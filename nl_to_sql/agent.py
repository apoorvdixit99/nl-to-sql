"""v2: LangGraph agent with tools, self-correction, and human approval via interrupt().

Graph:  START -> agent --(tool calls)--> tools --(not done)--> agent ...
                   |                       |
                   +--(text reply)--> END  +--(done / declined / budget spent)--> END

Built to keep API calls low: the compact schema is in the system prompt (no discovery
round trip), and a successful run_query(final=true) ends the run without asking the
model to summarize, so the common case is ONE model call per question.
"""
import json
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from . import config
from .chain import Answer
from .db import Database, QueryResult
from .safety import classify, to_sqlglot_dialect

SYSTEM_TEMPLATE = """You answer questions about a {dialect} database by running SQL with your tools.

Tables (columns, PK = primary key, -> = foreign key, {{...}} = the column's full set of values, [a .. b] = value range):
{schema}
{examples}
How to work:
- Usually you can answer with ONE run_query call with final=true. The rows are shown straight to the user, so don't summarize them afterwards.
- Before that call, write at most one short sentence stating any assumption.
- Use the listed values and ranges exactly (spelling, type, date format). Only explore with describe_tables or run_query final=false when the schema leaves you unsure.
- If a query fails, or returns no rows when you expected some, investigate and retry.
- Return exactly what was asked. "Which/who has the most X" means only the top row (all tied rows if there is a tie), not a ranking. Respect wording like "after the first", "rounded to 1 decimal", or the columns requested.
- Quote identifiers that contain spaces. When listing many rows, add LIMIT 50 unless the user asks for everything.
- Only write INSERT/UPDATE/DELETE/DDL when the user clearly asks to change data. The user is asked to approve those.
- If the question can't be answered from this database, reply in plain text starting with CANNOT_ANSWER: and the reason."""

TOOLS = [
    {
        "name": "describe_tables",
        "description": "Full CREATE TABLE statements plus a few sample rows for the given tables.",
        "input_schema": {
            "type": "object",
            "properties": {"tables": {"type": "array", "items": {"type": "string"}}},
            "required": ["tables"],
        },
    },
    {
        "name": "run_query",
        "description": (
            "Run one SQL statement. final=true means this query answers the user's question: "
            "on success its rows are shown to the user and the task ends. Use final=false for "
            "exploratory queries whose results only you need."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "final": {"type": "boolean"},
            },
            "required": ["sql", "final"],
        },
    },
]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int
    status: str | None        # set when the run should end: ok | declined | denied
    sql: str | None           # last final query
    kind: str | None
    note: str
    result: dict | None       # QueryResult as a plain dict (checkpointer-friendly)


@dataclass
class Pending:
    """The run is paused waiting for the user to approve a query."""
    sql: str
    kind: str


def _format_rows(res: QueryResult) -> str:
    if not res.columns:
        return f"OK, {res.rowcount} row(s) affected."
    shown = res.rows[: config.TOOL_RESULT_ROWS]
    lines = ["\t".join(res.columns)] + ["\t".join(map(str, r)) for r in shown]
    extra = len(res.rows) - len(shown)
    if extra > 0 or res.truncated:
        lines.append(f"... ({extra}{'+' if res.truncated else ''} more rows not shown)")
    return f"{len(res.rows)}{'+' if res.truncated else ''} row(s):\n" + "\n".join(lines)


def _examples_block(db_name: str) -> str:
    path = config.ROOT / "nl_to_sql" / "examples" / f"{db_name}.json"
    if not path.exists():
        return ""
    pairs = json.loads(path.read_text(encoding="utf-8"))
    body = "\n".join(f"Q: {p['question']}\nSQL: {p['sql']}" for p in pairs)
    return f"\nWorked examples:\n{body}\n"


class SQLAgent:
    def __init__(self, db: Database, permissions: dict[str, str] | None = None, llm: Any = None):
        self.db = db
        self.permissions = permissions or config.PERMISSIONS
        llm = llm or ChatAnthropic(
            model=config.MODEL,
            max_tokens=config.MAX_TOKENS,
            output_config={"effort": config.EFFORT},
        )
        self.llm = llm.bind_tools(TOOLS)
        # Tools + this system prompt are identical for every question: cache them.
        self.system = SystemMessage(content=[{
            "type": "text",
            "text": SYSTEM_TEMPLATE.format(
                dialect=db.dialect,
                schema=db.compact_schema(hints=config.VALUE_HINTS),
                examples=_examples_block(db.name) if config.FEW_SHOT else "",
            ),
            "cache_control": {"type": "ephemeral"},
        }])
        self.graph = self._build()
        self._thread: dict | None = None
        self._question = ""

    # ---- graph -------------------------------------------------------------

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tools_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._after_agent, ["tools", END])
        g.add_conditional_edges("tools", self._after_tools, ["agent", END])
        return g.compile(checkpointer=InMemorySaver())

    def _agent_node(self, state: AgentState) -> dict:
        reply = self.llm.invoke([self.system, *state["messages"]])
        return {"messages": [reply], "llm_calls": state.get("llm_calls", 0) + 1}

    @staticmethod
    def _after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    @staticmethod
    def _after_tools(state: AgentState) -> str:
        if state.get("status") or state.get("llm_calls", 0) >= config.MAX_LLM_CALLS:
            return END
        return "agent"

    def _tools_node(self, state: AgentState) -> dict:
        # Note: on resume after interrupt(), LangGraph re-runs this node from the top,
        # so tool calls before the interrupt run again. They're reads, so that's harmless.
        ai: AIMessage = state["messages"][-1]
        update: dict = {"messages": []}
        for call in ai.tool_calls:
            content = self._run_tool(call["name"], call["args"], ai, update)
            update["messages"].append(ToolMessage(content=content, tool_call_id=call["id"]))
            if update.get("status") in ("declined", "denied"):
                # Answer any remaining calls so the transcript stays valid, then stop.
                for rest in ai.tool_calls[ai.tool_calls.index(call) + 1:]:
                    update["messages"].append(ToolMessage(content="Skipped.", tool_call_id=rest["id"]))
                break
        return update

    def _run_tool(self, name: str, args: dict, ai: AIMessage, update: dict) -> str:
        if name == "describe_tables":
            try:
                return self.db.schema(args.get("tables") or None)
            except Exception as e:
                return f"Error: {e}"
        if name != "run_query":
            return f"Error: unknown tool {name}"

        sql, final = args.get("sql", ""), bool(args.get("final"))
        kind = classify(sql, to_sqlglot_dialect(self.db.dialect))
        policy = self.permissions.get(kind, "confirm")
        if policy == "deny":
            update.update(status="denied", sql=sql, kind=kind)
            return f"Blocked: {kind} queries are not allowed."
        if kind != "read":
            # Validate before asking the user, so they never approve a query that can't run.
            # EXPLAIN plans without executing on SQLite, Postgres and MySQL.
            try:
                self.db.execute(f"EXPLAIN {sql}", commit=False)
            except Exception as e:
                return f"Error (query not run): {e}\nFix the query and try again."
        if policy == "confirm" and not interrupt({"sql": sql, "kind": kind}):
            update.update(status="declined", sql=sql, kind=kind)
            return "The user declined to run this query."

        try:
            res = self.db.execute(sql, commit=kind != "read")
        except Exception as e:
            return f"Error: {e}\nFix the query and try again."

        if final or kind != "read":
            update.update(sql=sql, kind=kind, note=ai.text.strip(), result=res.__dict__.copy())
            # Empty final read: give the model one look before ending, in case it's a bad filter.
            if kind == "read" and not res.rows:
                return ("0 rows. If that's unexpected, check the data (value formats, ranges) "
                        "and retry. If it's correct, reply with one short sentence.")
            update["status"] = "ok"
        return _format_rows(res)

    # ---- public API (UI-agnostic) -----------------------------------------

    def start(self, question: str) -> Answer | Pending:
        self._question = question
        self._thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
        return self._run({"messages": [HumanMessage(content=question)]})

    def resume(self, approved: bool) -> Answer | Pending:
        return self._run(Command(resume=approved))

    def _run(self, inp) -> Answer | Pending:
        try:
            out = self.graph.invoke(inp, self._thread)
        except Exception as e:  # API/network errors surface to the UI, not as a crash
            return Answer(question=self._question, status="error", error=f"{type(e).__name__}: {e}")
        if out.get("__interrupt__"):
            req = out["__interrupt__"][0].value
            return Pending(sql=req["sql"], kind=req["kind"])
        return self._to_answer(out)

    def _to_answer(self, state: dict) -> Answer:
        ais = [m for m in state["messages"] if isinstance(m, AIMessage)]
        a = Answer(
            question=self._question,
            sql=state.get("sql"),
            kind=state.get("kind"),
            note=state.get("note", ""),
            llm_calls=state.get("llm_calls", 0),
            input_tokens=sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ais),
            output_tokens=sum((m.usage_metadata or {}).get("output_tokens", 0) for m in ais),
        )
        if state.get("result") is not None:
            a.result = QueryResult(**state["result"])

        last = state["messages"][-1]
        status = state.get("status")
        if status:
            a.status = status
        elif isinstance(last, AIMessage) and not last.tool_calls:
            text = last.text.strip()
            if text.startswith("CANNOT_ANSWER:"):
                a.status, a.note = "cannot_answer", text.removeprefix("CANNOT_ANSWER:").strip()
            else:  # e.g. confirming an empty result is correct
                a.status, a.note = "ok", text
        else:
            a.status = "gave_up"
            a.error = f"Stopped after {a.llm_calls} model calls without a final answer."
        return a
