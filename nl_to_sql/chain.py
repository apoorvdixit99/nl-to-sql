"""v1 pipeline: question -> SQL (LLM) -> safety check -> execute.

UI-agnostic: callers pass an `approve` callback for queries that need confirmation,
so the same core works for the CLI now and Streamlit later.
"""
import re
from dataclasses import dataclass, field
from typing import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .db import Database, QueryResult
from .safety import QueryKind, to_sqlglot_dialect
from .security import SQLGuard

SYSTEM_TEMPLATE = """You translate natural-language questions into a single {dialect} SQL query.

Database schema (with sample rows):
{schema}

Guidelines:
- Use only tables and columns that exist in the schema above; quote identifiers that contain spaces.
- For questions asking to see rows, add LIMIT 50 unless the user asks for everything or a specific count.
- Only write INSERT/UPDATE/DELETE/DDL when the user clearly asks to change data.
- Reply with the query in one ```sql fenced block, followed by at most one short sentence on any assumption you made.
- If the question can't be answered from this schema, reply with no SQL block, only a line starting with CANNOT_ANSWER: and the reason."""

Approver = Callable[[str, QueryKind], bool]


@dataclass
class Answer:
    question: str
    sql: str | None = None
    kind: QueryKind | None = None
    note: str = ""                    # model's assumption note, or refusal reason
    status: str = "ok"                # ok | cannot_answer | declined | denied | gave_up | error
    result: QueryResult | None = None
    error: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0     # part of input_tokens served from the prompt cache (~0.1x price)
    context_tables: list[str] = field(default_factory=list)  # v4: tables retrieved for the prompt
    review: str = ""               # v5: answer-check verdict, if it ran ("OK" or the issue)
    candidates_agreed: str = ""    # v5: self-consistency agreement, e.g. "3/3"
    security: list[str] = field(default_factory=list)  # v7: security-layer blocks/warnings


_SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class NLToSQL:
    def __init__(self, db: Database, permissions: dict[str, str] | None = None):
        self.db = db
        self.permissions = permissions or config.PERMISSIONS
        self.llm = ChatAnthropic(
            model=config.MODEL,
            max_tokens=config.MAX_TOKENS,
            output_config={"effort": config.EFFORT},
        )
        # Schema goes in the system prompt with a cache breakpoint: it's identical
        # across questions, so repeat questions read it from the prompt cache.
        self._system = SystemMessage(content=[{
            "type": "text",
            "text": SYSTEM_TEMPLATE.format(dialect=db.dialect, schema=db.schema()),
            "cache_control": {"type": "ephemeral"},
        }])

    def generate_sql(self, question: str) -> tuple[str | None, str]:
        """Return (sql, note). sql is None if the model says it can't answer."""
        msg = self.llm.invoke([self._system, HumanMessage(content=question)])
        self.last_usage = msg.usage_metadata or {}
        reply = msg.text
        match = _SQL_BLOCK.search(reply)
        if not match:
            return None, reply.replace("CANNOT_ANSWER:", "").strip()
        note = _SQL_BLOCK.sub("", reply).strip()
        return match.group(1).strip(), note

    def ask(self, question: str, approve: Approver) -> Answer:
        answer = Answer(question=question)
        try:
            answer.sql, answer.note = self.generate_sql(question)
            answer.llm_calls = 1
            answer.input_tokens = self.last_usage.get("input_tokens", 0)
            answer.output_tokens = self.last_usage.get("output_tokens", 0)
        except Exception as e:  # API/network errors surface to the UI, not as a crash
            answer.status, answer.error = "error", f"LLM call failed: {e}"
            return answer
        if answer.sql is None:
            answer.status = "cannot_answer"
            return answer

        verdict = SQLGuard(self.db.columns(), to_sqlglot_dialect(self.db.dialect)).check(answer.sql)
        answer.security = [f"blocked: {m}" for m in verdict.violations] + [f"warning: {m}" for m in verdict.warnings]
        if not verdict.allowed:
            answer.status, answer.error = "denied", f"Blocked by the security policy: {verdict.message()}"
            return answer
        answer.kind = verdict.kind
        policy = self.permissions.get(answer.kind, "confirm")
        if policy == "deny":
            answer.status = "denied"
            return answer
        if policy == "confirm" and not approve(answer.sql, answer.kind):
            answer.status = "declined"
            return answer

        try:
            answer.result = self.db.execute(answer.sql, commit=answer.kind != "read")
        except Exception as e:
            answer.status, answer.error = "error", str(e)
        return answer
