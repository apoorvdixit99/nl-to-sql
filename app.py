"""Streamlit UI: streamlit run app.py

Same agent as the CLI. Approval for writes is a pair of buttons instead of a y/N prompt:
the agent returns Pending, we keep it in session state, and resume() on click.
"""
import pandas as pd
import streamlit as st

from nl_to_sql import config
from nl_to_sql.agent import Pending, SQLAgent
from nl_to_sql.chain import Answer
from nl_to_sql.db import Database

st.set_page_config(page_title="nl-to-sql", page_icon="🗄️", layout="wide")


# ---- state -----------------------------------------------------------------

def new_agent(db_name: str) -> SQLAgent:
    return SQLAgent(Database(db_name), memory=st.session_state.get("memory", True))


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("db_name", config.DEFAULT_DB)
    ss.setdefault("memory", True)
    ss.setdefault("messages", [])      # [{"role": "user", "text": ...} | {"role": "assistant", "answer": Answer}]
    ss.setdefault("pending", None)     # Pending while waiting for approval
    ss.setdefault("usage", {"calls": 0, "in": 0, "cached": 0, "out": 0})
    if "agent" not in ss:
        ss.agent = new_agent(ss.db_name)


def add_answer(out) -> None:
    ss = st.session_state
    if isinstance(out, Pending):
        ss.pending = out
        return
    ss.pending = None
    ss.messages.append({"role": "assistant", "answer": out})
    u = ss.usage
    u["calls"] += out.llm_calls
    u["in"] += out.input_tokens
    u["cached"] += out.cache_read_tokens
    u["out"] += out.output_tokens


def reset_conversation() -> None:
    ss = st.session_state
    ss.messages, ss.pending = [], None
    ss.agent = new_agent(ss.db_name)


# ---- rendering ---------------------------------------------------------------

STATUS_TEXT = {
    "cannot_answer": ("warning", "Can't answer this from the database."),
    "declined": ("info", "Skipped. Nothing was run."),
    "denied": ("error", "Blocked by the permission policy."),
}


def render_answer(a: Answer) -> None:
    if a.note:
        st.markdown(a.note)
    if a.status in STATUS_TEXT:
        kind, text = STATUS_TEXT[a.status]
        getattr(st, kind)(text)
    elif a.status in ("error", "gave_up"):
        st.error(a.error or "Something went wrong.")
    if a.sql:
        with st.expander(f"SQL ({a.kind})", expanded=a.result is None):
            st.code(a.sql, language="sql")
    res = a.result
    if res is not None and a.status == "ok":
        if res.columns:
            st.dataframe(pd.DataFrame(res.rows, columns=res.columns), hide_index=True,
                         use_container_width=True)
            more = f" (first {config.MAX_RESULT_ROWS})" if res.truncated else ""
            st.caption(f"{len(res.rows)} row(s){more}")
        else:
            st.success(f"Done. {res.rowcount} row(s) affected.")
    bits = []
    if a.review:
        bits.append(f"answer check: {a.review}")
    if a.candidates_agreed:
        bits.append(f"attempts agreeing: {a.candidates_agreed}")
    if a.context_tables:
        bits.append(f"{len(a.context_tables)} tables retrieved")
    if a.llm_calls:
        bits.append(f"{a.llm_calls} model call(s) · {a.input_tokens:,} in / {a.output_tokens:,} out tokens")
    if bits:
        st.caption(" · ".join(bits))


def sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.header("nl-to-sql")
        names = list(config.DATABASES)
        choice = st.selectbox("Database", names, index=names.index(ss.db_name))
        if choice != ss.db_name:
            ss.db_name = choice
            reset_conversation()
            st.rerun()
        memory = st.toggle("Remember earlier questions", value=ss.memory,
                           help=f"Passes the last {config.MEMORY_TURNS} questions and their SQL to follow-ups.")
        if memory != ss.memory:
            ss.memory = memory
            reset_conversation()
            st.rerun()
        if st.button("New conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.divider()
        agent = ss.agent
        st.caption(f"Model: `{config.MODEL}` (effort {config.EFFORT})")
        st.caption(f"Schema mode: {agent.schema_mode} · {len(agent.db.table_names())} tables")
        st.caption(f"Writes: {config.PERMISSIONS['write']} · Answer check: {config.VERIFY}")
        u = ss.usage
        pin, pout = config.PRICING.get(config.MODEL, (0, 0))
        cost = ((u["in"] - u["cached"]) * pin + u["cached"] * pin * 0.1 + u["out"] * pout) / 1e6
        st.metric("Model calls this session", u["calls"], help=f"{u['in']:,} in / {u['out']:,} out tokens")
        st.caption(f"Estimated cost: ${cost:.4f}")
        with st.expander("Tables"):
            st.write(", ".join(agent.db.table_names()))


def main() -> None:
    init_state()
    sidebar()
    ss = st.session_state

    for m in ss.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["text"])
            else:
                render_answer(m["answer"])

    if ss.pending is not None:
        p: Pending = ss.pending
        with st.chat_message("assistant"):
            label = "write" if p.kind == "write" else "unclassified"
            st.warning(f"This is a **{label}** query and needs your approval before it runs.")
            st.code(p.sql, language="sql")
            approve, decline = st.columns(2)
            if approve.button("Approve and run", type="primary", use_container_width=True):
                with st.spinner("Running..."):
                    add_answer(ss.agent.resume(True))
                st.rerun()
            if decline.button("Decline", use_container_width=True):
                add_answer(ss.agent.resume(False))
                st.rerun()

    question = st.chat_input("Ask a question about the data", disabled=ss.pending is not None)
    if question:
        ss.messages.append({"role": "user", "text": question})
        with st.spinner("Thinking..."):
            add_answer(ss.agent.start(question))
        st.rerun()


main()
