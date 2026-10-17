"""Interactive REPL: python cli.py [--db chinook|northwind] [--v1]"""
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from nl_to_sql import config
from nl_to_sql.agent import Pending, SQLAgent
from nl_to_sql.chain import Answer, NLToSQL
from nl_to_sql.db import Database

console = Console()

HELP = """Type a question in plain English. Commands:
  /tables   list tables      /schema   show full schema
  /db NAME  switch database  /help     this help
  /quit     exit"""


def show_sql(sql: str, kind: str | None, style: str = "cyan") -> None:
    console.print(Panel(Syntax(sql, "sql", word_wrap=True), title=f"SQL ({kind})", border_style=style))


def approve(sql: str, kind: str) -> bool:
    show_sql(sql, kind, style="yellow")
    label = "WRITE" if kind == "write" else "UNCLASSIFIED"
    console.print(f"[bold yellow]This is a {label} query and needs your approval.[/]")
    return Confirm.ask("Run it?", default=False)


def render(answer: Answer, sql_already_shown: bool = False) -> None:
    if answer.sql and not sql_already_shown:
        show_sql(answer.sql, answer.kind)
    if answer.note:
        console.print(f"[dim]{answer.note}[/]")

    if answer.status == "cannot_answer":
        console.print("[yellow]Can't answer this from the schema.[/]")
    elif answer.status == "declined":
        console.print("[yellow]Skipped. Nothing was run.[/]")
    elif answer.status == "denied":
        console.print(f"[red]Blocked by policy: {answer.kind} queries are denied.[/]")
    elif answer.status in ("error", "gave_up"):
        console.print(f"[red]Error:[/] {answer.error}")

    res = answer.result
    if res is not None and answer.status not in ("declined", "denied", "cannot_answer"):
        if res.columns:
            table = Table(header_style="bold")
            for col in res.columns:
                table.add_column(str(col))
            for row in res.rows:
                table.add_row(*("NULL" if v is None else str(v) for v in row))
            console.print(table)
            more = f" (showing first {config.MAX_RESULT_ROWS})" if res.truncated else ""
            console.print(f"[dim]{len(res.rows)} row(s){more}[/]")
        else:
            console.print(f"[green]Done.[/] {res.rowcount} row(s) affected.")

    extras = []
    if answer.review:
        extras.append(f"answer check: {answer.review}")
    if answer.candidates_agreed:
        extras.append(f"attempts agreeing: {answer.candidates_agreed}")
    if answer.context_tables:
        extras.append(f"{len(answer.context_tables)} tables retrieved")
    if answer.llm_calls:
        extras.append(f"{answer.llm_calls} model call(s) · "
                      f"{answer.input_tokens:,} in / {answer.output_tokens:,} out tokens")
    if extras:
        console.print("[dim]" + " · ".join(extras) + "[/]")


def ask_agent(agent: SQLAgent, question: str) -> None:
    previewed = False
    with console.status("Thinking..."):
        out = agent.start(question)
    while isinstance(out, Pending):
        previewed = True
        approved = approve(out.sql, out.kind)  # outside the spinner so the prompt isn't garbled
        with console.status("Running..."):
            out = agent.resume(approved)
    render(out, sql_already_shown=previewed)


def ask_chain(chain: NLToSQL, question: str) -> None:
    previewed = False

    def approve_once(sql: str, kind: str) -> bool:
        nonlocal previewed
        previewed = True
        return approve(sql, kind)

    console.print("[dim]Thinking...[/]")  # no spinner: approve() may prompt mid-call
    answer = chain.ask(question, approve_once)
    render(answer, sql_already_shown=previewed)


def make(db_name: str, v1: bool):
    db = Database(db_name)
    return NLToSQL(db) if v1 else SQLAgent(db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural language to SQL")
    parser.add_argument("--db", default=config.DEFAULT_DB, choices=list(config.DATABASES))
    parser.add_argument("--v1", action="store_true", help="use the v1 single-shot chain")
    args = parser.parse_args()

    runner = make(args.db, args.v1)
    mode = "v1 chain" if args.v1 else "v2 agent"
    console.print(f"[bold]nl-to-sql[/] · {mode} · db=[cyan]{args.db}[/] · model=[cyan]{config.MODEL}[/]")
    console.print(HELP + "\n")

    while True:
        try:
            q = console.input(f"[bold green]{runner.db.name}>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            break
        if q == "/help":
            console.print(HELP)
        elif q == "/tables":
            console.print(", ".join(runner.db.table_names()))
        elif q == "/schema":
            console.print(runner.db.schema())
        elif q.startswith("/db"):
            name = q[3:].strip()
            try:
                runner = make(name, args.v1)
                console.print(f"Switched to [cyan]{name}[/]")
            except ValueError as e:
                console.print(f"[red]{e}[/]")
        elif args.v1:
            ask_chain(runner, q)
        else:
            ask_agent(runner, q)
        console.print()


if __name__ == "__main__":
    main()
