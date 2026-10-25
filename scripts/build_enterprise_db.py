"""Build data/enterprise.db: a large-schema testbed for v4 (no LLM calls).

- Chinook tables  -> music_<Table>   (constraints + data kept)
- Northwind tables -> shop_<Table>
- N synthetic distractor tables across ~30 business domains, some with deliberately
  confusable names (rail_track_maintenance, hr_employee_shifts, crm_customer_feedback...).

  python scripts/build_enterprise_db.py [--distractors 250]
"""
import argparse
import random
import sqlite3
import sys
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SOURCES = {"music": DATA / "chinook.db", "shop": DATA / "northwind.db"}

DOMAINS = [
    "hr", "payroll", "recruiting", "fleet", "facilities", "itam", "helpdesk", "marketing",
    "finance", "ledger", "procurement", "manufacturing", "quality", "warehouse", "logistics",
    "clinic", "pharmacy", "school", "library", "events", "hotel", "airline", "energy",
    "telecom", "insurance", "realestate", "legal", "research", "compliance", "rail", "crm",
]

# entity -> columns (besides id). Types drive the fake data.
ENTITIES = {
    "accounts": [("account_code", "TEXT"), ("name", "TEXT"), ("status", "TEXT"), ("opened_on", "DATE")],
    "assets": [("asset_tag", "TEXT"), ("category", "TEXT"), ("purchase_cost", "REAL"), ("acquired_on", "DATE")],
    "assignments": [("assignee", "TEXT"), ("role", "TEXT"), ("start_date", "DATE"), ("end_date", "DATE")],
    "audits": [("auditor", "TEXT"), ("score", "INTEGER"), ("audited_on", "DATE"), ("status", "TEXT")],
    "budgets": [("fiscal_year", "INTEGER"), ("amount", "REAL"), ("owner", "TEXT"), ("status", "TEXT")],
    "contracts": [("vendor", "TEXT"), ("value", "REAL"), ("signed_on", "DATE"), ("expires_on", "DATE")],
    "documents": [("title", "TEXT"), ("doc_type", "TEXT"), ("created_on", "DATE"), ("author", "TEXT")],
    "locations": [("site_name", "TEXT"), ("city", "TEXT"), ("country", "TEXT"), ("capacity", "INTEGER")],
    "requests": [("requested_by", "TEXT"), ("priority", "TEXT"), ("requested_on", "DATE"), ("status", "TEXT")],
    "schedules": [("label", "TEXT"), ("starts_at", "DATETIME"), ("ends_at", "DATETIME"), ("status", "TEXT")],
    "metrics": [("metric_name", "TEXT"), ("value", "REAL"), ("measured_on", "DATE"), ("unit", "TEXT")],
    "vendors": [("company_name", "TEXT"), ("country", "TEXT"), ("rating", "INTEGER"), ("since", "DATE")],
    "incidents": [("summary", "TEXT"), ("severity", "TEXT"), ("reported_on", "DATE"), ("resolved", "TEXT")],
    "payments": [("payee", "TEXT"), ("amount", "REAL"), ("paid_on", "DATE"), ("method", "TEXT")],
    "notes": [("body", "TEXT"), ("written_by", "TEXT"), ("written_on", "DATE"), ("visibility", "TEXT")],
}
# Confusable tables: names overlap with music/shop vocabulary but mean something else.
CONFUSABLES = {
    "rail_track_maintenance": [("track_section", "TEXT"), ("inspected_on", "DATE"), ("defects", "INTEGER")],
    "hr_employee_shifts": [("employee_name", "TEXT"), ("shift_date", "DATE"), ("hours", "REAL")],
    "crm_customer_feedback": [("customer_name", "TEXT"), ("rating", "INTEGER"), ("comment", "TEXT"), ("given_on", "DATE")],
    "manufacturing_product_specs": [("product_code", "TEXT"), ("spec_name", "TEXT"), ("spec_value", "TEXT")],
    "marketing_artist_partnerships": [("artist_name", "TEXT"), ("campaign", "TEXT"), ("fee", "REAL")],
    "events_playlist_requests": [("event_name", "TEXT"), ("song_title", "TEXT"), ("requested_on", "DATE")],
    "logistics_shipper_rates": [("carrier", "TEXT"), ("zone", "TEXT"), ("rate_per_kg", "REAL")],
    "finance_revenue_forecasts": [("quarter", "TEXT"), ("forecast_amount", "REAL"), ("region", "TEXT")],
    "warehouse_order_picks": [("order_ref", "TEXT"), ("picker", "TEXT"), ("picked_at", "DATETIME")],
    "hotel_guest_genres": [("guest_ref", "TEXT"), ("preferred_genre", "TEXT")],
}

WORDS = ["alpha", "bravo", "delta", "echo", "nova", "orion", "atlas", "zephyr", "vega", "lyra", "pixel", "quartz"]
STATUSES = ["open", "closed", "pending", "active", "archived"]


def fake(rng: random.Random, typ: str, col: str):
    if typ == "INTEGER":
        return rng.randint(1, 500)
    if typ == "REAL":
        return round(rng.uniform(10, 50000), 2)
    if typ in ("DATE", "DATETIME"):
        d = date(2015, 1, 1) + timedelta(days=rng.randint(0, 3000))
        return d.isoformat() + (" 09:00:00" if typ == "DATETIME" else "")
    if col in ("status", "priority", "severity", "visibility", "resolved"):
        return rng.choice(STATUSES)
    return f"{rng.choice(WORDS)}-{rng.randint(100, 999)}"


def copy_source(conn: sqlite3.Connection, prefix: str, path: Path) -> int:
    """Recreate every table of `path` as <prefix>_<name>, keeping PK/FK constraints and rows."""
    conn.execute("ATTACH DATABASE ? AS src", (str(path),))
    tables = conn.execute(
        "SELECT name, sql FROM src.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for name, ddl in tables:
        tree = sqlglot.parse_one(ddl, read="sqlite")
        for t in tree.find_all(exp.Table):  # the table itself and REFERENCES targets
            t.set("this", exp.to_identifier(f"{prefix}_{t.name}", quoted=True))
        conn.execute(tree.sql(dialect="sqlite"))
        conn.execute(f'INSERT INTO "{prefix}_{name}" SELECT * FROM src."{name}"')
    conn.commit()
    conn.execute("DETACH DATABASE src")
    return len(tables)


def add_distractors(conn: sqlite3.Connection, n: int, rng: random.Random) -> int:
    specs = dict(CONFUSABLES)
    combos = [(d, e) for d in DOMAINS for e in ENTITIES]
    rng.shuffle(combos)
    for domain, entity in combos:
        if len(specs) >= n:
            break
        specs[f"{domain}_{entity}"] = ENTITIES[entity]
    names = list(specs)
    for name in names:
        cols = specs[name]
        fk = ""
        # Some tables reference another table in the same domain, like real schemas.
        same_domain = [o for o in names if o != name and o.split("_")[0] == name.split("_")[0]]
        if same_domain and rng.random() < 0.4:
            ref = rng.choice(same_domain)
            cols = cols + [(f"{ref.split('_', 1)[1]}_id", "INTEGER")]
            fk = f', FOREIGN KEY ("{cols[-1][0]}") REFERENCES "{ref}" (id)'
        col_ddl = ", ".join(f'"{c}" {t}' for c, t in cols)
        conn.execute(f'CREATE TABLE "{name}" (id INTEGER PRIMARY KEY, {col_ddl}{fk})')
        rows = [[i] + [fake(rng, t, c) if not c.endswith("_id") else rng.randint(1, 5) for c, t in cols]
                for i in range(1, 6)]
        conn.executemany(f'INSERT INTO "{name}" VALUES ({",".join("?" * (len(cols) + 1))})', rows)
    conn.commit()
    return len(names)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--distractors", type=int, default=250)
    args = p.parse_args()
    out = DATA / "enterprise.db"
    for f in (out, out.with_suffix(".profile.json")):
        f.unlink(missing_ok=True)
    with closing(sqlite3.connect(out)) as conn:
        counts = {prefix: copy_source(conn, prefix, path) for prefix, path in SOURCES.items()}
        counts["distractors"] = add_distractors(conn, args.distractors, random.Random(42))
    print(f"built {out.name}: {counts}, total {sum(counts.values())} tables")


if __name__ == "__main__":
    sys.exit(main())
