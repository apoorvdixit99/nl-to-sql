"""Copy a configured database into any SQLAlchemy URL (e.g. load Chinook into Postgres).

  python scripts/copy_database.py chinook postgresql+psycopg://postgres:postgres@127.0.0.1:54329/postgres

Tables are created in foreign-key order with SQLAlchemy's generic types, so
SQLite's DATETIME/NVARCHAR become TIMESTAMP/VARCHAR on Postgres, and so on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import MetaData, create_engine, insert, select  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from nl_to_sql import config  # noqa: E402


def copy(src_url: str, dst_url: str, batch: int = 5000) -> None:
    src, dst = create_engine(src_url), create_engine(dst_url, poolclass=NullPool)
    meta = MetaData()
    meta.reflect(bind=src)
    meta.drop_all(dst)
    meta.create_all(dst)
    with src.connect() as s, dst.begin() as d:
        for table in meta.sorted_tables:  # parents before children
            rows = s.execute(select(table)).mappings()
            n = 0
            while chunk := rows.fetchmany(batch):
                d.execute(insert(table), [dict(r) for r in chunk])
                n += len(chunk)
            print(f"  {table.name}: {n} rows")


if __name__ == "__main__":
    name, dst = sys.argv[1], sys.argv[2]
    copy(config.DATABASES[name], dst)
