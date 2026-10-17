"""Download the Chinook and Northwind sample SQLite databases into data/."""
import sqlite3
from contextlib import closing
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "chinook.db": "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite",
    "northwind.db": "https://github.com/jpwhite3/northwind-SQLite3/raw/main/dist/northwind.db",
}


def download(name: str, url: str, force: bool = False) -> None:
    dest = DATA_DIR / name
    if dest.exists() and not force:
        print(f"[skip] {name} already exists")
        return
    print(f"[get]  {name} <- {url}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    # Sanity check: must be a readable SQLite file with at least one table.
    with closing(sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)) as conn:
        n = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    if n == 0:
        tmp.unlink()
        raise RuntimeError(f"{name}: downloaded file has no tables")
    tmp.replace(dest)
    print(f"[ok]   {name} ({n} tables)")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    force = "--force" in sys.argv
    for name, url in DATASETS.items():
        download(name, url, force)
