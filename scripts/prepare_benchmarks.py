"""Extract downloaded benchmark archives into data/benchmarks/<bench>/ and write items.json.

Archives (downloaded once into data/benchmarks/*_raw/):
  BIRD dev      https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
  Spider 1.0    https://drive.google.com/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J   (spider_data.zip)
  Spider 2.0    https://drive.google.com/uc?id=1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG   (local_sqlite.zip)
                + git sparse checkout of github.com/xlang-ai/Spider2 (spider2-lite gold + documents)

Only data files are extracted (.sqlite, .json, .csv, .md); nothing from the archives is executed.

  python scripts/prepare_benchmarks.py [spider bird spider2]
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data" / "benchmarks"


def safe_extract(zf: zipfile.ZipFile, member: str, dest: Path) -> Path:
    """Extract one member to dest (a file path), refusing anything that escapes data/benchmarks."""
    dest = dest.resolve()
    if B.resolve() not in dest.parents:
        raise ValueError(f"refusing to write outside {B}: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with zf.open(member) as src, open(dest, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
    return dest


def prepare_spider() -> None:
    out = B / "spider"
    with zipfile.ZipFile(B / "spider_raw" / "spider_data.zip") as zf:
        dev = json.loads(zf.read("spider_data/dev.json"))
        db_ids = sorted({d["db_id"] for d in dev})
        for db_id in db_ids:
            safe_extract(zf, f"spider_data/database/{db_id}/{db_id}.sqlite", out / "db" / f"{db_id}.sqlite")
    items = [{"id": f"spider-{i:04d}", "db_id": d["db_id"], "question": d["question"], "gold_sql": d["query"]}
             for i, d in enumerate(dev)]
    (out / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(f"spider: {len(items)} dev questions over {len(db_ids)} databases")


def _read_csv(raw: bytes) -> list[dict]:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(enc))))
        except UnicodeDecodeError:
            continue
    return []


def prepare_bird() -> None:
    out = B / "bird"
    with zipfile.ZipFile(B / "bird_raw" / "dev.zip") as zf:
        dev = json.loads(zf.read("dev_20240627/dev.json"))
        inner = io.BytesIO(zf.read("dev_20240627/dev_databases.zip"))
    with zipfile.ZipFile(inner) as dbz:
        names = dbz.namelist()
        db_ids = sorted({d["db_id"] for d in dev})
        for db_id in db_ids:
            safe_extract(dbz, f"dev_databases/{db_id}/{db_id}.sqlite", out / "db" / f"{db_id}.sqlite")
            # BIRD ships per-table CSVs of column descriptions: convert to our descriptions format.
            tables = {}
            prefix = f"dev_databases/{db_id}/database_description/"
            for name in names:
                if name.startswith(prefix) and name.endswith(".csv"):
                    table = Path(name).stem
                    cols = {}
                    for row in _read_csv(dbz.read(name)):
                        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                        col = row.get("original_column_name") or ""
                        parts = [row.get("column_description", ""), row.get("value_description", "")]
                        if row.get("column_name") and row["column_name"].lower() != col.lower():
                            parts.insert(0, row["column_name"])
                        text = "; ".join(p for p in parts if p and p.lower() not in ("nan", "none"))
                        if col and text and text.lower() != col.lower():
                            cols[col] = text[:300]
                    tables[table] = {"description": "", "columns": cols}
            desc_path = out / "db" / f"{db_id}.descriptions.json"
            desc_path.write_text(json.dumps({"source": "bird", "tables": tables}, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    items = [{"id": f"bird-{d.get('question_id', i):04d}", "db_id": d["db_id"], "question": d["question"],
              "evidence": d.get("evidence", ""), "gold_sql": d["SQL"], "level": d.get("difficulty", "")}
             for i, d in enumerate(dev)]
    (out / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(f"bird: {len(items)} dev questions over {len(db_ids)} databases (descriptions from BIRD CSVs)")


def prepare_spider2() -> None:
    out = B / "spider2"
    repo = B / "spider2_raw" / "repo" / "spider2-lite"
    tasks = [json.loads(line) for line in (repo / "spider2-lite.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    local = [t for t in tasks if t["instance_id"].startswith("local")]
    evals = {json.loads(line)["instance_id"]: json.loads(line)
             for line in (repo / "evaluation_suite" / "gold" / "spider2lite_eval.jsonl").read_text().splitlines()
             if line.strip()}
    gold_dir = repo / "evaluation_suite" / "gold" / "exec_result"
    with zipfile.ZipFile(B / "spider2_raw" / "local_sqlite.zip") as zf:
        for db in sorted({t["db"] for t in local}):
            safe_extract(zf, f"{db}.sqlite", out / "db" / f"{db}.sqlite")
    items = []
    for t in local:
        iid = t["instance_id"]
        golds = sorted(str(p.relative_to(B)) for p in gold_dir.glob(f"{iid}*.csv")
                       if p.stem == iid or p.stem.startswith(f"{iid}_"))
        if not golds:
            continue  # no gold result published for this task
        doc = ""
        if t.get("external_knowledge"):
            p = repo / "resource" / "documents" / t["external_knowledge"]
            doc = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        e = evals.get(iid, {})
        items.append({"id": iid, "db_id": t["db"], "question": t["question"], "evidence": doc,
                      "gold_csvs": golds, "condition_cols": e.get("condition_cols", []),
                      "ignore_order": e.get("ignore_order", False)})
    (out / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(f"spider2: {len(items)} local (SQLite) tasks with gold results over "
          f"{len({i['db_id'] for i in items})} databases")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["spider", "bird", "spider2"]
    for name in wanted:
        {"spider": prepare_spider, "bird": prepare_bird, "spider2": prepare_spider2}[name]()
