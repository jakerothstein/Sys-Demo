#!/usr/bin/env python3
"""
Materialize AmbiQT's schema modifications into real SQLite databases.

AmbiQT ships the original Spider .sqlite files plus per-question JSON that
describes the schema modification (e.g. "add synonym columns artist_name and
performer_name to singer.name"). The modifications are documented in the JSON
but NOT applied to the DB -- so the paper's gold `query1` / `query2` SQL
actually fail against the shipped sqlite. This script applies those
modifications, producing a DB per (subtype, db_id) so the modified schema is
visible to both the LLM (via PRAGMA table_info) and the SQL sandbox.

Currently supported subtypes:
  * col-synonyms  -> ALTER TABLE ADD COLUMN + UPDATE (copy values)
  * tbl-synonyms  -> CREATE TABLE <syn> AS SELECT * FROM <orig>

Not yet supported (benchmark entries are skipped):
  * tbl-split     -> requires moving columns into a new aux table per-question
  * tbl-agg       -> requires creating a precomputed-aggregate table per-query

Output layout mirrors AmbiQT's own structure so BenchmarkExecutor can find the
DBs with `dataset='ambiqt_colsyn'` / `dataset='ambiqt_tblsyn'`:

    data/ambiqt_colsyn/database/<db_id>/<db_id>.sqlite
    data/ambiqt_tblsyn/database/<db_id>/<db_id>.sqlite

Run:
    python scripts/build_ambiqt_dbs.py
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
AMBIQT = DATA / "ambiqt"
SRC_DB_DIR = AMBIQT / "db-content" / "database"

SUBTYPE_SHORT = {
    "col-synonyms": "colsyn",
    "tbl-synonyms": "tblsyn",
}


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    return [r[1] for r in cur.fetchall()]


def _tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]


def _resolve_ci(names: List[str], target: str) -> str | None:
    """Case-insensitive name lookup in a list of real table/column names."""
    tl = target.lower()
    for n in names:
        if n.lower() == tl:
            return n
    return None


def _apply_col_synonyms(conn: sqlite3.Connection, extra_map: Dict[str, Dict[str, List[str]]]) -> int:
    """Add synonym columns that are copies of an existing column. Idempotent."""
    added = 0
    real_tables = _tables(conn)
    for table, col_syns in (extra_map or {}).items():
        real_table = _resolve_ci(real_tables, table)
        if real_table is None:
            continue
        real_cols = _table_columns(conn, real_table)
        existing_lower = {c.lower() for c in real_cols}
        for orig_col, syns in col_syns.items():
            real_orig = _resolve_ci(real_cols, orig_col)
            if real_orig is None:
                continue
            # Get the column type once.
            cur = conn.execute(f'PRAGMA table_info("{real_table}")')
            col_type = "TEXT"
            for row in cur.fetchall():
                if row[1] == real_orig:
                    col_type = row[2] or "TEXT"
                    break
            for syn in syns or []:
                if not syn or syn.lower() in existing_lower:
                    continue
                try:
                    conn.execute(f'ALTER TABLE "{real_table}" ADD COLUMN "{syn}" {col_type}')
                    conn.execute(
                        f'UPDATE "{real_table}" SET "{syn}" = "{real_orig}"'
                    )
                    existing_lower.add(syn.lower())
                    added += 1
                except sqlite3.OperationalError as e:
                    # Duplicate column (race from an earlier UNION) — fine.
                    if "duplicate column" not in str(e).lower():
                        raise
    conn.commit()
    return added


def _apply_tbl_synonyms(conn: sqlite3.Connection, extra_tbl_map: Dict[str, List[str]]) -> int:
    """Create synonym tables as full copies of an existing table. Idempotent."""
    added = 0
    real_tables = _tables(conn)
    existing_lower = {t.lower() for t in real_tables}
    for orig_tbl, syns in (extra_tbl_map or {}).items():
        real_orig = _resolve_ci(real_tables, orig_tbl)
        if real_orig is None:
            continue
        for syn in syns or []:
            if not syn or syn.lower() in existing_lower:
                continue
            try:
                conn.execute(f'CREATE TABLE "{syn}" AS SELECT * FROM "{real_orig}"')
                existing_lower.add(syn.lower())
                added += 1
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e).lower():
                    raise
    conn.commit()
    return added


def build_subtype(subtype: str) -> Dict[str, Any]:
    short = SUBTYPE_SHORT[subtype]
    out_base = DATA / f"ambiqt_{short}" / "database"
    out_base.mkdir(parents=True, exist_ok=True)

    bench_path = AMBIQT / "benchmark" / subtype / "validation.json"
    with bench_path.open() as f:
        entries = json.load(f)

    # Group modifications per-db so each DB is materialized once with the
    # union of all modifications for that subtype.
    per_db: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"col_mods": defaultdict(lambda: defaultdict(set)),
                 "tbl_mods": defaultdict(set)}
    )
    for e in entries:
        db_id = e["db_id"]
        if subtype == "col-synonyms":
            for table, col_syns in (e.get("extra_map") or {}).items():
                for col, syns in (col_syns or {}).items():
                    per_db[db_id]["col_mods"][table][col].update(syns or [])
        elif subtype == "tbl-synonyms":
            for table, syns in (e.get("extra_table_map") or {}).items():
                per_db[db_id]["tbl_mods"][table].update(syns or [])

    summary = {"subtype": subtype, "dbs": 0, "new_cols": 0, "new_tbls": 0,
               "missing": [], "errors": []}
    for db_id, mods in sorted(per_db.items()):
        src = SRC_DB_DIR / db_id / f"{db_id}.sqlite"
        if not src.exists():
            summary["missing"].append(db_id)
            continue
        dst_dir = out_base / db_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{db_id}.sqlite"
        # Always overwrite so reruns pick up new modifications.
        shutil.copy2(src, dst)

        try:
            with sqlite3.connect(dst) as conn:
                if subtype == "col-synonyms":
                    # Flatten the per-(table, col) sets into the nested dict
                    # shape the applier expects.
                    extra_map = {
                        t: {c: sorted(syns) for c, syns in cols.items()}
                        for t, cols in mods["col_mods"].items()
                    }
                    summary["new_cols"] += _apply_col_synonyms(conn, extra_map)
                elif subtype == "tbl-synonyms":
                    extra_tbl = {t: sorted(s) for t, s in mods["tbl_mods"].items()}
                    summary["new_tbls"] += _apply_tbl_synonyms(conn, extra_tbl)
            summary["dbs"] += 1
        except Exception as e:  # noqa: BLE001
            summary["errors"].append({"db_id": db_id, "error": str(e)})
    return summary


def main() -> int:
    if not SRC_DB_DIR.exists():
        print(f"AmbiQT databases not found at {SRC_DB_DIR}.\n"
              f"Run: cd data/ambiqt && unzip -q db-content.zip", file=sys.stderr)
        return 1

    overall = []
    for subtype in SUBTYPE_SHORT:
        print(f"\n=== building {subtype} ===")
        s = build_subtype(subtype)
        overall.append(s)
        print(f"  dbs materialized: {s['dbs']}")
        if subtype == "col-synonyms":
            print(f"  synonym columns added: {s['new_cols']}")
        else:
            print(f"  synonym tables created: {s['new_tbls']}")
        if s["missing"]:
            print(f"  missing source dbs (skipped): {s['missing']}")
        if s["errors"]:
            print(f"  errors: {len(s['errors'])} -- first 3: {s['errors'][:3]}")

    meta_path = DATA / "ambiqt_build_summary.json"
    with meta_path.open("w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nSummary: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
