#!/usr/bin/env python3
"""
AmbiQT Benchmark Runner.

AmbiQT (EMNLP 2023, Bhaskar et al.) is a 3000+ example benchmark where every
question is deliberately ambiguous -- i.e. it has two valid SQL interpretations
that differ due to column-synonym ambiguity, table-synonym ambiguity, a missing
join, or a precomputed aggregate. This script runs a sampled slice through the
Text-to-SQL pipeline and measures how often the pipeline catches the ambiguity
(i.e. flags the question for HITL clarification).

Since every AmbiQT question is expected to trigger HITL, the primary metric is
RECALL per ambiguity type. Combined with the hand-labeled
`data/ambiguity_benchmark.json` run -- which provides the answerable side --
you get a full precision/recall picture across two benchmarks.

Currently supports the subtypes whose modifications are materialized in real
SQLite databases by `scripts/build_ambiqt_dbs.py`:
    - col-synonyms  (ambiguity type "C" in the paper)
    - tbl-synonyms  (ambiguity type "T")

Usage:
    # first time only
    python scripts/build_ambiqt_dbs.py

    python scripts/run_ambiqt.py                         # default: 10 per subtype
    python scripts/run_ambiqt.py --k 25                  # 25 per subtype
    python scripts/run_ambiqt.py --subtype col-synonyms  # one subtype only
    python scripts/run_ambiqt.py --server http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from scripts.run_ambiguity_benchmark import (  # noqa: E402  (sys.path first)
    LocalRunner,
    ServerRunner,
    _safe_div,
)


SUBTYPE_DATASET = {
    "col-synonyms": "ambiqt_colsyn",
    "tbl-synonyms": "ambiqt_tblsyn",
    # tbl-split and tbl-agg require per-question surgery on the sqlite files
    # and aren't materialized yet. Skipped with a loud message.
}

CATEGORY_TAG = {
    "col-synonyms": "ambiguous_column_synonym",
    "tbl-synonyms": "ambiguous_table_synonym",
}


def load_ambiqt_entries(subtypes: List[str]) -> List[Dict[str, Any]]:
    """
    Load AmbiQT validation entries and reshape into our benchmark schema.

    Each entry gets expected_hitl=True since every AmbiQT question is
    ambiguous by construction.
    """
    all_entries: List[Dict[str, Any]] = []
    ambiqt_dir = ROOT / "data" / "ambiqt" / "benchmark"
    for subtype in subtypes:
        path = ambiqt_dir / subtype / "validation.json"
        if not path.exists():
            print(f"[warn] missing {path}, skipping", file=sys.stderr)
            continue
        raw = json.load(path.open())
        dataset = SUBTYPE_DATASET[subtype]
        for idx, e in enumerate(raw):
            all_entries.append({
                "id": f"ambiqt_{CATEGORY_TAG[subtype]}_{idx:05d}",
                "db_id": e["db_id"],
                "dataset": dataset,
                "question": e["question"],
                "category": CATEGORY_TAG[subtype],
                "subtype": subtype,
                "expected_hitl": True,
                "gold_query1": e.get("query1"),
                "gold_query2": e.get("query2"),
                "orig_query": e.get("orig_query"),
            })
    return all_entries


def sample_per_subtype(entries: List[Dict[str, Any]], k: int,
                       seed: int) -> List[Dict[str, Any]]:
    """Sample k entries per (subtype, db_id) distribution, deterministic."""
    by_subtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_subtype[e["subtype"]].append(e)

    rng = random.Random(seed)
    sampled: List[Dict[str, Any]] = []
    for subtype, group in by_subtype.items():
        rng.shuffle(group)
        sampled.extend(group[:k])
    rng.shuffle(sampled)
    return sampled


def compute_ambiqt_metrics(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    AmbiQT-specific metrics.

    Since every example is positive (expected_hitl=True), recall is the only
    discriminative metric within this file. We report it overall and per
    subtype.
    """
    by_subtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in outcomes:
        if o.get("error"):
            continue
        by_subtype[o.get("subtype", "unknown")].append(o)

    def recall_for(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(group)
        n_hitl = sum(1 for o in group if o.get("predicted_hitl"))
        return {"n": n, "flagged_hitl": n_hitl, "missed": n - n_hitl,
                "recall": _safe_div(n_hitl, n)}

    all_valid = [o for o in outcomes if not o.get("error")]
    by_db_all: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in all_valid:
        by_db_all[o["db_id"]].append(o)

    return {
        "overall": recall_for(all_valid),
        "by_subtype": {st: recall_for(g) for st, g in by_subtype.items()},
        "by_db": {db: recall_for(g) for db, g in sorted(by_db_all.items())},
        "n_errors": sum(1 for o in outcomes if o.get("error")),
    }


def _fmt_recall(m: Dict[str, Any]) -> str:
    return (f"n={m['n']:3d}  flagged={m['flagged_hitl']:3d}  "
            f"missed={m['missed']:3d}  recall={m['recall']:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AmbiQT ambiguity-recall benchmark")
    parser.add_argument("--k", type=int, default=10,
                        help="Questions sampled per subtype (default 10)")
    parser.add_argument("--subtype", default="",
                        help=f"Restrict to one subtype. Options: {list(SUBTYPE_DATASET)}")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--server", default="",
                        help="HTTP base URL to hit; omit for in-process execution")
    parser.add_argument("--output", default="data/ambiqt_results.jsonl")
    parser.add_argument("--summary", default="data/ambiqt_summary.json")
    parser.add_argument("--paper-run", action="store_true",
                        help="Copy results+summary+calibration under data/benchmark_runs/ for paper / appendix.")
    parser.add_argument("--run-name", default="",
                        help="Subfolder under data/benchmark_runs/; default: ambiqt_<UTC timestamp>.")
    args = parser.parse_args()

    subtypes = list(SUBTYPE_DATASET)
    if args.subtype:
        if args.subtype not in SUBTYPE_DATASET:
            print(f"Unknown subtype {args.subtype!r}. Supported: {list(SUBTYPE_DATASET)}",
                  file=sys.stderr)
            return 1
        subtypes = [args.subtype]

    # Verify materialized DBs exist before loading anything big.
    missing = []
    for st in subtypes:
        db_root = ROOT / "data" / SUBTYPE_DATASET[st] / "database"
        if not db_root.exists() or not any(db_root.iterdir()):
            missing.append(st)
    if missing:
        print(f"Materialized DBs missing for: {missing}", file=sys.stderr)
        print("Run: python scripts/build_ambiqt_dbs.py", file=sys.stderr)
        return 1

    entries = load_ambiqt_entries(subtypes)
    if not entries:
        print("No AmbiQT entries loaded.", file=sys.stderr)
        return 1

    selected = sample_per_subtype(entries, args.k, args.seed)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already: set[str] = set()
    prior: List[Dict[str, Any]] = []
    if args.resume and output_path.exists():
        with output_path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                    already.add(row["id"])
                    prior.append(row)
                except Exception:  # noqa: BLE001
                    continue
        selected = [q for q in selected if q["id"] not in already]
        print(f"[resume] skipping {len(already)} previously-completed questions")

    runner = ServerRunner(args.server) if args.server else LocalRunner()

    print(f"\nRunning {len(selected)} AmbiQT questions across "
          f"{len(subtypes)} subtype(s). Provider: "
          f"{os.environ.get('LLM_PROVIDER', 'auto')}\n")

    outcomes: List[Dict[str, Any]] = list(prior)
    mode = "a" if args.resume else "w"
    t0 = time.time()
    with output_path.open(mode) as out_f:
        for i, q in enumerate(selected, 1):
            elapsed = time.time() - t0
            eta = (elapsed / i) * (len(selected) - i) if i else 0
            print(f"[{i:3d}/{len(selected)}] ({elapsed:5.0f}s, ~{eta:3.0f}s left) "
                  f"{q['subtype']:<14} [{q['db_id']:<22}] {q['question'][:70]}")
            outcome = runner.run_one(q)
            # Carry forward the AmbiQT-specific fields that _summarize drops.
            outcome["subtype"] = q["subtype"]
            outcome["gold_query1"] = q.get("gold_query1")
            outcome["gold_query2"] = q.get("gold_query2")
            outcome["orig_query"] = q.get("orig_query")
            state = "OK  " if outcome.get("predicted_hitl") else "MISS"
            print(f"       -> {state} pred_hitl={outcome.get('predicted_hitl')} "
                  f"final={outcome.get('final_status')} "
                  f"comp_conf={outcome.get('composite_confidence')}")
            out_f.write(json.dumps(outcome, default=str) + "\n")
            out_f.flush()
            outcomes.append(outcome)
            if args.sleep > 0:
                time.sleep(args.sleep)

    metrics = compute_ambiqt_metrics(outcomes)
    summary_path = ROOT / args.summary
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    with summary_path.open("w") as f:
        json.dump({
            "benchmark": "AmbiQT (col-synonyms + tbl-synonyms, materialized DBs)",
            "run_timestamp_utc": run_ts,
            "ambiqt_repo_path": str((ROOT / "data" / "ambiqt").resolve()),
            "validation_sources": [str(ROOT / "data" / "ambiqt" / "benchmark" / st / "validation.json")
                                   for st in subtypes],
            "materialized_db_roots": {st: str((ROOT / "data" / SUBTYPE_DATASET[st] / "database").resolve())
                                     for st in subtypes},
            "k_per_subtype": args.k,
            "subtypes": subtypes,
            "seed": args.seed,
            "total_run": len(outcomes),
            "metrics": metrics,
            "provider": os.environ.get("LLM_PROVIDER", "auto"),
            "model": os.environ.get("OLLAMA_MODEL")
            or os.environ.get("GOOGLE_MODEL")
            or "",
            "environment": {
                "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", ""),
                "OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", ""),
                "GOOGLE_MODEL": os.environ.get("GOOGLE_MODEL", ""),
            },
        }, f, indent=2, default=str)

    print("\n======================================================================")
    print("AMBIQT AMBIGUITY-RECALL SUMMARY")
    print("======================================================================")
    print(f"Overall   {_fmt_recall(metrics['overall'])}")
    print(f"errors    {metrics['n_errors']}")
    print("\nBy subtype:")
    for st in sorted(metrics["by_subtype"]):
        print(f"  {st:<20} {_fmt_recall(metrics['by_subtype'][st])}")
    print("\nBy database (top 10 by n):")
    rows = sorted(metrics["by_db"].items(), key=lambda kv: -kv[1]["n"])[:10]
    for db_id, m in rows:
        print(f"  {db_id:<22} {_fmt_recall(m)}")
    print(f"\nPer-question outcomes: {output_path}")
    print(f"Summary JSON:          {summary_path}")

    if args.paper_run or args.run_name:
        from scripts.benchmark_artifacts import write_run_bundle
        rname = args.run_name or f"ambiqt_{run_ts}"
        out_dir = write_run_bundle(
            ROOT,
            results_path=output_path,
            summary_path=summary_path,
            run_name=rname,
            extra={"script": "run_ambiqt.py", "k": args.k, "subtypes": subtypes, "seed": args.seed},
        )
        print(f"Paper artifact bundle: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
