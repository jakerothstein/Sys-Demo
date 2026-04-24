#!/usr/bin/env python3
"""
Ambiguity-Detection Benchmark Runner.

Loads data/ambiguity_benchmark.json (questions labeled as either answerable or
clarification-needed), runs each through the pipeline without user feedback,
and measures how accurately the pipeline classifies ambiguity.

Positive class  = "needs clarification" (expected_hitl = true)
Negative class  = "answerable"          (expected_hitl = false)

Prediction mapping:
  result.final_status == 'paused_hitl'  ->  predicted_hitl = True
  else                                  ->  predicted_hitl = False

Reports overall accuracy, confusion matrix, precision / recall / F1 for the
HITL class, per-database breakdown, and per-category accuracy. Writes a full
JSONL of per-question outcomes for offline analysis.

Usage:
  python scripts/run_ambiguity_benchmark.py
  python scripts/run_ambiguity_benchmark.py --count 10
  python scripts/run_ambiguity_benchmark.py --db company_analytics --sleep 4
  python scripts/run_ambiguity_benchmark.py --balanced 20   # 20 of each class
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Binary classification: positive class = needs clarification.
    """
    tp = fp = tn = fn = 0
    for o in outcomes:
        if o.get("error"):
            continue  # exclude crashed runs from metrics
        exp = bool(o["expected_hitl"])
        pred = bool(o["predicted_hitl"])
        if exp and pred:
            tp += 1
        elif exp and not pred:
            fn += 1
        elif not exp and pred:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)

    return {
        "n": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision_hitl": precision,
        "recall_hitl": recall,
        "f1_hitl": f1,
        "specificity_answerable": specificity,
    }


def breakdown_by(key: str, outcomes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in outcomes:
        groups[str(o.get(key, "?"))].append(o)
    return {k: compute_metrics(v) for k, v in groups.items()}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _summarize(q: Dict[str, Any], result: Dict[str, Any],
               error: Optional[str] = None) -> Dict[str, Any]:
    predicted_hitl = result.get("final_status") == "paused_hitl"
    sql_vars = result.get("generated_sqls") or result.get("sql_variations")
    if not sql_vars and isinstance(result.get("sql"), str):
        sql_vars = []
    return {
        "id": q["id"],
        "db_id": q["db_id"],
        "dataset": q.get("dataset"),
        "question": q["question"],
        "category": q.get("category", ""),
        "expected_hitl": q["expected_hitl"],
        "predicted_hitl": None if error else predicted_hitl,
        "final_status": "error" if error else result.get("final_status"),
        "correct": None if error else (predicted_hitl == q["expected_hitl"]),
        "error": error,
        "confidence_score": result.get("confidence_score"),
        "composite_confidence": result.get("composite_confidence"),
        "execution_entropy": result.get("execution_entropy"),
        "semantic_entropy": result.get("semantic_entropy"),
        "unanimous_structural_divergence": result.get("unanimous_structural_divergence"),
        "schema_diversity": result.get("schema_diversity"),
        "consistency_passed": result.get("consistency_passed"),
        "consistency_analysis": (result.get("consistency_analysis") or "")[:4000],
        "quality_gate_reasons": result.get("quality_gate_reasons") or [],
        "selected_sql": (result.get("selected_sql") or "")[:4000],
        "sql_variations": [((s or "")[:4000]) for s in (sql_vars or [])[:20]],
    }


class LocalRunner:
    """Runs the pipeline in-process (requires GOOGLE_API_KEY / ANTHROPIC_API_KEY)."""

    def __init__(self, confidence_threshold: float = 0.55,
                 num_sql_variations: int = 3,
                 max_retries: int = 2) -> None:
        from src.graph.pipeline import create_pipeline
        from src.graph.state import PipelineConfig
        from src.llm.client import create_llm_client
        from src.sandbox.executor import set_execution_mode

        self._set_execution_mode = set_execution_mode
        self.config = PipelineConfig(
            confidence_threshold=confidence_threshold,
            disambiguation_clear_confidence=0.85,
            max_retries=max_retries,
            num_sql_variations=num_sql_variations,
            llm_provider=os.environ.get("LLM_PROVIDER", "auto"),
        )
        self.llm_client = create_llm_client(provider="auto")
        self.pipeline = create_pipeline(config=self.config, llm_client=self.llm_client)
        self._db_cache: Optional[str] = None

    def _switch_db(self, db_id: str, preferred_dataset: Optional[str] = None) -> None:
        if db_id == self._db_cache:
            return
        # Try the caller-specified dataset first, then fall back through the
        # known buckets. `ambiqt_colsyn` / `ambiqt_tblsyn` = materialized
        # AmbiQT syn DBs; `ambiqt` = raw Spider copies from the AmbiQT zip.
        order = ["custom", "bird", "spider", "ambiqt_colsyn", "ambiqt_tblsyn", "ambiqt"]
        if preferred_dataset and preferred_dataset in order:
            order.remove(preferred_dataset)
            order.insert(0, preferred_dataset)
        ok = False
        for ds in order:
            if self._set_execution_mode("benchmark", db_id, ds):
                ok = True
                break
        if not ok:
            raise RuntimeError(f"Could not switch to database '{db_id}'")
        self._db_cache = db_id

    def run_one(self, q: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._switch_db(q["db_id"], preferred_dataset=q.get("dataset"))
            result = self.pipeline.run(
                q["question"], session_id=f"bench_{q['id']}_{int(time.time())}"
            )
        except Exception as e:  # noqa: BLE001
            return _summarize(q, {}, error=f"{type(e).__name__}: {e}")
        return _summarize(q, result)


class ServerRunner:
    """
    Hits a running Flask app (default http://127.0.0.1:5000).
    Avoids a second pipeline init and reuses the already-warm vector store.
    """

    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._db_cache: Optional[str] = None

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _switch_db(self, db_id: str, preferred_dataset: Optional[str] = None) -> None:
        if db_id == self._db_cache:
            return
        # The server uses {mode, db_id, dataset}. Try each dataset bucket.
        last_err: Optional[str] = None
        order = ["custom", "bird", "spider", "ambiqt_colsyn", "ambiqt_tblsyn", "ambiqt"]
        if preferred_dataset and preferred_dataset in order:
            order.remove(preferred_dataset)
            order.insert(0, preferred_dataset)
        for dataset in order:
            try:
                r = self._post(
                    "/api/databases/switch",
                    {"mode": "benchmark", "db_id": db_id, "dataset": dataset},
                )
                if r.get("success"):
                    self._db_cache = db_id
                    return
                last_err = r.get("error", "unknown")
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}"
                if e.code not in (400, 404):
                    raise
        raise RuntimeError(f"Could not switch server to '{db_id}': {last_err}")

    def run_one(self, q: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._switch_db(q["db_id"], preferred_dataset=q.get("dataset"))
            r = self._post(
                "/api/query",
                {"query": q["question"],
                 "session_id": f"bench_{q['id']}_{int(time.time())}"},
            )
        except Exception as e:  # noqa: BLE001
            return _summarize(q, {}, error=f"{type(e).__name__}: {e}")

        # /api/query returns format_response, which uppercases status.
        ui_status = (r.get("status") or "").upper()
        is_hitl = ui_status == "PAUSED_HITL"
        internal_status = {
            "SUCCESS": "success",
            "PAUSED_HITL": "paused_hitl",
            "FAILED": "failed",
            "IN_PROGRESS": "in_progress",
        }.get(ui_status, ui_status.lower() or "unknown")

        adapted = {
            "final_status": internal_status,
            "confidence_score": r.get("confidence"),
            "composite_confidence": r.get("composite_confidence"),
            "execution_entropy": r.get("execution_entropy"),
            "semantic_entropy": r.get("semantic_entropy"),
            "unanimous_structural_divergence": r.get("unanimous_structural_divergence"),
            "schema_diversity": r.get("schema_diversity"),
            "consistency_passed": r.get("consistency_passed"),
            "consistency_analysis": r.get("consistency_analysis", ""),
            "quality_gate_reasons": r.get("quality_gate_reasons"),
            "selected_sql": r.get("sql") or "",
            "generated_sqls": r.get("sql_variations") or r.get("generated_sqls", []),
        }
        # Sanity: is_ambiguous sometimes set even when status=SUCCESS
        if not is_hitl and r.get("is_ambiguous"):
            adapted["final_status"] = "paused_hitl"
        return _summarize(q, adapted)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _select(questions: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    pool = questions
    if args.db:
        pool = [q for q in pool if q["db_id"] == args.db]
    if args.balanced:
        pos = [q for q in pool if q["expected_hitl"]]
        neg = [q for q in pool if not q["expected_hitl"]]
        random.Random(args.seed).shuffle(pos)
        random.Random(args.seed + 1).shuffle(neg)
        n = min(args.balanced, len(pos), len(neg))
        pool = pos[:n] + neg[:n]
        random.Random(args.seed + 2).shuffle(pool)
    elif args.count:
        random.Random(args.seed).shuffle(pool)
        pool = pool[: args.count]
    return pool


def _fmt_metrics(m: Dict[str, Any]) -> str:
    return (f"n={m['n']:3d}  acc={m['accuracy']:.2f}  "
            f"P={m['precision_hitl']:.2f}  R={m['recall_hitl']:.2f}  "
            f"F1={m['f1_hitl']:.2f}  spec={m['specificity_answerable']:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ambiguity-detection benchmark")
    parser.add_argument("--benchmark", default="data/ambiguity_benchmark.json")
    parser.add_argument("--output", default="data/ambiguity_benchmark_results.jsonl")
    parser.add_argument("--summary", default="data/ambiguity_benchmark_summary.json")
    parser.add_argument("--count", type=int, default=0,
                        help="Random sample size; 0 = run all")
    parser.add_argument("--balanced", type=int, default=0,
                        help="Use N of each class (overrides --count)")
    parser.add_argument("--db", default="", help="Filter to one database")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between questions (rate limit)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action="store_true",
                        help="Skip question ids already in --output")
    parser.add_argument("--server", default="",
                        help="Hit a running Flask app, e.g. http://127.0.0.1:5000. "
                             "If unset, runs pipeline in-process.")
    parser.add_argument("--paper-run", action="store_true",
                        help="Package results+summary+calibration under data/benchmark_runs/ for a paper / appendix.")
    parser.add_argument("--run-name", default="",
                        help="Subfolder name under data/benchmark_runs/; default: ambiguity_<UTC timestamp>.")
    args = parser.parse_args()

    bench_path = Path(args.benchmark)
    with bench_path.open() as f:
        data = json.load(f)
    questions = data["questions"]

    selected = _select(questions, args)
    if not selected:
        print("No questions matched filters.")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already: set[str] = set()
    if args.resume and output_path.exists():
        with output_path.open() as f:
            for line in f:
                try:
                    already.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    continue
        selected = [q for q in selected if q["id"] not in already]
        print(f"[resume] skipping {len(already)} previously-completed questions")

    runner = ServerRunner(args.server) if args.server else LocalRunner()
    outcomes: List[Dict[str, Any]] = []

    # preserve previously-saved outcomes so the final summary covers everything
    if args.resume and output_path.exists():
        with output_path.open() as f:
            for line in f:
                try:
                    outcomes.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue

    print(f"\nRunning {len(selected)} questions "
          f"(sleep={args.sleep}s between, db_filter={args.db or 'all'})\n")

    mode = "a" if args.resume else "w"
    t0 = time.time()
    with output_path.open(mode) as out_f:
        for i, q in enumerate(selected, 1):
            elapsed = time.time() - t0
            eta = (elapsed / i) * (len(selected) - i) if i else 0
            print(f"[{i:3d}/{len(selected)}] ({elapsed:5.0f}s elapsed, ~{eta:3.0f}s left) "
                  f"{q['id']:<12} [{q['db_id']:<17}] "
                  f"exp_hitl={str(q['expected_hitl']):<5}  "
                  f"{q['question'][:70]}")
            outcome = runner.run_one(q)
            outcomes.append(outcome)
            out_f.write(json.dumps(outcome) + "\n")
            out_f.flush()

            tag = "OK " if outcome.get("correct") else "MISS"
            print(f"       -> {tag} pred_hitl={outcome['predicted_hitl']} "
                  f"final={outcome['final_status']} "
                  f"comp_conf={outcome.get('composite_confidence')}")

            if args.sleep and i < len(selected):
                time.sleep(args.sleep)

    print("\n" + "=" * 70)
    print("AMBIGUITY BENCHMARK SUMMARY")
    print("=" * 70)
    overall = compute_metrics(outcomes)
    print(f"Overall  {_fmt_metrics(overall)}")

    print("\nConfusion matrix (positive = needs clarification):")
    print(f"  TP (HITL, correctly flagged)   = {overall['tp']}")
    print(f"  FN (HITL, MISSED by pipeline)  = {overall['fn']}")
    print(f"  FP (answerable, over-flagged)  = {overall['fp']}")
    print(f"  TN (answerable, correctly run) = {overall['tn']}")

    print("\nBy database:")
    for db, m in breakdown_by("db_id", outcomes).items():
        print(f"  {db:<20} {_fmt_metrics(m)}")

    print("\nBy category (top-10 by count):")
    by_cat = breakdown_by("category", outcomes)
    for cat, m in sorted(by_cat.items(), key=lambda kv: -kv[1]["n"])[:10]:
        print(f"  {cat:<24} {_fmt_metrics(m)}")

    errors = [o for o in outcomes if o.get("error")]
    if errors:
        print(f"\n{len(errors)} runs raised an exception and were excluded from metrics:")
        for e in errors[:5]:
            print(f"  {e['id']}: {e['error']}")

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    summary = {
        "run_timestamp_utc": run_ts,
        "benchmark_path": str(bench_path.resolve()),
        "benchmark_header": {k: v for k, v in data.items() if k != "questions"},
        "cli": {k: getattr(args, k) for k in (
            "count", "balanced", "db", "seed", "resume", "server", "output", "summary",
        ) if hasattr(args, k)},
        "environment": {
            "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", ""),
            "OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", ""),
            "GOOGLE_MODEL": os.environ.get("GOOGLE_MODEL", ""),
        },
        "overall": overall,
        "by_database": breakdown_by("db_id", outcomes),
        "by_category": breakdown_by("category", outcomes),
        "errors": [{"id": e["id"], "error": e["error"]} for e in errors],
        "n_total": len(outcomes),
        "n_errors": len(errors),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nPer-question outcomes: {output_path}")
    print(f"Summary JSON:          {args.summary}")

    if args.paper_run or args.run_name:
        from scripts.benchmark_artifacts import write_run_bundle
        root = Path(__file__).resolve().parent.parent
        rname = args.run_name or f"ambiguity_{run_ts}"
        out_dir = write_run_bundle(
            root,
            results_path=output_path,
            summary_path=Path(args.summary),
            run_name=rname,
            extra={"script": "run_ambiguity_benchmark.py", "cli": summary["cli"]},
        )
        print(f"Paper artifact bundle: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
