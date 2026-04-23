#!/usr/bin/env python3
"""
Conformal Calibration of the Quality-Gate Threshold.

Given a labeled benchmark set, fit the minimum `composite_confidence` we should
require for the pipeline to answer (instead of asking the user) so that the
empirical hallucination rate among accepted answers is <= alpha.

This is a split-conformal style calibration:
    - For each labeled example, run the pipeline and record:
        * composite_confidence  (s)
        * was_correct           (y)
    - For each candidate threshold tau, compute:
        * coverage   = P(s >= tau)
        * error_rate = P(y == 0 | s >= tau)
    - Pick the smallest tau such that error_rate <= alpha (maximizes coverage
      while satisfying the hallucination budget).

Inputs
------
- A JSONL file with { "question": str, "expected_sql": str | None,
                       "expected_row_count": int | null,
                       "is_ambiguous": bool } per line.
  If `--questions` is omitted, falls back to BENCHMARK_QUESTIONS in app.py.

Outputs
-------
- Writes new thresholds to data/calibration.json (preserves weights).

Usage
-----
    python scripts/calibrate_threshold.py --alpha 0.10 --questions data/labeled.jsonl
    python scripts/calibrate_threshold.py --alpha 0.05    # uses default question set
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.pipeline import create_pipeline
from src.graph.state import PipelineConfig, load_calibration
from src.llm.client import create_llm_client


CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "calibration.json",
)


def load_questions(path: Optional[str]) -> List[Dict[str, Any]]:
    """Load labeled questions from JSONL, or fall back to the demo set."""
    if path and os.path.exists(path):
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    print(f"No --questions file given; using demo BENCHMARK_QUESTIONS from app.py")
    try:
        from app import BENCHMARK_QUESTIONS
    except Exception as e:
        print(f"Could not import BENCHMARK_QUESTIONS: {e}")
        return []
    out = []
    for q in BENCHMARK_QUESTIONS:
        out.append({
            "question": q["question"],
            "expected_sql": None,
            "expected_row_count": None,
            # Treat unambiguous questions as 'should be answered correctly'.
            # Ambiguous questions count as "expected to fail confidence" -> y=0.
            "is_ambiguous": q.get("expected_hitl", False),
        })
    return out


def evaluate_correctness(result: Dict[str, Any], label: Dict[str, Any]) -> Optional[int]:
    """
    Return 1 if the pipeline's answer is correct, 0 if wrong, None if unknown.
    Uses cheap heuristics (row count match, success signal) when no gold SQL.
    """
    final_status = result.get("final_status")
    if final_status == "paused_hitl":
        # Refused to answer -> not counted as wrong, not counted as correct
        return None
    if final_status == "failed":
        return 0
    expected_rows = label.get("expected_row_count")
    exec_res = result.get("execution_result") or {}
    actual_rows = exec_res.get("row_count")
    if expected_rows is not None and actual_rows is not None:
        return 1 if actual_rows == expected_rows else 0
    # Fall back: assume label.is_ambiguous => answering is wrong (we should've paused).
    if label.get("is_ambiguous", False):
        return 0
    # Otherwise, if execution succeeded, treat as plausibly correct.
    return 1 if exec_res else 0


def collect_scores(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    config = PipelineConfig(confidence_threshold=0.0, max_retries=2)
    try:
        llm_client = create_llm_client(provider="auto")
    except Exception as e:
        print(f"WARNING: no LLM available ({e}). Calibration will be degenerate.")
        llm_client = None

    pipeline = create_pipeline(config=config, llm_client=llm_client)
    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['question'][:70]}")
        try:
            result = pipeline.run(q["question"], session_id=f"calib_{i}")
        except Exception as e:
            print(f"     pipeline error: {e}")
            continue
        y = evaluate_correctness(result, q)
        if y is None:
            continue
        rows.append({
            "question": q["question"],
            "score": float(result.get("composite_confidence", result.get("confidence_score", 0.0))),
            "execution_entropy": float(result.get("execution_entropy", 0.0) or 0.0),
            "semantic_entropy": float(result.get("semantic_entropy", 0.0) or 0.0),
            "sequence_logprob": result.get("sequence_logprob"),
            "is_correct": int(y),
        })
    return rows


def fit_threshold(rows: List[Dict[str, Any]], alpha: float) -> Dict[str, Any]:
    """
    Find the smallest composite-confidence threshold tau such that the empirical
    error rate among accepted answers (score >= tau) is <= alpha.
    """
    if not rows:
        return {"composite_confidence_min": None, "coverage": 0.0, "error_rate": None,
                "n": 0, "n_accepted": 0}

    candidates = sorted({round(r["score"], 4) for r in rows})
    best = None
    for tau in candidates:
        accepted = [r for r in rows if r["score"] >= tau]
        if not accepted:
            continue
        err = sum(1 - r["is_correct"] for r in accepted) / len(accepted)
        cov = len(accepted) / len(rows)
        if err <= alpha:
            best = {
                "composite_confidence_min": tau,
                "coverage": cov,
                "error_rate": err,
                "n": len(rows),
                "n_accepted": len(accepted),
            }
            break  # smallest tau wins -> highest coverage
    if best is None:
        # No threshold satisfies alpha; pick the strictest one.
        tau = max(r["score"] for r in rows)
        accepted = [r for r in rows if r["score"] >= tau]
        err = sum(1 - r["is_correct"] for r in accepted) / max(1, len(accepted))
        best = {
            "composite_confidence_min": tau,
            "coverage": len(accepted) / len(rows),
            "error_rate": err,
            "n": len(rows),
            "n_accepted": len(accepted),
            "warning": f"No threshold achieved error_rate <= alpha={alpha}.",
        }
    return best


def write_calibration(thresh_min: float, alpha: float, n: int) -> None:
    cal = load_calibration(CALIBRATION_PATH)
    cal["alpha"] = alpha
    cal["fitted_at"] = datetime.utcnow().isoformat() + "Z"
    cal["fitted_n"] = n
    cal.setdefault("thresholds", {})["composite_confidence_min"] = float(thresh_min)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(cal, f, indent=4)
    print(f"\nWrote calibrated thresholds to {CALIBRATION_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Conformal calibration of quality-gate threshold")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Target maximum hallucination rate (default 0.10)")
    parser.add_argument("--questions", type=str, default=None,
                        help="Path to labeled JSONL (else uses demo benchmark)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write calibration.json, just print results")
    args = parser.parse_args()

    print(f"Loading questions...")
    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions.\n")

    if not questions:
        print("No questions to calibrate on. Aborting.")
        sys.exit(1)

    print("Collecting confidence scores from pipeline...")
    rows = collect_scores(questions)
    print(f"\nCollected {len(rows)} scored rows (questions where the pipeline produced a verdict).")

    if len(rows) < 10:
        print("WARNING: Calibration set is very small. Threshold will be unreliable.")

    fit = fit_threshold(rows, args.alpha)
    print("\n=== Calibration Result ===")
    print(json.dumps(fit, indent=2))

    if args.dry_run or fit.get("composite_confidence_min") is None:
        print("\nDry run -- nothing written.")
        return

    write_calibration(fit["composite_confidence_min"], args.alpha, fit["n"])


if __name__ == "__main__":
    main()
