#!/usr/bin/env python3
"""
Ablation study: Self-reported confidence ONLY vs. full composite score.

Reads the existing ambiguity benchmark JSONL (no pipeline re-runs) and
simulates what would happen if the HITL decision used ONLY the LLM's
self-reported confidence_score, ignoring execution entropy, semantic
entropy, and logprob signals.

Two decision strategies are compared:

  1. **Full composite** (actual system):
       predicted_hitl is determined by the pipeline's composite_confidence
       against the calibrated threshold (0.48) plus consistency checks.

  2. **Self-reported only** (ablation):
       predicted_hitl = (confidence_score < threshold)
       Sweeps threshold from 0.30 to 0.95 in 0.05 steps.

Outputs:
  - Console table comparing precision / recall / F1
  - data/ablation_confidence_only.json with full results

Usage:
    python scripts/ablation_confidence_only.py
    python scripts/ablation_confidence_only.py --input data/ambiguity_benchmark_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_binary_metrics(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute binary classification metrics (positive = needs clarification)."""
    tp = fp = tn = fn = 0
    for o in outcomes:
        if o.get("error"):
            continue
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
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "n": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": _safe_div(tn, tn + fp),
    }


def simulate_confidence_only(
    rows: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Simulate a decision rule using ONLY self-reported confidence.

    Decision:  predicted_hitl = (confidence_score < threshold)

    For rows where the pipeline short-circuited before SQL generation
    (truly ambiguous queries detected at disambiguation), the system
    already set predicted_hitl based on is_ambiguous. We replicate that:
    if confidence_score is below threshold, flag as HITL.
    """
    simulated = []
    for row in rows:
        new_row = dict(row)
        conf = row.get("confidence_score")
        if conf is None:
            # If no confidence score available, treat as uncertain
            conf = 0.5
        new_row["predicted_hitl"] = float(conf) < threshold
        simulated.append(new_row)
    return simulated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ablation: self-reported confidence only vs. full composite"
    )
    parser.add_argument(
        "--input", default="data/ambiguity_benchmark_results.jsonl",
        help="Path to the benchmark JSONL results"
    )
    parser.add_argument(
        "--output", default="data/ablation_confidence_only.json",
        help="Output path for ablation results"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.", file=sys.stderr)
        return 1

    # Load all rows
    rows: List[Dict[str, Any]] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("No rows loaded.", file=sys.stderr)
        return 1

    print(f"Loaded {len(rows)} benchmark results from {input_path}\n")

    # ---- Baseline: actual system performance ----
    baseline_metrics = compute_binary_metrics(rows)

    print("=" * 80)
    print("BASELINE (Full Composite System — actual pipeline decisions)")
    print("=" * 80)
    print(f"  n={baseline_metrics['n']}  "
          f"P={baseline_metrics['precision']:.3f}  "
          f"R={baseline_metrics['recall']:.3f}  "
          f"F1={baseline_metrics['f1']:.3f}  "
          f"Acc={baseline_metrics['accuracy']:.3f}  "
          f"Spec={baseline_metrics['specificity']:.3f}")
    print(f"  TP={baseline_metrics['tp']}  FP={baseline_metrics['fp']}  "
          f"TN={baseline_metrics['tn']}  FN={baseline_metrics['fn']}")

    # ---- Ablation: sweep confidence-only thresholds ----
    thresholds = [round(t, 2) for t in
                  [i * 0.05 + 0.30 for i in range(14)]]  # 0.30 to 0.95

    print(f"\n{'=' * 80}")
    print("ABLATION: Self-Reported Confidence Only (no entropy signals)")
    print(f"{'=' * 80}")
    print(f"{'Threshold':>10} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | "
          f"{'Acc':>6} | {'Spec':>6} | {'TP':>4} | {'FP':>4} | {'TN':>4} | {'FN':>4}")
    print("-" * 80)

    ablation_results: List[Dict[str, Any]] = []
    best_f1 = 0.0
    best_thresh = 0.0

    for thresh in thresholds:
        sim_rows = simulate_confidence_only(rows, thresh)
        metrics = compute_binary_metrics(sim_rows)
        ablation_results.append({
            "threshold": thresh,
            **metrics,
        })
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = thresh

        marker = " ◀ best F1" if metrics["f1"] >= best_f1 and thresh == best_thresh else ""
        print(f"{thresh:>10.2f} | {metrics['precision']:>6.3f} | "
              f"{metrics['recall']:>6.3f} | {metrics['f1']:>6.3f} | "
              f"{metrics['accuracy']:>6.3f} | {metrics['specificity']:>6.3f} | "
              f"{metrics['tp']:>4} | {metrics['fp']:>4} | {metrics['tn']:>4} | "
              f"{metrics['fn']:>4}{marker}")

    # ---- Comparison summary ----
    best_ablation = max(ablation_results, key=lambda x: x["f1"])

    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Metric':<20} | {'Full Composite':>15} | {'Best Conf-Only':>15} | {'Delta':>10}")
    print("-" * 65)
    for metric_name in ["precision", "recall", "f1", "accuracy", "specificity"]:
        base_val = baseline_metrics[metric_name]
        abl_val = best_ablation[metric_name]
        delta = abl_val - base_val
        sign = "+" if delta >= 0 else ""
        print(f"{metric_name:<20} | {base_val:>15.3f} | {abl_val:>15.3f} | {sign}{delta:>9.3f}")

    print(f"\nBest confidence-only threshold: {best_ablation['threshold']:.2f}")
    print(f"  → TP={best_ablation['tp']} FP={best_ablation['fp']} "
          f"TN={best_ablation['tn']} FN={best_ablation['fn']}")

    # ---- Diagnostic: per-row comparison ----
    # Find cases where confidence-only disagrees with the full system
    sim_best = simulate_confidence_only(rows, best_ablation["threshold"])
    disagreements: List[Dict[str, Any]] = []
    for orig, sim in zip(rows, sim_best):
        if orig.get("error"):
            continue
        orig_pred = bool(orig.get("predicted_hitl"))
        sim_pred = bool(sim.get("predicted_hitl"))
        if orig_pred != sim_pred:
            disagreements.append({
                "id": orig["id"],
                "question": orig["question"],
                "category": orig.get("category", ""),
                "expected_hitl": orig["expected_hitl"],
                "composite_pred": orig_pred,
                "confonly_pred": sim_pred,
                "confidence_score": orig.get("confidence_score"),
                "composite_confidence": orig.get("composite_confidence"),
                "execution_entropy": orig.get("execution_entropy"),
            })

    print(f"\nDisagreements at best threshold ({best_ablation['threshold']:.2f}): "
          f"{len(disagreements)} cases")
    if disagreements:
        print(f"\n{'ID':<15} | {'Category':<22} | {'Exp':>5} | {'Comp':>5} | "
              f"{'Conf':>5} | {'Score':>6} | {'CompConf':>8} | {'ExecH':>6}")
        print("-" * 95)
        for d in disagreements[:20]:
            print(f"{d['id']:<15} | {d['category']:<22} | "
                  f"{'T' if d['expected_hitl'] else 'F':>5} | "
                  f"{'HITL' if d['composite_pred'] else 'OK':>5} | "
                  f"{'HITL' if d['confonly_pred'] else 'OK':>5} | "
                  f"{d['confidence_score'] or 0:>6.2f} | "
                  f"{d['composite_confidence'] or 0:>8.3f} | "
                  f"{d['execution_entropy'] or 0:>6.2f}")

    # ---- Write output ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "description": "Ablation: self-reported confidence only vs. full composite system",
        "input_file": str(input_path),
        "n_rows": len(rows),
        "baseline_full_composite": baseline_metrics,
        "ablation_confidence_only": ablation_results,
        "best_confidence_only": {
            "threshold": best_ablation["threshold"],
            **{k: v for k, v in best_ablation.items() if k != "threshold"},
        },
        "disagreements_at_best_threshold": disagreements,
    }

    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
