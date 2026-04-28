#!/usr/bin/env python3
"""
Weight sensitivity analysis for the composite confidence score.

Baseline weights: w_self=0.65, w_exec=0.15, w_sem=0.10, w_lp=0.10
Gate threshold: composite_confidence_min=0.70, confidence_threshold=0.55

For each weight, we perturb it by ±0.10 (normalizing the remaining weights
proportionally to keep the vector sum = 1), then re-classify all 67 benchmark
questions using the recomputed composite score and report F1 and specificity.
"""
import json
import numpy as np
from typing import List, Dict, Any, Tuple

# ── configuration ────────────────────────────────────────────────────────────
DATA_PATH     = "data/ambiguity_benchmark_results.jsonl"
BASE_WEIGHTS  = {"w_self": 0.65, "w_exec": 0.15, "w_sem": 0.10, "w_lp": 0.10}
COMPOSITE_MIN = 0.70   # composite_confidence_min gate threshold
SELF_FLOOR    = 0.55   # confidence_threshold (self-reported floor)
PERTURB_DELTA = 0.10   # ±0.10

def load_data(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Renormalize so weights sum to 1."""
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

def recompute_composite(row: Dict[str, Any], weights: Dict[str, float]) -> float:
    """
    Recompute composite_confidence using the given weight vector.
    Maps the four raw signals onto [0,1] components using the same
    scaling logic as the production pipeline:
      c_exec  = 1 - execution_entropy/log2(k)   (k=3 variants, max entropy=log2(3)≈1.585)
      c_sem   = 1 - min(semantic_entropy/4.0, 1) (4 bits chosen as saturation point)
      c_lp    = sequence_logprob clipped to [0,1] (already exp'd in some runs; handle both)
      c_self  = confidence_score (already in [0,1])
    """
    c_self = float(row.get("confidence_score") or 0.5)
    
    exec_ent = float(row.get("execution_entropy") or 0.0)
    c_exec   = max(0.0, 1.0 - exec_ent / np.log2(3))   # log2(3) ≈ 1.585

    sem_ent  = float(row.get("semantic_entropy") or 0.0)
    c_sem    = max(0.0, 1.0 - min(sem_ent / 4.0, 1.0))

    lp = row.get("sequence_logprob")
    if lp is None:
        c_lp = 0.5  # neutral fallback (Ollama)
    else:
        lp = float(lp)
        if lp <= 0.0:
            c_lp = float(np.exp(lp))
        else:
            c_lp = float(min(lp, 1.0))

    return (weights["w_self"] * c_self +
            weights["w_exec"] * c_exec +
            weights["w_sem"]  * c_sem  +
            weights["w_lp"]   * c_lp)

def classify(row: Dict[str, Any], weights: Dict[str, float]) -> bool:
    """Return True (predict HITL) if composite < threshold OR self < floor."""
    c_comp = recompute_composite(row, weights)
    c_self = float(row.get("confidence_score", 0.5))
    return (c_comp < COMPOSITE_MIN) or (c_self < SELF_FLOOR)

def metrics(rows: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, float]:
    y_true = [r["expected_hitl"] for r in rows]
    y_pred = [classify(r, weights) for r in rows]

    tp = sum(t and p     for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p  for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p)  for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    spec      = tn / (tn + fp) if (tn + fp) else 0.0
    acc       = (tp + tn) / len(rows)

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": f1, "specificity": spec, "accuracy": acc}

def perturb(base: Dict[str, float], target_key: str, delta: float) -> Dict[str, float]:
    """
    Add `delta` to target_key, then normalize the *other* keys proportionally
    so the total remains 1.0.
    """
    new_val = base[target_key] + delta
    new_val = max(0.0, min(1.0, new_val))
    
    # Distribute the complement among the others proportionally
    remaining = 1.0 - new_val
    other_keys = [k for k in base if k != target_key]
    other_sum  = sum(base[k] for k in other_keys)
    
    result = {target_key: new_val}
    for k in other_keys:
        result[k] = (base[k] / other_sum) * remaining if other_sum > 0 else remaining / len(other_keys)
    return result

def main():
    np.random.seed(42)
    rows = load_data(DATA_PATH)
    print(f"Loaded {len(rows)} benchmark questions.\n")

    # ── Baseline ─────────────────────────────────────────────────────────────
    base_m = metrics(rows, BASE_WEIGHTS)
    print("=" * 72)
    print(f"{'Configuration':<32}  {'F1':>6}  {'Spec.':>6}  {'Acc.':>6}  {'Recall':>6}")
    print("=" * 72)
    print(f"{'Baseline (0.65/0.15/0.10/0.10)':<32}  "
          f"{base_m['f1']:.3f}  {base_m['specificity']:.3f}  "
          f"{base_m['accuracy']:.3f}  {base_m['recall']:.3f}")
    print("-" * 72)

    weight_labels = {
        "w_self": "w_self (self-report)",
        "w_exec": "w_exec (exec. entropy)",
        "w_sem":  "w_sem  (semantic ent.)",
        "w_lp":   "w_lp   (log-prob)",
    }

    results_table = []  # for LaTeX output

    for key in ["w_self", "w_exec", "w_sem", "w_lp"]:
        for direction, sign in [("+0.10", +PERTURB_DELTA), ("-0.10", -PERTURB_DELTA)]:
            perturbed = perturb(BASE_WEIGHTS, key, sign)
            m = metrics(rows, perturbed)
            delta_f1   = m["f1"]   - base_m["f1"]
            delta_spec = m["specificity"] - base_m["specificity"]
            label = f"{weight_labels[key]} {direction}"
            weight_str = f"({perturbed['w_self']:.2f}/{perturbed['w_exec']:.2f}/{perturbed['w_sem']:.2f}/{perturbed['w_lp']:.2f})"
            print(f"  {label:<38}  {m['f1']:.3f}  {m['specificity']:.3f}  "
                  f"{m['accuracy']:.3f}  {m['recall']:.3f}  "
                  f"[ΔF1={delta_f1:+.3f}, ΔSpec={delta_spec:+.3f}]  {weight_str}")
            results_table.append({
                "key": key, "direction": direction,
                "weights": perturbed,
                "f1": m["f1"], "spec": m["specificity"],
                "delta_f1": delta_f1, "delta_spec": delta_spec,
                "recall": m["recall"],
            })
        print()

    print("=" * 72)
    
    # ── LaTeX table snippet ───────────────────────────────────────────────────
    print("\n\n─── LaTeX table snippet ───")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Weight sensitivity analysis. Each row perturbs one weight by")
    print(r"  $\pm 0.10$ (remaining weights renormalized to sum to 1) and reports")
    print(r"  the resulting F1 and specificity on the full 67-question benchmark.}")
    print(r"\label{tab:weight_sensitivity}")
    print(r"\small")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & $w_s$ & $w_e$ & $w_\ell$ & $w_p$ & \textbf{F1} & \textbf{Spec.} \\")
    print(r"\midrule")
    print(f"\\textbf{{Baseline}} & 0.65 & 0.15 & 0.10 & 0.10 & "
          f"\\textbf{{{base_m['f1']:.3f}}} & \\textbf{{{base_m['specificity']:.3f}}} \\\\")
    print(r"\midrule")
    for r in results_table:
        w = r["weights"]
        f1_str   = f"{r['f1']:.3f}"
        spec_str = f"{r['spec']:.3f}"
        df1 = f"({r['delta_f1']:+.3f})" if r['delta_f1'] != 0 else ""
        ds  = f"({r['delta_spec']:+.3f})" if r['delta_spec'] != 0 else ""
        key_tex = {"w_self": "$w_s$", "w_exec": "$w_e$",
                   "w_sem":  "$w_\\ell$", "w_lp": "$w_p$"}[r["key"]]
        label = f"{key_tex}${r['direction']}$"
        print(f"{label} & {w['w_self']:.2f} & {w['w_exec']:.2f} & {w['w_sem']:.2f} & {w['w_lp']:.2f} & "
              f"{f1_str}~{df1} & {spec_str}~{ds} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
