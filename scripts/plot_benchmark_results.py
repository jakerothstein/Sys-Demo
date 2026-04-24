#!/usr/bin/env python3
"""
Build digestible comparison charts from saved benchmark outputs.

Inputs (defaults under data/):
  - ambiguity_benchmark_summary.json  (from run_ambiguity_benchmark.py)
  - ambiqt_summary.json               (from run_ambiqt.py)

Outputs PNGs to data/benchmark_plots/ (300 DPI, print-friendly).

Dependencies:
  pip install matplotlib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# project root
ROOT = Path(__file__).resolve().parent.parent


def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    with p.open() as f:
        return json.load(f)


def _ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        print("Install visualization deps:  pip install matplotlib", file=sys.stderr)
        raise SystemExit(1) from e


def plot_ambiguity_overall(overall: Dict[str, Any], out: Path, plt) -> None:
    """Grouped metrics bar for the hand-labeled benchmark."""
    keys = [
        ("accuracy", "Accuracy"),
        ("precision_hitl", "Precision (HITL)"),
        ("recall_hitl", "Recall (HITL)"),
        ("f1_hitl", "F1 (HITL)"),
        ("specificity_answerable", "Specificity (answ.)"),
    ]
    vals = [float(overall.get(k, 0) or 0) for k, _ in keys]
    labels = [l for _, l in keys]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = range(len(labels))
    bars = ax.bar(x, vals, color="#2E86AB", width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#ccc", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Score")
    n = overall.get("n", "?")
    ax.set_title(f"Ambiguity benchmark — overall (n = {n} questions)")
    for rect, v in zip(bars.patches, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(overall: Dict[str, Any], out: Path, plt) -> None:
    tp = int(overall.get("tp", 0))
    fp = int(overall.get("fp", 0))
    tn = int(overall.get("tn", 0))
    fn = int(overall.get("fn", 0))
    # Rows: actual need-HITL? (pos = yes), Cols: predicted HITL
    # Standard 2x2: [[TN, FP],[FN, TP]] with x = Predicted, y = Actual
    import numpy as np

    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_labels = ["Actual: answerable", "Actual: need HITL"]
    col_labels = ["Pred: answerable", "Pred: HITL"]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", color="white" if v > mat.max() / 2 else "black", fontsize=18, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    ax.set_title("Ambiguity benchmark — confusion (counts)")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_by_database(by_db: Dict[str, Any], out: Path, plt, title: str) -> None:
    items = sorted(by_db.items(), key=lambda kv: -kv[1].get("n", 0))[:12]
    if not items:
        return
    names = [k[:22] for k, _ in items]
    acc = [float(v.get("accuracy", 0) or 0) for _, v in items]
    f1h = [float(v.get("f1_hitl", 0) or 0) for _, v in items]

    x = range(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(names))))
    ax.barh([i - w / 2 for i in x], acc, w, label="Accuracy", color="#2E86AB")
    ax.barh([i + w / 2 for i in x], f1h, w, label="F1 (HITL class)", color="#A23B72")
    ax.set_yticks(list(x))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ambiqt(ambi: Dict[str, Any], out_dir: Path, plt) -> None:
    m = ambi.get("metrics") or {}
    overall = m.get("overall") or {}
    n = int(overall.get("n", 0))
    rec = float(overall.get("recall", 0) or 0)
    flagged = int(overall.get("flagged_hitl", 0))
    missed = int(overall.get("missed", 0))

    # Stacked bar: caught vs missed
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh(
        0,
        flagged,
        left=0,
        height=0.45,
        color="#2E86AB",
        label="Flagged (HITL) — correct for AmbiQT",
    )
    ax.barh(0, missed, left=flagged, height=0.45, color="#E94F37", label="Missed (answered)")
    ax.set_xlim(0, max(n, 1))
    ax.set_yticks([0])
    ax.set_yticklabels([f"AmbiQT n={n}"])
    ax.set_xlabel("Questions")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title(f"AmbiQT — ambiguous questions caught vs missed (recall = {rec:.0%})")
    fig.tight_layout()
    fig.savefig(out_dir / "02_ambiqt_caught_vs_missed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # By subtype
    by_st = m.get("by_subtype") or {}
    if not by_st:
        return
    names = list(by_st.keys())
    recalls = [float(by_st[k].get("recall", 0) or 0) for k in names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cols = ("#2E86AB", "#A23B72", "#6B8C42", "#F18F01")
    for i, (name, r) in enumerate(zip(names, recalls)):
        ax.bar(i, r, color=cols[i % len(cols)], width=0.55, label=name.replace("-", " "))
        ax.text(i, r + 0.03, f"{r:.0%}", ha="center", fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("-", " ") for n in names], rotation=0)
    ax.set_ylabel("Recall (HITL)")
    ax.set_ylim(0, 1.1)
    ax.set_title("AmbiQT — recall by ambiguity subtype")
    fig.tight_layout()
    fig.savefig(out_dir / "03_ambiqt_recall_by_subtype.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    by_db = m.get("by_db") or {}
    if by_db:
        items = sorted(by_db.items(), key=lambda kv: -kv[1].get("n", 0))[:15]
        names = [k[:20] for k, _ in items]
        rvals = [float(v.get("recall", 0) or 0) for _, v in items]
        fig, ax = plt.subplots(figsize=(8.5, max(3, 0.3 * len(names))))
        y = range(len(names))
        ax.barh(y, rvals, color="#2E86AB", height=0.65)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1.05)
        for yi, rv in zip(y, rvals):
            ax.text(rv + 0.02, yi, f"{rv:.0%}", va="center", fontsize=8)
        ax.set_title("AmbiQT — recall by database (top 15 by count)")
        fig.tight_layout()
        fig.savefig(out_dir / "04_ambiqt_recall_by_db.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_side_by_side(amb: Dict[str, Any], ambi: Optional[Dict[str, Any]], out: Path, plt) -> None:
    """Single figure comparing headline numbers across benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    o = (amb or {}).get("overall") or {}
    axes[0].set_title("Hand-labeled ambiguity benchmark")
    if o:
        mets = {
            "Acc": o.get("accuracy"),
            "P": o.get("precision_hitl"),
            "R": o.get("recall_hitl"),
            "F1": o.get("f1_hitl"),
            "Spec": o.get("specificity_answerable"),
        }
        xs = list(mets.keys())
        ys = [float(mets[k] or 0) for k in xs]
        axes[0].bar(xs, ys, color="#2E86AB", width=0.55)
        axes[0].set_ylim(0, 1.05)
        for x, y in zip(xs, ys):
            axes[0].text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=9)
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center")
    mo = (ambi or {}).get("metrics", {}).get("overall") or {}
    axes[1].set_title("AmbiQT (all positives — recall = catch rate)")
    if mo:
        rec = float(mo.get("recall", 0) or 0)
        n = int(mo.get("n", 0))
        axes[1].bar(["Recall\n(HITL on ambiguous Q)"], [rec], color="#A23B72", width=0.4)
        axes[1].set_ylim(0, 1.05)
        axes[1].text(0, rec + 0.04, f"{rec:.0%}\n(n={n})", ha="center", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "No AmbiQT summary", ha="center")
    fig.suptitle("Benchmark comparison (same pipeline, different test sets)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ambiguity-summary",
        default="data/ambiguity_benchmark_summary.json",
        type=Path,
    )
    parser.add_argument("--ambiqt-summary", default="data/ambiqt_summary.json", type=Path)
    parser.add_argument("--out-dir", default="data/benchmark_plots", type=Path)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt = _ensure_matplotlib()

    amb = _load_json(ROOT / args.ambiguity_summary)
    ambi = _load_json(ROOT / args.ambiqt_summary)

    if not amb and not ambi:
        print("No summary JSONs found. Run benchmarks first or pass --ambiguity-summary / --ambiqt-summary", file=sys.stderr)
        return 1

    if amb:
        o = amb.get("overall") or {}
        plot_ambiguity_overall(o, out_dir / "00_ambiguity_overall_metrics.png", plt)
        plot_confusion_matrix(o, out_dir / "00b_ambiguity_confusion_matrix.png", plt)
        if amb.get("by_database"):
            plot_by_database(
                amb["by_database"],
                out_dir / "01_ambiguity_by_database.png",
                plt,
                "Ambiguity benchmark — by database (top 12)",
            )

    if ambi:
        plot_ambiqt(ambi, out_dir, plt)

    if amb or ambi:
        plot_side_by_side(amb, ambi, out_dir / "05_benchmark_comparison.png", plt)

    meta = {
        "generated_from": {
            "ambiguity": str(args.ambiguity_summary) if amb else None,
            "ambiqt": str(args.ambiqt_summary) if ambi else None,
        },
        "output_dir": str(out_dir.resolve()),
        "files": [p.name for p in sorted(out_dir.glob("*.png"))],
    }
    (out_dir / "README.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote figures to: {out_dir}/")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  - {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
