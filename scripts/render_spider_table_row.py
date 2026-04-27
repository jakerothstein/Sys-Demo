#!/usr/bin/env python3
"""Render a LaTeX table row from a Spider summary.

This consumes the JSON summary written by `scripts/run_large_benchmark.py` and
prints a single LaTeX row matching Table `tab:spider` in the paper.

Usage:
  python scripts/render_spider_table_row.py data/spider/spider_large_summary.json
  python scripts/render_spider_table_row.py data/spider/spider_50_summary.json

Or, after running with --paper-run:
    python scripts/render_spider_table_row.py --latest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _pct(x: float, *, decimals: int = 1) -> str:
    try:
        x_f = float(x)
    except (TypeError, ValueError):
        x_f = 0.0
    return f"{x_f * 100:.{decimals}f}\\%"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_spider_summary(repo_root: Path) -> Path:
    runs_dir = repo_root / "data" / "benchmark_runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {runs_dir}")

    candidates = []
    for p in runs_dir.glob("spider_*/summary.json"):
        try:
            candidates.append((p.stat().st_mtime, p))
        except OSError:
            continue

    if not candidates:
        raise FileNotFoundError(
            "No Spider run bundles found under data/benchmark_runs/. "
            "Run: python scripts/run_large_benchmark.py --paper-run"
        )

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Print a LaTeX table row for Spider silent-failure summary",
    )
    ap.add_argument("summary_path", nargs="?", default="", help="Path to summary JSON")
    ap.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest data/benchmark_runs/spider_*/summary.json",
    )
    ap.add_argument("--decimals", type=int, default=1, help="Percent decimals")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    try:
        if args.latest or not args.summary_path:
            path = _find_latest_spider_summary(repo_root)
        else:
            path = Path(args.summary_path)
    except FileNotFoundError as e:
        spider_dir = repo_root / "data" / "spider"
        unbundled = sorted(spider_dir.glob("spider*_summary.json")) if spider_dir.is_dir() else []
        print(str(e), flush=True)
        if unbundled:
            print("\nFound unbundled Spider summary files:", flush=True)
            for p in unbundled[-8:]:
                rel = p.relative_to(repo_root)
                print(f"  - {rel}", flush=True)
            print(
                "\nTip: if your large run is still executing, wait until it finishes and writes a "
                "data/benchmark_runs/spider_*/summary.json (via --paper-run), then re-run:\n"
                "  python -m scripts.render_spider_table_row --latest\n\n"
                "Or render a specific file directly, e.g.:\n"
                f"  python -m scripts.render_spider_table_row {unbundled[-1].relative_to(repo_root)}",
                flush=True,
            )
        return 1
    doc = _load_json(path)

    n_total = int(doc.get("n_total") or 0)
    pauses = int(doc.get("n_hitl_triggered") or 0)
    specificity = float(doc.get("specificity") or 0.0)
    comparable = int(doc.get("n_comparable") or 0)
    exec_acc = float(doc.get("execution_accuracy") or 0.0)
    n_silent_wrong = int(doc.get("n_silent_wrong") or 0)
    silent_wrong_rate = float(doc.get("silent_wrong_rate") or 0.0)

    row = (
        f"{n_total} & {pauses} & {_pct(specificity, decimals=args.decimals)} "
        f"& {comparable} & {_pct(exec_acc, decimals=args.decimals)} "
        f"& {n_silent_wrong} ({_pct(silent_wrong_rate, decimals=args.decimals)}) \\\\"  # noqa: W605
    )
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
