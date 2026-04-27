#!/usr/bin/env python3
"""
Paper / demo preparation helper.

  python scripts/paper_prep.py check      # What exists vs missing; suggested commands
  python scripts/paper_prep.py snapshot   # JSON reproducibility record under data/paper_prep/
  python scripts/paper_prep.py bundle     # Copy summaries, calibration, plots into one folder
  python scripts/paper_prep.py plots      # Regenerate data/benchmark_plots/*.png

Run from repo root (or any cwd — script resolves ROOT).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_artifacts import get_git_commit  # noqa: E402


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _file_status(path: Path) -> Tuple[bool, Optional[int]]:
    if not path.is_file():
        return False, None
    return True, int(path.stat().st_mtime)


def check() -> int:
    items: List[Tuple[str, Path, str]] = [
        (
            "Hand-labeled ambiguity benchmark summary",
            ROOT / "data" / "ambiguity_benchmark_summary.json",
            "python scripts/run_ambiguity_benchmark.py --paper-run",
        ),
        (
            "AmbiQT summary (ambiguous-only recall)",
            ROOT / "data" / "ambiqt_summary.json",
            "python scripts/run_ambiqt.py --paper-run",
        ),
        (
            "Per-question ambiguity JSONL (for appendix / error analysis)",
            ROOT / "data" / "ambiguity_benchmark_results.jsonl",
            "(generated with ambiguity benchmark; same command as above)",
        ),
        (
            "Calibration / threshold snapshot",
            ROOT / "data" / "calibration.json",
            "python scripts/calibrate_threshold.py  # if you need to refresh",
        ),
        (
            "Benchmark figure manifest",
            ROOT / "data" / "benchmark_plots" / "README.json",
            "python scripts/paper_prep.py plots",
        ),
        (
            "Spider large benchmark summary (silent-wrong analysis)",
            ROOT / "data" / "spider" / "spider_large_summary.json",
            "LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5-coder:14b python scripts/run_large_benchmark.py --paper-run",
        ),
    ]
    print("Paper artifact check\n" + "=" * 60)
    missing_cmds: List[str] = []
    for label, path, fix in items:
        ok, _ = _file_status(path)
        status = "ok " if ok else "NO "
        print(f"  [{status}] {label}")
        print(f"        {path.relative_to(ROOT)}")
        if not ok:
            missing_cmds.append(fix)
    print("\nSuggested full refresh (needs LLM / API or local Ollama):\n")
    print("  python scripts/run_ambiguity_benchmark.py --paper-run --sleep 2")
    print("  python scripts/run_ambiqt.py --paper-run --sleep 2")
    print("  python scripts/paper_prep.py plots")
    print("  python scripts/paper_prep.py bundle")
    if missing_cmds:
        print("\nMissing files — run:\n")
        for c in dict.fromkeys(missing_cmds):
            print(f"  {c}")
    return 0 if all(_file_status(p)[0] for _, p, _ in items) else 1


def snapshot() -> int:
    out_dir = ROOT / "data" / "paper_prep"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"paper_snapshot_{_ts()}.json"

    amb_path = ROOT / "data" / "ambiguity_benchmark_summary.json"
    ambi_path = ROOT / "data" / "ambiqt_summary.json"
    cal_path = ROOT / "data" / "calibration.json"

    amb = _load_json(amb_path)
    ambi = _load_json(ambi_path)
    cal = _load_json(cal_path)

    doc: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": get_git_commit(ROOT),
        "python": sys.version.split()[0],
        "key_files": {},
        "metrics_excerpt": {},
        "reproduce": [
            "python scripts/run_ambiguity_benchmark.py --paper-run",
            "python scripts/run_ambiqt.py --paper-run",
            "LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5-coder:14b python scripts/run_large_benchmark.py --paper-run",
            "python scripts/paper_prep.py plots",
            "python scripts/paper_prep.py bundle",
        ],
    }

    for label, path in (
        ("ambiguity_benchmark_summary", amb_path),
        ("ambiqt_summary", ambi_path),
        ("spider_large_summary", ROOT / "data" / "spider" / "spider_large_summary.json"),
        ("calibration", cal_path),
        ("ambiguity_results_jsonl", ROOT / "data" / "ambiguity_benchmark_results.jsonl"),
    ):
        ok, mtime = _file_status(path)
        doc["key_files"][label] = {
            "path": str(path.relative_to(ROOT)),
            "present": ok,
            "mtime_epoch": mtime,
        }

    if amb and amb.get("overall"):
        doc["metrics_excerpt"]["ambiguity_overall"] = {
            k: amb["overall"].get(k)
            for k in ("n", "accuracy", "precision_hitl", "recall_hitl", "f1_hitl")
        }
    if ambi and ambi.get("metrics", {}).get("overall"):
        doc["metrics_excerpt"]["ambiqt_overall"] = ambi["metrics"]["overall"]

    dest = out_dir / name
    dest.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {dest.relative_to(ROOT)}")
    return 0


def bundle() -> int:
    bdir = ROOT / "data" / "paper_prep" / f"bundle_{_ts()}"
    bdir.mkdir(parents=True, exist_ok=True)
    copies = [
        ROOT / "data" / "ambiguity_benchmark_summary.json",
        ROOT / "data" / "ambiqt_summary.json",
        ROOT / "data" / "calibration.json",
        ROOT / "data" / "ambiguity_benchmark_results.jsonl",
        ROOT / "data" / "benchmark_plots" / "README.json",
    ]
    plot_dir = ROOT / "data" / "benchmark_plots"
    for src in copies:
        if src.is_file():
            shutil.copy2(src, bdir / src.name)
    if plot_dir.is_dir():
        png_dest = bdir / "benchmark_plots"
        png_dest.mkdir(exist_ok=True)
        for png in sorted(plot_dir.glob("*.png")):
            shutil.copy2(png, png_dest / png.name)

    runs = ROOT / "data" / "benchmark_runs"
    if runs.is_dir():
        shutil.copytree(runs, bdir / "benchmark_runs", dirs_exist_ok=True)

    manifest = {
        "bundle_created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": get_git_commit(ROOT),
        "contents": [p.relative_to(bdir).as_posix() for p in sorted(bdir.rglob("*")) if p.is_file()],
    }
    (bdir / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote bundle: {bdir.relative_to(ROOT)}/")
    return 0


def plots() -> int:
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_benchmark_results.py")],
        cwd=str(ROOT),
    )
    return int(rc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper / demo preparation")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Verify benchmark outputs and print next steps")
    sub.add_parser("snapshot", help="Write reproducibility JSON to data/paper_prep/")
    sub.add_parser("bundle", help="Copy summaries, calibration, plots, benchmark_runs")
    sub.add_parser("plots", help="Regenerate benchmark PNGs via plot_benchmark_results.py")
    args = parser.parse_args()
    if args.cmd == "check":
        return check()
    if args.cmd == "snapshot":
        return snapshot()
    if args.cmd == "bundle":
        return bundle()
    if args.cmd == "plots":
        return plots()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
