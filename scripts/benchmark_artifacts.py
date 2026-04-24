"""
Paper-ready benchmark run packaging.

Writes a self-contained directory under data/benchmark_runs/<run_id>/ with:
  - manifest.json   (git hash, platform, model env, run parameters)
  - results.jsonl   (copy of per-question rows)
  - summary.json    (copy of aggregate metrics)
  - calibration.json (snapshot of thresholds used)

Enable from runners via --paper-run (auto-generated run_id) or --artifact-dir PATH.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REDACT_KEYS = ("API_KEY", "api_key", "TOKEN", "SECRET", "PASSWORD", "PAT")


def get_git_commit(repo_root: Path) -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def safe_environ() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in os.environ.items():
        if any(s in k.upper() for s in REDACT_KEYS):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def write_run_bundle(
    repo_root: Path,
    *,
    results_path: Path,
    summary_path: Path,
    run_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    dest = repo_root / "data" / "benchmark_runs" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    r_dst = dest / "results.jsonl"
    s_dst = dest / "summary.json"
    if results_path.is_file():
        shutil.copy2(results_path, r_dst)
    if summary_path.is_file():
        shutil.copy2(summary_path, s_dst)
    cal_src = repo_root / "data" / "calibration.json"
    if cal_src.is_file():
        shutil.copy2(cal_src, dest / "calibration.json")
    prov = {
        "run_id": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": get_git_commit(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "llm_env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("LLM_", "GOOGLE_", "ANTHROPIC_", "OLLAMA_"))
            and not any(s in k.upper() for s in REDACT_KEYS)
        },
    }
    if extra:
        prov["run_config"] = extra
    (dest / "manifest.json").write_text(
        json.dumps(prov, indent=2, default=str), encoding="utf-8"
    )
    (dest / "ARTIFACTS.txt").write_text(
        "results.jsonl  Per-question outcomes (one JSON per line)\n"
        "summary.json     Aggregate metrics / provenance for tables\n"
        "calibration.json Threshold snapshot used for this run\n"
        "manifest.json    Reproducibility metadata\n",
        encoding="utf-8",
    )
    return dest
