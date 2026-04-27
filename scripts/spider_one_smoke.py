#!/usr/bin/env python3
"""Run a single Spider dev question through the pipeline (for debugging HITL vs success).

Examples:
  LLM_PROVIDER=ollama ./venv/bin/python scripts/spider_one_smoke.py 0
  # API: put OPENAI_API_KEY in .env or export it, then:
  LLM_PROVIDER=openai ./venv/bin/python scripts/spider_one_smoke.py 0
  ./venv/bin/python scripts/spider_one_smoke.py 0 --strict   # exit 1 unless success
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.load_repo_env import load_repo_env
from src.graph.pipeline import create_pipeline
from src.graph.state import PipelineConfig
from src.llm.client import create_llm_client
from src.sandbox.executor import set_execution_mode


def main() -> int:
    load_repo_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("index", type=int, nargs="?", default=0, help="0-based line index in spider_val.jsonl")
    ap.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "auto"),
        help="ollama | openai | auto (default: env LLM_PROVIDER or auto)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if final_status is not success",
    )
    args = ap.parse_args()
    provider = args.provider
    idx = args.index
    path = os.path.join(os.path.dirname(__file__), "..", "data", "spider", "spider_val.jsonl")
    with open(path) as f:
        for i, line in enumerate(f):
            if i != idx:
                continue
            q = json.loads(line)
            break
        else:
            print("index out of range")
            return 1

    db = q.get("database", "")
    print("id:", q.get("id"), "db:", db)
    print("Q:", q.get("question", "")[:120], "...")

    p = provider if provider and provider != "auto" else "auto"
    cfg = PipelineConfig(llm_provider=p)
    client = create_llm_client(provider=cfg.llm_provider)
    if not client.is_available():
        print("LLM client not available; set OPENAI_API_KEY for openai", file=sys.stderr)
        return 1

    pipe = create_pipeline(config=cfg, llm_client=client)
    set_execution_mode("benchmark", db, "spider")
    out = pipe.run(q["question"], session_id=q.get("id", "smoke"))
    print("final_status:", out.get("final_status"))
    print("hitl:", out.get("final_status") == "paused_hitl")
    print("confidence_score:", out.get("confidence_score"))
    print("composite_confidence:", out.get("composite_confidence"))
    print("execution_entropy:", out.get("execution_entropy"))
    print("semantic_entropy:", out.get("semantic_entropy"))
    print("consistency_passed:", out.get("consistency_passed"))
    print("quality_gate_reasons:", out.get("quality_gate_reasons"))
    print("selected_sql:", (out.get("selected_sql") or "")[:200])
    ok = out.get("final_status") == "success"
    if ok:
        print("\nOK: pipeline returned success (auto SQL path, no HITL pause).")
    if args.strict:
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
