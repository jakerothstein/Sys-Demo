#!/usr/bin/env python3
"""
Large-Scale Spider Benchmark Runner

This script runs the 1,034 validation questions from Spider through the Text-to-SQL pipeline.
It is designed to be run without manual HITL intervention to calculate metrics at scale.

Usage:

    # Local (recommended for quick iteration)
    export LLM_PROVIDER="ollama"
    export OLLAMA_MODEL="qwen2.5-coder:14b"
    python scripts/run_large_benchmark.py --count 50 --paper-run

    # Cloud (OpenAI)
    export OPENAI_API_KEY="your-key-here"
    export OPENAI_MODEL="gpt-4o"   # optional (default: gpt-4o)
    export LLM_PROVIDER="openai"
    python scripts/run_large_benchmark.py --paper-run

    # Mirror console to a file (e.g. started from an agent; tail -f the log in another window)
    python scripts/run_large_benchmark.py --count 50 --log-file data/spider/benchmark_run.log

    # If you omit LLM_PROVIDER entirely, the runner defaults to "auto" and
    # will fall back to a local Ollama server when no cloud keys are set.
"""

import os
import sys
import json
import argparse
import itertools
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.load_repo_env import load_repo_env
from src.graph.pipeline import create_pipeline
from src.graph.state import PipelineConfig
from src.llm.client import create_llm_client
from src.sandbox.executor import set_execution_mode, execute_sql


def _rows_as_tuples(exec_result: Optional[Dict[str, Any]]) -> Optional[List[Tuple[str, ...]]]:
    """Convert executor output into a stable, comparable rows representation."""
    if not exec_result or not exec_result.get("success"):
        return None
    cols = list(exec_result.get("columns", []) or [])
    data = list(exec_result.get("data", []) or [])
    rows: List[Tuple[str, ...]] = []
    for row in data:
        if isinstance(row, dict):
            values = [row.get(c) for c in cols] if cols else list(row.values())
        else:
            try:
                values = list(row)
            except TypeError:
                values = [row]
        rows.append(tuple(repr(v) for v in values))
    rows.sort()
    return rows


def _execution_equivalent(
    gold_exec: Optional[Dict[str, Any]],
    pred_exec: Optional[Dict[str, Any]],
    *,
    allow_column_permutation: bool = True,
    max_permute_arity: int = 6,
) -> Tuple[Optional[bool], str]:
    """Return (equivalent?, reason). None means not comparable (gold/pred failed)."""
    gold_rows = _rows_as_tuples(gold_exec)
    pred_rows = _rows_as_tuples(pred_exec)
    if gold_rows is None or pred_rows is None:
        return None, "not_comparable"

    if len(gold_rows) != len(pred_rows):
        return False, "row_count_mismatch"

    arity_gold = len(gold_rows[0]) if gold_rows else 0
    arity_pred = len(pred_rows[0]) if pred_rows else 0
    if arity_gold != arity_pred:
        return False, "arity_mismatch"

    if gold_rows == pred_rows:
        return True, "exact"

    if not allow_column_permutation or arity_gold > max_permute_arity or arity_gold <= 1:
        return False, "value_mismatch"

    indices = tuple(range(arity_gold))
    for perm in itertools.permutations(indices):
        if perm == indices:
            continue
        permuted = [tuple(r[i] for i in perm) for r in pred_rows]
        permuted.sort()
        if permuted == gold_rows:
            return True, "column_permutation"

    return False, "value_mismatch"

def load_spider_questions(input_path: str, max_count: int = None) -> List[Dict[str, Any]]:
    if not os.path.exists(input_path):
        print(f"Dataset not found at {input_path}")
        sys.exit(1)
        
    questions = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            questions.append(json.loads(line))
            
            if max_count and len(questions) >= max_count:
                break
                
    return questions


class LargeBenchmarkRunner:
    def __init__(
        self,
        *,
        compare_to_gold: bool = True,
        confidence_threshold: float = 0.55,
        disambiguation_clear_confidence: float = 0.80,
        num_sql_variations: int = 3,
        max_retries: int = 2,
    ):
        self.results = []
        self.compare_to_gold = compare_to_gold
        
        # Initialize pipeline
        self.config = PipelineConfig(
            confidence_threshold=confidence_threshold,
            disambiguation_clear_confidence=disambiguation_clear_confidence,
            max_retries=max_retries,
            num_sql_variations=num_sql_variations,
            llm_provider=os.environ.get("LLM_PROVIDER", "auto")
        )
        
        try:
            self.llm_client = create_llm_client(provider=self.config.llm_provider)
            if not self.llm_client.is_available():
                print(f"Error: {self.config.llm_provider} client is not available.")
                sys.exit(1)
        except Exception as e:
            print(f"Error initializing LLM client: {e}")
            sys.exit(1)
            
        self.pipeline = create_pipeline(config=self.config, llm_client=self.llm_client)
        
        print("\n" + "=" * 60)
        print("🚀 Large-Scale Spider Benchmark Runner")
        print("=" * 60)
        print(f"LLM Provider: {self.config.llm_provider}")
        print(f"Compare to Spider gold SQL: {self.compare_to_gold}")
        print(f"confidence_threshold: {self.config.confidence_threshold}")
        print(f"disambiguation_clear_confidence: {self.config.disambiguation_clear_confidence}")
        print(f"num_sql_variations: {self.config.num_sql_variations}")
        print(f"max_retries: {self.config.max_retries}")
        print("=" * 60 + "\n")
        
    def run_all(self, questions: List[Dict[str, Any]]):
        print(f"Evaluating {len(questions)} questions...")
        
        try:
            for i, q in enumerate(questions):
                print(f"[{i+1}/{len(questions)}] Processing: {q['question'][:60]}...")
                
                # Set execution mode so the pipeline loads the correct schema for the question
                db_id = q.get('database')
                if db_id:
                    set_execution_mode('benchmark', db_id, 'spider')
                
                result = self.pipeline.run(
                    q['question'],
                    session_id=q['id'],
                    db_id=db_id,
                    dataset='spider',
                )
                hitl_triggered = result.get('final_status') == 'paused_hitl'

                pause_stage = None
                if hitl_triggered:
                    if result.get('needs_clarification'):
                        pause_stage = 'disambiguate'
                    elif result.get('quality_gate_reasons'):
                        pause_stage = 'quality_gate'
                    else:
                        pause_stage = 'unknown'

                gold_sql = q.get('expected_sql')
                gold_exec = None
                exec_equivalent = None
                exec_equiv_reason = "not_run"
                if self.compare_to_gold and gold_sql and result.get('final_status') == 'success':
                    try:
                        gold_exec = execute_sql(gold_sql)
                        exec_equivalent, exec_equiv_reason = _execution_equivalent(
                            gold_exec,
                            result.get('execution_result'),
                        )
                    except Exception as e:
                        exec_equivalent, exec_equiv_reason = None, f"gold_exec_error:{type(e).__name__}"
                
                if hitl_triggered:
                    reasons = result.get('quality_gate_reasons', [])
                    print(f"HITL Reasons: {reasons}")
                    
                self.results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "database": q.get('database'),
                    "hitl_triggered": hitl_triggered,
                    "pause_stage": pause_stage,
                    "confidence_score": result.get("confidence_score", 0),
                    "composite_confidence": result.get("composite_confidence", 0),
                    "execution_entropy": result.get("execution_entropy", 0.0),
                    "semantic_entropy": result.get("semantic_entropy", 0.0),
                    "sequence_logprob": result.get("sequence_logprob", 0.0),
                    "expected_hitl": q.get("expected_hitl", False), # Spider is assumed clear
                    "expected_sql": gold_sql,
                    "sql": result.get("selected_sql", ""),
                    "quality_gate_reasons": result.get('quality_gate_reasons', []) or [],
                    "gold_execution_success": (gold_exec or {}).get('success') if isinstance(gold_exec, dict) else None,
                    "exec_equivalent_to_gold": exec_equivalent,
                    "exec_equivalence_reason": exec_equiv_reason,
                })
        except KeyboardInterrupt:
            print("\nBenchmark interrupted.")
            
        self.print_summary()
        
    def compute_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}
            
        total = len(self.results)
        hitl_triggered = sum(1 for r in self.results if r["hitl_triggered"])
        # For spider (all clear questions), True Negatives (no HITL) is what we want
        true_negatives = total - hitl_triggered 
        specificity = true_negatives / total if total > 0 else 0

        comparable = [r for r in self.results if r.get('exec_equivalent_to_gold') is not None]
        correct_exec = sum(1 for r in comparable if r.get('exec_equivalent_to_gold') is True)
        exec_acc = (correct_exec / len(comparable)) if comparable else 0

        silent_wrong = sum(
            1 for r in comparable
            if (not r.get('hitl_triggered')) and (r.get('exec_equivalent_to_gold') is False)
        )
        silent_wrong_rate = (silent_wrong / len(comparable)) if comparable else 0

        reason_counts = Counter()
        for r in self.results:
            for reason in (r.get('quality_gate_reasons') or []):
                reason_counts[str(reason)] += 1
        
        pause_stage_counts = Counter()
        for r in self.results:
            if r.get("hitl_triggered"):
                pause_stage_counts[str(r.get("pause_stage") or "unknown")] += 1

        return {
            "n_total": total,
            "n_hitl_triggered": hitl_triggered,
            "specificity": specificity,
            "compare_to_gold": self.compare_to_gold,
            "n_comparable": len(comparable),
            "execution_accuracy": exec_acc,
            "n_silent_wrong": silent_wrong,
            "silent_wrong_rate": silent_wrong_rate,
            "pause_stage_counts": dict(pause_stage_counts),
            "quality_gate_reason_counts": dict(reason_counts),
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "disambiguation_clear_confidence": self.config.disambiguation_clear_confidence,
                "num_sql_variations": self.config.num_sql_variations,
                "max_retries": self.config.max_retries,
                "llm_provider": self.config.llm_provider,
            },
            "env": {
                "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", ""),
                "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", ""),
                "OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", ""),
                "GOOGLE_MODEL": os.environ.get("GOOGLE_MODEL", ""),
            },
        }

    def print_summary(self):
        summary = self.compute_summary()
        if not summary:
            return

        print("\n" + "=" * 60)
        print("📊 SPIDER EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Questions Evaluated: {summary['n_total']}")
        print(f"False Positives (Unnecessary Pauses): {summary['n_hitl_triggered']}")
        print(f"Specificity (Clear questions passed): {summary['specificity']:.1%}")
        if summary.get("compare_to_gold"):
            print(f"Comparable vs gold (answered + gold executed): {summary['n_comparable']}")
            print(f"Execution accuracy (on comparable): {summary['execution_accuracy']:.1%}")
            print(
                f"Silent wrong answers (no HITL, wrong result): {summary['n_silent_wrong']} "
                f"({summary['silent_wrong_rate']:.1%})"
            )

        reason_counts = summary.get("quality_gate_reason_counts") or {}
        if reason_counts:
            top = ", ".join([f"{k}={v}" for k, v in Counter(reason_counts).most_common(6)])
            print(f"Top quality-gate reasons: {top}")
        print("=" * 60 + "\n")
        
    def save_results(self, out_path: str):
        with open(out_path, 'w', encoding='utf-8') as f:
            for r in self.results:
                f.write(json.dumps(r) + "\n")
        print(f"Results saved to {out_path}")

    def save_summary(self, summary_path: str):
        summary = self.compute_summary()
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")


class _Tee:
    """Write to multiple streams; flush each write so unbuffered tail -f works."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self._streams:
            n = st.write(s)
            st.flush()
        return n

    def flush(self) -> None:
        for st in self._streams:
            st.flush()

    def isatty(self) -> bool:  # keep color detection sane for the real tty
        return any(getattr(st, "isatty", lambda: False)() for st in self._streams)


def main():
    load_repo_env()
    parser = argparse.ArgumentParser(description='Run massive Spider benchmark')
    parser.add_argument('--count', type=int, default=None, help='Number of questions to run')
    parser.add_argument('--no-gold', action='store_true', help='Skip executing Spider gold SQL for result comparison')
    parser.add_argument('--log-file', default='',
                        help='Write stdout+stderr to this file as well (use tail -f if the real console is hidden).')
    parser.add_argument('--output', default='data/spider/spider_results.jsonl')
    parser.add_argument('--summary', default='data/spider/spider_large_summary.json')
    parser.add_argument('--input-file', default='data/spider/spider_val.jsonl',
                        help='Input file (jsonl format) with Spider questions.')
    parser.add_argument('--paper-run', action='store_true',
                        help='Package results+summary+calibration under data/benchmark_runs/ for paper / appendix.')
    parser.add_argument('--run-name', default='',
                        help='Subfolder name under data/benchmark_runs/; default: spider_<UTC timestamp>.')
    parser.add_argument('--confidence-threshold', type=float, default=0.55)
    parser.add_argument('--disambig-clear', type=float, default=0.70,
                        help='Adaptive prompting cutoff: >= this uses syntactic variants only')
    parser.add_argument('--num-variations', type=int, default=3)
    parser.add_argument('--max-retries', type=int, default=2)
    args = parser.parse_args()

    if args.log_file:
        log_path = os.path.abspath(args.log_file)
        os.makedirs(os.path.dirname(log_path) or os.path.curdir, exist_ok=True)
        _log = open(log_path, 'w', encoding='utf-8')
        sys.stdout = _Tee(sys.__stdout__, _log)
        sys.stderr = _Tee(sys.__stderr__, _log)
        print(f"Logging to {log_path}", flush=True)
    
    questions = load_spider_questions(args.input_file, args.count)
    runner = LargeBenchmarkRunner(
        compare_to_gold=(not args.no_gold),
        confidence_threshold=args.confidence_threshold,
        disambiguation_clear_confidence=args.disambig_clear,
        num_sql_variations=args.num_variations,
        max_retries=args.max_retries,
    )
    runner.run_all(questions)

    # Persist artifacts
    runner.save_results(args.output)
    runner.save_summary(args.summary)

    if args.paper_run or args.run_name:
        from datetime import datetime, timezone
        from pathlib import Path
        from scripts.benchmark_artifacts import write_run_bundle

        run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        run_name = args.run_name or f"spider_{run_ts}"
        root = Path(__file__).resolve().parent.parent
        write_run_bundle(
            root,
            results_path=Path(args.output),
            summary_path=Path(args.summary),
            run_name=run_name,
            extra={
                "script": "run_large_benchmark.py",
                "cli": {
                    "count": args.count,
                    "no_gold": args.no_gold,
                    "confidence_threshold": args.confidence_threshold,
                    "disambig_clear": args.disambig_clear,
                    "num_variations": args.num_variations,
                    "max_retries": args.max_retries,
                },
            },
        )

if __name__ == '__main__':
    main()
