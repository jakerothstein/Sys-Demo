"""
Focused validation for the Tier 1 uncertainty stack.

Exercises, with zero LLM calls:
  - Entropy / skeleton / canonicalization / logprob helpers
  - consistency_check_node on ambiguous, unanimous, and mixed-execution states
  - Quality-gate edge logic (should_execute_or_clarify, should_evaluate_or_clarify)
  - Calibration loading
  - format_response surfaces all new signals

Prints a colored PASS/FAIL summary and exits non-zero on any failure so it can
be wired into CI.
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from typing import Any, Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.state import AgentState, PipelineConfig, load_calibration
from src.graph.nodes import (
    consistency_check_node,
    _shannon_entropy_bits,
    _normalize_skeleton,
    _canonicalize_execution_result,
    _logprob_features,
    _normalize_entropy,
    _normalize_logprob,
    _compute_composite_confidence,
)
from src.graph.edges import should_execute_or_clarify, should_evaluate_or_clarify
from src.graph.schema_ref_diversity import (
    detect_unanimous_structural_ambiguity,
    extract_table_names,
    extract_schema_identifiers,
)
from app import format_response


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


_results: List[tuple[str, bool, str]] = []


def check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  {GREEN}PASS{RESET}  {name}")
    except AssertionError as e:
        _results.append((name, False, str(e)))
        print(f"  {RED}FAIL{RESET}  {name}: {e}")
    except Exception as e:  # noqa: BLE001
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  {RED}ERROR{RESET} {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------
def test_shannon_entropy_bounds() -> None:
    assert _shannon_entropy_bits([10]) == 0.0, "single cluster should have zero entropy"
    assert abs(_shannon_entropy_bits([1, 1]) - 1.0) < 1e-9, "2-way uniform = 1 bit"
    assert abs(_shannon_entropy_bits([1, 1, 1, 1]) - 2.0) < 1e-9, "4-way uniform = 2 bits"
    assert _shannon_entropy_bits([]) == 0.0


def test_skeleton_normalization() -> None:
    a = _normalize_skeleton("SELECT name FROM users WHERE id = 5")
    b = _normalize_skeleton("select  NAME from users where id = 99")
    assert a == b, f"skeletons should match across literal/whitespace: {a!r} vs {b!r}"

    c = _normalize_skeleton("SELECT COUNT(*) FROM orders")
    assert c != a, "different structures must produce different skeletons"


def test_canonicalize_results() -> None:
    r1 = {"success": True, "columns": ["id", "name"],
          "data": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
    r2 = {"success": True, "columns": ["name", "id"],
          "data": [{"name": "b", "id": 2}, {"name": "a", "id": 1}]}
    h1 = _canonicalize_execution_result(r1)
    h2 = _canonicalize_execution_result(r2)
    assert h1 is not None and h2 is not None
    assert h1 == h2, f"row/col order should not affect hash: {h1} vs {h2}"

    r3 = {"success": True, "columns": ["id"], "data": [{"id": 99}]}
    assert _canonicalize_execution_result(r3) != h1

    assert _canonicalize_execution_result(None) is None
    assert _canonicalize_execution_result({"success": False}) is None


def test_logprob_features() -> None:
    # Node expects per-token `confidence` in (0, 1] (probability, not log-prob).
    tokens = [
        {"token": "SELECT", "confidence": 0.95},
        {"token": "*",      "confidence": 0.80},
        {"token": "FROM",   "confidence": 0.90},
    ]
    feats = _logprob_features(tokens)
    assert feats["sequence_logprob"] is not None
    assert feats["sequence_logprob"] < 0.0, "log-probs are negative"
    assert feats["min_token_confidence"] == 0.80

    empty = _logprob_features([])
    assert empty["sequence_logprob"] is None
    assert empty["min_token_confidence"] is None


def test_normalize_helpers() -> None:
    # `_normalize_entropy` scales to [0, 1] where 1 = max spread across n candidates.
    assert _normalize_entropy(0.0, 3) == 0.0, "no disagreement -> 0"
    assert abs(_normalize_entropy(math.log2(3), 3) - 1.0) < 1e-9, "max entropy -> 1"
    assert _normalize_entropy(0.5, 1) == 0.0, "single candidate cannot disagree"

    assert _normalize_logprob(None) == 0.5, "missing logprob returns neutral 0.5"
    assert _normalize_logprob(0.0) > _normalize_logprob(-5.0), "higher logprob => higher score"


def test_schema_ref_extraction() -> None:
    # Use length-2+ column names (single-letter aliases are dropped as noise).
    sql = "SELECT aa, bb FROM foo JOIN bar ON foo.id = bar.xx"
    assert "foo" in extract_table_names(sql) and "bar" in extract_table_names(sql)
    idents = extract_schema_identifiers(sql)
    assert "aa" in idents and "bb" in idents and "foo" in idents


def test_detector_unanimous_structural_true_different_tables() -> None:
    """AmbiQT tbl-synonym: same rows possible from different copied tables -> abstain."""
    sqls = [
        "SELECT name FROM orchestra WHERE id = 1",
        "SELECT name FROM ensemble WHERE id = 1",
        "SELECT name FROM symphony WHERE id = 1",
    ]
    ok, d = detect_unanimous_structural_ambiguity(
        sqls,
        result_clusters=[{"size": 3, "hash": "h"}],
        skeleton_clusters=[{"s": 1}, {"s": 1}, {"s": 1}],
        execution_entropy=0.0,
        semantic_entropy=1.585,
        n_success=3,
        n_sqls=3,
        thresholds={},
    )
    assert ok, d
    assert d["union_table_count"] >= 2


def test_detector_paraphrase_count_rejected() -> None:
    """COUNT/COUNT/… paraphrase: at most 3 idents, must not fire structural abstention."""
    sqls = [
        "SELECT COUNT(*) FROM employees",
        "SELECT COUNT(id) FROM employees",
        "SELECT COUNT(DISTINCT id) FROM employees",
    ]
    ok, d = detect_unanimous_structural_ambiguity(
        sqls,
        result_clusters=[{"size": 3}],
        skeleton_clusters=[{"s": 1}, {"s": 1}, {"s": 1}],
        execution_entropy=0.0,
        semantic_entropy=1.585,
        n_success=3,
        n_sqls=3,
        thresholds={},
    )
    assert not ok, d
    assert d.get("reason") == "schema_reference_spread_too_low"


def test_composite_confidence_monotonic() -> None:
    weights = {"self_reported": 0.3, "execution_consistency": 0.4,
               "semantic_consistency": 0.15, "logprob": 0.15}
    hi = _compute_composite_confidence(
        self_reported=0.9,
        execution_entropy=0.0,
        semantic_entropy=0.0,
        n_candidates=3,
        avg_logprob=-0.5,
        weights=weights,
    )
    lo = _compute_composite_confidence(
        self_reported=0.3,
        execution_entropy=1.5,
        semantic_entropy=1.5,
        n_candidates=3,
        avg_logprob=-8.0,
        weights=weights,
    )
    assert hi > lo, f"confident state must score higher: {hi} vs {lo}"
    assert 0.0 <= hi <= 1.0 and 0.0 <= lo <= 1.0


# ---------------------------------------------------------------------------
# 2. consistency_check_node end-to-end (real execution against the sandbox DB)
# ---------------------------------------------------------------------------
def _mk_config() -> PipelineConfig:
    return PipelineConfig(confidence_threshold=0.7, num_sql_variations=3)


def test_consistency_unanimous() -> None:
    """Three identical SQLs over a real sandbox table => zero entropy, passes."""
    cfg = _mk_config()
    state: AgentState = {
        "generated_sqls": [
            "SELECT COUNT(*) FROM employees",
            "SELECT COUNT(*) FROM employees",
            "SELECT COUNT(*) FROM employees",
        ],
        "selected_sql": "SELECT COUNT(*) FROM employees",
        "confidence_score": 0.9,
        "token_confidence_map": [],
    }
    out = consistency_check_node(state, cfg, llm_client=None)
    assert out["execution_entropy"] == 0.0, out
    assert out["consistency_passed"] is True, out.get("consistency_analysis")
    assert out["composite_confidence"] >= 0.65, out["composite_confidence"]
    assert len(out["result_clusters"]) == 1, out["result_clusters"]
    # All three executions should be cached + successful.
    assert all(e["success"] for e in out["sql_executions"]), out["sql_executions"]


def test_consistency_ambiguous_execution() -> None:
    """SQLs hit different tables / filters => distinct result clusters + entropy."""
    cfg = _mk_config()
    state: AgentState = {
        "generated_sqls": [
            "SELECT COUNT(*) FROM employees",
            "SELECT COUNT(*) FROM customers",             # different result
            "SELECT COUNT(*) FROM employees WHERE 1 = 0", # returns 0 -> another cluster
        ],
        "selected_sql": "SELECT COUNT(*) FROM employees",
        "confidence_score": 0.8,
        "token_confidence_map": [],
    }
    out = consistency_check_node(state, cfg, llm_client=None)
    assert out["execution_entropy"] > 0.0, out
    assert len(out["result_clusters"]) >= 2, out["result_clusters"]
    # With this much disagreement, consistency gate must reject.
    assert out["consistency_passed"] is False, out.get("consistency_analysis")


def test_consistency_paraphrase_passes() -> None:
    """
    Three different SQL spellings that all return the same number must NOT be
    flagged as ambiguous (high semantic entropy alone is not a HITL trigger).
    Regression test for the false positive on 'How many employees ... in 2023'.
    """
    cfg = _mk_config()
    state: AgentState = {
        "generated_sqls": [
            "SELECT COUNT(*) FROM employees",
            "SELECT COUNT(id) FROM employees",
            "SELECT COUNT(DISTINCT id) FROM employees",
        ],
        "selected_sql": "SELECT COUNT(*) FROM employees",
        "confidence_score": 0.95,
        "token_confidence_map": [],
    }
    out = consistency_check_node(state, cfg, llm_client=None)
    assert out["execution_entropy"] == 0.0, out
    assert out["semantic_entropy"] > 0.5, out  # skeletons differ
    assert out["consistency_passed"] is True, (
        f"paraphrased equivalent SQLs must pass: {out.get('consistency_analysis')}"
    )


def test_consistency_placeholder_failure() -> None:
    """If SQL generation fell back to 'SELECT 1;', we must NOT report high confidence."""
    cfg = _mk_config()
    state: AgentState = {
        "generated_sqls": ["SELECT 1;", "SELECT 1;", "SELECT 1;"],
        "selected_sql": "SELECT 1;",
        "confidence_score": 0.0,
        "token_confidence_map": [],
    }
    out = consistency_check_node(state, cfg, llm_client=None)
    assert out["consistency_passed"] is False
    assert out["composite_confidence"] == 0.0, out["composite_confidence"]
    assert "placeholder" in out["consistency_analysis"].lower()


def test_consistency_invalid_sql_robust() -> None:
    """Node must not crash if a candidate SQL fails to execute."""
    cfg = _mk_config()
    state: AgentState = {
        "generated_sqls": [
            "SELECT COUNT(*) FROM employees",
            "SELECT * FROM table_that_does_not_exist_xyz",
            "SELECT COUNT(*) FROM employees",
        ],
        "selected_sql": "SELECT COUNT(*) FROM employees",
        "confidence_score": 0.8,
        "token_confidence_map": [],
    }
    out = consistency_check_node(state, cfg, llm_client=None)
    assert "execution_entropy" in out
    assert "semantic_entropy" in out
    assert isinstance(out["sql_executions"], list)
    bad = [e for e in out["sql_executions"] if not e["success"]]
    assert len(bad) == 1, bad  # exactly one candidate must have failed
    # The good candidates still form a stable result cluster.
    assert len(out["result_clusters"]) >= 1


# ---------------------------------------------------------------------------
# 3. Quality-gate edges
# ---------------------------------------------------------------------------
def test_quality_gate_accepts_confident() -> None:
    cfg = _mk_config()
    state: AgentState = {
        "expert_approved": True,
        "composite_confidence": 0.85,
        "sequence_logprob": -1.5,
        "confidence_score": 0.9,
    }
    assert should_execute_or_clarify(state, cfg) == "execute_sql"
    assert state["quality_gate_passed"] is True


def test_quality_gate_rejects_low_composite() -> None:
    cfg = _mk_config()
    state: AgentState = {
        "expert_approved": True,
        "composite_confidence": 0.40,
        "confidence_score": 0.9,
    }
    assert should_execute_or_clarify(state, cfg) == "ask_user"
    assert state["quality_gate_passed"] is False
    assert any("Composite confidence" in r for r in state["quality_gate_reasons"])


def test_quality_gate_rejects_expert_disagree() -> None:
    cfg = _mk_config()
    state: AgentState = {
        "expert_approved": False,
        "composite_confidence": 0.9,
        "confidence_score": 0.9,
    }
    assert should_execute_or_clarify(state, cfg) == "ask_user"
    assert any("Mixture-of-Experts" in r for r in state["quality_gate_reasons"])


def test_quality_gate_user_feedback_bypass() -> None:
    cfg = _mk_config()
    state: AgentState = {
        "expert_approved": False,
        "composite_confidence": 0.0,
        "user_feedback": "yes please run it",
    }
    assert should_execute_or_clarify(state, cfg) == "execute_sql"


def test_entropy_gate_triggers_hitl_on_exec() -> None:
    cfg = _mk_config()
    state: AgentState = {
        "execution_entropy": 1.5,     # above default 0.6
        "semantic_entropy": 0.0,
        "consistency_passed": True,
        "result_clusters": [{"size": 1}, {"size": 1}, {"size": 1}],
    }
    assert should_evaluate_or_clarify(state, cfg) == "ask_user"


def test_entropy_gate_allows_clear_case() -> None:
    cfg = _mk_config()
    state: AgentState = {
        "execution_entropy": 0.0,
        "semantic_entropy": 0.1,
        "consistency_passed": True,
        "result_clusters": [{"size": 3}],
    }
    assert should_evaluate_or_clarify(state, cfg) == "evaluate_sql"


def test_entropy_gate_paraphrase_does_not_trigger() -> None:
    """High semantic entropy with single execution cluster must NOT trigger HITL."""
    cfg = _mk_config()
    state: AgentState = {
        "execution_entropy": 0.0,
        "semantic_entropy": 1.585,    # max for n=3
        "consistency_passed": True,
        "result_clusters": [{"size": 3}],   # one cluster -> conclusive
    }
    assert should_evaluate_or_clarify(state, cfg) == "evaluate_sql"


def test_entropy_gate_high_sem_with_split_exec_triggers() -> None:
    """High semantic entropy WITH ambiguous execution clusters triggers HITL."""
    cfg = _mk_config()
    state: AgentState = {
        "execution_entropy": 0.5,   # below cap
        "semantic_entropy": 1.585,
        "consistency_passed": True,
        "result_clusters": [{"size": 2}, {"size": 1}],   # split -> inconclusive
    }
    assert should_evaluate_or_clarify(state, cfg) == "ask_user"


# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------
def test_calibration_loads() -> None:
    cal = load_calibration()
    assert "thresholds" in cal and "weights" in cal
    for k in ("execution_entropy_max", "semantic_entropy_max",
              "composite_confidence_min", "sequence_logprob_min"):
        assert k in cal["thresholds"], f"missing threshold: {k}"
    w = cal["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6, f"weights must sum to 1.0, got {sum(w.values())}"


def test_pipeline_config_uses_calibration() -> None:
    cfg = PipelineConfig()
    assert isinstance(cfg.calibration, dict)
    assert "thresholds" in cfg.calibration


# ---------------------------------------------------------------------------
# 5. format_response surfaces new signals
# ---------------------------------------------------------------------------
def test_format_response_surfaces_signals() -> None:
    fake_state = {
        "final_status": "success",
        "user_query": "q",
        "selected_sql": "SELECT 1",
        "execution_result": {"data": [], "columns": []},
        "execution_entropy": 0.25,
        "semantic_entropy": 0.10,
        "composite_confidence": 0.82,
        "sequence_logprob": -1.1,
        "min_token_confidence": 0.7,
        "result_clusters": [{"size": 2}, {"size": 1}],
        "skeleton_clusters": [{"size": 3}],
        "sql_executions": [],
        "unanimous_structural_divergence": False,
        "schema_diversity": {"union_table_count": 1},
        "quality_gate_passed": True,
        "quality_gate_reasons": [],
    }
    resp = format_response(fake_state)
    for key in (
        "execution_entropy", "semantic_entropy", "composite_confidence",
        "sequence_logprob", "min_token_confidence",
        "result_clusters", "skeleton_clusters",
        "unanimous_structural_divergence", "schema_diversity",
        "quality_gate_passed", "quality_gate_reasons",
    ):
        assert key in resp, f"format_response missing {key}"
    assert resp["execution_entropy"] == 0.25
    assert resp["quality_gate_passed"] is True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
TESTS: List[tuple[str, Callable[[], None]]] = [
    ("helpers/shannon_entropy_bounds",     test_shannon_entropy_bounds),
    ("helpers/skeleton_normalization",     test_skeleton_normalization),
    ("helpers/canonicalize_results",       test_canonicalize_results),
    ("helpers/logprob_features",           test_logprob_features),
    ("helpers/normalize_helpers",          test_normalize_helpers),
    ("helpers/composite_confidence",       test_composite_confidence_monotonic),
    ("schema_ref/extract",                 test_schema_ref_extraction),
    ("schema_ref/detector_ambiqt_tables", test_detector_unanimous_structural_true_different_tables),
    ("schema_ref/detector_paraphrase_no",  test_detector_paraphrase_count_rejected),
    ("node/consistency_unanimous",         test_consistency_unanimous),
    ("node/consistency_ambiguous",         test_consistency_ambiguous_execution),
    ("node/consistency_paraphrase_passes", test_consistency_paraphrase_passes),
    ("node/consistency_placeholder_fails", test_consistency_placeholder_failure),
    ("node/consistency_handles_bad_sql",   test_consistency_invalid_sql_robust),
    ("edges/quality_gate_accepts",         test_quality_gate_accepts_confident),
    ("edges/quality_gate_rejects_comp",    test_quality_gate_rejects_low_composite),
    ("edges/quality_gate_rejects_expert",  test_quality_gate_rejects_expert_disagree),
    ("edges/user_feedback_bypass",         test_quality_gate_user_feedback_bypass),
    ("edges/entropy_gate_rejects_exec",    test_entropy_gate_triggers_hitl_on_exec),
    ("edges/entropy_gate_accepts_clear",   test_entropy_gate_allows_clear_case),
    ("edges/entropy_gate_paraphrase_ok",   test_entropy_gate_paraphrase_does_not_trigger),
    ("edges/entropy_gate_split_exec",      test_entropy_gate_high_sem_with_split_exec_triggers),
    ("calibration/loads",                  test_calibration_loads),
    ("calibration/pipeline_config",        test_pipeline_config_uses_calibration),
    ("app/format_response",                test_format_response_surfaces_signals),
]


def main() -> int:
    print(f"\n{YELLOW}Uncertainty stack validation{RESET}")
    print("=" * 50)
    for name, fn in TESTS:
        check(name, fn)

    total = len(_results)
    failed = sum(1 for _, ok, _ in _results if not ok)
    passed = total - failed
    print("=" * 50)
    color = GREEN if failed == 0 else RED
    print(f"{color}{passed}/{total} passed{RESET}")
    if failed:
        print(f"\n{RED}Failures:{RESET}")
        for name, ok, err in _results:
            if not ok:
                print(f"  - {name}: {err}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
