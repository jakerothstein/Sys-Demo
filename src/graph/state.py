"""
Agent State Definition for LangGraph Pipeline.
"""
import json
import os
from typing import TypedDict, List, Optional, Literal, Dict, Any
from dataclasses import dataclass, field


class AgentState(TypedDict, total=False):
    """
    The state object that flows through the LangGraph pipeline.
    Each node can read and modify this state.
    """

    # Input
    user_query: str
    session_id: str

    # Disambiguation
    confidence_score: float
    is_ambiguous: bool
    ambiguity_reasons: List[str]
    detected_tables: List[str]
    detected_intent: str
    schema_context: str

    # Pre-Generation (DAIL-SQL)
    draft_sql: str
    query_skeleton: str

    # HITL
    needs_clarification: bool
    clarification_message: str
    user_feedback: Optional[str]

    # SQL Generation
    few_shot_examples: List[dict]
    generated_sqls: List[str]  # Multiple variations for consistency check
    selected_sql: str
    token_confidence_map: List[dict]

    # Consistency Check (entropy-based, replaces LLM-judged consistency)
    consistency_passed: bool
    consistency_analysis: str
    execution_entropy: float          # Shannon entropy over canonicalized result clusters
    semantic_entropy: float           # Shannon entropy over SQL skeleton clusters
    result_clusters: List[dict]       # [{cluster_id, size, sample_sql, sample_rows, hash}]
    skeleton_clusters: List[dict]     # [{skeleton, size, sqls}]
    sql_executions: List[dict]        # per-SQL execution outcomes (cached)
    # Unanimous rows + divergent schema references (AmbiQT-style synonym DBs)
    unanimous_structural_divergence: bool
    schema_diversity: Dict[str, Any]

    # Confidence signals (calibration features)
    sequence_logprob: Optional[float]      # length-normalized average log-prob
    min_token_confidence: Optional[float]
    composite_confidence: float            # weighted combination used by quality gate

    # Expert Evaluation (MoE)
    evaluation_results: List[dict]
    expert_approved: bool

    # Execution
    execution_result: Optional[dict]
    execution_success: bool
    error_trace: Optional[str]

    # Quality Gate
    quality_gate_passed: bool
    quality_gate_reasons: List[str]

    # Retry Logic
    retry_count: int
    max_retries: int
    debug_analysis: Optional[str]

    # Final Output
    final_status: Literal["success", "paused_hitl", "failed", "in_progress"]
    final_message: str


# ---------------------------------------------------------------------------
# Calibration: thresholds + weights loaded from data/calibration.json.
# Run scripts/calibrate_threshold.py to refit these from a labeled set
# (conformal prediction style guarantee on hallucination rate).
# ---------------------------------------------------------------------------

DEFAULT_CALIBRATION: Dict[str, Any] = {
    "alpha": 0.10,  # target hallucination rate among accepted answers
    "thresholds": {
        # Maximum allowed Shannon entropy (bits) over execution-result clusters.
        # 0.0  = all SQLs returned identical results (high confidence).
        # 1.0  = a 50/50 split between two distinct interpretations.
        "execution_entropy_max": 0.6,
        # Maximum allowed Shannon entropy over SQL skeleton clusters.
        # Captures structural disagreement that doesn't show up in results
        # (e.g. one variation crashes, others succeed).
        "semantic_entropy_max": 1.2,
        # Minimum composite confidence to accept without HITL.
        "composite_confidence_min": 0.65,
        # Minimum length-normalized average sequence log-prob (bits).
        # Tokens with very low local probability are a hallucination signal.
        # -10 effectively disables it; tighten after calibration.
        "sequence_logprob_min": -10.0,
        # Unanimous execution + high structural diversity: ask HITL when
        # candidates reference >=2 distinct tables OR >=4 distinct identifiers
        # (see schema_ref_diversity.detect_unanimous_structural_ambiguity).
        "structural_divergence_semantic_min": 1.45,
        "structural_divergence_min_skeletons": 3,
        "structural_divergence_min_distinct_tables": 2,
        "structural_divergence_min_distinct_identifiers": 4,
        "structural_divergence_exec_epsilon": 0.01,
    },
    "weights": {
        # Weighted sum that produces composite_confidence in [0, 1].
        "self_reported": 0.30,        # disambiguation node's LLM confidence
        "execution_consistency": 0.40, # 1 - normalized execution entropy
        "semantic_consistency": 0.15,  # 1 - normalized semantic entropy
        "logprob": 0.15,               # normalized seq logprob
    },
}


def load_calibration(path: Optional[str] = None) -> Dict[str, Any]:
    """Load calibration thresholds; falls back to safe defaults."""
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "calibration.json"
        )
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                loaded = json.load(f)
            # Shallow-merge so partial files still work.
            merged = json.loads(json.dumps(DEFAULT_CALIBRATION))
            for k, v in loaded.items():
                if isinstance(v, dict) and k in merged:
                    merged[k].update(v)
                else:
                    merged[k] = v
            return merged
    except Exception as e:
        print(f"Calibration load warning: {e}. Using defaults.")
    return DEFAULT_CALIBRATION


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    confidence_threshold: float = 0.8
    max_retries: int = 3
    num_sql_variations: int = 3
    consistency_similarity_threshold: float = 0.9
    llm_provider: str = "mock"  # "anthropic", "google", or "mock"
    llm_model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.1

    # Calibration data (loaded from data/calibration.json on init)
    calibration: Dict[str, Any] = field(default_factory=load_calibration)
