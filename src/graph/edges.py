"""
Decision Edge Functions for LangGraph Pipeline.
These functions determine which node to visit next based on the current state.
"""
from typing import Literal

from .state import AgentState, PipelineConfig


def should_clarify(state: AgentState, config: PipelineConfig) -> Literal["ask_user", "generate_draft_sql"]:
    """
    Decision gate: Should we ask the user for clarification or proceed to draft SQL generation?
    
    Logic:
    1. If confidence is low and no feedback provided -> ask user
    2. If consistency check failed -> ask user
    3. Otherwise -> proceed to draft SQL generation (DAIL-SQL two-step)
    """
    # Check if we already have user feedback
    if state.get('user_feedback'):
        return "generate_draft_sql"
    
    # Check confidence threshold
    if state.get('needs_clarification', False):
        return "ask_user"
    
    return "generate_draft_sql"


def _calibrated_thresholds(config: PipelineConfig) -> dict:
    cal = getattr(config, "calibration", {}) or {}
    return cal.get("thresholds", {}) if isinstance(cal, dict) else {}


def should_evaluate_or_clarify(state: AgentState, config: PipelineConfig) -> Literal["evaluate_sql", "ask_user"]:
    """
    After the execution-entropy consistency check, decide whether to proceed
    to MoE evaluation or ask for clarification.

    Honors:
      - explicit `consistency_passed = False` (e.g. ambiguous result clusters)
      - calibrated execution / semantic entropy thresholds
    """
    if state.get('user_feedback'):
        return "evaluate_sql"

    thresholds = _calibrated_thresholds(config)
    exec_h = float(state.get('execution_entropy', 0.0) or 0.0)
    sem_h = float(state.get('semantic_entropy', 0.0) or 0.0)

    if exec_h > float(thresholds.get('execution_entropy_max', 0.6)):
        return "ask_user"

    # Only treat semantic entropy as a HITL trigger when execution evidence
    # is also inconclusive. Skeleton diversity with unanimous execution
    # results is paraphrase, not ambiguity (e.g. COUNT(*) vs COUNT(id)).
    result_clusters = state.get('result_clusters') or []
    exec_conclusive = len(result_clusters) == 1
    if not exec_conclusive and sem_h > float(thresholds.get('semantic_entropy_max', 1.2)):
        return "ask_user"

    if not state.get('consistency_passed', True):
        return "ask_user"

    return "evaluate_sql"


def should_execute_or_clarify(state: AgentState, config: PipelineConfig) -> Literal["execute_sql", "ask_user"]:
    """
    Read the pre-computed quality gate decision written by quality_gate_node.
    The evaluation logic lives in quality_gate_node so that pass/fail + reasons
    are properly persisted in LangGraph state (edge functions cannot mutate state).
    """
    if state.get('user_feedback'):
        return "execute_sql"
    if state.get('quality_gate_passed', True):
        return "execute_sql"
    return "ask_user"

def should_retry(state: AgentState, config: PipelineConfig) -> Literal["debug", "fail"]:
    """
    After execution failure, decide whether to retry or give up.
    
    Logic:
    1. If retry count < max_retries -> debug and retry
    2. Otherwise -> fail
    """
    retry_count = state.get('retry_count', 0)
    max_retries = state.get('max_retries', config.max_retries)
    
    if retry_count < max_retries:
        return "debug"
    
    return "fail"


def route_after_execution(state: AgentState, config: PipelineConfig) -> Literal["success", "retry"]:
    """
    After SQL execution, route based on success/failure.
    """
    if state.get('execution_success', False):
        return "success"
    
    return "retry"


def route_after_debug(state: AgentState, config: PipelineConfig) -> Literal["generate_sql", "fail"]:
    """
    After debugging, decide whether to regenerate or fail.
    """
    retry_count = state.get('retry_count', 0)
    max_retries = state.get('max_retries', config.max_retries)
    
    if retry_count <= max_retries:
        return "generate_sql"
    
    return "fail"
