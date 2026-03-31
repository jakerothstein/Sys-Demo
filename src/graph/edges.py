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


def should_evaluate_or_clarify(state: AgentState, config: PipelineConfig) -> Literal["evaluate_sql", "ask_user"]:
    """
    After consistency check, decide whether to proceed to MoE evaluation or ask for clarification.
    """
    if not state.get('consistency_passed', True):
        # If variations differ significantly, we might need clarification
        # But only if we haven't already gotten feedback
        if not state.get('user_feedback'):
            return "ask_user"
    
    return "evaluate_sql"


def should_execute_or_clarify(state: AgentState, config: PipelineConfig) -> Literal["execute_sql", "ask_user"]:
    """
    After Mixture of Experts evaluation, check confidence and approval before execution.
    
    Logic:
    - If confidence falls below the config threshold (from semantic phase or MoE penalties), and user hasn't intervened -> ask user
    - If experts explicitly did NOT approve -> ask user
    - Otherwise -> execute_sql
    """
    # Force pause if explicitly denied, regardless of raw score remaining
    if not state.get('expert_approved', True) and not state.get('user_feedback'):
        return "ask_user"
    
    # Check updated overall confidence threshold taking MoE penalties into account
    if state.get('confidence_score', 1.0) < config.confidence_threshold and not state.get('user_feedback'):
        return "ask_user"
    
    return "execute_sql"


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
