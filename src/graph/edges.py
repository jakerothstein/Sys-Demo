"""
Decision Edge Functions for LangGraph Pipeline.
These functions determine which node to visit next based on the current state.
"""
from typing import Literal

from .state import AgentState, PipelineConfig


def should_clarify(state: AgentState, config: PipelineConfig) -> Literal["ask_user", "retrieve_examples"]:
    """
    Decision gate: Should we ask the user for clarification or proceed to SQL generation?
    
    Logic:
    1. If confidence is low and no feedback provided -> ask user
    2. If consistency check failed -> ask user
    3. Otherwise -> proceed to generation
    """
    # Check if we already have user feedback
    if state.get('user_feedback'):
        return "retrieve_examples"
    
    # Check confidence threshold
    if state.get('needs_clarification', False):
        return "ask_user"
    
    return "retrieve_examples"


def should_generate_or_clarify(state: AgentState, config: PipelineConfig) -> Literal["generate_sql", "ask_user"]:
    """
    After consistency check, decide whether to proceed or ask for clarification.
    """
    if not state.get('consistency_passed', True):
        # If variations differ significantly, we might need clarification
        # But only if we haven't already gotten feedback
        if not state.get('user_feedback'):
            return "ask_user"
    
    return "generate_sql"


def should_execute_or_clarify(state: AgentState, config: PipelineConfig) -> Literal["execute_sql", "ask_user"]:
    """
    After SQL generation, check consistency before execution.
    
    Logic:
    - If confidence is very high (>= 0.9), proceed anyway (query is clear)
    - If consistency failed and no user feedback, ask for clarification
    - Otherwise execute
    """
    # High confidence overrides consistency concerns
    if state.get('confidence_score', 0) >= 0.9:
        return "execute_sql"
    
    if not state.get('consistency_passed', True) and not state.get('user_feedback'):
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
