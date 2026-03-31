"""
Agent State Definition for LangGraph Pipeline.
"""
from typing import TypedDict, List, Optional, Literal, Annotated
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
    
    # Consistency Check
    consistency_passed: bool
    consistency_analysis: str
    
    # Expert Evaluation (MoE)
    evaluation_results: List[dict]
    expert_approved: bool
    
    # Execution
    execution_result: Optional[dict]
    execution_success: bool
    error_trace: Optional[str]
    
    # Retry Logic
    retry_count: int
    max_retries: int
    debug_analysis: Optional[str]
    
    # Final Output
    final_status: Literal["success", "paused_hitl", "failed", "in_progress"]
    final_message: str


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
