"""
LangGraph State Machine Pipeline.
Compiles the graph with nodes, edges, and interrupt points.
"""
from typing import Dict, Any, Optional
from functools import partial

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: langgraph not installed. Using fallback pipeline.")

from .state import AgentState, PipelineConfig
from .nodes import (
    disambiguate_node, generate_draft_sql_node, retrieve_examples_node, generate_sql_node,
    consistency_check_node, evaluate_sql_node, execute_sql_node, debug_node,
    finalize_success_node, finalize_hitl_node, finalize_failure_node
)
from .edges import (
    should_clarify, should_evaluate_or_clarify, should_execute_or_clarify, 
    route_after_execution, route_after_debug
)


class TextToSQLGraph:
    """
    LangGraph-based Text-to-SQL pipeline with state machine orchestration.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, llm_client=None):
        self.config = config or PipelineConfig()
        self.llm_client = llm_client
        self.graph = None
        self.compiled = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
        else:
            print("Using fallback pipeline (LangGraph not available)")
    
    def _build_graph(self):
        """Build the LangGraph state machine."""
        # Create graph with AgentState
        self.graph = StateGraph(AgentState)
        
        # Wrap nodes with config and llm_client
        def wrap_node(node_fn):
            def wrapped(state):
                return node_fn(state, self.config, self.llm_client)
            return wrapped
        
        def wrap_node_no_llm(node_fn):
            def wrapped(state):
                return node_fn(state, self.config)
            return wrapped
        
        # Add nodes
        self.graph.add_node("disambiguate", wrap_node(disambiguate_node))
        self.graph.add_node("generate_draft_sql", wrap_node(generate_draft_sql_node))
        self.graph.add_node("retrieve_examples", wrap_node_no_llm(retrieve_examples_node))
        self.graph.add_node("generate_sql", wrap_node(generate_sql_node))
        self.graph.add_node("consistency_check", wrap_node(consistency_check_node))
        self.graph.add_node("evaluate_sql", wrap_node(evaluate_sql_node))
        self.graph.add_node("execute_sql", wrap_node_no_llm(execute_sql_node))
        self.graph.add_node("debug", wrap_node(debug_node))
        self.graph.add_node("finalize_success", wrap_node_no_llm(finalize_success_node))
        self.graph.add_node("finalize_hitl", wrap_node_no_llm(finalize_hitl_node))
        self.graph.add_node("finalize_failure", wrap_node_no_llm(finalize_failure_node))
        
        # Set entry point
        self.graph.set_entry_point("disambiguate")
        
        # Add edges
        
        # From disambiguate: check if clarification needed
        self.graph.add_conditional_edges(
            "disambiguate",
            lambda s: should_clarify(s, self.config),
            {
                "ask_user": "finalize_hitl",
                "retrieve_examples": "generate_draft_sql"
            }
        )
        
        # From generate_draft_sql: go to retrieve_examples
        self.graph.add_edge("generate_draft_sql", "retrieve_examples")
        
        # From retrieve_examples: go to generate
        self.graph.add_edge("retrieve_examples", "generate_sql")
        
        # From generate_sql: go to consistency check
        self.graph.add_edge("generate_sql", "consistency_check")
        
        # From consistency_check: check if we should evaluate or clarify
        self.graph.add_conditional_edges(
            "consistency_check",
            lambda s: should_evaluate_or_clarify(s, self.config),
            {
                "evaluate_sql": "evaluate_sql",
                "ask_user": "finalize_hitl"
            }
        )
        
        # From evaluate_sql: check if we should execute or clarify (MoE penalties)
        self.graph.add_conditional_edges(
            "evaluate_sql",
            lambda s: should_execute_or_clarify(s, self.config),
            {
                "execute_sql": "execute_sql",
                "ask_user": "finalize_hitl"
            }
        )
        
        # From execute_sql: route based on success
        self.graph.add_conditional_edges(
            "execute_sql",
            lambda s: route_after_execution(s, self.config),
            {
                "success": "finalize_success",
                "retry": "debug"
            }
        )
        
        # From debug: retry or fail
        self.graph.add_conditional_edges(
            "debug",
            lambda s: route_after_debug(s, self.config),
            {
                "generate_sql": "generate_sql",
                "fail": "finalize_failure"
            }
        )
        
        # Terminal nodes
        self.graph.add_edge("finalize_success", END)
        self.graph.add_edge("finalize_hitl", END)
        self.graph.add_edge("finalize_failure", END)
        
        # Compile the graph
        self.compiled = self.graph.compile()
    
    def run(self, user_query: str, user_feedback: Optional[str] = None, 
            session_id: str = "default") -> Dict[str, Any]:
        """
        Run the pipeline for a user query.
        
        Args:
            user_query: The natural language query
            user_feedback: Optional clarification from user (for HITL continuation)
            session_id: Session identifier for tracking
            
        Returns:
            Final state dictionary
        """
        initial_state: AgentState = {
            'user_query': user_query,
            'session_id': session_id,
            'user_feedback': user_feedback,
            'retry_count': 0,
            'max_retries': self.config.max_retries,
            'final_status': 'in_progress'
        }
        
        if LANGGRAPH_AVAILABLE and self.compiled:
            # Run the compiled graph
            final_state = self.compiled.invoke(initial_state)
            return dict(final_state)
        else:
            # Fallback: manual orchestration
            return self._run_fallback(initial_state)
    
    def _run_fallback(self, state: AgentState) -> Dict[str, Any]:
        """Fallback pipeline when LangGraph is not available."""
        # Disambiguate
        state.update(disambiguate_node(state, self.config, self.llm_client))
        
        # Check if clarification needed
        if state.get('needs_clarification') and not state.get('user_feedback'):
            state.update(finalize_hitl_node(state, self.config))
            return dict(state)
        
        # Generate Draft SQL
        state.update(generate_draft_sql_node(state, self.config, self.llm_client))
        
        # Retrieve examples
        state.update(retrieve_examples_node(state, self.config))
        
        # Generate SQL
        state.update(generate_sql_node(state, self.config, self.llm_client))
        
        # Consistency check
        state.update(consistency_check_node(state, self.config, self.llm_client))
        
        # Check consistency
        if not state.get('consistency_passed') and not state.get('user_feedback'):
            state.update(finalize_hitl_node(state, self.config))
            return dict(state)
            
        # Evaluate SQL (MoE)
        state.update(evaluate_sql_node(state, self.config, self.llm_client))
        
        # Check MoE Approval and Pipeline final confidence score
        if state.get('confidence_score', 1.0) < self.config.confidence_threshold and not state.get('user_feedback'):
            state.update(finalize_hitl_node(state, self.config))
            return dict(state)
        
        # Execution loop with retries
        while state.get('retry_count', 0) <= self.config.max_retries:
            state.update(execute_sql_node(state, self.config))
            
            if state.get('execution_success'):
                state.update(finalize_success_node(state, self.config))
                return dict(state)
            
            state.update(debug_node(state, self.config, self.llm_client))
            
            if state.get('retry_count', 0) > self.config.max_retries:
                break
            
            state.update(generate_sql_node(state, self.config, self.llm_client))
        
        state.update(finalize_failure_node(state, self.config))
        return dict(state)


# Factory function
def create_pipeline(config: Optional[PipelineConfig] = None, 
                    llm_client=None) -> TextToSQLGraph:
    """Create a new pipeline instance."""
    return TextToSQLGraph(config=config, llm_client=llm_client)
