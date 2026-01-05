"""
Test script for the Text-to-SQL Graph Pipeline.
Uses the new LangGraph-based implementation from src.graph.
"""
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph.pipeline import create_pipeline, TextToSQLGraph
from src.graph.state import PipelineConfig
from src.llm.client import create_llm_client


def run_tests():
    """Run pipeline tests with the new graph-based architecture."""
    
    # Create pipeline with mock LLM for testing
    config = PipelineConfig(
        confidence_threshold=0.7,
        max_retries=3,
        num_sql_variations=3,
        llm_provider="mock"
    )
    
    llm_client = create_llm_client(provider="mock")
    pipeline = create_pipeline(config=config, llm_client=llm_client)
    
    print("\n[TEST 1] Happy Path: 'Count the users'")
    result = pipeline.run("Count the users")
    print(f"Result: {result.get('final_status', 'unknown').upper()}")
    if result.get('final_status') == 'success':
        exec_result = result.get('execution_result', {})
        print(f"Data: {exec_result.get('data', [])}")
    
    print("\n" + "="*30 + "\n")

    print("\n[TEST 2] HITL Scenario: 'Show me the top ?'")
    # Simulate a vague query
    result = pipeline.run("Show me the top ?")
    status = result.get('final_status', 'unknown')
    print(f"Result: {status.upper()}")
    
    if status == 'paused_hitl':
        print(">> Triggered HITL as expected.")
        print(">> Simulating user feedback: 'I meant top 5 users by name'")
        result_continuation = pipeline.run(
            "Show me the top ?", 
            user_feedback="I meant top 5 users by name"
        )
        print(f"Result after Feedback: {result_continuation.get('final_status', 'unknown').upper()}")
    
    print("\n" + "="*30 + "\n")

    print("\n[TEST 3] Self-Correction Scenario: 'Show me the error'")
    result = pipeline.run("Show me the error")
    status = result.get('final_status', 'unknown')
    print(f"Final Result: {status.upper()}")
    if status == 'failed':
        print(f"Last Error: {result.get('error_trace', 'Unknown')}")
        print("(This is expected if the generator keeps making bad SQL, but we should see retries in the logs)")
    elif status == 'paused_hitl':
        print("(Query was ambiguous, HITL triggered)")


if __name__ == "__main__":
    run_tests()
