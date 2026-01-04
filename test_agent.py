import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import TextToSqlPipeline

def run_tests():
    pipeline = TextToSqlPipeline()

    print("\n[TEST 1] Happy Path: 'Count the users'")
    result = pipeline.run("Count the users")
    print(f"Result: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"Data: {result['data']}")
    
    print("\n" + "="*30 + "\n")

    print("\n[TEST 2] HITL Scenario: 'Show me the top ?'")
    # Simulate a vague query
    result = pipeline.run("Show me the top ?")
    print(f"Result: {result['status']}")
    
    if result['status'] == 'PAUSED_HITL':
        print(">> Triggered HITL as expected.")
        print(">> Simulating user feedback: 'I meant top 5 users by name'")
        result_continuation = pipeline.run("Show me the top ?", user_feedback="I meant top 5 users by name")
        print(f"Result after Feedback: {result_continuation['status']}")
    
    print("\n" + "="*30 + "\n")

    print("\n[TEST 3] Self-Correction Scenario: 'Show me the error'")
    # This triggers the generator to produce bad SQL first, then hopefully retry (though my simple generator might just fail or loop)
    # My generator logic for 'error' is hardcoded to fail. Let's see if the loop works.
    result = pipeline.run("Show me the error")
    print(f"Final Result: {result['status']}")
    if result['status'] == 'FAILED':
        print(f"Last Error: {result['last_error']}")
        print("(This is expected if the generator keeps making bad SQL, but we should see retries in the logs)")

if __name__ == "__main__":
    run_tests()
