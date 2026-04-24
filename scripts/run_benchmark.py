#!/usr/bin/env python3
"""
BIRD Benchmark Interactive Runner

This script runs BIRD-bench questions through the Text-to-SQL pipeline
and demonstrates the Human-in-the-Loop (HITL) clarification flow.

Usage:
    python scripts/run_benchmark.py                    # Run interactive mode
    python scripts/run_benchmark.py --auto             # Auto-respond to HITL
    python scripts/run_benchmark.py --db california_schools  # Specific database
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.pipeline import create_pipeline
from src.graph.state import PipelineConfig
from src.llm.client import create_llm_client
from src.sandbox.executor import set_execution_mode, get_execution_mode


# Sample BIRD-style questions for testing HITL
# These are designed to test ambiguity detection
BIRD_SAMPLE_QUESTIONS = [
    {
        "db_id": "sample_company",
        "question": "Show me the employees",
        "difficulty": "simple",
        "expected_hitl": False,
        "evidence": ""
    },
    {
        "db_id": "sample_company",
        "question": "What is the budget?",
        "difficulty": "ambiguous",
        "expected_hitl": True,
        "evidence": "Could refer to department budget or project budget"
    },
    {
        "db_id": "sample_company",
        "question": "List all active projects with their team members",
        "difficulty": "medium",
        "expected_hitl": False,
        "evidence": ""
    },
    {
        "db_id": "sample_company",
        "question": "Who earns the most?",
        "difficulty": "ambiguous",
        "expected_hitl": True,
        "evidence": "Salary could be interpreted differently"
    },
    {
        "db_id": "sample_company",
        "question": "How many people work in Engineering?",
        "difficulty": "simple",
        "expected_hitl": False,
        "evidence": ""
    },
    {
        "db_id": "sample_company",
        "question": "Show project costs",
        "difficulty": "ambiguous", 
        "expected_hitl": True,
        "evidence": "Could mean project budgets or hours allocated"
    },
    {
        "db_id": "sample_company",
        "question": "Find the manager",
        "difficulty": "ambiguous",
        "expected_hitl": True,
        "evidence": "Which manager? Department head? Project lead?"
    },
    {
        "db_id": "sample_company",
        "question": "List employees who were hired in 2020 and work on the Cloud Migration project",
        "difficulty": "hard",
        "expected_hitl": False,
        "evidence": ""
    },
    {
        "db_id": "sample_company",
        "question": "What's the total?",
        "difficulty": "very_ambiguous",
        "expected_hitl": True,
        "evidence": "Total of what? Employees? Budget? Hours?"
    },
    {
        "db_id": "sample_company",
        "question": "Show department spending analysis",
        "difficulty": "complex",
        "expected_hitl": True,
        "evidence": "Requires joining departments with projects, ambiguous what 'spending' means"
    }
]


class BenchmarkRunner:
    """Interactive benchmark runner with HITL support."""
    
    def __init__(self, auto_respond: bool = False):
        self.auto_respond = auto_respond
        self.results: List[Dict[str, Any]] = []
        
        # Initialize pipeline
        self.config = PipelineConfig(
            confidence_threshold=0.55,
            disambiguation_clear_confidence=0.85,
            max_retries=2,
            num_sql_variations=3,
            llm_provider=os.environ.get("LLM_PROVIDER", "auto")
        )
        
        self.llm_client = create_llm_client(provider="auto")
        self.pipeline = create_pipeline(config=self.config, llm_client=self.llm_client)
        
        print("\n" + "=" * 60)
        print("🧪 BIRD Benchmark Interactive Runner")
        print("=" * 60)
        print(f"LLM Provider: {self.config.llm_provider}")
        print(f"LLM Available: {self.llm_client.is_available() if self.llm_client else False}")
        print(f"Confidence Threshold: {self.config.confidence_threshold}")
        print(f"Auto-respond to HITL: {self.auto_respond}")
        print("=" * 60 + "\n")
    
    def run_question(self, question: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Run a single question through the pipeline."""
        db_id = question['db_id']
        query = question['question']
        difficulty = question.get('difficulty', 'unknown')
        evidence = question.get('evidence', '')
        expected_hitl = question.get('expected_hitl', False)
        
        print(f"\n{'─' * 60}")
        print(f"📝 Question {index + 1}: {query}")
        print(f"   Database: {db_id} | Difficulty: {difficulty}")
        if evidence:
            print(f"   Evidence: {evidence}")
        print(f"   Expected HITL: {'Yes' if expected_hitl else 'No'}")
        print(f"{'─' * 60}")
        
        # Switch to the appropriate database
        if db_id:
            success = set_execution_mode('benchmark', db_id, 'custom')
            if not success:
                # Try other datasets
                for dataset in ['bird', 'spider']:
                    if set_execution_mode('benchmark', db_id, dataset):
                        break
        
        session_id = f"benchmark_{index}"
        
        # Run the pipeline
        result = self.pipeline.run(query, session_id=session_id)
        
        # Check if HITL was triggered
        hitl_triggered = result.get('final_status') == 'paused_hitl'
        
        if hitl_triggered:
            print(f"\n🤔 HITL Triggered!")
            print(f"   Confidence: {result.get('confidence_score', 0):.0%}")
            print(f"   Reasons: {', '.join(result.get('ambiguity_reasons', []))}")
            
            if self.auto_respond:
                # Auto-generate a clarification based on evidence
                feedback = self._generate_auto_feedback(question, result)
                print(f"   Auto-feedback: {feedback}")
                result = self.pipeline.run(query, user_feedback=feedback, session_id=session_id)
            else:
                # Interactive HITL
                result = self._handle_hitl_interactive(query, result, session_id)
        
        # Display result
        final_status = result.get('final_status', 'unknown')
        sql = result.get('selected_sql', '')
        
        if final_status == 'success':
            print(f"\n✅ Success!")
            print(f"   SQL: {sql[:80]}..." if len(sql) > 80 else f"   SQL: {sql}")
            
            exec_result = result.get('execution_result', {})
            if exec_result.get('success'):
                row_count = exec_result.get('row_count', 0)
                print(f"   Rows returned: {row_count}")
        else:
            print(f"\n❌ Status: {final_status}")
            if result.get('error_trace'):
                print(f"   Error: {result.get('error_trace')}")
        
        # Check if HITL prediction was correct
        hitl_correct = (hitl_triggered == expected_hitl)
        
        outcome = {
            'question': query,
            'db_id': db_id,
            'difficulty': difficulty,
            'expected_hitl': expected_hitl,
            'hitl_triggered': hitl_triggered,
            'hitl_correct': hitl_correct,
            'final_status': final_status,
            'sql': sql,
            'confidence': result.get('confidence_score', 0)
        }
        
        self.results.append(outcome)
        return outcome
    
    def _handle_hitl_interactive(self, query: str, result: Dict, session_id: str) -> Dict:
        """Handle HITL with interactive user input."""
        print("\n" + "─" * 40)
        print("Please provide clarification to help resolve the ambiguity.")
        print("Type 'skip' to skip this question, or 'quit' to exit.")
        print("─" * 40)
        
        try:
            feedback = input("Your clarification: ").strip()
        except (EOFError, KeyboardInterrupt):
            feedback = "skip"
        
        if feedback.lower() == 'quit':
            raise KeyboardInterrupt()
        
        if feedback.lower() == 'skip' or not feedback:
            return result
        
        # Continue pipeline with feedback
        return self.pipeline.run(query, user_feedback=feedback, session_id=session_id)
    
    def _generate_auto_feedback(self, question: Dict, result: Dict) -> str:
        """Generate automatic feedback for HITL based on question evidence."""
        evidence = question.get('evidence', '')
        query = question['question']
        
        # Simple heuristics for auto-feedback
        query_lower = query.lower()
        
        if 'budget' in query_lower:
            return "I mean the department budget"
        elif 'total' in query_lower:
            return "I want the total number of employees"
        elif 'cost' in query_lower:
            return "Show the project budgets"
        elif 'manager' in query_lower:
            return "Show employees who have manager_id set (they are managers of someone)"
        elif 'earns' in query_lower or 'salary' in query_lower:
            return "Find the employee with the highest salary"
        elif 'spending' in query_lower:
            return "Show department budgets and their associated project costs"
        else:
            return "Please provide all relevant information"
    
    def run_all(self, questions: List[Dict[str, Any]] = None):
        """Run all benchmark questions."""
        if questions is None:
            questions = BIRD_SAMPLE_QUESTIONS
        
        print(f"\n🚀 Running {len(questions)} questions...\n")
        
        try:
            for i, question in enumerate(questions):
                self.run_question(question, i)
        except KeyboardInterrupt:
            print("\n\n⏹️  Benchmark interrupted by user.")
        
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark results summary."""
        if not self.results:
            print("\nNo results to summarize.")
            return
        
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        total = len(self.results)
        successes = sum(1 for r in self.results if r['final_status'] == 'success')
        hitl_correct = sum(1 for r in self.results if r['hitl_correct'])
        hitl_triggered = sum(1 for r in self.results if r['hitl_triggered'])
        
        print(f"\nTotal Questions: {total}")
        print(f"Successful: {successes}/{total} ({successes/total*100:.1f}%)")
        print(f"HITL Triggered: {hitl_triggered}/{total}")
        print(f"HITL Prediction Correct: {hitl_correct}/{total} ({hitl_correct/total*100:.1f}%)")
        
        # Breakdown by difficulty
        print("\n📈 By Difficulty:")
        difficulties = {}
        for r in self.results:
            d = r['difficulty']
            if d not in difficulties:
                difficulties[d] = {'total': 0, 'success': 0, 'hitl': 0}
            difficulties[d]['total'] += 1
            if r['final_status'] == 'success':
                difficulties[d]['success'] += 1
            if r['hitl_triggered']:
                difficulties[d]['hitl'] += 1
        
        for d, stats in difficulties.items():
            print(f"  {d}: {stats['success']}/{stats['total']} success, {stats['hitl']} HITL")
        
        # Average confidence
        avg_confidence = sum(r['confidence'] for r in self.results) / total
        print(f"\n🎯 Average Confidence: {avg_confidence:.1%}")
        
        print("\n" + "=" * 60)
        

def main():
    parser = argparse.ArgumentParser(description='BIRD Benchmark Interactive Runner')
    parser.add_argument('--auto', action='store_true', help='Auto-respond to HITL prompts')
    parser.add_argument('--db', type=str, help='Run only questions for specific database')
    parser.add_argument('--count', type=int, default=10, help='Number of questions to run')
    args = parser.parse_args()
    
    # Filter questions if db specified
    questions = BIRD_SAMPLE_QUESTIONS[:args.count]
    if args.db:
        questions = [q for q in questions if q['db_id'] == args.db]
    
    runner = BenchmarkRunner(auto_respond=args.auto)
    runner.run_all(questions)


if __name__ == '__main__':
    main()
