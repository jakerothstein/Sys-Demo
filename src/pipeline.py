from src.agents.disambiguator import DisambiguatorAgent
from src.agents.sql_generator import SqlGeneratorAgent
from src.agents.tester import TestingAgent
from dataclasses import asdict

class TextToSqlPipeline:
    def __init__(self):
        self.disambiguator = DisambiguatorAgent()
        self.generator = SqlGeneratorAgent()
        self.tester = TestingAgent()
        self.max_retries = 3

    def run(self, user_query: str, user_feedback: str = None) -> dict:
        """
        Runs the pipeline.
        
        Args:
            user_query: The natural language query.
            user_feedback: Optional feedback provided by HITL if the previous run paused.
        
        Returns:
            A dictionary containing the status and results.
        """
        # 1. Disambiguation / Planning
        plan = self.disambiguator.analyze(user_query)
        
        # HITL Check
        if plan.needs_hitl and not user_feedback:
            return {
                "status": "PAUSED_HITL",
                "message": "Confidence low. Please clarify.",
                "plan": asdict(plan)
            }
        
        # Apply feedback if provided
        if user_feedback:
            plan.disambiguated_query = f"{user_query} ({user_feedback})"
            plan.needs_hitl = False

        # 2. Generation & 3. Verification Loop (Self-Correction)
        current_plan_context = asdict(plan)
        attempts = 0
        
        while attempts < self.max_retries:
            print(f"--- Attempt {attempts + 1} ---")
            sql = self.generator.generate(current_plan_context)
            print(f"Generated SQL: {sql}")
            
            execution_result = self.tester.execute(sql)
            
            if execution_result['success']:
                return {
                    "status": "SUCCESS",
                    "sql": sql,
                    "data": execution_result['data'],
                    "plan": current_plan_context
                }
            
            # Correction Logic
            print(f"Execution Failed: {execution_result['error']}")
            # Update context with error to guide regeneration
            current_plan_context['previous_error'] = execution_result['error']
            current_plan_context['disambiguated_query'] += f" (fix error: {execution_result['error']})"
            attempts += 1

        return {
            "status": "FAILED",
            "error": "Max retries exceeded",
            "last_error": execution_result.get('error')
        }
