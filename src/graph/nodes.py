"""
Graph Node Functions for LangGraph Pipeline.
Each node is a function that takes AgentState and returns a partial state update.
"""
import json
import os
from typing import Dict, Any, List

from .state import AgentState, PipelineConfig


def load_schema_catalog() -> Dict[str, Any]:
    """Load the schema catalog from JSON."""
    schema_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'schema_catalog.json')
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            return json.load(f)
    return {}


def format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    """Format schema catalog into a prompt-friendly string."""
    if not schema:
        return "No schema available."
    
    lines = [f"Database: {schema.get('database_name', 'unknown')}\n"]
    
    for table_name, table_info in schema.get('tables', {}).items():
        lines.append(f"\nTable: {table_name}")
        lines.append(f"Description: {table_info.get('description', '')}")
        lines.append("Columns:")
        
        for col_name, col_info in table_info.get('columns', {}).items():
            col_type = col_info.get('type', 'TEXT')
            col_desc = col_info.get('description', '')
            pk = " [PK]" if col_info.get('is_primary_key') else ""
            fk = f" [FK -> {col_info['foreign_key']}]" if col_info.get('foreign_key') else ""
            samples = f" (e.g., {', '.join(col_info['sample_values'])})" if col_info.get('sample_values') else ""
            
            lines.append(f"  - {col_name} ({col_type}){pk}{fk}: {col_desc}{samples}")
    
    return "\n".join(lines)


def disambiguate_node(state: AgentState, config: PipelineConfig, llm_client=None) -> Dict[str, Any]:
    """
    Analyze the query for ambiguity and semantic clarity.
    Returns confidence score and detected entities.
    """
    user_query = state.get('user_query', '')
    schema = load_schema_catalog()
    schema_context = format_schema_for_prompt(schema)
    
    # Detect tables mentioned
    query_lower = user_query.lower()
    detected_tables = []
    for table in schema.get('tables', {}).keys():
        if table in query_lower or table.rstrip('s') in query_lower:
            detected_tables.append(table)
    
    # Check for semantic ambiguity
    ambiguity_reasons = []
    semantic_notes = schema.get('semantic_notes', {})
    
    for term, info in semantic_notes.items():
        if term in query_lower and info.get('ambiguous'):
            interpretations = [i['term'] for i in info.get('possible_interpretations', [])]
            ambiguity_reasons.append(f"'{term}' is ambiguous: could mean {', '.join(interpretations)}")
    
    # Detect intent
    intent = "select"
    intent_keywords = {
        "count": ["count", "how many", "number of"],
        "sum": ["total", "sum", "revenue", "sales"],
        "average": ["average", "avg", "mean"],
        "ranking": ["top", "best", "highest", "lowest"],
        "filter": ["where", "with", "from", "in"]
    }
    
    for intent_type, keywords in intent_keywords.items():
        if any(kw in query_lower for kw in keywords):
            intent = intent_type
            break
    
    # Calculate confidence
    confidence = 0.5
    
    if detected_tables:
        confidence += 0.2
    else:
        ambiguity_reasons.append("No specific tables detected in query")
    
    if intent != "select":
        confidence += 0.15
    
    if ambiguity_reasons:
        confidence -= 0.2 * len(ambiguity_reasons)
    
    # Check for vague markers
    vague_markers = ["maybe", "probably", "unsure", "?", "something", "stuff"]
    if any(marker in query_lower for marker in vague_markers):
        confidence -= 0.3
        ambiguity_reasons.append("Query contains uncertainty markers")
    
    confidence = max(0.1, min(1.0, confidence))
    is_ambiguous = confidence < config.confidence_threshold
    
    # Determine if clarification needed
    needs_clarification = is_ambiguous and not state.get('user_feedback')
    
    clarification_message = ""
    if needs_clarification and ambiguity_reasons:
        clarification_message = f"I need clarification:\n" + "\n".join(f"- {r}" for r in ambiguity_reasons)
    
    return {
        'confidence_score': round(confidence, 2),
        'is_ambiguous': is_ambiguous,
        'ambiguity_reasons': ambiguity_reasons,
        'detected_tables': detected_tables,
        'detected_intent': intent,
        'schema_context': schema_context,
        'needs_clarification': needs_clarification,
        'clarification_message': clarification_message
    }


def retrieve_examples_node(state: AgentState, config: PipelineConfig) -> Dict[str, Any]:
    """
    Retrieve relevant few-shot examples from the vector store.
    """
    try:
        from src.data.vector_store import get_vector_store
        vector_store = get_vector_store()
        examples = vector_store.retrieve(state.get('user_query', ''), n_results=3)
    except Exception as e:
        print(f"Vector store error: {e}")
        examples = []
    
    return {'few_shot_examples': examples}


def generate_sql_node(state: AgentState, config: PipelineConfig, llm_client=None) -> Dict[str, Any]:
    """
    Generate SQL variations based on the query and context.
    Generates multiple variations for consistency checking.
    """
    user_query = state.get('user_query', '')
    if state.get('user_feedback'):
        user_query = f"{user_query} (User clarification: {state['user_feedback']})"
    
    schema_context = state.get('schema_context', '')
    few_shot_examples = state.get('few_shot_examples', [])
    detected_tables = state.get('detected_tables', [])
    error_context = state.get('debug_analysis', '')
    
    # Use LLM if available, otherwise use mock
    if llm_client:
        generated_sqls = _generate_with_llm(
            llm_client, user_query, schema_context, few_shot_examples, 
            error_context, config.num_sql_variations
        )
    else:
        generated_sqls = _generate_mock(user_query, detected_tables, error_context)
    
    return {
        'generated_sqls': generated_sqls,
        'selected_sql': generated_sqls[0] if generated_sqls else ""
    }


def _generate_mock(query: str, tables: List[str], error_context: str) -> List[str]:
    """Mock SQL generation for testing without LLM."""
    query_lower = query.lower()
    
    # If we're retrying after an error, try to fix it
    if error_context:
        if "no such table" in error_context.lower():
            return [f"SELECT * FROM customers LIMIT 10;"]
    
    # Simple pattern matching
    if "count" in query_lower and "customer" in query_lower:
        return [
            "SELECT COUNT(*) as customer_count FROM customers;",
            "SELECT COUNT(id) as total FROM customers;",
            "SELECT COUNT(*) FROM customers;"
        ]
    elif "revenue" in query_lower or "sales" in query_lower:
        return [
            "SELECT SUM(total_amount) as revenue FROM orders WHERE status = 'delivered';",
            "SELECT SUM(total_amount) as total_sales FROM orders WHERE status IN ('delivered', 'shipped');",
            "SELECT SUM(total_amount) FROM orders;"
        ]
    elif "top" in query_lower and "customer" in query_lower:
        return [
            "SELECT c.name, SUM(o.total_amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY total DESC LIMIT 5;",
            "SELECT c.name, COUNT(o.id) as orders FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY orders DESC LIMIT 5;",
            "SELECT c.name, SUM(o.total_amount) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name ORDER BY SUM(o.total_amount) DESC LIMIT 5;"
        ]
    elif tables:
        return [f"SELECT * FROM {tables[0]} LIMIT 10;"]
    else:
        return ["SELECT * FROM customers LIMIT 10;"]


def _generate_with_llm(llm_client, query: str, schema: str, examples: List[dict], 
                       error_context: str, num_variations: int) -> List[str]:
    """Generate SQL using LLM."""
    # Format few-shot examples
    examples_text = ""
    for ex in examples:
        examples_text += f"\nQuestion: {ex['question']}\nSQL: {ex['sql']}\n"
    
    prompt = f"""You are an expert SQL developer. Generate {num_variations} different valid SQL queries for the following question.

Schema:
{schema}

Similar Examples:
{examples_text}

{"Previous Error to Fix: " + error_context if error_context else ""}

Question: {query}

Return exactly {num_variations} SQL queries, each on a new line, prefixed with "SQL_1:", "SQL_2:", etc.
Only return the SQL, no explanations."""

    response = llm_client.generate(prompt)
    
    # Parse response
    sqls = []
    for line in response.split('\n'):
        if line.strip().startswith('SQL_'):
            sql = line.split(':', 1)[1].strip() if ':' in line else line.strip()
            if sql:
                sqls.append(sql)
    
    return sqls or ["SELECT * FROM customers LIMIT 10;"]


def consistency_check_node(state: AgentState, config: PipelineConfig, llm_client=None) -> Dict[str, Any]:
    """
    Check if the generated SQL variations are semantically consistent.
    If they differ significantly, we may need clarification.
    """
    sqls = state.get('generated_sqls', [])
    
    if len(sqls) < 2:
        return {
            'consistency_passed': True,
            'consistency_analysis': "Only one SQL generated, skipping consistency check."
        }
    
    # Simple heuristic: check if JOINs and WHERE clauses are similar
    def extract_structure(sql: str) -> dict:
        sql_upper = sql.upper()
        return {
            'has_join': 'JOIN' in sql_upper,
            'has_where': 'WHERE' in sql_upper,
            'has_group': 'GROUP BY' in sql_upper,
            'has_order': 'ORDER BY' in sql_upper,
            'aggregation': any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX'])
        }
    
    structures = [extract_structure(sql) for sql in sqls]
    
    # Check if all structures match
    inconsistencies = []
    keys = ['has_join', 'has_where', 'has_group', 'aggregation']
    
    for key in keys:
        values = [s[key] for s in structures]
        if len(set(values)) > 1:
            inconsistencies.append(f"Disagreement on {key.replace('has_', '')}")
    
    consistency_passed = len(inconsistencies) == 0
    
    if consistency_passed:
        analysis = "All variations have consistent structure (JOINs, WHERE, GROUP BY)."
    else:
        analysis = f"Inconsistencies found: {', '.join(inconsistencies)}. This may indicate semantic ambiguity."
    
    return {
        'consistency_passed': consistency_passed,
        'consistency_analysis': analysis
    }


def execute_sql_node(state: AgentState, config: PipelineConfig) -> Dict[str, Any]:
    """
    Execute the selected SQL in the sandbox.
    """
    sql = state.get('selected_sql', '')
    
    if not sql:
        return {
            'execution_success': False,
            'error_trace': "No SQL to execute",
            'execution_result': None
        }
    
    try:
        from src.sandbox.executor import execute_sql
        result = execute_sql(sql)
        
        if result['success']:
            return {
                'execution_success': True,
                'execution_result': result,
                'error_trace': None
            }
        else:
            return {
                'execution_success': False,
                'error_trace': result.get('error', 'Unknown error'),
                'execution_result': None
            }
    except Exception as e:
        return {
            'execution_success': False,
            'error_trace': str(e),
            'execution_result': None
        }


def debug_node(state: AgentState, config: PipelineConfig, llm_client=None) -> Dict[str, Any]:
    """
    Analyze the error and prepare context for retry.
    """
    error = state.get('error_trace', '')
    sql = state.get('selected_sql', '')
    retry_count = state.get('retry_count', 0)
    
    # Simple error analysis
    analysis = f"Error: {error}\n"
    
    if "no such table" in error.lower():
        analysis += "Issue: Table name is incorrect. Check schema for valid table names."
    elif "syntax error" in error.lower():
        analysis += "Issue: SQL syntax is invalid. Simplify the query."
    elif "ambiguous column" in error.lower():
        analysis += "Issue: Column reference is ambiguous. Use table.column notation."
    else:
        analysis += "Issue: Unknown error. Attempting alternative approach."
    
    return {
        'debug_analysis': analysis,
        'retry_count': retry_count + 1
    }


def finalize_success_node(state: AgentState, config: PipelineConfig) -> Dict[str, Any]:
    """Finalize the state for successful execution."""
    return {
        'final_status': 'success',
        'final_message': 'Query executed successfully.'
    }


def finalize_hitl_node(state: AgentState, config: PipelineConfig) -> Dict[str, Any]:
    """Finalize the state when HITL is needed."""
    return {
        'final_status': 'paused_hitl',
        'final_message': state.get('clarification_message', 'Clarification needed.')
    }


def finalize_failure_node(state: AgentState, config: PipelineConfig) -> Dict[str, Any]:
    """Finalize the state for failed execution."""
    return {
        'final_status': 'failed',
        'final_message': f"Failed after {state.get('retry_count', 0)} retries. Last error: {state.get('error_trace', 'Unknown')}"
    }
