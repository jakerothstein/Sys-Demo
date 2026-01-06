"""
Graph Node Functions for LangGraph Pipeline.
Each node is a function that takes AgentState and returns a partial state update.
LLM-First Design: All logic uses LLM prompts rather than regex/keyword matching.
"""
import json
import os
import re
from typing import Dict, Any, List

from .state import AgentState, PipelineConfig
from src.data.vector_store import get_schema_store


def load_schema_catalog() -> Dict[str, Any]:
    """Load the schema catalog - prefers current database, falls back to JSON."""
    try:
        # Try to get schema from current database executor
        from src.sandbox.executor import get_current_schema, get_execution_mode
        mode = get_execution_mode()
        if mode.get('database'):
            schema = get_current_schema()
            if schema and schema.get('tables'):
                return schema
    except Exception:
        pass
    
    # Fallback to static JSON schema
    schema_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'schema_catalog.json')
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            return json.load(f)
    return {}


def format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    """
    Format schema catalog into a comprehensive prompt-friendly string.
    Includes table descriptions, column types, primary keys, foreign keys,
    and relationship summaries for LLM understanding.
    """
    if not schema:
        return "No schema available."
    
    lines = [f"Database: {schema.get('database_name', 'unknown')}\n"]
    
    # Track foreign key relationships for summary
    relationships = []
    
    for table_name, table_info in schema.get('tables', {}).items():
        lines.append(f"\n### Table: {table_name}")
        lines.append(f"Description: {table_info.get('description', '')}")
        lines.append("Columns:")
        
        for col_name, col_info in table_info.get('columns', {}).items():
            col_type = col_info.get('type', 'TEXT')
            col_desc = col_info.get('description', '')
            
            # Build column annotation
            annotations = []
            if col_info.get('is_primary_key'):
                annotations.append("PRIMARY KEY")
            if col_info.get('foreign_key'):
                fk_target = col_info['foreign_key']
                annotations.append(f"FOREIGN KEY -> {fk_target}")
                relationships.append(f"{table_name}.{col_name} references {fk_target}")
            
            annotation_str = f" [{', '.join(annotations)}]" if annotations else ""
            samples = ""
            if col_info.get('sample_values'):
                sample_list = ', '.join(f"'{v}'" for v in col_info['sample_values'][:3])
                samples = f" (examples: {sample_list})"
            
            lines.append(f"  - {col_name}: {col_type}{annotation_str} - {col_desc}{samples}")
    
    # Add relationship summary section
    if relationships:
        lines.append("\n### Table Relationships (Foreign Keys):")
        for rel in relationships:
            lines.append(f"  - {rel}")
    
    # Add semantic notes if present
    semantic_notes = schema.get('semantic_notes', {})
    if semantic_notes:
        lines.append("\n### Semantic Notes (Ambiguous Terms):")
        for term, info in semantic_notes.items():
            if info.get('ambiguous'):
                interpretations = info.get('possible_interpretations', [])
                interp_str = ', '.join(i.get('term', '') for i in interpretations)
                lines.append(f"  - '{term}' can mean: {interp_str}")
    
    return "\n".join(lines)


def filter_schema_to_tables(schema: Dict[str, Any], table_names: List[str]) -> Dict[str, Any]:
    """
    Filter a schema dictionary to include only the specified tables.
    Used by schema linking to provide focused context to the LLM.
    
    Args:
        schema: Full schema catalog dictionary
        table_names: List of table names to keep
        
    Returns:
        Filtered schema with only the specified tables
    """
    if not schema or not table_names:
        return schema
    
    table_set = set(table_names)
    filtered_tables = {
        name: info for name, info in schema.get('tables', {}).items()
        if name in table_set
    }
    
    return {
        'database_name': schema.get('database_name', 'unknown'),
        'tables': filtered_tables,
        'semantic_notes': schema.get('semantic_notes', {})
    }


def _parse_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from an LLM response.
    Handles responses with surrounding text or markdown code blocks.
    """
    # Try to find JSON in code blocks first
    code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', response)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Return empty dict if no valid JSON found
    return {}


def disambiguate_node(state: AgentState, config: PipelineConfig, llm_client=None) -> Dict[str, Any]:
    """
    Analyze the query for ambiguity and semantic clarity using LLM.
    Uses Schema Linking to retrieve only relevant tables instead of full schema.
    Returns confidence score, detected entities, and clarification needs.
    """
    user_query = state.get('user_query', '')
    user_feedback = state.get('user_feedback', '')
    full_schema = load_schema_catalog()
    
    # Schema Linking: Retrieve only relevant tables for this query
    try:
        schema_store = get_schema_store()
        # Ensure schema is indexed
        if schema_store.get_stats().get('indexed_tables', 0) == 0:
            schema_store.index_schema(full_schema)
        
        relevant_tables = schema_store.retrieve_relevant_tables(user_query, n=5)
        if relevant_tables:
            schema = filter_schema_to_tables(full_schema, relevant_tables)
            print(f"Schema Linking: Retrieved {len(relevant_tables)} relevant tables: {relevant_tables}")
        else:
            schema = full_schema  # Fallback to full schema if no tables retrieved
    except Exception as e:
        print(f"Schema linking fallback: {e}")
        schema = full_schema  # Fallback to full schema on error
    
    schema_context = format_schema_for_prompt(schema)
    
    # Build user feedback section if clarification was provided
    feedback_section = ""
    if user_feedback:
        feedback_section = f"""
USER CLARIFICATION: "{user_feedback}"
Note: The user has provided clarification for their original query. This should significantly increase your confidence since the ambiguity has been resolved by the user.
"""
    
    # Build disambiguation prompt for LLM
    prompt = f"""You are an expert SQL analyst. Analyze the following natural language query for a database.

DATABASE SCHEMA:
{schema_context}

USER QUERY: "{user_query}"
{feedback_section}
Analyze this query and return a JSON object with the following structure:
{{
    "confidence": <float 0.0-1.0 indicating how clear and unambiguous the query is>,
    "is_ambiguous": <boolean - true if the query is unclear or could have multiple interpretations>,
    "detected_tables": <list of table names from the schema that are relevant to this query>,
    "detected_intent": <string: one of "select", "count", "sum", "average", "ranking", "filter", "join", "aggregate">,
    "ambiguity_reasons": <list of strings explaining why the query is ambiguous, empty if clear>,
    "suggested_clarification": <string with a clarifying question to ask the user, or null if not needed>
}}

Consider:
- Are the table references clear?
- Is the aggregation or intent unambiguous?
- Are there any terms that could have multiple meanings (e.g., "sales", "top customers")?
- Does the user need to specify filters, grouping, or ordering?
- If the user has provided clarification, factor that into your confidence score (it should be higher).

Return ONLY the JSON object, no other text."""

    # Default response for fallback
    default_response = {
        'confidence_score': 0.5,
        'is_ambiguous': True,
        'ambiguity_reasons': ["Unable to analyze query"],
        'detected_tables': [],
        'detected_intent': "select",
        'schema_context': schema_context,
        'needs_clarification': True,
        'clarification_message': "Could you please clarify your query?"
    }
    
    if not llm_client:
        # Without LLM, return conservative defaults
        return default_response
    
    try:
        response = llm_client.generate(prompt)
        parsed = _parse_json_from_response(response)
        
        if not parsed:
            return default_response
        
        confidence = float(parsed.get('confidence', 0.5))
        
        # Override is_ambiguous if confidence is very high (>= 0.9)
        # This prevents unnecessary HITL triggers when the LLM is confident
        if confidence >= 0.9:
            is_ambiguous = False
            ambiguity_reasons = []
        else:
            is_ambiguous = parsed.get('is_ambiguous', confidence < config.confidence_threshold)
            ambiguity_reasons = parsed.get('ambiguity_reasons', [])
        
        # Determine if clarification is needed
        needs_clarification = is_ambiguous and not state.get('user_feedback')
        
        clarification_message = ""
        if needs_clarification:
            suggested = parsed.get('suggested_clarification')
            if suggested:
                clarification_message = suggested
            elif ambiguity_reasons:
                clarification_message = f"I need clarification:\n" + "\n".join(f"- {r}" for r in ambiguity_reasons)
        
        return {
            'confidence_score': round(confidence, 2),
            'is_ambiguous': is_ambiguous,
            'ambiguity_reasons': ambiguity_reasons,
            'detected_tables': parsed.get('detected_tables', []),
            'detected_intent': parsed.get('detected_intent', 'select'),
            'schema_context': schema_context,
            'needs_clarification': needs_clarification,
            'clarification_message': clarification_message
        }
        
    except Exception as e:
        print(f"Disambiguation error: {e}")
        return default_response


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
    Generate SQL using LLM based on the query, schema, and context.
    Generates multiple variations for consistency checking.
    """
    user_query = state.get('user_query', '')
    if state.get('user_feedback'):
        user_query = f"{user_query}\n\nUser clarification: {state['user_feedback']}"
    
    schema_context = state.get('schema_context', '')
    few_shot_examples = state.get('few_shot_examples', [])
    error_context = state.get('debug_analysis', '')
    num_variations = config.num_sql_variations
    
    if not llm_client:
        # Without LLM, return a placeholder that will likely fail gracefully
        return {
            'generated_sqls': ["SELECT 1;"],
            'selected_sql': "SELECT 1;"
        }
    
    # Build SQL generation prompt
    examples_text = ""
    if few_shot_examples:
        examples_text = "\n\nSIMILAR EXAMPLES:\n"
        for ex in few_shot_examples:
            examples_text += f"Question: {ex.get('question', '')}\nSQL: {ex.get('sql', '')}\n\n"
    
    error_section = ""
    if error_context:
        error_section = f"""
PREVIOUS ERROR (Self-Correction Required):
{error_context}

Learn from this error and generate corrected SQL that avoids this issue.
"""

    prompt = f"""You are an expert SQL developer. Generate {num_variations} different valid SQL queries for the following question.

DATABASE SCHEMA:
{schema_context}
{examples_text}
{error_section}
QUESTION: {user_query}

Return a JSON object with this structure:
{{
    "sql_queries": [
        "<first SQL query>",
        "<second SQL query>",
        "<third SQL query>"
    ],
    "reasoning": "<brief explanation of your approach>"
}}

Guidelines:
- Generate exactly {num_variations} different but semantically equivalent SQL queries
- All queries should produce the same result but may use different syntax or approaches
- Use proper table and column names from the schema
- Handle JOINs appropriately using foreign key relationships
- Use appropriate aggregations (COUNT, SUM, AVG, etc.) based on the query intent
- Include appropriate LIMIT clauses for open-ended queries

Return ONLY the JSON object, no other text."""

    try:
        response = llm_client.generate(prompt)
        parsed = _parse_json_from_response(response)
        
        if parsed and 'sql_queries' in parsed:
            sqls = parsed['sql_queries']
            if isinstance(sqls, list) and sqls:
                return {
                    'generated_sqls': sqls,
                    'selected_sql': sqls[0]
                }
        
        # Try to parse SQL_N: format as fallback
        sqls = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('SQL_') and ':' in line:
                sql = line.split(':', 1)[1].strip()
                if sql:
                    sqls.append(sql)
        
        if sqls:
            return {
                'generated_sqls': sqls,
                'selected_sql': sqls[0]
            }
        
        # Last resort: return placeholder
        return {
            'generated_sqls': ["SELECT 1;"],
            'selected_sql': "SELECT 1;"
        }
        
    except Exception as e:
        print(f"SQL generation error: {e}")
        return {
            'generated_sqls': ["SELECT 1;"],
            'selected_sql': "SELECT 1;"
        }


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
    Analyze SQL execution error and prepare context for self-healing retry.
    Feeds the error message and failed SQL back into the generation context
    so the LLM can learn from the mistake and generate corrected SQL.
    """
    error = state.get('error_trace', '')
    failed_sql = state.get('selected_sql', '')
    retry_count = state.get('retry_count', 0)
    schema_context = state.get('schema_context', '')
    
    # Build comprehensive debug analysis for LLM self-correction
    analysis_parts = [
        f"FAILED SQL:\n{failed_sql}",
        f"\nERROR MESSAGE:\n{error}",
        f"\nRETRY ATTEMPT: {retry_count + 1} of {config.max_retries}"
    ]
    
    # Add specific guidance based on error type
    if "no such table" in error.lower():
        analysis_parts.append("\nDIAGNOSIS: Table name is incorrect. Review the schema carefully for valid table names.")
    elif "no such column" in error.lower():
        analysis_parts.append("\nDIAGNOSIS: Column name is incorrect. Check the schema for valid column names in the relevant table.")
    elif "syntax error" in error.lower():
        analysis_parts.append("\nDIAGNOSIS: SQL syntax is invalid. Simplify the query and check for missing keywords, parentheses, or quotes.")
    elif "ambiguous column" in error.lower():
        analysis_parts.append("\nDIAGNOSIS: Column reference is ambiguous. Use fully qualified table.column notation for all columns in JOINs.")
    elif "near" in error.lower():
        analysis_parts.append("\nDIAGNOSIS: Syntax error near a specific token. Check for typos and proper SQL formatting.")
    else:
        analysis_parts.append("\nDIAGNOSIS: Unknown error. Try a simpler query approach or alternative syntax.")
    
    # Add schema reminder for context
    if schema_context:
        analysis_parts.append(f"\nAVAILABLE SCHEMA (for reference):\n{schema_context[:1000]}...")
    
    analysis = "\n".join(analysis_parts)
    
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
