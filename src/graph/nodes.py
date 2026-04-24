"""
Graph Node Functions for LangGraph Pipeline.
Each node is a function that takes AgentState and returns a partial state update.
LLM-First Design: All logic uses LLM prompts rather than regex/keyword matching.
Structured Outputs: Uses response_schema for guaranteed valid JSON from LLMs.

Uncertainty model (Tier-1 research integration):
  - Execution entropy: Shannon entropy over canonicalized result clusters.
    See "Query and Conquer" (arXiv 2503.24364) and RTS++ (EDBT 2026).
  - Semantic entropy:  Shannon entropy over SQL skeleton clusters.
    Proxy for the bidirectional-entailment clustering from Farquhar et al.,
    "Detecting hallucinations in LLMs using semantic entropy", Nature 2024.
  - Composite confidence -> calibrated quality gate (data/calibration.json),
    inspired by "Conformal abstention" (arXiv 2405.01563).
"""
import hashlib
import json
import math
import os
import re
from typing import Dict, Any, List, Optional, Tuple

from .state import AgentState, PipelineConfig
from .schema_ref_diversity import detect_unanimous_structural_ambiguity
from src.data.vector_store import get_schema_store


# ==================== STRUCTURED OUTPUT SCHEMAS ====================
# These schemas are passed to LLM clients to guarantee valid JSON responses

DISAMBIGUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {"type": "number"},
        "is_ambiguous": {"type": "boolean"},
        "detected_tables": {"type": "array", "items": {"type": "string"}},
        "detected_intent": {
            "type": "string",
            "enum": ["select", "count", "sum", "average", "ranking", "filter", "join", "aggregate"]
        },
        "ambiguity_reasons": {"type": "array", "items": {"type": "string"}},
        "suggested_clarification": {"type": "string"}
    },
    "required": ["confidence", "is_ambiguous", "detected_tables", "detected_intent"]
}

SQL_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "sql_queries": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"}
    },
    "required": ["sql_queries"]
}

CONSISTENCY_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "are_equivalent": {"type": "boolean"},
        "confidence": {"type": "number"},
        "differences": {"type": "array", "items": {"type": "string"}},
        "recommendation": {
            "type": "string",
            "enum": ["proceed", "clarify", "retry"]
        }
    },
    "required": ["are_equivalent", "confidence", "recommendation"]
}

DRAFT_SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "draft_sql": {"type": "string"},
        "query_skeleton": {"type": "string"}
    },
    "required": ["draft_sql", "query_skeleton"]
}


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
    Format schema catalog into standard SQL DDL (CREATE TABLE) strings.
    DAIL-SQL highlights that LLMs inherently understand DDL better than custom textual representations.
    
    Supports minimal schema mode (available_tables only) for context overflow prevention.
    """
    if not schema:
        return "No schema available."
    
    lines = [f"-- Database: {schema.get('database_name', 'unknown')}\n"]
    
    # Handle minimal schema mode (only table names, no column details)
    # This is used when schema linking fails or returns no results
    if 'available_tables' in schema and not schema.get('tables'):
        lines.append("-- Available Tables (schema linking found no specific tables):")
        lines.append("-- Note: Ask for clarification about which tables to use.\n")
        for table in schema['available_tables']:
            lines.append(f"--   - {table}")
        if schema.get('error'):
            lines.append(f"\n-- (Schema linking error: {schema['error']})")
        return "\n".join(lines)
    
    relationships = []
    
    for table_name, table_info in schema.get('tables', {}).items():
        table_desc = table_info.get('description', '')
        if table_desc:
            lines.append(f"-- Table Description: {table_desc}")
            
        lines.append(f"CREATE TABLE {table_name} (")
        col_lines = []
        
        for col_name, col_info in table_info.get('columns', {}).items():
            col_type = col_info.get('type', 'TEXT')
            col_desc = col_info.get('description', '')
            
            # Build column annotation
            col_def = f"    {col_name} {col_type}"
            if col_info.get('is_primary_key'):
                col_def += " PRIMARY KEY"
                
            if col_info.get('foreign_key'):
                fk_target = col_info['foreign_key']
                # Store relationship to add as a comment later or inline
                relationships.append(f"FOREIGN KEY ({col_name}) REFERENCES {fk_target.split('.')[0]}({fk_target.split('.')[1] if '.' in fk_target else 'id'})")
                
            if col_desc:
                col_def += f" -- {col_desc}"
                
            if col_info.get('sample_values'):
                sample_list = ', '.join(f"'{v}'" for v in col_info['sample_values'][:3])
                col_def += f" (examples: {sample_list})"
                
            col_lines.append(col_def)
            
        lines.append(",\n".join(col_lines))
        lines.append(");")
        lines.append("")
        
    if relationships:
        lines.append("-- Table Relationships (Foreign Keys):")
        for rel in relationships:
            lines.append(f"--   - {rel}")
            
    # Add semantic notes if present
    semantic_notes = schema.get('semantic_notes', {})
    if semantic_notes:
        lines.append("\n-- Semantic Notes (Ambiguous Terms):")
        for term, info in semantic_notes.items():
            if info.get('ambiguous'):
                interpretations = info.get('possible_interpretations', [])
                interp_str = ', '.join(i.get('term', '') for i in interpretations)
                lines.append(f"--   - '{term}' can mean: {interp_str}")
                
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


def extract_tables_from_sql(sql: str) -> List[str]:
    """
    Extract table names from SQL by parsing FROM and JOIN clauses.
    Deterministic regex operation — no LLM call.
    
    Handles:
      - FROM table_name
      - FROM table_name AS alias
      - JOIN table_name
      - JOIN table_name AS alias
      - Comma-separated FROM lists: FROM t1, t2
    """
    if not sql:
        return []
    
    tables = set()
    sql_upper = sql.upper()
    
    # Match FROM <tables> and JOIN <table>
    # Pattern: FROM/JOIN followed by a table name (word chars only)
    from_join_pattern = re.compile(
        r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        re.IGNORECASE
    )
    for match in from_join_pattern.finditer(sql):
        table_name = match.group(1)
        # Exclude SQL keywords that can follow FROM/JOIN
        if table_name.upper() not in ('SELECT', 'WHERE', 'SET', 'VALUES', 'INTO', 'LATERAL'):
            tables.add(table_name)
    
    # Handle comma-separated tables after FROM: FROM t1, t2, t3
    comma_from = re.compile(
        r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)+)',
        re.IGNORECASE
    )
    for match in comma_from.finditer(sql):
        for part in match.group(1).split(','):
            table_name = part.strip().split()[0]  # Take name before any alias
            if table_name.upper() not in ('SELECT', 'WHERE', 'SET', 'VALUES', 'INTO', 'LATERAL'):
                tables.add(table_name)
    
    return list(tables)


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
    STRICT MODE: Does not default to full schema to prevent context overflow.
    Returns confidence score, detected entities, and clarification needs.
    """
    user_query = state.get('user_query', '')
    user_feedback = state.get('user_feedback', '')
    full_schema = load_schema_catalog()
    
    # Schema Linking: Retrieve only relevant tables for this query
    # Falls back to full schema if linking fails (needed for LLM context)
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
            # Fallback to full schema - LLM needs context to work properly
            print("Schema Linking: No specific tables matched. Using full schema for LLM context.")
            schema = full_schema
    except Exception as e:
        # Fallback to full schema on error
        print(f"Schema linking error: {e}. Using full schema fallback.")
        schema = full_schema
    
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
        # Use structured output schema for guaranteed valid JSON
        response, _ = llm_client.generate(prompt, response_schema=DISAMBIGUATION_SCHEMA)
        
        # Parse the response - with structured outputs this should always be valid JSON
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to regex parsing if structured output not supported by provider
            parsed = _parse_json_from_response(response)
        
        if not parsed:
            return default_response
        
        confidence = float(parsed.get('confidence', 0.5))
        
        clear_bar = float(getattr(config, "disambiguation_clear_confidence", 0.85))
        # High reported clarity skips clarification HITL (usability over strict
        # adherence to is_ambiguous when the model hedges).
        if confidence >= clear_bar:
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


def generate_draft_sql_node(state: AgentState, config: PipelineConfig, llm_client=None) -> Dict[str, Any]:
    """
    Generate an initial structural Draft SQL to be used for structural similarity search (DAIL-Selection).
    This performs a zero-shot prompt to generate an initial skeleton before full example retrieval.
    """
    user_query = state.get('user_query', '')
    schema_context = state.get('schema_context', '')
    
    if not llm_client:
        return {
            'draft_sql': "SELECT * FROM table",
            'query_skeleton': "SELECT * FROM table"
        }
        
    prompt = f"""You are an expert SQL analyst. Based on the user query and database schema, generate a Draft SQL query and its structural skeleton.
The skeleton should anonymize concrete values, replacing them with placeholders, to highlight the query's structure (e.g., SELECT col1 FROM tab1 WHERE col2 = <val>).

DATABASE SCHEMA:
{schema_context}

USER QUERY: "{user_query}"

Return a JSON object with:
{{
    "draft_sql": "<the draft sql query>",
    "query_skeleton": "<the structural skeleton of the query>"
}}
Return ONLY the JSON object, no other text."""

    try:
        response, _ = llm_client.generate(prompt, response_schema=DRAFT_SQL_SCHEMA)
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = _parse_json_from_response(response)
            
        return {
            'draft_sql': parsed.get('draft_sql', ''),
            'query_skeleton': parsed.get('query_skeleton', '')
        }
    except Exception as e:
        print(f"Draft SQL generation error: {e}")
        return {
            'draft_sql': "SELECT * FROM table",
            'query_skeleton': "SELECT * FROM table"
        }


def retrieve_examples_node(state: AgentState, config: PipelineConfig) -> Dict[str, Any]:
    """
    Retrieve relevant few-shot examples from the vector store.
    
    CRITICAL UPDATE: Performs Schema-Aware Filtering.
    Only uses examples where the referenced tables actually exist in the current DB.
    """
    examples = []
    try:
        from src.data.vector_store import get_vector_store
        
        # 1. Retrieve raw examples based on semantic similarity and structural similarity
        vector_store = get_vector_store()
        raw_examples = vector_store.retrieve(
            state.get('user_query', ''), 
            skeleton=state.get('query_skeleton', ''),
            n_results=10
        ) # Fetch more to allow for filtering
        
        # 2. Load current schema to validate table existence
        schema = load_schema_catalog()
        
        # 3. Filter examples
        if schema and schema.get('tables'):
            # Create a set of valid table names (normalized to lowercase)
            current_tables = set(k.lower() for k in schema.get('tables', {}).keys())
            
            valid_examples = []
            for ex in raw_examples:
                ex_tables = ex.get('tables', [])
                
                # If example has no specific tables, it's generic/safe to keep
                if not ex_tables:
                    valid_examples.append(ex)
                    continue
                
                # Check if ALL tables in the example exist in the current DB
                ex_tables_set = set(t.lower() for t in ex_tables)
                if ex_tables_set.issubset(current_tables):
                    valid_examples.append(ex)
                else:
                    # Debug log to show what is being skipped
                    missing = ex_tables_set - current_tables
                    print(f"Skipping example '{ex.get('question')}' (Tables {missing} not in current DB)")
            
            # Keep only the top 3 VALID examples
            examples = valid_examples[:3]
            
            if not examples and raw_examples:
                print("No relevant few-shot examples found for this schema. Using Zero-Shot mode.")
        else:
            # Fallback if schema fails to load
            examples = raw_examples[:3]

    except Exception as e:
        print(f"Vector store error: {e}")
        examples = []
    
    return {'few_shot_examples': examples}


def generate_sql_node(state: AgentState, config: PipelineConfig, llm_client=None) -> Dict[str, Any]:
    """
    Generate SQL using LLM based on the query, schema, and context.
    Generates multiple variations for consistency checking.
    
    Uses draft-SQL backtracing (Component 5 of DAIL-SQL plan):
    If a draft_sql exists, extract the tables it references and re-prune
    the schema context to only those tables for a tighter prompt.
    """
    user_query = state.get('user_query', '')
    if state.get('user_feedback'):
        user_query = f"{user_query}\n\nUser clarification: {state['user_feedback']}"
    
    schema_context = state.get('schema_context', '')
    few_shot_examples = state.get('few_shot_examples', [])
    error_context = state.get('debug_analysis', '')
    num_variations = config.num_sql_variations
    
    # Draft-SQL Backtracing: re-prune schema using tables from draft
    draft_sql = state.get('draft_sql', '')
    if draft_sql and draft_sql != 'SELECT * FROM table':
        referenced_tables = extract_tables_from_sql(draft_sql)
        if referenced_tables:
            full_schema = load_schema_catalog()
            pruned_schema = filter_schema_to_tables(full_schema, referenced_tables)
            if pruned_schema.get('tables'):
                schema_context = format_schema_for_prompt(pruned_schema)
                print(f"Draft backtracing: pruned schema to {len(pruned_schema['tables'])} tables: {list(pruned_schema['tables'].keys())}")
    
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

    prompt = f"""You are an expert SQL developer. Generate {num_variations} candidate SQL queries for the following question.

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

Guidelines for the candidate set:
- The first query should be your single best interpretation of the question.
- Subsequent queries should explore PLAUSIBLE ALTERNATIVE INTERPRETATIONS of any
  ambiguity in the question (different column choices, different join paths,
  different aggregation level). They are NOT required to be semantically
  equivalent. We use disagreement between them as an ambiguity signal.
- If the question is genuinely unambiguous, return slight syntactic variants
  of the same query so the candidate set is consistent.
- Use proper table and column names from the schema.
- Handle JOINs appropriately using foreign key relationships.
- Use appropriate aggregations (COUNT, SUM, AVG, etc.) based on the query intent.
- Include appropriate LIMIT clauses for open-ended queries.

Return ONLY the JSON object, no other text."""

    try:
        # Use structured output schema for guaranteed valid JSON
        response, token_data = llm_client.generate(prompt, response_schema=SQL_GENERATION_SCHEMA)
        
        # Parse the response - with structured outputs this should always be valid JSON
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to regex parsing if structured output not supported by provider
            parsed = _parse_json_from_response(response)
        
        if parsed and 'sql_queries' in parsed:
            sqls = parsed['sql_queries']
            if isinstance(sqls, list) and sqls:
                return {
                    'generated_sqls': sqls,
                    'selected_sql': sqls[0],
                    'token_confidence_map': token_data or []
                }
        
        # Try to parse SQL_N: format as fallback (for non-JSON responses)
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
                'selected_sql': sqls[0],
                'token_confidence_map': token_data or []
            }
        
        # Last resort: parsing produced nothing usable.
        return {
            'generated_sqls': ["SELECT 1;"],
            'selected_sql': "SELECT 1;",
            'token_confidence_map': token_data or [] if 'token_data' in locals() else [],
            'error_trace': "SQL generator returned no parseable SQL; emitting placeholder.",
        }

    except Exception as e:
        # Most commonly: Gemini 429 quota exhaustion that survived backoff.
        # Stamp error_trace so the quality gate sees this as a failed state
        # rather than incorrectly emitting a HITL with bogus high confidence.
        print(f"SQL generation error: {e}")
        return {
            'generated_sqls': ["SELECT 1;"],
            'selected_sql': "SELECT 1;",
            'error_trace': f"sql_generation_error: {type(e).__name__}: {str(e)[:200]}",
        }


# ==================== UNCERTAINTY HELPERS ====================

def _shannon_entropy_bits(counts: List[int]) -> float:
    """Shannon entropy in bits over a list of cluster counts."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def _normalize_skeleton(sql: str) -> str:
    """
    Reduce a SQL string to a structural skeleton for clustering.
    Collapses whitespace, uppercases keywords, replaces literals with '?'.
    Two SQLs with the same skeleton are taken as the same 'meaning class'.
    """
    if not sql:
        return ""
    s = sql.strip().rstrip(";")
    # Strip line comments and block comments
    s = re.sub(r"--[^\n]*", " ", s)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    # Strip string literals and numbers
    s = re.sub(r"'[^']*'", "?", s)
    s = re.sub(r'"[^"]*"', "?", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "?", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s


def _canonicalize_execution_result(result: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Produce a canonical hash of an execution result so that semantically
    equivalent outputs map to the same cluster id.

    Hashing is COLUMN-NAME-AGNOSTIC: two queries that return the same VALUES
    cluster together even if the labels differ (e.g. COUNT(*) vs COUNT(id)
    both returning 60). Rows are also order-independent. We keep the row
    *shape* (column count) as part of the hash so [(1,2)] never collapses
    onto [(1,)].

    Returns None for failed/empty executions so they are not clustered together.
    """
    if not result or not result.get("success"):
        return None
    cols = list(result.get("columns", []) or [])
    data = list(result.get("data", []) or [])

    rows: List[Tuple[str, ...]] = []
    for row in data:
        if isinstance(row, dict):
            # Use the column order returned by the executor (may be empty).
            if cols:
                values = [row.get(c) for c in cols]
            else:
                values = list(row.values())
        else:
            try:
                values = list(row)
            except TypeError:
                values = [row]
        # Sort cell values within the row -> tolerate column-order differences.
        rows.append(tuple(sorted(repr(v) for v in values)))
    rows.sort()
    arity = len(rows[0]) if rows else 0
    payload = json.dumps({"arity": arity, "rows": rows}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _logprob_features(token_map: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Length-normalized sequence log-prob and minimum token confidence.
    Higher avg_logprob is better (closer to 0 = more confident).
    """
    if not token_map:
        return {"sequence_logprob": None, "min_token_confidence": None}
    confs: List[float] = []
    for t in token_map:
        c = t.get("confidence")
        if isinstance(c, (int, float)) and 0.0 < c <= 1.0:
            confs.append(float(c))
    if not confs:
        return {"sequence_logprob": None, "min_token_confidence": None}
    log_confs = [math.log(c) for c in confs]
    avg_logprob = sum(log_confs) / len(log_confs)
    return {
        "sequence_logprob": avg_logprob,
        "min_token_confidence": min(confs),
    }


def _normalize_entropy(entropy_bits: float, n_candidates: int) -> float:
    """Normalize entropy to [0, 1] given the maximum possible entropy log2(n)."""
    if n_candidates <= 1:
        return 0.0
    max_h = math.log2(n_candidates)
    if max_h <= 0:
        return 0.0
    return max(0.0, min(1.0, entropy_bits / max_h))


def _normalize_logprob(avg_logprob: Optional[float]) -> float:
    """
    Map an average log-prob to roughly [0, 1].
    avg_logprob = 0  (perfectly confident)  -> 1.0
    avg_logprob = -3 (low confidence ~5%)   -> 0.0
    """
    if avg_logprob is None:
        return 0.5  # neutral when unavailable
    score = 1.0 + (avg_logprob / 3.0)
    return max(0.0, min(1.0, score))


def _compute_composite_confidence(
    self_reported: float,
    execution_entropy: float,
    semantic_entropy: float,
    n_candidates: int,
    avg_logprob: Optional[float],
    weights: Dict[str, float],
) -> float:
    """Weighted combination of all uncertainty signals -> [0, 1]."""
    exec_consistency = 1.0 - _normalize_entropy(execution_entropy, n_candidates)
    sem_consistency = 1.0 - _normalize_entropy(semantic_entropy, n_candidates)
    logprob_score = _normalize_logprob(avg_logprob)
    score = (
        weights.get("self_reported", 0.0) * float(self_reported)
        + weights.get("execution_consistency", 0.0) * exec_consistency
        + weights.get("semantic_consistency", 0.0) * sem_consistency
        + weights.get("logprob", 0.0) * logprob_score
    )
    return max(0.0, min(1.0, score))


def consistency_check_node(state: AgentState, config: PipelineConfig, llm_client=None) -> Dict[str, Any]:
    """
    Execution-aware consistency check (replaces the old LLM-as-judge approach).

    Steps:
      1. Execute every generated SQL in the sandbox.
      2. Cluster successful executions by canonicalized result -> EXECUTION ENTROPY.
      3. Cluster all SQLs by structural skeleton -> SEMANTIC ENTROPY.
      4. Promote the SQL from the largest execution cluster to `selected_sql`
         (majority-vote winner) and CACHE its execution result so the
         downstream `execute_sql_node` does not have to re-run it.
      5. Compute a composite confidence using calibrated weights.

    This addresses the failure mode where the LLM says "these queries are
    equivalent" but they actually return different rows on the live database.
    """
    sqls: List[str] = list(state.get('generated_sqls', []) or [])
    selected_sql = state.get('selected_sql', sqls[0] if sqls else "")
    self_reported = float(state.get('confidence_score', 0.5))
    token_map = state.get('token_confidence_map', []) or []
    calibration = (config.calibration if hasattr(config, "calibration") else None) or {}
    weights = calibration.get("weights", {}) if isinstance(calibration, dict) else {}

    logprob_feats = _logprob_features(token_map)

    if not sqls:
        return {
            'consistency_passed': False,
            'consistency_analysis': "No SQL candidates were generated.",
            'execution_entropy': 0.0,
            'semantic_entropy': 0.0,
            'result_clusters': [],
            'skeleton_clusters': [],
            'sql_executions': [],
            'unanimous_structural_divergence': False,
            'schema_diversity': {},
            'sequence_logprob': logprob_feats['sequence_logprob'],
            'min_token_confidence': logprob_feats['min_token_confidence'],
            'composite_confidence': 0.0,
        }

    # Placeholder detection: generate_sql_node emits "SELECT 1;" when the LLM
    # call failed (most often a Gemini 429). Without this guard the executor
    # happily runs SELECT 1; three times, gets perfect agreement, and produces
    # a high composite_confidence on a totally meaningless answer.
    placeholder_sqls = {"SELECT 1;", "SELECT 1", "select 1;", "select 1"}
    if all(s.strip() in placeholder_sqls for s in sqls):
        return {
            'consistency_passed': False,
            'consistency_analysis': "SQL generation failed (placeholder emitted). "
                                    "This usually indicates an upstream LLM error.",
            'execution_entropy': 0.0,
            'semantic_entropy': 0.0,
            'result_clusters': [],
            'skeleton_clusters': [],
            'sql_executions': [],
            'unanimous_structural_divergence': False,
            'schema_diversity': {},
            'sequence_logprob': logprob_feats['sequence_logprob'],
            'min_token_confidence': logprob_feats['min_token_confidence'],
            'composite_confidence': 0.0,
            'selected_sql': sqls[0],
        }

    # ---- 1. Execute every variation in the sandbox -----------------------
    try:
        from src.sandbox.executor import execute_sql as _execute_sql
    except Exception as e:
        _execute_sql = None
        print(f"Sandbox unavailable for execution-entropy check: {e}")

    executions: List[Dict[str, Any]] = []
    for idx, sql in enumerate(sqls):
        entry: Dict[str, Any] = {
            "index": idx,
            "sql": sql,
            "skeleton": _normalize_skeleton(sql),
            "success": False,
            "error": None,
            "row_count": 0,
            "result_hash": None,
            "result": None,
        }
        if _execute_sql is None or not sql.strip():
            executions.append(entry)
            continue
        try:
            res = _execute_sql(sql)
        except Exception as e:
            res = {"success": False, "error": str(e)}
        if res and res.get("success"):
            entry["success"] = True
            entry["row_count"] = res.get("row_count", len(res.get("data", []) or []))
            entry["result_hash"] = _canonicalize_execution_result(res)
            entry["result"] = res
        else:
            entry["error"] = (res or {}).get("error", "unknown error")
        executions.append(entry)

    # ---- 2. Cluster by canonical execution result ------------------------
    result_buckets: Dict[str, List[int]] = {}
    for e in executions:
        h = e["result_hash"]
        if h is None:
            continue
        result_buckets.setdefault(h, []).append(e["index"])

    result_clusters: List[Dict[str, Any]] = []
    for h, idxs in result_buckets.items():
        sample = executions[idxs[0]]
        sample_rows = (sample["result"] or {}).get("data", [])[:3]
        result_clusters.append({
            "hash": h,
            "size": len(idxs),
            "indices": idxs,
            "sample_sql": sample["sql"],
            "sample_rows": sample_rows,
            "row_count": sample["row_count"],
        })
    result_clusters.sort(key=lambda c: c["size"], reverse=True)

    exec_entropy = _shannon_entropy_bits([c["size"] for c in result_clusters])

    # ---- 3. Cluster by structural skeleton -------------------------------
    skeleton_buckets: Dict[str, List[int]] = {}
    for e in executions:
        skeleton_buckets.setdefault(e["skeleton"], []).append(e["index"])
    skeleton_clusters = [
        {"skeleton": s, "size": len(idxs), "indices": idxs,
         "sqls": [executions[i]["sql"] for i in idxs]}
        for s, idxs in skeleton_buckets.items()
    ]
    skeleton_clusters.sort(key=lambda c: c["size"], reverse=True)

    semantic_entropy = _shannon_entropy_bits([c["size"] for c in skeleton_clusters])

    # ---- 4. Pick winner via execution-cluster majority vote --------------
    cached_execution: Optional[Dict[str, Any]] = None
    if result_clusters:
        winner_idx = result_clusters[0]["indices"][0]
        selected_sql = executions[winner_idx]["sql"]
        cached_execution = executions[winner_idx]["result"]
    elif skeleton_clusters:
        winner_idx = skeleton_clusters[0]["indices"][0]
        selected_sql = executions[winner_idx]["sql"]

    # ---- 5. Composite confidence + acceptance criterion ------------------
    composite = _compute_composite_confidence(
        self_reported=self_reported,
        execution_entropy=exec_entropy,
        semantic_entropy=semantic_entropy,
        n_candidates=len(sqls),
        avg_logprob=logprob_feats["sequence_logprob"],
        weights=weights,
    )

    thresholds = calibration.get("thresholds", {}) if isinstance(calibration, dict) else {}
    exec_thresh = float(thresholds.get("execution_entropy_max", 0.6))
    sem_thresh = float(thresholds.get("semantic_entropy_max", 1.2))

    n_success = sum(1 for e in executions if e["success"])
    # Execution-based agreement is a much stronger signal than skeleton-based
    # diversity. Skeleton diversity *without* execution disagreement is just
    # paraphrase ("COUNT(*)" vs "COUNT(id)" returning the same number) and
    # must not be allowed to reject a correct answer. Only fall back to the
    # semantic-entropy cap when execution evidence is inconclusive (multiple
    # disagreeing result clusters, or fewer than half the candidates ran).
    exec_conclusive = (
        len(result_clusters) == 1
        and n_success >= max(2, (len(sqls) + 1) // 2)
    )
    consistency_passed = (
        n_success > 0
        and exec_entropy <= exec_thresh
        and (exec_conclusive or semantic_entropy <= sem_thresh)
    )

    u_div, div_detail = detect_unanimous_structural_ambiguity(
        sqls,
        result_clusters=result_clusters,
        skeleton_clusters=skeleton_clusters,
        execution_entropy=exec_entropy,
        semantic_entropy=semantic_entropy,
        n_success=n_success,
        n_sqls=len(sqls),
        thresholds=thresholds,
    )
    if u_div:
        consistency_passed = False

    if not result_clusters:
        analysis = (
            f"All {len(sqls)} candidate(s) failed to execute -> consistency cannot be verified."
        )
    elif len(result_clusters) == 1:
        analysis = (
            f"All {n_success}/{len(sqls)} successful candidates returned the same result "
            f"(execution entropy = 0 bits)."
        )
        if u_div:
            analysis += (
                " Candidates disagree on which tables or columns to use while returning "
                "identical rows; treating as schema-reference ambiguity and abstaining for clarification."
            )
    else:
        top = result_clusters[0]["size"]
        analysis = (
            f"Found {len(result_clusters)} distinct result clusters "
            f"(execution entropy = {exec_entropy:.2f} bits, top cluster has {top}/{n_success} votes). "
            f"This indicates semantic ambiguity in the question."
        )

    update: Dict[str, Any] = {
        'consistency_passed': consistency_passed,
        'consistency_analysis': analysis,
        'execution_entropy': exec_entropy,
        'semantic_entropy': semantic_entropy,
        'result_clusters': result_clusters,
        'skeleton_clusters': skeleton_clusters,
        'unanimous_structural_divergence': u_div,
        'schema_diversity': div_detail,
        'sql_executions': [
            {k: v for k, v in e.items() if k != "result"}  # drop heavy payload
            for e in executions
        ],
        'sequence_logprob': logprob_feats['sequence_logprob'],
        'min_token_confidence': logprob_feats['min_token_confidence'],
        'composite_confidence': composite,
        'selected_sql': selected_sql,
    }

    # Cache the chosen SQL's execution so execute_sql_node can reuse it
    if cached_execution is not None:
        update['execution_result'] = cached_execution
        update['execution_success'] = True
        update['error_trace'] = None

    return update


def _structural_consistency_check(sqls: List[str]) -> Dict[str, Any]:
    """
    Fallback structural consistency check using keyword matching.
    Used when LLM is unavailable or fails.
    """
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
        analysis = "Structural check: All variations have consistent structure (JOINs, WHERE, GROUP BY)."
    else:
        analysis = f"Structural check: {', '.join(inconsistencies)}. This may indicate semantic ambiguity."
    
    return {
        'consistency_passed': consistency_passed,
        'consistency_analysis': analysis
    }


import re
import sqlite3
import os

def evaluate_sql_node(state: AgentState, config: PipelineConfig, llm_client=None) -> Dict[str, Any]:
    """
    Mixture of Experts (MoE) Evaluation Node
    Runs generated SQL through multiple objective and LLM-based experts to adjust confidence and trigger HITL.
    """
    sql = state.get('selected_sql', '')
    original_confidence = state.get('confidence_score', 0.5)
    schema_context = state.get('schema_context', '')
    user_query = state.get('user_query', '')
    
    if not sql:
        return {'expert_approved': False, 'evaluation_results': [{'expert': 'System', 'confidence_penalty': 0.5, 'reasoning': 'No SQL generated to evaluate', 'is_approved': False}]}

    results = []
    total_penalty = 0.0

    # 1. Objective Expert: AST Complexity
    # Very basic AST complexity logic: count JOINs and nested select wrappers
    join_count = sql.upper().count('JOIN')
    subquery_count = sql.upper().count('(SELECT')
    complexity_penalty = 0.0
    reasoning = "Query structure looks manageable."
    
    if join_count > 3 or subquery_count > 2:
        complexity_penalty = 0.2
        reasoning = f"Query is highly complex ({join_count} JOINs, {subquery_count} subqueries) which increases risk of hallucinations."
    
    results.append({
        'expert': 'Complexity Evaluator',
        'is_approved': complexity_penalty == 0.0,
        'confidence_penalty': complexity_penalty,
        'reasoning': reasoning
    })

    # 2. Objective Expert: SQLite Dry-Run Validator 
    # Attempt to simply parse the SQL statement against the active db mapping to catch obvious typos
    syntax_penalty = 0.0
    syntax_reasoning = "No syntax errors detected during static check."
    try:
        # Load active DB path
        db_path = None
        # `load_schema_catalog` is defined in this same file (`nodes.py`)
        schema_cat = load_schema_catalog()
        active_db = "company_analytics" # Defaulting for demo safety, ideally parse from session/config
        
        if schema_cat.get('databases', {}).get(active_db):
            db_path = schema_cat['databases'][active_db].get('file')
            
        if db_path and os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # EXPLAIN allows parsing without actually executing hazardous logic
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            conn.close()
    except Exception as e:
        syntax_penalty = 0.3
        syntax_reasoning = f"SQL Syntax Error detected during dry-run validation: {str(e)}"
    
    results.append({
        'expert': 'Execution Validator',
        'is_approved': syntax_penalty == 0.0,
        'confidence_penalty': syntax_penalty,
        'reasoning': syntax_reasoning
    })

    # 3. Objective Expert: Token Confidence Evaluator
    token_map = state.get('token_confidence_map', [])
    token_penalty = 0.0
    token_reasoning = "Token probabilities are within acceptable bounds."
    
    if token_map:
        min_token_prob = min([t.get('confidence', 1.0) for t in token_map])
        if min_token_prob < 0.75:
            token_penalty = 0.15
            token_reasoning = f"Generated query contains highly uncertain tokens (lowest confidence: {min_token_prob:.0%}). Risk of hallucination."
    
    results.append({
        'expert': 'Token Confidence Evaluator',
        'is_approved': token_penalty == 0.0,
        'confidence_penalty': token_penalty,
        'reasoning': token_reasoning
    })

    # LLM Experts
    if llm_client:
        expert_schema = {
            "type": "object",
            "properties": {
                "is_approved": {"type": "boolean"},
                "confidence_penalty": {"type": "number"},
                "reasoning": {"type": "string"}
            },
            "required": ["is_approved", "confidence_penalty", "reasoning"]
        }

        # 3. LLM Expert: Schema & Syntax Validation
        syntax_prompt = f"""You are a strict SQL Syntax and Schema alignment expert.
Review the following query exactly as generated against the database schema.
User Query: "{user_query}"
Generated SQL: "{sql}"
Schema Context: {schema_context}

Task: Verify all identifiers (table names, column names) actually exist in the schema. Check for logical impossibilities or obvious type mismatches.
Return a JSON object with:
- is_approved (bool)
- confidence_penalty (float 0.0 to 1.0, e.g. 0.0 for flawless, 0.4 for severe issues)
- reasoning (string explaining your penalty, if any)
"""
        # 4. LLM Expert: Semantic Logic Expert
        semantic_prompt = f"""You are a strict Data Analytics Business Logic expert.
Review the intent of the user's query and compare it to the semantic logic generated in the SQL.
User Query: "{user_query}"
Generated SQL: "{sql}"

Task: Look for logical oversights like integer vs float division errors, missing explicit GROUP BY variables when aggregating, or incorrect directional sorting.
Return a JSON object with:
- is_approved (bool)
- confidence_penalty (float 0.0 to 1.0)
- reasoning (string explaining your penalty, if any)
"""
        
        try:
            for prompt_text, expert_name in [(syntax_prompt, "Schema Alignment Expert"), (semantic_prompt, "Semantic Logic Expert")]:
                llm_resp, _ = llm_client.generate(prompt_text, response_schema=expert_schema)
                try:
                    expert_eval = json.loads(llm_resp)
                except:
                    expert_eval = _parse_json_from_response(llm_resp)
                
                results.append({
                    'expert': expert_name,
                    'is_approved': expert_eval.get('is_approved', True),
                    'confidence_penalty': float(expert_eval.get('confidence_penalty', 0.0)),
                    'reasoning': expert_eval.get('reasoning', 'No reasoning provided')
                })
        except Exception as e:
            print(f"MoE LLM Evaluation failed: {e}")

    # Aggregate penalties
    for r in results:
        total_penalty += r.get('confidence_penalty', 0.0)
        
    final_confidence = max(0.0, original_confidence - total_penalty)
    expert_approved = all([r.get('is_approved', True) for r in results])

    return {
        'evaluation_results': results,
        'expert_approved': expert_approved,
        'confidence_score': final_confidence  # Overwrite original mapping
    }


def execute_sql_node(state: AgentState, config: PipelineConfig) -> Dict[str, Any]:
    """
    Execute the selected SQL in the sandbox.
    Reuses the cached execution from `consistency_check_node` when the
    selected SQL was already run there (saves ~one DB round-trip per query).
    """
    sql = state.get('selected_sql', '')

    if not sql:
        return {
            'execution_success': False,
            'error_trace': "No SQL to execute",
            'execution_result': None
        }

    # Reuse cached execution from execution-entropy phase if it matches.
    cached = state.get('execution_result')
    if cached and state.get('execution_success'):
        for entry in state.get('sql_executions', []) or []:
            if entry.get('sql') == sql and entry.get('success'):
                return {
                    'execution_success': True,
                    'execution_result': cached,
                    'error_trace': None,
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
    """Finalize the state when HITL is needed.

    Builds a human-readable explanation that combines:
      - the disambiguator's clarification request (if any)
      - the quality-gate failure reasons (if any)
      - up to 3 alternative result-cluster summaries so the user can pick
        which interpretation they meant (LogicalBeam-style top-k disambiguation).
    """
    parts: List[str] = []
    base = state.get('clarification_message', '').strip()
    if base:
        parts.append(base)

    reasons = state.get('quality_gate_reasons', []) or []
    if reasons:
        parts.append("Why I'm pausing:")
        for r in reasons:
            parts.append(f"  - {r}")

    clusters = state.get('result_clusters', []) or []
    if len(clusters) > 1:
        parts.append("")
        parts.append("I found multiple possible interpretations of your question:")
        for i, c in enumerate(clusters[:3], start=1):
            sample_rows = c.get('sample_rows', [])
            parts.append(
                f"  Option {i}: {c.get('row_count', 0)} rows "
                f"({c.get('size', 0)}/{len(state.get('generated_sqls', []))} candidates agree)."
            )
            if sample_rows:
                parts.append(f"    sample: {sample_rows[0]}")
        parts.append("Please clarify which interpretation you intended.")

    msg = "\n".join(parts) if parts else "Clarification needed."
    return {
        'final_status': 'paused_hitl',
        'final_message': msg,
    }


def finalize_failure_node(state: AgentState, config: PipelineConfig) -> Dict[str, Any]:
    """Finalize the state for failed execution."""
    return {
        'final_status': 'failed',
        'final_message': f"Failed after {state.get('retry_count', 0)} retries. Last error: {state.get('error_trace', 'Unknown')}"
    }
