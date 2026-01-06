"""
Flask Application for Text-to-SQL Web UI.
Integrates the LangGraph pipeline with a modern web interface.
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph.pipeline import create_pipeline, TextToSQLGraph
from src.graph.state import PipelineConfig
from src.graph.nodes import load_schema_catalog
from src.llm.client import create_llm_client
from src.data.vector_store import get_schema_store

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize pipeline with configuration
config = PipelineConfig(
    confidence_threshold=0.7,
    max_retries=3,
    num_sql_variations=3,
    llm_provider=os.environ.get("LLM_PROVIDER", "mock")
)

# Create LLM client (auto-detects available providers)
llm_client = create_llm_client(provider="auto")

# Create pipeline
pipeline = create_pipeline(config=config, llm_client=llm_client)

# Store pending HITL states
pending_hitl = {}


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle a new NL query."""
    data = request.json
    user_query = data.get('query', '')
    session_id = data.get('session_id', 'default')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # Run the pipeline
    result = pipeline.run(user_query, session_id=session_id)
    
    # Store state if HITL needed
    if result.get('final_status') == 'paused_hitl':
        pending_hitl[session_id] = {
            'original_query': user_query,
            'state': result
        }
    
    # Format response for frontend
    response = format_response(result)
    return jsonify(response)


@app.route('/api/feedback', methods=['POST'])
def handle_feedback():
    """Handle HITL feedback to continue a paused query."""
    data = request.json
    session_id = data.get('session_id', 'default')
    feedback = data.get('feedback', '')
    
    if session_id not in pending_hitl:
        return jsonify({"error": "No pending query for this session"}), 400
    
    original_query = pending_hitl[session_id]['original_query']
    del pending_hitl[session_id]
    
    # Continue pipeline with feedback
    result = pipeline.run(original_query, user_feedback=feedback, session_id=session_id)
    
    response = format_response(result)
    return jsonify(response)


@app.route('/api/schema', methods=['GET'])
def get_schema():
    """Return the current database schema."""
    import json
    schema_path = os.path.join(os.path.dirname(__file__), 'data', 'schema_catalog.json')
    
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            return jsonify(json.load(f))
    
    return jsonify({"error": "Schema not found"}), 404


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Return few-shot examples."""
    import json
    examples_path = os.path.join(os.path.dirname(__file__), 'data', 'few_shot_examples.json')
    
    if os.path.exists(examples_path):
        with open(examples_path, 'r') as f:
            return jsonify(json.load(f))
    
    return jsonify({"examples": []})


@app.route('/api/databases', methods=['GET'])
def list_databases():
    """List available benchmark databases."""
    from src.sandbox.executor import get_benchmark_executor, get_execution_mode
    
    executor = get_benchmark_executor()
    databases = executor.list_databases()
    current = get_execution_mode()
    
    return jsonify({
        "databases": databases,
        "current_mode": current['mode'],
        "current_database": current['database']
    })


@app.route('/api/databases/switch', methods=['POST'])
def switch_database():
    """Switch to a different database."""
    from src.sandbox.executor import set_execution_mode, get_current_schema
    
    data = request.json
    mode = data.get('mode', 'demo')
    db_id = data.get('db_id')
    dataset = data.get('dataset', 'bird')
    
    success = set_execution_mode(mode, db_id, dataset)
    
    if success:
        schema = get_current_schema()
        return jsonify({
            "success": True,
            "message": f"Switched to {mode} mode" + (f" ({db_id})" if db_id else ""),
            "schema": schema
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Failed to load database: {db_id}"
        }), 400


@app.route('/api/databases/schema', methods=['GET'])
def get_current_db_schema():
    """Get schema of current database."""
    from src.sandbox.executor import get_current_schema, get_execution_mode
    
    mode = get_execution_mode()
    schema = get_current_schema()
    
    return jsonify({
        "mode": mode['mode'],
        "database": mode['database'],
        "schema": schema
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    from src.sandbox.executor import get_execution_mode
    
    mode = get_execution_mode()
    return jsonify({
        "status": "healthy",
        "llm_provider": config.llm_provider,
        "llm_available": llm_client.is_available() if llm_client else False,
        "database_mode": mode['mode'],
        "database": mode['database']
    })


# ============== BENCHMARK ENDPOINTS ==============

# Sample benchmark questions for testing
BENCHMARK_QUESTIONS = [
    {"question": "Show me the employees", "difficulty": "simple", "expected_hitl": False},
    {"question": "What is the budget?", "difficulty": "ambiguous", "expected_hitl": True, "evidence": "Could refer to department or project"},
    {"question": "List all active projects with their team members", "difficulty": "medium", "expected_hitl": False},
    {"question": "Who earns the most?", "difficulty": "ambiguous", "expected_hitl": True, "evidence": "Needs salary clarification"},
    {"question": "How many people work in Engineering?", "difficulty": "simple", "expected_hitl": False},
    {"question": "Show project costs", "difficulty": "ambiguous", "expected_hitl": True},
    {"question": "Find the manager", "difficulty": "ambiguous", "expected_hitl": True, "evidence": "Which manager?"},
    {"question": "What's the total?", "difficulty": "very_ambiguous", "expected_hitl": True, "evidence": "Total of what?"},
    {"question": "List employees hired in 2020", "difficulty": "medium", "expected_hitl": False},
    {"question": "Show department spending", "difficulty": "complex", "expected_hitl": True}
]

# Active benchmark state
active_benchmark = {}


@app.route('/api/benchmark/questions', methods=['GET'])
def get_benchmark_questions():
    """Get available benchmark questions."""
    return jsonify({
        "questions": BENCHMARK_QUESTIONS,
        "total": len(BENCHMARK_QUESTIONS)
    })


@app.route('/api/benchmark/start', methods=['POST'])
def start_benchmark():
    """Start a new benchmark run."""
    from src.data.benchmark_store import get_benchmark_store
    from src.sandbox.executor import get_execution_mode
    
    data = request.json or {}
    question_count = min(data.get('count', len(BENCHMARK_QUESTIONS)), len(BENCHMARK_QUESTIONS))
    
    mode = get_execution_mode()
    store = get_benchmark_store()
    
    run_id = store.create_run(
        dataset='custom',
        database_id=mode.get('database', 'demo'),
        total_questions=question_count
    )
    
    # Store active benchmark state
    active_benchmark[run_id] = {
        'questions': BENCHMARK_QUESTIONS[:question_count],
        'current_index': 0,
        'waiting_hitl': False,
        'current_question': None
    }
    
    return jsonify({
        "run_id": run_id,
        "total_questions": question_count,
        "message": "Benchmark started. Call /api/benchmark/next to get questions."
    })


@app.route('/api/benchmark/next', methods=['POST'])
def next_benchmark_question():
    """Get and run the next benchmark question."""
    from src.data.benchmark_store import get_benchmark_store
    import time
    
    data = request.json or {}
    run_id = data.get('run_id')
    feedback = data.get('feedback')  # For HITL responses
    
    if not run_id or run_id not in active_benchmark:
        return jsonify({"error": "Invalid or expired run_id"}), 400
    
    bench = active_benchmark[run_id]
    store = get_benchmark_store()
    
    # If we were waiting for HITL feedback
    if bench['waiting_hitl'] and feedback:
        q = bench['current_question']
        start_time = time.time()
        
        # Continue pipeline with feedback
        result = pipeline.run(q['question'], user_feedback=feedback, session_id=f"bench_{run_id}_{bench['current_index']}")
        latency = int((time.time() - start_time) * 1000)
        
        # Record result
        store.add_result(run_id, {
            'question_index': bench['current_index'],
            'question': q['question'],
            'difficulty': q.get('difficulty', 'unknown'),
            'expected_hitl': q.get('expected_hitl', False),
            'hitl_triggered': True,
            'hitl_feedback': feedback,
            'sql': result.get('selected_sql', ''),
            'success': result.get('final_status') == 'success',
            'error': result.get('error_trace', ''),
            'row_count': result.get('execution_result', {}).get('row_count', 0),
            'confidence': result.get('confidence_score', 0),
            'latency_ms': latency
        })
        
        bench['waiting_hitl'] = False
        bench['current_index'] += 1
        
        formatted = format_response(result)
        formatted['question_index'] = bench['current_index'] - 1
        formatted['hitl_resolved'] = True
        
        # Check if done
        if bench['current_index'] >= len(bench['questions']):
            store.complete_run(run_id)
            del active_benchmark[run_id]
            formatted['benchmark_complete'] = True
        
        return jsonify(formatted)
    
    # Get next question
    if bench['current_index'] >= len(bench['questions']):
        store.complete_run(run_id)
        del active_benchmark[run_id]
        return jsonify({
            "benchmark_complete": True,
            "run_id": run_id,
            "message": "Benchmark complete!"
        })
    
    q = bench['questions'][bench['current_index']]
    bench['current_question'] = q
    
    start_time = time.time()
    result = pipeline.run(q['question'], session_id=f"bench_{run_id}_{bench['current_index']}")
    latency = int((time.time() - start_time) * 1000)
    
    hitl_triggered = result.get('final_status') == 'paused_hitl'
    
    if hitl_triggered:
        # Wait for user input
        bench['waiting_hitl'] = True
        formatted = format_response(result)
        formatted['question_index'] = bench['current_index']
        formatted['question'] = q['question']
        formatted['difficulty'] = q.get('difficulty', 'unknown')
        formatted['expected_hitl'] = q.get('expected_hitl', False)
        formatted['evidence'] = q.get('evidence', '')
        formatted['needs_feedback'] = True
        return jsonify(formatted)
    else:
        # Record result and move to next
        store.add_result(run_id, {
            'question_index': bench['current_index'],
            'question': q['question'],
            'difficulty': q.get('difficulty', 'unknown'),
            'expected_hitl': q.get('expected_hitl', False),
            'hitl_triggered': False,
            'sql': result.get('selected_sql', ''),
            'success': result.get('final_status') == 'success',
            'error': result.get('error_trace', ''),
            'row_count': result.get('execution_result', {}).get('row_count', 0),
            'confidence': result.get('confidence_score', 0),
            'latency_ms': latency
        })
        
        bench['current_index'] += 1
        
        formatted = format_response(result)
        formatted['question_index'] = bench['current_index'] - 1
        formatted['question'] = q['question']
        formatted['difficulty'] = q.get('difficulty', 'unknown')
        
        # Check if done
        if bench['current_index'] >= len(bench['questions']):
            store.complete_run(run_id)
            del active_benchmark[run_id]
            formatted['benchmark_complete'] = True
        
        return jsonify(formatted)


@app.route('/api/benchmark/skip', methods=['POST'])
def skip_benchmark_question():
    """Skip a HITL question in the benchmark."""
    from src.data.benchmark_store import get_benchmark_store
    
    data = request.json or {}
    run_id = data.get('run_id')
    
    if not run_id or run_id not in active_benchmark:
        return jsonify({"error": "Invalid run_id"}), 400
    
    bench = active_benchmark[run_id]
    store = get_benchmark_store()
    
    if bench['waiting_hitl']:
        q = bench['current_question']
        store.add_result(run_id, {
            'question_index': bench['current_index'],
            'question': q['question'],
            'difficulty': q.get('difficulty', 'unknown'),
            'expected_hitl': q.get('expected_hitl', False),
            'hitl_triggered': True,
            'hitl_feedback': 'SKIPPED',
            'success': False,
            'error': 'Skipped by user',
            'confidence': 0
        })
        
        bench['waiting_hitl'] = False
        bench['current_index'] += 1
    
    return jsonify({"message": "Skipped", "next_index": bench['current_index']})


@app.route('/api/benchmark/runs', methods=['GET'])
def get_benchmark_runs():
    """Get all benchmark runs."""
    from src.data.benchmark_store import get_benchmark_store
    
    store = get_benchmark_store()
    runs = store.get_all_runs()
    
    return jsonify({"runs": runs})


@app.route('/api/benchmark/run/<run_id>', methods=['GET'])
def get_benchmark_run(run_id):
    """Get details of a specific benchmark run."""
    from src.data.benchmark_store import get_benchmark_store
    
    store = get_benchmark_store()
    run = store.get_run(run_id)
    results = store.get_run_results(run_id)
    
    if not run:
        return jsonify({"error": "Run not found"}), 404
    
    return jsonify({
        "run": run,
        "results": results
    })


@app.route('/api/benchmark/analytics', methods=['GET'])
def get_analytics():
    """Get benchmark analytics."""
    from src.data.benchmark_store import get_benchmark_store
    
    store = get_benchmark_store()
    analytics = store.get_analytics()
    
    return jsonify(analytics)


def format_response(result: dict) -> dict:
    """Format pipeline result for frontend consumption."""
    status = result.get('final_status', 'unknown')
    
    # Map internal status to frontend status
    status_map = {
        'success': 'SUCCESS',
        'paused_hitl': 'PAUSED_HITL',
        'failed': 'FAILED',
        'in_progress': 'IN_PROGRESS'
    }
    
    response = {
        'status': status_map.get(status, 'UNKNOWN'),
        'message': result.get('final_message', ''),
        
        # Analysis details
        'confidence': result.get('confidence_score', 0),
        'is_ambiguous': result.get('is_ambiguous', False),
        'ambiguity_reasons': result.get('ambiguity_reasons', []),
        'detected_tables': result.get('detected_tables', []),
        'detected_intent': result.get('detected_intent', 'unknown'),
        
        # SQL details
        'sql': result.get('selected_sql', ''),
        'sql_variations': result.get('generated_sqls', []),
        'consistency_passed': result.get('consistency_passed', True),
        'consistency_analysis': result.get('consistency_analysis', ''),
        
        # Execution details
        'data': [],
        'columns': [],
        'row_count': 0,
        
        # Few-shot examples used
        'few_shot_examples': result.get('few_shot_examples', []),
        
        # Debug info
        'retry_count': result.get('retry_count', 0),
        'debug_analysis': result.get('debug_analysis', '')
    }
    
    # Add execution results if successful
    exec_result = result.get('execution_result')
    if exec_result and exec_result.get('success'):
        response['data'] = exec_result.get('data', [])
        response['columns'] = exec_result.get('columns', [])
        response['row_count'] = exec_result.get('row_count', 0)
    
    # Add error trace if failed
    if result.get('error_trace'):
        response['error'] = result.get('error_trace')
    
    return response


if __name__ == '__main__':
    print("=" * 50)
    print("Text-to-SQL Research Pipeline")
    print("=" * 50)
    print(f"LLM Provider: {config.llm_provider}")
    print(f"LLM Available: {llm_client.is_available() if llm_client else False}")
    print(f"Confidence Threshold: {config.confidence_threshold}")
    print(f"Max Retries: {config.max_retries}")
    print("=" * 50)
    
    # Initialize Schema Store for Schema Linking
    print("Initializing schema store for Schema Linking...")
    try:
        schema = load_schema_catalog()
        if schema and schema.get('tables'):
            schema_store = get_schema_store()
            schema_store.index_schema(schema)
            print(f"Schema store ready: {len(schema.get('tables', {}))} tables indexed.")
        else:
            print("Warning: No schema loaded, schema linking will use fallback.")
    except Exception as e:
        print(f"Schema store initialization warning: {e}")
    
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5000)
