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
from src.llm.client import create_llm_client

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


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "llm_provider": config.llm_provider,
        "llm_available": llm_client.is_available() if llm_client else False
    })


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
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, port=5000)
