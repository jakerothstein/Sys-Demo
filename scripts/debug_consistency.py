#!/usr/bin/env python3
"""Debug script: shows exactly what the consistency check computes for question 0."""
import os, sys, json
sys.path.insert(0, os.path.abspath('.'))

from scripts.load_repo_env import load_repo_env
load_repo_env()

from src.sandbox.executor import set_execution_mode, execute_sql
from src.graph.state import PipelineConfig, load_calibration
from src.graph.nodes import _normalize_skeleton, _canonicalize_execution_result, _shannon_entropy_bits
from src.graph.schema_ref_diversity import detect_unanimous_structural_ambiguity

# Patch consistency_check_node to print everything
set_execution_mode('benchmark', 'concert_singer', 'spider')

sqls = [
    "SELECT COUNT(*) AS total_singers FROM singer;",
    "SELECT COUNT(*) AS total_singers FROM singer;",
    "SELECT COUNT(*) AS total_singers FROM singer;",
]

# Simulate what the LLM likely generated — the 3 variants
# Let's just run the actual pipeline node in isolation with debug output

from src.llm.client import create_llm_client
from src.graph.state import AgentState

llm = create_llm_client(provider="openai")
config = PipelineConfig(
    confidence_threshold=0.55,
    disambiguation_clear_confidence=0.70,
    max_retries=2,
    num_sql_variations=3,
    llm_provider="openai",
)

# First run disambiguation to get state
from src.graph.nodes import disambiguate_node, generate_draft_sql_node, retrieve_examples_node, generate_sql_node

state: AgentState = {
    'user_query': "How many singers do we have?",
    'session_id': 'debug',
    'db_id': 'concert_singer',
    'dataset': 'spider',
    'retry_count': 0,
    'max_retries': 2,
    'final_status': 'in_progress',
}

print("=== DISAMBIGUATE ===")
dis = disambiguate_node(state, config, llm)
state.update(dis)
print(f"confidence_score: {state.get('confidence_score')}")
print(f"needs_clarification: {state.get('needs_clarification')}")

print("\n=== DRAFT SQL ===")
draft = generate_draft_sql_node(state, config, llm)
state.update(draft)
print(f"draft_sql: {state.get('draft_sql')}")

print("\n=== RETRIEVE EXAMPLES ===")
ex = retrieve_examples_node(state, config)
state.update(ex)
print(f"few_shot_examples count: {len(state.get('few_shot_examples', []))}")

print("\n=== GENERATE SQL ===")
gen = generate_sql_node(state, config, llm)
state.update(gen)
print(f"generated_sqls: {json.dumps(state.get('generated_sqls'), indent=2)}")

print("\n=== MANUAL CONSISTENCY CHECK ===")
sqls = state.get('generated_sqls', [])
calibration = config.calibration
thresholds = calibration.get('thresholds', {})

# Execute each
executions = []
for idx, sql in enumerate(sqls):
    try:
        res = execute_sql(sql)
    except Exception as e:
        res = {"success": False, "error": str(e)}
    skeleton = _normalize_skeleton(sql)
    h = _canonicalize_execution_result(res) if res and res.get("success") else None
    executions.append({
        "idx": idx, "sql": sql, "skeleton": skeleton,
        "success": res.get("success", False) if res else False,
        "error": res.get("error") if res else "no res",
        "hash": h,
        "row_count": res.get("row_count", 0) if res else 0,
    })
    print(f"SQL {idx}: success={executions[-1]['success']}, hash={h[:8] if h else None}, error={executions[-1]['error']}")
    print(f"  skeleton: {skeleton[:80]}")

# Cluster
result_buckets = {}
for e in executions:
    if e["hash"]:
        result_buckets.setdefault(e["hash"], []).append(e["idx"])
result_clusters = [{"hash": h, "size": len(idxs), "indices": idxs} for h, idxs in result_buckets.items()]
n_success = sum(1 for e in executions if e["success"])
exec_entropy = _shannon_entropy_bits([c["size"] for c in result_clusters])

skeleton_buckets = {}
for e in executions:
    skeleton_buckets.setdefault(e["skeleton"], []).append(e["idx"])
skeleton_clusters = [{"skeleton": s, "size": len(idxs)} for s, idxs in skeleton_buckets.items()]

skel_succ = {}
for e in executions:
    if e["success"]:
        skel_succ.setdefault(e["skeleton"], []).append(e["idx"])
skel_succ_clusters = [{"size": len(v)} for v in skel_succ.values()]
semantic_entropy = _shannon_entropy_bits([c["size"] for c in skel_succ_clusters]) if n_success > 0 else _shannon_entropy_bits([c["size"] for c in skeleton_clusters])

print(f"\nn_success={n_success}, n_sqls={len(sqls)}")
print(f"result_clusters count: {len(result_clusters)}")
print(f"skeleton_clusters count: {len(skeleton_clusters)}")
print(f"exec_entropy: {exec_entropy}")
print(f"semantic_entropy (succ only): {semantic_entropy}")

exec_conclusive = len(result_clusters) == 1 and n_success >= max(2, (len(sqls) + 1) // 2)
exec_thresh = float(thresholds.get("execution_entropy_max", 0.85))
sem_thresh = float(thresholds.get("semantic_entropy_max", 1.60))

print(f"\nexec_conclusive: {exec_conclusive}")
print(f"exec_thresh: {exec_thresh}, sem_thresh: {sem_thresh}")
consistency_passed = (
    n_success > 0
    and exec_entropy <= exec_thresh
    and (exec_conclusive or semantic_entropy <= sem_thresh)
)
print(f"consistency_passed (pre-u_div): {consistency_passed}")

# Structural divergence
semantic_entropy_all = _shannon_entropy_bits([c["size"] for c in skeleton_clusters])
print(f"semantic_entropy_all: {semantic_entropy_all}")
u_div, div_detail = detect_unanimous_structural_ambiguity(
    sqls,
    result_clusters=result_clusters,
    skeleton_clusters=skeleton_clusters,
    execution_entropy=exec_entropy,
    semantic_entropy=semantic_entropy_all,
    n_success=n_success,
    n_sqls=len(sqls),
    thresholds=thresholds,
)
print(f"u_div: {u_div}, reason: {div_detail.get('reason')}")
if u_div:
    consistency_passed = False
print(f"FINAL consistency_passed: {consistency_passed}")
