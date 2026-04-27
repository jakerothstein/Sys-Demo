import json
import os
import sqlite3
import sys
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from llm.client import create_llm_client

def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}');")
        columns = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
        schema_parts.append(f"Table {table}: {', '.join(columns)}")
    conn.close()
    return "\n".join(schema_parts)

def calculate_kappa(labels1: List[bool], labels2: List[bool]) -> float:
    if len(labels1) != len(labels2):
        return 0.0
    n = len(labels1)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(labels1, labels2) if a == b) / n
    
    y1 = sum(1 for x in labels1 if x) / n
    n1 = 1 - y1
    y2 = sum(1 for x in labels2 if x) / n
    n2 = 1 - y2
    
    pe = (y1 * y2) + (n1 * n2)
    
    if pe >= 1:
        return 1.0
    return (po - pe) / (1 - pe)

def main():
    with open("data/ambiguity_benchmark.json", "r") as f:
        benchmark = json.load(f)
    
    db_schemas = {}
    db_paths = {
        "company_analytics": "data/custom/company_analytics.db",
        "sample_company": "data/custom/sample_company/sample_company.sqlite",
        "sec_data_v2": "data/custom/sec_data_v2.db"
    }
    
    for db_id, path in db_paths.items():
        try:
            db_schemas[db_id] = get_schema(path)
        except Exception as e:
            print(f"Error loading schema for {db_id}: {e}")
            db_schemas[db_id] = "Schema information unavailable."
    
    # Use OpenAI as second annotator
    client = create_llm_client(provider="openai", model="gpt-4o")
    
    annotator1_labels = []
    annotator2_labels = []
    
    results = []
    
    print(f"Starting re-annotation of {len(benchmark['questions'])} questions...")
    
    for q in benchmark["questions"]:
        db_id = q["db_id"]
        schema = db_schemas.get(db_id, "Unknown schema")
        question_text = q["question"]
        
        prompt = f"""You are an independent data annotation expert for a Text-to-SQL project.
Task: Determine if the following user query is "Ambiguous" or "Clear" given the database schema.

"Ambiguous" (expected_hitl=True) means:
- The question has multiple valid interpretations (e.g., "Who earns the most?" could mean highest salary or highest total revenue).
- The question is underspecified (e.g., "Show the top customers" without defining "top").
- The question refers to a concept not clearly mapped to a single column (e.g., "workload").
- The question lacks necessary filters (e.g., "What is the budget?" when multiple budgets exist).

"Clear" (expected_hitl=False) means:
- The question maps directly and uniquely to a SQL query (e.g., "How many employees are there?").

Schema Context:
{schema}

User Question: "{question_text}"

Return a JSON object with:
- expected_hitl (boolean): True if ambiguous, False if clear.
- reasoning (string): Short explanation of your choice.
"""
        
        response_schema = {
            "type": "object",
            "properties": {
                "expected_hitl": {"type": "boolean"},
                "reasoning": {"type": "string"}
            },
            "required": ["expected_hitl", "reasoning"],
            "additionalProperties": False
        }
        
        try:
            resp_text, _ = client.generate(prompt, response_schema=response_schema)
            resp = json.loads(resp_text)
            
            annotator1_labels.append(q["expected_hitl"])
            annotator2_labels.append(resp["expected_hitl"])
            
            results.append({
                "id": q["id"],
                "question": question_text,
                "original_hitl": q["expected_hitl"],
                "reannotated_hitl": resp["expected_hitl"],
                "reasoning": resp["reasoning"],
                "original_rationale": q.get("rationale", "")
            })
            
            print(f"[{q['id']}] Match: {q['expected_hitl'] == resp['expected_hitl']}")
        except Exception as e:
            print(f"Error annotating {q['id']}: {e}")

    if not annotator1_labels:
        print("No annotations performed.")
        return

    kappa = calculate_kappa(annotator1_labels, annotator2_labels)
    agreement = sum(1 for a, b in zip(annotator1_labels, annotator2_labels) if a == b) / len(annotator1_labels)
    
    summary = {
        "n_total": len(annotator1_labels),
        "n_matches": sum(1 for a, b in zip(annotator1_labels, annotator2_labels) if a == b),
        "agreement": agreement,
        "cohen_kappa": kappa,
        "results": results
    }
    
    with open("data/ambiguity_iaa_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*40)
    print("IAA RESULTS")
    print("="*40)
    print(f"Total Questions: {summary['n_total']}")
    print(f"Raw Agreement: {agreement:.1%}")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print("="*40)

if __name__ == "__main__":
    main()
