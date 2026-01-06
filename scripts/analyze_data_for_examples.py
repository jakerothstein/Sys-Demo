#!/usr/bin/env python3
"""
Analyze SQLite database to suggest realistic few-shot examples.
This script examines your actual data and generates example queries
that match real patterns in your database.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

# Path to the database (try multiple common locations)
DATA_DIR = Path(__file__).parent.parent / 'data'
EXAMPLES_PATH = DATA_DIR / 'few_shot_examples.json'

# Try to find a database in common locations
def find_database():
    """Find the SQLite database in common locations."""
    search_paths = [
        DATA_DIR / 'ecommerce.db',
        DATA_DIR / 'demo.db',
        DATA_DIR / 'custom' / 'company_analytics.db',
        DATA_DIR / 'custom' / 'sample_company' / 'sample_company.sqlite',
    ]
    
    # Also search for any .db or .sqlite file in data/
    for path in search_paths:
        if path.exists():
            return path
    
    # Fallback: search data/ directory
    for ext in ['*.db', '*.sqlite']:
        for db_file in DATA_DIR.rglob(ext):
            if 'benchmark' not in str(db_file) and 'chroma' not in str(db_file):
                return db_file
    
    return None

DB_PATH = find_database()


def connect_db():
    """Connect to the SQLite database."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return None
    return sqlite3.connect(DB_PATH)


def get_table_info(conn):
    """Get all tables and their columns."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {}
    
    for (table_name,) in cursor.fetchall():
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [(row[1], row[2]) for row in cursor.fetchall()]
        tables[table_name] = columns
    
    return tables


def analyze_column_values(conn, table, column):
    """Get distinct values for a column (useful for filters)."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT DISTINCT {column} FROM {table} LIMIT 20;")
        return [row[0] for row in cursor.fetchall() if row[0] is not None]
    except:
        return []


def suggest_examples(conn, tables):
    """Generate example queries based on actual data patterns."""
    examples = []
    example_id = 100  # Start high to not conflict with existing
    
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    
    def make_example(question, sql, intent, table_list, difficulty):
        """Helper to create an example in the standard format."""
        nonlocal example_id
        ex = {
            "id": f"script_{datetime.now().strftime('%Y%m%d')}_{example_id}",
            "question": question,
            "sql": sql,
            "intent": intent,
            "tables": table_list,
            "difficulty": difficulty,
            "saved_at": timestamp,
            "source": "script_generated"
        }
        example_id += 1
        return ex
    
    # 1. Basic count queries for each table
    for table in tables:
        examples.append(make_example(
            f"How many records are in {table}?",
            f"SELECT COUNT(*) AS count FROM {table};",
            "count",
            [table],
            "easy"
        ))
    
    # 2. Find categorical columns and suggest filter queries
    for table, columns in tables.items():
        for col_name, col_type in columns:
            if 'status' in col_name.lower() or 'category' in col_name.lower() or 'type' in col_name.lower():
                values = analyze_column_values(conn, table, col_name)
                if values and len(values) < 10:  # Only if it's a real categorical
                    sample_value = values[0]
                    examples.append(make_example(
                        f"Show all {table} where {col_name} is '{sample_value}'",
                        f"SELECT * FROM {table} WHERE {col_name} = '{sample_value}' LIMIT 10;",
                        "filter",
                        [table],
                        "easy"
                    ))
    
    # 3. Find numeric columns for aggregations
    for table, columns in tables.items():
        for col_name, col_type in columns:
            if any(t in col_type.upper() for t in ['INT', 'REAL', 'DECIMAL', 'FLOAT', 'NUMERIC']):
                if any(kw in col_name.lower() for kw in ['price', 'amount', 'total', 'cost', 'revenue', 'quantity']):
                    examples.append(make_example(
                        f"What is the total {col_name} in {table}?",
                        f"SELECT SUM({col_name}) AS total_{col_name} FROM {table};",
                        "aggregation",
                        [table],
                        "easy"
                    ))
                    
                    examples.append(make_example(
                        f"What is the average {col_name} in {table}?",
                        f"SELECT AVG({col_name}) AS avg_{col_name} FROM {table};",
                        "aggregation",
                        [table],
                        "easy"
                    ))
    
    # 4. Find date columns for time-based queries
    for table, columns in tables.items():
        for col_name, col_type in columns:
            if any(kw in col_name.lower() for kw in ['date', 'created', 'updated', 'time']):
                examples.append(make_example(
                    f"Show {table} from the last 30 days",
                    f"SELECT * FROM {table} WHERE {col_name} >= date('now', '-30 days') LIMIT 10;",
                    "date_filter",
                    [table],
                    "medium"
                ))
    
    # 5. Find foreign key patterns for JOIN suggestions
    for table, columns in tables.items():
        for col_name, col_type in columns:
            if col_name.endswith('_id') and col_name != 'id':
                ref_table = col_name[:-3] + 's'  # Guess: customer_id -> customers
                if ref_table in tables:
                    examples.append(make_example(
                        f"Show {table} with their {ref_table[:-1]} details",
                        f"SELECT t.*, r.* FROM {table} t JOIN {ref_table} r ON t.{col_name} = r.id LIMIT 10;",
                        "join",
                        [table, ref_table],
                        "medium"
                    ))
    
    # 6. Discount-related examples (CRITICAL for fixing the bug!)
    if 'sales' in tables:
        sales_cols = [c[0] for c in tables['sales']]
        if 'discount_percent' in sales_cols and 'unit_price' in sales_cols:
            examples.append(make_example(
                "Calculate the discounted price for each sale",
                "SELECT sale_id, unit_price, discount_percent, unit_price * (1 - discount_percent/100.0) AS discounted_price FROM sales LIMIT 10;",
                "discount_calculation",
                ["sales"],
                "medium"
            ))
            
            examples.append(make_example(
                "What is the net revenue after applying discounts?",
                "SELECT SUM(quantity * unit_price * (1 - discount_percent/100.0)) AS net_revenue FROM sales;",
                "aggregation_with_discount",
                ["sales"],
                "medium"
            ))
    
    return examples


def main():
    print("=" * 60)
    print("Few-Shot Example Generator")
    print("=" * 60)
    
    conn = connect_db()
    if not conn:
        return
    
    print(f"\nConnected to: {DB_PATH}")
    
    # Get schema
    tables = get_table_info(conn)
    print(f"\nFound {len(tables)} tables:")
    for table, columns in tables.items():
        print(f"  - {table}: {len(columns)} columns")
    
    # Generate suggestions
    print("\n" + "=" * 60)
    print("Suggested Examples")
    print("=" * 60)
    
    examples = suggest_examples(conn, tables)
    
    for ex in examples:
        print(f"\n[{ex['id']}] {ex['question']}")
        print(f"   SQL: {ex['sql'][:80]}{'...' if len(ex['sql']) > 80 else ''}")
        print(f"   Tables: {', '.join(ex['tables'])}, Intent: {ex['intent']}, Difficulty: {ex['difficulty']}")
    
    # Option to save
    print("\n" + "=" * 60)
    print(f"\nGenerated {len(examples)} example suggestions.")
    print(f"\nTo add these to your few-shot examples file, you can:")
    print(f"  1. Review and select the ones you like")
    print(f"  2. Copy them to: {EXAMPLES_PATH}")
    
    # Save to a separate file for review
    output_path = Path(__file__).parent / 'suggested_examples.json'
    with open(output_path, 'w') as f:
        json.dump({"suggested_examples": examples}, f, indent=4)
    print(f"\nAll suggestions saved to: {output_path}")
    
    conn.close()


if __name__ == '__main__':
    main()
