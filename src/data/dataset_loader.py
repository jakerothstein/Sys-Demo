"""
BIRD-bench / Spider 2.0 Dataset Loader

Handles loading and managing research benchmark datasets for Text-to-SQL evaluation.
Supports both BIRD (Big Bench for Large-Scale Database Grounded Text-to-SQL) 
and Spider 2.0 datasets.

Dataset Structure Expected:
- BIRD: data/bird/{dev|train}/databases/{db_name}/{db_name}.sqlite
- Spider: data/spider/{database}/{database}.sqlite
"""

import json
import os
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class DatasetLoader:
    """Load and manage BIRD-bench and Spider datasets."""
    
    SUPPORTED_DATASETS = ['bird', 'spider', 'custom']
    
    def __init__(self, data_root: str = None):
        """
        Initialize the dataset loader.
        
        Args:
            data_root: Root directory containing dataset folders.
                      Defaults to project's data/ directory.
        """
        if data_root is None:
            data_root = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        
        self.data_root = os.path.abspath(data_root)
        self.current_dataset: Optional[str] = None
        self.current_db: Optional[str] = None
        self.available_databases: Dict[str, List[str]] = {}
        
        self._scan_datasets()
    
    def _scan_datasets(self):
        """Scan for available datasets and databases."""
        # Scan BIRD dataset
        bird_path = os.path.join(self.data_root, 'bird')
        if os.path.exists(bird_path):
            self.available_databases['bird'] = self._find_sqlite_dbs(bird_path)
        
        # Scan Spider dataset
        spider_path = os.path.join(self.data_root, 'spider')
        if os.path.exists(spider_path):
            self.available_databases['spider'] = self._find_sqlite_dbs(spider_path)
        
        # Scan custom databases
        custom_path = os.path.join(self.data_root, 'custom')
        if os.path.exists(custom_path):
            self.available_databases['custom'] = self._find_sqlite_dbs(custom_path)
    
    def _find_sqlite_dbs(self, root_path: str) -> List[str]:
        """Find all SQLite database files in a directory tree."""
        databases = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith('.sqlite') or file.endswith('.db'):
                    # Store relative path from data root
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.data_root)
                    databases.append(rel_path)
        return databases
    
    def list_datasets(self) -> Dict[str, int]:
        """List available datasets and their database counts."""
        return {k: len(v) for k, v in self.available_databases.items()}
    
    def list_databases(self, dataset: str = None) -> List[str]:
        """List available databases, optionally filtered by dataset."""
        if dataset:
            return self.available_databases.get(dataset, [])
        
        all_dbs = []
        for dbs in self.available_databases.values():
            all_dbs.extend(dbs)
        return all_dbs
    
    def get_database_path(self, db_path: str) -> str:
        """Get full path to a database file."""
        return os.path.join(self.data_root, db_path)
    
    def load_database(self, db_path: str) -> sqlite3.Connection:
        """Load a database and return a connection."""
        full_path = self.get_database_path(db_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Database not found: {full_path}")
        
        conn = sqlite3.connect(full_path)
        self.current_db = db_path
        return conn
    
    def extract_schema(self, db_path: str) -> Dict[str, Any]:
        """
        Extract schema information from a SQLite database.
        
        Returns a schema catalog compatible with our pipeline format.
        """
        conn = self.load_database(db_path)
        cursor = conn.cursor()
        
        # Get database name from path
        db_name = os.path.splitext(os.path.basename(db_path))[0]
        
        schema = {
            "database_name": db_name,
            "source": "bird" if "bird" in db_path else ("spider" if "spider" in db_path else "custom"),
            "tables": {}
        }
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            table_info = {
                "description": f"Table {table_name}",
                "columns": {}
            }
            
            # Get column info
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = cursor.fetchall()
            
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                
                col_info = {
                    "type": col_type or "TEXT",
                    "description": col_name.replace("_", " ").title(),
                    "is_primary_key": bool(is_pk)
                }
                
                # Get sample values (up to 3)
                try:
                    cursor.execute(f"SELECT DISTINCT \"{col_name}\" FROM \"{table_name}\" WHERE \"{col_name}\" IS NOT NULL LIMIT 3")
                    samples = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
                    if samples:
                        col_info["sample_values"] = samples[:3]
                except:
                    pass
                
                table_info["columns"][col_name] = col_info
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
            fks = cursor.fetchall()
            for fk in fks:
                _, _, ref_table, from_col, to_col, *_ = fk
                if from_col in table_info["columns"]:
                    table_info["columns"][from_col]["foreign_key"] = f"{ref_table}.{to_col}"
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM \"{table_name}\"")
            row_count = cursor.fetchone()[0]
            table_info["row_count"] = row_count
            
            schema["tables"][table_name] = table_info
        
        conn.close()
        return schema
    
    def load_dev_questions(self, dataset: str = 'bird') -> List[Dict[str, Any]]:
        """
        Load development set questions for evaluation.
        
        Returns list of {question, sql, db_id} dicts.
        """
        questions = []
        
        if dataset == 'bird':
            # Try multiple possible BIRD question file locations
            possible_paths = [
                os.path.join(self.data_root, 'bird', 'dev', 'dev.json'),
                os.path.join(self.data_root, 'bird', 'dev.json'),
                os.path.join(self.data_root, 'bird', 'dev_questions.json'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            questions.append({
                                "question": item.get("question", ""),
                                "sql": item.get("SQL", item.get("sql", "")),
                                "db_id": item.get("db_id", ""),
                                "evidence": item.get("evidence", ""),
                                "difficulty": item.get("difficulty", "unknown")
                            })
                    break
        
        elif dataset == 'spider':
            # Spider question format
            possible_paths = [
                os.path.join(self.data_root, 'spider', 'dev.json'),
                os.path.join(self.data_root, 'spider', 'train_spider.json'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            questions.append({
                                "question": item.get("question", ""),
                                "sql": item.get("query", item.get("sql", "")),
                                "db_id": item.get("db_id", ""),
                                "difficulty": item.get("hardness", "unknown")
                            })
                    break
        
        return questions
    
    def get_few_shot_examples(self, db_id: str, n: int = 5) -> List[Dict[str, str]]:
        """Get few-shot examples for a specific database."""
        all_questions = self.load_dev_questions('bird') + self.load_dev_questions('spider')
        
        # Filter by db_id
        db_examples = [q for q in all_questions if q.get('db_id') == db_id]
        
        # If not enough, fallback to any examples
        if len(db_examples) < n:
            db_examples = all_questions[:n]
        
        return [{"question": ex["question"], "sql": ex["sql"]} for ex in db_examples[:n]]


class BenchmarkRunner:
    """Run benchmarks against BIRD/Spider datasets."""
    
    def __init__(self, loader: DatasetLoader):
        self.loader = loader
        self.results: List[Dict[str, Any]] = []
    
    def evaluate_query(self, predicted_sql: str, gold_sql: str, 
                       db_path: str) -> Dict[str, Any]:
        """
        Evaluate predicted SQL against gold standard.
        
        Returns execution accuracy and exact match metrics.
        """
        result = {
            "predicted_sql": predicted_sql,
            "gold_sql": gold_sql,
            "exact_match": False,
            "execution_match": False,
            "error": None
        }
        
        # Normalize and compare
        pred_normalized = self._normalize_sql(predicted_sql)
        gold_normalized = self._normalize_sql(gold_sql)
        result["exact_match"] = pred_normalized == gold_normalized
        
        # Execute both and compare results
        try:
            conn = self.loader.load_database(db_path)
            cursor = conn.cursor()
            
            # Execute gold
            cursor.execute(gold_sql)
            gold_results = set(cursor.fetchall())
            
            # Execute predicted
            cursor.execute(predicted_sql)
            pred_results = set(cursor.fetchall())
            
            result["execution_match"] = gold_results == pred_results
            conn.close()
            
        except Exception as e:
            result["error"] = str(e)
        
        self.results.append(result)
        return result
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        # Basic normalization: lowercase, remove extra whitespace
        sql = sql.lower().strip()
        sql = ' '.join(sql.split())
        # Remove trailing semicolon
        sql = sql.rstrip(';')
        return sql
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate overall benchmark metrics."""
        if not self.results:
            return {"exact_match": 0.0, "execution_accuracy": 0.0}
        
        exact_matches = sum(1 for r in self.results if r["exact_match"])
        exec_matches = sum(1 for r in self.results if r["execution_match"])
        total = len(self.results)
        
        return {
            "exact_match": exact_matches / total,
            "execution_accuracy": exec_matches / total,
            "total_evaluated": total,
            "errors": sum(1 for r in self.results if r["error"])
        }


# Singleton instance
_dataset_loader: Optional[DatasetLoader] = None


def get_dataset_loader() -> DatasetLoader:
    """Get or create singleton dataset loader."""
    global _dataset_loader
    if _dataset_loader is None:
        _dataset_loader = DatasetLoader()
    return _dataset_loader


def get_current_database_info() -> Dict[str, Any]:
    """Get info about currently loaded database."""
    loader = get_dataset_loader()
    return {
        "current_db": loader.current_db,
        "available_datasets": loader.list_datasets(),
        "total_databases": sum(len(dbs) for dbs in loader.available_databases.values())
    }
