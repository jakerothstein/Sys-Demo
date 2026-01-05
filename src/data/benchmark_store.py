"""
Benchmark Results Storage.
Stores and retrieves benchmark evaluation results using SQLite.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


class BenchmarkStore:
    """SQLite-based storage for benchmark results."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'benchmark_results.db')
        
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Benchmark runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                dataset TEXT,
                database_id TEXT,
                total_questions INTEGER DEFAULT 0,
                completed_questions INTEGER DEFAULT 0,
                successful INTEGER DEFAULT 0,
                failed INTEGER DEFAULT 0,
                hitl_triggered INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        """)
        
        # Individual question results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS question_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                question_index INTEGER,
                question TEXT,
                difficulty TEXT,
                expected_hitl INTEGER,
                hitl_triggered INTEGER,
                hitl_feedback TEXT,
                generated_sql TEXT,
                execution_success INTEGER,
                execution_error TEXT,
                row_count INTEGER,
                confidence REAL,
                latency_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_run(self, dataset: str, database_id: str, total_questions: int) -> str:
        """Create a new benchmark run."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO benchmark_runs (run_id, started_at, dataset, database_id, total_questions, status)
            VALUES (?, ?, ?, ?, ?, 'running')
        """, (run_id, datetime.now().isoformat(), dataset, database_id, total_questions))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def add_result(self, run_id: str, result: Dict[str, Any]):
        """Add a question result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO question_results 
            (run_id, question_index, question, difficulty, expected_hitl, hitl_triggered, 
             hitl_feedback, generated_sql, execution_success, execution_error, 
             row_count, confidence, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            result.get('question_index', 0),
            result.get('question', ''),
            result.get('difficulty', 'unknown'),
            1 if result.get('expected_hitl') else 0,
            1 if result.get('hitl_triggered') else 0,
            result.get('hitl_feedback', ''),
            result.get('sql', ''),
            1 if result.get('success') else 0,
            result.get('error', ''),
            result.get('row_count', 0),
            result.get('confidence', 0),
            result.get('latency_ms', 0)
        ))
        
        # Update run stats
        cursor.execute("""
            UPDATE benchmark_runs 
            SET completed_questions = completed_questions + 1,
                successful = successful + ?,
                failed = failed + ?,
                hitl_triggered = hitl_triggered + ?
            WHERE run_id = ?
        """, (
            1 if result.get('success') else 0,
            0 if result.get('success') else 1,
            1 if result.get('hitl_triggered') else 0,
            run_id
        ))
        
        conn.commit()
        conn.close()
    
    def complete_run(self, run_id: str):
        """Mark a run as completed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE benchmark_runs 
            SET completed_at = ?, status = 'completed'
            WHERE run_id = ?
        """, (datetime.now().isoformat(), run_id))
        
        conn.commit()
        conn.close()
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a benchmark run by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_run_results(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all results for a run."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM question_results 
            WHERE run_id = ? 
            ORDER BY question_index
        """, (run_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get all benchmark runs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM benchmark_runs 
            ORDER BY started_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get overall analytics across all runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_runs,
                SUM(total_questions) as total_questions,
                SUM(successful) as total_successful,
                SUM(failed) as total_failed,
                SUM(hitl_triggered) as total_hitl
            FROM benchmark_runs
        """)
        overall = dict(zip(['total_runs', 'total_questions', 'total_successful', 'total_failed', 'total_hitl'], 
                          cursor.fetchone()))
        
        # Success rate by difficulty
        cursor.execute("""
            SELECT 
                difficulty,
                COUNT(*) as count,
                SUM(execution_success) as successes,
                AVG(confidence) as avg_confidence,
                SUM(hitl_triggered) as hitl_count
            FROM question_results
            GROUP BY difficulty
        """)
        by_difficulty = [dict(zip(['difficulty', 'count', 'successes', 'avg_confidence', 'hitl_count'], row)) 
                        for row in cursor.fetchall()]
        
        # Recent runs
        cursor.execute("""
            SELECT run_id, started_at, dataset, database_id, 
                   total_questions, successful, failed, status
            FROM benchmark_runs 
            ORDER BY started_at DESC 
            LIMIT 10
        """)
        recent_runs = [dict(zip(['run_id', 'started_at', 'dataset', 'database_id', 
                                'total_questions', 'successful', 'failed', 'status'], row)) 
                      for row in cursor.fetchall()]
        
        # HITL accuracy
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN expected_hitl = hitl_triggered THEN 1 ELSE 0 END) as correct,
                COUNT(*) as total
            FROM question_results
            WHERE expected_hitl IS NOT NULL
        """)
        hitl_row = cursor.fetchone()
        hitl_accuracy = hitl_row[0] / hitl_row[1] if hitl_row and hitl_row[1] > 0 else 0
        
        conn.close()
        
        return {
            'overall': overall,
            'by_difficulty': by_difficulty,
            'recent_runs': recent_runs,
            'hitl_accuracy': hitl_accuracy,
            'success_rate': overall['total_successful'] / overall['total_questions'] if overall['total_questions'] else 0
        }
    
    def delete_run(self, run_id: str):
        """Delete a benchmark run and its results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM question_results WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM benchmark_runs WHERE run_id = ?", (run_id,))
        
        conn.commit()
        conn.close()


# Singleton instance
_store: Optional[BenchmarkStore] = None


def get_benchmark_store() -> BenchmarkStore:
    """Get or create the singleton benchmark store."""
    global _store
    if _store is None:
        _store = BenchmarkStore()
    return _store
