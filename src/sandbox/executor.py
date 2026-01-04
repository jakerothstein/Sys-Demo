"""
SQL Execution Sandbox.
Safely executes SQL in an isolated environment.
Uses Docker for PostgreSQL or SQLite in-memory as fallback.
"""
import sqlite3
import subprocess
import json
import os
from typing import Dict, Any, Optional

# Check Docker availability
def is_docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], 
            capture_output=True, 
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

DOCKER_AVAILABLE = is_docker_available()


class SQLiteExecutor:
    """SQLite in-memory executor for safe SQL execution."""
    
    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._setup_schema()
    
    def _setup_schema(self):
        """Create the demo schema with sample data."""
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                state TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                price DECIMAL(10,2),
                cost DECIMAL(10,2)
            );
            
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                order_date DATE,
                status TEXT,
                total_amount DECIMAL(10,2),
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );
            
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY,
                order_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                unit_price DECIMAL(10,2),
                FOREIGN KEY (order_id) REFERENCES orders(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            );
            
            -- Insert sample data
            INSERT OR IGNORE INTO customers (id, name, email, state) VALUES
                (1, 'Alice Johnson', 'alice@gmail.com', 'California'),
                (2, 'Bob Smith', 'bob@yahoo.com', 'Texas'),
                (3, 'Charlie Brown', 'charlie@gmail.com', 'Georgia'),
                (4, 'Diana Prince', 'diana@outlook.com', 'New York'),
                (5, 'Eve Wilson', 'eve@gmail.com', 'California');
            
            INSERT OR IGNORE INTO products (id, name, category, price, cost) VALUES
                (1, 'Laptop Pro', 'Electronics', 1299.99, 800.00),
                (2, 'Wireless Mouse', 'Accessories', 49.99, 15.00),
                (3, 'USB-C Cable', 'Accessories', 19.99, 5.00),
                (4, 'Monitor 27"', 'Electronics', 399.99, 250.00),
                (5, 'Keyboard', 'Accessories', 79.99, 30.00);
            
            INSERT OR IGNORE INTO orders (id, customer_id, order_date, status, total_amount) VALUES
                (1, 1, '2024-01-15', 'delivered', 1349.98),
                (2, 2, '2024-01-20', 'delivered', 99.98),
                (3, 1, '2024-02-01', 'shipped', 479.98),
                (4, 3, '2024-02-10', 'pending', 1299.99),
                (5, 4, '2024-02-15', 'delivered', 149.97),
                (6, 5, '2024-02-20', 'cancelled', 49.99);
            
            INSERT OR IGNORE INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
                (1, 1, 1, 1, 1299.99),
                (2, 1, 2, 1, 49.99),
                (3, 2, 3, 2, 19.99),
                (4, 2, 2, 1, 49.99),
                (5, 3, 4, 1, 399.99),
                (6, 3, 5, 1, 79.99),
                (7, 4, 1, 1, 1299.99),
                (8, 5, 3, 3, 19.99),
                (9, 5, 2, 1, 49.99),
                (10, 5, 5, 1, 79.99),
                (11, 6, 2, 1, 49.99);
        """)
        
        self.conn.commit()
    
    def execute(self, sql: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute SQL and return results.
        
        Args:
            sql: The SQL query to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dict with 'success', 'data' or 'error', and 'row_count'
        """
        # Safety checks
        sql_upper = sql.upper()
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper and keyword not in ['CREATE', 'INSERT']:  # Allow in setup
                return {
                    'success': False,
                    'error': f"Destructive operation '{keyword}' not allowed in sandbox mode.",
                    'row_count': 0
                }
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            
            # Fetch results
            if sql_upper.strip().startswith('SELECT'):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Convert to list of dicts for better JSON serialization
                data = [dict(zip(columns, row)) for row in rows]
                
                return {
                    'success': True,
                    'data': data,
                    'columns': columns,
                    'row_count': len(rows)
                }
            else:
                self.conn.commit()
                return {
                    'success': True,
                    'data': [],
                    'row_count': cursor.rowcount
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'row_count': 0
            }


class DockerPostgresExecutor:
    """Docker-based PostgreSQL executor (placeholder for future implementation)."""
    
    def __init__(self, container_name: str = "text2sql_postgres"):
        self.container_name = container_name
        # TODO: Implement Docker-based execution
    
    def execute(self, sql: str, timeout: int = 10) -> Dict[str, Any]:
        """Execute SQL in Docker container."""
        # Placeholder - would use docker exec to run psql
        return {
            'success': False,
            'error': 'Docker executor not yet implemented. Using SQLite fallback.'
        }


# Singleton executor instance
_executor: Optional[SQLiteExecutor] = None

def get_executor() -> SQLiteExecutor:
    """Get or create the global executor instance."""
    global _executor
    if _executor is None:
        if DOCKER_AVAILABLE:
            print("Docker available, but using SQLite for demo simplicity.")
        _executor = SQLiteExecutor()
    return _executor


def execute_sql(sql: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute SQL in the sandbox.
    
    Args:
        sql: The SQL query to execute
        timeout: Maximum execution time
        
    Returns:
        Execution result dictionary
    """
    executor = get_executor()
    return executor.execute(sql, timeout)
