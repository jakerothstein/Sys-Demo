"""
SQL Execution Sandbox.
Safely executes SQL in an isolated environment.
Uses Docker for PostgreSQL or SQLite in-memory as fallback.

Now supports BIRD-bench and Spider 2.0 SQLite databases.
"""
import sqlite3
import subprocess
import json
import os
from typing import Dict, Any, Optional, List


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
    """SQLite executor supporting both in-memory and file-based databases."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite executor.
        
        Args:
            db_path: Path to SQLite database file, or None for in-memory demo database
        """
        self.db_path = db_path
        self.db_name = "demo_ecommerce"
        
        if db_path is not None and os.path.exists(db_path):
            # Connect to existing database (read-only for safety)
            self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
            self.db_name = os.path.splitext(os.path.basename(db_path))[0]
            self.is_benchmark_db = True
        else:
            # Create in-memory demo database
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._setup_demo_schema()
            self.is_benchmark_db = False
    
    def _setup_demo_schema(self):
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
    
    def get_schema(self) -> Dict[str, Any]:
        """Extract schema from the current database."""
        cursor = self.conn.cursor()
        schema = {"database_name": self.db_name, "tables": {}}
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            table_info = {"columns": {}}
            
            # Get column info
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = cursor.fetchall()
            
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                table_info["columns"][col_name] = {
                    "type": col_type or "TEXT",
                    "is_primary_key": bool(is_pk)
                }
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM \"{table_name}\"")
            table_info["row_count"] = cursor.fetchone()[0]
            
            schema["tables"][table_name] = table_info
        
        return schema
    
    def get_tables(self) -> List[str]:
        """Get list of table names."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        return [row[0] for row in cursor.fetchall()]
    
    def execute(self, sql: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute SQL and return results.
        
        Args:
            sql: The SQL query to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dict with 'success', 'data' or 'error', and 'row_count'
        """
        # Safety checks - prevent destructive operations
        sql_upper = sql.upper()
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']
        write_keywords = ['INSERT', 'UPDATE', 'CREATE']
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return {
                    'success': False,
                    'error': f"Destructive operation '{keyword}' not allowed in sandbox mode.",
                    'row_count': 0
                }
        
        # Block writes on benchmark databases
        if self.is_benchmark_db:
            for keyword in write_keywords:
                if keyword in sql_upper:
                    return {
                        'success': False,
                        'error': f"Write operation '{keyword}' not allowed on benchmark databases.",
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
                    'row_count': len(rows),
                    'database': self.db_name
                }
            else:
                self.conn.commit()
                return {
                    'success': True,
                    'data': [],
                    'row_count': cursor.rowcount,
                    'database': self.db_name
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'row_count': 0,
                'database': self.db_name
            }


class BenchmarkExecutor:
    """Executor for BIRD/Spider benchmark databases."""
    
    def __init__(self, data_root: str = None):
        if data_root is None:
            data_root = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        self.data_root = os.path.abspath(data_root)
        self.current_executor: Optional[SQLiteExecutor] = None
        self.current_db_id: Optional[str] = None
    
    def list_databases(self) -> Dict[str, List[str]]:
        """List all available benchmark databases."""
        databases = {'bird': [], 'spider': [], 'custom': []}
        
        for dataset in databases.keys():
            dataset_path = os.path.join(self.data_root, dataset)
            if os.path.exists(dataset_path):
                for root, dirs, files in os.walk(dataset_path):
                    for file in files:
                        if file.endswith('.sqlite') or file.endswith('.db'):
                            db_name = os.path.splitext(file)[0]
                            databases[dataset].append(db_name)
        
        return databases
    
    def load_database(self, db_id: str, dataset: str = 'bird') -> bool:
        """Load a specific benchmark database."""
        # Search for the database
        dataset_path = os.path.join(self.data_root, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            return False
        
        # Find the database file
        db_path = None
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.startswith(db_id) and (file.endswith('.sqlite') or file.endswith('.db')):
                    db_path = os.path.join(root, file)
                    break
            if db_path:
                break
        
        if db_path is None:
            print(f"Database not found: {db_id} in {dataset}")
            return False
        
        self.current_executor = SQLiteExecutor(db_path)
        self.current_db_id = db_id
        print(f"Loaded database: {db_id} from {dataset}")
        return True
    
    def execute(self, sql: str, timeout: int = 10) -> Dict[str, Any]:
        """Execute SQL on the current database."""
        if self.current_executor is None:
            return {
                'success': False,
                'error': 'No database loaded. Call load_database() first.',
                'row_count': 0
            }
        
        return self.current_executor.execute(sql, timeout)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema of current database."""
        if self.current_executor is None:
            return {}
        return self.current_executor.get_schema()


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


# Singleton instances
_executor: Optional[SQLiteExecutor] = None
_benchmark_executor: Optional[BenchmarkExecutor] = None
_current_mode: str = 'demo'  # 'demo' or 'benchmark'


def get_executor() -> SQLiteExecutor:
    """Get or create the global executor instance."""
    global _executor
    if _executor is None:
        if DOCKER_AVAILABLE:
            print("Docker available, but using SQLite for demo simplicity.")
        _executor = SQLiteExecutor()
    return _executor


def get_benchmark_executor() -> BenchmarkExecutor:
    """Get or create the benchmark executor."""
    global _benchmark_executor
    if _benchmark_executor is None:
        _benchmark_executor = BenchmarkExecutor()
    return _benchmark_executor


def set_execution_mode(mode: str, db_id: str = None, dataset: str = 'bird') -> bool:
    """
    Switch between demo and benchmark mode.
    
    Args:
        mode: 'demo' for in-memory demo database, 'benchmark' for BIRD/Spider
        db_id: Database ID to load (required for benchmark mode)
        dataset: 'bird' or 'spider'
        
    Returns:
        True if mode switch successful
    """
    global _current_mode, _executor
    
    if mode == 'demo':
        _current_mode = 'demo'
        _executor = SQLiteExecutor()
        return True
    
    elif mode == 'benchmark':
        if db_id is None:
            print("db_id required for benchmark mode")
            return False
        
        benchmark = get_benchmark_executor()
        if benchmark.load_database(db_id, dataset):
            _current_mode = 'benchmark'
            _executor = benchmark.current_executor
            return True
        return False
    
    return False


def get_execution_mode() -> Dict[str, Any]:
    """Get current execution mode info."""
    global _current_mode, _executor
    return {
        'mode': _current_mode,
        'database': _executor.db_name if _executor else None,
        'is_benchmark': _executor.is_benchmark_db if _executor else False
    }


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


def get_current_schema() -> Dict[str, Any]:
    """Get schema of current database."""
    executor = get_executor()
    return executor.get_schema()
