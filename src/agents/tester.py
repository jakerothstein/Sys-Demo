import sqlite3
import traceback

class TestingAgent:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._setup_mock_db()

    def _setup_mock_db(self):
        """Initializes a mock database for the demo."""
        if self.db_path == ":memory:":
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
            cursor.execute("INSERT OR IGNORE INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')")
            cursor.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT)")
            cursor.execute("INSERT OR IGNORE INTO orders (id, user_id, amount, status) VALUES (101, 1, 50.0, 'completed')")
            self.conn.commit()

    def execute(self, sql: str) -> dict:
        """
        Executes the SQL and returns the result or error.
        """
        try:
            # Basic sanitization for the demo
            if "drop" in sql.lower() or "delete" in sql.lower():
                return {"success": False, "error": "Destructive commands not allowed in demo mode."}

            cursor = self.conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return {"success": True, "data": results}
        except Exception as e:
            return {"success": False, "error": str(e), "trace": traceback.format_exc()}
