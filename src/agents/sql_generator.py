import re
from typing import Dict, Any

# Few-shot examples for pattern matching
FEW_SHOT_PATTERNS = [
    {
        "pattern": r"(how many|count).*users",
        "sql": "SELECT COUNT(*) FROM users;"
    },
    {
        "pattern": r"(total|sum).*revenue|(total|sum).*amount.*completed",
        "sql": "SELECT SUM(amount) FROM orders WHERE status = 'completed';"
    },
    {
        "pattern": r"orders.*for.*user.*(\w+)|(\w+)'s orders",
        "sql": "SELECT o.* FROM orders o JOIN users u ON o.user_id = u.id WHERE u.name LIKE '%{name}%';",
        "extract_name": True
    },
    {
        "pattern": r"top (\d+).*orders.*amount",
        "sql": "SELECT * FROM orders ORDER BY amount DESC LIMIT {n};",
        "extract_number": True
    },
    {
        "pattern": r"users.*with.*(gmail|yahoo|email)",
        "sql": "SELECT * FROM users WHERE email LIKE '%{domain}%';",
        "extract_domain": True
    },
    {
        "pattern": r"all orders|list orders|show orders",
        "sql": "SELECT * FROM orders;"
    },
    {
        "pattern": r"all users|list users|show users",
        "sql": "SELECT * FROM users;"
    }
]

class SqlGeneratorAgent:
    def __init__(self, schema_description: str = "users(id INTEGER, name TEXT, email TEXT), orders(id INTEGER, user_id INTEGER, amount REAL, status TEXT)"):
        self.schema_description = schema_description
        self.patterns = FEW_SHOT_PATTERNS

    def generate(self, plan_context: Dict[str, Any]) -> str:
        """
        Generates SQL based on the user's intent and schema.
        Uses few-shot pattern matching for improved accuracy.
        """
        query_text = plan_context.get('disambiguated_query', '').lower()
        detected_intent = plan_context.get('detected_intent', 'select')
        detected_tables = plan_context.get('detected_tables', [])

        # Self-correction: Handle previous errors
        if "previous_error" in plan_context:
            error = plan_context['previous_error']
            if "no such table" in error:
                # Extract the bad table name and replace with a valid one
                if detected_tables:
                    return f"SELECT * FROM {detected_tables[0]};"
                return "SELECT * FROM users; -- Fallback after error"
            elif "syntax error" in error.lower():
                # Try a simpler query
                return "SELECT * FROM users LIMIT 5; -- Simplified after syntax error"

        # Pattern matching from few-shot examples
        for pattern_info in self.patterns:
            match = re.search(pattern_info["pattern"], query_text, re.IGNORECASE)
            if match:
                sql = pattern_info["sql"]
                
                # Handle dynamic substitutions
                if pattern_info.get("extract_name"):
                    name = match.group(1) or match.group(2) or "Unknown"
                    sql = sql.format(name=name.strip())
                elif pattern_info.get("extract_number"):
                    n = match.group(1)
                    sql = sql.format(n=n)
                elif pattern_info.get("extract_domain"):
                    domain = match.group(1) if match.group(1) else "gmail.com"
                    sql = sql.format(domain=domain)
                
                return sql

        # Intent-based fallback
        if detected_intent == "count":
            if "orders" in detected_tables:
                return "SELECT COUNT(*) FROM orders;"
            return "SELECT COUNT(*) FROM users;"
        elif detected_intent == "sum":
            return "SELECT SUM(amount) FROM orders;"
        elif detected_intent == "top":
            return "SELECT * FROM orders ORDER BY amount DESC LIMIT 5;"

        # Ultimate fallback
        if detected_tables:
            return f"SELECT * FROM {detected_tables[0]} LIMIT 10;"
        return "SELECT * FROM users LIMIT 10;"
