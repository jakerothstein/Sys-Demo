import re
from dataclasses import dataclass
from typing import List, Optional

# Define the schema for entity matching
SCHEMA_ENTITIES = {
    "tables": ["users", "orders"],
    "columns": {
        "users": ["id", "name", "email"],
        "orders": ["id", "user_id", "amount", "status"]
    },
    "keywords": {
        "count": ["count", "how many", "number of"],
        "sum": ["total", "sum", "revenue"],
        "filter": ["where", "with", "for", "named"],
        "top": ["top", "best", "highest", "largest"],
        "join": ["for user", "user's", "belonging to"]
    }
}

@dataclass
class Plan:
    original_query: str
    disambiguated_query: str
    confidence: float
    is_ambiguous: bool
    needs_hitl: bool
    reasoning: str
    detected_tables: List[str]
    detected_intent: str

class DisambiguatorAgent:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.schema = SCHEMA_ENTITIES

    def _detect_tables(self, query: str) -> List[str]:
        """Detect which tables are referenced in the query."""
        query_lower = query.lower()
        detected = []
        for table in self.schema["tables"]:
            # Check for table name or related keywords
            if table in query_lower or table.rstrip('s') in query_lower:
                detected.append(table)
        return detected

    def _detect_intent(self, query: str) -> str:
        """Detect the query intent (SELECT, COUNT, SUM, etc.)."""
        query_lower = query.lower()
        for intent, keywords in self.schema["keywords"].items():
            if any(kw in query_lower for kw in keywords):
                return intent
        return "select"

    def _calculate_confidence(self, query: str, tables: List[str], intent: str) -> tuple:
        """Calculate confidence based on schema matching and query clarity."""
        score = 0.5  # Base score
        reasons = []

        # Boost for table detection
        if tables:
            score += 0.2
            reasons.append(f"Detected tables: {tables}")
        else:
            reasons.append("No tables detected in query.")

        # Boost for clear intent
        if intent != "select":
            score += 0.15
            reasons.append(f"Clear intent: {intent}")

        # Penalty for ambiguous markers
        ambiguous_markers = ["maybe", "unsure", "?", "or something", "I think", "probably"]
        if any(marker in query.lower() for marker in ambiguous_markers):
            score -= 0.3
            reasons.append("Query contains uncertainty markers.")

        # Penalty for very short queries
        if len(query.split()) < 3:
            score -= 0.2
            reasons.append("Query is too short.")

        # Ensure score is within bounds
        score = max(0.1, min(1.0, score))
        return score, " | ".join(reasons)

    def analyze(self, user_query: str) -> Plan:
        """
        Analyzes the user query to determine intent, schema matches, and confidence.
        """
        detected_tables = self._detect_tables(user_query)
        detected_intent = self._detect_intent(user_query)
        confidence, reasoning = self._calculate_confidence(user_query, detected_tables, detected_intent)

        is_ambiguous = confidence < self.confidence_threshold
        needs_hitl = is_ambiguous

        return Plan(
            original_query=user_query,
            disambiguated_query=user_query,
            confidence=round(confidence, 2),
            is_ambiguous=is_ambiguous,
            needs_hitl=needs_hitl,
            reasoning=reasoning,
            detected_tables=detected_tables,
            detected_intent=detected_intent
        )
