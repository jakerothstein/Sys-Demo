"""
Heuristic extraction of schema references (tables / identifiers) from SQL text.

Used to detect "unanimous structural divergence": several candidate queries that
all execute to the same rows, yet reference meaningfully different tables or
columns (AmbiQT-style ambiguity with materialized synonym copies). Execution
entropy alone is ~0 in that regime, so we need this extra signal.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

# SQLite + common SQL reserved words (uppercase keys for lookup)
_SQL_KEYWORDS: Set[str] = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE",
    "BETWEEN", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET", "AS", "ON",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "NATURAL", "USING",
    "UNION", "ALL", "EXCEPT", "INTERSECT", "WITH", "RECURSIVE", "CASE", "WHEN",
    "THEN", "ELSE", "END", "DISTINCT", "EXISTS", "TRUE", "FALSE", "CAST",
    "COUNT", "SUM", "AVG", "MIN", "MAX", "COALESCE", "ROWID", "OVER", "PARTITION",
    "ASC", "DESC", "FOR", "EACH", "GLOB", "REGEXP", "ESCAPE", "WINDOW", "FILTER",
}


def _strip_literals_for_tokens(sql: str) -> str:
    s = sql.strip()
    s = re.sub(r"--[^\n]*", " ", s)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    s = re.sub(r"'[^']*'", " ", s)
    s = re.sub(r'"[^"]*"', " ", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", " ", s)
    return s


def extract_table_names(sql: str) -> Set[str]:
    """Table names appearing after FROM / JOIN (single-token names only)."""
    s = _strip_literals_for_tokens(sql)
    out: Set[str] = set()
    for m in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w]*)", s, re.IGNORECASE):
        name = m.group(1)
        if name.upper() in _SQL_KEYWORDS:
            continue
        out.add(name.lower())
    return out


def extract_schema_identifiers(sql: str) -> Set[str]:
    """
    Non-keyword bare identifiers (rough superset of table + column names).
    Single-character tokens are dropped (reduces alias noise from `t`, `a`).
    """
    s = _strip_literals_for_tokens(sql)
    words = re.findall(r"\b([A-Za-z_][\w]*)\b", s)
    out: Set[str] = set()
    for w in words:
        if len(w) <= 1:
            continue
        if w.upper() in _SQL_KEYWORDS:
            continue
        out.add(w.lower())
    return out


def detect_unanimous_structural_ambiguity(
    sqls: List[str],
    *,
    result_clusters: List[Dict[str, Any]],
    skeleton_clusters: List[Dict[str, Any]],
    execution_entropy: float,
    semantic_entropy: float,
    n_success: int,
    n_sqls: int,
    thresholds: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Return (True, detail) when candidates are unanimously right on rows but
    structurally disagree "enough" that the question is plausibly ambiguous.

    Requires:
      - all candidates executed successfully
      - a single execution-result cluster (unanimous rows)
      - near-zero execution entropy
      - at least N distinct structural skeletons and high semantic entropy
      - spread in schema references: either >=2 distinct tables or
        >=4 distinct non-keyword identifiers across the union of candidates
    """
    th = thresholds or {}
    sem_min = float(th.get("structural_divergence_semantic_min", 1.45))
    min_skeletons = int(th.get("structural_divergence_min_skeletons", 3))
    min_tables = int(th.get("structural_divergence_min_distinct_tables", 2))
    min_idents = int(th.get("structural_divergence_min_distinct_identifiers", 4))
    exec_eps = float(th.get("structural_divergence_exec_epsilon", 0.01))

    per_tables = [extract_table_names(s) for s in sqls]
    per_idents = [extract_schema_identifiers(s) for s in sqls]
    union_tables: Set[str] = set().union(*per_tables) if per_tables else set()
    union_idents: Set[str] = set().union(*per_idents) if per_idents else set()

    detail: Dict[str, Any] = {
        "union_table_count": len(union_tables),
        "union_tables": sorted(union_tables),
        "union_identifier_count": len(union_idents),
        "per_sql_tables": [sorted(t) for t in per_tables],
        "thresholds_applied": {
            "semantic_min": sem_min,
            "min_skeleton_clusters": min_skeletons,
            "min_distinct_tables": min_tables,
            "min_distinct_identifiers": min_idents,
            "exec_epsilon": exec_eps,
        },
    }

    if n_sqls < min_skeletons:
        detail["reason"] = "not_enough_sql_candidates"
        return False, detail
    if n_success != n_sqls:
        detail["reason"] = "not_all_candidates_succeeded"
        return False, detail
    if len(result_clusters) != 1:
        detail["reason"] = "execution_not_unanimous"
        return False, detail
    if execution_entropy > exec_eps:
        detail["reason"] = "execution_entropy_not_near_zero"
        return False, detail
    if len(skeleton_clusters) < min_skeletons:
        detail["reason"] = "not_enough_skeleton_clusters"
        return False, detail
    if semantic_entropy < sem_min:
        detail["reason"] = "semantic_entropy_below_min"
        return False, detail

    spread_tables = len(union_tables) >= min_tables
    spread_idents = len(union_idents) >= min_idents
    if not (spread_tables or spread_idents):
        detail["reason"] = "schema_reference_spread_too_low"
        return False, detail

    detail["reason"] = "unanimous_structural_divergence"
    return True, detail
