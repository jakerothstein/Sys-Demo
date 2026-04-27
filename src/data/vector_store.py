"""
Vector Store for Few-Shot Example Retrieval and Schema Linking using ChromaDB.
Supports Hybrid Search (semantic + keyword) for improved retrieval.
"""
import json
import os
import re
import hashlib
from typing import List, Dict, Any, Optional

def mask_sql_to_skeleton(sql: str) -> str:
    """
    Mask concrete values in SQL to generate a structural skeleton for comparison.
    Replaces string literals (e.g., 'value') and numbers with '<val>'.
    """
    if not sql:
        return ""
    # Strip string literals
    sql_sk = re.sub(r"'[^']*'", "<val>", sql)
    # Strip numeric literals
    sql_sk = re.sub(r"\b\d+(\.\d+)?\b", "<val>", sql_sk)
    return sql_sk.lower()

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. Using fallback similarity search.")


# =============================================================================
# VectorStore - Few-Shot Example Retrieval with Hybrid Search
# =============================================================================

class VectorStore:
    """
    ChromaDB-based vector store for semantic retrieval of few-shot examples.
    Implements Hybrid Search: combines semantic (ChromaDB) + keyword matching.
    Falls back to simple keyword matching if ChromaDB is not available.
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "few_shot_examples"
        self.examples: List[Dict] = []
        
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        else:
            self._init_fallback()
    
    def _init_chromadb(self):
        """Initialize ChromaDB with sentence transformer embeddings."""
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Use default embedding function (all-MiniLM-L6-v2)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Few-shot SQL examples for retrieval"}
        )
    
    def _init_fallback(self):
        """Initialize fallback mode without ChromaDB."""
        self.client = None
        self.collection = None
    
    def load_examples(self, json_path: str):
        """Load few-shot examples from JSON file into the vector store."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.examples = data.get('examples', [])
        
        if CHROMADB_AVAILABLE and self.collection is not None:
            # Check if already populated
            if self.collection.count() >= len(self.examples):
                print(f"Vector store already contains {self.collection.count()} examples.")
                return
            
            # Add examples to ChromaDB
            ids = [ex['id'] for ex in self.examples]
            documents = [ex['question'] for ex in self.examples]
            metadatas = [
                {
                    'sql': ex['sql'],
                    'sql_skeleton': mask_sql_to_skeleton(ex['sql']),
                    'intent': ex.get('intent', ''),
                    'tables': ','.join(ex.get('tables', [])),
                    'difficulty': ex.get('difficulty', 'medium')
                }
                for ex in self.examples
            ]
            
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Loaded {len(self.examples)} examples into ChromaDB.")
        else:
            print(f"Loaded {len(self.examples)} examples into fallback store.")
    
    def retrieve(self, query: str, n_results: int = 3, skeleton: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar few-shot examples using Hybrid Search.
        
        Hybrid Search combines:
        - Semantic search via ChromaDB embeddings
        - Keyword matching for exact term overlap
        - Boost of 0.2 for results with keyword matches
        - Boost of 0.3 for structural similarity (DAIL-Selection)
        
        Args:
            query: The user's natural language question
            n_results: Number of examples to retrieve
            skeleton: Optional draft SQL skeleton for structural matching
            
        Returns:
            List of example dictionaries with question, sql, and hybrid score
        """
        if not CHROMADB_AVAILABLE or self.collection is None:
            return self._fallback_retrieve(query, n_results, skeleton)
        
        # Get more results from semantic search to enable hybrid ranking
        semantic_n = min(n_results * 2, max(self.collection.count(), 1))
        
        results = self.collection.query(
            query_texts=[query],
            n_results=semantic_n
        )
        
        # Build semantic results map
        semantic_results: Dict[str, Dict[str, Any]] = {}
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                semantic_score = 1 - distance  # Convert distance to similarity
                
                semantic_results[doc] = {
                    'question': doc,
                    'sql': metadata.get('sql', ''),
                    'intent': metadata.get('intent', ''),
                    'tables': metadata.get('tables', '').split(',') if metadata.get('tables') else [],
                    'semantic_score': semantic_score,
                    'keyword_boost': 0.0
                }
        
        # Add keyword boost - check for exact word overlaps
        query_words = set(query.lower().split())
        for question, result in semantic_results.items():
            question_words = set(question.lower().split())
            overlap = len(query_words & question_words)
            if overlap > 0:
                # Apply 0.2 boost for keyword matches
                result['keyword_boost'] = 0.2 * (overlap / max(len(query_words), 1))
        
        # Add structural boost - compare target skeleton to example skeletons
        if skeleton:
            target_sk_words = set(skeleton.lower().split())
            for question, result in semantic_results.items():
                ex_sk = mask_sql_to_skeleton(result['sql'])
                ex_sk_words = set(ex_sk.split())
                sk_overlap = len(target_sk_words & ex_sk_words)
                if sk_overlap > 0:
                    result['structural_boost'] = 0.3 * (sk_overlap / max(len(target_sk_words), 1))
                else:
                    result['structural_boost'] = 0.0
        else:
            for result in semantic_results.values():
                result['structural_boost'] = 0.0
        
        # Combine also with keyword-only fallback results (items not in semantic)
        keyword_results = self._fallback_retrieve(query, n_results * 2, skeleton)
        for kr in keyword_results:
            if kr['question'] not in semantic_results:
                # Add keyword-only result with low base semantic score
                semantic_results[kr['question']] = {
                    'question': kr['question'],
                    'sql': kr['sql'],
                    'intent': kr.get('intent', ''),
                    'tables': kr.get('tables', []),
                    'semantic_score': 0.1,  # Low base score for keyword-only
                    'keyword_boost': kr['similarity'] * 0.3,  # Scale keyword similarity
                    'structural_boost': 0.0 # Handled in fallback_retrieve combined similarity but keep 0 here
                }
        
        # Calculate hybrid scores and rank
        hybrid_results = []
        for result in semantic_results.values():
            hybrid_score = result['semantic_score'] + result.get('keyword_boost', 0.0) + result.get('structural_boost', 0.0)
            hybrid_results.append({
                'question': result['question'],
                'sql': result['sql'],
                'intent': result['intent'],
                'tables': result['tables'],
                'similarity': round(hybrid_score, 4)
            })
        
        # Sort by hybrid score descending and return top n
        hybrid_results.sort(key=lambda x: x['similarity'], reverse=True)
        return hybrid_results[:n_results]
    
    def _fallback_retrieve(self, query: str, n_results: int, skeleton: Optional[str] = None) -> List[Dict[str, Any]]:
        """Simple keyword-based fallback retrieval with optional structural matching."""
        query_words = set(query.lower().split())
        target_sk_words = set(skeleton.lower().split()) if skeleton else set()
        
        scored_examples = []
        for ex in self.examples:
            ex_words = set(ex['question'].lower().split())
            overlap = len(query_words & ex_words)
            
            sk_boost = 0.0
            if skeleton:
                ex_sk = mask_sql_to_skeleton(ex['sql'])
                ex_sk_words = set(ex_sk.split())
                sk_overlap = len(target_sk_words & ex_sk_words)
                sk_boost = sk_overlap / max(len(target_sk_words), 1)
            
            # Combine semantic keyword overlap and structural overlap
            total_score = (overlap / max(len(query_words), 1)) + (0.5 * sk_boost)
            scored_examples.append((total_score, ex))
        
        # Sort by overlap score descending
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        return [
            {
                'question': ex['question'],
                'sql': ex['sql'],
                'intent': ex.get('intent', ''),
                'tables': ex.get('tables', []),
                'similarity': score / max(len(query_words), 1)
            }
            for score, ex in scored_examples[:n_results]
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the vector store."""
        if CHROMADB_AVAILABLE and self.collection is not None:
            return {
                'backend': 'chromadb',
                'mode': 'hybrid_search',
                'count': self.collection.count(),
                'persist_directory': self.persist_directory
            }
        else:
            return {
                'backend': 'fallback',
                'mode': 'keyword_only',
                'count': len(self.examples),
                'persist_directory': None
            }


# =============================================================================
# SchemaStore - Schema Linking for Table Retrieval
# =============================================================================

class SchemaStore:
    """
    ChromaDB-based store for schema table retrieval (Schema Linking).
    Embeds table summaries to retrieve only relevant tables for a query.
    Falls back to keyword matching if ChromaDB is not available.
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "schema_tables"
        self.table_summaries: Dict[str, str] = {}  # table_name -> summary text
        self.all_table_names: List[str] = []

        # Track which schemas are already indexed so multi-DB runs (Spider/AmbiQT)
        # don't accidentally reuse a stale schema.
        self._indexed_fingerprints: Dict[str, str] = {}
        
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        else:
            self._init_fallback()
    
    def _init_chromadb(self):
        """Initialize ChromaDB with sentence transformer embeddings."""
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Schema table summaries for linking"}
        )
    
    def _init_fallback(self):
        """Initialize fallback mode without ChromaDB."""
        self.client = None
        self.collection = None

    @staticmethod
    def _get_db_name(schema_dict: Dict[str, Any]) -> str:
        db_name = schema_dict.get('database_name')
        return str(db_name) if db_name else "unknown"

    @staticmethod
    def _schema_fingerprint(schema_dict: Dict[str, Any]) -> str:
        """Stable fingerprint for a schema's table/column structure."""
        tables = schema_dict.get('tables', {}) or {}
        # Only include structural elements; ignore descriptions/sample values.
        simplified = {
            str(t): sorted(list((info or {}).get('columns', {}).keys()))
            for t, info in tables.items()
        }
        payload = {
            "database_name": schema_dict.get("database_name"),
            "tables": simplified,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _meta_id(self, db_name: str) -> str:
        return f"__schema_meta__::{db_name}"

    def _delete_db_docs(self, db_name: str) -> None:
        """Best-effort delete of all docs for a DB (tables + meta)."""
        if not (CHROMADB_AVAILABLE and self.collection is not None):
            return

        # Fast path: chromadb supports metadata-based delete.
        try:
            self.collection.delete(where={"db_name": db_name})
            return
        except Exception:
            pass

        # Fallback: fetch ids then delete by ids.
        try:
            existing = self.collection.get(where={"db_name": db_name})
            ids = existing.get("ids") if isinstance(existing, dict) else None
            if ids:
                self.collection.delete(ids=ids)
                return
        except Exception:
            pass

        # Last resort: clear the entire collection.
        self.clear()
    
    def index_schema(self, schema_dict: Dict[str, Any]) -> None:
        """
        Index schema tables by embedding text summaries.
        
        Creates summaries in format:
        "Table: [name]. Description: [desc]. Columns: [col1, col2...]"
        
        Args:
            schema_dict: Schema catalog dictionary with 'tables' key
        """
        tables = schema_dict.get('tables', {}) or {}
        if not tables:
            print("Warning: No tables found in schema to index.")
            return

        db_name = self._get_db_name(schema_dict)
        fingerprint = self._schema_fingerprint(schema_dict)

        # In-memory fast path: already indexed this exact schema.
        if self._indexed_fingerprints.get(db_name) == fingerprint:
            self.all_table_names = list(tables.keys())
            return

        self.all_table_names = list(tables.keys())
        self.table_summaries.clear()

        # Build summaries for each table
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for table_name, table_info in tables.items():
            description = (table_info or {}).get('description', '')
            columns = list(((table_info or {}).get('columns', {}) or {}).keys())
            columns_str = ', '.join(columns[:10])  # Limit columns in summary
            if len(columns) > 10:
                columns_str += f", ... ({len(columns)} total)"

            # Create summary text for embedding
            summary = f"Table: {table_name}. Description: {description}. Columns: {columns_str}"
            self.table_summaries[str(table_name)] = summary

            ids.append(f"{db_name}::table::{table_name}")
            documents.append(summary)
            metadatas.append({
                'db_name': db_name,
                'doc_type': 'table',
                'table_name': table_name,
                'column_count': len(columns),
                'description': description[:200]  # Truncate for metadata
            })

        if CHROMADB_AVAILABLE and self.collection is not None:
            meta_id = self._meta_id(db_name)

            # If the persisted meta fingerprint matches, skip expensive re-embedding.
            try:
                existing = self.collection.get(ids=[meta_id])
                docs = existing.get("documents") if isinstance(existing, dict) else None
                if docs and docs[0] == fingerprint:
                    self._indexed_fingerprints[db_name] = fingerprint
                    return
            except Exception:
                pass

            # If we get here, either the DB isn't indexed yet or the schema changed.
            self._delete_db_docs(db_name)

            # Upsert table summaries for this DB
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            # Upsert a single meta doc so future runs can detect schema drift.
            self.collection.upsert(
                ids=[meta_id],
                documents=[fingerprint],
                metadatas=[{
                    'db_name': db_name,
                    'doc_type': 'meta',
                    'fingerprint': fingerprint,
                }],
            )

            self._indexed_fingerprints[db_name] = fingerprint
            print(f"Indexed {len(tables)} tables into schema store for db '{db_name}'.")
        else:
            # Fallback mode: summaries are cheap; just mark current schema.
            self._indexed_fingerprints[db_name] = fingerprint
            print(f"Loaded {len(tables)} table summaries into fallback store for db '{db_name}'.")
    
    def clear(self):
        """Clear all indexed tables from the store."""
        self.table_summaries.clear()
        self.all_table_names = []
        self._indexed_fingerprints.clear()

        if CHROMADB_AVAILABLE and self.collection is not None:
            try:
                self.client.delete_collection(self.collection_name)
                # Re-create the collection immediately
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"description": "Schema table summaries for linking"}
                )
            except Exception as e:
                print(f"Error clearing collection: {e}")
    
    def retrieve_relevant_tables(self, query: str, n: int = 5, db_name: Optional[str] = None) -> List[str]:
        """
        Retrieve the most relevant table names for a given user query.
        
        Args:
            query: The user's natural language question
            n: Maximum number of tables to retrieve (default 5)
            
        Returns:
            List of relevant table names
        """
        if not self.all_table_names:
            return []
        
        # Limit n to available tables
        n = min(n, len(self.all_table_names))
        
        if CHROMADB_AVAILABLE and self.collection is not None and self.collection.count() > 0:
            # Chroma expects compound filters via an explicit operator (e.g. $and).
            # See: validate_where() error "Expected where to have exactly one operator".
            if db_name:
                where = {"$and": [{"doc_type": "table"}, {"db_name": db_name}]}
            else:
                where = {"doc_type": "table"}
            results = self.collection.query(
                query_texts=[query],
                n_results=n,
                where=where
            )
            
            table_names = []
            if results and results['metadatas'] and results['metadatas'][0]:
                for metadata in results['metadatas'][0]:
                    table_name = metadata.get('table_name')
                    if table_name:
                        table_names.append(table_name)
            
            return table_names
        else:
            # Fallback: keyword matching on table summaries
            return self._fallback_retrieve(query, n)
    
    def _fallback_retrieve(self, query: str, n: int) -> List[str]:
        """Simple keyword-based fallback retrieval for tables."""
        query_words = set(query.lower().split())
        
        scored_tables = []
        for table_name, summary in self.table_summaries.items():
            summary_words = set(summary.lower().split())
            overlap = len(query_words & summary_words)
            # Also boost if table name itself appears in query
            if table_name.lower() in query.lower():
                overlap += 3
            scored_tables.append((overlap, table_name))
        
        # Sort by score descending
        scored_tables.sort(key=lambda x: x[0], reverse=True)
        
        # Return top n table names
        return [table_name for _, table_name in scored_tables[:n]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the schema store."""
        if CHROMADB_AVAILABLE and self.collection is not None:
            return {
                'backend': 'chromadb',
                'indexed_tables': self.collection.count(),
                'all_tables': len(self.all_table_names),
                'indexed_dbs': len(self._indexed_fingerprints),
                'persist_directory': self.persist_directory
            }
        else:
            return {
                'backend': 'fallback',
                'indexed_tables': len(self.table_summaries),
                'all_tables': len(self.all_table_names),
                'indexed_dbs': len(self._indexed_fingerprints),
                'persist_directory': None
            }


# =============================================================================
# Singleton Instances
# =============================================================================

_vector_store: Optional[VectorStore] = None
_schema_store: Optional[SchemaStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        # Auto-load examples if file exists
        examples_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'few_shot_examples.json')
        if os.path.exists(examples_path):
            _vector_store.load_examples(examples_path)
    return _vector_store


def get_schema_store() -> SchemaStore:
    """Get or create the global schema store instance for schema linking."""
    global _schema_store
    if _schema_store is None:
        _schema_store = SchemaStore()
    return _schema_store
