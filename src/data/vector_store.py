"""
Vector Store for Few-Shot Example Retrieval using ChromaDB.
"""
import json
import os
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. Using fallback similarity search.")


class VectorStore:
    """
    ChromaDB-based vector store for semantic retrieval of few-shot examples.
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
    
    def retrieve(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar few-shot examples for a given query.
        
        Args:
            query: The user's natural language question
            n_results: Number of examples to retrieve
            
        Returns:
            List of example dictionaries with question, sql, and similarity score
        """
        if CHROMADB_AVAILABLE and self.collection is not None:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            retrieved = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    retrieved.append({
                        'question': doc,
                        'sql': metadata.get('sql', ''),
                        'intent': metadata.get('intent', ''),
                        'tables': metadata.get('tables', '').split(','),
                        'similarity': 1 - distance  # Convert distance to similarity
                    })
            
            return retrieved
        else:
            return self._fallback_retrieve(query, n_results)
    
    def _fallback_retrieve(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Simple keyword-based fallback retrieval."""
        query_words = set(query.lower().split())
        
        scored_examples = []
        for ex in self.examples:
            ex_words = set(ex['question'].lower().split())
            overlap = len(query_words & ex_words)
            scored_examples.append((overlap, ex))
        
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
                'count': self.collection.count(),
                'persist_directory': self.persist_directory
            }
        else:
            return {
                'backend': 'fallback',
                'count': len(self.examples),
                'persist_directory': None
            }


# Singleton instance
_vector_store: Optional[VectorStore] = None

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
