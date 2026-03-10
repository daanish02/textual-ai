# Semantic Search

## Table of Contents

- [Introduction](#introduction)
- [Semantic Search Fundamentals](#semantic-search-fundamentals)
- [Embedding-Based Retrieval](#embedding-based-retrieval)
- [Query Understanding and Expansion](#query-understanding-and-expansion)
- [Ranking and Re-ranking](#ranking-and-re-ranking)
- [Faceted Search](#faceted-search)
- [Hybrid Search](#hybrid-search)
- [Evaluation Metrics](#evaluation-metrics)
- [Production Patterns](#production-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Semantic search goes beyond keyword matching to understand the meaning and intent behind queries. Unlike traditional lexical search that matches exact words, semantic search uses embeddings to find conceptually similar content, enabling more intelligent and context-aware information retrieval.

```
Semantic Search Pipeline:

Query: "machine learning tutorial"

Traditional Search (Keyword):
  Matches: documents containing "machine", "learning", "tutorial"
  Misses: "AI course", "neural network guide"

Semantic Search (Meaning):
┌─────────────────────────────────────────────────────────────┐
│  Query → Embedding → Vector DB → Similarity → Results       │
│                                                              │
│  "machine learning tutorial" → [0.2, 0.8, -0.1, ...]       │
│                                    ↓                         │
│              Find similar vectors in database                │
│                                    ↓                         │
│  Results: "AI course", "deep learning intro",               │
│          "neural network fundamentals"                       │
└─────────────────────────────────────────────────────────────┘

Architecture:

User Query
    ↓
Query Processor (expansion, correction)
    ↓
Embedding Model (text → vector)
    ↓
Vector Database (similarity search)
    ↓
Ranker (scoring and ordering)
    ↓
Results (relevant documents)
```

**Key Advantages**:

- **Semantic Understanding**: Understands meaning, not just keywords
- **Multilingual**: Works across languages
- **Synonym Handling**: Finds related concepts automatically
- **Context-Aware**: Considers query intent
- **Robust to Typos**: Embedding-based matching is more forgiving
- **Personalization**: Can incorporate user context

**Use Cases**:

- **Document Search**: Find relevant documents by meaning
- **Product Discovery**: Match products to user needs
- **Question Answering**: Retrieve relevant knowledge
- **Recommendation**: Suggest similar content
- **Code Search**: Find code by functionality
- **Research**: Discover related papers and articles

**Challenges**:

- **Cold Start**: Requires good embeddings
- **Computational Cost**: Vector similarity can be expensive
- **Explainability**: Hard to explain why results match
- **Precision vs Recall**: Balancing broad vs narrow results
- **Ranking**: Ordering results by relevance

This guide covers building production-ready semantic search systems with modern embeddings and retrieval techniques.

## Semantic Search Fundamentals

Understanding the core concepts and components of semantic search.

### Embeddings and Vector Spaces

```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class Document:
    """A document to be searched."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }

class EmbeddingModel:
    """Simple embedding model (in production, use sentence-transformers or OpenAI)."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        # In production, load actual model
        np.random.seed(42)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        # Simplified - in production use actual model
        # This creates a pseudo-embedding based on text characteristics
        words = text.lower().split()
        embedding = np.random.randn(self.dimensions)

        # Add some structure based on text
        if words:
            seed = sum(ord(c) for c in text[:20])
            np.random.seed(seed)
            embedding = np.random.randn(self.dimensions)

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts."""
        return np.array([self.encode(text) for text in texts])

class SimilarityMetrics:
    """Various similarity metrics for embeddings."""

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return np.linalg.norm(vec1 - vec2)

    @staticmethod
    def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute dot product (for normalized vectors)."""
        return np.dot(vec1, vec2)

    @staticmethod
    def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute Manhattan (L1) distance."""
        return np.sum(np.abs(vec1 - vec2))

@dataclass
class SearchResult:
    """A search result with score."""
    document: Document
    score: float
    rank: int
    explanation: Optional[str] = None

class VectorIndex:
    """Simple in-memory vector index for similarity search."""

    def __init__(self, metric: str = 'cosine'):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metric = metric
        self.metrics = SimilarityMetrics()

    def add_document(self, document: Document):
        """Add a document to the index."""
        if document.embedding is None:
            raise ValueError("Document must have an embedding")

        self.documents.append(document)

        # Update embeddings matrix
        if self.embeddings is None:
            self.embeddings = document.embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, document.embedding])

    def add_documents(self, documents: List[Document]):
        """Add multiple documents."""
        for doc in documents:
            self.add_document(doc)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Compute similarities
        scores = self._compute_scores(query_embedding)

        # Apply threshold if specified
        if threshold is not None:
            mask = scores >= threshold
            scores = scores[mask]
            doc_indices = np.where(mask)[0]
        else:
            doc_indices = np.arange(len(scores))

        # Get top k
        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Create results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_idx = doc_indices[idx]
            results.append(SearchResult(
                document=self.documents[doc_idx],
                score=float(scores[idx]),
                rank=rank
            ))

        return results

    def _compute_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute similarity scores."""
        if self.metric == 'cosine':
            # Normalize query
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            # Compute cosine similarity with all documents
            scores = np.dot(self.embeddings, query_norm)
        elif self.metric == 'dot':
            scores = np.dot(self.embeddings, query_embedding)
        elif self.metric == 'euclidean':
            # Convert distance to similarity (lower is better)
            distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
            scores = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return scores

    def save(self, filepath: str):
        """Save index to disk."""
        data = {
            'documents': [doc.to_dict() for doc in self.documents],
            'metric': self.metric
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str, embedding_model: EmbeddingModel):
        """Load index from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.metric = data['metric']
        self.documents = []

        for doc_data in data['documents']:
            embedding = np.array(doc_data['embedding']) if doc_data['embedding'] else None
            if embedding is None:
                embedding = embedding_model.encode(doc_data['content'])

            doc = Document(
                doc_id=doc_data['doc_id'],
                content=doc_data['content'],
                metadata=doc_data['metadata'],
                embedding=embedding
            )
            self.add_document(doc)

# Example usage
def example_fundamentals():
    """Demonstrate semantic search fundamentals."""
    # Create embedding model
    model = EmbeddingModel(dimensions=128)

    # Create sample documents
    documents = [
        Document(
            doc_id="1",
            content="Machine learning is a subset of artificial intelligence",
            metadata={"category": "AI", "author": "Alice"}
        ),
        Document(
            doc_id="2",
            content="Deep learning uses neural networks with many layers",
            metadata={"category": "AI", "author": "Bob"}
        ),
        Document(
            doc_id="3",
            content="Python is a popular programming language",
            metadata={"category": "Programming", "author": "Charlie"}
        ),
        Document(
            doc_id="4",
            content="Natural language processing helps computers understand text",
            metadata={"category": "NLP", "author": "Diana"}
        ),
    ]

    # Generate embeddings
    for doc in documents:
        doc.embedding = model.encode(doc.content)

    # Create vector index
    index = VectorIndex(metric='cosine')
    index.add_documents(documents)

    # Perform search
    query = "artificial intelligence and neural networks"
    query_embedding = model.encode(query)

    print(f"Query: {query}\n")
    print("Results:")
    print("=" * 60)

    results = index.search(query_embedding, top_k=3)
    for result in results:
        print(f"Rank {result.rank}: {result.document.content}")
        print(f"Score: {result.score:.4f}")
        print(f"Category: {result.document.metadata['category']}")
        print()

if __name__ == "__main__":
    example_fundamentals()
```

## Embedding-Based Retrieval

Advanced retrieval techniques using embeddings.

### Embedding Search Engine

```python
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import time

class QueryProcessor:
    """Preprocesses and enhances queries."""

    def __init__(self):
        self.stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }

    def process(self, query: str) -> str:
        """Process a raw query."""
        # Lowercase
        processed = query.lower()

        # Remove extra whitespace
        processed = ' '.join(processed.split())

        return processed

    def remove_stopwords(self, query: str) -> str:
        """Remove stopwords from query."""
        words = query.lower().split()
        filtered = [w for w in words if w not in self.stopwords]
        return ' '.join(filtered)

    def extract_filters(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Extract filters from query (e.g., 'python tutorials category:programming')."""
        import re

        filters = {}
        cleaned_query = query

        # Extract key:value patterns
        pattern = r'(\w+):(["\']?)([^\s"\']+)\2'
        matches = re.findall(pattern, query)

        for key, _, value in matches:
            filters[key] = value
            # Remove from query
            cleaned_query = re.sub(f'{key}:["\']?{value}["\']?', '', cleaned_query)

        cleaned_query = ' '.join(cleaned_query.split())

        return cleaned_query, filters

class EmbeddingCache:
    """Caches embeddings to avoid recomputation."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = defaultdict(int)

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        if text in self.cache:
            self.access_count[text] += 1
            return self.cache[text]
        return None

    def put(self, text: str, embedding: np.ndarray):
        """Cache an embedding."""
        if len(self.cache) >= self.max_size:
            # Remove least accessed
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]

        self.cache[text] = embedding
        self.access_count[text] = 0

class FilterEngine:
    """Filters results based on metadata."""

    @staticmethod
    def apply_filters(
        results: List[SearchResult],
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Apply metadata filters to results."""
        if not filters:
            return results

        filtered = []
        for result in results:
            match = True
            for key, value in filters.items():
                if key not in result.document.metadata:
                    match = False
                    break

                doc_value = result.document.metadata[key]

                # Handle different comparison types
                if isinstance(value, str):
                    if str(doc_value).lower() != value.lower():
                        match = False
                        break
                elif doc_value != value:
                    match = False
                    break

            if match:
                filtered.append(result)

        return filtered

    @staticmethod
    def apply_range_filter(
        results: List[SearchResult],
        field: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> List[SearchResult]:
        """Apply range filter on numerical field."""
        filtered = []
        for result in results:
            if field not in result.document.metadata:
                continue

            value = result.document.metadata[field]

            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue

            filtered.append(result)

        return filtered

class EmbeddingSearchEngine:
    """Complete embedding-based search engine."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        index: VectorIndex,
        use_cache: bool = True
    ):
        self.embedding_model = embedding_model
        self.index = index
        self.query_processor = QueryProcessor()
        self.filter_engine = FilterEngine()

        self.use_cache = use_cache
        if use_cache:
            self.embedding_cache = EmbeddingCache()

        # Search statistics
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0.0
        }

    def add_documents(self, documents: List[Document]):
        """Add documents to the index."""
        # Generate embeddings if needed
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedding_model.encode(doc.content)

        # Add to index
        self.index.add_documents(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        process_query: bool = True
    ) -> List[SearchResult]:
        """Search for documents."""
        start_time = time.time()
        self.stats['total_searches'] += 1

        # Process query
        if process_query:
            processed_query, extracted_filters = self.query_processor.extract_filters(query)
            processed_query = self.query_processor.process(processed_query)

            # Merge filters
            if filters is None:
                filters = extracted_filters
            else:
                filters.update(extracted_filters)
        else:
            processed_query = query

        # Get query embedding
        if self.use_cache:
            query_embedding = self.embedding_cache.get(processed_query)
            if query_embedding is not None:
                self.stats['cache_hits'] += 1
            else:
                query_embedding = self.embedding_model.encode(processed_query)
                self.embedding_cache.put(processed_query, query_embedding)
        else:
            query_embedding = self.embedding_model.encode(processed_query)

        # Search index
        results = self.index.search(query_embedding, top_k=top_k * 2, threshold=threshold)

        # Apply filters
        if filters:
            results = self.filter_engine.apply_filters(results, filters)

        # Re-rank to get top k
        results = results[:top_k]

        # Update ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank

        # Update stats
        search_time = time.time() - start_time
        self.stats['avg_search_time'] = (
            (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time) /
            self.stats['total_searches']
        )

        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[SearchResult]]:
        """Search multiple queries efficiently."""
        # Process all queries
        processed_queries = [self.query_processor.process(q) for q in queries]

        # Get embeddings in batch
        query_embeddings = self.embedding_model.encode_batch(processed_queries)

        # Search for each
        all_results = []
        for query_embedding in query_embeddings:
            results = self.index.search(query_embedding, top_k=top_k)
            all_results.append(results)

        return all_results

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        stats = self.stats.copy()
        if self.stats['total_searches'] > 0:
            stats['cache_hit_rate'] = self.stats['cache_hits'] / self.stats['total_searches']
        else:
            stats['cache_hit_rate'] = 0.0
        return stats

# Example usage
def example_embedding_search():
    """Demonstrate embedding search engine."""
    # Create components
    model = EmbeddingModel(dimensions=128)
    index = VectorIndex(metric='cosine')
    engine = EmbeddingSearchEngine(model, index, use_cache=True)

    # Create documents
    documents = [
        Document(
            doc_id="1",
            content="Introduction to machine learning algorithms",
            metadata={"category": "tutorial", "difficulty": "beginner", "views": 1000}
        ),
        Document(
            doc_id="2",
            content="Advanced deep learning techniques",
            metadata={"category": "tutorial", "difficulty": "advanced", "views": 500}
        ),
        Document(
            doc_id="3",
            content="Python programming basics",
            metadata={"category": "programming", "difficulty": "beginner", "views": 2000}
        ),
        Document(
            doc_id="4",
            content="Natural language processing with transformers",
            metadata={"category": "tutorial", "difficulty": "intermediate", "views": 800}
        ),
        Document(
            doc_id="5",
            content="Building neural networks from scratch",
            metadata={"category": "tutorial", "difficulty": "advanced", "views": 600}
        ),
    ]

    # Add documents
    engine.add_documents(documents)

    # Search with filters
    print("Search 1: 'machine learning' with category filter")
    print("=" * 60)
    results = engine.search(
        "machine learning beginner guide",
        top_k=3,
        filters={"category": "tutorial"}
    )

    for result in results:
        print(f"Rank {result.rank}: {result.document.content}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Difficulty: {result.document.metadata['difficulty']}")
        print()

    # Search with extracted filters
    print("\nSearch 2: Query with embedded filter")
    print("=" * 60)
    results = engine.search(
        "deep learning category:tutorial difficulty:advanced",
        top_k=3
    )

    for result in results:
        print(f"Rank {result.rank}: {result.document.content}")
        print(f"  Score: {result.score:.4f}")
        print()

    # Show stats
    print("\nSearch Statistics:")
    print("=" * 60)
    for key, value in engine.get_stats().items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    example_embedding_search()
```

## Query Understanding and Expansion

Enhance queries to improve retrieval quality.

### Query Expander

```python
from typing import List, Set, Dict, Optional
import re

class QueryExpander:
    """Expands queries with synonyms and related terms."""

    def __init__(self):
        # Simple synonym dictionary (in production, use WordNet or learned embeddings)
        self.synonyms = {
            'machine learning': ['ml', 'artificial intelligence', 'ai', 'predictive modeling'],
            'deep learning': ['neural networks', 'dl', 'deep neural nets'],
            'python': ['py', 'python programming'],
            'tutorial': ['guide', 'how-to', 'walkthrough', 'introduction'],
            'beginner': ['novice', 'starter', 'basic', 'introductory'],
            'advanced': ['expert', 'sophisticated', 'complex'],
        }

    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with synonyms."""
        query_lower = query.lower()
        expansions = [query]  # Always include original

        # Find matching terms
        for term, syns in self.synonyms.items():
            if term in query_lower:
                # Create expanded versions
                for syn in syns[:max_expansions]:
                    expanded = query_lower.replace(term, syn)
                    if expanded not in expansions:
                        expansions.append(expanded)

        return expansions

    def expand_with_scores(self, query: str) -> List[Tuple[str, float]]:
        """Expand query with relevance scores."""
        expansions = self.expand(query)

        # Score expansions (original gets highest score)
        scored = []
        for i, expansion in enumerate(expansions):
            score = 1.0 / (i + 1)  # Decay score for expansions
            scored.append((expansion, score))

        return scored

class SpellingCorrector:
    """Corrects spelling errors in queries."""

    def __init__(self):
        # Common corrections
        self.corrections = {
            'machien': 'machine',
            'lerning': 'learning',
            'tutorail': 'tutorial',
            'pythno': 'python',
            'programing': 'programming',
        }

    def correct(self, query: str) -> str:
        """Correct spelling in query."""
        words = query.lower().split()
        corrected = []

        for word in words:
            corrected.append(self.corrections.get(word, word))

        return ' '.join(corrected)

    def suggest_corrections(self, query: str) -> List[Tuple[str, str]]:
        """Suggest corrections with original words."""
        words = query.lower().split()
        suggestions = []

        for word in words:
            if word in self.corrections:
                suggestions.append((word, self.corrections[word]))

        return suggestions

class IntentDetector:
    """Detects query intent."""

    def __init__(self):
        self.intent_patterns = {
            'question': [
                r'what is',
                r'how to',
                r'why',
                r'when',
                r'where',
                r'who',
                r'\?'
            ],
            'tutorial': [
                r'tutorial',
                r'guide',
                r'how to',
                r'learn',
                r'introduction'
            ],
            'comparison': [
                r'vs',
                r'versus',
                r'compare',
                r'difference between',
                r'better'
            ],
            'definition': [
                r'what is',
                r'define',
                r'meaning of',
                r'definition'
            ]
        }

    def detect(self, query: str) -> List[str]:
        """Detect intents in query."""
        query_lower = query.lower()
        detected_intents = []

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intents.append(intent)
                    break

        return detected_intents if detected_intents else ['general']

class QueryUnderstanding:
    """Complete query understanding system."""

    def __init__(self):
        self.expander = QueryExpander()
        self.corrector = SpellingCorrector()
        self.intent_detector = IntentDetector()

    def understand(self, query: str) -> Dict[str, Any]:
        """Understand and process a query."""
        # Detect intent
        intents = self.intent_detector.detect(query)

        # Check for spelling errors
        corrected_query = self.corrector.correct(query)
        spelling_corrections = self.corrector.suggest_corrections(query)

        # Expand query
        expansions = self.expander.expand(corrected_query, max_expansions=2)

        # Extract key terms
        key_terms = self._extract_key_terms(corrected_query)

        return {
            'original_query': query,
            'corrected_query': corrected_query,
            'intents': intents,
            'spelling_corrections': spelling_corrections,
            'expansions': expansions,
            'key_terms': key_terms
        }

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract important terms from query."""
        # Remove common words
        stopwords = {'a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'to', 'for'}
        words = query.lower().split()
        key_terms = [w for w in words if w not in stopwords and len(w) > 2]
        return key_terms

# Example usage
def example_query_understanding():
    """Demonstrate query understanding."""
    understander = QueryUnderstanding()

    test_queries = [
        "what is machien lerning tutorial",
        "python vs java comparison",
        "how to learn deep learning",
        "define natural language processing"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("=" * 60)

        understanding = understander.understand(query)

        print(f"Corrected: {understanding['corrected_query']}")
        print(f"Intents: {', '.join(understanding['intents'])}")

        if understanding['spelling_corrections']:
            print("Spelling corrections:")
            for original, corrected in understanding['spelling_corrections']:
                print(f"  {original} → {corrected}")

        print(f"Key terms: {', '.join(understanding['key_terms'])}")

        print("Expansions:")
        for expansion in understanding['expansions'][:3]:
            print(f"  - {expansion}")

if __name__ == "__main__":
    example_query_understanding()
```

## Ranking and Re-ranking

Advanced ranking techniques to order results by relevance.

### Rankers

```python
from typing import List, Dict, Any, Callable
import math

class BM25Ranker:
    """BM25 ranking algorithm for lexical matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq: Dict[str, int] = {}
        self.doc_lengths: List[float] = []
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def fit(self, documents: List[Document]):
        """Compute document statistics."""
        self.num_docs = len(documents)

        # Compute document lengths and term frequencies
        term_doc_count: Dict[str, Set[int]] = defaultdict(set)

        for idx, doc in enumerate(documents):
            words = doc.content.lower().split()
            self.doc_lengths.append(len(words))

            # Count terms in document
            for word in set(words):
                term_doc_count[word].add(idx)

        # Compute average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        # Compute document frequencies
        self.doc_freq = {term: len(docs) for term, docs in term_doc_count.items()}

    def score(self, query: str, document: Document, doc_idx: int) -> float:
        """Compute BM25 score for a document."""
        query_terms = query.lower().split()
        doc_terms = document.content.lower().split()
        doc_length = self.doc_lengths[doc_idx] if doc_idx < len(self.doc_lengths) else len(doc_terms)

        score = 0.0
        for term in query_terms:
            if term not in doc_terms:
                continue

            # Term frequency in document
            tf = doc_terms.count(term)

            # Document frequency (number of documents containing term)
            df = self.doc_freq.get(term, 0)

            # IDF score
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 score component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

class SemanticRanker:
    """Re-ranks results using semantic similarity."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        query_embedding: Optional[np.ndarray] = None
    ) -> List[SearchResult]:
        """Re-rank results by semantic similarity."""
        if query_embedding is None:
            query_embedding = self.embedding_model.encode(query)

        # Compute semantic scores
        for result in results:
            if result.document.embedding is not None:
                semantic_score = SimilarityMetrics.cosine_similarity(
                    query_embedding,
                    result.document.embedding
                )
                # Combine with existing score
                result.score = 0.7 * result.score + 0.3 * semantic_score

        # Re-sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank

        return results

class HybridRanker:
    """Combines multiple ranking signals."""

    def __init__(
        self,
        bm25_ranker: BM25Ranker,
        semantic_ranker: SemanticRanker,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        self.bm25_ranker = bm25_ranker
        self.semantic_ranker = semantic_ranker
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def rank(
        self,
        query: str,
        results: List[SearchResult],
        documents: List[Document]
    ) -> List[SearchResult]:
        """Rank using hybrid approach."""
        # Get document indices
        doc_to_idx = {doc.doc_id: idx for idx, doc in enumerate(documents)}

        # Compute combined scores
        for result in results:
            doc_idx = doc_to_idx.get(result.document.doc_id, 0)

            # BM25 score
            bm25_score = self.bm25_ranker.score(query, result.document, doc_idx)

            # Semantic score (already in result.score)
            semantic_score = result.score

            # Combine scores
            combined_score = (
                self.bm25_weight * bm25_score +
                self.semantic_weight * semantic_score
            )

            result.score = combined_score

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank

        return results

class MetadataBooster:
    """Boosts results based on metadata."""

    def __init__(self, boost_config: Dict[str, Dict[str, float]]):
        """
        boost_config: {
            'field_name': {
                'value': boost_multiplier
            }
        }
        """
        self.boost_config = boost_config

    def boost(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply metadata-based boosting."""
        for result in results:
            boost = 1.0

            for field, value_boosts in self.boost_config.items():
                if field in result.document.metadata:
                    field_value = result.document.metadata[field]

                    # Check for exact match
                    if field_value in value_boosts:
                        boost *= value_boosts[field_value]

                    # Check for range-based boosting (for numerical fields)
                    elif isinstance(field_value, (int, float)):
                        for value_key, boost_value in value_boosts.items():
                            if isinstance(value_key, tuple):  # Range: (min, max)
                                min_val, max_val = value_key
                                if min_val <= field_value <= max_val:
                                    boost *= boost_value
                                    break

            result.score *= boost

        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank

        return results

class PersonalizedRanker:
    """Personalizes results based on user context."""

    def __init__(self):
        self.user_preferences: Dict[str, Dict[str, float]] = {}

    def set_user_preferences(self, user_id: str, preferences: Dict[str, float]):
        """Set user preferences (category -> weight)."""
        self.user_preferences[user_id] = preferences

    def personalize(
        self,
        user_id: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Personalize results for user."""
        if user_id not in self.user_preferences:
            return results

        preferences = self.user_preferences[user_id]

        for result in results:
            boost = 1.0

            # Apply preference boosts
            if 'category' in result.document.metadata:
                category = result.document.metadata['category']
                if category in preferences:
                    boost = preferences[category]

            result.score *= boost

        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank

        return results

# Example usage
def example_ranking():
    """Demonstrate ranking techniques."""
    # Create components
    model = EmbeddingModel(dimensions=128)

    # Create documents
    documents = [
        Document(
            doc_id="1",
            content="Python machine learning tutorial for beginners",
            metadata={"category": "tutorial", "difficulty": "beginner", "views": 5000}
        ),
        Document(
            doc_id="2",
            content="Advanced machine learning algorithms in Python",
            metadata={"category": "tutorial", "difficulty": "advanced", "views": 1000}
        ),
        Document(
            doc_id="3",
            content="Machine learning course overview",
            metadata={"category": "course", "difficulty": "intermediate", "views": 3000}
        ),
    ]

    # Generate embeddings
    for doc in documents:
        doc.embedding = model.encode(doc.content)

    # Setup rankers
    bm25 = BM25Ranker()
    bm25.fit(documents)

    semantic_ranker = SemanticRanker(model)

    # Create initial results (from semantic search)
    query = "python machine learning tutorial"
    query_embedding = model.encode(query)

    initial_results = [
        SearchResult(doc, 0.9, 1) for doc in documents
    ]

    # Compute semantic scores
    for result in initial_results:
        result.score = SimilarityMetrics.cosine_similarity(
            query_embedding,
            result.document.embedding
        )

    print("Initial Semantic Results:")
    print("=" * 60)
    for result in sorted(initial_results, key=lambda x: x.score, reverse=True):
        print(f"{result.document.content}")
        print(f"  Score: {result.score:.4f}\n")

    # Apply hybrid ranking
    hybrid_ranker = HybridRanker(bm25, semantic_ranker, bm25_weight=0.4, semantic_weight=0.6)
    hybrid_results = hybrid_ranker.rank(query, initial_results.copy(), documents)

    print("\nHybrid Ranked Results:")
    print("=" * 60)
    for result in hybrid_results:
        print(f"Rank {result.rank}: {result.document.content}")
        print(f"  Score: {result.score:.4f}\n")

    # Apply metadata boosting
    boost_config = {
        'difficulty': {
            'beginner': 1.5,  # Boost beginner content
            'intermediate': 1.0,
            'advanced': 0.8
        },
        'views': {
            (1000, 5000): 1.2  # Boost popular content
        }
    }

    metadata_booster = MetadataBooster(boost_config)
    boosted_results = metadata_booster.boost(hybrid_results.copy())

    print("\nMetadata Boosted Results:")
    print("=" * 60)
    for result in boosted_results:
        print(f"Rank {result.rank}: {result.document.content}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Difficulty: {result.document.metadata['difficulty']}\n")

if __name__ == "__main__":
    example_ranking()
```

## Faceted Search

Faceted search allows users to filter results by multiple dimensions.

### Facet Manager

```python
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter

@dataclass
class Facet:
    """A facet dimension for filtering."""
    name: str
    field: str
    values: List[Tuple[Any, int]]  # (value, count) pairs
    facet_type: str = 'categorical'  # categorical, numerical, date

class FacetExtractor:
    """Extracts facets from documents."""

    def __init__(self, facet_fields: List[str]):
        self.facet_fields = facet_fields

    def extract(self, documents: List[Document]) -> Dict[str, Facet]:
        """Extract facets from documents."""
        facets = {}

        for field in self.facet_fields:
            values = []
            for doc in documents:
                if field in doc.metadata:
                    value = doc.metadata[field]
                    if isinstance(value, list):
                        values.extend(value)
                    else:
                        values.append(value)

            # Count values
            value_counts = Counter(values)

            # Determine facet type
            if values and isinstance(values[0], (int, float)):
                facet_type = 'numerical'
            else:
                facet_type = 'categorical'

            facets[field] = Facet(
                name=field.replace('_', ' ').title(),
                field=field,
                values=value_counts.most_common(),
                facet_type=facet_type
            )

        return facets

    def extract_from_results(
        self,
        results: List[SearchResult]
    ) -> Dict[str, Facet]:
        """Extract facets from search results."""
        documents = [r.document for r in results]
        return self.extract(documents)

class FacetedSearchEngine:
    """Search engine with faceted navigation."""

    def __init__(
        self,
        base_engine: EmbeddingSearchEngine,
        facet_fields: List[str]
    ):
        self.base_engine = base_engine
        self.facet_extractor = FacetExtractor(facet_fields)
        self.facet_fields = facet_fields

    def search(
        self,
        query: str,
        top_k: int = 10,
        selected_facets: Optional[Dict[str, List[Any]]] = None
    ) -> Tuple[List[SearchResult], Dict[str, Facet]]:
        """Search with faceted navigation."""
        # Perform base search
        results = self.base_engine.search(query, top_k=top_k * 3)  # Get more for filtering

        # Apply facet filters
        if selected_facets:
            results = self._apply_facet_filters(results, selected_facets)

        # Extract facets from results
        facets = self.facet_extractor.extract_from_results(results)

        # Trim to top_k
        results = results[:top_k]

        return results, facets

    def _apply_facet_filters(
        self,
        results: List[SearchResult],
        selected_facets: Dict[str, List[Any]]
    ) -> List[SearchResult]:
        """Apply facet filters to results."""
        filtered = []

        for result in results:
            match = True

            for field, selected_values in selected_facets.items():
                if field not in result.document.metadata:
                    match = False
                    break

                doc_value = result.document.metadata[field]

                # Handle list values
                if isinstance(doc_value, list):
                    if not any(v in selected_values for v in doc_value):
                        match = False
                        break
                else:
                    if doc_value not in selected_values:
                        match = False
                        break

            if match:
                filtered.append(result)

        return filtered

    def get_facet_breadcrumbs(
        self,
        selected_facets: Dict[str, List[Any]]
    ) -> List[str]:
        """Get breadcrumb trail of selected facets."""
        breadcrumbs = []

        for field, values in selected_facets.items():
            field_name = field.replace('_', ' ').title()
            values_str = ', '.join(str(v) for v in values)
            breadcrumbs.append(f"{field_name}: {values_str}")

        return breadcrumbs

class NumericalFacetBuilder:
    """Builds numerical range facets."""

    @staticmethod
    def build_range_facets(
        values: List[float],
        num_buckets: int = 5
    ) -> List[Tuple[Tuple[float, float], int]]:
        """Build range buckets for numerical values."""
        if not values:
            return []

        min_val = min(values)
        max_val = max(values)
        range_size = (max_val - min_val) / num_buckets

        # Create buckets
        buckets = defaultdict(int)

        for value in values:
            bucket_idx = min(int((value - min_val) / range_size), num_buckets - 1)
            bucket_start = min_val + bucket_idx * range_size
            bucket_end = bucket_start + range_size
            bucket_key = (bucket_start, bucket_end)
            buckets[bucket_key] += 1

        return sorted(buckets.items())

# Example usage
def example_faceted_search():
    """Demonstrate faceted search."""
    # Create base components
    model = EmbeddingModel(dimensions=128)
    index = VectorIndex()
    base_engine = EmbeddingSearchEngine(model, index)

    # Create documents with rich metadata
    documents = [
        Document(
            doc_id="1",
            content="Python machine learning tutorial",
            metadata={
                "category": "tutorial",
                "difficulty": "beginner",
                "language": "python",
                "duration_minutes": 30,
                "tags": ["ml", "python", "beginner"]
            }
        ),
        Document(
            doc_id="2",
            content="Advanced deep learning with PyTorch",
            metadata={
                "category": "course",
                "difficulty": "advanced",
                "language": "python",
                "duration_minutes": 120,
                "tags": ["dl", "pytorch", "advanced"]
            }
        ),
        Document(
            doc_id="3",
            content="JavaScript web development basics",
            metadata={
                "category": "tutorial",
                "difficulty": "beginner",
                "language": "javascript",
                "duration_minutes": 45,
                "tags": ["web", "javascript", "beginner"]
            }
        ),
        Document(
            doc_id="4",
            content="Natural language processing guide",
            metadata={
                "category": "guide",
                "difficulty": "intermediate",
                "language": "python",
                "duration_minutes": 60,
                "tags": ["nlp", "python", "intermediate"]
            }
        ),
    ]

    # Add documents
    base_engine.add_documents(documents)

    # Create faceted search engine
    faceted_engine = FacetedSearchEngine(
        base_engine,
        facet_fields=['category', 'difficulty', 'language', 'tags']
    )

    # Search without filters
    print("Search: 'python tutorial'")
    print("=" * 60)
    results, facets = faceted_engine.search("python tutorial", top_k=10)

    print(f"\nFound {len(results)} results\n")

    for result in results[:3]:
        print(f"Rank {result.rank}: {result.document.content}")
        print(f"  Category: {result.document.metadata['category']}")
        print(f"  Difficulty: {result.document.metadata['difficulty']}")
        print()

    print("\nAvailable Facets:")
    print("=" * 60)
    for field, facet in facets.items():
        print(f"\n{facet.name}:")
        for value, count in facet.values[:5]:
            print(f"  {value} ({count})")

    # Search with facet filters
    print("\n\nSearch with filters: difficulty='beginner', language='python'")
    print("=" * 60)

    selected_facets = {
        'difficulty': ['beginner'],
        'language': ['python']
    }

    results, facets = faceted_engine.search(
        "python tutorial",
        top_k=10,
        selected_facets=selected_facets
    )

    print(f"\nFound {len(results)} results\n")

    for result in results:
        print(f"Rank {result.rank}: {result.document.content}")
        print(f"  Difficulty: {result.document.metadata['difficulty']}")
        print(f"  Language: {result.document.metadata['language']}")
        print()

    # Show breadcrumbs
    breadcrumbs = faceted_engine.get_facet_breadcrumbs(selected_facets)
    print("Filters applied:")
    for crumb in breadcrumbs:
        print(f"  {crumb}")

if __name__ == "__main__":
    example_faceted_search()
```

## Hybrid Search

Combining semantic and keyword search for best results.

### Hybrid Search System

```python
from typing import List, Dict, Any, Optional, Tuple

class KeywordSearchEngine:
    """Traditional keyword-based search."""

    def __init__(self):
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.documents: Dict[str, Document] = {}

    def index_documents(self, documents: List[Document]):
        """Build inverted index."""
        for doc in documents:
            self.documents[doc.doc_id] = doc

            # Tokenize and index
            words = doc.content.lower().split()
            for word in set(words):
                self.inverted_index[word].add(doc.doc_id)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using keyword matching."""
        query_terms = query.lower().split()

        # Find documents containing query terms
        matching_docs: Dict[str, int] = defaultdict(int)

        for term in query_terms:
            if term in self.inverted_index:
                for doc_id in self.inverted_index[term]:
                    matching_docs[doc_id] += 1

        # Score documents by number of matching terms
        results = []
        for doc_id, match_count in matching_docs.items():
            doc = self.documents[doc_id]
            score = match_count / len(query_terms)  # Normalized score
            results.append(SearchResult(doc, score, 0))

        # Sort and rank
        results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(results[:top_k], 1):
            result.rank = rank

        return results[:top_k]

class HybridSearchEngine:
    """Combines semantic and keyword search."""

    def __init__(
        self,
        semantic_engine: EmbeddingSearchEngine,
        keyword_engine: KeywordSearchEngine,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        self.semantic_engine = semantic_engine
        self.keyword_engine = keyword_engine
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str = 'weighted'  # weighted, cascade, or parallel
    ) -> List[SearchResult]:
        """Hybrid search with multiple strategies."""
        if strategy == 'weighted':
            return self._weighted_search(query, top_k)
        elif strategy == 'cascade':
            return self._cascade_search(query, top_k)
        elif strategy == 'parallel':
            return self._parallel_search(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _weighted_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Combine results with weighted scores."""
        # Get results from both engines
        semantic_results = self.semantic_engine.search(query, top_k=top_k * 2)
        keyword_results = self.keyword_engine.search(query, top_k=top_k * 2)

        # Combine scores
        combined_scores: Dict[str, Tuple[Document, float]] = {}

        # Add semantic scores
        for result in semantic_results:
            doc_id = result.document.doc_id
            score = result.score * self.semantic_weight
            combined_scores[doc_id] = (result.document, score)

        # Add keyword scores
        for result in keyword_results:
            doc_id = result.document.doc_id
            keyword_score = result.score * self.keyword_weight

            if doc_id in combined_scores:
                doc, semantic_score = combined_scores[doc_id]
                combined_scores[doc_id] = (doc, semantic_score + keyword_score)
            else:
                combined_scores[doc_id] = (result.document, keyword_score)

        # Create final results
        results = [
            SearchResult(doc, score, 0)
            for doc, score in combined_scores.values()
        ]

        # Sort and rank
        results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(results[:top_k], 1):
            result.rank = rank

        return results[:top_k]

    def _cascade_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Use keyword search first, then semantic for refinement."""
        # First stage: keyword search
        keyword_results = self.keyword_engine.search(query, top_k=top_k * 3)

        if len(keyword_results) < top_k:
            # Not enough results, use semantic to fill in
            semantic_results = self.semantic_engine.search(query, top_k=top_k)

            # Merge, avoiding duplicates
            seen_ids = {r.document.doc_id for r in keyword_results}
            for result in semantic_results:
                if result.document.doc_id not in seen_ids:
                    keyword_results.append(result)
                    seen_ids.add(result.document.doc_id)

        # Re-rank using semantic similarity
        return self.semantic_engine.search(query, top_k=top_k)

    def _parallel_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Run both searches and interleave results."""
        semantic_results = self.semantic_engine.search(query, top_k=top_k)
        keyword_results = self.keyword_engine.search(query, top_k=top_k)

        # Interleave results
        results = []
        seen_ids = set()

        max_len = max(len(semantic_results), len(keyword_results))
        for i in range(max_len):
            # Add from semantic
            if i < len(semantic_results):
                result = semantic_results[i]
                if result.document.doc_id not in seen_ids:
                    results.append(result)
                    seen_ids.add(result.document.doc_id)

            # Add from keyword
            if i < len(keyword_results):
                result = keyword_results[i]
                if result.document.doc_id not in seen_ids:
                    results.append(result)
                    seen_ids.add(result.document.doc_id)

        # Update ranks
        for rank, result in enumerate(results[:top_k], 1):
            result.rank = rank

        return results[:top_k]

    def explain_result(self, query: str, result: SearchResult) -> Dict[str, Any]:
        """Explain why a result was returned."""
        explanation = {
            'document': result.document.content,
            'overall_score': result.score,
            'components': {}
        }

        # Get semantic score
        query_embedding = self.semantic_engine.embedding_model.encode(query)
        if result.document.embedding is not None:
            semantic_score = SimilarityMetrics.cosine_similarity(
                query_embedding,
                result.document.embedding
            )
            explanation['components']['semantic'] = {
                'score': semantic_score,
                'weight': self.semantic_weight,
                'contribution': semantic_score * self.semantic_weight
            }

        # Get keyword score
        query_terms = set(query.lower().split())
        doc_terms = set(result.document.content.lower().split())
        matching_terms = query_terms & doc_terms
        keyword_score = len(matching_terms) / len(query_terms) if query_terms else 0

        explanation['components']['keyword'] = {
            'score': keyword_score,
            'weight': self.keyword_weight,
            'contribution': keyword_score * self.keyword_weight,
            'matching_terms': list(matching_terms)
        }

        return explanation

# Example usage
def example_hybrid_search():
    """Demonstrate hybrid search."""
    # Create components
    model = EmbeddingModel(dimensions=128)
    vector_index = VectorIndex()
    semantic_engine = EmbeddingSearchEngine(model, vector_index)
    keyword_engine = KeywordSearchEngine()

    # Create documents
    documents = [
        Document(
            doc_id="1",
            content="Python machine learning tutorial for beginners",
            metadata={"type": "tutorial"}
        ),
        Document(
            doc_id="2",
            content="Advanced machine learning with scikit-learn",
            metadata={"type": "guide"}
        ),
        Document(
            doc_id="3",
            content="Python programming basics and fundamentals",
            metadata={"type": "tutorial"}
        ),
        Document(
            doc_id="4",
            content="Deep learning neural networks explained",
            metadata={"type": "article"}
        ),
        Document(
            doc_id="5",
            content="Machine learning algorithms overview",
            metadata={"type": "overview"}
        ),
    ]

    # Index documents
    semantic_engine.add_documents(documents)
    keyword_engine.index_documents(documents)

    # Create hybrid engine
    hybrid_engine = HybridSearchEngine(
        semantic_engine,
        keyword_engine,
        semantic_weight=0.6,
        keyword_weight=0.4
    )

    query = "python machine learning tutorial"

    # Test different strategies
    strategies = ['weighted', 'cascade', 'parallel']

    for strategy in strategies:
        print(f"\nStrategy: {strategy.upper()}")
        print("=" * 60)

        results = hybrid_engine.search(query, top_k=3, strategy=strategy)

        for result in results:
            print(f"\nRank {result.rank}: {result.document.content}")
            print(f"Score: {result.score:.4f}")

            # Get explanation
            explanation = hybrid_engine.explain_result(query, result)
            print("Explanation:")
            for component, details in explanation['components'].items():
                print(f"  {component}: {details['contribution']:.4f} " +
                      f"(score={details['score']:.4f}, weight={details['weight']:.2f})")

if __name__ == "__main__":
    example_hybrid_search()
```

## Evaluation Metrics

Measuring the quality and effectiveness of search systems.

### Search Evaluation

```python
from typing import List, Dict, Set, Optional
import math

class SearchEvaluator:
    """Evaluates search quality with standard metrics."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Precision@K: fraction of top-k results that are relevant."""
        if k == 0 or not retrieved:
            return 0.0

        top_k = retrieved[:k]
        relevant_count = sum(1 for doc_id in top_k if doc_id in relevant)

        return relevant_count / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Recall@K: fraction of relevant docs found in top-k."""
        if not relevant:
            return 0.0

        top_k = retrieved[:k]
        relevant_count = sum(1 for doc_id in top_k if doc_id in relevant)

        return relevant_count / len(relevant)

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """F1 score: harmonic mean of precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
        """Average Precision: average of precision values at relevant positions."""
        if not relevant:
            return 0.0

        precisions = []
        relevant_count = 0

        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                relevant_count += 1
                precision = relevant_count / i
                precisions.append(precision)

        if not precisions:
            return 0.0

        return sum(precisions) / len(relevant)

    @staticmethod
    def mean_average_precision(
        queries_results: List[Tuple[List[str], Set[str]]]
    ) -> float:
        """Mean Average Precision (MAP): average of AP across queries."""
        if not queries_results:
            return 0.0

        aps = [
            SearchEvaluator.average_precision(retrieved, relevant)
            for retrieved, relevant in queries_results
        ]

        return sum(aps) / len(aps)

    @staticmethod
    def mean_reciprocal_rank(
        queries_results: List[Tuple[List[str], Set[str]]]
    ) -> float:
        """Mean Reciprocal Rank (MRR): average of reciprocal ranks of first relevant result."""
        if not queries_results:
            return 0.0

        reciprocal_ranks = []

        for retrieved, relevant in queries_results:
            rank = None
            for i, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    rank = i
                    break

            if rank:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: Dict[str, float], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K."""
        if k == 0:
            return 0.0

        # DCG: sum of (relevance / log2(rank + 1))
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            rel = relevant.get(doc_id, 0.0)
            dcg += rel / math.log2(i + 1)

        # IDCG: DCG of ideal ranking
        ideal_scores = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal_scores))

        if idcg == 0:
            return 0.0

        return dcg / idcg

class SearchBenchmark:
    """Benchmark search systems on test queries."""

    def __init__(self):
        self.test_queries: List[Dict[str, Any]] = []

    def add_test_query(
        self,
        query: str,
        relevant_docs: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ):
        """Add a test query with known relevant documents."""
        self.test_queries.append({
            'query': query,
            'relevant_docs': relevant_docs,
            'relevance_scores': relevance_scores or {doc: 1.0 for doc in relevant_docs}
        })

    def evaluate(
        self,
        search_engine: Any,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """Evaluate search engine on benchmark."""
        results = {
            'precision@k': {},
            'recall@k': {},
            'f1@k': {},
            'ndcg@k': {},
            'map': 0.0,
            'mrr': 0.0,
            'num_queries': len(self.test_queries)
        }

        # Collect results for each query
        all_results = []

        for test in self.test_queries:
            query = test['query']
            relevant = test['relevant_docs']
            relevance_scores = test['relevance_scores']

            # Execute search
            search_results = search_engine.search(query, top_k=max(k_values))
            retrieved = [r.document.doc_id for r in search_results]

            all_results.append((retrieved, relevant))

            # Compute metrics at each k
            for k in k_values:
                if k not in results['precision@k']:
                    results['precision@k'][k] = []
                    results['recall@k'][k] = []
                    results['f1@k'][k] = []
                    results['ndcg@k'][k] = []

                p = SearchEvaluator.precision_at_k(retrieved, relevant, k)
                r = SearchEvaluator.recall_at_k(retrieved, relevant, k)
                f1 = SearchEvaluator.f1_score(p, r)
                ndcg = SearchEvaluator.ndcg_at_k(retrieved, relevance_scores, k)

                results['precision@k'][k].append(p)
                results['recall@k'][k].append(r)
                results['f1@k'][k].append(f1)
                results['ndcg@k'][k].append(ndcg)

        # Average metrics
        for k in k_values:
            results['precision@k'][k] = sum(results['precision@k'][k]) / len(self.test_queries)
            results['recall@k'][k] = sum(results['recall@k'][k]) / len(self.test_queries)
            results['f1@k'][k] = sum(results['f1@k'][k]) / len(self.test_queries)
            results['ndcg@k'][k] = sum(results['ndcg@k'][k]) / len(self.test_queries)

        # Compute MAP and MRR
        results['map'] = SearchEvaluator.mean_average_precision(all_results)
        results['mrr'] = SearchEvaluator.mean_reciprocal_rank(all_results)

        return results

    def compare_engines(
        self,
        engines: Dict[str, Any],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple search engines."""
        comparisons = {}

        for name, engine in engines.items():
            print(f"Evaluating {name}...")
            comparisons[name] = self.evaluate(engine, k_values)

        return comparisons

# Example usage
def example_evaluation():
    """Demonstrate search evaluation."""
    # Setup
    model = EmbeddingModel(dimensions=128)
    index = VectorIndex()
    engine = EmbeddingSearchEngine(model, index)

    # Create documents
    documents = [
        Document("1", "Python machine learning tutorial", {}),
        Document("2", "Machine learning with scikit-learn", {}),
        Document("3", "Deep learning fundamentals", {}),
        Document("4", "Python programming basics", {}),
        Document("5", "Data science introduction", {}),
        Document("6", "Neural networks explained", {}),
        Document("7", "Machine learning algorithms overview", {}),
    ]

    engine.add_documents(documents)

    # Create benchmark
    benchmark = SearchBenchmark()

    # Add test queries with relevance judgments
    benchmark.add_test_query(
        query="machine learning tutorial",
        relevant_docs={"1", "2", "7"},
        relevance_scores={"1": 3.0, "2": 2.0, "7": 1.0}
    )

    benchmark.add_test_query(
        query="deep learning neural networks",
        relevant_docs={"3", "6"},
        relevance_scores={"3": 3.0, "6": 2.0}
    )

    benchmark.add_test_query(
        query="python programming",
        relevant_docs={"1", "4"},
        relevance_scores={"1": 1.0, "4": 3.0}
    )

    # Evaluate
    print("Evaluating Search Engine")
    print("=" * 60)

    results = benchmark.evaluate(engine, k_values=[1, 3, 5])

    print(f"\nNumber of test queries: {results['num_queries']}")
    print(f"MAP: {results['map']:.4f}")
    print(f"MRR: {results['mrr']:.4f}")

    print("\nPrecision@K:")
    for k, score in results['precision@k'].items():
        print(f"  P@{k}: {score:.4f}")

    print("\nRecall@K:")
    for k, score in results['recall@k'].items():
        print(f"  R@{k}: {score:.4f}")

    print("\nNDCG@K:")
    for k, score in results['ndcg@k'].items():
        print(f"  NDCG@{k}: {score:.4f}")

if __name__ == "__main__":
    example_evaluation()
```

## Production Patterns

Best practices for deploying semantic search in production.

### Production Search System

```python
import logging
from typing import Optional, Dict, Any
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionSearchEngine:
    """Production-ready semantic search system."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        index_path: Optional[str] = None,
        cache_size: int = 10000,
        enable_monitoring: bool = True
    ):
        # Core components
        self.embedding_model = embedding_model
        self.index = VectorIndex()
        self.query_processor = QueryProcessor()
        self.query_understander = QueryUnderstanding()

        # Caching
        self.embedding_cache = EmbeddingCache(max_size=cache_size)
        self.query_cache: Dict[str, List[SearchResult]] = {}

        # Monitoring
        self.enable_monitoring = enable_monitoring
        self.metrics = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_latency_ms': 0.0,
            'errors': 0
        }

        # Load index if exists
        self.index_path = index_path
        if index_path and Path(index_path).exists():
            self.load_index(index_path)

    def add_documents_batch(
        self,
        documents: List[Document],
        batch_size: int = 100
    ):
        """Add documents in batches for efficiency."""
        logger.info(f"Adding {len(documents)} documents in batches of {batch_size}")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Generate embeddings in batch
            texts = [doc.content for doc in batch]
            embeddings = self.embedding_model.encode_batch(texts)

            # Assign embeddings
            for doc, embedding in zip(batch, embeddings):
                doc.embedding = embedding

            # Add to index
            self.index.add_documents(batch)

            logger.info(f"Processed batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")

        # Save index
        if self.index_path:
            self.save_index(self.index_path)

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_cache: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Production search with caching and monitoring."""
        start_time = time.time()

        try:
            # Check query cache
            cache_key = f"{query}:{top_k}:{filters}"
            if use_cache and cache_key in self.query_cache:
                self.metrics['cache_hits'] += 1
                results = self.query_cache[cache_key]
                latency_ms = (time.time() - start_time) * 1000

                return {
                    'results': results,
                    'latency_ms': latency_ms,
                    'cached': True,
                    'num_results': len(results)
                }

            # Understand query
            understanding = self.query_understander.understand(query)
            processed_query = understanding['corrected_query']

            # Get or compute query embedding
            query_embedding = self.embedding_cache.get(processed_query)
            if query_embedding is None:
                query_embedding = self.embedding_model.encode(processed_query)
                self.embedding_cache.put(processed_query, query_embedding)

            # Search
            results = self.index.search(query_embedding, top_k=top_k)

            # Apply filters if specified
            if filters:
                results = FilterEngine.apply_filters(results, filters)

            # Cache results
            if use_cache:
                self.query_cache[cache_key] = results

                # Limit cache size
                if len(self.query_cache) > 1000:
                    # Remove oldest entry
                    self.query_cache.pop(next(iter(self.query_cache)))

            # Update metrics
            self.metrics['total_searches'] += 1
            latency_ms = (time.time() - start_time) * 1000
            self.metrics['avg_latency_ms'] = (
                (self.metrics['avg_latency_ms'] * (self.metrics['total_searches'] - 1) + latency_ms) /
                self.metrics['total_searches']
            )

            # Log if slow
            if latency_ms > 1000:
                logger.warning(f"Slow query ({latency_ms:.2f}ms): {query}")

            return {
                'results': results,
                'latency_ms': latency_ms,
                'cached': False,
                'num_results': len(results),
                'understanding': understanding
            }

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Search error: {e}", exc_info=True)

            return {
                'results': [],
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }

    def save_index(self, filepath: str):
        """Save index to disk."""
        logger.info(f"Saving index to {filepath}")
        self.index.save(filepath)

    def load_index(self, filepath: str):
        """Load index from disk."""
        logger.info(f"Loading index from {filepath}")
        self.index.load(filepath, self.embedding_model)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.metrics.copy()

        if self.metrics['total_searches'] > 0:
            metrics['cache_hit_rate'] = self.metrics['cache_hits'] / self.metrics['total_searches']
            metrics['error_rate'] = self.metrics['errors'] / self.metrics['total_searches']
        else:
            metrics['cache_hit_rate'] = 0.0
            metrics['error_rate'] = 0.0

        return metrics

    def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring."""
        return {
            'status': 'healthy',
            'num_documents': len(self.index.documents),
            'cache_size': len(self.query_cache),
            'metrics': self.get_metrics()
        }

# Example usage
def example_production():
    """Demonstrate production search system."""
    # Create production engine
    model = EmbeddingModel(dimensions=128)
    engine = ProductionSearchEngine(
        model,
        index_path="./search_index.json",
        cache_size=1000,
        enable_monitoring=True
    )

    # Add documents
    documents = [
        Document(f"{i}", f"Document {i} about machine learning and AI", {"category": "AI"})
        for i in range(100)
    ]

    engine.add_documents_batch(documents, batch_size=20)

    # Perform searches
    queries = [
        "machine learning tutorial",
        "artificial intelligence basics",
        "machine learning tutorial",  # Duplicate to test cache
    ]

    print("Search Results")
    print("=" * 60)

    for query in queries:
        result = engine.search(query, top_k=3)

        print(f"\nQuery: {query}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
        print(f"Cached: {result.get('cached', False)}")
        print(f"Results: {result['num_results']}")

        for search_result in result['results'][:3]:
            print(f"  {search_result.rank}. {search_result.document.content} (score: {search_result.score:.4f})")

    # Show metrics
    print("\n\nPerformance Metrics")
    print("=" * 60)
    metrics = engine.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Health check
    print("\n\nHealth Check")
    print("=" * 60)
    health = engine.health_check()
    print(f"Status: {health['status']}")
    print(f"Documents: {health['num_documents']}")
    print(f"Cache size: {health['cache_size']}")

if __name__ == "__main__":
    example_production()
```

## Summary

**Semantic Search** enables intelligent information retrieval by understanding meaning rather than just matching keywords:

**Core Concepts**:

- **Embeddings**: Dense vector representations of text
- **Vector Similarity**: Measuring semantic relatedness
- **Semantic Understanding**: Capturing meaning and context
- **Hybrid Approaches**: Combining semantic and keyword search
- **Ranking**: Ordering results by relevance

**Key Components**:

- **Embedding Models**: Transform text to vectors
- **Vector Indexes**: Efficient similarity search
- **Query Processing**: Understanding and expanding queries
- **Rankers**: Scoring and ordering results
- **Filters**: Constraining results by metadata
- **Evaluation**: Measuring search quality

**Advanced Techniques**:

- **Query Expansion**: Synonyms and related terms
- **Re-ranking**: Multiple ranking stages
- **Faceted Search**: Multi-dimensional filtering
- **Hybrid Search**: Combining semantic and lexical
- **Personalization**: User-specific results
- **Metadata Boosting**: Context-aware ranking

**Evaluation Metrics**:

- **Precision@K**: Accuracy of top-K results
- **Recall@K**: Coverage of relevant documents
- **MAP**: Mean average precision
- **MRR**: Mean reciprocal rank
- **NDCG**: Normalized discounted cumulative gain

**Production Considerations**:

- **Caching**: Query and embedding caches
- **Batch Processing**: Efficient indexing
- **Monitoring**: Track performance and errors
- **Index Persistence**: Save and load indexes
- **Latency Optimization**: Fast query processing
- **Scalability**: Handle large document collections
- **Error Handling**: Graceful failure recovery

**Best Practices**:

- Use pre-trained embedding models for better quality
- Implement multiple ranking signals
- Cache frequently accessed embeddings
- Monitor search quality metrics
- A/B test search improvements
- Combine semantic and keyword search
- Provide query suggestions and corrections
- Enable faceted navigation
- Optimize for low latency
- Handle edge cases gracefully

Semantic search transforms information retrieval by understanding the conceptual meaning of queries and documents, enabling more intelligent and user-friendly search experiences.

## Next Steps

- Apply **[Question Answering](question-answering.md)** to retrieved documents
- Use **[Text Classification](text-classification.md)** for better filtering
- Implement **[Named Entity Recognition](named-entity-recognition.md)** for entity-based search
- Explore **[Text Embeddings](../text-embeddings/embedding-models.md)** for better representations
- Study **[Sentence Transformers](../text-embeddings/sentence-transformers.md)** for semantic encoding
- Learn **[Vector Databases](../infrastructure/vector-databases.md)** for scalable search
- Implement **[Retrieval-Augmented Generation](retrieval-augmented-generation.md)** for enhanced QA
- Use **[Conversational AI](conversational-ai.md)** for interactive search
