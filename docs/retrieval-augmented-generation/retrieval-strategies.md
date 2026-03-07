# Retrieval Strategies

## Table of Contents

- [Introduction](#introduction)
- [Dense Retrieval](#dense-retrieval)
- [Sparse Retrieval](#sparse-retrieval)
- [Hybrid Search](#hybrid-search)
- [Query Expansion](#query-expansion)
- [Query Transformation](#query-transformation)
- [Metadata Filtering](#metadata-filtering)
- [Multi-Vector Retrieval](#multi-vector-retrieval)
- [Iterative Retrieval](#iterative-retrieval)
- [Retrieval Optimization](#retrieval-optimization)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Retrieval** is the core of RAG - finding the right information to answer queries. Multiple retrieval strategies exist, each with different strengths.

```
Retrieval Strategies Overview:

Query: "How to install Python?"

DENSE RETRIEVAL (Semantic):
  • Convert query to embedding
  • Find similar document embeddings
  • Returns: Semantically similar docs
  • Good for: Conceptual matches

SPARSE RETRIEVAL (Keyword):
  • Extract keywords from query
  • Match keywords in documents (BM25/TF-IDF)
  • Returns: Keyword-matching docs
  • Good for: Exact term matches

HYBRID:
  • Combine dense + sparse
  • Best of both worlds
  • Returns: Semantic + keyword matches
  • Good for: Production systems
```

**Why multiple strategies?**

- Dense retrieval: Great for semantic similarity, misses exact matches
- Sparse retrieval: Great for exact terms, misses semantic variations
- Hybrid: Combines strengths, covers more cases

This guide covers retrieval methods, when to use each, and how to implement them effectively.

## Dense Retrieval

### Semantic Search with Embeddings

```python
def dense_retrieval_explained():
    """Understanding dense retrieval."""
    
    print("Dense Retrieval (Semantic Search):\n")
    
    print("=" * 60)
    print("\nHow It Works:\n")
    
    print("""
1. INDEXING (Preprocessing):
   Documents → [Embed] → Vector Database
   
   "Python is a programming language" → [0.23, -0.15, ..., 0.42]
   "Install Python from python.org"   → [0.25, -0.10, ..., 0.45]

2. QUERY (Search Time):
   Query → [Embed] → Vector Search → Top-k Results
   
   "How to install Python?" → [0.26, -0.12, ..., 0.43]
                           ↓
                    Cosine similarity
                           ↓
                    Ranked results

3. SIMILARITY METRIC:
   Cosine similarity = dot(query, doc) / (||query|| * ||doc||)
   
   Higher similarity = more relevant
""")
    
    print("=" * 60)
    print("\nCharacteristics:\n")
    
    characteristics = [
        ('Semantic understanding', 'Finds conceptually similar, not just keywords'),
        ('Handles synonyms', '"car" matches "automobile", "vehicle"'),
        ('Typo tolerant', '"machne learning" still matches "machine learning"'),
        ('Cross-lingual', 'Can work across languages with right model'),
        ('No exact match guarantee', 'May miss specific terms'),
        ('Computationally intensive', 'Embedding + vector search'),
    ]
    
    for char, description in characteristics:
        print(f"  • {char}: {description}")
    
    print("\n" + "=" * 60)
    print("\nWhen Dense Retrieval Excels:\n")
    
    examples = """
Query: "quick brown animal"
Sparse: Limited matches (exact words only)
Dense:  Matches "fast fox", "speedy dog", "rapid mammal" ✓

Query: "machne lerning" (typo)
Sparse: No matches
Dense:  Still matches "machine learning" documents ✓

Query: "Paris sights"
Sparse: Matches only "Paris" and "sights"
Dense:  Matches "Eiffel Tower", "Louvre", "Notre Dame" ✓
"""
    
    print(examples)

dense_retrieval_explained()
```

### Implementing Dense Retrieval

```python
def implement_dense_retrieval():
    """Implementing dense retrieval."""
    
    print("\n\nImplementing Dense Retrieval:\n")
    
    code = '''
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class DenseRetriever:
    """Dense retrieval using embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def index(self, documents: List[Dict]):
        """
        Index documents for retrieval.
        
        Args:
            documents: List of dicts with 'text' and optional metadata
        """
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Normalize for cosine similarity (optional but faster)
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        
        print(f"Indexed {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of results with scores
        """
        # Embed query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )
        
        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate cosine similarities (dot product of normalized vectors)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(similarities[idx]),
                'rank': len(results) + 1
            })
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """Search multiple queries efficiently."""
        
        # Embed all queries at once
        query_embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Normalize
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        
        # Calculate similarities for all queries
        # Shape: (num_queries, num_documents)
        similarities = np.dot(query_embeddings, self.embeddings.T)
        
        # Get top-k for each query
        all_results = []
        for i, query_sims in enumerate(similarities):
            top_indices = np.argsort(query_sims)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'document': self.documents[idx],
                    'score': float(query_sims[idx]),
                    'rank': len(results) + 1
                })
            
            all_results.append(results)
        
        return all_results


# Example usage
retriever = DenseRetriever()

# Index documents
documents = [
    {'text': 'Python is a high-level programming language.', 'id': 1},
    {'text': 'Machine learning is a subset of AI.', 'id': 2},
    {'text': 'Install Python from the official website.', 'id': 3},
    {'text': 'Deep learning uses neural networks.', 'id': 4},
    {'text': 'Python is popular for data science.', 'id': 5},
]

retriever.index(documents)

# Search
query = "How to get Python?"
results = retriever.search(query, top_k=3)

print(f"Query: {query}\\n")
for result in results:
    print(f"Rank {result['rank']} (score: {result['score']:.4f}):")
    print(f"  {result['document']['text']}")
    print()

# Output:
# Rank 1 (score: 0.6234):
#   Install Python from the official website.
# Rank 2 (score: 0.4521):
#   Python is a high-level programming language.
# ...
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nOptimizations:\n")
    
    optimizations = [
        ('Normalize vectors', 'Enables dot product instead of cosine (faster)'),
        ('Batch processing', 'Embed multiple queries/docs together'),
        ('GPU acceleration', 'Use model.encode(..., device="cuda")'),
        ('Quantization', 'Reduce embedding precision (int8 vs float32)'),
        ('FAISS/HNSW', 'Use approximate nearest neighbor search'),
    ]
    
    for opt, description in optimizations:
        print(f"  • {opt}: {description}")

implement_dense_retrieval()
```

## Sparse Retrieval

### Keyword-Based Search

```python
def sparse_retrieval_explained():
    """Understanding sparse retrieval."""
    
    print("\n\nSparse Retrieval (Keyword Search):\n")
    
    print("=" * 60)
    print("\nHow It Works:\n")
    
    print("""
1. INDEXING:
   Documents → [Extract keywords] → Inverted Index
   
   Doc 1: "Python is a programming language"
   Index: {
     "python": [Doc 1],
     "programming": [Doc 1],
     "language": [Doc 1]
   }

2. QUERY:
   Query → [Extract keywords] → Lookup Index → Score & Rank
   
   "Python programming" → ["python", "programming"]
                        ↓
                      Doc 1 matches both
                        ↓
                    High score

3. SCORING (BM25):
   TF (Term Frequency): How often term appears in doc
   IDF (Inverse Doc Frequency): How rare term is across all docs
   
   BM25 = IDF * (TF * (k+1)) / (TF + k * (1 - b + b * docLen/avgDocLen))
   
   Higher score = more relevant
""")
    
    print("=" * 60)
    print("\nCharacteristics:\n")
    
    characteristics = [
        ('Exact match', 'Finds documents with specific terms'),
        ('Fast', 'Efficient index lookups'),
        ('Interpretable', 'Clear why doc matched (shared keywords)'),
        ('No semantic understanding', 'Misses synonyms, paraphrases'),
        ('Vocabulary mismatch', 'Query and doc must share keywords'),
        ('Good for names/IDs', 'Exact terms like product codes'),
    ]
    
    for char, description in characteristics:
        print(f"  • {char}: {description}")
    
    print("\n" + "=" * 60)
    print("\nWhen Sparse Retrieval Excels:\n")
    
    examples = """
Query: "GPT-4"
Dense:  May match generic "AI models" docs
Sparse: Matches only docs with "GPT-4" ✓

Query: "Bug #12345"
Dense:  Unlikely to find specific bug number
Sparse: Finds exact bug report ✓

Query: "John Smith"
Dense:  May match other names
Sparse: Finds exact name matches ✓

Query: "SKU-ABC-123"
Dense:  Poor at finding product codes
Sparse: Exact product match ✓
"""
    
    print(examples)

sparse_retrieval_explained()
```

### Implementing Sparse Retrieval

```python
def implement_sparse_retrieval():
    """Implementing sparse retrieval with BM25."""
    
    print("\n\nImplementing Sparse Retrieval:\n")
    
    code = '''
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Dict

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class SparseRetriever:
    """Sparse retrieval using BM25."""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self.stop_words = set(stopwords.words('english'))
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize and preprocess text."""
        # Lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and short tokens
        tokens = [
            token for token in tokens 
            if token.isalnum() and 
            token not in self.stop_words and
            len(token) > 2
        ]
        
        return tokens
    
    def index(self, documents: List[Dict]):
        """
        Index documents for retrieval.
        
        Args:
            documents: List of dicts with 'text' and optional metadata
        """
        self.documents = documents
        
        # Tokenize all documents
        self.tokenized_docs = [
            self.tokenize(doc['text']) 
            for doc in documents
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        print(f"Indexed {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of results with scores
        """
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(scores[idx]),
                'rank': len(results) + 1,
                'matched_terms': self._get_matched_terms(
                    query_tokens, 
                    self.tokenized_docs[idx]
                )
            })
        
        return results
    
    def _get_matched_terms(self, query_tokens: List[str], doc_tokens: List[str]) -> List[str]:
        """Get terms that matched between query and document."""
        return list(set(query_tokens) & set(doc_tokens))


# Example usage
retriever = SparseRetriever()

# Index documents
documents = [
    {'text': 'Python is a high-level programming language.', 'id': 1},
    {'text': 'Machine learning is a subset of AI.', 'id': 2},
    {'text': 'Install Python from the official website.', 'id': 3},
    {'text': 'Deep learning uses neural networks.', 'id': 4},
    {'text': 'Python is popular for data science.', 'id': 5},
]

retriever.index(documents)

# Search
query = "Python programming language"
results = retriever.search(query, top_k=3)

print(f"Query: {query}\\n")
for result in results:
    print(f"Rank {result['rank']} (score: {result['score']:.2f}):")
    print(f"  {result['document']['text']}")
    print(f"  Matched terms: {result['matched_terms']}")
    print()

# Output:
# Rank 1 (score: 1.85):
#   Python is a high-level programming language.
#   Matched terms: ['python', 'programming', 'language']
# Rank 2 (score: 0.93):
#   Python is popular for data science.
#   Matched terms: ['python']
# ...
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nBM25 Variants:\n")
    
    variants = {
        'BM25': 'Standard BM25 (most common)',
        'BM25L': 'Better for long documents',
        'BM25+': 'Improved version with better scoring',
        'BM25-Adpt': 'Adaptive version'
    }
    
    for variant, description in variants.items():
        print(f"  • {variant}: {description}")
    
    print("\n" + "=" * 60)
    print("\nTuning Parameters:\n")
    
    params = [
        ('k1', '1.2-2.0', 'Controls term frequency saturation'),
        ('b', '0.75', 'Controls document length normalization'),
    ]
    
    print(f"{'Parameter':<15} {'Typical':<15} {'Effect'}")
    print("-" * 60)
    for param, typical, effect in params:
        print(f"{param:<15} {typical:<15} {effect}")

implement_sparse_retrieval()
```

## Hybrid Search

### Combining Dense and Sparse

```python
def hybrid_search_explained():
    """Understanding hybrid search."""
    
    print("\n\nHybrid Search:\n")
    
    print("=" * 60)
    print("\nConcept:\n")
    
    print("""
Combine dense (semantic) and sparse (keyword) retrieval:

Query: "Python tutorial"

DENSE RETRIEVAL:
  → Finds: "Python guide", "Programming intro", "Coding basics"
  → Semantic matches ✓

SPARSE RETRIEVAL:
  → Finds: Docs with "Python" and "tutorial"
  → Exact term matches ✓

HYBRID (Combine both):
  → Union of both result sets
  → Rank by combined score
  → Best of both worlds ✓
""")
    
    print("=" * 60)
    print("\nFusion Strategies:\n")
    
    print("""
1. RECIPROCAL RANK FUSION (RRF) [Most Common]
   
   score(doc) = Σ 1/(k + rank_i(doc))
   
   Where rank_i is doc's rank in result set i
   
   Example:
     Dense:  Doc A (rank 1), Doc B (rank 2), Doc C (rank 5)
     Sparse: Doc B (rank 1), Doc A (rank 3), Doc D (rank 2)
     
     RRF scores (k=60):
       Doc A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
       Doc B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325 ← Best
       Doc C: 1/(60+5) + 0       = 0.0154
       Doc D: 0       + 1/(60+2) = 0.0161
     
     Final ranking: Doc B, Doc A, Doc D, Doc C

2. WEIGHTED COMBINATION
   
   score(doc) = α * dense_score + (1-α) * sparse_score
   
   α = weight for dense retrieval (typically 0.5-0.7)

3. LINEAR COMBINATION
   
   score(doc) = w1*dense + w2*sparse + w3*other_features
   
   Learned weights (can train with data)

4. TWO-STAGE
   
   Stage 1: Fast retrieval (sparse) → Top-100
   Stage 2: Rerank with dense → Top-10
""")
    
    print("=" * 60)
    print("\nBenefits:\n")
    
    benefits = [
        'Better coverage', 'Catches both semantic and exact matches',
        'More robust', 'Less affected by query variations',
        'Higher recall', 'Finds more relevant documents',
        'Complementary strengths', 'Dense + sparse cover more cases'
    ]
    
    for i in range(0, len(benefits), 2):
        print(f"  • {benefits[i]}: {benefits[i+1]}")

hybrid_search_explained()
```

### Implementing Hybrid Search

```python
def implement_hybrid_search():
    """Implementing hybrid search with RRF."""
    
    print("\n\nImplementing Hybrid Search:\n")
    
    code = '''
from typing import List, Dict

class HybridRetriever:
    """Hybrid search combining dense and sparse retrieval."""
    
    def __init__(self, dense_retriever, sparse_retriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
    
    def search_rrf(
        self, 
        query: str, 
        top_k: int = 5,
        k: int = 60  # RRF parameter
    ) -> List[Dict]:
        """
        Search using Reciprocal Rank Fusion.
        
        Args:
            query: Search query
            top_k: Final number of results
            k: RRF constant (typically 60)
        
        Returns:
            Ranked results
        """
        # Get results from both retrievers
        dense_results = self.dense.search(query, top_k=top_k*2)
        sparse_results = self.sparse.search(query, top_k=top_k*2)
        
        # Calculate RRF scores
        doc_scores = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result['document']['id']
            score = 1.0 / (k + rank)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += score
                doc_scores[doc_id]['dense_rank'] = rank
            else:
                doc_scores[doc_id] = {
                    'document': result['document'],
                    'score': score,
                    'dense_rank': rank,
                    'sparse_rank': None
                }
        
        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result['document']['id']
            score = 1.0 / (k + rank)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += score
                doc_scores[doc_id]['sparse_rank'] = rank
            else:
                doc_scores[doc_id] = {
                    'document': result['document'],
                    'score': score,
                    'dense_rank': None,
                    'sparse_rank': rank
                }
        
        # Sort by combined score
        ranked_results = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return ranked_results[:top_k]
    
    def search_weighted(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.6  # Weight for dense
    ) -> List[Dict]:
        """
        Search using weighted combination.
        
        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for dense scores (0-1)
        
        Returns:
            Ranked results
        """
        # Get results
        dense_results = self.dense.search(query, top_k=top_k*2)
        sparse_results = self.sparse.search(query, top_k=top_k*2)
        
        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return results
            scores = [r['score'] for r in results]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score
            if score_range == 0:
                return results
            for r in results:
                r['normalized_score'] = (r['score'] - min_score) / score_range
            return results
        
        dense_results = normalize_scores(dense_results)
        sparse_results = normalize_scores(sparse_results)
        
        # Combine scores
        doc_scores = {}
        
        for result in dense_results:
            doc_id = result['document']['id']
            doc_scores[doc_id] = {
                'document': result['document'],
                'dense_score': result['normalized_score'],
                'sparse_score': 0
            }
        
        for result in sparse_results:
            doc_id = result['document']['id']
            if doc_id in doc_scores:
                doc_scores[doc_id]['sparse_score'] = result['normalized_score']
            else:
                doc_scores[doc_id] = {
                    'document': result['document'],
                    'dense_score': 0,
                    'sparse_score': result['normalized_score']
                }
        
        # Calculate weighted scores
        for doc_id in doc_scores:
            doc = doc_scores[doc_id]
            doc['score'] = (
                alpha * doc['dense_score'] + 
                (1 - alpha) * doc['sparse_score']
            )
        
        # Sort by combined score
        ranked_results = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return ranked_results[:top_k]


# Example usage
from sentence_transformers import SentenceTransformer

# Initialize retrievers
dense = DenseRetriever(model_name='all-MiniLM-L6-v2')
sparse = SparseRetriever()

# Index documents
documents = [
    {'text': 'Python is a programming language.', 'id': 1},
    {'text': 'Machine learning tutorial for beginners.', 'id': 2},
    {'text': 'Install Python from python.org.', 'id': 3},
    {'text': 'Deep learning with PyTorch.', 'id': 4},
    {'text': 'Python tutorial for data science.', 'id': 5},
]

dense.index(documents)
sparse.index(documents)

# Create hybrid retriever
hybrid = HybridRetriever(dense, sparse)

# Search with RRF
query = "Python tutorial"
results_rrf = hybrid.search_rrf(query, top_k=3)

print(f"Query: {query}\\n")
print("RRF Results:")
for i, result in enumerate(results_rrf, 1):
    print(f"{i}. {result['document']['text']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Dense rank: {result['dense_rank']}, Sparse rank: {result['sparse_rank']}")
    print()

# Search with weighted combination
results_weighted = hybrid.search_weighted(query, top_k=3, alpha=0.6)

print("\\nWeighted Results (α=0.6):")
for i, result in enumerate(results_weighted, 1):
    print(f"{i}. {result['document']['text']}")
    print(f"   Combined: {result['score']:.4f} (Dense: {result['dense_score']:.4f}, Sparse: {result['sparse_score']:.4f})")
    print()
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nChoosing Alpha (Weighted Combination):\n")
    
    guidelines = """
α (alpha) = weight for dense retrieval

α = 0.0:  Pure sparse (keyword only)
α = 0.3:  Mostly sparse, some semantic
α = 0.5:  Equal weight
α = 0.7:  Mostly dense, some keywords
α = 1.0:  Pure dense (semantic only)

Guidelines:
  • Start with α = 0.6 (slight preference for semantic)
  • Increase α if queries are conceptual
  • Decrease α if queries have important exact terms
  • Tune with evaluation data

Typical values: 0.5-0.7
"""
    
    print(guidelines)

implement_hybrid_search()
```

## Query Expansion

### Expanding Queries for Better Retrieval

```python
def query_expansion_explained():
    """Understanding query expansion."""
    
    print("\n\nQuery Expansion:\n")
    
    print("=" * 60)
    print("\nConcept:\n")
    
    print("""
Problem: Short queries may not match documents well

Original query: "ML models"

Expanded query: "machine learning models algorithms neural networks 
                 deep learning artificial intelligence"

Result: More terms → better chance of matching relevant docs
""")
    
    print("=" * 60)
    print("\nExpansion Techniques:\n")
    
    print("""
1. SYNONYM EXPANSION
   
   Query: "car"
   Expanded: "car automobile vehicle"
   
   Use: WordNet, custom synonym dictionary

2. RELATED TERMS
   
   Query: "Python"
   Expanded: "Python programming language coding"
   
   Use: Word embeddings (word2vec, fastText)

3. LLM-BASED EXPANSION
   
   Query: "Install Python"
   LLM generates: "install Python download setup configure"
   
   Use: GPT-3.5/4, smaller models

4. PSEUDO-RELEVANCE FEEDBACK
   
   1. Do initial search
   2. Take top-k results
   3. Extract key terms from results
   4. Add to query and search again
   
   Iterative improvement

5. MULTI-QUERY
   
   Generate multiple reformulations:
   
   Original: "How to use Python?"
   Variants:
     - "Python tutorial"
     - "Getting started with Python"
     - "Python programming guide"
   
   Search all variants, merge results
""")
    
    print("=" * 60)
    print("\nBenefits:\n")
    
    benefits = [
        ('Better recall', 'Find more relevant documents'),
        ('Handle vocabulary mismatch', 'Query terms ≠ document terms'),
        ('Disambiguate', 'Add context to ambiguous queries'),
        ('Robust to typos', 'Related terms still match'),
    ]
    
    for benefit, description in benefits:
        print(f"  • {benefit}: {description}")

query_expansion_explained()
```

### Implementing Query Expansion

```python
def implement_query_expansion():
    """Implementing query expansion techniques."""
    
    print("\n\nImplementing Query Expansion:\n")
    
    code = '''
from typing import List
import openai

class QueryExpander:
    """Query expansion techniques."""
    
    def expand_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms using WordNet."""
        from nltk.corpus import wordnet
        import nltk
        nltk.download('wordnet', quiet=True)
        
        words = query.lower().split()
        expanded_words = set(words)
        
        for word in words:
            # Get synonyms from WordNet
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    expanded_words.add(synonym)
        
        return ' '.join(expanded_words)
    
    def expand_with_llm(self, query: str, llm_call=None) -> str:
        """Expand query using LLM."""
        
        prompt = f"""Generate related search terms for this query. Include synonyms, related concepts, and alternate phrasings.

Query: {query}

Related terms (comma-separated):"""
        
        if llm_call:
            related_terms = llm_call(prompt)
            return query + " " + related_terms
        else:
            # Placeholder
            return query
    
    def multi_query(self, query: str, llm_call=None) -> List[str]:
        """Generate multiple query reformulations."""
        
        prompt = f"""Generate 3 different ways to search for this information:

Original query: {query}

Alternative queries (one per line):
1."""
        
        if llm_call:
            alternatives = llm_call(prompt)
            queries = [query] + alternatives.strip().split('\\n')
            return queries
        else:
            return [query]
    
    def pseudo_relevance_feedback(
        self,
        query: str,
        retriever,
        top_k: int = 3,
        expansion_terms: int = 5
    ) -> str:
        """Expand query using top results."""
        from collections import Counter
        import nltk
        from nltk.corpus import stopwords
        
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
        # Initial search
        results = retriever.search(query, top_k=top_k)
        
        # Extract terms from top results
        all_terms = []
        for result in results:
            text = result['document']['text'].lower()
            terms = [
                word for word in text.split() 
                if word.isalnum() and 
                word not in stop_words and
                len(word) > 2
            ]
            all_terms.extend(terms)
        
        # Get most common terms (excluding query terms)
        query_terms = set(query.lower().split())
        term_counts = Counter(all_terms)
        
        expansion = [
            term for term, count in term_counts.most_common(expansion_terms)
            if term not in query_terms
        ]
        
        # Expanded query
        expanded_query = query + " " + " ".join(expansion)
        
        return expanded_query


# Example usage
expander = QueryExpander()

# 1. Synonym expansion
query = "car insurance"
expanded = expander.expand_with_synonyms(query)
print(f"Original: {query}")
print(f"Expanded: {expanded}")
print()

# Output:
# Original: car insurance
# Expanded: car insurance automobile vehicle coverage policy


# 2. LLM-based expansion
def dummy_llm(prompt):
    # Simulated LLM response
    return "automotive, vehicle coverage, auto policy"

query = "car insurance"
expanded = expander.expand_with_llm(query, dummy_llm)
print(f"LLM Expanded: {expanded}")
print()

# Output:
# LLM Expanded: car insurance automotive, vehicle coverage, auto policy


# 3. Multi-query
queries = expander.multi_query("Python tutorial", dummy_llm)
print("Multi-query variants:")
for i, q in enumerate(queries, 1):
    print(f"  {i}. {q}")
print()


# 4. Search with expansion
def search_with_expansion(query: str, retriever, expander):
    """Search using query expansion."""
    
    # Expand query
    expanded_query = expander.expand_with_synonyms(query)
    
    # Search with expanded query
    results = retriever.search(expanded_query, top_k=5)
    
    return results

# Or: Multi-query approach
def search_multi_query(query: str, retriever, expander):
    """Search with multiple query variants."""
    
    # Generate variants
    queries = expander.multi_query(query)
    
    # Search each variant
    all_results = []
    for q in queries:
        results = retriever.search(q, top_k=5)
        all_results.extend(results)
    
    # Deduplicate and rerank
    seen = set()
    unique_results = []
    for result in all_results:
        doc_id = result['document']['id']
        if doc_id not in seen:
            seen.add(doc_id)
            unique_results.append(result)
    
    return unique_results[:5]
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nWhen to Use Query Expansion:\n")
    
    scenarios = [
        ('Short queries', 'Expand to add context'),
        ('Domain-specific', 'Add domain terminology'),
        ('Low recall', 'Find more relevant docs'),
        ('Vocabulary mismatch', 'Bridge query-doc gap'),
    ]
    
    for scenario, recommendation in scenarios:
        print(f"  • {scenario}: {recommendation}")
    
    print("\n" + "=" * 60)
    print("\nCautions:\n")
    
    cautions = [
        'Query drift: Expansion may change intent',
        'Lower precision: More noise from extra terms',
        'Latency: More processing time',
        'Over-expansion: Too many terms dilute original query'
    ]
    
    for caution in cautions:
        print(f"  ⚠ {caution}")

implement_query_expansion()
```

## Query Transformation

### Rewriting Queries

```python
def query_transformation():
    """Query transformation and rewriting."""
    
    print("\n\nQuery Transformation:\n")
    
    print("=" * 60)
    print("\nConcept:\n")
    
    print("""
Transform complex/ambiguous queries into better forms:

User query: "What's that thing for making coffee?"
Transformed: "coffee maker brewing machine"

User query: "How does X work?" (multi-step)
Decomposed:
  1. "What is X?"
  2. "How X functions"
  3. "X mechanism explanation"
""")
    
    print("=" * 60)
    print("\nTransformation Types:\n")
    
    print("""
1. QUERY REWRITING
   
   Complex → Simple
   
   "What are the best practices for optimizing database performance?"
   →  "database optimization best practices"

2. QUERY DECOMPOSITION
   
   Complex → Multiple sub-queries
   
   "Compare Python and Java for web development"
   →  1. "Python web development features"
       2. "Java web development features"
       3. "Python vs Java comparison"

3. STEP-BACK PROMPTING
   
   Specific → General (then search both)
   
   "How does BERT handle long sequences?"
   →  General: "How do transformer models work?"
       Specific: "BERT long sequence handling"

4. HyDE (Hypothetical Document Embeddings)
   
   Query → Hypothetical answer
   
   "How to install Python?"
   →  "To install Python, download from python.org..."
   
   Search using hypothetical answer embedding
""")
    
    code = '''
class QueryTransformer:
    """Transform queries for better retrieval."""
    
    def rewrite_query(self, query: str, llm_call) -> str:
        """Rewrite query to be more retrieval-friendly."""
        
        prompt = f"""Rewrite this query to be optimized for search. Make it concise and include key terms.

User query: {query}

Optimized search query:"""
        
        return llm_call(prompt)
    
    def decompose_query(self, query: str, llm_call) -> List[str]:
        """Decompose complex query into sub-queries."""
        
        prompt = f"""Break down this complex query into 2-3 simpler sub-queries that can be searched independently.

Complex query: {query}

Sub-queries (one per line):
1."""
        
        response = llm_call(prompt)
        sub_queries = response.strip().split('\\n')
        
        return [q.split('. ', 1)[-1] for q in sub_queries if q]
    
    def step_back(self, query: str, llm_call) -> tuple[str, str]:
        """Generate general question from specific query."""
        
        prompt = f"""Given this specific question, generate a more general question that would provide helpful background.

Specific question: {query}

General question:"""
        
        general_query = llm_call(prompt)
        
        return general_query, query
    
    def hyde(self, query: str, llm_call) -> str:
        """Generate hypothetical document."""
        
        prompt = f"""Write a paragraph that would answer this question.

Question: {query}

Answer:"""
        
        hypothetical_doc = llm_call(prompt)
        
        return hypothetical_doc

# Usage examples
transformer = QueryTransformer()

# 1. Query rewriting
query = "What are some good ways to make my Python code run faster?"
rewritten = transformer.rewrite_query(query, llm_call)
# → "Python code optimization performance improvement"

# 2. Query decomposition
query = "Compare machine learning and deep learning approaches for image classification"
sub_queries = transformer.decompose_query(query, llm_call)
# → ["machine learning image classification",
#     "deep learning image classification",
#     "machine learning vs deep learning comparison"]

# Search each sub-query and combine results

# 3. Step-back
query = "How does GPT-4 handle context windows larger than 8k tokens?"
general, specific = transformer.step_back(query, llm_call)
# general → "How do large language models handle context?"
# specific → (original query)

# Search both and combine

# 4. HyDE
query = "How to fine-tune BERT?"
hypothetical = transformer.hyde(query, llm_call)
# → "To fine-tune BERT, you first load the pre-trained model..."

# Embed hypothetical doc and search with that embedding
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nWhen to Use Each:\n")
    
    recommendations = [
        ('Query rewriting', 'User queries verbose or conversational'),
        ('Query decomposition', 'Complex multi-part questions'),
        ('Step-back', 'Very specific queries needing background'),
        ('HyDE', 'Query style differs significantly from doc style'),
    ]
    
    for technique, use_case in recommendations:
        print(f"  • {technique}: {use_case}")

query_transformation()
```

## Metadata Filtering

### Combining with Filters

```python
def metadata_filtering():
    """Using metadata filters in retrieval."""
    
    print("\n\nMetadata Filtering:\n")
    
    print("=" * 60)
    print("\nConcept:\n")
    
    print("""
Combine vector search with traditional filters:

Query: "Python tutorials"
Filters:
  • date >= "2023-01-01"  (recent only)
  • category = "programming"
  • difficulty = "beginner"

Process:
  1. Apply filters → Reduce search space
  2. Vector search within filtered set
  3. Much faster + more relevant
""")
    
    code = '''
class FilteredRetriever:
    """Retriever with metadata filtering."""
    
    def search_with_filters(
        self,
        query: str,
        filters: Dict,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search with metadata filters.
        
        Args:
            query: Search query
            filters: Dict of metadata filters
            top_k: Number of results
        
        Returns:
            Filtered and ranked results
        """
        # Example filters:
        # {
        #     'date': {'gte': '2023-01-01'},
        #     'category': {'eq': 'programming'},
        #     'tags': {'contains': 'python'},
        #     'author': {'in': ['Alice', 'Bob']}
        # }
        
        # 1. Filter documents by metadata
        filtered_docs = self._apply_filters(self.documents, filters)
        
        # 2. Search within filtered set
        query_embedding = self.model.encode(query)
        
        filtered_embeddings = [
            self.embeddings[doc['index']] 
            for doc in filtered_docs
        ]
        
        # Calculate similarities
        similarities = np.dot(filtered_embeddings, query_embedding)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            {
                'document': filtered_docs[idx],
                'score': float(similarities[idx])
            }
            for idx in top_indices
        ]
        
        return results
    
    def _apply_filters(self, documents: List[Dict], filters: Dict) -> List[Dict]:
        """Apply metadata filters to documents."""
        
        filtered = []
        
        for i, doc in enumerate(documents):
            metadata = doc.get('metadata', {})
            
            # Check each filter
            matches = True
            
            for field, condition in filters.items():
                value = metadata.get(field)
                
                if 'eq' in condition:
                    if value != condition['eq']:
                        matches = False
                        break
                
                elif 'in' in condition:
                    if value not in condition['in']:
                        matches = False
                        break
                
                elif 'gte' in condition:
                    if value < condition['gte']:
                        matches = False
                        break
                
                elif 'lte' in condition:
                    if value > condition['lte']:
                        matches = False
                        break
                
                elif 'contains' in condition:
                    if condition['contains'] not in value:
                        matches = False
                        break
            
            if matches:
                filtered.append({**doc, 'index': i})
        
        return filtered

# Example usage
retriever = FilteredRetriever()

# Search with filters
results = retriever.search_with_filters(
    query="Python tutorial",
    filters={
        'date': {'gte': '2023-01-01'},
        'category': {'eq': 'programming'},
        'difficulty': {'in': ['beginner', 'intermediate']},
        'tags': {'contains': 'python'}
    },
    top_k=5
)

# Common filter patterns:

# 1. Time-based
filters = {'date': {'gte': '2024-01-01', 'lte': '2024-12-31'}}

# 2. Multi-tenancy
filters = {'user_id': {'eq': 'user123'}}

# 3. Access control
filters = {'permissions': {'contains': 'public'}}

# 4. Category + tags
filters = {
    'category': {'in': ['tech', 'science']},
    'tags': {'contains': 'python'}
}

# 5. Complex AND/OR
# (Implement with more sophisticated filter logic)
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nCommon Filter Use Cases:\n")
    
    use_cases = [
        ('Recency', 'Only recent documents (last 30 days)'),
        ('Multi-tenancy', 'User/organization-specific data'),
        ('Access control', 'Permission-based filtering'),
        ('Document type', 'PDFs, code, markdown, etc.'),
        ('Language', 'English, Spanish, etc.'),
        ('Quality', 'High-confidence or verified only'),
    ]
    
    for use_case, example in use_cases:
        print(f"  • {use_case}: {example}")

metadata_filtering()
```

## Multi-Vector Retrieval

### Advanced Retrieval Patterns

```python
def multi_vector_retrieval():
    """Multi-vector and late interaction retrieval."""
    
    print("\n\nMulti-Vector Retrieval:\n")
    
    print("=" * 60)
    print("\nConcept:\n")
    
    print("""
Instead of single vector per document, use multiple:

Standard:
  Document → [Single Vector] → Search

Multi-Vector:
  Document → [Vector 1, Vector 2, Vector 3] → Search
            ↑
     (from different parts/aspects)

Why?
  • Long documents: Multiple vectors capture different sections
  • Multi-aspect: Different vectors for different aspects
  • Better matching: More granular similarity
""")
    
    print("=" * 60)
    print("\nApproaches:\n")
    
    print("""
1. MAX SIMILARITY
   
   Score(query, doc) = max(sim(query, vec_i) for vec_i in doc_vectors)
   
   Use the best matching vector

2. AVERAGE SIMILARITY
   
   Score(query, doc) = mean(sim(query, vec_i) for vec_i in doc_vectors)
   
   Average across all vectors

3. LATE INTERACTION (ColBERT-style)
   
   • Multiple vectors from query AND document
   • Compute all pairwise similarities
   • Aggregate (max-sim per query vector, then sum)
   
   More fine-grained matching

4. HIERARCHICAL
   
   • Document vector (summary)
   • Section vectors (mid-level)
   • Chunk vectors (detailed)
   
   Search at appropriate level
""")
    
    code = '''
class MultiVectorRetriever:
    """Multi-vector retrieval."""
    
    def __init__(self, model):
        self.model = model
        self.documents = []
        self.doc_vectors = []  # List of lists
    
    def index(self, documents: List[Dict]):
        """Index documents with multiple vectors."""
        
        self.documents = documents
        self.doc_vectors = []
        
        for doc in documents:
            # Split document into chunks
            chunks = split_into_chunks(doc['text'], chunk_size=200)
            
            # Embed each chunk
            chunk_embeddings = self.model.encode(chunks)
            
            self.doc_vectors.append(chunk_embeddings)
    
    def search_max_sim(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using max similarity across vectors."""
        
        query_embedding = self.model.encode(query)
        
        scores = []
        for doc_vecs in self.doc_vectors:
            # Calculate similarity with each vector
            sims = np.dot(doc_vecs, query_embedding)
            
            # Take max similarity
            max_sim = np.max(sims)
            scores.append(max_sim)
        
        # Rank documents
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = [
            {'document': self.documents[idx], 'score': scores[idx]}
            for idx in top_indices
        ]
        
        return results
    
    def search_late_interaction(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using late interaction (ColBERT-style)."""
        
        # Split query into tokens/phrases
        query_parts = split_query(query)  # e.g., ["Python", "tutorial"]
        
        # Embed each query part
        query_embeddings = self.model.encode(query_parts)
        
        scores = []
        for doc_vecs in self.doc_vectors:
            # For each query embedding, find max similarity with doc vectors
            max_sims = []
            for query_emb in query_embeddings:
                sims = np.dot(doc_vecs, query_emb)
                max_sims.append(np.max(sims))
            
            # Sum of max similarities
            score = np.sum(max_sims)
            scores.append(score)
        
        # Rank documents
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = [
            {'document': self.documents[idx], 'score': scores[idx]}
            for idx in top_indices
        ]
        
        return results

# Example usage
retriever = MultiVectorRetriever(model)
retriever.index(documents)

results = retriever.search_max_sim("Python programming", top_k=5)

# Benefits:
#  ✓ More fine-grained matching
#  ✓ Better for long documents
#  ✓ Captures multiple aspects

# Cons:
#  ✗ More storage (multiple vectors per doc)
#  ✗ Slower search (more comparisons)
#  ✗ More complex
'''
    
    print(code)

multi_vector_retrieval()
```

## Iterative Retrieval

### Multi-Step Retrieval

```python
def iterative_retrieval():
    """Iterative and multi-step retrieval."""
    
    print("\n\nIterative Retrieval:\n")
    
    print("=" * 60)
    print("\nConcept:\n")
    
    print("""
Multiple rounds of retrieval for complex queries:

Simple Query: "What is Python?"
  → Single retrieval sufficient

Complex Query: "Compare Python and Java for machine learning"
  → Step 1: Retrieve Python ML info
  → Step 2: Retrieve Java ML info
  → Step 3: Retrieve comparisons
  → Combine results
""")
    
    code = '''
class IterativeRetriever:
    """Multi-step iterative retrieval."""
    
    def iterative_search(
        self,
        query: str,
        max_iterations: int = 3,
        llm_call=None
    ) -> List[Dict]:
        """
        Iterative retrieval with refinement.
        
        Args:
            query: Original query
            max_iterations: Maximum retrieval rounds
            llm_call: LLM for generating follow-up queries
        
        Returns:
            Combined results from all iterations
        """
        all_results = []
        current_query = query
        retrieved_content = ""
        
        for iteration in range(max_iterations):
            # Retrieve with current query
            results = self.search(current_query, top_k=3)
            all_results.extend(results)
            
            # Add to retrieved content
            for result in results:
                retrieved_content += result['document']['text'] + "\\n\\n"
            
            # Check if we have enough information
            if llm_call:
                check_prompt = f"""Given the query and retrieved information, do we have enough to answer?

Query: {query}

Retrieved so far:
{retrieved_content[:500]}...

Answer 'YES' if sufficient, or suggest a follow-up query if more info needed:"""
                
                response = llm_call(check_prompt)
                
                if response.strip().upper().startswith('YES'):
                    break
                else:
                    # Extract follow-up query
                    current_query = response.strip()
            else:
                break
        
        # Deduplicate results
        seen = set()
        unique_results = []
        for result in all_results:
            doc_id = result['document']['id']
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(result)
        
        return unique_results

# Example
retriever = IterativeRetriever()

# Complex query requiring multiple steps
query = "How does BERT differ from GPT in architecture and training?"

results = retriever.iterative_search(query, max_iterations=3, llm_call=llm)

# Iteration 1: "BERT architecture training"
# Iteration 2: "GPT architecture training"
# Iteration 3: "BERT GPT comparison differences"
'''
    
    print(code)

iterative_retrieval()
```

## Retrieval Optimization

### Improving Retrieval Quality

```python
def retrieval_optimization():
    """Optimizing retrieval performance and quality."""
    
    print("\n\nRetrieval Optimization:\n")
    
    print("=" * 60)
    print("\nOptimization Strategies:\n")
    
    strategies = """
1. QUERY PREPROCESSING
   
   • Remove stop words (optional)
   • Normalize text (lowercase, etc.)
   • Extract key terms
   • Fix typos

2. EMBEDDING OPTIMIZATION
   
   • Use better embedding model
   • Fine-tune embedding model on your data
   • Normalize vectors for faster comparison
   • Use asymmetric models (different for query/doc)

3. INDEX OPTIMIZATION
   
   • Choose right index type (HNSW, IVF)
   • Tune index parameters
   • Use quantization to reduce size
   • Shard for large scale

4. RETRIEVAL PARAMETERS
   
   • Tune top-k (retrieve more, rerank later)
   • Adjust similarity threshold
   • Balance dense/sparse weights (hybrid)
   • Use metadata filters to reduce search space

5. POST-PROCESSING
   
   • Deduplicate results
   • Diversity-aware ranking
   • Temporal boosting (prefer recent)
   • Quality scoring

6. CACHING
   
   • Cache embeddings (don't recompute)
   • Cache common query results
   • Cache processed documents
"""
    
    print(strategies)
    
    print("\n" + "=" * 60)
    print("\nPerformance Tips:\n")
    
    tips = [
        ('Batch operations', 'Embed/search multiple queries together'),
        ('Async processing', 'Parallel retrieval for multiple queries'),
        ('Approximate search', 'Trade accuracy for speed (ANN)'),
        ('Reduce dimensions', 'Smaller embeddings (384 vs 1536)'),
        ('GPU acceleration', 'Much faster embedding'),
        ('Load balancing', 'Distribute across multiple nodes'),
    ]
    
    for tip, description in tips:
        print(f"  • {tip}: {description}")
    
    print("\n" + "=" * 60)
    print("\nQuality Improvements:\n")
    
    improvements = [
        ('Better chunking', 'Improve what\'s retrievable'),
        ('Add context', 'Enrich chunks with metadata'),
        ('Hybrid search', 'Combine dense + sparse'),
        ('Query expansion', 'Add related terms'),
        ('Reranking', 'Second-pass with better model'),
        ('Evaluate & iterate', 'Measure and improve'),
    ]
    
    for improvement, description in improvements:
        print(f"  • {improvement}: {description}")

retrieval_optimization()
```

## Summary

**Key Concepts**:

1. **Multiple retrieval strategies** - dense (semantic), sparse (keyword), hybrid (both)
2. **Dense retrieval** uses embeddings for semantic similarity (handles synonyms, typos)
3. **Sparse retrieval** uses keyword matching (BM25) for exact terms (good for names, IDs)
4. **Hybrid search** combines both - best of semantic + exact matching
5. **Query expansion** adds related terms to improve recall
6. **Query transformation** rewrites/decomposes queries for better results
7. **Metadata filtering** combines vector search with traditional filters
8. **Multi-vector** and **iterative** retrieval for complex scenarios

**Retrieval Strategies**:

```
DENSE (Semantic):
  • Embedding-based similarity
  • ✓ Semantic understanding, synonyms
  • ✗ May miss exact terms
  • Use: Conceptual queries

SPARSE (Keyword):
  • BM25/TF-IDF matching
  • ✓ Exact term matches
  • ✗ No semantic understanding
  • Use: Specific terms, names, IDs

HYBRID (RECOMMENDED):
  • Combine dense + sparse (RRF)
  • ✓ Best of both worlds
  • ✓ More robust
  • Use: Production systems
```

**Hybrid Search - RRF**:

```python
# Reciprocal Rank Fusion
score(doc) = Σ 1/(k + rank_i(doc))

Where:
  • rank_i = doc's rank in result set i
  • k = constant (typically 60)

Example:
  Dense:  Doc A rank 1, Doc B rank 2
  Sparse: Doc B rank 1, Doc A rank 3
  
  RRF scores:
    Doc B: 1/61 + 1/61 = 0.0328 ← Wins (high in both)
    Doc A: 1/61 + 1/64 = 0.0320
```

**Query Enhancement**:

| Technique | Purpose | Example |
|-----------|---------|---------|
| Expansion | Add related terms | "ML" → "machine learning AI models" |
| Rewriting | Simplify/optimize | "How do I...?" → "steps to..." |
| Decomposition | Break into sub-queries | Complex → 2-3 simpler queries |
| Step-back | Add general context | Specific → General + Specific |
| HyDE | Generate hypothetical answer | Query → Expected document text |

**Query Expansion Methods**:

1. **Synonyms** - WordNet, custom dictionaries
2. **Related terms** - Word embeddings (word2vec)
3. **LLM-based** - Generate with GPT/Claude
4. **Pseudo-relevance feedback** - Extract terms from top results
5. **Multi-query** - Generate multiple reformulations

**Metadata Filtering**:

```python
# Combine vector search with filters
results = search(
    query="Python tutorial",
    filters={
        'date': {'gte': '2023-01-01'},      # Recent only
        'category': {'eq': 'programming'},   # Specific category
        'difficulty': {'in': ['beginner']}, # Difficulty level
        'tags': {'contains': 'python'}      # Has tag
    }
)

Common filters:
  • Time-based: date ranges
  • Multi-tenancy: user/org ID
  • Access control: permissions
  • Document type: PDF, code, etc.
  • Language: English, Spanish, etc.
```

**Advanced Patterns**:

```
Multi-Vector:
  • Multiple embeddings per document
  • Max/average similarity
  • Better for long documents
  ✓ More fine-grained matching
  ✗ More storage/compute

Iterative Retrieval:
  • Multiple rounds of retrieval
  • Refine query based on results
  • Good for complex queries
  ✓ Better coverage
  ✗ Higher latency

Late Interaction (ColBERT):
  • Token-level matching
  • Query tokens × Document tokens
  • Aggregate similarities
  ✓ Fine-grained relevance
  ✗ Computationally expensive
```

**Performance Optimization**:

```
Query Time:
  • Batch multiple queries (2-5x faster)
  • Cache common query results
  • Use ANN (HNSW, IVF) not exact search
  • Normalize vectors (dot product faster)
  • GPU acceleration for embeddings

Index Time:
  • Batch document embeddings
  • Choose right index type
  • Tune parameters (M, ef for HNSW)
  • Use quantization (reduce size)
  • Shard for scale

Quality:
  • Better embedding model
  • Hybrid search (dense + sparse)
  • Query expansion/transformation
  • Metadata filtering
  • Reranking (covered in next section)
```

**Best Practices**:

1. **Start with hybrid search** (dense + sparse via RRF)
2. **Use metadata filters** to reduce search space
3. **Tune top-k**: Retrieve 10-20, rerank to 3-5
4. **Add query expansion** for better recall
5. **Normalize vectors** for faster similarity
6. **Cache embeddings** and common queries
7. **Monitor metrics**: precision, recall, latency
8. **A/B test** different strategies

**Common Pitfalls**:

- Dense only → Misses exact term matches
- Sparse only → Misses semantic variations
- Too small top-k → May miss relevant docs
- No query preprocessing → Noisy queries
- Over-expansion → Query drift
- No metadata → Can't filter effectively

**Recommended Setup**:

```
Simple Use Case:
  → Dense retrieval only (good enough)

Production System:
  → Hybrid search (RRF fusion)
  → Metadata filtering
  → Query expansion (optional)

Complex Queries:
  → Hybrid search
  → Query decomposition
  → Iterative retrieval
  → Reranking (next section)
```

**Latency Budget**:

```
Target: <50ms retrieval time

Breakdown:
  • Query embedding: 5-10ms
  • Vector search: 10-30ms (HNSW/IVF)
  • Sparse search: 5-10ms (BM25)
  • Fusion/ranking: 1-5ms
  • Metadata filtering: 1-5ms

Total: 20-60ms

Optimization opportunities:
  - GPU embedding: 5-10ms → 1-2ms
  - Better index: 30ms → 10ms
  - Caching: Skip embedding entirely
```

## Next Steps

- Deep dive into [Reranking and Fusion](reranking-fusion.md) to improve retrieval quality further
- Study [RAG Evaluation](rag-evaluation.md) to measure retrieval performance
- Review [RAG Architecture](rag-architecture.md) for end-to-end system design
- Learn [Vector Databases](vector-databases.md) for efficient storage and search
- Master [Embedding and Chunking](embedding-chunking.md) for better document preparation
- Apply to production in [Application Patterns](../application_patterns/)
