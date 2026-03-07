# Reranking and Fusion

## Table of Contents

- [Introduction](#introduction)
- [Why Reranking Matters](#why-reranking-matters)
- [Cross-Encoder Reranking](#cross-encoder-reranking)
- [Reciprocal Rank Fusion](#reciprocal-rank-fusion)
- [Relevance Scoring Models](#relevance-scoring-models)
- [Multi-Stage Retrieval](#multi-stage-retrieval)
- [Combining Multiple Retrievers](#combining-multiple-retrievers)
- [Diversity and MMR](#diversity-and-mmr)
- [LLM-Based Reranking](#llm-based-reranking)
- [Cost-Benefit Analysis](#cost-benefit-analysis)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Reranking** is a second-pass process that improves retrieval quality by reordering results from initial retrieval.

```
Standard Retrieval:
  Query → [Search] → Top-10 Results → Return to User

With Reranking:
  Query → [Search] → Top-100 Results → [Rerank] → Top-10 → Return to User
         ↑                                ↑
    Fast but rough            Slower but more accurate
```

**Key Insight**: First-stage retrieval casts a wide net (high recall), second-stage reranking selects the best (high precision).

```
Trade-off Diagram:

Stage 1 (Retrieval):
  Speed:    ██████████ (Fast)
  Quality:  ████       (Good)
  Coverage: ██████████ (Wide)

Stage 2 (Reranking):
  Speed:    ████       (Slower)
  Quality:  ██████████ (Excellent)
  Coverage: ████       (Narrow)

Combined:
  Speed:    ███████    (Acceptable)
  Quality:  ██████████ (Excellent)
  Coverage: ██████████ (Wide)
```

**Fusion** combines results from multiple retrievers (dense, sparse, etc.) into a unified ranking.

This guide covers reranking techniques, fusion strategies, and how to implement effective multi-stage retrieval.

## Why Reranking Matters

### The Retrieval Quality Gap

```python
def reranking_motivation():
    """Understanding why reranking is valuable."""
    
    print("Why Reranking Matters:\n")
    
    print("=" * 60)
    print("\nThe Quality Gap:\n")
    
    print("""
Bi-Encoder (Standard Retrieval):
  • Encodes query and document separately
  • Compare embeddings with cosine similarity
  • FAST but limited interaction
  
  Query:    [Embed] → Vector_Q
  Document: [Embed] → Vector_D
  
  Score = cosine(Vector_Q, Vector_D)

Cross-Encoder (Reranking):
  • Encodes query and document TOGETHER
  • Model sees full interaction
  • SLOW but much more accurate
  
  [Query, Document] → [Encoder] → Relevance Score
  
  Model learns: "Does this doc answer this query?"

Why Cross-Encoder is Better:
  ✓ Sees full context
  ✓ Token-level interactions (attention)
  ✓ Can reason about relevance
  ✓ Captures nuanced relationships

Why Not Use Cross-Encoder for Everything?
  ✗ Too slow for initial retrieval
  ✗ Must encode every query-doc pair
  ✗ Can't precompute embeddings
  
  Example: 1M documents × 1 query = 1M cross-encoder calls
""")
    
    print("=" * 60)
    print("\nTypical Quality Improvement:\n")
    
    improvements = """
Metric          | Retrieval Only | + Reranking | Improvement
----------------|----------------|-------------|-------------
Precision@5     | 0.65          | 0.85        | +31%
NDCG@10         | 0.72          | 0.88        | +22%
MRR             | 0.68          | 0.84        | +24%
User satisfaction| 72%           | 89%         | +24%

Observed improvements in production systems:
  • E-commerce search: 15-25% better relevance
  • Question answering: 20-30% better accuracy
  • Document search: 25-35% better precision
"""
    
    print(improvements)
    
    print("\n" + "=" * 60)
    print("\nWhen Reranking Helps Most:\n")
    
    scenarios = [
        ('Complex queries', 'Multi-part or nuanced questions'),
        ('Subtle differences', 'Documents look similar but differ in key ways'),
        ('Context matters', 'Word order, negations, qualifiers'),
        ('High-stakes applications', 'Medical, legal, financial domains'),
        ('Quality over speed', 'When accuracy is more important than latency'),
    ]
    
    for scenario, description in scenarios:
        print(f"  • {scenario}: {description}")

reranking_motivation()
```

### Bi-Encoder vs Cross-Encoder

```python
def bi_vs_cross_encoder():
    """Comparing bi-encoder and cross-encoder architectures."""
    
    print("\n\nBi-Encoder vs Cross-Encoder:\n")
    
    print("=" * 60)
    print("\nArchitectural Differences:\n")
    
    print("""
BI-ENCODER (Two-Tower):
┌─────────┐              ┌─────────┐
│  Query  │              │Document │
└────┬────┘              └────┬────┘
     │                        │
     ▼                        ▼
 ┌───────┐              ┌───────┐
 │Encoder│              │Encoder│
 └───┬───┘              └───┬───┘
     │                        │
     ▼                        ▼
 [Vector_Q]            [Vector_D]
     │                        │
     └───────────┬────────────┘
                 ▼
          cosine_similarity

Pros:
  ✓ Fast: Precompute document embeddings
  ✓ Scalable: Use vector database (ANN)
  ✓ Efficient: O(1) per document (dot product)

Cons:
  ✗ Limited interaction between query and doc
  ✗ Single vector must capture all info
  ✗ Can't model complex relevance patterns


CROSS-ENCODER (Single-Tower):
┌──────────────────────┐
│  [Query] [SEP] [Doc] │
└──────────┬───────────┘
           │
           ▼
       ┌───────┐
       │Encoder│
       └───┬───┘
           │
           ▼
     ┌──────────┐
     │Classifier│
     └─────┬────┘
           │
           ▼
    Relevance Score

Pros:
  ✓ Full interaction between query and doc
  ✓ Attention across all tokens
  ✓ Much more accurate
  ✓ Can model nuanced relevance

Cons:
  ✗ Slow: Must encode every pair
  ✗ Can't precompute
  ✗ O(N) for N documents
  ✗ Not scalable for initial retrieval
""")
    
    print("=" * 60)
    print("\nPerformance Comparison:\n")
    
    comparison = """
Task: Search 1M documents

Bi-Encoder:
  Index time:  10 minutes (embed all docs once)
  Query time:  10ms (vector search via HNSW)
  Accuracy:    Good (70-80% precision)
  Scalability: Excellent (millions of docs)

Cross-Encoder:
  Index time:  N/A (no precomputation)
  Query time:  1M × 50ms = 13.9 hours (!!)
  Accuracy:    Excellent (85-95% precision)
  Scalability: Poor (only for reranking)

Solution: Use Both!
  1. Bi-encoder retrieves top-100 (10ms)
  2. Cross-encoder reranks to top-10 (100 × 50ms = 5s)
  
  Total time: ~5s
  Accuracy: Excellent ✓
  Scalability: Good ✓
"""
    
    print(comparison)

bi_vs_cross_encoder()
```

## Cross-Encoder Reranking

### Implementing Cross-Encoder Reranking

```python
def implement_cross_encoder():
    """Implementing cross-encoder reranking."""
    
    print("\n\nImplementing Cross-Encoder Reranking:\n")
    
    code = '''
from sentence_transformers import CrossEncoder
from typing import List, Dict

class CrossEncoderReranker:
    """Rerank results using a cross-encoder model."""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize cross-encoder reranker.
        
        Popular models:
          • cross-encoder/ms-marco-MiniLM-L-6-v2 (Fast, good quality)
          • cross-encoder/ms-marco-MiniLM-L-12-v2 (Better quality, slower)
          • cross-encoder/ms-marco-TinyBERT-L-2-v2 (Fastest, lower quality)
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank documents for a query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return
        
        Returns:
            Reranked documents with new scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc['text']] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked = sorted(
            documents,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        return reranked[:top_k]
    
    def rerank_batched(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Rerank with batching for better throughput.
        
        Args:
            batch_size: Number of pairs to score at once
        """
        if not documents:
            return []
        
        # Prepare pairs
        pairs = [[query, doc['text']] for doc in documents]
        
        # Score in batches
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            batch_scores = self.model.predict(batch)
            all_scores.extend(batch_scores)
        
        # Add scores and sort
        for doc, score in zip(documents, all_scores):
            doc['rerank_score'] = float(score)
        
        reranked = sorted(
            documents,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        return reranked[:top_k]


# Example usage
reranker = CrossEncoderReranker()

# Initial retrieval results (from bi-encoder)
initial_results = [
    {'text': 'Python is a programming language.', 'id': 1, 'score': 0.85},
    {'text': 'Java is used for enterprise applications.', 'id': 2, 'score': 0.82},
    {'text': 'Python tutorial for beginners.', 'id': 3, 'score': 0.80},
    {'text': 'Learning Python programming.', 'id': 4, 'score': 0.78},
    {'text': 'JavaScript for web development.', 'id': 5, 'score': 0.75},
]

query = "Python tutorial"

# Rerank
reranked_results = reranker.rerank(query, initial_results, top_k=3)

print(f"Query: {query}\\n")
print("After Reranking:")
for i, result in enumerate(reranked_results, 1):
    print(f"{i}. {result['text']}")
    print(f"   Original score: {result.get('score', 'N/A')}")
    print(f"   Rerank score:   {result['rerank_score']:.4f}")
    print()

# Output might show reordering:
# 1. Python tutorial for beginners. (moved from 3rd to 1st)
# 2. Learning Python programming. (moved from 4th to 2nd)
# 3. Python is a programming language. (moved from 1st to 3rd)
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nCross-Encoder Models:\n")
    
    models = [
        ('ms-marco-MiniLM-L-6-v2', 'Balanced (RECOMMENDED)', '~50ms/pair'),
        ('ms-marco-MiniLM-L-12-v2', 'Higher quality', '~100ms/pair'),
        ('ms-marco-TinyBERT-L-2-v2', 'Fastest', '~20ms/pair'),
        ('ms-marco-electra-base', 'High quality', '~150ms/pair'),
    ]
    
    print(f"{'Model':<30} {'Quality':<20} {'Speed'}")
    print("-" * 70)
    for model, quality, speed in models:
        print(f"{model:<30} {quality:<20} {speed}")
    
    print("\n" + "=" * 60)
    print("\nOptimizations:\n")
    
    optimizations = [
        ('Batching', 'Process multiple pairs together (2-3x faster)'),
        ('GPU', 'Use GPU for inference (5-10x faster)'),
        ('ONNX', 'Convert to ONNX for faster inference'),
        ('Quantization', 'int8 quantization (2x faster, slight quality loss)'),
        ('Limit candidates', 'Only rerank top-50 to top-100, not all'),
    ]
    
    for opt, description in optimizations:
        print(f"  • {opt}: {description}")

implement_cross_encoder()
```

## Reciprocal Rank Fusion

### Understanding RRF

```python
def reciprocal_rank_fusion():
    """Deep dive into Reciprocal Rank Fusion."""
    
    print("\n\nReciprocal Rank Fusion (RRF):\n")
    
    print("=" * 60)
    print("\nAlgorithm:\n")
    
    print("""
RRF combines rankings from multiple sources without needing scores.

Formula:
  RRF_score(doc, k) = Σ 1 / (k + rank_i(doc))
                      i

Where:
  • rank_i(doc) = rank of doc in result list i
  • k = constant (typically 60)
  • Σ sums over all result lists containing doc

Example:

Result List 1 (Dense Retrieval):
  1. Doc A
  2. Doc B
  3. Doc C
  4. Doc D

Result List 2 (Sparse Retrieval):
  1. Doc B
  2. Doc D
  3. Doc A
  4. Doc E

RRF scores (k=60):
  Doc A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
  Doc B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325 ← Highest
  Doc C: 1/(60+3) + 0        = 0.0154
  Doc D: 1/(60+4) + 1/(60+2) = 0.0156 + 0.0161 = 0.0317
  Doc E: 0        + 1/(60+4) = 0.0156

Final Ranking (by RRF score):
  1. Doc B (0.0325) ← Ranked high in BOTH lists
  2. Doc A (0.0323)
  3. Doc D (0.0317)
  4. Doc C (0.0154) ← Only in one list
  5. Doc E (0.0156) ← Only in one list
""")
    
    print("=" * 60)
    print("\nWhy RRF Works:\n")
    
    print("""
Benefits:
  ✓ No score normalization needed
  ✓ Ranks matter, not absolute scores
  ✓ Simple and effective
  ✓ Favors docs appearing in multiple lists
  ✓ Robust to different scoring scales

Intuition:
  • High rank (low number) → High contribution
  • Appearing in multiple lists → Multiple contributions
  • Docs ranked high in ALL lists win

Tuning k:
  • Smaller k: More emphasis on top-ranked items
  • Larger k: More democratic, less emphasis on top
  • Default k=60 works well in practice
  • Range: 10-100
""")

reciprocal_rank_fusion()
```

### Implementing RRF

```python
def implement_rrf():
    """Implementing Reciprocal Rank Fusion."""
    
    print("\n\nImplementing RRF:\n")
    
    code = '''
from typing import List, Dict
from collections import defaultdict

class RRFFusion:
    """Reciprocal Rank Fusion for combining multiple rankings."""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF.
        
        Args:
            k: RRF constant (typically 60)
        """
        self.k = k
    
    def fuse(
        self,
        result_lists: List[List[Dict]],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Fuse multiple result lists using RRF.
        
        Args:
            result_lists: List of result lists from different retrievers
            top_k: Number of final results
        
        Returns:
            Fused and ranked results
        """
        # Calculate RRF scores
        doc_scores = defaultdict(lambda: {'score': 0.0, 'doc': None, 'ranks': []})
        
        for list_idx, results in enumerate(result_lists):
            for rank, result in enumerate(results, start=1):
                doc_id = result['id']
                
                # RRF contribution from this list
                rrf_score = 1.0 / (self.k + rank)
                
                doc_scores[doc_id]['score'] += rrf_score
                doc_scores[doc_id]['ranks'].append((list_idx, rank))
                
                # Store document (first occurrence)
                if doc_scores[doc_id]['doc'] is None:
                    doc_scores[doc_id]['doc'] = result
        
        # Convert to list and sort
        fused_results = []
        for doc_id, data in doc_scores.items():
            fused_results.append({
                'id': doc_id,
                'document': data['doc'],
                'rrf_score': data['score'],
                'ranks': data['ranks'],  # [(list_idx, rank), ...]
                'num_lists': len(data['ranks'])  # In how many lists?
            })
        
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        return fused_results[:top_k]
    
    def fuse_weighted(
        self,
        result_lists: List[List[Dict]],
        weights: List[float],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Weighted RRF - give different importance to different lists.
        
        Args:
            result_lists: List of result lists
            weights: Weight for each list (must sum to 1.0)
            top_k: Number of final results
        
        Returns:
            Weighted fused results
        """
        assert len(result_lists) == len(weights)
        assert abs(sum(weights) - 1.0) < 0.001, "Weights must sum to 1.0"
        
        doc_scores = defaultdict(lambda: {'score': 0.0, 'doc': None})
        
        for list_idx, (results, weight) in enumerate(zip(result_lists, weights)):
            for rank, result in enumerate(results, start=1):
                doc_id = result['id']
                
                # Weighted RRF contribution
                rrf_score = weight / (self.k + rank)
                
                doc_scores[doc_id]['score'] += rrf_score
                
                if doc_scores[doc_id]['doc'] is None:
                    doc_scores[doc_id]['doc'] = result
        
        # Sort and return
        fused_results = [
            {
                'id': doc_id,
                'document': data['doc'],
                'rrf_score': data['score']
            }
            for doc_id, data in doc_scores.items()
        ]
        
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        return fused_results[:top_k]


# Example usage
rrf = RRFFusion(k=60)

# Results from different retrievers
dense_results = [
    {'id': 1, 'text': 'Python programming language'},
    {'id': 2, 'text': 'Java enterprise applications'},
    {'id': 3, 'text': 'Python tutorial'},
]

sparse_results = [
    {'id': 3, 'text': 'Python tutorial'},
    {'id': 1, 'text': 'Python programming language'},
    {'id': 4, 'text': 'JavaScript web dev'},
]

semantic_results = [
    {'id': 2, 'text': 'Java enterprise applications'},
    {'id': 3, 'text': 'Python tutorial'},
    {'id': 5, 'text': 'Coding basics'},
]

# Fuse results
fused = rrf.fuse([dense_results, sparse_results, semantic_results], top_k=5)

print("Fused Results (RRF):\\n")
for i, result in enumerate(fused, 1):
    print(f"{i}. {result['document']['text']}")
    print(f"   RRF Score: {result['rrf_score']:.4f}")
    print(f"   Appeared in {result['num_lists']} lists")
    print(f"   Ranks: {result['ranks']}")
    print()

# Weighted RRF (e.g., trust dense more)
fused_weighted = rrf.fuse_weighted(
    [dense_results, sparse_results, semantic_results],
    weights=[0.5, 0.3, 0.2],  # 50% dense, 30% sparse, 20% semantic
    top_k=5
)

print("\\nWeighted RRF Results:\\n")
for i, result in enumerate(fused_weighted, 1):
    print(f"{i}. {result['document']['text']} (Score: {result['rrf_score']:.4f})")
'''
    
    print(code)

implement_rrf()
```

## Relevance Scoring Models

### Alternative Scoring Approaches

```python
def relevance_scoring():
    """Alternative relevance scoring methods."""
    
    print("\n\nRelevance Scoring Models:\n")
    
    print("=" * 60)
    print("\nScoring Methods:\n")
    
    print("""
1. LINEAR COMBINATION
   
   score = w1·dense_score + w2·sparse_score + w3·cross_encoder_score
   
   Where weights w1, w2, w3 are learned or tuned

2. CASCADE SCORING
   
   Stage 1: Fast scorer (filters to top-100)
   Stage 2: Medium scorer (filters to top-20)
   Stage 3: Slow scorer (final top-5)
   
   Each stage more accurate but slower

3. LEARNING TO RANK (LTR)
   
   Train a model to combine multiple signals:
   
   Features:
     • Dense similarity score
     • Sparse (BM25) score
     • Cross-encoder score
     • Document metadata (freshness, quality)
     • Query-doc features (length, overlap)
   
   Model: XGBoost, LightGBM, Neural Network
   
   Learns optimal combination from labeled data

4. POINTWISE VS PAIRWISE VS LISTWISE
   
   Pointwise: Score each doc independently
   Pairwise:  Compare doc pairs (which is better?)
   Listwise:  Score entire result list
   
   Listwise typically best for ranking

5. COLBERT (Late Interaction)
   
   Token-level similarity with MaxSim aggregation:
   
   score(Q, D) = Σ max(cos(q_i, d_j) for all d_j)
                 i
   
   Sum over query tokens of max similarity with doc tokens
""")
    
    code = '''
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class LearningToRankScorer:
    """Learning-to-rank for relevance scoring."""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
    
    def extract_features(
        self,
        query: str,
        document: Dict,
        dense_score: float,
        sparse_score: float,
        cross_score: float = None
    ) -> np.ndarray:
        """Extract features for LTR."""
        
        features = []
        
        # Retrieval scores
        features.append(dense_score)
        features.append(sparse_score)
        if cross_score is not None:
            features.append(cross_score)
        else:
            features.append(0.0)
        
        # Query-document features
        query_terms = set(query.lower().split())
        doc_terms = set(document['text'].lower().split())
        
        # Term overlap
        overlap = len(query_terms & doc_terms)
        features.append(overlap)
        
        # Jaccard similarity
        jaccard = overlap / len(query_terms | doc_terms) if (query_terms | doc_terms) else 0
        features.append(jaccard)
        
        # Length features
        features.append(len(document['text']))
        features.append(len(query))
        
        # Metadata features (if available)
        metadata = document.get('metadata', {})
        features.append(metadata.get('quality_score', 0.5))
        features.append(metadata.get('recency_score', 0.5))
        
        return np.array(features)
    
    def train(self, training_data: List[Dict]):
        """
        Train LTR model.
        
        Args:
            training_data: List of examples with features and labels
                [{
                    'query': str,
                    'document': dict,
                    'dense_score': float,
                    'sparse_score': float,
                    'cross_score': float,
                    'relevance': float  # Label (0-1 or 0-4)
                }, ...]
        """
        X = []
        y = []
        
        for example in training_data:
            features = self.extract_features(
                example['query'],
                example['document'],
                example['dense_score'],
                example['sparse_score'],
                example.get('cross_score')
            )
            X.append(features)
            y.append(example['relevance'])
        
        X = np.array(X)
        y = np.array(y)
        
        self.model.fit(X, y)
        
        print(f"Trained on {len(training_data)} examples")
    
    def score(
        self,
        query: str,
        document: Dict,
        dense_score: float,
        sparse_score: float,
        cross_score: float = None
    ) -> float:
        """Score a query-document pair."""
        
        features = self.extract_features(
            query,
            document,
            dense_score,
            sparse_score,
            cross_score
        )
        
        score = self.model.predict([features])[0]
        
        return float(score)
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """Rerank documents using LTR."""
        
        for doc in documents:
            score = self.score(
                query,
                doc['document'],
                doc.get('dense_score', 0),
                doc.get('sparse_score', 0),
                doc.get('cross_score')
            )
            doc['ltr_score'] = score
        
        # Sort by LTR score
        reranked = sorted(documents, key=lambda x: x['ltr_score'], reverse=True)
        
        return reranked[:top_k]


# Example usage
ltr = LearningToRankScorer()

# Training (would come from labeled data)
training_data = [
    {
        'query': 'Python tutorial',
        'document': {'text': 'Python programming guide', 'metadata': {'quality_score': 0.9}},
        'dense_score': 0.85,
        'sparse_score': 0.75,
        'cross_score': 0.92,
        'relevance': 1.0  # Highly relevant
    },
    # ... more examples
]

ltr.train(training_data)

# Scoring
documents = [
    {
        'document': {'text': 'Python tutorial for beginners'},
        'dense_score': 0.82,
        'sparse_score': 0.88
    },
    # ... more docs
]

reranked = ltr.rerank("Python tutorial", documents, top_k=5)
'''
    
    print(code)

relevance_scoring()
```

## Multi-Stage Retrieval

### Cascading Retrieval Pipeline

```python
def multi_stage_retrieval():
    """Multi-stage retrieval with increasing accuracy."""
    
    print("\n\nMulti-Stage Retrieval:\n")
    
    print("=" * 60)
    print("\nPipeline Architecture:\n")
    
    print("""
Concept: Progressively narrow down results with increasing accuracy

┌────────────────────────────────────────────────────────────┐
│ Stage 1: CANDIDATE GENERATION (Fast, High Recall)         │
│                                                            │
│ 1M documents → [Vector Search] → Top-1000                 │
│                                                            │
│ Speed:  ██████████ (10ms)                                 │
│ Quality: ████       (70% precision)                       │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Stage 2: FILTERING (Medium Speed, Better Quality)         │
│                                                            │
│ 1000 docs → [BM25 + Filters] → Top-100                    │
│                                                            │
│ Speed:  ███████    (50ms)                                 │
│ Quality: ███████   (80% precision)                        │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Stage 3: RERANKING (Slow, High Precision)                 │
│                                                            │
│ 100 docs → [Cross-Encoder] → Top-10                       │
│                                                            │
│ Speed:  ████       (5s)                                   │
│ Quality: ██████████ (95% precision)                       │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Stage 4 (Optional): FINAL SCORING                         │
│                                                            │
│ 10 docs → [LLM Scoring] → Top-3                           │
│                                                            │
│ Speed:  ██         (10s)                                  │
│ Quality: ██████████ (98% precision)                       │
└────────────────────────────────────────────────────────────┘

Total latency: 10ms + 50ms + 5s + 10s = ~15s
  (or ~5s without Stage 4)

Final quality: 95-98% precision ✓
""")
    
    code = '''
class MultiStageRetriever:
    """Multi-stage retrieval pipeline."""
    
    def __init__(
        self,
        vector_index,
        sparse_index,
        cross_encoder,
        llm_scorer=None
    ):
        self.vector_index = vector_index
        self.sparse_index = sparse_index
        self.cross_encoder = cross_encoder
        self.llm_scorer = llm_scorer
    
    def retrieve(
        self,
        query: str,
        stage1_k: int = 1000,
        stage2_k: int = 100,
        stage3_k: int = 10,
        final_k: int = 3
    ) -> List[Dict]:
        """
        Multi-stage retrieval.
        
        Args:
            query: Search query
            stage1_k: Candidates from stage 1
            stage2_k: Candidates from stage 2
            stage3_k: Candidates from stage 3
            final_k: Final results
        
        Returns:
            Top-k results after all stages
        """
        # STAGE 1: Fast vector search (cast wide net)
        print(f"Stage 1: Retrieving {stage1_k} candidates...")
        candidates = self.vector_index.search(query, top_k=stage1_k)
        print(f"  → Got {len(candidates)} candidates")
        
        # STAGE 2: Hybrid scoring with sparse retrieval
        print(f"\\nStage 2: Filtering to {stage2_k}...")
        
        # Get sparse scores
        sparse_results = self.sparse_index.search(query, top_k=stage2_k)
        sparse_scores = {r['id']: r['score'] for r in sparse_results}
        
        # Combine dense + sparse
        for candidate in candidates:
            doc_id = candidate['id']
            candidate['sparse_score'] = sparse_scores.get(doc_id, 0.0)
            candidate['stage2_score'] = (
                0.6 * candidate['score'] +  # Dense
                0.4 * candidate['sparse_score']  # Sparse
            )
        
        # Sort by combined score
        candidates.sort(key=lambda x: x['stage2_score'], reverse=True)
        candidates = candidates[:stage2_k]
        print(f"  → Filtered to {len(candidates)} candidates")
        
        # STAGE 3: Cross-encoder reranking
        print(f"\\nStage 3: Reranking to {stage3_k}...")
        
        reranked = self.cross_encoder.rerank(
            query,
            candidates,
            top_k=stage3_k
        )
        print(f"  → Reranked to {len(reranked)} results")
        
        # STAGE 4 (Optional): LLM-based final scoring
        if self.llm_scorer and final_k < stage3_k:
            print(f"\\nStage 4: LLM scoring to {final_k}...")
            
            final_results = self.llm_scorer.score(
                query,
                reranked,
                top_k=final_k
            )
            print(f"  → Final {len(final_results)} results")
            
            return final_results
        
        return reranked[:final_k]
    
    def retrieve_adaptive(self, query: str) -> List[Dict]:
        """
        Adaptive retrieval - adjust stages based on query complexity.
        
        Simple queries: Fewer stages
        Complex queries: All stages
        """
        # Classify query complexity (simple heuristic)
        complexity = self._estimate_complexity(query)
        
        if complexity == 'simple':
            # Skip Stage 2, use only vector + rerank
            candidates = self.vector_index.search(query, top_k=100)
            return self.cross_encoder.rerank(query, candidates, top_k=5)
        
        elif complexity == 'medium':
            # Use Stage 1-3
            return self.retrieve(
                query,
                stage1_k=500,
                stage2_k=100,
                stage3_k=10,
                final_k=5
            )
        
        else:  # complex
            # Use all stages
            return self.retrieve(
                query,
                stage1_k=1000,
                stage2_k=100,
                stage3_k=10,
                final_k=3
            )
    
    def _estimate_complexity(self, query: str) -> str:
        """Estimate query complexity."""
        word_count = len(query.split())
        
        if word_count <= 3:
            return 'simple'
        elif word_count <= 8:
            return 'medium'
        else:
            return 'complex'


# Example usage
retriever = MultiStageRetriever(
    vector_index=dense_index,
    sparse_index=sparse_index,
    cross_encoder=cross_encoder,
    llm_scorer=llm_scorer  # Optional
)

query = "How does BERT handle long sequences?"

# Standard multi-stage
results = retriever.retrieve(query, final_k=5)

# Or adaptive
results = retriever.retrieve_adaptive(query)

print("\\nFinal Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['document']['text']}")
    print(f"   Final score: {result.get('final_score', result.get('rerank_score', 'N/A'))}")
'''
    
    print(code)

multi_stage_retrieval()
```

## Combining Multiple Retrievers

### Ensemble Retrieval

```python
def ensemble_retrieval():
    """Combining multiple retrieval systems."""
    
    print("\n\nEnsemble Retrieval:\n")
    
    print("=" * 60)
    print("\nWhy Multiple Retrievers?\n")
    
    print("""
Different retrievers have different strengths:

Dense (Embedding-based):
  ✓ Semantic understanding
  ✗ May miss exact terms

Sparse (BM25):
  ✓ Exact keyword matching
  ✗ No semantic understanding

Hybrid (Dense + Sparse):
  ✓ Both semantic and keywords
  ✗ Still may miss some patterns

Domain-Specific:
  ✓ Tuned for specific domain
  ✗ May not generalize

SOLUTION: Use ALL of them! Combine results via fusion.
""")
    
    code = '''
class EnsembleRetriever:
    """Combine multiple retrievers into an ensemble."""
    
    def __init__(self, retrievers: List[Any], fusion_method: str = 'rrf'):
        """
        Initialize ensemble.
        
        Args:
            retrievers: List of retriever objects (must have .search method)
            fusion_method: 'rrf' or 'weighted' or 'voting'
        """
        self.retrievers = retrievers
        self.fusion_method = fusion_method
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search using all retrievers and fuse results."""
        
        # Get results from all retrievers
        all_results = []
        for i, retriever in enumerate(self.retrievers):
            results = retriever.search(query, top_k=top_k*2)
            # Tag with retriever index
            for result in results:
                result['retriever_idx'] = i
            all_results.append(results)
        
        # Fuse results
        if self.fusion_method == 'rrf':
            return self._fuse_rrf(all_results, top_k)
        elif self.fusion_method == 'weighted':
            return self._fuse_weighted(all_results, top_k)
        elif self.fusion_method == 'voting':
            return self._fuse_voting(all_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _fuse_rrf(self, all_results: List[List[Dict]], top_k: int, k: int = 60):
        """Fuse using RRF."""
        doc_scores = defaultdict(lambda: {'score': 0.0, 'doc': None})
        
        for results in all_results:
            for rank, result in enumerate(results, start=1):
                doc_id = result['id']
                rrf_score = 1.0 / (k + rank)
                
                doc_scores[doc_id]['score'] += rrf_score
                if doc_scores[doc_id]['doc'] is None:
                    doc_scores[doc_id]['doc'] = result
        
        fused = sorted(
            [{'id': k, **v} for k, v in doc_scores.items()],
            key=lambda x: x['score'],
            reverse=True
        )
        
        return fused[:top_k]
    
    def _fuse_weighted(
        self,
        all_results: List[List[Dict]],
        top_k: int,
        weights: List[float] = None
    ):
        """Fuse with weighted combination."""
        if weights is None:
            weights = [1.0 / len(all_results)] * len(all_results)
        
        doc_scores = defaultdict(lambda: {'score': 0.0, 'doc': None})
        
        for results, weight in zip(all_results, weights):
            # Normalize scores
            if results:
                max_score = max(r['score'] for r in results)
                for result in results:
                    doc_id = result['id']
                    norm_score = result['score'] / max_score if max_score > 0 else 0
                    
                    doc_scores[doc_id]['score'] += weight * norm_score
                    if doc_scores[doc_id]['doc'] is None:
                        doc_scores[doc_id]['doc'] = result
        
        fused = sorted(
            [{'id': k, **v} for k, v in doc_scores.items()],
            key=lambda x: x['score'],
            reverse=True
        )
        
        return fused[:top_k]
    
    def _fuse_voting(self, all_results: List[List[Dict]], top_k: int):
        """Fuse using voting (count appearances)."""
        doc_votes = defaultdict(lambda: {'votes': 0, 'doc': None})
        
        for results in all_results:
            seen_in_list = set()
            for result in results:
                doc_id = result['id']
                if doc_id not in seen_in_list:
                    doc_votes[doc_id]['votes'] += 1
                    seen_in_list.add(doc_id)
                    if doc_votes[doc_id]['doc'] is None:
                        doc_votes[doc_id]['doc'] = result
        
        fused = sorted(
            [{'id': k, **v} for k, v in doc_votes.items()],
            key=lambda x: x['votes'],
            reverse=True
        )
        
        return fused[:top_k]


# Example: Ensemble of different retrievers
ensemble = EnsembleRetriever(
    retrievers=[
        dense_retriever,        # Embedding-based
        sparse_retriever,       # BM25
        domain_specific_retriever,  # Fine-tuned for domain
    ],
    fusion_method='rrf'
)

results = ensemble.search("Python machine learning", top_k=5)

# Benefits:
#  ✓ More robust (not reliant on single method)
#  ✓ Better coverage (each retriever finds different docs)
#  ✓ Higher quality (consensus across methods)
'''
    
    print(code)

ensemble_retrieval()
```

## Diversity and MMR

### Maximal Marginal Relevance

```python
def diversity_mmr():
    """Diversity-aware reranking with MMR."""
    
    print("\n\nDiversity and MMR:\n")
    
    print("=" * 60)
    print("\nThe Diversity Problem:\n")
    
    print("""
Problem: Top results may be very similar (redundant)

Query: "Python tutorial"

Without Diversity:
  1. Python tutorial for beginners
  2. Python programming tutorial
  3. Learn Python tutorial
  4. Python tutorial guide
  5. Python coding tutorial
     ↑ All very similar!

With Diversity:
  1. Python tutorial for beginners
  2. Advanced Python techniques
  3. Python for data science
  4. Python web development
  5. Python vs other languages
     ↑ Diverse perspectives!
""")
    
    print("=" * 60)
    print("\nMaximal Marginal Relevance (MMR):\n")
    
    print("""
MMR balances relevance and diversity:

MMR = λ × Relevance(doc, query) - (1-λ) × max(Similarity(doc, selected))

Where:
  • λ = trade-off parameter (0-1)
  • Relevance = how relevant doc is to query
  • Similarity = how similar doc is to already-selected docs

Algorithm:
  1. Start with empty result set
  2. Repeat until k results:
     a. For each remaining doc:
        - Calculate relevance to query
        - Calculate max similarity to selected docs
        - Compute MMR score
     b. Add doc with highest MMR score
     c. Remove from candidates

λ values:
  • λ = 1.0: Pure relevance (no diversity)
  • λ = 0.5: Balance relevance and diversity (RECOMMENDED)
  • λ = 0.0: Pure diversity (low relevance)
""")
    
    code = '''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DiversityReranker:
    """Rerank for diversity using MMR."""
    
    def __init__(self, embedding_model, lambda_param: float = 0.5):
        """
        Initialize diversity reranker.
        
        Args:
            embedding_model: Model to generate embeddings
            lambda_param: Trade-off between relevance and diversity (0-1)
        """
        self.model = embedding_model
        self.lambda_param = lambda_param
    
    def rerank_mmr(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rerank using Maximal Marginal Relevance.
        
        Args:
            query: Search query
            documents: Initial ranked documents
            top_k: Number of diverse results to return
        
        Returns:
            Diversified results
        """
        if not documents:
            return []
        
        # Embed query and documents
        query_embedding = self.model.encode([query])[0]
        doc_embeddings = self.model.encode([d['text'] for d in documents])
        
        # Calculate relevance scores (similarity to query)
        relevance_scores = cosine_similarity(
            doc_embeddings,
            query_embedding.reshape(1, -1)
        ).flatten()
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        for _ in range(min(top_k, len(documents))):
            if not remaining_indices:
                break
            
            # Calculate MMR scores for remaining docs
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]
                
                # Diversity component (max similarity to selected)
                if selected_indices:
                    similarities = cosine_similarity(
                        doc_embeddings[idx].reshape(1, -1),
                        doc_embeddings[selected_indices]
                    )[0]
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0.0
                
                # MMR score
                mmr = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_similarity
                )
                
                mmr_scores.append((idx, mmr))
            
            # Select doc with highest MMR
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Build result list
        results = []
        for rank, idx in enumerate(selected_indices, 1):
            results.append({
                'document': documents[idx],
                'relevance_score': float(relevance_scores[idx]),
                'rank': rank,
                'mmr_rank': rank
            })
        
        return results
    
    def rerank_clustering(
        self,
        documents: List[Dict],
        top_k: int = 10,
        n_clusters: int = None
    ) -> List[Dict]:
        """
        Diversify by selecting from different clusters.
        
        Args:
            documents: Documents to diversify
            top_k: Number of results
            n_clusters: Number of clusters (default: top_k)
        
        Returns:
            Diverse results (one from each cluster)
        """
        from sklearn.cluster import KMeans
        
        if n_clusters is None:
            n_clusters = min(top_k, len(documents))
        
        # Embed documents
        doc_embeddings = self.model.encode([d['text'] for d in documents])
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(doc_embeddings)
        
        # Select top doc from each cluster
        results = []
        for cluster_id in range(n_clusters):
            # Get docs in this cluster
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            
            if cluster_indices:
                # Select highest-scored doc in cluster
                best_in_cluster = max(
                    cluster_indices,
                    key=lambda i: documents[i].get('score', 0)
                )
                results.append(documents[best_in_cluster])
        
        return results[:top_k]


# Example usage
diversifier = DiversityReranker(model, lambda_param=0.5)

# Initial results (may be redundant)
initial_results = retriever.search("Python tutorial", top_k=50)

# Diversify with MMR
diverse_results = diversifier.rerank_mmr(
    query="Python tutorial",
    documents=initial_results,
    top_k=10
)

print("Diverse Results:")
for i, result in enumerate(diverse_results, 1):
    print(f"{i}. {result['document']['text']}")
    print(f"   Relevance: {result['relevance_score']:.4f}")
    print()

# Or: Cluster-based diversification
diverse_results = diversifier.rerank_clustering(initial_results, top_k=10)
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nWhen to Use Diversity:\n")
    
    scenarios = [
        ('Exploratory search', 'User wants to see different angles'),
        ('Broad queries', '"AI" - show different AI topics'),
        ('User interface', 'More engaging to see variety'),
        ('Avoiding redundancy', 'Don\'t repeat same information'),
    ]
    
    for scenario, description in scenarios:
        print(f"  • {scenario}: {description}")

diversity_mmr()
```

## LLM-Based Reranking

### Using LLMs for Relevance Scoring

```python
def llm_reranking():
    """LLM-based reranking (highest quality but slowest)."""
    
    print("\n\nLLM-Based Reranking:\n")
    
    print("=" * 60)
    print("\nConcept:\n")
    
    print("""
Use LLM to judge relevance:

Prompt:
  Query: "How to install Python?"
  
  Document: "Python can be downloaded from python.org..."
  
  Is this document relevant to the query?
  Rate from 0-10 and explain.

LLM Response:
  Rating: 9/10
  This document directly answers how to install Python.

Use ratings to rerank documents.
""")
    
    code = '''
class LLMReranker:
    """Rerank using LLM relevance judgments."""
    
    def __init__(self, llm_call):
        """
        Initialize LLM reranker.
        
        Args:
            llm_call: Function to call LLM (query, document) → score
        """
        self.llm_call = llm_call
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank using LLM.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of results
        
        Returns:
            Reranked results
        """
        scored_docs = []
        
        for doc in documents:
            # Ask LLM to rate relevance
            prompt = f"""Rate the relevance of this document to the query on a scale of 0-10.

Query: {query}

Document: {doc['text'][:500]}...

Provide only a number from 0-10:"""
            
            response = self.llm_call(prompt)
            
            try:
                score = float(response.strip())
            except:
                # Fallback if LLM doesn't return a number
                score = 5.0
            
            scored_docs.append({
                'document': doc,
                'llm_score': score
            })
        
        # Sort by LLM score
        scored_docs.sort(key=lambda x: x['llm_score'], reverse=True)
        
        return scored_docs[:top_k]
    
    def rerank_batched(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        batch_size: int = 5
    ) -> List[Dict]:
        """
        Rerank with batching (rate multiple docs in one LLM call).
        
        More efficient than individual calls.
        """
        scored_docs = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Create batch prompt
            prompt = f"Rate the relevance of these documents to the query (0-10 each):\\n\\n"
            prompt += f"Query: {query}\\n\\n"
            
            for j, doc in enumerate(batch, 1):
                prompt += f"Document {j}: {doc['text'][:200]}...\\n\\n"
            
            prompt += "Provide ratings as a comma-separated list (e.g., '8, 5, 9, 3, 7'):\\n"
            
            response = self.llm_call(prompt)
            
            # Parse ratings
            try:
                ratings = [float(x.strip()) for x in response.split(',')]
            except:
                ratings = [5.0] * len(batch)
            
            # Ensure we have enough ratings
            while len(ratings) < len(batch):
                ratings.append(5.0)
            
            for doc, rating in zip(batch, ratings):
                scored_docs.append({
                    'document': doc,
                    'llm_score': rating
                })
        
        # Sort and return
        scored_docs.sort(key=lambda x: x['llm_score'], reverse=True)
        
        return scored_docs[:top_k]


# Example usage
llm_reranker = LLMReranker(llm_call=call_gpt)

# Rerank top-10 with LLM (slow but highest quality)
candidates = retriever.search("Python tutorial", top_k=10)
final_results = llm_reranker.rerank(query="Python tutorial", documents=candidates, top_k=3)

# Batched for efficiency
final_results = llm_reranker.rerank_batched(
    query="Python tutorial",
    documents=candidates,
    top_k=3,
    batch_size=5  # Rate 5 docs per LLM call
)

print("LLM-Reranked Results:")
for i, result in enumerate(final_results, 1):
    print(f"{i}. {result['document']['text']} (LLM Score: {result['llm_score']}/10)")
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nPros and Cons:\n")
    
    print("""
Pros:
  ✓ Highest quality relevance judgments
  ✓ Can consider context, intent, nuance
  ✓ Understands complex queries
  ✓ No training data needed

Cons:
  ✗ Very slow (seconds per document)
  ✗ Expensive (LLM API costs)
  ✗ Rate limits on API calls
  ✗ Not deterministic (slight variations)

When to Use:
  • Final stage reranking (top-10 → top-3)
  • High-stakes applications
  • When quality >> speed
  • As validation/ground truth for other methods
""")

llm_reranking()
```

## Cost-Benefit Analysis

### Evaluating Reranking Trade-offs

```python
def cost_benefit_analysis():
    """Analyzing reranking costs and benefits."""
    
    print("\n\nCost-Benefit Analysis:\n")
    
    print("=" * 60)
    print("\nReranking Methods Comparison:\n")
    
    comparison = """
┌─────────────────┬──────────┬─────────┬──────────┬────────────┐
│ Method          │ Latency  │ Cost    │ Quality  │ Scalability│
├─────────────────┼──────────┼─────────┼──────────┼────────────┤
│ No Reranking    │ 10ms     │ $0.00   │ ⭐⭐⭐     │ ⭐⭐⭐⭐⭐    │
│ RRF Fusion      │ 20ms     │ $0.00   │ ⭐⭐⭐⭐    │ ⭐⭐⭐⭐⭐    │
│ Cross-Encoder   │ 5s       │ $0.01   │ ⭐⭐⭐⭐⭐   │ ⭐⭐⭐       │
│ LLM Reranking   │ 15s      │ $0.10   │ ⭐⭐⭐⭐⭐   │ ⭐⭐         │
│ Multi-Stage     │ 5-10s    │ $0.02   │ ⭐⭐⭐⭐⭐   │ ⭐⭐⭐⭐      │
└─────────────────┴──────────┴─────────┴──────────┴────────────┘

Latency: Time per query
Cost: $ per query (approximate)
Quality: Ranking quality (precision)
Scalability: Can handle high QPS?
"""
    
    print(comparison)
    
    print("\n" + "=" * 60)
    print("\nWhen Is Reranking Worth It?\n")
    
    scenarios = """
DEFINITELY Worth It:
  ✓ High-value queries (e.g., legal, medical)
  ✓ Low query volume (<100 QPS)
  ✓ Quality-critical applications
  ✓ Complex/ambiguous queries

Maybe Worth It:
  ~ Medium query volume (100-1000 QPS)
  ~ Moderate quality requirements
  ~ Budget allows ($0.01-0.10 per query)

Probably Not Worth It:
  ✗ High query volume (>1000 QPS)
  ✗ Tight latency requirements (<500ms)
  ✗ Low-stakes applications
  ✗ Simple queries work well without reranking
"""
    
    print(scenarios)
    
    print("\n" + "=" * 60)
    print("\nOptimization Strategies:\n")
    
    strategies = [
        ('Cache rerank results', 'For popular queries (saves 90% cost)'),
        ('Adaptive reranking', 'Only rerank complex queries'),
        ('Batch processing', 'Amortize latency across multiple queries'),
        ('Smaller models', 'TinyBERT vs full BERT (3x faster)'),
        ('GPU acceleration', 'Cross-encoder on GPU (5-10x faster)'),
        ('Limit candidates', 'Rerank top-50, not top-100'),
    ]
    
    for strategy, description in strategies:
        print(f"  • {strategy}: {description}")
    
    print("\n" + "=" * 60)
    print("\nRecommended Configurations:\n")
    
    configs = """
LOW LATENCY (<100ms):
  → RRF fusion only
  → Skip cross-encoder
  
BALANCED (1-5s):
  → RRF fusion + cross-encoder
  → Rerank top-50 to top-10
  
HIGH QUALITY (5-15s):
  → Multi-stage with cross-encoder
  → Optional LLM for top-5 to top-3
  
COST-OPTIMIZED:
  → RRF fusion (free)
  → Cross-encoder on self-hosted GPU
  → Cache aggressive
"""
    
    print(configs)

cost_benefit_analysis()
```

## Summary

**Key Concepts**:

1. **Reranking** improves retrieval quality with a slower, more accurate second pass
2. **Bi-encoders** (retrieval) are fast but limited; **cross-encoders** (reranking) are slow but accurate
3. **RRF** (Reciprocal Rank Fusion) combines multiple rankings without score normalization
4. **Multi-stage retrieval** progressively narrows candidates with increasing accuracy
5. **Diversity** (MMR) prevents redundant results
6. **LLM reranking** offers highest quality but at significant cost/latency

**Two-Stage Architecture**:

```
┌─────────────────────────────────────────────────────┐
│ STAGE 1: RETRIEVAL (Fast, Wide Net)                │
│                                                     │
│ • Bi-encoder (dense) + BM25 (sparse)               │
│ • Vector search: ~10ms                             │
│ • Retrieve top-100 candidates                      │
│ • Goal: High recall (don't miss relevant docs)    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ STAGE 2: RERANKING (Slow, Precise)                 │
│                                                     │
│ • Cross-encoder                                     │
│ • Process: ~5s for 100 docs                        │
│ • Rerank to top-10                                 │
│ • Goal: High precision (best docs first)          │
└─────────────────────────────────────────────────────┘
```

**Bi-Encoder vs Cross-Encoder**:

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|------------|---------------|
| Architecture | Separate encoding | Joint encoding |
| Speed | Fast (~1ms per doc) | Slow (~50ms per doc) |
| Quality | Good (70-80% precision) | Excellent (85-95% precision) |
| Precomputation | Yes (embed docs once) | No (must encode each pair) |
| Scalability | Excellent (millions of docs) | Poor (only for reranking) |
| Use Case | Initial retrieval | Reranking |

**RRF (Reciprocal Rank Fusion)**:

```
Formula:
  RRF_score(doc) = Σ 1/(k + rank_i(doc))
                   i

Where:
  • rank_i(doc) = doc's rank in result list i
  • k = constant (typically 60)
  • Sum over all lists containing doc

Benefits:
  ✓ No score normalization needed
  ✓ Favors consensus (high in multiple lists)
  ✓ Simple and effective
  ✓ Parameter-free (just tune k)

Example:
  Dense:  Doc A rank 1, Doc B rank 2
  Sparse: Doc B rank 1, Doc A rank 3
  
  Doc A: 1/61 + 1/64 = 0.0323
  Doc B: 1/62 + 1/61 = 0.0325 ← Higher (consensus winner)
```

**Multi-Stage Pipeline**:

```
Query: "Python machine learning"

Stage 1: Vector Search (10ms)
  1M docs → Top-1000
  Quality: 70%

Stage 2: Hybrid (BM25 + Dense) (50ms)
  1000 docs → Top-100
  Quality: 80%

Stage 3: Cross-Encoder (5s)
  100 docs → Top-10
  Quality: 95%

Stage 4 (Optional): LLM (15s)
  10 docs → Top-3
  Quality: 98%

Total: ~5-20s depending on stages
```

**Diversity (MMR)**:

```
Maximal Marginal Relevance:

MMR = λ·Relevance(doc, query) - (1-λ)·max(Similarity(doc, selected))

Where:
  • λ = 0.5 (balance relevance and diversity)
  • Relevance = cosine similarity to query
  • Similarity = max similarity to already-selected docs

Algorithm:
  1. Start with empty result set
  2. Iteratively add doc with highest MMR score
  3. Repeat until k results

Effect:
  • Selects relevant docs
  • Avoids redundancy
  • Shows different perspectives

Use when:
  • Exploratory search
  • Broad queries
  • Want variety in results
```

**Cross-Encoder Models**:

```
Recommended Models (Sentence Transformers):

Model                           | Speed    | Quality | Use Case
--------------------------------|----------|---------|-------------
ms-marco-TinyBERT-L-2-v2       | ~20ms    | Good    | Speed critical
ms-marco-MiniLM-L-6-v2         | ~50ms    | Great   | RECOMMENDED
ms-marco-MiniLM-L-12-v2        | ~100ms   | Excellent| Quality critical
ms-marco-electra-base          | ~150ms   | Excellent| Best quality

All trained on MS MARCO passage ranking dataset
```

**Cost-Benefit Analysis**:

```
Method          | Latency | Cost/Query | Quality | Recommendation
----------------|---------|------------|---------|------------------
No reranking    | 10ms    | $0.00      | 70%     | Simple use cases
RRF fusion      | 20ms    | $0.00      | 80%     | Always add this
Cross-encoder   | 5s      | $0.01      | 95%     | Production default
LLM reranking   | 15s     | $0.10      | 98%     | High-stakes only
Multi-stage     | 5-10s   | $0.02      | 95%     | Best balance ✓

Optimization:
  • Cache popular query results (90% cost reduction)
  • Use GPU for cross-encoder (5-10x faster)
  • Batch multiple queries together
  • Adaptive: rerank only complex queries
```

**Implementation Template**:

```python
# Standard production setup
def production_retrieval(query):
    # Stage 1: Fast retrieval (100 candidates)
    candidates = hybrid_search(query, top_k=100)  # 20ms
    
    # Stage 2: Cross-encoder reranking (10 results)
    results = cross_encoder.rerank(query, candidates, top_k=10)  # 5s
    
    # Stage 3 (Optional): Diversity
    results = mmr_diversify(query, results, lambda_=0.5)  # 100ms
    
    return results[:5]

# Total latency: ~5s
# Quality: 95% precision
# Cost: ~$0.01 per query
```

**Best Practices**:

1. **Always use RRF** to combine dense + sparse (free improvement)
2. **Add cross-encoder** for quality-critical applications
3. **Tune top_k**: Retrieve 100, rerank to 10, return 5
4. **Cache results** for popular queries (huge cost savings)
5. **Use GPU** for cross-encoder inference (much faster)
6. **Batch when possible** (amortize latency)
7. **Monitor metrics**: precision@k, NDCG, MRR
8. **A/B test** to validate improvements

**Common Pitfalls**:

- Reranking too many docs (top-1000) → Too slow
- Reranking too few docs (top-10) → May miss good docs
- No fusion → Missing complementary signals
- LLM reranking everything → Too expensive
- No caching → Redundant expensive operations
- Ignoring diversity → Redundant results

**Recommended Configurations**:

```
Simple Use Case (e.g., documentation search):
  → Dense retrieval only
  → No reranking (fast enough)

Standard Production:
  → Hybrid (dense + sparse via RRF)
  → Cross-encoder rerank top-100 to top-10
  → Total: ~5s latency

High-Stakes (legal, medical):
  → Multi-stage pipeline
  → Cross-encoder + optional LLM
  → Diversity (MMR)
  → Total: ~10-15s latency

Cost-Optimized:
  → RRF fusion only (free)
  → Self-hosted cross-encoder on GPU
  → Aggressive caching
```

## Next Steps

- Study [RAG Evaluation](rag-evaluation.md) to measure reranking improvements
- Review [Retrieval Strategies](retrieval-strategies.md) for first-stage retrieval
- Learn [RAG Architecture](rag-architecture.md) for end-to-end system design
- Master [Vector Databases](vector-databases.md) for efficient candidate generation
- Apply to production in [Application Patterns](../application_patterns/)
