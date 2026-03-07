# Vector Databases

## Table of Contents

- [Introduction](#introduction)
- [What are Vector Databases?](#what-are-vector-databases)
- [Vector Database Fundamentals](#vector-database-fundamentals)
- [Indexing Strategies](#indexing-strategies)
- [Popular Vector Databases](#popular-vector-databases)
- [Choosing a Vector Database](#choosing-a-vector-database)
- [Implementation Examples](#implementation-examples)
- [Performance Optimization](#performance-optimization)
- [Scaling Considerations](#scaling-considerations)
- [Advanced Features](#advanced-features)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Vector databases** are specialized data stores designed for efficiently storing, indexing, and querying high-dimensional vector embeddings. They are the backbone of RAG systems, enabling semantic search at scale.

```
Traditional Database:
┌─────────────┬──────────┬────────┐
│ ID          │ Name     │ Price  │
├─────────────┼──────────┼────────┤
│ 1           │ Apple    │ $1.50  │
│ 2           │ Orange   │ $2.00  │
└─────────────┴──────────┴────────┘

Search: WHERE name = 'Apple'  ← Exact match


Vector Database:
┌────┬─────────────────────────────────────────┬──────────┐
│ ID │ Embedding (768 dims)                    │ Metadata │
├────┼─────────────────────────────────────────┼──────────┤
│ 1  │ [0.23, -0.15, 0.87, ..., 0.42]         │ {type:..}│
│ 2  │ [-0.44, 0.91, -0.12, ..., -0.23]       │ {type:..}│
└────┴─────────────────────────────────────────┴──────────┘

Search: Find vectors similar to [0.25, -0.10, 0.90, ...]
        ← Semantic similarity
```

**Why vector databases?**

- **Semantic search**: Find conceptually similar items, not just keyword matches
- **Scale**: Handle millions to billions of vectors
- **Speed**: Millisecond query latency with ANN algorithms
- **Metadata filtering**: Combine vector similarity with traditional filters
- **Purpose-built**: Optimized for high-dimensional data

This guide covers vector database concepts, indexing strategies, popular solutions, and how to choose and use them effectively.

## What are Vector Databases?

### Core Concept

```python
def vector_database_explained():
    """Understanding vector databases."""
    
    print("What is a Vector Database?\n")
    
    print("=" * 60)
    print("\nCore Purpose:\n")
    
    print("""
Vector databases are designed to store and search embeddings.

EMBEDDING: A vector (list of numbers) representing semantic meaning
Example: "dog" → [0.2, -0.4, 0.8, 0.1, ...]  (384 dimensions)

VECTOR SEARCH: Find embeddings that are "close" in vector space
- Close vectors = semantically similar content
- Use distance metrics (cosine, euclidean, dot product)
""")
    
    print("=" * 60)
    print("\nHow It Works:\n")
    
    print("""
1. INSERT:
   Text: "Paris is the capital of France"
      ↓ (embedding model)
   Vector: [0.23, -0.15, 0.87, ..., 0.42]
      ↓ (store)
   Vector DB: [vector, metadata, id]

2. SEARCH:
   Query: "French capital city"
      ↓ (embedding model)
   Query Vector: [0.25, -0.10, 0.90, ..., 0.45]
      ↓ (similarity search)
   Vector DB: Find nearest neighbors
      ↓
   Results: [
     (id=1, distance=0.05, text="Paris is the capital..."),
     (id=5, distance=0.12, text="France's capital..."),
     ...
   ]
""")
    
    print("=" * 60)
    print("\nKey Characteristics:\n")
    
    characteristics = [
        ('High-dimensional', 'Handles 100s-1000s of dimensions'),
        ('Approximate', 'Trade-off: speed vs perfect accuracy'),
        ('Optimized indexes', 'HNSW, IVF, etc. for fast search'),
        ('Metadata filtering', 'Combine with traditional filters'),
        ('Horizontal scaling', 'Handle billions of vectors'),
    ]
    
    for char, description in characteristics:
        print(f"  • {char}: {description}")

vector_database_explained()
```

### Vector Search vs Traditional Search

```python
def vector_vs_traditional_search():
    """Comparing vector search to traditional search."""
    
    print("\n\nVector Search vs Traditional Search:\n")
    
    comparison = {
        'Match Type': {
            'Traditional': 'Keyword/exact match',
            'Vector': 'Semantic similarity'
        },
        'Query': {
            'Traditional': 'SQL WHERE clauses',
            'Vector': 'Query vector + distance metric'
        },
        'Handles Synonyms': {
            'Traditional': 'No (need explicit)',
            'Vector': 'Yes (semantic understanding)'
        },
        'Typo Tolerance': {
            'Traditional': 'Limited (fuzzy match)',
            'Vector': 'Good (meaning preserved)'
        },
        'Multi-language': {
            'Traditional': 'Difficult',
            'Vector': 'Natural (shared space)'
        },
        'Scaling': {
            'Traditional': 'Easy (indexes)',
            'Vector': 'Complex (ANN required)'
        },
        'Latency': {
            'Traditional': 'Very fast (< 1ms)',
            'Vector': 'Fast (1-50ms with ANN)'
        }
    }
    
    print(f"{'Aspect':<20} {'Traditional Search':<30} {'Vector Search'}")
    print("=" * 80)
    
    for aspect, values in comparison.items():
        print(f"{aspect:<20} {values['Traditional']:<30} {values['Vector']}")
    
    print("\n" + "=" * 60)
    print("\nExample Comparison:\n")
    
    print("Query: 'puppy training tips'\n")
    
    print("TRADITIONAL SEARCH:")
    print("  Matches: Documents with exact words 'puppy', 'training', 'tips'")
    print("  Misses: 'dog obedience lessons', 'canine education guide'")
    print("  Result: Limited by exact vocabulary\n")
    
    print("VECTOR SEARCH:")
    print("  Matches: Documents semantically similar")
    print("  Finds: 'dog training', 'puppy obedience', 'canine education'")
    print("  Result: Captures intent, not just keywords\n")
    
    print("=" * 60)
    print("\nBest Practice: Hybrid Search\n")
    print("Combine both:")
    print("  • Vector search: Semantic understanding")
    print("  • Keyword search: Exact matches")
    print("  • Result: Better precision and recall")

vector_vs_traditional_search()
```

## Vector Database Fundamentals

### Distance Metrics

```python
import numpy as np

def distance_metrics():
    """Understanding distance metrics for vector search."""
    
    print("Vector Distance Metrics:\n")
    
    # Example vectors
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([2.0, 3.0, 4.0])
    v3 = np.array([-1.0, -2.0, -3.0])
    
    print("=" * 60)
    print("\n1. COSINE SIMILARITY (Most Common)\n")
    
    print("Measures: Angle between vectors (direction, not magnitude)")
    print("Range: -1 (opposite) to 1 (same direction)")
    print("Usage: Text embeddings (magnitude doesn't matter)\n")
    
    def cosine_similarity(a, b):
        """Cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    cos_sim_v1_v2 = cosine_similarity(v1, v2)
    cos_sim_v1_v3 = cosine_similarity(v1, v3)
    
    print(f"Example:")
    print(f"  v1 = {v1}")
    print(f"  v2 = {v2} (similar direction)")
    print(f"  v3 = {v3} (opposite direction)")
    print()
    print(f"  cosine_similarity(v1, v2) = {cos_sim_v1_v2:.4f} ← Similar")
    print(f"  cosine_similarity(v1, v3) = {cos_sim_v1_v3:.4f} ← Opposite")
    print()
    
    code = '''
def cosine_similarity(a, b):
    """Cosine similarity: dot(a, b) / (||a|| * ||b||)"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Or cosine distance (1 - similarity, for "distance")
def cosine_distance(a, b):
    return 1 - cosine_similarity(a, b)
'''
    print("Implementation:")
    print(code)
    
    print("=" * 60)
    print("\n2. EUCLIDEAN DISTANCE (L2)\n")
    
    print("Measures: Straight-line distance between points")
    print("Range: 0 (identical) to infinity")
    print("Usage: When magnitude matters\n")
    
    euclidean_v1_v2 = np.linalg.norm(v1 - v2)
    euclidean_v1_v3 = np.linalg.norm(v1 - v3)
    
    print(f"Example:")
    print(f"  euclidean_distance(v1, v2) = {euclidean_v1_v2:.4f}")
    print(f"  euclidean_distance(v1, v3) = {euclidean_v1_v3:.4f}")
    print()
    
    code = '''
def euclidean_distance(a, b):
    """L2 distance: sqrt(sum((a_i - b_i)^2))"""
    return np.linalg.norm(a - b)
'''
    print("Implementation:")
    print(code)
    
    print("=" * 60)
    print("\n3. DOT PRODUCT (Inner Product)\n")
    
    print("Measures: Projection of one vector onto another")
    print("Range: -infinity to infinity")
    print("Usage: Pre-normalized vectors (like cosine but faster)\n")
    
    dot_v1_v2 = np.dot(v1, v2)
    dot_v1_v3 = np.dot(v1, v3)
    
    print(f"Example:")
    print(f"  dot_product(v1, v2) = {dot_v1_v2:.4f}")
    print(f"  dot_product(v1, v3) = {dot_v1_v3:.4f}")
    print()
    
    print("Note: If vectors are normalized, dot product = cosine similarity")
    print("      (but faster, no division needed)")
    
    print("\n" + "=" * 60)
    print("\nChoosing a Distance Metric:\n")
    
    guidance = [
        ('Text embeddings', 'Cosine (direction matters, not length)'),
        ('Image embeddings', 'Cosine or L2'),
        ('Normalized vectors', 'Dot product (fastest)'),
        ('Need magnitude', 'Euclidean'),
        ('High-dimensional', 'Cosine (handles curse better)'),
    ]
    
    print(f"{'Use Case':<25} {'Recommended Metric'}")
    print("-" * 60)
    for use_case, metric in guidance:
        print(f"{use_case:<25} {metric}")

distance_metrics()
```

### ANN Algorithms

```python
def ann_algorithms():
    """Approximate Nearest Neighbor algorithms."""
    
    print("\n\nApproximate Nearest Neighbors (ANN):\n")
    
    print("=" * 60)
    print("\nWhy Approximate?\n")
    
    print("""
Exact nearest neighbor search is O(n):
- Must compare query to ALL vectors
- For 1M vectors, 1M comparisons per query
- Too slow for real-time applications

Approximate Nearest Neighbor (ANN):
- Trade perfect accuracy for speed
- Typically 95-99% accuracy with 100x speedup
- O(log n) or O(sqrt n) complexity
""")
    
    print("=" * 60)
    print("\nAccuracy vs Speed Trade-off:\n")
    
    tradeoff = """
┌─────────────────────────────────────────────┐
│           Accuracy vs Speed                 │
│                                             │
│  100% ●                                     │
│       │  ╲                                  │
│   95% │    ●╲                               │
│       │       ●╲                            │
│   90% │          ●╲                         │
│       │             ●                       │
│   85% │               ●────────────         │
│       └─────────────────────────────────    │
│         1ms   10ms  50ms  100ms  1s         │
│               Query Latency                 │
│                                             │
│  Sweet Spot: 95-99% accuracy @ 10-50ms     │
└─────────────────────────────────────────────┘
"""
    print(tradeoff)
    
    print("\nCommon ANN Algorithms:\n")
    
    algorithms = {
        'HNSW': {
            'name': 'Hierarchical Navigable Small World',
            'speed': 'Very Fast',
            'accuracy': 'High (95-99%)',
            'memory': 'High (stores graph)',
            'build_time': 'Medium',
            'best_for': 'Most use cases, good all-around'
        },
        'IVF': {
            'name': 'Inverted File Index',
            'speed': 'Fast',
            'accuracy': 'Medium-High (90-95%)',
            'memory': 'Low',
            'build_time': 'Fast',
            'best_for': 'Large datasets, memory constrained'
        },
        'LSH': {
            'name': 'Locality Sensitive Hashing',
            'speed': 'Very Fast',
            'accuracy': 'Medium (85-90%)',
            'memory': 'Low',
            'build_time': 'Very Fast',
            'best_for': 'Streaming data, quick updates'
        },
        'Product Quantization': {
            'name': 'PQ',
            'speed': 'Fast',
            'accuracy': 'Medium (90-95%)',
            'memory': 'Very Low (compressed)',
            'build_time': 'Medium',
            'best_for': 'Huge datasets (billions of vectors)'
        }
    }
    
    for algo_id, details in algorithms.items():
        print(f"{algo_id}: {details['name']}")
        print(f"  Speed: {details['speed']}")
        print(f"  Accuracy: {details['accuracy']}")
        print(f"  Memory: {details['memory']}")
        print(f"  Build Time: {details['build_time']}")
        print(f"  Best for: {details['best_for']}")
        print()

ann_algorithms()
```

## Indexing Strategies

### HNSW (Hierarchical Navigable Small World)

```python
def hnsw_explained():
    """HNSW indexing strategy."""
    
    print("HNSW: Hierarchical Navigable Small World\n")
    
    print("=" * 60)
    print("\nCore Idea:\n")
    
    print("""
Build a multi-layer graph where:
- Each vector is a node
- Edges connect similar vectors
- Upper layers: sparse, long-range connections (highway)
- Lower layers: dense, short-range connections (local roads)

Search process:
1. Start at top layer (sparse)
2. Navigate to nearest node
3. Descend to next layer
4. Repeat until bottom layer
5. Refine to exact nearest neighbors
""")
    
    visualization = """
Layer 2 (Top):     ●─────────────●
                   │               │
                   │               │
Layer 1:      ●────●────●     ●────●────●
              │    │    │     │    │    │
              │    │    │     │    │    │
Layer 0:   ●──●──●─●──●─●──●──●──●─●──●──●
           (Bottom layer has all vectors)

Search path: Start top → navigate coarse → descend → refine
"""
    
    print(visualization)
    
    print("=" * 60)
    print("\nCharacteristics:\n")
    
    characteristics = {
        'Search Complexity': 'O(log n)',
        'Build Complexity': 'O(n log n)',
        'Memory Usage': 'High (stores graph structure)',
        'Search Accuracy': '95-99%',
        'Search Speed': 'Very fast (<10ms for millions)',
        'Insert Speed': 'Medium (need to update graph)',
        'Best For': 'Static or slowly-changing datasets'
    }
    
    for char, value in characteristics.items():
        print(f"  {char}: {value}")
    
    print("\n" + "=" * 60)
    print("\nTuning Parameters:\n")
    
    params = [
        ('M', 'Max connections per node', '16-48', 'Higher = better accuracy, more memory'),
        ('ef_construction', 'Search during build', '100-200', 'Higher = better quality, slower build'),
        ('ef_search', 'Search during query', '50-200', 'Higher = better accuracy, slower search'),
    ]
    
    print(f"{'Parameter':<20} {'Description':<25} {'Typical':<10} {'Trade-off'}")
    print("-" * 85)
    for param, desc, typical, tradeoff in params:
        print(f"{param:<20} {desc:<25} {typical:<10} {tradeoff}")
    
    print("\n" + "=" * 60)
    print("\nExample Configuration:\n")
    
    config = '''
# FAISS HNSW configuration
import faiss

dimension = 768  # embedding dimension
M = 32          # connections per node
efConstruction = 200  # build-time search
efSearch = 100   # query-time search

# Create index
index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.efConstruction = efConstruction
index.hnsw.efSearch = efSearch

# Add vectors
vectors = ...  # shape (n, dimension)
index.add(vectors)

# Search
k = 5  # top-k results
query = ...  # shape (1, dimension)
distances, indices = index.search(query, k)
'''
    print(config)

hnsw_explained()
```

### IVF (Inverted File Index)

```python
def ivf_explained():
    """IVF indexing strategy."""
    
    print("\n\nIVF: Inverted File Index\n")
    
    print("=" * 60)
    print("\nCore Idea:\n")
    
    print("""
Partition vector space into clusters (using k-means):
1. Build: Cluster all vectors into n_clusters groups
2. Store: Each vector in its closest cluster
3. Search: 
   - Find closest clusters to query (cheap)
   - Search only within those clusters (focused)
   - Reduces search space dramatically
""")
    
    visualization = """
Vector Space (2D visualization):

Cluster 1:  ●●●●        Cluster 2:    ●●●
            ●●●●                      ●●●●
            ●●●                       ●●

Cluster 3:     ●●●●     Cluster 4:        ●●●●
               ●●●●                        ●●●
               ●●                          ●●●

Query (★):
1. Find closest clusters: 2 and 4
2. Search only in clusters 2 & 4 (not 1 & 3)
3. Much faster than searching all vectors
"""
    
    print(visualization)
    
    print("=" * 60)
    print("\nCharacteristics:\n")
    
    characteristics = {
        'Search Complexity': 'O(n_clusters + vectors_per_cluster)',
        'Build Complexity': 'O(n) (k-means clustering)',
        'Memory Usage': 'Low (just cluster assignments)',
        'Search Accuracy': '90-95% (depends on nprobe)',
        'Search Speed': 'Fast (searches subset)',
        'Insert Speed': 'Fast (just assign to cluster)',
        'Best For': 'Large datasets, memory-constrained'
    }
    
    for char, value in characteristics.items():
        print(f"  {char}: {value}")
    
    print("\n" + "=" * 60)
    print("\nTuning Parameters:\n")
    
    params = [
        ('n_clusters', 'Number of clusters', 'sqrt(n) to n/1000', 'More = finer partitions'),
        ('nprobe', 'Clusters to search', '1-100', 'More = better accuracy, slower'),
    ]
    
    print(f"{'Parameter':<15} {'Description':<25} {'Typical':<20} {'Trade-off'}")
    print("-" * 75)
    for param, desc, typical, tradeoff in params:
        print(f"{param:<15} {desc:<25} {typical:<20} {tradeoff}")
    
    print("\n" + "=" * 60)
    print("\nExample Configuration:\n")
    
    config = '''
# FAISS IVF configuration
import faiss

dimension = 768
n_vectors = 1_000_000
n_clusters = 1000  # typically sqrt(n) to n/1000
nprobe = 10       # search top-10 closest clusters

# Create quantizer for clustering
quantizer = faiss.IndexFlatL2(dimension)

# Create IVF index
index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)

# Train (cluster the data)
index.train(vectors)

# Add vectors
index.add(vectors)

# Search
index.nprobe = nprobe
distances, indices = index.search(query, k=5)

# Accuracy vs speed trade-off
# nprobe=1:  90% accuracy, 1ms
# nprobe=10: 95% accuracy, 5ms
# nprobe=50: 98% accuracy, 15ms
'''
    print(config)

ivf_explained()
```

### LSH (Locality Sensitive Hashing)

```python
def lsh_explained():
    """LSH indexing strategy."""
    
    print("\n\nLSH: Locality Sensitive Hashing\n")
    
    print("=" * 60)
    print("\nCore Idea:\n")
    
    print("""
Hash vectors such that similar vectors likely hash to same bucket:
1. Build: Create hash functions that preserve similarity
2. Hash: Map each vector to hash buckets
3. Search: Hash query, search only in matching buckets

Key property: P(hash(a) == hash(b)) ∝ similarity(a, b)
""")
    
    visualization = """
Hash Functions:

h1:  ───────────│───────────  (split space)
h2:  ────│──────────────────  (another split)
h3:  ─────────────────│─────  (another split)

Buckets created by combinations:
Bucket 1: h1=0, h2=0, h3=0  →  ●●●●
Bucket 2: h1=0, h2=0, h3=1  →  ●●
Bucket 3: h1=0, h2=1, h3=0  →  ●●●
...

Similar vectors → same bucket (high probability)
"""
    
    print(visualization)
    
    print("=" * 60)
    print("\nCharacteristics:\n")
    
    characteristics = {
        'Search Complexity': 'O(1) to O(log n)',
        'Build Complexity': 'O(n)',
        'Memory Usage': 'Low (just hash tables)',
        'Search Accuracy': '85-90%',
        'Search Speed': 'Very fast (hash lookup)',
        'Insert Speed': 'Very fast (hash and add)',
        'Best For': 'Streaming data, frequent updates'
    }
    
    for char, value in characteristics.items():
        print(f"  {char}: {value}")
    
    print("\n" + "=" * 60)
    print("\nTuning Parameters:\n")
    
    params = [
        ('num_tables', 'Number of hash tables', '10-50', 'More = better recall, more memory'),
        ('hash_size', 'Bits per hash', '8-32', 'Larger = more buckets'),
        ('probe_level', 'Nearby buckets', '0-2', 'Higher = better recall, slower'),
    ]
    
    print(f"{'Parameter':<15} {'Description':<25} {'Typical':<15} {'Trade-off'}")
    print("-" * 75)
    for param, desc, typical, tradeoff in params:
        print(f"{param:<15} {desc:<25} {typical:<15} {tradeoff}")
    
    print("\n" + "=" * 60)
    print("\nExample Usage:\n")
    
    code = '''
from datasketch import MinHashLSH, MinHash

# Create LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=128)

# Add vectors (represented as sets for MinHash)
for i, doc_set in enumerate(documents):
    m = MinHash(num_perm=128)
    for item in doc_set:
        m.update(item.encode('utf-8'))
    lsh.insert(f"doc_{i}", m)

# Query
query_minhash = MinHash(num_perm=128)
for item in query_set:
    query_minhash.update(item.encode('utf-8'))

# Find similar documents
similar = lsh.query(query_minhash)
print(f"Similar documents: {similar}")
'''
    print(code)

lsh_explained()
```

## Popular Vector Databases

### Solutions Overview

```python
def vector_database_solutions():
    """Overview of popular vector database solutions."""
    
    print("Popular Vector Database Solutions:\n")
    
    solutions = {
        'Pinecone': {
            'type': 'Managed cloud service',
            'index': 'Proprietary (similar to HNSW)',
            'scale': 'Billions of vectors',
            'features': 'Metadata filtering, namespaces, hybrid search',
            'pricing': 'Pay-as-you-go, $0.096/hour starter',
            'pros': 'Easy, no ops, great performance',
            'cons': 'Expensive at scale, vendor lock-in',
            'best_for': 'Production apps, don\'t want to manage'
        },
        'Weaviate': {
            'type': 'Open-source (self-host or cloud)',
            'index': 'HNSW',
            'scale': 'Millions to billions',
            'features': 'GraphQL, hybrid search, modules',
            'pricing': 'Free (self-hosted) or cloud pricing',
            'pros': 'Flexible, great features, open source',
            'cons': 'Need to manage if self-hosted',
            'best_for': 'Flexible deployment, advanced features'
        },
        'Chroma': {
            'type': 'Open-source, embedded',
            'index': 'HNSW (via hnswlib)',
            'scale': 'Thousands to millions',
            'features': 'Simple API, embeds in app',
            'pricing': 'Free',
            'pros': 'Super easy, no server needed',
            'cons': 'Limited scale, single machine',
            'best_for': 'Development, small scale, prototyping'
        },
        'FAISS': {
            'type': 'Library (not a database)',
            'index': 'HNSW, IVF, PQ, and more',
            'scale': 'Billions (with optimization)',
            'features': 'Many index types, GPU support',
            'pricing': 'Free',
            'pros': 'Fastest, most flexible, free',
            'cons': 'No database features, just search',
            'best_for': 'Custom solutions, maximum performance'
        },
        'Qdrant': {
            'type': 'Open-source (self-host or cloud)',
            'index': 'HNSW',
            'scale': 'Millions to billions',
            'features': 'Rich filtering, payloads, snapshots',
            'pricing': 'Free (self-hosted) or cloud pricing',
            'pros': 'Fast, Rust-based, great filtering',
            'cons': 'Newer, smaller ecosystem',
            'best_for': 'High performance, rich filtering'
        },
        'Milvus': {
            'type': 'Open-source (self-host or cloud)',
            'index': 'Multiple (HNSW, IVF, etc.)',
            'scale': 'Billions',
            'features': 'Distributed, multiple indexes, GPU',
            'pricing': 'Free (self-hosted) or Zilliz cloud',
            'pros': 'Scalable, feature-rich, cloud-native',
            'cons': 'Complex setup, heavy',
            'best_for': 'Large-scale enterprise'
        },
        'pgvector': {
            'type': 'PostgreSQL extension',
            'index': 'IVF or HNSW',
            'scale': 'Thousands to millions',
            'features': 'SQL interface, ACID, existing Postgres',
            'pricing': 'Free',
            'pros': 'Use existing Postgres, SQL, simple',
            'cons': 'Not as fast as specialized DBs',
            'best_for': 'Already using Postgres, simplicity'
        }
    }
    
    print(f"{'Solution':<12} {'Type':<25} {'Index':<20} {'Best For'}")
    print("=" * 90)
    
    for name, details in solutions.items():
        print(f"{name:<12} {details['type']:<25} {details['index']:<20} {details['best_for']}")
    
    print("\n" + "=" * 60)
    print("\nDetailed Comparison:\n")
    
    for name, details in solutions.items():
        print(f"{name}:")
        print(f"  Type: {details['type']}")
        print(f"  Index: {details['index']}")
        print(f"  Scale: {details['scale']}")
        print(f"  Features: {details['features']}")
        print(f"  Pricing: {details['pricing']}")
        print(f"  Pros: {details['pros']}")
        print(f"  Cons: {details['cons']}")
        print(f"  Best for: {details['best_for']}")
        print()

vector_database_solutions()
```

### Quick Comparison Table

```python
def vector_db_comparison_table():
    """Side-by-side comparison of vector databases."""
    
    print("\n\nVector Database Comparison:\n")
    
    print("EASE OF USE:")
    print("  ★★★★★ Chroma, Pinecone")
    print("  ★★★★☆ Weaviate, Qdrant")
    print("  ★★★☆☆ Milvus, pgvector")
    print("  ★★☆☆☆ FAISS (requires more code)")
    print()
    
    print("PERFORMANCE:")
    print("  ★★★★★ FAISS (raw speed)")
    print("  ★★★★☆ Pinecone, Qdrant, Milvus")
    print("  ★★★☆☆ Weaviate, Chroma")
    print("  ★★☆☆☆ pgvector")
    print()
    
    print("SCALE:")
    print("  ★★★★★ Pinecone, Milvus (billions)")
    print("  ★★★★☆ FAISS, Weaviate, Qdrant (billions)")
    print("  ★★★☆☆ Chroma, pgvector (millions)")
    print()
    
    print("COST:")
    print("  Free: FAISS, Chroma, Weaviate, Qdrant, Milvus, pgvector (self-hosted)")
    print("  Paid: Pinecone, Weaviate Cloud, Qdrant Cloud, Zilliz")
    print()
    
    print("DEPLOYMENT:")
    print("  Embedded: Chroma (in-process)")
    print("  Self-hosted: FAISS, Weaviate, Qdrant, Milvus, pgvector")
    print("  Managed: Pinecone, Weaviate Cloud, Qdrant Cloud, Zilliz")
    print()
    
    print("=" * 60)
    print("\nDecision Guide:\n")
    
    decision_tree = [
        ('Prototyping/development', '→', 'Chroma (easy, embedded)'),
        ('Production, managed service', '→', 'Pinecone (no ops)'),
        ('Production, self-hosted', '→', 'Qdrant or Weaviate'),
        ('Maximum performance', '→', 'FAISS (+ custom wrapper)'),
        ('Already using Postgres', '→', 'pgvector'),
        ('Enterprise scale', '→', 'Milvus or Pinecone'),
        ('Budget constrained', '→', 'Open source (Chroma, Qdrant, Weaviate)'),
    ]
    
    for scenario, arrow, recommendation in decision_tree:
        print(f"  {scenario:<30} {arrow} {recommendation}")

vector_db_comparison_table()
```

## Choosing a Vector Database

### Selection Criteria

```python
def selection_criteria():
    """Criteria for choosing a vector database."""
    
    print("Choosing a Vector Database:\n")
    
    print("=" * 60)
    print("\nKey Questions:\n")
    
    questions = [
        ('1. Scale', [
            'How many vectors? (thousands, millions, billions)',
            'Query volume? (requests/second)',
            'Growth rate? (static, slow, fast)'
        ]),
        ('2. Performance', [
            'Latency requirements? (ms)',
            'Throughput needs? (QPS)',
            'Accuracy requirements? (90%, 95%, 99%)'
        ]),
        ('3. Operations', [
            'Managed service or self-hosted?',
            'DevOps resources available?',
            'Cloud provider preference?'
        ]),
        ('4. Features', [
            'Metadata filtering needed?',
            'Hybrid search (vector + keyword)?',
            'Multi-tenancy/namespaces?'
        ]),
        ('5. Budget', [
            'Monthly budget?',
            'Cost per query acceptable?',
            'Infrastructure costs?'
        ]),
        ('6. Integration', [
            'Programming language?',
            'Existing infrastructure?',
            'Framework compatibility?'
        ])
    ]
    
    for category, items in questions:
        print(f"{category}:")
        for item in items:
            print(f"  • {item}")
        print()
    
    print("=" * 60)
    print("\nDecision Matrix:\n")
    
    scenarios = {
        'Startup MVP': {
            'scale': 'Small (10k-100k vectors)',
            'budget': 'Low',
            'ops': 'Minimal',
            'recommendation': 'Chroma (embedded, free, easy)'
        },
        'Growing Product': {
            'scale': 'Medium (100k-1M vectors)',
            'budget': 'Medium',
            'ops': 'Limited',
            'recommendation': 'Pinecone (managed) or Qdrant (self-hosted)'
        },
        'Enterprise Scale': {
            'scale': 'Large (1M-1B+ vectors)',
            'budget': 'High',
            'ops': 'Available',
            'recommendation': 'Milvus (self-hosted) or Pinecone (managed)'
        },
        'Maximum Performance': {
            'scale': 'Any',
            'budget': 'Variable',
            'ops': 'Strong team',
            'recommendation': 'FAISS + custom infrastructure'
        },
        'Existing Postgres': {
            'scale': 'Small-Medium',
            'budget': 'Low',
            'ops': 'Postgres team',
            'recommendation': 'pgvector extension'
        }
    }
    
    for scenario, details in scenarios.items():
        print(f"{scenario}:")
        print(f"  Scale: {details['scale']}")
        print(f"  Budget: {details['budget']}")
        print(f"  Ops: {details['ops']}")
        print(f"  ➜ Recommendation: {details['recommendation']}")
        print()

selection_criteria()
```

## Implementation Examples

### Chroma Example

```python
def chroma_example():
    """Using Chroma vector database."""
    
    print("Chroma Implementation Example:\n")
    
    code = '''
import chromadb
from chromadb.config import Settings

# Create client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create collection
collection = client.create_collection(
    name="documents",
    metadata={"description": "Document collection"}
)

# Add documents
documents = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "France is in Europe."
]

metadata = [
    {"source": "geography", "topic": "cities"},
    {"source": "landmarks", "topic": "monuments"},
    {"source": "geography", "topic": "continents"}
]

ids = ["doc1", "doc2", "doc3"]

collection.add(
    documents=documents,
    metadatas=metadata,
    ids=ids
)

# Query
results = collection.query(
    query_texts=["Tell me about Paris"],
    n_results=2
)

print("Results:", results['documents'])
print("Distances:", results['distances'])
print("Metadata:", results['metadatas'])

# Query with metadata filter
results = collection.query(
    query_texts=["European cities"],
    n_results=5,
    where={"topic": "cities"}  # Filter by metadata
)

# Update documents
collection.update(
    ids=["doc1"],
    documents=["Paris is the beautiful capital of France."]
)

# Delete documents
collection.delete(ids=["doc3"])

# Persist (if not auto-persisting)
client.persist()
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nChroma Features:\n")
    
    features = [
        'Embedded (runs in your process)',
        'No server needed for small scale',
        'Auto-embedding with built-in models',
        'Metadata filtering',
        'Persistent storage',
        'Easy to use API',
        'Great for prototyping'
    ]
    
    for feature in features:
        print(f"  • {feature}")

chroma_example()
```

### Pinecone Example

```python
def pinecone_example():
    """Using Pinecone vector database."""
    
    print("\n\nPinecone Implementation Example:\n")
    
    code = '''
import pinecone
from sentence_transformers import SentenceTransformer

# Initialize
pinecone.init(
    api_key="your-api-key",
    environment="us-west1-gcp"
)

# Create index
index_name = "documents"
dimension = 384  # embedding dimension

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine"
    )

# Connect to index
index = pinecone.Index(index_name)

# Prepare embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "France is in Europe."
]

embeddings = model.encode(documents)

# Upsert (insert/update) vectors
vectors = []
for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
    vectors.append({
        "id": f"doc{i}",
        "values": embedding.tolist(),
        "metadata": {
            "text": doc,
            "source": "geography"
        }
    })

index.upsert(vectors=vectors)

# Query
query = "Tell me about Paris"
query_embedding = model.encode([query])[0]

results = index.query(
    vector=query_embedding.tolist(),
    top_k=3,
    include_metadata=True
)

for match in results['matches']:
    print(f"Score: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text']}")
    print()

# Query with metadata filter
results = index.query(
    vector=query_embedding.tolist(),
    top_k=3,
    filter={"source": "geography"},
    include_metadata=True
)

# Delete vectors
index.delete(ids=["doc0", "doc1"])

# Get index stats
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nPinecone Features:\n")
    
    features = [
        'Fully managed (no ops)',
        'Auto-scaling',
        'High performance (low latency)',
        'Metadata filtering and sparse vectors',
        'Namespaces for multi-tenancy',
        'Good documentation and support',
        'Production-ready'
    ]
    
    for feature in features:
        print(f"  • {feature}")

pinecone_example()
```

### FAISS Example

```python
def faiss_example():
    """Using FAISS for vector search."""
    
    print("\n\nFAISS Implementation Example:\n")
    
    code = '''
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Prepare data
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "France is in Europe.",
    "London is the capital of England.",
    "The Thames flows through London."
]

# Generate embeddings
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]  # 384
n_vectors = embeddings.shape[0]  # 5

print(f"Embeddings shape: {embeddings.shape}")

# Create FAISS index (HNSW)
M = 32  # number of connections
index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 100

# Add vectors
index.add(embeddings)

print(f"Index size: {index.ntotal} vectors")

# Search
query = "Tell me about Paris"
query_embedding = model.encode([query]).astype('float32')

k = 3  # top-k results
distances, indices = index.search(query_embedding, k)

print(f"\\nQuery: {query}\\n")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"Result {i+1}:")
    print(f"  Distance: {dist:.4f}")
    print(f"  Document: {documents[idx]}")
    print()

# Save/load index
faiss.write_index(index, "documents.index")
index_loaded = faiss.read_index("documents.index")

# More index types:

# 1. Flat (exact search, slow but accurate)
index_flat = faiss.IndexFlatL2(dimension)
index_flat.add(embeddings)

# 2. IVF (faster, approximate)
n_clusters = 10
quantizer = faiss.IndexFlatL2(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
index_ivf.train(embeddings)  # IVF needs training
index_ivf.add(embeddings)
index_ivf.nprobe = 3  # search 3 clusters

# 3. Product Quantization (compressed)
m = 8  # number of subquantizers
bits = 8  # bits per subquantizer
index_pq = faiss.IndexPQ(dimension, m, bits)
index_pq.train(embeddings)
index_pq.add(embeddings)

# 4. GPU index (if available)
# res = faiss.StandardGpuResources()
# index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nFAISS Features:\n")
    
    features = [
        'Fastest vector search library',
        'Many index types (HNSW, IVF, PQ, etc.)',
        'GPU support for massive speedup',
        'Highly optimized (C++)',
        'Free and open-source',
        'Production-proven (Meta)',
        'Requires custom wrapper for DB features'
    ]
    
    for feature in features:
        print(f"  • {feature}")

faiss_example()
```

### Qdrant Example

```python
def qdrant_example():
    """Using Qdrant vector database."""
    
    print("\n\nQdrant Implementation Example:\n")
    
    code = '''
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Initialize client
client = QdrantClient(host="localhost", port=6333)
# Or: client = QdrantClient(":memory:")  # in-memory for testing
# Or: client = QdrantClient(url="https://your-cluster.qdrant.io")

# Create collection
collection_name = "documents"
dimension = 384

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=dimension,
        distance=Distance.COSINE
    )
)

# Prepare data
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    {"id": 1, "text": "Paris is the capital of France.", "category": "geography"},
    {"id": 2, "text": "The Eiffel Tower is in Paris.", "category": "landmarks"},
    {"id": 3, "text": "France is in Europe.", "category": "geography"}
]

# Upsert points
points = []
for doc in documents:
    embedding = model.encode(doc["text"])
    
    points.append(PointStruct(
        id=doc["id"],
        vector=embedding.tolist(),
        payload={
            "text": doc["text"],
            "category": doc["category"]
        }
    ))

client.upsert(
    collection_name=collection_name,
    points=points
)

# Search
query = "Tell me about Paris"
query_embedding = model.encode(query)

results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding.tolist(),
    limit=3
)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.payload['text']}")
    print(f"Category: {result.payload['category']}")
    print()

# Search with filter
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding.tolist(),
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="geography")
            )
        ]
    ),
    limit=3
)

# Scroll (get all points)
all_points = client.scroll(
    collection_name=collection_name,
    limit=100
)

# Delete points
client.delete(
    collection_name=collection_name,
    points_selector=[1, 2]
)

# Get collection info
info = client.get_collection(collection_name)
print(f"Vectors in collection: {info.vectors_count}")

# Create snapshot (backup)
snapshot = client.create_snapshot(collection_name)
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nQdrant Features:\n")
    
    features = [
        'Written in Rust (high performance)',
        'Rich filtering with payload',
        'Advanced query features',
        'Snapshots and backups',
        'Distributed deployment',
        'REST and gRPC APIs',
        'Self-hosted or cloud'
    ]
    
    for feature in features:
        print(f"  • {feature}")

qdrant_example()
```

## Performance Optimization

### Optimization Strategies

```python
def performance_optimization():
    """Strategies for optimizing vector database performance."""
    
    print("Performance Optimization:\n")
    
    print("=" * 60)
    print("\n1. INDEX OPTIMIZATION\n")
    
    index_tips = [
        ('Choose right index type', 'HNSW for most cases, IVF for huge datasets'),
        ('Tune index parameters', 'Higher M/ef = better accuracy, more resources'),
        ('Balance accuracy vs speed', 'Don\'t over-optimize for 100% accuracy'),
        ('Use quantization', 'PQ for huge datasets (trade accuracy for space)'),
    ]
    
    for tip, description in index_tips:
        print(f"  • {tip}: {description}")
    
    print("\n" + "=" * 60)
    print("\n2. QUERY OPTIMIZATION\n")
    
    query_tips = [
        ('Reduce top-k', 'Retrieve fewer vectors (3-5 usually enough)'),
        ('Use metadata filters', 'Pre-filter to reduce search space'),
        ('Batch queries', 'Process multiple queries together'),
        ('Cache common queries', 'Store results for frequent queries'),
    ]
    
    for tip, description in query_tips:
        print(f"  • {tip}: {description}")
    
    print("\n" + "=" * 60)
    print("\n3. EMBEDDING OPTIMIZATION\n")
    
    embedding_tips = [
        ('Use smaller embeddings', '384-dim vs 768-dim (faster, less storage)'),
        ('Normalize vectors', 'Enables dot product (faster than cosine)'),
        ('Batch encoding', 'Encode multiple texts together'),
        ('Cache embeddings', 'Don\'t re-embed same texts'),
    ]
    
    for tip, description in embedding_tips:
        print(f"  • {tip}: {description}")
    
    print("\n" + "=" * 60)
    print("\n4. INFRASTRUCTURE OPTIMIZATION\n")
    
    infra_tips = [
        ('Use SSD storage', 'Much faster than HDD'),
        ('Add more RAM', 'Keep index in memory'),
        ('Use GPU', 'For embedding and search (10-100x faster)'),
        ('Deploy close to users', 'Reduce network latency'),
    ]
    
    for tip, description in infra_tips:
        print(f"  • {tip}: {description}")
    
    print("\n" + "=" * 60)
    print("\nLatency Breakdown Example:\n")
    
    breakdown = """
Total Query Latency: 45ms

Breakdown:
  Embedding generation:  20ms  (44%)  ← Use faster model or GPU
  Network (to vector DB): 5ms  (11%)  ← Deploy closer
  Vector search:         15ms  (33%)  ← Optimize index
  Network (from DB):      3ms   (7%)
  Post-processing:        2ms   (5%)
  
Optimization opportunities:
  1. Faster embedding model: 45ms → 30ms (-15ms)
  2. Reduce top-k (10→5):    30ms → 22ms (-8ms)
  3. Better index params:    22ms → 18ms (-4ms)
  Result: 45ms → 18ms (60% improvement)
"""
    
    print(breakdown)

performance_optimization()
```

### Benchmarking

```python
def benchmarking():
    """Benchmarking vector databases."""
    
    print("\n\nBenchmarking Vector Databases:\n")
    
    code = '''
import time
import numpy as np
from sentence_transformers import SentenceTransformer

def benchmark_vector_db(index, query_embeddings, k=5, num_queries=100):
    """Benchmark vector database performance."""
    
    latencies = []
    
    for query_embedding in query_embeddings[:num_queries]:
        start = time.time()
        
        # Search (adapt to your vector DB API)
        results = index.search(query_embedding, k=k)
        
        end = time.time()
        latencies.append((end - start) * 1000)  # ms
    
    return {
        'mean_latency': np.mean(latencies),
        'median_latency': np.median(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies)
    }

def benchmark_accuracy(index, ground_truth, query_embeddings, k=5):
    """Measure search accuracy (recall@k)."""
    
    recalls = []
    
    for i, query_embedding in enumerate(query_embeddings):
        # Get results from index
        results = index.search(query_embedding, k=k)
        result_ids = set([r.id for r in results])
        
        # Compare to ground truth (exact search)
        true_ids = set(ground_truth[i][:k])
        
        # Calculate recall
        overlap = len(result_ids & true_ids)
        recall = overlap / k
        recalls.append(recall)
    
    return np.mean(recalls)

# Example usage
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate test queries
test_queries = ["query 1", "query 2", ...] * 100
query_embeddings = model.encode(test_queries)

# Benchmark
stats = benchmark_vector_db(index, query_embeddings, k=5)

print("Performance Metrics:")
print(f"  Mean latency: {stats['mean_latency']:.2f}ms")
print(f"  Median latency: {stats['median_latency']:.2f}ms")
print(f"  P95 latency: {stats['p95_latency']:.2f}ms")
print(f"  P99 latency: {stats['p99_latency']:.2f}ms")

# Accuracy
recall = benchmark_accuracy(index, ground_truth, query_embeddings, k=5)
print(f"  Recall@5: {recall:.2%}")
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nKey Metrics:\n")
    
    metrics = [
        ('Latency', 'Time to execute query (p50, p95, p99)'),
        ('Throughput', 'Queries per second (QPS)'),
        ('Recall@k', 'Fraction of true top-k results found'),
        ('Index size', 'Memory/disk space used'),
        ('Build time', 'Time to build index'),
        ('Update speed', 'Time to add/delete vectors'),
    ]
    
    for metric, description in metrics:
        print(f"  • {metric}: {description}")

benchmarking()
```

## Scaling Considerations

### Horizontal Scaling

```python
def scaling_strategies():
    """Strategies for scaling vector databases."""
    
    print("Scaling Vector Databases:\n")
    
    print("=" * 60)
    print("\nScaling Strategies:\n")
    
    print("""
1. VERTICAL SCALING (Scale Up)
   Add more resources to single machine:
   • More RAM (keep index in memory)
   • Faster CPUs (faster search)
   • GPU acceleration (10-100x speedup)
   • SSD storage (faster I/O)
   
   Limits: Single machine capacity (~100M vectors)

2. HORIZONTAL SCALING (Scale Out)
   Distribute across multiple machines:
   
   a) Sharding (partition data):
      • Split vectors across shards
      • Query all shards, merge results
      • Linear scaling
      
   b) Replication (copies):
      • Multiple copies of same data
      • Distribute read load
      • High availability
      
   c) Hybrid (sharding + replication):
      • Best of both
      • Most vector DBs use this

3. HIERARCHICAL SCALING
   Multiple tiers with different trade-offs:
   • Tier 1: Small, fast index (recent data)
   • Tier 2: Larger, slower index (historical)
   • Route queries intelligently
""")
    
    print("=" * 60)
    print("\nSharding Example:\n")
    
    sharding = """
Without Sharding:
┌─────────────────────────────────┐
│      Single Index               │
│      10M vectors                │
│      Query time: 50ms           │
└─────────────────────────────────┘

With 5 Shards:
┌───────────┬───────────┬───────────┬───────────┬───────────┐
│  Shard 1  │  Shard 2  │  Shard 3  │  Shard 4  │  Shard 5  │
│  2M vecs  │  2M vecs  │  2M vecs  │  2M vecs  │  2M vecs  │
│   10ms    │   10ms    │   10ms    │   10ms    │   10ms    │
└───────────┴───────────┴───────────┴───────────┴───────────┘
                         ↓
                    Merge results
                         ↓
              Total time: ~15ms (parallel + merge)
              
Scaling: 10M → 100M vectors (10x shards)
         Latency stays ~same
"""
    
    print(sharding)
    
    print("=" * 60)
    print("\nScaling Challenges:\n")
    
    challenges = [
        ('Query routing', 'Decide which shards to query'),
        ('Result merging', 'Combine results from shards'),
        ('Load balancing', 'Distribute queries evenly'),
        ('Data distribution', 'Partition vectors effectively'),
        ('Consistency', 'Keep replicas in sync'),
        ('Failure handling', 'Replicas for high availability'),
    ]
    
    for challenge, description in challenges:
        print(f"  • {challenge}: {description}")

scaling_strategies()
```

## Advanced Features

### Metadata Filtering

```python
def metadata_filtering():
    """Using metadata filters in vector search."""
    
    print("\n\nMetadata Filtering:\n")
    
    print("Concept:")
    print("  Combine vector similarity with traditional filters")
    print("  1. Filter by metadata (reduce search space)")
    print("  2. Vector search within filtered subset")
    print("  3. Much faster + more relevant results\n")
    
    code = '''
# Example: Pinecone with metadata filter

# Insert with metadata
vectors = [
    {
        "id": "doc1",
        "values": embedding1,
        "metadata": {
            "category": "technology",
            "date": "2024-01-15",
            "author": "Alice",
            "tags": ["AI", "ML"]
        }
    },
    {
        "id": "doc2",
        "values": embedding2,
        "metadata": {
            "category": "sports",
            "date": "2024-01-16",
            "author": "Bob"
        }
    }
]

index.upsert(vectors)

# Query with filter
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={
        "category": {"$eq": "technology"},  # Only tech category
        "date": {"$gte": "2024-01-01"},    # From Jan 2024
        "tags": {"$in": ["AI"]}             # Has AI tag
    }
)

# Complex filters (Qdrant example)
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[  # AND conditions
            FieldCondition(key="category", match=MatchValue(value="tech")),
            FieldCondition(key="score", range=Range(gte=0.8))
        ],
        should=[  # OR conditions
            FieldCondition(key="author", match=MatchValue(value="Alice")),
            FieldCondition(key="author", match=MatchValue(value="Bob"))
        ],
        must_not=[  # NOT conditions
            FieldCondition(key="archived", match=MatchValue(value=True))
        ]
    ),
    limit=10
)
'''
    
    print(code)
    
    print("\n" + "=" * 60)
    print("\nCommon Filter Use Cases:\n")
    
    use_cases = [
        ('Time-based', 'Only recent documents (last 30 days)'),
        ('Multi-tenancy', 'Filter by user_id or tenant_id'),
        ('Category', 'Only specific document types'),
        ('Security', 'Filter by access permissions'),
        ('Quality', 'Only high-confidence results'),
        ('Language', 'Documents in specific language'),
    ]
    
    for use_case, example in use_cases:
        print(f"  • {use_case}: {example}")

metadata_filtering()
```

### Hybrid Search

```python
def hybrid_search():
    """Combining vector and keyword search."""
    
    print("\n\nHybrid Search:\n")
    
    print("=" * 60)
    print("\nConcept:\n")
    
    print("""
Combine two search strategies:
1. Vector search (semantic similarity)
2. Keyword search (exact/BM25)

Why?
  • Vector: Good at semantic understanding
  • Keyword: Good at exact matches, names, IDs
  • Together: Better precision and recall
""")
    
    print("=" * 60)
    print("\nHybrid Search Approaches:\n")
    
    approaches = """
1. RANK FUSION (Most Common)
   
   a) Get top-k from vector search
   b) Get top-k from keyword search
   c) Combine using Reciprocal Rank Fusion (RRF):
   
      score(doc) = Σ 1/(k + rank_i(doc))
   
      Where rank_i is doc's rank in result set i

2. WEIGHTED COMBINATION
   
   score(doc) = α * vector_score + (1-α) * keyword_score
   
   Where α controls vector vs keyword importance

3. TWO-STAGE
   
   a) First stage: Fast retrieval (keyword or coarse vector)
   b) Second stage: Rerank with better model

4. PRE-FILTER
   
   a) Use one method to pre-filter
   b) Use other method on filtered set
"""
    
    print(approaches)
    
    code = '''
def hybrid_search(query: str, top_k: int = 5):
    """Hybrid search combining vector and keyword."""
    
    # Vector search
    query_embedding = embed(query)
    vector_results = vector_db.search(query_embedding, top_k=20)
    
    # Keyword search (BM25)
    keyword_results = bm25_search(query, top_k=20)
    
    # Reciprocal Rank Fusion
    def rrf_score(doc_id, results_list, k=60):
        """Calculate RRF score for a document."""
        score = 0
        for results in results_list:
            for rank, doc in enumerate(results, 1):
                if doc.id == doc_id:
                    score += 1 / (k + rank)
        return score
    
    # Get all unique doc IDs
    all_docs = set()
    for results in [vector_results, keyword_results]:
        all_docs.update([doc.id for doc in results])
    
    # Calculate RRF scores
    doc_scores = []
    for doc_id in all_docs:
        score = rrf_score(doc_id, [vector_results, keyword_results])
        doc_scores.append((doc_id, score))
    
    # Sort and return top-k
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]

# Example usage
results = hybrid_search("machine learning algorithms", top_k=5)

for doc_id, score in results:
    print(f"Doc: {doc_id}, Score: {score:.4f}")
'''
    
    print("\nImplementation:")
    print(code)
    
    print("\n" + "=" * 60)
    print("\nWhen to Use Hybrid Search:\n")
    
    scenarios = [
        ('✓ Need exact matches', 'Product codes, IDs, names'),
        ('✓ Both keywords and concepts', 'Legal, medical docs'),
        ('✓ Improve precision', 'Reduce false positives'),
        ('✗ Simple semantic queries', 'Vector alone sufficient'),
        ('✗ Tight latency budget', 'Hybrid slower than single method'),
    ]
    
    for scenario, description in scenarios:
        print(f"  {scenario}: {description}")

hybrid_search()
```

## Summary

**Key Concepts**:

1. **Vector databases store and search embeddings** - specialized for high-dimensional vectors, enabling semantic search
2. **Core operation**: Find k-nearest neighbors using distance metrics (cosine, L2, dot product)
3. **ANN algorithms trade accuracy for speed** - 95-99% accuracy with 100x speedup over exact search
4. **Three main index types**: HNSW (all-around), IVF (large-scale), LSH (streaming)
5. **Many solutions available**: Pinecone (managed), Weaviate/Qdrant (flexible), Chroma (easy), FAISS (fastest)
6. **Key features**: Metadata filtering, hybrid search, horizontal scaling, high availability

**Distance Metrics**:

| Metric | Formula | Use Case | Range |
|--------|---------|----------|-------|
| Cosine | dot(a,b)/(‖a‖‖b‖) | Text embeddings | -1 to 1 |
| L2 (Euclidean) | ‖a - b‖ | When magnitude matters | 0 to ∞ |
| Dot Product | dot(a, b) | Normalized vectors | -∞ to ∞ |

**Indexing Algorithms**:

```
HNSW: Hierarchical graph structure
- Speed: Very Fast (<10ms)
- Accuracy: 95-99%
- Memory: High
- Best for: Most use cases

IVF: Cluster-based partitioning
- Speed: Fast (10-50ms)
- Accuracy: 90-95%
- Memory: Low
- Best for: Large datasets, memory constrained

LSH: Hash-based bucketing
- Speed: Very Fast (<5ms)
- Accuracy: 85-90%
- Memory: Low
- Best for: Streaming data, frequent updates
```

**Popular Solutions**:

```
Chroma
- Type: Embedded, open-source
- Best for: Prototyping, small scale
- Pros: Easy, no server needed
- Scale: Thousands to millions

Pinecone
- Type: Managed cloud service
- Best for: Production, no ops
- Pros: Scalable, reliable
- Scale: Millions to billions

Qdrant
- Type: Self-hosted or cloud, open-source
- Best for: High performance, rich filtering
- Pros: Fast, Rust-based
- Scale: Millions to billions

FAISS
- Type: Library (not a database)
- Best for: Maximum performance, custom solutions
- Pros: Fastest, many index types, GPU support
- Scale: Billions (with optimization)

Weaviate
- Type: Open-source, self-hosted or cloud
- Best for: Flexible deployment, GraphQL
- Pros: Feature-rich, modular
- Scale: Millions to billions
```

**Choosing a Vector Database**:

```
Scenario                      → Recommendation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prototyping                   → Chroma (easy, embedded)
Production, managed           → Pinecone (no ops)
Production, self-hosted       → Qdrant or Weaviate
Maximum performance           → FAISS + wrapper
Already using Postgres        → pgvector extension
Enterprise scale              → Milvus or Pinecone
Budget constrained            → Open source (Chroma, Qdrant)
```

**Performance Optimization**:

1. **Index tuning**: Balance accuracy vs speed (M, ef parameters for HNSW)
2. **Reduce top-k**: Retrieve fewer vectors (3-5 usually sufficient)
3. **Smaller embeddings**: 384-dim vs 768-dim (faster, less storage)
4. **Normalize vectors**: Enables dot product (faster than cosine)
5. **Batch operations**: Process multiple queries/embeds together
6. **Caching**: Store common query results
7. **GPU acceleration**: 10-100x faster for embeddings and search
8. **Metadata filters**: Pre-filter to reduce search space

**Typical Latency Breakdown**:

```
Total: 45ms
├─ Embedding generation: 20ms (44%)
├─ Network to DB: 5ms (11%)
├─ Vector search: 15ms (33%)
├─ Network from DB: 3ms (7%)
└─ Post-processing: 2ms (5%)

Optimization targets:
- Faster embedding model or GPU
- Reduce top-k
- Optimize index parameters
```

**Scaling Strategies**:

- **Vertical**: Add RAM/CPU/GPU to single machine (~100M vectors)
- **Horizontal**: Shard across machines (billions of vectors)
- **Replication**: Multiple copies for read scaling and HA
- **Hybrid**: Sharding + replication for best results

**Advanced Features**:

- **Metadata filtering**: Combine vector similarity with traditional filters
- **Hybrid search**: Merge vector + keyword results (RRF)
- **Namespaces**: Multi-tenancy within single index
- **Sparse vectors**: Combine dense (semantic) + sparse (keyword) vectors
- **Snapshots**: Backup and restore capabilities

**Performance Targets**:

- Latency: p50 <20ms, p95 <50ms, p99 <100ms
- Accuracy: 95-99% recall@k
- Throughput: 100-1000+ QPS per node
- Scale: Millions to billions of vectors

## Next Steps

- Learn effective [Embedding and Chunking](embedding-chunking.md) strategies for better search quality
- Explore [Retrieval Strategies](retrieval-strategies.md) including dense, sparse, and hybrid approaches
- Master [Reranking and Fusion](reranking-fusion.md) to improve result relevance
- Study [RAG Evaluation](rag-evaluation.md) to measure and optimize system performance
- Review [RAG Architecture](rag-architecture.md) for overall system design
- Apply to production in [Application Patterns](../application_patterns/)
