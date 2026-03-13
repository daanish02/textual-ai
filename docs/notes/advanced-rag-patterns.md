# Advanced RAG Patterns

## Table of Contents

- [Introduction](#introduction)
- [Ensemble Retrieval](#ensemble-retrieval)
  - [Combining Multiple Retrievers](#combining-multiple-retrievers)
  - [Weighted Ensemble](#weighted-ensemble)
  - [Reciprocal Rank Fusion](#reciprocal-rank-fusion)
  - [When to Use Ensemble Retrieval](#when-to-use-ensemble-retrieval)
- [Reranking](#reranking)
  - [Why Reranking Matters](#why-reranking-matters)
  - [Cross-Encoder Reranking](#cross-encoder-reranking)
  - [LLM-Based Reranking](#llm-based-reranking)
  - [Reranking Trade-offs](#reranking-trade-offs)
- [Parent Document Retriever](#parent-document-retriever)
  - [The Small Chunk Problem](#the-small-chunk-problem)
  - [Implementation Pattern](#implementation-pattern)
  - [Chunk Relationships](#chunk-relationships)
  - [Use Cases](#use-cases)
- [Self-Query Retriever](#self-query-retriever)
  - [Natural Language to Filters](#natural-language-to-filters)
  - [Metadata Extraction](#metadata-extraction)
  - [Combining Semantic and Metadata Search](#combining-semantic-and-metadata-search)
  - [Structured Query Translation](#structured-query-translation)
- [Hybrid Search](#hybrid-search)
  - [Dense vs Sparse Retrieval](#dense-vs-sparse-retrieval)
  - [BM25 and Semantic Search](#bm25-and-semantic-search)
  - [Fusion Strategies](#fusion-strategies)
- [RAG Pattern Comparison](#rag-pattern-comparison)
- [Production Considerations](#production-considerations)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Basic RAG (retrieve → generate) works well for straightforward queries, but production systems need more sophisticated patterns to handle diverse queries, improve precision, and provide relevant context. **Advanced RAG patterns** address common limitations:

- **Ensemble Retrieval** - Combine multiple retrieval strategies for better coverage
- **Reranking** - Improve precision by reordering retrieved documents
- **Parent Document Retriever** - Retrieve with small chunks, return large context
- **Self-Query Retriever** - Extract metadata filters from natural language queries
- **Hybrid Search** - Combine dense (semantic) and sparse (keyword) retrieval

These patterns build on the fundamentals from [Data and Retrieval](data-and-retrieval.md) to create more robust, accurate RAG systems.

**Performance improvements:**

```
Basic similarity search:       ████████░░ 75% relevant
+ Reranking:                   ███████████ 88% relevant
+ Ensemble + Reranking:        ████████████ 92% relevant
```

This document covers when and how to use each pattern.

## Ensemble Retrieval

### Combining Multiple Retrievers

Ensemble retrieval runs multiple retrievers in parallel and merges results. Different retrievers excel at different query types -- combining them provides better coverage.

**Basic ensemble pattern:**

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

# Create semantic retriever (dense vectors)
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create keyword retriever (sparse, BM25)
keyword_retriever = BM25Retriever.from_documents(documents)
keyword_retriever.k = 5

# Combine with equal weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)

# Query retrieves from both, merges results
results = ensemble_retriever.invoke("machine learning algorithms")
```

**What happens:**

1. **Parallel execution** - Both retrievers run simultaneously
2. **Score normalization** - Normalize scores to comparable ranges
3. **Weight application** - Apply configured weights to scores
4. **Merge and deduplicate** - Combine results, remove duplicates
5. **Rerank** - Sort by weighted scores

**Flow diagram:**

```
Query: "machine learning algorithms"
        ↓
   ┌────┴────┐
   ↓         ↓
Semantic   BM25
(vectors)  (keywords)
   ↓         ↓
[doc1=0.9] [doc3=0.95]
[doc2=0.8] [doc1=0.85]
[doc5=0.7] [doc7=0.80]
   ↓         ↓
   └────┬────┘
        ↓
  Apply weights (0.5, 0.5)
        ↓
  Merge & deduplicate
        ↓
  [doc1, doc3, doc2, doc7, doc5]
```

### Weighted Ensemble

Adjust weights to favor specific retrievers based on your use case:

```python
# Favor semantic search (70%) over keyword (30%)
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.7, 0.3]
)

# Favor keyword search (30% semantic, 70% keyword)
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.3, 0.7]
)
```

**Weight selection strategies:**

| Use Case            | Semantic Weight | Keyword Weight | Reasoning                    |
| ------------------- | --------------- | -------------- | ---------------------------- |
| General QA          | 0.6             | 0.4            | Slight semantic bias         |
| Technical docs      | 0.5             | 0.5            | Equal weights                |
| Exact term matching | 0.3             | 0.7            | Favor keywords               |
| Conceptual queries  | 0.8             | 0.2            | Strong semantic bias         |
| Entity search       | 0.2             | 0.8            | Keywords find exact entities |

**Tuning weights:**

```python
# Experiment with different weights
test_queries = ["machine learning", "Python pandas", "neural networks"]

for sem_weight in [0.3, 0.5, 0.7]:
    kw_weight = 1 - sem_weight
    retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[sem_weight, kw_weight]
    )

    # Evaluate on test queries
    for query in test_queries:
        results = retriever.invoke(query)
        # Measure relevance, compare
```

### Reciprocal Rank Fusion

Alternative to weighted scoring -- uses rank-based fusion:

```python
from langchain.retrievers import EnsembleRetriever

# RRF algorithm automatically applied
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever1, retriever2, retriever3],
    weights=[0.33, 0.33, 0.34],  # Weights still used
    c=60  # RRF constant (default: 60)
)
```

**How RRF works:**

For each document, compute RRF score:

```
RRF(doc) = Σ(1 / (rank_i + c))
```

Where:

- `rank_i` = rank of document in retriever i (1-indexed)
- `c` = constant to reduce impact of high ranks (default 60)

**Example:**

```
Document appears in 2 retrievers:
- Retriever 1: rank 2
- Retriever 2: rank 5

RRF score = 1/(2+60) + 1/(5+60) = 1/62 + 1/65 = 0.0161 + 0.0154 = 0.0315
```

**Benefits of RRF:**

- **Rank-based** - Less sensitive to score scales
- **Handles missing documents** - Documents not in all retrievers handled gracefully
- **Empirically strong** - Performs well across diverse retrieval tasks
- **No score calibration** - Don't need to normalize scores

### When to Use Ensemble Retrieval

**Use ensemble when:**

- Queries vary widely (some semantic, some keyword-heavy)
- Need comprehensive coverage (don't miss relevant docs)
- Different retrievers capture different aspects (BM25 + semantic + metadata)
- Can afford latency of parallel retrieval
- Have diverse document types (some match keywords, some match concepts)

**Examples:**

**Good use case - Technical documentation:**

```python
# Users search with exact function names (keywords)
# AND conceptual questions (semantic)
ensemble_retriever = EnsembleRetriever(
    retrievers=[
        semantic_retriever,    # "How do I sort a list?"
        keyword_retriever,     # "list.sort() method"
        api_retriever          # Search API references
    ],
    weights=[0.4, 0.4, 0.2]
)
```

**Good use case - Multi-domain knowledge base:**

```python
# Different domains captured by different retrievers
ensemble_retriever = EnsembleRetriever(
    retrievers=[
        product_docs_retriever,    # Product documentation
        support_tickets_retriever, # Historical support issues
        faq_retriever             # Frequently asked questions
    ],
    weights=[0.5, 0.3, 0.2]
)
```

**When NOT to use:**

- Latency critical (parallel retrieval adds overhead)
- Single query type (semantic OR keyword, not both)
- Small document corpus (single retriever sufficient)
- Limited compute (running multiple retrievers expensive)

## Reranking

### Why Reranking Matters

Initial retrieval casts a wide net -- reranking refines results for precision.

**Two-stage retrieval:**

```
Stage 1 (Retrieval):
  Fast, broad - Retrieve top 100 candidates
  Goal: High recall (don't miss relevant docs)
  Method: Embedding similarity, BM25, etc.

Stage 2 (Reranking):
  Slower, precise - Rerank top 100 → select top 5
  Goal: High precision (best docs first)
  Method: Cross-encoder, LLM scoring
```

**Why this works:**

- **Retrieval optimizes for speed** - Fast approximate search
- **Reranking optimizes for accuracy** - Expensive but precise
- **Cost-effective** - Only rerank small candidate set (10-100 docs)

**Performance impact:**

```
Without reranking:
  Retrieved 20 docs → Top 5: 60% relevant

With reranking:
  Retrieved 20 docs → Rerank → Top 5: 85% relevant
                       ↑
               +25% precision improvement
```

### Cross-Encoder Reranking

Cross-encoders score query-document pairs jointly (more accurate than separate embeddings):

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Base retriever (retrieve more than needed)
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Reranker (Cohere cross-encoder)
compressor = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5  # Return top 5 after reranking
)

# Compression retriever wraps base + reranker
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Use like any retriever
query = "How do transformers handle long contexts?"
results = reranking_retriever.invoke(query)
# Returns: Top 5 docs after reranking 20 candidates
```

**Execution flow:**

```
Query
  ↓
Base Retriever (fast)
  └→ Retrieve 20 documents
       ↓
Cross-Encoder Reranker (slow)
  └→ Score each (query, doc) pair
       ↓
Filter & Sort
  └→ Return top 5 highest scores
```

**Why cross-encoders are better:**

**Bi-encoder (retrieval):**

```
Query → Encoder → [q_vec]             ← Computed once
                     ×  (dot product)
Doc   → Encoder → [d_vec]             ← Precomputed, fast
                     ↓
                  Score

Fast but less accurate - query and doc encoded separately
```

**Cross-encoder (reranking):**

```
[Query + Doc] → Joint Encoder → Score

Slow but more accurate - attention between query and doc
```

**Available rerankers:**

- **Cohere Rerank** - High quality, API-based
- **BGE Reranker** - Open source, local inference
- **Jina Reranker** - Multilingual support
- **Colbert** - Late interaction model

**Reranker selection:**

| Reranker      | Quality   | Speed  | Cost             | Use Case                 |
| ------------- | --------- | ------ | ---------------- | ------------------------ |
| Cohere Rerank | Excellent | Medium | API cost         | Production, high quality |
| BGE Reranker  | Good      | Fast   | Free (self-host) | Cost-sensitive           |
| Jina Reranker | Good      | Medium | Free             | Multilingual             |
| Custom LLM    | Variable  | Slow   | High             | Maximum control          |

### LLM-Based Reranking

Use an LLM to score relevance (expensive but flexible):

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_openai import ChatOpenAI

# Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# LLM-based filter
llm = ChatOpenAI(model="gpt-4o-mini")
compressor = LLMChainFilter.from_llm(llm)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

results = reranking_retriever.invoke(query)
```

**How LLM reranking works:**

1. **Retrieve candidates** - Get top 20 documents
2. **For each document** - Ask LLM: "Is this relevant to query?"
3. **Filter or score** - Keep only relevant documents
4. **Return results** - Filtered/reranked documents

**LLM reranking prompt example:**

```
Given the query: "{query}"
And the document: "{document}"

Is this document relevant to the query? Answer YES or NO.
Consider:
- Does it answer the question?
- Does it provide useful context?
- Is the information accurate and current?
```

**When to use LLM reranking:**

- Need custom relevance criteria
- Cross-encoder models not available
- Complex relevance logic (multiple factors)
- Can afford high cost and latency

### Reranking Trade-offs

**Decision matrix:**

| Method        | Latency        | Cost   | Accuracy | Scale                  |
| ------------- | -------------- | ------ | -------- | ---------------------- |
| No reranking  | Fast (100ms)   | Low    | Baseline | Millions of docs       |
| Cross-encoder | Medium (500ms) | Medium | +20-30%  | Hundreds of candidates |
| LLM reranking | Slow (5s)      | High   | +30-40%  | Tens of candidates     |

**Optimization strategies:**

**Hybrid reranking:**

```python
# Stage 1: Retrieve 100 docs (fast)
stage1 = base_retriever.invoke(query, k=100)

# Stage 2: Cross-encoder rerank to 20 (medium)
stage2 = cross_encoder_rerank(stage1, top_n=20)

# Stage 3: LLM rerank to 5 (slow, but only 20 candidates)
stage3 = llm_rerank(stage2, top_n=5)
```

**Cache reranking results:**

```python
# Cache for popular queries
cache = {}

def cached_rerank(query, docs):
    cache_key = hash(query)
    if cache_key in cache:
        return cache[cache_key]

    results = reranker.invoke(query, docs)
    cache[cache_key] = results
    return results
```

## Parent Document Retriever

### The Small Chunk Problem

**Dilemma:**

- **Small chunks** - Better retrieval precision, but lack context
- **Large chunks** - More context, but worse retrieval precision

**Example problem:**

```
Large chunk (2000 chars):
  "...background info... RELEVANT ANSWER ...more background..."

  Problem: Embedding is "averaged" over all content
  → Retrieval might miss it (answer diluted by surrounding text)

Small chunk (200 chars):
  "RELEVANT ANSWER"

  Problem: Retrieved, but lacks context
  → LLM might not understand without surrounding information
```

**Parent Document Retriever solves this:**

- **Embed and search** small chunks (precision)
- **Return** parent documents (context)

### Implementation Pattern

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Store for parent documents
store = InMemoryStore()

# Child splitter (small chunks for retrieval)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

# Parent splitter (larger chunks returned to LLM)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

# Vector store for child chunks
vectorstore = Chroma(
    collection_name="parent_doc_retriever",
    embedding_function=OpenAIEmbeddings()
)

# Parent document retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Add documents - automatically creates child/parent relationship
retriever.add_documents(documents)

# Query with small chunks, get large context
results = retriever.invoke("How do transformers work?")
# Returns: Parent documents (2000 chars each) based on child chunk matches
```

**What happens internally:**

```
Document (10,000 chars)
  ↓
Parent Splitter (2000 char chunks)
  ├─ Parent 1 (chars 0-2000)
  ├─ Parent 2 (chars 1800-3800)
  ├─ Parent 3 (chars 3600-5600)
  └─ Parent 4 (chars 5400-7400)
       ↓
Child Splitter (200 char chunks)
  ├─ Child 1 → Parent 1
  ├─ Child 2 → Parent 1
  ├─ Child 3 → Parent 2
  └─ ... (many child chunks)

Storage:
  Vector Store: [child_1_embedding, child_2_embedding, ...]
  Doc Store: {parent_1_id: parent_1_content, ...}
  Mapping: {child_1: parent_1_id, child_2: parent_1_id, ...}

Query:
  1. Embed query
  2. Find similar child chunks
  3. Look up parent IDs
  4. Return parent documents
```

### Chunk Relationships

**Alternative: Full document retrieval:**

```python
# Retrieve with small chunks, return full document
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=None  # No parent splitter = return full doc
)
```

**Sizing guidelines:**

| Use Case           | Child Size | Parent Size | Ratio |
| ------------------ | ---------- | ----------- | ----- |
| Technical docs     | 200-400    | 1500-2500   | 1:8   |
| Long-form content  | 400-600    | 3000-5000   | 1:8   |
| Dense information  | 150-250    | 1000-1500   | 1:6   |
| Code documentation | 300-500    | 2000-3000   | 1:6   |

**Overlap strategy:**

- **Child overlap:** 10-20% (maintain retrieval continuity)
- **Parent overlap:** 10-15% (maintain context across parents)

### Use Cases

**When to use Parent Document Retriever:**

✅ **Long technical documents**

- Child: Function signatures, key points
- Parent: Full function docs with examples

✅ **Legal/medical documents**

- Child: Specific clauses, symptoms
- Parent: Full legal sections, medical context

✅ **Research papers**

- Child: Key findings, formulas
- Parent: Full sections with methodology

✅ **Code repositories**

- Child: Function definitions
- Parent: Full file or class

**When NOT to use:**

- Documents already optimally sized (500-1000 chars)
- Need precise extraction (not full context)
- Storage constrained (stores both child and parent)
- Simple QA (basic retrieval sufficient)

## Self-Query Retriever

### Natural Language to Filters

Users express metadata filters in natural language:

```
User: "Show me Python tutorials from 2024"
       ↓
System extracts:
  - Semantic: "Python tutorials"
  - Metadata: {language: "python", year: 2024}
```

**Self-Query Retriever** automatically extracts metadata filters from queries.

### Metadata Extraction

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo

# Define metadata schema
metadata_field_info = [
    AttributeInfo(
        name="language",
        description="Programming language (Python, JavaScript, etc.)",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="Publication year",
        type="integer"
    ),
    AttributeInfo(
        name="difficulty",
        description="Difficulty level (beginner, intermediate, advanced)",
        type="string"
    ),
    AttributeInfo(
        name="duration",
        description="Tutorial duration in minutes",
        type="integer"
    )
]

document_content_description = "Programming tutorials and documentation"

# Create vector store with metadata
vectorstore = Chroma.from_documents(
    documents_with_metadata,
    OpenAIEmbeddings()
)

# Self-query retriever
retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

# Natural language queries with implicit filters
results = retriever.invoke("Python tutorials for beginners from 2024")
# Extracts: semantic="Python tutorials", filters={difficulty: "beginner", year: 2024}

results = retriever.invoke("Advanced JavaScript content under 30 minutes")
# Extracts: semantic="JavaScript content", filters={difficulty: "advanced", duration: <30}
```

### Combining Semantic and Metadata Search

**How self-query works:**

```
User Query: "Show me recent Python tutorials for beginners"
  ↓
LLM Analysis:
  Semantic component: "Python tutorials"
  Metadata filters:
    - language = "Python"
    - difficulty = "beginner"
    - year >= 2024 (implied "recent")
  ↓
Structured Query:
  {
    "query": "Python tutorials",
    "filter": {
      "language": {"$eq": "Python"},
      "difficulty": {"$eq": "beginner"},
      "year": {"$gte": 2024}
    }
  }
  ↓
Vector Store Execution:
  1. Semantic search: "Python tutorials"
  2. Apply filters: language="Python" AND difficulty="beginner" AND year>=2024
  3. Return filtered results
```

**Benefits:**

- **Better precision** - Metadata filtering removes irrelevant results
- **User-friendly** - Natural language instead of structured queries
- **Flexible** - Handles various filter types (equality, range, etc.)

### Structured Query Translation

**Comparison operators:**

```python
# User query → translated filter

"tutorials from 2024"              → {year: {"$eq": 2024}}
"tutorials after 2023"             → {year: {"$gt": 2023}}
"tutorials from 2023 or later"     → {year: {"$gte": 2023}}
"short tutorials under 20 minutes" → {duration: {"$lt": 20}}
"not JavaScript"                   → {language: {"$ne": "JavaScript"}}
"Python or JavaScript"             → {language: {"$in": ["Python", "JavaScript"]}}
```

**Logical operators:**

```python
# AND operator (implicit)
"Python tutorials for beginners"
→ {
    "language": "Python",
    "difficulty": "beginner"
  }

# OR operator
"Python or JavaScript tutorials"
→ {"language": {"$in": ["Python", "JavaScript"]}}

# NOT operator
"tutorials not for beginners"
→ {"difficulty": {"$ne": "beginner"}}
```

**Use cases:**

| Query                               | Semantic Component | Metadata Filters                         |
| ----------------------------------- | ------------------ | ---------------------------------------- |
| "Recent Python tutorials"           | "Python tutorials" | {year: >=2024}                           |
| "Beginner ML courses under 2 hours" | "ML courses"       | {difficulty: "beginner", duration: <120} |
| "JavaScript docs, not React"        | "JavaScript docs"  | {framework: {$ne: "React"}}              |
| "Advanced Python published in 2023" | "Python"           | {difficulty: "advanced", year: 2023}     |

**When to use self-query retriever:**

- Rich metadata structure (dates, categories, attributes)
- Users naturally include filters in queries
- Need to combine semantic + structured search
- Metadata significantly improves precision

**When NOT to use:**

- Minimal metadata (just content)
- LLM cost prohibitive (calls LLM for every query)
- Metadata schema changes frequently
- Simple semantic search sufficient

## Hybrid Search

### Dense vs Sparse Retrieval

**Dense (semantic) retrieval:**

- Uses embeddings (vectors)
- Captures semantic meaning
- Good for concept matching
- Example: "ML algorithms" matches "machine learning techniques"

**Sparse (keyword) retrieval:**

- Uses term frequencies (BM25, TF-IDF)
- Exact text matching
- Good for specific terms
- Example: "tensorflow.keras.Model" matches document with exact string

**Complementary strengths:**

| Query Type                                | Dense Better | Sparse Better |
| ----------------------------------------- | ------------ | ------------- |
| "How to sort lists in Python"             | ✓            |               |
| "list.sort() vs sorted()"                 |              | ✓             |
| "Machine learning for beginners"          | ✓            |               |
| "sklearn.ensemble.RandomForestClassifier" |              | ✓             |
| "Explain transformers"                    | ✓            |               |
| "BERT vs GPT-2"                           | ✓            | ✓             |

### BM25 and Semantic Search

Combine BM25 (sparse) and vector search (dense):

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Semantic retriever (dense vectors)
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Keyword retriever (BM25 - sparse)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# Hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# Queries benefit from both approaches
results = hybrid_retriever.invoke("list.sort() method in Python")
# BM25 finds exact term "list.sort()"
# Semantic finds conceptually related sorting docs
```

**How BM25 works:**

```
Term frequency–inverse document frequency with saturation

BM25(q, d) = Σ IDF(qi) × (f(qi, d) × (k1 + 1)) / (f(qi, d) + k1 × (1 - b + b × |d| / avgdl))

Where:
- qi: Query terms
- f(qi, d): Term frequency in document
- |d|: Document length
- avgdl: Average document length
- k1, b: Tuning parameters (typically k1=1.5, b=0.75)
```

**Why BM25 complements embeddings:**

- **Rare terms** - BM25 emphasizes rare, distinctive terms
- **Exact matches** - Critical for technical terms, names, IDs
- **Vocabulary gaps** - BM25 doesn't need term in training vocabulary
- **Transparent** - Explainable (based on term matches)

### Fusion Strategies

**Reciprocal Rank Fusion (covered earlier):**

```python
hybrid = EnsembleRetriever(
    retrievers=[semantic, bm25],
    weights=[0.5, 0.5]  # Equal contribution
)
```

**Adaptive weighting:**

```python
def adaptive_weights(query):
    """Adjust weights based on query characteristics."""
    # More semantic weight for conceptual queries
    if is_conceptual(query):
        return [0.7, 0.3]
    # More BM25 weight for technical/exact queries
    elif has_technical_terms(query):
        return [0.3, 0.7]
    # Equal for mixed queries
    else:
        return [0.5, 0.5]

# Apply dynamically
weights = adaptive_weights(query)
hybrid = EnsembleRetriever(retrievers=[semantic, bm25], weights=weights)
```

**Sequential fusion:**

```python
# Stage 1: BM25 for fast filtering
bm25_results = bm25_retriever.invoke(query, k=50)

# Stage 2: Semantic reranking
semantic_scores = {doc: semantic_similarity(query, doc) for doc in bm25_results}
ranked = sorted(bm25_results, key=lambda d: semantic_scores[d], reverse=True)
return ranked[:10]
```

## RAG Pattern Comparison

**Decision matrix for choosing patterns:**

| Pattern              | Strength              | Weakness             | Best For              | Added Latency |
| -------------------- | --------------------- | -------------------- | --------------------- | ------------- |
| **Basic Similarity** | Simple, fast          | Limited precision    | General QA            | Baseline      |
| **MMR**              | Diversity             | Computation overhead | Avoiding redundancy   | +10-20%       |
| **Ensemble**         | Coverage, robustness  | Higher latency       | Diverse queries       | +30-50%       |
| **Reranking**        | Precision             | Cost, latency        | High-stakes retrieval | +50-200%      |
| **Parent-Doc**       | Context + precision   | Storage overhead     | Long documents        | +10-20%       |
| **Self-Query**       | User-friendly filters | LLM cost per query   | Rich metadata         | +30-50%       |
| **Hybrid**           | Balanced retrieval    | Complexity           | Technical content     | +20-40%       |

**Combination strategies:**

**High-precision RAG:**

```python
# Ensemble → Rerank → Parent-doc
ensemble = EnsembleRetriever([semantic, bm25], weights=[0.6, 0.4])
reranked = ContextualCompressionRetriever(
    base_retriever=ensemble,
    base_compressor=CohereRerank(top_n=10)
)
final = ParentDocumentRetriever(
    vectorstore=...,
    child_splitter=...,
    base_retriever=reranked
)
```

**Balanced RAG:**

```python
# Hybrid → MMR → Light reranking
hybrid = EnsembleRetriever([semantic, bm25])
mmr = vectorstore.as_retriever(search_type="mmr")
final = ContextualCompressionRetriever(
    base_retriever=mmr,
    base_compressor=lightweight_reranker
)
```

**Fast RAG:**

```python
# Simple semantic + parent-doc
parent_doc = ParentDocumentRetriever(...)
# Minimal overhead, good context
```

## Production Considerations

### Performance Optimization

**Caching strategies:**

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieve(query: str, k: int = 5):
    return retriever.invoke(query, k=k)
```

**Async retrieval:**

```python
async def async_ensemble_retrieve(query: str):
    # Parallel retriever execution
    results = await asyncio.gather(
        retriever1.ainvoke(query),
        retriever2.ainvoke(query),
        retriever3.ainvoke(query)
    )
    return merge_results(results)
```

**Query timeout:**

```python
import asyncio

async def retrieve_with_timeout(query, timeout=5):
    try:
        return await asyncio.wait_for(
            retriever.ainvoke(query),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        # Fall back to cached/default results
        return fallback_retrieve(query)
```

### Cost Management

**Tiered retrieval:**

```python
def tiered_retrieval(query, user_tier):
    if user_tier == "premium":
        # Ensemble + reranking (expensive)
        return premium_retriever.invoke(query)
    elif user_tier == "standard":
        # Hybrid search (medium cost)
        return standard_retriever.invoke(query)
    else:
        # Basic similarity (cheap)
        return basic_retriever.invoke(query)
```

**Budget-aware reranking:**

```python
# Only rerank if worth the cost
def smart_rerank(query, initial_results):
    if needs_precision(query):  # Critical query
        return rerank(initial_results)
    else:  # Save cost on simple queries
        return initial_results
```

### Monitoring and Evaluation

**Key metrics:**

```python
# Retrieval metrics
metrics = {
    "latency": measure_latency(),
    "recall@k": calculate_recall(retrieved, relevant),
    "precision@k": calculate_precision(retrieved, relevant),
    "mrr": mean_reciprocal_rank(rankings),
    "cost_per_query": track_api_costs()
}
```

**A/B testing retrievers:**

```python
def ab_test_retrievers(query, user_id):
    # Route 50% to new retriever
    if hash(user_id) % 2 == 0:
        results = new_advanced_retriever.invoke(query)
        log_performance(query, results, variant="test")
    else:
        results = current_retriever.invoke(query)
        log_performance(query, results, variant="control")
    return results
```

## Summary

Advanced RAG patterns solve specific retrieval challenges:

**Ensemble Retrieval** - Combine multiple retrievers (semantic + keyword + metadata) for comprehensive coverage; use weighted fusion or RRF; effective when query types vary widely.

**Reranking** - Two-stage retrieval: fast broad retrieval → precise reranking; use cross-encoders for quality, LLMs for custom criteria; +20-40% precision at cost of latency.

**Parent Document Retriever** - Retrieve with small chunks (precision), return large context; solves the chunk size dilemma; ideal for long structured documents.

**Self-Query Retriever** - Extract metadata filters from natural language queries; combines semantic and structured search; requires rich metadata schema and LLM per query.

**Hybrid Search** - Combine dense (semantic/vectors) and sparse (keyword/BM25) retrieval; complementary strengths for different query types; particularly effective for technical content.

**Key principles:**

- Choose patterns based on query characteristics and document structure
- Combine patterns for production systems (ensemble + rerank + parent-doc)
- Balance precision, latency, and cost
- Monitor and iterate on weights, parameters
- Consider caching and async for performance

**Pattern selection:**

- Start simple (similarity search)
- Add hybrid if keyword matching important
- Add reranking if precision critical
- Add parent-doc if documents are long
- Add ensemble for query diversity
- Add self-query if rich metadata exists

These patterns transform basic RAG into production-grade retrieval systems.

## Next Steps

**Practice implementing:**

- Build ensemble retriever with custom weights
- Integrate cross-encoder reranking
- Set up parent document retriever for long docs
- Create self-query retriever with metadata schema
- Combine patterns into production pipeline

**Related documentation:**

- **[Data and Retrieval](data-and-retrieval.md)** - Foundation retrieval patterns
- **[Orchestration](orchestration.md)** - Chain retrieval patterns into workflows
- **[Fundamentals](fundamentals.md)** - Embeddings and output parsing

**Related lab concepts:**

- **[Retrieval Strategies](../../retrieval-augmented-generation/retrieval-strategies.md)** - Framework-agnostic retrieval theory
- **[Reranking and Fusion](../../retrieval-augmented-generation/reranking-fusion.md)** - Deep dive on reranking techniques
- **[RAG Evaluation](../../retrieval-augmented-generation/rag-evaluation.md)** - Measuring retrieval quality

**Further exploration:**

- Query understanding and intent classification
- Dynamic retriever selection
- Learned fusion models
- Retrieval-aware generation prompting
- End-to-end RAG optimization
