# Retrieval-Augmented Generation

## About This Section

This section covers RAG -- combining information retrieval with language model generation to ground outputs in external knowledge. RAG addresses critical LLM limitations: knowledge cutoffs, hallucination, and inability to access private data. Instead of relying solely on parameters, RAG systems retrieve relevant information and use it to inform generation.

RAG is essential for building production LLM systems that require factual accuracy, up-to-date information, or domain-specific knowledge. You'll learn the architecture, key components, implementation strategies, and how to evaluate and optimize RAG systems.

## Contents

### [RAG Architecture and Concepts](rag-architecture.md)


The fundamental RAG workflow and design patterns. Covers the retrieve-augment-generate pipeline, when to use RAG vs finetuning vs prompt engineering, RAG variants (single-step, iterative, adaptive), and the trade-offs between approaches. Understanding the architecture helps you design effective RAG systems.

### [Vector Databases and Storage](vector-databases.md)

Storing and retrieving embeddings at scale. Covers vector database concepts, indexing strategies (HNSW, IVF, LSH), popular solutions (Pinecone, Weaviate, Chroma, FAISS, Qdrant), choosing a vector store, and practical considerations (cost, latency, scale). Vector databases are the backbone of semantic search in RAG.

### [Embedding and Chunking Strategies](embedding-chunking.md)

Preparing documents for retrieval. Covers document chunking (fixed-size, semantic, recursive), chunk size and overlap trade-offs, choosing embedding models for retrieval, embedding documents vs queries, and handling long documents. Good chunking dramatically improves retrieval quality.

### [Retrieval Strategies](retrieval-strategies.md)

Methods for finding relevant information. Covers dense retrieval (semantic search with embeddings), sparse retrieval (BM25, TF-IDF), hybrid search (combining dense and sparse), query expansion, and filtering. Different retrieval strategies suit different use cases.

### [Reranking and Fusion](reranking-fusion.md)

Improving retrieval quality after initial search. Covers cross-encoder reranking, reciprocal rank fusion, relevance scoring, combining multiple retrievers, and when reranking provides the most value. Reranking improves precision at the cost of additional computation.

### [RAG Evaluation and Optimization](rag-evaluation.md)

Measuring and improving RAG system performance. Covers retrieval metrics (precision, recall, MRR, NDCG), generation quality metrics, end-to-end evaluation, failure analysis, debugging poor retrieval, and iterative optimization. Systematic evaluation is essential for reliable RAG systems.
