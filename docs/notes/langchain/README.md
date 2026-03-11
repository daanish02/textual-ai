# LangChain

## About This Section

This section documents practical learning from hands-on LangChain development. Unlike the framework-agnostic concepts in the main lab, these notes capture specific patterns, implementation details, and lessons learned while building with LangChain -- a popular framework for orchestrating LLM applications.

LangChain provides abstractions for common LLM application patterns: prompt management, chain composition, retrieval-augmented generation, and more. These notes cover the building blocks (models, prompts, parsers), data pipelines (loading, chunking, retrieval), and orchestration patterns (runnables, chains) that form complete LLM applications.

This is personal learning documentation -- practical, code-heavy, and implementation-focused. It complements the lab's conceptual foundation with concrete framework patterns.

## Contents

### [Fundamentals](fundamentals.md)

Core building blocks for LLM applications. Covers chat models vs LLMs, multi-provider initialization, prompt templating (string and chat-based), embeddings (query vs document), and output parsing (string, JSON, Pydantic). Includes structured output patterns, model compatibility considerations, and template persistence. Start here to understand the components every LangChain application uses.

### [Data and Retrieval](data-and-retrieval.md)

Building semantic search and RAG pipelines. Covers document loaders (text, PDF, CSV, web), text splitting strategies (character-based, recursive, language-specific, semantic), vector stores vs vector databases, and retrieval patterns (similarity search, MMR, multi-query, contextual compression). Includes chunking strategy trade-offs, metadata handling, and when to use each retriever type.

### [Orchestration](orchestration.md)

Composing LLM calls into workflows. Covers LangChain Expression Language (LCEL), core Runnable primitives (sequence, parallel, passthrough, lambda, branch), and chain patterns built on them (sequential, parallel, conditional). Includes chain visualization, composition strategies, and building complex workflows from simple building blocks.

### [Advanced RAG Patterns](advanced-rag-patterns.md)

Production-grade retrieval strategies for robust RAG systems. Covers ensemble retrieval (combining multiple retrievers with fusion), reranking (cross-encoder and LLM-based), parent document retriever (small chunk retrieval with large context), self-query retriever (natural language to metadata filters), and hybrid search (dense + sparse retrieval). Includes performance optimization, cost management, and pattern selection guidance for building high-precision retrieval systems.

## References

- [LLM Concepts](../../llm-concepts/)
- [Embeddings](../../embeddings/)
- [RAG](../../retrieval-augmented-generation/)
- [Prompt Engineering](../../prompt-engineering/)
