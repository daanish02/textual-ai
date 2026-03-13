# Data and Retrieval

## Table of Contents

- [Introduction](#introduction)
- [Document Loaders](#document-loaders)
  - [The Document Class](#the-document-class)
  - [TextLoader: Plain Text Files](#textloader-plain-text-files)
  - [PyPDFLoader: PDF Documents](#pypdfloader-pdf-documents)
  - [DirectoryLoader: Bulk Loading](#directoryloader-bulk-loading)
  - [WebBaseLoader: Scraping Web Content](#webbaseloader-scraping-web-content)
  - [CSVLoader: Tabular Data](#csvloader-tabular-data)
  - [Load vs Lazy Load](#load-vs-lazy-load)
  - [Other Loaders](#other-loaders)
- [Text Splitting](#text-splitting)
  - [Why Chunking Matters](#why-chunking-matters)
  - [Chunk Size and Overlap](#chunk-size-and-overlap)
  - [CharacterTextSplitter: Length-Based](#charactertextsplitter-length-based)
  - [RecursiveCharacterTextSplitter: Structure-Aware](#recursivecharactertextsplitter-structure-aware)
  - [Language-Specific Splitters](#language-specific-splitters)
  - [SemanticChunker: Meaning-Based](#semanticchunker-meaning-based)
  - [Choosing a Splitting Strategy](#choosing-a-splitting-strategy)
- [Vector Stores](#vector-stores)
  - [Vector Stores vs Vector Databases](#vector-stores-vs-vector-databases)
  - [Basic Operations](#basic-operations)
  - [Similarity Search](#similarity-search)
  - [Metadata Filtering](#metadata-filtering)
  - [Update and Delete](#update-and-delete)
  - [Other Vector Stores](#other-vector-stores)
- [Retrievers](#retrievers)
  - [Vector Store Retrievers](#vector-store-retrievers)
  - [Similarity Search](#similarity-search-retrieval)
  - [Maximum Marginal Relevance (MMR)](#maximum-marginal-relevance-mmr)
  - [Multi-Query Retrieval](#multi-query-retrieval)
  - [Contextual Compression](#contextual-compression)
  - [External Retrievers](#external-retrievers)
  - [Retriever Comparison](#retriever-comparison)
- [Building RAG Pipelines](#building-rag-pipelines)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Building semantic search and RAG (Retrieval-Augmented Generation) systems requires a data pipeline: **loading** documents from various sources, **chunking** them into manageable pieces, **embedding** and **storing** vectors for similarity search, and **retrieving** relevant context for LLM queries.

This document covers the complete data-to-retrieval workflow in LangChain. You'll learn how to ingest data from multiple formats, split text strategically, manage vector stores, and implement sophisticated retrieval patterns that go beyond simple similarity search.

Understanding this pipeline is essential for building production RAG systems that provide LLMs with relevant, accurate context.

## Document Loaders

### The Document Class

LangChain's `Document` is the universal data structure for text content:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="The actual text content goes here",
    metadata={
        "source": "document.pdf",
        "page": 5,
        "author": "Jane Smith",
        "timestamp": "2026-03-04"
    }
)
```

**Structure:**

- **`page_content`** - The text itself (string)
- **`metadata`** - Dictionary of additional information (source, page number, dates, etc.)

**Why metadata matters:**

- **Source tracking** - Know where information came from
- **Filtering** - Retrieve documents by attributes (date range, author, category)
- **Debugging** - Trace retrieved content back to source
- **Citations** - Provide references in generated responses

All loaders return lists of `Document` objects, regardless of source format.

### TextLoader: Plain Text Files

Loads entire text file as a single document:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("poem.txt", encoding="utf-8")
docs = loader.load()

# Returns list with single Document
print(docs[0].page_content)  # Full file content
print(docs[0].metadata)  # {'source': 'poem.txt'}
```

**Use TextLoader for:**

- Plain text files (.txt, .md, .log)
- Small to medium files (entire content is one document)
- Files where natural chunking isn't obvious

**Use with chain example:**

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

template = PromptTemplate(
    template="Summarize this poem:\n{poem}",
    input_variables=["poem"]
)

loader = TextLoader("poem.txt", encoding="utf-8")
docs = loader.load()

chain = template | ChatOpenAI() | StrOutputParser()
summary = chain.invoke({"poem": docs[0].page_content})
```

### PyPDFLoader: PDF Documents

Loads PDFs with **one document per page**:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("ml-book.pdf")
docs = loader.load()

# Each page is a separate Document
print(docs[10].page_content)  # Page 11 text
print(docs[10].metadata)  # {'source': 'ml-book.pdf', 'page': 10}
```

**Key characteristics:**

- **Page-level documents** - Natural chunking by page
- **Metadata includes page number** - Easy to cite specific pages
- **Preserves some formatting** - Paragraphs, line breaks (not perfect)
- **Handles multi-page documents** - Large PDFs split automatically

**Use PyPDFLoader for:**

- Books, research papers, reports
- Documents where page numbers matter for citations
- Large PDFs that need natural pre-chunking

**Limitations:**

- Complex layouts (multi-column, tables) may not parse correctly
- Embedded images and charts are ignored
- OCR scanned PDFs require additional processing

### DirectoryLoader: Bulk Loading

Load all files from a directory matching a pattern:

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load all PDFs from a directory
loader = DirectoryLoader(
    path="data",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()  # Returns all documents from all matching files
```

**Parameters:**

- **`path`** - Directory to scan
- **`glob`** - Pattern to match files (`*.pdf`, `**/*.txt`, etc.)
- **`loader_cls`** - Which loader to use for matched files

**Use DirectoryLoader for:**

- Bulk document ingestion
- Processing entire document collections
- Building knowledge bases from file repositories
- ETL pipelines that process batches

**Examples:**

```python
# All text files recursively
DirectoryLoader("documents", glob="**/*.txt", loader_cls=TextLoader)

# All CSVs in specific folder
DirectoryLoader("data/exports", glob="*.csv", loader_cls=CSVLoader)
```

### WebBaseLoader: Scraping Web Content

Load content from web URLs:

```python
from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/iphone-17-pro/"
loader = WebBaseLoader(url)
docs = loader.load()

# Returns page content as single document
print(docs[0].page_content)  # Extracted text from HTML
print(docs[0].metadata)  # {'source': url}
```

**Use WebBaseLoader for:**

- Documentation scraping
- Product information retrieval
- News article ingestion
- Dynamic content loading for up-to-date information

**Integration example:**

```python
template = PromptTemplate(
    template="Answer this question: {question}\nBased on: {context}",
    input_variables=["question", "context"]
)

loader = WebBaseLoader("https://example.com/product")
docs = loader.load()

chain = template | ChatOpenAI() | StrOutputParser()
answer = chain.invoke({
    "question": "What are the key features?",
    "context": docs[0].page_content
})
```

**Limitations:**

- Requires internet connection
- Some sites block scraping
- Dynamic JavaScript content may not load
- HTML parsing can be messy (ads, navigation, etc.)

### CSVLoader: Tabular Data

Load CSV files with **one document per row**:

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="transactions.csv")
docs = loader.load()

# Each row becomes a Document
print(docs[18])
# Document(
#     page_content="column1: value1\ncolumn2: value2\n...",
#     metadata={'source': 'transactions.csv', 'row': 18}
# )
```

**Key characteristics:**

- **Row-level documents** - Each row is a separate document
- **Column names in content** - Formatted as "column: value"
- **Natural for search** - Query individual records
- **Metadata includes row number** - Track source rows

**Use CSVLoader for:**

- Transaction logs
- Customer databases
- Product catalogs
- Any tabular data that needs semantic search

**Example use case:**

```python
# Search transaction records
loader = CSVLoader("ledger-transactions.csv")
docs = loader.load()

# Build vector store from transactions
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)

# Query: "Show me large expenses in January"
results = vectorstore.similarity_search("large expenses January", k=5)
```

### Load vs Lazy Load

Two loading patterns for different memory constraints:

**`load()`** - Load all documents into memory:

```python
loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()  # Returns complete list
# All documents in memory now
```

**`lazy_load()`** - Stream documents one at a time:

```python
loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.lazy_load()  # Returns generator

# Process one document at a time
for doc in docs:
    print(doc.metadata)
    # Process without loading entire dataset into memory
```

**When to use each:**

| Scenario                   | Use           | Reason                     |
| -------------------------- | ------------- | -------------------------- |
| Small dataset (<1000 docs) | `load()`      | Faster, simpler code       |
| Large dataset (>10k docs)  | `lazy_load()` | Memory efficient           |
| Need full list upfront     | `load()`      | Can count, shuffle, sample |
| Streaming ETL pipeline     | `lazy_load()` | Process incrementally      |
| Development/testing        | `load()`      | Easier debugging           |
| Production batch jobs      | `lazy_load()` | Safer memory usage         |

**Memory implications:**

- `load()`: Memory = number_of_docs × average_doc_size
- `lazy_load()`: Memory = average_doc_size (constant)

### Other Loaders

LangChain supports 100+ loaders for various formats:

**Document formats:**

- `Docx2txtLoader` - Microsoft Word documents
- `UnstructuredMarkdownLoader` - Markdown files with structure
- `JSONLoader` - JSON data extraction
- `UnstructuredPowerPointLoader` - PowerPoint presentations

**Databases:**

- `SQLDatabaseLoader` - Query results as documents
- `MongoDBLoader` - MongoDB collections
- `GoogleDriveLoader` - Files from Google Drive

**APIs and services:**

- `GitHubIssuesLoader` - GitHub issues and PRs
- `SlackDirectoryLoader` - Slack channel history
- `NotionDBLoader` - Notion pages
- `ConfluenceLoader` - Confluence pages

**Specialized:**

- `UnstructuredImageLoader` - OCR from images
- `YoutubeLoader` - Video transcripts
- `SitemapLoader` - Crawl entire websites

Browse the full list: [LangChain Loaders Documentation](https://python.langchain.com/docs/integrations/document_loaders/)

## Text Splitting

### Why Chunking Matters

Raw documents are often too large for LLM contexts or effective retrieval. Text splitting (chunking) breaks documents into smaller, semantically meaningful pieces.

**Why chunking is essential:**

1. **Context window limits** - LLMs have maximum token limits (4k-128k depending on model)
2. **Retrieval precision** - Smaller chunks → more focused matches → better context
3. **Embedding quality** - Shorter texts produce more accurate embeddings
4. **Cost optimization** - Fewer tokens sent to LLM = lower costs
5. **Response relevance** - Retrieve only the relevant section, not entire document

**Example scenario without chunking:**

```python
# Problem: Entire book as one document
book = loader.load()  # 100,000 words

# Embedding this creates "averaged" vector losing specificity
# Retrieval returns entire book regardless of what user asks
# Can't fit in LLM context window
```

**With strategic chunking:**

```python
# Solution: Split into semantic chunks
chunks = text_splitter.split_documents(book)  # 500 chunks of ~200 words

# Each chunk has focused topic
# Retrieval returns just the relevant sections
# Fits multiple relevant chunks in context
```

### Chunk Size and Overlap

Two critical parameters for all splitters:

**Chunk Size** - Target length per chunk (in characters or tokens):

```python
chunk_size=500  # Aim for 500 characters per chunk
```

**Chunk Overlap** - How much adjacent chunks share:

```python
chunk_overlap=50  # 50 characters overlap between chunks
```

**Why overlap matters:**

```
Without overlap:
Chunk 1: "...about quantum mechanics in detail."
Chunk 2: "Einstein proposed that light..." ❌ Context lost

With overlap (50 chars):
Chunk 1: "...about quantum mechanics in detail. Einstein proposed..."
Chunk 2: "...quantum mechanics in detail. Einstein proposed that light..." ✅ Context preserved
```

**Overlap benefits:**

- **Prevents context breaks** - Important information spanning chunk boundaries stays together
- **Improves retrieval** - Queries matching boundary content find both chunks
- **Semantic continuity** - Preserve meaning across splits

**Choosing values:**

| Use Case           | Chunk Size | Overlap | Reasoning                  |
| ------------------ | ---------- | ------- | -------------------------- |
| General QA         | 500-1000   | 50-100  | Balance detail and context |
| Long-context LLMs  | 2000-4000  | 200-400 | Leverage larger windows    |
| Short-form content | 200-500    | 20-50   | Preserve precision         |
| Code documentation | 1000-2000  | 100-200 | Keep functions intact      |
| Dense technical    | 300-600    | 50-100  | Focused retrieval          |

**Trade-offs:**

- **Larger chunks** - More context per chunk, fewer total chunks, less precise retrieval
- **Smaller chunks** - More precise retrieval, more chunks to manage, risk losing context
- **More overlap** - Better context preservation, more redundancy, increased storage
- **Less overlap** - Less redundancy, risk losing boundary information

### CharacterTextSplitter: Length-Based

Simplest splitter -- splits on length and separator:

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""  # Split anywhere
)

chunks = splitter.split_documents(docs)
```

**How it works:**

1. Measures text length
2. Splits at separator when chunk_size reached
3. No awareness of document structure

**Use cases:**

- Simple, uniform text
- Quick prototyping
- When structure doesn't matter

**Limitations:**

- Ignores paragraphs, sentences, code blocks
- May split mid-sentence or mid-word
- Not suitable for structured content

### RecursiveCharacterTextSplitter: Structure-Aware

Intelligent splitter that respects text structure:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75
)

chunks = splitter.split_documents(docs)
```

**How it works -- hierarchical splitting:**

1. Try splitting on **double newlines** (paragraphs) first
2. If chunks still too large, try **single newlines** (sentences)
3. If still too large, try **spaces** (words)
4. If still too large, split by **characters**

**Default separator hierarchy:**

```python
separators = ["\n\n", "\n", " ", ""]
```

**Benefits:**

- **Preserves paragraph boundaries** - Natural semantic units
- **Keeps sentences intact** - Better readability
- **Maintains meaning** - Doesn't break mid-thought
- **Works for general text** - Good default choice

**Use for:**

- Articles, blog posts, documentation
- Books and long-form content
- Any prose where structure matters
- **Use this as your default splitter**

### Language-Specific Splitters

Specialized splitters that understand code and markup syntax:

**Python code splitting:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_details(self):
        return self.name

    def is_passing(self):
        return self.grade >= 6.0

student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=45
)

chunks = splitter.split_text(code)
# Preserves: function boundaries, class structure, logical blocks
```

**How language splitters work:**

- **Syntax-aware separators** - Split on class/function boundaries, not mid-definition
- **Preserve logical units** - Keep complete methods together
- **Comment preservation** - Maintain docstrings with code

**Markdown splitting:**

```python
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=100,
    chunk_overlap=20
)

markdown = """
# Project Name: Smart Student Tracker

## Features
- Add new students
- View student details

## Tech Stack
- Python 3.10+
- No dependencies
"""

chunks = splitter.split_text(markdown)
# Preserves: headers, sections, list items
```

**Supported languages:**

- **Programming:** Python, JavaScript, TypeScript, Java, C++, Go, Rust, Ruby, PHP
- **Markup:** Markdown, HTML, LaTeX
- **Config:** JSON, YAML, TOML
- **Query:** SQL

**When to use language splitters:**

- Code documentation systems
- Repository search
- Code Q&A bots
- Technical documentation with code examples

### SemanticChunker: Meaning-Based

Advanced splitter that uses embeddings to find semantic boundaries:

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text = """
Farmers were working hard in the fields, preparing the soil and planting
seeds for the next season. The sun was bright, and the air smelled of
earth and fresh grass.

The Indian Premier League (IPL) is the biggest cricket league in the world.
People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and
creates fear in cities and villages.
"""

docs = splitter.create_documents([text])
# Result: 3 chunks (farming, cricket, terrorism) - split at topic boundaries
```

**How it works:**

1. **Embed sentences** - Convert each sentence to vector
2. **Compute similarity** - Measure semantic similarity between adjacent sentences
3. **Find breakpoints** - Split where similarity drops significantly
4. **Group semantically similar** - Keep related sentences together

**Breakpoint threshold types:**

**`percentile`** - Split at bottom X% of similarities:

```python
breakpoint_threshold_type="percentile"
breakpoint_threshold_amount=25  # Split at bottom 25% (least similar)
```

- Use when: You want predictable number of chunks
- Effect: Lower percentile = more splits

**`standard_deviation`** - Split where similarity drops by X standard deviations:

```python
breakpoint_threshold_type="standard_deviation"
breakpoint_threshold_amount=1.0  # Split when drop > 1 std dev
```

- Use when: You want to find topic boundaries automatically
- Effect: Higher value = fewer, larger chunks

**`interquartile`** - Split based on interquartile range:

```python
breakpoint_threshold_type="interquartile"
breakpoint_threshold_amount=1.5  # Split at 1.5 × IQR below median
```

- Use when: You want robust boundary detection with outliers
- Effect: Balanced approach between percentile and std dev

**Advantages:**

- **Topic-aware splitting** - Automatically detects when topic changes
- **No fixed size** - Chunks are as long as needed for coherent topics
- **Better for mixed content** - Handles documents with multiple subjects

**Disadvantages:**

- **Requires embeddings** - Slower and more expensive than character splitting
- **Variable chunk sizes** - Can't guarantee max size (may exceed context limits)
- **Complexity** - Harder to tune and understand behavior

**When to use SemanticChunker:**

- Multi-topic documents (articles, research papers)
- When semantic coherence is critical
- Content curation and summarization
- When you can afford embedding costs

### Choosing a Splitting Strategy

Decision tree for selecting a text splitter:

```
Do you have structured code/markup?
├─ YES → Use language-specific splitter (Python, Markdown, etc.)
└─ NO
   └─ Is semantic coherence critical and worth embedding cost?
      ├─ YES → Use SemanticChunker
      └─ NO
         └─ Is text structured with paragraphs?
            ├─ YES → Use RecursiveCharacterTextSplitter (default choice)
            └─ NO → Use CharacterTextSplitter
```

**Quick reference:**

| Splitter                       | Best For             | Pros                      | Cons                           |
| ------------------------------ | -------------------- | ------------------------- | ------------------------------ |
| CharacterTextSplitter          | Simple, uniform text | Fast, simple              | Breaks structure               |
| RecursiveCharacterTextSplitter | General documents    | Structure-aware, balanced | Not semantic                   |
| Language-specific              | Code, markup         | Syntax-aware              | Language-specific              |
| SemanticChunker                | Multi-topic docs     | Topic-aware, coherent     | Slow, expensive, variable size |

**Default recommendation:** Start with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)` -- works well for most use cases.

## Vector Stores

### Vector Stores vs Vector Databases

**Critical distinction:**

**Vector Stores** (like LangChain's Chroma integration):

- **In-memory or local storage** - Data persists to disk but runs in your process
- **Embedded in application** - No separate server
- **Simple** - Easy setup, no infrastructure
- **Local operations** - All processing happens locally
- **Limited scalability** - Suitable for thousands to low millions of vectors

**Vector Databases** (like Pinecone, Weaviate, Milvus):

- **Separate server/service** - Runs as independent database
- **Client-server architecture** - Application connects via API
- **Production-grade** - High availability, backups, monitoring
- **Distributed** - Scale horizontally across machines
- **Enterprise scalability** - Handle billions of vectors

**When to use each:**

| Scenario                    | Use             | Reason                     |
| --------------------------- | --------------- | -------------------------- |
| Development & prototyping   | Vector Store    | Faster, simpler setup      |
| Small datasets (<100k docs) | Vector Store    | Sufficient performance     |
| Single-server deployment    | Vector Store    | No infrastructure overhead |
| Embedded applications       | Vector Store    | Self-contained             |
| Production at scale         | Vector Database | Reliability, performance   |
| Multi-tenant applications   | Vector Database | Isolation, security        |
| Team/shared access          | Vector Database | Centralized, consistent    |
| High availability needs     | Vector Database | Redundancy, failover       |

**Storage comparison:**

```python
# Vector Store (Chroma) - local embeddings
vector_store = Chroma(
    persist_directory="./chroma-db",  # Local disk
    embedding_function=OpenAIEmbeddings()
)

# Vector Database (Pinecone) - remote service
from pinecone import Pinecone
pc = Pinecone(api_key="...")
index = pc.Index("my-index")  # Centralized, cloud-hosted
```

**LangChain abstracts both** - Same API works with vector stores and vector databases, making it easy to switch.

### Basic Operations

Creating and adding documents to a vector store:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Define documents with metadata
doc1 = Document(
    page_content="Virat Kohli is one of the most successful batsmen in IPL history.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history.",
    metadata={"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content="MS Dhoni has led Chennai Super Kings to multiple IPL titles.",
    metadata={"team": "Chennai Super Kings"}
)

# Create vector store
vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma-db",
    collection_name="ipl_players"
)

# Add documents (automatically embeds and stores)
vector_store.add_documents([doc1, doc2, doc3])
```

**What happens during add_documents:**

1. Text is sent to embedding model
2. Vectors are computed
3. Vectors + text + metadata stored locally
4. Data persisted to disk

**Inspecting stored data:**

```python
# Get all documents with their vectors
result = vector_store.get(include=["embeddings", "documents", "metadatas"])

print(result["documents"])  # Original text
print(result["metadatas"])  # Metadata
print(result["embeddings"][0])  # Vector: [0.123, -0.456, ...]
```

### Similarity Search

Find documents most similar to a query:

```python
# Basic similarity search
results = vector_store.similarity_search(
    query="Who among these are a bowler?",
    k=2  # Return top 2 matches
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

**With similarity scores:**

```python
results = vector_store.similarity_search_with_score(
    query="Who among these are a bowler?",
    k=2
)

for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {doc.page_content}\n")
```

**Score interpretation:**

- **Lower scores = more similar** (distance metrics)
- Exact values depend on embedding model and distance metric
- Use scores to set relevance thresholds

**How similarity search works:**

1. **Query embedding** - Convert query to vector
2. **Distance computation** - Compute distance to all stored vectors
3. **Ranking** - Sort by distance (closest first)
4. **Top-k selection** - Return k nearest neighbors

**Distance metrics** (configured at vector store creation):

- **Cosine similarity** - Measures angle between vectors (good for text)
- **Euclidean distance** - Straight-line distance (sensitive to magnitude)
- **Dot product** - Raw similarity (fast but magnitude-sensitive)

### Metadata Filtering

Filter results by metadata before similarity search:

```python
# Find documents for a specific team
results = vector_store.similarity_search_with_score(
    query="",  # Empty query - just filter by metadata
    filter={"team": "Chennai Super Kings"}
)

# Combines semantic search with filtering
results = vector_store.similarity_search(
    query="great captain",
    k=3,
    filter={"team": "Mumbai Indians"}  # Only search within Mumbai Indians
)
```

**Use cases:**

- **Temporal filtering** - `{"date": {"$gte": "2026-01-01"}}`
- **Source filtering** - `{"source": "trusted_docs"}`
- **Category filtering** - `{"category": "technical"}`
- **User isolation** - `{"user_id": "user123"}`

**Metadata indexing strategies:**

- Keep metadata simple (strings, numbers, dates)
- Index frequently filtered fields
- Use consistent metadata schema across documents

### Update and Delete

Modify and remove documents from vector store:

```python
# Update existing document
updated_doc = Document(
    page_content="Virat Kohli, former captain of RCB, holds the record for most runs in IPL history.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(
    document_id="abc-123",  # ID from initial add
    document=updated_doc
)

# Delete document
vector_store.delete(ids=["abc-123"])
```

**Use cases:**

- **Content updates** - Reflect changes in source documents
- **Data expiration** - Remove outdated information
- **GDPR compliance** - Remove user data on request
- **Index maintenance** - Clean up test data

**Best practices:**

- Track document IDs for updates/deletes
- Batch operations when possible (more efficient)
- Rebuild indexes periodically for performance

### Other Vector Stores

LangChain supports many vector store backends:

**Local/Embedded:**

- **Chroma** - Simple, embedded, good for development
- **FAISS** - Facebook's similarity search, CPU/GPU support, no persistence by default
- **LanceDB** - Embedded columnar database, good for ML workflows

**Managed Services:**

- **Pinecone** - Fully managed, excellent developer experience, serverless
- **Weaviate** - Open source + managed, GraphQL interface, hybrid search
- **Qdrant** - High-performance, great for production, good filtering
- **Milvus** - Open source, highly scalable, Kubernetes-native

**Cloud Provider:**

- **AWS OpenSearch** - Integrated with AWS ecosystem
- **Azure Cognitive Search** - Microsoft's vector search
- **Google Vertex AI Vector Search** - GCP vector search

**Database Extensions:**

- **pgvector** - PostgreSQL extension for vectors
- **ElasticSearch** - KNN search in existing Elasticsearch
- **MongoDB Atlas Vector Search** - Vectors in MongoDB

**Choosing a vector store:**

- **Development:** Chroma (simple, embedded)
- **Cost-sensitive:** FAISS or pgvector (self-hosted)
- **Scale & reliability:** Pinecone or Qdrant (managed)
- **Existing infrastructure:** Database extensions (pg vector, etc.)

## Retrievers

Retrievers provide a unified interface for fetching relevant documents. They abstract the retrieval mechanism, enabling easy swapping and composition.

### Vector Store Retrievers

Convert any vector store to a retriever:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)

# Convert to retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Return top 3 results
)

# Use retriever
results = retriever.invoke("your query here")
```

**Retriever benefits:**

- **Consistent interface** - All retrievers use `.invoke(query)`
- **Composable** - Easy to swap implementations
- **Chain-friendly** - Integrate smoothly with LCEL chains

### Similarity Search Retrieval

Standard semantic search - find most similar documents:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

query = "talk about vectors"
results = retriever.invoke(query)
# Returns: 3 documents most similar to query
```

**When to use:**

- General semantic search
- Finding relevant context for RAG
- Document discovery
- **Use as default retrieval method**

### Maximum Marginal Relevance (MMR)

Balances relevance with diversity to avoid redundant results:

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.5  # Balance: 0=max diversity, 1=max relevance
    }
)

query = "talk about langchain"
results = mmr_retriever.invoke(query)
```

**How MMR works:**

1. Find top-N most similar documents (N > k)
2. Select most similar document
3. For remaining slots, choose documents that are:
   - Similar to query AND
   - Diverse from already selected
4. Balance controlled by `lambda_mult`

**lambda_mult parameter:**

- **1.0** - Pure relevance (same as similarity search)
- **0.5** - Balance relevance and diversity
- **0.0** - Maximum diversity (may sacrifice relevance)

**When to use MMR:**

- Query matches many similar documents
- Want diverse perspectives
- Avoid redundant information
- Exploratory search

**Example scenario:**

```
Query: "tell me about LangChain"

Similarity search might return:
1. "LangChain makes it easy to work with LLMs."
2. "LangChain is used to build LLM applications."
3. "LangChain helps with LLM development."
← All very similar, redundant

MMR returns:
1. "LangChain makes it easy to work with LLMs."
2. "Chroma stores embeddings for semantic search."
3. "RAG combines retrieval with generation."
← Similar to query but diverse from each other
```

### Multi-Query Retrieval

Generates multiple query variations to improve recall:

```python
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# Base vector store retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Wrap with multi-query
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOpenAI()
)

query = "talk about health"
results = multiquery_retriever.invoke(query)
```

**How it works:**

1. **LLM generates query variations** - Creates 3-5 alternative phrasings
2. **Retrieve for each variation** - Run similarity search for all queries
3. **Deduplicate results** - Combine and remove duplicates
4. **Return merged set** - Union of all retrieval results

**Example query expansion:**

```
Original: "talk about health"

LLM generates:
- "health and wellness information"
- "staying healthy and fit"
- "medical advice and health tips"
- "health benefits and nutrition"

Retrieve for all 4 queries → merge unique results
```

**Benefits:**

- **Better recall** - Catches documents that match alternate phrasings
- **Handles ambiguity** - Multiple interpretations covered
- **Improves for poor queries** - LLM reformulates unclear questions

**Trade-offs:**

- **More expensive** - Multiple LLM calls + retrievals
- **Longer latency** - Sequential query generation + parallel retrieval
- **Potential noise** - Might retrieve less relevant documents

**When to use:**

- User queries are short or ambiguous
- High recall is critical (don't miss relevant docs)
- Can afford additional latency and cost
- Query quality varies widely

### Contextual Compression

Uses LLM to extract only relevant parts from retrieved documents:

```python
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

# Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Compression with LLM
compressor = LLMChainExtractor.from_llm(ChatOpenAI())

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

query = "talk about photosynthesis"
results = compression_retriever.invoke(query)
```

**How it works:**

1. **Base retrieval** - Get top-k documents via similarity search
2. **LLM extraction** - For each document, LLM extracts query-relevant sentences
3. **Compressed output** - Return only relevant excerpts

**Example:**

```
Original document:
"The Grand Canyon is one of the most visited natural wonders in the world.
Photosynthesis is the process by which green plants convert sunlight into energy.
Millions of tourists travel to see it every year."

Query: "talk about photosynthesis"

Compressed result:
"Photosynthesis is the process by which green plants convert sunlight into energy."
← Only relevant sentence extracted
```

**Benefits:**

- **Focused context** - Remove irrelevant information
- **Token efficiency** - Less content sent to LLM in RAG
- **Better precision** - Reduces noise in retrieved context
- **Cost savings** - Fewer tokens in downstream LLM calls

**Trade-offs:**

- **Additional LLM calls** - One per retrieved document (expensive)
- **Latency increase** - Sequential compression step
- **Risk of over-compression** - Might remove necessary context

**When to use:**

- Retrieved documents are long and noisy
- LLM context window is limited
- Need very precise context extraction
- Cost of compression < cost savings from reduced context

### External Retrievers

Retrieve from sources beyond your vector store:

**Wikipedia retriever:**

```python
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=3,
    lang="en"
)

query = "history of the germanic tribes"
docs = retriever.invoke(query)
# Returns: Top 3 Wikipedia articles
```

**Other external retrievers:**

- **ArxivRetriever** - Academic papers from arXiv
- **PubMedRetriever** - Medical research papers
- **GoogleSearchAPIWrapper** - Web search results
- **YouTubeTranscriptRetriever** - Video transcripts

**When to use:**

- Need up-to-date information beyond your corpus
- Specialized domains (academic, medical, etc.)
- Augment local knowledge with external sources
- Hybrid retrieval (local + external)

**Combining retrievers:**

```python
# Hybrid: local + Wikipedia
local_results = local_retriever.invoke(query)
wiki_results = wiki_retriever.invoke(query)
all_results = local_results + wiki_results
```

### Retriever Comparison

**Choosing the right retriever:**

| Retriever                      | Use Case                    | Pros                              | Cons                           | Cost                    |
| ------------------------------ | --------------------------- | --------------------------------- | ------------------------------ | ----------------------- |
| **Similarity Search**          | General RAG, default choice | Fast, simple, reliable            | May return redundant results   | Low (embeddings only)   |
| **MMR**                        | Diverse perspectives needed | Reduces redundancy, balanced      | Slightly slower                | Low (embeddings only)   |
| **Multi-Query**                | Ambiguous or short queries  | Better recall, handles variations | Higher latency, more expensive | High (LLM + embeddings) |
| **Contextual Compression**     | Long, noisy documents       | Precise extraction, saves tokens  | Expensive, slower              | High (LLM per doc)      |
| **External (Wikipedia, etc.)** | Need external knowledge     | Up-to-date, specialized domains   | Dependency on external service | Variable                |

**Performance considerations:**

```
Latency:
Similarity < MMR < Multi-Query < Contextual Compression

Cost:
Similarity < MMR < Multi-Query < Contextual Compression

Precision:
Contextual Compression > Similarity ≈ MMR > Multi-Query

Recall:
Multi-Query > MMR > Similarity > Contextual Compression
```

## Building RAG Pipelines

Putting it all together - complete RAG workflow:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load documents
loader = PyPDFLoader("knowledge-base.pdf")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

# 3. Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./rag-db"
)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Build RAG chain
template = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context: {context}"),
    ("human", "{question}")
])

chain = (
    {"context": retriever, "question": lambda x: x}
    | template
    | ChatOpenAI()
    | StrOutputParser()
)

# 6. Query
answer = chain.invoke("What is machine learning?")
```

**Pipeline stages:**

1. **Load** - Get documents from source (PDF, web, database, etc.)
2. **Split** - Chunk into retrieval-sized pieces
3. **Embed & Store** - Convert to vectors and store
4. **Retrieve** - Find relevant chunks for query
5. **Generate** - LLM synthesizes answer from context

## Summary

Building data and retrieval pipelines requires understanding the full workflow:

**Document Loading** - Use appropriate loaders for each format (TextLoader, PyPDFLoader, CSVLoader, WebBaseLoader); leverage DirectoryLoader for bulk loading; use lazy_load() for large datasets; Document class standardizes page_content + metadata.

**Text Splitting** - Chunking is essential for context limits, retrieval precision, and embedding quality; configure chunk_size and chunk_overlap based on use case; RecursiveCharacterTextSplitter is the default choice; use language-specific splitters for code; SemanticChunker for topic-aware splitting but higher cost.

**Vector Stores** - Vector stores (Chroma, FAISS) are embedded/local; vector databases (Pinecone, Weaviate) are production-grade services; use vector stores for development, vector databases for scale; support similarity search, metadata filtering, CRUD operations.

**Retrievers** - Unified interface for document retrieval; similarity search is the default; MMR adds diversity; MultiQueryRetriever improves recall via query expansion; ContextualCompressionRetriever extracts relevant excerpts; external retrievers (Wikipedia, etc.) augment local knowledge.

**Key principles:**

- Load → Split → Embed → Store → Retrieve → Generate
- Choose splitter based on content structure
- Balance chunk size (context) vs overlap (continuity)
- Start with similarity search, add sophistication as needed
- Metadata filtering enhances precision

These components form the foundation for production RAG systems.

## Next Steps

- **[Orchestration](orchestration.md)** - Compose these components into chains and workflows using LCEL and Runnables
- **[Fundamentals](fundamentals.md)** - Review models, prompts, and output parsing

**Related lab concepts:**

- [RAG Architecture](../../retrieval-augmented-generation/rag-architecture.md) - Framework-agnostic RAG patterns
- [Embeddings](../../embeddings/) - Deep dive into embedding theory
- [Evaluation](../../evaluation/) - Testing and measuring retrieval quality
