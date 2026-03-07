# RAG Architecture and Concepts

## Table of Contents

- [Introduction](#introduction)
- [What is RAG?](#what-is-rag)
- [The RAG Pipeline](#the-rag-pipeline)
- [Why Use RAG?](#why-use-rag)
- [RAG vs Alternatives](#rag-vs-alternatives)
- [RAG Variants](#rag-variants)
- [Architecture Patterns](#architecture-patterns)
- [Key Components](#key-components)
- [Design Considerations](#design-considerations)
- [Common RAG Challenges](#common-rag-challenges)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Retrieval-Augmented Generation (RAG)** is a technique that combines information retrieval with language model generation. Instead of relying solely on the knowledge encoded in model parameters, RAG systems retrieve relevant information from external sources and use it to inform the generation process.

```
Traditional LLM:
User Query → [LLM] → Response
              ↑
         (parametric knowledge only)

RAG System:
User Query → [Retrieve] → Relevant Docs → [LLM + Context] → Response
              ↑                              ↑
         Knowledge Base              Grounded in retrieved info
```

**Key benefits**:

- Access to up-to-date information beyond training cutoff
- Reduced hallucination through grounding
- Domain-specific knowledge without fine-tuning
- Traceable sources for factual claims
- Dynamic knowledge updates

This guide covers RAG architecture, design patterns, and when to use this powerful approach.

## What is RAG?

### RAG Explained

```python
def explain_rag():
    """Understanding Retrieval-Augmented Generation."""

    print("RAG: Retrieval-Augmented Generation\n")

    print("=" * 60)
    print("\nCore Concept:\n")

    print("Combine two capabilities:")
    print("  1. RETRIEVAL: Find relevant information from knowledge base")
    print("  2. GENERATION: Use LLM to generate response with that info")
    print()

    print("The Process:")
    print("""
    1. User asks a question
       ↓
    2. Convert question to embedding (vector)
       ↓
    3. Search knowledge base for similar embeddings
       ↓
    4. Retrieve top-k most relevant documents
       ↓
    5. Inject retrieved docs into LLM prompt as context
       ↓
    6. LLM generates response grounded in retrieved info
    """)

    print("=" * 60)
    print("\nExample:\n")

    example = """
User Query: "What are the new features in version 2.0?"

Without RAG:
  LLM: "I don't have information about version 2.0 as my knowledge
        was cut off in 2023." ❌

With RAG:
  Step 1: Retrieve relevant docs from knowledge base
          → Found: "Version 2.0 Release Notes (Jan 2024)"

  Step 2: Inject into prompt
          Context: [Release notes content]
          Question: What are the new features in version 2.0?

  Step 3: LLM generates grounded response
          "Version 2.0 includes: dark mode, API v2, and improved
           performance (2x faster). Released January 2024." ✓

  Sources: [Version 2.0 Release Notes]
"""
    print(example)

    print("=" * 60)
    print("\nKey Insight:\n")
    print("RAG lets LLMs access information they weren't trained on,")
    print("enabling accurate, up-to-date, and domain-specific responses.")

explain_rag()
```

### RAG vs Pure LLM

```python
def rag_vs_pure_llm():
    """Comparing RAG to using LLM alone."""

    print("\n\nRAG vs Pure LLM:\n")

    comparison = {
        'Knowledge': {
            'Pure LLM': 'Fixed at training time',
            'RAG': 'Dynamic from knowledge base'
        },
        'Up-to-date info': {
            'Pure LLM': 'Limited by cutoff date',
            'RAG': 'Current if KB is updated'
        },
        'Private data': {
            'Pure LLM': 'Cannot access',
            'RAG': 'Can retrieve and use'
        },
        'Hallucination': {
            'Pure LLM': 'Higher risk',
            'RAG': 'Lower (grounded in docs)'
        },
        'Source attribution': {
            'Pure LLM': 'Not possible',
            'RAG': 'Can cite sources'
        },
        'Latency': {
            'Pure LLM': 'Lower (single LLM call)',
            'RAG': 'Higher (retrieval + LLM)'
        },
        'Complexity': {
            'Pure LLM': 'Simple',
            'RAG': 'More components'
        },
        'Cost': {
            'Pure LLM': 'Lower',
            'RAG': 'Higher (retrieval + storage)'
        }
    }

    print(f"{'Aspect':<20} {'Pure LLM':<30} {'RAG'}")
    print("=" * 80)

    for aspect, comparison_data in comparison.items():
        print(f"{aspect:<20} {comparison_data['Pure LLM']:<30} {comparison_data['RAG']}")

    print("\n" + "=" * 60)
    print("\nWhen to Use Each:\n")

    print("Use Pure LLM when:")
    print("  • General knowledge questions")
    print("  • Creative generation")
    print("  • Low latency critical")
    print("  • Simple use case")
    print()

    print("Use RAG when:")
    print("  • Need current information")
    print("  • Domain-specific knowledge")
    print("  • Private/proprietary data")
    print("  • Factual accuracy critical")
    print("  • Need source attribution")

rag_vs_pure_llm()
```

## The RAG Pipeline

### Basic RAG Workflow

```python
def basic_rag_workflow():
    """The fundamental RAG pipeline."""

    print("Basic RAG Workflow:\n")

    print("""
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PHASE                       │
│                   (Done Once/Periodically)              │
└─────────────────────────────────────────────────────────┘

Documents → [Chunking] → [Embedding] → [Vector Store]
   ↓            ↓            ↓              ↓
 PDFs         Split      Generate        Store
 APIs        into        vector         vectors
 DBs        chunks      embeddings      + metadata


┌─────────────────────────────────────────────────────────┐
│                    QUERY PHASE                          │
│                   (Every User Query)                    │
└─────────────────────────────────────────────────────────┘

User Query → [Embed Query] → [Retrieve] → [Augment] → [Generate]
    ↓             ↓              ↓           ↓            ↓
"What is X?"  Vector       Top-k docs   Add to      LLM response
             representation  from KB     prompt      with context
""")

    print("=" * 60)
    print("\nDetailed Steps:\n")

    steps = {
        'INDEXING (Setup)': [
            '1. Collect documents (PDFs, web pages, databases)',
            '2. Chunk documents into smaller pieces',
            '3. Generate embeddings for each chunk',
            '4. Store embeddings in vector database',
            '5. Store metadata (source, date, etc.)'
        ],
        'RETRIEVAL (Query Time)': [
            '1. User submits query',
            '2. Convert query to embedding vector',
            '3. Search vector DB for similar chunks (cosine similarity)',
            '4. Retrieve top-k most relevant chunks',
            '5. Optional: Rerank results for better relevance'
        ],
        'GENERATION (Query Time)': [
            '1. Build prompt with retrieved context',
            '2. Add user query to prompt',
            '3. Call LLM with augmented prompt',
            '4. LLM generates response using context',
            '5. Return response (optionally with sources)'
        ]
    }

    for phase, phase_steps in steps.items():
        print(f"{phase}:")
        for step in phase_steps:
            print(f"  {step}")
        print()

basic_rag_workflow()
```

### RAG Implementation Example

```python
def simple_rag_implementation():
    """Simple RAG implementation example."""

    print("\n\nSimple RAG Implementation:\n")

    code = '''
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class SimpleRAG:
    """Basic RAG implementation."""

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model)

        # Storage for document chunks and embeddings
        self.chunks = []
        self.embeddings = None
        self.metadata = []

    def index_documents(self, documents: List[Dict]):
        """
        Index documents for retrieval.

        Args:
            documents: List of dicts with 'text' and optional metadata
        """
        print(f"Indexing {len(documents)} documents...")

        # Extract text chunks
        self.chunks = [doc['text'] for doc in documents]

        # Store metadata
        self.metadata = [
            {k: v for k, v in doc.items() if k != 'text'}
            for doc in documents
        ]

        # Generate embeddings for all chunks
        self.embeddings = self.embedder.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"Indexed {len(self.chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant chunks for query.

        Args:
            query: User query string
            top_k: Number of chunks to retrieve

        Returns:
            List of dicts with chunk text, score, and metadata
        """
        # Embed the query
        query_embedding = self.embedder.encode(
            query,
            convert_to_numpy=True
        )

        # Calculate cosine similarity with all chunks
        # cosine_sim = dot(A, B) / (norm(A) * norm(B))
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx],
                'score': float(similarities[idx]),
                'metadata': self.metadata[idx]
            })

        return results

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        llm_call: callable
    ) -> str:
        """
        Generate response using retrieved context.

        Args:
            query: User query
            retrieved_chunks: Results from retrieve()
            llm_call: Function to call LLM

        Returns:
            Generated response
        """
        # Build context from retrieved chunks
        context = "\\n\\n".join([
            f"[{i+1}] {chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        ])

        # Build augmented prompt
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer based on the context above. If the answer isn't in the context, say so.

Answer:"""

        # Generate response
        response = llm_call(prompt)

        return response

    def query(self, query: str, top_k: int = 3, llm_call: callable = None) -> Dict:
        """
        End-to-end RAG: retrieve and generate.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            llm_call: Function to call LLM

        Returns:
            Dict with response, sources, and scores
        """
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, top_k=top_k)

        # Generate response
        if llm_call:
            response = self.generate_response(query, retrieved, llm_call)
        else:
            response = "No LLM configured"

        return {
            'response': response,
            'sources': retrieved,
            'num_sources': len(retrieved)
        }


# Example usage
rag = SimpleRAG()

# Index documents
documents = [
    {
        'text': 'Paris is the capital of France. It is known for the Eiffel Tower.',
        'source': 'geography.txt'
    },
    {
        'text': 'The Eiffel Tower was completed in 1889. It is 330 meters tall.',
        'source': 'landmarks.txt'
    },
    {
        'text': 'France is a country in Western Europe with a population of 67 million.',
        'source': 'geography.txt'
    }
]

rag.index_documents(documents)

# Query
query = "How tall is the Eiffel Tower?"
retrieved = rag.retrieve(query, top_k=2)

print(f"Query: {query}\\n")
for i, result in enumerate(retrieved, 1):
    print(f"Result {i} (score: {result['score']:.3f}):")
    print(f"  {result['text']}")
    print(f"  Source: {result['metadata']['source']}")
'''

    print(code)

simple_rag_implementation()
```

## Why Use RAG?

### RAG Benefits

```python
def rag_benefits():
    """Key benefits of RAG systems."""

    print("Why Use RAG?\n")

    benefits = {
        'Knowledge Updates': {
            'problem': 'LLM knowledge frozen at training time',
            'solution': 'RAG retrieves current info from updated KB',
            'example': 'Answer questions about yesterday\'s news'
        },
        'Reduce Hallucination': {
            'problem': 'LLMs make up plausible-sounding facts',
            'solution': 'RAG grounds responses in retrieved documents',
            'example': '95% accuracy vs 65% without RAG (fact-checking)'
        },
        'Private Data': {
            'problem': 'LLM cannot access company/personal data',
            'solution': 'RAG retrieves from private knowledge base',
            'example': 'Answer questions about internal company docs'
        },
        'Source Attribution': {
            'problem': 'Cannot verify where LLM got information',
            'solution': 'RAG returns source documents with response',
            'example': 'Show which document supports each claim'
        },
        'Domain Expertise': {
            'problem': 'General LLM lacks specialized knowledge',
            'solution': 'RAG provides domain-specific context',
            'example': 'Medical, legal, or technical Q&A'
        },
        'No Fine-tuning Needed': {
            'problem': 'Fine-tuning is expensive and slow',
            'solution': 'RAG adds knowledge without retraining',
            'example': 'Update knowledge by adding documents'
        },
        'Cost Effective': {
            'problem': 'Fine-tuning large models very expensive',
            'solution': 'RAG works with smaller/cheaper models',
            'example': '$10/mo vs $10k+ for fine-tuning'
        }
    }

    for benefit, details in benefits.items():
        print(f"{benefit.upper()}:")
        print(f"  Problem: {details['problem']}")
        print(f"  Solution: {details['solution']}")
        print(f"  Example: {details['example']}")
        print()

    print("=" * 60)
    print("\nImpact on Key Metrics:\n")

    metrics = [
        ('Factual Accuracy', '65% → 95%', '+30pp'),
        ('Hallucination Rate', '35% → 5%', '-30pp'),
        ('Source Traceability', '0% → 100%', '+100pp'),
        ('Knowledge Currency', 'Months old → Real-time', 'Dynamic'),
    ]

    print(f"{'Metric':<25} {'Without → With RAG':<25} {'Improvement'}")
    print("=" * 70)

    for metric, change, improvement in metrics:
        print(f"{metric:<25} {change:<25} {improvement}")

rag_benefits()
```

### Use Cases

```python
def rag_use_cases():
    """Common RAG use cases."""

    print("\n\nRAG Use Cases:\n")

    use_cases = {
        'Customer Support': {
            'description': 'Answer customer questions using product documentation',
            'knowledge_base': 'Product docs, FAQs, troubleshooting guides',
            'benefit': 'Accurate, up-to-date answers with sources',
            'example': '"How do I reset my password?" → Retrieve password reset docs'
        },
        'Enterprise Q&A': {
            'description': 'Search and answer questions over company documents',
            'knowledge_base': 'Internal docs, wikis, presentations, emails',
            'benefit': 'Employees find information faster',
            'example': '"What is our Q3 revenue?" → Retrieve financial reports'
        },
        'Research Assistant': {
            'description': 'Synthesize information from research papers',
            'knowledge_base': 'Academic papers, articles, books',
            'benefit': 'Accurate summaries with citations',
            'example': '"What are recent advances in RAG?" → Retrieve papers'
        },
        'Legal/Compliance': {
            'description': 'Answer questions based on regulations and case law',
            'knowledge_base': 'Laws, regulations, court cases, policies',
            'benefit': 'Precise answers with legal citations',
            'example': '"What are GDPR requirements for data retention?"'
        },
        'Medical Diagnosis Support': {
            'description': 'Provide information from medical literature',
            'knowledge_base': 'Medical journals, drug databases, guidelines',
            'benefit': 'Evidence-based recommendations',
            'example': '"What are treatment options for condition X?"'
        },
        'Code Documentation': {
            'description': 'Answer programming questions from codebase',
            'knowledge_base': 'Code files, documentation, Stack Overflow',
            'benefit': 'Context-aware code help',
            'example': '"How does authentication work in our API?"'
        }
    }

    for use_case, details in use_cases.items():
        print(f"{use_case.upper()}:")
        print(f"  Description: {details['description']}")
        print(f"  KB: {details['knowledge_base']}")
        print(f"  Benefit: {details['benefit']}")
        print(f"  Example: {details['example']}")
        print()

rag_use_cases()
```

## RAG vs Alternatives

### Comparison Matrix

```python
def rag_vs_alternatives():
    """Comparing RAG to other approaches."""

    print("RAG vs Alternative Approaches:\n")

    comparison = {
        'Approach': ['RAG', 'Fine-tuning', 'Prompt Engineering', 'Long Context'],
        'Setup Effort': ['Medium', 'High', 'Low', 'Low'],
        'Ongoing Cost': ['Medium', 'Low', 'Low', 'High'],
        'Update Speed': ['Instant', 'Slow (retrain)', 'Instant', 'Instant'],
        'Knowledge Capacity': ['Large', 'Medium', 'Small', 'Large'],
        'Factual Accuracy': ['High', 'High', 'Medium', 'Medium'],
        'Latency': ['Medium', 'Low', 'Low', 'High'],
        'Source Attribution': ['Yes', 'No', 'No', 'Partial'],
        'Dynamic Knowledge': ['Yes', 'No', 'Yes', 'Yes'],
        'Cost at Scale': ['Medium', 'Low', 'Low', 'Very High']
    }

    # Print as table
    print(f"{'Dimension':<20}", end='')
    for approach in comparison['Approach']:
        print(f"{approach:<20}", end='')
    print("\n" + "=" * 100)

    for key in comparison:
        if key == 'Approach':
            continue
        print(f"{key:<20}", end='')
        for i, approach in enumerate(comparison['Approach']):
            print(f"{comparison[key][i]:<20}", end='')
        print()

    print("\n" + "=" * 60)
    print("\nWhen to Use Each:\n")

    when_to_use = {
        'RAG': [
            'Need up-to-date information',
            'Private/proprietary knowledge',
            'Source attribution required',
            'Frequent knowledge updates'
        ],
        'Fine-tuning': [
            'Specific task/style adaptation',
            'Static knowledge sufficient',
            'Low latency critical',
            'Budget for training'
        ],
        'Prompt Engineering': [
            'General knowledge tasks',
            'Small amount of context',
            'Quick iteration needed',
            'Low complexity'
        ],
        'Long Context': [
            'All context fits in window',
            'Rare queries (high cost ok)',
            'Very specific document set',
            'No retrieval infrastructure'
        ]
    }

    for approach, criteria in when_to_use.items():
        print(f"{approach}:")
        for criterion in criteria:
            print(f"  • {criterion}")
        print()

    print("=" * 60)
    print("\nCan Combine Approaches:\n")

    combinations = [
        'RAG + Fine-tuning: Retrieve docs, use fine-tuned model to generate',
        'RAG + Long Context: Retrieve many docs, put all in long context window',
        'RAG + Prompt Engineering: Retrieve docs, engineer prompt for better use',
        'All three: Fine-tuned model + RAG retrieval + optimized prompts'
    ]

    for combo in combinations:
        print(f"  • {combo}")

rag_vs_alternatives()
```

## RAG Variants

### Single-Step RAG

```python
def single_step_rag():
    """Basic single-step RAG."""

    print("RAG Variants:\n")

    print("=" * 60)
    print("\n1. SINGLE-STEP RAG (Standard)\n")

    print("Process:")
    print("  Query → Retrieve once → Generate")
    print()

    print("""
  User: "What is the capital of France?"
     ↓
  Retrieve: [Top 3 docs about France]
     ↓
  Generate: "Paris is the capital of France."
""")

    print("Characteristics:")
    print("  • One retrieval, one generation")
    print("  • Fast, simple")
    print("  • Works for straightforward questions")
    print("  • Most common pattern")
    print()

    print("Best for:")
    print("  • Simple fact lookup")
    print("  • Low latency requirements")
    print("  • Straightforward queries")

single_step_rag()
```

### Iterative RAG

```python
def iterative_rag():
    """Iterative multi-step RAG."""

    print("\n" + "=" * 60)
    print("\n2. ITERATIVE RAG (Multi-Step)\n")

    print("Process:")
    print("  Query → Retrieve → Generate → Assess → Retrieve more → Generate")
    print()

    print("""
  User: "Compare the heights of the Eiffel Tower and Statue of Liberty"
     ↓
  Step 1: Retrieve docs about Eiffel Tower
          Generate: "Eiffel Tower is 330m"
     ↓
  Step 2: Assess: Need info about Statue of Liberty
          Retrieve docs about Statue of Liberty
     ↓
  Step 3: Generate: "Eiffel Tower (330m) is taller than
                     Statue of Liberty (93m)"
""")

    print("Characteristics:")
    print("  • Multiple retrieval-generation cycles")
    print("  • Adaptive based on intermediate results")
    print("  • Better for complex questions")
    print("  • Higher latency and cost")
    print()

    print("Best for:")
    print("  • Multi-hop reasoning")
    print("  • Comparison questions")
    print("  • Complex research tasks")
    print()

    code = '''
def iterative_rag(query: str, max_iterations: int = 3):
    """Iterative RAG with multiple retrieval steps."""

    context = ""

    for i in range(max_iterations):
        # Retrieve based on current understanding
        retrieved = retrieve(query, context)

        # Generate partial answer
        partial_answer = generate(query, context + retrieved)

        # Check if we have enough information
        if is_complete(partial_answer):
            return partial_answer

        # Extract what's still needed
        missing = extract_missing_info(partial_answer)

        # Update context and continue
        context += retrieved + partial_answer
        query = missing

    return partial_answer
'''

    print("Implementation Pattern:")
    print(code)

iterative_rag()
```

### Adaptive RAG

```python
def adaptive_rag():
    """Adaptive RAG that chooses strategy."""

    print("\n" + "=" * 60)
    print("\n3. ADAPTIVE RAG (Intelligent)\n")

    print("Process:")
    print("  Query → Analyze → Choose strategy → Execute")
    print()

    print("""
  User: [Query]
     ↓
  Analyze query complexity:
    • Simple fact? → Single-step RAG
    • Multi-hop? → Iterative RAG
    • Well-known? → Skip retrieval, use LLM directly
    • Ambiguous? → Query clarification first
     ↓
  Execute chosen strategy
""")

    print("Characteristics:")
    print("  • Chooses approach based on query")
    print("  • Most efficient use of resources")
    print("  • Requires query classification")
    print("  • More complex to implement")
    print()

    print("Decision Logic:")
    decisions = [
        ('Simple fact', 'Single-step RAG', 'Low cost, fast'),
        ('Complex multi-hop', 'Iterative RAG', 'Better quality'),
        ('Common knowledge', 'No RAG', 'Fastest, cheapest'),
        ('Ambiguous', 'Clarify first', 'Better results')
    ]

    print(f"{'Query Type':<20} {'Strategy':<20} {'Benefit'}")
    print("-" * 60)
    for query_type, strategy, benefit in decisions:
        print(f"{query_type:<20} {strategy:<20} {benefit}")

    print()

    code = '''
def adaptive_rag(query: str):
    """Choose RAG strategy based on query."""

    # Classify query
    query_type = classify_query(query)

    if query_type == 'simple_fact':
        # Single retrieval sufficient
        return single_step_rag(query)

    elif query_type == 'multi_hop':
        # Need iterative retrieval
        return iterative_rag(query)

    elif query_type == 'common_knowledge':
        # LLM knows this, skip retrieval
        return llm_call(query)

    elif query_type == 'ambiguous':
        # Clarify first
        clarified = clarify_query(query)
        return adaptive_rag(clarified)

    else:
        # Default to single-step
        return single_step_rag(query)
'''

    print("Implementation Pattern:")
    print(code)

adaptive_rag()
```

## Architecture Patterns

### Basic RAG Architecture

```python
def basic_rag_architecture():
    """Basic RAG system architecture."""

    print("\n\nRAG Architecture Patterns:\n")

    print("=" * 60)
    print("\nBASIC RAG ARCHITECTURE:\n")

    print("""
┌─────────────────────────────────────────────────────────┐
│                   DATA PREPARATION                      │
└─────────────────────────────────────────────────────────┘

  Documents
      ↓
  [Chunker]
      ↓
  Chunks
      ↓
  [Embedding Model]
      ↓
  Embeddings
      ↓
  [Vector Database]


┌─────────────────────────────────────────────────────────┐
│                   QUERY PROCESSING                      │
└─────────────────────────────────────────────────────────┘

  User Query
      ↓
  [Embedding Model]
      ↓
  Query Vector
      ↓
  [Vector Database Search]
      ↓
  Retrieved Chunks (top-k)
      ↓
  [Prompt Construction]
      ↓
  Augmented Prompt
      ↓
  [LLM]
      ↓
  Response
""")

    print("Components:")
    print("  • Document Store: Original documents")
    print("  • Chunker: Splits documents into pieces")
    print("  • Embedding Model: Converts text to vectors")
    print("  • Vector Database: Stores and searches embeddings")
    print("  • LLM: Generates final response")

basic_rag_architecture()
```

### Advanced RAG Architecture

```python
def advanced_rag_architecture():
    """Advanced RAG with additional components."""

    print("\n\n" + "=" * 60)
    print("\nADVANCED RAG ARCHITECTURE:\n")

    print("""
┌─────────────────────────────────────────────────────────┐
│                DATA PREPARATION (Enhanced)              │
└─────────────────────────────────────────────────────────┘

  Documents
      ↓
  [Preprocessor] (clean, extract metadata)
      ↓
  [Smart Chunker] (semantic boundaries)
      ↓
  Chunks + Metadata
      ↓
  [Embedding Model] (query-optimized)
      ↓
  [Vector Database] (with metadata filters)


┌─────────────────────────────────────────────────────────┐
│                QUERY PROCESSING (Enhanced)              │
└─────────────────────────────────────────────────────────┘

  User Query
      ↓
  [Query Understanding] (intent, entities)
      ↓
  [Query Expansion] (synonyms, related terms)
      ↓
  [Multi-Strategy Retrieval]
      ├─ Dense (embedding similarity)
      ├─ Sparse (keyword/BM25)
      └─ Filtered (metadata)
      ↓
  Initial Results (top-20)
      ↓
  [Reranker] (cross-encoder)
      ↓
  Top-k (top-5)
      ↓
  [Context Compression] (remove irrelevant)
      ↓
  [Prompt Construction] (structured template)
      ↓
  [LLM] (with system prompt)
      ↓
  Response
      ↓
  [Citation Extraction] (link to sources)
      ↓
  Final Response + Sources
""")

    print("\nAdditional Components:")
    print("  • Query Understanding: Parse intent and entities")
    print("  • Query Expansion: Add related terms")
    print("  • Hybrid Retrieval: Combine multiple strategies")
    print("  • Reranker: Improve relevance with cross-encoder")
    print("  • Context Compression: Remove noise")
    print("  • Citation Extraction: Link claims to sources")

advanced_rag_architecture()
```

## Key Components

### Component Roles

```python
def component_roles():
    """Roles of key RAG components."""

    print("\n\nKey RAG Components:\n")

    components = {
        'Document Store': {
            'purpose': 'Store original documents',
            'examples': 'S3, database, file system',
            'considerations': 'Storage cost, access speed'
        },
        'Chunker': {
            'purpose': 'Split documents into retrievable units',
            'examples': 'Fixed-size, semantic, sentence-based',
            'considerations': 'Chunk size, overlap, boundaries'
        },
        'Embedding Model': {
            'purpose': 'Convert text to vector representations',
            'examples': 'OpenAI ada-002, Sentence Transformers',
            'considerations': 'Quality, cost, latency, dimension'
        },
        'Vector Database': {
            'purpose': 'Store and search embeddings efficiently',
            'examples': 'Pinecone, Weaviate, Chroma, FAISS',
            'considerations': 'Scale, cost, features, performance'
        },
        'Retriever': {
            'purpose': 'Find relevant chunks for query',
            'examples': 'Dense, sparse, hybrid',
            'considerations': 'Precision, recall, latency'
        },
        'Reranker': {
            'purpose': 'Improve relevance of retrieved results',
            'examples': 'Cross-encoder, LLM-based',
            'considerations': 'Accuracy vs cost/latency'
        },
        'LLM': {
            'purpose': 'Generate final response using context',
            'examples': 'GPT-4, Claude, open models',
            'considerations': 'Quality, cost, context window'
        },
        'Prompt Template': {
            'purpose': 'Structure context and query for LLM',
            'examples': 'Instructions, context, query, constraints',
            'considerations': 'Clarity, format, token usage'
        }
    }

    for component, details in components.items():
        print(f"{component.upper()}:")
        print(f"  Purpose: {details['purpose']}")
        print(f"  Examples: {details['examples']}")
        print(f"  Considerations: {details['considerations']}")
        print()

component_roles()
```

## Design Considerations

### System Design Factors

```python
def design_considerations():
    """Factors to consider when designing RAG systems."""

    print("RAG Design Considerations:\n")

    considerations = {
        'Latency': {
            'factors': [
                'Retrieval speed (vector search)',
                'Number of documents retrieved',
                'Reranking overhead',
                'LLM generation time',
                'Network latency'
            ],
            'target': 'Aim for <2s total latency',
            'optimization': 'Use faster embeddings, limit top-k, cache common queries'
        },
        'Cost': {
            'factors': [
                'Embedding API calls',
                'Vector database storage/operations',
                'LLM API calls',
                'Compute for self-hosted components'
            ],
            'target': '$0.01-$0.10 per query',
            'optimization': 'Batch embeddings, use cheaper models, cache results'
        },
        'Quality': {
            'factors': [
                'Retrieval precision/recall',
                'Chunk quality',
                'Embedding model quality',
                'LLM capability',
                'Prompt engineering'
            ],
            'target': '>90% factual accuracy',
            'optimization': 'Better embeddings, reranking, prompt tuning'
        },
        'Scale': {
            'factors': [
                'Number of documents',
                'Query volume',
                'Concurrent users',
                'Update frequency'
            ],
            'target': 'Handle your load + 3x headroom',
            'optimization': 'Horizontal scaling, caching, async processing'
        },
        'Maintainability': {
            'factors': [
                'System complexity',
                'Component dependencies',
                'Monitoring/debugging',
                'Update process'
            ],
            'target': 'Clear ownership, good observability',
            'optimization': 'Modular design, logging, metrics'
        }
    }

    for consideration, details in considerations.items():
        print(f"{consideration.upper()}:")
        print("  Factors:")
        for factor in details['factors']:
            print(f"    • {factor}")
        print(f"  Target: {details['target']}")
        print(f"  Optimization: {details['optimization']}")
        print()

    print("=" * 60)
    print("\nTrade-off Examples:\n")

    tradeoffs = [
        ('More docs retrieved', '↑ Recall, ↑ Cost, ↑ Latency, ↓ Precision'),
        ('Larger chunks', '↑ Context, ↑ Cost, ↓ Precision'),
        ('Better embeddings', '↑ Quality, ↑ Cost'),
        ('Add reranking', '↑ Precision, ↑ Latency, ↑ Cost'),
        ('Use GPT-4 vs GPT-3.5', '↑ Quality, ↑ Cost, ↑ Latency'),
    ]

    for decision, impact in tradeoffs:
        print(f"  {decision}:")
        print(f"    {impact}")

design_considerations()
```

## Common RAG Challenges

### Challenges and Solutions

```python
def rag_challenges():
    """Common challenges in RAG systems."""

    print("\n\nCommon RAG Challenges:\n")

    challenges = {
        'Poor Retrieval Quality': {
            'symptoms': [
                'Retrieved docs not relevant',
                'Missing key information',
                'Too many irrelevant docs'
            ],
            'causes': [
                'Bad chunking strategy',
                'Weak embedding model',
                'Query-document mismatch',
                'No metadata filtering'
            ],
            'solutions': [
                'Improve chunking (semantic boundaries)',
                'Use better embedding model',
                'Add query expansion',
                'Implement hybrid search',
                'Add reranking'
            ]
        },
        'Context Overflow': {
            'symptoms': [
                'Exceeding LLM context window',
                'Truncated important information',
                'High costs'
            ],
            'causes': [
                'Too many docs retrieved',
                'Chunks too large',
                'No compression'
            ],
            'solutions': [
                'Retrieve fewer docs',
                'Smaller chunks',
                'Compress/summarize context',
                'Use LLM with larger window'
            ]
        },
        'Stale Information': {
            'symptoms': [
                'Outdated answers',
                'Misses recent updates'
            ],
            'causes': [
                'Index not updated',
                'Old documents not removed'
            ],
            'solutions': [
                'Regular re-indexing',
                'Incremental updates',
                'Document versioning',
                'Metadata with timestamps'
            ]
        },
        'Hallucination Despite RAG': {
            'symptoms': [
                'LLM makes up facts not in context',
                'Ignores retrieved information'
            ],
            'causes': [
                'Weak instruction in prompt',
                'Retrieved docs unclear',
                'LLM too confident'
            ],
            'solutions': [
                'Stronger prompt constraints',
                'Better chunk quality',
                'Ask LLM to cite sources',
                'Lower temperature'
            ]
        },
        'High Latency': {
            'symptoms': [
                'Slow responses (>5s)',
                'Poor user experience'
            ],
            'causes': [
                'Slow vector search',
                'Too many docs retrieved',
                'Expensive reranking',
                'Large LLM calls'
            ],
            'solutions': [
                'Faster vector DB',
                'Reduce top-k',
                'Optimize reranker',
                'Use faster LLM',
                'Add caching'
            ]
        }
    }

    for challenge, details in challenges.items():
        print(f"{challenge.upper()}:")
        print("  Symptoms:")
        for symptom in details['symptoms']:
            print(f"    • {symptom}")
        print("  Causes:")
        for cause in details['causes']:
            print(f"    • {cause}")
        print("  Solutions:")
        for solution in details['solutions']:
            print(f"    ✓ {solution}")
        print()

rag_challenges()
```

## Summary

**Key Concepts**:

1. **RAG combines retrieval and generation** - retrieves relevant info then uses it to ground LLM responses
2. **Addresses critical LLM limitations** - knowledge cutoffs, hallucination, private data access
3. **Core pipeline**: Index documents → Embed query → Retrieve relevant docs → Augment prompt → Generate
4. **Major benefits**: Up-to-date info, reduced hallucination (65% → 95% accuracy), source attribution
5. **Three main variants**: Single-step (fast), iterative (multi-hop), adaptive (intelligent selection)
6. **Beats alternatives** when need current info, private data, or frequent updates
7. **Key components**: Chunker, embeddings, vector DB, retriever, LLM, prompt template
8. **Design considerations**: Latency (<2s), cost ($0.01-0.10/query), quality (>90%), scale, maintainability

**RAG vs Alternatives**:

| Aspect   | RAG     | Fine-tuning | Prompt Eng | Long Context |
| -------- | ------- | ----------- | ---------- | ------------ |
| Setup    | Medium  | High        | Low        | Low          |
| Updates  | Instant | Slow        | Instant    | Instant      |
| Accuracy | High    | High        | Medium     | Medium       |
| Cost     | Medium  | Low         | Low        | Very High    |
| Sources  | Yes     | No          | No         | Partial      |

**When to Use RAG**:

```
✓ Need up-to-date information
✓ Private/proprietary knowledge
✓ Source attribution required
✓ Frequent knowledge updates
✓ Domain-specific expertise
✓ Reduce hallucination critical
✓ $10-100k documents to search
```

**Basic RAG Workflow**:

```
INDEXING (once):
1. Collect documents
2. Chunk into pieces
3. Generate embeddings
4. Store in vector DB

QUERY (each request):
1. Embed user query
2. Search for similar chunks
3. Retrieve top-k results
4. Inject into LLM prompt
5. Generate grounded response
```

**RAG Variants**:

- **Single-step**: Query → Retrieve → Generate (most common, fast)
- **Iterative**: Multiple retrieval-generation cycles (complex queries)
- **Adaptive**: Choose strategy based on query type (optimal)

**Architecture Components**:

```
Documents → Chunker → Embeddings → Vector DB
                                       ↓
User Query → Embed → Search → Top-k → Prompt → LLM → Response
                                                         ↓
                                                    + Sources
```

**Performance Impact**:

- Factual accuracy: 65% → 95% (+30pp)
- Hallucination: 35% → 5% (-30pp)
- Source traceability: 0% → 100%
- Latency: +0.5-2s (retrieval overhead)
- Cost: +$0.005-0.05/query

**Common Challenges**:

1. **Poor retrieval** → Better chunking, hybrid search, reranking
2. **Context overflow** → Fewer docs, compression, larger window
3. **Stale info** → Regular re-indexing, incremental updates
4. **Still hallucinates** → Stronger prompts, cite sources
5. **High latency** → Faster vector DB, reduce top-k, caching

**Design Considerations**:

- **Chunk size**: 200-500 tokens typical, balance context vs precision
- **Top-k**: 3-5 chunks usual, more = recall but noise
- **Embedding model**: Trade-off quality vs cost/latency
- **Vector DB**: Consider scale, features, cost
- **Reranking**: +10-20% precision for +50-100ms

**Success Metrics**:

- Retrieval precision/recall
- Answer accuracy
- Hallucination rate
- Latency (end-to-end)
- Cost per query
- User satisfaction

## Next Steps

- Deep dive into [Vector Databases](vector-databases.md) for efficient storage and retrieval
- Learn [Embedding and Chunking](embedding-chunking.md) strategies for better quality
- Explore [Retrieval Strategies](retrieval-strategies.md) for improved relevance
- Study [Reranking and Fusion](reranking-fusion.md) to boost precision
- Master [RAG Evaluation](rag-evaluation.md) to measure and optimize
- Apply to real systems in [Application Patterns](../application_patterns/)
- Combine with [Prompt Engineering](../prompt-engineering/) for better generation
