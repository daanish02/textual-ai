# Embedding and Chunking

## Table of Contents

- [Introduction](#introduction)
- [Document Chunking](#document-chunking)
- [Chunking Strategies](#chunking-strategies)
- [Chunk Size and Overlap](#chunk-size-and-overlap)
- [Semantic Chunking](#semantic-chunking)
- [Choosing Embedding Models](#choosing-embedding-models)
- [Document vs Query Embeddings](#document-vs-query-embeddings)
- [Handling Long Documents](#handling-long-documents)
- [Preprocessing Techniques](#preprocessing-techniques)
- [Advanced Chunking Patterns](#advanced-chunking-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Document preparation** is critical for RAG quality. Before documents can be retrieved, they must be:

1. **Chunked** into retrievable units
2. **Embedded** into vector representations

```
Document Preparation Pipeline:

Raw Document
     ↓
[Preprocessing] (clean, extract text)
     ↓
Clean Text
     ↓
[Chunking] (split into pieces)
     ↓
Chunks
     ↓
[Embedding] (convert to vectors)
     ↓
Vector Embeddings
     ↓
[Vector Database] (store + index)
```

**Why this matters**:

- **Chunking quality** determines what gets retrieved
- Bad chunks = irrelevant retrieval = poor answers
- Chunk size affects both precision and context quality
- Embedding model determines semantic understanding

This guide covers document chunking strategies, chunk sizing, embedding model selection, and handling various document types effectively.

## Document Chunking

### Why Chunk Documents?

```python
def why_chunk_documents():
    """Understanding the need for document chunking."""

    print("Why Chunk Documents?\n")

    print("=" * 60)
    print("\nProblem with Whole Documents:\n")

    problems = """
1. Context Window Limits
   • LLMs have limited context (4k-128k tokens)
   • Can't fit many full documents
   • Need smaller, focused pieces

2. Retrieval Precision
   • Whole documents often contain multiple topics
   • Embedding represents average of all topics
   • Hard to match specific questions

   Example:
     Document: 50-page manual covering installation,
               configuration, troubleshooting, API reference

     Query: "How do I install?"

     Problem: Document embedding is average of ALL topics
              May not match "installation" well

3. Noise in Context
   • Retrieve whole document → mostly irrelevant content
   • LLM must sift through noise
   • Reduces answer quality

4. Cost
   • Longer context = more tokens = higher cost
   • With chunks, only use relevant pieces
"""

    print(problems)

    print("=" * 60)
    print("\nSolution: Chunking\n")

    solution = """
Break documents into smaller, focused pieces:

Document → Chunks
├─ Chunk 1: Installation (paragraphs 1-3)
├─ Chunk 2: Configuration (paragraphs 4-7)
├─ Chunk 3: Troubleshooting (paragraphs 8-12)
└─ Chunk 4: API Reference (paragraphs 13-20)

Benefits:
  • Each chunk has focused topic
  • Better embedding matches
  • Retrieve only relevant pieces
  • Less noise in context
  • Lower cost

Query: "How do I install?"
Retrieved: Chunk 1 (Installation) ✓
NOT retrieved: Chunks 2, 3, 4 ✗
"""

    print(solution)

    print("=" * 60)
    print("\nKey Trade-offs:\n")

    tradeoffs = [
        ('Chunk too small', '→ Missing context, fragmented info'),
        ('Chunk too large', '→ Less precise, more noise'),
        ('No overlap', '→ May split related info'),
        ('Too much overlap', '→ Redundancy, higher cost'),
    ]

    for decision, impact in tradeoffs:
        print(f"  {decision:<20} {impact}")

why_chunk_documents()
```

### Chunk Anatomy

```python
def chunk_anatomy():
    """Understanding chunk structure."""

    print("\n\nChunk Anatomy:\n")

    print("=" * 60)
    print("\nWhat's in a Chunk?\n")

    chunk_structure = """
┌─────────────────────────────────────────────────────────┐
│                      CHUNK                              │
├─────────────────────────────────────────────────────────┤
│ TEXT CONTENT                                            │
│   The actual text from the document                     │
│   Usually 100-500 tokens (200-1000 characters)          │
│   Should be semantically coherent                       │
├─────────────────────────────────────────────────────────┤
│ METADATA                                                │
│   - source: "user_guide.pdf"                            │
│   - page: 5                                             │
│   - section: "Installation"                             │
│   - chunk_id: "doc_123_chunk_5"                         │
│   - date: "2024-01-15"                                  │
│   - author: "Alice"                                     │
├─────────────────────────────────────────────────────────┤
│ EMBEDDING                                               │
│   [0.23, -0.15, 0.87, ..., 0.42]  (384-1536 dims)      │
│   Vector representation for semantic search             │
└─────────────────────────────────────────────────────────┘
"""

    print(chunk_structure)

    print("=" * 60)
    print("\nChunk Example:\n")

    example = '''
{
    "chunk_id": "manual_v2_chunk_12",
    "text": """
    To install the software, follow these steps:

    1. Download the installer from our website
    2. Run the installer with administrator privileges
    3. Follow the on-screen prompts
    4. Restart your computer when prompted

    The installation typically takes 5-10 minutes.
    """,
    "metadata": {
        "source": "user_manual_v2.pdf",
        "page": 3,
        "section": "Installation",
        "subsection": "Quick Start",
        "doc_type": "manual",
        "version": "2.0",
        "date": "2024-01-15",
        "word_count": 45
    },
    "embedding": [0.23, -0.15, 0.87, ...]  # 384 dimensions
}
'''

    print(example)

    print("\n" + "=" * 60)
    print("\nMetadata Uses:\n")

    uses = [
        ('Filtering', 'Only search specific documents/sections'),
        ('Citation', 'Show source in generated response'),
        ('Deduplication', 'Avoid duplicate sources'),
        ('Access control', 'Filter by permissions'),
        ('Freshness', 'Prioritize recent documents'),
        ('Context', 'Provide LLM with document context'),
    ]

    for use, description in uses:
        print(f"  • {use}: {description}")

chunk_anatomy()
```

## Chunking Strategies

### Fixed-Size Chunking

```python
def fixed_size_chunking():
    """Fixed-size chunking strategy."""

    print("Chunking Strategies:\n")

    print("=" * 60)
    print("\n1. FIXED-SIZE CHUNKING\n")

    print("Concept:")
    print("  Split text into chunks of fixed size (characters or tokens)")
    print("  Optionally add overlap between chunks\n")

    print("Characteristics:")
    print("  • Simplest approach")
    print("  • Predictable chunk count")
    print("  • May split mid-sentence/paragraph")
    print("  • No semantic awareness")
    print()

    code = '''
def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text: Input text
        chunk_size: Number of characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # Get chunk
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        # Move to next chunk with overlap
        start = end - overlap

    return chunks

# Example
text = """
Machine learning is a subset of artificial intelligence that focuses on
algorithms that improve through experience. Deep learning is a specialized
form of machine learning that uses neural networks with multiple layers.
Neural networks are inspired by biological neural networks in the brain.
"""

chunks = fixed_size_chunking(text, chunk_size=100, overlap=20)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk[:50]}...")

# Output:
# Chunk 1: Machine learning is a subset of artificial int...
# Chunk 2: experience. Deep learning is a specialized form...
# Chunk 3: networks with multiple layers. Neural networks...
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nPros and Cons:\n")

    print("PROS:")
    pros = [
        'Simple to implement',
        'Fast processing',
        'Consistent chunk sizes',
        'Works for any text type'
    ]
    for pro in pros:
        print(f"  ✓ {pro}")

    print("\nCONS:")
    cons = [
        'May split sentences/paragraphs awkwardly',
        'No semantic boundaries',
        'Overlap may be redundant or insufficient',
        'Not ideal for structured documents'
    ]
    for con in cons:
        print(f"  ✗ {con}")

    print("\n" + "=" * 60)
    print("\nWhen to Use:\n")
    print("  • Simple, homogeneous text")
    print("  • Quick implementation needed")
    print("  • Text with uniform structure")
    print("  • Baseline approach")

fixed_size_chunking()
```

### Sentence-Based Chunking

```python
def sentence_based_chunking():
    """Sentence-based chunking strategy."""

    print("\n\n" + "=" * 60)
    print("\n2. SENTENCE-BASED CHUNKING\n")

    print("Concept:")
    print("  Split on sentence boundaries, group sentences to target size")
    print("  Respects natural language structure\n")

    code = '''
import re

def sentence_chunking(text: str, max_chunk_size: int = 500):
    """
    Split text into chunks at sentence boundaries.

    Args:
        text: Input text
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    # Split into sentences (simple regex)
    sentences = re.split(r'(?<=[.!?])\\s+', text.strip())

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        # If adding this sentence exceeds max size, start new chunk
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size + 1  # +1 for space

    # Add remaining sentences
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Better sentence splitting with NLTK
import nltk
nltk.download('punkt', quiet=True)

def sentence_chunking_nltk(text: str, max_chunk_size: int = 500):
    """Sentence chunking using NLTK for better accuracy."""

    # Split into sentences
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Example
text = """
Machine learning enables computers to learn from data. It's a powerful tool.
Deep learning uses neural networks. These networks have multiple layers.
Applications include image recognition and natural language processing.
"""

chunks = sentence_chunking_nltk(text, max_chunk_size=100)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")

# Output:
# Chunk 1: Machine learning enables computers to learn from data. It's a powerful tool.
# Chunk 2: Deep learning uses neural networks. These networks have multiple layers.
# Chunk 3: Applications include image recognition and natural language processing.
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nPros and Cons:\n")

    print("PROS:")
    pros = [
        'Respects sentence boundaries',
        'More natural chunks',
        'Better context preservation',
        'Works well for prose'
    ]
    for pro in pros:
        print(f"  ✓ {pro}")

    print("\nCONS:")
    cons = [
        'Variable chunk sizes',
        'May split paragraphs',
        'Sentence detection can fail (abbreviations, etc.)',
        'Not ideal for non-prose text (code, tables)'
    ]
    for con in cons:
        print(f"  ✗ {con}")

    print("\n" + "=" * 60)
    print("\nWhen to Use:\n")
    print("  • Natural language text")
    print("  • Articles, books, documents")
    print("  • When sentence integrity important")
    print("  • Improvement over fixed-size")

sentence_based_chunking()
```

### Paragraph-Based Chunking

```python
def paragraph_based_chunking():
    """Paragraph-based chunking strategy."""

    print("\n\n" + "=" * 60)
    print("\n3. PARAGRAPH-BASED CHUNKING\n")

    print("Concept:")
    print("  Split on paragraph boundaries (double newlines)")
    print("  Paragraphs are natural semantic units\n")

    code = '''
def paragraph_chunking(text: str, max_chunk_size: int = 1000):
    """
    Split text into chunks at paragraph boundaries.

    Args:
        text: Input text
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    # Split on paragraph boundaries (double newline)
    paragraphs = re.split(r'\\n\\s*\\n', text.strip())

    # Remove empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    for paragraph in paragraphs:
        para_size = len(paragraph)

        # If single paragraph exceeds max, split it further
        if para_size > max_chunk_size:
            # Save current chunk if any
            if current_chunk:
                chunks.append('\\n\\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            # Split long paragraph by sentences
            sentences = nltk.sent_tokenize(paragraph)
            temp_chunk = []
            temp_size = 0

            for sentence in sentences:
                if temp_size + len(sentence) > max_chunk_size and temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                    temp_chunk = []
                    temp_size = 0
                temp_chunk.append(sentence)
                temp_size += len(sentence) + 1

            if temp_chunk:
                chunks.append(' '.join(temp_chunk))

        # If adding paragraph exceeds max, start new chunk
        elif current_size + para_size > max_chunk_size and current_chunk:
            chunks.append('\\n\\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_size = para_size

        else:
            current_chunk.append(paragraph)
            current_size += para_size + 2  # +2 for \\n\\n

    # Add remaining paragraphs
    if current_chunk:
        chunks.append('\\n\\n'.join(current_chunk))

    return chunks

# Example
text = """
Machine learning is a method of data analysis. It automates analytical
model building. It is based on the idea that systems can learn from data.

Deep learning is a subset of machine learning. It uses neural networks with
multiple layers. These layers progressively extract higher-level features.

Applications of deep learning include computer vision and speech recognition.
The technology has advanced significantly in recent years.
"""

chunks = paragraph_chunking(text, max_chunk_size=200)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\\n{chunk}\\n")
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nPros and Cons:\n")

    print("PROS:")
    pros = [
        'Respects paragraph boundaries',
        'Strong semantic coherence',
        'Natural reading units',
        'Good for well-formatted text'
    ]
    for pro in pros:
        print(f"  ✓ {pro}")

    print("\nCONS:")
    cons = [
        'Highly variable chunk sizes',
        'May create very small/large chunks',
        'Depends on document formatting',
        'Not all documents have clear paragraphs'
    ]
    for con in cons:
        print(f"  ✗ {con}")

    print("\n" + "=" * 60)
    print("\nWhen to Use:\n")
    print("  • Well-formatted prose")
    print("  • Essays, articles, blogs")
    print("  • When paragraph = semantic unit")
    print("  • Quality over uniform size")

paragraph_based_chunking()
```

### Recursive Chunking

```python
def recursive_chunking():
    """Recursive/hierarchical chunking strategy."""

    print("\n\n" + "=" * 60)
    print("\n4. RECURSIVE CHUNKING (Best Practice)\n")

    print("Concept:")
    print("  Try splitting on largest separators first, recurse if needed")
    print("  Hierarchy: \\n\\n → \\n → . → space")
    print("  Respects natural boundaries while meeting size constraints\n")

    code = '''
def recursive_chunking(
    text: str,
    max_chunk_size: int = 500,
    separators: list = None,
    keep_separator: bool = True
):
    """
    Recursively split text using hierarchy of separators.

    Args:
        text: Input text
        max_chunk_size: Target maximum chunk size
        separators: List of separators in priority order
        keep_separator: Whether to keep separator in chunks

    Returns:
        List of text chunks
    """
    if separators is None:
        # Default hierarchy: paragraph → sentence → word
        separators = ["\\n\\n", "\\n", ". ", " ", ""]

    chunks = []

    def split_text(text: str, sep_idx: int = 0):
        """Recursively split text."""

        # Base case: text small enough
        if len(text) <= max_chunk_size:
            return [text] if text else []

        # Base case: no more separators
        if sep_idx >= len(separators):
            # Force split at max_chunk_size
            return [text[i:i+max_chunk_size]
                    for i in range(0, len(text), max_chunk_size)]

        separator = separators[sep_idx]

        # If separator is empty, force split
        if not separator:
            return [text[i:i+max_chunk_size]
                    for i in range(0, len(text), max_chunk_size)]

        # Split by current separator
        splits = text.split(separator)

        # Merge splits into chunks
        result = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split) + len(separator)

            # If single split too large, recurse with next separator
            if len(split) > max_chunk_size:
                # Add current chunk if any
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if keep_separator and current_chunk:
                        chunk_text += separator
                    result.append(chunk_text)
                    current_chunk = []
                    current_size = 0

                # Recurse on large split
                sub_chunks = split_text(split, sep_idx + 1)
                result.extend(sub_chunks)

            # If adding split exceeds max, start new chunk
            elif current_size + split_size > max_chunk_size and current_chunk:
                chunk_text = separator.join(current_chunk)
                if keep_separator:
                    chunk_text += separator
                result.append(chunk_text)
                current_chunk = [split]
                current_size = len(split)

            else:
                current_chunk.append(split)
                current_size += split_size

        # Add remaining
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            result.append(chunk_text)

        return result

    return split_text(text)

# Example
text = """
Machine learning is a field of AI. It focuses on algorithms that improve
through experience.

Deep learning uses neural networks. These networks have multiple layers.

Applications include:
- Computer vision
- Natural language processing
- Speech recognition
"""

chunks = recursive_chunking(text, max_chunk_size=100)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\\n{chunk}\\n{'-'*40}")
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nPros and Cons:\n")

    print("PROS:")
    pros = [
        'Best of all strategies',
        'Respects natural boundaries when possible',
        'Handles edge cases gracefully',
        'Configurable separator hierarchy',
        'Production-ready'
    ]
    for pro in pros:
        print(f"  ✓ {pro}")

    print("\nCONS:")
    cons = [
        'More complex implementation',
        'Slightly slower than simple methods',
        'Need to tune separator hierarchy'
    ]
    for con in cons:
        print(f"  ✗ {con}")

    print("\n" + "=" * 60)
    print("\nWhen to Use:\n")
    print("  • Production RAG systems")
    print("  • Diverse document types")
    print("  • Need robust chunking")
    print("  • Recommended default approach")

recursive_chunking()
```

## Chunk Size and Overlap

### Chunk Size Guidelines

```python
def chunk_size_guidelines():
    """Guidelines for choosing chunk size."""

    print("\n\nChunk Size and Overlap:\n")

    print("=" * 60)
    print("\nChoosing Chunk Size:\n")

    print("Key Factors:")
    factors = [
        ('Embedding model context', 'Must fit within model limits (e.g., 512 tokens)'),
        ('LLM context window', 'Multiple chunks must fit (e.g., 5 chunks in 4k window)'),
        ('Content granularity', 'Smaller = more precise, larger = more context'),
        ('Query complexity', 'Simple queries = small chunks, complex = larger'),
    ]

    for factor, description in factors:
        print(f"  • {factor}: {description}")

    print("\n" + "=" * 60)
    print("\nCommon Chunk Sizes:\n")

    sizes = """
┌────────────────────────────────────────────────────────────┐
│ Use Case            │ Token Size  │ Char Size   │ Notes    │
├────────────────────────────────────────────────────────────┤
│ Short Q&A           │  128-256    │  500-1000   │ Precise  │
│ General RAG         │  256-512    │ 1000-2000   │ Default  │
│ Complex reasoning   │  512-1024   │ 2000-4000   │ Context  │
│ Summarization       │  1024-2048  │ 4000-8000   │ Long     │
└────────────────────────────────────────────────────────────┘

Recommended: 200-500 tokens (800-2000 characters)
"""

    print(sizes)

    print("\n" + "=" * 60)
    print("\nChunk Size Trade-offs:\n")

    print("SMALL CHUNKS (128-256 tokens):")
    print("  ✓ More precise retrieval")
    print("  ✓ Less noise")
    print("  ✓ Faster search")
    print("  ✗ May lose context")
    print("  ✗ Need more chunks for coverage")
    print("  ✗ May fragment information")
    print()

    print("LARGE CHUNKS (512-1024 tokens):")
    print("  ✓ More context preserved")
    print("  ✓ Better for complex queries")
    print("  ✓ Fewer chunks to manage")
    print("  ✗ Less precise")
    print("  ✗ More noise")
    print("  ✗ Higher cost (more tokens)")
    print()

    print("SWEET SPOT: 200-500 tokens")
    print("  • Balance of precision and context")
    print("  • Works for most use cases")
    print("  • Good starting point")

chunk_size_guidelines()
```

### Chunk Overlap

```python
def chunk_overlap_guidelines():
    """Guidelines for chunk overlap."""

    print("\n\n" + "=" * 60)
    print("\nChunk Overlap:\n")

    print("Why Overlap?")
    print("""
Without overlap:
┌──────────┐┌──────────┐┌──────────┐
│ Chunk 1  ││ Chunk 2  ││ Chunk 3  │
└──────────┘└──────────┘└──────────┘
            ↑ ↑
    Important info split across boundary!
    May not retrieve complete context.

With overlap:
┌──────────┐
│ Chunk 1  │
└──────────┘
     ┌──────────┐
     │ Chunk 2  │
     └──────────┘
          ┌──────────┐
          │ Chunk 3  │
          └──────────┘

Benefits:
  ✓ Info at boundaries captured in multiple chunks
  ✓ Better chance of retrieving complete context
  ✓ Smoother transitions between chunks
""")

    print("=" * 60)
    print("\nOverlap Guidelines:\n")

    guidelines = """
Typical overlap: 10-20% of chunk size

Examples:
  • Chunk size 500 chars → Overlap 50-100 chars
  • Chunk size 1000 chars → Overlap 100-200 chars
  • Chunk size 2000 chars → Overlap 200-400 chars

Trade-offs:

  NO OVERLAP (0%):
    ✓ No redundancy
    ✓ More unique content
    ✗ May split important info

  SMALL OVERLAP (5-10%):
    ✓ Catches most boundary issues
    ✓ Minimal redundancy
    ~ Good default

  MEDIUM OVERLAP (10-20%):
    ✓ Robust boundary coverage
    ✓ Better context continuity
    ✗ Some redundancy
    ~ Recommended

  LARGE OVERLAP (20-50%):
    ✓ Maximum coverage
    ✗ High redundancy
    ✗ More storage
    ✗ More cost
    ~ Only if needed
"""

    print(guidelines)

    code = '''
def chunking_with_overlap(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
):
    """Create chunks with overlap."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append({
            'text': chunk,
            'start': start,
            'end': min(end, len(text)),
            'overlap_prev': start > 0,  # Overlaps with previous
            'overlap_next': end < len(text)  # Overlaps with next
        })

        # Move to next chunk with overlap
        start = end - overlap

    return chunks

# Example
text = "A" * 1000  # 1000 char text

chunks = chunking_with_overlap(text, chunk_size=300, overlap=50)

print(f"Text length: {len(text)}")
print(f"Chunk size: 300")
print(f"Overlap: 50")
print(f"Number of chunks: {len(chunks)}")
print()

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"  Start: {chunk['start']}, End: {chunk['end']}")
    print(f"  Length: {len(chunk['text'])}")
    print(f"  Overlaps prev: {chunk['overlap_prev']}")
    print(f"  Overlaps next: {chunk['overlap_next']}")
'''

    print("\nImplementation:")
    print(code)

chunk_overlap_guidelines()
```

## Semantic Chunking

### Semantic Boundary Detection

```python
def semantic_chunking():
    """Semantic chunking based on content similarity."""

    print("\n\nSemantic Chunking:\n")

    print("=" * 60)
    print("\nConcept:\n")

    print("""
Instead of fixed rules, use embeddings to detect topic boundaries:

1. Split text into sentences
2. Embed each sentence
3. Measure similarity between consecutive sentences
4. Low similarity = topic boundary → split here
5. High similarity = same topic → keep together

Result: Chunks that respect semantic boundaries
""")

    print("=" * 60)
    print("\nVisualization:\n")

    viz = """
Sentence Embeddings:

S1: "ML is a branch of AI."           ┐
S2: "It focuses on learning."         │ High similarity
S3: "Algorithms improve with data."   ┘ → Group into Chunk 1

                                        Low similarity (topic change!)

S4: "Python is a programming language."  ┐
S5: "It's popular for data science."     │ High similarity
S6: "Many libraries are available."      ┘ → Group into Chunk 2
"""

    print(viz)

    code = '''
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk

def semantic_chunking(
    text: str,
    embedding_model: str = 'all-MiniLM-L6-v2',
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 2
):
    """
    Chunk text based on semantic similarity between sentences.

    Args:
        text: Input text
        embedding_model: Model for generating embeddings
        similarity_threshold: Similarity below this = boundary
        min_chunk_size: Minimum sentences per chunk

    Returns:
        List of chunks
    """
    # Initialize model
    model = SentenceTransformer(embedding_model)

    # Split into sentences
    sentences = nltk.sent_tokenize(text)

    # Embed sentences
    embeddings = model.encode(sentences)

    # Calculate similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        similarities.append(sim)

    # Find split points (low similarity = topic boundary)
    split_points = [0]

    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            # Only split if minimum chunk size met
            if i + 1 - split_points[-1] >= min_chunk_size:
                split_points.append(i + 1)

    split_points.append(len(sentences))

    # Create chunks
    chunks = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        chunk_sentences = sentences[start:end]
        chunk_text = ' '.join(chunk_sentences)

        chunks.append({
            'text': chunk_text,
            'sentences': chunk_sentences,
            'start_idx': start,
            'end_idx': end
        })

    return chunks

# Example
text = """
Machine learning is a branch of AI. It enables computers to learn from data.
Algorithms improve automatically through experience.

Python is a popular programming language. It's widely used in data science.
Libraries like NumPy and Pandas are essential tools.

Neural networks are inspired by the brain. They consist of interconnected nodes.
Deep learning uses multiple layers of neurons.
"""

chunks = semantic_chunking(text, similarity_threshold=0.6)

print(f"Created {len(chunks)} semantic chunks\\n")

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:")
    print(f"  Sentences: {len(chunk['sentences'])}")
    print(f"  Text: {chunk['text'][:100]}...")
    print()
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nPros and Cons:\n")

    print("PROS:")
    pros = [
        'Respects semantic boundaries',
        'Each chunk is coherent topic',
        'Better retrieval quality',
        'Adapts to content'
    ]
    for pro in pros:
        print(f"  ✓ {pro}")

    print("\nCONS:")
    cons = [
        'Slower (requires embedding all sentences)',
        'More complex',
        'Need to tune similarity threshold',
        'Variable chunk sizes'
    ]
    for con in cons:
        print(f"  ✗ {con}")

    print("\n" + "=" * 60)
    print("\nWhen to Use:\n")
    print("  • High-quality chunking needed")
    print("  • Multi-topic documents")
    print("  • Budget for slower processing")
    print("  • Premium use cases")

semantic_chunking()
```

## Choosing Embedding Models

### Embedding Model Selection

```python
def embedding_model_selection():
    """Choosing the right embedding model."""

    print("Choosing Embedding Models:\n")

    print("=" * 60)
    print("\nKey Factors:\n")

    factors = {
        'Performance': {
            'Quality': 'Retrieval accuracy on your domain',
            'Speed': 'Embeddings per second',
            'Cost': 'API cost or compute requirements'
        },
        'Technical': {
            'Dimension': '384, 768, 1536 dims (higher = more info, more cost)',
            'Context Length': 'Max tokens (512, 8192, etc.)',
            'Language': 'Supported languages'
        },
        'Deployment': {
            'API vs Self-hosted': 'Convenience vs control',
            'Model Size': 'Disk space, RAM requirements',
            'License': 'Commercial use allowed?'
        }
    }

    for category, items in factors.items():
        print(f"{category}:")
        for factor, description in items.items():
            print(f"  • {factor}: {description}")
        print()

    print("=" * 60)
    print("\nPopular Embedding Models:\n")

    models = {
        'OpenAI text-embedding-3-small': {
            'dims': '1536',
            'context': '8191 tokens',
            'speed': 'Fast (API)',
            'cost': '$0.02/1M tokens',
            'quality': 'Very good',
            'deployment': 'API only',
            'best_for': 'Production, need quality + speed'
        },
        'OpenAI text-embedding-ada-002': {
            'dims': '1536',
            'context': '8191 tokens',
            'speed': 'Fast (API)',
            'cost': '$0.10/1M tokens',
            'quality': 'Excellent',
            'deployment': 'API only',
            'best_for': 'Maximum quality, budget ok'
        },
        'all-MiniLM-L6-v2': {
            'dims': '384',
            'context': '256 tokens',
            'speed': 'Very fast',
            'cost': 'Free (self-hosted)',
            'quality': 'Good',
            'deployment': 'Self-hosted',
            'best_for': 'Development, cost-sensitive'
        },
        'all-mpnet-base-v2': {
            'dims': '768',
            'context': '384 tokens',
            'speed': 'Medium',
            'cost': 'Free (self-hosted)',
            'quality': 'Very good',
            'deployment': 'Self-hosted',
            'best_for': 'Balance of quality and efficiency'
        },
        'e5-large-v2': {
            'dims': '1024',
            'context': '512 tokens',
            'speed': 'Medium',
            'cost': 'Free (self-hosted)',
            'quality': 'Excellent',
            'deployment': 'Self-hosted',
            'best_for': 'Best open-source quality'
        },
        'BGE-large-en-v1.5': {
            'dims': '1024',
            'context': '512 tokens',
            'speed': 'Medium',
            'cost': 'Free (self-hosted)',
            'quality': 'Excellent',
            'deployment': 'Self-hosted',
            'best_for': 'State-of-the-art open-source'
        },
        'Cohere embed-english-v3.0': {
            'dims': '1024',
            'context': '512 tokens',
            'speed': 'Fast (API)',
            'cost': '$0.10/1M tokens',
            'quality': 'Excellent',
            'deployment': 'API',
            'best_for': 'Alternative to OpenAI'
        }
    }

    for model_name, details in models.items():
        print(f"{model_name}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
        print()

    print("=" * 60)
    print("\nRecommendations:\n")

    recommendations = [
        ('Development/prototyping', '→', 'all-MiniLM-L6-v2 (fast, free)'),
        ('Production, budget ok', '→', 'OpenAI text-embedding-3-small'),
        ('Maximum quality', '→', 'OpenAI ada-002 or BGE-large'),
        ('Self-hosted', '→', 'all-mpnet-base-v2 or e5-large-v2'),
        ('Multilingual', '→', 'multilingual-e5-large'),
        ('Cost-sensitive', '→', 'all-MiniLM-L6-v2 (384 dims)'),
    ]

    for scenario, arrow, recommendation in recommendations:
        print(f"  {scenario:<30} {arrow} {recommendation}")

embedding_model_selection()
```

### Using Embedding Models

```python
def using_embedding_models():
    """Examples of using different embedding models."""

    print("\n\nUsing Embedding Models:\n")

    code = '''
# 1. OpenAI embeddings (API)
import openai

openai.api_key = "your-api-key"

def embed_openai(texts: list):
    """Generate embeddings using OpenAI."""

    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    return embeddings

texts = ["Machine learning is great.", "Python is a programming language."]
embeddings = embed_openai(texts)
print(f"Embedding dimension: {len(embeddings[0])}")  # 1536


# 2. Sentence Transformers (self-hosted)
from sentence_transformers import SentenceTransformer

def embed_sentence_transformers(texts: list, model_name: str = 'all-MiniLM-L6-v2'):
    """Generate embeddings using Sentence Transformers."""

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

texts = ["Machine learning is great.", "Python is a programming language."]
embeddings = embed_sentence_transformers(texts)
print(f"Embedding dimension: {embeddings.shape[1]}")  # 384


# 3. Cohere embeddings (API)
import cohere

co = cohere.Client("your-api-key")

def embed_cohere(texts: list):
    """Generate embeddings using Cohere."""

    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"  # or "search_query" for queries
    )

    return response.embeddings

texts = ["Machine learning is great.", "Python is a programming language."]
embeddings = embed_cohere(texts)
print(f"Embedding dimension: {len(embeddings[0])}")  # 1024


# 4. Batch processing for efficiency
def embed_in_batches(texts: list, batch_size: int = 32):
    """Embed texts in batches for efficiency."""

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)

    return embeddings

# Process 1000 documents in batches
texts = ["Document " + str(i) for i in range(1000)]
embeddings = embed_in_batches(texts, batch_size=32)
print(f"Embedded {len(embeddings)} documents")


# 5. Caching embeddings
from functools import lru_cache
import hashlib

# Simple in-memory cache
@lru_cache(maxsize=10000)
def embed_cached(text: str):
    """Cache embeddings to avoid re-computing."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([text])[0]

# Or persistent cache
import pickle

class EmbeddingCache:
    """Persistent embedding cache."""

    def __init__(self, cache_file: str = "embeddings.pkl"):
        self.cache_file = cache_file
        try:
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
        except:
            self.cache = {}

    def get_embedding(self, text: str, model):
        """Get embedding from cache or compute."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash not in self.cache:
            self.cache[text_hash] = model.encode([text])[0]
            self.save()

        return self.cache[text_hash]

    def save(self):
        """Save cache to disk."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

# Usage
cache = EmbeddingCache()
model = SentenceTransformer('all-MiniLM-L6-v2')

text = "Machine learning is awesome"
embedding = cache.get_embedding(text, model)  # Computed
embedding = cache.get_embedding(text, model)  # From cache (fast)
'''

    print(code)

using_embedding_models()
```

## Document vs Query Embeddings

### Asymmetric Search

```python
def document_vs_query_embeddings():
    """Understanding document vs query embeddings."""

    print("\n\nDocument vs Query Embeddings:\n")

    print("=" * 60)
    print("\nAsymmetric Search:\n")

    print("""
In RAG, documents and queries are different:

DOCUMENTS:
  • Long, detailed, informative
  • "Machine learning is a subset of AI that..."

QUERIES:
  • Short, question-form
  • "What is machine learning?"

Problem: Direct embedding may not match well!
""")

    print("=" * 60)
    print("\nSolutions:\n")

    print("""
1. INSTRUCTION-TUNED MODELS

   Some models support instructions:

   # Document
   embedding = model.encode(
       "Machine learning is...",
       prompt="Represent this document for retrieval:"
   )

   # Query
   embedding = model.encode(
       "What is ML?",
       prompt="Represent this query for retrieving relevant documents:"
   )

2. SEPARATE MODELS

   Train separate models for documents and queries:
   • doc_encoder(document) → embedding
   • query_encoder(query) → embedding
   • Search in same space

   Example: DPR (Dense Passage Retrieval)

3. QUERY AUGMENTATION

   Expand short query to be more document-like:

   Query: "What is ML?"
   Augmented: "Machine learning is a field that..."

   Then embed the augmented version

4. HYPOTHETICAL DOCUMENTS

   Generate a hypothetical document that answers the query:

   Query: "How to install Python?"
   Hypothetical Doc: "To install Python, download from python.org..."

   Embed hypothetical doc, search with that
   (Called HyDE - Hypothetical Document Embeddings)
""")

    code = '''
# Example: Using instruction-tuned model (e5)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-base-v2')

# Embed documents
documents = [
    "Machine learning is a subset of AI that focuses on algorithms.",
    "Python is a popular programming language for data science."
]

doc_embeddings = model.encode(
    ["passage: " + doc for doc in documents],  # Instruction prefix
    normalize_embeddings=True
)

# Embed query
query = "What is machine learning?"
query_embedding = model.encode(
    ["query: " + query],  # Different instruction
    normalize_embeddings=True
)[0]

# Search
from numpy import dot

similarities = [dot(query_embedding, doc_emb) for doc_emb in doc_embeddings]

print("Similarities:")
for doc, sim in zip(documents, similarities):
    print(f"  {sim:.4f}: {doc[:50]}...")

# Output:
#   0.7234: Machine learning is a subset of AI...
#   0.3421: Python is a popular programming language...


# Example: HyDE (Hypothetical Document Embeddings)

def hyde_search(query: str, documents: list, llm_call: callable):
    """Search using hypothetical document."""

    # Generate hypothetical document
    prompt = f"Write a paragraph that would answer this question: {query}"
    hypothetical_doc = llm_call(prompt)

    # Embed hypothetical document
    hyp_embedding = model.encode([hypothetical_doc])[0]

    # Embed actual documents
    doc_embeddings = model.encode(documents)

    # Search using hypothetical embedding
    similarities = [dot(hyp_embedding, doc_emb) for doc_emb in doc_embeddings]

    # Return top results
    ranked = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
    return ranked

# Sometimes works better than direct query embedding!
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nWhen to Use Each:\n")

    approaches = [
        ('Standard models', 'Most cases, works well enough'),
        ('Instruction-tuned (e5, BGE)', 'Better asymmetric performance'),
        ('Separate encoders (DPR)', 'Maximum quality, complex setup'),
        ('Query augmentation', 'Simple improvement'),
        ('HyDE', 'When queries very different from docs'),
    ]

    for approach, use_case in approaches:
        print(f"  • {approach}: {use_case}")

document_vs_query_embeddings()
```

## Handling Long Documents

### Strategies for Long Documents

```python
def handling_long_documents():
    """Strategies for handling documents longer than embedding model context."""

    print("Handling Long Documents:\n")

    print("=" * 60)
    print("\nProblem:\n")

    print("""
Most embedding models have limited context:
  • all-MiniLM-L6-v2: 256 tokens
  • all-mpnet-base-v2: 384 tokens
  • OpenAI ada-002: 8191 tokens

But documents can be much longer!
  • Research papers: 5000-10000 tokens
  • Books: 50000+ tokens
  • Legal documents: 20000+ tokens

What to do?
""")

    print("=" * 60)
    print("\nStrategies:\n")

    print("""
1. CHUNKING (Most Common)

   Split document into chunks that fit:
   • Each chunk embedded separately
   • Retrieve relevant chunks
   • Lose document-level understanding

   ✓ Simple
   ✓ Works with any model
   ✗ No global context

2. HIERARCHICAL EMBEDDING

   Multiple levels:
   • Level 1: Chunk embeddings (detailed)
   • Level 2: Section embeddings (mid-level)
   • Level 3: Document embedding (high-level)

   Retrieve at appropriate level:
   • Specific query → chunk level
   • Broad query → document level

   ✓ Captures multiple granularities
   ✗ More complex
   ✗ More storage

3. SUMMARIZE THEN EMBED

   • Summarize long document (using LLM)
   • Embed summary
   • Store summary embedding + full text
   • Retrieve by summary, return full text

   ✓ Document-level understanding
   ✗ Loses details
   ✗ Expensive (LLM calls)

4. LATE CHUNKING

   • Embed full document if possible
   • Extract chunk embeddings from full embedding
   • Best of both worlds

   ✓ Global context preserved
   ✗ Requires long-context model
   ✗ More complex

5. POOLING STRATEGIES

   • Embed chunks separately
   • Pool (average, max, weighted) to get document embedding

   ✓ Document representation
   ✗ May dilute specific info
""")

    code = '''
# Strategy 1: Chunking (Standard)

def chunk_and_embed(document: str, model, chunk_size: int = 500):
    """Standard chunking approach."""

    # Chunk document
    chunks = recursive_chunking(document, chunk_size)

    # Embed each chunk
    chunk_embeddings = []
    for chunk in chunks:
        embedding = model.encode(chunk)
        chunk_embeddings.append({
            'text': chunk,
            'embedding': embedding
        })

    return chunk_embeddings


# Strategy 2: Hierarchical Embedding

def hierarchical_embedding(document: str, model):
    """Create embeddings at multiple granularities."""

    # Level 1: Document summary embedding
    summary = summarize(document)  # Use LLM
    doc_embedding = model.encode(summary)

    # Level 2: Section embeddings
    sections = split_into_sections(document)
    section_embeddings = []
    for section in sections:
        section_emb = model.encode(section['text'])
        section_embeddings.append({
            'text': section['text'],
            'title': section['title'],
            'embedding': section_emb
        })

    # Level 3: Chunk embeddings
    chunks = recursive_chunking(document, chunk_size=500)
    chunk_embeddings = [
        {'text': chunk, 'embedding': model.encode(chunk)}
        for chunk in chunks
    ]

    return {
        'document': {
            'summary': summary,
            'embedding': doc_embedding
        },
        'sections': section_embeddings,
        'chunks': chunk_embeddings
    }

# Retrieval with hierarchy
def hierarchical_retrieval(query: str, hierarchical_data: dict, model):
    """Retrieve at appropriate level."""

    query_emb = model.encode(query)

    # First: Check document-level
    doc_sim = cosine_similarity(query_emb, hierarchical_data['document']['embedding'])

    if doc_sim < 0.3:
        # Not relevant
        return []

    # Query is broad: return section-level
    if is_broad_query(query):
        # Search sections
        results = search_level(query_emb, hierarchical_data['sections'])
        return results

    # Query is specific: return chunk-level
    else:
        results = search_level(query_emb, hierarchical_data['chunks'])
        return results


# Strategy 3: Summarize Then Embed

def summarize_and_embed(document: str, model, llm_call):
    """Embed document summary."""

    # Summarize long document
    summary_prompt = f"Summarize the following document:\\n\\n{document}"
    summary = llm_call(summary_prompt)

    # Embed summary
    summary_embedding = model.encode(summary)

    return {
        'summary': summary,
        'summary_embedding': summary_embedding,
        'full_text': document  # Store for retrieval
    }


# Strategy 4: Late Chunking (requires long-context model)

def late_chunking(document: str, model_long_context):
    """Embed full document, extract chunk representations."""

    # Embed full document (requires long-context model)
    # This is conceptual - actual implementation model-specific
    full_embedding = model_long_context.encode(document)

    # Chunk document
    chunks = recursive_chunking(document, chunk_size=500)

    # Extract chunk representations from full embedding
    # (requires model that supports this)
    chunk_embeddings = extract_chunk_embeddings(
        full_embedding,
        chunks,
        model_long_context
    )

    return chunk_embeddings


# Strategy 5: Pooling

def pooled_embedding(document: str, model, strategy: str = 'mean'):
    """Pool chunk embeddings into document embedding."""

    # Chunk and embed
    chunks = recursive_chunking(document, chunk_size=500)
    chunk_embeddings = model.encode(chunks)

    # Pool
    if strategy == 'mean':
        doc_embedding = np.mean(chunk_embeddings, axis=0)
    elif strategy == 'max':
        doc_embedding = np.max(chunk_embeddings, axis=0)
    elif strategy == 'weighted':
        # Weight by chunk importance (e.g., length, position)
        weights = [len(chunk) for chunk in chunks]
        weights = np.array(weights) / sum(weights)
        doc_embedding = np.average(chunk_embeddings, axis=0, weights=weights)

    return {
        'document_embedding': doc_embedding,
        'chunk_embeddings': chunk_embeddings,
        'chunks': chunks
    }
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nRecommendations:\n")

    recommendations = [
        ('Default', '→', 'Chunking (Strategy 1)'),
        ('Multiple granularities needed', '→', 'Hierarchical (Strategy 2)'),
        ('Document-level search', '→', 'Summarize (Strategy 3) or Pooling (Strategy 5)'),
        ('Have long-context model', '→', 'Late Chunking (Strategy 4)'),
        ('Complex documents', '→', 'Hierarchical (Strategy 2)'),
    ]

    for scenario, arrow, recommendation in recommendations:
        print(f"  {scenario:<35} {arrow} {recommendation}")

handling_long_documents()
```

## Preprocessing Techniques

### Text Preprocessing

````python
def preprocessing_techniques():
    """Text preprocessing before chunking and embedding."""

    print("\n\nPreprocessing Techniques:\n")

    print("=" * 60)
    print("\nWhy Preprocess?\n")

    print("""
Raw documents often have issues:
  • Extra whitespace, special characters
  • Headers, footers, page numbers
  • HTML tags, markup
  • Poor formatting
  • Multiple languages
  • Low-quality OCR text

Preprocessing improves:
  ✓ Embedding quality
  ✓ Retrieval accuracy
  ✓ Storage efficiency
""")

    code = '''
import re
from typing import Dict

def preprocess_text(text: str, config: Dict = None) -> str:
    """
    Comprehensive text preprocessing.

    Args:
        text: Input text
        config: Configuration dict for preprocessing options

    Returns:
        Preprocessed text
    """
    if config is None:
        config = {
            'lowercase': False,  # Usually don't lowercase for embeddings
            'remove_extra_whitespace': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_phone_numbers': False,
            'remove_special_chars': False,
            'normalize_unicode': True,
            'remove_html': True,
            'min_length': 10  # Minimum chars to keep chunk
        }

    # Normalize unicode
    if config.get('normalize_unicode'):
        import unicodedata
        text = unicodedata.normalize('NFKC', text)

    # Remove HTML tags
    if config.get('remove_html'):
        text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    if config.get('remove_urls'):
        text = re.sub(r'http[s]?://\\S+', '', text)
        text = re.sub(r'www\\.\\S+', '', text)

    # Remove emails
    if config.get('remove_emails'):
        text = re.sub(r'\\S+@\\S+', '', text)

    # Remove phone numbers
    if config.get('remove_phone_numbers'):
        text = re.sub(r'\\+?\\d[\\d -]{8,}\\d', '', text)

    # Remove extra whitespace
    if config.get('remove_extra_whitespace'):
        text = re.sub(r'\\s+', ' ', text)
        text = text.strip()

    # Remove special characters (optional)
    if config.get('remove_special_chars'):
        text = re.sub(r'[^a-zA-Z0-9\\s.,!?;:\'"()-]', '', text)

    # Lowercase (optional, usually not recommended for embeddings)
    if config.get('lowercase'):
        text = text.lower()

    # Check minimum length
    if len(text) < config.get('min_length', 0):
        return ""

    return text


# PDF-specific preprocessing
def preprocess_pdf_text(text: str) -> str:
    """Preprocess text extracted from PDF."""

    # Remove page numbers
    text = re.sub(r'\\n\\d+\\n', '\\n', text)

    # Fix hyphenation at line breaks
    text = re.sub(r'(\\w+)-\\n(\\w+)', r'\\1\\2', text)

    # Remove repeated headers/footers
    lines = text.split('\\n')
    # (Complex logic to detect and remove repeated lines)

    # Fix spacing issues from OCR
    text = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', text)  # Add space between camelCase

    return text


# Markdown preprocessing
def preprocess_markdown(text: str) -> str:
    """Preprocess markdown text."""

    # Option 1: Remove markdown syntax
    text = re.sub(r'\\*\\*(.+?)\\*\\*', r'\\1', text)  # Bold
    text = re.sub(r'\\*(.+?)\\*', r'\\1', text)  # Italic
    text = re.sub(r'\\[(.+?)\\]\\(.+?\\)', r'\\1', text)  # Links
    text = re.sub(r'^#+\\s+', '', text, flags=re.MULTILINE)  # Headers

    # Option 2: Convert to plain text with structure preserved
    # (Use library like markdown or pypandoc)

    return text


# Handle code blocks
def extract_and_separate_code(text: str) -> Dict:
    """Separate code blocks from text."""

    # Extract code blocks
    code_pattern = r'```(\\w+)?\\n(.+?)```'
    code_blocks = re.findall(code_pattern, text, re.DOTALL)

    # Remove code blocks from text
    text_only = re.sub(code_pattern, '[CODE BLOCK]', text, flags=re.DOTALL)

    return {
        'text': text_only,
        'code_blocks': [{'language': lang, 'code': code}
                        for lang, code in code_blocks]
    }


# Example usage
raw_text = """
<html>
  <p>Check out https://example.com for more info!</p>
  <p>Contact us at info@example.com</p>
</html>

This  has   extra    whitespace.

Machine learning is great.
"""

cleaned = preprocess_text(raw_text)
print("Cleaned text:")
print(cleaned)

# Output:
# "Check out for more info! Contact us at This has extra whitespace.
#  Machine learning is great."
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nDocument-Type Specific:\n")

    types = {
        'PDF': [
            'Remove page numbers, headers, footers',
            'Fix hyphenation at line breaks',
            'Handle OCR errors',
            'Extract text from tables separately'
        ],
        'HTML': [
            'Strip HTML tags',
            'Extract main content (remove nav, ads)',
            'Convert to plain text or markdown',
            'Handle special entities (&nbsp;, etc.)'
        ],
        'Markdown': [
            'Remove or preserve markdown syntax',
            'Extract code blocks separately',
            'Preserve document structure',
            'Handle image links'
        ],
        'Code': [
            'Remove comments (optionally)',
            'Preserve indentation/structure',
            'Add context (file name, function names)',
            'Separate docs from code'
        ]
    }

    for doc_type, techniques in types.items():
        print(f"{doc_type}:")
        for technique in techniques:
            print(f"  • {technique}")
        print()

preprocessing_techniques()
````

## Advanced Chunking Patterns

### Context-Enriched Chunking

```python
def context_enriched_chunking():
    """Advanced chunking with context preservation."""

    print("Advanced Chunking Patterns:\n")

    print("=" * 60)
    print("\n1. CONTEXT-ENRICHED CHUNKING\n")

    print("""
Problem: Chunks lose context when separated from document

Solution: Add context metadata to each chunk

Example:
  Document: "Getting Started Guide"
  Section: "Installation"
  Subsection: "Windows Installation"
  Chunk: "Download the installer from..."

Instead of just the chunk text, include:
  "Document: Getting Started Guide. Section: Installation.
   Windows Installation: Download the installer from..."
""")

    code = '''
def context_enriched_chunking(document: Dict) -> List[Dict]:
    """
    Add document context to each chunk.

    Args:
        document: Dict with 'title', 'sections', etc.

    Returns:
        List of chunks with enriched context
    """
    chunks = []

    for section in document['sections']:
        for subsection in section.get('subsections', [section]):
            # Chunk the subsection
            text_chunks = recursive_chunking(
                subsection['text'],
                chunk_size=500
            )

            for chunk_text in text_chunks:
                # Build context prefix
                context = f"Document: {document['title']}."
                context += f" Section: {section['title']}."

                if 'title' in subsection:
                    context += f" {subsection['title']}:"

                # Enriched chunk
                enriched_text = f"{context} {chunk_text}"

                chunks.append({
                    'text': chunk_text,  # Original
                    'enriched_text': enriched_text,  # With context
                    'metadata': {
                        'document': document['title'],
                        'section': section['title'],
                        'subsection': subsection.get('title'),
                        'page': subsection.get('page')
                    }
                })

    return chunks

# What to embed: enriched_text (with context)
# What to return: original text + metadata

# Benefits:
#  ✓ Better retrieval (context helps matching)
#  ✓ Chunks are self-contained
#  ✓ LLM gets more context
'''

    print(code)

    print("\n" + "=" * 60)
    print("\n2. PARENT-CHILD CHUNKING\n")

    print("""
Idea: Store small chunks for precise retrieval, but return larger parent context

Structure:
  Parent Chunk (1000 tokens)
    ├─ Child Chunk 1 (200 tokens) [embedded]
    ├─ Child Chunk 2 (200 tokens) [embedded]
    └─ Child Chunk 3 (200 tokens) [embedded]

Workflow:
  1. Search using child chunks (precise)
  2. Return parent chunk (more context)
  3. Best of both worlds!
""")

    code2 = '''
def parent_child_chunking(text: str) -> List[Dict]:
    """Create parent-child chunk hierarchy."""

    # Create parent chunks (large)
    parent_chunks = recursive_chunking(text, chunk_size=1000)

    all_chunks = []

    for parent_id, parent_text in enumerate(parent_chunks):
        # Create child chunks (small)
        child_chunks = recursive_chunking(parent_text, chunk_size=200)

        # Store parent
        parent_chunk = {
            'id': f'parent_{parent_id}',
            'text': parent_text,
            'type': 'parent',
            'embed': False  # Don't embed parents
        }

        # Store children
        for child_id, child_text in enumerate(child_chunks):
            child_chunk = {
                'id': f'parent_{parent_id}_child_{child_id}',
                'text': child_text,
                'parent_id': f'parent_{parent_id}',
                'type': 'child',
                'embed': True  # Embed children for search
            }
            all_chunks.append(child_chunk)

        all_chunks.append(parent_chunk)

    return all_chunks

# Retrieval process:
def retrieve_with_parents(query: str, chunks: List[Dict], vector_db, top_k=5):
    """Retrieve child chunks, return parent context."""

    # Search child chunks only
    child_chunks = [c for c in chunks if c['type'] == 'child']
    results = vector_db.search(query, child_chunks, top_k=top_k)

    # Get parent chunks
    parent_ids = [r['parent_id'] for r in results]
    parents = [c for c in chunks if c['id'] in parent_ids]

    return parents  # Return larger context

# Benefits:
#  ✓ Precise retrieval (small chunks)
#  ✓ Rich context (large chunks returned)
#  ✓ Flexible (can return parent or child or both)
'''

    print(code2)

context_enriched_chunking()
```

### Sliding Window Chunking

```python
def sliding_window_chunking():
    """Sliding window chunking pattern."""

    print("\n\n" + "=" * 60)
    print("\n3. SLIDING WINDOW CHUNKING\n")

    print("""
Dense overlap to ensure no information split:

Fixed chunks:
┌────────┐       ┌────────┐       ┌────────┐
│ Chunk 1│       │ Chunk 2│       │ Chunk 3│
└────────┘       └────────┘       └────────┘
         Gap!           Gap!

Sliding window (50% overlap):
┌────────┐
│ Chunk 1│
└────────┘
    ┌────────┐
    │ Chunk 2│
    └────────┘
        ┌────────┐
        │ Chunk 3│
        └────────┘

No gaps, everything captured!
""")

    code = '''
def sliding_window_chunking(
    text: str,
    window_size: int = 500,
    step_size: int = 250  # 50% overlap
) -> List[Dict]:
    """Sliding window chunking with dense overlap."""

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + window_size
        chunk_text = text[start:end]

        # Skip if too small
        if len(chunk_text) < window_size * 0.3:
            break

        chunks.append({
            'id': chunk_id,
            'text': chunk_text,
            'start': start,
            'end': min(end, len(text)),
            'overlap_ratio': step_size / window_size
        })

        chunk_id += 1
        start += step_size  # Move by step_size

    return chunks

# Example
text = "A" * 1000

chunks = sliding_window_chunking(text, window_size=200, step_size=100)

print(f"Text length: {len(text)}")
print(f"Window size: 200")
print(f"Step size: 100 (50% overlap)")
print(f"Number of chunks: {len(chunks)}")
# Output: ~18 chunks with 50% overlap vs ~5 with no overlap

# Pros:
#  ✓ No information lost at boundaries
#  ✓ Robust retrieval
#  ✓ Good for critical applications

# Cons:
#  ✗ High redundancy
#  ✗ More storage
#  ✗ More embeddings to compute
#  ✗ May retrieve duplicate info
'''

    print(code)

sliding_window_chunking()
```

## Summary

**Key Concepts**:

1. **Document preparation is critical** - chunking and embedding quality determines RAG performance
2. **Chunking breaks documents** into retrievable units (chunks should be semantic, coherent)
3. **Multiple chunking strategies** - fixed-size (simple), sentence-based (better), paragraph-based (semantic), recursive (best), semantic (premium)
4. **Chunk size trade-off** - small (precise, less context) vs large (more context, less precise)
5. **Recommended**: 200-500 tokens with 10-20% overlap
6. **Embedding models vary** - OpenAI (quality, cost), Sentence Transformers (free), balance needed
7. **Document vs query embeddings** differ - asymmetric search, instruction-tuned models help
8. **Long documents** require special handling - chunking, hierarchical, summarization

**Chunking Strategies**:

```
Fixed-Size:
  • Split at fixed character/token count
  • ✓ Simple, predictable
  • ✗ May split awkwardly
  • Use: Quick implementation

Sentence-Based:
  • Group sentences to target size
  • ✓ Respects sentence boundaries
  • ✗ Variable chunk sizes
  • Use: Natural language text

Paragraph-Based:
  • Split on paragraph boundaries
  • ✓ Strong semantic coherence
  • ✗ Highly variable sizes
  • Use: Well-formatted prose

Recursive (RECOMMENDED):
  • Hierarchy: \n\n → \n → . → space
  • ✓ Best of all strategies
  • ✓ Handles edge cases
  • Use: Production systems

Semantic:
  • Split on topic boundaries (embedding-based)
  • ✓ Most coherent chunks
  • ✗ Slower, more complex
  • Use: Premium applications
```

**Chunk Size Guidelines**:

| Use Case          | Token Size | Char Size | Notes          |
| ----------------- | ---------- | --------- | -------------- |
| Short Q&A         | 128-256    | 500-1000  | Precise        |
| General RAG       | 256-512    | 1000-2000 | Recommended    |
| Complex reasoning | 512-1024   | 2000-4000 | More context   |
| Summarization     | 1024-2048  | 4000-8000 | Long documents |

**Chunk Overlap**: 10-20% of chunk size (e.g., 500 char chunks → 50-100 char overlap)

**Embedding Models**:

```
Development/Prototyping:
  → all-MiniLM-L6-v2 (384 dims, fast, free)

Production (managed):
  → OpenAI text-embedding-3-small (1536 dims, quality)

Production (self-hosted):
  → all-mpnet-base-v2 (768 dims) or e5-large-v2 (1024 dims)

Maximum Quality:
  → OpenAI ada-002 or BGE-large-en-v1.5

Multilingual:
  → multilingual-e5-large

Cost-Sensitive:
  → all-MiniLM-L6-v2 (smallest, fastest)
```

**Long Document Strategies**:

1. **Chunking** (default) - Split into pieces that fit embedding model
2. **Hierarchical** - Multiple granularities (doc, section, chunk)
3. **Summarize** - Embed summary, store full text
4. **Late chunking** - Embed full doc, extract chunk representations
5. **Pooling** - Average/max pool chunk embeddings

**Advanced Patterns**:

```
Context-Enriched:
  Add document context to chunks
  "Document: Guide. Section: Install. [chunk text]"
  ✓ Better retrieval, self-contained chunks

Parent-Child:
  Search small chunks (precise)
  Return large parent chunks (context)
  ✓ Best of both worlds

Sliding Window:
  Dense overlap (50%) between chunks
  ✓ No gaps, robust retrieval
  ✗ Higher redundancy/cost
```

**Preprocessing**:

- Remove HTML tags, URLs, extra whitespace
- Fix PDF artifacts (page numbers, hyphenation)
- Handle code blocks separately
- Normalize unicode
- Document-type specific processing

**Best Practices**:

1. Start with **recursive chunking** (200-500 tokens, 10-20% overlap)
2. Use **instruction-tuned models** (e5, BGE) for better asymmetry handling
3. Add **metadata** (source, section, page) to chunks
4. **Preprocess** text before chunking
5. **Experiment** with chunk size for your use case
6. Consider **context-enriched** or **parent-child** patterns
7. **Cache embeddings** to save cost/time
8. Use **batch processing** for efficiency

**Performance Tips**:

- Batch embed multiple chunks together (10-50 per batch)
- Cache embeddings (don't re-compute)
- Use smaller embeddings if quality acceptable (384 vs 1536 dims)
- Preprocess once, reuse chunks
- Consider GPU for large-scale embedding

**Common Pitfalls**:

- Chunks too large → less precision, more noise
- Chunks too small → fragmented context
- No overlap → information split across boundaries
- Poor preprocessing → low-quality embeddings
- Wrong embedding model → poor retrieval
- Not adding metadata → limited filtering options

## Next Steps

- Review [RAG Architecture](rag-architecture.md) for overall system design
- Explore [Vector Databases](vector-databases.md) for storing and searching embeddings
- Learn [Retrieval Strategies](retrieval-strategies.md) for better search
- Study [Reranking and Fusion](reranking-fusion.md) to improve result quality
- Master [RAG Evaluation](rag-evaluation.md) to measure chunking effectiveness
- Apply to production in [Application Patterns](../application_patterns/)
