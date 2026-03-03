# Vector Space Models

## Table of Contents

- [Introduction](#introduction)
- [Vector Space Fundamentals](#vector-space-fundamentals)
- [Bag-of-Words (BoW)](#bag-of-words-bow)
- [Term Frequency (TF)](#term-frequency-tf)
- [Inverse Document Frequency (IDF)](#inverse-document-frequency-idf)
- [TF-IDF: Combining TF and IDF](#tf-idf-combining-tf-and-idf)
- [Document Similarity](#document-similarity)
- [Cosine Similarity](#cosine-similarity)
- [Applications](#applications)
- [Variants and Extensions](#variants-and-extensions)
- [Implementation Best Practices](#implementation-best-practices)
- [Limitations](#limitations)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Vector Space Models (VSM) represent text as vectors in a high-dimensional space. This mathematical representation enables:

- **Document similarity**: Find similar documents
- **Information retrieval**: Match queries to documents
- **Text classification**: Classify documents by content
- **Clustering**: Group similar documents

**Core idea**: Each document is a vector where dimensions correspond to terms, and values reflect term importance.

```
Visual representation:

Document 1: "cat dog bird"
Document 2: "cat mouse"
Document 3: "dog mouse bird"

Vocabulary: [cat, dog, bird, mouse]

Vectors (term frequency):
Doc1: [1, 1, 1, 0]
Doc2: [1, 0, 0, 1]
Doc3: [0, 1, 1, 1]

     cat  dog  bird mouse
       ↓    ↓    ↓    ↓
Doc1: [1,   1,   1,   0]
Doc2: [1,   0,   0,   1]
Doc3: [0,   1,   1,   1]
```

This guide covers the foundational techniques: Bag-of-Words, TF-IDF, and cosine similarity.

## Vector Space Fundamentals

### Why Vectors for Text?

**Advantages**:

1. **Mathematical operations**: Can use linear algebra
2. **Similarity computation**: Distance metrics quantify similarity
3. **Machine learning ready**: Direct input to ML algorithms
4. **Efficient computation**: Optimized vector operations

### The Vector Space Model

**Assumptions**:

- Documents and queries are vectors in same space
- Each dimension = one term (word/phrase)
- Similar documents have similar vectors
- Vector similarity ≈ semantic similarity

**Example**:

```python
# Three documents
docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

# After vectorization (simplified):
# Dimensions: [cat, dog, sat, mat, log, pets]
vectors = [
    [1, 0, 1, 1, 0, 0],  # Doc 1
    [0, 1, 1, 0, 1, 0],  # Doc 2
    [1, 1, 0, 0, 0, 1],  # Doc 3
]

# Now can compute:
# - Distance between documents
# - Similarity to a query
# - Clusters of similar documents
```

### Vocabulary Construction

```python
from collections import Counter

def build_vocabulary(documents):
    """Build vocabulary from documents."""
    vocab = set()

    for doc in documents:
        words = doc.lower().split()
        vocab.update(words)

    # Convert to sorted list for consistent indexing
    vocab = sorted(list(vocab))

    # Create word to index mapping
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    return vocab, word2idx

# Example
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

vocab, word2idx = build_vocabulary(documents)

print(f"Vocabulary size: {len(vocab)}")
print(f"Vocabulary: {vocab}")
print(f"\nWord to index mapping:")
for word, idx in sorted(word2idx.items(), key=lambda x: x[1]):
    print(f"  {word:10} → {idx}")
```

### Document-Term Matrix

The **document-term matrix** is the foundation of VSM:

```
        cat  dog  mat  sat  ...
Doc1    1    0    1    1    ...
Doc2    0    1    0    1    ...
Doc3    1    1    0    0    ...
...
```

```python
import numpy as np

def create_document_term_matrix(documents, vocab, word2idx):
    """Create document-term matrix."""
    num_docs = len(documents)
    vocab_size = len(vocab)

    # Initialize matrix
    dtm = np.zeros((num_docs, vocab_size))

    # Fill matrix
    for doc_idx, doc in enumerate(documents):
        words = doc.lower().split()
        for word in words:
            if word in word2idx:
                word_idx = word2idx[word]
                dtm[doc_idx, word_idx] += 1

    return dtm

# Example
dtm = create_document_term_matrix(documents, vocab, word2idx)

print("Document-Term Matrix:")
print(f"Shape: {dtm.shape}")
print(f"\nFirst 3 documents, first 5 terms:")
print(dtm[:3, :5])
```

## Bag-of-Words (BoW)

### Concept

**Bag-of-Words** represents documents by word frequencies, ignoring:

- Word order
- Grammar
- Syntax

**Example**:

```
Document: "the cat sat on the mat"

BoW representation:
{
    'the': 2,
    'cat': 1,
    'sat': 1,
    'on': 1,
    'mat': 1
}
```

Both "the cat sat on the mat" and "the mat sat on the cat" have the same BoW representation!

### Implementation

```python
from collections import Counter

class BagOfWords:
    """Simple Bag-of-Words implementation."""

    def __init__(self):
        self.vocab = {}
        self.word2idx = {}

    def fit(self, documents):
        """Build vocabulary from documents."""
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update(words)

        self.vocab = sorted(list(all_words))
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

    def transform(self, documents):
        """Convert documents to BoW vectors."""
        vectors = []

        for doc in documents:
            # Count words in document
            word_counts = Counter(doc.lower().split())

            # Create vector
            vector = [0] * len(self.vocab)
            for word, count in word_counts.items():
                if word in self.word2idx:
                    idx = self.word2idx[word]
                    vector[idx] = count

            vectors.append(vector)

        return np.array(vectors)

    def fit_transform(self, documents):
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)

# Example usage
bow = BagOfWords()

documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

vectors = bow.fit_transform(documents)

print("Bag-of-Words vectors:")
print(vectors)

print(f"\nVocabulary: {bow.vocab}")

# Show vector for first document
print(f"\nDocument: '{documents[0]}'")
print("Vector:")
for word, count in zip(bow.vocab, vectors[0]):
    if count > 0:
        print(f"  {word:10}: {int(count)}")
```

### Using Scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer

# Create vectorizer
vectorizer = CountVectorizer()

# Fit and transform
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

bow_matrix = vectorizer.fit_transform(documents)

# Get vocabulary
vocab = vectorizer.get_feature_names_out()

print(f"Vocabulary: {vocab}")
print(f"\nBoW matrix shape: {bow_matrix.shape}")
print(f"BoW matrix (dense):\n{bow_matrix.toarray()}")

# Examine one document
doc_idx = 0
doc_vector = bow_matrix[doc_idx].toarray().flatten()

print(f"\nDocument {doc_idx}: '{documents[doc_idx]}'")
print("Non-zero features:")
for word, count in zip(vocab, doc_vector):
    if count > 0:
        print(f"  {word:10}: {int(count)}")
```

### N-gram Extension

Extend BoW to include phrases:

```python
# Bigram BoW
vectorizer = CountVectorizer(ngram_range=(1, 2))

documents = [
    "natural language processing",
    "machine learning and deep learning",
]

bow = vectorizer.fit_transform(documents)
features = vectorizer.get_feature_names_out()

print("Features (unigrams + bigrams):")
for feature in features:
    print(f"  {feature}")

print(f"\nBoW matrix:\n{bow.toarray()}")
```

**Output includes**:

- Unigrams: "natural", "language", "processing"
- Bigrams: "natural language", "language processing"

## Term Frequency (TF)

### Concept

**Term Frequency** measures how often a term appears in a document.

**Raw count**:
$$\text{TF}(t, d) = \text{count of term } t \text{ in document } d$$

**Normalized** (relative frequency):
$$\text{TF}(t, d) = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in document } d}$$

### Why Normalize?

Longer documents naturally have higher counts:

```
Doc1 (short): "cat dog" → cat: 1, dog: 1
Doc2 (long):  "cat dog cat dog cat dog" → cat: 3, dog: 3

Raw counts make Doc2 seem 3× more relevant
Normalized: both documents have cat: 0.5, dog: 0.5
```

### Implementation

```python
import numpy as np
from collections import Counter

def compute_tf(document):
    """Compute term frequency for a document."""
    words = document.lower().split()
    word_counts = Counter(words)
    total_words = len(words)

    # Normalize by document length
    tf = {word: count / total_words for word, count in word_counts.items()}

    return tf

# Example
doc = "the cat sat on the mat and the cat looked at the mat"

tf = compute_tf(doc)

print("Term Frequencies:")
for word, freq in sorted(tf.items(), key=lambda x: x[1], reverse=True):
    print(f"  {word:10}: {freq:.3f}")
```

### TF Variants

Different TF weighting schemes:

```python
def tf_variants(term_count, doc_length):
    """Different TF computation methods."""

    # 1. Raw count
    tf_raw = term_count

    # 2. Normalized frequency
    tf_normalized = term_count / doc_length

    # 3. Log normalization (reduces impact of high frequencies)
    tf_log = np.log(1 + term_count)

    # 4. Binary (presence/absence)
    tf_binary = 1 if term_count > 0 else 0

    # 5. Augmented (prevents bias toward long documents)
    max_count_in_doc = 10  # Would compute from full document
    tf_augmented = 0.5 + 0.5 * (term_count / max_count_in_doc)

    return {
        'raw': tf_raw,
        'normalized': tf_normalized,
        'log': tf_log,
        'binary': tf_binary,
        'augmented': tf_augmented
    }

# Example
term_count = 5
doc_length = 100

tfs = tf_variants(term_count, doc_length)

print("TF variants for term appearing 5 times in 100-word document:")
for variant, value in tfs.items():
    print(f"  {variant:12}: {value:.4f}")
```

## Inverse Document Frequency (IDF)

### Motivation

Not all words are equally informative:

```
Document: "The cat sat on the mat"

Common words: "the" (appears everywhere)
Rare words: "cat", "mat" (more distinctive)

"the" should have low weight
"cat", "mat" should have high weight
```

### Definition

**IDF** measures how rare/distinctive a term is across all documents:

$$\text{IDF}(t) = \log \frac{N}{\text{df}(t)}$$

Where:

- $N$ = total number of documents
- $\text{df}(t)$ = document frequency (number of documents containing term $t$)

**Intuition**:

- Term in all documents: IDF ≈ 0 (not distinctive)
- Term in few documents: IDF high (very distinctive)

### Implementation

```python
import math
from collections import defaultdict

def compute_idf(documents):
    """Compute IDF for all terms in corpus."""
    N = len(documents)

    # Count document frequency for each term
    df = defaultdict(int)

    for doc in documents:
        words = set(doc.lower().split())  # Use set to count each term once per doc
        for word in words:
            df[word] += 1

    # Compute IDF
    idf = {}
    for word, doc_freq in df.items():
        idf[word] = math.log(N / doc_freq)

    return idf, df

# Example
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets",
    "the mat is red",
    "the log is brown"
]

idf, df = compute_idf(documents)

print(f"Corpus size: {len(documents)} documents\n")

print("IDF scores (sorted by IDF):")
for word, idf_score in sorted(idf.items(), key=lambda x: x[1], reverse=True):
    print(f"  {word:10}: IDF={idf_score:.3f}  (appears in {df[word]}/{len(documents)} docs)")
```

**Expected output**:

```
Rare words (high IDF):
  pets       : IDF=1.609  (appears in 1/5 docs)
  red        : IDF=1.609  (appears in 1/5 docs)

Common words (low IDF):
  the        : IDF=0.000  (appears in 5/5 docs)
  sat        : IDF=0.405  (appears in 2/5 docs)
```

### IDF Variants

```python
def idf_variants(N, df):
    """Different IDF computation methods."""

    # 1. Standard IDF
    idf_standard = math.log(N / df)

    # 2. Smooth IDF (avoid division by zero)
    idf_smooth = math.log((N + 1) / (df + 1)) + 1

    # 3. Max IDF (normalize by maximum df)
    df_max = N  # Maximum possible df
    idf_max = math.log(df_max / df)

    # 4. Probabilistic IDF
    idf_prob = math.log((N - df) / df)

    return {
        'standard': idf_standard,
        'smooth': idf_smooth,
        'max': idf_max,
        'probabilistic': idf_prob
    }

# Example
N = 100  # Total documents
df = 10  # Term appears in 10 documents

idfs = idf_variants(N, df)

print(f"IDF variants (N={N}, df={df}):")
for variant, value in idfs.items():
    print(f"  {variant:15}: {value:.4f}")
```

## TF-IDF: Combining TF and IDF

### The TF-IDF Formula

Combine term frequency and inverse document frequency:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**Interpretation**:

- High TF-IDF: Term frequent in this document AND rare across corpus
- Low TF-IDF: Term infrequent in this document OR common across corpus

**Example**:

```
Document: "the cat sat on the mat and the cat looked at the mat"

Term "the":
  - TF = 6/12 = 0.5 (high frequency in doc)
  - IDF = 0 (appears in all docs)
  - TF-IDF = 0.5 × 0 = 0 (not distinctive)

Term "cat":
  - TF = 2/12 = 0.167 (moderate frequency)
  - IDF = 2.3 (rare in corpus)
  - TF-IDF = 0.167 × 2.3 = 0.384 (distinctive!)
```

### Manual Implementation

```python
import numpy as np
from collections import Counter
import math

class TfidfVectorizer:
    """Simple TF-IDF vectorizer."""

    def __init__(self):
        self.vocab = []
        self.word2idx = {}
        self.idf = {}

    def fit(self, documents):
        """Compute IDF from documents."""
        # Build vocabulary
        vocab_set = set()
        for doc in documents:
            words = doc.lower().split()
            vocab_set.update(words)

        self.vocab = sorted(list(vocab_set))
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

        # Compute IDF
        N = len(documents)
        df = Counter()

        for doc in documents:
            words = set(doc.lower().split())
            df.update(words)

        for word in self.vocab:
            self.idf[word] = math.log(N / df[word])

    def transform(self, documents):
        """Transform documents to TF-IDF vectors."""
        vectors = []

        for doc in documents:
            words = doc.lower().split()
            word_counts = Counter(words)
            doc_length = len(words)

            # Compute TF-IDF for each term
            vector = [0.0] * len(self.vocab)

            for word, count in word_counts.items():
                if word in self.word2idx:
                    idx = self.word2idx[word]
                    tf = count / doc_length
                    idf = self.idf[word]
                    vector[idx] = tf * idf

            vectors.append(vector)

        return np.array(vectors)

    def fit_transform(self, documents):
        """Fit and transform."""
        self.fit(documents)
        return self.transform(documents)

# Example
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets",
    "machine learning is fun",
    "deep learning is powerful"
]

tfidf = TfidfVectorizer()
vectors = tfidf.fit_transform(documents)

print("TF-IDF Matrix:")
print(f"Shape: {vectors.shape}")
print(f"\nFirst document: '{documents[0]}'")

# Show top TF-IDF terms
doc_idx = 0
doc_vector = vectors[doc_idx]
top_indices = np.argsort(doc_vector)[::-1][:5]

print("Top 5 TF-IDF terms:")
for idx in top_indices:
    if doc_vector[idx] > 0:
        word = tfidf.vocab[idx]
        score = doc_vector[idx]
        print(f"  {word:10}: {score:.4f}")
```

### Using Scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets",
    "machine learning is fun",
    "deep learning is powerful"
]

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(feature_names)}")

# Analyze one document
doc_idx = 0
doc_tfidf = tfidf_matrix[doc_idx].toarray().flatten()

print(f"\nDocument {doc_idx}: '{documents[doc_idx]}'")
print("Top TF-IDF terms:")

# Get top terms
top_indices = np.argsort(doc_tfidf)[::-1][:5]
for idx in top_indices:
    if doc_tfidf[idx] > 0:
        print(f"  {feature_names[idx]:10}: {doc_tfidf[idx]:.4f}")
```

### TF-IDF with N-grams

```python
# Include unigrams and bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20)

documents = [
    "natural language processing is fun",
    "machine learning and deep learning",
    "natural language understanding is hard"
]

tfidf = vectorizer.fit_transform(documents)
features = vectorizer.get_feature_names_out()

print("Features (top 20 by TF-IDF):")
for feat in features:
    print(f"  {feat}")

# Show document vectors
print("\nTF-IDF vectors:")
print(tfidf.toarray())
```

## Document Similarity

### Distance vs Similarity

**Distance**: How far apart are vectors?

- Euclidean distance
- Manhattan distance
- Smaller = more similar

**Similarity**: How similar are vectors?

- Cosine similarity
- Jaccard similarity
- Larger = more similar

### Euclidean Distance

$$d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$$

```python
def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two vectors."""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# Example
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

dist = euclidean_distance(vec1, vec2)
print(f"Euclidean distance: {dist:.4f}")
```

**Problem with Euclidean distance for text**:

- Sensitive to document length
- Two documents with same proportions but different lengths have large distance

```python
# Same content, different lengths
doc1 = np.array([1, 1, 0])  # "cat dog"
doc2 = np.array([2, 2, 0])  # "cat cat dog dog"

dist = euclidean_distance(doc1, doc2)
print(f"Distance between proportionally similar docs: {dist:.4f}")
# Large distance, even though content is similar!
```

## Cosine Similarity

### Definition

**Cosine similarity** measures the cosine of the angle between two vectors:

$$\text{similarity}(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**Range**: [-1, 1]

- 1 = identical direction
- 0 = orthogonal (unrelated)
- -1 = opposite direction

**Visual**:

```
        B
       /
      /  θ
     /____A

cosine(θ) = similarity

θ = 0°   → cos(0°) = 1.0   (identical)
θ = 45°  → cos(45°) = 0.7  (similar)
θ = 90°  → cos(90°) = 0.0  (orthogonal)
θ = 180° → cos(180°) = -1.0 (opposite)
```

### Implementation

```python
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

# Example
vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 4, 6])  # Proportional to vec1

similarity = cosine_similarity(vec1, vec2)
print(f"Cosine similarity: {similarity:.4f}")  # 1.0 - identical direction!

# Different example
vec3 = np.array([1, 0, 0])
vec4 = np.array([0, 1, 0])

similarity2 = cosine_similarity(vec3, vec4)
print(f"Cosine similarity (orthogonal): {similarity2:.4f}")  # 0.0
```

### Document Similarity Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "machine learning is a subset of artificial intelligence",
    "deep learning is a subset of machine learning",
    "natural language processing uses machine learning",
    "computer vision is an application of deep learning",
    "the cat sat on the mat"
]

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute pairwise similarities
similarities = cosine_similarity(tfidf_matrix)

print("Pairwise document similarities:")
print("(rows and columns are documents 0-4)\n")

# Print as matrix
for i in range(len(documents)):
    for j in range(len(documents)):
        print(f"{similarities[i][j]:.2f}", end="  ")
    print()

# Find most similar document pairs
print("\nMost similar document pairs:")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        sim = similarities[i][j]
        if sim > 0.1:  # Threshold
            print(f"  Doc {i} <-> Doc {j}: {sim:.4f}")
            print(f"    '{documents[i]}'")
            print(f"    '{documents[j]}'")
            print()
```

### Query-Document Similarity

```python
def search_documents(query, documents, vectorizer, tfidf_matrix, top_k=3):
    """Search documents for most similar to query."""
    # Transform query to TF-IDF vector
    query_vec = vectorizer.transform([query])

    # Compute similarities
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top-k documents
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx],
            'rank': len(results) + 1
        })

    return results

# Example
documents = [
    "machine learning algorithms learn from data",
    "deep learning uses neural networks",
    "natural language processing analyzes text",
    "computer vision processes images",
    "reinforcement learning learns through trial and error"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Search
query = "neural networks for images"
results = search_documents(query, documents, vectorizer, tfidf_matrix)

print(f"Query: '{query}'\n")
print("Top results:")
for result in results:
    print(f"{result['rank']}. (similarity={result['similarity']:.4f})")
    print(f"   {result['document']}")
    print()
```

## Applications

### Information Retrieval

```python
class SimpleSearchEngine:
    """Simple search engine using TF-IDF and cosine similarity."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.tfidf_matrix = None

    def index(self, documents):
        """Index documents."""
        self.documents = documents
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def search(self, query, top_k=5):
        """Search for documents matching query."""
        # Transform query
        query_vec = self.vectorizer.transform([query])

        # Compute similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant documents
                results.append({
                    'doc_id': idx,
                    'document': self.documents[idx],
                    'score': similarities[idx]
                })

        return results

# Example usage
corpus = [
    "Python is a programming language",
    "Java is used for enterprise applications",
    "Machine learning uses Python and R",
    "Web development with JavaScript",
    "Data science requires statistics",
    "Python is popular for data science",
    "Deep learning frameworks use Python"
]

engine = SimpleSearchEngine()
engine.index(corpus)

# Perform searches
queries = [
    "Python programming",
    "data science",
    "web development"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = engine.search(query, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['score']:.4f}] {result['document']}")
```

### Document Clustering

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_documents(documents, n_clusters=3):
    """Cluster documents using K-means on TF-IDF vectors."""
    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # Organize results
    clustered_docs = {i: [] for i in range(n_clusters)}
    for doc_idx, cluster_id in enumerate(clusters):
        clustered_docs[cluster_id].append(documents[doc_idx])

    return clustered_docs, kmeans, vectorizer

# Example
documents = [
    # Sports
    "football game was exciting",
    "basketball player scored points",
    "soccer match ended in draw",
    # Technology
    "new smartphone released today",
    "software update fixes bugs",
    "artificial intelligence advances",
    # Food
    "delicious pasta recipe",
    "chocolate cake is sweet",
    "healthy salad for lunch"
]

clustered, kmeans, vectorizer = cluster_documents(documents, n_clusters=3)

print("Document Clusters:\n")
for cluster_id, docs in clustered.items():
    print(f"Cluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")
    print()
```

### Text Classification Features

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Example: Sentiment classification using TF-IDF features

texts = [
    "I love this product, it's amazing!",
    "Terrible experience, would not recommend",
    "Great quality and fast shipping",
    "Waste of money, very disappointed",
    "Excellent service and support",
    "Poor quality, broke after one use"
]

labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train classifier
classifier = LogisticRegression()
classifier.fit(X, labels)

# Test
test_texts = [
    "This is fantastic!",
    "Very bad experience"
]

test_vectors = vectorizer.transform(test_texts)
predictions = classifier.predict(test_vectors)
probabilities = classifier.predict_proba(test_vectors)

print("Sentiment Classification:\n")
for text, pred, prob in zip(test_texts, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred]
    print(f"Text: '{text}'")
    print(f"Prediction: {sentiment} (confidence: {confidence:.2f})\n")
```

### Duplicate Detection

```python
def find_duplicates(documents, threshold=0.8):
    """Find near-duplicate documents using cosine similarity."""
    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute pairwise similarities
    similarities = cosine_similarity(tfidf_matrix)

    # Find duplicates
    duplicates = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            if similarities[i][j] >= threshold:
                duplicates.append({
                    'doc1': i,
                    'doc2': j,
                    'similarity': similarities[i][j],
                    'text1': documents[i],
                    'text2': documents[j]
                })

    return duplicates

# Example
documents = [
    "machine learning is a subset of AI",
    "ML is a subset of artificial intelligence",  # Near duplicate
    "deep learning uses neural networks",
    "python is a programming language",
    "machine learning is part of AI"  # Near duplicate
]

duplicates = find_duplicates(documents, threshold=0.5)

print("Potential duplicates:\n")
for dup in duplicates:
    print(f"Similarity: {dup['similarity']:.4f}")
    print(f"  Doc {dup['doc1']}: {dup['text1']}")
    print(f"  Doc {dup['doc2']}: {dup['text2']}")
    print()
```

## Variants and Extensions

### Binary Term Weighting

```python
# Presence/absence instead of frequency
vectorizer = TfidfVectorizer(binary=True)

documents = [
    "cat cat cat dog",
    "cat dog"
]

tfidf = vectorizer.fit_transform(documents)
print("Binary TF-IDF (both docs have same vector despite different counts):")
print(tfidf.toarray())
```

### Sublinear TF Scaling

```python
# Use log(TF) instead of TF
vectorizer = TfidfVectorizer(sublinear_tf=True)

# Reduces impact of very high term frequencies
```

### Norm Normalization

```python
# L2 normalization (default)
vectorizer_l2 = TfidfVectorizer(norm='l2')

# L1 normalization
vectorizer_l1 = TfidfVectorizer(norm='l1')

# No normalization
vectorizer_none = TfidfVectorizer(norm=None)

documents = ["the cat sat on the mat"]

print("L2 norm:", vectorizer_l2.fit_transform(documents).toarray())
print("L1 norm:", vectorizer_l1.fit_transform(documents).toarray())
print("No norm:", vectorizer_none.fit_transform(documents).toarray())
```

### BM25: Improvement over TF-IDF

**BM25** (Best Match 25) is a more sophisticated weighting scheme:

$$\text{BM25}(t, d) = \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Where:

- $f(t,d)$ = frequency of term $t$ in document $d$
- $|d|$ = document length
- $\text{avgdl}$ = average document length
- $k_1, b$ = tuning parameters

```python
from rank_bm25 import BM25Okapi  # pip install rank-bm25

documents = [
    "machine learning is great",
    "deep learning uses neural networks",
    "natural language processing"
]

# Tokenize
tokenized_docs = [doc.split() for doc in documents]

# Create BM25 index
bm25 = BM25Okapi(tokenized_docs)

# Query
query = "machine learning"
tokenized_query = query.split()

# Get scores
scores = bm25.get_scores(tokenized_query)

print(f"Query: '{query}'")
print("BM25 scores:")
for doc, score in zip(documents, scores):
    print(f"  {score:.4f}: {doc}")
```

## Implementation Best Practices

### Preprocessing

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import string

def preprocess_text(text):
    """Clean and normalize text."""
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

# Use with vectorizer
documents = [
    "Machine Learning!",
    "DEEP learning...",
    "  Natural   Language Processing  "
]

# Option 1: Preprocess before vectorizing
processed_docs = [preprocess_text(doc) for doc in documents]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(processed_docs)

# Option 2: Use vectorizer's built-in preprocessing
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents='unicode',
    token_pattern=r'\b[a-zA-Z]+\b'  # Only alphabetic tokens
)
tfidf = vectorizer.fit_transform(documents)
```

### Stop Words

```python
# Remove common words
vectorizer = TfidfVectorizer(stop_words='english')

documents = [
    "the cat sat on the mat",
    "the dog sat on the log"
]

tfidf = vectorizer.fit_transform(documents)
features = vectorizer.get_feature_names_out()

print("Features (stop words removed):")
print(features)
# Output: ['cat', 'dog', 'log', 'mat', 'sat']
# 'the', 'on' removed
```

### Min/Max Document Frequency

```python
# Ignore very rare and very common terms
vectorizer = TfidfVectorizer(
    min_df=2,      # Ignore terms appearing in < 2 documents
    max_df=0.8     # Ignore terms appearing in > 80% of documents
)

documents = [
    "cat dog bird",
    "cat mouse",
    "dog mouse bird",
    "cat dog mouse bird",
    "rabbit"  # "rabbit" appears only once - will be ignored
]

tfidf = vectorizer.fit_transform(documents)
features = vectorizer.get_feature_names_out()

print("Features (after filtering):")
print(features)
```

### Max Features

```python
# Limit vocabulary size to most frequent terms
vectorizer = TfidfVectorizer(max_features=100)

# Useful for:
# - Reducing dimensionality
# - Faster computation
# - Limiting memory usage
```

### Sparse Matrix Handling

```python
from scipy.sparse import csr_matrix

# TF-IDF matrices are typically sparse (many zeros)

documents = ["cat dog", "bird fish", "cat fish"]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

print(f"Matrix type: {type(tfidf)}")
print(f"Matrix shape: {tfidf.shape}")
print(f"Non-zero elements: {tfidf.nnz}")
print(f"Sparsity: {100 * (1 - tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1])):.2f}%")

# Convert to dense only when necessary
dense = tfidf.toarray()
print(f"\nDense matrix:\n{dense}")
```

## Limitations

### 1. No Word Order

```
"The dog bit the man"
"The man bit the dog"

→ Identical vectors!
```

### 2. No Semantics

```
"car" and "automobile" are different dimensions
"bank" (river) and "bank" (financial) are the same dimension
```

### 3. High Dimensionality

```python
# Vocabulary size determines dimensions
documents = ["a" * 1000]  # 1000 different words
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

print(f"Vector dimensionality: {tfidf.shape[1]}")
# Can be 10,000+ dimensions for real corpora
```

### 4. Synonyms Not Captured

```
Document 1: "cheap car for sale"
Document 2: "inexpensive automobile available"

TF-IDF: Low similarity (different words)
Human: High similarity (same meaning)
```

### 5. Context Independence

```
"Python programming" → python: 0.5, programming: 0.5
"Snake in Python" → python: 0.5, snake: 0.5

"python" has same weight in both, despite different meanings
```

### Comparison with Modern Approaches

```
Vector Space Models (TF-IDF):
✓ Simple and interpretable
✓ Fast computation
✓ No training required
✓ Works well for keyword matching
✗ No semantic understanding
✗ No word order
✗ High dimensionality
✗ Sparse representations

Modern Embeddings (Word2Vec, BERT):
✓ Semantic understanding
✓ Dense representations
✓ Capture context
✓ Handle synonyms
✗ Require training
✗ Less interpretable
✗ More computational cost
```

## Summary

**Key Concepts**:

1. **Vector Space Model**: Represent documents as vectors in high-dimensional space
2. **Bag-of-Words**: Count word occurrences, ignore order
3. **Term Frequency (TF)**: How often a term appears in a document
4. **Inverse Document Frequency (IDF)**: How rare/distinctive a term is across corpus
5. **TF-IDF**: Combines TF and IDF to weight terms by importance
6. **Cosine Similarity**: Measures angle between document vectors (0-1)

**TF-IDF Formula**:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{\text{df}(t)}$$

**Applications**:

- Information retrieval / search engines
- Document similarity and clustering
- Text classification features
- Duplicate detection
- Keyword extraction

**Advantages**:

- Simple and interpretable
- Fast computation
- No training required
- Effective for many tasks

**Limitations**:

- No word order or context
- No semantic understanding
- High dimensionality
- Cannot capture synonyms or polysemy

**Best Practices**:

- Remove stop words for better discrimination
- Use min_df/max_df to filter extreme terms
- Normalize documents (L2 norm)
- Consider sublinear TF scaling
- Use sparse matrices for efficiency

**When to use**:

- Baseline for text tasks
- Resource-constrained environments
- Interpretability is important
- Small to medium datasets
- Keyword-based matching

**When to use alternatives**:

- Need semantic understanding → Word embeddings (Word2Vec, GloVe)
- Need context → Contextual embeddings (BERT, GPT)
- Complex NLP tasks → Transformer models

## Next Steps

- Explore [Topic Modeling](topic-modeling.md) to discover latent themes using LSA and LDA
- Learn [Classical Text Classification](classical-classification.md) to build classifiers using TF-IDF features
- Study [Word Embeddings](../embeddings/word-embeddings.md) for dense semantic representations
- Progress to [Sentence Embeddings](../embeddings/sentence-embeddings.md) for document-level representations
- Learn about [BM25 and ranking](../retrieval_augmented_generation/retrieval-methods.md) for improved information retrieval
