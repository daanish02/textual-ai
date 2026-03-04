# Sentence Embeddings

## Table of Contents

- [Introduction](#introduction)
- [From Words to Sentences](#from-words-to-sentences)
- [Averaging Word Embeddings](#averaging-word-embeddings)
- [Doc2Vec](#doc2vec)
- [Sentence-BERT (SBERT)](#sentence-bert-sbert)
- [Universal Sentence Encoder](#universal-sentence-encoder)
- [InferSent](#infersent)
- [Comparing Sentence Embeddings](#comparing-sentence-embeddings)
- [Applications](#applications)
- [Fine-tuning for Specific Tasks](#fine-tuning-for-specific-tasks)
- [Evaluation Benchmarks](#evaluation-benchmarks)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

While word embeddings capture meaning of individual words, many NLP tasks require understanding entire sentences, paragraphs, or documents. **Sentence embeddings** encode variable-length text into fixed-size vectors that preserve semantic meaning.

**Key challenge**: How to aggregate word-level information into sentence-level representation?

```
Word embeddings:
  "cat" → [0.2, -0.5, 0.8]
  "sat" → [0.1, 0.3, -0.2]
  "mat" → [-0.1, 0.4, 0.5]

Sentence embedding:
  "the cat sat on the mat" → [0.15, 0.1, 0.37, ...]

Need to preserve:
- Semantic meaning
- Word order (sometimes)
- Context relationships
```

**Use cases**:

- **Semantic search**: Find similar documents
- **Clustering**: Group similar texts
- **Classification**: Classify entire documents
- **Paraphrase detection**: Are two sentences saying same thing?
- **Question answering**: Match questions to answers

This guide covers methods from simple averaging to sophisticated neural encoders like Sentence-BERT.

## From Words to Sentences

### The Compositionality Problem

**Challenge**: Sentence meaning ≠ simple sum of word meanings

```
Examples where word order matters:

1. "The dog bit the man" vs "The man bit the dog"
   - Same words, completely different meaning!

2. "not good" vs "good"
   - Negation reverses meaning

3. "I really love this" vs "I really hate this"
   - One positive word changes everything
```

### Desired Properties

Good sentence embeddings should:

1. **Semantic similarity**: Similar sentences → similar vectors
2. **Contextual awareness**: Consider word order and dependencies
3. **Fixed dimensionality**: Variable-length input → fixed-size output
4. **Transferable**: Work across different tasks/domains
5. **Efficient**: Fast to compute

### Evaluation Preview

```python
# How to evaluate sentence embeddings?

# 1. Semantic Textual Similarity (STS)
sentence1 = "The cat is on the mat"
sentence2 = "A feline is sitting on a rug"
# Should have high similarity (paraphrase)

sentence3 = "The dog is in the garden"
# Should have lower similarity to sentence1

# 2. Classification accuracy
# Use embeddings as features for sentiment analysis, topic classification, etc.

# 3. Clustering quality
# Group similar documents together

# 4. Retrieval performance
# Find relevant documents for queries
```

## Averaging Word Embeddings

### Simple Average

**Most basic approach**: Average all word vectors

```python
import numpy as np
from gensim.downloader import load

# Load pre-trained word embeddings
word_model = load('glove-wiki-gigaword-100')

def sentence_embedding_avg(sentence, word_model):
    """Create sentence embedding by averaging word vectors."""
    words = sentence.lower().split()

    # Get word vectors
    word_vectors = []
    for word in words:
        if word in word_model:
            word_vectors.append(word_model[word])

    if word_vectors:
        # Average
        return np.mean(word_vectors, axis=0)
    else:
        # Return zero vector if no words found
        return np.zeros(word_model.vector_size)

# Example
sentences = [
    "The cat sat on the mat",
    "The dog sat on the rug",
    "Machine learning is fascinating",
]

embeddings = [sentence_embedding_avg(s, word_model) for s in sentences]

print("Sentence embeddings shape:", embeddings[0].shape)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)

print("\nSentence similarities:")
for i, sent_i in enumerate(sentences):
    for j, sent_j in enumerate(sentences):
        if i < j:
            print(f"  '{sent_i}' <-> '{sent_j}': {similarities[i][j]:.3f}")
```

### Weighted Average (TF-IDF)

**Improvement**: Weight words by importance (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def sentence_embedding_tfidf(sentences, word_model):
    """Weighted average using TF-IDF weights."""
    # Compute TF-IDF weights
    vectorizer = TfidfVectorizer()
    vectorizer.fit(sentences)

    embeddings = []

    for sentence in sentences:
        words = sentence.lower().split()

        # Get TF-IDF weights for this sentence
        tfidf_vector = vectorizer.transform([sentence])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = dict(zip(feature_names, tfidf_vector.toarray()[0]))

        # Weighted average
        weighted_sum = np.zeros(word_model.vector_size)
        total_weight = 0

        for word in words:
            if word in word_model and word in tfidf_scores:
                weight = tfidf_scores[word]
                weighted_sum += weight * word_model[word]
                total_weight += weight

        if total_weight > 0:
            embeddings.append(weighted_sum / total_weight)
        else:
            embeddings.append(np.zeros(word_model.vector_size))

    return np.array(embeddings)

# Example
corpus = [
    "The cat sat on the mat",
    "The dog sat on the rug",
    "Machine learning is fascinating"
]

tfidf_embeddings = sentence_embedding_tfidf(corpus, word_model)
print("TF-IDF weighted embeddings shape:", tfidf_embeddings.shape)
```

### SIF (Smooth Inverse Frequency)

**Better weighting**: Downweight common words more aggressively

$$\text{weight}(w) = \frac{a}{a + p(w)}$$

Where $p(w)$ is word frequency, $a$ is smoothing parameter (typically 0.001)

```python
from collections import Counter

def sif_embedding(sentences, word_model, a=0.001):
    """SIF: Smooth Inverse Frequency weighting."""
    # Compute word frequencies
    all_words = []
    for sentence in sentences:
        all_words.extend(sentence.lower().split())

    word_counts = Counter(all_words)
    total_words = len(all_words)
    word_freqs = {word: count / total_words for word, count in word_counts.items()}

    embeddings = []

    for sentence in sentences:
        words = sentence.lower().split()

        weighted_sum = np.zeros(word_model.vector_size)
        total_weight = 0

        for word in words:
            if word in word_model:
                # SIF weight
                freq = word_freqs.get(word, 0)
                weight = a / (a + freq)

                weighted_sum += weight * word_model[word]
                total_weight += weight

        if total_weight > 0:
            embeddings.append(weighted_sum / total_weight)
        else:
            embeddings.append(np.zeros(word_model.vector_size))

    embeddings = np.array(embeddings)

    # Remove first principal component (common discourse vector)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(embeddings)
    pc = pca.components_[0]

    # Remove projection onto first PC
    embeddings = embeddings - embeddings @ pc.reshape(-1, 1) @ pc.reshape(1, -1)

    return embeddings

# Example
sif_embeddings = sif_embedding(corpus, word_model)
print("SIF embeddings shape:", sif_embeddings.shape)
```

### Limitations of Averaging

```python
# Problems with averaging:

# 1. Word order lost
sent1 = "dog bites man"
sent2 = "man bites dog"
# Same average!

emb1 = sentence_embedding_avg(sent1, word_model)
emb2 = sentence_embedding_avg(sent2, word_model)
similarity = cosine_similarity([emb1], [emb2])[0][0]
print(f"Similarity (should be low): {similarity:.3f}")  # Often high!

# 2. Negation not captured
sent3 = "this is good"
sent4 = "this is not good"
# "not" gets averaged in, but doesn't flip meaning

# 3. No learned composition
# Cannot learn how words interact syntactically
```

## Doc2Vec

### Paragraph Vector (PV)

**Doc2Vec** (Le & Mikolov, 2014) extends Word2Vec to learn document embeddings directly.

**Two architectures**:

1. **PV-DM** (Distributed Memory): Like CBOW with document vector
2. **PV-DBOW** (Distributed Bag of Words): Like Skip-gram with document vector

### PV-DM Architecture

```
Architecture:

Document ID + Context words → Predict target word

Example:
  Document D1: "The cat sat on the mat"

  Input: [D1, "the", "cat", "on"] → Predict: "sat"
```

**Key idea**: Each document gets a unique vector that's learned during training

```
       Target word (output)
              ↑
         Softmax layer
              ↑
    Average/Concatenate
         ↙        ↘
   Doc vector   Context words
   (learned)    (learned)
```

### Implementation with Gensim

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Prepare data
documents = [
    "The cat sat on the mat",
    "The dog sat on the rug",
    "Cats and dogs are animals",
    "Machine learning is fascinating",
    "Deep learning uses neural networks",
    "Natural language processing is important"
]

# Tag documents (required for Doc2Vec)
tagged_docs = [
    TaggedDocument(words=doc.lower().split(), tags=[str(i)])
    for i, doc in enumerate(documents)
]

# Train Doc2Vec
doc2vec_model = Doc2Vec(
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=100,
    dm=1,  # 1 = PV-DM, 0 = PV-DBOW
    seed=42
)

# Build vocabulary
doc2vec_model.build_vocab(tagged_docs)

# Train
doc2vec_model.train(
    tagged_docs,
    total_examples=doc2vec_model.corpus_count,
    epochs=doc2vec_model.epochs
)

# Get document vectors
doc_vector = doc2vec_model.dv['0']  # First document
print(f"Document vector shape: {doc_vector.shape}")
print(f"Document vector (first 10 dims): {doc_vector[:10]}")

# Find similar documents
similar_docs = doc2vec_model.dv.most_similar('0', topn=3)
print(f"\nDocuments similar to '{documents[0]}':")
for doc_id, score in similar_docs:
    print(f"  Doc {doc_id}: {documents[int(doc_id)]}")
    print(f"  Similarity: {score:.3f}")
```

### Inferring Vectors for New Documents

```python
# Infer vector for new, unseen document
new_doc = "The bird flew in the sky"
new_doc_tokens = new_doc.lower().split()

# Infer vector
new_vector = doc2vec_model.infer_vector(new_doc_tokens)

print(f"\nNew document: '{new_doc}'")
print(f"Inferred vector shape: {new_vector.shape}")

# Find similar documents
from scipy.spatial.distance import cosine

similarities = []
for i, doc in enumerate(documents):
    doc_vec = doc2vec_model.dv[str(i)]
    sim = 1 - cosine(new_vector, doc_vec)
    similarities.append((i, sim))

similarities.sort(key=lambda x: x[1], reverse=True)

print(f"\nMost similar documents:")
for doc_id, sim in similarities[:3]:
    print(f"  {documents[doc_id]}: {sim:.3f}")
```

### PV-DM vs PV-DBOW

```python
# Train both architectures
dm_model = Doc2Vec(tagged_docs, vector_size=100, dm=1, epochs=50)  # PV-DM
dbow_model = Doc2Vec(tagged_docs, vector_size=100, dm=0, epochs=50)  # PV-DBOW

# Compare on same document
doc_id = '0'
dm_vec = dm_model.dv[doc_id]
dbow_vec = dbow_model.dv[doc_id]

print("PV-DM vector:", dm_vec[:5])
print("PV-DBOW vector:", dbow_vec[:5])

# Often best to concatenate both!
combined = np.concatenate([dm_vec, dbow_vec])
print(f"Combined vector shape: {combined.shape}")
```

## Sentence-BERT (SBERT)

### Motivation

**BERT problem**: Computing similarity requires forward pass through network for each pair

```
To compare 10,000 sentences:
- Need 10,000 * 10,000 = 100M forward passes!
- Computationally infeasible
```

**SBERT solution** (Reimers & Gurevych, 2019): Create fixed sentence embeddings that can be compared with cosine similarity

### Siamese Network Architecture

```
Architecture:

Sentence A → BERT → Pooling → Embedding A
Sentence B → BERT → Pooling → Embedding B
                                  ↓
                        Cosine similarity / Classification
```

**Training objective**: Learn embeddings where similar sentences are close

### Using Pre-trained SBERT

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "The cat sat on the mat",
    "A feline rested on a rug",
    "The dog ran in the park",
    "Machine learning is fascinating"
]

embeddings = sbert_model.encode(sentences)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)

print("\nSentence similarities:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"  '{sentences[i][:30]}...' <-> '{sentences[j][:30]}...': {similarities[i][j]:.3f}")
```

### Semantic Search

```python
def semantic_search(query, corpus, model, top_k=3):
    """Find most similar sentences to query."""
    # Encode query and corpus
    query_embedding = model.encode([query])
    corpus_embeddings = model.encode(corpus)

    # Compute similarities
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'sentence': corpus[idx],
            'score': similarities[idx]
        })

    return results

# Example
corpus = [
    "Python is a programming language",
    "Machine learning uses Python",
    "The cat sat on the mat",
    "Natural language processing is a field of AI",
    "Deep learning requires GPUs",
    "I love pizza for dinner"
]

query = "What programming language is used for ML?"

results = semantic_search(query, corpus, sbert_model, top_k=3)

print(f"Query: '{query}'\n")
print("Top results:")
for i, result in enumerate(results, 1):
    print(f"{i}. (score={result['score']:.3f}) {result['sentence']}")
```

### Different SBERT Models

```python
# Different models for different use cases

models = {
    'all-MiniLM-L6-v2': 'Fast, good quality (384 dim)',
    'all-mpnet-base-v2': 'Best quality (768 dim)',
    'paraphrase-MiniLM-L6-v2': 'Paraphrase detection (384 dim)',
    'multi-qa-MiniLM-L6-cos-v1': 'Question answering (384 dim)',
    'msmarco-distilbert-base-v4': 'Passage retrieval (768 dim)',
}

# Load specific model
# model = SentenceTransformer('all-mpnet-base-v2')

# Model comparison
print("SBERT model comparison:\n")
for model_name, description in models.items():
    print(f"{model_name}:")
    print(f"  {description}")
    print()
```

### Fine-tuning SBERT

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Prepare training data (sentence pairs with labels)
train_examples = [
    InputExample(texts=['The cat sat on mat', 'A feline rested on rug'], label=0.9),
    InputExample(texts=['The cat sat on mat', 'The dog ran in park'], label=0.3),
    InputExample(texts=['I love pizza', 'Pizza is delicious'], label=0.8),
    InputExample(texts=['I love pizza', 'The weather is nice'], label=0.1),
]

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define loss (Cosine Similarity Loss)
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100
)

print("Model fine-tuned!")
```

## Universal Sentence Encoder

### Overview

**Universal Sentence Encoder** (USE) from Google encodes sentences into 512-dimensional embeddings optimized for transfer learning.

**Two variants**:

1. **Transformer-based**: More accurate, slower
2. **DAN (Deep Averaging Network)**: Faster, slightly less accurate

### Using TensorFlow Hub

```python
import tensorflow_hub as hub
import tensorflow as tf

# Load Universal Sentence Encoder
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Example (conceptual - requires TensorFlow Hub)
def encode_with_use(sentences):
    """Encode sentences with Universal Sentence Encoder."""
    embeddings = embed(sentences)
    return embeddings.numpy()

# sentences = [
#     "The cat sat on the mat",
#     "A feline rested on a rug",
#     "The dog ran in the park"
# ]
#
# use_embeddings = encode_with_use(sentences)
# print(f"USE embeddings shape: {use_embeddings.shape}")
```

### USE Applications

```python
# 1. Semantic similarity
def semantic_similarity_use(sent1, sent2, encoder):
    """Compute semantic similarity using USE."""
    embeddings = encoder([sent1, sent2])
    similarity = np.inner(embeddings[0], embeddings[1])
    return similarity

# 2. Text classification
def classify_with_use(texts, labels, test_texts, encoder):
    """Simple classifier using USE embeddings."""
    from sklearn.linear_model import LogisticRegression

    # Encode
    train_embeddings = encoder(texts).numpy()
    test_embeddings = encoder(test_texts).numpy()

    # Train classifier
    clf = LogisticRegression()
    clf.fit(train_embeddings, labels)

    # Predict
    predictions = clf.predict(test_embeddings)
    return predictions

# 3. Clustering
def cluster_with_use(texts, n_clusters, encoder):
    """Cluster texts using USE embeddings."""
    from sklearn.cluster import KMeans

    embeddings = encoder(texts).numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    return clusters
```

## InferSent

### Overview

**InferSent** (Conneau et al., 2017) learns sentence embeddings via Natural Language Inference (NLI) task.

**Training objective**: Learn to classify sentence relationships

- Entailment: A implies B
- Contradiction: A contradicts B
- Neutral: No relationship

```
Examples:

Entailment:
  A: "A man is playing guitar"
  B: "A person is making music"

Contradiction:
  A: "A man is playing guitar"
  B: "A man is sleeping"

Neutral:
  A: "A man is playing guitar"
  B: "A woman is cooking"
```

### Architecture

```
Sentence A → BiLSTM → max pooling → Embedding A
Sentence B → BiLSTM → max pooling → Embedding B
                                         ↓
                              [u, v, |u-v|, u*v]
                                         ↓
                                   Classifier
                                         ↓
                          [Entailment, Contradiction, Neutral]
```

### Using InferSent (Conceptual)

```python
# Note: InferSent requires specific setup and pre-trained models

# Load model (conceptual)
# from models import InferSent
# model = InferSent({'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048})
# model.load_state_dict(torch.load('infersent.allnli.pickle'))

# Set word embeddings
# model.set_w2v_path('glove.840B.300d.txt')

# Build vocabulary
# model.build_vocab(sentences, tokenize=True)

# Encode sentences
# embeddings = model.encode(sentences, tokenize=True)

# Conceptual example
def infersent_similarity(sent1, sent2, model):
    """Compute similarity using InferSent."""
    embeddings = model.encode([sent1, sent2], tokenize=True)
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity
```

## Comparing Sentence Embeddings

### Performance Comparison

```python
import time

def compare_methods(sentences):
    """Compare different sentence embedding methods."""

    results = {}

    # 1. Average word embeddings
    start = time.time()
    avg_embs = [sentence_embedding_avg(s, word_model) for s in sentences]
    results['Average'] = {
        'time': time.time() - start,
        'embeddings': avg_embs,
        'dimension': len(avg_embs[0])
    }

    # 2. Doc2Vec
    start = time.time()
    tagged = [TaggedDocument(s.split(), [str(i)]) for i, s in enumerate(sentences)]
    d2v = Doc2Vec(tagged, vector_size=100, epochs=20)
    d2v_embs = [d2v.infer_vector(s.split()) for s in sentences]
    results['Doc2Vec'] = {
        'time': time.time() - start,
        'embeddings': d2v_embs,
        'dimension': len(d2v_embs[0])
    }

    # 3. SBERT
    start = time.time()
    sbert_embs = sbert_model.encode(sentences)
    results['SBERT'] = {
        'time': time.time() - start,
        'embeddings': sbert_embs,
        'dimension': sbert_embs.shape[1]
    }

    return results

# Test
test_sentences = [
    "The quick brown fox jumps over the lazy dog",
    "A fast auburn fox leaps above a sleepy canine",
    "Machine learning is a subset of artificial intelligence",
]

comparison = compare_methods(test_sentences)

print("Method Comparison:\n")
for method, info in comparison.items():
    print(f"{method}:")
    print(f"  Time: {info['time']:.4f}s")
    print(f"  Dimension: {info['dimension']}")
    print()
```

### Quality Metrics

```python
def evaluate_sentence_embeddings(embeddings, sentence_pairs_with_labels):
    """Evaluate embedding quality using labeled sentence pairs."""
    from scipy.stats import spearmanr

    predicted_sims = []
    human_sims = []

    for (sent1, sent2), human_score in sentence_pairs_with_labels:
        # Get embeddings
        idx1 = sentences.index(sent1)
        idx2 = sentences.index(sent2)

        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]

        # Compute similarity
        pred_sim = cosine_similarity([emb1], [emb2])[0][0]

        predicted_sims.append(pred_sim)
        human_sims.append(human_score)

    # Compute correlation
    correlation, p_value = spearmanr(predicted_sims, human_sims)

    return {
        'correlation': correlation,
        'p_value': p_value
    }

# Example evaluation
sentence_pairs = [
    (("The cat sat on mat", "A feline rested on rug"), 0.9),
    (("The cat sat on mat", "The dog ran in park"), 0.3),
    (("I love pizza", "Pizza is delicious"), 0.8),
]

# Evaluate
# metrics = evaluate_sentence_embeddings(embeddings, sentence_pairs)
# print(f"Spearman correlation: {metrics['correlation']:.3f}")
```

## Applications

### Document Clustering

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cluster_documents(documents, n_clusters=3):
    """Cluster documents using sentence embeddings."""
    # Get embeddings
    embeddings = sbert_model.encode(documents)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Visualize (2D projection)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    for i, (x, y) in enumerate(coords):
        cluster = clusters[i]
        plt.scatter(x, y, c=f'C{cluster}', s=100)
        plt.annotate(documents[i][:20] + '...', (x, y), fontsize=8)

    plt.title('Document Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()

    # Return clustered documents
    clustered = {i: [] for i in range(n_clusters)}
    for doc, cluster in zip(documents, clusters):
        clustered[cluster].append(doc)

    return clustered

# Example
docs = [
    "Python is a programming language",
    "Machine learning uses Python",
    "The cat sat on the mat",
    "Dogs are loyal animals",
    "Natural language processing is fascinating",
    "Deep learning is a subset of machine learning",
]

clusters = cluster_documents(docs, n_clusters=2)

print("Clusters:")
for cluster_id, docs_in_cluster in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for doc in docs_in_cluster:
        print(f"  - {doc}")
```

### Duplicate Detection

```python
def find_duplicates(documents, threshold=0.85):
    """Find near-duplicate documents."""
    # Get embeddings
    embeddings = sbert_model.encode(documents)

    # Compute pairwise similarities
    similarities = cosine_similarity(embeddings)

    # Find duplicates
    duplicates = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            if similarities[i][j] >= threshold:
                duplicates.append({
                    'doc1_idx': i,
                    'doc2_idx': j,
                    'doc1': documents[i],
                    'doc2': documents[j],
                    'similarity': similarities[i][j]
                })

    return duplicates

# Example
documents = [
    "Machine learning is amazing",
    "ML is truly amazing",  # Near duplicate
    "The cat sat on the mat",
    "A feline rested on a rug",  # Near duplicate
    "Python is a programming language",
]

dupes = find_duplicates(documents, threshold=0.7)

print("Potential duplicates:\n")
for dupe in dupes:
    print(f"Similarity: {dupe['similarity']:.3f}")
    print(f"  Doc 1: {dupe['doc1']}")
    print(f"  Doc 2: {dupe['doc2']}")
    print()
```

### Question Answering Retrieval

```python
def qa_retrieval(question, context_paragraphs, model, top_k=3):
    """Retrieve most relevant paragraphs for a question."""
    # Encode question and paragraphs
    question_emb = model.encode([question])
    paragraph_embs = model.encode(context_paragraphs)

    # Compute similarities
    similarities = cosine_similarity(question_emb, paragraph_embs)[0]

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'paragraph': context_paragraphs[idx],
            'score': similarities[idx],
            'rank': len(results) + 1
        })

    return results

# Example
question = "What is machine learning?"

context = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
    "The weather today is sunny and warm with temperatures reaching 25 degrees Celsius.",
    "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data.",
    "Python is a popular programming language used in data science and machine learning applications.",
]

results = qa_retrieval(question, context, sbert_model, top_k=2)

print(f"Question: {question}\n")
print("Most relevant paragraphs:")
for result in results:
    print(f"\n{result['rank']}. (score={result['score']:.3f})")
    print(f"   {result['paragraph']}")
```

### Paraphrase Detection

```python
def detect_paraphrase(sent1, sent2, model, threshold=0.75):
    """Detect if two sentences are paraphrases."""
    embeddings = model.encode([sent1, sent2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    is_paraphrase = similarity >= threshold

    return {
        'is_paraphrase': is_paraphrase,
        'similarity': similarity,
        'confidence': 'high' if abs(similarity - threshold) > 0.1 else 'low'
    }

# Test cases
test_pairs = [
    ("The cat sat on the mat", "A feline rested on a rug"),
    ("I love programming", "Programming is my passion"),
    ("The sky is blue", "The grass is green"),
]

print("Paraphrase Detection:\n")
for sent1, sent2 in test_pairs:
    result = detect_paraphrase(sent1, sent2, sbert_model)
    status = "✓ Paraphrase" if result['is_paraphrase'] else "✗ Not paraphrase"

    print(f"{status} (similarity={result['similarity']:.3f})")
    print(f"  Sent 1: {sent1}")
    print(f"  Sent 2: {sent2}")
    print()
```

## Fine-tuning for Specific Tasks

### Domain Adaptation

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def fine_tune_domain(model, domain_sentences, base_model_name='all-MiniLM-L6-v2'):
    """Fine-tune SBERT on domain-specific data."""
    # Create training examples (using self-supervision)
    train_examples = []

    for sent in domain_sentences:
        # Create positive pair (sentence with itself)
        train_examples.append(InputExample(texts=[sent, sent], label=1.0))

        # Create negative pairs (with other sentences)
        for other_sent in np.random.choice(domain_sentences, size=2, replace=False):
            if other_sent != sent:
                train_examples.append(InputExample(texts=[sent, other_sent], label=0.0))

    # DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Loss
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100
    )

    return model

# Example: Medical domain
medical_sentences = [
    "Patient presents with acute chest pain",
    "Diagnosis indicates coronary artery disease",
    "Treatment plan includes medication and lifestyle changes",
    "Follow-up appointment scheduled in two weeks",
]

# Fine-tune
# model = SentenceTransformer('all-MiniLM-L6-v2')
# fine_tuned_model = fine_tune_domain(model, medical_sentences)
```

### Task-Specific Training

```python
def create_nli_training_data():
    """Create training data for Natural Language Inference."""
    # Format: (premise, hypothesis, label)
    # Labels: 0=contradiction, 1=entailment, 2=neutral

    examples = [
        InputExample(
            texts=['A man is playing guitar', 'A person is making music'],
            label=1  # Entailment
        ),
        InputExample(
            texts=['A man is playing guitar', 'A man is sleeping'],
            label=0  # Contradiction
        ),
        InputExample(
            texts=['A man is playing guitar', 'It is raining outside'],
            label=2  # Neutral
        ),
    ]

    return examples

# Train NLI model
def train_nli_model(model, train_examples):
    """Train model on NLI task."""
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Use Softmax loss for classification
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3  # 3 classes: entailment, contradiction, neutral
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5
    )

    return model
```

## Evaluation Benchmarks

### STS Benchmark

```python
def evaluate_on_sts(model, test_pairs):
    """Evaluate on Semantic Textual Similarity benchmark."""
    from scipy.stats import pearsonr, spearmanr

    predicted_scores = []
    gold_scores = []

    for sent1, sent2, gold_score in test_pairs:
        # Encode
        emb1, emb2 = model.encode([sent1, sent2])

        # Compute similarity
        pred_score = cosine_similarity([emb1], [emb2])[0][0]

        predicted_scores.append(pred_score)
        gold_scores.append(gold_score / 5.0)  # Normalize to [0, 1]

    # Compute correlations
    pearson_corr, _ = pearsonr(predicted_scores, gold_scores)
    spearman_corr, _ = spearmanr(predicted_scores, gold_scores)

    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }

# Example STS pairs (score 0-5)
sts_test = [
    ("A man is playing a flute", "A man is playing a bamboo flute", 4.6),
    ("A girl is styling her hair", "A girl is brushing her hair", 3.3),
    ("The young lady enjoys listening to the guitar", "The woman is playing the violin", 1.0),
]

# Evaluate
# scores = evaluate_on_sts(sbert_model, sts_test)
# print(f"Pearson correlation: {scores['pearson']:.3f}")
# print(f"Spearman correlation: {scores['spearman']:.3f}")
```

### SICK and STS-B Datasets

```python
# Standard benchmarks for sentence embeddings

benchmarks = {
    'STS-B': {
        'description': 'Semantic Textual Similarity Benchmark',
        'metric': 'Spearman correlation',
        'size': '8,628 sentence pairs',
        'score_range': '0-5'
    },
    'SICK': {
        'description': 'Sentences Involving Compositional Knowledge',
        'metric': 'Pearson/Spearman correlation',
        'size': '10,000 sentence pairs',
        'tasks': ['Relatedness', 'Entailment']
    },
    'MRPC': {
        'description': 'Microsoft Research Paraphrase Corpus',
        'metric': 'Accuracy/F1',
        'size': '5,801 sentence pairs',
        'task': 'Paraphrase detection'
    }
}

print("Standard Evaluation Benchmarks:\n")
for name, info in benchmarks.items():
    print(f"{name}: {info['description']}")
    print(f"  Metric: {info['metric']}")
    print(f"  Size: {info['size']}")
    print()
```

### Model Performance Comparison

```python
# Typical performance on STS-B (Spearman correlation)

performance_table = {
    'Average GloVe': 0.58,
    'Average Word2Vec': 0.61,
    'InferSent': 0.75,
    'Universal Sentence Encoder': 0.78,
    'Sentence-BERT (RoBERTa)': 0.86,
    'SimCSE': 0.84,
}

print("Model Performance on STS-B (Spearman correlation):\n")
for model, score in sorted(performance_table.items(), key=lambda x: x[1]):
    bar = '█' * int(score * 50)
    print(f"{model:30} {bar} {score:.2f}")
```

## Summary

**Key Concepts**:

1. **Sentence embeddings** encode variable-length text into fixed-size vectors
2. **Simple methods** (averaging, TF-IDF weighting) provide baselines but lose word order
3. **Doc2Vec** extends Word2Vec to learn document representations directly
4. **Sentence-BERT** creates efficient, high-quality embeddings using Siamese networks
5. **Universal Sentence Encoder** provides general-purpose sentence embeddings
6. **Evaluation** uses similarity benchmarks (STS, SICK) and downstream tasks

**Method Comparison**:

| Method                     | Quality   | Speed  | Dimensions | OOV Handling |
| -------------------------- | --------- | ------ | ---------- | ------------ |
| Average Word Emb           | Low       | Fast   | 100-300    | Poor         |
| SIF                        | Medium    | Fast   | 100-300    | Poor         |
| Doc2Vec                    | Medium    | Medium | 100-300    | Inferred     |
| InferSent                  | High      | Slow   | 4096       | Poor         |
| Universal Sentence Encoder | High      | Medium | 512        | Good         |
| Sentence-BERT              | Very High | Fast   | 384-768    | Excellent    |

**When to Use Each**:

- **Averaging**: Quick baseline, limited resources
- **Doc2Vec**: Need document-level training, moderate data
- **SBERT**: Production systems, semantic search, high quality needed
- **USE**: Google ecosystem, general-purpose tasks
- **InferSent**: Research, NLI-related tasks

**Applications**:

- **Semantic search**: Find similar documents/sentences
- **Clustering**: Group related texts
- **Paraphrase detection**: Identify similar meanings
- **Question answering**: Retrieve relevant passages
- **Duplicate detection**: Find near-duplicate content
- **Classification**: Use as features for ML models

**Best Practices**:

1. **Start simple**: Try averaging before complex methods
2. **Choose right model**: SBERT for most modern applications
3. **Fine-tune**: Adapt pre-trained models to your domain
4. **Evaluate properly**: Use standard benchmarks (STS-B)
5. **Consider speed**: Balance quality vs inference time
6. **Normalize embeddings**: Unit length for cosine similarity
7. **Batch encode**: More efficient for large datasets

**Limitations**:

- Still context-independent across sentence boundaries
- May not capture long-range dependencies well
- Domain shift can hurt performance
- Computational cost for large-scale retrieval
- Fine-tuning requires labeled data

## Next Steps

- Study [Contextual Embeddings](contextual-embeddings.md) to learn about context-dependent representations (ELMo, BERT)
- Explore [Embedding Spaces](embedding-spaces.md) to understand geometric properties and operations
- Apply to [Retrieval Systems](../retrieval_augmented_generation/retrieval-methods.md) for RAG applications
- Learn about [Dense Retrieval](../retrieval_augmented_generation/dense-retrieval.md) with neural embeddings
- Study [Semantic Search](../application_patterns/semantic-search.md) architectures
- Progress to [Transformer Encoders](../language_models/encoder-decoder-models.md) for understanding underlying architectures
