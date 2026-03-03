# Topic Modeling

## Table of Contents

- [Introduction](#introduction)
- [What is Topic Modeling?](#what-is-topic-modeling)
- [Latent Semantic Analysis (LSA)](#latent-semantic-analysis-lsa)
- [Latent Dirichlet Allocation (LDA)](#latent-dirichlet-allocation-lda)
- [Non-Negative Matrix Factorization (NMF)](#non-negative-matrix-factorization-nmf)
- [Evaluating Topic Models](#evaluating-topic-models)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Visualizing Topics](#visualizing-topics)
- [Applications](#applications)
- [Advanced Techniques](#advanced-techniques)
- [Comparison of Methods](#comparison-of-methods)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Topic modeling is an unsupervised machine learning technique that discovers abstract "topics" within a collection of documents. Unlike keyword extraction or classification, topic modeling:

- **Discovers hidden thematic structure** in large text corpora
- **Requires no labeled data** (fully unsupervised)
- **Reduces dimensionality** from vocabulary size to number of topics
- **Enables document organization** and exploration

**Core idea**: Documents are mixtures of topics, and topics are distributions over words.

```
Example Document: "The cat sat on the mat and the dog ran"

Possible topic composition:
- 70% Topic "Pets" (cat, dog)
- 20% Topic "Actions" (sat, ran)
- 10% Topic "Objects" (mat)

Topic "Pets" word distribution:
- cat: 0.3
- dog: 0.3
- pet: 0.2
- animal: 0.1
- ...
```

This guide covers three major approaches: LSA (matrix factorization), LDA (probabilistic), and NMF (non-negative factorization).

## What is Topic Modeling?

### Formal Definition

Given:

- Collection of $D$ documents
- Vocabulary of $V$ words
- Desired number of topics $K$

Find:

1. **Topic-word distribution**: What words characterize each topic?
2. **Document-topic distribution**: What topics appear in each document?

### Generative Story (LDA perspective)

For each document:

1. Choose distribution over topics
2. For each word position:
   - Pick a topic from document's topic distribution
   - Pick a word from that topic's word distribution

```
Document generation process:

Document → [Topic 1: 0.6, Topic 2: 0.3, Topic 3: 0.1]
          ↓
Generate word 1: Sample Topic 1 → Sample word "machine"
Generate word 2: Sample Topic 1 → Sample word "learning"
Generate word 3: Sample Topic 2 → Sample word "neural"
...
```

### Matrix Factorization View

Topic modeling decomposes document-term matrix:

```
Document-Term Matrix          Document-Topic    Topic-Term
     (D × V)           ≈           (D × K)    ×    (K × V)

 [w₁ w₂ w₃ ... wᵥ]        [t₁ t₂ ... tₖ]   [w₁ w₂ ... wᵥ]
d₁[ 5  2  0 ...  1]    d₁  [0.7 0.2 ... 0.1]  [0.3 0.1 ... 0.0]
d₂[ 3  0  4 ...  2] ≈  d₂  [0.3 0.6 ... 0.1]× [0.2 0.3 ... 0.1]
d₃[ 0  1  2 ...  0]    d₃  [0.1 0.8 ... 0.1]  [0.1 0.2 ... 0.3]
                                  t₁              t₂
```

### Key Assumptions

1. **Bag of words**: Word order doesn't matter
2. **Mixed membership**: Documents can belong to multiple topics
3. **Latent structure**: Topics are hidden/unobserved variables
4. **Sparse representations**: Few topics dominate each document

## Latent Semantic Analysis (LSA)

### Concept

LSA uses **Singular Value Decomposition (SVD)** to reduce dimensionality of document-term matrix.

$$\text{DTM} = U \Sigma V^T$$

Where:

- $U$ = document-topic matrix
- $\Sigma$ = diagonal matrix of singular values (topic strengths)
- $V^T$ = topic-word matrix

By keeping only top $K$ singular values, we get $K$ latent topics.

### Mathematical Foundation

```
Original DTM (D × V):
  Many dimensions (vocabulary size)
  Sparse (most entries are 0)

SVD factorization:
  DTM = U Σ V^T

Truncated SVD (keep top K):
  DTM ≈ U_k Σ_k V_k^T

Result:
  U_k (D × K): Document representations in topic space
  V_k^T (K × V): Topic representations in word space
```

### Implementation with Scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

documents = [
    "machine learning algorithms learn from data",
    "deep learning uses neural networks",
    "neural networks are inspired by the brain",
    "machine learning is a subset of artificial intelligence",
    "cats and dogs are common pets",
    "dogs require regular exercise and care",
    "cats are independent animals",
    "pets bring joy to families"
]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
dtm = vectorizer.fit_transform(documents)

# Apply LSA
n_topics = 2
lsa = TruncatedSVD(n_components=n_topics, random_state=42)
doc_topic = lsa.fit_transform(dtm)

# Get topic-word matrix
topic_word = lsa.components_

# Display topics
feature_names = vectorizer.get_feature_names_out()

print("Topics discovered by LSA:\n")
for topic_idx, topic in enumerate(topic_word):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")

# Display document-topic distributions
print("\nDocument-Topic Distributions:\n")
for doc_idx, doc in enumerate(documents):
    print(f"Doc {doc_idx}: {doc_topic[doc_idx]}")
    print(f"  '{doc[:50]}...'")
    print()
```

### LSA Step-by-Step

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.linalg import svd

# Small example for clarity
docs = [
    "dog cat pet",
    "dog pet animal",
    "machine learning ai",
    "learning algorithm ai"
]

# Create term-document matrix
vectorizer = CountVectorizer()
tdm = vectorizer.fit_transform(docs).toarray().T  # Transpose for term-document

print("Term-Document Matrix:")
print(tdm)
print(f"Shape: {tdm.shape} (terms × documents)")

# Perform SVD
U, sigma, Vt = svd(tdm, full_matrices=False)

print(f"\nU shape (term-topic): {U.shape}")
print(f"Sigma shape: {sigma.shape}")
print(f"Vt shape (topic-document): {Vt.shape}")

# Keep top 2 dimensions
k = 2
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]

# Reconstruct
tdm_reconstructed = U_k @ sigma_k @ Vt_k

print(f"\nReconstructed TDM (rank-{k} approximation):")
print(tdm_reconstructed)

# Interpret topics
terms = vectorizer.get_feature_names_out()
print("\nTopic interpretation:")
for i in range(k):
    top_term_idx = np.argsort(np.abs(U_k[:, i]))[-3:][::-1]
    top_terms = [terms[idx] for idx in top_term_idx]
    print(f"  Topic {i}: {', '.join(top_terms)}")
```

### LSA for Document Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

# Documents in LSA space
doc_topic = lsa.fit_transform(dtm)

# Compute similarities
similarities = cosine_similarity(doc_topic)

print("Document similarities in topic space:\n")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        if similarities[i][j] > 0.5:
            print(f"Doc {i} <-> Doc {j}: {similarities[i][j]:.3f}")
            print(f"  '{documents[i]}'")
            print(f"  '{documents[j]}'")
            print()
```

### LSA Advantages and Disadvantages

**Advantages**:

- Fast computation (SVD is efficient)
- Works well for semantic similarity
- Reduces noise and sparsity
- Handles synonyms (similar words cluster in topic space)

**Disadvantages**:

- No interpretable probability distributions
- Can produce negative values
- Hard to determine optimal number of topics
- Doesn't model document generation process

## Latent Dirichlet Allocation (LDA)

### Probabilistic Generative Model

LDA is a **Bayesian probabilistic model** that assumes:

1. Each document is a mixture of topics
2. Each topic is a mixture of words
3. Both mixtures are drawn from Dirichlet distributions

**Parameters**:

- $\alpha$ (alpha): Controls document-topic density
  - High α: Documents have many topics
  - Low α: Documents have few topics
- $\beta$ (beta/eta): Controls topic-word density
  - High β: Topics have many words
  - Low β: Topics have few words

### LDA Generative Process

```
For each document d:
  1. Draw topic proportions θ_d ~ Dirichlet(α)

  For each word position n in document d:
    2. Draw topic z_n ~ Categorical(θ_d)
    3. Draw word w_n ~ Categorical(φ_{z_n})

Where:
  θ_d = topic distribution for document d
  φ_k = word distribution for topic k
```

### Visual Representation

```
Document: "machine learning is powerful"

Document-Topic distribution:
  Topic AI: 0.7  ████████████████
  Topic Stats: 0.2  ████
  Topic Biology: 0.1  ██

Topic AI word distribution:
  machine: 0.15  ████████
  learning: 0.12  ███████
  neural: 0.10  ██████
  data: 0.08  ████
  ...

Generate word:
  1. Sample topic from document (e.g., "AI" with p=0.7)
  2. Sample word from topic (e.g., "machine" with p=0.15)
```

### Implementation with Scikit-learn

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "machine learning algorithms learn from data",
    "deep learning uses neural networks",
    "neural networks are inspired by the brain",
    "machine learning is a subset of artificial intelligence",
    "natural language processing analyzes text",
    "text classification uses machine learning",
    "cats and dogs are common pets",
    "dogs require regular exercise and care",
    "cats are independent animals",
    "pets bring joy to families",
    "exercise is good for health",
    "regular care keeps pets healthy"
]

# LDA requires raw counts (not TF-IDF)
vectorizer = CountVectorizer(max_features=100, stop_words='english')
dtm = vectorizer.fit_transform(documents)

# Fit LDA
n_topics = 3
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20,
    learning_method='batch'
)

doc_topic_dist = lda.fit_transform(dtm)

# Display topics
feature_names = vectorizer.get_feature_names_out()

def display_topics(model, feature_names, n_top_words=10):
    """Display top words for each topic."""
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]

        print(f"\nTopic {topic_idx}:")
        for word, weight in zip(top_words, top_weights):
            print(f"  {word:15} ({weight:.4f})")

print("LDA Topics:")
display_topics(lda, feature_names)

# Display document-topic distributions
print("\n" + "="*60)
print("Document-Topic Distributions:")
print("="*60)

for doc_idx, (doc, dist) in enumerate(zip(documents, doc_topic_dist)):
    dominant_topic = np.argmax(dist)
    print(f"\nDoc {doc_idx} (dominant topic: {dominant_topic})")
    print(f"  '{doc}'")
    print(f"  Distribution: {dist}")
```

### LDA with Gensim

```python
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Preprocessing
def preprocess(text):
    """Tokenize and clean text."""
    # Tokenize
    tokens = word_tokenize(text.lower())

    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]

    return tokens

# Process documents
documents = [
    "machine learning algorithms learn from data",
    "deep learning uses neural networks",
    "neural networks are inspired by the brain",
    "cats and dogs are common pets",
    "dogs require regular exercise and care"
]

processed_docs = [preprocess(doc) for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=2,
    random_state=42,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

# Display topics
print("Gensim LDA Topics:\n")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}:")
    print(f"  {topic}")
    print()

# Get document topics
print("Document-Topic Distributions:\n")
for doc_idx, doc in enumerate(corpus):
    topic_dist = lda_model.get_document_topics(doc)
    print(f"Doc {doc_idx}: {documents[doc_idx]}")
    print(f"  Topics: {topic_dist}")
    print()
```

### Tuning LDA Hyperparameters

```python
# Effect of alpha (document-topic density)

# Low alpha (sparse, few topics per document)
lda_low_alpha = LatentDirichletAllocation(
    n_components=5,
    doc_topic_prior=0.1,  # alpha
    random_state=42
)

# High alpha (dense, many topics per document)
lda_high_alpha = LatentDirichletAllocation(
    n_components=5,
    doc_topic_prior=10.0,  # alpha
    random_state=42
)

# Compare distributions
doc_topic_low = lda_low_alpha.fit_transform(dtm)
doc_topic_high = lda_high_alpha.fit_transform(dtm)

print("Effect of Alpha:\n")
print("Low Alpha (sparse topics per doc):")
print(doc_topic_low[0])  # First document

print("\nHigh Alpha (many topics per doc):")
print(doc_topic_high[0])  # First document
```

### Interpreting LDA Results

```python
def interpret_lda_topics(lda_model, vectorizer, n_words=10):
    """Interpret LDA topics with word probabilities."""
    feature_names = vectorizer.get_feature_names_out()

    for topic_idx, topic in enumerate(lda_model.components_):
        # Get top words
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_probs = topic[top_indices] / topic[top_indices].sum()  # Normalize

        print(f"\nTopic {topic_idx}:")

        # Bar chart (ASCII)
        for word, prob in zip(top_words, top_probs):
            bar_length = int(prob * 50)
            bar = '█' * bar_length
            print(f"  {word:15} {bar} {prob:.3f}")

interpret_lda_topics(lda, vectorizer, n_words=8)
```

## Non-Negative Matrix Factorization (NMF)

### Concept

NMF factorizes the document-term matrix into two non-negative matrices:

$$V \approx WH$$

Where:

- $V$ (D × V): Document-term matrix
- $W$ (D × K): Document-topic matrix
- $H$ (K × V): Topic-term matrix
- All entries ≥ 0

**Key difference from LSA**: Non-negativity constraint makes topics more interpretable.

### Mathematical Formulation

**Objective**: Minimize reconstruction error

$$\min_{W,H} ||V - WH||^2$$

Subject to: $W_{ij} \geq 0, H_{kj} \geq 0$

**Optimization**: Multiplicative update rules or coordinate descent.

### Implementation

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "machine learning algorithms learn from data",
    "deep learning uses neural networks",
    "neural networks are inspired by the brain",
    "machine learning is subset of artificial intelligence",
    "cats and dogs are common pets",
    "dogs require regular exercise and care",
    "cats are independent animals",
    "pets bring joy to families"
]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf = vectorizer.fit_transform(documents)

# Apply NMF
n_topics = 2
nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
doc_topic = nmf.fit_transform(tfidf)
topic_word = nmf.components_

# Display topics
feature_names = vectorizer.get_feature_names_out()

print("NMF Topics:\n")
for topic_idx, topic in enumerate(topic_word):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    top_weights = [topic[i] for i in top_words_idx]

    print(f"Topic {topic_idx}:")
    for word, weight in zip(top_words, top_weights):
        print(f"  {word:15}: {weight:.4f}")
    print()

# Document representations
print("Document-Topic Matrix:")
print(doc_topic)
```

### NMF vs LSA vs LDA

```python
# Compare all three methods

from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

documents = [
    "machine learning and artificial intelligence",
    "deep learning neural networks",
    "natural language processing",
    "computer vision image recognition",
    "cats dogs pets animals",
    "pet care and veterinary",
]

# LSA (uses TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
lsa = TruncatedSVD(n_components=2, random_state=42)
lsa_topics = lsa.fit_transform(tfidf)

# NMF (uses TF-IDF)
nmf = NMF(n_components=2, random_state=42)
nmf_topics = nmf.fit_transform(tfidf)

# LDA (uses raw counts)
count_vectorizer = CountVectorizer(stop_words='english')
counts = count_vectorizer.fit_transform(documents)
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda_topics = lda.fit_transform(counts)

# Display results
def show_topics(model, vectorizer, method_name):
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n{method_name} Topics:")

    for idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        print(f"  Topic {idx}: {', '.join(top_words)}")

show_topics(lsa, tfidf_vectorizer, "LSA")
show_topics(nmf, tfidf_vectorizer, "NMF")
show_topics(lda, count_vectorizer, "LDA")
```

## Evaluating Topic Models

### Perplexity

**Perplexity** measures how well the model predicts held-out documents (lower is better).

$$\text{Perplexity} = \exp\left(-\frac{\sum_d \log p(w_d)}{N}\right)$$

Where $N$ is total number of words.

```python
from sklearn.model_selection import train_test_split

# Split data
docs_train, docs_test = train_test_split(documents, test_size=0.2, random_state=42)

# Vectorize
vectorizer = CountVectorizer(stop_words='english')
dtm_train = vectorizer.fit_transform(docs_train)
dtm_test = vectorizer.transform(docs_test)

# Train LDA
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(dtm_train)

# Evaluate perplexity
train_perplexity = lda.perplexity(dtm_train)
test_perplexity = lda.perplexity(dtm_test)

print(f"Train Perplexity: {train_perplexity:.2f}")
print(f"Test Perplexity:  {test_perplexity:.2f}")

# Lower perplexity = better model
```

### Coherence

**Coherence** measures semantic similarity between top words in topics (higher is better).

**PMI (Pointwise Mutual Information) Coherence**:

$$\text{Coherence} = \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \log \frac{P(w_i, w_j)}{P(w_i)P(w_j)}$$

```python
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

# Preprocess
processed_docs = [preprocess(doc) for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    random_state=42
)

# Calculate coherence
coherence_model = CoherenceModel(
    model=lda_model,
    texts=processed_docs,
    dictionary=dictionary,
    coherence='c_v'
)

coherence_score = coherence_model.get_coherence()
print(f"Coherence Score: {coherence_score:.4f}")
```

### Topic Diversity

Measures how distinct topics are from each other.

```python
def topic_diversity(topic_word_matrix, top_n=10):
    """Calculate topic diversity (percentage of unique words in top-n)."""
    unique_words = set()
    total_words = 0

    for topic in topic_word_matrix:
        top_indices = topic.argsort()[-top_n:][::-1]
        unique_words.update(top_indices)
        total_words += top_n

    diversity = len(unique_words) / total_words
    return diversity

# Example
diversity = topic_diversity(lda.components_, top_n=10)
print(f"Topic Diversity: {diversity:.2f}")
print(f"(1.0 = all topics have different top words)")
print(f"(0.1 = all topics have same top words)")
```

### Choosing Number of Topics

```python
import matplotlib.pyplot as plt

def evaluate_topic_numbers(dtm, min_topics=2, max_topics=10):
    """Evaluate different numbers of topics."""
    perplexities = []
    coherences = []
    topic_counts = range(min_topics, max_topics + 1)

    for n_topics in topic_counts:
        # Train LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        lda.fit(dtm)

        # Calculate perplexity
        perp = lda.perplexity(dtm)
        perplexities.append(perp)

        print(f"Topics: {n_topics}, Perplexity: {perp:.2f}")

    return topic_counts, perplexities

# Example (conceptual - would need actual data)
# topic_counts, perplexities = evaluate_topic_numbers(dtm)
#
# plt.plot(topic_counts, perplexities, marker='o')
# plt.xlabel('Number of Topics')
# plt.ylabel('Perplexity')
# plt.title('Perplexity vs Number of Topics')
# plt.show()
```

## Hyperparameter Tuning

### Grid Search for LDA

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_components': [3, 5, 7],
    'learning_decay': [0.5, 0.7, 0.9],
    'max_iter': [10, 20, 30]
}

# Note: GridSearchCV for LDA requires custom scoring
# Here's a simplified version

best_perplexity = float('inf')
best_params = None

for n_topics in [3, 5, 7]:
    for decay in [0.5, 0.7, 0.9]:
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_decay=decay,
            random_state=42,
            max_iter=20
        )
        lda.fit(dtm)

        perplexity = lda.perplexity(dtm)

        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_params = {'n_components': n_topics, 'learning_decay': decay}

        print(f"Topics: {n_topics}, Decay: {decay}, Perplexity: {perplexity:.2f}")

print(f"\nBest parameters: {best_params}")
print(f"Best perplexity: {best_perplexity:.2f}")
```

### Alpha and Beta Tuning

```python
# Test different alpha values
alphas = [0.01, 0.1, 1.0, 10.0]

for alpha in alphas:
    lda = LatentDirichletAllocation(
        n_components=3,
        doc_topic_prior=alpha,
        random_state=42
    )
    doc_topics = lda.fit_transform(dtm)

    # Measure sparsity (average number of topics per document)
    avg_topics = (doc_topics > 0.01).sum(axis=1).mean()

    print(f"Alpha: {alpha:5.2f}, Avg topics per doc: {avg_topics:.2f}")
```

## Visualizing Topics

### pyLDAvis

```python
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Train Gensim LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,
    random_state=42,
    passes=10
)

# Visualize
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)

# Save to HTML
# pyLDAvis.save_html(vis, 'lda_visualization.html')
```

### Word Clouds for Topics

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def visualize_topics_wordcloud(model, vectorizer, n_topics):
    """Create word clouds for each topic."""
    feature_names = vectorizer.get_feature_names_out()

    fig, axes = plt.subplots(1, n_topics, figsize=(20, 5))

    for topic_idx, topic in enumerate(model.components_):
        # Create word frequency dict for this topic
        word_freq = {feature_names[i]: topic[i] for i in range(len(feature_names))}

        # Generate word cloud
        wordcloud = WordCloud(
            width=400,
            height=300,
            background_color='white'
        ).generate_from_frequencies(word_freq)

        # Plot
        axes[topic_idx].imshow(wordcloud, interpolation='bilinear')
        axes[topic_idx].set_title(f'Topic {topic_idx}')
        axes[topic_idx].axis('off')

    plt.tight_layout()
    plt.show()

# Example (conceptual)
# visualize_topics_wordcloud(lda, vectorizer, n_topics=3)
```

### Topic Distribution Heatmap

```python
import seaborn as sns

def plot_document_topic_heatmap(doc_topic_dist, documents):
    """Plot heatmap of document-topic distributions."""
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        doc_topic_dist,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=[f'Topic {i}' for i in range(doc_topic_dist.shape[1])],
        yticklabels=[doc[:30] + '...' for doc in documents]
    )

    plt.title('Document-Topic Distribution')
    plt.xlabel('Topics')
    plt.ylabel('Documents')
    plt.tight_layout()
    plt.show()

# Example (conceptual)
# plot_document_topic_heatmap(doc_topic_dist, documents)
```

## Applications

### Document Clustering

```python
from sklearn.cluster import KMeans

# Get document representations in topic space
doc_topics = lda.transform(dtm)

# Cluster in topic space
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(doc_topics)

# Organize by cluster
clustered_docs = {i: [] for i in range(3)}
for doc_idx, cluster_id in enumerate(clusters):
    clustered_docs[cluster_id].append(documents[doc_idx])

print("Clustered Documents:\n")
for cluster_id, docs in clustered_docs.items():
    print(f"Cluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")
    print()
```

### Document Recommendation

```python
def recommend_similar_documents(query_idx, doc_topic_dist, documents, top_n=3):
    """Recommend documents similar to query document."""
    from sklearn.metrics.pairwise import cosine_similarity

    # Query document's topic distribution
    query_dist = doc_topic_dist[query_idx].reshape(1, -1)

    # Compute similarities
    similarities = cosine_similarity(query_dist, doc_topic_dist).flatten()

    # Get top-n (excluding query itself)
    similar_indices = similarities.argsort()[::-1][1:top_n+1]

    print(f"Query document: '{documents[query_idx]}'\n")
    print("Similar documents:")
    for idx in similar_indices:
        print(f"  (similarity={similarities[idx]:.3f}) {documents[idx]}")

# Example
recommend_similar_documents(0, doc_topic_dist, documents, top_n=3)
```

### Trend Analysis

```python
def analyze_topic_trends(documents_by_time, n_topics=5):
    """Analyze how topics change over time."""
    # documents_by_time: dict of {timestamp: [documents]}

    topic_trends = {i: [] for i in range(n_topics)}
    timestamps = sorted(documents_by_time.keys())

    for timestamp in timestamps:
        docs = documents_by_time[timestamp]

        # Vectorize
        dtm = vectorizer.transform(docs)

        # Get topic distributions
        doc_topics = lda.transform(dtm)

        # Average topic proportions for this time period
        avg_topics = doc_topics.mean(axis=0)

        for topic_idx, proportion in enumerate(avg_topics):
            topic_trends[topic_idx].append(proportion)

    # Plot trends
    for topic_idx, proportions in topic_trends.items():
        plt.plot(timestamps, proportions, label=f'Topic {topic_idx}', marker='o')

    plt.xlabel('Time')
    plt.ylabel('Topic Proportion')
    plt.title('Topic Trends Over Time')
    plt.legend()
    plt.show()

# Example (conceptual)
# documents_by_year = {
#     2018: ["doc1", "doc2", ...],
#     2019: ["doc3", "doc4", ...],
#     ...
# }
# analyze_topic_trends(documents_by_year)
```

### Keyword Extraction

```python
def extract_document_keywords(doc_idx, doc_topics, topic_words, feature_names, top_n=10):
    """Extract keywords for a document based on its topic distribution."""
    # Get document's topic distribution
    doc_topic_dist = doc_topics[doc_idx]

    # Weight words by document's topic preferences
    word_scores = {}

    for topic_idx, topic_weight in enumerate(doc_topic_dist):
        topic_word_dist = topic_words[topic_idx]

        for word_idx, word_weight in enumerate(topic_word_dist):
            word = feature_names[word_idx]
            score = topic_weight * word_weight

            if word in word_scores:
                word_scores[word] += score
            else:
                word_scores[word] = score

    # Sort by score
    keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return keywords

# Example
doc_idx = 0
keywords = extract_document_keywords(
    doc_idx,
    doc_topic_dist,
    lda.components_,
    vectorizer.get_feature_names_out()
)

print(f"Keywords for: '{documents[doc_idx]}'\n")
for word, score in keywords:
    print(f"  {word:15}: {score:.4f}")
```

## Advanced Techniques

### Dynamic Topic Models

Model topic evolution over time:

```python
# Conceptual - requires gensim's LdaSeqModel
from gensim.models import LdaSeqModel

# Documents grouped by time slices
time_slice = [10, 15, 20]  # Number of docs in each time period

# Train dynamic topic model
# dtm = LdaSeqModel(
#     corpus=corpus,
#     time_slice=time_slice,
#     id2word=dictionary,
#     num_topics=5
# )

# Analyze how topics change over time
# for time_idx in range(len(time_slice)):
#     topics = dtm.print_topics(time=time_idx)
```

### Hierarchical Topic Models

Organize topics in a hierarchy:

```python
# Conceptual - hierarchical LDA
# Parent topics → Child topics → Words

# Topic hierarchy:
#   Science
#     ├── Computer Science
#     │   ├── Machine Learning
#     │   └── Databases
#     └── Biology
#         ├── Genetics
#         └── Ecology
```

### Correlated Topic Models (CTM)

Allow topics to be correlated (relax LDA's assumption of independence):

```python
# In CTM, topics can co-occur
# E.g., "machine learning" and "statistics" topics often appear together

# LDA: Topic distributions are independent
# CTM: Models correlation between topics
```

### Author-Topic Models

Model both documents and authors:

```python
# Each author has a distribution over topics
# Each document samples from its authors' topic distributions

# Useful for:
# - Analyzing author interests
# - Recommending collaborators
# - Predicting authorship
```

## Comparison of Methods

### LSA vs LDA vs NMF

```python
import pandas as pd

comparison = pd.DataFrame({
    'Aspect': [
        'Type',
        'Input',
        'Probabilistic',
        'Interpretability',
        'Sparsity',
        'Speed',
        'Use Case'
    ],
    'LSA': [
        'Matrix factorization',
        'TF-IDF',
        'No',
        'Moderate',
        'Dense',
        'Fast',
        'Semantic similarity'
    ],
    'LDA': [
        'Probabilistic model',
        'Raw counts',
        'Yes',
        'High',
        'Sparse',
        'Slower',
        'Topic discovery'
    ],
    'NMF': [
        'Matrix factorization',
        'TF-IDF',
        'No',
        'High',
        'Sparse',
        'Fast',
        'Parts-based decomposition'
    ]
})

print(comparison.to_string(index=False))
```

### Performance Characteristics

```
                    LSA         LDA         NMF
Complexity          O(DVK)      O(DIKN)     O(DVK)
(D=docs, V=vocab, K=topics, I=iterations, N=passes)

Scalability         Good        Moderate    Good
Memory Usage        High        Moderate    Moderate
Training Time       Fast        Slow        Fast
Inference Time      Fast        Moderate    Fast

Handles Synonyms    Yes         Yes         Yes
Topic Quality       Moderate    High        High
Probabilistic       No          Yes         No
Non-negative        No          Yes         Yes
```

### When to Use Each

**LSA**:

- Fast semantic similarity needed
- Don't need probabilistic interpretation
- Have TF-IDF features

**LDA**:

- Need interpretable topics
- Want generative model
- Have sufficient computational resources
- Need probability distributions

**NMF**:

- Want non-negative, interpretable parts
- Need fast training
- Have TF-IDF or count features
- Prefer sparse representations

## Summary

**Key Concepts**:

1. **Topic Modeling**: Unsupervised discovery of latent topics in document collections
2. **LSA**: SVD-based dimensionality reduction, fast but less interpretable
3. **LDA**: Probabilistic generative model, highly interpretable topics
4. **NMF**: Non-negative factorization, sparse and interpretable
5. **Evaluation**: Perplexity (lower better), coherence (higher better), diversity
6. **Hyperparameters**: Number of topics (K), alpha (doc-topic density), beta (topic-word density)

**Topic Modeling Pipeline**:

1. Preprocess text (tokenize, remove stopwords)
2. Create document-term matrix (counts or TF-IDF)
3. Choose and train model (LSA/LDA/NMF)
4. Evaluate (perplexity, coherence, manual inspection)
5. Tune hyperparameters
6. Apply to downstream tasks

**Evaluation Metrics**:

- **Perplexity**: Model's predictive performance
- **Coherence**: Semantic consistency of topics
- **Diversity**: Distinctiveness of topics
- **Human judgment**: Manual topic quality assessment

**Applications**:

- Document organization and clustering
- Recommendation systems
- Trend analysis over time
- Keyword extraction
- Information retrieval
- Exploratory data analysis

**Best Practices**:

- Start with more topics than needed, prune later
- Use domain knowledge to validate topics
- Combine multiple evaluation metrics
- Visualize results (word clouds, pyLDAvis)
- Iterate on preprocessing and hyperparameters
- Consider ensemble of multiple models

**Limitations**:

- Bag-of-words (no word order)
- Fixed number of topics
- Assumes topics are static (except dynamic models)
- Can be sensitive to preprocessing
- Evaluation is partly subjective

## Next Steps

- Learn [Classical Text Classification](classical-classification.md) to use topic features for classification
- Study [Word Embeddings](../embeddings/word-embeddings.md) for dense semantic representations
- Explore [Contextual Embeddings](../embeddings/contextual-embeddings.md) for context-aware representations
- Progress to [Language Models](../language_models/) for neural approaches to text understanding
- Apply topic models to [Document Retrieval](../retrieval_augmented_generation/retrieval-methods.md)
- Learn about [Evaluation Metrics](../evaluation/metrics-and-benchmarks.md) for comprehensive model assessment
