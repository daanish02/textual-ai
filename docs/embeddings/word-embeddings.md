# Word Embeddings

## Table of Contents

- [Introduction](#introduction)
- [From Sparse to Dense Representations](#from-sparse-to-dense-representations)
- [Word2Vec: Skip-gram and CBOW](#word2vec-skip-gram-and-cbow)
- [GloVe: Global Vectors](#glove-global-vectors)
- [FastText: Subword Embeddings](#fasttext-subword-embeddings)
- [Training Your Own Embeddings](#training-your-own-embeddings)
- [Pre-trained Embeddings](#pre-trained-embeddings)
- [Embedding Arithmetic and Analogies](#embedding-arithmetic-and-analogies)
- [Evaluation Methods](#evaluation-methods)
- [Fine-tuning Embeddings](#fine-tuning-embeddings)
- [Limitations and Challenges](#limitations-and-challenges)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Word embeddings are dense vector representations of words that capture semantic meaning. Unlike sparse one-hot encodings or TF-IDF vectors, embeddings place semantically similar words close together in a continuous vector space.

**The fundamental insight**: "You shall know a word by the company it keeps" - words appearing in similar contexts have similar meanings.

```
Sparse representation (one-hot):
  cat:  [0, 0, 0, 1, 0, 0, ..., 0]  (vocab_size dimensions, mostly zeros)
  dog:  [0, 0, 0, 0, 1, 0, ..., 0]

Dense representation (embedding):
  cat:  [0.2, -0.5, 0.8, ..., 0.1]  (typically 50-300 dimensions)
  dog:  [0.3, -0.4, 0.7, ..., 0.2]  (similar to 'cat'!)
```

**Key benefits**:

1. **Semantic similarity**: Similar words have similar vectors
2. **Dimensionality reduction**: From vocab_size (10K-100K) to ~300 dimensions
3. **Learned representations**: Capture meaning from data
4. **Transfer learning**: Pre-trained embeddings work across tasks

This guide covers the three foundational embedding methods: Word2Vec, GloVe, and FastText.

## From Sparse to Dense Representations

### The Problem with Sparse Vectors

**One-hot encoding**:

```python
import numpy as np

vocab = ['cat', 'dog', 'bird', 'fish']
vocab_size = len(vocab)

# One-hot vectors
cat_onehot = [1, 0, 0, 0]
dog_onehot = [0, 1, 0, 0]

# No notion of similarity!
similarity = np.dot(cat_onehot, dog_onehot)
print(f"Similarity between 'cat' and 'dog': {similarity}")  # 0.0

# All words are equally distant
print(f"'cat' vs 'dog' same distance as 'cat' vs 'fish'")
```

**Problems**:

- Vocabulary size explosion (10K-100K dimensions)
- No semantic relationships captured
- Sparse (wasteful memory)
- Cannot handle out-of-vocabulary words

### Dense Embeddings Solution

**Key idea**: Map words to low-dimensional continuous vectors

```python
# Dense embeddings (learned from data)
embedding_dim = 4  # Much smaller than vocab_size

cat_embedding = np.array([0.8, 0.3, -0.2, 0.1])
dog_embedding = np.array([0.7, 0.4, -0.1, 0.2])
fish_embedding = np.array([0.1, -0.5, 0.8, 0.3])

# Cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim_cat_dog = cosine_similarity(cat_embedding, dog_embedding)
sim_cat_fish = cosine_similarity(cat_embedding, fish_embedding)

print(f"Similarity (cat, dog): {sim_cat_dog:.3f}")    # ~0.95 (high!)
print(f"Similarity (cat, fish): {sim_cat_fish:.3f}")  # ~0.20 (low)
```

### Distributional Hypothesis

**Core principle**: Words in similar contexts have similar meanings

```
Context windows:

"The cat sat on the mat"
"The dog sat on the rug"

Both 'cat' and 'dog' appear with:
- 'the' (left context)
- 'sat' (right context)

→ 'cat' and 'dog' should have similar embeddings
```

### Visualizing Word Space

```
2D projection of word embeddings:

       dog •

    cat •      • puppy

       animal •


    car •          • vehicle

        • bicycle
```

## Word2Vec: Skip-gram and CBOW

### Overview

**Word2Vec** (Mikolov et al., 2013) learns embeddings by predicting context words.

**Two architectures**:

1. **Skip-gram**: Predict context words from center word
2. **CBOW** (Continuous Bag of Words): Predict center word from context

```
Sentence: "the quick brown fox jumps"
Window size: 2

Skip-gram (center → context):
  Input: "brown"
  Predict: "the", "quick", "fox", "jumps"

CBOW (context → center):
  Input: "the", "quick", "fox", "jumps"
  Predict: "brown"
```

### Skip-gram Architecture

```
Architecture:

Input word → Embedding layer → Hidden layer → Output (context words)

Example:
  "fox" → [0.2, -0.5, ...] → softmax → probabilities for all words

Training objective:
  Maximize: P(context | center_word)
```

**Visual**:

```
            Context words (output)
            ↑  ↑  ↑  ↑
            Softmax layer
                ↑
         Hidden layer (embeddings)
                ↑
         One-hot input (center word)
```

### Mathematical Formulation

**Skip-gram objective**: Maximize log probability of context words

$$\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

Where:

- $T$ = corpus size
- $c$ = context window size
- $w_t$ = word at position $t$

**Probability calculation** (using softmax):

$$P(w_O | w_I) = \frac{\exp(v'_{w_O}^T v_{w_I})}{\sum_{w=1}^{V} \exp(v'_w^T v_{w_I})}$$

Where:

- $v_{w_I}$ = embedding of input word
- $v'_{w_O}$ = embedding of output word

### CBOW Architecture

```
Context words (input) → Average → Hidden → Predict center word

Example:
  ["the", "quick", "fox", "jumps"] → average embeddings → predict "brown"
```

**Comparison**:

| Aspect          | Skip-gram       | CBOW                 |
| --------------- | --------------- | -------------------- |
| Input           | Center word     | Context words        |
| Output          | Context words   | Center word          |
| Speed           | Slower          | Faster               |
| Data efficiency | Needs more data | Works with less data |
| Rare words      | Better          | Worse                |
| Use case        | Large corpus    | Small corpus         |

### Negative Sampling

**Problem**: Computing softmax over entire vocabulary (10K-100K words) is expensive!

**Solution**: Negative sampling - only update small sample of "negative" words

```python
# Instead of computing softmax over all words:
# P(w_context | w_center) over 100K words

# Negative sampling:
# For each positive pair (w_center, w_context):
#   - Sample k "negative" words (not in context)
#   - Objective: distinguish positive from negative

# Example:
# Positive: ("fox", "brown")
# Negatives: ("fox", "democracy"), ("fox", "philosophy"), ...
```

**Objective** (binary classification):

$$\log \sigma(v'_{w_O}^T v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-v'_{w_i}^T v_{w_I})]$$

### Implementation with Gensim

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample corpus
sentences = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are animals",
    "the quick brown fox jumps over the lazy dog",
    "animals live in the wild",
    "the cat and dog are friends"
]

# Tokenize
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# Train Word2Vec (Skip-gram)
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,        # Embedding dimension
    window=5,               # Context window size
    min_count=1,            # Minimum word frequency
    sg=1,                   # 1 = Skip-gram, 0 = CBOW
    negative=5,             # Negative sampling
    epochs=100,
    seed=42
)

# Get word vector
cat_vector = model.wv['cat']
print(f"'cat' embedding shape: {cat_vector.shape}")
print(f"'cat' embedding (first 10 dims): {cat_vector[:10]}")

# Find similar words
similar_words = model.wv.most_similar('cat', topn=5)
print(f"\nWords similar to 'cat':")
for word, score in similar_words:
    print(f"  {word}: {score:.3f}")

# Similarity between words
similarity = model.wv.similarity('cat', 'dog')
print(f"\nSimilarity(cat, dog): {similarity:.3f}")
```

### Training on Large Corpus

```python
# Train on larger corpus (e.g., Wikipedia, news articles)

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# LineSentence reads from file (one sentence per line)
# sentences = LineSentence('corpus.txt')

# Optimized training
model = Word2Vec(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=5,            # Ignore rare words
    workers=4,              # Parallel training
    sg=1,                   # Skip-gram
    negative=10,
    epochs=5,
    sample=1e-3,            # Subsampling frequent words
    hs=0,                   # Hierarchical softmax (0 = use negative sampling)
    seed=42
)

# Save model
model.save('word2vec_model.bin')

# Load model
loaded_model = Word2Vec.load('word2vec_model.bin')
```

### Hyperparameter Tuning

```python
# Effect of embedding dimension
for dim in [50, 100, 200, 300]:
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=dim,
        window=5,
        sg=1,
        epochs=100
    )
    print(f"Dim={dim}: Vocab size={len(model.wv)}")

# Effect of window size
for window in [2, 5, 10]:
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=100,
        window=window,
        sg=1,
        epochs=100
    )
    # Larger window = more topical similarity
    # Smaller window = more syntactic similarity
    print(f"Window={window}: Similar to 'cat': {model.wv.most_similar('cat', topn=3)}")
```

## GloVe: Global Vectors

### Motivation

**Word2Vec limitation**: Only uses local context windows, ignores global statistics

**GloVe insight**: Combine local context AND global co-occurrence statistics

```
Word2Vec: Local context
  "the cat sat" → predict nearby words

GloVe: Global co-occurrence matrix
  How often do "cat" and "sat" appear together across ENTIRE corpus?
```

### Co-occurrence Matrix

**Build matrix** $X$ where $X_{ij}$ = number of times word $j$ appears in context of word $i$

```python
import numpy as np
from collections import defaultdict

def build_cooccurrence_matrix(sentences, window_size=2):
    """Build word co-occurrence matrix."""
    # Build vocabulary
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence.split())

    vocab = sorted(list(vocab))
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    # Initialize co-occurrence matrix
    matrix = np.zeros((len(vocab), len(vocab)))

    # Count co-occurrences
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            word_idx = word2idx[word]

            # Context window
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_word_idx = word2idx[words[j]]
                    # Weight by distance
                    distance = abs(i - j)
                    matrix[word_idx, context_word_idx] += 1.0 / distance

    return matrix, vocab, word2idx

# Example
sentences = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog"
]

cooc_matrix, vocab, word2idx = build_cooccurrence_matrix(sentences)

print("Co-occurrence matrix shape:", cooc_matrix.shape)
print("\nVocabulary:", vocab)
print("\nCo-occurrence matrix:")
print(cooc_matrix)

# Interpret: How often does 'cat' appear with other words?
cat_idx = word2idx['cat']
print(f"\n'cat' co-occurrences:")
for word, idx in sorted(word2idx.items(), key=lambda x: cooc_matrix[cat_idx, x[1]], reverse=True):
    if cooc_matrix[cat_idx, idx] > 0:
        print(f"  {word}: {cooc_matrix[cat_idx, idx]:.2f}")
```

### GloVe Objective

**Learn embeddings** $w$ and $\tilde{w}$ (word and context vectors) such that:

$$w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log(X_{ij})$$

**Full objective** (weighted least squares):

$$J = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Where $f(X_{ij})$ is a weighting function:

$$
f(x) = \begin{cases}
(x/x_{max})^\alpha & \text{if } x < x_{max} \\
1 & \text{otherwise}
\end{cases}
$$

**Intuition**:

- Frequent co-occurrences get high weight
- Rare co-occurrences get low weight (but not zero)
- Very frequent words (stop words) don't dominate

### Using Pre-trained GloVe

```python
import numpy as np
import urllib.request
import zipfile

# Download pre-trained GloVe (example: 50d)
# url = 'http://nlp.stanford.edu/data/glove.6B.zip'
# urllib.request.urlretrieve(url, 'glove.6B.zip')

# Load GloVe embeddings
def load_glove_embeddings(filepath):
    """Load GloVe embeddings from file."""
    embeddings = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector

    return embeddings

# Load embeddings (example)
# glove = load_glove_embeddings('glove.6B.50d.txt')

# Example pre-loaded embeddings
glove = {
    'cat': np.random.randn(50),
    'dog': np.random.randn(50),
    'king': np.random.randn(50),
    'queen': np.random.randn(50),
}

print(f"Loaded {len(glove)} word embeddings")
print(f"Embedding dimension: {len(glove['cat'])}")

# Get embedding
word = 'cat'
if word in glove:
    embedding = glove[word]
    print(f"\n'{word}' embedding (first 10 dims): {embedding[:10]}")
```

### GloVe vs Word2Vec

**Comparison**:

| Aspect      | Word2Vec                     | GloVe                              |
| ----------- | ---------------------------- | ---------------------------------- |
| Approach    | Predictive (neural)          | Count-based (matrix factorization) |
| Context     | Local windows                | Global co-occurrence               |
| Training    | Online (stochastic)          | Batch (all at once)                |
| Speed       | Faster on large corpus       | Faster on small corpus             |
| Performance | Slightly better on analogies | Similar overall                    |
| Memory      | Lower                        | Higher (stores matrix)             |

**When to use**:

- **Word2Vec**: Large streaming corpus, online learning
- **GloVe**: Fixed corpus, want to leverage global statistics

## FastText: Subword Embeddings

### Motivation

**Problem with Word2Vec/GloVe**: Each word is atomic unit

```
Word2Vec/GloVe issues:
- Cannot handle out-of-vocabulary (OOV) words
- No shared information between related words
  - "running", "runner", "runs" treated as completely different
- Cannot leverage morphology
  - "unhappiness" = "un" + "happy" + "ness"
```

**FastText solution**: Represent words as bag of character n-grams

### Character N-grams

```python
def get_character_ngrams(word, n=3):
    """Extract character n-grams from word."""
    # Add boundary markers
    word = f'<{word}>'

    # Extract n-grams
    ngrams = []
    for i in range(len(word) - n + 1):
        ngrams.append(word[i:i+n])

    return ngrams

# Example
word = 'cat'
for n in [3, 4, 5]:
    ngrams = get_character_ngrams(word, n)
    print(f"{n}-grams of '{word}': {ngrams}")

# Output:
# 3-grams of 'cat': ['<ca', 'cat', 'at>']
# 4-grams of 'cat': ['<cat', 'cat>']
# 5-grams of 'cat': ['<cat>']
```

### FastText Architecture

**Word representation** = sum of character n-gram embeddings

```
Word "running":
  Character n-grams: <ru, run, unn, nni, nin, ing, ng>

  Embedding("running") = sum of embeddings:
    + embed(<ru)
    + embed(run)
    + embed(unn)
    + embed(nni)
    + embed(nin)
    + embed(ing)
    + embed(ng>)
```

**Benefits**:

1. **OOV handling**: Can embed unseen words using known n-grams
2. **Morphological awareness**: "running" and "runner" share "run"
3. **Typo robustness**: "runnning" (typo) similar to "running"

### Training FastText

```python
from gensim.models import FastText

# Sample corpus
sentences = [
    "the cat is running fast",
    "the dog is running faster",
    "cats run quickly",
    "dogs can run",
    "running is good exercise",
    "the runner finished first"
]

tokenized = [sentence.split() for sentence in sentences]

# Train FastText
model = FastText(
    sentences=tokenized,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,              # Min character n-gram length
    max_n=6,              # Max character n-gram length
    sg=1,                 # Skip-gram
    epochs=100,
    seed=42
)

# Get word vector (even if not in training set!)
print("Words in vocabulary:", list(model.wv.key_to_index.keys()))

# In-vocabulary word
running_vec = model.wv['running']
print(f"\n'running' vector shape: {running_vec.shape}")

# Out-of-vocabulary word (still works!)
# Even though 'sprinting' wasn't in training data
oov_word = 'sprinting'
if oov_word not in model.wv:
    print(f"\n'{oov_word}' NOT in vocabulary, but can still embed:")
    oov_vec = model.wv[oov_word]  # Uses character n-grams!
    print(f"'{oov_word}' vector shape: {oov_vec.shape}")

    # Find similar words
    similar = model.wv.most_similar(oov_word, topn=3)
    print(f"Words similar to '{oov_word}':")
    for word, score in similar:
        print(f"  {word}: {score:.3f}")
```

### Handling Misspellings

```python
# FastText can handle typos!

# Train on correct spellings
sentences = [
    "machine learning is amazing",
    "deep learning uses neural networks",
    "learning new skills is important"
]

tokenized = [s.split() for s in sentences]
model = FastText(sentences=tokenized, vector_size=100, min_count=1)

# Test with misspelling
correct = 'learning'
misspelled = 'lerning'  # Missing 'a'

if misspelled not in model.wv:
    print(f"'{misspelled}' not in vocabulary (typo)")

    # But can still embed it!
    misspelled_vec = model.wv[misspelled]
    correct_vec = model.wv[correct]

    # Compute similarity
    from scipy.spatial.distance import cosine
    similarity = 1 - cosine(misspelled_vec, correct_vec)
    print(f"Similarity('{correct}', '{misspelled}'): {similarity:.3f}")
    # High similarity despite typo!
```

### Multilingual Applications

FastText particularly useful for morphologically rich languages:

```python
# Languages with complex morphology benefit from subword embeddings

# German example:
# "Donaudampfschifffahrtsgesellschaft" (Danube steamship company)
# Can be decomposed: Donau + dampf + schiff + fahrt + gesellschaft

# Finnish example:
# "juoksentelisinkohan" (I wonder if I should run around aimlessly)
# FastText can handle these through character n-grams
```

### Pre-trained FastText Models

```python
import fasttext
import fasttext.util

# Download pre-trained model (example)
# fasttext.util.download_model('en', if_exists='ignore')
# model = fasttext.load_model('cc.en.300.bin')

# Reduce dimensionality
# fasttext.util.reduce_model(model, 100)

# Get word vector
# vec = model.get_word_vector('cat')

# Handle OOV
# oov_vec = model.get_word_vector('unseen_word')
```

## Training Your Own Embeddings

### Data Preparation

```python
import re
from nltk.tokenize import sent_tokenize, word_tokenize

def prepare_corpus(text):
    """Prepare text corpus for embedding training."""
    # Sentence tokenization
    sentences = sent_tokenize(text)

    # Word tokenization and cleaning
    processed_sentences = []

    for sentence in sentences:
        # Lowercase
        sentence = sentence.lower()

        # Remove special characters (keep alphanumeric and spaces)
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)

        # Tokenize
        words = word_tokenize(sentence)

        # Filter short words
        words = [w for w in words if len(w) > 1]

        if words:
            processed_sentences.append(words)

    return processed_sentences

# Example
text = """
Machine learning is a fascinating field. It enables computers to learn from data.
Deep learning, a subset of machine learning, uses neural networks.
Natural language processing applies machine learning to text.
"""

sentences = prepare_corpus(text)
print("Processed sentences:")
for sent in sentences:
    print(f"  {sent}")
```

### Training Pipeline

```python
from gensim.models import Word2Vec
import logging

# Enable logging to see training progress
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_word_embeddings(
    sentences,
    embedding_type='word2vec',
    vector_size=100,
    window=5,
    min_count=2,
    epochs=5
):
    """Train word embeddings with specified parameters."""

    if embedding_type == 'word2vec':
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,  # Skip-gram
            negative=5,
            epochs=epochs,
            seed=42
        )
    elif embedding_type == 'fasttext':
        from gensim.models import FastText
        model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            min_n=3,
            max_n=6,
            workers=4,
            sg=1,
            epochs=epochs,
            seed=42
        )
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    return model

# Train
model = train_word_embeddings(
    sentences=sentences,
    embedding_type='word2vec',
    vector_size=100,
    epochs=10
)

print(f"\nVocabulary size: {len(model.wv)}")
print(f"Embedding dimension: {model.wv.vector_size}")
```

### Incremental Training

```python
# Train initial model
initial_sentences = [
    ['this', 'is', 'first', 'batch'],
    ['initial', 'training', 'data']
]

model = Word2Vec(sentences=initial_sentences, vector_size=100, min_count=1)

# Add more data later (incremental training)
new_sentences = [
    ['new', 'sentences', 'added'],
    ['incremental', 'training', 'update']
]

# Update vocabulary
model.build_vocab(new_sentences, update=True)

# Continue training
model.train(
    new_sentences,
    total_examples=len(new_sentences),
    epochs=5
)

print("Updated vocabulary size:", len(model.wv))
```

## Pre-trained Embeddings

### Loading Pre-trained Models

```python
# GloVe format (text file)
def load_glove_format(filepath, limit=None):
    """Load embeddings from GloVe format file."""
    embeddings = {}
    count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector

            count += 1
            if limit and count >= limit:
                break

    return embeddings

# Word2Vec binary format
from gensim.models import KeyedVectors

# Load Google News Word2Vec
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Load GloVe (converted to word2vec format)
# model = KeyedVectors.load_word2vec_format('glove.840B.300d.txt', binary=False, no_header=True)
```

### Popular Pre-trained Embeddings

**Comparison**:

| Model                   | Dimension | Vocab Size | Training Corpus            | Use Case          |
| ----------------------- | --------- | ---------- | -------------------------- | ----------------- |
| GloVe (Wikipedia)       | 50-300    | 400K       | Wikipedia + Gigaword       | General purpose   |
| GloVe (Twitter)         | 25-200    | 1.2M       | Twitter (2B tweets)        | Social media      |
| Word2Vec (Google News)  | 300       | 3M         | Google News (100B words)   | News, general     |
| FastText (Common Crawl) | 300       | 2M         | Common Crawl (600B tokens) | Multilingual, OOV |

### Using with Gensim

```python
from gensim.downloader import load

# Available models
import gensim.downloader as api
print("Available models:")
for model_name in api.info()['models'].keys():
    if 'word2vec' in model_name or 'glove' in model_name or 'fasttext' in model_name:
        print(f"  - {model_name}")

# Load pre-trained model (example)
# model = load('glove-wiki-gigaword-100')

# Or use sample
model = load('glove-wiki-gigaword-50')  # Smaller for demo

# Use embeddings
vector = model['computer']
print(f"'computer' embedding: {vector[:10]}")

# Find similar words
similar = model.most_similar('computer', topn=5)
print("\nWords similar to 'computer':")
for word, score in similar:
    print(f"  {word}: {score:.3f}")
```

### Converting Between Formats

```python
from gensim.scripts.glove2word2vec import glove2word2vec

# Convert GloVe to Word2Vec format
glove_file = 'glove.6B.100d.txt'
word2vec_file = 'glove.6B.100d.word2vec.txt'

glove2word2vec(glove_file, word2vec_file)

# Load converted file
model = KeyedVectors.load_word2vec_format(word2vec_file)
```

## Embedding Arithmetic and Analogies

### Vector Arithmetic

**Famous example**:
$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

```python
# Using pre-trained embeddings
from gensim.downloader import load

model = load('glove-wiki-gigaword-100')

# Vector arithmetic
result = model.most_similar(
    positive=['woman', 'king'],
    negative=['man'],
    topn=1
)

print(f"king - man + woman = {result[0][0]}")  # Expected: 'queen'

# More examples
analogies = [
    (['paris', 'france'], ['berlin'], 'Germany capital'),
    (['slow', 'slower'], ['fast'], 'comparative'),
    (['good', 'best'], ['bad'], 'superlative'),
]

for positive, negative, description in analogies:
    result = model.most_similar(positive=positive, negative=negative, topn=1)
    print(f"{description}: {result[0][0]}")
```

### Analogy Task

```python
def solve_analogy(model, a, b, c):
    """
    Solve analogy: a is to b as c is to ?

    Example: man is to king as woman is to ?
    """
    try:
        result = model.most_similar(
            positive=[b, c],
            negative=[a],
            topn=1
        )
        return result[0][0], result[0][1]
    except KeyError as e:
        return None, 0.0

# Test analogies
test_cases = [
    ('man', 'king', 'woman', 'queen'),
    ('paris', 'france', 'london', 'england'),
    ('walk', 'walking', 'swim', 'swimming'),
]

print("Analogy tests:")
for a, b, c, expected in test_cases:
    predicted, score = solve_analogy(model, a, b, c)
    correct = "✓" if predicted == expected else "✗"
    print(f"{correct} {a}:{b} :: {c}:{predicted} (expected: {expected}, score: {score:.3f})")
```

### Semantic vs Syntactic Analogies

```python
# Semantic analogies (meaning relationships)
semantic_analogies = [
    ('tokyo', 'japan', 'paris', 'france'),
    ('brother', 'sister', 'uncle', 'aunt'),
    ('big', 'bigger', 'small', 'smaller'),
]

# Syntactic analogies (grammatical relationships)
syntactic_analogies = [
    ('walk', 'walked', 'go', 'went'),
    ('fast', 'fastest', 'slow', 'slowest'),
    ('think', 'thinking', 'read', 'reading'),
]

print("Semantic analogies:")
for a, b, c, expected in semantic_analogies:
    predicted, score = solve_analogy(model, a, b, c)
    print(f"  {a}:{b} :: {c}:{predicted} (score: {score:.3f})")

print("\nSyntactic analogies:")
for a, b, c, expected in syntactic_analogies:
    predicted, score = solve_analogy(model, a, b, c)
    print(f"  {a}:{b} :: {c}:{predicted} (score: {score:.3f})")
```

### Clustering Words

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get embeddings for words
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'cat', 'dog', 'bird', 'fish',
         'car', 'truck', 'bike', 'plane']

vectors = [model[word] for word in words]

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(vectors)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
    x, y = coords[i]
    cluster = clusters[i]
    plt.scatter(x, y, c=f'C{cluster}', s=100)
    plt.annotate(word, (x, y), fontsize=12)

plt.title('Word Embedding Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```

## Evaluation Methods

### Intrinsic Evaluation

**1. Word Similarity**

```python
from scipy.stats import spearmanr

# Human similarity ratings (example subset of WordSim-353)
word_pairs = [
    ('computer', 'keyboard', 7.62),
    ('computer', 'internet', 7.58),
    ('computer', 'software', 8.50),
    ('phone', 'computer', 6.35),
    ('cat', 'dog', 7.52),
    ('cat', 'computer', 1.42),
]

# Get model similarities
model_similarities = []
human_similarities = []

for word1, word2, human_score in word_pairs:
    try:
        model_score = model.similarity(word1, word2)
        model_similarities.append(model_score)
        human_similarities.append(human_score)
    except KeyError:
        continue

# Compute correlation
correlation, p_value = spearmanr(human_similarities, model_similarities)

print(f"Spearman correlation: {correlation:.3f}")
print(f"P-value: {p_value:.4f}")
```

**2. Analogy Accuracy**

```python
def evaluate_analogies(model, analogy_file):
    """Evaluate model on analogy task."""
    correct = 0
    total = 0

    # Load analogies (format: word1 word2 word3 word4)
    # where word1:word2 :: word3:word4

    with open(analogy_file, 'r') as f:
        for line in f:
            words = line.strip().split()
            if len(words) != 4:
                continue

            a, b, c, expected = words

            try:
                predicted, _ = solve_analogy(model, a, b, c)
                if predicted == expected:
                    correct += 1
                total += 1
            except KeyError:
                continue

    accuracy = correct / total if total > 0 else 0
    return accuracy

# Example (conceptual)
# accuracy = evaluate_analogies(model, 'analogies.txt')
# print(f"Analogy accuracy: {accuracy:.2%}")
```

**3. Outlier Detection**

```python
def detect_outlier(model, words):
    """Find the word that doesn't belong."""
    # Compute pairwise similarities
    n = len(words)
    similarities = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    similarities[i, j] = model.similarity(words[i], words[j])
                except KeyError:
                    similarities[i, j] = 0

    # Word with lowest average similarity is outlier
    avg_similarities = similarities.mean(axis=1)
    outlier_idx = np.argmin(avg_similarities)

    return words[outlier_idx]

# Test
word_groups = [
    ['cat', 'dog', 'bird', 'car'],  # 'car' is outlier
    ['king', 'queen', 'prince', 'apple'],  # 'apple' is outlier
    ['run', 'walk', 'jump', 'banana'],  # 'banana' is outlier
]

for words in word_groups:
    outlier = detect_outlier(model, words)
    print(f"Outlier in {words}: {outlier}")
```

### Extrinsic Evaluation

**Use embeddings as features in downstream tasks**:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Example: Sentiment classification with embeddings

def document_vector(tokens, model):
    """Average word embeddings to get document vector."""
    vectors = []
    for token in tokens:
        if token in model:
            vectors.append(model[token])

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Sample data
texts = [
    "I love this movie it's amazing",
    "terrible movie waste of time",
    "great acting excellent plot",
    "boring and predictable",
    "fantastic film highly recommend"
]
labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative

# Convert to embeddings
X = np.array([document_vector(text.split(), model) for text in texts])
y = np.array(labels)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Sentiment classification accuracy: {accuracy:.2%}")
```

## Fine-tuning Embeddings

### Domain Adaptation

```python
# Start with pre-trained embeddings
general_model = load('glove-wiki-gigaword-100')

# Fine-tune on domain-specific corpus
domain_sentences = [
    ['medical', 'diagnosis', 'requires', 'careful', 'examination'],
    ['patient', 'shows', 'symptoms', 'of', 'disease'],
    # ... more medical text
]

# Continue training
from gensim.models import Word2Vec

# Initialize with pre-trained vectors
domain_model = Word2Vec(
    vector_size=100,
    min_count=1,
    window=5
)

domain_model.build_vocab(domain_sentences)

# Copy weights from pre-trained model for common words
for word in domain_model.wv.index_to_key:
    if word in general_model:
        domain_model.wv[word] = general_model[word]

# Continue training on domain data
domain_model.train(
    domain_sentences,
    total_examples=len(domain_sentences),
    epochs=10
)

print("Domain-adapted model ready")
```

### Task-Specific Fine-tuning

```python
# Fine-tune embeddings as part of neural network training
import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
    """Classifier with trainable embedding layer."""

    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with pre-trained embeddings
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # Classifier layers
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        pooled = torch.mean(embedded, dim=1)  # Average pooling
        x = torch.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model with pre-trained embeddings
# model = SentimentClassifier(
#     vocab_size=10000,
#     embedding_dim=100,
#     pretrained_embeddings=pretrained_matrix
# )

# Optionally freeze embeddings (don't fine-tune)
# model.embedding.weight.requires_grad = False

# Or allow fine-tuning (default)
# model.embedding.weight.requires_grad = True
```

## Limitations and Challenges

### 1. Context Independence

**Problem**: Same embedding regardless of context

```python
# "bank" has same embedding in both sentences:
sentence1 = "I went to the river bank"  # river bank
sentence2 = "I went to the money bank"  # financial bank

# Word2Vec gives same vector for both!
bank_vector = model['bank']  # Same for both contexts
```

**Solution**: Contextual embeddings (ELMo, BERT) - see next guide

### 2. Out-of-Vocabulary Words

**Problem** (Word2Vec/GloVe): Cannot embed unseen words

```python
# Word not in training vocabulary
try:
    vector = model['asdfghjkl']  # Random word
except KeyError:
    print("Word not in vocabulary!")

# Solution: Use FastText (subword embeddings)
```

### 3. Bias in Embeddings

Embeddings inherit biases from training data:

```python
# Gender bias example
def check_bias(model):
    """Check for gender bias in embeddings."""

    # Occupations
    occupations = ['doctor', 'nurse', 'engineer', 'teacher']

    for occupation in occupations:
        try:
            # Similarity to gender words
            sim_man = model.similarity(occupation, 'man')
            sim_woman = model.similarity(occupation, 'woman')

            bias = sim_man - sim_woman
            gender_lean = "male" if bias > 0 else "female"

            print(f"{occupation:12} bias toward {gender_lean}: {abs(bias):.3f}")
        except KeyError:
            continue

# check_bias(model)

# Common biases found:
# - doctor → male
# - nurse → female
# - engineer → male
# - teacher → female
```

**Debiasing techniques**:

```python
# Neutralize gender component
def debias_gender(model, word):
    """Remove gender component from word embedding."""
    # Get gender direction (man - woman)
    gender_direction = model['man'] - model['woman']
    gender_direction = gender_direction / np.linalg.norm(gender_direction)

    # Project word onto gender-neutral subspace
    word_vec = model[word]
    projection = np.dot(word_vec, gender_direction) * gender_direction
    debiased_vec = word_vec - projection

    return debiased_vec
```

### 4. No Sentence/Document Representation

Word embeddings are for individual words only:

```python
# No direct way to embed sentences
sentence = "machine learning is amazing"

# Naive approach: average word vectors
def average_embeddings(sentence, model):
    """Average word embeddings (simple but lossy)."""
    words = sentence.split()
    vectors = [model[w] for w in words if w in model]

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Better solutions: Sentence embeddings (see next guide)
```

### 5. Fixed Vocabulary

Cannot adapt to new words without retraining:

```python
# New slang, technical terms require retraining
# "COVID-19", "cryptocurrency" weren't in older models

# Solution options:
# 1. Retrain/fine-tune on new corpus
# 2. Use FastText (subword approach)
# 3. Use contextual embeddings (BERT, GPT)
```

### 6. Computational Cost

Training on large corpora is expensive:

```python
# Training costs scale with:
# - Vocabulary size (V)
# - Corpus size (C)
# - Embedding dimension (D)
# - Training epochs (E)

# Approximate training time:
# Word2Vec: O(C × D)
# GloVe: O(V^2) for co-occurrence matrix + training
# FastText: Higher than Word2Vec (character n-grams)

# Mitigation:
# - Use pre-trained models when possible
# - Start with smaller dimensions (50-100)
# - Use negative sampling
# - Subsample frequent words
```

## Summary

**Key Concepts**:

1. **Word embeddings** map words to dense continuous vectors that capture semantic meaning
2. **Distributional hypothesis**: Words in similar contexts have similar meanings
3. **Word2Vec** learns embeddings by predicting context (Skip-gram) or center word (CBOW)
4. **GloVe** combines local context and global co-occurrence statistics
5. **FastText** uses character n-grams for subword information and OOV handling
6. **Vector arithmetic** enables analogies: king - man + woman ≈ queen
7. **Evaluation**: Intrinsic (similarity, analogies) and extrinsic (downstream tasks)

**Model Comparison**:

| Model    | Approach             | OOV Handling | Best For                            |
| -------- | -------------------- | ------------ | ----------------------------------- |
| Word2Vec | Predictive (neural)  | No           | Large corpora, general purpose      |
| GloVe    | Count-based (matrix) | No           | Leveraging global statistics        |
| FastText | Subword n-grams      | Yes          | Morphologically rich languages, OOV |

**Architecture Comparison**:

| Aspect      | Skip-gram            | CBOW                     |
| ----------- | -------------------- | ------------------------ |
| Input       | Center word          | Context words            |
| Output      | Context words        | Center word              |
| Performance | Better on rare words | Better on frequent words |
| Speed       | Slower               | Faster                   |

**Training Pipeline**:

1. Prepare corpus (tokenize, clean)
2. Choose model (Word2Vec/GloVe/FastText)
3. Set hyperparameters (dimension, window, min_count)
4. Train or load pre-trained
5. Evaluate (similarity, analogies)
6. Fine-tune for specific task (optional)

**Best Practices**:

- Start with pre-trained embeddings (GloVe, Word2Vec)
- Use 100-300 dimensions for most tasks
- Window size 5-10 for semantic similarity
- Normalize embeddings (unit length)
- Use FastText for OOV robustness
- Evaluate on multiple tasks
- Be aware of biases in embeddings

**Limitations**:

- Context-independent (same vector regardless of usage)
- Cannot handle polysemy well (multiple meanings)
- Biases from training data
- No direct sentence/document embeddings
- Fixed vocabulary (except FastText)

**When to Use**:

- ✅ Feature extraction for ML models
- ✅ Semantic similarity tasks
- ✅ Transfer learning from pre-trained models
- ✅ Dimensionality reduction from sparse vectors
- ❌ When context matters (use contextual embeddings)
- ❌ When need sentence-level semantics (use sentence embeddings)

## Next Steps

- Explore [Sentence Embeddings](sentence-embeddings.md) to learn about encoding entire sentences and paragraphs
- Study [Contextual Embeddings](contextual-embeddings.md) to understand context-dependent representations (ELMo, BERT)
- Learn about [Embedding Spaces](embedding-spaces.md) to understand geometric properties and operations
- Apply embeddings in [Text Classification](../classical_nlp/classical-classification.md) tasks
- Use embeddings for [Retrieval](../retrieval_augmented_generation/retrieval-methods.md) in RAG systems
- Study [Transformers](../language_models/encoder-decoder-models.md) for modern context-aware architectures
