# Embedding Spaces

## Table of Contents

- [Introduction](#introduction)
- [Geometric Properties](#geometric-properties)
- [Vector Operations](#vector-operations)
- [Similarity Metrics](#similarity-metrics)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Visualization Techniques](#visualization-techniques)
- [Subspaces and Manifolds](#subspaces-and-manifolds)
- [Embedding Alignment](#embedding-alignment)
- [Quality Metrics](#quality-metrics)
- [Debugging Embeddings](#debugging-embeddings)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Embedding spaces** are high-dimensional vector spaces where semantic relationships are encoded as geometric relationships. Understanding the geometry and properties of these spaces is crucial for working with embeddings effectively.

```
Conceptual view of embedding space:

3D projection (actual spaces are 100-1000+ dimensions):

         king·
              ╲
               ╲  "royalty"
                ╲
         queen· ─────────────────·man
                                  │
                                  │ "gender"
                                  │
                                ·woman

Relationships encoded as:
- Distance: Semantic similarity
- Direction: Semantic relationships
- Clusters: Semantic categories
```

**Key concepts**:

1. **Distributional hypothesis**: "You shall know a word by the company it keeps"
2. **Geometric encoding**: Meaning → spatial relationships
3. **Linear structure**: Relationships as vector arithmetic
4. **Isotropy**: How uniformly distributed embeddings are

This guide explores the mathematical and geometric properties of embedding spaces.

## Geometric Properties

### Distance and Similarity

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt

# Example word embeddings (simplified to 3D for visualization)
embeddings_3d = {
    'king': np.array([0.5, 0.8, 0.3]),
    'queen': np.array([0.5, 0.75, -0.2]),
    'man': np.array([0.1, 0.2, 0.3]),
    'woman': np.array([0.1, 0.15, -0.2]),
    'apple': np.array([-0.8, 0.1, 0.4]),
}

# Compute pairwise distances
words = list(embeddings_3d.keys())
vectors = np.array([embeddings_3d[w] for w in words])

# Cosine similarity
cos_sim = cosine_similarity(vectors)

# Euclidean distance
eucl_dist = euclidean_distances(vectors)

print("Cosine Similarity Matrix:\n")
print(f"{'':10}", end='')
for word in words:
    print(f"{word:10}", end='')
print()

for i, word_i in enumerate(words):
    print(f"{word_i:10}", end='')
    for j in range(len(words)):
        print(f"{cos_sim[i][j]:10.3f}", end='')
    print()

print("\n\nEuclidean Distance Matrix:\n")
print(f"{'':10}", end='')
for word in words:
    print(f"{word:10}", end='')
print()

for i, word_i in enumerate(words):
    print(f"{word_i:10}", end='')
    for j in range(len(words)):
        print(f"{eucl_dist[i][j]:10.3f}", end='')
    print()
```

### Norm and Magnitude

```python
def analyze_vector_norms(embeddings_dict):
    """Analyze L2 norms of embeddings."""
    norms = {}

    for word, vector in embeddings_dict.items():
        norms[word] = np.linalg.norm(vector)

    print("Vector L2 Norms:\n")
    for word, norm in sorted(norms.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(norm * 20)
        print(f"{word:10} {bar} {norm:.3f}")

    return norms

norms = analyze_vector_norms(embeddings_3d)

print("\nNote: Many embedding models normalize vectors to unit length")
print("This makes cosine similarity = dot product")
```

### Isotropy

**Isotropy**: How uniformly distributed embeddings are in space

```python
def measure_isotropy(embeddings_matrix):
    """
    Measure isotropy of embedding space.

    High isotropy: Embeddings spread uniformly in all directions
    Low isotropy: Embeddings clustered in certain directions (anisotropic)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    normalized = embeddings_matrix / (norms + 1e-8)

    # Compute average cosine similarity
    cos_sim = cosine_similarity(normalized)

    # Exclude diagonal (self-similarity)
    n = len(cos_sim)
    mask = ~np.eye(n, dtype=bool)
    avg_similarity = cos_sim[mask].mean()

    print(f"Average pairwise cosine similarity: {avg_similarity:.3f}")
    print(f"Isotropy interpretation:")
    if avg_similarity < 0.1:
        print("  ✓ High isotropy (embeddings well-distributed)")
    elif avg_similarity < 0.3:
        print("  ~ Moderate isotropy")
    else:
        print("  ✗ Low isotropy (embeddings clustered/anisotropic)")

    return avg_similarity

# Test
measure_isotropy(vectors)

# Visualize distribution
print("\nIsotropy visualization:")
print("""
High isotropy:           Low isotropy:
    ·   ·   ·               ·····
  ·   ·   ·   ·            ·····
    ·   ·   ·                ····
  ·   ·   ·   ·              ···
    ·   ·   ·
(well-spread)           (clustered)
""")
```

### Principal Components

```python
from sklearn.decomposition import PCA

def analyze_principal_components(embeddings_matrix, n_components=None):
    """Analyze principal components of embedding space."""
    if n_components is None:
        n_components = min(embeddings_matrix.shape)

    pca = PCA(n_components=n_components)
    pca.fit(embeddings_matrix)

    print("Principal Component Analysis:\n")
    print("Explained variance ratio:")

    cumulative_variance = 0
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        cumulative_variance += var_ratio
        bar = '█' * int(var_ratio * 50)
        print(f"  PC{i+1}: {bar} {var_ratio:.3f} (cumulative: {cumulative_variance:.3f})")

    return pca

pca = analyze_principal_components(vectors)
```

## Vector Operations

### Vector Arithmetic

```python
from gensim.downloader import load

# Load real word embeddings
word_model = load('glove-wiki-gigaword-100')

def vector_arithmetic(word_model, positive, negative=None):
    """
    Perform vector arithmetic: positive - negative

    Example: king - man + woman ≈ queen
    """
    if negative is None:
        negative = []

    result = word_model.most_similar(positive=positive, negative=negative, topn=5)

    print(f"Vector arithmetic:")
    print(f"  Positive: {positive}")
    print(f"  Negative: {negative}")
    print(f"\n  Results:")
    for word, score in result:
        print(f"    {word:15} (similarity: {score:.3f})")

    return result

# Classic examples
print("=" * 50)
vector_arithmetic(word_model, positive=['king', 'woman'], negative=['man'])

print("\n" + "=" * 50)
vector_arithmetic(word_model, positive=['paris', 'germany'], negative=['france'])

print("\n" + "=" * 50)
vector_arithmetic(word_model, positive=['walking', 'swim'], negative=['walk'])
```

### Analogies

```python
def solve_analogy(word_model, a, b, c):
    """
    Solve analogy: a is to b as c is to ?

    Method: vec(b) - vec(a) + vec(c) ≈ vec(?)
    """
    print(f"\nAnalogy: '{a}' is to '{b}' as '{c}' is to ___?")

    # Compute: b - a + c
    result = word_model.most_similar(
        positive=[b, c],
        negative=[a],
        topn=5
    )

    print(f"\n  Top predictions:")
    for word, score in result:
        print(f"    {word:15} (score: {score:.3f})")

    return result

# Test analogies
analogies = [
    ('man', 'king', 'woman'),        # woman:queen
    ('france', 'paris', 'germany'),  # germany:berlin
    ('good', 'better', 'bad'),       # bad:worse
    ('walk', 'walked', 'go'),        # go:went
]

for a, b, c in analogies:
    solve_analogy(word_model, a, b, c)
    print("=" * 50)
```

### Compositional Operations

```python
def compose_embeddings(word_model, words, method='average'):
    """
    Compose multiple word embeddings into phrase embedding.

    Methods:
    - average: Simple mean
    - weighted: Weighted by IDF or learned weights
    - concatenate: Concat vectors (increases dimensionality)
    """
    vectors = [word_model[word] for word in words if word in word_model]

    if not vectors:
        return None

    if method == 'average':
        composed = np.mean(vectors, axis=0)
    elif method == 'sum':
        composed = np.sum(vectors, axis=0)
    elif method == 'concatenate':
        composed = np.concatenate(vectors)
    else:
        raise ValueError(f"Unknown method: {method}")

    return composed

# Example: Compose phrase embedding
phrase = "machine learning"
words = phrase.split()

phrase_embedding = compose_embeddings(word_model, words, method='average')

print(f"Phrase: '{phrase}'")
print(f"Composed embedding shape: {phrase_embedding.shape}")

# Find similar phrases
if phrase_embedding is not None:
    # Normalize
    phrase_embedding = phrase_embedding / np.linalg.norm(phrase_embedding)

    # Find similar words
    similar = word_model.similar_by_vector(phrase_embedding, topn=10)

    print(f"\nWords similar to '{phrase}':")
    for word, score in similar:
        print(f"  {word:20} {score:.3f}")
```

### Projection and Rejection

```python
def project_onto_direction(vector, direction):
    """Project vector onto direction."""
    direction = direction / np.linalg.norm(direction)
    projection = np.dot(vector, direction) * direction
    return projection

def reject_from_direction(vector, direction):
    """Remove component in direction from vector."""
    projection = project_onto_direction(vector, direction)
    rejection = vector - projection
    return rejection

# Example: Remove gender component
if 'man' in word_model and 'woman' in word_model:
    # Define gender direction
    gender_direction = word_model['man'] - word_model['woman']

    # Project a word
    word = 'king'
    word_vec = word_model[word]

    # Project onto gender axis
    gender_component = project_onto_direction(word_vec, gender_direction)

    # Remove gender component
    debiased = reject_from_direction(word_vec, gender_direction)

    print(f"Original '{word}' vector norm: {np.linalg.norm(word_vec):.3f}")
    print(f"Gender component norm: {np.linalg.norm(gender_component):.3f}")
    print(f"Debiased vector norm: {np.linalg.norm(debiased):.3f}")

    # Find words similar to debiased vector
    similar_debiased = word_model.similar_by_vector(debiased, topn=5)
    print(f"\nWords similar to debiased '{word}':")
    for w, score in similar_debiased:
        print(f"  {w:15} {score:.3f}")
```

## Similarity Metrics

### Cosine Similarity

```python
def cosine_similarity_manual(vec1, vec2):
    """Compute cosine similarity manually."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    cosine_sim = dot_product / (norm1 * norm2 + 1e-8)

    return cosine_sim

# Properties of cosine similarity
print("Cosine Similarity Properties:\n")
print("Range: [-1, 1]")
print("  +1: Vectors point in same direction (identical)")
print("   0: Vectors are orthogonal (unrelated)")
print("  -1: Vectors point in opposite directions (antonyms)")
print("\nAdvantages:")
print("  • Normalized (magnitude-invariant)")
print("  • Efficient to compute")
print("  • Works well for high-dimensional spaces")
print("\nDisadvantages:")
print("  • Ignores magnitude information")
print("  • May not capture all semantic relations")
```

### Other Distance Metrics

```python
from scipy.spatial.distance import euclidean, cityblock, chebyshev

def compare_distance_metrics(vec1, vec2):
    """Compare different distance metrics."""

    metrics = {
        'Cosine similarity': 1 - cosine_similarity([vec1], [vec2])[0][0],
        'Euclidean (L2)': euclidean(vec1, vec2),
        'Manhattan (L1)': cityblock(vec1, vec2),
        'Chebyshev (L∞)': chebyshev(vec1, vec2),
    }

    print("Distance Metrics Comparison:\n")
    for name, distance in metrics.items():
        print(f"  {name:25} {distance:.3f}")

    return metrics

# Example
if 'cat' in word_model and 'dog' in word_model:
    vec_cat = word_model['cat']
    vec_dog = word_model['dog']

    print("Comparing 'cat' vs 'dog':")
    compare_distance_metrics(vec_cat, vec_dog)

    print("\nComparing 'cat' vs 'computer':")
    if 'computer' in word_model:
        vec_computer = word_model['computer']
        compare_distance_metrics(vec_cat, vec_computer)
```

### Nearest Neighbors

```python
from sklearn.neighbors import NearestNeighbors

def find_nearest_neighbors(query_word, word_model, n_neighbors=5, metric='cosine'):
    """Find nearest neighbors using sklearn."""

    if query_word not in word_model:
        print(f"Word '{query_word}' not in vocabulary")
        return

    # Get all vectors
    vocab = list(word_model.index_to_key[:10000])  # Use subset for efficiency
    vectors = np.array([word_model[word] for word in vocab])

    # Fit NN model
    nn_model = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric)
    nn_model.fit(vectors)

    # Query
    query_vec = word_model[query_word].reshape(1, -1)
    distances, indices = nn_model.kneighbors(query_vec)

    print(f"Nearest neighbors of '{query_word}' (metric={metric}):\n")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if vocab[idx] != query_word:  # Skip self
            print(f"  {i}. {vocab[idx]:20} (distance: {dist:.3f})")

# Example
find_nearest_neighbors('python', word_model, n_neighbors=10)
```

## Dimensionality Reduction

### PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def reduce_with_pca(word_model, words, n_components=2):
    """Reduce embeddings to n dimensions using PCA."""
    # Get embeddings
    vectors = np.array([word_model[word] for word in words if word in word_model])
    valid_words = [word for word in words if word in word_model]

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(vectors)

    print(f"PCA Reduction: {vectors.shape[1]}D → {n_components}D")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    return reduced, valid_words, pca

# Example words
words_to_plot = [
    'king', 'queen', 'man', 'woman',
    'paris', 'france', 'london', 'england',
    'python', 'java', 'programming', 'code',
    'dog', 'cat', 'animal', 'pet'
]

reduced, valid_words, pca = reduce_with_pca(word_model, words_to_plot, n_components=2)

# Visualize
plt.figure(figsize=(12, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], s=100, alpha=0.6)

for i, word in enumerate(valid_words):
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=12)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Word Embeddings Reduced to 2D (PCA)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### t-SNE

```python
from sklearn.manifold import TSNE

def reduce_with_tsne(word_model, words, n_components=2, perplexity=30):
    """Reduce embeddings using t-SNE."""
    # Get embeddings
    vectors = np.array([word_model[word] for word in words if word in word_model])
    valid_words = [word for word in words if word in word_model]

    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000
    )
    reduced = tsne.fit_transform(vectors)

    print(f"t-SNE Reduction: {vectors.shape[1]}D → {n_components}D")
    print(f"Perplexity: {perplexity}")

    return reduced, valid_words

# Get more words for better t-SNE visualization
categories = {
    'animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'monkey'],
    'countries': ['france', 'germany', 'italy', 'spain', 'china', 'japan'],
    'programming': ['python', 'java', 'code', 'programming', 'software', 'computer'],
    'food': ['pizza', 'burger', 'salad', 'pasta', 'rice', 'bread']
}

all_words = [word for words in categories.values() for word in words]
reduced_tsne, valid_words = reduce_with_tsne(word_model, all_words, perplexity=15)

# Visualize with colors
plt.figure(figsize=(12, 8))

colors = ['red', 'blue', 'green', 'orange']
for i, (category, words) in enumerate(categories.items()):
    # Find indices of words in this category
    indices = [valid_words.index(w) for w in words if w in valid_words]
    if indices:
        category_points = reduced_tsne[indices]
        plt.scatter(
            category_points[:, 0],
            category_points[:, 1],
            label=category,
            s=100,
            alpha=0.6,
            c=colors[i]
        )

for i, word in enumerate(valid_words):
    plt.annotate(word, (reduced_tsne[i, 0], reduced_tsne[i, 1]), fontsize=10)

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Word Embeddings Reduced to 2D (t-SNE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### UMAP

```python
# UMAP is often better than t-SNE for embeddings

# Conceptual example (requires umap-learn package)
# from umap import UMAP

def reduce_with_umap(word_model, words, n_components=2, n_neighbors=15):
    """Reduce embeddings using UMAP."""
    # Get embeddings
    vectors = np.array([word_model[word] for word in words if word in word_model])
    valid_words = [word for word in words if word in word_model]

    # Apply UMAP
    # umap_model = UMAP(
    #     n_components=n_components,
    #     n_neighbors=n_neighbors,
    #     random_state=42
    # )
    # reduced = umap_model.fit_transform(vectors)

    # print(f"UMAP Reduction: {vectors.shape[1]}D → {n_components}D")

    # return reduced, valid_words

    print("UMAP example (requires: pip install umap-learn)")
    print("Advantages over t-SNE:")
    print("  • Faster for large datasets")
    print("  • Preserves global structure better")
    print("  • Can project new points")
    print("  • Better for clustering visualization")

reduce_with_umap(word_model, all_words)
```

### Comparison of Methods

```python
reduction_methods = {
    'PCA': {
        'type': 'Linear',
        'preserves': 'Global structure, variance',
        'speed': 'Very fast',
        'use_when': 'Need interpretable components, quick overview'
    },
    't-SNE': {
        'type': 'Non-linear',
        'preserves': 'Local structure, clusters',
        'speed': 'Slow',
        'use_when': 'Visualizing clusters, understanding local structure'
    },
    'UMAP': {
        'type': 'Non-linear',
        'preserves': 'Local + global structure',
        'speed': 'Medium',
        'use_when': 'Best of both worlds, production use'
    }
}

print("Dimensionality Reduction Methods:\n")
for method, info in reduction_methods.items():
    print(f"{method}:")
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

## Visualization Techniques

### 2D Scatter Plot

```python
def visualize_embeddings_2d(word_model, words, method='pca'):
    """Comprehensive 2D visualization."""
    # Reduce dimensionality
    if method == 'pca':
        reduced, valid_words, _ = reduce_with_pca(word_model, words, n_components=2)
    elif method == 'tsne':
        reduced, valid_words = reduce_with_tsne(word_model, words)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Scatter points
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=100,
        alpha=0.6,
        c=range(len(valid_words)),
        cmap='viridis'
    )

    # Annotate
    for i, word in enumerate(valid_words):
        ax.annotate(
            word,
            (reduced[i, 0], reduced[i, 1]),
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

    # Styling
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(f'Word Embeddings Visualization ({method.upper()})', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Example
# visualize_embeddings_2d(word_model, words_to_plot, method='pca')
```

### Interactive Visualization

````python
def create_interactive_plot(word_model, words):
    """Create interactive plot (conceptual - requires plotly)."""
    # Reduce dimensions
    reduced, valid_words, _ = reduce_with_pca(word_model, words, n_components=3)

    print("Interactive 3D visualization (requires plotly):")
    print("```python")
    print("import plotly.graph_objects as go")
    print("")
    print("fig = go.Figure(data=[go.Scatter3d(")
    print("    x=reduced[:, 0],")
    print("    y=reduced[:, 1],")
    print("    z=reduced[:, 2],")
    print("    mode='markers+text',")
    print("    text=valid_words,")
    print("    textposition='top center',")
    print("    marker=dict(size=8, color=range(len(valid_words)), colorscale='Viridis')")
    print(")])")
    print("fig.show()")
    print("```")

create_interactive_plot(word_model, words_to_plot)
````

### Heatmap Visualization

```python
def visualize_similarity_heatmap(word_model, words):
    """Visualize pairwise similarity as heatmap."""
    # Get vectors
    vectors = np.array([word_model[word] for word in words if word in word_model])
    valid_words = [word for word in words if word in word_model]

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(vectors)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')

    # Labels
    plt.xticks(range(len(valid_words)), valid_words, rotation=45, ha='right')
    plt.yticks(range(len(valid_words)), valid_words)

    # Add values
    for i in range(len(valid_words)):
        for j in range(len(valid_words)):
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=8)

    plt.title('Word Similarity Heatmap')
    plt.tight_layout()
    plt.show()

# Example
similarity_words = ['king', 'queen', 'man', 'woman', 'cat', 'dog']
# visualize_similarity_heatmap(word_model, similarity_words)
```

## Subspaces and Manifolds

### Semantic Subspaces

```python
def identify_subspace(word_model, words, n_components=3):
    """
    Identify semantic subspace spanned by words.

    Example: Gender subspace from 'man', 'woman', 'he', 'she', etc.
    """
    # Get vectors
    vectors = np.array([word_model[word] for word in words if word in word_model])

    # PCA to find principal directions
    pca = PCA(n_components=n_components)
    pca.fit(vectors)

    print(f"Subspace defined by: {words}")
    print(f"Explained variance by top {n_components} components:")

    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  Component {i+1}: {var:.3f}")

    # Principal directions
    directions = pca.components_

    return directions

# Example: Gender subspace
gender_words = ['man', 'woman', 'he', 'she', 'male', 'female', 'boy', 'girl']
if all(w in word_model for w in gender_words):
    gender_directions = identify_subspace(word_model, gender_words, n_components=2)

    print("\nGender subspace identified!")
    print("Can be used to:")
    print("  • Measure gender bias in other words")
    print("  • Debias embeddings")
    print("  • Analyze gender associations")
```

### Manifold Structure

```python
def analyze_manifold_structure(word_model, words):
    """Analyze whether embeddings lie on a manifold."""
    vectors = np.array([word_model[word] for word in words if word in word_model])

    # Intrinsic dimensionality estimation using PCA
    pca = PCA()
    pca.fit(vectors)

    # Find elbow in explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    # Estimate intrinsic dimension (where cumsum reaches 90%)
    intrinsic_dim = np.argmax(cumsum >= 0.9) + 1

    print("Manifold Analysis:")
    print(f"  Embedding dimension: {vectors.shape[1]}")
    print(f"  Intrinsic dimension (90% var): {intrinsic_dim}")
    print(f"  Dimensionality ratio: {intrinsic_dim / vectors.shape[1]:.2f}")

    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    plt.axvline(x=intrinsic_dim, color='g', linestyle='--', label=f'Intrinsic dim: {intrinsic_dim}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Intrinsic Dimensionality of Embedding Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return intrinsic_dim

# Test with many words
many_words = list(word_model.index_to_key[:1000])
# intrinsic_dim = analyze_manifold_structure(word_model, many_words)
```

### Clustering in Embedding Space

```python
from sklearn.cluster import KMeans

def cluster_embeddings(word_model, words, n_clusters=4):
    """Cluster words based on embeddings."""
    # Get vectors
    vectors = np.array([word_model[word] for word in words if word in word_model])
    valid_words = [word for word in words if word in word_model]

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)

    # Group by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for word, label in zip(valid_words, labels):
        clusters[label].append(word)

    print(f"Clustered {len(valid_words)} words into {n_clusters} clusters:\n")
    for cluster_id, words_in_cluster in clusters.items():
        print(f"Cluster {cluster_id}: {', '.join(words_in_cluster)}")

    return clusters, kmeans

# Example
diverse_words = [
    'dog', 'cat', 'lion', 'tiger',  # animals
    'paris', 'london', 'berlin', 'rome',  # cities
    'python', 'java', 'ruby', 'javascript',  # languages
    'car', 'bike', 'bus', 'train'  # vehicles
]

clusters, kmeans = cluster_embeddings(word_model, diverse_words, n_clusters=4)
```

## Embedding Alignment

### Cross-lingual Alignment

```python
def align_embedding_spaces(source_vectors, target_vectors):
    """
    Align two embedding spaces using Procrustes alignment.

    Used for:
    - Cross-lingual embeddings
    - Temporal alignment (different time periods)
    - Domain adaptation
    """
    # Procrustes alignment: Find rotation matrix W such that
    # source * W ≈ target

    from scipy.linalg import orthogonal_procrustes

    # Compute optimal rotation
    W, _ = orthogonal_procrustes(source_vectors, target_vectors)

    # Align source to target space
    aligned_source = source_vectors @ W

    print("Procrustes Alignment:")
    print(f"  Source shape: {source_vectors.shape}")
    print(f"  Target shape: {target_vectors.shape}")
    print(f"  Rotation matrix shape: {W.shape}")

    # Measure alignment quality
    mse = np.mean((aligned_source - target_vectors) ** 2)
    print(f"  Alignment MSE: {mse:.4f}")

    return aligned_source, W

# Conceptual example
print("\nCross-lingual Alignment Example:")
print("  English embeddings ──┐")
print("                       ├─→ Find rotation W")
print("  French embeddings  ──┘")
print("\n  After alignment:")
print("  'cat' (English) ≈ 'chat' (French) in shared space")
```

### Temporal Alignment

```python
def temporal_alignment_example():
    """Example of aligning embeddings from different time periods."""

    print("Temporal Embedding Alignment:")
    print("\nUse case: Track word meaning changes over time")
    print("\n1. Train embeddings on historical corpus (e.g., 1900s)")
    print("2. Train embeddings on modern corpus (e.g., 2000s)")
    print("3. Align spaces using anchor words (stable meanings)")
    print("4. Compare embeddings for same word across time")
    print("\nExample findings:")
    print("  • 'gay' (1900s) ≈ 'cheerful', 'happy'")
    print("  • 'gay' (2000s) ≈ 'homosexual', 'lgbtq'")
    print("\n  • 'mouse' (1900s) ≈ 'rodent', 'animal'")
    print("  • 'mouse' (2000s) ≈ 'computer', 'device', 'rodent'")

temporal_alignment_example()
```

## Quality Metrics

### Intrinsic Evaluation

```python
def intrinsic_evaluation(word_model):
    """Evaluate embeddings using intrinsic benchmarks."""

    # Word similarity benchmarks
    print("Intrinsic Evaluation Benchmarks:\n")

    benchmarks = {
        'WordSim-353': 'Word pair similarity ratings',
        'SimLex-999': 'Genuine similarity (not relatedness)',
        'MEN': 'Semantic relatedness',
        'RG-65': 'Classic similarity benchmark',
        'RareWord': 'Rare word similarity'
    }

    for name, description in benchmarks.items():
        print(f"{name}: {description}")

    # Analogy benchmarks
    print("\nAnalogy Benchmarks:")

    analogy_sets = {
        'Google Analogy': 'Semantic and syntactic analogies',
        'BATS': 'Balanced analogy test set',
        'SemEval-2012': 'Semantic relation classification'
    }

    for name, description in analogy_sets.items():
        print(f"  {name}: {description}")

intrinsic_evaluation(word_model)
```

### Extrinsic Evaluation

```python
def extrinsic_evaluation():
    """Evaluate embeddings on downstream tasks."""

    print("Extrinsic Evaluation (Downstream Tasks):\n")

    tasks = {
        'Text Classification': 'Sentiment analysis, topic classification',
        'Named Entity Recognition': 'Identify entities in text',
        'POS Tagging': 'Part-of-speech tagging',
        'Dependency Parsing': 'Syntactic structure',
        'Machine Translation': 'Translation quality',
        'Question Answering': 'Answer accuracy'
    }

    for task, description in tasks.items():
        print(f"{task}: {description}")

    print("\nEvaluation approach:")
    print("  1. Use embeddings as input features")
    print("  2. Train task-specific model")
    print("  3. Measure task performance")
    print("  4. Compare with baseline embeddings")

extrinsic_evaluation()
```

### Coverage and OOV Analysis

```python
def analyze_coverage(word_model, text_corpus):
    """Analyze vocabulary coverage."""
    # Tokenize corpus
    words = text_corpus.lower().split()
    unique_words = set(words)

    # Check coverage
    in_vocab = sum(1 for word in unique_words if word in word_model)
    oov_words = [word for word in unique_words if word not in word_model]

    coverage = in_vocab / len(unique_words)

    print("Vocabulary Coverage Analysis:")
    print(f"  Unique words in corpus: {len(unique_words)}")
    print(f"  Words in vocabulary: {in_vocab}")
    print(f"  Out-of-vocabulary (OOV): {len(oov_words)}")
    print(f"  Coverage: {coverage:.2%}")

    if oov_words[:10]:
        print(f"\n  Sample OOV words: {', '.join(oov_words[:10])}")

    return coverage, oov_words

# Example
sample_corpus = "python programming machine learning deep learning neural networks"
coverage, oov = analyze_coverage(word_model, sample_corpus)
```

## Debugging Embeddings

### Sanity Checks

```python
def sanity_check_embeddings(word_model):
    """Perform sanity checks on embeddings."""

    print("Embedding Sanity Checks:\n")

    # 1. Basic similarity checks
    tests = [
        ('Synonyms', 'good', 'great', 'should be similar'),
        ('Antonyms', 'good', 'bad', 'may be similar (both evaluative)'),
        ('Unrelated', 'cat', 'mathematics', 'should be dissimilar'),
    ]

    for test_name, word1, word2, expectation in tests:
        if word1 in word_model and word2 in word_model:
            sim = word_model.similarity(word1, word2)
            print(f"{test_name}: '{word1}' vs '{word2}'")
            print(f"  Similarity: {sim:.3f}")
            print(f"  Expected: {expectation}")
            print()

    # 2. Analogy check
    print("Analogy check: man:woman :: king:?")
    if all(w in word_model for w in ['man', 'woman', 'king']):
        result = word_model.most_similar(
            positive=['woman', 'king'],
            negative=['man'],
            topn=1
        )
        print(f"  Answer: {result[0][0]} (score: {result[0][1]:.3f})")
        print(f"  Expected: queen")

    # 3. Nearest neighbors check
    print("\nNearest neighbors of 'python':")
    if 'python' in word_model:
        neighbors = word_model.most_similar('python', topn=5)
        for word, score in neighbors:
            print(f"  {word} ({score:.3f})")

sanity_check_embeddings(word_model)
```

### Detecting Biases

```python
def detect_bias(word_model, bias_type='gender'):
    """Detect biases in embeddings."""

    if bias_type == 'gender':
        # Define gender axis
        if 'man' in word_model and 'woman' in word_model:
            gender_direction = word_model['man'] - word_model['woman']
            gender_direction = gender_direction / np.linalg.norm(gender_direction)

            # Test profession words
            professions = [
                'doctor', 'nurse', 'engineer', 'teacher',
                'scientist', 'secretary', 'ceo', 'programmer'
            ]

            print("Gender Bias Analysis:\n")
            print(f"{'Profession':<15} {'Gender Score':<15} {'Interpretation'}")
            print("-" * 50)

            for profession in professions:
                if profession in word_model:
                    prof_vec = word_model[profession]
                    # Project onto gender axis
                    score = np.dot(prof_vec, gender_direction)

                    if score > 0.1:
                        interpretation = "Male-leaning"
                    elif score < -0.1:
                        interpretation = "Female-leaning"
                    else:
                        interpretation = "Neutral"

                    print(f"{profession:<15} {score:>+.3f}           {interpretation}")

    print("\nNote: Biases reflect corpus, not reality!")
    print("Solutions:")
    print("  • Debiasing techniques")
    print("  • Balanced training data")
    print("  • Post-processing corrections")

detect_bias(word_model, 'gender')
```

### Visualization for Debugging

```python
def debug_visualization(word_model, problematic_words):
    """Create visualizations to debug embedding issues."""

    # Get embeddings
    vectors = np.array([
        word_model[word] for word in problematic_words
        if word in word_model
    ])
    valid_words = [word for word in problematic_words if word in word_model]

    # Reduce to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=100, alpha=0.6)

    for i, word in enumerate(valid_words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=12)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Debug Visualization: Problematic Words')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Compute pairwise similarities
    print("\nPairwise Similarities:")
    similarities = cosine_similarity(vectors)

    for i, word_i in enumerate(valid_words):
        for j, word_j in enumerate(valid_words):
            if i < j:
                print(f"  {word_i} <-> {word_j}: {similarities[i][j]:.3f}")

# Example: Debug unexpected similarities
# problematic = ['bank', 'river', 'money', 'deposit', 'financial']
# debug_visualization(word_model, problematic)
```

## Summary

**Key Concepts**:

1. **Embedding spaces** encode semantic meaning as geometric relationships
2. **Geometric properties** include distance, direction, isotropy, and manifold structure
3. **Vector operations** enable analogy solving and compositional meaning
4. **Similarity metrics** measure semantic relatedness (cosine, Euclidean, etc.)
5. **Dimensionality reduction** enables visualization (PCA, t-SNE, UMAP)
6. **Subspaces** capture semantic dimensions (gender, sentiment, topic)
7. **Alignment** enables cross-lingual and temporal comparison
8. **Quality metrics** evaluate embedding utility (intrinsic and extrinsic)

**Important Properties**:

| Property                 | Description             | Measurement               |
| ------------------------ | ----------------------- | ------------------------- |
| Isotropy                 | Uniform distribution    | Avg pairwise similarity   |
| Intrinsic dimensionality | Effective dimensions    | PCA cumulative variance   |
| Coverage                 | Vocabulary completeness | % words in vocab          |
| Linearity                | Linear relationships    | Analogy accuracy          |
| Bias                     | Unwanted associations   | Projection onto bias axis |

**Visualization Methods**:

- **PCA**: Fast, linear, interpretable components
- **t-SNE**: Preserves local structure, good for clusters
- **UMAP**: Best of both, preserves local + global structure
- **Heatmaps**: Show pairwise similarities
- **3D plots**: Interactive exploration

**Best Practices**:

1. **Normalize vectors** before computing cosine similarity
2. **Check isotropy** to detect degenerate spaces
3. **Visualize** to understand structure and debug issues
4. **Test analogies** for sanity checks
5. **Measure coverage** for your specific domain
6. **Detect biases** and apply debiasing if needed
7. **Use multiple metrics** for comprehensive evaluation
8. **Reduce dimensions** carefully (preserve important structure)

**Common Issues**:

- **Anisotropic spaces**: Embeddings clustered in few directions
- **Bias**: Unwanted associations from training data
- **OOV words**: Missing vocabulary for domain-specific terms
- **Frequency effects**: Common words dominate space
- **Context independence**: Static embeddings ignore context

## Next Steps

- Apply to [Language Modeling](../language_models/language-modeling-basics.md) to understand how embeddings are trained
- Learn [Transfer Learning](../llm_concepts/transfer-learning.md) to adapt embeddings to new domains
- Explore [Bias Mitigation](../rlhf_and_alignment/bias-and-fairness.md) techniques for fairer embeddings
- Study [Dense Retrieval](../retrieval_augmented_generation/dense-retrieval.md) using embedding spaces
- Implement [Semantic Search](../application_patterns/semantic-search.md) with optimized embeddings
- Progress to [Evaluation Metrics](../evaluation/evaluation-metrics.md) for comprehensive assessment
