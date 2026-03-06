# Language Modeling Fundamentals

## Table of Contents

- [Introduction](#introduction)
- [What is a Language Model?](#what-is-a-language-model)
- [The Language Modeling Task](#the-language-modeling-task)
- [N-gram Language Models](#n-gram-language-models)
- [Neural Language Models](#neural-language-models)
- [Autoregressive vs Masked Language Modeling](#autoregressive-vs-masked-language-modeling)
- [Perplexity and Evaluation](#perplexity-and-evaluation)
- [Why Language Modeling Matters](#why-language-modeling-matters)
- [From Statistical to Neural to Transformers](#from-statistical-to-neural-to-transformers)
- [Applications of Language Models](#applications-of-language-models)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Language modeling** is the foundational task in modern NLP. At its core, a language model learns the probability distribution over sequences of words (or tokens). This seemingly simple objective has become the basis for nearly all state-of-the-art NLP systems.

```
The fundamental question:

Given a sequence of words: "The cat sat on the"
What comes next?

A good language model assigns:
  P("mat" | "The cat sat on the") = 0.15  ← High probability
  P("sky" | "The cat sat on the") = 0.001 ← Low probability
  P("xqz" | "The cat sat on the") = 0     ← Zero probability
```

**Why it matters**:

- **Unsupervised learning**: No labeled data needed, just raw text
- **General knowledge**: Models learn grammar, facts, reasoning
- **Transfer learning**: Pretrain once, fine-tune for many tasks
- **Generation**: Enable text completion, translation, summarization
- **Understanding**: Contextual representations for classification, QA

This guide covers the fundamentals of language modeling from n-grams to modern neural approaches.

## What is a Language Model?

### Formal Definition

A **language model** assigns probabilities to sequences of words:

$$P(w_1, w_2, ..., w_n)$$

Where $w_i$ are words/tokens in a vocabulary $V$.

Using the chain rule of probability:

$$P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1, w_2) \cdot ... \cdot P(w_n|w_1, ..., w_{n-1})$$

**Goal**: Model the conditional probability $P(w_t | w_1, ..., w_{t-1})$

### Example Calculation

```python
import numpy as np

# Example: Compute probability of sentence
sentence = "the cat sat"

# Simple probability calculation (conceptual)
def sentence_probability(words):
    """
    Compute P(sentence) using chain rule.
    P(the cat sat) = P(the) × P(cat|the) × P(sat|the cat)
    """
    # Mock probabilities (in reality, from trained model)
    probabilities = {
        'P(the)': 0.1,
        'P(cat|the)': 0.05,
        'P(sat|the cat)': 0.08
    }

    total_prob = probabilities['P(the)'] * \
                 probabilities['P(cat|the)'] * \
                 probabilities['P(sat|the cat)']

    return total_prob

prob = sentence_probability(['the', 'cat', 'sat'])
print(f"P('the cat sat') = {prob}")
print(f"In log space: log P = {np.log(prob):.4f}")

# Why log space?
print("\nWhy use log probabilities:")
print("  • Avoid numerical underflow (multiplying many small numbers)")
print("  • Convert products to sums: log(a×b) = log(a) + log(b)")
print("  • More numerically stable")
```

### Types of Language Models

```python
# Three main paradigms:

lm_types = {
    'Statistical (N-gram)': {
        'era': '1980s-2010s',
        'method': 'Count-based, Markov assumption',
        'pros': 'Simple, interpretable, fast',
        'cons': 'Fixed context, sparsity, no generalization'
    },
    'Neural (RNN/LSTM)': {
        'era': '2010-2017',
        'method': 'Recurrent neural networks',
        'pros': 'Variable context, dense representations',
        'cons': 'Sequential (slow), vanishing gradients'
    },
    'Transformer': {
        'era': '2017-present',
        'method': 'Self-attention mechanisms',
        'pros': 'Parallel, long-range dependencies, scalable',
        'cons': 'Computationally expensive, quadratic attention'
    }
}

for lm_type, info in lm_types.items():
    print(f"\n{lm_type} ({info['era']}):")
    print(f"  Method: {info['method']}")
    print(f"  Pros: {info['pros']}")
    print(f"  Cons: {info['cons']}")
```

## The Language Modeling Task

### Next Token Prediction

**Core objective**: Given a context, predict the next token

```
Training example:

Input:  "The cat sat on the"
Target: "mat"

Model learns: P(mat | The cat sat on the)
```

### Training Process

```python
import torch
import torch.nn as nn

def language_modeling_training_loop(model, data, epochs=10):
    """
    Conceptual language model training loop.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch in data:
            # batch: [batch_size, seq_len]
            # Each sequence: [w1, w2, w3, ..., wn]

            inputs = batch[:, :-1]   # All tokens except last: [w1, ..., w(n-1)]
            targets = batch[:, 1:]   # All tokens except first: [w2, ..., wn]

            # Forward pass
            logits = model(inputs)  # [batch_size, seq_len-1, vocab_size]

            # Compute loss (cross-entropy)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),  # Flatten
                targets.reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

print("Language Model Training Process:")
print("1. Input: Sequence of tokens [w1, w2, ..., wn]")
print("2. Model predicts: P(w2|w1), P(w3|w1,w2), ..., P(wn|w1,...,w(n-1))")
print("3. Loss: Cross-entropy between predictions and actual next tokens")
print("4. Optimize: Update model parameters to maximize log-likelihood")
```

### Self-Supervised Learning

```python
def self_supervised_language_modeling():
    """
    Language modeling is self-supervised: labels come from the data itself.
    """

    # Raw text (unlabeled)
    text = "The quick brown fox jumps over the lazy dog"

    # Create training examples automatically
    tokens = text.split()

    training_examples = []
    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]
        training_examples.append((context, target))

    print("Self-Supervised Training Examples:\n")
    for i, (context, target) in enumerate(training_examples[:5], 1):
        print(f"{i}. Context: {' '.join(context):30} → Target: {target}")

    print("\nKey insight:")
    print("  • No manual labeling required!")
    print("  • Labels come from the text itself")
    print("  • Can use massive amounts of unlabeled text")
    print("  • Scales to billions/trillions of tokens")

self_supervised_language_modeling()
```

## N-gram Language Models

### Markov Assumption

**N-gram models** use the Markov assumption: only last $n-1$ words matter

$$P(w_t | w_1, ..., w_{t-1}) \approx P(w_t | w_{t-n+1}, ..., w_{t-1})$$

```python
from collections import defaultdict, Counter

class NgramLanguageModel:
    """Simple n-gram language model."""

    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)

    def train(self, corpus):
        """Train on corpus (list of sentences)."""
        for sentence in corpus:
            # Add start/end tokens
            tokens = ['<START>'] * (self.n - 1) + sentence + ['<END>']

            # Count n-grams
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                word = tokens[i+self.n-1]

                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

    def probability(self, context, word):
        """Compute P(word | context)."""
        context = tuple(context[-(self.n-1):])  # Use last n-1 words

        if self.context_counts[context] == 0:
            return 0

        return self.ngram_counts[context][word] / self.context_counts[context]

    def generate(self, start_words, max_length=20):
        """Generate text using the language model."""
        words = list(start_words)

        for _ in range(max_length):
            context = tuple(words[-(self.n-1):])

            # Get possible next words
            if context in self.ngram_counts:
                next_words = self.ngram_counts[context]
                # Sample based on probabilities
                total = sum(next_words.values())
                rand = np.random.randint(0, total)

                cumsum = 0
                for word, count in next_words.items():
                    cumsum += count
                    if cumsum > rand:
                        if word == '<END>':
                            return words
                        words.append(word)
                        break
            else:
                break

        return words

# Train on simple corpus
corpus = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['the', 'dog', 'sat', 'on', 'the', 'rug'],
    ['the', 'cat', 'chased', 'the', 'mouse'],
    ['the', 'dog', 'chased', 'the', 'cat']
]

# Bigram model (n=2)
bigram_lm = NgramLanguageModel(n=2)
bigram_lm.train(corpus)

print("Bigram Language Model:\n")

# Test probabilities
test_contexts = [
    (['the'], 'cat'),
    (['the'], 'dog'),
    (['cat'], 'sat'),
    (['cat'], 'chased'),
]

for context, word in test_contexts:
    prob = bigram_lm.probability(context, word)
    print(f"P({word} | {' '.join(context)}) = {prob:.3f}")

# Generate text
print("\nGenerated text:")
generated = bigram_lm.generate(['<START>'], max_length=10)
print(' '.join(generated))
```

### Smoothing Techniques

```python
def add_k_smoothing(count, context_count, vocab_size, k=1):
    """
    Add-k smoothing (Laplace smoothing when k=1).

    Handles zero counts by adding k to all counts.
    """
    return (count + k) / (context_count + k * vocab_size)

def interpolation_smoothing(trigram_prob, bigram_prob, unigram_prob,
                           lambda1=0.6, lambda2=0.3, lambda3=0.1):
    """
    Linear interpolation: combine different n-gram orders.

    P(w|context) = λ1·P_trigram + λ2·P_bigram + λ3·P_unigram
    """
    return lambda1 * trigram_prob + lambda2 * bigram_prob + lambda3 * unigram_prob

# Example
print("Smoothing Techniques:\n")

# Without smoothing
print("Without smoothing:")
print("  P('xqz' | 'the cat') = 0 (never seen)")
print("  Problem: Zero probability kills entire sequence")

# With add-1 smoothing
count = 0  # Never seen
context_count = 100
vocab_size = 10000
smoothed = add_k_smoothing(count, context_count, vocab_size, k=1)
print(f"\nWith add-1 smoothing:")
print(f"  P('xqz' | 'the cat') = {smoothed:.6f} (small but non-zero)")

# Interpolation
print("\nWith interpolation:")
print("  Trigram: P('sat' | 'the cat') = 0.0 (never seen)")
print("  Bigram:  P('sat' | 'cat') = 0.3")
print("  Unigram: P('sat') = 0.05")
result = interpolation_smoothing(0.0, 0.3, 0.05)
print(f"  Combined: {result:.3f}")
```

### Limitations of N-grams

```python
ngram_limitations = {
    'Fixed context window': {
        'problem': 'Only use last n-1 words',
        'example': "Trigram can't connect 'The cat' with 'sat' if 5 words apart"
    },
    'Sparsity': {
        'problem': 'Most n-grams never observed in training',
        'example': 'Need smoothing, but hard to estimate rare events'
    },
    'No generalization': {
        'problem': 'Cannot generalize to similar contexts',
        'example': "'cat sat' and 'dog sat' are independent, no shared knowledge"
    },
    'Storage': {
        'problem': 'Need to store all n-gram counts',
        'example': 'Billions of n-grams for large corpora'
    },
    'No semantic understanding': {
        'problem': 'Purely statistical, no meaning',
        'example': "'bank' near 'river' vs 'bank' near 'money' treated identically"
    }
}

print("N-gram Language Model Limitations:\n")
for limitation, info in ngram_limitations.items():
    print(f"{limitation}:")
    print(f"  Problem: {info['problem']}")
    print(f"  Example: {info['example']}")
    print()
```

## Neural Language Models

### From Sparse to Dense

```python
# Evolution from n-grams to neural models

print("Evolution of Language Modeling:\n")

print("N-gram Model:")
print("  Representation: Discrete word IDs")
print("  Context: Fixed window (n-1 words)")
print("  Similarity: No notion of word similarity")
print("  Example: 'cat' and 'dog' are completely different\n")

print("Neural Language Model:")
print("  Representation: Dense word embeddings")
print("  Context: Variable length (RNN/Transformer)")
print("  Similarity: Similar words have similar embeddings")
print("  Example: 'cat' and 'dog' are close in embedding space")
```

### Feed-Forward Neural LM

```python
import torch
import torch.nn as nn

class FeedForwardNeuralLM(nn.Module):
    """
    Simple feed-forward neural language model.
    Bengio et al., 2003
    """

    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=200, context_size=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Hidden layers
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        self.relu = nn.ReLU()

    def forward(self, context):
        """
        context: [batch_size, context_size] - indices of context words
        returns: [batch_size, vocab_size] - logits for next word
        """
        # Embed context words
        embeds = self.embeddings(context)  # [batch_size, context_size, embedding_dim]

        # Flatten embeddings
        embeds_flat = embeds.view(embeds.size(0), -1)  # [batch_size, context_size * embedding_dim]

        # Hidden layer
        hidden = self.relu(self.fc1(embeds_flat))

        # Output layer
        logits = self.fc2(hidden)

        return logits

# Example usage
vocab_size = 10000
model = FeedForwardNeuralLM(vocab_size, embedding_dim=100, hidden_dim=200, context_size=5)

# Example input: indices of 5 context words
context = torch.tensor([[1, 42, 156, 8, 99]])  # [batch_size=1, context_size=5]

# Forward pass
logits = model(context)

print("Feed-Forward Neural Language Model:")
print(f"  Input shape: {context.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Output: Probability distribution over {vocab_size} words")

# Get probabilities
probs = torch.softmax(logits, dim=-1)
top5_probs, top5_indices = torch.topk(probs[0], k=5)

print(f"\nTop 5 predicted next words:")
for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
    print(f"  {i+1}. Word index {idx.item()}: {prob.item():.4f}")
```

### Recurrent Neural LM

```python
class RecurrentNeuralLM(nn.Module):
    """
    RNN-based language model.
    Can handle variable-length context!
    """

    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: [batch_size, seq_len] - sequence of token indices
        returns: [batch_size, seq_len, vocab_size] - logits for each position
        """
        # Embed tokens
        embeds = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # RNN processes sequence
        rnn_out, hidden = self.rnn(embeds, hidden)  # [batch_size, seq_len, hidden_dim]

        # Project to vocabulary
        logits = self.fc(rnn_out)  # [batch_size, seq_len, vocab_size]

        return logits, hidden

# Example
rnn_lm = RecurrentNeuralLM(vocab_size=10000, embedding_dim=100, hidden_dim=256)

# Variable-length sequences!
seq1 = torch.tensor([[1, 2, 3, 4, 5]])  # Length 5
seq2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # Length 10

logits1, _ = rnn_lm(seq1)
logits2, _ = rnn_lm(seq2)

print("\nRecurrent Neural Language Model:")
print(f"  Sequence 1 length: {seq1.size(1)} → Output: {logits1.shape}")
print(f"  Sequence 2 length: {seq2.size(1)} → Output: {logits2.shape}")
print(f"  Advantage: Variable context length!")
```

### Advantages of Neural LMs

```python
neural_advantages = {
    'Dense representations': {
        'benefit': 'Words represented as continuous vectors',
        'impact': 'Similar words share similar representations'
    },
    'Generalization': {
        'benefit': 'Can generalize to unseen contexts',
        'impact': 'Better handling of rare events'
    },
    'Variable context': {
        'benefit': 'RNNs/Transformers handle variable-length context',
        'impact': 'No fixed n-gram window limitation'
    },
    'Shared parameters': {
        'benefit': 'Same weights for all positions',
        'impact': 'More efficient than storing all n-grams'
    },
    'Compositionality': {
        'benefit': 'Learn hierarchical representations',
        'impact': 'Capture complex linguistic patterns'
    }
}

print("Advantages of Neural Language Models:\n")
for advantage, info in neural_advantages.items():
    print(f"{advantage}:")
    print(f"  • {info['benefit']}")
    print(f"  • {info['impact']}")
    print()
```

## Autoregressive vs Masked Language Modeling

### Autoregressive (Causal) LM

**Left-to-right**: Predict next token given all previous tokens

```
Architecture visualization:

Input:    The  cat  sat  on   the
          ↓    ↓    ↓    ↓    ↓
Predict:  cat  sat  on   the  mat
          ↑    ↑    ↑    ↑    ↑

Each token can only attend to previous tokens (causal masking)

Attention pattern (✓ = can attend):
     The  cat  sat  on  the
The   ✓    ✗    ✗   ✗   ✗
cat   ✓    ✓    ✗   ✗   ✗
sat   ✓    ✓    ✓   ✗   ✗
on    ✓    ✓    ✓   ✓   ✗
the   ✓    ✓    ✓   ✓   ✓
```

```python
def autoregressive_lm_example():
    """Autoregressive language modeling."""

    sentence = "The cat sat on the mat"
    tokens = sentence.split()

    print("Autoregressive Language Modeling:\n")
    print("Training objectives:\n")

    for i in range(len(tokens)):
        context = tokens[:i+1]
        if i < len(tokens) - 1:
            target = tokens[i+1]
            print(f"  Context: {' '.join(context):25} → Predict: {target}")

    print("\nKey properties:")
    print("  ✓ Natural for generation (predict next token)")
    print("  ✓ Matches inference (can generate incrementally)")
    print("  ✗ Only sees left context (unidirectional)")

    print("\nExamples: GPT, GPT-2, GPT-3, LLaMA")

autoregressive_lm_example()
```

### Masked Language Modeling (MLM)

**Bidirectional**: Predict masked tokens using both left and right context

```
Architecture visualization:

Input:    The  [MASK]  sat   on  the  mat
          ↓      ↓      ↓    ↓    ↓    ↓
Predict:        cat
                 ↑

Token can attend to all other tokens (bidirectional)

Attention pattern (✓ = can attend):
         The  [M]  sat  on  the  mat
The       ✓    ✓    ✓   ✓   ✓    ✓
[MASK]    ✓    ✓    ✓   ✓   ✓    ✓   ← Can see both sides!
sat       ✓    ✓    ✓   ✓   ✓    ✓
on        ✓    ✓    ✓   ✓   ✓    ✓
the       ✓    ✓    ✓   ✓   ✓    ✓
mat       ✓    ✓    ✓   ✓   ✓    ✓
```

```python
def masked_lm_example():
    """Masked language modeling."""

    sentence = "The cat sat on the mat"
    tokens = sentence.split()

    print("\nMasked Language Modeling:\n")
    print("Training objectives:\n")

    # Randomly mask tokens
    import random
    for i in range(len(tokens)):
        masked_tokens = tokens.copy()
        original_token = masked_tokens[i]
        masked_tokens[i] = '[MASK]'

        left_context = ' '.join(masked_tokens[:i])
        right_context = ' '.join(masked_tokens[i+1:])

        print(f"  {left_context:15} [MASK] {right_context:15} → Predict: {original_token}")

    print("\nKey properties:")
    print("  ✓ Bidirectional context (sees both left and right)")
    print("  ✓ Better for understanding tasks")
    print("  ✗ Mismatch between training (masked) and inference (no masks)")

    print("\nExamples: BERT, RoBERTa, ALBERT")

masked_lm_example()
```

### Comparison

```python
comparison = {
    'Training': {
        'Autoregressive': 'Predict next token',
        'Masked': 'Predict masked tokens'
    },
    'Context': {
        'Autoregressive': 'Unidirectional (left-to-right)',
        'Masked': 'Bidirectional (both directions)'
    },
    'Generation': {
        'Autoregressive': 'Natural (predict next token)',
        'Masked': 'Difficult (needs iterative refinement)'
    },
    'Understanding': {
        'Autoregressive': 'Good (but one direction)',
        'Masked': 'Better (full context)'
    },
    'Use cases': {
        'Autoregressive': 'Text generation, completion, chatbots',
        'Masked': 'Classification, NER, question answering'
    }
}

print("\nAutoregressive vs Masked Language Modeling:\n")
print(f"{'Aspect':<15} {'Autoregressive':<35} {'Masked':<35}")
print("=" * 90)
for aspect, values in comparison.items():
    print(f"{aspect:<15} {values['Autoregressive']:<35} {values['Masked']:<35}")
```

## Perplexity and Evaluation

### Perplexity Definition

**Perplexity**: Measure of how well a language model predicts a test set

$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_1, ..., w_{i-1})\right)$$

Interpretation: "How many words is the model uncertain about, on average?"

```python
def compute_perplexity(log_likelihoods):
    """
    Compute perplexity from log-likelihoods.

    Args:
        log_likelihoods: List of log P(w_i | context)

    Returns:
        Perplexity value
    """
    n = len(log_likelihoods)
    avg_log_likelihood = sum(log_likelihoods) / n
    perplexity = np.exp(-avg_log_likelihood)

    return perplexity

# Example
print("Perplexity Calculation:\n")

# Good model (high probabilities)
good_probs = [0.8, 0.7, 0.6, 0.75, 0.8]
good_log_probs = [np.log(p) for p in good_probs]
good_ppl = compute_perplexity(good_log_probs)

print("Good model:")
print(f"  Probabilities: {good_probs}")
print(f"  Perplexity: {good_ppl:.2f}")

# Bad model (low probabilities)
bad_probs = [0.1, 0.05, 0.08, 0.12, 0.09]
bad_log_probs = [np.log(p) for p in bad_probs]
bad_ppl = compute_perplexity(bad_log_probs)

print("\nBad model:")
print(f"  Probabilities: {bad_probs}")
print(f"  Perplexity: {bad_ppl:.2f}")

print("\nInterpretation:")
print(f"  Lower perplexity = Better model")
print(f"  Perplexity ≈ 'effective vocabulary size' the model is choosing from")
```

### Cross-Entropy Loss

```python
def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss for language modeling.
    Equivalent to negative log-likelihood.
    """
    # Softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get probability of correct class
    correct_probs = probs[range(len(targets)), targets]

    # Negative log-likelihood
    nll = -torch.log(correct_probs)

    # Average over batch
    loss = nll.mean()

    return loss

# Relationship to perplexity
print("\nCross-Entropy Loss and Perplexity:\n")
print("Loss = -log P(w | context)")
print("Perplexity = exp(Loss)")
print("\nExample:")

example_loss = 2.5
example_ppl = np.exp(example_loss)

print(f"  Cross-entropy loss: {example_loss}")
print(f"  Perplexity: {example_ppl:.2f}")
```

### Evaluation Best Practices

```python
evaluation_practices = {
    'Hold-out test set': {
        'why': 'Avoid overfitting to evaluation data',
        'how': 'Use separate test set that model never sees during training'
    },
    'Domain matching': {
        'why': 'Perplexity varies by domain',
        'how': 'Evaluate on same domain as target application'
    },
    'Token-level metrics': {
        'why': 'Fair comparison across models',
        'how': 'Report per-token perplexity, not per-sequence'
    },
    'Multiple metrics': {
        'why': 'Perplexity doesn\'t capture everything',
        'how': 'Also use downstream task performance'
    }
}

print("\nEvaluation Best Practices:\n")
for practice, info in evaluation_practices.items():
    print(f"{practice}:")
    print(f"  Why: {info['why']}")
    print(f"  How: {info['how']}")
    print()
```

## Why Language Modeling Matters

### Unsupervised Pretraining

```python
print("Why Language Modeling is Powerful:\n")

print("1. Unsupervised Learning:")
print("   • No labeled data required")
print("   • Can use massive amounts of text from the internet")
print("   • Scales to trillions of tokens")
print()

print("2. General Knowledge Acquisition:")
print("   • Grammar: Subject-verb agreement, syntax")
print("   • Facts: 'Paris is the capital of France'")
print("   • Reasoning: Causality, temporal relations")
print("   • World knowledge: Common sense, domain expertise")
print()

print("3. Transfer Learning:")
print("   • Pretrain once on language modeling")
print("   • Fine-tune for specific tasks")
print("   • Saves computation and data for downstream tasks")
print()

print("4. Contextual Representations:")
print("   • Words represented in context")
print("   • Captures polysemy ('bank' as financial vs riverbank)")
print("   • Enables downstream tasks without task-specific architectures")
```

### What Models Learn

```python
def what_language_models_learn():
    """Examples of knowledge captured by language models."""

    knowledge_types = {
        'Syntax': [
            "The dog [is/are] → 'is' (subject-verb agreement)",
            "She gave [he/him] → 'him' (case marking)"
        ],
        'Semantics': [
            "The cat sat on the [mat/sky] → 'mat' (plausibility)",
            "Ice is [hot/cold] → 'cold' (world knowledge)"
        ],
        'World Knowledge': [
            "Paris is the capital of [France/Germany] → 'France'",
            "Python is a programming [language/animal] → 'language'"
        ],
        'Coreference': [
            "The cat ate. It was [hungry/delicious] → 'hungry' (refers to cat)",
            "John told Mike that he [likes/liked] → verb tense agreement"
        ],
        'Common Sense': [
            "She was happy so she [smiled/cried] → 'smiled'",
            "The glass fell off the table and [broke/bounced] → 'broke'"
        ]
    }

    print("What Language Models Learn:\n")
    for knowledge_type, examples in knowledge_types.items():
        print(f"{knowledge_type}:")
        for example in examples:
            print(f"  • {example}")
        print()

what_language_models_learn()
```

### The Pretraining-Finetuning Paradigm

```
The modern NLP workflow:

┌─────────────────────────────────────┐
│   PRETRAINING (Language Modeling)   │
│                                     │
│   Massive unlabeled corpus          │
│   (e.g., 1 trillion tokens)         │
│   Learn general language knowledge  │
│                                     │
│   Result: Pretrained model          │
└─────────────────┬───────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │    FINE-TUNING      │
        │                     │
        │  Task-specific data │
        │  (e.g., 10K labels) │
        │  Adapt to task      │
        │                     │
        │  Result: Task model │
        └─────────────────────┘

Benefits:
  • Pretrain once, reuse for many tasks
  • Need less labeled data for downstream tasks
  • Better performance than training from scratch
  • Faster development cycle
```

## From Statistical to Neural to Transformers

### Historical Timeline

```python
timeline = {
    '1980s-1990s': {
        'approach': 'N-gram models',
        'key_work': 'Statistical language modeling',
        'limitation': 'Sparsity, fixed context'
    },
    '2003': {
        'approach': 'Neural language models',
        'key_work': 'Bengio et al. - Feed-forward neural LM',
        'innovation': 'Distributed word representations'
    },
    '2010-2013': {
        'approach': 'Recurrent models',
        'key_work': 'RNN/LSTM language models',
        'innovation': 'Variable-length context'
    },
    '2013': {
        'approach': 'Word embeddings',
        'key_work': 'Word2Vec, GloVe',
        'innovation': 'Efficient pre-trained embeddings'
    },
    '2017': {
        'approach': 'Transformer',
        'key_work': 'Attention is All You Need',
        'innovation': 'Self-attention, parallelizable'
    },
    '2018': {
        'approach': 'Large pretrained models',
        'key_work': 'BERT (masked LM), GPT (autoregressive)',
        'innovation': 'Transfer learning revolution'
    },
    '2019-2020': {
        'approach': 'Scaling up',
        'key_work': 'GPT-2, GPT-3, T5',
        'innovation': 'Billions of parameters, few-shot learning'
    },
    '2022-present': {
        'approach': 'Chat-optimized LLMs',
        'key_work': 'ChatGPT, GPT-4, Claude, LLaMA',
        'innovation': 'RLHF, instruction following'
    }
}

print("Evolution of Language Modeling:\n")
for year, info in timeline.items():
    print(f"{year}: {info['approach']}")
    print(f"  Key Work: {info['key_work']}")
    print(f"  Innovation: {info.get('innovation', info.get('limitation'))}")
    print()
```

### Scaling Laws

```python
def discuss_scaling_laws():
    """Neural language models benefit from scale."""

    print("Scaling Laws for Language Models:\n")

    print("Observations:")
    print("  • Larger models → Lower perplexity")
    print("  • More training data → Better performance")
    print("  • Larger models need more data to avoid overfitting")
    print("  • Compute-optimal scaling: balance model size and data")
    print()

    print("Example progression:")

    models = [
        ('GPT-1', '117M parameters', '2018'),
        ('GPT-2', '1.5B parameters', '2019'),
        ('GPT-3', '175B parameters', '2020'),
        ('GPT-4', '1.7T parameters (estimated)', '2023'),
    ]

    for model, size, year in models:
        print(f"  {model} ({year}): {size}")

    print("\nKey insight: Performance scales predictably with:")
    print("  1. Model parameters (N)")
    print("  2. Dataset size (D)")
    print("  3. Training compute (C)")
    print("\n  Loss ∝ N^(-α) × D^(-β) × C^(-γ)")

discuss_scaling_laws()
```

## Applications of Language Models

### Text Generation

```python
def text_generation_demo():
    """Language models for text generation."""

    print("Text Generation Applications:\n")

    applications = {
        'Completion': {
            'task': 'Complete partial text',
            'example': 'Input: "Once upon a time" → Output: "there was a dragon..."'
        },
        'Translation': {
            'task': 'Translate between languages',
            'example': 'Input: "Hello" → Output: "Bonjour"'
        },
        'Summarization': {
            'task': 'Condense long text',
            'example': 'Input: Long article → Output: Brief summary'
        },
        'Dialogue': {
            'task': 'Conversational responses',
            'example': 'User: "How are you?" → Bot: "I\'m doing well, thanks!"'
        },
        'Code generation': {
            'task': 'Generate code from description',
            'example': 'Input: "Sort a list" → Output: Python sorting code'
        }
    }

    for app, info in applications.items():
        print(f"{app}:")
        print(f"  Task: {info['task']}")
        print(f"  Example: {info['example']}")
        print()

text_generation_demo()
```

### Understanding Tasks

```python
def understanding_tasks_demo():
    """Language models for NLU tasks."""

    print("Natural Language Understanding Applications:\n")

    tasks = {
        'Classification': {
            'description': 'Categorize text',
            'examples': ['Sentiment analysis', 'Topic classification', 'Spam detection']
        },
        'Sequence labeling': {
            'description': 'Label each token',
            'examples': ['Named entity recognition', 'POS tagging', 'Chunking']
        },
        'Span extraction': {
            'description': 'Extract text spans',
            'examples': ['Question answering', 'Information extraction']
        },
        'Sentence pairs': {
            'description': 'Relate two sentences',
            'examples': ['Entailment', 'Paraphrase detection', 'Similarity']
        }
    }

    for task, info in tasks.items():
        print(f"{task}: {info['description']}")
        for example in info['examples']:
            print(f"  • {example}")
        print()

understanding_tasks_demo()
```

## Summary

**Key Concepts**:

1. **Language modeling** is the task of predicting probability distributions over text sequences
2. **Self-supervised learning** uses unlabeled text, creating training signal from the data itself
3. **N-gram models** use count-based statistics with Markov assumption (limited context)
4. **Neural language models** use continuous representations and can handle variable context
5. **Autoregressive LM** predicts next token (left-to-right), good for generation
6. **Masked LM** predicts masked tokens (bidirectional), good for understanding
7. **Perplexity** measures model quality: lower is better
8. **Pretraining** on language modeling enables transfer learning to downstream tasks

**Evolution**:

```
N-grams → Neural (Feed-forward) → RNN/LSTM → Transformer → Large LMs
(1980s)   (2003)                  (2010s)     (2017)       (2018-present)
```

**Two Main Paradigms**:

| Aspect                   | Autoregressive (GPT) | Masked (BERT)         |
| ------------------------ | -------------------- | --------------------- |
| Objective                | Predict next token   | Predict masked tokens |
| Context                  | Unidirectional       | Bidirectional         |
| Training-inference match | ✓ Yes                | ✗ Mismatch            |
| Best for                 | Generation           | Understanding         |

**Why Language Modeling Matters**:

- **Unsupervised**: Massive unlabeled data available
- **General knowledge**: Models learn grammar, facts, reasoning
- **Transfer learning**: Pretrain once, fine-tune for many tasks
- **Foundation**: Basis for modern NLP (BERT, GPT, T5, etc.)

**Key Metrics**:

- **Perplexity**: $\exp(-\frac{1}{N}\sum \log P(w_i))$ - Lower is better
- **Cross-entropy loss**: Negative log-likelihood
- **Downstream task performance**: Ultimate test

**Applications**:

- **Generation**: Completion, translation, summarization, dialogue
- **Understanding**: Classification, NER, QA, sentiment analysis
- **Representation learning**: Contextual embeddings for downstream tasks

## Next Steps

- Study [Autoregressive Models](autoregressive-models.md) for GPT-style architecture and generation
- Explore [Masked Models](masked-models.md) for BERT-style architecture and understanding tasks
- Learn [Encoder-Decoder Models](encoder-decoder-models.md) for sequence-to-sequence tasks
- Understand [Pretraining and Transfer Learning](pretraining-transfer.md) in depth
- Apply to [Text Generation](../application_patterns/text-generation.md) tasks
- Progress to [Large Language Models](../llm_concepts/large-language-models.md) for modern LLMs
