# Traditional NLP Metrics

## Table of Contents

- [Introduction](#introduction)
- [Classification Metrics](#classification-metrics)
- [Sequence Generation Metrics](#sequence-generation-metrics)
- [Language Model Metrics](#language-model-metrics)
- [Information Retrieval Metrics](#information-retrieval-metrics)
- [Named Entity Recognition Metrics](#named-entity-recognition-metrics)
- [Multi-Label and Multi-Class Metrics](#multi-label-and-multi-class-metrics)
- [Choosing the Right Metric](#choosing-the-right-metric)
- [Limitations of Traditional Metrics](#limitations-of-traditional-metrics)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Traditional NLP metrics provide foundational ways to measure model performance across various tasks. These metrics have been refined over decades and remain essential for understanding model behavior, establishing baselines, and conducting rigorous evaluation.

```
Evaluation Landscape:

Classification Tasks:        Sequence Tasks:           Language Modeling:
┌─────────────────┐         ┌─────────────────┐       ┌─────────────────┐
│  Accuracy       │         │  BLEU           │       │  Perplexity     │
│  Precision      │         │  ROUGE          │       │  Cross-Entropy  │
│  Recall         │         │  METEOR         │       │  Bits/Character │
│  F1 Score       │         │  CIDEr          │       └─────────────────┘
│  ROC-AUC        │         └─────────────────┘
└─────────────────┘
```

**Why traditional metrics matter**:

- Provide interpretable, objective measurements
- Enable comparison across models and approaches
- Established baselines for well-studied tasks
- Fast to compute (no neural models required)
- Understood by researchers and practitioners

This guide covers the most important traditional metrics, when to use them, and their limitations.

## Classification Metrics

### Confusion Matrix

The foundation of classification metrics:

```
                    Predicted
                 Positive  Negative
              ┌──────────┬──────────┐
Actual   Pos  │    TP    │    FN    │
              ├──────────┼──────────┤
         Neg  │    FP    │    TN    │
              └──────────┴──────────┘

TP = True Positives  (correct positive predictions)
TN = True Negatives  (correct negative predictions)
FP = False Positives (incorrect positive predictions - Type I error)
FN = False Negatives (incorrect negative predictions - Type II error)
```

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Visualize a confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class names
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

    return cm

# Example: Sentiment classification
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # 1=positive, 0=negative
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

cm = plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'])

print("\nConfusion Matrix:")
print(cm)
print(f"\nTP: {cm[1,1]}, TN: {cm[0,0]}")
print(f"FP: {cm[0,1]}, FN: {cm[1,0]}")
```

### Accuracy

The simplest metric - percentage of correct predictions:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

```python
from sklearn.metrics import accuracy_score

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy."""
    accuracy = accuracy_score(y_true, y_pred)

    # Manual calculation
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    manual_accuracy = correct / total

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Manual calculation: {manual_accuracy:.3f}")
    print(f"Correct predictions: {correct}/{total}")

    return accuracy

accuracy = calculate_accuracy(y_true, y_pred)
```

**When to use**:

- Balanced datasets (equal class distribution)
- All classes equally important

**When NOT to use**:

- Imbalanced datasets (e.g., 99% negative, 1% positive)
  - A model predicting "negative" for everything gets 99% accuracy!

### Precision

Percentage of positive predictions that are correct:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Interpretation**: "Of all items I predicted positive, what fraction was actually positive?"

```python
from sklearn.metrics import precision_score

def calculate_precision(y_true, y_pred):
    """Calculate precision."""
    precision = precision_score(y_true, y_pred)

    # Manual calculation
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"Precision: {precision:.3f}")
    print(f"Manual calculation: {manual_precision:.3f}")
    print(f"Interpretation: {precision*100:.1f}% of positive predictions are correct")

    return precision

precision = calculate_precision(y_true, y_pred)
```

**When to prioritize**:

- False positives are costly
- Example: Spam detection (marking good email as spam is bad)
- Example: Medical diagnosis (false alarms cause unnecessary stress/treatment)

### Recall (Sensitivity, True Positive Rate)

Percentage of actual positives that are correctly identified:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Interpretation**: "Of all actual positive items, what fraction did I find?"

```python
from sklearn.metrics import recall_score

def calculate_recall(y_true, y_pred):
    """Calculate recall."""
    recall = recall_score(y_true, y_pred)

    # Manual calculation
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"Recall: {recall:.3f}")
    print(f"Manual calculation: {manual_recall:.3f}")
    print(f"Interpretation: Found {recall*100:.1f}% of all positive items")

    return recall

recall = calculate_recall(y_true, y_pred)
```

**When to prioritize**:

- False negatives are costly
- Example: Disease detection (missing a sick patient is dangerous)
- Example: Fraud detection (missing fraud is costly)

### F1 Score

Harmonic mean of precision and recall:

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
$$

**Why harmonic mean?** Penalizes extreme values more than arithmetic mean.

```python
from sklearn.metrics import f1_score

def calculate_f1(y_true, y_pred):
    """Calculate F1 score."""
    f1 = f1_score(y_true, y_pred)

    # Manual calculation
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    manual_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"F1 Score: {f1:.3f}")
    print(f"Manual calculation: {manual_f1:.3f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")

    # Show why harmonic mean matters
    print("\nHarmonic vs Arithmetic mean:")
    arithmetic_mean = (precision + recall) / 2
    print(f"  Arithmetic mean: {arithmetic_mean:.3f}")
    print(f"  Harmonic mean (F1): {f1:.3f}")
    print(f"  → Harmonic mean penalizes imbalance")

    return f1

f1 = calculate_f1(y_true, y_pred)

# Example showing F1 penalty for imbalance
print("\n" + "="*50)
print("Example: High precision, low recall")
print("="*50)
precision_high = 0.9
recall_low = 0.2
f1_imbalanced = 2 * (precision_high * recall_low) / (precision_high + recall_low)
arithmetic = (precision_high + recall_low) / 2
print(f"Precision: {precision_high:.2f}, Recall: {recall_low:.2f}")
print(f"Arithmetic mean: {arithmetic:.2f}")
print(f"F1 (harmonic): {f1_imbalanced:.2f}")
print(f"→ F1 heavily penalizes the low recall!")
```

**When to use**:

- Need balance between precision and recall
- Imbalanced datasets
- Single metric to optimize

### F-Beta Score

Weighted version of F1:

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

- $\beta < 1$: Emphasize precision
- $\beta = 1$: Equal weight (F1)
- $\beta > 1$: Emphasize recall

```python
from sklearn.metrics import fbeta_score

def compare_fbeta(y_true, y_pred):
    """Compare different beta values."""

    betas = [0.5, 1.0, 2.0]

    print("F-beta scores with different beta values:\n")
    for beta in betas:
        score = fbeta_score(y_true, y_pred, beta=beta)

        if beta < 1:
            emphasis = "precision"
        elif beta == 1:
            emphasis = "balanced"
        else:
            emphasis = "recall"

        print(f"F{beta} (emphasizes {emphasis}): {score:.3f}")

compare_fbeta(y_true, y_pred)
```

### ROC Curve and AUC

For models that output probabilities:

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_proba):
    """
    Plot ROC curve and calculate AUC.

    Args:
        y_true: True labels (0 or 1)
        y_proba: Predicted probabilities for positive class
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Interpretation
    print(f"AUC Score: {auc:.3f}")
    print("\nInterpretation:")
    if auc >= 0.9:
        print("  Excellent classifier")
    elif auc >= 0.8:
        print("  Good classifier")
    elif auc >= 0.7:
        print("  Fair classifier")
    elif auc >= 0.6:
        print("  Poor classifier")
    else:
        print("  Random or worse")

    return auc

# Example with probability predictions
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_proba = [0.1, 0.8, 0.7, 0.3, 0.9, 0.2, 0.85, 0.6, 0.4, 0.15]

auc = plot_roc_curve(y_true, y_proba)
```

**AUC Interpretation**:

- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random (predictions inverted)

**Advantages**:

- Threshold-independent
- Good for imbalanced datasets
- Single number summary

### Precision-Recall Curve

Alternative to ROC, better for imbalanced datasets:

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(y_true, y_proba):
    """Plot Precision-Recall curve."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Average Precision: {avg_precision:.3f}")

    return avg_precision

ap = plot_pr_curve(y_true, y_proba)
```

**When to use PR curve vs ROC**:

- Balanced data: ROC curve
- Imbalanced data: PR curve (more informative)

## Sequence Generation Metrics

### BLEU (Bilingual Evaluation Understudy)

For machine translation and text generation. Measures n-gram overlap with reference text(s).

$$
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

Where:

- $p_n$ = n-gram precision
- $BP$ = brevity penalty (penalizes short outputs)
- $w_n$ = weights (typically uniform)

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk

nltk.download('punkt', quiet=True)

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score.

    Args:
        reference: List of reference tokens (or list of lists for multiple references)
        candidate: List of candidate tokens
    """
    # Handle single reference
    if isinstance(reference[0], str):
        reference = [reference]

    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
    smoothing = SmoothingFunction()

    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0),
                          smoothing_function=smoothing.method1)
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0),
                          smoothing_function=smoothing.method1)
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0),
                          smoothing_function=smoothing.method1)
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoothing.method1)

    print("BLEU Scores:")
    print(f"  BLEU-1 (unigrams): {bleu1:.4f}")
    print(f"  BLEU-2 (bigrams):  {bleu2:.4f}")
    print(f"  BLEU-3 (trigrams): {bleu3:.4f}")
    print(f"  BLEU-4 (4-grams):  {bleu4:.4f}")

    return bleu4

# Example: Machine translation
reference = "the cat is on the mat".split()
candidate_good = "the cat is on the mat".split()  # Perfect match
candidate_ok = "a cat is on the mat".split()      # Good but not perfect
candidate_bad = "cat mat on the is".split()       # Poor

print("Reference:", ' '.join(reference))
print("\nCandidate 1 (perfect):", ' '.join(candidate_good))
calculate_bleu([reference], candidate_good)

print("\n" + "="*50)
print("\nCandidate 2 (good):", ' '.join(candidate_ok))
calculate_bleu([reference], candidate_ok)

print("\n" + "="*50)
print("\nCandidate 3 (poor):", ' '.join(candidate_bad))
calculate_bleu([reference], candidate_bad)
```

**BLEU Characteristics**:

- Range: 0-1 (higher is better)
- Corpus-level metric (better on multiple sentences)
- Emphasizes precision (matches in candidate)
- Has brevity penalty to prevent very short outputs
- Insensitive to synonyms and paraphrasing

**Multiple references**:

```python
def bleu_multiple_references():
    """BLEU with multiple reference translations."""

    # Multiple valid translations
    references = [
        "the cat is on the mat".split(),
        "there is a cat on the mat".split(),
        "a cat is sitting on the mat".split()
    ]

    candidate = "the cat is on the mat".split()

    bleu = sentence_bleu(references, candidate)

    print("References:")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {' '.join(ref)}")
    print(f"\nCandidate: {' '.join(candidate)}")
    print(f"BLEU-4: {bleu:.4f}")

bleu_multiple_references()
```

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

For summarization. Measures overlap between generated summary and reference(s).

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference, candidate):
    """
    Calculate ROUGE scores.

    Args:
        reference: Reference text (string)
        candidate: Candidate text (string)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    print("ROUGE Scores:")
    print(f"\nROUGE-1 (unigram overlap):")
    print(f"  Precision: {scores['rouge1'].precision:.4f}")
    print(f"  Recall:    {scores['rouge1'].recall:.4f}")
    print(f"  F1:        {scores['rouge1'].fmeasure:.4f}")

    print(f"\nROUGE-2 (bigram overlap):")
    print(f"  Precision: {scores['rouge2'].precision:.4f}")
    print(f"  Recall:    {scores['rouge2'].recall:.4f}")
    print(f"  F1:        {scores['rouge2'].fmeasure:.4f}")

    print(f"\nROUGE-L (longest common subsequence):")
    print(f"  Precision: {scores['rougeL'].precision:.4f}")
    print(f"  Recall:    {scores['rougeL'].recall:.4f}")
    print(f"  F1:        {scores['rougeL'].fmeasure:.4f}")

    return scores

# Example: Text summarization
reference = "The cat sat on the mat. The dog barked loudly."
candidate_good = "The cat sat on the mat. The dog barked."
candidate_bad = "Animals were present in the location."

print("Reference:", reference)
print("\nCandidate 1 (good):", candidate_good)
calculate_rouge(reference, candidate_good)

print("\n" + "="*50)
print("\nCandidate 2 (poor):", candidate_bad)
calculate_rouge(reference, candidate_bad)
```

**ROUGE Variants**:

- **ROUGE-N**: N-gram overlap (ROUGE-1, ROUGE-2, etc.)
- **ROUGE-L**: Longest common subsequence
- **ROUGE-W**: Weighted longest common subsequence
- **ROUGE-S**: Skip-bigram co-occurrence

**BLEU vs ROUGE**:

```
Metric    Focus      Use Case           Characteristics
──────────────────────────────────────────────────────────
BLEU      Precision  Translation        • Penalizes missing words less
                                        • Brevity penalty
                                        • Corpus-level better

ROUGE     Recall     Summarization      • Penalizes missing content more
                                        • Multiple variants (N, L, W, S)
                                        • Sentence-level OK
```

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

Improved over BLEU with stemming, synonyms, and paraphrasing:

```python
from nltk.translate.meteor_score import meteor_score
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def calculate_meteor(reference, candidate):
    """
    Calculate METEOR score.

    Args:
        reference: Reference text (string or list of tokens)
        candidate: Candidate text (string or list of tokens)
    """
    # Tokenize if strings
    if isinstance(reference, str):
        reference = reference.split()
    if isinstance(candidate, str):
        candidate = candidate.split()

    score = meteor_score([reference], candidate)

    print(f"METEOR Score: {score:.4f}")

    return score

# Example showing METEOR's synonym handling
reference = "the cat is on the mat"
candidate1 = "the cat is on the mat"      # Exact match
candidate2 = "the feline is on the mat"   # Synonym
candidate3 = "the dog is on the mat"      # Different word

print("Reference:", reference)
print("\nCandidate 1 (exact):", candidate1)
calculate_meteor(reference, candidate1)

print("\nCandidate 2 (synonym 'feline'):", candidate2)
calculate_meteor(reference, candidate2)

print("\nCandidate 3 (different word):", candidate3)
calculate_meteor(reference, candidate3)
```

**METEOR Features**:

- Stems words (running → run)
- Matches synonyms (using WordNet)
- Considers word order
- Harmonic mean of precision and recall
- Typically correlates better with human judgment than BLEU

### CIDEr (Consensus-based Image Description Evaluation)

Originally for image captioning, also used for text generation:

```
CIDEr measures consensus by comparing n-grams to multiple references
using TF-IDF weighting.

Key idea: Common words (in, the, a) weighted less than rare, descriptive words.
```

## Language Model Metrics

### Perplexity

Measures how well a probability model predicts a sample. Lower is better.

$$
\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_1, \ldots, w_{i-1})\right)
$$

**Interpretation**: "The model is as confused as if it had to choose uniformly among X possibilities."

```python
import torch
import torch.nn.functional as F
import math

def calculate_perplexity(model, tokens):
    """
    Calculate perplexity for a language model.

    Args:
        model: Language model
        tokens: Input token IDs

    Returns:
        Perplexity value
    """
    # Get model predictions
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits

    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tokens[..., 1:].contiguous()

    # Calculate cross-entropy loss
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    # Perplexity is exp(loss)
    perplexity = torch.exp(loss)

    return perplexity.item()

# Simplified example
def perplexity_example():
    """Simplified perplexity calculation."""

    # Example: Model assigns these probabilities to each next word
    probabilities = [0.7, 0.6, 0.8, 0.5, 0.7]  # Higher = better

    # Calculate perplexity
    log_prob_sum = sum(math.log(p) for p in probabilities)
    avg_log_prob = log_prob_sum / len(probabilities)
    perplexity = math.exp(-avg_log_prob)

    print("Word probabilities:", probabilities)
    print(f"Average log probability: {avg_log_prob:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    # Low vs high perplexity
    print("\n" + "="*50)
    print("Comparison:")
    print("="*50)

    # Good model (high probabilities)
    good_probs = [0.9, 0.85, 0.95, 0.9, 0.88]
    good_ppl = math.exp(-sum(math.log(p) for p in good_probs) / len(good_probs))

    # Bad model (low probabilities)
    bad_probs = [0.3, 0.2, 0.4, 0.25, 0.35]
    bad_ppl = math.exp(-sum(math.log(p) for p in bad_probs) / len(bad_probs))

    print(f"Good model (confident): perplexity = {good_ppl:.2f}")
    print(f"Bad model (uncertain):  perplexity = {bad_ppl:.2f}")
    print("\n→ Lower perplexity = better language model")

perplexity_example()
```

**Perplexity guidelines**:

- Perplexity = 1: Perfect model (always correct)
- Perplexity = 10: Confused between ~10 possibilities
- Perplexity = 100: Very uncertain
- Perplexity = Vocabulary size: Random guessing

### Cross-Entropy and Bits per Character

Alternative ways to measure language model quality:

```python
def calculate_metrics(probabilities):
    """Calculate cross-entropy, perplexity, and bits per character."""

    # Cross-entropy (average negative log probability)
    cross_entropy = -sum(math.log2(p) for p in probabilities) / len(probabilities)

    # Perplexity (base 2)
    perplexity = 2 ** cross_entropy

    # Bits per character (same as cross-entropy with log2)
    bits_per_char = cross_entropy

    print(f"Cross-Entropy: {cross_entropy:.4f} bits")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Bits per Character: {bits_per_char:.4f}")

    return cross_entropy, perplexity

# Example
probs = [0.8, 0.7, 0.9, 0.75, 0.85]
calculate_metrics(probs)
```

## Information Retrieval Metrics

### Precision@K and Recall@K

For ranked retrieval results:

```python
def precision_at_k(relevant_items, retrieved_items, k):
    """
    Calculate precision at k.

    Args:
        relevant_items: Set of relevant item IDs
        retrieved_items: List of retrieved item IDs (ranked)
        k: Cutoff position
    """
    retrieved_at_k = set(retrieved_items[:k])
    relevant_retrieved = relevant_items & retrieved_at_k

    precision = len(relevant_retrieved) / k

    return precision

def recall_at_k(relevant_items, retrieved_items, k):
    """Calculate recall at k."""
    retrieved_at_k = set(retrieved_items[:k])
    relevant_retrieved = relevant_items & retrieved_at_k

    recall = len(relevant_retrieved) / len(relevant_items)

    return recall

# Example: Search results
relevant_docs = {2, 5, 7, 9, 12}  # 5 relevant documents
retrieved_docs = [2, 3, 5, 8, 7, 10, 12, 15, 9, 20]  # Ranked results

print("Relevant documents:", relevant_docs)
print("Retrieved documents (ranked):", retrieved_docs)
print()

for k in [1, 3, 5, 10]:
    p_at_k = precision_at_k(relevant_docs, retrieved_docs, k)
    r_at_k = recall_at_k(relevant_docs, retrieved_docs, k)

    print(f"At position {k}:")
    print(f"  Precision@{k}: {p_at_k:.3f} ({int(p_at_k*k)}/{k} relevant)")
    print(f"  Recall@{k}:    {r_at_k:.3f} ({int(r_at_k*len(relevant_docs))}/{len(relevant_docs)} found)")
    print()
```

### Mean Average Precision (MAP)

Average precision across queries:

```python
def average_precision(relevant_items, retrieved_items):
    """
    Calculate average precision for a single query.

    Args:
        relevant_items: Set of relevant item IDs
        retrieved_items: List of retrieved item IDs (ranked)
    """
    num_relevant = len(relevant_items)
    precisions = []
    num_hits = 0

    for i, item in enumerate(retrieved_items, 1):
        if item in relevant_items:
            num_hits += 1
            precision_at_i = num_hits / i
            precisions.append(precision_at_i)

    if not precisions:
        return 0.0

    avg_precision = sum(precisions) / num_relevant

    return avg_precision

def mean_average_precision(queries_data):
    """
    Calculate MAP across multiple queries.

    Args:
        queries_data: List of (relevant_items, retrieved_items) tuples
    """
    aps = []

    for relevant, retrieved in queries_data:
        ap = average_precision(relevant, retrieved)
        aps.append(ap)

    map_score = sum(aps) / len(aps)

    return map_score

# Example
query1_relevant = {1, 3, 5}
query1_retrieved = [1, 2, 3, 4, 5]

query2_relevant = {2, 4, 6, 8}
query2_retrieved = [2, 3, 4, 5, 6, 7, 8]

queries = [
    (query1_relevant, query1_retrieved),
    (query2_relevant, query2_retrieved)
]

map_score = mean_average_precision(queries)

print("Query 1:")
print(f"  Relevant: {query1_relevant}")
print(f"  Retrieved: {query1_retrieved}")
ap1 = average_precision(query1_relevant, query1_retrieved)
print(f"  AP: {ap1:.3f}")

print("\nQuery 2:")
print(f"  Relevant: {query2_relevant}")
print(f"  Retrieved: {query2_retrieved}")
ap2 = average_precision(query2_relevant, query2_retrieved)
print(f"  AP: {ap2:.3f}")

print(f"\nMAP: {map_score:.3f}")
```

### Mean Reciprocal Rank (MRR)

Focuses on the rank of the first relevant result:

```python
def reciprocal_rank(relevant_items, retrieved_items):
    """
    Calculate reciprocal rank.

    Args:
        relevant_items: Set of relevant item IDs
        retrieved_items: List of retrieved item IDs (ranked)
    """
    for i, item in enumerate(retrieved_items, 1):
        if item in relevant_items:
            return 1.0 / i

    return 0.0  # No relevant item found

def mean_reciprocal_rank(queries_data):
    """Calculate MRR across queries."""
    rrs = [reciprocal_rank(relevant, retrieved)
           for relevant, retrieved in queries_data]

    mrr = sum(rrs) / len(rrs)

    return mrr

# Example
queries = [
    ({5, 7}, [1, 2, 5, 7]),    # First relevant at position 3
    ({3, 8}, [3, 5, 8]),        # First relevant at position 1
    ({6}, [1, 2, 3, 4, 6])      # First relevant at position 5
]

mrr = mean_reciprocal_rank(queries)

print("MRR Calculation:\n")
for i, (relevant, retrieved) in enumerate(queries, 1):
    rr = reciprocal_rank(relevant, retrieved)
    first_pos = next((j for j, item in enumerate(retrieved, 1) if item in relevant), -1)

    print(f"Query {i}:")
    print(f"  Relevant: {relevant}, Retrieved: {retrieved}")
    print(f"  First relevant at position: {first_pos}")
    print(f"  Reciprocal Rank: {rr:.3f}")
    print()

print(f"Mean Reciprocal Rank: {mrr:.3f}")
```

### Normalized Discounted Cumulative Gain (NDCG)

Accounts for position and graded relevance:

```python
import math

def dcg_at_k(relevances, k):
    """
    Calculate DCG@k.

    Args:
        relevances: List of relevance scores (in ranking order)
        k: Cutoff position
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], 1):
        dcg += rel / math.log2(i + 1)

    return dcg

def ndcg_at_k(relevances, k):
    """
    Calculate NDCG@k.

    Args:
        relevances: List of relevance scores (in ranking order)
        k: Cutoff position
    """
    # Actual DCG
    dcg = dcg_at_k(relevances, k)

    # Ideal DCG (sort relevances descending)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    # Normalize
    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg

    return ndcg

# Example with graded relevance
# Relevance scale: 0 (not relevant), 1 (somewhat), 2 (relevant), 3 (highly relevant)
actual_ranking = [3, 2, 0, 1, 2, 0, 1]  # Relevance scores in retrieval order
ideal_ranking = sorted(actual_ranking, reverse=True)  # [3, 2, 2, 1, 1, 0, 0]

print("Actual ranking relevances:", actual_ranking)
print("Ideal ranking relevances:", ideal_ranking)
print()

for k in [3, 5, 7]:
    dcg = dcg_at_k(actual_ranking, k)
    idcg = dcg_at_k(ideal_ranking, k)
    ndcg = ndcg_at_k(actual_ranking, k)

    print(f"At position {k}:")
    print(f"  DCG@{k}:  {dcg:.3f}")
    print(f"  IDCG@{k}: {idcg:.3f}")
    print(f"  NDCG@{k}: {ndcg:.3f}")
    print()
```

## Named Entity Recognition Metrics

### Entity-Level F1

```python
def entity_level_f1(true_entities, pred_entities):
    """
    Calculate entity-level precision, recall, and F1.

    Args:
        true_entities: Set of (start, end, label) tuples
        pred_entities: Set of (start, end, label) tuples
    """
    tp = len(true_entities & pred_entities)
    fp = len(pred_entities - true_entities)
    fn = len(true_entities - pred_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Entity-Level Metrics:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    return precision, recall, f1

# Example: NER predictions
# Format: (start_position, end_position, entity_label)
true_entities = {
    (0, 4, 'PER'),   # "John"
    (10, 16, 'LOC'), # "London"
    (25, 30, 'ORG')  # "Apple"
}

pred_entities = {
    (0, 4, 'PER'),   # Correct
    (10, 16, 'LOC'), # Correct
    (25, 30, 'PER'), # Wrong label
    (35, 40, 'ORG')  # Extra prediction
}

print("True entities:", true_entities)
print("Predicted entities:", pred_entities)
print()

entity_level_f1(true_entities, pred_entities)
```

### Token-Level F1

```python
from sklearn.metrics import classification_report

def token_level_metrics(y_true, y_pred, labels):
    """
    Calculate token-level metrics for NER.

    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        labels: List of label names
    """
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=labels, zero_division=0)

    print("Token-Level Classification Report:")
    print(report)

# Example
y_true = ['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'O', 'B-ORG']
y_pred = ['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'B-LOC', 'B-ORG']

labels = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']

token_level_metrics(y_true, y_pred, labels)
```

## Multi-Label and Multi-Class Metrics

### Macro vs Micro vs Weighted Averaging

```python
from sklearn.metrics import precision_recall_fscore_support

def compare_averaging_strategies(y_true, y_pred, labels):
    """Compare different averaging strategies."""

    # Calculate for each averaging method
    for average in ['micro', 'macro', 'weighted']:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )

        print(f"{average.upper()} Average:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print()

# Example: Multi-class classification
y_true = [0, 1, 2, 0, 1, 2, 2, 2, 0, 1] * 10  # Imbalanced (more class 2)
y_pred = [0, 1, 1, 0, 1, 2, 2, 1, 0, 1] * 10

compare_averaging_strategies(y_true, y_pred, labels=[0, 1, 2])

print("="*50)
print("Explanation:")
print("="*50)
print("""
Micro:    Calculate globally (all predictions together)
          → Good for overall performance, affected by class imbalance

Macro:    Calculate per-class, then average (equal weight per class)
          → Good when all classes equally important

Weighted: Calculate per-class, average weighted by support
          → Good when some classes more frequent/important
""")
```

## Choosing the Right Metric

```python
def metric_decision_guide():
    """Guide for choosing the right metric."""

    scenarios = [
        {
            'task': 'Spam detection',
            'characteristic': 'FP costly (good email marked spam)',
            'metric': 'Precision',
            'alternative': 'F1 with β < 1 (emphasize precision)'
        },
        {
            'task': 'Disease detection',
            'characteristic': 'FN costly (miss sick patient)',
            'metric': 'Recall',
            'alternative': 'F1 with β > 1 (emphasize recall)'
        },
        {
            'task': 'General classification (balanced)',
            'characteristic': 'Equal class importance',
            'metric': 'Accuracy or F1',
            'alternative': 'ROC-AUC for ranking'
        },
        {
            'task': 'General classification (imbalanced)',
            'characteristic': 'Unequal class distribution',
            'metric': 'F1, PR-AUC',
            'alternative': 'Macro F1 if all classes equally important'
        },
        {
            'task': 'Machine translation',
            'characteristic': 'Need multiple references',
            'metric': 'BLEU',
            'alternative': 'METEOR (handles synonyms), BERTScore (semantic)'
        },
        {
            'task': 'Summarization',
            'characteristic': 'Recall important (capture key info)',
            'metric': 'ROUGE-L, ROUGE-2',
            'alternative': 'BERTScore for semantic similarity'
        },
        {
            'task': 'Language model',
            'characteristic': 'Predict probability distribution',
            'metric': 'Perplexity',
            'alternative': 'Cross-entropy, Bits/Character'
        },
        {
            'task': 'Search/Retrieval',
            'characteristic': 'Ranked results',
            'metric': 'NDCG@k, MAP',
            'alternative': 'MRR (first result important), Precision@k'
        },
        {
            'task': 'Question answering',
            'characteristic': 'Single correct answer needed',
            'metric': 'MRR, Accuracy@1',
            'alternative': 'Exact Match, F1 (token overlap)'
        }
    ]

    print("METRIC SELECTION GUIDE")
    print("=" * 80)
    print()

    for scenario in scenarios:
        print(f"Task: {scenario['task']}")
        print(f"  Characteristic: {scenario['characteristic']}")
        print(f"  Primary Metric: {scenario['metric']}")
        print(f"  Alternative: {scenario['alternative']}")
        print()

metric_decision_guide()
```

## Limitations of Traditional Metrics

### 1. Insensitivity to Semantic Similarity

```python
def demonstrate_semantic_insensitivity():
    """Show how traditional metrics miss semantic similarity."""

    reference = "The cat sat on the mat"

    candidates = [
        ("Exact match", "The cat sat on the mat"),
        ("Synonym", "The feline sat on the mat"),
        ("Paraphrase", "A cat was sitting on a mat"),
        ("Semantic equivalent", "There was a cat on the mat"),
        ("Different", "The dog ran in the park")
    ]

    print("Reference:", reference)
    print("\n" + "="*60)

    for desc, candidate in candidates:
        # Calculate BLEU
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        bleu = sentence_bleu([ref_tokens], cand_tokens)

        print(f"\n{desc}:")
        print(f"  Candidate: {candidate}")
        print(f"  BLEU-4: {bleu:.3f}")

    print("\n" + "="*60)
    print("\nProblem: BLEU gives low scores to valid paraphrases!")
    print("→ Needs lexical overlap, misses semantic meaning")

demonstrate_semantic_insensitivity()
```

### 2. Gaming Metrics

```python
def demonstrate_metric_gaming():
    """Show how metrics can be gamed."""

    reference = "The quick brown fox jumps over the lazy dog"

    # Strategy 1: Repeat reference for high n-gram overlap
    gaming_candidate = reference + " " + reference

    ref_tokens = reference.split()
    game_tokens = gaming_candidate.split()

    bleu = sentence_bleu([ref_tokens], game_tokens)

    print("Gaming BLEU Score:")
    print(f"Reference: {reference}")
    print(f"Gaming strategy: {gaming_candidate}")
    print(f"BLEU: {bleu:.3f}")
    print("\nProblem: Repetition increases n-gram overlap!")
    print("Note: BLEU has brevity penalty but not perfect")

demonstrate_metric_gaming()
```

### 3. No Accounting for Factuality

```python
def demonstrate_factuality_blind():
    """Traditional metrics don't detect factual errors."""

    print("Factuality Blindness:\n")

    reference = "Paris is the capital of France"

    candidates = [
        ("Correct", "Paris is the capital of France"),
        ("Wrong fact", "Paris is the capital of Germany"),
        ("Fluent but wrong", "The capital of France is Berlin")
    ]

    for desc, candidate in candidates:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        bleu = sentence_bleu([ref_tokens], cand_tokens)

        print(f"{desc}:")
        print(f"  {candidate}")
        print(f"  BLEU: {bleu:.3f}")
        print()

    print("Problem: High BLEU despite wrong facts!")
    print("→ Metrics measure form, not factual correctness")

demonstrate_factuality_blind()
```

### Key Limitations Summary

```
Limitation                  Problem                         Solution
─────────────────────────────────────────────────────────────────────────────
Lexical focus              Misses semantic equivalence     Use neural metrics
                                                           (BERTScore)

Insensitive to order       Word order matters for meaning  Use ROUGE-L, METEOR

No factuality check        Wrong facts score well          Use factuality metrics,
                                                           human eval

Easy to game               Optimizing metric ≠ quality     Multiple metrics,
                                                           human evaluation

Context-independent        Doesn't consider use case       Task-specific metrics

No style/fluency measure   Grammatical errors not caught   Human evaluation,
                                                           perplexity

Single reference bias      Multiple valid outputs exist    Multiple references,
                                                           neural metrics
```

## Summary

**Classification Metrics**:

```
Metric      Formula                    Use When                     Range
───────────────────────────────────────────────────────────────────────────
Accuracy    (TP+TN)/Total             Balanced data                0-1
Precision   TP/(TP+FP)                 FP costly                    0-1
Recall      TP/(TP+FN)                 FN costly                    0-1
F1          2*P*R/(P+R)                Need balance                 0-1
ROC-AUC     Area under ROC curve       Probability ranking          0-1
PR-AUC      Area under PR curve        Imbalanced data              0-1
```

**Sequence Generation Metrics**:

```
Metric      Focus       Use Case           Key Feature
──────────────────────────────────────────────────────────────
BLEU        Precision   Translation        N-gram overlap
ROUGE       Recall      Summarization      Multiple variants
METEOR      Balance     Translation        Synonyms, stemming
CIDEr       Consensus   Image captions     TF-IDF weighted
```

**Language Model Metrics**:

```
Perplexity = exp(cross-entropy)

Lower = better
  • PPL = 1: Perfect
  • PPL = 10: ~10 choices
  • PPL = 100: Very uncertain
```

**Information Retrieval Metrics**:

```
Metric        Focuses On              Best For
──────────────────────────────────────────────────────────
P@k, R@k      Top-k results           Simple ranking eval
MAP           Whole ranking           Average performance
MRR           First result            Single answer tasks
NDCG          Graded relevance        Ranked with scores
```

**Key Principles**:

1. **Match metric to task** - Different tasks need different metrics
2. **Consider multiple metrics** - No single metric captures everything
3. **Understand trade-offs** - Precision vs recall, etc.
4. **Use appropriate baselines** - Random, majority class, etc.
5. **Report confidence intervals** - Metrics have variance
6. **Validate with humans** - Metrics approximate human judgment

**Common Pitfalls**:

- Using accuracy on imbalanced data
- Optimizing single metric blindly
- Ignoring statistical significance
- Not using multiple references (BLEU/ROUGE)
- Forgetting that metrics ≠ quality
- Gaming metrics instead of improving quality

**Best Practices**:

1. **Start simple** - Accuracy, F1, BLEU/ROUGE
2. **Add sophistication** - Neural metrics, human eval
3. **Multiple perspectives** - Combine complementary metrics
4. **Error analysis** - Understand failures beyond metrics
5. **Domain expertise** - Know what matters for your task
6. **Iterate** - Metrics guide but don't define success

**Limitations**:

- **Semantic insensitivity**: Miss paraphrases
- **No factuality**: Can't detect wrong facts
- **Context-free**: Don't consider use case
- **Gaming**: Can optimize metric without improving quality
- **Single reference bias**: Many valid outputs exist

**When Traditional Metrics Fail**:

- Open-ended generation (creative writing)
- Semantic equivalence tasks (paraphrasing)
- Factuality requirements (Q&A, knowledge)
- Subjective quality (style, tone)
- → Use neural metrics, LLM-as-judge, human evaluation

## Next Steps

- Learn [Neural and Semantic Metrics](neural-metrics.md) for semantic similarity measurement
- Study [LLM Evaluation Methods](llm-evaluation.md) for modern evaluation approaches
- Explore [Benchmarks and Leaderboards](benchmarks.md) for standardized comparisons
- Master [Human Evaluation](human-evaluation.md) techniques for gold standard assessment
- Apply [Failure Analysis](failure-analysis.md) to understand model weaknesses
- Review traditional NLP tasks in [Fundamentals](../fundamentals/) for context
