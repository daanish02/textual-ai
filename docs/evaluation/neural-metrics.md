# Neural and Semantic Metrics

## Table of Contents

- [Introduction](#introduction)
- [Why Neural Metrics](#why-neural-metrics)
- [BERTScore](#bertscore)
- [BLEURT](#bleurt)
- [Semantic Similarity Metrics](#semantic-similarity-metrics)
- [Embedding-Based Evaluation](#embedding-based-evaluation)
- [Learned Metrics](#learned-metrics)
- [Comparing Neural vs Traditional Metrics](#comparing-neural-vs-traditional-metrics)
- [Practical Implementation](#practical-implementation)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Neural metrics use pretrained language models to evaluate text based on semantic meaning rather than lexical overlap. They can recognize paraphrases, synonyms, and semantic equivalence that traditional metrics miss.

```
Traditional Metrics:           Neural Metrics:
┌──────────────────┐          ┌──────────────────┐
│  Lexical Match   │          │  Semantic Match  │
│                  │          │                  │
│  "cat" ≠ "feline"│   vs     │  "cat" ≈ "feline"│
│  Word-for-word   │          │  Meaning-based   │
│  Fast, simple    │          │  Better aligned  │
│  Miss synonyms   │          │  with humans     │
└──────────────────┘          └──────────────────┘

Example:
  Reference:  "The cat sat on the mat"
  Candidate:  "A feline was sitting on the rug"
  
  BLEU:       Low (few exact matches)
  BERTScore:  High (same meaning)
```

**Evolution of metrics**:

```
1990s-2000s:  BLEU, ROUGE, METEOR
              → Lexical overlap

2010s:        Embeddings
              → Word-level semantics

2019+:        BERTScore, BLEURT
              → Contextual semantics

2023+:        LLM-as-judge
              → Sophisticated reasoning
```

This guide covers modern neural metrics, how they work, and when to use them.

## Why Neural Metrics

### Problems with Traditional Metrics

```python
from nltk.translate.bleu_score import sentence_bleu

def demonstrate_traditional_limitations():
    """Show where traditional metrics fail."""
    
    reference = "The doctor examined the patient"
    
    test_cases = [
        ("Exact match", "The doctor examined the patient", "Should score high"),
        ("Synonym", "The physician examined the patient", "Should score high"),
        ("Paraphrase", "The patient was examined by the doctor", "Should score high"),
        ("Different words, same meaning", "A medical professional checked the sick person", "Should score medium-high"),
        ("Wrong meaning", "The patient examined the doctor", "Should score low"),
        ("Gibberish", "The doctor patient examined", "Should score low"),
    ]
    
    print("Traditional Metric (BLEU) Limitations:\n")
    print("="*70)
    
    ref_tokens = reference.split()
    
    for desc, candidate, expected in test_cases:
        cand_tokens = candidate.split()
        bleu = sentence_bleu([ref_tokens], cand_tokens)
        
        print(f"\n{desc}:")
        print(f"  Candidate: {candidate}")
        print(f"  BLEU-4: {bleu:.3f}")
        print(f"  Expected: {expected}")
        
        # Check if BLEU aligns with expectation
        if "high" in expected and bleu < 0.5:
            print(f"  ⚠️  BLEU TOO LOW - misses semantic similarity!")
        elif "low" in expected and bleu > 0.5:
            print(f"  ⚠️  BLEU TOO HIGH - doesn't catch semantic error!")

demonstrate_traditional_limitations()
```

### Benefits of Neural Metrics

```
Advantage                Example                              Impact
────────────────────────────────────────────────────────────────────────
Semantic understanding   "car" ≈ "automobile"                Better correlation
                                                             with human judgment

Context-aware           "bank" (river) vs "bank" (money)    Disambiguates words

Paraphrase recognition  "not bad" ≈ "good"                  Catches valid rewrites

Word order sensitivity  "dog bites man" ≠ "man bites dog"   Understands structure

Multilingual support    Works across languages              Broader applicability

Learned from data       Trained on human judgments          Aligns with humans
```

## BERTScore

### How BERTScore Works

BERTScore matches tokens based on their contextual embeddings from BERT:

```
Process:
1. Embed reference and candidate with BERT
2. Match each token based on cosine similarity
3. Aggregate to get precision, recall, F1

Example:
Reference:  "The  cat  sat  on  the  mat"
            ↓    ↓    ↓    ↓    ↓    ↓
          [emb] [emb][emb][emb][emb][emb]

Candidate:  "A  feline  was  sitting  on  the  rug"
            ↓    ↓     ↓      ↓      ↓    ↓    ↓
          [emb][emb] [emb]  [emb]  [emb][emb][emb]

Match tokens by similarity:
  "cat" ↔ "feline"  (high similarity)
  "sat" ↔ "sitting" (high similarity)
  "mat" ↔ "rug"     (medium similarity)
```

$$
\text{Precision} = \frac{1}{|C|}\sum_{c \in C} \max_{r \in R} \text{sim}(c, r)
$$

$$
\text{Recall} = \frac{1}{|R|}\sum_{r \in R} \max_{c \in C} \text{sim}(r, c)
$$

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Implementing BERTScore

```python
from bert_score import score
import torch

def calculate_bertscore(references, candidates, verbose=True):
    """
    Calculate BERTScore for text pairs.
    
    Args:
        references: List of reference texts
        candidates: List of candidate texts
        verbose: Print detailed info
    
    Returns:
        Precision, recall, F1 scores
    """
    # Calculate BERTScore
    P, R, F1 = score(
        candidates, 
        references,
        lang='en',
        model_type='microsoft/deberta-xlarge-mnli',  # Best performing model
        verbose=verbose
    )
    
    if verbose:
        print("BERTScore Results:")
        print(f"  Precision: {P.mean():.4f} (±{P.std():.4f})")
        print(f"  Recall:    {R.mean():.4f} (±{R.std():.4f})")
        print(f"  F1:        {F1.mean():.4f} (±{F1.std():.4f})")
    
    return P, R, F1

# Example: Compare BERTScore vs BLEU
def compare_bertscore_bleu():
    """Compare BERTScore and BLEU on same examples."""
    
    reference = "The doctor examined the patient"
    
    test_cases = [
        "The doctor examined the patient",  # Exact
        "The physician examined the patient",  # Synonym
        "The patient was examined by the doctor",  # Passive
        "A medical professional checked the sick person",  # Paraphrase
    ]
    
    print("Comparison: BERTScore vs BLEU\n")
    print("="*70)
    
    for i, candidate in enumerate(test_cases, 1):
        # BLEU
        from nltk.translate.bleu_score import sentence_bleu
        bleu = sentence_bleu([reference.split()], candidate.split())
        
        # BERTScore
        P, R, F1 = calculate_bertscore([reference], [candidate], verbose=False)
        
        print(f"\nCase {i}:")
        print(f"  Candidate: {candidate}")
        print(f"  BLEU:      {bleu:.3f}")
        print(f"  BERTScore: {F1.item():.3f}")
        
        # Analysis
        diff = F1.item() - bleu
        if diff > 0.2:
            print(f"  → BERTScore much higher (+{diff:.3f}) - captures semantic similarity")

compare_bertscore_bleu()
```

### BERTScore Variants

```python
def bertscore_model_comparison():
    """Compare different BERT models for BERTScore."""
    
    reference = "The cat sat on the mat"
    candidate = "A feline was sitting on the rug"
    
    models = [
        'bert-base-uncased',           # Original BERT
        'roberta-large',               # RoBERTa (better)
        'microsoft/deberta-xlarge-mnli',  # DeBERTa (best)
        'distilbert-base-uncased',     # Smaller, faster
    ]
    
    print("BERTScore with Different Models:\n")
    print("="*70)
    
    for model in models:
        P, R, F1 = score(
            [candidate],
            [reference],
            model_type=model,
            verbose=False
        )
        
        print(f"\nModel: {model}")
        print(f"  F1: {F1.item():.4f}")
        
        if 'distil' in model:
            print(f"  → Faster but slightly lower quality")
        elif 'deberta' in model:
            print(f"  → Best quality, slower")

# Note: Run this if you want to compare models
# bertscore_model_comparison()
```

### BERTScore with IDF Weighting

Improve BERTScore by weighting rare words more:

```python
def bertscore_with_idf():
    """Use IDF weighting to emphasize rare words."""
    
    references = ["The cat sat on the mat"]
    candidates = ["The cat sat on the mat"]  # Exact match
    
    # Standard BERTScore
    P, R, F1 = score(candidates, references, lang='en', verbose=False)
    print(f"Standard BERTScore F1: {F1.item():.4f}")
    
    # With IDF weighting (emphasizes rare/important words)
    P_idf, R_idf, F1_idf = score(
        candidates, 
        references, 
        lang='en',
        idf=True,  # Enable IDF weighting
        verbose=False
    )
    print(f"IDF-weighted BERTScore F1: {F1_idf.item():.4f}")
    
    print("\nIDF weighting gives more weight to rare, informative words")
    print("(e.g., 'physician' > 'the')")

bertscore_with_idf()
```

### BERTScore Interpretation

```python
def interpret_bertscore():
    """Guidelines for interpreting BERTScore values."""
    
    guidelines = """
BERTScore Interpretation:

F1 Score Range:
  0.90 - 1.00:  Excellent - Very similar semantically
  0.80 - 0.90:  Good - Captures main meaning
  0.70 - 0.80:  Fair - Some semantic overlap
  0.60 - 0.70:  Poor - Limited similarity
  < 0.60:       Bad - Different meanings

Precision vs Recall:
  High P, Low R:  Candidate has correct info but misses content
                  → Generated text too concise
  
  Low P, High R:  Candidate covers reference but has extra/wrong info
                  → Generated text too verbose or hallucinated
  
  Balanced:       Good match in both coverage and accuracy

Model Selection Impact:
  BERT-base:      F1 ~0.85 baseline
  RoBERTa:        F1 ~0.87 (+0.02 improvement)
  DeBERTa:        F1 ~0.89 (+0.04 improvement)
  
  → Better models → higher scores, better human correlation

When to Use:
  ✓ Paraphrase detection
  ✓ Translation evaluation (with semantic similarity)
  ✓ Summarization
  ✓ Text generation quality
  ✓ When meaning matters more than exact words

When NOT to Use:
  ✗ Need exact matches (e.g., named entities, dates)
  ✗ Code evaluation (syntax matters)
  ✗ Very short texts (1-2 words) - unreliable
  ✗ Need factuality checking (semantic ≠ factual)
"""
    
    print(guidelines)

interpret_bertscore()
```

## BLEURT

### What is BLEURT

BLEURT (Bilingual Evaluation Understudy with Representations from Transformers) is a learned metric trained to predict human judgments:

```
BLEURT Pipeline:

1. Pretrain on BERT
   ↓
2. Train on synthetic data (millions of examples)
   • Backtranslation
   • Masking
   • Perturbations
   ↓
3. Fine-tune on human ratings
   • WMT datasets
   • Human quality scores
   ↓
4. Predict quality score for new text pairs

Result: Better correlation with human judgment than BLEU or BERTScore
```

### Using BLEURT

```python
from bleurt import score as bleurt_score

def calculate_bleurt(references, candidates):
    """
    Calculate BLEURT scores.
    
    Args:
        references: List of reference texts
        candidates: List of candidate texts
    
    Returns:
        BLEURT scores
    """
    # Load BLEURT checkpoint
    scorer = bleurt_score.BleurtScorer('BLEURT-20')
    
    # Calculate scores
    scores = scorer.score(references=references, candidates=candidates)
    
    print("BLEURT Scores:")
    for i, (ref, cand, score) in enumerate(zip(references, candidates, scores), 1):
        print(f"\n{i}. Reference: {ref}")
        print(f"   Candidate: {cand}")
        print(f"   BLEURT: {score:.4f}")
    
    return scores

# Example
references = [
    "The cat sat on the mat",
    "I love machine learning",
]

candidates = [
    "A feline was sitting on the rug",  # Good paraphrase
    "I hate machine learning",  # Opposite meaning
]

scores = calculate_bleurt(references, candidates)

print("\n" + "="*60)
print("\nInterpretation:")
print(f"Score 1: {scores[0]:.3f} - Good (captures same meaning)")
print(f"Score 2: {scores[1]:.3f} - Poor (opposite meaning)")
```

### BLEURT vs BERTScore

```python
def compare_bleurt_bertscore():
    """Compare BLEURT and BERTScore."""
    
    reference = "The movie was excellent"
    
    test_cases = [
        ("Exact", "The movie was excellent"),
        ("Synonym", "The film was excellent"),
        ("Paraphrase", "The movie was really good"),
        ("Opposite", "The movie was terrible"),
        ("Related but different", "The actor was excellent"),
    ]
    
    print("BLEURT vs BERTScore Comparison:\n")
    print("="*70)
    
    for desc, candidate in test_cases:
        # BERTScore
        from bert_score import score
        _, _, F1 = score([candidate], [reference], lang='en', verbose=False)
        bertscore = F1.item()
        
        # BLEURT (simulated - replace with actual BLEURT call)
        # bleurt = calculate_bleurt([reference], [candidate])[0]
        
        print(f"\n{desc}:")
        print(f"  Candidate: {candidate}")
        print(f"  BERTScore: {bertscore:.3f}")
        # print(f"  BLEURT:    {bleurt:.3f}")

compare_bleurt_bertscore()
```

**Key Differences**:

```
Feature          BERTScore                 BLEURT
────────────────────────────────────────────────────────────
Training         Pretrained only          Pretrained + Fine-tuned
Data             No human labels          Trained on human ratings
Correlation      Good                     Better
Speed            Faster                   Slower
Setup            Simple                   Needs checkpoint
Scores           0-1 (similarity)         Continuous (quality)
```

## Semantic Similarity Metrics

### Cosine Similarity

Basic embedding similarity:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity_metric():
    """Calculate cosine similarity between texts."""
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Example texts
    reference = "The cat sat on the mat"
    candidates = [
        "The cat sat on the mat",  # Exact
        "A feline was sitting on the rug",  # Paraphrase
        "The dog ran in the park",  # Different
    ]
    
    # Embed
    ref_embedding = model.encode(reference)
    cand_embeddings = model.encode(candidates)
    
    # Calculate cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("Cosine Similarity Scores:\n")
    print("="*60)
    
    for i, (candidate, cand_emb) in enumerate(zip(candidates, cand_embeddings), 1):
        similarity = cosine_sim(ref_embedding, cand_emb)
        
        print(f"\n{i}. Candidate: {candidate}")
        print(f"   Similarity: {similarity:.4f}")
        
        if similarity > 0.8:
            print(f"   → Very similar")
        elif similarity > 0.6:
            print(f"   → Somewhat similar")
        else:
            print(f"   → Different")

cosine_similarity_metric()
```

### Semantic Textual Similarity (STS)

```python
from sentence_transformers import SentenceTransformer, util

def semantic_textual_similarity():
    """Calculate semantic similarity using sentence transformers."""
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    sentence_pairs = [
        ("The cat is on the mat", "A feline sits on a rug"),  # High similarity
        ("I love programming", "I hate programming"),  # Low similarity (opposite)
        ("Python is a language", "Python is a snake"),  # Low similarity (different meaning)
        ("He opened the door", "The door was opened by him"),  # High similarity (passive)
    ]
    
    print("Semantic Textual Similarity:\n")
    print("="*70)
    
    for sent1, sent2 in sentence_pairs:
        # Encode
        emb1 = model.encode(sent1, convert_to_tensor=True)
        emb2 = model.encode(sent2, convert_to_tensor=True)
        
        # Calculate similarity
        similarity = util.cos_sim(emb1, emb2).item()
        
        print(f"\nSentence 1: {sent1}")
        print(f"Sentence 2: {sent2}")
        print(f"Similarity: {similarity:.4f}")
        
        # Interpretation
        if similarity > 0.75:
            print("→ Very similar (paraphrase)")
        elif similarity > 0.5:
            print("→ Moderately similar (related)")
        else:
            print("→ Different meaning")

semantic_textual_similarity()
```

### Sentence Embedding Models

```python
def compare_embedding_models():
    """Compare different sentence embedding models."""
    
    from sentence_transformers import SentenceTransformer
    
    models = [
        'all-MiniLM-L6-v2',      # Small, fast (384 dims)
        'all-mpnet-base-v2',     # Balanced (768 dims)
        'all-MiniLM-L12-v2',     # Medium (384 dims)
    ]
    
    sentence1 = "The cat sat on the mat"
    sentence2 = "A feline was sitting on the rug"
    
    print("Embedding Model Comparison:\n")
    print("="*60)
    
    for model_name in models:
        model = SentenceTransformer(model_name)
        
        emb1 = model.encode(sentence1)
        emb2 = model.encode(sentence2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        print(f"\nModel: {model_name}")
        print(f"  Dimensions: {len(emb1)}")
        print(f"  Similarity: {similarity:.4f}")
        
        if 'Mini' in model_name:
            print(f"  → Faster, smaller")
        elif 'mpnet' in model_name:
            print(f"  → Best quality")

compare_embedding_models()
```

## Embedding-Based Evaluation

### Average Embedding Distance

```python
def average_embedding_distance():
    """Calculate average distance between generated texts and reference."""
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    reference = "The cat sat on the mat"
    
    # Multiple generated candidates
    candidates = [
        "The cat sat on the mat",
        "A cat was on the mat",
        "The feline sat on the rug",
        "A dog ran in the park",
    ]
    
    ref_emb = model.encode(reference)
    cand_embs = model.encode(candidates)
    
    print("Average Embedding Distance:\n")
    print("="*60)
    
    distances = []
    for i, (cand, cand_emb) in enumerate(zip(candidates, cand_embs), 1):
        # Euclidean distance
        distance = np.linalg.norm(ref_emb - cand_emb)
        distances.append(distance)
        
        # Cosine similarity
        similarity = np.dot(ref_emb, cand_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(cand_emb))
        
        print(f"\n{i}. {cand}")
        print(f"   Euclidean distance: {distance:.4f}")
        print(f"   Cosine similarity:  {similarity:.4f}")
    
    avg_distance = np.mean(distances)
    print(f"\n{'='*60}")
    print(f"Average distance: {avg_distance:.4f}")
    print(f"→ Lower average = candidates closer to reference")

average_embedding_distance()
```

### Diversity Measurement

```python
def measure_diversity():
    """Measure diversity in generated outputs."""
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Multiple generations
    generations = [
        "The cat sat on the mat",
        "A feline was on the rug",
        "The dog played outside",
        "A bird flew in the sky",
    ]
    
    embeddings = model.encode(generations)
    
    print("Diversity Measurement:\n")
    print("="*60)
    
    # Calculate pairwise similarities
    n = len(generations)
    similarities = []
    
    for i in range(n):
        for j in range(i+1, n):
            sim = np.dot(embeddings[i], embeddings[j]) / \
                  (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            similarities.append(sim)
            print(f"\nSimilarity between {i+1} and {j+1}: {sim:.4f}")
    
    avg_similarity = np.mean(similarities)
    diversity = 1 - avg_similarity
    
    print(f"\n{'='*60}")
    print(f"Average similarity: {avg_similarity:.4f}")
    print(f"Diversity score:    {diversity:.4f}")
    print(f"\n→ Higher diversity = more varied generations")
    
    if diversity > 0.4:
        print("→ Good diversity")
    elif diversity > 0.2:
        print("→ Moderate diversity")
    else:
        print("→ Low diversity (repetitive)")

measure_diversity()
```

## Learned Metrics

### Training Custom Metrics

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LearnedMetric(nn.Module):
    """
    Custom learned metric for text evaluation.
    
    Trained on human judgments to predict quality scores.
    """
    
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        
        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Regression head
        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output 0-1
        )
    
    def forward(self, reference, candidate):
        """
        Compute quality score.
        
        Args:
            reference: Reference text
            candidate: Candidate text
        
        Returns:
            Quality score (0-1)
        """
        # Tokenize
        ref_tokens = self.tokenizer(reference, return_tensors='pt', padding=True, truncation=True)
        cand_tokens = self.tokenizer(candidate, return_tensors='pt', padding=True, truncation=True)
        
        # Encode
        ref_output = self.encoder(**ref_tokens).last_hidden_state[:, 0, :]  # [CLS] token
        cand_output = self.encoder(**cand_tokens).last_hidden_state[:, 0, :]
        
        # Concatenate
        combined = torch.cat([ref_output, cand_output], dim=1)
        
        # Predict score
        score = self.regressor(combined)
        
        return score

# Example usage (training loop not shown)
def demonstrate_learned_metric():
    """Demonstrate learned metric concept."""
    
    print("Learned Metric Concept:\n")
    print("="*60)
    
    print("""
Training Process:

1. Collect data:
   • Pairs of (reference, candidate, human_score)
   • e.g., (ref="Good movie", cand="Excellent film", score=0.9)

2. Model architecture:
   • Encode reference and candidate with BERT
   • Concatenate representations
   • Regression head to predict score

3. Training objective:
   • Minimize MSE between predicted and human scores
   • Loss = (predicted_score - human_score)²

4. Inference:
   • Given (reference, candidate), predict quality score
   • Score correlates with human judgment

Benefits:
  ✓ Learns from actual human preferences
  ✓ Can capture complex patterns
  ✓ Task-specific (train for your domain)

Examples:
  • COMET: Trained on WMT human judgments
  • BLEURT: Trained on synthetic + human data
  • Custom metrics: Train on your domain
""")

demonstrate_learned_metric()
```

### COMET (Crosslingual Optimized Metric for Evaluation of Translation)

```python
# Note: Requires comet-ml package
# from comet import download_model, load_from_checkpoint

def comet_metric_example():
    """
    COMET is a learned metric for machine translation.
    Trained on human judgments from WMT.
    """
    
    print("COMET Metric:\n")
    print("="*60)
    
    print("""
COMET Features:

• Reference-based: Uses source, reference, and candidate
• QE (Quality Estimation): Source and candidate only
• Trained on DA (Direct Assessment) scores from humans
• State-of-the-art correlation with human judgment

Example usage:

from comet import download_model, load_from_checkpoint

# Download model
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

# Prepare data
data = [
    {
        "src": "Hola mundo",           # Source (Spanish)
        "mt": "Hello world",           # Machine translation
        "ref": "Hello world"           # Reference
    }
]

# Predict scores
scores = model.predict(data)
print(f"COMET score: {scores['scores'][0]:.4f}")

Scores typically range from 0 to 1, higher is better.

When to use:
  • Machine translation evaluation
  • Need human-aligned scores
  • Multilingual evaluation
""")

comet_metric_example()
```

## Comparing Neural vs Traditional Metrics

### Head-to-Head Comparison

```python
def comprehensive_metric_comparison():
    """Compare traditional and neural metrics side-by-side."""
    
    reference = "The doctor examined the patient carefully"
    
    test_cases = [
        ("Exact match", "The doctor examined the patient carefully"),
        ("Synonym", "The physician examined the patient carefully"),
        ("Paraphrase", "The patient was carefully examined by the doctor"),
        ("Semantic equiv", "A medical professional checked the sick person thoroughly"),
        ("Wrong fact", "The patient examined the doctor carefully"),
        ("Gibberish", "The doctor patient carefully examined"),
    ]
    
    print("Comprehensive Metric Comparison:\n")
    print("="*80)
    print(f"\nReference: {reference}\n")
    
    from nltk.translate.bleu_score import sentence_bleu
    from bert_score import score as bert_score
    
    ref_tokens = reference.split()
    
    results = []
    
    for desc, candidate in test_cases:
        cand_tokens = candidate.split()
        
        # Traditional: BLEU
        bleu = sentence_bleu([ref_tokens], cand_tokens)
        
        # Neural: BERTScore
        _, _, F1 = bert_score([candidate], [reference], lang='en', verbose=False)
        bertscore = F1.item()
        
        # Neural: Sentence embedding similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')
        ref_emb = model.encode(reference)
        cand_emb = model.encode(candidate)
        sem_sim = np.dot(ref_emb, cand_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(cand_emb))
        
        results.append({
            'desc': desc,
            'candidate': candidate,
            'bleu': bleu,
            'bertscore': bertscore,
            'sem_sim': sem_sim
        })
    
    # Print results
    for result in results:
        print(f"\n{result['desc']}:")
        print(f"  Candidate: {result['candidate']}")
        print(f"  BLEU:            {result['bleu']:.3f}")
        print(f"  BERTScore:       {result['bertscore']:.3f}")
        print(f"  Semantic Sim:    {result['sem_sim']:.3f}")
        
        # Analysis
        if result['bleu'] < 0.3 and result['bertscore'] > 0.7:
            print(f"  → Neural metrics catch semantic similarity that BLEU misses")
        elif result['bleu'] > 0.5 and result['bertscore'] < 0.7:
            print(f"  → BLEU high despite semantic issues (word order problem)")

comprehensive_metric_comparison()
```

### Correlation with Human Judgment

```python
def human_correlation_comparison():
    """Show correlation with human judgments."""
    
    print("Correlation with Human Judgment:\n")
    print("="*60)
    
    print("""
Based on WMT shared tasks and academic research:

Metric                Correlation (Kendall τ)    Correlation (Pearson ρ)
─────────────────────────────────────────────────────────────────────────
BLEU                  0.20 - 0.30                0.40 - 0.50
METEOR                0.30 - 0.35                0.50 - 0.60
BERTScore             0.45 - 0.55                0.70 - 0.75
BLEURT                0.50 - 0.60                0.75 - 0.80
COMET                 0.55 - 0.65                0.80 - 0.85
Human vs Human        0.60 - 0.70                0.85 - 0.90

Interpretation:
  • Higher correlation = better agreement with humans
  • Neural metrics (BERTScore, BLEURT, COMET) ~2x better than BLEU
  • Best learned metrics (COMET) approach human-human agreement
  • No metric is perfect - still gap to human judgment

When Neural Metrics Excel:
  ✓ Paraphrase evaluation
  ✓ Capturing semantic similarity
  ✓ Cross-lingual evaluation
  ✓ Open-ended generation

When Traditional Metrics Still Useful:
  ✓ Fast computation needed
  ✓ Interpretability important
  ✓ Exact match matters (e.g., named entities)
  ✓ Established baselines for comparison
""")

human_correlation_comparison()
```

## Practical Implementation

### Complete Evaluation Pipeline

```python
class ComprehensiveEvaluator:
    """
    Comprehensive evaluator combining multiple metrics.
    """
    
    def __init__(self):
        # Load models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def evaluate(self, reference, candidate):
        """
        Evaluate candidate against reference with multiple metrics.
        
        Args:
            reference: Reference text
            candidate: Candidate text
        
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # Traditional: BLEU
        from nltk.translate.bleu_score import sentence_bleu
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        results['bleu'] = sentence_bleu([ref_tokens], cand_tokens)
        
        # Traditional: ROUGE
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)
        results['rouge1'] = rouge_scores['rouge1'].fmeasure
        results['rouge2'] = rouge_scores['rouge2'].fmeasure
        results['rougeL'] = rouge_scores['rougeL'].fmeasure
        
        # Neural: BERTScore
        from bert_score import score as bert_score
        _, _, F1 = bert_score([candidate], [reference], lang='en', verbose=False)
        results['bertscore'] = F1.item()
        
        # Neural: Semantic similarity
        ref_emb = self.sentence_model.encode(reference)
        cand_emb = self.sentence_model.encode(candidate)
        results['semantic_sim'] = float(np.dot(ref_emb, cand_emb) / 
                                        (np.linalg.norm(ref_emb) * np.linalg.norm(cand_emb)))
        
        return results
    
    def print_results(self, results):
        """Print evaluation results nicely."""
        
        print("\nEvaluation Results:")
        print("="*50)
        
        print("\nTraditional Metrics:")
        print(f"  BLEU:      {results['bleu']:.4f}")
        print(f"  ROUGE-1:   {results['rouge1']:.4f}")
        print(f"  ROUGE-2:   {results['rouge2']:.4f}")
        print(f"  ROUGE-L:   {results['rougeL']:.4f}")
        
        print("\nNeural Metrics:")
        print(f"  BERTScore: {results['bertscore']:.4f}")
        print(f"  Sem. Sim.: {results['semantic_sim']:.4f}")
        
        print("\nOverall Assessment:")
        avg_traditional = (results['bleu'] + results['rouge1'] + results['rouge2'] + results['rougeL']) / 4
        avg_neural = (results['bertscore'] + results['semantic_sim']) / 2
        
        print(f"  Traditional avg: {avg_traditional:.4f}")
        print(f"  Neural avg:      {avg_neural:.4f}")
        
        if avg_neural > 0.8:
            print("  → Excellent semantic match")
        elif avg_neural > 0.65:
            print("  → Good semantic match")
        elif avg_neural > 0.5:
            print("  → Moderate match")
        else:
            print("  → Poor match")

# Example usage
evaluator = ComprehensiveEvaluator()

reference = "The doctor examined the patient carefully"
candidate = "A medical professional thoroughly checked the sick person"

results = evaluator.evaluate(reference, candidate)
evaluator.print_results(results)
```

### Batch Evaluation

```python
def batch_evaluation(references, candidates):
    """
    Evaluate multiple text pairs efficiently.
    
    Args:
        references: List of reference texts
        candidates: List of candidate texts
    """
    evaluator = ComprehensiveEvaluator()
    
    print("Batch Evaluation Results:\n")
    print("="*80)
    
    all_results = []
    
    for i, (ref, cand) in enumerate(zip(references, candidates), 1):
        print(f"\nPair {i}:")
        print(f"  Reference: {ref}")
        print(f"  Candidate: {cand}")
        
        results = evaluator.evaluate(ref, cand)
        all_results.append(results)
        
        print(f"  BLEU: {results['bleu']:.3f}, BERTScore: {results['bertscore']:.3f}")
    
    # Aggregate statistics
    print(f"\n{'='*80}")
    print("\nAggregate Statistics:")
    
    avg_bleu = np.mean([r['bleu'] for r in all_results])
    avg_bertscore = np.mean([r['bertscore'] for r in all_results])
    
    print(f"  Average BLEU:      {avg_bleu:.4f}")
    print(f"  Average BERTScore: {avg_bertscore:.4f}")

# Example
refs = [
    "The cat sat on the mat",
    "I love programming",
    "Machine learning is fascinating"
]

cands = [
    "A feline was on the rug",
    "I enjoy coding",
    "AI and ML are interesting"
]

batch_evaluation(refs, cands)
```

## Summary

**Neural Metrics Overview**:

```
Metric          Type        Strengths                   Weaknesses
─────────────────────────────────────────────────────────────────────────
BERTScore       Embedding   Semantic similarity         Slower than BLEU
                            Paraphrase detection        No factuality check

BLEURT          Learned     Human-aligned               Needs checkpoint
                            Best correlation            Slow

Semantic Sim    Embedding   Fast, simple                Sentence-level only
                            Good for filtering          Less fine-grained

COMET           Learned     Best for MT                 MT-specific
                            Multilingual                Needs source text
```

**When to Use Neural Metrics**:

```
Use Case                          Recommended Metric       Why
────────────────────────────────────────────────────────────────────────
Paraphrase detection             BERTScore, Semantic Sim  Captures meaning
Translation evaluation           COMET, BLEURT            Human-aligned
Summarization                    BERTScore, ROUGE         Semantic + recall
Open-ended generation            BERTScore, BLEURT        Flexible matching
Semantic search evaluation       Semantic Similarity      Fast, effective
Quick filtering                  Semantic Similarity      Speed
Production monitoring            BERTScore                Balance speed/quality
Research/benchmarking            BLEURT, COMET            Best correlation
```

**Key Advantages**:

1. **Semantic understanding**: Recognizes paraphrases and synonyms
2. **Context-aware**: Handles word sense disambiguation
3. **Better correlation**: Closer to human judgment (~0.70-0.80 vs ~0.40-0.50)
4. **Language-independent**: Works across languages with multilingual models
5. **Flexible**: Can fine-tune for specific domains

**Limitations**:

1. **Computational cost**: Slower than lexical metrics (BERT forward pass)
2. **No factuality**: High score doesn't mean factually correct
3. **Blackbox**: Less interpretable than word overlap
4. **Model dependency**: Quality depends on pretrained model
5. **Short text issues**: Less reliable for very short texts (1-2 words)

**Best Practices**:

```python
# 1. Use multiple metrics
def comprehensive_eval(ref, cand):
    return {
        'bleu': calculate_bleu(ref, cand),        # Lexical
        'bertscore': calculate_bertscore(ref, cand),  # Semantic
        'factual': check_factuality(ref, cand)    # Factual (if applicable)
    }

# 2. Choose right model for BERTScore
# Fast: distilbert-base-uncased
# Balanced: roberta-large
# Best: microsoft/deberta-xlarge-mnli

# 3. Enable IDF weighting for better rare word handling
P, R, F1 = score(cands, refs, idf=True)

# 4. Batch processing for speed
scores = model.encode(texts, batch_size=32)

# 5. Cache embeddings
embeddings_cache = {text: model.encode(text) for text in unique_texts}
```

**Trade-offs**:

```
Speed vs Quality:
  Fast:    Semantic similarity (embedding lookup)
  Medium:  BERTScore (BERT forward pass)
  Slow:    BLEURT (larger model + regression head)

Interpretability vs Accuracy:
  Interpretable: BLEU (count word overlaps)
  Less clear:    BERTScore (embedding similarities)
  Black-box:     BLEURT (learned weights)

Generality vs Task-specific:
  General:  BERTScore, Semantic Similarity
  Specific: COMET (MT), trained custom metrics
```

**Future Directions**:

- **LLM-as-judge**: Use GPT-4 to evaluate (even better correlation)
- **Multimodal metrics**: Evaluate text + images together
- **Factuality-aware**: Combine semantic + factual correctness
- **Explainable**: Highlight what makes scores high/low
- **Efficient**: Distilled models for faster inference

## Next Steps

- Study [LLM Evaluation Methods](llm-evaluation.md) for modern LLM evaluation approaches
- Learn [Human Evaluation](human-evaluation.md) techniques as the gold standard
- Explore [Benchmarks and Leaderboards](benchmarks.md) for standardized comparisons
- Review [Traditional Metrics](traditional-metrics.md) for foundational understanding
- Apply [Failure Analysis](failure-analysis.md) to understand metric limitations
- Check [Application Patterns](../application-patterns/) for production evaluation pipelines
