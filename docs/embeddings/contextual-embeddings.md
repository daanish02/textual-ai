# Contextual Embeddings

## Table of Contents

- [Introduction](#introduction)
- [From Static to Contextual](#from-static-to-contextual)
- [ELMo (Embeddings from Language Models)](#elmo-embeddings-from-language-models)
- [BERT Embeddings](#bert-embeddings)
- [GPT Embeddings](#gpt-embeddings)
- [RoBERTa and Variants](#roberta-and-variants)
- [Extracting Embeddings from Transformers](#extracting-embeddings-from-transformers)
- [Layer-wise Embeddings](#layer-wise-embeddings)
- [Fine-tuning Strategies](#fine-tuning-strategies)
- [Comparison of Contextual Models](#comparison-of-contextual-models)
- [Applications](#applications)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Contextual embeddings** represent words differently based on their surrounding context, solving a key limitation of static embeddings like Word2Vec and GloVe.

```
Problem with static embeddings:

Word: "bank"
  - Static embedding: Same vector always

Examples:
  1. "I went to the bank to deposit money"     (financial institution)
  2. "The river bank was covered with flowers" (riverside)

With static embeddings: SAME representation
With contextual embeddings: DIFFERENT representations
```

**Key Innovation**: Instead of one vector per word type, generate different vectors based on context

$$\text{bank}_{\text{context1}} \neq \text{bank}_{\text{context2}}$$

**Historical evolution**:

- 2013: Word2Vec, GloVe → Static embeddings
- 2018: ELMo → First mainstream contextual embeddings
- 2018: BERT → Bidirectional context, massive impact
- 2019+: GPT-2, RoBERTa, ALBERT, etc. → Improvements and variants

This guide covers major contextual embedding models and how to extract/use their representations.

## From Static to Contextual

### The Polysemy Problem

**Polysemy**: Words with multiple meanings

```python
# Examples of polysemous words

examples = {
    'bank': [
        'financial institution',
        'river edge',
        'to tilt (airplane banks)'
    ],
    'mouse': [
        'small rodent',
        'computer input device'
    ],
    'play': [
        'to engage in activity',
        'theatrical performance',
        'to perform music'
    ],
    'address': [
        'location/residence',
        'to speak to',
        'to deal with (address a problem)'
    ]
}

for word, meanings in examples.items():
    print(f"\n'{word}' has {len(meanings)} meanings:")
    for i, meaning in enumerate(meanings, 1):
        print(f"  {i}. {meaning}")
```

### Static vs Contextual Comparison

```python
from gensim.downloader import load
from transformers import BertTokenizer, BertModel
import torch

# Load static embeddings (Word2Vec)
static_model = load('word2vec-google-news-300')

# Load contextual model (BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
contextual_model = BertModel.from_pretrained('bert-base-uncased')

# Test sentences
sentences = [
    "I deposited money at the bank",
    "We walked along the river bank"
]

# Static embedding - same for both!
if 'bank' in static_model:
    static_emb = static_model['bank']
    print("Static embedding for 'bank' (same in both contexts):")
    print(f"  Shape: {static_emb.shape}")
    print(f"  First 5 dims: {static_emb[:5]}")

# Contextual embeddings - different for each!
print("\nContextual embeddings for 'bank' (different per context):")

for sentence in sentences:
    # Tokenize
    inputs = tokenizer(sentence, return_tensors='pt')

    # Get embeddings
    with torch.no_grad():
        outputs = contextual_model(**inputs)
        embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

    # Find 'bank' token
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    bank_idx = tokens.index('bank')
    bank_embedding = embeddings[bank_idx]

    print(f"\n  Sentence: '{sentence}'")
    print(f"  Shape: {bank_embedding.shape}")
    print(f"  First 5 dims: {bank_embedding[:5].numpy()}")
```

### Why Contextual is Better

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compare_static_vs_contextual():
    """Compare static and contextual embeddings for disambiguation."""

    # Word pairs with different meanings
    test_cases = [
        {
            'word': 'bank',
            'sent1': 'I went to the bank',
            'sent2': 'The river bank was muddy',
            'expected': 'different meanings'
        },
        {
            'word': 'bank',
            'sent1': 'She works at the bank',
            'sent2': 'He deposited money at the bank',
            'expected': 'same meaning'
        }
    ]

    for case in test_cases:
        word = case['word']
        sent1 = case['sent1']
        sent2 = case['sent2']

        # Static: always same similarity (1.0 with itself)
        static_sim = 1.0

        # Contextual: different based on context
        # (Conceptual - would compute actual embeddings)
        # contextual_sim = compute_contextual_similarity(sent1, sent2, word)

        print(f"\nWord: '{word}'")
        print(f"  Sent 1: {sent1}")
        print(f"  Sent 2: {sent2}")
        print(f"  Expected: {case['expected']}")
        print(f"  Static similarity: {static_sim:.3f} (always same!)")
        # print(f"  Contextual similarity: {contextual_sim:.3f}")

compare_static_vs_contextual()
```

## ELMo (Embeddings from Language Models)

### Architecture Overview

**ELMo** (Peters et al., 2018) uses bidirectional LSTM language models to generate contextual embeddings.

```
Architecture:

Input: "The cat sat on the mat"

Forward LSTM:  [CLS] → the → cat → sat → on → the → mat → [SEP]
                        ↓     ↓     ↓     ↓     ↓     ↓
                      Layer 1 hidden states
                        ↓     ↓     ↓     ↓     ↓     ↓
                      Layer 2 hidden states

Backward LSTM: [CLS] ← the ← cat ← sat ← on ← the ← mat ← [SEP]
                        ↓     ↓     ↓     ↓     ↓     ↓
                      Layer 1 hidden states
                        ↓     ↓     ↓     ↓     ↓     ↓
                      Layer 2 hidden states

ELMo embedding = weighted sum of:
  - Character-based input
  - Forward LSTM layer 1
  - Forward LSTM layer 2
  - Backward LSTM layer 1
  - Backward LSTM layer 2
```

### Key Features

1. **Character-based input**: Handles OOV words
2. **Bidirectional**: Captures left and right context
3. **Deep**: Multiple LSTM layers
4. **Task-specific weighting**: Learn how to combine layers for each task

### Using ELMo with AllenNLP

```python
# ELMo with AllenNLP (conceptual example)

# from allennlp.modules.elmo import Elmo, batch_to_ids

# Load pre-trained ELMo
# options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)

def get_elmo_embeddings(sentences):
    """Get ELMo embeddings for sentences."""
    # Convert sentences to character IDs
    # character_ids = batch_to_ids(sentences)

    # Get embeddings
    # embeddings = elmo(character_ids)

    # embeddings['elmo_representations'] contains the weighted sum
    # Shape: (batch_size, seq_len, embedding_dim)

    return embeddings

# Example usage
sentences = [
    ["The", "cat", "sat"],
    ["I", "went", "to", "the", "bank"]
]

# embeddings = get_elmo_embeddings(sentences)
# print(f"ELMo embeddings shape: {embeddings['elmo_representations'][0].shape}")
```

### ELMo for Downstream Tasks

```python
def use_elmo_for_classification():
    """Use ELMo embeddings for text classification."""

    # Conceptual example

    # 1. Get ELMo embeddings for training data
    # train_embeddings = get_elmo_embeddings(train_sentences)

    # 2. Pool embeddings (e.g., mean pooling)
    # pooled = torch.mean(train_embeddings, dim=1)

    # 3. Train classifier on top
    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression()
    # clf.fit(pooled.numpy(), train_labels)

    # 4. Predict on test data
    # test_embeddings = get_elmo_embeddings(test_sentences)
    # test_pooled = torch.mean(test_embeddings, dim=1)
    # predictions = clf.predict(test_pooled.numpy())

    pass

# ELMo improved many NLP tasks in 2018:
# - Named Entity Recognition (NER)
# - Sentiment Analysis
# - Question Answering
# - Textual Entailment
```

### Advantages and Limitations

```python
elmo_properties = {
    'Advantages': [
        'Handles OOV words (character-based)',
        'Captures bidirectional context',
        'Deep representations (multiple layers)',
        'Improved many downstream tasks'
    ],
    'Limitations': [
        'Sequential computation (LSTMs, not parallelizable)',
        'Large memory footprint',
        'Slower than Word2Vec',
        'Not truly bidirectional (concatenated forward/backward)'
    ]
}

for category, items in elmo_properties.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  • {item}")
```

## BERT Embeddings

### BERT Architecture

**BERT** (Devlin et al., 2018) uses Transformer encoder with true bidirectional context.

```
Architecture:

Input: [CLS] The cat sat on mat [SEP]
         ↓    ↓   ↓   ↓   ↓  ↓    ↓
    Token Embeddings
         +
    Position Embeddings
         +
    Segment Embeddings
         ↓
    ═══════════════════════════════
    ║   Transformer Layer 1       ║  ← Self-attention: all words attend to all
    ═══════════════════════════════
         ↓
    ═══════════════════════════════
    ║   Transformer Layer 2       ║
    ═══════════════════════════════
         ↓
         ...
         ↓
    ═══════════════════════════════
    ║   Transformer Layer 12      ║  (BERT-base)
    ═══════════════════════════════
         ↓
    Contextual embeddings for each token
```

### Loading BERT

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Set to evaluation mode
model.eval()

print(f"BERT model: {model_name}")
print(f"  Hidden size: {model.config.hidden_size}")
print(f"  Number of layers: {model.config.num_hidden_layers}")
print(f"  Number of attention heads: {model.config.num_attention_heads}")
print(f"  Vocabulary size: {model.config.vocab_size}")
```

### Extracting BERT Embeddings

```python
def get_bert_embeddings(text, model, tokenizer, layer=-1):
    """
    Extract BERT embeddings for text.

    Args:
        text: Input text
        model: BERT model
        tokenizer: BERT tokenizer
        layer: Which layer to extract (-1 = last layer, -2 = second-to-last, etc.)

    Returns:
        Embeddings tensor of shape [seq_len, hidden_dim]
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract specified layer
    # outputs.hidden_states is tuple of (num_layers + 1) tensors
    # Each tensor is [batch_size, seq_len, hidden_dim]
    hidden_states = outputs.hidden_states
    embeddings = hidden_states[layer][0]  # [seq_len, hidden_dim]

    return embeddings, inputs

# Example
text = "The cat sat on the mat"
embeddings, inputs = get_bert_embeddings(text, model, tokenizer)

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

print(f"\nInput text: '{text}'")
print(f"Tokens: {tokens}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

### Word-level Contextual Embeddings

```python
def get_word_embedding(text, word, model, tokenizer):
    """Get contextual embedding for specific word in text."""
    # Get all embeddings
    embeddings, inputs = get_bert_embeddings(text, model, tokenizer)

    # Find word in tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Handle subword tokenization
    word_indices = []
    for i, token in enumerate(tokens):
        if token.startswith(word) or token.strip('#').startswith(word):
            word_indices.append(i)

    if word_indices:
        # Average subword embeddings if multiple
        word_emb = embeddings[word_indices].mean(dim=0)
        return word_emb, word_indices, tokens
    else:
        return None, [], tokens

# Test with polysemous word
sentences = [
    "I deposited money at the bank",
    "We walked along the river bank"
]

print("\nContextual embeddings for 'bank':\n")
for sentence in sentences:
    word_emb, indices, tokens = get_word_embedding(sentence, 'bank', model, tokenizer)

    if word_emb is not None:
        print(f"Sentence: '{sentence}'")
        print(f"  Token indices: {indices}")
        print(f"  Tokens: {[tokens[i] for i in indices]}")
        print(f"  Embedding shape: {word_emb.shape}")
        print(f"  First 5 dims: {word_emb[:5].numpy()}")
        print()
```

### Sentence-level Embeddings

```python
def get_sentence_embedding(text, model, tokenizer, pooling='cls'):
    """
    Get sentence-level embedding from BERT.

    Args:
        text: Input text
        model: BERT model
        tokenizer: BERT tokenizer
        pooling: Pooling strategy - 'cls', 'mean', or 'max'

    Returns:
        Sentence embedding vector
    """
    embeddings, inputs = get_bert_embeddings(text, model, tokenizer)

    if pooling == 'cls':
        # Use [CLS] token embedding (first token)
        sentence_emb = embeddings[0]

    elif pooling == 'mean':
        # Mean pooling over all tokens (excluding [CLS] and [SEP])
        # Create attention mask for actual tokens
        attention_mask = inputs['attention_mask'][0]
        mask = attention_mask[1:-1].unsqueeze(-1).float()  # Exclude [CLS] and [SEP]

        token_embeddings = embeddings[1:-1]  # Exclude [CLS] and [SEP]
        sentence_emb = (token_embeddings * mask).sum(dim=0) / mask.sum()

    elif pooling == 'max':
        # Max pooling over all tokens
        sentence_emb = embeddings[1:-1].max(dim=0)[0]  # Exclude [CLS] and [SEP]

    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")

    return sentence_emb

# Compare pooling strategies
text = "The cat sat on the mat"

for pooling in ['cls', 'mean', 'max']:
    emb = get_sentence_embedding(text, model, tokenizer, pooling=pooling)
    print(f"{pooling.upper()} pooling: shape={emb.shape}, first 5 dims={emb[:5].numpy()}")
```

### Comparing Contextual Similarity

```python
def contextual_similarity(text1, text2, word, model, tokenizer):
    """
    Compute similarity of word embeddings in two different contexts.
    """
    # Get word embeddings in each context
    emb1, _, _ = get_word_embedding(text1, word, model, tokenizer)
    emb2, _, _ = get_word_embedding(text2, word, model, tokenizer)

    if emb1 is not None and emb2 is not None:
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item()

        return similarity
    else:
        return None

# Test: same vs different meanings
test_pairs = [
    {
        'word': 'bank',
        'sent1': 'I deposited money at the bank',
        'sent2': 'She works at the bank',
        'expected': 'high (same meaning)'
    },
    {
        'word': 'bank',
        'sent1': 'I deposited money at the bank',
        'sent2': 'We walked along the river bank',
        'expected': 'lower (different meanings)'
    },
]

print("\nContextual similarity analysis:\n")
for pair in test_pairs:
    sim = contextual_similarity(
        pair['sent1'],
        pair['sent2'],
        pair['word'],
        model,
        tokenizer
    )

    print(f"Word: '{pair['word']}'")
    print(f"  Sent 1: {pair['sent1']}")
    print(f"  Sent 2: {pair['sent2']}")
    print(f"  Expected: {pair['expected']}")
    print(f"  Similarity: {sim:.3f}")
    print()
```

## GPT Embeddings

### GPT Architecture

**GPT** (Generative Pre-trained Transformer) is unidirectional (left-to-right) unlike BERT.

```
Architecture:

GPT (Causal/Autoregressive):
  Input: The cat sat on the mat

  The   →  [hidden state]  (sees only "The")
  cat   →  [hidden state]  (sees "The cat")
  sat   →  [hidden state]  (sees "The cat sat")
  on    →  [hidden state]  (sees "The cat sat on")
  the   →  [hidden state]  (sees "The cat sat on the")
  mat   →  [hidden state]  (sees "The cat sat on the mat")

Each token only attends to previous tokens (causal mask).

BERT (Bidirectional):
  Each token attends to ALL tokens (past and future)
```

### Using GPT-2 Embeddings

```python
from transformers import GPT2Tokenizer, GPT2Model

# Load GPT-2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2Model.from_pretrained('gpt2')
gpt2_model.eval()

# Add padding token (GPT-2 doesn't have one by default)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

print(f"GPT-2 model:")
print(f"  Hidden size: {gpt2_model.config.n_embd}")
print(f"  Number of layers: {gpt2_model.config.n_layer}")
print(f"  Vocabulary size: {gpt2_model.config.vocab_size}")
```

### Extracting GPT Embeddings

```python
def get_gpt_embeddings(text, model, tokenizer):
    """Extract GPT-2 embeddings."""
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Last layer hidden states
    embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

    return embeddings, inputs

# Example
text = "The cat sat on the mat"
gpt_embeddings, gpt_inputs = get_gpt_embeddings(text, gpt2_model, gpt2_tokenizer)

tokens = gpt2_tokenizer.convert_ids_to_tokens(gpt_inputs['input_ids'][0])

print(f"\nGPT-2 embeddings:")
print(f"  Input: '{text}'")
print(f"  Tokens: {tokens}")
print(f"  Shape: {gpt_embeddings.shape}")
```

### BERT vs GPT Comparison

```python
def compare_bert_gpt(text):
    """Compare BERT and GPT embeddings for same text."""

    # BERT
    bert_emb, _ = get_bert_embeddings(text, model, tokenizer)

    # GPT-2
    gpt_emb, _ = get_gpt_embeddings(text, gpt2_model, gpt2_tokenizer)

    print(f"Text: '{text}'\n")
    print(f"BERT:")
    print(f"  Shape: {bert_emb.shape}")
    print(f"  Hidden dim: {bert_emb.shape[1]}")
    print(f"  Bidirectional: Yes")

    print(f"\nGPT-2:")
    print(f"  Shape: {gpt_emb.shape}")
    print(f"  Hidden dim: {gpt_emb.shape[1]}")
    print(f"  Bidirectional: No (causal/left-to-right)")

    # Key difference
    print(f"\nKey difference:")
    print(f"  BERT: Each token sees ALL tokens (past + future)")
    print(f"  GPT: Each token sees only PREVIOUS tokens")

compare_bert_gpt("The quick brown fox")
```

### When to Use GPT vs BERT

```python
use_cases = {
    'BERT': {
        'best_for': [
            'Classification tasks',
            'Named Entity Recognition',
            'Question Answering',
            'Sentence pair tasks (NLI, paraphrase)',
            'When future context is available'
        ],
        'architecture': 'Encoder-only (bidirectional)'
    },
    'GPT': {
        'best_for': [
            'Text generation',
            'Language modeling',
            'Completion tasks',
            'Causal prediction',
            'When only past context available'
        ],
        'architecture': 'Decoder-only (unidirectional)'
    }
}

for model_type, info in use_cases.items():
    print(f"\n{model_type}:")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Best for:")
    for task in info['best_for']:
        print(f"    • {task}")
```

## RoBERTa and Variants

### RoBERTa Improvements

**RoBERTa** (Robustly Optimized BERT) improves on BERT:

1. **Training longer**: More data, more steps
2. **Larger batches**: Better gradient estimates
3. **Remove NSP**: Next Sentence Prediction task removed
4. **Dynamic masking**: Different masking each epoch
5. **Larger byte-level BPE**: Better tokenization

```python
from transformers import RobertaTokenizer, RobertaModel

# Load RoBERTa
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')
roberta_model.eval()

print("RoBERTa improvements over BERT:")
print("  • Trained on 10x more data")
print("  • Dynamic masking (different each epoch)")
print("  • Removed Next Sentence Prediction (NSP)")
print("  • Byte-level BPE tokenization")
print("  • Trained longer with larger batches")
```

### Using RoBERTa

```python
def get_roberta_embeddings(text):
    """Get RoBERTa embeddings."""
    inputs = roberta_tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = roberta_model(**inputs)

    embeddings = outputs.last_hidden_state[0]
    return embeddings

# Example
text = "RoBERTa is an optimized version of BERT"
roberta_emb = get_roberta_embeddings(text)

print(f"\nRoBERTa embeddings:")
print(f"  Text: '{text}'")
print(f"  Shape: {roberta_emb.shape}")
```

### Other BERT Variants

```python
bert_variants = {
    'BERT': {
        'size': '110M (base), 340M (large)',
        'key_feature': 'Original bidirectional transformer',
        'year': 2018
    },
    'RoBERTa': {
        'size': '125M (base), 355M (large)',
        'key_feature': 'Optimized BERT training',
        'year': 2019
    },
    'ALBERT': {
        'size': '12M (base), 235M (xxlarge)',
        'key_feature': 'Parameter sharing for efficiency',
        'year': 2019
    },
    'DistilBERT': {
        'size': '66M',
        'key_feature': 'Distilled BERT (60% smaller, 95% performance)',
        'year': 2019
    },
    'ELECTRA': {
        'size': '110M (base), 335M (large)',
        'key_feature': 'Discriminator instead of MLM',
        'year': 2020
    },
    'DeBERTa': {
        'size': '134M (base), 1.5B (large)',
        'key_feature': 'Disentangled attention',
        'year': 2020
    }
}

print("BERT Variants:\n")
for name, info in bert_variants.items():
    print(f"{name} ({info['year']}):")
    print(f"  Size: {info['size']}")
    print(f"  Key Feature: {info['key_feature']}")
    print()
```

## Extracting Embeddings from Transformers

### Different Extraction Strategies

```python
def extract_embeddings_comprehensive(text, model, tokenizer):
    """
    Extract embeddings using various strategies.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True)

    # Get all hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Tuple of tensors
    attention_mask = inputs['attention_mask'][0]

    strategies = {}

    # 1. Last layer [CLS]
    strategies['last_cls'] = hidden_states[-1][0][0]

    # 2. Last layer mean pooling
    last_layer = hidden_states[-1][0]
    mask = attention_mask.unsqueeze(-1).float()
    strategies['last_mean'] = (last_layer * mask).sum(dim=0) / mask.sum()

    # 3. Second-to-last layer [CLS]
    strategies['second_last_cls'] = hidden_states[-2][0][0]

    # 4. Concatenate last 4 layers [CLS]
    last_four = torch.cat([hidden_states[i][0][0] for i in range(-4, 0)])
    strategies['concat_4_cls'] = last_four

    # 5. Sum last 4 layers [CLS]
    sum_four = torch.stack([hidden_states[i][0][0] for i in range(-4, 0)]).sum(dim=0)
    strategies['sum_4_cls'] = sum_four

    return strategies

# Test
text = "The cat sat on the mat"
embeddings = extract_embeddings_comprehensive(text, model, tokenizer)

print("Embedding extraction strategies:\n")
for name, emb in embeddings.items():
    print(f"{name}:")
    print(f"  Shape: {emb.shape}")
    print(f"  First 3 dims: {emb[:3].numpy()}")
    print()
```

### Choosing the Right Layer

```python
def analyze_layer_embeddings(text, model, tokenizer):
    """Analyze embeddings from different layers."""
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states)

    print(f"Model has {num_layers} layers (including embedding layer)\n")

    # Get [CLS] embedding from each layer
    for i, layer_output in enumerate(hidden_states):
        cls_emb = layer_output[0][0]  # [CLS] token
        print(f"Layer {i}:")
        print(f"  Shape: {cls_emb.shape}")
        print(f"  Norm: {cls_emb.norm().item():.3f}")
        print(f"  Mean: {cls_emb.mean().item():.3f}")
        print(f"  Std: {cls_emb.std().item():.3f}")
        print()

text = "Machine learning is fascinating"
analyze_layer_embeddings(text, model, tokenizer)
```

### Empirical Layer Guidelines

```python
layer_guidelines = {
    'Layer 0 (Embedding)': {
        'captures': 'Token identity, position',
        'use_for': 'Rarely used directly'
    },
    'Early Layers (1-3)': {
        'captures': 'Syntax, POS tags, basic structure',
        'use_for': 'Syntactic tasks'
    },
    'Middle Layers (4-8)': {
        'captures': 'Phrase structure, dependencies',
        'use_for': 'Most NLU tasks'
    },
    'Late Layers (9-12)': {
        'captures': 'Semantic, task-specific',
        'use_for': 'Semantic similarity, classification'
    },
    'Last Layer': {
        'captures': 'Task-specific (especially if fine-tuned)',
        'use_for': 'Default choice for most tasks'
    },
    'Second-to-Last': {
        'captures': 'General semantic (less task-specific)',
        'use_for': 'Transfer learning, when last layer is too specific'
    }
}

print("Layer Selection Guidelines:\n")
for layer, info in layer_guidelines.items():
    print(f"{layer}:")
    print(f"  Captures: {info['captures']}")
    print(f"  Use for: {info['use_for']}")
    print()
```

## Layer-wise Embeddings

### Probing Layer Representations

```python
def probe_layer_capabilities(model, tokenizer, task='similarity'):
    """
    Probe which layers capture which linguistic properties.
    """
    # Example: Test syntactic vs semantic similarity

    # Syntactically similar, semantically different
    synt_pair = [
        "The cat sat on the mat",
        "The dog sat on the rug"
    ]

    # Semantically similar, syntactically different
    sem_pair = [
        "The cat sat on the mat",
        "A feline rested on a carpet"
    ]

    # Get embeddings from all layers
    results = {'syntactic': [], 'semantic': []}

    for pair, pair_type in [(synt_pair, 'syntactic'), (sem_pair, 'semantic')]:
        # Get embeddings for both sentences
        inputs1 = tokenizer(pair[0], return_tensors='pt')
        inputs2 = tokenizer(pair[1], return_tensors='pt')

        with torch.no_grad():
            outputs1 = model(**inputs1, output_hidden_states=True)
            outputs2 = model(**inputs2, output_hidden_states=True)

        # Compare [CLS] embeddings from each layer
        for layer_idx in range(len(outputs1.hidden_states)):
            emb1 = outputs1.hidden_states[layer_idx][0][0]
            emb2 = outputs2.hidden_states[layer_idx][0][0]

            sim = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0),
                emb2.unsqueeze(0)
            ).item()

            results[pair_type].append(sim)

    return results

# Probe
results = probe_layer_capabilities(model, tokenizer)

print("Layer-wise similarity analysis:\n")
print("Layer | Syntactic Pair | Semantic Pair")
print("------|----------------|---------------")
for layer in range(len(results['syntactic'])):
    syn_sim = results['syntactic'][layer]
    sem_sim = results['semantic'][layer]
    print(f"  {layer:2d}  |     {syn_sim:.3f}      |     {sem_sim:.3f}")

print("\nObservation:")
print("  • Early layers: Higher syntactic similarity")
print("  • Later layers: Higher semantic similarity")
```

### Weighted Layer Combination

```python
def weighted_layer_combination(text, model, tokenizer, weights=None):
    """
    Combine multiple layers with learned or specified weights.
    """
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states)

    if weights is None:
        # Equal weights
        weights = torch.ones(num_layers) / num_layers
    else:
        weights = torch.tensor(weights)
        weights = weights / weights.sum()  # Normalize

    # Weighted sum of [CLS] embeddings
    weighted_emb = sum(
        w * hidden_states[i][0][0]
        for i, w in enumerate(weights)
    )

    return weighted_emb, weights

# Example: More weight to later layers
custom_weights = [0.01] * 9 + [0.3, 0.3, 0.31]  # Emphasize last 3 layers

text = "The cat sat on the mat"
emb, weights = weighted_layer_combination(text, model, tokenizer, custom_weights)

print("Weighted layer combination:")
print(f"  Weights: {weights.numpy()}")
print(f"  Resulting embedding shape: {emb.shape}")
```

## Fine-tuning Strategies

### Feature-based vs Fine-tuning

```python
approaches = {
    'Feature-based (like ELMo)': {
        'method': 'Freeze pre-trained model, use embeddings as features',
        'pros': [
            'Faster training',
            'Less memory',
            'Can combine with other features'
        ],
        'cons': [
            'Less adaptation to task',
            'May not capture task-specific patterns'
        ]
    },
    'Fine-tuning (like BERT)': {
        'method': 'Update all/some parameters during training',
        'pros': [
            'Better task adaptation',
            'Often better performance',
            'End-to-end learning'
        ],
        'cons': [
            'Slower training',
            'More memory required',
            'Risk of overfitting on small datasets'
        ]
    }
}

for approach, info in approaches.items():
    print(f"\n{approach}:")
    print(f"  Method: {info['method']}")
    print(f"  Pros:")
    for pro in info['pros']:
        print(f"    + {pro}")
    print(f"  Cons:")
    for con in info['cons']:
        print(f"    - {con}")
```

### Gradual Unfreezing

```python
def gradual_unfreezing_example():
    """
    Gradually unfreeze layers during fine-tuning.
    """
    # Conceptual example

    # Stage 1: Train only classifier (freeze encoder)
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    # Train classifier...

    # Stage 2: Unfreeze top layers
    # for param in model.bert.encoder.layer[-3:].parameters():
    #     param.requires_grad = True
    # Continue training...

    # Stage 3: Unfreeze all layers
    # for param in model.bert.parameters():
    #     param.requires_grad = True
    # Final training...

    stages = [
        "Stage 1: Freeze encoder, train only classifier",
        "Stage 2: Unfreeze top 3 layers",
        "Stage 3: Unfreeze all layers with low learning rate"
    ]

    print("Gradual Unfreezing Strategy:\n")
    for i, stage in enumerate(stages, 1):
        print(f"{i}. {stage}")

gradual_unfreezing_example()
```

### Layer-specific Learning Rates

```python
def discriminative_fine_tuning():
    """
    Use different learning rates for different layers.
    """
    # Conceptual example

    # Lower layers: smaller learning rate (more general)
    # Higher layers: larger learning rate (more task-specific)

    # from torch.optim import Adam
    #
    # optimizer = Adam([
    #     {'params': model.bert.embeddings.parameters(), 'lr': 1e-5},
    #     {'params': model.bert.encoder.layer[:6].parameters(), 'lr': 2e-5},
    #     {'params': model.bert.encoder.layer[6:].parameters(), 'lr': 3e-5},
    #     {'params': model.classifier.parameters(), 'lr': 5e-5}
    # ])

    print("Discriminative Fine-tuning:")
    print("  Embeddings:      lr = 1e-5 (lowest)")
    print("  Lower encoder:   lr = 2e-5")
    print("  Upper encoder:   lr = 3e-5")
    print("  Classifier:      lr = 5e-5 (highest)")
    print("\nRationale: Lower layers are more general, need less updating")

discriminative_fine_tuning()
```

## Comparison of Contextual Models

### Performance Comparison

```python
model_comparison = {
    'ELMo': {
        'architecture': 'Bi-LSTM',
        'parameters': '93M',
        'context': 'Bidirectional (concatenated)',
        'speed': 'Slow',
        'typical_performance': '85-88% (GLUE)'
    },
    'BERT-base': {
        'architecture': 'Transformer Encoder',
        'parameters': '110M',
        'context': 'Bidirectional (true)',
        'speed': 'Medium',
        'typical_performance': '87-89% (GLUE)'
    },
    'BERT-large': {
        'architecture': 'Transformer Encoder',
        'parameters': '340M',
        'context': 'Bidirectional',
        'speed': 'Slow',
        'typical_performance': '89-91% (GLUE)'
    },
    'RoBERTa-base': {
        'architecture': 'Transformer Encoder',
        'parameters': '125M',
        'context': 'Bidirectional',
        'speed': 'Medium',
        'typical_performance': '88-90% (GLUE)'
    },
    'GPT-2': {
        'architecture': 'Transformer Decoder',
        'parameters': '117M-1.5B',
        'context': 'Unidirectional (causal)',
        'speed': 'Fast-Medium',
        'typical_performance': 'N/A (generative)'
    },
}

print("Model Comparison:\n")
print(f"{'Model':<15} {'Arch':<20} {'Params':<10} {'Context':<25} {'Speed':<10}")
print("-" * 85)
for model_name, info in model_comparison.items():
    print(f"{model_name:<15} {info['architecture']:<20} {info['parameters']:<10} "
          f"{info['context']:<25} {info['speed']:<10}")
```

### Use Case Recommendations

```python
use_case_guide = {
    'Text Classification': {
        'recommended': ['BERT', 'RoBERTa', 'DistilBERT'],
        'rationale': 'Bidirectional context, good [CLS] representation'
    },
    'Named Entity Recognition': {
        'recommended': ['BERT', 'RoBERTa'],
        'rationale': 'Token-level predictions, bidirectional context'
    },
    'Question Answering': {
        'recommended': ['BERT', 'RoBERTa', 'ALBERT'],
        'rationale': 'Span prediction, passage understanding'
    },
    'Semantic Similarity': {
        'recommended': ['Sentence-BERT', 'RoBERTa'],
        'rationale': 'Good sentence representations'
    },
    'Text Generation': {
        'recommended': ['GPT-2', 'GPT-3'],
        'rationale': 'Autoregressive, causal attention'
    },
    'Low-resource Deployment': {
        'recommended': ['DistilBERT', 'ALBERT', 'MobileBERT'],
        'rationale': 'Smaller, faster, efficient'
    }
}

print("\nUse Case Recommendations:\n")
for task, info in use_case_guide.items():
    print(f"{task}:")
    print(f"  Recommended: {', '.join(info['recommended'])}")
    print(f"  Rationale: {info['rationale']}")
    print()
```

## Applications

### Word Sense Disambiguation

```python
def word_sense_disambiguation(word, context, model, tokenizer):
    """
    Disambiguate word meaning using contextual embeddings.
    """
    # Get contextual embedding
    word_emb, indices, tokens = get_word_embedding(context, word, model, tokenizer)

    # Define sense prototypes (would typically be learned)
    sense_examples = {
        'bank_financial': "I deposited money at the bank",
        'bank_river': "We walked along the river bank"
    }

    # Get embeddings for sense examples
    sense_embeddings = {}
    for sense, example in sense_examples.items():
        sense_emb, _, _ = get_word_embedding(example, word, model, tokenizer)
        if sense_emb is not None:
            sense_embeddings[sense] = sense_emb

    # Compare with each sense
    if word_emb is not None:
        similarities = {}
        for sense, sense_emb in sense_embeddings.items():
            sim = torch.nn.functional.cosine_similarity(
                word_emb.unsqueeze(0),
                sense_emb.unsqueeze(0)
            ).item()
            similarities[sense] = sim

        # Predict sense
        predicted_sense = max(similarities, key=similarities.get)

        return predicted_sense, similarities

    return None, {}

# Test
test_sentences = [
    "I need to go to the bank to withdraw cash",
    "The boat was moored near the bank"
]

print("Word Sense Disambiguation:\n")
for sentence in test_sentences:
    sense, similarities = word_sense_disambiguation('bank', sentence, model, tokenizer)

    print(f"Sentence: '{sentence}'")
    print(f"  Predicted sense: {sense}")
    print(f"  Similarities: {similarities}")
    print()
```

### Contextualized Search

```python
def contextualized_search(query_context, candidate_contexts, model, tokenizer):
    """
    Search for similar contexts using contextual embeddings.
    """
    # Extract query embedding (mean pool)
    query_emb = get_sentence_embedding(query_context, model, tokenizer, pooling='mean')

    # Extract candidate embeddings
    candidate_embs = [
        get_sentence_embedding(ctx, model, tokenizer, pooling='mean')
        for ctx in candidate_contexts
    ]

    # Compute similarities
    similarities = [
        torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0),
            cand_emb.unsqueeze(0)
        ).item()
        for cand_emb in candidate_embs
    ]

    # Rank
    ranked = sorted(
        zip(candidate_contexts, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked

# Example
query = "Methods for training deep neural networks"
candidates = [
    "Backpropagation is used to train neural networks",
    "The cat sat on the mat",
    "Gradient descent optimizes model parameters",
    "Pizza is my favorite food"
]

results = contextualized_search(query, candidates, model, tokenizer)

print(f"Query: '{query}'\n")
print("Ranked results:")
for i, (context, score) in enumerate(results, 1):
    print(f"{i}. (score={score:.3f}) {context}")
```

## Summary

**Key Concepts**:

1. **Contextual embeddings** generate different vectors for same word in different contexts
2. **ELMo** uses bidirectional LSTMs to create context-dependent representations
3. **BERT** uses Transformer encoders for true bidirectional context via self-attention
4. **GPT** uses causal (unidirectional) Transformers for left-to-right generation
5. **Extraction strategies** vary: [CLS] token, mean pooling, max pooling, layer combinations
6. **Layer analysis** shows different layers capture different linguistic properties

**Model Comparison**:

| Model   | Context            | Architecture        | Parameters | Best For       |
| ------- | ------------------ | ------------------- | ---------- | -------------- |
| ELMo    | Bi-LSTM concat     | LSTM                | 93M        | OOV handling   |
| BERT    | True bidirectional | Transformer Encoder | 110M-340M  | NLU tasks      |
| GPT     | Unidirectional     | Transformer Decoder | 117M-1.5B  | Generation     |
| RoBERTa | Bidirectional      | Transformer Encoder | 125M-355M  | Optimized BERT |

**Key Advantages**:

- ✅ **Handles polysemy**: Different meanings → different embeddings
- ✅ **Captures context**: Word order and dependencies preserved
- ✅ **Transfer learning**: Pre-train once, fine-tune for many tasks
- ✅ **State-of-the-art**: Dramatic improvements on benchmarks
- ✅ **Flexible extraction**: Multiple layers, pooling strategies

**Limitations**:

- ❌ **Computational cost**: Slower than static embeddings
- ❌ **Memory intensive**: Large models (millions of parameters)
- ❌ **Fixed context window**: Limited sequence length (512 tokens)
- ❌ **Need fine-tuning**: Best results require task-specific training
- ❌ **Black box**: Hard to interpret what's learned

**Best Practices**:

1. **Choose right model**: BERT for NLU, GPT for generation
2. **Extract from appropriate layer**: Last or second-to-last for most tasks
3. **Use proper pooling**: [CLS] for sentences, mean/max for tokens
4. **Fine-tune when possible**: Adapt to your domain/task
5. **Consider efficiency**: DistilBERT/ALBERT for production
6. **Batch processing**: Much faster than one-at-a-time

## Next Steps

- Explore [Embedding Spaces](embedding-spaces.md) to understand geometric properties and visualization
- Study [Language Models](../language_models/transformer-architectures.md) for deeper understanding of Transformers
- Learn [Fine-tuning Techniques](../llm_concepts/transfer-learning.md) for adapting pre-trained models
- Apply to [Text Classification](../application_patterns/text-classification.md) with contextual embeddings
- Implement [Named Entity Recognition](../application_patterns/named-entity-recognition.md) with BERT
- Progress to [Modern LLMs](../llm_concepts/large-language-models.md) like GPT-3, LLaMA
