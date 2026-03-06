# Masked Models (BERT-style)

## Table of Contents

- [Introduction](#introduction)
- [Masked Language Modeling](#masked-language-modeling)
- [BERT Architecture](#bert-architecture)
- [Training Objectives](#training-objectives)
- [Bidirectional Context](#bidirectional-context)
- [Fine-tuning for Downstream Tasks](#fine-tuning-for-downstream-tasks)
- [BERT Variants](#bert-variants)
- [Extracting Representations](#extracting-representations)
- [Comparison with Autoregressive Models](#comparison-with-autoregressive-models)
- [Applications](#applications)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Masked language models** like BERT learn bidirectional representations by predicting masked tokens using context from both directions. This approach excels at understanding tasks like classification, question answering, and named entity recognition.

```
Masked Language Modeling:

Input:  The  [MASK] sat  on  the  mat
              ↓
Context: ← The    [MASK]    sat on the mat →
              ↓
Predict:     cat

Key difference from GPT:
  • GPT: Can only see left context (The)
  • BERT: Sees both left and right context (The ... sat on the mat)
```

**Key characteristics**:
- **Bidirectional**: Attends to both past and future tokens
- **Masked prediction**: Random tokens replaced with [MASK]
- **Better understanding**: Captures full context
- **Not natural for generation**: Mismatch between training and inference

This guide covers BERT-style masked language models and their applications.

## Masked Language Modeling

### The MLM Objective

**Masked Language Modeling (MLM)**: Randomly mask tokens and predict them

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(w_i | w_{\setminus i})$$

Where $w_{\setminus i}$ represents all tokens except position $i$.

```python
import torch
import torch.nn as nn
import numpy as np

def create_mlm_training_data(tokens, mask_prob=0.15, mask_token_id=103):
    """
    Create masked language modeling training data.
    
    Strategy:
      • 15% of tokens selected for prediction
      • Of those:
        - 80% replaced with [MASK]
        - 10% replaced with random token
        - 10% kept unchanged
    """
    masked_tokens = tokens.clone()
    labels = tokens.clone()
    
    # Probability matrix
    prob_matrix = torch.full(tokens.shape, mask_prob)
    
    # Don't mask special tokens ([CLS], [SEP], [PAD])
    special_tokens_mask = (tokens == 101) | (tokens == 102) | (tokens == 0)
    prob_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Select indices to mask
    masked_indices = torch.bernoulli(prob_matrix).bool()
    
    # Set labels to -100 for non-masked tokens (ignore in loss)
    labels[~masked_indices] = -100
    
    # 80% of time: replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & masked_indices
    masked_tokens[indices_replaced] = mask_token_id
    
    # 10% of time: replace with random token
    indices_random = torch.bernoulli(torch.full(tokens.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(len(vocab), tokens.shape, dtype=torch.long)
    masked_tokens[indices_random] = random_tokens[indices_random]
    
    # 10% of time: keep original token (indices not replaced and not random)
    
    return masked_tokens, labels

# Example
print("Masked Language Modeling Training Data:\n")

sentence = "The quick brown fox jumps over the lazy dog"
tokens = sentence.split()

print(f"Original: {sentence}")
print(f"\nMasking examples:")

examples = [
    ("The [MASK] brown fox jumps", "quick"),
    ("The quick brown [MASK] jumps", "fox"),
    ("[MASK] quick brown fox jumps", "The"),
    ("The quick brown fox [MASK]", "jumps")
]

for masked, target in examples:
    print(f"  Input:  {masked:35} → Predict: {target}")

print("\nMasking strategy:")
print("  • 80%: Replace with [MASK] token")
print("  • 10%: Replace with random token")
print("  • 10%: Keep original token")
print("\nWhy not always use [MASK]?")
print("  • Model would learn to rely on [MASK] presence")
print("  • At inference time, no [MASK] tokens exist")
print("  • Mixing strategies makes model more robust")
```

### Why MLM Works

```python
def explain_mlm_benefits():
    """Why masked language modeling is effective."""
    
    benefits = {
        'Bidirectional context': {
            'description': 'See both left and right context',
            'example': 'bank: "deposited money at the bank" vs "river bank"',
            'benefit': 'Better word sense disambiguation'
        },
        'Dense training signal': {
            'description': 'Multiple predictions per sequence',
            'example': 'Mask 15% of tokens → many training examples',
            'benefit': 'More efficient learning'
        },
        'Natural for understanding': {
            'description': 'Captures relationships in both directions',
            'example': 'Subject-verb agreement, coreference',
            'benefit': 'Better representations for NLU tasks'
        },
        'Self-supervised': {
            'description': 'No manual labeling needed',
            'example': 'Just raw text',
            'benefit': 'Can use massive datasets'
        }
    }
    
    print("Why Masked Language Modeling Works:\n")
    for benefit_name, info in benefits.items():
        print(f"{benefit_name}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

explain_mlm_benefits()
```

## BERT Architecture

### Model Structure

**BERT (Bidirectional Encoder Representations from Transformers)**: Stack of transformer encoders

```
Architecture:

Input: [CLS] The cat sat on mat [SEP]
         ↓    ↓   ↓   ↓  ↓  ↓    ↓
    Token Embeddings
         +
    Position Embeddings  
         +
    Segment Embeddings (for sentence pairs)
         ↓
    ═══════════════════════════════════
    ║   Transformer Encoder Block 1   ║
    ║   - Multi-Head Self-Attention   ║ ← Bidirectional!
    ║   - Feed-Forward Network        ║
    ║   - Layer Norm + Residual       ║
    ═══════════════════════════════════
         ↓
    ═══════════════════════════════════
    ║   Transformer Encoder Block 2   ║
    ═══════════════════════════════════
         ↓
         ... (N blocks)
         ↓
    ═══════════════════════════════════
    ║   Transformer Encoder Block N   ║
    ═══════════════════════════════════
         ↓
    Contextual representations
    [batch_size, seq_len, hidden_size]
```

### Implementation

```python
import torch.nn as nn
import torch.nn.functional as F

class BERTEmbeddings(nn.Module):
    """BERT-style embeddings: token + position + segment."""
    
    def __init__(self, vocab_size, hidden_size=768, max_position_embeddings=512, 
                 type_vocab_size=2, dropout=0.1):
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        
        # Position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Token type IDs (for sentence pairs)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Sum embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BERTEncoder(nn.Module):
    """BERT encoder: stack of transformer blocks."""
    
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12,
                 intermediate_size=3072, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None):
        # Apply each transformer layer
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        return hidden_states

class BERTModel(nn.Module):
    """Full BERT model."""
    
    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=12,
                 num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512):
        super().__init__()
        
        self.embeddings = BERTEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder = BERTEncoder(hidden_size, num_layers, num_attention_heads, intermediate_size)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Encode
        sequence_output = self.encoder(embedding_output, attention_mask)
        
        return sequence_output

# Create BERT model
bert_model = BERTModel(
    vocab_size=30522,
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)

print("BERT Model:")
print(f"  Parameters: {sum(p.numel() for p in bert_model.parameters()):,}")
print(f"  Architecture: Transformer Encoder")
print(f"  Hidden size: 768")
print(f"  Layers: 12")
print(f"  Attention heads: 12")

# Example forward pass
input_ids = torch.randint(0, 30522, (2, 10))  # [batch=2, seq_len=10]
output = bert_model(input_ids)

print(f"\nInput shape: {input_ids.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: Contextual representation for each token")
```

### Special Tokens

```python
def explain_bert_special_tokens():
    """BERT uses special tokens for specific purposes."""
    
    special_tokens = {
        '[CLS]': {
            'position': 'Start of sequence',
            'purpose': 'Aggregate sequence representation',
            'use': 'Classification tasks (use [CLS] embedding)',
            'example': '[CLS] This is a sentence [SEP]'
        },
        '[SEP]': {
            'position': 'End of sequence / between sentences',
            'purpose': 'Separate sentences in pairs',
            'use': 'Sentence pair tasks (NLI, QA)',
            'example': '[CLS] Sent A [SEP] Sent B [SEP]'
        },
        '[MASK]': {
            'position': 'Replaces tokens during training',
            'purpose': 'Masked language modeling',
            'use': 'Training only',
            'example': '[CLS] The [MASK] sat [SEP]'
        },
        '[PAD]': {
            'position': 'Padding shorter sequences',
            'purpose': 'Batch sequences of different lengths',
            'use': 'Ignored by attention mask',
            'example': '[CLS] Short [SEP] [PAD] [PAD]'
        },
        '[UNK]': {
            'position': 'Replaces out-of-vocabulary tokens',
            'purpose': 'Handle unknown words',
            'use': 'Rare or unseen words',
            'example': '[CLS] The xyzabc is [UNK] [SEP]'
        }
    }
    
    print("BERT Special Tokens:\n")
    for token, info in special_tokens.items():
        print(f"{token}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

explain_bert_special_tokens()
```

## Training Objectives

### Masked Language Modeling (MLM)

```python
class BERTForMaskedLM(nn.Module):
    """BERT with MLM head."""
    
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        
        self.bert = bert_model
        self.mlm_head = nn.Linear(bert_model.embeddings.token_embeddings.embedding_dim, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get BERT representations
        sequence_output = self.bert(input_ids, attention_mask)
        
        # MLM predictions
        prediction_scores = self.mlm_head(sequence_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(prediction_scores.view(-1, vocab_size), labels.view(-1))
        
        return loss, prediction_scores

print("Masked Language Modeling Training:")
print("  1. Randomly mask 15% of tokens")
print("  2. Pass through BERT encoder")
print("  3. Predict original tokens for masked positions")
print("  4. Loss: Cross-entropy on masked positions only")
print("\nExample:")
print("  Input:  [CLS] The [MASK] sat on the [MASK] [SEP]")
print("  Labels: [ignored] cat [ignored] mat")
print("  Loss computed only for [MASK] positions")
```

### Next Sentence Prediction (NSP)

```python
class BERTForPreTraining(nn.Module):
    """BERT with both MLM and NSP heads."""
    
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        
        self.bert = bert_model
        
        # MLM head
        self.mlm_head = nn.Linear(768, vocab_size)
        
        # NSP head (binary classification)
        self.nsp_head = nn.Linear(768, 2)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        
        # BERT encoding
        sequence_output = self.bert(input_ids, attention_mask, token_type_ids)
        
        # MLM: predict masked tokens
        mlm_scores = self.mlm_head(sequence_output)
        
        # NSP: use [CLS] token for sentence pair classification
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        nsp_scores = self.nsp_head(cls_output)  # [batch_size, 2]
        
        # Compute losses
        total_loss = 0
        
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_scores.view(-1, vocab_size), masked_lm_labels.view(-1))
            total_loss += mlm_loss
        
        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            nsp_loss = loss_fct(nsp_scores, next_sentence_label)
            total_loss += nsp_loss
        
        return total_loss, mlm_scores, nsp_scores

print("\nNext Sentence Prediction (NSP):")
print("  Task: Predict if sentence B follows sentence A")
print("\n  Example:")
print("    Positive: A='The cat sat.' B='It was tired.'")
print("    Negative: A='The cat sat.' B='France is a country.'")
print("\n  Training data:")
print("    50% real consecutive sentences (label=1)")
print("    50% random pairs (label=0)")
print("\n  Note: Later work (RoBERTa) found NSP not helpful")
```

### Pretraining Recipe

```python
def bert_pretraining_recipe():
    """BERT pretraining process."""
    
    print("BERT Pretraining Recipe:\n")
    
    print("1. Data:")
    print("   • BooksCorpus (800M words)")
    print("   • English Wikipedia (2,500M words)")
    print("   • Total: ~3.3B words\n")
    
    print("2. Objectives:")
    print("   • Masked Language Modeling (MLM)")
    print("   • Next Sentence Prediction (NSP)")
    print("   • Joint training on both\n")
    
    print("3. Training:")
    print("   • Batch size: 256 sequences")
    print("   • Sequence length: 512 tokens")
    print("   • Steps: 1M")
    print("   • Optimizer: Adam (lr=1e-4, warmup)")
    print("   • Hardware: 16 TPU chips, 4 days\n")
    
    print("4. Model sizes:")
    print("   • BERT-base: 12 layers, 768 hidden, 12 heads, 110M params")
    print("   • BERT-large: 24 layers, 1024 hidden, 16 heads, 340M params\n")
    
    print("5. Result:")
    print("   • Pre-trained model with rich contextual representations")
    print("   • Can be fine-tuned for downstream tasks")

bert_pretraining_recipe()
```

## Bidirectional Context

### Attention Pattern

```python
def compare_attention_patterns():
    """Compare GPT vs BERT attention."""
    
    print("Attention Patterns:\n")
    
    sequence = ["The", "cat", "sat", "on", "mat"]
    n = len(sequence)
    
    print("GPT (Causal/Autoregressive):")
    print("Position can attend to:\n")
    print("       ", "  ".join(f"{w:4}" for w in sequence))
    
    for i, word in enumerate(sequence):
        print(f"{word:6}", end=" ")
        for j in range(n):
            if j <= i:
                print(" ✓  ", end=" ")
            else:
                print(" ✗  ", end=" ")
        print()
    
    print("\nBERT (Bidirectional):")
    print("Position can attend to:\n")
    print("       ", "  ".join(f"{w:4}" for w in sequence))
    
    for i, word in enumerate(sequence):
        print(f"{word:6}", end=" ")
        for j in range(n):
            print(" ✓  ", end=" ")  # Can attend to all positions!
        print()
    
    print("\nKey difference:")
    print("  • GPT: Each position sees only previous positions")
    print("  • BERT: Each position sees ALL positions (bidirectional)")

compare_attention_patterns()
```

### Benefits for Understanding

```python
understanding_examples = {
    'Word sense disambiguation': {
        'sentence': 'I went to the bank to deposit money',
        'target': 'bank',
        'gpt_context': 'I went to the',
        'bert_context': 'I went to the ___ to deposit money',
        'benefit': 'BERT uses "deposit money" to disambiguate'
    },
    'Coreference resolution': {
        'sentence': 'The cat sat because it was tired',
        'target': 'it',
        'gpt_context': 'The cat sat because',
        'bert_context': 'The cat sat because ___ was tired',
        'benefit': 'BERT uses "was tired" to link "it" to "cat"'
    },
    'Subject-verb agreement': {
        'sentence': 'The list of items is long',
        'target': 'is',
        'gpt_context': 'The list of items',
        'bert_context': 'The list of items ___ long',
        'benefit': 'BERT sees "list" (singular) despite "items"'
    },
    'Semantic role labeling': {
        'sentence': 'John gave Mary a book',
        'target': 'gave',
        'gpt_context': 'John',
        'bert_context': 'John ___ Mary a book',
        'benefit': 'BERT sees all arguments (giver, receiver, object)'
    }
}

print("\nBidirectional Context Benefits:\n")
for task, info in understanding_examples.items():
    print(f"{task}:")
    print(f"  Sentence: {info['sentence']}")
    print(f"  Target: {info['target']}")
    print(f"  GPT context: {info['gpt_context']}")
    print(f"  BERT context: {info['bert_context']}")
    print(f"  Benefit: {info['benefit']}")
    print()
```

## Fine-tuning for Downstream Tasks

### Single Sentence Classification

```python
class BERTForSequenceClassification(nn.Module):
    """BERT fine-tuned for sequence classification."""
    
    def __init__(self, bert_model, num_labels):
        super().__init__()
        
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # BERT encoding
        sequence_output = self.bert(input_ids, attention_mask)
        
        # Use [CLS] token representation
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)
        
        # Classification
        logits = self.classifier(cls_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return loss, logits

print("Sequence Classification with BERT:")
print("  Task: Sentiment analysis, topic classification, etc.")
print("  Architecture:")
print("    Input → BERT → [CLS] embedding → Classifier → Label")
print("  Example (sentiment):")
print("    Input: 'This movie is amazing!'")
print("    [CLS] representation → Classifier → Positive")
```

### Token Classification

```python
class BERTForTokenClassification(nn.Module):
    """BERT for token-level classification (NER, POS tagging)."""
    
    def __init__(self, bert_model, num_labels):
        super().__init__()
        
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # BERT encoding
        sequence_output = self.bert(input_ids, attention_mask)
        
        # Classify each token
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        return loss, logits

print("\nToken Classification with BERT:")
print("  Task: NER, POS tagging, chunking")
print("  Architecture:")
print("    Input → BERT → Token embeddings → Classifier → Label per token")
print("  Example (NER):")
print("    Input:  'John lives in Paris'")
print("    Labels: PERSON   O    O  LOCATION")
```

### Question Answering

```python
class BERTForQuestionAnswering(nn.Module):
    """BERT for extractive question answering (SQuAD-style)."""
    
    def __init__(self, bert_model):
        super().__init__()
        
        self.bert = bert_model
        self.qa_outputs = nn.Linear(768, 2)  # Start and end positions
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None):
        # BERT encoding
        sequence_output = self.bert(input_ids, attention_mask, token_type_ids)
        
        # Predict start and end positions
        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch_size, seq_len]
        end_logits = end_logits.squeeze(-1)
        
        # Compute loss if positions provided
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        return total_loss, start_logits, end_logits

print("\nQuestion Answering with BERT:")
print("  Task: Extract answer span from passage")
print("  Input format:")
print("    [CLS] Question [SEP] Passage [SEP]")
print("  Architecture:")
print("    BERT → Predict start and end positions of answer")
print("  Example:")
print("    Q: 'Where is Paris?'")
print("    P: 'Paris is the capital of France.'")
print("    Answer: 'the capital of France' (positions 15-22)")
```

### Fine-tuning Best Practices

```python
finetuning_tips = {
    'Learning rate': {
        'recommendation': 'Small LR (2e-5 to 5e-5)',
        'reason': 'Pretrained weights should not change drastically'
    },
    'Batch size': {
        'recommendation': '16 or 32',
        'reason': 'Balance between speed and stability'
    },
    'Epochs': {
        'recommendation': '3-4 epochs',
        'reason': 'BERT overfits quickly on small datasets'
    },
    'Warmup': {
        'recommendation': 'Linear warmup (10% of steps)',
        'reason': 'Prevents destabilizing pretrained weights'
    },
    'Weight decay': {
        'recommendation': '0.01',
        'reason': 'Regularization to prevent overfitting'
    },
    'Layer-wise LR decay': {
        'recommendation': 'Lower LR for earlier layers',
        'reason': 'Earlier layers more general, later layers more task-specific'
    }
}

print("\nFine-tuning Best Practices:\n")
for aspect, info in finetuning_tips.items():
    print(f"{aspect}:")
    print(f"  Recommendation: {info['recommendation']}")
    print(f"  Reason: {info['reason']}")
    print()
```

## BERT Variants

### RoBERTa (Robustly Optimized BERT)

```python
roberta_improvements = {
    'Training data': 'More data (160GB vs 16GB for BERT)',
    'Training time': 'Longer training (500K steps vs 1M)',
    'Batch size': 'Larger batches (8K vs 256)',
    'Dynamic masking': 'Different masking pattern each epoch',
    'No NSP': 'Removed Next Sentence Prediction task',
    'Byte-level BPE': 'Better tokenization for rare words',
    'Result': 'Consistent improvements over BERT on many tasks'
}

print("RoBERTa Improvements over BERT:\n")
for improvement, description in roberta_improvements.items():
    print(f"  {improvement}: {description}")
```

### ALBERT (A Lite BERT)

```python
albert_innovations = {
    'Factorized embeddings': {
        'technique': 'Separate token embeddings from hidden size',
        'benefit': 'Reduce parameters: V×H → V×E + E×H',
        'example': '30K×1024 → 30K×128 + 128×1024'
    },
    'Cross-layer parameter sharing': {
        'technique': 'Share parameters across all layers',
        'benefit': '12× parameter reduction',
        'example': 'BERT-large: 334M → ALBERT: 18M params'
    },
    'Sentence order prediction': {
        'technique': 'Replace NSP with harder task',
        'benefit': 'Better inter-sentence modeling',
        'example': 'Predict if two sentences are in order or swapped'
    }
}

print("\nALBERT Innovations:\n")
for innovation, info in albert_innovations.items():
    print(f"{innovation}:")
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

### Other Variants

```python
bert_variants_table = {
    'DistilBERT': {
        'approach': 'Knowledge distillation',
        'size': '66M params (40% reduction)',
        'performance': '97% of BERT performance',
        'benefit': 'Faster inference, smaller memory'
    },
    'ELECTRA': {
        'approach': 'Discriminator instead of generator',
        'task': 'Detect replaced tokens (not masked)',
        'benefit': 'More efficient pretraining',
        'performance': 'Better than BERT with same compute'
    },
    'DeBERTa': {
        'approach': 'Disentangled attention',
        'innovation': 'Separate content and position',
        'benefit': 'Better positional awareness',
        'performance': 'SOTA on many benchmarks'
    },
    'SpanBERT': {
        'approach': 'Mask contiguous spans',
        'task': 'Predict entire spans, not just tokens',
        'benefit': 'Better for span-based tasks',
        'performance': 'Improves QA and coreference'
    }
}

print("\nOther BERT Variants:\n")
for variant, info in bert_variants_table.items():
    print(f"{variant}:")
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

## Extracting Representations

### Layer Selection

```python
def extract_bert_embeddings(text, model, tokenizer, layer=-1):
    """
    Extract embeddings from specific BERT layer.
    
    Different layers capture different information:
      • Lower layers: Syntax, POS tags
      • Middle layers: Semantics
      • Upper layers: Task-specific (if fine-tuned)
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    
    # Forward pass with output_hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get hidden states from all layers
    hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors
    
    # Extract specific layer
    layer_output = hidden_states[layer]  # [1, seq_len, hidden_size]
    
    return layer_output

print("Extracting BERT Representations:\n")
print("Layer selection guidelines:")
print("  • Layer -1 (last): Task-specific, general purpose")
print("  • Layer -2 (second-to-last): More general, good for transfer")
print("  • Layer 0-4: Syntactic information")
print("  • Layer 5-8: Semantic information")
print("  • Concatenate layers: Richer representation")
```

### Pooling Strategies

```python
def bert_pooling_strategies(sequence_output, attention_mask):
    """Different ways to pool BERT token embeddings into sentence embedding."""
    
    strategies = {}
    
    # 1. [CLS] token
    strategies['cls'] = sequence_output[:, 0, :]
    
    # 2. Mean pooling
    mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
    sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    strategies['mean'] = sum_embeddings / sum_mask
    
    # 3. Max pooling
    strategies['max'] = torch.max(sequence_output, dim=1)[0]
    
    # 4. Mean of last 4 layers (requires multiple layer outputs)
    # strategies['mean_last_4'] = ...
    
    return strategies

print("\nPooling Strategies:\n")
pooling_info = {
    '[CLS] token': 'Default, trained for sentence-level tasks',
    'Mean pooling': 'Average all token embeddings',
    'Max pooling': 'Take maximum across sequence',
    'Attention-weighted': 'Learn attention weights over tokens',
    'Mean last 4 layers': 'Average representations from multiple layers'
}

for strategy, description in pooling_info.items():
    print(f"  {strategy}: {description}")
```

## Comparison with Autoregressive Models

### Architectural Differences

```python
def compare_bert_gpt():
    """Compare BERT and GPT architectures."""
    
    comparison = {
        'Attention': {
            'BERT': 'Bidirectional (all-to-all)',
            'GPT': 'Causal (left-to-right only)'
        },
        'Training objective': {
            'BERT': 'Masked Language Modeling',
            'GPT': 'Next Token Prediction'
        },
        'Architecture': {
            'BERT': 'Encoder-only',
            'GPT': 'Decoder-only'
        },
        'Best for': {
            'BERT': 'Understanding tasks (classification, NER, QA)',
            'GPT': 'Generation tasks (completion, dialogue)'
        },
        'Context': {
            'BERT': 'Full sentence context',
            'GPT': 'Only previous tokens'
        },
        'Training-inference': {
            'BERT': 'Mismatch ([MASK] in training, not inference)',
            'GPT': 'Match (always predict next token)'
        }
    }
    
    print("BERT vs GPT Comparison:\n")
    print(f"{'Aspect':<25} {'BERT':<40} {'GPT':<40}")
    print("=" * 110)
    
    for aspect, values in comparison.items():
        print(f"{aspect:<25} {values['BERT']:<40} {values['GPT']:<40}")

compare_bert_gpt()
```

### When to Use Each

```python
use_case_recommendations = {
    'Use BERT when': [
        'Classification tasks (sentiment, topic, intent)',
        'Token classification (NER, POS tagging)',
        'Question answering (extractive)',
        'Sentence similarity and entailment',
        'Need bidirectional context',
        'Understanding > Generation'
    ],
    'Use GPT when': [
        'Text generation and completion',
        'Dialogue and chatbots',
        'Creative writing',
        'Code generation',
        'Zero-shot or few-shot learning',
        'Generation > Understanding'
    ],
    'Use both when': [
        'Complex systems (BERT for understanding, GPT for responses)',
        'Question answering (BERT to find, GPT to generate answer)',
        'Summarization with planning (BERT extracts, GPT generates)'
    ]
}

print("\nWhen to Use BERT vs GPT:\n")
for category, use_cases in use_case_recommendations.items():
    print(f"{category}:")
    for use_case in use_cases:
        print(f"  • {use_case}")
    print()
```

## Applications

### Text Classification

```python
print("BERT for Text Classification:\n")

classification_examples = {
    'Sentiment Analysis': {
        'task': 'Classify text as positive/negative/neutral',
        'example': "'This movie is amazing!' → Positive"
    },
    'Topic Classification': {
        'task': 'Categorize text by topic',
        'example': "'Python is a language' → Technology"
    },
    'Intent Detection': {
        'task': 'Identify user intent in chatbots',
        'example': "'Book a flight to Paris' → BookFlight"
    },
    'Spam Detection': {
        'task': 'Identify spam messages',
        'example': "'You won $1M!' → Spam"
    }
}

for app, info in classification_examples.items():
    print(f"{app}:")
    print(f"  Task: {info['task']}")
    print(f"  Example: {info['example']}")
    print()
```

### Named Entity Recognition

```python
print("BERT for Named Entity Recognition:\n")

ner_example = {
    'Input': 'Apple Inc. was founded by Steve Jobs in Cupertino',
    'Entities': [
        ('Apple Inc.', 'ORG'),
        ('Steve Jobs', 'PERSON'),
        ('Cupertino', 'LOC')
    ]
}

print(f"Input: {ner_example['Input']}\n")
print("Extracted entities:")
for entity, label in ner_example['Entities']:
    print(f"  '{entity}' → {label}")
```

### Question Answering

```python
print("\n\nBERT for Question Answering:\n")

qa_example = {
    'Question': 'When was BERT published?',
    'Context': 'BERT was published in 2018 by researchers at Google. It revolutionized NLP by introducing bidirectional pretraining.',
    'Answer': '2018',
    'Start': 23,  # Character position
    'End': 27
}

print(f"Question: {qa_example['Question']}")
print(f"Context: {qa_example['Context']}")
print(f"Answer: {qa_example['Answer']}")
print(f"Extracted from positions {qa_example['Start']}-{qa_example['End']}")
```

## Summary

**Key Concepts**:

1. **Masked Language Modeling** trains BERT to predict masked tokens using bidirectional context
2. **Bidirectional attention** allows each token to attend to all other tokens (past and future)
3. **BERT architecture** is encoder-only transformer with special tokens ([CLS], [SEP], [MASK])
4. **Pretraining objectives** include MLM and (originally) Next Sentence Prediction
5. **Fine-tuning** adapts pretrained BERT to specific tasks (classification, NER, QA)
6. **BERT variants** improve efficiency (DistilBERT, ALBERT) or performance (RoBERTa, DeBERTa)

**Architecture**:

```
Input → Token + Position + Segment Embeddings → 
Transformer Encoder Blocks (bidirectional attention) → 
Contextual Representations → Task-specific heads
```

**Training**:

- **Pretraining**: MLM + NSP on large corpus
- **Fine-tuning**: Task-specific head + small labeled dataset
- **Efficient transfer**: Pretrain once, fine-tune for many tasks

**BERT vs GPT**:

| Aspect | BERT | GPT |
|--------|------|-----|
| Attention | Bidirectional | Unidirectional (causal) |
| Training | Masked LM | Next token prediction |
| Architecture | Encoder | Decoder |
| Best for | Understanding | Generation |

**Advantages**:

- ✅ **Bidirectional context**: Better understanding of language
- ✅ **Strong representations**: Captures syntax, semantics, world knowledge
- ✅ **Transfer learning**: Excellent performance with little fine-tuning
- ✅ **Versatile**: Works for many NLU tasks
- ✅ **Efficient fine-tuning**: Small labeled datasets sufficient

**Limitations**:

- ❌ **Not for generation**: Training-inference mismatch
- ❌ **Fixed length**: Maximum 512 tokens
- ❌ **Computational cost**: Large models expensive to run
- ❌ **Masked tokens**: Mismatch between [MASK] in training and real text

**Applications**:

- Text classification (sentiment, topic, intent)
- Token classification (NER, POS tagging)
- Question answering (extractive)
- Sentence similarity and entailment
- Information extraction

## Next Steps

- Study [Encoder-Decoder Models](encoder-decoder-models.md) for sequence-to-sequence tasks
- Learn [Pretraining and Transfer Learning](pretraining-transfer.md) strategies in depth
- Apply to [Text Classification](../application_patterns/text-classification.md) tasks
- Explore [Named Entity Recognition](../application_patterns/named-entity-recognition.md) with BERT
- Study [Question Answering](../application_patterns/question-answering.md) systems
- Progress to [Large Language Models](../llm_concepts/large-language-models.md) for modern approaches
