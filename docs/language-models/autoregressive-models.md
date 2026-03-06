# Autoregressive Models (GPT-style)

## Table of Contents

- [Introduction](#introduction)
- [Causal Language Modeling](#causal-language-modeling)
- [GPT Architecture](#gpt-architecture)
- [Attention Mechanisms](#attention-mechanisms)
- [Training Autoregressive Models](#training-autoregressive-models)
- [Text Generation Strategies](#text-generation-strategies)
- [Sampling Methods](#sampling-methods)
- [Beam Search](#beam-search)
- [Control and Conditioning](#control-and-conditioning)
- [Scaling and Variants](#scaling-and-variants)
- [Evaluation](#evaluation)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Autoregressive models** generate text one token at a time, using previously generated tokens as context. They're the foundation of modern text generation systems like GPT, GPT-2, GPT-3, and GPT-4.

```
Autoregressive generation:

Start: "The cat"

Step 1: "The cat" → Model predicts → "sat"
Step 2: "The cat sat" → Model predicts → "on"
Step 3: "The cat sat on" → Model predicts → "the"
Step 4: "The cat sat on the" → Model predicts → "mat"

Result: "The cat sat on the mat"
```

**Key characteristics**:
- **Unidirectional**: Left-to-right (or right-to-left)
- **Causal attention**: Each token attends only to previous tokens
- **Natural generation**: Matches how humans produce text
- **Flexible length**: Can generate variable-length outputs

This guide covers GPT-style autoregressive models from architecture to generation strategies.

## Causal Language Modeling

### The Autoregressive Property

**Autoregressive**: Output at time $t$ depends only on inputs at times $< t$

$$P(w_1, w_2, ..., w_n) = \prod_{t=1}^{n} P(w_t | w_1, ..., w_{t-1})$$

```python
import torch
import torch.nn as nn
import numpy as np

def autoregressive_probability(model, sequence):
    """
    Compute probability of sequence using autoregressive decomposition.
    
    P(w1, w2, w3) = P(w1) × P(w2|w1) × P(w3|w1,w2)
    """
    log_prob = 0.0
    
    for t in range(len(sequence)):
        # Context: all tokens before position t
        context = sequence[:t+1]
        
        # Get model prediction
        logits = model(context)
        probs = torch.softmax(logits[-1], dim=-1)  # Prob dist at position t
        
        # Probability of actual token
        if t < len(sequence) - 1:
            next_token = sequence[t+1]
            log_prob += torch.log(probs[next_token])
    
    return torch.exp(log_prob)

print("Autoregressive Factorization:")
print("  P(The cat sat) = P(The) × P(cat|The) × P(sat|The cat)")
print("\nKey insight: Break down joint probability into conditional probabilities")
```

### Causal Masking

**Causal mask**: Prevents attention to future positions

```python
def create_causal_mask(seq_len):
    """
    Create causal attention mask.
    
    Lower triangular matrix: position i can only attend to positions ≤ i
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    return mask

# Example
seq_len = 5
mask = create_causal_mask(seq_len)

print("Causal Attention Mask (seq_len=5):\n")
print("Position can attend to:")
print("     0  1  2  3  4")
for i in range(seq_len):
    print(f"{i}: ", end="")
    for j in range(seq_len):
        print(f" {'✓' if mask[i, j] else '✗'} ", end="")
    print()

print("\nVisualization:")
print("""
Position 0: ✓                (only itself)
Position 1: ✓ ✓              (0, 1)
Position 2: ✓ ✓ ✓            (0, 1, 2)
Position 3: ✓ ✓ ✓ ✓          (0, 1, 2, 3)
Position 4: ✓ ✓ ✓ ✓ ✓        (0, 1, 2, 3, 4)
""")
```

### Training Objective

```python
def autoregressive_loss(model, input_ids):
    """
    Compute autoregressive language modeling loss.
    
    For sequence [w1, w2, w3, w4]:
    - Input to model: [w1, w2, w3, w4]
    - Predict: [w2, w3, w4, w5]
    - Loss: Cross-entropy between predictions and targets
    """
    # Forward pass
    logits = model(input_ids)  # [batch_size, seq_len, vocab_size]
    
    # Shift targets (predict next token)
    shift_logits = logits[:, :-1, :].contiguous()  # Remove last prediction
    shift_labels = input_ids[:, 1:].contiguous()    # Remove first token
    
    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss

print("Autoregressive Training:")
print("  Input:  [w1  w2  w3  w4 ]")
print("  Target: [    w2  w3  w4  w5]")
print("           └───┴───┴───┴───┘")
print("  Predict next token at each position")
```

## GPT Architecture

### Model Overview

**GPT (Generative Pre-trained Transformer)**: Decoder-only transformer with causal attention

```
Architecture stack:

Input tokens
     ↓
Token Embedding + Position Embedding
     ↓
┌────────────────────────────────┐
│  Transformer Decoder Block 1   │
│  - Masked Multi-Head Attention │
│  - Feed-Forward Network        │
│  - Layer Normalization         │
│  - Residual Connections        │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│  Transformer Decoder Block 2   │
└────────────────────────────────┘
     ↓
     ... (N blocks)
     ↓
┌────────────────────────────────┐
│  Transformer Decoder Block N   │
└────────────────────────────────┘
     ↓
Layer Norm
     ↓
Linear (project to vocabulary)
     ↓
Logits [batch_size, seq_len, vocab_size]
```

### Implementation

```python
import torch.nn.functional as F

class GPTBlock(nn.Module):
    """Single transformer decoder block with causal attention."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Masked multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, 
            n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        """
        x: [batch_size, seq_len, d_model]
        attn_mask: Causal mask
        """
        # Pre-LN: Normalize before attention
        normed = self.ln1(x)
        
        # Masked self-attention with residual
        attn_out, _ = self.attention(
            normed, normed, normed,
            attn_mask=attn_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        x = x + self.ffn(self.ln2(x))
        
        return x

class GPTModel(nn.Module):
    """GPT-style autoregressive transformer."""
    
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, 
                 d_ff=3072, max_seq_len=1024, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (learned)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (token embedding = output projection)
        self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len]
        returns: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        )
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits

# Example: Create small GPT model
vocab_size = 50257  # GPT-2 vocabulary size
model = GPTModel(
    vocab_size=vocab_size,
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=3072,
    max_seq_len=1024
)

print(f"GPT Model created:")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Vocabulary size: {vocab_size:,}")
print(f"  Hidden dimension: 768")
print(f"  Layers: 12")
print(f"  Attention heads: 12")
```

### Key Components

```python
def explain_gpt_components():
    """Explain key components of GPT architecture."""
    
    components = {
        'Token Embedding': {
            'purpose': 'Convert token IDs to dense vectors',
            'shape': '[vocab_size, d_model]',
            'note': 'Learned during training'
        },
        'Position Embedding': {
            'purpose': 'Encode position information',
            'shape': '[max_seq_len, d_model]',
            'note': 'Learned (GPT) or sinusoidal (original Transformer)'
        },
        'Masked Attention': {
            'purpose': 'Attend to previous positions only',
            'shape': 'Attention mask prevents future info leakage',
            'note': 'Key difference from BERT'
        },
        'Feed-Forward': {
            'purpose': 'Non-linear transformation',
            'shape': '[d_model] → [d_ff] → [d_model]',
            'note': 'Applied independently to each position'
        },
        'Layer Norm': {
            'purpose': 'Stabilize training',
            'shape': 'Normalize across d_model dimension',
            'note': 'Pre-LN (before attention/FFN) in modern models'
        },
        'Residual Connections': {
            'purpose': 'Enable deep networks',
            'shape': 'x + SubLayer(x)',
            'note': 'Gradient flow through network'
        }
    }
    
    print("GPT Architecture Components:\n")
    for component, info in components.items():
        print(f"{component}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

explain_gpt_components()
```

## Attention Mechanisms

### Self-Attention Recap

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply mask (set future positions to -inf)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

print("Self-Attention in GPT:")
print("  Query (Q): What am I looking for?")
print("  Key (K): What do I contain?")
print("  Value (V): What information do I have?")
print("\n  Attention score = Q · K^T / √d_k")
print("  Apply causal mask to prevent looking ahead")
print("  Output = softmax(scores) · V")
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head: [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output

print("\nMulti-Head Attention:")
print("  Allows model to attend to different aspects simultaneously")
print("  Each head: d_k = d_model / n_heads")
print("  Example: d_model=768, n_heads=12 → d_k=64")
print("  Heads might specialize in:")
print("    • Head 1: Syntactic dependencies")
print("    • Head 2: Semantic relationships")
print("    • Head 3: Long-range dependencies")
print("    • etc.")
```

### Attention Patterns

```python
def visualize_attention_patterns():
    """Typical attention patterns in autoregressive models."""
    
    print("Attention Patterns in GPT:\n")
    
    patterns = {
        'Local attention': {
            'description': 'Attend to nearby tokens',
            'example': 'Word attending to previous 2-3 words',
            'purpose': 'Capture local syntax and grammar'
        },
        'Previous token': {
            'description': 'Strong attention to immediately previous token',
            'example': 'In "New York", "York" attends strongly to "New"',
            'purpose': 'Capture bigram patterns, continuations'
        },
        'Delimiter attention': {
            'description': 'Attend to special tokens (periods, commas)',
            'example': 'Tokens attending to sentence boundaries',
            'purpose': 'Segment and structure text'
        },
        'Long-range': {
            'description': 'Attend to distant relevant tokens',
            'example': 'Pronoun attending to its antecedent',
            'purpose': 'Coreference, discourse relations'
        },
        'Broadcast': {
            'description': 'Many tokens attend to same important token',
            'example': 'All tokens in sentence attend to subject',
            'purpose': 'Propagate important information'
        }
    }
    
    for pattern, info in patterns.items():
        print(f"{pattern}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

visualize_attention_patterns()
```

## Training Autoregressive Models

### Data Preparation

```python
def prepare_autoregressive_data(text, tokenizer, max_length=512):
    """
    Prepare data for autoregressive language modeling.
    
    Single sequence becomes multiple training examples.
    """
    # Tokenize
    tokens = tokenizer.encode(text)
    
    # Create examples by sliding window
    examples = []
    
    for i in range(0, len(tokens) - max_length, max_length):
        chunk = tokens[i:i + max_length + 1]  # +1 for target
        
        input_ids = chunk[:-1]  # All except last
        labels = chunk[1:]      # All except first
        
        examples.append({
            'input_ids': input_ids,
            'labels': labels
        })
    
    return examples

# Conceptual example
print("Data Preparation for Autoregressive LM:\n")

text = "The quick brown fox jumps over the lazy dog"
tokens = text.split()  # Simplified tokenization

print(f"Original text: {text}")
print(f"Tokens: {tokens}\n")

print("Training examples (max_length=4):")
for i in range(len(tokens) - 4):
    input_seq = tokens[i:i+4]
    target_seq = tokens[i+1:i+5]
    
    print(f"  Input:  {input_seq}")
    print(f"  Target: {target_seq}")
    print()
```

### Training Loop

```python
def train_autoregressive_lm(model, train_loader, epochs=3, lr=5e-5):
    """Training loop for autoregressive language model."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids']  # [batch_size, seq_len]
            labels = batch['labels']        # [batch_size, seq_len]
            
            # Forward pass
            logits = model(input_ids)       # [batch_size, seq_len, vocab_size]
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # Flatten
                labels.view(-1),
                ignore_index=-100  # Ignore padding
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        perplexity = np.exp(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

print("Autoregressive Training Process:")
print("1. Forward pass: Compute logits for all positions")
print("2. Shift: Align predictions with targets")
print("3. Loss: Cross-entropy between predictions and actual next tokens")
print("4. Backward: Compute gradients")
print("5. Optimize: Update parameters")
print("\nKey techniques:")
print("  • Gradient clipping (prevent instability)")
print("  • Learning rate scheduling (cosine annealing)")
print("  • Mixed precision training (faster, less memory)")
```

### Optimization Techniques

```python
optimization_techniques = {
    'Mixed Precision Training': {
        'benefit': '2x faster training, half memory usage',
        'method': 'Use FP16 for forward/backward, FP32 for updates',
        'code': 'torch.cuda.amp.autocast()'
    },
    'Gradient Accumulation': {
        'benefit': 'Train with larger effective batch size',
        'method': 'Accumulate gradients over multiple batches',
        'code': 'loss.backward(); if step % accum_steps == 0: optimizer.step()'
    },
    'Gradient Checkpointing': {
        'benefit': 'Reduce memory usage',
        'method': 'Recompute activations during backward pass',
        'code': 'torch.utils.checkpoint.checkpoint()'
    },
    'Learning Rate Warmup': {
        'benefit': 'Stable training start',
        'method': 'Gradually increase LR from 0 to target',
        'code': 'torch.optim.lr_scheduler.LambdaLR()'
    }
}

print("\nOptimization Techniques:\n")
for technique, info in optimization_techniques.items():
    print(f"{technique}:")
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

## Text Generation Strategies

### Greedy Decoding

```python
def greedy_generate(model, input_ids, max_length=50):
    """
    Greedy decoding: Always pick most likely next token.
    
    Fast but can be repetitive and boring.
    """
    generated = input_ids.clone()
    
    for _ in range(max_length):
        # Forward pass
        logits = model(generated)  # [1, seq_len, vocab_size]
        
        # Get logits for last position
        next_token_logits = logits[0, -1, :]  # [vocab_size]
        
        # Pick most likely token
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Append to sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # Stop if EOS token
        if next_token.item() == eos_token_id:
            break
    
    return generated

print("Greedy Decoding:")
print("  Strategy: Always pick argmax(P(w|context))")
print("  Pros:")
print("    • Fast (no sampling)")
print("    • Deterministic (same input → same output)")
print("  Cons:")
print("    • Repetitive ('very very very...')")
print("    • Misses creative options")
print("    • Can get stuck in loops")
```

### Temperature Sampling

```python
def temperature_sampling(logits, temperature=1.0):
    """
    Apply temperature to logits before sampling.
    
    temperature < 1: More confident (sharper distribution)
    temperature = 1: Unchanged
    temperature > 1: More random (flatter distribution)
    """
    # Apply temperature
    logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

# Example: Effect of temperature
print("\nTemperature Sampling:\n")

# Mock logits
logits = torch.tensor([2.0, 1.0, 0.5, 0.1])

for temp in [0.5, 1.0, 2.0]:
    scaled_logits = logits / temp
    probs = F.softmax(scaled_logits, dim=-1)
    
    print(f"Temperature = {temp}:")
    print(f"  Probabilities: {probs.numpy()}")
    print(f"  Effect: {'More peaked' if temp < 1 else 'More uniform' if temp > 1 else 'Unchanged'}")
    print()

print("Guidelines:")
print("  • temperature=0.7: Creative writing")
print("  • temperature=1.0: Balanced")
print("  • temperature=0.2: Factual, focused")
```

## Sampling Methods

### Top-k Sampling

```python
def top_k_sampling(logits, k=50, temperature=1.0):
    """
    Sample from top-k most likely tokens.
    
    Prevents sampling from very unlikely tokens.
    """
    # Apply temperature
    logits = logits / temperature
    
    # Get top-k
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # Softmax over top-k
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample from top-k
    sampled_index = torch.multinomial(top_k_probs, num_samples=1)
    next_token = top_k_indices[sampled_index]
    
    return next_token

print("Top-k Sampling:")
print("  Strategy: Sample only from k most likely tokens")
print("  Benefits:")
print("    • Prevents sampling from tail (very unlikely words)")
print("    • More coherent than pure sampling")
print("    • Still allows diversity")
print("  Typical values: k=50")
```

### Top-p (Nucleus) Sampling

```python
def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Nucleus sampling: Sample from smallest set with cumulative probability > p.
    
    Dynamic vocabulary size based on probability mass.
    """
    # Apply temperature
    logits = logits / temperature
    
    # Sort by probability
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # Cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability > p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Zero out removed tokens
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # Sample
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_sorted_index = torch.multinomial(probs, num_samples=1)
    next_token = sorted_indices[sampled_sorted_index]
    
    return next_token

print("\nTop-p (Nucleus) Sampling:")
print("  Strategy: Sample from tokens whose cumulative probability ≥ p")
print("  Benefits:")
print("    • Adaptive: vocabulary size changes based on distribution")
print("    • High confidence → fewer tokens")
print("    • Low confidence → more tokens")
print("  Typical values: p=0.9 or p=0.95")
print("\nExample:")
print("  If top 3 tokens have probs [0.5, 0.3, 0.2], sum = 1.0")
print("  With p=0.9, we'd sample from top 2 (sum=0.8) + part of 3rd")
```

### Sampling Comparison

```python
def compare_sampling_methods():
    """Compare different sampling strategies."""
    
    methods = {
        'Greedy': {
            'diversity': 'None (deterministic)',
            'quality': 'Can be repetitive',
            'speed': 'Fastest',
            'use_case': 'Factual, deterministic output'
        },
        'Temperature': {
            'diversity': 'Controlled by temperature',
            'quality': 'Can be incoherent if temperature too high',
            'speed': 'Fast',
            'use_case': 'Creative text, varied output'
        },
        'Top-k': {
            'diversity': 'Controlled by k',
            'quality': 'Good coherence',
            'speed': 'Fast',
            'use_case': 'Balanced creativity and coherence'
        },
        'Top-p': {
            'diversity': 'Adaptive based on confidence',
            'quality': 'Best overall quality',
            'speed': 'Slightly slower',
            'use_case': 'High-quality generation (default for GPT-3)'
        },
        'Beam Search': {
            'diversity': 'Limited (beam width)',
            'quality': 'High coherence',
            'speed': 'Slower',
            'use_case': 'Machine translation, summarization'
        }
    }
    
    print("\nSampling Methods Comparison:\n")
    print(f"{'Method':<15} {'Diversity':<25} {'Quality':<35} {'Speed':<10}")
    print("=" * 90)
    
    for method, info in methods.items():
        print(f"{method:<15} {info['diversity']:<25} {info['quality']:<35} {info['speed']:<10}")

compare_sampling_methods()
```

## Beam Search

### Algorithm

```python
def beam_search(model, input_ids, beam_width=5, max_length=50):
    """
    Beam search: Maintain top-k most likely sequences.
    
    More comprehensive search than greedy, but not sampling.
    """
    vocab_size = model.vocab_size
    batch_size = input_ids.size(0)
    
    # Initial beam: [batch_size, beam_width, seq_len]
    beams = input_ids.unsqueeze(1).expand(-1, beam_width, -1)
    beam_scores = torch.zeros(batch_size, beam_width)
    
    for step in range(max_length):
        # Get logits for all beams
        all_logits = []
        for beam_idx in range(beam_width):
            beam = beams[:, beam_idx, :]
            logits = model(beam)
            all_logits.append(logits[:, -1, :])  # Last position
        
        all_logits = torch.stack(all_logits, dim=1)  # [batch, beam_width, vocab]
        
        # Compute log probabilities
        log_probs = F.log_softmax(all_logits, dim=-1)
        
        # For each beam, get top-k next tokens
        # Score = beam_score + log_prob
        scores = beam_scores.unsqueeze(-1) + log_probs  # [batch, beam_width, vocab]
        scores = scores.view(batch_size, -1)  # Flatten: [batch, beam_width * vocab]
        
        # Get top beam_width scores
        top_scores, top_indices = torch.topk(scores, beam_width, dim=-1)
        
        # Update beams
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size
        
        # Construct new beams
        new_beams = []
        for b in range(batch_size):
            batch_beams = []
            for beam_idx in range(beam_width):
                old_beam_idx = beam_indices[b, beam_idx]
                old_beam = beams[b, old_beam_idx, :]
                new_token = token_indices[b, beam_idx]
                new_beam = torch.cat([old_beam, new_token.unsqueeze(0)])
                batch_beams.append(new_beam)
            new_beams.append(torch.stack(batch_beams))
        
        beams = torch.stack(new_beams)
        beam_scores = top_scores
    
    # Return best beam
    best_beam = beams[:, 0, :]
    return best_beam

print("Beam Search:")
print("  Strategy: Keep top-k most likely partial sequences")
print("  At each step:")
print("    1. Expand each beam by all possible next tokens")
print("    2. Score all candidates")
print("    3. Keep top-k scoring sequences")
print("\n  Beam width = 1 → Greedy decoding")
print("  Beam width = vocab_size → Exhaustive search")
print("  Typical: beam_width = 5-10")
```

### Beam Search Variations

```python
beam_variations = {
    'Length Normalization': {
        'problem': 'Longer sequences have lower scores (more terms)',
        'solution': 'Normalize by length: score / length^α',
        'parameter': 'α typically 0.6-0.8'
    },
    'Coverage Penalty': {
        'problem': 'Repetition or skipping source words',
        'solution': 'Penalize attending to same words repeatedly',
        'parameter': 'Coverage coefficient β'
    },
    'Diverse Beam Search': {
        'problem': 'Beams become similar',
        'solution': 'Encourage diversity across beam groups',
        'parameter': 'Diversity penalty λ'
    },
    'Constrained Beam Search': {
        'problem': 'Need to include specific words/phrases',
        'solution': 'Force beams to include constraints',
        'parameter': 'Required tokens/phrases'
    }
}

print("\nBeam Search Variations:\n")
for variation, info in beam_variations.items():
    print(f"{variation}:")
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

## Control and Conditioning

### Conditional Generation

```python
def conditional_generate(model, prompt, condition, max_length=50):
    """
    Generate text conditioned on prompt and control code.
    
    Example: Control sentiment, style, topic
    """
    # Encode prompt with condition
    # E.g., "[POSITIVE] Write a review:"
    conditioned_prompt = f"[{condition}] {prompt}"
    
    # Tokenize
    input_ids = tokenizer.encode(conditioned_prompt)
    
    # Generate
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.8,
        top_p=0.9
    )
    
    return output

print("Conditional Generation:")
print("  Approach 1: Control tokens")
print("    Input: '[POSITIVE] The movie was' → 'amazing, brilliant...'")
print("    Input: '[NEGATIVE] The movie was' → 'terrible, boring...'")
print()
print("  Approach 2: Prompt engineering")
print("    Input: 'Write a formal email:' → formal style")
print("    Input: 'Write casual text:' → casual style")
print()
print("  Approach 3: Fine-tuning")
print("    Train on specific domain/style")
print("    Better control, but requires labeled data")
```

### Prompt Engineering

```python
prompt_engineering_examples = {
    'Zero-shot': {
        'prompt': 'Translate to French: Hello',
        'expected': 'Bonjour',
        'when': 'Model trained on translation'
    },
    'Few-shot': {
        'prompt': '''Translate to French:
English: Hello
French: Bonjour
English: Goodbye  
French: Au revoir
English: Thank you
French:''',
        'expected': 'Merci',
        'when': 'Model needs examples'
    },
    'Instruction following': {
        'prompt': 'Summarize the following text in one sentence: [text]',
        'expected': 'One-sentence summary',
        'when': 'Model trained for instructions'
    },
    'Chain-of-thought': {
        'prompt': 'Q: 15 + 27 = ?\nLet\'s think step by step.',
        'expected': 'Step-by-step reasoning → answer',
        'when': 'Complex reasoning needed'
    }
}

print("\nPrompt Engineering Strategies:\n")
for strategy, info in prompt_engineering_examples.items():
    print(f"{strategy}:")
    print(f"  Example: {info['prompt'][:50]}...")
    print(f"  Expected: {info['expected']}")
    print(f"  When to use: {info['when']}")
    print()
```

## Scaling and Variants

### GPT Model Sizes

```python
gpt_variants = {
    'GPT-1': {
        'parameters': '117M',
        'layers': 12,
        'd_model': 768,
        'year': 2018,
        'training': 'BooksCorpus'
    },
    'GPT-2': {
        'parameters': '1.5B (largest)',
        'layers': 48,
        'd_model': 1600,
        'year': 2019,
        'training': 'WebText (40GB)'
    },
    'GPT-3': {
        'parameters': '175B',
        'layers': 96,
        'd_model': 12288,
        'year': 2020,
        'training': 'CommonCrawl (filtered, 570GB)'
    },
    'GPT-3.5 (ChatGPT)': {
        'parameters': '~175B (estimated)',
        'layers': '~96',
        'd_model': '~12288',
        'year': 2022,
        'training': 'GPT-3 + RLHF'
    },
    'GPT-4': {
        'parameters': '~1.7T (estimated, MoE)',
        'layers': 'Unknown',
        'd_model': 'Unknown',
        'year': 2023,
        'training': 'Unknown + RLHF'
    }
}

print("GPT Model Evolution:\n")
for model, specs in gpt_variants.items():
    print(f"{model}:")
    for key, value in specs.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

### Other Autoregressive Models

```python
other_models = {
    'GPT-Neo/GPT-J': {
        'creator': 'EleutherAI',
        'description': 'Open-source GPT-3-like models',
        'sizes': '125M - 20B parameters'
    },
    'LLaMA': {
        'creator': 'Meta',
        'description': 'Efficient autoregressive models',
        'sizes': '7B - 65B parameters'
    },
    'PaLM': {
        'creator': 'Google',
        'description': 'Pathways Language Model',
        'sizes': '540B parameters'
    },
    'Chinchilla': {
        'creator': 'DeepMind',
        'description': 'Compute-optimal scaling',
        'sizes': '70B parameters (trained longer)'
    },
    'Claude': {
        'creator': 'Anthropic',
        'description': 'Constitutional AI, safety-focused',
        'sizes': 'Unknown'
    }
}

print("\nOther Autoregressive Models:\n")
for model, info in other_models.items():
    print(f"{model}:")
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

## Evaluation

### Perplexity

```python
def evaluate_perplexity(model, test_loader):
    """
    Compute perplexity on test set.
    
    Lower perplexity = better language model
    """
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            logits = model(input_ids)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

print("Perplexity Evaluation:")
print("  Measures: How well model predicts next token")
print("  Formula: exp(-1/N * Σ log P(w_i | context))")
print("  Interpretation: 'Effective vocabulary size'")
print("\nTypical values:")
print("  • GPT-2 on Wikipedia: ~30")
print("  • GPT-3 on WebText: ~20")
print("  • Better model → Lower perplexity")
```

### Generation Quality Metrics

```python
generation_metrics = {
    'BLEU': {
        'measures': 'N-gram overlap with reference',
        'range': '0-100',
        'use': 'Machine translation',
        'limitation': 'Doesn\'t capture semantic similarity'
    },
    'ROUGE': {
        'measures': 'Recall-based overlap',
        'range': '0-1',
        'use': 'Summarization',
        'limitation': 'Surface-level matching'
    },
    'METEOR': {
        'measures': 'Overlap with synonyms, stemming',
        'range': '0-1',
        'use': 'MT, generation',
        'limitation': 'Still limited semantic understanding'
    },
    'BERTScore': {
        'measures': 'Semantic similarity via BERT embeddings',
        'range': '0-1',
        'use': 'Any generation',
        'limitation': 'Expensive to compute'
    },
    'Human evaluation': {
        'measures': 'Fluency, coherence, relevance',
        'range': 'Rating scales',
        'use': 'Gold standard',
        'limitation': 'Expensive, slow, subjective'
    }
}

print("\nGeneration Quality Metrics:\n")
for metric, info in generation_metrics.items():
    print(f"{metric}:")
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

## Summary

**Key Concepts**:

1. **Autoregressive models** generate text left-to-right, predicting next token given previous tokens
2. **Causal attention** prevents looking ahead, ensuring unidirectional information flow
3. **GPT architecture** is a decoder-only transformer with masked self-attention
4. **Training objective** is next-token prediction (cross-entropy loss)
5. **Generation strategies** include greedy, sampling (temperature, top-k, top-p), and beam search
6. **Scaling** improves performance: GPT-1 (117M) → GPT-3 (175B) → GPT-4 (~1.7T)

**Architecture Components**:

```
Input → Token Emb + Position Emb → Transformer Blocks → LM Head → Logits
        [vocab, d]   [max_len, d]   (causal attention)   [d, vocab]
```

**Generation Strategies**:

| Method | Diversity | Quality | Speed | Use Case |
|--------|-----------|---------|-------|----------|
| Greedy | None | Can be repetitive | Fastest | Deterministic |
| Temperature | Controlled | Variable | Fast | Creative |
| Top-k | Fixed k | Good | Fast | Balanced |
| Top-p | Adaptive | Best | Medium | Default (GPT-3) |
| Beam Search | Limited | High coherence | Slowest | Translation |

**Key Advantages**:

- ✅ **Natural generation**: Matches inference (incremental)
- ✅ **Flexible length**: Can generate variable-length outputs
- ✅ **Simple training**: Standard language modeling objective
- ✅ **Scalable**: Benefits from scale (more parameters, more data)
- ✅ **Few-shot learning**: Large models learn from examples in prompt

**Limitations**:

- ❌ **Unidirectional**: Only sees left context
- ❌ **No revision**: Can't go back and change earlier tokens
- ❌ **Exposure bias**: Training vs inference mismatch
- ❌ **Repetition**: Tendency to repeat phrases
- ❌ **Long-range coherence**: Can lose thread in long generations

**Applications**:

- Text completion and generation
- Dialogue systems and chatbots
- Code generation
- Creative writing
- Question answering (with prompting)

## Next Steps

- Study [Masked Models](masked-models.md) for bidirectional context and understanding tasks
- Explore [Encoder-Decoder Models](encoder-decoder-models.md) for sequence-to-sequence
- Learn [Pretraining and Transfer Learning](pretraining-transfer.md) for training strategies
- Apply to [Text Generation](../application_patterns/text-generation.md) tasks
- Study [Prompt Engineering](../prompt-engineering/prompt-design.md) for better control
- Progress to [Large Language Models](../llm_concepts/large-language-models.md) for modern LLMs
