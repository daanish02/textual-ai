# Encoder-Decoder Models

## Table of Contents

- [Introduction](#introduction)
- [Sequence-to-Sequence Architecture](#sequence-to-sequence-architecture)
- [Encoder-Decoder Attention](#encoder-decoder-attention)
- [T5: Text-to-Text Framework](#t5-text-to-text-framework)
- [Training Encoder-Decoder Models](#training-encoder-decoder-models)
- [Span Corruption and Denoising](#span-corruption-and-denoising)
- [Generation with Encoder-Decoder Models](#generation-with-encoder-decoder-models)
- [BART and Other Models](#bart-and-other-models)
- [Comparison with Other Architectures](#comparison-with-other-architectures)
- [Applications](#applications)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Encoder-decoder models** combine the strengths of both BERT and GPT: bidirectional encoding for understanding and autoregressive decoding for generation. This makes them ideal for sequence-to-sequence tasks like translation, summarization, and question answering.

```
Encoder-Decoder Architecture:

Input sequence              Output sequence
"Translate: Hello"     →    "Bonjour"

┌─────────────────┐         ┌─────────────────┐
│    ENCODER      │         │    DECODER      │
│  (Bidirectional)│────────→│  (Causal)       │
│                 │         │                 │
│  All-to-all     │         │  Left-to-right  │
│  attention      │         │  + cross-attn   │
└─────────────────┘         └─────────────────┘
      BERT-like                   GPT-like
```

**Key characteristics**:

- **Encoder**: Bidirectional, understands input
- **Decoder**: Autoregressive, generates output
- **Cross-attention**: Decoder attends to encoder outputs
- **Flexible**: Handles variable-length input and output

This guide covers encoder-decoder architectures, focusing on T5 and BART.

## Sequence-to-Sequence Architecture

### Classic Seq2Seq

```python
import torch
import torch.nn as nn

class Seq2SeqEncoder(nn.Module):
    """Encoder: Process input sequence bidirectionally."""

    def __init__(self, hidden_size=768, num_layers=6, num_heads=12):
        super().__init__()

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: [batch_size, src_len, hidden_size]
            src_mask: Padding mask for source

        Returns:
            encoder_output: [batch_size, src_len, hidden_size]
        """
        return self.encoder(src, src_key_padding_mask=src_mask)

class Seq2SeqDecoder(nn.Module):
    """Decoder: Generate output sequence autoregressively."""

    def __init__(self, hidden_size=768, num_layers=6, num_heads=12):
        super().__init__()

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: [batch_size, tgt_len, hidden_size] - decoder input
            memory: [batch_size, src_len, hidden_size] - encoder output
            tgt_mask: Causal mask for target
            memory_mask: Padding mask for encoder output

        Returns:
            decoder_output: [batch_size, tgt_len, hidden_size]
        """
        return self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_mask
        )

class Seq2SeqModel(nn.Module):
    """Full encoder-decoder model."""

    def __init__(self, vocab_size=32000, hidden_size=768, num_layers=6):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = Seq2SeqEncoder(hidden_size, num_layers)
        self.decoder = Seq2SeqDecoder(hidden_size, num_layers)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # Embed inputs
        src_embeds = self.embedding(src_ids)
        tgt_embeds = self.embedding(tgt_ids)

        # Encode source
        encoder_output = self.encoder(src_embeds, src_mask)

        # Create causal mask for decoder
        tgt_len = tgt_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)

        # Decode
        decoder_output = self.decoder(
            tgt_embeds,
            encoder_output,
            tgt_mask=causal_mask,
            memory_mask=src_mask
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

print("Seq2Seq Model Architecture:")
print("  Encoder: Bidirectional transformer (like BERT)")
print("  Decoder: Causal transformer (like GPT)")
print("  Connection: Cross-attention from decoder to encoder")
print("\nInformation flow:")
print("  1. Encoder processes entire input bidirectionally")
print("  2. Decoder generates output token-by-token")
print("  3. Decoder attends to encoder outputs via cross-attention")
```

### Three Types of Attention

```python
def explain_attention_types():
    """Three attention mechanisms in encoder-decoder models."""

    print("Three Types of Attention:\n")

    print("1. Encoder Self-Attention (Bidirectional)")
    print("   • Where: Inside encoder")
    print("   • Pattern: All-to-all")
    print("   • Purpose: Understand input context")
    print("   • Example: Each input token attends to all input tokens")

    print("\n2. Decoder Self-Attention (Causal)")
    print("   • Where: Inside decoder")
    print("   • Pattern: Causal (left-to-right)")
    print("   • Purpose: Generate output sequentially")
    print("   • Example: Each output token attends to previous outputs")

    print("\n3. Cross-Attention (Encoder-Decoder)")
    print("   • Where: Between encoder and decoder")
    print("   • Pattern: Decoder queries encoder")
    print("   • Purpose: Align input and output")
    print("   • Example: Output token attends to relevant input tokens")

    print("\n")
    print("Visualization:")
    print("┌──────────────┐")
    print("│   ENCODER    │")
    print("│              │")
    print("│ Self-Attn ←→ │  Bidirectional")
    print("└──────┬───────┘")
    print("       │ Cross-attention")
    print("       ↓")
    print("┌──────────────┐")
    print("│   DECODER    │")
    print("│              │")
    print("│ Self-Attn →  │  Causal")
    print("└──────────────┘")

explain_attention_types()
```

### Masking Patterns

```python
def visualize_seq2seq_masks():
    """Visualize attention masks in seq2seq."""

    print("\n\nAttention Masks:\n")

    # Encoder self-attention (all-to-all)
    print("Encoder Self-Attention:")
    print("  Can attend to:")
    print("     A  B  C  D")
    print("  A  ✓  ✓  ✓  ✓")
    print("  B  ✓  ✓  ✓  ✓")
    print("  C  ✓  ✓  ✓  ✓")
    print("  D  ✓  ✓  ✓  ✓")
    print("  (All positions attend to all positions)")

    # Decoder self-attention (causal)
    print("\nDecoder Self-Attention:")
    print("  Can attend to:")
    print("     E  F  G  H")
    print("  E  ✓  ✗  ✗  ✗")
    print("  F  ✓  ✓  ✗  ✗")
    print("  G  ✓  ✓  ✓  ✗")
    print("  H  ✓  ✓  ✓  ✓")
    print("  (Causal: only to previous positions)")

    # Cross-attention (all encoder positions)
    print("\nCross-Attention:")
    print("  Decoder → Encoder:")
    print("       A  B  C  D")
    print("  E    ✓  ✓  ✓  ✓")
    print("  F    ✓  ✓  ✓  ✓")
    print("  G    ✓  ✓  ✓  ✓")
    print("  H    ✓  ✓  ✓  ✓")
    print("  (Each decoder position attends to all encoder positions)")

visualize_seq2seq_masks()
```

## Encoder-Decoder Attention

### Cross-Attention Mechanism

```python
class CrossAttention(nn.Module):
    """Cross-attention: Decoder attends to encoder outputs."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Query from decoder, Key and Value from encoder
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, encoder_output, encoder_mask=None):
        """
        Args:
            decoder_hidden: [batch, tgt_len, hidden_size]
            encoder_output: [batch, src_len, hidden_size]
            encoder_mask: [batch, src_len]
        """
        batch_size = decoder_hidden.size(0)

        # Queries from decoder
        Q = self.query(decoder_hidden)

        # Keys and Values from encoder
        K = self.key(encoder_output)
        V = self.value(encoder_output)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply encoder padding mask
        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out(context)

        return output, attn_weights

print("\nCross-Attention Mechanism:")
print("  Q (Query):  From decoder (what we're generating)")
print("  K (Key):    From encoder (input content)")
print("  V (Value):  From encoder (input content)")
print("\n  Process:")
print("    1. Decoder asks: 'What input is relevant now?'")
print("    2. Attention scores: Decoder Q × Encoder K")
print("    3. Weighted sum: Attention weights × Encoder V")
print("    4. Result: Context vector for generation")
```

### Attention Visualization

```python
def demonstrate_cross_attention():
    """Example of cross-attention in translation."""

    print("\n\nCross-Attention Example: English → French Translation\n")

    source = ["The", "cat", "sat"]
    target = ["Le", "chat", "s'est", "assis"]

    # Simulated attention weights (target attends to source)
    attention = [
        # "Le" attends to
        {"The": 0.9, "cat": 0.05, "sat": 0.05},
        # "chat" attends to
        {"The": 0.1, "cat": 0.85, "sat": 0.05},
        # "s'est" attends to
        {"The": 0.1, "cat": 0.1, "sat": 0.8},
        # "assis" attends to
        {"The": 0.05, "cat": 0.05, "sat": 0.9}
    ]

    print("Attention weights (target → source):\n")
    print(f"Source: {' '.join(source)}")
    print(f"Target: {' '.join(target)}\n")

    for i, tgt_word in enumerate(target):
        print(f"{tgt_word}:")
        for src_word in source:
            weight = attention[i][src_word]
            bar = "█" * int(weight * 20)
            print(f"  {src_word:6} {bar:20} {weight:.2f}")
        print()

    print("Observations:")
    print("  • 'Le' (the) strongly attends to 'The'")
    print("  • 'chat' (cat) strongly attends to 'cat'")
    print("  • 's'est assis' (sat) attends to 'sat'")
    print("  • Cross-attention learns alignment between languages")

demonstrate_cross_attention()
```

## T5: Text-to-Text Framework

### Unified Framework

**T5 (Text-to-Text Transfer Transformer)**: Treats every NLP task as text-to-text

```python
def t5_task_examples():
    """T5 converts all tasks to text-to-text format."""

    tasks = {
        'Translation': {
            'input': 'translate English to German: That is good.',
            'output': 'Das ist gut.'
        },
        'Summarization': {
            'input': 'summarize: [long article text]',
            'output': '[summary]'
        },
        'Question Answering': {
            'input': 'question: What is the capital? context: Paris is the capital of France.',
            'output': 'Paris'
        },
        'Classification': {
            'input': 'sentiment: This movie is great!',
            'output': 'positive'
        },
        'NER': {
            'input': 'ner: Google was founded by Larry Page.',
            'output': 'Google: ORG, Larry Page: PERSON'
        },
        'Similarity': {
            'input': 'stsb sentence1: The cat sat. sentence2: A cat was sitting.',
            'output': '4.5'
        }
    }

    print("T5: Text-to-Text Framework\n")
    print("Every task is converted to text-to-text format:\n")

    for task, example in tasks.items():
        print(f"{task}:")
        print(f"  Input:  {example['input'][:70]}...")
        print(f"  Output: {example['output']}")
        print()

    print("Benefits:")
    print("  • Unified architecture for all tasks")
    print("  • Same training procedure")
    print("  • Easy multi-task learning")
    print("  • Flexible task specification")

t5_task_examples()
```

### T5 Architecture

```python
class T5Model(nn.Module):
    """T5: Encoder-decoder with relative position embeddings."""

    def __init__(self, vocab_size=32128, d_model=512, num_layers=6, num_heads=8):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, batch_first=True),
            num_layers
        )

        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, d_model * 4, batch_first=True),
            num_layers
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Share embeddings with output layer (optional)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, decoder_input_ids):
        # Encode
        encoder_hidden = self.embedding(input_ids)
        encoder_output = self.encoder(encoder_hidden)

        # Decode
        decoder_hidden = self.embedding(decoder_input_ids)

        # Create causal mask
        tgt_len = decoder_input_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(decoder_input_ids.device)

        decoder_output = self.decoder(decoder_hidden, encoder_output, tgt_mask=causal_mask)

        # Predict next tokens
        logits = self.lm_head(decoder_output)

        return logits

print("\nT5 Model Architecture:")
print("  Encoder: 6 or 12 layers (T5-base or T5-large)")
print("  Decoder: 6 or 12 layers")
print("  Hidden size: 512 or 768")
print("  Attention heads: 8 or 12")
print("\nKey innovations:")
print("  • Relative position embeddings (not absolute)")
print("  • Shared embedding weights between input and output")
print("  • Pre-LayerNorm (more stable training)")
```

### T5 Variants

```python
t5_variants = {
    'T5-Small': {'params': '60M', 'layers': 6, 'hidden': 512, 'use': 'Fast experiments'},
    'T5-Base': {'params': '220M', 'layers': 12, 'hidden': 768, 'use': 'General purpose'},
    'T5-Large': {'params': '770M', 'layers': 24, 'hidden': 1024, 'use': 'Better performance'},
    'T5-3B': {'params': '3B', 'layers': 24, 'hidden': 1024, 'use': 'High performance'},
    'T5-11B': {'params': '11B', 'layers': 24, 'hidden': 1024, 'use': 'SOTA results'}
}

print("\nT5 Model Sizes:\n")
for variant, specs in t5_variants.items():
    print(f"{variant}:")
    for key, value in specs.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

## Training Encoder-Decoder Models

### Span Corruption Objective

```python
def create_span_corruption_data(text, corruption_rate=0.15, mean_span_length=3):
    """
    T5's span corruption training objective.

    Process:
      1. Randomly corrupt ~15% of tokens into spans
      2. Replace each span with a sentinel token <extra_id_X>
      3. Target: Reconstruct corrupted spans
    """
    tokens = text.split()
    n = len(tokens)

    # Determine which tokens to corrupt
    num_corrupt = int(n * corruption_rate)

    # Create corrupted input and target
    corrupted_input = []
    target = []

    sentinel_id = 0
    i = 0

    # Simplified span corruption
    while i < n:
        if i % 5 == 0 and i < n - 1:  # Corrupt this span
            # Find span
            span_len = min(mean_span_length, n - i)
            span = tokens[i:i+span_len]

            # Add sentinel to input
            corrupted_input.append(f"<extra_id_{sentinel_id}>")

            # Add sentinel + span to target
            target.append(f"<extra_id_{sentinel_id}>")
            target.extend(span)

            sentinel_id += 1
            i += span_len
        else:
            corrupted_input.append(tokens[i])
            i += 1

    target.append("<extra_id_end>")

    return ' '.join(corrupted_input), ' '.join(target)

# Example
text = "The quick brown fox jumps over the lazy dog"
corrupted, target = create_span_corruption_data(text)

print("\nSpan Corruption Example:\n")
print(f"Original: {text}")
print(f"Input:    {corrupted}")
print(f"Target:   {target}")

print("\nHow it works:")
print("  1. Replace spans with <extra_id_X> tokens")
print("  2. Model must predict: <extra_id_0> The quick brown <extra_id_1> jumps over ...")
print("  3. Learns to denoise and reconstruct text")
```

### Training Loop

```python
class T5Trainer:
    """Training encoder-decoder models."""

    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self, input_ids, target_ids):
        """
        Single training step.

        Args:
            input_ids: Corrupted input [batch_size, src_len]
            target_ids: Target output [batch_size, tgt_len]
        """
        self.model.train()

        # Decoder input: Shift target right (start with <pad> or <bos>)
        decoder_input_ids = torch.cat([
            torch.zeros(target_ids.size(0), 1, dtype=torch.long, device=self.device),
            target_ids[:, :-1]
        ], dim=1)

        # Forward pass
        logits = self.model(input_ids, decoder_input_ids)

        # Loss: Predict target tokens
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            loss = self.train_step(input_ids, target_ids)
            total_loss += loss

        return total_loss / len(dataloader)

print("\nTraining Process:")
print("  1. Create corrupted input from text")
print("  2. Encoder processes corrupted input")
print("  3. Decoder generates target autoregressively")
print("  4. Loss: Cross-entropy on target tokens")
print("  5. Update weights via backpropagation")
```

## Span Corruption and Denoising

### Why Span Corruption?

```python
def compare_pretraining_objectives():
    """Compare different pretraining objectives."""

    objectives = {
        'MLM (BERT)': {
            'task': 'Predict individual masked tokens',
            'example': 'The [MASK] sat → predict "cat"',
            'pros': 'Simple, effective',
            'cons': 'Only encoder, mismatch at inference'
        },
        'CLM (GPT)': {
            'task': 'Predict next token',
            'example': 'The cat → predict "sat"',
            'pros': 'Natural for generation',
            'cons': 'Only decoder, unidirectional'
        },
        'Span Corruption (T5)': {
            'task': 'Reconstruct corrupted spans',
            'example': '<X> sat <Y> → <X> cat <Y> on the mat',
            'pros': 'Encoder+decoder, longer context',
            'cons': 'More complex'
        },
        'Denoising (BART)': {
            'task': 'Reconstruct from various corruptions',
            'example': 'Multiple corruption types',
            'pros': 'Robust representations',
            'cons': 'More computational cost'
        }
    }

    print("Pretraining Objectives Comparison:\n")
    for obj, info in objectives.items():
        print(f"{obj}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

compare_pretraining_objectives()
```

### Implementation

```python
import random

def span_corruption_transform(tokens, noise_density=0.15, mean_noise_span_length=3.0):
    """
    Apply span corruption to a sequence.

    Based on T5 paper:
      • Corrupt ~15% of tokens
      • Mean span length ~3 tokens
      • Replace with sentinel tokens
    """
    num_tokens = len(tokens)
    num_noise_tokens = int(round(num_tokens * noise_density))
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))

    # Avoid empty spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = num_tokens - num_noise_tokens

    def _random_spans(length, num_spans):
        """Create random spans for corruption."""
        # Generate span boundaries
        span_starts = sorted(random.sample(range(length), num_spans))
        span_starts.append(length)

        spans = []
        prev = 0
        for start in span_starts:
            if start > prev:
                spans.append((prev, start))
            prev = start

        return spans

    # Create noise spans
    noise_span_lengths = [0] * num_noise_spans
    remaining = num_noise_tokens

    for i in range(num_noise_spans):
        # Distribute noise tokens across spans
        if i < num_noise_spans - 1:
            length = random.randint(1, remaining - (num_noise_spans - i - 1))
        else:
            length = remaining
        noise_span_lengths[i] = length
        remaining -= length

    # Build corrupted sequence
    corrupted = []
    target = []

    token_idx = 0
    for span_idx, span_length in enumerate(noise_span_lengths):
        # Add non-noise tokens before this span
        nonnoise_length = num_nonnoise_tokens // num_noise_spans
        corrupted.extend(tokens[token_idx:token_idx + nonnoise_length])
        token_idx += nonnoise_length

        # Add sentinel
        sentinel = f"<extra_id_{span_idx}>"
        corrupted.append(sentinel)

        # Target: sentinel + corrupted tokens
        target.append(sentinel)
        target.extend(tokens[token_idx:token_idx + span_length])
        token_idx += span_length

    # Add remaining tokens
    corrupted.extend(tokens[token_idx:])

    return corrupted, target

# Example
text = "The quick brown fox jumps over the lazy dog and cat"
tokens = text.split()

corrupted, target = span_corruption_transform(tokens)

print("\nSpan Corruption Transform:\n")
print(f"Original:  {' '.join(tokens)}")
print(f"Corrupted: {' '.join(corrupted)}")
print(f"Target:    {' '.join(target)}")
```

## Generation with Encoder-Decoder Models

### Greedy Decoding

```python
def greedy_decode_seq2seq(model, encoder_input, max_length=50, start_token=0, end_token=1):
    """
    Greedy decoding for encoder-decoder model.

    Process:
      1. Encode input once
      2. Generate tokens one at a time
      3. Feed previous predictions to decoder
    """
    model.eval()

    with torch.no_grad():
        # Encode input
        encoder_output = model.encode(encoder_input)

        # Start with start token
        decoder_input = torch.tensor([[start_token]])
        generated = [start_token]

        for _ in range(max_length):
            # Decode next token
            logits = model.decode(decoder_input, encoder_output)

            # Get most likely token
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_token)

            # Stop if end token
            if next_token == end_token:
                break

            # Append to decoder input
            decoder_input = torch.cat([
                decoder_input,
                torch.tensor([[next_token]])
            ], dim=1)

        return generated

print("Greedy Decoding:")
print("  1. Encode input once (bidirectional)")
print("  2. Decoder generates token-by-token (left-to-right)")
print("  3. Each step conditions on:")
print("     • Previous decoder outputs (self-attention)")
print("     • Encoder outputs (cross-attention)")
print("  4. Stop when <EOS> or max length")
```

### Beam Search

```python
def beam_search_seq2seq(model, encoder_input, beam_size=5, max_length=50, start_token=0, end_token=1):
    """
    Beam search for encoder-decoder model.

    Maintains top-k hypotheses at each step.
    """
    model.eval()

    with torch.no_grad():
        # Encode input once
        encoder_output = model.encode(encoder_input)

        # Initialize beams
        beams = [(torch.tensor([[start_token]]), 0.0)]  # (sequence, score)
        completed = []

        for _ in range(max_length):
            candidates = []

            for seq, score in beams:
                if seq[0, -1].item() == end_token:
                    completed.append((seq, score))
                    continue

                # Get next token probabilities
                logits = model.decode(seq, encoder_output)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

                # Get top-k tokens
                top_log_probs, top_indices = log_probs.topk(beam_size)

                for log_prob, token_id in zip(top_log_probs[0], top_indices[0]):
                    new_seq = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score))

            # Keep top-k beams
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            # Check if all beams completed
            if not beams:
                break

        # Return best completed sequence
        completed.extend(beams)
        best_seq, best_score = max(completed, key=lambda x: x[1])

        return best_seq[0].tolist()

print("\nBeam Search for Seq2Seq:")
print("  • Encoder runs once (efficient)")
print("  • Decoder explores multiple hypotheses")
print("  • Cross-attention from each beam to encoder")
print("  • Finds better outputs than greedy")
```

## BART and Other Models

### BART: Denoising Autoencoder

```python
def bart_pretraining_corruptions():
    """BART uses multiple corruption strategies."""

    original = "The quick brown fox jumps over the lazy dog"

    corruptions = {
        'Token Masking': {
            'description': 'Replace tokens with [MASK]',
            'example': 'The [MASK] brown fox [MASK] over the lazy dog',
            'similar_to': 'BERT'
        },
        'Token Deletion': {
            'description': 'Delete random tokens',
            'example': 'The brown fox over lazy dog',
            'similar_to': 'Harder than masking'
        },
        'Text Infilling': {
            'description': 'Replace spans with single [MASK]',
            'example': 'The [MASK] fox jumps [MASK] lazy dog',
            'similar_to': 'T5 span corruption'
        },
        'Sentence Permutation': {
            'description': 'Shuffle sentence order',
            'example': '[Sent 2] [Sent 1] [Sent 3]',
            'similar_to': 'Document-level'
        },
        'Document Rotation': {
            'description': 'Rotate to random starting point',
            'example': 'fox jumps over the lazy dog The quick brown',
            'similar_to': 'Position-based'
        }
    }

    print("BART Pretraining Corruptions:\n")
    print(f"Original: {original}\n")

    for corruption, info in corruptions.items():
        print(f"{corruption}:")
        print(f"  Description: {info['description']}")
        print(f"  Example: {info['example']}")
        print(f"  Similar to: {info['similar_to']}")
        print()

    print("Key idea:")
    print("  • Apply diverse corruptions during pretraining")
    print("  • Model learns robust denoising")
    print("  • Better generalization to downstream tasks")

bart_pretraining_corruptions()
```

### mT5 and ByT5

```python
multilingual_models = {
    'mT5': {
        'description': 'Multilingual T5',
        'languages': '101 languages',
        'training': 'mC4 corpus (multilingual C4)',
        'benefit': 'Cross-lingual transfer',
        'use_case': 'Translation, multilingual QA'
    },
    'ByT5': {
        'description': 'Byte-level T5',
        'tokenization': 'UTF-8 bytes (no tokenizer)',
        'vocab_size': '256 (byte vocabulary)',
        'benefit': 'Handles any language, no OOV',
        'use_case': 'Multilingual, low-resource languages'
    },
    'FLAN-T5': {
        'description': 'Instruction-tuned T5',
        'training': 'Finetuned on instruction tasks',
        'benefit': 'Better zero-shot performance',
        'use_case': 'Following instructions, prompts'
    }
}

print("\nmultilingual and Variant Models:\n")
for model, info in multilingual_models.items():
    print(f"{model}:")
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

## Comparison with Other Architectures

### Architecture Summary

```python
def compare_architectures():
    """Compare encoder-only, decoder-only, and encoder-decoder."""

    comparison = {
        'Architecture': {
            'BERT': 'Encoder-only',
            'GPT': 'Decoder-only',
            'T5': 'Encoder-decoder'
        },
        'Attention': {
            'BERT': 'Bidirectional',
            'GPT': 'Causal (unidirectional)',
            'T5': 'Bidirectional encoder + Causal decoder'
        },
        'Training': {
            'BERT': 'Masked Language Modeling',
            'GPT': 'Next Token Prediction',
            'T5': 'Span Corruption'
        },
        'Best for': {
            'BERT': 'Understanding (classification, NER)',
            'GPT': 'Generation (completion, dialogue)',
            'T5': 'Seq2seq (translation, summarization)'
        },
        'Input/Output': {
            'BERT': 'Fixed input → representations',
            'GPT': 'Prefix → continuation',
            'T5': 'Input sequence → Output sequence'
        },
        'Parameters (base)': {
            'BERT': '110M',
            'GPT': '117M (GPT-2)',
            'T5': '220M'
        }
    }

    print("Architecture Comparison:\n")
    print(f"{'Aspect':<20} {'BERT':<35} {'GPT':<35} {'T5':<35}")
    print("=" * 125)

    for aspect, values in comparison.items():
        print(f"{aspect:<20} {values['BERT']:<35} {values['GPT']:<35} {values['T5']:<35}")

compare_architectures()
```

### When to Use Encoder-Decoder

```python
use_cases = {
    'Use Encoder-Decoder when': [
        'Input and output are different sequences (translation)',
        'Need bidirectional understanding + generation',
        'Summarization (understand document, generate summary)',
        'Question answering (understand question+context, generate answer)',
        'Data-to-text (understand structured data, generate text)',
        'Abstractive tasks (not just copying input)'
    ],
    'Use Encoder-only (BERT) when': [
        'Classification tasks',
        'No generation needed',
        'Extractive tasks (select from input)',
        'Token-level tagging'
    ],
    'Use Decoder-only (GPT) when': [
        'Open-ended generation',
        'Completion tasks',
        'Large-scale prompting',
        'In-context learning'
    ]
}

print("\n\nWhen to Use Each Architecture:\n")
for category, cases in use_cases.items():
    print(f"{category}:")
    for case in cases:
        print(f"  • {case}")
    print()
```

## Applications

### Machine Translation

```python
print("Machine Translation with Encoder-Decoder:\n")

translation_example = {
    'Task': 'Translate English to French',
    'Input': 'translate English to French: The cat is on the table',
    'Process': [
        '1. Encoder: Process English sentence bidirectionally',
        '2. Decoder: Generate French words autoregressively',
        '3. Cross-attention: Align French words to English words',
        '4. Output: Le chat est sur la table'
    ]
}

print(f"Task: {translation_example['Task']}")
print(f"Input: {translation_example['Input']}\n")
print("Process:")
for step in translation_example['Process']:
    print(f"  {step}")
```

### Summarization

```python
print("\n\nSummarization with T5:\n")

summarization_example = {
    'Input': 'summarize: [Long article about climate change...]',
    'Encoder': 'Understands full article bidirectionally',
    'Decoder': 'Generates concise summary',
    'Output': 'Climate change causes rising temperatures and sea levels.'
}

for key, value in summarization_example.items():
    print(f"{key}: {value}")

print("\nTypes of summarization:")
print("  • Extractive: Select important sentences (BERT-based)")
print("  • Abstractive: Generate new summary (T5, BART)")
print("  • Encoder-decoder better for abstractive")
```

### Question Answering

```python
print("\n\nQuestion Answering:\n")

qa_types = {
    'Extractive QA': {
        'approach': 'Extract answer from context',
        'model': 'BERT (encoder-only)',
        'example': 'Q: Where is Paris? → Extract: "in France"'
    },
    'Abstractive QA': {
        'approach': 'Generate answer',
        'model': 'T5 (encoder-decoder)',
        'example': 'Q: Why is Paris famous? → Generate: "Paris is famous for..."'
    },
    'Open-domain QA': {
        'approach': 'Generate from knowledge',
        'model': 'T5 or GPT',
        'example': 'Q: Who wrote Hamlet? → Generate: "Shakespeare"'
    }
}

for qa_type, info in qa_types.items():
    print(f"{qa_type}:")
    for key, value in info.items():
        print(f"  {key.capitalize()}: {value}")
    print()
```

## Summary

**Key Concepts**:

1. **Encoder-decoder architecture** combines bidirectional encoder and autoregressive decoder
2. **Three attention types**: encoder self-attention, decoder self-attention, cross-attention
3. **Cross-attention** allows decoder to attend to encoder outputs for alignment
4. **T5** treats all NLP tasks as text-to-text problems
5. **Span corruption** pretraining teaches model to reconstruct corrupted text spans
6. **BART** uses diverse corruption strategies for robust representations

**Architecture**:

```
Input → Encoder (bidirectional) → Encoder output
                                       ↓ (cross-attention)
Start token → Decoder (causal) → Generated output
```

**Training Objectives**:

- **Span corruption** (T5): Reconstruct corrupted spans
- **Denoising** (BART): Reconstruct from multiple corruptions
- Both are sequence-to-sequence tasks

**Strengths**:

- ✅ **Bidirectional understanding**: Encoder sees full context
- ✅ **Flexible generation**: Variable-length output
- ✅ **Alignment**: Cross-attention learns input-output mapping
- ✅ **Versatile**: Works for many seq2seq tasks

**Limitations**:

- ❌ **More parameters**: Larger than encoder-only or decoder-only
- ❌ **Slower inference**: Two-stage process (encode then decode)
- ❌ **Complex training**: More components to optimize

**Applications**:

- Machine translation
- Summarization (abstractive)
- Question answering (generative)
- Data-to-text generation
- Dialog systems

**Model Variants**:

- **T5**: Text-to-text, span corruption
- **BART**: Denoising autoencoder, multiple corruptions
- **mT5**: Multilingual T5
- **FLAN-T5**: Instruction-tuned T5

## Next Steps

- Study [Pretraining and Transfer Learning](pretraining-transfer.md) for training strategies
- Explore [Fine-tuning](../llm_concepts/fine-tuning.md) encoder-decoder models
- Learn [Summarization](../application_patterns/summarization.md) techniques
- Study [Machine Translation](../application_patterns/translation.md) systems
- Understand [Prompt Engineering](../prompt_engineering/prompt-design.md) for T5
- Progress to [Large Language Models](../llm_concepts/large-language-models.md)
