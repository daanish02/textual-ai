# Tokenization

## Table of Contents

- [Introduction](#introduction)
- [What is Tokenization?](#what-is-tokenization)
- [Word Tokenization](#word-tokenization)
- [Subword Tokenization](#subword-tokenization)
- [Character Tokenization](#character-tokenization)
- [Comparing Tokenization Approaches](#comparing-tokenization-approaches)
- [Practical Tokenization with Modern Tools](#practical-tokenization-with-modern-tools)
- [Special Tokens and Control Tokens](#special-tokens-and-control-tokens)
- [Tokenization Challenges](#tokenization-challenges)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Tokenization is the foundational step in every NLP pipeline. Before a model can process text, it must be broken into discrete units (tokens) that the model can operate on. The choice of tokenization strategy fundamentally affects model vocabulary size, handling of rare words, multilingual capability, and downstream performance.

**Why tokenization matters**:

- **Input representation**: Models work with sequences of tokens, not raw strings
- **Vocabulary management**: Tokenization determines vocabulary size and OOV handling
- **Subword structure**: Modern methods balance word meaning with character flexibility
- **Multilingual support**: Different languages require different tokenization strategies
- **Model compatibility**: Different models use different tokenizers (GPT vs BERT vs T5)

This guide covers word-level, subword-level, and character-level tokenization, with emphasis on modern subword methods (BPE, WordPiece, Unigram) used in transformers.

## What is Tokenization?

### Definition

**Tokenization** is the process of splitting text into smaller units called **tokens**. Tokens are the atomic units that NLP models process.

**Example**:

```
Text: "Hello, world! How are you?"

Possible tokenizations:

Word-level:     ["Hello", ",", "world", "!", "How", "are", "you", "?"]
Subword-level:  ["Hello", ",", "world", "!", "How", "are", "you", "?"]
                (or with BPE: ["Hel", "lo", ",", "world", "!", "How", "are", "you", "?"])
Character-level: ["H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!", ...]
```

### Why Not Just Split on Spaces?

Naive space-splitting fails for many reasons:

1. **Punctuation**: "Hello, world!" → Need to separate punctuation
2. **Contractions**: "don't" → "do" + "n't" or keep as one?
3. **Compounds**: "New York" → One unit or two?
4. **Languages without spaces**: Chinese, Japanese, Thai
5. **Rare words**: "antidisestablishmentarianism" → Unknown to vocabulary

Modern tokenization addresses these issues through sophisticated algorithms.

### The Vocabulary Problem

**Trade-off**: Vocabulary size vs coverage

- **Small vocabulary**: Many out-of-vocabulary (OOV) words
- **Large vocabulary**: Memory intensive, sparse training signal per token

**Example**:

```python
# Vocabulary size impacts

# Small vocabulary (word-level, 10K words)
# - "antidisestablishmentarianism" → <UNK> (unknown token)
# - Cannot handle new words

# Large vocabulary (word-level, 100K words)
# - Covers more words but still has OOV problem
# - Large embedding matrix (100K × embedding_dim)

# Subword vocabulary (BPE, 32K tokens)
# - "antidisestablishmentarianism" → ["anti", "dis", "establish", "ment", "arian", "ism"]
# - Can handle any word through composition
# - Reasonable vocabulary size
```

## Word Tokenization

### Basic Word Tokenization

Split text into words based on whitespace and punctuation.

```python
import re

def simple_word_tokenize(text):
    """Basic word tokenization using regex."""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return tokens

# Example
text = "Hello, world! How are you doing today?"
tokens = simple_word_tokenize(text)
print(tokens)
# Output: ['Hello', ',', 'world', '!', 'How', 'are', 'you', 'doing', 'today', '?']
```

### Advanced Word Tokenization

Libraries like NLTK provide sophisticated word tokenizers that handle edge cases.

```python
import nltk
from nltk.tokenize import word_tokenize

# Download required resources (run once)
# nltk.download('punkt')

text = "I don't know. Let's go to New York! It's 3.14159."
tokens = word_tokenize(text)
print(tokens)
# Output: ['I', 'do', "n't", 'know', '.', 'Let', "'s", 'go', 'to', 'New', 'York', '!',
#          'It', "'s", '3.14159', '.']
```

**Handling edge cases**:

- Contractions: "don't" → ["do", "n't"]
- Possessives: "John's" → ["John", "'s"]
- Numbers: "3.14159" kept together
- Punctuation: Separated but preserved

### Strengths and Weaknesses

**Strengths**:

- Simple and interpretable
- Tokens correspond to linguistic units (words)
- Works well for languages with clear word boundaries

**Weaknesses**:

- Large vocabulary (100K+ words for English)
- OOV problem for rare words and morphological variants
- Doesn't capture subword structure ("unhappiness" vs "happiness")
- Inefficient for morphologically rich languages
- Each typo creates a new OOV word

### Vocabulary Size Analysis

```python
from collections import Counter

def analyze_vocabulary(texts, tokenizer_func):
    """Analyze vocabulary size and word frequencies."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer_func(text))

    vocab = set(all_tokens)
    freq = Counter(all_tokens)

    print(f"Total tokens: {len(all_tokens)}")
    print(f"Unique tokens (vocab size): {len(vocab)}")
    print(f"Top 10 most common: {freq.most_common(10)}")

    # Frequency distribution
    counts = list(freq.values())
    singletons = sum(1 for c in counts if c == 1)
    print(f"Tokens appearing once: {singletons} ({100*singletons/len(vocab):.1f}%)")

    return vocab, freq

# Example with sample texts
sample_texts = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat and the dog are friends.",
]

vocab, freq = analyze_vocabulary(sample_texts, simple_word_tokenize)
```

## Subword Tokenization

Subword tokenization solves the vocabulary-coverage trade-off by breaking words into smaller pieces. This enables:

- **Fixed vocabulary size**: Control vocab size (e.g., 32K tokens)
- **Open vocabulary**: Any word can be represented through subword composition
- **Morphological awareness**: "play", "playing", "played" share "play"
- **Rare word handling**: Break rare words into common subwords

### Byte-Pair Encoding (BPE)

BPE iteratively merges the most frequent adjacent pairs of tokens.

**Algorithm**:

1. Start with character-level tokens
2. Find most frequent adjacent pair
3. Merge that pair into a new token
4. Repeat until desired vocabulary size

**Example**:

```
Initial vocabulary: {a, b, c, d, ...}
Text: "aaabdaaabac"

Iteration 1:
- Count pairs: ('a','a'): 4, ('a','b'): 2, ('b','d'): 1, ...
- Most frequent: ('a','a')
- Merge: 'aa' becomes new token
- Text: "aa ab d aa ab ac" → "aa|ab|d|aa|ab|ac"

Iteration 2:
- Count pairs: ('aa','ab'): 2, ...
- Most frequent: ('aa','ab')
- Merge: 'aaab' becomes new token
- Continue...

Final vocabulary: {a, b, c, d, aa, aaab, ...}
```

**Implementation**:

```python
from collections import Counter, defaultdict
import re

class SimpleBPE:
    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.merges = []  # List of merge operations
        self.vocab = set()

    def train(self, texts):
        """Train BPE on a corpus."""
        # Start with character-level tokens
        words = []
        for text in texts:
            # Split into words, then characters with end-of-word marker
            for word in text.split():
                word_tokens = list(word) + ['</w>']
                words.append(word_tokens)

        # Perform merges
        for i in range(self.num_merges):
            # Count adjacent pairs
            pairs = defaultdict(int)
            for word in words:
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j+1])] += 1

            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            # Merge the pair in all words
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        # Merge
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

        # Build vocabulary
        self.vocab = set()
        for word in words:
            self.vocab.update(word)

    def tokenize(self, text):
        """Tokenize text using learned merges."""
        # Start with character-level
        words = []
        for word in text.split():
            word_tokens = list(word) + ['</w>']
            words.append(word_tokens)

        # Apply merges in order
        for merge_pair in self.merges:
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == merge_pair:
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

        # Flatten
        tokens = [token for word in words for token in word]
        return tokens

# Example usage
texts = [
    "low lower lowest",
    "new newer newest",
    "wide wider widest",
]

bpe = SimpleBPE(num_merges=10)
bpe.train(texts)

print("Learned merges:")
for i, merge in enumerate(bpe.merges):
    print(f"{i+1}. {merge}")

print("\nVocabulary:", sorted(bpe.vocab))

test_text = "lower newest"
tokens = bpe.tokenize(test_text)
print(f"\nTokenization of '{test_text}':")
print(tokens)
```

**Output**:

```
Learned merges:
1. ('e', 'r')
2. ('e', 's')
3. ('n', 'e')
4. ('w', 'e')
...

Tokenization of 'lower newest':
['l', 'o', 'we', 'r', '</w>', 'ne', 'we', 'st', '</w>']
```

**Used in**: GPT, GPT-2, GPT-3, RoBERTa, BART

### WordPiece

WordPiece is similar to BPE but uses a different merge criterion: maximize likelihood rather than frequency.

**Algorithm**:

1. Start with character-level vocabulary
2. Compute likelihood of corpus with current vocabulary
3. For each potential merge, compute likelihood increase
4. Merge pair that maximizes likelihood increase
5. Repeat until desired vocabulary size

**Key difference from BPE**: Chooses merges based on language model probability, not raw frequency.

**Example**:

```python
# Conceptual WordPiece (simplified)
# Actual WordPiece uses language model likelihood

class SimpleWordPiece:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = set()

    def train(self, texts):
        """Train WordPiece tokenizer."""
        # Start with character vocabulary
        self.vocab = set()
        for text in texts:
            self.vocab.update(text.lower())

        # Add special prefix marker for subword units
        # In practice, WordPiece uses ## prefix for continuation

        # Iteratively add subwords that improve language model likelihood
        # (Simplified version)
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.lower().split())

        # Most common subwords (simplified)
        from itertools import combinations
        subword_scores = {}
        for word, count in word_counts.items():
            for length in range(2, min(len(word)+1, 6)):
                for i in range(len(word) - length + 1):
                    subword = word[i:i+length]
                    subword_scores[subword] = subword_scores.get(subword, 0) + count

        # Add top subwords to vocabulary
        top_subwords = sorted(subword_scores.items(), key=lambda x: x[1], reverse=True)
        for subword, _ in top_subwords[:self.vocab_size - len(self.vocab)]:
            self.vocab.add(subword)

    def tokenize(self, word):
        """Tokenize a word using greedy longest-match-first."""
        tokens = []
        start = 0
        word = word.lower()

        while start < len(word):
            # Find longest subword in vocab starting from start
            end = len(word)
            found = False
            while end > start:
                subword = word[start:end]
                if subword in self.vocab:
                    tokens.append(subword if start == 0 else '##' + subword)
                    start = end
                    found = True
                    break
                end -= 1

            if not found:
                # Unknown character, use special token or character
                tokens.append(word[start])
                start += 1

        return tokens

# Example
wp = SimpleWordPiece(vocab_size=50)
wp.train(["playing player played play playful"])

print("Vocabulary:", sorted(wp.vocab))
print("\nTokenization examples:")
for word in ["playing", "player", "playful"]:
    print(f"{word}: {wp.tokenize(word)}")
```

**Used in**: BERT, DistilBERT, Electra

### Unigram Language Model

Unigram tokenization starts with a large vocabulary and iteratively removes tokens to minimize loss.

**Algorithm**:

1. Start with large vocabulary (all substrings)
2. Train unigram language model
3. For each token, compute loss increase if removed
4. Remove tokens with smallest loss increase
5. Repeat until desired vocabulary size

**Key difference**: Top-down (remove) vs bottom-up (add) like BPE/WordPiece

```python
# Conceptual Unigram tokenization
# Actual implementation requires EM algorithm

class UnigramTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> probability

    def train(self, texts):
        """Train unigram language model."""
        # Start with all possible substrings (up to length limit)
        from collections import Counter
        subword_counts = Counter()

        for text in texts:
            words = text.lower().split()
            for word in words:
                # Add all substrings
                for length in range(1, min(len(word)+1, 10)):
                    for i in range(len(word) - length + 1):
                        subword_counts[word[i:i+length]] += 1

        # Initialize probabilities (simplified - actual uses EM)
        total = sum(subword_counts.values())
        self.vocab = {token: count/total for token, count in subword_counts.most_common(self.vocab_size)}

    def tokenize(self, word):
        """Tokenize using Viterbi algorithm to find best segmentation."""
        # Find segmentation that maximizes probability (simplified)
        word = word.lower()
        n = len(word)

        # Dynamic programming
        # best_prob[i] = best probability for word[:i]
        # best_split[i] = best last token ending at i
        best_prob = [0.0] * (n + 1)
        best_prob[0] = 1.0
        best_split = [''] * (n + 1)

        for i in range(1, n + 1):
            for j in range(i):
                token = word[j:i]
                if token in self.vocab:
                    prob = best_prob[j] * self.vocab[token]
                    if prob > best_prob[i]:
                        best_prob[i] = prob
                        best_split[i] = token

        # Reconstruct path
        tokens = []
        i = n
        while i > 0:
            token = best_split[i]
            if token:
                tokens.append(token)
                i -= len(token)
            else:
                i -= 1

        return list(reversed(tokens))

# Example
unigram = UnigramTokenizer(vocab_size=50)
unigram.train(["playing player played play playful"])

print("Top tokens:", list(sorted(unigram.vocab.items(), key=lambda x: x[1], reverse=True)[:10]))
print("\nTokenization examples:")
for word in ["playing", "player", "playful"]:
    print(f"{word}: {unigram.tokenize(word)}")
```

**Used in**: T5, ALBERT, XLNet

### SentencePiece

SentencePiece is a language-agnostic tokenizer that treats text as raw Unicode:

- **Language-agnostic**: No language-specific pre-processing
- **Reversible**: Can reconstruct original text (including spaces)
- **Space as token**: Encodes spaces explicitly (▁ symbol)

```python
# Using SentencePiece library
# pip install sentencepiece

import sentencepiece as spm

# Train SentencePiece model
# spm.SentencePieceTrainer.train(
#     input='corpus.txt',
#     model_prefix='spm_model',
#     vocab_size=5000,
#     model_type='bpe'  # or 'unigram'
# )

# Load and use
# sp = spm.SentencePieceProcessor()
# sp.load('spm_model.model')

# tokens = sp.encode_as_pieces("This is a test.")
# print(tokens)
# # Output: ['▁This', '▁is', '▁a', '▁test', '.']

# ids = sp.encode_as_ids("This is a test.")
# print(ids)

# # Decode back
# text = sp.decode_pieces(tokens)
# print(text)  # "This is a test."
```

**Key feature**: The ▁ symbol represents spaces, making tokenization reversible.

## Character Tokenization

Break text into individual characters.

```python
def char_tokenize(text):
    """Character-level tokenization."""
    return list(text)

text = "Hello, world!"
tokens = char_tokenize(text)
print(tokens)
# Output: ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']
```

### Strengths and Weaknesses

**Strengths**:

- Tiny vocabulary (26 letters + punctuation)
- No OOV problem
- Works for any language
- Robust to typos and spelling variations

**Weaknesses**:

- Very long sequences (each character is a token)
- Loses word-level meaning
- Model must learn to compose characters into words
- Computationally expensive (longer sequences)

**When to use**:

- Multilingual models with many scripts
- Tasks requiring character-level awareness (spelling correction)
- Extremely small data regimes
- When vocabulary size must be minimal

## Comparing Tokenization Approaches

### Visual Comparison

```
Original text: "The quick brown fox jumps over the lazy dog."

Word-level (space-split):
["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]

Subword (BPE):
["The", "qu", "ick", "brown", "fox", "jump", "s", "over", "the", "lazy", "dog", "."]

Character-level:
["T", "h", "e", " ", "q", "u", "i", "c", "k", " ", "b", "r", "o", "w", "n", ...]
```

### Trade-off Table

| Aspect          | Word          | Subword      | Character   |
| --------------- | ------------- | ------------ | ----------- |
| Vocab size      | Large (100K+) | Medium (32K) | Small (100) |
| OOV handling    | Poor          | Excellent    | Perfect     |
| Sequence length | Short         | Medium       | Long        |
| Semantic units  | Yes           | Partial      | No          |
| Morphology      | No            | Yes          | Yes         |
| Multilingual    | Limited       | Good         | Excellent   |
| Computation     | Efficient     | Moderate     | Expensive   |

### Empirical Comparison

```python
def compare_tokenizations(text):
    """Compare different tokenization strategies."""
    # Word-level
    word_tokens = text.split()

    # Pseudo-subword (just for demo - real BPE would be different)
    # Simulating by splitting on common prefixes/suffixes
    subword_tokens = []
    for word in word_tokens:
        if len(word) > 5:
            subword_tokens.extend([word[:3], word[3:]])
        else:
            subword_tokens.append(word)

    # Character-level
    char_tokens = list(text.replace(' ', '_'))

    print(f"Original: {text}")
    print(f"\nWord tokens ({len(word_tokens)}): {word_tokens}")
    print(f"Subword tokens ({len(subword_tokens)}): {subword_tokens}")
    print(f"Char tokens ({len(char_tokens)}): {char_tokens[:50]}{'...' if len(char_tokens) > 50 else ''}")

    # Compression ratio
    print(f"\nCompression ratios:")
    print(f"  Word: {len(text) / len(word_tokens):.2f} chars/token")
    print(f"  Subword: {len(text) / len(subword_tokens):.2f} chars/token")
    print(f"  Char: {len(text) / len(char_tokens):.2f} chars/token")

# Example
compare_tokenizations("The interplanetary astronaut explored the extraterrestrial landscape.")
```

## Practical Tokenization with Modern Tools

### Using Hugging Face Transformers

The Hugging Face `transformers` library provides tokenizers for all major models.

```python
from transformers import AutoTokenizer

# Load different tokenizers
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")  # BPE
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # WordPiece
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")  # SentencePiece

text = "The interplanetary astronaut explored the extraterrestrial landscape."

print("GPT-2 (BPE):")
tokens = gpt2_tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
print(f"IDs: {gpt2_tokenizer.encode(text)}")
print(f"Decoded: {gpt2_tokenizer.decode(gpt2_tokenizer.encode(text))}")

print("\nBERT (WordPiece):")
tokens = bert_tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
print(f"IDs: {bert_tokenizer.encode(text)}")

print("\nT5 (SentencePiece):")
tokens = t5_tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
print(f"IDs: {t5_tokenizer.encode(text)}")
```

**Output**:

```
GPT-2 (BPE):
Tokens: ['The', 'Ġinter', 'plan', 'etary', 'Ġastron', 'aut', ...]
IDs: [464, 987, 11578, 316, ...]

BERT (WordPiece):
Tokens: ['the', 'inter', '##plan', '##etary', 'astro', '##naut', ...]
IDs: [101, 1996, 6970, ...]

T5 (SentencePiece):
Tokens: ['▁The', '▁inter', 'plan', 'etary', '▁astro', 'na', 'ut', ...]
IDs: [37, 1413, ...]
```

### Tokenizer Features

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Natural language processing is fascinating!"

# Basic tokenization
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# Encode (text -> IDs)
input_ids = tokenizer.encode(text, add_special_tokens=True)
print("Input IDs:", input_ids)

# Decode (IDs -> text)
decoded = tokenizer.decode(input_ids)
print("Decoded:", decoded)

# Full encoding (with attention masks, etc.)
encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
print("Encoding keys:", encoding.keys())
print("Input IDs:", encoding['input_ids'])
print("Attention mask:", encoding['attention_mask'])

# Batch encoding
texts = [
    "First sentence.",
    "Second sentence is longer.",
]
batch_encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print("\nBatch input IDs shape:", batch_encoding['input_ids'].shape)
```

### Building a Custom Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Create a BPE tokenizer from scratch
tokenizer = Tokenizer(models.BPE())

# Use whitespace pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train on corpus
trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>"])

# files = ["corpus.txt"]
# tokenizer.train(files, trainer)

# Use the tokenizer
# output = tokenizer.encode("This is a test sentence.")
# print(output.tokens)
# print(output.ids)
```

## Special Tokens and Control Tokens

Modern tokenizers include special tokens for model control.

### Common Special Tokens

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("Special tokens:")
print(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"  UNK: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
print(f"  CLS: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
print(f"  SEP: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
print(f"  MASK: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

# Usage in encoding
text = "Hello world"
tokens = tokenizer.tokenize(text)
print(f"\nTokens: {tokens}")

# With special tokens
encoded = tokenizer.encode(text, add_special_tokens=True)
decoded_tokens = [tokenizer.decode([id]) for id in encoded]
print(f"With special tokens: {decoded_tokens}")
```

### Token Types

| Token    | Purpose                     | Example                  |
| -------- | --------------------------- | ------------------------ |
| `<PAD>`  | Padding shorter sequences   | Used in batching         |
| `<UNK>`  | Unknown/OOV words           | Rare words not in vocab  |
| `<CLS>`  | Classification token (BERT) | Sentence representation  |
| `<SEP>`  | Separator between segments  | Question \| Answer       |
| `<MASK>` | Masked token (BERT)         | Masked language modeling |
| `<BOS>`  | Beginning of sequence (GPT) | Start of generation      |
| `<EOS>`  | End of sequence (GPT)       | End of generation        |

### Example: Sentence Pair Encoding

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentence_a = "What is NLP?"
sentence_b = "Natural Language Processing."

# Encode sentence pair
encoding = tokenizer(sentence_a, sentence_b, add_special_tokens=True)

print("Tokens:", tokenizer.convert_ids_to_tokens(encoding['input_ids']))
# Output: ['[CLS]', 'what', 'is', 'nlp', '?', '[SEP]', 'natural', 'language', 'processing', '.', '[SEP]']

print("Token type IDs:", encoding['token_type_ids'])
# Output: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# 0 = first sentence, 1 = second sentence
```

## Tokenization Challenges

### Out-of-Vocabulary (OOV) Words

**Problem**: Word-level tokenizers encounter unknown words.

```python
# Simulated OOV handling

word_vocab = {"the", "cat", "sat", "on", "mat"}

def word_tokenize_with_unk(text, vocab):
    tokens = []
    for word in text.lower().split():
        if word in vocab:
            tokens.append(word)
        else:
            tokens.append("<UNK>")
    return tokens

text = "the cat sat on the rug"
tokens = word_tokenize_with_unk(text, word_vocab)
print(tokens)
# Output: ['the', 'cat', 'sat', 'on', 'the', '<UNK>']
# "rug" is OOV
```

**Solution**: Subword tokenization breaks OOV words into known subwords.

```python
# With subword tokenization (conceptual)
# "rug" might be split into ['r', 'ug'] where both are in vocab
```

### Multilingual Tokenization

Different languages have different tokenization needs:

**Languages without spaces**:

```python
# Chinese doesn't use spaces between words
chinese_text = "我喜欢自然语言处理"
# Needs word segmentation before tokenization

# Japanese mixes scripts
japanese_text = "自然言語処理は面白い"
# Kanji, Hiragana, sometimes Katakana

# Thai doesn't use spaces
thai_text = "การประมวลผลภาษาธรรมชาติ"
```

**Solution**: Use language-agnostic tokenizers like SentencePiece or multilingual models like mBERT, XLM-R.

### Tokenization Artifacts

Tokenization can introduce artifacts:

**Capitalization**:

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print(tokenizer.tokenize("apple"))
# ['apple']

print(tokenizer.tokenize("Apple"))
# ['Apple']

print(tokenizer.tokenize("APPLE"))
# ['APP', 'LE']  # Tokenized differently!
```

**Leading spaces**:

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print(tokenizer.tokenize("hello"))
# ['hello']

print(tokenizer.tokenize(" hello"))
# ['Ġhello']  # Ġ represents leading space

# This matters for generation and token probabilities!
```

**Inconsistent splitting**:

```python
# Same word tokenized differently based on context
print(tokenizer.tokenize("playing"))
# ['play', 'ing']

print(tokenizer.tokenize("I am playing"))
# ['I', 'Ġam', 'Ġplaying']  # "playing" as whole word due to space
```

### Handling Special Characters

```python
text_with_special = "Email: user@example.com, URL: https://example.com, Price: $99.99"

tokens = tokenizer.tokenize(text_with_special)
print(tokens)
# Different tokenizers handle these differently
# Some preserve structure, others split aggressively
```

### Tokenization and Model Performance

**Case study: BERT vs GPT-2 tokenization**

```python
from transformers import BertTokenizer, GPT2Tokenizer

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "The model's performance on out-of-distribution data was suboptimal."

print("BERT tokens:")
print(bert_tokenizer.tokenize(text))
# ['the', 'model', "'", 's', 'performance', 'on', 'out', '-', 'of', '-',
#  'distribution', 'data', 'was', 'sub', '##optimal', '.']

print("\nGPT-2 tokens:")
print(gpt2_tokenizer.tokenize(text))
# ['The', 'Ġmodel', "'s", 'Ġperformance', 'Ġon', 'Ġout', '-', 'of', '-',
#  'distribution', 'Ġdata', 'Ġwas', 'Ġsub', 'optimal', '.']
```

**Observations**:

- BERT lowercases (trained on lowercased text)
- GPT-2 preserves case and uses Ġ for spaces
- Different subword splits affect model behavior

## Summary

**Key Concepts**:

1. **Tokenization** breaks text into discrete units (tokens) that models process
2. **Word tokenization** is simple but suffers from large vocabulary and OOV problems
3. **Subword tokenization** (BPE, WordPiece, Unigram) balances vocabulary size and coverage
4. **Character tokenization** has tiny vocabulary but creates very long sequences
5. **Modern transformers** almost exclusively use subword tokenization
6. **Special tokens** enable model control (padding, masking, sequence boundaries)
7. **Tokenization artifacts** can affect model behavior (capitalization, spaces)

**Practical Insights**:

- **Use existing tokenizers**: Don't build from scratch; use Hugging Face tokenizers matched to your model
- **Understand your tokenizer**: Know whether it lowercases, how it handles spaces, what special tokens it uses
- **Vocabulary size matters**: 32K-50K subword tokens is common; larger = more memory, smaller = longer sequences
- **Reversibility**: SentencePiece tokenization is reversible; others may lose information
- **Language-specific considerations**: Different languages need different approaches
- **Consistency**: Use the same tokenizer for training and inference

**Decision Framework**:

Choose tokenization based on:

- **Task**: Generation (subword), classification (word or subword), character-level tasks (character)
- **Language**: Multilingual (SentencePiece), English (any), morphologically rich (subword)
- **Vocabulary constraints**: Limited memory (smaller vocab), need coverage (subword)
- **Sequence length**: Short sequences (word), long context (subword), unlimited (character)

## Next Steps

- Explore [Text Preprocessing](text-preprocessing.md) to learn about preparing tokenized text
- Study [Linguistic Foundations](linguistic-foundations.md) to understand linguistic units (POS, NER, parsing)
- Learn about [Embeddings](../embeddings/) to see how tokens are converted to vectors
- Progress to [Language Models](../language_models/) to understand how models process token sequences
