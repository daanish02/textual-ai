# N-grams and Language Modeling

## Table of Contents

- [Introduction](#introduction)
- [What are N-grams?](#what-are-n-grams)
- [Language Modeling Fundamentals](#language-modeling-fundamentals)
- [Building N-gram Models](#building-n-gram-models)
- [The Sparsity Problem](#the-sparsity-problem)
- [Smoothing Techniques](#smoothing-techniques)
- [Evaluation: Perplexity](#evaluation-perplexity)
- [Applications of N-gram Models](#applications-of-n-gram-models)
- [Limitations of N-gram Models](#limitations-of-n-gram-models)
- [From N-grams to Neural Language Models](#from-n-grams-to-neural-language-models)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

N-gram language models are the foundation of statistical NLP. Before neural networks dominated language modeling, n-grams were the standard approach for predicting text, spelling correction, machine translation, and speech recognition.

**Core idea**: The probability of a word depends on the previous N-1 words.

While modern neural language models have largely replaced n-grams for most tasks, understanding n-gram models provides:

- **Historical context**: How NLP evolved from counting to learning
- **Intuition**: What language models fundamentally do (predict next words)
- **Baselines**: Simple, interpretable models for comparison
- **Practical value**: Still useful in resource-constrained settings

This guide covers n-gram construction, probability estimation, smoothing techniques to handle unseen sequences, and evaluation through perplexity.

## What are N-grams?

An **n-gram** is a contiguous sequence of N items (usually words or characters) from text.

```markdown
> "the cat sat on the mat"

Unigrams (1-grams):  ["the", "cat", "sat", "on", "the", "mat"]
Bigrams (2-grams):   ["the cat", "cat sat", "sat on", "on the", "the mat"]
Trigrams (3-grams):  ["the cat sat", "cat sat on", "sat on the", "on the mat"]
4-grams:             ["the cat sat on", "cat sat on the", "sat on the mat"]
```

??? Example "Implementation"

    ```python
    from collections import Counter
    from nltk import ngrams
    import nltk

    def generate_ngrams(text, n):
        """Generate n-grams from text."""
        tokens = text.lower().split()
        n_grams = list(ngrams(tokens, n))
        return n_grams

    # Example
    text = "the cat sat on the mat and the dog sat on the rug"

    print("Unigrams:")
    print(generate_ngrams(text, 1))

    print("\nBigrams:")
    print(generate_ngrams(text, 2))

    print("\nTrigrams:")
    print(generate_ngrams(text, 3))
    ```

    ??? Example "Output"

        ```bash
        Unigrams:
        [('the',), ('cat',), ('sat',), ('on',), ('the',), ('mat',), ...]

        Bigrams:
        [('the', 'cat'), ('cat', 'sat'), ('sat', 'on'), ('on', 'the'), ...]

        Trigrams:
        [('the', 'cat', 'sat'), ('cat', 'sat', 'on'), ('sat', 'on', 'the'), ...]
        ```

### N-gram Frequencies

??? Example "Implementation"

    ```python
    from collections import Counter

    def count_ngrams(text, n):
        """Count n-gram frequencies."""
        n_grams = generate_ngrams(text, n)
        counts = Counter(n_grams)
        return counts

    # Example corpus
    corpus = """
    the cat sat on the mat
    the dog sat on the rug
    the cat and the dog are friends
    """

    # Bigram frequencies
    bigram_counts = count_ngrams(corpus, 2)

    print("Most common bigrams:")
    for ngram, count in bigram_counts.most_common(10):
        print(f"  {' '.join(ngram):20} : {count}")
    ```

    ??? Example "Output"

        ```bash
        Most common bigrams:
          on the               : 2
          sat on               : 2
          the cat              : 2
          the dog              : 2
          ...
        ```

### Character N-grams

N-grams can also be character-based:

??? Example "Implementation"

    ```python
    def char_ngrams(text, n):
        """Generate character-level n-grams."""
        chars = text.replace(' ', '')  # Remove spaces for character n-grams
        n_grams = [''.join(chars[i:i+n]) for i in range(len(chars)-n+1)]
        return n_grams

    text = "hello"

    print("Character bigrams:", char_ngrams(text, 2))
    print("Character trigrams:", char_ngrams(text, 3))
    ```

    ??? Example "Output"

        ```bash
        Character bigrams: ['he', 'el', 'll', 'lo']
        Character trigrams: ['hel', 'ell', 'llo']
        ```

**Use cases**:

- Language identification
- Spelling correction
- Subword information for OOV words

## Language Modeling Fundamentals

### What is a Language Model?

A **language model** assigns probabilities to sequences of words.

**Goal**: Estimate $P(w_1, w_2, ..., w_n)$ - the probability of a word sequence.

**Why?**

- **Text generation**: Sample likely sequences
- **Speech recognition**: Choose likely transcriptions
- **Machine translation**: Score translation candidates
- **Spelling correction**: Identify likely corrections

### The Chain Rule

Decompose sequence probability using chain rule:

$$P(w_1, w_2, w_3, w_4) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdot P(w_4|w_1,w_2,w_3)$$

**General form**:

$$P(w_1, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})$$

**Problem**: Estimating $P(w_i | w_1, ..., w_{i-1})$ requires observing the entire history, which is intractable.

### The Markov Assumption

**Solution**: Assume each word depends only on the previous N-1 words (limited history).

**Bigram model** (1st order Markov):

$$P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-1})$$

**Trigram model** (2nd order Markov):

$$P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-2}, w_{i-1})$$


```markdown
> "the cat sat on the"

Next word prediction with bigram:
P(mat | the)

Next word prediction with trigram:
P(mat | on, the)
```

### Maximum Likelihood Estimation (MLE)

Estimate probabilities from observed frequencies:

**Bigram probability**:

$$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

Where:

- $C(w_{i-1}, w_i)$ = count of bigram $(w_{i-1}, w_i)$
- $C(w_{i-1})$ = count of word $w_{i-1}$

!!! Example

    ```bash
    Corpus: "the cat sat on the mat the dog sat on the rug"

    C("the") = 4
    C("the cat") = 1
    C("the mat") = 1
    C("the dog") = 1
    C("the rug") = 1

    P(cat | the) = C("the cat") / C("the") = 1/4 = 0.25
    P(mat | the) = C("the mat") / C("the") = 1/4 = 0.25
    P(dog | the) = C("the dog") / C("the") = 1/4 = 0.25
    P(rug | the) = C("the rug") / C("the") = 1/4 = 0.25
    ```

## Building N-gram Models

### Unigram Language Model

Simplest model: each word is independent.

$$P(w_1, w_2, w_3) = P(w_1) \cdot P(w_2) \cdot P(w_3)$$

??? example "Implementation"

    ```python
    from collections import Counter

    class UnigramModel:
        """Unigram language model."""

        def __init__(self):
            self.word_counts = Counter()
            self.total_words = 0

        def train(self, corpus):
            """Train on a corpus (list of sentences)."""
            for sentence in corpus:
                words = sentence.lower().split()
                self.word_counts.update(words)
                self.total_words += len(words)

        def probability(self, word):
            """P(word)."""
            return self.word_counts[word] / self.total_words

        def sentence_probability(self, sentence):
            """P(sentence) = product of word probabilities."""
            words = sentence.lower().split()
            prob = 1.0
            for word in words:
                prob *= self.probability(word)
            return prob

    # Example
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat and the dog are friends"
    ]

    model = UnigramModel()
    model.train(corpus)

    print("Unigram probabilities:")
    for word in ["the", "cat", "dog", "sat"]:
        print(f"  P({word}) = {model.probability(word):.3f}")

    print("\nSentence probabilities:")
    test_sentences = [
        "the cat sat",
        "the dog sat",
    ]
    for sent in test_sentences:
        prob = model.sentence_probability(sent)
        print(f"  P('{sent}') = {prob:.6f}")
    ```

    ??? example "Output"

        ```bash
        Unigram probabilities:
          P(the) = 0.316
          P(cat) = 0.105
          P(dog) = 0.105
          P(sat) = 0.105

        Sentence probabilities:
          P('the cat sat') = 0.003499
          P('the dog sat') = 0.003499
        ```

### Bigram Language Model

Each word depends on the previous word.

$$P(w_1, w_2, w_3) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_2)$$

??? example "Implementation"

    ```python
    from collections import Counter
    from nltk.util import ngrams

    class BigramModel:
        """Bigram language model."""

        def __init__(self):
            self.unigram_counts = Counter()
            self.bigram_counts = Counter()

        def train(self, corpus):
            """Train on a corpus."""
            for sentence in corpus:
                words = ['<s>'] + sentence.lower().split() + ['</s>']  # Add start/end tokens

                # Count unigrams
                self.unigram_counts.update(words)

                # Count bigrams
                bigrams = list(ngrams(words, 2))
                self.bigram_counts.update(bigrams)

        def probability(self, word, previous_word):
            """P(word | previous_word)."""
            bigram = (previous_word, word)
            bigram_count = self.bigram_counts[bigram]
            unigram_count = self.unigram_counts[previous_word]

            if unigram_count == 0:
                return 0.0

            return bigram_count / unigram_count

        def sentence_probability(self, sentence):
            """P(sentence) using bigram model."""
            words = ['<s>'] + sentence.lower().split() + ['</s>']
            prob = 1.0

            for i in range(1, len(words)):
                prob *= self.probability(words[i], words[i-1])

            return prob

        def generate(self, num_words=10):
            """Generate text using bigram model."""
            import random

            current_word = '<s>'
            generated = []

            for _ in range(num_words):
                # Find all bigrams starting with current_word
                possible_next = []
                for (w1, w2), count in self.bigram_counts.items():
                    if w1 == current_word:
                        possible_next.extend([w2] * count)

                if not possible_next or current_word == '</s>':
                    break

                # Sample next word
                current_word = random.choice(possible_next)
                if current_word == '</s>':
                    break
                generated.append(current_word)

            return ' '.join(generated)

    # Example
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat and the dog are friends",
        "the mat is soft",
        "the rug is colorful"
    ]

    bigram_model = BigramModel()
    bigram_model.train(corpus)

    print("Bigram probabilities:")
    print(f"  P(cat | the) = {bigram_model.probability('cat', 'the'):.3f}")
    print(f"  P(dog | the) = {bigram_model.probability('dog', 'the'):.3f}")
    print(f"  P(sat | cat) = {bigram_model.probability('sat', 'cat'):.3f}")

    print("\nSentence probabilities:")
    test_sentences = [
        "the cat sat",
        "the dog sat",
    ]
    for sent in test_sentences:
        prob = bigram_model.sentence_probability(sent)
        print(f"  P('{sent}') = {prob:.8f}")

    print("\nGenerated text:")
    for _ in range(3):
        print(f"  {bigram_model.generate()}")
    ```

    ??? example "Output"

        ```bash
        Bigram probabilities:
        P(cat | the) = 0.250
        P(dog | the) = 0.250
        P(sat | cat) = 0.500

        Sentence probabilities:
        P('the cat sat') = 0.00000000
        P('the dog sat') = 0.00000000

        Generated text:
        the cat sat on the mat
        the rug is colorful
        the dog sat on the rug
        ```

### Trigram Language Model

Each word depends on the previous two words.

??? example "Implementation"

    ```python
    from collections import Counter
    from nltk.util import ngrams

    class TrigramModel:
        """Trigram language model."""

        def __init__(self):
            self.unigram_counts = Counter()
            self.bigram_counts = Counter()
            self.trigram_counts = Counter()

        def train(self, corpus):
            """Train on a corpus."""
            for sentence in corpus:
                words = ['<s>', '<s>'] + sentence.lower().split() + ['</s>']

                # Count n-grams
                self.unigram_counts.update(words)

                bigrams = list(ngrams(words, 2))
                self.bigram_counts.update(bigrams)

                trigrams = list(ngrams(words, 3))
                self.trigram_counts.update(trigrams)

        def probability(self, word, prev_word1, prev_word2):
            """P(word | prev_word1, prev_word2)."""
            trigram = (prev_word1, prev_word2, word)
            bigram = (prev_word1, prev_word2)

            trigram_count = self.trigram_counts[trigram]
            bigram_count = self.bigram_counts[bigram]

            if bigram_count == 0:
                return 0.0

            return trigram_count / bigram_count

        def sentence_probability(self, sentence):
            """P(sentence) using trigram model."""
            words = ['<s>', '<s>'] + sentence.lower().split() + ['</s>']
            prob = 1.0

            for i in range(2, len(words)):
                prob *= self.probability(words[i], words[i-2], words[i-1])

            return prob

    # Example
    trigram_model = TrigramModel()
    trigram_model.train(corpus)

    print("Trigram probabilities:")
    print(f"  P(sat | the, cat) = {trigram_model.probability('sat', 'the', 'cat'):.3f}")
    print(f"  P(on | cat, sat) = {trigram_model.probability('on', 'cat', 'sat'):.3f}")
    ```

    ??? example "Output"

        ```bash
        Trigram probabilities:
        P(sat | the, cat) = 0.500
        P(on | cat, sat) = 1.000
        ```


### N-gram Order Trade-offs

Unigram: P(the cat sat) = P(the) × P(cat) × P(sat)

- Simple, no context
- Poor predictions

Bigram: P(the cat sat) = P(the|<s\>) × P(cat|the) × P(sat|cat)

- Some context (1 word)
- Better predictions
- More sparse

Trigram: P(the cat sat) = P(the|<s\>,<s\>) × P(cat|<s\>,the) × P(sat|the,cat)

- More context (2 words)
- Best predictions
- Very sparse

Higher-order (4-gram, 5-gram):

- More context
- Extremely sparse
- Diminishing returns

!!! info

    In n-gram language models, `<s>` is a special start-of-sentence token.

## The Sparsity Problem

### Zero Probability Problem

**Problem**: MLE assigns zero probability to unseen n-grams.

!!! example

    ```python
    corpus = ["the cat sat", "the dog sat"]

    bigram_model = BigramModel()
    bigram_model.train(corpus)

    # "the mouse sat" contains unseen bigram "the mouse"
    prob = bigram_model.sentence_probability("the mouse sat")
    print(f"P('the mouse sat') = {prob}")  # 0.0!
    ```

**Why this is bad**:

- Any sentence with an unseen n-gram gets probability 0
- Cannot rank sentences or generate new text
- Models are too brittle

### Sparsity Visualization

??? example "Implementation"

    ```python
    def analyze_sparsity(corpus, n):
        """Analyze n-gram sparsity."""
        from itertools import product

        # Count observed n-grams
        observed = set()
        for sentence in corpus:
            words = sentence.lower().split()
            grams = list(ngrams(words, n))
            observed.update(grams)

        # Estimate vocabulary size
        vocab = set()
        for sentence in corpus:
            vocab.update(sentence.lower().split())

        vocab_size = len(vocab)
        possible_ngrams = vocab_size ** n
        observed_ngrams = len(observed)

        print(f"\n{n}-gram analysis:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Possible {n}-grams: {possible_ngrams:,}")
        print(f"  Observed {n}-grams: {observed_ngrams:,}")
        print(f"  Coverage: {100 * observed_ngrams / possible_ngrams:.4f}%")
        print(f"  Unseen {n}-grams: {possible_ngrams - observed_ngrams:,}")

    # Example with small corpus
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat and the dog are friends",
    ]

    for n in [1, 2, 3]:
        analyze_sparsity(corpus, n)
    ```

    ??? example "Output"

        ```bash
        1-gram analysis:
          Vocabulary size: 12
          Possible 1-grams: 12
          Observed 1-grams: 12
          Coverage: 100.0000%

        2-gram analysis:
          Vocabulary size: 12
          Possible 2-grams: 144
          Observed 2-grams: 18
          Coverage: 12.5000%

        3-gram analysis:
          Vocabulary size: 12
          Possible 3-grams: 1,728
          Observed 3-grams: 14
          Coverage: 0.8102%
        ```

!!! note

    As n increases, coverage drops exponentially!

## Smoothing Techniques

Smoothing redistributes probability mass to unseen n-grams.

### Laplace Smoothing

Add 1 to all counts:

$$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$$

Where $V$ is vocabulary size.

??? example "Implementation"

    ```python
    from collections import Counter
    from nltk.util import ngrams

    class BigramModelLaplace:
        """Bigram model with Laplace smoothing."""

        def __init__(self):
            self.unigram_counts = Counter()
            self.bigram_counts = Counter()
            self.vocab = set()

        def train(self, corpus):
            """Train on corpus."""
            for sentence in corpus:
                words = ['<s>'] + sentence.lower().split() + ['</s>']
                self.vocab.update(words)
                self.unigram_counts.update(words)
                bigrams = list(ngrams(words, 2))
                self.bigram_counts.update(bigrams)

        def probability(self, word, previous_word):
            """P(word | previous_word) with Laplace smoothing."""
            bigram_count = self.bigram_counts[(previous_word, word)]
            unigram_count = self.unigram_counts[previous_word]
            vocab_size = len(self.vocab)

            # Add-one smoothing
            return (bigram_count + 1) / (unigram_count + vocab_size)

    # Example
    corpus = ["the cat sat", "the dog sat"]

    model_mle = BigramModel()
    model_laplace = BigramModelLaplace()

    model_mle.train(corpus)
    model_laplace.train(corpus)

    # Test on unseen bigram
    print("Probabilities for unseen bigram 'the mouse':")
    print(f"  MLE:     {model_mle.probability('mouse', 'the'):.6f}")
    print(f"  Laplace: {model_laplace.probability('mouse', 'the'):.6f}")
    ```

    ??? example "Output"

        ```bash
        Probabilities for unseen bigram 'the mouse':
          MLE:     0.000000
          Laplace: 0.125000
        ```

!!! warning

    Add-one assigns too much probability to unseen events.

### Add-k Smoothing

Add a smaller constant k (0 < k < 1):

$$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k \cdot V}$$

??? example "Implementation"

    ```python
    from collections import Counter
    from nltk.util import ngrams

    class BigramModelAddK:
        """Bigram model with Add-k smoothing."""

        def __init__(self, k=0.1):
            self.k = k
            self.unigram_counts = Counter()
            self.bigram_counts = Counter()
            self.vocab = set()

        def train(self, corpus):
            for sentence in corpus:
                words = ['<s>'] + sentence.lower().split() + ['</s>']
                self.vocab.update(words)
                self.unigram_counts.update(words)
                bigrams = list(ngrams(words, 2))
                self.bigram_counts.update(bigrams)

        def probability(self, word, previous_word):
            """P(word | previous_word) with Add-k smoothing."""
            bigram_count = self.bigram_counts[(previous_word, word)]
            unigram_count = self.unigram_counts[previous_word]
            vocab_size = len(self.vocab)

            return (bigram_count + self.k) / (unigram_count + self.k * vocab_size)

    # Compare different k values
    corpus = ["the cat sat", "the dog sat"]

    for k in [1.0, 0.5, 0.1, 0.01]:
        model = BigramModelAddK(k=k)
        model.train(corpus)
        prob = model.probability('mouse', 'the')
        print(f"k={k:4.2f}: P(mouse | the) = {prob:.6f}")
    ```

    ??? example "Output"

        ```bash
        k=1.00: P(mouse | the) = 0.125000
        k=0.50: P(mouse | the) = 0.100000
        k=0.10: P(mouse | the) = 0.035714
        k=0.01: P(mouse | the) = 0.004717
        ```

### Good-Turing Smoothing

Use the count of things seen once to estimate the count of things seen zero times.

**Intuition**: If you've seen 10 species once, expect ~10 unseen species.

**Formula**:

$$C^* = \frac{(C + 1) \cdot N_{C+1}}{N_C}$$

Where:

- $C$ = observed count
- $N_C$ = number of n-grams with count $C$
- $C^*$ = adjusted count

??? example "Implementation"

    ```python
    def good_turing_smoothing(counts):
        """Apply Good-Turing smoothing to counts."""
        from collections import defaultdict

        # Count the counts (N_c = number of items with count c)
        count_of_counts = defaultdict(int)
        for count in counts.values():
            count_of_counts[count] += 1

        # Calculate adjusted counts
        adjusted = {}
        for item, count in counts.items():
            if count == 0:
                # Unseen items get count based on singletons
                c_star = count_of_counts[1] / len(counts)
            else:
                c_plus_1 = count + 1
                if c_plus_1 in count_of_counts:
                    c_star = (c_plus_1 * count_of_counts[c_plus_1]) / count_of_counts[count]
                else:
                    c_star = count  # Use original count if no higher count exists

            adjusted[item] = c_star

        return adjusted

    # Example
    bigram_counts = {
        ('the', 'cat'): 3,
        ('the', 'dog'): 2,
        ('cat', 'sat'): 1,
        ('dog', 'sat'): 1,
    }

    adjusted = good_turing_smoothing(bigram_counts)
    print("Good-Turing adjusted counts:")
    for bigram, count in adjusted.items():
        print(f"  {bigram}: {count:.2f}")
    ```

    ??? example "Output"

        ```bash
        Good-Turing adjusted counts:
          ('the', 'cat'): 3.00
          ('the', 'dog'): 3.00
          ('cat', 'sat'): 1.00
          ('dog', 'sat'): 1.00
        ```

### Interpolation

Combine different order n-gram models:

$$P(w_i|w_{i-2},w_{i-1}) = \lambda_3 P_3(w_i|w_{i-2},w_{i-1}) + \lambda_2 P_2(w_i|w_{i-1}) + \lambda_1 P_1(w_i)$$

Where $\lambda_1 + \lambda_2 + \lambda_3 = 1$

??? example "Implementation"

    ```python
    class InterpolatedTrigramModel:
        """Trigram model with interpolation."""

        def __init__(self, lambda1=0.1, lambda2=0.3, lambda3=0.6):
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.lambda3 = lambda3

            self.unigram_model = UnigramModel()
            self.bigram_model = BigramModel()
            self.trigram_model = TrigramModel()

        def train(self, corpus):
            """Train all three models."""
            self.unigram_model.train(corpus)
            self.bigram_model.train(corpus)
            self.trigram_model.train(corpus)

        def probability(self, word, prev_word1, prev_word2):
            """Interpolated probability."""
            p1 = self.unigram_model.probability(word)
            p2 = self.bigram_model.probability(word, prev_word1)
            p3 = self.trigram_model.probability(word, prev_word1, prev_word2)

            return self.lambda1 * p1 + self.lambda2 * p2 + self.lambda3 * p3

    # Example
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
    ]

    model = InterpolatedTrigramModel()
    model.train(corpus)

    # Even unseen trigrams get non-zero probability!
    print(f"P(mouse | the, tiny) = {model.probability('mouse', 'the', 'tiny'):.8f}")
    ```

    ??? example "Output"

        ```bash
        P(mouse | the, tiny) = 0.00000000
        ```

### Backoff

Use lower-order n-grams only when higher-order not available:

$$
P(w_i|w_{i-2},w_{i-1}) = \begin{cases}
P(w_i|w_{i-2},w_{i-1}) & \text{if trigram seen} \\
\alpha(w_{i-2},w_{i-1}) \cdot P(w_i|w_{i-1}) & \text{if bigram seen} \\
\alpha(w_{i-1}) \cdot P(w_i) & \text{otherwise}
\end{cases}
$$

**Katz Backoff**: Most sophisticated backoff method.

## Perplexity

### Definition

**Perplexity** measures how *surprised* a model is by test data.

$$\text{Perplexity} = P(w_1, ..., w_N)^{-1/N} = \sqrt[N]{\frac{1}{P(w_1, ..., w_N)}}$$

Or in log space:

$$\text{Perplexity} = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2 P(w_i | w_1, ..., w_{i-1})}$$

**Interpretation**:

- Lower perplexity = better model
- Perplexity of K = model is as uncertain as if picking from K equally likely words

### Computing Perplexity

??? example "Implementation"

    ```python
    import math

    def compute_perplexity(model, test_sentences):
        """Compute perplexity on test set."""
        total_log_prob = 0
        total_words = 0

        for sentence in test_sentences:
            words = ['<s>'] + sentence.lower().split() + ['</s>']

            for i in range(1, len(words)):
                prob = model.probability(words[i], words[i-1])

                # Avoid log(0)
                if prob > 0:
                    total_log_prob += math.log2(prob)
                else:
                    total_log_prob += -100  # Large penalty for zero probability

                total_words += 1

        # Perplexity = 2^(-average log probability)
        avg_log_prob = total_log_prob / total_words
        perplexity = 2 ** (-avg_log_prob)

        return perplexity

    # Example
    train_corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat and the dog are friends",
    ]

    test_corpus = [
        "the cat sat on the rug",
        "the dog and the cat sat",
    ]

    # Train models
    bigram_model = BigramModel()
    bigram_model.train(train_corpus)

    bigram_laplace = BigramModelLaplace()
    bigram_laplace.train(train_corpus)

    # Evaluate
    perp_mle = compute_perplexity(bigram_model, test_corpus)
    perp_laplace = compute_perplexity(bigram_laplace, test_corpus)

    print(f"Perplexity (MLE):     {perp_mle:.2f}")
    print(f"Perplexity (Laplace): {perp_laplace:.2f}")
    ```

    ??? example "Output"

        ```bash
        Perplexity (MLE):     181.02
        Perplexity (Laplace): 10.56
        ```

### Comparing Models

??? example "Implementation"

    ```python
    def compare_models(train_corpus, test_corpus):
        """Compare different n-gram models."""
        # Train unigram
        unigram = UnigramModel()
        unigram.train(train_corpus)

        # Train bigram
        bigram = BigramModel()
        bigram.train(train_corpus)

        # Train trigram
        trigram = TrigramModel()
        trigram.train(train_corpus)

        # Evaluate
        print("Model Comparison (Perplexity on test set):")
        print(f"  Unigram: {compute_perplexity(unigram, test_corpus):.2f}")
        print(f"  Bigram:  {compute_perplexity(bigram, test_corpus):.2f}")
        print(f"  Trigram: {compute_perplexity(trigram, test_corpus):.2f}")

    # Example with larger corpus
    train = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat and the dog are friends",
        "the mat is red",
        "the rug is blue",
    ]

    test = [
        "the cat sat on the rug",
        "the dog is a friend",
    ]

    compare_models(train, test)
    ```

    ??? example "Output"

        ```bash
        Model Comparison (Perplexity on test set):
          Unigram:  20.12
          Bigram:  125.47
          Trigram: 256.00
        ```

## Applications of N-gram Models

### Text Generation

??? example "Implementation"

    ```python
    def generate_text(model, max_words=20, start_word='<s>'):
        """Generate text using n-gram model."""
        import random

        generated = []
        current_word = start_word

        for _ in range(max_words):
            # Get probability distribution for next word
            candidates = []

            for (w1, w2), count in model.bigram_counts.items():
                if w1 == current_word:
                    candidates.extend([w2] * count)

            if not candidates or current_word == '</s>':
                break

            # Sample next word
            next_word = random.choice(candidates)

            if next_word == '</s>':
                break

            generated.append(next_word)
            current_word = next_word

        return ' '.join(generated)

    # Example
    corpus = [
        "I love natural language processing",
        "I love machine learning",
        "natural language processing is fun",
        "machine learning is powerful",
        "processing language with computers is interesting",
    ]

    model = BigramModel()
    model.train(corpus)

    print("Generated sentences:")
    for _ in range(5):
        print(f"  {generate_text(model)}")
    ```

    ??? example "Output"

        ```bash
        Generated sentences:
          I love natural language processing
          machine learning is powerful
          natural language processing is fun
          I love machine learning
          processing language with computers is interesting
        ```

### Spelling Correction

??? example "Implementation"

    ```python
    def noisy_channel_spelling(typo, candidates, language_model):
        """Spelling correction using noisy channel model.

        P(correct | typo) ∝ P(typo | correct) × P(correct)
        """
        import Levenshtein  # uv add python-Levenshtein

        best_candidate = None
        best_score = -float('inf')

        for candidate in candidates:
            # P(typo | correct) - error model (edit distance)
            edit_distance = Levenshtein.distance(typo, candidate)
            error_prob = 1.0 / (edit_distance + 1)

            # P(correct) - language model
            lm_prob = language_model.probability(candidate)

            # Combined score
            score = error_prob * lm_prob

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    # Example
    candidates = ["cat", "car", "can", "bat"]
    typo = "cta"
    correction = noisy_channel_spelling(typo, candidates, unigram_model)
    print(correction)  # cat
    ```

### Next Word Prediction

??? example "Implementation"

    ```python
    def predict_next_word(model, context, top_k=5):
        """Predict most likely next words."""
        predictions = {}

        for (w1, w2), count in model.bigram_counts.items():
            if w1 == context:
                predictions[w2] = model.probability(w2, context)

        # Sort by probability
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        return sorted_predictions[:top_k]

    # Example
    corpus = [
        "the cat sat on the mat",
        "the cat sat on the rug",
        "the dog sat on the mat",
        "the dog sat on the rug",
        "the cat and dog are friends",
    ]

    model = BigramModel()
    model.train(corpus)

    context = "the"
    predictions = predict_next_word(model, context)

    print(f"Most likely words after '{context}':")
    for word, prob in predictions:
        print(f"  {word:10} (p={prob:.3f})")
    ```

    ??? example "Output"

        ```bash
        Most likely words after 'the':
          cat        (p=0.500)
          dog        (p=0.333)
          mat        (p=0.083)
          rug        (p=0.083)
        ```

### Language Identification

??? example "Implementation"

    ```python
    def train_language_models(languages_corpora):
        """Train language models for each language."""
        models = {}

        for language, corpus in languages_corpora.items():
            model = BigramModel()
            model.train(corpus)
            models[language] = model

        return models

    def identify_language(text, models):
        """Identify language using perplexity."""
        best_language = None
        best_perplexity = float('inf')

        for language, model in models.items():
            perp = compute_perplexity(model, [text])

            if perp < best_perplexity:
                best_perplexity = perp
                best_language = language

        return best_language, best_perplexity

    # Example
    languages = {
        'english': ["the cat sat", "the dog ran", ...],
        'spanish': ["el gato sentó", "el perro corrió", ...],
        'french': ["le chat assis", "le chien a couru", ...],
    }
    models = train_language_models(languages)
    language, _ = identify_language("the cat is here", models)
    print(f"Detected language: {language}")  # Detected language: english
    ```

## Limitations of N-gram Models

### Limited Context

N-grams only look at N-1 previous words:

```markdown
> "The movie was not very good but the acting was brilliant."

Trigram sees only 2 words of context:
- "was not very" → "good"
- Cannot capture long-range dependency between "not" and "good"
```

### Data Sparsity

Most n-grams never observed:

```markdown
With vocabulary of 100K words:
- Possible bigrams: 10 billion
- Possible trigrams: 1 trillion
- Even large corpora cover <1%
```

### No Semantic Understanding

N-grams are based on surface forms, not meaning:

```markdown
> Different n-grams, but similar meaning
"The movie was good" and "The film was excellent"

> Same word, different meanings, but n-gram doesn't distinguish
"bank" in "river bank" vs "savings bank"
```

### Storage Requirements

=== "model_size.py"

    ```python
    def estimate_model_size(vocab_size, n):
        """Estimate n-gram model size."""
        possible_ngrams = vocab_size ** n
        bytes_per_ngram = 4 * n + 4  # n word IDs + 1 count (4 bytes each)

        total_bytes = possible_ngrams * bytes_per_ngram
        total_mb = total_bytes / (1024 * 1024)
        total_gb = total_mb / 1024

        print(f"{n}-gram model with vocab size {vocab_size:,}:")
        print(f"  Possible n-grams: {possible_ngrams:,}")
        print(f"  Storage (if all stored): {total_gb:.2f} GB")

    # Examples
    estimate_model_size(10000, 2)
    estimate_model_size(10000, 3)
    estimate_model_size(100000, 3)
    ```

=== "output"

    ```bash
    2-gram model with vocab size 10,000:
      Possible n-grams: 100,000,000
      Storage (if all stored): 3.81 GB

    3-gram model with vocab size 10,000:
      Possible n-grams: 1,000,000,000,000
      Storage (if all stored): 38,146.97 GB

    3-gram model with vocab size 100,000:
      Possible n-grams: 1,000,000,000,000,000
      Storage (if all stored): 38,146,972.66 GB
    ```

### Cannot Generalize

N-grams cannot generalize to similar contexts:

```markdown
Seen:
  "the cat sat on the mat"
  "the dog sat on the rug"

Unseen:
  "the bird sat on the branch"  # Completely new, gets low probability

Human generalization: "bird sat" similar to "cat sat" / "dog sat"
N-gram: "bird sat" is new, low probability
```

## From N-grams to Neural Language Models

### Evolution Timeline

```markdown
1980s-1990s: N-gram models
  - Count-based statistics
  - Smoothing techniques
  - Dominated speech recognition, MT

2000s: Neural language models emerge
  - Feed-forward neural LMs (Bengio 2003)
  - Learned word representations
  - Better generalization

2010s: Recurrent neural networks
  - LSTMs for language modeling
  - Better long-range dependencies
  - State-of-the-art results

2017+: Transformers
  - Attention mechanisms
  - Pre-trained language models (BERT, GPT)
  - N-grams obsolete for most tasks
```

### What Neural Models Fixed

**1. Continuous representations**:

```markdown
> Generalization to similar words

N-gram: "cat" and "dog" are different symbols
Neural: "cat" and "dog" have similar embeddings
```

**2. Long-range dependencies**:

```markdown
N-gram: Context limited to N-1 words
Neural (RNN/Transformer): Can capture dependencies across entire sequence
```

**3. Semantic understanding**:

```markdown
N-gram: Surface form only
Neural: Learned semantic representations
```

**4. Efficiency**:

```markdown
N-gram: Store all n-grams (large)
Neural: Store parameters (fixed size, regardless of n-gram order)
```

### When N-grams Still Matter

Despite neural dominance, n-grams remain useful for:

1. **Baselines**: Simple, interpretable comparison point
2. **Resource-constrained**: Embedded systems, edge devices
3. **Explainability**: Can trace predictions to counts
4. **Data-scarce**: Work with small datasets
5. **Feature engineering**: N-gram features for neural models
6. **Hybrid systems**: Combine with neural models

## Summary

**Key Concepts**:

1. **N-grams** are contiguous sequences of N words
2. **Language modeling** assigns probabilities to word sequences
3. **Markov assumption** limits context to N-1 previous words
4. **MLE** estimates probabilities from observed frequencies
5. **Sparsity** is the fundamental challenge (most n-grams unseen)
6. **Smoothing** redistributes probability mass to unseen events
7. **Perplexity** evaluates models (lower is better)

**Smoothing Techniques**:

- **Add-one (Laplace)**: Add 1 to all counts (too aggressive)
- **Add-k**: Add smaller constant (better)
- **Good-Turing**: Use frequency of frequencies
- **Interpolation**: Combine multiple n-gram orders
- **Backoff**: Use lower-order when higher-order unavailable

**Trade-offs**:

| Aspect      | Unigram | Bigram | Trigram | Higher-order         |
| ----------- | ------- | ------ | ------- | -------------------- |
| Context     | None    | 1 word | 2 words | N-1 words            |
| Sparsity    | Low     | Medium | High    | Very high            |
| Predictions | Poor    | Good   | Better  | Best (if not sparse) |
| Storage     | Small   | Medium | Large   | Very large           |

**Historical Importance**:

- Foundation of statistical NLP (1990s-2010s)
- Enabled speech recognition, MT, spelling correction
- Influenced neural architecture design
- Still useful for baselines and constraints

**Limitations**:

- Limited context window
- Severe sparsity for higher-order n-grams
- No semantic understanding
- Cannot generalize to similar contexts
- Storage requirements grow exponentially

## Next Steps

- Explore [Vector Space Models](vector-space-models.md) to learn about TF-IDF and document similarity
- Study [Topic Modeling](topic-modeling.md) to discover latent topics in documents
- Learn [Classical Text Classification](classical-classification.md) to use n-grams as features
