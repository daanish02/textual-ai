# Text Preprocessing

## Table of Contents

- [Introduction](#introduction)
- [Why Preprocess Text?](#why-preprocess-text)
- [Lowercasing](#lowercasing)
- [Removing Punctuation and Special Characters](#removing-punctuation-and-special-characters)
- [Stemming](#stemming)
- [Lemmatization](#lemmatization)
- [Stopword Removal](#stopword-removal)
- [Text Normalization](#text-normalization)
- [Handling Noise and Artifacts](#handling-noise-and-artifacts)
- [Task-Specific Preprocessing](#task-specific-preprocessing)
- [Modern NLP and Preprocessing](#modern-nlp-and-preprocessing)
- [Building Preprocessing Pipelines](#building-preprocessing-pipelines)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Text preprocessing transforms raw, messy text into a clean, consistent format suitable for NLP tasks. While modern neural models require less preprocessing than traditional methods, understanding preprocessing remains essential for:

- **Feature engineering**: Classical NLP and machine learning
- **Data cleaning**: Removing noise that hurts performance
- **Normalization**: Reducing vocabulary and variation
- **Task optimization**: Tailoring text to specific needs

**The central question**: What to keep and what to discard?

This guide covers common preprocessing techniques, when to apply them, and how preprocessing choices affect downstream performance.

## Why Preprocess Text?

### The Problem with Raw Text

Raw text from the internet, social media, documents, or transcripts is messy:

```python
raw_text = """
Wow!!! This is AMAZING 😍😍😍
Best product ever!!! 10/10 would recommend 👍
Check out our website: https://example.com
Price: $99.99 (was $149.99) -- 33% OFF!!!
Email us: support@example.com for more info...
#BestDeal #MustBuy @FriendName you need this!
"""

print(raw_text)
```

**Issues**:

- Mixed case ("AMAZING" vs "this")
- Repeated punctuation ("!!!")
- Emojis and special symbols
- URLs and email addresses
- HTML or markdown artifacts
- Hashtags and mentions
- Numbers and prices
- Inconsistent spacing

### Goals of Preprocessing

1. **Reduce noise**: Remove irrelevant information
2. **Normalize**: Convert variations to standard form ("running" → "run")
3. **Reduce vocabulary**: Fewer unique tokens to learn
4. **Improve signal**: Focus on meaningful content
5. **Task alignment**: Format text for specific tasks

### Trade-offs

**Preprocessing is not free**:

- **Information loss**: Lowercasing loses "US" (United States) vs "us" (pronoun)
- **Semantic loss**: Stopword removal may hurt: "not good" → "good"
- **Context loss**: Punctuation carries meaning: "Let's eat, Grandma!" vs "Let's eat Grandma!"
- **Irreversibility**: Can't recover original text after aggressive preprocessing

**Principle**: Preprocess as little as necessary for your task.

## Lowercasing

Convert all text to lowercase to reduce vocabulary.

### Implementation

```python
def lowercase_text(text):
    """Convert text to lowercase."""
    return text.lower()

# Example
text = "The Quick BROWN Fox Jumps Over the Lazy DOG"
lowercased = lowercase_text(text)

print(f"Original:   {text}")
print(f"Lowercased: {lowercased}")

# Vocabulary impact
words_original = set(text.split())
words_lowercased = set(lowercased.split())

print(f"\nUnique words (original):   {len(words_original)}")
print(f"Unique words (lowercased): {len(words_lowercased)}")
```

**Output**:

```
Original:   The Quick BROWN Fox Jumps Over the Lazy DOG
Lowercased: the quick brown fox jumps over the lazy dog

Unique words (original):   9
Unique words (lowercased): 8
```

### When to Lowercase

**Lowercase when**:

- Task doesn't depend on case (sentiment analysis, topic classification)
- Using small datasets (reduces sparsity)
- Using bag-of-words or TF-IDF models
- Case is inconsistent in data

**Don't lowercase when**:

- Case carries meaning: "US" vs "us", "Apple" vs "apple"
- Named entity recognition (entities are often capitalized)
- Proper nouns matter
- Using pretrained models that preserve case (GPT-2, GPT-3)

### Case Study: Impact on Sentiment

```python
from collections import Counter

# Sentiment corpus (simplified)
reviews = [
    "This product is AMAZING!",
    "amazing quality and price",
    "Not amazing, pretty average",
]

# Word frequencies without lowercasing
words_original = []
for review in reviews:
    words_original.extend(review.split())

freq_original = Counter(words_original)
print("Without lowercasing:")
print(freq_original.most_common())

# Word frequencies with lowercasing
words_lowercased = []
for review in reviews:
    words_lowercased.extend(review.lower().split())

freq_lowercased = Counter(words_lowercased)
print("\nWith lowercasing:")
print(freq_lowercased.most_common())
```

**Output**:

```
Without lowercasing:
[('AMAZING!', 1), ('amazing', 2), ...]

With lowercasing:
[('amazing', 3), ...]  # "amazing" now unified
```

## Removing Punctuation and Special Characters

Remove or normalize punctuation and special characters.

### Basic Punctuation Removal

```python
import string
import re

def remove_punctuation(text):
    """Remove all punctuation from text."""
    return text.translate(str.maketrans('', '', string.punctuation))

# Example
text = "Hello, world! How are you? I'm fine."
cleaned = remove_punctuation(text)

print(f"Original: {text}")
print(f"Cleaned:  {cleaned}")
```

**Output**:

```
Original: Hello, world! How are you? I'm fine.
Cleaned:  Hello world How are you Im fine
```

### Selective Punctuation Removal

```python
def remove_punctuation_except(text, keep="'"):
    """Remove punctuation except specified characters."""
    punctuation_to_remove = string.punctuation
    for char in keep:
        punctuation_to_remove = punctuation_to_remove.replace(char, '')

    return text.translate(str.maketrans('', '', punctuation_to_remove))

# Example: Keep apostrophes for contractions
text = "I'm happy! Isn't that great?"
cleaned = remove_punctuation_except(text, keep="'")

print(f"Original: {text}")
print(f"Cleaned:  {cleaned}")
```

**Output**:

```
Original: I'm happy! Isn't that great?
Cleaned:  I'm happy Isn't that great
```

### Removing Special Characters and Numbers

```python
def remove_special_chars(text, keep_numbers=False):
    """Remove special characters, optionally keep numbers."""
    if keep_numbers:
        # Keep letters, numbers, and spaces
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    else:
        # Keep only letters and spaces
        return re.sub(r'[^a-zA-Z\s]', '', text)

# Examples
text = "Product #12 costs $99.99! Buy now @store123"

print(f"Original: {text}")
print(f"Remove all special: {remove_special_chars(text)}")
print(f"Keep numbers: {remove_special_chars(text, keep_numbers=True)}")
```

**Output**:

```
Original: Product #12 costs $99.99! Buy now @store123
Remove all special: Product  costs  Buy now store
Keep numbers: Product 12 costs 9999 Buy now store123
```

### When to Remove Punctuation

**Remove when**:

- Using bag-of-words models
- Punctuation is noisy
- Vocabulary reduction is priority

**Keep when**:

- Sentiment analysis ("Great!" vs "Great." different intensity)
- Question detection ("Is this good?" vs "This is good.")
- Using modern tokenizers (handle punctuation well)
- Punctuation carries semantic meaning

## Stemming

Reduce words to their root form by removing suffixes.

### Porter Stemmer

```python
from nltk.stem import PorterStemmer

# Initialize stemmer
stemmer = PorterStemmer()

# Examples
words = [
    "running", "runs", "ran", "runner",
    "easily", "easy",
    "arguing", "argued", "argues", "argument",
    "helping", "helped", "helper", "helpful",
]

print("Word       -> Stem")
print("-" * 25)
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:12} -> {stem}")
```

**Output**:

```
Word       -> Stem
-------------------------
running      -> run
runs         -> run
ran          -> ran        # Not perfect!
runner       -> runner     # Not reduced
easily       -> easili     # Not a real word
easy         -> easi       # Not a real word
arguing      -> argu       # Not a real word
argued       -> argu
argues       -> argu
argument     -> argument   # Not reduced
```

### Snowball Stemmer

```python
from nltk.stem import SnowballStemmer

# Snowball supports multiple languages
stemmer_en = SnowballStemmer("english")
stemmer_es = SnowballStemmer("spanish")

# English examples
words_en = ["running", "runs", "easily", "generously"]

print("English Snowball Stemmer:")
for word in words_en:
    print(f"{word:12} -> {stemmer_en.stem(word)}")

# Spanish examples
words_es = ["corriendo", "correr", "corrió"]

print("\nSpanish Snowball Stemmer:")
for word in words_es:
    print(f"{word:12} -> {stemmer_es.stem(word)}")
```

### Stemming in Context

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

def stem_text(text):
    """Stem all words in text."""
    tokens = word_tokenize(text.lower())
    stemmed = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed)

# Example
text = "The runners were running quickly through the quickly darkening forest"
stemmed = stem_text(text)

print(f"Original: {text}")
print(f"Stemmed:  {stemmed}")
```

**Output**:

```
Original: The runners were running quickly through the quickly darkening forest
Stemmed:  the runner were run quickli through the quickli darken forest
```

### Strengths and Weaknesses

**Strengths**:

- Fast and simple
- Reduces vocabulary size
- Language-specific variants available
- Doesn't require dictionary

**Weaknesses**:

- Produces non-words ("easili", "argu")
- Over-stemming: "university" and "universe" → "univers"
- Under-stemming: "data" and "datum" remain different
- Loses semantic nuance: "argue" and "argument" have different meanings

**When to use**:

- Information retrieval (search engines)
- Document clustering
- Classical text classification
- When vocabulary reduction is critical

**When not to use**:

- Modern neural NLP (models learn morphology)
- When preserving exact meaning matters
- Semantic similarity tasks

## Lemmatization

Reduce words to their dictionary form (lemma) using linguistic knowledge.

### Using WordNet Lemmatizer

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required data (run once)
# nltk.download('wordnet')
# nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Lemmatization requires part-of-speech tags
words_and_pos = [
    ("running", "v"),    # verb
    ("runs", "v"),       # verb
    ("ran", "v"),        # verb
    ("runner", "n"),     # noun
    ("easily", "r"),     # adverb
    ("easy", "a"),       # adjective
    ("better", "a"),     # adjective
    ("best", "a"),       # adjective
]

print("Word       POS  -> Lemma")
print("-" * 30)
for word, pos in words_and_pos:
    lemma = lemmatizer.lemmatize(word, pos=pos)
    print(f"{word:12} {pos:4} -> {lemma}")
```

**Output**:

```
Word       POS  -> Lemma
------------------------------
running      v    -> run
runs         v    -> run
ran          v    -> run         # Better than stemming!
runner       n    -> runner
easily       r    -> easily
easy         a    -> easy
better       a    -> good        # Recognizes irregular forms!
best         a    -> good
```

### POS-Aware Lemmatization

```python
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tag to WordNet POS tag."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default

def lemmatize_text(text):
    """Lemmatize text with POS tagging."""
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)

    lemmas = []
    for token, pos in pos_tags:
        wordnet_pos = get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
        lemmas.append(lemma)

    return ' '.join(lemmas)

# Example
text = "The striped bats are hanging on their feet for best"
lemmatized = lemmatize_text(text)

print(f"Original:    {text}")
print(f"Lemmatized:  {lemmatized}")
```

**Output**:

```
Original:    The striped bats are hanging on their feet for best
Lemmatized:  the striped bat be hang on their foot for good
```

### Stemming vs Lemmatization

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["studies", "studying", "flies", "flying", "better", "caring", "university"]

print(f"{'Word':<15} {'Stemming':<15} {'Lemmatization':<15}")
print("-" * 45)

for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, pos='v')  # Assuming verb
    print(f"{word:<15} {stem:<15} {lemma:<15}")
```

**Output**:

```
Word            Stemming        Lemmatization
---------------------------------------------
studies         studi           study
studying        studi           study
flies           fli             fly
flying          fli             fly
better          better          better
caring          car             care
university      univers         university
```

**Key differences**:

- Stemming: Crude chopping, produces non-words
- Lemmatization: Linguistic analysis, produces valid words

## Stopword Removal

Remove common words that carry little meaning.

### Basic Stopword Removal

```python
from nltk.corpus import stopwords
import nltk

# Download stopwords (run once)
# nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))

print(f"Number of English stopwords: {len(stop_words)}")
print(f"Sample stopwords: {list(stop_words)[:20]}")

def remove_stopwords(text, stop_words):
    """Remove stopwords from text."""
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Example
text = "This is a simple example showing how stopword removal works"
filtered = remove_stopwords(text, stop_words)

print(f"\nOriginal: {text}")
print(f"Filtered: {filtered}")
```

**Output**:

```
Number of English stopwords: 179
Sample stopwords: ['i', 'me', 'my', 'myself', 'we', 'our', ...]

Original: This is a simple example showing how stopword removal works
Filtered: simple example showing stopword removal works
```

### Custom Stopword Lists

```python
# Standard stopwords
nltk_stops = set(stopwords.words('english'))

# Add domain-specific stopwords
custom_stops = nltk_stops | {"please", "email", "website", "click"}

# Or create from scratch
minimal_stops = {"the", "a", "an", "is", "are", "was", "were"}

# Example
text = "Please click on our website to email us"

print(f"Original:        {text}")
print(f"NLTK stops:      {remove_stopwords(text, nltk_stops)}")
print(f"Custom stops:    {remove_stopwords(text, custom_stops)}")
print(f"Minimal stops:   {remove_stopwords(text, minimal_stops)}")
```

### Stopword Removal Impact

```python
from collections import Counter

corpus = [
    "The movie was absolutely amazing and wonderful",
    "This film was terrible and boring",
    "I loved this movie so much",
    "Worst film I have ever seen",
]

# Without stopword removal
all_words = []
for text in corpus:
    all_words.extend(word_tokenize(text.lower()))

print("Top words (no stopword removal):")
print(Counter(all_words).most_common(10))

# With stopword removal
content_words = []
stop_words = set(stopwords.words('english'))
for text in corpus:
    tokens = word_tokenize(text.lower())
    content_words.extend([w for w in tokens if w not in stop_words])

print("\nTop words (with stopword removal):")
print(Counter(content_words).most_common(10))
```

### When to Remove Stopwords

**Remove when**:

- Using bag-of-words or TF-IDF
- Memory/compute constrained
- Topic modeling or document similarity
- Stopwords dominate feature space

**Don't remove when**:

- Stopwords carry meaning: "not good" vs "good"
- Using neural models (they learn what to ignore)
- Sentiment analysis (negations are stopwords!)
- Question answering (question words are often stopwords)
- Modern language models (they need full context)

**Critical example**:

```python
# Stopword removal can destroy meaning!

text1 = "This is not bad"
text2 = "This is not good"

# After stopword removal (removing "is", "not")
# Both become: "This bad" or "This good"
# Meaning completely changed!
```

## Text Normalization

### Expanding Contractions

```python
import re

contractions_dict = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
}

def expand_contractions(text, contractions=contractions_dict):
    """Expand contractions in text."""
    contractions_pattern = re.compile('({})'.format('|'.join(contractions.keys())),
                                     flags=re.IGNORECASE|re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        expanded = contractions.get(match.lower())
        if expanded:
            return expanded
        return match

    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

# Examples
texts = [
    "I can't believe it's already here!",
    "We'll see what happens, won't we?",
    "They've been waiting, but I'd leave now.",
]

for text in texts:
    print(f"Original: {text}")
    print(f"Expanded: {expand_contractions(text)}")
    print()
```

### Handling Repetitions

```python
def remove_repetitions(text, max_repeat=2):
    """Remove character repetitions beyond max_repeat."""
    # Pattern: character repeated more than max_repeat times
    pattern = r'(.)\1{' + str(max_repeat) + ',}'

    # Replace with character repeated max_repeat times
    cleaned = re.sub(pattern, r'\1' * max_repeat, text)
    return cleaned

# Examples
texts = [
    "Wooooow this is amaaazing!!!",
    "Nooooooo wayyyyy",
    "Helloooooooo there",
]

for text in texts:
    print(f"Original: {text}")
    print(f"Cleaned:  {remove_repetitions(text, max_repeat=2)}")
```

**Output**:

```
Original: Wooooow this is amaaazing!!!
Cleaned:  Wooow this is amaazing!!!
```

### Normalizing Whitespace

```python
def normalize_whitespace(text):
    """Normalize whitespace to single spaces."""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

# Examples
texts = [
    "Too    many     spaces",
    "Leading   and trailing   spaces   ",
    "Multiple\n\nnewlines\n\nhere",
]

for text in texts:
    print(f"Original: '{text}'")
    print(f"Cleaned:  '{normalize_whitespace(text)}'")
    print()
```

### Unicode Normalization

```python
import unicodedata

def normalize_unicode(text, form='NFKC'):
    """Normalize unicode characters.

    Forms:
    - NFC: Canonical Decomposition, followed by Canonical Composition
    - NFD: Canonical Decomposition
    - NFKC: Compatibility Decomposition, followed by Canonical Composition
    - NFKD: Compatibility Decomposition
    """
    return unicodedata.normalize(form, text)

# Examples
text1 = "café"  # é as single character
text2 = "café"  # é as e + combining accent

print(f"Text 1: {text1} (length: {len(text1)})")
print(f"Text 2: {text2} (length: {len(text2)})")
print(f"Equal: {text1 == text2}")

# After normalization
norm1 = normalize_unicode(text1)
norm2 = normalize_unicode(text2)

print(f"\nNormalized 1: {norm1} (length: {len(norm1)})")
print(f"Normalized 2: {norm2} (length: {len(norm2)})")
print(f"Equal: {norm1 == norm2}")
```

## Handling Noise and Artifacts

### Removing URLs

```python
def remove_urls(text):
    """Remove URLs from text."""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

# Example
text = "Check out https://example.com and www.another-site.com for more info!"
cleaned = remove_urls(text)

print(f"Original: {text}")
print(f"Cleaned:  {cleaned}")
```

### Removing Email Addresses

```python
def remove_emails(text):
    """Remove email addresses from text."""
    email_pattern = r'\S+@\S+'
    return re.sub(email_pattern, '', text)

# Example
text = "Contact us at support@example.com or sales@company.org"
cleaned = remove_emails(text)

print(f"Original: {text}")
print(f"Cleaned:  {cleaned}")
```

### Removing HTML Tags

```python
from html.parser import HTMLParser

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_data(self):
        return ''.join(self.text)

def remove_html(text):
    """Remove HTML tags from text."""
    stripper = HTMLStripper()
    stripper.feed(text)
    return stripper.get_data()

# Example
html_text = "<p>This is a <b>bold</b> statement with <a href='link'>a link</a>.</p>"
cleaned = remove_html(html_text)

print(f"Original: {html_text}")
print(f"Cleaned:  {cleaned}")
```

**Output**:

```
Original: <p>This is a <b>bold</b> statement with <a href='link'>a link</a>.</p>
Cleaned:  This is a bold statement with a link.
```

### Removing Emojis

```python
def remove_emojis(text):
    """Remove emojis from text."""
    emoji_pattern = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              u"\U00002702-\U000027B0"
                              u"\U000024C2-\U0001F251"
                              "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Example
text = "I love this! 😍😍 Best product ever! 🎉👍"
cleaned = remove_emojis(text)

print(f"Original: {text}")
print(f"Cleaned:  {cleaned}")
```

### Removing Hashtags and Mentions

```python
def remove_hashtags(text):
    """Remove hashtags from text."""
    return re.sub(r'#\w+', '', text)

def remove_mentions(text):
    """Remove @mentions from text."""
    return re.sub(r'@\w+', '', text)

def remove_social_artifacts(text):
    """Remove hashtags and mentions."""
    text = remove_hashtags(text)
    text = remove_mentions(text)
    return text

# Example
text = "Great product! @company #BestDeal #MustBuy highly recommend"
cleaned = remove_social_artifacts(text)

print(f"Original: {text}")
print(f"Cleaned:  {cleaned}")
```

## Task-Specific Preprocessing

### Sentiment Analysis

```python
def preprocess_for_sentiment(text):
    """Preprocessing for sentiment analysis."""
    # Keep capitalization and punctuation (sentiment signals)
    # Remove URLs, emails, mentions
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_mentions(text)

    # Expand contractions (important for negations!)
    text = expand_contractions(text)

    # Normalize repetitions (but keep emphasis)
    text = remove_repetitions(text, max_repeat=2)

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text

# Example
review = "I can't believe how AMAZING this is!!! Woooow 😍 Best purchase ever! @friend you need this"
cleaned = preprocess_for_sentiment(review)

print(f"Original: {review}")
print(f"Cleaned:  {cleaned}")
```

### Information Retrieval / Search

```python
def preprocess_for_search(text):
    """Preprocessing for search/IR tasks."""
    # Lowercase (queries are often lowercase)
    text = text.lower()

    # Remove punctuation
    text = remove_punctuation(text)

    # Remove stopwords (focus on content words)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]

    # Stem (normalize morphological variants)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]

    return ' '.join(tokens)

# Example
query = "How to improve the performance of machine learning models?"
processed = preprocess_for_search(query)

print(f"Original: {query}")
print(f"Processed: {processed}")
```

### Named Entity Recognition

```python
def preprocess_for_ner(text):
    """Minimal preprocessing for NER (preserve capitalization!)."""
    # Keep capitalization (important for identifying entities)
    # Only normalize whitespace and remove obvious noise
    text = remove_urls(text)
    text = normalize_whitespace(text)

    return text

# Example
text = "John Smith works at Apple Inc. in New York    City."
cleaned = preprocess_for_ner(text)

print(f"Original: '{text}'")
print(f"Cleaned:  '{cleaned}'")
# Notice capitalization preserved!
```

### Topic Modeling

```python
def preprocess_for_topic_modeling(text):
    """Preprocessing for topic modeling."""
    # Lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    # Lemmatize (better than stemming for topics)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return tokens

# Example
text = "Machine learning models are trained on large datasets to recognize patterns."
processed = preprocess_for_topic_modeling(text)

print(f"Original: {text}")
print(f"Processed: {processed}")
```

## Modern NLP and Preprocessing

### The Neural Era: Less Preprocessing

Modern neural models (BERT, GPT) require **minimal preprocessing**:

```python
# Traditional NLP preprocessing (extensive)
def traditional_preprocessing(text):
    text = text.lower()
    text = remove_punctuation(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

# Modern neural NLP preprocessing (minimal)
def modern_preprocessing(text):
    # Just normalize whitespace!
    text = normalize_whitespace(text)
    # Tokenizer handles the rest
    return text

# Example
text = "I can't believe this AMAZING product! It's the best!!!"

print(f"Original: {text}")
print(f"Traditional: {traditional_preprocessing(text)}")
print(f"Modern: {modern_preprocessing(text)}")
```

**Why less preprocessing?**

1. **Learned representations**: Models learn what matters
2. **Subword tokenization**: Handles morphology automatically
3. **Context**: Models use surrounding context
4. **Task-agnostic**: Same preprocessing for all tasks
5. **Pretrained models**: Already trained on diverse text

### When to Still Preprocess

Even with modern models, preprocess to:

1. **Remove noise**: URLs, HTML, corrupted characters
2. **Handle artifacts**: Platform-specific formatting
3. **Normalize**: Whitespace, unicode
4. **Task-specific**: Domain requirements

```python
def minimal_modern_preprocessing(text):
    """Minimal preprocessing for modern NLP."""
    # Remove obvious noise
    text = remove_urls(text)
    text = remove_html(text)

    # Normalize unicode
    text = normalize_unicode(text)

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text
```

## Building Preprocessing Pipelines

### Composable Preprocessing Functions

```python
from typing import Callable, List

class TextPreprocessor:
    """Composable text preprocessing pipeline."""

    def __init__(self, steps: List[Callable[[str], str]]):
        self.steps = steps

    def __call__(self, text: str) -> str:
        """Apply all preprocessing steps."""
        for step in self.steps:
            text = step(text)
        return text

    def add_step(self, step: Callable[[str], str]):
        """Add a preprocessing step."""
        self.steps.append(step)

# Define preprocessing steps
def step_lowercase(text):
    return text.lower()

def step_remove_urls(text):
    return remove_urls(text)

def step_remove_punctuation(text):
    return remove_punctuation(text)

def step_remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered)

# Build pipeline
sentiment_pipeline = TextPreprocessor([
    step_remove_urls,
    normalize_whitespace,
])

search_pipeline = TextPreprocessor([
    step_lowercase,
    step_remove_punctuation,
    step_remove_stopwords,
])

# Use pipelines
text = "Check out https://example.com - This is AMAZING!!!"

print(f"Original: {text}")
print(f"Sentiment: {sentiment_pipeline(text)}")
print(f"Search: {search_pipeline(text)}")
```

### Scikit-learn Compatible Preprocessor

```python
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible text preprocessor."""

    def __init__(self, lowercase=True, remove_punct=True,
                 remove_stops=True, stemming=False):
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stops = remove_stops
        self.stemming = stemming

        if self.stemming:
            self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        """Fit method (no-op for preprocessing)."""
        return self

    def transform(self, X):
        """Transform texts."""
        return [self._preprocess(text) for text in X]

    def _preprocess(self, text):
        """Preprocess a single text."""
        if self.lowercase:
            text = text.lower()

        tokens = word_tokenize(text)

        if self.remove_punct:
            tokens = [w for w in tokens if w.isalpha()]

        if self.remove_stops:
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if w not in stop_words]

        if self.stemming:
            tokens = [self.stemmer.stem(w) for w in tokens]

        return ' '.join(tokens)

# Use in sklearn pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline([
    ('preprocess', TextPreprocessorTransformer(stemming=True)),
    ('vectorize', TfidfVectorizer()),
])

# Example
texts = [
    "This is a great product!",
    "I love this amazing item.",
    "Terrible experience, would not recommend.",
]

# Fit and transform
# X = pipeline.fit_transform(texts)
# print(X.shape)
```

## Summary

**Key Concepts**:

1. **Preprocessing** transforms raw text into clean, consistent format
2. **Lowercasing** reduces vocabulary but loses case information
3. **Punctuation removal** simplifies text but may lose meaning
4. **Stemming** crudely reduces words to roots (fast, imprecise)
5. **Lemmatization** reduces words to dictionary forms (slower, precise)
6. **Stopword removal** focuses on content words but may hurt semantic tasks
7. **Normalization** handles contractions, repetitions, unicode, whitespace
8. **Noise removal** cleans URLs, HTML, emojis, social media artifacts
9. **Task-specific preprocessing** tailors cleaning to specific needs

**Decision Framework**:

| Task           | Lowercase | Remove Punct | Stopwords | Stem/Lemma  | Keep               |
| -------------- | --------- | ------------ | --------- | ----------- | ------------------ |
| Sentiment      | Maybe     | Selective    | No        | No          | Emphasis, negation |
| Search/IR      | Yes       | Yes          | Yes       | Yes         | Content words      |
| NER            | No        | No           | No        | No          | Capitalization     |
| Topic Modeling | Yes       | Yes          | Yes       | Yes (lemma) | Content words      |
| Modern LLMs    | No        | No           | No        | No          | Everything!        |

**Modern NLP Guidelines**:

1. **Minimal preprocessing**: Neural models learn from raw text
2. **Clean obvious noise**: URLs, HTML, corrupted text
3. **Normalize basics**: Whitespace, unicode
4. **Preserve information**: Capitalization, punctuation carry meaning
5. **Task-specific choices**: Some tasks need more preprocessing
6. **Experiment**: Test with and without preprocessing

**Anti-patterns**:

- ❌ Over-preprocessing (throwing away useful signal)
- ❌ Removing negations before sentiment analysis
- ❌ Lowercasing for NER
- ❌ Aggressive stemming when lemmatization is affordable
- ❌ Applying same preprocessing to all tasks

## Next Steps

- Review [Tokenization](tokenization.md) to understand how preprocessing interacts with tokenization
- Explore [Linguistic Foundations](linguistic-foundations.md) to learn about POS tagging and NER
- Progress to [Classical NLP](../classical_nlp/) to see preprocessing in traditional methods
- Study [Embeddings](../embeddings/) to understand modern representation learning
- Learn [Language Models](../language_models/) to see why modern models need less preprocessing
