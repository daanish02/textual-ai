# Linguistic Foundations

## Table of Contents

- [Introduction](#introduction)
- [Linguistic Levels of Analysis](#linguistic-levels-of-analysis)
- [Part-of-Speech Tagging](#part-of-speech-tagging)
- [Named Entity Recognition](#named-entity-recognition)
- [Syntactic Parsing](#syntactic-parsing)
- [Morphology](#morphology)
- [Semantics](#semantics)
- [Pragmatics](#pragmatics)
- [Linguistic Knowledge in NLP Systems](#linguistic-knowledge-in-nlp-systems)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Natural language is structured at multiple levels -- from individual sounds to complex discourse. Understanding these linguistic structures helps you:

- **Interpret NLP techniques**: Why certain approaches work
- **Design better systems**: Leverage linguistic knowledge
- **Debug failures**: Understand where models struggle
- **Evaluate properly**: Know what capabilities to test

This guide covers core linguistic concepts that underpin NLP: morphology (word structure), syntax (sentence structure), semantics (meaning), and pragmatics (context-dependent meaning). We'll also explore practical NLP tasks built on linguistic foundations: POS tagging, NER, and parsing.

**Key insight**: While modern neural models learn linguistic patterns from data, understanding the underlying structures helps you work more effectively with these models.

## Linguistic Levels of Analysis

Language is organized hierarchically:

```
Discourse (conversation, documents)
    ↓
Pragmatics (context-dependent meaning)
    ↓
Semantics (meaning)
    ↓
Syntax (sentence structure)
    ↓
Morphology (word structure)
    ↓
Phonology (sound patterns)
    ↓
Phonetics (speech sounds)
```

**For NLP**, we primarily work with:

1. **Morphology**: Word formation (prefixes, suffixes, roots)
2. **Syntax**: Sentence structure (grammar, dependencies)
3. **Semantics**: Meaning (word sense, compositional semantics)
4. **Pragmatics**: Context and intent (discourse, speech acts)

### Example: Multiple Levels

```
Sentence: "The bank closes at 3pm."

Morphology:
  - "closes" = "close" (root) + "s" (3rd person singular)
  - "bank" = single morpheme

Syntax:
  - Subject: "The bank"
  - Predicate: "closes at 3pm"
  - Structure: NP (Det + N) + VP (V + PP)

Semantics:
  - "bank" = financial institution (not river bank)
  - "closes" = ceases operation (not physically closing)
  - Time specification: 3pm = 15:00

Pragmatics:
  - Informational: Stating a fact
  - Implicit: Regular closing time (not one-time event)
  - Context: Likely answering "When can I visit?"
```

## Part-of-Speech Tagging

**Part-of-Speech (POS) tagging** assigns grammatical categories (noun, verb, adjective, etc.) to each word.

### POS Tag Sets

#### Penn Treebank Tags (Common)

```
Major categories:
- NN:   Noun, singular          (cat, dog, house)
- NNS:  Noun, plural            (cats, dogs, houses)
- NNP:  Proper noun, singular   (John, London)
- NNPS: Proper noun, plural     (Americans)
- VB:   Verb, base form         (run, eat)
- VBD:  Verb, past tense        (ran, ate)
- VBG:  Verb, gerund            (running, eating)
- VBN:  Verb, past participle   (eaten, run)
- VBP:  Verb, non-3rd person    (run, eat)
- VBZ:  Verb, 3rd person        (runs, eats)
- JJ:   Adjective               (happy, big)
- JJR:  Adjective, comparative  (happier, bigger)
- JJS:  Adjective, superlative  (happiest, biggest)
- RB:   Adverb                  (quickly, very)
- IN:   Preposition             (in, on, at)
- DT:   Determiner              (the, a, an)
- CC:   Coordinating conjunction (and, or, but)
- PRP:  Personal pronoun        (I, you, he)
- TO:   to                      (to)
- .  :  Punctuation
```

### POS Tagging with NLTK

```python
import nltk
from nltk import word_tokenize, pos_tag

# Download required data (run once)
# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
tags = pos_tag(tokens)

print("Word-level POS tags:")
for word, tag in tags:
    print(f"{word:10} -> {tag}")

# Describe a tag
# nltk.help.upenn_tagset('NN')
```

**Output**:

```
Word-level POS tags:
The        -> DT
quick      -> JJ
brown      -> JJ
fox        -> NN
jumps      -> VBZ
over       -> IN
the        -> DT
lazy       -> JJ
dog        -> NN
```

### POS Tagging with spaCy

```python
import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

print("spaCy POS tags:")
print(f"{'Token':10} {'POS':6} {'Tag':6} {'Description'}")
print("-" * 50)
for token in doc:
    print(f"{token.text:10} {token.pos_:6} {token.tag_:6} {spacy.explain(token.tag_)}")
```

**Output**:

```
Token      POS    Tag    Description
--------------------------------------------------
The        DET    DT     determiner
quick      ADJ    JJ     adjective
brown      ADJ    JJ     adjective
fox        NOUN   NN     noun, singular
jumps      VERB   VBZ    verb, 3rd person singular
over       ADP    IN     preposition or subordinating conjunction
the        DET    DT     determiner
lazy       ADJ    JJ     adjective
dog        NOUN   NN     noun, singular
```

### Why POS Tagging Matters

1. **Word sense disambiguation**: "book" (noun) vs "book" (verb)
2. **Information extraction**: Extract noun phrases, verb phrases
3. **Parsing**: POS tags guide syntactic parsing
4. **Lemmatization**: Requires POS to correctly reduce words
5. **Feature engineering**: POS tags as features for classification

### POS Tagging Ambiguity

Many words have multiple possible POS tags:

```python
sentences = [
    "I will book the flight",     # book = verb
    "I read the book",             # book = noun
    "They can fish here",          # can = modal verb, fish = verb
    "They can the fish",           # can = verb, fish = noun
]

nlp = spacy.load("en_core_web_sm")

for sentence in sentences:
    doc = nlp(sentence)
    print(f"Sentence: {sentence}")
    for token in doc:
        print(f"  {token.text:8} -> {token.pos_:6} ({token.tag_})")
    print()
```

**Output**:

```
Sentence: I will book the flight
  I        -> PRON   (PRP)
  will     -> AUX    (MD)
  book     -> VERB   (VB)    # Verb!
  the      -> DET    (DT)
  flight   -> NOUN   (NN)

Sentence: I read the book
  I        -> PRON   (PRP)
  read     -> VERB   (VBD)
  the      -> DET    (DT)
  book     -> NOUN   (NN)    # Noun!
```

### POS Patterns for Extraction

Use POS tags to extract linguistic patterns:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

# Define patterns
# Pattern: Adjective + Noun
matcher = Matcher(nlp.vocab)
pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}]
matcher.add("ADJ_NOUN", [pattern])

text = "The quick brown fox chases the lazy dog through the dark forest"
doc = nlp(text)

matches = matcher(doc)
print("Adjective-Noun pairs:")
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"  {span.text}")
```

**Output**:

```
Adjective-Noun pairs:
  quick brown
  brown fox
  lazy dog
  dark forest
```

### Building POS Patterns for NP Extraction

```python
def extract_noun_phrases(text):
    """Extract noun phrases using POS patterns."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Simple pattern: (Det)? (Adj)* Noun+
    noun_phrases = []
    current_np = []

    for token in doc:
        if token.pos_ in ['DET', 'ADJ']:
            current_np.append(token.text)
        elif token.pos_ == 'NOUN':
            current_np.append(token.text)
            # End of NP
            if current_np:
                noun_phrases.append(' '.join(current_np))
                current_np = []
        else:
            current_np = []

    return noun_phrases

text = "The intelligent machine learning model processes large datasets efficiently"
nps = extract_noun_phrases(text)

print("Extracted noun phrases:")
for np in nps:
    print(f"  {np}")
```

## Named Entity Recognition

**Named Entity Recognition (NER)** identifies and classifies named entities (people, organizations, locations, dates, etc.) in text.

### Common Entity Types

```
PERSON:      People's names
ORG:         Organizations (companies, agencies)
GPE:         Geopolitical entities (countries, cities, states)
LOC:         Non-GPE locations (mountains, water bodies)
DATE:        Dates or periods
TIME:        Times smaller than a day
MONEY:       Monetary values
PERCENT:     Percentages
PRODUCT:     Product names
EVENT:       Named events
WORK_OF_ART: Titles of books, songs, etc.
LAW:         Named laws
LANGUAGE:    Named languages
```

### NER with spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = """
Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.
The company is now worth over $2 trillion and employs more than 150,000 people worldwide.
Tim Cook became CEO in August 2011.
"""

doc = nlp(text)

print("Named Entities:")
print(f"{'Entity':25} {'Type':15} {'Description'}")
print("-" * 70)

for ent in doc.ents:
    print(f"{ent.text:25} {ent.label_:15} {spacy.explain(ent.label_)}")
```

**Output**:

```
Named Entities:
Entity                    Type            Description
----------------------------------------------------------------------
Apple Inc.                ORG             Companies, agencies, institutions
Steve Jobs                PERSON          People, including fictional
Cupertino                 GPE             Countries, cities, states
California                GPE             Countries, cities, states
April 1, 1976             DATE            Absolute or relative dates
over $2 trillion          MONEY           Monetary values
more than 150,000         CARDINAL        Numerals not covered by other types
Tim Cook                  PERSON          People, including fictional
August 2011               DATE            Absolute or relative dates
```

### Visualizing NER

```python
from spacy import displacy

# Visualize entities
# displacy.serve(doc, style="ent")  # In Jupyter: displacy.render(doc, style="ent")

# Or get HTML
html = displacy.render(doc, style="ent", page=False)
# print(html)
```

### Custom NER Patterns

```python
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Pattern to match email addresses
email_pattern = [
    {"TEXT": {"REGEX": r"\w+"}},
    {"TEXT": "@"},
    {"TEXT": {"REGEX": r"\w+"}},
    {"TEXT": "."},
    {"TEXT": {"REGEX": r"\w+"}}
]

matcher.add("EMAIL", [email_pattern])

text = "Contact us at support@example.com or sales@company.org for more information"
doc = nlp(text)

matches = matcher(doc)
print("Found emails:")
for match_id, start, end in matches:
    print(f"  {doc[start:end]}")
```

### NER for Information Extraction

```python
def extract_company_info(text):
    """Extract structured information about companies."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    info = {
        'companies': [],
        'people': [],
        'locations': [],
        'dates': [],
        'money': []
    }

    for ent in doc.ents:
        if ent.label_ == 'ORG':
            info['companies'].append(ent.text)
        elif ent.label_ == 'PERSON':
            info['people'].append(ent.text)
        elif ent.label_ in ['GPE', 'LOC']:
            info['locations'].append(ent.text)
        elif ent.label_ == 'DATE':
            info['dates'].append(ent.text)
        elif ent.label_ == 'MONEY':
            info['money'].append(ent.text)

    return info

text = """
Tesla announced record profits of $3.2 billion in Q3 2023.
CEO Elon Musk stated that the company's Gigafactory in Austin, Texas
has increased production capacity by 40%.
"""

info = extract_company_info(text)

print("Extracted Information:")
for key, values in info.items():
    if values:
        print(f"{key.capitalize()}: {', '.join(values)}")
```

**Output**:

```
Extracted Information:
Companies: Tesla
People: Elon Musk
Locations: Austin, Texas
Dates: Q3 2023
Money: $3.2 billion
```

### NER Challenges

```python
# Ambiguity
ambiguous_text = "Washington is beautiful in spring"
# Washington = person, city, or state?

# Context-dependent
context_examples = [
    "I work at Apple",              # Company
    "I ate an apple",               # Fruit - not detected
    "New York is expensive",        # GPE
    "I love New York pizza",        # GPE (modifier)
]

nlp = spacy.load("en_core_web_sm")

for text in context_examples:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"Text: {text}")
    print(f"  Entities: {entities}")
```

## Syntactic Parsing

**Parsing** analyzes the grammatical structure of sentences.

### Dependency Parsing

Dependency parsing identifies relationships between words:

```
"The cat sat on the mat"

Dependency tree:
      sat (ROOT)
     / | \  \
   cat on  .
   /    |
  The  mat
       /
      the
```

**Dependencies**:

- sat → cat (nsubj: nominal subject)
- sat → on (prep: prepositional modifier)
- on → mat (pobj: object of preposition)
- cat → The (det: determiner)
- mat → the (det: determiner)

### Dependency Parsing with spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "The cat sat on the mat"
doc = nlp(text)

print("Dependency Parse:")
print(f"{'Token':10} {'Dependency':15} {'Head':10} {'Children'}")
print("-" * 60)

for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:10} {token.dep_:15} {token.head.text:10} {children}")
```

**Output**:

```
Token      Dependency      Head       Children
------------------------------------------------------------
The        det             cat        []
cat        nsubj           sat        ['The']
sat        ROOT            sat        ['cat', 'on', '.']
on         prep            sat        ['mat']
the        det             mat        []
mat        pobj            on         ['the']
.          punct           sat        []
```

### Visualizing Dependency Trees

```python
from spacy import displacy

text = "The intelligent model processes complex linguistic structures efficiently"
doc = nlp(text)

# Render dependency tree
# displacy.serve(doc, style="dep")  # Opens in browser
# In Jupyter: displacy.render(doc, style="dep")

# Text-based visualization
for token in doc:
    print("  " * token.i + f"└─ {token.text} ({token.dep_})")
```

### Common Dependency Relations

```python
# Core dependencies
examples = {
    'nsubj': "The dog barks",              # dog ← barks (nominal subject)
    'dobj': "I eat apples",                # apples ← eat (direct object)
    'iobj': "I gave him money",            # him ← gave (indirect object)
    'amod': "The red car",                 # red → car (adjectival modifier)
    'advmod': "She runs quickly",          # quickly → runs (adverbial modifier)
    'det': "The book",                     # The → book (determiner)
    'prep': "I sat on the chair",          # on ← sat (prepositional modifier)
    'pobj': "I sat on the chair",          # chair ← on (object of preposition)
    'conj': "I like cats and dogs",        # dogs ← and, and ← cats (conjunction)
}

nlp = spacy.load("en_core_web_sm")

for dep_type, text in examples.items():
    doc = nlp(text)
    print(f"\n{dep_type}: \"{text}\"")
    for token in doc:
        if token.dep_ == dep_type or dep_type in [child.dep_ for child in token.children]:
            print(f"  {token.text} ({token.dep_}) ← {token.head.text}")
```

### Extracting Relationships with Dependencies

```python
def extract_subject_verb_object(text):
    """Extract SVO triples from text."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    triples = []

    for token in doc:
        if token.pos_ == "VERB":
            subject = None
            obj = None

            # Find subject and object
            for child in token.children:
                if child.dep_ == "nsubj":
                    subject = child.text
                elif child.dep_ in ["dobj", "pobj"]:
                    obj = child.text

            if subject and obj:
                triples.append((subject, token.text, obj))

    return triples

texts = [
    "The cat chased the mouse",
    "Alice writes programs",
    "The company announced earnings",
]

for text in texts:
    triples = extract_subject_verb_object(text)
    print(f"Text: {text}")
    print(f"  SVO: {triples}")
```

### Constituency Parsing

Constituency parsing creates hierarchical phrase structure:

```
"The cat sat on the mat"

Constituency tree:
            S
         /     \
       NP       VP
      / \      /  \
    DT  NN   VBD  PP
    |   |    |   /  \
   The cat  sat IN  NP
                |  /  \
               on DT  NN
                  |   |
                 the mat
```

**Phrase types**:

- S: Sentence
- NP: Noun Phrase
- VP: Verb Phrase
- PP: Prepositional Phrase
- DT: Determiner
- NN: Noun
- VBD: Verb (past tense)

```python
# Constituency parsing with NLTK
import nltk
from nltk import Tree

# Note: constituency parsing requires specialized parsers
# Stanford Parser or Berkeley Parser

# Simplified demonstration
grammar = nltk.CFG.fromstring("""
  S -> NP VP
  NP -> DT NN | PRP
  VP -> VBD PP | VBZ NP
  PP -> IN NP
  DT -> 'the' | 'a'
  NN -> 'cat' | 'mat' | 'dog'
  VBD -> 'sat'
  VBZ -> 'chases'
  IN -> 'on'
  PRP -> 'I'
""")

parser = nltk.ChartParser(grammar)

sentence = "the cat sat on the mat".split()
for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()
```

## Morphology

**Morphology** studies word structure -- how words are formed from smaller units (morphemes).

### Morphemes

**Morpheme**: The smallest meaningful unit of language.

```
Word: "unhappiness"

Morphemes:
  un-      (prefix: negation)
  happy    (root: state of joy)
  -ness    (suffix: noun-forming)

Meaning: The state of not being happy
```

### Types of Morphemes

**Free morphemes**: Can stand alone as words

```
cat, dog, run, happy, quick
```

**Bound morphemes**: Must attach to other morphemes

```
Prefixes:  un-, re-, pre-, dis-
Suffixes:  -ing, -ed, -er, -ness, -ly
```

### Morphological Analysis

```python
def analyze_morphology(word):
    """Simple morphological analysis (heuristic)."""
    # Common prefixes and suffixes
    prefixes = ['un', 're', 'pre', 'dis', 'mis', 'anti', 'de']
    suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'tion', 'able', 'ful']

    morphemes = []
    root = word.lower()

    # Check for prefix
    for prefix in prefixes:
        if root.startswith(prefix):
            morphemes.append(f"{prefix}- (prefix)")
            root = root[len(prefix):]
            break

    # Check for suffix
    for suffix in suffixes:
        if root.endswith(suffix):
            morphemes.append(f"ROOT: {root[:-len(suffix)]}")
            morphemes.append(f"-{suffix} (suffix)")
            root = root[:-len(suffix)]
            return morphemes

    morphemes.append(f"ROOT: {root}")
    return morphemes

# Examples
words = ["unhappiness", "replay", "wonderful", "running", "disagree"]

for word in words:
    print(f"\n{word}:")
    for morpheme in analyze_morphology(word):
        print(f"  {morpheme}")
```

**Output**:

```
unhappiness:
  un- (prefix)
  ROOT: happi
  -ness (suffix)

replay:
  re- (prefix)
  ROOT: play

wonderful:
  ROOT: wonder
  -ful (suffix)
```

### Inflectional vs Derivational Morphology

**Inflectional** (grammatical function, same word class):

```
walk → walks (3rd person)
walk → walked (past tense)
walk → walking (progressive)

cat → cats (plural)

happy → happier (comparative)
happy → happiest (superlative)
```

**Derivational** (changes meaning or word class):

```
happy (adj) → happiness (noun)      # Class change
happy (adj) → unhappy (adj)         # Meaning change
act (verb) → action (noun)          # Class change
teach (verb) → teacher (noun)       # Class change
```

### Morphology in Different Languages

**English** (relatively simple):

```
run, runs, running, ran
```

**Spanish** (more complex):

```
hablar (to speak)
hablo, hablas, habla, hablamos, habláis, hablan (present)
hablé, hablaste, habló, hablamos, hablasteis, hablaron (past)
```

**Turkish** (highly agglutinative):

```
ev (house)
evler (houses)
evlerim (my houses)
evlerimde (in my houses)
evlerimdeyim (I am in my houses)
```

**Arabic** (root+pattern):

```
Root: k-t-b (writing)
kataba (he wrote)
kitāb (book)
maktab (office, place of writing)
kātib (writer)
```

## Semantics

**Semantics** studies meaning in language.

### Lexical Semantics (Word Meaning)

**Word sense**: Different meanings of the same word (polysemy)

```
"bank"
  1. Financial institution
  2. Edge of a river
  3. To tilt (airplane banks)

"mouse"
  1. Small rodent
  2. Computer input device
```

**Synonymy**: Different words, same meaning

```
big ≈ large ≈ huge ≈ enormous
```

**Antonymy**: Opposite meaning

```
hot ↔ cold
up ↔ down
```

**Hyponymy**: IS-A relationships

```
cat IS-A feline IS-A mammal IS-A animal
```

**Meronymy**: PART-OF relationships

```
wheel PART-OF car
```

### Compositional Semantics

Meaning of phrases/sentences from meanings of parts:

```
"red ball"
  red(x) ∧ ball(x)

"all cats are mammals"
  ∀x: cat(x) → mammal(x)

"some students passed"
  ∃x: student(x) ∧ passed(x)
```

**Non-compositional** (idioms):

```
"kick the bucket" ≠ literally kicking a bucket
                  = to die

"spill the beans" ≠ literally spilling beans
                  = to reveal a secret
```

### Semantic Roles

```
Sentence: "John gave Mary a book in the library"

Semantic roles:
  Agent:      John (who did it)
  Recipient:  Mary (who received)
  Theme:      book (what was given)
  Location:   library (where)
```

**Frame semantics example**:

```
[Giving] frame
  Donor:     who gives
  Recipient: who receives
  Theme:     what is given
  Time:      when
  Place:     where
```

### Semantic Similarity

```python
import spacy

nlp = spacy.load("en_core_web_md")  # Medium model with word vectors

# Semantic similarity using word embeddings
words = [
    ("king", "queen"),
    ("cat", "dog"),
    ("car", "automobile"),
    ("happy", "sad"),
    ("apple", "computer"),
]

for word1, word2 in words:
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    print(f"{word1:12} <-> {word2:12} = {similarity:.3f}")
```

**Output**:

```
king         <-> queen        = 0.762
cat          <-> dog          = 0.801
car          <-> automobile   = 0.672
happy        <-> sad          = 0.457
apple        <-> computer     = 0.386
```

## Pragmatics

**Pragmatics** studies how context affects meaning.

### Speech Acts

Utterances perform actions:

```
Assertive:  "It's raining"             (stating a fact)
Directive:  "Close the door"           (requesting action)
Commissive: "I promise to help"        (committing to action)
Expressive: "Congratulations!"         (expressing emotion)
Declarative: "I now pronounce you married" (changing reality)
```

### Implicature

What's implied beyond literal meaning:

```
A: "Do you want to go to the movies?"
B: "I have a lot of work to do"

Implicature: B is declining (though not explicitly stated)

A: "Can you pass the salt?"
Literal: Asking about ability
Pragmatic: Requesting action
```

### Reference and Deixis

**Anaphora** (referring back):

```
"John went to the store. He bought milk."
  "He" refers to "John"

"The cat chased the mouse. It was fast."
  "It" refers to "mouse" (or "cat" - ambiguous!)
```

**Deictic expressions** (context-dependent):

```
"I will see you tomorrow"
  "I" = speaker
  "you" = addressee
  "tomorrow" = day after utterance

"Come here"
  "here" = location relative to speaker
```

### Presupposition

Assumptions embedded in utterances:

```
"Have you stopped smoking?"
  Presupposes: You used to smoke

"The king of France is bald"
  Presupposes: France has a king (false!)

"Mary regrets failing the exam"
  Presupposes: Mary failed the exam
```

## Linguistic Knowledge in NLP Systems

### Traditional NLP: Explicit Linguistic Rules

```python
# Example: Rule-based NER (simplified)

def rule_based_ner(text):
    """Simple rule-based named entity recognition."""
    entities = []

    # Rule: Capitalized word followed by lowercase = likely proper noun
    words = text.split()
    for i, word in enumerate(words):
        if word[0].isupper():
            if i == 0 or (i > 0 and words[i-1][-1] in '.!?'):
                # Sentence start, skip
                continue
            # Likely a named entity
            entities.append((word, 'PROPER_NOUN'))

    return entities

text = "John visited Paris last summer. The Eiffel Tower was beautiful."
entities = rule_based_ner(text)
print(entities)
```

### Modern NLP: Learned Linguistic Patterns

```python
# Neural models learn linguistic patterns from data
# No explicit rules, but they capture linguistic structure

import spacy

nlp = spacy.load("en_core_web_sm")

text = "John visited Paris last summer"
doc = nlp(text)

# Model learned:
# - POS tags
# - Dependencies
# - Named entities
# All from data, not rules!

print("What the model learned:")
for token in doc:
    print(f"{token.text:10} POS={token.pos_:6} DEP={token.dep_:10} Head={token.head.text}")

print("\nEntities:")
for ent in doc.ents:
    print(f"  {ent.text} ({ent.label_})")
```

### Hybrid Approaches

```python
# Combine learned models with linguistic rules

def extract_company_acquisition_events(text):
    """Extract acquisition events using NER + dependency parsing."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    events = []

    # Look for pattern: ORG acquired/bought ORG
    for token in doc:
        if token.lemma_ in ['acquire', 'buy', 'purchase']:
            acquirer = None
            target = None

            # Find subject (acquirer)
            for child in token.children:
                if child.dep_ == 'nsubj' and child.ent_type_ == 'ORG':
                    acquirer = child.text

            # Find object (target)
            for child in token.children:
                if child.dep_ == 'dobj' and child.ent_type_ == 'ORG':
                    target = child.text

            if acquirer and target:
                events.append({
                    'acquirer': acquirer,
                    'target': target,
                    'verb': token.text
                })

    return events

text = "Microsoft acquired LinkedIn for $26 billion. Google bought YouTube in 2006."
events = extract_company_acquisition_events(text)

print("Acquisition events:")
for event in events:
    print(f"  {event['acquirer']} {event['verb']} {event['target']}")
```

### Linguistic Features in Modern Models

Modern language models implicitly learn linguistic structure:

```
BERT learns:
- Syntax (can predict syntactic relationships)
- Semantics (word sense, similarity)
- Some morphology (subword tokenization helps)

GPT learns:
- Sequential structure
- Long-range dependencies
- Compositional semantics

Evidence:
- Probing studies show linguistic information in hidden layers
- Attention heads specialize in syntactic relationships
- Models can be fine-tuned for linguistic tasks (POS, parsing, NER)
```

## Summary

**Key Concepts**:

1. **Linguistic levels**: Morphology → Syntax → Semantics → Pragmatics
2. **POS tagging**: Assigns grammatical categories to words
3. **NER**: Identifies and classifies named entities (people, places, organizations)
4. **Parsing**: Analyzes sentence structure (dependency or constituency trees)
5. **Morphology**: Studies word formation from morphemes
6. **Semantics**: Studies meaning (lexical, compositional)
7. **Pragmatics**: Studies context-dependent meaning (speech acts, implicature)

**Practical Skills**:

- Use spaCy or NLTK for POS tagging and NER
- Extract linguistic patterns using POS and dependency information
- Understand morphological structure for better text processing
- Recognize semantic relationships (synonymy, hyponymy)
- Consider pragmatic context for natural language understanding

**Modern NLP and Linguistics**:

- **Traditional NLP**: Explicit linguistic rules and features
- **Neural NLP**: Learned linguistic patterns from data
- **Hybrid approaches**: Combine learned models with linguistic constraints
- **LLMs**: Implicitly capture linguistic structure through training

**Why Linguistics Matters**:

1. **Debugging**: Understand why models fail on certain constructions
2. **Feature engineering**: Design better features for classical methods
3. **Evaluation**: Test for specific linguistic capabilities
4. **System design**: Leverage linguistic structure for extraction and parsing
5. **Intuition**: Develop intuition for what's hard and what's easy in NLP

**Limitations**:

- Linguistic analysis is not perfect (ambiguity, context-dependence)
- Different languages have different structures (not all concepts translate)
- Pragmatics is especially hard to capture computationally
- Neural models may capture patterns without explicit linguistic structure

## Next Steps

- Review [Tokenization](tokenization.md) to see how morphology affects tokenization
- Explore [Text Preprocessing](text-preprocessing.md) and understand when to preserve linguistic structure
- Progress to [Classical NLP](../classical_nlp/) to see linguistic features in traditional methods
- Study [Embeddings](../embeddings/) to understand how meaning is represented geometrically
- Learn [Language Models](../language_models/) to see how models capture linguistic patterns
