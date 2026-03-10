# Summarization

## Table of Contents

- [Introduction](#introduction)
- [Extractive vs Abstractive](#extractive-vs-abstractive)
- [Single-Document Summarization](#single-document-summarization)
- [Multi-Document Summarization](#multi-document-summarization)
- [Controlling Length and Style](#controlling-length-and-style)
- [Prompting Strategies](#prompting-strategies)
- [Domain-Specific Summarization](#domain-specific-summarization)
- [Evaluation Challenges](#evaluation-challenges)
- [Production Patterns](#production-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Summarization is the task of creating a shorter version of text while preserving the most important information. It helps readers quickly grasp key points without reading everything, making it crucial for information overload management.

```
Summarization Process:

Long Text → Identify Key Information → Generate Summary → Short Text

Types:
┌─────────────────────────────────────┐
│ Extractive: Select existing sentences│
│ Abstractive: Generate new sentences │
└─────────────────────────────────────┘

Example:
Input (500 words): "The company reported Q3 earnings today..."
Output (50 words): "Company beats earnings expectations with
                   15% revenue growth..."
```

**Why summarization matters**:

- **Information overload**: People need quick insights
- **Time savings**: Read summaries instead of full documents
- **Better decisions**: Key points surface faster
- **Scalability**: Process large document collections
- **Accessibility**: Complex texts become more digestible

This guide covers practical patterns for building robust summarization systems with LLMs.

## Extractive vs Abstractive

### Understanding the Difference

```python
def summarization_types_explained():
    """Explain extractive vs abstractive summarization."""

    print("Summarization Types:\n")
    print("="*80)

    print("""
EXTRACTIVE SUMMARIZATION
------------------------
Selects and combines existing sentences from the original text.

Characteristics:
  • Uses exact sentences from source
  • Guaranteed factual accuracy (no hallucination)
  • May feel choppy or disconnected
  • Limited by original phrasing
  • Faster and cheaper

Example:
Original: "The Federal Reserve announced a 0.25% interest rate hike today.
          This marks the third consecutive increase this year. Economists
          predict this will help combat inflation, which has reached 8.2%.
          However, some worry about potential recession risks."

Extractive Summary: "The Federal Reserve announced a 0.25% interest rate hike
                    today. Economists predict this will help combat inflation."

→ Directly copied sentences


ABSTRACTIVE SUMMARIZATION
--------------------------
Generates new sentences that capture the meaning of the original.

Characteristics:
  • Creates new phrasings
  • More natural and coherent
  • Can paraphrase and synthesize
  • Risk of hallucination
  • Requires powerful LLMs

Example:
Original: [same as above]

Abstractive Summary: "The Fed raised rates by 0.25% for the third time this
                     year to address 8.2% inflation, despite recession concerns."

→ New sentence synthesizing multiple points


WHEN TO USE EACH
----------------

Use Extractive when:
  • Factual accuracy is critical (legal, medical, financial)
  • Need to preserve exact wording
  • Working with structured documents
  • Limited computational resources

Use Abstractive when:
  • Natural-sounding summaries are important
  • Need to synthesize information from multiple sources
  • Audience needs simplified language
  • Have access to powerful LLMs
""")

summarization_types_explained()
```

### Implementing Extractive Summarization

```python
class ExtractiveSummarizer:
    """Extract important sentences from text."""

    def __init__(self, num_sentences=3):
        """
        Initialize extractive summarizer.

        Args:
            num_sentences: Number of sentences to extract
        """
        self.num_sentences = num_sentences

    def split_sentences(self, text):
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        import re

        # Simple sentence splitting (can use spacy for better results)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def score_sentences(self, sentences):
        """
        Score sentences by importance.

        Args:
            sentences: List of sentences

        Returns:
            List of (sentence, score) tuples
        """
        scores = []

        for sent in sentences:
            score = 0

            # Heuristic 1: Length (not too short, not too long)
            words = sent.split()
            if 10 <= len(words) <= 30:
                score += 1

            # Heuristic 2: Position (first sentences often important)
            if sentences.index(sent) < 3:
                score += 0.5

            # Heuristic 3: Keywords (simple approach)
            important_words = ['announced', 'reported', 'found', 'concluded']
            if any(word in sent.lower() for word in important_words):
                score += 0.5

            # Heuristic 4: Numbers (often contain key facts)
            import re
            if re.search(r'\d+', sent):
                score += 0.3

            scores.append((sent, score))

        return scores

    def summarize_extractive(self, text):
        """
        Create extractive summary.

        Args:
            text: Input text

        Returns:
            Summary string
        """
        # Split into sentences
        sentences = self.split_sentences(text)

        if len(sentences) <= self.num_sentences:
            return text

        # Score sentences
        scored = self.score_sentences(sentences)

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top N sentences
        top_sentences = scored[:self.num_sentences]

        # Sort by original order to maintain flow
        original_order = sorted(
            top_sentences,
            key=lambda x: sentences.index(x[0])
        )

        # Join sentences
        summary = ' '.join([sent for sent, score in original_order])

        return summary

    def summarize_with_llm(self, text):
        """
        Use LLM for extractive summarization (more sophisticated).

        Args:
            text: Input text

        Returns:
            Summary string
        """
        sentences = self.split_sentences(text)

        if len(sentences) <= self.num_sentences:
            return text

        # Number sentences for easy selection
        numbered_sentences = '\n'.join([
            f"{i+1}. {sent}"
            for i, sent in enumerate(sentences)
        ])

        prompt = f"""
Select the {self.num_sentences} most important sentences from this text.
Output ONLY the sentence numbers (comma-separated), nothing else.

Text:
{numbered_sentences}

Most important sentence numbers:"""

        response = llm.generate(prompt, temperature=0)

        # Parse response
        import re
        numbers = re.findall(r'\d+', response)
        selected_indices = [int(n) - 1 for n in numbers[:self.num_sentences]]

        # Extract selected sentences in original order
        selected_indices.sort()
        summary = ' '.join([sentences[i] for i in selected_indices
                           if i < len(sentences)])

        return summary

# Example Usage
print("\n" + "="*80)
print("Extractive Summarization Example\n")

text = """
The Federal Reserve announced a 0.25% interest rate hike today, marking
the third consecutive increase this year. The decision comes as inflation
reaches 8.2%, the highest level in four decades. Economists predict this
move will help combat rising prices. However, some analysts worry about
potential recession risks. Stock markets initially fell on the news but
recovered by end of day. Fed Chair Jerome Powell stated that the central
bank remains committed to price stability. Consumer spending has shown
signs of cooling in recent months. The housing market has been particularly
affected by higher rates.
"""

summarizer = ExtractiveSummarizer(num_sentences=3)

print("Original Text:")
print(text)
print("\nExtractive Summary:")
print(summarizer.summarize_extractive(text))
```

### Implementing Abstractive Summarization

```python
class AbstractiveSummarizer:
    """Generate new sentences summarizing text."""

    def __init__(self, max_length=100):
        """
        Initialize abstractive summarizer.

        Args:
            max_length: Maximum summary length in words
        """
        self.max_length = max_length

    def summarize_basic(self, text):
        """
        Basic abstractive summarization.

        Args:
            text: Input text

        Returns:
            Summary string
        """
        word_count = len(text.split())

        prompt = f"""
Summarize this text in approximately {self.max_length} words.

Text:
{text}

Summary:"""

        summary = llm.generate(prompt, temperature=0.3)

        return summary.strip()

    def summarize_with_focus(self, text, focus=None):
        """
        Abstractive summarization with specific focus.

        Args:
            text: Input text
            focus: Aspect to focus on (e.g., "financial impact", "key decisions")

        Returns:
            Summary string
        """
        if focus:
            prompt = f"""
Summarize this text in approximately {self.max_length} words, focusing on: {focus}

Text:
{text}

Summary:"""
        else:
            prompt = f"""
Summarize this text in approximately {self.max_length} words.

Text:
{text}

Summary:"""

        summary = llm.generate(prompt, temperature=0.3)

        return summary.strip()

    def summarize_bullet_points(self, text, num_points=5):
        """
        Summarize as bullet points.

        Args:
            text: Input text
            num_points: Number of bullet points

        Returns:
            List of bullet points
        """
        prompt = f"""
Summarize this text as {num_points} concise bullet points. Each point should be one sentence.

Text:
{text}

Bullet points:"""

        response = llm.generate(prompt, temperature=0.3)

        # Parse bullet points
        import re
        lines = response.strip().split('\n')
        bullets = []

        for line in lines:
            # Remove bullet markers
            line = re.sub(r'^[-•*\d.)\s]+', '', line).strip()
            if line:
                bullets.append(line)

        return bullets[:num_points]

    def summarize_progressive(self, text, lengths=[50, 100, 200]):
        """
        Generate summaries at different lengths.

        Args:
            text: Input text
            lengths: List of word counts for summaries

        Returns:
            Dictionary mapping lengths to summaries
        """
        summaries = {}

        for length in lengths:
            prompt = f"""
Summarize this text in exactly {length} words.

Text:
{text}

{length}-word summary:"""

            summary = llm.generate(prompt, temperature=0.3)
            summaries[length] = summary.strip()

        return summaries

# Example Usage
print("\n\n" + "="*80)
print("Abstractive Summarization Example\n")

text = """
Researchers at MIT have developed a new quantum computer chip that operates
at room temperature, a breakthrough that could revolutionize the field.
Traditional quantum computers require extremely cold temperatures near
absolute zero, making them expensive and difficult to maintain. The new
chip uses a novel approach with synthetic diamond materials that maintain
quantum states even at normal temperatures. The team demonstrated successful
quantum calculations with a 64-qubit system. This advancement could
accelerate quantum computing adoption in commercial applications. The
research was published in Nature Physics and has garnered significant
attention from tech companies. However, experts note that practical
applications are still years away. The chip's architecture could be scaled
to thousands of qubits, potentially matching or exceeding the capabilities
of current superconducting quantum computers.
"""

summarizer = AbstractiveSummarizer(max_length=50)

print("Original Text:")
print(text)

print("\n\nBasic Summary:")
print(summarizer.summarize_basic(text))

print("\n\nFocused Summary (technical details):")
print(summarizer.summarize_with_focus(text, focus="technical details"))

print("\n\nBullet Point Summary:")
bullets = summarizer.summarize_bullet_points(text, num_points=4)
for i, bullet in enumerate(bullets, 1):
    print(f"{i}. {bullet}")
```

## Single-Document Summarization

### Length-Controlled Summarization

```python
class LengthControlledSummarizer:
    """Summarizer with precise length control."""

    def __init__(self):
        pass

    def summarize_by_words(self, text, word_count):
        """
        Summarize to exact word count.

        Args:
            text: Input text
            word_count: Target number of words

        Returns:
            Summary string
        """
        prompt = f"""
Summarize this text in EXACTLY {word_count} words. Be precise with the count.

Text:
{text}

{word_count}-word summary:"""

        summary = llm.generate(prompt, temperature=0)

        # Verify and adjust if needed
        actual_count = len(summary.split())

        if actual_count > word_count * 1.1:  # More than 10% over
            # Try again with stricter instruction
            prompt = f"""
You MUST write exactly {word_count} words, no more, no less.

Text:
{text}

{word_count}-word summary:"""
            summary = llm.generate(prompt, temperature=0)

        return summary.strip()

    def summarize_by_compression_ratio(self, text, ratio=0.25):
        """
        Summarize to a compression ratio.

        Args:
            text: Input text
            ratio: Target compression ratio (0.25 = 25% of original length)

        Returns:
            Summary string
        """
        original_words = len(text.split())
        target_words = int(original_words * ratio)

        return self.summarize_by_words(text, target_words)

    def summarize_by_sentences(self, text, num_sentences):
        """
        Summarize to exact number of sentences.

        Args:
            text: Input text
            num_sentences: Target number of sentences

        Returns:
            Summary string
        """
        prompt = f"""
Summarize this text in EXACTLY {num_sentences} sentences.
Each sentence should be complete and informative.

Text:
{text}

{num_sentences}-sentence summary:"""

        summary = llm.generate(prompt, temperature=0)

        return summary.strip()

    def summarize_by_percentage(self, text, percentage):
        """
        Summarize to percentage of original.

        Args:
            text: Input text
            percentage: Percentage of original length (e.g., 25 for 25%)

        Returns:
            Summary string
        """
        ratio = percentage / 100
        return self.summarize_by_compression_ratio(text, ratio)

# Example
print("\n\n" + "="*80)
print("Length-Controlled Summarization Example\n")

text = """
Climate change continues to accelerate, with 2023 being the warmest year on
record. Global temperatures have risen 1.2°C above pre-industrial levels.
Scientists warn that without immediate action, we could exceed the 1.5°C
threshold set by the Paris Agreement within the next decade. Extreme weather
events are becoming more frequent and severe. The economic costs of climate-
related disasters exceeded $200 billion last year. Renewable energy adoption
is growing, with solar and wind now cheaper than fossil fuels in many regions.
However, the transition is not happening fast enough to meet climate goals.
Governments and corporations are under pressure to accelerate decarbonization
efforts.
"""

summarizer = LengthControlledSummarizer()

print("Original:", len(text.split()), "words\n")

print("25-word summary:")
print(summarizer.summarize_by_words(text, 25))

print("\n\n3-sentence summary:")
print(summarizer.summarize_by_sentences(text, 3))

print("\n\n25% compression:")
print(summarizer.summarize_by_percentage(text, 25))
```

### Aspect-Based Summarization

```python
def aspect_based_summarization():
    """Summarize focusing on specific aspects."""

    print("\n\nAspect-Based Summarization:\n")
    print("="*80)

    class AspectSummarizer:
        """Summarize with focus on specific aspects."""

        def summarize_aspects(self, text, aspects):
            """
            Generate summaries for different aspects.

            Args:
                text: Input text
                aspects: List of aspects to focus on

            Returns:
                Dictionary mapping aspects to summaries
            """
            summaries = {}

            for aspect in aspects:
                prompt = f"""
Summarize this text focusing ONLY on: {aspect}

If the text doesn't contain information about this aspect, say "Not mentioned".

Text:
{text}

Summary ({aspect}):"""

                summary = llm.generate(prompt, temperature=0)
                summaries[aspect] = summary.strip()

            return summaries

        def summarize_multi_aspect(self, text, aspects, words_per_aspect=30):
            """
            Create structured multi-aspect summary.

            Args:
                text: Input text
                aspects: List of aspects
                words_per_aspect: Words per aspect summary

            Returns:
                Structured summary string
            """
            aspect_list = ', '.join(aspects)

            prompt = f"""
Create a structured summary covering these aspects: {aspect_list}

For each aspect, write approximately {words_per_aspect} words.
Use this format:

{aspects[0]}: [summary]
{aspects[1]}: [summary]
...

Text:
{text}

Structured Summary:"""

            summary = llm.generate(prompt, temperature=0.3)

            return summary.strip()

    # Example
    print("\nExample: Product Review Summarization\n")

    review = """
I purchased this laptop three months ago and have mixed feelings. The build
quality is excellent - the aluminum chassis feels premium and sturdy. The
keyboard is comfortable for long typing sessions. However, the battery life
is disappointing, lasting only 4-5 hours with normal use, far below the
advertised 10 hours. Performance is good for everyday tasks and light gaming.
The screen is bright and sharp. Customer service was unhelpful when I
reported the battery issue. At $1,200, it's overpriced given the battery
problem. Shipping was fast, arrived in 2 days. Overall, good hardware let
down by poor battery and service.
"""

    summarizer = AspectSummarizer()

    aspects = [
        "build quality",
        "battery life",
        "performance",
        "price/value",
        "customer service"
    ]

    print("Aspect Summaries:")
    summaries = summarizer.summarize_aspects(review, aspects)

    for aspect, summary in summaries.items():
        print(f"\n{aspect.upper()}:")
        print(f"  {summary}")

aspect_based_summarization()
```

## Multi-Document Summarization

### Synthesizing Multiple Sources

```python
class MultiDocumentSummarizer:
    """Summarize multiple related documents."""

    def __init__(self, max_length=200):
        """
        Initialize multi-document summarizer.

        Args:
            max_length: Maximum summary length in words
        """
        self.max_length = max_length

    def summarize_multiple(self, documents, titles=None):
        """
        Summarize multiple documents together.

        Args:
            documents: List of document texts
            titles: Optional list of document titles

        Returns:
            Combined summary
        """
        # Format documents
        if titles:
            formatted = '\n\n'.join([
                f"Document {i+1} ({title}):\n{doc}"
                for i, (doc, title) in enumerate(zip(documents, titles))
            ])
        else:
            formatted = '\n\n'.join([
                f"Document {i+1}:\n{doc}"
                for i, doc in enumerate(documents)
            ])

        prompt = f"""
Summarize these related documents together in approximately {self.max_length} words.
Synthesize common themes and highlight key differences.

{formatted}

Combined summary:"""

        summary = llm.generate(prompt, temperature=0.3)

        return summary.strip()

    def summarize_with_attribution(self, documents, titles):
        """
        Summarize with source attribution.

        Args:
            documents: List of document texts
            titles: List of document titles

        Returns:
            Summary with attributions
        """
        formatted = '\n\n'.join([
            f"Source {i+1} ({title}):\n{doc}"
            for i, (doc, title) in enumerate(zip(documents, titles))
        ])

        prompt = f"""
Summarize these documents in approximately {self.max_length} words.
When mentioning information, cite the source number (e.g., "According to Source 1...").

{formatted}

Summary with attributions:"""

        summary = llm.generate(prompt, temperature=0.3)

        return summary.strip()

    def find_commonalities_and_differences(self, documents):
        """
        Identify shared themes and differences.

        Args:
            documents: List of document texts

        Returns:
            Dictionary with 'common' and 'different' summaries
        """
        formatted = '\n\n'.join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(documents)
        ])

        prompt = f"""
Analyze these documents and identify:
1. Common themes (what all documents agree on)
2. Key differences (where documents disagree or emphasize different points)

{formatted}

Analysis:
Common themes:
Different perspectives:"""

        response = llm.generate(prompt, temperature=0.3)

        # Parse response
        parts = response.split('Different perspectives:')
        common = parts[0].replace('Common themes:', '').strip()
        different = parts[1].strip() if len(parts) > 1 else ""

        return {
            'common': common,
            'different': different
        }

    def summarize_timeline(self, documents, dates):
        """
        Summarize documents in chronological order.

        Args:
            documents: List of document texts
            dates: List of dates corresponding to documents

        Returns:
            Chronological summary
        """
        # Sort by date
        sorted_docs = sorted(zip(dates, documents), key=lambda x: x[0])

        formatted = '\n\n'.join([
            f"{date}:\n{doc}"
            for date, doc in sorted_docs
        ])

        prompt = f"""
Create a chronological summary showing how events unfolded over time.
Highlight key developments and changes.

Timeline:
{formatted}

Chronological summary:"""

        summary = llm.generate(prompt, temperature=0.3)

        return summary.strip()

# Example
print("\n\n" + "="*80)
print("Multi-Document Summarization Example\n")

docs = [
    """Company A reported strong Q3 results with 15% revenue growth.
    CEO stated that new product launches drove the increase. Stock price
    rose 5% after the announcement.""",

    """Company A's earnings beat expectations, analysts say. Revenue of
    $2.5B exceeded forecasts by 8%. However, profit margins declined
    slightly due to increased marketing spend.""",

    """Investors react positively to Company A earnings. The company raised
    full-year guidance. Some analysts question sustainability of growth
    given macroeconomic headwinds."""
]

titles = ["Press Release", "Financial Times", "Bloomberg"]

summarizer = MultiDocumentSummarizer(max_length=80)

print("Combined Summary:")
print(summarizer.summarize_multiple(docs, titles))

print("\n\nWith Attribution:")
print(summarizer.summarize_with_attribution(docs, titles))

print("\n\nCommonalities and Differences:")
analysis = summarizer.find_commonalities_and_differences(docs)
print(f"Common: {analysis['common']}")
print(f"Different: {analysis['different']}")
```

### Conflict Resolution

```python
def handle_conflicting_information():
    """Handle contradictory information across documents."""

    print("\n\nHandling Conflicting Information:\n")
    print("="*80)

    print("""
When documents contradict each other, strategies include:

1. ACKNOWLEDGE DISAGREEMENT
   "Source A reports X while Source B reports Y"

2. PRESENT MULTIPLE PERSPECTIVES
   "According to analysts, growth is either strong (Source A) or
    concerning (Source B) depending on metrics used"

3. INDICATE UNCERTAINTY
   "Reports vary on the exact figure, ranging from 10-15%"

4. PRIORITIZE BY CREDIBILITY
   If one source is clearly more authoritative

5. NOTE TEMPORAL DIFFERENCES
   "Initial reports suggested X, but later updates indicated Y"
""")

    class ConflictAwareSummarizer:
        """Summarizer that handles contradictions."""

        def summarize_with_conflicts(self, documents, sources):
            """
            Summarize acknowledging conflicts.

            Args:
                documents: List of document texts
                sources: List of source names

            Returns:
                Summary handling conflicts
            """
            formatted = '\n\n'.join([
                f"{source}:\n{doc}"
                for source, doc in zip(sources, documents)
            ])

            prompt = f"""
Summarize these documents. If sources contradict each other, explicitly
note the disagreement and present both perspectives.

Use phrases like:
- "According to [Source], ... while [Source] reports..."
- "Sources disagree on..."
- "Estimates range from..."

{formatted}

Summary with conflicts noted:"""

            summary = llm.generate(prompt, temperature=0.3)

            return summary.strip()

    # Example
    print("\n\nExample:\n")

    docs = [
        "Unemployment fell to 3.5% in October, the lowest in 50 years.",
        "October unemployment rate remained at 3.7%, unchanged from September.",
        "Labor market shows mixed signals as unemployment ticked up to 3.9%."
    ]

    sources = ["Source A", "Source B", "Source C"]

    summarizer = ConflictAwareSummarizer()
    summary = summarizer.summarize_with_conflicts(docs, sources)

    print("Documents present conflicting unemployment figures.")
    print(f"\nConflict-Aware Summary:\n{summary}")

handle_conflicting_information()
```

## Controlling Length and Style

### Style Transfer in Summaries

```python
class StyledSummarizer:
    """Generate summaries in different styles."""

    def __init__(self):
        pass

    def summarize_for_audience(self, text, audience):
        """
        Tailor summary for specific audience.

        Args:
            text: Input text
            audience: Target audience (e.g., "executives", "technical experts", "general public")

        Returns:
            Audience-appropriate summary
        """
        audience_prompts = {
            "executives": "Focus on business impact, key metrics, and strategic implications. Be concise and actionable.",
            "technical": "Include technical details, methodologies, and specific findings. Use technical terminology.",
            "general": "Use simple language, avoid jargon, and explain concepts clearly.",
            "academic": "Maintain formal tone, include methodology, and be precise with terminology.",
            "journalists": "Lead with most newsworthy angle, include relevant context and quotes if present."
        }

        style_instruction = audience_prompts.get(audience, "")

        prompt = f"""
Summarize this text for {audience}.

Style guide: {style_instruction}

Text:
{text}

Summary for {audience}:"""

        summary = llm.generate(prompt, temperature=0.3)

        return summary.strip()

    def summarize_with_tone(self, text, tone):
        """
        Generate summary with specific tone.

        Args:
            text: Input text
            tone: Desired tone (e.g., "formal", "casual", "enthusiastic", "neutral")

        Returns:
            Summary with specified tone
        """
        prompt = f"""
Summarize this text in a {tone} tone.

Text:
{text}

{tone.capitalize()} summary:"""

        summary = llm.generate(prompt, temperature=0.4)

        return summary.strip()

    def summarize_formats(self, text):
        """
        Generate summaries in different formats.

        Args:
            text: Input text

        Returns:
            Dictionary of summaries in different formats
        """
        formats = {}

        # Paragraph format
        prompt = f"Summarize in one concise paragraph:\n\n{text}\n\nSummary:"
        formats['paragraph'] = llm.generate(prompt, temperature=0.3).strip()

        # Bullet points
        prompt = f"Summarize as 5 bullet points:\n\n{text}\n\nBullets:"
        formats['bullets'] = llm.generate(prompt, temperature=0.3).strip()

        # Tweet format
        prompt = f"Summarize in tweet format (280 characters max):\n\n{text}\n\nTweet:"
        formats['tweet'] = llm.generate(prompt, temperature=0.3).strip()

        # Headline
        prompt = f"Write a headline summarizing the main point:\n\n{text}\n\nHeadline:"
        formats['headline'] = llm.generate(prompt, temperature=0.3).strip()

        return formats

# Example
print("\n\n" + "="*80)
print("Styled Summarization Example\n")

text = """
A new study published in Nature shows that a novel mRNA vaccine platform
can be rapidly adapted to target emerging viral threats. Researchers
demonstrated that vaccines could be designed, tested, and manufactured
within 60 days of identifying a new pathogen's genetic sequence. The
platform uses lipid nanoparticles to deliver mRNA encoding viral antigens,
triggering an immune response. In animal trials, the vaccines showed 90%
efficacy against multiple test viruses. The technology could revolutionize
pandemic response by dramatically reducing vaccine development time.
Traditional vaccine development typically takes years. The team is now
seeking regulatory approval for human trials.
"""

summarizer = StyledSummarizer()

print("For Executives:")
print(summarizer.summarize_for_audience(text, "executives"))

print("\n\nFor General Public:")
print(summarizer.summarize_for_audience(text, "general"))

print("\n\nFor Technical Experts:")
print(summarizer.summarize_for_audience(text, "technical"))

print("\n\nDifferent Formats:")
formats = summarizer.summarize_formats(text)
for fmt, summary in formats.items():
    print(f"\n{fmt.upper()}:")
    print(summary)
```

### Iterative Refinement

```python
def iterative_summarization():
    """Iteratively refine summaries to meet requirements."""

    print("\n\nIterative Summarization:\n")
    print("="*80)

    class IterativeSummarizer:
        """Refine summaries through iteration."""

        def summarize_with_feedback(self, text, requirements):
            """
            Generate and refine summary based on requirements.

            Args:
                text: Input text
                requirements: List of requirements (e.g., "include statistics", "emphasize impact")

            Returns:
                Refined summary
            """
            # Initial summary
            summary = llm.generate(
                f"Summarize this text:\n\n{text}\n\nSummary:",
                temperature=0.3
            ).strip()

            # Check requirements and refine
            requirements_text = '\n'.join([f"- {req}" for req in requirements])

            refine_prompt = f"""
Review this summary and improve it to meet these requirements:

{requirements_text}

Original text:
{text}

Current summary:
{summary}

Improved summary meeting all requirements:"""

            refined = llm.generate(refine_prompt, temperature=0.3).strip()

            return refined

        def compress_progressively(self, text, target_length):
            """
            Progressively compress summary to target length.

            Args:
                text: Input text
                target_length: Target word count

            Returns:
                Compressed summary
            """
            # Start with larger summary
            current_length = target_length * 2

            summary = llm.generate(
                f"Summarize in {current_length} words:\n\n{text}\n\nSummary:",
                temperature=0.3
            ).strip()

            # Progressively compress
            while current_length > target_length:
                current_length = int(current_length * 0.75)
                if current_length < target_length:
                    current_length = target_length

                prompt = f"""
Compress this summary to {current_length} words while keeping the most important information.

Current summary:
{summary}

{current_length}-word version:"""

                summary = llm.generate(prompt, temperature=0.3).strip()

            return summary

    print("""
Iterative refinement strategies:

1. GENERATE → REVIEW → REFINE
   - Create initial summary
   - Check against requirements
   - Refine to meet all criteria

2. PROGRESSIVE COMPRESSION
   - Start with longer summary
   - Iteratively reduce length
   - Preserve key information

3. QUALITY CHECKS
   - Factual accuracy
   - Length requirements
   - Style and tone
   - Completeness

4. USER FEEDBACK LOOP
   - Generate summary
   - Collect user feedback
   - Regenerate with feedback incorporated
""")

iterative_summarization()
```

## Prompting Strategies

### Advanced Prompting Techniques

```python
def advanced_summarization_prompts():
    """Advanced prompting techniques for better summaries."""

    print("\n\nAdvanced Prompting Strategies:\n")
    print("="*80)

    strategies = {
        "Chain-of-Thought": {
            "description": "Guide model to think step-by-step",
            "template": """
Summarize this text. Think step-by-step:

1. What are the 3 most important points?
2. How are they connected?
3. What's the main takeaway?

Text: {text}

Step-by-step analysis:
1. Important points:
2. Connections:
3. Main takeaway:

Final summary:""",
            "when_to_use": "Complex texts requiring careful analysis"
        },

        "Few-Shot": {
            "description": "Provide examples of good summaries",
            "template": """
Summarize texts as shown in examples:

Example 1:
Text: [long text about economics]
Summary: [good summary]

Example 2:
Text: [long text about technology]
Summary: [good summary]

Now summarize:
Text: {text}
Summary:""",
            "when_to_use": "Specific summary style needed"
        },

        "Constraints-First": {
            "description": "State all constraints upfront",
            "template": """
Summarize with these constraints:
- Exactly 50 words
- Include at least one number
- Use past tense
- Avoid jargon

Text: {text}

Summary (meeting all constraints):""",
            "when_to_use": "Multiple requirements must be met"
        },

        "Question-Guided": {
            "description": "Answer key questions to form summary",
            "template": """
Answer these questions about the text, then combine into a summary:

1. What happened?
2. Who was involved?
3. When did it occur?
4. Why is it significant?

Text: {text}

Answers:
1.
2.
3.
4.

Summary:""",
            "when_to_use": "Structured information needed"
        },

        "Hierarchical": {
            "description": "Summarize at multiple levels",
            "template": """
Create a hierarchical summary:

1. One-sentence summary (10-15 words)
2. Short summary (50 words)
3. Detailed summary (150 words)

Text: {text}

1. One-sentence:
2. Short:
3. Detailed:""",
            "when_to_use": "Multiple summary lengths needed"
        }
    }

    for name, details in strategies.items():
        print(f"\n{name}:")
        print(f"  Description: {details['description']}")
        print(f"  When to use: {details['when_to_use']}")
        print(f"\n  Template:")
        for line in details['template'].strip().split('\n'):
            print(f"    {line}")

advanced_summarization_prompts()
```

### Prompt Engineering Tips

```python
def summarization_prompt_tips():
    """Tips for effective summarization prompts."""

    print("\n\nSummarization Prompt Engineering Tips:\n")
    print("="*80)

    print("""
1. BE SPECIFIC ABOUT LENGTH
   ❌ "Write a short summary"
   ✓ "Summarize in exactly 50 words"

2. SPECIFY WHAT TO INCLUDE
   ❌ "Summarize this article"
   ✓ "Summarize focusing on: causes, impacts, and solutions"

3. DEFINE THE FORMAT
   ❌ "Make it concise"
   ✓ "Create a 3-bullet-point summary"

4. SET THE TONE/AUDIENCE
   ❌ "Summarize this"
   ✓ "Summarize for a non-technical audience using simple language"

5. HANDLE EDGE CASES
   ✓ "If the text is too short to summarize, output: [TEXT TOO SHORT]"

6. REQUEST STRUCTURED OUTPUT
   ✓ "Output as JSON: {'summary': '...', 'key_points': [...]}"

7. USE EXAMPLES FOR CONSISTENCY
   ✓ Show 2-3 examples of desired summary style

8. SPECIFY PRESERVATION REQUIREMENTS
   ✓ "Preserve all numerical data and dates in the summary"

9. CONTROL ABSTRACTION LEVEL
   ✓ "Focus on high-level themes, not specific details"
   ✓ "Include specific examples and data points"

10. PREVENT HALLUCINATION
    ✓ "Only include information present in the text"
    ✓ "If uncertain, omit rather than guess"
""")

    print("\n\nCommon Pitfalls:\n")
    print("""
❌ Vague length requirements → Inconsistent lengths
❌ No audience specification → Inappropriate level of detail
❌ Missing format guidance → Unpredictable structure
❌ No examples → Style drift
❌ Ignoring hallucination → Fabricated facts
""")

summarization_prompt_tips()
```

## Domain-Specific Summarization

### Specialized Domain Patterns

```python
def domain_specific_patterns():
    """Patterns for summarizing domain-specific content."""

    print("\n\nDomain-Specific Summarization Patterns:\n")
    print("="*80)

    domains = {
        "Scientific Papers": {
            "structure": "Background, Methods, Results, Conclusions",
            "focus": "Research question, methodology, key findings, implications",
            "template": """
Summarize this scientific paper using this structure:

1. Research Question: What problem does this address?
2. Methods: How was the study conducted?
3. Key Findings: What were the main results?
4. Implications: Why does this matter?

Paper: {text}

Structured Summary:
1. Research Question:
2. Methods:
3. Key Findings:
4. Implications:"""
        },

        "Legal Documents": {
            "structure": "Parties, Issue, Ruling, Reasoning",
            "focus": "Key facts, legal issues, decision, precedent",
            "template": """
Summarize this legal document:

1. Parties: Who is involved?
2. Issue: What is the legal question?
3. Ruling: What was decided?
4. Reasoning: Why was this decision made?

Document: {text}

Legal Summary:
1. Parties:
2. Issue:
3. Ruling:
4. Reasoning:"""
        },

        "News Articles": {
            "structure": "Who, What, When, Where, Why, How",
            "focus": "Main event, key players, timeline, context",
            "template": """
Summarize this news article answering:

1. What happened?
2. Who is involved?
3. When and where?
4. Why is it significant?
5. What's next?

Article: {text}

News Summary:"""
        },

        "Medical Records": {
            "structure": "Chief Complaint, History, Examination, Assessment, Plan",
            "focus": "Symptoms, diagnosis, treatment, follow-up",
            "template": """
Summarize this medical record:

1. Chief Complaint: Why did patient present?
2. Key Findings: What was discovered?
3. Assessment: What is the diagnosis?
4. Plan: What treatment is recommended?

Record: {text}

Medical Summary:
1. Chief Complaint:
2. Key Findings:
3. Assessment:
4. Plan:"""
        },

        "Financial Reports": {
            "structure": "Performance, Key Metrics, Outlook",
            "focus": "Revenue, profit, changes, guidance",
            "template": """
Summarize this financial report:

1. Performance: How did the company perform?
2. Key Metrics: What are the important numbers?
3. Changes: What changed vs. last period?
4. Outlook: What's the future guidance?

Report: {text}

Financial Summary:
1. Performance:
2. Key Metrics:
3. Changes:
4. Outlook:"""
        },

        "Meeting Notes": {
            "structure": "Decisions, Action Items, Next Steps",
            "focus": "What was decided, who does what, when",
            "template": """
Summarize these meeting notes:

1. Key Decisions: What was decided?
2. Action Items: What needs to be done? (assign to whom)
3. Next Steps: What happens next?
4. Open Issues: What remains unresolved?

Notes: {text}

Meeting Summary:
1. Key Decisions:
2. Action Items:
3. Next Steps:
4. Open Issues:"""
        }
    }

    for domain, details in domains.items():
        print(f"\n{domain}:")
        print(f"  Structure: {details['structure']}")
        print(f"  Focus: {details['focus']}")
        print(f"\n  Template:")
        for line in details['template'].strip().split('\n'):
            print(f"    {line}")

domain_specific_patterns()
```

### Preserving Domain-Specific Information

```python
class DomainAwareSummarizer:
    """Summarizer that preserves domain-specific information."""

    def __init__(self, domain):
        """
        Initialize domain-aware summarizer.

        Args:
            domain: Domain type (e.g., "medical", "legal", "financial")
        """
        self.domain = domain
        self.preservation_rules = {
            "medical": ["drug names", "dosages", "diagnoses", "symptoms"],
            "legal": ["case numbers", "parties", "statutes", "dates"],
            "financial": ["numbers", "percentages", "companies", "metrics"],
            "scientific": ["methods", "sample sizes", "p-values", "conclusions"],
            "technical": ["version numbers", "specifications", "error codes"]
        }

    def summarize_preserving_key_info(self, text):
        """
        Summarize while preserving domain-specific information.

        Args:
            text: Input text

        Returns:
            Summary string
        """
        items_to_preserve = self.preservation_rules.get(self.domain, [])
        items_text = ", ".join(items_to_preserve)

        prompt = f"""
Summarize this {self.domain} text.

IMPORTANT: Preserve all {items_text} exactly as they appear.

Text:
{text}

Summary (preserving {items_text}):"""

        summary = llm.generate(prompt, temperature=0.2)

        return summary.strip()

# Example
print("\n\n" + "="*80)
print("Domain-Aware Summarization Example\n")

medical_text = """
Patient presented with acute chest pain radiating to left arm. ECG showed
ST-elevation in leads II, III, and aVF. Troponin I elevated at 0.8 ng/mL
(normal <0.04). Diagnosed with inferior STEMI. Administered aspirin 325mg,
clopidogrel 600mg loading dose, and heparin 5000 units IV bolus. Patient
taken for emergency cardiac catheterization. RCA 95% occluded, successfully
stented with drug-eluting stent. Post-procedure troponin peaked at 12.3 ng/mL.
Discharged on aspirin 81mg daily, ticagrelor 90mg BID, atorvastatin 80mg daily,
and metoprolol 25mg BID. Follow-up in 2 weeks.
"""

summarizer = DomainAwareSummarizer(domain="medical")

print("Medical Text Summary (preserving drug names, dosages, diagnoses):")
print(summarizer.summarize_preserving_key_info(medical_text))
```

## Evaluation Challenges

### Measuring Summary Quality

```python
def summary_quality_metrics():
    """Metrics for evaluating summary quality."""

    print("\n\nSummary Quality Metrics:\n")
    print("="*80)

    print("""
1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

   Measures n-gram overlap between generated and reference summaries

   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence

   Pros: Automatic, fast, correlates with human judgments
   Cons: Requires reference summaries, doesn't measure meaning

2. BLEU (Bilingual Evaluation Understudy)

   Originally for translation, sometimes used for summarization
   Measures precision of n-gram matches

   Pros: Automatic, widely used
   Cons: Precision-focused, less suitable for summarization

3. BERTScore

   Uses BERT embeddings to measure semantic similarity

   Pros: Captures meaning better than n-gram metrics
   Cons: Slower, still needs reference summaries

4. FACTUAL CONSISTENCY

   Does summary contain only facts from source?

   Methods:
   - NLI models (check if summary entailed by source)
   - QA-based (ask questions, verify answers)
   - LLM-based fact-checking

   Critical for production use!

5. COVERAGE

   Does summary include all important information?

   Can measure by:
   - Checking if key entities mentioned
   - Verifying main topics covered
   - Human judgment

6. COHERENCE

   Is summary well-structured and readable?

   Measured by:
   - Human ratings
   - Discourse coherence models
   - Readability scores

7. CONCISENESS

   Is summary appropriately brief?

   - Length vs. information density
   - Redundancy detection
   - Compression ratio
""")

    print("\n\nPractical Evaluation Approach:\n")
    print("""
For production systems:

1. AUTOMATIC METRICS (fast, scalable)
   - Factual consistency (NLI-based)
   - Length adherence
   - Basic quality checks

2. SAMPLING + HUMAN REVIEW
   - Review random sample (e.g., 100 summaries/week)
   - Rate on: accuracy, completeness, coherence
   - Track trends over time

3. USER FEEDBACK
   - "Was this summary helpful?" thumbs up/down
   - Optional detailed feedback
   - A/B test different approaches

4. ERROR ANALYSIS
   - Categorize failures
   - Identify patterns
   - Prioritize fixes
""")

summary_quality_metrics()
```

### Factuality Checking

```python
class FactualityChecker:
    """Check if summary is factually consistent with source."""

    def __init__(self):
        pass

    def check_factuality_nli(self, source, summary):
        """
        Check factuality using NLI (Natural Language Inference).

        Args:
            source: Original text
            summary: Generated summary

        Returns:
            Boolean indicating if summary is consistent
        """
        # Split summary into claims
        import re
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]

        inconsistent_claims = []

        for claim in sentences:
            # Check if claim is entailed by source
            prompt = f"""
Does the source text support this claim?

Source: {source}

Claim: {claim}

Answer (YES/NO):"""

            response = llm.generate(prompt, temperature=0)

            if response.strip().upper() != 'YES':
                inconsistent_claims.append(claim)

        is_consistent = len(inconsistent_claims) == 0

        return {
            'is_consistent': is_consistent,
            'inconsistent_claims': inconsistent_claims,
            'consistency_score': 1 - (len(inconsistent_claims) / len(sentences))
        }

    def check_factuality_qa(self, source, summary):
        """
        Check factuality using QA approach.

        Args:
            source: Original text
            summary: Generated summary

        Returns:
            Factuality report
        """
        # Extract key facts from summary
        prompt = f"""
Extract 5 key factual claims from this summary.

Summary: {summary}

Key facts (numbered list):"""

        response = llm.generate(prompt, temperature=0)

        # Parse facts
        import re
        facts = re.findall(r'\d+\.\s*(.+?)(?:\n|$)', response)

        # Verify each fact against source
        verified_facts = []
        unverified_facts = []

        for fact in facts:
            verify_prompt = f"""
Is this fact supported by the source text?

Source: {source}

Fact: {fact}

Answer (YES/NO):"""

            verification = llm.generate(verify_prompt, temperature=0)

            if verification.strip().upper() == 'YES':
                verified_facts.append(fact)
            else:
                unverified_facts.append(fact)

        return {
            'verified_facts': verified_facts,
            'unverified_facts': unverified_facts,
            'accuracy': len(verified_facts) / len(facts) if facts else 1.0
        }

# Example
print("\n\n" + "="*80)
print("Factuality Checking Example\n")

source = "Company reported Q3 revenue of $2.5 billion, up 15% year-over-year."

good_summary = "The company's Q3 revenue reached $2.5 billion, showing 15% growth."
bad_summary = "The company's Q3 revenue reached $3.5 billion, showing 25% growth."

checker = FactualityChecker()

print("Checking GOOD summary:")
result = checker.check_factuality_nli(source, good_summary)
print(f"Consistent: {result['is_consistent']}")
print(f"Score: {result['consistency_score']:.2f}")

print("\n\nChecking BAD summary:")
result = checker.check_factuality_nli(source, bad_summary)
print(f"Consistent: {result['is_consistent']}")
print(f"Inconsistent claims: {result['inconsistent_claims']}")
print(f"Score: {result['consistency_score']:.2f}")
```

## Production Patterns

### Scalable Summarization Pipeline

```python
class ProductionSummarizer:
    """Production-ready summarization system."""

    def __init__(self, cache_enabled=True):
        """
        Initialize production summarizer.

        Args:
            cache_enabled: Whether to cache summaries
        """
        self.cache_enabled = cache_enabled
        self.cache = {}

    def summarize_with_pipeline(self, text, options=None):
        """
        Complete summarization pipeline with all production features.

        Args:
            text: Input text
            options: Dict with summarization options

        Returns:
            Dict with summary and metadata
        """
        import time
        import hashlib

        start_time = time.time()

        # Default options
        if options is None:
            options = {}

        max_length = options.get('max_length', 100)
        style = options.get('style', 'neutral')
        check_factuality = options.get('check_factuality', True)

        # Check cache
        cache_key = hashlib.md5(
            (text + str(options)).encode()
        ).hexdigest()

        if self.cache_enabled and cache_key in self.cache:
            cached = self.cache[cache_key]
            cached['cache_hit'] = True
            return cached

        # Validate input
        if len(text.strip()) < 50:
            return {
                'error': 'Text too short to summarize',
                'summary': None
            }

        if len(text.split()) > 10000:
            return {
                'error': 'Text too long (max 10,000 words)',
                'summary': None
            }

        # Generate summary
        try:
            summary = self.generate_summary(text, max_length, style)
        except Exception as e:
            return {
                'error': f'Generation failed: {str(e)}',
                'summary': None
            }

        # Quality checks
        result = {
            'summary': summary,
            'metadata': {
                'length': len(summary.split()),
                'compression_ratio': len(summary.split()) / len(text.split()),
                'latency_ms': int((time.time() - start_time) * 1000),
                'cache_hit': False
            }
        }

        # Factuality check
        if check_factuality:
            checker = FactualityChecker()
            factuality = checker.check_factuality_nli(text, summary)
            result['metadata']['factuality_score'] = factuality['consistency_score']

            if not factuality['is_consistent']:
                result['warnings'] = [
                    f"Potential factual inconsistencies: {factuality['inconsistent_claims']}"
                ]

        # Cache result
        if self.cache_enabled:
            self.cache[cache_key] = result

        return result

    def generate_summary(self, text, max_length, style):
        """Generate summary (placeholder for actual implementation)."""
        prompt = f"Summarize in {max_length} words with {style} style:\n\n{text}\n\nSummary:"
        return llm.generate(prompt, temperature=0.3).strip()

    def batch_summarize(self, texts, options=None):
        """
        Batch summarization for efficiency.

        Args:
            texts: List of texts to summarize
            options: Summarization options

        Returns:
            List of summaries
        """
        results = []

        for text in texts:
            result = self.summarize_with_pipeline(text, options)
            results.append(result)

        return results

# Example
print("\n\n" + "="*80)
print("Production Pipeline Example\n")

summarizer = ProductionSummarizer(cache_enabled=True)

text = """
Artificial intelligence research has made remarkable progress in recent years.
Large language models can now perform complex reasoning and generate human-like
text. However, challenges remain in areas like factual accuracy, bias mitigation,
and computational efficiency. Researchers are exploring new architectures and
training methods to address these limitations. The field is rapidly evolving with
new breakthroughs announced regularly.
"""

options = {
    'max_length': 30,
    'style': 'neutral',
    'check_factuality': True
}

result = summarizer.summarize_with_pipeline(text, options)

print(f"Summary: {result['summary']}")
print(f"\nMetadata:")
for key, value in result['metadata'].items():
    print(f"  {key}: {value}")

if 'warnings' in result:
    print(f"\nWarnings: {result['warnings']}")
```

## Summary

**Summarization Overview**:

```
Long Text → Summarization → Short Text

Types:
  • Extractive: Select existing sentences (factually safe)
  • Abstractive: Generate new sentences (more natural)
  • Hybrid: Combine both approaches
```

**Key Distinctions**:

| Aspect      | Extractive              | Abstractive    |
| ----------- | ----------------------- | -------------- |
| Output      | Exact sentences         | New sentences  |
| Accuracy    | High (no hallucination) | Risk of errors |
| Fluency     | Can be choppy           | Natural flow   |
| Compression | Limited                 | Flexible       |
| Speed       | Faster                  | Slower         |

**Common Applications**:

- News summarization (headlines, articles)
- Document summarization (reports, papers)
- Meeting summarization (notes → action items)
- Email/chat summarization (conversations)
- Multi-document synthesis (research, reviews)

**Best Practices**:

1. **Length Control**: Be specific ("50 words" not "short")
2. **Style Specification**: Define audience and tone
3. **Structure**: Request specific format (bullets, paragraphs)
4. **Factuality**: Verify against source, avoid hallucination
5. **Domain Adaptation**: Preserve domain-specific information
6. **Multi-Document**: Handle conflicts, attribute sources

**Prompting Strategies**:

- Basic: "Summarize in X words"
- Chain-of-thought: Guide step-by-step thinking
- Aspect-based: Focus on specific aspects
- Question-guided: Answer key questions
- Hierarchical: Multiple levels of detail

**Evaluation Metrics**:

- ROUGE (n-gram overlap with reference)
- Factual consistency (verify against source)
- Coverage (all key points included)
- Coherence (well-structured, readable)
- Length adherence (meets requirements)

**Production Considerations**:

- **Input validation**: Check length, format
- **Caching**: Store summaries for repeated content
- **Factuality checking**: Verify accuracy before serving
- **Quality monitoring**: Track metrics, user feedback
- **Fallback strategies**: Handle edge cases gracefully
- **Batch processing**: Optimize for throughput

**Common Pitfalls**:

- Hallucination → Verify facts, use extractive when critical
- Length inconsistency → Use specific word counts
- Loss of key information → Check coverage
- Poor structure → Request specific format
- Style mismatch → Define audience and tone

**Domain-Specific Considerations**:

| Domain     | Key Focus            | Preserve                 |
| ---------- | -------------------- | ------------------------ |
| Scientific | Methods, findings    | Numbers, methodology     |
| Legal      | Ruling, reasoning    | Parties, dates, statutes |
| Medical    | Diagnosis, treatment | Drugs, dosages, symptoms |
| Financial  | Performance, metrics | Numbers, percentages     |
| News       | Who, what, when, why | Facts, quotes            |

**Key Takeaways**:

- Choose extractive for factual safety, abstractive for fluency
- Be specific with length, style, and format requirements
- Always verify factuality, especially for critical applications
- Adapt approach to domain and use case
- Monitor quality continuously in production
- Cache aggressively to reduce costs and latency
- Provide multiple summary lengths for flexibility

## Next Steps

- Learn [Information Extraction](information-extraction.md) for structured data
- Study [Question Answering](question-answering.md) for targeted information
- Apply summarization in [Conversational AI](conversational-ai.md)
- Master [Text Generation](text-generation.md) for controlled content creation
- Implement [Semantic Search](semantic-search.md) to find relevant documents
- Review [Evaluation Methods](../evaluation/) for quality assessment
- Explore [RAG](../retrieval-augmented-generation/) for retrieval-based summarization
