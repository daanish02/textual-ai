# Few-Shot Learning and Examples

## Table of Contents

- [Introduction](#introduction)
- [What is Few-Shot Learning?](#what-is-few-shot-learning)
- [How Few-Shot Learning Works](#how-few-shot-learning-works)
- [Example Selection Strategies](#example-selection-strategies)
- [Example Quality Factors](#example-quality-factors)
- [Example Ordering and Format](#example-ordering-and-format)
- [Balancing Number of Examples](#balancing-number-of-examples)
- [Few-Shot vs Zero-Shot vs Fine-Tuning](#few-shot-vs-zero-shot-vs-fine-tuning)
- [Domain-Specific Few-Shot](#domain-specific-few-shot)
- [Few-Shot Failure Modes](#few-shot-failure-modes)
- [Advanced Few-Shot Techniques](#advanced-few-shot-techniques)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Few-shot learning** is the practice of teaching language models new tasks through a small number of examples provided in the prompt. Instead of explaining what to do, you show the model what you want through demonstrations.

```
Zero-Shot (Tell):
"Translate this to French: Hello"

Few-Shot (Show):
English: Goodbye → French: Au revoir
English: Thank you → French: Merci
English: Hello → French: 
```

**Key insight**: Examples are often more effective than instructions alone. The model learns the pattern from demonstrations and applies it to new inputs.

```
Performance Comparison:

Zero-Shot:   ████████░░ 60%  (instruction only)
Few-Shot:    ██████████ 85%  (instruction + 3 examples)
Fine-Tuned:  ███████████ 95% (task-specific training)
                ↑
         Sweet spot: Few-shot learning
         • No training required
         • Fast iteration
         • Good performance
```

This guide covers how to effectively use examples in prompts to maximize LLM performance.

## What is Few-Shot Learning?

### The Concept

```python
def few_shot_concept():
    """Understanding few-shot learning fundamentals."""
    
    print("Few-Shot Learning Explained:\n")
    
    print("Definition:")
    print("  Learning a task from a small number of examples")
    print("  (typically 1-10) provided in the prompt context")
    print()
    
    print("The Spectrum:\n")
    
    spectrum = {
        'Zero-shot': {
            'examples': 0,
            'description': 'Task description only',
            'best_for': 'Simple, well-known tasks',
            'performance': '50-70%'
        },
        'One-shot': {
            'examples': 1,
            'description': 'Single example',
            'best_for': 'Format demonstration',
            'performance': '60-75%'
        },
        'Few-shot': {
            'examples': '2-10',
            'description': 'Multiple examples',
            'best_for': 'Pattern learning',
            'performance': '70-90%'
        },
        'Many-shot': {
            'examples': '10+',
            'description': 'Many examples (if context allows)',
            'best_for': 'Complex patterns',
            'performance': '75-95%'
        }
    }
    
    for approach, info in spectrum.items():
        print(f"{approach.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Why Few-Shot Works:")
    print("  • Models trained on in-context learning")
    print("  • Examples clarify ambiguous instructions")
    print("  • Demonstrates format and style")
    print("  • Shows edge case handling")
    print("  • More natural than verbose explanations")

few_shot_concept()
```

### Few-Shot Example

```python
def few_shot_example_demo():
    """Demonstrating few-shot learning in action."""
    
    print("\n\nFew-Shot Learning in Action:\n")
    
    print("Task: Extract product and sentiment from reviews")
    print()
    
    print("Zero-Shot Attempt:")
    print("─" * 60)
    zero_shot = """
Extract the product name and sentiment (positive/negative).

Review: "The laptop is amazing, super fast!"
"""
    print(zero_shot)
    print("Result: Inconsistent format, sometimes misses product name")
    print()
    
    print("Few-Shot Approach:")
    print("─" * 60)
    few_shot = """
Extract the product name and sentiment from each review.

Review: "The headphones sound great!"
Product: headphones, Sentiment: positive

Review: "This phone keeps crashing, very disappointed."
Product: phone, Sentiment: negative

Review: "The laptop is amazing, super fast!"
Product: """
    print(few_shot)
    print("Expected: laptop, Sentiment: positive")
    print()
    
    print("Result: Consistent format, reliable extraction ✓")
    print()
    
    print("What the Model Learned:")
    print("  • Output format: 'Product: X, Sentiment: Y'")
    print("  • Extract noun (the product)")
    print("  • Classify sentiment based on descriptive words")
    print("  • Handle different review lengths")

few_shot_example_demo()
```

## How Few-Shot Learning Works

### The Mechanism

```python
def few_shot_mechanism():
    """How few-shot learning works under the hood."""
    
    print("How Few-Shot Learning Works:\n")
    
    print("1. Pattern Recognition")
    print("   Model identifies pattern in input-output pairs")
    print("   Example: 'Q: X → A: Y' pattern")
    print()
    
    print("2. In-Context Adaptation")
    print("   Model temporarily 'learns' task within context window")
    print("   No weight updates - purely from attention")
    print()
    
    print("3. Pattern Application")
    print("   Applies learned pattern to new input")
    print("   Generates output matching demonstrated format")
    print()
    
    print("=" * 60)
    print("\nMechanistic View:\n")
    
    print("Examples in Prompt:")
    print("  ┌─────────────┐")
    print("  │ Example 1   │ ─┐")
    print("  │ Example 2   │  ├─→ Attention learns pattern")
    print("  │ Example 3   │ ─┘")
    print("  │ New Input   │ ───→ Apply pattern")
    print("  └─────────────┘")
    print()
    
    print("What Models Extract:")
    print("  • Input format patterns")
    print("  • Output format patterns")
    print("  • Task-specific logic (from examples)")
    print("  • Edge case handling")
    print("  • Style and tone")
    
    print("\nKey Insight:")
    print("  Few-shot learning happens in the forward pass")
    print("  No gradient updates - just attention over context")
    print("  Think: 'pattern matching' not 'learning'")

few_shot_mechanism()
```

### Emergence with Scale

```python
def few_shot_emergence():
    """Few-shot ability emerges with model scale."""
    
    print("\n\nFew-Shot Capability vs Model Size:\n")
    
    print("Few-shot ability emerges around 1-6B parameters")
    print()
    
    models = [
        ('Small (< 1B)', '20-40%', 'Weak few-shot, barely better than zero-shot'),
        ('Medium (1-7B)', '50-70%', 'Decent few-shot, benefits from examples'),
        ('Large (7-30B)', '70-85%', 'Strong few-shot, significant gains'),
        ('Very Large (30B+)', '80-95%', 'Excellent few-shot, near fine-tuned')
    ]
    
    print(f"{'Model Size':<20} {'Few-Shot Perf':<15} {'Notes'}")
    print("="*70)
    
    for size, perf, notes in models:
        print(f"{size:<20} {perf:<15} {notes}")
    
    print("\n" + "=" * 60)
    print("\nPerformance Gain from Examples:\n")
    
    gains = {
        'Small models': '+5-15% over zero-shot',
        'Medium models': '+15-25% over zero-shot',
        'Large models': '+20-35% over zero-shot',
        'Very large models': '+25-40% over zero-shot'
    }
    
    for model_type, gain in gains.items():
        print(f"  {model_type}: {gain}")
    
    print("\nKey Finding:")
    print("  Larger models extract more from examples")
    print("  But even 7B models benefit significantly from few-shot")

few_shot_emergence()
```

## Example Selection Strategies

### Selection Criteria

```python
def example_selection_criteria():
    """What makes a good few-shot example."""
    
    print("Example Selection Criteria:\n")
    
    criteria = {
        'Relevance': {
            'description': 'Examples should be similar to target task',
            'good': 'Email examples for email classification',
            'bad': 'News article examples for email classification',
            'impact': 'High - relevant examples teach right patterns'
        },
        'Diversity': {
            'description': 'Cover different types/edge cases',
            'good': 'Short, medium, long texts; different formats',
            'bad': 'All examples very similar',
            'impact': 'High - generalization improves'
        },
        'Clarity': {
            'description': 'Unambiguous input-output relationship',
            'good': 'Clear why input maps to output',
            'bad': 'Confusing or contradictory examples',
            'impact': 'Critical - unclear examples mislead'
        },
        'Correctness': {
            'description': 'Examples must be accurate',
            'good': 'Verified correct labels/outputs',
            'bad': 'Wrong labels or inconsistent',
            'impact': 'Critical - model copies mistakes'
        },
        'Representativeness': {
            'description': 'Examples represent real distribution',
            'good': 'Typical cases from actual data',
            'bad': 'Only edge cases or artificial examples',
            'impact': 'Medium - affects generalization'
        },
        'Balance': {
            'description': 'Balanced across classes/types',
            'good': 'Equal positive/negative examples',
            'bad': 'All positive or all negative',
            'impact': 'Medium-High - prevents bias'
        }
    }
    
    for criterion, info in criteria.items():
        print(f"{criterion.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

example_selection_criteria()
```

### Selection Methods

```python
def example_selection_methods():
    """Different strategies for selecting examples."""
    
    print("\n\nExample Selection Methods:\n")
    
    methods = {
        'Random selection': {
            'how': 'Pick examples randomly from dataset',
            'pros': 'Simple, unbiased',
            'cons': 'May miss important patterns',
            'when': 'Large, diverse dataset available'
        },
        'Diversity-based': {
            'how': 'Select diverse examples (k-means, etc.)',
            'pros': 'Good coverage of input space',
            'cons': 'Computationally expensive',
            'when': 'Want broad pattern coverage'
        },
        'Similarity-based': {
            'how': 'Select examples similar to test input',
            'pros': 'Most relevant to current task',
            'cons': 'Requires computing similarity',
            'when': 'Dynamic example selection possible'
        },
        'Hard example mining': {
            'how': 'Select examples model struggles with',
            'pros': 'Teaches difficult cases',
            'cons': 'May not cover common cases',
            'when': 'Improving on known failure modes'
        },
        'Stratified sampling': {
            'how': 'Sample proportionally from each class',
            'pros': 'Balanced representation',
            'cons': 'Requires labeled data',
            'when': 'Classification tasks'
        },
        'Manual curation': {
            'how': 'Hand-pick best examples',
            'pros': 'Highest quality, educational',
            'cons': 'Time-consuming, doesn\'t scale',
            'when': 'Small number needed, quality critical'
        }
    }
    
    for method, info in methods.items():
        print(f"{method.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Recommendation:")
    print("  Start with manual curation for core examples")
    print("  Use diversity-based selection for general coverage")
    print("  Consider similarity-based for production (if feasible)")

example_selection_methods()
```

### Similarity-Based Selection

```python
def similarity_based_selection():
    """Using similarity to select relevant examples."""
    
    print("\n\nSimilarity-Based Example Selection:\n")
    
    print("Concept: Select examples most similar to current input")
    print()
    
    print("Algorithm:")
    print("  1. Embed all potential examples")
    print("  2. Embed current test input")
    print("  3. Compute similarity (cosine, euclidean)")
    print("  4. Select top-k most similar examples")
    print("  5. Include in prompt")
    print()
    
    print("=" * 60)
    print("\nPython Implementation:\n")
    
    code = '''
import numpy as np
from sentence_transformers import SentenceTransformer

def select_similar_examples(test_input, example_pool, k=3):
    """Select k most similar examples to test input."""
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Embed test input
    test_embedding = model.encode([test_input])[0]
    
    # Embed all examples
    example_texts = [ex['input'] for ex in example_pool]
    example_embeddings = model.encode(example_texts)
    
    # Compute cosine similarity
    similarities = np.dot(example_embeddings, test_embedding)
    similarities = similarities / (
        np.linalg.norm(example_embeddings, axis=1) * 
        np.linalg.norm(test_embedding)
    )
    
    # Get top-k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Return selected examples
    return [example_pool[i] for i in top_k_indices]

# Usage
example_pool = [
    {'input': 'Great product!', 'output': 'positive'},
    {'input': 'Terrible quality.', 'output': 'negative'},
    {'input': 'Works as expected.', 'output': 'neutral'},
    # ... more examples
]

test = "Excellent service, very happy!"
selected = select_similar_examples(test, example_pool, k=3)

# Build prompt with selected examples
prompt = "Classify sentiment:\\n\\n"
for ex in selected:
    prompt += f"Input: {ex['input']} → Output: {ex['output']}\\n"
prompt += f"Input: {test} → Output:"
'''
    
    print(code)
    
    print("\nBenefits:")
    print("  • Dynamically adapt examples to input")
    print("  • More relevant examples = better performance")
    print("  • Can improve accuracy by 5-15%")
    
    print("\nDrawbacks:")
    print("  • Requires embedding computation")
    print("  • Adds latency")
    print("  • Need example pool")

similarity_based_selection()
```

## Example Quality Factors

### High-Quality Examples

```python
def high_quality_examples():
    """Characteristics of high-quality examples."""
    
    print("High-Quality Example Characteristics:\n")
    
    print("1. CLEAR INPUT-OUTPUT MAPPING\n")
    
    print("Good:")
    print('  Input: "The movie was fantastic!"')
    print('  Output: positive')
    print("  → Clear why positive")
    print()
    
    print("Bad:")
    print('  Input: "The movie was something."')
    print('  Output: positive')
    print("  → Unclear why positive")
    print()
    
    print("=" * 60)
    print("\n2. REPRESENTATIVE OF TASK\n")
    
    print("Good: Typical real-world examples")
    print('  "I love this product!" → positive')
    print('  "Arrived damaged." → negative')
    print()
    
    print("Bad: Artificial or extreme examples")
    print('  "This is the best thing ever in the universe!" → positive')
    print()
    
    print("=" * 60)
    print("\n3. CONSISTENT FORMAT\n")
    
    print("Good: All examples follow same format")
    print('  "Text: X" → "Sentiment: Y"')
    print('  "Text: A" → "Sentiment: B"')
    print()
    
    print("Bad: Inconsistent format")
    print('  "X" → "Sentiment: Y"')
    print('  "Text: A → B"')
    print()
    
    print("=" * 60)
    print("\n4. DIFFICULTY PROGRESSION\n")
    
    print("Good: Start simple, add complexity")
    print('  Example 1: "Great!" → positive (simple)')
    print('  Example 2: "Good value for money" → positive (context)')
    print('  Example 3: "Not bad, actually pretty good" → positive (negation)')
    print()
    
    print("Bad: All complex or all simple")
    print()
    
    print("=" * 60)
    print("\n5. EDGE CASE COVERAGE\n")
    
    print("Include examples of:")
    print("  • Sarcasm: 'Oh great, it broke' → negative")
    print("  • Negation: 'Not good' → negative")
    print("  • Mixed: 'Good but expensive' → mixed")
    print("  • Neutral: 'It works' → neutral")

high_quality_examples()
```

### Example Errors to Avoid

```python
def example_errors():
    """Common mistakes in few-shot examples."""
    
    print("\n\nCommon Example Errors:\n")
    
    errors = {
        'Inconsistent labels': {
            'problem': 'Similar inputs get different outputs',
            'example': '"Great!" → positive, "Excellent!" → neutral',
            'why_bad': 'Confuses the model, inconsistent pattern',
            'fix': 'Ensure consistent labeling criteria'
        },
        'Imbalanced classes': {
            'problem': 'All examples from one class',
            'example': '5 positive examples, 0 negative',
            'why_bad': 'Model biased toward majority class',
            'fix': 'Balance examples across classes'
        },
        'Too similar examples': {
            'problem': 'All examples nearly identical',
            'example': 'All short positive reviews',
            'why_bad': 'Poor generalization to varied inputs',
            'fix': 'Include diverse examples'
        },
        'Ambiguous examples': {
            'problem': 'Unclear why input maps to output',
            'example': '"It was okay" → positive (why?)',
            'why_bad': 'Model learns wrong pattern',
            'fix': 'Use clear, unambiguous examples'
        },
        'Wrong labels': {
            'problem': 'Examples have incorrect outputs',
            'example': '"Terrible!" → positive',
            'why_bad': 'Model learns wrong patterns',
            'fix': 'Verify all examples are correct'
        },
        'Overly complex examples': {
            'problem': 'Examples too difficult or long',
            'example': 'Paragraph-long reviews for simple task',
            'why_bad': 'Wastes context, may confuse',
            'fix': 'Use appropriately complex examples'
        }
    }
    
    for error, info in errors.items():
        print(f"{error.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

example_errors()
```

## Example Ordering and Format

### Order Effects

```python
def example_ordering():
    """How example order affects performance."""
    
    print("Example Ordering Effects:\n")
    
    print("Observation: Example order can impact results!")
    print()
    
    print("Recency Bias:")
    print("  • Model influenced more by recent examples")
    print("  • Last few examples have stronger effect")
    print("  • Impact: ±3-8% accuracy from order alone")
    print()
    
    print("=" * 60)
    print("\nOrdering Strategies:\n")
    
    strategies = {
        'Easy to hard': {
            'description': 'Simple examples first, complex later',
            'rationale': 'Build understanding gradually',
            'works_well': 'Teaching complex patterns'
        },
        'Hard to easy': {
            'description': 'Complex first, simple later',
            'rationale': 'Recent simple examples clarify',
            'works_well': 'When last example is template'
        },
        'Balanced alternating': {
            'description': 'Alternate between classes/types',
            'rationale': 'Avoid class bias',
            'works_well': 'Classification tasks'
        },
        'Similarity-sorted': {
            'description': 'Most similar to input last',
            'rationale': 'Most relevant example fresh',
            'works_well': 'Dynamic example selection'
        },
        'Random': {
            'description': 'Random order',
            'rationale': 'Baseline, no bias',
            'works_well': 'When no clear ordering strategy'
        }
    }
    
    for strategy, info in strategies.items():
        print(f"{strategy.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Recommendation:")
    print("  • Test multiple orderings during development")
    print("  • For production: use consistent ordering")
    print("  • If possible: dynamically order by similarity")
    print("  • Default: balanced alternating or easy-to-hard")

example_ordering()
```

### Format Consistency

```python
def format_consistency():
    """Importance of consistent example formatting."""
    
    print("\n\nFormat Consistency:\n")
    
    print("Rule: ALL examples must follow identical format")
    print()
    
    print("Bad (Inconsistent Format):")
    print("─" * 60)
    bad_examples = """
Review: "Great product!" 
Sentiment: positive

"Not happy with this" → negative

Review: Works well, Sentiment: positive

Text: "Could be better"
Output: neutral
"""
    print(bad_examples)
    print("Issue: Model confused by varying formats")
    print()
    
    print("Good (Consistent Format):")
    print("─" * 60)
    good_examples = """
Review: "Great product!"
Sentiment: positive

Review: "Not happy with this"
Sentiment: negative

Review: "Works well"
Sentiment: positive

Review: "Could be better"
Sentiment: neutral
"""
    print(good_examples)
    print("Better: Consistent 'Review: X' → 'Sentiment: Y' format")
    print()
    
    print("=" * 60)
    print("\nFormat Elements to Keep Consistent:\n")
    
    elements = [
        'Input label (e.g., "Review:", "Text:", "Input:")',
        'Output label (e.g., "Sentiment:", "Label:", "Output:")',
        'Delimiter (→, =>, :, etc.)',
        'Punctuation and spacing',
        'Capitalization',
        'Line breaks',
        'Special characters'
    ]
    
    for element in elements:
        print(f"  • {element}")
    
    print("\nTemplate Approach:")
    template = """
Define a template and fill for each example:

Template:
Input: {text}
Output: {label}

Example 1:
Input: Sample text here
Output: positive

Example 2:
Input: Another sample
Output: negative
"""
    print(template)

format_consistency()
```

## Balancing Number of Examples

### Optimal Number

```python
def optimal_example_count():
    """Finding the right number of examples."""
    
    print("Optimal Number of Examples:\n")
    
    print("Performance vs Number of Examples:")
    print()
    
    # Typical performance curve
    examples_data = [
        (0, 60, 'Zero-shot baseline'),
        (1, 68, 'One example helps significantly'),
        (2, 75, 'Two examples much better'),
        (3, 80, 'Three examples strong'),
        (5, 83, 'Diminishing returns begin'),
        (10, 85, 'Marginal gains'),
        (20, 86, 'Little additional benefit')
    ]
    
    print(f"{'# Examples':<12} {'Performance':<15} {'Notes'}")
    print("="*60)
    
    for count, perf, notes in examples_data:
        bar = '█' * (perf // 5)
        print(f"{count:<12} {perf}% {bar:<20} {notes}")
    
    print("\n" + "=" * 60)
    print("\nKey Findings:\n")
    
    print("• Biggest jump: 0 → 1-2 examples (+10-15%)")
    print("• Diminishing returns after 3-5 examples")
    print("• Sweet spot: 3-5 examples for most tasks")
    print("• More examples: only if complex pattern or context allows")
    print()
    
    print("=" * 60)
    print("\nFactors Affecting Optimal Count:\n")
    
    factors = {
        'Task complexity': 'Simple: 1-3, Complex: 5-10+',
        'Pattern clarity': 'Clear: fewer, Ambiguous: more',
        'Model size': 'Larger models extract more from fewer',
        'Context window': 'Limited: fewer, Large: can use more',
        'Example quality': 'High quality: fewer needed'
    }
    
    for factor, guidance in factors.items():
        print(f"  {factor}: {guidance}")
    
    print("\nRule of Thumb:")
    print("  Start with 3 examples")
    print("  Add more if performance insufficient")
    print("  Stop when gains < 2% from adding example")

optimal_example_count()
```

### Context Length Tradeoffs

```python
def context_length_tradeoffs():
    """Balancing examples with context window."""
    
    print("\n\nContext Length Tradeoffs:\n")
    
    print("Context Window Allocation:\n")
    
    allocation = {
        'System prompt': '100-500 tokens',
        'Few-shot examples': '200-2000 tokens (depends on count)',
        'Actual input': '100-10000 tokens (varies)',
        'Output space': '100-2000 tokens (reserve for response)',
        'Safety margin': '100-500 tokens (buffer)'
    }
    
    print("Typical allocation:")
    for component, tokens in allocation.items():
        print(f"  • {component}: {tokens}")
    
    print("\n" + "=" * 60)
    print("\nTradeoff Scenarios:\n")
    
    print("Scenario 1: Short inputs (e.g., tweets)")
    print("  • Can afford 5-10 examples")
    print("  • Each example ~50-100 tokens")
    print("  • Total examples: 500-1000 tokens")
    print("  • Plenty of room for examples ✓")
    print()
    
    print("Scenario 2: Long inputs (e.g., documents)")
    print("  • Input alone: 5000-10000 tokens")
    print("  • Limited space for examples")
    print("  • Maybe 1-3 examples only")
    print("  • Must prioritize quality over quantity")
    print()
    
    print("Scenario 3: Limited context (small models)")
    print("  • Only 2048 token context")
    print("  • Must be very selective")
    print("  • 1-2 examples maximum")
    print("  • Consider zero-shot instead")
    print()
    
    print("=" * 60)
    print("\nOptimization Strategies:\n")
    
    strategies = [
        'Use concise examples (remove unnecessary words)',
        'Compress example format (abbreviate labels)',
        'Dynamic selection (most relevant examples only)',
        'Summarize long inputs before prompting',
        'Use retrieval for examples (don\'t include all)',
        'Consider fine-tuning if need many examples'
    ]
    
    for strategy in strategies:
        print(f"  • {strategy}")

context_length_tradeoffs()
```

## Few-Shot vs Zero-Shot vs Fine-Tuning

### Comparison

```python
def comparison_table():
    """Comparing different learning approaches."""
    
    print("Few-Shot vs Zero-Shot vs Fine-Tuning:\n")
    
    comparison = {
        'Setup effort': {
            'zero_shot': 'Low (write instruction)',
            'few_shot': 'Medium (create examples)',
            'fine_tuning': 'High (collect dataset, train)'
        },
        'Iteration speed': {
            'zero_shot': 'Instant',
            'few_shot': 'Instant',
            'fine_tuning': 'Hours to days'
        },
        'Performance': {
            'zero_shot': '50-70% (simple tasks)',
            'few_shot': '70-90% (with good examples)',
            'fine_tuning': '85-98% (task-specific)'
        },
        'Cost per inference': {
            'zero_shot': 'Low (short prompts)',
            'few_shot': 'Medium (longer prompts)',
            'fine_tuning': 'Low (no examples needed)'
        },
        'Training cost': {
            'zero_shot': '$0',
            'few_shot': '$0',
            'fine_tuning': '$100-$10,000+'
        },
        'Flexibility': {
            'zero_shot': 'Very high (change instantly)',
            'few_shot': 'Very high (swap examples)',
            'fine_tuning': 'Low (need retrain)'
        },
        'Data required': {
            'zero_shot': '0 examples',
            'few_shot': '1-20 examples',
            'fine_tuning': '100-10,000+ examples'
        },
        'Best for': {
            'zero_shot': 'Simple, well-known tasks',
            'few_shot': 'Most tasks, quick iteration',
            'fine_tuning': 'Production, high volume'
        }
    }
    
    # Print header
    print(f"{'Aspect':<20} {'Zero-Shot':<25} {'Few-Shot':<25} {'Fine-Tuning'}")
    print("="*95)
    
    # Print rows
    for aspect, values in comparison.items():
        print(f"{aspect:<20} {values['zero_shot']:<25} {values['few_shot']:<25} {values['fine_tuning']}")
    
    print("\n" + "=" * 60)
    print("\nDecision Guide:\n")
    
    print("Use Zero-Shot when:")
    print("  • Task is simple and common")
    print("  • Need minimal setup")
    print("  • Acceptable to have moderate performance")
    print()
    
    print("Use Few-Shot when:")
    print("  • Need better performance than zero-shot")
    print("  • Want fast iteration")
    print("  • Have 3-20 good examples")
    print("  • Custom task format or logic")
    print("  • Don't want to fine-tune")
    print()
    
    print("Use Fine-Tuning when:")
    print("  • Maximum performance required")
    print("  • High-volume inference (cost matters)")
    print("  • Task is stable (not changing)")
    print("  • Have 1000+ training examples")
    print("  • Context window too small for examples")

comparison_table()
```

## Domain-Specific Few-Shot

### Classification Tasks

```python
def classification_few_shot():
    """Few-shot for classification tasks."""
    
    print("\n\nFew-Shot for Classification:\n")
    
    print("Example: Email Classification")
    print("─" * 60)
    
    prompt = """
Classify emails as: spam, urgent, or normal

Email: "Congratulations! You've won $1,000,000! Click here now!"
Category: spam

Email: "URGENT: Server is down, customers cannot access site"
Category: urgent

Email: "Meeting scheduled for Tuesday at 2pm"
Category: normal

Email: "Your package has been delivered"
Category: normal

Email: "ACT NOW! Limited time offer, 90% discount!!!"
Category: spam

Email: "CRITICAL: Database backup failed, data at risk"
Category: urgent

Email: "Please review the attached quarterly report"
Category: """
    
    print(prompt)
    print("\nExpected: normal")
    
    print("\n" + "=" * 60)
    print("\nBest Practices for Classification:\n")
    
    practices = [
        'Include 2-3 examples per class',
        'Balance examples across classes',
        'Show edge cases (e.g., urgent but not spam)',
        'Use consistent format for all examples',
        'Include class list at the beginning',
        'Consider confidence if needed: "Category: spam (high confidence)"'
    ]
    
    for practice in practices:
        print(f"  • {practice}")

classification_few_shot()
```

### Generation Tasks

```python
def generation_few_shot():
    """Few-shot for text generation tasks."""
    
    print("\n\nFew-Shot for Generation:\n")
    
    print("Example: Product Description Generation")
    print("─" * 60)
    
    prompt = """
Write engaging product descriptions from specifications.

Specs: Laptop, 16GB RAM, 512GB SSD, Intel i7
Description: Experience lightning-fast performance with this powerful laptop. 
Featuring 16GB RAM and 512GB SSD storage, it handles multitasking with ease. 
The Intel i7 processor ensures smooth operation for work and play.

Specs: Wireless headphones, 20hr battery, noise cancelling
Description: Immerse yourself in pure audio bliss with these wireless headphones. 
Enjoy 20 hours of playtime and advanced noise cancellation that blocks out 
distractions. Perfect for music lovers and frequent travelers.

Specs: Coffee maker, 12-cup, programmable, thermal carafe
Description: Wake up to freshly brewed coffee every morning with this programmable 
coffee maker. The 12-cup thermal carafe keeps coffee hot for hours, making it 
perfect for households and offices.

Specs: Running shoes, cushioned sole, breathable mesh, size 9
Description: """
    
    print(prompt)
    
    print("\n" + "=" * 60)
    print("\nBest Practices for Generation:\n")
    
    practices = [
        'Show desired style and tone in examples',
        'Demonstrate target length (examples should be similar length)',
        'Include variety (different structures, word choices)',
        'Show how to handle different inputs',
        'Consistent quality across examples',
        'Demonstrate specific requirements (e.g., mention all specs)'
    ]
    
    for practice in practices:
        print(f"  • {practice}")

generation_few_shot()
```

### Extraction Tasks

```python
def extraction_few_shot():
    """Few-shot for information extraction."""
    
    print("\n\nFew-Shot for Extraction:\n")
    
    print("Example: Structured Information Extraction")
    print("─" * 60)
    
    prompt = """
Extract person name, company, and role from text.

Text: "Jane Smith, CEO of TechCorp, announced the new product."
Name: Jane Smith
Company: TechCorp
Role: CEO

Text: "According to software engineer Mike Johnson at DataSystems..."
Name: Mike Johnson
Company: DataSystems
Role: software engineer

Text: "Dr. Sarah Lee from MedResearch Institute published the findings."
Name: Sarah Lee
Company: MedResearch Institute
Role: Dr. (researcher)

Text: "The keynote speaker, Alex Chen, CTO of CloudNet, discussed AI trends."
Name: Alex Chen
Company: CloudNet
Role: CTO

Text: "Professor Emily Wilson teaches at University of Science."
Name: """
    
    print(prompt)
    print("\nExpected:")
    print("Name: Emily Wilson")
    print("Company: University of Science")
    print("Role: Professor")
    
    print("\n" + "=" * 60)
    print("\nBest Practices for Extraction:\n")
    
    practices = [
        'Show how to handle missing fields',
        'Demonstrate edge cases (multiple people, no company)',
        'Use consistent output format (JSON or structured text)',
        'Show normalization (e.g., titles, name format)',
        'Include examples with variations (Mr., Dr., Prof.)',
        'Handle ambiguity (when role unclear)'
    ]
    
    for practice in practices:
        print(f"  • {practice}")

extraction_few_shot()
```

## Few-Shot Failure Modes

### Common Failures

```python
def few_shot_failures():
    """Common ways few-shot learning fails."""
    
    print("Few-Shot Failure Modes:\n")
    
    failures = {
        'Copying examples': {
            'symptom': 'Model returns exact example outputs',
            'cause': 'Examples too similar to input',
            'example': 'Input matches example exactly',
            'fix': 'Use diverse examples, different from likely inputs'
        },
        'Format confusion': {
            'symptom': 'Output format wrong despite examples',
            'cause': 'Inconsistent format in examples',
            'example': 'Examples use different delimiters',
            'fix': 'Ensure perfect format consistency'
        },
        'Class imbalance learning': {
            'symptom': 'Always predicts majority class',
            'cause': 'All examples from one class',
            'example': '5 positive, 0 negative examples',
            'fix': 'Balance examples across classes'
        },
        'Superficial pattern matching': {
            'symptom': 'Learns wrong pattern from examples',
            'cause': 'Examples have spurious correlations',
            'example': 'All short texts labeled positive',
            'fix': 'Vary length, style across classes'
        },
        'Example order bias': {
            'symptom': 'Heavily influenced by last example',
            'cause': 'Recency bias in attention',
            'example': 'Always predicts class of last example',
            'fix': 'Alternate classes, test different orders'
        },
        'Context overflow': {
            'symptom': 'Performance degrades with more examples',
            'cause': 'Too many examples, input truncated',
            'example': '20 examples push input out of context',
            'fix': 'Reduce example count, increase quality'
        }
    }
    
    for failure, info in failures.items():
        print(f"{failure.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

few_shot_failures()
```

### Debugging Few-Shot

```python
def debugging_few_shot():
    """How to debug few-shot learning issues."""
    
    print("\n\nDebugging Few-Shot Issues:\n")
    
    print("Debugging Checklist:\n")
    
    checks = [
        ('Example quality', 'Are all examples correct and clear?'),
        ('Format consistency', 'Do all examples follow identical format?'),
        ('Class balance', 'Are classes represented equally?'),
        ('Example diversity', 'Do examples cover different cases?'),
        ('Example relevance', 'Are examples similar to target task?'),
        ('Example count', 'Too few (< 2) or too many (> context)?'),
        ('Example order', 'Try different orderings'),
        ('Model capability', 'Is model too small for few-shot?')
    ]
    
    for check, question in checks:
        print(f"  ☐ {check}: {question}")
    
    print("\n" + "=" * 60)
    print("\nSystematic Debugging Process:\n")
    
    process = [
        ('1. Test zero-shot', 'Baseline performance without examples'),
        ('2. Add 1 example', 'Does it help? Which class/type?'),
        ('3. Add 2-3 examples', 'Balance classes, measure improvement'),
        ('4. Vary examples', 'Try different example sets'),
        ('5. Test ordering', 'Try different example orders'),
        ('6. Check edge cases', 'How does it handle unusual inputs?'),
        ('7. Compare formats', 'Test different output formats'),
        ('8. Optimize count', 'Find sweet spot for # examples')
    ]
    
    for step, description in process:
        print(f"  {step}: {description}")
    
    print("\n" + "=" * 60)
    print("\nCommon Fixes:\n")
    
    fixes = {
        'Poor performance': 'Add more diverse examples, check quality',
        'Inconsistent output': 'Enforce format consistency, add constraints',
        'Wrong format': 'Make format more explicit in examples',
        'Biased predictions': 'Balance classes in examples',
        'Copying examples': 'Use examples different from test inputs',
        'Degrading with examples': 'Reduce count, improve quality'
    }
    
    for problem, solution in fixes.items():
        print(f"  • {problem}: {solution}")

debugging_few_shot()
```

## Advanced Few-Shot Techniques

### Dynamic Example Selection

```python
def dynamic_example_selection():
    """Advanced technique: dynamically select examples per input."""
    
    print("Dynamic Example Selection:\n")
    
    print("Concept: Choose different examples for each test input")
    print("         based on similarity or relevance")
    print()
    
    print("Algorithm:")
    print("  1. Maintain pool of candidate examples")
    print("  2. For each test input:")
    print("     a. Compute similarity to all candidates")
    print("     b. Select top-k most similar")
    print("     c. Build prompt with selected examples")
    print("     d. Query model")
    print()
    
    print("=" * 60)
    print("\nImplementation Pattern:\n")
    
    code = '''
class DynamicFewShot:
    def __init__(self, example_pool, k=3):
        self.example_pool = example_pool  # List of (input, output) pairs
        self.k = k
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute example embeddings
        self.example_embeddings = self.encoder.encode(
            [ex['input'] for ex in example_pool]
        )
    
    def select_examples(self, test_input):
        """Select k most relevant examples for test input."""
        # Embed test input
        test_emb = self.encoder.encode([test_input])[0]
        
        # Compute similarities
        sims = cosine_similarity([test_emb], self.example_embeddings)[0]
        
        # Get top-k indices
        top_k = np.argsort(sims)[-self.k:][::-1]
        
        return [self.example_pool[i] for i in top_k]
    
    def build_prompt(self, test_input, task_description):
        """Build prompt with dynamically selected examples."""
        examples = self.select_examples(test_input)
        
        prompt = f"{task_description}\\n\\n"
        
        for ex in examples:
            prompt += f"Input: {ex['input']}\\n"
            prompt += f"Output: {ex['output']}\\n\\n"
        
        prompt += f"Input: {test_input}\\nOutput:"
        
        return prompt

# Usage
dynamic_fs = DynamicFewShot(example_pool, k=3)
prompt = dynamic_fs.build_prompt(test_input, "Classify sentiment:")
'''
    
    print(code)
    
    print("\nBenefits:")
    print("  • Maximally relevant examples for each input")
    print("  • Can improve accuracy by 5-15%")
    print("  • Handles diverse input distribution")
    
    print("\nCosts:")
    print("  • Requires similarity computation")
    print("  • Adds latency (~50-200ms)")
    print("  • Need large example pool")

dynamic_example_selection()
```

### Few-Shot Chain-of-Thought

```python
def few_shot_cot():
    """Combining few-shot with chain-of-thought reasoning."""
    
    print("\n\nFew-Shot Chain-of-Thought:\n")
    
    print("Combine few-shot learning with reasoning demonstrations")
    print()
    
    prompt = """
Solve math word problems step-by-step.

Problem: Sarah has 5 apples. She buys 3 more. How many does she have?
Reasoning: Sarah starts with 5 apples. She buys 3 more apples.
So she has 5 + 3 = 8 apples total.
Answer: 8

Problem: A store had 20 items. They sold 12 and got a shipment of 15. How many now?
Reasoning: The store started with 20 items. They sold 12, so now have 20 - 12 = 8 items.
Then they received 15 more items. So they have 8 + 15 = 23 items total.
Answer: 23

Problem: Tom has $50. He spends $18 on lunch and $12 on a movie. How much left?
Reasoning: Tom starts with $50. He spends $18 on lunch, leaving $50 - $18 = $32.
Then he spends $12 on a movie, leaving $32 - $12 = $20.
Answer: $20

Problem: A car travels 60 miles in 1.5 hours. What is its average speed in mph?
Reasoning: """
    
    print(prompt)
    
    print("\nKey Features:")
    print("  • Examples include reasoning process")
    print("  • Shows HOW to solve, not just answer")
    print("  • Model learns to reason step-by-step")
    print("  • Much better for complex problems")
    
    print("\n" + "=" * 60)
    print("\nPerformance Comparison:")
    print()
    
    comparison = [
        ('Zero-shot direct', '35%', 'Just ask for answer'),
        ('Few-shot direct', '45%', 'Examples without reasoning'),
        ('Zero-shot CoT', '58%', '"Let\'s think step by step"'),
        ('Few-shot CoT', '75%', 'Examples WITH reasoning')
    ]
    
    print(f"{'Method':<20} {'Accuracy':<12} {'Description'}")
    print("="*60)
    
    for method, accuracy, description in comparison:
        print(f"{method:<20} {accuracy:<12} {description}")
    
    print("\nBest Use Cases:")
    print("  • Math word problems")
    print("  • Multi-step reasoning")
    print("  • Logic puzzles")
    print("  • Complex decision-making")

few_shot_cot()
```

## Summary

**Key Concepts**:

1. **Few-shot learning** uses examples in the prompt to teach tasks without training
2. **Example quality > quantity** - 3-5 good examples often better than 10 mediocre
3. **Similarity-based selection** improves relevance by choosing examples close to test input
4. **Format consistency** is critical - all examples must follow identical structure
5. **Class balance** prevents bias - represent all classes equally in examples
6. **Order effects** exist - example ordering can impact results by ±5-10%
7. **Diminishing returns** after 3-5 examples for most tasks

**Few-Shot Template**:

```
[Task Description]

[Example 1: Input → Output]
[Example 2: Input → Output]
[Example 3: Input → Output]

[Test Input] →
```

**Performance Gains**:

| Examples | Typical Improvement | Best For |
|----------|---------------------|----------|
| 0 (zero-shot) | Baseline | Simple tasks |
| 1 (one-shot) | +8-12% | Format demo |
| 2-3 | +15-25% | Most tasks |
| 5-10 | +20-35% | Complex patterns |
| 10+ | +25-40% (diminishing) | Very complex |

**Best Practices**:

✓ **Selection**: Diverse, relevant, balanced examples  
✓ **Quality**: Clear, correct, representative examples  
✓ **Format**: Consistent structure across all examples  
✓ **Count**: Start with 3, add if needed  
✓ **Order**: Test different orderings, use balanced alternation  
✓ **Relevance**: Dynamic selection if possible  

**Common Mistakes**:

❌ Inconsistent example format  
❌ All examples from one class  
❌ Too many similar examples  
❌ Wrong or ambiguous labels  
❌ Too many examples (context overflow)  
❌ Examples too different from test cases  

**Decision Matrix**:

```
Use Few-Shot When:
• Task is custom or domain-specific
• Need better performance than zero-shot
• Have 3-20 good examples
• Want fast iteration
• Context window has room

Use Zero-Shot When:
• Task is simple and common
• Context window is tight
• No examples available

Use Fine-Tuning When:
• Need maximum performance
• High inference volume
• Have 1000+ examples
• Task is stable
```

**Advanced Techniques**:

- **Dynamic selection**: Choose examples based on test input similarity (+5-15%)
- **Few-shot CoT**: Include reasoning in examples (+15-30% on reasoning tasks)
- **Hybrid approaches**: Combine with retrieval, tools, or fine-tuning

**Quick Checklist**:

```
☐ 3-5 examples selected
☐ Examples are correct and clear
☐ Format consistent across all examples
☐ Classes balanced (for classification)
☐ Examples diverse (cover different cases)
☐ Examples relevant to task
☐ Order tested (try alternatives)
☐ Context length checked (examples fit)
```

## Next Steps

- Learn [Chain-of-Thought Prompting](cot-prompting.md) to add reasoning to examples
- Study [Structured Output](structured-output.md) for consistent example formatting
- Master [Prompt Optimization](prompt-optimization.md) for systematic example testing
- Explore [Advanced Patterns](advanced-patterns.md) for dynamic example selection
- Review [In-Context Learning](../llm-concepts/in-context-learning.md) theory
- Apply to [Application Patterns](../application_patterns/best-practices.md) for production use

