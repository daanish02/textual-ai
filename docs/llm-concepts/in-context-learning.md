# In-Context Learning

## Table of Contents

- [Introduction](#introduction)
- [What is In-Context Learning?](#what-is-in-context-learning)
- [Zero-Shot Learning](#zero-shot-learning)
- [Few-Shot Learning](#few-shot-learning)
- [Why In-Context Learning Works](#why-in-context-learning-works)
- [Demonstration Selection](#demonstration-selection)
- [Prompt Formatting](#prompt-formatting)
- [Factors Affecting ICL Performance](#factors-affecting-icl-performance)
- [In-Context Learning vs Fine-Tuning](#in-context-learning-vs-fine-tuning)
- [Limitations and Failure Modes](#limitations-and-failure-modes)
- [Advanced ICL Techniques](#advanced-icl-techniques)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**In-context learning (ICL)** is the remarkable ability of large language models to learn new tasks from just a few examples provided in the prompt -- without any parameter updates. Show GPT-4 three examples of translating English to French, and it can translate new sentences. This capability fundamentally changed how we interact with AI.

```
Traditional Machine Learning:
  1. Collect large labeled dataset (10K+ examples)
  2. Train/fine-tune model parameters
  3. Deploy model for specific task

In-Context Learning:
  1. Show model 0-10 examples in prompt
  2. Model infers task and performs it
  3. Same model works for unlimited tasks

The difference: No training, just prompting!
```

**Key insight**: Large language models contain the capability to perform many tasks. In-context learning is about eliciting the right behavior through examples and instructions, not teaching new skills.

This guide explores how ICL works, how to use it effectively, and when it succeeds or fails.

## What is In-Context Learning?

### The Basic Mechanism

```python
def in_context_learning_demo():
    """Demonstrate the concept of in-context learning."""

    print("In-Context Learning: Learning from the Prompt\n")

    # Task: Sentiment classification
    prompt = """
Classify the sentiment of these reviews as positive or negative:

Review: "This movie was amazing!"
Sentiment: positive

Review: "Terrible acting and boring plot."
Sentiment: negative

Review: "A masterpiece of cinema!"
Sentiment: positive

Review: "I fell asleep halfway through."
Sentiment:"""

    print(prompt)
    print(" negative")  # Model prediction

    print("\n" + "="*60)
    print("\nWhat happened:")
    print("  1. Model sees pattern in examples")
    print("  2. Infers the task (sentiment classification)")
    print("  3. Applies pattern to new input")
    print("  4. Generates prediction")
    print("\nNo training occurred! Pure inference.")

in_context_learning_demo()
```

### The ICL Spectrum

```python
def icl_spectrum():
    """Different levels of in-context learning."""

    levels = {
        'Zero-shot': {
            'examples': 0,
            'description': 'Task description only, no examples',
            'difficulty': 'Hardest for model',
            'use_case': 'Simple tasks, well-defined in pretraining'
        },
        'One-shot': {
            'examples': 1,
            'description': 'Single example to demonstrate format',
            'difficulty': 'Better than zero-shot',
            'use_case': 'Format clarification, simple patterns'
        },
        'Few-shot': {
            'examples': '2-10',
            'description': 'Multiple examples showing task',
            'difficulty': 'Easiest for model',
            'use_case': 'Complex tasks, nuanced patterns'
        },
        'Many-shot': {
            'examples': '10-100',
            'description': 'Many examples (if context allows)',
            'difficulty': 'Best performance (diminishing returns)',
            'use_case': 'Complex tasks with long context models'
        }
    }

    print("In-Context Learning Spectrum:\n")
    for level, info in levels.items():
        print(f"{level.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

    print("Performance typically: Many-shot > Few-shot > One-shot > Zero-shot")
    print("But cost increases with more examples (longer prompts)")

icl_spectrum()
```

### Emergence of ICL

```
ICL Capability vs Model Size:

Accuracy
  100% ┤
       │                                    ╭─────
       │                               ╭────╯
   80% ┤                          ╭────╯
       │                     ╭────╯
   60% ┤                ╭────╯
       │           ╭────╯
   40% ┤      ╭────╯
       │ ╭────╯
   20% ┤─╯
       │
    0% ┤
       └───┬────┬────┬────┬────┬────┬────┬────
         100M  1B  6B  13B 30B 60B 175B 500B
                  Model Parameters

Key observations:
  • ICL emerges around 1-6B parameters
  • Rapid improvement from 6B to 30B
  • Continues improving beyond 100B
  • Small models (~100M) show minimal ICL
```

## Zero-Shot Learning

### Task Description Only

```python
def zero_shot_examples():
    """Examples of zero-shot learning with task descriptions."""

    examples = {
        'Classification': {
            'prompt': 'Classify the sentiment of this review as positive, negative, or neutral:\n\n"The food was okay but service was slow."\n\nSentiment:',
            'expected': 'neutral',
            'note': 'Clear task, well-defined in training'
        },
        'Translation': {
            'prompt': 'Translate this English sentence to Spanish:\n\nEnglish: "Hello, how are you?"\nSpanish:',
            'expected': '"Hola, ¿cómo estás?"',
            'note': 'Common task, seen during pretraining'
        },
        'Question Answering': {
            'prompt': 'Answer this question based on the passage:\n\nPassage: "Paris is the capital of France."\nQuestion: "What is the capital of France?"\nAnswer:',
            'expected': 'Paris',
            'note': 'Straightforward extraction'
        },
        'Summarization': {
            'prompt': 'Summarize this text in one sentence:\n\n"[Long text about climate change...]"\n\nSummary:',
            'expected': '[Concise summary]',
            'note': 'May struggle with specific format without examples'
        }
    }

    print("Zero-Shot Learning Examples:\n")
    for task, info in examples.items():
        print(f"{task}:")
        print(f"  Prompt: {info['prompt'][:80]}...")
        print(f"  Expected: {info['expected']}")
        print(f"  Note: {info['note']}")
        print()

    print("When zero-shot works best:")
    print("  • Task is common in pretraining data")
    print("  • Task description is clear and unambiguous")
    print("  • Desired output format is standard")
    print("  • Model is sufficiently large (30B+ params)")

zero_shot_examples()
```

### Zero-Shot Prompting Strategies

```python
def zero_shot_strategies():
    """Strategies for effective zero-shot prompting."""

    strategies = {
        'Clear instructions': {
            'bad': 'sentiment',
            'good': 'Classify the sentiment of this text as positive, negative, or neutral.',
            'reason': 'Explicit instructions reduce ambiguity'
        },
        'Format specification': {
            'bad': 'Answer:',
            'good': 'Answer with a single word: positive, negative, or neutral.\n\nAnswer:',
            'reason': 'Constrains output format'
        },
        'Role assignment': {
            'bad': 'Translate to French:',
            'good': 'You are a professional French translator. Translate this English text to French:',
            'reason': 'Primes model for task-specific behavior'
        },
        'Step-by-step': {
            'bad': 'What is 234 * 567?',
            'good': 'Calculate 234 * 567. Show your work step by step.',
            'reason': 'Encourages reasoning, reduces errors'
        }
    }

    print("Zero-Shot Prompting Strategies:\n")
    for strategy, info in strategies.items():
        print(f"{strategy.upper()}:")
        print(f"  Bad:  {info['bad']}")
        print(f"  Good: {info['good']}")
        print(f"  Why:  {info['reason']}")
        print()

zero_shot_strategies()
```

## Few-Shot Learning

### The Power of Examples

```python
def few_shot_demonstration():
    """Show the power of few-shot learning."""

    print("Few-Shot Learning: Learning from Examples\n")

    # Task: Extract person names from text
    prompt = """
Extract all person names from the text.

Text: "Alice went to the store with Bob."
Names: Alice, Bob

Text: "Dr. Smith treated patient Johnson."
Names: Dr. Smith, Johnson

Text: "The CEO, Maria Garcia, announced the merger."
Names: Maria Garcia

Text: "Professor Chen and his student Lee discussed the paper."
Names:"""

    print(prompt)
    print(" Professor Chen, Lee")

    print("\n" + "="*60)
    print("\nWhat the examples teach:")
    print("  • Format: comma-separated list")
    print("  • Task: extract person names (not places, not objects)")
    print("  • Include titles: Dr., Professor")
    print("  • Handle complex cases: compound names")
    print("\nModel learns task specification from examples!")

few_shot_demonstration()
```

### Constructing Few-Shot Prompts

```python
def few_shot_prompt_template():
    """Template for constructing effective few-shot prompts."""

    template = """
# Few-Shot Prompt Template

## Structure:
1. [Optional] Task description
2. Examples (2-10 typically)
3. New input
4. Prompt for output

## Format:
[Task description]

[Input label]: [Example input 1]
[Output label]: [Example output 1]

[Input label]: [Example input 2]
[Output label]: [Example output 2]

...

[Input label]: [New input]
[Output label]:
"""

    print(template)

    print("\nBest practices:")
    print("  • Consistent formatting across all examples")
    print("  • Examples should cover edge cases")
    print("  • Examples should be diverse")
    print("  • Clear input/output labels")
    print("  • Leave output label blank for new input")

few_shot_prompt_template()
```

### Number of Examples

```python
import numpy as np

def performance_vs_examples():
    """How performance scales with number of examples."""

    # Simulated performance curve
    n_examples = np.array([0, 1, 2, 3, 5, 8, 10, 15, 20])

    # Typical performance curve (logarithmic improvement)
    baseline = 40
    performance = baseline + 30 * np.log1p(n_examples) / np.log1p(20)

    print("Performance vs Number of Examples:\n")
    print(f"{'Examples':<10} {'Accuracy':<10} {'Improvement':<15}")
    print("=" * 35)

    for i, (n, perf) in enumerate(zip(n_examples, performance)):
        if i == 0:
            print(f"{int(n):<10} {perf:.1f}%      (baseline)")
        else:
            improvement = perf - performance[0]
            print(f"{int(n):<10} {perf:.1f}%      +{improvement:.1f}%")

    print("\nKey findings:")
    print("  • Biggest jump from 0 to 1-2 examples")
    print("  • Diminishing returns after ~5 examples")
    print("  • 10+ examples: minimal additional benefit")
    print("  • Exception: Very complex tasks may benefit from more")

performance_vs_examples()
```

## Why In-Context Learning Works

### Theories and Mechanisms

```python
def icl_theories():
    """Theories explaining why ICL works."""

    theories = {
        'Pattern matching': {
            'idea': 'Model recognizes patterns from pretraining',
            'mechanism': 'Similar examples seen during training',
            'evidence': 'Performance correlates with training data similarity'
        },
        'Task recognition': {
            'idea': 'Examples help model identify the task',
            'mechanism': 'Disambiguation among possible interpretations',
            'evidence': 'Even random labels can help (format matters)'
        },
        'Meta-learning': {
            'idea': 'Model learned to learn during pretraining',
            'mechanism': 'Pretraining implicitly teaches ICL',
            'evidence': 'Larger models show better ICL'
        },
        'Induction heads': {
            'idea': 'Specific attention heads implement ICL',
            'mechanism': 'Copy-paste mechanism via attention',
            'evidence': 'Identified in circuit analysis'
        },
        'Bayesian inference': {
            'idea': 'Model performs approximate Bayesian inference',
            'mechanism': 'Examples update posterior over functions',
            'evidence': 'Behavior matches Bayesian predictions'
        },
        'Gradient descent': {
            'idea': 'Forward pass approximates gradient descent',
            'mechanism': 'Self-attention implements optimization',
            'evidence': 'Transformers can implement GD in-context'
        }
    }

    print("Theories of In-Context Learning:\n")
    for theory, info in theories.items():
        print(f"{theory.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

    print("Likely reality: Combination of multiple mechanisms")
    print("Different mechanisms may dominate for different tasks")

icl_theories()
```

### What Models Learn During Pretraining

```python
def pretraining_enables_icl():
    """How pretraining enables in-context learning."""

    print("How Pretraining Enables ICL:\n")

    print("During pretraining, model sees:")
    print("  • Many different tasks naturally occurring in text")
    print("  • Examples followed by similar examples")
    print("  • Patterns, then new instances of patterns")
    print("  • Implicit 'few-shot' scenarios in natural text")

    print("\nExample from web data:")
    print("  'Here are some French translations:")
    print("  - Hello → Bonjour")
    print("  - Goodbye → Au revoir")
    print("  - Thank you → Merci'")

    print("\n  Model learns: examples → pattern → apply to new case")

    print("\nKey insight:")
    print("  ICL isn't a special ability added after pretraining")
    print("  It emerges naturally from language modeling objective")
    print("  Larger models → better at recognizing and using patterns")

pretraining_enables_icl()
```

## Demonstration Selection

### Quality Over Quantity

```python
def demonstration_quality():
    """What makes a good demonstration?"""

    qualities = {
        'Relevance': {
            'good': 'Similar to test input (domain, style, complexity)',
            'bad': 'Unrelated examples',
            'impact': 'High - most important factor'
        },
        'Diversity': {
            'good': 'Cover different aspects of task',
            'bad': 'All examples very similar',
            'impact': 'Medium - helps generalization'
        },
        'Correctness': {
            'good': 'Correct input-output pairs',
            'bad': 'Wrong labels or outputs',
            'impact': 'High - wrong examples hurt performance'
        },
        'Clarity': {
            'good': 'Clear, unambiguous examples',
            'bad': 'Confusing or edge cases',
            'impact': 'Medium - affects understanding'
        },
        'Balance': {
            'good': 'Balanced across classes/categories',
            'bad': 'All positive examples, no negative',
            'impact': 'Medium - prevents bias'
        }
    }

    print("Demonstration Quality Factors:\n")
    for quality, info in qualities.items():
        print(f"{quality.upper()}:")
        print(f"  Good: {info['good']}")
        print(f"  Bad:  {info['bad']}")
        print(f"  Impact: {info['impact']}")
        print()

demonstration_quality()
```

### Selection Strategies

```python
def demonstration_selection_strategies():
    """Strategies for selecting good demonstrations."""

    strategies = {
        'Random selection': {
            'method': 'Randomly sample from available examples',
            'pros': 'Simple, unbiased',
            'cons': 'May miss relevant examples',
            'when': 'Quick experiments, large example pool'
        },
        'Similarity-based': {
            'method': 'Select examples most similar to test input',
            'pros': 'Highly relevant',
            'cons': 'Requires embedding computation',
            'when': 'High-stakes tasks, have embeddings'
        },
        'Diversity-based': {
            'method': 'Select diverse examples covering task space',
            'pros': 'Good coverage',
            'cons': 'May include irrelevant examples',
            'when': 'Complex tasks with many variations'
        },
        'Difficulty-based': {
            'method': 'Include hard examples that challenge model',
            'pros': 'Teaches edge cases',
            'cons': 'May confuse model',
            'when': 'Model struggles with specific cases'
        },
        'Active learning': {
            'method': 'Select examples model is uncertain about',
            'pros': 'Most informative examples',
            'cons': 'Requires multiple queries',
            'when': 'Budget for multiple attempts'
        }
    }

    print("Demonstration Selection Strategies:\n")
    for strategy, info in strategies.items():
        print(f"{strategy.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

demonstration_selection_strategies()
```

### Example: Similarity-Based Selection

```python
def similarity_based_selection_example():
    """Implement similarity-based example selection."""

    print("Similarity-Based Demonstration Selection\n")

    # Simulate embeddings (in practice, use actual embedding model)
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Example pool (text → embedding)
    example_pool = {
        "The movie was great!": np.random.rand(384),
        "Terrible service at the restaurant.": np.random.rand(384),
        "Average experience, nothing special.": np.random.rand(384),
        "Loved the book, couldn't put it down!": np.random.rand(384),
        "Disappointed with the quality.": np.random.rand(384),
    }

    # Labels
    labels = {
        "The movie was great!": "positive",
        "Terrible service at the restaurant.": "negative",
        "Average experience, nothing special.": "neutral",
        "Loved the book, couldn't put it down!": "positive",
        "Disappointed with the quality.": "negative",
    }

    # New input
    test_input = "This product exceeded my expectations!"
    test_embedding = np.random.rand(384)

    # Find most similar examples
    similarities = []
    for text, embedding in example_pool.items():
        sim = cosine_similarity(test_embedding, embedding)
        similarities.append((text, sim, labels[text]))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"Test input: '{test_input}'")
    print("\nMost similar examples (top 3):\n")

    for i, (text, sim, label) in enumerate(similarities[:3], 1):
        print(f"{i}. '{text}'")
        print(f"   Similarity: {sim:.3f}")
        print(f"   Label: {label}\n")

    print("Prompt construction:")
    print("  Use these 3 examples in the few-shot prompt")
    print("  They're most relevant to the test input")

similarity_based_selection_example()
```

## Prompt Formatting

### Format Matters

```python
def format_importance():
    """Show how format affects ICL performance."""

    # Same task, different formats
    formats = {
        'Natural': {
            'example': 'Q: What is 2+2?\nA: 4',
            'pros': 'Easy to read',
            'cons': 'May be ambiguous'
        },
        'Structured': {
            'example': 'Question: What is 2+2?\nAnswer: 4',
            'pros': 'Clear separation',
            'cons': 'Verbose'
        },
        'Symbolic': {
            'example': 'Input: 2+2\nOutput: 4',
            'pros': 'Very clear roles',
            'cons': 'Less natural'
        },
        'Minimalist': {
            'example': '2+2 = 4',
            'pros': 'Concise',
            'cons': 'Less guidance'
        },
        'Tagged': {
            'example': '<question>What is 2+2?</question>\n<answer>4</answer>',
            'pros': 'Unambiguous structure',
            'cons': 'Unusual format'
        }
    }

    print("Impact of Prompt Format:\n")
    for format_name, info in formats.items():
        print(f"{format_name}:")
        print(f"  Example: {info['example']}")
        print(f"  Pros: {info['pros']}")
        print(f"  Cons: {info['cons']}")
        print()

    print("Key findings:")
    print("  • Consistent format across examples is crucial")
    print("  • Clear input/output separation helps")
    print("  • Natural formats often work best")
    print("  • Test different formats if performance poor")

format_importance()
```

### Delimiters and Structure

````python
def delimiter_examples():
    """Examples of using delimiters effectively."""

    print("Using Delimiters for Structure:\n")

    examples = {
        'Triple quotes': {
            'format': '"""Input text here"""',
            'use': 'Long inputs with special characters',
            'benefit': 'Clearly marks boundaries'
        },
        'Triple backticks': {
            'format': '```code here```',
            'use': 'Code, formatted text',
            'benefit': 'Preserves formatting'
        },
        'XML-style tags': {
            'format': '<input>text</input> <output>text</output>',
            'use': 'Complex structured data',
            'benefit': 'Hierarchical structure'
        },
        'Markdown headers': {
            'format': '## Input\n\n## Output',
            'use': 'Long-form content',
            'benefit': 'Natural for models trained on markdown'
        },
        'Dashes/lines': {
            'format': '---\nInput: ...\n---\nOutput: ...\n---',
            'use': 'Visual separation',
            'benefit': 'Easy to read'
        }
    }

    for delimiter, info in examples.items():
        print(f"{delimiter}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

delimiter_examples()
````

## Factors Affecting ICL Performance

### Model Size

```python
def model_size_effect():
    """How model size affects ICL performance."""

    print("Model Size Impact on ICL:\n")

    size_effects = {
        'Small (<1B)': {
            'icl_ability': 'Minimal to none',
            'max_shots': 'N/A',
            'behavior': 'Ignores examples, random output'
        },
        'Medium (1-7B)': {
            'icl_ability': 'Basic ICL emerges',
            'max_shots': '1-3 examples helpful',
            'behavior': 'Learns simple patterns'
        },
        'Large (7-30B)': {
            'icl_ability': 'Good ICL',
            'max_shots': '5-10 examples',
            'behavior': 'Complex patterns, instruction following'
        },
        'Very Large (30-100B)': {
            'icl_ability': 'Strong ICL',
            'max_shots': '10-50 examples',
            'behavior': 'Sophisticated reasoning, rare patterns'
        },
        'Frontier (100B+)': {
            'icl_ability': 'Excellent ICL',
            'max_shots': '100+ examples (long context)',
            'behavior': 'Near fine-tuned performance'
        }
    }

    for size, info in size_effects.items():
        print(f"{size}:")
        for key, value in info.items():
            print(f"  {key.replace('_', ' ').capitalize()}: {value}")
        print()

    print("Key takeaway: ICL improves dramatically with scale")
    print("Minimum ~7B parameters for reliable ICL on complex tasks")

model_size_effect()
```

### Task Complexity

```python
def task_complexity_icl():
    """How task complexity affects ICL."""

    complexity_levels = {
        'Simple (high ICL success)': [
            'Classification with clear categories',
            'Format conversion',
            'Entity extraction',
            'Simple Q&A'
        ],
        'Medium (moderate ICL success)': [
            'Sentiment with nuance',
            'Multi-step reasoning (2-3 steps)',
            'Summarization',
            'Translation'
        ],
        'Complex (low ICL success)': [
            'Multi-hop reasoning (4+ steps)',
            'Mathematical problem solving',
            'Long-form generation with constraints',
            'Complex code generation'
        ],
        'Very Complex (ICL often fails)': [
            'Novel algorithms',
            'Tasks requiring specialized knowledge',
            'High-precision domains (medical, legal)',
            'Tasks with many edge cases'
        ]
    }

    print("Task Complexity and ICL Success:\n")
    for level, tasks in complexity_levels.items():
        print(f"{level}:")
        for task in tasks:
            print(f"  • {task}")
        print()

    print("Rule of thumb:")
    print("  If task requires >10 examples to explain to human,")
    print("  ICL likely not sufficient → consider fine-tuning")

task_complexity_icl()
```

### Order Effects

```python
def order_effects():
    """Demonstration order can affect performance."""

    print("Order Effects in ICL:\n")

    findings = {
        'Recency bias': {
            'effect': 'Recent examples have more influence',
            'implication': 'Put most important examples last',
            'magnitude': 'Small but measurable (~1-3%)'
        },
        'Primacy for format': {
            'effect': 'First example defines format',
            'implication': 'Ensure first example is well-formatted',
            'magnitude': 'Medium (~5-10% if format unclear)'
        },
        'Class order bias': {
            'effect': 'Model may favor labels seen later',
            'implication': 'Vary class order or balance',
            'magnitude': 'Small (~2-5%)'
        },
        'Random order variance': {
            'effect': 'Different orderings give different results',
            'implication': 'Try multiple orderings, average results',
            'magnitude': 'Small (~2-5% variance)'
        }
    }

    for finding, info in findings.items():
        print(f"{finding.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

    print("Best practice:")
    print("  • Test multiple example orderings")
    print("  • Use self-consistency (multiple samples, majority vote)")
    print("  • For high-stakes: average over several orderings")

order_effects()
```

## In-Context Learning vs Fine-Tuning

### Comparison

```python
def icl_vs_finetuning():
    """Compare ICL and fine-tuning approaches."""

    comparison = {
        'Data required': {
            'ICL': '0-50 examples',
            'Fine-tuning': '100-10,000+ examples'
        },
        'Setup time': {
            'ICL': 'Minutes (craft prompt)',
            'Fine-tuning': 'Hours to days (training)'
        },
        'Compute cost': {
            'ICL': 'Low (inference only)',
            'Fine-tuning': 'High (training GPUs)'
        },
        'Iteration speed': {
            'ICL': 'Instant (change prompt)',
            'Fine-tuning': 'Slow (retrain model)'
        },
        'Performance': {
            'ICL': 'Good (70-90% of fine-tuned)',
            'Fine-tuning': 'Best (100%)'
        },
        'Robustness': {
            'ICL': 'Sensitive to prompt',
            'Fine-tuning': 'More robust'
        },
        'Versatility': {
            'ICL': 'One model, many tasks',
            'Fine-tuning': 'One model per task'
        },
        'Deployment': {
            'ICL': 'Simple (API call)',
            'Fine-tuning': 'Complex (host model)'
        }
    }

    print("In-Context Learning vs Fine-Tuning:\n")
    print(f"{'Aspect':<20} {'ICL':<30} {'Fine-Tuning':<30}")
    print("=" * 80)

    for aspect, values in comparison.items():
        print(f"{aspect:<20} {values['ICL']:<30} {values['Fine-Tuning']:<30}")

    print("\n\nWhen to use ICL:")
    print("  • Limited labeled data (<100 examples)")
    print("  • Need fast iteration")
    print("  • Many different tasks")
    print("  • Deployment simplicity important")

    print("\nWhen to use fine-tuning:")
    print("  • Have substantial labeled data (1000+)")
    print("  • Need best possible performance")
    print("  • Single critical task")
    print("  • Cost of training acceptable")

    print("\nHybrid approach:")
    print("  • Start with ICL for prototyping")
    print("  • Collect data using ICL system")
    print("  • Fine-tune once enough data collected")

icl_vs_finetuning()
```

### Performance Gaps

```
ICL vs Fine-Tuning Performance:

Task Accuracy
  100% ┤                              Fine-tuned ────────
       │                         ╭────╯
       │                    ╭────╯
   85% ┤               ╭────╯
       │          ╭────╯            ICL (few-shot)
       │     ╭────╯   ╱
   70% ┤─────╯    ╱──╯
       │      ╱──╯                   ICL (zero-shot)
   55% ┤──────╯
       │
   40% ┤
       └────┬────┬────┬────┬────┬────┬────┬────
         Simple  |  Medium  | Complex  | Very Complex
                    Task Complexity

Gap increases with task complexity
ICL approaches fine-tuning for simple tasks
Fine-tuning essential for complex/specialized tasks
```

## Limitations and Failure Modes

### When ICL Fails

```python
def icl_failure_modes():
    """Common failure modes of in-context learning."""

    failures = {
        'Shallow pattern matching': {
            'symptom': 'Model learns surface pattern, not task',
            'example': 'Learns "answer is always in first sentence"',
            'solution': 'Diverse examples, vary patterns'
        },
        'Example copying': {
            'symptom': 'Model copies from examples literally',
            'example': 'Outputs exact example instead of processing input',
            'solution': 'More diverse examples, different inputs'
        },
        'Format confusion': {
            'symptom': 'Model mixes up input and output',
            'example': 'Outputs input as output',
            'solution': 'Clear delimiters, consistent format'
        },
        'Majority label bias': {
            'symptom': 'Always predicts most common label',
            'example': 'All positive examples → always predicts positive',
            'solution': 'Balance examples across classes'
        },
        'Context window limits': {
            'symptom': 'Too many examples exceed context',
            'example': 'Model ignores early examples',
            'solution': 'Fewer examples, summarize, or fine-tune'
        },
        'Task misidentification': {
            'symptom': 'Model interprets task wrong',
            'example': 'Thinks classification is generation',
            'solution': 'Clear task description, better examples'
        }
    }

    print("ICL Failure Modes:\n")
    for failure, info in failures.items():
        print(f"{failure.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

icl_failure_modes()
```

### Sensitivity to Prompts

```python
def prompt_sensitivity():
    """ICL is sensitive to prompt variations."""

    print("Prompt Sensitivity Issues:\n")

    # Same task, different prompts, different results
    variations = {
        'Wording': {
            'prompt1': 'Classify sentiment:',
            'prompt2': 'What is the sentiment?',
            'difference': '±5-10% accuracy'
        },
        'Label names': {
            'prompt1': 'Sentiment: positive/negative',
            'prompt2': 'Sentiment: good/bad',
            'difference': '±3-7% accuracy'
        },
        'Example order': {
            'prompt1': '[pos, neg, pos, neg]',
            'prompt2': '[neg, pos, neg, pos]',
            'difference': '±2-5% accuracy'
        },
        'Instruction position': {
            'prompt1': 'Instruction at top',
            'prompt2': 'Instruction at bottom',
            'difference': '±3-5% accuracy'
        }
    }

    for variation, info in variations.items():
        print(f"{variation}:")
        print(f"  Option 1: {info['prompt1']}")
        print(f"  Option 2: {info['prompt2']}")
        print(f"  Performance difference: {info['difference']}")
        print()

    print("Mitigation strategies:")
    print("  • Test multiple prompt variations")
    print("  • Use prompt optimization tools")
    print("  • Ensemble over multiple prompts")
    print("  • Standardize on best-performing format")

prompt_sensitivity()
```

## Advanced ICL Techniques

### Self-Consistency

```python
def self_consistency_method():
    """Self-consistency: Sample multiple outputs, take majority vote."""

    print("Self-Consistency for Improved ICL:\n")

    print("Algorithm:")
    print("  1. Generate N different outputs (with temperature > 0)")
    print("  2. For each output, extract the answer")
    print("  3. Take majority vote across outputs")
    print("  4. Return most common answer")

    print("\nExample (math problem):")
    print("  Question: 'What is 15% of 80?'")
    print("\n  Sampled outputs:")
    print("    Output 1: 'Let me calculate: 80 * 0.15 = 12'")
    print("    Output 2: '15% of 80 is 12'")
    print("    Output 3: 'We need 80 * 0.15 = 11.8 ≈ 12'")
    print("    Output 4: '15/100 * 80 = 12'")
    print("    Output 5: '80 * 0.15 = 12.0'")

    print("\n  Extracted answers: [12, 12, 12, 12, 12]")
    print("  Majority vote: 12 ✓")

    print("\nBenefits:")
    print("  • Improves accuracy by ~5-20% on reasoning tasks")
    print("  • Robust to individual errors")
    print("  • No prompt engineering needed")

    print("\nCost:")
    print("  • N times more expensive (N=5-40 typical)")
    print("  • Only practical for important queries")

self_consistency_method()
```

### Chain-of-Thought in ICL

```python
def cot_in_icl():
    """Using chain-of-thought reasoning in few-shot prompts."""

    print("Chain-of-Thought in In-Context Learning:\n")

    # Without CoT
    print("Standard few-shot (worse):")
    standard = """
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls. How many does he have?
A: 11

Q: The cafeteria had 23 apples. If they used 20, then bought 6 more, how many do they have?
A:"""
    print(standard)
    print(" 9")  # Direct answer

    print("\n" + "="*60)

    # With CoT
    print("\nChain-of-thought few-shot (better):")
    cot = """
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls. How many does he have?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20, then bought 6 more, how many do they have?
A:"""
    print(cot)
    print(" They started with 23 apples. After using 20, they have 23 - 20 = 3. Then they buy 6 more, so 3 + 6 = 9. The answer is 9.")

    print("\n" + "="*60)
    print("\nBenefits of CoT in ICL:")
    print("  • Better accuracy on reasoning tasks (+10-30%)")
    print("  • Model shows its work (interpretable)")
    print("  • Can catch errors in reasoning")
    print("  • Works best with large models (60B+)")

cot_in_icl()
```

### Instruction + ICL Hybrid

```python
def instruction_icl_hybrid():
    """Combine instructions with few-shot examples."""

    print("Hybrid: Instruction + Few-Shot Examples\n")

    prompt = """
You are an expert at extracting structured information from text.
Extract all dates, locations, and people mentioned in the text.
Format your output as JSON.

Example:

Text: "On May 15, 2023, Alice met Bob in Paris."
Output:
{
  "dates": ["May 15, 2023"],
  "locations": ["Paris"],
  "people": ["Alice", "Bob"]
}

Example:

Text: "Dr. Smith visited London and Berlin in March 2024."
Output:
{
  "dates": ["March 2024"],
  "locations": ["London", "Berlin"],
  "people": ["Dr. Smith"]
}

Now extract from this text:

Text: "Professor Chen gave a lecture in Tokyo on January 10, 2025."
Output:"""

    print(prompt)

    print("\n\nWhy this works:")
    print("  • Instruction provides general guidance")
    print("  • Examples demonstrate specific format")
    print("  • Combination better than either alone")
    print("  • Especially effective with instruction-tuned models")

instruction_icl_hybrid()
```

## Summary

**Key Concepts**:

1. **In-context learning (ICL)** is the ability to learn tasks from examples in the prompt without parameter updates
2. **Zero-shot, one-shot, few-shot** represent increasing levels of ICL with 0, 1, or 2-10 examples
3. **ICL emerges at scale** -- models need ~1-6B parameters for basic ICL, 30B+ for strong ICL
4. **Demonstration quality matters** more than quantity -- relevance, diversity, and correctness are key
5. **Prompt format significantly affects** ICL performance -- consistent formatting is crucial
6. **ICL is practical but imperfect** -- achieves 70-90% of fine-tuned performance with near-zero training
7. **Advanced techniques** like self-consistency and chain-of-thought can boost ICL performance substantially

**ICL Spectrum**:

```
Zero-shot: Task description only
  ↓ (+1 example)
One-shot: Single example for format
  ↓ (+1-9 examples)
Few-shot: Multiple examples (2-10)
  ↓ (+10-100 examples)
Many-shot: Many examples (long context)
  ↓ (100-10K examples)
Fine-tuning: Parameter updates
```

**When ICL Works Best**:

- ✅ Large model (7B+ parameters, 30B+ for complex tasks)
- ✅ Simple to moderate task complexity
- ✅ Clear examples that demonstrate the pattern
- ✅ Consistent formatting
- ✅ Task seen during pretraining
- ✅ Fast iteration and deployment needed

**When to Fine-Tune Instead**:

- ❌ Complex or specialized task
- ❌ Have large labeled dataset (1000+)
- ❌ Need maximum performance
- ❌ Task very different from pretraining
- ❌ Production deployment with cost sensitivity

**Best Practices**:

1. Start with zero-shot, add examples if needed
2. Use 3-5 diverse, relevant examples
3. Maintain consistent formatting
4. Test multiple prompt variations
5. Use self-consistency for critical tasks
6. Combine instructions with examples
7. Consider fine-tuning for complex tasks

**Key Limitations**:

- Sensitive to prompt wording and format
- Performance below fine-tuned models
- Context window limits number of examples
- May learn surface patterns instead of task
- Unreliable for very complex reasoning

## Next Steps

- Study [Chain-of-Thought Reasoning](chain-of-thought.md) for complex problem solving with ICL
- Learn [Instruction Tuning](instruction-tuning.md) to improve zero-shot and ICL performance
- Explore [Prompt Engineering](../prompt_engineering/prompt-design.md) for crafting effective prompts
- Understand [Few-Shot Learning Techniques](../prompt_engineering/few-shot-prompting.md) in depth
- Study [Model Capabilities](capabilities-limitations.md) to know when ICL is sufficient
- Learn [Prompt Optimization](../prompt_engineering/prompt-optimization.md) for systematic improvement
