# Prompt Optimization

## Table of Contents

- [Introduction](#introduction)
- [The Optimization Process](#the-optimization-process)
- [Iterative Refinement](#iterative-refinement)
- [A/B Testing Prompts](#ab-testing-prompts)
- [Evaluation-Driven Optimization](#evaluation-driven-optimization)
- [Debugging Poor Outputs](#debugging-poor-outputs)
- [Identifying Failure Patterns](#identifying-failure-patterns)
- [Performance Metrics](#performance-metrics)
- [Maintaining Prompt Libraries](#maintaining-prompt-libraries)
- [Optimization Strategies](#optimization-strategies)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Prompt optimization** is the systematic process of improving prompts to achieve better, more consistent results from language models. Like tuning parameters in traditional ML, optimizing prompts can dramatically improve performance.

```
Initial prompt (60% accuracy):
"Classify the sentiment"

Optimized prompt (85% accuracy):
"Classify the sentiment of this text as positive, negative, or neutral.

Examples:
Text: "Great product!" → positive
Text: "Terrible experience" → negative
Text: "It's okay" → neutral

Text: [input]
Sentiment:"
```

**Why optimization matters**:

- Initial prompts rarely optimal
- Small changes can have large effects
- Different tasks need different approaches
- Performance varies across inputs
- Cost/quality trade-offs to balance

This guide teaches you systematic methods for optimizing prompts.

## The Optimization Process

```python
def optimization_workflow():
    """The systematic prompt optimization workflow."""

    print("Prompt Optimization Workflow:\n")

    print("""
┌─────────────────┐
│  1. BASELINE    │  Create initial prompt, measure performance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. ANALYZE     │  Identify failure patterns, edge cases
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. HYPOTHESIZE │  Form hypothesis about improvement
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. MODIFY      │  Make targeted change to prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. TEST        │  Evaluate on same test set
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. COMPARE     │  Better? Keep. Worse? Revert.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  7. REPEAT      │  Iterate until satisfied
└─────────────────┘
""")

    print("=" * 60)
    print("\nKey Principles:\n")

    principles = [
        'Start with baseline measurement',
        'Change one thing at a time',
        'Test on consistent evaluation set',
        'Keep what works, discard what doesn\'t',
        'Document changes and results',
        'Know when to stop (diminishing returns)'
    ]

    for i, principle in enumerate(principles, 1):
        print(f"  {i}. {principle}")

    print("\n" + "=" * 60)
    print("\nExample Optimization Session:\n")

    session = [
        ('v1 (baseline)', 'Classify sentiment', '65%'),
        ('v2', '+ Add examples', '72% (+7%)'),
        ('v3', '+ Specify output format', '74% (+2%)'),
        ('v4', '+ Add constraints', '76% (+2%)'),
        ('v5', '+ More diverse examples', '81% (+5%)'),
        ('v6', '+ Chain-of-thought', '78% (-3%, revert)'),
        ('Final', 'v5', '81% total improvement: +16%')
    ]

    print(f"{'Version':<12} {'Change':<30} {'Accuracy'}")
    print("=" * 60)

    for version, change, accuracy in session:
        print(f"{version:<12} {change:<30} {accuracy}")

optimization_workflow()
```

## Iterative Refinement

### Version Control for Prompts

```python
def prompt_versioning():
    """Tracking prompt versions and changes."""

    print("Prompt Versioning:\n")

    versions = {
        'v1.0 (baseline)': {
            'prompt': 'Extract the main entities from this text.',
            'performance': '45% F1',
            'notes': 'Too vague, inconsistent output format'
        },
        'v1.1': {
            'prompt': 'Extract person names, organizations, and locations from this text. Return as JSON.',
            'performance': '62% F1 (+17%)',
            'notes': 'Better with explicit entity types and format'
        },
        'v1.2': {
            'prompt': 'Extract entities as JSON:\n{"people": [], "organizations": [], "locations": []}',
            'performance': '68% F1 (+6%)',
            'notes': 'Template helps with consistent format'
        },
        'v2.0': {
            'prompt': 'v1.2 + 3 few-shot examples',
            'performance': '79% F1 (+11%)',
            'notes': 'Examples significantly improved accuracy'
        },
        'v2.1': {
            'prompt': 'v2.0 + handling nicknames/abbreviations in instructions',
            'performance': '82% F1 (+3%)',
            'notes': 'Addressed edge case failures'
        }
    }

    print(f"{'Version':<18} {'Performance':<15} {'Change'}")
    print("=" * 70)

    for version, info in versions.items():
        change = info['notes']
        print(f"{version:<18} {info['performance']:<15} {change}")

    print("\n" + "=" * 60)
    print("\nVersioning Best Practices:\n")

    practices = [
        'Use semantic versioning (major.minor)',
        'Document what changed in each version',
        'Track performance metrics for each',
        'Keep old versions for rollback',
        'Note which version is in production',
        'Link to eval results'
    ]

    for practice in practices:
        print(f"  • {practice}")

prompt_versioning()
```

### Systematic Refinement

```python
def systematic_refinement():
    """Systematic approach to refining prompts."""

    print("\n\nSystematic Refinement Process:\n")

    print("STEP 1: Identify specific weaknesses\n")

    weaknesses = '''
Test results show:
  • Fails on sarcastic text (20% accuracy)
  • Misses multi-word entities (45% recall)
  • Inconsistent output format (30% invalid JSON)
  • Confused by ambiguous abbreviations (35% accuracy)
'''
    print(weaknesses)

    print("STEP 2: Prioritize by impact\n")

    priorities = [
        ('1. Format consistency', 'Affects 30% of outputs, breaks pipeline'),
        ('2. Sarcasm detection', 'Affects 20%, high visibility errors'),
        ('3. Multi-word entities', 'Affects 15%, missing important info'),
        ('4. Abbreviations', 'Affects 10%, edge case')
    ]

    for priority, reason in priorities:
        print(f"  {priority}")
        print(f"     → {reason}")

    print("\nSTEP 3: Address highest priority\n")

    solution = '''
Issue: Format consistency (30% invalid JSON)

Hypothesis: Examples don't show enough format variety

Change: Add 3 more examples showing different field combinations
        (some with nulls, some with empty arrays)

Test: Run on 100-sample dev set

Result: Invalid JSON dropped from 30% → 8% ✓
'''
    print(solution)

    print("STEP 4: Move to next priority and repeat")

    print("\n" + "=" * 60)
    print("\nRefinement Checklist:\n")

    checklist = [
        '□ Tested on representative data',
        '□ Identified failure patterns',
        '□ Prioritized by impact',
        '□ Hypothesized root cause',
        '□ Made targeted change',
        '□ Measured improvement',
        '□ Documented results',
        '□ Updated version'
    ]

    for item in checklist:
        print(f"  {item}")

systematic_refinement()
```

## A/B Testing Prompts

### Setting Up A/B Tests

```python
def ab_testing_setup():
    """Setting up A/B tests for prompt variations."""

    print("A/B Testing Prompts:\n")

    print("Concept: Compare two prompt versions on same data\n")

    code = '''
import random
from typing import List, Dict

def ab_test_prompts(
    prompt_a: str,
    prompt_b: str,
    test_inputs: List[str],
    eval_fn: callable,
    runs_per_prompt: int = 3
) -> Dict:
    """
    Run A/B test comparing two prompts.

    Args:
        prompt_a: First prompt variant
        prompt_b: Second prompt variant
        test_inputs: List of test inputs
        eval_fn: Function to evaluate output quality (returns 0-1 score)
        runs_per_prompt: Number of runs per prompt (for variance)

    Returns:
        Results dictionary with scores and winner
    """

    results = {'A': [], 'B': []}

    for test_input in test_inputs:
        # Test prompt A
        for _ in range(runs_per_prompt):
            output = llm_call(prompt_a.format(input=test_input))
            score = eval_fn(output, test_input)
            results['A'].append(score)

        # Test prompt B
        for _ in range(runs_per_prompt):
            output = llm_call(prompt_b.format(input=test_input))
            score = eval_fn(output, test_input)
            results['B'].append(score)

    # Calculate statistics
    import numpy as np

    stats = {
        'A': {
            'mean': np.mean(results['A']),
            'std': np.std(results['A']),
            'scores': results['A']
        },
        'B': {
            'mean': np.mean(results['B']),
            'std': np.std(results['B']),
            'scores': results['B']
        }
    }

    # Determine winner
    diff = stats['B']['mean'] - stats['A']['mean']

    if diff > 0.05:  # B is meaningfully better
        winner = 'B'
    elif diff < -0.05:  # A is meaningfully better
        winner = 'A'
    else:
        winner = 'Tie'

    stats['winner'] = winner
    stats['difference'] = diff

    return stats

# Example usage
prompt_a = "Classify sentiment: {input}"
prompt_b = "Classify sentiment as positive/negative/neutral: {input}"

test_inputs = ["Great product!", "Terrible", "It's okay"]

def eval_sentiment(output, input):
    # Return 1 if correct, 0 if wrong
    # (simplified - real eval would compare to ground truth)
    return 1.0

results = ab_test_prompts(prompt_a, prompt_b, test_inputs, eval_sentiment)
print(f"Winner: {results['winner']}")
print(f"Difference: {results['difference']:.3f}")
'''

    print(code)

ab_testing_setup()
```

### Analyzing A/B Results

```python
def ab_results_analysis():
    """Analyzing and interpreting A/B test results."""

    print("\n\nAnalyzing A/B Test Results:\n")

    print("Example Results:\n")

    results = '''
Prompt A (baseline):
  Mean accuracy: 0.742
  Std dev: 0.089
  Samples: 100

Prompt B (with examples):
  Mean accuracy: 0.813
  Std dev: 0.074
  Samples: 100

Difference: +0.071 (+7.1 percentage points)
Statistical significance: p < 0.01 ✓
'''

    print(results)

    print("=" * 60)
    print("\nInterpretation:\n")

    interpretation = [
        'Prompt B is better: +7.1% accuracy',
        'Difference is statistically significant (p < 0.01)',
        'Prompt B also more consistent (lower std dev)',
        'Decision: Adopt Prompt B'
    ]

    for point in interpretation:
        print(f"  • {point}")

    print("\n" + "=" * 60)
    print("\nWhen to Do A/B Testing:\n")

    scenarios = {
        'Comparing approaches': 'Few-shot vs zero-shot',
        'Testing hypothesis': 'Will examples help?',
        'Format changes': 'JSON template vs freeform',
        'Instruction phrasing': 'Different ways to phrase task',
        'Example selection': 'Different example sets',
        'Trade-off decisions': 'Accuracy vs length vs cost'
    }

    for scenario, example in scenarios.items():
        print(f"  • {scenario}: {example}")

    print("\n" + "=" * 60)
    print("\nA/B Testing Checklist:\n")

    checklist = [
        '□ Same test set for both prompts',
        '□ Multiple runs per prompt (for variance)',
        '□ Large enough sample size (50+ examples)',
        '□ Clear evaluation metric',
        '□ Statistical significance test',
        '□ Consider practical significance (is diff meaningful?)',
        '□ Test on multiple data distributions',
        '□ Account for cost differences'
    ]

    for item in checklist:
        print(f"  {item}")

ab_results_analysis()
```

## Evaluation-Driven Optimization

### Building Evaluation Sets

```python
def building_eval_sets():
    """Creating effective evaluation datasets."""

    print("Building Evaluation Sets:\n")

    print("What makes a good eval set?\n")

    characteristics = {
        'Representative': 'Covers real-world distribution',
        'Diverse': 'Includes edge cases and variations',
        'Sufficient size': '50-100 examples minimum',
        'Ground truth': 'Correct answers known',
        'Stratified': 'Balanced across categories/difficulty',
        'Stable': 'Doesn\'t change between tests',
        'Separate from train': 'Not used for prompt development'
    }

    for char, description in characteristics.items():
        print(f"  • {char}: {description}")

    print("\n" + "=" * 60)
    print("\nEval Set Structure:\n")

    example = '''
# eval_set.json
[
  {
    "id": "001",
    "input": "This movie was fantastic!",
    "expected_output": "positive",
    "category": "clear_positive",
    "difficulty": "easy"
  },
  {
    "id": "002",
    "input": "Not the worst thing I've seen",
    "expected_output": "neutral",
    "category": "double_negative",
    "difficulty": "hard"
  },
  {
    "id": "003",
    "input": "Yeah, real great job on that one 🙄",
    "expected_output": "negative",
    "category": "sarcasm",
    "difficulty": "hard"
  },
  ...
]
'''
    print(example)

    print("\n" + "=" * 60)
    print("\nStratification Strategy:\n")

    strategy = '''
Include examples from each category:

By sentiment distribution:
  • 40% positive
  • 30% negative
  • 30% neutral

By difficulty:
  • 50% easy (clear sentiment)
  • 30% medium (context needed)
  • 20% hard (sarcasm, ambiguity)

By length:
  • 40% short (1 sentence)
  • 40% medium (2-3 sentences)
  • 20% long (paragraph)

By domain:
  • 25% products
  • 25% services
  • 25% experiences
  • 25% other
'''
    print(strategy)

    print("\nSplit Strategy:")
    print("  • Dev set: 100 examples (for prompt development)")
    print("  • Test set: 200 examples (for final evaluation)")
    print("  • Don't use test set during optimization!")

building_eval_sets()
```

### Automated Evaluation

```python
def automated_evaluation():
    """Automating prompt evaluation."""

    print("\n\nAutomated Evaluation:\n")

    code = '''
import json
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class EvalResult:
    """Results from evaluating a prompt."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_category: Dict[str, float]
    failures: List[Dict]

def evaluate_prompt(
    prompt_template: str,
    eval_set: List[Dict],
    category_field: str = 'category'
) -> EvalResult:
    """
    Evaluate a prompt on an evaluation set.

    Args:
        prompt_template: Prompt with {input} placeholder
        eval_set: List of examples with input and expected_output
        category_field: Field name for stratification

    Returns:
        EvalResult with metrics
    """

    correct = 0
    total = len(eval_set)
    category_results = {}
    failures = []

    # Track per-category performance
    category_correct = {}
    category_total = {}

    for example in eval_set:
        # Generate output
        prompt = prompt_template.format(input=example['input'])
        output = llm_call(prompt).strip().lower()
        expected = example['expected_output'].strip().lower()

        # Check if correct
        is_correct = (output == expected)

        if is_correct:
            correct += 1
        else:
            failures.append({
                'input': example['input'],
                'expected': expected,
                'got': output,
                'category': example.get(category_field, 'unknown')
            })

        # Track by category
        category = example.get(category_field, 'unknown')
        if category not in category_correct:
            category_correct[category] = 0
            category_total[category] = 0

        category_total[category] += 1
        if is_correct:
            category_correct[category] += 1

    # Calculate metrics
    accuracy = correct / total

    # Per-category accuracy
    per_category = {
        cat: category_correct[cat] / category_total[cat]
        for cat in category_total
    }

    # Calculate precision, recall, F1 (for classification)
    # (simplified - real implementation would handle multi-class)
    precision = accuracy  # Simplified
    recall = accuracy
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return EvalResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        per_category=per_category,
        failures=failures
    )

# Usage
with open('eval_set.json') as f:
    eval_set = json.load(f)

prompt = """
Classify the sentiment as positive, negative, or neutral.

Text: {input}
Sentiment:"""

results = evaluate_prompt(prompt, eval_set)

print(f"Accuracy: {results.accuracy:.1%}")
print(f"F1 Score: {results.f1:.1%}")
print("\\nPer-category accuracy:")
for category, acc in results.per_category.items():
    print(f"  {category}: {acc:.1%}")

print(f"\\nFailures: {len(results.failures)}")
for failure in results.failures[:5]:  # Show first 5
    print(f"  Expected {failure['expected']}, got {failure['got']}")
    print(f"  Input: {failure['input'][:50]}...")
'''

    print(code)

automated_evaluation()
```

## Debugging Poor Outputs

### Failure Analysis

```python
def failure_analysis():
    """Analyzing why prompts fail."""

    print("Failure Analysis Process:\n")

    print("STEP 1: Collect failures\n")

    failures = '''
Run prompt on eval set, collect all errors:

Example failures (sentiment classification):
1. Input: "Not bad at all"
   Expected: positive
   Got: neutral

2. Input: "Yeah, real great job 🙄"
   Expected: negative (sarcasm)
   Got: positive

3. Input: "The product works as intended"
   Expected: neutral
   Got: positive

4. Input: "Could be better but okay"
   Expected: neutral
   Got: negative
...
'''
    print(failures)

    print("STEP 2: Categorize failures\n")

    categories = {
        'Sarcasm misunderstanding': ['Examples 2, 7, 12', '3 failures (15%)'],
        'Ambiguous neutral/positive': ['Examples 1, 3, 8', '3 failures (15%)'],
        'Double negatives': ['Examples 5, 11', '2 failures (10%)'],
        'Context needed': ['Examples 4, 9, 15', '3 failures (15%)'],
        'Emoji misinterpretation': ['Examples 2, 6, 13', '3 failures (15%)']
    }

    print(f"{'Failure Type':<30} {'Examples':<20} {'Count'}")
    print("=" * 70)

    for failure_type, (examples, count) in categories.items():
        print(f"{failure_type:<30} {examples:<20} {count}")

    print("\nSTEP 3: Identify root causes\n")

    root_causes = '''
Pattern: Sarcasm misunderstanding (highest frequency)

Root cause analysis:
  • Prompt doesn't mention sarcasm
  • No sarcasm examples
  • Model takes text at face value
  • Emoji context clues ignored

Potential fixes:
  1. Add instruction: "Watch for sarcasm"
  2. Include sarcasm examples
  3. Note that emojis provide context
  4. Ask model to consider tone
'''
    print(root_causes)

    print("\nSTEP 4: Hypothesize and test fixes")

failure_analysis()
```

### Debugging Checklist

```python
def debugging_checklist():
    """Checklist for debugging poor prompt performance."""

    print("\n\nPrompt Debugging Checklist:\n")

    checklist = {
        'Instruction clarity': {
            'questions': [
                'Is the task clearly stated?',
                'Are requirements explicit?',
                'Could it be misinterpreted?'
            ],
            'fixes': ['Rephrase more clearly', 'Add examples', 'Break into sub-tasks']
        },
        'Output format': {
            'questions': [
                'Is desired format specified?',
                'Are examples consistent?',
                'Is format validated?'
            ],
            'fixes': ['Add format instructions', 'Provide template', 'Use few-shot']
        },
        'Context': {
            'questions': [
                'Is necessary context provided?',
                'Is context relevant?',
                'Too much irrelevant context?'
            ],
            'fixes': ['Add missing context', 'Remove noise', 'Prioritize info']
        },
        'Examples': {
            'questions': [
                'Are examples representative?',
                'Do they cover edge cases?',
                'Are they high quality?'
            ],
            'fixes': ['Add more examples', 'Include edge cases', 'Fix bad examples']
        },
        'Constraints': {
            'questions': [
                'Are constraints clear?',
                'Are they followed?',
                'Are they conflicting?'
            ],
            'fixes': ['Make constraints explicit', 'Prioritize constraints', 'Resolve conflicts']
        }
    }

    for category, info in checklist.items():
        print(f"{category.upper()}:")
        print("  Questions:")
        for q in info['questions']:
            print(f"    • {q}")
        print("  Potential fixes:")
        for fix in info['fixes']:
            print(f"    → {fix}")
        print()

debugging_checklist()
```

## Identifying Failure Patterns

### Pattern Detection

```python
def pattern_detection():
    """Detecting patterns in prompt failures."""

    print("Detecting Failure Patterns:\n")

    code = '''
from collections import Counter, defaultdict
import re

def analyze_failure_patterns(failures: List[Dict]) -> Dict:
    """
    Analyze patterns in prompt failures.

    Args:
        failures: List of failure dictionaries with input, expected, got

    Returns:
        Dictionary of detected patterns
    """

    patterns = {
        'by_category': Counter(),
        'by_length': defaultdict(list),
        'by_keywords': Counter(),
        'by_error_type': Counter()
    }

    for failure in failures:
        input_text = failure['input']
        category = failure.get('category', 'unknown')
        expected = failure['expected']
        got = failure['got']

        # Category pattern
        patterns['by_category'][category] += 1

        # Length pattern
        length = len(input_text.split())
        length_bucket = '0-10' if length <= 10 else '11-20' if length <= 20 else '20+'
        patterns['by_length'][length_bucket].append(failure)

        # Keyword pattern
        keywords = ['not', 'but', 'however', 'although', 'emoji', 'sarcasm']
        for keyword in keywords:
            if keyword in input_text.lower():
                patterns['by_keywords'][keyword] += 1

        # Error type pattern
        if expected == 'positive' and got == 'negative':
            patterns['by_error_type']['pos_to_neg'] += 1
        elif expected == 'negative' and got == 'positive':
            patterns['by_error_type']['neg_to_pos'] += 1
        elif got == 'neutral':
            patterns['by_error_type']['classified_as_neutral'] += 1

    return patterns

# Usage
failures = [...]  # Your failures list
patterns = analyze_failure_patterns(failures)

print("Most common failure categories:")
for category, count in patterns['by_category'].most_common(5):
    print(f"  {category}: {count} failures")

print("\\nKeywords in failures:")
for keyword, count in patterns['by_keywords'].most_common():
    print(f"  '{keyword}': {count} occurrences")

print("\\nError type distribution:")
for error_type, count in patterns['by_error_type'].items():
    print(f"  {error_type}: {count}")
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nExample Pattern Analysis Output:\n")

    output = '''
Most common failure categories:
  sarcasm: 12 failures (30%)
  double_negative: 8 failures (20%)
  neutral_confusion: 7 failures (17.5%)
  context_dependent: 6 failures (15%)
  emoji_context: 5 failures (12.5%)

Keywords in failures:
  'not': 15 occurrences
  'but': 10 occurrences
  'emoji': 8 occurrences
  'however': 5 occurrences

Error type distribution:
  neg_to_pos: 18 (45%)  ← Biggest issue
  pos_to_neg: 10 (25%)
  classified_as_neutral: 12 (30%)

INSIGHT: Model frequently misclassifies negative as positive,
especially with sarcasm and negation. Need examples with these patterns.
'''
    print(output)

pattern_detection()
```

## Performance Metrics

### Choosing Metrics

```python
def choosing_metrics():
    """Selecting appropriate metrics for prompt optimization."""

    print("Choosing Performance Metrics:\n")

    metrics = {
        'Accuracy': {
            'when': 'Classification, balanced classes',
            'formula': 'correct / total',
            'pros': 'Simple, interpretable',
            'cons': 'Poor for imbalanced data'
        },
        'F1 Score': {
            'when': 'Classification, imbalanced classes',
            'formula': '2 * (precision * recall) / (precision + recall)',
            'pros': 'Balances precision and recall',
            'cons': 'Less intuitive'
        },
        'BLEU': {
            'when': 'Text generation, translation',
            'formula': 'N-gram overlap with reference',
            'pros': 'Standard for generation',
            'cons': 'Can miss semantic similarity'
        },
        'ROUGE': {
            'when': 'Summarization',
            'formula': 'Overlap with reference summary',
            'pros': 'Good for summaries',
            'cons': 'Multiple variants'
        },
        'Exact Match': {
            'when': 'Structured output, QA',
            'formula': 'output == expected',
            'pros': 'Strict, clear pass/fail',
            'cons': 'Doesn\'t reward close answers'
        },
        'Human Eval': {
            'when': 'Complex, subjective tasks',
            'formula': 'Human judgment score',
            'pros': 'Captures quality',
            'cons': 'Expensive, slow'
        },
        'Latency': {
            'when': 'Production systems',
            'formula': 'Time to generate',
            'pros': 'User experience metric',
            'cons': 'Not about quality'
        },
        'Cost': {
            'when': 'Production at scale',
            'formula': 'Tokens used × price',
            'pros': 'Real business metric',
            'cons': 'Not about quality'
        }
    }

    print(f"{'Metric':<15} {'Best For':<30} {'Pros':<30} {'Cons'}")
    print("=" * 95)

    for metric, info in metrics.items():
        print(f"{metric:<15} {info['when']:<30} {info['pros']:<30} {info['cons']}")

    print("\n" + "=" * 60)
    print("\nMulti-Metric Optimization:\n")

    example = '''
Optimize for multiple objectives:

Primary: F1 score (quality)
  Target: > 85%

Secondary: Cost per query
  Target: < $0.01

Tertiary: Latency
  Target: < 2 seconds

Approach: Optimize primary first, then tune others without
         sacrificing primary metric significantly
'''
    print(example)

choosing_metrics()
```

### Tracking Metrics Over Time

```python
def metrics_tracking():
    """Tracking metrics across prompt versions."""

    print("\n\nTracking Metrics Over Time:\n")

    history = [
        ('2024-01-01', 'v1.0', 0.65, 250, 0.008, 1.2),
        ('2024-01-05', 'v1.1', 0.72, 280, 0.009, 1.4),
        ('2024-01-10', 'v2.0', 0.79, 320, 0.011, 1.8),
        ('2024-01-15', 'v2.1', 0.82, 310, 0.010, 1.7),
        ('2024-01-20', 'v3.0', 0.85, 290, 0.010, 1.5),
    ]

    print(f"{'Date':<12} {'Version':<10} {'F1':<8} {'Tokens':<10} {'Cost':<10} {'Latency'}")
    print("=" * 70)

    for date, version, f1, tokens, cost, latency in history:
        print(f"{date:<12} {version:<10} {f1:<8.2f} {tokens:<10} ${cost:<9.3f} {latency}s")

    print("\n" + "=" * 60)
    print("\nInsights from Tracking:\n")

    insights = '''
• v1.0 → v1.1: +7% F1, modest cost increase (+12.5%)
• v1.1 → v2.0: +7% F1, but 22% cost increase (added examples)
• v2.0 → v2.1: +3% F1, reduced cost (better examples)
• v2.1 → v3.0: +3% F1, similar cost (refinements)

Overall: +20 percentage points F1, +25% cost
→ Good trade-off for production use

v3.0 selected for deployment: best F1, acceptable cost/latency
'''
    print(insights)

    print("\n" + "=" * 60)
    print("\nMetrics Dashboard:\n")

    print('''
┌─────────────────────────────────────────────┐
│  Prompt Performance Dashboard               │
├─────────────────────────────────────────────┤
│  Current Version: v3.0                      │
│  Last Updated: 2024-01-20                   │
│                                             │
│  Quality Metrics:                           │
│    F1 Score:     0.85  ████████████░░  85%  │
│    Accuracy:     0.87  ████████████░░  87%  │
│    Precision:    0.83  ████████████░░  83%  │
│    Recall:       0.87  ████████████░░  87%  │
│                                             │
│  Efficiency Metrics:                        │
│    Avg Tokens:   290                        │
│    Avg Cost:     $0.010                     │
│    Avg Latency:  1.5s                       │
│                                             │
│  Compared to baseline (v1.0):               │
│    F1: +30% ↑                               │
│    Cost: +25% ↑                             │
│    Latency: +25% ↑                          │
└─────────────────────────────────────────────┘
''')

metrics_tracking()
```

## Maintaining Prompt Libraries

### Organizing Prompts

````python
def organizing_prompts():
    """Organizing and maintaining prompt libraries."""

    print("Organizing Prompt Libraries:\n")

    print("Directory Structure:\n")

    structure = '''
prompts/
├── README.md                    # Overview and guidelines
├── sentiment_analysis/
│   ├── v1.0_baseline.txt
│   ├── v2.0_with_examples.txt
│   ├── v3.0_production.txt      ← Current production
│   ├── eval_set.json
│   └── results.json
├── entity_extraction/
│   ├── v1.0_basic.txt
│   ├── v2.0_structured.txt      ← Current production
│   ├── eval_set.json
│   └── results.json
├── summarization/
│   ├── v1.0_simple.txt
│   ├── v1.1_with_constraints.txt ← Current production
│   ├── eval_set.json
│   └── results.json
└── templates/
    ├── classification_template.txt
    ├── extraction_template.txt
    └── generation_template.txt
'''
    print(structure)

    print("=" * 60)
    print("\nPrompt File Format:\n")

    prompt_file = '''
# Sentiment Analysis - v3.0
# Status: PRODUCTION
# Created: 2024-01-15
# Last Updated: 2024-01-20
# Performance: F1=0.85, Accuracy=0.87
# Cost: ~$0.010 per query
# Owner: team@company.com

## Description
Classifies text sentiment as positive, negative, or neutral.
Handles sarcasm and double negatives.

## Prompt
---
Classify the sentiment of the following text as positive, negative, or neutral.

Consider:
- Sarcasm (e.g., "Yeah, great job 🙄" is negative)
- Double negatives (e.g., "not bad" is positive)
- Context and tone

Examples:

Text: "This product is amazing!"
Sentiment: positive

Text: "Worst experience ever."
Sentiment: negative

Text: "It's okay, nothing special."
Sentiment: neutral

Text: "Oh sure, this works perfectly... NOT"
Sentiment: negative

Text: {input}
Sentiment:
---

## Usage
```python
from prompts import load_prompt

prompt = load_prompt('sentiment_analysis/v3.0_production.txt')
result = llm_call(prompt.format(input=user_text))
````

## Changelog

- v3.0: Added sarcasm handling examples (+3% F1)
- v2.0: Added few-shot examples (+14% F1)
- v1.0: Initial baseline
  '''
  print(prompt_file)

organizing_prompts()

````

### Prompt Management System

```python
def prompt_management_system():
    """Building a prompt management system."""

    print("\n\nPrompt Management System:\n")

    code = '''
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PromptVersion:
    """Represents a versioned prompt."""
    name: str
    version: str
    prompt_text: str
    status: str  # development, staging, production, deprecated
    performance: dict
    metadata: dict

class PromptLibrary:
    """Manages a library of prompts."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.prompts = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load all prompts from library."""
        for prompt_dir in self.base_path.iterdir():
            if prompt_dir.is_dir():
                self.prompts[prompt_dir.name] = self._load_versions(prompt_dir)

    def _load_versions(self, prompt_dir: Path) -> List[PromptVersion]:
        """Load all versions of a prompt."""
        versions = []
        for file in prompt_dir.glob('v*.txt'):
            version = PromptVersion(
                name=prompt_dir.name,
                version=file.stem,
                prompt_text=file.read_text(),
                status=self._get_status(file),
                performance=self._load_results(prompt_dir),
                metadata=self._parse_metadata(file)
            )
            versions.append(version)
        return versions

    def get_prompt(self, name: str, version: Optional[str] = None) -> PromptVersion:
        """Get a prompt by name and optional version."""
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found")

        versions = self.prompts[name]

        if version:
            for v in versions:
                if v.version == version:
                    return v
            raise ValueError(f"Version '{version}' not found")

        # Return production version or latest
        prod = [v for v in versions if v.status == 'production']
        if prod:
            return prod[0]

        return max(versions, key=lambda v: v.version)

    def compare_versions(self, name: str, v1: str, v2: str) -> dict:
        """Compare two versions of a prompt."""
        version1 = self.get_prompt(name, v1)
        version2 = self.get_prompt(name, v2)

        return {
            'v1': {
                'version': v1,
                'performance': version1.performance
            },
            'v2': {
                'version': v2,
                'performance': version2.performance
            },
            'diff': {
                metric: version2.performance.get(metric, 0) - version1.performance.get(metric, 0)
                for metric in version1.performance
            }
        }

    def promote_version(self, name: str, version: str):
        """Promote a version to production."""
        prompt = self.get_prompt(name, version)

        # Demote current production
        for v in self.prompts[name]:
            if v.status == 'production':
                v.status = 'deprecated'

        # Promote new version
        prompt.status = 'production'

        # Save changes
        self._save_prompts()

# Usage
library = PromptLibrary('prompts/')

# Get production prompt
prompt = library.get_prompt('sentiment_analysis')
print(f"Using: {prompt.name} {prompt.version}")
print(f"Performance: {prompt.performance}")

# Compare versions
comparison = library.compare_versions('sentiment_analysis', 'v2.0', 'v3.0')
print(f"Improvement: {comparison['diff']}")

# Promote new version
library.promote_version('sentiment_analysis', 'v3.1')
'''

    print(code)

prompt_management_system()
````

## Optimization Strategies

### Optimization Patterns

```python
def optimization_patterns():
    """Common optimization patterns and strategies."""

    print("Common Optimization Patterns:\n")

    patterns = {
        'Start simple, add complexity': {
            'approach': 'Begin with zero-shot, add examples if needed',
            'when': 'New tasks, unclear requirements',
            'benefit': 'Avoid premature optimization'
        },
        'Add examples incrementally': {
            'approach': 'Start with 1 example, add more until plateau',
            'when': 'Few-shot prompting',
            'benefit': 'Find minimum examples needed'
        },
        'Target weakest category': {
            'approach': 'Improve performance on worst-performing subset',
            'when': 'Uneven performance across categories',
            'benefit': 'Balanced improvement'
        },
        'Remove then add back': {
            'approach': 'Strip prompt to minimum, add back one piece at a time',
            'when': 'Overly complex prompt',
            'benefit': 'Identify what actually helps'
        },
        'Copy successful patterns': {
            'approach': 'Adapt prompts from similar tasks',
            'when': 'Starting new task',
            'benefit': 'Faster initial results'
        },
        'Constraint relaxation': {
            'approach': 'Remove constraints, see if quality improves',
            'when': 'Over-constrained outputs',
            'benefit': 'May be unnecessarily limiting'
        }
    }

    for pattern, info in patterns.items():
        print(f"{pattern.upper()}:")
        print(f"  Approach: {info['approach']}")
        print(f"  When: {info['when']}")
        print(f"  Benefit: {info['benefit']}")
        print()

    print("=" * 60)
    print("\nOptimization Priority Order:\n")

    priorities = [
        ('1. Fix format issues', 'Most important - broken format breaks pipeline'),
        ('2. Add missing instructions', 'Clarify task, often biggest gain'),
        ('3. Add/improve examples', 'Show desired behavior'),
        ('4. Handle edge cases', 'Target specific failure modes'),
        ('5. Refine constraints', 'Fine-tune requirements'),
        ('6. Optimize for cost/speed', 'After quality is acceptable')
    ]

    for priority, reason in priorities:
        print(f"  {priority}")
        print(f"     → {reason}")

optimization_patterns()
```

### Knowing When to Stop

```python
def knowing_when_to_stop():
    """When to stop optimizing prompts."""

    print("\n\nKnowing When to Stop Optimizing:\n")

    print("Diminishing Returns Curve:\n")

    print('''
Accuracy
   100% ┤
        │                          ┌──
    90% ┤                     ┌────┘
        │                ┌────┘
    80% ┤           ┌────┘          ← Plateau
        │      ┌────┘
    70% ┤  ┌───┘                    ← Fast gains
        │┌─┘
    60% ┼┘                          ← Baseline
        └─────────────────────────────> Effort
         0   5   10   15   20   25   Hours

Key insight: First few hours give biggest gains,
then improvements slow dramatically.
''')

    print("=" * 60)
    print("\nStop Optimization When:\n")

    criteria = [
        'Performance meets requirements (e.g., >85% accuracy)',
        'Improvements < 1% for multiple iterations',
        'Cost of optimization exceeds value of improvement',
        'Approaching theoretical limits (inter-annotator agreement)',
        'Time budget exhausted',
        'Better ROI from other tasks',
        'Ready for production, can iterate later'
    ]

    for criterion in criteria:
        print(f"  • {criterion}")

    print("\n" + "=" * 60)
    print("\nDecision Framework:\n")

    framework = '''
Questions to ask:

1. Does it meet requirements?
   • Performance: ✓ 87% (target: 85%)
   • Cost: ✓ $0.01/query (target: <$0.02)
   • Latency: ✓ 1.5s (target: <2s)

2. Is improvement worth the effort?
   • Last 5 hours: +1% improvement
   • Estimated next 5 hours: +0.5% improvement
   • Value of 0.5%: ~$1000/year
   • Cost of 5 hours: ~$500
   → Marginal, but acceptable

3. Are there better opportunities?
   • Other prompts that need work? Yes
   • Other projects? Yes
   → Should move on

DECISION: Stop optimization, deploy current version,
          revisit later if needed.
'''
    print(framework)

knowing_when_to_stop()
```

## Summary

**Key Concepts**:

1. **Prompt optimization** is systematic improvement through iteration and testing
2. **Scientific approach**: Measure baseline → Hypothesize → Change → Test → Compare
3. **Version control** for prompts like code (v1.0, v2.0, etc.)
4. **A/B testing** compares variants on same data with statistical significance
5. **Evaluation sets** must be representative, diverse, and separate from development
6. **Failure analysis** identifies patterns to target improvements
7. **Multiple metrics**: Quality (F1, accuracy), efficiency (cost, latency)
8. **Diminishing returns**: First few iterations give biggest gains
9. **Stop when**: Meets requirements, improvements <1%, time better spent elsewhere

**Optimization Workflow**:

```
1. Baseline: Create initial prompt, measure (e.g., 65% accuracy)
2. Analyze: Collect failures, identify patterns
3. Hypothesize: Form theory about improvement
4. Modify: Make targeted change (add examples)
5. Test: Evaluate on same test set (72% accuracy)
6. Compare: Better? Keep (+7%). Worse? Revert.
7. Repeat: Until satisfied or diminishing returns
```

**Typical Improvement Trajectory**:

| Version | Change           | Accuracy | Gain                        |
| ------- | ---------------- | -------- | --------------------------- |
| v1.0    | Baseline         | 65%      | -                           |
| v1.1    | Add examples     | 72%      | +7%                         |
| v1.2    | Better format    | 76%      | +4%                         |
| v2.0    | More examples    | 81%      | +5%                         |
| v2.1    | Edge cases       | 83%      | +2%                         |
| v2.2    | Refinements      | 84%      | +1%                         |
| v2.3    | More refinements | 84.5%    | +0.5% ← Diminishing returns |

**A/B Testing**:

```python
# Test two prompt variations
results_A = test_prompt(prompt_A, eval_set)  # 74% accuracy
results_B = test_prompt(prompt_B, eval_set)  # 81% accuracy

# B is better: +7% (p < 0.01) ✓
# Decision: Adopt Prompt B
```

**Evaluation Best Practices**:

✓ **Representative dataset**: Covers real-world distribution  
✓ **50-100+ examples**: Large enough for reliable metrics  
✓ **Stratified**: Balanced across categories/difficulty  
✓ **Ground truth**: Correct answers known  
✓ **Stable**: Same test set across iterations  
✓ **Separate dev/test**: Don't optimize on test set  
✓ **Automated evaluation**: Run quickly and consistently

**Failure Analysis Process**:

```
1. Collect all failures from eval run
2. Categorize by type (sarcasm, negation, etc.)
3. Count frequency of each category
4. Identify most common patterns
5. Hypothesize root cause
6. Design targeted fix
7. Test and measure improvement
```

**Metrics Selection**:

- **Classification**: Accuracy (balanced), F1 (imbalanced)
- **Generation**: BLEU, ROUGE, Human eval
- **Extraction**: Exact match, F1
- **Production**: Also track cost, latency
- **Multi-objective**: Quality first, then efficiency

**Prompt Library Organization**:

```
prompts/
├── task_name/
│   ├── v1.0_baseline.txt
│   ├── v2.0_production.txt      ← Current
│   ├── v3.0_experimental.txt
│   ├── eval_set.json            ← Test data
│   └── results.json             ← Performance history
```

**Version Control**:

- Semantic versioning (major.minor)
- Document changes, performance
- Mark production version
- Keep history for rollback
- Link to evaluation results

**When to Stop**:

✓ Meets performance requirements  
✓ Improvements < 1% per iteration  
✓ Approaching theoretical limits  
✓ Time/cost budget exhausted  
✓ Better ROI elsewhere  
✓ Good enough for production

**Common Optimization Patterns**:

1. **Start simple**: Zero-shot → Few-shot only if needed
2. **Incremental examples**: Add one at a time until plateau
3. **Target weakest**: Improve worst-performing category
4. **Remove and add**: Strip to minimum, add back selectively
5. **Copy patterns**: Adapt from similar successful tasks

## Next Steps

- Apply patterns from [Advanced Patterns](advanced-patterns.md) for complex optimizations
- Use [Structured Output](structured-output.md) techniques for reliable format
- Combine with [Chain-of-Thought](cot-prompting.md) for reasoning tasks
- Build evaluation pipelines with [Evaluation](../evaluation/) methods
- Deploy optimized prompts in [Application Patterns](../application_patterns/)
- Study [RLHF](../rlhf_and_alignment/) for model-level optimization
