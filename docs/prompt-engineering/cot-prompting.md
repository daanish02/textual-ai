# Chain-of-Thought Prompting

## Table of Contents

- [Introduction](#introduction)
- [What is Chain-of-Thought Prompting?](#what-is-chain-of-thought-prompting)
- [When to Use CoT](#when-to-use-cot)
- [Zero-Shot CoT](#zero-shot-cot)
- [Few-Shot CoT](#few-shot-cot)
- [Self-Consistency](#self-consistency)
- [CoT for Different Task Types](#cot-for-different-task-types)
- [Prompt Patterns for CoT](#prompt-patterns-for-cot)
- [CoT Best Practices](#cot-best-practices)
- [Common CoT Pitfalls](#common-cot-pitfalls)
- [Advanced CoT Techniques](#advanced-cot-techniques)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Chain-of-thought (CoT) prompting** is a technique that prompts language models to generate intermediate reasoning steps before producing a final answer. This dramatically improves performance on tasks requiring multi-step reasoning.

```
Without CoT:
Q: Roger has 5 tennis balls. He buys 2 cans with 3 balls each. How many total?
A: 11

With CoT:
Q: Roger has 5 tennis balls. He buys 2 cans with 3 balls each. How many total?
A: Let's think step by step.
   - Roger starts with 5 balls
   - He buys 2 cans × 3 balls = 6 balls
   - Total: 5 + 6 = 11 balls
   Answer: 11
```

**Key insight**: Explicit reasoning steps act as "working memory" and computational steps, allowing models to solve problems they would fail on with direct answer generation.

```
Performance Impact:

Direct answer:  ████░░░░░░ 35%  GSM8K (grade school math)
With CoT:       ████████░░ 75%  Same task, much better!
                      ↑
              +40% absolute improvement
```

This guide teaches you how to effectively use CoT prompting for complex tasks.

## What is Chain-of-Thought Prompting?

### The Core Technique

```python
def cot_explanation():
    """Understanding chain-of-thought prompting."""

    print("Chain-of-Thought Prompting Explained:\n")

    print("Traditional Prompting:")
    print("  Input → [Model] → Direct Answer")
    print()

    print("Chain-of-Thought Prompting:")
    print("  Input → [Model] → Reasoning Steps → Final Answer")
    print("                        ↑")
    print("                  'Thinking out loud'")
    print()

    print("=" * 60)
    print("\nWhat Makes It Work:\n")

    mechanisms = {
        'Decomposition': 'Break complex problem into simpler sub-problems',
        'Scratchpad': 'Intermediate steps serve as working memory',
        'More computation': 'Generating reasoning uses more forward passes',
        'Error correction': 'Can catch and fix errors in reasoning',
        'Attention guidance': 'Reasoning focuses attention on relevant info'
    }

    for mechanism, explanation in mechanisms.items():
        print(f"  • {mechanism}: {explanation}")

    print("\n" + "=" * 60)
    print("\nExample Comparison:\n")

    problem = "A store has 23 items. They sell 8 and receive 12 more. How many now?"

    print(f"Problem: {problem}\n")

    print("Without CoT:")
    print("  Answer: 27")
    print("  (May be wrong, no visibility into reasoning)")
    print()

    print("With CoT:")
    print("  Let's solve step by step:")
    print("  1. Store starts with 23 items")
    print("  2. They sell 8: 23 - 8 = 15 items")
    print("  3. They receive 12: 15 + 12 = 27 items")
    print("  Answer: 27")
    print("  (Correct, reasoning is verifiable)")

cot_explanation()
```

### Two Approaches to CoT

```python
def cot_approaches():
    """Two main approaches to CoT prompting."""

    print("\n\nTwo Approaches to Chain-of-Thought:\n")

    print("1. FEW-SHOT CoT")
    print("   Provide examples with reasoning")
    print()

    few_shot_example = """
Q: John has 3 apples. He gets 2 more. How many total?
A: John starts with 3 apples. He gets 2 more.
   So 3 + 2 = 5 apples total.
   Answer: 5

Q: Sarah has 10 cookies. She eats 3. How many left?
A: Sarah starts with 10 cookies. She eats 3.
   So 10 - 3 = 7 cookies left.
   Answer: 7

Q: A store has 15 books. They sell 7. How many remain?
A:"""

    print(few_shot_example)
    print("   Model follows the reasoning pattern")
    print()

    print("=" * 60)
    print("\n2. ZERO-SHOT CoT")
    print('   Add "Let\'s think step by step" to prompt')
    print()

    zero_shot_example = """
Q: A store has 15 books. They sell 7. How many remain?

Let's think step by step.
A:"""

    print(zero_shot_example)
    print("   Model generates reasoning automatically")
    print()

    print("=" * 60)
    print("\nComparison:\n")

    comparison = {
        'Setup': {
            'few_shot': 'Need to create examples with reasoning',
            'zero_shot': 'Just add magic phrase'
        },
        'Context usage': {
            'few_shot': 'High (examples take space)',
            'zero_shot': 'Low (just a short phrase)'
        },
        'Performance': {
            'few_shot': 'Better (75-90% on math)',
            'zero_shot': 'Good (60-75% on math)'
        },
        'Flexibility': {
            'few_shot': 'Can show specific reasoning style',
            'zero_shot': 'Model chooses reasoning approach'
        },
        'Best for': {
            'few_shot': 'When you know desired reasoning format',
            'zero_shot': 'Quick wins, less setup'
        }
    }

    print(f"{'Aspect':<20} {'Few-Shot CoT':<35} {'Zero-Shot CoT'}")
    print("=" * 85)

    for aspect, approaches in comparison.items():
        print(f"{aspect:<20} {approaches['few_shot']:<35} {approaches['zero_shot']}")

cot_approaches()
```

## When to Use CoT

### Tasks That Benefit

```python
def when_to_use_cot():
    """Tasks that benefit from chain-of-thought."""

    print("When to Use Chain-of-Thought:\n")

    tasks = {
        'Arithmetic': {
            'examples': ['Word problems', 'Multi-step calculations'],
            'improvement': '+100-300%',
            'why': 'Need intermediate calculations'
        },
        'Commonsense reasoning': {
            'examples': ['Physical reasoning', 'Cause-effect'],
            'improvement': '+20-50%',
            'why': 'Need to think through implications'
        },
        'Logical reasoning': {
            'examples': ['Deduction', 'If-then logic'],
            'improvement': '+30-70%',
            'why': 'Need explicit logical steps'
        },
        'Multi-hop QA': {
            'examples': ['Answer requires multiple facts'],
            'improvement': '+25-60%',
            'why': 'Need to connect information'
        },
        'Symbolic manipulation': {
            'examples': ['Algebra', 'Code transformation'],
            'improvement': '+40-80%',
            'why': 'Need to track state changes'
        },
        'Planning': {
            'examples': ['Task sequencing', 'Strategy'],
            'improvement': '+30-60%',
            'why': 'Need to think ahead'
        }
    }

    for task_type, info in tasks.items():
        print(f"{task_type.upper()}:")
        for key, value in info.items():
            if key == 'examples':
                print(f"  {key.capitalize()}: {', '.join(value)}")
            else:
                print(f"  {key.capitalize()}: {value}")
        print()

    print("Rule of Thumb:")
    print("  If a human would need to 'think it through',")
    print("  the model will benefit from CoT prompting.")

when_to_use_cot()
```

### Tasks That Don't Need CoT

```python
def when_not_to_use_cot():
    """Tasks where CoT doesn't help or hurts."""

    print("\n\nWhen NOT to Use Chain-of-Thought:\n")

    tasks = {
        'Simple lookups': {
            'examples': ['What is the capital of France?'],
            'impact': '~0% improvement',
            'why': 'Single-step retrieval, no reasoning needed'
        },
        'Pattern classification': {
            'examples': ['Sentiment analysis'],
            'impact': '+0-5%',
            'why': 'Direct pattern matching sufficient'
        },
        'Short text generation': {
            'examples': ['One-word answers', 'Labels'],
            'impact': 'May hurt (adds verbosity)',
            'why': 'Reasoning overhead not worth it'
        },
        'Well-rehearsed tasks': {
            'examples': ['Common translations'],
            'impact': '~0%',
            'why': 'Model already knows pattern'
        }
    }

    for task_type, info in tasks.items():
        print(f"{task_type.upper()}:")
        for key, value in info.items():
            if key == 'examples':
                print(f"  {key.capitalize()}: {', '.join(value)}")
            else:
                print(f"  {key.capitalize()}: {value}")
        print()

    print("Potential Downsides of CoT:")
    print("  • Increased token usage (3-5x longer)")
    print("  • Higher latency (more generation)")
    print("  • Higher cost (more tokens)")
    print("  • May overthink simple problems")
    print("  • Can hallucinate reasoning")

when_not_to_use_cot()
```

## Zero-Shot CoT

### The Magic Phrase

```python
def zero_shot_cot_magic():
    """The famous 'Let's think step by step' prompt."""

    print("Zero-Shot Chain-of-Thought:\n")

    print("Discovery (Kojima et al., 2022):")
    print('  Just adding "Let\'s think step by step" triggers reasoning!')
    print()

    print("=" * 60)
    print("\nExample:\n")

    print("Without magic phrase:")
    print("─" * 60)
    standard = """
Q: If there are 3 cars in the parking lot and 2 more arrive,
   how many are there total?
A:"""
    print(standard)
    print("Response: '5' (direct answer, may be wrong for complex problems)")
    print()

    print("With magic phrase:")
    print("─" * 60)
    with_magic = """
Q: If there are 3 cars in the parking lot and 2 more arrive,
   how many are there total?

Let's think step by step.
A:"""
    print(with_magic)
    print("Response: 'There are 3 cars initially. 2 more arrive.")
    print("          So we have 3 + 2 = 5 cars total.'")
    print()

    print("=" * 60)
    print("\nVariations That Work:\n")

    variations = [
        "Let's think step by step.",
        "Let's solve this step by step.",
        "Let's break this down.",
        "Let's work through this carefully.",
        "Let's think about this logically.",
        "First, let's consider what we know."
    ]

    for variation in variations:
        print(f'  • "{variation}"')

    print("\nWhy It Works:")
    print("  • Triggers reasoning mode in model")
    print("  • Trained on step-by-step solutions")
    print("  • Explicit prompt to decompose")

zero_shot_cot_magic()
```

### Zero-Shot CoT Template

```python
def zero_shot_cot_template():
    """Template for zero-shot CoT prompting."""

    print("\n\nZero-Shot CoT Template:\n")

    template = """
[Task Description]

[Problem/Question]

Let's think step by step.

[Model generates reasoning and answer]
"""

    print("Basic Template:")
    print("─" * 60)
    print(template)

    print("=" * 60)
    print("\nConcrete Example:\n")

    example = """
Solve this math word problem:

A restaurant has 23 tables. Each table seats 4 people.
If there are 78 customers, how many tables will be empty?

Let's think step by step.
"""

    print(example)

    print("Expected Model Response:")
    print("─" * 60)
    response = """
Let's solve this step by step:

1. First, calculate total seating capacity:
   23 tables × 4 people per table = 92 people

2. We have 78 customers, so we need:
   78 ÷ 4 = 19.5 tables
   Since we can't have half a table, we need 20 tables

3. Calculate empty tables:
   Total tables: 23
   Used tables: 20
   Empty tables: 23 - 20 = 3

Therefore, 3 tables will be empty.
"""
    print(response)

    print("\n" + "=" * 60)
    print("\nTips:")
    print("  • Place phrase after the problem")
    print("  • Can add 'Answer:' at the end for clarity")
    print("  • Works best with models 30B+ parameters")
    print("  • May need temperature ~0.7 for diverse reasoning")

zero_shot_cot_template()
```

## Few-Shot CoT

### Creating CoT Examples

```python
def creating_cot_examples():
    """How to create good CoT examples."""

    print("Creating Chain-of-Thought Examples:\n")

    print("Example Structure:")
    print("  [Problem] → [Reasoning Steps] → [Answer]")
    print()

    print("=" * 60)
    print("\nGood CoT Example:\n")

    good_example = """
Q: Alice has 5 red marbles and 3 blue marbles. Bob has twice as many
   marbles as Alice. How many marbles does Bob have?

A: Let's solve this step by step:
   1. Count Alice's marbles: 5 red + 3 blue = 8 marbles total
   2. Bob has twice as many as Alice: 2 × 8 = 16 marbles

   Therefore, Bob has 16 marbles.
"""

    print(good_example)

    print("Why This is Good:")
    print("  ✓ Clear problem statement")
    print("  ✓ Explicit reasoning steps")
    print("  ✓ Shows intermediate calculations")
    print("  ✓ Clear final answer")
    print()

    print("=" * 60)
    print("\nPoor CoT Example:\n")

    poor_example = """
Q: Alice has 5 red marbles and 3 blue marbles. Bob has twice as many
   marbles as Alice. How many marbles does Bob have?

A: Bob has 16 marbles because Alice has some and Bob has twice that amount.
"""

    print(poor_example)

    print("Why This is Poor:")
    print("  ✗ Vague reasoning ('some')")
    print("  ✗ Missing intermediate steps")
    print("  ✗ Doesn't show calculations")
    print("  ✗ Hard to follow logic")

creating_cot_examples()
```

### Few-Shot CoT Template

```python
def few_shot_cot_template():
    """Template for few-shot CoT prompting."""

    print("\n\nFew-Shot CoT Template:\n")

    template = """
[Task Description]

[Example 1 Problem]
[Example 1 Reasoning]
[Example 1 Answer]

[Example 2 Problem]
[Example 2 Reasoning]
[Example 2 Answer]

[Example 3 Problem]
[Example 3 Reasoning]
[Example 3 Answer]

[New Problem]
"""

    print("Template Structure:")
    print("─" * 60)
    print(template)

    print("=" * 60)
    print("\nComplete Example:\n")

    complete = """
Solve these math word problems by showing your reasoning.

Q: Tom has $20. He spends $5 on lunch. How much does he have left?
A: Tom starts with $20. He spends $5 on lunch.
   $20 - $5 = $15
   Answer: Tom has $15 left.

Q: A box contains 12 apples. Emily eats 3 and John eats 4. How many remain?
A: The box starts with 12 apples.
   Emily eats 3: 12 - 3 = 9 apples left
   John eats 4: 9 - 4 = 5 apples left
   Answer: 5 apples remain.

Q: A train travels 60 miles in 2 hours. What is its average speed?
A: The train travels 60 miles in 2 hours.
   Average speed = distance ÷ time
   Average speed = 60 miles ÷ 2 hours = 30 mph
   Answer: 30 mph

Q: Sarah has 8 toys. She gets 5 more for her birthday and gives away 3.
   How many toys does she have now?
A:"""

    print(complete)

    print("\nExpected Response:")
    print("─" * 60)
    print("Sarah starts with 8 toys.")
    print("She gets 5 more: 8 + 5 = 13 toys")
    print("She gives away 3: 13 - 3 = 10 toys")
    print("Answer: Sarah has 10 toys now.")

few_shot_cot_template()
```

## Self-Consistency

### The Technique

```python
def self_consistency_method():
    """Self-consistency for improved reliability."""

    print("Self-Consistency Method:\n")

    print("Concept: Sample multiple reasoning paths, take majority vote")
    print()

    print("Algorithm:")
    print("  1. Generate N different reasoning paths (e.g., N=5-10)")
    print("  2. Extract final answer from each path")
    print("  3. Take majority vote among answers")
    print("  4. Return most common answer")
    print()

    print("=" * 60)
    print("\nExample:\n")

    problem = "If x + 5 = 12, what is x?"

    print(f"Problem: {problem}\n")

    print("Sample 5 reasoning paths:\n")

    paths = [
        ("Path 1", "x + 5 = 12, so x = 12 - 5 = 7", "7"),
        ("Path 2", "Move 5 to right: x = 12 - 5, x = 7", "7"),
        ("Path 3", "Subtract 5 from both sides: x = 7", "7"),
        ("Path 4", "x equals 12 minus 5 which is 7", "7"),
        ("Path 5", "12 - 5 = 7, so x = 7", "7")
    ]

    for name, reasoning, answer in paths:
        print(f"{name}: {reasoning}")
        print(f"  Answer: {answer}")
        print()

    print("Vote Count:")
    print("  Answer '7': 5 votes (100%)")
    print()

    print("Final Answer: 7 (unanimous)")

    print("\n" + "=" * 60)
    print("\nWhen Paths Disagree:\n")

    print("Problem: What is 127 × 8?\n")

    results = [
        ("Path 1", "1016"),
        ("Path 2", "1016"),
        ("Path 3", "1016"),
        ("Path 4", "1006"),  # Error
        ("Path 5", "1016")
    ]

    for name, answer in results:
        print(f"{name}: {answer}")

    print("\nVote Count:")
    print("  Answer '1016': 4 votes (80%)")
    print("  Answer '1006': 1 vote (20%)")
    print()
    print("Final Answer: 1016 (majority)")
    print()
    print("Benefit: Error in one path doesn't affect final answer")

self_consistency_method()
```

### Implementation

```python
def self_consistency_implementation():
    """Implementing self-consistency."""

    print("\n\nSelf-Consistency Implementation:\n")

    code = '''
def self_consistency(problem, n_paths=5, temperature=0.7):
    """
    Generate multiple reasoning paths and take majority vote.

    Args:
        problem: The problem to solve
        n_paths: Number of reasoning paths to sample
        temperature: Higher temperature for diverse paths

    Returns:
        most_common_answer: Answer with highest vote count
        confidence: Percentage agreeing with final answer
    """
    from collections import Counter

    # Build CoT prompt
    prompt = f"""
    {problem}

    Let's think step by step.
    """

    answers = []

    # Generate n_paths different reasoning paths
    for i in range(n_paths):
        # Call LLM with temperature for diversity
        response = llm_call(prompt, temperature=temperature)

        # Extract final answer from reasoning
        answer = extract_answer(response)
        answers.append(answer)

    # Count votes
    vote_counts = Counter(answers)
    most_common_answer, count = vote_counts.most_common(1)[0]

    # Calculate confidence
    confidence = count / n_paths

    return most_common_answer, confidence

# Usage
problem = "If x + 7 = 15, what is x?"
answer, confidence = self_consistency(problem, n_paths=5)

print(f"Answer: {answer}")
print(f"Confidence: {confidence:.0%}")
# Output: Answer: 8, Confidence: 100%
'''

    print(code)

    print("\n" + "=" * 60)
    print("\nPerformance Impact:\n")

    improvements = [
        ('GSM8K (math)', '57% → 73%', '+16%'),
        ('StrategyQA', '69% → 78%', '+9%'),
        ('AQuA (algebra)', '46% → 58%', '+12%'),
    ]

    print(f"{'Benchmark':<20} {'Standard CoT → Self-Consistency':<30} {'Gain'}")
    print("="*65)

    for benchmark, change, gain in improvements:
        print(f"{benchmark:<20} {change:<30} {gain}")

    print("\nCost-Benefit:")
    print("  • Cost: N times more expensive (N API calls)")
    print("  • Benefit: +5-15% accuracy improvement")
    print("  • Use when: Correctness is critical, cost acceptable")

self_consistency_implementation()
```

## CoT for Different Task Types

### Math Word Problems

```python
def cot_for_math():
    """CoT prompting for mathematical reasoning."""

    print("CoT for Math Word Problems:\n")

    prompt = """
Solve the following math problem step by step.

Problem: A bakery makes 120 croissants every morning. They sell 3/4 of them
by noon and 1/2 of the remaining by evening. How many croissants are left?

Let's solve this step by step:

1. Calculate croissants sold by noon:
   - Sold by noon = 3/4 × 120 = 90 croissants
   - Remaining after noon = 120 - 90 = 30 croissants

2. Calculate croissants sold by evening:
   - Sold by evening = 1/2 × 30 = 15 croissants
   - Remaining after evening = 30 - 15 = 15 croissants

Answer: 15 croissants are left at the end of the day.
"""

    print(prompt)

    print("\n" + "=" * 60)
    print("\nKey Techniques for Math:\n")

    techniques = [
        'Show each calculation explicitly',
        'Label intermediate results',
        'Use proper mathematical notation',
        'Check units (miles, hours, etc.)',
        'State final answer clearly',
        'Consider using code for complex calculations'
    ]

    for technique in techniques:
        print(f"  • {technique}")

cot_for_math()
```

### Logical Reasoning

```python
def cot_for_logic():
    """CoT prompting for logical reasoning tasks."""

    print("\n\nCoT for Logical Reasoning:\n")

    prompt = """
Determine if the conclusion follows logically from the premises.

Premises:
1. All programmers drink coffee
2. Alice drinks coffee

Conclusion: Alice is a programmer

Let's analyze this step by step:

1. Examine Premise 1: "All programmers drink coffee"
   - This means: programmer → drinks coffee
   - This does NOT mean: drinks coffee → programmer

2. Examine Premise 2: "Alice drinks coffee"
   - We know Alice drinks coffee

3. Check if conclusion follows:
   - We cannot conclude Alice is a programmer
   - Many non-programmers also drink coffee
   - This is the logical fallacy: affirming the consequent

Answer: The conclusion does NOT follow logically. Just because all
programmers drink coffee doesn't mean everyone who drinks coffee is
a programmer.
"""

    print(prompt)

    print("\n" + "=" * 60)
    print("\nLogical Reasoning Strategies:\n")

    strategies = [
        'Identify premises and conclusion clearly',
        'Represent logical structure (if A then B)',
        'Check for common fallacies',
        'Consider counterexamples',
        'State reasoning for each step',
        'Be explicit about logical operators (AND, OR, NOT)'
    ]

    for strategy in strategies:
        print(f"  • {strategy}")

cot_for_logic()
```

### Multi-Hop Question Answering

```python
def cot_for_multihop():
    """CoT for questions requiring multiple facts."""

    print("\n\nCoT for Multi-Hop Question Answering:\n")

    prompt = """
Answer the following question using the provided information.

Context:
- The Eiffel Tower is located in Paris
- Paris is the capital of France
- France is a member of the European Union
- The European Union was founded in 1993

Question: In what year was the organization that includes the country
containing the Eiffel Tower founded?

Let's trace through this step by step:

1. Identify where the Eiffel Tower is located:
   - The Eiffel Tower is in Paris

2. Identify what country Paris is in:
   - Paris is the capital of France
   - Therefore, the Eiffel Tower is in France

3. Identify what organization France is a member of:
   - France is a member of the European Union

4. Find when that organization was founded:
   - The European Union was founded in 1993

Answer: 1993 (the year the European Union was founded)
"""

    print(prompt)

    print("\n" + "=" * 60)
    print("\nMulti-Hop QA Techniques:\n")

    techniques = [
        'Break question into sub-questions',
        'Answer each sub-question in order',
        'Connect facts explicitly (therefore, so, this means)',
        'Track entities through multiple hops',
        'State which fact is used at each step',
        'Verify final answer addresses original question'
    ]

    for technique in techniques:
        print(f"  • {technique}")

cot_for_multihop()
```

## Prompt Patterns for CoT

### Structured CoT Format

```python
def structured_cot_patterns():
    """Common patterns for structuring CoT prompts."""

    print("Structured CoT Patterns:\n")

    print("PATTERN 1: Numbered Steps\n")

    pattern1 = """
Problem: [problem description]

Solution:
1. [First step with calculation/reasoning]
2. [Second step building on first]
3. [Third step]
...
n. [Final step with answer]

Answer: [concise final answer]
"""
    print(pattern1)

    print("=" * 60)
    print("\nPATTERN 2: Question-Answer Chain\n")

    pattern2 = """
Problem: [problem description]

Q1: What do we know?
A1: [list given information]

Q2: What do we need to find?
A2: [state the goal]

Q3: What steps are needed?
A3: [outline approach]

Q4: [Execute step 1]?
A4: [result of step 1]

...

Final Answer: [answer]
"""
    print(pattern2)

    print("=" * 60)
    print("\nPATTERN 3: Think-Calculate-Verify\n")

    pattern3 = """
Problem: [problem description]

Think: [Understand what's being asked]
       [Identify relevant information]
       [Plan approach]

Calculate: [Show calculations step by step]
          [Label intermediate results]
          [Use proper notation]

Verify: [Check if answer makes sense]
        [Verify calculations]
        [Confirm answer addresses question]

Answer: [final answer]
"""
    print(pattern3)

    print("=" * 60)
    print("\nPATTERN 4: Socratic Method\n")

    pattern4 = """
Problem: [problem description]

Let's think about this:

What information are we given?
- [fact 1]
- [fact 2]

What is the question really asking?
- [restate in own words]

How can we approach this?
- [describe strategy]

Let's work through it:
- [step 1]
- [step 2]

Does our answer make sense?
- [sanity check]

Answer: [final answer]
"""
    print(pattern4)

    print("\nChoosing a Pattern:")
    print("  • Numbered steps: Clear, linear problems")
    print("  • Question-answer: Teaching, complex reasoning")
    print("  • Think-calculate-verify: Math, calculations")
    print("  • Socratic: Understanding, exploration")

structured_cot_patterns()
```

## CoT Best Practices

### Effective CoT Design

```python
def cot_best_practices():
    """Best practices for CoT prompting."""

    print("Chain-of-Thought Best Practices:\n")

    practices = {
        'Be explicit': {
            'do': 'Show every calculation and logical step',
            'dont': 'Skip steps or assume understanding',
            'example': '"5 + 3 = 8" not "add them"'
        },
        'Label steps': {
            'do': 'Number steps or use clear markers',
            'dont': 'Run steps together without structure',
            'example': '"Step 1:", "Step 2:" or "First,", "Then,"'
        },
        'Check work': {
            'do': 'Include verification or sanity check',
            'dont': 'End abruptly after calculation',
            'example': '"Does 11 make sense? Yes, 5+6=11 ✓"'
        },
        'Show intermediate values': {
            'do': 'Store and label intermediate results',
            'dont': 'Chain calculations without labels',
            'example': '"remaining = 20 - 8 = 12"'
        },
        'Use natural language': {
            'do': 'Explain reasoning in plain English',
            'dont': 'Use overly formal or cryptic notation',
            'example': '"Alice has 5 apples" not "A = 5"'
        },
        'State assumptions': {
            'do': 'Make assumptions explicit',
            'dont': 'Assume reader knows context',
            'example': '"Assuming a 30-day month..."'
        }
    }

    for practice, guidance in practices.items():
        print(f"{practice.upper()}:")
        print(f"  ✓ Do: {guidance['do']}")
        print(f"  ✗ Don't: {guidance['dont']}")
        print(f"  Example: {guidance['example']}")
        print()

cot_best_practices()
```

### Optimizing CoT Quality

```python
def optimizing_cot_quality():
    """How to improve CoT reasoning quality."""

    print("\n\nOptimizing CoT Quality:\n")

    print("1. START SIMPLE")
    print("   Begin with clear, easy examples")
    print("   Gradually increase complexity")
    print()

    print("2. BE CONSISTENT")
    print("   Use same format across all examples")
    print("   Maintain similar level of detail")
    print()

    print("3. SHOW ERRORS")
    print("   Include examples of catching errors")
    print('   "Wait, that can\'t be right because..."')
    print()

    print("4. ADD VERIFICATION")
    print('   End with "Let\'s verify:" or "Checking:"')
    print("   Confirm answer makes sense")
    print()

    print("5. USE TEMPLATES")
    print("   Create reusable CoT templates")
    print("   Adapt template to specific problem")
    print()

    print("6. ITERATE")
    print("   Test CoT prompts on multiple examples")
    print("   Refine based on failure patterns")
    print()

    print("=" * 60)
    print("\nQuality Indicators:\n")

    indicators = {
        'Good CoT': [
            'Every step is clear and justified',
            'Calculations shown with proper notation',
            'Intermediate results labeled',
            'Logical flow from start to finish',
            'Final answer clearly stated',
            'Could be followed by human reader'
        ],
        'Poor CoT': [
            'Skips steps or jumps to conclusion',
            'Vague reasoning ("it\'s obvious")',
            'Inconsistent formatting',
            'Missing calculations',
            'Ambiguous final answer',
            'Hard to follow the logic'
        ]
    }

    for quality, examples in indicators.items():
        print(f"{quality}:")
        for example in examples:
            marker = '✓' if quality == 'Good CoT' else '✗'
            print(f"  {marker} {example}")
        print()

optimizing_cot_quality()
```

## Common CoT Pitfalls

### Pitfalls to Avoid

```python
def cot_pitfalls():
    """Common mistakes in CoT prompting."""

    print("Common CoT Pitfalls:\n")

    pitfalls = {
        'Reasoning without calculation': {
            'problem': 'Describes steps but doesn\'t show math',
            'example': '"We add the numbers" (but doesn\'t show 5+3=8)',
            'fix': 'Always show explicit calculations'
        },
        'Too verbose': {
            'problem': 'Overly long reasoning obscures logic',
            'example': 'Paragraph of explanation for 2+2',
            'fix': 'Be concise, include only necessary steps'
        },
        'Inconsistent detail': {
            'problem': 'Some steps detailed, others skipped',
            'example': 'Shows 5+3=8 but then "do some algebra" ',
            'fix': 'Consistent level of detail throughout'
        },
        'Wrong format': {
            'problem': 'Reasoning and answer format don\'t match examples',
            'example': 'Examples use bullets, new problem uses prose',
            'fix': 'Maintain consistent formatting'
        },
        'No verification': {
            'problem': 'Calculates but doesn\'t check if sensible',
            'example': 'Gets negative people count, doesn\'t question it',
            'fix': 'Add sanity checks and verification steps'
        },
        'Hallucinated reasoning': {
            'problem': 'Model generates plausible but wrong logic',
            'example': 'Makes up a formula that doesn\'t exist',
            'fix': 'Verify reasoning with test cases, use tools for math'
        }
    }

    for pitfall, info in pitfalls.items():
        print(f"{pitfall.upper()}:")
        print(f"  Problem: {info['problem']}")
        print(f"  Example: {info['example']}")
        print(f"  Fix: {info['fix']}")
        print()

cot_pitfalls()
```

## Advanced CoT Techniques

### Least-to-Most Prompting

```python
def least_to_most():
    """Least-to-most prompting technique."""

    print("Least-to-Most Prompting:\n")

    print("Concept: Break problem into subproblems, solve simplest first")
    print()

    print("=" * 60)
    print("\nExample:\n")

    prompt = """
Problem: If x + 5 = 12 and y = 2x, what is y?

Let's break this into subproblems:

Subproblem 1: Solve for x
Question: What is x if x + 5 = 12?
Solution:
  x + 5 = 12
  x = 12 - 5
  x = 7

Subproblem 2: Solve for y using x
Question: What is y if y = 2x and x = 7?
Solution:
  y = 2x
  y = 2 × 7
  y = 14

Final Answer: y = 14
"""

    print(prompt)

    print("\nWhen to Use:")
    print("  • Compositional problems")
    print("  • Multi-step with clear substructure")
    print("  • When intermediate results needed for next steps")

least_to_most()
```

### Program-Aided Language Models (PAL)

````python
def pal_technique():
    """Program-Aided Language Models."""

    print("\n\nProgram-Aided Language Models (PAL):\n")

    print("Concept: Generate code to solve problem, execute it")
    print("Benefits: Perfect arithmetic, complex calculations")
    print()

    prompt = """
Problem: A store has 150 items. They sell 60% on Monday and 40% of the
remaining on Tuesday. How many items are left?

Let's write Python code to solve this:

```python
# Initial items
items = 150

# Monday: sell 60%
sold_monday = items * 0.60
remaining_after_monday = items - sold_monday

# Tuesday: sell 40% of remaining
sold_tuesday = remaining_after_monday * 0.40
remaining_after_tuesday = remaining_after_monday - sold_tuesday

# Answer
answer = remaining_after_tuesday
print(f"Items remaining: {answer}")
````

Execute the code:
Items remaining: 36.0

Answer: 36 items remain.
"""

    print(prompt)

    print("\nPAL Benefits:")
    print("  • No arithmetic errors")
    print("  • Handles complex calculations")
    print("  • Verifiable (can see the code)")
    print("  • Better than text CoT for math")

    print("\nPAL Requirements:")
    print("  • Code execution environment")
    print("  • Model must generate valid code")
    print("  • Additional security considerations")

pal_technique()

```

## Summary

**Key Concepts**:

1. **Chain-of-thought** prompts models to generate reasoning steps before answers
2. **Two approaches**: Few-shot (examples with reasoning) and zero-shot ("Let's think step by step")
3. **Dramatic improvements** on reasoning tasks: +40% on math, +20-30% on logic
4. **Self-consistency** samples multiple paths and votes for higher reliability (+5-15%)
5. **Works best** for multi-step reasoning, math, logic, planning
6. **Less helpful** for simple lookups, pattern matching, single-step tasks
7. **Trade-off**: Higher accuracy but 3-5x more tokens (cost and latency)

**CoT Performance Impact**:

| Task Type | Without CoT | With CoT | Improvement |
|-----------|-------------|----------|-------------|
| Grade school math | 17% | 57% | +236% |
| College math | 4% | 18% | +350% |
| Commonsense QA | 73% | 78% | +7% |
| Multi-hop QA | 54% | 68% | +26% |

**Two Main Approaches**:

```

Few-Shot CoT:
[Example 1 with reasoning]
[Example 2 with reasoning]
[New problem]
→ Model follows reasoning pattern
→ Better performance (75-90%)
→ Requires creating examples

Zero-Shot CoT:
[Problem]
"Let's think step by step."
→ Model generates reasoning
→ Good performance (60-75%)
→ No examples needed

```

**Best Practices**:

✓ **Show calculations explicitly** - don't skip steps
✓ **Label intermediate results** - "remaining = 20 - 8 = 12"
✓ **Use structured format** - numbered steps or clear markers
✓ **Include verification** - sanity check final answer
✓ **Be consistent** - same format across examples
✓ **Natural language** - explain reasoning clearly

**Common Mistakes**:

❌ Skipping calculation steps
❌ Inconsistent formatting
❌ Too verbose or too terse
❌ No verification of answers
❌ Mixing different reasoning styles
❌ Not checking if answer makes sense

**Advanced Techniques**:

- **Self-consistency**: Sample 5-10 paths, majority vote (+5-15% accuracy)
- **Least-to-most**: Break into subproblems, solve sequentially
- **PAL**: Generate Python code instead of text reasoning (best for math)
- **Tree-of-thought**: Explore multiple reasoning branches

**When to Use**:

```

✓ Use CoT for:
• Multi-step math problems
• Logical reasoning
• Multi-hop questions
• Planning and strategy
• Complex decision-making

✗ Skip CoT for:
• Simple lookups
• Pattern classification
• Single-step tasks
• Cost/latency sensitive

```

**Template**:

```

[Problem description]

Let's solve this step by step:

1. [First step with calculation/reasoning]
2. [Second step building on first]
3. [Third step]
   ...

Therefore, [final answer].

```

## Next Steps

- Learn [Structured Output](structured-output.md) for formatting CoT responses
- Study [Prompt Optimization](prompt-optimization.md) for testing CoT variations
- Explore [Advanced Patterns](advanced-patterns.md) for complex reasoning
- Review [LLM Capabilities](../llm-concepts/capabilities-limitations.md) for realistic expectations
- Apply to [Tool Use](../agentic-ai-lab/tool-use/) for combining CoT with external tools
- Study [Self-Consistency](../llm-concepts/chain-of-thought.md) in depth for reliability

```
