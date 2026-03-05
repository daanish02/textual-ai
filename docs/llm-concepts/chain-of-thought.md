# Chain-of-Thought Reasoning

## Table of Contents

- [Introduction](#introduction)
- [What is Chain-of-Thought?](#what-is-chain-of-thought)
- [Chain-of-Thought Prompting](#chain-of-thought-prompting)
- [Why Chain-of-Thought Works](#why-chain-of-thought-works)
- [Types of Reasoning Tasks](#types-of-reasoning-tasks)
- [Self-Consistency](#self-consistency)
- [Zero-Shot Chain-of-Thought](#zero-shot-chain-of-thought)
- [Automatic Chain-of-Thought](#automatic-chain-of-thought)
- [When CoT Helps and When It Doesn't](#when-cot-helps-and-when-it-doesnt)
- [Advanced CoT Techniques](#advanced-cot-techniques)
- [CoT Limitations and Failures](#cot-limitations-and-failures)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Chain-of-thought (CoT) reasoning** is the practice of prompting language models to show their reasoning process step-by-step before arriving at a final answer. This simple technique dramatically improves performance on complex reasoning tasks.

```
Without Chain-of-Thought:
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: 11  ← Direct answer (may be wrong)

With Chain-of-Thought:
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. He buys 2 cans of 3 balls each.
   That's 2 × 3 = 6 more balls. So he has 5 + 6 = 11 balls in total.

   The answer is 11.  ← Answer with reasoning (more likely correct)
```

**Key insight**: By generating intermediate reasoning steps, models can solve problems they would fail on with direct answer generation. CoT acts as a form of "thinking out loud" for language models.

This guide explores how CoT works, when to use it, and advanced techniques for complex reasoning.

## What is Chain-of-Thought?

### The Basic Idea

```python
def cot_comparison():
    """Compare direct answer vs chain-of-thought."""

    problem = "A store had 20 apples. They sold 12 and bought 15 more. How many apples do they have?"

    print("Direct Answer Approach:\n")
    print(f"Q: {problem}")
    print("A: 23")
    print("\nIssue: No visibility into reasoning")
    print("Hard to debug if wrong\n")

    print("="*60)

    print("\nChain-of-Thought Approach:\n")
    print(f"Q: {problem}")
    print("A: Let's think step by step:")
    print("   1. The store started with 20 apples")
    print("   2. They sold 12, so now they have: 20 - 12 = 8 apples")
    print("   3. They bought 15 more: 8 + 15 = 23 apples")
    print("   Therefore, they have 23 apples.")

    print("\nBenefits:")
    print("  • Shows reasoning process")
    print("  • Easier to verify correctness")
    print("  • Can identify where errors occur")
    print("  • More likely to be correct")

cot_comparison()
```

### Key Components

```python
def cot_components():
    """Components of effective chain-of-thought reasoning."""

    components = {
        'Problem decomposition': {
            'description': 'Break complex problem into steps',
            'example': 'Multi-step math → separate operations',
            'benefit': 'Reduces cognitive load per step'
        },
        'Explicit intermediate steps': {
            'description': 'Show calculation/reasoning at each step',
            'example': '20 - 12 = 8 (not just "subtract")',
            'benefit': 'Verifiable, debuggable'
        },
        'Natural language reasoning': {
            'description': 'Explain logic in words',
            'example': '"They sold 12, so we subtract..."',
            'benefit': 'Human-readable, intuitive'
        },
        'Sequential flow': {
            'description': 'One step follows from previous',
            'example': 'Step 1 → Step 2 → Step 3 → Answer',
            'benefit': 'Clear logical progression'
        },
        'Final answer extraction': {
            'description': 'Clearly mark the final answer',
            'example': '"Therefore, the answer is 23"',
            'benefit': 'Unambiguous for parsing'
        }
    }

    print("Chain-of-Thought Components:\n")
    for component, info in components.items():
        print(f"{component.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

cot_components()
```

### Historical Context

```
Evolution of CoT Research:

2021: GPT-3 paper shows few-shot learning
      ↓
2022: Wei et al. discover Chain-of-Thought prompting
      → Dramatic improvements on reasoning tasks
      ↓
2022: Kojima et al. show Zero-Shot CoT
      → "Let's think step by step" triggers reasoning
      ↓
2022: Wang et al. propose Self-Consistency
      → Sample multiple chains, take majority vote
      ↓
2023: CoT becomes standard for reasoning tasks
      → Integrated into most LLM applications
```

## Chain-of-Thought Prompting

### Few-Shot CoT

```python
def few_shot_cot_example():
    """Example of few-shot chain-of-thought prompting."""

    prompt = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today.
   After they are done, there will be 21 trees. How many trees did they plant today?
A: We start with 15 trees. Later we have 21 trees.
   The difference must be the number of trees they planted.
   So, they must have planted 21 - 15 = 6 trees.
   The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive,
   how many cars are in the parking lot?
A: There are 3 cars in the parking lot already.
   2 more arrive. Now there are 3 + 2 = 5 cars.
   The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35,
   how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42.
   That means there were originally 32 + 42 = 74 chocolates.
   35 have been eaten. So in total they still have 74 - 35 = 39 chocolates.
   The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops.
   How many lollipops did Jason give to Denny?
A:"""

    print("Few-Shot Chain-of-Thought Prompt:\n")
    print(prompt)

    # Model response
    response = """Jason had 20 lollipops. Then he gave some to Denny.
Now he has 12. The difference is 20 - 12 = 8.
So Jason gave 8 lollipops to Denny.
The answer is 8."""

    print(response)

    print("\n" + "="*60)
    print("\nKey features:")
    print("  • 2-4 examples with full reasoning")
    print("  • Consistent format across examples")
    print("  • Shows thinking process explicitly")
    print("  • Model learns to replicate reasoning style")

few_shot_cot_example()
```

### Prompting Techniques

```python
def cot_prompting_techniques():
    """Different ways to elicit chain-of-thought reasoning."""

    techniques = {
        'Explicit instruction': {
            'prompt': 'Solve this step by step:',
            'when': 'Model needs clear direction',
            'effectiveness': 'High'
        },
        'Numbered steps': {
            'prompt': 'Let\'s solve this in numbered steps:\n1.',
            'when': 'Need structured breakdown',
            'effectiveness': 'High'
        },
        'Implicit reasoning': {
            'prompt': 'Think carefully about this:',
            'when': 'Want natural reasoning',
            'effectiveness': 'Medium'
        },
        'Question prompts': {
            'prompt': 'What do we know? What do we need to find?',
            'when': 'Complex word problems',
            'effectiveness': 'Medium-High'
        },
        'Template-based': {
            'prompt': 'Given: ... Find: ... Steps: ...',
            'when': 'Standardized problem types',
            'effectiveness': 'High'
        }
    }

    print("CoT Prompting Techniques:\n")
    for technique, info in techniques.items():
        print(f"{technique.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

cot_prompting_techniques()
```

## Why Chain-of-Thought Works

### Computational Benefits

```python
def why_cot_works():
    """Theories explaining why CoT improves reasoning."""

    theories = {
        'More computation': {
            'idea': 'Generating steps uses more compute/tokens',
            'mechanism': 'More forward passes = more thinking',
            'evidence': 'Performance correlates with reasoning length'
        },
        'Decomposition': {
            'idea': 'Breaking problems into smaller sub-problems',
            'mechanism': 'Easier to solve small problems correctly',
            'evidence': 'Works best on multi-step problems'
        },
        'Scratchpad': {
            'idea': 'Reasoning steps act as working memory',
            'mechanism': 'Store intermediate results explicitly',
            'evidence': 'Helps with problems exceeding context'
        },
        'Attention focus': {
            'idea': 'Reasoning steps guide attention to relevant info',
            'mechanism': 'Intermediate steps highlight key facts',
            'evidence': 'Attention analysis shows focus on reasoning'
        },
        'Training distribution': {
            'idea': 'Step-by-step reasoning common in training data',
            'mechanism': 'Model learned to reason from educational content',
            'evidence': 'CoT mirrors textbook solutions'
        },
        'Faithfulness': {
            'idea': 'Model follows its own reasoning',
            'mechanism': 'Generated steps influence next tokens',
            'evidence': 'Corrupting reasoning affects answer'
        }
    }

    print("Why Chain-of-Thought Works:\n")
    for theory, info in theories.items():
        print(f"{theory.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

    print("Consensus: Multiple mechanisms contribute")
    print("CoT provides both computational and structural benefits")

why_cot_works()
```

### Empirical Results

```python
def cot_performance_gains():
    """Typical performance improvements from CoT."""

    print("Chain-of-Thought Performance Gains:\n")

    benchmarks = {
        'GSM8K (grade school math)': {
            'baseline': '17.9%',
            'cot': '57.4%',
            'improvement': '+220%'
        },
        'MATH (competition math)': {
            'baseline': '3.9%',
            'cot': '18.1%',
            'improvement': '+364%'
        },
        'CSQA (commonsense QA)': {
            'baseline': '72.5%',
            'cot': '78.2%',
            'improvement': '+8%'
        },
        'StrategyQA (multi-hop)': {
            'baseline': '54.3%',
            'cot': '68.1%',
            'improvement': '+25%'
        },
        'AQuA (algebraic reasoning)': {
            'baseline': '33.7%',
            'cot': '45.9%',
            'improvement': '+36%'
        }
    }

    print(f"{'Benchmark':<30} {'Baseline':<12} {'CoT':<12} {'Improvement'}")
    print("=" * 70)

    for benchmark, results in benchmarks.items():
        print(f"{benchmark:<30} {results['baseline']:<12} {results['cot']:<12} {results['improvement']}")

    print("\nKey findings:")
    print("  • Largest gains on mathematical reasoning")
    print("  • Moderate gains on commonsense reasoning")
    print("  • Benefits increase with model size")
    print("  • Essential for multi-step problems")

cot_performance_gains()
```

## Types of Reasoning Tasks

### Mathematical Reasoning

```python
def math_reasoning_cot():
    """CoT for mathematical reasoning."""

    print("Mathematical Reasoning with CoT:\n")

    problem = """
A restaurant has 23 tables. Each table seats 4 people.
If there are 78 customers, how many tables will be empty?
"""

    print(f"Problem: {problem}")

    cot_solution = """
Let's solve this step by step:

Step 1: Calculate total seating capacity
   - 23 tables × 4 people per table = 92 people

Step 2: Calculate how many tables are needed
   - 78 customers ÷ 4 people per table = 19.5 tables
   - Since we can't have half a table, we need 20 tables

Step 3: Calculate empty tables
   - Total tables: 23
   - Used tables: 20
   - Empty tables: 23 - 20 = 3

Therefore, 3 tables will be empty.
"""

    print(f"CoT Solution:{cot_solution}")

    print("Why CoT helps:")
    print("  • Breaks complex problem into sub-problems")
    print("  • Shows intermediate calculations")
    print("  • Handles rounding/ceiling operations explicitly")
    print("  • Easy to verify each step")

math_reasoning_cot()
```

### Logical Reasoning

```python
def logical_reasoning_cot():
    """CoT for logical reasoning."""

    print("\nLogical Reasoning with CoT:\n")

    problem = """
All roses are flowers.
Some flowers are red.
Are all roses red?
"""

    print(f"Problem: {problem}")

    cot_solution = """
Let's think through this logically:

Given facts:
1. All roses are flowers (roses ⊆ flowers)
2. Some flowers are red (∃ flowers that are red)

Analysis:
- From fact 1: Every rose is a flower
- From fact 2: At least some flowers are red
- But: "Some flowers are red" doesn't mean "all flowers are red"
- Therefore: We cannot conclude that all roses are red

In fact, we can't even conclude that ANY roses are red, because the
"some flowers" that are red might not include any roses at all.

Answer: No, we cannot conclude that all roses are red.
"""

    print(f"CoT Solution:{cot_solution}")

    print("\nBenefits:")
    print("  • Identifies given facts clearly")
    print("  • Analyzes logical relationships")
    print("  • Avoids common fallacies")
    print("  • Explains reasoning explicitly")

logical_reasoning_cot()
```

### Commonsense Reasoning

```python
def commonsense_reasoning_cot():
    """CoT for commonsense reasoning."""

    print("\nCommonsense Reasoning with CoT:\n")

    problem = """
Would a penguin make a good pet for someone living in the Sahara Desert?
"""

    print(f"Problem: {problem}")

    cot_solution = """
Let's think about this:

Facts about penguins:
- Penguins are adapted to cold climates (Antarctica, cold oceans)
- They need cold temperatures to survive
- They require water for swimming and hunting
- They eat fish and need specific care

Facts about Sahara Desert:
- Extremely hot climate (can exceed 50°C/122°F)
- Very little water
- Harsh, dry environment

Analysis:
- Temperature mismatch: Penguins need cold, Sahara is extremely hot
- Water scarcity: Penguins need water, Sahara has little
- Stress and health: Penguin would suffer in this environment
- Ethical concerns: Cruel to subject penguin to incompatible climate

Answer: No, a penguin would not make a good pet in the Sahara Desert.
The climate incompatibility would cause suffering and likely death.
"""

    print(f"CoT Solution:{cot_solution}")

    print("\nWhy CoT helps:")
    print("  • Brings relevant knowledge to attention")
    print("  • Considers multiple factors")
    print("  • Makes implicit reasoning explicit")
    print("  • Accounts for real-world constraints")

commonsense_reasoning_cot()
```

## Self-Consistency

### The Method

```python
def self_consistency_method():
    """Self-consistency: Sample multiple reasoning paths."""

    print("Self-Consistency with CoT:\n")

    problem = "A car travels 60 miles in 1.5 hours. What is its average speed?"

    print(f"Problem: {problem}\n")
    print("Step 1: Generate multiple reasoning paths\n")

    paths = [
        """Path 1: Speed = Distance / Time
        Speed = 60 miles / 1.5 hours
        Speed = 40 mph
        Answer: 40""",

        """Path 2: In 1.5 hours, the car goes 60 miles.
        In 1 hour, it would go 60/1.5 miles
        60 ÷ 1.5 = 40 miles
        Answer: 40""",

        """Path 3: 1.5 hours = 1 hour 30 minutes
        Distance = 60 miles
        60 miles / 1.5 hours = 60/1.5 = 40 mph
        Answer: 40""",

        """Path 4: Let's set up the equation:
        distance = speed × time
        60 = speed × 1.5
        speed = 60/1.5 = 40
        Answer: 40""",

        """Path 5: If speed is s, then:
        s × 1.5 = 60
        s = 60/1.5
        Calculating: 60 ÷ 1.5 = 40
        Answer: 40"""
    ]

    for path in paths:
        print(path)
        print()

    print("Step 2: Extract answers from each path")
    answers = [40, 40, 40, 40, 40]
    print(f"  Answers: {answers}")

    print("\nStep 3: Take majority vote")
    print(f"  Majority answer: 40 (unanimous)")

    print("\nStep 4: Return final answer")
    print(f"  Final answer: 40 mph")

    print("\n" + "="*60)
    print("\nBenefits:")
    print("  • Increases reliability (catches errors)")
    print("  • Typical improvement: +5-20% accuracy")
    print("  • Works even with imperfect reasoning paths")
    print("  • Provides confidence measure (vote margin)")

    print("\nCost:")
    print("  • N times more expensive (N=5-40 typical)")
    print("  • Only practical for important queries")

self_consistency_method()
```

### When Paths Disagree

```python
def self_consistency_disagreement():
    """What happens when reasoning paths disagree."""

    print("\n\nHandling Disagreement in Self-Consistency:\n")

    problem = "A rectangle has area 24 and perimeter 20. What is its length?"

    print(f"Problem: {problem}\n")

    # Simulated paths (some wrong)
    paths_answers = [
        ("Path 1", 6, True),
        ("Path 2", 8, False),  # Made an error
        ("Path 3", 6, True),
        ("Path 4", 6, True),
        ("Path 5", 4, False),  # Different error
    ]

    print("Reasoning paths and answers:\n")
    for path, answer, correct in paths_answers:
        status = "✓" if correct else "✗"
        print(f"  {path}: {answer} {status}")

    print("\nVote distribution:")
    print("  Answer 6: 3 votes (60%)")
    print("  Answer 8: 1 vote (20%)")
    print("  Answer 4: 1 vote (20%)")

    print("\nMajority answer: 6")
    print("\nWhy this works:")
    print("  • Different paths make different errors")
    print("  • Correct reasoning converges on right answer")
    print("  • Majority vote filters out mistakes")
    print("  • Higher confidence when vote is unanimous")

self_consistency_disagreement()
```

## Zero-Shot Chain-of-Thought

### "Let's Think Step by Step"

```python
def zero_shot_cot():
    """Zero-shot CoT: No examples needed."""

    print("Zero-Shot Chain-of-Thought:\n")

    print("The magic phrase: 'Let's think step by step'\n")

    # Without zero-shot CoT
    print("Standard zero-shot (worse):")
    print("Q: If John has 5 apples and gives 2 to Mary, then buys 3 more, how many does he have?")
    print("A: 6")
    print("(Often wrong for complex problems)\n")

    print("="*60 + "\n")

    # With zero-shot CoT
    print("Zero-shot CoT (better):")
    print("Q: If John has 5 apples and gives 2 to Mary, then buys 3 more, how many does he have?")
    print("Let's think step by step.")
    print("A: Okay, let's break this down:")
    print("   1. John starts with 5 apples")
    print("   2. He gives 2 to Mary, so he has: 5 - 2 = 3 apples")
    print("   3. Then he buys 3 more: 3 + 3 = 6 apples")
    print("   Therefore, John has 6 apples.")

    print("\n" + "="*60)
    print("\nKey discovery (Kojima et al., 2022):")
    print("  • Just adding 'Let's think step by step' triggers reasoning")
    print("  • No examples needed!")
    print("  • Works across many tasks")
    print("  • Simpler than few-shot CoT")

    print("\nVariations that work:")
    print("  • 'Let's solve this step by step.'")
    print("  • 'Let's break this down.'")
    print("  • 'Let's think about this carefully.'")
    print("  • 'First, let's identify what we know.'")

zero_shot_cot()
```

### Two-Stage Process

```python
def zero_shot_cot_twostage():
    """Zero-shot CoT uses two API calls."""

    print("\n\nZero-Shot CoT: Two-Stage Process\n")

    print("Stage 1: Generate reasoning")
    print("-" * 40)
    prompt1 = """
Q: A store has 15 shirts. They sell 8 and receive a shipment of 12 more.
   How many shirts do they have?

Let's think step by step.
"""
    print(f"Prompt:{prompt1}")

    reasoning = """
A: Let's break this down:
1. The store starts with 15 shirts
2. They sell 8, so they have: 15 - 8 = 7 shirts
3. They receive 12 more: 7 + 12 = 19 shirts
Therefore, the store has 19 shirts.
"""
    print(f"Model output:{reasoning}")

    print("\nStage 2: Extract answer")
    print("-" * 40)
    prompt2 = f"""
{reasoning}

Therefore, the answer (arabic numerals) is
"""
    print(f"Prompt:{prompt2}")
    print("Model output: 19")

    print("\n" + "="*60)
    print("\nWhy two stages:")
    print("  • Stage 1: Generate full reasoning")
    print("  • Stage 2: Extract clean answer")
    print("  • Separates reasoning from answer extraction")
    print("  • More reliable answer parsing")

zero_shot_cot_twostage()
```

## Automatic Chain-of-Thought

### Auto-CoT

```python
def auto_cot_method():
    """Automatically generate CoT demonstrations."""

    print("Automatic Chain-of-Thought (Auto-CoT):\n")

    print("Problem: Creating good CoT examples is labor-intensive\n")

    print("Auto-CoT algorithm:")
    print("  1. Cluster questions by similarity")
    print("  2. Select representative question from each cluster")
    print("  3. Generate reasoning with zero-shot CoT")
    print("  4. Use generated examples as few-shot demonstrations")

    print("\nExample process:\n")

    print("Step 1: Cluster questions")
    print("  Cluster A: Basic arithmetic")
    print("    - 'What is 5 + 3?'")
    print("    - 'Calculate 12 - 7'")
    print("  Cluster B: Word problems")
    print("    - 'John has 5 apples...'")
    print("    - 'A store has 20 items...'")

    print("\nStep 2: Select representative")
    print("  From Cluster A: 'What is 5 + 3?'")
    print("  From Cluster B: 'John has 5 apples...'")

    print("\nStep 3: Generate reasoning (zero-shot CoT)")
    print("  Q: What is 5 + 3?")
    print("  A: We need to add 5 and 3.")
    print("     5 + 3 = 8")
    print("     The answer is 8.")

    print("\nStep 4: Use as few-shot example")
    print("  Now use this as demonstration for similar questions")

    print("\n" + "="*60)
    print("\nBenefits:")
    print("  • Reduces manual effort")
    print("  • Diverse demonstrations automatically")
    print("  • Scales to many tasks")

    print("\nLimitations:")
    print("  • Generated reasoning may contain errors")
    print("  • Need to verify examples")
    print("  • May not match human reasoning style")

auto_cot_method()
```

## When CoT Helps and When It Doesn't

### Tasks Where CoT Helps

```python
def cot_helps():
    """Tasks that benefit from chain-of-thought."""

    tasks = {
        'Multi-step math': {
            'example': 'Grade school word problems, algebra',
            'improvement': '+100-300%',
            'reason': 'Many intermediate calculations'
        },
        'Logical reasoning': {
            'example': 'Deductive inference, puzzles',
            'improvement': '+20-50%',
            'reason': 'Explicit logical steps help'
        },
        'Commonsense reasoning': {
            'example': 'Physical reasoning, social situations',
            'improvement': '+10-30%',
            'reason': 'Bring implicit knowledge to attention'
        },
        'Strategic reasoning': {
            'example': 'Planning, game playing',
            'improvement': '+30-100%',
            'reason': 'Consider multiple options'
        },
        'Symbolic manipulation': {
            'example': 'Algebra, code execution',
            'improvement': '+50-150%',
            'reason': 'Track state through transformations'
        }
    }

    print("Tasks Where CoT Helps:\n")
    for task, info in tasks.items():
        print(f"{task.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

cot_helps()
```

### Tasks Where CoT Doesn't Help

```python
def cot_not_helpful():
    """Tasks where CoT provides little benefit."""

    tasks = {
        'Simple factual recall': {
            'example': 'What is the capital of France?',
            'improvement': '~0%',
            'reason': 'Single-step retrieval, no reasoning needed'
        },
        'Pattern matching': {
            'example': 'Sentiment classification',
            'improvement': '+0-5%',
            'reason': 'Direct pattern recognition sufficient'
        },
        'Subjective tasks': {
            'example': 'Creative writing, poetry',
            'improvement': 'Variable',
            'reason': 'No "correct" reasoning path'
        },
        'Overlearned tasks': {
            'example': 'Common idioms, proverbs',
            'improvement': '~0%',
            'reason': 'Direct retrieval faster than reasoning'
        },
        'Perception tasks': {
            'example': 'Image classification (for LLMs)',
            'improvement': 'N/A',
            'reason': 'LLMs lack visual perception'
        }
    }

    print("Tasks Where CoT Doesn't Help:\n")
    for task, info in tasks.items():
        print(f"{task.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

    print("Rule of thumb:")
    print("  If a human would need to 'think through' the problem,")
    print("  CoT likely helps. If answer is immediate, CoT adds little.")

cot_not_helpful()
```

### When CoT Can Hurt

```python
def when_cot_hurts():
    """Cases where CoT can reduce performance."""

    print("\nWhen Chain-of-Thought Can Hurt:\n")

    cases = {
        'Overthinking simple problems': {
            'issue': 'CoT adds complexity where none needed',
            'example': '2+2 with elaborate reasoning',
            'impact': '-5-10% (adds noise)'
        },
        'Anchoring on wrong path': {
            'issue': 'First reasoning step is wrong, rest follows',
            'example': 'Misread problem, reason from wrong premise',
            'impact': 'Can amplify errors'
        },
        'Hallucinated reasoning': {
            'issue': 'Generate plausible but incorrect reasoning',
            'example': 'Make up facts that sound right',
            'impact': 'Confident wrong answers'
        },
        'Context length limits': {
            'issue': 'Long reasoning exceeds context window',
            'example': 'Complex problem needs 1000+ tokens',
            'impact': 'Truncation, incomplete reasoning'
        },
        'Cost and latency': {
            'issue': 'More tokens = higher cost and slower',
            'example': 'Simple query becomes 500 tokens',
            'impact': '3-10x cost increase'
        }
    }

    for case, info in cases.items():
        print(f"{case.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

when_cot_hurts()
```

## Advanced CoT Techniques

### Least-to-Most Prompting

```python
def least_to_most_prompting():
    """Least-to-most: Break into subproblems and solve in order."""

    print("Least-to-Most Prompting:\n")

    problem = "If x + 5 = 12 and y = 2x, what is y?"

    print(f"Problem: {problem}\n")

    print("Stage 1: Problem decomposition")
    decomposition = """
To solve this, we need to:
1. First, solve for x from: x + 5 = 12
2. Then, use that x value to find y from: y = 2x
"""
    print(decomposition)

    print("\nStage 2: Solve subproblems in order")
    solution = """
Subproblem 1: Solve x + 5 = 12
  x + 5 = 12
  x = 12 - 5
  x = 7

Subproblem 2: Find y when y = 2x and x = 7
  y = 2x
  y = 2(7)
  y = 14

Final answer: y = 14
"""
    print(solution)

    print("\n" + "="*60)
    print("\nKey idea:")
    print("  • Explicit decomposition before solving")
    print("  • Solve simpler problems first")
    print("  • Use solutions as building blocks")

    print("\nWhen to use:")
    print("  • Problems with clear sub-structure")
    print("  • Compositional reasoning")
    print("  • When standard CoT gets confused")

least_to_most_prompting()
```

### Program-Aided Language Models (PAL)

```python
def pal_method():
    """Program-Aided Language Models: Generate code, execute it."""

    print("\n\nProgram-Aided Language Models (PAL):\n")

    problem = "A store has 23 apples. They sell 8, buy 15, then sell 7. How many apples remain?"

    print(f"Problem: {problem}\n")

    print("PAL approach: Generate Python code\n")

    code = """
# A store has 23 apples
apples = 23

# They sell 8
apples = apples - 8

# Buy 15
apples = apples + 15

# Sell 7
apples = apples - 7

# How many remain?
answer = apples
print(answer)
"""

    print(code)

    print("Execute code:")
    print("  Output: 23")

    print("\n" + "="*60)
    print("\nBenefits:")
    print("  • Perfect arithmetic (no calculation errors)")
    print("  • Can handle complex calculations")
    print("  • Verifiable (can see the code)")
    print("  • Better than text-only CoT for math")

    print("\nLimitations:")
    print("  • Requires code execution environment")
    print("  • Code generation may have bugs")
    print("  • Limited to problems expressible in code")

    print("\nPerformance:")
    print("  • GSM8K: 72.2% (PAL) vs 57.4% (CoT)")
    print("  • MATH: 25.6% (PAL) vs 18.1% (CoT)")

pal_method()
```

### Tree of Thoughts

```python
def tree_of_thoughts():
    """Tree of Thoughts: Explore multiple reasoning paths."""

    print("\n\nTree of Thoughts:\n")

    print("Concept: Search through reasoning possibilities")
    print("like a tree, backtrack when stuck\n")

    print("Example: Creative problem solving")
    print("'Use 4 numbers (4, 9, 10, 13) and operations (+, -, ×, ÷)")
    print("to get 24'\n")

    print("Tree of reasoning paths:\n")

    tree = """
                      [Start: 4, 9, 10, 13]
                             |
         ┌───────────────────┼───────────────────┐
         |                   |                   |
    [13-9=4]            [10-4=6]           [9+10=19]
    4,4,10              6,9,13             4,13,19
         |                   |                   |
    [4×4=16]            [13-9=4]           [Dead end]
    16,10               4,6                    ✗
         |                   |
    [10+16=26]          [4×6=24]
    Dead end ✗          Success! ✓

Found solution: ((13-9)×4)×6 = 24... (continuing search)
"""

    print(tree)

    print("\nKey ideas:")
    print("  • Generate multiple next steps")
    print("  • Evaluate each step")
    print("  • Backtrack from dead ends")
    print("  • More thorough than linear CoT")

    print("\nCost:")
    print("  • Much more expensive (explore many paths)")
    print("  • Only for hard problems worth the cost")

tree_of_thoughts()
```

## CoT Limitations and Failures

### Reasoning Errors

```python
def cot_failure_modes():
    """Common errors in chain-of-thought reasoning."""

    failures = {
        'Arithmetic errors': {
            'description': 'Mistakes in basic calculations',
            'example': '17 + 25 = 42 (wrong) vs 42 (correct)',
            'mitigation': 'Use PAL or external calculator'
        },
        'Logical fallacies': {
            'description': 'Invalid logical inferences',
            'example': 'All A are B, some B are C, therefore all A are C (invalid)',
            'mitigation': 'Formal logic frameworks, verification'
        },
        'Hallucinated steps': {
            'description': 'Making up facts or steps',
            'example': 'Inventing formula that doesn\'t exist',
            'mitigation': 'Retrieval augmentation, grounding'
        },
        'Circular reasoning': {
            'description': 'Reasoning assumes the conclusion',
            'example': 'X is true because X implies X',
            'mitigation': 'Better prompting, examples'
        },
        'Incomplete reasoning': {
            'description': 'Missing steps in logic',
            'example': 'Jumping to conclusion without justification',
            'mitigation': 'More explicit step instructions'
        },
        'Inconsistent reasoning': {
            'description': 'Contradictory steps',
            'example': 'Assuming X then assuming not-X',
            'mitigation': 'Self-consistency, verification'
        }
    }

    print("Chain-of-Thought Failure Modes:\n")
    for failure, info in failures.items():
        print(f"{failure.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

cot_failure_modes()
```

### Faithfulness

```python
def cot_faithfulness():
    """Is the reasoning faithful to the model's decision process?"""

    print("\nFaithfulness of CoT Reasoning:\n")

    print("Question: Does the model actually follow its stated reasoning?")
    print("Or is reasoning a post-hoc rationalization?\n")

    findings = {
        'Evidence for faithfulness': [
            'Corrupting reasoning changes answers',
            'Intervening on reasoning steers outputs',
            'Attention patterns align with reasoning steps'
        ],
        'Evidence against faithfulness': [
            'Can generate correct answer with wrong reasoning',
            'Sometimes contradicts its own reasoning',
            'May reach answer before finishing reasoning'
        ]
    }

    for category, points in findings.items():
        print(f"{category}:")
        for point in points:
            print(f"  • {point}")
        print()

    print("Current consensus:")
    print("  • Reasoning is partially faithful")
    print("  • Model does use its reasoning")
    print("  • But not always fully faithful")
    print("  • Can have 'shortcuts' that bypass reasoning")

    print("\nImplications:")
    print("  • Trust but verify CoT reasoning")
    print("  • Reasoning is useful but not guaranteed correct")
    print("  • Use self-consistency for higher reliability")

cot_faithfulness()
```

## Summary

**Key Concepts**:

1. **Chain-of-thought (CoT)** prompts models to generate explicit reasoning steps before answering
2. **Few-shot CoT** uses examples with reasoning; **zero-shot CoT** uses "Let's think step by step"
3. **CoT dramatically improves** performance on multi-step reasoning tasks (+100-300% on math)
4. **Self-consistency** samples multiple reasoning paths and takes majority vote for reliability
5. **CoT works** by decomposing problems, providing working memory, and allocating more computation
6. **Advanced techniques** include least-to-most prompting, PAL (code generation), and tree of thoughts
7. **CoT has limitations**: arithmetic errors, hallucination, cost, and faithfulness concerns

**CoT Variants**:

```
Standard CoT: Show reasoning in examples
     ↓
Zero-Shot CoT: "Let's think step by step"
     ↓
Self-Consistency: Multiple samples + majority vote
     ↓
Least-to-Most: Explicit decomposition first
     ↓
PAL: Generate code instead of text
     ↓
Tree of Thoughts: Search over reasoning paths
```

**Performance Gains**:

| Task Type      | Improvement | Example             |
| -------------- | ----------- | ------------------- |
| Math reasoning | +100-300%   | GSM8K: 17% → 57%    |
| Logic puzzles  | +20-50%     | Strategic reasoning |
| Commonsense    | +10-30%     | Physical reasoning  |
| Factual recall | ~0%         | Simple lookup       |

**When to Use CoT**:

- ✅ Multi-step problems
- ✅ Mathematical reasoning
- ✅ Logical inference
- ✅ Complex problem solving
- ✅ When interpretability matters
- ✅ Large models (60B+ for best results)

**When NOT to Use CoT**:

- ❌ Simple factual questions
- ❌ Single-step problems
- ❌ Cost/latency sensitive
- ❌ Pattern matching tasks
- ❌ Small models (<7B parameters)

**Best Practices**:

1. Start with zero-shot CoT for simplicity
2. Use few-shot CoT for better performance
3. Apply self-consistency for critical tasks
4. Consider PAL for math-heavy problems
5. Verify reasoning, don't trust blindly
6. Balance cost vs. accuracy needs

**Key Limitations**:

- Arithmetic errors common (use PAL/tools)
- Can hallucinate plausible reasoning
- Not always faithful to actual process
- Expensive (3-10x tokens)
- May overthink simple problems

## Next Steps

- Study [Instruction Tuning](instruction-tuning.md) to improve baseline reasoning
- Learn [Prompt Engineering](../prompt_engineering/chain-of-thought-prompting.md) for effective CoT design
- Explore [Tool Use](../tool-use/calculator-tools.md) to eliminate arithmetic errors
- Understand [LLM Capabilities](capabilities-limitations.md) for realistic expectations
- Study [Self-Consistency](../prompt_engineering/self-consistency.md) techniques in depth
- Learn [Advanced Reasoning](../agentic-ai-lab/planning-and-reasoning/) patterns
