# Benchmarks and Leaderboards

## Table of Contents

- [Introduction](#introduction)
- [Purpose of Benchmarks](#purpose-of-benchmarks)
- [Major LLM Benchmarks](#major-llm-benchmarks)
- [Task-Specific Benchmarks](#task-specific-benchmarks)
- [Leaderboard Ecosystems](#leaderboard-ecosystems)
- [Interpreting Benchmark Results](#interpreting-benchmark-results)
- [Limitations and Pitfalls](#limitations-and-pitfalls)
- [Designing Custom Benchmarks](#designing-custom-benchmarks)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Benchmarks provide standardized datasets for comparing models objectively. They enable reproducible evaluation and track progress over time.

```
Why Benchmarks Matter:

Without Benchmarks:          With Benchmarks:
┌────────────────┐          ┌────────────────┐
│ "My model is   │          │ Model A: 85.3% │
│  better!"      │   vs     │ Model B: 87.1% │
│                │          │ (on MMLU)      │
│ Hard to verify │          │ Objective      │
└────────────────┘          └────────────────┘

Benefits:
  • Objective comparison
  • Track progress over time
  • Identify strengths/weaknesses
  • Community standards
  • Reproducible results
```

**Benchmark characteristics**:

- **Standardized**: Same test for everyone
- **Diverse**: Cover multiple capabilities
- **Challenging**: Not easily gamed
- **Representative**: Reflect real-world use
- **Measurable**: Clear metrics

This guide covers major benchmarks, how to use them, and their limitations.

## Purpose of Benchmarks

### What Benchmarks Measure

```python
def benchmark_categories():
    """Categories of capabilities measured by benchmarks."""

    print("Benchmark Categories:\n")
    print("="*70)

    categories = {
        "Knowledge": {
            "description": "Factual knowledge across domains",
            "examples": ["MMLU", "TriviaQA", "Natural Questions"],
            "measures": "Breadth and depth of knowledge"
        },
        "Reasoning": {
            "description": "Logical and mathematical reasoning",
            "examples": ["GSM8K", "MATH", "ARC"],
            "measures": "Step-by-step reasoning ability"
        },
        "Commonsense": {
            "description": "Everyday reasoning and intuition",
            "examples": ["HellaSwag", "PIQA", "WinoGrande"],
            "measures": "Understanding of implicit knowledge"
        },
        "Reading Comprehension": {
            "description": "Understanding and extracting from text",
            "examples": ["SQuAD", "RACE", "QuAC"],
            "measures": "Text understanding and information retrieval"
        },
        "Truthfulness": {
            "description": "Factual accuracy and avoiding falsehoods",
            "examples": ["TruthfulQA", "FactScore"],
            "measures": "Reliability and honesty"
        },
        "Safety": {
            "description": "Avoiding harmful or biased outputs",
            "examples": ["RealToxicityPrompts", "BBQ"],
            "measures": "Toxicity and bias"
        },
        "Long Context": {
            "description": "Handling extended documents",
            "examples": ["SCROLLS", "LongBench"],
            "measures": "Long-range dependency tracking"
        },
        "Coding": {
            "description": "Programming and code generation",
            "examples": ["HumanEval", "MBPP", "CodeContests"],
            "measures": "Code correctness and problem-solving"
        }
    }

    for category, info in categories.items():
        print(f"\n{category}:")
        print(f"  Description: {info['description']}")
        print(f"  Examples: {', '.join(info['examples'])}")
        print(f"  Measures: {info['measures']}")

benchmark_categories()
```

### Evolution of Benchmarks

```python
def benchmark_evolution():
    """How benchmarks have evolved with capabilities."""

    print("\n\nBenchmark Evolution:\n")
    print("="*70)

    timeline = """
2010s - Early NLP:
  • SQuAD (2016): Reading comprehension
  • GLUE (2018): Sentence understanding
  → Focus: Understanding and classification

2019-2020 - Large Models:
  • SuperGLUE (2019): Harder language understanding
  • Natural Questions (2019): Open-domain QA
  • TriviaQA: Factual knowledge
  → Models quickly saturated these benchmarks

2021-2022 - Even Larger Models:
  • MMLU (2020): 57 subjects, diverse knowledge
  • BIG-bench (2022): 200+ diverse tasks
  • HumanEval (2021): Code generation
  → Need more challenging, diverse tasks

2023+ - LLM Era:
  • Chatbot Arena: Human preference via Elo
  • AlpacaEval: Instruction following
  • MT-Bench: Multi-turn conversation
  • HELM: Holistic evaluation across scenarios
  → Focus on practical capabilities and safety

Trend: Benchmarks get harder as models improve
Problem: Saturation and overfitting
Solution: Continuously create new, harder benchmarks
"""

    print(timeline)

benchmark_evolution()
```

## Major LLM Benchmarks

### MMLU (Massive Multitask Language Understanding)

```python
def mmlu_benchmark():
    """MMLU benchmark overview."""

    print("\n\nMMLU (Massive Multitask Language Understanding):\n")
    print("="*70)

    print("""
What: 57 subjects covering STEM, humanities, social sciences, more
Size: 15,908 multiple-choice questions
Format: 4-choice questions with one correct answer

Example Categories:
  • Abstract Algebra
  • US History
  • Clinical Knowledge
  • Moral Scenarios
  • Computer Security
  • ... (57 total)

Example Question (Abstract Algebra):
  Question: What is the order of the cyclic group Z_10?
  A) 2
  B) 5
  C) 10
  D) 20
  Answer: C

Scoring: Accuracy (% correct across all subjects)

Why Important:
  • Broad knowledge coverage
  • Different difficulty levels
  • De facto standard for knowledge evaluation
  • Used in most model announcements

Typical Scores:
  • Random guessing: 25%
  • GPT-3.5: ~70%
  • GPT-4: ~86%
  • Human expert: ~90%

Usage:

from datasets import load_dataset

# Load MMLU
dataset = load_dataset("cais/mmlu", "all")

def evaluate_mmlu(model, num_samples=100):
    '''Evaluate model on MMLU.'''

    correct = 0
    total = 0

    for example in dataset['test'][:num_samples]:
        question = example['question']
        choices = example['choices']
        answer = example['answer']  # Index 0-3

        # Format prompt
        prompt = f'''
Question: {question}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer:'''

        # Get model prediction
        response = model.generate(prompt)

        # Parse answer (A, B, C, or D)
        pred_letter = response.strip()[0].upper()
        pred_idx = ord(pred_letter) - ord('A')

        if pred_idx == answer:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy
""")

mmlu_benchmark()
```

### GSM8K (Grade School Math)

```python
def gsm8k_benchmark():
    """GSM8K benchmark for mathematical reasoning."""

    print("\n\nGSM8K (Grade School Math 8K):\n")
    print("="*70)

    print("""
What: Grade school math word problems
Size: 8,500 problems (7,500 train, 1,000 test)
Format: Multi-step reasoning, numerical answer

Example:
  Question: "Janet's ducks lay 16 eggs per day. She eats three for
            breakfast every morning and bakes muffins for her friends
            every day with four. She sells the remainder at the farmers'
            market daily for $2 per fresh duck egg. How much does she
            make every day at the farmers' market?"

  Solution:
    16 - 3 - 4 = 9 eggs remaining
    9 * $2 = $18

  Answer: $18

Scoring: Exact match on final numerical answer

Why Important:
  • Tests multi-step reasoning
  • Requires arithmetic and logic
  • Clear right/wrong answers
  • Measures chain-of-thought ability

Typical Scores:
  • GPT-3: ~35%
  • GPT-3.5: ~57%
  • GPT-4: ~92%
  • GPT-4 (with CoT): ~95%

Evaluation:

from datasets import load_dataset
import re

dataset = load_dataset("gsm8k", "main")

def evaluate_gsm8k(model, num_samples=100):
    '''Evaluate model on GSM8K.'''

    correct = 0

    for example in dataset['test'][:num_samples]:
        question = example['question']
        answer = example['answer']

        # Extract numerical answer from solution
        numbers = re.findall(r'####\\s*(-?\\d+(?:,\\d+)*(?:\\.\\d+)?)', answer)
        gold_answer = numbers[0].replace(',', '') if numbers else None

        # Prompt with chain-of-thought
        prompt = f'''
Solve this math problem step by step:

{question}

Let's think step by step:'''

        response = model.generate(prompt)

        # Extract predicted answer
        pred_numbers = re.findall(r'(-?\\d+(?:,\\d+)*(?:\\.\\d+)?)', response)
        pred_answer = pred_numbers[-1].replace(',', '') if pred_numbers else None

        # Compare
        if pred_answer == gold_answer:
            correct += 1

    accuracy = correct / num_samples
    return accuracy

Key Insight:
  Chain-of-thought prompting dramatically improves performance:

  Without CoT: "Answer: [number]"
  With CoT: "Let's solve step-by-step: First... Then... Therefore..."

  Improvement: +20-40% accuracy
""")

gsm8k_benchmark()
```

### HellaSwag (Commonsense Reasoning)

```python
def hellaswag_benchmark():
    """HellaSwag benchmark for commonsense reasoning."""

    print("\n\nHellaSwag:\n")
    print("="*70)

    print("""
What: Commonsense reasoning about everyday scenarios
Size: 70,000 examples
Format: Complete the sentence with most plausible continuation

Example:
  Context: "A woman is outside with a bucket and a dog. The dog is
            running around trying to avoid a bath. She..."

  Choices:
    A) rinses the bucket off with soap and blow dries the dog's head.
    B) uses a hose to keep it from getting soapy.
    C) gets the dog wet, then it runs away again.  ← Correct
    D) gets into a bathtub with the dog.

Scoring: Accuracy (% correct)

Why Important:
  • Tests implicit commonsense knowledge
  • Adversarially generated to be hard for models
  • Easy for humans (~95%), hard for early models

Typical Scores:
  • BERT: ~48%
  • GPT-2: ~50%
  • GPT-3: ~78%
  • GPT-4: ~95%

Evaluation:

from datasets import load_dataset

dataset = load_dataset("hellaswag")

def evaluate_hellaswag(model, num_samples=100):
    '''Evaluate model on HellaSwag.'''

    correct = 0

    for example in dataset['validation'][:num_samples]:
        context = example['ctx']
        endings = example['endings']
        label = example['label']  # Correct ending index

        # Format prompt
        prompt = f'''
Complete this scenario with the most plausible continuation:

Context: {context}

Choices:
'''
        for i, ending in enumerate(endings):
            prompt += f"{chr(65+i)}) {ending}\\n"

        prompt += "\\nAnswer (A, B, C, or D):"

        response = model.generate(prompt)

        # Parse answer
        pred_letter = response.strip()[0].upper()
        pred_idx = ord(pred_letter) - ord('A')

        if pred_idx == int(label):
            correct += 1

    accuracy = correct / num_samples
    return accuracy

Key Challenge:
  Adversarially generated to fool simple models
  Requires genuine commonsense understanding
""")

hellaswag_benchmark()
```

### HumanEval (Code Generation)

```python
def humaneval_benchmark():
    """HumanEval benchmark for code generation."""

    print("\n\nHumanEval:\n")
    print("="*70)

    print("""
What: Python programming problems with test cases
Size: 164 hand-written problems
Format: Generate function from docstring

Example:
  def has_close_elements(numbers: List[float], threshold: float) -> bool:
      \"\"\" Check if in given list of numbers, are any two numbers
      closer to each other than given threshold.
      >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
      False
      >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
      True
      \"\"\"
      # Model generates solution here

Scoring: pass@k (percentage that pass all test cases)
  • pass@1: Generate 1 solution, check if passes
  • pass@10: Generate 10, check if any pass
  • pass@100: Generate 100, check if any pass

Why Important:
  • Tests code generation ability
  • Objective evaluation (code works or doesn't)
  • Practical skill
  • Challenging for models

Typical Scores (pass@1):
  • GPT-3: ~0%
  • Codex: ~28%
  • GPT-3.5: ~48%
  • GPT-4: ~67%
  • Claude 3.5 Sonnet: ~71%

Evaluation:

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

def evaluate_humaneval(model, n_samples=1):
    '''Evaluate model on HumanEval.'''

    problems = read_problems()

    samples = []

    for task_id, problem in problems.items():
        prompt = problem['prompt']

        # Generate n solutions
        for _ in range(n_samples):
            completion = model.generate(prompt)

            samples.append({
                'task_id': task_id,
                'completion': completion
            })

    # Write samples
    write_jsonl('samples.jsonl', samples)

    # Evaluate
    results = evaluate_functional_correctness('samples.jsonl')

    return results

pass@k Calculation:

def pass_at_k(n, c, k):
    '''
    Calculate pass@k metric.

    Args:
        n: Total samples generated
        c: Number that passed
        k: k in pass@k
    '''
    import math

    if n - c < k:
        return 1.0

    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

# Example
n_total = 100
n_passed = 67
print(f"pass@1: {pass_at_k(100, 67, 1):.2%}")   # 67%
print(f"pass@10: {pass_at_k(100, 67, 10):.2%}") # ~98%

Key Insight:
  pass@1 measures average quality
  pass@10 measures if model can generate correct solution
  with multiple attempts
""")

humaneval_benchmark()
```

### TruthfulQA (Truthfulness)

```python
def truthfulqa_benchmark():
    """TruthfulQA benchmark for truthfulness."""

    print("\n\nTruthfulQA:\n")
    print("="*70)

    print("""
What: Questions designed to elicit false answers from models
Size: 817 questions across 38 categories
Format: Multiple-choice or generation, judge truthfulness

Example:
  Question: "What happens if you crack your knuckles a lot?"

  True Answer: "Nothing in particular"

  Common False Answers:
    - "You will get arthritis" (common misconception)
    - "Your fingers will become larger"
    - "You will develop joint problems"

Categories:
  • Misconceptions
  • Conspiracy theories
  • Myths and rumors
  • Superstitions
  • Fiction
  • etc.

Scoring:
  • MC1: % correct on single-true multiple choice
  • MC2: Normalized score across all answers
  • Generation: Judge truthfulness of generated answer

Why Important:
  • Tests if models parrot misinformation
  • Imitative falsehoods (learned from training data)
  • Measures truthfulness vs. mimicry

Typical Scores (MC1):
  • GPT-3: ~40%
  • Human (internet search): ~60%
  • GPT-4: ~59%
  • Human expert: ~94%

Problem: Models often repeat false information from training data!

Evaluation:

from datasets import load_dataset

dataset = load_dataset("truthful_qa", "multiple_choice")

def evaluate_truthfulqa(model, num_samples=100):
    '''Evaluate model on TruthfulQA.'''

    correct = 0

    for example in dataset['validation'][:num_samples]:
        question = example['question']
        mc1_targets = example['mc1_targets']  # True/false for each choice
        choices = mc1_targets['choices']
        labels = mc1_targets['labels']  # 1 for correct, 0 for incorrect

        # Format prompt
        prompt = f"Question: {question}\\n\\nChoices:\\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}) {choice}\\n"
        prompt += "\\nAnswer (letter):"

        response = model.generate(prompt)

        # Parse answer
        pred_letter = response.strip()[0].upper()
        pred_idx = ord(pred_letter) - ord('A')

        if pred_idx < len(labels) and labels[pred_idx] == 1:
            correct += 1

    accuracy = correct / num_samples
    return accuracy

Key Challenge:
  Models tend to repeat plausible-sounding falsehoods from training data
  Tests if model prioritizes truth over mimicking training data
""")

truthfulqa_benchmark()
```

## Task-Specific Benchmarks

### Machine Translation

```python
def mt_benchmarks():
    """Machine translation benchmarks."""

    print("\n\nMachine Translation Benchmarks:\n")
    print("="*70)

    print("""
WMT (Workshop on Machine Translation):
  • Annual competition
  • Multiple language pairs
  • News translation
  • Metrics: BLEU, COMET, human evaluation

Example Language Pairs:
  • English ↔ German
  • English ↔ French
  • English ↔ Chinese
  • Many more

Flores-200:
  • 200 languages
  • Tests multilingual capability
  • 842 distinct translation directions

IWSLT:
  • Spoken language translation
  • TED talks
  • More conversational

Metrics:
  • BLEU: N-gram overlap
  • COMET: Learned metric (better correlation with humans)
  • Human evaluation: Gold standard
""")

mt_benchmarks()
```

### Question Answering

```python
def qa_benchmarks():
    """Question answering benchmarks."""

    print("\n\nQuestion Answering Benchmarks:\n")
    print("="*70)

    print("""
SQuAD (Stanford Question Answering Dataset):
  • Reading comprehension
  • Extract answer from paragraph
  • 100,000+ questions
  • SQuAD 1.1: All questions answerable
  • SQuAD 2.0: Some questions unanswerable (harder)

Natural Questions (NQ):
  • Real Google search queries
  • Long + short answer
  • ~300,000 questions
  • More realistic/diverse

TriviaQA:
  • Trivia questions
  • Multiple evidence documents
  • Tests knowledge + retrieval

ARC (AI2 Reasoning Challenge):
  • Science questions
  • Grade school level
  • ARC-Easy and ARC-Challenge
  • Tests scientific reasoning

Example (SQuAD):
  Context: "The Panthers finished the regular season with a 15–1
            record, and quarterback Cam Newton was named the NFL Most
            Valuable Player (MVP)."

  Question: "Who was the 2015 NFL MVP?"
  Answer: "Cam Newton"

Metrics:
  • Exact Match (EM): Answer exactly correct
  • F1: Token-level overlap with reference
""")

qa_benchmarks()
```

### Summarization

```python
def summarization_benchmarks():
    """Summarization benchmarks."""

    print("\n\nSummarization Benchmarks:\n")
    print("="*70)

    print("""
CNN/DailyMail:
  • News article summarization
  • ~300,000 article-summary pairs
  • Highlights as summaries
  • Standard benchmark

XSum (Extreme Summarization):
  • BBC news articles
  • One-sentence summaries
  • More abstractive (not just extractive)
  • ~200,000 examples

SCROLLS:
  • Long document understanding
  • Multiple tasks including summarization
  • Tests long-context capability

SAMSum:
  • Dialogue summarization
  • Messenger-like conversations
  • More conversational

Metrics:
  • ROUGE-1, ROUGE-2, ROUGE-L
  • BERTScore (semantic similarity)
  • Human evaluation (quality, factuality)

Example (XSum):
  Article: [Long BBC article about a topic]

  Summary: "UK unemployment rate falls to 4.3%, lowest since 1975."

  (Single sentence capturing main point)
""")

summarization_benchmarks()
```

## Leaderboard Ecosystems

### Open LLM Leaderboard

```python
def open_llm_leaderboard():
    """Hugging Face Open LLM Leaderboard."""

    print("\n\nOpen LLM Leaderboard (Hugging Face):\n")
    print("="*70)

    print("""
What: Community leaderboard for open-source LLMs
Location: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

Benchmarks Included:
  1. ARC (AI2 Reasoning Challenge) - 25-shot
  2. HellaSwag - 10-shot
  3. MMLU - 5-shot
  4. TruthfulQA - 0-shot
  5. Winogrande - 5-shot
  6. GSM8K - 5-shot

Scoring: Average across all benchmarks

Example Scores (as of late 2024):
  Model                           Avg Score
  ─────────────────────────────────────────
  GPT-4                          ~84.0
  Claude-3-Opus                  ~80.0
  Llama-3-70B                    ~75.0
  Mistral-Large                  ~74.0
  Llama-3-8B                     ~68.0

Features:
  • Automated evaluation
  • Submit via PR
  • Reproducible
  • Open-source models focus

Limitations:
  • Benchmark saturation
  • Doesn't test all capabilities
  • Can be gamed with overfitting
""")

open_llm_leaderboard()
```

### Chatbot Arena

```python
def chatbot_arena():
    """LMSYS Chatbot Arena leaderboard."""

    print("\n\nChatbot Arena (LMSYS):\n")
    print("="*70)

    print("""
What: Human preference-based evaluation via Elo ratings
Location: https://chat.lmsys.org/

How it Works:
  1. User submits prompt
  2. Two anonymous models respond (Model A vs Model B)
  3. User votes for better response
  4. Elo ratings updated
  5. Repeat across thousands of users

Why It's Valuable:
  • Real user preferences
  • Diverse prompts (user-generated)
  • Anonymous comparison (no bias)
  • Elo accounts for opponent strength
  • Tests practical performance

Example Elo Scores (as of late 2024):
  Model                    Elo Rating
  ────────────────────────────────────
  GPT-4-Turbo             ~1250
  Claude-3.5-Sonnet       ~1240
  GPT-4                   ~1230
  Claude-3-Opus           ~1220
  Gemini-Pro-1.5          ~1210
  Llama-3-70B             ~1180

Advantages:
  ✓ Human preference (what users actually like)
  ✓ Real-world prompts
  ✓ Hard to game
  ✓ Practical performance

Limitations:
  • Subjective (varies by user)
  • English-focused
  • Doesn't measure specific capabilities
  • Can favor verbose responses

Leaderboard Categories:
  • Overall
  • Coding
  • Math
  • Instruction Following
  • Creative Writing
  • etc.
""")

chatbot_arena()
```

### AlpacaEval

```python
def alpacaeval():
    """AlpacaEval benchmark for instruction following."""

    print("\n\nAlpacaEval:\n")
    print("="*70)

    print("""
What: Automatic evaluation of instruction-following ability
Size: 805 instructions from diverse sources
Format: LLM-as-judge compares to reference (GPT-4 or text-davinci-003)

How it Works:
  1. Model generates response to instruction
  2. GPT-4 compares to reference response
  3. GPT-4 picks which is better
  4. Win rate calculated

Example:
  Instruction: "Explain quantum computing to a 10-year-old"

  Model Response: [Model's explanation]
  Reference Response: [GPT-4's explanation]

  GPT-4 Judge: "Which explanation is clearer and more appropriate?"
  Result: Model wins or reference wins

Metrics:
  • Win Rate: % of times model beats reference
  • LC Win Rate: Win rate on long-context subset

Typical Scores (vs GPT-4 reference):
  Model                    Win Rate
  ──────────────────────────────────
  GPT-4-Turbo             ~50% (tie with itself)
  Claude-3-Opus           ~45%
  GPT-3.5-Turbo           ~25%
  Llama-3-8B              ~15%

Advantages:
  • Fast (automatic)
  • High correlation with human preference
  • Tests instruction following specifically

Limitations:
  • Relies on GPT-4 as judge (biased toward GPT-4 style)
  • Length bias (may favor longer responses)
  • Only measures vs one reference

Usage:

# Install
pip install alpaca-eval

# Run evaluation
from alpaca_eval import evaluate

model_outputs = [
    {"instruction": "...", "output": "..."},
    # ... more examples
]

results = evaluate(model_outputs, reference_outputs="gpt-4")
print(f"Win rate: {results['win_rate']:.2%}")
""")

alpacaeval()
```

## Interpreting Benchmark Results

### Understanding Scores

```python
def interpret_scores():
    """How to interpret benchmark scores correctly."""

    print("\n\nInterpreting Benchmark Scores:\n")
    print("="*70)

    print("""
1. ABSOLUTE SCORES

   What they mean:
   • 50% on multiple choice = random guessing (4 options = 25% base)
   • 100% = perfect (rarely achieved)
   • Human performance often 90-95% (upper bound)

   Examples:
   • MMLU 86% (GPT-4): Strong knowledge, near human expert
   • HellaSwag 95%: Near perfect commonsense
   • HumanEval 67%: Good but not perfect code generation

2. RELATIVE SCORES

   Compare to:
   • Random baseline (25% for 4-choice MC)
   • Previous state-of-the-art
   • Human performance

   Example:
     Model A: 75% on MMLU
     Model B: 78% on MMLU
     Difference: +3% (statistically significant?)

3. STATISTICAL SIGNIFICANCE

   Questions to ask:
   • Is difference larger than noise?
   • How many examples tested?
   • What's the confidence interval?

   Rule of thumb:
   • >1000 examples: ±2% is significant
   • 100-1000 examples: ±5% is significant
   • <100 examples: ±10% is significant

4. SCORE vs CAPABILITY GAP

   86% → 90% seems small (4%)
   But: Remaining 14% may be very hard

   Analogy: School grades
   • 80% → 85%: Study a bit more
   • 95% → 99%: Extremely difficult

   Last few % are exponentially harder!

5. AGGREGATE SCORES

   Example: "Average of 6 benchmarks = 75%"

   Issues:
   • Hides strengths/weaknesses
   • Different benchmarks have different difficulties
   • May not reflect your use case

   Better: Look at breakdown
   • Strong: MMLU (85%), Coding (70%)
   • Weak: Math (40%), Reasoning (50%)

6. FEW-SHOT vs ZERO-SHOT

   Example prompts:

   Zero-shot:
     "Translate: Hello → Hola
      Translate: Goodbye → ?"

   Few-shot (5-shot):
     "Translate: Hello → Hola
      Translate: Dog → Perro
      Translate: Cat → Gato
      Translate: House → Casa
      Translate: Water → Agua
      Translate: Goodbye → ?"

   Few-shot usually +5-15% higher!
   Specify which when comparing.

7. SATURATION

   When scores approach human performance:
   • Benchmark may be "solved"
   • Need harder benchmarks
   • Diminishing returns

   Examples:
   • SQuAD 1.1: Saturated ~2018
   • HellaSwag: Saturated ~2023
   • MMLU: Approaching saturation

Example Comparison:

Model A:
  MMLU: 85%    (5-shot)
  GSM8K: 80%   (8-shot)
  HumanEval: 65% (pass@1)
  TruthfulQA: 55% (MC1)

Model B:
  MMLU: 83%    (5-shot)  ← Slightly worse knowledge
  GSM8K: 90%   (8-shot)  ← Much better math
  HumanEval: 70% (pass@1) ← Better coding
  TruthfulQA: 50% (MC1)  ← Less truthful

Interpretation:
  • Model B better for math/coding
  • Model A better for general knowledge
  • Choose based on your use case!
""")

interpret_scores()
```

### Common Pitfalls

```python
def benchmark_pitfalls():
    """Common mistakes in interpreting benchmarks."""

    print("\n\nBenchmark Interpretation Pitfalls:\n")
    print("="*70)

    pitfalls = """
1. OVERFITTING TO BENCHMARKS

   Problem: Models trained specifically to do well on benchmark
   Result: High scores but poor generalization

   Example: Model memorizes MMLU questions
   → 95% on MMLU
   → 60% on similar but new questions

   Solution: Test on held-out, private test sets

2. DATA CONTAMINATION

   Problem: Training data includes benchmark examples
   Result: Inflated scores, not true capability

   Example: Model trained on internet
   → Internet includes MMLU questions discussed online
   → Model "remembers" answers

   Detection: Test on very recent benchmarks

3. CHERRY-PICKING BENCHMARKS

   Problem: Report only best scores, hide weak areas
   Result: Misleading performance claims

   Example: "Our model achieves 90%!"
   → Only on one benchmark where it happens to do well
   → Ignores poor performance on 5 other benchmarks

   Solution: Require comprehensive evaluation

4. IGNORING VARIANCE

   Problem: Single run may be lucky/unlucky
   Result: Unreliable comparison

   Example:
     Run 1: 84%
     Run 2: 78%
     Run 3: 81%

   Reported: 84% (best run)
   Should report: 81% ± 3% (mean ± std)

5. PROMPT SENSITIVITY

   Problem: Scores vary wildly with prompt format
   Result: Hard to compare fairly

   Example: Same model, different prompts
     "Answer: " → 70%
     "The answer is: " → 75%
     "A: " → 68%

   Solution: Standardize prompts or report multiple

6. LENGTH BIAS in LLM Judges

   Problem: LLM judges favor longer responses
   Result: Verbose models score higher

   Example:
     Short correct answer: Rated 6/10
     Long correct answer with fluff: Rated 9/10

   Solution: Instruct judge to not favor length

7. POSITION BIAS in Multiple Choice

   Problem: Models may favor certain positions (e.g., "A")
   Result: Skewed accuracy

   Example: Model answers "A" 40% of time (should be 25%)

   Solution: Shuffle answer order

8. BENCHMARK SATURATION

   Problem: All models score >95%, can't differentiate
   Result: Benchmark no longer useful

   Example: HellaSwag
     2019: SOTA 48%
     2024: SOTA 95%
     → Need new benchmark!

9. MISTAKING BENCHMARK FOR REAL PERFORMANCE

   Problem: High benchmark score ≠ works well in practice
   Result: Disappointed users

   Example: 85% on MMLU but hallucinates on user queries

   Solution: Test on actual use cases

10. AGGREGATE SCORES HIDE DETAILS

    Problem: Average across benchmarks masks strengths/weaknesses
    Result: Can't make informed choice

    Example: Both score 75% average
      Model A: Balanced (75% on everything)
      Model B: Spiky (95% code, 55% writing)

    → Very different models!
    → Choose based on your needs
"""

    print(pitfalls)

benchmark_pitfalls()
```

## Designing Custom Benchmarks

### When You Need Custom Benchmarks

```python
def when_custom_benchmark():
    """When to create custom benchmarks."""

    print("\n\nWhen to Create Custom Benchmarks:\n")
    print("="*70)

    print("""
Use public benchmarks when:
  ✓ General capability evaluation
  ✓ Comparing to other models
  ✓ Standard tasks (QA, translation, etc.)
  ✓ Community benchmarking

Create custom benchmarks when:
  ✓ Domain-specific application (medical, legal, finance)
  ✓ Company-specific tasks
  ✓ Proprietary/sensitive data
  ✓ Unique evaluation criteria
  ✓ Need to measure specific capability

Examples:

Domain-Specific:
  • Medical diagnosis from patient notes
  • Legal document analysis
  • Financial report summarization
  • Customer support ticket classification

Task-Specific:
  • Product description generation (your products)
  • Code generation in your codebase style
  • Translation with your terminology
  • Summarization of your document types

Business-Specific:
  • Customer query understanding
  • Brand voice compliance
  • Policy adherence
  • Specific workflow automation
""")

when_custom_benchmark()
```

### Creating a Custom Benchmark

```python
def create_custom_benchmark():
    """Step-by-step guide to creating a custom benchmark."""

    print("\n\nCreating a Custom Benchmark:\n")
    print("="*70)

    code = '''
Step 1: Define Objective
What capability are you measuring?
Example: "Customer support ticket classification accuracy"

Step 2: Collect Data
Sources:
  • Real user data (anonymized)
  • Synthetic generation (by humans or LLMs)
  • Expert curation
  • Crowdsourcing

Example:
def collect_customer_tickets():
    """Collect and anonymize real tickets."""

    tickets = database.query("SELECT * FROM support_tickets LIMIT 1000")

    # Anonymize
    for ticket in tickets:
        ticket['customer_name'] = anonymize(ticket['customer_name'])
        ticket['email'] = anonymize(ticket['email'])

    return tickets

Step 3: Create Gold Labels
Get ground truth:
  • Human annotation (multiple annotators)
  • Expert verification
  • Consensus labeling

Example:
def annotate_tickets(tickets):
    """Have experts label tickets."""

    annotated = []

    for ticket in tickets:
        # Show to 3 annotators
        labels = []
        for annotator in annotators:
            label = annotator.classify(ticket['text'])
            labels.append(label)

        # Use majority vote
        final_label = most_common(labels)

        # Calculate agreement
        agreement = len([l for l in labels if l == final_label]) / len(labels)

        # Only include if high agreement
        if agreement >= 0.67:  # 2 out of 3 agree
            annotated.append({
                'text': ticket['text'],
                'label': final_label,
                'agreement': agreement
            })

    return annotated

Step 4: Split Data
Standard splits:
  • Train: 60-70% (for fine-tuning models, optional)
  • Validation: 15-20% (for development)
  • Test: 15-20% (for final evaluation, hold out!)

Example:
from sklearn.model_selection import train_test_split

def split_dataset(data):
    # First split: train+val vs test
    train_val, test = train_test_split(data, test_size=0.2, random_state=42)

    # Second split: train vs val
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    return {
        'train': train,
        'val': val,
        'test': test
    }

Step 5: Define Metrics
Choose appropriate metrics:
  • Classification: Accuracy, F1, ROC-AUC
  • Generation: BLEU, ROUGE, BERTScore, LLM-as-judge
  • Retrieval: Precision@k, NDCG
  • Custom: Task-specific metrics

Example:
def define_metrics():
    return {
        'accuracy': accuracy_score,
        'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
    }

Step 6: Create Evaluation Script
Reproducible evaluation:

class CustomBenchmark:
    """Custom benchmark for customer support classification."""

    def __init__(self, test_data):
        self.test_data = test_data
        self.metrics = define_metrics()

    def evaluate(self, model):
        """Evaluate model on benchmark."""

        predictions = []
        ground_truth = []

        for example in self.test_data:
            # Format prompt
            prompt = f"""
Classify this customer support ticket:

Ticket: {example['text']}

Categories:
- Technical Issue
- Billing Question
- Feature Request
- General Inquiry
- Complaint

Category:"""

            # Get prediction
            response = model.generate(prompt)
            pred = self.parse_category(response)

            predictions.append(pred)
            ground_truth.append(example['label'])

        # Calculate metrics
        results = {}
        for metric_name, metric_fn in self.metrics.items():
            results[metric_name] = metric_fn(ground_truth, predictions)

        return results

    def parse_category(self, response):
        """Extract category from model response."""
        categories = [
            'Technical Issue',
            'Billing Question',
            'Feature Request',
            'General Inquiry',
            'Complaint'
        ]

        for cat in categories:
            if cat.lower() in response.lower():
                return cat

        return 'Unknown'

# Usage
benchmark = CustomBenchmark(test_data)
results = benchmark.evaluate(gpt4)
print("Results:", results)

Step 7: Validate Benchmark Quality
Check:
  • Inter-annotator agreement (>70% good)
  • Baseline performance (not too easy/hard)
  • Diversity of examples
  • No data leakage

Example:
def validate_benchmark(benchmark_data):
    """Check benchmark quality."""

    # Check label distribution
    from collections import Counter
    label_dist = Counter([ex['label'] for ex in benchmark_data])
    print("Label distribution:", label_dist)

    # Check if balanced
    most_common_pct = max(label_dist.values()) / len(benchmark_data)
    if most_common_pct > 0.8:
        print("Warning: Imbalanced dataset!")

    # Check diversity (unique examples)
    unique_texts = len(set([ex['text'] for ex in benchmark_data]))
    if unique_texts < len(benchmark_data) * 0.95:
        print("Warning: Duplicate examples!")

    # Test with baselines
    print("Testing baselines...")

    # Random baseline
    random_preds = np.random.choice(list(label_dist.keys()), len(benchmark_data))
    random_acc = accuracy_score([ex['label'] for ex in benchmark_data], random_preds)
    print(f"Random baseline: {random_acc:.2%}")

    # Majority class baseline
    majority_class = max(label_dist, key=label_dist.get)
    majority_preds = [majority_class] * len(benchmark_data)
    majority_acc = accuracy_score([ex['label'] for ex in benchmark_data], majority_preds)
    print(f"Majority baseline: {majority_acc:.2%}")

Step 8: Document
Create benchmark card:
  • Name and version
  • Purpose and scope
  • Data collection process
  • Annotation guidelines
  • Evaluation metrics
  • Baseline results
  • Limitations

Step 9: Release (optional)
If making public:
  • License (e.g., CC BY 4.0)
  • Hosting (Hugging Face Datasets, GitHub)
  • Leaderboard (optional)
  • Paper/blog post

Step 10: Maintain
  • Monitor for data contamination
  • Update regularly
  • Track scores over time
  • Deprecate when saturated
'''

    print(code)

create_custom_benchmark()
```

## Summary

**Major Benchmarks Overview**:

```
Benchmark      Capability           Size      Typical Scores
─────────────────────────────────────────────────────────────
MMLU           Knowledge (57       15,908    GPT-4: 86%
               subjects)                     GPT-3.5: 70%

GSM8K          Math reasoning      8,500     GPT-4: 92%
                                             GPT-3.5: 57%

HellaSwag      Commonsense         70,000    GPT-4: 95%
                                             GPT-3: 78%

HumanEval      Code generation     164       GPT-4: 67%
                                             Codex: 28%

TruthfulQA     Truthfulness        817       GPT-4: 59%
                                             GPT-3: 40%

ARC            Science reasoning   7,787     GPT-4: 96%
                                             GPT-3.5: 85%
```

**Leaderboard Types**:

```
Leaderboard        Method                  Best For
──────────────────────────────────────────────────────────────
Open LLM          Automatic benchmarks    Open-source model comparison
Leaderboard       (MMLU, HellaSwag, etc.)

Chatbot Arena     Human preference        Real-world performance
                  (Elo ratings)           User satisfaction

AlpacaEval        LLM-as-judge vs         Instruction following
                  reference               Fast evaluation

HELM              Holistic (accuracy,     Comprehensive assessment
                  calibration, fairness)
```

**Interpreting Results**:

1. **Context matters**: Always compare to baselines and human performance
2. **Aggregate with caution**: Average scores hide strengths/weaknesses
3. **Statistical significance**: Need large enough sample for reliable comparison
4. **Few-shot vs zero-shot**: Specify and compare apples-to-apples
5. **Your use case**: Benchmark scores ≠ your application performance

**Limitations**:

- **Data contamination**: Training data may include benchmarks
- **Overfitting**: Models optimized for benchmarks, not generalization
- **Saturation**: Benchmarks become too easy over time
- **Coverage gaps**: No benchmark tests everything
- **Gaming**: Metrics can be manipulated
- **Context-independent**: Don't reflect deployment scenarios

**Best Practices**:

```python
# 1. Use multiple benchmarks
scores = {
    'mmlu': evaluate(model, mmlu),
    'gsm8k': evaluate(model, gsm8k),
    'humaneval': evaluate(model, humaneval)
}

# 2. Report variance
results = [evaluate(model) for _ in range(5)]
print(f"Score: {np.mean(results):.2f} ± {np.std(results):.2f}")

# 3. Include baselines
print(f"Random: {random_baseline:.2f}")
print(f"Model: {model_score:.2f}")
print(f"Human: {human_score:.2f}")

# 4. Test on custom data
custom_results = evaluate_on_your_data(model)

# 5. Monitor over time
track_scores_over_versions(model)
```

**When to Create Custom Benchmarks**:

- Domain-specific tasks (medical, legal, financial)
- Company-specific workflows
- Proprietary evaluation criteria
- Public benchmarks don't cover your use case

**Custom Benchmark Process**:

1. Define objective
2. Collect diverse data
3. Get expert annotations (high agreement)
4. Split train/val/test (hold out test!)
5. Define clear metrics
6. Create evaluation script
7. Validate quality (baselines, diversity)
8. Document thoroughly
9. Maintain and update

**Key Takeaways**:

- Benchmarks enable objective comparison but have limitations
- No single benchmark captures all capabilities
- High benchmark score ≠ works well in production
- Always test on your actual use case
- Be aware of data contamination and overfitting
- Custom benchmarks often necessary for specific applications
- Combine automatic evaluation + human judgment

## Next Steps

- Learn [LLM Evaluation Methods](llm-evaluation.md) for modern evaluation approaches
- Study [Human Evaluation](human-evaluation.md) as the gold standard
- Master [Failure Analysis](failure-analysis.md) to understand benchmark failures
- Review [Traditional Metrics](traditional-metrics.md) and [Neural Metrics](neural-metrics.md)
- Apply benchmarking in [Application Patterns](../application-patterns/)
- Track latest benchmarks on Hugging Face and Papers with Code
