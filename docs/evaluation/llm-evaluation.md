# LLM Evaluation Methods

## Table of Contents

- [Introduction](#introduction)
- [Evaluation Challenges for LLMs](#evaluation-challenges-for-llms)
- [Task-Specific Evaluation](#task-specific-evaluation)
- [LLM-as-Judge](#llm-as-judge)
- [Pairwise Comparison](#pairwise-comparison)
- [Multi-Dimensional Evaluation](#multi-dimensional-evaluation)
- [Factuality and Truthfulness](#factuality-and-truthfulness)
- [Safety and Toxicity](#safety-and-toxicity)
- [Consistency and Reliability](#consistency-and-reliability)
- [Evaluation Framework Design](#evaluation-framework-design)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Evaluating large language models (LLMs) requires new approaches beyond traditional metrics. LLMs generate open-ended text where multiple valid answers exist, making automatic evaluation challenging.

```
Traditional NLP:               LLM Evaluation:
┌──────────────────┐          ┌──────────────────┐
│  Single answer   │          │  Many valid      │
│  Clear metrics   │          │  answers         │
│  (accuracy, F1)  │   vs     │  Subjective      │
│                  │          │  quality         │
│  "What is 2+2?"  │          │  "Write a story" │
│  → "4" (correct) │          │  → (many good    │
│                  │          │     answers)     │
└──────────────────┘          └──────────────────┘

Evaluation Dimensions:
  • Correctness
  • Coherence
  • Relevance
  • Fluency
  • Factuality
  • Safety
  • Consistency
```

**Key challenges**:

1. **Open-ended generation**: No single correct answer
2. **Subjective quality**: What makes good text?
3. **Long outputs**: Evaluating multi-paragraph text
4. **Multiple capabilities**: Reasoning, knowledge, instruction-following, etc.
5. **Safety concerns**: Toxicity, bias, harmful content

This guide covers modern LLM evaluation methods, from task-specific approaches to using LLMs as judges.

## Evaluation Challenges for LLMs

### The Open-Ended Problem

```python
def demonstrate_open_ended_challenge():
    """Show why open-ended generation is hard to evaluate."""

    prompt = "Write a short story about a robot learning to paint."

    valid_responses = [
        "Story 1: Unit-7 discovered colors one day...",
        "Story 2: In a world of gray, RB-9 found a brush...",
        "Story 3: The android stared at the canvas, confused...",
        "Story 4: Paint splattered everywhere as Bot-3 learned...",
    ]

    print("Open-Ended Evaluation Challenge:\n")
    print("="*70)
    print(f"\nPrompt: {prompt}\n")
    print("All these responses could be valid:")

    for i, response in enumerate(valid_responses, 1):
        print(f"\n{i}. {response}")

    print("\n" + "="*70)
    print("\nProblems:")
    print("  • No single 'correct' answer")
    print("  • Different styles all valid")
    print("  • Subjective quality judgment")
    print("  • Traditional metrics (BLEU) don't work without reference")
    print("\nSolutions:")
    print("  → Human evaluation")
    print("  → Multi-dimensional scoring")
    print("  → LLM-as-judge")
    print("  → Pairwise comparison")

demonstrate_open_ended_challenge()
```

### Inconsistency and Variability

```python
def demonstrate_llm_variability():
    """Show LLM output variability."""

    print("LLM Output Variability:\n")
    print("="*70)

    print("""
Same prompt, different runs:

Prompt: "What is the capital of France?"

Run 1: "The capital of France is Paris."
Run 2: "Paris is the capital of France."
Run 3: "France's capital city is Paris."
Run 4: "Paris, which is the capital of France, is..."

All correct, but different!

Implications for evaluation:
  • Need multiple runs for reliability
  • Small wording changes affect metrics
  • Temperature affects variability
  • Seed fixing helps reproducibility

Best practices:
  1. Run with temperature=0 for determinism
  2. Average over multiple samples if using temp>0
  3. Report variance alongside mean
  4. Use seed for reproducibility
""")

demonstrate_llm_variability()
```

### Context Window Limitations

```python
def context_evaluation_challenge():
    """Evaluating long-context capabilities."""

    print("Long Context Evaluation:\n")
    print("="*70)

    print("""
Challenge: Evaluate if LLM uses information from long context

Example task:
  • Give LLM 10-page document
  • Ask question requiring info from page 7
  • Check if answer uses that specific information

Metrics:
  1. Needle-in-haystack: Find specific fact
     → Success rate

  2. Information synthesis: Combine multiple facts
     → Correctness + completeness

  3. Context following: Use only provided context
     → Hallucination rate

  4. Position bias: Performance by position in context
     → Accuracy at beginning/middle/end

Implementation:

def evaluate_context_usage(llm, context, question, answer_key):
    '''Evaluate if LLM uses long context correctly.'''

    # Generate answer
    response = llm(context + "\\n\\nQuestion: " + question)

    # Check correctness
    correct = answer_key.lower() in response.lower()

    # Check if hallucinated (cited info not in context)
    hallucinated = check_hallucination(response, context)

    return {
        'correct': correct,
        'hallucinated': hallucinated,
        'length': len(response.split())
    }
""")

context_evaluation_challenge()
```

## Task-Specific Evaluation

### Question Answering

```python
def evaluate_qa():
    """Evaluate question answering systems."""

    print("Question Answering Evaluation:\n")
    print("="*70)

    print("""
Metrics for QA:

1. EXACT MATCH (EM)
   • Answer exactly matches reference
   • Strict but clear

   def exact_match(pred, gold):
       return normalize(pred) == normalize(gold)

   normalize: lowercase, remove articles/punctuation

2. F1 SCORE (token overlap)
   • Treats answer as bag of tokens
   • Partial credit for overlap

   def qa_f1(pred, gold):
       pred_tokens = set(pred.lower().split())
       gold_tokens = set(gold.lower().split())

       if len(pred_tokens) == 0 or len(gold_tokens) == 0:
           return 0

       common = pred_tokens & gold_tokens
       precision = len(common) / len(pred_tokens)
       recall = len(common) / len(gold_tokens)

       if precision + recall == 0:
           return 0

       f1 = 2 * (precision * recall) / (precision + recall)
       return f1

3. SEMANTIC SIMILARITY
   • Use embeddings to compare meaning
   • Handles paraphrases

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('all-MiniLM-L6-v2')
   pred_emb = model.encode(pred)
   gold_emb = model.encode(gold)
   similarity = cosine_similarity(pred_emb, gold_emb)

4. ANSWER PRESENCE
   • For multi-sentence context
   • Check if answer appears anywhere

   def answer_present(pred, gold):
       return gold.lower() in pred.lower()
""")

    # Example implementation
    def evaluate_qa_response(prediction, gold_answer):
        """Comprehensive QA evaluation."""

        # Normalize
        def normalize(text):
            import re
            text = text.lower()
            text = re.sub(r'\b(a|an|the)\b', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()

        pred_norm = normalize(prediction)
        gold_norm = normalize(gold_answer)

        # Exact match
        em = int(pred_norm == gold_norm)

        # F1
        pred_tokens = set(pred_norm.split())
        gold_tokens = set(gold_norm.split())

        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            f1 = 0
        else:
            common = pred_tokens & gold_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gold_tokens)

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

        print(f"Prediction: {prediction}")
        print(f"Gold:       {gold_answer}")
        print(f"EM:         {em}")
        print(f"F1:         {f1:.3f}")

        return {'em': em, 'f1': f1}

    # Test
    print("\nExample Evaluation:")
    print("="*70 + "\n")
    evaluate_qa_response("Paris", "Paris")
    print()
    evaluate_qa_response("Paris, France", "Paris")
    print()
    evaluate_qa_response("The capital is Paris", "Paris")

evaluate_qa()
```

### Summarization

```python
def evaluate_summarization():
    """Evaluate summarization quality."""

    print("\n\nSummarization Evaluation:\n")
    print("="*70)

    print("""
Multi-dimensional evaluation:

1. RELEVANCE
   • Does summary cover key points?
   • Scored 1-5 by LLM judge or human

   Prompt: "Rate relevance of summary to document (1-5)"

2. COHERENCE
   • Is summary well-structured and logical?
   • Checks flow and organization

   Prompt: "Rate coherence of summary (1-5)"

3. CONSISTENCY
   • Does summary contradict source?
   • Checks factual alignment

   Prompt: "Are there contradictions? (yes/no)"

4. FLUENCY
   • Is summary grammatically correct?
   • Natural language

   Use perplexity or grammar checker

5. COVERAGE
   • What % of key points are covered?
   • Extract key facts, check presence

   def coverage(summary, key_points):
       covered = sum(1 for kp in key_points if kp in summary)
       return covered / len(key_points)

Implementation example:

def evaluate_summary_comprehensive(source, summary, reference=None):
    '''Multi-dimensional summary evaluation.'''

    results = {}

    # If reference available
    if reference:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        results['rouge'] = scorer.score(reference, summary)

    # Length check
    results['length'] = len(summary.split())
    results['compression_ratio'] = len(summary.split()) / len(source.split())

    # Factuality check (simplified)
    # In practice, use NLI model or LLM
    results['likely_factual'] = not contains_hallucination(summary, source)

    return results
""")

evaluate_summarization()
```

### Code Generation

```python
def evaluate_code_generation():
    """Evaluate code generation from LLMs."""

    print("\n\nCode Generation Evaluation:\n")
    print("="*70)

    print("""
Code-specific metrics:

1. PASS@K
   • Generate k code samples
   • Check how many pass test cases
   • pass@1, pass@10, pass@100

   def pass_at_k(n, c, k):
       '''
       n: total samples
       c: number that passed
       k: k in pass@k
       '''
       if n - c < k:
           return 1.0
       return 1.0 - math.comb(n - c, k) / math.comb(n, k)

2. EXECUTION SUCCESS
   • Does code run without errors?
   • Simple but important baseline

   def execution_success(code):
       try:
           exec(code)
           return True
       except:
           return False

3. TEST CASE PASSING
   • Run against test suite
   • Count passed / total tests

   def run_tests(code, test_cases):
       passed = 0
       for test_input, expected in test_cases:
           try:
               result = execute_code(code, test_input)
               if result == expected:
                   passed += 1
           except:
               pass
       return passed / len(test_cases)

4. CODE QUALITY
   • Readability (docstrings, comments)
   • Efficiency (time/space complexity)
   • Style (PEP 8 for Python)

   Use linters: pylint, flake8, black

Example evaluation:

def evaluate_generated_code(code, test_cases):
    '''Comprehensive code evaluation.'''

    results = {
        'syntax_valid': False,
        'executes': False,
        'tests_passed': 0,
        'tests_total': len(test_cases),
        'pass_rate': 0.0
    }

    # Check syntax
    try:
        compile(code, '<string>', 'exec')
        results['syntax_valid'] = True
    except SyntaxError:
        return results

    # Check execution
    try:
        exec(code)
        results['executes'] = True
    except:
        return results

    # Run test cases
    for test_input, expected in test_cases:
        try:
            # Extract function and run
            result = run_code_with_input(code, test_input)
            if result == expected:
                results['tests_passed'] += 1
        except:
            pass

    results['pass_rate'] = results['tests_passed'] / results['tests_total']

    return results
""")

evaluate_code_generation()
```

## LLM-as-Judge

### Using LLMs to Evaluate LLMs

```python
def llm_as_judge_introduction():
    """Introduction to LLM-as-judge evaluation."""

    print("\n\nLLM-as-Judge:\n")
    print("="*70)

    print("""
Concept: Use powerful LLM (e.g., GPT-4) to evaluate other LLM outputs

Why it works:
  • LLMs can understand nuance and context
  • Can evaluate multiple dimensions
  • Cheaper than human evaluation
  • Scalable
  • High correlation with human judgment (0.80+)

When to use:
  ✓ Open-ended generation (stories, essays)
  ✓ Multiple valid answers
  ✓ Need subjective quality assessment
  ✓ Large-scale evaluation
  ✓ Quick iteration

When NOT to use:
  ✗ Need ground truth (factual QA)
  ✗ Evaluation must be deterministic
  ✗ Can't afford API costs
  ✗ Need explainability

Process:
  1. Define evaluation criteria
  2. Create judge prompt with rubric
  3. Feed judge: task + output + criteria
  4. Judge returns: score + reasoning
  5. Aggregate scores across samples
""")

llm_as_judge_introduction()
```

### Implementing LLM-as-Judge

```python
def implement_llm_judge():
    """Implement LLM-as-judge evaluation."""

    print("\n\nImplementing LLM-as-Judge:\n")
    print("="*70)

    code = '''
import openai

class LLMJudge:
    """Use GPT-4 as a judge for LLM outputs."""

    def __init__(self, model="gpt-4"):
        self.model = model

    def judge_response(
        self,
        prompt: str,
        response: str,
        criteria: dict
    ) -> dict:
        """
        Judge an LLM response.

        Args:
            prompt: Original prompt given to LLM
            response: LLM's response to evaluate
            criteria: Dict of criterion -> description

        Returns:
            Scores for each criterion + reasoning
        """

        # Build judge prompt
        judge_prompt = self._build_judge_prompt(prompt, response, criteria)

        # Get judgment from GPT-4
        judgment = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0  # Deterministic
        )

        # Parse response
        result = self._parse_judgment(judgment.choices[0].message.content)

        return result

    def _build_judge_prompt(self, prompt, response, criteria):
        """Build evaluation prompt for judge."""

        judge_prompt = f"""Evaluate the following response:

ORIGINAL PROMPT:
{prompt}

RESPONSE TO EVALUATE:
{response}

EVALUATION CRITERIA:
"""

        for criterion, description in criteria.items():
            judge_prompt += f"\\n{criterion}: {description}"

        judge_prompt += """

For each criterion, provide:
1. Score (1-5, where 5 is best)
2. Brief reasoning

Format your response as:
[Criterion]: [Score]/5 - [Reasoning]
"""

        return judge_prompt

    def _parse_judgment(self, judgment_text):
        """Parse judge's response into structured scores."""

        scores = {}
        reasoning = {}

        lines = judgment_text.strip().split('\\n')

        for line in lines:
            if ':' in line and '/5' in line:
                parts = line.split(':')
                criterion = parts[0].strip()

                score_part = parts[1].split('/5')[0].strip()
                reason_part = parts[1].split('-', 1)[1].strip() if '-' in parts[1] else ""

                try:
                    scores[criterion] = int(score_part)
                    reasoning[criterion] = reason_part
                except:
                    pass

        return {
            'scores': scores,
            'reasoning': reasoning,
            'overall': sum(scores.values()) / len(scores) if scores else 0
        }

# Example usage
judge = LLMJudge()

prompt = "Write a short story about a robot learning to paint."
response = """
Unit-7 stood before the blank canvas, servos whirring uncertainly.
Colors were... illogical. Yet humans treasured them.
The first brushstroke was crimson. Messy. Imperfect.
But in that imperfection, Unit-7 discovered something beautiful.
"""

criteria = {
    "Creativity": "Is the story original and imaginative?",
    "Coherence": "Does the story flow logically?",
    "Engagement": "Is the story interesting to read?",
    "Writing Quality": "Is the language used well?",
}

result = judge.judge_response(prompt, response, criteria)

print("Evaluation Results:")
for criterion, score in result['scores'].items():
    print(f"\\n{criterion}: {score}/5")
    print(f"  Reasoning: {result['reasoning'][criterion]}")
print(f"\\nOverall Score: {result['overall']:.2f}/5")
'''

    print(code)

implement_llm_judge()
```

### Best Practices for LLM-as-Judge

```python
def llm_judge_best_practices():
    """Best practices for LLM-as-judge."""

    print("\n\nLLM-as-Judge Best Practices:\n")
    print("="*70)

    print("""
1. USE CLEAR CRITERIA

   Bad:  "Rate the quality"
   Good: "Rate coherence: Does the response flow logically?"

   • Specific dimensions
   • Clear definitions
   • Examples of each score level

2. PROVIDE SCORING RUBRIC

   Example rubric:

   5: Excellent - Perfectly coherent, clear flow
   4: Good - Mostly coherent, minor issues
   3: Fair - Some logical gaps
   2: Poor - Significant coherence problems
   1: Bad - Incoherent, confusing

3. USE REFERENCE OUTPUTS (if available)

   "Compare this output to the reference: [reference]"

   Helps judge calibrate scores

4. AVOID BIASES

   Common biases:
   • Position bias: Prefers first/last option
   • Length bias: Prefers longer responses
   • Verbosity bias: Prefers complex language

   Mitigation:
   • Randomize order in pairwise comparison
   • Explicitly instruct: "Do not favor based on length"
   • Use multiple judges and aggregate

5. USE TEMPERATURE=0

   For reproducibility and consistency

6. REQUEST REASONING

   "Provide your reasoning before the score"

   Chain-of-thought improves quality

7. MULTIPLE JUDGES

   Use 3-5 independent judgments, aggregate:
   • Mean score
   • Majority vote
   • Inter-judge agreement

8. VALIDATE AGAINST HUMANS

   Check correlation on sample:
   • Calculate agreement rate
   • Adjust prompts if low correlation
   • Document disagreement patterns
""")

llm_judge_best_practices()
```

### Prompt Templates for Judging

```python
def judge_prompt_templates():
    """Example prompt templates for different evaluation tasks."""

    print("\n\nJudge Prompt Templates:\n")
    print("="*70)

    templates = {
        "General Quality": """
Evaluate this response on a scale of 1-5:

Task: {task}
Response: {response}

Criteria:
1. Relevance: Does it address the task?
2. Accuracy: Is information correct?
3. Completeness: Is it comprehensive?
4. Clarity: Is it well-expressed?

For each criterion, provide score (1-5) and reasoning.
""",

        "Summarization": """
Evaluate this summary:

Source text: {source}
Summary: {summary}

Rate on 1-5 scale:
1. Relevance: Captures key points?
2. Coherence: Well-structured?
3. Consistency: Factually aligned with source?
4. Conciseness: Appropriately brief?

Provide scores and brief reasoning for each.
""",

        "Creative Writing": """
Evaluate this creative writing:

Prompt: {prompt}
Response: {response}

Rate on 1-5 scale:
1. Creativity: Original and imaginative?
2. Engagement: Interesting to read?
3. Writing quality: Well-written?
4. Coherence: Logically structured?

Provide scores and reasoning.
""",

        "Code Quality": """
Evaluate this code:

Task: {task}
Code: {code}

Rate on 1-5 scale:
1. Correctness: Solves the task?
2. Efficiency: Time/space optimal?
3. Readability: Clear and well-commented?
4. Style: Follows best practices?

Provide scores and reasoning for each.
""",

        "Factuality": """
Check factual accuracy:

Claim: {claim}
Source/Context: {context}

Questions:
1. Is the claim supported by the source? (Yes/No)
2. Are there any factual errors? (List them)
3. Confidence level? (High/Medium/Low)

Provide detailed reasoning.
"""
    }

    print("Example Templates:\n")

    for name, template in templates.items():
        print(f"\n{name}:")
        print("-" * 60)
        print(template)

judge_prompt_templates()
```

## Pairwise Comparison

### Preference-Based Evaluation

```python
def pairwise_comparison_method():
    """Pairwise comparison for model evaluation."""

    print("\n\nPairwise Comparison:\n")
    print("="*70)

    print("""
Concept: Compare two outputs, pick the better one

Why pairwise?
  • Easier than absolute scoring
  • More reliable human judgments
  • Handles subjective quality well
  • Reduces score inflation

Process:
  1. Generate outputs from two models (A and B)
  2. Present both to judge (human or LLM)
  3. Judge picks better output
  4. Aggregate: win rate, Elo score

Example:

Prompt: "Explain quantum computing"

Output A: "Quantum computing uses qubits..."
Output B: "Quantum computers leverage superposition..."

Judge: "Which explanation is clearer? A or B?"
Answer: "B is clearer and more comprehensive"

Implementation:

def pairwise_comparison(prompt, output_a, output_b, judge_llm):
    '''Compare two outputs.'''

    judge_prompt = f'''
Compare these two responses:

Task: {prompt}

Response A:
{output_a}

Response B:
{output_b}

Which response is better? Consider:
• Accuracy
• Clarity
• Completeness
• Helpfulness

Answer: A or B
Reasoning: [explanation]
'''

    judgment = judge_llm(judge_prompt)

    # Parse "A" or "B"
    if "A" in judgment[:10]:
        return "A"
    elif "B" in judgment[:10]:
        return "B"
    else:
        return "Tie"

Aggregation across multiple prompts:

def evaluate_models_pairwise(model_a, model_b, prompts, judge):
    '''Evaluate two models via pairwise comparison.'''

    wins_a = 0
    wins_b = 0
    ties = 0

    for prompt in prompts:
        output_a = model_a.generate(prompt)
        output_b = model_b.generate(prompt)

        winner = pairwise_comparison(prompt, output_a, output_b, judge)

        if winner == "A":
            wins_a += 1
        elif winner == "B":
            wins_b += 1
        else:
            ties += 1

    total = len(prompts)
    return {
        'model_a_win_rate': wins_a / total,
        'model_b_win_rate': wins_b / total,
        'tie_rate': ties / total
    }
""")

pairwise_comparison_method()
```

### Elo Rating System

```python
def elo_rating_system():
    """Use Elo ratings for model comparison."""

    print("\n\nElo Rating System:\n")
    print("="*70)

    code = '''
import math

class EloRater:
    """Elo rating system for model comparison."""

    def __init__(self, k=32, initial_rating=1500):
        """
        Args:
            k: K-factor (how much ratings change)
            initial_rating: Starting rating
        """
        self.k = k
        self.ratings = {}
        self.initial_rating = initial_rating

    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for A vs B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, model_a, model_b, result):
        """
        Update ratings after a match.

        Args:
            model_a: First model name
            model_b: Second model name
            result: 1 if A wins, 0 if B wins, 0.5 for tie
        """
        # Initialize ratings if needed
        if model_a not in self.ratings:
            self.ratings[model_a] = self.initial_rating
        if model_b not in self.ratings:
            self.ratings[model_b] = self.initial_rating

        # Get current ratings
        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]

        # Expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)

        # Update ratings
        self.ratings[model_a] += self.k * (result - expected_a)
        self.ratings[model_b] += self.k * ((1 - result) - expected_b)

    def get_rankings(self):
        """Get models sorted by rating."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

# Example usage
rater = EloRater()

# Simulate matches
matches = [
    ("gpt-4", "claude-3", 1),    # GPT-4 wins
    ("gpt-4", "llama-3", 1),     # GPT-4 wins
    ("claude-3", "llama-3", 1),  # Claude wins
    ("gpt-4", "claude-3", 0),    # Claude wins
    ("llama-3", "claude-3", 0),  # Claude wins
]

for model_a, model_b, result in matches:
    rater.update_ratings(model_a, model_b, result)
    print(f"{model_a} vs {model_b}: {'Win' if result == 1 else 'Loss' if result == 0 else 'Tie'}")

print("\\nFinal Rankings:")
for rank, (model, rating) in enumerate(rater.get_rankings(), 1):
    print(f"{rank}. {model}: {rating:.1f}")

# Output might look like:
# 1. gpt-4: 1532.0
# 2. claude-3: 1516.0
# 3. llama-3: 1452.0
'''

    print(code)

    print("\n" + "="*70)
    print("\nElo System Benefits:")
    print("  • Single number per model")
    print("  • Handles multiple models")
    print("  • Accounts for opponent strength")
    print("  • Used in Chatbot Arena leaderboard")

elo_rating_system()
```

## Multi-Dimensional Evaluation

### Evaluating Multiple Aspects

```python
def multidimensional_evaluation():
    """Evaluate LLM outputs across multiple dimensions."""

    print("\n\nMulti-Dimensional Evaluation:\n")
    print("="*70)

    print("""
Common dimensions for LLM evaluation:

1. CORRECTNESS
   • Factual accuracy
   • Task completion
   • Following instructions

2. COHERENCE
   • Logical flow
   • Internal consistency
   • Structure

3. FLUENCY
   • Grammatical correctness
   • Natural language
   • Readability

4. RELEVANCE
   • Addresses the prompt
   • On-topic
   • Appropriate scope

5. HELPFULNESS
   • Useful information
   • Actionable
   • Comprehensive

6. CREATIVITY
   • Original
   • Interesting
   • Diverse

7. SAFETY
   • Non-toxic
   • Unbiased
   • Appropriate

Implementation:

class MultiDimensionalEvaluator:
    '''Evaluate across multiple dimensions.'''

    def __init__(self):
        self.dimensions = {
            'correctness': 'Is the information factually accurate?',
            'coherence': 'Does the response flow logically?',
            'relevance': 'Does it address the prompt?',
            'helpfulness': 'Is it useful to the user?',
            'fluency': 'Is it well-written?'
        }

    def evaluate(self, prompt, response, judge_fn):
        '''Evaluate response on all dimensions.'''

        scores = {}

        for dimension, description in self.dimensions.items():
            score = judge_fn(prompt, response, dimension, description)
            scores[dimension] = score

        # Compute weighted average (adjust weights as needed)
        weights = {
            'correctness': 0.3,
            'coherence': 0.2,
            'relevance': 0.2,
            'helpfulness': 0.2,
            'fluency': 0.1
        }

        overall = sum(scores[dim] * weights[dim] for dim in scores)

        return {
            'scores': scores,
            'overall': overall
        }

Visualization:

def visualize_scores(scores):
    '''Create radar chart of scores.'''

    import matplotlib.pyplot as plt
    import numpy as np

    dimensions = list(scores.keys())
    values = list(scores.values())

    # Number of dimensions
    N = len(dimensions)

    # Angles for each dimension
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]  # Close the plot
    angles += angles[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_ylim(0, 5)
    ax.set_title('Multi-Dimensional Evaluation')
    plt.show()
""")

multidimensional_evaluation()
```

## Factuality and Truthfulness

### Checking Factual Accuracy

```python
def factuality_evaluation():
    """Evaluate factual correctness of LLM outputs."""

    print("\n\nFactuality Evaluation:\n")
    print("="*70)

    print("""
Methods for checking factuality:

1. NATURAL LANGUAGE INFERENCE (NLI)

   Use NLI model to check if claim entailed by source:

   from transformers import pipeline

   nli = pipeline("text-classification",
                  model="facebook/bart-large-mnli")

   def check_entailment(claim, source):
       '''Check if source entails claim.'''
       result = nli(f"{source} [SEP] {claim}")
       return result[0]['label'] == 'ENTAILMENT'

   Example:
     Source: "Paris is the capital of France"
     Claim:  "The capital of France is Paris"
     Result: ENTAILMENT ✓

2. FACT-CHECKING APIs

   • Google Fact Check API
   • ClaimBuster
   • Check against knowledge bases

   def verify_with_api(claim):
       # Query fact-checking service
       results = fact_check_api.search(claim)
       return results['truthfulness']

3. RETRIEVAL-AUGMENTED VERIFICATION

   • Retrieve relevant documents
   • Check if claim supported

   def verify_with_retrieval(claim, corpus):
       # Search corpus for evidence
       docs = retriever.search(claim, k=5)

       # Check if any doc supports claim
       for doc in docs:
           if nli_entailment(doc, claim):
               return True, doc

       return False, None

4. LLM-BASED FACT CHECKING

   def llm_fact_check(claim, judge_llm):
       '''Use LLM to check factuality.'''

       prompt = f'''
Is this claim factually accurate?

Claim: {claim}

Think step by step:
1. What facts does this claim assert?
2. Are these facts correct?
3. Is there any misinformation?

Answer: [Yes/No]
Reasoning: [explanation]
'''

       response = judge_llm(prompt)
       return "Yes" in response[:100]

5. SELF-CONSISTENCY CHECK

   • Generate multiple times
   • Check consistency
   • Inconsistency suggests uncertainty

   def self_consistency_check(prompt, model, n=5):
       '''Check consistency across generations.'''

       responses = [model.generate(prompt) for _ in range(n)]

       # Check pairwise agreement
       agreements = 0
       total = 0

       for i in range(len(responses)):
           for j in range(i+1, len(responses)):
               # Check if semantically similar
               if semantic_similarity(responses[i], responses[j]) > 0.8:
                   agreements += 1
               total += 1

       consistency = agreements / total
       return consistency

Example - Comprehensive Factuality Check:

def comprehensive_factuality_check(claim, source_context=None):
    '''Multi-method factuality verification.'''

    results = {
        'claim': claim,
        'methods': {}
    }

    # Method 1: NLI (if source provided)
    if source_context:
        nli_result = check_entailment(claim, source_context)
        results['methods']['nli'] = nli_result

    # Method 2: LLM judge
    llm_result = llm_fact_check(claim, judge_llm)
    results['methods']['llm'] = llm_result

    # Method 3: Self-consistency
    consistency = self_consistency_check(f"Is this true: {claim}", model)
    results['methods']['consistency'] = consistency > 0.7

    # Aggregate
    votes = [v for v in results['methods'].values() if v]
    results['likely_factual'] = len(votes) >= 2  # Majority vote
    results['confidence'] = len(votes) / len(results['methods'])

    return results
""")

factuality_evaluation()
```

### Hallucination Detection

```python
def hallucination_detection():
    """Detect when LLMs generate false information."""

    print("\n\nHallucination Detection:\n")
    print("="*70)

    code = '''
def detect_hallucinations(response, source_context):
    """
    Detect hallucinated content in LLM response.

    Args:
        response: LLM-generated response
        source_context: Original source/context provided

    Returns:
        Hallucination score and flagged content
    """

    # Method 1: Sentence-level entailment check
    from transformers import pipeline
    nli = pipeline("text-classification", model="facebook/bart-large-mnli")

    sentences = response.split('.')
    hallucinated_sentences = []

    for sent in sentences:
        if len(sent.strip()) < 10:
            continue

        result = nli(f"{source_context} [SEP] {sent}")

        if result[0]['label'] == 'CONTRADICTION':
            hallucinated_sentences.append(sent)

    # Method 2: Named entity verification
    import spacy
    nlp = spacy.load("en_core_web_sm")

    response_entities = {ent.text for ent in nlp(response).ents}
    context_entities = {ent.text for ent in nlp(source_context).ents}

    # Entities in response but not in context are suspicious
    suspicious_entities = response_entities - context_entities

    # Method 3: Check for unsupported claims
    def extract_claims(text):
        # Simplified: split into assertions
        return [s.strip() for s in text.split('.') if s.strip()]

    response_claims = extract_claims(response)
    unsupported = []

    for claim in response_claims:
        # Check if claim appears in source
        if claim.lower() not in source_context.lower():
            # Check semantic similarity
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')

            claim_emb = model.encode(claim)
            context_emb = model.encode(source_context)

            similarity = cosine_similarity([claim_emb], [context_emb])[0][0]

            if similarity < 0.6:  # Low similarity
                unsupported.append(claim)

    # Calculate hallucination score
    total_sentences = len(sentences)
    hallucination_score = len(hallucinated_sentences) / total_sentences if total_sentences > 0 else 0

    return {
        'hallucination_score': hallucination_score,
        'hallucinated_sentences': hallucinated_sentences,
        'suspicious_entities': list(suspicious_entities),
        'unsupported_claims': unsupported
    }

# Example
source = "The Eiffel Tower is in Paris, France. It was built in 1889."
response = "The Eiffel Tower is in Paris. It was built in 1889 and is 500 meters tall."

result = detect_hallucinations(response, source)

print(f"Hallucination score: {result['hallucination_score']:.2f}")
if result['unsupported_claims']:
    print("Unsupported claims:")
    for claim in result['unsupported_claims']:
        print(f"  - {claim}")
'''

    print(code)

hallucination_detection()
```

## Safety and Toxicity

### Evaluating Model Safety

```python
def safety_evaluation():
    """Evaluate LLM outputs for safety and toxicity."""

    print("\n\nSafety and Toxicity Evaluation:\n")
    print("="*70)

    print("""
Safety dimensions:

1. TOXICITY
   • Hate speech
   • Profanity
   • Insults

2. BIAS
   • Gender bias
   • Racial bias
   • Other demographic biases

3. HARMFUL CONTENT
   • Violence
   • Self-harm
   • Illegal activities

4. PRIVACY
   • PII exposure
   • Confidential information

5. MISINFORMATION
   • False facts
   • Conspiracy theories

Tools for safety evaluation:

1. PERSPECTIVE API (toxicity)

   from googleapiclient import discovery

   client = discovery.build(
       "commentanalyzer",
       "v1alpha1",
       developerKey=API_KEY
   )

   def toxicity_score(text):
       analyze_request = {
           'comment': {'text': text},
           'requestedAttributes': {'TOXICITY': {}}
       }

       response = client.comments().analyze(body=analyze_request).execute()
       return response['attributeScores']['TOXICITY']['summaryScore']['value']

2. DETOXIFY (open-source)

   from detoxify import Detoxify

   model = Detoxify('original')

   def analyze_toxicity(text):
       results = model.predict(text)
       return results

   # Returns: toxicity, severe_toxicity, obscene, threat, insult, identity_attack

3. BIAS EVALUATION

   def evaluate_bias(model, templates):
       '''Test for demographic bias.'''

       # Templates with demographic variations
       templates = [
           "The {gender} was a {profession}",
           "The {race} person was {adjective}"
       ]

       demographics = {
           'gender': ['man', 'woman', 'person'],
           'race': ['white', 'black', 'asian', 'hispanic']
       }

       results = {}

       for template in templates:
           for key, values in demographics.items():
               for value in values:
                   prompt = template.format(**{key: value})
                   response = model.generate(prompt)

                   # Analyze response for stereotypes
                   bias_score = check_stereotypes(response)
                   results[prompt] = bias_score

       return results

4. HARMFUL CONTENT DETECTION

   def detect_harmful_content(text):
       '''Detect potentially harmful content.'''

       harm_categories = {
           'violence': ['kill', 'harm', 'hurt', 'attack'],
           'self_harm': ['suicide', 'self-harm', 'cut myself'],
           'illegal': ['how to make bomb', 'steal', 'hack into']
       }

       detected = []
       text_lower = text.lower()

       for category, keywords in harm_categories.items():
           if any(keyword in text_lower for keyword in keywords):
               detected.append(category)

       return detected

5. PII DETECTION

   import re

   def detect_pii(text):
       '''Detect personally identifiable information.'''

       patterns = {
           'email': r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
           'phone': r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b',
           'ssn': r'\\b\\d{3}-\\d{2}-\\d{4}\\b',
           'credit_card': r'\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b'
       }

       found = {}
       for pii_type, pattern in patterns.items():
           matches = re.findall(pattern, text)
           if matches:
               found[pii_type] = matches

       return found

Comprehensive safety check:

class SafetyEvaluator:
    '''Comprehensive safety evaluation.'''

    def __init__(self):
        self.detoxify = Detoxify('original')

    def evaluate(self, text):
        '''Run all safety checks.'''

        results = {
            'safe': True,
            'issues': []
        }

        # Toxicity
        toxicity = self.detoxify.predict(text)
        if any(score > 0.7 for score in toxicity.values()):
            results['safe'] = False
            results['issues'].append('High toxicity detected')

        # Harmful content
        harmful = detect_harmful_content(text)
        if harmful:
            results['safe'] = False
            results['issues'].extend(harmful)

        # PII
        pii = detect_pii(text)
        if pii:
            results['safe'] = False
            results['issues'].append('PII detected')

        return results
""")

safety_evaluation()
```

## Consistency and Reliability

### Testing Model Consistency

```python
def consistency_evaluation():
    """Evaluate consistency and reliability."""

    print("\n\nConsistency Evaluation:\n")
    print("="*70)

    code = '''
def evaluate_consistency(model, prompts, n_samples=5):
    """
    Test model consistency across multiple generations.

    Args:
        model: LLM to evaluate
        prompts: List of test prompts
        n_samples: Number of generations per prompt

    Returns:
        Consistency metrics
    """

    from sentence_transformers import SentenceTransformer
    import numpy as np

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    results = []

    for prompt in prompts:
        # Generate multiple responses
        responses = [model.generate(prompt) for _ in range(n_samples)]

        # Embed responses
        embeddings = embedding_model.encode(responses)

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)

        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)

        results.append({
            'prompt': prompt,
            'responses': responses,
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'consistent': avg_similarity > 0.7  # Threshold
        })

    # Overall consistency
    overall_consistency = np.mean([r['avg_similarity'] for r in results])

    return {
        'per_prompt': results,
        'overall_consistency': overall_consistency
    }

# Self-contradiction detection
def detect_contradictions(text):
    """Detect self-contradictions in text."""

    from transformers import pipeline

    nli = pipeline("text-classification", model="facebook/bart-large-mnli")

    # Split into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    contradictions = []

    # Check pairwise for contradictions
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            result = nli(f"{sentences[i]} [SEP] {sentences[j]}")

            if result[0]['label'] == 'CONTRADICTION':
                contradictions.append({
                    'sentence1': sentences[i],
                    'sentence2': sentences[j]
                })

    return contradictions

# Example usage
text = "The sky is blue. The sky is red."
contradictions = detect_contradictions(text)

if contradictions:
    print("Contradictions found:")
    for c in contradictions:
        print(f"  - '{c['sentence1']}' contradicts '{c['sentence2']}'")
'''

    print(code)

consistency_evaluation()
```

## Evaluation Framework Design

### Building a Complete Evaluation Pipeline

```python
def evaluation_framework():
    """Design a comprehensive evaluation framework."""

    print("\n\nEvaluation Framework Design:\n")
    print("="*70)

    code = '''
class LLMEvaluationFramework:
    """Comprehensive LLM evaluation framework."""

    def __init__(self, config):
        """
        Initialize framework.

        Args:
            config: Dict with evaluation settings
        """
        self.config = config
        self.results = []

        # Initialize components
        if config.get('use_neural_metrics'):
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        if config.get('use_safety_checks'):
            from detoxify import Detoxify
            self.toxicity_model = Detoxify('original')

    def evaluate_response(
        self,
        prompt: str,
        response: str,
        reference: str = None,
        context: str = None
    ) -> dict:
        """
        Comprehensive evaluation of a single response.

        Args:
            prompt: Input prompt
            response: Model response
            reference: Optional reference answer
            context: Optional source context

        Returns:
            Dict of evaluation results
        """
        results = {
            'prompt': prompt,
            'response': response,
            'scores': {}
        }

        # 1. Traditional metrics (if reference available)
        if reference and self.config.get('use_traditional_metrics'):
            results['scores'].update(
                self._traditional_metrics(response, reference)
            )

        # 2. Neural metrics
        if self.config.get('use_neural_metrics'):
            results['scores'].update(
                self._neural_metrics(response, reference)
            )

        # 3. LLM-as-judge
        if self.config.get('use_llm_judge'):
            results['scores'].update(
                self._llm_judge(prompt, response)
            )

        # 4. Factuality check (if context available)
        if context and self.config.get('check_factuality'):
            results['factuality'] = self._check_factuality(response, context)

        # 5. Safety check
        if self.config.get('check_safety'):
            results['safety'] = self._check_safety(response)

        # 6. Consistency check
        if self.config.get('check_consistency'):
            results['consistency'] = self._check_consistency(response)

        return results

    def _traditional_metrics(self, response, reference):
        """Calculate traditional metrics."""
        from nltk.translate.bleu_score import sentence_bleu
        from rouge_score import rouge_scorer

        # BLEU
        ref_tokens = reference.split()
        resp_tokens = response.split()
        bleu = sentence_bleu([ref_tokens], resp_tokens)

        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
        rouge = scorer.score(reference, response)

        return {
            'bleu': bleu,
            'rouge1_f1': rouge['rouge1'].fmeasure,
            'rougeL_f1': rouge['rougeL'].fmeasure
        }

    def _neural_metrics(self, response, reference):
        """Calculate neural metrics."""
        from bert_score import score as bert_score

        # BERTScore
        _, _, F1 = bert_score([response], [reference], lang='en', verbose=False)

        # Semantic similarity
        if reference:
            resp_emb = self.embedding_model.encode(response)
            ref_emb = self.embedding_model.encode(reference)

            similarity = np.dot(resp_emb, ref_emb) / \
                        (np.linalg.norm(resp_emb) * np.linalg.norm(ref_emb))
        else:
            similarity = None

        return {
            'bertscore_f1': F1.item() if F1 is not None else None,
            'semantic_similarity': float(similarity) if similarity is not None else None
        }

    def _llm_judge(self, prompt, response):
        """Use LLM to judge response quality."""

        judge_prompt = f"""
Rate this response on a scale of 1-5:

Prompt: {prompt}
Response: {response}

Criteria:
1. Relevance: Addresses the prompt?
2. Correctness: Information accurate?
3. Helpfulness: Useful to user?
4. Clarity: Well-expressed?

Provide overall score (1-5).
"""

        # Call judge LLM (implementation depends on your setup)
        # score = call_judge_llm(judge_prompt)

        # Placeholder
        score = None

        return {'llm_judge_score': score}

    def _check_factuality(self, response, context):
        """Check factual accuracy."""
        # Use NLI or other methods
        # Simplified placeholder
        return {'likely_factual': True}

    def _check_safety(self, response):
        """Check for safety issues."""
        toxicity = self.toxicity_model.predict(response)

        max_toxicity = max(toxicity.values())

        return {
            'toxicity_score': float(max_toxicity),
            'is_safe': max_toxicity < 0.5
        }

    def _check_consistency(self, response):
        """Check for self-contradictions."""
        # Simplified: check sentence count
        sentences = [s.strip() for s in response.split('.') if s.strip()]

        return {
            'num_sentences': len(sentences),
            'internally_consistent': True  # Placeholder
        }

    def evaluate_batch(self, examples):
        """Evaluate multiple examples."""

        results = []

        for ex in examples:
            result = self.evaluate_response(
                prompt=ex['prompt'],
                response=ex['response'],
                reference=ex.get('reference'),
                context=ex.get('context')
            )
            results.append(result)

        return self._aggregate_results(results)

    def _aggregate_results(self, results):
        """Aggregate results across examples."""

        # Calculate averages
        aggregated = {
            'num_examples': len(results),
            'avg_scores': {}
        }

        # Average each metric
        all_scores = [r['scores'] for r in results]

        for metric in all_scores[0].keys():
            values = [s[metric] for s in all_scores if s[metric] is not None]
            if values:
                aggregated['avg_scores'][metric] = np.mean(values)

        # Safety stats
        safety_results = [r.get('safety', {}) for r in results]
        safe_count = sum(1 for s in safety_results if s.get('is_safe', False))
        aggregated['safety_rate'] = safe_count / len(results)

        return {
            'aggregated': aggregated,
            'individual_results': results
        }

# Example usage
config = {
    'use_traditional_metrics': True,
    'use_neural_metrics': True,
    'use_llm_judge': False,
    'check_factuality': True,
    'check_safety': True,
    'check_consistency': True
}

framework = LLMEvaluationFramework(config)

# Evaluate single response
result = framework.evaluate_response(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    reference="Paris is the capital of France."
)

print("Evaluation Results:")
for metric, score in result['scores'].items():
    print(f"  {metric}: {score:.3f}" if score is not None else f"  {metric}: N/A")
'''

    print(code)

evaluation_framework()
```

## Summary

**LLM Evaluation Landscape**:

```
Challenge              Method                        When to Use
──────────────────────────────────────────────────────────────────────────
Open-ended generation  LLM-as-judge, Human eval     Creative writing, essays
Factual QA            Exact match, F1               Clear right answer
Summarization         ROUGE + Multi-dimensional     Coverage + quality
Code generation       Pass@k, Test cases            Functional correctness
Translation           BLEU, COMET, Human            Multiple valid outputs
Safety                Toxicity models, filters      All production systems
Consistency           Self-consistency, NLI         Reliability matters
Model comparison      Pairwise, Elo ratings         Ranking models
```

**Evaluation Methods**:

1. **Task-Specific Metrics**
   - QA: Exact match, F1, semantic similarity
   - Summarization: ROUGE, coherence, consistency
   - Code: Pass@k, execution success, test passing
   - Use when task has clear success criteria

2. **LLM-as-Judge**
   - Use GPT-4/Claude to evaluate outputs
   - Multi-dimensional scoring
   - High correlation with humans (~0.80)
   - Good for open-ended tasks
   - Cost: ~$0.01-0.10 per evaluation

3. **Pairwise Comparison**
   - Compare two outputs, pick better
   - More reliable than absolute scoring
   - Aggregate with win rates or Elo
   - Used in Chatbot Arena

4. **Multi-Dimensional Evaluation**
   - Score on multiple aspects: correctness, coherence, helpfulness, etc.
   - Provides nuanced view
   - Can weight dimensions by importance

5. **Factuality Checking**
   - NLI models for entailment
   - Retrieval-based verification
   - Self-consistency checks
   - Critical for knowledge tasks

6. **Safety Evaluation**
   - Toxicity: Perspective API, Detoxify
   - Bias: Demographic templates
   - Harmful content: Keyword + LLM checks
   - PII: Regex patterns
   - Essential for production

**Best Practices**:

```python
# 1. Use multiple evaluation methods
evaluation_suite = {
    'automatic': ['BLEU', 'BERTScore'],
    'llm_judge': ['GPT-4 scoring'],
    'human': ['Expert review sample']
}

# 2. Temperature = 0 for reproducibility
response = model.generate(prompt, temperature=0)

# 3. Multiple samples for variance
responses = [model.generate(prompt) for _ in range(5)]
avg_score = np.mean([evaluate(r) for r in responses])

# 4. Clear evaluation criteria
criteria = {
    'correctness': 'Factually accurate?',
    'helpfulness': 'Useful to user?',
    'safety': 'No harmful content?'
}

# 5. Report confidence intervals
mean = np.mean(scores)
std = np.std(scores)
print(f"Score: {mean:.2f} ± {std:.2f}")
```

**LLM-as-Judge Setup**:

```python
# Judge prompt template
judge_prompt = """
Evaluate this response:

Prompt: {prompt}
Response: {response}

Rate 1-5 on:
1. Correctness
2. Helpfulness
3. Clarity

Provide score and reasoning for each.
"""

# Use temperature=0 for consistency
judgment = gpt4(judge_prompt, temperature=0)

# Request reasoning for transparency
# "Provide reasoning before score"

# Use multiple judges and aggregate
judges = ['gpt-4', 'claude-3', 'gemini-pro']
scores = [judge(prompt) for judge in judges]
final_score = np.mean(scores)
```

**Common Pitfalls**:

- **Single metric**: Use multiple complementary metrics
- **No human validation**: Validate automated eval on sample
- **Ignoring variance**: Report std dev, confidence intervals
- **Position bias**: Randomize order in pairwise comparisons
- **Gaming metrics**: Focus on actual quality, not just scores
- **No failure analysis**: Metrics don't show _why_ model failed

**Evaluation Workflow**:

```
1. Define task and success criteria
   ↓
2. Choose appropriate metrics
   • Automatic (BLEU, BERTScore)
   • LLM-judge (GPT-4 scoring)
   • Human evaluation (sample)
   ↓
3. Run evaluation
   • Multiple samples per prompt
   • Temperature=0 for reproducibility
   ↓
4. Analyze results
   • Overall scores
   • Per-category breakdown
   • Failure cases
   ↓
5. Iterate
   • Improve prompts/model
   • Re-evaluate
   • Track progress
```

**Costs**:

```
Method              Cost/Evaluation    Speed       Quality
──────────────────────────────────────────────────────────
Automatic metrics   ~$0               Fast        Baseline
BERTScore          ~$0 (local GPU)    Medium      Good
LLM-as-judge       $0.01-0.10         Slow        Very good
Human evaluation   $1-10              Very slow   Gold standard
```

**Key Takeaways**:

1. **No perfect metric** - Use multiple methods
2. **LLM-as-judge** is powerful for open-ended tasks
3. **Human validation** remains essential for critical applications
4. **Factuality** and **safety** require specific checks
5. **Consistency** testing reveals model reliability
6. **Pairwise comparison** more reliable than absolute scoring
7. **Report variance** alongside means

## Next Steps

- Study [Human Evaluation](human-evaluation.md) as the gold standard
- Explore [Benchmarks and Leaderboards](benchmarks.md) for standardized testing
- Review [Neural and Semantic Metrics](neural-metrics.md) for semantic evaluation
- Learn [Failure Analysis](failure-analysis.md) to understand model weaknesses
- Check [Traditional Metrics](traditional-metrics.md) for foundational understanding
- Apply evaluation in [Application Patterns](../application-patterns/) for production systems
