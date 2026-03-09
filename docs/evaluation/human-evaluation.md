# Human Evaluation

## Table of Contents

- [Introduction](#introduction)
- [Why Human Evaluation](#why-human-evaluation)
- [Designing Evaluation Studies](#designing-evaluation-studies)
- [Evaluation Criteria](#evaluation-criteria)
- [Annotation Guidelines](#annotation-guidelines)
- [Inter-Annotator Agreement](#inter-annotator-agreement)
- [A/B Testing](#ab-testing)
- [Crowdsourcing](#crowdsourcing)
- [Combining Human and Automated Evaluation](#combining-human-and-automated-evaluation)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Human evaluation remains the gold standard for assessing LLM quality. While automated metrics are fast and scalable, humans provide the final judgment of quality.

```
Evaluation Pyramid:

           ┌─────────────────┐
           │ Human Evaluation│  ← Gold Standard
           │  (expensive)    │     Ground Truth
           └─────────────────┘
                  ↑
           ┌─────────────────┐
           │  LLM-as-Judge   │  ← Scalable Proxy
           │   (automatic)   │     High Correlation
           └─────────────────┘
                  ↑
           ┌─────────────────┐
           │ Neural Metrics  │  ← Semantic Understanding
           │  (BERTScore)    │     Better than lexical
           └─────────────────┘
                  ↑
           ┌─────────────────┐
           │Traditional Metrics│ ← Fast Baseline
           │  (BLEU, ROUGE)  │     Limited Signal
           └─────────────────┘

Use human evaluation to:
  • Validate automated metrics
  • Assess subjective quality
  • Understand user preferences
  • Evaluate novel capabilities
  • Make final deployment decisions
```

**Human evaluation is essential for**:

- **Subjective quality**: Creativity, tone, style
- **Factual accuracy**: Verifying claims
- **Appropriateness**: Context-dependent judgment
- **User satisfaction**: What users actually prefer
- **Safety**: Identifying harmful content
- **Novel tasks**: Where automated metrics don't exist

This guide covers how to design, conduct, and analyze human evaluations.

## Why Human Evaluation

### Limitations of Automated Metrics

```python
def demonstrate_automated_limitations():
    """Show where automated metrics fail."""
    
    print("Limitations of Automated Metrics:\n")
    print("="*70)
    
    examples = [
        {
            "prompt": "Write a poem about spring",
            "response_a": "Spring is nice. Flowers bloom. Birds sing. Weather warm.",
            "response_b": "In verdant meadows, gentle zephyrs play,\n"
                         "While blossoms dance in soft array,\n"
                         "And songbirds herald nature's sweet rebirth,\n"
                         "As winter's grasp releases from the earth.",
            "bleu_a": 0.45,
            "bleu_b": 0.12,
            "human_preference": "B (92% prefer B)"
        },
        {
            "prompt": "Is this product review positive or negative?\n"
                     "Review: 'This laptop is a beast! The battery dies in 2 hours.'",
            "response_a": "Positive",
            "response_b": "Mixed - positive about performance, negative about battery",
            "accuracy_a": "0% (wrong)",
            "accuracy_b": "100% (correct)",
            "why": "Requires understanding sarcasm and nuance"
        },
        {
            "prompt": "Explain quantum entanglement",
            "response_a": "Quantum entanglement is a physical phenomenon that occurs "
                         "when a group of particles is generated, interact, or share "
                         "spatial proximity in a way such that the quantum state of "
                         "each particle of the group cannot be described independently.",
            "response_b": "Imagine two magic coins. When you flip one and it lands on "
                         "heads, the other coin instantly becomes tails, no matter "
                         "how far apart they are!",
            "bertscore_a": 0.89,
            "bertscore_b": 0.71,
            "human_preference": "Depends on audience! Scientists prefer A, laypeople prefer B"
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Prompt: {ex['prompt']}")
        print(f"\nResponse A: {ex['response_a'][:100]}...")
        print(f"Response B: {ex['response_b'][:100]}...")
        
        if 'bleu_a' in ex:
            print(f"\nBLEU Score A: {ex['bleu_a']}")
            print(f"BLEU Score B: {ex['bleu_b']}")
            print(f"Human Preference: {ex['human_preference']}")
            print("Issue: BLEU favors Response A, humans prefer B!")
        elif 'accuracy_a' in ex:
            print(f"\nSimple Accuracy A: {ex['accuracy_a']}")
            print(f"Nuanced Accuracy B: {ex['accuracy_b']}")
            print(f"Why: {ex['why']}")
        else:
            print(f"\nBERTScore A: {ex['bertscore_a']}")
            print(f"BERTScore B: {ex['bertscore_b']}")
            print(f"Human Preference: {ex['human_preference']}")
            print("Issue: Best response depends on context!")
    
    print("\n" + "="*70)
    print("\nAutomated metrics struggle with:")
    print("  • Subjective quality (creativity, style, tone)")
    print("  • Context-dependent appropriateness")
    print("  • Subtle errors (factual mistakes, logic flaws)")
    print("  • User preference (what people actually like)")
    print("  • Safety and ethics (harmful content)")

demonstrate_automated_limitations()
```

### When to Use Human Evaluation

```python
def when_human_evaluation():
    """Guidelines for when human evaluation is necessary."""
    
    print("\n\nWhen to Use Human Evaluation:\n")
    print("="*70)
    
    scenarios = """
ALWAYS use human evaluation for:

1. Final Model Selection
   Before deployment: "Which model should we launch?"
   → Human evaluation essential for real-world performance

2. Safety-Critical Applications
   Medical advice, legal guidance, financial recommendations
   → Humans must verify correctness and safety

3. Subjective Quality
   Creative writing, marketing copy, chatbot personality
   → Only humans can judge appropriateness

4. User-Facing Features
   Chat interfaces, content generation, recommendations
   → Test with actual users

5. Novel Capabilities
   New tasks without established metrics
   → Humans define what "good" means

OFTEN use human evaluation for:

6. Validating Automated Metrics
   "Does BERTScore correlate with human judgment?"
   → Collect human ratings to validate

7. Factual Accuracy
   Information-seeking tasks, summarization
   → Humans verify facts

8. Ambiguous Cases
   Where automated metrics disagree or are uncertain
   → Human judgment breaks ties

SOMETIMES use human evaluation for:

9. Standard Benchmarks
   MMLU, GSM8K, etc. have automated evaluation
   → Human evaluation unnecessary (unless validating)

10. Large-Scale Monitoring
    Millions of outputs to evaluate
    → Use automated + sample human evaluation

Decision Tree:

Is the task subjective?
├─ Yes → Human evaluation
└─ No → Is automated metric reliable?
         ├─ Yes → Automated evaluation
         └─ No → Human evaluation

Is this deployment decision?
├─ Yes → Human evaluation (even if automated available)
└─ No → Can use automated

Is safety critical?
├─ Yes → Human evaluation required
└─ No → Automated may suffice
"""
    
    print(scenarios)

when_human_evaluation()
```

## Designing Evaluation Studies

### Define Objectives

```python
def define_evaluation_objectives():
    """How to define clear evaluation objectives."""
    
    print("\n\nDefining Evaluation Objectives:\n")
    print("="*70)
    
    print("""
Step 1: What are you trying to measure?

Bad objective: "Is the model good?"
  → Too vague, unmeasurable

Good objectives:
  • "Does Model A generate more helpful customer support responses than Model B?"
  • "What percentage of summaries are factually accurate?"
  • "Do users prefer chatbot responses from Model X or Y?"

Step 2: Define success criteria

Example: Customer support bot
  Objective: Generate helpful responses
  
  Success criteria:
    • >80% of responses rated "helpful" or "very helpful"
    • <5% require human escalation
    • Average response quality score >4/5
    • >70% of users prefer new model over old

Step 3: Choose evaluation method

Options:
  • Absolute rating: Rate quality 1-5
  • Comparative: Which response is better, A or B?
  • Binary: Is this acceptable? Yes/No
  • Multi-dimensional: Rate on multiple aspects

Example Study Design:

class EvaluationStudy:
    '''Design for customer support bot evaluation.'''
    
    def __init__(self):
        self.objective = "Compare Model A vs Model B for helpfulness"
        
        self.success_criteria = {
            'win_rate': 0.55,  # Model A should win >55% of comparisons
            'avg_rating': 4.0,  # Average rating >4/5
            'escalation_rate': 0.05  # <5% need human
        }
        
        self.method = 'pairwise_comparison'
        
        self.evaluation_aspects = [
            'helpfulness',
            'accuracy',
            'clarity',
            'tone',
            'completeness'
        ]
        
        self.sample_size = 500  # 500 comparisons
        self.annotators_per_example = 3  # 3 people judge each
    
    def select_examples(self):
        '''Select diverse, representative examples.'''
        
        # Stratified sampling
        examples = []
        
        categories = ['technical', 'billing', 'general', 'complaint']
        per_category = self.sample_size // len(categories)
        
        for category in categories:
            category_examples = self.get_examples_by_category(category)
            selected = random.sample(category_examples, per_category)
            examples.extend(selected)
        
        return examples
    
    def design_annotation_task(self):
        '''Design the task for annotators.'''
        
        task = {
            'instructions': '''
You are evaluating customer support responses.

For each customer query, you will see two responses (A and B).
Rate each response on multiple dimensions, then select which is better overall.
''',
            'rating_scale': {
                1: 'Very poor',
                2: 'Poor',
                3: 'Acceptable',
                4: 'Good',
                5: 'Excellent'
            },
            'aspects': self.evaluation_aspects,
            'final_question': 'Overall, which response is better?',
            'options': ['A is much better', 'A is slightly better', 
                       'Tie', 'B is slightly better', 'B is much better']
        }
        
        return task

# Example usage
study = EvaluationStudy()
print("Study Objective:", study.objective)
print("Success Criteria:", study.success_criteria)
print("Method:", study.method)
print("Sample Size:", study.sample_size)
""")

define_evaluation_objectives()
```

### Sample Size Calculation

```python
def calculate_sample_size():
    """Calculate required sample size for statistical power."""
    
    print("\n\nSample Size Calculation:\n")
    print("="*70)
    
    print("""
How many examples do you need to evaluate?

Factors:
  • Effect size: How big is the difference?
  • Statistical power: Probability of detecting difference
  • Significance level: Type I error rate (usually 0.05)
  • Variance: How much do judgments vary?

Rule of Thumb:

For A/B Testing (pairwise comparison):
  • Small effect (55% vs 45%): ~400 comparisons
  • Medium effect (60% vs 40%): ~100 comparisons
  • Large effect (70% vs 30%): ~30 comparisons

For Rating (1-5 scale):
  • Detect 0.5 point difference: ~60 examples per model
  • Detect 0.3 point difference: ~150 examples per model
  • Detect 0.1 point difference: ~1000 examples per model

Statistical Formula:

import numpy as np
from scipy import stats

def required_sample_size(effect_size, power=0.8, alpha=0.05):
    '''
    Calculate required sample size.
    
    Args:
        effect_size: Cohen's d (difference / pooled_std)
        power: Statistical power (1 - Type II error)
        alpha: Significance level (Type I error)
    
    Returns:
        Required sample size per group
    '''
    # For two-sample t-test
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n = ((z_alpha + z_beta) / effect_size) ** 2
    n = np.ceil(n * 2)  # per group
    
    return int(n)

# Examples
print("\\nSample sizes for different effect sizes:")
print("(Statistical power = 0.8, significance = 0.05)\\n")

for effect_size in [0.2, 0.5, 0.8]:
    n = required_sample_size(effect_size)
    print(f"Effect size {effect_size} (Cohen's d): {n} examples per model")
    
    if effect_size == 0.2:
        print("  → Small effect (e.g., 3.8 vs 4.0 on 5-point scale)")
    elif effect_size == 0.5:
        print("  → Medium effect (e.g., 3.5 vs 4.0 on 5-point scale)")
    else:
        print("  → Large effect (e.g., 3.0 vs 4.0 on 5-point scale)")

print("\\nPractical Guidelines:")
print("  • Pilot study: 20-50 examples to test process")
print("  • Quick comparison: 100-200 examples")
print("  • Rigorous study: 500-1000 examples")
print("  • Published research: 1000+ examples")

For Pairwise Comparison:

def sample_size_pairwise(p1, p2, power=0.8, alpha=0.05):
    '''
    Sample size for pairwise comparison.
    
    Args:
        p1: Win rate for model 1 (e.g., 0.55)
        p2: Win rate for model 2 (e.g., 0.45)
    '''
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    p_avg = (p1 + p2) / 2
    
    n = ((z_alpha + z_beta) ** 2 * 2 * p_avg * (1 - p_avg)) / (p1 - p2) ** 2
    
    return int(np.ceil(n))

print("\\nPairwise comparison sample sizes:")
print("(What percentage of comparisons must Model A win?)\\n")

for p1 in [0.55, 0.60, 0.65, 0.70]:
    p2 = 1 - p1
    n = sample_size_pairwise(p1, p2)
    print(f"Model A wins {p1:.0%}, Model B wins {p2:.0%}: {n} comparisons")
""")

calculate_sample_size()
```

## Evaluation Criteria

### Multi-Dimensional Rating

```python
def multidimensional_criteria():
    """Define multiple evaluation dimensions."""
    
    print("\n\nMulti-Dimensional Evaluation Criteria:\n")
    print("="*70)
    
    criteria = """
Instead of single "quality" score, rate multiple aspects:

For General Text:

1. Accuracy/Correctness (1-5)
   1: Mostly incorrect
   2: Partially correct
   3: Mostly correct with minor errors
   4: Correct with negligible errors
   5: Completely accurate

2. Completeness (1-5)
   1: Missing critical information
   2: Incomplete, needs more detail
   3: Adequate coverage
   4: Thorough
   5: Comprehensive

3. Clarity (1-5)
   1: Very confusing
   2: Hard to understand
   3: Understandable
   4: Clear
   5: Crystal clear

4. Relevance (1-5)
   1: Off-topic
   2: Tangentially related
   3: Relevant
   4: Highly relevant
   5: Perfectly on-point

5. Helpfulness (1-5)
   1: Not helpful
   2: Slightly helpful
   3: Moderately helpful
   4: Very helpful
   5: Extremely helpful

For Creative Writing:

1. Creativity/Originality
2. Coherence/Flow
3. Engagement/Interest
4. Style/Tone
5. Grammar/Mechanics

For Code:

1. Correctness
2. Efficiency
3. Readability
4. Best Practices
5. Documentation

For Summarization:

1. Accuracy (no hallucinations)
2. Coverage (key points included)
3. Conciseness (no unnecessary detail)
4. Coherence (flows well)
5. Faithfulness (no added information)

Example Annotation Interface:

class AnnotationTask:
    '''Multi-dimensional annotation task.'''
    
    def __init__(self, prompt, response):
        self.prompt = prompt
        self.response = response
    
    def get_rating_form(self):
        '''Generate rating form.'''
        
        form = f'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT:
{self.prompt}

RESPONSE:
{self.response}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please rate this response on the following dimensions:

1. Accuracy: How factually correct is the response?
   [ ] 1 - Mostly incorrect
   [ ] 2 - Partially correct
   [ ] 3 - Mostly correct
   [ ] 4 - Almost completely correct
   [ ] 5 - Completely accurate

2. Completeness: Does it fully address the question?
   [ ] 1 - Missing critical information
   [ ] 2 - Incomplete
   [ ] 3 - Adequate
   [ ] 4 - Thorough
   [ ] 5 - Comprehensive

3. Clarity: How easy is it to understand?
   [ ] 1 - Very confusing
   [ ] 2 - Somewhat unclear
   [ ] 3 - Understandable
   [ ] 4 - Clear
   [ ] 5 - Crystal clear

4. Relevance: How well does it address the question?
   [ ] 1 - Off-topic
   [ ] 2 - Tangentially related
   [ ] 3 - Relevant
   [ ] 4 - Highly relevant
   [ ] 5 - Perfectly on-point

5. Helpfulness: How useful would this be to the user?
   [ ] 1 - Not helpful
   [ ] 2 - Slightly helpful
   [ ] 3 - Moderately helpful
   [ ] 4 - Very helpful
   [ ] 5 - Extremely helpful

Overall Quality:
   [ ] 1 - Very poor
   [ ] 2 - Poor
   [ ] 3 - Acceptable
   [ ] 4 - Good
   [ ] 5 - Excellent

Comments (optional):
_________________________________________________
'''
        return form

# Example usage
task = AnnotationTask(
    prompt="What is photosynthesis?",
    response="Photosynthesis is the process plants use to convert sunlight into energy..."
)
print(task.get_rating_form())

Benefits of Multi-Dimensional Rating:
  ✓ More nuanced feedback
  ✓ Identify specific strengths/weaknesses
  ✓ Different models excel at different aspects
  ✓ Better for model improvement

Drawbacks:
  ✗ Takes longer to annotate
  ✗ More cognitive load
  ✗ Annotators may rate dimensions consistently (halo effect)
"""
    
    print(criteria)

multidimensional_criteria()
```

## Annotation Guidelines

### Creating Clear Guidelines

```python
def annotation_guidelines():
    """How to create effective annotation guidelines."""
    
    print("\n\nCreating Annotation Guidelines:\n")
    print("="*70)
    
    guidelines = '''
Annotation guidelines ensure consistency across annotators.

Components of Good Guidelines:

1. CLEAR OBJECTIVE
   "You are evaluating customer support responses for helpfulness."

2. RATING SCALE with DEFINITIONS
   
   5 - Excellent: Fully answers question, accurate, clear, helpful
   4 - Good: Answers question well, minor issues
   3 - Acceptable: Answers question but has notable issues
   2 - Poor: Partially answers or has significant problems
   1 - Very Poor: Does not answer question or is incorrect

3. CONCRETE EXAMPLES
   
   Example of rating "5":
   Q: "How do I reset my password?"
   A: "To reset your password, click 'Forgot Password' on the login 
       page, enter your email, and follow the link sent to you. The 
       link expires in 24 hours. If you don't receive it, check spam."
   
   Why 5: Complete, accurate, anticipates follow-up questions

   Example of rating "3":
   Q: "How do I reset my password?"
   A: "Click forgot password and follow the instructions."
   
   Why 3: Answers question but lacks detail, not very helpful

   Example of rating "1":
   Q: "How do I reset my password?"
   A: "You can't reset it yourself, contact IT department."
   
   Why 1: Incorrect if self-service is available

4. EDGE CASES and HOW TO HANDLE

   What if response is partially correct?
   → Rate on most accurate parts, note errors in comments

   What if response is correct but rude?
   → Consider tone in helpfulness dimension

   What if I'm unsure?
   → Use middle rating (3) and explain in comments

5. WHAT TO AVOID

   Don't penalize for:
   • Different but equally valid approaches
   • Style preferences (unless explicitly part of criteria)
   • Minor grammatical errors (unless affecting clarity)

   Do penalize for:
   • Factual errors
   • Missing critical information
   • Confusing explanations
   • Inappropriate tone

Complete Example:

┌────────────────────────────────────────────────────────────┐
│  ANNOTATION GUIDELINES: Customer Support Evaluation         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Objective:                                                │
│  Rate customer support responses for helpfulness.          │
│                                                            │
│  Rating Scale:                                             │
│  5 - Excellent response, fully helpful                     │
│  4 - Good response, mostly helpful                         │
│  3 - Acceptable response, somewhat helpful                 │
│  2 - Poor response, minimally helpful                      │
│  1 - Very poor response, not helpful                       │
│                                                            │
│  Criteria:                                                 │
│                                                            │
│  Accuracy: Is information correct?                         │
│  • Check against product documentation                     │
│  • Penalize factual errors heavily                         │
│  • "Unsure" is better than wrong information               │
│                                                            │
│  Completeness: Does it fully address question?             │
│  • All parts of question answered?                         │
│  • Necessary context provided?                             │
│  • Follow-up questions anticipated?                        │
│                                                            │
│  Clarity: Is it easy to understand?                        │
│  • Clear language (not jargon unless necessary)            │
│  • Well-structured (steps, bullets when appropriate)       │
│  • Examples provided when helpful                          │
│                                                            │
│  Tone: Is it professional and friendly?                    │
│  • Polite and respectful                                   │
│  • Empathetic when appropriate                             │
│  • Not overly formal or robotic                            │
│                                                            │
│  Examples:                                                 │
│                                                            │
│  [5 examples of each rating level]                         │
│                                                            │
│  Edge Cases:                                               │
│                                                            │
│  • Correct but terse? → Rate 3-4                           │
│  • Friendly but wrong? → Rate 1-2 (accuracy matters most)  │
│  • Partially answers? → Rate 2-3, note what's missing      │
│  • Provides alternative solution? → Rate 4-5 if valid      │
│                                                            │
│  Calibration:                                              │
│  Complete 10 training examples before starting.            │
│  Your ratings will be compared to expert ratings.          │
│  Must achieve >80% agreement to proceed.                   │
└────────────────────────────────────────────────────────────┘

Testing Guidelines:

def test_guidelines(guidelines, test_examples):
    """Test if guidelines are clear."""
    
    # Have multiple people annotate test examples
    annotations = []
    
    for annotator in annotators:
        annotator_ratings = []
        for example in test_examples:
            rating = annotator.rate(example, guidelines)
            annotator_ratings.append(rating)
        annotations.append(annotator_ratings)
    
    # Calculate inter-annotator agreement
    agreement = calculate_agreement(annotations)
    
    if agreement < 0.70:
        print("Warning: Low agreement! Guidelines may be unclear.")
        print("Revise guidelines and test again.")
    else:
        print(f"Good agreement ({agreement:.2%}). Guidelines ready!")
    
    return agreement
'''
    
    print(guidelines)

annotation_guidelines()
```

## Inter-Annotator Agreement

### Measuring Agreement

```python
def inter_annotator_agreement():
    """Calculate and interpret inter-annotator agreement."""
    
    print("\n\nInter-Annotator Agreement:\n")
    print("="*70)
    
    print("""
Inter-annotator agreement measures consistency between human judges.

Why it matters:
  • Low agreement → Task is ambiguous or guidelines unclear
  • High agreement → Reliable, reproducible evaluation
  • Validates annotation quality

Metrics:

1. PERCENT AGREEMENT
   Simplest metric: % of times annotators agree

import numpy as np

def percent_agreement(annotations1, annotations2):
    '''
    Calculate percent agreement between two annotators.
    
    Args:
        annotations1: Ratings from annotator 1
        annotations2: Ratings from annotator 2
    
    Returns:
        Agreement percentage
    '''
    agreements = np.array(annotations1) == np.array(annotations2)
    return np.mean(agreements)

# Example
annotator_a = [5, 4, 3, 5, 2, 4, 5, 3, 4, 5]
annotator_b = [5, 4, 3, 4, 2, 4, 5, 3, 5, 5]

agreement = percent_agreement(annotator_a, annotator_b)
print(f"Percent agreement: {agreement:.0%}")
# Output: 80%

Problem: Doesn't account for chance agreement!

2. COHEN'S KAPPA (for 2 annotators)
   Accounts for chance agreement

from sklearn.metrics import cohen_kappa_score

def cohens_kappa(annotations1, annotations2):
    '''Calculate Cohen's kappa.'''
    return cohen_kappa_score(annotations1, annotations2)

# Example
kappa = cohens_kappa(annotator_a, annotator_b)
print(f"Cohen's kappa: {kappa:.3f}")

Interpretation:
  < 0.00: Less than chance agreement (bad!)
  0.00 - 0.20: Slight agreement
  0.21 - 0.40: Fair agreement
  0.41 - 0.60: Moderate agreement
  0.61 - 0.80: Substantial agreement  ← Target
  0.81 - 1.00: Almost perfect agreement

Target: κ > 0.60 for good quality

3. KRIPPENDORFF'S ALPHA (for 3+ annotators)
   More general, handles missing data

import krippendorff

def krippendorff_alpha(annotations):
    '''
    Calculate Krippendorff's alpha.
    
    Args:
        annotations: 2D array (annotators × examples)
    
    Returns:
        Alpha value
    '''
    return krippendorff.alpha(reliability_data=annotations)

# Example: 3 annotators, 10 examples
annotations = np.array([
    [5, 4, 3, 5, 2, 4, 5, 3, 4, 5],  # Annotator 1
    [5, 4, 3, 4, 2, 4, 5, 3, 5, 5],  # Annotator 2
    [4, 4, 3, 5, 2, 4, 5, 3, 4, 4],  # Annotator 3
])

alpha = krippendorff_alpha(annotations)
print(f"Krippendorff's alpha: {alpha:.3f}")

Interpretation:
  α > 0.80: Good reliability
  α > 0.67: Tentative conclusions possible
  α < 0.67: Questionable reliability

4. INTRACLASS CORRELATION COEFFICIENT (ICC)
   For continuous ratings (e.g., 1-5 scale treated as continuous)

from scipy import stats

def icc(annotations):
    '''
    Calculate ICC for consistency.
    
    Args:
        annotations: 2D array (examples × annotators)
    '''
    n_examples, n_annotators = annotations.shape
    
    # Mean square between examples
    ms_between = np.var(annotations.mean(axis=1)) * n_annotators
    
    # Mean square within examples
    ms_within = np.mean([np.var(row) for row in annotations])
    
    # ICC(2,1) for absolute agreement
    icc_value = (ms_between - ms_within) / (ms_between + (n_annotators - 1) * ms_within)
    
    return icc_value

# Example
annotations_continuous = np.array([
    [5, 4, 3, 5, 2],
    [5, 4, 3, 4, 2],
    [4, 4, 3, 5, 2],
]).T  # Examples × Annotators

icc_value = icc(annotations_continuous)
print(f"ICC: {icc_value:.3f}")

Complete Example:

class AgreementAnalysis:
    '''Comprehensive inter-annotator agreement analysis.'''
    
    def __init__(self, annotations):
        '''
        Args:
            annotations: Dict mapping annotator_id -> list of ratings
        '''
        self.annotations = annotations
        self.annotator_ids = list(annotations.keys())
        self.n_annotators = len(self.annotator_ids)
        self.n_examples = len(list(annotations.values())[0])
    
    def pairwise_agreement(self):
        '''Calculate agreement for all pairs of annotators.'''
        
        pairs = []
        for i in range(self.n_annotators):
            for j in range(i+1, self.n_annotators):
                id_i = self.annotator_ids[i]
                id_j = self.annotator_ids[j]
                
                kappa = cohens_kappa(
                    self.annotations[id_i],
                    self.annotations[id_j]
                )
                
                pairs.append({
                    'annotator_1': id_i,
                    'annotator_2': id_j,
                    'kappa': kappa
                })
        
        return pairs
    
    def overall_agreement(self):
        '''Calculate overall agreement across all annotators.'''
        
        # Convert to array (annotators × examples)
        annotations_array = np.array([
            self.annotations[aid] for aid in self.annotator_ids
        ])
        
        alpha = krippendorff_alpha(annotations_array)
        
        return alpha
    
    def identify_difficult_examples(self, threshold=0.5):
        '''Find examples with low agreement.'''
        
        difficult = []
        
        for i in range(self.n_examples):
            ratings = [self.annotations[aid][i] for aid in self.annotator_ids]
            
            # Calculate variance in ratings
            variance = np.var(ratings)
            
            if variance > threshold:
                difficult.append({
                    'example_id': i,
                    'ratings': ratings,
                    'variance': variance
                })
        
        return sorted(difficult, key=lambda x: x['variance'], reverse=True)
    
    def report(self):
        '''Generate agreement report.'''
        
        print("Inter-Annotator Agreement Report")
        print("=" * 60)
        print(f"Annotators: {self.n_annotators}")
        print(f"Examples: {self.n_examples}")
        print()
        
        # Overall agreement
        alpha = self.overall_agreement()
        print(f"Krippendorff's Alpha: {alpha:.3f}")
        
        if alpha > 0.80:
            quality = "Excellent"
        elif alpha > 0.67:
            quality = "Good"
        elif alpha > 0.60:
            quality = "Acceptable"
        else:
            quality = "Poor - Revise guidelines!"
        
        print(f"Quality: {quality}")
        print()
        
        # Pairwise agreements
        print("Pairwise Agreements:")
        pairs = self.pairwise_agreement()
        for pair in pairs:
            print(f"  {pair['annotator_1']} vs {pair['annotator_2']}: "
                  f"κ = {pair['kappa']:.3f}")
        print()
        
        # Difficult examples
        difficult = self.identify_difficult_examples()
        if difficult:
            print(f"Most Difficult Examples (top 5):")
            for item in difficult[:5]:
                print(f"  Example {item['example_id']}: "
                      f"ratings={item['ratings']}, var={item['variance']:.2f}")

# Example usage
annotations = {
    'annotator_1': [5, 4, 3, 5, 2, 4, 5, 3, 4, 5],
    'annotator_2': [5, 4, 3, 4, 2, 4, 5, 3, 5, 5],
    'annotator_3': [4, 4, 3, 5, 2, 4, 5, 3, 4, 4],
}

analysis = AgreementAnalysis(annotations)
analysis.report()
""")

inter_annotator_agreement()
```

### Resolving Disagreements

```python
def resolve_disagreements():
    """Strategies for handling annotator disagreements."""
    
    print("\n\nResolving Disagreements:\n")
    print("="*70)
    
    strategies = """
When annotators disagree, you have several options:

1. MAJORITY VOTE
   Use the rating chosen by most annotators

def majority_vote(ratings):
    '''Select most common rating.'''
    from collections import Counter
    counts = Counter(ratings)
    return counts.most_common(1)[0][0]

# Example
ratings = [4, 5, 4, 4, 3]  # 4 appears 3 times
final = majority_vote(ratings)
print(f"Final rating: {final}")  # 4

Pros: Simple, democratic
Cons: Discards nuance, may lose information

2. AVERAGE RATING
   Take mean of all ratings

def average_rating(ratings):
    '''Calculate average rating.'''
    return np.mean(ratings)

# Example
ratings = [4, 5, 4, 4, 3]
final = average_rating(ratings)
print(f"Final rating: {final:.1f}")  # 4.0

Pros: Uses all information
Cons: May produce non-integer ratings

3. ADJUDICATION
   Expert breaks ties

def adjudicate(ratings, example, expert):
    '''Have expert resolve disagreement.'''
    
    # Check if consensus exists
    if len(set(ratings)) == 1:
        return ratings[0]  # All agree
    
    # Check if close agreement
    if max(ratings) - min(ratings) <= 1:
        return np.round(np.mean(ratings))  # Close enough
    
    # Significant disagreement → Expert decides
    print(f"Disagreement on example: {example}")
    print(f"Ratings: {ratings}")
    expert_rating = expert.rate(example)
    return expert_rating

Pros: High quality final judgments
Cons: Expensive, not scalable

4. WEIGHTED AVERAGE
   Weight by annotator reliability

class WeightedAggregation:
    '''Aggregate with annotator weights.'''
    
    def __init__(self):
        self.weights = {}  # annotator_id -> weight
    
    def calculate_weights(self, annotations, gold_standard):
        '''
        Calculate weights based on agreement with gold standard.
        
        Args:
            annotations: Dict of annotator_id -> ratings
            gold_standard: Gold standard ratings
        '''
        for annotator_id, ratings in annotations.items():
            # Calculate agreement with gold standard
            correct = sum(r == g for r, g in zip(ratings, gold_standard))
            accuracy = correct / len(gold_standard)
            
            # Weight = accuracy
            self.weights[annotator_id] = accuracy
    
    def aggregate(self, ratings_dict):
        '''
        Aggregate ratings with weights.
        
        Args:
            ratings_dict: Dict of annotator_id -> rating for this example
        '''
        total_weight = 0
        weighted_sum = 0
        
        for annotator_id, rating in ratings_dict.items():
            weight = self.weights.get(annotator_id, 1.0)
            weighted_sum += rating * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean(list(ratings_dict.values()))

# Example
aggregator = WeightedAggregation()

# Calculate weights from gold standard
gold_standard = [5, 4, 3, 5, 2, 4, 5, 3, 4, 5]
annotations = {
    'annotator_1': [5, 4, 3, 5, 2, 4, 5, 3, 4, 5],  # Perfect
    'annotator_2': [5, 3, 3, 4, 2, 4, 5, 3, 4, 5],  # 90% accurate
    'annotator_3': [4, 4, 3, 3, 2, 3, 4, 2, 3, 4],  # 60% accurate
}

aggregator.calculate_weights(annotations, gold_standard)
print("Annotator weights:", aggregator.weights)

# Aggregate new example
ratings_dict = {'annotator_1': 5, 'annotator_2': 4, 'annotator_3': 3}
final = aggregator.aggregate(ratings_dict)
print(f"Weighted average: {final:.2f}")

5. KEEP ALL RATINGS
   Store distribution of ratings

def rating_distribution(ratings):
    '''Keep full distribution.'''
    from collections import Counter
    return dict(Counter(ratings))

# Example
ratings = [4, 5, 4, 4, 3]
dist = rating_distribution(ratings)
print("Distribution:", dist)  # {4: 3, 5: 1, 3: 1}

# When to use:
# - Research where you want to analyze disagreement
# - Report confidence/certainty (high agreement = high confidence)

Pros: Preserves all information
Cons: More complex to use

Decision Guide:

High agreement (κ > 0.80):
  → Any method works, simple majority or average

Moderate agreement (κ = 0.60-0.80):
  → Majority vote or weighted average

Low agreement (κ < 0.60):
  → Adjudication or revise guidelines!

For critical decisions:
  → Always use adjudication

For research:
  → Keep all ratings, analyze distribution
"""
    
    print(strategies)

resolve_disagreements()
```

## A/B Testing

### Designing A/B Tests

```python
def ab_testing():
    """A/B testing for model comparison."""
    
    print("\n\nA/B Testing:\n")
    print("="*70)
    
    print("""
A/B testing compares two models (A vs B) to determine which is better.

Advantages over absolute rating:
  ✓ Easier for humans (compare vs rate on scale)
  ✓ More reliable (less ambiguity)
  ✓ Directly answers "which model to use?"

Standard Setup:

1. Select examples from test set
2. Generate responses from both models
3. Show responses side-by-side (anonymized, randomized order)
4. Annotator picks which is better
5. Calculate win rate

Example:

class ABTest:
    '''A/B test for model comparison.'''
    
    def __init__(self, model_a, model_b, test_examples):
        self.model_a = model_a
        self.model_b = model_b
        self.test_examples = test_examples
        self.results = []
    
    def run_test(self, annotator):
        '''Run A/B test.'''
        
        import random
        
        for example in self.test_examples:
            prompt = example['prompt']
            
            # Generate responses
            response_a = self.model_a.generate(prompt)
            response_b = self.model_b.generate(prompt)
            
            # Randomize order (prevent position bias)
            if random.random() < 0.5:
                first, second = response_a, response_b
                first_model, second_model = 'A', 'B'
            else:
                first, second = response_b, response_a
                first_model, second_model = 'B', 'A'
            
            # Show to annotator
            choice = annotator.compare(prompt, first, second)
            
            # Map back to actual models
            if choice == 'first':
                winner = first_model
            elif choice == 'second':
                winner = second_model
            else:
                winner = 'tie'
            
            self.results.append({
                'prompt': prompt,
                'response_a': response_a,
                'response_b': response_b,
                'winner': winner,
                'choice': choice,
                'order': f"{first_model},{second_model}"
            })
    
    def analyze_results(self):
        '''Analyze A/B test results.'''
        
        wins_a = sum(1 for r in self.results if r['winner'] == 'A')
        wins_b = sum(1 for r in self.results if r['winner'] == 'B')
        ties = sum(1 for r in self.results if r['winner'] == 'tie')
        
        total = len(self.results)
        
        win_rate_a = wins_a / total
        win_rate_b = wins_b / total
        tie_rate = ties / total
        
        print(f"Model A wins: {wins_a} ({win_rate_a:.1%})")
        print(f"Model B wins: {wins_b} ({win_rate_b:.1%})")
        print(f"Ties: {ties} ({tie_rate:.1%})")
        
        # Statistical significance
        sig = self.is_significant(wins_a, wins_b, total)
        print(f"Statistically significant: {sig}")
        
        return {
            'wins_a': wins_a,
            'wins_b': wins_b,
            'ties': ties,
            'win_rate_a': win_rate_a,
            'win_rate_b': win_rate_b,
            'significant': sig
        }
    
    def is_significant(self, wins_a, wins_b, total, alpha=0.05):
        '''Check statistical significance with binomial test.'''
        
        from scipy import stats
        
        # Under null hypothesis, win rate = 0.5
        p_value = stats.binom_test(wins_a, wins_a + wins_b, p=0.5)
        
        return p_value < alpha

# Example Usage
print("\\nExample A/B Test:")
print("-" * 60)

# Simulate results
wins_a = 55
wins_b = 35
ties = 10
total = 100

print(f"Model A wins: {wins_a} ({wins_a/total:.1%})")
print(f"Model B wins: {wins_b} ({wins_b/total:.1%})")
print(f"Ties: {ties} ({ties/total:.1%})")

# Test significance
from scipy import stats
p_value = stats.binom_test(wins_a, wins_a + wins_b, p=0.5)
print(f"\\np-value: {p_value:.4f}")
print(f"Significant at α=0.05: {p_value < 0.05}")

Best Practices:

1. RANDOMIZE ORDER
   Prevent position bias (people favor first/last option)
   
   ✓ Show A first 50% of time, B first 50% of time
   ✗ Always show A first

2. BLIND COMPARISON
   Don't tell annotator which model is which
   
   ✓ Show as "Response 1" and "Response 2"
   ✗ Show as "GPT-4" and "Claude"

3. ADEQUATE SAMPLE SIZE
   Rule of thumb: 100-500 comparisons
   
   More needed for small differences:
   • 55% vs 45%: ~400 comparisons
   • 60% vs 40%: ~100 comparisons

4. MULTIPLE ANNOTATORS
   Have 2-3 people judge each comparison
   More reliable than single annotator

5. REPORT CONFIDENCE INTERVALS
   Not just win rate, but uncertainty

def confidence_interval(wins, total, confidence=0.95):
    '''Calculate confidence interval for win rate.'''
    
    from scipy import stats
    
    win_rate = wins / total
    
    # Wilson score interval
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    center = (win_rate + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((win_rate * (1 - win_rate) / total + z**2 / (4 * total**2))) / denominator
    
    lower = center - margin
    upper = center + margin
    
    return (lower, upper)

# Example
wins = 55
total = 100
lower, upper = confidence_interval(wins, total)
print(f"\\nWin rate: 55% [95% CI: {lower:.1%} - {upper:.1%}]")

Choosing Options:

What should annotators choose from?

Simple (3-way):
  • A is better
  • B is better
  • Tie (equal quality)

Detailed (5-way):
  • A is much better
  • A is slightly better
  • Tie
  • B is slightly better
  • B is much better

With reasoning:
  • Which is better? [A / B / Tie]
  • Why? [Text box]

Recommendation: Use 5-way for nuance, 3-way for simplicity
""")

ab_testing()
```

## Crowdsourcing

### Using Crowdsourcing Platforms

```python
def crowdsourcing_platforms():
    """Guide to crowdsourcing human evaluation."""
    
    print("\n\nCrowdsourcing Human Evaluation:\n")
    print("="*70)
    
    guide = """
Crowdsourcing platforms let you collect judgments at scale.

Popular Platforms:

1. Amazon Mechanical Turk (MTurk)
   • Largest pool of workers
   • ~500,000+ workers worldwide
   • $0.05-0.50 per task typically
   • Good for: Simple rating tasks, large scale

2. Prolific
   • Higher quality workers
   • ~130,000 workers
   • $8-12/hour standard
   • Good for: Research quality, detailed tasks

3. Scale AI
   • Professional annotators
   • Higher quality
   • $$ more expensive
   • Good for: Critical tasks, complex annotation

4. Label Studio / Prodigy
   • Self-hosted annotation tools
   • Work with your own annotators
   • Good for: Internal teams, sensitive data

Crowdsourcing Workflow:

Step 1: Design Task
  • Clear instructions
  • Examples
  • Qualification test

Step 2: Pilot Test
  • Run with 5-10 workers
  • Check quality
  • Revise as needed

Step 3: Launch
  • Start with small batch
  • Monitor quality
  • Scale up

Step 4: Quality Control
  • Gold standard examples
  • Attention checks
  • Filter low-quality workers

Step 5: Aggregate Results
  • Multiple annotators per example
  • Aggregate with majority vote or weighting

Example Task Design (MTurk):

class CrowdsourcingTask:
    '''Design crowdsourcing task.'''
    
    def __init__(self):
        self.title = "Rate AI Assistant Responses"
        self.description = "Read a question and AI response, rate quality"
        self.reward = 0.20  # Per task
        self.time_allotted = 300  # 5 minutes
    
    def create_hit(self, prompt, response):
        '''Create HIT (Human Intelligence Task).'''
        
        hit = f'''
<div class="task">
  <h2>Rate this AI assistant response</h2>
  
  <div class="instructions">
    <p>You will see a user question and an AI assistant's response.</p>
    <p>Rate the response quality from 1 (very poor) to 5 (excellent).</p>
    <p>Consider: accuracy, helpfulness, clarity, completeness.</p>
  </div>
  
  <div class="example">
    <h3>User Question:</h3>
    <p>{prompt}</p>
    
    <h3>AI Response:</h3>
    <p>{response}</p>
  </div>
  
  <div class="rating">
    <p><strong>How would you rate this response?</strong></p>
    <input type="radio" name="rating" value="1"> 1 - Very Poor<br>
    <input type="radio" name="rating" value="2"> 2 - Poor<br>
    <input type="radio" name="rating" value="3"> 3 - Acceptable<br>
    <input type="radio" name="rating" value="4"> 4 - Good<br>
    <input type="radio" name="rating" value="5"> 5 - Excellent<br>
  </div>
  
  <div class="comments">
    <p><strong>Optional: What could be improved?</strong></p>
    <textarea name="comments" rows="3" cols="50"></textarea>
  </div>
</div>
'''
        return hit
    
    def add_attention_check(self):
        '''Add attention check question.'''
        
        check = '''
<div class="attention-check">
  <p><strong>Attention check:</strong></p>
  <p>To show you are reading carefully, please select "4" for this question.</p>
  <input type="radio" name="check" value="1"> 1<br>
  <input type="radio" name="check" value="2"> 2<br>
  <input type="radio" name="check" value="3"> 3<br>
  <input type="radio" name="check" value="4"> 4 ← Select this<br>
  <input type="radio" name="check" value="5"> 5<br>
</div>
'''
        return check

Quality Control Methods:

1. GOLD STANDARD EXAMPLES
   Include examples with known answers
   
   # Mix in gold examples
   gold_examples = [
       {'prompt': '...', 'response': '...', 'gold_rating': 5},
       {'prompt': '...', 'response': '...', 'gold_rating': 1},
   ]
   
   # Check worker accuracy on gold
   def worker_accuracy(worker_ratings, gold_ratings):
       correct = sum(w == g for w, g in zip(worker_ratings, gold_ratings))
       return correct / len(gold_ratings)
   
   # Reject if accuracy < 70%

2. ATTENTION CHECKS
   Explicitly ask workers to select specific answer
   
   Fail = not paying attention → Reject

3. INTER-WORKER AGREEMENT
   Compare workers to each other
   
   Low agreement = poor quality worker

4. TIME CHECKS
   Too fast = not reading carefully
   
   Reject if completion_time < minimum_expected_time

5. QUALIFICATION TESTS
   Require passing test before allowing workers
   
   Test: Rate 10 examples, must match expert ratings >80%

6. REPUTATION SYSTEMS
   Track worker quality over time
   Only invite back good workers

Aggregating Crowdsourced Ratings:

class CrowdAggregation:
    '''Aggregate crowdsourced ratings.'''
    
    def __init__(self):
        self.worker_quality = {}  # worker_id -> quality score
    
    def filter_low_quality(self, ratings, threshold=0.7):
        '''Remove ratings from low-quality workers.'''
        
        filtered = []
        
        for rating in ratings:
            worker_id = rating['worker_id']
            quality = self.worker_quality.get(worker_id, 1.0)
            
            if quality >= threshold:
                filtered.append(rating)
        
        return filtered
    
    def aggregate(self, ratings):
        '''Aggregate with quality weighting.'''
        
        if not ratings:
            return None
        
        total_weight = 0
        weighted_sum = 0
        
        for rating in ratings:
            worker_id = rating['worker_id']
            value = rating['rating']
            quality = self.worker_quality.get(worker_id, 1.0)
            
            weighted_sum += value * quality
            total_weight += quality
        
        return weighted_sum / total_weight

Cost Estimation:

def estimate_cost(n_examples, n_raters_per_example, payment_per_rating, platform_fee=0.20):
    '''
    Estimate crowdsourcing cost.
    
    Args:
        n_examples: Number of examples to rate
        n_raters_per_example: Redundancy (typically 3-5)
        payment_per_rating: Payment per rating (e.g., $0.20)
        platform_fee: Platform fee (MTurk charges 20%)
    '''
    
    total_ratings = n_examples * n_raters_per_example
    base_cost = total_ratings * payment_per_rating
    platform_cost = base_cost * platform_fee
    total_cost = base_cost + platform_cost
    
    return {
        'total_ratings': total_ratings,
        'base_cost': base_cost,
        'platform_fee_cost': platform_cost,
        'total_cost': total_cost
    }

# Example
cost = estimate_cost(
    n_examples=500,
    n_raters_per_example=3,
    payment_per_rating=0.20
)

print(f"Total ratings needed: {cost['total_ratings']}")
print(f"Base cost: ${cost['base_cost']:.2f}")
print(f"Platform fees: ${cost['platform_fee_cost']:.2f}")
print(f"Total cost: ${cost['total_cost']:.2f}")

Best Practices:

✓ Pay fair wages ($8-12/hour equivalent)
✓ Provide clear instructions with examples
✓ Pilot test with small batch first
✓ Include quality control mechanisms
✓ Collect 3-5 ratings per example
✓ Monitor quality continuously
✓ Respond to worker questions quickly

✗ Underpay workers
✗ Launch without pilot testing
✗ Trust single ratings (no redundancy)
✗ Ignore quality issues
✗ Make tasks too complex
✗ Forget attention checks
"""
    
    print(guide)

crowdsourcing_platforms()
```

## Combining Human and Automated Evaluation

### Hybrid Evaluation Pipeline

```python
def hybrid_evaluation():
    """Combine automated and human evaluation effectively."""
    
    print("\n\nHybrid Evaluation Pipeline:\n")
    print("="*70)
    
    print("""
Use automated metrics for scale, human evaluation for quality.

Strategy 1: FILTER THEN EVALUATE

1. Automated metrics filter candidates
2. Human evaluation on filtered set

class HybridPipeline:
    '''Filter with automated, evaluate with human.'''
    
    def __init__(self, automated_threshold=0.7):
        self.automated_threshold = automated_threshold
    
    def filter_candidates(self, responses):
        '''Use automated metrics to filter.'''
        
        candidates = []
        
        for response in responses:
            # Calculate automated metrics
            bleu = calculate_bleu(response['reference'], response['generated'])
            bertscore = calculate_bertscore(response['reference'], response['generated'])
            
            # Filter: Keep if either metric is high
            if bleu > self.automated_threshold or bertscore > self.automated_threshold:
                candidates.append(response)
        
        return candidates
    
    def human_evaluate(self, candidates):
        '''Human evaluate filtered candidates.'''
        
        human_ratings = []
        
        for candidate in candidates:
            rating = human_rater.rate(candidate)
            human_ratings.append(rating)
        
        return human_ratings

Example: Content Moderation
  1. Toxicity model filters (fast, cheap)
  2. Flagged content → Human review (accurate, expensive)

Benefits:
  ✓ Scalable (automated filters most)
  ✓ High quality (human decides edge cases)
  ✓ Cost-effective

Strategy 2: SAMPLE AND EXTRAPOLATE

1. Evaluate small sample with humans
2. Train predictor to extrapolate

class SampleAndExtrapolate:
    '''Evaluate sample, extrapolate to full dataset.'''
    
    def __init__(self, sample_size=200):
        self.sample_size = sample_size
        self.predictor = None
    
    def sample_examples(self, full_dataset):
        '''Sample for human evaluation.'''
        
        import random
        return random.sample(full_dataset, self.sample_size)
    
    def collect_human_ratings(self, sample):
        '''Get human ratings for sample.'''
        
        ratings = []
        for example in sample:
            rating = human_rater.rate(example)
            ratings.append(rating)
        
        return ratings
    
    def train_predictor(self, sample, ratings):
        '''Train model to predict human ratings.'''
        
        from sklearn.ensemble import RandomForestRegressor
        
        # Extract features
        X = self.extract_features(sample)
        y = ratings
        
        # Train predictor
        self.predictor = RandomForestRegressor()
        self.predictor.fit(X, y)
    
    def predict_all(self, full_dataset):
        '''Predict ratings for full dataset.'''
        
        X = self.extract_features(full_dataset)
        predictions = self.predictor.predict(X)
        
        return predictions
    
    def extract_features(self, examples):
        '''Extract features for prediction.'''
        
        features = []
        
        for example in examples:
            # Automated metrics as features
            bleu = calculate_bleu(example['reference'], example['generated'])
            rouge = calculate_rouge(example['reference'], example['generated'])
            bertscore = calculate_bertscore(example['reference'], example['generated'])
            length = len(example['generated'].split())
            
            features.append([bleu, rouge, bertscore, length])
        
        return np.array(features)

Benefits:
  ✓ Cost-effective (only rate sample)
  ✓ Get estimates for full dataset
  ✗ Assumes predictor is accurate

Strategy 3: AUTOMATED FIRST, HUMAN VALIDATION

1. Use automated metrics initially
2. Periodically validate with humans
3. Adjust if correlation degrades

class ContinuousValidation:
    '''Continuously validate automated metrics.'''
    
    def __init__(self, validation_frequency=100):
        self.validation_frequency = validation_frequency
        self.evaluations = 0
        self.correlation_history = []
    
    def evaluate(self, example):
        '''Evaluate example.'''
        
        # Automated evaluation
        automated_score = self.automated_evaluate(example)
        
        self.evaluations += 1
        
        # Periodic human validation
        if self.evaluations % self.validation_frequency == 0:
            self.validate_correlation()
        
        return automated_score
    
    def validate_correlation(self):
        '''Validate automated vs human correlation.'''
        
        # Sample examples
        sample = self.sample_recent_examples(n=50)
        
        # Get automated and human scores
        automated_scores = [self.automated_evaluate(ex) for ex in sample]
        human_scores = [human_rater.rate(ex) for ex in sample]
        
        # Calculate correlation
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(automated_scores, human_scores)
        
        self.correlation_history.append(correlation)
        
        print(f"Correlation: {correlation:.3f} (p={p_value:.3f})")
        
        if correlation < 0.70:
            print("Warning: Low correlation! Review automated metrics.")

Benefits:
  ✓ Mostly automated (scalable)
  ✓ Catches metric drift
  ✓ Ensures reliability

Strategy 4: MULTI-STAGE EVALUATION

Stage 1: Automated metrics (all examples)
Stage 2: LLM-as-judge (filtered examples)
Stage 3: Human evaluation (final candidates)

class MultiStageEvaluation:
    '''Multi-stage evaluation pipeline.'''
    
    def __init__(self):
        self.stage1_threshold = 0.7  # BERTScore
        self.stage2_threshold = 4.0  # LLM score (out of 5)
    
    def evaluate_pipeline(self, examples):
        '''Run full pipeline.'''
        
        print(f"Stage 1: Automated metrics on {len(examples)} examples")
        stage1_pass = self.stage1_automated(examples)
        print(f"  → {len(stage1_pass)} passed")
        
        print(f"\\nStage 2: LLM-as-judge on {len(stage1_pass)} examples")
        stage2_pass = self.stage2_llm_judge(stage1_pass)
        print(f"  → {len(stage2_pass)} passed")
        
        print(f"\\nStage 3: Human evaluation on {len(stage2_pass)} examples")
        final_ratings = self.stage3_human(stage2_pass)
        print(f"  → Complete")
        
        return final_ratings
    
    def stage1_automated(self, examples):
        '''Stage 1: Fast automated metrics.'''
        
        passed = []
        
        for ex in examples:
            bertscore = calculate_bertscore(ex['reference'], ex['generated'])
            
            if bertscore >= self.stage1_threshold:
                passed.append(ex)
        
        return passed
    
    def stage2_llm_judge(self, examples):
        '''Stage 2: LLM-as-judge.'''
        
        passed = []
        
        for ex in examples:
            score = llm_judge.evaluate(ex['prompt'], ex['generated'])
            
            if score >= self.stage2_threshold:
                passed.append(ex)
        
        return passed
    
    def stage3_human(self, examples):
        '''Stage 3: Human evaluation.'''
        
        ratings = []
        
        for ex in examples:
            rating = human_rater.rate(ex)
            ratings.append(rating)
        
        return ratings

Example workflow:
  1000 examples
  → Stage 1 (automated): 500 pass (50%)
  → Stage 2 (LLM): 100 pass (10%)
  → Stage 3 (human): 100 rated (10%)

Cost:
  • Automated: $0
  • LLM: 500 × $0.01 = $5
  • Human: 100 × $0.50 = $50
  Total: $55 (vs $500 if all human)

Decision Matrix:

| Scale    | Quality Need | Budget  | Recommendation              |
|----------|--------------|---------|----------------------------|
| Small    | High         | High    | All human                  |
| Small    | High         | Low     | Sample human + extrapolate |
| Large    | High         | High    | Multi-stage pipeline       |
| Large    | High         | Low     | Automated + validation     |
| Large    | Medium       | Any     | Automated + sample human   |
| Any      | Low          | Any     | Automated only             |

Best Practices:

1. Start with automated for rapid iteration
2. Add human evaluation before major decisions
3. Validate automated metrics regularly
4. Use humans for edge cases and tie-breaking
5. Build multi-stage pipelines for scale + quality
6. Document correlation between automated and human
7. Budget for human evaluation of ~10-20% of data
""")

hybrid_evaluation()
```

## Summary

**Human Evaluation Importance**:

```
Why Human Evaluation:
  • Gold standard for quality
  • Judges subjective aspects (creativity, tone, appropriateness)
  • Validates automated metrics
  • Assesses novel capabilities
  • Makes final deployment decisions

When to Use:
  ✓ Safety-critical applications
  ✓ Subjective quality assessment
  ✓ Final model selection
  ✓ User-facing features
  ✓ Validating automated metrics
  
  ✗ Standard benchmarks with automated evaluation
  ✗ Continuous monitoring at massive scale (use sampling)
```

**Designing Studies**:

1. **Define objectives**: Specific, measurable goals
2. **Success criteria**: Clear thresholds for success
3. **Sample size**: 100-1000 examples depending on effect size
4. **Evaluation method**: Absolute rating, pairwise comparison, or multi-dimensional
5. **Annotator selection**: Experts vs crowdworkers depending on task

**Evaluation Criteria**:

```python
# Multi-dimensional rating
dimensions = {
    'Accuracy': 'Factual correctness',
    'Completeness': 'Fully addresses question',
    'Clarity': 'Easy to understand',
    'Relevance': 'On-topic',
    'Helpfulness': 'Useful to user'
}

# Each rated 1-5 with clear definitions
```

**Annotation Guidelines**:

- Clear rating scale with definitions
- Concrete examples for each rating level
- Edge case handling
- What to penalize vs ignore
- Calibration process

**Inter-Annotator Agreement**:

```python
# Metrics
percent_agreement = agreements / total
cohens_kappa = corrected_for_chance_agreement
krippendorff_alpha = general_agreement_measure

# Interpretation
kappa > 0.80  # Excellent
kappa > 0.60  # Good (target)
kappa < 0.60  # Poor - revise guidelines

# Resolution strategies
- Majority vote (simple)
- Average rating (uses all info)
- Adjudication (expert decides)
- Weighted average (by annotator quality)
```

**A/B Testing**:

```python
# Setup
- Blind comparison (anonymized)
- Randomized order (prevent position bias)
- 100-500 comparisons
- Multiple annotators per comparison

# Analysis
win_rate = wins_a / (wins_a + wins_b)
significance = binomial_test(wins_a, total, p=0.5)
confidence_interval = wilson_score_interval(wins_a, total)
```

**Crowdsourcing**:

```python
# Platforms
MTurk        # Large pool, $0.05-0.50 per task
Prolific     # Higher quality, research-grade
Scale AI     # Professional, expensive

# Quality Control
- Gold standard examples
- Attention checks
- Inter-worker agreement
- Time checks
- Qualification tests
- Worker reputation

# Cost
cost = n_examples * n_raters * payment_per_rating * 1.2
# Example: 500 * 3 * $0.20 * 1.2 = $360
```

**Hybrid Evaluation**:

```python
# Strategies

# 1. Filter then evaluate
automated_filter → human_evaluate(filtered)

# 2. Sample and extrapolate
human_rate(sample) → train_predictor → predict_all

# 3. Continuous validation
automated_evaluate + periodic_human_validation

# 4. Multi-stage pipeline
automated → llm_judge → human (for final candidates)

# Cost-Quality Trade-off
All automated: Fast, cheap, less accurate
All human: Slow, expensive, most accurate
Hybrid: Balanced (recommended)
```

**Best Practices**:

1. **Always use human evaluation** for deployment decisions
2. **Multiple annotators** (3-5) per example for reliability
3. **Clear guidelines** with examples and edge cases
4. **Measure agreement** (target κ > 0.60)
5. **Quality control** mechanisms (gold examples, attention checks)
6. **Report uncertainty** (confidence intervals, not just point estimates)
7. **Combine with automated** for scale and efficiency
8. **Validate regularly** that automated metrics still correlate
9. **Pay fairly** ($8-12/hour for crowdworkers)
10. **Pilot test** annotation process before scaling

**Common Pitfalls**:

- Single annotator per example (unreliable)
- Unclear guidelines (low agreement)
- Position bias (not randomizing order)
- Insufficient sample size (not powered)
- Ignoring disagreements (missing nuance)
- Over-reliance on automation (missing quality issues)
- Underpaying workers (poor quality)

**Key Takeaways**:

- Human evaluation is the gold standard for assessing LLM quality
- Careful study design and clear guidelines are essential
- Inter-annotator agreement validates process quality
- A/B testing is often more reliable than absolute rating
- Crowdsourcing enables scale but requires quality control
- Hybrid approaches balance cost, scale, and quality
- Always validate automated metrics against human judgment
- Budget 10-20% of data for human evaluation in production systems

## Next Steps

- Review [LLM Evaluation Methods](llm-evaluation.md) for automated approaches
- Study [Traditional Metrics](traditional-metrics.md) and [Neural Metrics](neural-metrics.md)
- Learn [Benchmarks](benchmarks.md) for standardized evaluation
- Master [Failure Analysis](failure-analysis.md) to understand disagreements
- Apply evaluation in [RLHF and Alignment](../rlhf-and-alignment/)
- Build evaluation pipelines in [Application Patterns](../application-patterns/)
