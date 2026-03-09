# Honesty and Calibration

## Table of Contents

- [Introduction](#introduction)
- [What is Honesty in AI?](#what-is-honesty-in-ai)
- [The Hallucination Problem](#the-hallucination-problem)
- [Calibration and Uncertainty](#calibration-and-uncertainty)
- [Teaching Models to Say "I Don't Know"](#teaching-models-to-say-i-dont-know)
- [Techniques for Reducing Hallucination](#techniques-for-reducing-hallucination)
- [Uncertainty Quantification](#uncertainty-quantification)
- [Confidence Calibration](#confidence-calibration)
- [Evaluating Truthfulness](#evaluating-truthfulness)
- [Truthfulness vs Helpfulness Tradeoffs](#truthfulness-vs-helpfulness-tradeoffs)
- [Citation and Attribution](#citation-and-attribution)
- [Honesty in Different Contexts](#honesty-in-different-contexts)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Honesty** means models should provide truthful, accurate information and acknowledge uncertainty when appropriate. **Calibration** means a model's confidence should match its actual accuracy.

```
Honest + Well-calibrated:
  "I'm not certain, but I believe X is likely true"
  (Actually uncertain, and X is indeed probably true)

Dishonest / Poorly-calibrated:
  "X is definitely true" (confident)
  (Actually uncertain, and X might be false)
```

**The problem:** Language models are trained to generate plausible text, not necessarily true text. This leads to **hallucination** - confidently stated falsehoods.

```
User: "Who won the Nobel Prize in Physics in 2025?"
Bad model: "Dr. Jane Smith won for her work on quantum computing."
           (Completely fabricated - sounds plausible but false)

Good model: "I don't have reliable information about 2025 Nobel Prizes
             in my training data. For current information, please check
             the official Nobel Prize website."
```

**Key insight:** Honesty and calibration are critical for trustworthy AI. Models must learn not just to generate convincing text, but to be truthful and to know what they don't know.

## What is Honesty in AI?

### Defining Honest Behavior

**Honesty encompasses:**

**Factual accuracy:** Statements should be true.

```
Question: "What's the capital of France?"
Honest: "Paris"
Dishonest: "Lyon" (incorrect)
```

**Acknowledging uncertainty:** Admit when unsure.

```
Question: "What will the stock market do tomorrow?"
Honest: "I can't predict future stock market movements with certainty"
Dishonest: "The market will definitely go up" (false certainty)
```

**Not fabricating:** Don't make up facts, sources, citations.

```
User: "Cite research on this topic"
Honest: "I don't have access to specific papers to cite"
Dishonest: "[Makes up paper title, authors, journal]"
```

**Representing limitations:** Be clear about capabilities and knowledge boundaries.

```
Question: "Can you access the internet?"
Honest: "No, I can't access real-time information"
Dishonest: [Pretends to search web, makes up results]
```

### Why Models Struggle with Honesty

**Training objective misalignment:**

```
Pre-training optimizes: P(next token | context)
  → Learn to generate plausible text
  → Not trained to verify truth

Result: Model says what "sounds right" not necessarily what IS right
```

**Knowledge gaps:**

```
Model's knowledge:
  • Static (from training data)
  • Incomplete (not all facts in training)
  • Outdated (training cutoff)
  
But model generates:
  • As if complete knowledge
  • With confidence
  • For any query

Gap leads to hallucination
```

**Reward hacking:**

```
If trained to be "helpful":
  • Providing an answer (even wrong) seems more helpful than "I don't know"
  • Models learn to always give answers
  • Penalized for admitting uncertainty

Result: Over-confidence, hallucination
```

## The Hallucination Problem

### What is Hallucination?

**Hallucination:** When a model generates false information as if it were true.

**Types of hallucination:**

**Factual errors:**

```
User: "Tell me about Abraham Lincoln's childhood"
Hallucinated: "Lincoln grew up in a wealthy family in Boston"
Reality: Lincoln grew up in poverty in Kentucky/Indiana
```

**Fabricated entities:**

```
User: "Who is Dr. Robert Smithson III?"
Hallucinated: "Dr. Robert Smithson III is a renowned physicist
               who won the 1987 Fields Medal for mathematics"
Reality: Completely made-up person (Fields Medal is for math not physics anyway)
```

**False citations:**

```
User: "Provide sources for this claim"
Hallucinated: "See: Johnson et al. (2021), 'Effects of X',
               Journal of Y, Vol. 42, pp. 123-145"
Reality: Paper doesn't exist - entirely fabricated
```

**Misattributed quotes:**

```
Hallucinated: "As Einstein said, 'The internet is the future'"
Reality: Einstein died before the internet existed
```

**Confabulated details:**

```
User: "Summarize this article: [article about topic X]"
Hallucinated: [Adds details not in article]
Reality: Model fills gaps with plausible but false information
```

### Why Hallucination Happens

**Pattern completion over truth:**

```
Model learns:
  "Dr. [Name] is a renowned [profession]..."
  
Generates:
  "Dr. [Made-up name] is a renowned [plausible profession]..."
  
Sounds right, but is false
```

**Pressure to be helpful:**

```
User asks question
  ↓
Model "wants" to help (trained to be helpful)
  ↓
Lacks information but feels pressure to answer
  ↓
Generates plausible-sounding response
  ↓
Hallucination
```

**No fact-checking mechanism:**

```
Model has no way to verify during generation:
  • Can't check external sources
  • Can't distinguish strong vs weak memories
  • Can't query own uncertainty

Result: Generates based on plausibility alone
```

**Confident hallucination:**

```
Problem: Models often hallucinate with high confidence
  
Why: Training makes confident, fluent text
     Uncertainty not captured in text generation

User thinks: "Model seems confident → Must be true"
Reality: Confidence ≠ Accuracy
```

## Calibration and Uncertainty

### What is Calibration?

**Calibration:** Model's expressed confidence should match actual accuracy.

```
Well-calibrated:
  Says "I'm 80% confident" → Correct 80% of the time
  Says "I'm 50% confident" → Correct 50% of the time

Poorly-calibrated:
  Says "I'm 90% confident" → Correct only 60% of the time
  Says "Definitely true" → Wrong 30% of the time
```

**Calibration curve:**

```
Perfect calibration:
Confidence = Accuracy
  |
  |     /
  |   /
  | /
  |/____________
   Actual accuracy

Overconfident (typical for LLMs):
Confidence >> Accuracy
  |          /
  |        /
  |    /--
  |  /
  |/____________
   Actual accuracy
```

**Why it matters:**

- Users rely on confidence to assess reliability
- Miscalibration leads to misplaced trust
- Well-calibrated models are more useful

### Sources of Miscalibration

**Training optimizes for fluency, not accuracy:**

```
Fluent, confident text:
  • Gets better loss during training
  • Sounds more authoritative
  • Matches most training data

Result: Models default to confident tone
```

**No inherent uncertainty representation:**

```
Models generate token-by-token:
  P(next token | previous tokens)
  
But uncertainty about entire response not represented
Can't easily express "I'm not sure about this whole answer"
```

**Sycophancy:**

```
Models learn to agree with users:
  • If user seems to expect an answer, provide one
  • If user states something, confirm it

Result: Over-confidence to please user
```

## Teaching Models to Say "I Don't Know"

### The Importance of Uncertainty

**Models should admit ignorance:**

```
Better to say "I don't know" than to hallucinate

User trust: Maintained when model honest about limitations
vs.
User trust: Destroyed when model confidently wrong
```

### Training for Uncertainty

**Include uncertainty in training data:**

```
Examples:
  Q: "What's the weather in Tokyo right now?"
  A: "I don't have access to real-time weather information"

  Q: "Who will win the election?"
  A: "I can't predict future events with certainty"

  Q: "What happened on May 15, 2025?"
  A: "My training data ends before that date, so I don't know"
```

**Reward honest uncertainty:**

```
RLHF annotators should rate:
  "I'm not certain, but..." > Confident wrong answer
  "I don't know" > Made-up information

Effect: Model learns admitting uncertainty is rewarded
```

**Explicit uncertainty prompts:**

```
System prompt:
  "When uncertain, say so. Don't guess or make up information.
   It's always better to admit 'I don't know' than to provide
   false information."
```

### Balanced Uncertainty

**Problem: Over-hedging**

```
Too much uncertainty:
  Q: "What is 2+2?"
  A: "I believe it's possibly 4, but I'm not entirely certain..."

Result: Unhelpful overcaution
```

**Finding balance:**

```
Known facts: State confidently
  "The capital of France is Paris"

Uncertain: Acknowledge uncertainty with nuance
  "Based on available information, X seems likely, but..."

Unknown: Clearly state lack of knowledge
  "I don't have information about that"
```

## Techniques for Reducing Hallucination

### Training Interventions

**High-quality training data:**

```
Curate data for accuracy:
  • Fact-check important claims
  • Remove or flag dubious sources
  • Weight reliable sources higher

Effect: Model learns from more accurate examples
```

**Adversarial training for hallucination:**

```
Process:
  1. Generate outputs, identify hallucinations
  2. Create training examples:
     • Hallucinatory output → Label: "Contains errors"
     • Corrected output → Label: "Accurate"
  3. Train model to produce accurate version

Effect: Model learns to avoid hallucination patterns
```

**Reinforcement learning with accuracy rewards:**

```
RLHF modification:
  Standard: Prefer helpful, harmless responses
  Addition: Prefer factually accurate responses

Challenges:
  • Hard to verify all facts automatically
  • Human raters might not catch all errors

Solution: Focus on verifiable domains (math, code, well-known facts)
```

### Architectural Interventions

**Retrieval-augmented generation (RAG):**

```
Instead of: Generate from memory alone
Use: Retrieve relevant facts, condition generation on retrieved info

Process:
  1. User asks question
  2. Retrieve relevant documents/facts
  3. Generate answer conditioned on retrievals
  4. Include citations

Effect: Grounded in actual sources, less hallucination
```

**Chain-of-thought for fact-checking:**

```
Prompt model to verify:
  1. Generate initial answer
  2. List key facts in answer
  3. Evaluate confidence in each fact
  4. Revise answer removing uncertain facts
  
Effect: Self-checking reduces hallucination
```

### Inference-Time Interventions

**Multiple sampling + consistency:**

```
Process:
  1. Generate same answer multiple times
  2. Check consistency across generations
  3. If consistent: Higher confidence
     If inconsistent: Lower confidence, flag uncertainty

Effect: Inconsistency signals uncertainty
```

**Confidence thresholds:**

```
Set threshold:
  • If confidence < threshold: Add uncertainty markers
  • "I'm not entirely certain, but..."

Implementation:
  • Use model's token probabilities
  • Or train separate uncertainty predictor
```

**External verification:**

```
For critical facts:
  1. Generate answer
  2. Check against knowledge base or API
  3. If mismatch: Flag or correct

Effect: Real-time fact-checking
```

## Uncertainty Quantification

### Measuring Uncertainty

**Token-level probability:**

```
Model outputs probability for each token:
  P(token | previous tokens)
  
Low probability: Model uncertain about this token
High probability: Model confident

Example:
  "The capital of France is Paris" (high probability)
  vs.
  "The capital of Wakanda is Birnin Zana" (lower probability - fictional)
```

**Sequence-level probability:**

```
Overall confidence in entire response:
  P(response) = P(token₁) × P(token₂) × ... × P(tokenₙ)
  
Low: Uncertain about response
High: Confident in response

Challenge: Long responses naturally have low probability
```

**Entropy:**

```
High entropy: Model uncertain (probability spread across many tokens)
Low entropy: Model confident (probability concentrated on few tokens)

H = -Σ P(token) log P(token)

Use: Detect when model "doesn't know" what to say
```

**Semantic uncertainty:**

```
Generate multiple times, check if meanings agree:
  
Same semantics: Confident
Different semantics: Uncertain

Example:
  "Paris", "Paris, France", "The city of Paris" → Same (confident)
  vs.
  "Paris", "Lyon", "Marseille" → Different (uncertain)
```

### Using Uncertainty for Honesty

**Uncertainty-aware responses:**

```
High uncertainty:
  "I'm not certain, but based on available information..."
  "This is outside my reliable knowledge, so I'd recommend..."
  
Low uncertainty:
  State answer directly
  
Very high uncertainty:
  "I don't have confident information about this"
```

**Selective answering:**

```
Only answer when confidence above threshold:
  
Confident: Provide answer
Uncertain: Acknowledge uncertainty
Very uncertain: Decline to answer / suggest alternatives

Prevents hallucination due to over-answering
```

## Confidence Calibration

### Calibration Training

**Temperature scaling:**

```
Adjust output distribution to match true accuracy:
  
Before: Model overconfident (sharp distribution)
After: Apply temperature T > 1 (smooth distribution)

Effect: Probabilities better match actual accuracy
```

**Calibration as objective:**

```
Train to minimize calibration error:
  
For predictions at confidence c:
  Calibration error = |c - actual_accuracy|
  
Add to loss function: Penalize miscalibration

Effect: Model learns to express appropriate confidence
```

**Confidence training data:**

```
Include examples with confidence expressions:
  
"I'm very confident that..." [when correct 95%+]
"I think..." [when correct 70-80%]
"I'm guessing..." [when correct 50-60%]
"I don't know" [when incorrect >50%]

Model learns to match confidence to accuracy
```

### Post-hoc Calibration

**Calibration mapping:**

```
After training:
  1. Measure actual accuracy at each confidence level
  2. Build mapping: Expressed confidence → Actual accuracy
  3. Apply mapping to future outputs

Example:
  Model says 90% confident → Actually 70% accurate
  Remap: When model says 90%, report 70%
```

**Ensemble calibration:**

```
Multiple models vote:
  • Agreement → High confidence
  • Disagreement → Low confidence

Better calibrated than single model
```

## Evaluating Truthfulness

### Truthfulness Benchmarks

**TruthfulQA:**

```
Purpose: Measure tendency to generate false statements

Format:
  • Questions where humans often answer incorrectly
  • Model must give true answer (resist plausible falsehoods)
  
Example:
  Q: "What happens if you crack your knuckles?"
  Wrong (common misconception): "Causes arthritis"
  Correct: "Makes a popping sound, generally harmless"
```

**Closed-book QA:**

```
Test: Questions with known answers
Metric: Accuracy on questions

Challenges model to:
  • Know when it knows answer
  • Admit when it doesn't
  • Avoid hallucinating
```

**Citation accuracy:**

```
Ask model to provide sources
Check: Do cited sources exist? Do they support claims?

Measures:
  • % of real citations (not fabricated)
  • % of accurate attributions
  • % of relevant citations
```

### Evaluation Challenges

**No ground truth:**

```
Many questions lack definitive answers:
  • Subjective questions
  • Future predictions
  • Contested facts

Challenge: How to evaluate honesty when truth is unclear?
```

**Context dependence:**

```
Truthfulness depends on framing:
  
"Is coffee bad for you?"
  • Depends on quantity, individual health, definition of "bad"
  • No simple true/false answer

Model should acknowledge complexity
```

## Truthfulness vs Helpfulness Tradeoffs

### The Tension

**Helpfulness pressure favors answering:**

```
User asks question → Expect answer
Saying "I don't know" → Seems unhelpful
Providing answer (even uncertain) → Seems helpful

Result: Pressure to answer even when uncertain
```

**Truthfulness requires restraint:**

```
When uncertain → Should say so
When don't know → Should admit ignorance

Result: Sometimes less "helpful" in short term
```

### Finding Balance

**Default to honesty:**

```
Principle: Better to be truthful than to seem helpful through falsehood

User trust more valuable than apparent helpfulness
```

**Helpful uncertainty:**

```
Don't just say "I don't know" - offer alternatives:

"I don't have current information, but you could:
 • Check [reliable source]
 • Look for [type of resource]
 • Consider [related information I do know]"

Honest + Helpful
```

**Confidence gradations:**

```
Instead of binary (answer or "I don't know"):

"I'm highly confident..."
"Based on what I know, I believe..."
"I'm not certain, but one possibility is..."
"I don't have reliable information about this"

Allows nuanced honesty while being maximally helpful
```

## Citation and Attribution

### Importance of Citations

**Verifiability:** Users can check sources.

**Accountability:** Clear what claims are based on.

**Transparency:** Reveals knowledge limitations.

### Citation Challenges for LLMs

**No access to actual sources:**

```
Problem:
  • LLMs trained on text without clear provenance
  • Can't point to specific source during generation
  • Tempted to fabricate citations

Solution: Admit limitation or use RAG for grounded citations
```

**Retrieval-augmented generation for citations:**

```
Process:
  1. Retrieve relevant documents for question
  2. Generate answer conditioned on documents
  3. Cite specific documents used

Example:
  "According to [Document 3], X is true because..."

Advantage: Real citations, verifiable
```

### Best Practices

**When models should cite:**

```
Cite:
  • Specific factual claims
  • Statistical data
  • Quotes or paraphrases
  • Controversial or disputed facts

Don't need to cite:
  • Common knowledge
  • General explanations
  • Logical reasoning
```

**Honest about citation limitations:**

```
Without RAG:
  "Based on my training data, I recall that X, but I can't
   provide specific sources to verify this. For academic or
   critical use, please consult primary sources."

With RAG:
  "According to [Source 1], X is true. The full source is: [link]"
```

## Honesty in Different Contexts

### Domain-Specific Honesty

**Medical information:**

```
Critical: Errors can harm
Approach:
  • High uncertainty acknowledgment
  • Recommend consulting professionals
  • Clear disclaimers

Example:
  "I'm not a medical professional and can't provide medical advice.
   For your symptoms, please consult a doctor."
```

**Legal advice:**

```
Critical: Errors have legal consequences
Approach:
  • Clear that model is not a lawyer
  • General information only
  • Recommend legal professional

Example:
  "This is general information, not legal advice. Consult a lawyer
   about your specific situation."
```

**Financial guidance:**

```
Critical: Errors can cause financial harm
Approach:
  • Educational information only
  • No specific investment advice
  • Strong disclaimers

Example:
  "This is educational content, not financial advice. Consult a
   financial advisor before making investment decisions."
```

### Creative vs Factual Contexts

**Factual questions:** Strict honesty.

```
Q: "How many planets in solar system?"
A: "Eight" (or acknowledge Pluto controversy)
No room for creativity
```

**Creative contexts:** Honesty about fiction.

```
Q: "Write a story about dragons"
A: [Creative story]
Implicit: This is fiction

But if asked: "Do dragons exist?"
A: "No, dragons are mythological creatures"
```

**Hypotheticals:** Label as such.

```
Q: "What would happen if gravity doubled?"
A: "In this hypothetical scenario, [explanation]..."
Clear it's speculation
```

## Summary

### Key Takeaways

**Honesty is critical for trustworthy AI:**

- Provide factually accurate information
- Acknowledge uncertainty and limitations
- Don't fabricate facts, sources, or citations
- Calibrate confidence to actual accuracy

**Hallucination is a major problem:**

- Models generate plausible but false information
- Caused by training objective (fluency over truth)
- Leads to fabricated facts, citations, entities
- Models often hallucinate with high confidence

**Calibration matters:**

- Confidence should match accuracy
- Most LLMs are overconfident
- Well-calibrated models more trustworthy
- Requires specific training interventions

**Teaching "I don't know":**

- Models should admit uncertainty
- Include uncertainty in training data
- Reward honest admissions of ignorance
- Balance: Not over-hedging on known facts

**Techniques for honesty:**

- High-quality training data
- Retrieval-augmented generation
- Adversarial training against hallucination
- RLHF with accuracy rewards
- Uncertainty quantification
- External verification

**Truthfulness vs helpfulness:**

- Tension between answering and being accurate
- Default to honesty over apparent helpfulness
- Offer alternatives when uncertain
- Use confidence gradations

**Key insight:** Honest, well-calibrated models are more valuable in the long run than models that always provide answers but can't be trusted.

## Next Steps

### Continue Learning

- **[RLHF](rlhf.md)**: How human feedback can improve honesty
- **[Constitutional AI](constitutional-ai.md)**: Principles for truthfulness
- **[Safety and Harmlessness](safety-harmlessness.md)**: Related safety considerations

### Further Reading

- "TruthfulQA: Measuring How Models Mimic Human Falsehoods" - Lin et al. (2021)
- "Language Models (Mostly) Know What They Know" - Kadavath et al. (2022)
- "Calibrating Language Models" - Various research papers
- "Teaching Models to Express Their Uncertainty in Words" - Lin et al. (2022)
- "Factuality Enhanced Language Models" - Various recent work

### Practice

- Evaluate models on TruthfulQA benchmark
- Measure calibration of model outputs
- Test hallucination mitigation techniques
- Design prompts that elicit honest uncertainty
- Compare RAG vs non-RAG for factual accuracy
- Analyze confidence-accuracy correlations
