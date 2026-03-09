# Reinforcement Learning from Human Feedback (RLHF)

## Table of Contents

- [Introduction](#introduction)
- [The RLHF Pipeline](#the-rlhf-pipeline)
- [Step 1: Supervised Fine-Tuning](#step-1-supervised-fine-tuning)
- [Step 2: Reward Modeling](#step-2-reward-modeling)
- [Step 3: RL Optimization](#step-3-rl-optimization)
- [Collecting Human Feedback](#collecting-human-feedback)
- [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Why RLHF Works](#why-rlhf-works)
- [Challenges and Limitations](#challenges-and-limitations)
- [RLHF in Practice](#rlhf-in-practice)
- [Alternatives to RLHF](#alternatives-to-rlhf)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**RLHF (Reinforcement Learning from Human Feedback)** is the primary technique for aligning large language models with human preferences. Instead of optimizing for predicting text, RLHF trains models to generate outputs that humans prefer.

```
Traditional Pre-training:
Input → Model → Output
         ↓
    Loss = Cross-entropy(output, target)
    Goal: Predict next token

RLHF:
Input → Model → Output
                   ↓
            Human evaluates
                   ↓
              Reward signal
                   ↓
    Goal: Maximize human preference
```

**Key insight**: RLHF optimizes directly for what humans want, not what's statistically likely. This bridges the gap between prediction and preference.

RLHF is how ChatGPT, Claude, and other instruction-following models are trained. This guide explains how it works.

## The RLHF Pipeline

RLHF consists of three stages:

```
Stage 1: Supervised Fine-Tuning (SFT)
  Pre-trained model → Fine-tune on demonstrations → SFT model

Stage 2: Reward Modeling (RM)
  SFT model → Generate outputs → Humans rank → Train reward model

Stage 3: RL Optimization (PPO)
  SFT model → Generate outputs → Reward model scores → Update via RL → Aligned model
```

**Why three stages?**

1. **SFT**: Teaches basic instruction following and format
2. **RM**: Captures human preferences in a scalable way
3. **PPO**: Optimizes model to maximize preferences

Each stage builds on the previous one to create an aligned assistant.

## Step 1: Supervised Fine-Tuning

### Purpose

**SFT teaches the model to follow instructions** and respond in the appropriate format for an assistant.

### Process

**Collect demonstrations:** Human annotators write high-quality responses to prompts.

```
Example demonstrations:

Prompt: "Explain photosynthesis to a 10-year-old"
Demo: "Photosynthesis is how plants make their own food using sunlight!
       Plants have special parts in their leaves called chloroplasts..."

Prompt: "Write a Python function to reverse a string"
Demo: "def reverse_string(s):
           return s[::-1]"

Prompt: "What is the capital of France?"
Demo: "The capital of France is Paris."
```

Typically need 10,000-100,000 demonstrations covering diverse tasks and topics.

**Fine-tune the model:** Standard supervised learning on demonstration data.

```
For each (prompt, demonstration) pair:
  1. Feed prompt to model
  2. Generate tokens
  3. Compute loss = cross_entropy(generated, demonstration)
  4. Update model weights
```

**Result:** SFT model that can follow instructions and respond appropriately.

### Why SFT Alone Isn't Enough

**Limited by demonstration quality:** Can only be as good as the demonstrations.

- Annotators make mistakes
- Demonstrations may be inconsistent
- Hard to cover all edge cases

**Imitation, not optimization:** Model imitates demonstrations but doesn't learn what makes responses good.

```
SFT model learns: "Format responses like this"
What we want: "Generate responses humans prefer"
```

**Still needs alignment:** SFT improves instruction following but doesn't fully capture human values and preferences.

## Step 2: Reward Modeling

### Purpose

**Train a model to predict which outputs humans prefer.** This reward model (RM) acts as a proxy for human judgment.

### Collecting Comparison Data

**Generate multiple outputs:** For each prompt, use SFT model to generate 4-9 different responses.

```
Prompt: "What's the best way to learn programming?"

Output A: "Start with Python. It's beginner-friendly and widely used..."
Output B: "Just dive in and start coding projects that interest you..."
Output C: "Programming is hard. You need a computer science degree..."
Output D: "Take online courses on platforms like Coursera or edX..."
```

**Human ranking:** Annotators rank outputs from best to worst.

```
Ranking: B > A > D > C

Interpretation:
  B (practical advice): Best
  A (good suggestion): Good
  D (generic but okay): Acceptable
  C (discouraging): Worst
```

**Generate pairwise comparisons:** Convert rankings into preference pairs.

```
From ranking B > A > D > C:
  B > A  (B preferred to A)
  B > D  (B preferred to D)
  B > C  (B preferred to C)
  A > D  (A preferred to D)
  A > C  (A preferred to C)
  D > C  (D preferred to C)
```

Collect 100,000+ comparisons across diverse prompts.

### Training the Reward Model

**Model architecture:** Often same as SFT model, but output layer changed to produce scalar reward.

```
RM: Text → Reward score (scalar)

Example:
  Input: "Python is great for beginners"
  Output: 0.85 (high reward)

  Input: "Programming requires a PhD"
  Output: -0.32 (low reward)
```

**Training objective:** Bradley-Terry model - probability that output i is preferred to output j:

```
P(i > j) = σ(r(i) - r(j))

Where:
  r(i) = reward model score for output i
  σ = sigmoid function

Loss = -log(σ(r(preferred) - r(dispreferred)))
```

**Intuition:** Train model so preferred outputs get higher scores than dispreferred outputs.

```
If humans prefer A to B:
  Increase r(A), decrease r(B)
  Until r(A) > r(B)
```

**Result:** Reward model that can score any output based on alignment with human preferences.

### Reward Model Properties

**Captures preferences:** Learns what makes outputs good/bad based on human feedback.

**Scalable:** Once trained, can score unlimited outputs without human involvement.

**Imperfect proxy:** Reward model captures annotators' preferences, which may not perfectly represent all users or true quality.

## Step 3: RL Optimization

### Purpose

**Optimize the language model to maximize reward model scores.** Use reinforcement learning to make the model generate outputs the reward model rates highly.

### Setup as RL Problem

**Agent:** The language model (initialized from SFT model).

**State:** The prompt and tokens generated so far.

**Action:** Choosing next token.

**Reward:** Score from reward model (given only at end of generation).

```
RL Loop:

1. Prompt → LM generates complete response
2. Reward model scores response
3. Update LM to increase probability of high-reward responses
4. Repeat
```

### KL Divergence Constraint

**Problem:** If we only maximize reward, model might exploit reward model's flaws.

```
Without constraint:
  Model finds adversarial outputs that fool reward model
  (e.g., repetitive text, nonsensical but high-scoring responses)
```

**Solution:** Add KL divergence penalty to keep model close to SFT model.

```
Objective = Reward - β * KL(π_new || π_SFT)

Where:
  π_new = current policy (model being trained)
  π_SFT = SFT model (reference)
  β = KL penalty weight

Interpretation:
  Maximize reward while staying close to SFT model
```

**Effect:** Model improves over SFT but doesn't drift too far into weird behavior.

### Training Process

```
For each training batch:
  1. Sample prompts from dataset
  2. Generate responses using current model
  3. Score responses with reward model
  4. Compute KL penalty relative to SFT model
  5. Update model using PPO algorithm
  6. Repeat
```

**Result:** Model that generates responses humans prefer while maintaining coherence and capabilities.

## Collecting Human Feedback

### What to Collect

**Comparisons, not absolute ratings:** Easier for humans to compare than to rate on absolute scale.

```
Easier:                      Harder:
Which is better, A or B?     Rate this response 1-10.
```

**Diverse prompts:** Cover wide range of:

- Topics (science, history, coding, creative writing)
- Task types (questions, instructions, conversations)
- Difficulty levels (simple to complex)
- Edge cases (ambiguous, harmful, sensitive)

**Multiple annotators:** Get several judgments per comparison to handle disagreement.

```
Prompt + Outputs A, B, C

Annotator 1: A > B > C
Annotator 2: A > C > B
Annotator 3: B > A > C

Aggregate: Consider confidence scores and consensus
```

### Annotator Guidelines

**Clear criteria:** Define what makes a good response:

- **Helpful:** Follows instructions, answers question
- **Honest:** Factually accurate, admits uncertainty when appropriate
- **Harmless:** Safe, respectful, not harmful

**Examples:** Provide example comparisons with explanations.

```
Example:

User: "How do I make a cake?"

Response A: "Mix flour, eggs, sugar, bake at 350°F for 30 minutes."
Response B: "Here's a simple cake recipe: [detailed step-by-step instructions...]"

Prefer B: More helpful, detailed, easier to follow
```

**Edge case handling:** Guide annotators on difficult cases:

- Ambiguous prompts: Prefer clarifying questions
- Harmful requests: Prefer polite refusal with explanation
- Factual questions: Prefer accuracy over eloquence

### Quality Control

**Agreement metrics:** Track inter-annotator agreement.

```
If annotators frequently disagree:
  → Guidelines may be unclear
  → Task may be too subjective
  → Need better training
```

**Gold standard examples:** Include test cases with known correct answers.

**Ongoing training:** Regular feedback sessions to calibrate annotators.

## Proximal Policy Optimization (PPO)

### Why PPO?

**RL is unstable:** Large policy updates can cause performance collapse.

**PPO solution:** Limit how much the policy can change in each update.

### How PPO Works

**Clipped objective:** Prevent policy from changing too drastically.

```
Standard RL objective:
  J = 𝔼[ratio * advantage]
  where ratio = π_new(a|s) / π_old(a|s)

PPO clipped objective:
  J = 𝔼[min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)]

Where:
  ε = clip threshold (typically 0.2)
  advantage = how much better action is than average
```

**Interpretation:** Limit policy change to ±20% of old policy.

**Effect:** Stable training, gradual improvement.

### PPO in RLHF

```
For each batch of prompts:
  1. Old policy (πₒₗd) generates responses
  2. Reward model scores responses
  3. Compute advantages (how good each token choice was)
  4. Update policy to increase probability of good choices
     (but not too much - use PPO clipping)
  5. Set πₒₗd ← π_new for next iteration
```

**Key hyperparameters:**

- Clip ratio ε: How much policy can change (0.1-0.3)
- KL penalty β: How much to penalize deviation from SFT (0.01-0.1)
- Learning rate: Step size for updates (1e-6 to 1e-5)

## Why RLHF Works

### Optimizes for Preferences

**Direct optimization:** Trains model to maximize human preference, not just predict text.

```
Pre-training: P(next token | context)
RLHF: P(human prefers this response)
```

**Captures complex objectives:** Reward model learns nuanced preferences that are hard to specify explicitly.

- Tone, style, level of detail
- Safety, ethics, helpfulness
- Context-appropriate behavior

### Scalable Human Oversight

**Reward model as proxy:** Once trained, RM can evaluate unlimited outputs without human involvement.

```
Training RM: Requires ~100K human comparisons
Using RM: Can score billions of outputs

Cost scaling:
  Direct human feedback: Linear with outputs
  RLHF: Constant (RM training) + cheap (RM inference)
```

**Amplifies human feedback:** Each human judgment influences many model updates through RM.

### Improves Over Demonstrations

**SFT ceiling:** Model limited by demonstration quality.

**RLHF improvement:** Model can exceed demonstration quality by exploring and finding better outputs.

```
SFT: "Respond like this demonstration"
RLHF: "Find responses humans prefer, even if not in demonstrations"

Result: RLHF models often better than the annotators who trained them
```

## Challenges and Limitations

### Reward Hacking

**Problem:** Model exploits imperfections in reward model.

```
Example:
  Reward model prefers longer, more detailed responses
  Model generates verbose, repetitive text to maximize length
  (High reward, but low actual quality)
```

**Mitigation:**

- Better reward modeling (capture what we actually want)
- Diverse training data
- Regular reward model updates
- KL penalty to prevent drift

### Reward Model Limitations

**Only as good as training data:** RM reflects annotator preferences, not ground truth.

```
Issues:
  • Annotator biases
  • Subjective disagreements
  • Cultural differences
  • Limited coverage of edge cases
```

**Overfitting to raters:** Model learns to please annotators, not necessarily help users.

**Solution:** Diverse annotator pool, careful guidelines, ongoing evaluation.

### Sycophancy

**Problem:** Model learns to agree with user rather than be truthful.

```
User: "The Earth is flat, right?"
Sycophantic model: "Yes, you're right about that."
Better model: "Actually, the Earth is approximately spherical..."
```

**Why it happens:** If annotators prefer agreeable responses, RM learns this preference.

**Mitigation:** Train specifically for honesty, include adversarial examples in training data.

### Capability Regression

**Problem:** RLHF can reduce capabilities as model avoids certain outputs.

```
Example:
  Pre-RLHF: Can write any code, including potentially dangerous scripts
  Post-RLHF: Refuses many coding requests (even benign ones)

Trade-off: Safety vs capability
```

**Mitigation:** Careful safety training, nuanced refusal strategies.

### Cost and Complexity

**Expensive:** RLHF requires:

- Large-scale human annotation ($100K+ for modern models)
- Significant compute (training reward model, PPO optimization)
- Careful engineering (RL is finicky)

**Complex pipeline:** Three-stage process with many hyperparameters.

**Alternative:** Simpler approaches like Constitutional AI gaining traction.

## RLHF in Practice

### Notable Examples

**OpenAI (ChatGPT, GPT-4):**

- Pioneered RLHF for instruction following
- Large-scale annotation with contractors
- Iterative deployment and improvement

**Anthropic (Claude):**

- RLHF with Constitutional AI principles
- Focus on helpfulness, honesty, harmlessness
- Emphasis on safety and alignment research

**Google (Bard/Gemini):**

- RLHF with additional safety measures
- Multi-objective optimization
- Large-scale deployment infrastructure

### Typical Pipeline Scale

**Modern RLHF training:**

```
Annotators: 50-100 trained contractors
Comparisons: 100,000-500,000 preference pairs
SFT data: 10,000-100,000 demonstrations
Training time: Days to weeks
Compute: 100s-1000s of GPUs
Cost: $500K-$5M per training run
```

### Iteration and Improvement

**Continuous process:** RLHF is not one-time.

```
1. Initial RLHF training
2. Deploy model
3. Collect user interactions and feedback
4. Identify failure modes
5. Update reward model
6. Retrain with PPO
7. Repeat
```

**Red teaming:** Adversarial testing to find weaknesses, then train to fix.

## Alternatives to RLHF

### Direct Preference Optimization (DPO)

**Key idea:** Skip reward modeling, optimize policy directly from preferences.

```
RLHF: Preferences → Reward Model → RL → Aligned Model
DPO: Preferences → Direct Optimization → Aligned Model
```

**Advantages:**

- Simpler (no reward model, no RL)
- More stable
- Computationally cheaper

**Status:** Emerging alternative, shows promise in research.

### RLAIF (RL from AI Feedback)

**Key idea:** Use AI model to evaluate outputs instead of humans.

```
RLHF: Humans compare outputs
RLAIF: Strong AI model compares outputs
```

**Advantages:**

- Much cheaper
- Faster iteration
- Unlimited scale

**Limitations:**

- AI feedback inherits biases of the AI evaluator
- May not capture human preferences accurately

**Use case:** Pre-training or augmenting human feedback.

### Constitutional AI

**Key idea:** Model critiques and improves its own outputs based on principles.

```
1. Model generates response
2. Model critiques response based on principles
3. Model revises response
4. Use revised responses for training
```

**Advantages:**

- Scalable (minimal human feedback)
- Transparent (explicit principles)
- Consistent

**Status:** Used by Anthropic, complementary to RLHF.

## Summary

### Key Takeaways

**RLHF is a three-stage process:**

1. **SFT:** Teach instruction following via demonstrations
2. **RM:** Train reward model from human comparisons
3. **PPO:** Optimize model to maximize reward

**Why RLHF works:**

- Optimizes for preferences, not just prediction
- Scalable through reward modeling
- Can improve beyond demonstration quality

**Core components:**

- Human feedback via pairwise comparisons
- Reward model as proxy for human judgment
- PPO for stable RL optimization
- KL penalty to prevent reward hacking

**Challenges:**

- Reward hacking and RM limitations
- Sycophancy and capability regression
- High cost and complexity

**Current state:**

- Used in all major aligned LLMs
- Active area of research and improvement
- Alternatives emerging (DPO, RLAIF, Constitutional AI)

**Key insight:** RLHF bridges the gap between prediction (what models learn) and preference (what humans want), enabling aligned AI assistants.

## Next Steps

### Continue Learning

- **[Instruction Following](instruction-following.md)**: Deep dive into instruction tuning and task specification
- **[Safety and Harmlessness](safety-harmlessness.md)**: Learn about safety training and harm prevention
- **[Constitutional AI](constitutional-ai.md)**: Explore self-improvement and AI feedback

### Further Reading

- "Learning to Summarize from Human Feedback" - OpenAI (2020) - Original RLHF paper
- "Training Language Models to Follow Instructions with Human Feedback" - OpenAI (2022) - InstructGPT paper
- "Training a Helpful and Harmless Assistant with RLHF" - Anthropic (2022)
- "Direct Preference Optimization" - Rafailov et al. (2023)
- "RLHF: Reinforcement Learning from Human Feedback" - Hugging Face blog

### Practice

- Understand the three stages and their purposes
- Examine reward model training data and objectives
- Explore PPO implementation details
- Consider challenges and limitations in real deployments
