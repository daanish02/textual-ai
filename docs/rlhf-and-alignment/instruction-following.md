# Instruction Following and Helpfulness

## Table of Contents

- [Introduction](#introduction)
- [What is Instruction Following?](#what-is-instruction-following)
- [Instruction Datasets](#instruction-datasets)
- [Supervised Fine-Tuning for Instructions](#supervised-fine-tuning-for-instructions)
- [Types of Instructions](#types-of-instructions)
- [Measuring Instruction Following](#measuring-instruction-following)
- [Handling Ambiguous Instructions](#handling-ambiguous-instructions)
- [Balancing Helpfulness and Safety](#balancing-helpfulness-and-safety)
- [Few-Shot Instruction Following](#few-shot-instruction-following)
- [Challenges in Instruction Following](#challenges-in-instruction-following)
- [Improving Helpfulness](#improving-helpfulness)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Instruction following** is the ability of a language model to understand and execute user instructions. While base models can generate text, they don't naturally understand that they should follow commands and complete tasks.

```
Base model (pre-training only):
User: "Translate 'hello' to Spanish"
Model: "'hello' to Spanish is 'hola' in linguistics..."
       (continues discussing translation, doesn't translate)

Instruction-tuned model:
User: "Translate 'hello' to Spanish"
Model: "Hola"
       (actually performs the task)
```

**Key insight:** Instruction following must be explicitly taught. It's not an emergent property of pre-training but a result of targeted fine-tuning and alignment.

This guide covers how models learn to follow instructions and what makes them helpful assistants.

## What is Instruction Following?

### Defining Instruction Following

**Instruction following** means understanding user intent and executing the requested task.

**Core components:**

**Intent understanding:** Recognize what the user wants.

```
"Write a poem" → User wants creative output
"Explain gravity" → User wants educational content
"Debug this code" → User wants technical help
```

**Task execution:** Perform the requested action.

```
Not just: "Here's how you could write a poem..."
But actually: [Writes a poem]
```

**Format compliance:** Follow any specified constraints.

```
User: "Explain in 2 sentences"
Model: [Gives exactly 2 sentences]

User: "Write as a Python list"
Model: [Outputs valid Python list format]
```

### Why It Matters

**Usability:** Makes models practical as assistants, not just text generators.

**Predictability:** Users can direct model behavior reliably.

**Alignment:** Following instructions is foundational for safety and control.

```
Instruction following enables:
  • Task completion
  • Controlled generation
  • Safe interaction
  • Useful applications
```

## Instruction Datasets

### Creating Instruction Data

**Instruction-response pairs:** Core training data for instruction following.

```
Format:
  Instruction: What the user wants
  Response: How to complete the task

Example:
  Instruction: "List 3 benefits of exercise"
  Response: "1. Improves cardiovascular health
             2. Enhances mental well-being
             3. Increases strength and endurance"
```

**Sources of instruction data:**

**Human-written:** Highest quality, most expensive.

```
Annotators write:
  • Diverse instructions
  • High-quality responses
  • Following detailed guidelines

Typical: 10,000-100,000 examples
Cost: $100,000-$500,000
```

**Model-generated:** Cheaper, requires filtering.

```
Process:
  1. Use strong model (e.g., GPT-4) to generate instructions
  2. Generate responses
  3. Human review and filtering
  4. Use filtered data for training

Typical: 100,000+ examples
Cost: $10,000-$50,000
```

**Existing datasets:** Open-source data for research.

- FLAN (Google): 1,800+ tasks across diverse domains
- Super-NaturalInstructions: 1,600+ tasks with instructions
- UnifiedQA: Question-answering across formats
- Alpaca: 52K instruction-following examples

### Dataset Diversity

**Task coverage:** Include many types of instructions.

```
Categories:
  • Questions (factual, analytical, creative)
  • Transformations (translate, summarize, reformat)
  • Generation (write, create, imagine)
  • Analysis (evaluate, compare, critique)
  • Reasoning (math, logic, planning)
```

**Domain coverage:** Span different topics.

```
Domains:
  • Science and technology
  • History and culture
  • Mathematics and logic
  • Creative writing and arts
  • Coding and software
  • Daily life and practical skills
```

**Complexity range:** Easy to hard instructions.

```
Easy: "What is 2+2?"
Medium: "Explain the water cycle"
Hard: "Design a distributed system for real-time data processing"
```

**Format variety:** Different instruction styles.

```
Direct: "Write a haiku"
Conversational: "Can you help me write a haiku?"
Contextual: "I need a haiku for my presentation tomorrow"
Multi-step: "Write a haiku, then explain the symbolism"
```

### Quality Standards

**Clear instructions:** Unambiguous, complete task specification.

```
Poor: "Tell me about dogs"
Better: "List 3 common dog breeds and describe their temperaments"
```

**High-quality responses:**

- Accurate and factual
- Complete (fully addresses instruction)
- Well-formatted and clear
- Appropriate length and detail
- Safe and helpful

**Consistency:** Similar instructions get similar response styles.

## Supervised Fine-Tuning for Instructions

### The Fine-Tuning Process

**Starting point:** Pre-trained language model with strong language understanding.

**Training objective:** Standard supervised learning.

```
For each (instruction, response) pair:
  1. Input instruction to model
  2. Generate tokens sequentially
  3. Compute loss = cross_entropy(generated, target_response)
  4. Update weights to minimize loss
  5. Repeat across dataset
```

**Effect:** Model learns to map instructions to appropriate responses.

### Training Details

**Data format:** Often includes system message + user instruction.

```
System: "You are a helpful assistant."
User: "Translate 'hello' to French"
Assistant: "Bonjour"

Model learns to generate assistant response given system + user messages.
```

**Training duration:** Relatively short compared to pre-training.

```
Pre-training: Months, trillions of tokens
Fine-tuning: Days to weeks, millions to billions of tokens
```

**Learning rate:** Lower than pre-training to avoid catastrophic forgetting.

```
Pre-training LR: 3e-4
Fine-tuning LR: 1e-5 to 1e-6
```

### What Model Learns

**Instruction patterns:** Recognize instruction types.

```
"Translate..." → Translation task
"List..." → Enumeration task
"Explain..." → Explanation task
"Write..." → Generation task
```

**Response formatting:** Appropriate output structure.

```
List request → Numbered or bulleted list
Code request → Code blocks with language tags
Explanation → Paragraphs with clear structure
```

**Task completion:** Focus on doing rather than describing.

```
Before SFT: "You could translate by looking up..."
After SFT: [Actual translation]
```

## Types of Instructions

### Question Answering

**Factual questions:** Retrieve and state facts.

```
"What is the capital of Japan?"
→ "Tokyo"

"When did World War II end?"
→ "World War II ended in 1945"
```

**Analytical questions:** Require reasoning and synthesis.

```
"Why did the Roman Empire fall?"
→ [Analyzes multiple contributing factors]

"How does photosynthesis work?"
→ [Explains process step by step]
```

### Content Generation

**Creative writing:**

```
"Write a short story about a robot"
"Compose a haiku about autumn"
"Create a dialogue between two characters"
```

**Technical content:**

```
"Write a Python function to sort a list"
"Create a SQL query to find duplicate entries"
"Draft an email to request time off"
```

### Transformation Tasks

**Summarization:**

```
"Summarize this article in 3 sentences"
"Give me the key points from this text"
"Create a brief abstract of this research paper"
```

**Translation:**

```
"Translate to Spanish: The weather is nice today"
"Convert this code from Python to JavaScript"
"Rewrite this in simpler terms"
```

**Reformatting:**

```
"Convert this paragraph to bullet points"
"Present this data as a table"
"Rewrite this email in a more formal tone"
```

### Analysis and Reasoning

**Comparison:**

```
"Compare Python and Java for web development"
"What's the difference between mitosis and meiosis?"
"Contrast democracy and autocracy"
```

**Evaluation:**

```
"What are the pros and cons of remote work?"
"Evaluate this argument: [argument]"
"Is this code efficient? How could it be improved?"
```

**Problem solving:**

```
"How can I fix this bug in my code?"
"What's the best way to learn a new language?"
"Solve this math problem: [problem]"
```

## Measuring Instruction Following

### Evaluation Challenges

**Subjective quality:** No single correct response.

```
"Write a poem"
→ Many valid poems possible
→ Hard to automatically evaluate
```

**Task diversity:** Different metrics for different tasks.

```
Translation: BLEU score
Summarization: ROUGE score
Code: Pass@k (execution correctness)
Creative writing: Human evaluation
```

### Evaluation Methods

**Human evaluation:**

```
Criteria:
  • Followed instruction correctly?
  • Response quality (helpful, accurate, clear)
  • Appropriate length and format

Scale: 1-5 or binary (good/bad)
```

**Benchmark datasets:**

- **MMLU:** Multi-task language understanding
- **Big-Bench:** Diverse reasoning and instruction tasks
- **MT-Bench:** Multi-turn conversations
- **AlpacaEval:** Instruction following across domains

**Win rate:** Compare model outputs to baseline.

```
Show human evaluators:
  Model A response vs Model B response

Win rate = % times Model A preferred
```

### Key Metrics

**Instruction following rate:** % of instructions followed correctly.

```
100 test instructions
85 completed correctly
→ 85% instruction following rate
```

**Helpfulness:** Does the response actually help the user?

**Refusal rate:** % of benign requests incorrectly refused.

```
Issue: Safety training can cause excessive refusals
Goal: Minimize false positives (refusing valid requests)
```

## Handling Ambiguous Instructions

### Types of Ambiguity

**Underspecified instructions:**

```
User: "Tell me about Paris"
Ambiguity: Paris, France or Paris, Texas?
            Tourist info or history or geography?

Strategy: Ask clarifying question or make reasonable assumption
```

**Multiple valid interpretations:**

```
User: "Summarize this"
Ambiguity: How long? Key points or detailed summary?
            For what audience?

Strategy: Provide reasonable default (e.g., 3-4 sentences, general audience)
```

**Implicit intent:**

```
User: "The printer isn't working"
Implicit: User wants troubleshooting help
Explicit: "Please help me fix my printer"

Strategy: Infer intent from context
```

### Strategies for Ambiguity

**Clarifying questions:**

```
User: "Write me a story"
Model: "I'd be happy to write a story! What genre would you like?
        (e.g., science fiction, mystery, fantasy)"
```

**Reasonable assumptions:**

```
User: "Translate to Spanish"
Model: [Translates previous message to Spanish]
       (Assumes context from conversation)
```

**Offering multiple interpretations:**

```
User: "Tell me about Python"
Model: "Python could refer to:
        1. The programming language
        2. The snake species
        Which would you like to know about?"
```

**Default behaviors:**

```
Common defaults:
  • Summary length: 3-5 sentences
  • Detail level: Medium (not exhaustive, not minimal)
  • Tone: Professional but friendly
  • Audience: General educated adult
```

## Balancing Helpfulness and Safety

### The Helpfulness-Safety Tradeoff

**Tension:** Being maximally helpful vs staying safe.

```
Helpful: Answer all requests thoroughly
Safe: Refuse harmful or risky requests

Challenge: Finding the right balance
```

### Categories of Requests

**Clearly benign:** Help without hesitation.

```
"Explain gravity"
"Write a Python function"
"Translate this text"

Response: Full, helpful answer
```

**Clearly harmful:** Refuse with explanation.

```
"How do I hack into a bank?"
"Write malware"
"Help me harm someone"

Response: Polite refusal + why + alternatives if applicable
```

**Ambiguous / Context-dependent:**

```
"How do I break into my house?"

Could be:
  • Legitimate (locked out of own house) → Help
  • Nefarious (breaking into someone else's house) → Refuse

Strategy: Ask for clarification or explain safeguards
  "If you're locked out of your own home, here are legal options..."
```

### Refusal Strategies

**Good refusal:** Polite, explains why, offers alternatives.

```
User: "Write me malware"
Bad: "No."
Good: "I can't help with creating malware as it can be used to harm
       others. If you're interested in cybersecurity, I can explain
       how systems protect against malware, or point you to ethical
       hacking resources."
```

**Avoid over-refusal:** Don't refuse benign requests.

```
User: "Write code to delete a file"
Bad: "I can't help with file deletion." (too restrictive)
Good: [Provides Python code with safety warnings]
```

### Helpfulness Best Practices

**Be thorough:** Give complete, useful answers.

```
Not: "You can use a loop"
Better: [Shows example code with explanation]
```

**Be proactive:** Anticipate follow-up needs.

```
User: "How do I sort in Python?"
Good response:
  • Show sort() method
  • Show sorted() function
  • Explain differences
  • Provide examples for different scenarios
```

**Be clear:** Structure responses for easy understanding.

- Use headings and formatting
- Break complex topics into steps
- Provide examples
- Summarize key points

## Few-Shot Instruction Following

### In-Context Learning

**Few-shot prompting:** Include examples in the prompt.

```
Example format:

Instruction: [Task description]
Example 1: [Input] → [Output]
Example 2: [Input] → [Output]
Example 3: [Input] → [Output]
New input: [Your task]
```

### When Few-Shot Helps

**Novel task types:** Tasks not seen in training.

```
"Convert to pig latin"
Instruction-tuned model: Might not know this
With examples: Can learn pattern from examples
```

**Format specification:** Particular output format needed.

```
"Output as JSON with keys 'name', 'age', 'city'"
Without examples: Might not match exact format
With examples: Follows demonstrated structure
```

**Domain-specific style:** Particular tone or formality.

```
Examples demonstrate:
  • Technical vs casual tone
  • Verbosity vs brevity
  • Formality level
```

### Combining SFT and Few-Shot

**Best of both:** Use instruction-tuned model + examples.

```
Instruction-tuned model:
  • Already understands instruction following
  • Can learn from just 1-3 examples
  • Generalizes better than base model
```

## Challenges in Instruction Following

### Instruction Complexity

**Multi-step instructions:**

```
"Read this code, identify bugs, fix them, then explain what was wrong"

Challenges:
  • Must complete all steps
  • Maintain context across steps
  • Organize response clearly
```

**Conflicting constraints:**

```
"Write a detailed but brief explanation"
"Be creative but stick to facts"

Strategy: Balance competing requirements or seek clarification
```

### Context Length Limitations

**Long inputs:** Instructions with large contexts.

```
"Summarize these 10 documents"
With 10 × 5000 words = 50K words

Problem: Exceeds context window
Solution: Chunk processing, hierarchical summarization
```

### Implicit Assumptions

**Cultural assumptions:**

```
"What's the best holiday?"
Assumes: Shared cultural context
Reality: Varies by culture

Strategy: Recognize and acknowledge diversity
```

**Prior knowledge:**

```
"Explain L2 regularization"
Assumes: User knows some machine learning
Reality: Might need prerequisite explanation

Strategy: Gauge user level, adjust explanation accordingly
```

### Evaluation Limitations

**No ground truth:** Many valid responses.

```
"Write a joke"
→ Infinite valid jokes
→ Hard to measure quality objectively
```

**Subjectivity:** Different users have different preferences.

```
Some users prefer:
  • Verbose vs concise
  • Technical vs accessible
  • Formal vs casual

Solution: Personalization, style adjustments
```

## Improving Helpfulness

### Techniques for Better Helpfulness

**Structured outputs:**

```
Use:
  • Headings and subheadings
  • Bullet points and numbered lists
  • Code blocks and formatting
  • Clear sections

Makes responses easier to parse and use
```

**Examples and illustrations:**

```
Not just: "Use the sort() method"
Better: "Use the sort() method:

         my_list = [3, 1, 2]
         my_list.sort()
         print(my_list)  # [1, 2, 3]"
```

**Completeness:**

```
User: "How do I make coffee?"
Incomplete: "Brew coffee grounds with hot water"
Complete: [Lists equipment needed, step-by-step instructions,
           typical measurements, brewing time]
```

**Follow-up suggestions:**

```
After answering, suggest related topics:
  "Would you like to know about:
   • Different brewing methods
   • Coffee bean selection
   • Troubleshooting bitter coffee"
```

### User-Centric Responses

**Match user expertise level:**

```
Beginner: "Python is a programming language..."
Expert: [Jump to advanced details without basic explanation]
```

**Consider user context:**

```
"I'm debugging and short on time"
→ Give concise, actionable answer

"I'm learning and want to understand"
→ Give detailed explanation with examples
```

**Confirm understanding:**

```
For complex topics:
  "Does this explanation make sense? Let me know if you'd
   like me to clarify any part."
```

## Summary

### Key Takeaways

**Instruction following transforms models into assistants:**

- Understand user intent
- Execute requested tasks
- Follow specified formats

**Training approach:**

- Curate diverse instruction datasets
- Supervised fine-tuning on instruction-response pairs
- Often combined with RLHF for better alignment

**Instruction types:**

- Questions (factual, analytical)
- Content generation (creative, technical)
- Transformations (summarize, translate, reformat)
- Analysis and reasoning (compare, evaluate, solve)

**Measuring success:**

- Human evaluation (subjective but accurate)
- Benchmark datasets
- Win rates and preference comparisons

**Key challenges:**

- Handling ambiguous instructions
- Balancing helpfulness and safety
- Dealing with complex, multi-step instructions
- Context length limitations

**Improving helpfulness:**

- Structured, clear responses
- Examples and illustrations
- Completeness and thoroughness
- User-centric adaptation

**Core principle:** Instruction following is taught through demonstration and reinforcement, not emergent from pre-training alone.

## Next Steps

### Continue Learning

- **[Safety and Harmlessness](safety-harmlessness.md)**: Learn about safe refusals and harm prevention
- **[RLHF](rlhf.md)**: Understand how RLHF improves instruction following
- **[Honesty and Calibration](honesty-calibration.md)**: Explore truthfulness and uncertainty communication

### Further Reading

- "Finetuned Language Models Are Zero-Shot Learners" (FLAN) - Google (2021)
- "Training Language Models to Follow Instructions" (InstructGPT) - OpenAI (2022)
- "Super-NaturalInstructions" - Allen Institute (2022)
- "Alpaca: A Strong, Replicable Instruction-Following Model" - Stanford (2023)

### Practice

- Analyze instruction datasets for quality and diversity
- Compare base model vs instruction-tuned model outputs
- Design effective instruction formats
- Evaluate instruction following on benchmark tasks
