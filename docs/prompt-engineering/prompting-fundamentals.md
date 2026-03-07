# Prompting Fundamentals

## Table of Contents

- [Introduction](#introduction)
- [What is a Prompt?](#what-is-a-prompt)
- [Prompt Anatomy](#prompt-anatomy)
- [Zero-Shot Prompting](#zero-shot-prompting)
- [Task Specification](#task-specification)
- [Providing Context](#providing-context)
- [Setting Constraints](#setting-constraints)
- [Output Format Specification](#output-format-specification)
- [Common Prompting Mistakes](#common-prompting-mistakes)
- [Prompt Clarity and Precision](#prompt-clarity-and-precision)
- [Testing and Validation](#testing-and-validation)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Prompt engineering** is the practice of designing inputs to large language models to achieve desired outputs. Unlike traditional programming where you write code, with LLMs you write natural language instructions. The quality of these instructions dramatically affects results.

```
Traditional Programming:
def classify_sentiment(text):
    # Code logic here
    return sentiment

LLM Programming (Prompting):
"Classify the sentiment of this text as positive, negative, or neutral: {text}"
```

**Key insight**: LLMs are incredibly capable but need clear communication. A well-crafted prompt can mean the difference between excellent and poor performance.

```
Prompt Quality Impact:

Poor Prompt:  "sentiment?"           → Confusion, poor results

Okay Prompt:  "What's the sentiment?" → Works sometimes

Good Prompt:  "Classify the sentiment of the following text as
               positive, negative, or neutral: {text}"
               → Reliable, clear results
```

This guide covers fundamental principles for effective prompt engineering.

## What is a Prompt?

### Definition and Purpose

```python
def prompt_definition():
    """Understanding what a prompt is and does."""

    print("Prompt Definition:\n")

    print("A prompt is:")
    print("  • Natural language input to an LLM")
    print("  • Instructions + data + context")
    print("  • The 'program' that controls model behavior")
    print("  • Both the question and the specification")

    print("\nPrompt as Interface:")
    print("  User Intent → Prompt → LLM → Output")
    print("                  ↑")
    print("           (Critical translation)")

    print("\nComponents:")
    components = {
        'Instruction': 'What to do',
        'Context': 'Background information',
        'Input data': 'What to process',
        'Examples': 'How to do it (optional)',
        'Constraints': 'What to avoid/follow',
        'Output format': 'How to respond'
    }

    for component, description in components.items():
        print(f"  • {component}: {description}")

    print("\nExample Prompt:")
    print("─" * 60)
    example = """
Task: Translate the following English text to French.

Context: This is customer support communication, use formal language.

Text: "Thank you for contacting us. We will respond within 24 hours."

Output: Provide only the French translation.
"""
    print(example)
    print("─" * 60)

    print("\nExpected Output:")
    print("Nous vous remercions de nous avoir contactés. Nous vous")
    print("répondrons dans un délai de 24 heures.")

prompt_definition()
```

### Types of Prompts

```python
def prompt_types():
    """Different categories of prompts."""

    print("\n\nTypes of Prompts:\n")

    types = {
        'Zero-shot': {
            'description': 'Task with no examples',
            'example': '"Translate to Spanish: Hello"',
            'when': 'Simple, well-defined tasks',
            'success_rate': '50-80%'
        },
        'Few-shot': {
            'description': 'Task with 1-10 examples',
            'example': 'EN: Hi → ES: Hola\\nEN: Bye → ES: Adiós\\nEN: Thanks → ES:',
            'when': 'Complex or ambiguous tasks',
            'success_rate': '70-95%'
        },
        'Instruction-based': {
            'description': 'Explicit step-by-step instructions',
            'example': 'Read text, identify entities, return JSON',
            'when': 'Multi-step processes',
            'success_rate': '65-85%'
        },
        'Conversational': {
            'description': 'Multi-turn dialogue',
            'example': 'Back-and-forth with context',
            'when': 'Interactive applications',
            'success_rate': '70-90%'
        },
        'Role-based': {
            'description': 'Assign persona to model',
            'example': '"You are an expert Python developer"',
            'when': 'Specific expertise needed',
            'success_rate': '60-85%'
        }
    }

    for ptype, info in types.items():
        print(f"{ptype.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

prompt_types()
```

## Prompt Anatomy

### Core Components

```python
def prompt_anatomy():
    """Breaking down prompt components."""

    print("Prompt Anatomy:\n")

    print("1. INSTRUCTION (Required)")
    print("   What the model should do")
    print("   Example: 'Summarize the following article'")
    print()

    print("2. CONTEXT (Optional but often helpful)")
    print("   Background information")
    print("   Example: 'This is a technical blog post about machine learning'")
    print()

    print("3. INPUT DATA (Usually required)")
    print("   The content to process")
    print("   Example: [Article text goes here]")
    print()

    print("4. EXAMPLES (Optional)")
    print("   Demonstrations of desired behavior")
    print("   Example: 'Input: X → Output: Y'")
    print()

    print("5. CONSTRAINTS (Optional)")
    print("   Rules and limitations")
    print("   Example: 'Keep summary under 100 words'")
    print()

    print("6. OUTPUT FORMAT (Often important)")
    print("   How to structure the response")
    print("   Example: 'Respond in JSON format'")
    print()

    print("=" * 60)
    print("\nComplete Example:\n")

    complete_prompt = """
[INSTRUCTION]
Extract all person names mentioned in the text.

[CONTEXT]
This is a news article about a political event.

[INPUT DATA]
"President Jane Smith met with Ambassador John Doe yesterday.
The meeting was also attended by Senator Mary Johnson."

[CONSTRAINTS]
- Only extract full names (first and last)
- Do not include titles (President, Ambassador, etc.)

[OUTPUT FORMAT]
Return a JSON array of names.

[EXAMPLES]
Input: "Dr. Bob Lee spoke with Prof. Alice Wang"
Output: ["Bob Lee", "Alice Wang"]

Now process the input data above.
"""

    print(complete_prompt)

    print("\nExpected Output:")
    print('["Jane Smith", "John Doe", "Mary Johnson"]')

prompt_anatomy()
```

### Ordering Components

```python
def component_ordering():
    """Best practices for ordering prompt components."""

    print("\n\nComponent Ordering Best Practices:\n")

    print("Recommended Order:")
    print("  1. Instruction (what to do)")
    print("  2. Context (background)")
    print("  3. Examples (if using few-shot)")
    print("  4. Input data (what to process)")
    print("  5. Constraints (rules)")
    print("  6. Output format (how to respond)")
    print()

    print("Why this order?")
    print("  • Instruction first: Sets up task immediately")
    print("  • Context early: Frames the task properly")
    print("  • Examples before input: Shows the pattern to follow")
    print("  • Input last: Fresh in model's 'mind' (recency)")
    print("  • Format last: Final reminder before generating")

    print("\n" + "=" * 60)
    print("\nGood Structure:\n")

    good_prompt = """
Classify the sentiment of customer reviews as positive, negative, or neutral.

These are reviews of a restaurant.

Examples:
"Great food, excellent service!" → positive
"Terrible experience, never going back." → negative
"It was okay, nothing special." → neutral

Review: "The pasta was amazing but the wait was too long."

Classify as: positive, negative, or neutral
"""

    print(good_prompt)

    print("\n" + "=" * 60)
    print("\nPoor Structure (input before instruction):\n")

    poor_prompt = """
"The pasta was amazing but the wait was too long."

Classify this somehow using these examples:
"Great food!" → positive
"Terrible!" → negative

What's the sentiment?
"""

    print(poor_prompt)

    print("\nIssue: Unclear what to do with the text initially")

component_ordering()
```

## Zero-Shot Prompting

### When Zero-Shot Works

```python
def zero_shot_effectiveness():
    """Understanding when zero-shot prompting is sufficient."""

    print("\n\nZero-Shot Prompting:\n")

    print("Definition: Prompting without examples")
    print()

    print("Works well for:")

    tasks_good = {
        'Well-known tasks': {
            'examples': ['Translation', 'Summarization', 'Basic QA'],
            'why': 'Model trained extensively on these',
            'success': '70-90%'
        },
        'Simple operations': {
            'examples': ['Capitalize text', 'Count words', 'Extract emails'],
            'why': 'Clear, unambiguous instructions',
            'success': '80-95%'
        },
        'General knowledge': {
            'examples': ['Famous people', 'Common facts', 'Basic concepts'],
            'why': 'Within training distribution',
            'success': '60-85%'
        },
        'Format conversions': {
            'examples': ['CSV to JSON', 'List to numbered', 'Case changes'],
            'why': 'Deterministic transformations',
            'success': '75-90%'
        }
    }

    for task_type, info in tasks_good.items():
        print(f"\n{task_type.upper()}:")
        print(f"  Examples: {', '.join(info['examples'])}")
        print(f"  Why: {info['why']}")
        print(f"  Success rate: {info['success']}")

    print("\n" + "=" * 60)
    print("\nWorks poorly for:")

    tasks_poor = {
        'Novel tasks': 'Custom business logic, uncommon formats',
        'Ambiguous requirements': 'Vague or unclear specifications',
        'Domain-specific': 'Specialized medical, legal, technical tasks',
        'Complex reasoning': 'Multi-step logic, intricate calculations',
        'Style-specific': 'Particular writing styles or formats'
    }

    for task_type, description in tasks_poor.items():
        print(f"  • {task_type}: {description}")

    print("\nRule of thumb:")
    print("  If a human would need examples to understand the task,")
    print("  the model probably needs examples too.")

zero_shot_effectiveness()
```

### Zero-Shot Examples

```python
def zero_shot_examples():
    """Examples of effective zero-shot prompts."""

    print("\n\nZero-Shot Prompt Examples:\n")

    examples = [
        {
            'task': 'Translation',
            'prompt': 'Translate the following English text to German: "Hello, how are you today?"',
            'output': 'Hallo, wie geht es dir heute?',
            'quality': 'Excellent'
        },
        {
            'task': 'Summarization',
            'prompt': 'Summarize this in one sentence: "Artificial intelligence is transforming healthcare through improved diagnostics, personalized treatment plans, and drug discovery. However, challenges remain in data privacy, algorithmic bias, and integration with existing systems."',
            'output': 'AI is improving healthcare through better diagnostics and treatments but faces challenges with privacy, bias, and system integration.',
            'quality': 'Good'
        },
        {
            'task': 'Classification',
            'prompt': 'Is this review positive or negative? "The product broke after one week of use."',
            'output': 'Negative',
            'quality': 'Excellent'
        },
        {
            'task': 'Question Answering',
            'prompt': 'What is the capital of France?',
            'output': 'Paris',
            'quality': 'Excellent'
        },
        {
            'task': 'Entity Extraction',
            'prompt': 'Extract all company names from this text: "Apple and Microsoft announced a partnership. Google declined to comment."',
            'output': 'Apple, Microsoft, Google',
            'quality': 'Very Good'
        }
    ]

    for ex in examples:
        print(f"TASK: {ex['task']}")
        print(f"Prompt: \"{ex['prompt']}\"")
        print(f"Expected Output: \"{ex['output']}\"")
        print(f"Quality: {ex['quality']}")
        print("─" * 60)
        print()

zero_shot_examples()
```

## Task Specification

### Clear Instructions

```python
def clear_instructions():
    """How to write clear task instructions."""

    print("Writing Clear Instructions:\n")

    print("Principles of Clarity:\n")

    principles = {
        'Be specific': {
            'bad': 'Tell me about this',
            'good': 'Provide a 3-sentence summary of the main points',
            'why': 'Specific = less ambiguity'
        },
        'Use action verbs': {
            'bad': 'This needs classification',
            'good': 'Classify this email as spam or not spam',
            'why': 'Clear what action to take'
        },
        'Define scope': {
            'bad': 'Extract information',
            'good': 'Extract only person names and their job titles',
            'why': 'Boundaries prevent over/under-extraction'
        },
        'Specify format': {
            'bad': 'Give me the names',
            'good': 'List the names in a comma-separated format',
            'why': 'Consistent, parsable output'
        },
        'Avoid ambiguity': {
            'bad': 'Make this better',
            'good': 'Improve the grammar and clarity of this text',
            'why': '"Better" is subjective and vague'
        }
    }

    for principle, examples in principles.items():
        print(f"{principle.upper()}:")
        print(f"  ✗ Bad:  {examples['bad']}")
        print(f"  ✓ Good: {examples['good']}")
        print(f"  Why:   {examples['why']}")
        print()

clear_instructions()
```

### Task Decomposition

```python
def task_decomposition():
    """Breaking complex tasks into clear steps."""

    print("\n\nTask Decomposition:\n")

    print("Complex Task: 'Analyze this customer review'")
    print("(Too vague - what kind of analysis?)")
    print()

    print("Decomposed Version:\n")

    steps = [
        "1. Read the customer review below",
        "2. Identify the sentiment (positive/negative/neutral)",
        "3. Extract specific product features mentioned",
        "4. Rate the urgency (low/medium/high) based on language",
        "5. Provide a one-sentence summary",
        "6. Format as JSON: {sentiment, features, urgency, summary}"
    ]

    for step in steps:
        print(f"  {step}")

    print("\n" + "=" * 60)
    print("\nExample with Numbered Steps:\n")

    prompt = """
Analyze the following customer review:

"The laptop is fast and the screen is beautiful, but it gets very hot
and the battery dies quickly. I need this fixed ASAP!"

Please:
1. Classify sentiment (positive/negative/mixed)
2. List specific issues mentioned
3. Determine urgency level (low/medium/high)
4. Suggest appropriate response priority

Provide your analysis step by step.
"""

    print(prompt)

    print("\nBenefits:")
    print("  • Clear sequence of operations")
    print("  • Model follows logical flow")
    print("  • Easier to debug if something goes wrong")
    print("  • More consistent results")

task_decomposition()
```

## Providing Context

### Why Context Matters

```python
def context_importance():
    """Understanding the role of context in prompts."""

    print("Importance of Context:\n")

    print("Without Context:")
    print("─" * 60)
    prompt_no_context = 'Summarize this: "The patient presented with acute symptoms."'
    print(prompt_no_context)
    print("\nIssue: Model doesn't know what kind of summary is needed")
    print("Result: Generic, possibly inappropriate medical summary")
    print()

    print("With Context:")
    print("─" * 60)
    prompt_with_context = """
Context: This is a clinical note for a medical handoff to the night shift.
The summary should focus on immediate action items and patient status.

Summarize this for the night shift nurse:
"The patient presented with acute symptoms."
"""
    print(prompt_with_context)
    print("\nBenefit: Model knows audience and purpose")
    print("Result: Actionable, appropriate summary")

    print("\n" + "=" * 60)
    print("\nTypes of Useful Context:\n")

    context_types = {
        'Audience': 'Who will read/use this?',
        'Purpose': 'What is the goal?',
        'Domain': 'What field/industry?',
        'Tone': 'Formal, casual, technical?',
        'Constraints': 'Time, length, style limits?',
        'Background': 'What happened before?'
    }

    for ctype, description in context_types.items():
        print(f"  • {ctype}: {description}")

context_importance()
```

### Effective Context Provision

```python
def effective_context():
    """How to provide context effectively."""

    print("\n\nProviding Effective Context:\n")

    print("Example 1: Translation with Context\n")

    print("Without context:")
    print('  "Translate to French: bank"')
    print('  → "banque" (financial institution)')
    print()

    print("With context:")
    print('  "Translate to French (geography context): bank"')
    print('  → "rive" (riverbank)')
    print()

    print("=" * 60)
    print("\nExample 2: Content Generation with Context\n")

    prompt = """
Context:
- Audience: Technical developers with 5+ years experience
- Purpose: API documentation
- Tone: Professional but approachable
- Domain: Machine learning REST API

Write a description for this endpoint:
POST /api/v1/predict
Accepts JSON with model_id and input_data, returns predictions.
"""

    print(prompt)

    print("\n" + "=" * 60)
    print("\nContext Pitfalls to Avoid:\n")

    pitfalls = [
        ('Too much context', 'Overwhelming, loses focus', 'Keep context concise and relevant'),
        ('Irrelevant context', 'Confuses or misleads', 'Only include pertinent information'),
        ('Contradictory context', 'Mixed signals', 'Ensure consistency'),
        ('Assumed context', 'Model lacks background', 'Make context explicit')
    ]

    for pitfall, problem, solution in pitfalls:
        print(f"  • {pitfall}:")
        print(f"    Problem: {problem}")
        print(f"    Solution: {solution}")
        print()

effective_context()
```

## Setting Constraints

### Types of Constraints

```python
def constraint_types():
    """Different types of constraints in prompts."""

    print("Types of Constraints:\n")

    constraints = {
        'Length constraints': {
            'examples': ['Max 100 words', 'Exactly 3 sentences', 'Under 500 characters'],
            'enforcement': 'Medium - model tries but may exceed',
            'tip': 'Be specific about units (words/sentences/characters)'
        },
        'Format constraints': {
            'examples': ['JSON only', 'Numbered list', 'Table format'],
            'enforcement': 'Good - model usually follows',
            'tip': 'Provide format template/example'
        },
        'Content constraints': {
            'examples': ['No technical jargon', 'Avoid humor', 'Professional tone only'],
            'enforcement': 'Medium - requires clear definition',
            'tip': 'Be explicit about what to avoid'
        },
        'Scope constraints': {
            'examples': ['Only 2023 data', 'US markets only', 'Main points, no details'],
            'enforcement': 'Good if clearly specified',
            'tip': 'Define boundaries precisely'
        },
        'Style constraints': {
            'examples': ['Formal academic style', 'Like a news article', 'Casual blog post'],
            'enforcement': 'Good - model adapts style well',
            'tip': 'Reference familiar writing styles'
        },
        'Exclusion constraints': {
            'examples': ['Do not include opinions', 'No personal data', 'Exclude prices'],
            'enforcement': 'Medium - model may forget',
            'tip': 'List exclusions explicitly'
        }
    }

    for constraint_type, info in constraints.items():
        print(f"{constraint_type.upper()}:")
        print(f"  Examples: {', '.join(info['examples'])}")
        print(f"  Enforcement: {info['enforcement']}")
        print(f"  Tip: {info['tip']}")
        print()

constraint_types()
```

### Writing Effective Constraints

```python
def effective_constraints():
    """Best practices for constraint specification."""

    print("\n\nWriting Effective Constraints:\n")

    print("Example: Email Response Generation\n")

    prompt = """
Generate a response to this customer email:
"Your product doesn't work. I want a refund immediately!"

Constraints:
- Professional and empathetic tone
- Maximum 150 words
- Include apology and solution
- Do NOT promise immediate refunds (requires manager approval)
- Do NOT use phrases like "I understand your frustration" (overused)
- End with clear next steps

Response:
"""

    print(prompt)

    print("\n" + "=" * 60)
    print("\nConstraint Specification Patterns:\n")

    patterns = {
        'Positive constraints': {
            'description': 'What TO do',
            'example': 'Include: greeting, summary, action items',
            'better_than': 'Just saying what not to do'
        },
        'Negative constraints': {
            'description': 'What NOT to do',
            'example': 'Do not include: prices, personal opinions, speculation',
            'use_when': 'Preventing specific errors'
        },
        'Range constraints': {
            'description': 'Within bounds',
            'example': 'Between 50-100 words, 3-5 bullet points',
            'benefit': 'Flexibility within limits'
        },
        'Priority constraints': {
            'description': 'Ordered importance',
            'example': 'Must include X, should include Y, optionally Z',
            'benefit': 'Clear priority hierarchy'
        }
    }

    for pattern, info in patterns.items():
        print(f"{pattern.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

    print("Tips:")
    print("  • Be specific but not overly restrictive")
    print("  • Test constraints - do they help or hinder?")
    print("  • Prioritize critical constraints")
    print("  • Use examples to illustrate constraints")

effective_constraints()
```

## Output Format Specification

### Common Output Formats

```python
def output_formats():
    """Specifying desired output formats."""

    print("Common Output Formats:\n")

    formats = {
        'Plain text': {
            'use_case': 'Simple, human-readable output',
            'prompt': 'Provide a brief summary.',
            'example': 'The article discusses AI trends in 2024...'
        },
        'Structured list': {
            'use_case': 'Enumerated items',
            'prompt': 'List the main points as numbered items.',
            'example': '1. First point\\n2. Second point\\n3. Third point'
        },
        'JSON': {
            'use_case': 'Programmatic parsing',
            'prompt': 'Return as JSON: {"name": "...", "age": ...}',
            'example': '{"name": "John", "age": 30, "city": "NYC"}'
        },
        'Markdown': {
            'use_case': 'Formatted documentation',
            'prompt': 'Format as markdown with headers and lists.',
            'example': '## Heading\\n- Bullet 1\\n- Bullet 2'
        },
        'Table': {
            'use_case': 'Tabular data display',
            'prompt': 'Create a table with columns: Name, Date, Status',
            'example': '| Name | Date | Status |\\n|------|------|--------|'
        },
        'CSV': {
            'use_case': 'Spreadsheet import',
            'prompt': 'Output as CSV: name,date,status',
            'example': 'John,2024-01-01,Active\\nJane,2024-01-02,Pending'
        }
    }

    for format_type, info in formats.items():
        print(f"{format_type.upper()}:")
        print(f"  Use case: {info['use_case']}")
        print(f"  Prompt: \"{info['prompt']}\"")
        print(f"  Example output: {info['example']}")
        print()

output_formats()
```

### JSON Output Templates

```python
def json_output_specification():
    """Best practices for JSON output specification."""

    print("\n\nJSON Output Specification:\n")

    print("Approach 1: Describe structure in prompt\n")

    prompt1 = """
Extract person information from the text and return as JSON with fields:
- name (string)
- age (integer)
- occupation (string)

Text: "Alice Johnson is a 32-year-old software engineer."
"""

    print(prompt1)
    print("Expected: {\"name\": \"Alice Johnson\", \"age\": 32, \"occupation\": \"software engineer\"}")

    print("\n" + "=" * 60)
    print("\nApproach 2: Provide JSON template\n")

    prompt2 = """
Extract person information and fill this JSON template:

{
  "name": "",
  "age": 0,
  "occupation": ""
}

Text: "Alice Johnson is a 32-year-old software engineer."

JSON:
"""

    print(prompt2)

    print("\n" + "=" * 60)
    print("\nApproach 3: Schema specification\n")

    prompt3 = """
Extract information following this schema:

{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer", "minimum": 0},
    "occupation": {"type": "string"}
  },
  "required": ["name", "age", "occupation"]
}

Text: "Alice Johnson is a 32-year-old software engineer."

Respond with valid JSON only.
"""

    print(prompt3)

    print("\nBest Practices:")
    print("  • Specify 'valid JSON only' to avoid extra text")
    print("  • Provide type information (string, integer, array)")
    print("  • Show example of desired structure")
    print("  • Specify required vs optional fields")
    print("  • Handle null/missing values explicitly")

json_output_specification()
```

## Common Prompting Mistakes

### Ambiguity and Vagueness

```python
def ambiguity_mistakes():
    """Common mistakes from ambiguous prompts."""

    print("Common Prompting Mistakes:\n")

    print("MISTAKE 1: Vague Instructions\n")

    print("✗ Bad:")
    print('  "Tell me about AI"')
    print("  Problem: Too broad, unclear what aspect/depth")
    print()

    print("✓ Good:")
    print('  "Explain in 3 paragraphs how AI is used in healthcare"')
    print("  Better: Specific topic, defined scope, clear length")
    print()

    print("=" * 60)
    print("\nMISTAKE 2: Multiple Unclear Tasks\n")

    print("✗ Bad:")
    print('  "Analyze this and give me insights"')
    print("  Problem: 'Analyze' and 'insights' are vague")
    print()

    print("✓ Good:")
    print('  "Calculate the average, identify the highest value,')
    print('   and list any outliers from this dataset"')
    print("  Better: Specific, actionable tasks")
    print()

    print("=" * 60)
    print("\nMISTAKE 3: Assumed Context\n")

    print("✗ Bad:")
    print('  "Fix this code: [code]"')
    print("  Problem: What language? What's wrong? What should it do?")
    print()

    print("✓ Good:")
    print('  "This Python code should calculate factorial but returns')
    print('   wrong results. Fix the logic error: [code]"')
    print("  Better: Language, expected behavior, problem specified")
    print()

    print("=" * 60)
    print("\nMISTAKE 4: Conflicting Instructions\n")

    print("✗ Bad:")
    print('  "Be very detailed but keep it brief"')
    print("  Problem: Contradictory requirements")
    print()

    print("✓ Good:")
    print('  "Provide key details in 100-150 words"')
    print("  Better: Clear, consistent constraint")

ambiguity_mistakes()
```

### Complexity Issues

```python
def complexity_mistakes():
    """Mistakes from overly complex prompts."""

    print("\n\nComplexity Issues:\n")

    print("MISTAKE 5: Too Many Tasks at Once\n")

    print("✗ Bad:")
    bad_prompt = """
Read this article, summarize it, translate to Spanish, extract keywords,
identify sentiment, rate quality 1-10, suggest improvements, and format as JSON.
"""
    print(bad_prompt)
    print("Problem: Cognitive overload, easy to miss tasks")
    print()

    print("✓ Good: Break into separate prompts")
    print('  Prompt 1: "Summarize this article in 3 sentences"')
    print('  Prompt 2: "Translate this summary to Spanish: [summary]"')
    print('  Prompt 3: "Extract 5 keywords from: [summary]"')
    print("  Better: One clear task per prompt")
    print()

    print("=" * 60)
    print("\nMISTAKE 6: Overly Complex Instructions\n")

    print("✗ Bad:")
    bad_complex = """
Considering the semantic nuances and contextual embeddings inherent in
the lexical structure, perform a comprehensive multidimensional analysis
utilizing advanced heuristics to ascertain the underlying sentiment
modality while accounting for potential polysemy...
"""
    print(bad_complex)
    print("Problem: Unnecessarily complex language")
    print()

    print("✓ Good:")
    print('  "Classify the sentiment of this text as positive, negative,')
    print('   or neutral, considering sarcasm and context."')
    print("  Better: Clear, simple language")
    print()

    print("=" * 60)
    print("\nMISTAKE 7: Missing Critical Information\n")

    print("✗ Bad:")
    print('  "Classify these: [list of items]"')
    print("  Problem: No classification categories provided")
    print()

    print("✓ Good:")
    print('  "Classify these products as Electronics, Clothing, or Books:')
    print('   [list of items]"')
    print("  Better: Categories explicitly stated")

complexity_mistakes()
```

## Prompt Clarity and Precision

### Writing Clear Prompts

```python
def clear_prompt_writing():
    """Techniques for writing clear, precise prompts."""

    print("Writing Clear Prompts:\n")

    print("Clarity Checklist:\n")

    checklist = [
        ('One primary task', 'Focus on single objective'),
        ('Active voice', 'Use clear action verbs (Extract, Classify, Generate)'),
        ('Specific terms', 'Avoid vague words (good, nice, appropriate)'),
        ('Defined outputs', 'Specify format, length, structure'),
        ('Explicit constraints', 'State what to include/exclude'),
        ('Unambiguous', 'Could someone else understand this the same way?')
    ]

    for item, description in checklist:
        print(f"  ☐ {item}: {description}")

    print("\n" + "=" * 60)
    print("\nClarity Transformation Example:\n")

    print("BEFORE (Unclear):")
    unclear = """
Look at this customer feedback and tell me what you think.
Make it good and useful.
"""
    print(unclear)

    print("\nIssues:")
    print("  • 'Look at' - not a clear action")
    print("  • 'tell me what you think' - too vague")
    print("  • 'good and useful' - subjective, undefined")
    print()

    print("AFTER (Clear):")
    clear = """
Analyze the customer feedback below and provide:
1. Overall sentiment (positive/negative/neutral)
2. Top 3 mentioned issues
3. Suggested priority for each issue (high/medium/low)

Format as:
Sentiment: [sentiment]
Issues:
- [Issue 1]: [Priority]
- [Issue 2]: [Priority]
- [Issue 3]: [Priority]

Customer Feedback: [text]
"""
    print(clear)

    print("\nImprovements:")
    print("  ✓ Clear action: 'Analyze'")
    print("  ✓ Specific deliverables: sentiment + 3 issues + priorities")
    print("  ✓ Defined format: structured output")
    print("  ✓ Unambiguous: anyone would interpret this the same way")

clear_prompt_writing()
```

### Precision in Language

```python
def precision_in_language():
    """Using precise language in prompts."""

    print("\n\nPrecision in Language:\n")

    print("Replacing Vague Terms:\n")

    replacements = {
        'Vague: "some"': 'Precise: "3 to 5", "at least 2", "approximately 10"',
        'Vague: "brief"': 'Precise: "under 100 words", "2-3 sentences", "1 paragraph"',
        'Vague: "good"': 'Precise: "grammatically correct", "professional tone", "clear and concise"',
        'Vague: "analyze"': 'Precise: "identify patterns", "calculate average", "compare values"',
        'Vague: "important"': 'Precise: "critical errors", "high-priority items", "required fields"',
        'Vague: "relevant"': 'Precise: "related to topic X", "from date range Y", "in category Z"'
    }

    for vague, precise in replacements.items():
        print(f"  {vague}")
        print(f"    → {precise}")
        print()

    print("=" * 60)
    print("\nExample: Product Review Analysis\n")

    print("Imprecise:")
    imprecise = """
Read the reviews and tell me the important stuff.
"""
    print(f"  \"{imprecise.strip()}\"")
    print()

    print("Precise:")
    precise = """
Read the product reviews below and extract:
- Average rating (1-5 stars)
- Most frequently mentioned pros (top 3)
- Most frequently mentioned cons (top 3)
- Percentage of reviews mentioning 'durability'

Format as bullet points.
"""
    print(f"  {precise.strip()}")

precision_in_language()
```

## Testing and Validation

### Prompt Testing Strategies

```python
def prompt_testing():
    """How to test and validate prompts."""

    print("Prompt Testing Strategies:\n")

    print("1. EDGE CASE TESTING\n")
    print("Test with:")
    test_cases = [
        'Empty input',
        'Very long input',
        'Malformed input',
        'Ambiguous input',
        'Input with special characters',
        'Input in wrong language',
        'Minimal input',
        'Maximum input'
    ]

    for case in test_cases:
        print(f"  • {case}")

    print("\n2. VARIATION TESTING\n")
    print("Run same prompt with:")
    variations = [
        'Different examples (same task)',
        'Different phrasings of instruction',
        'Different ordering of components',
        'With and without context',
        'Various temperature settings'
    ]

    for variation in variations:
        print(f"  • {variation}")

    print("\n3. CONSISTENCY TESTING\n")
    print("Check if:")
    consistency_checks = [
        'Same input → similar outputs (reproducibility)',
        'Similar inputs → consistent patterns',
        'Output format matches specification',
        'All constraints are respected',
        'Edge cases handled gracefully'
    ]

    for check in consistency_checks:
        print(f"  • {check}")

    print("\n4. PERFORMANCE METRICS\n")
    metrics = {
        'Accuracy': 'Correct outputs / total attempts',
        'Consistency': 'Variance in outputs for same input',
        'Format compliance': 'Outputs matching format specification',
        'Constraint adherence': 'Percentage respecting all constraints',
        'Token efficiency': 'Output quality per token used'
    }

    for metric, definition in metrics.items():
        print(f"  • {metric}: {definition}")

prompt_testing()
```

### Iterative Refinement

```python
def iterative_refinement():
    """Process for iteratively improving prompts."""

    print("\n\nIterative Refinement Process:\n")

    print("Step 1: Initial Prompt")
    print("─" * 60)
    initial = "Summarize this article."
    print(initial)
    print("\nResult: Too vague, inconsistent length\n")

    print("Step 2: Add Length Constraint")
    print("─" * 60)
    v2 = "Summarize this article in 2-3 sentences."
    print(v2)
    print("\nResult: Better, but sometimes misses key points\n")

    print("Step 3: Add Content Guidance")
    print("─" * 60)
    v3 = "Summarize this article in 2-3 sentences, focusing on the main argument and key evidence."
    print(v3)
    print("\nResult: More relevant, but format varies\n")

    print("Step 4: Add Format Specification")
    print("─" * 60)
    v4 = """
Summarize this article in exactly 3 sentences:
1. First sentence: Main argument
2. Second sentence: Key supporting evidence
3. Third sentence: Conclusion or implications

Article: [text]

Summary:
"""
    print(v4)
    print("Result: Consistent, comprehensive, well-structured ✓")

    print("\n" + "=" * 60)
    print("\nRefinement Workflow:\n")

    workflow = [
        ('Test', 'Run prompt with diverse inputs'),
        ('Identify issues', 'What fails? What\'s inconsistent?'),
        ('Hypothesize cause', 'Why is it failing?'),
        ('Refine', 'Add constraints, context, or examples'),
        ('Retest', 'Verify improvement'),
        ('Repeat', 'Until acceptable performance')
    ]

    for step, description in workflow:
        print(f"  {step} → {description}")

    print("\nCommon Refinements:")
    print("  • Add examples when task is unclear")
    print("  • Add constraints when output varies too much")
    print("  • Add context when domain knowledge needed")
    print("  • Simplify when prompt is too complex")
    print("  • Split when doing too many tasks")

iterative_refinement()
```

### Debugging Prompts

```python
def debugging_prompts():
    """Debugging common prompt problems."""

    print("\n\nDebugging Prompt Problems:\n")

    problems = {
        'Wrong output format': {
            'symptom': 'Model returns text instead of JSON',
            'cause': 'Format not clearly specified or examples missing',
            'fix': 'Add explicit format specification and example',
            'example': 'Add: "Return ONLY valid JSON, no other text"'
        },
        'Missing information': {
            'symptom': 'Output incomplete or skips requested items',
            'cause': 'Too many tasks or unclear priority',
            'fix': 'Break into smaller prompts or use numbered list',
            'example': 'Use: "1. First do X, 2. Then do Y, 3. Finally Z"'
        },
        'Inconsistent results': {
            'symptom': 'Same input gives different outputs',
            'cause': 'Ambiguous instructions or high temperature',
            'fix': 'Add clarity, examples, or lower temperature',
            'example': 'Reduce temperature from 0.7 to 0.2'
        },
        'Ignoring constraints': {
            'symptom': 'Output violates specified constraints',
            'cause': 'Constraints buried in text or unclear',
            'fix': 'Make constraints explicit and prominent',
            'example': 'Use: "IMPORTANT: Do not exceed 100 words"'
        },
        'Hallucinating': {
            'symptom': 'Model makes up facts or information',
            'cause': 'No grounding data or asked to speculate',
            'fix': 'Provide source data or instruct to say "unknown"',
            'example': 'Add: "If information not in text, respond: Unknown"'
        }
    }

    for problem, info in problems.items():
        print(f"{problem.upper()}:")
        print(f"  Symptom: {info['symptom']}")
        print(f"  Cause: {info['cause']}")
        print(f"  Fix: {info['fix']}")
        print(f"  Example: {info['example']}")
        print()

    print("Debugging Checklist:")
    print("  ☐ Is the task clearly stated?")
    print("  ☐ Are all necessary components present?")
    print("  ☐ Are instructions unambiguous?")
    print("  ☐ Is the output format specified?")
    print("  ☐ Are constraints explicit and visible?")
    print("  ☐ Have you tested with multiple inputs?")
    print("  ☐ Have you tried simplifying?")

debugging_prompts()
```

## Summary

**Key Concepts**:

1. **Prompts are the interface** to LLMs - quality of prompts determines quality of outputs
2. **Prompt anatomy**: Instruction + Context + Input + Examples + Constraints + Output Format
3. **Zero-shot** works for simple, well-known tasks; complex tasks need few-shot or decomposition
4. **Clarity is critical** - be specific, use action verbs, avoid ambiguity
5. **Context matters** - provide relevant background (audience, purpose, domain, tone)
6. **Constraints guide behavior** - explicitly state what to do/avoid, format requirements, scope
7. **Test and iterate** - refine prompts based on results, test edge cases

**Prompt Structure Template**:

```
[INSTRUCTION] - What to do (specific action verb)
[CONTEXT] - Background information (audience, purpose, domain)
[EXAMPLES] - Demonstrations (if few-shot)
[INPUT] - Data to process
[CONSTRAINTS] - Rules and limitations
[OUTPUT FORMAT] - How to structure response
```

**Best Practices**:

| Principle   | Do                                 | Don't                 |
| ----------- | ---------------------------------- | --------------------- |
| Clarity     | "Summarize in 3 sentences"         | "Tell me about this"  |
| Specificity | "Extract person names and titles"  | "Extract information" |
| Format      | "Return as JSON: {name, age}"      | "Give me the data"    |
| Constraints | "Max 100 words, professional tone" | "Keep it appropriate" |
| Action      | "Classify as spam/not spam"        | "What do you think?"  |

**Common Mistakes to Avoid**:

- ❌ Vague instructions ("analyze this", "make it better")
- ❌ Multiple unclear tasks in one prompt
- ❌ Assumed context that model doesn't have
- ❌ Conflicting or contradictory requirements
- ❌ Overly complex language
- ❌ Missing output format specification
- ❌ No testing or validation

**Component Priority**:

```
Essential:
✓ Clear instruction (what to do)
✓ Input data (what to process)

Important:
✓ Output format (structured responses)
✓ Constraints (guide behavior)

Helpful:
✓ Context (improve relevance)
✓ Examples (complex tasks)
```

**Quick Checklist**:

```
Before sending a prompt, verify:
☐ Task is clearly stated (action verb + specific goal)
☐ Context provided if needed
☐ Output format specified
☐ Constraints listed explicitly
☐ Instruction is unambiguous
☐ Tested with sample inputs
☐ Expected output defined
```

**Refinement Process**:

```
Initial Prompt → Test → Identify Issues → Refine → Retest → Deploy
                  ↑                                           ↓
                  ←───────────── Iterate if needed ──────────┘
```

**When to Add Components**:

- **Examples**: Task is novel, ambiguous, or format-specific
- **Context**: Domain-specific, multiple interpretations possible
- **Constraints**: Output too variable, need consistency
- **Format spec**: Programmatic parsing required
- **Decomposition**: Task complex, multiple steps needed

## Next Steps

- Study [Few-Shot Learning and Examples](few-shot-learning.md) for using demonstrations effectively
- Learn [Chain-of-Thought Prompting](cot-prompting.md) for complex reasoning tasks
- Explore [Structured Output and Formatting](structured-output.md) for parsable responses
- Practice [Prompt Optimization and Iteration](prompt-optimization.md) for systematic improvement
- Master [Advanced Prompting Patterns](advanced-patterns.md) for sophisticated applications
- Review [LLM Concepts](../llm-concepts/instruction-tuning.md) to understand model behavior
- Apply to [Application Patterns](../application_patterns/best-practices.md) for production systems
