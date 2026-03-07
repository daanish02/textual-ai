# Advanced Prompting Patterns

## Table of Contents

- [Introduction](#introduction)
- [Prompt Decomposition](#prompt-decomposition)
- [Self-Criticism and Refinement](#self-criticism-and-refinement)
- [Role Prompting](#role-prompting)
- [Meta-Prompting](#meta-prompting)
- [Constitutional Prompting](#constitutional-prompting)
- [Multi-Turn Conversation Patterns](#multi-turn-conversation-patterns)
- [Prompt Chaining](#prompt-chaining)
- [Recursive Prompting](#recursive-prompting)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Advanced prompting patterns** are sophisticated techniques that go beyond basic instructions and examples. These patterns leverage complex reasoning, multi-step processing, and self-improvement to tackle tasks that single prompts cannot handle effectively.

```
Simple prompt:
"Write a blog post about AI"

Advanced pattern (decomposition + self-criticism):
Step 1: "Generate 5 blog post outlines about AI"
Step 2: "Evaluate each outline for clarity and engagement"
Step 3: "Select best outline and expand into full post"
Step 4: "Review post for accuracy and tone, suggest improvements"
Step 5: "Apply improvements and finalize"

Result: Higher quality, more controlled output
```

**When to use advanced patterns**:

- Single prompts don't achieve desired quality
- Task requires multiple perspectives or iterations
- Need explicit reasoning or verification
- Complex tasks requiring planning
- Quality control is critical

This guide covers powerful patterns for handling complex prompting challenges.

## Prompt Decomposition

### Breaking Complex Tasks Into Steps

```python
def prompt_decomposition():
    """Decomposing complex tasks into sequential subtasks."""

    print("Prompt Decomposition Pattern:\n")

    print("Concept: Break complex task into manageable steps\n")

    print("Example: Research Report Generation\n")
    print("=" * 60)

    print("\nSingle Prompt (struggles):")
    print('"Write comprehensive research report on climate change"')
    print("→ Too complex, lacks structure, superficial\n")

    print("Decomposed Approach (succeeds):")

    steps = [
        "1. List 10 key climate change impacts",
        "2. Categorize into: environmental, economic, social",
        "3. Analyze causes of top 3 impacts",
        "4. Propose 2-3 solutions per cause",
        "5. Evaluate solution feasibility",
        "6. Create policy recommendations",
        "7. Synthesize into coherent structure",
        "8. Polish language and transitions"
    ]

    for step in steps:
        print(f"  {step}")

    print("\nBenefits:")
    print("  ✓ Each step is manageable")
    print("  ✓ Can validate between steps")
    print("  ✓ Better quality output")
    print("  ✓ More control over process")

prompt_decomposition()
```

### Implementation

```python
def decomposition_implementation():
    """Implementing decomposed prompts."""

    print("\n\nDecomposition Implementation:\n")

    code = '''
def execute_decomposed_task(steps: List[str], initial_input: str) -> str:
    """Execute task broken into steps."""

    context = initial_input
    outputs = []

    for i, step_template in enumerate(steps, 1):
        print(f"Executing step {i}...")

        # Build prompt with context from previous steps
        prompt = step_template.format(
            input=context,
            previous_outputs='\\n'.join(outputs)
        )

        # Execute step
        output = llm_call(prompt)
        outputs.append(output)

        # Context for next step
        context = output

    return outputs[-1]

# Example: Content creation pipeline
steps = [
    "Generate 5 article outlines about: {input}",
    "Evaluate each outline for engagement and clarity. Select best one: {input}",
    "Expand selected outline into full article (800 words): {input}",
    "Review article for accuracy, tone, and flow. List improvements: {input}",
    "Apply improvements and finalize article: {input}"
]

result = execute_decomposed_task(steps, "AI in healthcare")
'''

    print(code)

decomposition_implementation()
```

## Self-Criticism and Refinement

### Generate-Critique-Refine Pattern

```python
def critique_refine_pattern():
    """Self-criticism and iterative refinement."""

    print("Generate-Critique-Refine Pattern:\n")

    print("Process: Generate → Critique → Refine → (Repeat)\n")

    print("=" * 60)
    print("\nExample: Essay Writing\n")

    print("Step 1 - Generate:")
    print('  Prompt: "Write 300-word essay on renewable energy"')
    print("  Output: [Initial draft]\n")

    print("Step 2 - Critique:")
    print('  Prompt: "Review this essay and identify 5 specific weaknesses:')
    print('           [Initial draft]"')
    print("  Output: 1. Weak introduction")
    print("          2. Missing evidence in paragraph 2")
    print("          3. Abrupt transition")
    print("          4. Brief conclusion")
    print("          5. Complex sentences\n")

    print("Step 3 - Refine:")
    print('  Prompt: "Rewrite essay addressing these issues:')
    print('           [Initial draft] [Critique]"')
    print("  Output: [Improved draft]\n")

    print("Optional: Repeat steps 2-3 for further improvement")

    print("\n" + "=" * 60)
    print("\nImplementation:\n")

    code = '''
def critique_and_refine(task: str, iterations: int = 2) -> str:
    """Iterative critique and refinement."""

    # Generate initial version
    initial_prompt = f"Write: {task}"
    current = llm_call(initial_prompt)

    for i in range(iterations):
        # Critique
        critique_prompt = f"""
        Review this and identify specific weaknesses:

        {current}

        List 3-5 concrete issues to fix.
        """
        critique = llm_call(critique_prompt)

        # Refine
        refine_prompt = f"""
        Improve this by addressing the critique:

        Original: {current}
        Critique: {critique}

        Rewrite to fix all issues.
        """
        current = llm_call(refine_prompt)

    return current

essay = critique_and_refine("Essay on renewable energy", iterations=2)
'''

    print(code)

critique_refine_pattern()
```

### Multi-Perspective Critique

```python
def multi_perspective():
    """Critiquing from multiple viewpoints."""

    print("\n\nMulti-Perspective Critique:\n")

    print("Concept: Evaluate from different expert viewpoints\n")

    example = '''
Task: Review product description

Marketing Perspective:
  • Lacks emotional appeal
  • Benefits not highlighted
  • Weak call-to-action
  • Too technical

Technical Writer Perspective:
  • Specifications clear
  • Missing setup instructions
  • Needs compatibility info
  • Could use diagrams

Customer Perspective:
  • Want use cases
  • How does it solve my problem?
  • Price vs alternatives?
  • Worth the cost?

Synthesize Feedback:
  1. Balance technical detail with benefits
  2. Add real-world use cases
  3. Clear call-to-action
  4. Include setup guidance
  5. Address value proposition

Refined Version:
  [Improved description incorporating all perspectives]
'''

    print(example)

multi_perspective()
```

## Role Prompting

### Assigning Expert Roles

```python
def role_prompting():
    """Using role-based prompts for better outputs."""

    print("Role Prompting:\n")

    print("Concept: Assign model a specific expert persona\n")

    print("=" * 60)
    print("\nComparison:\n")

    print("Generic:")
    print('  "Explain quantum computing"')
    print("  → Surface-level, generic\n")

    print("With Role:")
    print('  "You are a quantum physics professor teaching undergrads.')
    print('   Explain quantum computing with appropriate depth for')
    print('   students who know basic physics."')
    print("  → Targeted, appropriate depth\n")

    print("=" * 60)
    print("\nEffective Roles:\n")

    roles = {
        'Expert Consultant': 'Authoritative, comprehensive advice',
        'Patient Teacher': 'Clear, simple, example-rich',
        'Thoughtful Critic': 'Analytical, balanced feedback',
        'Creative Writer': 'Engaging, stylistic, narrative',
        'Technical Documenter': 'Precise, structured, complete'
    }

    for role, effect in roles.items():
        print(f"  • {role}: {effect}")

    print("\n" + "=" * 60)
    print("\nRole Prompt Template:\n")

    template = '''
You are a {role} with expertise in {domain}.

Your task: {task}

Your response should:
- {characteristic_1}
- {characteristic_2}
- {characteristic_3}

{specific_instructions}
'''

    print(template)

role_prompting()
```

## Meta-Prompting

### Prompts That Generate Prompts

```python
def meta_prompting():
    """Using models to generate and improve prompts."""

    print("Meta-Prompting:\n")

    print("Concept: Use LLM to create or improve prompts\n")

    print("=" * 60)
    print("\nExample: Generate Task-Specific Prompt\n")

    print("Meta-Prompt:")
    meta = '''
I need a prompt for this task:

Task: Classify support tickets into categories
Categories: Technical, Billing, Feature Request, Complaint, Praise
Requirements: High accuracy, handle ambiguity, return JSON

Generate an effective prompt including:
1. Clear instructions
2. 3-5 diverse examples
3. Output format spec
4. Edge case handling
'''

    print(meta)

    print("\nGenerated Prompt:")
    generated = '''
Classify customer support tickets into one category:
- Technical Issue: Product functionality problems
- Billing Question: Payment or subscription inquiries
- Feature Request: Suggestions for improvements
- Complaint: Dissatisfaction expressed
- Praise: Positive feedback

Examples:

"App crashes when exporting" → Technical Issue
"Charged twice this month" → Billing Question
"Please add dark mode" → Feature Request
"Customer service is terrible" → Complaint
"Love the new update!" → Praise

For ambiguous tickets, choose primary concern.
Return: {"category": "...", "confidence": "high|medium|low"}

Ticket: {input}
'''

    print(generated)

    print("\nMeta-Prompting Uses:")
    print("  • Generate prompts for new tasks")
    print("  • Improve existing prompts")
    print("  • Create few-shot examples")
    print("  • Design output formats")
    print("  • Generate evaluation criteria")

meta_prompting()
```

## Constitutional Prompting

### Principle-Based Behavior

```python
def constitutional_prompting():
    """Guiding model behavior with explicit principles."""

    print("Constitutional Prompting:\n")

    print("Concept: Define principles that govern responses\n")

    print("=" * 60)
    print("\nExample: Content Moderation\n")

    constitutional = '''
You are a content moderator. Follow these principles:

PRINCIPLES:
1. Harm Prevention: Don't provide harmful information
2. Fairness: Treat all groups equally, avoid bias
3. Accuracy: State only verified facts
4. Privacy: Protect personal information
5. Transparency: Explain your reasoning
6. Proportionality: Match response to severity

PROCESS:
1. Identify potential issues
2. Map to relevant principles
3. Assess severity
4. Recommend action
5. Explain reasoning

Example:
Content: [Post with unverified medical claim]
Analysis:
- Issue: Unverified medical advice
- Principle: Accuracy (#3)
- Severity: Medium (health misinformation)
- Action: Flag for review
- Reasoning: Medical claims need verification to prevent harm
'''

    print(constitutional)

    print("\nBenefits:")
    print("  ✓ Explicit value system")
    print("  ✓ Consistent decisions")
    print("  ✓ Auditable reasoning")
    print("  ✓ Handles ambiguity better")
    print("  ✓ Reduces harmful outputs")

constitutional_prompting()
```

## Multi-Turn Conversation Patterns

### Stateful Conversations

```python
def conversation_management():
    """Managing state across conversation turns."""

    print("Multi-Turn Conversation Management:\n")

    print("Challenge: Maintain context and coherence\n")

    print("Solution: Explicit state tracking\n")

    code = '''
class StatefulConversation:
    """Manages conversation with explicit state."""

    def __init__(self, system_prompt: str):
        self.system = system_prompt
        self.state = "GATHERING_INFO"
        self.collected_info = {}
        self.history = []

    def process_turn(self, user_input: str) -> str:
        """Process one conversation turn."""

        # Build prompt with state context
        prompt = f"""
        {self.system}

        Current state: {self.state}
        Collected info: {self.collected_info}

        Recent conversation:
        {self._format_history()}

        User: {user_input}
        Assistant:
        """

        response = llm_call(prompt)

        # Update state
        self._update_state(user_input, response)

        # Save to history
        self.history.append({
            'user': user_input,
            'assistant': response,
            'state': self.state
        })

        return response

    def _update_state(self, user_input, response):
        """Transition to next state based on interaction."""
        # State machine logic here
        pass

# Usage
bot = StatefulConversation("You are a helpful troubleshooting assistant")

bot.process_turn("My code doesn't work")  # State: GATHERING_INFO
bot.process_turn("It gives TypeError")    # State: ANALYZING
bot.process_turn("How do I fix it?")      # State: PROVIDING_SOLUTION
'''

    print(code)

conversation_management()
```

### Clarification Loops

```python
def clarification_loops():
    """Handling ambiguity with clarification."""

    print("\n\nClarification Loops:\n")

    print("Pattern: Ask instead of guessing\n")

    example = '''
User: "Make it faster"

Poor Response (guesses):
"I'll optimize the database queries..."
(Assumes what "it" is)

Good Response (clarifies):
"I'd be happy to help optimize. To assist you better:
 1. What needs to be faster? (page load, processing, etc.)
 2. Current performance?
 3. Target performance?
 4. Any constraints?"

User: "The data processing, takes 2 hours, need under 30 min"

Now Response (informed):
"To reduce processing from 2 hours to under 30 minutes:
 1. Parallelize operations
 2. Optimize data structures
 3. Add caching

 Which approach would you like to start with?"
'''

    print(example)

    print("\nWhen to Clarify:")
    print("  • Ambiguous references")
    print("  • Missing key information")
    print("  • Multiple interpretations")
    print("  • High-stakes decisions")
    print("  • Unclear requirements")

clarification_loops()
```

## Prompt Chaining

### Sequential Processing

```python
def prompt_chaining():
    """Chaining prompts for complex workflows."""

    print("Prompt Chaining:\n")

    print("Concept: Output of one prompt feeds into next\n")

    print("=" * 60)
    print("\nExample: Content Creation Pipeline\n")

    chain = '''
Prompt 1: Ideation
"Generate 10 blog post ideas about AI in healthcare"
Output: [List of 10 ideas]
           ↓
Prompt 2: Selection
"Rank these ideas by potential engagement: [ideas]"
Output: [Ranked list]
           ↓
Prompt 3: Outlining
"Create detailed outline for top idea: [idea]"
Output: [Outline with sections]
           ↓
Prompt 4: Expansion
"Write introduction section: [outline]"
Output: [Introduction text]
           ↓
Prompt 5: Review
"Review introduction for clarity and engagement: [text]"
Output: [Feedback]
           ↓
Prompt 6: Refinement
"Revise introduction based on feedback: [text] [feedback]"
Output: [Final introduction]
'''

    print(chain)

    print("\n" + "=" * 60)
    print("\nImplementation:\n")

    code = '''
class PromptChain:
    """Execute chain of prompts."""

    def __init__(self, steps: List[Dict]):
        self.steps = steps
        self.outputs = {}

    def execute(self, initial_input: str) -> str:
        """Run the full chain."""

        current_input = initial_input

        for i, step in enumerate(self.steps):
            # Build prompt with accumulated context
            prompt = step['template'].format(
                input=current_input,
                **self.outputs  # All previous outputs available
            )

            # Execute
            output = llm_call(prompt)

            # Store output
            self.outputs[step['name']] = output

            # Pass to next step
            current_input = output

        return current_input

# Usage
steps = [
    {'name': 'ideas', 'template': 'Generate 10 ideas: {input}'},
    {'name': 'ranked', 'template': 'Rank by engagement: {input}'},
    {'name': 'outline', 'template': 'Outline for top idea: {input}'},
    {'name': 'draft', 'template': 'Write full post: {input}'},
]

chain = PromptChain(steps)
result = chain.execute("AI in healthcare")
'''

    print(code)

prompt_chaining()
```

## Recursive Prompting

### Self-Referential Patterns

```python
def recursive_prompting():
    """Recursive prompting for iterative refinement."""

    print("Recursive Prompting:\n")

    print("Concept: Prompt calls itself until condition met\n")

    print("=" * 60)
    print("\nExample: Iterative Code Improvement\n")

    recursive = '''
Prompt:
"Review this code and suggest ONE improvement:

{code}

If code is perfect, respond with 'DONE'.
Otherwise, provide:
1. Issue found
2. Improved code

Current iteration: {iteration}"

Process:
Iteration 1: Find issue → Improve → Recurse
Iteration 2: Find issue → Improve → Recurse
Iteration 3: Find issue → Improve → Recurse
Iteration 4: No issues found → "DONE"
'''

    print(recursive)

    print("\n" + "=" * 60)
    print("\nImplementation:\n")

    code = '''
def recursive_improve(
    content: str,
    improve_prompt: str,
    max_iterations: int = 5
) -> str:
    """Recursively improve until 'DONE' or max iterations."""

    for i in range(max_iterations):
        prompt = improve_prompt.format(
            content=content,
            iteration=i+1
        )

        response = llm_call(prompt)

        # Check termination
        if "DONE" in response or "perfect" in response.lower():
            print(f"Converged after {i+1} iterations")
            return content

        # Extract improved version
        content = extract_improved_content(response)
        print(f"Iteration {i+1}: Improvement applied")

    print(f"Max iterations ({max_iterations}) reached")
    return content

# Usage
code_to_improve = """
def process(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""

improved = recursive_improve(
    code_to_improve,
    "Review and suggest ONE improvement: {content}",
    max_iterations=5
)
'''

    print(code)

    print("\nUse Cases:")
    print("  • Iterative code improvement")
    print("  • Text refinement until quality threshold")
    print("  • Problem solving with attempts")
    print("  • Progressive simplification")

recursive_prompting()
```

## Summary

**Key Concepts**:

1. **Prompt Decomposition**: Break complex tasks into sequential steps
2. **Self-Criticism**: Generate → Critique → Refine for higher quality
3. **Role Prompting**: Assign expert personas for appropriate tone/depth
4. **Meta-Prompting**: Use LLMs to generate and improve prompts
5. **Constitutional Prompting**: Guide behavior with explicit principles
6. **Multi-Turn Patterns**: Manage state across conversation turns
7. **Prompt Chaining**: Sequential processing through pipeline
8. **Recursive Prompting**: Iterative improvement until convergence

**When to Use Advanced Patterns**:

```
Use when:
  ✓ Single prompts insufficient
  ✓ Need quality control
  ✓ Complex multi-step tasks
  ✓ Require explicit reasoning
  ✓ Managing conversations
  ✓ Production systems

Don't use when:
  ✗ Simple tasks
  ✗ Cost-sensitive
  ✗ Latency critical
  ✗ Overkill for requirements
```

**Pattern Selection Guide**:

| Pattern        | Best For                     | Complexity | Cost      |
| -------------- | ---------------------------- | ---------- | --------- |
| Decomposition  | Complex multi-step tasks     | Medium     | Medium    |
| Self-Criticism | Quality-critical content     | Medium     | High      |
| Role Prompting | Audience-specific output     | Low        | Low       |
| Meta-Prompting | Prompt creation/improvement  | Low        | Medium    |
| Constitutional | Value-aligned behavior       | Medium     | Medium    |
| Multi-Turn     | Conversations, clarification | High       | High      |
| Chaining       | Sequential workflows         | Medium     | High      |
| Recursive      | Iterative refinement         | High       | Very High |

**Decomposition Pattern**:

```
Complex Task → Step 1 → Step 2 → ... → Final Output

Example: Research Report
  1. List key points
  2. Categorize points
  3. Analyze top items
  4. Propose solutions
  5. Synthesize report
  6. Polish language
```

**Self-Criticism Pattern**:

```
Generate → Critique → Refine → (Repeat)

Typical gain: +20-40% quality
Iterations: 1-3 usually sufficient
```

**Role Prompting Template**:

```
You are a {role} with expertise in {domain}.

Task: {task}

Your response should:
- {characteristic_1}
- {characteristic_2}

{specific_instructions}
```

**Constitutional Principles Example**:

```
1. Harm Prevention: No harmful content
2. Accuracy: Only verified facts
3. Fairness: Unbiased treatment
4. Privacy: Protect PII
5. Transparency: Explain reasoning

Process: Identify issue → Map to principle → Assess → Decide → Explain
```

**Prompt Chaining Structure**:

```python
# Chain execution
input_0 = initial_input
output_1 = prompt_1(input_0)
output_2 = prompt_2(output_1)
output_3 = prompt_3(output_2)
final = output_3

# Each prompt specialized for its step
```

**Multi-Turn Conversation**:

```
Key elements:
- State tracking (what stage of conversation)
- Context accumulation (history)
- Transition logic (when to move states)
- Clarification loops (when ambiguous)
```

**Best Practices**:

✓ **Start simple** - Use advanced patterns only when needed  
✓ **Measure impact** - Validate improvements justify complexity  
✓ **Cost-aware** - Advanced patterns use more tokens  
✓ **Fail gracefully** - Handle when patterns don't converge  
✓ **Document** - Explain pattern choice for maintainability  
✓ **Combine patterns** - Mix approaches for complex needs  
✓ **Version control** - Track pattern variations

**Common Combinations**:

- **Decomposition + Self-Criticism**: Multi-step with quality checks
- **Role + Constitutional**: Expert persona with value alignment
- **Chaining + Decomposition**: Pipeline of subtasks
- **Meta + Optimization**: Auto-generate and test prompts
- **Multi-Turn + Clarification**: Conversational workflows

**Performance Impact**:

| Pattern                 | Quality Gain | Cost Multiplier | Latency |
| ----------------------- | ------------ | --------------- | ------- |
| Decomposition (5 steps) | +15-30%      | 3-5x            | 3-5x    |
| Self-Criticism (2 iter) | +20-40%      | 3x              | 3x      |
| Role Prompting          | +5-15%       | 1.1x            | 1.1x    |
| Chaining (4 steps)      | +25-45%      | 4x              | 4x      |
| Recursive (avg 3 iter)  | +30-50%      | 3-4x            | 3-4x    |

**Anti-Patterns to Avoid**:

❌ Over-engineering simple tasks  
❌ Too many recursive iterations (expensive, diminishing returns)  
❌ Chaining without validation between steps  
❌ Role drift in long conversations  
❌ Circular dependencies in chains  
❌ Ignoring convergence in recursive patterns

## Next Steps

- Combine with [Chain-of-Thought](cot-prompting.md) for reasoning-heavy patterns
- Use [Structured Output](structured-output.md) for reliable data exchange in chains
- Apply [Prompt Optimization](prompt-optimization.md) to tune advanced patterns
- Implement in [Application Patterns](../application_patterns/) for production
- Study [Tool Use](../agentic-ai-lab/tool-use/) for agent-based implementations
- Explore [Multi-Agent Systems](../agentic-ai-lab/multi-agent-systems/) for complex workflows
- Review [Orchestration Patterns](../agentic-ai-lab/orchestration_patterns/) for coordination
