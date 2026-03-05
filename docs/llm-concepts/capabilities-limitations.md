# Capabilities and Limitations

## Table of Contents

- [Introduction](#introduction)
- [What LLMs Can Do Well](#what-llms-can-do-well)
- [Knowledge and Factual Recall](#knowledge-and-factual-recall)
- [Reasoning Capabilities](#reasoning-capabilities)
- [Language Understanding and Generation](#language-understanding-and-generation)
- [Common Failure Modes](#common-failure-modes)
- [Arithmetic and Symbolic Reasoning](#arithmetic-and-symbolic-reasoning)
- [Hallucination](#hallucination)
- [Context and Memory Limitations](#context-and-memory-limitations)
- [Robustness and Consistency](#robustness-and-consistency)
- [Safety and Alignment Challenges](#safety-and-alignment-challenges)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Understanding what large language models can and cannot do** is critical for building reliable applications. LLMs exhibit impressive capabilities but also systematic failures that developers must account for.

```
LLM Capabilities Spectrum:

Excellent:                 Good:                    Limited:
• Text generation         • Summarization          • Precise arithmetic
• Pattern matching        • Translation            • Long-term memory
• Style mimicry           • Question answering     • Logical consistency
• Code completion         • Commonsense reasoning  • Factual accuracy
• Creative writing        • Entity extraction      • Planning ahead
```

**Key insight**: LLMs are powerful **pattern matching and generation systems** trained on text, not reasoning engines with verified knowledge. They excel at tasks similar to their training distribution but struggle with precise logic and novel situations.

```
Mental Model:

LLM = Sophisticated Pattern Matcher + Text Generator
    ≠ Knowledge Database
    ≠ Reasoning Engine
    ≠ Calculator
    ≠ Search Engine
```

This guide provides a realistic assessment of LLM capabilities and limitations, with examples of successes and failures.

## What LLMs Can Do Well

### Text Generation and Completion

```python
def text_generation_strengths():
    """Tasks where LLMs excel at text generation."""
    
    strengths = {
        'Creative writing': {
            'tasks': ['Stories, poems, scripts, song lyrics'],
            'quality': 'High - often indistinguishable from human',
            'reliability': '90%+',
            'example': 'Write a short story about a robot learning to paint'
        },
        'Code generation': {
            'tasks': ['Functions, scripts, boilerplate code'],
            'quality': 'High for common patterns, good for novel',
            'reliability': '70-85%',
            'example': 'Write a Python function to reverse a linked list'
        },
        'Text completion': {
            'tasks': ['Finish sentences, paragraphs, documents'],
            'quality': 'Very high - core training objective',
            'reliability': '95%+',
            'example': 'Complete: "The key to machine learning is..."'
        },
        'Style mimicry': {
            'tasks': ['Write in style of X, formal/casual tone'],
            'quality': 'Excellent - captures style nuances',
            'reliability': '85%+',
            'example': 'Write like Shakespeare, like a lawyer, like a child'
        },
        'Content drafting': {
            'tasks': ['Emails, reports, articles, proposals'],
            'quality': 'Good - useful first drafts',
            'reliability': '80%+',
            'example': 'Draft a professional email declining a meeting'
        }
    }
    
    print("Text Generation Strengths:\n")
    for strength, info in strengths.items():
        print(f"{strength.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Why LLMs excel:")
    print("  • Trained specifically on text generation")
    print("  • Billions of examples of good text")
    print("  • Captures patterns, style, structure")
    print("  • Strong at interpolation within training distribution")

text_generation_strengths()
```

### Pattern Recognition and Classification

```python
def pattern_recognition_strengths():
    """Tasks where LLMs excel at pattern recognition."""
    
    print("\n\nPattern Recognition Strengths:\n")
    
    tasks = {
        'Sentiment analysis': {
            'accuracy': '85-95%',
            'why': 'Clear linguistic patterns for sentiment',
            'example': '"This movie was terrible" → Negative'
        },
        'Entity extraction': {
            'accuracy': '80-90%',
            'why': 'Named entities have recognizable patterns',
            'example': 'Extract: "Apple Inc. CEO Tim Cook" → [Apple Inc., Tim Cook]'
        },
        'Text classification': {
            'accuracy': '85-92%',
            'why': 'Strong at categorizing based on content',
            'example': 'Classify news article: Sports, Politics, Tech, etc.'
        },
        'Intent detection': {
            'accuracy': '80-90%',
            'why': 'Can identify user goals from phrasing',
            'example': '"Book a flight to NYC" → Intent: flight_booking'
        },
        'Spam detection': {
            'accuracy': '90-95%',
            'why': 'Spam has distinctive linguistic markers',
            'example': 'Identify phishing emails, spam messages'
        },
        'Language detection': {
            'accuracy': '95-99%',
            'why': 'Languages have distinct character/word patterns',
            'example': 'Detect: "Bonjour" → French'
        }
    }
    
    for task, info in tasks.items():
        print(f"{task.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

pattern_recognition_strengths()
```

### Natural Language Understanding

```python
def nlu_strengths():
    """Natural language understanding capabilities."""
    
    print("\n\nNatural Language Understanding:\n")
    
    capabilities = [
        ('Semantic similarity', 'Identify similar meanings across different phrasings',
         'Example: "car" ≈ "automobile", "big" ≈ "large"'),
        
        ('Paraphrasing', 'Rephrase text while preserving meaning',
         'Example: "It\'s raining" → "There\'s precipitation"'),
        
        ('Context understanding', 'Use surrounding text to disambiguate',
         'Example: "bank" (river) vs "bank" (financial) from context'),
        
        ('Implied meaning', 'Understand implications and subtext',
         'Example: "Can you pass the salt?" = request, not yes/no question'),
        
        ('Pronoun resolution', 'Identify what pronouns refer to',
         'Example: "Alice told Bob she was leaving" (she = Alice)'),
        
        ('Commonsense reasoning', 'Apply world knowledge to understand text',
         'Example: "The trophy didn\'t fit in the suitcase because it was too big"'),
    ]
    
    for capability, description, example in capabilities:
        print(f"{capability.upper()}:")
        print(f"  Description: {description}")
        print(f"  {example}")
        print()
    
    print("Strength: LLMs have absorbed vast amounts of language")
    print("They understand syntax, semantics, pragmatics reasonably well")

nlu_strengths()
```

## Knowledge and Factual Recall

### What LLMs Know

```python
def llm_knowledge_domains():
    """Domains where LLMs have knowledge."""
    
    print("LLM Knowledge Domains:\n")
    
    domains = {
        'General knowledge': {
            'coverage': 'Broad but uneven',
            'reliability': 'Medium (60-85%)',
            'example': 'Capital cities, historical facts, scientific concepts',
            'caveat': 'More reliable for popular topics'
        },
        'Language and literature': {
            'coverage': 'Excellent',
            'reliability': 'High (80-95%)',
            'example': 'Famous quotes, book plots, writing styles',
            'caveat': 'May confuse similar works'
        },
        'Programming knowledge': {
            'coverage': 'Very good for popular languages',
            'reliability': 'High (75-90%)',
            'example': 'Python, JavaScript, SQL, common libraries',
            'caveat': 'Weaker on niche languages/libraries'
        },
        'Scientific concepts': {
            'coverage': 'Good for established science',
            'reliability': 'Medium-High (70-85%)',
            'example': 'Basic physics, chemistry, biology principles',
            'caveat': 'May be outdated or oversimplified'
        },
        'Current events': {
            'coverage': 'Only up to training cutoff',
            'reliability': 'N/A after cutoff',
            'example': 'Events before training date (e.g., 2021, 2023)',
            'caveat': 'No knowledge of post-cutoff events'
        },
        'Cultural knowledge': {
            'coverage': 'Strong for popular culture',
            'reliability': 'Medium (60-80%)',
            'example': 'Movies, music, celebrities, memes',
            'caveat': 'Biased toward English/Western culture'
        }
    }
    
    for domain, info in domains.items():
        print(f"{domain.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

llm_knowledge_domains()
```

### Knowledge Limitations

```python
def knowledge_limitations():
    """Where LLM knowledge fails."""
    
    print("\n\nKnowledge Limitations:\n")
    
    limitations = {
        'Temporal knowledge': {
            'issue': 'No awareness of current time or events after training',
            'example': 'Asks about current date, recent news → outdated or wrong',
            'impact': 'Critical for time-sensitive applications'
        },
        'Factual accuracy': {
            'issue': 'Memorized patterns, not verified facts',
            'example': 'May confidently state wrong information',
            'impact': 'Cannot trust as authoritative source'
        },
        'Rare knowledge': {
            'issue': 'Weak on niche topics with little training data',
            'example': 'Obscure historical events, rare diseases',
            'impact': 'Unreliable for specialized domains'
        },
        'Numerical facts': {
            'issue': 'Poor at memorizing exact numbers',
            'example': 'Population of cities, dates, statistics',
            'impact': 'Numbers should be verified'
        },
        'Source attribution': {
            'issue': 'Cannot cite sources for information',
            'example': 'Knows fact but not where it learned it',
            'impact': 'No verification trail'
        },
        'Knowledge updating': {
            'issue': 'Knowledge frozen at training time',
            'example': 'New discoveries, changed facts unknown',
            'impact': 'Requires retraining or RAG for updates'
        }
    }
    
    for limitation, info in limitations.items():
        print(f"{limitation.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Key Takeaway:")
    print("  LLMs have broad but shallow knowledge")
    print("  Useful for general info, unreliable for precision")
    print("  ALWAYS verify critical facts")

knowledge_limitations()
```

## Reasoning Capabilities

### Where LLMs Can Reason

```python
def reasoning_capabilities():
    """Reasoning tasks LLMs can handle."""
    
    print("Reasoning Capabilities:\n")
    
    capabilities = {
        'Commonsense reasoning': {
            'description': 'Apply everyday knowledge to situations',
            'performance': 'Good (70-85%)',
            'example': 'If you drop a glass, it will likely break',
            'technique': 'Chain-of-thought helps significantly'
        },
        'Analogical reasoning': {
            'description': 'Recognize similar patterns/structures',
            'performance': 'Good (65-80%)',
            'example': '"Hand is to arm as foot is to..." → leg',
            'technique': 'Few-shot examples effective'
        },
        'Deductive reasoning': {
            'description': 'Simple logical inferences',
            'performance': 'Medium (60-75%)',
            'example': 'All A are B, X is A → X is B',
            'technique': 'CoT improves, but limited depth'
        },
        'Reading comprehension': {
            'description': 'Answer questions about text',
            'performance': 'Very good (80-90%)',
            'example': 'Given passage, answer who/what/when/where',
            'technique': 'Core strength, benefits from context'
        },
        'Multi-step reasoning': {
            'description': 'Chain multiple reasoning steps',
            'performance': 'Medium (50-70%)',
            'example': 'Math word problems (with CoT)',
            'technique': 'Requires chain-of-thought prompting'
        }
    }
    
    for capability, info in capabilities.items():
        print(f"{capability.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

reasoning_capabilities()
```

### Reasoning Failures

```python
def reasoning_failures():
    """Where LLM reasoning breaks down."""
    
    print("\n\nReasoning Failures:\n")
    
    failures = {
        'Logical consistency': {
            'problem': 'May make contradictory statements',
            'example': 'Says "All A are B" then "Some A are not B"',
            'why': 'No logical consistency checking',
            'frequency': 'Common in long conversations'
        },
        'Negation': {
            'problem': 'Struggles with negative statements',
            'example': '"Not all birds can fly" → confuses some/all',
            'why': 'Negation is complex in language',
            'frequency': 'Frequent error'
        },
        'Counterfactual reasoning': {
            'problem': 'Difficulty with "what if" scenarios',
            'example': '"If gravity were weaker, would..." → inconsistent',
            'why': 'Requires simulating alternative physics',
            'frequency': 'Moderate'
        },
        'Causal reasoning': {
            'problem': 'Confuses correlation and causation',
            'example': 'Assumes temporal order implies causation',
            'why': 'Causal relationships not explicitly learned',
            'frequency': 'Common'
        },
        'Complex logic': {
            'problem': 'Fails on multi-step logical proofs',
            'example': 'Nested quantifiers, long inference chains',
            'why': 'No formal reasoning system',
            'frequency': 'Very common'
        },
        'Probability': {
            'problem': 'Poor at probabilistic reasoning',
            'example': 'Misestimates likelihoods, ignores base rates',
            'why': 'Probabilistic reasoning requires math',
            'frequency': 'Common'
        }
    }
    
    for failure, info in failures.items():
        print(f"{failure.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Conclusion:")
    print("  LLMs have 'shallow' reasoning")
    print("  Good at pattern-based inference")
    print("  Poor at formal logic and consistency")

reasoning_failures()
```

## Language Understanding and Generation

### Strengths

```python
def language_strengths():
    """What LLMs excel at in language."""
    
    print("Language Understanding and Generation Strengths:\n")
    
    strengths = [
        ('Grammar and syntax', 'Excellent', 'Rarely makes grammatical errors', '95%+'),
        ('Fluency', 'Excellent', 'Generated text sounds natural', '95%+'),
        ('Coherence', 'Very good', 'Maintains topic and flow', '85-90%'),
        ('Style adaptation', 'Excellent', 'Matches requested tone/style', '90%+'),
        ('Multilingual', 'Good', 'Handles multiple languages', '70-90%'),
        ('Paraphrasing', 'Very good', 'Rephrase while preserving meaning', '85%+'),
        ('Summarization', 'Good', 'Extract key points from text', '75-85%'),
        ('Translation', 'Good', 'Convert between languages', '70-90%'),
    ]
    
    print(f"{'Capability':<25} {'Quality':<15} {'Description':<40} {'Reliability'}")
    print("="*105)
    
    for capability, quality, description, reliability in strengths:
        print(f"{capability:<25} {quality:<15} {description:<40} {reliability}")
    
    print("\n\nWhy so strong:")
    print("  • Core training objective is language modeling")
    print("  • Trained on trillions of tokens of text")
    print("  • Captures linguistic patterns at multiple levels")
    print("  • Syntax, semantics, pragmatics all learned")

language_strengths()
```

### Generation Quality Issues

```python
def generation_issues():
    """Quality issues in LLM generation."""
    
    print("\n\nGeneration Quality Issues:\n")
    
    issues = {
        'Verbosity': {
            'problem': 'Often generates overly long responses',
            'example': 'Simple question → 3 paragraph answer',
            'mitigation': 'Instruct for brevity, RLHF tuning'
        },
        'Repetition': {
            'problem': 'May repeat phrases or ideas',
            'example': 'Says same thing multiple times',
            'mitigation': 'Repetition penalties, better sampling'
        },
        'Generic responses': {
            'problem': 'Falls back to safe, generic answers',
            'example': '"It depends...", "There are pros and cons..."',
            'mitigation': 'Specific prompts, temperature tuning'
        },
        'Hedging': {
            'problem': 'Excessive uncertainty markers',
            'example': '"It might possibly perhaps could be..."',
            'mitigation': 'Instruct for confidence, RLHF'
        },
        'Format deviations': {
            'problem': 'Doesn\'t always follow format instructions',
            'example': 'Asked for JSON, returns prose',
            'mitigation': 'Clear formatting, few-shot examples'
        },
        'Off-topic drift': {
            'problem': 'Strays from original question',
            'example': 'Starts answering, ends discussing tangent',
            'mitigation': 'Shorter responses, focused prompts'
        }
    }
    
    for issue, info in issues.items():
        print(f"{issue.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

generation_issues()
```

## Common Failure Modes

### Hallucination

```python
def hallucination_patterns():
    """Understanding hallucination in LLMs."""
    
    print("Hallucination Patterns:\n")
    
    print("Definition: Generating plausible-sounding but incorrect information")
    print()
    
    types = {
        'Factual hallucination': {
            'description': 'Makes up false facts',
            'example': 'Claims Paris is the capital of Germany',
            'why': 'Pattern matching generates plausible falsehood',
            'severity': 'High - misinformation risk'
        },
        'Source hallucination': {
            'description': 'Invents citations or sources',
            'example': 'References non-existent papers or books',
            'why': 'Learned citation format, not actual sources',
            'severity': 'High - trust erosion'
        },
        'Detail hallucination': {
            'description': 'Adds fictional details to true information',
            'example': 'Correct event, wrong date or names',
            'why': 'Fills in gaps with plausible details',
            'severity': 'Medium - subtle errors'
        },
        'Reasoning hallucination': {
            'description': 'Invalid logical steps that sound reasonable',
            'example': 'Plausible but incorrect chain-of-thought',
            'why': 'Reasoning format learned, not logical validity',
            'severity': 'Medium - misleading reasoning'
        },
        'Code hallucination': {
            'description': 'Invents non-existent functions or APIs',
            'example': 'Uses made-up library methods',
            'why': 'Generalizes from similar real functions',
            'severity': 'Medium - code won\'t run'
        }
    }
    
    for htype, info in types.items():
        print(f"{htype.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Why hallucination happens:")
    print("  • Training objective: Generate plausible text")
    print("  • No explicit truthfulness constraint")
    print("  • No access to ground truth during generation")
    print("  • Pressure to respond even when uncertain")
    
    print("\nMitigation strategies:")
    print("  • Retrieval augmentation (RAG)")
    print("  • Confidence calibration")
    print("  • Fact-checking layers")
    print("  • Instruct to say 'I don't know'")
    print("  • RLHF for truthfulness")

hallucination_patterns()
```

## Arithmetic and Symbolic Reasoning

### Why LLMs Struggle with Math

```python
def math_limitations():
    """Why LLMs are poor at arithmetic."""
    
    print("\n\nArithmetic Limitations:\n")
    
    print("Simple arithmetic: What is 127 + 358?")
    print("  LLM (without tools): '475' (WRONG - should be 485)")
    print("  Why: Token-by-token generation, no calculator")
    print()
    
    limitations = {
        'Multi-digit arithmetic': {
            'success_rate': '30-60%',
            'example': '1247 × 389 → often wrong',
            'why': 'Cannot track carries, requires precise steps'
        },
        'Floating point': {
            'success_rate': '20-40%',
            'example': '3.14159 × 2.71828 → unreliable',
            'why': 'Decimal precision hard in text'
        },
        'Large numbers': {
            'success_rate': '10-30%',
            'example': '999999 + 888888 → likely wrong',
            'why': 'Tokenization breaks up numbers'
        },
        'Multi-step calculations': {
            'success_rate': '40-70% (with CoT)',
            'example': 'Word problems with several operations',
            'why': 'Each step can introduce errors'
        },
        'Symbolic math': {
            'success_rate': '20-50%',
            'example': 'Solve: 2x + 5 = 13 for x',
            'why': 'Requires systematic algebraic manipulation'
        }
    }
    
    print("Performance by math type:\n")
    for mathtype, info in limitations.items():
        print(f"{mathtype.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Solutions:")
    print("  • Use external calculators/tools (PAL, Code Interpreter)")
    print("  • Chain-of-thought prompting helps but doesn't solve")
    print("  • Specialized models (e.g., Minerva) better but still limited")
    print("  • For reliable math: Use symbolic computation libraries")

math_limitations()
```

### When to Use Tools

```python
def when_use_tools():
    """When LLMs need external tools."""
    
    print("\n\nWhen to Use External Tools:\n")
    
    tool_needs = {
        'Calculator/Code execution': {
            'for': 'Any arithmetic beyond simple addition/subtraction',
            'example': 'Multiply large numbers, percentages, statistics',
            'tool': 'Python interpreter, calculator API'
        },
        'Search/Retrieval': {
            'for': 'Current information, specific facts, citations',
            'example': 'Recent news, specific paper details, stats',
            'tool': 'Web search, vector database (RAG)'
        },
        'Database queries': {
            'for': 'Structured data access',
            'example': 'Query user database, product inventory',
            'tool': 'SQL database, APIs'
        },
        'Specialized APIs': {
            'for': 'Domain-specific operations',
            'example': 'Weather data, stock prices, mapping',
            'tool': 'Weather API, financial APIs, Google Maps'
        },
        'Verification': {
            'for': 'Fact-checking, logical verification',
            'example': 'Verify calculation, check claim',
            'tool': 'Symbolic solvers, fact-checking APIs'
        }
    }
    
    for need, info in tool_needs.items():
        print(f"{need.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Tool use pattern:")
    print("  1. LLM identifies need for tool")
    print("  2. LLM calls tool with parameters")
    print("  3. Tool returns result")
    print("  4. LLM incorporates result in response")
    print()
    print("Benefits:")
    print("  • Compensates for LLM weaknesses")
    print("  • Provides verifiable, accurate results")
    print("  • Extends LLM capabilities")

when_use_tools()
```

## Context and Memory Limitations

### Context Window Constraints

```python
def context_limitations():
    """Understanding context window limitations."""
    
    print("Context Window Constraints:\n")
    
    models = [
        ('GPT-3', '4k tokens', '~3k words', 'Short conversations'),
        ('GPT-3.5', '4k-16k tokens', '~3k-12k words', 'Moderate docs'),
        ('GPT-4', '8k-32k tokens', '~6k-24k words', 'Long documents'),
        ('GPT-4-turbo', '128k tokens', '~96k words', 'Books, codebases'),
        ('Claude 2', '100k tokens', '~75k words', 'Very long context'),
        ('Claude 3', '200k tokens', '~150k words', 'Massive context'),
    ]
    
    print(f"{'Model':<15} {'Context':<15} {'~Words':<15} {'Use Case'}")
    print("="*70)
    
    for model, context, words, use_case in models:
        print(f"{model:<15} {context:<15} {words:<15} {use_case}")
    
    print("\nImplications:")
    print("  • Everything must fit in context window")
    print("  • Older conversations get truncated")
    print("  • Long documents need chunking")
    print("  • RAG helps manage large knowledge bases")
    
    print("\nWhat happens when context is exceeded:")
    print("  • Oldest messages dropped (sliding window)")
    print("  • API returns error if input too long")
    print("  • Information loss from dropped context")
    print("  • May forget earlier conversation parts")

context_limitations()
```

### Memory Characteristics

```python
def memory_characteristics():
    """How LLM 'memory' works."""
    
    print("\n\nLLM Memory Characteristics:\n")
    
    characteristics = {
        'No persistent memory': {
            'description': 'Each request is independent',
            'implication': 'Doesn\'t remember previous sessions',
            'workaround': 'Store conversation history, pass in context'
        },
        'Context-based only': {
            'description': 'Only "remembers" what\'s in current prompt',
            'implication': 'Memory is limited by context window',
            'workaround': 'Summarize old context, use vector stores'
        },
        'Recency bias': {
            'description': 'More influenced by recent text',
            'implication': 'May forget earlier instructions',
            'workaround': 'Repeat important info, structured prompts'
        },
        'No learning from interaction': {
            'description': 'Doesn\'t update weights during use',
            'implication': 'Can\'t improve from feedback in session',
            'workaround': 'Fine-tuning, RLHF for new model versions'
        },
        'Lossy recall': {
            'description': 'May not recall exact earlier statements',
            'implication': 'Paraphrases or forgets details',
            'workaround': 'Explicit memory systems, retrieval'
        }
    }
    
    for characteristic, info in characteristics.items():
        print(f"{characteristic.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Simulating Long-Term Memory:")
    print("  • Vector databases for semantic search")
    print("  • Conversation summarization")
    print("  • External memory stores (database)")
    print("  • Retrieval-augmented generation (RAG)")

memory_characteristics()
```

## Robustness and Consistency

### Sensitivity to Prompts

```python
def prompt_sensitivity():
    """How sensitive LLMs are to prompt variations."""
    
    print("Prompt Sensitivity:\n")
    
    print("Same question, different phrasings can yield different answers:")
    print()
    
    examples = [
        {
            'prompt1': 'Is coffee healthy?',
            'response1': 'Coffee has both benefits and risks...',
            'prompt2': 'What are the health benefits of coffee?',
            'response2': 'Coffee provides antioxidants, improves focus...',
            'prompt3': 'What are the health risks of coffee?',
            'response3': 'Coffee can cause anxiety, insomnia, addiction...'
        }
    ]
    
    for ex in examples:
        print(f"Prompt 1: '{ex['prompt1']}'")
        print(f"  Response: {ex['response1']}")
        print()
        print(f"Prompt 2: '{ex['prompt2']}'")
        print(f"  Response: {ex['response2']}")
        print()
        print(f"Prompt 3: '{ex['prompt3']}'")
        print(f"  Response: {ex['response3']}")
    
    print("\nNote: Same topic, but framing influences response tone/content")
    print()
    
    sensitivity_factors = {
        'Wording changes': 'Synonyms, paraphrases → different emphasis',
        'Instruction format': 'Imperative vs question form → affects style',
        'Example inclusion': 'With/without examples → major impact',
        'Order of information': 'Different orders → recency bias',
        'Formatting': 'Bullets, numbering, plain text → affects structure',
        'Negative phrasing': 'Do/don\'t formulations → higher error rates'
    }
    
    print("Sensitivity Factors:\n")
    for factor, impact in sensitivity_factors.items():
        print(f"  {factor}: {impact}")
    
    print("\nImplication: Prompt engineering is critical")
    print("Small changes can significantly affect output quality")

prompt_sensitivity()
```

### Inconsistency Issues

```python
def inconsistency_patterns():
    """Where LLMs are inconsistent."""
    
    print("\n\nInconsistency Patterns:\n")
    
    patterns = {
        'Self-contradiction': {
            'description': 'Makes conflicting statements',
            'example': 'Says X is true, later says X is false',
            'cause': 'No consistency checking mechanism',
            'frequency': 'Common in long texts'
        },
        'Randomness in responses': {
            'description': 'Same prompt → different outputs',
            'example': 'Run query twice, get different answers',
            'cause': 'Sampling introduces randomness (temp > 0)',
            'frequency': 'Always present (unless temp=0)'
        },
        'Context-dependent views': {
            'description': 'Opinion shifts based on framing',
            'example': 'Pro-X when asked benefits, anti-X when asked risks',
            'cause': 'Responds to immediate context, not global stance',
            'frequency': 'Very common'
        },
        'Fact variability': {
            'description': 'States different facts on same topic',
            'example': 'Population of city varies between answers',
            'cause': 'Probabilistic generation, weak factual grounding',
            'frequency': 'Moderate'
        },
        'Style drift': {
            'description': 'Writing style changes during generation',
            'example': 'Starts formal, becomes casual',
            'cause': 'Local context influences generation',
            'frequency': 'Occasional'
        }
    }
    
    for pattern, info in patterns.items():
        print(f"{pattern.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Mitigation:")
    print("  • Lower temperature for consistency")
    print("  • Self-consistency (sample multiple times)")
    print("  • Explicit consistency instructions")
    print("  • Structured outputs (JSON, forms)")
    print("  • Post-processing to check for contradictions")

inconsistency_patterns()
```

## Safety and Alignment Challenges

### Safety Concerns

```python
def safety_concerns():
    """Safety and alignment challenges with LLMs."""
    
    print("Safety and Alignment Challenges:\n")
    
    concerns = {
        'Harmful content generation': {
            'risk': 'May generate violent, explicit, or offensive content',
            'example': 'Detailed violence, hate speech, explicit material',
            'mitigation': 'Content filtering, RLHF, refusal training'
        },
        'Bias amplification': {
            'risk': 'Reflects and amplifies training data biases',
            'example': 'Gender, racial, cultural stereotypes',
            'mitigation': 'Diverse training data, bias detection, RLHF'
        },
        'Malicious use': {
            'risk': 'Can be used for phishing, misinformation, spam',
            'example': 'Generate convincing phishing emails, fake news',
            'mitigation': 'Usage policies, detection systems, watermarking'
        },
        'Privacy leakage': {
            'risk': 'May memorize and reveal training data',
            'example': 'Reproduce personal info, copyrighted text',
            'mitigation': 'Data filtering, privacy audits, differential privacy'
        },
        'Manipulation': {
            'risk': 'Can be used for social engineering',
            'example': 'Persuasive fake reviews, impersonation',
            'mitigation': 'Disclosure requirements, authentication'
        },
        'Jailbreaking': {
            'risk': 'Users bypass safety measures',
            'example': 'Roleplay, encoded instructions to get harmful output',
            'mitigation': 'Adversarial training, monitoring, updates'
        }
    }
    
    for concern, info in concerns.items():
        print(f"{concern.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Alignment Goals:")
    print("  • Helpful: Assist users effectively")
    print("  • Harmless: Avoid harmful outputs")
    print("  • Honest: Truthful and accurate")
    print()
    print("These goals can conflict (e.g., helpfulness vs safety)")

safety_concerns()
```

### Alignment Limitations

```python
def alignment_limitations():
    """Current limitations in LLM alignment."""
    
    print("\n\nAlignment Limitations:\n")
    
    limitations = {
        'Incomplete specification': {
            'issue': 'Hard to specify all desired behaviors',
            'example': 'RLHF captures average human preference, not universal',
            'consequence': 'Edge cases mishandled'
        },
        'Reward hacking': {
            'issue': 'Model optimizes specified reward, not intent',
            'example': 'Generates verbose text if rewarded for length',
            'consequence': 'Gaming the objective'
        },
        'Distributional shift': {
            'issue': 'Deployed usage differs from training',
            'example': 'Trained on helpfulness, used for academic dishonesty',
            'consequence': 'Misuse in new contexts'
        },
        'Adversarial robustness': {
            'issue': 'Intentional attempts to break alignment',
            'example': 'Jailbreaks, prompt injection attacks',
            'consequence': 'Safety measures bypassed'
        },
        'Value alignment': {
            'issue': 'Whose values should model align with?',
            'example': 'Cultural differences, political views',
            'consequence': 'No universal "correct" alignment'
        },
        'Capability without alignment': {
            'issue': 'Models get smarter faster than alignment improves',
            'example': 'GPT-4 more capable but also new risks',
            'consequence': 'Safety lags capabilities'
        }
    }
    
    for limitation, info in limitations.items():
        print(f"{limitation.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Open Problems:")
    print("  • Scalable oversight: Supervise superhuman systems")
    print("  • Value specification: Define 'good' behavior precisely")
    print("  • Robustness: Maintain alignment under adversaries")
    print("  • Interpretability: Understand why models do what they do")

alignment_limitations()
```

## Summary

**Key Capabilities**:

1. **Text generation**: Excellent at creative writing, code, completion, style mimicry
2. **Pattern recognition**: Strong at classification, sentiment, entity extraction (80-95%)
3. **Language understanding**: Very good at syntax, semantics, paraphrasing, translation
4. **Commonsense reasoning**: Good at everyday reasoning, analogy (70-85%)
5. **Knowledge recall**: Broad but shallow, reliable for popular topics

**Key Limitations**:

1. **Arithmetic**: Poor at multi-digit math, symbolic manipulation (30-60%)
2. **Hallucination**: Generates plausible but false information frequently
3. **Logical consistency**: Makes contradictory statements, poor formal logic
4. **Factual accuracy**: Not a reliable source of truth, requires verification
5. **Context limits**: Finite memory, recency bias, no persistent memory
6. **Robustness**: Sensitive to prompt phrasing, inconsistent outputs
7. **Temporal knowledge**: Frozen at training time, no current events

**Capability Matrix**:

| Task Type | Performance | Reliability | Use Tools? |
|-----------|-------------|-------------|------------|
| Text generation | Excellent (95%) | High | No |
| Classification | Very Good (85-95%) | High | No |
| Summarization | Good (75-85%) | Medium-High | No |
| Translation | Good (70-90%) | Medium-High | No |
| Math/Arithmetic | Poor (30-60%) | Low | **Yes** |
| Factual QA | Medium (60-80%) | Medium | **Yes (RAG)** |
| Reasoning | Medium (60-75%) | Medium | Sometimes |
| Current events | N/A | N/A | **Yes (Search)** |

**Common Failure Modes**:

```
Hallucination → Use RAG, fact-checking
Arithmetic errors → Use calculator tools
Logical inconsistency → Self-consistency, verification
Outdated knowledge → Web search, RAG
Prompt sensitivity → Careful prompt engineering
Memory limits → Summarization, vector stores
```

**Best Practices**:

1. **Verify critical facts** - Don't trust LLM as authoritative source
2. **Use tools for precision** - Calculator for math, search for facts
3. **Expect hallucination** - Build verification into workflows
4. **Test prompt robustness** - Try variations, use self-consistency
5. **Manage context** - Summarize, use RAG for long-term memory
6. **Set appropriate expectations** - LLMs assist, don't replace experts
7. **Safety measures** - Content filtering, monitoring, human oversight

**When to Use LLMs**:

- ✅ Drafting, brainstorming, creative tasks
- ✅ Code generation (with testing)
- ✅ Classification, extraction, summarization
- ✅ Conversational interfaces
- ✅ Translation, paraphrasing
- ✅ Question answering (with RAG)

**When NOT to Use LLMs**:

- ❌ Mission-critical factual accuracy
- ❌ Precise arithmetic without tools
- ❌ Legal/medical decisions alone
- ❌ Formal logical proofs
- ❌ Real-time information without search
- ❌ Security-critical operations

**Mental Model**:

```
LLM = Sophisticated Pattern Matcher + Fluent Text Generator

Strengths: Language, patterns, creativity, broad knowledge
Weaknesses: Precision, logic, arithmetic, consistency, facts

Best deployed: As assistant with verification, not autonomous authority
```

## Next Steps

- Study [Prompt Engineering](../prompt_engineering/basic-prompting.md) to maximize capabilities
- Learn [Retrieval Augmented Generation](../retrieval_augmented_generation/rag-fundamentals.md) for factual grounding
- Explore [Tool Use](../agentic-ai-lab/tool-use/) to compensate for limitations
- Understand [Evaluation](../evaluation/benchmark-evaluation.md) for measuring performance
- Study [RLHF and Alignment](../rlhf_and_alignment/rlhf-fundamentals.md) for safety
- Learn [Agent Architectures](../agentic-ai-lab/agent-architectures/) for complex systems
- Explore [Application Patterns](../application_patterns/best-practices.md) for reliable systems

