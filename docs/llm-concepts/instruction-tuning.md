# Instruction Tuning

## Table of Contents

- [Introduction](#introduction)
- [What is Instruction Tuning?](#what-is-instruction-tuning)
- [Instruction Datasets](#instruction-datasets)
- [Instruction Tuning Methods](#instruction-tuning-methods)
- [Zero-Shot Task Generalization](#zero-shot-task-generalization)
- [Instruction Format Design](#instruction-format-design)
- [Multi-Task Learning](#multi-task-learning)
- [Instruction Quality and Diversity](#instruction-quality-and-diversity)
- [Scaling Instruction Tuning](#scaling-instruction-tuning)
- [Evaluation of Instruction-Tuned Models](#evaluation-of-instruction-tuned-models)
- [From Instruction Tuning to RLHF](#from-instruction-tuning-to-rlhf)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Instruction tuning** is the process of fine-tuning pre-trained language models on datasets of instruction-response pairs to improve their ability to follow natural language instructions. This technique transforms base models into helpful assistants.

```
Base Model (Pre-trained):
User: "Translate to French: Hello"
Model: "Hello bonjour good morning salut..." ← Continues text, doesn't follow instruction

Instruction-Tuned Model:
User: "Translate to French: Hello"
Model: "Bonjour" ← Follows instruction correctly
```

**Key insight**: Pre-trained models have knowledge but don't naturally follow instructions. Instruction tuning teaches them to interpret and execute user intent.

```
Evolution of LLM Capabilities:

Pre-training → Knowledge and Language Understanding
     ↓
Instruction Tuning → Following Instructions
     ↓
RLHF → Human Preference Alignment
     ↓
Helpful, Harmless, Honest Assistant
```

This guide covers how instruction tuning works, datasets used, methods, and its role in creating practical LLM applications.

## What is Instruction Tuning?

### The Core Concept

```python
def instruction_tuning_concept():
    """Understanding instruction tuning fundamentals."""
    
    print("Instruction Tuning Components:\n")
    
    # Standard pre-training example
    print("Pre-Training (Next Token Prediction):")
    print("  Input:  'The capital of France is'")
    print("  Output: 'Paris' (continues the text)")
    print("  Goal:   Learn language patterns and knowledge")
    print()
    
    # Instruction tuning example
    print("Instruction Tuning:")
    print("  Input:  'Q: What is the capital of France? A:'")
    print("  Output: 'Paris'")
    print("  Goal:   Learn to respond to questions")
    print()
    
    print("="*60)
    
    # Key difference
    print("\nKey Difference:")
    print("  Pre-training: 'What comes next in this text?'")
    print("  Instruction:  'How do I respond to this request?'")
    
    print("\nFormat:")
    examples = [
        {
            'instruction': 'Translate to Spanish:',
            'input': 'Hello, how are you?',
            'output': 'Hola, ¿cómo estás?'
        },
        {
            'instruction': 'Summarize this text:',
            'input': 'Long article about climate change...',
            'output': 'The article discusses climate change impacts...'
        },
        {
            'instruction': 'Answer the question:',
            'input': 'What is photosynthesis?',
            'output': 'Photosynthesis is the process by which plants...'
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n  Example {i}:")
        print(f"    Instruction: {ex['instruction']}")
        print(f"    Input: {ex['input'][:50]}...")
        print(f"    Output: {ex['output'][:50]}...")

instruction_tuning_concept()
```

### Historical Development

```python
def instruction_tuning_history():
    """Timeline of instruction tuning development."""
    
    milestones = [
        ('2021', 'FLAN', 'Fine-tuned LaMDA on 62 NLP tasks'),
        ('2022', 'T0', 'Instruction tuning on diverse prompts'),
        ('2022', 'InstructGPT', 'Instruction tuning + RLHF'),
        ('2022', 'FLAN-T5', 'T5 with instruction tuning'),
        ('2023', 'Alpaca', 'LLaMA + 52k instruction dataset'),
        ('2023', 'Vicuna', 'LLaMA + ShareGPT conversations'),
        ('2023', 'WizardLM', 'Evolved instructions for complexity'),
        ('2023', 'Orca', 'Learning from GPT-4 explanations')
    ]
    
    print("Instruction Tuning Evolution:\n")
    
    for year, model, description in milestones:
        print(f"{year}: {model}")
        print(f"       {description}")
        print()
    
    print("Impact:")
    print("  • Transformed LLMs from predictors to assistants")
    print("  • Enabled zero-shot task performance")
    print("  • Made LLMs practical for end users")

instruction_tuning_history()
```

### Before and After

```python
def before_after_comparison():
    """Compare model behavior before and after instruction tuning."""
    
    print("\n\nBehavior Comparison:\n")
    
    test_cases = [
        {
            'prompt': 'Translate to German: I love programming',
            'base_model': 'Translate to German: I love programming. Translation: Ich liebe Programmierung. Translate to Spanish: I love programming...',
            'instruction_tuned': 'Ich liebe Programmierung.'
        },
        {
            'prompt': 'Is this review positive or negative? "The movie was terrible"',
            'base_model': 'Is this review positive or negative? "The movie was terrible". Review 2: "Best film ever" is positive...',
            'instruction_tuned': 'Negative'
        },
        {
            'prompt': 'Write a haiku about winter',
            'base_model': 'Write a haiku about winter. A haiku is a form of poetry. It has 17 syllables...',
            'instruction_tuned': 'Snowflakes gently fall\nSilent white blankets the earth\nWinter\'s peaceful sleep'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"  Prompt: {test['prompt']}")
        print(f"\n  Base Model (Pre-trained):")
        print(f"    {test['base_model']}")
        print(f"\n  Instruction-Tuned Model:")
        print(f"    {test['instruction_tuned']}")
        print()
        print("="*60)
        print()

before_after_comparison()
```

## Instruction Datasets

### Types of Instruction Data

```python
def instruction_dataset_types():
    """Different types of instruction datasets."""
    
    dataset_types = {
        'Task-specific datasets': {
            'description': 'Collections of NLP tasks with instructions',
            'examples': ['SuperGLUE, SQuAD, MNLI with instructions'],
            'size': '~50-200 tasks',
            'strength': 'High quality, well-defined',
            'weakness': 'Limited diversity'
        },
        'Multi-task mixtures': {
            'description': 'Mixed task types with instruction templates',
            'examples': ['FLAN (62 tasks), T0 (47 datasets)'],
            'size': '~10M examples',
            'strength': 'Broad coverage, diverse',
            'weakness': 'Template-based, may be rigid'
        },
        'Human-written instructions': {
            'description': 'Crowdsourced instruction-response pairs',
            'examples': ['Super-NaturalInstructions, xP3'],
            'size': '~1M+ examples',
            'strength': 'Natural diversity',
            'weakness': 'Expensive to create'
        },
        'Model-generated instructions': {
            'description': 'LLM-generated instruction datasets',
            'examples': ['Self-Instruct, Alpaca (52k)'],
            'size': '~50k-1M examples',
            'strength': 'Cheap, scalable',
            'weakness': 'May inherit model biases'
        },
        'Conversational data': {
            'description': 'Multi-turn conversations',
            'examples': ['ShareGPT, OASST1'],
            'size': '~10k-100k conversations',
            'strength': 'Natural interaction',
            'weakness': 'Quality varies'
        }
    }
    
    print("Instruction Dataset Types:\n")
    for dtype, info in dataset_types.items():
        print(f"{dtype.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

instruction_dataset_types()
```

### FLAN Dataset

```python
def flan_dataset():
    """FLAN: Finetuned Language Models Are Zero-Shot Learners."""
    
    print("FLAN Dataset Structure:\n")
    
    print("62 NLP tasks across categories:")
    print()
    
    categories = {
        'Reading Comprehension': {
            'tasks': ['SQuAD, BoolQ, MultiRC'],
            'examples': '~10 instruction templates per task'
        },
        'Closed-Book QA': {
            'tasks': ['Natural Questions, Web Questions'],
            'examples': '~10 templates'
        },
        'Translation': {
            'tasks': ['WMT (multiple language pairs)'],
            'examples': '~5 templates'
        },
        'Sentiment Analysis': {
            'tasks': ['SST-2, IMDB'],
            'examples': '~10 templates'
        },
        'Summarization': {
            'tasks': ['CNN/DailyMail, XSum'],
            'examples': '~5 templates'
        },
        'Commonsense Reasoning': {
            'tasks': ['Winogrande, HellaSwag, PIQA'],
            'examples': '~10 templates'
        },
        'Coreference Resolution': {
            'tasks': ['WSC, DPR'],
            'examples': '~8 templates'
        },
        'Structure-to-Text': {
            'tasks': ['CommonGen, DART'],
            'examples': '~5 templates'
        }
    }
    
    for category, info in categories.items():
        print(f"{category}:")
        print(f"  Tasks: {info['tasks']}")
        print(f"  Templates: {info['examples']}")
        print()
    
    print("Template Example (Reading Comprehension):")
    templates = [
        "Read this and answer the question: {context} {question}",
        "Answer based on context: {context} Q: {question} A:",
        "{context} Question: {question} Answer:",
        "Given: {context} What is the answer to: {question}?"
    ]
    for template in templates:
        print(f"  • {template}")
    
    print("\n" + "="*60)
    print("\nKey Innovation:")
    print("  • Multiple instruction phrasings per task")
    print("  • ~10 templates per task = diverse instruction formats")
    print("  • Teaches model: 'instruction intent' not 'exact wording'")

flan_dataset()
```

### Self-Instruct

```python
def self_instruct_method():
    """Self-Instruct: Generate instruction data with LLMs."""
    
    print("\n\nSelf-Instruct Method:\n")
    
    print("Algorithm:")
    print("  1. Start with seed instruction set (~175 examples)")
    print("  2. Sample seed instructions as examples")
    print("  3. Prompt LLM to generate new instructions")
    print("  4. Prompt LLM to generate input-output pairs")
    print("  5. Filter low-quality/duplicate instructions")
    print("  6. Add to instruction pool")
    print("  7. Repeat until target dataset size")
    
    print("\nExample Generation:\n")
    
    prompt = """
Task: Generate a new instruction that is different from these examples:

1. Translate English to French: {input}
2. Summarize the following text: {input}
3. Answer the question: {input}

New instruction:
"""
    print(prompt)
    
    print("Generated: 'Write a product review for: {input}'\n")
    
    print("Then generate example:")
    print("  Input: 'smartphone with great camera'")
    print("  Output: 'I recently purchased this smartphone and...'")
    
    print("\n" + "="*60)
    print("\nResults (Alpaca 52k):")
    print("  • Started with 175 seed tasks")
    print("  • Generated 52k instruction-following examples")
    print("  • Cost: ~$500 (using text-davinci-003)")
    print("  • Quality: Good enough to produce capable models")
    
    print("\nBenefits:")
    print("  • Low cost compared to human annotation")
    print("  • Scalable to large datasets")
    print("  • Diverse instruction formats")
    
    print("\nLimitations:")
    print("  • Inherits biases from generator model")
    print("  • May have factual errors")
    print("  • Less natural than human-written")

self_instruct_method()
```

## Instruction Tuning Methods

### Supervised Fine-Tuning (SFT)

```python
def supervised_fine_tuning():
    """Standard supervised instruction tuning."""
    
    print("Supervised Fine-Tuning for Instructions:\n")
    
    print("Process:")
    print("  1. Take pre-trained base model")
    print("  2. Format instruction data as input-output pairs")
    print("  3. Fine-tune model to predict outputs given inputs")
    print("  4. Optimize cross-entropy loss")
    
    print("\nData Format:")
    example = {
        'instruction': 'Classify the sentiment of the text.',
        'input': 'I absolutely loved this movie!',
        'output': 'Positive'
    }
    
    print(f"\n  Instruction: {example['instruction']}")
    print(f"  Input: {example['input']}")
    print(f"  Output: {example['output']}")
    
    print("\nModel Input (concatenated):")
    model_input = f"{example['instruction']} {example['input']}"
    print(f"  '{model_input}'")
    
    print("\nModel Target:")
    print(f"  '{example['output']}'")
    
    print("\nLoss Computation:")
    print("  L = -log P(output | instruction, input)")
    print("  Minimize negative log-likelihood")
    
    print("\n" + "="*60)
    print("\nTraining Details:")
    
    details = {
        'Learning rate': '1e-5 to 5e-5 (lower than pre-training)',
        'Epochs': '1-3 (few epochs to avoid overfitting)',
        'Batch size': '32-128 (depends on GPU memory)',
        'Sequence length': '512-2048 tokens',
        'Optimizer': 'AdamW with weight decay',
        'Warmup': '100-1000 steps',
        'Training time': 'Hours to days (vs months for pre-training)'
    }
    
    for key, value in details.items():
        print(f"  {key}: {value}")

supervised_fine_tuning()
```

### Multi-Task Fine-Tuning

```python
def multi_task_fine_tuning():
    """Fine-tuning on multiple tasks simultaneously."""
    
    print("\n\nMulti-Task Instruction Fine-Tuning:\n")
    
    print("Approach: Mix data from many tasks in training")
    print()
    
    tasks = {
        'Translation': 0.15,
        'Summarization': 0.20,
        'QA': 0.25,
        'Classification': 0.20,
        'Generation': 0.20
    }
    
    print("Task Mix (example proportions):")
    for task, proportion in tasks.items():
        bar = '█' * int(proportion * 50)
        print(f"  {task:<20} {bar} {proportion:.0%}")
    
    print("\nSampling Strategy:")
    print("  • Temperature sampling: Adjust task proportions")
    print("  • Uniform: Sample all tasks equally")
    print("  • Proportional: Sample based on dataset size")
    print("  • Curriculum: Start simple, increase difficulty")
    
    print("\nBenefits:")
    print("  • Better generalization across tasks")
    print("  • Reduces catastrophic forgetting")
    print("  • Single model handles multiple capabilities")
    print("  • Cross-task knowledge transfer")
    
    print("\nChallenges:")
    print("  • Balancing task proportions")
    print("  • High-resource tasks may dominate")
    print("  • Need large, diverse instruction datasets")
    
    print("\nExample Models:")
    print("  • FLAN: 62 tasks mixed")
    print("  • T0: 47 datasets with templates")
    print("  • FLAN-T5: Multi-task mixture for T5")

multi_task_fine_tuning()
```

### Instruction Tuning vs Traditional Fine-Tuning

```python
def instruction_vs_traditional():
    """Compare instruction tuning with traditional fine-tuning."""
    
    print("\n\nInstruction Tuning vs Traditional Fine-Tuning:\n")
    
    comparison = {
        'Data format': {
            'Traditional': 'Task-specific (labels, spans, etc.)',
            'Instruction': 'Natural language instructions + examples'
        },
        'Task description': {
            'Traditional': 'Implicit in training data',
            'Instruction': 'Explicit in instruction text'
        },
        'Generalization': {
            'Traditional': 'Good on trained task, poor on new tasks',
            'Instruction': 'Zero-shot on new tasks with instructions'
        },
        'Multi-task': {
            'Traditional': 'Requires multi-task architecture',
            'Instruction': 'Natural via instruction diversity'
        },
        'Human interface': {
            'Traditional': 'Needs task-specific formatting',
            'Instruction': 'Natural language instructions'
        },
        'Dataset size': {
            'Traditional': 'Often smaller, task-specific',
            'Instruction': 'Large, diverse instruction pools'
        },
        'Transfer': {
            'Traditional': 'Limited to similar tasks',
            'Instruction': 'Broad transfer to instruction-following'
        }
    }
    
    print(f"{'Aspect':<20} {'Traditional FT':<35} {'Instruction Tuning'}")
    print("="*90)
    
    for aspect, details in comparison.items():
        print(f"{aspect:<20} {details['Traditional']:<35} {details['Instruction']}")
    
    print("\n\nWhen to use each:")
    print("\nTraditional Fine-Tuning:")
    print("  • Single, well-defined task")
    print("  • Maximum performance on that task")
    print("  • Large task-specific dataset available")
    
    print("\nInstruction Tuning:")
    print("  • General-purpose assistant")
    print("  • Multiple diverse tasks")
    print("  • Zero-shot generalization needed")
    print("  • User-facing natural language interface")

instruction_vs_traditional()
```

## Zero-Shot Task Generalization

### The Power of Instruction Tuning

```python
def zero_shot_generalization():
    """How instruction tuning enables zero-shot performance."""
    
    print("Zero-Shot Task Generalization:\n")
    
    print("Key Idea: Learn to follow instructions, not memorize tasks")
    print()
    
    print("Training Tasks (seen):")
    seen_tasks = [
        "Translate English to French",
        "Summarize news articles",
        "Answer science questions",
        "Classify sentiment (positive/negative)"
    ]
    for task in seen_tasks:
        print(f"  • {task}")
    
    print("\nTest Tasks (unseen - zero-shot):")
    unseen_tasks = [
        "Translate English to German (new language)",
        "Summarize movie plots (new domain)",
        "Answer history questions (new domain)",
        "Classify sentiment (3-way: pos/neg/neutral)"
    ]
    for task in unseen_tasks:
        print(f"  • {task}")
    
    print("\nWhy it works:")
    print("  1. Model learns 'what is an instruction'")
    print("  2. Model learns 'how to interpret user intent'")
    print("  3. Model leverages pre-trained knowledge")
    print("  4. Instruction diversity → generalization")
    
    print("\nExample:")
    print("  Trained: 'Translate to French: Hello' → 'Bonjour'")
    print("  Zero-shot: 'Translate to Spanish: Hello' → 'Hola'")
    print("              (Spanish not in training, but works!)")

zero_shot_generalization()
```

### Performance Comparison

```python
def instruction_tuning_performance():
    """Performance gains from instruction tuning."""
    
    print("\n\nInstruction Tuning Performance:\n")
    
    # Results on unseen tasks
    results = {
        'FLAN (137B)': {
            'Base model (zero-shot)': 42.3,
            'After instruction tuning': 57.4,
            'Improvement': '+36%'
        },
        'T0 (11B)': {
            'Base model (zero-shot)': 34.2,
            'After instruction tuning': 52.1,
            'Improvement': '+52%'
        },
        'InstructGPT (1.3B)': {
            'Base model': 'Often fails to follow',
            'After instruction tuning': '85% preferred by humans',
            'Improvement': 'Qualitative leap'
        }
    }
    
    print("Zero-Shot Performance on Held-Out Tasks:\n")
    
    for model, scores in results.items():
        print(f"{model}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value}")
        print()
    
    print("Key Findings:")
    print("  • Instruction tuning dramatically improves zero-shot")
    print("  • Benefits scale with model size")
    print("  • More tasks in training → better generalization")
    print("  • Even small instruction-tuned models useful")

instruction_tuning_performance()
```

## Instruction Format Design

### Instruction Components

```python
def instruction_format_components():
    """Components of effective instruction formats."""
    
    print("Instruction Format Design:\n")
    
    components = {
        'Task description': {
            'purpose': 'What to do',
            'example': '"Translate the following text to Spanish"',
            'best_practice': 'Clear, concise, unambiguous'
        },
        'Input specification': {
            'purpose': 'What to process',
            'example': '"Text: {input_text}"',
            'best_practice': 'Clearly marked, separated'
        },
        'Output specification': {
            'purpose': 'What format to produce',
            'example': '"Translation:"',
            'best_practice': 'Explicit format, constraints'
        },
        'Examples (optional)': {
            'purpose': 'Demonstrate task',
            'example': '"Example: Hello → Hola"',
            'best_practice': '1-3 examples if task complex'
        },
        'Constraints (optional)': {
            'purpose': 'Additional requirements',
            'example': '"Keep the translation concise"',
            'best_practice': 'Only when needed'
        }
    }
    
    for component, info in components.items():
        print(f"{component.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Complete Example:")
    instruction_example = """
Instruction: Translate the following text to Spanish.

Text: "Good morning, how are you today?"

Translation:
"""
    print(instruction_example)
    print("Expected Output: 'Buenos días, ¿cómo estás hoy?'")

instruction_format_components()
```

### Natural vs Templated Instructions

```python
def natural_vs_templated():
    """Compare natural and templated instruction formats."""
    
    print("\n\nNatural vs Templated Instructions:\n")
    
    task = "Sentiment classification"
    
    print("Templated Instructions (structured):\n")
    templated = [
        "Classify the sentiment: {text}",
        "Sentiment of '{text}': [positive/negative]",
        "Is this positive or negative? {text}",
        "Determine sentiment: {text} →"
    ]
    for template in templated:
        print(f"  • {template}")
    
    print("\nNatural Instructions (varied):\n")
    natural = [
        "What's the sentiment of this review?",
        "Tell me if this text is positive or negative.",
        "How would you describe the tone here?",
        "Is the author happy or unhappy based on this?"
    ]
    for instruction in natural:
        print(f"  • {instruction}")
    
    print("\nTradeoffs:\n")
    
    print("Templated:")
    print("  Pros: Consistent, easy to generate, clear format")
    print("  Cons: Less diverse, rigid, may not match user language")
    
    print("\nNatural:")
    print("  Pros: Diverse, realistic, better generalization")
    print("  Cons: Harder to create, inconsistent format")
    
    print("\nBest Practice:")
    print("  • Mix both templated and natural instructions")
    print("  • Use templates for initial coverage")
    print("  • Add natural variations for robustness")
    print("  • Human-written for most important tasks")

natural_vs_templated()
```

## Multi-Task Learning

### Task Interference and Transfer

```python
def task_interference_transfer():
    """Understanding task relationships in multi-task learning."""
    
    print("Task Interference and Transfer:\n")
    
    print("Positive Transfer (helpful):")
    positive_pairs = [
        ("Translate EN→FR", "Translate EN→ES") + ("Similar task structure",),
        ("Sentiment analysis", "Emotion detection") + ("Related classification",),
        ("Summarization", "Question answering") + ("Both require comprehension",),
        ("Math word problems", "Logical reasoning") + ("Shared reasoning skills",)
    ]
    
    for task1, task2, reason in positive_pairs:
        print(f"  {task1} → {task2}")
        print(f"    Reason: {reason}")
        print()
    
    print("Negative Interference (harmful):")
    negative_pairs = [
        ("Creative writing", "Factual QA") + ("Conflicts: creativity vs accuracy",),
        ("Code generation", "Natural language") + ("Different syntax/style",),
        ("Formal language", "Casual conversation") + ("Different tone requirements",)
    ]
    
    for task1, task2, reason in negative_pairs:
        print(f"  {task1} ⚡ {task2}")
        print(f"    Issue: {reason}")
        print()
    
    print("Mitigation Strategies:")
    print("  • Task clustering: Group similar tasks")
    print("  • Balanced sampling: Don't over-represent one type")
    print("  • Curriculum learning: Order tasks by difficulty")
    print("  • Task-specific adapters: Separate parameters per task")

task_interference_transfer()
```

### Scaling Number of Tasks

```python
def scaling_tasks():
    """How performance scales with number of instruction tasks."""
    
    print("\n\nScaling Number of Tasks:\n")
    
    print("Relationship: More diverse tasks → Better generalization")
    print()
    
    # Simulated scaling curve
    task_counts = [1, 10, 50, 100, 500, 1000]
    performance = [45, 58, 67, 72, 78, 80]
    
    print("Tasks  Zero-Shot Performance  Performance Curve")
    print("="*60)
    
    for tasks, perf in zip(task_counts, performance):
        bar = '█' * int(perf / 2)
        print(f"{tasks:>5}  {perf:>6.1f}%             {bar}")
    
    print("\nObservations:")
    print("  • Rapid gains from 1 → 100 tasks")
    print("  • Diminishing returns after ~100-500 tasks")
    print("  • Diversity matters more than count")
    print("  • Quality > Quantity (for individual examples)")
    
    print("\nFLAN Findings:")
    print("  • 62 tasks sufficient for strong zero-shot")
    print("  • Adding more similar tasks: little benefit")
    print("  • Adding new task types: significant benefit")
    
    print("\nPractical Recommendation:")
    print("  • Aim for 50-100+ diverse task types")
    print("  • Include multiple templates per task")
    print("  • Cover broad range of capabilities")
    print("  • Balance task categories")

scaling_tasks()
```

## Instruction Quality and Diversity

### Quality Criteria

```python
def instruction_quality():
    """What makes high-quality instruction data."""
    
    criteria = {
        'Clarity': {
            'good': 'Translate this English text to French: {text}',
            'bad': 'Do something with this: {text}',
            'why': 'Ambiguous what "something" means'
        },
        'Completeness': {
            'good': 'Classify sentiment as positive, negative, or neutral: {text}',
            'bad': 'Classify this: {text}',
            'why': 'Doesn\'t specify classes'
        },
        'Correctness': {
            'good': 'What is 15% of 200? Answer: 30',
            'bad': 'What is 15% of 200? Answer: 15',
            'why': 'Incorrect calculation'
        },
        'Consistency': {
            'good': 'Always use "Answer:" before response',
            'bad': 'Mix of "Answer:", "Response:", "Output:"',
            'why': 'Inconsistent formatting confuses model'
        },
        'Diversity': {
            'good': 'Multiple phrasings: "Translate", "Convert to", "Say this in"',
            'bad': 'Only "Translate to X: {text}"',
            'why': 'Limited instruction variation'
        }
    }
    
    print("Instruction Quality Criteria:\n")
    
    for criterion, examples in criteria.items():
        print(f"{criterion.upper()}:")
        print(f"  Good: {examples['good']}")
        print(f"  Bad:  {examples['bad']}")
        print(f"  Why:  {examples['why']}")
        print()
    
    print("Quality Assurance:")
    print("  • Manual review of samples")
    print("  • Automated filtering (length, diversity)")
    print("  • Test on validation set")
    print("  • Iterate based on model performance")

instruction_quality()
```

### Diversity Dimensions

```python
def instruction_diversity():
    """Different dimensions of instruction diversity."""
    
    print("\n\nInstruction Diversity Dimensions:\n")
    
    dimensions = {
        'Task type diversity': [
            'Classification, Generation, Extraction, Translation, QA, Reasoning'
        ],
        'Domain diversity': [
            'News, Science, Casual, Technical, Fiction, Formal'
        ],
        'Language diversity': [
            'Multiple languages (if multilingual)',
            'Formal vs casual language',
            'Technical vs layperson terminology'
        ],
        'Format diversity': [
            'Different instruction phrasings',
            'Various output formats',
            'With/without examples'
        ],
        'Complexity diversity': [
            'Simple single-step tasks',
            'Complex multi-step tasks',
            'Range of difficulty levels'
        ],
        'Length diversity': [
            'Short inputs (single sentence)',
            'Medium inputs (paragraph)',
            'Long inputs (documents)'
        ]
    }
    
    for dimension, examples in dimensions.items():
        print(f"{dimension.upper()}:")
        for example in examples:
            print(f"  • {example}")
        print()
    
    print("Why Diversity Matters:")
    print("  • Robust to different user phrasings")
    print("  • Generalizes to new task variations")
    print("  • Handles diverse real-world use cases")
    print("  • Reduces overfitting to specific formats")

instruction_diversity()
```

## Scaling Instruction Tuning

### Model Size Effects

```python
def model_size_effects():
    """How model size affects instruction tuning benefits."""
    
    print("Model Size Effects on Instruction Tuning:\n")
    
    # Performance by model size
    models = [
        ('125M', 15, 28, +87),
        ('350M', 22, 36, +64),
        ('1.3B', 31, 48, +55),
        ('6.7B', 39, 58, +49),
        ('30B', 48, 68, +42),
        ('175B', 54, 75, +39)
    ]
    
    print(f"{'Size':<10} {'Base (Zero-Shot)':<20} {'After Inst. Tuning':<20} {'Improvement'}")
    print("="*70)
    
    for size, base, tuned, improvement in models:
        print(f"{size:<10} {base:>6}%{' '*13} {tuned:>6}%{' '*13} +{improvement}%")
    
    print("\nKey Findings:")
    print("  • All sizes benefit from instruction tuning")
    print("  • Smaller models: Larger relative gains")
    print("  • Larger models: Higher absolute performance")
    print("  • Minimum ~1B parameters for good results")
    print("  • 7B+ models competitive with GPT-3.5 after tuning")
    
    print("\nImplications:")
    print("  • Can create useful assistants from smaller models")
    print("  • Instruction tuning democratizes LLM capabilities")
    print("  • Cost-performance tradeoff improves")

model_size_effects()
```

### Data Efficiency

```python
def instruction_data_efficiency():
    """How much instruction data is needed."""
    
    print("\n\nInstruction Data Efficiency:\n")
    
    # Performance vs dataset size
    data_sizes = [
        (1000, 45, 'Minimal improvement'),
        (5000, 58, 'Noticeable gains'),
        (10000, 65, 'Good performance'),
        (50000, 72, 'Strong performance'),
        (100000, 75, 'Diminishing returns'),
        (1000000, 78, 'Marginal gains')
    ]
    
    print(f"{'Num Examples':<15} {'Performance':<15} {'Notes'}")
    print("="*60)
    
    for size, perf, notes in data_sizes:
        print(f"{size:>10,}{' '*5} {perf:>5}%{' '*9} {notes}")
    
    print("\nSweetspot: ~10k-100k diverse examples")
    
    print("\nQuality vs Quantity:")
    print("  10k high-quality examples > 100k low-quality")
    print("  Diversity matters more than scale")
    print("  Even 1k examples can help (targeted domains)")
    
    print("\nPractical Guidelines:")
    print("  • Start with 10k: Good baseline")
    print("  • 50k: Strong general assistant")
    print("  • 100k+: Broad coverage, diminishing returns")
    print("  • Focus on diversity over raw count")

instruction_data_efficiency()
```

## Evaluation of Instruction-Tuned Models

### Evaluation Challenges

```python
def evaluation_challenges():
    """Challenges in evaluating instruction-tuned models."""
    
    print("Evaluation Challenges:\n")
    
    challenges = {
        'Open-ended generation': {
            'issue': 'No single correct answer',
            'example': '"Write a poem about rain"',
            'solution': 'Human evaluation, GPT-4 as judge'
        },
        'Instruction diversity': {
            'issue': 'Infinite ways to phrase instructions',
            'example': 'Synonyms, different levels of detail',
            'solution': 'Test on varied instruction phrasings'
        },
        'Subjective quality': {
            'issue': 'Helpfulness is subjective',
            'example': 'Concise vs detailed responses',
            'solution': 'Multiple human raters, consensus'
        },
        'Multi-dimensional quality': {
            'issue': 'Accuracy, coherence, safety, etc.',
            'example': 'Correct but offensive response',
            'solution': 'Evaluate multiple dimensions separately'
        },
        'Benchmark saturation': {
            'issue': 'Models may have seen benchmarks',
            'example': 'Training data contamination',
            'solution': 'New benchmarks, held-out tasks'
        }
    }
    
    for challenge, info in challenges.items():
        print(f"{challenge.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

evaluation_challenges()
```

### Evaluation Methods

```python
def evaluation_methods():
    """Methods for evaluating instruction-tuned models."""
    
    print("\n\nEvaluation Methods:\n")
    
    methods = {
        'Benchmark Performance': {
            'description': 'Test on standard NLP benchmarks',
            'examples': 'MMLU, BBH, HellaSwag, TruthfulQA',
            'pros': 'Objective, comparable across models',
            'cons': 'May not reflect real-world use'
        },
        'Human Evaluation': {
            'description': 'Humans rate model responses',
            'examples': 'Helpfulness, harmlessness, honesty',
            'pros': 'Captures real-world quality',
            'cons': 'Expensive, slow, subjective'
        },
        'Win-Rate Comparisons': {
            'description': 'Humans compare two model outputs',
            'examples': 'A vs B preference judgments',
            'pros': 'Easier than absolute rating',
            'cons': 'Requires pairwise comparisons'
        },
        'LLM-as-Judge': {
            'description': 'Use GPT-4 to evaluate responses',
            'examples': 'Rate quality 1-10, provide feedback',
            'pros': 'Scalable, consistent',
            'cons': 'GPT-4 biases, not ground truth'
        },
        'Task Success Rate': {
            'description': 'Did model complete the task?',
            'examples': 'Code executes, extraction is correct',
            'pros': 'Clear pass/fail',
            'cons': 'Only for verifiable tasks'
        }
    }
    
    for method, info in methods.items():
        print(f"{method.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Best Practice:")
    print("  • Combine multiple evaluation methods")
    print("  • Use benchmarks for objective metrics")
    print("  • Use human eval for subjective quality")
    print("  • Track diverse dimensions (accuracy, safety, etc.)")

evaluation_methods()
```

## From Instruction Tuning to RLHF

### The Pipeline

```python
def instruction_to_rlhf_pipeline():
    """Complete pipeline from base model to aligned assistant."""
    
    print("From Base Model to Aligned Assistant:\n")
    
    stages = [
        {
            'stage': 'Stage 0: Pre-training',
            'input': 'Raw text (web, books, code)',
            'process': 'Next-token prediction',
            'output': 'Base language model',
            'cost': 'Millions of dollars, months',
            'result': 'Knowledge but no instruction-following'
        },
        {
            'stage': 'Stage 1: Instruction Tuning (SFT)',
            'input': 'Instruction-response pairs',
            'process': 'Supervised fine-tuning',
            'output': 'Instruction-following model',
            'cost': 'Thousands of dollars, days',
            'result': 'Follows instructions, still has issues'
        },
        {
            'stage': 'Stage 2: Reward Modeling',
            'input': 'Human preference comparisons',
            'process': 'Train reward model',
            'output': 'Model that predicts human preferences',
            'cost': 'Thousands of dollars, days',
            'result': 'Can evaluate response quality'
        },
        {
            'stage': 'Stage 3: RLHF',
            'input': 'Reward model + RL',
            'process': 'Optimize for human preferences',
            'output': 'Aligned assistant',
            'cost': 'Tens of thousands, weeks',
            'result': 'Helpful, harmless, honest'
        }
    ]
    
    for i, stage_info in enumerate(stages, 0):
        print(f"{stage_info['stage']}:")
        print(f"  Input:   {stage_info['input']}")
        print(f"  Process: {stage_info['process']}")
        print(f"  Output:  {stage_info['output']}")
        print(f"  Cost:    {stage_info['cost']}")
        print(f"  Result:  {stage_info['result']}")
        if i < len(stages) - 1:
            print("  " + "↓")
        print()
    
    print("Key Insight:")
    print("  Instruction tuning teaches 'what' (follow instructions)")
    print("  RLHF teaches 'how' (do it according to human values)")

instruction_to_rlhf_pipeline()
```

### Limitations of Instruction Tuning Alone

```python
def instruction_tuning_limitations():
    """Why RLHF is needed after instruction tuning."""
    
    print("\n\nLimitations of Instruction Tuning Alone:\n")
    
    limitations = {
        'Objective mismatch': {
            'problem': 'Training maximizes likelihood, not quality',
            'example': 'Model may generate likely but unhelpful text',
            'rlhf_solution': 'Directly optimize for human preferences'
        },
        'No notion of quality': {
            'problem': 'All outputs in training data treated equal',
            'example': 'Good and mediocre responses weighted same',
            'rlhf_solution': 'Learn to distinguish quality levels'
        },
        'Distribution mismatch': {
            'problem': 'Training data may not match deployment',
            'example': 'Training has formal instructions, users are casual',
            'rlhf_solution': 'Adapt to actual user interactions'
        },
        'Safety issues': {
            'problem': 'May generate harmful content',
            'example': 'Follows harmful instructions in training data',
            'rlhf_solution': 'Incorporate safety constraints'
        },
        'Verbosity and style': {
            'problem': 'May be too verbose or have wrong style',
            'example': 'Overly long responses or wrong tone',
            'rlhf_solution': 'Learn preferred length and style'
        }
    }
    
    for limitation, info in limitations.items():
        print(f"{limitation.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    print("Conclusion:")
    print("  • Instruction tuning: Necessary foundation")
    print("  • RLHF: Polishes and aligns the model")
    print("  • Both stages complement each other")
    print("  • Instruction tuning (SFT) + RLHF = State-of-the-art")

instruction_tuning_limitations()
```

## Summary

**Key Concepts**:

1. **Instruction tuning** fine-tunes pre-trained models on instruction-response pairs
2. **Transforms base models** from text predictors to instruction-following assistants
3. **Enables zero-shot generalization** to unseen tasks via natural language instructions
4. **Requires diverse instruction datasets** covering many task types and formats
5. **Supervised fine-tuning (SFT)** is the standard training method
6. **Multi-task learning** improves generalization across task categories
7. **RLHF follows instruction tuning** to align models with human preferences

**Instruction Dataset Types**:

```
Task-specific → Multi-task mixtures → Human-written → LLM-generated → Conversational
     |                |                    |                |              |
  (Narrow)      (Template-based)      (Natural)        (Scalable)    (Interactive)
```

**Key Datasets/Methods**:

- **FLAN**: 62 tasks, multiple templates per task, strong zero-shot
- **T0**: 47 datasets with varied prompts, demonstrates generalization
- **Self-Instruct**: Bootstrap from seed tasks using LLM generation
- **Alpaca**: 52k LLM-generated instructions, low-cost dataset creation
- **InstructGPT**: Instruction tuning + RLHF, aligned assistant

**Performance Gains**:

| Model | Base (Zero-Shot) | After Instruction Tuning | Improvement |
|-------|------------------|--------------------------|-------------|
| Small (1-3B) | ~30% | ~50% | +65% |
| Medium (7-13B) | ~40% | ~65% | +63% |
| Large (60B+) | ~50% | ~75% | +50% |

**Best Practices**:

1. **Dataset**: 10k-100k diverse examples, multiple task types
2. **Quality > Quantity**: Diverse, high-quality examples beat scale
3. **Instruction diversity**: Multiple phrasings per task type
4. **Format consistency**: Clear, unambiguous instructions
5. **Multi-task mixing**: Balance task categories
6. **Evaluation**: Combine benchmarks + human evaluation
7. **Follow with RLHF**: For alignment and safety

**Instruction Format**:

```
[Task Description] + [Input] + [Output Specification]

Example:
"Translate the following English text to French:
Text: Hello, how are you?
Translation:"
```

**Data Efficiency**:

- ~1k examples: Minimal gains
- ~10k examples: Noticeable improvement
- ~50k examples: Strong performance
- ~100k+ examples: Diminishing returns

**Limitations**:

- Not aligned with human preferences (needs RLHF)
- May generate harmful content if in training data
- Optimizes likelihood, not quality
- Can be verbose or have wrong style
- Dataset biases transfer to model

## Next Steps

- Study [RLHF and Alignment](../rlhf_and_alignment/rlhf-fundamentals.md) for preference optimization
- Learn [LLM Capabilities and Limitations](capabilities-limitations.md) for realistic expectations
- Explore [Prompt Engineering](../prompt_engineering/instruction-design.md) for effective instructions
- Understand [Evaluation Methods](../evaluation/human-evaluation.md) for measuring quality
- Study [Model Training](../mlops-aiops-lab/training/) for implementation details
- Learn [In-Context Learning](in-context-learning.md) as complement to instruction tuning

