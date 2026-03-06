# Pretraining and Transfer Learning

## Table of Contents

- [Introduction](#introduction)
- [The Pretraining-Finetuning Paradigm](#the-pretraining-finetuning-paradigm)
- [Pretraining Objectives](#pretraining-objectives)
- [What Models Learn During Pretraining](#what-models-learn-during-pretraining)
- [Transfer Learning Strategies](#transfer-learning-strategies)
- [Domain Adaptation](#domain-adaptation)
- [Few-Shot and Zero-Shot Learning](#few-shot-and-zero-shot-learning)
- [Efficient Transfer Learning](#efficient-transfer-learning)
- [Pretraining Data](#pretraining-data)
- [Scaling Laws](#scaling-laws)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Pretraining and transfer learning** have revolutionized NLP by enabling models to learn general language understanding from massive unlabeled datasets, then transfer that knowledge to specific tasks with minimal labeled data.

```
Transfer Learning Pipeline:

┌─────────────────────────────────────────────┐
│      PRETRAINING (Unsupervised)             │
│                                             │
│  Massive unlabeled corpus (billions tokens)│
│  Self-supervised objectives (MLM, CLM)     │
│  Learn general language representations    │
│                                             │
│  Result: Pretrained model                  │
└──────────────────┬──────────────────────────┘
                   │ Transfer
                   ↓
┌─────────────────────────────────────────────┐
│      FINE-TUNING (Supervised)               │
│                                             │
│  Small labeled dataset (1K-100K examples)  │
│  Task-specific objective                   │
│  Adapt representations to task             │
│                                             │
│  Result: Task-specific model               │
└─────────────────────────────────────────────┘
```

**Key idea**: Learning from scratch requires enormous labeled data. Pretraining on unlabeled data + fine-tuning on small labeled data = better performance with less annotation.

This guide covers pretraining objectives, what models learn, and how to effectively transfer knowledge to downstream tasks.

## The Pretraining-Finetuning Paradigm

### Why Pretrain?

```python
def explain_pretraining_benefits():
    """Why pretraining works so well."""

    benefits = {
        'Leverage unlabeled data': {
            'problem': 'Labeled data is expensive and limited',
            'solution': 'Billions of tokens of unlabeled text available',
            'impact': 'Learn from vast amounts of data'
        },
        'Learn general representations': {
            'problem': 'Training from scratch overfits small datasets',
            'solution': 'Pretrain on diverse data for general knowledge',
            'impact': 'Better initialization for downstream tasks'
        },
        'Transfer knowledge': {
            'problem': 'Each task learned independently',
            'solution': 'Share linguistic knowledge across tasks',
            'impact': 'Faster convergence, better performance'
        },
        'Low-resource effectiveness': {
            'problem': 'Many domains lack labeled data',
            'solution': 'Transfer from high-resource pretraining',
            'impact': 'Good performance with few examples'
        }
    }

    print("Why Pretraining Works:\n")
    for benefit, info in benefits.items():
        print(f"{benefit.upper()}:")
        print(f"  Problem: {info['problem']}")
        print(f"  Solution: {info['solution']}")
        print(f"  Impact: {info['impact']}")
        print()

explain_pretraining_benefits()
```

### The Two-Stage Process

```python
class PretrainingFinetuningPipeline:
    """Complete pipeline from pretraining to deployment."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def pretrain(self, unlabeled_corpus, objective='mlm', steps=1000000):
        """
        Stage 1: Pretrain on large unlabeled corpus.

        Args:
            unlabeled_corpus: Billions of tokens
            objective: 'mlm', 'clm', 'span_corruption'
            steps: Training steps (millions)

        Result: Model with general language understanding
        """
        print(f"Pretraining with {objective.upper()} objective...")
        print(f"  Corpus size: {len(unlabeled_corpus)} examples")
        print(f"  Training steps: {steps:,}")
        print(f"  Learning: Syntax, semantics, world knowledge")

        # Pseudocode for pretraining
        for step in range(steps):
            # Sample batch from corpus
            batch = self.sample_batch(unlabeled_corpus)

            # Apply pretraining objective
            if objective == 'mlm':
                inputs, labels = self.mask_tokens(batch)
            elif objective == 'clm':
                inputs, labels = self.shift_right(batch)
            else:
                inputs, labels = self.span_corruption(batch)

            # Train
            loss = self.model.train_step(inputs, labels)

        print("Pretraining complete!")
        return self.model

    def finetune(self, labeled_data, task='classification', epochs=3):
        """
        Stage 2: Fine-tune on task-specific labeled data.

        Args:
            labeled_data: Small labeled dataset (1K-100K)
            task: Downstream task
            epochs: Few epochs (3-5)

        Result: Task-specific model
        """
        print(f"\nFine-tuning for {task}...")
        print(f"  Labeled examples: {len(labeled_data)}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning: Task-specific patterns")

        # Add task-specific head
        self.model.add_task_head(task)

        # Fine-tune with small learning rate
        for epoch in range(epochs):
            for batch in labeled_data:
                inputs, labels = batch
                loss = self.model.train_step(inputs, labels, lr=2e-5)

        print("Fine-tuning complete!")
        return self.model

print("Pretraining-Finetuning Pipeline:\n")
print("Stage 1: PRETRAINING")
print("  • Data: Unlabeled (Wikipedia, books, web)")
print("  • Objective: Self-supervised (MLM, CLM)")
print("  • Duration: Days/weeks on many GPUs")
print("  • Result: General language model")
print("\nStage 2: FINE-TUNING")
print("  • Data: Labeled (task-specific)")
print("  • Objective: Supervised (classification, QA, etc.)")
print("  • Duration: Hours/days")
print("  • Result: Task-specific model")
```

### Historical Context

```python
def nlp_paradigm_evolution():
    """Evolution of NLP paradigms."""

    paradigms = {
        'Pre-2013: Feature Engineering': {
            'approach': 'Handcrafted features + ML classifier',
            'example': 'TF-IDF + Naive Bayes',
            'limitation': 'Labor-intensive, task-specific'
        },
        '2013-2017: Task-Specific Neural': {
            'approach': 'Train neural network per task',
            'example': 'CNN/LSTM for sentiment analysis',
            'limitation': 'Requires large labeled dataset per task'
        },
        '2018-Present: Pretrain-Finetune': {
            'approach': 'Pretrain once, finetune for many tasks',
            'example': 'BERT pretraining → fine-tune for 100+ tasks',
            'breakthrough': 'Massive performance gains with less data'
        },
        '2020-Present: Prompt-Based': {
            'approach': 'Pretrain at scale, prompt without finetuning',
            'example': 'GPT-3 few-shot learning',
            'advancement': 'Zero-shot or few-shot, no fine-tuning'
        }
    }

    print("Evolution of NLP Paradigms:\n")
    for era, info in paradigms.items():
        print(f"{era}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

nlp_paradigm_evolution()
```

## Pretraining Objectives

### Masked Language Modeling (MLM)

```python
def mlm_objective(tokens, mask_prob=0.15, vocab_size=30000):
    """
    Masked Language Modeling (BERT-style).

    Objective: Predict randomly masked tokens
    Loss: Cross-entropy on masked positions
    """
    masked_tokens = tokens.clone()
    labels = tokens.clone()

    # Mask 15% of tokens
    mask = torch.rand(tokens.shape) < mask_prob

    # 80%: [MASK], 10%: random, 10%: original
    mask_token = 103  # [MASK]

    replace_with_mask = mask & (torch.rand(tokens.shape) < 0.8)
    replace_with_random = mask & ~replace_with_mask & (torch.rand(tokens.shape) < 0.5)

    masked_tokens[replace_with_mask] = mask_token
    masked_tokens[replace_with_random] = torch.randint(0, vocab_size, tokens.shape)[replace_with_random]

    # Label only masked positions
    labels[~mask] = -100  # Ignore in loss

    return masked_tokens, labels

print("Masked Language Modeling (MLM):\n")
print("  Task: Predict masked tokens using bidirectional context")
print("  Formula: P(w_i | w_{-i})")
print("  Example:")
print("    Input:  The [MASK] sat on the [MASK]")
print("    Labels: cat, mat")
print("  Models: BERT, RoBERTa, ALBERT")
print("\n  Strengths:")
print("    • Bidirectional context")
print("    • Strong understanding representations")
print("  Weaknesses:")
print("    • Training-inference mismatch ([MASK] only during training)")
print("    • Not natural for generation")
```

### Causal Language Modeling (CLM)

```python
def clm_objective(tokens):
    """
    Causal Language Modeling (GPT-style).

    Objective: Predict next token given previous tokens
    Loss: Cross-entropy on all positions
    """
    # Input: tokens[:-1]
    # Labels: tokens[1:]

    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]

    return inputs, labels

print("\nCausal Language Modeling (CLM):\n")
print("  Task: Predict next token given previous tokens")
print("  Formula: P(w_i | w_{<i})")
print("  Example:")
print("    Input:  The cat sat on")
print("    Predict: the")
print("  Models: GPT, GPT-2, GPT-3")
print("\n  Strengths:")
print("    • Natural for generation")
print("    • No training-inference mismatch")
print("  Weaknesses:")
print("    • Unidirectional (only left context)")
print("    • Less effective for understanding tasks")
```

### Span Corruption

```python
def span_corruption_objective(tokens, corruption_rate=0.15, mean_span_length=3):
    """
    Span Corruption (T5-style).

    Objective: Reconstruct corrupted spans
    Loss: Cross-entropy on corrupted spans
    """
    n = len(tokens)
    corrupted = []
    target = []

    sentinel_id = 0
    i = 0

    while i < n:
        # Randomly corrupt spans
        if random.random() < corruption_rate:
            span_len = random.randint(1, mean_span_length)
            span = tokens[i:i+span_len]

            # Replace span with sentinel
            corrupted.append(f"<extra_id_{sentinel_id}>")

            # Target: sentinel + original span
            target.append(f"<extra_id_{sentinel_id}>")
            target.extend(span)

            sentinel_id += 1
            i += span_len
        else:
            corrupted.append(tokens[i])
            i += 1

    return corrupted, target

print("\nSpan Corruption:\n")
print("  Task: Reconstruct corrupted text spans")
print("  Example:")
print("    Input:  Thank <X> for <Y> lovely party")
print("    Target: <X> you <Y> the")
print("  Models: T5, UL2")
print("\n  Strengths:")
print("    • Longer context than single tokens")
print("    • Encoder-decoder naturally suited")
print("  Weaknesses:")
print("    • More complex than MLM or CLM")
```

### Denoising Autoencoding

```python
def denoising_objectives():
    """BART-style denoising with multiple corruption types."""

    corruptions = {
        'Token Masking': 'Replace random tokens with [MASK]',
        'Token Deletion': 'Delete random tokens',
        'Text Infilling': 'Replace spans with single [MASK]',
        'Sentence Permutation': 'Shuffle sentences',
        'Document Rotation': 'Rotate document to random start'
    }

    print("Denoising Autoencoding (BART):\n")
    print("  Task: Reconstruct original from corrupted input")
    print("  Multiple corruption strategies:\n")

    for corruption, description in corruptions.items():
        print(f"    • {corruption}: {description}")

    print("\n  Model: BART, mBART")
    print("  Benefit: Robust to various types of noise")

denoising_objectives()
```

### Comparison of Objectives

```python
objective_comparison = {
    'Objective': {
        'MLM': 'Predict masked tokens',
        'CLM': 'Predict next token',
        'Span Corruption': 'Reconstruct spans',
        'Denoising': 'Reconstruct corrupted'
    },
    'Architecture': {
        'MLM': 'Encoder-only',
        'CLM': 'Decoder-only',
        'Span Corruption': 'Encoder-decoder',
        'Denoising': 'Encoder-decoder'
    },
    'Context': {
        'MLM': 'Bidirectional',
        'CLM': 'Unidirectional',
        'Span Corruption': 'Bidirectional encoder',
        'Denoising': 'Bidirectional encoder'
    },
    'Best for': {
        'MLM': 'Understanding tasks',
        'CLM': 'Generation tasks',
        'Span Corruption': 'Seq2seq tasks',
        'Denoising': 'Robust representations'
    }
}

print("\n\nPretraining Objectives Comparison:\n")
print(f"{'Aspect':<15} {'MLM':<25} {'CLM':<25} {'Span Corruption':<25} {'Denoising':<25}")
print("=" * 115)

for aspect, values in objective_comparison.items():
    print(f"{aspect:<15} {values['MLM']:<25} {values['CLM']:<25} {values['Span Corruption']:<25} {values['Denoising']:<25}")
```

## What Models Learn During Pretraining

### Linguistic Knowledge

```python
def linguistic_knowledge_learned():
    """What linguistic knowledge emerges from pretraining."""

    knowledge_types = {
        'Syntactic': {
            'phenomena': [
                'Part-of-speech tags',
                'Dependency parsing',
                'Constituency structure',
                'Subject-verb agreement'
            ],
            'evidence': 'Probing studies show syntax in middle layers'
        },
        'Semantic': {
            'phenomena': [
                'Word sense disambiguation',
                'Semantic roles',
                'Named entities',
                'Coreference'
            ],
            'evidence': 'Strong performance on semantic tasks'
        },
        'World Knowledge': {
            'phenomena': [
                'Factual knowledge',
                'Common sense reasoning',
                'Entity relationships',
                'Temporal reasoning'
            ],
            'evidence': 'Can answer factual questions'
        },
        'Pragmatic': {
            'phenomena': [
                'Sentiment',
                'Sarcasm (limited)',
                'Implicature',
                'Discourse structure'
            ],
            'evidence': 'Good sentiment analysis, discourse parsing'
        }
    }

    print("Linguistic Knowledge Learned During Pretraining:\n")
    for knowledge_type, info in knowledge_types.items():
        print(f"{knowledge_type}:")
        print(f"  Phenomena:")
        for phenomenon in info['phenomena']:
            print(f"    • {phenomenon}")
        print(f"  Evidence: {info['evidence']}")
        print()

linguistic_knowledge_learned()
```

### Layer-wise Analysis

```python
def layer_wise_knowledge():
    """What different layers learn."""

    print("Layer-wise Knowledge in Pretrained Models:\n")

    layers = {
        'Lower Layers (1-4)': {
            'knowledge': 'Surface-level syntax',
            'examples': [
                'Part-of-speech tags',
                'Phrase structure',
                'Morphology'
            ]
        },
        'Middle Layers (5-8)': {
            'knowledge': 'Semantics and syntax',
            'examples': [
                'Dependency relations',
                'Semantic roles',
                'Named entities'
            ]
        },
        'Upper Layers (9-12)': {
            'knowledge': 'Task-specific',
            'examples': [
                'Task-dependent patterns',
                'More abstract representations',
                'Downstream task features'
            ]
        }
    }

    for layer_range, info in layers.items():
        print(f"{layer_range}:")
        print(f"  Knowledge: {info['knowledge']}")
        print(f"  Examples:")
        for example in info['examples']:
            print(f"    • {example}")
        print()

    print("Implication: Can extract different layer for different tasks")
    print("  • POS tagging: Use lower layers")
    print("  • Semantic similarity: Use middle layers")
    print("  • Task-specific: Use upper layers (after fine-tuning)")

layer_wise_knowledge()
```

### Attention Patterns

```python
def attention_pattern_analysis():
    """What attention heads learn."""

    print("\n\nAttention Head Specialization:\n")

    patterns = {
        'Syntactic heads': {
            'pattern': 'Attend to syntactic relations',
            'example': 'Attend from verb to subject/object'
        },
        'Positional heads': {
            'pattern': 'Attend to nearby tokens',
            'example': 'Attend to next/previous word'
        },
        'Semantic heads': {
            'pattern': 'Attend to semantically related words',
            'example': 'Attend from pronoun to antecedent'
        },
        'Rare heads': {
            'pattern': 'Attend to rare or special tokens',
            'example': 'Attend to [CLS] or [SEP] tokens'
        },
        'Broadcast heads': {
            'pattern': 'Attend uniformly (no specialization)',
            'example': 'Distribute attention evenly'
        }
    }

    for head_type, info in patterns.items():
        print(f"{head_type}:")
        print(f"  Pattern: {info['pattern']}")
        print(f"  Example: {info['example']}")
        print()

    print("Key insight: Different heads specialize in different linguistic phenomena")

attention_pattern_analysis()
```

## Transfer Learning Strategies

### Full Fine-tuning

```python
class FullFineTuning:
    """Traditional fine-tuning: Update all parameters."""

    def __init__(self, pretrained_model, num_labels):
        self.model = pretrained_model

        # Add task-specific head
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, num_labels)

    def finetune(self, train_data, epochs=3, lr=2e-5):
        """
        Fine-tune all parameters.

        Best practices:
          • Small learning rate (2e-5 to 5e-5)
          • Few epochs (3-5)
          • Gradient clipping
          • Warmup
        """
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.classifier.parameters()),
            lr=lr,
            weight_decay=0.01
        )

        for epoch in range(epochs):
            for batch in train_data:
                # Forward pass
                outputs = self.model(batch['input_ids'])
                logits = self.classifier(outputs[:, 0, :])  # Use [CLS]

                # Compute loss
                loss = nn.CrossEntropyLoss()(logits, batch['labels'])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

print("Full Fine-tuning:")
print("  • Update: All model parameters")
print("  • Data: 1K+ labeled examples")
print("  • Time: Hours to days")
print("  • Performance: Best on task")
print("  • Drawback: Separate model per task")
```

### Feature Extraction (Frozen)

```python
class FeatureExtraction:
    """Use pretrained model as fixed feature extractor."""

    def __init__(self, pretrained_model, num_labels):
        self.model = pretrained_model

        # Freeze pretrained weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Only train classifier
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, num_labels)

    def train(self, train_data, epochs=10, lr=1e-3):
        """
        Train only classifier head.

        Advantages:
          • Fast training
          • Low memory
          • Good for small datasets
        """
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch in train_data:
                # Extract features (no gradients)
                with torch.no_grad():
                    features = self.model(batch['input_ids'])[:, 0, :]

                # Train classifier
                logits = self.classifier(features)
                loss = nn.CrossEntropyLoss()(logits, batch['labels'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

print("\n\nFeature Extraction (Frozen):")
print("  • Update: Only classifier head")
print("  • Data: Few hundred examples")
print("  • Time: Minutes")
print("  • Performance: Good baseline")
print("  • Advantage: Very fast, low memory")
```

### Gradual Unfreezing

```python
class GradualUnfreezing:
    """Unfreeze layers progressively during fine-tuning."""

    def __init__(self, model, num_labels):
        self.model = model
        self.classifier = nn.Linear(model.config.hidden_size, num_labels)
        self.num_layers = len(model.encoder.layer)

    def finetune_gradual(self, train_data, epochs_per_stage=2):
        """
        Gradually unfreeze layers from top to bottom.

        Strategy:
          1. Train only classifier
          2. Unfreeze top layer
          3. Unfreeze next layer
          4. Continue until all unfrozen
        """
        # Stage 1: Only classifier
        print("Stage 1: Train classifier only")
        self._freeze_all()
        self._train(train_data, epochs_per_stage, lr=1e-3)

        # Stage 2+: Gradually unfreeze
        for layer_idx in range(self.num_layers - 1, -1, -1):
            print(f"Stage {self.num_layers - layer_idx + 1}: Unfreeze layer {layer_idx}")
            self._unfreeze_layer(layer_idx)
            self._train(train_data, epochs_per_stage, lr=2e-5)

    def _freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_layer(self, layer_idx):
        for param in self.model.encoder.layer[layer_idx].parameters():
            param.requires_grad = True

    def _train(self, data, epochs, lr):
        # Training loop...
        pass

print("\n\nGradual Unfreezing:")
print("  • Strategy: Unfreeze layers progressively")
print("  • Benefit: More stable, prevents catastrophic forgetting")
print("  • When: Limited data, want careful adaptation")
```

### Layer-wise Learning Rates

```python
def discriminative_fine_tuning(model, base_lr=2e-5, decay_factor=0.95):
    """
    Use different learning rates for different layers.

    Intuition:
      • Lower layers: More general, need less updating
      • Upper layers: More task-specific, can change more
    """
    param_groups = []

    num_layers = len(model.encoder.layer)

    # Create parameter groups with decaying LR
    for layer_idx in range(num_layers):
        lr = base_lr * (decay_factor ** (num_layers - layer_idx - 1))

        param_groups.append({
            'params': model.encoder.layer[layer_idx].parameters(),
            'lr': lr
        })

    # Highest LR for task head
    param_groups.append({
        'params': model.classifier.parameters(),
        'lr': base_lr * 10  # Even higher for new head
    })

    optimizer = torch.optim.AdamW(param_groups)

    return optimizer

print("\n\nDiscriminative Fine-tuning:")
print("  • Technique: Different LR per layer")
print("  • Lower layers: Smaller LR (more general)")
print("  • Upper layers: Larger LR (more task-specific)")
print("  • Task head: Highest LR (random initialization)")
print("  • Benefit: Better adaptation, less overfitting")
```

## Domain Adaptation

### Continued Pretraining

```python
def continued_pretraining(model, domain_corpus, steps=10000):
    """
    Continue pretraining on domain-specific unlabeled data.

    Use case: Adapting to specific domain (medical, legal, etc.)
    """
    print("Continued Pretraining:\n")
    print(f"  Domain corpus: {domain_corpus}")
    print(f"  Additional steps: {steps:,}")
    print("\n  Process:")
    print("    1. Load pretrained model (e.g., BERT)")
    print("    2. Continue MLM training on domain corpus")
    print("    3. Adapt vocabulary and representations")
    print("    4. Fine-tune on downstream task")
    print("\n  Example:")
    print("    BERT → BioBERT (medical texts)")
    print("    BERT → SciBERT (scientific papers)")
    print("    GPT-2 → CodeGPT (code repositories)")

continued_pretraining("BERT", "Medical texts (PubMed)", 100000)
```

### Domain-Adaptive Pretraining

```python
def domain_adaptive_pretraining_strategies():
    """Strategies for domain adaptation."""

    strategies = {
        'Task-Adaptive Pretraining (TAPT)': {
            'data': 'Unlabeled data from target task',
            'method': 'Continue pretraining on task-specific unlabeled data',
            'benefit': 'Most relevant adaptation',
            'example': 'IMDb reviews for sentiment on IMDb'
        },
        'Domain-Adaptive Pretraining (DAPT)': {
            'data': 'Unlabeled data from target domain',
            'method': 'Continue pretraining on domain corpus',
            'benefit': 'Broad domain knowledge',
            'example': 'Biomedical texts for medical NLP'
        },
        'Combined (DAPT + TAPT)': {
            'data': 'Domain corpus then task data',
            'method': 'DAPT followed by TAPT',
            'benefit': 'Best performance',
            'example': 'Biomedical corpus → specific dataset'
        }
    }

    print("\nDomain Adaptation Strategies:\n")
    for strategy, info in strategies.items():
        print(f"{strategy}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

domain_adaptive_pretraining_strategies()
```

## Few-Shot and Zero-Shot Learning

### Zero-Shot Learning

```python
def zero_shot_learning():
    """Perform tasks without any task-specific training."""

    print("Zero-Shot Learning:\n")
    print("  Definition: Perform task with NO task-specific examples")
    print("  Mechanism: Task description in natural language")
    print("\n  Example (Sentiment Analysis):")
    print("    Prompt: 'Classify sentiment as positive or negative:'")
    print("    Input: 'This movie is amazing!'")
    print("    Output: 'positive'")
    print("\n  Requirements:")
    print("    • Large pretrained model (GPT-3, T5)")
    print("    • Clear task description")
    print("    • Model has seen similar patterns in pretraining")
    print("\n  Advantages:")
    print("    • No labeled data needed")
    print("    • No training time")
    print("    • Flexible task specification")
    print("\n  Limitations:")
    print("    • Lower performance than fine-tuning")
    print("    • Sensitive to prompt wording")
    print("    • Requires very large models")

zero_shot_learning()
```

### Few-Shot Learning

```python
def few_shot_learning_example():
    """Learn from few examples in context."""

    print("\n\nFew-Shot Learning:\n")
    print("  Definition: Learn from K examples (K=1-10)")
    print("  Mechanism: Provide examples in prompt (in-context learning)")
    print("\n  Example (3-shot sentiment):")

    prompt = """
    Classify the sentiment of these reviews:

    Review: "Great movie!" Sentiment: positive
    Review: "Terrible acting." Sentiment: negative
    Review: "Not bad." Sentiment: neutral

    Review: "Absolutely loved it!" Sentiment:"""

    print(prompt)
    print("\n    Model predicts: 'positive'")
    print("\n  Advantages:")
    print("    • Better than zero-shot")
    print("    • No fine-tuning needed")
    print("    • Quick experimentation")
    print("\n  Limitations:")
    print("    • Still below fine-tuned performance")
    print("    • Limited context window")
    print("    • Example selection matters")

few_shot_learning_example()
```

### Prompt Tuning

```python
class PromptTuning:
    """Learn continuous prompts instead of discrete tokens."""

    def __init__(self, model, prompt_length=20):
        self.model = model

        # Freeze model
        for param in model.parameters():
            param.requires_grad = False

        # Learn continuous prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, model.config.hidden_size)
        )

    def forward(self, input_ids):
        # Get input embeddings
        input_embeds = self.model.embeddings(input_ids)

        # Prepend learnable prompt
        batch_size = input_embeds.size(0)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        combined = torch.cat([prompt, input_embeds], dim=1)

        # Forward through model
        outputs = self.model(inputs_embeds=combined)

        return outputs

print("\n\nPrompt Tuning:")
print("  • Learn: Continuous prompt embeddings")
print("  • Freeze: All model parameters")
print("  • Parameters: Only ~0.01% of model")
print("  • Performance: Comparable to full fine-tuning (at scale)")
print("  • Benefit: One model, multiple tasks via different prompts")
```

## Efficient Transfer Learning

### Adapter Layers

```python
class AdapterLayer(nn.Module):
    """
    Adapter: Small bottleneck layers inserted into pretrained model.

    Advantage: Only train adapters, freeze main model
    """

    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()

        # Bottleneck: down-project → nonlinearity → up-project
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # Residual connection
        residual = hidden_states

        # Adapter transformation
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.up_project(hidden_states)

        # Add residual
        return hidden_states + residual

print("Adapter Layers:")
print("  • Insert small bottleneck layers into frozen model")
print("  • Train only adapters (~1% of parameters)")
print("  • Performance: 95-98% of full fine-tuning")
print("  • Benefit: Can swap adapters for different tasks")
print("\n  Architecture:")
print("    Input → [Frozen Transformer] → Adapter → Output")
print("           ↓                         ↑")
print("           └─── Residual connection ──┘")
```

### LoRA (Low-Rank Adaptation)

```python
class LoRALayer(nn.Module):
    """
    LoRA: Learn low-rank updates to weight matrices.

    W_new = W_frozen + A @ B
    where A: [d, r], B: [r, k], r << min(d, k)
    """

    def __init__(self, in_features, out_features, rank=8):
        super().__init__()

        # Freeze original weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # Low-rank decomposition
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))

        self.rank = rank
        self.scaling = 0.01  # Scale low-rank update

    def forward(self, x):
        # Original transformation
        result = F.linear(x, self.weight)

        # Add low-rank update
        lora_update = (x @ self.lora_A.T) @ self.lora_B.T
        result += self.scaling * lora_update

        return result

print("\n\nLoRA (Low-Rank Adaptation):")
print("  • Freeze: Pretrained weights")
print("  • Learn: Low-rank updates (rank=4-8)")
print("  • Parameters: ~0.1% of model")
print("  • Performance: Matches full fine-tuning")
print("  • Benefits:")
print("    - Extremely parameter-efficient")
print("    - No inference latency")
print("    - Can merge into original weights")
```

### QLoRA (Quantized LoRA)

```python
class QLoRALayer(nn.Module):
    """
    QLoRA: Combines quantization + LoRA for extreme memory efficiency.

    Key innovations:
    1. 4-bit NormalFloat (NF4) quantization of base model
    2. Double quantization (quantize the quantization constants)
    3. Paged optimizers for memory spikes
    4. LoRA adapters in full precision (16-bit)

    Result: Fine-tune 65B models on single 48GB GPU!
    """

    def __init__(self, in_features, out_features, rank=8):
        super().__init__()

        # Base weight quantized to 4-bit NF4
        self.weight_4bit = self.quantize_nf4(
            torch.randn(out_features, in_features)
        )

        # LoRA adapters in full precision (16-bit/32-bit)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))

        self.rank = rank
        self.scaling = 0.01

    def quantize_nf4(self, weight):
        """
        4-bit NormalFloat quantization.

        NF4 is information-theoretically optimal for normally distributed weights.
        """
        # Compute quantization constants
        absmax = weight.abs().max()

        # NF4 quantization levels (asymmetric, optimized for normal distribution)
        nf4_levels = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0
        ])

        # Quantize: map continuous weights to nearest NF4 level
        normalized = weight / absmax

        # Find nearest level for each weight
        quantized_indices = torch.zeros_like(weight, dtype=torch.uint8)
        for idx, level in enumerate(nf4_levels):
            mask = (normalized >= level)
            quantized_indices = torch.where(mask, idx, quantized_indices)

        return {
            'indices': quantized_indices,
            'scale': absmax,
            'nf4_levels': nf4_levels
        }

    def dequantize_nf4(self, quantized):
        """Dequantize 4-bit weights back to float."""
        indices = quantized['indices']
        scale = quantized['scale']
        nf4_levels = quantized['nf4_levels']

        # Map indices back to float values
        dequantized = nf4_levels[indices] * scale

        return dequantized

    def forward(self, x):
        # Dequantize base weight for computation
        weight_float = self.dequantize_nf4(self.weight_4bit)

        # Base transformation
        result = F.linear(x, weight_float)

        # Add LoRA in full precision
        lora_update = (x @ self.lora_A.T) @ self.lora_B.T
        result += self.scaling * lora_update

        return result

print("\n\nQLoRA (Quantized LoRA):")
print("  • Base model: 4-bit NF4 quantization")
print("  • LoRA adapters: Full precision (16-bit)")
print("  • Memory: ~1/4 of LoRA, ~1/16 of full fine-tuning")
print("  • Performance: Matches 16-bit LoRA")
print("  • Breakthrough: Fine-tune 65B models on single 48GB GPU")
print("\n  Memory Comparison (65B model):")
print("    Full FP16: ~780 GB")
print("    LoRA FP16: ~400 GB")
print("    QLoRA 4-bit: ~48 GB ✓")
```

### QLoRA Implementation Details

```python
def qlora_memory_breakdown():
    """
    Break down QLoRA memory savings.
    """

    print("\n\nQLoRA Memory Breakdown:\n")
    print("="*80)

    model_params = 65_000_000_000  # 65B parameters

    print("Component                     | Memory (GB)")
    print("-" * 50)

    # Base model storage
    fp16_base = (model_params * 2) / 1e9  # 2 bytes per param
    int4_base = (model_params * 0.5) / 1e9  # 4 bits = 0.5 bytes per param

    print(f"Base Model (FP16)            | {fp16_base:.1f}")
    print(f"Base Model (4-bit NF4)       | {int4_base:.1f}")

    # LoRA adapters (assuming rank=64, applied to 80% of layers)
    lora_params = int(model_params * 0.001)  # ~0.1% of base model
    lora_fp16 = (lora_params * 2) / 1e9

    print(f"LoRA Adapters (FP16)         | {lora_fp16:.1f}")

    # Optimizer states (Adam: 2 states per param)
    optimizer_states = (lora_params * 2 * 4) / 1e9  # 4 bytes per state (FP32)

    print(f"Optimizer States (Adam)      | {optimizer_states:.1f}")

    # Gradients
    gradients = lora_fp16  # Same as LoRA params

    print(f"Gradients                    | {gradients:.1f}")

    # Activations (batch size 1, sequence length 512)
    batch_size = 1
    seq_length = 512
    hidden_size = 8192  # For 65B model
    num_layers = 80

    activations = (batch_size * seq_length * hidden_size * num_layers * 2) / 1e9

    print(f"Activations                  | {activations:.1f}")
    print("-" * 50)

    total = int4_base + lora_fp16 + optimizer_states + gradients + activations

    print(f"TOTAL (QLoRA)                | {total:.1f}")
    print(f"\nFits on single 48GB GPU! ✓")
    print("\nKey Innovations:")
    print("  1. 4-bit base model: 130GB → 32.5GB")
    print("  2. Only train LoRA adapters: ~1% of params")
    print("  3. Paged optimizers: Handle memory spikes")
    print("  4. Double quantization: Further compress quantization constants")

qlora_memory_breakdown()
```

### QLoRA Training Setup

```python
def setup_qlora_training():
    """
    Complete QLoRA training setup.
    """

    print("\n\nQLoRA Training Setup:\n")
    print("="*80)

    config = {
        'base_model': '65B-llama-2',

        'quantization': {
            'method': 'nf4',  # NormalFloat 4-bit
            'double_quant': True,  # Quantize quantization constants
            'compute_dtype': 'bfloat16',  # Compute in bf16
            'quant_type': 'nf4'
        },

        'lora': {
            'rank': 64,
            'alpha': 16,  # Scaling factor
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj',
                              'gate_proj', 'up_proj', 'down_proj'],
            'dropout': 0.05
        },

        'training': {
            'batch_size': 1,
            'gradient_accumulation_steps': 16,  # Effective batch size: 16
            'learning_rate': 2e-4,
            'max_steps': 10000,
            'warmup_steps': 100,
            'optimizer': 'paged_adamw_32bit'  # Handles memory spikes
        },

        'memory_optimization': {
            'gradient_checkpointing': True,  # Recompute activations
            'paged_optimizer': True,  # Move optimizer states to CPU if needed
            'max_memory': {0: '48GB'}  # GPU memory limit
        }
    }

    print("Base Model:")
    print(f"  Model: {config['base_model']}")
    print(f"  Quantization: {config['quantization']['method'].upper()}")
    print(f"  Double Quantization: {config['quantization']['double_quant']}")

    print("\nLoRA Configuration:")
    print(f"  Rank: {config['lora']['rank']}")
    print(f"  Alpha: {config['lora']['alpha']}")
    print(f"  Target Modules: {len(config['lora']['target_modules'])} types")
    print(f"  Dropout: {config['lora']['dropout']}")

    print("\nTraining:")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Gradient Accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"  Effective Batch Size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Optimizer: {config['training']['optimizer']}")

    print("\nMemory Optimization:")
    print(f"  Gradient Checkpointing: {config['memory_optimization']['gradient_checkpointing']}")
    print(f"  Paged Optimizer: {config['memory_optimization']['paged_optimizer']}")
    print(f"  Max GPU Memory: {config['memory_optimization']['max_memory'][0]}")

    return config

setup_qlora_training()
```

### QLoRA vs LoRA vs Full Fine-Tuning

```python
def compare_finetuning_approaches():
    """
    Compare different fine-tuning approaches.
    """

    print("\n\nFine-Tuning Comparison (65B Model):\n")
    print("="*80)

    comparison = {
        'Approach': ['Full FT', 'LoRA', 'QLoRA'],
        'Precision': ['FP16', 'FP16', '4-bit + FP16 adapters'],
        'Memory (GB)': [780, 400, 48],
        'Trainable Params': ['100%', '0.1%', '0.1%'],
        'Performance': ['100%', '99%', '99%'],
        'Training Time': ['1x', '1.2x', '1.5x'],
        'Hardware': ['8x A100', '4x A100', '1x A100'],
        'Cost': ['$$$$$', '$$$', '$']
    }

    print(f"{'Metric':<20} | {'Full FT':<15} | {'LoRA':<15} | {'QLoRA':<15}")
    print("-" * 75)

    for i in range(len(comparison['Approach'])):
        for metric in comparison:
            if metric == 'Approach':
                continue
            approach_idx = i
            value = comparison[metric][approach_idx]

            if i == 0:
                print(f"{metric:<20} | {comparison[metric][0]:<15} | {comparison[metric][1]:<15} | {comparison[metric][2]:<15}")

    print("\n\nKey Takeaways:")
    print("  • QLoRA enables fine-tuning of massive models (65B+) on consumer hardware")
    print("  • Minimal performance loss vs full fine-tuning")
    print("  • 10-16x memory reduction compared to LoRA")
    print("  • Democratizes LLM fine-tuning")
    print("\n  When to use:")
    print("    Full FT: When you have unlimited resources and need absolute best performance")
    print("    LoRA: Good balance of performance and efficiency")
    print("    QLoRA: Limited GPU memory, large models, or cost constraints")

compare_finetuning_approaches()
```

### Practical QLoRA Usage

```python
def practical_qlora_example():
    """
    Practical example of using QLoRA with Hugging Face.
    """

    print("\n\nPractical QLoRA Usage:\n")
    print("="*80)

    print("""
# Install required packages
pip install transformers peft bitsandbytes accelerate

# Python code:
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    load_in_4bit=True,
    device_map="auto",
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4"
    }
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=64,  # Rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Add LoRA adapters
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 65,000,000,000 || trainable%: 0.006%

# Train normally with your favorite trainer
# trainer = Trainer(model=model, ...)
# trainer.train()

# Save only the LoRA adapters (~100MB instead of 130GB!)
model.save_pretrained("./qlora-adapters")

# Later: Load base model + adapters
base_model = AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)
model = PeftModel.from_pretrained(base_model, "./qlora-adapters")
""")

    print("\nKey Benefits:")
    print("  • Adapter files are tiny (~100MB vs 130GB base model)")
    print("  • Can share adapters without sharing base model")
    print("  • Multiple adapters for same base model (different tasks)")
    print("  • Fast switching between adapters")
    print("  • Community can fine-tune and share adaptations")

practical_qlora_example()
```

### BitFit

```python
def bitfit_fine_tuning(model):
    """
    BitFit: Only train bias terms.

    Surprisingly effective!
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only bias terms
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("BitFit:")
    print(f"  • Train: Only bias terms")
    print(f"  • Parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  • Performance: Surprisingly good (80-90% of full fine-tuning)")
    print(f"  • Benefit: Extremely simple and parameter-efficient")

bitfit_fine_tuning(nn.Linear(768, 768))
```

### Comparison

```python
efficiency_comparison = {
    'Method': {
        'Full Fine-tuning': 'Update all parameters',
        'Feature Extraction': 'Freeze model, train head',
        'Adapters': 'Insert trainable bottlenecks',
        'LoRA': 'Low-rank weight updates',
        'QLoRA': '4-bit quantized + LoRA',
        'BitFit': 'Train only bias terms',
        'Prompt Tuning': 'Learn continuous prompts'
    },
    'Trainable %': {
        'Full Fine-tuning': '100%',
        'Feature Extraction': '~1%',
        'Adapters': '~1-3%',
        'LoRA': '~0.1-1%',
        'QLoRA': '~0.1-1%',
        'BitFit': '~0.1%',
        'Prompt Tuning': '~0.01%'
    },
    'Performance': {
        'Full Fine-tuning': '100%',
        'Feature Extraction': '85-90%',
        'Adapters': '95-98%',
        'LoRA': '98-100%',
        'QLoRA': '98-100%',
        'BitFit': '80-90%',
        'Prompt Tuning': '90-100% (at scale)'
    },
    'Memory': {
        'Full Fine-tuning': 'Very High',
        'Feature Extraction': 'Low',
        'Adapters': 'Medium',
        'LoRA': 'Medium',
        'QLoRA': 'Very Low',
        'BitFit': 'Low',
        'Prompt Tuning': 'Low'
    }
}

print("\n\nEfficient Transfer Learning Comparison:\n")
print(f"{'Method':<20} {'Trainable %':<15} {'Performance':<15} {'Memory':<15}")
print("=" * 65)

for aspect in ['Method', 'Trainable %', 'Performance', 'Memory']:
    if aspect == 'Method':
        continue
    for method in efficiency_comparison['Method']:
        if aspect == 'Trainable %':
            print(f"{method:<20} {efficiency_comparison['Trainable %'][method]:<15} {efficiency_comparison['Performance'][method]:<15} {efficiency_comparison['Memory'][method]:<15}")
```

## Pretraining Data

### Data Sources

```python
def pretraining_data_sources():
    """Common pretraining corpora."""

    sources = {
        'Wikipedia': {
            'size': '~3B words (English)',
            'quality': 'High',
            'domain': 'Encyclopedic knowledge',
            'models': 'BERT, RoBERTa, GPT'
        },
        'BooksCorpus': {
            'size': '~800M words',
            'quality': 'High',
            'domain': 'Long-form narratives',
            'models': 'BERT, GPT'
        },
        'Common Crawl': {
            'size': 'Petabytes (filtered to ~100s GB)',
            'quality': 'Variable (requires filtering)',
            'domain': 'Web text (diverse)',
            'models': 'RoBERTa, T5, GPT-3'
        },
        'C4 (Colossal Clean Crawled Corpus)': {
            'size': '~750GB',
            'quality': 'Filtered Common Crawl',
            'domain': 'Clean web text',
            'models': 'T5, FLAN-T5'
        },
        'The Pile': {
            'size': '~800GB',
            'quality': 'High (curated)',
            'domain': 'Diverse (code, books, papers, web)',
            'models': 'GPT-Neo, GPT-J'
        }
    }

    print("Pretraining Data Sources:\n")
    for source, info in sources.items():
        print(f"{source}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()

pretraining_data_sources()
```

### Data Quality vs Quantity

```python
def data_quality_matters():
    """Quality vs quantity trade-off."""

    print("Data Quality vs Quantity:\n")

    print("Findings from research:")
    print("  • More data → Better performance (generally)")
    print("  • Quality matters more than quantity")
    print("  • Deduplication improves performance")
    print("  • Filtering toxic content important")
    print("\nExample (RoBERTa vs BERT):")
    print("  BERT: 16GB (Wikipedia + Books)")
    print("  RoBERTa: 160GB (+ Common Crawl)")
    print("  Result: RoBERTa significantly outperforms BERT")
    print("\nData preprocessing:")
    print("  • Remove duplicates")
    print("  • Filter low-quality (short, garbled)")
    print("  • Language identification")
    print("  • Toxicity filtering")

data_quality_matters()
```

## Scaling Laws

### Model Size vs Performance

```python
def scaling_laws():
    """How performance scales with model size, data, compute."""

    print("\n\nScaling Laws (Kaplan et al., 2020):\n")

    print("Key findings:")
    print("  1. Performance improves as power law with:")
    print("     • Model size (parameters)")
    print("     • Dataset size (tokens)")
    print("     • Compute (FLOPs)")
    print("\n  2. Model size and data should scale together")
    print("     • 10x parameters → ~10x data")
    print("\n  3. Diminishing returns at scale")
    print("     • Double parameters → ~10% improvement")
    print("\n  4. Compute-optimal training (Chinchilla):")
    print("     • Train smaller models longer on more data")
    print("     • More efficient than large model + less data")

    print("\nExample:")
    print("  BERT-base:  110M params, 16GB data")
    print("  BERT-large: 340M params, 16GB data  → +10% performance")
    print("  RoBERTa:    340M params, 160GB data → +15% performance")
    print("  Conclusion: More data helped more than model size")

scaling_laws()
```

### Compute Budget

```python
def compute_budget_allocation():
    """How to allocate compute budget."""

    print("\n\nCompute Budget Allocation:\n")

    scenarios = {
        'Small budget': {
            'strategy': 'Smaller model, more training',
            'example': 'BERT-small (60M params) for 1M steps'
        },
        'Medium budget': {
            'strategy': 'Balanced model and data',
            'example': 'BERT-base (110M params) for 1M steps'
        },
        'Large budget': {
            'strategy': 'Large model + massive data',
            'example': 'GPT-3 (175B params) for 300B tokens'
        }
    }

    for scenario, info in scenarios.items():
        print(f"{scenario.upper()}:")
        print(f"  Strategy: {info['strategy']}")
        print(f"  Example: {info['example']}")
        print()

    print("Chinchilla finding:")
    print("  • Given fixed compute budget")
    print("  • Better to train smaller model on more data")
    print("  • Than large model on less data")
    print("  • Example: Chinchilla (70B) outperforms Gopher (280B)")

compute_budget_allocation()
```

## Summary

**Key Concepts**:

1. **Pretraining-finetuning paradigm**: Pretrain on unlabeled data, fine-tune on labeled data
2. **Pretraining objectives**: MLM (BERT), CLM (GPT), span corruption (T5), denoising (BART)
3. **What models learn**: Syntax, semantics, world knowledge, in different layers
4. **Transfer strategies**: Full fine-tuning, feature extraction, gradual unfreezing, discriminative LR
5. **Domain adaptation**: Continued pretraining on domain-specific data (DAPT, TAPT)
6. **Few-shot learning**: Learn from few examples via prompting (zero-shot, few-shot, prompt tuning)
7. **Efficient transfer**: Adapters, LoRA, BitFit (train <1% of parameters)
8. **Scaling laws**: Performance scales with model size, data, and compute

**Pretraining Objectives**:

| Objective       | Architecture    | Context               | Best For               |
| --------------- | --------------- | --------------------- | ---------------------- |
| MLM             | Encoder-only    | Bidirectional         | Understanding          |
| CLM             | Decoder-only    | Unidirectional        | Generation             |
| Span Corruption | Encoder-decoder | Bidirectional encoder | Seq2seq                |
| Denoising       | Encoder-decoder | Bidirectional encoder | Robust representations |

**Transfer Learning Hierarchy**:

```
Efficiency vs Performance Trade-off:

Full Fine-tuning ─────────────────────────► 100% performance, 100% parameters
Adapters ──────────────────────────────────► 95-98% performance, 1-3% parameters
LoRA ──────────────────────────────────────► 98-100% performance, 0.1-1% parameters
QLoRA ─────────────────────────────────────► 98-100% performance, 0.1-1% params, 1/4 memory
BitFit ────────────────────────────────────► 80-90% performance, 0.1% parameters
Prompt Tuning ─────────────────────────────► 90-100% performance, 0.01% parameters
Feature Extraction ────────────────────────► 85-90% performance, 1% parameters
```

**Memory Efficiency (65B Model)**:

| Method | Memory | Hardware | Performance |
|--------|--------|----------|-------------|
| Full FT | ~780 GB | 8x A100 | 100% |
| LoRA | ~400 GB | 4x A100 | 99% |
| QLoRA | ~48 GB | 1x A100 | 99% |

**Key Insights**:

- Pretraining learns generalizable representations from unlabeled data
- Different layers capture different linguistic phenomena
- Fine-tuning adapts pretrained knowledge to specific tasks
- Quality > quantity for pretraining data
- Efficient methods (LoRA, adapters) achieve similar performance with far fewer parameters
- **QLoRA breakthrough**: 4-bit quantization + LoRA enables 65B+ model fine-tuning on single GPU
- Scaling laws: Performance improves as power law with size, data, compute

**Best Practices**:

1. Start with pretrained model (don't train from scratch)
2. Use domain adaptation for specialized domains
3. Use parameter-efficient methods when possible (LoRA/QLoRA for large models)
4. Match objective to task (encoder for understanding, decoder for generation)
5. Quality data > quantity
6. Small LR and few epochs for fine-tuning
7. Consider QLoRA for models >13B parameters or limited GPU memory

## Next Steps

- Explore [Large Language Models](../llm_concepts/large-language-models.md) and modern scaling
- Study [Fine-tuning](../llm_concepts/fine-tuning.md) techniques in detail
- Learn [Prompt Engineering](../prompt_engineering/prompt-design.md) for few-shot learning
- Understand [Parameter-Efficient Fine-Tuning](../llm_concepts/parameter-efficient-finetuning.md)
- Study [Model Architectures](../llm_concepts/model-architectures.md) in the LLM era
- Apply to [Specific Applications](../application_patterns/) in your domain
