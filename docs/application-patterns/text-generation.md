# Text Generation

## Table of Contents

- [Introduction](#introduction)
- [Controlled Generation](#controlled-generation)
- [Style and Tone Control](#style-and-tone-control)
- [Prompt Templates](#prompt-templates)
- [Sampling Strategies](#sampling-strategies)
- [Avoiding Repetition](#avoiding-repetition)
- [Length Control](#length-control)
- [Creative Applications](#creative-applications)
- [Production Patterns](#production-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Text generation is the task of producing fluent, coherent, and contextually appropriate text. Modern LLMs excel at generation, but controlling output quality, style, and format requires careful engineering.

```
Text Generation Flow:

Input (Prompt) → LLM → Generated Text

Controls:
┌──────────────────────────────────────────────────────┐
│ • Temperature (creativity vs consistency)            │
│ • Top-k / Top-p (sampling strategy)                  │
│ • Max length (output size)                           │
│ • Stop sequences (termination)                       │
│ • Penalties (repetition, frequency)                  │
└──────────────────────────────────────────────────────┘

Example:
Prompt: "Write a haiku about coding:"
Output: "Functions awake,
        Variables dance through loops,
        Code comes alive."
```

**Why Text Generation Matters**:

- **Content Creation**: Articles, emails, marketing copy
- **Creative Writing**: Stories, poetry, scripts
- **Code Generation**: Programming assistance
- **Data Augmentation**: Synthetic training data
- **Personalization**: Tailored messages

This guide covers practical techniques for controlling and improving LLM text generation.

## Controlled Generation

### Understanding Generation Control

```python
def generation_control_overview():
    """Overview of controlled text generation."""
    
    print("Controlled Text Generation:\n")
    print("="*80)
    
    print("""
Generation control means steering the model's output toward desired characteristics.

Key Control Dimensions:
  1. Content: What to generate (topic, facts)
  2. Format: How to structure (JSON, bullet points)
  3. Style: Tone, formality, voice
  4. Length: Target word/character count
  5. Constraints: Must include/exclude certain elements

Control Mechanisms:
┌─────────────────────┬──────────────────────────────────────┐
│ Mechanism           │ Description                          │
├─────────────────────┼──────────────────────────────────────┤
│ Prompt Engineering  │ Instructions in prompt               │
│ Few-Shot Examples   │ Show desired output format           │
│ Sampling Parameters │ Temperature, top-k, top-p            │
│ Post-Processing     │ Filter and modify output             │
│ Constrained Decoding│ Force specific tokens/formats        │
└─────────────────────┴──────────────────────────────────────┘

Example - Controlled Product Description:
Constraints:
  • Length: 50-75 words
  • Tone: Professional
  • Format: 2-3 sentences
  • Must mention: features, benefits, price

Input: "Wireless headphones, $99, noise-cancelling, 20hr battery"
Output: "Experience premium audio with our wireless headphones 
         featuring active noise-cancelling technology. Enjoy up to 
         20 hours of uninterrupted listening on a single charge. 
         At just $99, these headphones deliver exceptional value 
         for audiophiles and casual listeners alike."
""")

generation_control_overview()
```

### Implementing Controlled Generation

```python
class ControlledGenerator:
    """Generate text with control over various attributes."""
    
    def __init__(self):
        pass
    
    def generate_with_constraints(self, prompt, constraints=None):
        """
        Generate text with specified constraints.
        
        Args:
            prompt: Base prompt
            constraints: Dict of constraints
        
        Returns:
            Generated text
        """
        if constraints is None:
            constraints = {}
        
        # Build constrained prompt
        full_prompt = prompt
        
        if 'length' in constraints:
            full_prompt += f"\n\nLength: {constraints['length']} words"
        
        if 'format' in constraints:
            full_prompt += f"\nFormat: {constraints['format']}"
        
        if 'tone' in constraints:
            full_prompt += f"\nTone: {constraints['tone']}"
        
        if 'must_include' in constraints:
            items = ', '.join(constraints['must_include'])
            full_prompt += f"\nMust include: {items}"
        
        if 'must_not_include' in constraints:
            items = ', '.join(constraints['must_not_include'])
            full_prompt += f"\nMust NOT include: {items}"
        
        # Generate
        text = llm.generate(full_prompt, temperature=0.7)
        
        # Validate constraints
        validation = self.validate_constraints(text, constraints)
        
        if not validation['valid']:
            # Retry with stronger instructions
            return self.generate_with_retry(prompt, constraints, validation)
        
        return text
    
    def validate_constraints(self, text, constraints):
        """
        Validate that generated text meets constraints.
        
        Args:
            text: Generated text
            constraints: Constraint dict
        
        Returns:
            Validation result
        """
        violations = []
        
        # Check length
        if 'length' in constraints:
            word_count = len(text.split())
            target = constraints['length']
            
            # Allow 10% margin
            if isinstance(target, int):
                if not (target * 0.9 <= word_count <= target * 1.1):
                    violations.append(f"Length: {word_count} words, expected {target}")
        
        # Check required elements
        if 'must_include' in constraints:
            for item in constraints['must_include']:
                if item.lower() not in text.lower():
                    violations.append(f"Missing required: {item}")
        
        # Check forbidden elements
        if 'must_not_include' in constraints:
            for item in constraints['must_not_include']:
                if item.lower() in text.lower():
                    violations.append(f"Contains forbidden: {item}")
        
        return {
            'valid': len(violations) == 0,
            'violations': violations
        }
    
    def generate_with_retry(self, prompt, constraints, previous_validation, max_retries=3):
        """
        Retry generation if constraints not met.
        
        Args:
            prompt: Base prompt
            constraints: Constraints dict
            previous_validation: Previous validation result
            max_retries: Maximum retry attempts
        
        Returns:
            Generated text
        """
        for attempt in range(max_retries):
            # Add violation feedback to prompt
            feedback = "Previous attempt had issues:\n"
            for v in previous_validation['violations']:
                feedback += f"  - {v}\n"
            
            enhanced_prompt = f"{prompt}\n\n{feedback}\nPlease address these issues:"
            
            text = llm.generate(enhanced_prompt, temperature=0.7)
            validation = self.validate_constraints(text, constraints)
            
            if validation['valid']:
                return text
            
            previous_validation = validation
        
        # Return best attempt
        return text

# Example
print("\n" + "="*80)
print("Controlled Generation Example\n")

generator = ControlledGenerator()

prompt = "Write a product description for wireless headphones."

constraints = {
    'length': 60,
    'tone': 'professional',
    'format': '2-3 sentences',
    'must_include': ['noise-cancelling', 'battery life', 'price'],
    'must_not_include': ['cheap', 'mediocre']
}

print("Prompt:", prompt)
print("\nConstraints:")
for key, value in constraints.items():
    print(f"  {key}: {value}")

print("\nGenerated Text:")
text = generator.generate_with_constraints(prompt, constraints)
print(text)

validation = generator.validate_constraints(text, constraints)
print(f"\nValidation: {'✓ PASS' if validation['valid'] else '✗ FAIL'}")
if validation['violations']:
    for v in validation['violations']:
        print(f"  - {v}")
```

### Format-Specific Generation

```python
class FormatGenerator:
    """Generate text in specific formats."""
    
    def __init__(self):
        pass
    
    def generate_json(self, prompt, schema):
        """
        Generate JSON output with schema validation.
        
        Args:
            prompt: Description of desired data
            schema: JSON schema
        
        Returns:
            Valid JSON string
        """
        schema_str = json.dumps(schema, indent=2)
        
        full_prompt = f"""
Generate JSON data based on this description.

Description: {prompt}

Required schema:
{schema_str}

Output valid JSON only:"""
        
        response = llm.generate(full_prompt, temperature=0.3)
        
        # Extract JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            # Validate
            try:
                data = json.loads(json_str)
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                return self.generate_json(prompt, schema)  # Retry
        
        return "{}"
    
    def generate_markdown(self, topic, structure):
        """
        Generate markdown document with specified structure.
        
        Args:
            topic: Document topic
            structure: Desired structure (headings)
        
        Returns:
            Markdown text
        """
        structure_str = "\n".join([f"  - {s}" for s in structure])
        
        prompt = f"""
Write a markdown document about: {topic}

Structure (required sections):
{structure_str}

Output in markdown format:"""
        
        return llm.generate(prompt, temperature=0.7)
    
    def generate_bullet_points(self, prompt, num_points=5):
        """
        Generate bullet point list.
        
        Args:
            prompt: Topic description
            num_points: Number of points
        
        Returns:
            Bullet points
        """
        full_prompt = f"""
{prompt}

Provide exactly {num_points} bullet points:"""
        
        response = llm.generate(full_prompt, temperature=0.7)
        
        # Parse and validate
        import re
        bullets = re.findall(r'^[•\-\*]\s+(.+)$', response, re.MULTILINE)
        
        if len(bullets) < num_points:
            # Retry
            return self.generate_bullet_points(prompt, num_points)
        
        return bullets[:num_points]
    
    def generate_table(self, prompt, columns):
        """
        Generate data table.
        
        Args:
            prompt: Table description
            columns: List of column names
        
        Returns:
            Markdown table
        """
        columns_str = " | ".join(columns)
        separator = " | ".join(["---"] * len(columns))
        
        full_prompt = f"""
{prompt}

Generate a markdown table with these columns:
{columns_str}

Format:
| {columns_str} |
| {separator} |
| data | data | ... |

Output:"""
        
        return llm.generate(full_prompt, temperature=0.7)

# Example
print("\n\n" + "="*80)
print("Format-Specific Generation Example\n")

format_gen = FormatGenerator()

# JSON generation
print("1. JSON Generation:")
schema = {
    "name": "string",
    "age": "number",
    "email": "string"
}
json_output = format_gen.generate_json("A software engineer named Alice", schema)
print(json_output)

# Bullet points
print("\n\n2. Bullet Points:")
bullets = format_gen.generate_bullet_points(
    "Key benefits of regular exercise", 
    num_points=3
)
for bullet in bullets:
    print(f"  • {bullet}")
```

## Style and Tone Control

### Adjusting Style and Tone

```python
class StyleController:
    """Control style and tone of generated text."""
    
    def __init__(self):
        self.tones = {
            'professional': 'formal, business-appropriate',
            'casual': 'conversational, friendly',
            'academic': 'scholarly, technical',
            'enthusiastic': 'energetic, excited',
            'empathetic': 'understanding, compassionate'
        }
    
    def generate_with_tone(self, content, tone):
        """
        Generate text with specific tone.
        
        Args:
            content: Content to generate
            tone: Desired tone
        
        Returns:
            Styled text
        """
        tone_desc = self.tones.get(tone, tone)
        
        prompt = f"""
Write about: {content}

Tone: {tone_desc}

Text:"""
        
        return llm.generate(prompt, temperature=0.7)
    
    def rewrite_with_tone(self, text, target_tone):
        """
        Rewrite existing text in different tone.
        
        Args:
            text: Original text
            target_tone: Desired tone
        
        Returns:
            Rewritten text
        """
        tone_desc = self.tones.get(target_tone, target_tone)
        
        prompt = f"""
Rewrite this text with a {tone_desc} tone:

Original: {text}

Rewritten ({target_tone}):"""
        
        return llm.generate(prompt, temperature=0.7)
    
    def generate_multi_tone(self, content, tones):
        """
        Generate same content in multiple tones.
        
        Args:
            content: Content description
            tones: List of tones
        
        Returns:
            Dict mapping tone to text
        """
        results = {}
        
        for tone in tones:
            results[tone] = self.generate_with_tone(content, tone)
        
        return results
    
    def adjust_formality(self, text, formality_level):
        """
        Adjust formality level.
        
        Args:
            text: Original text
            formality_level: 1-5 (1=very casual, 5=very formal)
        
        Returns:
            Adjusted text
        """
        levels = {
            1: "very casual and colloquial",
            2: "casual and friendly",
            3: "neutral",
            4: "formal and professional",
            5: "very formal and ceremonial"
        }
        
        level_desc = levels.get(formality_level, "neutral")
        
        prompt = f"""
Rewrite this text with a {level_desc} level of formality:

Original: {text}

Rewritten:"""
        
        return llm.generate(prompt, temperature=0.7)

# Example
print("\n\n" + "="*80)
print("Style and Tone Control Example\n")

controller = StyleController()

content = "Our new product launch was successful."

tones = ['professional', 'casual', 'enthusiastic']

print(f"Original: {content}\n")
print("Same content in different tones:\n")

results = controller.generate_multi_tone(content, tones)

for tone, text in results.items():
    print(f"{tone.upper()}:")
    print(f"  {text}\n")

# Formality adjustment
print("\nFormality Levels:")
original = "Hey, check out this cool new feature!"

for level in [1, 3, 5]:
    adjusted = controller.adjust_formality(original, level)
    print(f"  Level {level}: {adjusted}")
```

### Persona-Based Generation

```python
class PersonaGenerator:
    """Generate text from specific persona perspectives."""
    
    def __init__(self):
        self.personas = {
            'expert': {
                'description': 'A knowledgeable expert in the field',
                'traits': ['authoritative', 'detailed', 'technical']
            },
            'beginner': {
                'description': 'Someone new to the topic',
                'traits': ['curious', 'simple language', 'asking questions']
            },
            'skeptic': {
                'description': 'A critical thinker who questions claims',
                'traits': ['questioning', 'analytical', 'evidence-focused']
            },
            'enthusiast': {
                'description': 'An excited advocate',
                'traits': ['passionate', 'positive', 'engaging']
            }
        }
    
    def generate_as_persona(self, prompt, persona):
        """
        Generate text from persona perspective.
        
        Args:
            prompt: Content to generate
            persona: Persona name
        
        Returns:
            Persona-styled text
        """
        persona_info = self.personas.get(persona, {})
        description = persona_info.get('description', persona)
        traits = persona_info.get('traits', [])
        
        traits_str = ', '.join(traits)
        
        full_prompt = f"""
You are {description}.
Your writing style is: {traits_str}

Topic: {prompt}

Response:"""
        
        return llm.generate(full_prompt, temperature=0.8)
    
    def generate_dialogue(self, topic, personas):
        """
        Generate multi-persona dialogue.
        
        Args:
            topic: Discussion topic
            personas: List of persona names
        
        Returns:
            Dialogue script
        """
        personas_str = ", ".join(personas)
        
        prompt = f"""
Generate a dialogue about: {topic}

Participants: {personas_str}

Format:
[Persona]: statement/question

Dialogue:"""
        
        return llm.generate(prompt, temperature=0.8)

# Example
print("\n\n" + "="*80)
print("Persona-Based Generation Example\n")

persona_gen = PersonaGenerator()

topic = "The benefits of learning Python"

personas = ['expert', 'beginner', 'enthusiast']

print(f"Topic: {topic}\n")

for persona in personas:
    text = persona_gen.generate_as_persona(topic, persona)
    print(f"{persona.upper()}:")
    print(f"  {text}\n")
```

## Prompt Templates

### Reusable Prompt Templates

```python
class PromptTemplate:
    """Reusable templates for text generation."""
    
    def __init__(self):
        self.templates = {}
    
    def create_template(self, name, template_str, required_vars):
        """
        Create a new template.
        
        Args:
            name: Template name
            template_str: Template string with {variables}
            required_vars: List of required variable names
        """
        self.templates[name] = {
            'template': template_str,
            'required_vars': required_vars
        }
    
    def generate_from_template(self, name, variables):
        """
        Generate text from template.
        
        Args:
            name: Template name
            variables: Dict of variable values
        
        Returns:
            Generated text
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        template_info = self.templates[name]
        template_str = template_info['template']
        required = template_info['required_vars']
        
        # Check required variables
        missing = [v for v in required if v not in variables]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Fill template
        prompt = template_str.format(**variables)
        
        # Generate
        return llm.generate(prompt, temperature=0.7)
    
    def list_templates(self):
        """List all available templates."""
        return list(self.templates.keys())

# Example templates
print("\n\n" + "="*80)
print("Prompt Templates Example\n")

pt = PromptTemplate()

# Email template
pt.create_template(
    'professional_email',
    """Write a professional email.

To: {recipient}
Subject: {subject}
Context: {context}
Tone: {tone}

Email:""",
    ['recipient', 'subject', 'context', 'tone']
)

# Product description template
pt.create_template(
    'product_description',
    """Write a compelling product description.

Product: {product_name}
Key Features: {features}
Target Audience: {audience}
Price: {price}
Tone: {tone}

Description (50-100 words):""",
    ['product_name', 'features', 'audience', 'price', 'tone']
)

# Blog post intro template
pt.create_template(
    'blog_intro',
    """Write an engaging blog post introduction.

Topic: {topic}
Hook: {hook}
Key Points: {key_points}
Target Length: 2-3 paragraphs

Introduction:""",
    ['topic', 'hook', 'key_points']
)

print("Available Templates:")
for template in pt.list_templates():
    print(f"  - {template}")

print("\n\nExample: Generate Product Description\n")

variables = {
    'product_name': 'SmartWatch Pro',
    'features': 'heart rate monitoring, GPS, 7-day battery',
    'audience': 'fitness enthusiasts',
    'price': '$299',
    'tone': 'enthusiastic'
}

description = pt.generate_from_template('product_description', variables)
print(description)
```

### Template Chaining

```python
class TemplateChain:
    """Chain multiple templates together."""
    
    def __init__(self):
        self.pt = PromptTemplate()
    
    def generate_chained(self, template_chain, initial_variables):
        """
        Generate text through a chain of templates.
        
        Args:
            template_chain: List of (template_name, output_var) tuples
            initial_variables: Initial variable dict
        
        Returns:
            Final generated text
        """
        variables = initial_variables.copy()
        outputs = []
        
        for template_name, output_var in template_chain:
            # Generate from current template
            output = self.pt.generate_from_template(template_name, variables)
            outputs.append(output)
            
            # Store output for next template
            variables[output_var] = output
        
        return {
            'final_output': outputs[-1],
            'all_outputs': outputs,
            'variables': variables
        }

# Example
print("\n\n" + "="*80)
print("Template Chaining Example\n")

# Setup templates for chaining
pt.create_template(
    'outline',
    """Create an outline for: {topic}
Number of points: {num_points}

Outline:""",
    ['topic', 'num_points']
)

pt.create_template(
    'expand_outline',
    """Expand this outline into a full article.

Outline:
{outline}

Target length: {target_length} words

Article:""",
    ['outline', 'target_length']
)

chain = TemplateChain()

# Chain: outline → expand
template_chain = [
    ('outline', 'outline'),
    ('expand_outline', 'article')
]

initial_vars = {
    'topic': 'Benefits of meditation',
    'num_points': '3',
    'target_length': '200'
}

result = chain.generate_chained(template_chain, initial_vars)

print("Final Article:")
print(result['final_output'])
```

## Sampling Strategies

### Understanding Sampling Parameters

```python
def sampling_strategies_overview():
    """Overview of sampling strategies for generation."""
    
    print("\n\nSampling Strategies:\n")
    print("="*80)
    
    print("""
Sampling controls how the model selects the next token during generation.

Key Parameters:
┌──────────────┬───────────────────────────────────────────────────────┐
│ Parameter    │ Description                                           │
├──────────────┼───────────────────────────────────────────────────────┤
│ Temperature  │ Randomness (0=deterministic, higher=more random)      │
│ Top-k        │ Sample from k most likely tokens                      │
│ Top-p        │ Sample from tokens with cumulative prob > p           │
│ Frequency    │ Penalize tokens based on how often they appear        │
│ Presence     │ Penalize tokens that have appeared at all             │
└──────────────┴───────────────────────────────────────────────────────┘

TEMPERATURE:
  Temperature = 0:   "The cat sat on the mat."        (deterministic)
  Temperature = 0.5: "The cat rested on the mat."     (slight variation)
  Temperature = 1.0: "The feline lounged on the rug." (more creative)
  Temperature = 2.0: "A whisker beast throne-sat!"    (very random)

  ┌────────────────┬──────────────────────────────────────┐
  │ Temperature    │ Use Case                             │
  ├────────────────┼──────────────────────────────────────┤
  │ 0 - 0.3        │ Factual tasks, classification        │
  │ 0.4 - 0.7      │ General purpose, balanced            │
  │ 0.8 - 1.2      │ Creative writing, brainstorming      │
  │ 1.3+           │ Experimental, highly creative        │
  └────────────────┴──────────────────────────────────────┘

TOP-K SAMPLING:
  Limit to k most likely tokens
  
  Example (k=3):
  Probabilities: [0.4, 0.3, 0.2, 0.05, 0.03, 0.02]
  Sample only from: [0.4, 0.3, 0.2] (top 3)
  
  Higher k → More diversity
  Lower k → More focused

TOP-P (NUCLEUS) SAMPLING:
  Sample from smallest set of tokens with cumulative prob ≥ p
  
  Example (p=0.9):
  Probabilities: [0.4, 0.3, 0.2, 0.05, 0.03, 0.02]
  Cumulative:    [0.4, 0.7, 0.9, 0.95, 0.98, 1.0]
  Sample from: [0.4, 0.3, 0.2] (adds up to 0.9)
  
  Typical values: 0.9 - 0.95

COMBINING STRATEGIES:
  Temperature + Top-p is most common:
    • Temperature = 0.7, Top-p = 0.9 (balanced)
    • Temperature = 0.3, Top-p = 0.95 (focused)
    • Temperature = 1.0, Top-p = 0.85 (creative)

REPETITION PENALTIES:
  Frequency penalty: -0.5 to 0.5
    • Reduces likelihood of repeated tokens
  
  Presence penalty: 0 to 1
    • Binary penalty for any repetition
""")

sampling_strategies_overview()
```

### Implementing Sampling Control

```python
class SamplingController:
    """Control sampling parameters for generation."""
    
    def __init__(self):
        self.presets = {
            'factual': {
                'temperature': 0.2,
                'top_p': 0.95,
                'frequency_penalty': 0,
                'presence_penalty': 0
            },
            'balanced': {
                'temperature': 0.7,
                'top_p': 0.9,
                'frequency_penalty': 0.1,
                'presence_penalty': 0
            },
            'creative': {
                'temperature': 1.0,
                'top_p': 0.85,
                'frequency_penalty': 0.2,
                'presence_penalty': 0.1
            },
            'diverse': {
                'temperature': 1.2,
                'top_p': 0.8,
                'frequency_penalty': 0.5,
                'presence_penalty': 0.3
            }
        }
    
    def generate_with_preset(self, prompt, preset='balanced'):
        """
        Generate with predefined sampling preset.
        
        Args:
            prompt: Input prompt
            preset: Preset name
        
        Returns:
            Generated text
        """
        if preset not in self.presets:
            raise ValueError(f"Unknown preset: {preset}")
        
        params = self.presets[preset]
        
        return llm.generate(
            prompt,
            temperature=params['temperature'],
            top_p=params['top_p'],
            frequency_penalty=params.get('frequency_penalty', 0),
            presence_penalty=params.get('presence_penalty', 0)
        )
    
    def generate_with_custom_params(self, prompt, temperature=0.7, top_p=0.9,
                                   frequency_penalty=0, presence_penalty=0):
        """
        Generate with custom sampling parameters.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
        
        Returns:
            Generated text
        """
        return llm.generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
    
    def compare_presets(self, prompt, presets=None):
        """
        Generate with multiple presets for comparison.
        
        Args:
            prompt: Input prompt
            presets: List of preset names (default: all)
        
        Returns:
            Dict mapping preset to output
        """
        if presets is None:
            presets = list(self.presets.keys())
        
        results = {}
        
        for preset in presets:
            results[preset] = self.generate_with_preset(prompt, preset)
        
        return results

# Example
print("\n\n" + "="*80)
print("Sampling Control Example\n")

controller = SamplingController()

prompt = "Write a short story opening about a mysterious door:"

print(f"Prompt: {prompt}\n")
print("Comparing different sampling presets:\n")

presets = ['factual', 'balanced', 'creative']
results = controller.compare_presets(prompt, presets)

for preset, text in results.items():
    print(f"{preset.upper()}:")
    print(f"  {text}\n")
```

## Avoiding Repetition

### Detecting and Preventing Repetition

```python
class RepetitionAvoider:
    """Avoid repetitive text in generation."""
    
    def __init__(self):
        pass
    
    def detect_repetition(self, text, window_size=5):
        """
        Detect repetitive patterns in text.
        
        Args:
            text: Generated text
            window_size: Size of window to check for repetition
        
        Returns:
            Repetition metrics
        """
        words = text.split()
        
        # Check for repeated n-grams
        ngrams = []
        for i in range(len(words) - window_size + 1):
            ngram = tuple(words[i:i + window_size])
            ngrams.append(ngram)
        
        # Count repetitions
        from collections import Counter
        counts = Counter(ngrams)
        repeated = {k: v for k, v in counts.items() if v > 1}
        
        return {
            'has_repetition': len(repeated) > 0,
            'repeated_ngrams': repeated,
            'repetition_ratio': len(repeated) / len(ngrams) if ngrams else 0
        }
    
    def generate_without_repetition(self, prompt, max_attempts=3):
        """
        Generate text with repetition checking.
        
        Args:
            prompt: Input prompt
            max_attempts: Maximum generation attempts
        
        Returns:
            Non-repetitive text
        """
        for attempt in range(max_attempts):
            # Generate with increasing frequency penalty
            frequency_penalty = 0.2 + (attempt * 0.2)
            
            text = llm.generate(
                prompt,
                temperature=0.7,
                frequency_penalty=frequency_penalty
            )
            
            # Check for repetition
            rep_check = self.detect_repetition(text)
            
            if not rep_check['has_repetition'] or rep_check['repetition_ratio'] < 0.1:
                return text
        
        # Return best attempt
        return text
    
    def post_process_repetition(self, text):
        """
        Remove repetitive sections from text.
        
        Args:
            text: Text with potential repetition
        
        Returns:
            Cleaned text
        """
        words = text.split()
        cleaned = []
        
        i = 0
        while i < len(words):
            # Check for immediate repetition
            if i < len(words) - 1 and words[i] == words[i + 1]:
                cleaned.append(words[i])
                # Skip repeated word
                i += 2
                while i < len(words) and words[i] == words[i - 1]:
                    i += 1
            else:
                cleaned.append(words[i])
                i += 1
        
        return ' '.join(cleaned)

# Example
print("\n\n" + "="*80)
print("Repetition Avoidance Example\n")

avoider = RepetitionAvoider()

# Simulate repetitive text
repetitive_text = "The cat sat on the mat. The cat sat on the mat. The dog barked loudly."

print("Repetitive Text:")
print(repetitive_text)

rep_check = avoider.detect_repetition(repetitive_text)
print(f"\nRepetition detected: {rep_check['has_repetition']}")
print(f"Repetition ratio: {rep_check['repetition_ratio']:.2f}")

if rep_check['repeated_ngrams']:
    print("Repeated patterns:")
    for ngram, count in list(rep_check['repeated_ngrams'].items())[:3]:
        print(f"  '{' '.join(ngram)}' (x{count})")

# Clean text
cleaned = avoider.post_process_repetition(repetitive_text)
print(f"\nCleaned text:\n{cleaned}")
```

## Length Control

### Controlling Output Length

```python
class LengthController:
    """Control the length of generated text."""
    
    def __init__(self):
        pass
    
    def generate_exact_length(self, prompt, target_words, tolerance=10):
        """
        Generate text with specific word count.
        
        Args:
            prompt: Input prompt
            target_words: Target word count
            tolerance: Acceptable deviation
        
        Returns:
            Generated text
        """
        full_prompt = f"""{prompt}

Length requirement: EXACTLY {target_words} words (± {tolerance} words)

Response:"""
        
        for attempt in range(3):
            text = llm.generate(full_prompt, temperature=0.7)
            
            word_count = len(text.split())
            
            if target_words - tolerance <= word_count <= target_words + tolerance:
                return text
            
            # Adjust prompt with feedback
            diff = word_count - target_words
            if diff > 0:
                full_prompt += f"\n(Previous was {word_count} words, too long by {diff})"
            else:
                full_prompt += f"\n(Previous was {word_count} words, too short by {-diff})"
        
        return text
    
    def generate_with_max_length(self, prompt, max_words):
        """
        Generate text not exceeding max length.
        
        Args:
            prompt: Input prompt
            max_words: Maximum word count
        
        Returns:
            Generated text
        """
        full_prompt = f"""{prompt}

Keep response under {max_words} words.

Response:"""
        
        text = llm.generate(full_prompt, temperature=0.7, max_tokens=max_words * 2)
        
        # Truncate if needed
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
            text = ' '.join(words)
            
            # Add ellipsis if truncated mid-sentence
            if not text[-1] in '.!?':
                text += '...'
        
        return text
    
    def generate_progressive_lengths(self, prompt, lengths):
        """
        Generate same content in different lengths.
        
        Args:
            prompt: Input prompt
            lengths: List of word counts
        
        Returns:
            Dict mapping length to text
        """
        results = {}
        
        for length in lengths:
            text = self.generate_exact_length(prompt, length, tolerance=5)
            results[length] = text
        
        return results

# Example
print("\n\n" + "="*80)
print("Length Control Example\n")

length_ctrl = LengthController()

prompt = "Explain what machine learning is."

lengths = [50, 100, 200]

print(f"Prompt: {prompt}\n")
print("Generating at different lengths:\n")

results = length_ctrl.generate_progressive_lengths(prompt, lengths)

for length, text in results.items():
    actual_length = len(text.split())
    print(f"{length} WORDS (actual: {actual_length}):")
    print(f"  {text}\n")
```

## Creative Applications

### Creative Writing

```python
class CreativeWriter:
    """Generate creative content."""
    
    def __init__(self):
        pass
    
    def write_story(self, premise, style='narrative', length_words=200):
        """
        Write a short story.
        
        Args:
            premise: Story premise/prompt
            style: Writing style
            length_words: Target length
        
        Returns:
            Story text
        """
        prompt = f"""Write a {style} story based on this premise:

{premise}

Style: {style}
Length: approximately {length_words} words

Story:"""
        
        return llm.generate(prompt, temperature=0.9)
    
    def write_poetry(self, topic, form='free verse', stanzas=3):
        """
        Write a poem.
        
        Args:
            topic: Poem topic
            form: Poetry form
            stanzas: Number of stanzas
        
        Returns:
            Poem text
        """
        prompt = f"""Write a {form} poem about: {topic}

Structure: {stanzas} stanzas
Form: {form}

Poem:"""
        
        return llm.generate(prompt, temperature=0.9)
    
    def brainstorm_ideas(self, topic, num_ideas=5):
        """
        Generate creative ideas.
        
        Args:
            topic: Topic to brainstorm
            num_ideas: Number of ideas
        
        Returns:
            List of ideas
        """
        prompt = f"""Brainstorm {num_ideas} creative ideas for: {topic}

Ideas:"""
        
        response = llm.generate(prompt, temperature=1.0)
        
        # Parse ideas
        import re
        ideas = re.findall(r'^\d+\.\s+(.+)$', response, re.MULTILINE)
        
        return ideas
    
    def write_dialogue(self, scenario, characters, num_exchanges=5):
        """
        Write character dialogue.
        
        Args:
            scenario: Scene setup
            characters: List of character names
            num_exchanges: Number of dialogue exchanges
        
        Returns:
            Dialogue script
        """
        characters_str = ", ".join(characters)
        
        prompt = f"""Write a dialogue for this scenario:

Scenario: {scenario}
Characters: {characters_str}
Exchanges: {num_exchanges}

Format:
CHARACTER: dialogue

Dialogue:"""
        
        return llm.generate(prompt, temperature=0.8)

# Example
print("\n\n" + "="*80)
print("Creative Writing Example\n")

writer = CreativeWriter()

# Story
print("1. Short Story:")
story = writer.write_story(
    "A programmer discovers their code is sentient",
    style="sci-fi",
    length_words=150
)
print(story)

# Poetry
print("\n\n2. Poem:")
poem = writer.write_poetry("the beauty of algorithms", form="haiku", stanzas=3)
print(poem)

# Brainstorming
print("\n\n3. Idea Brainstorming:")
ideas = writer.brainstorm_ideas("innovative uses for AI in education", num_ideas=3)
for i, idea in enumerate(ideas, 1):
    print(f"  {i}. {idea}")
```

## Production Patterns

### Production Generation System

```python
class ProductionGenerator:
    """Production-ready text generation system."""
    
    def __init__(self):
        self.cache = {}
        self.generation_log = []
        self.controller = ControlledGenerator()
        self.avoider = RepetitionAvoider()
    
    def generate_production(self, prompt, config=None):
        """
        Full production generation pipeline.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
        
        Returns:
            Generation result
        """
        import time
        import hashlib
        
        start_time = time.time()
        
        if config is None:
            config = {}
        
        # Check cache
        cache_key = hashlib.md5(f"{prompt}_{config}".encode()).hexdigest()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cached['metadata']['cache_hit'] = True
            return cached
        
        result = {
            'prompt': prompt,
            'metadata': {
                'timestamp': time.time(),
                'cache_hit': False
            }
        }
        
        try:
            # Generate
            text = self.controller.generate_with_constraints(
                prompt,
                config.get('constraints', {})
            )
            
            # Check for repetition
            if config.get('avoid_repetition', True):
                rep_check = self.avoider.detect_repetition(text)
                
                if rep_check['has_repetition'] and rep_check['repetition_ratio'] > 0.2:
                    # Regenerate
                    text = self.avoider.generate_without_repetition(prompt)
            
            # Post-process
            if config.get('post_process', True):
                text = self.post_process(text, config)
            
            result['text'] = text
            result['success'] = True
            
        except Exception as e:
            result['text'] = ""
            result['success'] = False
            result['error'] = str(e)
        
        # Finalize
        result['metadata']['latency_ms'] = int((time.time() - start_time) * 1000)
        
        # Cache
        if result['success']:
            self.cache[cache_key] = result
        
        # Log
        self.generation_log.append({
            'prompt': prompt[:50],
            'timestamp': time.time(),
            'success': result['success'],
            'latency_ms': result['metadata']['latency_ms']
        })
        
        return result
    
    def post_process(self, text, config):
        """
        Post-process generated text.
        
        Args:
            text: Generated text
            config: Configuration
        
        Returns:
            Processed text
        """
        # Trim whitespace
        text = text.strip()
        
        # Remove markdown code fences if unwanted
        if not config.get('allow_code_blocks', True):
            text = text.replace('```', '')
        
        # Ensure ends with punctuation
        if config.get('ensure_punctuation', False):
            if text and text[-1] not in '.!?':
                text += '.'
        
        return text
    
    def batch_generate(self, prompts, config=None):
        """
        Generate multiple texts in batch.
        
        Args:
            prompts: List of prompts
            config: Shared configuration
        
        Returns:
            List of results
        """
        results = []
        
        for prompt in prompts:
            result = self.generate_production(prompt, config)
            results.append(result)
        
        return results
    
    def get_stats(self):
        """Get generation statistics."""
        
        if not self.generation_log:
            return {"message": "No generations yet"}
        
        total = len(self.generation_log)
        successful = sum(1 for g in self.generation_log if g['success'])
        avg_latency = sum(g['latency_ms'] for g in self.generation_log) / total
        
        return {
            'total_generations': total,
            'success_rate': successful / total,
            'average_latency_ms': avg_latency,
            'cache_size': len(self.cache)
        }

# Example
print("\n\n" + "="*80)
print("Production Generation Example\n")

prod_gen = ProductionGenerator()

prompts = [
    "Write a tagline for a fitness app",
    "Explain blockchain in simple terms",
    "Write a tagline for a fitness app"  # Duplicate to test caching
]

config = {
    'constraints': {
        'length': 20,
        'tone': 'professional'
    },
    'avoid_repetition': True,
    'post_process': True
}

print("Generating texts:\n")

for i, prompt in enumerate(prompts, 1):
    result = prod_gen.generate_production(prompt, config)
    
    print(f"{i}. {prompt}")
    print(f"   Result: {result['text']}")
    print(f"   Cache hit: {result['metadata']['cache_hit']}")
    print(f"   Latency: {result['metadata']['latency_ms']}ms\n")

print("Statistics:")
stats = prod_gen.get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")
```

## Summary

**Text Generation Overview**:

```
Prompt → Sampling → Generation → Post-Processing → Output

Key Controls:
  • Temperature (randomness)
  • Top-k / Top-p (sampling)
  • Penalties (repetition)
  • Length constraints
  • Style/tone directives
```

**Core Components**:

1. **Controlled Generation**: Constraints, validation, retry logic
2. **Style Control**: Tone, formality, persona-based
3. **Prompt Templates**: Reusable patterns, chaining
4. **Sampling**: Temperature, top-k, top-p, penalties
5. **Quality Checks**: Repetition detection, length validation

**Sampling Parameters**:

| Parameter | Range | Use Case |
|-----------|-------|----------|
| Temperature | 0-0.3 | Factual, deterministic |
|  | 0.4-0.7 | General purpose |
|  | 0.8-1.2 | Creative writing |
| Top-p | 0.9-0.95 | Standard sampling |
| Frequency Penalty | 0-0.5 | Reduce repetition |

**Best Practices**:

1. **Control Quality**:
   - Use constraints
   - Validate output
   - Retry on failure
   - Post-process results

2. **Sampling Strategy**:
   - Low temp for facts
   - High temp for creativity
   - Combine temp + top-p
   - Use penalties for diversity

3. **Template Management**:
   - Create reusable templates
   - Validate variables
   - Chain templates
   - Version templates

4. **Avoid Repetition**:
   - Detect n-gram repetition
   - Use frequency penalties
   - Post-process cleanup
   - Regenerate if needed

**Production Considerations**:
- **Caching**: Hash prompts, cache results
- **Validation**: Check constraints, format, length
- **Monitoring**: Track latency, success rate
- **Error Handling**: Retry logic, fallback strategies
- **Batch Processing**: Process multiple prompts efficiently

**Common Pitfalls**:
- High temperature → Incoherent text
- No repetition control → Repetitive output
- Vague constraints → Inconsistent results
- No validation → Format violations
- No caching → Slow and expensive

**Creative Applications**:

| Application | Temperature | Top-p | Notes |
|-------------|-------------|-------|-------|
| Stories | 0.9-1.1 | 0.85 | High creativity |
| Poetry | 1.0-1.2 | 0.8 | Very creative |
| Dialogue | 0.8-0.9 | 0.9 | Natural variation |
| Brainstorming | 1.0+ | 0.8 | Diverse ideas |

**Key Takeaways**:
- Control generation with constraints and sampling
- Use templates for consistency
- Adjust temperature based on task (factual vs creative)
- Always validate and post-process output
- Cache repeated prompts
- Monitor quality metrics
- Implement retry logic for critical generations

## Next Steps

- Use in [Conversational AI](conversational-ai.md) for dialogue generation
- Apply to [Text Classification](text-classification.md) for label generation
- Combine with [Prompt Engineering](../prompt-engineering/) for better control
- Implement [Evaluation Methods](../evaluation/) for generation quality
- Explore [Few-Shot Learning](../few-shot-learning/) for style transfer
- Study [Fine-Tuning](../fine-tuning/) for domain-specific generation
