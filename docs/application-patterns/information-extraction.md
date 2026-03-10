# Information Extraction

## Table of Contents

- [Introduction](#introduction)
- [Named Entity Recognition](#named-entity-recognition)
- [Relation Extraction](#relation-extraction)
- [Event Extraction](#event-extraction)
- [Structured Extraction with LLMs](#structured-extraction-with-llms)
- [Few-Shot Extraction](#few-shot-extraction)
- [Validation and Error Handling](#validation-and-error-handling)
- [Domain Adaptation](#domain-adaptation)
- [Production Patterns](#production-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Information extraction is the task of converting unstructured text into structured, queryable data. It identifies entities, relationships, events, and other key information, making text machine-readable and enabling downstream analysis.

```
Information Extraction Pipeline:

Unstructured Text → IE System → Structured Data

Example:
Input: "Apple Inc. CEO Tim Cook announced iPhone 15 on September 12, 2023"

Output (Structured):
{
  "entities": [
    {"text": "Apple Inc.", "type": "ORGANIZATION"},
    {"text": "Tim Cook", "type": "PERSON"},
    {"text": "iPhone 15", "type": "PRODUCT"},
    {"text": "September 12, 2023", "type": "DATE"}
  ],
  "relations": [
    {"subject": "Tim Cook", "relation": "CEO_OF", "object": "Apple Inc."},
    {"subject": "Tim Cook", "relation": "ANNOUNCED", "object": "iPhone 15"}
  ],
  "event": {
    "type": "PRODUCT_ANNOUNCEMENT",
    "actor": "Tim Cook",
    "object": "iPhone 15",
    "date": "September 12, 2023"
  }
}
```

**Why information extraction matters**:

- **Database Population**: Extract facts to fill databases
- **Knowledge Graphs**: Build structured knowledge representations
- **Search & Analytics**: Enable structured queries on unstructured data
- **Automation**: Automatically process forms, documents, reports
- **Integration**: Feed structured data into downstream systems

This guide covers practical patterns for building robust IE systems with LLMs.

## Named Entity Recognition

### Understanding NER

```python
def ner_overview():
    """Overview of Named Entity Recognition."""

    print("Named Entity Recognition (NER):\n")
    print("="*80)

    print("""
NER identifies and classifies named entities in text.

Common Entity Types:
┌─────────────────┬──────────────────────────────────────────────┐
│ Entity Type     │ Examples                                     │
├─────────────────┼──────────────────────────────────────────────┤
│ PERSON          │ "John Smith", "Dr. Sarah Johnson"            │
│ ORGANIZATION    │ "Apple Inc.", "Harvard University"           │
│ LOCATION        │ "New York", "Mount Everest"                  │
│ DATE            │ "January 1, 2023", "last Tuesday"            │
│ TIME            │ "3:00 PM", "midnight"                        │
│ MONEY           │ "$100", "€50 million"                        │
│ PERCENT         │ "25%", "three percent"                       │
│ PRODUCT         │ "iPhone 15", "Boeing 747"                    │
│ EVENT           │ "World War II", "Super Bowl"                 │
│ LAW             │ "GDPR", "First Amendment"                    │
│ LANGUAGE        │ "English", "Mandarin"                        │
│ QUANTITY        │ "10 kilograms", "5 feet"                     │
└─────────────────┴──────────────────────────────────────────────┘

Example:
Input: "Google acquired DeepMind for $500 million in January 2014."

NER Output:
- "Google" → ORGANIZATION
- "DeepMind" → ORGANIZATION
- "$500 million" → MONEY
- "January 2014" → DATE

Applications:
• Information retrieval (find documents mentioning specific entities)
• Question answering (extract answers from text)
• Content analysis (track mentions of companies, people)
• Document indexing (organize by entities)
• Anonymization (redact sensitive entities)
""")

ner_overview()
```

### Implementing NER with LLMs

```python
class NERExtractor:
    """Extract named entities using LLMs."""

    def __init__(self, entity_types=None):
        """
        Initialize NER extractor.

        Args:
            entity_types: List of entity types to extract
        """
        if entity_types is None:
            self.entity_types = [
                "PERSON", "ORGANIZATION", "LOCATION",
                "DATE", "MONEY", "PRODUCT"
            ]
        else:
            self.entity_types = entity_types

    def extract_entities(self, text):
        """
        Extract named entities from text.

        Args:
            text: Input text

        Returns:
            List of entity dictionaries
        """
        entity_types_str = ", ".join(self.entity_types)

        prompt = f"""
Extract all named entities from this text. Classify each entity into one of these types: {entity_types_str}

Text: {text}

Output format (one per line):
<entity> | <type>

Entities:"""

        response = llm.generate(prompt, temperature=0)

        # Parse response
        entities = []
        for line in response.strip().split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 2:
                    entity_text = parts[0].strip()
                    entity_type = parts[1].strip().upper()

                    if entity_type in self.entity_types:
                        entities.append({
                            'text': entity_text,
                            'type': entity_type
                        })

        return entities

    def extract_entities_json(self, text):
        """
        Extract entities with JSON output.

        Args:
            text: Input text

        Returns:
            Structured entity data
        """
        entity_types_str = ", ".join(self.entity_types)

        prompt = f"""
Extract all named entities from this text and output as JSON.

Entity types: {entity_types_str}

Text: {text}

Output format:
{{
  "entities": [
    {{"text": "entity text", "type": "TYPE"}},
    ...
  ]
}}

JSON:"""

        response = llm.generate(prompt, temperature=0)

        # Parse JSON
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get('entities', [])
            except json.JSONDecodeError:
                pass

        # Fallback to basic extraction
        return self.extract_entities(text)

    def extract_with_context(self, text):
        """
        Extract entities with surrounding context.

        Args:
            text: Input text

        Returns:
            Entities with context
        """
        entities = self.extract_entities_json(text)

        # Add context (surrounding words)
        words = text.split()

        for entity in entities:
            # Find entity in text
            entity_text = entity['text']

            # Simple context extraction (can be improved)
            try:
                start = text.index(entity_text)
                end = start + len(entity_text)

                # Get surrounding words (5 words before and after)
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)

                entity['context'] = text[context_start:context_end]
                entity['position'] = {'start': start, 'end': end}
            except ValueError:
                pass

        return entities

# Example Usage
print("\n" + "="*80)
print("NER Extraction Example\n")

text = """
Apple Inc. announced record earnings on January 15, 2024. CEO Tim Cook
stated that iPhone sales in China exceeded $50 billion in Q4. The company
plans to open new stores in Tokyo and London next year.
"""

extractor = NERExtractor()

print("Text:")
print(text)

print("\n\nExtracted Entities:")
entities = extractor.extract_entities_json(text)

for entity in entities:
    print(f"  {entity['text']} → {entity['type']}")
```

### Advanced NER Patterns

```python
def advanced_ner_patterns():
    """Advanced patterns for NER."""

    print("\n\nAdvanced NER Patterns:\n")
    print("="*80)

    class AdvancedNER:
        """Advanced NER with disambiguation and normalization."""

        def extract_with_disambiguation(self, text):
            """
            Extract entities and disambiguate them.

            Args:
                text: Input text

            Returns:
                Disambiguated entities
            """
            # Extract entities first
            prompt = f"""
Extract named entities and disambiguate them when possible.

For example:
- "Apple" could be the company or the fruit → specify which
- "Washington" could be the president, state, or city → clarify

Text: {text}

Format:
<entity> | <type> | <disambiguation>

Entities:"""

            response = llm.generate(prompt, temperature=0)

            entities = []
            for line in response.strip().split('\n'):
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2:
                        entities.append({
                            'text': parts[0],
                            'type': parts[1],
                            'disambiguation': parts[2] if len(parts) > 2 else None
                        })

            return entities

        def extract_with_normalization(self, text):
            """
            Extract and normalize entities.

            Args:
                text: Input text

            Returns:
                Normalized entities
            """
            prompt = f"""
Extract entities and provide normalized forms.

Examples:
- "Dr. Smith" → normalize to "Smith, Dr." or full name if known
- "Jan 15, 2024" → normalize to "2024-01-15"
- "$50B" → normalize to "$50,000,000,000"

Text: {text}

Format:
<original> | <type> | <normalized>

Entities:"""

            response = llm.generate(prompt, temperature=0)

            entities = []
            for line in response.strip().split('\n'):
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2:
                        entities.append({
                            'original': parts[0],
                            'type': parts[1],
                            'normalized': parts[2] if len(parts) > 2 else parts[0]
                        })

            return entities

        def extract_nested_entities(self, text):
            """
            Extract nested entities.

            Example: "University of California, Berkeley" contains
            both "University of California, Berkeley" (ORG) and
            "Berkeley" (LOCATION)

            Args:
                text: Input text

            Returns:
                Nested entity structures
            """
            prompt = f"""
Extract entities including nested entities.

Example:
"Microsoft Office Suite" contains:
- "Microsoft Office Suite" (PRODUCT)
  - "Microsoft" (ORGANIZATION)

Text: {text}

Format (use indentation for nesting):
<entity> | <type>
  <nested entity> | <type>

Entities:"""

            response = llm.generate(prompt, temperature=0)

            # Parse nested structure (simplified)
            entities = []
            current_parent = None

            for line in response.strip().split('\n'):
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2:
                        entity = {
                            'text': parts[0],
                            'type': parts[1]
                        }

                        if line.startswith('  '):  # Nested
                            if current_parent:
                                if 'children' not in current_parent:
                                    current_parent['children'] = []
                                current_parent['children'].append(entity)
                        else:  # Top-level
                            entities.append(entity)
                            current_parent = entity

            return entities

    print("""
Advanced NER Techniques:

1. DISAMBIGUATION
   Handle ambiguous entities (e.g., "Washington" → which one?)

2. NORMALIZATION
   Convert to standard forms (dates, currencies, names)

3. NESTED ENTITIES
   Extract entities within entities

4. COREFERENCE RESOLUTION
   Link pronouns to entities ("he" → "Tim Cook")

5. ENTITY LINKING
   Link to knowledge base IDs (e.g., Wikidata)

6. MULTI-WORD EXPRESSIONS
   Handle complex entity names correctly

7. CONTEXT-AWARE EXTRACTION
   Use context to determine entity type
""")

advanced_ner_patterns()
```

## Relation Extraction

### Extracting Relationships

```python
class RelationExtractor:
    """Extract relationships between entities."""

    def __init__(self, relation_types=None):
        """
        Initialize relation extractor.

        Args:
            relation_types: List of relation types to extract
        """
        if relation_types is None:
            self.relation_types = [
                "WORKS_FOR", "CEO_OF", "LOCATED_IN", "FOUNDED",
                "ACQUIRED", "PARENT_OF", "SUBSIDIARY_OF", "PARTNER_WITH"
            ]
        else:
            self.relation_types = relation_types

    def extract_relations(self, text, entities=None):
        """
        Extract relations from text.

        Args:
            text: Input text
            entities: Optional pre-extracted entities

        Returns:
            List of relation triples
        """
        relation_types_str = ", ".join(self.relation_types)

        prompt = f"""
Extract relationships between entities in this text.

Relation types: {relation_types_str}

Text: {text}

Output format (one per line):
<subject> | <relation> | <object>

Relations:"""

        response = llm.generate(prompt, temperature=0)

        # Parse relations
        relations = []
        for line in response.strip().split('\n'):
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3:
                    relations.append({
                        'subject': parts[0],
                        'relation': parts[1].upper(),
                        'object': parts[2]
                    })

        return relations

    def extract_relations_json(self, text):
        """
        Extract relations with JSON output.

        Args:
            text: Input text

        Returns:
            Structured relation data
        """
        relation_types_str = ", ".join(self.relation_types)

        prompt = f"""
Extract all relationships from this text and output as JSON.

Relation types: {relation_types_str}

Text: {text}

Output format:
{{
  "relations": [
    {{
      "subject": "entity1",
      "relation": "RELATION_TYPE",
      "object": "entity2"
    }},
    ...
  ]
}}

JSON:"""

        response = llm.generate(prompt, temperature=0)

        # Parse JSON
        import json
        import re

        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get('relations', [])
            except json.JSONDecodeError:
                pass

        return self.extract_relations(text)

    def extract_relations_with_evidence(self, text):
        """
        Extract relations with supporting evidence.

        Args:
            text: Input text

        Returns:
            Relations with evidence
        """
        prompt = f"""
Extract relationships and provide the text span that supports each relation.

Text: {text}

Output format (JSON):
{{
  "relations": [
    {{
      "subject": "...",
      "relation": "...",
      "object": "...",
      "evidence": "text span supporting this relation"
    }}
  ]
}}

JSON:"""

        response = llm.generate(prompt, temperature=0)

        import json
        import re

        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get('relations', [])
            except json.JSONDecodeError:
                pass

        return []

    def build_knowledge_graph(self, text):
        """
        Build a knowledge graph from text.

        Args:
            text: Input text

        Returns:
            Graph structure with nodes and edges
        """
        # Extract entities
        ner = NERExtractor()
        entities = ner.extract_entities_json(text)

        # Extract relations
        relations = self.extract_relations_json(text)

        # Build graph
        graph = {
            'nodes': [
                {'id': i, 'label': e['text'], 'type': e['type']}
                for i, e in enumerate(entities)
            ],
            'edges': []
        }

        # Create entity lookup
        entity_lookup = {e['text']: i for i, e in enumerate(entities)}

        # Add edges
        for rel in relations:
            subj_id = entity_lookup.get(rel['subject'])
            obj_id = entity_lookup.get(rel['object'])

            if subj_id is not None and obj_id is not None:
                graph['edges'].append({
                    'source': subj_id,
                    'target': obj_id,
                    'label': rel['relation']
                })

        return graph

# Example
print("\n\n" + "="*80)
print("Relation Extraction Example\n")

text = """
Tim Cook is the CEO of Apple Inc., which is headquartered in Cupertino,
California. Apple acquired Beats Electronics in 2014 for $3 billion.
Beats was founded by Dr. Dre and Jimmy Iovine.
"""

extractor = RelationExtractor()

print("Text:")
print(text)

print("\n\nExtracted Relations:")
relations = extractor.extract_relations_json(text)

for rel in relations:
    print(f"  {rel['subject']} --[{rel['relation']}]--> {rel['object']}")

print("\n\nKnowledge Graph:")
graph = extractor.build_knowledge_graph(text)
print(f"Nodes: {len(graph['nodes'])}")
print(f"Edges: {len(graph['edges'])}")
```

### N-ary Relations

```python
def nary_relations():
    """Extract n-ary relations (more than 2 entities)."""

    print("\n\nN-ary Relations:\n")
    print("="*80)

    print("""
N-ary relations involve more than 2 entities and often include attributes.

Example:
"Apple acquired Beats for $3 billion in 2014"

Binary relations:
- Apple --ACQUIRED--> Beats

N-ary relation (captures full context):
{
  "type": "ACQUISITION",
  "acquirer": "Apple",
  "target": "Beats",
  "amount": "$3 billion",
  "date": "2014"
}
""")

    class NaryRelationExtractor:
        """Extract n-ary relations."""

        def extract_acquisition_events(self, text):
            """
            Extract acquisition events with all attributes.

            Args:
                text: Input text

            Returns:
                List of acquisition events
            """
            prompt = f"""
Extract all acquisition events from this text with complete details.

For each acquisition, extract:
- Acquirer (company buying)
- Target (company being bought)
- Amount (price paid, if mentioned)
- Date (when it happened, if mentioned)

Text: {text}

Output as JSON array:
[
  {{
    "acquirer": "...",
    "target": "...",
    "amount": "..." or null,
    "date": "..." or null
  }}
]

JSON:"""

            response = llm.generate(prompt, temperature=0)

            import json
            import re

            json_match = re.search(r'\[.*\]', response, re.DOTALL)

            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            return []

        def extract_employment_relations(self, text):
            """
            Extract employment relations with attributes.

            Args:
                text: Input text

            Returns:
                Employment relations
            """
            prompt = f"""
Extract employment relationships with full details.

For each relationship, extract:
- Person
- Role/Title
- Organization
- Start date (if mentioned)
- End date (if mentioned)

Text: {text}

JSON array:
[
  {{
    "person": "...",
    "role": "...",
    "organization": "...",
    "start_date": "..." or null,
    "end_date": "..." or null
  }}
]

JSON:"""

            response = llm.generate(prompt, temperature=0)

            import json
            import re

            json_match = re.search(r'\[.*\]', response, re.DOTALL)

            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            return []

    # Example
    print("\n\nExample:\n")

    text = """
    Microsoft acquired LinkedIn for $26.2 billion in June 2016. The deal
    was led by Satya Nadella, who became Microsoft CEO in 2014. Jeff Weiner
    served as LinkedIn's CEO from 2009 to 2020.
    """

    extractor = NaryRelationExtractor()

    print("Text:")
    print(text)

    print("\n\nAcquisition Events:")
    acquisitions = extractor.extract_acquisition_events(text)
    for acq in acquisitions:
        print(f"  {acq}")

    print("\n\nEmployment Relations:")
    employment = extractor.extract_employment_relations(text)
    for emp in employment:
        print(f"  {emp}")

nary_relations()
```

## Event Extraction

### Identifying Events

```python
class EventExtractor:
    """Extract events and their participants."""

    def __init__(self):
        pass

    def extract_events(self, text):
        """
        Extract events with participants and attributes.

        Args:
            text: Input text

        Returns:
            List of events
        """
        prompt = f"""
Extract all events from this text with their participants.

For each event, identify:
- Event type (e.g., ANNOUNCEMENT, ACQUISITION, LAUNCH, MEETING)
- Actor (who performed the action)
- Object (what was acted upon)
- Location (where it happened, if mentioned)
- Time (when it happened, if mentioned)

Text: {text}

Output as JSON array:
[
  {{
    "type": "EVENT_TYPE",
    "actor": "...",
    "object": "...",
    "location": "..." or null,
    "time": "..." or null
  }}
]

JSON:"""

        response = llm.generate(prompt, temperature=0)

        import json
        import re

        json_match = re.search(r'\[.*\]', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return []

    def extract_events_with_schema(self, text, event_schema):
        """
        Extract events following a specific schema.

        Args:
            text: Input text
            event_schema: Dict defining event structure

        Returns:
            Events matching schema
        """
        # Format schema for prompt
        schema_desc = json.dumps(event_schema, indent=2)

        prompt = f"""
Extract events from this text following this schema:

{schema_desc}

Text: {text}

Output events in the same JSON structure:"""

        response = llm.generate(prompt, temperature=0)

        import json
        import re

        json_match = re.search(r'\[.*\]', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return []

    def extract_event_sequences(self, text):
        """
        Extract sequences of related events (storylines).

        Args:
            text: Input text

        Returns:
            Ordered list of events
        """
        prompt = f"""
Extract the sequence of events from this text in chronological order.

For each event, provide:
- Order (1, 2, 3, ...)
- Description
- Time (if mentioned)

Text: {text}

Output as JSON array (ordered by time):
[
  {{"order": 1, "description": "...", "time": "..."}},
  {{"order": 2, "description": "...", "time": "..."}}
]

JSON:"""

        response = llm.generate(prompt, temperature=0)

        import json
        import re

        json_match = re.search(r'\[.*\]', response, re.DOTALL)

        if json_match:
            try:
                events = json.loads(json_match.group())
                return sorted(events, key=lambda x: x.get('order', 0))
            except json.JSONDecodeError:
                pass

        return []

# Example
print("\n\n" + "="*80)
print("Event Extraction Example\n")

text = """
On January 15, 2024, Apple CEO Tim Cook announced the iPhone 15 at an event
in Cupertino. The new phone features an improved camera and longer battery life.
Pre-orders opened on January 20, and the phone launched on January 27.
Initial sales exceeded expectations, with 10 million units sold in the first week.
"""

extractor = EventExtractor()

print("Text:")
print(text)

print("\n\nExtracted Events:")
events = extractor.extract_events(text)
for i, event in enumerate(events, 1):
    print(f"\n{i}. {event['type']}")
    print(f"   Actor: {event.get('actor', 'N/A')}")
    print(f"   Object: {event.get('object', 'N/A')}")
    print(f"   Time: {event.get('time', 'N/A')}")

print("\n\nEvent Sequence:")
sequence = extractor.extract_event_sequences(text)
for event in sequence:
    print(f"{event['order']}. [{event.get('time', 'N/A')}] {event['description']}")
```

### Complex Event Patterns

```python
def complex_event_patterns():
    """Extract complex event patterns."""

    print("\n\nComplex Event Patterns:\n")
    print("="*80)

    print("""
Complex event patterns involve:

1. NESTED EVENTS
   Event contains sub-events
   Example: "Conference" event contains "Keynote", "Panel", "Networking" sub-events

2. CAUSAL CHAINS
   One event causes another
   Example: "Revenue decline" → "Layoffs" → "Stock drop"

3. CONDITIONAL EVENTS
   Events that depend on conditions
   Example: "If FDA approves, launch will proceed in Q3"

4. RECURRING EVENTS
   Events that repeat
   Example: "Quarterly earnings calls"

5. MULTI-STAGE EVENTS
   Events with distinct phases
   Example: "Product development" → "Testing" → "Launch" → "Post-launch support"
""")

    class ComplexEventExtractor:
        """Extract complex event patterns."""

        def extract_causal_chain(self, text):
            """
            Extract chain of causally related events.

            Args:
                text: Input text

            Returns:
                Causal event chain
            """
            prompt = f"""
Extract the causal chain of events from this text.

For each event, identify:
- Event description
- Caused by (previous event, or null if first)
- Led to (next event, or null if last)

Text: {text}

Output as JSON array showing causal links:
[
  {{
    "event": "...",
    "caused_by": "..." or null,
    "led_to": "..." or null
  }}
]

JSON:"""

            response = llm.generate(prompt, temperature=0)

            import json
            import re

            json_match = re.search(r'\[.*\]', response, re.DOTALL)

            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            return []

        def extract_conditional_events(self, text):
            """
            Extract events with conditions.

            Args:
                text: Input text

            Returns:
                Events with conditions
            """
            prompt = f"""
Extract events and their conditions from this text.

For each conditional event:
- Condition (what must happen first)
- Event (what will happen if condition is met)
- Confidence (if mentioned: certain, likely, possible)

Text: {text}

JSON array:
[
  {{
    "condition": "...",
    "event": "...",
    "confidence": "certain/likely/possible"
  }}
]

JSON:"""

            response = llm.generate(prompt, temperature=0)

            import json
            import re

            json_match = re.search(r'\[.*\]', response, re.DOTALL)

            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            return []

    # Example
    print("\n\nExample:\n")

    text = """
    The company's revenue declined by 20% in Q1 due to supply chain issues.
    This led to a 15% workforce reduction in May. As a result, the stock
    price dropped 30%. If market conditions improve, the company plans to
    resume hiring in Q4.
    """

    extractor = ComplexEventExtractor()

    print("Text:")
    print(text)

    print("\n\nCausal Chain:")
    chain = extractor.extract_causal_chain(text)
    for link in chain:
        print(f"  Event: {link['event']}")
        if link.get('caused_by'):
            print(f"    Caused by: {link['caused_by']}")
        if link.get('led_to'):
            print(f"    Led to: {link['led_to']}")
        print()

    print("\nConditional Events:")
    conditionals = extractor.extract_conditional_events(text)
    for cond in conditionals:
        print(f"  IF: {cond['condition']}")
        print(f"  THEN: {cond['event']}")
        print(f"  Confidence: {cond.get('confidence', 'unknown')}\n")

complex_event_patterns()
```

## Structured Extraction with LLMs

### JSON Schema Extraction

```python
class StructuredExtractor:
    """Extract information into structured JSON format."""

    def __init__(self):
        pass

    def extract_with_schema(self, text, schema):
        """
        Extract information following a JSON schema.

        Args:
            text: Input text
            schema: JSON schema defining expected structure

        Returns:
            Extracted data matching schema
        """
        import json

        schema_str = json.dumps(schema, indent=2)

        prompt = f"""
Extract information from this text according to the following JSON schema:

Schema:
{schema_str}

Text: {text}

Output the extracted data as valid JSON matching the schema:"""

        response = llm.generate(prompt, temperature=0)

        # Parse JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return data
            except json.JSONDecodeError as e:
                return {'error': f'JSON parsing failed: {str(e)}'}

        return {'error': 'No JSON found in response'}

    def extract_form_data(self, text, fields):
        """
        Extract form-like structured data.

        Args:
            text: Input text
            fields: List of field names to extract

        Returns:
            Dictionary with extracted fields
        """
        fields_str = ", ".join(fields)

        prompt = f"""
Extract the following fields from this text: {fields_str}

If a field is not mentioned, use null.

Text: {text}

Output as JSON:
{{
  "{fields[0]}": "..." or null,
  "{fields[1]}": "..." or null,
  ...
}}

JSON:"""

        response = llm.generate(prompt, temperature=0)

        import json
        import re

        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {field: None for field in fields}

    def extract_table_data(self, text):
        """
        Extract structured tabular data from text.

        Args:
            text: Input text

        Returns:
            List of dictionaries (rows)
        """
        prompt = f"""
Extract structured data from this text as a table (array of objects).

Identify the columns/fields and extract all rows.

Text: {text}

Output as JSON array:
[
  {{"field1": "value1", "field2": "value2", ...}},
  {{"field1": "value1", "field2": "value2", ...}}
]

JSON:"""

        response = llm.generate(prompt, temperature=0)

        import json
        import re

        json_match = re.search(r'\[.*\]', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return []

# Example
print("\n\n" + "="*80)
print("Structured Extraction Example\n")

text = """
Product: iPhone 15 Pro
Manufacturer: Apple Inc.
Price: $999
Release Date: September 15, 2023
Features: 6.1" display, A17 Pro chip, 48MP camera, Titanium frame
Colors: Black, White, Blue, Natural Titanium
Storage: 128GB, 256GB, 512GB, 1TB
"""

extractor = StructuredExtractor()

# Define schema
schema = {
    "product_name": "string",
    "manufacturer": "string",
    "price": "string",
    "release_date": "string",
    "features": ["string"],
    "colors": ["string"],
    "storage_options": ["string"]
}

print("Text:")
print(text)

print("\n\nExtracted Data (with schema):")
data = extractor.extract_with_schema(text, schema)
import json
print(json.dumps(data, indent=2))
```

### Form and Document Extraction

```python
def form_extraction_patterns():
    """Patterns for extracting data from forms and documents."""

    print("\n\nForm and Document Extraction:\n")
    print("="*80)

    class FormExtractor:
        """Extract data from various document types."""

        def extract_invoice(self, text):
            """Extract invoice data."""

            schema = {
                "invoice_number": "string",
                "date": "string",
                "vendor": {
                    "name": "string",
                    "address": "string"
                },
                "customer": {
                    "name": "string",
                    "address": "string"
                },
                "items": [
                    {
                        "description": "string",
                        "quantity": "number",
                        "unit_price": "number",
                        "total": "number"
                    }
                ],
                "subtotal": "number",
                "tax": "number",
                "total": "number"
            }

            extractor = StructuredExtractor()
            return extractor.extract_with_schema(text, schema)

        def extract_resume(self, text):
            """Extract resume/CV data."""

            schema = {
                "name": "string",
                "contact": {
                    "email": "string",
                    "phone": "string",
                    "address": "string"
                },
                "summary": "string",
                "experience": [
                    {
                        "company": "string",
                        "title": "string",
                        "start_date": "string",
                        "end_date": "string",
                        "responsibilities": ["string"]
                    }
                ],
                "education": [
                    {
                        "institution": "string",
                        "degree": "string",
                        "field": "string",
                        "graduation_year": "string"
                    }
                ],
                "skills": ["string"]
            }

            extractor = StructuredExtractor()
            return extractor.extract_with_schema(text, schema)

        def extract_contract(self, text):
            """Extract key contract terms."""

            fields = [
                "parties",
                "effective_date",
                "termination_date",
                "payment_terms",
                "deliverables",
                "penalties",
                "governing_law"
            ]

            extractor = StructuredExtractor()
            return extractor.extract_form_data(text, fields)

    print("""
Document Type Examples:

1. INVOICES
   Extract: Number, dates, vendor, customer, line items, totals

2. RESUMES/CVs
   Extract: Contact info, experience, education, skills

3. CONTRACTS
   Extract: Parties, dates, terms, obligations, penalties

4. MEDICAL RECORDS
   Extract: Patient info, diagnoses, medications, procedures

5. FINANCIAL STATEMENTS
   Extract: Revenue, expenses, assets, liabilities

6. RESEARCH PAPERS
   Extract: Authors, abstract, methods, results, references

Key Challenges:
• Varying formats
• Missing information
• Ambiguous structures
• Multi-page documents

Solutions:
• Flexible schemas (optional fields)
• Format normalization
• Multi-pass extraction
• Confidence scoring
""")

form_extraction_patterns()
```

## Few-Shot Extraction

### Learning from Examples

```python
class FewShotExtractor:
    """Extract information using few-shot learning."""

    def __init__(self):
        self.examples = []

    def add_example(self, text, extracted_data):
        """
        Add an example for few-shot learning.

        Args:
            text: Example input text
            extracted_data: Corresponding extracted data
        """
        self.examples.append({
            'text': text,
            'data': extracted_data
        })

    def extract_with_examples(self, text):
        """
        Extract using few-shot examples.

        Args:
            text: Input text

        Returns:
            Extracted data
        """
        import json

        # Build prompt with examples
        examples_text = ""
        for i, ex in enumerate(self.examples, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Text: {ex['text']}\n"
            examples_text += f"Extracted: {json.dumps(ex['data'])}\n"

        prompt = f"""
Extract information from text following these examples:
{examples_text}

Now extract from this text:
Text: {text}

Extracted:"""

        response = llm.generate(prompt, temperature=0)

        # Parse JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {}

# Example
print("\n\n" + "="*80)
print("Few-Shot Extraction Example\n")

extractor = FewShotExtractor()

# Add examples
extractor.add_example(
    "Dr. Sarah Johnson, MD, works at Boston General Hospital in Cardiology.",
    {
        "name": "Sarah Johnson",
        "title": "MD",
        "organization": "Boston General Hospital",
        "department": "Cardiology"
    }
)

extractor.add_example(
    "Prof. Michael Chen teaches Computer Science at MIT.",
    {
        "name": "Michael Chen",
        "title": "Prof.",
        "organization": "MIT",
        "department": "Computer Science"
    }
)

# Extract from new text
new_text = "Dr. Emily Rodriguez is a researcher in Neuroscience at Stanford University."

print("Examples provided:")
for i, ex in enumerate(extractor.examples, 1):
    print(f"\n{i}. {ex['text']}")
    print(f"   → {ex['data']}")

print(f"\n\nNew text to extract from:")
print(new_text)

print("\n\nExtracted:")
result = extractor.extract_with_examples(new_text)
import json
print(json.dumps(result, indent=2))
```

## Validation and Error Handling

### Validating Extracted Data

```python
class ExtractionValidator:
    """Validate extracted information."""

    def __init__(self):
        pass

    def validate_schema(self, data, schema):
        """
        Validate data against schema.

        Args:
            data: Extracted data
            schema: Expected schema

        Returns:
            Validation result
        """
        errors = []

        def check_type(value, expected_type, path=""):
            """Check if value matches expected type."""
            if expected_type == "string":
                if not isinstance(value, str):
                    errors.append(f"{path}: expected string, got {type(value).__name__}")
            elif expected_type == "number":
                if not isinstance(value, (int, float)):
                    errors.append(f"{path}: expected number, got {type(value).__name__}")
            elif isinstance(expected_type, list):
                if not isinstance(value, list):
                    errors.append(f"{path}: expected array, got {type(value).__name__}")
                elif value and expected_type:
                    for i, item in enumerate(value):
                        check_type(item, expected_type[0], f"{path}[{i}]")
            elif isinstance(expected_type, dict):
                if not isinstance(value, dict):
                    errors.append(f"{path}: expected object, got {type(value).__name__}")
                else:
                    for key, val_type in expected_type.items():
                        if key in value:
                            check_type(value[key], val_type, f"{path}.{key}")
                        else:
                            errors.append(f"{path}.{key}: missing required field")

        check_type(data, schema)

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def validate_completeness(self, data, required_fields):
        """
        Check if all required fields are present and non-empty.

        Args:
            data: Extracted data
            required_fields: List of required field names

        Returns:
            Completeness report
        """
        missing = []
        empty = []

        for field in required_fields:
            if field not in data:
                missing.append(field)
            elif data[field] in [None, "", []]:
                empty.append(field)

        return {
            'complete': len(missing) == 0 and len(empty) == 0,
            'missing_fields': missing,
            'empty_fields': empty,
            'completeness_score': 1 - (len(missing) + len(empty)) / len(required_fields)
        }

    def validate_consistency(self, data, source_text):
        """
        Check if extracted data is consistent with source.

        Args:
            data: Extracted data
            source_text: Original text

        Returns:
            Consistency report
        """
        import json

        # Check if extracted values appear in source
        inconsistencies = []

        def check_value(value, path=""):
            """Recursively check if values appear in source."""
            if isinstance(value, str) and len(value) > 3:
                # Check if string appears in source (case-insensitive)
                if value.lower() not in source_text.lower():
                    inconsistencies.append({
                        'field': path,
                        'value': value,
                        'issue': 'not found in source text'
                    })
            elif isinstance(value, dict):
                for key, val in value.items():
                    check_value(val, f"{path}.{key}" if path else key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_value(item, f"{path}[{i}]")

        check_value(data)

        return {
            'consistent': len(inconsistencies) == 0,
            'inconsistencies': inconsistencies
        }

    def validate_with_llm(self, data, source_text):
        """
        Use LLM to validate extraction quality.

        Args:
            data: Extracted data
            source_text: Original text

        Returns:
            Validation result
        """
        import json

        data_str = json.dumps(data, indent=2)

        prompt = f"""
Validate this extracted data against the source text.

Source text:
{source_text}

Extracted data:
{data_str}

For each extracted field, check:
1. Is it factually correct?
2. Is it complete?
3. Is there any hallucinated information?

Output validation result as JSON:
{{
  "overall_quality": "excellent/good/fair/poor",
  "factually_correct": true/false,
  "complete": true/false,
  "issues": ["list of issues found"],
  "confidence": 0.0-1.0
}}

JSON:"""

        response = llm.generate(prompt, temperature=0)

        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            'overall_quality': 'unknown',
            'factually_correct': None,
            'complete': None,
            'issues': [],
            'confidence': 0.0
        }

# Example
print("\n\n" + "="*80)
print("Validation Example\n")

source_text = "Dr. John Smith works at Boston General Hospital."

extracted_data = {
    "name": "John Smith",
    "title": "Dr.",
    "organization": "Boston General Hospital"
}

validator = ExtractionValidator()

# Schema validation
schema = {
    "name": "string",
    "title": "string",
    "organization": "string"
}

print("Schema Validation:")
result = validator.validate_schema(extracted_data, schema)
print(f"Valid: {result['valid']}")
if result['errors']:
    print(f"Errors: {result['errors']}")

# Completeness check
print("\n\nCompleteness Check:")
required_fields = ["name", "title", "organization", "department"]
result = validator.validate_completeness(extracted_data, required_fields)
print(f"Complete: {result['complete']}")
print(f"Score: {result['completeness_score']:.2f}")
if result['missing_fields']:
    print(f"Missing: {result['missing_fields']}")

# Consistency check
print("\n\nConsistency Check:")
result = validator.validate_consistency(extracted_data, source_text)
print(f"Consistent: {result['consistent']}")
if result['inconsistencies']:
    for issue in result['inconsistencies']:
        print(f"  {issue}")
```

### Error Recovery

```python
def error_recovery_strategies():
    """Strategies for handling extraction errors."""

    print("\n\nError Recovery Strategies:\n")
    print("="*80)

    print("""
Common extraction errors and recovery strategies:

1. INCOMPLETE EXTRACTION
   Error: Missing required fields
   Recovery:
   - Use multiple prompts targeting specific fields
   - Provide examples of complete extractions
   - Fallback to default/null values

2. FORMAT ERRORS
   Error: Invalid JSON, wrong data types
   Recovery:
   - Retry with clearer format instructions
   - Parse with lenient JSON parser
   - Post-process to fix common issues

3. HALLUCINATION
   Error: Extracted data not in source
   Recovery:
   - Validate against source
   - Use extractive approach (quote from source)
   - Lower temperature for more faithful extraction

4. AMBIGUITY
   Error: Multiple valid interpretations
   Recovery:
   - Request disambiguation
   - Extract multiple options with confidence scores
   - Provide more context in prompt

5. COMPLEX STRUCTURE
   Error: Nested/complex data poorly extracted
   Recovery:
   - Break into smaller extraction tasks
   - Hierarchical extraction (coarse to fine)
   - Iterative refinement

6. NOISE IN TEXT
   Error: Irrelevant text affects extraction
   Recovery:
   - Pre-filter text (remove noise)
   - Focus on relevant sections
   - Provide negative examples
""")

    class ErrorRecoveryExtractor:
        """Extractor with error recovery."""

        def extract_with_retry(self, text, schema, max_retries=3):
            """
            Extract with automatic retry on failure.

            Args:
                text: Input text
                schema: Expected schema
                max_retries: Maximum retry attempts

            Returns:
                Extracted data or error
            """
            extractor = StructuredExtractor()
            validator = ExtractionValidator()

            for attempt in range(max_retries):
                # Extract
                data = extractor.extract_with_schema(text, schema)

                # Validate
                validation = validator.validate_schema(data, schema)

                if validation['valid']:
                    return data

                # If invalid, provide feedback for retry
                if attempt < max_retries - 1:
                    errors_str = "\n".join(validation['errors'])
                    # Would normally provide error feedback to LLM for retry
                    continue

            return {
                'error': 'Extraction failed after retries',
                'last_attempt': data
            }

        def extract_with_fallback(self, text, schema):
            """
            Extract with fallback to simpler extraction.

            Args:
                text: Input text
                schema: Expected schema

            Returns:
                Extracted data
            """
            extractor = StructuredExtractor()

            # Try full structured extraction
            data = extractor.extract_with_schema(text, schema)

            validator = ExtractionValidator()
            validation = validator.validate_schema(data, schema)

            if validation['valid']:
                return data

            # Fallback: Extract fields individually
            print("Full extraction failed, falling back to field-by-field extraction...")

            result = {}
            for field in schema.keys():
                field_data = extractor.extract_form_data(text, [field])
                if field in field_data:
                    result[field] = field_data[field]

            return result

    print("\n\nExample: Retry with feedback\n")
    print("""
Attempt 1: Extract → Validate → Fails
  Error: "Missing 'email' field"

Attempt 2: Extract with instruction "Must include email" → Validate → Success
""")

error_recovery_strategies()
```

## Domain Adaptation

### Customizing for Specific Domains

```python
def domain_adaptation():
    """Adapt extraction to specific domains."""

    print("\n\nDomain Adaptation:\n")
    print("="*80)

    print("""
Domain-specific extraction requires:

1. DOMAIN VOCABULARY
   Use domain-specific terminology in prompts

2. CUSTOM ENTITY TYPES
   Define entities relevant to domain

3. DOMAIN SCHEMAS
   Structure data according to domain conventions

4. EXAMPLES FROM DOMAIN
   Provide domain-specific few-shot examples

5. DOMAIN CONSTRAINTS
   Apply domain rules (e.g., medical codes, legal citations)
""")

    class DomainExtractor:
        """Domain-adapted extractors."""

        def extract_medical(self, text):
            """Extract medical information."""

            schema = {
                "patient": {
                    "name": "string",
                    "age": "number",
                    "gender": "string"
                },
                "chief_complaint": "string",
                "symptoms": ["string"],
                "diagnoses": [
                    {
                        "condition": "string",
                        "icd_code": "string"
                    }
                ],
                "medications": [
                    {
                        "name": "string",
                        "dosage": "string",
                        "frequency": "string"
                    }
                ],
                "procedures": ["string"],
                "follow_up": "string"
            }

            extractor = StructuredExtractor()
            return extractor.extract_with_schema(text, schema)

        def extract_legal(self, text):
            """Extract legal information."""

            schema = {
                "case_number": "string",
                "court": "string",
                "parties": {
                    "plaintiff": "string",
                    "defendant": "string"
                },
                "filing_date": "string",
                "legal_issues": ["string"],
                "statutes_cited": ["string"],
                "precedents_cited": ["string"],
                "ruling": "string",
                "judge": "string"
            }

            extractor = StructuredExtractor()
            return extractor.extract_with_schema(text, schema)

        def extract_financial(self, text):
            """Extract financial information."""

            schema = {
                "company": "string",
                "reporting_period": "string",
                "revenue": {
                    "amount": "number",
                    "currency": "string",
                    "change_yoy": "string"
                },
                "net_income": {
                    "amount": "number",
                    "currency": "string"
                },
                "eps": "number",
                "key_metrics": {
                    "pe_ratio": "number",
                    "debt_to_equity": "number"
                },
                "guidance": "string"
            }

            extractor = StructuredExtractor()
            return extractor.extract_with_schema(text, schema)

    print("\n\nDomain Examples:\n")
    print("""
MEDICAL:
- Extract: Diagnoses, medications, procedures, symptoms
- Standards: ICD codes, drug names, dosages
- Challenges: Medical terminology, abbreviations

LEGAL:
- Extract: Parties, case numbers, statutes, rulings
- Standards: Legal citations, court formats
- Challenges: Complex language, precedents

FINANCIAL:
- Extract: Revenue, profits, metrics, guidance
- Standards: GAAP, financial ratios
- Challenges: Numbers, percentages, year-over-year changes

SCIENTIFIC:
- Extract: Methods, results, hypotheses, conclusions
- Standards: Scientific notation, statistical significance
- Challenges: Technical terminology, complex relationships
""")

domain_adaptation()
```

## Production Patterns

### Scalable Extraction Pipeline

```python
class ProductionExtractor:
    """Production-ready information extraction system."""

    def __init__(self, cache_enabled=True):
        """Initialize production extractor."""
        self.cache_enabled = cache_enabled
        self.cache = {}

    def extract_pipeline(self, text, schema, options=None):
        """
        Complete extraction pipeline.

        Args:
            text: Input text
            schema: Extraction schema
            options: Configuration options

        Returns:
            Extraction result with metadata
        """
        import time
        import hashlib

        start_time = time.time()

        if options is None:
            options = {}

        validate = options.get('validate', True)
        max_retries = options.get('max_retries', 2)

        # Check cache
        cache_key = hashlib.md5(
            (text + str(schema)).encode()
        ).hexdigest()

        if self.cache_enabled and cache_key in self.cache:
            cached = self.cache[cache_key]
            cached['metadata']['cache_hit'] = True
            return cached

        # Validate input
        if not text or len(text.strip()) < 10:
            return {
                'error': 'Text too short',
                'data': None
            }

        # Extract with retry
        extractor = StructuredExtractor()
        validator = ExtractionValidator()

        data = None
        validation_result = None

        for attempt in range(max_retries + 1):
            data = extractor.extract_with_schema(text, schema)

            if 'error' in data:
                if attempt < max_retries:
                    continue
                else:
                    break

            if validate:
                validation_result = validator.validate_schema(data, schema)
                if validation_result['valid']:
                    break
            else:
                break

        # Build result
        result = {
            'data': data,
            'metadata': {
                'latency_ms': int((time.time() - start_time) * 1000),
                'cache_hit': False,
                'attempts': attempt + 1
            }
        }

        if validate and validation_result:
            result['metadata']['validation'] = validation_result

        # Cache result
        if self.cache_enabled and 'error' not in data:
            self.cache[cache_key] = result

        return result

    def batch_extract(self, texts, schema):
        """
        Batch extraction for efficiency.

        Args:
            texts: List of input texts
            schema: Extraction schema

        Returns:
            List of extraction results
        """
        results = []

        for text in texts:
            result = self.extract_pipeline(text, schema)
            results.append(result)

        return results

# Example
print("\n\n" + "="*80)
print("Production Pipeline Example\n")

extractor = ProductionExtractor(cache_enabled=True)

text = """
Apple Inc. announced record earnings on January 15, 2024.
CEO Tim Cook stated that revenue reached $120 billion in Q4,
up 12% year-over-year.
"""

schema = {
    "company": "string",
    "announcement_date": "string",
    "ceo": "string",
    "revenue": {
        "amount": "string",
        "period": "string",
        "growth": "string"
    }
}

result = extractor.extract_pipeline(text, schema, options={'validate': True})

print("Extracted Data:")
import json
print(json.dumps(result['data'], indent=2))

print("\n\nMetadata:")
for key, value in result['metadata'].items():
    print(f"  {key}: {value}")
```

## Summary

**Information Extraction Overview**:

```
Unstructured Text → IE System → Structured Data

Core Tasks:
  • Named Entity Recognition (NER): Identify entities
  • Relation Extraction: Find relationships between entities
  • Event Extraction: Identify events and participants
```

**Key Components**:

| Task     | Input           | Output           | Example                   |
| -------- | --------------- | ---------------- | ------------------------- |
| NER      | Text            | Entity list      | "Apple" → ORG             |
| Relation | Text + Entities | Relation triples | (Tim Cook, CEO_OF, Apple) |
| Event    | Text            | Event structures | ANNOUNCEMENT event        |

**Common Entity Types**:

- PERSON, ORGANIZATION, LOCATION
- DATE, TIME, MONEY, PERCENT
- PRODUCT, EVENT, LAW

**Extraction Approaches**:

1. **Basic Extraction**: Prompt for entities/relations
2. **JSON Schema**: Define structure, extract to JSON
3. **Few-Shot**: Provide examples for consistency
4. **Iterative**: Extract, validate, refine

**Best Practices**:

1. **Schema Design**:
   - Clear, hierarchical structure
   - Optional vs required fields
   - Type specifications

2. **Prompt Engineering**:
   - Specify output format (JSON)
   - Provide examples
   - Define entity/relation types
   - Request evidence/context

3. **Validation**:
   - Schema compliance
   - Completeness checking
   - Consistency with source
   - No hallucination

4. **Error Handling**:
   - Retry with feedback
   - Fallback strategies
   - Partial extraction acceptance
   - Confidence scores

**Domain Adaptation**:

| Domain     | Key Extractions               | Challenges         |
| ---------- | ----------------------------- | ------------------ |
| Medical    | Diagnoses, meds, procedures   | Terminology, codes |
| Legal      | Parties, statutes, rulings    | Complex language   |
| Financial  | Metrics, guidance, numbers    | Quantitative data  |
| Scientific | Methods, results, conclusions | Technical terms    |

**Production Considerations**:

- **Caching**: Store extracted data for repeated content
- **Batch Processing**: Process multiple documents efficiently
- **Validation Pipeline**: Automated quality checks
- **Error Recovery**: Retry logic, fallback extraction
- **Monitoring**: Track extraction quality metrics
- **Versioning**: Schema versioning for changes

**Common Pitfalls**:

- Hallucination → Validate against source text
- Incomplete extraction → Specify required fields clearly
- Format errors → Use strict JSON schemas
- Domain mismatch → Adapt prompts to domain
- Missing context → Include surrounding text

**Validation Checklist**:

- ☐ Schema compliance (correct types, structure)
- ☐ Completeness (all required fields present)
- ☐ Consistency (values match source text)
- ☐ No hallucination (everything verifiable in source)
- ☐ Format validity (valid JSON, data types)

**Key Takeaways**:

- Structure extraction with clear schemas
- Always validate extracted data
- Use few-shot examples for consistency
- Adapt to domain-specific requirements
- Implement retry and fallback strategies
- Cache aggressively for repeated content
- Monitor quality continuously in production

## Next Steps

- Apply IE in [Question Answering](question-answering.md) systems
- Use extracted data for [Knowledge Graphs](../rag/)
- Integrate with [Semantic Search](semantic-search.md)
- Build [Conversational AI](conversational-ai.md) with entity tracking
- Study [Evaluation Methods](../evaluation/) for IE quality
- Explore [RAG](../retrieval-augmented-generation/) for knowledge-enhanced extraction
