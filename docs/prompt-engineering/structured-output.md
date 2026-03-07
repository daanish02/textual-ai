# Structured Output

## Table of Contents

- [Introduction](#introduction)
- [Why Structured Output Matters](#why-structured-output-matters)
- [JSON Output](#json-output)
- [XML Output](#xml-output)
- [Output Schemas](#output-schemas)
- [Format Enforcement Techniques](#format-enforcement-techniques)
- [Handling Format Errors](#handling-format-errors)
- [Parsing Strategies](#parsing-strategies)
- [Complex Nested Structures](#complex-nested-structures)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Structured output** refers to getting language models to return data in machine-readable formats like JSON, XML, or other structured formats instead of free-form text. This is essential for integrating LLM outputs into applications.

```
Human-readable (unstructured):
"John is 25 years old and lives in New York. He works as an engineer."

Machine-readable (structured):
{
  "name": "John",
  "age": 25,
  "location": "New York",
  "occupation": "engineer"
}
```

**Why it matters**:

- Enables programmatic processing of LLM outputs
- Ensures consistent format across responses
- Facilitates integration with downstream systems
- Reduces parsing errors and ambiguity
- Enables validation against schemas

This guide teaches you how to reliably extract structured data from language models.

## Why Structured Output Matters

```python
def why_structured_output():
    """Understanding the importance of structured output."""

    print("Why Structured Output Matters:\n")

    print("=" * 60)
    print("\nProblem with Unstructured Text:\n")

    unstructured = """
Prompt: Extract the product, price, and rating from this review.
Review: "The SuperWidget 3000 is amazing! Costs $49.99. I give it 5 stars!"

Response: "The product is SuperWidget 3000, it costs forty-nine ninety-nine,
and the rating is 5 stars out of 5."
"""

    print(unstructured)

    print("Issues:")
    print("  ✗ Hard to parse programmatically")
    print("  ✗ Price format varies (forty-nine ninety-nine)")
    print("  ✗ Extra text (out of 5)")
    print("  ✗ Can't easily extract specific fields")
    print("  ✗ Fragile parsing rules")

    print("\n" + "=" * 60)
    print("\nSolution with Structured Output:\n")

    structured = '''
Prompt: Extract the product, price, and rating from this review as JSON.
Review: "The SuperWidget 3000 is amazing! Costs $49.99. I give it 5 stars!"

Response:
{
  "product": "SuperWidget 3000",
  "price": 49.99,
  "rating": 5
}
'''

    print(structured)

    print("Benefits:")
    print("  ✓ Easy to parse (json.loads())")
    print("  ✓ Consistent format")
    print("  ✓ Proper data types (number not string)")
    print("  ✓ Direct field access")
    print("  ✓ Reliable parsing")

    print("\n" + "=" * 60)
    print("\nUse Cases for Structured Output:\n")

    use_cases = {
        'API Integration': 'Return data to calling application',
        'Database Storage': 'Save extracted information to DB',
        'Data Pipeline': 'Feed output to next processing stage',
        'Validation': 'Check required fields present and valid',
        'Bulk Processing': 'Extract same fields from many inputs',
        'UI Rendering': 'Display structured data in interface',
        'Analytics': 'Aggregate and analyze extracted data'
    }

    for use_case, description in use_cases.items():
        print(f"  • {use_case}: {description}")

why_structured_output()
```

## JSON Output

### Basic JSON Prompting

```python
def json_output_basics():
    """How to prompt for JSON output."""

    print("Prompting for JSON Output:\n")

    print("APPROACH 1: Explicit Instruction\n")

    approach1 = """
Extract the following information from the text and return it as JSON:
- name
- email
- phone

Text: "Contact Jane Doe at jane@example.com or call 555-1234."

Return your response as valid JSON.
"""

    print(approach1)
    print("Expected Output:")
    print('''{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "phone": "555-1234"
}''')

    print("\n" + "=" * 60)
    print("\nAPPROACH 2: Template Provided\n")

    approach2 = """
Extract information from the text and fill in this JSON template:

{
  "name": "",
  "email": "",
  "phone": ""
}

Text: "Contact Jane Doe at jane@example.com or call 555-1234."
"""

    print(approach2)
    print("Expected Output:")
    print('''{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "phone": "555-1234"
}''')

    print("\n" + "=" * 60)
    print("\nAPPROACH 3: Example-Based (Few-Shot)\n")

    approach3 = """
Extract contact information as JSON.

Example:
Text: "Meet Bob Smith. Email: bob@test.com, Phone: 555-9999"
Output: {"name": "Bob Smith", "email": "bob@test.com", "phone": "555-9999"}

Text: "Contact Jane Doe at jane@example.com or call 555-1234."
Output:"""

    print(approach3)

    print("\nWhich Approach?")
    print("  • Explicit instruction: Simple, clear, works well")
    print("  • Template: Best for complex structures")
    print("  • Few-shot: Best for ambiguous formats")

json_output_basics()
```

### JSON Schema Specification

```python
def json_schema_specification():
    """Using JSON schemas for precise output format."""

    print("\n\nJSON Schema Specification:\n")

    prompt = """
Extract product information according to this JSON schema:

Schema:
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "price": {"type": "number"},
    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
    "in_stock": {"type": "boolean"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "rating": {"type": "number", "minimum": 0, "maximum": 5}
  },
  "required": ["name", "price", "currency", "in_stock"]
}

Text: "The Awesome Gadget costs $99.99 and is currently available.
       Rated 4.5 stars. Tagged as: electronics, gadget, popular"

Return valid JSON matching the schema.
"""

    print(prompt)

    print("\nExpected Output:")
    output = '''{
  "name": "Awesome Gadget",
  "price": 99.99,
  "currency": "USD",
  "in_stock": true,
  "tags": ["electronics", "gadget", "popular"],
  "rating": 4.5
}'''
    print(output)

    print("\n" + "=" * 60)
    print("\nBenefits of Schema Specification:\n")

    benefits = [
        'Precise data types (number vs string)',
        'Required vs optional fields',
        'Validation rules (min/max, enum)',
        'Array types and nested objects',
        'Documentation for the model',
        'Can validate output against schema'
    ]

    for benefit in benefits:
        print(f"  ✓ {benefit}")

json_schema_specification()
```

### Handling Optional Fields

```python
def handling_optional_fields():
    """Dealing with missing or optional data."""

    print("\n\nHandling Optional Fields in JSON:\n")

    print("Strategy 1: Use null for missing values\n")

    strategy1 = """
Extract person information as JSON. Use null if information is not present.

Text: "John Smith works as an engineer."

Output:
{
  "name": "John Smith",
  "occupation": "engineer",
  "age": null,
  "location": null
}
"""

    print(strategy1)

    print("=" * 60)
    print("\nStrategy 2: Omit missing fields\n")

    strategy2 = """
Extract person information as JSON. Only include fields that are present.

Text: "John Smith works as an engineer."

Output:
{
  "name": "John Smith",
  "occupation": "engineer"
}
"""

    print(strategy2)

    print("=" * 60)
    print("\nStrategy 3: Use empty strings/arrays\n")

    strategy3 = """
Extract person information as JSON. Use empty string for missing text fields
and empty array for missing lists.

Text: "John Smith works as an engineer."

Output:
{
  "name": "John Smith",
  "occupation": "engineer",
  "age": "",
  "location": "",
  "skills": []
}
"""

    print(strategy3)

    print("\nChoosing a Strategy:")
    print("  • null: Most semantically correct")
    print("  • Omit: Cleaner, smaller output")
    print("  • Empty: Ensures all fields present")
    print("\n  → Choose based on downstream system requirements")

handling_optional_fields()
```

## XML Output

### XML vs JSON

```python
def xml_vs_json():
    """When to use XML vs JSON for structured output."""

    print("XML vs JSON:\n")

    comparison = {
        'Simplicity': {
            'JSON': 'Simpler, less verbose',
            'XML': 'More verbose with tags'
        },
        'Data types': {
            'JSON': 'Built-in types (number, boolean, null)',
            'XML': 'Everything is text'
        },
        'Arrays': {
            'JSON': 'Native array support []',
            'XML': 'Repeated elements'
        },
        'Attributes': {
            'JSON': 'No concept of attributes',
            'XML': 'Can use attributes'
        },
        'Parsing': {
            'JSON': 'Easier to parse (json.loads())',
            'XML': 'Requires XML parser'
        },
        'Hierarchical': {
            'JSON': 'Good for nested objects',
            'XML': 'Excellent for hierarchical data'
        },
        'Mixed content': {
            'JSON': 'Cannot mix text and elements',
            'XML': 'Can have mixed content'
        }
    }

    print(f"{'Aspect':<15} {'JSON':<35} {'XML'}")
    print("=" * 85)

    for aspect, details in comparison.items():
        print(f"{aspect:<15} {details['JSON']:<35} {details['XML']}")

    print("\n" + "=" * 60)
    print("\nSame Data in Both Formats:\n")

    print("JSON:")
    json_example = '''{
  "person": {
    "name": "John Doe",
    "age": 30,
    "emails": ["john@example.com", "jdoe@test.com"]
  }
}'''
    print(json_example)

    print("\nXML:")
    xml_example = '''<person>
  <name>John Doe</name>
  <age>30</age>
  <emails>
    <email>john@example.com</email>
    <email>jdoe@test.com</email>
  </emails>
</person>'''
    print(xml_example)

    print("\nRecommendation:")
    print("  → Use JSON for most cases (simpler, modern standard)")
    print("  → Use XML when:")
    print("     • Working with XML-based systems")
    print("     • Need attributes or mixed content")
    print("     • Domain standard is XML (SOAP, etc.)")

xml_vs_json()
```

### Prompting for XML

```python
def xml_output_prompting():
    """How to prompt for XML output."""

    print("\n\nPrompting for XML Output:\n")

    prompt = """
Extract product information from the text and format as XML.

Required fields:
- name
- price (with currency attribute)
- description
- tags (as multiple <tag> elements)

Text: "The SuperWidget 3000 costs $149.99. An innovative gadget for
       modern homes. Tags: electronics, home, gadget"

Format as XML:
"""

    print(prompt)

    print("\nExpected Output:")
    output = '''<product>
  <name>SuperWidget 3000</name>
  <price currency="USD">149.99</price>
  <description>An innovative gadget for modern homes</description>
  <tags>
    <tag>electronics</tag>
    <tag>home</tag>
    <tag>gadget</tag>
  </tags>
</product>'''
    print(output)

    print("\n" + "=" * 60)
    print("\nXML Template Approach:\n")

    template_prompt = """
Fill in this XML template with information from the text:

<product>
  <name></name>
  <price currency=""></price>
  <description></description>
  <tags>
    <tag></tag>
  </tags>
</product>

Text: "The SuperWidget 3000 costs $149.99. An innovative gadget for
       modern homes. Tags: electronics, home, gadget"
"""

    print(template_prompt)

    print("\nTips for XML Prompting:")
    print("  • Specify root element name")
    print("  • Show example structure or template")
    print("  • Clarify how to handle lists (repeated elements)")
    print("  • Specify if attributes should be used")
    print("  • Request valid, well-formed XML")

xml_output_prompting()
```

## Output Schemas

### Defining Clear Schemas

```python
def defining_output_schemas():
    """Creating clear schemas for structured output."""

    print("Defining Output Schemas:\n")

    print("SCHEMA FOR CONTACT EXTRACTION:\n")

    contact_schema = """{
  "name": "string (full name)",
  "email": "string (email address) or null",
  "phone": "string (phone number) or null",
  "company": "string (company name) or null",
  "role": "string (job title) or null",
  "social": {
    "linkedin": "string (URL) or null",
    "twitter": "string (handle) or null"
  }
}"""

    print(contact_schema)

    print("\n" + "=" * 60)
    print("\nSCHEMA FOR PRODUCT REVIEW ANALYSIS:\n")

    review_schema = """{
  "product_name": "string",
  "overall_sentiment": "string (positive|negative|neutral)",
  "rating": "number (1-5)",
  "pros": ["string", "string", ...],
  "cons": ["string", "string", ...],
  "would_recommend": "boolean",
  "price_mentioned": "number or null",
  "key_features_mentioned": ["string", ...]
}"""

    print(review_schema)

    print("\n" + "=" * 60)
    print("\nBest Practices for Schema Definition:\n")

    practices = [
        'Specify data type for each field',
        'Indicate if field is required or optional',
        'For strings, specify format (email, URL, etc.)',
        'For numbers, specify range if applicable',
        'For enums, list allowed values',
        'Show structure of nested objects',
        'Document what each field represents',
        'Provide example output'
    ]

    for practice in practices:
        print(f"  • {practice}")

defining_output_schemas()
```

### TypeScript-Style Type Definitions

```python
def typescript_type_definitions():
    """Using TypeScript-like types for clarity."""

    print("\n\nTypeScript-Style Type Definitions:\n")

    prompt = """
Extract event information according to this type definition:

interface Event {
  title: string;
  start_date: string; // ISO 8601 format
  end_date?: string; // optional, ISO 8601 format
  location: {
    venue: string;
    address: string;
    city: string;
    country: string;
  };
  attendees: number;
  is_virtual: boolean;
  tags: string[];
}

Text: "Tech Conference 2024 will be held at Convention Center, 123 Main St,
       San Francisco, USA from March 15-17, 2024. Expecting 500 attendees.
       Tags: technology, conference, networking"

Return JSON matching this type:
"""

    print(prompt)

    print("\nExpected Output:")
    output = '''{
  "title": "Tech Conference 2024",
  "start_date": "2024-03-15",
  "end_date": "2024-03-17",
  "location": {
    "venue": "Convention Center",
    "address": "123 Main St",
    "city": "San Francisco",
    "country": "USA"
  },
  "attendees": 500,
  "is_virtual": false,
  "tags": ["technology", "conference", "networking"]
}'''
    print(output)

    print("\nWhy TypeScript-Style Types Work Well:")
    print("  ✓ Familiar to many developers")
    print("  ✓ Clear optional vs required (?)")
    print("  ✓ Inline comments for clarification")
    print("  ✓ Nested object structure obvious")
    print("  ✓ Array types clear")

typescript_type_definitions()
```

## Format Enforcement Techniques

### Strong Format Instructions

```python
def strong_format_instructions():
    """Techniques for enforcing output format."""

    print("Format Enforcement Techniques:\n")

    print("TECHNIQUE 1: Explicit Format Requirement\n")

    technique1 = """
Your response MUST be valid JSON. Do not include any text before or after
the JSON object.

Extract: [task description]

Text: [input text]
"""

    print(technique1)

    print("=" * 60)
    print("\nTECHNIQUE 2: Output Delimiters\n")

    technique2 = """
Extract product information from the text.

Return your response between JSON_START and JSON_END markers:

JSON_START
{your JSON here}
JSON_END

Text: [input text]
"""

    print(technique2)

    print("=" * 60)
    print("\nTECHNIQUE 3: Template Filling\n")

    technique3 = """
Fill in this exact template with extracted information. Do not modify
the structure.

{
  "field1": "VALUE_HERE",
  "field2": "VALUE_HERE",
  "field3": 0
}

Text: [input text]
"""

    print(technique3)

    print("=" * 60)
    print("\nTECHNIQUE 4: Output Prefix\n")

    technique4 = """
Extract information from the text.

Output format: JSON

{"
"""

    print(technique4)
    print('Model completes: "field1": "value", ...')
    print("(Forces model to start with JSON)")

    print("\n" + "=" * 60)
    print("\nTECHNIQUE 5: Few-Shot Examples\n")

    technique5 = """
Extract as JSON (examples):

Text: "Call Bob at 555-1111"
{"name": "Bob", "phone": "555-1111"}

Text: "Email alice@test.com"
{"name": "Alice", "email": "alice@test.com", "phone": null}

Text: [new input]
"""

    print(technique5)

    print("\nEffectiveness Ranking:")
    print("  1. Few-shot examples (best, ~95% format compliance)")
    print("  2. Template filling (~90%)")
    print("  3. Output prefix (~85%)")
    print("  4. Explicit instructions (~80%)")
    print("  5. Delimiters (~75%)")
    print("\n  → Combine multiple techniques for best results")

strong_format_instructions()
```

### Function Calling APIs

```python
def function_calling_apis():
    """Using function calling for guaranteed structure."""

    print("\n\nFunction Calling APIs:\n")

    print("Many LLM APIs now support 'function calling' or 'tools' that")
    print("guarantee structured output by defining the output schema.\n")

    example = '''
# OpenAI Function Calling Example

function_definition = {
    "name": "extract_contact_info",
    "description": "Extract contact information from text",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's name"},
            "email": {"type": "string", "description": "Email address"},
            "phone": {"type": "string", "description": "Phone number"}
        },
        "required": ["name"]
    }
}

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Extract info: Contact Jane at jane@example.com"}
    ],
    functions=[function_definition],
    function_call={"name": "extract_contact_info"}
)

# Response automatically structured:
{
    "name": "Jane",
    "email": "jane@example.com",
    "phone": null
}
'''

    print(example)

    print("\n" + "=" * 60)
    print("\nBenefits of Function Calling:\n")

    benefits = [
        'Guaranteed valid JSON output',
        'Automatic type validation',
        'No parsing errors',
        'Schema enforced by API',
        'Required fields enforced',
        'Better than prompt-based formatting'
    ]

    for benefit in benefits:
        print(f"  ✓ {benefit}")

    print("\nWhen to Use:")
    print("  • Production systems requiring reliability")
    print("  • Complex schemas with validation")
    print("  • When API supports it (OpenAI, Anthropic, etc.)")

function_calling_apis()
```

## Handling Format Errors

### Common Format Errors

```python
def common_format_errors():
    """Common errors in structured output and how to handle them."""

    print("Common Format Errors:\n")

    errors = {
        'Extra text before/after JSON': {
            'example': 'Here is the JSON:\n{"name": "John"}',
            'cause': 'Model adds explanation',
            'fix': 'Stronger formatting instructions, output prefix'
        },
        'Invalid JSON syntax': {
            'example': '{"name": John}  // Missing quotes',
            'cause': 'Model makes syntax error',
            'fix': 'Few-shot examples, validation prompt'
        },
        'Missing required fields': {
            'example': '{"name": "John"}  // missing email',
            'cause': 'Model doesn\'t extract all fields',
            'fix': 'Specify required fields, use template'
        },
        'Wrong data type': {
            'example': '{"age": "30"}  // Should be number',
            'cause': 'Model uses wrong type',
            'fix': 'Specify types in schema, examples'
        },
        'Nested JSON as string': {
            'example': '{"data": "{\\"nested\\": true}"}',
            'cause': 'Model escapes nested JSON',
            'fix': 'Show proper nesting in examples'
        },
        'Array vs single value': {
            'example': '{"tags": "tag1"}  // Should be ["tag1"]',
            'cause': 'Ambiguous array handling',
            'fix': 'Specify array format in schema'
        }
    }

    for error_type, info in errors.items():
        print(f"{error_type}:")
        print(f"  Example: {info['example']}")
        print(f"  Cause: {info['cause']}")
        print(f"  Fix: {info['fix']}")
        print()

common_format_errors()
```

### Retry with Formatting Correction

```python
def retry_with_correction():
    """Handling format errors with retry and correction."""

    print("\n\nRetry Strategy for Format Errors:\n")

    code = '''
import json

def extract_with_retry(prompt, max_retries=3):
    """
    Try to get valid JSON, retry with correction if invalid.
    """

    for attempt in range(max_retries):
        response = llm_call(prompt)

        try:
            # Try to parse JSON
            data = json.loads(response)
            return data  # Success!

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                # Build correction prompt
                correction_prompt = f"""
The previous response was not valid JSON. Error: {str(e)}

Invalid response:
{response}

Please provide the same information as valid JSON only, with no additional text.
"""
                prompt = correction_prompt
            else:
                # Final attempt failed
                raise ValueError(f"Could not get valid JSON after {max_retries} attempts")

    return None

# Usage
prompt = "Extract name and age from: 'John is 30 years old' as JSON"
data = extract_with_retry(prompt)
print(data)  # {'name': 'John', 'age': 30}
'''

    print(code)

    print("\nFallback Strategies:")
    print("  1. Regex extraction: Extract JSON from text")
    print("  2. Partial parsing: Get what you can")
    print("  3. Default values: Fill missing fields")
    print("  4. Human review: Flag for manual check")

retry_with_correction()
```

## Parsing Strategies

### Robust JSON Parsing

````python
def robust_json_parsing():
    """Strategies for parsing JSON from LLM output."""

    print("Robust JSON Parsing Strategies:\n")

    code = '''
import json
import re

def extract_json_from_text(text):
    """
    Extract JSON from text that may have extra content.
    """

    # Strategy 1: Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON between markers
    markers = [
        (r'JSON_START\s*(.*?)\s*JSON_END', re.DOTALL),
        (r'```json\s*(.*?)\s*```', re.DOTALL),
        (r'```\s*(.*?)\s*```', re.DOTALL),
    ]

    for pattern, flags in markers:
        match = re.search(pattern, text, flags)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Strategy 3: Find JSON object/array
    # Look for {...} or [...]
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Objects
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Arrays
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Strategy 4: Try fixing common issues
    fixes = [
        (r"'", '"'),  # Single quotes to double
        (r'(\w+):', r'"\1":'),  # Unquoted keys
    ]

    fixed = text
    for pattern, replacement in fixes:
        fixed = re.sub(pattern, replacement, fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise ValueError("Could not extract valid JSON from text")

# Example usage
text = """
Here is the extracted information:
```json
{
  "name": "John Doe",
  "age": 30
}
````

Hope this helps!
"""

data = extract_json_from_text(text)
print(data) # {'name': 'John Doe', 'age': 30}
'''

    print(code)

robust_json_parsing()

````

### Validation and Error Handling

```python
def validation_and_error_handling():
    """Validating extracted structured data."""

    print("\n\nValidation and Error Handling:\n")

    code = '''
from typing import Optional, List
from pydantic import BaseModel, Field, validator

class ContactInfo(BaseModel):
    """Validated contact information model."""

    name: str = Field(..., min_length=1)
    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=150)
    tags: List[str] = []

    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v

    @validator('phone')
    def validate_phone(cls, v):
        if v:
            # Remove common formatting
            digits = ''.join(filter(str.isdigit, v))
            if len(digits) < 10:
                raise ValueError('Phone number too short')
        return v

# Usage
def parse_and_validate(json_text: str) -> Optional[ContactInfo]:
    """Parse JSON and validate against schema."""
    try:
        data = json.loads(json_text)
        contact = ContactInfo(**data)
        return contact
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None

# Example
json_output = '{"name": "John", "email": "john@example.com", "age": 30}'
contact = parse_and_validate(json_output)

if contact:
    print(f"Valid contact: {contact.name}, {contact.email}")
else:
    print("Failed to parse/validate")
'''

    print(code)

    print("\nValidation Checklist:")
    print("  ✓ Required fields present")
    print("  ✓ Correct data types")
    print("  ✓ Values within valid ranges")
    print("  ✓ Format validation (email, URL, etc.)")
    print("  ✓ Enum values are allowed")
    print("  ✓ Array lengths acceptable")

validation_and_error_handling()
````

## Complex Nested Structures

### Extracting Nested Data

```python
def nested_structure_extraction():
    """Extracting complex nested structures."""

    print("Extracting Complex Nested Structures:\n")

    prompt = """
Extract company information from the text as JSON with this structure:

{
  "company": {
    "name": "string",
    "founded": "number",
    "headquarters": {
      "city": "string",
      "country": "string"
    },
    "products": [
      {
        "name": "string",
        "category": "string",
        "launched": "number"
      }
    ],
    "key_people": [
      {
        "name": "string",
        "role": "string"
      }
    ]
  }
}

Text: "TechCorp was founded in 2010 in San Francisco, USA. CEO Jane Smith
       leads the company. They launched CloudApp (cloud software) in 2015
       and DataTool (analytics) in 2018."

Return valid JSON matching this structure:
"""

    print(prompt)

    print("\nExpected Output:")
    output = '''{
  "company": {
    "name": "TechCorp",
    "founded": 2010,
    "headquarters": {
      "city": "San Francisco",
      "country": "USA"
    },
    "products": [
      {
        "name": "CloudApp",
        "category": "cloud software",
        "launched": 2015
      },
      {
        "name": "DataTool",
        "category": "analytics",
        "launched": 2018
      }
    ],
    "key_people": [
      {
        "name": "Jane Smith",
        "role": "CEO"
      }
    ]
  }
}'''
    print(output)

    print("\nTips for Nested Structures:")
    print("  • Show complete structure in prompt")
    print("  • Use indentation to show nesting")
    print("  • Provide examples of nested data")
    print("  • Consider breaking into multiple extractions for very complex structures")

nested_structure_extraction()
```

## Best Practices

```python
def structured_output_best_practices():
    """Best practices for structured output."""

    print("Structured Output Best Practices:\n")

    practices = {
        'Be explicit': 'State format clearly (JSON, XML, etc.)',
        'Provide schema': 'Show exact structure expected',
        'Use examples': 'Few-shot examples with correct format',
        'Specify types': 'Indicate data types for each field',
        'Handle missing data': 'Define strategy (null, omit, empty)',
        'Add delimiters': 'Use markers to isolate structured output',
        'Validate output': 'Parse and validate before using',
        'Retry on error': 'Implement retry logic for format errors',
        'Use templates': 'Provide template to fill in',
        'Output prefix': 'Start model\'s response with opening brace',
        'Function calling': 'Use API features when available',
        'Keep it simple': 'Simpler schemas are more reliable'
    }

    for practice, description in practices.items():
        print(f"✓ {practice}: {description}")

    print("\n" + "=" * 60)
    print("\nCommon Mistakes to Avoid:\n")

    mistakes = [
        'Assuming model will infer format from context',
        'Not providing examples for complex structures',
        'Forgetting to handle optional/missing fields',
        'No validation of output before using',
        'Overly complex nested structures',
        'Inconsistent formatting across examples',
        'Not specifying data types clearly'
    ]

    for mistake in mistakes:
        print(f"✗ {mistake}")

    print("\n" + "=" * 60)
    print("\nReliability Hierarchy (Best to Worst):\n")

    print("1. Function/Tool Calling API (~98% success)")
    print("2. Few-shot examples + template (~95%)")
    print("3. JSON schema + examples (~90%)")
    print("4. Template to fill in (~85%)")
    print("5. Clear format instructions (~80%)")
    print("6. Just asking for JSON (~70%)")

structured_output_best_practices()
```

## Summary

**Key Concepts**:

1. **Structured output** returns machine-readable formats (JSON, XML) instead of free text
2. **Essential for integration** - enables programmatic processing of LLM outputs
3. **Three main approaches**: Explicit instructions, templates, few-shot examples
4. **JSON preferred** over XML for most cases (simpler, standard, easier to parse)
5. **Schema specification** ensures correct types, required fields, validation
6. **Format enforcement** requires explicit instructions, examples, or API features
7. **Robust parsing** handles common errors (extra text, syntax issues)
8. **Validation** checks output against expected schema before use

**Format Reliability**:

| Technique                 | Success Rate | Setup Effort      |
| ------------------------- | ------------ | ----------------- |
| Function/Tool calling API | ~98%         | Low (API feature) |
| Few-shot + template       | ~95%         | Medium            |
| Schema + examples         | ~90%         | Medium            |
| Template filling          | ~85%         | Low               |
| Format instructions       | ~80%         | Low               |
| Basic request             | ~70%         | Very low          |

**JSON vs XML**:

```
JSON (preferred for most cases):
  ✓ Simpler, less verbose
  ✓ Native data types
  ✓ Easy parsing
  ✓ Modern standard

XML (use when):
  • Legacy system integration
  • Attributes needed
  • Mixed content
  • Domain standard (SOAP, etc.)
```

**Three Approaches**:

```
1. Explicit Instructions:
   "Return as JSON with fields: name, email, phone"

2. Template:
   Fill in: {"name": "", "email": "", "phone": ""}

3. Few-Shot:
   Text: "..." → {"name": "...", "email": "..."}
   Text: "..." → {"name": "...", "email": "..."}
   Text: "..." →
```

**Best Practices**:

✓ **Provide complete schema** - show exact structure expected  
✓ **Use few-shot examples** - demonstrate correct format  
✓ **Specify data types** - number vs string, required vs optional  
✓ **Handle missing data** - decide on null, omit, or empty strategy  
✓ **Validate output** - parse and check before using  
✓ **Implement retry** - handle format errors gracefully  
✓ **Use API features** - function calling when available  
✓ **Keep schemas simple** - complexity reduces reliability

**Common Errors**:

- Extra text before/after JSON → Stronger instructions, output prefix
- Invalid JSON syntax → Few-shot examples, validation
- Missing required fields → Explicit requirements, template
- Wrong data types → Specify types in schema
- Nested JSON as string → Show proper nesting

**Handling Errors**:

```python
1. Try parsing
2. If fails, extract JSON from text (markers, regex)
3. If still fails, try fixing common issues
4. If still fails, retry with correction prompt
5. Final fallback: partial parsing or default values
```

**Schema Specification Formats**:

- **JSON Schema**: Standard, comprehensive validation
- **TypeScript types**: Developer-friendly, clear syntax
- **Template**: Simple, fill-in-the-blanks
- **Examples**: Show don't tell

**Validation Steps**:

1. Parse JSON/XML
2. Check required fields present
3. Verify data types correct
4. Validate ranges/enums
5. Check nested structure
6. Format validation (email, URL, etc.)

## Next Steps

- Apply to [Prompt Optimization](prompt-optimization.md) for testing format reliability
- Combine with [Chain-of-Thought](cot-prompting.md) for complex extraction
- Study [Advanced Patterns](advanced-patterns.md) for multi-step extraction
- Explore [Data Ingestion](../data-engineering-lab/data-ingestion/) for processing structured output
- Review [Evaluation](../evaluation/) for measuring extraction accuracy
- Learn [Tool Use](../agentic-ai-lab/tool-use/) for calling functions with extracted data
