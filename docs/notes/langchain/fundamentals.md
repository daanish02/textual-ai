# LangChain Fundamentals

## Table of Contents

- [Introduction](#introduction)
- [Chat Models and LLMs](#chat-models-and-llms)
  - [LLM vs ChatModel](#llm-vs-chatmodel)
  - [Multi-Provider Model Initialization](#multi-provider-model-initialization)
  - [Universal Model Initialization](#universal-model-initialization)
  - [Temperature and Model Configuration](#temperature-and-model-configuration)
- [Prompt Templates](#prompt-templates)
  - [PromptTemplate: String-Based Prompts](#prompttemplate-string-based-prompts)
  - [ChatPromptTemplate: Structured Messages](#chatprompttemplate-structured-messages)
  - [Message Types](#message-types)
  - [MessagesPlaceholder: Dynamic History](#messagesplaceholder-dynamic-history)
  - [Template Persistence](#template-persistence)
- [Embeddings](#embeddings)
  - [Query vs Document Embeddings](#query-vs-document-embeddings)
  - [Dimensions and Configuration](#dimensions-and-configuration)
- [Output Parsing](#output-parsing)
  - [StrOutputParser: Basic String Output](#stroutputparser-basic-string-output)
  - [JsonOutputParser: Structured JSON](#jsonoutputparser-structured-json)
  - [PydanticOutputParser: Schema Validation](#pydanticoutputparser-schema-validation)
  - [Structured Output: Native Model Support](#structured-output-native-model-support)
  - [Model Compatibility](#model-compatibility)
- [Conversation Memory](#conversation-memory)
  - [Why Memory Matters](#why-memory-matters)
  - [Message History Pattern](#message-history-pattern)
  - [Memory Integration with Chains](#memory-integration-with-chains)
  - [When to Use Memory](#when-to-use-memory)
- [Component Comparison](#component-comparison)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

LangChain provides the building blocks for constructing LLM applications. Before orchestrating complex chains or retrieval pipelines, you need to understand the fundamental components: **models** (the LLMs that generate responses), **prompts** (how you structure input), **embeddings** (converting text to vectors), and **output parsers** (extracting structured data from responses).

These components are framework-specific implementations of universal concepts from the main lab. Every LangChain application uses these building blocks -- understanding them thoroughly enables you to compose more complex patterns confidently.

This document covers the core primitives you'll use in every LangChain application.

## Chat Models and LLMs

### LLM vs ChatModel

LangChain distinguishes between two interfaces for language models:

**LLM** - Text-in, text-out completion models:

- Takes a string prompt, returns a string completion
- Older interface designed for completion-style models (GPT-3, text-davinci-003)
- Returns plain string output
- Less structured, primarily for simple text generation

**ChatModel** - Message-based conversational models:

- Takes a list of messages, returns a message response
- Modern interface for chat-optimized models (GPT-4, Claude, Gemini)
- Returns rich message objects with content + metadata
- Supports system/user/assistant message roles
- Better for conversational and instruction-following tasks

```python
from langchain_openai import OpenAI, ChatOpenAI

# LLM - simple text completion
llm = OpenAI()
response = llm.invoke("Why do parrots talk?")
print(response)  # String output directly

# ChatModel - structured message-based
chat_model = ChatOpenAI(model="gpt-4o-mini")
response = chat_model.invoke("Why do parrots talk?")
print(response.content)  # Access .content attribute
print(response.response_metadata)  # Additional metadata available
```

**When to use each:**

- **Use ChatModel** for almost everything -- modern models expect message-based interaction
- **Use LLM** only for legacy compatibility or when working with pure completion models

### Multi-Provider Model Initialization

LangChain supports multiple LLM providers with a consistent interface. Each provider has its own package and model class:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# OpenAI (GPT models)
openai_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Anthropic (Claude models)
anthropic_model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.9
)

# Google (Gemini models)
google_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

# All use the same .invoke() interface
response = openai_model.invoke("Your prompt here")
```

Each provider requires its API key in environment variables:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`

**Provider selection considerations:**

- **OpenAI** - Most popular, strong general capabilities, function calling
- **Anthropic** - Excellent at following instructions, longer context windows, constitutional AI
- **Google** - Cost-effective, fast inference, good for high-volume applications

### Universal Model Initialization

For applications that need provider flexibility, use `init_chat_model()`:

```python
from langchain.chat_models import init_chat_model

# Initialize model by name alone - automatically selects provider
model = init_chat_model(model="gpt-4o-mini")
model = init_chat_model(model="claude-3-opus-20240229")
model = init_chat_model(model="gemini-pro")

# Works across providers with consistent interface
response = model.invoke("Why do parrots talk?")
```

This abstraction is useful when:

- Building provider-agnostic applications
- Testing across multiple models
- Allowing runtime model selection via configuration
- Migrating between providers without code changes

### Temperature and Model Configuration

Temperature controls randomness in model outputs:

```python
# Low temperature - deterministic, focused
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
# Good for: factual QA, classification, structured extraction

# Medium temperature - balanced creativity
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# Good for: general conversation, content generation

# High temperature - creative, diverse
model = ChatOpenAI(model="gpt-4o-mini", temperature=1.5)
# Good for: creative writing, brainstorming, multiple perspectives
```

**Temperature guidelines:**

- **0.0-0.3** - Deterministic tasks (classification, extraction, factual QA)
- **0.4-0.8** - Balanced tasks (conversation, content generation)
- **0.9-2.0** - Creative tasks (storytelling, ideation, diverse responses)

Note: Some models may not support temperatures above 1.0 -- check provider documentation.

## Prompt Templates

Prompt templates separate prompt structure from variable content, enabling reusability and composition.

### PromptTemplate: String-Based Prompts

Basic string templating with variable substitution:

```python
from langchain_core.prompts import PromptTemplate

# Define template with input variables
template = PromptTemplate(
    template="Greet me in these languages: {languages}. The name is {name}",
    input_variables=["name", "languages"]
)

# Invoke with values
prompt = template.invoke({
    "name": "danish",
    "languages": "english, hindi, urdu, arabic, french"
})

# Use with model
result = model.invoke(prompt)
```

**Benefits of templates:**

- **Reusability** - Define once, use with different inputs
- **Maintainability** - Update prompt in one place
- **Composition** - Chain templates together
- **Testing** - Test prompts independently from application logic

### ChatPromptTemplate: Structured Messages

For chat models, use `ChatPromptTemplate` to structure system/user/assistant messages:

```python
from langchain_core.prompts import ChatPromptTemplate

# Define chat-style template with roles
chat_template = ChatPromptTemplate([
    ("system", "You are {persona}"),
    ("human", "{topic}")
])

# Invoke creates properly formatted messages
prompt = chat_template.invoke({
    "persona": "a helpful physics tutor",
    "topic": "Explain quantum entanglement"
})

result = model.invoke(prompt)
```

**Role types:**

- **system** - Sets model behavior and persona
- **human** - User input/questions
- **ai** - Assistant responses (for few-shot examples or history)

**Chat vs String templates:**

- Use `ChatPromptTemplate` for conversational models (GPT-4, Claude)
- Use `PromptTemplate` for simple completion tasks
- Chat templates provide better control over message structure

### Message Types

For more explicit control, construct messages directly:

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about LangChain"),
]

result = model.invoke(messages)

# Append assistant response to maintain history
messages.append(AIMessage(content=result.content))
```

**Use explicit messages when:**

- Building conversational applications with history
- Fine-tuning message order and structure
- Implementing few-shot examples with AI responses
- Need programmatic control over message construction

### MessagesPlaceholder: Dynamic History

For chat applications with variable-length history, use `MessagesPlaceholder`:

```python
from langchain_core.prompts import MessagesPlaceholder

# Define template with placeholder for dynamic history
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# Load chat history from file, database, etc.
chat_history = [
    HumanMessage(content="I ordered item #12345"),
    AIMessage(content="I can help you with that order"),
    HumanMessage(content="When will it ship?"),
    AIMessage(content="It shipped yesterday via FedEx")
]

# Inject history into template
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

result = model.invoke(prompt)
```

**MessagesPlaceholder benefits:**

- Variable-length history (0 to many messages)
- Clean separation of template structure from history content
- Easy integration with chat memory systems
- Type-safe message handling

### Template Persistence

Save and load templates for versioning and sharing:

```python
# Save template to file
template.save("greeting-template.json")

# Load template elsewhere
loaded_template = PromptTemplate.load("greeting-template.json")

# Use in chain
chain = loaded_template | model
result = chain.invoke({"name": "ahmed", "languages": "russian, german"})
```

**Use cases for persistence:**

- Version control for prompts
- Sharing templates across team
- A/B testing different prompt versions
- Prompt libraries and registries

## Embeddings

Embeddings convert text into numerical vectors that capture semantic meaning. Essential for semantic search, RAG, and similarity tasks.

### Query vs Document Embeddings

LangChain distinguishes between two embedding use cases:

**Query Embeddings** - Single text → vector:

```python
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

# Embed a single query
query_vector = embedding_model.embed_query("Why do parrots talk?")
# Returns: [0.123, -0.456, 0.789, ...] (1024 dimensions)
```

**Document Embeddings** - Batch processing:

```python
documents = [
    "Why are flamingos pink?",
    "Why are elephants so big?",
    "Why are zebras not domestic?"
]

# Embed multiple documents efficiently
doc_vectors = embedding_model.embed_documents(documents)
# Returns: [[...], [...], [...]] (list of vectors)
```

**Why the distinction?**

- **API optimization** - Batch processing for documents is more efficient
- **Different processing** - Some models optimize queries vs documents differently
- **Usage patterns** - Queries are typically one-at-a-time, documents are bulk-processed

**Typical workflow:**

1. **Index time** - Use `embed_documents()` to vectorize your corpus
2. **Search time** - Use `embed_query()` to vectorize user questions
3. **Compare** - Compute similarity between query vector and document vectors

### Dimensions and Configuration

Configure embedding dimensionality based on your needs:

```python
# High-dimensional embeddings (better quality, larger storage)
embedding_large = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072  # Maximum for this model
)

# Reduced-dimensional embeddings (faster, smaller storage)
embedding_small = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=256  # Reduced size
)
```

**Dimension trade-offs:**

- **Higher dimensions** - Better semantic capture, higher accuracy, larger storage/memory
- **Lower dimensions** - Faster search, smaller storage, slightly lower accuracy
- **Typical values** - 384 (Sentence-BERT), 768 (BERT), 1536 (OpenAI ada-002), 1024-3072 (OpenAI v3)

**Choose dimensions based on:**

- Dataset size (larger datasets benefit from lower dimensions)
- Latency requirements (lower dimensions = faster search)
- Accuracy requirements (higher dimensions = better semantic capture)
- Storage constraints (each dimension adds storage overhead)

## Output Parsing

Output parsers extract structured data from LLM text responses. They transform raw text into usable formats: strings, JSON objects, or validated Pydantic models.

### StrOutputParser: Basic String Output

The simplest parser -- extracts content string from model response:

```python
from langchain_core.output_parsers import StrOutputParser

template1 = PromptTemplate(
    template="Give a detailed report on {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Summarize this report: {report}",
    input_variables=["report"]
)

parser = StrOutputParser()

# Chain: template → model → parse string → next template → model → parse
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "quantum tunneling"})
# Returns: Plain string summary
```

**Use StrOutputParser when:**

- Output is free-form text
- No structure needed
- Chaining LLM calls (feed output as next input)

### JsonOutputParser: Structured JSON

Instructs model to output JSON and parses it:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give 5 facts about {topic}\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain = template | model | parser
result = chain.invoke({"topic": "wolves"})
# Returns: {"facts": ["...", "...", ...]}
```

**How it works:**

1. `parser.get_format_instructions()` generates instructions like "Output valid JSON with this structure"
2. Instructions are injected into prompt
3. Model generates JSON text
4. Parser validates and parses into Python dict

**Use JsonOutputParser when:**

- Need structured output without strict schema
- Flexible field structure
- Rapid prototyping (no Pydantic model needed)

**Limitations:**

- No type validation (everything is strings until you convert)
- Model might still generate invalid JSON
- No IDE autocomplete for fields

### PydanticOutputParser: Schema Validation

Enforces strict schema using Pydantic models:

```python
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# Define your schema
class Person(BaseModel):
    name: str = Field(..., description="Name of character")
    age: int = Field(..., description="Age of character")
    universe: str = Field(..., description="Which universe they exist in")
    parents: List[str] = Field(..., description="Parents' names")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate a fictional character\n{format_instruction}",
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain = template | model | parser
result = chain.invoke({})
# Returns: Person(name="...", age=25, universe="...", parents=[...])
```

**Pydantic benefits:**

- **Type safety** - Validated types (int, str, List, etc.)
- **Documentation** - Field descriptions guide the model
- **IDE support** - Autocomplete for result fields
- **Validation** - Automatic validation and error messages
- **Defaults** - Optional fields with default values

**Use PydanticOutputParser when:**

- Production applications requiring reliability
- Complex nested structures
- Type safety is important
- Integrating with typed Python codebases

### Structured Output: Native Model Support

Some models natively support structured output without prompt engineering:

```python
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Review(BaseModel):
    key_themes: List[str] = Field(
        ..., description="Key themes discussed"
    )
    summary: str = Field(..., description="Brief summary")
    sentiment: Literal["positive", "negative"] = Field(
        ..., description="Overall sentiment"
    )
    pros: Optional[List[str]] = Field(None, description="Pros mentioned")
    cons: Optional[List[str]] = Field(None, description="Cons mentioned")
    name: Optional[str] = Field(None, description="Reviewer name")

# Use with_structured_output() instead of parser
structured_model = model.with_structured_output(Review)

review_text = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and it's an absolute
powerhouse! The processor is lightning fast... [full review]
"""

result = structured_model.invoke(review_text)
# Returns: Review object with validated fields
print(result.key_themes)  # ['performance', 'camera', 'battery']
print(result.sentiment)   # 'positive'
```

**Structured output vs PydanticOutputParser:**

- **Structured output** - Native model support, more reliable, cleaner prompts
- **PydanticOutputParser** - Works with any model, requires prompt instructions

**Prefer structured output when available** -- cleaner, more reliable, no prompt engineering needed.

### Model Compatibility

**Models with native structured output support:**

- OpenAI: GPT-4o, GPT-4o-mini, GPT-4 Turbo (with function calling)
- Anthropic: Claude 3+ models (via tool use API)
- Google: Gemini Pro (via function declarations)

**Why some models can't produce structured output:**

- **Training** - Not trained on structured output tasks
- **Architecture** - Lack necessary function-calling or tool-use capabilities
- **Context** - Can't reliably maintain JSON syntax over long outputs
- **Older models** - GPT-3.5 and earlier struggle with consistent structure

**Fallback strategy:**

```python
# Try structured output first
try:
    structured_model = model.with_structured_output(Schema)
    result = structured_model.invoke(prompt)
except NotImplementedError:
    # Fall back to PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=Schema)
    template = create_template_with_parser(parser)
    chain = template | model | parser
    result = chain.invoke(prompt)
```

## Conversation Memory

### Why Memory Matters

LLMs are stateless -- they don't remember previous interactions. For multi-turn conversations, you must explicitly manage conversation history.

**Without memory:**

```python
user: "My name is Danish"
bot: "Nice to meet you, Danish!"

user: "What's my name?"
bot: "I don't know your name."  # ❌ No memory
```

**With memory:**

```python
user: "My name is Danish"
bot: "Nice to meet you, Danish!"

user: "What's my name?"
bot: "Your name is Danish!"  # ✓ Remembers context
```

**Memory enables:**

- Multi-turn conversations
- Context awareness across messages
- Personalized responses
- Follow-up questions
- Anaphora resolution ("it", "that", "the previous one")

### Message History Pattern

The fundamental pattern using `MessagesPlaceholder` (covered in Prompts section):

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Template with history placeholder
template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

model = ChatOpenAI()
chain = template | model

# Conversation state (maintained in your application)
chat_history = []

# Turn 1
response = chain.invoke({
    "chat_history": chat_history,
    "input": "My name is Danish"
})
chat_history.extend([
    HumanMessage(content="My name is Danish"),
    AIMessage(content=response.content)
])

# Turn 2 - model sees previous context
response = chain.invoke({
    "chat_history": chat_history,
    "input": "What's my name?"
})
# Response: "Your name is Danish!"
```

**Key pattern:**

1. **Store history** - Maintain list of messages (HumanMessage, AIMessage)
2. **Pass to template** - Inject via MessagesPlaceholder
3. **Append new turns** - Add user input and model response after each turn
4. **Persist if needed** - Save to database for long-term memory

**History structure:**

```python
chat_history = [
    HumanMessage(content="Hello"),
    AIMessage(content="Hi! How can I help?"),
    HumanMessage(content="Tell me about RAG"),
    AIMessage(content="RAG stands for..."),
    # ... continues
]
```

### Memory Integration with Chains

**RAG with conversation memory:**

```python
from langchain_core.runnables import RunnablePassthrough

template = ChatPromptTemplate([
    ("system", "Answer based on this context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Chain with retrieval + memory
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "chat_history": lambda x: x.get("chat_history", [])
    }
    | template
    | model
)

# Use with history
result = chain.invoke({
    "question": "What is machine learning?",
    "chat_history": previous_messages
})
```

**Stateful application pattern:**

```python
class ConversationApp:
    def __init__(self):
        self.sessions = {}  # session_id -> chat_history

    def chat(self, session_id: str, user_input: str):
        # Get or create session history
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        history = self.sessions[session_id]

        # Invoke chain with history
        response = chain.invoke({
            "input": user_input,
            "chat_history": history
        })

        # Update history
        history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response.content)
        ])

        return response.content

# Usage
app = ConversationApp()
app.chat("user123", "My name is Danish")
app.chat("user123", "What's my name?")  # Remembers context
```

**Persistence pattern:**

```python
import json

# Save to file/database
def save_history(session_id, history):
    serialized = [
        {"role": "human" if isinstance(m, HumanMessage) else "ai",
         "content": m.content}
        for m in history
    ]
    with open(f"sessions/{session_id}.json", "w") as f:
        json.dump(serialized, f)

# Load from file/database
def load_history(session_id):
    try:
        with open(f"sessions/{session_id}.json", "r") as f:
            data = json.load(f)
        return [
            HumanMessage(content=m["content"]) if m["role"] == "human"
            else AIMessage(content=m["content"])
            for m in data
        ]
    except FileNotFoundError:
        return []
```

**Context window management:**

```python
# Keep only recent messages to fit context window
def trim_history(history, max_messages=10):
    """Keep only last N messages."""
    return history[-max_messages:]

# Use in chain
history = trim_history(chat_history, max_messages=20)
response = chain.invoke({"input": user_input, "chat_history": history})
```

**Summarization for long conversations:**

```python
def summarize_and_trim(history, model, threshold=20):
    """Summarize old messages when history gets long."""
    if len(history) < threshold:
        return history

    # Summarize older messages
    old_messages = history[:-10]  # All but last 10
    recent_messages = history[-10:]  # Last 10

    summary_prompt = f"Summarize this conversation: {old_messages}"
    summary = model.invoke(summary_prompt)

    return [
        SystemMessage(content=f"Previous conversation summary: {summary}"),
        *recent_messages
    ]
```

### When to Use Memory

**Use conversation memory for:**

- Chatbots and virtual assistants
- Multi-turn QA sessions
- Personalized interactions
- Follow-up questions ("Tell me more", "What about the second one?")
- Context-dependent queries ("Explain that in simpler terms")

**Memory patterns by use case:**

| Use Case           | Memory Type        | Retention              |
| ------------------ | ------------------ | ---------------------- |
| Customer support   | Full history       | Session-based (hours)  |
| Personal assistant | Trimmed history    | Long-term (days/weeks) |
| Educational tutor  | Summarized history | Course duration        |
| Code assistant     | Recent history     | Task-based             |
| General chatbot    | Sliding window     | Session-based          |

**When NOT to use memory:**

- One-shot queries (classification, single QA)
- Each query is independent
- Privacy concerns (don't retain conversation)
- Stateless API requirements

**Best practices:**

- Always trim/summarize for long conversations (avoid context overflow)
- Implement session management (separate users/conversations)
- Handle edge cases (empty history, corrupted state)
- Persist to database for production (not just in-memory)
- Add timestamps for time-aware context ("earlier you said...")
- Consider privacy/compliance (GDPR, data retention policies)

## Component Comparison

Quick reference for choosing the right component:

| Component                  | Use When                                | Output Type    | Reliability |
| -------------------------- | --------------------------------------- | -------------- | ----------- |
| **LLM**                    | Legacy models only                      | String         | -           |
| **ChatModel**              | Modern chat models (always prefer)      | Message        | High        |
| **PromptTemplate**         | Simple string prompts                   | String         | -           |
| **ChatPromptTemplate**     | Structured conversations                | Messages       | -           |
| **MessagesPlaceholder**    | Dynamic chat history                    | Messages       | -           |
| **embed_query**            | Search queries (single)                 | Vector         | -           |
| **embed_documents**        | Corpus indexing (batch)                 | List[Vector]   | -           |
| **StrOutputParser**        | Free-form text                          | String         | Medium      |
| **JsonOutputParser**       | Flexible structured data                | Dict           | Medium      |
| **PydanticOutputParser**   | Validated structures (any model)        | Pydantic       | Medium-High |
| **with_structured_output** | Validated structures (supported models) | Pydantic       | High        |
| **Conversation Memory**    | Multi-turn conversations                | List[Messages] | High        |

## Summary

LangChain fundamentals provide the building blocks for every LLM application:

**Models** - Use `ChatModel` for modern applications; support multiple providers (OpenAI, Anthropic, Google) with consistent interfaces; configure temperature based on task requirements.

**Prompts** - Use `PromptTemplate` for simple strings, `ChatPromptTemplate` for conversations; leverage `MessagesPlaceholder` for dynamic history; persist templates for versioning and reuse.

**Embeddings** - Use `embed_query()` for search-time queries, `embed_documents()` for index-time bulk processing; choose dimensions balancing accuracy, speed, and storage.

**Output Parsing** - Progress from `StrOutputParser` (free text) → `JsonOutputParser` (flexible structure) → `PydanticOutputParser` (validated schema); use `with_structured_output()` when native model support available.

**Conversation Memory** - Maintain chat history with `MessagesPlaceholder`; append HumanMessage and AIMessage after each turn; implement session management and persistence; trim or summarize for long conversations; essential for multi-turn chatbots and context-aware interactions.

**Key principles:**

- Prefer ChatModel over LLM for modern applications
- Use chat templates for better message structure
- Distinguish query vs document embeddings
- Choose parsers based on reliability requirements
- Leverage native structured output when possible
- Manage conversation state explicitly for multi-turn interactions

These primitives compose into chains, retrieval systems, and conversational applications -- covered in subsequent documents.

## Next Steps

- **[Data and Retrieval](data-and-retrieval.md)** - Load documents, chunk text, build vector stores, and implement retrieval strategies
- **[Orchestration](orchestration.md)** - Compose these components into chains and workflows using LCEL and Runnables

**Related concepts:**

- [Prompt Engineering](../../prompt-engineering/) - Framework-agnostic prompting strategies
- [Embeddings](../../embeddings/) - Deep dive into embedding spaces and theory
- [LLM Concepts](../../llm-concepts/) - Understanding LLM capabilities and limitations
