# Orchestration

## Table of Contents

- [Introduction](#introduction)
- [LangChain Expression Language (LCEL)](#langchain-expression-language-lcel)
  - [The Pipe Operator](#the-pipe-operator)
  - [Why LCEL Matters](#why-lcel-matters)
- [Core Runnables](#core-runnables)
  - [RunnableSequence: Sequential Flow](#runnablesequence-sequential-flow)
  - [RunnableParallel: Concurrent Execution](#runnableparallel-concurrent-execution)
  - [RunnablePassthrough: Data Flow](#runnablepassthrough-data-flow)
  - [RunnableLambda: Custom Functions](#runnablelambda-custom-functions)
  - [RunnableBranch: Conditional Routing](#runnablebranch-conditional-routing)
- [Chain Patterns](#chain-patterns)
  - [Sequential Chains](#sequential-chains)
  - [Parallel Chains](#parallel-chains)
  - [Conditional Chains](#conditional-chains)
  - [Complex Compositions](#complex-compositions)
- [Chain Visualization](#chain-visualization)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Orchestration is about composing LLM calls into workflows. Individual components (models, prompts, retrievers) are powerful alone, but real applications require **chaining** them together: feed one output as the next input, run steps in parallel, route through conditional logic, or combine all three.

LangChain provides two abstraction levels:

1. **Runnables** - Low-level primitives for composition (sequence, parallel, branch, etc.)
2. **Chains** - Common patterns built on Runnables (sequential workflows, parallel execution, conditional routing)

This document covers both: understanding the building blocks (Runnables) enables you to construct sophisticated workflows (Chains) for any use case.

## LangChain Expression Language (LCEL)

### The Pipe Operator

LCEL uses the `|` (pipe) operator to chain components together:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Traditional approach (verbose)
prompt = PromptTemplate(template="Tell me a joke about {topic}", input_variables=["topic"])
model = ChatOpenAI()
parser = StrOutputParser()

# Invoke step by step
prompt_value = prompt.invoke({"topic": "AI"})
model_output = model.invoke(prompt_value)
final_output = parser.invoke(model_output)

# LCEL approach (elegant)
chain = prompt | model | parser
result = chain.invoke({"topic": "AI"})
```

**Flow visualization:**

```
Input → Prompt → Model → Parser → Output
 {"topic": "AI"}
         ↓
      "Tell me a joke about AI"
                 ↓
            AIMessage(content="Why did...")
                        ↓
                   "Why did..." (string)
```

**How the pipe works:**

- Each component implements the **Runnable interface**
- `|` creates a **RunnableSequence** automatically
- Data flows left to right through the chain
- Each component's output becomes the next component's input

### Why LCEL Matters

**Readability:**

```python
# Without LCEL - imperative, verbose
result1 = prompt.invoke(input)
result2 = model.invoke(result1)
result3 = parser.invoke(result2)

# With LCEL - declarative, clear
result = (prompt | model | parser).invoke(input)
```

**Composability:**

```python
# Build chains from chains
summary_chain = prompt1 | model | parser
analysis_chain = prompt2 | model | parser

# Compose into larger workflow
workflow = summary_chain | analysis_chain
```

**Streaming support:**

```python
# Automatic streaming through chain
chain = prompt | model | parser
for chunk in chain.stream({"topic": "quantum"}):
    print(chunk, end="", flush=True)
```

**Async support:**

```python
# Automatic async execution
result = await chain.ainvoke({"topic": "AI"})
```

**Error handling and retries:**

- Built-in error propagation
- Retry logic at component level
- Graceful failure handling

**Key principle:** LCEL makes complex workflows look simple by treating all components as composable Runnables.

## Core Runnables

### RunnableSequence: Sequential Flow

Execute components one after another, passing output to the next:

**Explicit construction:**

```python
from langchain_core.runnables import RunnableSequence

prompt1 = PromptTemplate(
    template="Tell a joke about {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Explain this joke: {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = RunnableSequence(
    prompt1,
    model,
    parser,
    prompt2,
    model,
    parser
)

result = chain.invoke({"topic": "AI"})
```

**LCEL equivalent (preferred):**

```python
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic": "AI"})
```

**Execution flow:**

```
Input: {"topic": "AI"}
   ↓
prompt1 → "Tell a joke about AI"
   ↓
model → AIMessage("Why did the AI...")
   ↓
parser → "Why did the AI..." (string)
   ↓
prompt2 → "Explain this joke: Why did the AI..."
   ↓
model → AIMessage("This joke is funny because...")
   ↓
parser → "This joke is funny because..." (string)
   ↓
Output: "This joke is funny because..."
```

**Use RunnableSequence for:**

- Multi-step workflows
- Transform-then-process patterns
- Building up complex outputs progressively
- **Most common chain pattern**

### RunnableParallel: Concurrent Execution

Execute multiple chains simultaneously and combine results:

```python
from langchain_core.runnables import RunnableParallel, RunnableSequence

prompt_tweet = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)
prompt_linkedin = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet": prompt_tweet | model | parser,
    "linkedin": prompt_linkedin | model | parser
})

result = parallel_chain.invoke({"topic": "cybersecurity"})
print(result["tweet"])     # Tweet content
print(result["linkedin"])  # LinkedIn content
```

**Execution flow:**

```
Input: {"topic": "cybersecurity"}
      ↓
   ┌──┴──┐
   ↓     ↓
tweet  linkedin (execute in parallel)
   ↓     ↓
   └──┬──┘
      ↓
Output: {"tweet": "...", "linkedin": "..."}
```

**Benefits:**

- **Speed** - Parallel execution is faster than sequential
- **Independence** - Chains don't depend on each other's output
- **Structured output** - Results in a dictionary

**Use RunnableParallel for:**

- Generating multiple formats simultaneously (tweet + blog post + email)
- Multi-model approaches (query multiple LLMs, compare results)
- Parallel data processing
- Any independent operations

**Performance gain example:**

```python
# Sequential: 3 seconds + 3 seconds = 6 seconds total
result1 = chain1.invoke(input)  # 3 seconds
result2 = chain2.invoke(input)  # 3 seconds

# Parallel: max(3 seconds, 3 seconds) = 3 seconds total
results = RunnableParallel({
    "output1": chain1,
    "output2": chain2
}).invoke(input)  # 3 seconds
```

### RunnablePassthrough: Data Flow

Pass input through unchanged while allowing parallel processing:

```python
from langchain_core.runnables import RunnablePassthrough

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Explain this joke: {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

# Generate joke
joke_chain = prompt1 | model | parser

# Pass joke through + generate explanation
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),  # Pass joke unchanged
    "explanation": prompt2 | model | parser
})

# Combine: generate joke → pass through + explain
final_chain = joke_chain | parallel_chain
result = final_chain.invoke({"topic": "apple"})

print(result["joke"])         # Original joke
print(result["explanation"])  # Explanation of joke
```

**Execution flow:**

```
Input: {"topic": "apple"}
   ↓
joke_chain → "Why did the apple go to therapy?..."
   ↓
   ┌────────────────┐
   ↓                ↓
Passthrough    Explanation
   (unchanged)    (processed)
   ↓                ↓
   └────────┬───────┘
            ↓
Output: {
  "joke": "Why did the apple...",
  "explanation": "This is funny because..."
}
```

**Use RunnablePassthrough for:**

- Preserving intermediate results
- Creating audit trails (input + output together)
- Fan-out patterns (one input → multiple operations)
- Metadata preservation

**Common pattern - RAG with source tracking:**

```python
from langchain_core.runnables import RunnablePassthrough

retriever_chain = retriever  # Gets documents
prompt_chain = prompt | model | parser  # Generates answer

rag_chain = RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
}) | RunnableParallel({
    "answer": prompt_chain,
    "sources": lambda x: x["context"]  # Preserve retrieved docs
})

result = rag_chain.invoke("What is RAG?")
# Returns: {"answer": "...", "sources": [doc1, doc2, doc3]}
```

### RunnableLambda: Custom Functions

Wrap Python functions as Runnables for custom processing:

```python
from langchain_core.runnables import RunnableLambda

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()
joke_chain = prompt | model | parser

# Generate joke + count words
result_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})

final_chain = joke_chain | result_chain
result = final_chain.invoke({"topic": "houses"})

print(f"{result['joke']}\nWord count: {result['word_count']}")
```

**Custom function patterns:**

**Data transformation:**

```python
def extract_keywords(text):
    # Custom keyword extraction logic
    return text.split()[:5]

chain = generate_chain | RunnableLambda(extract_keywords)
```

**External API calls:**

```python
def fetch_weather(location):
    # Call weather API
    response = requests.get(f"api.weather.com/{location}")
    return response.json()

chain = location_extractor | RunnableLambda(fetch_weather)
```

**Validation and filtering:**

```python
def validate_output(text):
    if len(text) < 100:
        return text
    return text[:100] + "..."

chain = generator | RunnableLambda(validate_output)
```

**Use RunnableLambda for:**

- Custom business logic
- External API integrations
- Data validation and filtering
- Format conversions
- Statistical computations
- Any Python function in a chain

**Important:** Functions must take single input and return single output (can be dictionaries or objects).

### RunnableBranch: Conditional Routing

Route input through different chains based on conditions:

```python
from langchain_core.runnables import RunnableBranch

prompt_report = PromptTemplate(
    template="Give detailed report on {topic}",
    input_variables=["topic"]
)
prompt_summarize = PromptTemplate(
    template="Summarize this text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

# Generate report
report_chain = prompt_report | model | parser

# Branch: if long, summarize; else, pass through
branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 300,  # Condition
        prompt_summarize | model | parser  # Chain if True
    ),
    RunnablePassthrough()  # Default if False
)

final_chain = report_chain | branch_chain
result = final_chain.invoke({"topic": "Russia v Ukraine"})
```

**Branch structure:**

```python
RunnableBranch(
    (condition1, chain1),  # If condition1 is True, run chain1
    (condition2, chain2),  # Else if condition2 is True, run chain2
    default_chain          # Else run default_chain
)
```

**Execution flow:**

```
Input → Generate report → Check length
                            ↓
                    ┌───────┴────────┐
                    ↓                ↓
              > 300 words      < 300 words
                    ↓                ↓
               Summarize        Pass through
                    ↓                ↓
                    └───────┬────────┘
                            ↓
                         Output
```

**Multiple conditions example:**

```python
from pydantic import BaseModel, Field
from typing import Literal

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]

# Classify sentiment first
classify_chain = classify_prompt | model | PydanticOutputParser(pydantic_object=Sentiment)

# Branch based on sentiment
response_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        positive_response_prompt | model | parser
    ),
    (
        lambda x: x.sentiment == "negative",
        negative_response_prompt | model | parser
    ),
    neutral_response_prompt | model | parser  # Default
)

final_chain = classify_chain | response_chain
```

**Use RunnableBranch for:**

- Content routing based on classification
- Adaptive complexity (simple vs complex processing)
- Multi-language routing
- Error handling paths
- Dynamic workflow selection

**Lambda function tips:**

- `lambda x: condition` - Single value input
- `lambda x: x["field"] == value` - Dictionary input
- `lambda x: len(x.split()) > 100` - Text analysis
- `lambda x: x.score > 0.8` - Threshold checks

## Chain Patterns

Chains are common patterns built on Runnables. Understanding these patterns helps you design effective workflows.

### Sequential Chains

Multi-step processing where each step transforms the output:

**Basic sequential chain:**

```python
template1 = PromptTemplate(
    template="Give a detailed report on {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Summarize this report: {report}",
    input_variables=["report"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "quantum tunneling"})
```

**Pattern:**

```
Step 1: Generate → Step 2: Transform → Step 3: Refine → Output
```

**Common sequential patterns:**

**Generate → Critique → Refine:**

```python
generate = generate_prompt | model | parser
critique = critique_prompt | model | parser
refine = refine_prompt | model | parser

chain = generate | critique | refine
```

**Extract → Classify → Summarize:**

```python
extract = extract_prompt | model | parser
classify = classify_prompt | model | parser
summarize = summary_prompt | model | parser

chain = extract | classify | summarize
```

**When to use:**

- Multi-stage transformations
- Iterative refinement
- Progressive detail reduction/expansion
- **Most straightforward pattern**

### Parallel Chains

Generate multiple outputs simultaneously:

```python
prompt_notes = PromptTemplate(
    template="Generate short notes from:\n{text}",
    input_variables=["text"]
)
prompt_quiz = PromptTemplate(
    template="Generate 5 quiz questions from:\n{text}",
    input_variables=["text"]
)
prompt_merge = PromptTemplate(
    template="Merge notes and quiz:\nNotes: {notes}\nQuiz: {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# Step 1: Generate notes and quiz in parallel
parallel_chain = RunnableParallel({
    "notes": prompt_notes | model | parser,
    "quiz": prompt_quiz | model | parser
})

# Step 2: Merge results
merge_chain = prompt_merge | model | parser

# Combine: parallel generation → merge
final_chain = parallel_chain | merge_chain

text = """Support vector machines (SVMs) are supervised learning methods..."""
result = final_chain.invoke({"text": text})
```

**Pattern:**

```
      Input
        ↓
    ┌───┴───┐
    ↓       ↓
  Chain1  Chain2  (parallel)
    ↓       ↓
    └───┬───┘
        ↓
     Merge
        ↓
     Output
```

**Common parallel patterns:**

**Multi-format generation:**

```python
RunnableParallel({
    "email": email_prompt | model | parser,
    "tweet": tweet_prompt | model | parser,
    "blog": blog_prompt | model | parser
})
```

**Multi-model consensus:**

```python
RunnableParallel({
    "gpt4": gpt4_chain,
    "claude": claude_chain,
    "gemini": gemini_chain
}) | consensus_merger
```

**Parallel processing with aggregation:**

```python
RunnableParallel({
    "summary": summarize_chain,
    "keywords": keyword_chain,
    "sentiment": sentiment_chain
}) | aggregate_results
```

**When to use:**

- Independent operations
- Multi-perspective analysis
- Speed optimization (parallel > sequential)
- Generating multiple output formats

### Conditional Chains

Route through different processing paths based on data:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]

parser_sentiment = PydanticOutputParser(pydantic_object=Feedback)
parser_text = StrOutputParser()

# Classify sentiment
classify_template = PromptTemplate(
    template="Classify sentiment: {feedback}\n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser_sentiment.get_format_instructions()}
)
classify_chain = classify_template | model | parser_sentiment

# Response templates
thank_template = PromptTemplate(
    template="Write thankful message for: {feedback}",
    input_variables=["feedback"]
)
considerate_template = PromptTemplate(
    template="Write considerate message for: {feedback}",
    input_variables=["feedback"]
)

# Branch based on sentiment
branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        thank_template | model | parser_text
    ),
    (
        lambda x: x.sentiment == "negative",
        considerate_template | model | parser_text
    ),
    RunnableLambda(lambda x: "Thanks")  # Neutral default
)

# Full chain: classify → branch
final_chain = classify_chain | branch_chain

result = final_chain.invoke({"feedback": "This is the worst product"})
# Output: Considerate message (routed through negative branch)
```

**Pattern:**

```
Input → Classify → Route
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
    Path A      Path B      Path C
        ↓           ↓           ↓
        └───────────┴───────────┘
                    ↓
                 Output
```

**Common conditional patterns:**

**Complexity routing:**

```python
classify = simple_or_complex_classifier

RunnableBranch(
    (lambda x: x["complexity"] == "simple", simple_chain),
    (lambda x: x["complexity"] == "complex", complex_chain)
)
```

**Language routing:**

```python
detect_language_chain = language_detector

RunnableBranch(
    (lambda x: x["language"] == "en", english_chain),
    (lambda x: x["language"] == "es", spanish_chain),
    default_english_chain
)
```

**Quality control:**

```python
quality_check_chain = quality_checker

RunnableBranch(
    (lambda x: x["quality_score"] > 0.8, accept_chain),
    (lambda x: x["quality_score"] > 0.5, retry_chain),
    reject_chain
)
```

**When to use:**

- Content-dependent routing
- Adaptive processing complexity
- Multi-language support
- Quality gates and validation

### Complex Compositions

Combine sequential, parallel, and conditional patterns:

```python
# Example: Adaptive document processing

# Step 1: Classify document type
classify_chain = classify_prompt | model | type_parser

# Step 2: Branch based on type
process_branch = RunnableBranch(
    (
        lambda x: x["type"] == "technical",
        RunnableParallel({
            "summary": technical_summary_chain,
            "keywords": keyword_extraction_chain,
            "complexity": complexity_analysis_chain
        })
    ),
    (
        lambda x: x["type"] == "narrative",
        RunnableParallel({
            "summary": narrative_summary_chain,
            "themes": theme_extraction_chain,
            "sentiment": sentiment_analysis_chain
        })
    ),
    default_processing_chain
)

# Step 3: Format results
format_chain = format_prompt | model | parser

# Full workflow: classify → process (branched parallel) → format
workflow = classify_chain | process_branch | format_chain
```

**Complex pattern visualization:**

```
Input
  ↓
Classify (sequential)
  ↓
┌─┴─┐
│   │ Branch (conditional)
↓   ↓
Tech  Narrative
│     │
Parallel  Parallel (concurrent within branch)
↓     ↓
└──┬──┘
   ↓
Format (sequential)
   ↓
Output
```

**Design principles for complex chains:**

1. **Build incrementally** - Start simple, add complexity step by step
2. **Test components independently** - Verify each Runnable works alone
3. **Use descriptive names** - `sentiment_classifier` not `chain1`
4. **Visualize early** - Use graph visualization to understand flow
5. **Handle errors** - Add validation and fallbacks
6. **Monitor performance** - Measure latency at each stage

## Chain Visualization

LangChain provides tools to visualize chain execution:

**ASCII visualization:**

```python
chain = template1 | model | parser | template2 | model | parser
chain.get_graph().print_ascii()
```

**Output:**

```
           +-----------+
           | prompt1   |
           +-----------+
                 *
                 *
                 *
           +-----------+
           | ChatOpenAI|
           +-----------+
                 *
                 *
                 *
       +-----------------+
       | StrOutputParser |
       +-----------------+
                 *
                 *
                 *
           +-----------+
           | prompt2   |
           +-----------+
```

**Mermaid diagram (visual graph):**

```python
# Generate graph visualization
parallel_chain = RunnableParallel({
    "notes": prompt_notes | model | parser,
    "quiz": prompt_quiz | model | parser
})

merge_chain = prompt_merge | model | parser
final_chain = parallel_chain | merge_chain

# Save as PNG
image_bytes = final_chain.get_graph().draw_mermaid_png()
with open("chain-visualization.png", "wb") as f:
    f.write(image_bytes)
```

**Use visualization for:**

- **Understanding flow** - See how data moves through chain
- **Debugging** - Identify where chains fail
- **Documentation** - Explain workflows to team
- **Optimization** - Spot unnecessary sequential steps that could be parallel

**Complex chain example visualization:**

```
                    input
                      |
                      |
          +-----------+-----------+
          |                       |
       classify            RunnableBranch
          |                       |
          +----------+------------+
                     |
          +----------+-----------+
          |          |           |
      positive   negative    neutral
          |          |           |
        thankful considerate  default
          |          |           |
          +----------+-----------+
                     |
                  output
```

## Best Practices

### Design Principles

**1. Start simple, add complexity:**

```python
# Start with basic chain
chain = prompt | model | parser

# Add parallel processing when needed
chain = RunnableParallel({
    "answer": prompt | model | parser,
    "metadata": metadata_extractor
})

# Add branching when complexity demands
chain = classifier | RunnableBranch(...)
```

**2. Compose small chains into larger ones:**

```python
# Build reusable components
summarize = summarize_prompt | model | parser
translate = translate_prompt | model | parser
format_output = format_prompt | model | parser

# Compose into workflow
workflow = summarize | translate | format_output
```

**3. Use descriptive variable names:**

```python
# Bad
c1 = p1 | m | parser
c2 = p2 | m | parser

# Good
sentiment_analyzer = sentiment_prompt | model | parser
response_generator = response_prompt | model | parser
```

**4. Separate concerns:**

```python
# Data retrieval
retrieval_chain = query_builder | retriever

# Processing
processing_chain = prompt | model | parser

# Formatting
format_chain = formatter | validator

# Combine
full_pipeline = retrieval_chain | processing_chain | format_chain
```

### Error Handling

**Validate inputs:**

```python
def validate_input(x):
    if not x.get("topic"):
        raise ValueError("Missing required field: topic")
    return x

chain = RunnableLambda(validate_input) | prompt | model | parser
```

**Fallback chains:**

```python
try:
    result = primary_chain.invoke(input)
except Exception:
    result = fallback_chain.invoke(input)
```

**Conditional error recovery:**

```python
validation_chain = validator | RunnableBranch(
    (lambda x: x["valid"], main_chain),
    error_handling_chain
)
```

### Performance Optimization

**Use parallel where possible:**

```python
# Slow: sequential independent operations
summary = summarize_chain.invoke(text)
keywords = keyword_chain.invoke(text)
sentiment = sentiment_chain.invoke(text)

# Fast: parallel independent operations
result = RunnableParallel({
    "summary": summarize_chain,
    "keywords": keyword_chain,
    "sentiment": sentiment_chain
}).invoke(text)
```

**Batch processing:**

```python
# Process multiple inputs efficiently
results = chain.batch([input1, input2, input3])
```

**Streaming for responsiveness:**

```python
# Stream output as it's generated
for chunk in chain.stream(input):
    print(chunk, end="", flush=True)
```

### Testing Chains

**Test components independently:**

```python
# Test prompt
prompt_output = prompt.invoke({"topic": "AI"})
assert "AI" in prompt_output.text

# Test model
model_output = model.invoke("Tell me a joke")
assert len(model_output.content) > 0

# Test full chain
result = chain.invoke({"topic": "AI"})
assert isinstance(result, str)
```

**Use small test inputs:**

```python
# Don't test with full documents in unit tests
test_input = {"text": "Short test paragraph"}
result = chain.invoke(test_input)
```

**Monitor execution time:**

```python
import time

start = time.time()
result = chain.invoke(input)
duration = time.time() - start
print(f"Chain executed in {duration:.2f}s")
```

## Summary

Orchestration in LangChain is built on two levels:

**Runnables** - Core primitives for composition:

- **RunnableSequence** - Sequential execution with pipe operator (`|`)
- **RunnableParallel** - Concurrent execution of independent chains
- **RunnablePassthrough** - Pass data unchanged while enabling parallel operations
- **RunnableLambda** - Wrap custom Python functions as Runnables
- **RunnableBranch** - Conditional routing based on predicates

**Chains** - Common patterns built on Runnables:

- **Sequential chains** - Multi-step transformations (generate → refine → summarize)
- **Parallel chains** - Concurrent generation (notes + quiz simultaneously)
- **Conditional chains** - Content-dependent routing (classify → branch)
- **Complex compositions** - Nested combinations of all patterns

**LCEL (LangChain Expression Language):**

- Pipe operator (`|`) creates readable, composable chains
- Automatic streaming and async support
- Built-in error handling and retries
- Enables declarative workflow definition

**Key principles:**

- Start simple, add complexity incrementally
- Compose small, reusable chains into larger workflows
- Use parallel execution for independent operations
- Branch based on data characteristics
- Visualize chains early and often
- Test components independently before integration

**Practical workflow:**

1. Build individual components (prompts, models, parsers)
2. Chain with `|` for sequential flow
3. Wrap in `RunnableParallel` for concurrent operations
4. Add `RunnableBranch` for conditional logic
5. Use `RunnableLambda` for custom processing
6. Visualize with `.get_graph()`
7. Test thoroughly and optimize

These orchestration patterns enable building sophisticated LLM applications: multi-step reasoning, parallel generation, adaptive complexity, and conditional workflows -- all with readable, maintainable code.

## Next Steps

**Practice building:**

- Multi-stage RAG pipelines (retrieve → rerank → generate → validate)
- Content generation workflows (outline → draft → critique → revise)
- Multi-agent systems (planner → executor → validator)
- Adaptive applications (classify → route → process → format)

**Related concepts:**

- **[Fundamentals](fundamentals.md)** - Review models, prompts, and parsers used in chains
- **[Data and Retrieval](data-and-retrieval.md)** - Integrate loaders and retrievers into chains
- **[Prompt Engineering](../../prompt-engineering/)** - Design better prompts for chain steps
- **[RAG Architecture](../../retrieval-augmented-generation/rag-architecture.md)** - Framework-agnostic RAG patterns

**Advanced topics to explore:**

- Streaming chains with incremental output
- Async chains for high-concurrency applications
- Error handling and retry strategies
- Chain optimization and caching
- Custom Runnable implementations
- Chain composition patterns for specific domains
