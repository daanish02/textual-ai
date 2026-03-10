# Question Answering

## Table of Contents

- [Introduction](#introduction)
- [Extractive Question Answering](#extractive-question-answering)
- [Generative Question Answering](#generative-question-answering)
- [RAG for Factual QA](#rag-for-factual-qa)
- [Handling Unanswerable Questions](#handling-unanswerable-questions)
- [Multi-Hop Reasoning](#multi-hop-reasoning)
- [Confidence Estimation](#confidence-estimation)
- [Domain-Specific QA](#domain-specific-qa)
- [Production Patterns](#production-patterns)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Question Answering (QA) is the task of automatically answering questions posed in natural language. QA systems retrieve or generate answers from given contexts, knowledge bases, or the model's internal knowledge.

```
Question Answering Flow:

Question + Context → QA System → Answer

Types:
┌──────────────────────────────────────────────────────┐
│ Extractive: Select span from context                 │
│ Generative: Synthesize new answer                    │
│ Hybrid: Combine extraction and generation            │
└──────────────────────────────────────────────────────┘

Example:
Question: "Who is the CEO of Apple?"
Context: "Apple Inc., led by CEO Tim Cook, announced..."
Answer: "Tim Cook"
```

**Why QA matters**:

- **Natural interfaces**: Ask questions in plain language
- **Knowledge access**: Quick access to information
- **Customer support**: Automated help systems
- **Search enhancement**: Beyond keyword matching
- **Decision support**: Get specific answers from documents

This guide covers practical patterns for building robust QA systems with LLMs.

## Extractive Question Answering

### Understanding Extractive QA

```python
def extractive_qa_overview():
    """Overview of extractive question answering."""
    
    print("Extractive Question Answering:\n")
    print("="*80)
    
    print("""
Extractive QA selects a span (substring) from the context as the answer.

Characteristics:
  • Answer is verbatim from context
  • Guaranteed factual (if context is accurate)
  • Limited to information explicitly stated
  • No synthesis or paraphrasing

Example:
Context: "The Eiffel Tower, built in 1889, is located in Paris, France."
Question: "When was the Eiffel Tower built?"
Answer: "1889" (extracted from context)

Advantages:
  ✓ No hallucination risk
  ✓ Verifiable (can trace to source)
  ✓ Fast and efficient
  ✓ Works well with structured documents

Limitations:
  ✗ Cannot synthesize from multiple sources
  ✗ Limited by exact wording in context
  ✗ Cannot handle complex reasoning
  ✗ May return awkward spans

Use cases:
  • Document QA (specific facts)
  • Reading comprehension
  • FAQ systems
  • Legal/medical QA (precision critical)
""")

extractive_qa_overview()
```

### Implementing Extractive QA

```python
class ExtractivetQA:
    """Extract answer spans from context."""
    
    def __init__(self):
        pass
    
    def answer_question(self, question, context):
        """
        Answer question by extracting from context.
        
        Args:
            question: Question string
            context: Context paragraph
        
        Returns:
            Answer string
        """
        prompt = f"""
Answer this question by extracting the exact span from the context.
If the answer is not in the context, respond with "NOT_FOUND".

Context: {context}

Question: {question}

Answer (exact text from context):"""
        
        response = llm.generate(prompt, temperature=0)
        
        answer = response.strip()
        
        # Verify answer appears in context
        if answer != "NOT_FOUND" and answer not in context:
            # Try to find closest match
            answer = self.find_closest_span(answer, context)
        
        return answer
    
    def answer_with_position(self, question, context):
        """
        Answer with character positions.
        
        Args:
            question: Question string
            context: Context paragraph
        
        Returns:
            Dict with answer and position
        """
        prompt = f"""
Answer this question by identifying the exact span in the context.
Provide the answer text and its start position (character index).

Context: {context}

Question: {question}

Output format:
Answer: [exact text]
Start position: [number]

Response:"""
        
        response = llm.generate(prompt, temperature=0)
        
        # Parse response
        import re
        
        answer_match = re.search(r'Answer:\s*(.+)', response)
        pos_match = re.search(r'Start position:\s*(\d+)', response)
        
        if answer_match:
            answer = answer_match.group(1).strip()
            start_pos = int(pos_match.group(1)) if pos_match else -1
            
            # Verify position
            if start_pos >= 0 and context[start_pos:start_pos+len(answer)] == answer:
                return {
                    'answer': answer,
                    'start': start_pos,
                    'end': start_pos + len(answer)
                }
        
        return {'answer': None, 'start': -1, 'end': -1}
    
    def find_closest_span(self, target, context, threshold=0.8):
        """
        Find closest matching span in context.
        
        Args:
            target: Target answer text
            context: Context to search
            threshold: Similarity threshold
        
        Returns:
            Best matching span
        """
        from difflib import SequenceMatcher
        
        words = context.split()
        target_len = len(target.split())
        
        best_match = ""
        best_score = 0
        
        for i in range(len(words) - target_len + 1):
            span = " ".join(words[i:i + target_len])
            score = SequenceMatcher(None, target.lower(), span.lower()).ratio()
            
            if score > best_score:
                best_score = score
                best_match = span
        
        return best_match if best_score >= threshold else target
    
    def answer_multiple_questions(self, questions, context):
        """
        Answer multiple questions from same context.
        
        Args:
            questions: List of questions
            context: Shared context
        
        Returns:
            List of answers
        """
        answers = []
        
        for question in questions:
            answer = self.answer_question(question, context)
            answers.append({
                'question': question,
                'answer': answer
            })
        
        return answers

# Example Usage
print("\n" + "="*80)
print("Extractive QA Example\n")

context = """
The Python programming language was created by Guido van Rossum and first 
released in 1991. Python is known for its simple, readable syntax and is 
widely used in web development, data science, and artificial intelligence. 
The language is maintained by the Python Software Foundation. The latest 
major version, Python 3, was released in 2008.
"""

qa = ExtractivetQA()

questions = [
    "Who created Python?",
    "When was Python first released?",
    "What is Python used for?",
    "When was Python 3 released?"
]

print("Context:")
print(context)
print("\n" + "="*80)

for question in questions:
    answer = qa.answer_question(question, context)
    print(f"\nQ: {question}")
    print(f"A: {answer}")
```

### Multi-Paragraph Extractive QA

```python
class MultiParagraphQA:
    """Extractive QA across multiple paragraphs."""
    
    def __init__(self):
        self.single_qa = ExtractivetQA()
    
    def answer_from_multiple_contexts(self, question, contexts):
        """
        Find answer across multiple context paragraphs.
        
        Args:
            question: Question string
            contexts: List of context paragraphs
        
        Returns:
            Answer with source paragraph
        """
        # Try each context
        candidates = []
        
        for i, context in enumerate(contexts):
            answer = self.single_qa.answer_question(question, context)
            
            if answer != "NOT_FOUND":
                # Score the answer
                score = self.score_answer(question, answer, context)
                candidates.append({
                    'answer': answer,
                    'context_index': i,
                    'context': context,
                    'score': score
                })
        
        if not candidates:
            return {'answer': "NOT_FOUND", 'context_index': -1}
        
        # Return best scoring answer
        best = max(candidates, key=lambda x: x['score'])
        return best
    
    def score_answer(self, question, answer, context):
        """
        Score answer relevance.
        
        Args:
            question: Question string
            answer: Answer string
            context: Source context
        
        Returns:
            Relevance score
        """
        score = 0
        
        # Length heuristic (not too short, not too long)
        if 5 <= len(answer) <= 100:
            score += 0.3
        
        # Question word overlap
        q_words = set(question.lower().split())
        c_words = set(context.lower().split())
        overlap = len(q_words & c_words) / len(q_words) if q_words else 0
        score += overlap * 0.3
        
        # Answer confidence (placeholder - would use model confidence)
        score += 0.4
        
        return score
    
    def answer_with_evidence(self, question, contexts):
        """
        Answer with supporting evidence context.
        
        Args:
            question: Question string
            contexts: List of context paragraphs
        
        Returns:
            Answer with evidence
        """
        result = self.answer_from_multiple_contexts(question, contexts)
        
        if result['answer'] == "NOT_FOUND":
            return result
        
        # Add surrounding context as evidence
        evidence_start = max(0, result['context'].find(result['answer']) - 50)
        evidence_end = min(len(result['context']), 
                          result['context'].find(result['answer']) + len(result['answer']) + 50)
        
        result['evidence'] = result['context'][evidence_start:evidence_end]
        
        return result

# Example
print("\n\n" + "="*80)
print("Multi-Paragraph QA Example\n")

contexts = [
    "Python was created by Guido van Rossum in 1991. It emphasizes code readability.",
    "The Python Software Foundation manages the language's development and community.",
    "Python 3.0 was released in December 2008 with major improvements and breaking changes."
]

multi_qa = MultiParagraphQA()

question = "When was Python 3.0 released?"
result = multi_qa.answer_with_evidence(question, contexts)

print(f"Question: {question}")
print(f"Answer: {result['answer']}")
print(f"Source paragraph: {result.get('context_index', 'N/A')}")
print(f"Evidence: {result.get('evidence', 'N/A')}")
```

## Generative Question Answering

### Synthesizing Answers

```python
class GenerativeQA:
    """Generate synthesized answers to questions."""
    
    def __init__(self):
        pass
    
    def answer_question(self, question, context=None):
        """
        Generate answer to question.
        
        Args:
            question: Question string
            context: Optional context (if None, uses model's knowledge)
        
        Returns:
            Generated answer
        """
        if context:
            prompt = f"""
Answer this question based on the provided context. 
Synthesize a clear, concise answer in your own words.

Context: {context}

Question: {question}

Answer:"""
        else:
            prompt = f"""
Answer this question concisely and accurately.

Question: {question}

Answer:"""
        
        answer = llm.generate(prompt, temperature=0.3)
        
        return answer.strip()
    
    def answer_with_explanation(self, question, context):
        """
        Generate answer with reasoning explanation.
        
        Args:
            question: Question string
            context: Context paragraph
        
        Returns:
            Answer and explanation
        """
        prompt = f"""
Answer this question and explain your reasoning.

Context: {context}

Question: {question}

Provide:
1. Direct answer
2. Explanation of how you arrived at this answer

Response:"""
        
        response = llm.generate(prompt, temperature=0.3)
        
        # Parse answer and explanation
        import re
        
        parts = response.split('\n', 1)
        answer = parts[0].strip()
        explanation = parts[1].strip() if len(parts) > 1 else ""
        
        return {
            'answer': answer,
            'explanation': explanation
        }
    
    def answer_with_sources(self, question, contexts):
        """
        Generate answer synthesizing from multiple sources.
        
        Args:
            question: Question string
            contexts: List of context paragraphs
        
        Returns:
            Answer with source citations
        """
        # Format contexts with numbers
        formatted_contexts = "\n\n".join([
            f"[{i+1}] {ctx}" 
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""
Answer this question by synthesizing information from the provided sources.
Cite sources using [1], [2], etc.

Sources:
{formatted_contexts}

Question: {question}

Answer (with citations):"""
        
        answer = llm.generate(prompt, temperature=0.3)
        
        return answer.strip()
    
    def answer_step_by_step(self, question, context):
        """
        Generate answer with step-by-step reasoning.
        
        Args:
            question: Question string
            context: Context paragraph
        
        Returns:
            Answer with reasoning steps
        """
        prompt = f"""
Answer this question step-by-step, showing your reasoning.

Context: {context}

Question: {question}

Let's solve this step by step:
1. [First step]
2. [Second step]
...

Final Answer:"""
        
        response = llm.generate(prompt, temperature=0.3)
        
        # Parse steps and final answer
        import re
        
        steps = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\nFinal Answer:|\Z)', 
                          response, re.DOTALL)
        
        final_match = re.search(r'Final Answer:\s*(.+)', response, re.DOTALL)
        final_answer = final_match.group(1).strip() if final_match else ""
        
        return {
            'answer': final_answer,
            'reasoning_steps': [s.strip() for s in steps]
        }

# Example
print("\n\n" + "="*80)
print("Generative QA Example\n")

context = """
Climate change is primarily caused by greenhouse gas emissions from human 
activities. The burning of fossil fuels releases CO2, which traps heat in 
the atmosphere. Rising temperatures lead to melting ice caps, sea level rise, 
and more frequent extreme weather events. To address climate change, we need 
to reduce emissions, transition to renewable energy, and implement carbon 
capture technologies.
"""

gen_qa = GenerativeQA()

question = "What are the main causes and effects of climate change?"

print(f"Question: {question}\n")

# Basic answer
print("Basic Answer:")
answer = gen_qa.answer_question(question, context)
print(answer)

# With explanation
print("\n\nWith Explanation:")
result = gen_qa.answer_with_explanation(question, context)
print(f"Answer: {result['answer']}")
print(f"Explanation: {result['explanation']}")

# Step by step
print("\n\nStep-by-Step Reasoning:")
result = gen_qa.answer_step_by_step(question, context)
print(f"Answer: {result['answer']}")
print("Steps:")
for i, step in enumerate(result['reasoning_steps'], 1):
    print(f"  {i}. {step}")
```

### Comparing Extractive vs Generative

```python
def compare_qa_approaches():
    """Compare extractive and generative QA."""
    
    print("\n\nExtractive vs Generative QA:\n")
    print("="*80)
    
    comparison = """
┌─────────────────┬────────────────────────┬────────────────────────┐
│ Aspect          │ Extractive             │ Generative             │
├─────────────────┼────────────────────────┼────────────────────────┤
│ Answer Form     │ Exact span from text   │ Synthesized response   │
│ Factuality      │ High (from source)     │ Risk of hallucination  │
│ Flexibility     │ Limited to exact text  │ Can paraphrase         │
│ Multi-source    │ Difficult              │ Easy to combine        │
│ Reasoning       │ Limited                │ Can explain/reason     │
│ Fluency         │ May be awkward         │ Natural language       │
│ Verifiability   │ Easy to verify         │ Needs fact-checking    │
│ Speed           │ Fast                   │ Slower (generation)    │
│ Use Case        │ Fact lookup            │ Explanation/synthesis  │
└─────────────────┴────────────────────────┴────────────────────────┘

When to use EXTRACTIVE:
  • Factual accuracy is critical (legal, medical)
  • Answer is explicitly in text
  • Need to cite exact sources
  • Simple fact retrieval

When to use GENERATIVE:
  • Need natural, synthesized responses
  • Combining multiple sources
  • Explanation required
  • Complex reasoning needed

HYBRID APPROACH:
  1. Extract relevant spans
  2. Generate natural answer incorporating spans
  3. Best of both worlds!
"""
    
    print(comparison)

compare_qa_approaches()
```

## RAG for Factual QA

### Retrieval-Augmented Generation

```python
class RAGQA:
    """Question answering with retrieval-augmented generation."""
    
    def __init__(self, knowledge_base):
        """
        Initialize RAG QA system.
        
        Args:
            knowledge_base: List of documents
        """
        self.knowledge_base = knowledge_base
    
    def retrieve_relevant_docs(self, question, top_k=3):
        """
        Retrieve most relevant documents for question.
        
        Args:
            question: Question string
            top_k: Number of documents to retrieve
        
        Returns:
            List of relevant documents
        """
        # Simple keyword-based retrieval (would use embeddings in production)
        from collections import Counter
        
        question_words = set(question.lower().split())
        
        scores = []
        for doc in self.knowledge_base:
            doc_words = set(doc.lower().split())
            overlap = len(question_words & doc_words)
            scores.append((overlap, doc))
        
        # Sort by score and return top k
        scores.sort(reverse=True, key=lambda x: x[0])
        
        return [doc for score, doc in scores[:top_k]]
    
    def answer_with_rag(self, question):
        """
        Answer question using RAG.
        
        Args:
            question: Question string
        
        Returns:
            Answer with retrieved context
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(question)
        
        if not relevant_docs:
            return {
                'answer': "I don't have enough information to answer this question.",
                'sources': []
            }
        
        # Combine contexts
        combined_context = "\n\n".join([
            f"Source {i+1}:\n{doc}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Generate answer
        prompt = f"""
Answer this question based on the provided sources.

{combined_context}

Question: {question}

Answer (cite sources using [1], [2], etc.):"""
        
        answer = llm.generate(prompt, temperature=0.3)
        
        return {
            'answer': answer.strip(),
            'sources': relevant_docs
        }
    
    def answer_with_confidence(self, question):
        """
        Answer with confidence scoring.
        
        Args:
            question: Question string
        
        Returns:
            Answer with confidence
        """
        result = self.answer_with_rag(question)
        
        # Estimate confidence based on retrieval quality
        relevant_docs = result['sources']
        
        if not relevant_docs:
            confidence = 0.0
        else:
            # Simple heuristic: more sources = higher confidence
            confidence = min(1.0, len(relevant_docs) / 3.0)
        
        result['confidence'] = confidence
        
        return result

# Example
print("\n\n" + "="*80)
print("RAG QA Example\n")

knowledge_base = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python 3.0 was released in December 2008 with breaking changes from Python 2.",
    "The Python Software Foundation manages Python's development.",
    "Python is widely used in data science, web development, and AI.",
    "JavaScript was created by Brendan Eich in 1995 for web browsers."
]

rag_qa = RAGQA(knowledge_base)

question = "Who created Python and when?"

result = rag_qa.answer_with_confidence(question)

print(f"Question: {question}\n")
print(f"Answer: {result['answer']}\n")
print(f"Confidence: {result['confidence']:.2f}\n")
print("Retrieved Sources:")
for i, source in enumerate(result['sources'], 1):
    print(f"  [{i}] {source}")
```

### Improving RAG Quality

```python
def rag_improvement_strategies():
    """Strategies for improving RAG QA quality."""
    
    print("\n\nRAG Improvement Strategies:\n")
    print("="*80)
    
    print("""
1. BETTER RETRIEVAL
   • Use semantic search (embeddings) instead of keywords
   • Re-rank retrieved documents
   • Filter irrelevant results
   • Adjust retrieval threshold

2. CONTEXT AUGMENTATION
   • Include surrounding paragraphs
   • Add document metadata (title, source, date)
   • Provide document summaries

3. QUERY ENHANCEMENT
   • Expand query with synonyms
   • Reformulate ambiguous questions
   • Extract key entities from question

4. ANSWER POST-PROCESSING
   • Verify facts against retrieved docs
   • Add source citations
   • Format for readability
   • Detect and flag low confidence

5. ITERATIVE REFINEMENT
   • If answer is incomplete, retrieve more docs
   • Ask clarifying questions
   • Multi-step reasoning

6. QUALITY CHECKS
   • Factual consistency check
   • Relevance scoring
   • Citation verification
   • Hallucination detection
""")
    
    class ImprovedRAGQA:
        """RAG with quality improvements."""
        
        def __init__(self, knowledge_base):
            self.knowledge_base = knowledge_base
            self.rag_qa = RAGQA(knowledge_base)
        
        def answer_with_quality_checks(self, question):
            """
            Answer with comprehensive quality checks.
            
            Args:
                question: Question string
            
            Returns:
                Answer with quality metrics
            """
            # Get RAG answer
            result = self.rag_qa.answer_with_rag(question)
            
            # Quality checks
            quality_checks = {
                'has_sources': len(result['sources']) > 0,
                'answer_length_ok': 10 <= len(result['answer']) <= 500,
                'contains_citations': '[' in result['answer'],
            }
            
            # Factual consistency check
            consistent = self.check_consistency(
                result['answer'], 
                result['sources']
            )
            quality_checks['factually_consistent'] = consistent
            
            # Overall quality score
            quality_score = sum(quality_checks.values()) / len(quality_checks)
            
            result['quality_checks'] = quality_checks
            result['quality_score'] = quality_score
            
            return result
        
        def check_consistency(self, answer, sources):
            """
            Check if answer is consistent with sources.
            
            Args:
                answer: Generated answer
                sources: Retrieved sources
            
            Returns:
                Boolean consistency check
            """
            # Simple check: answer words should appear in sources
            answer_words = set(answer.lower().split())
            source_words = set()
            
            for source in sources:
                source_words.update(source.lower().split())
            
            # At least 50% of answer words should be in sources
            overlap = len(answer_words & source_words)
            consistency = overlap / len(answer_words) if answer_words else 0
            
            return consistency >= 0.5
    
    print("\n\nExample Quality Checks:\n")
    print("✓ Sources retrieved")
    print("✓ Answer length appropriate")
    print("✓ Citations included")
    print("✓ Factually consistent with sources")
    print("\nQuality Score: 1.0 (4/4 checks passed)")

rag_improvement_strategies()
```

## Handling Unanswerable Questions

### Detecting Unanswerable Questions

```python
class UnanswerableDetector:
    """Detect and handle unanswerable questions."""
    
    def __init__(self):
        pass
    
    def answer_or_refuse(self, question, context):
        """
        Answer question or indicate if unanswerable.
        
        Args:
            question: Question string
            context: Context paragraph
        
        Returns:
            Answer or refusal with reason
        """
        prompt = f"""
Answer this question based on the context. 
If the answer is NOT in the context, respond with "UNANSWERABLE" and explain why.

Context: {context}

Question: {question}

Response:"""
        
        response = llm.generate(prompt, temperature=0)
        
        if response.strip().upper().startswith("UNANSWERABLE"):
            return {
                'answerable': False,
                'reason': response.strip(),
                'answer': None
            }
        
        return {
            'answerable': True,
            'answer': response.strip(),
            'reason': None
        }
    
    def classify_answerability(self, question, context):
        """
        Classify if question is answerable before attempting to answer.
        
        Args:
            question: Question string
            context: Context paragraph
        
        Returns:
            Answerability classification
        """
        prompt = f"""
Determine if this question can be answered from the given context.

Context: {context}

Question: {question}

Can this question be answered from the context? Respond with:
- YES if the answer is clearly in the context
- PARTIAL if some information is there but incomplete
- NO if the answer is not in the context

Classification:"""
        
        response = llm.generate(prompt, temperature=0)
        
        classification = response.strip().upper()
        
        return {
            'answerable': classification == 'YES',
            'partial': classification == 'PARTIAL',
            'classification': classification
        }
    
    def answer_with_confidence_threshold(self, question, context, threshold=0.7):
        """
        Only answer if confidence exceeds threshold.
        
        Args:
            question: Question string
            context: Context paragraph
            threshold: Minimum confidence to answer
        
        Returns:
            Answer or refusal
        """
        # Check answerability
        check = self.classify_answerability(question, context)
        
        if not check['answerable'] and not check['partial']:
            return {
                'answer': "I cannot answer this question based on the provided context.",
                'confidence': 0.0,
                'refused': True
            }
        
        # Attempt to answer
        qa = GenerativeQA()
        answer = qa.answer_question(question, context)
        
        # Estimate confidence (simplified)
        confidence = 0.8 if check['answerable'] else 0.5
        
        if confidence < threshold:
            return {
                'answer': "I'm not confident enough to answer this question.",
                'confidence': confidence,
                'refused': True
            }
        
        return {
            'answer': answer,
            'confidence': confidence,
            'refused': False
        }

# Example
print("\n\n" + "="*80)
print("Unanswerable Questions Example\n")

context = """
Python is a popular programming language created in 1991. It is known for 
its simple syntax and wide applicability in various domains.
"""

detector = UnanswerableDetector()

questions = [
    ("When was Python created?", "Answerable"),
    ("Who maintains Python?", "Not in context"),
    ("What is Python's main feature?", "Partially answerable")
]

for question, expected in questions:
    result = detector.answer_or_refuse(question, context)
    
    print(f"\nQuestion: {question}")
    print(f"Expected: {expected}")
    print(f"Answerable: {result['answerable']}")
    
    if result['answerable']:
        print(f"Answer: {result['answer']}")
    else:
        print(f"Reason: {result['reason']}")
```

## Multi-Hop Reasoning

### Chaining Reasoning Steps

```python
class MultiHopQA:
    """QA requiring multiple reasoning steps."""
    
    def __init__(self):
        self.qa = GenerativeQA()
    
    def answer_multi_hop(self, question, contexts):
        """
        Answer question requiring multiple reasoning steps.
        
        Args:
            question: Complex question
            contexts: List of context paragraphs
        
        Returns:
            Answer with reasoning trace
        """
        # Decompose question into sub-questions
        sub_questions = self.decompose_question(question)
        
        # Answer each sub-question
        sub_answers = []
        for sq in sub_questions:
            # Find relevant context
            relevant_ctx = self.find_relevant_context(sq, contexts)
            
            # Answer sub-question
            answer = self.qa.answer_question(sq, relevant_ctx)
            sub_answers.append({
                'question': sq,
                'answer': answer,
                'context': relevant_ctx
            })
        
        # Synthesize final answer
        final_answer = self.synthesize_answer(question, sub_answers)
        
        return {
            'question': question,
            'sub_questions': sub_questions,
            'sub_answers': sub_answers,
            'final_answer': final_answer
        }
    
    def decompose_question(self, question):
        """
        Break complex question into sub-questions.
        
        Args:
            question: Complex question
        
        Returns:
            List of sub-questions
        """
        prompt = f"""
This question requires multiple steps to answer. Break it into simpler sub-questions.

Question: {question}

Sub-questions (numbered list):"""
        
        response = llm.generate(prompt, temperature=0)
        
        # Parse sub-questions
        import re
        sub_questions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)', response, re.DOTALL)
        
        return [sq.strip() for sq in sub_questions]
    
    def find_relevant_context(self, question, contexts):
        """
        Find most relevant context for question.
        
        Args:
            question: Question string
            contexts: List of contexts
        
        Returns:
            Most relevant context
        """
        # Simple keyword matching (would use embeddings in production)
        question_words = set(question.lower().split())
        
        best_context = contexts[0] if contexts else ""
        best_score = 0
        
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            score = len(question_words & ctx_words)
            
            if score > best_score:
                best_score = score
                best_context = ctx
        
        return best_context
    
    def synthesize_answer(self, question, sub_answers):
        """
        Synthesize final answer from sub-answers.
        
        Args:
            question: Original question
            sub_answers: List of sub-question answers
        
        Returns:
            Final synthesized answer
        """
        # Format sub-answers
        formatted = "\n".join([
            f"Q: {sa['question']}\nA: {sa['answer']}"
            for sa in sub_answers
        ])
        
        prompt = f"""
Synthesize a final answer to this question based on the sub-answers:

Original question: {question}

Sub-questions and answers:
{formatted}

Final answer:"""
        
        answer = llm.generate(prompt, temperature=0.3)
        
        return answer.strip()

# Example
print("\n\n" + "="*80)
print("Multi-Hop Reasoning Example\n")

contexts = [
    "The Eiffel Tower is located in Paris, France.",
    "Paris is the capital of France.",
    "France is a country in Western Europe with a population of 67 million."
]

multi_hop = MultiHopQA()

question = "What is the population of the country where the Eiffel Tower is located?"

print(f"Question: {question}\n")

result = multi_hop.answer_multi_hop(question, contexts)

print("Sub-questions:")
for i, sq in enumerate(result['sub_questions'], 1):
    print(f"  {i}. {sq}")

print("\n\nSub-answers:")
for sa in result['sub_answers']:
    print(f"  Q: {sa['question']}")
    print(f"  A: {sa['answer']}\n")

print(f"Final Answer: {result['final_answer']}")
```

## Confidence Estimation

### Measuring Answer Confidence

```python
class ConfidenceEstimator:
    """Estimate confidence in QA answers."""
    
    def __init__(self):
        pass
    
    def estimate_confidence_consistency(self, question, context, n_samples=5):
        """
        Estimate confidence via answer consistency.
        
        Args:
            question: Question string
            context: Context paragraph
            n_samples: Number of samples to generate
        
        Returns:
            Confidence score
        """
        qa = GenerativeQA()
        
        # Generate multiple answers with temperature
        answers = []
        for _ in range(n_samples):
            answer = qa.answer_question(question, context)
            answers.append(answer.lower().strip())
        
        # Calculate consistency
        from collections import Counter
        counts = Counter(answers)
        most_common_count = counts.most_common(1)[0][1]
        
        confidence = most_common_count / n_samples
        most_common_answer = counts.most_common(1)[0][0]
        
        return {
            'answer': most_common_answer,
            'confidence': confidence,
            'all_answers': answers
        }
    
    def estimate_confidence_verbalized(self, question, context):
        """
        Ask model to verbalize its confidence.
        
        Args:
            question: Question string
            context: Context paragraph
        
        Returns:
            Answer with confidence
        """
        prompt = f"""
Answer this question and rate your confidence (LOW/MEDIUM/HIGH).

Context: {context}

Question: {question}

Answer:
Confidence:"""
        
        response = llm.generate(prompt, temperature=0)
        
        # Parse answer and confidence
        lines = response.strip().split('\n')
        answer = lines[0].strip() if lines else ""
        
        confidence_text = ""
        for line in lines:
            if line.lower().startswith('confidence:'):
                confidence_text = line.split(':', 1)[1].strip().upper()
                break
        
        # Map to numeric score
        confidence_map = {
            'HIGH': 0.9,
            'MEDIUM': 0.6,
            'LOW': 0.3
        }
        
        confidence = confidence_map.get(confidence_text, 0.5)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'confidence_label': confidence_text
        }
    
    def estimate_confidence_context_overlap(self, question, answer, context):
        """
        Estimate confidence based on answer-context overlap.
        
        Args:
            question: Question string
            answer: Generated answer
            context: Context used
        
        Returns:
            Confidence score
        """
        # Check how much of answer appears in context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(answer_words & context_words)
        overlap_ratio = overlap / len(answer_words) if answer_words else 0
        
        # Higher overlap = higher confidence for extractive answers
        confidence = overlap_ratio
        
        return {
            'confidence': confidence,
            'overlap_ratio': overlap_ratio,
            'method': 'context_overlap'
        }

# Example
print("\n\n" + "="*80)
print("Confidence Estimation Example\n")

context = "Python was created by Guido van Rossum in 1991."
question = "Who created Python?"

estimator = ConfidenceEstimator()

print(f"Question: {question}\n")

# Method 1: Consistency
print("Method 1: Answer Consistency")
result = estimator.estimate_confidence_consistency(question, context, n_samples=3)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}\n")

# Method 2: Verbalized
print("Method 2: Verbalized Confidence")
result = estimator.estimate_confidence_verbalized(question, context)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence_label']} ({result['confidence']:.2f})")
```

## Domain-Specific QA

### Specialized QA Systems

```python
def domain_specific_qa():
    """Domain-specific QA patterns."""
    
    print("\n\nDomain-Specific QA:\n")
    print("="*80)
    
    print("""
Different domains require specialized QA approaches:

1. MEDICAL QA
   • High accuracy requirements
   • Use medical terminology correctly
   • Cite authoritative sources
   • Include disclaimers
   
   Example:
   Q: "What are the side effects of aspirin?"
   A: "Common side effects include stomach upset, heartburn... 
       [Source: Mayo Clinic]. Consult your doctor..."

2. LEGAL QA
   • Precise language
   • Cite statutes and case law
   • Avoid interpretation without qualifications
   
   Example:
   Q: "What is the statute of limitations for breach of contract?"
   A: "Varies by jurisdiction. In California, it's 4 years for 
       written contracts (Cal. Civ. Code § 337)..."

3. TECHNICAL QA
   • Include code examples
   • Reference documentation
   • Provide version-specific info
   
   Example:
   Q: "How do I create a list in Python?"
   A: "Use square brackets: my_list = [1, 2, 3] 
       or list(): my_list = list()..."

4. FINANCIAL QA
   • Include numbers and calculations
   • Cite data sources
   • Add temporal context
   
   Example:
   Q: "What was Apple's revenue in Q1?"
   A: "Apple reported $119.6 billion in Q1 2024 
       [Source: Apple Q1 2024 Earnings Report]..."
""")
    
    class MedicalQA:
        """Medical domain QA with safety features."""
        
        def answer_medical_question(self, question, context):
            """Answer medical question with appropriate disclaimers."""
            
            prompt = f"""
Answer this medical question based on the context. 

IMPORTANT: 
- Use accurate medical terminology
- Cite sources if available
- Include appropriate medical disclaimers

Context: {context}

Question: {question}

Answer:"""
            
            answer = llm.generate(prompt, temperature=0.1)
            
            # Add disclaimer
            disclaimer = "\n\n⚠️ This is for informational purposes only. Consult a healthcare provider for medical advice."
            
            return answer.strip() + disclaimer
    
    print("\n\nKey Considerations by Domain:\n")
    print("  Medical: Accuracy, disclaimers, citations")
    print("  Legal: Precision, jurisdiction-awareness, caveats")
    print("  Technical: Examples, version info, clarity")
    print("  Financial: Numbers, sources, context")

domain_specific_qa()
```

## Production Patterns

### Production QA System

```python
class ProductionQA:
    """Production-ready QA system."""
    
    def __init__(self, knowledge_base=None):
        """Initialize production QA system."""
        self.knowledge_base = knowledge_base or []
        self.cache = {}
        self.qa_log = []
    
    def answer_question_production(self, question, context=None, options=None):
        """
        Full production QA pipeline.
        
        Args:
            question: Question string
            context: Optional context
            options: Configuration options
        
        Returns:
            Complete QA result
        """
        import time
        import hashlib
        
        start_time = time.time()
        
        if options is None:
            options = {}
        
        use_rag = options.get('use_rag', True)
        check_answerability = options.get('check_answerability', True)
        estimate_confidence = options.get('estimate_confidence', True)
        
        # Check cache
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cached['metadata']['cache_hit'] = True
            return cached
        
        # Initialize result
        result = {
            'question': question,
            'metadata': {
                'timestamp': time.time(),
                'cache_hit': False
            }
        }
        
        # Check if answerable
        if check_answerability and context:
            detector = UnanswerableDetector()
            answerability = detector.classify_answerability(question, context)
            
            if not answerability['answerable']:
                result['answer'] = "Cannot answer based on provided context."
                result['answerable'] = False
                result['metadata']['latency_ms'] = int((time.time() - start_time) * 1000)
                return result
        
        # Use RAG if enabled
        if use_rag and self.knowledge_base:
            rag = RAGQA(self.knowledge_base)
            rag_result = rag.answer_with_rag(question)
            
            result['answer'] = rag_result['answer']
            result['sources'] = rag_result['sources']
        else:
            # Direct QA
            qa = GenerativeQA()
            result['answer'] = qa.answer_question(question, context)
            result['sources'] = [context] if context else []
        
        # Estimate confidence
        if estimate_confidence:
            estimator = ConfidenceEstimator()
            conf_result = estimator.estimate_confidence_verbalized(
                question, 
                context or result['sources'][0] if result['sources'] else ""
            )
            result['confidence'] = conf_result['confidence']
        
        # Finalize
        result['metadata']['latency_ms'] = int((time.time() - start_time) * 1000)
        result['answerable'] = True
        
        # Cache
        self.cache[cache_key] = result
        
        # Log
        self.qa_log.append({
            'question': question,
            'timestamp': time.time(),
            'answerable': result['answerable'],
            'confidence': result.get('confidence', 0)
        })
        
        return result
    
    def get_statistics(self):
        """Get QA system statistics."""
        
        if not self.qa_log:
            return {"message": "No questions processed yet"}
        
        total = len(self.qa_log)
        answerable = sum(1 for q in self.qa_log if q['answerable'])
        avg_confidence = sum(q.get('confidence', 0) for q in self.qa_log) / total
        
        return {
            'total_questions': total,
            'answerable_rate': answerable / total,
            'average_confidence': avg_confidence,
            'cache_size': len(self.cache)
        }

# Example
print("\n\n" + "="*80)
print("Production QA System Example\n")

knowledge_base = [
    "Python is a high-level programming language created in 1991.",
    "Python is known for its simple, readable syntax.",
    "Python is used in web development, data science, and AI."
]

prod_qa = ProductionQA(knowledge_base)

questions = [
    "What is Python?",
    "When was Python created?",
    "What is the capital of Mars?"
]

for question in questions:
    result = prod_qa.answer_question_production(
        question, 
        options={'use_rag': True, 'estimate_confidence': True}
    )
    
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"Answerable: {result.get('answerable', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"Latency: {result['metadata']['latency_ms']}ms")

print("\n\nSystem Statistics:")
stats = prod_qa.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")
```

## Summary

**Question Answering Overview**:

```
Question + Context → QA System → Answer

Types:
  • Extractive: Select span from context
  • Generative: Synthesize new answer
  • Hybrid: Combine both approaches
```

**Key Distinctions**:

| Aspect | Extractive | Generative |
|--------|-----------|------------|
| Output | Exact text span | Synthesized response |
| Factuality | High | Requires verification |
| Flexibility | Limited | High |
| Multi-source | Difficult | Easy |
| Use Case | Fact lookup | Explanation |

**Core Components**:

1. **Extractive QA**: Find answer span in context
2. **Generative QA**: Generate natural language answer
3. **RAG**: Retrieve relevant docs, then generate
4. **Multi-hop**: Chain reasoning across multiple sources
5. **Confidence**: Estimate answer reliability

**Best Practices**:

1. **Answer Quality**:
   - Verify against source
   - Cite evidence
   - Check factuality
   - Avoid hallucination

2. **Handling Edge Cases**:
   - Detect unanswerable questions
   - Set confidence thresholds
   - Provide "don't know" option
   - Explain limitations

3. **RAG Implementation**:
   - Semantic retrieval
   - Re-ranking
   - Multi-document synthesis
   - Source attribution

4. **Confidence Estimation**:
   - Answer consistency
   - Verbalized confidence
   - Context overlap
   - Multiple methods

**Production Considerations**:
- **Caching**: Store answers for repeated questions
- **Answerability Detection**: Refuse when uncertain
- **Quality Monitoring**: Track accuracy, confidence
- **Fallback Strategies**: Multiple QA methods
- **Latency Optimization**: Batch processing, caching
- **Error Handling**: Graceful degradation

**Common Pitfalls**:
- Hallucination → Use RAG with fact-checking
- Overconfidence → Implement confidence thresholds
- Missing context → Retrieve sufficient information
- Poor retrieval → Use semantic search
- No refusal mechanism → Detect unanswerable questions

**Domain Adaptation**:

| Domain | Requirements | Special Handling |
|--------|-------------|------------------|
| Medical | High accuracy | Disclaimers, citations |
| Legal | Precise language | Jurisdiction, caveats |
| Technical | Code examples | Version info, docs |
| Financial | Numbers, data | Sources, context |

**Key Takeaways**:
- Choose extractive for factual safety, generative for fluency
- Always implement RAG for knowledge-intensive QA
- Detect and refuse unanswerable questions
- Estimate and threshold confidence
- Cache aggressively for repeated questions
- Monitor quality continuously
- Adapt to domain requirements

## Next Steps

- Integrate with [Semantic Search](semantic-search.md) for retrieval
- Apply in [Conversational AI](conversational-ai.md) systems
- Use [Information Extraction](information-extraction.md) for structured QA
- Study [RAG](../retrieval-augmented-generation/) for advanced retrieval
- Implement [Evaluation Methods](../evaluation/) for QA quality
- Explore [Prompt Engineering](../prompt-engineering/) for better questions
