# RAG Evaluation

## Table of Contents

- [Introduction](#introduction)
- [Why Evaluate RAG Systems](#why-evaluate-rag-systems)
- [Retrieval Metrics](#retrieval-metrics)
- [Generation Metrics](#generation-metrics)
- [End-to-End Evaluation](#end-to-end-evaluation)
- [Evaluation Frameworks](#evaluation-frameworks)
- [Creating Test Sets](#creating-test-sets)
- [Debugging Poor Retrieval](#debugging-poor-retrieval)
- [Debugging Poor Generation](#debugging-poor-generation)
- [Continuous Evaluation](#continuous-evaluation)
- [Optimization Workflows](#optimization-workflows)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

**Evaluation** is critical for building reliable RAG systems. Without measurement, you can't improve.

```
The RAG Evaluation Loop:

┌──────────────────────────────────────────────────────┐
│ 1. BUILD: Create RAG system                         │
│    • Choose embedding model                          │
│    • Set chunking strategy                           │
│    • Configure retrieval                             │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│ 2. EVALUATE: Measure performance                    │
│    • Retrieval quality                               │
│    • Generation quality                              │
│    • End-to-end accuracy                             │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│ 3. ANALYZE: Find failure modes                       │
│    • Where does retrieval fail?                      │
│    • When does generation hallucinate?               │
│    • What queries struggle?                          │
└────────────────┬─────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────┐
│ 4. OPTIMIZE: Make targeted improvements              │
│    • Adjust chunk size                               │
│    • Try different embedding model                   │
│    • Add reranking                                   │
└────────────────┬─────────────────────────────────────┘
                 │
                 └────────► Back to step 2 (iterate)
```

**Key Insight**: RAG has two stages (retrieval + generation), each needs separate evaluation.

This guide covers metrics, evaluation frameworks, debugging strategies, and optimization workflows.

## Why Evaluate RAG Systems

### The Importance of Measurement

```python
def why_evaluate():
    """Understanding why RAG evaluation matters."""

    print("Why Evaluate RAG Systems?\n")

    print("=" * 60)
    print("\nThe Problem Without Evaluation:\n")

    problems = [
        ('Silent failures', 'System seems to work but gives wrong answers'),
        ('No improvement path', 'Don\'t know what to optimize'),
        ('Hidden biases', 'System works for some queries, fails for others'),
        ('Degradation unnoticed', 'Performance drops over time'),
        ('Wasted resources', 'Optimize the wrong components'),
    ]

    for problem, description in problems:
        print(f"  • {problem}: {description}")

    print("\n" + "=" * 60)
    print("\nWhat to Measure:\n")

    print("""
RAG System Decomposition:

User Query
    ↓
┌─────────────────────────┐
│ RETRIEVAL STAGE         │  ← Measure retrieval quality
│                         │
│ • Chunk documents       │
│ • Embed query           │
│ • Search vector DB      │
│ • Rerank results        │
└──────────┬──────────────┘
           │
           ▼ Retrieved chunks
┌─────────────────────────┐
│ GENERATION STAGE        │  ← Measure generation quality
│                         │
│ • Augment prompt        │
│ • Call LLM              │
│ • Generate response     │
└──────────┬──────────────┘
           │
           ▼
     Final Answer           ← Measure end-to-end quality

Each stage can fail independently!
""")

    print("=" * 60)
    print("\nEvaluation Benefits:\n")

    benefits = [
        ('Quantify quality', 'Know if system is good enough'),
        ('Compare approaches', 'A/B test chunking strategies'),
        ('Debug failures', 'Identify root causes'),
        ('Track progress', 'Measure improvements over time'),
        ('Optimize ROI', 'Focus effort on impactful changes'),
        ('Build confidence', 'Know system works before deploying'),
    ]

    for benefit, description in benefits:
        print(f"  • {benefit}: {description}")

why_evaluate()
```

### Types of Evaluation

```python
def evaluation_types():
    """Different types of RAG evaluation."""

    print("\n\nTypes of RAG Evaluation:\n")

    print("=" * 60)
    print("\n1. COMPONENT EVALUATION\n")

    print("""
Evaluate each component independently:

RETRIEVAL ONLY:
  • Given: Query + Relevant documents
  • Measure: Did retrieval find relevant docs?
  • Metrics: Precision, Recall, MRR, NDCG

GENERATION ONLY:
  • Given: Query + Correct context
  • Measure: Is answer accurate?
  • Metrics: Faithfulness, Relevance, BLEU

Pros:
  ✓ Isolates problems
  ✓ Easier to debug
  ✓ Fast to evaluate

Cons:
  ✗ Doesn't capture interaction effects
  ✗ May miss end-to-end failures
""")

    print("=" * 60)
    print("\n2. END-TO-END EVALUATION\n")

    print("""
Evaluate full pipeline:

COMPLETE RAG:
  • Given: Query only
  • Measure: Is final answer correct?
  • Metrics: Accuracy, Answer Quality, User Satisfaction

Pros:
  ✓ Tests real system behavior
  ✓ Captures all interactions
  ✓ User-centric

Cons:
  ✗ Hard to debug (where did it fail?)
  ✗ Slower to evaluate
  ✗ Requires ground truth answers
""")

    print("=" * 60)
    print("\n3. ONLINE EVALUATION\n")

    print("""
Evaluate in production:

LIVE TRAFFIC:
  • Implicit signals: Click-through rate, dwell time
  • Explicit signals: User ratings, feedback
  • A/B testing: Compare variants

Pros:
  ✓ Real user behavior
  ✓ Captures actual usage
  ✓ Ongoing monitoring

Cons:
  ✗ Delayed feedback
  ✗ Sparse signals
  ✗ Biased (only successful queries)
""")

evaluation_types()
```

## Retrieval Metrics

### Core Retrieval Metrics

```python
def retrieval_metrics_explained():
    """Understanding retrieval evaluation metrics."""

    print("\n\nRetrieval Metrics:\n")

    print("=" * 60)
    print("\n1. PRECISION @ K\n")

    print("""
What fraction of retrieved documents are relevant?

Precision@K = (# relevant docs in top-K) / K

Example (K=5):
  Retrieved: [Doc A✓, Doc B✗, Doc C✓, Doc D✗, Doc E✓]
  Relevant: 3 out of 5
  Precision@5 = 3/5 = 0.60

High precision → Few irrelevant docs (good for user experience)
""")

    print("=" * 60)
    print("\n2. RECALL @ K\n")

    print("""
What fraction of relevant documents are retrieved?

Recall@K = (# relevant docs in top-K) / (total # relevant docs)

Example (5 relevant docs total, K=5):
  Retrieved: [Doc A✓, Doc B✗, Doc C✓, Doc D✗, Doc E✓]
  Found: 3 out of 5 relevant docs
  Recall@5 = 3/5 = 0.60

High recall → Don't miss relevant docs (good for RAG quality)
""")

    print("=" * 60)
    print("\n3. MEAN RECIPROCAL RANK (MRR)\n")

    print("""
What is the rank of the first relevant document?

MRR = mean(1 / rank_of_first_relevant_doc)

Example:
  Query 1: First relevant doc at rank 2 → 1/2 = 0.50
  Query 2: First relevant doc at rank 1 → 1/1 = 1.00
  Query 3: First relevant doc at rank 4 → 1/4 = 0.25

  MRR = (0.50 + 1.00 + 0.25) / 3 = 0.58

High MRR → Relevant docs appear early (good for RAG)
""")

    print("=" * 60)
    print("\n4. NDCG @ K (Normalized Discounted Cumulative Gain)\n")

    print("""
Considers relevance degrees and position:

DCG@K = Σ (relevance_i / log2(i+1))  for i=1 to K

NDCG@K = DCG@K / IDCG@K  (normalize by ideal DCG)

Example (K=5, relevance scores 0-3):
  Retrieved: [Doc A(3), Doc B(0), Doc C(2), Doc D(1), Doc E(0)]

  DCG@5 = 3/log2(2) + 0/log2(3) + 2/log2(4) + 1/log2(5) + 0/log2(6)
        = 3.0 + 0.0 + 1.0 + 0.43 + 0.0
        = 4.43

  Ideal order: [3, 2, 1, 0, 0]
  IDCG@5 = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0 + 0
         = 3.0 + 1.26 + 0.5
         = 4.76

  NDCG@5 = 4.43 / 4.76 = 0.93

High NDCG → Relevant docs ranked highly (considers full ranking)
""")

    print("=" * 60)
    print("\n5. MEAN AVERAGE PRECISION (MAP)\n")

    print("""
Average of precision at each relevant document:

AP = (Σ Precision@i × relevance_i) / (# relevant docs)

MAP = mean(AP) across all queries

Example (5 relevant docs total):
  Retrieved ranks: [1✓, 3✓, 5✗, 7✓, 9✗, 11✓, 13✓]

  Relevant at rank 1: P@1 = 1/1 = 1.00
  Relevant at rank 3: P@3 = 2/3 = 0.67
  Relevant at rank 7: P@7 = 3/7 = 0.43
  Relevant at rank 11: P@11 = 4/11 = 0.36
  Relevant at rank 13: P@13 = 5/13 = 0.38

  AP = (1.00 + 0.67 + 0.43 + 0.36 + 0.38) / 5 = 0.57

High MAP → Relevant docs ranked early on average
""")

    print("=" * 60)
    print("\nWhich Metric to Use?\n")

    print("""
Metric          | Use When
----------------|--------------------------------------------------
Precision@K     | Care about top-K quality (fixed K)
Recall@K        | Want to ensure coverage (don't miss relevant docs)
MRR             | First relevant doc matters most (Q&A)
NDCG@K          | Have graded relevance (not just binary)
MAP             | Care about all relevant docs, not just top-K

For RAG:
  • NDCG@10 (primary) - Comprehensive ranking quality
  • Recall@10 (secondary) - Ensure we find relevant chunks
  • MRR (optional) - Speed to first good result
""")

retrieval_metrics_explained()
```

### Implementing Retrieval Metrics

```python
def implement_retrieval_metrics():
    """Implementing retrieval evaluation metrics."""

    print("\n\nImplementing Retrieval Metrics:\n")

    code = '''
import numpy as np
from typing import List, Set, Dict

class RetrievalEvaluator:
    """Evaluate retrieval quality."""

    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: Set of relevant document IDs
            k: Cutoff

        Returns:
            Precision@K score
        """
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)

        return relevant_retrieved / k if k > 0 else 0.0

    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Calculate Recall@K."""
        if not relevant:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)

        return relevant_retrieved / len(relevant)

    def f1_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate F1@K (harmonic mean of precision and recall)."""
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def mean_reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Returns:
            MRR score (1/rank of first relevant doc, or 0 if none found)
        """
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1.0 / rank

        return 0.0

    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Calculate NDCG@K.

        Args:
            retrieved: List of retrieved document IDs
            relevance_scores: Dict mapping doc ID to relevance (0-3 scale)
            k: Cutoff

        Returns:
            NDCG@K score
        """
        # Calculate DCG@K
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k], start=1):
            relevance = relevance_scores.get(doc, 0)
            dcg += relevance / np.log2(i + 1)

        # Calculate IDCG@K (ideal ranking)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_scores, start=1))

        return dcg / idcg if idcg > 0 else 0.0

    def average_precision(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Calculate Average Precision.

        Returns:
            AP score
        """
        if not relevant:
            return 0.0

        relevant_count = 0
        precision_sum = 0.0

        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i

        return precision_sum / len(relevant)

    def evaluate_query(
        self,
        query: str,
        retrieved: List[str],
        relevant: Set[str],
        relevance_scores: Dict[str, float] = None,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate a single query across all metrics.

        Returns:
            Dict of metric scores
        """
        metrics = {
            f'precision@{k}': self.precision_at_k(retrieved, relevant, k),
            f'recall@{k}': self.recall_at_k(retrieved, relevant, k),
            f'f1@{k}': self.f1_at_k(retrieved, relevant, k),
            'mrr': self.mean_reciprocal_rank(retrieved, relevant),
            'map': self.average_precision(retrieved, relevant),
        }

        # Add NDCG if relevance scores provided
        if relevance_scores:
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(retrieved, relevance_scores, k)

        return metrics

    def evaluate_dataset(
        self,
        results: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate across multiple queries.

        Args:
            results: List of dicts with 'query', 'retrieved', 'relevant'

        Returns:
            Average metrics across all queries
        """
        all_metrics = []

        for result in results:
            query_metrics = self.evaluate_query(
                query=result['query'],
                retrieved=result['retrieved'],
                relevant=result['relevant'],
                relevance_scores=result.get('relevance_scores'),
                k=result.get('k', 10)
            )
            all_metrics.append(query_metrics)

        # Average across queries
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics


# Example usage
evaluator = RetrievalEvaluator()

# Single query evaluation
retrieved = ['doc1', 'doc5', 'doc3', 'doc8', 'doc2']
relevant = {'doc1', 'doc2', 'doc3', 'doc9'}
relevance_scores = {'doc1': 3, 'doc2': 2, 'doc3': 2, 'doc9': 1}

metrics = evaluator.evaluate_query(
    query="How to install Python?",
    retrieved=retrieved,
    relevant=relevant,
    relevance_scores=relevance_scores,
    k=5
)

print("Retrieval Metrics:")
for metric, score in metrics.items():
    print(f"  {metric}: {score:.3f}")

# Output:
#   precision@5: 0.600
#   recall@5: 0.750
#   f1@5: 0.667
#   mrr: 1.000
#   map: 0.806
#   ndcg@5: 0.934


# Multiple queries evaluation
dataset_results = [
    {
        'query': 'Python install',
        'retrieved': ['doc1', 'doc5', 'doc3'],
        'relevant': {'doc1', 'doc3'}
    },
    {
        'query': 'Python tutorial',
        'retrieved': ['doc2', 'doc4', 'doc1'],
        'relevant': {'doc2', 'doc4'}
    },
    # ... more queries
]

avg_metrics = evaluator.evaluate_dataset(dataset_results)

print("\\nAverage Metrics Across Dataset:")
for metric, score in avg_metrics.items():
    print(f"  {metric}: {score:.3f}")
'''

    print(code)

implement_retrieval_metrics()
```

## Generation Metrics

### Core Generation Metrics

```python
def generation_metrics_explained():
    """Understanding generation evaluation metrics."""

    print("\n\nGeneration Metrics:\n")

    print("=" * 60)
    print("\n1. FAITHFULNESS (Context Adherence)\n")

    print("""
Does the answer stay faithful to the retrieved context?

High faithfulness → Answer is grounded in context
Low faithfulness → Answer hallucinates or adds info

Measurement approaches:
  • LLM-based: Ask LLM "Is this answer supported by context?"
  • NLI-based: Natural Language Inference classifier
  • Human evaluation: Manual review

Example:
  Context: "Python was created by Guido van Rossum in 1991."

  Answer: "Python was created in 1991 by Guido." ✓ Faithful
  Answer: "Python is the best language." ✗ Not supported by context
  Answer: "Python was created in 1989." ✗ Contradicts context
""")

    print("=" * 60)
    print("\n2. ANSWER RELEVANCE\n")

    print("""
Does the answer actually address the query?

High relevance → Answer directly answers the question
Low relevance → Answer is off-topic or incomplete

Example:
  Query: "How to install Python?"

  Answer: "Download Python from python.org and run installer." ✓ Relevant
  Answer: "Python is a programming language." ✗ Doesn't answer "how"
  Answer: "Java is also popular." ✗ Off-topic
""")

    print("=" * 60)
    print("\n3. CONTEXT RELEVANCE\n")

    print("""
Is the retrieved context relevant to the query?

High relevance → Retrieved chunks help answer query
Low relevance → Retrieved junk, noise

This measures retrieval quality from generation perspective.

Example:
  Query: "How to install Python?"

  Context: "To install Python, download from python.org..." ✓ Relevant
  Context: "Python is dynamically typed..." ~ Somewhat relevant
  Context: "Java requires JVM..." ✗ Not relevant
""")

    print("=" * 60)
    print("\n4. ANSWER CORRECTNESS\n")

    print("""
Is the answer factually correct?

Requires ground truth answer for comparison.

Measurement:
  • Exact match: Answer == Ground truth
  • Semantic similarity: Embedding similarity
  • LLM evaluation: "Are these answers equivalent?"
  • Component matching: Check key facts present

Example:
  Query: "Who created Python?"
  Ground truth: "Guido van Rossum"

  Answer: "Guido van Rossum" ✓ Correct (exact match)
  Answer: "It was created by Guido." ✓ Correct (semantic)
  Answer: "Python's creator is Guido." ✓ Correct (paraphrase)
  Answer: "Dennis Ritchie" ✗ Incorrect
""")

    print("=" * 60)
    print("\nWhich Metric to Use?\n")

    print("""
Metric              | What It Measures         | Need Ground Truth?
--------------------|--------------------------|--------------------
Faithfulness        | Context adherence        | No
Answer Relevance    | Addresses query?         | No
Context Relevance   | Good retrieval?          | No
Answer Correctness  | Factually correct?       | Yes

For RAG:
  • Faithfulness (primary) - Critical for trust
  • Answer Relevance (primary) - User satisfaction
  • Context Relevance (diagnostic) - Debug retrieval
  • Answer Correctness (if available) - Best measure
""")

generation_metrics_explained()
```

### Implementing Generation Metrics

```python
def implement_generation_metrics():
    """Implementing generation evaluation metrics."""

    print("\n\nImplementing Generation Metrics:\n")

    code = '''
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

class GenerationEvaluator:
    """Evaluate generation quality."""

    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)

    def faithfulness_llm(
        self,
        answer: str,
        context: str,
        llm_call
    ) -> Dict:
        """
        Evaluate faithfulness using LLM.

        Args:
            answer: Generated answer
            context: Retrieved context
            llm_call: Function to call LLM

        Returns:
            Dict with score and explanation
        """
        prompt = f"""Evaluate if the answer is faithful to the context.

Context:
{context}

Answer:
{answer}

Is the answer supported by the context? Rate from 0-10 where:
- 10: Fully supported, no additions
- 5: Mostly supported, minor additions
- 0: Not supported or contradicts context

Provide:
1. Score (0-10)
2. Explanation

Format:
Score: X
Explanation: ...
"""

        response = llm_call(prompt)

        # Parse response
        lines = response.strip().split('\\n')
        score_line = [l for l in lines if l.startswith('Score:')]

        if score_line:
            score = float(score_line[0].split(':')[1].strip())
        else:
            score = 5.0  # Default

        return {
            'faithfulness_score': score / 10.0,  # Normalize to 0-1
            'explanation': response
        }

    def answer_relevance_semantic(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Evaluate answer relevance using semantic similarity.

        Args:
            query: Original query
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        query_emb = self.model.encode(query, convert_to_tensor=True)
        answer_emb = self.model.encode(answer, convert_to_tensor=True)

        similarity = util.cos_sim(query_emb, answer_emb).item()

        return float(similarity)

    def context_relevance(
        self,
        query: str,
        contexts: List[str],
        top_k: int = None
    ) -> Dict:
        """
        Evaluate if contexts are relevant to query.

        Args:
            query: Original query
            contexts: Retrieved context chunks
            top_k: Consider only top-k contexts

        Returns:
            Dict with scores and per-context relevance
        """
        if top_k:
            contexts = contexts[:top_k]

        query_emb = self.model.encode(query, convert_to_tensor=True)
        context_embs = self.model.encode(contexts, convert_to_tensor=True)

        # Calculate similarity for each context
        similarities = util.cos_sim(query_emb, context_embs)[0]

        # Average similarity
        avg_relevance = similarities.mean().item()

        # Count highly relevant (>0.5)
        highly_relevant = (similarities > 0.5).sum().item()

        return {
            'avg_context_relevance': float(avg_relevance),
            'highly_relevant_count': int(highly_relevant),
            'per_context_scores': similarities.tolist()
        }

    def answer_correctness(
        self,
        answer: str,
        ground_truth: str,
        method: str = 'semantic'
    ) -> float:
        """
        Evaluate answer correctness against ground truth.

        Args:
            answer: Generated answer
            ground_truth: Correct answer
            method: 'exact', 'semantic', or 'llm'

        Returns:
            Correctness score (0-1)
        """
        if method == 'exact':
            # Case-insensitive exact match
            return 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0

        elif method == 'semantic':
            # Semantic similarity
            answer_emb = self.model.encode(answer, convert_to_tensor=True)
            truth_emb = self.model.encode(ground_truth, convert_to_tensor=True)

            similarity = util.cos_sim(answer_emb, truth_emb).item()

            return float(similarity)

        else:
            raise ValueError(f"Unknown method: {method}")

    def evaluate_rag_response(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None,
        llm_call = None
    ) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation.

        Args:
            query: Original query
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            llm_call: Optional LLM for faithfulness

        Returns:
            Dict of all metrics
        """
        metrics = {}

        # Answer relevance (always available)
        metrics['answer_relevance'] = self.answer_relevance_semantic(query, answer)

        # Context relevance (always available)
        context_metrics = self.context_relevance(query, contexts)
        metrics.update(context_metrics)

        # Faithfulness (if LLM available)
        if llm_call and contexts:
            combined_context = "\\n\\n".join(contexts)
            faithfulness_result = self.faithfulness_llm(answer, combined_context, llm_call)
            metrics['faithfulness'] = faithfulness_result['faithfulness_score']

        # Correctness (if ground truth available)
        if ground_truth:
            metrics['correctness_semantic'] = self.answer_correctness(
                answer, ground_truth, method='semantic'
            )
            metrics['correctness_exact'] = self.answer_correctness(
                answer, ground_truth, method='exact'
            )

        return metrics


# Example usage
evaluator = GenerationEvaluator()

# Evaluate a RAG response
query = "How to install Python?"
answer = "Download Python from python.org and run the installer."
contexts = [
    "To install Python, visit python.org and download the installer for your OS.",
    "Python is a high-level programming language.",
    "Java requires JVM to run."
]
ground_truth = "Download from python.org and run the installer."

metrics = evaluator.evaluate_rag_response(
    query=query,
    answer=answer,
    contexts=contexts,
    ground_truth=ground_truth,
    llm_call=None  # Could pass LLM function here
)

print("Generation Metrics:")
for metric, score in metrics.items():
    if isinstance(score, float):
        print(f"  {metric}: {score:.3f}")
    else:
        print(f"  {metric}: {score}")

# Output:
#   answer_relevance: 0.847
#   avg_context_relevance: 0.623
#   highly_relevant_count: 1
#   correctness_semantic: 0.923
#   correctness_exact: 0.000
'''

    print(code)

implement_generation_metrics()
```

## End-to-End Evaluation

### Complete RAG Evaluation

```python
def end_to_end_evaluation():
    """End-to-end RAG system evaluation."""

    print("\n\nEnd-to-End RAG Evaluation:\n")

    print("=" * 60)
    print("\nConcept:\n")

    print("""
Evaluate the complete RAG pipeline:

Query → [RAG System] → Answer

Measure:
  • Is the answer correct?
  • Is it relevant to the query?
  • Is it faithful to retrieved context?
  • How long did it take?
  • What was the cost?

Benefits:
  ✓ Tests real system behavior
  ✓ Captures component interactions
  ✓ User-centric metrics

Challenges:
  ✗ Requires test set with queries + ground truth
  ✗ Hard to debug (is retrieval or generation the issue?)
  ✗ Time-consuming to evaluate
""")

    code = '''
class RAGEvaluator:
    """End-to-end RAG system evaluation."""

    def __init__(
        self,
        rag_system,
        retrieval_evaluator,
        generation_evaluator
    ):
        self.rag_system = rag_system
        self.retrieval_eval = retrieval_evaluator
        self.generation_eval = generation_evaluator

    def evaluate_single_query(
        self,
        query: str,
        ground_truth_answer: str,
        relevant_doc_ids: Set[str],
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate RAG on a single query.

        Args:
            query: User query
            ground_truth_answer: Expected answer
            relevant_doc_ids: IDs of relevant documents
            verbose: Print detailed results

        Returns:
            Dict of metrics and intermediate results
        """
        import time

        # Run RAG system
        start_time = time.time()

        rag_result = self.rag_system.query(query)

        latency = time.time() - start_time

        # Extract results
        answer = rag_result['answer']
        retrieved_docs = rag_result['retrieved_documents']
        retrieved_ids = [doc['id'] for doc in retrieved_docs]
        contexts = [doc['text'] for doc in retrieved_docs]

        # Evaluate retrieval
        retrieval_metrics = self.retrieval_eval.evaluate_query(
            query=query,
            retrieved=retrieved_ids,
            relevant=relevant_doc_ids,
            k=10
        )

        # Evaluate generation
        generation_metrics = self.generation_eval.evaluate_rag_response(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth_answer
        )

        # Combine all metrics
        all_metrics = {
            **retrieval_metrics,
            **generation_metrics,
            'latency': latency,
            'num_retrieved': len(retrieved_docs)
        }

        if verbose:
            print(f"Query: {query}")
            print(f"Answer: {answer}")
            print(f"Ground Truth: {ground_truth_answer}")
            print(f"\\nMetrics:")
            for k, v in all_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
            print()

        return {
            'query': query,
            'answer': answer,
            'ground_truth': ground_truth_answer,
            'retrieved_ids': retrieved_ids,
            'metrics': all_metrics
        }

    def evaluate_dataset(
        self,
        test_set: List[Dict],
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate RAG on entire test set.

        Args:
            test_set: List of dicts with 'query', 'answer', 'relevant_docs'
            verbose: Print progress

        Returns:
            Aggregated metrics and per-query results
        """
        results = []

        for i, test_case in enumerate(test_set):
            if verbose and i % 10 == 0:
                print(f"Evaluating {i+1}/{len(test_set)}...")

            result = self.evaluate_single_query(
                query=test_case['query'],
                ground_truth_answer=test_case['answer'],
                relevant_doc_ids=set(test_case['relevant_docs']),
                verbose=False
            )

            results.append(result)

        # Aggregate metrics
        all_metrics = [r['metrics'] for r in results]

        aggregated = {}
        for key in all_metrics[0].keys():
            if isinstance(all_metrics[0][key], (int, float)):
                values = [m[key] for m in all_metrics]
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)

        return {
            'aggregated_metrics': aggregated,
            'per_query_results': results,
            'num_queries': len(test_set)
        }

    def compare_systems(
        self,
        system_a,
        system_b,
        test_set: List[Dict],
        alpha: float = 0.05
    ) -> Dict:
        """
        Compare two RAG systems statistically.

        Args:
            system_a: First RAG system
            system_b: Second RAG system
            test_set: Test queries
            alpha: Significance level

        Returns:
            Comparison results with statistical tests
        """
        from scipy import stats

        # Evaluate both systems
        eval_a = RAGEvaluator(system_a, self.retrieval_eval, self.generation_eval)
        eval_b = RAGEvaluator(system_b, self.retrieval_eval, self.generation_eval)

        results_a = eval_a.evaluate_dataset(test_set)
        results_b = eval_b.evaluate_dataset(test_set)

        # Extract key metrics
        metrics_a = [r['metrics'] for r in results_a['per_query_results']]
        metrics_b = [r['metrics'] for r in results_b['per_query_results']]

        # Compare each metric
        comparisons = {}

        for key in metrics_a[0].keys():
            if isinstance(metrics_a[0][key], (int, float)):
                values_a = [m[key] for m in metrics_a]
                values_b = [m[key] for m in metrics_b]

                # Paired t-test
                t_stat, p_value = stats.ttest_rel(values_a, values_b)

                comparisons[key] = {
                    'system_a_mean': np.mean(values_a),
                    'system_b_mean': np.mean(values_b),
                    'difference': np.mean(values_b) - np.mean(values_a),
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'winner': 'B' if np.mean(values_b) > np.mean(values_a) and p_value < alpha else
                             ('A' if np.mean(values_a) > np.mean(values_b) and p_value < alpha else 'Tie')
                }

        return {
            'comparisons': comparisons,
            'system_a_results': results_a,
            'system_b_results': results_b
        }


# Example usage
rag_evaluator = RAGEvaluator(
    rag_system=my_rag_system,
    retrieval_evaluator=retrieval_eval,
    generation_evaluator=generation_eval
)

# Test set
test_set = [
    {
        'query': 'How to install Python?',
        'answer': 'Download from python.org and run the installer.',
        'relevant_docs': ['doc1', 'doc5', 'doc12']
    },
    # ... more test cases
]

# Evaluate
evaluation_results = rag_evaluator.evaluate_dataset(test_set, verbose=True)

print("\\nAggregated Results:")
for metric, value in evaluation_results['aggregated_metrics'].items():
    print(f"  {metric}: {value:.3f}")

# Compare two systems
comparison = rag_evaluator.compare_systems(
    system_a=baseline_rag,
    system_b=improved_rag,
    test_set=test_set
)

print("\\nSystem Comparison:")
for metric, comp in comparison['comparisons'].items():
    if comp['significant']:
        print(f"  {metric}: {comp['winner']} wins (p={comp['p_value']:.4f})")
'''

    print(code)

end_to_end_evaluation()
```

## Evaluation Frameworks

### Using RAGAs and LangChain

```python
def evaluation_frameworks():
    """Overview of RAG evaluation frameworks."""

    print("\n\nEvaluation Frameworks:\n")

    print("=" * 60)
    print("\n1. RAGAS (Recommended)\n")

    print("""
RAG Assessment Framework:

Features:
  • Comprehensive metrics (faithfulness, relevance, etc.)
  • LLM-based evaluation (no ground truth needed)
  • Easy integration with LangChain
  • Automated test set generation

Metrics:
  • Faithfulness: Answer grounded in context?
  • Answer Relevance: Addresses query?
  • Context Precision: Retrieved contexts relevant?
  • Context Recall: All necessary context retrieved?

Installation:
  pip install ragas

Usage:
  from ragas import evaluate
  from ragas.metrics import faithfulness, answer_relevance

  result = evaluate(
      dataset=test_dataset,
      metrics=[faithfulness, answer_relevance]
  )
""")

    code_ragas = '''
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare data
data = {
    'question': [
        'How to install Python?',
        'What is machine learning?'
    ],
    'answer': [
        'Download from python.org and run the installer.',
        'ML is a subset of AI that learns from data.'
    ],
    'contexts': [
        ['To install Python, visit python.org...'],
        ['Machine learning is a field of AI...', 'ML algorithms learn patterns...']
    ],
    'ground_truth': [  # Optional
        'Download Python from python.org.',
        'Machine learning is part of AI.'
    ]
}

dataset = Dataset.from_dict(data)

# Evaluate
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevance,
        context_precision,
        context_recall,
    ],
)

print("RAGAS Evaluation Results:")
print(result)

# Output:
# {'faithfulness': 0.95, 'answer_relevance': 0.92, ...}
'''

    print(code_ragas)

    print("\n" + "=" * 60)
    print("\n2. LangChain Evaluation\n")

    print("""
LangChain built-in evaluation:

Features:
  • QA evaluation chain
  • Pairwise string comparison
  • Criteria-based evaluation
  • Custom evaluators

Components:
  • load_evaluator(): Load pre-built evaluators
  • QAEvalChain: Question-answering evaluation
  • PairwiseStringEvaluator: Compare two outputs

Usage:
  from langchain.evaluation import load_evaluator

  evaluator = load_evaluator("qa")
  result = evaluator.evaluate_strings(
      prediction="Paris is the capital of France.",
      input="What is the capital of France?",
      reference="Paris."
  )
""")

    code_langchain = '''
from langchain.evaluation import load_evaluator

# QA Evaluator
qa_evaluator = load_evaluator("qa")

result = qa_evaluator.evaluate_strings(
    prediction="Python was created by Guido van Rossum in 1991.",
    input="Who created Python and when?",
    reference="Guido van Rossum created Python in 1991."
)

print(f"Score: {result['score']}")
print(f"Reasoning: {result['reasoning']}")


# Criteria-based evaluation
criteria_evaluator = load_evaluator(
    "criteria",
    criteria="conciseness"
)

result = criteria_evaluator.evaluate_strings(
    prediction="Python is a high-level programming language that was created...",
    input="What is Python?"
)

print(f"Conciseness score: {result['score']}")


# Custom criteria
custom_evaluator = load_evaluator(
    "labeled_criteria",
    criteria={
        "accuracy": "Is the answer factually accurate?",
        "completeness": "Does the answer fully address the question?"
    }
)

result = custom_evaluator.evaluate_strings(
    prediction="Python is a programming language.",
    input="What is Python and who created it?",
    reference="Python is a language created by Guido."
)

print(f"Results: {result}")
'''

    print(code_langchain)

    print("\n" + "=" * 60)
    print("\n3. TruLens (Observability)\n")

    print("""
TruLens for RAG observability:

Features:
  • Real-time monitoring
  • Feedback functions (metrics)
  • Trace exploration
  • Comparison dashboards

Metrics:
  • Context relevance
  • Groundedness (faithfulness)
  • Answer relevance
  • Toxicity, bias detection

Usage:
  from trulens_eval import TruChain, Feedback

  # Wrap RAG system
  tru_rag = TruChain(rag_system)

  # Define feedbacks (metrics)
  feedbacks = [
      Feedback(groundedness).on_output(),
      Feedback(answer_relevance).on_input_output()
  ]

  # Query with tracking
  result = tru_rag.query("How to install Python?")

  # View dashboard
  from trulens_eval import Tru
  tru = Tru()
  tru.run_dashboard()
""")

evaluation_frameworks()
```

## Creating Test Sets

### Building RAG Test Datasets

```python
def creating_test_sets():
    """Creating test datasets for RAG evaluation."""

    print("\n\nCreating Test Sets:\n")

    print("=" * 60)
    print("\nTest Set Requirements:\n")

    print("""
A good RAG test set needs:

1. DIVERSE QUERIES
   • Different query types (factual, how-to, comparison)
   • Various difficulty levels
   • Edge cases

2. GROUND TRUTH
   • Expected answers (gold standard)
   • Relevant document IDs
   • Relevance scores (0-3 scale)

3. COVERAGE
   • Cover main use cases
   • Include failure modes
   • Test edge cases

4. SIZE
   • Minimum: 50-100 queries
   • Good: 500-1000 queries
   • Excellent: 5000+ queries
""")

    print("=" * 60)
    print("\nCreation Methods:\n")

    print("""
1. MANUAL CREATION

   Pros:
     ✓ High quality
     ✓ Domain-specific
     ✓ Captures nuances

   Cons:
     ✗ Time-consuming
     ✗ Expensive
     ✗ Limited scale

2. SEMI-AUTOMATED (LLM-GENERATED)

   Approach:
     • Use LLM to generate queries from documents
     • Human review and filter
     • Add ground truth answers

   Pros:
     ✓ Fast
     ✓ Scalable
     ✓ Diverse

   Cons:
     ✗ May have errors
     ✗ Needs human review

3. PRODUCTION DATA

   Approach:
     • Sample real user queries
     • Have experts annotate answers
     • Filter for quality

   Pros:
     ✓ Real user behavior
     ✓ Relevant to actual use

   Cons:
     ✗ May be biased
     ✗ Privacy concerns
     ✗ Needs cleanup

4. SYNTHETIC (FULLY AUTOMATED)

   Approach:
     • LLM generates queries + answers from docs
     • Automated quality filtering
     • No human review

   Pros:
     ✓ Very fast
     ✓ Large scale
     ✓ Cheap

   Cons:
     ✗ Lower quality
     ✗ May miss edge cases
     ✗ Needs validation
""")

    code = '''
class TestSetGenerator:
    """Generate test sets for RAG evaluation."""

    def __init__(self, llm_call):
        self.llm_call = llm_call

    def generate_queries_from_document(
        self,
        document: str,
        num_queries: int = 5
    ) -> List[Dict]:
        """
        Generate queries from a document.

        Args:
            document: Source document
            num_queries: Number of queries to generate

        Returns:
            List of generated query-answer pairs
        """
        prompt = f"""Generate {num_queries} diverse questions that can be answered using this document. For each question, also provide the answer.

Document:
{document[:1000]}...

Generate questions of different types:
- Factual (What, Who, When)
- How-to (How to...)
- Comparison (Compare X and Y)
- Explanation (Why, How)

Format as:
Q1: [question]
A1: [answer]
Q2: [question]
A2: [answer]
...
"""

        response = self.llm_call(prompt)

        # Parse questions and answers
        qa_pairs = []
        lines = response.strip().split('\\n')

        current_q = None
        for line in lines:
            if line.startswith('Q'):
                current_q = line.split(':', 1)[1].strip()
            elif line.startswith('A') and current_q:
                current_a = line.split(':', 1)[1].strip()
                qa_pairs.append({
                    'query': current_q,
                    'answer': current_a,
                    'source_document': document[:500]
                })
                current_q = None

        return qa_pairs

    def generate_test_set_from_corpus(
        self,
        documents: List[str],
        queries_per_doc: int = 3
    ) -> List[Dict]:
        """
        Generate test set from document corpus.

        Args:
            documents: List of documents
            queries_per_doc: Queries to generate per document

        Returns:
            Complete test set
        """
        test_set = []

        for i, doc in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}...")

            qa_pairs = self.generate_queries_from_document(doc, queries_per_doc)

            for qa in qa_pairs:
                test_set.append({
                    'query': qa['query'],
                    'answer': qa['answer'],
                    'relevant_docs': [i],  # Document index
                    'source_document': doc
                })

        return test_set

    def enhance_with_relevance_judgments(
        self,
        test_set: List[Dict],
        all_documents: List[str],
        rag_system
    ) -> List[Dict]:
        """
        Add relevance judgments for more documents.

        Args:
            test_set: Initial test set
            all_documents: All documents in corpus
            rag_system: RAG system to retrieve candidates

        Returns:
            Enhanced test set with relevance judgments
        """
        for test_case in test_set:
            query = test_case['query']

            # Retrieve top candidates
            results = rag_system.retrieve(query, top_k=20)

            # Ask LLM to judge relevance
            candidate_docs = [r['text'] for r in results]

            prompt = f"""Rate the relevance of each document to the query on a scale of 0-3:
- 3: Perfectly relevant, directly answers query
- 2: Highly relevant, contains key information
- 1: Somewhat relevant, tangentially related
- 0: Not relevant

Query: {query}

Documents:
"""
            for i, doc in enumerate(candidate_docs):
                prompt += f"\\nDoc {i+1}: {doc[:200]}..."

            prompt += "\\n\\nProvide ratings as: 1:3, 2:1, 3:0, ..."

            response = self.llm_call(prompt)

            # Parse ratings
            relevance_scores = {}
            for rating in response.split(','):
                if ':' in rating:
                    doc_id, score = rating.split(':')
                    doc_id = results[int(doc_id.strip())-1]['id']
                    relevance_scores[doc_id] = int(score.strip())

            test_case['relevance_scores'] = relevance_scores

        return test_set

    def validate_test_set(
        self,
        test_set: List[Dict]
    ) -> Dict:
        """
        Validate test set quality.

        Returns:
            Validation report
        """
        issues = []

        # Check for duplicates
        queries = [t['query'] for t in test_set]
        duplicates = len(queries) - len(set(queries))
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate queries")

        # Check for missing fields
        required_fields = ['query', 'answer', 'relevant_docs']
        for i, test_case in enumerate(test_set):
            for field in required_fields:
                if field not in test_case:
                    issues.append(f"Test case {i} missing field: {field}")

        # Check query diversity
        avg_length = np.mean([len(q.split()) for q in queries])

        # Check answer quality
        short_answers = sum(1 for t in test_set if len(t['answer'].split()) < 3)

        return {
            'num_queries': len(test_set),
            'num_duplicates': duplicates,
            'avg_query_length': avg_length,
            'short_answers': short_answers,
            'issues': issues,
            'valid': len(issues) == 0
        }


# Example usage
generator = TestSetGenerator(llm_call=call_gpt)

# Generate from documents
documents = [
    "Python is a high-level programming language created by Guido van Rossum...",
    "Machine learning is a subset of artificial intelligence...",
    # ... more documents
]

test_set = generator.generate_test_set_from_corpus(
    documents,
    queries_per_doc=3
)

# Enhance with relevance judgments
test_set = generator.enhance_with_relevance_judgments(
    test_set,
    documents,
    rag_system=my_rag
)

# Validate
validation_report = generator.validate_test_set(test_set)

print("Test Set Validation:")
print(f"  Queries: {validation_report['num_queries']}")
print(f"  Duplicates: {validation_report['num_duplicates']}")
print(f"  Avg query length: {validation_report['avg_query_length']:.1f} words")
print(f"  Valid: {validation_report['valid']}")

if not validation_report['valid']:
    print(f"  Issues: {validation_report['issues']}")

# Save test set
import json

with open('rag_test_set.json', 'w') as f:
    json.dump(test_set, f, indent=2)

print(f"\\nSaved {len(test_set)} test cases to rag_test_set.json")
'''

    print(code)

creating_test_sets()
```

## Debugging Poor Retrieval

### Diagnosing Retrieval Issues

```python
def debugging_retrieval():
    """Debugging retrieval problems."""

    print("\n\nDebugging Poor Retrieval:\n")

    print("=" * 60)
    print("\nCommon Retrieval Issues:\n")

    issues = """
Issue                        | Symptom                | Root Cause
-----------------------------|------------------------|---------------------------
Low Recall                   | Missing relevant docs  | • Chunks too large/small
                             |                        | • Poor embedding model
                             |                        | • Query-doc mismatch

Low Precision                | Too many irrelevant    | • No metadata filtering
                             |                        | • Embedding not specific
                             |                        | • Need reranking

Inconsistent Results         | Different results      | • ANN index approximate
                             | each time              | • Need more neighbors (k)

Slow Retrieval               | High latency           | • Index not optimized
                             |                        | • Too many docs to rerank
                             |                        | • Network latency

Poor for Specific Terms      | Misses exact matches   | • Dense-only retrieval
                             |                        | • Need hybrid (add BM25)
"""

    print(issues)

    print("\n" + "=" * 60)
    print("\nDiagnostic Process:\n")

    print("""
Step 1: ISOLATE THE PROBLEM

  Run query → Check what's retrieved

  Questions:
    • Are any relevant docs retrieved?
    • What's the rank of first relevant doc?
    • What irrelevant docs appear?
    • Does changing query help?

Step 2: ANALYZE EMBEDDINGS

  Check query-doc similarity:
    • Embed query
    • Embed relevant docs
    • Calculate similarity
    • Compare to retrieved docs

  Low similarity to relevant docs → Embedding problem

Step 3: INSPECT CHUNKS

  Look at actual chunk content:
    • Are chunks too short? (missing context)
    • Are chunks too long? (too much noise)
    • Do chunks have good boundaries?
    • Is preprocessing removing important info?

Step 4: TEST ALTERNATIVES

  Try variations:
    • Different embedding model
    • Different chunk size (256 → 512)
    • Different chunking strategy (fixed → recursive)
    • Add metadata filtering
    • Add reranking

  Measure improvement on test set

Step 5: FIX ROOT CAUSE

  Based on findings, apply fix:
    • Adjust chunking parameters
    • Switch embedding model
    • Add hybrid search
    • Improve preprocessing
    • Add reranking
""")

    code = '''
class RetrievalDebugger:
    """Debug retrieval issues."""

    def __init__(self, rag_system, embedding_model):
        self.rag = rag_system
        self.model = embedding_model

    def diagnose_query(
        self,
        query: str,
        relevant_doc_ids: Set[str],
        top_k: int = 10
    ) -> Dict:
        """
        Diagnose retrieval for a single query.

        Args:
            query: Query to debug
            relevant_doc_ids: Known relevant documents
            top_k: How many docs to retrieve

        Returns:
            Diagnostic report
        """
        # Retrieve
        results = self.rag.retrieve(query, top_k=top_k)
        retrieved_ids = [r['id'] for r in results]

        # Check if relevant docs retrieved
        relevant_retrieved = [
            doc_id for doc_id in retrieved_ids
            if doc_id in relevant_doc_ids
        ]

        # Find rank of first relevant
        first_relevant_rank = None
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_doc_ids:
                first_relevant_rank = rank
                break

        # Embed query and documents
        query_emb = self.model.encode(query)

        # Get embeddings for relevant docs (if available)
        relevant_docs_info = []
        for doc_id in relevant_doc_ids:
            doc = self.rag.get_document(doc_id)
            if doc:
                doc_emb = self.model.encode(doc['text'])
                similarity = float(np.dot(query_emb, doc_emb))
                relevant_docs_info.append({
                    'id': doc_id,
                    'similarity': similarity,
                    'retrieved': doc_id in retrieved_ids,
                    'rank': retrieved_ids.index(doc_id) + 1 if doc_id in retrieved_ids else None
                })

        # Sort by similarity
        relevant_docs_info.sort(key=lambda x: x['similarity'], reverse=True)

        # Analyze retrieved docs
        retrieved_analysis = []
        for rank, result in enumerate(results, 1):
            doc_emb = self.model.encode(result['text'])
            similarity = float(np.dot(query_emb, doc_emb))

            retrieved_analysis.append({
                'rank': rank,
                'id': result['id'],
                'score': result.get('score', 0),
                'similarity': similarity,
                'relevant': result['id'] in relevant_doc_ids,
                'preview': result['text'][:100]
            })

        # Compute metrics
        recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0
        precision = len(relevant_retrieved) / top_k if top_k > 0 else 0

        # Diagnosis
        issues = []

        if recall < 0.5:
            issues.append("LOW RECALL: Many relevant docs not retrieved")

        if precision < 0.3:
            issues.append("LOW PRECISION: Many irrelevant docs retrieved")

        if first_relevant_rank and first_relevant_rank > 5:
            issues.append(f"POOR RANKING: First relevant doc at rank {first_relevant_rank}")

        # Check if embedding is the issue
        avg_relevant_sim = np.mean([d['similarity'] for d in relevant_docs_info])
        avg_retrieved_sim = np.mean([d['similarity'] for d in retrieved_analysis])

        if avg_relevant_sim < 0.5:
            issues.append("LOW EMBEDDING SIMILARITY: Query and relevant docs have low similarity")

        if avg_retrieved_sim > avg_relevant_sim:
            issues.append("RANKING PROBLEM: Retrieved docs more similar than relevant docs")

        return {
            'query': query,
            'recall': recall,
            'precision': precision,
            'first_relevant_rank': first_relevant_rank,
            'relevant_docs_analysis': relevant_docs_info,
            'retrieved_docs_analysis': retrieved_analysis,
            'issues': issues,
            'recommendations': self._generate_recommendations(issues)
        }

    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []

        for issue in issues:
            if 'LOW RECALL' in issue:
                recommendations.append("• Try larger chunk size or more overlap")
                recommendations.append("• Consider different embedding model")
                recommendations.append("• Check if preprocessing removes important info")

            if 'LOW PRECISION' in issue:
                recommendations.append("• Add cross-encoder reranking")
                recommendations.append("• Use metadata filtering to narrow search")
                recommendations.append("• Try hybrid search (dense + sparse)")

            if 'POOR RANKING' in issue:
                recommendations.append("• Add reranking stage")
                recommendations.append("• Try different similarity metric")
                recommendations.append("• Increase retrieval k, then rerank")

            if 'LOW EMBEDDING SIMILARITY' in issue:
                recommendations.append("• Try different embedding model")
                recommendations.append("• Use query expansion")
                recommendations.append("• Check if query and docs use different vocabulary")

        return list(set(recommendations))  # Deduplicate

    def compare_chunking_strategies(
        self,
        document: str,
        query: str,
        strategies: Dict[str, callable]
    ) -> Dict:
        """
        Compare different chunking strategies.

        Args:
            document: Document to chunk
            query: Query to test
            strategies: Dict of strategy_name -> chunking_function

        Returns:
            Comparison results
        """
        results = {}

        query_emb = self.model.encode(query)

        for strategy_name, chunking_fn in strategies.items():
            # Chunk document
            chunks = chunking_fn(document)

            # Embed chunks
            chunk_embs = self.model.encode(chunks)

            # Calculate similarities
            similarities = [float(np.dot(query_emb, chunk_emb)) for chunk_emb in chunk_embs]

            # Get best chunk
            best_idx = np.argmax(similarities)

            results[strategy_name] = {
                'num_chunks': len(chunks),
                'max_similarity': similarities[best_idx],
                'avg_similarity': np.mean(similarities),
                'best_chunk': chunks[best_idx][:200],
                'chunk_sizes': [len(c) for c in chunks]
            }

        return results


# Example usage
debugger = RetrievalDebugger(rag_system, embedding_model)

# Diagnose a failing query
query = "How to install Python on Windows?"
relevant_docs = {'doc1', 'doc5', 'doc12'}

diagnosis = debugger.diagnose_query(query, relevant_docs, top_k=10)

print("Retrieval Diagnosis:")
print(f"  Recall: {diagnosis['recall']:.2f}")
print(f"  Precision: {diagnosis['precision']:.2f}")
print(f"  First relevant rank: {diagnosis['first_relevant_rank']}")

print("\\nIssues Found:")
for issue in diagnosis['issues']:
    print(f"  • {issue}")

print("\\nRecommendations:")
for rec in diagnosis['recommendations']:
    print(f"  {rec}")

print("\\nRelevant Docs Analysis:")
for doc in diagnosis['relevant_docs_analysis'][:5]:
    print(f"  {doc['id']}: similarity={doc['similarity']:.3f}, retrieved={doc['retrieved']}, rank={doc['rank']}")

# Compare chunking strategies
strategies = {
    'fixed_200': lambda doc: fixed_size_chunking(doc, 200, 20),
    'fixed_500': lambda doc: fixed_size_chunking(doc, 500, 50),
    'recursive': lambda doc: recursive_chunking(doc, 500),
    'semantic': lambda doc: semantic_chunking(doc, embedding_model, 0.8)
}

comparison = debugger.compare_chunking_strategies(
    document=sample_doc,
    query=query,
    strategies=strategies
)

print("\\nChunking Strategy Comparison:")
for strategy, metrics in comparison.items():
    print(f"  {strategy}:")
    print(f"    Chunks: {metrics['num_chunks']}")
    print(f"    Max similarity: {metrics['max_similarity']:.3f}")
    print(f"    Avg similarity: {metrics['avg_similarity']:.3f}")
'''

    print(code)

debugging_retrieval()
```

## Debugging Poor Generation

### Diagnosing Generation Issues

```python
def debugging_generation():
    """Debugging generation problems."""

    print("\n\nDebugging Poor Generation:\n")

    print("=" * 60)
    print("\nCommon Generation Issues:\n")

    issues = """
Issue                    | Symptom                    | Root Cause
-------------------------|----------------------------|---------------------------
Hallucination            | Makes up information       | • LLM not grounded to context
                         |                            | • Context not clear
                         |                            | • System prompt weak

Incomplete Answers       | Misses key information     | • Context missing info
                         |                            | • LLM stops too early
                         |                            | • Max tokens too low

Off-Topic Responses      | Doesn't answer query       | • Poor retrieval (wrong context)
                         |                            | • Prompt doesn't emphasize query
                         |                            | • LLM misunderstands

Copy-Paste from Context  | Verbatim context quotes    | • Prompt says "use context"
                         |                            | • LLM too cautious
                         |                            | • Need "synthesize" instruction

Inconsistent Quality     | Sometimes good, sometimes  | • Temperature too high
                         | bad                        | • Context quality varies
                         |                            | • Non-deterministic model
"""

    print(issues)

    print("\n" + "=" * 60)
    print("\nDiagnostic Process:\n")

    print("""
Step 1: SEPARATE RETRIEVAL FROM GENERATION

  Test with CORRECT context:
    • Manually provide perfect context
    • Generate answer
    • Is answer good now?

  If YES → Retrieval is the problem
  If NO → Generation is the problem

Step 2: ANALYZE FAITHFULNESS

  Check if answer uses context:
    • Does answer include facts from context?
    • Does answer add information not in context?
    • Does answer contradict context?

  Hallucination → Need stronger grounding prompt

Step 3: TEST PROMPT VARIATIONS

  Try different system prompts:
    • More explicit: "Only use provided context"
    • Add examples: "Good answer: ..., Bad answer: ..."
    • Change tone: "You are a helpful assistant..."

  Measure which works best

Step 4: CHECK CONTEXT QUALITY

  Review retrieved context:
    • Does it contain answer to query?
    • Is there contradictory information?
    • Is context too long/short?
    • Is important info buried?

  Fix retrieval if needed

Step 5: TUNE LLM PARAMETERS

  Adjust generation:
    • Temperature (0 = deterministic, 1 = creative)
    • Max tokens (ensure complete answers)
    • Top-p (nucleus sampling)

  Lower temperature often helps
""")

    code = '''
class GenerationDebugger:
    """Debug generation issues."""

    def __init__(self, llm_call, evaluator):
        self.llm_call = llm_call
        self.evaluator = evaluator

    def test_with_perfect_context(
        self,
        query: str,
        perfect_context: str,
        expected_answer: str
    ) -> Dict:
        """
        Test generation with perfect context.

        This isolates generation from retrieval.

        Args:
            query: User query
            perfect_context: Manually curated context
            expected_answer: Ground truth answer

        Returns:
            Generation analysis
        """
        # Generate answer
        prompt = f"""Answer the question using ONLY the provided context.

Context:
{perfect_context}

Question: {query}

Answer:"""

        answer = self.llm_call(prompt)

        # Evaluate
        faithfulness = self.evaluator.faithfulness_llm(
            answer=answer,
            context=perfect_context,
            llm_call=self.llm_call
        )

        relevance = self.evaluator.answer_relevance_semantic(query, answer)

        correctness = self.evaluator.answer_correctness(
            answer=answer,
            ground_truth=expected_answer,
            method='semantic'
        )

        # Analyze
        issues = []

        if faithfulness['faithfulness_score'] < 0.7:
            issues.append("HALLUCINATION: Answer not faithful to context")

        if relevance < 0.6:
            issues.append("OFF-TOPIC: Answer doesn't address query")

        if correctness < 0.6:
            issues.append("INCORRECT: Answer doesn't match ground truth")

        return {
            'query': query,
            'context': perfect_context,
            'generated_answer': answer,
            'expected_answer': expected_answer,
            'faithfulness': faithfulness['faithfulness_score'],
            'relevance': relevance,
            'correctness': correctness,
            'issues': issues
        }

    def test_prompt_variations(
        self,
        query: str,
        context: str,
        prompts: Dict[str, str],
        ground_truth: str
    ) -> Dict:
        """
        Test different system prompts.

        Args:
            query: User query
            context: Retrieved context
            prompts: Dict of prompt_name -> system_prompt
            ground_truth: Expected answer

        Returns:
            Comparison of prompts
        """
        results = {}

        for prompt_name, system_prompt in prompts.items():
            full_prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""

            answer = self.llm_call(full_prompt)

            # Evaluate
            faithfulness = self.evaluator.faithfulness_llm(
                answer, context, self.llm_call
            )['faithfulness_score']

            relevance = self.evaluator.answer_relevance_semantic(query, answer)

            correctness = self.evaluator.answer_correctness(
                answer, ground_truth, 'semantic'
            )

            results[prompt_name] = {
                'answer': answer,
                'faithfulness': faithfulness,
                'relevance': relevance,
                'correctness': correctness,
                'avg_score': (faithfulness + relevance + correctness) / 3
            }

        # Sort by avg score
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        )

        return sorted_results

    def analyze_hallucination(
        self,
        query: str,
        context: str,
        answer: str
    ) -> Dict:
        """
        Detailed hallucination analysis.

        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer

        Returns:
            Hallucination analysis
        """
        # Ask LLM to identify claims in answer
        claims_prompt = f"""List all factual claims in this answer:

Answer: {answer}

Claims (one per line):"""

        claims_response = self.llm_call(claims_prompt)
        claims = [c.strip() for c in claims_response.split('\\n') if c.strip()]

        # Check each claim against context
        claim_analysis = []

        for claim in claims:
            verify_prompt = f"""Is this claim supported by the context?

Context:
{context}

Claim: {claim}

Answer: YES, NO, or PARTIAL
Explanation:"""

            verification = self.llm_call(verify_prompt)

            supported = 'YES' in verification.upper()

            claim_analysis.append({
                'claim': claim,
                'supported': supported,
                'verification': verification
            })

        # Calculate hallucination rate
        hallucinated = sum(1 for c in claim_analysis if not c['supported'])
        hallucination_rate = hallucinated / len(claims) if claims else 0

        return {
            'query': query,
            'answer': answer,
            'num_claims': len(claims),
            'num_hallucinated': hallucinated,
            'hallucination_rate': hallucination_rate,
            'claim_analysis': claim_analysis
        }

    def test_llm_parameters(
        self,
        query: str,
        context: str,
        ground_truth: str,
        parameter_sets: List[Dict]
    ) -> Dict:
        """
        Test different LLM parameters.

        Args:
            query: User query
            context: Retrieved context
            ground_truth: Expected answer
            parameter_sets: List of parameter dicts (temperature, max_tokens, etc.)

        Returns:
            Comparison of parameter sets
        """
        results = {}

        prompt = f"""Answer the question using the provided context.

Context:
{context}

Question: {query}

Answer:"""

        for i, params in enumerate(parameter_sets):
            config_name = f"config_{i+1}"

            # Generate with these parameters
            answer = self.llm_call(prompt, **params)

            # Evaluate
            correctness = self.evaluator.answer_correctness(
                answer, ground_truth, 'semantic'
            )

            results[config_name] = {
                'parameters': params,
                'answer': answer,
                'correctness': correctness,
                'answer_length': len(answer.split())
            }

        return results


# Example usage
debugger = GenerationDebugger(llm_call=call_gpt, evaluator=generation_eval)

# Test with perfect context
query = "Who created Python?"
perfect_context = "Python was created by Guido van Rossum in 1991 at CWI in the Netherlands."
expected_answer = "Guido van Rossum created Python in 1991."

result = debugger.test_with_perfect_context(query, perfect_context, expected_answer)

print("Generation Test (Perfect Context):")
print(f"  Generated: {result['generated_answer']}")
print(f"  Expected: {result['expected_answer']}")
print(f"  Faithfulness: {result['faithfulness']:.2f}")
print(f"  Relevance: {result['relevance']:.2f}")
print(f"  Correctness: {result['correctness']:.2f}")

if result['issues']:
    print(f"  Issues: {result['issues']}")

# Test prompt variations
prompts = {
    'basic': "Answer the question using the context.",
    'strict': "Answer ONLY using information from the context. Do not add any external knowledge.",
    'helpful': "You are a helpful assistant. Use the context to answer the question accurately.",
    'synthesize': "Synthesize information from the context to provide a complete answer."
}

prompt_results = debugger.test_prompt_variations(
    query=query,
    context=perfect_context,
    prompts=prompts,
    ground_truth=expected_answer
)

print("\\nPrompt Comparison:")
for prompt_name, metrics in prompt_results.items():
    print(f"  {prompt_name}: avg_score={metrics['avg_score']:.2f}")

# Analyze hallucination
hallucination_analysis = debugger.analyze_hallucination(
    query=query,
    context=perfect_context,
    answer="Python was created by Guido van Rossum in 1991 and is the most popular language."
)

print(f"\\nHallucination Analysis:")
print(f"  Claims: {hallucination_analysis['num_claims']}")
print(f"  Hallucinated: {hallucination_analysis['num_hallucinated']}")
print(f"  Rate: {hallucination_analysis['hallucination_rate']:.1%}")

# Test parameters
parameter_sets = [
    {'temperature': 0.0, 'max_tokens': 100},
    {'temperature': 0.3, 'max_tokens': 100},
    {'temperature': 0.7, 'max_tokens': 100},
]

param_results = debugger.test_llm_parameters(
    query=query,
    context=perfect_context,
    ground_truth=expected_answer,
    parameter_sets=parameter_sets
)

print("\\nParameter Comparison:")
for config, metrics in param_results.items():
    print(f"  {config} (temp={metrics['parameters']['temperature']}):")
    print(f"    Correctness: {metrics['correctness']:.2f}")
    print(f"    Length: {metrics['answer_length']} words")
'''

    print(code)

debugging_generation()
```

## Continuous Evaluation

### Monitoring RAG in Production

```python
def continuous_evaluation():
    """Continuous evaluation and monitoring."""

    print("\n\nContinuous Evaluation:\n")

    print("=" * 60)
    print("\nWhy Continuous Evaluation?\n")

    print("""
RAG systems can degrade over time:

1. DATA DRIFT
   • Documents change/updated
   • New content added
   • Old content removed

   Effect: Retrieval finds outdated info

2. QUERY DRIFT
   • Users ask different questions
   • New topics emerge
   • Language patterns change

   Effect: System not optimized for new queries

3. MODEL DRIFT
   • Embedding model updated
   • LLM changes (API updates)
   • Infrastructure changes

   Effect: Different behavior

4. PERFORMANCE DEGRADATION
   • Database gets larger (slower)
   • More concurrent users
   • Infrastructure issues

   Effect: Higher latency

Solution: Monitor continuously and catch issues early
""")

    print("=" * 60)
    print("\nMetrics to Monitor:\n")

    metrics = """
Metric                  | Frequency      | Alert Threshold
------------------------|----------------|------------------
Retrieval Quality       | Daily          | Drop > 5%
  • Precision@5         |                |
  • Recall@10           |                |
  • NDCG@10             |                |

Generation Quality      | Daily          | Drop > 5%
  • Faithfulness        |                |
  • Answer Relevance    |                |

Latency                 | Real-time      | > 5s (p95)
  • Retrieval time      |                |
  • Generation time     |                |
  • Total time          |                |

User Signals            | Real-time      | Drop > 10%
  • Click-through rate  |                |
  • User ratings        |                |
  • Session duration    |                |

System Health           | Real-time      | Critical
  • Error rate          |                | > 1%
  • API failures        |                | > 0.1%
  • Timeout rate        |                | > 5%
"""

    print(metrics)

    code = '''
class ContinuousEvaluator:
    """Monitor RAG system continuously."""

    def __init__(
        self,
        rag_system,
        test_set: List[Dict],
        alert_callback=None
    ):
        self.rag = rag_system
        self.test_set = test_set
        self.alert_callback = alert_callback
        self.baseline_metrics = None

    def establish_baseline(self) -> Dict:
        """
        Run initial evaluation to establish baseline.

        Returns:
            Baseline metrics
        """
        evaluator = RAGEvaluator(self.rag, retrieval_eval, generation_eval)

        results = evaluator.evaluate_dataset(self.test_set)

        self.baseline_metrics = results['aggregated_metrics']

        print("Baseline Established:")
        for metric, value in self.baseline_metrics.items():
            if 'mean' in metric:
                print(f"  {metric}: {value:.3f}")

        return self.baseline_metrics

    def run_evaluation(self) -> Dict:
        """
        Run evaluation on test set.

        Returns:
            Current metrics
        """
        evaluator = RAGEvaluator(self.rag, retrieval_eval, generation_eval)

        results = evaluator.evaluate_dataset(self.test_set, verbose=False)

        return results['aggregated_metrics']

    def compare_to_baseline(
        self,
        current_metrics: Dict,
        threshold: float = 0.05
    ) -> Dict:
        """
        Compare current metrics to baseline.

        Args:
            current_metrics: Current evaluation metrics
            threshold: Alert if drop > threshold (e.g., 0.05 = 5%)

        Returns:
            Comparison with alerts
        """
        if not self.baseline_metrics:
            raise ValueError("No baseline established. Run establish_baseline() first.")

        comparison = {}
        alerts = []

        for metric in self.baseline_metrics:
            if 'mean' in metric:
                baseline = self.baseline_metrics[metric]
                current = current_metrics.get(metric, 0)

                change = current - baseline
                pct_change = (change / baseline) * 100 if baseline > 0 else 0

                comparison[metric] = {
                    'baseline': baseline,
                    'current': current,
                    'change': change,
                    'pct_change': pct_change
                }

                # Check for degradation
                if change < -threshold:
                    alerts.append({
                        'metric': metric,
                        'severity': 'HIGH' if pct_change < -10 else 'MEDIUM',
                        'message': f"{metric} dropped by {abs(pct_change):.1f}%"
                    })

        # Trigger alerts
        if alerts and self.alert_callback:
            self.alert_callback(alerts)

        return {
            'comparison': comparison,
            'alerts': alerts
        }

    def monitor_latency(
        self,
        queries: List[str],
        num_samples: int = 100
    ) -> Dict:
        """
        Monitor system latency.

        Args:
            queries: Sample queries to test
            num_samples: Number of samples

        Returns:
            Latency statistics
        """
        import time

        latencies = []

        for _ in range(num_samples):
            query = np.random.choice(queries)

            start = time.time()
            _ = self.rag.query(query)
            latency = time.time() - start

            latencies.append(latency)

        return {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': np.max(latencies)
        }

    def daily_evaluation_job(self):
        """
        Daily evaluation job.

        Run this as a cron job or scheduled task.
        """
        print(f"Running daily evaluation ({datetime.now()})...")

        # Run evaluation
        current_metrics = self.run_evaluation()

        # Compare to baseline
        comparison = self.compare_to_baseline(current_metrics)

        # Log results
        self._log_metrics(current_metrics, comparison)

        # Send report
        self._send_report(comparison)

        print("Daily evaluation complete.")

    def _log_metrics(self, metrics: Dict, comparison: Dict):
        """Log metrics to monitoring system."""
        # Log to your monitoring system (Datadog, CloudWatch, etc.)
        # Example:
        import json

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'comparison': comparison
        }

        # Write to file or send to monitoring service
        with open('rag_metrics.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\\n')

    def _send_report(self, comparison: Dict):
        """Send daily report."""
        if comparison['alerts']:
            # Send alert email/Slack message
            message = "RAG System Alert:\\n\\n"
            for alert in comparison['alerts']:
                message += f"[{alert['severity']}] {alert['message']}\\n"

            print(message)
            # send_alert(message)


# Example usage
continuous_eval = ContinuousEvaluator(
    rag_system=my_rag,
    test_set=test_queries,
    alert_callback=send_slack_alert
)

# Initial setup (run once)
continuous_eval.establish_baseline()

# Daily job (schedule with cron or similar)
continuous_eval.daily_evaluation_job()

# Monitor latency
latency_stats = continuous_eval.monitor_latency(
    queries=[t['query'] for t in test_queries],
    num_samples=100
)

print("Latency Statistics:")
print(f"  Mean: {latency_stats['mean']:.2f}s")
print(f"  P95: {latency_stats['p95']:.2f}s")
print(f"  P99: {latency_stats['p99']:.2f}s")
'''

    print(code)

continuous_evaluation()
```

## Optimization Workflows

### Iterative RAG Improvement

```python
def optimization_workflows():
    """Systematic RAG optimization process."""

    print("\n\nOptimization Workflows:\n")

    print("=" * 60)
    print("\nOptimization Process:\n")

    print("""
┌────────────────────────────────────────────────┐
│ 1. MEASURE: Establish baseline                │
│    • Run evaluation on test set                │
│    • Record all metrics                        │
│    • Identify weakest areas                    │
└────────────────┬───────────────────────────────┘
                 ↓
┌────────────────────────────────────────────────┐
│ 2. ANALYZE: Find root causes                   │
│    • Debug failing queries                     │
│    • Check retrieval quality                   │
│    • Review generation faithfulness            │
│    • Identify patterns in failures             │
└────────────────┬───────────────────────────────┘
                 ↓
┌────────────────────────────────────────────────┐
│ 3. HYPOTHESIZE: What might help?               │
│    • Larger chunks?                            │
│    • Different embedding model?                │
│    • Add reranking?                            │
│    • Better prompts?                           │
└────────────────┬───────────────────────────────┘
                 ↓
┌────────────────────────────────────────────────┐
│ 4. EXPERIMENT: Test one change                 │
│    • Implement change                          │
│    • Re-run evaluation                         │
│    • Compare to baseline                       │
│    • Statistical significance test             │
└────────────────┬───────────────────────────────┘
                 ↓
┌────────────────────────────────────────────────┐
│ 5. DECIDE: Keep or revert?                     │
│    • Improvement > threshold? → Keep           │
│    • No improvement? → Revert                  │
│    • Mixed results? → Investigate more         │
└────────────────┬───────────────────────────────┘
                 ↓
                Loop back to step 2
""")

    print("=" * 60)
    print("\nCommon Optimizations:\n")

    optimizations = """
Problem                      | Solution                    | Expected Gain
-----------------------------|-----------------------------|--------------
Low retrieval recall         | • Larger chunk size         | +10-20% recall
                             | • More overlap              |
                             | • Different embedding       |

Low retrieval precision      | • Add reranking             | +15-30% precision
                             | • Metadata filtering        |
                             | • Hybrid search             |

Hallucination               | • Stronger grounding prompt  | +20-40% faithfulness
                             | • Lower temperature          |
                             | • Add faithfulness check     |

Slow retrieval              | • Optimize index (HNSW)      | 2-5x speedup
                             | • Reduce dimensions          |
                             | • Cache popular queries      |

Inconsistent quality        | • Add reranking              | +10-20% consistency
                             | • Lower temperature          |
                             | • Ensemble retrievers        |
"""

    print(optimizations)

    code = '''
class RAGOptimizer:
    """Systematic RAG optimization."""

    def __init__(
        self,
        rag_system,
        test_set: List[Dict],
        evaluator
    ):
        self.rag = rag_system
        self.test_set = test_set
        self.evaluator = evaluator
        self.optimization_history = []

    def run_experiment(
        self,
        experiment_name: str,
        config_changes: Dict,
        baseline_metrics: Dict = None
    ) -> Dict:
        """
        Run an optimization experiment.

        Args:
            experiment_name: Name of experiment
            config_changes: Configuration changes to test
            baseline_metrics: Baseline metrics (if not provided, will compute)

        Returns:
            Experiment results
        """
        print(f"\\nRunning experiment: {experiment_name}")
        print(f"Changes: {config_changes}")

        # Get baseline if not provided
        if baseline_metrics is None:
            print("  Computing baseline...")
            baseline_metrics = self.evaluator.evaluate_dataset(self.test_set)['aggregated_metrics']

        # Apply changes
        print("  Applying changes...")
        self._apply_config_changes(config_changes)

        # Evaluate with changes
        print("  Evaluating...")
        new_metrics = self.evaluator.evaluate_dataset(self.test_set)['aggregated_metrics']

        # Compare
        comparison = self._compare_metrics(baseline_metrics, new_metrics)

        # Statistical test
        significance = self._test_significance(
            baseline_results=self.test_set,
            new_results=self.test_set,
            metric='precision@10_mean'
        )

        result = {
            'experiment_name': experiment_name,
            'config_changes': config_changes,
            'baseline_metrics': baseline_metrics,
            'new_metrics': new_metrics,
            'comparison': comparison,
            'significance': significance,
            'recommendation': self._make_recommendation(comparison, significance)
        }

        # Save to history
        self.optimization_history.append(result)

        return result

    def optimize_retrieval(self, baseline_metrics: Dict) -> List[Dict]:
        """
        Run a series of retrieval optimizations.

        Args:
            baseline_metrics: Starting point

        Returns:
            List of experiment results
        """
        experiments = []

        # Experiment 1: Larger chunks
        exp1 = self.run_experiment(
            "Larger chunks (500 → 750)",
            {'chunk_size': 750, 'chunk_overlap': 75},
            baseline_metrics
        )
        experiments.append(exp1)

        # If improved, keep and use as new baseline
        if exp1['recommendation'] == 'ADOPT':
            baseline_metrics = exp1['new_metrics']
        else:
            # Revert
            self._apply_config_changes({'chunk_size': 500, 'chunk_overlap': 50})

        # Experiment 2: Add reranking
        exp2 = self.run_experiment(
            "Add cross-encoder reranking",
            {'use_reranking': True, 'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2'},
            baseline_metrics
        )
        experiments.append(exp2)

        if exp2['recommendation'] == 'ADOPT':
            baseline_metrics = exp2['new_metrics']
        else:
            self._apply_config_changes({'use_reranking': False})

        # Experiment 3: Hybrid search
        exp3 = self.run_experiment(
            "Add sparse retrieval (hybrid)",
            {'use_hybrid': True, 'alpha': 0.6},
            baseline_metrics
        )
        experiments.append(exp3)

        return experiments

    def optimize_generation(self, baseline_metrics: Dict) -> List[Dict]:
        """
        Run a series of generation optimizations.

        Args:
            baseline_metrics: Starting point

        Returns:
            List of experiment results
        """
        experiments = []

        # Experiment 1: Lower temperature
        exp1 = self.run_experiment(
            "Lower temperature (0.7 → 0.3)",
            {'temperature': 0.3},
            baseline_metrics
        )
        experiments.append(exp1)

        # Experiment 2: Stronger system prompt
        new_prompt = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so. Do not add any external knowledge."""

        exp2 = self.run_experiment(
            "Stronger system prompt",
            {'system_prompt': new_prompt},
            baseline_metrics
        )
        experiments.append(exp2)

        return experiments

    def _compare_metrics(
        self,
        baseline: Dict,
        new: Dict
    ) -> Dict:
        """Compare metrics and calculate improvements."""
        comparison = {}

        for metric in baseline:
            if 'mean' in metric:
                base_value = baseline[metric]
                new_value = new[metric]

                improvement = new_value - base_value
                pct_improvement = (improvement / base_value * 100) if base_value > 0 else 0

                comparison[metric] = {
                    'baseline': base_value,
                    'new': new_value,
                    'improvement': improvement,
                    'pct_improvement': pct_improvement
                }

        return comparison

    def _test_significance(
        self,
        baseline_results: List,
        new_results: List,
        metric: str,
        alpha: float = 0.05
    ) -> Dict:
        """Test statistical significance of improvement."""
        from scipy import stats

        # This is simplified - in practice, would need actual per-query results
        # For demo, generate synthetic data
        baseline_scores = np.random.normal(0.7, 0.1, len(baseline_results))
        new_scores = np.random.normal(0.75, 0.1, len(new_results))

        t_stat, p_value = stats.ttest_rel(baseline_scores, new_scores)

        return {
            'p_value': p_value,
            'significant': p_value < alpha,
            't_statistic': t_stat
        }

    def _make_recommendation(
        self,
        comparison: Dict,
        significance: Dict
    ) -> str:
        """Recommend whether to adopt changes."""

        # Count improvements
        improvements = sum(
            1 for m in comparison.values()
            if m['improvement'] > 0
        )

        total = len(comparison)

        # Decision logic
        if significance['significant'] and improvements / total > 0.6:
            return "ADOPT"
        elif improvements / total > 0.8:
            return "ADOPT_WITH_CAUTION"
        else:
            return "REJECT"

    def _apply_config_changes(self, changes: Dict):
        """Apply configuration changes to RAG system."""
        # Update RAG system configuration
        for key, value in changes.items():
            setattr(self.rag, key, value)

    def generate_report(self) -> str:
        """Generate optimization report."""
        report = "RAG Optimization Report\\n"
        report += "=" * 60 + "\\n\\n"

        for i, exp in enumerate(self.optimization_history, 1):
            report += f"Experiment {i}: {exp['experiment_name']}\\n"
            report += f"  Recommendation: {exp['recommendation']}\\n"

            # Show key improvements
            for metric, values in exp['comparison'].items():
                if 'precision' in metric or 'recall' in metric or 'ndcg' in metric:
                    report += f"  {metric}: {values['baseline']:.3f} → {values['new']:.3f} "
                    report += f"({values['pct_improvement']:+.1f}%)\\n"

            report += "\\n"

        return report


# Example usage
optimizer = RAGOptimizer(
    rag_system=my_rag,
    test_set=test_queries,
    evaluator=rag_evaluator
)

# Establish baseline
print("Computing baseline...")
baseline = rag_evaluator.evaluate_dataset(test_queries)['aggregated_metrics']

print("\\nBaseline Metrics:")
for metric, value in baseline.items():
    if 'mean' in metric:
        print(f"  {metric}: {value:.3f}")

# Run retrieval optimizations
print("\\n" + "=" * 60)
print("RETRIEVAL OPTIMIZATIONS")
print("=" * 60)

retrieval_experiments = optimizer.optimize_retrieval(baseline)

# Run generation optimizations
print("\\n" + "=" * 60)
print("GENERATION OPTIMIZATIONS")
print("=" * 60)

generation_experiments = optimizer.optimize_generation(baseline)

# Generate report
print("\\n" + "=" * 60)
report = optimizer.generate_report()
print(report)
'''

    print(code)

optimization_workflows()
```

## Summary

**Key Concepts**:

1. **Evaluation is essential** - can't improve what you don't measure
2. **Evaluate both stages** - retrieval and generation separately
3. **Use multiple metrics** - NDCG, faithfulness, relevance, etc.
4. **Create good test sets** - diverse queries with ground truth
5. **Debug systematically** - isolate problems, test fixes
6. **Monitor continuously** - catch degradation early
7. **Optimize iteratively** - measure → analyze → experiment → repeat

**RAG Evaluation Framework**:

```
┌────────────────────────────────────────────┐
│ RETRIEVAL METRICS                          │
│                                            │
│ • Precision@K: Relevant in top-K?          │
│ • Recall@K: Found all relevant?            │
│ • NDCG@K: Quality of ranking?              │
│ • MRR: Rank of first relevant?             │
│ • MAP: Average precision                   │
└────────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────────┐
│ GENERATION METRICS                         │
│                                            │
│ • Faithfulness: Grounded in context?       │
│ • Answer Relevance: Addresses query?       │
│ • Correctness: Matches ground truth?       │
│ • Context Relevance: Good retrieval?       │
└────────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────────┐
│ END-TO-END METRICS                         │
│                                            │
│ • Overall accuracy                         │
│ • User satisfaction                        │
│ • Latency                                  │
│ • Cost per query                           │
└────────────────────────────────────────────┘
```

**Essential Metrics**:

```
Primary Metrics (always measure):
  • NDCG@10: Retrieval ranking quality
  • Faithfulness: Answer grounded in context
  • Answer Relevance: Addresses query

Secondary Metrics (optional):
  • Recall@10: Coverage of relevant docs
  • MRR: Speed to first relevant
  • Correctness: If ground truth available
  • Latency: Response time
```

**Evaluation Frameworks**:

```
RAGAS (Recommended):
  • LLM-based evaluation
  • No ground truth needed
  • Comprehensive metrics
  • pip install ragas

LangChain:
  • Built-in evaluators
  • QA evaluation chain
  • Criteria-based scoring

TruLens:
  • Real-time monitoring
  • Dashboard for exploration
  • Production observability

Custom:
  • Full control
  • Domain-specific metrics
  • Integration with existing systems
```

**Test Set Creation**:

```
Approach                | Quality | Speed | Cost
------------------------|---------|-------|-------
Manual                  | ⭐⭐⭐⭐⭐  | ⭐     | $$$
Semi-automated (LLM)    | ⭐⭐⭐⭐   | ⭐⭐⭐⭐  | $$
Production data         | ⭐⭐⭐⭐   | ⭐⭐⭐   | $$
Fully synthetic         | ⭐⭐⭐    | ⭐⭐⭐⭐⭐ | $

Recommended: Semi-automated with human review

Requirements:
  • 50-100 queries minimum
  • 500-1000 for production
  • Diverse query types
  • Ground truth answers
  • Relevance judgments
```

**Debugging Workflow**:

```
Problem: Low retrieval quality

Step 1: Isolate retrieval
  → Test with known relevant docs
  → Are they retrieved?

Step 2: Check embeddings
  → Calculate query-doc similarity
  → Low similarity → Embedding issue

Step 3: Inspect chunks
  → Review chunk content
  → Too large/small?
  → Poor boundaries?

Step 4: Test alternatives
  → Different chunk size
  → Different embedding model
  → Add hybrid search

Step 5: Measure improvement
  → Re-run evaluation
  → Compare to baseline
  → Adopt if better
```

**Common Issues and Fixes**:

```
Issue: Low Recall
  → Increase chunk size (500 → 750)
  → Add more overlap (10% → 20%)
  → Try different embedding model

Issue: Low Precision
  → Add cross-encoder reranking
  → Use metadata filtering
  → Add hybrid search (dense + sparse)

Issue: Hallucination
  → Strengthen system prompt ("use ONLY context")
  → Lower temperature (0.7 → 0.3)
  → Add faithfulness check

Issue: Slow
  → Optimize index (HNSW parameters)
  → Cache popular queries
  → Batch processing
  → Use GPU

Issue: Inconsistent
  → Lower temperature
  → Add reranking
  → Use ensemble retrievers
```

**Continuous Monitoring**:

```
Metrics to Track (Daily):
  ✓ Retrieval precision/recall
  ✓ Generation faithfulness
  ✓ Answer relevance
  ✓ Latency (p95, p99)
  ✓ Error rate

Alert When:
  • Quality drops > 5%
  • Latency > 5s (p95)
  • Error rate > 1%

Response Plan:
  1. Check recent changes
  2. Run diagnostic queries
  3. Review logs
  4. Rollback if needed
  5. Fix root cause
```

**Optimization Process**:

```
1. Measure baseline
   → Run full evaluation
   → Record all metrics

2. Analyze failures
   → Debug worst queries
   → Find patterns

3. Hypothesize fixes
   → What might help?
   → Prioritize by impact

4. Experiment
   → Change ONE thing
   → Re-evaluate
   → Compare statistically

5. Decide
   → Improvement > 5%? → Adopt
   → No improvement? → Revert
   → Update baseline

6. Repeat
   → Iterate until satisfied
```

**Best Practices**:

1. **Start with baseline** - know where you are
2. **Test one change at a time** - isolate effects
3. **Use statistical tests** - ensure real improvement
4. **Maintain test set** - keep updated with new queries
5. **Monitor production** - catch issues early
6. **Document everything** - track what worked
7. **Iterate continuously** - always improving

**Recommended Metrics by Use Case**:

```
Customer Support:
  • Answer Relevance (primary)
  • Faithfulness (critical for trust)
  • Latency (< 3s)

Documentation Search:
  • Precision@5 (show right docs)
  • NDCG@10 (ranking quality)
  • Coverage (recall)

Question Answering:
  • Correctness (if ground truth available)
  • Faithfulness (must be accurate)
  • MRR (speed to answer)

Research Assistant:
  • Recall (don't miss sources)
  • Diversity (show multiple perspectives)
  • Context Relevance
```

## Next Steps

- Apply to production in [Application Patterns](../application_patterns/)
- Review [RAG Architecture](rag-architecture.md) for system design
- Master [Retrieval Strategies](retrieval-strategies.md) for better search
- Study [Reranking and Fusion](reranking-fusion.md) for quality improvements
- Continue learning in [RLHF and Alignment](../rlhf-and-alignment/)
