# Textual AI

## Purpose and Philosophy

This repository is a **comprehensive knowledge base** for natural language processing and textual AI -- from classical NLP concepts to modern large language models, from embeddings to retrieval-augmented generation. It focuses on **conceptual understanding** that transcends specific libraries or model families.

The focus is on:

- **NLP fundamentals**: tokenization, text representation, linguistic concepts
- **Classical NLP**: n-grams, TF-IDF, parsing, information extraction
- **Embeddings**: word embeddings, sentence embeddings, representation learning for text
- **LLM concepts**: transformer language models, prompting, fine-tuning, instruction tuning
- **Retrieval-augmented generation (RAG)**: combining retrieval with generation, vector databases
- **Evaluation**: metrics, benchmarks, human evaluation, understanding LLM capabilities and limitations
- **Practical workflows**: prompt engineering, LLM application patterns, system design with LLMs

This repository is **concept-first and tool-agnostic**. While examples may reference OpenAI API, Hugging Face, LangChain, or other tools, the emphasis is on **universal principles** for working with language and text.

!!! quote

    "Language is the dress of thought." -- Samuel Johnson

Natural language processing is about teaching machines to understand and generate human language -- the most natural interface for human-computer interaction. This repository is your guide to that frontier.

## Who This Repository Is For

This repository is designed for:

- **NLP engineers** deepening their understanding of language models and NLP systems
- **ML engineers** specializing in text and language applications
- **Software engineers** building LLM-powered applications
- **Researchers** exploring language understanding and generation
- **Lifelong learners** building comprehensive NLP expertise

This assumes ML fundamentals and progresses toward advanced NLP techniques and LLM application design.

## How to Use This Repository

### As a Learning System

- **Build NLP foundations**: Understand text representation, tokenization, and classical NLP
- **Progress to modern NLP**: Embeddings → Transformers → LLMs → RAG
- **Learn prompting and evaluation**: Master prompt engineering and LLM evaluation
- **Design systems**: Apply concepts to build production NLP/LLM systems
- **Experiment**: Test prompting strategies, embeddings, and retrieval systems

### As a Reference

- **Prompt engineering**: Reference effective prompting patterns
- **Evaluation metrics**: Choose appropriate metrics for NLP tasks
- **RAG patterns**: Design retrieval-augmented systems
- **LLM capabilities**: Understand what LLMs can and cannot do

### As a Living Document

- **Track LLM evolution**: As models improve, update understanding and patterns
- **Document prompt patterns**: Capture effective prompting strategies
- **Add evaluation insights**: Record findings from evaluating LLM systems
- **Capture lessons learned**: Document insights from production NLP systems

## Repository Structure

This repository is organized to **mirror the progression of NLP mastery** and to **support efficient navigation**:

```bash
textual-ai-lab/
├── overview/                        # NLP evolution and philosophy
├── fundamentals/                    # Tokenization, preprocessing, linguistic concepts
├── classical_nlp/                   # N-grams, TF-IDF, topic modeling
├── embeddings/                      # Word2Vec, sentence embeddings, representation learning
├── language_models/                 # LM fundamentals, BERT vs GPT, pretraining
├── llm_concepts/                    # Scale, emergent abilities, in-context learning
├── prompt_engineering/              # Prompting strategies and optimization
├── retrieval_augmented_generation/  # RAG architecture and patterns
├── rlhf_and_alignment/              # RLHF, reward modeling, safety
├── evaluation/                      # Metrics, benchmarks, human evaluation
├── application_patterns/            # Classification, summarization, QA workflows
├── experiments/                     # Experiments related to NLP and Textual AI
├── notes/                           # Personal learnings
└── resources.md                     # Resources for NLP and Textual AI
```

## Table of Contents

### [Overview](overview/)

- Evolution of NLP: rule-based → statistical → neural → LLMs
- Understanding vs generation
- Challenges in NLP: ambiguity, context, commonsense
- The LLM paradigm shift

### [Fundamentals](fundamentals/)

- **Tokenization**: word, subword (BPE, WordPiece), character tokenization
- **Text preprocessing**: lowercasing, stemming, lemmatization, stopword removal
- **Part-of-speech tagging**: syntactic categories, tagging algorithms
- **Named entity recognition**: entity types, sequence labeling
- **Parsing**: dependency parsing, constituency parsing, syntax trees
- **Linguistic concepts**: morphology, syntax, semantics, pragmatics

### [Classical NLP](classical-nlp/)

- **N-grams**: language modeling, smoothing, perplexity
- **TF-IDF**: term weighting, document similarity, information retrieval
- **Bag-of-words**: text representation, limitations
- **Topic modeling**: LDA, LSA, discovering topics in corpora
- **Information extraction**: relation extraction, event extraction
- **Text classification**: Naive Bayes, logistic regression, feature engineering

### [Embeddings](embeddings/)

- **Word2Vec**: CBOW, Skip-gram, learning word representations
- **GloVe**: global matrix factorization, word analogies
- **fastText**: subword embeddings, handling OOV words
- **Sentence embeddings**: averaging, SBERT, universal sentence encoder
- **Contextual embeddings**: ELMo, contextualized representations
- **Embedding spaces**: semantic similarity, analogies, bias in embeddings

### [Language Models](language-models/)

- **Autoregressive models**: GPT-style, left-to-right generation, causal masking
- **Masked models**: BERT-style, bidirectional context, masked language modeling
- **Encoder-decoder**: T5, sequence-to-sequence, translation
- **Pretraining objectives**: MLM, CLM, NSP, span corruption
- **Fine-tuning**: task-specific adaptation, full fine-tuning vs LoRA
- **Transfer learning in NLP**: pretrain-finetune paradigm

### [LLM Concepts](llm-concepts/)

- **Scale effects**: emergent abilities, scaling laws, bigger models
- **In-context learning**: learning from examples in the prompt
- **Chain-of-thought**: reasoning through intermediate steps
- **Instruction tuning**: following instructions, FLAN, InstructGPT
- **RLHF**: reinforcement learning from human feedback, alignment
- **Capabilities**: what LLMs can do, reasoning, knowledge, creativity
- **Limitations**: hallucination, factuality, reasoning failures, context length

### [Prompt Engineering](prompt-engineering/)

- **Zero-shot prompting**: task description only
- **Few-shot prompting**: providing examples, in-context learning
- **Chain-of-thought**: step-by-step reasoning, let's think step by step
- **Structured output**: JSON, XML, formatting outputs
- **Prompt decomposition**: breaking complex tasks into steps
- **Self-consistency**: sampling multiple outputs, voting
- **Prompt optimization**: iterative refinement, evaluation-driven improvement

### [Retrieval-Augmented Generation](retrieval-augmented-generation/)

- **RAG architecture**: retrieval → augmentation → generation
- **Vector databases**: Pinecone, Weaviate, Chroma, FAISS
- **Embedding models**: choosing embeddings for retrieval
- **Chunking strategies**: document splitting, overlap, semantic chunking
- **Retrieval strategies**: dense retrieval, sparse retrieval, hybrid search
- **Reranking**: improving retrieval quality, cross-encoder reranking
- **RAG evaluation**: retrieval quality, generation quality, end-to-end evaluation

### [Evaluation](evaluation/)

- **Traditional metrics**: BLEU, ROUGE, METEOR, BERTScore
- **LLM evaluation**: accuracy on benchmarks, human evaluation, pairwise comparison
- **Benchmarks**: MMLU, HellaSwag, TruthfulQA, BigBench
- **Task-specific evaluation**: classification metrics, generation quality
- **Human evaluation**: relevance, coherence, factuality, fluency
- **Failure analysis**: understanding when and why systems fail
- **Adversarial testing**: edge cases, stress testing, robustness

### [Application Patterns](application-patterns/)

- **Classification**: sentiment analysis, intent detection, topic classification
- **Summarization**: extractive vs abstractive, single-doc vs multi-doc
- **Information extraction**: named entities, relations, structured extraction
- **Question answering**: extractive QA, generative QA, open-domain QA
- **Generation**: creative writing, code generation, content creation
- **Conversational AI**: dialogue systems, chatbots, multi-turn interaction
- **Semantic search**: finding relevant documents, similarity search

## Learning Principles

- **Language is Complex:** Language is ambiguous, context-dependent, and nuanced. Appreciate the difficulty of NLP and the limitations of current systems.
- **Prompting is an Art and Science:** Effective prompting requires iteration, evaluation, and understanding of LLM behavior. Develop systematic prompting skills.
- **Evaluation is Critical:** LLMs can produce fluent but incorrect outputs. Rigorous evaluation is essential for building reliable systems.
- **RAG Bridges Knowledge Gaps:** LLMs have knowledge cutoffs and hallucinate. RAG grounds generation in retrieved documents, improving factuality.
- **Understand Trade-offs:** Bigger models vs smaller models, latency vs quality, cost vs capability. Every system design involves trade-offs.

## Contribution to Your Growth

This repository is your **comprehensive guide to NLP and textual AI mastery**. It is:

- An **NLP reference** for classical and modern techniques
- A **prompt engineering guide** for effective LLM usage
- A **RAG design handbook** for retrieval-augmented systems
- A **living document** tracking the rapid evolution of language AI

Mastering NLP and LLMs will enable you to build sophisticated language understanding and generation systems.
