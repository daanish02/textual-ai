# Language Models

## About This Section

This section covers language modeling fundamentals -- the task of predicting text and the architectures that enable it. Language models form the foundation of modern NLP: they provide the pretraining objective for transfer learning, enable generation, and serve as the basis for large language models.

Understanding different language model architectures (autoregressive vs masked, encoder vs decoder vs encoder-decoder) helps you choose the right model for your task, understand model capabilities and limitations, and grasp what modern LLMs are really doing under the hood.

## Contents

### [Language Modeling Fundamentals](lm-fundamentals.md)

The core task of language modeling and why it matters. Covers the language modeling objective, autoregressive vs masked approaches, perplexity as an evaluation metric, and why unsupervised pretraining on language modeling enables transfer learning. Understanding the basics helps you appreciate why language modeling is such a powerful pretraining task.

### [Autoregressive Models (GPT-style)](autoregressive-models.md)

Left-to-right language models for generation. Covers the GPT architecture, causal (unidirectional) attention, training objectives, generation strategies (greedy, sampling, beam search), and why autoregressive models excel at generation. These models predict the next token given all previous tokens.

### [Masked Models (BERT-style)](masked-models.md)

Bidirectional language models for understanding. Covers the BERT architecture, masked language modeling (MLM), bidirectional attention, the [CLS] and [SEP] tokens, and why masked models excel at understanding tasks (classification, extraction). These models predict masked tokens using context from both directions.

### [Encoder-Decoder Models](encoder-decoder-models.md)

Sequence-to-sequence architectures for transformation tasks. Covers the T5 architecture, text-to-text framework, span corruption, translation and summarization applications, and how encoder-decoder models combine the strengths of both paradigms. These models transform input sequences into output sequences.

### [Pretraining and Transfer Learning](pretraining-transfer.md)

How language models learn general knowledge that transfers to downstream tasks. Covers different pretraining objectives (MLM, CLM, NSP, span corruption), the pretrain-finetune paradigm, what knowledge is captured during pretraining, and why this approach revolutionized NLP. Understanding transfer learning explains why we rarely train language models from scratch anymore.
