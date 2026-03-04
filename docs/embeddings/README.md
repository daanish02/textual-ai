# Embeddings

## About This Section

This section covers the representation learning revolution in NLP. Embeddings transform discrete symbols (words, sentences) into continuous vector spaces where semantic relationships are captured geometrically. This shift from sparse, hand-crafted features to dense, learned representations enabled the neural NLP era.

Understanding embeddings is fundamental to modern NLP -- they're the input to transformers, the basis for semantic search, and the foundation for transfer learning. You'll learn how different embedding methods work, what properties they capture, and how to choose and use embeddings effectively.

## Contents

### [Word Embeddings](word-embeddings.md)

Learning distributed representations of words. Covers Word2Vec (CBOW and Skip-gram architectures), GloVe (global matrix factorization), fastText (subword embeddings), training objectives, and the remarkable properties of embedding spaces (analogies, similarity). Understanding word embeddings reveals how meaning can be captured geometrically.

### [Sentence and Document Embeddings](sentence-embeddings.md)

Moving beyond word-level to sentence and document representations. Covers simple approaches (averaging word embeddings), learned methods (Doc2Vec), modern approaches (SBERT, Universal Sentence Encoder), and the challenges of composing word meanings into sentence meanings. Sentence embeddings enable semantic search and document similarity at scale.

### [Contextual Embeddings](contextual-embeddings.md)

Embeddings that change based on context. Covers ELMo (contextual representations from language models), the shift from static to dynamic representations, and why context matters ("bank" in "river bank" vs "savings bank"). Contextual embeddings bridge to transformer-based models where every token representation depends on surrounding context.

### [Embedding Spaces and Properties](embedding-spaces.md)

Understanding what embeddings capture and how to work with them. Covers semantic similarity, analogies, compositionality, bias in embeddings, dimensionality, and practical considerations (how to choose embeddings, fine-tuning strategies, evaluation). Understanding embedding properties helps you use them effectively and identify limitations.
