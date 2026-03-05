# LLM Concepts

## About This Section

This section explores the concepts and capabilities that emerge in large language models. LLMs aren't just bigger versions of previous models -- at scale, they exhibit qualitatively new behaviors like in-context learning, chain-of-thought reasoning, and instruction following. Understanding these concepts is essential for working effectively with modern language AI.

This section bridges the gap between understanding language models as architectures and understanding LLMs as general-purpose reasoning engines. You'll learn what emerges at scale, how to leverage these capabilities, and what fundamental limitations remain.

## Contents

### [Scale and Emergent Abilities](scale-and-emergence.md)

How size changes everything. Covers scaling laws (bigger is predictably better), emergent abilities that appear at sufficient scale (few-shot learning, complex reasoning), the compute and data requirements for frontier models, and the implications of scale for capabilities and access. Understanding scale helps you appreciate both the power and the resource intensity of LLMs.

### [In-Context Learning](in-context-learning.md)

Learning from examples in the prompt without parameter updates. Covers zero-shot, one-shot, and few-shot learning, how to construct effective examples, why in-context learning works, and the relationship between model size and ICL capability. This is the mechanism that enables LLMs to adapt to new tasks through prompting alone.

### [Chain-of-Thought Reasoning](chain-of-thought.md)

Improving reasoning through intermediate steps. Covers chain-of-thought prompting, step-by-step reasoning, problem decomposition, self-consistency (sampling multiple reasoning paths), and when COT helps vs hurts. Understanding COT is critical for complex reasoning tasks.

### [Instruction Tuning and Alignment](instruction-tuning.md)

Making models follow instructions and align with human intent. Covers instruction datasets (FLAN, T0), instruction tuning methods, how instruction tuning differs from traditional finetuning, zero-shot task generalization, and the connection to RLHF. Instruction tuning makes models more useful and controllable.

### [LLM Capabilities and Limitations](capabilities-limitations.md)

What LLMs can and cannot do. Covers knowledge and factual recall, reasoning abilities and failures, creativity and generation, understanding of instructions, common failure modes (hallucination, inconsistency, poor arithmetic), and the boundaries of current capabilities. Realistic understanding of capabilities is essential for building reliable systems.
