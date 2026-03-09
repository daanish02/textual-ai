# RLHF and Alignment

## About This Section

This section covers aligning language models with human values and intent. While pretraining teaches models to predict text, it doesn't ensure they're helpful, harmless, or honest. Alignment techniques like RLHF (Reinforcement Learning from Human Feedback) and Constitutional AI shape model behavior to match human preferences and follow instructions safely.

Understanding alignment is critical for building responsible AI systems. You'll learn how models are aligned, what trade-offs exist between capability and safety, and ongoing challenges in ensuring AI systems behave as intended.

## Contents

### [The Alignment Problem](alignment-problem.md)

Why aligning AI systems with human intent is challenging. Covers the gap between prediction objectives and human values, specification problems (defining what we want), capability vs alignment, risks of misalignment, and why alignment matters more as models become more capable. Understanding the problem motivates the solutions.

### [Reinforcement Learning from Human Feedback (RLHF)](rlhf.md)

The primary technique for aligning modern LLMs. Covers the RLHF pipeline (supervised finetuning → reward modeling → RL optimization), collecting human preference data, training reward models, PPO for policy optimization, and why RLHF works. RLHF is how ChatGPT and similar models are made helpful and safe.

### [Instruction Following and Helpfulness](instruction-following.md)

Teaching models to understand and execute user intent. Covers instruction datasets, supervised finetuning on instructions, measuring instruction following capability, handling ambiguous instructions, and balancing helpfulness with safety. Instruction following makes models more useful as assistants.

### [Safety and Harmlessness](safety-harmlessness.md)

Preventing models from producing harmful outputs. Covers defining harm, red teaming and adversarial testing, safety training, guardrails and safety systems (input/output filtering, behavioral constraints), content filters, refusal strategies, jailbreaking and prompt injection attacks, and trade-offs between capability and safety. Safety is essential for deployed systems.

### [Constitutional AI and Self-Improvement](constitutional-ai.md)

Using models to critique and improve their own outputs. Covers Constitutional AI principles, self-critique loops, AI feedback vs human feedback, scaling alignment with model capability, and recursive improvement. Constitutional AI offers a path toward more scalable alignment.

### [Honesty and Calibration](honesty-calibration.md)

Making models truthful and well-calibrated. Covers hallucination mitigation, uncertainty quantification, teaching models to say "I don't know", calibration (confidence matching accuracy), and evaluation of truthfulness. Honest, well-calibrated models are more trustworthy.
