# Proteus-Attention

**An Adaptive Attention Architecture for Extreme Long-Context Modeling.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/proteus-attention.svg)](https://badge.fury.io/py/proteus-attention)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Zen-Sherbert/Proteus-Attention/ci.yml?branch=main)](https://github.com/Zen-Sherbert/Proteus-Attention/actions)

---

Standard Transformers can't read a whole book. Their attention mechanism fails under long sequences, hitting a quadratic wall of computation and memory. Proteus-Attention is a full-stack redesign that solves this problem.

It enables models to process context windows exceeding **500,000 tokens on a single consumer-grade GPU**, turning what was once a supercomputing challenge into a task for commodity hardware.

<br>

> ## Key Features
>
> -   **Extreme Scalability:** Processes 500k+ tokens on a single 16GB GPU, avoiding the OOM errors of standard attention.
> -   **The Protean Slider:** A revolutionary control system that fluidly adapts the attention mechanism—from dense quadratic to sparse sub-quadratic to linear-time—ensuring optimal performance at any sequence length.
> -   **Hardware-Optimized:** Includes a custom, JIT-compiled Triton kernel to achieve maximum performance on NVIDIA GPUs, with a highly efficient PyTorch fallback for other platforms.
> -   **Intelligent Routing:** Uses a novel **DNA (Dynamic Network Affinity)** system to create a semantic, long-term memory, enabling smarter, content-aware routing decisions.
> -   **Drop-in Ready:** Designed as a one-line replacement for standard attention layers in existing Transformer architectures.

<br>

## The Core Innovation: The Protean Slider

At the heart of Proteus is the **Protean Slider**, a system that acts like an automatic transmission for attention. It understands that no single algorithm is optimal for all situations and seamlessly shifts "gears" on-the-fly.

This is controlled by a dynamic `alpha` value that continuously morphs the attention mechanism:

```
Sequence Length:  Short ───────> Medium ────────> Extreme
Alpha Value:      0.0   ────────> 0.5    ────────> 1.0
Attention Mode:   Dense         Sparse           Linear (Fixed-Cost)
                  (Quadratic)   (Sub-Quadratic)  (Information Retrieval)
```

This automated, smooth transition eliminates performance cliffs and training instabilities, allowing a single model to be an expert at both short-sentence syntax and book-length semantic retrieval.

## Architectural Pillars

The architecture is built on five interlocking concepts powered by the Protean Slider:

1.  **Unified Kernel:** A single, custom Triton kernel serves as the engine for all operational modes. It is a templated, adaptive kernel that JIT-compiles itself based on the `alpha` value, ensuring maximum performance at every point on the curve.
2.  **Double Sparsity:** A multi-axis approach that prunes both heads (functional specialization) and tokens (salience), enabling the model to learn a dynamic computational budget.
3.  **DNA (Dynamic Network Affinity):** A persistent, semantic memory composed of learned "prototype" vectors. DNA provides a powerful, content-aware routing prior, helping the model establish critical long-range dependencies.
4.  **Gates vs. Heads: A Smarter Division of Labor:** We decouple the router's decision space ("gates") from the execution units ("heads"). This abstraction encourages a rich hierarchy where some heads become generalists and others become hyper-specialists.
5.  **Hybrid Routing:** A sophisticated decision system that blends the long-term semantic memory of DNA with a context-aware learned router, creating a robust mechanism that leverages both content-based and pattern-based information processing.

## Empirical Validation

The result: Proteus avoids the 'Out of Memory' errors that plague standard attention and maintains high performance at scales that were previously impossible on consumer hardware.

> **Hardware Note:** The following results were obtained on a single **16GB AMD Radeon RX 7800 XT** using the efficient PyTorch-native fallback path. Performance is expected to be significantly higher on NVIDIA GPUs leveraging the integrated Triton kernel (see our Vision & Roadmap).

| Sequence Length | Model                     | Latency (ms) | Peak VRAM (MB) |
| :-------------- | :------------------------ | :----------- | :------------- |
| 512             | Standard Attention        | 0.42         | 122            |
| 512             | Proteus (FP32)            | 3.53         | 69             |
| ---             | ---                       | ---          | ---            |
| 32,768          | Standard Attention        | **OOM**      | >16,000        |
| 32,768          | Proteus (FP32)            | 26.20        | 471            |
| ---             | ---                       | ---          | ---            |
| 524,288         | Standard Attention        | **OOM**      | -              |
| 524,288         | Proteus (FP32)            | 187.58       | 7,010          |

## Installation

```bash
pip install proteus-attention
```

## Usage: A Complete Example

Proteus is designed for effortless integration. All complexity is handled internally by the `ModelConfig`.

```python
import torch.nn as nn
from proteus_attention import CausalDynamicAttention as ProteusAttention
from proteus_attention import ModelConfig

# 1. Define your model's configuration
# The 'auto' mode will manage the Protean Slider for you.
config = ModelConfig(
    d_model=1024,
    n_head=16,
    attn_mode="auto",
    attn_linear_switch_ctx=16384 # The sequence length to fully transition to linear mode
)

# 2. Drop it into your Transformer block
# This is a one-line replacement for nn.MultiheadAttention or other custom layers.
self.attn = ProteusAttention(config)

# ... then use it like any other attention layer
output = self.attn(x)
```

## Training an Adaptive Model

The Protean Slider enables powerful and flexible training strategies.

#### Method 1: Sliding Context Curriculum (Recommended)
Train the model on a mix of sequence lengths to exercise the full range of the `alpha` slider. A good starting ratio is:
*   **70% Short Context Batches (<2k tokens):** Hones dense and sparse capabilities.
*   **20% Medium Context Batches (2k-16k tokens):** Ensures a stable transition.
*   **10% Long Context Batches (16k+ tokens):** Develops long-range retrieval.

#### Method 2: Loss-Gated Context Expansion (Advanced)
Create a universal model by linking the `alpha` slider to your validation loss. Start training at `alpha = 0.0`. When the loss plateaus, "up-shift" the `alpha` value and continue training. This allows the model to master one level of contextual complexity before moving to the next.

## Vision & Roadmap

Proteus is an active research project. Our immediate goals are:
*   **Achieve True Hardware Agnosticism:** Provide comprehensive benchmarks on NVIDIA hardware (e.g., A100, 4090) to validate the Triton kernel and publish official performance targets.
*   **Evaluate Model Quality:** Move beyond systems metrics to include perplexity scores on long-context language tasks (e.g., Project Gutenberg) to rigorously evaluate quality.
*   **Perfect the "Drop-in Ready" Experience:** Provide a rich set of pre-configured `ModelConfig` settings for common use-cases (e.g., "Max Quality," "Max Context Length," "Fastest Inference").
*   **Expand to New Modalities:** Apply the Proteus architecture to non-text data, such as high-resolution images, long-form audio, and genomic sequences.

## Citation

If you use Proteus-Attention in your work, please cite the repository:

```bibtex
@software{proteus_2025,
  author = {Scott Dietz},
  title = {{Proteus-Attention: An Adaptive Attention Architecture for Extreme Long-Context Modeling}},
  url = {https://github.com/Zen-Sherbert/Proteus-Attention},
  year = {2025}
}
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full text.
