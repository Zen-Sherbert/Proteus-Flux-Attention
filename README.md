# Proteus-Attention

**A Novel Attention Architecture for Extreme Long-Context Modeling on Commodity Hardware.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/proteus-attention.svg)](https://badge.fury.io/py/proteus-attention)
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-username/proteus-attention/ci.yml?branch=main)](https://github.com/your-username/proteus-attention/actions)

---

Proteus Attention is a new computational paradigm for Transformer models, designed to systematically solve the quadratic complexity bottleneck of standard attention. It is a full-stack, architecturally complete system that integrates high-level intelligent routing logic with a novel, adaptive low-level execution kernel.

Our empirical validation demonstrates the ability to process context windows exceeding **500,000 tokens on a single 16GB GPU**—a capability that redefines the accessibility of massive-scale AI. This is not merely a sparse approximation, but a principled redesign of the attention mechanism to be both computationally efficient and structurally intelligent.

## Architectural Pillars

The architecture is built on a series of synergistic, novel concepts:

1.  **The Protean Kernel:** A custom Triton kernel that functions as a single, unified engine for attention. It operates on a continuous performance curve, fluidly interpolating its computational graph from a sub-quadratic to a linear-time regime on-the-fly, ensuring optimal performance at every sequence length.
2.  **Double Sparsity:** A multi-axis approach that prunes both heads (functional specialization) and tokens (salience), enabling the model to learn a dynamic computational budget allocation.
3.  **DNA (Dynamic Network Alleles):** A persistent, semantic memory composed of learned "prototype" vectors. DNA provides a powerful, content-aware routing prior, allowing the model to make instantaneous, conceptually-grounded routing decisions.
4.  **Decoupled Gating:** An architectural abstraction where a large set of fine-grained "gates" (skills) are mapped to a smaller set of physical attention heads. This promotes a hierarchical division of labor, allowing for the natural emergence of both generalist and hyper-specialist heads.
5.  **Hybrid Routing:** A sophisticated decision-making system that blends the long-term semantic memory of DNA with a context-aware learned router, creating a robust mechanism that leverages the strengths of both content-based and pattern-based information processing.

## Empirical Validation

Proteus demonstrates a clear shift in asymptotic performance compared to standard scaled-dot-product attention. The following results were obtained on a single 16GB AMD Radeon RX 7800 XT.

| Sequence Length | Model                     | Latency (ms) | Peak VRAM (MB) |
| :-------------- | :------------------------ | :----------- | :------------- |
| 512             | Standard Attention        | 0.42         | 122            |
| 512             | Proteus (FP32)       | 3.53         | 69             |
| ---             | ---                       | ---          | ---            |
| 4,096           | Standard Attention        | 11.93        | 1,122          |
| 4,096           | Proteus (FP32)       | 8.72         | 334            |
| ---             | ---                       | ---          | ---            |
| 32,768          | Standard Attention        | OOM          | >16,000        |
| 32,768          | Proteus (FP32)       | 26.20        | 471            |
| ---             | ---                       | ---          | ---            |
| 524,288         | Standard Attention        | OOM          | -              |
| 524,288         | Proteus (FP32)       | 187.58       | 7,010          |

## Installation

The package can be installed via pip:
```bash
pip install proteus-attention
```

## Usage: Drop-in API

The system is designed for ease of integration via the `CausalGeneticMultiheadAttention` module, which mirrors the standard `torch.nn.MultiheadAttention` API.

**Standard Implementation:**
```python
import torch.nn as nn

# Standard Transformer Block
self.attn = nn.MultiheadAttention(
    embed_dim=d_model, 
    num_heads=n_head, 
    batch_first=True
)
```

**Proteus Implementation:**
```python
from proteus_attention import CausalGeneticMultiheadAttention

# A one-line, drop-in replacement
self.attn = CausalGeneticMultiheadAttention(
    embed_dim=d_model, 
    num_heads=n_head, 
    batch_first=True
)
```

## Architectural Deep Dive

<details>
<summary><b>1. The Protean Kernel and its Gradient Control System</b></summary>
<p>
The core execution engine is a custom Triton kernel. Its behavior is controlled by a high-level `alpha` parameter, which continuously interpolates the underlying attention pattern from a dense, local configuration (ideal for short, syntactically rich contexts) to a sparse, fixed-size candidate set (ideal for long-range, semantic retrieval). This smooth transition between computational regimes is critical for maintaining stable training gradients across a curriculum of varying sequence lengths, avoiding the performance cliffs and instabilities of discrete mode switching.
</p>
</details>

<details>
<summary><b>2. DNA as an Interpretable, Explicit Memory</b></summary>
<p>
The DNA prototypes are a set of learned vectors that are persistent across training and inference. They evolve via an Exponential Moving Average of the token embeddings routed to their corresponding gates, forming a compressed representation of the core semantic concepts the model has learned. This mechanism not only provides a powerful routing prior but also functions as an explicit, interpretable memory. By analyzing the DNA tensor and its relationship to known concept embeddings, it is possible to inspect and understand the model's internal "world model."
</p>
</details>

<details>
<summary><b>3. Emergent Specialization via Decoupled Gating</b></summary>
<p>
By decoupling the router's decision space ("gates") from the execution units ("heads"), the architecture allows for a more flexible and robust division of labor. This many-to-one mapping naturally encourages the formation of a head hierarchy. Some heads become generalists, serving as the target for many related gates, while others become hyper-specialists, activated by a single, specific gate. This mitigates the problem of "expert collapse" seen in other MoE systems and leads to a more efficient and capable final model.
</p>
</details>


## Replication & Training

To replicate the benchmark results or train a model using this architecture, please refer to the provided scripts:

*   **Performance Benchmark:** The `tinytoy.py` script provides the harness for performance and memory validation.
*   **Training Curriculum:** The `train_sliding_ctx.py` script provides a working example of a sliding-context curriculum, a training strategy designed to optimally leverage the adaptive nature of the Protean architecture.

## Future Work

Proteus is an active research project. Key areas for future investigation include:

*   **Kernel Fusion:** Fusing the candidate-generation logic into a dedicated CUDA/Triton kernel to further reduce computational overhead at extreme scales.
*   **Autonomous Control:** Developing the integrated `SparseHeadController` into a fully autonomous agent that can learn optimal sparsity policies via reinforcement learning.
*   **Analysis of Emergent Structures:** Investigating the conceptual clusters within the trained DNA space and the graph topology of the `gate_to_head` mapping to better understand the emergent knowledge structures within the model.
*   **Knowledge Transfer:** Exploring methods for transferring trained DNA and gate mappings between models to accelerate fine-tuning and enable novel forms of model composition.

## Citation

If you use Proteus-Attention in your work, please cite the repository:

```bibtex
@software{proteus_2025,
  author = {Scott Dietz},
  title = {{Proteus-Attention: A Novel Attention Architecture for Extreme Long-Context Modeling}},
  url = {https://github.com/Zen-Sherbert/Proteus-Attention},
  year = {2025}
}
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full text.
