# Proteus-Attention

**An adaptive attention architecture for massive context, born from necessity.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/proteus-attention.svg)](https://badge.fury.io/py/proteus-attention)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Zen-Sherbert/Proteus-Attention/ci.yml?branch=main)](https://github.com/Zen-Sherbert/Proteus-Attention/actions)

---

I built Proteus because standard attention is a greedy, clunky black box. It hits a quadratic wall and dies. I needed a system that was smarter, more efficient, and could run on hardware I actually owned.

Not knowing if there were other options, I tossed my hat in the ring and redesigned it from the ground up.

The result is an attention mechanism that can be configured to process **over 1,000,000 tokens on a single 16GB consumer GPU**. It turns what was a supercomputing problem into a desktop reality. And the best part? It was developed and benchmarked on a **Radeon 7800XT running ROCm inside a Fedora Silverblue container.** No H100s required.

## Proof, Not Promises: The Benchmark

Standard attention fails catastrophically where Proteus begins to stretch its legs. The following benchmark was run on a **16GB Radeon 7800XT with ROCm**, showing a **64x increase in context length capacity** over standard attention on the same hardware.

| Seq Len | Model                     | Latency (ms) | Peak VRAM (MB) | Status     |
| :------ | :------------------------ | :----------- | :------------- | :--------- |
| 16,384  | Standard Attention        | 41.03        | 4,194          | OK         |
| **32,768**  | **Standard Attention**        | -            | >16,000        | **OOM**    |
| ---     | ---                       | ---          | ---            | ---        |
| 32,768  | Proteus (BF16)            | 58.79        | 632            | **OK**     |
| 131,072 | Proteus (BF16)            | 186.93       | 2,438          | **OK**     |
| 524,288 | Proteus (BF16)            | 813.75       | 10,257         | **OK**     |
| **1,000,000**| **Proteus (BF16)**  | **538.72**   | **5,658**      | **OK***    |

***A Note on Reaching 1M Tokens:** Achieving the absolute maximum context length is a trade-off. The 1M token benchmark was achieved by disabling the **DNA (Dynamic Network Affinity)** system. DNA maintains semantic prototype vectors for intelligent routing, which requires additional VRAM. By default, DNA is enabled for maximum model quality. For tasks requiring absolute maximum sequence length, it can be disabled as a configuration choice.*

## How It Works

This isn't just another linear attention model. It's a hybrid, adaptive system built on a few core principles born from solving real-world problems.

### 1. The Protean Slider: An Automatic Transmission for Attention
I was tired of being locked into one algorithm. The solution is a dynamic `alpha` value that acts like a slider, smoothly transitioning the attention strategy based on sequence length. No jarring switches, no training instability, just fluid adaptation.

```
Sequence Length:  Short ───────> Medium ────────> Extreme
Alpha Value:      0.0   ────────> 0.5    ────────> 1.0
Attention Mode:   Dense         Dynamic Sparse   Fully Linear (Flux)
```

### 2. One Kernel to Rule Them All
Juggling multiple, specialized kernels was a nightmare. Proteus uses a single, unified Triton kernel for all modes. This "Flux Kernel" reconfigures itself on-the-fly based on the `alpha` value. It has two internal pathways, one for sub-quadratic attention and one for fixed-cost linear attention, giving us the best of both worlds without the overhead.

### 3. Intelligent Hybrid Routing
Relying on a single learned router is a recipe for model collapse. Proteus uses a two-pronged system:
*   **The Learned Router:** A fast, context-aware dispatcher.
*   **DNA (Dynamic Network Affinity):** A "semantic fingerprint" for each of the model's internal "gates." As tokens pass through a gate, the gate's DNA vector is updated to be more like them, creating a long-term memory of what kind of information it "likes."

Like Sauron's Nazgûl, the DNA system seeks out the incoming tokens, using deep semantic similarity to guide them. This robust, dual-system approach prevents collapse and leads to much smarter routing.

### 4. No Expert Required: Automated Performance Tuning
Manually tuning GPU block sizes is absurd. Proteus includes a built-in auto-tuner that benchmarks different configurations on your specific hardware at runtime. It finds the optimal setup (e.g., `(32, 64, 16)` for my 7800XT) and caches it, guaranteeing you get maximum performance out-of-the-box without needing to be a GPU architect.

## Taming the Beast: How to Train a Model

An adaptive architecture needs an adaptive training strategy. Throwing a 1M token sequence at an untrained model will fail. The key is a curriculum that teaches the model to use its own dynamic features.

**The "Context Mastery" Curriculum (Recommended):**
This is the most effective strategy. It's a dynamic curriculum that expands the context as the model learns.

1.  **Start Short:** Begin training with a short sequence length (e.g., `seq_len=256`, `alpha=0.0`).
2.  **Monitor Loss:** Train until your validation loss begins to plateau.
3.  **Increase Difficulty:** When a plateau is detected, slightly increase both the sequence length and the `alpha` value.
4.  **Repeat:** Continue this process, allowing the model to master each level of contextual complexity before graduating to the next.

This method ensures the model learns short-range syntax before tackling long-range semantics, leading to the best final performance.
There are training examples, in the examples folder.

## Installation

```bash
pip install proteus-attention
```

## Quick Start: A Complete Example

Proteus is a drop-in replacement. All the complexity is handled internally.

```python
import torch
from proteus_attention import AttentionBlock as ProteusBlock
from proteus_attention import ModelConfig

# 1. Define your model's configuration.

# This config is tuned for high quality with DNA enabled.
config_quality = ModelConfig(
    d_model=1024,
    n_head=16,               # Total physical heads
    attn_h_total=64,         # Total "virtual" heads for specialization
    attn_gates=256,          # Routing gates (where DNA lives)
    attn_mode="auto",
    attn_dna_enable=True, # Default: ON for max intelligence
    attn_linear_switch_ctx=8192
)

# For tasks requiring the absolute maximum context,
# you can trade some routing intelligence for VRAM.
config_max_context = ModelConfig(
    d_model=1024,
    n_head=16,
    attn_mode="auto",
    attn_dna_enable=False # Switched OFF for max sequence length
)

# 2. Use the ProteusBlock as a drop-in replacement for your Transformer block.
model = ProteusBlock(config_quality) # Or config_max_context

# 3. That's it.
# It can handle this.
x = torch.randn(1, 1_000_000, 1024, device='cuda', dtype=torch.bfloat16)
output = model(x)
```

## The Road Ahead

This is the beginning. My focus is on making Proteus robust, accessible, and even more powerful.

*   **Improve the Auto-Tuner:** Evolve the tuner to be "3D-aware," selecting optimal block sizes based not just on hardware, but also on the sequence length and sparsity of the current batch.
*   **Official CUDA Kernel:** Implement a dedicated CUDA kernel to complement the Triton path, offering another layer of performance and hardware compatibility.
*   **Agnostic Verification:** Formally benchmark on a wide range of hardware, including NVIDIA GPUs (4090, A100) and other accelerators, to solidify Proteus as a truly hardware-agnostic solution.
*   **Logging & Introspection:** Build an optional logging bus so users can see what's happening behind the scenes: watch DNA evolve, monitor gate usage, and truly understand their model's internal state.

## Citation

If you use this work, please cite the repository.

```bibtex
@software{proteus_attention_2025,
  author = {Scott Dietz},
  title = {{Proteus-Attention: An Adaptive, Long-Context Attention Mechanism}},
  url = {https://github.com/Zen-Sherbert/Proteus-Attention},
  year = {2025}
}
```

## License

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
