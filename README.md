# Proteus Attention

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zen-Sherbert/Proteus-Attention/blob/main/TinyPlayground.ipynb)

Proteus is an experimental attention architecture designed for extreme-scale context processing. It implements a collection of techniques, centered around the **Dynamic Mixture-of-Attention-Heads (DMoAH)** mechanism, to manage computational and memory complexity as sequence lengths grow.

The system is built with custom Triton kernels for high performance on CUDA-enabled hardware, with a graceful fallback to a pure PyTorch implementation for compatibility.

## Core Components

*   **Dynamic Mixture-of-Attention-Heads (DMoAH):** A sparse attention mechanism that dynamically routes each token to a small subset of specialized attention heads, rather than using all heads for all tokens.
*   **Prototype-Guided Routing:** The head selection process is guided by learned "prototype" vectors that provide semantic priors to the router, improving selection quality.
*   **Adaptive Sparsity:** The system employs both head-level and token-level sparsity. It can automatically adjust the number of active heads and tokens based on sequence length and a target computational density.
*   **Multi-Mode Attention:** The attention module can operate in several modes and switch between them automatically:
    *   **Dense:** A standard attention pass used for very short sequences.
    *   **Sparse (Sub-quadratic):** The default DMoAH mode with token and head routing.
    *   **Shortlist (Linear Time):** For very long contexts, this mode uses a shortlist-based candidate generation to achieve linear time complexity.
*   **Chunked Shortlist Pipeline:** A two-pass architecture for processing sequences that exceed available VRAM. It streams the input, builds a "shortlist" of salient tokens, and performs a final attention pass on the distilled result.

---

## Performance Demonstration

The following benchmark was run on a single GPU with 16GB of VRAM using the `scripts/tinytoy.py` script. It compares the performance and memory usage of Proteus Attention (DMoAH) against a standard PyTorch `nn.MultiheadAttention` implementation across various sequence lengths.

| Seq Len | Model              | Latency (ms) | Memory (MB) | Mode        | Result        |
| :------ | :----------------- | :----------- | :---------- | :---------- | :------------ |
| 4,096   | Standard Attention | 11.67        | 1,137.90    | -           | Success       |
| 4,096   | DMoAH (BF16)       | 64.45        | 857.48      | `shortlist` | Success       |
| 16,384  | Standard Attention | 41.14        | 4,234.08    | -           | Success       |
| 16,384  | DMoAH (BF16)       | 27.92        | 644.93      | `shortlist` | Success       |
| 32,768  | Standard Attention | -            | -           | -           | OOM           |
| 32,768  | DMoAH (BF16)       | 50.36        | 992.80      | `shortlist` | Success       |
| 65,536  | Standard Attention | -            | -           | -           | OOM           |
| 65,536  | DMoAH (BF16)       | 97.99        | 1,581.49    | `shortlist` | Success       |
| 262,144 | Standard Attention | -            | -           | -           | OOM           |
| 262,144 | DMoAH (BF16)       | 401.46       | 6,082.09    | `shortlist` | Success       |
| 524,288 | Standard Attention | -            | -           | -           | OOM           |
| 524,288 | DMoAH (BF16)       | 810.43       | 12,106.17   | `shortlist` | Success       |

### Key Observations:

1.  **Memory Scaling:** Standard attention still hits OOM just past 16K tokens on this 16 GB GPU. The CV sparse kernel keeps DMoAH within budget through 524K tokens, trading quadratic memory for a flat shortlist working set.
2.  **Context Length:** Proteus runs sequences **32× longer** than dense SDPA on the same card. INT8 shortlist mode remains stable through ~262K tokens, while BF16 reaches 524K before exhausting VRAM.
3.  **Adaptive Modes:** The benchmarks capture the automatic ramp from dense to shortlist and fully linear behaviour. As the single `alpha` slider increases, the router smoothly lowers head budgets, enabling the continuous-variable kernel that powers the latest Top‑A sampler.

---

## Interactive Playground

The best way to validate these results is to run them yourself. The entire benchmark and validation suite is packaged into a single Google Colab notebook that runs on a free T4 GPU.

*   **Live Benchmark:** Run a head-to-head comparison against standard attention.
*   **Validation Suite:** Execute a full set of "Needle in a Haystack" and "Jigsaw Puzzle" tests to verify the mechanics.
*   **Scaling Demo:** Observe the Chunked Shortlist system processing million-token sequences with a fixed memory footprint.

**[► Click here to open the Proteus Playground in Google Colab](https://colab.research.google.com/github/Zen-Sherbert/Proteus-Attention/blob/main/TinyPlayground.ipynb)**

---

## Quickstart

To install and run the benchmarks locally:

```bash
# 1. Clone the repository
git clone https://github.com/Zen-Sherbert/Proteus-Attention.git
cd Proteus-Attention

# 2. Install the package in editable mode
# (Requires PyTorch and a CUDA-enabled environment for full performance)
pip install -e .

# 3. Run the primary performance benchmark
python scripts/tinytoy.py

# 4. Run the synthetic validation tests for the chunking mechanism
python scripts/chunked_shortlist_tests.py
```

## Repository Contents

*   `src/proteus_attention/`: The core library code, including the DMoAH implementation and Triton kernels.
*   `examples/`: A collection of training scripts demonstrating various strategies like baseline training, curriculum learning (`train_context_slide.py`), and adaptive training (`train_context_mastery.py`).
*   `scripts/`: Standalone utilities for benchmarking (`tinytoy.py`), validation (`chunked_shortlist_tests.py`), and system checks.
*   `TinyPlayground.ipynb`: The interactive Google Colab notebook for demonstration and experimentation.

## Project Status

This is an experimental architecture and an active area of research. The APIs and underlying mechanisms are subject to change.

**Recent highlights**

* **Continuously variable sparse kernel:** the DMoAH router now blends dense, shortlist, and fully linear regimes via a single `alpha`, while head/token budgets are selected with nucleus (top‑p) gating for smoother CVT behaviour.
* **Top‑A sampler + alignment:** inference uses the active-head consensus to filter or re-weight nucleus candidates; optional alignment loss keeps the heads predictive during training, improving stylistic cohesion and reducing projection drift.
* **Chunked shortlist parity:** the streaming shortlist tool and tinytoy benchmark now share the same nucleus and Top‑A plumbing, so large-context evaluations reflect the exact runtime kernel mix.
