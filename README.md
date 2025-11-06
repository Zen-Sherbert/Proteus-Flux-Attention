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

| Seq Len | Model              | Latency (ms) | Memory (MB) | Mode      | Result                                         |
| :------ | :----------------- | :----------- | :---------- | :-------- | :--------------------------------------------- |
| 4,096   | Standard Attention | 11.77        | 1,138.46    | -         | Success                                        |
| 4,096   | DMoAH (BF16)       | 15.62        | 384.14      | `sparse`  | Success                                        |
| **16,384**  | Standard Attention | **69.17**        | **4,234.35**    | -         | **Success**                                    |
| **16,384**  | DMoAH (BF16)       | **51.98**        | **839.76**      | `sparse`  | **Success**                                    |
| **32,768**  | Standard Attention | -            | -           | -         | **Out of Memory**                              |
| 32,768  | DMoAH (BF16)       | 99.63        | 961.17      | `shortlist` | Success                                        |
| 131,072 | DMoAH (BF16)       | 287.66       | 2,813.49    | `shortlist` | Success                                        |
| **524,288** | DMoAH (BF16)       | **1947.55**      | **11,081.79**   | `shortlist` | **Success**                                    |
| **1,048,576** | DMoAH (BF16)       | -            | -           | `shortlist` | **Out of Memory**                              |

### Key Observations:

1.  **Memory Scaling:** Standard attention's memory usage grows quadratically, leading to an Out-of-Memory (OOM) error at 32,768 tokens. DMoAH's memory usage scales far more gracefully due to its sparse architecture.
2.  **Context Length:** On the same hardware, Proteus Attention successfully processes a context length of **524,288 tokens**—a **32x increase** over the standard implementation's limit.
3.  **Adaptive Modes:** The `Mode` column shows the system automatically transitioning from `sparse` to `shortlist` attention at 32,768 tokens to maintain efficiency at longer scales.

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
