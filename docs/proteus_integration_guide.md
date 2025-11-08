# Proteus Attention Integration Guide

This note outlines how to adapt an existing Hugging Face checkpoint (e.g.
Gemma-1B-it) to use the Proteus Adaptive Sparse Attention stack.

## 1. Convert the Checkpoint Directory

Use the helper CLI to scaffold a Proteus-ready copy of the model:

```bash
python scripts/proteus_convertor.py --model-path /path/to/Gemma-1b-it
```

The tool creates a sibling directory with the `-Proteus` suffix, retains the
original tokenizer/weights, injects a `proteus_attention` stanza into
`config.json`, and drops a `README_PROTEUS.md` with next steps.  No tensors are
modified yet.

Once the workspace is ready, launch staged training with:

```bash
python scripts/proteus_trainer.py --workspace /path/to/Gemma-1b-it-Proteus
```

The trainer reads the latest `proteus_runs/run-*/` plan, executes the staged
curriculum, pauses after each stage with an interactive menu, and records
metrics/checkpoints under the run directory.

## 2. Patch the Model Class

1. Subclass the Hugging Face decoder block so that each attention layer uses
   `proteus_attention.modules.CausalASPAMultiheadAttention` (drop-in MHA) or
   the lower-level `AdaptiveSparseAttentionBlock`.
2. Extend the config class to surface the Proteus-specific knobs
   (`attn_proto_*`, `shortlist_alpha`, etc.).
3. Write a checkpoint loader that copies compatible weights (QKV, output
   projections) from the base model and initialises the new router/prototype
   parameters with small random tensors.

The scaffolding in `src/proteus_attention/integration/hf_adapter.py` exposes the
entry points used by the trainer.  Replace the default `ModelAdapter`
implementation with architecture-specific logic that swaps the native attention
modules for Proteus equivalents, defines stage-specific freezing rules, and
returns parameter groups for different learning-rate schedules.

## 3. Fine-tuning Curriculum

The key is to let prototypes and shortlist controllers stabilise before pushing
the full long-context workload.

Each stage in the CLI pausing after completion, presenting a clear menu (think
"Arch installer" style). At every pause you can inspect metrics, switch to a
different corpus, tweak hyperparameters, or abort gracefully. The live console
UI avoids log spam by re-drawing a single status panel (steps, loss, ppl,
alpha, controller state) instead of streaming new lines. All historical metrics
are persisted to CSV/JSON alongside a Markdown summary so you still have an
audit trail.

### Stage A — Prototype Warm-up (≈5–10k steps)

- Freeze the original transformer weights except the attention routers and
  newly added Proteus parameters.
- Train on short sequences (2k–4k tokens) with `shortlist_alpha = 0` (dense
  mode) so the router learns the gating distribution.
- Use a slightly elevated learning rate for the router (e.g. 5× base LR) and
  enable prototype EMA updates.

When the plateau detector confirms improvement has stalled (or the configured
step budget is exhausted) the CLI pauses, summarizes the stage, and asks whether
to continue or adjust settings before resuming.

### Stage B — Sparse Head Activation (≈20–40k steps)

- Unfreeze the attention projections and MLPs; keep embeddings and layer norms
  on a lower learning rate.
- Linearly increase `shortlist_alpha` from 0 → 1 over this phase while raising
  the token budget (`per_chunk_budget`) to the target value.
- Introduce the salience controller (SparseHeadController) so the number of
  active heads per token adapts instead of collapsing to dense execution.

### Stage C — Long-context Curriculum (≈80–150k steps)

- Increase sequence lengths following a doubling schedule (e.g. 8k → 16k →
  32k → 64k) once validation loss plateaus at each level.
- Enable chunked shortlist streaming for contexts beyond the Triton limit;
  monitor `proto` stats and retention ratio (`keep_indices`).
- Periodically run the `scripts/chunked_shortlist_smoke.py` sanity check to
  ensure shortlist tuning saturates and no regression appears in throughput.

### Stage D — Optional Consolidation (≈10–20k steps)

Once Stage C finishes, you can run a short consolidation pass focusing on the
prototype/controller parameters. The pause menu lets you skip this phase if
results already meet your targets.

## 4. Optimisation Tips

- Maintain separate optimisers or parameter groups: one for newly introduced
  Proteus parameters (higher LR), one for the frozen/unfrozen backbone (lower
  LR).  AdamW with decoupled weight decay works well.
- Keep gradient checkpointing enabled; our attention layer is compatible with
  PyTorch’s `torch.utils.checkpoint`.
- Save EMA snapshots of the routing prototypes regularly.  The conversion tool
  can be rerun on the latest checkpoint to keep metadata in sync.

## 5. Validation & Export

- During evaluation, log `last_head_stats["proto"]` to confirm the salience
  prior is being used (non-zero blend, stable usage entropy).
- When exporting, persist the Proteus config stanza so downstream consumers can
  rebuild the attention module consistently.
- If you need a standard HF `AutoModel`, bundle a thin adapter module that
  imports Proteus at runtime and registers the custom config/model classes.

Following this playbook yields a reproducible path from an off-the-shelf
checkpoint to a Proteus-enhanced model ready for long-context inference.
