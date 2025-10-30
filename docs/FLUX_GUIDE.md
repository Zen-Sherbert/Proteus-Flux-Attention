# Proteus Flux Attention Primer

Proteus Flux is the unified attention kernel that continuously blends dense, sparse,
and shortlist behaviour. Instead of switching between distinct kernels, the
model keeps a single Triton implementation warm and adjusts a _slider_ (the
Proteus Flux alpha) to control sparsity.

## Key Concepts

- **Proteus Flux Alpha (`α`)** – a scalar in `[0, 1]`:
  - `α = 0` behaves like dense attention (full causal window).
  - `α = 1` behaves like the constant-`L` shortlist path.
  - Intermediate values blend both, reducing token/head budgets smoothly.
- **Token Keep Ratio** – upper bound on the fraction of tokens a router can keep.
  TinyToy provides a schedule that decays with sequence length; feel free to
  substitute your own function if your workloads prefer different sparsity.
- **Shortlist Cap (`attn_linear_L_*`)** – how many candidate keys per query are
  considered in the shortlist path. This should scale down for long contexts to
  keep VRAM use predictable.
- **DNA Fraction** – optional semantic context retrieved from prior hidden
  states. DNA was dialled down to zero in the benchmark config to maximise
  memory headroom; you can turn it back on by increasing
  `attn_linear_piece_dna_frac` and `dna_cap` in `build_flux_candidates`.

## Controlling the Slider

`CausalDynamicAttention` exposes a `set_flux_alpha(value)` helper that pins the
runtime alpha. If you do nothing, the module derives `α` from
`seq_len`, `attn_active_seq_low`, and `attn_linear_switch_ctx`. For custom
behaviour:

```python
from proteus_attention.models.dmoah import ModelConfig, CausalDynamicAttention

cfg = ModelConfig(d_model=512, n_head=16, attn_mode="auto")
attn = CausalDynamicAttention(cfg)
attn.set_flux_alpha(0.35)  # midpoint between dense and shortlist
```

When building configs programmatically (e.g. in TinyToy) you can construct an
alpha schedule with `_flux_alpha_from_seq` to ensure reproducible profiles.

## Practical Tips

- Keep `attn_linear_L_*` in sync with the token keep ratio. If you decimate
  tokens but leave the shortlist cap large you waste memory without gaining
  quality.
- DNA is the first knob to revisit when accuracy dips; even a small DNA cap
  helps semantic recall on long contexts.
- If you encounter "HIP/CUDA out of memory" errors at very long contexts,
  reduce `attn_token_keep_ratio`, `attn_linear_L_max`, or increase
  `attn_linear_switch_ctx` so the model stays in the shortlist regime.

For a hands-on example, see `scripts/check_install.py` or the `tinytoy.py`
benchmark harness. Both demonstrate how to integrate Proteus Flux without swapping
kernels or touching the rest of your architecture.
