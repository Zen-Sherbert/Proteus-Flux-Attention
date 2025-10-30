#!/usr/bin/env python
"""
Utility to load a saved Proteus Attention checkpoint and generate text.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import examples.common as common  # type: ignore
    from examples.modeling import MiniProteusLM, ModelConfig  # type: ignore
else:  # pragma: no cover - package execution
    from . import common
    from .modeling import MiniProteusLM, ModelConfig


def load_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if "model_cfg" not in ckpt or "state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing model configuration.")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from Proteus Attention checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint (.pt).")
    parser.add_argument("--device", default=None, help="Device string (default auto).")
    parser.add_argument("--prompt", default="Proteus Attention", help="Prompt text.")
    parser.add_argument("--tokens", type=int, default=120, help="Number of tokens to sample.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt = load_checkpoint(args.checkpoint)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = MiniProteusLM(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    tokenizer = common.get_tokenizer()
    encoded_prompt = tokenizer.encode(args.prompt)
    with torch.inference_mode():
        tokens = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
        generated = model.generate(tokens, max_new_tokens=args.tokens, temperature=args.temperature)
        text = tokenizer.decode(generated[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
