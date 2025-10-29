from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Add the repository's `src` tree to `sys.path` for src-layout imports."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_src_on_path()
