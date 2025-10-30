import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import common
from examples import train_baseline, train_context_slide, train_context_mastery


def _use_tmp_checkpoint(monkeypatch, tmp_path):
    def _ensure(name: str):
        target = tmp_path / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(common, "ensure_checkpoint_dir", _ensure)


def _write_corpus(tmp_path):
    data_path = tmp_path / "corpus.txt"
    data_path.write_text("Proteus attention loves tiny tests.", encoding="utf-8")
    return data_path


def test_train_baseline_smoke(tmp_path, monkeypatch):
    _use_tmp_checkpoint(monkeypatch, tmp_path)
    data = _write_corpus(tmp_path)
    result = train_baseline.run_training(
        steps=2,
        seq_len=16,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        prompt="hello",
        sample_tokens=4,
        data_path=data,
    )
    assert "checkpoint" in result


def test_train_context_slide_smoke(tmp_path, monkeypatch):
    _use_tmp_checkpoint(monkeypatch, tmp_path)
    data = _write_corpus(tmp_path)
    result = train_context_slide.run_training(
        total_steps=6,
        short_seq=16,
        medium_seq=20,
        long_seq=24,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        data_path=data,
        prompt="context",
        sample_tokens=4,
    )
    assert "checkpoint" in result


def test_train_context_mastery_smoke(tmp_path, monkeypatch):
    _use_tmp_checkpoint(monkeypatch, tmp_path)
    data = _write_corpus(tmp_path)
    result = train_context_mastery.run_training(
        steps=6,
        base_seq_len=16,
        max_seq_len=32,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        alpha_start=0.0,
        alpha_step=0.5,
        plateau_window=2,
        plateau_tol=0.5,
        plateau_patience=1,
        plateau_cooldown=1,
        data_path=data,
        prompt="mastery",
        sample_tokens=4,
    )
    assert "checkpoint" in result
