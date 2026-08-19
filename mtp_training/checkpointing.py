from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def _trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Any,
) -> Path:
    checkpoint = Path(output_dir) / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file(_trainable_state(model), checkpoint / "trainable_model.safetensors")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        checkpoint / "trainer_state.pt",
    )
    metadata = asdict(config)
    metadata.update({"global_step": step, "format_version": 1})
    (checkpoint / "mtp_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return checkpoint


def load_trainable_weights(
    model: torch.nn.Module,
    checkpoint: str | Path,
    allow_missing_asr: bool = False,
) -> None:
    checkpoint = Path(checkpoint)
    weights = load_file(checkpoint / "trainable_model.safetensors", device="cpu")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    unexpected = [name for name in unexpected if name in weights]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    loaded = set(weights)
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    absent = sorted(required - loaded)
    if allow_missing_asr:
        absent = [name for name in absent if not name.startswith("asr_model")]
    if absent:
        raise RuntimeError(f"Missing trainable checkpoint keys: {absent[:20]}")


def resume_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint: str | Path,
) -> int:
    load_trainable_weights(model, checkpoint)
    state = torch.load(Path(checkpoint) / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"])


def prune_checkpoints(output_dir: str | Path, keep: int) -> None:
    checkpoints = []
    for path in Path(output_dir).glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    for _, path in sorted(checkpoints)[:-keep]:
        shutil.rmtree(path)
