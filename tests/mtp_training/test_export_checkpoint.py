from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from mtp_training.export_checkpoint import export_checkpoint


def _write_base_model(path: Path) -> None:
    path.mkdir()
    save_file(
        {
            "thinker.model.embed_tokens.weight": torch.zeros(4, 2),
            "thinker.model.norm.weight": torch.zeros(2),
        },
        path / "model.safetensors",
    )
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ASRForConditionalGeneration"],
                "model_type": "qwen3_asr",
                "thinker_config": {
                    "audio_config": {"model_type": "qwen3_asr_audio_encoder"},
                    "text_config": {"model_type": "qwen3"},
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "processor_config.json").write_text("{}", encoding="utf-8")


def _write_checkpoint(
    path: Path, *, stage: int = 1, depth: int = 2, format_version: int = 1
) -> None:
    path.mkdir()
    weights = {}
    for layer in range(depth):
        weights[f"branches.{layer}.hidden_norm.weight"] = torch.ones(2) * (layer + 1)
        weights[f"branches.{layer}.projection.weight"] = torch.ones(2, 4)
    if stage == 2:
        weights["asr_model.thinker.model.norm.weight"] = torch.ones(2) * 7
    save_file(weights, path / "trainable_model.safetensors")
    (path / "trainer_state.pt").write_bytes(b"complete")
    (path / "mtp_config.json").write_text(
        json.dumps(
            {
                "format_version": format_version,
                "stage": stage,
                "mtp_depth": depth,
                "global_step": 10,
            }
        ),
        encoding="utf-8",
    )


def test_export_stage1_creates_self_contained_index(tmp_path: Path):
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "export"
    _write_base_model(base)
    _write_checkpoint(checkpoint)

    export_checkpoint(base, checkpoint, output)

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "mtp_export_metadata.json").read_text(encoding="utf-8"))
    assert config["mtp_num_hidden_layers"] == 2
    assert config["mtp_branch_position_mode"] == "base"
    assert index["weight_map"]["mtp.layers.1.hidden_norm.weight"] == "mtp_model.safetensors"
    assert (output / "processor_config.json").is_file()
    assert metadata["training_stage"] == 1
    assert metadata["branch_position_mode"] == "base"
    assert metadata["files"]["mtp_model.safetensors"]


def test_export_rejects_base_model_without_thinker_config(tmp_path: Path):
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "export"
    _write_base_model(base)
    (base / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ASRForConditionalGeneration"],
                "model_type": "qwen3_asr",
            }
        ),
        encoding="utf-8",
    )
    _write_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="thinker_config"):
        export_checkpoint(base, checkpoint, output)


def test_export_stage2_overlays_backbone_weight(tmp_path: Path):
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "export"
    _write_base_model(base)
    _write_checkpoint(checkpoint, stage=2)

    export_checkpoint(base, checkpoint, output)

    exported = load_file(output / "model.safetensors", device="cpu")
    assert torch.equal(exported["thinker.model.norm.weight"], torch.ones(2) * 7)


def test_export_combines_v2_stage1_and_stage2(tmp_path: Path):
    base = tmp_path / "base"
    stage1 = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    output = tmp_path / "export"
    _write_base_model(base)
    _write_checkpoint(stage1, stage=1, format_version=2)
    _write_checkpoint(stage2, stage=2, format_version=2)

    export_checkpoint(base, stage2, output, stage1_checkpoint=stage1)

    metadata = json.loads((output / "mtp_export_metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage1_training_checkpoint"] == str(stage1.resolve())
    assert metadata["training_checkpoint"] == str(stage2.resolve())
    assert (output / "mtp_model.safetensors").is_file()


def test_export_rejects_incomplete_checkpoint(tmp_path: Path):
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    _write_base_model(base)
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="incomplete"):
        export_checkpoint(base, checkpoint, tmp_path / "export")
