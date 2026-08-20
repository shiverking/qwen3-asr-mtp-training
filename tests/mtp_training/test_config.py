from pathlib import Path

import pytest

from mtp_training.config import TrainConfig


def test_stage2_requires_stage1_checkpoint(tmp_path: Path):
    config = tmp_path / "stage2.yaml"
    config.write_text("stage: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="init_mtp_from"):
        TrainConfig.from_yaml(config)


def test_relative_manifest_is_resolved_under_dataset_root():
    config = TrainConfig(dataset_root="/data", train_manifest="manifests/train.jsonl")
    assert Path(config.resolve_manifest(config.train_manifest)) == Path("/data/manifests/train.jsonl")


def test_rejects_unknown_sampler_mode(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text("sampler_mode: random\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sampler_mode"):
        TrainConfig.from_yaml(config)


def test_rejects_unknown_branch_position_mode(tmp_path: Path):
    config = tmp_path / "bad-position.yaml"
    config.write_text("branch_position_mode: absolute\n", encoding="utf-8")
    with pytest.raises(ValueError, match="branch_position_mode"):
        TrainConfig.from_yaml(config)
