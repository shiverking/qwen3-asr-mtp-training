from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file


MTP_FORMAT_VERSION = 1
MTP_ARCHITECTURE = "paraasr_serial"
SUPPORTED_TRAINING_CHECKPOINT_FORMATS = {1, 2}
REQUIRED_CHECKPOINT_FILES = (
    "trainable_model.safetensors",
    "trainer_state.pt",
    "mtp_config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_base_model_config(model_config: dict[str, Any]) -> None:
    if model_config.get("model_type") != "qwen3_asr":
        raise ValueError("Base model config.json must have model_type='qwen3_asr'")
    architectures = model_config.get("architectures")
    if not isinstance(architectures, list) or "Qwen3ASRForConditionalGeneration" not in architectures:
        raise ValueError(
            "Base model config.json must declare Qwen3ASRForConditionalGeneration"
        )
    thinker_config = model_config.get("thinker_config")
    if not isinstance(thinker_config, dict):
        raise ValueError("Base model config.json is missing thinker_config")
    for name in ("audio_config", "text_config"):
        if not isinstance(thinker_config.get(name), dict):
            raise ValueError(f"Base model thinker_config is missing {name}")


def _validate_complete_checkpoint(checkpoint: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Checkpoint is incomplete; missing files: {missing}")
    config = _load_json(checkpoint / "mtp_config.json")
    if config.get("format_version") not in SUPPORTED_TRAINING_CHECKPOINT_FORMATS:
        raise ValueError(f"Unsupported training checkpoint format: {config.get('format_version')!r}")
    if int(config.get("stage", 0)) not in (1, 2):
        raise ValueError("Checkpoint stage must be 1 or 2")
    if int(config.get("mtp_depth", 0)) < 1:
        raise ValueError("Checkpoint mtp_depth must be positive")
    return config


def _convert_trainable_weights(
    weights: dict[str, Any], stage: int, depth: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    mtp_weights: dict[str, Any] = {}
    backbone_weights: dict[str, Any] = {}
    layer_suffixes: dict[int, set[str]] = defaultdict(set)

    for name, tensor in weights.items():
        if name.startswith("branches."):
            parts = name.split(".", 2)
            if len(parts) != 3 or not parts[1].isdigit():
                raise ValueError(f"Invalid MTP checkpoint key: {name}")
            layer_index = int(parts[1])
            suffix = parts[2]
            if layer_index >= depth:
                raise ValueError(f"MTP layer {layer_index} exceeds configured depth {depth}")
            exported_name = f"mtp.layers.{layer_index}.{suffix}"
            mtp_weights[exported_name] = tensor
            layer_suffixes[layer_index].add(suffix)
        elif name.startswith("asr_model."):
            backbone_weights[name.removeprefix("asr_model.")] = tensor
        else:
            raise ValueError(f"Unexpected trainable checkpoint key: {name}")

    expected_layers = set(range(depth))
    if set(layer_suffixes) != expected_layers:
        raise ValueError(
            f"MTP layers do not match configured depth: expected {sorted(expected_layers)}, "
            f"got {sorted(layer_suffixes)}"
        )
    reference_suffixes = layer_suffixes[0]
    for layer_index in range(1, depth):
        if layer_suffixes[layer_index] != reference_suffixes:
            missing = sorted(reference_suffixes - layer_suffixes[layer_index])
            extra = sorted(layer_suffixes[layer_index] - reference_suffixes)
            raise ValueError(f"MTP layer {layer_index} differs from layer 0; missing={missing}, extra={extra}")
        for suffix in reference_suffixes:
            reference_shape = mtp_weights[f"mtp.layers.0.{suffix}"].shape
            layer_shape = mtp_weights[f"mtp.layers.{layer_index}.{suffix}"].shape
            if layer_shape != reference_shape:
                raise ValueError(
                    f"MTP layer {layer_index} shape differs for {suffix}: "
                    f"layer0={tuple(reference_shape)}, layer{layer_index}={tuple(layer_shape)}"
                )
    if stage == 1 and backbone_weights:
        raise ValueError("Stage 1 checkpoint unexpectedly contains trainable ASR weights")
    if stage == 2 and not backbone_weights:
        raise ValueError("Stage 2 checkpoint does not contain trainable ASR weights")
    return mtp_weights, backbone_weights


def _checkpoint_layout(model_dir: Path) -> tuple[dict[str, str], Path | None]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path)
        return dict(index["weight_map"]), index_path

    model_path = model_dir / "model.safetensors"
    if not model_path.is_file():
        raise ValueError("Base model must contain model.safetensors or model.safetensors.index.json")
    keys = load_file(model_path, device="cpu").keys()
    return {key: model_path.name for key in keys}, None


def _overlay_backbone_weights(
    output_dir: Path, weight_map: dict[str, str], overrides: dict[str, Any]
) -> None:
    by_shard: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, tensor in overrides.items():
        shard = weight_map.get(name)
        if shard is None:
            raise ValueError(f"Stage 2 weight is absent from base checkpoint: {name}")
        by_shard[shard][name] = tensor

    for shard_name, shard_overrides in by_shard.items():
        shard_path = output_dir / shard_name
        shard_weights = load_file(shard_path, device="cpu")
        for name, tensor in shard_overrides.items():
            if shard_weights[name].shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: base={tuple(shard_weights[name].shape)}, "
                    f"trained={tuple(tensor.shape)}"
                )
            shard_weights[name] = tensor
        save_file(shard_weights, shard_path)


def export_checkpoint(
    base_model: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    stage1_checkpoint: str | Path | None = None,
) -> Path:
    base_model = Path(base_model).resolve()
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    if not base_model.is_dir():
        raise ValueError("base_model must be a local self-contained model directory")
    _validate_base_model_config(_load_json(base_model / "config.json"))
    if output_dir.is_relative_to(base_model):
        raise ValueError("output_dir must not be inside base_model")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    train_config = _validate_complete_checkpoint(checkpoint)
    stage = int(train_config["stage"])
    depth = int(train_config["mtp_depth"])
    trainable = load_file(checkpoint / "trainable_model.safetensors", device="cpu")
    mtp_weights, backbone_weights = _convert_trainable_weights(trainable, stage, depth)

    stage1_checkpoint_path: Path | None = None
    if stage1_checkpoint is not None:
        stage1_checkpoint_path = Path(stage1_checkpoint).resolve()
        stage1_config = _validate_complete_checkpoint(stage1_checkpoint_path)
        if int(stage1_config["stage"]) != 1 or stage != 2:
            raise ValueError("Combined export requires a Stage 1 checkpoint and a Stage 2 checkpoint")
        if int(stage1_config["mtp_depth"]) != depth:
            raise ValueError("Stage 1 and Stage 2 checkpoints use different MTP depths")
        stage1_position_mode = stage1_config.get("branch_position_mode", "base")
        stage2_position_mode = train_config.get("branch_position_mode", "base")
        if stage1_position_mode != stage2_position_mode:
            raise ValueError("Stage 1 and Stage 2 checkpoints use different branch position modes")
        stage1_trainable = load_file(
            stage1_checkpoint_path / "trainable_model.safetensors", device="cpu"
        )
        stage1_mtp_weights, _ = _convert_trainable_weights(
            stage1_trainable, stage=1, depth=depth
        )
        stage1_mtp_weights.update(mtp_weights)
        mtp_weights = stage1_mtp_weights

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        shutil.copytree(base_model, temporary, dirs_exist_ok=True)
        weight_map, index_path = _checkpoint_layout(temporary)
        _overlay_backbone_weights(temporary, weight_map, backbone_weights)

        mtp_shard_name = "mtp_model.safetensors"
        save_file(mtp_weights, temporary / mtp_shard_name)
        for name in mtp_weights:
            if name in weight_map:
                raise ValueError(f"Base checkpoint already contains exported MTP weight: {name}")
            weight_map[name] = mtp_shard_name

        if index_path is None:
            index_path = temporary / "model.safetensors.index.json"
        total_size = sum((temporary / shard).stat().st_size for shard in set(weight_map.values()))
        index_path.write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        config_path = temporary / "config.json"
        model_config = _load_json(config_path)
        _validate_base_model_config(model_config)
        model_config.update(
            {
                "mtp_format_version": MTP_FORMAT_VERSION,
                "mtp_architecture": MTP_ARCHITECTURE,
                "mtp_num_hidden_layers": depth,
                "num_nextn_predict_layers": depth,
                "mtp_branch_position_mode": train_config.get(
                    "branch_position_mode", "base"
                ),
            }
        )
        config_path.write_text(
            json.dumps(model_config, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        hashed_files = sorted(
            path for path in temporary.iterdir() if path.is_file() and path.name != "mtp_export_metadata.json"
        )
        metadata = {
            "format_version": MTP_FORMAT_VERSION,
            "architecture": MTP_ARCHITECTURE,
            "base_model": str(base_model),
            "base_model_revision": model_config.get("_commit_hash"),
            "training_checkpoint": str(checkpoint),
            "stage1_training_checkpoint": (
                str(stage1_checkpoint_path) if stage1_checkpoint_path is not None else None
            ),
            "training_stage": stage,
            "global_step": int(train_config.get("global_step", 0)),
            "mtp_depth": depth,
            "branch_position_mode": train_config.get("branch_position_mode", "base"),
            "files": {path.name: _sha256(path) for path in hashed_files},
        }
        (temporary / "mtp_export_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Qwen3-ASR ParaASR-style MTP checkpoint")
    parser.add_argument("--base-model", required=True, help="Local Qwen3-ASR model directory")
    parser.add_argument(
        "--checkpoint",
        "--stage2-checkpoint",
        dest="checkpoint",
        required=True,
        help="Completed Stage 2 training checkpoint directory",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        help="Optional Stage 1 checkpoint applied before the Stage 2 checkpoint",
    )
    parser.add_argument("--output-dir", required=True, help="New self-contained model directory")
    args = parser.parse_args()
    exported = export_checkpoint(
        args.base_model,
        args.checkpoint,
        args.output_dir,
        stage1_checkpoint=args.stage1_checkpoint,
    )
    print(exported)


if __name__ == "__main__":
    main()
