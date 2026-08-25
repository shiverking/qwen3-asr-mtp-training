from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from qwen_asr import Qwen3ASRModel

from .checkpointing import load_trainable_weights
from .config import TrainConfig
from .data import ManifestDataset
from .evaluation import evaluate
from .modeling_mtp import Qwen3ASRMTPModel
from .reference_verifier import evaluate_speculative_reference
from .train import build_loader, seed_everything


@dataclass
class CheckpointTestConfig:
    model_path: str
    dataset_root: str
    test_manifest: str
    checkpoint: str
    output: str
    stage: int = 1
    mtp_depth: int = 3
    alpha: float = 0.9
    seed: int = 20260819
    batch_size: int = 32
    num_workers: int = 8
    include_eos_in_loss: bool = False
    attn_implementation: str = "flash_attention_2"
    branch_position_mode: str = "base"
    loss_reduction: str = "token_mean"
    reference_samples: int = -1
    reference_max_new_tokens: int = 64
    macro_bb_min: float = 2.5
    reference_min: float = 2.2
    per_language_reference_min: float = 1.8

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CheckpointTestConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown test config keys: {sorted(unknown)}")
        config = cls(**values)
        if config.stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        if config.mtp_depth < 1:
            raise ValueError("mtp_depth must be >= 1")
        if config.reference_samples == 0 or config.reference_samples < -1:
            raise ValueError("reference_samples must be -1 (all) or a positive integer")
        if config.reference_max_new_tokens < 1:
            raise ValueError("reference_max_new_tokens must be >= 1")
        return config

    def resolve_test_manifest(self) -> Path:
        path = Path(self.test_manifest)
        return path if path.is_absolute() else Path(self.dataset_root) / path


def _checkpoint_metadata(config: CheckpointTestConfig) -> dict[str, Any]:
    checkpoint = Path(config.checkpoint)
    required = (
        checkpoint / "trainable_model.safetensors",
        checkpoint / "trainer_state.pt",
        checkpoint / "mtp_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint, missing: {missing}")
    metadata = json.loads(required[2].read_text(encoding="utf-8"))
    expected = {
        "stage": config.stage,
        "mtp_depth": config.mtp_depth,
        "alpha": config.alpha,
        "include_eos_in_loss": config.include_eos_in_loss,
        "branch_position_mode": config.branch_position_mode,
        "loss_reduction": config.loss_reduction,
    }
    mismatches = {
        key: {"checkpoint": metadata.get(key), "test_config": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint/config mismatch: {mismatches}")
    return metadata


def _loader_config(config: CheckpointTestConfig) -> TrainConfig:
    return TrainConfig(
        model_path=config.model_path,
        dataset_root=config.dataset_root,
        eval_manifest=config.test_manifest,
        stage=config.stage,
        mtp_depth=config.mtp_depth,
        alpha=config.alpha,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        include_eos_in_loss=config.include_eos_in_loss,
        attn_implementation=config.attn_implementation,
        branch_position_mode=config.branch_position_mode,
        loss_reduction=config.loss_reduction,
    )


def _gates(
    teacher_forced: dict[str, Any],
    reference: dict[str, Any],
    config: CheckpointTestConfig,
) -> dict[str, Any]:
    reference_value = reference["average_accepted_length"]
    language_values = reference["by_language"]
    per_language = {
        language: {
            "value": value,
            "threshold": config.per_language_reference_min,
            "passed": value >= config.per_language_reference_min,
        }
        for language, value in language_values.items()
    }
    result = {
        "reference": {
            "value": reference_value,
            "threshold": config.reference_min,
            "passed": reference_value >= config.reference_min,
        },
        "per_language_reference": per_language,
    }
    result["passed"] = (
        result["reference"]["passed"]
        and bool(per_language)
        and all(item["passed"] for item in per_language.values())
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        "Run the complete test set against an MTP checkpoint"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = CheckpointTestConfig.from_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    metadata = _checkpoint_metadata(config)
    seed_everything(config.seed)
    print(
        f"checkpoint validated path={config.checkpoint} "
        f"step={metadata.get('global_step', 'unknown')}"
    )
    wrapper = Qwen3ASRModel.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        device_map=None,
        attn_implementation=config.attn_implementation,
    )
    wrapper.processor.tokenizer.padding_side = "right"
    model = Qwen3ASRMTPModel(
        wrapper.model,
        config.mtp_depth,
        config.alpha,
        config.branch_position_mode,
        config.loss_reduction,
    )
    model.configure_trainable(config.stage)
    load_trainable_weights(model, config.checkpoint)
    device = torch.device("cuda:0")
    model.to(device)
    print("checkpoint weights loaded")

    dataset = ManifestDataset(str(config.resolve_test_manifest()), config.dataset_root)
    loader = build_loader(dataset, wrapper.processor, _loader_config(config), train=False)
    print(f"test samples={len(dataset):,} batches={len(loader):,}")

    print("running full teacher-forced test")
    teacher_forced = evaluate(model, loader, config.stage, device, len(loader))
    macro = teacher_forced["macro_average"]
    print(
        f"test loss={macro['loss']:.4f} "
        f"target_len={macro['strict_average_accepted_length']:.3f}/6 "
        f"pos={[round(value, 4) for value in macro['strict_position_acceptance']]}"
    )
    for language, metrics in teacher_forced.items():
        if language in ("all", "macro_average"):
            continue
        print(
            f"test language={language} "
            f"target_len={metrics['strict_average_accepted_length']:.3f}/6 "
            f"pos={[round(value, 4) for value in metrics['strict_position_acceptance']]}"
        )

    reference_samples = (
        None if config.reference_samples == -1 else config.reference_samples
    )
    print(
        "running reference verifier "
        f"samples={'all' if reference_samples is None else reference_samples}"
    )
    reference = evaluate_speculative_reference(
        model=model,
        dataset=dataset,
        collator=loader.collate_fn,
        device=device,
        eos_token_id=wrapper.processor.tokenizer.eos_token_id,
        samples=reference_samples,
        max_new_tokens=config.reference_max_new_tokens,
        seed=config.seed,
        show_progress=True,
    )
    gates = _gates(teacher_forced, reference, config)
    print(
        f"reference accepted={reference['average_accepted_length']:.3f}/6 "
        f"pos={[round(value, 4) for value in reference['strict_position_acceptance']]}"
    )
    for language, value in reference["by_language"].items():
        positions = reference["by_language_metrics"][language]["strict_position_acceptance"]
        print(f"reference language={language} accepted={value:.3f}/6 pos={[round(item, 4) for item in positions]}")
    print(f"test gates={'passed' if gates['passed'] else 'failed'}")

    report = {
        "checkpoint": str(Path(config.checkpoint).resolve()),
        "checkpoint_metadata": metadata,
        "test_manifest": str(config.resolve_test_manifest().resolve()),
        "test_samples": len(dataset),
        "teacher_forced": teacher_forced,
        "reference": reference,
        "gates": gates,
    }
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test report={output}")


if __name__ == "__main__":
    main()
