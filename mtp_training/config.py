from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    model_path: str = "Qwen/Qwen3-ASR-1.7B"
    dataset_root: str = "/root/autodl-tmp/qwen3_asr_mtp_200h"
    train_manifest: str = "manifests/train.jsonl"
    eval_manifest: str = "manifests/dev.jsonl"
    output_dir: str = "/root/autodl-tmp/outputs/mtp3-stage1"
    stage: int = 1
    mtp_depth: int = 3
    alpha: float = 0.9
    seed: int = 20260819
    max_steps: int = 2000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    min_learning_rate: float = 1.0e-6
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int = 4
    log_steps: int = 10
    eval_steps: int = 250
    eval_batches: int = 100
    save_steps: int = 250
    save_total_limit: int = 3
    include_eos_in_loss: bool = False
    attn_implementation: str = "flash_attention_2"
    resume_from: str = ""
    init_mtp_from: str = ""
    sampler_mode: str = "duration"
    language_temperature: float = 0.5
    branch_position_mode: str = "base"
    loss_reduction: str = "token_mean"
    reference_eval_samples: int = 0
    reference_eval_max_new_tokens: int = 64
    reference_eval_steps: int = 0
    early_stop_after_step: int = 0
    early_stop_patience_evals: int = 3
    early_stop_min_delta: float = 0.03

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        config = cls(**values)
        if config.stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        if config.stage == 2 and not config.init_mtp_from and not config.resume_from:
            raise ValueError("Stage 2 requires init_mtp_from or resume_from")
        if config.sampler_mode not in ("duration", "language_temperature"):
            raise ValueError("sampler_mode must be duration or language_temperature")
        if not 0.0 <= config.language_temperature <= 1.0:
            raise ValueError("language_temperature must be between 0 and 1")
        if config.branch_position_mode not in ("base", "shifted"):
            raise ValueError("branch_position_mode must be base or shifted")
        if config.loss_reduction not in ("token_mean", "sample_mean"):
            raise ValueError("loss_reduction must be token_mean or sample_mean")
        if config.reference_eval_samples < 0:
            raise ValueError("reference_eval_samples must be >= 0")
        if config.reference_eval_max_new_tokens < 1:
            raise ValueError("reference_eval_max_new_tokens must be >= 1")
        if config.reference_eval_steps < 0:
            raise ValueError("reference_eval_steps must be >= 0")
        if config.early_stop_after_step < 0:
            raise ValueError("early_stop_after_step must be >= 0")
        if config.early_stop_patience_evals < 1:
            raise ValueError("early_stop_patience_evals must be >= 1")
        if config.early_stop_min_delta < 0:
            raise ValueError("early_stop_min_delta must be >= 0")
        return config

    def resolve_manifest(self, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else Path(self.dataset_root) / path)
