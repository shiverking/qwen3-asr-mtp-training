from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from torch.utils.data import DataLoader

from .checkpointing import (
    load_trainable_weights,
    prune_checkpoints,
    resume_training,
    save_checkpoint,
)
from .config import TrainConfig
from .data import (
    DurationBucketBatchSampler,
    LanguageTemperatureBatchSampler,
    MTPDataCollator,
    ManifestDataset,
)
from .diagnostics import audit_initialization, audit_trainable_parameters
from .evaluation import evaluate
from .modeling_mtp import Qwen3ASRMTPModel


def parse_args():
    parser = argparse.ArgumentParser("Train ParaASR-style MTP branches for Qwen3-ASR")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, config: TrainConfig):
    minimum_ratio = config.min_learning_rate / config.learning_rate

    def schedule(step: int) -> float:
        if step < config.warmup_steps:
            return max(step, 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def build_loader(dataset, processor, config, train: bool):
    collator = MTPDataCollator(
        processor=processor,
        include_eos_in_loss=config.include_eos_in_loss,
    )
    sampler_class = (
        LanguageTemperatureBatchSampler
        if train and config.sampler_mode == "language_temperature"
        else DurationBucketBatchSampler
    )
    sampler_kwargs = {}
    if sampler_class is LanguageTemperatureBatchSampler:
        sampler_kwargs["temperature"] = config.language_temperature
    sampler = sampler_class(
        dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        drop_last=train,
        **sampler_kwargs,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("BF16-capable GPU is required")
    seed_everything(config.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
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
    )
    initialization_audit = audit_initialization(model)
    if not initialization_audit["passed"]:
        raise RuntimeError(f"MTP initialization audit failed: {initialization_audit}")
    counts = model.configure_trainable(config.stage)
    trainable_audit = audit_trainable_parameters(model, config.stage)
    if not trainable_audit["passed"]:
        raise RuntimeError(f"Trainable parameter audit failed: {trainable_audit}")
    print(
        json.dumps(
            {
                "parameter_counts": counts,
                "initialization_audit": initialization_audit,
                "trainable_parameter_audit": trainable_audit,
            },
            indent=2,
        )
    )
    device = torch.device("cuda:0")
    model.to(device)

    train_dataset = ManifestDataset(
        config.resolve_manifest(config.train_manifest), config.dataset_root
    )
    eval_dataset = ManifestDataset(
        config.resolve_manifest(config.eval_manifest), config.dataset_root
    )
    train_loader = build_loader(train_dataset, wrapper.processor, config, train=True)
    eval_loader = build_loader(eval_dataset, wrapper.processor, config, train=False)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)
    global_step = 0
    if config.resume_from:
        global_step = resume_training(model, optimizer, scheduler, config.resume_from)
    elif config.init_mtp_from:
        load_trainable_weights(model, config.init_mtp_from, allow_missing_asr=True)

    optimizer.zero_grad(set_to_none=True)
    model.train()
    running_loss = 0.0
    running_started = time.perf_counter()
    exposure_samples: Counter[str] = Counter()
    exposure_tokens: Counter[str] = Counter()
    data_iterator = iter(train_loader)
    while global_step < config.max_steps:
        for accumulation_index in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                train_loader.batch_sampler.set_epoch(global_step + 1)
                data_iterator = iter(train_loader)
                batch = next(data_iterator)
            languages = batch.pop("languages")
            batch.pop("sample_ids")
            token_counts = batch["loss_mask"].sum(dim=1).tolist()
            exposure_samples.update(languages)
            for language, count in zip(languages, token_counts):
                exposure_tokens[language] += int(count)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(stage=config.stage, **batch)
                loss = output.loss / config.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % config.log_steps == 0:
            elapsed = time.perf_counter() - running_started
            payload = {
                "step": global_step,
                "loss": running_loss / config.log_steps,
                "learning_rate": scheduler.get_last_lr()[0],
                "steps_per_second": config.log_steps / elapsed,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            running_loss = 0.0
            running_started = time.perf_counter()

        if global_step % config.eval_steps == 0:
            metrics = evaluate(model, eval_loader, config.stage, device, config.eval_batches)
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "training_exposure": {
                            "samples": dict(sorted(exposure_samples.items())),
                            "transcript_tokens": dict(sorted(exposure_tokens.items())),
                        },
                        "eval": metrics,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if global_step % config.save_steps == 0:
            checkpoint = save_checkpoint(
                output_dir, global_step, model, optimizer, scheduler, config
            )
            prune_checkpoints(output_dir, config.save_total_limit)
            print(f"Saved {checkpoint}", flush=True)

    if global_step % config.save_steps:
        save_checkpoint(output_dir, global_step, model, optimizer, scheduler, config)


if __name__ == "__main__":
    main()
