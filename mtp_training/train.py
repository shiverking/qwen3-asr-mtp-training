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
from tqdm.auto import tqdm

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


def should_stop_for_plateau(
    history: list[float], global_step: int, config: TrainConfig
) -> bool:
    if not config.early_stop_after_step or global_step < config.early_stop_after_step:
        return False
    required = config.early_stop_patience_evals + 1
    if len(history) < required:
        return False
    recent = history[-required:]
    return all(
        current - previous < config.early_stop_min_delta
        for previous, current in zip(recent, recent[1:])
    )


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


def append_metric(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def concise_eval(step: int, metrics: dict, reference_metrics: dict | None) -> str:
    macro = metrics["macro_average"]
    language_bb = ", ".join(
        f"{language}={values['decode_window_backbone_consistency_average_accepted_length']:.2f}"
        for language, values in metrics.items()
        if language not in ("all", "macro_average")
    )
    reference = (
        f" ref={reference_metrics['average_accepted_length']:.2f}"
        if reference_metrics is not None
        else ""
    )
    return (
        f"eval step={step} loss={macro['loss']:.4f} "
        f"gt={macro['decode_window_ground_truth_average_accepted_length']:.3f} "
        f"bb={macro['decode_window_backbone_consistency_average_accepted_length']:.3f}"
        f"{reference} | {language_bb}"
    )


def summarize_training_data(dataset, loader, config) -> dict:
    samples_by_language: Counter[str] = Counter()
    seconds_by_language: Counter[str] = Counter()
    for row in dataset.rows:
        language = row["language"]
        samples_by_language[language] += 1
        seconds_by_language[language] += float(row["duration_s"])
    batches_per_epoch = len(loader)
    planned_batches = config.max_steps * config.gradient_accumulation_steps
    return {
        "manifest": str(dataset.manifest_path),
        "samples": len(dataset),
        "hours": sum(seconds_by_language.values()) / 3600,
        "samples_by_language": dict(sorted(samples_by_language.items())),
        "hours_by_language": {
            language: seconds / 3600
            for language, seconds in sorted(seconds_by_language.items())
        },
        "batches_per_epoch": batches_per_epoch,
        "samples_per_full_epoch": batches_per_epoch * config.batch_size,
        "planned_optimizer_steps": config.max_steps,
        "planned_sample_exposures": (
            planned_batches * config.batch_size
        ),
        "planned_epochs": planned_batches / max(batches_per_epoch, 1),
    }


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
    metrics_path = output_dir / "metrics.jsonl"
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
        config.loss_reduction,
    )
    initialization_audit = audit_initialization(model)
    if not initialization_audit["passed"]:
        raise RuntimeError(f"MTP initialization audit failed: {initialization_audit}")
    counts = model.configure_trainable(config.stage)
    trainable_audit = audit_trainable_parameters(model, config.stage)
    if not trainable_audit["passed"]:
        raise RuntimeError(f"Trainable parameter audit failed: {trainable_audit}")
    startup_audit = {
        "parameter_counts": counts,
        "initialization_audit": initialization_audit,
        "trainable_parameter_audit": trainable_audit,
    }
    (output_dir / "startup_audit.json").write_text(
        json.dumps(startup_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"startup audit=passed trainable={counts['trainable']:,} "
        f"total={counts['total']:,}",
        flush=True,
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
    data_summary = summarize_training_data(train_dataset, train_loader, config)
    startup_audit["training_data"] = data_summary
    (output_dir / "startup_audit.json").write_text(
        json.dumps(startup_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    language_hours = ", ".join(
        f"{language}={hours:.1f}h"
        for language, hours in data_summary["hours_by_language"].items()
    )
    print(
        f"data samples={data_summary['samples']:,} hours={data_summary['hours']:.2f} "
        f"batches/epoch={data_summary['batches_per_epoch']:,} "
        f"planned_epochs={data_summary['planned_epochs']:.3f}",
        flush=True,
    )
    print(f"data languages {language_hours}", flush=True)

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
    backbone_acceptance_history: list[float] = []
    stop_requested = False
    progress = tqdm(
        total=config.max_steps,
        initial=global_step,
        desc=f"MTP-{config.mtp_depth} stage-{config.stage}",
        unit="step",
        dynamic_ncols=True,
    )
    while global_step < config.max_steps:
        for _ in range(config.gradient_accumulation_steps):
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
        progress.update(1)

        if global_step % config.log_steps == 0:
            elapsed = time.perf_counter() - running_started
            payload = {
                "step": global_step,
                "loss": running_loss / config.log_steps,
                "learning_rate": scheduler.get_last_lr()[0],
                "steps_per_second": config.log_steps / elapsed,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            append_metric(metrics_path, {"event": "train", **payload})
            progress.set_postfix(
                loss=f"{payload['loss']:.4f}",
                lr=f"{payload['learning_rate']:.2e}",
                speed=f"{payload['steps_per_second']:.2f}/s",
                memory=f"{payload['max_memory_gib']:.1f}GiB",
                refresh=True,
            )
            running_loss = 0.0
            running_started = time.perf_counter()

        if global_step % config.eval_steps == 0:
            metrics = evaluate(model, eval_loader, config.stage, device, config.eval_batches)
            backbone_acceptance_history.append(
                metrics["macro_average"][
                    "decode_window_backbone_consistency_average_accepted_length"
                ]
            )
            reference_metrics = None
            if (
                config.reference_eval_samples
                and config.reference_eval_steps
                and global_step % config.reference_eval_steps == 0
            ):
                from .reference_verifier import evaluate_speculative_reference

                reference_metrics = evaluate_speculative_reference(
                    model,
                    eval_dataset,
                    eval_loader.collate_fn,
                    device,
                    wrapper.processor.tokenizer.eos_token_id,
                    config.reference_eval_samples,
                    config.reference_eval_max_new_tokens,
                    config.seed,
                )
                reference_path = output_dir / f"reference-step-{global_step}.json"
                reference_path.write_text(
                    json.dumps(reference_metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            eval_payload = {
                "event": "eval",
                "step": global_step,
                "training_exposure": {
                    "samples": dict(sorted(exposure_samples.items())),
                    "transcript_tokens": dict(sorted(exposure_tokens.items())),
                },
                "eval": metrics,
                "reference_eval": (
                    {key: value for key, value in reference_metrics.items() if key != "results"}
                    if reference_metrics is not None
                    else None
                ),
            }
            append_metric(metrics_path, eval_payload)
            progress.write(concise_eval(global_step, metrics, reference_metrics))
            stop_requested = should_stop_for_plateau(
                backbone_acceptance_history, global_step, config
            )
            if stop_requested:
                progress.write(
                    f"early stop step={global_step} bb_recent="
                    f"{backbone_acceptance_history[-(config.early_stop_patience_evals + 1):]}"
                )

        if global_step % config.save_steps == 0:
            checkpoint = save_checkpoint(
                output_dir, global_step, model, optimizer, scheduler, config
            )
            prune_checkpoints(output_dir, config.save_total_limit)
            progress.write(f"saved {checkpoint}")

        if stop_requested:
            break
    progress.close()

    if global_step % config.save_steps:
        save_checkpoint(output_dir, global_step, model, optimizer, scheduler, config)


if __name__ == "__main__":
    main()
