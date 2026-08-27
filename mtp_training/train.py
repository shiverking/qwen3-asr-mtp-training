from __future__ import annotations

import argparse
import hashlib
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
    IndexedManifestDataset,
    LanguageTemperatureBatchSampler,
    MixedLanguageSourceTemperatureBatchSampler,
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


def build_scheduler(optimizer, config: TrainConfig, schedule_steps: int):
    minimum_ratio = config.min_learning_rate / config.learning_rate

    def schedule(step: int) -> float:
        if step < config.warmup_steps:
            return max(step, 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            schedule_steps - config.warmup_steps, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def build_loader(dataset, processor, config, train: bool):
    collator = MTPDataCollator(
        processor=processor,
        include_eos_in_loss=config.include_eos_in_loss,
        target_text_field=(
            config.train_target_text_field
            if train
            else config.eval_target_text_field
        ),
    )
    if train and config.sampler_mode == "mixed_language_source_temperature":
        sampler_class = MixedLanguageSourceTemperatureBatchSampler
    elif train and config.sampler_mode == "language_temperature":
        sampler_class = LanguageTemperatureBatchSampler
    else:
        sampler_class = DurationBucketBatchSampler
    sampler_kwargs = {}
    if sampler_class is LanguageTemperatureBatchSampler:
        sampler_kwargs["temperature"] = config.language_temperature
    elif sampler_class is MixedLanguageSourceTemperatureBatchSampler:
        sampler_kwargs.update(
            language_temperature=config.language_temperature,
            source_temperature=config.source_temperature,
        )
    sampler = sampler_class(
        dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        drop_last=False,
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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_tensorboard_writer(config: TrainConfig, global_step: int):
    if not config.tensorboard_enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard is enabled but not installed; run: pip install tensorboard"
        ) from error
    log_dir = config.tensorboard_log_dir or str(Path(config.output_dir) / "tensorboard")
    return SummaryWriter(
        log_dir=log_dir,
        purge_step=global_step + 1 if global_step else None,
        flush_secs=config.tensorboard_flush_secs,
    )


def write_tensorboard_eval(writer, metrics: dict, step: int) -> None:
    if writer is None:
        return
    macro = metrics["macro_average"]
    writer.add_scalar("eval/loss", macro["loss"], step)
    writer.add_scalar(
        "eval/greedy_accepted_length",
        macro["strict_average_accepted_length"],
        step,
    )
    for position, value in enumerate(macro["strict_position_acceptance"], start=1):
        writer.add_scalar(f"eval/position_acceptance/p{position}", value, step)
    for language, values in metrics.items():
        if language in ("all", "macro_average"):
            continue
        writer.add_scalar(
            f"eval_language/{language}/greedy_accepted_length",
            values["strict_average_accepted_length"],
            step,
        )


def write_tensorboard_verify(writer, metrics: dict, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar("verify/accepted_length", metrics["average_accepted_length"], step)
    for position, value in enumerate(metrics["strict_position_acceptance"], start=1):
        writer.add_scalar(f"verify/position_acceptance/p{position}", value, step)
    for language, value in metrics["by_language"].items():
        writer.add_scalar(f"verify_language/{language}/accepted_length", value, step)


def validate_dataset_gate(config: TrainConfig) -> None:
    root = Path(config.dataset_root)
    temporary_files = list((root / "manifests").rglob("*.tmp"))
    if temporary_files:
        raise RuntimeError(f"Dataset contains unfinished .tmp files: {temporary_files[:5]}")
    if config.use_indexed_train_dataset:
        required_languages = {"en", "es", "pt-BR", "pt-PT"}
        for split in ("dev", "test"):
            path = root / "manifests" / "eval" / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing multilingual {split} manifest: {path}")
            languages = set()
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        languages.add(json.loads(line)["language"])
            missing = required_languages - languages
            if missing:
                raise RuntimeError(f"{split} manifest is missing languages: {sorted(missing)}")


def concise_eval(epoch: float, step: int, metrics: dict) -> str:
    macro = metrics["macro_average"]
    languages = ", ".join(
        f"{language}={values['strict_average_accepted_length']:.2f}"
        for language, values in metrics.items()
        if language not in ("all", "macro_average")
    )
    positions = ",".join(f"{value:.3f}" for value in macro["strict_position_acceptance"])
    return (
        f"eval epoch={epoch:.2f} step={step} loss={macro['loss']:.4f} "
        f"greedy_len={macro['strict_average_accepted_length']:.3f}/6 "
        f"pos=[{positions}] | {languages}"
    )


def concise_verify(epoch: float, metrics: dict) -> str:
    positions = ",".join(f"{value:.3f}" for value in metrics["strict_position_acceptance"])
    languages = ", ".join(
        f"{key}={value:.2f}" for key, value in metrics["by_language"].items()
    )
    return (
        f"verify epoch={epoch:.2f} len={metrics['average_accepted_length']:.3f}/6 "
        f"pos=[{positions}] | {languages}"
    )


def summarize_training_data(dataset, loader, config, target_steps: int) -> dict:
    if isinstance(dataset, IndexedManifestDataset):
        metadata = dataset.metadata
        batches_per_epoch = len(loader)
        planned_batches = target_steps * config.gradient_accumulation_steps
        return {
            "manifest": str(dataset.manifest_path),
            "manifest_sha256": dataset.manifest_sha256,
            "samples": len(dataset),
            "hours": metadata["hours"],
            "samples_by_language": metadata["samples_by_language"],
            "hours_by_language": metadata["hours_by_language"],
            "batches_per_epoch": batches_per_epoch,
            "samples_per_full_epoch": len(dataset),
            "planned_optimizer_steps": target_steps,
            "planned_sample_exposures_upper_bound": planned_batches * config.batch_size,
            "planned_epochs": planned_batches / max(batches_per_epoch, 1),
            "covers_full_manifest": planned_batches >= batches_per_epoch,
        }
    samples_by_language: Counter[str] = Counter()
    seconds_by_language: Counter[str] = Counter()
    for row in dataset.rows:
        language = row["language"]
        samples_by_language[language] += 1
        seconds_by_language[language] += float(row["duration_s"])
    batches_per_epoch = len(loader)
    planned_batches = target_steps * config.gradient_accumulation_steps
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
        "samples_per_full_epoch": len(dataset),
        "planned_optimizer_steps": target_steps,
        "planned_sample_exposures_upper_bound": (
            planned_batches * config.batch_size
        ),
        "planned_epochs": planned_batches / max(batches_per_epoch, 1),
        "covers_full_manifest": planned_batches >= batches_per_epoch,
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
    validate_dataset_gate(config)

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

    train_dataset_class = (
        IndexedManifestDataset if config.use_indexed_train_dataset else ManifestDataset
    )
    train_dataset = train_dataset_class(
        config.resolve_manifest(config.train_manifest), config.dataset_root
    )
    if isinstance(train_dataset, IndexedManifestDataset):
        expected_languages = {"en", "es", "pt-BR", "pt-PT"}
        actual_languages = set(train_dataset.languages)
        if actual_languages != expected_languages:
            raise RuntimeError(
                f"Indexed train manifest languages {sorted(actual_languages)} do not "
                f"match required {sorted(expected_languages)}"
            )
    eval_dataset = ManifestDataset(
        config.resolve_manifest(config.eval_manifest), config.dataset_root
    )
    train_loader = build_loader(train_dataset, wrapper.processor, config, train=True)
    eval_loader = build_loader(eval_dataset, wrapper.processor, config, train=False)
    target_steps = (
        math.ceil(
            len(train_loader)
            * config.num_train_epochs
            / config.gradient_accumulation_steps
        )
        if config.num_train_epochs
        else config.max_steps
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    eval_interval = (
        max(1, round(optimizer_steps_per_epoch * config.eval_every_epochs))
        if config.eval_every_epochs
        else config.eval_steps
    )
    verify_interval = (
        max(1, round(optimizer_steps_per_epoch * config.verify_every_epochs))
        if config.verify_every_epochs
        else config.reference_eval_steps
    )
    save_interval = (
        max(1, round(optimizer_steps_per_epoch * config.save_every_epochs))
        if config.save_every_epochs
        else config.save_steps
    )
    if config.warmup_ratio:
        config.warmup_steps = max(1, round(target_steps * config.warmup_ratio))
    if config.initial_step >= target_steps:
        raise ValueError(
            f"initial_step {config.initial_step} must be below target step {target_steps}"
        )
    data_summary = summarize_training_data(
        train_dataset, train_loader, config, target_steps
    )
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
        f"planned_epochs={data_summary['planned_epochs']:.3f} "
        f"covers_full_manifest={str(data_summary['covers_full_manifest']).lower()}",
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
    global_step = config.initial_step
    if config.resume_from:
        scheduler = build_scheduler(optimizer, config, target_steps)
        global_step = resume_training(model, optimizer, scheduler, config.resume_from)
    elif config.init_mtp_from:
        load_trainable_weights(model, config.init_mtp_from, allow_missing_asr=True)
        scheduler = build_scheduler(
            optimizer, config, max(target_steps - global_step, 1)
        )
    else:
        scheduler = build_scheduler(optimizer, config, target_steps)
    starting_step = global_step
    runtime_metadata = {
        "train_manifest_sha256": getattr(train_dataset, "manifest_sha256", ""),
        "dev_manifest_sha256": file_sha256(eval_dataset.manifest_path),
        "completed_epoch": global_step / optimizer_steps_per_epoch,
        "steps_per_epoch": optimizer_steps_per_epoch,
        "greedy_target_model_revision": config.greedy_target_model_revision,
        "sampler": {
            "mode": config.sampler_mode,
            "language_temperature": config.language_temperature,
            "source_temperature": config.source_temperature,
        },
    }
    if config.resume_from:
        checkpoint_metadata = json.loads(
            (Path(config.resume_from) / "mtp_config.json").read_text(encoding="utf-8")
        ).get("runtime_metadata", {})
        expected_hash = checkpoint_metadata.get("train_manifest_sha256")
        if expected_hash and expected_hash != runtime_metadata["train_manifest_sha256"]:
            raise RuntimeError("Training manifest hash differs from checkpoint")
    tensorboard_writer = create_tensorboard_writer(config, global_step)

    optimizer.zero_grad(set_to_none=True)
    model.train()
    running_loss = 0.0
    running_started = time.perf_counter()
    exposure_samples: Counter[str] = Counter()
    exposure_tokens: Counter[str] = Counter()
    consumed_batches = global_step * config.gradient_accumulation_steps
    data_epoch, start_batch = divmod(consumed_batches, len(train_loader))
    train_loader.batch_sampler.set_epoch(data_epoch)
    if start_batch:
        if not hasattr(train_loader.batch_sampler, "set_start_batch"):
            raise ValueError("Selected sampler cannot continue from initial_step")
        train_loader.batch_sampler.set_start_batch(start_batch)
    data_iterator = iter(train_loader)
    progress = tqdm(
        total=target_steps,
        initial=global_step,
        desc=f"MTP-{config.mtp_depth} stage-{config.stage}",
        unit="step",
        dynamic_ncols=True,
    )
    while global_step < target_steps:
        for _ in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_epoch += 1
                train_loader.batch_sampler.set_epoch(data_epoch)
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

        run_step = global_step - starting_step
        if run_step % config.log_steps == 0:
            elapsed = time.perf_counter() - running_started
            payload = {
                "step": global_step,
                "loss": running_loss / config.log_steps,
                "learning_rate": scheduler.get_last_lr()[0],
                "steps_per_second": config.log_steps / elapsed,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            append_metric(metrics_path, {"event": "train", **payload})
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("train/loss", payload["loss"], global_step)
                tensorboard_writer.add_scalar(
                    "train/learning_rate", payload["learning_rate"], global_step
                )
                tensorboard_writer.add_scalar(
                    "train/steps_per_second", payload["steps_per_second"], global_step
                )
                tensorboard_writer.add_scalar(
                    "train/max_memory_gib", payload["max_memory_gib"], global_step
                )
                tensorboard_writer.add_scalar(
                    "train/epoch", global_step / optimizer_steps_per_epoch, global_step
                )
            progress.set_postfix(
                loss=f"{payload['loss']:.4f}",
                lr=f"{payload['learning_rate']:.2e}",
                speed=f"{payload['steps_per_second']:.2f}/s",
                memory=f"{payload['max_memory_gib']:.1f}GiB",
                refresh=True,
            )
            running_loss = 0.0
            running_started = time.perf_counter()

        if run_step % eval_interval == 0:
            metrics = evaluate(model, eval_loader, config.stage, device, config.eval_batches)
            current_epoch = global_step / optimizer_steps_per_epoch
            eval_payload = {
                "event": "eval",
                "step": global_step,
                "training_exposure": {
                    "samples": dict(sorted(exposure_samples.items())),
                    "transcript_tokens": dict(sorted(exposure_tokens.items())),
                },
                "eval": metrics,
                "reference_eval": None,
            }
            append_metric(metrics_path, eval_payload)
            write_tensorboard_eval(tensorboard_writer, metrics, global_step)
            if tensorboard_writer is not None:
                tensorboard_writer.flush()
            progress.write(concise_eval(current_epoch, global_step, metrics))

        if config.reference_eval_samples and verify_interval and run_step % verify_interval == 0:
            from .reference_verifier import evaluate_speculative_reference

            current_epoch = global_step / optimizer_steps_per_epoch
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
            append_metric(
                metrics_path,
                {
                    "event": "verify",
                    "step": global_step,
                    "epoch": current_epoch,
                    "reference_eval": {
                        key: value
                        for key, value in reference_metrics.items()
                        if key != "results"
                    },
                },
            )
            write_tensorboard_verify(
                tensorboard_writer, reference_metrics, global_step
            )
            if tensorboard_writer is not None:
                tensorboard_writer.flush()
            progress.write(concise_verify(current_epoch, reference_metrics))

        if run_step % save_interval == 0:
            runtime_metadata["completed_epoch"] = global_step / optimizer_steps_per_epoch
            checkpoint = save_checkpoint(
                output_dir, global_step, model, optimizer, scheduler, config,
                runtime_metadata=runtime_metadata,
            )
            prune_checkpoints(output_dir, config.save_total_limit)
            if tensorboard_writer is not None:
                tensorboard_writer.flush()
            progress.write(f"saved {checkpoint}")

    progress.close()

    if (global_step - starting_step) % save_interval:
        runtime_metadata["completed_epoch"] = global_step / optimizer_steps_per_epoch
        save_checkpoint(
            output_dir, global_step, model, optimizer, scheduler, config,
            runtime_metadata=runtime_metadata,
        )
    if tensorboard_writer is not None:
        tensorboard_writer.close()


if __name__ == "__main__":
    main()
