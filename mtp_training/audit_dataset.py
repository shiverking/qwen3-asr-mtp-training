from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

from .config import TrainConfig
from .data import MTPDataCollator, ManifestDataset


SCRIPT_PATTERNS = {
    "zh-CN": re.compile(r"[\u3400-\u9fff]"),
    "ar": re.compile(r"[\u0600-\u06ff]"),
    "th": re.compile(r"[\u0e00-\u0e7f]"),
    "en": re.compile(r"[A-Za-z]"),
    "es": re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]"),
    "pt-BR": re.compile(r"[A-Za-zÁÀÂÃÉÊÍÓÔÕÚÇáàâãéêíóôõúç]"),
    "pt-PT": re.compile(r"[A-Za-zÁÀÂÃÉÊÍÓÔÕÚÇáàâãéêíóôõúç]"),
}


def _percentiles(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def _dataset_report(dataset: ManifestDataset, tokenizer, config: TrainConfig) -> dict:
    by_language: dict[str, dict] = defaultdict(
        lambda: {"samples": 0, "duration_s": 0.0, "tokens": [], "suspicious_script": 0}
    )
    texts = Counter()
    blank = []
    special_text = []
    over_limit = []
    for row in dataset.rows:
        language = row["language"]
        text = row.get("text", "")
        texts[text] += 1
        if not text.strip():
            blank.append(row["id"])
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        item = by_language[language]
        item["samples"] += 1
        item["duration_s"] += float(row["duration_s"])
        item["tokens"].append(len(token_ids))
        pattern = SCRIPT_PATTERNS.get(language)
        if pattern and text.strip() and not pattern.search(text):
            item["suspicious_script"] += 1
        if any(token and token in text for token in tokenizer.all_special_tokens):
            special_text.append(row["id"])
        if len(token_ids) > 4096:
            over_limit.append(row["id"])
    languages = {}
    for language, item in sorted(by_language.items()):
        tokens = item.pop("tokens")
        languages[language] = {
            **item,
            "hours": item["duration_s"] / 3600,
            "transcript_tokens": sum(tokens),
            "mean_tokens": mean(tokens) if tokens else 0.0,
            "token_length": _percentiles(tokens),
        }
    duplicate_rows = sum(count - 1 for count in texts.values() if count > 1)
    exposure_samples = (
        2000 * config.batch_size * config.gradient_accumulation_steps
    )
    estimated_exposure = {}
    for language, item in languages.items():
        sample_share = item["samples"] / len(dataset)
        estimated_samples = exposure_samples * sample_share
        estimated_exposure[language] = {
            "samples": estimated_samples,
            "transcript_tokens": estimated_samples * item["mean_tokens"],
        }
    return {
        "passed": not blank and not over_limit,
        "samples": len(dataset),
        "languages": languages,
        "exact_duplicate_text_rows": duplicate_rows,
        "unique_texts": len(texts),
        "blank_text_ids": blank[:100],
        "text_with_special_token_ids": special_text[:100],
        "transcript_over_4096_ids": over_limit[:100],
        "estimated_natural_exposure_first_2000_steps": {
            "optimizer_steps": 2000,
            "total_samples": exposure_samples,
            "by_language": estimated_exposure,
            "note": "Expected exposure from manifest proportions; exact order depends on batch size and seed.",
        },
    }


def _sample_indices(dataset: ManifestDataset, count: int, seed: int) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        grouped[row["language"]].append(index)
    rng = random.Random(seed)
    result = []
    for language in sorted(grouped):
        candidates = grouped[language]
        result.extend(rng.sample(candidates, min(count, len(candidates))))
    return result


def _alignment_report(dataset, processor, indices, include_eos: bool) -> dict:
    collator = MTPDataCollator(processor, include_eos_in_loss=include_eos)
    failures = []
    samples = []
    for index in indices:
        row = dataset[index]
        batch = collator([row])
        active = batch["attention_mask"][0].bool()
        loss_mask = batch["loss_mask"][0]
        loss_positions = loss_mask.nonzero(as_tuple=False).flatten().tolist()
        selected = batch["input_ids"][0][loss_mask].tolist()
        expected = processor.tokenizer.encode(row["text"], add_special_tokens=False)
        contiguous = not loss_positions or loss_positions == list(
            range(loss_positions[0], loss_positions[-1] + 1)
        )
        no_padding_loss = not bool((loss_mask & ~active).any())
        text_matches = selected == expected
        passed = bool(loss_positions) and contiguous and no_padding_loss and text_matches
        if not passed:
            failures.append(row["id"])
        samples.append(
            {
                "id": row["id"],
                "language": row["language"],
                "sequence_length": int(active.sum()),
                "loss_start": loss_positions[0] if loss_positions else None,
                "loss_end": loss_positions[-1] if loss_positions else None,
                "loss_tokens": len(selected),
                "expected_text_tokens": len(expected),
                "contiguous": contiguous,
                "no_padding_loss": no_padding_loss,
                "text_tokens_match": text_matches,
            }
        )
    return {
        "passed": not failures,
        "include_eos_in_loss": include_eos,
        "checked_samples": len(samples),
        "failure_ids": failures,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Audit MTP transcript tokens and loss-mask alignment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--samples-per-language", type=int, default=100)
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    dataset = ManifestDataset(config.resolve_manifest(config.train_manifest), config.dataset_root)
    processor = Qwen3ASRProcessor.from_pretrained(config.model_path, fix_mistral_regex=True)
    processor.tokenizer.padding_side = "right"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_report = _dataset_report(dataset, processor.tokenizer, config)
    indices = _sample_indices(dataset, args.samples_per_language, config.seed)
    alignment_report = _alignment_report(
        dataset, processor, indices, config.include_eos_in_loss
    )
    (output_dir / "dataset_token_audit.json").write_text(
        json.dumps(dataset_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "alignment_audit.json").write_text(
        json.dumps(alignment_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"dataset": dataset_report["passed"], "alignment": alignment_report["passed"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not dataset_report["passed"] or not alignment_report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
