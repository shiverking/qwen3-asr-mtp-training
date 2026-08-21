from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from qwen_asr import Qwen3ASRModel
from tqdm.auto import tqdm

from .data import LANGUAGE_NAMES, ManifestDataset
from .evaluate_backbone_asr import _distance, _units


@dataclass
class GreedyTargetConfig:
    model_path: str
    dataset_root: str
    input_manifest: str
    output_dir: str
    model_revision: str = "local"
    batch_size: int = 8
    max_new_tokens: int = 512
    attn_implementation: str = "sdpa"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GreedyTargetConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown greedy target config keys: {sorted(unknown)}")
        config = cls(**values)
        if config.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if config.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        return config

    def resolve_input_manifest(self) -> Path:
        path = Path(self.input_manifest)
        return path if path.is_absolute() else Path(self.dataset_root) / path


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["id"]
            if sample_id in completed:
                raise ValueError(
                    f"Duplicate id in partial output at line {line_number}: {sample_id}"
                )
            completed[sample_id] = row
    return completed


def _common_prefix_length(reference: list[int], hypothesis: list[int]) -> int:
    length = 0
    for reference_id, hypothesis_id in zip(reference, hypothesis):
        if reference_id != hypothesis_id:
            break
        length += 1
    return length


def _update_stats(
    stats: dict[str, Any],
    row: dict[str, Any],
    hypothesis: str,
    tokenizer,
) -> None:
    reference = row["text"]
    reference_units = _units(reference, row["language"])
    hypothesis_units = _units(hypothesis, row["language"])
    reference_ids = tokenizer.encode(reference, add_special_tokens=False)
    hypothesis_ids = tokenizer.encode(hypothesis, add_special_tokens=False)
    prefix = _common_prefix_length(reference_ids, hypothesis_ids)
    stats["samples"] += 1
    stats["edits"] += _distance(reference_units, hypothesis_units)
    stats["reference_units"] += len(reference_units)
    stats["exact_token_matches"] += int(reference_ids == hypothesis_ids)
    stats["prefix_ratio_sum"] += prefix / max(len(reference_ids), 1)


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    samples = stats["samples"]
    return {
        "samples": samples,
        "normalized_error_rate": stats["edits"]
        / max(stats["reference_units"], 1),
        "exact_token_match_rate": stats["exact_token_matches"]
        / max(samples, 1),
        "mean_common_token_prefix_ratio": stats["prefix_ratio_sum"]
        / max(samples, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        "Generate resumable Qwen3-ASR greedy targets for an MTP manifest"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = GreedyTargetConfig.from_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "train_mtp_greedy.partial.jsonl"
    final_path = output_dir / "train_mtp_greedy.jsonl"
    rejected_path = output_dir / "train_mtp_greedy_rejected.jsonl"
    report_path = output_dir / "greedy_target_summary.json"

    dataset = ManifestDataset(
        str(config.resolve_input_manifest()), config.dataset_root
    )
    completed = _load_completed(partial_path)
    known_ids = {row["id"] for row in dataset.rows}
    unexpected = sorted(set(completed) - known_ids)
    if unexpected:
        raise ValueError(f"Partial output contains unknown ids: {unexpected[:10]}")
    pending = [row for row in dataset.rows if row["id"] not in completed]
    print(
        f"data samples={len(dataset):,} completed={len(completed):,} "
        f"pending={len(pending):,}",
        flush=True,
    )

    model = Qwen3ASRModel.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=config.attn_implementation,
        max_inference_batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
    )
    tokenizer = model.processor.tokenizer
    rejected: list[dict[str, Any]] = []
    with partial_path.open("a", encoding="utf-8", newline="\n") as output_stream:
        progress = tqdm(
            range(0, len(pending), config.batch_size),
            desc="greedy targets",
            unit="batch",
        )
        for start in progress:
            rows = pending[start : start + config.batch_size]
            outputs = model.transcribe(
                audio=[
                    str(dataset.dataset_root / Path(row["audio"]))
                    for row in rows
                ],
                language=[LANGUAGE_NAMES[row["language"]] for row in rows],
            )
            for row, result in zip(rows, outputs):
                hypothesis = result.text.strip()
                if not hypothesis:
                    rejected.append(
                        {"id": row["id"], "reason": "empty_greedy_output"}
                    )
                    continue
                generated = dict(row)
                generated["text_reference"] = row["text"]
                generated["text_mtp_target"] = hypothesis
                generated["mtp_target_source"] = "qwen3_asr_greedy"
                generated["mtp_target_model"] = config.model_path
                generated["mtp_target_revision"] = config.model_revision
                output_stream.write(
                    json.dumps(generated, ensure_ascii=False) + "\n"
                )
            output_stream.flush()

    completed = _load_completed(partial_path)
    missing = [row["id"] for row in dataset.rows if row["id"] not in completed]
    rejected_ids = {row["id"] for row in rejected}
    unresolved = [sample_id for sample_id in missing if sample_id not in rejected_ids]
    if unresolved:
        raise RuntimeError(f"Generation incomplete, unresolved ids: {unresolved[:10]}")

    stats = defaultdict(
        lambda: {
            "samples": 0,
            "edits": 0,
            "reference_units": 0,
            "exact_token_matches": 0,
            "prefix_ratio_sum": 0.0,
        }
    )
    ordered_rows = []
    for source_row in dataset.rows:
        generated = completed.get(source_row["id"])
        if generated is None:
            continue
        ordered_rows.append(generated)
        _update_stats(
            stats[
                f"{source_row['language']}::"
                f"{source_row.get('source', 'unknown')}"
            ],
            source_row,
            generated["text_mtp_target"],
            tokenizer,
        )

    temporary_final = final_path.with_suffix(".jsonl.tmp")
    with temporary_final.open("w", encoding="utf-8", newline="\n") as stream:
        for row in ordered_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_final.replace(final_path)
    with rejected_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rejected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input_manifest": str(config.resolve_input_manifest()),
        "output_manifest": str(final_path),
        "input_samples": len(dataset),
        "output_samples": len(ordered_rows),
        "rejected_samples": len(rejected),
        "by_language_source": {
            key: _finalize_stats(value) for key, value in sorted(stats.items())
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"complete output={final_path} samples={len(ordered_rows):,} "
        f"rejected={len(rejected):,} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
