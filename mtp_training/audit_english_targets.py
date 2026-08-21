from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from qwen_asr import Qwen3ASRModel
from tqdm.auto import tqdm

from .data import LANGUAGE_NAMES, ManifestDataset
from .evaluate_backbone_asr import _distance


def _normalized_units(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower().strip()
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


def _raw_units(text: str) -> list[str]:
    return unicodedata.normalize("NFKC", text).strip().split()


def _common_prefix_length(reference: list[int], hypothesis: list[int]) -> int:
    length = 0
    for reference_id, hypothesis_id in zip(reference, hypothesis):
        if reference_id != hypothesis_id:
            break
        length += 1
    return length


@dataclass
class AuditGroup:
    name: str
    manifest: str
    source: str
    samples: int
    export_ab_manifests: bool = False
    export_reference_manifest: bool = False


@dataclass
class EnglishTargetAuditConfig:
    model_path: str
    dataset_root: str
    output_dir: str
    groups: list[AuditGroup]
    language: str = "en"
    seed: int = 20260819
    batch_size: int = 8
    attn_implementation: str = "sdpa"
    max_new_tokens: int = 512
    max_normalized_wer_for_export: float = 0.2

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnglishTargetAuditConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown audit config keys: {sorted(unknown)}")
        raw_groups = values.pop("groups", [])
        group_fields = {field.name for field in fields(AuditGroup)}
        groups = []
        for raw_group in raw_groups:
            unknown_group = set(raw_group) - group_fields
            if unknown_group:
                raise ValueError(
                    f"Unknown keys in audit group: {sorted(unknown_group)}"
                )
            groups.append(AuditGroup(**raw_group))
        config = cls(groups=groups, **values)
        if not config.groups:
            raise ValueError("At least one audit group is required")
        if config.language not in LANGUAGE_NAMES:
            raise ValueError(f"Unsupported language: {config.language}")
        if config.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= config.max_normalized_wer_for_export <= 1.0:
            raise ValueError("max_normalized_wer_for_export must be between 0 and 1")
        for group in config.groups:
            if group.samples < 1:
                raise ValueError(f"{group.name}: samples must be >= 1")
        return config

    def resolve_manifest(self, manifest: str) -> Path:
        path = Path(manifest)
        return path if path.is_absolute() else Path(self.dataset_root) / path


def _select_rows(
    dataset: ManifestDataset,
    group: AuditGroup,
    language: str,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in dataset.rows
        if row["language"] == language and row.get("source") == group.source
    ]
    if len(candidates) < group.samples:
        raise ValueError(
            f"{group.name}: source={group.source} has {len(candidates)} rows, "
            f"needs {group.samples}"
        )
    rng = random.Random(f"{seed}:{group.name}")
    return rng.sample(candidates, group.samples)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(results)
    raw_reference_words = sum(item["raw_reference_words"] for item in results)
    normalized_reference_words = sum(
        item["normalized_reference_words"] for item in results
    )
    return {
        "samples": samples,
        "raw_wer": sum(item["raw_edits"] for item in results)
        / max(raw_reference_words, 1),
        "normalized_wer": sum(item["normalized_edits"] for item in results)
        / max(normalized_reference_words, 1),
        "exact_token_match_rate": sum(item["exact_token_match"] for item in results)
        / max(samples, 1),
        "normalized_text_match_rate": sum(
            item["normalized_text_match"] for item in results
        )
        / max(samples, 1),
        "mean_common_token_prefix": sum(
            item["common_token_prefix"] for item in results
        )
        / max(samples, 1),
        "mean_common_token_prefix_ratio": sum(
            item["common_token_prefix_ratio"] for item in results
        )
        / max(samples, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        "Audit English human references against Qwen3-ASR greedy targets"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = EnglishTargetAuditConfig.from_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    model = Qwen3ASRModel.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=config.attn_implementation,
        max_inference_batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
    )
    tokenizer = model.processor.tokenizer
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"groups": {}}

    for group in config.groups:
        dataset = ManifestDataset(
            str(config.resolve_manifest(group.manifest)), config.dataset_root
        )
        rows = _select_rows(dataset, group, config.language, config.seed)
        results = []
        complete_reference_manifest = []
        reference_manifest = []
        greedy_manifest = []
        progress = tqdm(
            range(0, len(rows), config.batch_size),
            desc=group.name,
            unit="batch",
        )
        for start in progress:
            chunk = rows[start : start + config.batch_size]
            outputs = model.transcribe(
                audio=[str(dataset.dataset_root / row["audio"]) for row in chunk],
                language=[LANGUAGE_NAMES[config.language]] * len(chunk),
            )
            for row, output in zip(chunk, outputs):
                reference = row["text"]
                hypothesis = output.text
                raw_reference = _raw_units(reference)
                raw_hypothesis = _raw_units(hypothesis)
                normalized_reference = _normalized_units(reference)
                normalized_hypothesis = _normalized_units(hypothesis)
                raw_edits = _distance(raw_reference, raw_hypothesis)
                normalized_edits = _distance(
                    normalized_reference, normalized_hypothesis
                )
                reference_ids = tokenizer.encode(
                    reference, add_special_tokens=False
                )
                hypothesis_ids = tokenizer.encode(
                    hypothesis, add_special_tokens=False
                )
                common_prefix = _common_prefix_length(
                    reference_ids, hypothesis_ids
                )
                normalized_wer = normalized_edits / max(
                    len(normalized_reference), 1
                )
                result = {
                    "id": row["id"],
                    "source": row.get("source"),
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "raw_edits": raw_edits,
                    "raw_reference_words": len(raw_reference),
                    "normalized_edits": normalized_edits,
                    "normalized_reference_words": len(normalized_reference),
                    "normalized_wer": normalized_wer,
                    "normalized_text_match": (
                        normalized_reference == normalized_hypothesis
                    ),
                    "reference_token_ids": reference_ids,
                    "hypothesis_token_ids": hypothesis_ids,
                    "exact_token_match": reference_ids == hypothesis_ids,
                    "common_token_prefix": common_prefix,
                    "common_token_prefix_ratio": common_prefix
                    / max(len(reference_ids), 1),
                }
                results.append(result)
                if group.export_reference_manifest:
                    reference_row = dict(row)
                    reference_row["text_reference"] = reference
                    complete_reference_manifest.append(reference_row)
                if (
                    group.export_ab_manifests
                    and normalized_wer <= config.max_normalized_wer_for_export
                    and hypothesis.strip()
                ):
                    reference_row = dict(row)
                    reference_row["text_reference"] = reference
                    reference_manifest.append(reference_row)
                    greedy_row = dict(reference_row)
                    greedy_row["text"] = hypothesis
                    greedy_row["text_mtp_target"] = hypothesis
                    greedy_row["mtp_target_source"] = "qwen3_asr_greedy"
                    greedy_manifest.append(greedy_row)

        summary = _summarize(results)
        group_report: dict[str, Any] = {
            "manifest": str(config.resolve_manifest(group.manifest)),
            "source": group.source,
            "summary": summary,
            "results": results,
        }
        if group.export_reference_manifest:
            complete_reference_path = output_dir / f"{group.name}-reference.jsonl"
            _write_jsonl(complete_reference_path, complete_reference_manifest)
            group_report["reference_export"] = {
                "samples": len(complete_reference_manifest),
                "manifest": str(complete_reference_path),
            }
        if group.export_ab_manifests:
            reference_path = output_dir / f"{group.name}-reference.jsonl"
            greedy_path = output_dir / f"{group.name}-greedy.jsonl"
            _write_jsonl(reference_path, reference_manifest)
            _write_jsonl(greedy_path, greedy_manifest)
            group_report["ab_export"] = {
                "quality_threshold_normalized_wer": (
                    config.max_normalized_wer_for_export
                ),
                "samples": len(greedy_manifest),
                "reference_manifest": str(reference_path),
                "greedy_manifest": str(greedy_path),
            }
        report["groups"][group.name] = group_report
        print(
            f"{group.name} samples={summary['samples']} "
            f"raw_wer={summary['raw_wer']:.4f} "
            f"normalized_wer={summary['normalized_wer']:.4f} "
            f"normalized_match={summary['normalized_text_match_rate']:.4f} "
            f"exact_tokens={summary['exact_token_match_rate']:.4f} "
            f"prefix_ratio={summary['mean_common_token_prefix_ratio']:.4f}"
        )

    report_path = output_dir / "english-target-audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
