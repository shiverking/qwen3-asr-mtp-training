from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

from .config import TrainConfig
from .data import LANGUAGE_NAMES, ManifestDataset


def _distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _units(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    if language == "zh-CN":
        return [character for character in text if not character.isspace()]
    return re.findall(r"\w+", text, flags=re.UNICODE)


def _select(dataset, languages: list[str], count: int, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in dataset.rows:
        if row["language"] in languages:
            grouped[row["language"]].append(row)
    rng = random.Random(seed)
    selected = []
    for language in languages:
        candidates = grouped[language]
        if len(candidates) < count:
            raise ValueError(f"{language} has {len(candidates)} rows, needs {count}")
        selected.extend(rng.sample(candidates, count))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate the official Qwen3-ASR decoding path")
    parser.add_argument("--config", required=True)
    parser.add_argument("--languages", nargs="+", default=["zh-CN", "en"])
    parser.add_argument("--samples-per-language", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="reports/backbone-asr-eval.json")
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    unknown = sorted(set(args.languages) - set(LANGUAGE_NAMES))
    if unknown:
        raise ValueError(f"Unsupported languages: {unknown}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    dataset = ManifestDataset(
        config.resolve_manifest(config.eval_manifest), config.dataset_root
    )
    rows = _select(dataset, args.languages, args.samples_per_language, config.seed)
    asr = Qwen3ASRModel.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=config.attn_implementation,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=512,
    )
    results = []
    totals = defaultdict(lambda: {"edits": 0, "reference_units": 0, "samples": 0})
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        outputs = asr.transcribe(
            audio=[str(dataset.dataset_root / row["audio"]) for row in chunk],
            language=[LANGUAGE_NAMES[row["language"]] for row in chunk],
        )
        for row, output in zip(chunk, outputs):
            reference = row.get("text_eval") or row["text"]
            reference_units = _units(reference, row["language"])
            hypothesis_units = _units(output.text, row["language"])
            edits = _distance(reference_units, hypothesis_units)
            item = totals[row["language"]]
            item["edits"] += edits
            item["reference_units"] += len(reference_units)
            item["samples"] += 1
            results.append(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "source": row.get("source", "unknown"),
                    "reference": reference,
                    "hypothesis": output.text,
                    "edits": edits,
                    "reference_units": len(reference_units),
                }
            )
    summary = {
        language: {
            **item,
            "error_rate": item["edits"] / max(item["reference_units"], 1),
            "metric": "CER" if language == "zh-CN" else "WER",
        }
        for language, item in sorted(totals.items())
    }
    report = {"summary": summary, "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
