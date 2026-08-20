from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

from .checkpointing import load_trainable_weights
from .config import TrainConfig
from .data import MTPDataCollator, ManifestDataset
from .modeling_mtp import Qwen3ASRMTPModel
from .reference_verifier import speculative_greedy_reference
from .train import seed_everything


def _stratified_indices(dataset, total: int, seed: int) -> list[int]:
    grouped = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        grouped[row["language"]].append(index)
    rng = random.Random(seed)
    languages = sorted(grouped)
    base, remainder = divmod(total, len(languages))
    selected = []
    for index, language in enumerate(languages):
        count = base + (1 if index < remainder else 0)
        selected.extend(rng.sample(grouped[language], min(count, len(grouped[language]))))
    return selected


def _prefix_batch(collator, row, device) -> dict:
    batch = collator([row])
    loss_positions = batch["loss_mask"][0].nonzero(as_tuple=False).flatten()
    if loss_positions.numel() == 0:
        raise ValueError(f"No transcript tokens for {row['id']}")
    prefix_length = int(loss_positions[0])
    batch.pop("languages")
    batch.pop("sample_ids")
    for key in ("input_ids", "attention_mask", "loss_mask"):
        batch[key] = batch[key][:, :prefix_length]
    return {key: value.to(device) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser("Run the slow MTP speculative-greedy reference verifier")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="reports/reference-verifier.json")
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    seed_everything(config.seed)
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
    model.configure_trainable(config.stage)
    load_trainable_weights(model, args.checkpoint)
    device = torch.device("cuda:0")
    model.to(device).eval()
    dataset = ManifestDataset(config.resolve_manifest(config.eval_manifest), config.dataset_root)
    collator = MTPDataCollator(wrapper.processor, include_eos_in_loss=False)
    results = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    for index in _stratified_indices(dataset, args.samples, config.seed):
        row = dataset[index]
        prefix = _prefix_batch(collator, row, device)
        with autocast:
            result = speculative_greedy_reference(
                model,
                prefix,
                wrapper.processor.tokenizer.eos_token_id,
                args.max_new_tokens,
            )
        result.update({"id": row["id"], "language": row["language"]})
        results.append(result)
    by_language = defaultdict(list)
    for result in results:
        by_language[result["language"]].append(result["average_accepted_length"])
    summary = {
        "samples": len(results),
        "average_accepted_length": sum(
            value for values in by_language.values() for value in values
        )
        / max(len(results), 1),
        "by_language": {
            key: sum(values) / len(values) for key, values in sorted(by_language.items())
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
