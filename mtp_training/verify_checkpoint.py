from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

from .checkpointing import load_trainable_weights
from .config import TrainConfig
from .data import MTPDataCollator, ManifestDataset
from .modeling_mtp import Qwen3ASRMTPModel
from .reference_verifier import evaluate_speculative_reference
from .train import seed_everything


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
        config.loss_reduction,
    )
    model.configure_trainable(config.stage)
    load_trainable_weights(model, args.checkpoint)
    device = torch.device("cuda:0")
    model.to(device).eval()
    dataset = ManifestDataset(config.resolve_manifest(config.eval_manifest), config.dataset_root)
    collator = MTPDataCollator(wrapper.processor, include_eos_in_loss=False)
    summary = evaluate_speculative_reference(
        model,
        dataset,
        collator,
        device,
        wrapper.processor.tokenizer.eos_token_id,
        args.samples,
        args.max_new_tokens,
        config.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
