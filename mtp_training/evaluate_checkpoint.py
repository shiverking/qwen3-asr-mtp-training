from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

from .checkpointing import load_trainable_weights
from .config import TrainConfig
from .data import ManifestDataset
from .diagnostics import (
    audit_future_token_causality,
    audit_gradients,
    audit_initialization,
    audit_trainable_parameters,
)
from .evaluation import evaluate
from .modeling_mtp import Qwen3ASRMTPModel
from .train import build_loader, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate and diagnose a completed MTP checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="reports/checkpoint-diagnostics.json")
    parser.add_argument("--gradient-check", action="store_true")
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
    initialization = audit_initialization(model)
    model.configure_trainable(config.stage)
    trainable = audit_trainable_parameters(model, config.stage)
    load_trainable_weights(model, args.checkpoint)
    device = torch.device("cuda:0")
    model.to(device)
    dataset = ManifestDataset(
        config.resolve_manifest(config.eval_manifest), config.dataset_root
    )
    loader = build_loader(dataset, wrapper.processor, config, train=False)
    first = next(iter(loader))
    first.pop("languages")
    first.pop("sample_ids")
    first = {key: value.to(device, non_blocking=True) for key, value in first.items()}
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        causality = audit_future_token_causality(model, config.stage, first)
    gradients = None
    if args.gradient_check:
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(stage=config.stage, **first)
        output.loss.backward()
        gradients = audit_gradients(model, config.stage)
        model.zero_grad(set_to_none=True)
    metrics = evaluate(model, loader, config.stage, device, config.eval_batches)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "initialization": initialization,
        "trainable_parameters": trainable,
        "future_token_causality": causality,
        "gradients": gradients,
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not initialization["passed"] or not trainable["passed"] or not causality["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
