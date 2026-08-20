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
    audit_reference_equivalence,
    audit_trainable_parameters,
)
from .evaluation import evaluate
from .modeling_mtp import Qwen3ASRMTPModel
from .train import build_loader, seed_everything


def _sample_ids_from_report(path: str | None) -> list[str] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Verifier report must contain a results list")
    sample_ids = [item.get("id") for item in results]
    if not sample_ids or any(not isinstance(item, str) for item in sample_ids):
        raise ValueError("Every verifier result must contain a string id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Verifier report contains duplicate sample ids")
    return sample_ids


def _filter_dataset_rows(dataset, sample_ids: list[str] | None) -> None:
    if sample_ids is None:
        return
    rows_by_id = {row["id"]: row for row in dataset.rows}
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
    if missing:
        raise ValueError(f"Sample ids missing from eval manifest: {missing[:10]}")
    dataset.rows = [rows_by_id[sample_id] for sample_id in sample_ids]


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate and diagnose a completed MTP checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="reports/checkpoint-diagnostics.json")
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument(
        "--sample-ids-from",
        help="Restrict evaluation to ids in a verify_checkpoint JSON report",
    )
    parser.add_argument("--equivalence-samples", type=int, default=8)
    parser.add_argument(
        "--deterministic-causality",
        action="store_true",
        help="Gate on a float32, batch-size-one causality audit",
    )
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
    initialization = audit_initialization(model)
    model.configure_trainable(config.stage)
    trainable = audit_trainable_parameters(model, config.stage)
    load_trainable_weights(model, args.checkpoint)
    device = torch.device("cuda:0")
    model.to(device)
    dataset = ManifestDataset(
        config.resolve_manifest(config.eval_manifest), config.dataset_root
    )
    _filter_dataset_rows(dataset, _sample_ids_from_report(args.sample_ids_from))
    loader = build_loader(dataset, wrapper.processor, config, train=False)
    first = next(iter(loader))
    first.pop("languages")
    sample_ids = first.pop("sample_ids")
    first = {key: value.to(device, non_blocking=True) for key, value in first.items()}
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        operational_causality = audit_future_token_causality(model, config.stage, first)
        equivalence = audit_reference_equivalence(
            model,
            config.stage,
            first,
            sample_ids=sample_ids,
            max_samples=args.equivalence_samples,
        )
    deterministic_causality = None
    if args.deterministic_causality:
        model.float()
        fp32_first = {key: value[:1] for key, value in first.items()}
        deterministic_causality = audit_future_token_causality(
            model, config.stage, fp32_first
        )
        model.bfloat16()
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
        "future_token_causality": {
            "operational_bf16": operational_causality,
            "deterministic_fp32": deterministic_causality,
            "gate_mode": (
                "deterministic_fp32" if args.deterministic_causality else "not_run"
            ),
        },
        "reference_equivalence": equivalence,
        "gradients": gradients,
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    causality_passed = (
        deterministic_causality["passed"]
        if deterministic_causality is not None
        else True
    )
    if (
        not initialization["passed"]
        or not trainable["passed"]
        or not causality_passed
        or not equivalence["passed"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
