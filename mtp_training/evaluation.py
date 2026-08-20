from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .objectives import normalized_branch_weights, strict_acceptance


def _empty_stats(depth: int) -> dict[str, Any]:
    return {
        "main_loss_sum": 0.0,
        "main_correct": 0,
        "main_valid": 0,
        "branch_loss_sum": [0.0] * depth,
        "ground_truth_correct": [0] * depth,
        "ground_truth_valid": [0] * depth,
        "backbone_correct": [0] * depth,
        "backbone_valid": [0] * depth,
        "ground_truth_accepted": 0.0,
        "ground_truth_positions": 0,
        "backbone_accepted": 0.0,
        "backbone_positions": 0,
    }


def _select(values: list[torch.Tensor], rows: list[int]) -> list[torch.Tensor]:
    return [value[rows] for value in values]


def _update(item: dict[str, Any], output, rows: list[int]) -> None:
    main_valid = output.main_valid[rows]
    item["main_loss_sum"] += float(
        (output.main_token_losses[rows] * main_valid).sum()
    )
    item["main_correct"] += int((output.main_correct[rows] & main_valid).sum())
    item["main_valid"] += int(main_valid.sum())

    gt_correct = _select(output.branch_correct, rows)
    gt_valid = _select(output.branch_valid, rows)
    bb_correct = _select(output.branch_backbone_correct, rows)
    bb_valid = _select(output.branch_backbone_valid, rows)
    branch_token_losses = _select(output.branch_token_losses, rows)
    for branch_index, (losses, correct, valid, consistency, consistency_valid) in enumerate(
        zip(branch_token_losses, gt_correct, gt_valid, bb_correct, bb_valid)
    ):
        item["branch_loss_sum"][branch_index] += float((losses * valid).sum())
        item["ground_truth_correct"][branch_index] += int((correct & valid).sum())
        item["ground_truth_valid"][branch_index] += int(valid.sum())
        item["backbone_correct"][branch_index] += int(
            (consistency & consistency_valid).sum()
        )
        item["backbone_valid"][branch_index] += int(consistency_valid.sum())

    accepted, positions = strict_acceptance(gt_correct, gt_valid)
    item["ground_truth_accepted"] += float(accepted)
    item["ground_truth_positions"] += int(positions)
    accepted, positions = strict_acceptance(bb_correct, bb_valid)
    item["backbone_accepted"] += float(accepted)
    item["backbone_positions"] += int(positions)


@torch.no_grad()
def evaluate(model, loader, stage: int, device: torch.device, max_batches: int) -> dict[str, Any]:
    model.eval()
    stats = defaultdict(lambda: _empty_stats(model.depth))
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        languages = batch.pop("languages")
        batch.pop("sample_ids")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast:
            output = model(stage=stage, **batch)
        for key, row_indices in _group_indices(languages).items():
            _update(stats[key], output, row_indices)
        _update(stats["all"], output, list(range(len(languages))))
    model.train()
    weights = normalized_branch_weights(model.depth, model.alpha).tolist()
    return {
        key: _finalize(value, weights, stage)
        for key, value in sorted(stats.items())
    }


def _group_indices(languages: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(languages):
        groups[language].append(index)
    return groups


def _ratios(numerators: list[int], denominators: list[int]) -> list[float]:
    return [
        numerator / max(denominator, 1)
        for numerator, denominator in zip(numerators, denominators)
    ]


def _finalize(stats: dict[str, Any], weights: list[float], stage: int) -> dict[str, Any]:
    main_loss = stats["main_loss_sum"] / max(stats["main_valid"], 1)
    branch_losses = [
        value / max(valid, 1)
        for value, valid in zip(stats["branch_loss_sum"], stats["ground_truth_valid"])
    ]
    mtp_loss = sum(weight * loss for weight, loss in zip(weights, branch_losses))
    gt_accuracy = _ratios(stats["ground_truth_correct"], stats["ground_truth_valid"])
    bb_accuracy = _ratios(stats["backbone_correct"], stats["backbone_valid"])
    gt_accepted = stats["ground_truth_accepted"] / max(
        stats["ground_truth_positions"], 1
    )
    bb_accepted = stats["backbone_accepted"] / max(stats["backbone_positions"], 1)
    return {
        "loss": mtp_loss if stage == 1 else main_loss + mtp_loss,
        "main_loss": main_loss,
        "main_accuracy": stats["main_correct"] / max(stats["main_valid"], 1),
        "main_valid_tokens": stats["main_valid"],
        "branch_losses": branch_losses,
        "ground_truth_branch_accuracy": gt_accuracy,
        "ground_truth_branch_valid_tokens": stats["ground_truth_valid"],
        "ground_truth_average_accepted_length": gt_accepted,
        "ground_truth_eligible_positions": stats["ground_truth_positions"],
        "backbone_consistency_branch_accuracy": bb_accuracy,
        "backbone_consistency_branch_valid_tokens": stats["backbone_valid"],
        "backbone_consistency_average_accepted_length": bb_accepted,
        "backbone_consistency_eligible_positions": stats["backbone_positions"],
        # Backward-compatible aliases for existing log parsers.
        "branch_accuracy": gt_accuracy,
        "average_accepted_length": gt_accepted,
        "eligible_positions": stats["ground_truth_positions"],
    }
