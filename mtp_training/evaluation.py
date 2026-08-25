from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .objectives import (
    decode_window_validity,
    legacy_strict_acceptance,
    normalized_branch_weights,
    strict_acceptance,
)


METRIC_VERSION = 2


def _empty_stats(depth: int) -> dict[str, Any]:
    return {
        "main_loss_sum": 0.0,
        "main_correct": 0,
        "main_valid": 0,
        "branch_loss_sum": [0.0] * depth,
        "training_correct": [0] * depth,
        "training_valid": [0] * depth,
        "legacy_backbone_correct": [0] * depth,
        "legacy_backbone_valid": [0] * depth,
        "decode_gt_correct": [0] * depth,
        "decode_gt_valid": [0] * depth,
        "decode_bb_correct": [0] * depth,
        "decode_bb_valid": [0] * depth,
        "legacy_gt_accepted": 0.0,
        "legacy_gt_positions": 0,
        "legacy_bb_accepted": 0.0,
        "legacy_bb_positions": 0,
        "decode_gt_accepted": 0.0,
        "decode_gt_positions": 0,
        "decode_bb_accepted": 0.0,
        "decode_bb_positions": 0,
        "strict_position_accepted": [0] * depth,
        "strict_position_eligible": [0] * depth,
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

    training_correct = _select(output.branch_correct, rows)
    training_valid = _select(output.branch_valid, rows)
    legacy_bb_correct = _select(output.branch_backbone_correct, rows)
    legacy_bb_valid = _select(output.branch_backbone_valid, rows)
    branch_token_losses = _select(output.branch_token_losses, rows)
    decode_gt_valid = decode_window_validity(main_valid, training_valid)
    decode_bb_valid = decode_window_validity(main_valid, legacy_bb_valid)
    decode_gt_correct = [
        correct[:, : valid.shape[1]] & valid
        for correct, valid in zip(training_correct, decode_gt_valid)
    ]
    decode_bb_correct = [
        correct[:, : valid.shape[1]] & valid
        for correct, valid in zip(legacy_bb_correct, decode_bb_valid)
    ]

    for branch_index, values in enumerate(
        zip(
            branch_token_losses,
            training_correct,
            training_valid,
            legacy_bb_correct,
            legacy_bb_valid,
            decode_gt_correct,
            decode_gt_valid,
            decode_bb_correct,
            decode_bb_valid,
        )
    ):
        (
            losses,
            target_correct,
            target_valid,
            backbone_correct,
            backbone_valid,
            window_gt_correct,
            window_gt_valid,
            window_bb_correct,
            window_bb_valid,
        ) = values
        item["branch_loss_sum"][branch_index] += float((losses * target_valid).sum())
        item["training_correct"][branch_index] += int(
            (target_correct & target_valid).sum()
        )
        item["training_valid"][branch_index] += int(target_valid.sum())
        item["legacy_backbone_correct"][branch_index] += int(
            (backbone_correct & backbone_valid).sum()
        )
        item["legacy_backbone_valid"][branch_index] += int(backbone_valid.sum())
        item["decode_gt_correct"][branch_index] += int(window_gt_correct.sum())
        item["decode_gt_valid"][branch_index] += int(window_gt_valid.sum())
        item["decode_bb_correct"][branch_index] += int(window_bb_correct.sum())
        item["decode_bb_valid"][branch_index] += int(window_bb_valid.sum())

    accepted, positions = legacy_strict_acceptance(training_correct, training_valid)
    item["legacy_gt_accepted"] += float(accepted)
    item["legacy_gt_positions"] += int(positions)
    accepted, positions = legacy_strict_acceptance(legacy_bb_correct, legacy_bb_valid)
    item["legacy_bb_accepted"] += float(accepted)
    item["legacy_bb_positions"] += int(positions)
    accepted, positions = strict_acceptance(decode_gt_correct, decode_gt_valid)
    item["decode_gt_accepted"] += float(accepted)
    item["decode_gt_positions"] += int(positions)
    if decode_gt_correct:
        width = min(value.shape[1] for value in decode_gt_correct)
        eligible = torch.ones_like(decode_gt_valid[0][:, :width])
        for valid in decode_gt_valid:
            eligible &= valid[:, :width]
        prefix_correct = torch.ones_like(eligible)
        for branch_index, (correct, valid) in enumerate(
            zip(decode_gt_correct, decode_gt_valid)
        ):
            prefix_correct &= correct[:, :width]
            item["strict_position_accepted"][branch_index] += int(
                (eligible & prefix_correct).sum()
            )
            item["strict_position_eligible"][branch_index] += int(eligible.sum())
    accepted, positions = strict_acceptance(decode_bb_correct, decode_bb_valid)
    item["decode_bb_accepted"] += float(accepted)
    item["decode_bb_positions"] += int(positions)


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
    result = {
        key: _finalize(value, weights, stage)
        for key, value in sorted(stats.items())
    }
    language_metrics = [value for key, value in result.items() if key != "all"]
    if language_metrics:
        result["macro_average"] = _macro_average(language_metrics)
    return result


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
        for value, valid in zip(stats["branch_loss_sum"], stats["training_valid"])
    ]
    mtp_loss = sum(weight * loss for weight, loss in zip(weights, branch_losses))
    training_accuracy = _ratios(stats["training_correct"], stats["training_valid"])
    legacy_bb_accuracy = _ratios(
        stats["legacy_backbone_correct"], stats["legacy_backbone_valid"]
    )
    decode_gt_accuracy = _ratios(stats["decode_gt_correct"], stats["decode_gt_valid"])
    decode_bb_accuracy = _ratios(stats["decode_bb_correct"], stats["decode_bb_valid"])
    legacy_gt_accepted = stats["legacy_gt_accepted"] / max(
        stats["legacy_gt_positions"], 1
    )
    legacy_bb_accepted = stats["legacy_bb_accepted"] / max(
        stats["legacy_bb_positions"], 1
    )
    decode_gt_accepted = stats["decode_gt_accepted"] / max(
        stats["decode_gt_positions"], 1
    )
    decode_bb_accepted = stats["decode_bb_accepted"] / max(
        stats["decode_bb_positions"], 1
    )
    strict_position = _ratios(
        stats["strict_position_accepted"], stats["strict_position_eligible"]
    )
    conditional_position = []
    for index, accepted in enumerate(stats["strict_position_accepted"]):
        denominator = (
            stats["strict_position_eligible"][0]
            if index == 0
            else stats["strict_position_accepted"][index - 1]
        )
        conditional_position.append(accepted / max(denominator, 1))
    strict_average = 1.0 + sum(strict_position)
    return {
        "metric_version": METRIC_VERSION,
        "loss": mtp_loss if stage == 1 else main_loss + mtp_loss,
        "main_loss": main_loss,
        "main_accuracy": stats["main_correct"] / max(stats["main_valid"], 1),
        "main_valid_tokens": stats["main_valid"],
        "branch_losses": branch_losses,
        "training_target_branch_accuracy": training_accuracy,
        "training_target_branch_valid_tokens": stats["training_valid"],
        "decode_window_ground_truth_branch_accuracy": decode_gt_accuracy,
        "decode_window_ground_truth_branch_valid_tokens": stats["decode_gt_valid"],
        "decode_window_ground_truth_average_accepted_length": decode_gt_accepted,
        "decode_window_ground_truth_eligible_positions": stats["decode_gt_positions"],
        "strict_position_acceptance": strict_position,
        "conditional_position_acceptance": conditional_position,
        "strict_position_verification_counts": stats["strict_position_eligible"],
        "strict_average_accepted_length": strict_average,
        "decode_window_backbone_consistency_branch_accuracy": decode_bb_accuracy,
        "decode_window_backbone_consistency_branch_valid_tokens": stats["decode_bb_valid"],
        "decode_window_backbone_consistency_average_accepted_length": decode_bb_accepted,
        "decode_window_backbone_consistency_eligible_positions": stats["decode_bb_positions"],
        "legacy_ground_truth_branch_accuracy": training_accuracy,
        "legacy_ground_truth_average_accepted_length": legacy_gt_accepted,
        "legacy_ground_truth_eligible_positions": stats["legacy_gt_positions"],
        "legacy_backbone_consistency_branch_accuracy": legacy_bb_accuracy,
        "legacy_backbone_consistency_average_accepted_length": legacy_bb_accepted,
        "legacy_backbone_consistency_eligible_positions": stats["legacy_bb_positions"],
        # Exact version-1 compatibility aliases.
        "ground_truth_branch_accuracy": training_accuracy,
        "ground_truth_branch_valid_tokens": stats["training_valid"],
        "ground_truth_average_accepted_length": legacy_gt_accepted,
        "ground_truth_eligible_positions": stats["legacy_gt_positions"],
        "backbone_consistency_branch_accuracy": legacy_bb_accuracy,
        "backbone_consistency_branch_valid_tokens": stats["legacy_backbone_valid"],
        "backbone_consistency_average_accepted_length": legacy_bb_accepted,
        "backbone_consistency_eligible_positions": stats["legacy_bb_positions"],
        "branch_accuracy": training_accuracy,
        "average_accepted_length": legacy_gt_accepted,
        "eligible_positions": stats["legacy_gt_positions"],
    }


def _macro_average(items: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_fields = (
        "loss",
        "main_loss",
        "main_accuracy",
        "decode_window_ground_truth_average_accepted_length",
        "decode_window_backbone_consistency_average_accepted_length",
        "legacy_ground_truth_average_accepted_length",
        "legacy_backbone_consistency_average_accepted_length",
        "strict_average_accepted_length",
    )
    vector_fields = (
        "branch_losses",
        "training_target_branch_accuracy",
        "decode_window_ground_truth_branch_accuracy",
        "decode_window_backbone_consistency_branch_accuracy",
        "strict_position_acceptance",
        "conditional_position_acceptance",
    )
    result: dict[str, Any] = {
        "metric_version": METRIC_VERSION,
        "languages": len(items),
    }
    for field in scalar_fields:
        result[field] = sum(item[field] for item in items) / len(items)
    for field in vector_fields:
        result[field] = [
            sum(item[field][index] for item in items) / len(items)
            for index in range(len(items[0][field]))
        ]
    return result
