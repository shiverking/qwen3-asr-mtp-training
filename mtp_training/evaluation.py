from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .objectives import strict_acceptance


def _empty_stats(depth: int) -> dict[str, Any]:
    return {
        "loss_sum": 0.0,
        "loss_batches": 0,
        "correct": [0] * depth,
        "valid": [0] * depth,
        "accepted": 0.0,
        "positions": 0,
    }


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
            item = stats[key]
            selected_correct = [value[row_indices] for value in output.branch_correct]
            selected_valid = [value[row_indices] for value in output.branch_valid]
            for branch_index, (correct, valid) in enumerate(zip(selected_correct, selected_valid)):
                item["correct"][branch_index] += int((correct & valid).sum())
                item["valid"][branch_index] += int(valid.sum())
            accepted, positions = strict_acceptance(selected_correct, selected_valid)
            item["accepted"] += float(accepted)
            item["positions"] += int(positions)
        all_rows = list(range(len(languages)))
        item = stats["all"]
        item["loss_sum"] += float(output.loss.detach())
        item["loss_batches"] += 1
        for branch_index, (correct, valid) in enumerate(zip(output.branch_correct, output.branch_valid)):
            item["correct"][branch_index] += int((correct[all_rows] & valid[all_rows]).sum())
            item["valid"][branch_index] += int(valid[all_rows].sum())
        accepted, positions = strict_acceptance(output.branch_correct, output.branch_valid)
        item["accepted"] += float(accepted)
        item["positions"] += int(positions)
    model.train()
    return {key: _finalize(value) for key, value in sorted(stats.items())}


def _group_indices(languages: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(languages):
        groups[language].append(index)
    return groups


def _finalize(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": (
            stats["loss_sum"] / stats["loss_batches"]
            if stats["loss_batches"]
            else None
        ),
        "branch_accuracy": [
            correct / max(valid, 1)
            for correct, valid in zip(stats["correct"], stats["valid"])
        ],
        "average_accepted_length": stats["accepted"] / max(stats["positions"], 1),
        "eligible_positions": stats["positions"],
    }
