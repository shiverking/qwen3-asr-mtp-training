from __future__ import annotations

from typing import Any

import torch

from .reference_verifier import draft_next_token


def audit_initialization(model) -> dict[str, Any]:
    """Verify copied decoder weights and absence of parameter sharing at init."""
    source = dict(model.thinker.model.layers[-1].named_parameters())
    branches = []
    for index, branch in enumerate(model.branches):
        copied = dict(branch.decoder_layer.named_parameters())
        mismatched = [
            name for name, value in source.items() if not torch.equal(value, copied[name])
        ]
        shared = [
            name
            for name, value in source.items()
            if value.data_ptr() == copied[name].data_ptr()
        ]
        branches.append(
            {
                "branch": index + 1,
                "decoder_mismatches": mismatched,
                "shared_decoder_parameters": shared,
                "projection_shares_backbone_storage": any(
                    branch.projection.weight.data_ptr() == value.data_ptr()
                    for value in source.values()
                ),
            }
        )
    passed = all(
        not item["decoder_mismatches"]
        and not item["shared_decoder_parameters"]
        and not item["projection_shares_backbone_storage"]
        for item in branches
    )
    return {"passed": passed, "branches": branches}


def audit_trainable_parameters(model, stage: int) -> dict[str, Any]:
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if stage == 1:
        unexpected = [name for name in trainable if not name.startswith("branches.")]
        missing_branches = not any(name.startswith("branches.") for name in trainable)
    elif stage == 2:
        allowed = (
            "branches.",
            "asr_model.thinker.model.",
            "asr_model.thinker.audio_tower.proj1.",
            "asr_model.thinker.audio_tower.proj2.",
            "asr_model.thinker.lm_head.",
        )
        unexpected = [name for name in trainable if not name.startswith(allowed)]
        missing_branches = not any(name.startswith("branches.") for name in trainable)
    else:
        raise ValueError("stage must be 1 or 2")
    frozen_audio_encoder_trainable = [
        name
        for name in trainable
        if name.startswith("asr_model.thinker.audio_tower.")
        and not name.startswith(
            (
                "asr_model.thinker.audio_tower.proj1.",
                "asr_model.thinker.audio_tower.proj2.",
            )
        )
    ]
    return {
        "passed": not unexpected and not missing_branches and not frozen_audio_encoder_trainable,
        "stage": stage,
        "trainable_parameter_tensors": len(trainable),
        "unexpected_trainable": unexpected,
        "frozen_audio_encoder_trainable": frozen_audio_encoder_trainable,
        "missing_mtp_branches": missing_branches,
    }


def audit_gradients(model, stage: int) -> dict[str, Any]:
    with_grad = [name for name, value in model.named_parameters() if value.grad is not None]
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    unexpected = sorted(set(with_grad) - trainable)
    missing = sorted(trainable - set(with_grad))
    parameter_audit = audit_trainable_parameters(model, stage)
    return {
        "passed": parameter_audit["passed"] and not unexpected,
        "unexpected_gradients": unexpected,
        # Missing gradients can be legitimate for unused parameters, so report only.
        "trainable_without_gradient": missing,
    }


@torch.no_grad()
def audit_future_token_causality(model, stage: int, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Compare repeatability with the effect of perturbing a future token."""
    original = model(stage=stage, **batch)
    repeated = model(stage=stage, **batch)
    modified = {key: value.clone() for key, value in batch.items()}
    input_ids = modified["input_ids"]
    loss_mask = modified["loss_mask"]
    changed_positions = []
    for row in range(input_ids.shape[0]):
        positions = loss_mask[row].nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            changed_positions.append(None)
            continue
        position = int(positions[-1])
        input_ids[row, position] = (input_ids[row, position] + 1) % model.thinker.model.config.vocab_size
        changed_positions.append(position)
    perturbed = model(stage=stage, **modified)
    baseline_mismatches = {"main": 0, "branches": [0] * model.depth}
    perturbed_mismatches = {"main": 0, "branches": [0] * model.depth}
    compared = {"main": 0, "branches": [0] * model.depth}
    for row, changed_position in enumerate(changed_positions):
        if changed_position is None:
            continue
        main_width = min(changed_position, original.main_predictions.shape[1])
        compared["main"] += main_width
        baseline_mismatches["main"] += int(
            original.main_predictions[row, :main_width]
            .ne(repeated.main_predictions[row, :main_width])
            .sum()
        )
        perturbed_mismatches["main"] += int(
            original.main_predictions[row, :main_width]
            .ne(perturbed.main_predictions[row, :main_width])
            .sum()
        )
        for branch_index, (before, repeat, after) in enumerate(
            zip(
                original.branch_predictions,
                repeated.branch_predictions,
                perturbed.branch_predictions,
            ),
            start=1,
        ):
            width = min(max(changed_position - branch_index, 0), before.shape[1])
            compared["branches"][branch_index - 1] += width
            baseline_mismatches["branches"][branch_index - 1] += int(
                before[row, :width].ne(repeat[row, :width]).sum()
            )
            perturbed_mismatches["branches"][branch_index - 1] += int(
                before[row, :width].ne(after[row, :width]).sum()
            )
    net_new_mismatches = {
        "main": max(
            perturbed_mismatches["main"] - baseline_mismatches["main"], 0
        ),
        "branches": [
            max(perturbed - baseline, 0)
            for perturbed, baseline in zip(
                perturbed_mismatches["branches"], baseline_mismatches["branches"]
            )
        ],
    }
    repeatable = baseline_mismatches["main"] == 0 and not any(
        baseline_mismatches["branches"]
    )
    passed = (
        repeatable
        and net_new_mismatches["main"] == 0
        and not any(net_new_mismatches["branches"])
    )
    return {
        "passed": passed,
        "repeatable": repeatable,
        "compared": compared,
        "baseline_mismatches": baseline_mismatches,
        "perturbed_mismatches": perturbed_mismatches,
        "net_new_mismatches": net_new_mismatches,
        # Compatibility alias for version-1 reports.
        "mismatches": perturbed_mismatches,
    }


def _slice_row_prefix(
    batch: dict[str, torch.Tensor], row: int, prefix_length: int
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        selected = value[row : row + 1]
        if key in ("input_ids", "attention_mask", "loss_mask"):
            selected = selected[:, :prefix_length]
        result[key] = selected
    return result


@torch.no_grad()
def audit_reference_equivalence(
    model,
    stage: int,
    batch: dict[str, torch.Tensor],
    sample_ids: list[str] | None = None,
    max_samples: int = 8,
) -> dict[str, Any]:
    """Match training-path branch predictions against reference draft inference."""
    training = model(stage=stage, **batch)
    compared = [0] * model.depth
    matches = [0] * model.depth
    failures: list[dict[str, Any]] = []
    rows = min(batch["input_ids"].shape[0], max_samples)
    for row in range(rows):
        sample_id = sample_ids[row] if sample_ids else str(row)
        for branch_index in range(1, model.depth + 1):
            branch_valid = training.branch_valid[branch_index - 1][row]
            width = branch_valid.shape[0]
            decode_valid = branch_valid & training.main_valid[row, :width]
            valid_positions = decode_valid.nonzero(as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            base_position = int(valid_positions[0])
            prefix_length = base_position + branch_index + 1
            prefix = _slice_row_prefix(batch, row, prefix_length)
            inferred = draft_next_token(
                model, prefix, branch_index, base_position
            )
            expected = training.branch_predictions[branch_index - 1][
                row, base_position
            ]
            compared[branch_index - 1] += 1
            equal = int(inferred[0]) == int(expected)
            matches[branch_index - 1] += int(equal)
            if not equal:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "branch": branch_index,
                        "base_position": base_position,
                        "training_token": int(expected),
                        "reference_token": int(inferred[0]),
                    }
                )
    accuracy = [
        match / count if count else 0.0 for match, count in zip(matches, compared)
    ]
    return {
        "passed": all(count > 0 and match == count for match, count in zip(matches, compared)),
        "compared": compared,
        "matches": matches,
        "accuracy": accuracy,
        "failures": failures,
    }
