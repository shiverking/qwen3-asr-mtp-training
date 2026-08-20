from __future__ import annotations

from typing import Any

import torch


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
    """Perturb the final supervised token and verify earlier predictions are stable."""
    original = model(stage=stage, **batch)
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
    mismatches = {"main": 0, "branches": [0] * model.depth}
    compared = {"main": 0, "branches": [0] * model.depth}
    for row, changed_position in enumerate(changed_positions):
        if changed_position is None:
            continue
        main_width = min(changed_position, original.main_predictions.shape[1])
        compared["main"] += main_width
        mismatches["main"] += int(
            original.main_predictions[row, :main_width]
            .ne(perturbed.main_predictions[row, :main_width])
            .sum()
        )
        for branch_index, (before, after) in enumerate(
            zip(original.branch_predictions, perturbed.branch_predictions), start=1
        ):
            width = min(max(changed_position - branch_index, 0), before.shape[1])
            compared["branches"][branch_index - 1] += width
            mismatches["branches"][branch_index - 1] += int(
                before[row, :width].ne(after[row, :width]).sum()
            )
    passed = mismatches["main"] == 0 and not any(mismatches["branches"])
    return {"passed": passed, "compared": compared, "mismatches": mismatches}
