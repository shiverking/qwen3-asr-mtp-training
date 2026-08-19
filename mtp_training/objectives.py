from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def normalized_branch_weights(depth: int, alpha: float, device=None) -> torch.Tensor:
    raw = torch.tensor([alpha**i for i in range(depth)], dtype=torch.float32, device=device)
    return raw / raw.sum()


def shifted_targets(
    input_ids: torch.Tensor, loss_mask: torch.Tensor, future_offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return x[t+future_offset] and its transcript-only validity mask."""
    if future_offset < 1:
        raise ValueError("future_offset must be >= 1")
    return input_ids[:, future_offset:], loss_mask[:, future_offset:]


@dataclass
class BranchResult:
    loss: torch.Tensor
    correct: torch.Tensor
    valid: torch.Tensor


def branch_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> BranchResult:
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_valid = valid.reshape(-1)
    if not torch.any(flat_valid):
        zero = hidden_states.sum() * 0.0
        return BranchResult(zero, torch.zeros_like(valid), valid)
    logits = lm_head(flat_hidden[flat_valid])
    selected_targets = flat_targets[flat_valid]
    loss = F.cross_entropy(logits.float(), selected_targets, reduction="mean")
    selected_correct = logits.detach().argmax(dim=-1).eq(selected_targets)
    correct = torch.zeros_like(flat_valid, dtype=torch.bool)
    correct[flat_valid] = selected_correct
    return BranchResult(loss, correct.view_as(valid), valid)


def strict_acceptance(
    branch_correct: list[torch.Tensor], branch_valid: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return accepted-token sum and eligible AR-position count.

    The normal AR token counts as one. Auxiliary tokens count only while every
    earlier auxiliary branch at that position is correct.
    """
    if not branch_correct:
        raise ValueError("At least one MTP branch is required")
    width = min(item.shape[1] for item in branch_correct)
    eligible = branch_valid[-1][:, :width].clone()
    prefix_ok = torch.ones_like(eligible)
    accepted = eligible.to(torch.float32)
    for correct, valid in zip(branch_correct, branch_valid):
        prefix_ok &= correct[:, :width] & valid[:, :width]
        accepted += prefix_ok.to(torch.float32)
    return accepted.sum(), eligible.sum()
