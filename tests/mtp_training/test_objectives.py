import torch

from mtp_training.objectives import (
    normalized_branch_weights,
    shifted_targets,
    strict_acceptance,
)


def test_branch_weights_are_normalized_and_decay():
    weights = normalized_branch_weights(3, 0.9)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    assert weights[0] > weights[1] > weights[2]


def test_shifted_targets_align_future_tokens():
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    mask = torch.tensor([[False, True, True, True, False]])
    targets, valid = shifted_targets(input_ids, mask, 2)
    assert targets.tolist() == [[12, 13, 14]]
    assert valid.tolist() == [[True, True, False]]


def test_strict_acceptance_stops_after_first_rejection():
    valid = [torch.ones((1, 3), dtype=torch.bool) for _ in range(3)]
    correct = [
        torch.tensor([[True, True, False]]),
        torch.tensor([[True, False, True]]),
        torch.tensor([[False, True, True]]),
    ]
    accepted, positions = strict_acceptance(correct, valid)
    # Per position: 3, 2, 1 accepted tokens including the normal AR token.
    assert accepted.item() == 6
    assert positions.item() == 3
