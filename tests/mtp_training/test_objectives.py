import torch

from mtp_training.objectives import (
    BranchResult,
    align_branch_with_backbone,
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


def test_branch_prediction_is_aligned_with_shifted_backbone_position():
    main = BranchResult(
        loss=torch.tensor(0.0),
        correct=torch.ones((1, 5), dtype=torch.bool),
        valid=torch.ones((1, 5), dtype=torch.bool),
        predicted=torch.tensor([[10, 11, 12, 13, 14]]),
        token_losses=torch.zeros((1, 5)),
    )
    branch = BranchResult(
        loss=torch.tensor(0.0),
        correct=torch.ones((1, 4), dtype=torch.bool),
        valid=torch.ones((1, 4), dtype=torch.bool),
        predicted=torch.tensor([[11, 99, 13, 14]]),
        token_losses=torch.zeros((1, 4)),
    )
    correct, valid = align_branch_with_backbone(branch, main, branch_index=1)
    assert correct.tolist() == [[True, False, True, True]]
    assert valid.all()


def test_branch_cross_entropy_returns_predictions_and_per_token_losses():
    from mtp_training.objectives import branch_cross_entropy

    hidden = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    head = torch.nn.Identity()
    targets = torch.tensor([[0, 1]])
    valid = torch.tensor([[True, False]])
    result = branch_cross_entropy(hidden, head, targets, valid)
    assert result.predicted.tolist() == [[0, -1]]
    assert result.correct.tolist() == [[True, False]]
    assert result.token_losses[0, 0] > 0
    assert result.token_losses[0, 1] == 0
