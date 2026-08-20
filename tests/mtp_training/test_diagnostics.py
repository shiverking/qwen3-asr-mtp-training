from __future__ import annotations

from types import SimpleNamespace

import torch

from mtp_training.diagnostics import (
    audit_future_token_causality,
    audit_initialization,
    audit_reference_equivalence,
    audit_trainable_parameters,
)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Linear(2, 2, bias=False)
        self.asr_model = torch.nn.Module()
        self.asr_model.thinker = torch.nn.Module()
        self.asr_model.thinker.model = torch.nn.Module()
        self.asr_model.thinker.model.layers = torch.nn.ModuleList([layer])
        self.branches = torch.nn.ModuleList()
        for _ in range(2):
            branch = torch.nn.Module()
            branch.decoder_layer = torch.nn.Linear(2, 2, bias=False)
            branch.decoder_layer.load_state_dict(layer.state_dict())
            branch.projection = torch.nn.Linear(4, 2, bias=False)
            self.branches.append(branch)

    @property
    def thinker(self):
        return self.asr_model.thinker


def test_initialization_audit_checks_copy_without_parameter_sharing():
    report = audit_initialization(FakeModel())
    assert report["passed"]


def test_stage1_trainable_audit_rejects_backbone_parameter():
    model = FakeModel()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.branches.parameters():
        parameter.requires_grad = True
    assert audit_trainable_parameters(model, 1)["passed"]
    model.thinker.model.layers[0].weight.requires_grad = True
    assert not audit_trainable_parameters(model, 1)["passed"]


class PredictionModel:
    depth = 2

    def __init__(self, unstable: bool = False):
        self.unstable = unstable
        self.calls = 0
        self.thinker = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(vocab_size=100)))

    def __call__(self, stage, **batch):
        self.calls += 1
        input_ids = batch["input_ids"]
        main = torch.zeros_like(input_ids[:, 1:])
        branches = [
            torch.zeros_like(input_ids[:, 2:]),
            torch.zeros_like(input_ids[:, 3:]),
        ]
        if self.unstable and self.calls == 2:
            main[:, 0] += 1
        return SimpleNamespace(main_predictions=main, branch_predictions=branches)


def test_causality_audit_reports_repeatability_separately():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "loss_mask": torch.tensor([[False, True, True, True, True]]),
    }
    stable = audit_future_token_causality(PredictionModel(), 1, batch)
    assert stable["passed"]
    assert stable["baseline_mismatches"]["main"] == 0

    unstable = audit_future_token_causality(PredictionModel(unstable=True), 1, batch)
    assert not unstable["passed"]
    assert not unstable["repeatable"]


def test_reference_equivalence_checks_all_branches(monkeypatch):
    class EquivalenceModel:
        depth = 3

        def __call__(self, stage, **batch):
            return SimpleNamespace(
                main_valid=torch.tensor([[True, True, True, True]]),
                branch_valid=[
                    torch.tensor([[True, False, False]]),
                    torch.tensor([[True, False]]),
                    torch.tensor([[True]]),
                ],
                branch_predictions=[
                    torch.tensor([[11, -1, -1]]),
                    torch.tensor([[12, -1]]),
                    torch.tensor([[13]]),
                ],
            )

    model = EquivalenceModel()
    monkeypatch.setattr(
        "mtp_training.diagnostics.draft_next_token",
        lambda model, batch, depth, base_position: torch.tensor([10 + depth]),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "loss_mask": torch.tensor([[True, True, True, True, False]]),
    }
    report = audit_reference_equivalence(model, 1, batch, ["sample"], max_samples=1)
    assert report["passed"]
    assert report["accuracy"] == [1.0, 1.0, 1.0]
