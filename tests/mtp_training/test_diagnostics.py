from __future__ import annotations

from types import SimpleNamespace

import torch

from mtp_training.diagnostics import audit_initialization, audit_trainable_parameters


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
