from types import SimpleNamespace

from mtp_training.train import should_stop_for_plateau


def _config():
    return SimpleNamespace(
        early_stop_after_step=2500,
        early_stop_patience_evals=3,
        early_stop_min_delta=0.03,
    )


def test_plateau_gate_requires_three_small_improvements_after_start_step():
    history = [2.00, 2.02, 2.04, 2.05]
    assert not should_stop_for_plateau(history, 2250, _config())
    assert should_stop_for_plateau(history, 2500, _config())


def test_plateau_gate_resets_when_recent_improvement_is_large():
    assert not should_stop_for_plateau([2.00, 2.02, 2.10, 2.11], 2500, _config())
