import pytest
from mtp_training.evaluation import _empty_stats, _finalize


def test_strict_position_rates_reconstruct_average_length():
    stats = _empty_stats(5)
    stats["strict_position_accepted"] = [90, 70, 50, 30, 10]
    stats["strict_position_eligible"] = [100] * 5
    metrics = _finalize(stats, [0.2] * 5, stage=1)
    assert metrics["strict_position_acceptance"] == pytest.approx(
        [0.9, 0.7, 0.5, 0.3, 0.1]
    )
    assert metrics["strict_average_accepted_length"] == pytest.approx(
        1 + sum(metrics["strict_position_acceptance"])
    )
