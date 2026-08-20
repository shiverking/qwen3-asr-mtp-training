from __future__ import annotations

from mtp_training.evaluation import METRIC_VERSION, _finalize, _macro_average


def _stats(scale: int = 1):
    return {
        "main_loss_sum": 4.0 * scale,
        "main_correct": 2 * scale,
        "main_valid": 4 * scale,
        "branch_loss_sum": [4.0 * scale, 8.0 * scale],
        "training_correct": [2 * scale, 1 * scale],
        "training_valid": [4 * scale, 2 * scale],
        "legacy_backbone_correct": [2 * scale, 1 * scale],
        "legacy_backbone_valid": [4 * scale, 2 * scale],
        "decode_gt_correct": [2 * scale, 1 * scale],
        "decode_gt_valid": [3 * scale, 1 * scale],
        "decode_bb_correct": [1 * scale, 1 * scale],
        "decode_bb_valid": [3 * scale, 1 * scale],
        "legacy_gt_accepted": 6.0 * scale,
        "legacy_gt_positions": 4 * scale,
        "legacy_bb_accepted": 5.0 * scale,
        "legacy_bb_positions": 4 * scale,
        "decode_gt_accepted": 3.0 * scale,
        "decode_gt_positions": 1 * scale,
        "decode_bb_accepted": 2.0 * scale,
        "decode_bb_positions": 1 * scale,
    }


def test_metric_v2_keeps_legacy_aliases_and_adds_decode_window_metrics():
    result = _finalize(_stats(), [0.5, 0.5], stage=1)
    assert result["metric_version"] == METRIC_VERSION
    assert result["branch_accuracy"] == result["training_target_branch_accuracy"]
    assert result["ground_truth_branch_accuracy"] == result["branch_accuracy"]
    assert (
        result["average_accepted_length"]
        == result["legacy_ground_truth_average_accepted_length"]
    )
    assert result["decode_window_ground_truth_average_accepted_length"] == 3.0
    assert result["decode_window_backbone_consistency_average_accepted_length"] == 2.0


def test_macro_average_weights_languages_equally():
    first = _finalize(_stats(), [0.5, 0.5], stage=1)
    second_stats = _stats()
    second_stats["main_correct"] = 0
    second = _finalize(second_stats, [0.5, 0.5], stage=1)
    macro = _macro_average([first, second])
    assert macro["languages"] == 2
    assert macro["main_accuracy"] == 0.25
