from mtp_training.evaluate_backbone_asr import _distance, _units


def test_edit_distance_and_language_units():
    assert _distance(["a", "b"], ["a", "c"]) == 1
    assert _units("你 好", "zh-CN") == ["你", "好"]
    assert _units("Hello, WORLD!", "en") == ["hello", "world"]
