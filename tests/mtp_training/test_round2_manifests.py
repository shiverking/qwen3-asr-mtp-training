from __future__ import annotations

import random

from mtp_training.make_round2_manifests import _diverse_sample


def test_diverse_sample_is_deterministic_and_prefers_new_speakers():
    rows = [
        {
            "id": f"a-{index}",
            "source": "a",
            "speaker_id": f"speaker-{index}",
        }
        for index in range(5)
    ] + [
        {
            "id": f"b-{index}",
            "source": "b",
            "speaker_id": f"speaker-b-{index}",
        }
        for index in range(5)
    ]
    first = _diverse_sample(rows, 6, random.Random(7))
    second = _diverse_sample(rows, 6, random.Random(7))
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["speaker_id"] for row in first}) == 6
    assert {row["source"] for row in first} == {"a", "b"}
