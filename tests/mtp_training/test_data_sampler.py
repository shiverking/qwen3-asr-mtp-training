from __future__ import annotations

from types import SimpleNamespace

from mtp_training.data import LanguageTemperatureBatchSampler


def test_language_temperature_sampler_is_deterministic_and_homogeneous():
    rows = [
        {"language": "en", "duration_s": 1.0 + index / 10} for index in range(20)
    ] + [{"language": "th", "duration_s": 2.0 + index / 10} for index in range(5)]
    dataset = SimpleNamespace(rows=rows)
    sampler = LanguageTemperatureBatchSampler(
        dataset, batch_size=2, seed=7, temperature=0.5, drop_last=True
    )
    first = list(sampler)
    second = list(sampler)
    assert first == second
    assert len(first) == len(rows) // 2
    for batch in first:
        assert len({rows[index]["language"] for index in batch}) == 1


def test_language_temperature_zero_gives_equal_language_weights():
    rows = [{"language": "en", "duration_s": 1.0} for _ in range(100)] + [
        {"language": "th", "duration_s": 1.0} for _ in range(4)
    ]
    dataset = SimpleNamespace(rows=rows)
    sampler = LanguageTemperatureBatchSampler(
        dataset, batch_size=1, seed=11, temperature=0.0, drop_last=True
    )
    languages = [rows[batch[0]]["language"] for batch in sampler]
    assert 35 <= languages.count("th") <= 70
