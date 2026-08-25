import json

import pytest

from mtp_training.data import (
    IndexedManifestDataset,
    MixedLanguageSourceTemperatureBatchSampler,
)


def _write_manifest(path, audio_path):
    rows = []
    for index, (language, source) in enumerate(
        (("en", "a"), ("en", "b"), ("es", "a"), ("pt-BR", "c"), ("pt-PT", "d"))
    ):
        rows.append(
            {
                "id": str(index),
                "audio": audio_path.name,
                "duration_s": index + 1,
                "language": language,
                "source": source,
                "mtp_target_token_count": index + 2,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_index_random_access_hash_guard_and_mixed_sampler(tmp_path):
    audio = tmp_path / "audio.flac"
    audio.touch()
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, audio)
    metadata = IndexedManifestDataset.build_index(manifest)
    dataset = IndexedManifestDataset(manifest, tmp_path)
    assert metadata["row_count"] == 5
    assert dataset[3]["id"] == "3"
    sampler_a = MixedLanguageSourceTemperatureBatchSampler(dataset, 2, 7)
    sampler_b = MixedLanguageSourceTemperatureBatchSampler(dataset, 2, 7)
    assert list(sampler_a) == list(sampler_b)
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        IndexedManifestDataset(manifest, tmp_path)
