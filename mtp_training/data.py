from __future__ import annotations

import json
import hashlib
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


LANGUAGE_NAMES = {
    "zh-CN": "Chinese",
    "en": "English",
    "ar": "Arabic",
    "th": "Thai",
    "es": "Spanish",
    "pt-BR": "Portuguese",
    "pt-PT": "Portuguese",
}


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, dataset_root: str | Path):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        audio_path = self.dataset_root / Path(row["audio"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        row["audio_path"] = str(audio_path)
        return row


class IndexedManifestDataset(Dataset):
    """Random-access JSONL dataset backed by compact numpy sidecars."""

    INDEX_VERSION = 1

    def __init__(self, manifest_path: str | Path, dataset_root: str | Path):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.index_dir = Path(str(self.manifest_path) + ".mtpidx")
        metadata_path = self.index_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing manifest index {metadata_path}; run build_manifest_index first"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stat = self.manifest_path.stat()
        if (
            stat.st_size != self.metadata["manifest_size"]
            or stat.st_mtime_ns != self.metadata["manifest_mtime_ns"]
        ):
            raise RuntimeError("Manifest changed after index creation; rebuild the index")
        self.offsets = np.load(self.index_dir / "offsets.npy", mmap_mode="r")
        self.durations = np.load(self.index_dir / "durations.npy", mmap_mode="r")
        self.language_codes = np.load(self.index_dir / "languages.npy", mmap_mode="r")
        self.source_codes = np.load(self.index_dir / "sources.npy", mmap_mode="r")
        token_path = self.index_dir / "target_tokens.npy"
        self.target_tokens = (
            np.load(token_path, mmap_mode="r")
            if token_path.is_file()
            else np.ones(len(self.offsets), dtype=np.int32)
        )
        self.languages = self.metadata["language_vocabulary"]
        self.sources = self.metadata["source_vocabulary"]
        self._stream = None

    @property
    def manifest_sha256(self) -> str:
        return self.metadata["manifest_sha256"]

    @classmethod
    def build_index(cls, manifest_path: str | Path) -> dict[str, Any]:
        manifest_path = Path(manifest_path)
        index_dir = Path(str(manifest_path) + ".mtpidx")
        index_dir.mkdir(parents=True, exist_ok=True)
        offsets: list[int] = []
        durations: list[float] = []
        language_codes: list[int] = []
        source_codes: list[int] = []
        target_tokens: list[int] = []
        language_vocabulary: list[str] = []
        source_vocabulary: list[str] = []
        language_lookup: dict[str, int] = {}
        source_lookup: dict[str, int] = {}
        language_counts: Counter[str] = Counter()
        language_seconds: Counter[str] = Counter()
        digest = hashlib.sha256()
        with manifest_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                target = row.get("text_mtp_target")
                if not isinstance(target, str) or not target.strip():
                    raise ValueError(
                        f"Missing text_mtp_target at manifest byte offset {offset}"
                    )
                language = row["language"]
                source = row.get("source", "unknown")
                if language not in language_lookup:
                    language_lookup[language] = len(language_vocabulary)
                    language_vocabulary.append(language)
                if source not in source_lookup:
                    source_lookup[source] = len(source_vocabulary)
                    source_vocabulary.append(source)
                duration = float(row["duration_s"])
                offsets.append(offset)
                durations.append(duration)
                language_codes.append(language_lookup[language])
                source_codes.append(source_lookup[source])
                target_tokens.append(max(int(row.get("mtp_target_token_count", 1)), 1))
                language_counts[language] += 1
                language_seconds[language] += duration
        stat = manifest_path.stat()
        arrays = {
            "offsets": np.asarray(offsets, dtype=np.int64),
            "durations": np.asarray(durations, dtype=np.float32),
            "languages": np.asarray(language_codes, dtype=np.uint16),
            "sources": np.asarray(source_codes, dtype=np.uint16),
            "target_tokens": np.asarray(target_tokens, dtype=np.int32),
        }
        for name, values in arrays.items():
            temporary = index_dir / f"{name}.npy.tmp"
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            os.replace(temporary, index_dir / f"{name}.npy")
        metadata = {
            "index_version": cls.INDEX_VERSION,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": digest.hexdigest(),
            "manifest_size": stat.st_size,
            "manifest_mtime_ns": stat.st_mtime_ns,
            "row_count": len(offsets),
            "hours": sum(durations) / 3600,
            "language_vocabulary": language_vocabulary,
            "source_vocabulary": source_vocabulary,
            "samples_by_language": dict(sorted(language_counts.items())),
            "hours_by_language": {
                key: value / 3600 for key, value in sorted(language_seconds.items())
            },
        }
        temporary_metadata = index_dir / "metadata.json.tmp"
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_metadata, index_dir / "metadata.json")
        return metadata

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_stream"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._stream is None:
            self._stream = self.manifest_path.open("rb")
        self._stream.seek(int(self.offsets[index]))
        row = json.loads(self._stream.readline())
        audio_path = self.dataset_root / Path(row["audio"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        row["audio_path"] = str(audio_path)
        return row

    def duration(self, index: int) -> float:
        return float(self.durations[index])

    def language(self, index: int) -> str:
        return self.languages[int(self.language_codes[index])]

    def source(self, index: int) -> str:
        return self.sources[int(self.source_codes[index])]


class DurationBucketBatchSampler(Sampler[list[int]]):
    """Shuffle locally while keeping similarly sized audio in the same batch."""

    def __init__(
        self,
        dataset: ManifestDataset,
        batch_size: int,
        seed: int,
        bucket_multiplier: int = 50,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_size = batch_size * bucket_multiplier
        self.drop_last = drop_last
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_batch(self, start_batch: int) -> None:
        if start_batch < 0:
            raise ValueError("start_batch must be >= 0")
        self.start_batch = start_batch

    def __len__(self) -> int:
        size = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            size += 1
        return size

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda i: _duration(self.dataset, i))
            for offset in range(0, len(bucket), self.batch_size):
                batch = bucket[offset : offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        start_batch = self.start_batch
        self.start_batch = 0
        yield from batches[start_batch:]


class LanguageTemperatureBatchSampler(Sampler[list[int]]):
    """Sample homogeneous duration-bucketed batches with tempered language weights."""

    def __init__(
        self,
        dataset: ManifestDataset,
        batch_size: int,
        seed: int,
        temperature: float = 0.5,
        drop_last: bool = False,
    ):
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0 and 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.temperature = temperature
        self.drop_last = drop_last
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            grouped[_language(dataset, index)].append(index)
        self.grouped_indices = dict(grouped)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def _language_batches(self, indices: list[int], rng: random.Random) -> list[list[int]]:
        shuffled = list(indices)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda index: _duration(self.dataset, index))
        batches = [
            shuffled[start : start + self.batch_size]
            for start in range(0, len(shuffled), self.batch_size)
        ]
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        languages = sorted(self.grouped_indices)
        weights = [len(self.grouped_indices[key]) ** self.temperature for key in languages]
        batches_by_language = {
            key: self._language_batches(self.grouped_indices[key], rng) for key in languages
        }
        offsets = {key: 0 for key in languages}
        for _ in range(len(self)):
            language = rng.choices(languages, weights=weights, k=1)[0]
            batches = batches_by_language[language]
            if not batches:
                continue
            offset = offsets[language]
            if offset >= len(batches):
                batches = self._language_batches(self.grouped_indices[language], rng)
                batches_by_language[language] = batches
                offset = 0
            batch = batches[offset]
            offsets[language] = offset + 1
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch


def _duration(dataset, index: int) -> float:
    if hasattr(dataset, "duration"):
        return dataset.duration(index)
    return float(dataset.rows[index]["duration_s"])


def _language(dataset, index: int) -> str:
    if hasattr(dataset, "language"):
        return dataset.language(index)
    return dataset.rows[index]["language"]


def _source(dataset, index: int) -> str:
    if hasattr(dataset, "source"):
        return dataset.source(index)
    return dataset.rows[index].get("source", "unknown")


class MixedLanguageSourceTemperatureBatchSampler(Sampler[list[int]]):
    """Mixed batches sampled hierarchically by language and source token mass."""

    def __init__(
        self,
        dataset: IndexedManifestDataset,
        batch_size: int,
        seed: int,
        language_temperature: float = 0.5,
        source_temperature: float = 0.7,
        bucket_multiplier: int = 50,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.language_temperature = language_temperature
        self.source_temperature = source_temperature
        self.bucket_size = batch_size * bucket_multiplier
        self.drop_last = drop_last
        self.epoch = 0
        self.start_batch = 0
        grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index in range(len(dataset)):
            grouped[dataset.language(index)][dataset.source(index)].append(index)
        self.grouped = {lang: dict(sources) for lang, sources in grouped.items()}
        self.source_mass = {
            (language, source): sum(int(dataset.target_tokens[index]) for index in indices)
            for language, sources in self.grouped.items()
            for source, indices in sources.items()
        }
        self.language_mass = {
            language: sum(self.source_mass[(language, source)] for source in sources)
            for language, sources in self.grouped.items()
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_batch(self, start_batch: int) -> None:
        self.start_batch = start_batch

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size if self.drop_last else math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        pools = {
            (language, source): list(indices)
            for language, sources in self.grouped.items()
            for source, indices in sources.items()
        }
        for values in pools.values():
            rng.shuffle(values)
        offsets = {key: 0 for key in pools}
        languages = sorted(self.grouped)
        language_weights = [self.language_mass[key] ** self.language_temperature for key in languages]
        sampled: list[int] = []
        for _ in range(len(self.dataset)):
            language = rng.choices(languages, weights=language_weights, k=1)[0]
            sources = sorted(self.grouped[language])
            source_weights = [
                self.source_mass[(language, source)] ** self.source_temperature
                for source in sources
            ]
            source = rng.choices(sources, weights=source_weights, k=1)[0]
            key = (language, source)
            offset = offsets[key]
            if offset >= len(pools[key]):
                rng.shuffle(pools[key])
                offset = 0
            sampled.append(pools[key][offset])
            offsets[key] = offset + 1
        batches: list[list[int]] = []
        for start in range(0, len(sampled), self.bucket_size):
            bucket = sampled[start : start + self.bucket_size]
            bucket.sort(key=lambda index: _duration(self.dataset, index))
            batches.extend(
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        rng.shuffle(batches)
        start_batch = self.start_batch
        self.start_batch = 0
        yield from batches[start_batch:]


@dataclass
class MTPDataCollator:
    processor: Any
    sampling_rate: int = 16000
    include_eos_in_loss: bool = False
    target_text_field: str = "text"

    def _load_audio(self, path: str):
        audio, sampling_rate = sf.read(path, dtype="float32", always_2d=False)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected {self.sampling_rate} Hz, got {sampling_rate}: {path}")
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio: {path}")
        return audio

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        audios = [self._load_audio(row["audio_path"]) for row in rows]
        language_names = []
        for row in rows:
            try:
                language_names.append(LANGUAGE_NAMES[row["language"]])
            except KeyError as error:
                raise ValueError(f"Unsupported language code: {row['language']}") from error

        messages = [
            [
                {"role": "system", "content": row.get("prompt", "")},
                {"role": "user", "content": [{"type": "audio", "audio": None}]},
            ]
            for row in rows
        ]
        generation_prefixes = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        transcript_prefixes = [
            prefix + f"language {language}<asr_text>"
            for prefix, language in zip(generation_prefixes, language_names)
        ]
        eos = self.processor.tokenizer.eos_token or ""
        targets = []
        for row in rows:
            target = row.get(self.target_text_field)
            if not isinstance(target, str) or not target.strip():
                raise ValueError(
                    f"Missing non-empty {self.target_text_field!r} for {row['id']}"
                )
            targets.append(target)
        full_texts = [
            prefix + target + eos
            for prefix, target in zip(transcript_prefixes, targets)
        ]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=transcript_prefixes,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        loss_mask = torch.zeros_like(full_inputs["input_ids"], dtype=torch.bool)
        full_lengths = full_inputs["attention_mask"].sum(dim=1).tolist()
        for row_index, (prefix_length, full_length) in enumerate(zip(prefix_lengths, full_lengths)):
            end = full_length if self.include_eos_in_loss else max(prefix_length, full_length - 1)
            loss_mask[row_index, prefix_length:end] = True

        full_inputs["loss_mask"] = loss_mask
        full_inputs["languages"] = [row["language"] for row in rows]
        full_inputs["sample_ids"] = [row["id"] for row in rows]
        return full_inputs
