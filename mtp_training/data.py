from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import soundfile as sf
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
            bucket.sort(key=lambda i: float(self.dataset.rows[i]["duration_s"]))
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
        for index, row in enumerate(dataset.rows):
            grouped[row["language"]].append(index)
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
        shuffled.sort(key=lambda index: float(self.dataset.rows[index]["duration_s"]))
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
