from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


LANGUAGES = ("en", "es", "pt-BR", "pt-PT")


@dataclass
class EvalDataConfig:
    dataset_root: str
    source_dataset_root: str
    train_manifest: str = "manifests/train.jsonl"
    source_dev_manifest: str = "manifests/dev.jsonl"
    source_test_manifest: str = "manifests/test.jsonl"
    output_dev_manifest: str = "manifests/eval/dev.jsonl"
    output_test_manifest: str = "manifests/eval/test.jsonl"
    minimum_per_language: int = 250

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalDataConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown eval data config keys: {sorted(unknown)}")
        return cls(**values)


def _sample_id(row: dict) -> str:
    sample_id = row.get("id")
    if sample_id:
        return str(sample_id)
    source = row.get("source")
    source_id = row.get("source_id")
    if source and source_id:
        return f"{source}:{source_id}"
    audio = row.get("audio")
    if audio:
        return f"audio:{audio}"
    raise ValueError(
        "Manifest row has no usable id, source/source_id, or audio path: "
        f"keys={sorted(row)}"
    )


def _audio_fingerprint(row: dict) -> tuple[str, str] | None:
    pcm_hash = row.get("sha256_pcm")
    if pcm_hash:
        return "pcm", str(pcm_hash)
    file_hash = row.get("sha256_file")
    if file_hash:
        return "file", str(file_hash)
    return None


def _fingerprints(row: dict) -> list[tuple[str, str]]:
    result = [("id", _sample_id(row))]
    audio_fingerprint = _audio_fingerprint(row)
    if audio_fingerprint is not None:
        result.append(audio_fingerprint)
    source = row.get("source")
    source_id = row.get("source_id")
    if source and source_id:
        result.append(("source_id", f"{source}:{source_id}"))
    elif source and row.get("audio"):
        result.append(("source_audio", f"{source}:{row['audio']}"))
    for name in ("speaker_id", "origin_recording_id"):
        value = row.get(name)
        if value:
            result.append((name, str(value)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Build four-language dev/test data from the frozen 200h corpus")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = EvalDataConfig.from_yaml(args.config)
    target_root = Path(config.dataset_root)
    source_root = Path(config.source_dataset_root)
    active_temps = list((target_root / "manifests").rglob("*.tmp"))
    if active_temps:
        raise RuntimeError(f"Dataset contains unfinished .tmp files: {active_temps[:5]}")
    train_manifest = target_root / config.train_manifest
    output_dir = (target_root / config.output_dev_manifest).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    candidates = {}
    eval_fingerprints: dict[tuple[str, str], tuple[str, str]] = {}
    for split, source_name, output_name in (
        ("dev", config.source_dev_manifest, config.output_dev_manifest),
        ("test", config.source_test_manifest, config.output_test_manifest),
    ):
        rows = []
        counts: Counter[str] = Counter()
        with (source_root / source_name).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("language") not in LANGUAGES:
                    continue
                for kind, value in _fingerprints(row):
                    overlap = eval_fingerprints.get((kind, value))
                    if overlap and (
                        overlap[0] != split
                        or kind in (
                            "id",
                            "pcm",
                            "file",
                            "source_id",
                            "source_audio",
                        )
                    ):
                        raise RuntimeError(
                            f"{split} sample {_sample_id(row)} overlaps {overlap[0]} "
                            f"sample {overlap[1]} by {kind}"
                        )
                    eval_fingerprints.setdefault(
                        (kind, value), (split, _sample_id(row))
                    )
                rows.append(row)
                counts[row["language"]] += 1
        missing = {
            language: counts[language]
            for language in LANGUAGES
            if counts[language] < config.minimum_per_language
        }
        if missing:
            raise RuntimeError(f"{split} is below minimum counts: {missing}")
        candidates[split] = (rows, output_name)
        reports[split] = dict(sorted(counts.items()))

    scanned = 0
    missing_audio_hashes = 0
    with train_manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            scanned += 1
            missing_audio_hashes += int(_audio_fingerprint(row) is None)
            for fingerprint in _fingerprints(row):
                overlap = eval_fingerprints.get(fingerprint)
                if overlap:
                    raise RuntimeError(
                        f"train sample {_sample_id(row)} at line {line_number} "
                        f"overlaps {overlap[0]} sample {overlap[1]} by "
                        f"{fingerprint[0]}"
                    )
            if scanned % 100_000 == 0:
                print(f"leakage scan train rows={scanned:,}", flush=True)
    print(
        f"leakage scan complete train rows={scanned:,} "
        f"rows_without_audio_hash={missing_audio_hashes:,}",
        flush=True,
    )

    for split, (rows, output_name) in candidates.items():
        selected_rows = []
        for row in rows:
            source_audio = source_root / Path(row["audio"])
            suffix = source_audio.suffix.lower()
            fingerprint = _audio_fingerprint(row)
            audio_fingerprint = (
                fingerprint[1]
                if fingerprint is not None
                else hashlib.sha256(_sample_id(row).encode("utf-8")).hexdigest()
            )
            relative = (
                Path("audio_eval")
                / split
                / row["language"]
                / audio_fingerprint[:2]
                / f"{audio_fingerprint}{suffix}"
            )
            destination = target_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                try:
                    os.link(source_audio, destination)
                except OSError:
                    shutil.copy2(source_audio, destination)
            selected = dict(row)
            selected["audio"] = relative.as_posix()
            selected["source_split"] = split
            selected_rows.append(selected)
        output_path = target_root / output_name
        temporary = output_path.with_suffix(output_path.suffix + ".building")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in selected_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output_path)
    report_path = output_dir / "eval_data_summary.json"
    report_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
