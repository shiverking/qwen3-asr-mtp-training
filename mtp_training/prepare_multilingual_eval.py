from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
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


def _fingerprints(row: dict) -> list[tuple[str, str]]:
    result = [("id", str(row["id"])), ("pcm", str(row["sha256_pcm"]))]
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
    database = output_dir / "split_fingerprints.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE IF EXISTS fingerprints")
    connection.execute("CREATE TABLE fingerprints(kind TEXT, value TEXT, split TEXT, sample_id TEXT, PRIMARY KEY(kind, value))")
    with train_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            connection.executemany(
                "INSERT OR IGNORE INTO fingerprints VALUES (?, ?, 'train', ?)",
                [(kind, value, row["id"]) for kind, value in _fingerprints(row)],
            )
    connection.commit()
    reports = {}
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
                    overlap = connection.execute(
                        "SELECT split, sample_id FROM fingerprints WHERE kind=? AND value=?",
                        (kind, value),
                    ).fetchone()
                    if overlap and (overlap[0] != split or kind in ("id", "pcm")):
                        raise RuntimeError(
                            f"{split} sample {row['id']} overlaps {overlap[0]} "
                            f"sample {overlap[1]} by {kind}"
                        )
                source_audio = source_root / Path(row["audio"])
                suffix = source_audio.suffix.lower()
                relative = Path("audio_eval") / split / row["language"] / row["sha256_pcm"][:2] / f"{row['sha256_pcm']}{suffix}"
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
                rows.append(selected)
                counts[row["language"]] += 1
                connection.executemany(
                    "INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?, ?)",
                    [(kind, value, split, row["id"]) for kind, value in _fingerprints(row)],
                )
        missing = {language: counts[language] for language in LANGUAGES if counts[language] < config.minimum_per_language}
        if missing:
            raise RuntimeError(f"{split} is below minimum counts: {missing}")
        output_path = target_root / output_name
        temporary = output_path.with_suffix(output_path.suffix + ".building")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output_path)
        connection.commit()
        reports[split] = dict(sorted(counts.items()))
    connection.close()
    report_path = output_dir / "eval_data_summary.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
