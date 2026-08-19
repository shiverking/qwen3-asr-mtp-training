from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch

from .config import TrainConfig
from .data import LANGUAGE_NAMES, ManifestDataset


def main() -> None:
    parser = argparse.ArgumentParser("Validate environment and a Qwen3-ASR MTP dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio-checks", type=int, default=100)
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    dataset = ManifestDataset(config.resolve_manifest(config.train_manifest), config.dataset_root)
    language_counts = Counter(row["language"] for row in dataset.rows)
    unknown = sorted(set(language_counts) - set(LANGUAGE_NAMES))
    if unknown:
        raise ValueError(f"No Qwen language-name mapping for: {unknown}")
    for row in dataset.rows[: args.audio_checks]:
        path = Path(config.dataset_root) / row["audio"]
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.format != "FLAC":
            raise ValueError(f"Unexpected audio format: {path}: {info}")
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "manifest_rows": len(dataset),
        "languages": language_counts,
        "checked_audio_files": min(args.audio_checks, len(dataset)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=list))


if __name__ == "__main__":
    main()
