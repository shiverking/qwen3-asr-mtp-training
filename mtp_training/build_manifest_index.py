from __future__ import annotations

import argparse
import json

from .config import TrainConfig
from .data import IndexedManifestDataset


def main() -> None:
    parser = argparse.ArgumentParser("Build a random-access index for a large JSONL manifest")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    manifest = config.resolve_manifest(config.train_manifest)
    metadata = IndexedManifestDataset.build_index(manifest)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
