from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _diverse_sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    candidates = list(rows)
    rng.shuffle(candidates)
    selected = []
    speakers = set()
    source_counts: Counter[str] = Counter()
    while candidates and len(selected) < count:
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                candidates[index].get("speaker_id") not in speakers,
                -source_counts[candidates[index].get("source", "unknown")],
                -index,
            ),
        )
        row = candidates.pop(best_index)
        selected.append(row)
        speakers.add(row.get("speaker_id"))
        source_counts[row.get("source", "unknown")] += 1
    if len(selected) != count:
        raise ValueError(f"Only selected {len(selected)} rows, needs {count}")
    return selected


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser("Build deterministic round-2 MTP manifests")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)
    required = {"zh-CN", "en", "ar", "th", "es", "pt-BR", "pt-PT"}
    missing = sorted(required - set(grouped))
    if missing:
        raise ValueError(f"Missing languages: {missing}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    outputs = {
        "diagnostic-zh64.jsonl": _diverse_sample(grouped["zh-CN"], 64, rng),
        "diagnostic-en64.jsonl": _diverse_sample(grouped["en"], 64, rng),
    }
    position_rows = []
    for language in sorted(required):
        position_rows.extend(_diverse_sample(grouped[language], 32, rng))
    rng.shuffle(position_rows)
    outputs["diagnostic-position-ab224.jsonl"] = position_rows
    for name, selected in outputs.items():
        _write(output_dir / name, selected)
    print(
        json.dumps(
            {name: len(selected) for name, selected in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
