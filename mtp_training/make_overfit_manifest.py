from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("Build a deterministic language-stratified overfit manifest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-language", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)
    rng = random.Random(args.seed)
    selected = []
    for language in sorted(grouped):
        if len(grouped[language]) < args.per_language:
            raise ValueError(
                f"{language} has {len(grouped[language])} rows, needs {args.per_language}"
            )
        selected.extend(rng.sample(grouped[language], args.per_language))
    rng.shuffle(selected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "samples": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
