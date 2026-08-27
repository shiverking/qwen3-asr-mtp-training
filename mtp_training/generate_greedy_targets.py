from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from qwen_asr import Qwen3ASRModel
from tqdm.auto import tqdm

from .data import LANGUAGE_NAMES


@dataclass
class GreedyTargetConfig:
    model_path: str
    dataset_root: str
    input_manifest: str
    output_dir: str
    output_name: str = "train_mtp_greedy.jsonl"
    model_revision: str = "local"
    batch_size: int = 32
    max_new_tokens: int = 512
    attn_implementation: str = "flash_attention_2"
    log_every_batches: int = 200

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GreedyTargetConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown greedy target config keys: {sorted(unknown)}")
        return cls(**values)

    def resolve_input_manifest(self) -> Path:
        path = Path(self.input_manifest)
        return path if path.is_absolute() else Path(self.dataset_root) / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _configure_padding(model, tokenizer) -> None:
    eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_token_id
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "thinker", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        generation_config = getattr(candidate, "generation_config", None)
        if generation_config is not None:
            generation_config.pad_token_id = eos_token_id
        model_config = getattr(candidate, "config", None)
        if model_config is not None:
            model_config.pad_token_id = eos_token_id


def main() -> None:
    parser = argparse.ArgumentParser("Generate resumable streaming greedy targets")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = GreedyTargetConfig.from_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    input_path = config.resolve_input_manifest()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / config.output_name
    partial_path = output_dir / f"{config.output_name}.partial"
    state_path = output_dir / f"{config.output_name}.state.json"
    rejected_path = output_dir / f"{config.output_name}.rejected.jsonl"
    if final_path.is_file():
        print(f"already complete output={final_path}", flush=True)
        return
    stat = input_path.stat()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    if state is None:
        state = {
            "input_manifest": str(input_path.resolve()),
            "input_size": stat.st_size,
            "input_mtime_ns": stat.st_mtime_ns,
            "input_sha256": _sha256(input_path),
            "input_offset": 0,
            "output_bytes": 0,
            "rejected_bytes": 0,
            "processed": 0,
            "written": 0,
            "rejected": 0,
            "audio_seconds": 0.0,
            "batches": 0,
            "elapsed_seconds": 0.0,
        }
        _atomic_json(state_path, state)
    if (state["input_size"], state["input_mtime_ns"], state["input_manifest"]) != (
        stat.st_size, stat.st_mtime_ns, str(input_path.resolve())
    ):
        raise RuntimeError("Input manifest changed since greedy generation started")
    partial_path.touch(exist_ok=True)
    with partial_path.open("r+b") as stream:
        stream.truncate(state["output_bytes"])
    rejected_path.touch(exist_ok=True)
    with rejected_path.open("r+b") as stream:
        stream.truncate(state.get("rejected_bytes", 0))
    model = Qwen3ASRModel.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=config.attn_implementation,
        max_inference_batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
    )
    tokenizer = model.processor.tokenizer
    _configure_padding(model, tokenizer)
    started = time.perf_counter()
    progress = tqdm(
        total=stat.st_size,
        initial=state["input_offset"],
        desc=f"greedy {config.output_name}",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    )
    try:
        with input_path.open("rb") as source, partial_path.open("ab") as output, rejected_path.open("ab") as rejected:
            source.seek(state["input_offset"])
            while True:
                previous_offset = source.tell()
                rows = []
                for _ in range(config.batch_size):
                    line = source.readline()
                    if not line:
                        break
                    if line.strip():
                        rows.append(json.loads(line))
                if not rows:
                    break
                results = model.transcribe(
                    audio=[str(Path(config.dataset_root) / Path(row["audio"])) for row in rows],
                    language=[LANGUAGE_NAMES[row["language"]] for row in rows],
                )
                for row, result in zip(rows, results):
                    hypothesis = result.text.strip()
                    if not hypothesis:
                        rejected.write((json.dumps({"id": row["id"], "reason": "empty_greedy_output"}, ensure_ascii=False) + "\n").encode("utf-8"))
                        state["rejected"] += 1
                        continue
                    generated = dict(row)
                    generated.update(
                        text_reference=row["text"], text_mtp_target=hypothesis,
                        mtp_target_token_count=len(tokenizer.encode(hypothesis, add_special_tokens=False)),
                        mtp_target_source="qwen3_asr_greedy", mtp_target_model=config.model_path,
                        mtp_target_revision=config.model_revision,
                    )
                    output.write((json.dumps(generated, ensure_ascii=False) + "\n").encode("utf-8"))
                    state["written"] += 1
                output.flush()
                rejected.flush()
                state["processed"] += len(rows)
                state["audio_seconds"] += sum(float(row["duration_s"]) for row in rows)
                state["batches"] += 1
                state["input_offset"] = source.tell()
                state["output_bytes"] = output.tell()
                state["rejected_bytes"] = rejected.tell()
                state["elapsed_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                _atomic_json(state_path, state)
                progress.update(source.tell() - previous_offset)
                if state["batches"] % max(config.log_every_batches // 20, 1) == 0:
                    rate = state["processed"] / max(state["elapsed_seconds"], 1e-6)
                    audio_rate = state["audio_seconds"] / max(state["elapsed_seconds"], 1e-6)
                    progress.set_postfix(
                        samples=f"{state['processed']:,}",
                        speed=f"{rate:.2f}/s",
                        audio=f"{audio_rate:.2f}h/h",
                        rejected=f"{state['rejected']:,}",
                        refresh=True,
                    )
    finally:
        progress.close()
    os.replace(partial_path, final_path)
    state.update(complete=True, output_manifest=str(final_path))
    _atomic_json(state_path, state)
    print(f"complete output={final_path} samples={state['written']:,} rejected={state['rejected']:,}", flush=True)


if __name__ == "__main__":
    main()
