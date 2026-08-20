# Qwen3-ASR ParaASR-style MTP training

This directory adds serial Multi-Token Prediction (MTP) branches to
Qwen3-ASR-1.7B without modifying the upstream model implementation. It follows
the two-stage recipe in ParaASR:

1. **Stage 1 — frozen-branch alignment:** freeze Qwen3-ASR and train only MTP
   branches at a peak learning rate of `2e-4`.
2. **Stage 2 — joint calibration:** unfreeze the text decoder, LM head and the
   audio tower's final `proj1/proj2` adapter, then jointly optimize the normal
   next-token loss and weighted MTP losses at `2e-5`.

`mtp_depth: 3` means three auxiliary branches plus the normal AR token, so one
verification step can accept at most four tokens. Change it to `5` to build an
MTP-5 model. Each auxiliary Transformer layer is initialized from the final
Qwen3-ASR decoder layer. The branch weights use normalized exponential decay
with `alpha: 0.9`.

## Dataset contract

The loader consumes the FLAC bundle directly. Manifest audio paths must be
relative to `dataset_root`; required fields are `id`, `audio`, `text`,
`language`, and `duration_s`. Supported first-wave language codes are:

| Manifest code | Qwen output prefix |
|---|---|
| `zh-CN` | `language Chinese<asr_text>` |
| `en` | `language English<asr_text>` |
| `ar` | `language Arabic<asr_text>` |
| `th` | `language Thai<asr_text>` |
| `es` | `language Spanish<asr_text>` |
| `pt-BR`, `pt-PT` | `language Portuguese<asr_text>` |

The prefix is constructed at collation time and excluded from every loss. By
default EOS is also excluded, so only transcript tokens contribute. Original
manifests are not rewritten.

## AutoDL workflow

Use a fresh Python environment on the single RTX PRO 6000. Clone this repository
and unpack the data separately:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> qwen3-asr-mtp-training
cd qwen3-asr-mtp-training
bash scripts/bootstrap_autodl.sh

python -m mtp_training.preflight \
  --config configs/mtp3-smoke.yaml \
  --audio-checks 100
```

Edit only `dataset_root`, `model_path`, and output/checkpoint paths if your
server layout differs. Then run in this order:

```bash
# 100 optimizer steps; do not start the paid full run until this passes.
python -m mtp_training.train --config configs/mtp3-smoke.yaml

# Frozen Qwen3-ASR, 2,000 optimizer steps.
python -m mtp_training.train --config configs/mtp3-stage1.yaml

# Set init_mtp_from to the chosen Stage-1 checkpoint first.
python -m mtp_training.train --config configs/mtp3-stage2.yaml
```

If FlashAttention installation or execution fails on the selected Blackwell
image, set `attn_implementation: sdpa` for the smoke test. Do not silently
change this in the full run; record the fallback because step time will change.

## Checkpoints and resume

Each `checkpoint-N` contains:

- `trainable_model.safetensors`: only parameters trainable in that stage;
- `trainer_state.pt`: optimizer, scheduler and RNG state for exact resume;
- `mtp_config.json`: resolved recipe and MTP metadata.

Set `resume_from` to resume the same stage. Set `init_mtp_from` in the Stage-2
config to initialize its MTP branches from a Stage-1 checkpoint. Stage-1
checkpoints are small; Stage-2 checkpoints include the changed decoder and are
substantially larger.

## Metrics and go/no-go rule

Evaluation reports per-branch top-1 accuracy and strict average accepted length
globally and for every manifest language/locale layer. Strict acceptance stops
at the first rejected auxiliary token, matching speculative verification.

For the first MTP-3 run, continue to Stage 2 only if the Stage-1 dev result is
stable over two evaluations and global average accepted length is at least
`3.0 / 4`. A commercial inference claim still requires a real propose/verify
engine and RTF test; teacher-forced acceptance from this trainer is a training
gate, not a serving benchmark.

## Local checks

```bash
python -m compileall mtp_training
pytest -q tests/mtp_training
```

The implementation deliberately stays outside `vLLM` and Ascend 310P code.
Once MTP-3 quality is accepted, export/inference integration should be handled
as a separate commit so training behavior and deployment changes remain easy to
review and revert independently.
# Export for vLLM deployment

Export only a completed checkpoint (one that contains `trainable_model.safetensors`,
`trainer_state.pt`, and `mtp_config.json`). The exporter never reads a moving
"latest" pointer and writes the destination atomically:

```bash
python -m mtp_training.export_checkpoint \
  --base-model /models/Qwen3-ASR-1.7B \
  --checkpoint /root/autodl-tmp/outputs/mtp3-stage1/checkpoint-2000 \
  --output-dir /models/Qwen3-ASR-1.7B-MTP3
```

The result is a self-contained Hugging Face directory. Stage 1 exports the
original backbone plus MTP layers; Stage 2 overlays the jointly trained backbone
weights before adding the MTP layers.
