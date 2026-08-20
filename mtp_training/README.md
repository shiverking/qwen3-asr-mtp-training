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

## Diagnostics before Stage 2

Do not start Stage 2 when the teacher-forced accepted length is low. Run the
dataset/alignment audit and checkpoint diagnostics first:

```bash
python -m mtp_training.audit_dataset \
  --config configs/mtp3-stage1.yaml \
  --output-dir reports \
  --samples-per-language 100

python -m mtp_training.evaluate_checkpoint \
  --config configs/mtp3-stage1.yaml \
  --checkpoint /root/autodl-tmp/outputs/mtp3-stage1/checkpoint-2000 \
  --output reports/checkpoint-2000-diagnostics.json \
  --gradient-check

python -m mtp_training.verify_checkpoint \
  --config configs/mtp3-stage1.yaml \
  --checkpoint /root/autodl-tmp/outputs/mtp3-stage1/checkpoint-2000 \
  --samples 100 \
  --output reports/reference-verifier.json
```

Build the intentional 1,750-sample overfit set and run the diagnostic recipe:

```bash
python -m mtp_training.make_overfit_manifest \
  --input /root/autodl-tmp/qwen3_asr_mtp_200h/manifests/train.jsonl \
  --output /root/autodl-tmp/qwen3_asr_mtp_200h/manifests/diagnostic-overfit.jsonl \
  --per-language 250

python -m mtp_training.train --config configs/mtp3-overfit.yaml \
  2>&1 | tee reports/tiny-overfit-metrics.jsonl
```

`mtp3-overfit.yaml` preserves the original branch position IDs. Run
`mtp3-overfit-shifted-position.yaml` only as the controlled A/B variant; do not
change the production position convention unless its alignment checks and
overfit result are better.

Only after the alignment audit and overfit gate pass, initialize a new
low-learning-rate Stage-1 phase from checkpoint 2000. This resets the optimizer
and cosine schedule; do not change `max_steps` and resume the exhausted old
scheduler.

```bash
python -m mtp_training.train --config configs/mtp3-stage1-continuation.yaml \
  2>&1 | tee reports/stage1-continuation-metrics.jsonl
```

## Metrics and go/no-go rule

Evaluation reports backbone next-token accuracy, per-branch loss, ground-truth
accuracy, backbone-consistency accuracy and both strict accepted-length
variants, globally and per language. The legacy `branch_accuracy` and
`average_accepted_length` fields remain aliases for ground-truth metrics.

`verify_checkpoint` is deliberately slow: it generates a base token, drafts
serial MTP tokens, then verifies them with the backbone. Use it on a small fixed
sample to check that the faster teacher-forced consistency metric has the same
trend.

### Round-2 low-acceptance investigation

Metric version 2 separates the loss-training target window from positions that
can occur in real multi-token decoding. Version-1 fields remain under
`legacy_*`; existing `branch_accuracy` and `average_accepted_length` aliases
also retain their old meanings. Use only the `decode_window_*` fields for new
go/no-go decisions. `macro_average` weights each manifest language equally.

Re-evaluate an existing checkpoint without training it:

```bash
python -m mtp_training.evaluate_checkpoint \
  --config configs/mtp3-overfit.yaml \
  --checkpoint /root/autodl-tmp/outputs/mtp3-overfit/checkpoint-1000 \
  --output reports/overfit-1000-metrics-v2.json \
  --gradient-check

python -m mtp_training.audit_dataset \
  --config configs/mtp3-overfit.yaml \
  --output-dir reports/overfit-audit \
  --samples-per-language 100

python -m mtp_training.evaluate_backbone_asr \
  --config configs/mtp3-overfit.yaml \
  --languages zh-CN en \
  --samples-per-language 50 \
  --output reports/overfit-backbone-asr.json

python -m mtp_training.verify_checkpoint \
  --config configs/mtp3-overfit.yaml \
  --checkpoint /root/autodl-tmp/outputs/mtp3-overfit/checkpoint-1000 \
  --samples 30 \
  --max-new-tokens 64 \
  --output reports/overfit-reference-30.json
```

`reports/overfit-audit/manual_audio_review.jsonl` contains 20 review rows per
language plus 50 extra Chinese rows. Fill its review fields while listening; it
is intentionally not consumed by training.

Build fixed single-language and position-A/B manifests:

```bash
python -m mtp_training.make_round2_manifests \
  --input /root/autodl-tmp/qwen3_asr_mtp_200h/manifests/diagnostic-overfit.jsonl \
  --output-dir /root/autodl-tmp/qwen3_asr_mtp_200h/manifests
```

Run the two 64-row isolation tests from the unmodified backbone:

```bash
python -m mtp_training.train --config configs/mtp3-overfit-zh64.yaml
python -m mtp_training.train --config configs/mtp3-overfit-en64.yaml
```

If MTP-1 is between 50% and 70% at step 400, run exactly one fresh-scheduler
extension initialized from checkpoint 400:

```bash
python -m mtp_training.train --config configs/mtp3-overfit-zh64-extend.yaml
python -m mtp_training.train --config configs/mtp3-overfit-en64-extend.yaml
```

Do not run an extension below 50%, above 70%, or more than once. After the two
language tests, rerun the 1,750-row set with naturally mixed batches:

```bash
python -m mtp_training.train --config configs/mtp3-overfit-mixed.yaml
```

Position A/B is conditional: run it only if data, mask and causality audits pass
but either 64-row experiment still fails. First run the base/shifted pair. Run
the seed-2 pair only when shifted wins by at least 0.05 decode-window accepted
length in the first pair.

```bash
python -m mtp_training.train --config configs/mtp3-position-ab-base.yaml
python -m mtp_training.train --config configs/mtp3-position-ab-shifted.yaml

# Conditional confirmation only:
python -m mtp_training.train --config configs/mtp3-position-ab-base-seed2.yaml
python -m mtp_training.train --config configs/mtp3-position-ab-shifted-seed2.yaml
```

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
