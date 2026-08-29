#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m mtp_training.export_checkpoint \
  --base-model /root/autodl-tmp/models/Qwen3-ASR-1.7B \
  --stage1-checkpoint /root/autodl-tmp/outputs/mtp5-multilingual-4kh-stage1-resume40614/checkpoint-54154 \
  --stage2-checkpoint /root/autodl-tmp/outputs/mtp5-multilingual-4kh-stage2/checkpoint-3384 \
  --output-dir /root/autodl-tmp/models/Qwen3-ASR-1.7B-MTP5-stage2-3384
