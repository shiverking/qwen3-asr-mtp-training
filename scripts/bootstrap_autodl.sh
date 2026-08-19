#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-mtp.txt

if ! python -c "import flash_attn" >/dev/null 2>&1; then
  MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation
fi

python -c "import torch, qwen_asr; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0))"
