#!/bin/bash
#SBATCH --job-name=fastfill-size-api
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
# Usage: sbatch --output=LOG run.sbatch DEPLOY_ROOT REQUESTS NEW_OUT CALL_CAP MAX_ROOMS [PRIOR]
# One GPU, fresh 27B-P inference, no layout API/O2. Size calls have their own hard cap.
set -euo pipefail
test "$#" -eq 5 -o "$#" -eq 6 || exit 2
DEPLOY_ROOT="$1"
REQUESTS="$2"
OUT="$3"
CALL_CAP="$4"
MAX_ROOMS="$5"
PRIOR="${6-}"
case "$CALL_CAP:$MAX_ROOMS" in *[!0-9:]*|:*|*:) exit 2 ;; esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
API_ENV=/home/u6uv/shanliantian.u6uv/.fastfill_api.env
MODEL="$PROJECT_ROOT/models/Qwen3.8-27B"
ADAPTER="$RUN_ROOT/a021/27b/train/final"
test -f "$REQUESTS" -a -f "$API_ENV" -a -d "$MODEL" -a -d "$ADAPTER"
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'rc=$?; printf "{\"exit_code\":%d,\"job_id\":\"%s\"}\n" "$rc" "$SLURM_JOB_ID" > "$OUT/job-status.json"; exit "$rc"' EXIT
export PYTHONPATH="$DEPLOY_ROOT/code" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL
JOB_CACHE="$PROJECT_ROOT/.cache-jobs/size-api-$SLURM_JOB_ID"
PIP_CUDA="$(dirname "$(dirname "$PYTHON")")/lib/python3.11/site-packages/nvidia/cu13"
if test -x "$PIP_CUDA/bin/nvcc"; then
  CUDA_HOME="$PIP_CUDA"
else
  module load cudatoolkit
  NVCC=$(command -v nvcc)
  CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
fi
export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
export HF_HOME=/scratch/u6uv/shanliantian.u6uv/.cache/huggingface XDG_CACHE_HOME="$JOB_CACHE/xdg"
export VLLM_CACHE_ROOT="$JOB_CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor"
export TRITON_CACHE_DIR="$JOB_CACHE/triton" VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
cd "$DEPLOY_ROOT"
sha256sum --check --quiet SHA256SUMS
sha256sum --check --quiet server/SHA256SUMS
"$PYTHON" -B server/detect_resources.py --output "$OUT/resources.json"
"$PYTHON" -B - "$ADAPTER" <<'PY'
from pathlib import Path
import inspect, sys
import torch, vllm, shapely, transformers
from fastfill.comparison import dpo_stages, infer, sample_vllm, train_pair_helpers as h
from fastfill.system import pipeline
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
free, total = torch.cuda.mem_get_info()
fraction = sample_vllm.ENGINE.get('gpu_memory_utilization',
                                inspect.signature(vllm.LLM.__init__).parameters['gpu_memory_utilization'].default)
assert isinstance(fraction, (int, float)) and 0 < fraction < 1
required = fraction * total
assert free >= required + 512 * 1024**2, 'Insufficient free GPU memory before any API call'
print('GPU_MEMORY_GIB', {'free': round(free / 1024**3, 2), 'required': round(required / 1024**3, 2)})
assert Path(pipeline.__file__).resolve().is_relative_to(Path.cwd() / 'code')
assert infer._hash(h.artifact_manifest(sys.argv[1])) == 'cc3e286fa715be1892d13c0e4f29721ac32324903d8bb08e81f1951fee5fd1e6'
assert sample_vllm.EOS in dpo_stages._model_tokenizer(sys.argv[1]).get_vocab()
print('VERSIONS', {'torch': torch.__version__, 'vllm': vllm.__version__, 'transformers': transformers.__version__})
print('PREFLIGHT_OK: one GPU, pinned delivered adapter, isolated deployment imports')
PY
ARGS=(--requests "$REQUESTS" --model "$MODEL" --adapter "$ADAPTER" --placer vllm --escalate none
      --size-fallback api --size-api-cap "$CALL_CAP" --size-api-max-rooms "$MAX_ROOMS"
      --env-file "$API_ENV" --workers 1 --out "$OUT/system")
if test -n "$PRIOR"; then
  ARGS+=(--prior "$PRIOR" --prior-sha256 073522c18e10ce2443189dfb8ebca90074c16ff1c2a4c292dfd6f788409d2422)
fi
timeout --signal=TERM --kill-after=60s 40m "$PYTHON" -B -u -m fastfill.system.cli run "${ARGS[@]}"
