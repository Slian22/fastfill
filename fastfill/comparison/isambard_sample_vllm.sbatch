#!/bin/bash
#SBATCH --job-name=unknownsize-s2-sample
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-s2-sample-%j.out
# S2 best-of-N sampling (amendment-011 draft): CODE_ROOT dev ARM SHARD_COUNT [--limit M]
#   sbatch --array=0-$((K-1)) isambard_sample_vllm.sbatch CODE_ROOT dev ARM K         shard = SLURM_ARRAY_TASK_ID
#   sbatch --time=02:00:00 isambard_sample_vllm.sbatch CODE_ROOT dev ARM 1 --limit 16  smoke: the 16 largest rooms
# GPU layout: tensor_parallel_size 1, one vLLM engine per 1-GPU array task (data-parallel shards). Qwen3-8B in BF16
# (~16 GB) fits one 96 GB GH200 with most of it left for KV cache, so TP would only add all-reduce traffic; 1-GPU
# tasks pack onto a node's 4 GPUs when they are free, start on partly used nodes, and fail and resume per shard.
# ARM joint = the dev-selected sft (SFT adapter on the generic base); placement = rpo-dpo1 (the dpo-003 stage's exact
# policy: its adapter on its BF16 training base). Preflight: dev-eval-010/selection.json still selects exactly these,
# dev-full-v1 exists, and the vLLM env imports vllm, shapely and fastfill.comparison.sample_vllm.
# Output: RUN/s2-011/ARM-CANDIDATE/samples, shared by the smoke and every shard under one run identity; resubmitting
# resumes from receipts. Only dev: the test set needs a later amendment.
set -euo pipefail
case "$#" in
  4) ;;
  6) test "$5" = --limit || exit 2; case "$6" in ''|*[!0-9]*) exit 2 ;; esac ;;
  *) exit 2 ;;
esac
PAIR_CODE_ROOT="$1"
test "$2" = dev || exit 2
ARM="$3"
COUNT="$4"
case "$COUNT" in ''|*[!0-9]*) exit 2 ;; esac
test "$COUNT" -ge 1 || exit 2
INDEX="${SLURM_ARRAY_TASK_ID:-}"
if test -z "$INDEX"; then
  test "$COUNT" -eq 1 || exit 2  # Several shards need an array job.
  INDEX=0
fi
case "$INDEX" in *[!0-9]*) exit 2 ;; esac
test "$INDEX" -lt "$COUNT" || exit 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
case "$ARM" in
  joint) CAND=sft
         MODEL_ARGS=(--adapter "$RUN_ROOT/sft-joint-001/joint/adapter" --training-run "$RUN_ROOT/sft-joint-001") ;;
  placement) CAND=rpo-dpo1
             MODEL_ARGS=(--model "$RUN_ROOT/dpo-003/placement/dpo1-merged" --exact-policy) ;;
  *) exit 2 ;;
esac
LIMIT=()
if test "$#" -eq 6; then LIMIT=(--limit "$6"); fi
: "${SLURM_JOB_ID:?Slurm allocation required}"
DATA="$RUN_ROOT/dev-full-v1"
OUT="$RUN_ROOT/s2-011/$ARM-$CAND/samples"
PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
CACHE=/scratch/u6uv/shanliantian.u6uv/.cache
# Scratch hit its file limit (2026-10-14): compile caches live under $PROJECTDIR; HF_HOME stays on scratch (offline).
JOB_CACHE="$PROJECT_ROOT/.cache-jobs/s2/$ARM"
test -f "$DATA/MANIFEST.json" || { echo "PREFLIGHT FAIL: $DATA missing (sample_vllm build)" >&2; exit 1; }
# vLLM JIT kernels need an nvcc matching torch cu130: prefer the env's pip CUDA 13 toolkit, else the module.
PIP_CUDA="$(dirname "$(dirname "$PYTHON")")/lib/python3.11/site-packages/nvidia/cu13"
if test -x "$PIP_CUDA/bin/nvcc"; then
  CUDA_HOME="$PIP_CUDA"
else
  module load cudatoolkit || { echo "PREFLIGHT FAIL: module load cudatoolkit" >&2; exit 1; }
  NVCC=$(command -v nvcc) || { echo "PREFLIGHT FAIL: no nvcc after 'module load cudatoolkit'" >&2; exit 1; }
  CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
fi
"$CUDA_HOME/bin/nvcc" --version | tail -1
export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$CACHE/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$JOB_CACHE/xdg" VLLM_CACHE_ROOT="$JOB_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor" TRITON_CACHE_DIR="$JOB_CACHE/triton"
# FlashInfer's sampler JIT build fails on this GH200 image.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
cd "$PAIR_CODE_ROOT"
"$PYTHON" -B -c 'import vllm, shapely, fastfill.comparison.sample_vllm' || {
  echo "PREFLIGHT FAIL: $PYTHON cannot import vllm + shapely + fastfill.comparison.sample_vllm" >&2; exit 1; }
"$PYTHON" -B - "$RUN_ROOT/dev-eval-010/selection.json" "$ARM" "$CAND" <<'PY' || {
import json, sys
selected = json.load(open(sys.argv[1]))['selected']
assert selected == {'joint': 'sft', 'placement': 'rpo-dpo1'} and selected[sys.argv[2]] == sys.argv[3], selected
PY
  echo "PREFLIGHT FAIL: dev-eval-010/selection.json no longer selects sft (joint) and rpo-dpo1 (placement)" >&2; exit 1; }
nvidia-smi -L
exec timeout --signal=TERM --kill-after=60s 230m "$PYTHON" -B -u -m fastfill.comparison.sample_vllm run \
  --data "$DATA" --base "$PROJECT_ROOT/models/Qwen3-8B" --arm "$ARM" "${MODEL_ARGS[@]}" --out "$OUT" \
  --shard-index "$INDEX" --shard-count "$COUNT" ${LIMIT[@]+"${LIMIT[@]}"}
