#!/bin/bash
#SBATCH --job-name=unknownsize-llm-open
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-llm-open-%j.out
# Usage: sbatch $(python -m fastfill.comparison.llm_open slurm --system S --mode M) \
#          isambard_llm_open.sbatch CODE_ROOT S smoke|full
# GPUs/CPUs/memory/time per system come from llm_open.SYSTEMS (submit_llm_open.sh passes them).
# smoke: 2 cohort rooms; exits non-zero only when the system cannot load or execute (amendment 006:
# cap-hit reasoning or invalid answers are measured failures, and the full run still follows).
# Preflight (fails fast, before the GPU work): cudatoolkit/nvcc for vLLM JIT kernels, and the vLLM env
# must import shapely + fastfill.comparison.
# Resubmitting the same system and mode resumes from receipts under the same run identity.
set -euo pipefail
test "$#" -eq 3
PAIR_CODE_ROOT="$1"
SYSTEM="$2"
case "$SYSTEM" in O1a|O1b|O2|O3|O4|O5|O6|O7) ;; *) exit 2 ;; esac
case "$3" in
  smoke) OUT="llm-open-$SYSTEM-smoke-001"; EXTRA=(--limit 2 --smoke-gate) ;;
  full) OUT="llm-open-$SYSTEM-001"; EXTRA=() ;;
  *) exit 2 ;;
esac
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
COHORT="$PAIR_RUN_ROOT/llm-cohort-v1.json"
PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
CACHE=/scratch/u6uv/shanliantian.u6uv/.cache
# Scratch hit its 1.024M-file project limit (2026-10-14); compile caches (many small files) live under
# $PROJECTDIR instead. HF_HOME stays on scratch: offline, read-only model metadata.
SYSTEM_CACHE="/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/.cache-jobs/llm-open/$SYSTEM"
: "${SLURM_JOB_ID:?Slurm allocation required}"
# JIT builds (FlashInfer, DeepGEMM) need an nvcc matching torch cu130: DeepGEMM asserts nvcc >= 12.9 and the
# cluster module is 12.6, so prefer the env's pip CUDA 13 toolkit; fall back to the module (v1_1 pattern).
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
export XDG_CACHE_HOME="$SYSTEM_CACHE/xdg" VLLM_CACHE_ROOT="$SYSTEM_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$SYSTEM_CACHE/torchinductor" TRITON_CACHE_DIR="$SYSTEM_CACHE/triton"
# FlashInfer's sampler JIT build fails on this GH200 image; large checkpoints may need a long engine start.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
cd "$PAIR_CODE_ROOT"
"$PYTHON" -B -c 'import shapely, fastfill.comparison.llm_open' || {
  echo "PREFLIGHT FAIL: $PYTHON cannot import shapely + fastfill.comparison from $PAIR_CODE_ROOT;" \
    "install shapely into the vLLM env (system $SYSTEM stays 'not run')" >&2; exit 1; }
# The API-agent smoke normally writes the cohort first; it is deterministic and write-once, so a
# concurrent writer is harmless (llm_open rebuilds and compares it anyway).
test -e "$COHORT" || "$PYTHON" -B -m fastfill.comparison.llm_agent cohort \
  --data "$PAIR_RUN_ROOT/eval-data-v1" --out "$COHORT" || test -e "$COHORT"
nvidia-smi -L
exec "$PYTHON" -B -u -m fastfill.comparison.llm_open run --system "$SYSTEM" \
  --data "$PAIR_RUN_ROOT/eval-data-v1" --cohort "$COHORT" --model-root "$PROJECT_ROOT/models" \
  --out "$PAIR_RUN_ROOT/$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
