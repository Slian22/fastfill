#!/bin/bash
#SBATCH --job-name=fastfill-v11-eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=115000M
#SBATCH --time=06:00:00

# Usage: provide CODE_ROOT OUTPUT_ROOT MODEL_TAG as positional arguments.
# The caller controls job logs and schedules at most one continuation per model.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 CODE_ROOT OUTPUT_ROOT MODEL_TAG" >&2
    exit 2
fi
CODE_ROOT="$1"
OUTPUT_ROOT="$2"
MODEL_TAG="$3"
case "$MODEL_TAG" in sft|dpo1|final) ;; *) echo "Invalid model tag" >&2; exit 2 ;; esac
cd "$CODE_ROOT"

module load cudatoolkit
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
PY_VLLM="${PY_VLLM:-/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python}"
export PATH="$(dirname "$PY_VLLM"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$CODE_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp
TASK_CACHE="/tmp/fastfill_eval_${USER:-u6uv}_${SLURM_JOB_ID:-local}"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/torchinductor"
export TRITON_CACHE_DIR="$TASK_CACHE/triton"
export VLLM_CACHE_ROOT="$TASK_CACHE/vllm"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# A failed/timeout evaluation remains incomplete; the caller may resume chunks.
trap 'status=$?; "$PY_VLLM" -m fastfill.v1_1.automate report --output "$OUTPUT_ROOT" || true; exit "$status"' EXIT
"$PY_VLLM" -m fastfill.v1_1.automate run-model --output "$OUTPUT_ROOT" --tag "$MODEL_TAG"
