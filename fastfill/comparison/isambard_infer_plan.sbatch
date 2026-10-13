#!/bin/bash
#SBATCH --job-name=unknownsize-infer-plan
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-infer-plan-%j.out
# Amendment 002 shared plan (pinned base tokenizer, no GPU): CODE_ROOT INFER_ROOT -> INFER_ROOT/plan, once.
set -euo pipefail
test "$#" -eq 2
PAIR_CODE_ROOT="$1"
PAIR_INFER_ROOT="$2"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
case "$PAIR_INFER_ROOT" in "$PAIR_RUN_ROOT"/infer-v2-[0-9][0-9][0-9]) ;; *) exit 2 ;; esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
test ! -e "$PAIR_INFER_ROOT"
module load cudatoolkit/24.11_12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH="$CUDA_HOME" PYTHONPATH="$PAIR_CODE_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/tmp PYTHONUNBUFFERED=1
cd "$PAIR_CODE_ROOT"
exec timeout --signal=TERM --kill-after=30s 55m "$PROJECT_ROOT/env/bin/python" -B -u \
  -m fastfill.comparison.infer_shard plan --data "$PAIR_RUN_ROOT/eval-data-v1" \
  --base "$PROJECT_ROOT/models/Qwen3-8B" --out "$PAIR_INFER_ROOT/plan" \
  --batch-size 8 --max-input-tokens 40959 --max-new-tokens 40959 --context-budget 40960
