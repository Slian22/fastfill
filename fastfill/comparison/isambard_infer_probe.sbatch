#!/bin/bash
#SBATCH --job-name=unknownsize-infer-probe
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-infer-probe-%j.out
# Amendment 005 section 6 dev probe gate: CODE_ROOT ARM INFER_ROOT -> INFER_ROOT/ARM/probe/probe.json, once.
# Exits non-zero (the arm's shards never start) iff a recorded schema-valid dev probe turns schema-invalid.
set -euo pipefail
test "$#" -eq 3
PAIR_CODE_ROOT="$1"
PAIR_ARM="$2"
PAIR_INFER_ROOT="$3"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
case "$PAIR_ARM" in joint|placement) ;; *) exit 2 ;; esac
case "$PAIR_INFER_ROOT" in "$PAIR_RUN_ROOT"/infer-v2-[0-9][0-9][0-9]) ;; *) exit 2 ;; esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
test -f "$PAIR_RUN_ROOT/sft-$PAIR_ARM-001/$PAIR_ARM/dev_generations.json"
module load cudatoolkit/24.11_12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH="$CUDA_HOME" PYTHONPATH="$PAIR_CODE_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/tmp PYTHONUNBUFFERED=1
cd "$PAIR_CODE_ROOT"
exec timeout --signal=TERM --kill-after=30s 42m "$PROJECT_ROOT/env/bin/python" -B -u \
  -m fastfill.comparison.infer_shard probe --base "$PROJECT_ROOT/models/Qwen3-8B" --arm "$PAIR_ARM" \
  --adapter "$PAIR_RUN_ROOT/sft-$PAIR_ARM-001/$PAIR_ARM/adapter" \
  --training-run "$PAIR_RUN_ROOT/sft-$PAIR_ARM-001" --out "$PAIR_INFER_ROOT/$PAIR_ARM/probe"
