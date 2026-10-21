#!/bin/bash
#SBATCH --job-name=full-audit-data-inputs
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-inputs-%j.out
set -euo pipefail
ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
AUDIT="$ROOT/docs/research/20261013-unknown-size-comparison/evidence"
module load cudatoolkit/24.11_12.6
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
exec timeout --signal=TERM --kill-after=30s 43m "$ROOT/env/bin/python" -B -u \
  "$AUDIT/full-audit-data-inputs.py" --root "$ROOT" --out "$AUDIT/full-audit-data-inputs-$SLURM_JOB_ID"
