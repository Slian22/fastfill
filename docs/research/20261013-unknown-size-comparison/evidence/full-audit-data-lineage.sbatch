#!/bin/bash
#SBATCH --job-name=full-audit-data-lineage
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-lineage-%j.out
set -uo pipefail
ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
AUDIT="$ROOT/docs/research/20261013-unknown-size-comparison/evidence"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
timeout --signal=TERM --kill-after=30s 40m "$ROOT/env/bin/python" -B -u \
  "$AUDIT/full-audit-data-lineage.py" --root "$ROOT" \
  --out "$AUDIT/full-audit-data-lineage-$SLURM_JOB_ID"
LINEAGE_RC=$?
timeout --signal=TERM --kill-after=15s 3m "$ROOT/env/bin/python" -B -u \
  "$AUDIT/full-audit-data-p-target-check.py" --root "$ROOT" \
  --out "$AUDIT/full-audit-data-p-target-$SLURM_JOB_ID"
TARGET_RC=$?
if [ "$LINEAGE_RC" -ne 0 ]; then exit "$LINEAGE_RC"; fi
exit "$TARGET_RC"
