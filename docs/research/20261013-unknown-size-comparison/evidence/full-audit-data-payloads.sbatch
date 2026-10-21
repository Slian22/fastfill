#!/bin/bash
#SBATCH --job-name=full-audit-data-payloads
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-payloads-%j.out
set -euo pipefail
ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
AUDIT="$ROOT/docs/research/20261013-unknown-size-comparison/evidence"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
exec timeout --signal=TERM --kill-after=15s 9m "$ROOT/env/bin/python" -B -u \
 "$AUDIT/full-audit-data-payloads.py" --root "$ROOT" \
 --prior-result "$AUDIT/full-audit-data-lineage-7368003/result.json" \
 --out "$AUDIT/full-audit-data-payloads-$SLURM_JOB_ID"
