#!/bin/bash
#SBATCH --job-name=unknownsize-geometry
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-geometry-%j.out
set -euo pipefail
test "$#" -eq 1
PAIR_CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PAIR_CODE_ROOT"
exec timeout --signal=TERM --kill-after=30s 115m "$PROJECT_ROOT/env/bin/python" -B -u \
  -m fastfill.comparison.evaluate_pair --data "$PAIR_RUN_ROOT/eval-data-v1" \
  --joint "$PAIR_RUN_ROOT/infer-joint-001" --placement "$PAIR_RUN_ROOT/infer-placement-001" \
  --out "$PAIR_RUN_ROOT/geometry-v1" --workers 24 --repeats 2000
