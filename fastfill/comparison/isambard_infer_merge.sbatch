#!/bin/bash
#SBATCH --job-name=unknownsize-infer-merge
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-infer-merge-%j.out
# Amendment 002 merge: CODE_ROOT ARM SHARD_COUNT INFER_ROOT OUT -> canonical evaluate_pair input dir.
set -euo pipefail
test "$#" -eq 5
PAIR_CODE_ROOT="$1"
PAIR_ARM="$2"
PAIR_SHARD_COUNT="$3"
PAIR_INFER_ROOT="$4"
PAIR_OUT="$5"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
case "$PAIR_ARM" in joint|placement) ;; *) exit 2 ;; esac
case "$PAIR_SHARD_COUNT" in ''|*[!0-9]*) exit 2 ;; esac
case "$PAIR_INFER_ROOT" in "$PAIR_RUN_ROOT"/infer-v2-[0-9][0-9][0-9]) ;; *) exit 2 ;; esac
case "$PAIR_OUT" in "$PAIR_RUN_ROOT/infer-$PAIR_ARM"-[0-9][0-9][0-9]) ;; *) exit 2 ;; esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
test ! -e "$PAIR_OUT"
SHARDS=()
for ((i = 0; i < PAIR_SHARD_COUNT; i++)); do
  SHARDS+=("$PAIR_INFER_ROOT/$PAIR_ARM/shard-$(printf %02d "$i")")
done
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PAIR_CODE_ROOT"
exec timeout --signal=TERM --kill-after=30s 55m "$PROJECT_ROOT/env/bin/python" -B -u \
  -m fastfill.comparison.infer_shard merge --data "$PAIR_RUN_ROOT/eval-data-v1" \
  --plan "$PAIR_INFER_ROOT/plan" --arm "$PAIR_ARM" --out "$PAIR_OUT" --shards "${SHARDS[@]}"
