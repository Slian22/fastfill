#!/bin/bash
# External recovery wrapper: original snapshot and research settings remain unchanged.
set -euo pipefail
test "$#" -eq 1 || exit 2
STEP="$1"
case "$STEP" in shard|finalize|summarize) ;; *) exit 2 ;; esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
OLD="$ROOT/snapshots/main-498348274"
OPS="$ROOT/ops/a027-resume-20261015-v1"
export PYTHONPATH="$OLD" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL
cd "$OLD"
case "$STEP" in
  shard)
    "$ROOT/env/bin/python" -B "$OPS/verify.py" before-shard
    /home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python -B "$OPS/preflight.py"
    exec bash "$OLD/fastfill/system/isambard_022.sbatch" "$OLD" shard final 1 12 strict ;;
  finalize)
    "$ROOT/env/bin/python" -B "$OPS/verify.py" before-finalize
    exec bash "$OLD/fastfill/system/isambard_022.sbatch" "$OLD" finalize final strict ;;
  summarize)
    bash "$OLD/fastfill/system/isambard_022.sbatch" "$OLD" summarize final test-strict-sharded
    "$ROOT/env/bin/python" -B "$OPS/verify.py" after-summarize ;;
esac
