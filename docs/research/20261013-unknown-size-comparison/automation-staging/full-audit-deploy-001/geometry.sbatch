#!/bin/bash
#SBATCH --job-name=unknownsize-geometry-audit
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=none
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-geometry-audit-%j.out
set -euo pipefail
test "$#" -eq 3
AUDIT_CODE_ROOT="$1"
AUDIT_SOURCE_RUN="$2"
AUDIT_PYTHON="$3"
export PYTHONPATH="$AUDIT_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$AUDIT_CODE_ROOT"
# Verify the complete frozen production snapshot before importing it.
"$AUDIT_PYTHON" -B -c 'import sys; from fastfill.comparison.overnight import verify_manifest; verify_manifest(sys.argv[1])' "$AUDIT_CODE_ROOT"
# Parent must receipt pending/never-started cancellation of the old geometry
# before reserving this previously unused destination. Do not remove it if found.
test ! -e "$AUDIT_SOURCE_RUN/geometry-v1"
test ! -L "$AUDIT_SOURCE_RUN/geometry-v1"
exec timeout --signal=TERM --kill-after=30s 115m "$AUDIT_PYTHON" -B -u \
  -m fastfill.comparison.evaluate_pair --data "$AUDIT_SOURCE_RUN/eval-data-v1" \
  --joint "$AUDIT_SOURCE_RUN/infer-joint-001" \
  --placement "$AUDIT_SOURCE_RUN/infer-placement-001" \
  --out "$AUDIT_SOURCE_RUN/geometry-v1" --workers 24 --repeats 2000
