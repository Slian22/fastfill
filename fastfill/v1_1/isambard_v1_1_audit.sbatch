#!/bin/bash
#SBATCH --job-name=ff-v11-cpu-audit
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue

# EVAL_ROOT is the prepared evaluation directory directly inside CLEAN_RUN.
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 NEW_HELPER_ROOT EVAL_ROOT ADDENDUM MODE(sft|report)" >&2
    exit 2
fi
NEW_HELPER_ROOT="$1"
EVAL_ROOT="$2"
ADDENDUM="$3"
MODE="$4"
case "$MODE" in sft|report) ;; *) echo "Invalid audit mode" >&2; exit 2 ;; esac
for path in "$NEW_HELPER_ROOT" "$EVAL_ROOT" "$ADDENDUM"; do
    case "$path" in /*) ;; *) echo "Absolute paths are required" >&2; exit 2 ;; esac
done
ORIGINAL_ROOT="$(cd "$EVAL_ROOT/../../.." && pwd -P)"
PY_BOOTSTRAP="$ORIGINAL_ROOT/env/bin/python"
cd "$NEW_HELPER_ROOT"
export PYTHONPATH="$NEW_HELPER_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Read JSON as data; never evaluate its values as shell code. Both calls validate
# the full path contract, and command-substitution failure propagates under -e.
runner_field() {
    "$PY_BOOTSTRAP" -c '
import json,sys
from pathlib import Path
out=Path(sys.argv[1]).resolve()
receipt=json.loads((out/"runner.json").read_text())
for key,expected in (("output",out),("run",out.parent)):
    value=receipt.get(key)
    if not isinstance(value,str) or Path(value).resolve()!=expected:
        raise ValueError(f"Frozen runner {key} differs from audit directory")
value=receipt.get(sys.argv[2])
if not isinstance(value,str) or not value or "\n" in value or "\r" in value or not Path(value).is_absolute():
    raise ValueError("Invalid frozen runner path")
print(value)
' "$EVAL_ROOT" "$1"
}
PY_EVAL="$(runner_field python)"
SCORING_ROOT="$(runner_field root)"
if [ "$MODE" = sft ]; then
    exec "$PY_EVAL" "$NEW_HELPER_ROOT/fastfill/v1_1/verify_outputs.py" \
        --output "$EVAL_ROOT" --tag sft --scoring-root "$SCORING_ROOT" \
        --workers 32 --reuse-verified
fi
exec "$PY_EVAL" -m fastfill.v1_1.audit_report --output "$EVAL_ROOT" \
    --interpretation "$ADDENDUM" --workers 32
