#!/bin/bash
#SBATCH --job-name=ff-v11-verified-eval
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=115000M
#SBATCH --time=06:00:00
#SBATCH --no-requeue

# Verify provenance with new helpers, then execute the unchanged frozen evaluator.
set -euo pipefail
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 NEW_HELPER_ROOT OLD_FROZEN_ROOT EVAL_ROOT CLEAN_RUN TAG" >&2
    exit 2
fi
NEW_HELPER_ROOT="$1"
OLD_FROZEN_ROOT="$2"
EVAL_ROOT="$3"
CLEAN_RUN="$4"
TAG="$5"
case "$TAG" in sft|dpo1|final) ;; *) echo "Invalid model tag" >&2; exit 2 ;; esac
for path in "$NEW_HELPER_ROOT" "$OLD_FROZEN_ROOT" "$EVAL_ROOT" "$CLEAN_RUN"; do
    case "$path" in /*) ;; *) echo "Absolute paths are required" >&2; exit 2 ;; esac
done
ORIGINAL_ROOT="$(cd "$CLEAN_RUN/../.." && pwd -P)"
PY_TRAIN="$ORIGINAL_ROOT/env/bin/python"
cd "$NEW_HELPER_ROOT"
export PYTHONPATH="$NEW_HELPER_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Argument paths must refer to the experiment recorded by prepare, not a new run.
"$PY_TRAIN" -c '
import json,sys
from pathlib import Path
root,out,run=map(lambda p:Path(p).resolve(),sys.argv[1:])
receipt=json.loads((out/"runner.json").read_text())
for key,expected in (("root",root),("output",out),("run",run)):
    value=receipt.get(key)
    if not isinstance(value,str) or Path(value).resolve()!=expected:
        raise ValueError(f"Frozen runner {key} differs from wrapper arguments")
if not (root/"fastfill/v1_1/isambard_v1_1_evaluate.sbatch").is_file():
    raise ValueError("Missing frozen evaluation script")
' "$OLD_FROZEN_ROOT" "$EVAL_ROOT" "$CLEAN_RUN"

"$PY_TRAIN" -m fastfill.v1_1.training_guard verify-lineage --out "$CLEAN_RUN" --tag "$TAG"
exec bash "$OLD_FROZEN_ROOT/fastfill/v1_1/isambard_v1_1_evaluate.sbatch" \
    "$OLD_FROZEN_ROOT" "$EVAL_ROOT" "$TAG"
