#!/bin/bash
# Open-weight agents (amendments 003/004); run by the parent on the Isambard login node:
#   submit_llm_open.sh CODE_ROOT AFTER [SYSTEM ...]  (default: every system in llm_open.SYSTEMS)
# AFTER: colon-separated GPU job ids every open-model job waits for (afterany), e.g. the 8 inference
# v2 shards and both merges; '-' once they have left Slurm.
# Per system: smoke (2 rooms, gated) -> full (afterok smoke, --kill-on-invalid-dep=yes), so a failed
# smoke cancels its full run and the system stays 'not run' with the smoke's error; nothing is
# substituted. Every job is also 'singleton' under the sbatch's one job name, so at most one
# open-model job (<= 4 GPUs) runs at a time, in submission order (max_concurrent_new_gpus = 8).
# A CPU job (afterany all) then writes runs/unknownsize-20261013/llm-open-status.json.
# Rerunning resumes existing outputs from receipts (same run identity), it never starts them over.
set -euo pipefail
test "$#" -ge 2
CODE_ROOT="$1"
AFTER="$2"
shift 2
case "$AFTER" in -) DEP=singleton ;; ''|:*|*:|*::*|*[!0-9:]*) exit 2 ;; *) DEP="afterany:$AFTER,singleton" ;; esac
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
S="$CODE_ROOT/fastfill/comparison"
test -f "$S/llm_open.py" -a -f "$S/isambard_llm_open.sbatch"
flags() {  # flags SYSTEM MODE: sbatch resources from llm_open.SYSTEMS.
  (cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -m fastfill.comparison.llm_open slurm --system "$1" --mode "$2")
}
SYSTEMS=("$@")
if test "${#SYSTEMS[@]}" -eq 0; then SYSTEMS=(O1a O1b O2 O3 O4 O5); fi
ALL=""
for SYSTEM in "${SYSTEMS[@]}"; do
  case "$SYSTEM" in O1a|O1b|O2|O3|O4|O5) ;; *) echo "unknown system $SYSTEM" >&2; exit 2 ;; esac
  SMOKE_FLAGS=$(flags "$SYSTEM" smoke)
  FULL_FLAGS=$(flags "$SYSTEM" full)
  # shellcheck disable=SC2086  # the flags are space-separated sbatch options
  SMOKE=$(sbatch --parsable --dependency="$DEP" $SMOKE_FLAGS \
    "$S/isambard_llm_open.sbatch" "$CODE_ROOT" "$SYSTEM" smoke)
  SMOKE="${SMOKE%%;*}"
  # shellcheck disable=SC2086
  FULL=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$SMOKE,$DEP" $FULL_FLAGS \
    "$S/isambard_llm_open.sbatch" "$CODE_ROOT" "$SYSTEM" full)
  FULL="${FULL%%;*}"
  echo "$SYSTEM smoke $SMOKE full $FULL"
  ALL="$ALL:$SMOKE:$FULL"
done
STATUS=$(sbatch --parsable --account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=4G --gres=none --time=00:10:00 --no-requeue --dependency="afterany$ALL" \
  --output="$PROJECT_ROOT/runs/isambard/unknownsize-llm-open-status-%j.out" \
  --wrap "cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT $PYTHON -B -m fastfill.comparison.llm_open status --run-root $RUN_ROOT")
echo "status ${STATUS%%;*}"
