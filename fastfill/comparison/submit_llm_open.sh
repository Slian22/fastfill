#!/bin/bash
# Open-weight agents (amendments 003/004/006/007); run by the parent on the Isambard login node:
#   submit_llm_open.sh CODE_ROOT AFTER [SYSTEM ...]  (default: amendment 007 order O6 O5 O4 O2 O3 O7 O1a O1b)
# AFTER: colon-separated GPU job ids every open-model job waits for (afterany), e.g. the 8 inference
# v2 shards and both merges; '-' once they have left Slurm.
# Per system: smoke (2 rooms, gated) -> full (afterok smoke, --kill-on-invalid-dep=yes), so a failed
# smoke cancels its full run and the system stays 'not run' with the smoke's error; nothing is
# substituted. Every job is also 'singleton' under the sbatch's one job name, so at most one
# open-model job (<= 4 GPUs) runs at a time, in submission order (max_concurrent_new_gpus = 8).
# A CPU job (afterany all) then writes runs/unknownsize-20261013/llm-open-status.json.
# Rerunning resumes existing outputs from receipts (same run identity), it never starts them over.
# Any failed sbatch (or flag lookup) aborts and lists the job ids already submitted, for scancel.
# Refuses to submit anything while any unknownsize-llm-open job of this user is still in Slurm.
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
SUBMITTED=""
abort() {
  echo "ABORT: $1 failed; open-model chain NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
  if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
  exit 1
}
submit() {  # submit LABEL SBATCH_ARGS...; sets ID. set -e does not reach $(...), so check here.
  local out=""
  out=$(sbatch --parsable "${@:2}") || out=""
  ID="${out%%;*}"
  case "$ID" in ''|*[!0-9]*) abort "sbatch for '$1'" ;; esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
}
SYSTEMS=("$@")
if test "${#SYSTEMS[@]}" -eq 0; then SYSTEMS=(O6 O5 O4 O2 O3 O7 O1a O1b); fi
for SYSTEM in "${SYSTEMS[@]}"; do
  case "$SYSTEM" in O1a|O1b|O2|O3|O4|O5|O6|O7) ;; *) echo "unknown system $SYSTEM" >&2; exit 2 ;; esac
done
# An older chain still in Slurm shares the singleton name and output folders and would run first.
QUEUED=$(squeue -h -u "$USER" -n unknownsize-llm-open -o %i) || { echo "ABORT: squeue failed" >&2; exit 1; }
if test -n "$QUEUED"; then
  echo "ABORT: open-model jobs already in Slurm; nothing submitted. cancel with: scancel" $QUEUED >&2
  exit 1
fi
ALL=""
for SYSTEM in "${SYSTEMS[@]}"; do
  SMOKE_FLAGS=$(flags "$SYSTEM" smoke) || abort "flags for $SYSTEM smoke"
  FULL_FLAGS=$(flags "$SYSTEM" full) || abort "flags for $SYSTEM full"
  # shellcheck disable=SC2086  # the flags are space-separated sbatch options
  submit "$SYSTEM smoke" --dependency="$DEP" $SMOKE_FLAGS "$S/isambard_llm_open.sbatch" "$CODE_ROOT" "$SYSTEM" smoke
  SMOKE="$ID"
  # shellcheck disable=SC2086
  submit "$SYSTEM full" --kill-on-invalid-dep=yes --dependency="afterok:$SMOKE,$DEP" $FULL_FLAGS \
    "$S/isambard_llm_open.sbatch" "$CODE_ROOT" "$SYSTEM" full
  echo "$SYSTEM smoke $SMOKE full $ID"
  ALL="$ALL:$SMOKE:$ID"
done
submit status --account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=4G --gres=none --time=00:10:00 --no-requeue --dependency="afterany$ALL" \
  --output="$PROJECT_ROOT/runs/isambard/unknownsize-llm-open-status-%j.out" \
  --wrap "cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT $PYTHON -B -m fastfill.comparison.llm_open status --run-root $RUN_ROOT"
echo "status $ID"
