#!/bin/bash
# Amendment 010 chain, run by the parent on the Isambard login node:
#   submit_dev_selection.sh CODE_ROOT [DEV_N=2]
# Dependencies are afterok with --kill-on-invalid-dep=yes; the root job is held until the whole chain is
# submitted, so an ABORT (it lists the ids for scancel) leaves nothing that ran.
#   build-dev (CPU, held)            dev-select-v1: 600 frozen dev rooms (before any candidate is scored)
#   dev-plan (CPU)                   <- build-dev; amendment 002 shared length-sorted plan for dev
#   train CAND ARM dpo1 (4 GPUs)     <- build-dev; CAND e1 -> dpo-002, rpo -> dpo-003 (inputs from dpo-001)
#   train CAND ARM dpo2 (4 GPUs)     <- train CAND ARM dpo1
#   per candidate sft, e1-dpo1, e1-dpo2, rpo-dpo1, rpo-dpo2 and arm:
#     DEV_N dev shards (1 GPU)       <- dev-plan (+ the candidate's train job); dev-merge (CPU) <- its shards
#   score (CPU)                      <- all 10 dev merges: frozen geometry, per-candidate dev metrics
#   select (CPU)                     <- score: selection.json, then submits the selected models' test chain
# Timed-out GPU steps are resubmitted by hand with the same arguments (shards add --resume).
set -euo pipefail
test "$#" -ge 1 -a "$#" -le 2
CODE_ROOT="$1"
N="${2:-2}"
case "$N" in ''|*[!0-9]*) exit 2 ;; esac
test "$N" -ge 1
RUN_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013
S="$CODE_ROOT/fastfill/comparison"
for path in "$RUN_ROOT"/dpo-00{2,3} "$RUN_ROOT/dev-select-v1" "$RUN_ROOT/dev-eval-010" "$RUN_ROOT/infer-v2-005" \
    "$RUN_ROOT"/infer-{joint,placement}-005 "$RUN_ROOT/geometry-v5"; do
  if test -e "$path"; then echo "refusing to reuse existing $path" >&2; exit 1; fi
done
for name in dev_select.py dpo_stages.py infer_shard.py evaluate_pair.py isambard_010_cpu.sbatch \
    isambard_010_infer_shard.sbatch isambard_dpo_train.sbatch isambard_infer_merge.sbatch; do
  test -f "$S/$name"
done
SUBMITTED=""
submit() {  # submit LABEL DEPENDENCY SCRIPT ARGS...; sets ID. set -e does not reach $(...), so check here.
  local out=""
  if test -n "$2"; then
    out=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$2" "${@:3}") || out=""
  else
    out=$(sbatch --parsable "${@:3}") || out=""
  fi
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$1'; chain NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  echo "$1 $ID"
}
submit build-dev "" --hold "$S/isambard_010_cpu.sbatch" "$CODE_ROOT" build-dev
ROOT="$ID"
submit dev-plan "$ROOT" "$S/isambard_010_cpu.sbatch" "$CODE_ROOT" dev-plan
PLAN="$ID"
ARMS=(joint placement)
TRAINED=()  # "CAND-STAGE ARM JOB" lines; bash 3 (macOS test host) has no associative arrays.
for CAND in e1 rpo; do
  for ARM in "${ARMS[@]}"; do
    submit "train $CAND $ARM dpo1" "$ROOT" "$S/isambard_dpo_train.sbatch" "$CODE_ROOT" "$ARM" dpo1 "$CAND"
    TRAINED+=("$CAND-dpo1 $ARM $ID")
    submit "train $CAND $ARM dpo2" "$ID" "$S/isambard_dpo_train.sbatch" "$CODE_ROOT" "$ARM" dpo2 "$CAND"
    TRAINED+=("$CAND-dpo2 $ARM $ID")
  done
done
trained() {  # trained CANDIDATE ARM -> the train job id producing that model
  local line
  for line in "${TRAINED[@]}"; do
    set -- "$1" "$2" $line
    if test "$1" = "$3" -a "$2" = "$4"; then echo "$5"; return; fi
  done
  exit 1
}
MERGES=""
for CAND in sft e1-dpo1 e1-dpo2 rpo-dpo1 rpo-dpo2; do
  for ARM in "${ARMS[@]}"; do
    AFTER="$PLAN"
    if test "$CAND" != sft; then AFTER="$AFTER:$(trained "$CAND" "$ARM")"; fi
    SHARDS=""
    for ((i = 0; i < N; i++)); do
      submit "dev shard $CAND $ARM $i" "$AFTER" "$S/isambard_010_infer_shard.sbatch" "$CODE_ROOT" dev "$ARM" "$CAND" "$i" "$N"
      SHARDS="$SHARDS${SHARDS:+:}$ID"
    done
    submit "dev merge $CAND $ARM" "$SHARDS" "$S/isambard_010_cpu.sbatch" "$CODE_ROOT" dev-merge "$ARM" "$CAND" "$N"
    MERGES="$MERGES${MERGES:+:}$ID"
  done
done
submit score "$MERGES" "$S/isambard_010_cpu.sbatch" "$CODE_ROOT" score
submit select "$ID" "$S/isambard_010_cpu.sbatch" "$CODE_ROOT" select
scontrol release "$ROOT" || { echo "ABORT: could not release $ROOT; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released build-dev $ROOT"
