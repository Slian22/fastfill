#!/bin/bash
# Amendment 008 chain (SFT -> DPO1 -> DPO2, both arms, then stage evaluations); run by the parent on the
# Isambard login node:
#   submit_dpo_stages.sh CODE_ROOT JOINT_SFT_JOB PLACEMENT_SFT_JOB [N=4]
# Pass '-' for an SFT job that has left Slurm (every step re-verifies TRAINING_COMPLETE itself).
# Dependencies are afterok with --kill-on-invalid-dep=yes; the root job is held until the whole chain is
# submitted, so an ABORT (it lists the ids for scancel) leaves nothing that ran.
#   plans (CPU, held)     infer-v2-002/plan -> infer-v2-003/plan, infer-v2-004/plan (byte-identical)
#   positives (CPU)       <- plans + both SFT jobs
#   merge-sft ARM (CPU)   <- plans + ARM's SFT job
#   generate ARM (1 GPU)  <- positives + merge-sft ARM
#   stage1-pairs (CPU)    <- both generates           stage2-pairs (CPU) <- positives
#   train ARM dpo1 (1 GPU) <- stage1-pairs (+ merge-sft ARM through generate)
#   train ARM dpo2 (1 GPU) <- train ARM dpo1 + stage2-pairs
#   per stage (dpo1 -> infer-v2-003, geometry-v3; dpo2 -> infer-v2-004, geometry-v4):
#     N shards per arm <- train ARM STAGE; merge ARM <- its shards; geometry <- both merges
# Timed-out GPU steps are resubmitted by hand with the same arguments (shards add --resume).
set -euo pipefail
test "$#" -ge 3 -a "$#" -le 4
CODE_ROOT="$1"
N="${4:-4}"
case "$N" in ''|*[!0-9]*) exit 2 ;; esac
test "$N" -ge 1
for SFT in "$2" "$3"; do
  case "$SFT" in -) ;; ''|*[!0-9]*) exit 2 ;; esac
done
RUN_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013
S="$CODE_ROOT/fastfill/comparison"
for path in "$RUN_ROOT/dpo-001" "$RUN_ROOT/infer-v2-003" "$RUN_ROOT/infer-v2-004" "$RUN_ROOT/geometry-v3" \
    "$RUN_ROOT/geometry-v4" "$RUN_ROOT"/infer-{joint,placement}-00{3,4}; do
  if test -e "$path"; then echo "refusing to reuse existing $path" >&2; exit 1; fi
done
for name in dpo_stages.py infer_shard.py isambard_dpo_cpu.sbatch isambard_dpo_generate.sbatch \
    isambard_dpo_train.sbatch isambard_dpo_infer_shard.sbatch isambard_infer_merge.sbatch isambard_geometry.sbatch; do
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
sft() { if test "$1" = joint; then echo "$2"; else echo "$3"; fi; }
with() {  # with JOBS...: colon-joined, skipping '-'
  local joined=""
  for job in "$@"; do
    if test "$job" != -; then joined="$joined${joined:+:}$job"; fi
  done
  echo "$joined"
}
submit plans "" --hold "$S/isambard_dpo_cpu.sbatch" "$CODE_ROOT" copy-plans
PLANS="$ID"
submit positives "$(with "$PLANS" "$2" "$3")" "$S/isambard_dpo_cpu.sbatch" "$CODE_ROOT" positives
POSITIVES="$ID"
GENERATES=""
for ARM in joint placement; do
  submit "merge-sft $ARM" "$(with "$PLANS" "$(sft "$ARM" "$2" "$3")")" \
    "$S/isambard_dpo_cpu.sbatch" "$CODE_ROOT" merge-sft "$ARM"
  submit "generate $ARM" "$POSITIVES:$ID" "$S/isambard_dpo_generate.sbatch" "$CODE_ROOT" "$ARM"
  GENERATES="$GENERATES${GENERATES:+:}$ID"
done
submit stage1-pairs "$GENERATES" "$S/isambard_dpo_cpu.sbatch" "$CODE_ROOT" stage1-pairs
STAGE1="$ID"
submit stage2-pairs "$POSITIVES" "$S/isambard_dpo_cpu.sbatch" "$CODE_ROOT" stage2-pairs
STAGE2="$ID"
ARMS=(joint placement)
DPO1=()  # Indexed like ARMS: bash 3 (macOS test host) has no associative arrays.
DPO2=()
for ARM in "${ARMS[@]}"; do
  submit "train $ARM dpo1" "$STAGE1" "$S/isambard_dpo_train.sbatch" "$CODE_ROOT" "$ARM" dpo1
  DPO1+=("$ID")
  submit "train $ARM dpo2" "$ID:$STAGE2" "$S/isambard_dpo_train.sbatch" "$CODE_ROOT" "$ARM" dpo2
  DPO2+=("$ID")
done
for STAGE in dpo1 dpo2; do
  if test "$STAGE" = dpo1; then TAG=003; GEOMETRY=geometry-v3; else TAG=004; GEOMETRY=geometry-v4; fi
  INFER_ROOT="$RUN_ROOT/infer-v2-$TAG"
  MERGES=""
  for k in 0 1; do
    ARM="${ARMS[$k]}"
    if test "$STAGE" = dpo1; then TRAINED="${DPO1[$k]}"; else TRAINED="${DPO2[$k]}"; fi
    SHARDS=""
    for ((i = 0; i < N; i++)); do
      submit "shard $STAGE $ARM $i" "$TRAINED" "$S/isambard_dpo_infer_shard.sbatch" \
        "$CODE_ROOT" "$STAGE" "$ARM" "$i" "$N" "$INFER_ROOT"
      SHARDS="$SHARDS${SHARDS:+:}$ID"
    done
    submit "merge $STAGE $ARM" "$SHARDS" "$S/isambard_infer_merge.sbatch" "$CODE_ROOT" "$ARM" "$N" "$INFER_ROOT" \
      "$RUN_ROOT/infer-$ARM-$TAG"
    MERGES="$MERGES${MERGES:+:}$ID"
  done
  submit "geometry $STAGE" "$MERGES" "$S/isambard_geometry.sbatch" "$CODE_ROOT" "$TAG" "$GEOMETRY"
done
scontrol release "$PLANS" || { echo "ABORT: could not release plans $PLANS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released plans $PLANS"
