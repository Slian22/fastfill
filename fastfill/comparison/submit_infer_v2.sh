#!/bin/bash
# Amendment 002 inference v2 chain; run by the parent on the Isambard login node:
#   submit_infer_v2.sh CODE_ROOT JOINT_SFT_JOB PLACEMENT_SFT_JOB [N=4] [TAG=002] [GEOMETRY=geometry-v2]
# plan (no dependency) -> per arm: dev probe gate (amendment 005 section 6; afterok plan + that arm's
# SFT job; pass '-' once the SFT job has left Slurm, probe and shards re-verify TRAINING_COMPLETE
# themselves) -> N shards (afterok plan + that arm's probe) -> merge (afterok its shards) -> geometry
# (afterok both merges). Dependent jobs use --kill-on-invalid-dep=yes. Any failed sbatch aborts the
# chain and lists the job ids already submitted, for scancel.
# Outputs: runs/unknownsize-20261013/{infer-v2-TAG/{plan,joint,placement}, infer-ARM-TAG, GEOMETRY}.
set -euo pipefail
test "$#" -ge 3 -a "$#" -le 6
CODE_ROOT="$1"
N="${4:-4}"
TAG="${5:-002}"
GEOMETRY="${6:-geometry-v2}"
case "$N" in ''|*[!0-9]*) exit 2 ;; esac
test "$N" -ge 1
case "$TAG" in [0-9][0-9][0-9]) ;; *) exit 2 ;; esac
case "$GEOMETRY" in */*|*..*) exit 2 ;; geometry-*) ;; *) exit 2 ;; esac
for SFT in "$2" "$3"; do
  case "$SFT" in -) ;; ''|*[!0-9]*) exit 2 ;; esac
done
RUN_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013
INFER_ROOT="$RUN_ROOT/infer-v2-$TAG"
S="$CODE_ROOT/fastfill/comparison"
for path in "$INFER_ROOT" "$RUN_ROOT/infer-joint-$TAG" "$RUN_ROOT/infer-placement-$TAG" "$RUN_ROOT/$GEOMETRY"; do
  if test -e "$path"; then echo "refusing to reuse existing $path" >&2; exit 1; fi
done
for name in infer_shard.py isambard_infer_plan.sbatch isambard_infer_probe.sbatch isambard_infer_shard.sbatch \
    isambard_infer_merge.sbatch isambard_geometry.sbatch; do
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
# The plan is held until the whole chain is submitted, so an ABORT leaves nothing that ran and the
# same TAG can be resubmitted after the printed scancel.
submit plan "" --hold "$S/isambard_infer_plan.sbatch" "$CODE_ROOT" "$INFER_ROOT"
PLAN="$ID"
MERGES=""
for ARM in joint placement; do
  if test "$ARM" = joint; then SFT="$2"; else SFT="$3"; fi
  AFTER="$PLAN"
  if test "$SFT" != -; then AFTER="$AFTER:$SFT"; fi
  submit "probe $ARM" "$AFTER" "$S/isambard_infer_probe.sbatch" "$CODE_ROOT" "$ARM" "$INFER_ROOT"
  AFTER="$PLAN:$ID"
  SHARDS=""
  for ((i = 0; i < N; i++)); do
    submit "shard $ARM $i" "$AFTER" "$S/isambard_infer_shard.sbatch" "$CODE_ROOT" "$ARM" "$i" "$N" "$INFER_ROOT"
    SHARDS="$SHARDS${SHARDS:+:}$ID"
  done
  submit "merge $ARM" "$SHARDS" "$S/isambard_infer_merge.sbatch" "$CODE_ROOT" "$ARM" "$N" "$INFER_ROOT" \
    "$RUN_ROOT/infer-$ARM-$TAG"
  MERGES="$MERGES${MERGES:+:}$ID"
done
submit geometry "$MERGES" "$S/isambard_geometry.sbatch" "$CODE_ROOT" "$TAG" "$GEOMETRY"
scontrol release "$PLAN" || { echo "ABORT: could not release plan $PLAN; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released plan $PLAN"
