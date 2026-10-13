#!/bin/bash
# Amendment 002 inference v2 chain; run by the parent on the Isambard login node:
#   submit_infer_v2.sh CODE_ROOT JOINT_SFT_JOB PLACEMENT_SFT_JOB [N=4] [TAG=002] [GEOMETRY=geometry-v2]
# plan (no dependency) -> N shards per arm (afterok plan + that arm's SFT job; pass '-' once the SFT
# job has left Slurm, the shard re-verifies TRAINING_COMPLETE itself) -> merge per arm (afterok its
# shards) -> geometry (afterok both merges). Dependent jobs use --kill-on-invalid-dep=yes.
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
for name in infer_shard.py isambard_infer_plan.sbatch isambard_infer_shard.sbatch isambard_infer_merge.sbatch isambard_geometry.sbatch; do
  test -f "$S/$name"
done
submit() {  # submit DEPENDENCY SCRIPT ARGS...; prints the job id only.
  local id
  if test -n "$1"; then
    id=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$1" "${@:2}")
  else
    id=$(sbatch --parsable "${@:2}")
  fi
  echo "${id%%;*}"
}
PLAN=$(submit "" "$S/isambard_infer_plan.sbatch" "$CODE_ROOT" "$INFER_ROOT")
echo "plan $PLAN"
MERGES=""
for ARM in joint placement; do
  if test "$ARM" = joint; then SFT="$2"; else SFT="$3"; fi
  AFTER="$PLAN"
  if test "$SFT" != -; then AFTER="$AFTER:$SFT"; fi
  SHARDS=""
  for ((i = 0; i < N; i++)); do
    ID=$(submit "$AFTER" "$S/isambard_infer_shard.sbatch" "$CODE_ROOT" "$ARM" "$i" "$N" "$INFER_ROOT")
    echo "shard $ARM $i $ID"
    SHARDS="$SHARDS${SHARDS:+:}$ID"
  done
  ID=$(submit "$SHARDS" "$S/isambard_infer_merge.sbatch" "$CODE_ROOT" "$ARM" "$N" "$INFER_ROOT" "$RUN_ROOT/infer-$ARM-$TAG")
  echo "merge $ARM $ID"
  MERGES="$MERGES${MERGES:+:}$ID"
done
ID=$(submit "$MERGES" "$S/isambard_geometry.sbatch" "$CODE_ROOT" "$TAG" "$GEOMETRY")
echo "geometry $ID"
