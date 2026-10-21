#!/bin/bash
# Amendment 028 (27B joint arm J27, fastfill.comparison.a028 + a021_train --arm J27), run by the parent on the Isambard
# login node. ONE command chains every stage; each later job waits on the one before, so nobody has to intervene:
#   submit_028.sh CODE_ROOT all       rows -> smoke -> smoke-vllm -> train x2 -> greedy dev-select + test -> compare
#   submit_028.sh CODE_ROOT smoke     the same chain from the smoke on (rows exist; e.g. after a crashed smoke was moved)
#   submit_028.sh CODE_ROOT smoke-vllm   from the vLLM load of the smoke adapter on (SMOKE_PASS exists)
#   submit_028.sh CODE_ROOT train     from training (SMOKE_PASS and vllm-load.json OK exist; a partial run resumes)
#   submit_028.sh CODE_ROOT infer     the vLLM runs not yet complete (each resumes from its receipts), then compare
#   submit_028.sh CODE_ROOT compare   compare alone (both vLLM runs complete)
# CODE_ROOT: a new code snapshot (main-<sha>) holding the amendment-028 doc. Jobs (account brics.u6uv, workq, normal,
# names unknownsize-028-*): rows (CPU 72 x 8 h, isambard_021_train.sbatch rows J27) -> smoke (4 GPUs x 1.5 h; exits 1
# on SMOKE_STOP, which cancels everything after it) -afterok-> smoke-vllm (1 GPU x 1 h) -afterok-> train segment 1
# (4 GPUs x 7 h) -afternotok-> train segment 2 (resumes from ckpt_latest; cancelled once segment 1 succeeds) ->
# greedy dev-select (1 GPU x 2 h) and test (1 GPU x 8 h), each afterok segment 1 OR segment 2 -> compare (CPU 72 x
# 6 h, isambard_021_eval.sbatch a028 compare; afterok both). Every dependency is --kill-on-invalid-dep=yes. Only the
# first job(s) are held; they are released once every job is in, so an ABORT (it lists the ids for scancel) leaves
# nothing running. Refused: outputs of the starting stage, any unknownsize-028 job in Slurm, and a GPU-h overrun.
# GPU cap (amendment 028): 80 GPU-h. The GPU-h of this user's unknownsize-028* jobs since 2026-10-15 (sacct: GPUs x
# elapsed once ended, x the time limit while queued or running, queued ones also by id) plus this submission's Slurm
# ceiling (GPUs x time limit: smoke 6 + smoke-vllm 1 + training 2 x 4 x 7 = 56 + greedy 2 + 8) must stay <= 80.
set -euo pipefail
test "$#" -eq 2 || exit 2
CODE_ROOT="$1"
MODE="$2"
case "$MODE" in
  all) START=0 ;; smoke) START=1 ;; smoke-vllm) START=2 ;; train) START=3 ;; infer) START=4 ;; compare) START=5 ;; *) exit 2 ;;
esac
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
A21="$RUN_ROOT/a021"
A28="$RUN_ROOT/a028"
S="$CODE_ROOT/fastfill/comparison"
CAP=80 SEGMENT_HOURS=7  # Arm A trained N = 40,000 in 5.1 h; two 7 h segments leave room for one node failure.
hours() { if test "$1" = test; then echo 8; else echo 2; fi; }  # a021_select.VLLM_HOURS['27B'].
for name in a028.py a021_train.py a021_select.py output_repair.py isambard_021_train.sbatch isambard_021_eval.sbatch; do
  test -f "$S/$name" || { echo "ABORT: missing $S/$name; nothing submitted" >&2; exit 1; }
done
test -f "$CODE_ROOT/fastfill/v1_1/train_dpo.py" || { echo "ABORT: missing train_dpo.py; nothing submitted" >&2; exit 1; }
DOC=""
for path in "$CODE_ROOT"/docs/research/20261013-unknown-size-comparison/amendment-028-*.md; do
  if test -f "$path"; then DOC="$path"; fi
done
test -n "$DOC" || { echo "ABORT: no amendment-028 protocol doc in $CODE_ROOT; nothing submitted" >&2; exit 1; }
need() {  # need PATH...: every input must exist.
  local path
  for path in "$@"; do test -e "$path" || { echo "ABORT: missing input $path; nothing submitted" >&2; exit 1; }; done
}
refuse() {  # refuse PATH...: no output of the starting stage may exist yet.
  local path
  for path in "$@"; do if test -e "$path"; then echo "refusing to reuse existing $path" >&2; exit 1; fi; done
}
need "$RUN_ROOT/eval-data-v1/MANIFEST.json" "$RUN_ROOT/dev-select-v1/MANIFEST.json" "$PROJECT_ROOT/models/Qwen3.8-27B/config.json" \
  "$A21/test-geometry-arm/MANIFEST.json" "$A21/27b/rows/MANIFEST.json"  # compare's 27B-P rows and Arm A's rooms.
if test "$START" -le 3; then
  compgen -G "$PROJECT_ROOT/pylib-021/.a021-fla-core-*" > /dev/null || {
    echo "ABORT: no fla overlay $PROJECT_ROOT/pylib-021 (submit_021_train.sh setup); nothing submitted" >&2; exit 1; }
fi
case "$MODE" in
  all) need "$RUN_ROOT/main-data-v1/MANIFEST.json" "$RUN_ROOT/sft-joint-001/SHA256.json" \
         "$RUN_ROOT/sft-placement-001/SHA256.json" "$RUN_ROOT/dev-full-v1/MANIFEST.json"
       refuse "$A28/rows" "$A28/smoke" "$A28/27b-joint" "$A28/vllm" "$A28/compare" ;;
  smoke) need "$A28/rows/MANIFEST.json"
         refuse "$A28/smoke" "$A28/27b-joint" "$A28/vllm" "$A28/compare" ;;
  smoke-vllm) "$PYTHON" -B -c 'import json, sys; assert json.load(open(sys.argv[1]))["status"] == "SMOKE_PASS"' \
                "$A28/smoke/verdict.json" 2>/dev/null || {
                echo "ABORT: no SMOKE_PASS verdict in $A28/smoke; nothing submitted" >&2; exit 1; }
              need "$A28/smoke/train/final/adapter_model.safetensors"
              refuse "$A28/smoke/vllm-load.json" "$A28/27b-joint" "$A28/vllm" "$A28/compare" ;;
  train) "$PYTHON" -B - "$A28/smoke" <<'PY' || { echo "ABORT: no SMOKE_PASS + vllm-load.json OK in $A28/smoke; nothing submitted" >&2; exit 1; }
import json, sys
assert json.load(open(sys.argv[1] + '/verdict.json'))['status'] == 'SMOKE_PASS'
assert json.load(open(sys.argv[1] + '/vllm-load.json'))['status'] == 'OK'
PY
         refuse "$A28/27b-joint/a021-policy.json" "$A28/vllm" "$A28/compare" ;;
  infer) need "$A28/27b-joint/a021-policy.json"
         refuse "$A28/compare" ;;
  compare) need "$A28/vllm/dev-select/j27/MANIFEST.json" "$A28/vllm/test/j27/MANIFEST.json"
           refuse "$A28/compare" ;;
esac
SLICES=()
if test "$START" -le 4; then
  for slice in dev-select test; do
    if ! test -f "$A28/vllm/$slice/j27/MANIFEST.json"; then SLICES+=("$slice"); fi
  done
fi
CEILING=0
if test "$START" -le 1; then CEILING=$((CEILING + 4 * 3 / 2)); fi
if test "$START" -le 2; then CEILING=$((CEILING + 1)); fi
if test "$START" -le 3; then CEILING=$((CEILING + 2 * 4 * SEGMENT_HOURS)); fi
for slice in ${SLICES[@]+"${SLICES[@]}"}; do CEILING=$((CEILING + $(hours "$slice"))); done
QUEUED=$(squeue -h -u "$USER" -o '%i|%j' | awk -F'|' '$2 ~ /^unknownsize-028/ { print $1 }') || {
  echo "ABORT: squeue failed; nothing submitted" >&2; exit 1; }
if test -n "$QUEUED"; then
  echo "ABORT: unknownsize-028 jobs already in Slurm; nothing submitted. cancel with: scancel" $QUEUED >&2
  exit 1
fi
used() {  # GPU-h of this user's unknownsize-028* jobs, counted as submit_021_*.sh count theirs.
  local fields=JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES ids
  ids=$(squeue -h -u "$USER" -o '%i|%j' | awk -F'|' '$2 ~ /^unknownsize-028/ { print $1 }' | paste -sd, -) || return 1
  { sacct -n -X -P -u "$USER" -S 2026-10-15 -o "$fields" || exit 1
    if test -n "$ids"; then sacct -n -X -P -j "$ids" -o "$fields" || exit 1; fi; } | awk -F'|' '
    $2 ~ /^unknownsize-028/ && !seen[$1]++ {
      tres = $6 != "" ? $6 : $7; g = 0
      if (match(tres, /gres\/gpu[^=,]*=[0-9]+/)) { s = substr(tres, RSTART, RLENGTH); sub(/.*=/, "", s); g = s + 0 }
      split($3, state, " ")
      t += (state[1] ~ /^(PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|CONFIGURING|COMPLETING)$/ ? $5 * 60 : $4) * g }
    END { printf "%.2f\n", t / 3600 }'
}
SPENT=$(used) || { echo "ABORT: sacct failed; nothing submitted" >&2; exit 1; }
awk -v u="$SPENT" -v c="$CEILING" -v m="$CAP" 'BEGIN { exit !(u + c <= m) }' || {
  echo "ABORT: GPU cap: used $SPENT + ceiling $CEILING > $CAP GPU-h; nothing submitted" >&2; exit 1; }
echo "GPU-h: used $SPENT + ceiling $CEILING <= $CAP"
TRAIN_SB="$S/isambard_021_train.sbatch"
EVAL_SB="$S/isambard_021_eval.sbatch"
GPU4="--gpus-per-node=4 --cpus-per-task=72 --mem=0"
GPU1="--gpus-per-node=1 --cpus-per-task=16"
SUBMITTED="" ROOTS=""
submit() {  # submit NAME DEPENDENCY "RESOURCES" SCRIPT ARGS...; sets ID; no dependency = a held root.
  local out="" res
  read -r -a res <<< "$3"
  local common=(--parsable --account=brics.u6uv --partition=workq --qos=normal --job-name="unknownsize-028-$1"
                --output="$PROJECT_ROOT/runs/isambard/unknownsize-028-$1-%j.out" ${res[@]+"${res[@]}"})
  if test -n "$2"; then
    out=$(sbatch "${common[@]}" --kill-on-invalid-dep=yes --dependency="$2" "$4" "$CODE_ROOT" "${@:5}") || out=""
  else
    out=$(sbatch "${common[@]}" --hold "$4" "$CODE_ROOT" "${@:5}") || out=""
  fi
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$1'; NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  if test -z "$2"; then ROOTS="$ROOTS${ROOTS:+,}$ID"; fi
  echo "$1 $ID"
}
DEP=""
if test "$START" -le 0; then submit rows "" "" "$TRAIN_SB" rows J27; DEP="afterok:$ID"; fi
if test "$START" -le 1; then submit smoke "$DEP" "$GPU4 --time=01:30:00" "$TRAIN_SB" smoke J27; DEP="afterok:$ID"; fi
if test "$START" -le 2; then
  submit smoke-vllm "$DEP" "$GPU1 --mem=200G --time=01:00:00" "$TRAIN_SB" smoke-vllm J27
  DEP="afterok:$ID"
fi
if test "$START" -le 3; then
  submit train "$DEP" "$GPU4 --time=$SEGMENT_HOURS:00:00" "$TRAIN_SB" train J27 "$SEGMENT_HOURS"
  FIRST="$ID"
  # Upstream cancellation must never bypass the successful smoke-vLLM gate.
  RETRY_DEP="${DEP:+$DEP,}afternotok:$FIRST"
  submit train "$RETRY_DEP" "$GPU4 --time=$SEGMENT_HOURS:00:00" "$TRAIN_SB" train J27 "$SEGMENT_HOURS"
  DEP="afterok:$FIRST?afterok:$ID"  # Either segment finishing OK means the policy exists.
fi
INFERRED=""
for slice in ${SLICES[@]+"${SLICES[@]}"}; do
  submit "infer-$slice" "$DEP" "$GPU1 --mem=100G --time=$(printf %02d "$(hours "$slice")"):00:00" "$EVAL_SB" a028 greedy \
    "$slice" "$(hours "$slice")"
  INFERRED="$INFERRED:$ID"
done
submit compare "${INFERRED:+afterok$INFERRED}" "--cpus-per-task=72 --mem=200G --time=06:00:00" "$EVAL_SB" a028 compare 6
scontrol release "$ROOTS" || { echo "ABORT: could not release $ROOTS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOTS"
