#!/bin/bash
# Amendment 021 section 2 jobs (isambard_021_train.sbatch), run by the parent on the Isambard login node:
#   submit_021_train.sh CODE_ROOT setup | smoke | smoke-vllm | train [SEGMENTS [HOURS]] | probe | armB [SEGMENTS [HOURS]]
# CODE_ROOT: a new code snapshot (main-<sha>) holding the amendment-021 doc and the lineage evidence JSON
# (evidence/full-audit-data-lineage-7368003-result.json, sha256-pinned in a021_train). The first job of a submission is
# held and released only once every job is in; an ABORT lists the submitted ids for scancel.
#   setup   CPU 72 x 8 h: the fla overlay PROJECT/pylib-021 (once) and RUN/a021/pool. Refused when both exist.
#   smoke   4 GPUs x 1.5 h, then smoke-vllm (afterok: only after SMOKE_PASS): needs the pool and the overlay; refused
#           once RUN/a021/smoke exists (a crashed smoke renames itself smoke.failed-JOB; one killed by its time limit
#           stays and must be moved aside by hand).
#   smoke-vllm  1 GPU x 1 h, alone (a rerun): needs SMOKE_PASS and the smoke adapter; refused once
#           RUN/a021/smoke/vllm-load.json exists.
#   train   SEGMENTS (1-3, default 1) x 4 GPUs x H h, H = HOURS (1-12) if given, else min(12, ceil(1.15 x the smoke's
#           projected hours for N + 1)); needs SMOKE_PASS and RUN/a021/smoke/vllm-load.json = OK. Segment k+1 starts
#           only if segment k did not finish OK (afternotok; Slurm cancels it once one succeeds) and resumes from
#           ckpt_latest; a later resubmission resumes too, so HOURS sizes it to the cap. Refused once the Arm A policy
#           (RUN/a021/27b/a021-policy.json) exists.
#   probe   1 GPU x 3 h: needs the Arm A policy; refused once RUN/a021/probe/report.json exists.
#   armB    the fallback only: SEGMENTS x 4 GPUs x HOURS (default 12) h, the 8B on every admitted room; refused while
#           Arm A has a policy or once Arm B has one.
# GPU cap (amendment 021 section 5): 75 GPU-h. The GPU-h of this user's unknownsize-021* jobs since 2026-10-14 (sacct:
# elapsed x GPUs once ended; time limit x GPUs while pending or running, also for jobs that squeue lists but the sacct
# window misses, e.g. pending on a dependency) plus the Slurm ceiling (GPUs x time limit) of this submission must
# stay <= 75, else nothing is submitted. Refused while any unknownsize-021-train job of this user is in Slurm.
set -euo pipefail
test "$#" -ge 2 -a "$#" -le 4 || exit 2
CODE_ROOT="$1"
MODE="$2"
SEGMENTS="${3:-1}"
HOURS="${4:-}"
case "$MODE" in
  train|armB) ;;
  setup|smoke|smoke-vllm|probe) test "$#" -eq 2 || exit 2 ;;
  *) exit 2 ;;
esac
case "$SEGMENTS" in ''|*[!0-9]*) exit 2 ;; esac
test "$SEGMENTS" -ge 1 -a "$SEGMENTS" -le 3 || exit 2
if test -n "$HOURS"; then
  case "$HOURS" in *[!0-9]*) exit 2 ;; esac
  test "$HOURS" -ge 1 -a "$HOURS" -le 12 || exit 2
  HOURS=$((10#$HOURS))
fi
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
A21="$RUN_ROOT/a021"
S="$CODE_ROOT/fastfill/comparison"
CAP=200  # amendment 021 addendum A
for name in a021_train.py dpo_stages.py sample_vllm.py infer.py main_data.py isambard_021_train.sbatch; do
  test -f "$S/$name" || { echo "ABORT: missing $S/$name; nothing submitted" >&2; exit 1; }
done
test -f "$CODE_ROOT/fastfill/v1_1/train_dpo.py" || { echo "ABORT: missing train_dpo.py; nothing submitted" >&2; exit 1; }
DOCS="$CODE_ROOT/docs/research/20261013-unknown-size-comparison"
DOC=""
for path in "$DOCS"/amendment-021-*.md; do if test -f "$path"; then DOC="$path"; fi; done
test -n "$DOC" || { echo "ABORT: no amendment-021 protocol doc in $CODE_ROOT; nothing submitted" >&2; exit 1; }
test -f "$DOCS/evidence/full-audit-data-lineage-7368003-result.json" || {
  echo "ABORT: no lineage evidence JSON in $CODE_ROOT; nothing submitted" >&2; exit 1; }
for path in main-data-v1/MANIFEST.json full-corpus-v2/MANIFEST.json dev-select-v1/MANIFEST.json \
    dev-full-v1/MANIFEST.json eval-data-v1/MANIFEST.json; do
  test -f "$RUN_ROOT/$path" || { echo "ABORT: missing input $RUN_ROOT/$path; nothing submitted" >&2; exit 1; }
done
for model in Qwen3.8-27B Qwen3-8B; do
  test -f "$PROJECT_ROOT/models/$model/config.json" || {
    echo "ABORT: missing $PROJECT_ROOT/models/$model; nothing submitted" >&2; exit 1; }
done
COLUMNS_021=JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES
used() {  # GPU-h of this user's unknownsize-021* jobs: elapsed once ended, the time limit while pending or running.
  local ids  # squeue's ids too: a job pending on a dependency has no eligible time, so the sacct -S window misses it.
  ids=$(squeue -h -u "$USER" -o '%i|%j' | awk -F'|' '$2 ~ /^unknownsize-021/ { printf "%s%s", c, $1; c = "," }') || return 1
  { sacct -n -X -P -u "$USER" -S 2026-10-14 -o "$COLUMNS_021" &&
      { test -z "$ids" || sacct -n -X -P -j "$ids" -o "$COLUMNS_021"; }; } | awk -F'|' '
    $2 ~ /^unknownsize-021/ && !seen[$1]++ {
      tres = $6 != "" ? $6 : $7; g = 0
      if (match(tres, /gres\/gpu[^=,]*=[0-9]+/)) { s = substr(tres, RSTART, RLENGTH); sub(/.*=/, "", s); g = s + 0 }
      split($3, state, " ")
      waiting = state[1] ~ /^(PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|CONFIGURING|COMPLETING)$/
      t += (waiting ? $5 * 60 : $4) * g }
    END { printf "%.2f\n", t / 3600 }'
}
POLICY_A="$A21/27b/a021-policy.json"
POLICY_B="$A21/8b/a021-policy.json"
OVERLAY_READY=""
if compgen -G "$PROJECT_ROOT/pylib-021/.a021-fla-core-*" > /dev/null; then OVERLAY_READY=1; fi
VLLM_RES="--gpus-per-node=1 --cpus-per-task=16 --mem=200G --time=01:00:00"
pass_verdict() {  # Prints min(12, ceil(1.15 x projected hours for N + 1)) of a SMOKE_PASS verdict, else fails.
  "$PYTHON" -B - "$A21/smoke/verdict.json" <<'PY'
import json, math, sys
verdict = json.load(open(sys.argv[1]))
assert verdict['status'] == 'SMOKE_PASS', verdict['reasons']
print(min(12, math.ceil(1.15 * verdict['projected_hours_n'] + 1)))
PY
}
case "$MODE" in
  setup)
    if test -n "$OVERLAY_READY" && test -f "$A21/pool/MANIFEST.json"; then
      echo "ABORT: the overlay and RUN/a021/pool exist; nothing to do" >&2; exit 1
    fi
    if test -e "$A21/pool" && ! test -f "$A21/pool/MANIFEST.json"; then
      echo "refusing to reuse existing $A21/pool (no MANIFEST.json; inspect it)" >&2; exit 1
    fi
    CEILING=0 JOBS=("setup||setup|") ;;
  smoke)
    test -f "$A21/pool/MANIFEST.json" -a -n "$OVERLAY_READY" || {
      echo "ABORT: smoke needs RUN/a021/pool and the fla overlay (submit setup); nothing submitted" >&2; exit 1; }
    if test -e "$A21/smoke/verdict.json"; then echo "refusing: the smoke has a verdict ($A21/smoke/verdict.json)" >&2; exit 1; fi
    if test -e "$A21/smoke"; then
      echo "refusing to reuse existing $A21/smoke (no verdict: an interrupted attempt; move it aside to rerun)" >&2; exit 1
    fi
    CEILING=7 JOBS=("smoke|--gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=01:30:00|smoke|"
                    "smoke-vllm|$VLLM_RES|smoke-vllm|afterok") ;;
  smoke-vllm)
    pass_verdict > /dev/null || { echo "ABORT: no SMOKE_PASS verdict in $A21/smoke; nothing submitted" >&2; exit 1; }
    test -f "$A21/smoke/train/final/adapter_model.safetensors" || {
      echo "ABORT: no smoke adapter in $A21/smoke/train/final; nothing submitted" >&2; exit 1; }
    if test -e "$A21/smoke/vllm-load.json"; then echo "refusing: $A21/smoke/vllm-load.json exists" >&2; exit 1; fi
    CEILING=1 JOBS=("smoke-vllm|$VLLM_RES|smoke-vllm|") ;;
  train)
    if test -e "$POLICY_A"; then echo "refusing: Arm A is trained ($POLICY_A)" >&2; exit 1; fi
    PROJECTED=$(pass_verdict) || { echo "ABORT: no SMOKE_PASS verdict in $A21/smoke; nothing submitted" >&2; exit 1; }
    "$PYTHON" -B -c 'import json, sys; assert json.load(open(sys.argv[1]))["status"] == "OK"' \
      "$A21/smoke/vllm-load.json" 2>/dev/null || {
      echo "ABORT: vLLM has not loaded the smoke adapter ($A21/smoke/vllm-load.json; submit smoke-vllm); nothing submitted" >&2
      exit 1; }
    HOURS="${HOURS:-$PROJECTED}"
    CEILING=$((SEGMENTS * 4 * HOURS)) JOBS=()
    for ((k = 0; k < SEGMENTS; k++)); do
      JOBS+=("train A segment $((k + 1))|--gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=$HOURS:00:00|train A $HOURS|afternotok")
    done ;;
  probe)
    test -f "$POLICY_A" || { echo "ABORT: probe needs the Arm A policy $POLICY_A; nothing submitted" >&2; exit 1; }
    if test -e "$A21/probe/report.json"; then echo "refusing to reuse existing $A21/probe/report.json" >&2; exit 1; fi
    CEILING=3 JOBS=("probe|--gpus-per-node=1 --cpus-per-task=16 --mem=200G --time=03:00:00|probe|") ;;
  armB)
    if test -e "$POLICY_A"; then echo "refusing: Arm A is trained ($POLICY_A); Arm B is the fallback only" >&2; exit 1; fi
    if test -e "$POLICY_B"; then echo "refusing: Arm B is trained ($POLICY_B)" >&2; exit 1; fi
    test -f "$A21/pool/MANIFEST.json" || { echo "ABORT: armB needs RUN/a021/pool; nothing submitted" >&2; exit 1; }
    HOURS="${HOURS:-12}"
    CEILING=$((SEGMENTS * 4 * HOURS)) JOBS=()
    for ((k = 0; k < SEGMENTS; k++)); do
      JOBS+=("train B segment $((k + 1))|--gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=$HOURS:00:00|train B $HOURS|afternotok")
    done ;;
esac
QUEUED=$(squeue -h -u "$USER" -n unknownsize-021-train -o %i) || { echo "ABORT: squeue failed; nothing submitted" >&2; exit 1; }
if test -n "$QUEUED"; then
  echo "ABORT: unknownsize-021-train jobs already in Slurm; nothing submitted. cancel with: scancel" $QUEUED >&2
  exit 1
fi
SPENT=$(used) || { echo "ABORT: sacct failed; nothing submitted" >&2; exit 1; }
awk -v u="$SPENT" -v c="$CEILING" -v m="$CAP" 'BEGIN { exit !(u + c <= m) }' || {
  echo "ABORT: GPU cap: used $SPENT + ceiling $CEILING > $CAP GPU-h; nothing submitted" >&2; exit 1; }
echo "GPU-h: used $SPENT + ceiling $CEILING <= $CAP"
SUBMITTED="" ROOT="" PREV=""
for job in "${JOBS[@]}"; do
  IFS='|' read -r LABEL RESOURCES ARGS DEP <<< "$job"
  read -r -a RES <<< "$RESOURCES"
  read -r -a MODE_ARGS <<< "$ARGS"
  if test -z "$PREV"; then  # The held root; each later job on the one before (afternotok: a segment; afterok: smoke-vllm).
    out=$(sbatch --parsable --hold ${RES[@]+"${RES[@]}"} "$S/isambard_021_train.sbatch" "$CODE_ROOT" "${MODE_ARGS[@]}") || out=""
  else
    out=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="$DEP:$PREV" ${RES[@]+"${RES[@]}"} \
          "$S/isambard_021_train.sbatch" "$CODE_ROOT" "${MODE_ARGS[@]}") || out=""
  fi
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$LABEL'; NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  if test -z "$ROOT"; then ROOT="$ID"; fi
  PREV="$ID"
  echo "$LABEL $ID"
done
scontrol release "$ROOT" || { echo "ABORT: could not release $ROOT; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOT"
