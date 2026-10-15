#!/bin/bash
# Amendment 022 (fastfill.system; RUN/system-022), run by the user on the Isambard login node:
#   submit_022.sh CODE_ROOT interim    package the system with R = rpo-dpo1, labelled interim (CPU; no GPU, no API)
#                                      -> RUN/system-022/interim/package
#   submit_022.sh CODE_ROOT cohort     §2 cohort (1,454 rooms), system with API access: [s1] -> system (2 GPUs, 6 h;
#                                      gate, gpt-6.1-sol <= 600 rooms x 2 calls, then O2) -> summarize
#   submit_022.sh CODE_ROOT test       §2 all 8,647 test rooms, O2 escalation only: [s1] -> system (2 GPUs, 12 h) ->
#                                      summarize
#   submit_022.sh CODE_ROOT package    the final package (CPU) from whatever §2 evaluations exist ('not measured' else)
#   submit_022.sh CODE_ROOT test-sharded [N]   fallback for `test` (its own addendum; N default 4): [s1] -> plan (CPU;
#                                      reuses a complete plan of the same N) -> shard 0..N-1 (2 GPUs, 12 h each; only
#                                      shards without COMPLETE.json, which resume per chunk) -> finalize (CPU) ->
#                                      summarize, all under RUN/system-022/final/test-sharded (never .../test).
#                                      Resubmit the same command after a killed shard; refuses while any
#                                      unknownsize-022-test-sharded* job is in Slurm.
# cohort / test / package need RUN/a021/final-selection.json (amendment 021 §4). [s1] = output_repair on the delivered
# W's amendment-021 test rows, submitted only when W != R and RUN/s1-022/placement-W is missing (a queued
# unknownsize-022-s1 job is reused as the dependency). The first job of the chain is a held root and every later job
# waits on it (afterok, --kill-on-invalid-dep=yes); the root is released only once the whole chain is in, so an ABORT
# (it lists the ids for scancel) leaves nothing that ran. Caps: cohort API (the gate is previewed here when the S1 rows
# exist and is applied again in the job before any call: <= 600 escalated rooms, the calls left of 1,200, LLM-agent
# total <= 7,700); GPU ceiling printed (cohort 12 GPU-h, test 24 GPU-h). Refuses when a job of the same step is in Slurm,
# when an output exists, or when CODE_ROOT lacks the amendment-022 code.
set -euo pipefail
usage() { echo "usage: submit_022.sh CODE_ROOT interim|cohort|test|package|test-sharded [N]" >&2; exit 2; }
test "$#" -eq 2 -o "$#" -eq 3 || usage
CODE_ROOT="$1"
MODE="$2"
SHARDS="${3:-4}"
case "$MODE" in interim|cohort|test|package) test "$#" -eq 2 || usage ;; test-sharded) ;; *) usage ;; esac
case "$SHARDS" in ''|*[!0-9]*) usage ;; esac
test "$SHARDS" -ge 1 -a "$SHARDS" -le 16 || usage
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
S="$CODE_ROOT/fastfill/system"
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
for name in pipeline.py cli.py evaluate.py package.py isambard_022.sbatch; do
  test -f "$S/$name" || fail "missing $S/$name"
done
test -f "$CODE_ROOT/docs/research/20261013-unknown-size-comparison/amendment-022-delivered-system-20261014.md" \
  || fail "no amendment-022 protocol doc in $CODE_ROOT"
for f in eval-data-v1/MANIFEST.json main-data-v1/size-priors.json llm-cohort-v1.json; do
  test -f "$RUN_ROOT/$f" || fail "missing $RUN_ROOT/$f"
done
py() { (cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B "$@"); }
case "$MODE" in
  interim) PMODE=interim ;;
  *) PMODE=final; test -f "$RUN_ROOT/a021/final-selection.json" || fail "amendment 021 §4 has not selected (no final-selection.json)" ;;
esac
OUT="$RUN_ROOT/system-022/$PMODE"
case "$MODE" in
  interim|package) test ! -e "$OUT/package" || fail "$OUT/package exists" ;;
  *) for path in "$OUT/$MODE/system" "$OUT/$MODE/eval"; do test ! -e "$path" || fail "$path exists"; done ;;
esac
queued() { squeue -h -u "$USER" -n "unknownsize-022-$1" -o %i; }  # set -e misses $(...): callers check it.
case "$MODE" in interim|package) STEP=package ;; *) STEP="$MODE" ;; esac  # interim runs as job 'package'
QUEUED=$(queued "$STEP") || fail "squeue failed"
test -z "$QUEUED" || fail "unknownsize-022-$STEP jobs already in Slurm (scancel $QUEUED)"
if test "$MODE" = test-sharded; then
  QUEUED=$(squeue -h -u "$USER" -o '%i %j' | awk '$2 ~ /^unknownsize-022-test-sharded/ {print $1}') || fail "squeue failed"
  test -z "$QUEUED" || fail "unknownsize-022-test-sharded* jobs already in Slurm (scancel $QUEUED)"
  T="$OUT/test-sharded"
  if test -e "$T/plan/plan.json"; then
    N=$(py -c 'import json, sys; print(json.load(open(sys.argv[1]))["shards"])' "$T/plan/plan.json") \
      || fail "cannot read $T/plan/plan.json"
    test "$N" = "$SHARDS" || fail "$T/plan was made for $N shards, not $SHARDS"
  elif test -e "$T/plan"; then
    fail "$T/plan exists without plan.json (an interrupted plan; inspect it, move it aside)"
  fi
fi
S1_DIR=$(py -c 'import sys; from fastfill.system import evaluate as ev; print(ev.placer(sys.argv[1], sys.argv[2])["s1"])' \
  "$RUN_ROOT" "$PMODE") || fail "cannot resolve the delivered placer"
SUBMITTED=""
ROOT=""
submit() {  # submit LABEL DEPENDENCY SBATCH_ARGS...; sets ID; the first job is the held root. set -e misses $(...).
  local out="" dep=()
  if test -n "$2"; then dep=(--kill-on-invalid-dep=yes --dependency="afterok:$2"); fi
  if test -z "$ROOT"; then dep+=(--hold); fi
  out=$(sbatch --parsable --job-name="unknownsize-022-$1" "${dep[@]}" "${@:3}") || out=""
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$1'; chain NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  if test -z "$ROOT"; then ROOT="$ID"; fi
  echo "$1 $ID"
}
release() {
  scontrol release "$ROOT" || { echo "ABORT: could not release $ROOT; cancel with: scancel $SUBMITTED" >&2; exit 1; }
  echo "released $ROOT"
}
J=("$S/isambard_022.sbatch" "$CODE_ROOT")
case "$MODE" in
  interim|package)
    submit package "" "${J[@]}" package "$PMODE"
    release
    exit 0 ;;
esac
DEP=""
if test ! -e "$S1_DIR/MANIFEST.json"; then
  test ! -e "$S1_DIR" || fail "$S1_DIR exists without MANIFEST.json (an interrupted S1 run; inspect it)"
  DEP=$(queued s1) || fail "squeue failed"
  if test -z "$DEP"; then
    submit s1 "" --cpus-per-task=72 --mem=200G --time=04:00:00 "${J[@]}" s1 final
    DEP="$ID"
  else
    echo "reusing queued S1 job $DEP"
  fi
elif test "$MODE" = cohort; then
  py -m fastfill.system.evaluate gate --run-root "$RUN_ROOT" --mode final --scope cohort \
    || fail "the amendment-022 API gate refuses (see the JSON above)"
fi
if test "$MODE" = test-sharded; then
  submit test-sharded-plan "$DEP" --cpus-per-task=72 --mem=200G --time=04:00:00 "${J[@]}" plan final "$SHARDS"
  PLAN="$ID" AFTER="$ID" TODO=0
  for K in $(seq 0 $((SHARDS - 1))); do
    if test -e "$T/shards/shard-$K/COMPLETE.json"; then echo "shard $K complete; not resubmitted"; continue; fi
    submit "test-sharded-shard$K" "$PLAN" --gpus-per-node=2 --cpus-per-task=72 --mem=200G --time=12:00:00 \
      "${J[@]}" shard final "$K" 12
    AFTER="$AFTER:$ID" TODO=$((TODO + 1))
  done
  echo "GPU ceiling: $TODO shards x 2 GPUs x 12 h = $((TODO * 24)) GPU-h"
  submit test-sharded-finalize "$AFTER" --cpus-per-task=72 --mem=200G --time=04:00:00 "${J[@]}" finalize final
  submit test-sharded-summarize "$ID" --time=06:00:00 "${J[@]}" summarize final test-sharded
  release
  exit 0
fi
if test "$MODE" = cohort; then HOURS=6; else HOURS=12; fi
echo "GPU ceiling: 2 GPUs x $HOURS h = $((2 * HOURS)) GPU-h"
submit "$MODE" "$DEP" --gpus-per-node=2 --cpus-per-task=72 --mem=200G --time="$(printf '%02d:00:00' "$HOURS")" \
  "${J[@]}" system final "$MODE" "$HOURS"
submit "$MODE-summarize" "$ID" --time=06:00:00 "${J[@]}" summarize final "$MODE"
if test -n "$ROOT"; then release; fi
