#!/bin/bash
# Amendment 021 sections 1.4, 3, 4 (fastfill.comparison.a021_select, RUN/a021), run by the parent on the Isambard
# login node, one stage at a time:
#   submit_021_eval.sh CODE_ROOT confirm      CPU: confirm slice (3,000 dev rooms), engine slice (its first 1,000 in hash
#                                             order), --all-rooms cohorts -> then the HF plans of engine and confirm
#   submit_021_eval.sh CODE_ROOT o2nt         O2nt (2 GPUs each) on dev-select, confirm and the 1,454 cohort -> o2nt-score
#   submit_021_eval.sh CODE_ROOT engine       R and P-SFT on the engine slice: HF exact shards + merges and vLLM greedy
#                                             -> engine-check (common vLLM iff R's |HF - vLLM| checker gap <= 0.015)
#   submit_021_eval.sh CODE_ROOT candidates A|B
#                                             fixes a021/candidates.json (R, 013 / 018 adopted models from their
#                                             selection.json, the 021 arm: A = RUN/a021/27b, B = RUN/a021/8b fallback,
#                                             a021_train's policy dirs, loaded --exact-policy);
#                                             dev-select runs on each serving engine, R twice on vLLM (spread), the vLLM
#                                             confirm runs -> select (final-selection.json unless an HF winner needs
#                                             its confirm runs)
#   submit_021_eval.sh CODE_ROOT select       only for an HF-served dev winner: its and R's HF confirm runs -> select
#   submit_021_eval.sh CODE_ROOT test         the delivered winner (and R on vLLM) on all test rooms -> test-geometry
# The GPU jobs of a stage come from 'a021_select jobs STAGE' (finished runs left out, partial vLLM runs resume, existing
# HF shard dirs get --resume). Every GPU job is a held root and the CPU step waits on all of them (afterok,
# --kill-on-invalid-dep=yes); the roots are released only once the whole stage is in, so an ABORT (it lists the ids for
# scancel) leaves nothing that ran. GPU cap (amendment 021 section 5): 75 GPU-h for the whole amendment. Before
# submitting, the GPU-h of this user's unknownsize-021* jobs since 2026-10-14 (sacct: GPUs x elapsed once ended, x the
# time limit while queued or running, queued ones also by id; the arm's training jobs must carry that prefix) plus the
# Slurm ceiling of this stage (GPUs x time limit) must
# stay <= 75, else nothing is submitted. Refuses while any unknownsize-021-eval job of this user is in Slurm, and
# refuses a stage whose output exists.
set -euo pipefail
test "$#" -ge 2 || exit 2
CODE_ROOT="$1"
MODE="$2"
shift 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
A21="$RUN_ROOT/a021"
S="$CODE_ROOT/fastfill/comparison"
CAP=200  # amendment 021 addendum A
case "$MODE" in
  confirm|o2nt|engine|select|test) test "$#" -eq 0 || exit 2 ;;
  candidates) test "$#" -eq 1 || exit 2; case "$1" in A) ARM_DIR=27b ;; B) ARM_DIR=8b ;; *) exit 2 ;; esac ;;
  *) exit 2 ;;
esac
for name in a021_select.py llm_open.py sample_vllm.py infer_shard.py isambard_021_eval.sbatch; do
  test -f "$S/$name" || { echo "ABORT: missing $S/$name; nothing submitted" >&2; exit 1; }
done
DOC=""
for path in "$CODE_ROOT"/docs/research/20261013-unknown-size-comparison/amendment-021-*.md; do
  if test -f "$path"; then DOC="$path"; fi
done
test -n "$DOC" || { echo "ABORT: no amendment-021 protocol doc in $CODE_ROOT; nothing submitted" >&2; exit 1; }
need() {  # need PATH...: every input must exist.
  local path
  for path in "$@"; do test -e "$path" || { echo "ABORT: missing input $path; nothing submitted" >&2; exit 1; }; done
}
refuse() {  # refuse PATH...: no output of this stage may exist yet.
  local path
  for path in "$@"; do if test -e "$path"; then echo "refusing to reuse existing $path" >&2; exit 1; fi; done
}
case "$MODE" in
  confirm) need "$RUN_ROOT/main-data-v1/MANIFEST.json" "$RUN_ROOT/sft-joint-001/SHA256.json" \
             "$RUN_ROOT/sft-placement-001/SHA256.json" "$RUN_ROOT/dev-select-v1/MANIFEST.json" "$PROJECT_ROOT/models/Qwen3-8B"
           refuse "$A21/confirm-v1" "$A21/engine-v1" "$A21/cohorts" "$A21/hf/engine/plan" "$A21/hf/confirm/plan" ;;
  o2nt) need "$A21/cohorts/MANIFEST.json" "$A21/confirm-v1/MANIFEST.json" "$RUN_ROOT/llm-cohort-v1.json" \
          "$PROJECT_ROOT/models/Qwen3.8-27B"
        refuse "$A21/o2nt-scores" ;;
  engine) need "$A21/engine-v1/MANIFEST.json" "$A21/hf/engine/plan/MANIFEST.json" "$RUN_ROOT/context-capacity-001/SHA256.json" \
            "$RUN_ROOT/dpo-003/placement/dpo1-merged.json" "$RUN_ROOT/sft-placement-001/SHA256.json"
          refuse "$A21/engine-check" ;;
  candidates) need "$A21/engine-check/MANIFEST.json" "$A21/confirm-v1/MANIFEST.json" "$A21/$ARM_DIR/a021-policy.json" \
                "$RUN_ROOT/dev-eval-013/selection.json" "$RUN_ROOT/dev-eval-018/selection.json"
              refuse "$A21/dev-selection.json" "$A21/final-selection.json" ;;
  select) need "$A21/dev-selection.json" "$A21/hf/confirm/plan/MANIFEST.json"
          refuse "$A21/final-selection.json" ;;
  test) need "$A21/final-selection.json"
        refuse "$A21"/test-geometry-* ;;
esac
QUEUED=$(squeue -h -u "$USER" -n unknownsize-021-eval -o %i) || { echo "ABORT: squeue failed; nothing submitted" >&2; exit 1; }
if test -n "$QUEUED"; then
  echo "ABORT: amendment-021 evaluation jobs already in Slurm; nothing submitted. cancel with: scancel" $QUEUED >&2
  exit 1
fi
used() {  # GPU-h of this user's unknownsize-021* jobs (placers_020.gpu_hours): GPUs (allocated, else requested) x
  # elapsed once ended, x the time limit while waiting or running. Queued ones are added by id: a job pending on a
  # dependency has no eligible time, so the -S window misses it.
  local fields=JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES ids
  ids=$(squeue -h -u "$USER" -o '%i %j' | awk '$2 ~ /^unknownsize-021/ { print $1 }' | paste -sd, -) || return 1
  { sacct -n -X -P -u "$USER" -S 2026-10-14 -o "$fields" || exit 1  # A pipeline subshell: pipefail carries it.
    if test -n "$ids"; then sacct -n -X -P -j "$ids" -o "$fields" || exit 1; fi; } | awk -F'|' '
    $2 ~ /^unknownsize-021/ && !seen[$1]++ {
      tres = $6 != "" ? $6 : $7; g = 0
      if (match(tres, /gres\/gpu[^=,]*=[0-9]+/)) { s = substr(tres, RSTART, RLENGTH); sub(/.*=/, "", s); g = s + 0 }
      split($3, state, " ")
      t += (state[1] ~ /^(PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|CONFIGURING|COMPLETING)$/ ? $5 * 60 : $4) * g }
    END { printf "%.2f\n", t / 3600 }'
}
cap() {  # cap CEILING: abort unless used + CEILING <= CAP.
  local spent
  spent=$(used) || { echo "ABORT: sacct failed; nothing submitted" >&2; exit 1; }
  awk -v u="$spent" -v c="$1" -v m="$CAP" 'BEGIN { exit !(u + c <= m) }' || {
    echo "ABORT: GPU cap: used $spent + ceiling $1 > $CAP GPU-h; nothing submitted" >&2; exit 1; }
  echo "GPU-h: used $spent + ceiling $1 <= $CAP"
}
SUBMITTED=""
ROOTS=""
submit() {  # submit LABEL DEPENDENCY SBATCH_ARGS...; sets ID; no dependency = a held root. set -e misses $(...).
  local out=""
  if test -n "$2"; then
    out=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$2" "${@:3}") || out=""
  else
    out=$(sbatch --parsable --hold "${@:3}") || out=""
  fi
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$1'; stage NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  if test -z "$2"; then ROOTS="$ROOTS${ROOTS:+,}$ID"; fi
  echo "$1 $ID"
}
release() {
  scontrol release "$ROOTS" || { echo "ABORT: could not release $ROOTS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
  echo "released $ROOTS"
}
J=("$S/isambard_021_eval.sbatch" "$CODE_ROOT")
if test "$MODE" = confirm; then
  cap 0
  submit confirm "" "${J[@]}" confirm
  CONFIRM="$ID"
  submit "plan engine" "$CONFIRM" "${J[@]}" plan engine
  submit "plan confirm" "$CONFIRM" "${J[@]}" plan confirm
  release
  exit 0
fi
case "$MODE" in
  candidates) STAGE=candidates FINAL=select EXTRA=(--arm-kind "$1") ;;
  select) STAGE=confirm FINAL=select EXTRA=() ;;
  o2nt) STAGE=o2nt FINAL=o2nt-score EXTRA=() ;;
  engine) STAGE=engine FINAL=engine-check EXTRA=() ;;
  test) STAGE=test FINAL=test-geometry EXTRA=() ;;
esac
LINES=$(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -B -m fastfill.comparison.a021_select \
  jobs "$STAGE" --run-root "$RUN_ROOT" ${EXTRA[@]+"${EXTRA[@]}"}) || {
  echo "ABORT: a021_select jobs $STAGE failed; nothing submitted" >&2; exit 1; }
CEILING=$(printf '%s\n' "$LINES" | awk '$1 == "ceiling" { print $2 }')
test -n "$CEILING" || { echo "ABORT: no ceiling from a021_select jobs; nothing submitted" >&2; exit 1; }
cap "$CEILING"
time_of() { printf -- '--time=%02d:00:00' "$1"; }
DEPS=""
SHARDS=""
while read -r KIND A B C D E F; do
  case "$KIND" in
    vllm) submit "vllm $A $B $C" "" --gpus-per-node=1 --cpus-per-task=16 --mem=100G "$(time_of "$D")" \
            "${J[@]}" vllm "$A" "$B" "$C" "$D"
          DEPS="$DEPS${DEPS:+:}$ID" ;;
    hf) submit "hf $A $B $C/$D" "" --gpus-per-node=1 --cpus-per-task=8 --mem=96G "$(time_of "$E")" \
          "${J[@]}" hf "$A" "$B" "$C" "$D" "$E" ${F:+"$F"}
        SHARDS="$SHARDS${SHARDS:+:}$ID" ;;
    hf-merge) submit "hf-merge $A $B" "$SHARDS" "${J[@]}" hf-merge "$A" "$B" "$C"
              SHARDS=""
              DEPS="$DEPS${DEPS:+:}$ID" ;;
    o2nt) submit "o2nt $A" "" --gpus-per-node=2 --cpus-per-task=32 --mem=200G "$(time_of "$B")" "${J[@]}" o2nt "$A" "$B"
          DEPS="$DEPS${DEPS:+:}$ID" ;;
    ceiling) ;;
    *) echo "ABORT: unknown job line '$KIND $A'; stage NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
       if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
       exit 1 ;;
  esac
done <<< "$LINES"
if test "$FINAL" = test-geometry; then
  submit "$FINAL" "$DEPS" --time=06:00:00 "${J[@]}" "$FINAL"
else
  submit "$FINAL" "$DEPS" "${J[@]}" "$FINAL"
fi
release
