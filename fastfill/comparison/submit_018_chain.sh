#!/bin/bash
# Amendment 018 chain (sections 2-3: SFT-D -> DPO1-D -> DPO2-D for arm P, dev selection), run by the parent on the
# Isambard login node:
#   submit_018_chain.sh CODE_ROOT [FROM=sft-data] [TEACHER_JOB]
#   submit_018_chain.sh CODE_ROOT test
# CODE_ROOT must be a new code snapshot (main-<sha>) that holds the amendment-018 protocol doc. TEACHER_JOB: the
# teacher filter job (still in Slurm); sft-data then waits on it (afterok). Without it RUN/distill-018/teacher must
# hold its targets. RUN/distill-018/plan (distill_018 plan) must exist whenever T-C still has to be derived.
# All jobs run isambard_018.sbatch; dependencies are afterok with --kill-on-invalid-dep=yes and the root is held until
# the whole chain is in, so an ABORT (it lists the ids for scancel) leaves nothing that ran:
#   recheck (CPU, held root)          this code reproduces dev-eval-010's sft / rpo-dpo1 dev scores (before any GPU)
#   sft-data (CPU 72, 4 h)            <- recheck [+ TEACHER_JOB]: distill-018/sft-d-rows
#   tc-data (CPU)                     <- recheck: distill-018/t-c from the plan (with sample, unless t-c is complete)
#   train sft-d (4 GPUs, 6 h)         <- sft-data: SFT-D from rpo-dpo1's adapter
#   pairs-dpo1 (CPU 72, 4 h)          <- sft-data: distill-018/dpo1-d-pairs on T-A
#   train dpo1-d (4 GPUs, 6 h)        <- train sft-d, pairs-dpo1
#   sample (SAMPLES x 1 GPU, 2 h)     <- train dpo1-d, tc-data: DPO1-D on T-C, K = 9 per room -> distill-018/dpo2-d-samples
#   finalize (CPU)                    <- sample
#   pairs-dpo2 (CPU 72, 8 h)          <- finalize: onpolicy_dpo PAIR_RULE -> distill-018/dpo2-d-pairs
#   train dpo2-d (4 GPUs, 6 h)        <- pairs-dpo2
#   per CAND sft-d, dpo1-d, dpo2-d: 2 dev shards (1 GPU, 4 h) <- its train; dev merge (CPU) <- its shards
#   dev-score (CPU)                   <- the three dev merges
#   select (CPU)                      <- dev-score: selection.json + selection-report.md only (--no-submit)
# test: on the login node, select again without --no-submit; if a candidate was adopted it submits the test chain
# (6 test shards x 1 GPU x 12 h, merge, geometry-v7, test-report-018, S1 on test, merged-weight dev check).
# FROM (sft-data train-sft pairs-dpo1 train-dpo1 sample finalize pairs-dpo2 train-dpo2 dev dev-score select) restarts a
# failed chain: earlier steps are skipped (each must have its output), recheck stays the held root, sampling and
# training resume, existing dev shards get --resume, a candidate whose dev-eval-018/CAND/infer-placement/MANIFEST.json
# exists is skipped, and any other output of FROM or a later step refuses. If pairs-dpo2 recorded A018_DPO2_NOT_TRAINED
# (below the amendment-013 floors), a restart after it submits nothing for DPO2-D (no train, no dev shards) and needs
# neither its merge nor its dev run; in a full chain those jobs are submitted and exit 0 without work.
# GPU cap (amendment 018 section 5): 150 GPU-h. Before submitting, the GPU-h already used by this user's
# unknownsize-018* jobs since 2026-10-14 (sacct: elapsed x allocated GPUs, running jobs so far) plus the Slurm ceiling
# of what is about to be submitted (GPUs x time limit; full chain 104, test 80) must stay <= 150, else nothing is
# submitted. The teacher's GPU jobs (unknownsize-018-teacher, submit_018_teacher.sh) count with them.
# Refuses while any unknownsize-018 job of this user is in Slurm.
set -euo pipefail
test "$#" -ge 1 -a "$#" -le 3 || exit 2
CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
D18="$RUN_ROOT/distill-018"
EVAL="$RUN_ROOT/dev-eval-018"
S="$CODE_ROOT/fastfill/comparison"
CAP=170 SAMPLES=4 N=2 TEST_CEILING=80  # TEST_CEILING: 6 x 1 x 12 test shards + 2 x 1 x 4 merged dev shards.
CANDS=(sft-d dpo1-d dpo2-d)
for name in distill_018_chain.py onpolicy_dpo.py sample_vllm.py dpo_stages.py dev_select.py infer_shard.py \
    output_repair.py downstream.py isambard_018.sbatch isambard_infer_merge.sbatch; do
  test -f "$S/$name" || { echo "ABORT: missing $S/$name; nothing submitted" >&2; exit 1; }
done
DOC=""
for path in "$CODE_ROOT"/docs/research/20261013-unknown-size-comparison/amendment-018-*.md; do
  if test -f "$path"; then DOC="$path"; fi
done
test -n "$DOC" || { echo "ABORT: no amendment-018 protocol doc in $CODE_ROOT; nothing submitted" >&2; exit 1; }
for path in main-data-v1/MANIFEST.json sft-joint-001/SHA256.json sft-placement-001/SHA256.json \
    dpo-001/positives/MANIFEST.json dev-select-v1/MANIFEST.json dev-full-v1/MANIFEST.json eval-data-v1/MANIFEST.json \
    dpo-003/placement/dpo1-merged.json dpo-001/placement/sft-merged.json dpo-003/placement/dpo1/final/adapter_model.safetensors \
    dpo-004/train-subset/MANIFEST.json dpo-004/samples/MANIFEST.json dev-eval-010/plan/MANIFEST.json \
    dev-eval-010/scores/MANIFEST.json dev-eval-010/sft/infer-placement/MANIFEST.json \
    dev-eval-010/rpo-dpo1/infer-placement/MANIFEST.json context-capacity-001/SHA256.json \
    infer-v2-002/plan/MANIFEST.json infer-joint-002/MANIFEST.json infer-placement-005/MANIFEST.json; do
  test -f "$RUN_ROOT/$path" || { echo "ABORT: missing input $RUN_ROOT/$path; nothing submitted" >&2; exit 1; }
done
test -d "$PROJECT_ROOT/models/Qwen3-8B" || {
  echo "ABORT: missing $PROJECT_ROOT/models/Qwen3-8B; nothing submitted" >&2; exit 1; }
used() {  # GPU-h of this user's unknownsize-018* jobs since the amendment: elapsed seconds x allocated GPUs.
  sacct -n -X -P -u "$USER" -S 2026-10-14 -o JobName,ElapsedRaw,AllocTRES | awk -F'|' '
    $1 ~ /^unknownsize-018/ { g = 0; if (match($3, /gres\/gpu[^=,]*=[0-9]+/)) { s = substr($3, RSTART, RLENGTH); sub(/.*=/, "", s); g = s + 0 }
                              t += $2 * g }
    END { printf "%.2f\n", t / 3600 }'
}
cap() {  # cap CEILING: abort unless used + CEILING <= CAP.
  local spent
  spent=$(used) || { echo "ABORT: sacct failed; nothing submitted" >&2; exit 1; }
  awk -v u="$spent" -v c="$1" -v m="$CAP" 'BEGIN { exit !(u + c <= m) }' || {
    echo "ABORT: GPU cap: used $spent + ceiling $1 > $CAP GPU-h; nothing submitted" >&2; exit 1; }
  echo "GPU-h: used $spent + ceiling $1 <= $CAP"
}
if test "${2:-}" = test; then
  test "$#" -eq 2 || exit 2
  test -f "$EVAL/selection.json" || { echo "ABORT: no $EVAL/selection.json; nothing submitted" >&2; exit 1; }
  cap "$TEST_CEILING"
  cd "$CODE_ROOT"
  PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 exec "$PYTHON" -B -m fastfill.comparison.distill_018_chain select \
    --scores "$EVAL/scores" --out "$EVAL/selection.json" --code-root "$CODE_ROOT" --shards 6
fi
FROM="${2:-sft-data}"
TEACHER="${3:-}"
case "$TEACHER" in *[!0-9]*) exit 2 ;; esac
STEPS=(sft-data train-sft pairs-dpo1 train-dpo1 sample finalize pairs-dpo2 train-dpo2 dev dev-score select)
index() { local k; for k in "${!STEPS[@]}"; do if test "${STEPS[$k]}" = "$1"; then echo "$k"; return; fi; done; exit 2; }
START=$(index "$FROM") || exit 2
from() { test "$(index "$1")" -ge "$START"; }  # from STEP: STEP is FROM or later, so it is submitted.
if test -n "$TEACHER" && test "$FROM" != sft-data; then exit 2; fi  # Only sft-data reads the teacher targets.
UNTRAINED=""
if ! from pairs-dpo2 && (cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -c 'import sys
from fastfill.comparison import distill_018_chain as ch
sys.exit(not ch.not_trained(sys.argv[1]))' "$RUN_ROOT"); then
  UNTRAINED=1 CANDS=(sft-d dpo1-d)
  if test "$FROM" = train-dpo2; then START=$(index dev); fi
  echo "dpo2-d not trained (distill-018/dpo2-d-pairs: A018_DPO2_NOT_TRAINED): no DPO2-D jobs"
fi
for item in sft-data:distill-018/sft-d-rows train-sft:distill-018/placement/sft-d-merged.json \
    pairs-dpo1:distill-018/dpo1-d-pairs train-dpo1:distill-018/placement/dpo1-d-merged.json \
    finalize:distill-018/dpo2-d-samples/MANIFEST.json pairs-dpo2:distill-018/dpo2-d-pairs \
    train-dpo2:distill-018/placement/dpo2-d-merged.json dev-score:dev-eval-018/scores select:dev-eval-018/selection.json \
    test:infer-v2-007 test:infer-placement-007 test:geometry-v7 test:test-report-018 test:s1-018-test; do
  step="${item%%:*}" path="$RUN_ROOT/${item#*:}"
  if test -n "$UNTRAINED" && test "$step" = train-dpo2; then continue; fi
  if test "$step" = test || from "$step"; then
    if test -e "$path"; then echo "refusing to reuse existing $path" >&2; exit 1; fi
  elif ! test -e "$path"; then
    echo "ABORT: FROM $FROM needs $path; nothing submitted" >&2; exit 1
  fi
done
for CAND in "${CANDS[@]}"; do
  path="$EVAL/$CAND/infer-placement"
  if from dev; then
    if test -e "$path" -a ! -f "$path/MANIFEST.json"; then echo "refusing to reuse existing $path" >&2; exit 1; fi
  elif ! test -f "$path/MANIFEST.json"; then
    echo "ABORT: FROM $FROM needs $path/MANIFEST.json; nothing submitted" >&2; exit 1
  fi
done
if test "$FROM" = sft-data && test -z "$TEACHER"; then
  for path in "$D18/teacher/MANIFEST.json" "$D18/teacher/targets.jsonl"; do
    test -f "$path" || { echo "ABORT: no $path and no TEACHER_JOB; nothing submitted" >&2; exit 1; }
  done
fi
MAKE_TC=""
if from sample && ! test -f "$D18/t-c/MANIFEST.json"; then  # T-C is derived from the plan, once.
  if test -e "$D18/t-c"; then echo "refusing to reuse existing $D18/t-c" >&2; exit 1; fi
  test -f "$D18/plan/MANIFEST.json" || { echo "ABORT: no $D18/plan/MANIFEST.json to derive T-C; nothing submitted" >&2; exit 1; }
  MAKE_TC=1
fi
QUEUED=$(squeue -h -u "$USER" -n unknownsize-018 -o %i) || { echo "ABORT: squeue failed; nothing submitted" >&2; exit 1; }
if test -n "$QUEUED"; then
  echo "ABORT: amendment-018 jobs already in Slurm; nothing submitted. cancel with: scancel" $QUEUED >&2
  exit 1
fi
if test -n "$TEACHER"; then
  FOUND=$(squeue -h -j "$TEACHER" -o %i 2>/dev/null) || FOUND=""
  test "$FOUND" = "$TEACHER" || { echo "ABORT: TEACHER_JOB $TEACHER is not in Slurm; nothing submitted" >&2; exit 1; }
fi
CEILING=0
for step in train-sft train-dpo1 train-dpo2; do if from "$step"; then CEILING=$((CEILING + 4 * 6)); fi; done
if from sample; then CEILING=$((CEILING + SAMPLES * 2)); fi
if from dev; then
  for CAND in "${CANDS[@]}"; do
    if ! test -f "$EVAL/$CAND/infer-placement/MANIFEST.json"; then CEILING=$((CEILING + N * 4)); fi
  done
fi
cap "$CEILING"
SUBMITTED=""
ROOT=""
submit() {  # submit LABEL DEPENDENCY SBATCH_ARGS...; sets ID; no dependency = the held root. set -e misses $(...).
  local out=""
  if test -n "$2"; then
    out=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$2" "${@:3}") || out=""
  else
    out=$(sbatch --parsable --hold "${@:3}") || out=""
  fi
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$1'; chain NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  if test -z "$2"; then ROOT="$ID"; fi
  echo "$1 $ID"
}
after() {  # after ID...: the submitted ones joined by ':', or the recheck root when none was submitted.
  local out="" x
  for x in "$@"; do if test -n "$x"; then out="$out${out:+:}$x"; fi; done
  echo "${out:-$RECHECK}"
}
J=("$S/isambard_018.sbatch" "$CODE_ROOT")
CPU72=(--cpus-per-task=72 --mem=192G)
TRAIN=(--gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=06:00:00)
SHARD=(--gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=04:00:00)
SFTDATA="" TRAIN_SFT_D="" PAIRS1="" TRAIN_DPO1_D="" TCDATA="" SAMPLE="" FINAL="" PAIRS2="" TRAIN_DPO2_D="" MERGES="" SCORE=""
submit recheck "" "${J[@]}" recheck
RECHECK="$ID"
if from sft-data; then submit sft-data "$RECHECK${TEACHER:+:$TEACHER}" "${CPU72[@]}" --time=04:00:00 "${J[@]}" sft-data; SFTDATA="$ID"; fi
if from train-sft; then submit "train sft-d" "$(after "$SFTDATA")" "${TRAIN[@]}" "${J[@]}" train sft-d; TRAIN_SFT_D="$ID"; fi
if from pairs-dpo1; then submit pairs-dpo1 "$(after "$SFTDATA")" "${CPU72[@]}" --time=04:00:00 "${J[@]}" pairs-dpo1; PAIRS1="$ID"; fi
if from train-dpo1; then
  submit "train dpo1-d" "$(after "$TRAIN_SFT_D" "$PAIRS1")" "${TRAIN[@]}" "${J[@]}" train dpo1-d; TRAIN_DPO1_D="$ID"
fi
if test -n "$MAKE_TC"; then submit tc-data "$RECHECK" "${J[@]}" tc-data; TCDATA="$ID"; fi
if from sample; then
  submit sample "$(after "$TRAIN_DPO1_D" "$TCDATA")" --gpus-per-node=1 --cpus-per-task=16 --mem=100G --time=02:00:00 \
    --array=0-$((SAMPLES - 1)) "${J[@]}" sample "$SAMPLES"
  SAMPLE="$ID"
fi
if from finalize; then submit finalize "$(after "$SAMPLE")" "${J[@]}" finalize; FINAL="$ID"; fi
if from pairs-dpo2; then submit pairs-dpo2 "$(after "$FINAL")" "${CPU72[@]}" --time=08:00:00 "${J[@]}" pairs-dpo2; PAIRS2="$ID"; fi
if from train-dpo2; then submit "train dpo2-d" "$(after "$PAIRS2")" "${TRAIN[@]}" "${J[@]}" train dpo2-d; TRAIN_DPO2_D="$ID"; fi
if from dev; then
  for CAND in "${CANDS[@]}"; do
    if test -f "$EVAL/$CAND/infer-placement/MANIFEST.json"; then echo "skipped dev $CAND: merged"; continue; fi
    case "$CAND" in sft-d) TRAINED="$TRAIN_SFT_D" ;; dpo1-d) TRAINED="$TRAIN_DPO1_D" ;; dpo2-d) TRAINED="$TRAIN_DPO2_D" ;; esac
    SHARDS=""
    for ((i = 0; i < N; i++)); do
      RESUME=()
      if test -e "$EVAL/$CAND/placement/shard-$(printf %02d "$i")"; then RESUME=(--resume); fi
      submit "dev shard $CAND $i" "$(after "$TRAINED")" "${SHARD[@]}" "${J[@]}" shard dev "$CAND" "$i" "$N" \
        ${RESUME[@]+"${RESUME[@]}"}
      SHARDS="$SHARDS${SHARDS:+:}$ID"
    done
    submit "dev merge $CAND" "$SHARDS" "${J[@]}" dev-merge "$CAND" "$N"
    MERGES="$MERGES${MERGES:+:}$ID"
  done
fi
if from dev-score; then submit dev-score "$(after "$MERGES")" "${J[@]}" dev-score; SCORE="$ID"; fi
submit select "$(after "$SCORE")" "${J[@]}" select
scontrol release "$ROOT" || { echo "ABORT: could not release $ROOT; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released recheck $ROOT"
