#!/bin/bash
# Amendment 013 chain (on-policy DPO2 for arm P on rpo-dpo1), run by the parent on the Isambard login node:
#   submit_013.sh CODE_ROOT [SHARDS=4] [DEV_N=2] [FROM=subset]
#   submit_013.sh CODE_ROOT test
# CODE_ROOT must be a new code snapshot (main-<sha>) that holds the amendment-013 protocol doc: sample_vllm's run
# identity hashes sample_vllm.py and dpo_stages.py, so a resubmit of the amendment-012 S2 chain keeps its own snapshot.
# First, by hand, once the amendment-012 S2 placement samples are finalized (CPU, no GPU):
#   sbatch --cpus-per-task=72 --mem=192G --time=08:00:00 CODE_ROOT/fastfill/comparison/isambard_013.sbatch CODE_ROOT yield
# Sampling is submitted only if RUN/op-yield-013 projects the pair floors (1,500 pairs, 600 rooms) at 3,000 rooms.
# All jobs run isambard_013.sbatch; dependencies are afterok with --kill-on-invalid-dep=yes and the root is held until
# the whole chain is in, so an ABORT (it lists the ids for scancel) leaves nothing that ran:
#   recheck (CPU, held root)             this code reproduces dev-eval-010's sft / rpo-dpo1 dev scores (before any GPU)
#   subset (CPU)                         <- recheck: dpo-004/train-subset (3,000 train rooms; no positives, dev, test)
#   smoke (1 GPU, 2 h, --limit 16)       <- subset: the 16 largest rooms through vLLM
#   array (SHARDS x 1 GPU, 4 h)          <- smoke: sample_vllm run, K = 9 per room -> dpo-004/samples
#   finalize (CPU)                       <- array
#   pairs (CPU 72, 192G, 8 h)            <- finalize: S1-repaired / best-passing chosen -> dpo-004/stage2-pairs
#   train (4 GPUs, 12 h)                 <- pairs: dpo2 from rpo-dpo1's exact policy (dpo-001 SFT merge + the dpo-003
#                                           dpo1 adapter as policy start and frozen reference); merges epoch 1 and 2
#   per op-dpo2-e1, op-dpo2-e2: DEV_N dev shards (1 GPU) <- train; dev merge (CPU) <- its shards
#   dev-score (CPU)                      <- both dev merges
#   select (CPU)                         <- dev-score: selection.json only (--no-submit)
# test: only once stage_table / report_v2 / delivery take the amendment-013 selection (today they reject it, so nothing
# would make the amendment-005 J-SFT vs P-op decision): reruns select here, on the login node, without --no-submit. If
# op was adopted it submits the test chain (test shards, merge, geometry-v6, test report, merged-weight check).
# FROM (subset sample finalize pairs train dev dev-score select) restarts a failed chain with the same SHARDS and DEV_N:
# earlier steps are skipped (the step just before FROM must have its output), recheck stays the held root, sampling and
# training resume, existing dev shards get --resume, and an output of FROM or a later step refuses.
# Budget cap 150 GPU-h (sampling <= 30, training ~15-20, dev ~4, test ~60-80 only if adopted).
# Refuses while any unknownsize-s2-* or unknownsize-013 job of this user is in Slurm.
set -euo pipefail
test "$#" -ge 1 -a "$#" -le 4 || exit 2
CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
EVAL="$RUN_ROOT/dev-eval-013"
S="$CODE_ROOT/fastfill/comparison"
for name in onpolicy_dpo.py sample_vllm.py dpo_stages.py dev_select.py infer_shard.py output_repair.py best_of_n.py \
    downstream.py isambard_013.sbatch isambard_infer_merge.sbatch; do
  test -f "$S/$name" || { echo "ABORT: missing $S/$name; nothing submitted" >&2; exit 1; }
done
DOC=""
for path in "$CODE_ROOT"/docs/research/20261013-unknown-size-comparison/amendment-013-*.md; do
  if test -f "$path"; then DOC="$path"; fi
done
test -n "$DOC" || { echo "ABORT: no amendment-013 protocol doc in $CODE_ROOT; nothing submitted" >&2; exit 1; }
for path in main-data-v1/MANIFEST.json sft-joint-001/SHA256.json sft-placement-001/SHA256.json \
    dpo-001/positives/MANIFEST.json dev-select-v1/MANIFEST.json dev-full-v1/MANIFEST.json eval-data-v1/MANIFEST.json \
    dpo-003/placement/dpo1-merged.json dpo-001/placement/sft-merged.json \
    dpo-003/placement/dpo1/final/adapter_model.safetensors dev-eval-010/plan/MANIFEST.json dev-eval-010/scores/MANIFEST.json \
    dev-eval-010/selection.json dev-eval-010/sft/infer-placement/MANIFEST.json \
    dev-eval-010/rpo-dpo1/infer-placement/MANIFEST.json context-capacity-001/SHA256.json \
    infer-v2-002/plan/MANIFEST.json infer-joint-002/MANIFEST.json infer-placement-005/MANIFEST.json; do
  test -f "$RUN_ROOT/$path" || { echo "ABORT: missing input $RUN_ROOT/$path; nothing submitted" >&2; exit 1; }
done
test -d "$PROJECT_ROOT/models/Qwen3-8B" || {
  echo "ABORT: missing $PROJECT_ROOT/models/Qwen3-8B; nothing submitted" >&2; exit 1; }
if test "${2:-}" = test; then
  test "$#" -eq 2 || exit 2
  test -f "$EVAL/selection.json" || { echo "ABORT: no $EVAL/selection.json; nothing submitted" >&2; exit 1; }
  cd "$CODE_ROOT"
  PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 exec "$PYTHON" -B -m fastfill.comparison.onpolicy_dpo select \
    --scores "$EVAL/scores" --out "$EVAL/selection.json" --code-root "$CODE_ROOT" --shards 6
fi
K="${2:-4}"
N="${3:-2}"
FROM="${4:-subset}"
case "$K" in ''|*[!0-9]*) exit 2 ;; esac
case "$N" in ''|*[!0-9]*) exit 2 ;; esac
test "$K" -ge 1 -a "$N" -ge 1 || exit 2
STEPS=(subset sample finalize pairs train dev dev-score select)
index() { local k; for k in "${!STEPS[@]}"; do if test "${STEPS[$k]}" = "$1"; then echo "$k"; return; fi; done; exit 2; }
START=$(index "$FROM") || exit 2
from() { test "$(index "$1")" -ge "$START"; }  # from STEP: STEP is FROM or later, so it is submitted.
PREVIOUS=""
if test "$START" -gt 0; then PREVIOUS="${STEPS[$((START - 1))]}"; fi
for item in subset:dpo-004/train-subset finalize:dpo-004/samples/MANIFEST.json pairs:dpo-004/stage2-pairs \
    train:dpo-004/placement/dpo2-merged.json dev:dev-eval-013/op-dpo2-e1/infer-placement \
    dev:dev-eval-013/op-dpo2-e2/infer-placement dev-score:dev-eval-013/scores select:dev-eval-013/selection.json \
    test:infer-v2-006 test:infer-placement-006 test:geometry-v6 test:test-report-013; do
  step="${item%%:*}" path="$RUN_ROOT/${item#*:}"
  if test "$step" = test || from "$step"; then
    if test -e "$path"; then echo "refusing to reuse existing $path" >&2; exit 1; fi
  elif test "$step" = "$PREVIOUS" && ! test -e "$path"; then
    echo "ABORT: FROM $FROM needs $path; nothing submitted" >&2; exit 1
  fi
done
if from sample; then
  "$PYTHON" -B - "$RUN_ROOT/op-yield-013/summary.json" <<'PY' || {
import json, sys
value = json.load(open(sys.argv[1]))
assert value['status'] == 'OP_YIELD_DRY_RUN' and value['projection']['meets_floors'] is True, value.get('projection')
PY
    echo "ABORT: $RUN_ROOT/op-yield-013 (sbatch ... yield) does not project the pair floors; nothing submitted" >&2
    exit 1; }
fi
QUEUED=$(squeue -h -u "$USER" -n unknownsize-s2-sample,unknownsize-s2-cpu,unknownsize-013 -o %i) || {
  echo "ABORT: squeue failed; nothing submitted" >&2; exit 1; }
if test -n "$QUEUED"; then
  echo "ABORT: S2 or amendment-013 jobs already in Slurm; nothing submitted. cancel with: scancel" $QUEUED >&2
  exit 1
fi
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
      echo "ABORT: sbatch failed for '$1'; chain NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  if test -z "$2"; then ROOTS="$ROOTS${ROOTS:+,}$ID"; fi
  echo "$1 $ID"
}
J=("$S/isambard_013.sbatch" "$CODE_ROOT")
SAMPLE=(--gpus-per-node=1 --cpus-per-task=16 --mem=100G)
SHARD=(--gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=14:00:00)
submit recheck "" "${J[@]}" recheck
if from subset; then submit subset "$ID" "${J[@]}" subset; fi
if from sample; then
  submit smoke "$ID" "${SAMPLE[@]}" --time=02:00:00 "${J[@]}" sample 1 --limit 16
  submit array "$ID" "${SAMPLE[@]}" --time=04:00:00 --array=0-$((K - 1)) "${J[@]}" sample "$K"
fi
if from finalize; then submit finalize "$ID" "${J[@]}" finalize; fi
if from pairs; then submit pairs "$ID" --cpus-per-task=72 --mem=192G --time=08:00:00 "${J[@]}" pairs; fi
if from train; then submit train "$ID" --gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=12:00:00 "${J[@]}" train; fi
if from dev; then
  TRAIN="$ID"
  MERGES=""
  for CAND in op-dpo2-e1 op-dpo2-e2; do
    SHARDS=""
    for ((i = 0; i < N; i++)); do
      RESUME=()
      if test -e "$EVAL/$CAND/placement/shard-$(printf %02d "$i")"; then RESUME=(--resume); fi
      submit "dev shard $CAND $i" "$TRAIN" "${SHARD[@]}" "${J[@]}" shard dev "$CAND" "$i" "$N" ${RESUME[@]+"${RESUME[@]}"}
      SHARDS="$SHARDS${SHARDS:+:}$ID"
    done
    submit "dev merge $CAND" "$SHARDS" "${J[@]}" dev-merge "$CAND" "$N"
    MERGES="$MERGES${MERGES:+:}$ID"
  done
  ID="$MERGES"
fi
if from dev-score; then submit dev-score "$ID" "${J[@]}" dev-score; fi
submit select "$ID" "${J[@]}" select
scontrol release "$ROOTS" || { echo "ABORT: could not release $ROOTS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released recheck $ROOTS"
