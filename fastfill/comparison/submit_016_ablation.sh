#!/bin/bash
# Amendment 016 section 3 chain (DPO2 attribution ablation on dev), run by the parent on the Isambard login node:
#   submit_016_ablation.sh CODE_ROOT [AFTER_JOB]
# CODE_ROOT must be a code snapshot (main-<sha>) that holds the amendment-016 protocol doc. The ablation needs
# op-dpo2's step count, so it starts after amendment-013 training: without AFTER_JOB, dpo-004/placement/dpo2-merged.json
# must exist; with AFTER_JOB the trains depend on it. AFTER_JOB must be the amendment-013 train job, still in Slurm:
# named unknownsize-013 with submit_013.sh's train time limit 12:00:00 (no other submit_013.sh job has it).
# All jobs run isambard_016.sbatch; dependencies are afterok with --kill-on-invalid-dep=yes and the roots (both trains)
# are held until the whole chain is in, so an ABORT (it lists the ids for scancel) leaves nothing that ran:
#   per CAND in rft (dpo-005), pert (dpo-006):
#     train CAND (4 GPUs, 12 h)          [afterok AFTER_JOB]: from rpo-dpo1's exact policy, op-dpo2 e2's step count
#     N=2 dev shards (1 GPU, 14 h)       <- its train: exact policy on dev-select-v1, shared dev-eval-010 plan
#     dev merge (CPU)                    <- its shards -> dev-eval-016/CAND/infer-placement
#   score (CPU)                          <- the dev merges submitted -> dev-eval-016/ablation.json + ablation-report.md
# score also needs dev-eval-013/scores (amendment 013's own dev-score); if that is late, score fails before writing
# anything and is resubmitted by hand (sbatch CODE_ROOT/fastfill/comparison/isambard_016.sbatch CODE_ROOT score).
# A resubmit resumes: training resumes (training_guard, ckpt_latest) or only verifies a finished merge, and existing dev
# shards get --resume; a candidate whose dev-eval-016/CAND/infer-placement/MANIFEST.json exists is skipped (score then
# waits only on the other's merge, or is the held root itself); an infer-placement without MANIFEST.json or an existing
# ablation.json refuses.
# Refuses while any unknownsize-016 job of this user is in Slurm.
set -euo pipefail
test "$#" -ge 1 -a "$#" -le 2 || exit 2
CODE_ROOT="$1"
AFTER="${2:-}"
case "$AFTER" in *[!0-9]*) exit 2 ;; esac
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
EVAL="$RUN_ROOT/dev-eval-016"
S="$CODE_ROOT/fastfill/comparison"
N=2
for name in ablation_016.py onpolicy_dpo.py dpo_stages.py dev_select.py infer_shard.py isambard_016.sbatch; do
  test -f "$S/$name" || { echo "ABORT: missing $S/$name; nothing submitted" >&2; exit 1; }
done
test -f "$CODE_ROOT/fastfill/v1_1/train_dpo.py" && grep -q -- '--sft_only' "$CODE_ROOT/fastfill/v1_1/train_dpo.py" || {
  echo "ABORT: $CODE_ROOT/fastfill/v1_1/train_dpo.py has no --sft_only; nothing submitted" >&2; exit 1; }
DOC=""
for path in "$CODE_ROOT"/docs/research/20261013-unknown-size-comparison/amendment-016-*.md; do
  if test -f "$path"; then DOC="$path"; fi
done
test -n "$DOC" || { echo "ABORT: no amendment-016 protocol doc in $CODE_ROOT; nothing submitted" >&2; exit 1; }
INPUTS=(dpo-001/placement/sft-merged.json dpo-001/stage2-pairs/MANIFEST.json dpo-003/placement/dpo1-merged.json
        dpo-003/placement/dpo1/final/adapter_model.safetensors dpo-004/stage2-pairs/MANIFEST.json
        dev-select-v1/MANIFEST.json dev-eval-010/plan/MANIFEST.json dev-eval-010/scores/MANIFEST.json
        context-capacity-001/SHA256.json)
if test -z "$AFTER"; then INPUTS+=(dpo-004/placement/dpo2-merged.json dpo-004/placement/dpo2/final/completion.json); fi
for path in "${INPUTS[@]}"; do
  test -f "$RUN_ROOT/$path" || { echo "ABORT: missing input $RUN_ROOT/$path; nothing submitted" >&2; exit 1; }
done
test -d "$PROJECT_ROOT/models/Qwen3-8B" || {
  echo "ABORT: missing $PROJECT_ROOT/models/Qwen3-8B; nothing submitted" >&2; exit 1; }
if test -n "$AFTER"; then
  NAME=$(squeue -h -j "$AFTER" -o "%j %l" 2>/dev/null) || NAME=""
  test "$NAME" = "unknownsize-013 12:00:00" || {
    echo "ABORT: AFTER_JOB $AFTER is not the amendment-013 train job in Slurm (got '${NAME}'); once" \
      "dpo-004/placement/dpo2-merged.json exists, submit without AFTER_JOB; nothing submitted" >&2; exit 1; }
fi
for path in "$EVAL/ablation.json" "$EVAL/rft/infer-placement" "$EVAL/pert/infer-placement"; do
  if test -e "$path" -a ! -f "$path/MANIFEST.json"; then echo "refusing to reuse existing $path" >&2; exit 1; fi
done
QUEUED=$(squeue -h -u "$USER" -n unknownsize-016 -o %i) || { echo "ABORT: squeue failed; nothing submitted" >&2; exit 1; }
if test -n "$QUEUED"; then
  echo "ABORT: amendment-016 jobs already in Slurm; nothing submitted. cancel with: scancel" $QUEUED >&2
  exit 1
fi
SUBMITTED=""
ROOTS=""
submit() {  # submit LABEL DEPENDENCY SBATCH_ARGS...; sets ID; no dependency = a held root (afterok AFTER_JOB if given).
  local out="" dep=()
  if test -n "$2"; then
    dep=(--kill-on-invalid-dep=yes --dependency="afterok:$2")
  else
    dep=(--hold)
    if test -n "$AFTER"; then dep+=(--kill-on-invalid-dep=yes --dependency="afterok:$AFTER"); fi
  fi
  out=$(sbatch --parsable "${dep[@]}" "${@:3}") || out=""
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
J=("$S/isambard_016.sbatch" "$CODE_ROOT")
SHARD=(--gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=14:00:00)
MERGES=""
for CAND in rft pert; do
  if test -f "$EVAL/$CAND/infer-placement/MANIFEST.json"; then echo "skipped $CAND: dev inference merged"; continue; fi
  submit "train $CAND" "" --gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=12:00:00 "${J[@]}" train "$CAND"
  TRAIN="$ID"
  SHARDS=""
  for ((i = 0; i < N; i++)); do
    RESUME=()
    if test -e "$EVAL/$CAND/placement/shard-$(printf %02d "$i")"; then RESUME=(--resume); fi
    submit "dev shard $CAND $i" "$TRAIN" "${SHARD[@]}" "${J[@]}" shard "$CAND" "$i" "$N" ${RESUME[@]+"${RESUME[@]}"}
    SHARDS="$SHARDS${SHARDS:+:}$ID"
  done
  submit "dev merge $CAND" "$SHARDS" "${J[@]}" dev-merge "$CAND" "$N"
  MERGES="$MERGES${MERGES:+:}$ID"
done
submit score "$MERGES" "${J[@]}" score
scontrol release "$ROOTS" || { echo "ABORT: could not release $ROOTS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released trains $ROOTS"
