#!/bin/bash
# Amendment 018 section 1 teacher data (EXPLORATORY), run by the parent on the Isambard login node:
#   submit_018_teacher.sh CODE_ROOT [SHARDS=8]
# CODE_ROOT must be a new code snapshot (main-<sha>) holding the amendment-018 protocol doc and distill_018.py.
# Chain (afterok with --kill-on-invalid-dep=yes; the plan is the held root, released once everything is in, so an ABORT
# (it lists the ids for scancel) leaves nothing that ran):
#   plan     CPU 32 / 96G / 3 h, held root   distill_018 plan -> RUN/distill-018/plan (T-A = dpo-004/train-subset,
#            T-B 5,000, T-C 3,000; SHARDS stratified round-robin shards of T-A + T-B; cohort.json for llm_open)
#   teacher  SHARDS jobs <- plan             isambard_llm_open.sbatch CODE_ROOT O2 t018 sNN: llm_open --system O2 --arm
#            placement --all-rooms --uids plan/shards/sNN.json -> RUN/llm-open-O2-T018-sNN-001 (amendment-007 engine,
#            sampling and GPUs; amendment-016 section 2 harness and B2). Job name unknownsize-018-teacher and no
#            singleton: the shards run side by side and never queue behind unknownsize-llm-open.
#   filter   CPU 32 / 64G / 4 h <- every shard   distill_018 teacher -> RUN/distill-018/teacher (--reference = the
#            amendment-016 section 2 O2 pilot run, whose config the shards must match)
# GPU ceiling (teacher cap 35 GPU-h, amendment 018): every shard gets --time = distill_018.teacher_minutes (TIME_RULE:
# the measured O2 P-input pilot RUN/llm-open-O2-P-pilot-001 gives seconds per output token and each object-count bin's
# mean output tokens per room; the shard's rooms per bin come from T-A's rooms.jsonl and T-B's thirds, so the teacher
# set's denser mix is priced, not the pilot's mostly sparse one); the ceiling SHARDS x minutes x GPUs (llm_open's O2
# resources) must be <= 2,100 GPU-minutes, else nothing is submitted and the user decides (raise the cap or shrink T-B).
# A shard that hits its limit resumes from its receipts under the same identity when resubmitted by hand with the
# printed sbatch line (count the rerun against the cap); then resubmit the printed filter line.
# Refuses (nothing submitted) when an output above exists, an input is missing, CODE_ROOT lacks the amendment-018 code,
# or an unknownsize-018-plan / -teacher / -filter job of this user is in Slurm.
set -euo pipefail
test "$#" -ge 1 -a "$#" -le 2 || { echo "usage: submit_018_teacher.sh CODE_ROOT [SHARDS=8]" >&2; exit 2; }
CODE_ROOT="$1"
SHARDS="${2:-8}"
case "$SHARDS" in ''|*[!0-9]*) exit 2 ;; esac
test "$SHARDS" -ge 1 -a "$SHARDS" -le 99 || exit 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
S="$CODE_ROOT/fastfill/comparison"
PLAN="$RUN/distill-018/plan"
PILOT="$RUN/llm-open-O2-P-pilot-001"
CAP_MINUTES=4320  # 72 GPU-h ceiling (amendment 018 addendum B)
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
OUTS=(distill-018/plan distill-018/teacher)
for ((k = 0; k < SHARDS; k++)); do OUTS+=("llm-open-O2-T018-s$(printf %02d "$k")-001"); done
for out in "${OUTS[@]}"; do test ! -e "$RUN/$out" || fail "$RUN/$out exists"; done
for f in distill_018.py llm_open.py llm_agent.py evaluate_llm.py onpolicy_dpo.py isambard_llm_open.sbatch; do
  test -f "$S/$f" || fail "missing $S/$f"
done
grep -q -- "--all-rooms" "$S/llm_open.py" || fail "$S/llm_open.py has no --all-rooms"
grep -q 't018)' "$S/isambard_llm_open.sbatch" || fail "$S/isambard_llm_open.sbatch has no t018 mode"
DOC=""
for path in "$CODE_ROOT"/docs/research/20261013-unknown-size-comparison/amendment-018-*.md; do
  if test -f "$path"; then DOC="$path"; fi
done
test -n "$DOC" || fail "no amendment-018 protocol doc in $CODE_ROOT"
for f in main-data-v1/MANIFEST.json sft-joint-001/SHA256.json sft-placement-001/SHA256.json \
    dpo-001/positives/MANIFEST.json dpo-004/train-subset/MANIFEST.json dev-select-v1/MANIFEST.json \
    dev-full-v1/MANIFEST.json eval-data-v1/MANIFEST.json llm-open-O2-P-pilot-001/MANIFEST.json \
    llm-open-O2-P-pilot-001/cost.json llm-open-O2-P-pilot-001/result.json llm-open-O2-P-pilot-001/run_receipt.json \
    eval-data-v1/rooms.jsonl dpo-004/train-subset/rooms.jsonl; do
  test -f "$RUN/$f" || fail "missing $RUN/$f"
done
test -f "$PROJECT_ROOT/models/Qwen3.8-27B/config.json" || fail "missing $PROJECT_ROOT/models/Qwen3.8-27B"
QUEUED=$(squeue -h -u "$USER" -n unknownsize-018-plan,unknownsize-018-teacher,unknownsize-018-filter -o %i) \
  || fail "squeue failed"
test -z "$QUEUED" || fail "amendment-018 teacher jobs already in Slurm (scancel $(echo $QUEUED))"
F=$(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -m fastfill.comparison.llm_open slurm --system O2 --mode full) \
  || fail "llm_open slurm flags for O2"
F="${F% --time=*}"
GPUS="${F#*--gpus-per-node=}"
GPUS="${GPUS%% *}"
ESTIMATE=$(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -c 'import json, sys
from fastfill.comparison import distill_018 as ds
value = ds.teacher_minutes(*sys.argv[1:4], int(sys.argv[4]))
print(value["minutes"], json.dumps(value))' "$PILOT" "$RUN/eval-data-v1" "$RUN/dpo-004/train-subset" "$SHARDS") \
  || fail "no density-weighted O2 time estimate from $PILOT (error above)"
MINUTES="${ESTIMATE%% *}"
CEILING=$((SHARDS * MINUTES * GPUS))
test "$CEILING" -le "$CAP_MINUTES" \
  || fail "GPU ceiling $CEILING GPU-minutes ($SHARDS shards x $MINUTES min x $GPUS GPUs) exceeds the 72 GPU-h teacher cap (user decision: raise the cap or shrink T-B; estimate ${ESTIMATE#* })"
TIME=$(printf '%02d:%02d:00' $((MINUTES / 60)) $((MINUTES % 60)))
SUBMITTED=""
submit() {  # submit LABEL SBATCH_ARGS...; sets ID. set -e does not reach $(...), so check here.
  local out=""
  out=$(sbatch --parsable "${@:2}") || out=""
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
ENV="cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
LOGS="$PROJECT_ROOT/runs/isambard"
CPU=(--account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --gres=none --no-requeue)
submit plan --hold "${CPU[@]}" --cpus-per-task=32 --mem=96G --time=03:00:00 --job-name=unknownsize-018-plan \
  --output="$LOGS/unknownsize-018-plan-%j.out" \
  --wrap "$ENV $PYTHON -B -m fastfill.comparison.distill_018 plan --data $RUN/main-data-v1 --sft-runs $RUN/sft-joint-001 $RUN/sft-placement-001 --exclude $RUN/dpo-001/positives --ta $RUN/dpo-004/train-subset --dev $RUN/dev-select-v1 $RUN/dev-full-v1 --test $RUN/eval-data-v1 --out $PLAN --shards $SHARDS"
ROOT="$ID"
TEACHERS=""
for ((k = 0; k < SHARDS; k++)); do
  # shellcheck disable=SC2086  # F holds space-separated sbatch options
  submit "teacher s$(printf %02d "$k")" --kill-on-invalid-dep=yes --dependency="afterok:$ROOT" \
    --job-name=unknownsize-018-teacher $F --time="$TIME" "$S/isambard_llm_open.sbatch" "$CODE_ROOT" O2 t018 \
    "s$(printf %02d "$k")"
  TEACHERS="$TEACHERS:$ID"
done
FILTER=("${CPU[@]}" --cpus-per-task=32 --mem=64G --time=04:00:00 --job-name=unknownsize-018-filter
        --output="$LOGS/unknownsize-018-filter-%j.out")
FILTER_WRAP="$ENV $PYTHON -B -m fastfill.comparison.distill_018 teacher --plan $PLAN --run-root $RUN --reference $PILOT --out $RUN/distill-018/teacher --workers 32"
submit filter --kill-on-invalid-dep=yes --dependency="afterok$TEACHERS" "${FILTER[@]}" --wrap "$FILTER_WRAP"
scontrol release "$ROOT" || { echo "ABORT: could not release $ROOT; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOT (GPU ceiling $CEILING GPU-minutes: $SHARDS shards x $TIME x $GPUS GPUs)"
echo "time estimate: ${ESTIMATE#* }"
echo "rerun one shard: sbatch --job-name=unknownsize-018-teacher $F --time=$TIME $S/isambard_llm_open.sbatch $CODE_ROOT O2 t018 sNN"
echo "then the filter: sbatch --dependency=afterok:SHARD_JOB ${FILTER[*]} --wrap \"$FILTER_WRAP\""
