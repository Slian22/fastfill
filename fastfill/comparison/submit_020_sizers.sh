#!/bin/bash
# Amendment 020 stage 1 (EXPLORATORY): the sizers and their size-quality report (table A). Run by the user on the
# Isambard login node:
#   submit_020_sizers.sh CODE_ROOT [SHARDS=6]
# Chain (isambard_020.sbatch; every dependent job --kill-on-invalid-dep=yes; rooms is the held root, released once
# everything is in, so an ABORT, which lists the ids for scancel, leaves nothing that ran):
#   rooms    CPU 4 / 32G / 1 h, held root            -> RUN/a020-rooms-v1.json (E2 1,009 + E3 868 rooms)
#   L-size   CPU 1 / 8G / 8 h, afterok rooms, job name unknownsize-020-sizer-L + singleton; API, cap preflight in the
#            job (status CAP_REFUSED, no call, when used + rooms x 3 > 4,800) -> RUN/sizes-020-L. Not submitted while
#            an unknownsize-llm-agent job is in Slurm (its ledger would grow under the preflight).
#   O2-size  SHARDS GPU jobs, afterok rooms, job name unknownsize-020-sizer-O2, no singleton (never queues behind
#            unknownsize-llm-open): llm_open's O2 resources (2 GPUs) -> RUN/sizes-020-O2
#   report   CPU 4 / 32G / 2 h, afterany L-size + every shard: finalize O2, then RUN/sizes-020-report-v1 (refuses
#            unless O2-size is complete: rerun the failed shards, then the report)
# Every job excludes nid010216, nid011181 and nid010569 (an isambard_020.sbatch header line, reruns included).
# GPU ceiling: every shard gets --time = ceil(1.5 x (LOAD + RATE x ceil(ROOMS / SHARDS)) / 60) minutes from the
# J-input O2 run RUN/llm-open-O2-001 (RATE = cost generation_seconds / result rooms, LOAD = wall - generation; sizing
# answers are shorter, so this is conservative); SHARDS x minutes x GPUs must be <= 960 GPU-minutes (amendment 020's
# 30 GPU-h minus OptiScene ~4 and P ~10), else nothing is submitted.
# Before submitting, prints the API cap preflight for ROOMS = 1,877 (sizers_020 cap; read-only).
# Refuses (nothing submitted) when an output above exists (the rooms file may: the job checks it against its
# rebuild), an input is missing, CODE_ROOT lacks the amendment-020 code, or an unknownsize-020 job is in Slurm.
set -euo pipefail
test "$#" -ge 1 -a "$#" -le 2 || { echo "usage: submit_020_sizers.sh CODE_ROOT [SHARDS=6]" >&2; exit 2; }
CODE_ROOT="$1"
SHARDS="${2:-6}"
case "$SHARDS" in ''|*[!0-9]*) exit 2 ;; esac
test "$SHARDS" -ge 1 -a "$SHARDS" -le 99 || exit 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
S="$CODE_ROOT/fastfill/comparison"
B="$S/isambard_020.sbatch"
MEASURED="$RUN/llm-open-O2-001"
ROOMS=1877  # Amendment 020 section 1: E2 1,009 + E3 868.
CAP_MINUTES=960
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
for out in sizes-020-L sizes-020-O2 sizes-020-report-v1; do test ! -e "$RUN/$out" || fail "$RUN/$out exists"; done
for f in sizers_020.py isambard_020.sbatch llm_open.py llm_agent.py optiscene_eval.py room_families.py; do
  test -f "$S/$f" || fail "missing $S/$f"
done
for f in eval-data-v1/MANIFEST.json optiscene-014/e2/MANIFEST.json optiscene-015/e3/MANIFEST.json \
         llm-eval-L-selected-v1/MANIFEST.json llm-open-O2-001/cost.json llm-open-O2-001/result.json; do
  test -f "$RUN/$f" || fail "missing $RUN/$f"
done
test -f "$PROJECT_ROOT/models/Qwen3.8-27B/config.json" || fail "missing $PROJECT_ROOT/models/Qwen3.8-27B"
QUEUED=$(squeue -h -u "$USER" -o '%i %j') || fail "squeue failed"
OURS=$(printf '%s\n' "$QUEUED" | awk '$2 ~ /^unknownsize-020-/ {print $1}')
test -z "$OURS" || fail "amendment-020 jobs already in Slurm (scancel $(echo $OURS))"
API=$(printf '%s\n' "$QUEUED" | awk '$2 ~ /^unknownsize-llm-agent/ {print $1}')
F=$(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -m fastfill.comparison.llm_open slurm --system O2 --mode full) \
  || fail "llm_open slurm flags for O2"
F="${F% --time=*}"
GPUS="${F#*--gpus-per-node=}"
GPUS="${GPUS%% *}"
MINUTES=$("$PYTHON" -B - "$MEASURED" $(((ROOMS + SHARDS - 1) / SHARDS)) <<'PY'
import json, math, sys
root, rooms = sys.argv[1], int(sys.argv[2])
cost, result = (json.load(open(f'{root}/{name}.json')) for name in ('cost', 'result'))
assert result['status'] == 'LLM_COMPLETE' and cost['system'] == 'O2' and result['rooms'] > 0, root
rate, load = cost['generation_seconds'] / result['rooms'], cost['wall_seconds'] - cost['generation_seconds']
print(math.ceil(1.5 * (load + rate * rooms) / 60))
PY
) || fail "no measured O2 rate in $MEASURED"
CEILING=$((SHARDS * MINUTES * GPUS))
test "$CEILING" -le "$CAP_MINUTES" \
  || fail "GPU ceiling $CEILING GPU-minutes ($SHARDS shards x $MINUTES min x $GPUS GPUs) exceeds $CAP_MINUTES"
TIME=$(printf '%02d:%02d:00' $((MINUTES / 60)) $((MINUTES % 60)))
echo "API cap preflight for $ROOMS rooms (the L-size job repeats it with the real room count):"
(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -m fastfill.comparison.sizers_020 cap --run-root "$RUN" \
  --rooms "$ROOMS") && echo "  -> within the cap" \
  || echo "  -> refused (cap exceeded or an unfinished API run, see above): the L-size job will refuse (CAP_REFUSED)"
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
LOGS="$PROJECT_ROOT/runs/isambard"
CPU=(--account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --gres=none --no-requeue)
submit rooms --hold "${CPU[@]}" --cpus-per-task=4 --mem=32G --time=01:00:00 --job-name=unknownsize-020-rooms \
  --output="$LOGS/unknownsize-020-rooms-%j.out" "$B" "$CODE_ROOT" rooms
ROOT="$ID"
AFTER=""
if test -z "$API"; then
  submit L-size --kill-on-invalid-dep=yes --dependency="afterok:$ROOT,singleton" "${CPU[@]}" --cpus-per-task=1 \
    --mem=8G --time=08:00:00 --job-name=unknownsize-020-sizer-L --output="$LOGS/unknownsize-020-sizer-L-%j.out" \
    "$B" "$CODE_ROOT" l-size
  AFTER="$AFTER:$ID"
else
  echo "skip L-size: unknownsize-llm-agent jobs in Slurm ($(echo $API)); submit later with:" \
    "sbatch ${CPU[*]} --cpus-per-task=1 --mem=8G --time=08:00:00 --job-name=unknownsize-020-sizer-L $B $CODE_ROOT l-size"
fi
for ((k = 0; k < SHARDS; k++)); do
  # shellcheck disable=SC2086  # F holds space-separated sbatch options
  submit "O2-size s$k" --kill-on-invalid-dep=yes --dependency="afterok:$ROOT" --job-name=unknownsize-020-sizer-O2 \
    $F --time="$TIME" --output="$LOGS/unknownsize-020-sizer-O2-%j.out" "$B" "$CODE_ROOT" o2-size "$k" "$SHARDS"
  AFTER="$AFTER:$ID"
done
REPORT=("${CPU[@]}" --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=unknownsize-020-report
        --output="$LOGS/unknownsize-020-report-%j.out")
submit report --kill-on-invalid-dep=yes --dependency="afterany$AFTER" "${REPORT[@]}" "$B" "$CODE_ROOT" report
scontrol release "$ROOT" || { echo "ABORT: could not release $ROOT; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOT (O2 GPU ceiling $CEILING GPU-minutes: $SHARDS shards x $TIME x $GPUS GPUs)"
echo "rerun one O2 shard K: sbatch --job-name=unknownsize-020-sizer-O2 $F --time=$TIME $B $CODE_ROOT o2-size K $SHARDS"
echo "then the report: sbatch ${REPORT[*]} $B $CODE_ROOT report"
