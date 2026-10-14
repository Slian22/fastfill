#!/bin/bash
# Amendment 012 §4 test application (secondary, EXPLORATORY; confirmatory and delivery results unchanged), run by the
# user on the Isambard login node once the dev decision RUN/decision-012 (S1 adopted, S2 not) is published:
#   submit_test_012.sh CODE_ROOT
# S1 once on test at the dev settings (output_repair, max_trials 2000, time_limit 60; best-of-N not applied):
#   s1-J          RUN/geometry-v5/joint-geometry.jsonl (J = SFT, infer-joint-002), every test request, requests arm joint
#                 -> RUN/s1-012-test/joint-sft
#   s1-P          RUN/geometry-v5/placement-geometry.jsonl (P = rpo-dpo1), every test request, requests arm placement
#                 -> RUN/s1-012-test/placement-rpo-dpo1
#   s1-NAME-V     NAME in L O1a O1b O2 O3 O4 O6 O7, V in harness raw: RUN/llm-eval-NAME-selected-v1/llm_V-geometry.jsonl
#                 on the cohort (--uids RUN/llm-cohort-v1.json), requests arm joint (agents received J's input)
#                 -> RUN/s1-012-test/agent-NAME-V
#   summarize     afterok every S1 job: test_apply_012 summarize -> RUN/apply-012-test
# Every job: CPU 72 cores, 200G, 4 h, --workers 70 (no GPU). The S1 jobs are held roots, released once the whole chain
# is in; on any sbatch failure: ABORT with the ids to scancel. Nothing is submitted on a missing input, a dev decision
# other than S1 adopted / S2 not adopted, a CODE_ROOT without output_repair --uids or test_apply_012, an
# unknownsize-012-test job in Slurm, or any existing output. Logs: PROJECT_ROOT/runs/isambard/unknownsize-012-test-*.
set -euo pipefail
test "$#" -eq 1 || { echo "usage: submit_test_012.sh CODE_ROOT" >&2; exit 2; }
CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
S1="$RUN/s1-012-test"
OUT="$RUN/apply-012-test"
AGENTS=(L O1a O1b O2 O3 O4 O6 O7)
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
for f in output_repair.py test_apply_012.py decide_012.py downstream.py; do
  test -f "$CODE_ROOT/fastfill/comparison/$f" || fail "missing $CODE_ROOT/fastfill/comparison/$f"
done
INPUTS=(eval-data-v1/MANIFEST.json geometry-v5/MANIFEST.json geometry-v5/joint-geometry.jsonl
        geometry-v5/placement-geometry.jsonl llm-cohort-v1.json decision-012/MANIFEST.json decision-012/decision.json)
OUTS=("$S1/joint-sft" "$S1/placement-rpo-dpo1" "$OUT")
for NAME in "${AGENTS[@]}"; do
  INPUTS+=("llm-eval-$NAME-selected-v1/MANIFEST.json")
  for V in harness raw; do
    INPUTS+=("llm-eval-$NAME-selected-v1/llm_$V-geometry.jsonl")
    OUTS+=("$S1/agent-$NAME-$V")
  done
done
for f in "${INPUTS[@]}"; do test -f "$RUN/$f" || fail "missing $RUN/$f"; done
for path in "${OUTS[@]}"; do test ! -e "$path" || fail "$path exists"; done
test -x "$PYTHON" || fail "no executable $PYTHON"
"$PYTHON" -I -B - "$RUN/decision-012/decision.json" <<'PY' || fail "RUN/decision-012 is not 'S1 adopted, S2 not adopted'"
import json, sys
value = json.load(open(sys.argv[1]))
decisions = value['decisions']
assert value['status'] == 'DECISION_012_COMPLETE' and decisions['s1_adopted'] is True
assert not decisions['s2_joint']['passed'] and not decisions['s2_placement']['passed']
PY
(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -c 'import inspect
from fastfill.comparison import output_repair, test_apply_012
assert "uids" in inspect.signature(output_repair.run).parameters') \
  || fail "$CODE_ROOT cannot import output_repair with --uids and test_apply_012"
QUEUED=$(squeue -h -u "$USER" -o '%i %j') || fail "squeue failed"
QUEUED=$(printf '%s\n' "$QUEUED" | awk '$2 ~ /^unknownsize-012-test/ {print $1}')
test -z "$QUEUED" || fail "unknownsize-012-test jobs already in Slurm (scancel $(echo $QUEUED))"
CPU=(--account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=72 --mem=200G
     --gres=none --no-requeue --time=04:00:00)
LOGS="$PROJECT_ROOT/runs/isambard"
ENV="cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
REPAIR="$ENV $PYTHON -B -m fastfill.comparison.output_repair --data $RUN/eval-data-v1 --workers 70 --max-trials 2000 --time-limit 60"
SUBMITTED=""
submit() {  # submit LABEL DEPENDENCY COMMAND; sets ID; no dependency = a held root. set -e misses $(...).
  local out=""
  local name=(--job-name="unknownsize-012-test-$1" --output="$LOGS/unknownsize-012-test-$1-%j.out")
  if test -n "$2"; then
    out=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="$2" "${CPU[@]}" "${name[@]}" --wrap "$3") || out=""
  else
    out=$(sbatch --parsable --hold "${CPU[@]}" "${name[@]}" --wrap "$3") || out=""
  fi
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
submit s1-J "" "$REPAIR --geometry $RUN/geometry-v5/joint-geometry.jsonl --requests-arm joint --out $S1/joint-sft"
submit s1-P "" "$REPAIR --geometry $RUN/geometry-v5/placement-geometry.jsonl --requests-arm placement --out $S1/placement-rpo-dpo1"
for NAME in "${AGENTS[@]}"; do
  for V in harness raw; do
    submit "s1-$NAME-$V" "" "$REPAIR --geometry $RUN/llm-eval-$NAME-selected-v1/llm_$V-geometry.jsonl --requests-arm joint --uids $RUN/llm-cohort-v1.json --out $S1/agent-$NAME-$V"
  done
done
ROOTS="$SUBMITTED"
submit summarize "afterok:${ROOTS// /:}" "$ENV $PYTHON -B -m fastfill.comparison.test_apply_012 summarize --data $RUN/eval-data-v1 --s1-root $S1 --cohort $RUN/llm-cohort-v1.json --decision $RUN/decision-012 --out $OUT --workers 70"
scontrol release "${ROOTS// /,}" || { echo "ABORT: could not release $ROOTS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOTS"
