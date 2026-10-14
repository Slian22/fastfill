#!/bin/bash
# Amendment 010 final reporting chain, run by the parent on the Isambard login node once the select job has
# written RUN/dev-eval-010/submission.json (the selected models' test chain ending in geometry-v5):
#   submit_final_010.sh CODE_ROOT [NAME=LLM_DIR[:JOBID] ...]
# Agents are re-paired with the two dev-selected models (amendment 010 addendum). Default (no agent arguments):
#   L=RUN/llm-agent-001 O6=RUN/llm-open-O6-001 O4=RUN/llm-open-O4-001 O2=RUN/llm-open-O2-001
#   O3=RUN/llm-open-O3-001 O7=RUN/llm-open-O7-001 O1a=RUN/llm-open-O1a-001 O1b=RUN/llm-open-O1b-001
#   O5=RUN/llm-open-O5-001
# LLM_DIR is an absolute RUN/llm-* path. Append :JOBID (the agent's full generation job) while that job is in
# Slurm: its evaluation then waits for it (afterok; killed if it fails). An agent without :JOBID must already
# be complete (LLM_DIR/MANIFEST.json), else nothing is submitted.
# CPU jobs (16 cores, 64G, no GPU), each held until the whole chain is submitted (an ABORT lists the ids for
# scancel, so nothing ran):
#   eval NAME   <- afterok geometry-v5 job (submission.json 'selected geometry'; dropped once RUN/geometry-v5 is
#                  published) [+ afterok JOBID]: evaluate_llm on RUN/llm-cohort-v1.json with the selected J/P test
#                  runs (selection.json test.*.inference) -> RUN/llm-eval-NAME-selected-v1
#   final       <- afterany every eval, afterok geometry-v5: dev_select final = agents whose evaluation is
#                  complete (the others logged with the reason, never substituted), report_v2 stage selected ->
#                  RUN/report-v2-selected-001, stage_table (SFT report-v2-JPL-002, DPO1 report-v2-dpo1-001,
#                  selected, dev table) -> RUN/stage-table-010-001, delivery -> RUN/delivery-010-001
# Refuses if any output dir exists. Logs: PROJECT_ROOT/runs/isambard/unknownsize-010-{eval-NAME,final}-JOB.out.
set -euo pipefail
test "$#" -ge 1
CODE_ROOT="$1"
shift
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
S="$CODE_ROOT/fastfill/comparison"
EVIDENCE="$CODE_ROOT/docs/research/20261013-unknown-size-comparison/evidence"
REFERENCE="$EVIDENCE/amendment-005-reference-rows-20261013.json"
CORRECTIONS="$EVIDENCE/amendment-006-corrections-uids-candidate-003.json"
for name in dev_select.py evaluate_llm.py report_v2.py stage_table.py delivery.py; do
  test -f "$S/$name"
done
test -f "$REFERENCE" -a -f "$CORRECTIONS"
if test "$#" -eq 0; then
  set -- "L=$RUN_ROOT/llm-agent-001"
  for NAME in O6 O4 O2 O3 O7 O1a O1b O5; do set -- "$@" "$NAME=$RUN_ROOT/llm-open-$NAME-001"; done
fi
READ_CHAIN='
import hashlib, json, sys
root = sys.argv[1] + "/dev-eval-010/"
submission = json.load(open(root + "submission.json"))
selection = open(root + "selection.json", "rb").read()
if submission["selection_sha256"] != hashlib.sha256(selection).hexdigest():
    sys.exit("submission.json does not bind selection.json")
test = json.loads(selection)["test"]
print(*[j["id"] for j in submission["jobs"] if j["label"] == "selected geometry"],
      test["joint"]["inference"], test["placement"]["inference"])'
CHAIN=$("$PYTHON" -I -B -c "$READ_CHAIN" "$RUN_ROOT") || { echo "ABORT: cannot read $RUN_ROOT/dev-eval-010" >&2; exit 1; }
read -r GEOMETRY JOINT PLACEMENT EXTRA <<< "$CHAIN"
case "$GEOMETRY" in ''|*[!0-9]*) echo "ABORT: no 'selected geometry' job id in submission.json" >&2; exit 1 ;; esac
case "$JOINT" in "$RUN_ROOT"/infer-joint-00[25]) ;; *) echo "ABORT: unexpected J run $JOINT" >&2; exit 1 ;; esac
case "$PLACEMENT" in "$RUN_ROOT"/infer-placement-00[25]) ;; *) echo "ABORT: unexpected P run $PLACEMENT" >&2; exit 1 ;; esac
test -z "${EXTRA:-}"
AFTER_GEOMETRY="$GEOMETRY"
if test -e "$RUN_ROOT/geometry-v5/MANIFEST.json"; then AFTER_GEOMETRY=""; fi  # Published; the job may have left Slurm.
NAMES=()  # Parallel arrays: bash 3 (macOS test host) has no associative arrays.
DIRS=()
JOBS=()
OUTS=("$RUN_ROOT/report-v2-selected-001" "$RUN_ROOT/stage-table-010-001" "$RUN_ROOT/delivery-010-001")
SEEN=" "
for SPEC in "$@"; do
  NAME="${SPEC%%=*}"
  REST="${SPEC#*=}"
  DIR="${REST%%:*}"
  JOB=""
  if test "$DIR" != "$REST"; then JOB="${REST#*:}"; fi
  case "$NAME" in L|O1a|O1b|O2|O3|O4|O5|O6|O7) ;; *) echo "unknown agent in $SPEC" >&2; exit 2 ;; esac
  case "$SEEN" in *" $NAME "*) echo "duplicate agent $NAME" >&2; exit 2 ;; esac
  case "$DIR" in "$RUN_ROOT"/llm-*) ;; *) echo "agent dir must be $RUN_ROOT/llm-*: $SPEC" >&2; exit 2 ;; esac
  case "$JOB" in
    '') if ! test -e "$DIR/MANIFEST.json"; then
          echo "ABORT: $NAME generation not complete ($DIR/MANIFEST.json missing); pass $NAME=$DIR:JOBID" >&2; exit 1
        fi ;;
    *[!0-9]*) echo "bad job id in $SPEC" >&2; exit 2 ;;
  esac
  SEEN="$SEEN$NAME "
  NAMES+=("$NAME")
  DIRS+=("$DIR")
  JOBS+=("$JOB")
  OUTS+=("$RUN_ROOT/llm-eval-$NAME-selected-v1")
done
for path in "${OUTS[@]}"; do
  if test -e "$path"; then echo "refusing to reuse existing $path" >&2; exit 1; fi
done
CPU=(--account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
     --gres=none --no-requeue --kill-on-invalid-dep=yes)
LOGS="$PROJECT_ROOT/runs/isambard"
ENV="cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
SUBMITTED=""
submit() {  # submit LABEL DEPENDENCY SBATCH_ARGS...; sets ID. set -e does not reach $(...), so check here.
  local out=""
  if test -n "$2"; then
    out=$(sbatch --parsable --hold --dependency="$2" "${CPU[@]}" "${@:3}") || out=""
  else
    out=$(sbatch --parsable --hold "${CPU[@]}" "${@:3}") || out=""
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
EVALS=""
FINAL_AGENTS=""
for k in "${!NAMES[@]}"; do
  AFTER="$AFTER_GEOMETRY"
  if test -n "${JOBS[$k]}"; then AFTER="$AFTER${AFTER:+:}${JOBS[$k]}"; fi
  OUT="$RUN_ROOT/llm-eval-${NAMES[$k]}-selected-v1"
  submit "eval ${NAMES[$k]}" "${AFTER:+afterok:$AFTER}" --time=04:00:00 --job-name=unknownsize-010-eval \
    --output="$LOGS/unknownsize-010-eval-${NAMES[$k]}-%j.out" \
    --wrap "$ENV $PYTHON -B -m fastfill.comparison.evaluate_llm --data $RUN_ROOT/eval-data-v1 --cohort $RUN_ROOT/llm-cohort-v1.json --llm ${DIRS[$k]} --joint $JOINT --placement $PLACEMENT --out $OUT --workers 16"
  EVALS="$EVALS:$ID"
  FINAL_AGENTS="$FINAL_AGENTS --agent ${NAMES[$k]}=$OUT"
done
submit final "afterany$EVALS${AFTER_GEOMETRY:+,afterok:$AFTER_GEOMETRY}" --time=12:00:00 \
  --job-name=unknownsize-010-final --output="$LOGS/unknownsize-010-final-%j.out" \
  --wrap "$ENV $PYTHON -B -m fastfill.comparison.dev_select final --run-root $RUN_ROOT --reference $REFERENCE --corrections $CORRECTIONS$FINAL_AGENTS --repeats 2000 --workers 16"
scontrol release "${SUBMITTED// /,}" || { echo "ABORT: could not release; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $SUBMITTED"
