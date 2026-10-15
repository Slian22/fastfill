#!/bin/bash
# Amendment 026 addendum A (scope C, EXPLORATORY): L = gpt-6.1-sol with P's input (size-fixed, B2) at exactly the
# amendment-016 pilot settings (isambard_llm_agent.sbatch pilot-placement's, through its uids-* modes) on 300 test
# rooms, 20 per amendment-015 room family. Run by the user on the Isambard login node, smoke first:
#   submit_026_diverse.sh CODE_ROOT smoke   first 2 rooms, <= 6 calls -> RUN/llm-agent-P-diverse-smoke-001
#   submit_026_diverse.sh CODE_ROOT full    only after the smoke completed with no API_FAILED room:
#     run   RUN/llm-agent-P-diverse-001                 held root; <= 900 calls (llm_agent's --max-calls ledger cap)
#     eval  RUN/llm-eval-L-P-diverse-v1                 afterok run: evaluate_llm --uids RUN/a026-diverse-v1.json
#                                                       --uids-scope test; J = infer-joint-002, P = infer-placement-005
#     s1    RUN/s1-026/agent-L-P-diverse-{harness,raw}  afterok eval: output_repair at the amendment-012 settings
# Both modes first write or verify RUN/a026-diverse-v1.json (diverse_026 diverse) on the login node. Eval CPU 16 / 64G
# / 4 h, S1 CPU 72 / 200G / 4 h, as submit_016_agents.sh; dependent jobs are --kill-on-invalid-dep=yes. Refuses
# (nothing submitted) when an output exists, an input is missing, CODE_ROOT lacks the amendment-026 code, or an
# unknownsize-llm-agent / -026-diverse job is in Slurm. The API env file is read only by llm_agent inside the job.
set -euo pipefail
usage() { echo "usage: submit_026_diverse.sh CODE_ROOT smoke|full" >&2; exit 2; }
test "$#" -eq 2 || usage
CODE_ROOT="$1"
MODE="$2"
case "$MODE" in smoke|full) ;; *) usage ;; esac
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
S="$CODE_ROOT/fastfill/comparison"
UIDS=a026-diverse-v1.json
SMOKE=llm-agent-P-diverse-smoke-001
LLM=llm-agent-P-diverse-001
EVAL=llm-eval-L-P-diverse-v1
S1=s1-026/agent-L-P-diverse
CAP=900  # Addendum A: <= 900 paid calls; the ledger also counts transient retries, so it is the stricter bound.
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
if test "$MODE" = smoke; then OUTS=("$SMOKE"); else OUTS=("$LLM" "$EVAL" "$S1-harness" "$S1-raw"); fi
for out in "${OUTS[@]}"; do test ! -e "$RUN/$out" || fail "$RUN/$out exists"; done
for f in diverse_026.py llm_agent.py evaluate_llm.py output_repair.py isambard_llm_agent.sbatch; do
  test -f "$S/$f" || fail "missing $S/$f"
done
grep -q 'uids-placement)' "$S/isambard_llm_agent.sbatch" || fail "$S/isambard_llm_agent.sbatch has no uids-placement"
grep -q uids-scope "$S/llm_agent.py" && grep -q uids-scope "$S/evaluate_llm.py" || fail "$S has no --uids-scope test"
for f in eval-data-v1/MANIFEST.json llm-cohort-v1.json infer-joint-002/MANIFEST.json infer-placement-005/MANIFEST.json; do
  test -f "$RUN/$f" || fail "missing $RUN/$f"
done
if test "$MODE" = full; then
  "$PYTHON" -c 'import json, sys; r = json.load(open(sys.argv[1]))
sys.exit(r["status"] != "LLM_COMPLETE" or r["api_failed_rooms"] != 0)' "$RUN/$SMOKE/result.json" \
    || fail "the smoke $RUN/$SMOKE has not completed without API_FAILED rooms"
fi
# A queued API-agent job shares the API budget (and the smoke must finish before the full run).
QUEUED=$(squeue -h -u "$USER" -o '%i %j') || fail "squeue failed"
QUEUED=$(printf '%s\n' "$QUEUED" | awk '$2 ~ /^unknownsize-(llm-agent|026-diverse)/ {print $1}')
test -z "$QUEUED" || fail "unknownsize API-agent / 026-diverse jobs already in Slurm (scancel $(echo $QUEUED))"
(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -m fastfill.comparison.diverse_026 diverse \
  --data "$RUN/eval-data-v1" --out "$RUN/$UIDS") || fail "diverse_026 diverse ($RUN/$UIDS)"
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
B="$S/isambard_llm_agent.sbatch"
if test "$MODE" = smoke; then
  submit "L diverse smoke" --time=00:30:00 "$B" "$CODE_ROOT" uids-smoke "$UIDS" "$SMOKE"
  exit 0
fi
ENV="cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
LOGS="$PROJECT_ROOT/runs/isambard"
CPU=(--kill-on-invalid-dep=yes --account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --gres=none
     --no-requeue --time=04:00:00)
submit "L diverse" --hold --time=04:00:00 "$B" "$CODE_ROOT" uids-placement "$UIDS" "$LLM" "$CAP"
ROOT_ID="$ID"
submit "L diverse eval" "${CPU[@]}" --dependency="afterok:$ROOT_ID" --cpus-per-task=16 --mem=64G \
  --job-name=unknownsize-026-diverse-eval --output="$LOGS/unknownsize-026-diverse-eval-%j.out" \
  --wrap "$ENV $PYTHON -B -m fastfill.comparison.evaluate_llm --data $RUN/eval-data-v1 --cohort $RUN/llm-cohort-v1.json --llm $RUN/$LLM --joint $RUN/infer-joint-002 --placement $RUN/infer-placement-005 --out $RUN/$EVAL --workers 16 --uids $RUN/$UIDS --uids-scope test"
EVAL_ID="$ID"
for V in harness raw; do
  submit "L diverse s1-$V" "${CPU[@]}" --dependency="afterok:$EVAL_ID" --cpus-per-task=72 --mem=200G \
    --job-name=unknownsize-026-diverse-s1 --output="$LOGS/unknownsize-026-diverse-s1-$V-%j.out" \
    --wrap "$ENV $PYTHON -B -m fastfill.comparison.output_repair --data $RUN/eval-data-v1 --workers 70 --max-trials 2000 --time-limit 60 --geometry $RUN/$EVAL/llm_$V-geometry.jsonl --requests-arm placement --uids $RUN/$UIDS --out $RUN/$S1-$V"
done
scontrol release "$ROOT_ID" || { echo "ABORT: could not release $ROOT_ID; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOT_ID"
