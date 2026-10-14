#!/bin/bash
# Amendment 011 chain, run by the parent on the Isambard login node:
#   submit_011.sh CODE_ROOT
# Every job is held until the whole chain is submitted (an ABORT lists the ids for scancel, so nothing ran);
# dependencies are afterok with --kill-on-invalid-dep=yes:
#   dev shard K (K = 0, 1; 1 GPU)   isambard_010_infer_shard.sbatch dev placement rpo-dpo1-merged K 2: the delivered
#                                   BF16 merge dpo-003/placement/dpo1-merged as is, on the shared dev plan (section 5)
#   dev merge (CPU)                 <- both shards: dev-eval-010/rpo-dpo1-merged/infer-placement
#   check-merged (CPU)              <- dev merge: dev_select check-merged -> dev-eval-010/merged-check
#   final (CPU, 16 cores, 64G, 12h) <- check-merged: final_011 from this one CODE_ROOT snapshot -> report-v2-sft-011,
#                                   report-v2-dpo1-011, report-v2-selected-011, delivery-011, stage-table-011
# Refuses if any output exists. Logs: PROJECT_ROOT/runs/isambard/unknownsize-010-{infer-shard,cpu}-JOB.out and
# unknownsize-011-final-JOB.out.
set -euo pipefail
test "$#" -eq 1 || { echo "usage: submit_011.sh CODE_ROOT" >&2; exit 2; }
CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
S="$CODE_ROOT/fastfill/comparison"
EVIDENCE="$CODE_ROOT/docs/research/20261013-unknown-size-comparison/evidence"
REFERENCE="$EVIDENCE/amendment-005-reference-rows-20261013.json"
CORRECTIONS="$EVIDENCE/amendment-006-corrections-uids-candidate-003.json"
for name in dev_select.py final_011.py report_v2.py stage_table.py delivery.py infer_shard.py \
    isambard_010_infer_shard.sbatch isambard_010_cpu.sbatch; do
  test -f "$S/$name" || { echo "ABORT: missing $S/$name" >&2; exit 1; }
done
test -f "$REFERENCE" -a -f "$CORRECTIONS" || { echo "ABORT: missing reference/corrections evidence" >&2; exit 1; }
INPUTS=("$RUN_ROOT/dev-eval-010/plan/MANIFEST.json" "$RUN_ROOT/dev-eval-010/selection.json"
        "$RUN_ROOT/dpo-003/placement/dpo1-merged.json" "$RUN_ROOT/dev-eval-010/scores-report-only/MANIFEST.json"
        "$RUN_ROOT/llm-open-status.json" "$RUN_ROOT/llm-agent-001/MANIFEST.json")
for name in L O6 O4 O2 O3 O7 O1a O1b; do  # final_011.AGENTS: both paired evaluations; open models' own runs.
  INPUTS+=("$RUN_ROOT/llm-eval-$name-v1/MANIFEST.json" "$RUN_ROOT/llm-eval-$name-selected-v1/MANIFEST.json")
  if test "$name" != L; then INPUTS+=("$RUN_ROOT/llm-open-$name-001/MANIFEST.json"); fi
done
for path in "${INPUTS[@]}"; do  # Everything final_011 reads, before any GPU time is spent.
  test -f "$path" || { echo "ABORT: missing input $path" >&2; exit 1; }
done
for name in dev-eval-010/rpo-dpo1-merged dev-eval-010/merged-check report-v2-sft-011 report-v2-dpo1-011 \
    report-v2-selected-011 delivery-011 stage-table-011; do
  if test -e "$RUN_ROOT/$name"; then echo "refusing to reuse existing $RUN_ROOT/$name" >&2; exit 1; fi
done
CPU=(--account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
     --gres=none --no-requeue)
ENV="cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
SUBMITTED=""
submit() {  # submit LABEL AFTEROK SBATCH_ARGS...; sets ID. set -e does not reach $(...), so check here.
  local out=""
  if test -n "$2"; then
    out=$(sbatch --parsable --hold --kill-on-invalid-dep=yes --dependency="afterok:$2" "${@:3}") || out=""
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
  echo "$1 $ID"
}
SHARDS=""
for i in 0 1; do
  submit "dev shard rpo-dpo1-merged placement $i" "" "$S/isambard_010_infer_shard.sbatch" "$CODE_ROOT" dev placement \
    rpo-dpo1-merged "$i" 2
  SHARDS="$SHARDS${SHARDS:+:}$ID"
done
submit "dev merge rpo-dpo1-merged placement" "$SHARDS" "$S/isambard_010_cpu.sbatch" "$CODE_ROOT" dev-merge placement \
  rpo-dpo1-merged 2
submit check-merged "$ID" "$S/isambard_010_cpu.sbatch" "$CODE_ROOT" check-merged
submit final "$ID" "${CPU[@]}" --time=12:00:00 --job-name=unknownsize-011-final \
  --output="$PROJECT_ROOT/runs/isambard/unknownsize-011-final-%j.out" \
  --wrap "$ENV $PYTHON -B -m fastfill.comparison.final_011 --run-root $RUN_ROOT --reference $REFERENCE --corrections $CORRECTIONS --repeats 2000 --workers 16"
scontrol release "${SUBMITTED// /,}" || { echo "ABORT: could not release; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $SUBMITTED"
