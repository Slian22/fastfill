#!/bin/bash
# S2 best-of-N chain (amendment-011 draft), run by the parent on the Isambard login node:
#   submit_sample_vllm.sh CODE_ROOT [SHARDS=4]
# build      CPU, only when RUN/dev-full-v1 is missing: sample_vllm build (all 8,167 dev rooms)
# per arm (joint = dev-selected sft, placement = rpo-dpo1), each afterok its predecessor, --kill-on-invalid-dep=yes:
#   smoke    1 GPU, 2 h, --limit 16 (the 16 largest rooms)         <- build (or held when the data exists)
#   array    SHARDS x 1 GPU, tensor_parallel 1, shard = task id    <- smoke
#   finalize CPU: sample_vllm finalize -> s2-011/ARM-CAND/samples  <- array
#   select   CPU 32: best_of_n select (--reference dev-eval-010/CAND/infer-ARM) -> s2-011/ARM-CAND/best-of-n
# Root jobs are held until the whole chain is in, so an ABORT (it lists the ids for scancel) leaves nothing that ran.
# Refuses while any unknownsize-s2-* job of this user is in Slurm, once an arm is finalized or selected (run the
# remaining step by hand), or when a select --reference dir is missing. A rerun after a failed or timed-out job
# resumes the samples from receipts.
set -euo pipefail
test "$#" -ge 1 -a "$#" -le 2 || exit 2
CODE_ROOT="$1"
K="${2:-4}"
case "$K" in ''|*[!0-9]*) exit 2 ;; esac
test "$K" -ge 1 || exit 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
DATA="$RUN_ROOT/dev-full-v1"
S="$CODE_ROOT/fastfill/comparison"
for name in sample_vllm.py best_of_n.py isambard_sample_vllm.sbatch; do test -f "$S/$name"; done
for ref in sft/infer-joint rpo-dpo1/infer-placement; do
  test -f "$RUN_ROOT/dev-eval-010/$ref/MANIFEST.json" || {
    echo "ABORT: missing select --reference $RUN_ROOT/dev-eval-010/$ref; nothing submitted" >&2; exit 1; }
done
QUEUED=$(squeue -h -u "$USER" -n unknownsize-s2-sample,unknownsize-s2-cpu -o %i) || { echo "ABORT: squeue failed" >&2; exit 1; }
if test -n "$QUEUED"; then
  echo "ABORT: S2 jobs already in Slurm; nothing submitted. cancel with: scancel" $QUEUED >&2
  exit 1
fi
for pair in joint-sft placement-rpo-dpo1; do
  for path in "$RUN_ROOT/s2-011/$pair/best-of-n" "$RUN_ROOT/s2-011/$pair/samples/MANIFEST.json"; do
    if test -e "$path"; then echo "refusing: $path exists; run the remaining step by hand" >&2; exit 1; fi
  done
done
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
cpu() {  # cpu LABEL DEPENDENCY CPUS MEM TIME MODULE ARGS...  (env python, no GPU)
  local label="$1" after="$2" cpus="$3" mem="$4" limit="$5"
  shift 5
  submit "$label" "$after" --job-name=unknownsize-s2-cpu --account=brics.u6uv --partition=workq --qos=normal \
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem" --gres=none --time="$limit" --no-requeue \
    --output="$PROJECT_ROOT/runs/isambard/unknownsize-s2-cpu-%j.out" \
    --wrap "cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 $PYTHON -B -u -m $*"
}
BUILD=""
if ! test -e "$DATA"; then
  cpu "build dev-full-v1" "" 4 96G 02:00:00 fastfill.comparison.sample_vllm build --data "$RUN_ROOT/main-data-v1" --out "$DATA"
  BUILD="$ID"
fi
for ARM in joint placement; do
  case "$ARM" in joint) CAND=sft ;; *) CAND=rpo-dpo1 ;; esac
  OUT="$RUN_ROOT/s2-011/$ARM-$CAND/samples"
  submit "$ARM smoke" "$BUILD" --time=02:00:00 "$S/isambard_sample_vllm.sbatch" "$CODE_ROOT" dev "$ARM" 1 --limit 16
  submit "$ARM array" "$ID" --array=0-$((K - 1)) "$S/isambard_sample_vllm.sbatch" "$CODE_ROOT" dev "$ARM" "$K"
  cpu "$ARM finalize" "$ID" 4 32G 01:00:00 fastfill.comparison.sample_vllm finalize --data "$DATA" --out "$OUT"
  cpu "$ARM select" "$ID" 32 96G 06:00:00 fastfill.comparison.best_of_n select --data "$DATA" --samples "$OUT" \
    --arm "$ARM" --out "$RUN_ROOT/s2-011/$ARM-$CAND/best-of-n" --workers 32 \
    --reference "$RUN_ROOT/dev-eval-010/$CAND/infer-$ARM"
done
scontrol release "$ROOTS" || { echo "ABORT: could not release $ROOTS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOTS"
