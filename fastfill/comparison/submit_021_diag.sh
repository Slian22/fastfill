#!/bin/bash
#SBATCH --job-name=unknownsize-021-diag
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-021-diag-%j.out
# Amendment 021 sections 1.1-1.3 (descriptive diagnostics; they never gate section 2), run by the parent on the login
# node:   submit_021_diag.sh CODE_ROOT STEP...      STEP = ledger | pool | probe (each at most once)
# CODE_ROOT = a code snapshot (main-<sha>) holding the amendment-021 protocol doc and a021_diag.py. The steps of one call
# are submitted held and released together once all are in (an ABORT lists the ids for scancel; nothing has run):
#   ledger  CPU 72 / 192G / 8 h (the header)       a021_diag ledger -> RUN/diag-021/ledger: dpo-004 samples on
#           dpo-004/train-subset (+ the stage2-pairs ledger agreement if dpo-004/stage2-pairs exists when it runs)
#   pool    CPU 4 / 32G / 30 min                   a021_diag pool -> RUN/diag-021/pool: reads RUN/a021/pool, written by
#           `submit_021_train.sh CODE_ROOT setup` (a021_train.pool); submit it once RUN/a021/pool/MANIFEST.json exists.
#   probe   1 GPU / 8 CPU / 96G / 30 min           a021_diag probe -> RUN/diag-021/probe: rpo-dpo1's exact policy (HF)
#           on dev-select-v1 and its dev-eval-010 records; 0.5 GPU-h at most (amendment 021 section 5: probe + ledger
#           <= 1 GPU-h). The 8B policy needs neither flash-linear-attention nor PROJECT/pylib-021.
# Job names unknownsize-021-diag-STEP (counted by the train / eval GPU-h caps, which match ^unknownsize-021).
# This file is also the batch script: each job runs `submit_021_diag.sh CODE_ROOT job MODE` (PROJECT/env python).
# Refuses (nothing submitted) when a step's output exists, its input is missing, CODE_ROOT lacks the amendment-021 code
# or doc, or a job of that step of this user is in Slurm.
set -euo pipefail
usage() { echo "usage: submit_021_diag.sh CODE_ROOT {ledger|pool|probe}..." >&2; exit 2; }
test "$#" -ge 2 || usage
CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
OUT="$RUN/diag-021"
if test "$2" = job; then
  test "$#" -eq 3 || exit 2
  MODE="$3"
  GPUS=0
  case "$MODE" in
    ledger)
      LIMIT=470m
      PAIRS=()
      if test -f "$RUN/dpo-004/stage2-pairs/MANIFEST.json"; then PAIRS=(--pairs "$RUN/dpo-004/stage2-pairs"); fi
      ARGS=(ledger --data "$RUN/dpo-004/train-subset" --samples "$RUN/dpo-004/samples" ${PAIRS[@]+"${PAIRS[@]}"}
            --out "$OUT/ledger" --workers 72) ;;
    pool)
      LIMIT=25m
      ARGS=(pool --pool "$RUN/a021/pool" --out "$OUT/pool") ;;
    probe)
      LIMIT=28m GPUS=1
      ARGS=(probe --data "$RUN/dev-select-v1" --ref-root "$RUN/dev-eval-010" --out "$OUT/probe" --rooms 200
            --max-minutes 24) ;;
    *) exit 2 ;;
  esac
  : "${SLURM_JOB_ID:?Slurm allocation required}"
  export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=/tmp
  cd "$CODE_ROOT"
  if test "$GPUS" -gt 0; then  # As isambard_013.sbatch's HF shard jobs.
    module load cudatoolkit/24.11_12.6
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export CUDA_PATH="$CUDA_HOME" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    nvidia-smi -L
    SEEN=$(nvidia-smi -L | grep -c '^GPU ' || true)
    test "$SEEN" -ge "$GPUS" || { echo "PREFLIGHT FAIL: $MODE needs $GPUS GPU, sees $SEEN" >&2; exit 1; }
  else
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  fi
  exec timeout --signal=TERM --kill-after=60s "$LIMIT" "$PYTHON" -B -u -m fastfill.comparison.a021_diag "${ARGS[@]}"
fi
S="$CODE_ROOT/fastfill/comparison"
STEPS=("${@:2}")
NAMES=""
for step in "${STEPS[@]}"; do
  case "$step" in ledger|pool|probe) ;; *) usage ;; esac
  case " $NAMES " in *" unknownsize-021-diag-$step "*) usage ;; esac
  NAMES="$NAMES${NAMES:+ }unknownsize-021-diag-$step"
done
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
for f in a021_diag.py a021_train.py onpolicy_dpo.py dev_select.py submit_021_diag.sh; do
  test -f "$S/$f" || fail "missing $S/$f"
done
DOC=""
for path in "$CODE_ROOT"/docs/research/20261013-unknown-size-comparison/amendment-021-*.md; do
  if test -f "$path"; then DOC="$path"; fi
done
test -n "$DOC" || fail "no amendment-021 protocol doc in $CODE_ROOT"
for step in "${STEPS[@]}"; do
  case "$step" in
    ledger) INPUTS="dpo-004/train-subset/MANIFEST.json dpo-004/samples/MANIFEST.json" ;;
    pool) INPUTS="a021/pool/MANIFEST.json" ;;
    probe) INPUTS="dev-select-v1/MANIFEST.json dev-eval-010/rpo-dpo1/infer-placement/MANIFEST.json dpo-003/placement/dpo1-merged.json" ;;
  esac
  for f in $INPUTS; do test -f "$RUN/$f" || fail "missing $RUN/$f"; done
  test ! -e "$OUT/$step" || fail "$OUT/$step exists"
done
QUEUED=$(squeue -h -u "$USER" -n "${NAMES// /,}" -o %i) || fail "squeue failed"
test -z "$QUEUED" || fail "these diagnostics are already in Slurm (cancel with: scancel $QUEUED)"
SUBMITTED=""
submit() {  # submit STEP SBATCH_ARGS...: held, released below; set -e misses $(...).
  local out=""
  out=$(sbatch --parsable --hold --job-name="unknownsize-021-diag-$1" "${@:2}" "$S/submit_021_diag.sh" "$CODE_ROOT" job "$1") || out=""
  local id="${out%%;*}"
  case "$id" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$1'. Already submitted (held, never ran): ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$id"
  echo "$1 $id"
}
for step in "${STEPS[@]}"; do
  case "$step" in
    ledger) submit ledger ;;
    pool) submit pool --cpus-per-task=4 --mem=32G --time=00:30:00 ;;
    probe) submit probe --gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=00:30:00 ;;
  esac
done
scontrol release "${SUBMITTED// /,}" || { echo "ABORT: could not release; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $SUBMITTED"
