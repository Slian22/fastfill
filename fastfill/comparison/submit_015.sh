#!/bin/bash
#SBATCH --job-name=optiscene-015
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/optiscene-015-%j.out
# Amendment 015 E3 (EXPLORATORY): the amendment-014 OptiScene checkpoints and the 8B on our test rooms of the room
# families outside the paper's four (optiscene_eval plan-e3). Run by the user on the Isambard login node:
#   submit_015.sh CODE_ROOT MERGE_SFT MERGE_DPO1 MERGE_DPO2
# MERGE_X = the amendment-014 merge-X job id (dependency while it is still in Slurm) or '-'. The same file is the job
# script (CODE_ROOT job plan-e3 | gen sft|dpo1|dpo2|8b | score-e3; inside a Slurm allocation only).
# Preflight, nothing submitted on failure: eval-data-v1 / geometry-v5 / dev-eval-010/selection.json, the uploaded
# plan-e1 (hf-coverage.json), the official data, the 8B model, both pythons, CODE_ROOT importing room_families.family
# and optiscene_eval.plan_e3, no optiscene-015 job in Slurm, nothing at RUN/optiscene-015.
# Jobs (each --kill-on-invalid-dep=yes; the root is held until the whole chain is in, then released):
#   plan-e3                                                     CPU 16 / 64G, 2 h   root
#   gen-X     afterok plan-e3, X = sft|dpo1|dpo2: only when RUN/optiscene-014/X-merged is published, or its merge
#             job (Slurm name optiscene-014-merge-X) is still in Slurm (then also afterok that merge); otherwise
#             skipped and listed                                1 GPU, 2 h
#   gen-8b    afterok plan-e3: only when amendment 014's 8B dev smoke is published (checked again at run time);
#             otherwise skipped and listed                      1 GPU, 2 h
# Arms are decided before anything is submitted; no arm: nothing submitted.
#   score-e3  afterany every gen: scores the published arms, lists the others          CPU 16 / 64G, 4 h
# GPU cap: 4 x 2 = 8 GPU-h. On any sbatch failure: ABORT with the ids to scancel.
set -euo pipefail
test "$#" -ge 2 || exit 2
CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
OUT="$RUN/optiscene-015"
OUT14="$RUN/optiscene-014"
S="$CODE_ROOT/fastfill/comparison"
PY="$PROJECT_ROOT/env/bin/python"
VLLM_PY=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
DATA="$PROJECT_ROOT/data/optiscene-official"
E1="$PROJECT_ROOT/data/optiscene-014-e1"
EIGHT_B="$PROJECT_ROOT/models/OptiScene-Qwen-8B-Final"
OE=(-B -u -m fastfill.comparison.optiscene_eval)

if test "$2" = job; then
  case "${3:-}:$#" in
    plan-e3:3|score-e3:3) ;;
    gen:4) case "$4" in sft|dpo1|dpo2|8b) ;; *) exit 2 ;; esac ;;
    *) exit 2 ;;
  esac
  : "${SLURM_JOB_ID:?Slurm allocation required}"
  CACHE="$PROJECT_ROOT/.cache-jobs/optiscene-015"
  export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
  export HF_HOME="$PROJECT_ROOT/.cache-jobs/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1 XDG_CACHE_HOME="$CACHE/xdg" TRITON_CACHE_DIR="$CACHE/triton"
  export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" VLLM_CACHE_ROOT="$CACHE/vllm" TMPDIR=/tmp
  mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
  cd "$CODE_ROOT"
  case "$3" in
    plan-e3)
      exec "$PY" "${OE[@]}" plan-e3 --eval-data "$RUN/eval-data-v1" --data "$DATA" --e1 "$E1" --out "$OUT/e3" ;;
    score-e3)
      P_RUN=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["test"]["placement"]["inference"])' \
        "$RUN/dev-eval-010/selection.json")
      exec "$PY" "${OE[@]}" score-e3 --e3 "$OUT/e3" --gen "$OUT/gen" --eval-data "$RUN/eval-data-v1" \
        --p-geometry "$RUN/geometry-v5" --p-run "$P_RUN" --out "$OUT/score-e3" --workers 16 ;;
  esac
  if test "$4" = 8b; then
    test -f "$OUT14/gen/8b-dev-smoke-v1/MANIFEST.json" || {
      echo "amendment 014's 8B dev smoke did not pass (no $OUT14/gen/8b-dev-smoke-v1): 8B not run" >&2; exit 1; }
    MODEL="$EIGHT_B" PROMPT=v1
  else
    MODEL="$OUT14/$4-merged" PROMPT=sft
  fi
  # vLLM JIT kernels need an nvcc matching torch cu130: prefer the env's pip CUDA 13 toolkit, else the module.
  PIP_CUDA="$(dirname "$(dirname "$VLLM_PY")")/lib/python3.11/site-packages/nvidia/cu13"
  if test -x "$PIP_CUDA/bin/nvcc"; then
    CUDA_HOME="$PIP_CUDA"
  else
    module load cudatoolkit || { echo "PREFLIGHT FAIL: module load cudatoolkit" >&2; exit 1; }
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
  export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$(dirname "$VLLM_PY"):$CUDA_HOME/bin:$PATH"
  export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
  exec "$VLLM_PY" "${OE[@]}" generate --plan "$OUT/e3" --prompt "$PROMPT" --model "$MODEL" \
    --out "$OUT/gen/$4-e3-$PROMPT"
fi

test "$#" -eq 4 || exit 2
MERGES=("$2" "$3" "$4")
for id in "${MERGES[@]}"; do case "$id" in -) ;; ''|*[!0-9]*) exit 2 ;; esac; done
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
for f in optiscene_eval.py room_families.py submit_015.sh; do test -f "$S/$f" || fail "missing $S/$f"; done
for f in eval-data-v1/MANIFEST.json geometry-v5/MANIFEST.json dev-eval-010/selection.json; do
  test -f "$RUN/$f" || fail "missing $RUN/$f"
done
for f in "$E1/MANIFEST.json" "$E1/hf-coverage.json" "$DATA/sft_prompts.json" "$DATA/dpo_prompts.json" \
         "$DATA/dpo2_prompts.json" "$EIGHT_B/config.json"; do
  test -f "$f" || fail "missing $f"
done
test ! -e "$OUT" || fail "$OUT exists"
for py in "$PY" "$VLLM_PY"; do test -x "$py" || fail "no executable $py"; done
PYTHONPATH="$CODE_ROOT" "$PY" -B -c \
  'from fastfill.comparison.room_families import family; from fastfill.comparison.optiscene_eval import plan_e3' \
  || fail "$CODE_ROOT cannot import room_families.family and optiscene_eval.plan_e3"
QUEUED=$(squeue -h -u "$USER" -o '%i %j') || fail "squeue failed"
QUEUED=$(printf '%s\n' "$QUEUED" | awk '$2 ~ /^optiscene-015/ {print $1}')
test -z "$QUEUED" || fail "optiscene-015 jobs already in Slurm (scancel $(echo $QUEUED))"
ARMS=()  # STAGE:[MERGE:] per submitted gen
STAGES=(sft dpo1 dpo2)
for k in 0 1 2; do
  STAGE="${STAGES[$k]}" MERGE="${MERGES[$k]}"
  if test -f "$OUT14/$STAGE-merged/MANIFEST.json"; then
    ARMS+=("$STAGE:")
  elif test "$MERGE" != - \
      && test "$(squeue -h -j "$MERGE" -o %j 2>/dev/null || true)" = "optiscene-014-merge-$STAGE"; then
    ARMS+=("$STAGE:$MERGE:")
  else
    echo "skip gen-$STAGE: $OUT14/$STAGE-merged is not published and '$MERGE' is no optiscene-014-merge-$STAGE in Slurm"
  fi
done
if test -f "$OUT14/gen/8b-dev-smoke-v1/MANIFEST.json"; then
  ARMS+=("8b:")
else
  echo "skip gen-8b: amendment 014's 8B dev smoke is not published ($OUT14/gen/8b-dev-smoke-v1)"
fi
test "${#ARMS[@]}" -gt 0 || fail "no arm to generate"
SUBMITTED=""
submit() {  # submit LABEL DEPENDENCY SBATCH_ARGS...; sets ID; no dependency = the held root. set -e misses $(...).
  local label="$1" dep="$2" out=""
  shift 2
  if test -n "$dep"; then
    out=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="$dep" --job-name="optiscene-015-$label" "$@") || out=""
  else
    out=$(sbatch --parsable --hold --job-name="optiscene-015-$label" "$@") || out=""
  fi
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$label'; chain NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  echo "$label $ID"
}
B="$S/submit_015.sh"
CPU=(--cpus-per-task=16 --mem=64G)
GPU=(--gpus-per-node=1 --cpus-per-task=16 --mem=110G)
submit plan-e3 "" "${CPU[@]}" --time=02:00:00 "$B" "$CODE_ROOT" job plan-e3
PLAN="$ID"
GENS=""
for ARM in "${ARMS[@]}"; do
  submit "gen-${ARM%%:*}" "afterok:${ARM#*:}$PLAN" "${GPU[@]}" --time=02:00:00 "$B" "$CODE_ROOT" job gen "${ARM%%:*}"
  GENS="$GENS:$ID"
done
submit score-e3 "afterany$GENS" "${CPU[@]}" --time=04:00:00 "$B" "$CODE_ROOT" job score-e3
scontrol release "$PLAN" || { echo "ABORT: could not release $PLAN; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $PLAN"
