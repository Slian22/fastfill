#!/bin/bash
#SBATCH --job-name=placers-020
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/placers-020-%j.out
# Amendment 020 §3-§4 (EXPLORATORY): the placers OS-DPO2 and P given the llm-L / llm-O2 sizes on E2 ∪ E3, scored with
# prior / gt / llm sizes, and Table B. Run by the user on the Isambard login node:
#   submit_020_placers.sh CODE_ROOT [SIZERS_JOB]
# SIZERS_JOB = the submit_020_sizers.sh job ids to wait for, colon-separated: its report job (unknownsize-020-report,
# which finalizes RUN/sizes-020-O2 after L-size and every O2 shard; never the L-size job itself, which exits 1 when
# refused). Still in Slurm: afterok; ended: must be COMPLETED.
# MODES: llm-O2 always; llm-L unless the L sizer is known refused: RUN/sizes-020-L/status.json is CAP_REFUSED, or no
# RUN/sizes-020-L yet and the API cap preflight (sizers_020 cap, 1,877 rooms) refuses. Amendment 020 §2 locks that cap
# rule, and 1,877 x 3 = 5,631 > 4,800, so llm-L is dropped unless the cap changes. Without SIZERS_JOB every kept MODE's
# RUN/sizes-020-X must be published.
# The same file is the job script (CODE_ROOT job plan MODE | os-gen MODE | p-shard MODE K | p-merge MODE | score PLACER
# MODE | report; inside a Slurm allocation only). Every step but report skips an output that is already published;
# report writes OUT/report-JOBID each time (its summary lists the arms not published).
# Preflight, nothing submitted on failure: inputs and pythons, CODE_ROOT importing placers_020 and optiscene_eval, no
# placers-020 job in Slurm, the sizer jobs, and the amendment-020 GPU cap (placers_020 budget: the sizers = max(sacct of
# unknownsize-020-sizer-* jobs, the receipts' gpu_hours), earlier placers-020 jobs (sacct), + this chain's GPU time
# limits <= 30 GPU-h; a pending or running job counts its whole time limit). A GPU step whose output is published is
# neither submitted nor planned, so a rerun after a partial failure plans only what is left.
# Jobs (each --kill-on-invalid-dep=yes; the roots (plan-MODE and the prior/gt scores) are held until the whole chain is
# in, then released; a failed MODE cancels only its own jobs):
#   plan-MODE       [afterok sizers]  optiscene_eval plan-e2/plan-e3 --size-mode MODE -> OUT/plan-{e2,e3}-MODE,
#                   (their rooms must equal the 014/015 plans'), placers_020 p-data -> OUT/p-data-MODE, infer_shard plan
#                   (amendment-002 decode) -> OUT/p-MODE/plan
#                                                                                      CPU 16 / 64G, 2 h   root
#   os-gen-MODE     afterok plan-MODE [+ the optiscene-014-merge-dpo2 job while it is in Slurm]: generate (prompt sft) E2 and
#                   E3 with OUT14/dpo2-merged -> OUT/gen/dpo2-{e2,e3}-MODE-sft; skipped (and its scores) when dpo2-merged
#                   is neither published nor being merged                              1 GPU, 1 h
#   p-shard-MODE-K  afterok plan-MODE: infer_shard shard, rpo-dpo1 exact policy (dpo-003/placement/dpo1-merged
#                   --exact-policy), K < 6; resumes a started shard                   1 GPU, 1 h
#   p-merge-MODE    afterok its shards -> OUT/p-MODE/infer                            CPU 16 / 64G, 1 h
#   score-PLACER-MODE  -> OUT/rows/PLACER-MODE                                        CPU 32 / 128G, 4 h
#                   os-prior/os-gt (roots) afterany optiscene-014-gen-dpo2 / optiscene-015-gen-dpo2 while in Slurm (their
#                   E2/E3 generations); os-MODE afterok os-gen-MODE (plan-MODE if published); p-prior (root; geometry-v5
#                   rows); p-MODE afterok p-merge-MODE
#   report          afterany every score -> OUT/report-JOBID                          CPU 16 / 64G, 4 h
# GPU per MODE: 1 + 6 x 1 = 7 GPU-h by time limits (both MODES 14; with the sizers' 16 GPU-h ceiling: 30). A shard that
# times out is resumed by rerunning this script once the chain has left Slurm (only the unpublished steps are submitted
# and planned), or by hand: sbatch --job-name=placers-020-p-shard-MODE-K GPU_FLAGS THIS CODE_ROOT job p-shard MODE K,
# then the p-merge, score and report steps the same way. On any sbatch failure: ABORT with the ids to scancel.
set -euo pipefail
test "$#" -ge 1 || exit 2
CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
OUT="$RUN/optiscene-020"
OUT14="$RUN/optiscene-014"
OUT15="$RUN/optiscene-015"
S="$CODE_ROOT/fastfill/comparison"
PY="$PROJECT_ROOT/env/bin/python"
VLLM_PY=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
DATA="$PROJECT_ROOT/data/optiscene-official"
E1="$PROJECT_ROOT/data/optiscene-014-e1"
BASE="$PROJECT_ROOT/models/Qwen3-8B"
P_MODEL="$RUN/dpo-003/placement/dpo1-merged"
SHARDS=6
CAP=30
OE=(-B -u -m fastfill.comparison.optiscene_eval)
P20=(-B -u -m fastfill.comparison.placers_020)
IS=(-B -u -m fastfill.comparison.infer_shard)

if test "${2:-}" = job; then
  case "${3:-}:$#" in
    report:3) ;;
    plan:4|os-gen:4|p-merge:4) case "$4" in llm-L|llm-O2) ;; *) exit 2 ;; esac ;;
    p-shard:5) case "$4:$5" in llm-L:[0-5]|llm-O2:[0-5]) ;; *) exit 2 ;; esac ;;
    score:5) case "$4:$5" in os:prior|os:gt|os:llm-L|os:llm-O2|p:prior|p:llm-L|p:llm-O2) ;; *) exit 2 ;; esac ;;
    *) exit 2 ;;
  esac
  : "${SLURM_JOB_ID:?Slurm allocation required}"
  CACHE="$PROJECT_ROOT/.cache-jobs/placers-020"
  export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
  export HF_HOME="$PROJECT_ROOT/.cache-jobs/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true XDG_CACHE_HOME="$CACHE/xdg" TRITON_CACHE_DIR="$CACHE/triton"
  export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" VLLM_CACHE_ROOT="$CACHE/vllm" TMPDIR=/tmp
  mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
  cd "$CODE_ROOT"
  done_() { if test -f "$1/MANIFEST.json"; then echo "skip $1 (published)"; return 0; fi; return 1; }
  case "$3" in
    plan)
      M="$4"
      X="${M#llm-}"
      for E in e2 e3; do
        done_ "$OUT/plan-$E-$M" || "$PY" "${OE[@]}" "plan-$E" --eval-data "$RUN/eval-data-v1" --data "$DATA" \
          --e1 "$E1" --out "$OUT/plan-$E-$M" --size-mode "$M" --estimates "$RUN/sizes-020-$X"
      done
      "$PY" -B -c 'import sys; from fastfill.comparison.placers_020 import universe as u
sys.exit(0 if u(*sys.argv[1:3]) == u(*sys.argv[3:]) else "E2/E3 rooms differ from the amendment-014/015 plans")' \
        "$OUT/plan-e2-$M" "$OUT/plan-e3-$M" "$OUT14/e2" "$OUT15/e3"
      done_ "$OUT/p-data-$M" || "$PY" "${P20[@]}" p-data --eval-data "$RUN/eval-data-v1" --e2 "$OUT/plan-e2-$M" \
        --e3 "$OUT/plan-e3-$M" --estimates "$RUN/sizes-020-$X" --mode "$M" --out "$OUT/p-data-$M"
      done_ "$OUT/p-$M/plan" || "$PY" "${IS[@]}" plan --data "$OUT/p-data-$M" --base "$BASE" --out "$OUT/p-$M/plan" \
        --batch-size 8 --max-input-tokens 40959 --max-new-tokens 40959 --context-budget 40960 ;;
    os-gen)
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
      for E in e2 e3; do
        done_ "$OUT/gen/dpo2-$E-$4-sft" || "$VLLM_PY" "${OE[@]}" generate --plan "$OUT/plan-$E-$4" --prompt sft \
          --model "$OUT14/dpo2-merged" --out "$OUT/gen/dpo2-$E-$4-sft"
      done ;;
    p-shard)
      D="$OUT/p-$4/shard-0$5"
      done_ "$D" && exit 0
      RESUME=()
      if test -e "$D"; then RESUME=(--resume); fi
      module load cudatoolkit/24.11_12.6
      export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
      export CUDA_PATH="$CUDA_HOME" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
      exec timeout --signal=TERM --kill-after=60s 55m "$PY" "${IS[@]}" shard --data "$OUT/p-data-$4" --base "$BASE" \
        --arm placement --model "$P_MODEL" --exact-policy --plan "$OUT/p-$4/plan" --shard-index "$5" \
        --shard-count "$SHARDS" --out "$D" ${RESUME[@]+"${RESUME[@]}"} ;;
    p-merge)
      done_ "$OUT/p-$4/infer" && exit 0
      SH=()
      for ((k = 0; k < SHARDS; k++)); do SH+=("$OUT/p-$4/shard-0$k"); done
      exec "$PY" "${IS[@]}" merge --data "$OUT/p-data-$4" --plan "$OUT/p-$4/plan" --arm placement \
        --out "$OUT/p-$4/infer" --shards "${SH[@]}" ;;
    score)
      done_ "$OUT/rows/$4-$5" && exit 0
      P_RUN=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["test"]["placement"]["inference"])' \
        "$RUN/dev-eval-010/selection.json")
      case "$4:$5" in
        os:prior|os:gt) ARGS=(--e2 "$OUT14/e2" --e3 "$OUT15/e3" --gen-e2 "$OUT14/gen/dpo2-e2-sft"
                              --gen-e3 "$OUT15/gen/dpo2-e3-sft") ;;
        os:*) ARGS=(--e2 "$OUT/plan-e2-$5" --e3 "$OUT/plan-e3-$5" --gen-e2 "$OUT/gen/dpo2-e2-$5-sft"
                    --gen-e3 "$OUT/gen/dpo2-e3-$5-sft" --p-data "$OUT/p-data-$5") ;;
        p:prior) ARGS=(--e2 "$OUT14/e2" --e3 "$OUT15/e3") ;;
        p:*) ARGS=(--e2 "$OUT/plan-e2-$5" --e3 "$OUT/plan-e3-$5" --p-data "$OUT/p-data-$5" --p-infer "$OUT/p-$5/infer") ;;
      esac
      exec "$PY" "${P20[@]}" score --placer "$4" --mode "$5" --eval-data "$RUN/eval-data-v1" \
        --p-geometry "$RUN/geometry-v5" --p-run "$P_RUN" --out "$OUT/rows/$4-$5" --workers 32 "${ARGS[@]}" ;;
    report)
      exec "$PY" "${P20[@]}" report --rows "$OUT/rows" --out "$OUT/report-$SLURM_JOB_ID" --repeats 2000 ;;
  esac
  exit 0
fi

test "$#" -le 2 || exit 2
SIZERS="${2:-}"
case "$SIZERS" in '') ;; *[!0-9:]*|:*|*:|*::*) exit 2 ;; esac
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
for f in optiscene_eval.py placers_020.py infer_shard.py submit_020_placers.sh; do test -f "$S/$f" || fail "missing $S/$f"; done
for f in eval-data-v1/MANIFEST.json geometry-v5/MANIFEST.json dev-eval-010/selection.json optiscene-014/e2/MANIFEST.json \
         optiscene-015/e3/MANIFEST.json; do
  test -f "$RUN/$f" || fail "missing $RUN/$f"
done
for f in "$E1/MANIFEST.json" "$E1/hf-coverage.json" "$DATA/sft_prompts.json" "$DATA/dpo_prompts.json" \
         "$DATA/dpo2_prompts.json" "$BASE/config.json" "$P_MODEL.json"; do
  test -f "$f" || fail "missing $f"
done
for py in "$PY" "$VLLM_PY"; do test -x "$py" || fail "no executable $py"; done
PYTHONPATH="$CODE_ROOT" "$PY" -B -c 'from fastfill.comparison import optiscene_eval, placers_020' \
  || fail "$CODE_ROOT cannot import placers_020 and optiscene_eval"
QUEUED=$(squeue -h -u "$USER" -o '%i %j') || fail "squeue failed"
JOBS="$QUEUED"
QUEUED=$(printf '%s\n' "$QUEUED" | awk '$2 ~ /^placers-020/ {print $1}')
test -z "$QUEUED" || fail "placers-020 jobs already in Slurm (scancel $(echo $QUEUED))"
named() { printf '%s\n' "$JOBS" | awk -v n="$1" '$2 == n {print $1; exit}'; }
AFTER_SIZERS=""
for id in ${SIZERS//:/ }; do
  if test -n "$(printf '%s\n' "$JOBS" | awk -v i="$id" '$1 == i')"; then
    AFTER_SIZERS="$AFTER_SIZERS:$id"
  else
    STATE=$(sacct -j "$id" -X -n -P -o State | head -n 1) || fail "sacct failed for sizer job $id"
    test "$STATE" = COMPLETED || fail "sizer job $id is not in Slurm and ended '${STATE:-unknown}', not COMPLETED"
  fi
done
L="$RUN/sizes-020-L"
MODES=(llm-L llm-O2)
if test -f "$L/MANIFEST.json"; then
  :
elif grep -qs '"status": *"CAP_REFUSED"' "$L/status.json"; then
  MODES=(llm-O2)
  echo "skip llm-L: $L/status.json is CAP_REFUSED"
elif ! test -e "$L"; then
  CHECK=$(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PY" -B -m fastfill.comparison.sizers_020 cap --run-root "$RUN" \
    --rooms 1877) || true
  case "$CHECK" in
    *'"ok": false'*) MODES=(llm-O2); echo "skip llm-L: the API cap preflight refuses the L sizer: $CHECK" ;;
    *'"ok": true'*) ;;
    *) fail "sizers_020 cap preflight failed" ;;
  esac
fi
if test -z "$SIZERS"; then
  for M in "${MODES[@]}"; do
    test -f "$RUN/sizes-020-${M#llm-}/MANIFEST.json" || fail "no SIZERS_JOB and $RUN/sizes-020-${M#llm-} not published"
  done
fi
OS_AFTER=""  # '' = run OS, ':ID' = afterok the dpo2 merge, '-' = skip OS
if ! test -f "$OUT14/dpo2-merged/MANIFEST.json"; then
  MERGE=$(named optiscene-014-merge-dpo2)
  if test -n "$MERGE"; then OS_AFTER=":$MERGE"; else OS_AFTER=-; fi
fi
todo() { ! test -f "$1/MANIFEST.json"; }  # A GPU step runs (and is planned) only while its output is unpublished.
gen_todo() { test "$OS_AFTER" != - && { todo "$OUT/gen/dpo2-e2-$1-sft" || todo "$OUT/gen/dpo2-e3-$1-sft"; }; }
shard_todo() { todo "$OUT/p-$1/infer" && todo "$OUT/p-$1/shard-0$2"; }
PLANNED=0  # GPU jobs x 1 h
for M in "${MODES[@]}"; do
  if gen_todo "$M"; then PLANNED=$((PLANNED + 1)); fi
  for ((k = 0; k < SHARDS; k++)); do if shard_todo "$M" "$k"; then PLANNED=$((PLANNED + 1)); fi; done
done
PYTHONPATH="$CODE_ROOT" "$PY" "${P20[@]}" budget --cap "$CAP" --planned "$PLANNED" --sizes "$RUN/sizes-020-L" \
  "$RUN/sizes-020-O2" --jobs ${SIZERS//:/ } || fail "amendment-020 GPU cap ($CAP GPU-h)"
SUBMITTED=""
ROOTS=""
submit() {  # submit LABEL DEPENDENCY ROOT SBATCH_ARGS...; sets ID; ROOT=1: held until the chain is in. set -e misses $(...).
  local label="$1" dep="$2" root="$3" out=""
  shift 3
  local args=(--parsable --kill-on-invalid-dep=yes)
  if test "$root" = 1; then args+=(--hold); fi
  if test -n "$dep"; then args+=(--dependency="$dep"); fi
  out=$(sbatch "${args[@]}" --job-name="placers-020-$label" "$@") || out=""
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$label'; chain NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  if test "$root" = 1; then ROOTS="$ROOTS${ROOTS:+,}$ID"; fi
  echo "$label $ID"
}
B="$S/submit_020_placers.sh"
CPU=(--cpus-per-task=16 --mem=64G)
SCORE=(--cpus-per-task=32 --mem=128G --time=04:00:00)
GPU=(--gpus-per-node=1 --cpus-per-task=16 --mem=110G --time=01:00:00)
PLANS=()
for M in "${MODES[@]}"; do
  submit "plan-$M" "${AFTER_SIZERS:+afterok$AFTER_SIZERS}" 1 "${CPU[@]}" --time=02:00:00 "$B" "$CODE_ROOT" job plan "$M"
  PLANS+=("$ID")
done
SCORES=""
if test "$OS_AFTER" = -; then
  echo "skip OS-DPO2: $OUT14/dpo2-merged is not published and no optiscene-014-merge-dpo2 job is in Slurm"
else
  GEN14=$(named optiscene-014-gen-dpo2)
  GEN15=$(named optiscene-015-gen-dpo2)
  GENS="${GEN14:+:$GEN14}${GEN15:+:$GEN15}"
  for M in prior gt; do
    submit "score-os-$M" "${GENS:+afterany$GENS}" 1 "${SCORE[@]}" "$B" "$CODE_ROOT" job score os "$M"
    SCORES="$SCORES:$ID"
  done
  for i in "${!MODES[@]}"; do
    M="${MODES[$i]}"
    ID="${PLANS[$i]}"
    if gen_todo "$M"; then submit "os-gen-$M" "afterok:$ID$OS_AFTER" 0 "${GPU[@]}" "$B" "$CODE_ROOT" job os-gen "$M"; fi
    submit "score-os-$M" "afterok:$ID" 0 "${SCORE[@]}" "$B" "$CODE_ROOT" job score os "$M"
    SCORES="$SCORES:$ID"
  done
fi
submit score-p-prior "" 1 "${SCORE[@]}" "$B" "$CODE_ROOT" job score p prior
SCORES="$SCORES:$ID"
for i in "${!MODES[@]}"; do
  M="${MODES[$i]}"
  SHARD_IDS=""
  for ((k = 0; k < SHARDS; k++)); do
    if shard_todo "$M" "$k"; then
      submit "p-shard-$M-$k" "afterok:${PLANS[$i]}" 0 "${GPU[@]}" "$B" "$CODE_ROOT" job p-shard "$M" "$k"
      SHARD_IDS="$SHARD_IDS:$ID"
    fi
  done
  submit "p-merge-$M" "afterok${SHARD_IDS:-:${PLANS[$i]}}" 0 "${CPU[@]}" --time=01:00:00 "$B" "$CODE_ROOT" job p-merge "$M"
  submit "score-p-$M" "afterok:$ID" 0 "${SCORE[@]}" "$B" "$CODE_ROOT" job score p "$M"
  SCORES="$SCORES:$ID"
done
submit report "afterany$SCORES" 0 "${CPU[@]}" --time=04:00:00 "$B" "$CODE_ROOT" job report
scontrol release "$ROOTS" || { echo "ABORT: could not release $ROOTS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOTS"
