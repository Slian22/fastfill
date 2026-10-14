#!/bin/bash
#SBATCH --job-name=unknownsize-021-train
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --exclude=nid010216
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-021-train-%j.out
# Amendment 021 section 2 (Arm A = the 27B student, Arm B = the 8B fallback; RUN/a021): CODE_ROOT MODE [ARGS]. The
# header is the CPU setup job (72 cores, 192G, 8 h); GPU modes take their resources on the sbatch line
# (submit_021_train.sh). Every job prepends the fla overlay PROJECT/pylib-021 to PYTHONPATH (no other env changes).
#   setup        CPU   once: pip install --no-deps --target pylib-021 fla-core==FLA_VERSION (+ einops if the env lacks
#                      it), staged then renamed; fla must resolve to the overlay; kernels --report prints the qwen3_5
#                      path (no CUDA here: the torch fallback is expected; the GPU jobs' kernels check is binding);
#                      then, unless it exists, a021_train pool -> RUN/a021/pool
#   smoke        --gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=01:30:00   kernels (strict), a021_train smoke
#                      -> RUN/a021/smoke/verdict.json (exit 1 on SMOKE_STOP; a crash writes no verdict and renames
#                      the attempt smoke.failed-JOB)
#   smoke-vllm   --gpus-per-node=1 --cpus-per-task=16 --mem=200G --time=01:00:00   (vLLM env, no overlay) vLLM loads
#                      the smoke adapter as a LoRA on the 27B -> RUN/a021/smoke/vllm-load.json (Arm A train needs it)
#   train A|B H  --gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=H:00:00   (A: kernels strict) a021_train train
#                      -> RUN/a021/27b|8b; a rerun (next segment) resumes from ckpt_latest
#   probe        --gpus-per-node=1 --cpus-per-task=16 --mem=200G --time=03:00:00   kernels, probe-hf (training env;
#                      per room, a rerun resumes), probe-vllm (vLLM env), probe-report -> RUN/a021/probe; each part
#                      skipped once its output exists
set -euo pipefail
test "$#" -ge 2 || exit 2
CODE_ROOT="$1"
MODE="$2"
shift 2
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
A21="$RUN_ROOT/a021"
PYTHON="$PROJECT_ROOT/env/bin/python"
VLLM_PYTHON=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
OVERLAY="$PROJECT_ROOT/pylib-021"
FLA_VERSION=0.5.2  # Same chunk/recurrent/FusedRMSNormGated signatures as 0.4.0; newer Triton support.
EINOPS_VERSION=0.8.1
B27="$PROJECT_ROOT/models/Qwen3.8-27B"
B8="$PROJECT_ROOT/models/Qwen3-8B"
LINEAGE="$CODE_ROOT/docs/research/20261013-unknown-size-comparison/evidence/full-audit-data-lineage-7368003-result.json"
GPUS=0 KERNELS=""
case "$MODE" in
  setup) test "$#" -eq 0 || exit 2; LIMIT=460m ;;
  smoke) test "$#" -eq 0 || exit 2; GPUS=4 LIMIT=85m KERNELS=1 ;;
  smoke-vllm) test "$#" -eq 0 || exit 2; GPUS=1 LIMIT=50m ;;
  train)
    test "$#" -eq 2 || exit 2
    case "$1" in A) ARM=A KERNELS=1 ;; B) ARM=B ;; *) exit 2 ;; esac
    case "$2" in ''|*[!0-9]*) exit 2 ;; esac
    test "$2" -ge 1 -a "$2" -le 24 || exit 2
    GPUS=4 LIMIT="$(( $2 * 60 - 10 ))m" ;;
  probe) test "$#" -eq 0 || exit 2; GPUS=1 KERNELS=1 ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Slurm allocation required}"
export PYTHONPATH="$CODE_ROOT:$OVERLAY" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=/tmp
JOB_CACHE="$PROJECT_ROOT/.cache-jobs/a021"
export TRITON_CACHE_DIR="$JOB_CACHE/triton" XDG_CACHE_HOME="$JOB_CACHE/xdg"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$A21"
cd "$CODE_ROOT"
A=("$PYTHON" -B -u -m fastfill.comparison.a021_train)
if test "$GPUS" -gt 0; then  # As isambard_018.sbatch train; at least GPUS GPUs visible.
  module load cudatoolkit/24.11_12.6
  export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
  export CUDA_PATH="$CUDA_HOME" WANDB_DISABLED=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  nvidia-smi -L
  SEEN=$(nvidia-smi -L | grep -c '^GPU ' || true)
  test "$SEEN" -ge "$GPUS" || { echo "PREFLIGHT FAIL: $MODE needs $GPUS GPUs, sees $SEEN" >&2; exit 1; }
  if test -n "$KERNELS"; then "${A[@]}" kernels; fi  # fla bound and agreeing, else stop here.
fi
vllm() {  # vllm LIMIT STEP ARGS...: the vLLM env as isambard_018.sbatch's vllm kind; its own kernels, so no overlay.
  (
    PIP_CUDA="$(dirname "$(dirname "$VLLM_PYTHON")")/lib/python3.11/site-packages/nvidia/cu13"
    if test -x "$PIP_CUDA/bin/nvcc"; then CUDA_HOME="$PIP_CUDA"; else CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; fi
    export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$(dirname "$VLLM_PYTHON"):$CUDA_HOME/bin:$PATH" PYTHONPATH="$CODE_ROOT"
    export HF_HOME=/scratch/u6uv/shanliantian.u6uv/.cache/huggingface VLLM_CACHE_ROOT="$JOB_CACHE/vllm"
    export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor" TRITON_CACHE_DIR="$JOB_CACHE/triton-vllm"
    export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
    mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    exec timeout --signal=TERM --kill-after=60s "$1" "$VLLM_PYTHON" -B -u -m fastfill.comparison.a021_train "${@:2}"
  )
}
case "$MODE" in
  setup)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    MARKER="$OVERLAY/.a021-fla-core-$FLA_VERSION"
    if ! test -f "$MARKER"; then
      test ! -e "$OVERLAY" || { echo "PREFLIGHT FAIL: $OVERLAY exists without $MARKER; inspect it" >&2; exit 1; }
      STAGE="$OVERLAY.tmp.$SLURM_JOB_ID"
      PKGS=("fla-core==$FLA_VERSION")
      "$PYTHON" -c 'import einops' 2>/dev/null || PKGS+=("einops==$EINOPS_VERSION")
      "$PYTHON" -m pip install --no-deps --no-cache-dir --target "$STAGE" "${PKGS[@]}"
      printf '%s\n' "${PKGS[@]}" > "$STAGE/.a021-fla-core-$FLA_VERSION"
      mv -T "$STAGE" "$OVERLAY"
    fi
    "$PYTHON" -B - "$OVERLAY" <<'PY'
import importlib.metadata, importlib.util, sys
spec = importlib.util.find_spec('fla')
assert spec is not None and spec.origin.startswith(sys.argv[1] + '/'), f'fla does not resolve to the overlay: {spec}'
print('fla-core', importlib.metadata.version('fla-core'), spec.origin)
PY
    "$PYTHON" -B -c 'import fla.ops.gated_delta_rule' \
      || echo "WARNING: fla does not import on this CPU node; the GPU jobs' kernels check is binding" >&2
    "${A[@]}" kernels --report
    if test -f "$A21/pool/MANIFEST.json"; then echo "pool exists: $A21/pool"; exit 0; fi
    exec timeout --signal=TERM --kill-after=60s "$LIMIT" "${A[@]}" pool --data "$RUN_ROOT/main-data-v1" \
      --corpus "$RUN_ROOT/full-corpus-v2" --lineage "$LINEAGE" --base27 "$B27" --base8 "$B8" \
      --dev "$RUN_ROOT/dev-select-v1" "$RUN_ROOT/dev-full-v1" --test "$RUN_ROOT/eval-data-v1" --out "$A21/pool" \
      --workers 70 ;;
  smoke)
    exec timeout --signal=TERM --kill-after=120s "$LIMIT" "${A[@]}" smoke --pool "$A21/pool" --base "$B27" \
      --out "$A21/smoke" ;;
  smoke-vllm)
    vllm "$LIMIT" smoke-vllm --data "$RUN_ROOT/dev-select-v1" --smoke "$A21/smoke" --base "$B27" ;;
  train)
    exec 9>"$A21/.train-$ARM.lock"  # One writer per arm directory, also against a manual resubmit.
    flock -n 9 || { echo "another job owns Arm $ARM" >&2; exit 1; }
    if test "$ARM" = A; then BASE=("$B27" --smoke "$A21/smoke"); else BASE=("$B8"); fi
    exec timeout --signal=TERM --kill-after=120s "$LIMIT" "${A[@]}" train --arm "$ARM" --pool "$A21/pool" \
      --base "${BASE[@]}" --run-root "$A21" ;;
  probe)
    P=(--data "$RUN_ROOT/dev-select-v1" --policy "$A21/27b" --base "$B27" --out "$A21/probe")
    if ! test -f "$A21/probe/hf.jsonl"; then timeout --signal=TERM --kill-after=60s 115m "${A[@]}" probe-hf "${P[@]}"; fi
    if ! test -f "$A21/probe/vllm.jsonl"; then vllm 40m probe-vllm "${P[@]}"; fi
    exec timeout --signal=TERM --kill-after=60s 15m "${A[@]}" probe-report "${P[@]}" ;;
esac
