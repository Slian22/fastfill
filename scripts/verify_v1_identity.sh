#!/usr/bin/env bash
# v1 freeze + dual-port model-identity verification (train machine, repo root).
#
# Does, in order (auditor's ops corrections baked in):
#   1. Freeze MANIFEST.sha256 for every v1 model dir (fail on missing/empty,
#      verify with sha256sum -c immediately).
#   2. Serve full_fp and full_r128_merged on separate GPUs/ports/aliases
#      (ss preflight, readiness wait, PID liveness, exact-alias jq check,
#      PID->model-path recorded).
#   3. Run eval_layout against BOTH endpoints with --dump-generations and
#      assert n_missing_generation == 0 + identity fields present + both
#      arms share the same records/samples/uids manifests.
# Non-zero exit on ANY failure. Servers are LEFT RUNNING on exit — user
# policy: no fastfill script ever kills a vLLM process; kill manually.
#
# Usage: bash scripts/verify_v1_identity.sh
set -euo pipefail
cd "$(dirname "$0")/.."

FP_DIR="out/full_fp"
R128_DIR="out/full_r128_merged"
FP_PORT="${FP_PORT:-8901}"
R128_PORT="${R128_PORT:-8902}"
FP_GPU="${FP_GPU:-0}"
R128_GPU="${R128_GPU:-1}"
RECORDS="${RECORDS:-data/full/heldout.jsonl}"
SAMPLES="${SAMPLES:-out/conv/deduped.jsonl}"
STAMP="$(date +%Y%m%d_%H%M%S)"
IDENTITY_LOG="out/identity_${STAMP}.log"

# Post-mortem aids: full command trace to a repo-root file (out/ may be wiped
# later by the rebuild) + the exact failing line/command on any error.
DEBUG_LOG="verify_debug_${STAMP}.log"
exec 9>"$DEBUG_LOG"
BASH_XTRACEFD=9
PS4='+ [$(date "+%F %T")] ${BASH_SOURCE##*/}:${LINENO}: '
set -x
trap 'echo "[$(date "+%F %T")] FAILED at ${BASH_SOURCE##*/}:${LINENO}: $BASH_COMMAND" >&2' ERR
echo "[$(date '+%F %T')] verify_v1_identity start — command trace: $DEBUG_LOG"

# --- 1. Freeze manifests ----------------------------------------------------
for d in out/full_fp out/full_r128 "$R128_DIR" out/sft_full out/sft_full_merged; do
  [ -d "$d" ] || { echo "SKIP (missing dir): $d"; continue; }
  (
    cd "$d"
    find . -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.json" -o -name "*.bin" \) \
      -exec sha256sum {} + | sort -k2 > MANIFEST.sha256
    [ -s MANIFEST.sha256 ] || { echo "ERROR: empty manifest in $d" >&2; exit 1; }
    sha256sum -c MANIFEST.sha256 --quiet
    echo "manifest OK: $d ($(wc -l < MANIFEST.sha256) files)"
  )
done
[ -d "$FP_DIR" ] || { echo "ERROR: $FP_DIR missing" >&2; exit 1; }
[ -d "$R128_DIR" ] || { echo "ERROR: $R128_DIR missing" >&2; exit 1; }

# --- 2. Dual-port serve -----------------------------------------------------
for p in "$FP_PORT" "$R128_PORT"; do
  if ss -ltn "sport = :$p" | grep -q LISTEN; then
    echo "ERROR: port $p already listening — kill it first" >&2; exit 1
  fi
done
mkdir -p out
CUDA_VISIBLE_DEVICES="$FP_GPU" vllm serve "$(realpath "$FP_DIR")" \
  --port "$FP_PORT" --served-model-name fastfill-full-fp \
  --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 4096 \
  > "out/vllm_id_fp_${STAMP}.log" 2>&1 &
FP_PID=$!
CUDA_VISIBLE_DEVICES="$R128_GPU" vllm serve "$(realpath "$R128_DIR")" \
  --port "$R128_PORT" --served-model-name fastfill-r128 \
  --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 4096 \
  > "out/vllm_id_r128_${STAMP}.log" 2>&1 &
R128_PID=$!
# NO kill trap (user policy: scripts never kill vLLM). On ANY exit — pass or
# fail — both servers stay up; the trap only prints them for manual cleanup.
trap 'echo "NOTE: vLLM servers left RUNNING — kill manually when done: kill $FP_PID $R128_PID"' EXIT

wait_ready() { # port pid name
  for _ in $(seq 1 120); do
    kill -0 "$2" 2>/dev/null || { echo "ERROR: $3 vLLM died — see log" >&2; exit 1; }
    if curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null; then return 0; fi
    sleep 5
  done
  echo "ERROR: $3 never became ready" >&2; exit 1
}
wait_ready "$FP_PORT" "$FP_PID" full_fp
wait_ready "$R128_PORT" "$R128_PID" r128

# Exact alias match + PID->model-path record (identity evidence).
curl -s "http://127.0.0.1:$FP_PORT/v1/models" \
  | jq -e '.data | any(.id == "fastfill-full-fp")' >/dev/null \
  || { echo "ERROR: port $FP_PORT does not serve alias fastfill-full-fp" >&2; exit 1; }
curl -s "http://127.0.0.1:$R128_PORT/v1/models" \
  | jq -e '.data | any(.id == "fastfill-r128")' >/dev/null \
  || { echo "ERROR: port $R128_PORT does not serve alias fastfill-r128" >&2; exit 1; }
{
  echo "stamp=$STAMP"
  echo "fp:   pid=$FP_PID port=$FP_PORT path=$(realpath "$FP_DIR") cmdline=$(tr '\0' ' ' < "/proc/$FP_PID/cmdline" 2>/dev/null || ps -o command= -p "$FP_PID")"
  echo "r128: pid=$R128_PID port=$R128_PORT path=$(realpath "$R128_DIR") cmdline=$(tr '\0' ' ' < "/proc/$R128_PID/cmdline" 2>/dev/null || ps -o command= -p "$R128_PID")"
} | tee "$IDENTITY_LOG"

# --- 3. Paired eval ----------------------------------------------------------
run_eval() { # port alias tag
  PYTHONPATH=src python3 -m fastfill_train.eval_layout \
    --records "$RECORDS" --samples "$SAMPLES" \
    --endpoint "http://127.0.0.1:$1/v1" --model "$2" \
    --dump-generations "out/gens_id_$3_${STAMP}.jsonl" \
    --out "out/eval_id_$3_${STAMP}.json"
  jq -e '.n_missing_generation == 0
         and (.selection.uids_sha256 | length) > 0
         and (.selection.records_sha256 | length) > 0
         and (.selection.samples_sha256 | length) > 0
         and (.scored_uids_sha256 | length) > 0' \
    "out/eval_id_$3_${STAMP}.json" >/dev/null \
    || { echo "ERROR: $3 report failed identity-field checks" >&2; exit 1; }
}
run_eval "$FP_PORT" fastfill-full-fp fp
run_eval "$R128_PORT" fastfill-r128 r128

# Both arms must have evaluated the SAME manifest.
for key in uids_sha256 records_sha256 samples_sha256; do
  a=$(jq -r ".selection.$key" "out/eval_id_fp_${STAMP}.json")
  b=$(jq -r ".selection.$key" "out/eval_id_r128_${STAMP}.json")
  [ "$a" = "$b" ] || { echo "ERROR: $key differs between arms — not paired" >&2; exit 1; }
done

echo "=== IDENTITY VERIFICATION PASSED ==="
for t in fp r128; do
  jq -r --arg t "$t" '"\($t): parse=\(.parse_rate) floor_pre=\(.pass_pre_repair) floor_post=\(.pass_post_repair) surface_post=\(.surface_pass_post_repair)"' \
    "out/eval_id_${t}_${STAMP}.json" | sed "s/^fp:/full_fp:/;s/^r128:/r128:   /"
done
echo "reports: out/eval_id_{fp,r128}_${STAMP}.json   identity: $IDENTITY_LOG"
echo "vLLM servers intentionally left running (manual-kill policy):"
echo "  fp:   pid=$FP_PID port=$FP_PORT"
echo "  r128: pid=$R128_PID port=$R128_PORT"
echo "after inspecting the reports, kill them yourself: kill $FP_PID $R128_PID"
