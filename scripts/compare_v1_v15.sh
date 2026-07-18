#!/usr/bin/env bash
# Paired v1 (archived) vs v1.5 comparison on the NEW heldout — same harness,
# same UID manifest, full fairness assertions, explicit server cleanup.
#
# Usage: bash scripts/compare_v1_v15.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

RECORDS="data/full_research_v15/heldout.jsonl"
SAMPLES="out/conv/deduped_research.jsonl"
test -f "$RECORDS" || { echo "ERROR: $RECORDS missing" >&2; exit 1; }
test -f "$SAMPLES" || { echo "ERROR: $SAMPLES missing" >&2; exit 1; }
test -d out/research_v15_fp || { echo "ERROR: out/research_v15_fp missing" >&2; exit 1; }
V1_ARCHIVE="$(cat ~/.fastfill_v1_archive_path)"
test -d "$V1_ARCHIVE/full_fp" || { echo "ERROR: archive missing: $V1_ARCHIVE/full_fp" >&2; exit 1; }

for p in 8901 8902; do
  ss -ltn "sport = :$p" | grep -q LISTEN \
    && { echo "ERROR: port $p already listening" >&2; exit 1; }
done

CUDA_VISIBLE_DEVICES=0 vllm serve "$(realpath out/research_v15_fp)" --port 8901 \
  --served-model-name fastfill-v15 --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 > vllm_v15.log 2>&1 &
V15_PID=$!
CUDA_VISIBLE_DEVICES=1 vllm serve "$(realpath "$V1_ARCHIVE/full_fp")" --port 8902 \
  --served-model-name fastfill-v1 --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 > vllm_v1.log 2>&1 &
V1_PID=$!
cleanup() { kill "$V15_PID" "$V1_PID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

wait_ready() { # port pid name
  for _ in $(seq 1 120); do
    kill -0 "$2" 2>/dev/null || { echo "ERROR: $3 vLLM died — see log" >&2; exit 1; }
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0
    sleep 5
  done
  echo "ERROR: $3 never became ready" >&2; exit 1
}
wait_ready 8901 "$V15_PID" v15
wait_ready 8902 "$V1_PID"  v1
curl -s http://127.0.0.1:8901/v1/models | jq -e '.data|any(.id=="fastfill-v15")' >/dev/null \
  || { echo "ERROR: 8901 does not serve fastfill-v15" >&2; exit 1; }
curl -s http://127.0.0.1:8902/v1/models | jq -e '.data|any(.id=="fastfill-v1")' >/dev/null \
  || { echo "ERROR: 8902 does not serve fastfill-v1" >&2; exit 1; }

for tag in v15:8901:fastfill-v15 v1:8902:fastfill-v1; do
  IFS=: read -r t p m <<< "$tag"
  PYTHONPATH=src python3 -m fastfill_train.eval_layout \
    --records "$RECORDS" --samples "$SAMPLES" \
    --endpoint "http://127.0.0.1:$p/v1" --model "$m" \
    --dump-generations "out/gens_${t}.jsonl" --out "out/eval_${t}.json"
done

# Explicit cleanup NOW (trap only fires when this shell exits).
cleanup
trap - EXIT

# --- FULL fairness assertions ------------------------------------------------
# Everything that defines "same experiment" must match; endpoint_model and
# uids_manifest paths legitimately differ and are excluded.
identity='{harness, n, n_validated, n_validated_surface, n_missing_generation,
           scored_uids_sha256,
           sel: (.selection | {uids_sha256, records_sha256, samples_sha256,
                               temperature, template, sample_mode, limit, seed,
                               generation_mode, extra_body_sha256})}'
for t in v15 v1; do
  jq -e '.n_missing_generation == 0' "out/eval_${t}.json" >/dev/null \
    || { echo "ERROR: $t has missing generations — not a full pass" >&2; exit 1; }
done
if ! diff <(jq -S "$identity" out/eval_v15.json) \
          <(jq -S "$identity" out/eval_v1.json); then
  echo "ERROR: identity mismatch between arms — comparison is NOT paired" >&2
  exit 1
fi

echo "=== PAIRED COMPARISON (same heldout, same harness) ==="
for t in v1 v15; do
  jq -r --arg t "$t" \
    '"\($t): parse=\(.parse_rate) floor_pre=\(.pass_pre_repair) floor_post=\(.pass_post_repair) surface_pre=\(.surface_pass_pre_repair) surface_post=\(.surface_pass_post_repair)"' \
    "out/eval_${t}.json"
done
echo "reports: out/eval_v1.json out/eval_v15.json"
