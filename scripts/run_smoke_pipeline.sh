#!/usr/bin/env bash
# T3.1 end-to-end smoke pipeline for the FastFill planner.
#
# Verifies the full loop cheaply BEFORE any long run:
#   SFT (100 steps, LoRA) -> merge -> vLLM serve -> layout eval vs endpoint.
#
# Assumes (done beforehand, see vendor/tools/fastfill_data/):
#   - the data chain already ran IN THIS ORDER (see README step 2):
#       convert_*.py -> deduplicate.py --contamination-list (dedup #1 +
#       closure on raw hashes) -> sanitize.py -> deduplicate.py (dedup #2)
#       -> export_sft.py -> make_snapshot (train/heldout/test)
#     producing data/stage0/{train,heldout}.jsonl
#   - training deps installed (requirements.txt) on a CUDA machine,
#     including vllm for step 3.
#
# Each numbered block is a standalone copy-paste unit.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8901}"

# --- 1. SFT smoke train (max_steps=100 -> LoRA adapter in out/sft_smoke) ---
PYTHONPATH=src python3 -m fastfill_train.train_sft \
  --config configs/sft_smoke.yaml

# --- 2. Merge the adapter into a runnable HF model dir ---------------------
PYTHONPATH=src python3 -m fastfill_train.merge_lora \
  --base "$BASE_MODEL" --lora out/sft_smoke --out out/sft_smoke_merged

# --- 3. Serve the merged model (OpenAI-compatible) -------------------------
# Backgrounded here so step 4 can run in the same shell; for a long-lived
# server run this in its own terminal instead:
#   PORT=8901 bash scripts/serve_vllm.sh out/sft_smoke_merged
mkdir -p out
PORT="$PORT" bash scripts/serve_vllm.sh out/sft_smoke_merged \
  > out/vllm_smoke.log 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# Wait for the endpoint (vLLM cold start can take minutes; log has details).
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then break; fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null \
  || { echo "vLLM never became ready — see out/vllm_smoke.log" >&2; exit 1; }

# --- 4. Layout eval against the live endpoint ------------------------------
PYTHONPATH=src python3 -m fastfill_train.eval_layout \
  --records data/stage0/heldout.jsonl \
  --samples out/conv/deduped.jsonl \
  --endpoint "http://127.0.0.1:$PORT/v1" \
  --model fastfill-planner --limit 100 \
  --out out/eval_smoke.json

echo "smoke pipeline complete — adapter: out/sft_smoke, merged: out/sft_smoke_merged"
