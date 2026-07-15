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

# GPU selection: honour an explicit override, otherwise find the first GPU
# with < 1 GB used memory (training jobs each claim 80-90 GB, so anything
# above 1 GB means the card is occupied). Fail fast rather than OOM-crash
# mid-run or stomp on a live training job.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (explicit)"
else
  FOUND_GPU=""
  for gpu in $(seq 0 7); do
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                -i "$gpu" 2>/dev/null || echo 99999)
    if [[ "$used_mb" -lt 1000 ]]; then
      FOUND_GPU="$gpu"
      break
    fi
  done
  if [[ -z "$FOUND_GPU" ]]; then
    echo "ERROR: no free GPU found (all 8 cards report >= 1 GB used)." >&2
    echo "  Set CUDA_VISIBLE_DEVICES=<card> explicitly, e.g.:" >&2
    echo "    CUDA_VISIBLE_DEVICES=2 bash scripts/run_smoke_pipeline.sh" >&2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$FOUND_GPU"
  echo "Auto-selected GPU ${FOUND_GPU} for smoke pipeline"
fi

# --- 1. SFT smoke train (max_steps=100 -> LoRA adapter in out/sft_smoke) ---
# BASE_MODEL feeds BOTH the trainer and the merge: overriding it only at the
# merge step would silently weld the adapter onto a different base.
PYTHONPATH=src python3 -m fastfill_train.train_sft \
  --config configs/sft_smoke.yaml \
  --set "model_name_or_path=$BASE_MODEL"

# --- 2. Merge the adapter into a runnable HF model dir ---------------------
PYTHONPATH=src python3 -m fastfill_train.merge_lora \
  --base "$BASE_MODEL" --lora out/sft_smoke --out out/sft_smoke_merged

# --- 3. Serve the merged model (OpenAI-compatible) -------------------------
# Backgrounded here so step 4 can run in the same shell; for a long-lived
# server run this in its own terminal instead:
#   PORT=8901 bash scripts/serve_vllm.sh out/sft_smoke_merged
# A stale server on $PORT would answer the readiness probe and step 4 would
# silently evaluate the WRONG model (every serve_vllm.sh instance shares the
# fastfill-planner alias) — fail fast instead.
if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
  echo "ERROR: port $PORT already serves a model — kill it or set PORT." >&2
  exit 1
fi
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
kill -0 "$VLLM_PID" 2>/dev/null \
  || { echo "vLLM process died — see out/vllm_smoke.log" >&2; exit 1; }
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null \
  || { echo "vLLM never became ready — see out/vllm_smoke.log" >&2; exit 1; }

# --- 4. Layout eval against the live endpoint ------------------------------
PYTHONPATH=src python3 -m fastfill_train.eval_layout \
  --records data/stage0/heldout.jsonl \
  --samples out/conv/deduped_permissive.jsonl \
  --endpoint "http://127.0.0.1:$PORT/v1" \
  --model fastfill-planner --limit 100 \
  --out out/eval_smoke.json

# Gate on the eval outcome: an all-garbage model must not print success.
python3 - <<'PY'
import json, sys
r = json.load(open("out/eval_smoke.json"))
n, parse = r.get("n") or 0, r.get("parse_rate") or 0.0
if n == 0 or parse <= 0.0:
    print(f"SMOKE FAIL: n={n} parse_rate={parse} — see out/eval_smoke.json",
          file=sys.stderr)
    sys.exit(1)
print(f"smoke eval: n={n} parse_rate={parse:.3f} "
      f"pass_post_repair={r.get('pass_post_repair')}")
PY

echo "smoke pipeline complete — adapter: out/sft_smoke, merged: out/sft_smoke_merged"
