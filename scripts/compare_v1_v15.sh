#!/usr/bin/env bash
# Paired v1 (archived) vs v1.5 comparison on the NEW heldout — same harness,
# same UID manifest, full fairness assertions. Servers are LEFT RUNNING on
# exit (user policy: no fastfill script ever kills a vLLM process).
#
# Usage: bash scripts/compare_v1_v15.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

# Post-mortem aids: full command trace + exact failing line on any error.
DEBUG_LOG="compare_debug_$(date +%Y%m%d_%H%M%S).log"
exec 9>"$DEBUG_LOG"
BASH_XTRACEFD=9
PS4='+ [$(date "+%F %T")] ${BASH_SOURCE##*/}:${LINENO}: '
set -x
trap 'echo "[$(date "+%F %T")] FAILED at ${BASH_SOURCE##*/}:${LINENO}: $BASH_COMMAND" >&2' ERR
echo "[$(date '+%F %T')] compare_v1_v15 start — command trace: $DEBUG_LOG"

RECORDS="data/full_research_v15/heldout.jsonl"
SAMPLES="out/conv/deduped_research.jsonl"
test -f "$RECORDS" || { echo "ERROR: $RECORDS missing" >&2; exit 1; }
test -f "$SAMPLES" || { echo "ERROR: $SAMPLES missing" >&2; exit 1; }
test -d out/research_v15_fp || { echo "ERROR: out/research_v15_fp missing" >&2; exit 1; }
# Archive pointer: repo-local archive/LATEST_V1 (all storage stays under the
# repo tree), with fallback to the legacy ~/.fastfill_v1_archive_path.
V1_ARCHIVE=""
for f in archive/LATEST_V1 "$HOME/.fastfill_v1_archive_path"; do
  if [ -f "$f" ] && [ -d "$(cat "$f")/full_fp" ]; then V1_ARCHIVE="$(cat "$f")"; break; fi
done
test -n "$V1_ARCHIVE" \
  || { echo "ERROR: no v1 archive found (archive/LATEST_V1 missing) — run rebuild first" >&2; exit 1; }

# --- Weight + provenance verification BEFORE anything is served ---------------
# v1 arm: archived weights must still match their frozen manifest.
(cd "$V1_ARCHIVE/full_fp" && sha256sum -c MANIFEST.sha256 --quiet) \
  || { echo "ERROR: v1 archive weights no longer match MANIFEST.sha256" >&2; exit 1; }
# v1.5 arm: freeze a manifest now (first run) or verify against it (reruns),
# and require verified provenance bound to the v1.5 snapshot.
if [ ! -f out/research_v15_fp/MANIFEST.sha256 ]; then
  (cd out/research_v15_fp && find . -maxdepth 1 -type f \
     \( -name '*.safetensors' -o -name '*.json' \) -exec sha256sum {} + \
     | sort -k2 > MANIFEST.sha256)
fi
(cd out/research_v15_fp && grep -v ' \./MANIFEST.sha256$' MANIFEST.sha256 | sha256sum -c --quiet) \
  || { echo "ERROR: out/research_v15_fp weights do not match MANIFEST.sha256" >&2; exit 1; }
test -f out/research_v15_fp/DATA_PROVENANCE.json \
  || { echo "ERROR: out/research_v15_fp has no DATA_PROVENANCE.json" >&2; exit 1; }
SNAP_ID="$(jq -r '.snapshot_id' data/full_research_v15/SNAPSHOT.json)"
jq -e --arg sid "$SNAP_ID" '.verified == true and .snapshot_id == $sid' \
    out/research_v15_fp/DATA_PROVENANCE.json >/dev/null \
  || { echo "ERROR: v1.5 provenance not verified or bound to a different snapshot" >&2; \
       echo "  expected snapshot_id $SNAP_ID; see DATA_PROVENANCE.json" >&2; exit 1; }

for p in 8901 8902; do
  ss -ltn "sport = :$p" | grep -q LISTEN \
    && { echo "ERROR: port $p already listening" >&2; exit 1; }
done

# GPU picks are env-overridable for busy nodes (e.g. V15_GPU=4 V1_GPU=5
# when GPUs 0-3 are taken by another job).
CUDA_VISIBLE_DEVICES="${V15_GPU:-0}" vllm serve "$(realpath out/research_v15_fp)" --port 8901 \
  --served-model-name fastfill-v15 --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 > vllm_v15.log 2>&1 &
V15_PID=$!
CUDA_VISIBLE_DEVICES="${V1_GPU:-1}" vllm serve "$(realpath "$V1_ARCHIVE/full_fp")" --port 8902 \
  --served-model-name fastfill-v1 --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 > vllm_v1.log 2>&1 &
V1_PID=$!
# NO kill trap (user policy: scripts never kill vLLM); the trap only prints
# the PIDs on exit so the user can kill them manually after inspection.
trap 'echo "NOTE: vLLM servers left RUNNING — kill manually when done: kill $V15_PID $V1_PID"' EXIT

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

# PID -> model-path evidence, saved for the record (winner is only pinned
# with this file present).
{
  echo "port=8901 alias=fastfill-v15 pid=$V15_PID path=$(realpath out/research_v15_fp)"
  echo "  cmdline: $(ps -o args= -p "$V15_PID")"
  echo "port=8902 alias=fastfill-v1 pid=$V1_PID path=$(realpath "$V1_ARCHIVE/full_fp")"
  echo "  cmdline: $(ps -o args= -p "$V1_PID")"
} > out/compare_identity.log
cat out/compare_identity.log

for tag in v15:8901:fastfill-v15 v1:8902:fastfill-v1; do
  IFS=: read -r t p m <<< "$tag"
  PYTHONPATH=src python3 -m fastfill_train.eval_layout \
    --records "$RECORDS" --samples "$SAMPLES" \
    --endpoint "http://127.0.0.1:$p/v1" --model "$m" \
    --dump-generations "out/gens_${t}.jsonl" --out "out/eval_${t}.json"
done

# Servers intentionally left running (manual-kill policy) — the user can
# re-query either arm; PIDs are printed by the EXIT trap.

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
