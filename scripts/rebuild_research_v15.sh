#!/usr/bin/env bash
# Research v1.5 rebuild: verify v1 -> full verified archive -> wipe ->
# convert ALL supported sources/floors -> dedup#1(+contamination) ->
# sanitize -> dedup#2 -> export -> snapshot. One shell, fail-fast end to end.
#
# Usage: bash scripts/rebuild_research_v15.sh
# After it finishes, train with scripts/train_research_v15.sh inside tmux.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

EXPECTED_COMMIT="${EXPECTED_COMMIT:-b92f325}"
test "$(git rev-parse --short HEAD)" = "$EXPECTED_COMMIT" \
  || { echo "ERROR: fastfill HEAD != $EXPECTED_COMMIT (git pull first)" >&2; exit 1; }

PRIORITY="m3dlayout,il3d,mansionworld,scenesmith_scenes,3d_synthplace"
LICENSE_ARGS=(--license-route research --allow-unresolved-licenses)

# --- Phase 0a: v1 identity verification --------------------------------------
for d in out/full_fp out/full_r128 out/full_r128_merged; do
  test -d "$d" || { echo "ERROR: missing $d — nothing to freeze" >&2; exit 1; }
done
bash scripts/verify_v1_identity.sh

# --- Phase 0b: COMPLETE verified archive (no silent failures) ----------------
ARCHIVE=~/v1_archive_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ARCHIVE/conv"
rsync -a --exclude='checkpoint-*' out/full_fp/  "$ARCHIVE/full_fp/"
rsync -a --exclude='checkpoint-*' out/full_r128/ "$ARCHIVE/full_r128/"
cp -a data/full "$ARCHIVE/data_full_v1"
ls out/conv/deduped*.jsonl >/dev/null 2>&1 \
  || { echo "ERROR: no out/conv/deduped*.jsonl to archive (old eval needs it)" >&2; exit 1; }
cp -a out/conv/deduped*.jsonl "$ARCHIVE/conv/"
# evidence files: copy whatever exists, but archive must not be empty of them
cp -a out/eval_*.json out/eval_*.uids.json "$ARCHIVE/" 2>/dev/null || true
cp -a out/identity_*.log train_*.log "$ARCHIVE/" 2>/dev/null || true

# Verify the archive BYTE-LEVEL before any deletion:
(cd "$ARCHIVE/full_fp" && sha256sum -c MANIFEST.sha256 --quiet)
if [ -f "$ARCHIVE/full_r128/MANIFEST.sha256" ]; then
  (cd "$ARCHIVE/full_r128" && sha256sum -c MANIFEST.sha256 --quiet)
fi
for f in data/full/*; do
  cmp -s "$f" "$ARCHIVE/data_full_v1/$(basename "$f")" \
    || { echo "ERROR: archive mismatch for $f" >&2; exit 1; }
done
for f in out/conv/deduped*.jsonl; do
  cmp -s "$f" "$ARCHIVE/conv/$(basename "$f")" \
    || { echo "ERROR: archive mismatch for $f" >&2; exit 1; }
done
printf '%s\n' "$ARCHIVE" > ~/.fastfill_v1_archive_path
echo "ARCHIVE-OK: $ARCHIVE"

# --- Phase 1: wipe -----------------------------------------------------------
if pgrep -af 'train_sft|vllm|accelerate' >/dev/null; then
  echo "ERROR: training/serving processes still running — stop them first" >&2
  exit 1
fi
rm -rf out data/sft data/stage0 data/stage0_aux data/full
mkdir -p out/conv
cat data/sceneeval_contamination.txt data/eval_rooms.txt > data/contamination_all.txt
echo "contamination list: $(grep -cv '^\s*#' data/contamination_all.txt || true) non-comment lines"

# --- Phase 2: dataset preflight ----------------------------------------------
total_floors=$(find data/MansionWorld/mansionworld -name 'floor_*.json' | wc -l)
first_floors=$(find data/MansionWorld/mansionworld -name 'floor_1.json' | wc -l)
echo "MansionWorld floors: $total_floors total, $first_floors floor_1"
test "$total_floors" -gt "$first_floors" \
  || { echo "ERROR: only floor_1 files found — all-floors data missing?" >&2; exit 1; }
find data/MansionWorld/mansionworld -name 'floor_*.json' \
  | grep -Ev '/floor_[0-9]+\.json$' \
  && { echo "ERROR: non-numeric floor_*.json present — inspect before converting" >&2; exit 1; } || true
for s in Room House NoAgentMemory NoAssetValidation NoCritic NoObserveScene NoSpecializedTools NotGenerated; do
  [ -d "data/scenesmith_scenes/$s" ] || echo "NOTE: scenesmith subset missing (will be skipped): $s"
done

# --- Phase 3: convert (atomic writers hard-fail on zero output) --------------
python3 vendor/tools/fastfill_data/convert_3d_synthplace.py --out out/conv/synthplace.jsonl
for subset in train val test; do
  python3 vendor/tools/fastfill_data/convert_m3dlayout.py --split all --subset "$subset" \
    --out "out/conv/m3dlayout_${subset}.jsonl"
done
python3 vendor/tools/fastfill_data/convert_il3d.py --out out/conv/il3d.jsonl
python3 vendor/tools/fastfill_data/convert_mansionworld.py --out out/conv/mansionworld.jsonl
python3 vendor/tools/fastfill_data/convert_scenesmith.py --out out/conv/scenesmith.jsonl
wc -l out/conv/*.jsonl

# --- Phase 4: dedup #1 (contamination closure lives HERE only) ---------------
python3 vendor/tools/fastfill_data/deduplicate.py \
  --in out/conv/synthplace.jsonl out/conv/m3dlayout_train.jsonl \
       out/conv/m3dlayout_val.jsonl out/conv/m3dlayout_test.jsonl \
       out/conv/il3d.jsonl out/conv/mansionworld.jsonl out/conv/scenesmith.jsonl \
  --out out/conv/deduped_raw.jsonl --priority "$PRIORITY" \
  "${LICENSE_ARGS[@]}" \
  --contamination-list data/contamination_all.txt
echo "--- contamination hits (non-empty list with all-zero matches = wrong ID format) ---"
jq '.contamination' out/conv/deduped_raw.report.json

# --- Phase 5-7: sanitize -> dedup#2 -> export -> snapshot --------------------
python3 vendor/tools/fastfill_data/sanitize.py --in out/conv/deduped_raw.jsonl \
  --out out/conv/sanitized.jsonl
python3 vendor/tools/fastfill_data/deduplicate.py --in out/conv/sanitized.jsonl \
  --out out/conv/deduped_research.jsonl --priority "$PRIORITY" "${LICENSE_ARGS[@]}"
python3 vendor/tools/fastfill_data/export_sft.py --in out/conv/deduped_research.jsonl \
  --out-dir data/sft_research_v15 --license-mode research --allow-unresolved-licenses
echo "--- export counts (surface_records should be in the thousands) ---"
jq '.counts | {floor_records, surface_records}' data/sft_research_v15/export_report.json
python3 -m fastfill_train.make_snapshot \
  --in data/sft_research_v15/floor_sft.jsonl data/sft_research_v15/surface_sft.jsonl \
  --out-dir data/full_research_v15 --license-mode research --allow-unresolved-licenses
jq '{snapshot_id, license_mode, counts, license_counts}' data/full_research_v15/SNAPSHOT.json

echo "=== REBUILD COMPLETE — next: tmux new -s research_v15_fp, then"
echo "    bash scripts/train_research_v15.sh"
