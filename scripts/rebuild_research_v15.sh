#!/usr/bin/env bash
# Research v1.5 rebuild: verify v1 -> full verified archive -> preflight ->
# wipe -> convert ALL supported sources/floors -> dedup#1(+contamination) ->
# sanitize -> dedup#2 -> export (gated) -> snapshot. Fail-fast end to end.
#
# ALL static preflight happens BEFORE anything is deleted. Re-run safe: if
# v1 was already archived by a previous attempt (out/ wiped), it resumes
# from the conversion phase instead of failing identity verification.
#
# Usage: bash scripts/rebuild_research_v15.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

# The DATA CONTRACT commit (MansionWorld all-floors converter) must be an
# ancestor of HEAD — never pin HEAD itself, script fixes move it.
DATA_CONTRACT_COMMIT="${DATA_CONTRACT_COMMIT:-b92f325}"
git merge-base --is-ancestor "$DATA_CONTRACT_COMMIT" HEAD \
  || { echo "ERROR: HEAD does not contain data-contract commit $DATA_CONTRACT_COMMIT (git pull)" >&2; exit 1; }

PRIORITY="m3dlayout,il3d,mansionworld,scenesmith_scenes,3d_synthplace"
LICENSE_ARGS=(--license-route research --allow-unresolved-licenses)

# =============================================================================
# STATIC PREFLIGHT — no deletion, no writes beyond the contamination file.
# =============================================================================
for f in data/sceneeval_contamination.txt data/eval_rooms.txt; do
  test -f "$f" || { echo "ERROR: missing $f" >&2; exit 1; }
done
for d in data/MansionWorld/mansionworld data/M3DLayout data/IL3D \
         data/3D-SynthPlace_indoor_scenes_dataset data/scenesmith_scenes; do
  test -d "$d" || { echo "ERROR: missing dataset dir $d" >&2; exit 1; }
done
# MansionWorld floors at the SAME depth the converter globs (building root
# only — assets/floor_N_object_states.json lives deeper and is legitimate).
total_floors=$(find data/MansionWorld/mansionworld -mindepth 2 -maxdepth 2 -name 'floor_*.json' | wc -l)
first_floors=$(find data/MansionWorld/mansionworld -mindepth 2 -maxdepth 2 -name 'floor_1.json' | wc -l)
echo "MansionWorld building-root floors: $total_floors total, $first_floors floor_1"
test "$total_floors" -gt "$first_floors" \
  || { echo "ERROR: only floor_1 files at building root — all-floors data missing?" >&2; exit 1; }
bad_floors=$(find data/MansionWorld/mansionworld -mindepth 2 -maxdepth 2 -name 'floor_*.json' \
  | grep -Ev '/floor_[0-9]+\.json$' || true)
test -z "$bad_floors" \
  || { echo "ERROR: non-numeric floor files at building root:" >&2; echo "$bad_floors" >&2; exit 1; }
for s in Room House NoAgentMemory NoAssetValidation NoCritic NoObserveScene NoSpecializedTools NotGenerated; do
  [ -d "data/scenesmith_scenes/$s" ] || echo "NOTE: scenesmith subset missing (will be skipped): $s"
done
cat data/sceneeval_contamination.txt data/eval_rooms.txt > data/contamination_all.txt
CONTAM_LINES=$(grep -cv -e '^[[:space:]]*#' -e '^[[:space:]]*$' data/contamination_all.txt || true)
echo "contamination list: $CONTAM_LINES non-comment lines"

# =============================================================================
# Phase 0 — v1 identity verification + COMPLETE byte-verified archive.
# Re-run mode: skip when a previous attempt already archived and wiped.
# =============================================================================
if [ -d out/full_fp ]; then
  for d in out/full_fp out/full_r128 out/full_r128_merged; do
    test -d "$d" || { echo "ERROR: missing $d — nothing to freeze" >&2; exit 1; }
  done
  bash scripts/verify_v1_identity.sh

  ARCHIVE=~/v1_archive_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$ARCHIVE/conv"
  rsync -a --exclude='checkpoint-*' out/full_fp/         "$ARCHIVE/full_fp/"
  rsync -a --exclude='checkpoint-*' out/full_r128/       "$ARCHIVE/full_r128/"
  rsync -a --exclude='checkpoint-*' out/full_r128_merged/ "$ARCHIVE/full_r128_merged/"
  cp -a data/full "$ARCHIVE/data_full_v1"
  ls out/conv/deduped*.jsonl >/dev/null 2>&1 \
    || { echo "ERROR: no out/conv/deduped*.jsonl to archive (old eval needs it)" >&2; exit 1; }
  cp -a out/conv/deduped*.jsonl "$ARCHIVE/conv/"
  cp -a out/eval_*.json out/eval_*.uids.json "$ARCHIVE/" 2>/dev/null || true
  cp -a out/identity_*.log train_*.log "$ARCHIVE/" 2>/dev/null || true

  (cd "$ARCHIVE/full_fp" && sha256sum -c MANIFEST.sha256 --quiet)
  for m in full_r128 full_r128_merged; do
    if [ -f "$ARCHIVE/$m/MANIFEST.sha256" ]; then
      (cd "$ARCHIVE/$m" && sha256sum -c MANIFEST.sha256 --quiet)
    fi
  done
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

  # Phase 1 — wipe (only after a verified archive).
  if pgrep -u "$USER" -af 'train_sft|vllm|accelerate' >/dev/null; then
    echo "ERROR: training/serving processes still running — stop them first" >&2
    exit 1
  fi
  rm -rf out data/sft data/stage0 data/stage0_aux data/full
elif [ -f ~/.fastfill_v1_archive_path ] && [ -d "$(cat ~/.fastfill_v1_archive_path)/full_fp" ]; then
  echo "RESUME MODE: v1 already archived at $(cat ~/.fastfill_v1_archive_path); continuing"
else
  echo "ERROR: out/full_fp missing and no verified archive recorded — refusing" >&2
  exit 1
fi
mkdir -p out/conv

# =============================================================================
# Phase 3 — convert (atomic writers hard-fail on zero output).
# =============================================================================
python3 vendor/tools/fastfill_data/convert_3d_synthplace.py --out out/conv/synthplace.jsonl
for subset in train val test; do
  python3 vendor/tools/fastfill_data/convert_m3dlayout.py --split all --subset "$subset" \
    --out "out/conv/m3dlayout_${subset}.jsonl"
done
python3 vendor/tools/fastfill_data/convert_il3d.py --out out/conv/il3d.jsonl
python3 vendor/tools/fastfill_data/convert_mansionworld.py --out out/conv/mansionworld.jsonl
python3 vendor/tools/fastfill_data/convert_scenesmith.py --out out/conv/scenesmith.jsonl
wc -l out/conv/*.jsonl

# =============================================================================
# Phase 4 — dedup #1 (contamination closure lives HERE only) + hit gate.
# =============================================================================
python3 vendor/tools/fastfill_data/deduplicate.py \
  --in out/conv/synthplace.jsonl out/conv/m3dlayout_train.jsonl \
       out/conv/m3dlayout_val.jsonl out/conv/m3dlayout_test.jsonl \
       out/conv/il3d.jsonl out/conv/mansionworld.jsonl out/conv/scenesmith.jsonl \
  --out out/conv/deduped_raw.jsonl --priority "$PRIORITY" \
  "${LICENSE_ARGS[@]}" \
  --contamination-list data/contamination_all.txt
jq '.contamination' out/conv/deduped_raw.report.json
hits=$(jq '[.contamination.direct_matches_by_source // {} | .[]] | add // 0' out/conv/deduped_raw.report.json)
if [ "$CONTAM_LINES" -gt 0 ] && [ "$hits" -eq 0 ] \
   && [ "${ALLOW_ZERO_CONTAMINATION_HITS:-0}" != "1" ]; then
  echo "ERROR: contamination list has $CONTAM_LINES ids but ZERO direct matches —" >&2
  echo "  likely wrong ID format (SceneEval display names vs source_room_id)." >&2
  echo "  Fix the list, or re-run with ALLOW_ZERO_CONTAMINATION_HITS=1 if the" >&2
  echo "  ids are genuinely absent from this corpus." >&2
  exit 1
fi

# =============================================================================
# Phase 5-7 — sanitize -> dedup#2 -> export (surface gate) -> snapshot.
# =============================================================================
python3 vendor/tools/fastfill_data/sanitize.py --in out/conv/deduped_raw.jsonl \
  --out out/conv/sanitized.jsonl
python3 vendor/tools/fastfill_data/deduplicate.py --in out/conv/sanitized.jsonl \
  --out out/conv/deduped_research.jsonl --priority "$PRIORITY" "${LICENSE_ARGS[@]}"
python3 vendor/tools/fastfill_data/export_sft.py --in out/conv/deduped_research.jsonl \
  --out-dir data/sft_research_v15 --license-mode research --allow-unresolved-licenses
jq '.counts | {floor_records, surface_records}' data/sft_research_v15/export_report.json
surface_n=$(jq '.counts.surface_records' data/sft_research_v15/export_report.json)
MIN_SURFACE="${MIN_SURFACE_RECORDS:-1000}"
test "$surface_n" -ge "$MIN_SURFACE" \
  || { echo "ERROR: surface_records=$surface_n < $MIN_SURFACE — the whole point of" >&2; \
       echo "  this retrain is the Surface corpus; investigate before freezing." >&2; exit 1; }
python3 -m fastfill_train.make_snapshot \
  --in data/sft_research_v15/floor_sft.jsonl data/sft_research_v15/surface_sft.jsonl \
  --out-dir data/full_research_v15 --license-mode research --allow-unresolved-licenses
jq '{snapshot_id, license_mode, counts, license_counts}' data/full_research_v15/SNAPSHOT.json

echo "=== REBUILD COMPLETE — next: tmux new -s research_v15_fp, then"
echo "    bash scripts/train_research_v15.sh"
