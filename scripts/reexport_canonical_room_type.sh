#!/usr/bin/env bash
# Transactional re-export after the room_type canonicalization patch
# (worldedge ee7bf04, vendored here). ONLY export_sft + make_snapshot are
# re-run — convert/dedup/sanitize outputs and the Phase-0 archive are
# untouched. Never touches vLLM or training processes (user policy).
#
# Flow: preflight -> timestamped backup -> STAGING export -> record-level
# acceptance -> atomic swap -> STAGING snapshot (from the LIVE SFT path so
# SNAPSHOT.inputs matches training provenance) -> snapshot acceptance ->
# atomic swap -> gates. Fail-closed at every step; a mid-run crash leaves
# the live dirs either fully old or fully new, never mixed.
#
# Usage: bash scripts/reexport_canonical_room_type.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
trap 'echo "[$(date "+%F %T")] FAILED at ${BASH_SOURCE##*/}:${LINENO}: $BASH_COMMAND" >&2' ERR

STAMP="$(date +%Y%m%d_%H%M%S)"
SFT=data/sft_research_v15
SNAP=data/full_research_v15
BAK="backup_pre_canon_${STAMP}"
STG_SFT="${SFT}.staging_${STAMP}"
STG_SNAP="${SNAP}.staging_${STAMP}"

# ---------------------------------------------------------------- preflight
git merge-base --is-ancestor ac8d129 HEAD \
  || { echo "ERROR: HEAD lacks canonicalization commit ac8d129 — git pull" >&2; exit 1; }
grep -Fxq 'synced from scenesmith @ ee7bf04' vendor/VENDOR_VERSION \
  || { echo "ERROR: vendor not synced @ ee7bf04:" >&2; cat vendor/VENDOR_VERSION >&2; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" \
  || { echo "ERROR: working tree has uncommitted changes" >&2; exit 1; }
test -s out/conv/deduped_research.jsonl \
  || { echo "ERROR: out/conv/deduped_research.jsonl missing/empty — run 块② first" >&2; exit 1; }
test -f "$SFT/export_report.json" \
  || { echo "ERROR: $SFT missing — run 块② first" >&2; exit 1; }
test -f "$SNAP/SNAPSHOT.json" \
  || { echo "ERROR: $SNAP missing — run 块② first" >&2; exit 1; }
command -v jq >/dev/null || { echo "ERROR: jq not found" >&2; exit 1; }

# ------------------------------------------------------------------- backup
mkdir "$BAK"
cp -a "$SFT" "$BAK/sft_research_v15"
cp -a "$SNAP" "$BAK/full_research_v15"
echo "[$(date '+%F %T')] backup: $BAK (KEEP until 块⑤ sensitivity test is done)"

# ---------------------------------------------------- staging export + check
echo "[$(date '+%F %T')] staging export -> $STG_SFT"
python3 vendor/tools/fastfill_data/export_sft.py \
  --in out/conv/deduped_research.jsonl --out-dir "$STG_SFT" \
  --license-mode research --allow-unresolved-licenses
python3 scripts/reexport_acceptance.py sft "$BAK/sft_research_v15" "$STG_SFT"

# atomic swap: live dir is old XOR new at every instant
mv "$SFT" "${SFT}.old_${STAMP}"
mv "$STG_SFT" "$SFT"
rm -rf "${SFT}.old_${STAMP}"
echo "[$(date '+%F %T')] SFT swapped in"

# -------------------------------------------------- staging snapshot + check
echo "[$(date '+%F %T')] staging snapshot -> $STG_SNAP"
python3 -m fastfill_train.make_snapshot \
  --in "$SFT/floor_sft.jsonl" "$SFT/surface_sft.jsonl" \
  --out-dir "$STG_SNAP" --license-mode research --allow-unresolved-licenses
python3 scripts/reexport_acceptance.py snapshot "$BAK/full_research_v15" "$STG_SNAP"

mv "$SNAP" "${SNAP}.old_${STAMP}"
mv "$STG_SNAP" "$SNAP"
rm -rf "${SNAP}.old_${STAMP}"
echo "[$(date '+%F %T')] snapshot swapped in"

# -------------------------------------------------- gates (mirror rebuild's)
surface_n=$(jq '.counts.surface_records' "$SFT/export_report.json")
test "$surface_n" -ge "${MIN_SURFACE_RECORDS:-1000}" \
  || { echo "ERROR: surface_records=$surface_n < ${MIN_SURFACE_RECORDS:-1000}" >&2; exit 1; }
intern_n=$(jq '[.per_source_layer[]? | to_entries[] | select(.key | startswith("internscenes/")) | .value] | add // 0' \
  "$SNAP/SNAPSHOT.json")
test "$intern_n" -gt 0 \
  || { echo "ERROR: internscenes contributed 0 records to the snapshot" >&2; exit 1; }

jq '{snapshot_id, counts}' "$SNAP/SNAPSHOT.json"
echo "=== CANONICAL RE-EXPORT COMPLETE ==="
echo "next: bash scripts/leak_check_v15.sh"
