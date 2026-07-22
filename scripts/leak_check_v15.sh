#!/usr/bin/env bash
# Hardened 块③ leakage check before 块④ training.
#
# Checks v1 TRAIN against BOTH v1.5 heldout and v1.5 test, on BOTH
# split_key and geometry_hash. Fail-closed: empty key sets, records missing
# split_key, or ANY overlap -> exit 1. v1-era records without geometry_hash
# also fail unless ALLOW_V1_NO_GEOMETRY_HASH=1 is set explicitly.
#
# Usage: bash scripts/leak_check_v15.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."
trap 'echo "[$(date "+%F %T")] FAILED at ${BASH_SOURCE##*/}:${LINENO}: $BASH_COMMAND" >&2' ERR

V1_ARCHIVE=""
for f in archive/LATEST_V1 "$HOME/.fastfill_v1_archive_path"; do
  if [ -f "$f" ] && [ -d "$(cat "$f")/data_full_v1" ]; then V1_ARCHIVE="$(cat "$f")"; break; fi
done
test -n "$V1_ARCHIVE" || { echo "ERROR: no v1 archive with data_full_v1 found" >&2; exit 1; }
test -f data/full_research_v15/heldout.jsonl || { echo "ERROR: v1.5 heldout missing" >&2; exit 1; }
test -f data/full_research_v15/test.jsonl || { echo "ERROR: v1.5 test missing" >&2; exit 1; }
echo "v1 archive: $V1_ARCHIVE"

python3 - "$V1_ARCHIVE/data_full_v1/train.jsonl" \
  data/full_research_v15/heldout.jsonl data/full_research_v15/test.jsonl <<'PY'
import json
import os
import sys

def load(path):
    n = missing = 0
    keys, hashes = set(), set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            key = str(r.get("split_key") or "").strip()
            if key:
                keys.add(key)
            else:
                missing += 1
            h = str(r.get("geometry_hash") or "").strip()
            if h:
                hashes.add(h)
    return n, keys, hashes, missing

failures = []
def require(cond, msg):
    if not cond:
        failures.append(msg)
        print(f"FAIL: {msg}", file=sys.stderr)

v1_path, ho_path, te_path = sys.argv[1:4]
v1_n, v1_keys, v1_hashes, v1_missing = load(v1_path)
print(f"v1_train: n={v1_n} split_keys={len(v1_keys)} geometry_hashes={len(v1_hashes)}")
require(v1_n > 0, "v1 train is empty")
require(len(v1_keys) > 0, "v1 train has zero split_keys")
require(v1_missing == 0, f"v1 train has {v1_missing} records without split_key")
if not v1_hashes and os.environ.get("ALLOW_V1_NO_GEOMETRY_HASH") != "1":
    require(False, "v1 train has zero geometry_hashes "
                   "(set ALLOW_V1_NO_GEOMETRY_HASH=1 only if the v1-era schema truly lacks it)")

for name, path in (("heldout", ho_path), ("test", te_path)):
    n, keys, hashes, missing = load(path)
    require(n > 0, f"v1.5 {name} is empty")
    require(len(keys) > 0, f"v1.5 {name} has zero split_keys")
    require(missing == 0, f"v1.5 {name} has {missing} records without split_key")
    require(len(hashes) > 0, f"v1.5 {name} has zero geometry_hashes")
    key_overlap = v1_keys & keys
    require(not key_overlap, f"split_key overlap v1_train x {name}: {len(key_overlap)}")
    hash_overlap = v1_hashes & hashes
    require(not hash_overlap, f"geometry_hash overlap v1_train x {name}: {len(hash_overlap)}")
    print(f"v1.5 {name}: n={n} split_keys={len(keys)} geometry_hashes={len(hashes)} "
          f"key_overlap={len(key_overlap)} hash_overlap={len(hash_overlap)}")

if failures:
    print(f"LEAK-CHECK FAIL: {len(failures)} problem(s)", file=sys.stderr)
    sys.exit(1)
print("LEAK-CHECK PASS: overlap=0 (split_key + geometry_hash, heldout + test)")
PY
