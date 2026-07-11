#!/usr/bin/env bash
# Sync the vendored FastFill contract + data tools from the scenesmith repo.
# Single source of truth = scenesmith; NEVER edit vendor/ by hand.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$HERE/../scenesmith}"

if [ ! -d "$SRC/scenesmith/growing_world/fastfill" ]; then
  echo "ERROR: scenesmith repo not found at $SRC (pass the path as arg 1)" >&2
  exit 1
fi

mkdir -p "$HERE/vendor/scenesmith/growing_world" "$HERE/vendor/tools"

# Package stubs (NOT the real __init__: the real growing_world __init__
# lazy-imports the Drake-bound world module; training must stay lean).
cat > "$HERE/vendor/scenesmith/__init__.py" << 'PY'
"""Vendored stub — real package lives in the scenesmith repo."""
PY
cat > "$HERE/vendor/scenesmith/growing_world/__init__.py" << 'PY'
"""Vendored stub — only the fastfill subpackage is vendored for training."""
PY

rsync -a --delete --exclude '__pycache__' \
  "$SRC/scenesmith/growing_world/fastfill/" \
  "$HERE/vendor/scenesmith/growing_world/fastfill/"
cp "$SRC/scenesmith/growing_world/hooks.py" \
  "$HERE/vendor/scenesmith/growing_world/hooks.py"
rsync -a --delete --exclude '__pycache__' \
  "$SRC/tools/fastfill_data/" "$HERE/vendor/tools/fastfill_data/"

SHA="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "synced from scenesmith @ $SHA" > "$HERE/vendor/VENDOR_VERSION"
echo "vendor synced from $SRC (scenesmith @ $SHA)"
