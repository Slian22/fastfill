"""Fail-closed acceptance for the canonical room_type re-export.

Usage:
    PYTHONPATH=src python3 scripts/reexport_acceptance.py sft      OLD_DIR NEW_DIR
    PYTHONPATH=src python3 scripts/reexport_acceptance.py snapshot OLD_DIR NEW_DIR

Exits non-zero on ANY violation. Allowed differences per record:
``room_type`` (canonicalized), the new ``room_type_raw`` field, and the
single ``room `` line inside ``input``. Everything else — uid order, split
membership, geometry, license, instruction, output, counts — must be
byte-identical between OLD_DIR and NEW_DIR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fastfill_train  # noqa: F401  (vendor path bootstrap — keep first)

from scenesmith.growing_world.fastfill.codec import canonicalize_room_type

ALLOWED_CHANGED = {"room_type", "input"}
ALLOWED_NEW = {"room_type_raw"}
SNAPSHOT_INVARIANT_KEYS = (
    "counts",
    "per_source_layer",
    "license_counts",
    "n_split_keys",
    "n_geometry_hashes",
    "seed",
    "val_fraction",
    "test_fraction",
    "license_mode",
)


def _fail(msg: str) -> None:
    print(f"ACCEPTANCE FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        _fail(f"missing file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _room_line_type(line: str) -> str:
    return line[len("room ") :].split(" id=", 1)[0].strip()


def _compare_records(old: list[dict], new: list[dict], where: str) -> int:
    if [r["uid"] for r in old] != [r["uid"] for r in new]:
        _fail(f"{where}: uid list or order differs")
    changed = 0
    for o, n in zip(old, new):
        uid = n["uid"]
        extra = set(n) - set(o)
        if extra - ALLOWED_NEW:
            _fail(f"{where} {uid}: unexpected new fields {sorted(extra - ALLOWED_NEW)}")
        missing = set(o) - set(n)
        if missing:
            _fail(f"{where} {uid}: fields dropped {sorted(missing)}")
        raw = n.get("room_type_raw")
        if raw is None:
            _fail(f"{where} {uid}: room_type_raw missing")
        if o.get("room_type") != raw:
            _fail(f"{where} {uid}: room_type_raw != old room_type")
        if n.get("room_type") != canonicalize_room_type(raw):
            _fail(f"{where} {uid}: room_type != canonicalize_room_type(room_type_raw)")
        for key in set(o):
            if key in ALLOWED_CHANGED:
                continue
            if o[key] != n[key]:
                _fail(f"{where} {uid}: field {key!r} changed")
        old_lines = str(o["input"]).split("\n")
        new_lines = str(n["input"]).split("\n")
        if len(old_lines) != len(new_lines):
            _fail(f"{where} {uid}: input line count changed")
        diffs = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
        if len(diffs) > 1:
            _fail(f"{where} {uid}: more than one input line changed")
        for a, b in diffs:
            if not (a.startswith("room ") and b.startswith("room ")):
                _fail(f"{where} {uid}: non-room input line changed: {a!r} -> {b!r}")
        room_lines = [line for line in new_lines if line.startswith("room ")]
        if not room_lines:
            _fail(f"{where} {uid}: no room line in input")
        if _room_line_type(room_lines[0]) != n["room_type"]:
            _fail(
                f"{where} {uid}: prompt room type {_room_line_type(room_lines[0])!r}"
                f" != room_type {n['room_type']!r}"
            )
        if o.get("room_type") != n.get("room_type"):
            changed += 1
    return changed


def _require_unique_uids(row_lists: list[list[dict]], where: str) -> None:
    seen: set[str] = set()
    for rows in row_lists:
        for r in rows:
            uid = str(r["uid"])
            if uid in seen:
                _fail(f"{where}: duplicate uid {uid}")
            seen.add(uid)


def check_sft(old_dir: Path, new_dir: Path) -> None:
    changed = 0
    new_lists = []
    for name in ("floor_sft.jsonl", "surface_sft.jsonl"):
        old, new = _read(old_dir / name), _read(new_dir / name)
        changed += _compare_records(old, new, name)
        new_lists.append(new)
    _require_unique_uids(new_lists, "sft")
    old_counts = json.loads((old_dir / "export_report.json").read_text())["counts"]
    new_counts = json.loads((new_dir / "export_report.json").read_text())["counts"]
    for key in ("input_samples", "floor_records", "surface_records"):
        if old_counts.get(key) != new_counts.get(key):
            _fail(
                f"export_report counts.{key} changed:"
                f" {old_counts.get(key)} -> {new_counts.get(key)}"
            )
    print(f"SFT ACCEPTANCE PASS: {changed} records canonicalized, all else identical")


def check_snapshot(old_dir: Path, new_dir: Path) -> None:
    changed = 0
    new_lists = []
    for name in ("train.jsonl", "heldout.jsonl", "test.jsonl"):
        old, new = _read(old_dir / name), _read(new_dir / name)
        changed += _compare_records(old, new, name)
        new_lists.append(new)
    _require_unique_uids(new_lists, "snapshot")
    old_snap = json.loads((old_dir / "SNAPSHOT.json").read_text())
    new_snap = json.loads((new_dir / "SNAPSHOT.json").read_text())
    for key in SNAPSHOT_INVARIANT_KEYS:
        if old_snap.get(key) != new_snap.get(key):
            _fail(f"SNAPSHOT.{key} changed")
    if new_snap.get("leakage_check") != "passed":
        _fail("SNAPSHOT.leakage_check != passed")
    if old_snap.get("snapshot_id") == new_snap.get("snapshot_id"):
        _fail("snapshot_id did NOT change — stale snapshot reused?")
    old_keys = json.loads((old_dir / "SPLIT_KEYS.json").read_text())
    new_keys = json.loads((new_dir / "SPLIT_KEYS.json").read_text())
    if old_keys["split_keys"] != new_keys["split_keys"]:
        _fail("SPLIT_KEYS.split_keys changed")
    if old_keys["geometry_hashes"] != new_keys["geometry_hashes"]:
        _fail("SPLIT_KEYS.geometry_hashes changed")
    if old_keys["record_hashes"] == new_keys["record_hashes"]:
        _fail("SPLIT_KEYS.record_hashes did NOT change — prompts not re-frozen?")
    print(
        f"SNAPSHOT ACCEPTANCE PASS: {changed} records canonicalized,"
        " membership/counts/keys identical, snapshot_id re-frozen"
    )


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("sft", "snapshot"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    mode, old_dir, new_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {"sft": check_sft, "snapshot": check_snapshot}[mode](old_dir, new_dir)


if __name__ == "__main__":
    main()
