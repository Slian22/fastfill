"""Build FastFill v1.1 dataset from FastFill v3.2 dataset (.release/v3.2/data/v3.2).

What this script does:
1. Reads `{train,dev,test}.jsonl` from `--src` (e.g., `.release/v3.2/data/v3.2`).
2. Converts every message triple from v1 (size in `user["objects"]`) to v1.1 (no size in `user["objects"]`,
   `"size": [sx,sy,sz]` in `assistant["placements"]`, while preserving `user["fixed"]` doors/windows/columns).
3. Runs `fastfill.validate.check` on every training room and augments `row["flags"]` with:
   - `"main_collision"`: count of cross-category/same-category main collisions (if > 0)
   - `"severe_collision"`: count of severe (>30% footprint overlap) main collisions (if > 0)
   This fixes the v1/v3.2 dataset blind spot where `_bad_overlap()` only flagged same-category overlaps and left
   33.5% of default-kept training rooms with cross-category collisions.
4. Copies `{dev,test}_rooms.jsonl` and `{dev,test}_constrained_rooms.jsonl` and writes `MANIFEST.json` and
   `collision_audit.json`.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from fastfill.v1_1.scene_v1_1 import convert_v1_messages_to_v1_1
from fastfill.validate import check


def _process_chunk(lines: list[str], audit_collisions: bool) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for line in lines:
        r = json.loads(line)
        u_v1 = json.loads(r["messages"][1]["content"])
        a_v1 = json.loads(r["messages"][2]["content"])
        flags = dict(r.get("flags") or {})
        if audit_collisions:
            pl = {p["id"]: p for p in a_v1["placements"]}
            objs = []
            for o in u_v1["objects"]:
                p = pl[o["id"]]
                cat = o["id"].rsplit("_", 1)[0].replace("_", " ")
                objs.append(
                    {
                        "id": o["id"],
                        "category": cat,
                        "size": o["size"],
                        "pos": p["pos"],
                        "yaw": math.radians(p["yaw"]),
                        "parent": p.get("on"),
                    }
                )
            fixed = [
                {
                    "id": fx["id"],
                    "category": fx["id"].rsplit("_", 1)[0].replace("_", " "),
                    "size": fx["size"],
                    "pos": fx["pos"],
                    "yaw": math.radians(fx["yaw"]),
                }
                for fx in u_v1.get("fixed", [])
            ]
            room = {
                "boundary_type": u_v1["boundary_type"],
                "boundary": u_v1["boundary"],
                "height": u_v1.get("height"),
                "fixed": fixed,
                "objects": objs,
            }
            c = check(room)
            if c["collisions"]["main"]:
                flags["main_collision"] = len(c["collisions"]["main"])
            if c["collisions"]["severe"]:
                flags["severe_collision"] = len(c["collisions"]["severe"])
        new_msgs = convert_v1_messages_to_v1_1(r["messages"])
        new_row = {
            "uid": r["uid"],
            "source": r["source"],
            "flags": flags,
            "messages": new_msgs,
        }
        out.append((json.dumps(new_row, ensure_ascii=False, allow_nan=False) + "\n", flags))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path(".release/v3.2/data/v3.2"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0, help="0 = full dataset")
    args = ap.parse_args()

    if args.workers < 1 or args.limit < 0:
        raise ValueError('workers must be positive and limit must be nonnegative')
    names = ('train', 'dev', 'test', 'dev_rooms', 'test_rooms',
             'dev_constrained_rooms', 'test_constrained_rooms')
    files = [args.src / f'{name}.jsonl' for name in names]
    for path in files:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f'Required nonempty source input: {path}')
    destination = args.out.absolute()
    _validate_destination(destination)
    source_sha = {path.name: _sha256(path) for path in files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / f'.{destination.name}.build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _validate_destination(destination)
        with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.', dir=destination.parent) as temporary:
            staged = argparse.Namespace(**{**vars(args), 'out': Path(temporary)})
            _build(staged, source_sha)
            if any(_sha256(path) != source_sha[path.name] for path in files):
                raise ValueError('Source input changed during conversion; output not published')
            _validate_destination(destination)
            if destination.exists():
                destination.rmdir()  # only a verified empty directory is replaceable
            os.rename(temporary, destination)
    print('Published dataset to', destination, flush=True)


def _validate_destination(path: Path) -> None:
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError(f'Refusing to replace existing dataset; choose an absent or empty output: {path}')


def _build(args: argparse.Namespace, source_sha: dict[str, str]) -> None:

    args.out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {}

    for split in ("train", "dev", "test"):
        src_file = args.src / f"{split}.jsonl"
        dst_file = args.out / f"{split}.jsonl"
        with open(src_file, encoding="utf-8") as f:
            if args.limit > 0:
                lines = [ line for _, line in zip(range(args.limit), f) ]
            else:
                lines = f.readlines()
        chunk_size = max(200, len(lines) // (args.workers * 8) + 1)
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
        flag_counter: Counter[str] = Counter()
        clean_v1_default = 0
        clean_no_severe = 0
        clean_no_main = 0
        v1_excl = {"oob_objects", "overlapping_furniture", "fixed_collision"}
        no_severe_excl = v1_excl | {"severe_collision"}
        no_main_excl = v1_excl | {"severe_collision", "main_collision"}

        with ProcessPoolExecutor(max_workers=args.workers) as ex, open(dst_file, "w", encoding="utf-8") as out_f:
            for res in ex.map(_process_chunk, chunks, [True] * len(chunks)):
                for s_line, fl in res:
                    out_f.write(s_line)
                    for k in fl:
                        flag_counter[k] += 1
                    fset = set(fl.keys())
                    if not (fset & v1_excl):
                        clean_v1_default += 1
                    if not (fset & no_severe_excl):
                        clean_no_severe += 1
                    if not (fset & no_main_excl):
                        clean_no_main += 1

        stats[split] = {
            "total_rows": len(lines),
            "clean_v1_default": clean_v1_default,
            "clean_no_severe_collision": clean_no_severe,
            "clean_zero_main_collision": clean_no_main,
            "flags": dict(flag_counter.most_common()),
        }
        print(f"[{split}] {len(lines)} rows -> clean_v1={clean_v1_default}, no_severe={clean_no_severe}, zero_main={clean_no_main}", flush=True)

    for room_file in ("dev_rooms.jsonl", "test_rooms.jsonl", "dev_constrained_rooms.jsonl", "test_constrained_rooms.jsonl"):
        src_rf = args.src / room_file
        # Validate every copied row; malformed room JSON must never receive a manifest.
        with src_rf.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get('objects'), list) or not record['objects']:
                    raise ValueError(f'Invalid room record at {src_rf}:{number}')
        if args.limit > 0:
            with open(src_rf, encoding="utf-8") as sf, open(args.out / room_file, "w", encoding="utf-8") as df:
                for _, line in zip(range(args.limit), sf):
                    df.write(line)
        else:
            shutil.copy2(src_rf, args.out / room_file)

    manifest = {
        "protocol": "fastfill_v1.1",
        "source_dataset": str(args.src),
        "source_sha256": source_sha,
        "stats": stats,
        "files": {
            p.name: {"sha256": _sha256(p)}
            for p in sorted(args.out.glob("*.jsonl"))
        },
    }
    with open(args.out / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote MANIFEST.json to", args.out / "MANIFEST.json", flush=True)


if __name__ == "__main__":
    main()
