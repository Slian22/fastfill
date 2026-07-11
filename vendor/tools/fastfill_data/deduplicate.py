"""Cross-source geometry dedup for FastFill JSONL corpora (铁律 4).

Three of our sources share the 3D-FRONT upstream, so the same physical room
can enter the corpus several times. This tool groups samples by
``provenance.geometry_hash`` (category-free, frame-normalized — see
``fastfill/provenance.py``) and keeps exactly ONE sample per hash, preferring
the earliest source in ``--priority`` and breaking ties by first-seen order.

Two streaming passes over the inputs: pass 1 records only the winning
(file, line) reference per hash plus per-hash source sets; pass 2 re-streams
and copies winning lines verbatim, so full samples are never held in memory.

A dedup report JSON is written next to ``--out``: per-source input counts,
per-source kept counts, and the cross-source collision matrix (how many
hashes were seen in BOTH source a and source b — the three-source dedup
evidence). Malformed lines are skipped and counted, never silently dropped.

    python tools/fastfill_data/deduplicate.py \
        --in out/m3dlayout.jsonl out/il3d.jsonl --out out/deduped.jsonl \
        --priority m3dlayout,il3d
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterator, Sequence

# Script-mode bootstrap: make `fastfill_data` (tools/) and `scenesmith`
# (repo root, not pip-installed) importable when run as a plain script.
for _extra in (
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2],
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from pydantic import ValidationError  # noqa: E402

from scenesmith.growing_world.fastfill.schema import FastFillSample  # noqa: E402

LineRef = tuple[int, int]  # (input file index, 1-based line number)


@dataclass(frozen=True)
class _Winner:
    """Current best sample for one geometry hash."""

    rank: int  # position in --priority (len(priority) if unlisted)
    sequence: int  # global first-seen order (tie-break)
    ref: LineRef
    source: str


@dataclass
class _ScanState:
    """Pass-1 accumulator (mutable while scanning, then read-only)."""

    winners: dict[str, _Winner] = field(default_factory=dict)
    sources_by_hash: dict[str, set[str]] = field(default_factory=dict)
    input_counts: dict[str, int] = field(default_factory=dict)
    hashless_refs: set[LineRef] = field(default_factory=set)
    hashless_by_source: dict[str, int] = field(default_factory=dict)
    malformed_by_file: dict[str, int] = field(default_factory=dict)


def _iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(1-based line number, stripped non-empty line)``."""
    with path.open("r", encoding="utf-8") as fh:
        for number, raw in enumerate(fh, start=1):
            line = raw.strip()
            if line:
                yield number, line


def _priority_rank(source: str, priority: Sequence[str]) -> int:
    """Index in the priority list; unlisted sources rank after all listed."""
    try:
        return priority.index(source)
    except ValueError:
        return len(priority)


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _scan_line(
    state: _ScanState,
    line: str,
    ref: LineRef,
    sequence: int,
    file_name: str,
    priority: Sequence[str],
) -> None:
    """Classify one input line into winner / hashless / malformed buckets."""
    try:
        sample = FastFillSample.model_validate_json(line)
    except ValidationError:
        _bump(state.malformed_by_file, file_name)
        return
    source = sample.provenance.source_dataset
    _bump(state.input_counts, source)
    geometry_hash = sample.provenance.geometry_hash
    if not geometry_hash:  # ungroupable — pass through, never collapse
        state.hashless_refs.add(ref)
        _bump(state.hashless_by_source, source)
        return
    state.sources_by_hash.setdefault(geometry_hash, set()).add(source)
    candidate = _Winner(
        rank=_priority_rank(source, priority),
        sequence=sequence,
        ref=ref,
        source=source,
    )
    incumbent = state.winners.get(geometry_hash)
    if incumbent is None or (candidate.rank, candidate.sequence) < (
        incumbent.rank,
        incumbent.sequence,
    ):
        state.winners[geometry_hash] = candidate


def _scan_inputs(inputs: Sequence[Path], priority: Sequence[str]) -> _ScanState:
    """Pass 1: stream every input, keeping only refs and counters."""
    state = _ScanState()
    sequence = 0
    for file_index, path in enumerate(inputs):
        for line_number, line in _iter_lines(path):
            _scan_line(
                state,
                line,
                (file_index, line_number),
                sequence,
                path.name,
                priority,
            )
            sequence += 1
    return state


def _collision_matrix(sources_by_hash: dict[str, set[str]]) -> dict[str, int]:
    """``{"a&b": n}`` = number of hashes seen in BOTH source a and source b."""
    matrix: dict[str, int] = {}
    for sources in sources_by_hash.values():
        for a, b in combinations(sorted(sources), 2):
            _bump(matrix, f"{a}&{b}")
    return matrix


def _write_kept(inputs: Sequence[Path], keep_refs: set[LineRef], out_path: Path) -> int:
    """Pass 2: re-stream inputs and copy winning lines verbatim."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for file_index, path in enumerate(inputs):
            for line_number, line in _iter_lines(path):
                if (file_index, line_number) in keep_refs:
                    out.write(line + "\n")
                    written += 1
    return written


def _build_report(
    state: _ScanState, inputs: Sequence[Path], priority: Sequence[str]
) -> dict:
    per_source_kept = dict(state.hashless_by_source)
    for winner in state.winners.values():
        _bump(per_source_kept, winner.source)
    total_in = sum(state.input_counts.values())
    total_kept = sum(per_source_kept.values())
    return {
        "inputs": [str(p) for p in inputs],
        "priority": list(priority),
        "per_source_input": dict(sorted(state.input_counts.items())),
        "per_source_kept": dict(sorted(per_source_kept.items())),
        "collision_matrix": dict(
            sorted(_collision_matrix(state.sources_by_hash).items())
        ),
        "unique_hashes": len(state.winners),
        "duplicates_removed": total_in - total_kept,
        "no_geometry_hash": dict(sorted(state.hashless_by_source.items())),
        "malformed_lines": dict(sorted(state.malformed_by_file.items())),
    }


def deduplicate(
    inputs: Sequence[Path], out_path: Path, priority: Sequence[str] = ()
) -> dict:
    """Dedup ``inputs`` into ``out_path``; write and return the report."""
    state = _scan_inputs(inputs, priority)
    keep_refs = {w.ref for w in state.winners.values()} | state.hashless_refs
    _write_kept(inputs, keep_refs, out_path)
    report = _build_report(state, inputs, priority)
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dedup FastFill JSONL by provenance.geometry_hash (铁律 4)"
    )
    parser.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        required=True,
        help="input FastFillSample JSONL files",
    )
    parser.add_argument("--out", required=True, help="deduped JSONL path")
    parser.add_argument(
        "--priority",
        default="",
        help="comma-separated source_dataset order; earlier wins a collision",
    )
    args = parser.parse_args()

    priority = tuple(s for s in args.priority.split(",") if s)
    out_path = Path(args.out)
    report = deduplicate([Path(p) for p in args.inputs], out_path, priority)
    kept = sum(report["per_source_kept"].values())
    print(f"kept {kept} / {sum(report['per_source_input'].values())} -> {out_path}")
    print(f"report -> {out_path.with_suffix('.report.json')}")
    print(f"collision matrix: {report['collision_matrix']}")


if __name__ == "__main__":
    main()
