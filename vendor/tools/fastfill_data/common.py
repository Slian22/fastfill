"""Shared infrastructure for FastFill data converters.

Every converter in this directory follows the same shape:

    python tools/fastfill_data/convert_<source>.py \
        --data-dir <...>/data --out out/<source>.jsonl --limit 100

and emits one :class:`FastFillSample` per line (JSONL), with provenance
hashes stamped (铁律 4). Conversion is best-effort per record: a malformed
source record is skipped and counted, never fabricated. Fields we cannot
derive from the source (task, expected_*) stay EMPTY — back-annotation is a
separate later stage (T2.6).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from scenesmith.growing_world.fastfill.provenance import stamp_hashes
from scenesmith.growing_world.fastfill.schema import FastFillSample

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT.parent / "data"


def resolve_data_dir(cli_value: str | None = None) -> Path:
    """Data root resolution order: CLI flag > $WORLDEDGE_DATA_DIR > ../data."""
    if cli_value:
        return Path(cli_value).expanduser()
    env = os.environ.get("WORLDEDGE_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_DATA_DIR


@dataclass
class ConversionStats:
    """Per-run accounting so silent drops are impossible (no-silent-caps)."""

    converted: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str, count: int = 1) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + count

    def to_dict(self) -> dict:
        return {"converted": self.converted, "skipped": dict(self.skipped)}


def finalize_sample(sample: FastFillSample) -> FastFillSample:
    """Stamp provenance hashes + split key onto a converted sample."""
    stamped = stamp_hashes(
        sample.provenance,
        sample.room_context.floor_polygon,
        sample.layout.floor_layout.objects,
    )
    return sample.model_copy(
        update={
            "provenance": stamped,
            "layout": sample.layout.model_copy(update={"meta": stamped}),
        }
    )


def write_jsonl(samples: Iterable[FastFillSample], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(sample.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[FastFillSample]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield FastFillSample.model_validate_json(line)


def converter_main(
    description: str,
    convert: Callable[[Path, int | None, ConversionStats], Iterator[FastFillSample]],
    default_out: str,
) -> None:
    """Standard converter CLI: parse args, run, write JSONL + stats report."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", default=None, help="dataset root")
    parser.add_argument("--out", default=default_out, help="output JSONL path")
    parser.add_argument(
        "--limit", type=int, default=None, help="max samples (None = all)"
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    stats = ConversionStats()
    out_path = Path(args.out)
    n = write_jsonl(
        (finalize_sample(s) for s in convert(data_dir, args.limit, stats)),
        out_path,
    )
    stats.converted = n
    report_path = out_path.with_suffix(".stats.json")
    report_path.write_text(json.dumps(stats.to_dict(), indent=2))
    print(f"wrote {n} samples -> {out_path}")
    print(f"stats -> {report_path}: {stats.to_dict()}")
    if n == 0:
        # A missing/mis-mounted dataset root yields 0 samples and would
        # otherwise overwrite the output and exit 0 — a rebuild that ran
        # this way silently drops the whole source.
        raise SystemExit(
            f"ERROR: converter produced 0 samples from {data_dir} — dataset "
            f"root missing or mis-mounted? ({out_path} now has 0 rows)"
        )
