"""Export deduped FastFill JSONL to SFT training records (floor + surface).

Reads ``FastFillSample`` JSONL — normally the output of the full chain
``convert -> sanitize.py -> deduplicate.py`` (samples missing the
``sanitized=v1`` note are counted as ``unsanitized_input_samples``) — and
emits two files under ``--out-dir``:

- ``floor_sft.jsonl``   — one record per sample: RoomContext codec text in,
  floor-layout codec text out;
- ``surface_sft.jsonl`` — one record per stored ``SurfaceObjectGroup``: a
  minimal ``SupportContext`` (rebuilt from the sample's stored surfaces and
  parent furniture) codec text in, single-group codec text out.

Filters (each excluded/skipped sample is COUNTED in the export report,
never silently dropped):

- ``--contamination-list`` — file with one ``source_room_id`` per line;
  matching samples are excluded (铁律 1 eval-contamination hook);
- ``--exclude-nc`` (default on) — CC BY-NC licensed samples are excluded
  from training (铁律 2);
- samples with zero floor objects are skipped;
- surface groups whose surface or parent furniture cannot be resolved from
  the stored layout are skipped (nothing is fabricated).

    python tools/fastfill_data/export_sft.py \
        --in out/deduped.jsonl --out-dir out/sft \
        --contamination-list eval_rooms.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

# Script-mode bootstrap: make `fastfill_data` (tools/) and `scenesmith`
# (repo root, not pip-installed) importable when run as a plain script.
for _extra in (
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2],
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from fastfill_data.common import read_jsonl  # noqa: E402
from fastfill_data.sanitize import SANITIZED_NOTE  # noqa: E402
from scenesmith.growing_world.fastfill.codec import (  # noqa: E402
    FLOOR_INSTRUCTION,
    SURFACE_INSTRUCTION,
    encode_floor_layout,
    encode_room_context,
    encode_support_context,
    encode_surface_groups,
)
from scenesmith.growing_world.fastfill.schema import (  # noqa: E402
    FastFillSample,
    LicenseTag,
    SupportContext,
    SurfaceObjectGroup,
)

FLOOR_FILE = "floor_sft.jsonl"
SURFACE_FILE = "surface_sft.jsonl"
REPORT_FILE = "export_report.json"


def load_contamination_list(path: Path | None) -> frozenset[str]:
    """One ``source_room_id`` per line; blank lines ignored."""
    if path is None:
        return frozenset()
    lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip())


def build_support_context(
    sample: FastFillSample, group: SurfaceObjectGroup
) -> SupportContext | None:
    """Minimal SupportContext from the sample's stored surface inventory.

    ``None`` when the group's surface or the surface's parent furniture is
    missing from the stored layout — such groups are skipped and counted.
    """
    floor = sample.layout.floor_layout
    surface = next(
        (s for s in floor.support_surfaces if s.surface_id == group.surface_id),
        None,
    )
    if surface is None:
        return None
    parent = next(
        (o for o in floor.objects if o.object_id == surface.parent_object_id),
        None,
    )
    if parent is None:
        return None
    return SupportContext(
        surface=surface,
        parent_category=parent.category,
        parent_dimensions=parent.dimensions,
        room_type=sample.room_context.room_type,
    )


def _floor_record(sample: FastFillSample) -> dict:
    return {
        "uid": sample.sample_id,
        "split_key": sample.provenance.split_key,
        "source_dataset": sample.provenance.source_dataset,
        "room_type": sample.room_context.room_type,
        "layer": "floor",
        "instruction": FLOOR_INSTRUCTION,
        "input": encode_room_context(sample.room_context),
        "output": encode_floor_layout(sample.layout.floor_layout),
    }


def _surface_record(
    sample: FastFillSample, group: SurfaceObjectGroup, context: SupportContext
) -> dict:
    return {
        "uid": f"{sample.sample_id}#{group.group_id}",
        "split_key": sample.provenance.split_key,
        "source_dataset": sample.provenance.source_dataset,
        "room_type": sample.room_context.room_type,
        "layer": "surface",
        "instruction": SURFACE_INSTRUCTION,
        "input": encode_support_context(context),
        "output": encode_surface_groups([group]),
    }


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _exclusion_reason(
    sample: FastFillSample, exclude_nc: bool, contamination: frozenset[str]
) -> str | None:
    """Sample-level filter (checked in 铁律 order), ``None`` = exportable."""
    room_id = sample.provenance.source_room_id
    house_id = sample.provenance.source_house_id
    if (room_id and room_id in contamination) or (
        house_id and house_id in contamination
    ):  # scenesmith scenes key on house_id (scene_XXX); room_id is a room name
        return "excluded_contamination"
    if exclude_nc and sample.provenance.license_tag is LicenseTag.CC_BY_NC:
        return "excluded_nc_license"
    if not sample.layout.floor_layout.objects:
        return "skipped_empty_floor"
    return None


def _export_surface_groups(
    sample: FastFillSample, surface_out: TextIO, counts: dict[str, int]
) -> None:
    for group in sample.layout.surface_groups:
        if not group.objects:
            _bump(counts, "skipped_surface_group_empty")
            continue
        context = build_support_context(sample, group)
        if context is None:
            _bump(counts, "skipped_surface_group_unresolved")
            continue
        try:
            record = _surface_record(sample, group, context)
        except ValueError:  # codec-illegal token (e.g. '|' in a category)
            _bump(counts, "skipped_surface_group_codec_error")
            continue
        surface_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        _bump(counts, "surface_records")


UNVERIFIED_YAW_NOTE = "yaw_facing=unverified_per_asset"


def _export_sample(
    sample: FastFillSample,
    floor_out: TextIO,
    surface_out: TextIO,
    counts: dict[str, int],
    exclude_nc: bool,
    contamination: frozenset[str],
    exclude_unverified_yaw: bool,
) -> None:
    _bump(counts, "input_samples")
    if SANITIZED_NOTE not in sample.provenance.notes:
        _bump(counts, "unsanitized_input_samples")
    reason = _exclusion_reason(sample, exclude_nc, contamination)
    if reason is not None:
        _bump(counts, reason)
        return
    # Floor yaw labels from sources without a verified facing convention
    # (e.g. MansionWorld: per-asset canonical fronts) would poison facing
    # supervision — skip the FLOOR record only. Surface records stay: their
    # local frames are parent-relative, so intra-surface poses are consistent
    # regardless of the parent's absolute facing.
    skip_floor = (
        exclude_unverified_yaw and UNVERIFIED_YAW_NOTE in sample.provenance.notes
    )
    if skip_floor:
        _bump(counts, "excluded_unverified_yaw_floor")
    else:
        try:
            record = _floor_record(sample)
        except ValueError:  # codec-illegal token (e.g. '|' in a category)
            # Count and fall through: a bad floor category must not also
            # discard the sample's independently valid surface groups.
            _bump(counts, "skipped_floor_codec_error")
        else:
            floor_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            _bump(counts, "floor_records")
    _export_surface_groups(sample, surface_out, counts)


def export_sft(
    in_path: Path,
    out_dir: Path,
    *,
    exclude_nc: bool = True,
    contamination: frozenset[str] = frozenset(),
    exclude_unverified_yaw: bool = True,
) -> dict:
    """Stream ``in_path`` into SFT JSONL files; write and return the report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"input_samples": 0}
    counts.setdefault("floor_records", 0)
    counts.setdefault("surface_records", 0)
    floor_path = out_dir / FLOOR_FILE
    surface_path = out_dir / SURFACE_FILE
    with floor_path.open("w", encoding="utf-8") as floor_out:
        with surface_path.open("w", encoding="utf-8") as surface_out:
            for sample in read_jsonl(in_path):
                _export_sample(
                    sample,
                    floor_out,
                    surface_out,
                    counts,
                    exclude_nc,
                    contamination,
                    exclude_unverified_yaw,
                )
    report = {
        "in_path": str(in_path),
        "exclude_nc": exclude_nc,
        "exclude_unverified_yaw": exclude_unverified_yaw,
        "contamination_list_size": len(contamination),
        "counts": dict(sorted(counts.items())),
    }
    report_path = out_dir / REPORT_FILE
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export FastFill JSONL to floor/surface SFT records"
    )
    parser.add_argument(
        "--in", dest="in_path", required=True, help="deduped FastFill JSONL"
    )
    parser.add_argument("--out-dir", required=True, help="SFT output directory")
    parser.add_argument(
        "--exclude-nc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude CC BY-NC licensed samples (铁律 2; default on)",
    )
    parser.add_argument(
        "--contamination-list",
        default=None,
        help="file with one source_room_id per line to exclude (铁律 1)",
    )
    parser.add_argument(
        "--exclude-unverified-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "drop FLOOR records whose provenance notes flag an unverified "
            "facing convention (surface records kept; default on)"
        ),
    )
    args = parser.parse_args()

    contamination_path = (
        Path(args.contamination_list) if args.contamination_list else None
    )
    report = export_sft(
        Path(args.in_path),
        Path(args.out_dir),
        exclude_nc=args.exclude_nc,
        contamination=load_contamination_list(contamination_path),
        exclude_unverified_yaw=args.exclude_unverified_yaw,
    )
    print(f"report -> {Path(args.out_dir) / REPORT_FILE}")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
