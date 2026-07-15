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
- ``--license-mode`` (default ``permissive``) — ``permissive`` exports only
  ``LicenseTag.PERMISSIVE`` samples (CC BY-NC / LICENSE_PENDING / UNKNOWN
  are excluded and counted per tag, 铁律 2); ``research`` also exports
  ``CC_BY_NC`` for a research-only checkpoint. Unresolved tags
  (LICENSE_PENDING / UNKNOWN) never export unless
  ``--allow-unresolved-licenses`` is passed explicitly. Every exported
  record carries its ``license`` tag for downstream lineage audits;
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
from fastfill_data.sanitize import (  # noqa: E402
    FLOOR_UNREPAIRED_NOTE,
    has_note,
    has_sanitized_note,
)
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
LICENSE_MODES = ("permissive", "research")


def load_contamination_list(path: Path | None) -> frozenset[str]:
    """One ``source_room_id`` per line; blank lines and #-comments ignored."""
    if path is None:
        return frozenset()
    if not path.exists():
        raise FileNotFoundError(
            f"contamination list not found: {path} — create it (one "
            "source_room_id per line; may be empty) or drop the flag"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    )


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
        "license": sample.provenance.license_tag.value,
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
        "license": sample.provenance.license_tag.value,
        "room_type": sample.room_context.room_type,
        "layer": "surface",
        "instruction": SURFACE_INSTRUCTION,
        "input": encode_support_context(context),
        "output": encode_surface_groups([group]),
    }


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _exclusion_reason(
    sample: FastFillSample,
    license_mode: str,
    allow_unresolved: bool,
    contamination: frozenset[str],
) -> str | None:
    """Sample-level filter (checked in 铁律 order), ``None`` = exportable."""
    room_id = sample.provenance.source_room_id
    house_id = sample.provenance.source_house_id
    if (room_id and room_id in contamination) or (
        house_id and house_id in contamination
    ):  # scenesmith scenes key on house_id (scene_XXX); room_id is a room name
        return "excluded_contamination"
    tag = sample.provenance.license_tag
    if not _license_allowed(tag, license_mode, allow_unresolved):
        return f"excluded_license_{tag.value}"
    if not sample.layout.floor_layout.objects:
        return "skipped_empty_floor"
    return None


def _license_allowed(
    tag: LicenseTag, license_mode: str, allow_unresolved: bool
) -> bool:
    """permissive: PERMISSIVE only. research: + CC_BY_NC. Pending/unknown
    licenses are unresolved (schema: they wait for confirmation) and only
    flow with the explicit --allow-unresolved-licenses override."""
    if tag is LicenseTag.PERMISSIVE:
        return True
    if license_mode != "research":
        return False
    if tag is LicenseTag.CC_BY_NC:
        return True
    return allow_unresolved


def _export_surface_groups(
    sample: FastFillSample, surface_out: TextIO, counts: dict[str, int]
) -> int:
    written = 0
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
        written += 1
    return written


UNVERIFIED_YAW_NOTE = "yaw_facing=unverified_per_asset"
# Mirrors convert_3d_synthplace.BBOX_UNVERIFIED_3DFRONT_NOTE — 3D-FRONT bbox
# axis order is a per-record HWD/DWH mix not resolvable without mesh vertices,
# so these floor labels are isolated (kept upstream for dedup + contamination
# closure, never exported). Defined here to avoid importing the converter.
BBOX_UNVERIFIED_NOTE = "bbox_axis=unverified_3dfront"


def _export_sample(
    sample: FastFillSample,
    floor_out: TextIO,
    surface_out: TextIO,
    counts: dict[str, int],
    license_mode: str,
    allow_unresolved: bool,
    contamination: frozenset[str],
    exclude_unverified_yaw: bool,
    require_sanitized: bool,
) -> None:
    _bump(counts, "input_samples")
    if not has_sanitized_note(sample.provenance.notes):
        # Hard gate (exact-token check): unsanitized labels are dirty by
        # definition — they never went through validate/repair/round-trip.
        _bump(
            counts,
            "excluded_unsanitized" if require_sanitized else "unsanitized_input_samples",
        )
        if require_sanitized:
            return
    reason = _exclusion_reason(sample, license_mode, allow_unresolved, contamination)
    if reason is not None:
        _bump(counts, reason)
        return
    # Floor yaw labels from sources without a verified facing convention
    # (e.g. MansionWorld: per-asset canonical fronts) would poison facing
    # supervision — skip the FLOOR record only. Surface records stay: their
    # local frames are parent-relative, so intra-surface poses are consistent
    # regardless of the parent's absolute facing. Same one-sided skip for
    # floors the sanitizer could not repair.
    skip_floor = ""
    if BBOX_UNVERIFIED_NOTE in sample.provenance.notes:
        # 3D-FRONT: bbox axis order unresolved -> isolate the whole floor
        # record (this source is floor-only). Kept upstream for dedup +
        # contamination closure; never a training label until mesh-resolved.
        skip_floor = "excluded_bbox_unverified_3dfront"
    elif exclude_unverified_yaw and UNVERIFIED_YAW_NOTE in sample.provenance.notes:
        skip_floor = "excluded_unverified_yaw_floor"
    elif has_note(sample.provenance.notes, FLOOR_UNREPAIRED_NOTE):
        skip_floor = "excluded_unrepaired_floor"
    wrote_floor = False
    if skip_floor:
        _bump(counts, skip_floor)
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
            wrote_floor = True
    wrote_surfaces = _export_surface_groups(sample, surface_out, counts)
    if wrote_floor or wrote_surfaces:
        # Per-tag accounting of what actually LANDED in the training files
        # (bumped only after a real write, so the report cannot claim an
        # exported license sample while floor+surface records are both 0).
        _bump(counts, f"exported_license_{sample.provenance.license_tag.value}")


def export_sft(
    in_path: Path,
    out_dir: Path,
    *,
    license_mode: str = "permissive",
    allow_unresolved: bool = False,
    contamination: frozenset[str] = frozenset(),
    exclude_unverified_yaw: bool = True,
    require_sanitized: bool = True,
) -> dict:
    """Stream ``in_path`` into SFT JSONL files; write and return the report."""
    if license_mode not in LICENSE_MODES:
        raise ValueError(
            f"license_mode must be one of {LICENSE_MODES}, got {license_mode!r}"
        )
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
                    license_mode,
                    allow_unresolved,
                    contamination,
                    exclude_unverified_yaw,
                    require_sanitized,
                )
    report = {
        "in_path": str(in_path),
        "license_mode": license_mode,
        "allow_unresolved_licenses": allow_unresolved,
        "exclude_unverified_yaw": exclude_unverified_yaw,
        "require_sanitized": require_sanitized,
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
        "--license-mode",
        choices=LICENSE_MODES,
        default="permissive",
        help=(
            "permissive: export only PERMISSIVE-tagged samples (铁律 2, "
            "default); research: also export CC_BY_NC for a research-only "
            "checkpoint (per-tag counts in the report)"
        ),
    )
    parser.add_argument(
        "--allow-unresolved-licenses",
        action="store_true",
        help=(
            "DANGER: in research mode, also export LICENSE_PENDING/UNKNOWN "
            "samples (schema says unresolved licenses wait for confirmation)"
        ),
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
    parser.add_argument(
        "--require-sanitized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "hard gate: exclude samples without the sanitize.py "
            "'sanitized=v1' note (default on; --no-require-sanitized only "
            "counts them)"
        ),
    )
    args = parser.parse_args()

    contamination_path = (
        Path(args.contamination_list) if args.contamination_list else None
    )
    report = export_sft(
        Path(args.in_path),
        Path(args.out_dir),
        license_mode=args.license_mode,
        allow_unresolved=args.allow_unresolved_licenses,
        contamination=load_contamination_list(contamination_path),
        exclude_unverified_yaw=args.exclude_unverified_yaw,
        require_sanitized=args.require_sanitized,
    )
    print(f"report -> {Path(args.out_dir) / REPORT_FILE}")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
