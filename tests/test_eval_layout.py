"""Offline tests for the eval harness (no network, no openai import)."""

from __future__ import annotations

import json
from pathlib import Path

import fastfill_train  # noqa: F401  (vendor path bootstrap)
import pytest
from scenesmith.growing_world.fastfill.codec import (
    encode_floor_layout,
    encode_room_context,
    encode_surface_groups,
)
from scenesmith.growing_world.fastfill.schema import (
    Anchor,
    FastFillSample,
    FloorLayout,
    FloorObjectSpec,
    ProvenanceMeta,
    RoomContentLayout,
    RoomContext,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
)

from fastfill_train.eval_layout import main
from fastfill_train.templates import LAYOUT_PREFIX, PLAN_PREFIX

# ------------------------------------------------------------ test fixtures


def _context() -> RoomContext:
    return RoomContext(
        room_id="r0",
        room_type="bedroom",
        floor_polygon=((-2.0, -1.6), (2.0, -1.6), (2.0, 1.6), (-2.0, 1.6)),
        expected_furniture=("bed",),
    )


def _bed(position: tuple[float, float]) -> FloorObjectSpec:
    return FloorObjectSpec(
        object_id="bed_0",
        category="bed",
        dimensions=(1.6, 2.0, 0.5),
        position_xy=position,
        yaw_deg=0.0,
        anchor=Anchor.WALL,
    )


def _floor_text(position: tuple[float, float] = (0.0, 0.0)) -> str:
    return encode_floor_layout(FloorLayout(room_id="r0", objects=(_bed(position),)))


def _surface_text() -> str:
    group = SurfaceObjectGroup(
        group_id="g0",
        surface_id="surf0",
        objects=(
            SurfaceObjectSpec(
                object_id="g0/plate_0",
                category="plate",
                dimensions=(0.25, 0.25, 0.03),
                position_local=(0.0, 0.0),
            ),
        ),
    )
    return encode_surface_groups([group])


def _record(uid: str, output: str) -> dict:
    return {
        "uid": uid,
        "split_key": "house0",
        "source_dataset": "test_ds",
        "instruction": "Place floor-standing furniture in the room below.",
        "input": encode_room_context(_context()),
        "output": output,
    }


def _sample(sample_id: str = "s0") -> FastFillSample:
    return FastFillSample(
        sample_id=sample_id,
        room_context=_context(),
        layout=RoomContentLayout(
            room_id="r0",
            floor_layout=FloorLayout(room_id="r0", objects=(_bed((0.0, 0.0)),)),
        ),
        provenance=ProvenanceMeta(source_dataset="test_ds", split_key="house0"),
    )


def _write_jsonl(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run(
    tmp_path: Path,
    records: list[dict],
    generations: list[dict],
    samples: list[FastFillSample] | None = None,
    extra: list[str] | None = None,
) -> dict:
    records_path = tmp_path / "records.jsonl"
    _write_jsonl(records_path, [json.dumps(r) for r in records])
    gens_path = tmp_path / "gens.jsonl"
    _write_jsonl(gens_path, [json.dumps(g) for g in generations])
    out_path = tmp_path / "report.json"
    argv = [
        "--records",
        str(records_path),
        "--generations",
        str(gens_path),
        "--out",
        str(out_path),
    ]
    if samples is not None:
        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(samples_path, [s.model_dump_json() for s in samples])
        argv += ["--samples", str(samples_path)]
    argv += extra or []
    assert main(argv) == 0
    assert out_path.exists()
    return json.loads(out_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- tests


def test_well_formed_completion_parses_and_validates(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        records=[_record("s0", _floor_text())],
        generations=[{"uid": "s0", "completion": _floor_text()}],
        samples=[_sample()],
    )
    assert report["n"] == 1
    assert report["n_floor"] == 1
    assert report["parse_rate"] == 1.0
    assert report["n_validated"] == 1
    assert report["pass_pre_repair"] == 1.0
    assert report["pass_post_repair"] == 1.0
    assert report["expected_furniture_coverage"] == 1.0
    assert report["latency_ms_mean"] is None  # offline mode has no latency
    assert report["per_source"]["test_ds"]["n"] == 1
    assert report["per_source"]["test_ds"]["parse_rate"] == 1.0


def test_garbage_completion_counts_as_parse_fail(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        records=[_record("s0", _floor_text())],
        generations=[{"uid": "s0", "completion": "this is not a layout at all"}],
        samples=[_sample()],
    )
    assert report["parse_rate"] == 0.0
    # parse failures inside the validated set count as failed passes
    assert report["n_validated"] == 1
    assert report["pass_pre_repair"] == 0.0
    assert report["pass_post_repair"] == 0.0


def test_out_of_bounds_layout_repaired_post_repair(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        records=[_record("s0", _floor_text())],
        generations=[{"uid": "s0", "completion": _floor_text((0.0, 1.5))}],
        samples=[_sample()],
    )
    assert report["parse_rate"] == 1.0
    assert report["pass_pre_repair"] == 0.0
    assert report["pass_post_repair"] == 1.0
    assert "L1_FLOOR_OUT_OF_BOUNDS" in report["violations_top"]


def test_plan_format_completion_is_scored(tmp_path: Path) -> None:
    completion = (
        f"{PLAN_PREFIX} zones=sleeping; anchors=bed:wall; n=1\n"
        f"{LAYOUT_PREFIX}\n{_floor_text()}"
    )
    report = _run(
        tmp_path,
        records=[_record("s0", _floor_text())],
        generations=[{"uid": "s0", "completion": completion}],
        samples=[_sample()],
    )
    assert report["parse_rate"] == 1.0
    assert report["pass_post_repair"] == 1.0


def test_surface_records_are_parse_only(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        records=[
            _record("s0#g0", _surface_text()),
            _record("s0#g1", _surface_text()),
        ],
        generations=[
            {"uid": "s0#g0", "completion": _surface_text()},
            {"uid": "s0#g1", "completion": "garbage"},
        ],
    )
    assert report["n_surface"] == 2
    assert report["n_floor"] == 0
    assert report["surface_parse_rate"] == 0.5
    assert report["parse_rate"] is None
    assert report["n_validated"] == 0
    assert report["pass_post_repair"] is None


def test_missing_generation_counted_not_scored(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        records=[_record("s0", _floor_text()), _record("s1", _floor_text())],
        generations=[{"uid": "s0", "completion": _floor_text()}],
    )
    assert report["n"] == 1
    assert report["n_missing_generation"] == 1


def test_teacher_ratio_computed(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher.json"
    teacher_path.write_text(json.dumps({"pass_post_repair": 0.8}))
    report = _run(
        tmp_path,
        records=[_record("s0", _floor_text())],
        generations=[{"uid": "s0", "completion": _floor_text()}],
        samples=[_sample()],
        extra=["--teacher-report", str(teacher_path)],
    )
    assert report["teacher"]["pass_post_repair"] == 0.8
    assert report["teacher"]["ratio_pass_post_repair"] == pytest.approx(1.25)


def test_dump_generations_row_shape(tmp_path: Path) -> None:
    dump_path = tmp_path / "dump.jsonl"
    _run(
        tmp_path,
        records=[_record("s0", _floor_text()), _record("s0#g0", _surface_text())],
        generations=[
            {"uid": "s0", "completion": _floor_text()},
            {"uid": "s0#g0", "completion": _surface_text()},
        ],
        extra=["--dump-generations", str(dump_path)],
    )
    rows = [
        json.loads(line)
        for line in dump_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all(set(row) == {"uid", "completion"} for row in rows)
    assert rows[0] == {"uid": "s0", "completion": _floor_text()}


def test_limit_caps_records(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        records=[_record("s0", _floor_text()), _record("s1", _floor_text())],
        generations=[
            {"uid": "s0", "completion": _floor_text()},
            {"uid": "s1", "completion": _floor_text()},
        ],
        extra=["--limit", "1"],
    )
    assert report["n"] == 1


def test_offline_or_endpoint_required(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    _write_jsonl(records_path, [json.dumps(_record("s0", _floor_text()))])
    with pytest.raises(SystemExit):
        main(["--records", str(records_path), "--out", str(tmp_path / "r.json")])
