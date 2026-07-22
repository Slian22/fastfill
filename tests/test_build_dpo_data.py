"""End-to-end tests for the DPO pair builder CLI (stage2 + stage1).

Runs ``fastfill_train.build_dpo_data.main`` in-process on tmp JSONL files
built from synthetic FastFillSamples. No torch / trl / datasets required.
"""

from __future__ import annotations

import json
from pathlib import Path

import fastfill_train  # noqa: F401  (vendor path bootstrap)
import fastfill_train.build_dpo_data as dpo_builder
from pydantic import ValidationError
from fastfill_data.export_sft import FLOOR_INSTRUCTION, SURFACE_INSTRUCTION
from test_injectors import make_clean_sample
from fastfill_train.injectors import ALL_INJECTORS

from fastfill_train.build_dpo_data import _floor_record, main


def _write_jsonl(path: Path, rows: list) -> Path:
    lines = [
        row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_samples(path: Path, n: int = 3) -> list:
    samples = [make_clean_sample(f"s{i}", f"house{i}") for i in range(n)]
    _write_jsonl(path, [s.model_dump_json() for s in samples])
    return samples


# -------------------------------------------------------------------- stage 2


def test_stage2_end_to_end(tmp_path: Path) -> None:
    # Arrange
    in_path = tmp_path / "samples.jsonl"
    out_path = tmp_path / "dpo" / "stage2_pairs.jsonl"
    _write_samples(in_path, n=3)

    # Act
    main(
        [
            "stage2",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--template",
            "direct",
            "--per-sample",
            "2",
            "--seed",
            "7",
         "--allow-no-snapshot"]
    )

    # Assert: 2 verified pairs per sample.
    rows = _read_jsonl(out_path)
    assert len(rows) == 6
    for row in rows:
        assert set(row) >= {
            "prompt",
            "chosen",
            "rejected",
            "split_key",
            "uid",
            "injector",
            "expected_codes",
        }
        assert [m["role"] for m in row["prompt"]] == ["user"]
        assert [m["role"] for m in row["chosen"]] == ["assistant"]
        assert [m["role"] for m in row["rejected"]] == ["assistant"]
        assert row["chosen"][0]["content"] != row["rejected"][0]["content"]
        assert row["expected_codes"], "every pair must carry expected codes"
        instruction = (
            SURFACE_INSTRUCTION if "#" in row["uid"] else FLOOR_INSTRUCTION
        )
        assert row["prompt"][0]["content"].startswith(instruction)
    assert {row["split_key"] for row in rows} == {"house0", "house1", "house2"}

    stats = json.loads(
        (tmp_path / "dpo" / "stage2_pairs.stats.json").read_text()
    )
    assert stats["counts"] == {"input_samples": 3, "pairs": 6}
    assert sum(stats["per_injector"].values()) == 6


def test_stage2_counts_unrenderable_floating_pairs(tmp_path: Path) -> None:
    """z_local is not codec-encoded, so floating pairs are skipped + counted."""
    in_path = tmp_path / "samples.jsonl"
    out_path = tmp_path / "stage2_pairs.jsonl"
    _write_samples(in_path, n=1)

    main(
        [
            "stage2",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--per-sample",
            "9",
         "--allow-no-snapshot"]
    )

    rows = _read_jsonl(out_path)
    assert len(rows) == len(ALL_INJECTORS) - 1  # floating surface object unrenderable in codec
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert {row["injector"] for row in rows} == set(stats["per_injector"])
    assert (
        stats["skipped"]["identical_completion:floating_surface_object"] == 1
    )
    assert stats["skipped"]["sample_underfilled"] == 1


def test_stage2_plan_template_renders_plan_line_on_floor_pairs(
    tmp_path: Path,
) -> None:
    in_path = tmp_path / "samples.jsonl"
    out_path = tmp_path / "stage2_pairs.jsonl"
    _write_samples(in_path, n=1)

    main(
        [
            "stage2",
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--template",
            "plan",
            "--per-sample",
            "9",
         "--allow-no-snapshot"]
    )

    rows = _read_jsonl(out_path)
    floor_rows = [row for row in rows if "#" not in row["uid"]]
    surface_rows = [row for row in rows if "#" in row["uid"]]
    assert floor_rows and surface_rows
    for row in floor_rows:
        assert row["chosen"][0]["content"].startswith("PLAN:")
        assert row["rejected"][0]["content"].startswith("PLAN:")
        assert row["chosen"][0]["content"] != row["rejected"][0]["content"]
    for row in surface_rows:  # surface records always use the direct format
        assert not row["chosen"][0]["content"].startswith("PLAN:")


# -------------------------------------------------------------------- stage 1


def _make_oob_completion(output: str) -> str:
    """Ground-truth floor codec text with the first object pushed to x=9 m."""
    lines = output.splitlines()
    parts = lines[0].split("|")
    parts[2] = "900,200"
    return "\n".join(["|".join(parts)] + lines[1:])


def _stage1_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    samples_path = tmp_path / "samples.jsonl"
    samples = _write_samples(samples_path, n=3)
    records = [_floor_record(s) for s in samples]
    contexts_path = _write_jsonl(tmp_path / "floor_sft.jsonl", records)
    generations = [
        {"uid": "s0", "completion": records[0]["output"]},  # valid -> skipped
        {"uid": "s1", "completion": "THIS IS NOT A LAYOUT"},  # parse failure
        {"uid": "s2", "completion": _make_oob_completion(records[2]["output"])},
        {"uid": "zzz", "completion": "whatever"},  # unknown uid
    ]
    generations_path = _write_jsonl(tmp_path / "gens.jsonl", generations)
    return contexts_path, generations_path, samples_path


def test_stage1_validated_mode(tmp_path: Path, monkeypatch) -> None:
    # Arrange
    contexts_path, generations_path, samples_path = _stage1_fixtures(tmp_path)
    out_path = tmp_path / "stage1_pairs.jsonl"

    # Act
    main(
        [
            "stage1",
            "--contexts",
            str(contexts_path),
            "--generations",
            str(generations_path),
            "--out",
            str(out_path),
            "--samples",
            str(samples_path),
         "--allow-no-snapshot"]
    )

    # Assert
    rows = {row["uid"]: row for row in _read_jsonl(out_path)}
    assert set(rows) == {"s1", "s2"}
    assert rows["s1"]["reason"] == "parse_failure"
    assert rows["s1"]["rejected"][0]["content"] == "THIS IS NOT A LAYOUT"
    assert rows["s2"]["reason"] == "validation_failure"
    assert "L1_FLOOR_OUT_OF_BOUNDS" in rows["s2"]["violation_codes"]
    for row in rows.values():
        assert [m["role"] for m in row["prompt"]] == ["user"]
        assert row["chosen"][0]["content"] != row["rejected"][0]["content"]

    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["counts"] == {"generations": 4, "pairs": 2}
    assert stats["skipped"] == {"generation_passes": 1, "unknown_uid": 1}
    assert stats["judge_modes"] == {"validated": 3, "parse_only": 0}


def test_judge_generation_schema_validation_error_is_parse_failure(
    monkeypatch,
) -> None:
    """ValidationError from a pydantic schema check must be treated as a
    parse failure, not propagated as an unhandled exception.  This keeps
    _judge_generation aligned with the same decode path in eval_layout."""

    def _raise_validation_error(*_args: object, **_kwargs: object) -> None:
        raise ValidationError.from_exception_data("FloorLayout", [])

    monkeypatch.setattr(dpo_builder, "decode_floor_layout", _raise_validation_error)
    verdict, codes = dpo_builder._judge_generation({}, "malformed", None)
    assert (verdict, codes) == ("reject_parse_failure", [])


def test_stage1_parse_only_mode_without_samples(tmp_path: Path) -> None:
    contexts_path, generations_path, _ = _stage1_fixtures(tmp_path)
    out_path = tmp_path / "stage1_pairs.jsonl"

    main(
        [
            "stage1",
            "--contexts",
            str(contexts_path),
            "--generations",
            str(generations_path),
            "--out",
            str(out_path),
         "--allow-no-snapshot"]
    )

    # Only the unparseable completion can become a negative in this mode; the
    # decodable-but-invalid one is (honestly) counted as a parse-only pass.
    rows = _read_jsonl(out_path)
    assert [row["uid"] for row in rows] == ["s1"]
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["counts"]["pairs"] == 1
    assert stats["skipped"]["parse_only_passes"] == 2
    assert stats["judge_modes"] == {"validated": 0, "parse_only": 3}


def test_stage1_near_miss_gate_skips_multi_violation_rejects(tmp_path):
    """Architect-Ant precedent: rejected completions with many distinct
    violation codes are not near-misses and are skipped by default."""
    import json

    from fastfill_train.build_dpo_data import run_stage1

    sample = make_clean_sample("s_gate", "house_gate")
    record = _floor_record(sample)
    contexts = tmp_path / "ctx.jsonl"
    contexts.write_text(json.dumps(record) + "\n")
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(sample.model_dump_json() + "\n")

    # Completion with two objects far outside AND colliding AND a third
    # missing expected furniture -> many distinct codes -> gated out.
    bad = "\n".join(
        [
            "wardrobe|120,60,220|900,900|0|w",
            "wardrobe|120,60,220|900,900|0|w",
        ]
    )
    gens = tmp_path / "gens.jsonl"
    gens.write_text(json.dumps({"uid": record["uid"], "completion": bad}) + "\n")

    out = tmp_path / "pairs.jsonl"
    stats = run_stage1(contexts, gens, out, samples_path, "direct", 2)
    all_stats = json.dumps(stats)
    if stats["counts"]["pairs"] == 0:
        assert "too_many_violations_not_near_miss" in all_stats
    else:
        # if this completion only fires <=2 distinct codes, the gate must
        # still exist and default to 2
        assert stats["max_reject_codes"] == 2

    # gate disabled -> pair emitted (completion definitely fails validation)
    stats_off = run_stage1(
        contexts, gens, tmp_path / "pairs_off.jsonl", samples_path, "direct", 0
    )
    assert stats_off["counts"]["pairs"] == 1


# ------------------------------------------------------- holdout + gates


def test_stage2_snapshot_filter_excludes_holdout(tmp_path: Path) -> None:
    """DPO chosen labels must never come from Stage-0 heldout/test keys."""
    from fastfill_train.data import split_bucket

    seed, val, test = 42, 0.1, 0.05
    cutoff = int((val + test) * 10_000)
    holdout_key = next(
        f"house_h{i}" for i in range(1000) if split_bucket(f"house_h{i}", seed) < cutoff
    )
    train_key = next(
        f"house_t{i}" for i in range(1000) if split_bucket(f"house_t{i}", seed) >= cutoff
    )
    samples = [
        make_clean_sample("s_hold", holdout_key),
        make_clean_sample("s_train", train_key),
    ]
    in_path = tmp_path / "samples.jsonl"
    _write_jsonl(in_path, [s.model_dump_json() for s in samples])
    snapshot = _freeze_snapshot(tmp_path, samples, val=val, test=test)
    out_path = tmp_path / "pairs.jsonl"

    main(
        [
            "stage2",
            "--in", str(in_path),
            "--out", str(out_path),
            "--snapshot", str(snapshot),
            "--per-sample", "1",
        ]
    )

    rows = _read_jsonl(out_path)
    assert rows, "train-side sample must still produce pairs"
    assert {row["split_key"] for row in rows} == {train_key}
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["skipped"]["excluded_holdout"] == 1
    assert "warning" not in stats


def test_stage2_without_snapshot_warns(tmp_path: Path) -> None:
    in_path = tmp_path / "samples.jsonl"
    _write_samples(in_path, n=1)
    out_path = tmp_path / "pairs.jsonl"
    main(["stage2", "--in", str(in_path), "--out", str(out_path), "--allow-no-snapshot"])
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert "heldout/test" in stats["warning"]


def test_stage2_hard_gates_unsanitized_samples(tmp_path: Path) -> None:
    sample = make_clean_sample("s_raw", "house_raw")
    sample = sample.model_copy(
        update={"provenance": sample.provenance.model_copy(update={"notes": ""})}
    )
    in_path = tmp_path / "samples.jsonl"
    _write_jsonl(in_path, [sample.model_dump_json()])
    out_path = tmp_path / "pairs.jsonl"

    main(["stage2", "--in", str(in_path), "--out", str(out_path), "--allow-no-snapshot"])
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["skipped"]["excluded_unsanitized"] == 1
    assert stats["counts"]["pairs"] == 0

    main(
        [
            "stage2",
            "--in", str(in_path),
            "--out", str(tmp_path / "pairs_off.jsonl"),
            "--no-require-sanitized",
         "--allow-no-snapshot"]
    )
    stats_off = json.loads(
        (tmp_path / "pairs_off.stats.json").read_text()
    )
    assert stats_off["counts"]["pairs"] > 0


def test_stage1_snapshot_filter_excludes_holdout_contexts(tmp_path: Path) -> None:
    from fastfill_train.data import split_bucket

    seed, val, test = 42, 0.1, 0.05
    cutoff = int((val + test) * 10_000)
    holdout_key = next(
        f"h{i}" for i in range(1000) if split_bucket(f"h{i}", seed) < cutoff
    )
    train_key = next(
        f"t{i}" for i in range(1000) if split_bucket(f"t{i}", seed) >= cutoff
    )
    hold_sample = make_clean_sample("s_hold", holdout_key)
    train_sample = make_clean_sample("s_train", train_key)
    contexts = tmp_path / "contexts.jsonl"
    _write_jsonl(
        contexts,
        [_floor_record(hold_sample), _floor_record(train_sample)],
    )
    generations = tmp_path / "gens.jsonl"
    _write_jsonl(
        generations,
        [
            {"uid": "s_hold", "completion": "not a layout"},
            {"uid": "s_train", "completion": "not a layout"},
        ],
    )
    snapshot = _freeze_snapshot(
        tmp_path, [hold_sample, train_sample], val=val, test=test
    )
    out_path = tmp_path / "pairs.jsonl"

    main(
        [
            "stage1",
            "--contexts", str(contexts),
            "--generations", str(generations),
            "--snapshot", str(snapshot),
            "--out", str(out_path),
        ]
    )

    rows = _read_jsonl(out_path)
    assert [row["uid"] for row in rows] == ["s_train"]
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["skipped"]["excluded_holdout"] == 1


# -------------------------------------------------------- bbox-unverified gate


def test_stage2_bbox_unverified_blocks_floor_pairs_allows_surface(
    tmp_path: Path,
) -> None:
    """bbox_axis=unverified_3dfront must suppress Floor pairs but not Surface."""
    from fastfill_data.export_sft import BBOX_UNVERIFIED_NOTE

    sample = make_clean_sample("s_bbox", "house_bbox")
    sample = sample.model_copy(
        update={
            "provenance": sample.provenance.model_copy(
                update={"notes": f"sanitized=v1;{BBOX_UNVERIFIED_NOTE}"}
            )
        }
    )
    in_path = tmp_path / "samples.jsonl"
    _write_jsonl(in_path, [sample.model_dump_json()])
    out_path = tmp_path / "pairs.jsonl"

    main(["stage2", "--in", str(in_path), "--out", str(out_path), "--per-sample", "9", "--allow-no-snapshot"])

    rows = _read_jsonl(out_path)
    floor_rows = [r for r in rows if "#" not in r["uid"]]
    surface_rows = [r for r in rows if "#" in r["uid"]]

    assert floor_rows == [], "bbox-unverified must produce no Floor pairs"
    assert surface_rows, "bbox-unverified must still produce Surface pairs"

    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    skipped_keys = " ".join(stats["skipped"])
    assert "excluded_bbox_unverified_floor" in skipped_keys
    assert "excluded_unverified_yaw_floor" not in skipped_keys


def test_stage2_bbox_unverified_stats_key_not_yaw(tmp_path: Path) -> None:
    """Stats key for bbox exclusions must be distinct from the yaw exclusion key."""
    from fastfill_data.export_sft import BBOX_UNVERIFIED_NOTE, UNVERIFIED_YAW_NOTE
    from fastfill_data.sanitize import FLOOR_UNREPAIRED_NOTE

    def _make(note_extra: str, sample_id: str) -> object:
        s = make_clean_sample(sample_id, f"house_{sample_id}")
        return s.model_copy(
            update={
                "provenance": s.provenance.model_copy(
                    update={"notes": f"sanitized=v1;{note_extra}"}
                )
            }
        )

    cases = [
        ("bbox", _make(BBOX_UNVERIFIED_NOTE, "s_bbox")),
        ("yaw", _make(UNVERIFIED_YAW_NOTE, "s_yaw")),
        ("unrepaired", _make(FLOOR_UNREPAIRED_NOTE, "s_unrepaired")),
    ]
    for label, sample in cases:
        out = tmp_path / f"pairs_{label}.jsonl"
        _write_jsonl(tmp_path / f"in_{label}.jsonl", [sample.model_dump_json()])
        main(
            [
                "stage2",
                "--in", str(tmp_path / f"in_{label}.jsonl"),
                "--out", str(out),
                "--per-sample", "9",
             "--allow-no-snapshot"]
        )
        stats = json.loads(out.with_suffix(".stats.json").read_text())
        skipped_keys = " ".join(stats["skipped"])
        if label == "bbox":
            assert "excluded_bbox_unverified_floor" in skipped_keys, label
            assert "excluded_unverified_yaw_floor" not in skipped_keys, label
        elif label == "yaw":
            assert "excluded_unverified_yaw_floor" in skipped_keys, label
            assert "excluded_bbox_unverified_floor" not in skipped_keys, label
        elif label == "unrepaired":
            assert "excluded_floor_unrepaired" in skipped_keys, label


def test_stage2_clean_sample_unaffected_by_bbox_gate(tmp_path: Path) -> None:
    """A sample with no exclusion notes must still produce floor pairs normally."""
    in_path = tmp_path / "samples.jsonl"
    _write_samples(in_path, n=1)
    out_path = tmp_path / "pairs.jsonl"

    main(["stage2", "--in", str(in_path), "--out", str(out_path), "--per-sample", "9", "--allow-no-snapshot"])

    rows = _read_jsonl(out_path)
    floor_rows = [r for r in rows if "#" not in r["uid"]]
    assert floor_rows, "clean sample must still produce Floor pairs"


def test_self_built_records_carry_explicit_layer() -> None:
    # build_dpo_data builds its own SFT-shape records; they MUST carry an
    # explicit "layer" so a floor uid containing '#' (e.g. mansionworld) is
    # not misrouted by the '#'-in-uid heuristic under plan/plan_nl.
    from fastfill_train.build_dpo_data import _floor_record, _surface_record

    sample = make_clean_sample("s0", "house0")
    floor = _floor_record(sample)
    assert floor["layer"] == "floor"
    assert "#" not in floor["uid"]

    group = sample.layout.surface_groups[0]
    surface = _surface_record(sample, group)
    assert surface is not None
    assert surface["layer"] == "surface"
    assert "#" in surface["uid"]


def test_self_built_records_canonicalize_room_type_and_keep_raw() -> None:
    # room_type feeds the prompt: it must be the canonical form, byte-equal
    # to what the codec renders into `input`; the converter's original value
    # is preserved in room_type_raw (and must NOT appear in the prompt).
    from fastfill_train.build_dpo_data import _floor_record, _surface_record

    sample = make_clean_sample("s0", "house0")
    sample = sample.model_copy(
        update={
            "room_context": sample.room_context.model_copy(
                update={"room_type": "Bed Room"}
            )
        }
    )
    floor = _floor_record(sample)
    assert floor["room_type"] == "bedroom"
    assert floor["room_type_raw"] == "Bed Room"
    assert "room bedroom id=" in floor["input"]
    assert "Bed Room" not in floor["input"]

    group = sample.layout.surface_groups[0]
    surface = _surface_record(sample, group)
    assert surface is not None
    assert surface["room_type"] == "bedroom"
    assert surface["room_type_raw"] == "Bed Room"
    assert "\nroom bedroom\n" in surface["input"]


def test_stage2_license_allowlist_gates_pending(tmp_path: Path) -> None:
    """Stage-2 chosen labels obey the same license allowlist as export_sft:
    LICENSE_PENDING never trains under permissive OR plain research."""
    from scenesmith.growing_world.fastfill.schema import LicenseTag

    sample = make_clean_sample("s0", "house0")
    sample = sample.model_copy(
        update={
            "provenance": sample.provenance.model_copy(
                update={"license_tag": LicenseTag.LICENSE_PENDING}
            )
        }
    )
    in_path = tmp_path / "samples.jsonl"
    _write_jsonl(in_path, [sample.model_dump_json()])
    out_path = tmp_path / "pairs.jsonl"

    main(
        [
            "stage2",
            "--in", str(in_path),
            "--out", str(out_path),
            "--allow-no-snapshot",
        ]
    )
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["counts"]["pairs"] == 0
    assert stats["skipped"]["excluded_license_license_pending"] == 1

    main(
        [
            "stage2",
            "--in", str(in_path),
            "--out", str(out_path),
            "--license-mode", "research",
            "--allow-no-snapshot",
        ]
    )
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["counts"]["pairs"] == 0  # research alone still excludes

    main(
        [
            "stage2",
            "--in", str(in_path),
            "--out", str(out_path),
            "--license-mode", "research",
            "--allow-unresolved-licenses",
            "--allow-no-snapshot",
        ]
    )
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["counts"]["pairs"] > 0


def test_snapshot_empty_string_is_rejected(tmp_path: Path) -> None:
    import pytest

    in_path = tmp_path / "samples.jsonl"
    _write_samples(in_path, n=1)
    with pytest.raises(SystemExit, match="snapshot"):
        main(
            [
                "stage2",
                "--in", str(in_path),
                "--out", str(tmp_path / "pairs.jsonl"),
                "--snapshot", "",
            ]
        )


def test_snapshot_sidecar_shape_is_validated(tmp_path: Path) -> None:
    import pytest

    in_path = tmp_path / "samples.jsonl"
    _write_samples(in_path, n=1)
    bogus = tmp_path / "not_a_sidecar.json"
    bogus.write_text(json.dumps({"seed": 42}))  # missing val_fraction/rule
    with pytest.raises(ValueError, match="not a current make_snapshot"):
        main(
            [
                "stage2",
                "--in", str(in_path),
                "--out", str(tmp_path / "pairs.jsonl"),
                "--snapshot", str(bogus),
            ]
        )


def _freeze_snapshot(
    tmp_path: Path, samples, val: float = 0.1, test: float = 0.0
) -> Path:
    """Freeze a real snapshot over the samples' floor records."""
    import sys as _sys

    from fastfill_train.make_snapshot import main as snap_main

    records = [_floor_record(s) for s in samples]
    for sample in samples:  # mirror export: surface records are frozen too
        for group in sample.layout.surface_groups:
            record = dpo_builder._surface_record(sample, group)
            if record is not None:
                records.append(record)
    sft_path = _write_jsonl(tmp_path / "floor_sft.jsonl", records)
    out_dir = tmp_path / "snap"
    argv_backup = _sys.argv
    _sys.argv = [
        "make_snapshot",
        "--in", str(sft_path),
        "--out-dir", str(out_dir),
        "--val-fraction", str(val),
        "--test-fraction", str(test),
    ]
    try:
        snap_main()
    finally:
        _sys.argv = argv_backup
    return out_dir / "SNAPSHOT.json"


def test_stage2_rejects_foreign_corpus_snapshot(tmp_path: Path) -> None:
    """A snapshot frozen from corpus A must not filter corpus B."""
    import pytest

    corpus_a = [make_clean_sample(f"a{i}", f"houseA{i}") for i in range(3)]
    snapshot = _freeze_snapshot(tmp_path, corpus_a)

    corpus_b_path = tmp_path / "corpus_b.jsonl"
    corpus_b = [make_clean_sample(f"b{i}", f"houseB{i}") for i in range(3)]
    _write_jsonl(corpus_b_path, [s.model_dump_json() for s in corpus_b])

    with pytest.raises(SystemExit, match="corpus mismatch"):
        main(
            [
                "stage2",
                "--in", str(corpus_b_path),
                "--out", str(tmp_path / "pairs.jsonl"),
                "--snapshot", str(snapshot),
            ]
        )


def test_stage2_accepts_matching_corpus_snapshot(tmp_path: Path) -> None:
    corpus = [make_clean_sample(f"s{i}", f"house{i}") for i in range(3)]
    snapshot = _freeze_snapshot(tmp_path, corpus)
    in_path = tmp_path / "samples.jsonl"
    _write_jsonl(in_path, [s.model_dump_json() for s in corpus])
    out_path = tmp_path / "pairs.jsonl"
    main(
        [
            "stage2",
            "--in", str(in_path),
            "--out", str(out_path),
            "--snapshot", str(snapshot),
        ]
    )
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["snapshot_id"]
    assert stats["out_sha256_16"]
    rows = _read_jsonl(out_path)
    assert rows and all(row["license"] == "permissive" for row in rows)


def test_stage1_license_gate_fails_closed_on_unrecorded(tmp_path: Path) -> None:
    """Context records without a license field are unresolved: no chosen
    labels under permissive OR plain research."""
    contexts_path, generations_path, samples_path = _stage1_fixtures(tmp_path)
    # Strip the license field from every context record.
    stripped = [
        json.dumps({k: v for k, v in row.items() if k != "license"})
        for row in _read_jsonl(contexts_path)
    ]
    _write_jsonl(contexts_path, stripped)
    out_path = tmp_path / "stage1_pairs.jsonl"
    main(
        [
            "stage1",
            "--contexts", str(contexts_path),
            "--generations", str(generations_path),
            "--out", str(out_path),
            "--samples", str(samples_path),
            "--allow-no-snapshot",
        ]
    )
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["counts"]["pairs"] == 0
    assert stats["skipped"]["excluded_license_unknown"] == 3


def test_stage2_route_must_match_snapshot(tmp_path: Path) -> None:
    """A research invocation over a permissive snapshot must hard-fail —
    otherwise NC chosen labels reach a permissive model's DPO stage."""
    import pytest

    corpus = [make_clean_sample(f"s{i}", f"house{i}") for i in range(2)]
    snapshot = _freeze_snapshot(tmp_path, corpus)  # license_mode=permissive
    in_path = tmp_path / "samples.jsonl"
    _write_jsonl(in_path, [s.model_dump_json() for s in corpus])
    with pytest.raises(SystemExit, match="snapshot's"):
        main(
            [
                "stage2",
                "--in", str(in_path),
                "--out", str(tmp_path / "pairs.jsonl"),
                "--snapshot", str(snapshot),
                "--license-mode", "research",
            ]
        )


def test_stage2_tampered_content_is_unauthorized(tmp_path: Path) -> None:
    """Same split_key + same geometry_hash but modified chosen content must
    not produce pairs — the record-identity manifest catches what the
    category-free dedup geometry_hash cannot."""
    corpus = [make_clean_sample("s0", "house0")]
    snapshot = _freeze_snapshot(tmp_path, corpus)

    sample = corpus[0]
    group = sample.layout.surface_groups[0]
    tampered_objects = tuple(
        obj.model_copy(update={"category": "SECRETCHANGEDLABEL"})
        for obj in group.objects
    )
    tampered_groups = tuple(
        g.model_copy(update={"objects": tampered_objects})
        if g.group_id == group.group_id
        else g
        for g in sample.layout.surface_groups
    )
    tampered = sample.model_copy(
        update={
            "layout": sample.layout.model_copy(
                update={"surface_groups": tampered_groups}
            )
        }
    )
    # The near-dup signature is blind to categories/surfaces — unchanged.
    assert tampered.provenance.geometry_hash == sample.provenance.geometry_hash

    in_path = tmp_path / "samples.jsonl"
    _write_jsonl(in_path, [tampered.model_dump_json()])
    out_path = tmp_path / "pairs.jsonl"
    main(
        [
            "stage2",
            "--in", str(in_path),
            "--out", str(out_path),
            "--snapshot", str(snapshot),
            "--per-sample", "9",
        ]
    )
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    unauthorized = sum(
        v for k, v in stats["skipped"].items() if k.startswith("unauthorized_record")
    )
    assert unauthorized > 0
    rows = _read_jsonl(out_path)
    assert all("SECRETCHANGEDLABEL" not in json.dumps(row) for row in rows)
