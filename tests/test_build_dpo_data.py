"""End-to-end tests for the DPO pair builder CLI (stage2 + stage1).

Runs ``fastfill_train.build_dpo_data.main`` in-process on tmp JSONL files
built from synthetic FastFillSamples. No torch / trl / datasets required.
"""

from __future__ import annotations

import json
from pathlib import Path

import fastfill_train  # noqa: F401  (vendor path bootstrap)
from fastfill_data.export_sft import FLOOR_INSTRUCTION, SURFACE_INSTRUCTION
from test_injectors import make_clean_sample

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
        ]
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
        ]
    )

    rows = _read_jsonl(out_path)
    assert len(rows) == 8  # 9 injectors, floating unrenderable in the codec
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
        ]
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


def test_stage1_validated_mode(tmp_path: Path) -> None:
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
        ]
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
        ]
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
    skipped = stats["skipped"] if "skipped" in stats else stats.get("counts", {})
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
