"""Research conclusions require verified full cohorts and paired cluster inference."""
import json

import pytest

from fastfill.v1_1 import evaluation_protocol as protocol
from fastfill.v1_1 import research_report as research


def row(uid, source="A", group="house", accepted=False, error=None, size=0.8):
    return {"uid": uid, "source": source, "group": group, "output": "{}",
            "error": error, "parsed": error is None, "zero_collision_accepted": accepted,
            "coll_main": 0 if accepted else 1, "size_iou_3d": size,
            "constraints": [["near", accepted]], "constraints_repaired": [["near", accepted]]}


@pytest.fixture
def run(tmp_path):
    data, output = tmp_path / "data", tmp_path / "eval"
    data.mkdir()
    rows = [row("a", "A", "one"), row("b", "A", "one"), row("c", "B", "one")]
    for split in protocol.SPLITS:
        (data / f"{split}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows))
    frozen = protocol.freeze_protocol(data, output, {"all_rows": True}, code_hashes={"eval": "pinned"})
    for tag in ("sft", "dpo1", "final"):
        model = {"path": str(tmp_path / tag), "files": {"weights": {"sha256": tag}}}
        cells = {}
        for split in protocol.SPLITS:
            target = output / tag / split
            target.mkdir(parents=True)
            predictions = [{**item, "zero_collision_accepted": tag != "sft"} for item in rows]
            (target / "predictions.jsonl").write_text("".join(json.dumps(item) + "\n" for item in predictions))
            protocol.write_json_atomic(target / "metrics.json", {"in_dist": {"rooms": 3}, "held_out": None})
            identity = {"schema_version": frozen["schema_version"], "dataset": frozen["datasets"][split],
                        "model": model, "settings": frozen["settings"], "code_hashes": frozen["code_hashes"]}
            protocol.finalize_evaluation(target, identity)
            cells[split] = {"status": "complete", "identity": identity}
        protocol.write_json_atomic(output / "state" / f"{tag}.json", {"model_identity": model, "cells": cells})
    return output


def test_all_twelve_verified_evaluations_report_only_dev_inference(run):
    result = research.generate_report(run)
    assert result["complete"] is True
    assert result["completed_evaluations"] == 12
    assert len(result["dev_comparisons"]) == 4
    assert all(item["split"].startswith("dev_") for item in result["dev_comparisons"])
    assert result["dev_comparisons"][0]["primary"]["difference"] == 1
    assert result["dev_comparisons"][0]["primary"]["evidence"] == "supported"
    assert (run / "research-summary.json").is_file()
    assert "No test-based selection" in (run / "research-summary.md").read_text()


def test_all_six_dev_available_allow_provisional_comparisons(run):
    for tag in research.MODELS:
        for split in research.TEST_SPLITS:
            (run / tag / split / "completion.json").unlink()
    result = research.generate_report(run)
    assert result["complete"] is False
    assert result["dev_ready"] is True
    assert result["completed_evaluations"] == 6
    assert len(result["dev_comparisons"]) == 4
    assert research.main(["--output", str(run)]) == 2


def test_partial_dev_cohort_blocks_inference_even_if_metrics_exist(run):
    (run / "dpo1" / "dev_rooms" / "completion.json").unlink()
    result = research.generate_report(run)
    assert result["dev_ready"] is False
    assert result["dev_comparisons"] == []
    assert result["complete"] is False


def test_same_uid_but_changed_source_or_group_is_rejected():
    with pytest.raises(ValueError, match="paired"):
        research.compare_rows([row("a")], [row("a", group="different")], "dev_rooms", "sft", "dpo1")


def test_bootstrap_resamples_clusters_not_individual_rooms():
    baseline = [row(str(index), group="large") for index in range(100)] + [row("small", group="small")]
    candidate = [{**item, "zero_collision_accepted": item["group"] == "large"} for item in baseline]
    result = research.compare_rows(baseline, candidate, "dev_rooms", "sft", "dpo1")
    assert result["primary"]["clusters"] == 2
    assert result["primary"]["difference"] == pytest.approx(100 / 101)
    assert result["primary"]["ci95"] == [0.0, 1.0]
    assert result["primary"]["evidence"] == "inconclusive"


def test_failures_remain_false_even_if_corrupt_success_flag_present():
    baseline = [row("a", group="one"), row("b", group="two")]
    candidate = [row("a", group="one", accepted=True, error="generation_error"),
                 row("b", group="two", accepted=True, error="geometry_error")]
    result = research.compare_rows(baseline, candidate, "dev_rooms", "sft", "dpo1")
    assert result["primary"]["candidate_rate"] == 0
    assert result["descriptive"]["candidate"]["size_iou_all_requests"] == 0
    assert result["descriptive"]["common_complete_size_iou"]["rooms"] == 0
    assert result["guard_warnings"]


def test_single_cluster_is_inconclusive_despite_uniform_improvement():
    before, after = [row("a")], [row("a", accepted=True)]
    result = research.compare_rows(before, after, "dev_rooms", "sft", "dpo1")
    assert result["primary"]["ci95"] is None
    assert result["primary"]["evidence"] == "inconclusive"


def test_regression_and_seed_are_reproducible():
    before = [row("a", group="one", accepted=True), row("b", group="two", accepted=True)]
    after = [{**item, "zero_collision_accepted": False} for item in before]
    result = research.compare_rows(before, after, "dev_rooms", "dpo1", "final")
    assert result["primary"]["evidence"] == "regression"
    assert result == research.compare_rows(before, after, "dev_rooms", "dpo1", "final")


def test_corrupt_metrics_receipt_cannot_be_used_for_report(run):
    (run / "final" / "test_rooms" / "metrics.json").write_text('{}')
    result = research.generate_report(run)
    assert result["complete"] is False
    assert result["completed_evaluations"] == 11


def test_cli_complete_exit_zero_and_bad_protocol_exit_two(run, tmp_path):
    assert research.main(["--output", str(run)]) == 0
    assert research.main(["--output", str(tmp_path / "missing")]) == 2


def test_test_set_never_used_for_inference():
    with pytest.raises(ValueError, match="dev"):
        research.compare_rows([row("a")], [row("a", accepted=True)], "test_rooms", "sft", "final")


def test_unknown_size_is_reported_without_inventing_acceptance_threshold():
    before = [row("a", group="one"), row("b", group="two")]
    after = [row("a", group="one", accepted=True, size=None), row("b", group="two", accepted=True, size=None)]
    result = research.compare_rows(before, after, "dev_rooms", "sft", "dpo1")
    assert result["primary"]["evidence"] == "supported"
    assert result["descriptive"]["size_iou_all_requests_difference"] is None
    assert result["guard_warnings"]


def test_completed_state_with_code_identity_drift_is_incomplete(run):
    path = run / "state" / "dpo1.json"
    state = json.loads(path.read_text())
    state["cells"]["dev_rooms"]["identity"]["code_hashes"] = {"eval": "changed"}
    protocol.write_json_atomic(path, state)
    result = research.generate_report(run)
    assert not result["dev_ready"]
    assert result["completed_evaluations"] == 11


def test_valid_receipts_with_unpaired_source_group_block_final_report(run):
    target = run / "dpo1" / "dev_rooms"
    predictions = [json.loads(line) for line in (target / "predictions.jsonl").read_text().splitlines()]
    changed = [{**item, "group": "changed"} for item in predictions]
    (target / "predictions.jsonl").write_text("".join(json.dumps(item) + "\n" for item in changed))
    state = json.loads((run / "state" / "dpo1.json").read_text())
    protocol.finalize_evaluation(target, state["cells"]["dev_rooms"]["identity"])
    result = research.generate_report(run)
    assert result["completed_evaluations"] == 12
    assert result["complete"] is False
    assert result["dev_ready"] is False
    assert "paired" in result["errors"][0]


def test_malformed_receipt_cli_fails_closed_with_exit_two(run):
    protocol.write_json_atomic(run / "state" / "dpo1.json", {"cells": []})
    assert research.main(["--output", str(run)]) == 2


def test_constraint_and_raw_metrics_keep_failure_denominators():
    before = [row("a", group="one"), row("b", group="two")]
    after = [row("a", group="one", accepted=True), row("b", group="two", accepted=True, error="bad_json")]
    result = research.compare_rows(before, after, "dev_rooms", "sft", "dpo1")
    descriptive = result["descriptive"]["candidate"]
    assert descriptive["raw_main_collision_denominator"] == 1
    assert descriptive["raw_zero_main_collision_all_requests"] == 0.5
    assert descriptive["constraints_repaired"] == {"constraints": 2, "rate": 0.5}


def test_failed_runner_identity_revalidation_invalidates_old_complete_cells(run):
    path = run / 'state/dpo1.json'
    state = json.loads(path.read_text())
    protocol.write_json_atomic(path, {**state, 'status': 'failed', 'validation_complete': False,
                                      'error': 'Frozen training lineage changed'})
    result = research.build_report(run)
    assert not result['complete']
    assert not result['dev_ready']
    assert result['completed_evaluations'] == 8
