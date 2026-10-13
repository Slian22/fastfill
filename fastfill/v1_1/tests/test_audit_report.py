"""The audit gate must never publish inference from unverified artifacts."""
import json
from pathlib import Path
import pytest
from fastfill.v1_1 import audit_report as report


def fixture(tmp_path):
    output = tmp_path / "evaluation"
    output.mkdir()
    splits = ("dev_rooms", "dev_constrained_rooms", "test_rooms", "test_constrained_rooms")
    protocol = {"datasets": {split: {"sha256": str(i) * 64} for i, split in enumerate(splits)}}
    (output / "protocol.json").write_text(json.dumps(protocol))
    addendum = tmp_path / "interpretation.json"
    addendum.write_text(json.dumps({"schema_version": 1, "research_status": "diagnostic_only",
        "dataset_sha256": {s: x["sha256"] for s, x in protocol["datasets"].items()},
        "limitations": ["Object order uses reference geometry; size-unknown deployment validity is unestablished."]}))
    return output, addendum


def verified(*args, **kwargs):
    return {"status": "complete", "rows": 4}


def synth(*args, **kwargs):
    return {"summary": {"complete": True, "status": "complete", "completed_evaluations": 12,
        "dev_comparisons": [{"primary": {"evidence": "supported"}}]}, "markdown": "# Frozen report\n\n12 complete.\n"}


def test_all_three_verifications_and_two_lineages_before_inference(tmp_path):
    out, addon = fixture(tmp_path)
    calls = []
    def verify(output, tag, workers):
        calls.append(("verify", tag)); return verified()
    def lineage(output, tag):
        calls.append(("lineage", tag)); return {"tag": tag, "verified": True}
    def synthesis(output):
        calls.append(("synthesis", "all")); return synth()
    result = report.generate_audited_report(out, addon, 2, verifier=verify,
        lineage_verifier=lineage, synthesizer=synthesis)
    assert calls == [("verify", t) for t in ("sft", "dpo1", "final")] + [
        ("lineage", t) for t in ("dpo1", "final")] + [("synthesis", "all")]
    assert result["complete"] and result["research_status"] == "diagnostic_only"
    assert result["audit"]["status"] == "passed"
    assert "Object order uses reference geometry" in (out / "research-summary.md").read_text()
    assert json.loads((out / "research-summary.json").read_text()) == result


@pytest.mark.parametrize("bad_tag", ["sft", "dpo1", "final"])
def test_verification_failure_blocks_all_inference(tmp_path, bad_tag):
    out, addon = fixture(tmp_path)
    def verify(output, tag, workers):
        if tag == bad_tag: raise ValueError("scoring mismatch")
        return verified()
    def forbidden(*args): raise AssertionError("must not synthesize invalid evidence")
    result = report.generate_audited_report(out, addon, 1, verifier=verify,
        lineage_verifier=verified, synthesizer=forbidden)
    assert not result["complete"] and result["dev_comparisons"] == []
    assert result["audit"]["status"] == "failed"
    assert "scoring mismatch" in str(result["errors"])
    assert "incomplete" in (out / "research-summary.md").read_text()


def test_failed_receipt_is_not_a_success(tmp_path):
    out, addon = fixture(tmp_path)
    result = report.generate_audited_report(out, addon, 1,
        verifier=lambda *args: {"status": "failed"}, lineage_verifier=verified, synthesizer=synth)
    assert not result["complete"] and result["dev_comparisons"] == []


def test_lineage_failure_blocks_inference(tmp_path):
    out, addon = fixture(tmp_path)
    def broken(*args): raise ValueError("not a completed trainer output")
    result = report.generate_audited_report(out, addon, 1, verifier=verified,
        lineage_verifier=broken, synthesizer=synth)
    assert not result["complete"] and "not a completed trainer output" in str(result["errors"])


def test_addendum_cannot_be_applied_to_different_cohort(tmp_path):
    out, addon = fixture(tmp_path)
    value = json.loads(addon.read_text()); value["dataset_sha256"]["test_rooms"] = "f" * 64
    addon.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cohort"):
        report.generate_audited_report(out, addon, 1, verifier=verified, lineage_verifier=verified, synthesizer=synth)


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_workers_rejected(tmp_path, workers):
    out, addon = fixture(tmp_path)
    with pytest.raises(ValueError, match="workers"):
        report.generate_audited_report(out, addon, workers, verifier=verified, lineage_verifier=verified, synthesizer=synth)


def test_synthesis_exception_produces_incomplete_report(tmp_path):
    out, addon = fixture(tmp_path)
    def broken(*args): raise RuntimeError("frozen report failed")
    result = report.generate_audited_report(out, addon, 1, verifier=verified, lineage_verifier=verified, synthesizer=broken)
    assert not result["complete"] and "frozen report failed" in str(result["errors"])


def test_cli_exit_status(tmp_path, monkeypatch):
    out, addon = fixture(tmp_path)
    monkeypatch.setattr(report, "generate_audited_report", lambda *args: {"complete": False, "status": "incomplete"})
    assert report.main(["--output", str(out), "--interpretation", str(addon)]) == 2
    monkeypatch.setattr(report, "generate_audited_report", lambda *args: {"complete": True, "status": "complete"})
    assert report.main(["--output", str(out), "--interpretation", str(addon)]) == 0


@pytest.mark.parametrize("payload", [[], {"summary": []}, {"summary": {"complete": True}, "markdown": []}])
def test_malformed_synthesis_fails_closed(tmp_path, payload):
    out, addon = fixture(tmp_path)
    result = report.generate_audited_report(out, addon, 1, verifier=verified,
        lineage_verifier=verified, synthesizer=lambda *args: payload)
    assert not result["complete"] and not result["dev_comparisons"]


def test_addendum_drift_blocks_publication(tmp_path):
    out, addon = fixture(tmp_path)
    def changed(*args):
        value = json.loads(addon.read_text())
        value["limitations"] = ["Changed after verification"]
        addon.write_text(json.dumps(value))
        return synth()
    result = report.generate_audited_report(out, addon, 1, verifier=verified,
        lineage_verifier=verified, synthesizer=changed)
    assert not result["complete"] and "changed" in str(result["errors"]).lower()


def test_absolute_verifier_wrapper_reads_receipt(tmp_path, monkeypatch):
    out, _ = fixture(tmp_path)
    (out / "runner.json").write_text(json.dumps({"root": str(tmp_path / "old root")}))
    (out / "verification").mkdir()
    (out / "verification/sft.json").write_text(json.dumps(verified()))
    commands = []
    monkeypatch.setattr(report.subprocess, "run", lambda cmd, **kw: commands.append((cmd, kw)))
    assert report._verify_outputs(out, "sft", 2) == verified()
    cmd, kw = commands[0]
    assert Path(cmd[1]).is_absolute() and Path(cmd[1]).name == "verify_outputs.py"
    assert cmd[cmd.index("--scoring-root") + 1] == str(tmp_path / "old root")
    assert kw["check"] is True


def test_frozen_synthesis_uses_isolated_original_interpreter(tmp_path):
    import sys
    out, _ = fixture(tmp_path)
    frozen = tmp_path / "frozen root"
    package = frozen / "fastfill/v1_1"
    package.mkdir(parents=True)
    (frozen / "fastfill/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "research_report.py").write_text(
        "def build_report(output): return {'complete': True, 'output': output}\n"
        "def _markdown(summary): return 'original frozen synthesis'\n")
    (out / "runner.json").write_text(json.dumps({"root": str(frozen), "python": sys.executable}))
    result = report._synthesize_frozen(out)
    assert result["summary"] == {"complete": True, "output": str(out)}
    assert result["markdown"] == "original frozen synthesis"


def test_lineage_wrapper_uses_runner_run(tmp_path, monkeypatch):
    from fastfill.v1_1 import training_guard
    out, _ = fixture(tmp_path)
    (out / "runner.json").write_text(json.dumps({"run": "clean run"}))
    monkeypatch.setattr(training_guard, "verify_training_lineage", lambda run, tag: {"run": run, "tag": tag})
    assert report._verify_lineage(out, "dpo1") == {"run": "clean run", "tag": "dpo1"}


def test_cli_invalid_addendum_replaces_stale_report(tmp_path):
    out, addon = fixture(tmp_path)
    addon.write_text("[]")
    (out / "research-summary.json").write_text(json.dumps({"complete": True}))
    assert report.main(["--output", str(out), "--interpretation", str(addon)]) == 2
    assert json.loads((out / "research-summary.json").read_text())["complete"] is False


@pytest.mark.parametrize("change", [{"limitations": []}, {"research_status": "validated"}])
def test_invalid_interpretation_rejected(tmp_path, change):
    out, addon = fixture(tmp_path)
    addon.write_text(json.dumps({**json.loads(addon.read_text()), **change}))
    with pytest.raises(ValueError, match="interpretation"):
        report._interpretation(out, addon)
