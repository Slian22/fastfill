"""Real artifact tests for isolated DPO resume guards."""
import json
from pathlib import Path
import shutil
import sys

import pytest

from fastfill.v1_1.training_guard import guard_training, completed_adapter, verify_pair_receipt, require_isolated_run
from fastfill.v1_1.evaluation_protocol import file_identity
from fastfill.v1_1 import training_guard


def weight(path):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "base"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (model / name).write_text("{}")
    weight(model / "model.safetensors")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{"uid":"a","chosen":"good","rejected":"bad"}\n')
    code = tmp_path / "train.py"
    code.write_text("# frozen trainer\n")
    return model, pairs, code, tmp_path / "run"


def test_identical_resume_is_allowed_and_returns_same_receipt(inputs):
    model, pairs, code, out = inputs
    first = guard_training(model, pairs, out, {"epochs": 2}, [code])
    assert guard_training(model, pairs, out, {"epochs": 2}, [code]) == first
    assert json.loads((out / "training_guard.json").read_text()) == first


@pytest.mark.parametrize("changed", ["pairs", "weights", "settings", "code"])
def test_resume_rejects_changed_training_identity(inputs, changed):
    model, pairs, code, out = inputs
    guard_training(model, pairs, out, {"epochs": 2}, [code])
    if changed == "pairs":
        pairs.write_text('{"uid":"b","chosen":"good","rejected":"bad"}\n')
    elif changed == "weights":
        p = model / "model.safetensors"
        p.write_bytes(p.read_bytes()[:-4] + b"1234")
    elif changed == "code":
        code.write_text("# changed trainer\n")
    settings = {"epochs": 3 if changed == "settings" else 2}
    with pytest.raises(ValueError, match="identity"):
        guard_training(model, pairs, out, settings, [code])


def test_existing_unbound_checkpoint_cannot_be_adopted(inputs):
    model, pairs, code, out = inputs
    (out / "ckpt_latest").mkdir(parents=True)
    with pytest.raises(ValueError, match="unbound"):
        guard_training(model, pairs, out, {"epochs": 2}, [code])


def test_adapter_must_be_structurally_complete(tmp_path):
    assert not completed_adapter(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}")
    weight(tmp_path / "adapter_model.safetensors")
    assert completed_adapter(tmp_path)
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    assert not completed_adapter(tmp_path)


def test_training_guard_rejects_empty_pairs(inputs):
    model, pairs, code, out = inputs
    pairs.write_text("")
    with pytest.raises(ValueError, match="Empty"):
        guard_training(model, pairs, out, {"epochs": 2}, [code])


def test_sanitized_pair_receipt_is_required_and_byte_bound(tmp_path):
    from fastfill.v1_1.audit_pairs import POLICY
    import fastfill.v1_1.audit_pairs as audit_module
    with pytest.raises(FileNotFoundError):
        verify_pair_receipt(tmp_path)
    outputs = {}
    for stage in ("stage1", "stage2"):
        p = tmp_path / f"{stage}.cleaned.jsonl"
        p.write_text('{"uid":"a"}\n')
        outputs[stage] = {"file": p.name, "sha256": file_identity(p)["sha256"], "kept": 1}
    package = Path(audit_module.__file__).resolve().parent.parent
    names = ("v1_1/audit_pairs.py", "v1_1/build_dpo.py", "v1_1/scene_v1_1.py",
             "validate.py", "scene.py", "anchors.py")
    receipt = {
        "protocol": "fastfill_v1.1_pair_sanitizer_v1", "outputs": outputs,
        "policy": POLICY,
        "inputs": {k: {"path": f"/recorded/{k}", "sha256": "a" * 64} for k in ("train", "stage1", "stage2")},
        "code_sha256": {name: file_identity(package / name)["sha256"] for name in names},
    }
    (tmp_path / "audit.json").write_text(json.dumps(receipt))
    assert verify_pair_receipt(tmp_path)["outputs"] == outputs
    for field in ("inputs", "policy", "code_sha256"):
        (tmp_path / "audit.json").write_text(json.dumps({k: v for k, v in receipt.items() if k != field}))
        with pytest.raises(ValueError, match="audited"):
            verify_pair_receipt(tmp_path)
    (tmp_path / "audit.json").write_text(json.dumps(receipt))
    (tmp_path / "stage1.cleaned.jsonl").write_text('{"uid":"b"}\n')
    with pytest.raises(ValueError, match="audited"):
        verify_pair_receipt(tmp_path)


def test_original_run_and_symlink_alias_are_protected(tmp_path):
    original = tmp_path / "original"
    old = original / "runs" / "fastfill_v1_1"
    old.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(old, target_is_directory=True)
    for path in (old, old / "subdir", original, alias):
        with pytest.raises(ValueError, match="isolated"):
            require_isolated_run(original, path)
    clean = original / "runs" / "fastfill_v1_1_clean_20261011"
    assert require_isolated_run(original, clean) == clean
    clean.mkdir()
    (clean / "dpo1_lora").symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="isolated"):
        require_isolated_run(original, clean)


def test_audit_receipt_change_refuses_resume(inputs):
    model, pairs, code, out = inputs
    receipt = pairs.parent / "audit.json"
    receipt.write_text('{"identity":1}')
    guard_training(model, pairs, out, {"epochs": 2}, [code])
    receipt.write_text('{"identity":2}')
    with pytest.raises(ValueError, match="identity"):
        guard_training(model, pairs, out, {"epochs": 2}, [code])


def test_merged_output_must_match_exact_base_adapter_and_output(inputs, tmp_path):
    from fastfill.v1_1.training_guard import seal_merge, merge_complete
    import shutil

    model, _, _, _ = inputs
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for name in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json"):
        (adapter / name).write_text("{}")
    weight(adapter / "adapter_model.safetensors")
    merged = tmp_path / "merged"
    shutil.copytree(model, merged)
    assert not merge_complete(model, adapter, merged)
    seal_merge(model, adapter, merged)
    assert merge_complete(model, adapter, merged)
    p = merged / "model.safetensors"
    p.write_bytes(p.read_bytes()[:-4] + b"1234")
    assert not merge_complete(model, adapter, merged)
    seal_merge(model, adapter, merged)
    (adapter / "adapter_config.json").write_text('{"r":32}')
    assert not merge_complete(model, adapter, merged)


def run_cli(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["training_guard", *map(str, arguments)])
    return training_guard.main()


@pytest.fixture
def sanitized_pairs(tmp_path):
    """Generate a real sanitizer receipt using valid geometry and source files."""
    from fastfill.v1_1.audit_pairs import sanitize_pairs

    user = {"boundary_type": "polygon", "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
            "height": 3, "objects": [{"id": "bed_1"}], "constraints": []}
    chosen = {"placements": [{"id": "bed_1", "size": [1, 1, 0.5], "pos": [2, 2, 0], "yaw": 0}]}
    rejected = {"placements": [{"id": "bed_1", "size": [1, 1, 0.5], "pos": [10, 10, 0], "yaw": 0}]}
    messages = [{"role": "system", "content": "fixture"},
                {"role": "user", "content": json.dumps(user)},
                {"role": "assistant", "content": json.dumps(chosen)}]
    reference = {"uid": "train-room", "source": "fixture", "messages": messages}
    pair = {"uid": reference["uid"], "source": reference["source"], "prompt": messages[:2],
            "chosen": messages[2]["content"], "rejected": json.dumps(rejected)}
    paths = [tmp_path / name for name in ("train.jsonl", "stage1.jsonl", "stage2.jsonl")]
    for path, row in zip(paths, (reference, pair, pair)):
        path.write_text(json.dumps(row) + "\n")
    out = tmp_path / "sanitized"
    receipt = sanitize_pairs(*paths, out, workers=1)
    assert all(entry["sha256"] == file_identity(entry["path"])["sha256"]
               for entry in receipt["inputs"].values())
    return out


@pytest.mark.parametrize("action", ["bind", "isolate", "merge-complete", "seal-merge"])
def test_cli_missing_required_action_arguments_exit_two(tmp_path, monkeypatch, action):
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, action, "--out", tmp_path / "output")
    assert error.value.code == 2


@pytest.mark.parametrize("action", ["model-complete", "adapter-complete", "merge-complete"])
def test_cli_incomplete_artifacts_exit_one(inputs, tmp_path, monkeypatch, action):
    model, _, _, _ = inputs
    arguments = [action, "--out", tmp_path / "absent"]
    if action == "merge-complete":
        arguments += ["--model", model, "--adapter", tmp_path / "absent-adapter"]
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, *arguments)
    assert error.value.code == 1


def test_cli_bind_verifies_actual_sanitizer_receipt_and_resume_settings(inputs, sanitized_pairs, monkeypatch, capsys):
    model, _, _, out = inputs
    run_cli(monkeypatch, "verify-pairs", "--out", sanitized_pairs)
    arguments = ["bind", "--model", model, "--pairs", sanitized_pairs / "stage1.cleaned.jsonl",
                 "--out", out, "--lr", "0.000005", "--nproc", "2"]
    run_cli(monkeypatch, *arguments)
    assert json.loads(capsys.readouterr().out) == {"status": "identity_verified", "out": str(out), "pairs": 1}
    receipt = json.loads((out / "training_guard.json").read_text())
    assert receipt["settings"]["world_size"] == 2
    assert receipt["settings"]["lr"] == 5e-6
    assert receipt["audit_receipt"] == file_identity(sanitized_pairs / "audit.json")
    run_cli(monkeypatch, *arguments)
    capsys.readouterr()
    with pytest.raises(ValueError, match="identity"):
        run_cli(monkeypatch, *arguments[:-1], "4")


def test_cli_bind_rejects_tampered_sanitized_pairs_before_writing_guard(inputs, sanitized_pairs, monkeypatch):
    model, _, _, out = inputs
    pairs = sanitized_pairs / "stage1.cleaned.jsonl"
    pairs.write_text(pairs.read_text().replace('"train-room"', '"changed-room"'))
    with pytest.raises(ValueError, match="audited receipt"):
        run_cli(monkeypatch, "bind", "--model", model, "--pairs", pairs, "--out", out, "--lr", "0.000005")
    assert not (out / "training_guard.json").exists()


def test_cli_complete_models_adapters_and_sealed_merges(inputs, tmp_path, monkeypatch):
    model, _, _, _ = inputs
    adapter, merged = tmp_path / "adapter", tmp_path / "merged"
    adapter.mkdir()
    for name in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json"):
        (adapter / name).write_text("{}")
    weight(adapter / "adapter_model.safetensors")
    shutil.copytree(model, merged)
    run_cli(monkeypatch, "model-complete", "--out", model)
    with pytest.raises(SystemExit) as completed:
        run_cli(monkeypatch, "adapter-complete", "--out", adapter)
    assert completed.value.code == 0
    common = ["--model", model, "--adapter", adapter, "--out", merged]
    with pytest.raises(SystemExit) as unbound:
        run_cli(monkeypatch, "merge-complete", *common)
    assert unbound.value.code == 1
    run_cli(monkeypatch, "seal-merge", *common)
    assert training_guard.merge_receipt_path(merged).is_file()
    with pytest.raises(SystemExit) as complete:
        run_cli(monkeypatch, "merge-complete", *common)
    assert complete.value.code == 0
    weights = merged / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1])
    with pytest.raises(SystemExit) as changed:
        run_cli(monkeypatch, "merge-complete", *common)
    assert changed.value.code == 1


def test_cli_isolate_accepts_new_run_and_rejects_original(tmp_path, monkeypatch):
    original = tmp_path / "project"
    clean = original / "runs" / "fastfill_v1_1_clean_test"
    run_cli(monkeypatch, "isolate", "--model", original, "--out", clean)
    with pytest.raises(ValueError, match="isolated"):
        run_cli(monkeypatch, "isolate", "--model", original, "--out", original / "runs" / "fastfill_v1_1")


@pytest.mark.parametrize("change", ["protocol", "code_digest", "code_path", "output_name"])
def test_cli_pair_verification_rejects_untrusted_receipt(sanitized_pairs, monkeypatch, change):
    path = sanitized_pairs / "audit.json"
    receipt = json.loads(path.read_text())
    if change == "protocol":
        changed = {**receipt, "protocol": "untrusted"}
    elif change == "code_digest":
        changed = {**receipt, "code_sha256": {**receipt["code_sha256"], "scene.py": "0" * 64}}
    elif change == "code_path":
        changed = {**receipt, "code_sha256": {**receipt["code_sha256"], "../outside.py": "0" * 64}}
    else:
        changed_stage = {**receipt["outputs"]["stage1"], "file": "unexpected.jsonl"}
        changed = {**receipt, "outputs": {**receipt["outputs"], "stage1": changed_stage}}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="protocol|code|filename"):
        run_cli(monkeypatch, "verify-pairs", "--out", sanitized_pairs)
