"""Regression tests for full-cohort evaluation identity and completion."""
import json
from pathlib import Path

import pytest

from fastfill.v1_1 import evaluation_protocol as protocol


def safetensors(path, payload=b"model weights"):
    header = json.dumps({"weight": {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for split in protocol.SPLITS:
        (data / f"{split}.jsonl").write_text(
            '\n'.join(json.dumps({"uid": uid}) for uid in ("a", "b")) + '\n'
        )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{}')
    (model / "tokenizer_config.json").write_text('{}')
    (model / "tokenizer.json").write_text('{}')
    safetensors(model / "model.safetensors")
    return data, model


def identity(inputs):
    data, model = inputs
    return protocol.evaluation_identity(data / "test_rooms.jsonl", model, {"all_rows": True})


def artifacts(out, uids=("a", "b")):
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text(
        '\n'.join(json.dumps({"uid": uid, "output": "{}"}) for uid in uids) + '\n'
    )
    (out / "metrics.json").write_text(json.dumps({
        "in_dist": {"rooms": len(uids), "accepted_rate": 0.5}, "held_out": None,
    }))


def test_legacy_sample_metrics_are_never_complete(inputs, tmp_path):
    out = tmp_path / "out"
    artifacts(out, ("a",))
    assert not protocol.evaluation_complete(out, identity(inputs))
    with pytest.raises(ValueError, match="cohort"):
        protocol.finalize_evaluation(out, identity(inputs))


def test_complete_artifacts_bind_all_inputs_and_detect_tampering(inputs, tmp_path):
    out = tmp_path / "out"
    artifacts(out)
    expected = identity(inputs)
    protocol.finalize_evaluation(out, expected)
    assert protocol.evaluation_complete(out, expected)
    changed = {**expected, "settings": {"all_rows": True, "max_len": 100}}
    assert not protocol.evaluation_complete(out, changed)
    (out / "metrics.json").write_text('{"in_dist":{"rooms":2},"held_out":null}')
    assert not protocol.evaluation_complete(out, expected)


def test_code_hash_drift_invalidates_completion(inputs, tmp_path):
    out = tmp_path / "out"
    artifacts(out)
    data, model = inputs
    expected = protocol.evaluation_identity(data / "test_rooms.jsonl", model, {}, code_hashes={"eval.py": "old"})
    protocol.finalize_evaluation(out, expected)
    changed = protocol.evaluation_identity(data / "test_rooms.jsonl", model, {}, code_hashes={"eval.py": "new"})
    assert not protocol.evaluation_complete(out, changed)


def test_duplicate_uids_preserve_order_without_dropping_rows(inputs, tmp_path):
    rooms = inputs[0] / "test_rooms.jsonl"
    rooms.write_text('{"uid":"a"}\n{"uid":"a"}\n')
    out = tmp_path / "out"
    artifacts(out, ("a", "a"))
    expected = identity(inputs)
    protocol.finalize_evaluation(out, expected)
    assert protocol.evaluation_complete(out, expected)


def test_prediction_corruption_invalidates_completion(inputs, tmp_path):
    out = tmp_path / "out"
    artifacts(out)
    expected = identity(inputs)
    protocol.finalize_evaluation(out, expected)
    (out / "predictions.jsonl").write_text('partial JSON')
    assert not protocol.evaluation_complete(out, expected)
    (out / "completion.json").write_text('null')
    assert not protocol.evaluation_complete(out, expected)


def test_same_count_wrong_or_reordered_cohort_is_rejected(inputs, tmp_path):
    out = tmp_path / "out"
    for uids in (("b", "a"), ("a", "c"), ("a", "a")):
        artifacts(out, uids)
        with pytest.raises(ValueError, match="cohort"):
            protocol.finalize_evaluation(out, identity(inputs))


def test_metrics_count_must_match_predictions(inputs, tmp_path):
    out = tmp_path / "out"
    artifacts(out)
    (out / "metrics.json").write_text('{"in_dist":{"rooms":200},"held_out":null}')
    with pytest.raises(ValueError, match="metrics"):
        protocol.finalize_evaluation(out, identity(inputs))


def test_model_hash_changes_when_weights_change(inputs):
    before = identity(inputs)
    safetensors(inputs[1] / "model.safetensors", b"changed weights")
    assert before != identity(inputs)


def pin_stat(path, monkeypatch):
    """Model a filesystem that reports unchanged metadata after a short write."""
    target = path.resolve()
    state, original = target.stat(), Path.stat
    monkeypatch.setattr(Path, "stat", lambda self, *a, **kw: state if self == target else original(self, *a, **kw))


@pytest.mark.parametrize("name", ["model.safetensors", "tokenizer.json"])
def test_model_identity_rehashes_changed_bytes_with_unchanged_stat(inputs, monkeypatch, name):
    model = inputs[1]
    target = model / name
    if name == "tokenizer.json":
        target.write_text('{"v":1}')
    pin_stat(target, monkeypatch)
    before = protocol.model_identity(model)
    if name == "model.safetensors":
        safetensors(target, b"changed bytes")  # Same length as "model weights".
    else:
        target.write_text('{"v":2}')
    actual = protocol.file_identity(target)
    assert actual != before["files"][name]
    after = protocol.model_identity(model)
    assert after["files"][name] == actual
    assert after != before


def test_model_identity_revalidates_header_with_unchanged_stat(inputs, monkeypatch):
    model = inputs[1]
    target = model / "model.safetensors"
    pin_stat(target, monkeypatch)
    protocol.model_identity(model)
    target.write_bytes(b"\0" * 8 + target.read_bytes()[8:])
    with pytest.raises(ValueError, match="weight header"):
        protocol.model_identity(model)


def test_missing_indexed_model_shard_is_rejected(inputs):
    model = inputs[1]
    (model / "model.safetensors").unlink()
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-00001-of-00001.safetensors"},
    }))
    with pytest.raises(ValueError, match="weight"):
        protocol.model_identity(model)


def test_indexed_model_all_shards_and_tokenizer_are_hashed(inputs):
    model = inputs[1]
    (model / "model.safetensors").unlink()
    safetensors(model / "shard-1.safetensors", b"first")
    safetensors(model / "shard-2.safetensors", b"second")
    (model / "tokenizer.json").write_text('{}')
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"first": "shard-1.safetensors", "second": "shard-2.safetensors"},
    }))
    fingerprint = protocol.model_identity(model)
    assert {"shard-1.safetensors", "shard-2.safetensors", "tokenizer.json"} <= fingerprint["files"].keys()


def test_indexed_model_cannot_escape_model_directory(inputs):
    model = inputs[1]
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "../outside.safetensors"},
    }))
    with pytest.raises(ValueError, match="weight"):
        protocol.model_identity(model)


def test_freeze_protocol_is_idempotent_but_rejects_data_or_settings_drift(inputs, tmp_path):
    data, _ = inputs
    out = tmp_path / "frozen"
    frozen = protocol.freeze_protocol(data, out, {"all_rows": True})
    assert len(frozen["datasets"]) == 4
    assert frozen == protocol.freeze_protocol(data, out, {"all_rows": True})
    with pytest.raises(ValueError, match="protocol"):
        protocol.freeze_protocol(data, out, {"all_rows": False})
    (data / "dev_rooms.jsonl").write_text('{"uid":"changed"}\n')
    with pytest.raises(ValueError, match="protocol"):
        protocol.freeze_protocol(data, out, {"all_rows": True})


def test_summary_requires_every_requested_evaluation(inputs, tmp_path):
    out = tmp_path / "out"
    artifacts(out)
    expected = identity(inputs)
    protocol.finalize_evaluation(out, expected)
    result = protocol.summarize_evaluations({
        "sft/test": (out, expected), "final/test": (tmp_path / "missing", expected),
    })
    assert result["complete"] is False
    assert result["evaluations"]["sft/test"]["status"] == "complete"
    assert result["evaluations"]["final/test"]["status"] == "incomplete"
    assert result["evaluations"]["sft/test"]["metrics"]["in_dist"]["accepted_rate"] == 0.5


def test_empty_or_malformed_datasets_fail_closed(tmp_path):
    rooms = tmp_path / "rooms.jsonl"
    for text in ('', '{}\n', '\n', '{"uid":"a"}\nnot json\n'):
        rooms.write_text(text)
        with pytest.raises(ValueError):
            protocol.dataset_identity(rooms)


def test_json_writer_preserves_existing_file_if_serialization_fails(tmp_path):
    target = tmp_path / "state.json"
    protocol.write_json_atomic(target, {"status": "pending"})
    with pytest.raises(TypeError):
        protocol.write_json_atomic(target, {"bad": object()})
    assert json.loads(target.read_text()) == {"status": "pending"}


@pytest.mark.parametrize("metrics", [[], {}, {"in_dist": {"rooms": True}, "held_out": None}])
def test_bad_metrics_fail_closed(inputs, tmp_path, metrics):
    out = tmp_path / "out"
    artifacts(out)
    (out / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="metrics"):
        protocol.finalize_evaluation(out, identity(inputs))


def test_missing_prediction_output_rejected(inputs, tmp_path):
    out = tmp_path / "out"
    artifacts(out)
    (out / "predictions.jsonl").write_text('{"uid":"a"}\n{"uid":"b"}\n')
    with pytest.raises(ValueError, match="output"):
        protocol.finalize_evaluation(out, identity(inputs))


def test_model_missing_configuration_or_empty_weight_rejected(inputs):
    model = inputs[1]
    (model / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="weight"):
        protocol.model_identity(model)
    (model / "config.json").unlink()
    with pytest.raises(ValueError, match="configuration"):
        protocol.model_identity(model)


def test_empty_evaluation_set_is_not_complete():
    assert not protocol.summarize_evaluations({})["complete"]


def test_partial_safetensors_merge_is_rejected(inputs):
    weights = inputs[1] / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1])
    with pytest.raises(ValueError, match="weight"):
        protocol.model_identity(inputs[1])


def test_missing_tokenizer_and_invalid_config_are_rejected(inputs):
    model = inputs[1]
    (model / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer"):
        protocol.model_identity(model)
    (model / "tokenizer.json").write_text('{}')
    (model / "config.json").write_text('[]')
    with pytest.raises(ValueError, match="configuration"):
        protocol.model_identity(model)
