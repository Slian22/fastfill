"""CPU-only regressions for authoritative, resumable whole-cohort evaluation."""
import argparse
import json
from pathlib import Path
import contextlib
import sys
import types

import pytest

from fastfill.v1_1 import evaluate_v1_1 as evaluator


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    rooms = tmp_path / "rooms.jsonl"
    # Identical UIDs and prompts are deliberately separate evaluation rows.
    rooms.write_text("\n".join(json.dumps({"uid": uid}) for uid in ("a", "a", "b", "c", "d")) + "\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer_config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (model / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
    args = argparse.Namespace(rooms=str(rooms), model=str(model), out=str(tmp_path / "out"),
        all_rows=True, resume_evaluation=True, chunk_size=2, gt_only=False,
        max_len=128, max_new_tokens=64, batch=2, tp=1, use_vllm=True, workers=1, limit=None)
    monkeypatch.setattr(evaluator, "_load_tokenizer", lambda model: object(), raising=False)
    monkeypatch.setattr(evaluator, "messages_v1_1", lambda room, *args, **kwargs: room["uid"])
    monkeypatch.setattr(evaluator, "prompt_text", lambda tok, messages: messages)
    monkeypatch.setattr(evaluator, "_score_worker", lambda task: {
        "uid": task[0]["uid"], "error": task[2], "accepted": task[2] is None})
    monkeypatch.setattr(evaluator, "summarize_v1_1", lambda rows: {
        "in_dist": {"rooms": len(rows), "accepted_rate": sum(r["accepted"] for r in rows) / len(rows)},
        "held_out": None})
    return args


def session(monkeypatch, fail_on=None, outputs=None):
    calls, loads = [], []
    class FakeSession:
        def __init__(self, *args, **kwargs):
            loads.append(1)
        def generate(self, prompts):
            calls.append(list(prompts))
            if len(calls) == fail_on:
                raise RuntimeError("interrupted")
            return outputs(prompts) if outputs else [f"output-{p}" for p in prompts]
    monkeypatch.setattr(evaluator, "_GenerationSession", FakeSession, raising=False)
    return calls, loads


def test_interrupted_resume_preserves_duplicates_and_does_not_regenerate(evaluation, monkeypatch):
    calls, loads = session(monkeypatch, fail_on=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluator.run_resumable_evaluation(evaluation)
    assert calls == [["a", "a"], ["b", "c"]]
    assert not (Path(evaluation.out) / "metrics.json").exists()
    calls, loads = session(monkeypatch)
    evaluator.run_resumable_evaluation(evaluation)
    assert calls == [["b", "c"], ["d"]]
    assert len(loads) == 1
    predictions = [json.loads(line) for line in (Path(evaluation.out) / "predictions.jsonl").read_text().splitlines()]
    assert [row["uid"] for row in predictions] == ["a", "a", "b", "c", "d"]
    assert (Path(evaluation.out) / "completion.json").exists()
    calls, loads = session(monkeypatch)
    evaluator.run_resumable_evaluation(evaluation)
    assert calls == loads == []


def test_scoring_interruption_reuses_saved_generation(evaluation, monkeypatch):
    session(monkeypatch)
    scorer = evaluator._score_worker
    monkeypatch.setattr(evaluator, "_score_worker", lambda task: (_ for _ in ()).throw(RuntimeError("score crash")))
    with pytest.raises(RuntimeError, match="score crash"):
        evaluator.run_resumable_evaluation(evaluation)
    monkeypatch.setattr(evaluator, "_score_worker", scorer)
    calls, _ = session(monkeypatch)
    evaluator.run_resumable_evaluation(evaluation)
    assert calls == [["b", "c"], ["d"]]


@pytest.mark.parametrize("change", ["data", "model", "settings", "chunk"])
def test_changed_inputs_cannot_reuse_partial_results(evaluation, monkeypatch, change):
    session(monkeypatch, fail_on=2)
    with pytest.raises(RuntimeError):
        evaluator.run_resumable_evaluation(evaluation)
    if change == "data":
        Path(evaluation.rooms).write_text('{"uid":"other"}\n')
    elif change == "model":
        weights = Path(evaluation.model) / "model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-4] + b"\1" * 4)
    elif change == "settings":
        evaluation.max_new_tokens += 1
    else:
        evaluation.chunk_size += 1
    with pytest.raises(ValueError, match="identity|manifest"):
        evaluator.run_resumable_evaluation(evaluation)


def test_legacy_output_is_preserved_and_rejected(evaluation, monkeypatch):
    out = Path(evaluation.out)
    out.mkdir()
    (out / "metrics.json").write_text('{"legacy":true}')
    session(monkeypatch)
    with pytest.raises(ValueError, match="existing|legacy"):
        evaluator.run_resumable_evaluation(evaluation)
    assert (out / "metrics.json").read_text() == '{"legacy":true}'


def test_missing_generation_stays_in_denominator(evaluation, monkeypatch):
    session(monkeypatch, outputs=lambda prompts: [None] * len(prompts))
    evaluator.run_resumable_evaluation(evaluation)
    predictions = [json.loads(line) for line in (Path(evaluation.out) / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 5
    assert all(row["error"] == "missing_generation" and not row["accepted"] for row in predictions)
    assert json.loads((Path(evaluation.out) / "metrics.json").read_text())["in_dist"]["accepted_rate"] == 0


def test_short_generation_list_fails_without_publishing(evaluation, monkeypatch):
    session(monkeypatch, outputs=lambda prompts: ["only one"])
    with pytest.raises(ValueError, match="count"):
        evaluator.run_resumable_evaluation(evaluation)
    assert not (Path(evaluation.out) / "metrics.json").exists()


def test_corrupt_saved_chunk_cannot_resume(evaluation, monkeypatch):
    session(monkeypatch, fail_on=2)
    with pytest.raises(RuntimeError):
        evaluator.run_resumable_evaluation(evaluation)
    chunk = next((Path(evaluation.out) / "chunks").glob("*.generated.json"))
    saved = json.loads(chunk.read_text())
    chunk.write_text(json.dumps({**saved, "uids": ["wrong", "wrong"]}))
    with pytest.raises(ValueError, match="chunk"):
        evaluator.run_resumable_evaluation(evaluation)


def test_hf_generation_budget_uses_longest_padded_prompt():
    assert evaluator._hf_generation_budget([(5, 0), (19, 1)], 20, 8) == 1
    assert evaluator._hf_generation_budget([(5, 0), (7, 1)], 20, 8) == 8


def test_real_hf_generation_uses_padded_width_and_one_loaded_model(monkeypatch):
    calls, loads = [], []
    class Tensor:
        shape = (2, 19)
        def __getitem__(self, key):
            return self
    class Encoded(dict):
        def to(self, device):
            return self
    class Tokenizer:
        pad_token = "pad"
        def __call__(self, prompts, **kwargs):
            if isinstance(prompts, str):
                return {"input_ids": [0] * len(prompts)}
            return Encoded(input_ids=Tensor())
        def batch_decode(self, outputs, **kwargs):
            return ["first", "second"]
    class Model:
        device = "cpu"
        def to(self, device):
            return self
        def eval(self):
            pass
        def generate(self, **kwargs):
            calls.append(kwargs["max_new_tokens"])
            return Tensor()
    def load(*args, **kwargs):
        loads.append(1)
        return Model()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(inference_mode=contextlib.nullcontext,
        bfloat16="bf16", cuda=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load)))
    monkeypatch.setattr(evaluator, "_load_tokenizer", lambda model: Tokenizer())
    args = argparse.Namespace(model="fake", use_vllm=False, max_len=20, max_new_tokens=8, batch=2)
    generator = evaluator._GenerationSession(args)
    assert generator.generate(["x" * 5, "x" * 19]) == ["first", "second"]
    generator.generate(["x" * 5, "x" * 19])
    assert calls == [1, 1] and loads == [1]


def test_real_vllm_session_preserves_oversize_prompt_failures_and_reuses_model(monkeypatch):
    calls, loads = [], []
    class Model:
        def __init__(self, **kwargs):
            loads.append(1)
        def get_tokenizer(self):
            return lambda prompt: {"input_ids": [0] * len(prompt)}
        def generate(self, prompts, parameters):
            calls.append((prompts, parameters))
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=prompt)]) for prompt in prompts]
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=Model, SamplingParams=types.SimpleNamespace))
    monkeypatch.setattr(evaluator, "_effective_tp", lambda tp: tp)
    args = argparse.Namespace(model="fake", use_vllm=True, max_len=8, max_new_tokens=6, tp=1)
    generator = evaluator._GenerationSession(args)
    assert generator.generate(["ok", "too-long!"]) == ["ok", None]
    assert generator.generate(["fine"]) == ["fine"]
    assert len(loads) == 1
    assert calls[0][1][0].max_tokens == 6
    assert calls[1][1][0].max_tokens == 4


def test_dataset_mutation_during_generation_prevents_publication(evaluation, monkeypatch):
    def changed(prompts):
        rooms = Path(evaluation.rooms)
        # Keep the count and UID order; full content hashing must detect this.
        rooms.write_text(rooms.read_text().replace('"uid": "d"', '"uid":\t"d"'))
        return ["text"] * len(prompts)
    session(monkeypatch, outputs=changed)
    with pytest.raises(ValueError, match="identity changed"):
        evaluator.run_resumable_evaluation(evaluation)
    assert not (Path(evaluation.out) / "completion.json").exists()


def test_score_missing_output_cannot_be_substituted_with_ground_truth(monkeypatch):
    monkeypatch.setattr(evaluator, "judge_v1_1", lambda room, placements: {"accepted": True})
    monkeypatch.setattr(evaluator, "gt_placements_v1_1", lambda room: {})
    row = evaluator.score_v1_1({"uid": "u", "source": "source", "group": "g", "objects": []},
                              None, "missing_generation")
    assert row["gt"]["accepted"]
    assert not row["accepted"] and not row["parsed"]


@pytest.mark.parametrize("kind", ["generated", "scored"])
def test_saved_chunk_payload_corruption_is_rejected(evaluation, monkeypatch, kind):
    session(monkeypatch, fail_on=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluator.run_resumable_evaluation(evaluation)
    chunk = Path(evaluation.out) / "chunks" / f"000000.{kind}.json"
    saved = json.loads(chunk.read_text())
    if kind == "generated":
        saved["outputs"][0] = "corrupted but still valid JSON"
        # Interrupted before scoring: no downstream digest can protect this payload.
        (chunk.parent / "000000.scored.json").unlink()
    else:
        saved["rows"][0]["accepted"] = False
    chunk.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="digest|integrity"):
        evaluator.run_resumable_evaluation(evaluation)


@pytest.mark.parametrize("kind", ["generated", "scored"])
def test_legacy_saved_chunks_without_payload_digest_are_rejected(evaluation, monkeypatch, kind):
    session(monkeypatch, fail_on=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluator.run_resumable_evaluation(evaluation)
    chunk = Path(evaluation.out) / "chunks" / f"000000.{kind}.json"
    saved = json.loads(chunk.read_text())
    saved.pop("content_sha256", None)
    chunk.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="digest|integrity"):
        evaluator.run_resumable_evaluation(evaluation)
