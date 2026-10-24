"""Fault-injection and compatibility checks for resumable DPO publication."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from fastfill.v1_1 import train_dpo


class Adapter:
    def save_pretrained(self, target, **kwargs):
        (Path(target) / "adapter_config.json").write_text("{}")
        save_file({"adapter": torch.ones(1)}, Path(target) / "adapter_model.safetensors")


def checkpoint(path, marker=1):
    path.mkdir(parents=True)
    Adapter().save_pretrained(path)
    torch.save({"optimizer": {}, "scheduler": {}, "state": {"epoch": 0, "it": 3, "step": marker}},
               path / "trainer_state.pt")
    return path


def save(path, step=2):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    train_dpo._save_dpo_ckpt(path, Adapter(), optimizer, scheduler, {"epoch": 0, "it": 7, "step": step})


def saved_step(path):
    return torch.load(path / "trainer_state.pt", weights_only=False)["state"]["step"]


def test_publish_failure_retains_previous_good_checkpoint(tmp_path):
    target = checkpoint(tmp_path / "ckpt_latest")
    replace = train_dpo.os.replace

    def fail_new_publication(source, destination):
        if ".tmp." in Path(source).name:
            raise OSError("injected rename failure")
        return replace(source, destination)

    with patch.object(train_dpo.os, "replace", side_effect=fail_new_publication):
        with pytest.raises(OSError, match="injected"):
            save(target)
    assert saved_step(target) == 1


def test_process_killed_between_renames_restores_previous_checkpoint(tmp_path):
    target = tmp_path / "ckpt_latest"
    previous = checkpoint(tmp_path / "ckpt_latest.previous")
    checkpoint(tmp_path / "ckpt_latest.tmp.123", marker=2)
    assert train_dpo.recover_dpo_checkpoint(target) == target
    assert saved_step(target) == 1
    assert not previous.exists()


def test_successful_publish_keeps_one_previous_generation(tmp_path):
    target = checkpoint(tmp_path / "ckpt_latest")
    save(target, 2)
    assert saved_step(target) == 2
    assert saved_step(tmp_path / "ckpt_latest.previous") == 1
    save(target, 3)
    assert saved_step(target) == 3
    assert saved_step(tmp_path / "ckpt_latest.previous") == 2


def test_legacy_checkpoint_is_resumable_without_format_migration(tmp_path):
    target = checkpoint(tmp_path / "ckpt_latest")
    assert train_dpo.recover_dpo_checkpoint(target) == target
    assert saved_step(target) == 1


def test_fresh_directory_has_no_checkpoint(tmp_path):
    assert train_dpo.recover_dpo_checkpoint(tmp_path / "ckpt_latest") is None


@pytest.mark.parametrize("missing", ["trainer_state.pt", "adapter_model.safetensors", "adapter_config.json"])
def test_partial_checkpoint_fails_instead_of_silently_restarting(tmp_path, missing):
    target = checkpoint(tmp_path / "ckpt_latest")
    (target / missing).unlink()
    with pytest.raises(ValueError, match="Incomplete DPO checkpoint"):
        train_dpo.recover_dpo_checkpoint(target)


def test_interrupted_first_save_requires_inspection(tmp_path):
    checkpoint(tmp_path / "ckpt_latest.tmp.123")
    with pytest.raises(ValueError, match="Interrupted initial"):
        train_dpo.recover_dpo_checkpoint(tmp_path / "ckpt_latest")


def test_corrupt_adapter_does_not_resume(tmp_path):
    target = checkpoint(tmp_path / "ckpt_latest")
    (target / "adapter_model.safetensors").write_bytes(b"broken")
    with pytest.raises(ValueError, match="Invalid DPO checkpoint"):
        train_dpo.recover_dpo_checkpoint(target)


@pytest.mark.parametrize("content", ["invalid json", "[]"])
def test_invalid_adapter_configuration_fails_closed(tmp_path, content):
    target = checkpoint(tmp_path / "ckpt_latest")
    (target / "adapter_config.json").write_text(content)
    with pytest.raises(ValueError, match="Invalid DPO checkpoint"):
        train_dpo.recover_dpo_checkpoint(target)


def test_failed_initial_publish_preserves_temporary_evidence(tmp_path):
    target = tmp_path / "ckpt_latest"
    with patch.object(train_dpo.os, "replace", side_effect=OSError("initial publication failure")):
        with pytest.raises(OSError):
            save(target)
    assert not target.exists()
    with pytest.raises(ValueError, match="Interrupted initial"):
        train_dpo.recover_dpo_checkpoint(target)


def test_old_backup_not_used_when_new_checkpoint_is_partial(tmp_path):
    checkpoint(tmp_path / "ckpt_latest.previous")
    target = checkpoint(tmp_path / "ckpt_latest", marker=2)
    (target / "trainer_state.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete DPO checkpoint"):
        train_dpo.recover_dpo_checkpoint(target)


@pytest.mark.parametrize("state", [
    {"epoch": 0, "it": 3, "step": 1},
    {"epoch": 0, "it": 6, "step": 2},
    {"epoch": 1, "it": 3, "step": 3},
    {"epoch": 1, "it": 6, "step": 4},
])
def test_valid_full_and_partial_window_checkpoint_positions(state):
    assert train_dpo.validate_resume_position(state, batches=7, epochs=2, grad_accum=4) == state


@pytest.mark.parametrize("state", [
    {}, {"epoch": 2, "it": 6, "step": 4}, {"epoch": -1, "it": 3, "step": 1},
    {"epoch": 0, "it": 7, "step": 2}, {"epoch": 0, "it": 2, "step": 1},
    {"epoch": 1, "it": 3, "step": 2}, {"epoch": True, "it": 3, "step": 3},
    {"epoch": 1, "it": 3, "step": "3"},
])
def test_invalid_checkpoint_position_cannot_skip_training(state):
    with pytest.raises(ValueError, match="checkpoint position"):
        train_dpo.validate_resume_position(state, batches=7, epochs=2, grad_accum=4)


@pytest.fixture
def tiny_training_inputs(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    backend = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "good": 2, "bad": 3, "prompt": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")
    tokenizer.chat_template = "prompt "
    base, out, pairs = tmp_path / "base", tmp_path / "run", tmp_path / "pairs.jsonl"
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=5, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                        head_dim=8, max_position_embeddings=32, eos_token_id=1))
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    pair = {"prompt": [{"role": "user", "content": "prompt"}], "chosen": "good", "rejected": "bad"}
    pairs.write_text("".join(json.dumps(pair) + "\n" for _ in range(4)))
    monkeypatch.setattr(sys, "argv", ["train_dpo", "--model", str(base), "--pairs", str(pairs), "--out", str(out),
                                    "--epochs", "1", "--grad_accum", "2", "--save_steps", "1", "--max_len", "32",
                                    "--lora_r", "2", "--lora_alpha", "4"])
    return out


@pytest.fixture
def tiny_training_run(tiny_training_inputs):
    train_dpo.main()
    return tiny_training_inputs


def test_nonfinite_gradient_fails_before_optimizer_or_final_publication(tiny_training_inputs, monkeypatch):
    make_peft = train_dpo.get_peft_model

    def with_invalid_gradient(*args, **kwargs):
        model = make_peft(*args, **kwargs)
        next(p for p in model.parameters() if p.requires_grad).register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return model

    def forbidden_update(*args, **kwargs):
        pytest.fail("Non-finite gradients must fail before optimizer update")

    monkeypatch.setattr(train_dpo, "get_peft_model", with_invalid_gradient)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_update)
    with pytest.raises(RuntimeError, match="non-finite"):
        train_dpo.main()
    assert not (tiny_training_inputs / "final").exists()
    assert not (tiny_training_inputs / "ckpt_latest").exists()


def test_real_cpu_train_and_complete_checkpoint_resume(tiny_training_run, capsys):
    from safetensors.torch import load_file
    out = tiny_training_run
    before = load_file(out / "final/adapter_model.safetensors")
    assert saved_step(out / "ckpt_latest") == 2
    train_dpo.main()
    after = load_file(out / "final/adapter_model.safetensors")
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert "step=2/2" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["pairs", "model", "recipe"])
def test_real_cpu_resume_rejects_changed_inputs_before_rewriting_manifest(tiny_training_run, monkeypatch, change):
    out = tiny_training_run
    manifest = out / "run_manifest.json"
    before = manifest.read_bytes()
    args = json.loads(before)["args"]
    if change == "pairs":
        path = Path(args["pairs"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps({**r, "chosen": r["rejected"], "rejected": r["chosen"]}) + "\n" for r in rows))
    elif change == "model":
        path = Path(args["model"]) / "model.safetensors"
        from safetensors.torch import load_file
        weights = load_file(path)
        key = next(iter(weights))
        weights[key] = weights[key] + .1
        save_file(weights, path)
    else:
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--beta", "0.2"])
    with pytest.raises(ValueError, match="[Ii]dentity"):
        train_dpo.main()
    assert manifest.read_bytes() == before


def test_unbound_legacy_checkpoint_requires_explicit_provenance(tiny_training_run):
    out = tiny_training_run
    identity = out / "trainer_identity.json"
    if identity.exists():
        identity.unlink()  # Simulate the historical direct CLI, which did not write a guard.
    with pytest.raises(ValueError, match="unbound"):
        train_dpo.main()


def _guarded_legacy_run(out):
    """Build a real tiny run with the historical external guard and frozen trainer copy."""
    from fastfill.v1_1.training_guard import guard_training

    pairs, base = out.parent / "pairs.jsonl", out.parent / "base"
    pairs.write_text("".join(json.dumps({**json.loads(line), "uid": f"row-{i}"}) + "\n"
                             for i, line in enumerate(pairs.read_text().splitlines())))
    frozen = out.parent / "frozen_train_dpo.py"
    frozen.write_bytes(Path(train_dpo.__file__).read_bytes())
    guard_training(base, pairs, out, {"world_size": 1, "epochs": 1, "grad_accum": 2, "max_len": 32,
                                    "lora_r": 2, "lora_alpha": 4, "save_steps": 1,
                                    "loss": "historical DPO", "dtype": "float32"}, [frozen])
    train_dpo.main()
    identity = out / "trainer_identity.json"
    if identity.exists():
        identity.unlink()
    return frozen


def test_guarded_legacy_resume_retains_guard_and_run_manifest(tiny_training_inputs, capsys):
    out = tiny_training_inputs
    _guarded_legacy_run(out)
    originals = {name: (out / name).read_bytes() for name in ("training_guard.json", "run_manifest.json")}
    train_dpo.main()
    assert (out / "trainer_identity.json").is_file()
    assert all((out / name).read_bytes() == value for name, value in originals.items())
    assert "step=2/2" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["pairs", "recipe", "frozen_code", "guard_settings", "manifest_type"])
def test_guarded_legacy_resume_checks_original_provenance(tiny_training_inputs, monkeypatch, change):
    out = tiny_training_inputs
    frozen = _guarded_legacy_run(out)
    if change == "pairs":
        path = out.parent / "pairs.jsonl"
        path.write_text(path.read_text().replace('"good"', '"bad"'))
    elif change == "recipe":
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--beta", "0.2"])
    elif change == "frozen_code":
        frozen.write_text(frozen.read_text() + "\n# changed\n")
    elif change == "guard_settings":
        path = out / "training_guard.json"
        guard = json.loads(path.read_text())
        guard["settings"]["world_size"] = 2
        path.write_text(json.dumps(guard))
    else:
        (out / "run_manifest.json").write_text("[]")
    before = (out / "run_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="[Ii]dentity"):
        train_dpo.main()
    assert not (out / "trainer_identity.json").exists()
    assert (out / "run_manifest.json").read_bytes() == before


@pytest.mark.parametrize("change", ["world_size", "init_adapter", "audit", "code"])
def test_direct_trainer_binding_rejects_other_resume_drift(tiny_training_inputs, change):
    from fastfill.v1_1.training_guard import bind_trainer_identity

    out = tiny_training_inputs
    initial = out.parent / "initial"
    initial.mkdir()
    Adapter().save_pretrained(initial)
    code = out.parent / "code.py"
    code.write_text("# trainer snapshot\n")
    audit = out.parent / "audit.json"
    audit.write_text('{"version":1}')
    parser = train_dpo.argument_parser()
    args = parser.parse_args([*sys.argv[1:], "--init_adapter", str(initial)])
    defaults = {k: parser.get_default(k) for k in vars(args)}
    bind_trainer_identity(args, defaults, 1, "float32", [code])
    before = (out / "trainer_identity.json").read_bytes()
    world = 1
    if change == "world_size":
        world = 2
    elif change == "init_adapter":
        save_file({"adapter": torch.zeros(1)}, initial / "adapter_model.safetensors")
    elif change == "audit":
        audit.write_text('{"version":2}')
    else:
        code.write_text("# changed snapshot\n")
    with pytest.raises(ValueError, match="[Ii]dentity"):
        bind_trainer_identity(args, defaults, world, "float32", [code])
    assert (out / "trainer_identity.json").read_bytes() == before


def test_guarded_initial_adapter_and_explicit_default_warmup_are_compatible(tiny_training_inputs):
    import hashlib
    from fastfill.v1_1.training_guard import bind_trainer_identity, guard_training
    from fastfill.v1_1.evaluation_protocol import file_identity

    out = tiny_training_inputs
    pairs, base = out.parent / "pairs.jsonl", out.parent / "base"
    pairs.write_text("".join(json.dumps({**json.loads(line), "uid": str(i)}) + "\n"
                             for i, line in enumerate(pairs.read_text().splitlines())))
    initial = out.parent / "initial"
    initial.mkdir()
    Adapter().save_pretrained(initial)
    code = out.parent / "train.py"
    code.write_text("# original trainer\n")
    parser = train_dpo.argument_parser()
    args = parser.parse_args([*sys.argv[1:], "--init_adapter", str(initial), "--warmup_ratio", "0.05"])
    defaults = {k: parser.get_default(k) for k in vars(args)}
    files = {p.name: file_identity(p)["sha256"] for p in initial.iterdir()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, allow_nan=False).encode()).hexdigest()
    settings = {"world_size": 1, "epochs": 1, "grad_accum": 2, "max_len": 32, "lora_r": 2,
                "lora_alpha": 4, "save_steps": 1,
                "init_adapter": {"path": str(initial), "sha256": digest, "of": "rpo-dpo1"}}
    guard_training(base, pairs, out, settings, [code])
    original = (out / "training_guard.json").read_bytes()
    bind_trainer_identity(args, defaults, 1, "float32", [code])
    args.warmup_ratio = None
    bind_trainer_identity(args, defaults, 1, "float32", [code])
    assert (out / "training_guard.json").read_bytes() == original


def test_real_cpu_final_receipt_binds_actual_completion_and_artifacts(tiny_training_run):
    from fastfill.v1_1.evaluation_protocol import file_identity

    out = tiny_training_run
    receipt = json.loads((out / "final/completion.json").read_text())
    assert receipt["schema_version"] == 1
    assert receipt["kind"] == "fastfill_v1.1_dpo_completion"
    assert receipt["training_guard"] is None
    assert receipt["admitted_pairs"] == receipt["loader_batches"] == 4
    assert receipt["world_size"] == receipt["epochs"] == 1
    assert receipt["position"] == {"epoch": 0, "it": 3, "step": 2}
    assert receipt["expected_total_steps"] == 2
    assert set(receipt["adapter_files"]) == {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "tokenizer.json"}
    for name, identity in receipt["adapter_files"].items():
        assert identity == file_identity(out / "final" / name)


def test_final_receipt_binds_guard_and_refuses_unfinished_training(tiny_training_run):
    from fastfill.v1_1.evaluation_protocol import file_identity

    out = tiny_training_run
    guard = out / "training_guard.json"
    guard.write_text('{"fixture": true}')
    train_dpo.write_dpo_completion(out / "final", out, 4, 4, 1, 1, 2, 2)
    receipt = json.loads((out / "final/completion.json").read_text())
    assert receipt["training_guard"] == file_identity(guard)
    with pytest.raises(ValueError, match="incomplete DPO training"):
        train_dpo.write_dpo_completion(out / "final", out, 4, 4, 1, 1, 1, 2)


def test_real_cpu_resume_rejects_inconsistent_checkpoint_steps(tiny_training_run):
    path = tiny_training_run / "ckpt_latest/trainer_state.pt"
    state = torch.load(path, weights_only=False)
    torch.save({**state, "state": {**state["state"], "step": 999}}, path)
    with pytest.raises(ValueError, match="checkpoint position"):
        train_dpo.main()


def test_real_cpu_resume_rejects_unreadable_trainer_state(tiny_training_run):
    path = tiny_training_run / "ckpt_latest/trainer_state.pt"
    path.write_bytes(b"corrupt checkpoint state")
    with pytest.raises(Exception):
        train_dpo.main()
