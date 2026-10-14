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
