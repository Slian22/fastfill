"""Config loading + CLI mapping tests — NO torch/trl/peft required.

Covers: loading every YAML config, paper-value assertions, ``--set`` override
behavior (incl. nested ``lora.r``), the pure ``build_sft_config`` /
``build_dpo_config`` kwargs mappings, and the ``dump_config`` roundtrip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import fastfill_train  # noqa: F401,E402  (vendor path bootstrap)
from fastfill_train.config import TrainConfig, dump_config, load_config  # noqa: E402
from fastfill_train.train_dpo import build_dpo_config  # noqa: E402
from fastfill_train.train_sft import build_arg_parser, build_sft_config  # noqa: E402

CONFIGS = REPO / "configs"
ALL_CONFIG_FILES = tuple(path.name for path in sorted(CONFIGS.glob("*.yaml")))


# --- config loading (paper values) -----------------------------------------


@pytest.mark.parametrize("name", ALL_CONFIG_FILES)
def test_all_configs_load(name: str) -> None:
    cfg = load_config(CONFIGS / name)
    assert isinstance(cfg, TrainConfig)
    assert cfg.dataset_files  # every config names its data
    # Every checked-in training file consumes a seed-42 frozen snapshot.
    # Reusing 42 makes its 5% val bucket a subset already removed from train.
    assert cfg.seed == 43


def test_sft_full_matches_paper() -> None:
    cfg = load_config(CONFIGS / "sft_full.yaml")
    assert cfg.model_name_or_path == "Qwen/Qwen3-8B"
    assert cfg.learning_rate == pytest.approx(5e-6)
    assert cfg.epochs == 10
    assert cfg.max_steps == -1  # full run: epochs, not a step cap
    assert (cfg.lora.r, cfg.lora.alpha) == (16, 32)
    assert cfg.lora.dropout == pytest.approx(0.05)


def test_dpo_stage1_matches_paper() -> None:
    cfg = load_config(CONFIGS / "dpo_stage1.yaml")
    assert cfg.model_name_or_path == "out/sft_full_merged"
    assert cfg.learning_rate == pytest.approx(5e-7)  # paper, NOT README 5e-6
    assert cfg.epochs == 5  # paper, NOT README 10
    assert (cfg.lora.r, cfg.lora.alpha) == (16, 32)  # paper, NOT README r32/a16
    assert cfg.dpo_beta == pytest.approx(0.1)
    assert cfg.max_length == 3200
    assert cfg.max_prompt_length == 2048


def test_dpo_stage2_chains_from_stage1() -> None:
    cfg = load_config(CONFIGS / "dpo_stage2.yaml")
    assert cfg.model_name_or_path == "out/dpo_stage1_merged"
    assert cfg.learning_rate == pytest.approx(5e-7)
    assert cfg.epochs == 5


def test_sft_smoke_is_step_capped() -> None:
    cfg = load_config(CONFIGS / "sft_smoke.yaml")
    assert cfg.max_steps == 100
    assert cfg.template == "direct"


def test_deepspeed_zero2_contract() -> None:
    raw = json.loads((CONFIGS / "ds_zero2.json").read_text(encoding="utf-8"))
    assert raw["bf16"]["enabled"] is True
    assert raw["zero_optimization"]["stage"] == 2
    assert raw["gradient_accumulation_steps"] == "auto"
    assert raw["train_micro_batch_size_per_gpu"] == "auto"
    for name in ("sft_full_fp.yaml", "stage0_full.yaml", "full_fp.yaml"):
        cfg = load_config(CONFIGS / name)
        assert cfg.use_lora is False
        assert cfg.deepspeed == "configs/ds_zero2.json"


# --- --set overrides --------------------------------------------------------


def test_set_overrides_scalar_and_nested_lora_r() -> None:
    # NOTE: YAML 1.1 floats need a dotted mantissa — write 1.0e-4, not 1e-4
    # (the latter parses as the STRING "1e-4"; pinned below).
    cfg = load_config(
        CONFIGS / "sft_full.yaml", ["learning_rate=1.0e-4", "lora.r=8"]
    )
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.lora.r == 8
    assert cfg.lora.alpha == 32  # sibling fields untouched
    # source file unaffected (immutable copies, not in-place edits)
    assert load_config(CONFIGS / "sft_full.yaml").lora.r == 16


def test_set_override_list_becomes_tuple() -> None:
    cfg = load_config(
        CONFIGS / "sft_full.yaml", ["dataset_files=[a.jsonl, b.jsonl]"]
    )
    assert cfg.dataset_files == ("a.jsonl", "b.jsonl")


def test_set_override_rejects_unknown_key_and_bad_format() -> None:
    with pytest.raises(KeyError):
        load_config(CONFIGS / "sft_full.yaml", ["no_such_key=1"])
    with pytest.raises(ValueError):
        load_config(CONFIGS / "sft_full.yaml", ["not_key_value"])


def test_set_override_yaml_float_quirk_is_pinned() -> None:
    # yaml.safe_load("1e-4") is a STRING under YAML 1.1 — this pins the
    # sharp edge so nobody "fixes" a run by passing the bare form.
    cfg = load_config(CONFIGS / "sft_full.yaml", ["learning_rate=1e-4"])
    assert cfg.learning_rate == "1e-4"  # would break training; use 1.0e-4


def test_cli_parser_collects_repeated_set_flags() -> None:
    parser = build_arg_parser("test")
    ns = parser.parse_args(
        ["--config", "c.yaml", "--set", "lora.r=8", "--set", "epochs=1"]
    )
    assert ns.config == "c.yaml"
    assert ns.overrides == ["lora.r=8", "epochs=1"]


# --- build_sft_config / build_dpo_config mappings ---------------------------


def test_sft_config_max_steps_beats_epochs() -> None:
    smoke = load_config(CONFIGS / "sft_smoke.yaml")
    kwargs = build_sft_config(smoke, has_eval=True)
    assert kwargs["max_steps"] == 100
    assert "num_train_epochs" not in kwargs

    full = load_config(CONFIGS / "sft_full.yaml")
    kwargs = build_sft_config(full, has_eval=True)
    assert kwargs["num_train_epochs"] == 10
    assert "max_steps" not in kwargs


def test_sft_config_eval_off_without_val_split() -> None:
    cfg = load_config(CONFIGS / "sft_full.yaml")
    with_eval = build_sft_config(cfg, has_eval=True)
    assert with_eval["eval_strategy"] == "steps"
    assert with_eval["eval_steps"] == cfg.eval_steps
    without = build_sft_config(cfg, has_eval=False)
    assert without["eval_strategy"] == "no"
    assert "eval_steps" not in without


def test_sft_config_core_field_mapping() -> None:
    cfg = load_config(CONFIGS / "sft_full.yaml")
    kwargs = build_sft_config(cfg, has_eval=True)
    assert kwargs["output_dir"] == "out/sft_full"
    assert kwargs["learning_rate"] == pytest.approx(5e-6)
    assert kwargs["max_length"] == cfg.max_seq_length  # TRL>=0.20 name
    assert kwargs["per_device_train_batch_size"] == 4
    assert kwargs["gradient_accumulation_steps"] == 2
    assert kwargs["bf16"] is True
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["report_to"] == "none"
    assert kwargs["seed"] == cfg.seed


def test_dpo_config_mapping_paper_values() -> None:
    cfg = load_config(CONFIGS / "dpo_stage1.yaml")
    kwargs = build_dpo_config(cfg, has_eval=True)
    assert kwargs["beta"] == pytest.approx(0.1)
    assert kwargs["max_length"] == 3200
    assert kwargs["max_prompt_length"] == 2048
    assert kwargs["learning_rate"] == pytest.approx(5e-7)
    assert kwargs["num_train_epochs"] == 5
    assert "max_steps" not in kwargs
    assert kwargs["per_device_train_batch_size"] == 1
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["bf16"] is True
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["remove_unused_columns"] is False
    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["eval_steps"] == 100
    # Optimizer fields — must stay in sync with build_sft_config; TRL DPO
    # defaults differ (e.g. warmup_ratio=0, weight_decay=0).
    assert kwargs["warmup_ratio"] == pytest.approx(cfg.warmup_ratio)
    assert kwargs["lr_scheduler_type"] == cfg.lr_scheduler_type
    assert kwargs["weight_decay"] == pytest.approx(cfg.weight_decay)
    assert kwargs["max_grad_norm"] == pytest.approx(cfg.max_grad_norm)


def test_dpo_config_eval_off_and_step_cap() -> None:
    cfg = load_config(CONFIGS / "dpo_stage1.yaml", ["max_steps=7"])
    kwargs = build_dpo_config(cfg, has_eval=False)
    assert kwargs["eval_strategy"] == "no"
    assert "eval_steps" not in kwargs
    assert kwargs["max_steps"] == 7
    assert "num_train_epochs" not in kwargs


# --- dump_config roundtrip ---------------------------------------------------


def test_dump_config_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "sft_full.yaml", ["lora.r=8", "epochs=2"])
    dumped = tmp_path / "resolved_config.yaml"
    dump_config(cfg, dumped)
    assert load_config(dumped) == cfg


# --- lazy heavy deps ----------------------------------------------------------


def test_train_modules_import_without_heavy_deps() -> None:
    # The imports at the top of this file already exercised train_sft /
    # train_dpo on a machine without torch; this pins the invariant.
    for mod in ("torch", "trl", "peft", "datasets", "transformers"):
        assert mod not in sys.modules
