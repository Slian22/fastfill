"""Training configuration: one dataclass, loaded from YAML + CLI overrides.

Every entrypoint (SFT / DPO / eval) takes ``--config configs/<x>.yaml`` plus
optional ``--set key=value`` overrides (dotted keys for nested fields), so a
run is always reproducible from a single file + explicit deltas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class TrainConfig:
    """Shared config for SFT and DPO runs (unused fields are ignored)."""

    # model / data
    model_name_or_path: str = "Qwen/Qwen3-8B"
    dataset_files: tuple[str, ...] = ()
    template: str = "direct"  # "direct" (A) | "plan" (B)
    output_dir: str = "out/run"
    cache_dir: str | None = None

    # split (house-first: samples sharing split_key never straddle splits)
    val_fraction: float = 0.05
    seed: int = 42

    # optimization
    learning_rate: float = 5e-6
    epochs: float = 1.0
    max_steps: int = -1  # >0 = smoke cap, overrides epochs
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_lora: bool = True  # False = full-parameter fine-tune (80G-class GPUs)
    deepspeed: str | None = None  # e.g. configs/ds_zero2.json for full-param

    # DPO-only
    dpo_beta: float = 0.1
    max_prompt_length: int = 3200
    max_length: int = 3200

    # bookkeeping
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_strategy: str = "steps"
    report_to: str = "none"  # "none" | "wandb" | "swanlab"
    lora: LoraSettings = field(default_factory=LoraSettings)


def _apply_override(cfg: Any, dotted_key: str, raw_value: str) -> Any:
    """Return a copy of ``cfg`` with ``a.b=c`` applied (YAML-parsed value)."""
    value = yaml.safe_load(raw_value)
    head, _, rest = dotted_key.partition(".")
    if not hasattr(cfg, head):
        raise KeyError(f"unknown config key: {dotted_key}")
    current = getattr(cfg, head)
    if rest:
        return replace(cfg, **{head: _apply_override(current, rest, raw_value)})
    if isinstance(current, tuple) and isinstance(value, list):
        value = tuple(value)
    return replace(cfg, **{head: value})


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    """Load a YAML config file and apply ``key=value`` overrides in order."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    lora = LoraSettings(
        **{
            **asdict(LoraSettings()),
            **{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in (data.pop("lora", {}) or {}).items()
            },
        }
    )
    for key, value in list(data.items()):
        if isinstance(value, list):
            data[key] = tuple(value)
    cfg = TrainConfig(lora=lora, **data)
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _ or not key:
            raise ValueError(f"override must be key=value, got: {item!r}")
        cfg = _apply_override(cfg, key.strip(), value.strip())
    return cfg


def dump_config(cfg: TrainConfig, path: str | Path) -> None:
    """Persist the resolved config next to the run outputs."""

    def _plain(obj: Any) -> Any:
        if is_dataclass(obj):
            return {k: _plain(v) for k, v in asdict(obj).items()}
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items()}
        return obj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(_plain(cfg), sort_keys=False))
