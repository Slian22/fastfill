"""SFT entrypoint for the FastFill planner (TRL SFTTrainer + LoRA).

Invocation (every entrypoint follows this shape):

    PYTHONPATH=src python3 -m fastfill_train.train_sft \
        --config configs/sft_smoke.yaml [--set key=value ...]

Heavy deps (torch / transformers / trl / peft / datasets) are imported
LAZILY inside functions so this module stays importable — and its pure
config mapping (:func:`build_sft_config`) testable — on machines without
the training stack. ``train_dpo`` reuses the model/tokenizer/LoRA helpers
defined here.
"""

from __future__ import annotations

import fastfill_train  # noqa: F401  (vendor path bootstrap — keep first)

import argparse
import functools
from pathlib import Path
from typing import Any

from fastfill_train.config import TrainConfig, dump_config, load_config
from fastfill_train.data import load_sft_datasets
from fastfill_train.provenance_guard import verify_training_data, write_provenance

RESOLVED_CONFIG_NAME = "resolved_config.yaml"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Shared ``--config`` / ``--set`` CLI for all training entrypoints."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --set lora.r=8 (repeatable)",
    )
    return parser


def build_sft_config(cfg: TrainConfig, *, has_eval: bool) -> dict[str, Any]:
    """Kwargs for ``trl.SFTConfig`` — pure mapping, NO trl import.

    Rules: ``max_steps`` (when > 0) beats ``num_train_epochs`` (smoke cap);
    eval runs on a step schedule only when a validation split exists.
    ``max_length`` is TRL >= 0.20's name for the OptiScene-era
    ``max_seq_length``.
    """
    kwargs: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_length": cfg.max_seq_length,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_strategy": cfg.save_strategy,
        "report_to": cfg.report_to,
        "seed": cfg.seed,
    }
    if cfg.save_total_limit is not None:
        kwargs["save_total_limit"] = cfg.save_total_limit
    if cfg.gradient_checkpointing:
        # Non-reentrant checkpointing: required for LoRA (frozen inputs
        # otherwise break the autograd graph under the reentrant variant).
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    kwargs["completion_only_loss"] = True  # prompt must never enter the loss
    if cfg.max_steps > 0:
        kwargs["max_steps"] = cfg.max_steps
    else:
        kwargs["num_train_epochs"] = cfg.epochs
    kwargs["eval_strategy"] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = cfg.eval_steps
    if cfg.deepspeed:
        kwargs["deepspeed"] = cfg.deepspeed
    return kwargs


def build_lora_config(cfg: TrainConfig) -> Any:
    """``peft.LoraConfig`` from ``cfg.lora`` (lazy peft import)."""
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )


def _prefer_no_thinking(tokenizer: Any) -> None:
    """Pin ``enable_thinking=False`` on Qwen3-style chat templates.

    Qwen3 tokenizers accept ``enable_thinking`` in ``apply_chat_template``;
    FastFill is token-budgeted (no chain of thought), so when the kwarg is
    supported we pin it to False for every render, including TRL's internal
    calls. Templates that ignore the kwarg are unaffected; tokenizers that
    reject it (or have no chat template) keep their original method — the
    probe call below guards that case.
    """
    probe = [{"role": "user", "content": "ping"}]
    try:
        tokenizer.apply_chat_template(probe, tokenize=False, enable_thinking=False)
    except Exception:  # kwarg rejected / no chat template -> leave untouched
        return
    tokenizer.apply_chat_template = functools.partial(
        tokenizer.apply_chat_template, enable_thinking=False
    )


def load_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    """AutoModelForCausalLM + AutoTokenizer (transformers, NOT modelscope)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else None,
        cache_dir=cfg.cache_dir,
        trust_remote_code=False,
        use_cache=not cfg.gradient_checkpointing,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, cache_dir=cfg.cache_dir, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _prefer_no_thinking(tokenizer)
    return model, tokenizer


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("FastFill SFT training (LoRA)").parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    provenance = verify_training_data(cfg)  # fail-closed BEFORE any setup
    dump_config(cfg, Path(cfg.output_dir) / RESOLVED_CONFIG_NAME)
    write_provenance(cfg, provenance)
    train_dataset, eval_dataset = load_sft_datasets(
        cfg.dataset_files, cfg.template, cfg.val_fraction, cfg.seed
    )

    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    sft_args = SFTConfig(**build_sft_config(cfg, has_eval=eval_dataset is not None))
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg) if cfg.use_lora else None,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint or None)
    trainer.save_model(cfg.output_dir)
    print(f"SFT adapter saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
