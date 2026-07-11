"""DPO entrypoint for the FastFill planner (TRL DPOTrainer + LoRA).

Invocation:

    PYTHONPATH=src python3 -m fastfill_train.train_dpo \
        --config configs/dpo_stage1.yaml [--set key=value ...]

``cfg.model_name_or_path`` must point at a MERGED SFT directory (see
``fastfill_train.merge_lora``); a fresh LoRA adapter is trained on top of
it. With PEFT active, ``ref_model=None`` — TRL uses the base weights with
adapters disabled as the implicit reference model (same as OptiScene's
DPO script). Heavy deps are imported lazily so :func:`build_dpo_config`
stays testable without trl/torch installed.
"""

from __future__ import annotations

import fastfill_train  # noqa: F401  (vendor path bootstrap — keep first)

from pathlib import Path
from typing import Any

from fastfill_train.config import TrainConfig, dump_config, load_config
from fastfill_train.data import load_dpo_datasets
from fastfill_train.train_sft import (
    RESOLVED_CONFIG_NAME,
    build_arg_parser,
    build_lora_config,
    load_model_and_tokenizer,
)


def build_dpo_config(cfg: TrainConfig, *, has_eval: bool) -> dict[str, Any]:
    """Kwargs for ``trl.DPOConfig`` — pure mapping, NO trl import.

    Same scheduling rules as SFT: ``max_steps`` (when > 0) beats
    ``num_train_epochs``; step-schedule eval only when a validation split
    exists. ``remove_unused_columns=False`` keeps the prompt/chosen/
    rejected columns for DPOTrainer's own preprocessing.
    """
    kwargs: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.learning_rate,
        "beta": cfg.dpo_beta,
        "max_length": cfg.max_length,
        "max_prompt_length": cfg.max_prompt_length,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_strategy": cfg.save_strategy,
        "report_to": cfg.report_to,
        "seed": cfg.seed,
        "remove_unused_columns": False,
    }
    if cfg.gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if cfg.max_steps > 0:
        kwargs["max_steps"] = cfg.max_steps
    else:
        kwargs["num_train_epochs"] = cfg.epochs
    kwargs["eval_strategy"] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = cfg.eval_steps
    return kwargs


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("FastFill DPO training (LoRA)").parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    dump_config(cfg, Path(cfg.output_dir) / RESOLVED_CONFIG_NAME)
    train_dataset, eval_dataset = load_dpo_datasets(
        cfg.dataset_files, cfg.val_fraction, cfg.seed
    )

    from trl import DPOConfig, DPOTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    dpo_args = DPOConfig(**build_dpo_config(cfg, has_eval=eval_dataset is not None))
    trainer = DPOTrainer(
        model,
        ref_model=None,  # PEFT active: adapters-off base acts as reference
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"DPO adapter saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
