"""Merge a LoRA adapter into its base model (bf16, transformers + peft).

Ported from OptiScene ``scripts/merge_lora.py`` with transformers instead
of modelscope. The merged directory is a plain HF model dir, servable via
``scripts/serve_vllm.sh`` and usable as ``model_name_or_path`` for the DPO
stages. Invocation:

    PYTHONPATH=src python3 -m fastfill_train.merge_lora \
        --base Qwen/Qwen3-8B --lora out/sft_smoke --out out/sft_smoke_merged
"""

from __future__ import annotations

import fastfill_train  # noqa: F401  (vendor path bootstrap — keep first)

import argparse


def merge_lora(base: str, lora: str, out: str) -> None:
    """Load base + adapter, ``merge_and_unload``, save model AND tokenizer."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    lora_model = PeftModel.from_pretrained(
        base_model, lora, torch_dtype=torch.bfloat16
    )
    merged = lora_model.merge_and_unload()
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model (bf16)"
    )
    parser.add_argument("--base", required=True, help="base model dir or HF id")
    parser.add_argument("--lora", required=True, help="adapter dir (trainer output)")
    parser.add_argument("--out", required=True, help="merged model output dir")
    args = parser.parse_args(argv)
    merge_lora(args.base, args.lora, args.out)
    print(f"merged {args.base} + {args.lora} -> {args.out}")


if __name__ == "__main__":
    main()
