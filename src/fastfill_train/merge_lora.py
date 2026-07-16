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
import json
from pathlib import Path


def check_adapter_lineage(base: str, lora: str) -> str | None:
    """Return an error message when the adapter was trained on another base.

    Compares ``--base`` against the adapter's recorded
    ``base_model_name_or_path``. peft only errors on architecture mismatch,
    so merging a DPO adapter onto the wrong same-architecture base (e.g. raw
    Qwen3-8B instead of the merged SFT mainline) would otherwise succeed
    silently and produce a broken model.
    """
    config_path = Path(lora) / "adapter_config.json"
    if not config_path.exists():
        return f"adapter config not found: {config_path}"
    recorded = json.loads(config_path.read_text(encoding="utf-8")).get(
        "base_model_name_or_path"
    )
    if not recorded:
        return (
            f"adapter config {config_path} records no base_model_name_or_path "
            "— lineage cannot be verified"
        )

    def normalize(path: str) -> str:
        return str(Path(path).resolve()) if Path(path).exists() else path

    if normalize(recorded) != normalize(base):
        return (
            f"adapter {lora} was trained on base '{recorded}' but --base is "
            f"'{base}'; pass the recorded base, or --allow-base-mismatch to "
            "override deliberately"
        )
    return None


def merge_lora(base: str, lora: str, out: str) -> None:
    """Load base + adapter, ``merge_and_unload``, save model AND tokenizer.

    The merged dir also gets a combined ``DATA_PROVENANCE.json`` — merged
    weights carry BOTH lineages, and downstream DPO verifies its pairs
    against the snapshot ids recorded here.
    """
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
    _write_merged_provenance(base, lora, out)


def _load_provenance(model_dir: str) -> dict | None:
    path = Path(model_dir) / "DATA_PROVENANCE.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_merged_provenance(base: str, lora: str, out: str) -> None:
    from fastfill_train.provenance_guard import _walk_values, lineage_problem

    base_prov = _load_provenance(base)
    adapter_prov = _load_provenance(lora)
    # A merged model is verified only when the adapter carries acceptable
    # lineage AND neither side records verified=false — a smoke merge
    # (--allow-missing-provenance) must stay verified=false forever so the
    # DPO base guard keeps rejecting it in production.
    verified = (
        adapter_prov is not None
        and lineage_problem(adapter_prov) is None
        and (
            base_prov is None
            or False not in _walk_values(base_prov, "verified")
        )
    )
    lineage = {
        "verified": verified,
        "merged_base": base,
        "merged_adapter": lora,
        "base_provenance": base_prov,
        "adapter_provenance": adapter_prov,
    }
    (Path(out) / "DATA_PROVENANCE.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model (bf16)"
    )
    parser.add_argument("--base", required=True, help="base model dir or HF id")
    parser.add_argument("--lora", required=True, help="adapter dir (trainer output)")
    parser.add_argument("--out", required=True, help="merged model output dir")
    parser.add_argument(
        "--allow-base-mismatch",
        action="store_true",
        help="merge even when the adapter's recorded base differs from --base",
    )
    parser.add_argument(
        "--allow-missing-provenance",
        action="store_true",
        help=(
            "smoke only: merge an adapter that has no DATA_PROVENANCE.json "
            "(the merged model would otherwise carry an unverifiable side)"
        ),
    )
    args = parser.parse_args(argv)
    lineage_error = check_adapter_lineage(args.base, args.lora)
    if lineage_error is not None:
        if args.allow_base_mismatch:
            print(f"WARNING: {lineage_error} (continuing: --allow-base-mismatch)")
        else:
            raise SystemExit(f"ERROR: {lineage_error}")
    if not (Path(args.lora) / "DATA_PROVENANCE.json").exists():
        adapter_problem: str | None = "no DATA_PROVENANCE.json"
    else:
        from fastfill_train.provenance_guard import lineage_problem

        adapter_problem = lineage_problem(_load_provenance(args.lora))
    if adapter_problem is not None:
        if args.allow_missing_provenance:
            print(
                f"WARNING: adapter lineage unacceptable ({adapter_problem}) "
                "— merged model will be marked verified=false (smoke only)"
            )
        else:
            raise SystemExit(
                f"ERROR: adapter {args.lora} lineage unacceptable "
                f"({adapter_problem}) — a merged production model must carry "
                "BOTH verifiable lineages; retrain with the current tooling, "
                "or pass --allow-missing-provenance for smoke"
            )
    merge_lora(args.base, args.lora, args.out)
    print(f"merged {args.base} + {args.lora} -> {args.out}")


if __name__ == "__main__":
    main()
