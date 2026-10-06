"""Condition-only checkpoint inference, optionally followed by the asset loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from fastfill.v2.batch import load_tokenizer
from fastfill.v2.evaluate import predict_layout
from fastfill.v2.io import safe_output
from fastfill.v2.model import load_model
from fastfill.v2.schema import validate_condition, validate_layout


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--condition", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New JSON file, never overwrite input")
    p.add_argument("--baseline", choices=["structured", "text"], default="structured")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-asset-retries", type=int, default=2)
    p.add_argument("--max-repair-calls", type=int, default=0)
    p.add_argument("--repair-step-m", type=float, default=.25)
    p.add_argument("--max-seconds", type=float, default=10.)
    p.add_argument("--commit-in-memory", action="store_true")
    args = p.parse_args(argv)
    output = safe_output(args.output)
    condition = json.loads(args.condition.read_text())
    validate_condition(condition)
    if args.commit_in_memory and args.catalog is None:
        raise ValueError("commit requires an actual asset catalog")
    torch.set_num_threads(2)
    start = time.perf_counter()
    if args.baseline == "structured":
        model = load_model(args.checkpoint, device=args.device)
        tokenizer = load_tokenizer("tiny" if model.config.backbone == "tiny" else str(args.checkpoint.parent / "tokenizer"), local_files_only=True)
        inference_start = time.perf_counter()
        layout = predict_layout(model, tokenizer, condition, max_length=args.max_length, device=args.device)
    else:
        from fastfill.v2.text_sft import generate_text, load_text_model
        model, tokenizer = load_text_model(args.checkpoint, device=args.device)
        inference_start = time.perf_counter()
        layout = json.loads(generate_text(model, tokenizer, condition, max_length=args.max_length,
                                         max_new_tokens=args.max_new_tokens, device=args.device))
        validate_layout(layout, condition)
    inference_ms = (time.perf_counter() - inference_start) * 1000
    result = layout
    if args.catalog:
        from fastfill.v2.runtime import AtomicMemoryHost, BoundedTranslationRepair, RuntimeBudget, run_pipeline
        from fastfill.v2.serve import load_catalog
        host = AtomicMemoryHost() if args.commit_in_memory else None
        result = run_pipeline(condition, layout, load_catalog(args.catalog), host=host,
            budget=RuntimeBudget(max_asset_retries=args.max_asset_retries, max_repair_calls=args.max_repair_calls, max_seconds=args.max_seconds),
            repair=BoundedTranslationRepair(args.repair_step_m) if args.max_repair_calls else None,
            expected_world_version=0 if host else None, idempotency_key="cli-prediction")
        result = {**result, "fastfill_latency_ms": inference_ms,
                  "cold_start_total_ms": (time.perf_counter() - start) * 1000}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "requested_objects": len(condition["objects"]), "fastfill_latency_ms": inference_ms}))


if __name__ == "__main__":
    main()
