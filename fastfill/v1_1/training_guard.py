"""Bind an isolated DPO run to the exact inputs before starting or resuming."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from fastfill.v1_1.evaluation_protocol import (
    _validate_safetensors, dataset_identity, file_identity, model_identity, write_json_atomic,
)


def guard_training(model, pairs, out, settings, code_files):
    target = Path(out)
    identity = {
        "schema_version": 1,
        "model": model_identity(model),
        "pairs": dataset_identity(pairs),
        "audit_receipt": (file_identity(Path(pairs).parent / "audit.json")
                          if (Path(pairs).parent / "audit.json").exists() else None),
        "settings": json.loads(json.dumps(settings, allow_nan=False)),
        "code": {str(Path(p).resolve()): file_identity(p) for p in code_files},
    }
    guard = target / "training_guard.json"
    if guard.exists():
        if json.loads(guard.read_text()) != identity:
            raise ValueError(f"Training identity changed; refusing resume: {target}")
    else:
        if any((target / name).exists() for name in ("ckpt_latest", "final", "run_manifest.json")):
            raise ValueError(f"Cannot adopt unbound training artifacts: {target}")
        write_json_atomic(guard, identity)
    return identity


def completed_adapter(path):
    target = Path(path)
    try:
        config = json.loads((target / "adapter_config.json").read_text())
        if not isinstance(config, dict):
            return False
        _validate_safetensors(target / "adapter_model.safetensors")
        return True
    except (OSError, ValueError, TypeError):
        return False


def verify_pair_receipt(directory, *, package=None):
    from fastfill.v1_1.audit_pairs import POLICY

    root = Path(directory)
    receipt = json.loads((root / "audit.json").read_text())
    if receipt.get("protocol") != "fastfill_v1.1_pair_sanitizer_v1":
        raise ValueError("Unrecognized audited pair protocol")
    if receipt.get("policy") != POLICY:
        raise ValueError("Missing or changed audited pair policy")
    for source in ("train", "stage1", "stage2"):
        entry = receipt.get("inputs", {}).get(source, {})
        if not isinstance(entry.get("path"), str) or not entry["path"] or not re.fullmatch("[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError(f"Missing audited input identity: {source}")
    code = receipt.get("code_sha256", {})
    required = {"v1_1/audit_pairs.py", "v1_1/build_dpo.py", "v1_1/scene_v1_1.py",
                "validate.py", "scene.py", "anchors.py"}
    if not required.issubset(code):
        raise ValueError("Missing audited code identity")
    package = Path(package) if package is not None else Path(__file__).resolve().parent.parent
    for relative, digest in code.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or file_identity(package / path)["sha256"] != digest:
            raise ValueError(f"Changed audited code: {relative}")
    for stage in ("stage1", "stage2"):
        entry = receipt["outputs"][stage]
        name = f"{stage}.cleaned.jsonl"
        if entry["file"] != name:
            raise ValueError("Unexpected audited pair filename")
        identity = dataset_identity(root / name)
        if identity["sha256"] != entry["sha256"] or identity["line_count"] != entry["kept"]:
            raise ValueError(f"Pair bytes differ from audited receipt: {name}")
    return receipt


def require_isolated_run(original_root, clean_run):
    runs = Path(original_root).resolve() / "runs"
    old = runs / "fastfill_v1_1"
    new = Path(clean_run).resolve()
    if new.parent != runs or not new.name.startswith("fastfill_v1_1_clean_"):
        raise ValueError("Clean run must be isolated from the existing run")
    for name in ("dpo1_lora", "dpo1_merged", "dpo2_lora", "final_merged"):
        if not (new / name).resolve().is_relative_to(new):
            raise ValueError("Clean run artifact must be isolated from existing paths")
    return new


def merge_identity(base, adapter):
    path = Path(adapter)
    if not completed_adapter(path):
        raise ValueError("Cannot merge an incomplete adapter")
    return {"base": model_identity(base),
            "adapter": {name: file_identity(path / name) for name in
                        ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "tokenizer.json")},
            "merge_code": file_identity(Path(__file__).resolve().parent.parent / "merge_lora.py")}


def merge_receipt_path(out):
    path = Path(out)
    return path.with_name(path.name + ".merge.json")


def seal_merge(base, adapter, out):
    receipt = {"schema_version": 1, "inputs": merge_identity(base, adapter),
               "output": model_identity(out)}
    write_json_atomic(merge_receipt_path(out), receipt)
    return receipt


def merge_complete(base, adapter, out):
    try:
        receipt = json.loads(merge_receipt_path(out).read_text())
        return receipt == {"schema_version": 1, "inputs": merge_identity(base, adapter),
                           "output": model_identity(out)}
    except (OSError, ValueError, TypeError):
        return False


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "tokenizer.json")


def _object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Invalid lineage object: {path}")
    return value


def _source_package(receipt):
    code = receipt.get("code", {})
    guards = [Path(path) for path in code if Path(path).parts[-2:] == ("v1_1", "training_guard.py")]
    if len(guards) != 1:
        raise ValueError("Missing frozen training code lineage")
    package = guards[0].resolve().parent.parent
    required = {str(package / name) for name in ("v1_1/training_guard.py", "v1_1/train_dpo.py", "merge_lora.py")}
    if set(code) != required or any(file_identity(path) != digest for path, digest in code.items()):
        raise ValueError("Frozen training source changed")
    return package


def _recipe(stage):
    return {"lr": 5e-6 if stage == 1 else 2e-6, "epochs": 2, "beta": 0.1, "bs": 1,
            "grad_accum": 4, "max_len": 8192, "lora_r": 32, "lora_alpha": 64,
            "save_steps": 150, "seed": 42, "world_size": 4}


def _verify_manifest(adapter, base, pairs, settings):
    args = _object(adapter / "run_manifest.json").get("args", {})
    if any(args.get(key) != value for key, value in settings.items() if key != "world_size"):
        raise ValueError("Training manifest differs from guarded recipe")
    for key, expected in (("model", base), ("pairs", pairs), ("out", adapter)):
        if not isinstance(args.get(key), str) or Path(args[key]).resolve() != expected.resolve():
            raise ValueError(f"Training manifest {key} differs from lineage")


def _count_admitted_pairs(base, pairs, max_len, source_package=None):
    """Recount legacy length admission on CPU without loading weights or all tokens."""
    from transformers import AutoTokenizer
    from fastfill.scene import prompt_text

    if source_package is not None and file_identity(Path(prompt_text.__code__.co_filename)) != file_identity(Path(source_package) / "scene.py"):
        raise ValueError("Legacy prompt rendering differs from frozen training code")
    tokenizer = AutoTokenizer.from_pretrained(base)
    if not isinstance(tokenizer.eos_token, str):
        raise ValueError("Legacy completion tokenizer lacks EOS")
    admitted = 0
    with Path(pairs).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = prompt_text(tokenizer, row["prompt"])
            prompt_ids, chosen, rejected = tokenizer(
                [prompt, prompt + row["chosen"] + tokenizer.eos_token,
                 prompt + row["rejected"] + tokenizer.eos_token], add_special_tokens=False)["input_ids"]
            length = len(prompt_ids)
            if all(length < len(ids) <= max_len and ids[:length] == prompt_ids for ids in (chosen, rejected)):
                admitted += 1
    return admitted


def _completion_position(admitted, settings):
    if type(admitted) is not int or admitted < 1:
        raise ValueError("Invalid admitted pair count in completion evidence")
    batches = math.ceil(math.ceil(admitted / settings["world_size"]) / settings["bs"])
    steps = settings["epochs"] * math.ceil(batches / settings["grad_accum"])
    return batches, {"epoch": settings["epochs"] - 1, "it": batches - 1, "step": steps}


def verify_final_completion(adapter, base, pairs, receipt):
    """Read-only proof of a full run, including old checkpoints without receipts."""
    adapter, base, pairs = Path(adapter), Path(base), Path(pairs)
    final, settings = adapter / "final", receipt["settings"]
    if not completed_adapter(final):
        raise ValueError("Incomplete final adapter in training lineage")
    files = {name: file_identity(final / name) for name in ADAPTER_FILES}
    completion_path = final / "completion.json"
    if completion_path.exists():
        completion = _object(completion_path)
        admitted = completion.get("admitted_pairs")
        batches, position = _completion_position(admitted, settings)
        expected = {"schema_version": 1, "kind": "fastfill_v1.1_dpo_completion",
                    "training_guard": file_identity(adapter / "training_guard.json"),
                    "loader_batches": batches, "world_size": settings["world_size"],
                    "epochs": settings["epochs"], "position": position,
                    "expected_total_steps": position["step"], "adapter_files": files}
        if admitted > receipt["pairs"]["line_count"] or any(completion.get(k) != v for k, v in expected.items()):
            raise ValueError("Final completion receipt does not prove this full guarded run")
        return {"evidence": "trainer_completion_receipt", "receipt": file_identity(completion_path),
                "admitted_pairs": admitted, "position": position}
    import torch

    checkpoint = adapter / "ckpt_latest"
    if any(file_identity(checkpoint / name) != files[name] for name in ADAPTER_FILES[:2]):
        raise ValueError("Legacy final adapter differs from last checkpoint")
    admitted = _count_admitted_pairs(base, pairs, settings["max_len"], _source_package(receipt))
    _, position = _completion_position(admitted, settings)
    state_path = checkpoint / "trainer_state.pt"
    state_identity = file_identity(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("state") != position or file_identity(state_path) != state_identity:
        raise ValueError("Legacy checkpoint position does not prove complete training")
    return {"evidence": "legacy_final_matches_complete_checkpoint", "checkpoint": state_identity,
            "admitted_pairs": admitted, "position": position}


def verify_training_lineage(run, tag):
    """Verify old or new frozen runs without adopting, sealing, or changing files."""
    run = Path(run).resolve()
    if tag not in {"sft", "dpo1", "final"}:
        raise ValueError("Unknown training lineage tag")
    if tag == "sft":
        return {"tag": tag, "model": model_identity(run / "sft_merged"),
                "completion": {"evidence": "legacy_sft_model_identity_only"}}
    stage = 1 if tag == "dpo1" else 2
    parent = verify_training_lineage(run, "sft" if stage == 1 else "dpo1")
    adapter = run / f"dpo{stage}_lora"
    pairs = run / "pairs" / f"stage{stage}.cleaned.jsonl"
    merged = run / ("dpo1_merged" if stage == 1 else "final_merged")
    if any(not path.resolve().is_relative_to(run) for path in (adapter, pairs, merged)):
        raise ValueError("Training lineage escapes the clean run")
    guard_path = adapter / "training_guard.json"
    receipt = _object(guard_path)
    package = _source_package(receipt)
    expected = {"schema_version": 1, "model": parent["model"], "pairs": dataset_identity(pairs),
                "audit_receipt": file_identity(pairs.parent / "audit.json"), "settings": _recipe(stage)}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Guarded training lineage differs from model, pairs, audit or recipe")
    verify_pair_receipt(pairs.parent, package=package)
    base = Path(parent["model"]["path"])
    _verify_manifest(adapter, base, pairs, receipt["settings"])
    completion = verify_final_completion(adapter, base, pairs, receipt)
    merge_path = merge_receipt_path(merged)
    merge = _object(merge_path)
    model = model_identity(merged)
    expected_merge = {"schema_version": 1, "inputs": {"base": parent["model"],
        "adapter": {name: file_identity(adapter / "final" / name) for name in ADAPTER_FILES},
        "merge_code": file_identity(package / "merge_lora.py")}, "output": model}
    if merge != expected_merge:
        raise ValueError("Merged model lacks the expected completed training lineage")
    return {"tag": tag, "model": model, "source_package": str(package),
            "guard": file_identity(guard_path), "merge": file_identity(merge_path),
            "completion": completion, "parent": parent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bind", "model-complete", "adapter-complete", "verify-pairs", "isolate", "merge-complete", "seal-merge", "verify-lineage"))
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--pairs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--tag", choices=("sft", "dpo1", "final"))
    args = parser.parse_args()
    if args.action == "verify-lineage":
        if args.tag is None:
            parser.error("verify-lineage requires --tag")
        print(json.dumps(verify_training_lineage(args.out, args.tag)), flush=True)
        return
    if args.action in ("merge-complete", "seal-merge"):
        if not args.model or not args.adapter:
            parser.error("merge actions require --model and --adapter")
        if args.action == "merge-complete":
            raise SystemExit(0 if merge_complete(args.model, args.adapter, args.out) else 1)
        seal_merge(args.model, args.adapter, args.out)
        return
    if args.action == "verify-pairs":
        verify_pair_receipt(args.out)
        return
    if args.action == "isolate":
        if not args.model:
            parser.error("isolate requires --model ORIGINAL_ROOT")
        require_isolated_run(args.model, args.out)
        return
    if args.action == "adapter-complete":
        raise SystemExit(0 if completed_adapter(args.out) else 1)
    if args.action == "model-complete":
        try:
            model_identity(args.out)
        except (OSError, ValueError, TypeError):
            raise SystemExit(1)
        return
    if not args.model or not args.pairs or args.lr is None:
        parser.error("bind requires --model, --pairs, and --lr")
    verify_pair_receipt(Path(args.pairs).parent)
    package = Path(__file__).resolve().parent
    settings = {"lr": args.lr, "epochs": 2, "beta": 0.1, "bs": 1, "grad_accum": 4,
                "max_len": 8192, "lora_r": 32, "lora_alpha": 64, "save_steps": 150,
                "seed": 42, "world_size": args.nproc}
    receipt = guard_training(args.model, args.pairs, args.out, settings,
                             [package / "train_dpo.py", package / "training_guard.py",
                              package.parent / "merge_lora.py"])
    print(json.dumps({"status": "identity_verified", "out": args.out,
                      "pairs": receipt["pairs"]["line_count"]}), flush=True)


if __name__ == "__main__":
    main()
