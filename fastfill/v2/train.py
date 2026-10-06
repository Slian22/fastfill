"""Joint structured geometry training. v1 entry points and snapshots are untouched."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from fastfill.v2.batch import collate_samples, load_tokenizer, tokenize_condition
from fastfill.v2.io import ensure_disjoint, fingerprint, read_samples, run_metadata, safe_output
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.model import ModelConfig, build_model, model_inputs
from fastfill.v2.objective import ObjectiveWindow, local_objective_count


def _preflight(samples, tokenizer, model_config, training, loss_config=LossConfig()):
    kept, rejected = [], []
    for index, sample in enumerate(samples):
        condition = sample["condition"]
        tokens, _ = tokenize_condition(condition, tokenizer)
        if len(tokens) > training["max_length"] or len(condition["objects"]) > model_config.max_objects:
            rejected.append({"row": index, "reason": "context_or_object_budget", "tokens": len(tokens),
                             "objects": len(condition["objects"]), "provenance": sample["provenance"]})
            continue
        # Fail loudly on malformed geometry/conditions rather than filtering an unexplained error.
        b = collate_samples([sample], tokenizer, max_length=training["max_length"], max_objects=model_config.max_objects)
        if not local_objective_count(b, loss_config):
            rejected.append({"row": index, "reason": "no_active_objective", "provenance": sample["provenance"]})
            continue
        kept.append(sample)
    if not kept:
        raise ValueError("no supervised samples have an active objective within context/object budgets; inspect enabled losses and label validity")
    return kept, rejected


def _gradients(model):
    names = ("backbone", "decoder", "position_head", "size_head", "yaw_logits_head", "yaw_residual_head")
    return {name: float(sum((p.grad.detach().float().square().sum() for key, p in model.named_parameters()
                            if name in key and p.grad is not None), torch.tensor(0., device=next(model.parameters()).device)).sqrt())
            for name in names}


def _record(result, model, step, elapsed, accelerator):
    # Every rank participates: local numerators divided by global count are
    # correct for gradients; their mean yields the global diagnostic value.
    # Optional box/collision operators may produce float64 only on ranks with
    # valid terms. Diagnostic collectives need one explicit dtype on all ranks;
    # the original tensors used for backward retain their precision and graph.
    return {"step": step, "elapsed_s": elapsed, "loss": float(accelerator.reduce(result["loss"].detach().float(), reduction="mean")),
            "unweighted": {key: float(accelerator.reduce(result[key].detach().float(), reduction="mean")) for key in
                           ("position", "size", "yaw_cls", "yaw_reg", "box", "collision", "boundary")},
            "counts_global": result["counts"], "gradient_norms": _gradients(model),
            "active_objective_count": result.get("active_objective_count"),
            "active_objective_scope": "global accumulated window"}


def _verify_input_fingerprints(metadata, accelerator=None):
    changed = []
    for path_key, hash_key in (("data_path", "data_sha256"),
                               ("validation_data_path", "validation_data_sha256")):
        path = metadata[path_key]
        if path is None:
            continue
        try:
            matches = fingerprint(path) == metadata[hash_key]
        except OSError:
            matches = False
        if not matches:
            changed.append(path)
    failed = bool(changed)
    if accelerator is not None:
        # All ranks must leave this boundary together if any input changed;
        # otherwise one process can fail while peers wait to finish the run.
        flags = torch.tensor(int(failed), dtype=torch.int64, device=accelerator.device)
        failed = bool(accelerator.reduce(flags, reduction="sum").item())
    if failed:
        detail = ", ".join(changed) if changed else "input observed by another process"
        raise RuntimeError(f"training input fingerprint changed during the run: {detail}")


DEFAULT_TRAINING = {"steps": 1000, "batch_size": 2, "learning_rate": 1e-4,
                    "gradient_accumulation_steps": 1, "max_length": 4096, "seed": 42,
                    "clip_grad_norm": 1., "mixed_precision": "no", "cpu": False, "cpu_threads": 2,
                    "checkpoint_every": 100, "validate_every": 100}


def _training_config(config):
    unknown = set(config) - set(DEFAULT_TRAINING)
    if unknown:
        raise ValueError(f"unknown training fields: {sorted(unknown)}")
    cfg = {**DEFAULT_TRAINING, **config}
    for key in ("steps", "batch_size", "gradient_accumulation_steps", "max_length", "cpu_threads"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("checkpoint_every", "validate_every", "seed"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not isinstance(cfg["cpu"], bool):
        raise ValueError("cpu must be a boolean")
    for key in ("learning_rate", "clip_grad_norm"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < float("inf"):
            raise ValueError(f"{key} must be positive and finite")
    if cfg["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed precision must be no/fp16/bf16")
    return cfg


@torch.no_grad()
def _validation(model, loader, criterion, accelerator):
    model.eval()
    total, batches = 0., 0
    for batch in loader:
        result = criterion(model(**model_inputs(batch)), batch)
        total += float(accelerator.reduce(result["loss"].detach().float(), reduction="mean"))
        batches += 1
    model.train()
    return {"geometry_objective_mean_of_batches": total / max(1, batches), "batches": batches,
            "selection_scope": "geometry diagnostic; asset/runtime evaluation is separate"}


def run_training(config, data, output, *, validation=None, dry_run=False, max_samples=None):
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    model_config, loss_config = ModelConfig(**config.get("model", {})), LossConfig(**config.get("loss", {}))
    training = _training_config(config.get("training", {}))
    if set(config) - {"model", "loss", "training"}:
        raise ValueError("config only accepts model/loss/training")
    if dry_run and model_config.backbone != "tiny":
        raise ValueError("offline dry-run requires explicit tiny backbone")
    target = safe_output(output)
    metadata = {**run_metadata(data),
                "validation_data_path": str(Path(validation).resolve()) if validation else None,
                "validation_data_sha256": fingerprint(validation) if validation else None}
    samples = read_samples(data, training=True, max_samples=max_samples)
    heldout = read_samples(validation) if validation else []
    _verify_input_fingerprints(metadata)
    if heldout:
        if any(r["provenance"].get("split") != "validation" for r in heldout):
            raise ValueError("model selection accepts validation only; test is reserved for final evaluation")
        ensure_disjoint(samples, heldout)
    torch.set_num_threads(training["cpu_threads"])
    accelerator = Accelerator(cpu=training["cpu"], mixed_precision=training["mixed_precision"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"])
    set_seed(training["seed"])
    tokenizer = load_tokenizer(model_config.backbone, local_files_only=model_config.local_files_only)
    samples, rejected = _preflight(samples, tokenizer, model_config, training, loss_config)
    validation_samples, validation_rejected = _preflight(heldout, tokenizer, model_config, training, loss_config) if heldout else ([], [])
    collate = partial(collate_samples, tokenizer=tokenizer, max_length=training["max_length"], max_objects=model_config.max_objects)
    loader = DataLoader(samples, batch_size=training["batch_size"], shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(training["seed"]))
    val_loader = DataLoader(validation_samples, batch_size=training["batch_size"], collate_fn=collate) if heldout else None
    model, criterion = build_model(model_config), GeometryCriterion(loss_config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=training["learning_rate"])
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if accelerator.is_main_process:
        target.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    model.train()
    logs, start, step = [], time.perf_counter(), 0
    window, skipped_windows = ObjectiveWindow(), 0
    steps = 1 if dry_run else training["steps"]
    while step < steps:
        epoch_start_step = step
        for batch in loader:
            record = None
            with accelerator.accumulate(model):
                result = criterion(model(**model_inputs(batch)), batch)
                if not torch.isfinite(result["loss"]):
                    raise RuntimeError("nonfinite structured loss")
                accelerator.backward(result["loss"])
                window = window.add(result["active_objective_count_local"])
                if accelerator.sync_gradients:
                    count = window.global_count(accelerator)
                    window = ObjectiveWindow()
                    if count:
                        accelerator.clip_grad_norm_(model.parameters(), training["clip_grad_norm"])
                        step += 1
                        record = _record({**result, "active_objective_count": count}, model, step,
                                         time.perf_counter() - start, accelerator)
                        optimizer.step()
                    else:
                        skipped_windows += 1
                    # Clear both completed and skipped windows. Non-sync
                    # microbatches keep their gradients until this boundary.
                    optimizer.zero_grad(set_to_none=True)
            if record is not None:
                # Capture training gradients before step/zero, then evaluate the
                # updated model outside accumulation so step-k validation and
                # state-step-k describe the same weights on every rank.
                if val_loader is not None and training["validate_every"] and step % training["validate_every"] == 0:
                    record = {**record, "validation": _validation(model, val_loader, criterion, accelerator)}
                logs.append(record)
                if accelerator.is_main_process:
                    print(json.dumps(record), flush=True)
            if record is not None and training["checkpoint_every"] and step % training["checkpoint_every"] == 0:
                accelerator.save_state(str(target / f"state-step-{step}"))
            if step >= steps:
                break
        if step == epoch_start_step:
            raise RuntimeError("epoch contains no active objective; no supervised optimizer update is possible")
    accelerator.wait_for_everyone()
    _verify_input_fingerprints(metadata, accelerator)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(target / "model")
        tokenizer.save_pretrained(target / "tokenizer")
        manifest = {**metadata, "model": asdict(model_config), "loss": asdict(loss_config), "training": training,
            "offline_smoke": model_config.backbone == "tiny", "dry_run": dry_run, "steps_completed": step,
            "supervised_samples": len(samples), "skipped_no_objective_windows": skipped_windows, "rejected": rejected, "validation_rejected": validation_rejected,
            "subset_limit": max_samples, "world_size": accelerator.num_processes,
            "optimizer_step_policy": "update iff global accumulated-window eligible objective count is positive; zero numeric loss is eligible",
            "reduction": "complete-field valid instances; fixed coordinate sum divided by 3; global valid count per microbatch; accumulated microbatch means",
            "elapsed_s": time.perf_counter() - start, "trainable_parameters": sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)}
        (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (target / "training_log.json").write_text(json.dumps(logs, indent=2) + "\n")
    accelerator.end_training()
    return logs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/structured.json")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    config = {**config, "model": {**config.get("model", {})}, "training": {**config.get("training", {})}}
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    for key in ("steps", "max_length"):
        if getattr(args, key) is not None:
            config["training"][key] = getattr(args, key)
    if args.cpu:
        config["training"]["cpu"] = True
    run_training(config, args.data, args.output, validation=args.validation, dry_run=args.dry_run, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
