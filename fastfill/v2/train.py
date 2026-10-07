"""Joint structured geometry training. v1 entry points and snapshots are untouched.

Resume (``training.resume`` / ``--resume <state_dir>``) continues into a NEW
output directory. Restored: model, optimizer, LR schedule, GradScaler,
Python/NumPy/torch RNG, Accelerate's accumulation counter, completed-step and
skipped-window counters, earlier log records, and the data order (epoch,
consumed batches, shuffle generator; augmentation is seeded per
(seed, epoch, row) so replayed samples are identical). Required unchanged:
every config section except ``training.resume``, ``--max-samples`` and the
training data hash. Not restored: wall-clock ``elapsed_s`` and the earlier
process's ``train.log``.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

from fastfill.v2.batch import AUGMENT_DEFAULTS, augment_sample, collate_samples, load_tokenizer, tokenize_condition
from fastfill.v2.data import filter_rows_by_flags
from fastfill.v2.io import ensure_disjoint, fingerprint, read_samples, run_metadata, safe_output
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.matching import match_batch
from fastfill.v2.model import ModelConfig, build_model, model_inputs
from fastfill.v2.objective import ObjectiveWindow, local_objective_count
from fastfill.v2.size_range import size_target_conflicts


TERMS = ("position", "size", "yaw_cls", "yaw_reg", "box", "collision", "boundary")


def _target_predictions(batch):
    """Finite stand-in predictions so preflight can certify every exchangeable group."""
    size = batch["targets"]["size"]
    return {"position_normalized": torch.nan_to_num(batch["targets"]["position_normalized"], nan=0.),
            "size": torch.where(torch.isfinite(size) & (size > 0), size, torch.ones_like(size)),
            "slot_mask": batch["slot_mask"]}


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
        try:
            match_batch(_target_predictions(b), b, loss_config.hungarian, loss_config.alpha_position, loss_config.alpha_size)
        except ValueError as error:
            raise ValueError(f"row {index} cannot be matched before training: {error}; provenance {sample['provenance']}") from error
        conflicts = size_target_conflicts(b, model_config, loss_config)
        if conflicts:
            rejected.append({"row": index, "reason": "size_target_outside_model_range",
                             "coordinates": conflicts, "provenance": sample["provenance"]})
            continue
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


def _nonfinite_gradients(model, accelerator):
    """bf16 has no GradScaler: detect inf/nan gradients on any rank before clipping or stepping."""
    local = any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    flag = torch.tensor(int(local), dtype=torch.int64, device=accelerator.device)
    return bool(accelerator.reduce(flag, reduction="sum").item())


@torch.no_grad()
def _rescale_flushed_window_gradients(model, configured_steps, microbatches):
    """Undo Accelerate's fixed-K divisor for an incomplete epoch-tail window.

    Backward and DDP synchronization have completed. Gradients may still be AMP
    scaled; multiplying before Accelerate unscale/clip preserves that scaling.
    Empty-label microbatches belong to the declared microbatch-mean denominator.
    """
    if not 1 <= microbatches <= configured_steps:
        raise RuntimeError("invalid synchronized accumulation-window length")
    if microbatches != configured_steps:
        factor = configured_steps / microbatches
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)


class _Window:
    """Local per-term unweighted sums and valid counts over every microbatch of a window.

    The criterion exposes local ``term_sums``/``term_counts``; one summed
    collective at summary time gives the global count-weighted window means.
    ``loss`` keeps the microbatch mean of the weighted objective (local
    numerators over global counts, so its rank mean is the global value).
    """

    def __init__(self):
        self.loss, self.microbatches, self.rows = 0., 0, []
        self.sums, self.counts = dict.fromkeys(TERMS, 0.), dict.fromkeys(TERMS, 0)

    def add(self, result, batch=None):
        self.loss += float(result["loss"].detach().float())
        self.microbatches += 1
        for key in TERMS:
            self.sums[key] += float(result["term_sums"][key])
            self.counts[key] += int(result["term_counts"][key])
        if batch is not None:
            self.rows.extend(p.get("scene_id") for p in batch["provenance"])
        return self

    def summary(self, accelerator):
        # Every rank enters this collective with one dtype/shape, including ranks without valid terms.
        local = torch.tensor([self.loss, *(self.sums[key] for key in TERMS), *(float(self.counts[key]) for key in TERMS)],
                             dtype=torch.float64, device=accelerator.device)
        values = accelerator.reduce(local, reduction="sum").tolist()
        sums, counts = values[1:1 + len(TERMS)], values[1 + len(TERMS):]
        return {"loss": values[0] / accelerator.num_processes / max(1, self.microbatches),
                "unweighted": {key: total / count if count else None for key, total, count in zip(TERMS, sums, counts)},
                "counts": {key: int(count) for key, count in zip(TERMS, counts)}, "microbatches": self.microbatches}


def _record(window, model, optimizer, step, elapsed, count, accelerator):
    summary = window.summary(accelerator)
    return {"step": step, "elapsed_s": elapsed, "loss": summary["loss"], "unweighted": summary["unweighted"],
            "counts_window": summary["counts"], "gradient_norms": _gradients(model),
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "active_objective_count": count, "accumulation_microbatches": summary["microbatches"],
            "rows_sha256": hashlib.sha256(json.dumps(window.rows).encode()).hexdigest()[:16],
            "active_objective_scope": "global accumulated window",
            "logging_scope": "loss: microbatch mean; unweighted: count-weighted means over the window; rows digest: this rank's order"}


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
                    "checkpoint_every": 100, "validate_every": 100,
                    "resume": None, "export_model_every_checkpoint": True}
# warmup_steps None -> 3% of steps; decoder_lr None -> training.learning_rate (which trains LoRA/backbone).
DEFAULT_OPTIMIZER = {"warmup_steps": None, "schedule": "cosine", "decoder_lr": None}
DEFAULT_AUGMENTATION = AUGMENT_DEFAULTS
DEFAULT_VALIDATION = {"exclude_flags": ["oob_objects", "fixed_collision", "overlapping_furniture"]}
CONFIG_SECTIONS = ("model", "loss", "training", "optimizer", "augmentation", "validation")


def _section(section, name, defaults):
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(section) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return {**defaults, **section}


def _positive_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 < value < float("inf")


def _training_config(config):
    cfg = _section(config, "training", DEFAULT_TRAINING)
    for key in ("steps", "batch_size", "gradient_accumulation_steps", "max_length", "cpu_threads"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("checkpoint_every", "validate_every", "seed"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("cpu", "export_model_every_checkpoint"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("learning_rate", "clip_grad_norm"):
        if not _positive_number(cfg[key]):
            raise ValueError(f"{key} must be positive and finite")
    if cfg["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed precision must be no/fp16/bf16")
    if cfg["resume"] is not None and not isinstance(cfg["resume"], (str, Path)):
        raise ValueError("resume must be null or a saved state directory path")
    return cfg


def _optimizer_config(config, training):
    cfg = _section(config.get("optimizer", {}), "optimizer", DEFAULT_OPTIMIZER)
    if cfg["warmup_steps"] is None:
        cfg["warmup_steps"] = round(.03 * training["steps"])
    if cfg["decoder_lr"] is None:
        cfg["decoder_lr"] = training["learning_rate"]
    if (isinstance(cfg["warmup_steps"], bool) or not isinstance(cfg["warmup_steps"], int)
            or not 0 <= cfg["warmup_steps"] < training["steps"]):
        raise ValueError("warmup_steps must be an integer in [0, steps)")
    if cfg["schedule"] not in {"cosine", "constant"}:
        raise ValueError("schedule must be cosine or constant")
    if not _positive_number(cfg["decoder_lr"]):
        raise ValueError("decoder_lr must be positive and finite")
    return cfg


def _augmentation_config(config):
    cfg = _section(config.get("augmentation", {}), "augmentation", DEFAULT_AUGMENTATION)
    for key in ("rotate90", "mirror", "shuffle_objects"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"augmentation.{key} must be a boolean")
    for key in ("drop_constraints_p", "drop_support_p", "category_only_description_p"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"augmentation.{key} must be a probability in [0,1]")
    return cfg


def _validation_config(config):
    cfg = _section(config.get("validation", {}), "validation", DEFAULT_VALIDATION)
    flags = cfg["exclude_flags"]
    if not isinstance(flags, list) or any(not isinstance(flag, str) or not flag for flag in flags):
        raise ValueError("validation.exclude_flags must be a list of flag names")
    return cfg


def _lr_factor(completed, warmup, total, schedule):
    """Linear warmup over `warmup` updates, then cosine from lr to 10% of lr at the final update."""
    if completed < warmup:
        return (completed + 1) / warmup
    if schedule == "constant":
        return 1.
    progress = (completed - warmup) / max(1, total - warmup)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def _parameter_groups(model, lr, decoder_lr):
    """Backbone (LoRA or full) parameters train at lr; projection, slots, decoder and heads at decoder_lr."""
    groups = {"backbone": [], "decoder": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups["backbone" if name.startswith("backbone.") else "decoder"].append(parameter)
    return [{"params": params, "lr": lr if name == "backbone" else decoder_lr, "name": name}
            for name, params in groups.items() if params]


class _Rows(torch.utils.data.Dataset):
    """Training rows; augmentation is seeded per (seed, epoch, row) so a resumed run replays identical samples.

    Preflight admits the original condition only. An augmented one that outgrows
    ``max_length`` (sign/digit growth, longer shuffled IDs) would make collate
    raise inside the DataLoader, so it falls back to the original row and is counted.
    """

    def __init__(self, rows, augmentation, seed, tokenizer=None, max_length=None):
        self.rows, self.augmentation, self.seed, self.epoch = rows, augmentation, seed, 0
        self.tokenizer, self.max_length, self.fallbacks = tokenizer, max_length, 0
        self.active = (any(augmentation[key] for key in ("rotate90", "mirror", "shuffle_objects")) or
                       any(augmentation[key] > 0 for key in ("drop_constraints_p", "drop_support_p", "category_only_description_p")))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # ponytail: num_workers=0, so `epoch` set by the trainer and `fallbacks` are visible in-process.
        if not self.active:
            return self.rows[index]
        digest = hashlib.sha256(f"{self.seed}:{self.epoch}:{index}".encode()).digest()
        sample = augment_sample(self.rows[index], self.augmentation,
                                torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") >> 1))
        if self.tokenizer is not None and len(tokenize_condition(sample["condition"], self.tokenizer)[0]) > self.max_length:
            self.fallbacks += 1
            return self.rows[index]
        return sample


class _TrainerState:
    """Resume bookkeeping saved by Accelerate beside model/optimizer/scheduler/scaler/RNG state."""

    def __init__(self):
        self.values = {}

    def state_dict(self):
        return self.values

    def load_state_dict(self, values):
        self.values = values


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@torch.no_grad()
def _validation(model, loader, criterion, accelerator):
    model.eval()
    window = _Window()
    for batch in loader:
        window.add(criterion(model(**model_inputs(batch)), batch))
    model.train()
    summary = window.summary(accelerator)
    return {"geometry_objective_mean_of_batches": summary["loss"], "batches": summary["microbatches"],
            "unweighted": summary["unweighted"], "counts": summary["counts"],
            "selection_scope": "geometry diagnostic; asset/runtime evaluation is separate"}


def _category_median_sizes(rows):
    """Per-category per-axis medians of fully valid size labels; the global median backs unseen categories."""
    sizes = {}
    for row in rows:
        categories = {obj["id"]: obj.get("category") for obj in row["condition"]["objects"]}
        valid = row.get("validity", {}).get("size", [])
        for index, target in enumerate(row["target"]["objects"]):
            mask = valid[index] if index < len(valid) else False
            if mask is True or (isinstance(mask, list) and len(mask) == 3 and all(mask)):
                sizes.setdefault(categories.get(target["id"]), []).append(target["target_size_local_m"])
    medians = {category: [statistics.median(axis) for axis in zip(*values)] for category, values in sizes.items()}
    everything = [size for values in sizes.values() for size in values]
    medians[None] = [statistics.median(axis) for axis in zip(*everything)] if everything else [1., 1., 1.]
    return medians


def _baseline_predictions(batch, medians, yaw_bins):
    """Room-centre floor position, per-category median size, uniform yaw bins with zero residual."""
    mask = batch["slot_mask"]
    b, n = mask.shape
    size = torch.ones((b, n, 3), device=mask.device)
    for i, objects in enumerate(batch["objects"]):
        for j, obj in enumerate(objects):
            size[i, j] = torch.tensor(medians.get(obj.get("category"), medians[None]), device=mask.device)
    return {"position_normalized": torch.tensor([.5, .5, 0.], device=mask.device).expand(b, n, 3).clone(),
            "size": size, "yaw_logits": torch.zeros((b, n, yaw_bins), device=mask.device),
            "yaw_residuals": torch.zeros((b, n, yaw_bins), device=mask.device), "slot_mask": mask}


@torch.no_grad()
def _baseline(loader, criterion, accelerator, medians, yaw_bins):
    window = _Window()
    for batch in loader:
        window.add(criterion(_baseline_predictions(batch, medians, yaw_bins), batch))
    return {**window.summary(accelerator), "predictor": "room-centre floor position; per-category median size of the first "
            "20000 training rows (global median for unseen categories); uniform yaw bins with zero residual"}


def _due(step, every):
    return bool(every) and step % every == 0


def run_training(config, data, output, *, validation=None, dry_run=False, max_samples=None):
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    unknown = set(config) - set(CONFIG_SECTIONS)
    if unknown:
        raise ValueError(f"config only accepts {'/'.join(CONFIG_SECTIONS)}; unknown sections {sorted(unknown)}")
    model_config, loss_config = ModelConfig(**config.get("model", {})), LossConfig(**config.get("loss", {}))
    training = _training_config(config.get("training", {}))
    optimizer_config = _optimizer_config(config, training)
    augmentation = _augmentation_config(config)
    validation_config = _validation_config(config)
    if dry_run and model_config.backbone != "tiny":
        raise ValueError("offline dry-run requires explicit tiny backbone")
    resume = training["resume"]
    resolved = {"model": asdict(model_config), "loss": asdict(loss_config), "optimizer": optimizer_config,
                "augmentation": augmentation, "validation": validation_config,
                "training": {key: value for key, value in training.items() if key != "resume"},
                "max_samples": max_samples, "dry_run": dry_run}
    config_digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest()
    target = safe_output(output)
    metadata = {**run_metadata(data),
                "validation_data_path": str(Path(validation).resolve()) if validation else None,
                "validation_data_sha256": fingerprint(validation) if validation else None}
    samples = read_samples(data, training=True, max_samples=max_samples)
    heldout = read_samples(validation) if validation else []
    _verify_input_fingerprints(metadata)
    validation_excluded = 0
    if heldout:
        if any(r["provenance"].get("split") != "validation" for r in heldout):
            raise ValueError("model selection accepts validation only; test is reserved for final evaluation")
        ensure_disjoint(samples, heldout)
        filtered = filter_rows_by_flags(heldout, validation_config["exclude_flags"])
        validation_excluded, heldout = len(heldout) - len(filtered), filtered
        if not heldout:
            raise ValueError("every validation row carries an excluded flag")
    torch.set_num_threads(training["cpu_threads"])
    accelerator = Accelerator(cpu=training["cpu"], mixed_precision=training["mixed_precision"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"], step_scheduler_with_optimizer=False)
    set_seed(training["seed"])
    tokenizer = load_tokenizer(model_config.backbone, local_files_only=model_config.local_files_only)
    samples, rejected = _preflight(samples, tokenizer, model_config, training, loss_config)
    validation_samples, validation_rejected = _preflight(heldout, tokenizer, model_config, training, loss_config) if heldout else ([], [])
    collate = partial(collate_samples, tokenizer=tokenizer, max_length=training["max_length"], max_objects=model_config.max_objects)
    rows = _Rows(samples, augmentation, training["seed"], tokenizer, training["max_length"])
    generator = torch.Generator().manual_seed(training["seed"])
    loader = DataLoader(rows, batch_size=training["batch_size"], shuffle=True, collate_fn=collate, generator=generator)
    val_loader = DataLoader(validation_samples, batch_size=training["batch_size"], collate_fn=collate) if heldout else None
    baseline_loader = DataLoader(validation_samples[:2000], batch_size=training["batch_size"], collate_fn=collate) if heldout else None
    steps = 1 if dry_run else training["steps"]
    model, criterion = build_model(model_config), GeometryCriterion(loss_config)
    optimizer = torch.optim.AdamW(_parameter_groups(model, training["learning_rate"], optimizer_config["decoder_lr"]),
                                  lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(_lr_factor, warmup=optimizer_config["warmup_steps"],
                                                                     total=steps, schedule=optimizer_config["schedule"]))
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    if val_loader is not None:
        val_loader, baseline_loader = accelerator.prepare(val_loader, baseline_loader)
    state = _TrainerState()
    accelerator.register_for_checkpointing(state)
    logs, step, epoch, skip_batches = [], 0, 0, 0
    skipped_windows, skipped_gradient_overflow_windows, skipped_nonfinite_windows = 0, 0, 0
    if resume is not None:
        accelerator.load_state(str(resume))
        saved = state.values
        if saved.get("config_sha256") != config_digest or saved.get("data_sha256") != metadata["data_sha256"]:
            raise ValueError("resume requires the same config (except training.resume), --max-samples and training data")
        step, epoch, skip_batches = saved["step"], saved["epoch"], saved["batches_done"]
        generator.set_state(saved["generator_state"])
        skipped_windows, skipped_gradient_overflow_windows, skipped_nonfinite_windows = (
            saved["skipped_no_objective_windows"], saved["skipped_gradient_overflow_windows"], saved["skipped_nonfinite_windows"])
        logs = list(saved["logs"])
    resumed_at = step
    with contextlib.ExitStack() as stack:
        if accelerator.is_main_process:
            target.mkdir(parents=True, exist_ok=False)
            tokenizer.save_pretrained(target / "tokenizer")
            stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, stack.enter_context((target / "train.log").open("x")))))
            if resume is not None:
                print(json.dumps({"resumed_from": str(resume), "step": step, "epoch": epoch, "batches_done": skip_batches}), flush=True)
        accelerator.wait_for_everyone()
        baseline = None
        if val_loader is not None:
            baseline = _baseline(baseline_loader, criterion, accelerator, _category_median_sizes(samples[:20000]), model_config.yaw_bins)
            baseline["rows"] = len(baseline_loader.dataset)
            if accelerator.is_main_process:
                print(json.dumps({"baseline": baseline}), flush=True)
        model.train()
        start = time.perf_counter()
        objective, window = ObjectiveWindow(), _Window()
        while step < steps:
            epoch_start_step = step
            epoch_start_skips = skipped_gradient_overflow_windows + skipped_nonfinite_windows
            rows.epoch = epoch
            epoch_generator_state = generator.get_state()
            epoch_loader = accelerator.skip_first_batches(loader, skip_batches) if skip_batches else loader
            batches_done, skip_batches = skip_batches, 0
            for batch in epoch_loader:
                batches_done += 1
                record = None
                with accelerator.accumulate(model):
                    result = criterion(model(**model_inputs(batch)), batch)
                    if not torch.isfinite(result["loss"]):
                        raise RuntimeError("nonfinite structured loss")
                    accelerator.backward(result["loss"])
                    objective = objective.add(result["active_objective_count_local"])
                    window.add(result, batch)
                    if accelerator.sync_gradients:
                        count = objective.global_count(accelerator)
                        objective = ObjectiveWindow()
                        if not count:
                            skipped_windows += 1
                        elif accelerator.scaler is None and _nonfinite_gradients(model, accelerator):
                            # No GradScaler (bf16/no): an inf/nan gradient would poison
                            # clipping and the update. Skip the window; not a training step.
                            skipped_nonfinite_windows += 1
                        else:
                            _rescale_flushed_window_gradients(model, training["gradient_accumulation_steps"], window.microbatches)
                            accelerator.clip_grad_norm_(model.parameters(), training["clip_grad_norm"])
                            optimizer.step()
                            if accelerator.optimizer_step_was_skipped:
                                # GradScaler found nonfinite gradients and deliberately
                                # suppressed the optimizer update. Do not advance the
                                # completed-update schedule, validate or checkpoint it.
                                skipped_gradient_overflow_windows += 1
                            else:
                                step += 1
                                record = _record(window, model, optimizer, step, time.perf_counter() - start, count, accelerator)
                                scheduler.step()
                        # Clear both completed and skipped windows. Non-sync
                        # microbatches keep their gradients until this boundary.
                        optimizer.zero_grad(set_to_none=True)
                        window = _Window()
                if record is not None:
                    # Capture training gradients before step/zero, then evaluate the
                    # updated model outside accumulation so step-k validation and
                    # state-step-k describe the same weights on every rank.
                    checkpoint = _due(step, training["checkpoint_every"])
                    if val_loader is not None and (checkpoint or _due(step, training["validate_every"])):
                        record = {**record, "validation": _validation(model, val_loader, criterion, accelerator)}
                    logs.append(record)
                    if accelerator.is_main_process:
                        print(json.dumps(record), flush=True)
                    if checkpoint:
                        end = epoch_loader.end_of_dataloader
                        state.values = {"step": step, "epoch": epoch + int(end), "batches_done": 0 if end else batches_done,
                                        "generator_state": generator.get_state() if end else epoch_generator_state,
                                        "skipped_no_objective_windows": skipped_windows,
                                        "skipped_gradient_overflow_windows": skipped_gradient_overflow_windows,
                                        "skipped_nonfinite_windows": skipped_nonfinite_windows, "logs": logs,
                                        "config_sha256": config_digest, "data_sha256": metadata["data_sha256"]}
                        accelerator.save_state(str(target / f"state-step-{step}"))
                        if training["export_model_every_checkpoint"] and accelerator.is_main_process:
                            accelerator.unwrap_model(model).save_pretrained(target / f"model-step-{step}")
                if step >= steps:
                    break
            epoch += 1
            if step == epoch_start_step:
                if skipped_gradient_overflow_windows + skipped_nonfinite_windows > epoch_start_skips:
                    raise RuntimeError("epoch contains active objectives but every attempted optimizer update was skipped after nonfinite gradients")
                raise RuntimeError("epoch contains no active objective; no supervised optimizer update is possible")
        accelerator.wait_for_everyone()
        _verify_input_fingerprints(metadata, accelerator)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(target / "model")
            manifest = {**metadata, "config": config, "config_sha256": config_digest, "model": asdict(model_config),
                "loss": asdict(loss_config), "training": {**training, "resume": str(resume) if resume is not None else None},
                "optimizer": optimizer_config, "augmentation": augmentation, "validation": validation_config,
                "lr_schedule": "linear warmup over warmup_steps updates, then cosine from lr to 10% of lr at the final update; constant keeps lr",
                "parameter_groups": "backbone (LoRA/full) at learning_rate; projection/slots/decoder/heads at decoder_lr",
                "baselines": baseline, "offline_smoke": model_config.backbone == "tiny", "dry_run": dry_run,
                "steps_completed": step, "resumed_from": str(resume) if resume is not None else None, "resumed_at_step": resumed_at,
                "supervised_samples": len(samples), "validation_samples": len(validation_samples),
                "validation_excluded_flagged": validation_excluded, "augmentation_fallbacks": rows.fallbacks,
                "skipped_no_objective_windows": skipped_windows,
                "skipped_gradient_overflow_windows": skipped_gradient_overflow_windows,
                "skipped_nonfinite_windows": skipped_nonfinite_windows,
                "rejected": rejected, "validation_rejected": validation_rejected,
                "subset_limit": max_samples, "world_size": accelerator.num_processes,
                "optimizer_step_policy": "update iff global accumulated-window eligible objective count is positive and every gradient is finite; zero numeric loss is eligible",
                "reduction": "complete-field valid instances; fixed coordinate sum divided by 3; global valid count per microbatch; mean over actual flushed-window microbatches",
                "elapsed_s": time.perf_counter() - start, "trainable_parameters": sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)}
            (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
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
    parser.add_argument("--resume", type=Path, help="state-step-<n> directory of an earlier run; output must be a new directory")
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
    if args.resume is not None:
        config["training"]["resume"] = str(args.resume)
    if args.cpu:
        config["training"]["cpu"] = True
    run_training(config, args.data, args.output, validation=args.validation, dry_run=args.dry_run, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
