"""All-request, stage-separated evaluation for structured and text v2 models."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from fastfill.v2.batch import collate_samples, load_tokenizer
from fastfill.v2.boxes import bev_iou
from fastfill.v2.geometry import decode_yaw, encode_yaw, wrap_yaw
from fastfill.v2.io import read_samples, safe_output, to_device
from fastfill.v2.matching import match_batch
from fastfill.v2.model import load_model, model_inputs
from fastfill.v2.schema import validate_condition, validate_layout
from fastfill.v2.validation import validate_scene, validate_required_levels


def serialize_predictions(predictions, batch):
    position = predictions["position_normalized"] * batch["scale"][:, None] + batch["origin"][:, None]
    yaw = decode_yaw(predictions["yaw_logits"], predictions["yaw_residuals"])
    result = []
    for b, objects in enumerate(batch["objects"]):
        layout = {"schema_version": "fastfill.v2", "objects": [
            {"id": obj["id"], "target_size_local_m": predictions["size"][b, i].detach().cpu().tolist(),
             "bottom_center_m": position[b, i].detach().cpu().tolist(),
             "yaw_rad": float(wrap_yaw(float(yaw[b, i].detach().cpu())))} for i, obj in enumerate(objects)]}
        validate_layout(layout, batch["conditions"][b])
        result.append(layout)
    return result


@torch.no_grad()
def predict_layout(model, tokenizer, condition, *, max_length=4096, device="cpu"):
    validate_condition(condition)
    batch = to_device(collate_samples([{"condition": condition}], tokenizer,
        max_length=max_length, max_objects=model.config.max_objects), device)
    model.eval()
    return serialize_predictions(model(**model_inputs(batch)), batch)[0]


def _layout_tensors(layout, batch):
    objects = {o["id"]: o for o in layout["objects"]}
    ordered = [objects[o["id"]] for o in batch["objects"][0]]
    p = torch.tensor([[o["bottom_center_m"] for o in ordered]], dtype=torch.float32).reshape(1, -1, 3)
    s = torch.tensor([[o["target_size_local_m"] for o in ordered]], dtype=torch.float32).reshape(1, -1, 3)
    yaw = torch.tensor([[o["yaw_rad"] for o in ordered]], dtype=torch.float32)
    k, r = encode_yaw(yaw, 12)
    logits = torch.nn.functional.one_hot(k, 12).float() * 100
    residuals = torch.zeros_like(logits).scatter(-1, k[..., None], r[..., None])
    return {"position_normalized": (p - batch["origin"][:, None]) / batch["scale"][:, None],
            "size": s, "yaw_logits": logits, "yaw_residuals": residuals, "slot_mask": batch["slot_mask"]}, p, yaw


def reference_metrics(layout, sample, *, hungarian=True, include_iou=True):
    """Reference errors follow only legal correspondence; no GT used for inference."""
    from fastfill.v2.batch import TinyTokenizer
    batch = collate_samples([sample], TinyTokenizer(), max_length=10**8, max_objects=10**6)
    predictions, positions, yaw = _layout_tensors(layout, batch)
    groups = {}
    for i, obj in enumerate(batch["objects"][0]):
        if obj.get("exchangeable_group"):
            groups.setdefault(obj["exchangeable_group"], []).append(i)
    incomplete = [name for name, indices in groups.items() if len(indices) > 1 and any(
                  not batch["validity"][field][0, indices].all() for field in ("position", "size"))]
    use_hungarian = hungarian and not incomplete
    assignment = match_batch(predictions, batch, enabled=use_hungarian)[0]
    values = {"bottom_center_error_m": [], "log_size_error": [], "yaw_error_rad": [], "bev_iou": []}
    target = batch["targets"]
    valid = batch["validity"]
    for i, j in enumerate(assignment.tolist()[:len(layout["objects"])]):
        pv, sv, yv = bool(valid["position"][0, j].all()), bool(valid["size"][0, j].all()), bool(valid["yaw"][0, j])
        if pv:
            gt_position = target["position_normalized"][0, j] * batch["scale"][0] + batch["origin"][0]
            values["bottom_center_error_m"].append(float((positions[0, i] - gt_position).norm()))
        if sv:
            values["log_size_error"].append(float((predictions["size"][0, i].log() - target["size"][0, j].log()).abs().mean()))
        if yv:
            period = 2 * math.pi / int(batch["yaw_symmetry_order"][0, j])
            error = abs((float(yaw[0, i] - target["yaw"][0, j]) + period / 2) % period - period / 2)
            values["yaw_error_rad"].append(error)
        if include_iou and pv and sv and yv:
            values["bev_iou"].append(float(bev_iou(positions[0, i], predictions["size"][0, i], yaw[0, i],
                          gt_position, target["size"][0, j], target["yaw"][0, j])))
    return {**{key: {"mean": float(np.mean(v)) if v else None, "valid_objects": len(v)} for key, v in values.items()},
            "matching_scope": "fixed_incomplete_labels" if hungarian and incomplete else
                              ("exchangeable_groups" if hungarian else "fixed"),
            "incomplete_groups": incomplete}


def evaluate_layout(layout, sample, *, resolver=None, commit_in_memory=False, hungarian=True,
                    asset_retries=2, repair_calls=0, repair_step_m=.25, max_seconds=10., required_levels=("bbox",)):
    condition = sample["condition"]
    validate_condition(condition)
    validate_layout(layout, condition)
    target = validate_scene(condition, layout["objects"], required_levels=required_levels)
    result = {"model": {"schema_success": True, "requested_ids_exactly_once": True, "positive_valid_size": True,
                         "target_geometry_valid": target["ok"], "reference": None},
              "raw_prediction": layout, "target_validation": target, "asset": None, "system": None,
              "actual_resolved": None, "final_output": None}
    try:
        result["model"]["reference"] = reference_metrics(layout, sample, hungarian=hungarian)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        # Reference-label eligibility is independent of model schema and runtime.
        result["reference_error"] = {"type": type(exc).__name__, "message": str(exc)}
    if resolver is not None:
        from fastfill.v2.runtime import AtomicMemoryHost, BoundedTranslationRepair, RuntimeBudget, run_pipeline
        host = AtomicMemoryHost() if commit_in_memory else None
        runtime = run_pipeline(condition, layout, resolver, host=host,
            budget=RuntimeBudget(max_asset_retries=asset_retries, max_repair_calls=repair_calls, max_seconds=max_seconds),
            repair=BoundedTranslationRepair(repair_step_m) if repair_calls else None,
            required_levels=required_levels,
            expected_world_version=0 if host else None,
            idempotency_key=str(sample["provenance"].get("scene_id", "evaluation")))
        result.update(asset=runtime["metrics"]["asset"], system=runtime["metrics"]["system"], runtime=runtime)
        result["actual_resolved"] = runtime.get("actual_first_pass_objects")
        result["final_output"] = runtime.get("final_objects")
    return result


def summarize(outcomes, *, asset_evaluation_requested=False, commit_evaluation_requested=False):
    n = len(outcomes)
    if not n:
        raise ValueError("no evaluation requests")
    model = {key: sum(bool(o.get("model", {}).get(key)) for o in outcomes) / n for key in
             ("schema_success", "requested_ids_exactly_once", "positive_valid_size", "target_geometry_valid")}
    reference = {}
    for key in ("bottom_center_error_m", "log_size_error", "yaw_error_rad", "bev_iou"):
        measurements = [o["model"]["reference"][key] for o in outcomes if o.get("model", {}).get("reference")]
        count = sum(m["valid_objects"] for m in measurements)
        reference[key] = {"mean": sum(m["mean"] * m["valid_objects"] for m in measurements if m["mean"] is not None) / count if count else None,
                          "valid_objects": count, "scope": "parsed predictions with valid labels; schema failures included in success denominator"}
    model["reference"] = reference
    latency = [o["fastfill_latency_ms"] for o in outcomes if o.get("fastfill_latency_ms") is not None]
    latency_by_outcome = {}
    for status in ("success", "failure"):
        values = [o["fastfill_latency_ms"] for o in outcomes if o.get("fastfill_latency_ms") is not None
                  and o.get("fastfill_latency_status") == status]
        latency_by_outcome[status] = {"requests": len(values), "p50": float(np.percentile(values, 50)) if values else None,
                                     "p95": float(np.percentile(values, 95)) if values else None}
    wall = [o["evaluation_wall_time_ms"] for o in outcomes if o.get("evaluation_wall_time_ms") is not None]
    asset_evaluated = asset_evaluation_requested or any(o.get("asset") is not None for o in outcomes)
    inference_failures = sum("error" in o for o in outcomes)
    system_failures = sum(not o.get("runtime", {}).get("ok", False) for o in outcomes) if asset_evaluated else None
    report = {"requests": n, "failed_requests": sum((not o.get("runtime", {}).get("ok", False))
              if asset_evaluated else ("error" in o or not o.get("model", {}).get("target_geometry_valid", False)) for o in outcomes),
              "inference_failed_requests": inference_failures, "system_failed_requests": system_failures,
              "stage_denominator": n, "stage_failures": {
                  "model_schema": sum(not o.get("model", {}).get("schema_success", False) for o in outcomes),
                  "target_geometry": sum(not o.get("model", {}).get("target_geometry_valid", False) for o in outcomes),
                  "asset_resolution": sum(not o.get("asset") or o["asset"]["retrieval_coverage"] < 1 for o in outcomes) if asset_evaluated else None,
                  "actual_first_pass": sum(not o.get("asset", {}).get("first_pass_actual_geometry_validation", False)
                     if o.get("asset") else True for o in outcomes) if asset_evaluated else None}, "model": model,
              "latency_ms": {"fastfill_p50": float(np.percentile(latency, 50)) if latency else None,
                             "fastfill_p95": float(np.percentile(latency, 95)) if latency else None,
                             "fastfill_observed_requests": len(latency), "fastfill_by_outcome": latency_by_outcome,
                             "fastfill_scope": "All observed checkpoint generations, including failures; supplied prediction generation time is unknown",
                             "evaluation_wall_time_p50": float(np.percentile(wall, 50)) if wall else None,
                             "evaluation_wall_time_p95": float(np.percentile(wall, 95)) if wall else None}}
    if asset_evaluated:
        report["asset"] = {key: sum(float(o.get("asset", {}).get(key, 0) or 0) for o in outcomes if o.get("asset")) / n
                           for key in ("retrieval_coverage", "capability_satisfaction", "first_pass_actual_geometry_validation", "resolver_calls")}
        differences = [o["asset"]["target_actual_log_size_l1_mean"] for o in outcomes if o.get("asset") and o["asset"]["target_actual_log_size_l1_mean"] is not None]
        report["asset"]["target_actual_log_size_l1_mean"] = float(np.mean(differences)) if differences else None
        report["system"] = {key: sum(float(o.get("system", {}).get(key, 0) or 0) for o in outcomes if o.get("system")) / n
                            for key in ("repaired_success", "fallback_rate", "final_commit", "asset_retries", "repair_calls")}
        if not commit_evaluation_requested and not any(o.get("system", {}).get("commit_attempted") for o in outcomes if o.get("system")):
            report["system"]["final_commit"] = None
        runtime_latency = [o["system"]["end_to_end_latency_ms"] for o in outcomes if
                           o.get("system") and o["system"].get("end_to_end_latency_ms") is not None]
        end_to_end = [o["fastfill_latency_ms"]+o["system"]["end_to_end_latency_ms"] for o in outcomes if
                      o.get("fastfill_latency_ms") is not None and o.get("system") and
                      o["system"].get("end_to_end_latency_ms") is not None]
        report["latency_ms"].update(runtime_p50=float(np.percentile(runtime_latency, 50)) if runtime_latency else None,
                                    runtime_p95=float(np.percentile(runtime_latency, 95)) if runtime_latency else None,
                                    end_to_end_p50=float(np.percentile(end_to_end, 50)) if end_to_end else None,
                                    end_to_end_p95=float(np.percentile(end_to_end, 95)) if end_to_end else None)
        report["system"]["solver_success"] = None
    return report


def run_evaluation(data, output, *, checkpoint=None, baseline="structured", predictions=None,
                   catalog=None, device="cpu", max_length=4096, max_samples=None,
                   commit_in_memory=False, hungarian=True, asset_retries=2, repair_calls=0, repair_step_m=.25,
                   max_seconds=10., max_new_tokens=2048, required_levels=("bbox",)):
    validate_required_levels(required_levels)
    if (checkpoint is None) == (predictions is None):
        raise ValueError("provide exactly one checkpoint or prediction JSONL")
    target = safe_output(output)
    samples = read_samples(data, max_samples=max_samples)
    model = tokenizer = None
    if checkpoint:
        if baseline == "structured":
            model = load_model(checkpoint, device=device)
            tokenizer_path = Path(checkpoint).parent / "tokenizer"
            tokenizer = load_tokenizer("tiny" if model.config.backbone == "tiny" else str(tokenizer_path), local_files_only=True)
        else:
            from fastfill.v2.text_sft import load_text_model
            model, tokenizer = load_text_model(checkpoint, device=device)
    supplied = None
    if predictions:
        with Path(predictions).open() as stream:
            supplied = [line.rstrip("\n") for line in stream if line.strip()]
        if len(supplied) != len(samples):
            raise ValueError("predictions must have one row per selected request, including failures")
    resolver = None
    if catalog:
        from fastfill.v2.serve import load_catalog
        resolver = load_catalog(catalog)
    if commit_in_memory and resolver is None:
        raise ValueError("commit evaluation requires an asset catalog")
    outcomes = []
    torch.set_num_threads(2)
    for i, sample in enumerate(samples):
        start = time.perf_counter()
        raw, fastfill_latency = None, None
        try:
            if supplied is not None:
                raw = supplied[i]
                def reject_constant(value):
                    raise ValueError(f"nonfinite prediction JSON constant {value}")
                parsed = json.loads(raw, parse_constant=reject_constant)
                layout = parsed.get("layout", parsed) if isinstance(parsed, dict) else parsed
            elif baseline == "structured":
                raw = layout = predict_layout(model, tokenizer, sample["condition"], max_length=max_length, device=device)
            else:
                from fastfill.v2.text_sft import generate_text
                raw = generate_text(model, tokenizer, sample["condition"], max_length=max_length,
                                    max_new_tokens=max_new_tokens, device=device)
                layout = json.loads(raw)
            fastfill_latency = (time.perf_counter() - start) * 1000 if checkpoint else None
            result = evaluate_layout(layout, sample, resolver=resolver, commit_in_memory=commit_in_memory,
                                    hungarian=hungarian, asset_retries=asset_retries, repair_calls=repair_calls,
                                    repair_step_m=repair_step_m, max_seconds=max_seconds, required_levels=required_levels)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            if checkpoint and fastfill_latency is None:
                fastfill_latency = (time.perf_counter() - start) * 1000
            result = {"error": {"type": type(exc).__name__, "message": str(exc)}, "raw_prediction": raw,
                      "model": {"schema_success": False, "requested_ids_exactly_once": False, "positive_valid_size": False}}
        result.update(row=i, provenance=sample["provenance"], evaluation_wall_time_ms=(time.perf_counter() - start) * 1000,
                      fastfill_latency_ms=fastfill_latency,
                      fastfill_latency_status=("failure" if "error" in result else "success") if checkpoint else "unobserved")
        outcomes.append(result)
    report = summarize(outcomes, asset_evaluation_requested=resolver is not None,
                       commit_evaluation_requested=commit_in_memory)
    report.update(baseline=baseline if checkpoint else "supplied_predictions", subset_limit=max_samples,
                  evaluation_matching="exchangeable_groups" if hungarian else "fixed", commit_scope="in_memory" if commit_in_memory else "not_attempted",
                  runtime_budget={"asset_retries": asset_retries, "repair_calls": repair_calls,
                                  "repair_step_m": repair_step_m, "max_seconds": max_seconds}, required_levels=list(required_levels),
                  geometry_level="upright_obb_bev_iou", source_root_modified=False)
    target.mkdir(parents=True, exist_ok=False)
    (target / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (target / "outcomes.jsonl").open("x") as stream:
        for outcome in outcomes:
            stream.write(json.dumps(outcome, allow_nan=False) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--baseline", choices=["structured", "text"], default="structured")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--fixed-correspondence", action="store_true")
    p.add_argument("--commit-in-memory", action="store_true")
    p.add_argument("--asset-retries", type=int, default=2)
    p.add_argument("--repair-calls", type=int, default=0)
    p.add_argument("--repair-step-m", type=float, default=.25)
    p.add_argument("--required-levels", nargs="+", choices=("bbox", "mesh", "physics", "solver"), default=["bbox"])
    p.add_argument("--max-seconds", type=float, default=10.)
    args = vars(p.parse_args(argv))
    args["hungarian"] = not args.pop("fixed_correspondence")
    print(json.dumps(run_evaluation(**args), indent=2))


if __name__ == "__main__":
    main()
