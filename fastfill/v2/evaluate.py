"""All-request, stage-separated evaluation for structured and text v2 models.

Besides the reference errors the report carries label-only baselines (room centre,
per-category mean position, per-category median size, uniform yaw), mode-collapse
diagnostics compared with the ground truth on the same objects, symmetry-aware yaw
error counts per order, and a per-source breakdown of all of the above.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
from shapely.geometry import Point, Polygon
import torch

from fastfill.v2.batch import _geometry_rows, collate_samples, load_tokenizer, room_normalization
from fastfill.v2.boxes import bev_iou
from fastfill.v2.geometry import decode_yaw, wrap_yaw
from fastfill.v2.io import read_samples, safe_output, to_device
from fastfill.v2.matching import match_batch
from fastfill.v2.model import load_model, model_inputs
from fastfill.v2.schema import migrate_legacy_row, validate_condition, validate_layout
from fastfill.v2.validation import footprint, validate_scene, validate_required_levels

BASELINE_METRICS = (("room_center_position", "bottom_center_error_m"), ("category_mean_position", "bottom_center_error_m"),
                    ("category_median_size", "log_size_error"), ("uniform_yaw", "yaw_error_rad"))
COLLAPSE_SCOPE = ("request slots whose labels form a trustworthy upright box (complete position and size, finite yaw); "
                  "prediction and ground truth use the identical object set")


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
    """Float32 prediction tensors in batch slot order; only correspondence matching reads them."""
    objects = {o["id"]: o for o in layout["objects"]}
    ordered = [objects[o["id"]] for o in batch["objects"][0]]
    p = torch.tensor([[o["bottom_center_m"] for o in ordered]], dtype=torch.float32).reshape(1, -1, 3)
    s = torch.tensor([[o["target_size_local_m"] for o in ordered]], dtype=torch.float32).reshape(1, -1, 3)
    return {"position_normalized": (p - batch["origin"][:, None]) / batch["scale"][:, None],
            "size": s, "slot_mask": batch["slot_mask"]}


def _stat(values):
    return {"mean": float(np.mean(values)) if values else None, "valid_objects": len(values)}


def _pool(entries):
    """Label-weighted mean of per-request {"mean", "valid_objects"} measurements."""
    count = sum(e["valid_objects"] for e in entries)
    total = sum(e["mean"] * e["valid_objects"] for e in entries if e["mean"] is not None)
    return {"mean": total / count if count else None, "valid_objects": count}


def _yaw_error(predicted, label, order):
    period = 2 * math.pi / order
    return abs((predicted - label + period / 2) % period - period / 2)


def reference_metrics(layout, sample, *, hungarian=True, include_iou=True):
    """Reference errors follow only legal correspondence; no GT used for inference.

    Errors are computed in float64 metres from the raw layout and target rows, so an
    exact imitation of the labels scores exactly zero. Groups come from collate's
    ``batch["exchangeable_group"]``; match_batch keeps position-incomplete groups at
    fixed identity and matches size-incomplete groups on position alone.
    """
    from fastfill.v2.batch import TinyTokenizer
    batch = collate_samples([sample], TinyTokenizer(), max_length=10**8, max_objects=10**6)
    objects = batch["objects"][0]
    assignment = match_batch(_layout_tensors(layout, batch), batch, enabled=hungarian)[0].tolist()
    predicted = {o["id"]: o for o in layout["objects"]}
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    valid = batch["validity"]
    values = {"bottom_center_error_m": [], "log_size_error": [], "yaw_error_rad": [], "bev_iou": []}
    by_order = {}
    for i, j in enumerate(assignment[:len(objects)]):
        p, t = predicted[objects[i]["id"]], labels.get(objects[j]["id"], {})
        pv, sv, yv = bool(valid["position"][0, j].all()), bool(valid["size"][0, j].all()), bool(valid["yaw"][0, j])
        if pv:
            values["bottom_center_error_m"].append(float(np.linalg.norm(np.subtract(p["bottom_center_m"], t["bottom_center_m"]))))
        if sv:
            values["log_size_error"].append(float(np.abs(np.log(p["target_size_local_m"]) - np.log(t["target_size_local_m"])).mean()))
        if yv:
            order = int(batch["yaw_symmetry_order"][0, j])
            error = _yaw_error(p["yaw_rad"], t["yaw_rad"], order)
            values["yaw_error_rad"].append(error)
            by_order.setdefault(order, []).append(error)
        if include_iou and pv and sv and yv:
            values["bev_iou"].append(float(bev_iou(*(torch.tensor(v, dtype=torch.float64) for v in (
                p["bottom_center_m"], p["target_size_local_m"], p["yaw_rad"],
                t["bottom_center_m"], t["target_size_local_m"], t["yaw_rad"])))))
    groups = {}
    for i, name in enumerate(batch["exchangeable_group"][0] if hungarian else ()):
        if name:
            groups.setdefault(name, []).append(i)
    matched = [name for name, indices in groups.items() if len(indices) > 1]
    incomplete = [name for name in matched if not valid["position"][0, groups[name]].all()]
    position_only = [name for name in matched if name not in incomplete and not valid["size"][0, groups[name]].all()]
    matching_scope = ("exchangeable_complete_groups_fixed_incomplete_groups"
                      if hungarian and incomplete and len(incomplete) < len(matched) else
                      "fixed_incomplete_labels" if hungarian and incomplete else
                      "exchangeable_groups" if hungarian else "fixed")
    return {**{key: _stat(v) for key, v in values.items()},
            "yaw_error_rad_by_symmetry_order": {str(order): _stat(v) for order, v in sorted(by_order.items())},
            "matching_scope": matching_scope, "incomplete_groups": incomplete, "position_only_groups": position_only}


def _label_rows(sample):
    condition = sample["condition"]
    origin, scale = room_normalization(condition["room"])
    return condition["objects"], _geometry_rows(sample, origin, scale), origin, scale


def _collapse_counts(boxes, categories, room, origin, scale):
    """Pairwise/room statistics of upright boxes given as (bottom_center_m, size, yaw)."""
    polygons = [footprint({"_pos": p, "_size": s, "_yaw": y}) for p, s, y in boxes]
    wall = Polygon(room["floor_polygon_xy_m"]).exterior
    counts = {"objects": len(boxes), "pairs": 0, "bev_overlap_pairs": 0, "same_category_pairs": 0,
              "stacked_same_category_pairs": 0, "nearest_wall_distance_sum_m": 0., "central_quarter_objects": 0}
    for i, (p, _, _) in enumerate(boxes):
        counts["nearest_wall_distance_sum_m"] += float(wall.distance(Point(p[:2])))
        counts["central_quarter_objects"] += all(.25 <= (p[q] - origin[q]) / scale[q] <= .75 for q in (0, 1))
        for j in range(i + 1, len(boxes)):
            counts["pairs"] += 1
            intersection = polygons[i].intersection(polygons[j]).area
            union = polygons[i].area + polygons[j].area - intersection
            counts["bev_overlap_pairs"] += int(union > 0 and intersection / union > .3)
            if categories[i] == categories[j]:
                counts["same_category_pairs"] += 1
                counts["stacked_same_category_pairs"] += int(math.dist(p[:2], boxes[j][0][:2]) < .1)
    return counts


def collapse_metrics(layout, sample):
    """Mode-collapse diagnostics for the prediction and, on the same objects, the ground truth.

    Stacking uses horizontal bottom-centre distance; the central quarter is the
    middle half of each normalized room axis; wall distance is to the floor polygon.
    """
    objects, rows, origin, scale = _label_rows(sample)
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    predicted = {o["id"]: o for o in layout["objects"]}
    keep = [i for i in range(len(objects)) if all(rows["position_valid"][i]) and all(rows["size_valid"][i])
            and math.isfinite(rows["yaw"][i])]
    categories = [objects[i]["category"] for i in keep]
    room = sample["condition"]["room"]
    columns = {}
    for name, source in (("predicted", predicted), ("ground_truth", labels)):
        boxes = [(source[objects[i]["id"]]["bottom_center_m"], source[objects[i]["id"]]["target_size_local_m"],
                  source[objects[i]["id"]]["yaw_rad"]) for i in keep]
        columns[name] = _collapse_counts(boxes, categories, room, origin, scale)
    return columns


def fit_baselines(rows):
    """Per-category label pools (normalized position, size) keyed by (row index, object id)."""
    pools, keys = {"position": {}, "size": {}}, {}
    for index, sample in enumerate(rows):
        try:
            objects, geometry, _, _ = _label_rows(sample)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"baseline fit row {index} has unusable labels: {exc}") from exc
        for i, obj in enumerate(objects):
            keys[(index, obj["id"])] = len(keys)
            for field in pools:
                if all(geometry[f"{field}_valid"][i]):
                    pools[field].setdefault(obj["category"], []).append((keys[(index, obj["id"])], geometry[field][i]))
    def arrays(entries):
        return np.array([g for g, _ in entries], dtype=np.int64), np.array([v for _, v in entries], dtype=np.float64).reshape(-1, 3)
    fit = {"keys": keys}
    for field, pool in pools.items():
        fit[field] = {category: arrays(entries) for category, entries in pool.items()}
        fit[field + "_all"] = arrays([entry for entries in pool.values() for entry in entries])
    return fit


def _estimate(fit, field, category, exclude, reduce):
    """Category statistic without the label keys in ``exclude``; falls back to the all-category pool."""
    for fallback, pool in ((False, fit[field].get(category)), (True, fit[field + "_all"])):
        if pool is not None:
            # ponytail: O(n_category) masked reduction per object; use a sorted-index trick if slow.
            vectors = pool[1][~np.isin(pool[0], list(exclude))]
            if len(vectors):
                return reduce(vectors, axis=0), fallback
    return None, True


def baseline_metrics(sample, fit, *, exclude_row=None):
    """Label-only predictors on one request; exclude_row removes every label of that row (row-level leave-one-out).

    Same-room duplicates often share one asset's exact size, so excluding only the
    object's own label would let its siblings stand in for it.
    Room centre predicts the floor-level centre of the room's XY bounds. Uniform yaw is
    the analytic expectation pi / (2 * symmetry order) of a uniformly random yaw.
    """
    objects, geometry, origin, scale = _label_rows(sample)
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    values = {name: [] for name, _ in BASELINE_METRICS}
    fallbacks = 0
    center = np.array([origin[0] + scale[0] / 2, origin[1] + scale[1] / 2, origin[2]])
    own = {fit["keys"][(exclude_row, o["id"])] for o in objects if (exclude_row, o["id"]) in fit["keys"]}
    for i, obj in enumerate(objects):
        target = labels.get(obj["id"], {})
        if all(geometry["position_valid"][i]):
            gt = np.asarray(target["bottom_center_m"], dtype=np.float64)
            values["room_center_position"].append(float(np.linalg.norm(center - gt)))
            estimate, fallback = _estimate(fit, "position", obj["category"], own, np.mean)
            if estimate is not None:
                fallbacks += fallback
                values["category_mean_position"].append(float(np.linalg.norm(np.asarray(origin) + np.asarray(scale) * estimate - gt)))
        if all(geometry["size_valid"][i]):
            estimate, fallback = _estimate(fit, "size", obj["category"], own, np.median)
            if estimate is not None:
                fallbacks += fallback
                values["category_median_size"].append(float(np.abs(np.log(estimate) - np.log(target["target_size_local_m"])).mean()))
        if geometry["yaw_valid"][i]:
            values["uniform_yaw"].append(math.pi / (2 * geometry["symmetry"][i]))
    return {**{name: {metric: _stat(values[name])} for name, metric in BASELINE_METRICS},
            "category_fallback_objects": fallbacks}


def evaluate_layout(layout, sample, *, resolver=None, commit_in_memory=False, hungarian=True,
                    asset_retries=2, repair_calls=0, repair_step_m=.25, max_seconds=10., required_levels=("bbox",)):
    sample = migrate_legacy_row(sample)
    condition = sample["condition"]
    validate_condition(condition)
    validate_layout(layout, condition)
    target = validate_scene(condition, layout["objects"], required_levels=required_levels)
    result = {"model": {"schema_success": True, "requested_ids_exactly_once": True, "positive_valid_size": True,
                         "target_geometry_valid": target["ok"], "reference": None},
              "raw_prediction": layout, "target_validation": target, "collapse": None, "asset": None, "system": None,
              "actual_resolved": None, "final_output": None}
    try:
        result["model"]["reference"] = reference_metrics(layout, sample, hungarian=hungarian)
        result["collapse"] = collapse_metrics(layout, sample)
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


def _model_summary(outcomes):
    n = len(outcomes)
    model = {key: sum(bool(o.get("model", {}).get(key)) for o in outcomes) / n for key in
             ("schema_success", "requested_ids_exactly_once", "positive_valid_size", "target_geometry_valid")}
    references = [o["model"]["reference"] for o in outcomes if o.get("model", {}).get("reference")]
    reference = {key: {**_pool([r[key] for r in references]),
                       "scope": "parsed predictions with valid labels; schema failures included in success denominator"}
                 for key in ("bottom_center_error_m", "log_size_error", "yaw_error_rad", "bev_iou")}
    orders = sorted({order for r in references for order in r["yaw_error_rad_by_symmetry_order"]})
    reference["yaw_error_rad_by_symmetry_order"] = {
        order: _pool([r["yaw_error_rad_by_symmetry_order"][order] for r in references if order in r["yaw_error_rad_by_symmetry_order"]])
        for order in orders}
    model["reference"] = reference
    return model


def _baseline_summary(outcomes):
    rows = [o["baselines"] for o in outcomes if o.get("baselines")]
    if not rows:
        return None
    return {**{name: {metric: _pool([r[name][metric] for r in rows])} for name, metric in BASELINE_METRICS},
            "category_fallback_objects": sum(r["category_fallback_objects"] for r in rows), "requests": len(rows),
            "scope": "all evaluation rows with valid labels, independent of model parsing"}


def _collapse_summary(outcomes):
    rows = [o["collapse"] for o in outcomes if o.get("collapse")]
    if not rows:
        return None
    def rates(column):
        c = {key: sum(r[column][key] for r in rows) for key in rows[0][column]}
        ratio = lambda a, b: c[a] / c[b] if c[b] else None
        return {"bev_overlap_rate_iou_gt_0.3": ratio("bev_overlap_pairs", "pairs"),
                "duplicate_stacking_rate_lt_0.10m": ratio("stacked_same_category_pairs", "same_category_pairs"),
                "mean_nearest_wall_distance_m": ratio("nearest_wall_distance_sum_m", "objects"),
                "central_quarter_fraction": ratio("central_quarter_objects", "objects"),
                "objects": c["objects"], "pairs": c["pairs"], "same_category_pairs": c["same_category_pairs"]}
    return {"predicted": rates("predicted"), "ground_truth": rates("ground_truth"), "requests": len(rows), "scope": COLLAPSE_SCOPE}


def summarize(outcomes, *, asset_evaluation_requested=False, commit_evaluation_requested=False, baseline_fit_source=None):
    n = len(outcomes)
    if not n:
        raise ValueError("no evaluation requests")
    model = _model_summary(outcomes)
    baselines = _baseline_summary(outcomes)
    if baselines is not None:
        baselines["fit"] = baseline_fit_source
    sources = sorted({str(o.get("provenance", {}).get("source")) for o in outcomes})
    by_source = {}
    for source in sources:
        subset = [o for o in outcomes if str(o.get("provenance", {}).get("source")) == source]
        by_source[source] = {"requests": len(subset), "model": _model_summary(subset),
                             "baselines": _baseline_summary(subset), "collapse": _collapse_summary(subset)}
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
              "baselines": baselines, "collapse": _collapse_summary(outcomes), "by_source": by_source,
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
                   max_seconds=10., max_new_tokens=2048, required_levels=("bbox",), baseline_fit=None):
    validate_required_levels(required_levels)
    if (checkpoint is None) == (predictions is None):
        raise ValueError("provide exactly one checkpoint or prediction JSONL")
    target = safe_output(output)
    samples = read_samples(data, max_samples=max_samples)
    if baseline_fit:
        with Path(baseline_fit).open() as stream:
            fit = fit_baselines(json.loads(line) for line in stream if line.strip())
        fit_source = f"baseline_fit_file:{baseline_fit}"
    else:
        fit, fit_source = fit_baselines(samples), "evaluation_set_leave_one_out"
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
                      fastfill_latency_status=("failure" if "error" in result else "success") if checkpoint else "unobserved",
                      baselines=baseline_metrics(sample, fit, exclude_row=None if baseline_fit else i))
        outcomes.append(result)
    report = summarize(outcomes, asset_evaluation_requested=resolver is not None,
                       commit_evaluation_requested=commit_in_memory, baseline_fit_source=fit_source)
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
    p.add_argument("--baseline-fit", type=Path, help="JSONL rows for the category baselines; default: evaluation set, leave-one-out")
    args = vars(p.parse_args(argv))
    args["hungarian"] = not args.pop("fixed_correspondence")
    print(json.dumps(run_evaluation(**args), indent=2))


if __name__ == "__main__":
    main()
