"""All-request, stage-separated evaluation for structured and text v2 models.

Besides the reference errors the report carries label-only baselines (room centre,
per-category mean position, per-category median size, uniform yaw), mode-collapse
diagnostics (every predicted object, the label-complete ground truth, and the predicted
objects matched to it, each with the out-of-room fraction), symmetry-aware yaw error counts per order, and a per-source
breakdown of all of the above.

``--projection full minimal`` also evaluates the three-field projection
(``project_minimal``: ``batch.render_minimal_condition`` with the training
augmentation's regrouping) over boundary-known axis-aligned rectangular rooms; the report's top
level is the first projection and ``projections`` holds the others. Objects whose
``validity.size_axis_swap_allowed`` is true score size and yaw under the
box-equivalent minimum over (sx, sy, yaw + k pi/2 with sx/sy swapped for odd k),
with the plain-convention errors reported beside them. Target validation is
counted per check code as pass / violation / unknown. ``max_length`` comes from the
checkpoint (``io.load_checkpoint_config``) unless given explicitly; model settings
come from its ``model_config.json``. Reports record the checkpoint, its binding,
and the data and code sha256.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import math
from pathlib import Path
import time

import numpy as np
from shapely.geometry import Point, Polygon
import torch

from fastfill.v2.batch import (AUGMENT_DEFAULTS, OBJECT_BUDGET_ERROR, _geometry_rows, augment_sample, collate_samples, minimal_form_eligible,
                               load_tokenizer, room_normalization)
from fastfill.v2.boxes import bev_iou
from fastfill.v2.geometry import decode_yaw, wrap_yaw
from fastfill.v2.io import fingerprint, load_checkpoint_config, read_samples, run_metadata, safe_output, to_device
from fastfill.v2.matching import match_batch
from fastfill.v2.model import load_model, model_inputs
from fastfill.v2.schema import migrate_legacy_row, validate_condition, validate_layout
from fastfill.v2.validation import CHECK_STATUSES, effective_support_requests, footprint, validate_scene, validate_required_levels

BASELINE_METRICS = (("room_center_position", "bottom_center_error_m"), ("category_mean_position", "bottom_center_error_m"),
                    ("category_median_size", "log_size_error"), ("uniform_yaw", "yaw_error_rad"))
REFERENCE_METRICS = ("bottom_center_error_m", "log_size_error", "yaw_error_rad", "bev_iou",
                     "log_size_error_plain_convention", "yaw_error_rad_plain_convention")
COLLAPSE_SCOPE = ("predicted: every requested object (a legal exchange of IDs cannot change it); ground truth: request "
                  "slots whose labels form a trustworthy upright box (complete position and size, finite yaw); "
                  "predicted_matched: the predicted objects the reference matching assigns to those slots, the same "
                  "objects as ground truth, so compare predicted_matched with ground_truth")
COLLAPSE_KEYS = ("objects", "pairs", "bev_overlap_pairs", "same_category_pairs", "stacked_same_category_pairs",
                 "nearest_wall_distance_sum_m", "central_quarter_objects", "out_of_room_objects")
OUT_OF_ROOM_M = .05  # bottom centre this far outside the floor polygon counts as out of the room
PROJECTIONS = ("full", "minimal")
# augment_sample with only the minimal form enabled is the K5 projection of a row: shared condition
# renderer, floor declarations dropped (z learned), exchangeable groups recomputed by the build rule.
MINIMAL_PROJECTION = {key: (1. if key == "minimal_form_p" else 0. if key.endswith("_p") else False) for key in AUGMENT_DEFAULTS}
DEFAULT_MAX_LENGTH = 4096


def project_minimal(sample):
    """The three-field projection of an evaluation row, or None unless ``batch.minimal_form_eligible``
    (a boundary-known axis-aligned rectangle; such rows count as ``skipped_non_rectangular_rooms``)."""
    sample = migrate_legacy_row(sample)
    if not minimal_form_eligible(sample["condition"]["room"]):
        return None
    return augment_sample(sample, MINIMAL_PROJECTION, torch.Generator().manual_seed(0))


def collapse_score(counts):
    """Collapse score: BEV overlap rate (IoU > 0.3) + central-quarter fraction; lower is better.

    No objects -> None; objects without pairs contribute only the central fraction.
    It cannot see a layout pushed against or outside the walls (such a layout scores 0):
    read it only together with ``out_of_room_fraction`` and the reference errors.
    """
    if not counts["objects"]:
        return None
    return (counts["bev_overlap_pairs"] / counts["pairs"] if counts["pairs"] else 0.) + counts["central_quarter_objects"] / counts["objects"]


def batch_collapse_counts(predictions, batch, assignment=None):
    """Summed ``_collapse_counts`` of a collated batch, as the matching ``collapse_metrics`` column.

    Model outputs (yaw logits) are the predicted column: every request slot; with
    ``assignment`` (prediction slot -> label slot, e.g. the criterion's) only the slots
    assigned to label-complete ones, the predicted_matched column. Labels passed with
    their yaw in radians as ``predictions["yaw"]`` are the ground-truth column: only the
    label-complete slots.
    """
    origin, scale = batch["origin"].detach().float().cpu(), batch["scale"].detach().float().cpu()
    position = (predictions["position_normalized"].detach().float().cpu() * scale[:, None] + origin[:, None]).tolist()
    size = predictions["size"].detach().float().cpu().tolist()
    labels = "yaw" in predictions
    yaw = (predictions["yaw"].detach().float() if labels else decode_yaw(
        predictions["yaw_logits"].detach().float(), predictions["yaw_residuals"].detach().float())).cpu().tolist()
    complete = (batch["slot_mask"] & batch["validity"]["position"].all(-1) & batch["validity"]["size"].all(-1)
                & torch.isfinite(batch["targets"]["yaw"])).cpu()
    keep = complete if labels else batch["slot_mask"].cpu() if assignment is None else complete.gather(1, assignment.cpu())
    totals = dict.fromkeys(COLLAPSE_KEYS, 0)
    for b, objects in enumerate(batch["objects"]):
        slots = keep[b].nonzero().flatten().tolist()
        counts = _collapse_counts([(position[b][i], size[b][i], yaw[b][i]) for i in slots], [objects[i]["category"] for i in slots],
                                  batch["conditions"][b]["room"], origin[b].tolist(), scale[b].tolist())
        for key in totals:
            totals[key] += counts[key]
    return totals


GRID_DECODES = ("spread", "argmax")
PARENT_LATTICE = np.stack(np.meshgrid(*[np.linspace(-1, 1, 5)] * 2, indexing="ij"), -1).reshape(-1, 2)


def spread_grid_xy(logits, residuals, grid, size, yaw, position, room, *, requests=None, fixed_z=None, yaw_logits=None,
                   yaw_residuals=None, top_k=64, max_overlap=.15, margin=.02, near_wall=.3):
    """Collision-aware decoding of one scene's grid head (meters in, meters out); returns (xy, yaw, z).

    ``room`` may list ``fixed_objects`` (condition rows: bottom_center_m, size_local_m, yaw_rad): they never move,
    block cells like placed objects and can carry raised ones. ``requests`` (the slots' effective requests,
    ``validation.effective_support_requests``) supply ``id`` and ``support_parent``; ``fixed_z`` marks slots whose
    z the condition fixes (``fixed_position_mask[..., 2]``). Without them every object is undeclared.

    Objects are placed floor-standing before raised ones, parents before their declared children, largest
    footprint first; each takes its most probable cell whose footprint stays in the room (a footprint at most
    ``margin`` past a wall slides back inside first) and overlaps no placed
    or fixed object that shares its height interval (overlap / smaller footprint <= max_overlap). Identical
    requests, whose argmax cells coincide, therefore spread over their next most probable cells. If no
    candidate qualifies, the least overlapping one is used (for a declared child: on its parent).

    Declarations beat the predicted height: an object whose z is fixed stands at that z, one declared on
    "floor" stands on the floor, one declared on "wall" keeps its predicted z, and one declared on a request
    or fixed object stands on the parent's top: at its most probable top_k cell whose centre lies inside the
    parent's (placed) footprint, else at the free point of a 5 x 5 lattice over that footprint (inset by its
    own half extents) nearest its argmax decode, else at the least overlapping of these, so the declared
    support holds even at the cost of a collision.

    With yaw logits, a floor-standing object whose candidate footprint ends within ``near_wall`` of a wall
    takes its most probable yaw bin within 45 degrees of facing away from that wall (its back to the wall,
    as 97-99% of wall-adjacent training furniture); its footprint is then checked at that yaw.

    An undeclared raised object (predicted bottom > 0.15 m above the floor) rests on a placed or fixed
    floor-standing object: it takes its most probable cell (at least 10% as likely as its best) whose centre lies inside
    such an object's footprint and sits on the highest such top; with no such cell it goes to the floor at its
    most probable free cell.
    """
    # ponytail: rotated footprints are compared by their axis-aligned bounds (exact for 90-degree yaws,
    # conservative otherwise); switch to shapely polygons if oblique furniture matters.
    n, cells = logits.shape
    index = torch.arange(cells)
    centres = torch.stack((torch.div(index, grid, rounding_mode="floor"), index % grid), -1).double()
    xy_norm = (centres + .5) / grid + residuals.double() / (2 * grid)  # n x cells x 2
    origin, scale = np.asarray(room["origin"], float), np.asarray(room["scale"], float)
    xy = xy_norm.numpy() * scale[:2] + origin[:2]
    size, yaw, position = np.asarray(size, float).reshape(-1, 3), np.asarray(yaw, float), np.asarray(position, float)
    requests = [{}] * n if requests is None else list(requests)
    fixed_z = np.zeros(n, bool) if fixed_z is None else np.asarray(fixed_z, bool)
    if len(requests) != n or fixed_z.shape != (n,):
        raise ValueError("requests and fixed_z need one entry per slot")

    def halves(angle, box):
        c, s = np.abs(np.cos(angle)), np.abs(np.sin(angle))
        return .5 * np.stack((c * box[..., 0] + s * box[..., 1], s * box[..., 0] + c * box[..., 1]), -1)

    fixed = room.get("fixed_objects", ())
    fixed_ids = {obj["id"]: j for j, obj in enumerate(fixed)}
    fixed_xy = np.array([obj["bottom_center_m"][:2] for obj in fixed], float).reshape(-1, 2)
    fixed_bottom = np.array([obj["bottom_center_m"][2] for obj in fixed], float)
    fixed_size = np.array([obj["size_local_m"] for obj in fixed], float).reshape(-1, 3)
    fixed_half = halves(np.array([obj["yaw_rad"] for obj in fixed], float), fixed_size)
    fixed_top, fixed_area = fixed_bottom + fixed_size[:, 2], 4 * fixed_half.prod(-1)
    fixed_standing = fixed_bottom - room["floor_z"] <= .15
    slots = {r["id"]: i for i, r in enumerate(requests) if "id" in r}
    parent = [r.get("support_parent") for r in requests]
    if any(p is not None and p not in slots and p not in fixed_ids and p not in ("floor", "wall") for p in parent):
        raise ValueError("unknown support parent")

    def depth(i):
        chain = 0
        while parent[i] in slots:
            i, chain = slots[parent[i]], chain + 1
            if chain > n:
                raise ValueError("support graph contains a cycle")
        return chain

    half = halves(yaw, size)
    area = 4 * half[:, 0] * half[:, 1]
    low = position[:, 2]
    lo, hi = np.asarray(room["bounds"][0], float), np.asarray(room["bounds"][1], float)
    on_object = np.array([p in slots or p in fixed_ids for p in parent], bool)
    on_floor = fixed_z | np.array([p == "floor" for p in parent], bool)
    on_wall = np.array([p == "wall" for p in parent], bool)
    raised = on_object | (~on_floor & (low - room["floor_z"] > .15))
    stand = np.where(fixed_z | on_wall, low, float(room["floor_z"]))  # where a floor-standing or wall-hung object stands
    order = sorted(range(n), key=lambda i: (bool(raised[i]), depth(i), -area[i]))
    ranked = logits.double().argsort(-1, descending=True)[:, :top_k].numpy()
    chosen, chosen_yaw, chosen_z = xy[np.arange(n), ranked[:, 0]].copy(), yaw.copy(), low.copy()
    if yaw_logits is not None:
        yaw_logits, yaw_residuals = np.asarray(yaw_logits, float), np.asarray(yaw_residuals, float)
        bins = yaw_logits.shape[-1]
        bin_centres = np.arange(bins) * 2 * np.pi / bins
        away = np.array([0., np.pi, np.pi / 2, -np.pi / 2])  # from walls x=lo, x=hi, y=lo, y=hi into the room

    def boxes(js, f=slice(None)):  # placed slots js, then fixed objects f: centre, half extents, bottom, top, footprint area
        js = np.asarray(js, int)
        return (np.concatenate((chosen[js], fixed_xy[f])), np.concatenate((half[js], fixed_half[f])),
                np.concatenate((chosen_z[js], fixed_bottom[f])), np.concatenate((chosen_z[js] + size[js, 2], fixed_top[f])),
                np.concatenate((area[js], fixed_area[f])))

    placed = []
    for i in order:
        cand = xy[i, ranked[i]]  # k x 2
        cand_yaw = np.full(len(cand), yaw[i])
        if on_object[i]:  # the declared parent's footprint and top; a lattice over that top follows the top_k cells
            if parent[i] in slots:
                j = slots[parent[i]]
                base, base_half, top = chosen[j], half[j], chosen_z[j] + size[j, 2]
            else:
                j = fixed_ids[parent[i]]
                base, base_half, top = fixed_xy[j], fixed_half[j], fixed_top[j]
            on_top = base + PARENT_LATTICE * np.maximum(base_half - half[i], 0)  # includes the footprint centre
            on_top = on_top[np.argsort(((on_top - cand[0]) ** 2).sum(-1), kind="stable")]  # nearest the argmax decode first
            cand, cand_yaw = np.vstack((cand, on_top)), np.append(cand_yaw, np.full(len(on_top), yaw[i]))
        elif yaw_logits is not None and not raised[i]:
            gaps = np.stack((cand[:, 0] - half[i, 0] - lo[0], hi[0] - cand[:, 0] - half[i, 0],
                             cand[:, 1] - half[i, 1] - lo[1], hi[1] - cand[:, 1] - half[i, 1]), -1)
            wall = gaps.argmin(-1)
            facing = (bin_centres[None] - away[wall][:, None] + np.pi) % (2 * np.pi) - np.pi
            allowed = np.abs(facing) <= np.pi / 4 + 1e-6
            best = np.where(allowed, yaw_logits[i][None], -np.inf).argmax(-1)
            snapped = bin_centres[best] + yaw_residuals[i, best] * np.pi / bins
            cand_yaw = np.where(gaps.min(-1) < near_wall, (snapped + np.pi) % (2 * np.pi) - np.pi, cand_yaw)
        cand_half = halves(cand_yaw, size[i])  # k x 2
        over = (np.maximum(lo + cand_half - cand, 0) + np.maximum(cand + cand_half - hi, 0)).max(-1)
        snap = (over > 0) & (over <= margin)  # within the margin past a wall: slide back inside (the validator allows 1e-4)
        cand = np.where(snap[:, None], np.clip(cand, lo + cand_half, hi - cand_half), cand)
        outside = over > margin
        if on_object[i]:
            cand_z, supported = np.full(len(cand), top), (np.abs(cand - base) <= base_half).all(-1)
        else:
            cand_z, supported = np.full(len(cand), stand[i]), np.zeros(len(cand), bool)
        if raised[i] and not on_object[i] and not on_wall[i]:
            s_xy, s_half, _, s_top, _ = boxes([j for j in placed if not raised[j]], fixed_standing)
            inside = (np.abs(cand[:, None] - s_xy) <= s_half).all(-1)  # k x m centre in footprint
            highest = np.where(inside, s_top, -np.inf).max(-1, initial=-np.inf)
            supported = np.isfinite(highest)
            cand_z = np.where(supported, highest, cand_z)
            supported &= (logits[i, ranked[i]] - logits[i, ranked[i, 0]]).numpy() >= -math.log(10)  # >= 10% of the best cell
        o_xy, o_half, o_bottom, o_top, o_area = boxes(placed)
        gap = (np.minimum(cand[:, None] + cand_half[:, None], o_xy + o_half)
               - np.maximum(cand[:, None] - cand_half[:, None], o_xy - o_half))
        inter = np.clip(gap, 0, None).prod(-1)
        vertical = (np.minimum(cand_z[:, None] + size[i, 2], o_top) - np.maximum(cand_z[:, None], o_bottom)) > margin
        worst = np.where(vertical, inter / np.minimum(area[i], o_area), 0).max(-1, initial=0.)
        free = ~outside & (worst <= max_overlap)
        ok = np.flatnonzero(free & supported if raised[i] else free)
        if not len(ok) and not on_object[i]:
            ok = np.flatnonzero(free)
        # least overlapping fallback; a declared child only among points on its parent (its lattice always is)
        pick = ok[0] if len(ok) else int(np.argmin(np.where(supported | ~on_object[i], worst + outside, np.inf)))
        chosen[i], chosen_yaw[i], half[i], chosen_z[i] = cand[pick], cand_yaw[pick], cand_half[pick], cand_z[pick]
        placed.append(i)
    return chosen, chosen_yaw, chosen_z


def _room_frame(batch, b):
    room = batch["conditions"][b]["room"]
    polygon = np.asarray(room["floor_polygon_xy_m"], float)
    floor = room.get("floor_z_m")
    return {"origin": batch["origin"][b].cpu().tolist(), "scale": batch["scale"][b].cpu().tolist(),
            "bounds": (polygon.min(0), polygon.max(0)), "floor_z": float(floor) if floor is not None else float(batch["origin"][b, 2]),
            "fixed_objects": room.get("fixed_objects", [])}


def serialize_predictions(predictions, batch, *, grid_decode="spread"):
    """Validated layouts of a collated batch. ``spread`` (grid head only) re-decodes each scene with
    ``spread_grid_xy`` under its declared supports, fixed objects and fixed z; ``argmax`` keeps the raw head."""
    if grid_decode not in GRID_DECODES:
        raise ValueError(f"grid_decode must be one of {GRID_DECODES}")
    position = predictions["position_normalized"] * batch["scale"][:, None] + batch["origin"][:, None]
    yaw = decode_yaw(predictions["yaw_logits"], predictions["yaw_residuals"])
    if grid_decode == "spread" and "position_cell_logits" in predictions:
        position, yaw = position.detach().cpu().double().clone(), yaw.detach().cpu().double().clone()
        grid = int(round(predictions["position_cell_logits"].shape[-1] ** .5))
        for b, objects in enumerate(batch["objects"]):
            n = len(objects)
            if n:
                xy, spread_yaw, spread_z = spread_grid_xy(
                    predictions["position_cell_logits"][b, :n].detach().cpu().float(),
                    predictions["position_cell_residuals"][b, :n].detach().cpu().float(), grid,
                    predictions["size"][b, :n].detach().cpu().double().numpy(), yaw[b, :n].detach().cpu().double().numpy(),
                    position[b, :n].numpy(), _room_frame(batch, b),
                    requests=effective_support_requests(batch["conditions"][b]),
                    fixed_z=batch["fixed_position_mask"][b, :n, 2].cpu().numpy(),
                    yaw_logits=predictions["yaw_logits"][b, :n].detach().cpu().float().numpy(),
                    yaw_residuals=predictions["yaw_residuals"][b, :n].detach().cpu().float().numpy())
                position[b, :n, :2] = torch.from_numpy(xy)
                position[b, :n, 2] = torch.from_numpy(spread_z)
                yaw[b, :n] = torch.from_numpy(spread_yaw).to(yaw.dtype)
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
def predict_layout(model, tokenizer, condition, *, max_length=4096, device="cpu", grid_decode="spread"):
    validate_condition(condition)
    batch = to_device(collate_samples([{"condition": condition}], tokenizer,
        max_length=max_length, max_objects=model.config.max_objects), device)
    model.eval()
    return serialize_predictions(model(**model_inputs(batch)), batch, grid_decode=grid_decode)[0]


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


def _log_size_error(predicted, label):
    return float(np.abs(np.log(predicted) - np.log(label)).mean())


def _box_equivalent_errors(p, t, size_valid, yaw_valid):
    """(size, yaw) errors of the label rewriting (sx, sy, sz, yaw + k pi/2), sy/sx swapped for odd k, k = 0..3,
    whose summed valid errors are smallest; the same candidate set as the training loss. None for invalid fields."""
    best = None
    for k in range(4):
        sx, sy, sz = t["target_size_local_m"] if size_valid else (1., 1., 1.)
        size = _log_size_error(p["target_size_local_m"], (sy, sx, sz) if k % 2 else (sx, sy, sz)) if size_valid else None
        yaw = _yaw_error(p["yaw_rad"], t["yaw_rad"] + k * math.pi / 2, 1) if yaw_valid else None
        if best is None or (size or 0.) + (yaw or 0.) < (best[0] or 0.) + (best[1] or 0.):
            best = (size, yaw)
    return best


def _matched(layout, sample, hungarian):
    """The one-request batch and its legal assignment (prediction slot -> label slot)."""
    from fastfill.v2.batch import TinyTokenizer
    batch = collate_samples([sample], TinyTokenizer(), max_length=10**8, max_objects=10**6)
    return batch, match_batch(_layout_tensors(layout, batch), batch, enabled=hungarian)[0].tolist()


def reference_metrics(layout, sample, *, hungarian=True, include_iou=True):
    """Reference errors follow only legal correspondence; no GT used for inference.

    Errors are computed in float64 metres from the raw layout and target rows, so an
    exact imitation of the labels scores exactly zero. Groups come from collate's
    ``batch["exchangeable_group"]``; match_batch keeps position-incomplete groups at
    fixed identity and matches size-incomplete groups on position alone.
    ``log_size_error`` / ``yaw_error_rad`` use the box-equivalent minimum for objects
    with ``size_axis_swap_allowed``; the ``*_plain_convention`` keys keep the label as
    written (yaw modulo its symmetry order; modulo pi for swap objects, whose order 4
    only stands for the swap) for every object.
    """
    batch, assignment = _matched(layout, sample, hungarian)
    objects = batch["objects"][0]
    predicted = {o["id"]: o for o in layout["objects"]}
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    valid = batch["validity"]
    values = {key: [] for key in REFERENCE_METRICS}
    by_order, box_equivalent = {}, 0
    for i, j in enumerate(assignment[:len(objects)]):
        p, t = predicted[objects[i]["id"]], labels.get(objects[j]["id"], {})
        pv, sv, yv = bool(valid["position"][0, j].all()), bool(valid["size"][0, j].all()), bool(valid["yaw"][0, j])
        if pv:
            values["bottom_center_error_m"].append(float(np.linalg.norm(np.subtract(p["bottom_center_m"], t["bottom_center_m"]))))
        order, swap = int(batch["yaw_symmetry_order"][0, j]), bool(batch["size_axis_swap_allowed"][0, j])
        size_error = _log_size_error(p["target_size_local_m"], t["target_size_local_m"]) if sv else None
        # A swap object's order 4 encodes the axis swap; as written, its box only repeats every pi.
        yaw_error = _yaw_error(p["yaw_rad"], t["yaw_rad"], 2 if swap else order) if yv else None
        values["log_size_error_plain_convention"] += [size_error] if sv else []
        values["yaw_error_rad_plain_convention"] += [yaw_error] if yv else []
        if (sv or yv) and swap:
            size_error, yaw_error = _box_equivalent_errors(p, t, sv, yv)
            box_equivalent += 1
        if sv:
            values["log_size_error"].append(size_error)
        if yv:
            values["yaw_error_rad"].append(yaw_error)
            by_order.setdefault(order, []).append(yaw_error)
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
            "box_equivalent_objects": box_equivalent, "matching_scope": matching_scope, "incomplete_groups": incomplete, "position_only_groups": position_only}


def _label_rows(sample):
    condition = sample["condition"]
    origin, scale = room_normalization(condition["room"])
    return condition["objects"], _geometry_rows(sample, origin, scale), origin, scale


def _collapse_counts(boxes, categories, room, origin, scale):
    """Pairwise/room statistics of upright boxes given as (bottom_center_m, size, yaw)."""
    polygons = [footprint({"_pos": p, "_size": s, "_yaw": y}) for p, s, y in boxes]
    floor = Polygon(room["floor_polygon_xy_m"])
    counts = {"objects": len(boxes), "pairs": 0, "bev_overlap_pairs": 0, "same_category_pairs": 0,
              "stacked_same_category_pairs": 0, "nearest_wall_distance_sum_m": 0., "central_quarter_objects": 0,
              "out_of_room_objects": 0}
    for i, (p, _, _) in enumerate(boxes):
        counts["nearest_wall_distance_sum_m"] += float(floor.exterior.distance(Point(p[:2])))
        counts["out_of_room_objects"] += int(floor.distance(Point(p[:2])) > OUT_OF_ROOM_M)  # 0 inside the polygon
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


def collapse_metrics(layout, sample, *, hungarian=True):
    """Mode-collapse diagnostics: the prediction (every requested object), the ground truth
    (label-complete objects) and predicted_matched (the predicted objects that the reference
    matching assigns to the label-complete slots: the ground truth's objects, predicted).

    The predicted column never depends on which slots carry labels, so a legal exchange
    of IDs leaves it unchanged; predicted_matched follows the legal assignment, so it is
    unchanged too except inside groups the matching keeps at fixed identity. Compare
    predicted_matched, not predicted, with the ground truth: sources with masked labels
    (e.g. size-masked MansionWorld) are in predicted only. Stacking uses horizontal
    bottom-centre distance; the central quarter is the middle half of each normalized room
    axis; wall distance is to the floor polygon; an object is out of the room when its
    bottom centre lies more than ``OUT_OF_ROOM_M`` outside the floor polygon.
    """
    objects, rows, origin, scale = _label_rows(sample)
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    predicted = {o["id"]: o for o in layout["objects"]}
    keep = [i for i in range(len(objects)) if all(rows["position_valid"][i]) and all(rows["size_valid"][i])
            and math.isfinite(rows["yaw"][i])]
    complete = set(keep)
    matched = [i for i, j in enumerate(_matched(layout, sample, hungarian)[1][:len(objects)]) if j in complete]
    room = sample["condition"]["room"]
    columns = {}
    for name, source, indices in (("predicted", predicted, range(len(objects))), ("predicted_matched", predicted, matched),
                                  ("ground_truth", labels, keep)):
        boxes = [(source[objects[i]["id"]]["bottom_center_m"], source[objects[i]["id"]]["target_size_local_m"],
                  source[objects[i]["id"]]["yaw_rad"]) for i in indices]
        columns[name] = _collapse_counts(boxes, [objects[i]["category"] for i in indices], room, origin, scale)
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
    Median size scores swap-allowed objects box-equivalently, like the model.
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
                label = target["target_size_local_m"]
                error = _log_size_error(estimate, label)
                if geometry["swap"][i]:
                    error = min(error, _log_size_error(estimate, (label[1], label[0], label[2])))
                values["category_median_size"].append(error)
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
        result["collapse"] = collapse_metrics(layout, sample, hungarian=hungarian)
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
                 for key in REFERENCE_METRICS}
    reference["box_equivalent_objects"] = sum(r["box_equivalent_objects"] for r in references)
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
                "central_quarter_fraction": ratio("central_quarter_objects", "objects"), "collapse_score": collapse_score(c),
                "out_of_room_fraction": ratio("out_of_room_objects", "objects"),
                "objects": c["objects"], "pairs": c["pairs"], "same_category_pairs": c["same_category_pairs"]}
    return {**{column: rates(column) for column in rows[0]}, "requests": len(rows), "scope": COLLAPSE_SCOPE}


def _check_counts(outcomes):
    """Per check code: how many target-validation checks passed, were violated or stayed unknown.

    Rows whose own labels exceed the declared height (``provenance.height_conflict``)
    count their ceiling checks separately, so a known data conflict is not read as a model violation.
    """
    table = {}
    for outcome in outcomes:
        conflict = bool(outcome.get("provenance", {}).get("height_conflict"))
        for check in (outcome.get("target_validation") or {}).get("checks", ()):
            code = check["code"] + ("_on_height_conflict_rows" if conflict and check["code"] == "ceiling" else "")
            table.setdefault(code, dict.fromkeys(CHECK_STATUSES, 0))[check["status"]] += 1
    return dict(sorted(table.items()))


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
              # of the inference failures: requests with more objects than the checkpoint's max_objects
              "over_capacity_requests": sum(OBJECT_BUDGET_ERROR in o.get("error", {}).get("message", "") for o in outcomes),
              "target_validation_checks": _check_counts(outcomes),
              "target_validation_not_run": sum(not o.get("target_validation") for o in outcomes),
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
                   catalog=None, device="cpu", max_length=None, max_samples=None,
                   commit_in_memory=False, hungarian=True, asset_retries=2, repair_calls=0, repair_step_m=.25,
                   max_seconds=10., max_new_tokens=2048, required_levels=("bbox",), baseline_fit=None,
                   projections=("full",), grid_decode="spread"):
    """Evaluate every request once per projection; ``max_length=None`` binds to the checkpoint."""
    validate_required_levels(required_levels)
    if (checkpoint is None) == (predictions is None):
        raise ValueError("provide exactly one checkpoint or prediction JSONL")
    projections = tuple(projections)
    if not projections or len(set(projections)) != len(projections) or set(projections) - set(PROJECTIONS):
        raise ValueError(f"projections must be distinct names from {PROJECTIONS}")
    if predictions is not None and projections != ("full",):
        raise ValueError("supplied predictions answer the full condition; evaluate other projections from a checkpoint")
    target = safe_output(output)
    samples = read_samples(data, max_samples=max_samples)
    if baseline_fit:
        with Path(baseline_fit).open() as stream:
            fit = fit_baselines(json.loads(line) for line in stream if line.strip())
        fit_source = f"baseline_fit_file:{baseline_fit}"
    else:
        fit, fit_source = fit_baselines(samples), "evaluation_set_leave_one_out"
    model = tokenizer = binding = None
    max_length_source = "argument" if max_length is not None else None
    if checkpoint:
        binding = load_checkpoint_config(checkpoint)
        if max_length is None and binding["max_length"] is not None:
            max_length, max_length_source = binding["max_length"], binding["source"]
        if baseline == "structured":
            model = load_model(checkpoint, device=device)
            tokenizer_path = Path(checkpoint).parent / "tokenizer"
            tokenizer = load_tokenizer("tiny" if model.config.backbone == "tiny" else str(tokenizer_path), local_files_only=True)
        else:
            from fastfill.v2.text_sft import load_text_model
            model, tokenizer = load_text_model(checkpoint, device=device)
    if max_length is None:
        max_length, max_length_source = DEFAULT_MAX_LENGTH, "default"
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
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))  # CPU inference may use more threads

    def evaluate_row(i, sample):
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
                raw = layout = predict_layout(model, tokenizer, sample["condition"], max_length=max_length, device=device,
                                                    grid_decode=grid_decode)
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
        return result

    reports, outcomes = {}, {}
    for projection in projections:
        rows = list(enumerate(samples)) if projection == "full" else [
            (i, projected) for i, projected in enumerate(map(project_minimal, samples)) if projected is not None]
        outcomes[projection] = [evaluate_row(i, sample) for i, sample in rows]
        reports[projection] = summarize(outcomes[projection], asset_evaluation_requested=resolver is not None,
                                        commit_evaluation_requested=commit_in_memory,
                                        baseline_fit_source=fit_source) if rows else {"requests": 0}
        reports[projection].update(projection=projection, skipped_non_rectangular_rooms=len(samples) - len(rows))
    report = {**reports[projections[0]], "projections": {name: reports[name] for name in projections[1:]}}
    metadata = run_metadata(data)
    report.update(baseline=baseline if checkpoint else "supplied_predictions", subset_limit=max_samples,
                  evaluation_matching="exchangeable_groups" if hungarian else "fixed", commit_scope="in_memory" if commit_in_memory else "not_attempted",
                  runtime_budget={"asset_retries": asset_retries, "repair_calls": repair_calls,
                                  "repair_step_m": repair_step_m, "max_seconds": max_seconds}, required_levels=list(required_levels),
                  geometry_level="upright_obb_bev_iou", source_root_modified=False,
                  checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None, checkpoint_binding=binding,
                  max_length=max_length if checkpoint else None, max_length_source=max_length_source if checkpoint else None,
                  grid_decode=grid_decode if checkpoint else None,
                  predictions_sha256=fingerprint(predictions) if predictions else None,
                  **{key: metadata[key] for key in ("data_path", "data_sha256", "implementation_sha256", "code_commit", "code_dirty")})
    target.mkdir(parents=True, exist_ok=False)
    (target / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for index, projection in enumerate(projections):
        with (target / ("outcomes.jsonl" if not index else f"outcomes-{projection}.jsonl")).open("x") as stream:
            for outcome in outcomes[projection]:
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
    p.add_argument("--max-length", type=int, help="default: the checkpoint's training max_length")
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
    p.add_argument("--projection", nargs="+", choices=PROJECTIONS, default=["full"], dest="projections",
                   help="full condition and/or its three-field projection (rectangular rooms); the first is the report's top level")
    p.add_argument("--grid-decode", choices=GRID_DECODES, default="spread",
                   help="grid position head: collision-aware spread (default) or plain per-object argmax")
    args = vars(p.parse_args(argv))
    args["hungarian"] = not args.pop("fixed_correspondence")
    print(json.dumps(run_evaluation(**args), indent=2))


if __name__ == "__main__":
    main()
