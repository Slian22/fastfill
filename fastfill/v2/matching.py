"""Detached assignment only within explicitly certified exchangeable groups."""
from __future__ import annotations

import json
import math

import torch
from scipy.optimize import linear_sum_assignment


def _rename(value, mapping):
    if isinstance(value, dict):
        return {mapping.get(k, k): _rename(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(v, mapping) for v in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def certify_group(objects, constraints, indices):
    """Check conditioning invariance under generators of the group's permutations."""
    fingerprints = [json.dumps({k: v for k, v in objects[i].items() if k not in {"id", "exchangeable_group"}},
                               sort_keys=True) for i in indices]
    if len(set(fingerprints)) != 1:
        raise ValueError("exchangeable group has distinct descriptions/requirements/roles")
    canonical = lambda rows: sorted(json.dumps(r, sort_keys=True) for r in rows)
    original = canonical(constraints)
    object_graph = canonical(objects)
    for i in indices[1:]:
        mapping = {objects[indices[0]]["id"]: objects[i]["id"], objects[i]["id"]: objects[indices[0]]["id"]}
        if canonical(_rename(constraints, mapping)) != original or canonical(_rename(objects, mapping)) != object_graph:
            raise ValueError("exchange would change a constraint or support role; use fixed identities")


def validate_matching_options(enabled, alpha_position, alpha_size):
    if not isinstance(enabled, bool):
        raise ValueError("matching enabled/hungarian flag must be boolean")
    weights = (alpha_position, alpha_size)
    if (any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in weights)
            or alpha_position + alpha_size <= 0):
        raise ValueError("matching weights must be native finite nonnegative numbers and not both zero")


@torch.no_grad()
def match_batch(predictions, batch, enabled=True, alpha_position=1., alpha_size=1.):
    validate_matching_options(enabled, alpha_position, alpha_size)
    valid = batch["slot_mask"]
    bsz, slots = valid.shape
    assignments = []
    for b in range(bsz):
        indices = list(range(slots))
        objects = batch["objects"][b]
        if len(objects) != int(valid[b].sum()) or not valid[b, :len(objects)].all():
            raise ValueError("slots must be contiguous and bound to every request")
        groups = {}
        for i, obj in enumerate(objects):
            if enabled and obj.get("exchangeable_group"):
                groups.setdefault(obj["exchangeable_group"], []).append(i)
        if any(len(group) > 1 for group in groups.values()) and "conditions" not in batch:
            raise ValueError("exchangeable matching requires original conditions and constraint graph")
        conditions = batch.get("conditions", [{} for _ in range(bsz)])
        for group in groups.values():
            certify_group(objects, conditions[b].get("constraints", []), group)
            if len(group) < 2:
                continue
            for field in ("position", "size"):
                if not batch["validity"][field][b, group].all():
                    raise ValueError("exchangeable assignment needs complete position and size labels")
            pos, size = predictions["position_normalized"][b, group], predictions["size"][b, group]
            gt_pos = batch["targets"]["position_normalized"][b, group]
            gt_size = batch["targets"]["size"][b, group]
            if not all(torch.isfinite(t).all() for t in (pos, size, gt_pos, gt_size)) or (size <= 0).any() or (gt_size <= 0).any():
                raise ValueError("nonfinite or nonpositive geometry in matching group")
            cost = alpha_position * (pos[:, None] - gt_pos[None]).abs().sum(-1)
            cost = cost + alpha_size * (size.log()[:, None] - gt_size.log()[None]).abs().sum(-1)
            if not torch.isfinite(cost).all():
                raise ValueError("matching cost is not finite")
            # SciPy deterministic ordered input; exact ties use its row/column order.
            rows, cols = linear_sum_assignment(cost.cpu().double().numpy())
            chosen = {group[int(row)]: group[int(col)] for row, col in zip(rows, cols)}
            indices = [chosen.get(k, old) for k, old in enumerate(indices)]
        assignments.append(indices)
    return torch.tensor(assignments, dtype=torch.long, device=valid.device)


def permute_relations(assignment, parents=None, relation_matrix=None):
    """Negative parent indices represent external supports and remain unchanged."""
    if (assignment.ndim != 1 or assignment.dtype not in {torch.int32, torch.int64} or
            not torch.equal(assignment.sort().values, torch.arange(len(assignment), device=assignment.device, dtype=assignment.dtype))):
        raise ValueError("relation assignment must be a one-to-one permutation")
    if parents is not None and parents.shape != assignment.shape:
        raise ValueError("parent targets must have one value per assigned object")
    if relation_matrix is not None and relation_matrix.shape != (len(assignment), len(assignment)):
        raise ValueError("relation matrix shape must match assigned objects")
    inverse = assignment.argsort()
    new_parents = None
    if parents is not None:
        selected = parents[assignment]
        external = selected < 0
        if ((selected >= len(assignment)) & ~external).any():
            raise ValueError("parent index out of range")
        new_parents = torch.where(external, selected, inverse[selected.clamp_min(0)])
    matrix = None if relation_matrix is None else relation_matrix[assignment][:, assignment]
    return new_parents, matrix
