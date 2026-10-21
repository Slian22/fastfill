"""Bounded, pose-only OFFLINE TRAINING teacher for frozen estimated dimensions.

Never use this as inference postprocessing or as an OOF fitting/admission rule.
The caller supplies frozen estimates keyed by immutable original/bound IR IDs.
Only size substitution and movable poses change; no resizing, pruning, relabeling,
support reassignment, canonicalization, or prompt construction happens here.

Qualification uses fastfill.validate.check defaults: center-based XY support
(overhang is allowed), full footprints for boundary/collisions, and zero main,
severe, and fixed-blocking pairs in addition to check.valid. Tuck/contained/window
collisions retain the checker's category semantics. Fixed supports are rejected:
that checker resolves support parents only among movables.

Search is deterministic greedy improvement, not a feasibility proof. Each floor
root has at most max_candidates trials TOTAL across max_passes, including child
trials. Defaults are 200/4. Original/nearby/wall/grid positions and original plus
cardinal yaws are tried, moving support subtrees rigidly. Positions are rounded to
cm and yaw to integer degrees stored as IR radians BEFORE every check. Failed
searches return all objects and explicit reasons; malformed input is rejected
with a detached copy (size substitution is only possible for valid estimates).
Rejection receipts encode nonfinite floats as {"nonfinite": "nan"/"inf"/"-inf"}
so they can be persisted with strict JSON. No finite dimension is changed.
At most 256 movable objects enter graph/geometry preflight; larger inventories
are retained in a rejection receipt without attempting the teacher.
"""
import copy
import hashlib
import itertools
import math
from collections.abc import Mapping

from shapely.geometry import Polygon

from fastfill.scene import footprint
from fastfill.validate import check, support_contains


POLICY = {"checker": "fastfill.validate.check", "support": "center",
          "acceptance": "valid and zero main, severe, fixed_blocking",
          "position_decimals": 2, "yaw_degree_step": 1}
MAX_OBJECTS = 256


def _ledger_safe(value):
    """Copy rejection data, tagging nonfinite floats instead of emitting JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {key: _ledger_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_ledger_safe(item) for item in value)
    if isinstance(value, list):
        return [_ledger_safe(item) for item in value]
    return value


def _number(value, positive=False):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and abs(value) < 1e6 and math.isfinite(value)
            and (not positive or value > 0))


def _vector(value, n=3, positive=False):
    return (isinstance(value, (list, tuple)) and len(value) == n
            and all(_number(v, positive) for v in value))


def _freeze(room, sizes):
    objects, fixed = room.get("objects"), room.get("fixed") or []
    if not isinstance(objects, list) or not isinstance(fixed, list):
        raise ValueError("invalid_inventory")
    records = objects + fixed
    if any(not isinstance(o, dict) or not isinstance(o.get("id"), str)
           or not o["id"] for o in records):
        raise ValueError("invalid_id")
    ids = [o["id"] for o in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_id")
    if not isinstance(sizes, Mapping) or set(sizes) != {o["id"] for o in objects}:
        raise ValueError("estimated_size_inventory_mismatch")
    if any(not _vector(s, positive=True) for s in sizes.values()):
        raise ValueError("invalid_estimated_size")
    objects = [{**o, "size": copy.deepcopy(list(sizes[o["id"]]))} for o in objects]
    return {**room, "objects": objects}


def _validate(room):
    for o in room["objects"] + (room.get("fixed") or []):
        if not (_vector(o.get("pos")) and _vector(o.get("size"), positive=True)
                and _number(o.get("yaw")) and "category" in o
                and (o["category"] is None or isinstance(o["category"], str))):
            raise ValueError("invalid_geometry_or_category:" + o["id"])
    boundary = room.get("boundary")
    if boundary is not None:
        if (not isinstance(boundary, (list, tuple)) or len(boundary) < 3
                or not all(_vector(p, 2) for p in boundary)):
            raise ValueError("invalid_boundary")
        polygon = Polygon(boundary)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("invalid_boundary")
    if room.get("boundary_type") not in (None, "polygon", "hull", "proxy"):
        raise ValueError("invalid_boundary_type")
    if room.get("height") is not None and not _number(room["height"], positive=True):
        raise ValueError("invalid_height")


def _identity(oid):
    return hashlib.sha256(oid.encode()).digest(), oid


def _forest(room):
    objects = {o["id"]: o for o in room["objects"]}
    paths = {}
    for oid in objects:
        path, cur = [], oid
        while cur is not None:
            if not isinstance(cur, str) or cur not in objects:
                raise ValueError("parent_not_movable:" + oid)
            if cur in path:
                raise ValueError("support_cycle:" + oid)
            path.append(cur)
            cur = objects[cur].get("parent")
        paths[oid] = path
    order = sorted(objects, key=lambda oid: (len(paths[oid]), _identity(oid)))
    members = {oid: [other for other in order if oid in paths[other]] for oid in order}
    roots = sorted((oid for oid in order if len(paths[oid]) == 1), key=lambda oid: (
        -sum(math.prod(objects[x]["size"][:2]) for x in members[oid]), _identity(oid)))
    return order, members, roots


def _yaw(yaw):
    return math.radians(round(math.degrees(yaw % math.tau)) % 360)


def _rotate(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def _replace(room, changed):
    return {**room, "objects": [changed.get(o["id"], o) for o in room["objects"]]}


def _normalize(room, order):
    original = {o["id"]: o for o in room["objects"]}
    changed = {}
    for oid in order:
        obj = original[oid]
        x, y, _ = obj["pos"]
        parent = changed.get(obj.get("parent"))
        z = 0
        if parent is not None:
            old_parent = original[parent["id"]]
            x += parent["pos"][0] - old_parent["pos"][0]
            y += parent["pos"][1] - old_parent["pos"][1]
            z = parent["pos"][2] + parent["size"][2]
            if not support_contains(parent, (x, y)):
                dx, dy = _rotate(x - parent["pos"][0], y - parent["pos"][1], -parent["yaw"])
                dx = max(-parent["size"][0] / 2, min(parent["size"][0] / 2, dx))
                dy = max(-parent["size"][1] / 2, min(parent["size"][1] / 2, dy))
                dx, dy = _rotate(dx, dy, parent["yaw"])
                x, y = parent["pos"][0] + dx, parent["pos"][1] + dy
        changed[oid] = {**obj, "pos": [round(v, 2) for v in (x, y, z)], "yaw": _yaw(obj["yaw"])}
    return _replace(room, changed)


def _move(room, oid, members, pose):
    index = {o["id"]: o for o in room["objects"]}
    root, changed = index[oid], {}
    x, y, yaw = pose
    delta = yaw - root["yaw"]
    for key in members[oid]:
        obj = index[key]
        dx, dy = _rotate(obj["pos"][0] - root["pos"][0], obj["pos"][1] - root["pos"][1], delta)
        changed[key] = {**obj, "pos": [round(x + dx, 2), round(y + dy, 2), obj["pos"][2]],
                        "yaw": _yaw(obj["yaw"] + delta)}
    return _replace(room, changed)


def _positions(room, oid, members, yaw):
    index = {o["id"]: o for o in room["objects"]}
    obj = index[oid]
    x, y = obj["pos"][:2]
    yield x, y, yaw
    parent = index.get(obj.get("parent"))
    turned = _move(room, oid, members, (x, y, yaw))
    points = [p for o in turned["objects"] if o["id"] in members[oid] for p in footprint(o)]
    bx0, by0 = min(p[0] for p in points), min(p[1] for p in points)
    bx1, by1 = max(p[0] for p in points), max(p[1] for p in points)
    if parent is not None:
        for u, v in itertools.product((0, -.45, .45), repeat=2):
            dx, dy = _rotate(u * parent["size"][0], v * parent["size"][1], parent["yaw"])
            yield parent["pos"][0] + dx, parent["pos"][1] + dy, yaw
    elif room.get("boundary"):
        lo_x, lo_y, hi_x, hi_y = Polygon(room["boundary"]).bounds
        # Subtree footprint extents, never dimensions, determine translation intervals.
        left, right = lo_x + x - bx0, hi_x + x - bx1
        bottom, top = lo_y + y - by0, hi_y + y - by1
        if left <= right and bottom <= top:
            yield min(max(x, left), right), min(max(y, bottom), top), yaw
            for u, v in itertools.product((0, 1, .5, .25, .75), repeat=2):
                yield left + u * (right - left), bottom + v * (top - bottom), yaw
    step = max(.1, min(bx1 - bx0, by1 - by0) / 2)
    for scale in (1, 2, 4):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            yield x + dx * step * scale, y + dy * step * scale, yaw


def _candidates(room, oid, members):
    obj = next(o for o in room["objects"] if o["id"] == oid)
    yaws = dict.fromkeys([obj["yaw"], 0, math.pi / 2, math.pi, 3 * math.pi / 2])
    streams = [_positions(room, oid, members, yaw) for yaw in yaws]
    seen = set()
    for batch in itertools.zip_longest(*streams):
        for pose in batch:
            if pose is not None:
                pose = round(pose[0], 2), round(pose[1], 2), _yaw(pose[2])
                if pose not in seen:
                    seen.add(pose)
                    yield pose


def _counts(audit):
    return {**{k: len(audit[k]) for k in ("oob", "support_fail", "ceiling", "fixed_blocking")},
            **{k: len(audit["collisions"][k]) for k in ("main", "severe")}}


def _score(audit):
    counts = _counts(audit)
    return sum(counts.values()) + int(not audit["valid"]), counts["severe"]


def _problems(audit):
    return (set(audit["oob"] + audit["support_fail"] + audit["ceiling"])
            | {oid for k in ("main", "severe") for pair in audit["collisions"][k] for oid in pair}
            | {pair[0] for pair in audit["fixed_blocking"]})


def _search(room, members, roots, max_candidates, max_passes):
    audit = check(room)
    counts, seen = {root: 0 for root in roots}, {oid: set() for oid in members}
    passes = 0
    for pass_index in range(max_passes):
        if _score(audit)[0] == 0 or not any(n < max_candidates for n in counts.values()):
            break
        passes += 1
        for root in roots:
            allowance = math.ceil((max_candidates - counts[root]) / (max_passes - pass_index))
            units = [oid for oid in members[root] if set(members[oid]) & _problems(audit)]
            for index, oid in enumerate(units):
                budget = math.ceil(allowance / (len(units) - index))
                best, best_audit, used = room, audit, 0
                # Parent movement changes the meaning of absolute child candidates.
                context = tuple((o["id"], *o["pos"], o["yaw"]) for o in room["objects"])
                for pose in itertools.islice(_candidates(room, oid, members), max_candidates):
                    if used >= budget:
                        break
                    signature = context, pose
                    if signature in seen[oid]:
                        continue
                    seen[oid].add(signature)
                    trial = _move(room, oid, members, pose)
                    trial_audit = check(trial)
                    used += 1
                    if _score(trial_audit) < _score(best_audit):
                        best, best_audit = trial, trial_audit
                    if _score(best_audit)[0] == 0:
                        break
                room, audit = best, best_audit
                counts[root] += used
                allowance -= used
                if _score(audit)[0] == 0:
                    return room, counts, passes
    return room, counts, passes


def _finish(room, frozen, diagnostics):
    rounded = {**room, "objects": [{**o, "pos": [round(v, 2) for v in o["pos"]],
                                    "yaw": _yaw(o["yaw"])} for o in room["objects"]]}
    audit = check(rounded)  # Authoritative check of the actual returned precision.
    reasons = [key for key, count in _counts(audit).items() if count]
    if not audit["valid"] and not reasons:
        reasons.append("invalid")
    locked = lambda r: [{k: v for k, v in o.items() if k not in ("pos", "yaw")} for o in r["objects"]]
    if locked(rounded) != locked(frozen):
        reasons.append("inventory_or_size_changed")
    result = {"status": "rejected" if reasons else "ok", "room": rounded,
              "diagnostics": {**diagnostics, "after": audit, "after_counts": _counts(audit),
                              "reasons": reasons + (["search_exhausted"] if reasons else [])}}
    return _ledger_safe(result) if reasons else result


def repair(room, estimated_sizes, *, max_candidates=200, max_passes=4):
    """Return JSON-serializable {status, room, diagnostics}; never mutate inputs.

    max_candidates is a per-floor-root TOTAL trial limit, shared with descendants.
    Explicit nonnegative integer limits override the defaults (zero disables search).
    More than MAX_OBJECTS movables reject before graph/geometry preflight, with
    the full frozen inventory retained. Nonfinite rejection values are tagged
    dictionaries for strict-JSON ledgers, never replacement numeric dimensions.
    Audit ``before`` uses estimated sizes at original poses, never GT dimensions.
    Rejection means this bounded teacher failed; it does not prove infeasibility.
    """
    output = copy.deepcopy(room)
    diagnostics = {"policy": dict(POLICY), "before": None, "after": None,
                   "before_counts": None, "after_counts": None, "reasons": [],
                   "candidate_counts": {}, "passes": 0,
                   "limits": {"max_candidates": max_candidates, "max_passes": max_passes,
                              "max_objects": MAX_OBJECTS}}
    try:
        if not isinstance(output, dict):
            raise ValueError("invalid_room")
        output = _freeze(output, estimated_sizes)
        if any(type(n) is not int or n < 0 for n in (max_candidates, max_passes)):
            raise ValueError("invalid_search_limits")
        if len(output["objects"]) > MAX_OBJECTS:
            raise ValueError("movable_limit_exceeded")
        _validate(output)
        order, members, roots = _forest(output)
    except ValueError as error:
        return _ledger_safe({"status": "rejected", "room": output,
                             "diagnostics": {**diagnostics, "reasons": [str(error)]}})
    before = check(output)
    normalized = _normalize(output, order)
    repaired, counts, passes = _search(normalized, members, roots, max_candidates, max_passes)
    return _finish(repaired, output, {**diagnostics, "before": before, "before_counts": _counts(before),
                                      "candidate_counts": counts, "passes": passes})


repair_pose_targets = repair
