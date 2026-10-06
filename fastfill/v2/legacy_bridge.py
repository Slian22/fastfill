"""Migrate selected v3.2 geometry, retaining lineage and target-independent IDs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from fastfill.scene import canonical, norm_cat, rot90, target_json
from fastfill.v2.schema import validate_condition


def _key(seed, uid, ident):
    return hashlib.sha256(json.dumps([seed, uid, ident], separators=(",", ":")).encode()).hexdigest()


def _legacy_map(prepared, row):
    tagged = {**prepared, "objects": [{**o, "_source_id": o["id"]} for o in prepared["objects"]],
              "fixed": [{**o, "_source_id": o["id"]} for o in prepared.get("fixed", [])]}
    expected = json.loads(row["messages"][-1]["content"])
    for k in range(4):
        candidate = canonical(rot90(deepcopy(tagged), k))
        if json.loads(target_json(candidate)) == expected:
            return {o["id"]: o["_source_id"] for o in candidate["objects"] + candidate.get("fixed", [])}
    raise ValueError("frozen target cannot be reproduced from selected IR preparation")


def _constraint(c, mapping):
    kind = c[0]
    result = {"type": kind, "object_id": mapping[c[1]]}
    if kind in {"near", "faces", "on"}:
        result["parent_id" if kind == "on" else "target_id"] = mapping[c[2]]
    if kind == "near":
        result["max_distance_m"] = c[3]
    if kind == "between":
        result["target_ids"] = [mapping[c[2]], mapping[c[3]]]
    if kind not in {"near", "faces", "on", "between", "against_wall"}:
        raise ValueError(f"unsupported frozen constraint {kind}")
    return result


def _frame(raw):
    meta = raw.get("meta", {})
    dz = 0.
    if raw["source"] == "MultiScan":
        if meta.get("n_floor_objects_outside_structure", 0):
            raise ValueError("target-expanded MultiScan boundary cannot enter condition")
        dz = float(meta["floor_z"]) - float(meta["floor_height"])
    room = {"frame": "right_handed_z_up", "floor_polygon_xy_m": deepcopy(raw["boundary"]),
            "boundary_known": raw.get("boundary_type") == "polygon", "boundary_quality": raw.get("boundary_type", "unknown"),
            "floor_known": raw["source"] != "MultiScan", "floor_z_m": 0. if raw["source"] != "MultiScan" else None,
            "height_m": raw.get("height")}
    if raw["source"] == "MultiScan":
        room["height_m"] = raw["height"] + dz if meta.get("height_reliable") and raw.get("height") else None
    if raw.get("room_type"):
        room["room_type"] = raw["room_type"]
    return room, dz


def _validity(obj, raw, front_policy):
    numeric = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    upright = not obj.get("tilted", False)
    size = upright and all(numeric(v) and v > 0 for v in obj["size"])
    position = upright and all(numeric(v) for v in obj["pos"])
    front = obj.get("front_known", raw.get("meta", {}).get("front_known")) is True
    if front_policy == "strict":
        front = front and raw["source"] == "MultiScan"
    yaw = upright and front and numeric(obj.get("yaw"))
    if raw["source"] in {"OptiScene_holodeck", "MansionWorld"}:
        size = False  # Padded boxes and annotation footprints cannot certify local asset extents.
    return [position] * 3, [size] * 3, yaw


def _groups(objects, constraints, validity):
    from fastfill.v2.matching import certify_group
    candidates = {}
    for i, obj in enumerate(objects):
        if all(validity["position"][i]) and all(validity["size"][i]):
            signature = json.dumps({k: v for k, v in obj.items() if k != "id"}, sort_keys=True)
            candidates.setdefault(signature, []).append(i)
    out = [dict(o) for o in objects]
    for number, indices in enumerate(candidates.values()):
        if len(indices) < 2:
            continue
        try:
            certify_group(out, constraints, indices)
        except ValueError:
            continue
        out = [{**o, "exchangeable_group": f"anonymous_{number}"} if i in indices else o for i, o in enumerate(out)]
    return out


def convert_selected_room(raw, prepared, row, split, *, seed=42, front_policy="strict"):
    """Selected objects come from legacy prep; labels come from unrounded IR.

    strict yaw only admits independently documented MultiScan semantic fronts.
    legacy-convention is an explicit lower-evidence ablation, never a claim of
    per-asset semantic-front verification. No targets enter descriptions/IDs.
    """
    if front_policy not in {"strict", "legacy-convention"}:
        raise ValueError("unknown front policy")
    raw_objects = {o["id"]: o for o in raw["objects"]}
    prepared_objects = {o["id"]: o for o in prepared["objects"]}
    selected = [raw_objects[o["id"]] for o in prepared["objects"]]
    selected = sorted(selected, key=lambda o: _key(seed, raw["uid"], o["id"]))
    fixed_selected = sorted(prepared.get("fixed", []), key=lambda o: _key(seed, raw["uid"], o["id"]))
    ids = {o["id"]: f"obj_{i:04d}" for i, o in enumerate(selected)}
    ids = {**ids, **{o["id"]: f"fixed_{i:04d}" for i, o in enumerate(fixed_selected)}}
    room, dz = _frame(raw)
    if prepared.get("meta", {}).get("height_dropped"):
        room = {**room, "height_m": None}
    fixed = []
    for item in fixed_selected:
        o = raw_objects.get(item["id"], item)
        if o.get("tilted"):
            raise ValueError("tilted fixed geometry cannot enter the yaw-only condition protocol")
        if min(o["size"]) <= 0:
            raise ValueError("selected fixed geometry has nonpositive size")
        if (raw["source"] == "IL3D_3dfront" and "window" in norm_cat(o["category"])
                and abs(math.sin(2 * o["yaw"])) > 1e-5
                and not o.get("v2_evidence", {}).get("corrected_width_m")):
            raise ValueError("oblique window requires source-mesh correction before conditioning")
        fixed.append({"id": ids[o["id"]], "category": norm_cat(o["category"]) or "object",
                      "size_local_m": list(o["size"]), "bottom_center_m": [*o["pos"][:2], o["pos"][2] + dz],
                      "yaw_rad": (o["yaw"] + math.pi) % (2 * math.pi) - math.pi})
    if fixed:
        room["fixed_objects"] = fixed
    validity = {"position": [], "size": [], "yaw": []}
    requests, targets, source_ids, evidence = [], [], [], []
    for obj in selected:
        category = norm_cat(obj["category"]) or "object"
        request = {"id": ids[obj["id"]], "category": category, "description": category}
        # Preserve explicit source support, never promote bbox-inferred support.
        if obj.get("anchor") == "floor" and not obj.get("anchor_inferred") and room["floor_known"] and abs(obj["pos"][2]) <= 1e-5:
            request = {**request, "support_parent": "floor"}
        if obj.get("parent") in ids and not obj.get("anchor_inferred"):
            request = {**request, "support_parent": ids[obj["parent"]]}
        requests.append(request)
        pv, sv, yv = _validity(obj, raw, front_policy)
        for key, value in (("position", pv), ("size", sv), ("yaw", yv)):
            validity[key].append(value)
        targets.append({"id": ids[obj["id"]], "target_size_local_m": list(obj["size"]),
                        "bottom_center_m": [*obj["pos"][:2], obj["pos"][2] + dz],
                        "yaw_rad": (obj["yaw"] + math.pi) % (2 * math.pi) - math.pi})
        source_ids.append(obj["id"])
        evidence.append({"tilted": bool(obj.get("tilted")), "front_policy": front_policy,
                         "size_semantics": {"OptiScene_holodeck": "padded_proxy",
                                            "MansionWorld": "annotation_footprint_proxy"}.get(
                                                raw["source"], "canonical_source_IR"),
                         "source_support_inferred": bool(obj.get("anchor_inferred")),
                         "legacy_support_inferred": bool(prepared_objects[obj["id"]].get("anchor_inferred")),
                         "raw_anchor": obj.get("anchor"), "legacy_z_snap_applied_to_target": False,
                         "recorded_source_evidence": deepcopy(obj.get("v2_evidence", {}))})
    old_user = json.loads(row["messages"][1]["content"])
    constraints, omitted_constraints = [], []
    if old_user.get("constraints"):
        old_sources = _legacy_map(prepared, row)
        mapping = {old: ids[src] for old, src in old_sources.items()}
        for c in old_user["constraints"]:
            child = prepared_objects.get(old_sources[c[1]], {})
            if c[0] == "on" and child.get("anchor_inferred"):
                omitted_constraints.append({"legacy_constraint": c, "reason": "bbox_inferred_support_not_verified_source_label"})
            else:
                constraints.append(_constraint(c, mapping))
    requests = _groups(requests, constraints, validity)
    condition = {"schema_version": "fastfill.v2", "room": room, "objects": requests, "constraints": constraints}
    validate_condition(condition)
    result = {"schema_version": "fastfill.v2", "condition": condition,
            "target": {"schema_version": "fastfill.v2", "objects": targets}, "validity": validity,
            "provenance": {"source": raw["source"], "scene_id": raw["uid"], "legacy_uid": raw["uid"],
                           "house_id": raw.get("group", raw["uid"]), "group": raw.get("group"),
                           "split": split, "legacy_flags": deepcopy(row.get("flags", {})),
                           "target_source_ids": source_ids, "field_evidence": evidence,
                           "source_meta": deepcopy(raw.get("meta", {})), "vertical_reframe_m": dz,
                           "legacy_height_dropped": bool(prepared.get("meta", {}).get("height_dropped")),
                           "omitted_legacy_constraints": omitted_constraints,
                           "constraint_source": "frozen_sparse_legacy_request", "request_order": "SHA256(seed,uid,source_id)",
                           "descriptions": "category_only_no_asset_or_pose_text"}}
    from fastfill.v2.batch import _geometry_rows, room_normalization
    _geometry_rows(result, *room_normalization(room))
    return result
