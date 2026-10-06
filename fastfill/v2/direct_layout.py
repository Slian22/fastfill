"""Room type + room dimensions + furniture inventory -> a bbox handoff.

This boundary never resolves assets, infers support, or commits a world. The
geometric outputs remain predictions even when their proxy diagnostics fail.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .geometry import wrap_yaw
from .io import safe_output
from .schema import _id, _keys, validate_condition, validate_layout, vector


def _request_objects(entries, max_objects):
    if not isinstance(entries, list) or not entries:
        raise ValueError("furniture_list must be a nonempty list")
    objects = []
    for entry in entries:
        item = {"category": entry} if isinstance(entry, str) else entry
        _keys(item, {"id", "category", "description", "count"}, {"category"}, "furniture")
        category = _id(item["category"], "furniture category")
        description = item.get("description", category)
        if not isinstance(description, str):
            raise ValueError("furniture description must be a string")
        count = item.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("furniture count must be a positive integer")
        if "id" in item and count != 1:
            raise ValueError("an explicit furniture ID requires count=1")
        if len(objects) + count > max_objects:
            raise ValueError("furniture inventory exceeds max_objects; do not truncate the request")
        expanded = [{"id": _id(item.get("id", f"obj_{len(objects) + index:04d}")),
                     "category": category, "description": description} for index in range(count)]
        objects = objects + expanded
    return objects


def request_to_condition(request, *, max_objects=128, room_size_semantics="rectangular"):
    """Meters-only room adapter with an explicit interpretation of room size.

    room_size_m is [X length, Y length] or [X length, Y length, height].
    The room origin is its lower XY corner and its nominal floor is Z=0.
    Missing height stays unknown; no mesh, assets, doors or windows are assumed.
    rectangular declares a known rectangle and floor. reference_extent uses
    the same coordinates for normalization without declaring a physical room
    boundary or a measured floor. The profile is configuration, not a fourth
    request field.
    """
    _keys(request, {"room_type", "room_size_m", "furniture_list"},
          {"room_type", "room_size_m", "furniture_list"}, "direct request")
    room_type = _id(request["room_type"], "room type")
    if isinstance(max_objects, bool) or not isinstance(max_objects, int) or max_objects < 1:
        raise ValueError("max_objects must be a positive integer")
    if room_size_semantics not in ("rectangular", "reference_extent"):
        raise ValueError("room_size_semantics must be rectangular or reference_extent")
    dimensions = request["room_size_m"]
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) not in (2, 3):
        raise ValueError("room_size_m must contain two or three full lengths in meters")
    dimensions = vector(dimensions, len(dimensions), "room size", positive=True)
    w, d = dimensions[:2]
    known_rectangle = room_size_semantics == "rectangular"
    room = {"frame": "right_handed_z_up", "room_type": room_type,
            "floor_polygon_xy_m": [[0., 0.], [w, 0.], [w, d], [0., d]],
            "floor_z_m": 0., "floor_known": known_rectangle, "boundary_known": known_rectangle,
            "boundary_quality": "explicit_rectangular_request" if known_rectangle else "source_reference_extent",
            "height_m": dimensions[2] if len(dimensions) == 3 else None}
    condition = {"schema_version": "fastfill.v2", "room": room,
                 "objects": _request_objects(request["furniture_list"], max_objects), "constraints": []}
    validate_condition(condition)
    return condition


def _corners(position, size, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    w, d, h = size
    xy = [(x * c - y * s + position[0], x * s + y * c + position[1])
          for x, y in [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]]
    return [[x, y, position[2] + dz] for dz in [0., h] for x, y in xy]


def layout_to_scene(condition, layout):
    """Attach request semantics and explicit OBB geometry without changing sizes."""
    validate_condition(condition)
    validate_layout(layout, condition)
    predictions = {obj["id"]: obj for obj in layout["objects"]}
    objects = []
    for request in condition["objects"]:
        prediction = deepcopy(predictions[request["id"]])
        size, position = prediction["target_size_local_m"], prediction["bottom_center_m"]
        yaw = prediction["yaw_rad"]
        bbox = {"center_m": [position[0], position[1], position[2] + size[2] / 2],
                "size_local_m": list(size), "yaw_rad": yaw,
                "corners_m": _corners(position, size, yaw)}
        objects = objects + [{**prediction, "yaw_rad": yaw, "category": request["category"],
                              "description": request["description"], "bbox": bbox}]
    return {"schema_version": "fastfill.bbox-scene.v1", "frame": "right_handed_z_up",
            "units": "meters", "geometry_kind": "predicted_local_obb_envelope",
            "room": deepcopy(condition["room"]), "objects": objects}


def layout_to_roomgenbench(condition, layout):
    """SceneSpec adapter: SAGE local +Y axis, full sizes, degree yaw.

    Swapping the two local horizontal lengths and subtracting pi/2 preserves
    world corners and maps FastFill's canonical +X axis to SAGE's +Y axis.
    Geometric bbox-axis yaw does not certify an asset's semantic front. No
    asset IDs or verified support surfaces are fabricated. Explicit request
    support and hard on constraints are preserved; absent support stays unknown.
    Fixed geometry and constraints remain metadata, not generated asset claims.
    """
    scene = layout_to_scene(condition, layout)
    from .validation import effective_support_requests
    requests = {obj["id"]: obj for obj in effective_support_requests(condition)}
    room = scene["room"]
    points = room["floor_polygon_xy_m"]
    low = [min(p[q] for p in points) for q in (0, 1)]
    high = [max(p[q] for p in points) for q in (0, 1)]
    dimensions = {"width": high[0] - low[0], "length": high[1] - low[1], "height": room.get("height_m")}
    objects = [{"id": obj["id"], "type": obj["category"], "description": obj["description"],
                "asset_key": "bbox_" + hashlib.sha256(json.dumps(
                    [obj["category"], obj["description"], obj["target_size_local_m"],
                     _placement(requests[obj["id"]].get("support_parent"))],
                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:24],
                "asset_key_kind": "downstream_generation_key_only",
                "position": dict(zip(("x", "y", "z"), obj["bottom_center_m"])),
                "rotation": {"x": 0., "y": 0., "z": math.degrees(wrap_yaw(obj["yaw_rad"] - math.pi / 2))},
                "dimensions": {"width": obj["target_size_local_m"][1],
                               "length": obj["target_size_local_m"][0],
                               "height": obj["target_size_local_m"][2]},
                "place_id": requests[obj["id"]].get("support_parent"),
                "support_surface_id": requests[obj["id"]].get("support_surface_id"),
                "support_status": "declared" if requests[obj["id"]].get("support_parent") else "unknown"}
               for obj in scene["objects"]]
    scene_key = "fastfill_" + hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"),
                                                      ensure_ascii=False).encode()).hexdigest()[:16]
    downstream = {"scene_key": scene_key, "room_type": room.get("room_type"),
            "geometry_only": True, "objects": objects,
            "fixed_objects": deepcopy(room.get("fixed_objects", [])),
            "constraints": deepcopy(condition["constraints"]),
            "room": {"dimensions": dimensions, "position": {"x": low[0], "y": low[1], "z": room.get("floor_z_m")},
                     "ceiling_height": room.get("height_m"), "walls": [], "doors": [], "windows": []}}
    if room.get("boundary_quality") == "source_reference_extent":
        return {**downstream, "room_size_semantics": "reference_extent",
                "room_interpretation": "XY reference range and nominal Z origin; physical boundary and floor unknown",
                "room": {**downstream["room"], "boundary_known": room["boundary_known"],
                         "floor_known": room["floor_known"], "boundary_quality": room["boundary_quality"]}}
    return downstream


def _placement(parent):
    if parent is None:
        return "unknown"
    return parent if parent in {"floor", "wall"} else "on_object"


def bbox_diagnostics(scene, *, tolerance_m=1e-3):
    """Report proxy boundary/collision observations without requiring asset evidence."""
    from shapely.geometry import Polygon
    room, objects = scene["room"], scene["objects"]
    boundary = Polygon(room["floor_polygon_xy_m"])
    footprints = {obj["id"]: Polygon([p[:2] for p in obj["bbox"]["corners_m"][:4]]) for obj in objects}
    checks = []
    for obj in objects:
        ident, p, size = obj["id"], obj["bottom_center_m"], obj["target_size_local_m"]
        boundary_status = "pass" if boundary.buffer(tolerance_m).covers(footprints[ident]) else "fail"
        checks = checks + [{"code": "boundary", "object_ids": [ident],
                            "status": boundary_status if room.get("boundary_known", True) else "unknown"}]
        floor = room.get("floor_z_m")
        floor_status = "unknown" if floor is None or not room.get("floor_known", True) else (
            "pass" if p[2] >= floor - tolerance_m else "fail")
        checks = checks + [{"code": "floor_lower_bound", "object_ids": [ident], "status": floor_status}]
        height = room.get("height_m")
        ceiling_status = "unknown" if height is None or floor is None or not room.get("floor_known", True) else (
            "pass" if p[2] + size[2] <= floor + height + tolerance_m else "fail")
        checks = checks + [{"code": "ceiling", "object_ids": [ident], "status": ceiling_status}]
    for i, first in enumerate(objects):
        for second in objects[i + 1:]:
            low = max(first["bottom_center_m"][2], second["bottom_center_m"][2])
            high = min(first["bottom_center_m"][2] + first["target_size_local_m"][2],
                       second["bottom_center_m"][2] + second["target_size_local_m"][2])
            if high - low > tolerance_m and footprints[first["id"]].intersection(footprints[second["id"]]).area > 1e-8:
                checks = checks + [{"code": "obb_overlap", "object_ids": [first["id"], second["id"]],
                                    "status": "fail", "interpretation": "proxy overlap; support/contact semantics unknown"}]
    report = {"scope": "target_bbox_proxy", "tolerance_m": tolerance_m, "checks": checks,
            "counts": dict(Counter(c["status"] for c in checks)),
            "asset_retrieval": "not_attempted", "mesh": "not_checked", "support": "unknown",
            "physics": "not_checked", "commit": "not_attempted"}
    if room.get("boundary_quality") == "source_reference_extent":
        return {**report, "room_size_semantics": "reference_extent",
                "room_interpretation": "XY reference range and nominal Z origin; physical boundary and floor unknown"}
    return report


def write_bbox_glb(path, scene):
    from .bbox_visualization import glb_bytes
    target = safe_output(path)
    payload = glb_bytes(scene)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)


def export_handoff(directory, condition, layout):
    """Write a new handoff directory. Validation failures never repair predictions."""
    from .bbox_visualization import glb_bytes, preview_svg
    output = safe_output(directory)
    scene = layout_to_scene(condition, layout)
    downstream = layout_to_roomgenbench(condition, layout)
    registry = {}
    for obj in downstream["objects"]:
        previous = registry.get(obj["asset_key"], {})
        registry = {**registry, obj["asset_key"]: {"asset_key": obj["asset_key"], "type": obj["type"],
                    "description": obj["description"], "dimensions": deepcopy(obj["dimensions"]),
                    "place": _placement(obj["place_id"]), "support_status": obj["support_status"],
                    "placement_eligible": obj["place_id"] is not None, "scenes": [downstream["scene_key"]],
                    "n_instances": previous.get("n_instances", 0) + 1,
                    "asset_key_kind": "downstream_generation_key_only"}}
    payloads = {"condition.json": json.dumps(condition, indent=2, allow_nan=False) + "\n",
                "layout.json": json.dumps(layout, indent=2, allow_nan=False) + "\n",
                "scene.json": json.dumps(scene, indent=2, allow_nan=False) + "\n",
                "roomgenbench_scene.json": json.dumps(downstream, indent=2, allow_nan=False) + "\n",
                "assets.jsonl": "".join(json.dumps(registry[key], allow_nan=False) + "\n" for key in sorted(registry)),
                "diagnostics.json": json.dumps(bbox_diagnostics(scene), indent=2, allow_nan=False) + "\n",
                "preview.svg": preview_svg(scene), "layout.glb": glb_bytes(scene)}
    # Compute and validate everything before creating the output directory.
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open("xb" if isinstance(payload, bytes) else "x") as stream:
            stream.write(payload)
    return output
