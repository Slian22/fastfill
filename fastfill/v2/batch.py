"""Condition-only tokenization and geometry batching for FastFill v2.

Every JSON segment is encoded without automatic special tokens. Object segment
boundaries therefore have exact token spans for every tokenizer, including the
explicit offline byte tokenizer. Entire samples are rejected on overflow.

Exchangeable-group bookkeeping lives in ``validity.exchangeable_group`` (target
order) and is exposed per request slot as ``batch["exchangeable_group"]``; it is
never rendered into the condition text. Training-only augmentation is applied by
``collate_samples(..., augment=...)``; the default path never augments.

``render_minimal_condition`` is the single three-field projection (room type /
bounding rectangle / inventory) shared by the direct request adapter, the
``minimal_form_p`` augmentation and minimal-projection evaluation.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any

import torch

from .geometry import wrap_yaw
from .schema import migrate_legacy_row, normalize_room, validate_condition


class TinyTokenizer:
    """Deterministic UTF-8 tokenizer, ONLY for offline smoke tests."""

    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 258

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        values = [byte + 2 for byte in text.encode("utf-8")]
        return values + ([self.eos_token_id] if add_special_tokens else [])

    def decode(self, ids: Any, **_: Any) -> str:
        return bytes(int(i) - 2 for i in ids if 2 <= int(i) < 258).decode("utf-8", errors="replace")

    def save_pretrained(self, directory: Any) -> None:
        from pathlib import Path
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tiny_tokenizer.json").write_text('{"backend":"offline_utf8","vocab_size":258}\n')


def load_tokenizer(backbone: str, *, local_files_only: bool = False) -> Any:
    if backbone == "tiny":
        return TinyTokenizer()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("backbone tokenizer requires a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


CONDITION_FIELD_ORDER = ("schema_version", "room", "constraints", "objects")
_UNRENDERED = {"room": {"boundary_quality"}, "objects": {"exchangeable_group"}}


def _rendered(value: Any, drop: set = frozenset()) -> Any:
    """Drop supervision bookkeeping and any "_"-prefixed key from the rendered text."""
    if isinstance(value, dict):
        return {k: _rendered(v) for k, v in value.items() if not (isinstance(k, str) and k.startswith("_")) and k not in drop}
    if isinstance(value, (list, tuple)):
        return [_rendered(v) for v in value]
    return value


def condition_segments(condition: dict[str, Any]) -> list[tuple[str, int | None]]:
    """Serialize the condition in fixed order (room first, objects last), tagging exact object segments.

    Causal object tokens therefore always attend to the room and constraints.
    """
    objects = condition.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("condition objects must be a list")
    if set(condition) - set(CONDITION_FIELD_ORDER):
        raise ValueError(f"condition: unknown fields {sorted(set(condition) - set(CONDITION_FIELD_ORDER))}")
    segments = [("Predict target local size, bottom-center and yaw for every requested ID.\nCondition:\n{", None)]
    fields = [key for key in CONDITION_FIELD_ORDER if key in condition or key == "objects"]
    for index, key in enumerate(fields):
        segments.append((("," if index else "") + _json(key) + ":", None))
        if key != "objects":
            segments.append((_json(_rendered(condition[key], _UNRENDERED.get(key, frozenset()))), None))
            continue
        segments.append(("[", None))
        for object_index, obj in enumerate(objects):
            if object_index:
                segments.append((",", None))
            segments.append((_json(_rendered(obj, _UNRENDERED["objects"])), object_index))
        segments.append(("]", None))
    return segments + [("}\nAssistant:\n", None)]


RECTANGLE_TOLERANCE_M = .01


def _bounds(points: list) -> tuple[list[float], list[float]]:
    return [min(float(p[q]) for p in points) for q in range(2)], [max(float(p[q]) for p in points) for q in range(2)]


def is_axis_aligned_rectangle(points: list, tolerance_m: float = RECTANGLE_TOLERANCE_M) -> bool:
    """Four vertices, one within ``tolerance_m`` of each corner of their bounding box.

    A 1e-9 m float margin keeps a deviation of exactly ``tolerance_m`` stable under
    the snapped rigid motions of ``augment_sample`` (2.7 - 2.69 rounds either side of 0.01).
    """
    if len(points) != 4:
        return False
    low, high = _bounds(points)
    corners = {(p[0] > (low[0] + high[0]) / 2, p[1] > (low[1] + high[1]) / 2) for p in points}
    return len(corners) == 4 and all(min(abs(p[q] - low[q]), abs(p[q] - high[q])) <= tolerance_m + 1e-9
                                     for p in points for q in range(2))


def minimal_form_eligible(room: dict[str, Any]) -> bool:
    """Rooms whose three-field projection is truthful: an axis-aligned rectangle not declared boundary-unknown.

    ``render_minimal_condition`` writes ``boundary_known: true``, so hull rectangles
    and reference-extent rooms (``boundary_known: false``) keep their own text.
    A missing flag counts as known, as in ``validation.validate_scene``.
    """
    return room.get("boundary_known") is not False and is_axis_aligned_rectangle(room["floor_polygon_xy_m"])


def render_minimal_condition(condition: dict[str, Any]) -> dict[str, Any]:
    """Three-field projection: room type, bounding rectangle, inventory; nothing else.

    Keeps ``schema_version``; ``room`` becomes exactly frame, the 4-point
    axis-aligned bounding rectangle, floor_z_m, floor_known, boundary_known=true,
    height_m and room_type (the last two as given, null when absent); constraints
    are empty and each object keeps only id/category/description (no
    support_parent). Numbers are floats, so the rendered text equals
    ``direct_layout.request_to_condition`` byte for byte for the same room type,
    size and inventory; fixed objects, openings and boundary_quality are gone.
    """
    room = condition["room"]
    (x0, y0), (x1, y1) = _bounds(room["floor_polygon_xy_m"])
    number = lambda value: None if value is None else float(value)
    floor = number(room.get("floor_z_m"))
    minimal = {"frame": room["frame"], "floor_polygon_xy_m": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
               "floor_z_m": floor, "floor_known": room.get("floor_known", floor is not None), "boundary_known": True,
               "height_m": number(room.get("height_m")), "room_type": room.get("room_type")}
    return {"schema_version": condition["schema_version"], "room": minimal, "constraints": [],
            "objects": [{key: obj[key] for key in ("id", "category", "description")} for obj in condition["objects"]]}


def tokenize_condition(condition: dict[str, Any], tokenizer: Any) -> tuple[list[int], list[tuple[int, int]]]:
    tokens: list[int] = []
    spans: list[tuple[int, int]] = []
    for segment, object_index in condition_segments(condition):
        encoded = tokenizer.encode(segment, add_special_tokens=False)
        if object_index is not None:
            spans.append((len(tokens), len(tokens) + len(encoded)))
        tokens.extend(encoded)
    if not tokens or any(start == end for start, end in spans):
        raise ValueError("tokenizer produced an empty condition/object span")
    return tokens, spans


def _finite_vector(value: Any, *, positive: bool = False) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == 3 and
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and
                math.isfinite(v) and (not positive or v > 0) for v in value))


def room_normalization(room: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Use the single protocol implementation; never infer scales from targets."""
    return normalize_room(room)


def _mask_row(validity: dict, field: str, index: int | None, dimensions: int) -> list[bool]:
    rows = validity.get(field, [])
    row = rows[index] if index is not None and index < len(rows) else False
    if isinstance(row, bool):
        return [row] * dimensions
    if not isinstance(row, (list, tuple)) or len(row) != dimensions or not all(isinstance(v, bool) for v in row):
        raise ValueError(f"invalid {field} validity mask for object {index}")
    return list(row)


def _target_vector(value: Any) -> list[float]:
    if value is None:
        return [float("nan")] * 3
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("geometry target must be a three-coordinate vector")
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))) for v in value):
        raise ValueError("geometry target coordinates must be numbers or null")
    return [float(v) if v is not None else float("nan") for v in value]


def _target_rows(sample: dict, objects: list[dict]) -> list[dict]:
    targets = sample.get("target", {}).get("objects", [])
    by_id = {obj["id"]: obj for obj in targets}
    if len(by_id) != len(targets):
        raise ValueError("duplicate target object ID")
    if set(by_id) - {obj["id"] for obj in objects}:
        raise ValueError("target object ID missing from request")
    return [by_id.get(obj["id"], {}) for obj in objects]


def _geometry_rows(sample: dict, origin: list, scale: list) -> dict:
    from .validation import effective_support_requests
    objects = effective_support_requests(sample["condition"])
    validity = sample.get("validity", {})
    rows = {key: [] for key in ("position", "size", "yaw", "position_valid", "size_valid", "yaw_valid",
                                "fixed_position", "fixed_position_mask", "fixed_size", "fixed_size_mask", "symmetry", "swap", "group")}
    room = sample["condition"].get("room", {})
    source_targets = sample.get("target", {}).get("objects", [])
    target_indices = {obj["id"]: index for index, obj in enumerate(source_targets)}
    floor = room.get("floor_z_m")
    floor_known = room.get("floor_known", floor is not None) and isinstance(floor, (int, float)) and math.isfinite(floor)
    for i, (obj, target) in enumerate(zip(objects, _target_rows(sample, objects))):
        # Validity arrays describe target rows, so reorder them by stable target ID; no target row, no label.
        label_index = target_indices.get(obj["id"])
        position = _target_vector(target.get("bottom_center_m"))
        size = _target_vector(target.get("target_size_local_m"))
        pos_mask = _mask_row(validity, "position", label_index, 3)
        size_mask = _mask_row(validity, "size", label_index, 3)
        yaw = target.get("yaw_rad")
        if yaw is not None and (isinstance(yaw, bool) or not isinstance(yaw, (int, float))):
            raise ValueError("yaw target must be a number or null")
        yaw_valid = _mask_row(validity, "yaw", label_index, 1)[0]
        if any(flag and not math.isfinite(value) for flag, value in zip(pos_mask, position)):
            raise ValueError("valid position target must be finite")
        if any(flag and (not math.isfinite(value) or value <= 0) for flag, value in zip(size_mask, size)):
            raise ValueError("valid size target must be finite and positive")
        if yaw_valid and (not isinstance(yaw, (int, float)) or not math.isfinite(yaw)):
            raise ValueError("valid yaw target must be finite")
        fixed_size = obj.get("fixed_size_local_m")
        fixed_size_values = [1. if value is None else float(value) for value in (fixed_size or [None] * 3)]
        size_fixed_mask = [value is not None for value in (fixed_size or [None] * 3)]
        if any(valid and fixed and not math.isclose(value, label, rel_tol=1e-5, abs_tol=1e-5)
               for value, label, valid, fixed in zip(fixed_size_values, size, size_mask, size_fixed_mask)):
            raise ValueError("fixed size condition conflicts with demonstrated geometry")
        bounds = obj.get("size_bounds_local_m")
        if bounds and any(valid and not lo <= label <= hi for label, valid, lo, hi
                          in zip(size, size_mask, bounds["min"], bounds["max"])):
            raise ValueError("size bounds conflict with demonstrated geometry")
        fixed_z = obj.get("support_parent") == "floor" and floor_known
        if fixed_z and pos_mask[2] and not math.isclose(position[2], floor, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("floor support condition conflicts with demonstrated bottom-center")
        rows["position"].append([(v - o) / d for v, o, d in zip(position, origin, scale)])
        rows["size"].append(size)
        rows["yaw"].append(float(yaw) if yaw is not None else float("nan"))
        rows["position_valid"].append(pos_mask)
        rows["size_valid"].append(size_mask)
        rows["yaw_valid"].append(yaw_valid)
        rows["fixed_position"].append([0., 0., (floor - origin[2]) / scale[2] if fixed_z else 0.])
        rows["fixed_position_mask"].append([False, False, bool(fixed_z)])
        rows["fixed_size"].append(fixed_size_values)
        rows["fixed_size_mask"].append(size_fixed_mask)
        symmetries = validity.get("yaw_symmetry_order", [])
        symmetry = symmetries[label_index] if label_index is not None and label_index < len(symmetries) else 1
        if not isinstance(symmetry, int) or isinstance(symmetry, bool) or symmetry < 1:
            raise ValueError("yaw symmetry order must be a positive integer")
        rows["symmetry"].append(symmetry)
        # K1 box-symmetry tier; rows built before the field default to the plain convention.
        swaps = validity.get("size_axis_swap_allowed", [])
        swap = swaps[label_index] if label_index is not None and label_index < len(swaps) else False
        if not isinstance(swap, bool):
            raise ValueError("size_axis_swap_allowed entries must be booleans")
        rows["swap"].append(swap)
        groups = validity.get("exchangeable_group", [])
        group = groups[label_index] if label_index is not None and label_index < len(groups) else None
        if group is not None and (not isinstance(group, str) or not group):
            raise ValueError("exchangeable_group labels must be nonempty strings or null")
        rows["group"].append(group)
    return rows


AUGMENT_DEFAULTS = {"rotate90": True, "mirror": False, "shuffle_objects": True,
                    "drop_constraints_p": .3, "drop_support_p": .2, "category_only_description_p": .5,
                    "minimal_form_p": .5}


def _augment_options(augment: dict) -> dict:
    if set(augment) - set(AUGMENT_DEFAULTS):
        raise ValueError(f"unknown augmentation fields: {sorted(set(augment) - set(AUGMENT_DEFAULTS))}")
    options = {**AUGMENT_DEFAULTS, **augment}
    for key, value in options.items():
        if key.endswith("_p"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"augmentation {key} must be a probability in [0, 1]")
        elif not isinstance(value, bool):
            raise ValueError(f"augmentation {key} must be a boolean")
    return options


def _shuffle_objects(condition: dict, target: dict, validity: dict, generator) -> None:
    """Permute requests in place, re-id them obj_%04d in the new order, remap every reference."""
    from .matching import _rename_references
    objects, targets = condition["objects"], target.get("objects", [])
    order = torch.randperm(len(objects), generator=generator).tolist()
    mapping = {objects[old]["id"]: f"obj_{new:04d}" for new, old in enumerate(order)}
    reserved = {o["id"] for o in condition["room"].get("fixed_objects", [])} | {"floor", "wall"}
    if reserved & set(mapping.values()):
        raise ValueError("shuffled request IDs collide with fixed or reserved IDs")
    label_index = {t["id"]: i for i, t in enumerate(targets)}
    if set(label_index) - set(mapping):
        raise ValueError("target object ID missing from request")
    target_order = [label_index[objects[old]["id"]] for old in order if objects[old]["id"] in label_index]
    for key, rows in validity.items():
        if not isinstance(rows, list) or len(rows) != len(targets):
            raise ValueError(f"validity {key} must be a list aligned with target objects")
        validity[key] = [rows[j] for j in target_order]
    condition["objects"] = [_rename_references(objects[old], mapping, ("id", "support_parent")) for old in order]
    condition["constraints"] = [_rename_references(c, mapping, ("object_id", "target_id", "parent_id"), ("target_ids", "object_ids"))
                                for c in condition.get("constraints", [])]
    target["objects"] = [{**targets[j], "id": mapping[targets[j]["id"]]} for j in target_order]


def _snap(value: float) -> float:
    """Drop float noise from rigid-motion arithmetic (9.48 - 2.2 -> 7.28, not 7.279999999999999).

    Source coordinates mix short decimals with full-precision floats; the latter
    are kept as they are, so rendered numbers keep the source digit distribution.
    """
    rounded = round(value, 6)
    return rounded if abs(rounded - value) < 1e-9 else value


def _rigid_xy(condition: dict, target: dict, validity: dict, quarter_turns: int, mirror: bool) -> None:
    """Apply x -> -x (if mirror) then k quarter turns about +Z to every XY quantity, in place.

    The result is translated so the floor polygon keeps its original lower XY
    corner; room normalization is recomputed by the caller. Null target
    coordinates stay null and their validity flags follow the axis swap.
    """
    room = condition["room"]
    if room.get("openings"):
        raise ValueError("rotate90/mirror cannot transform free-form room openings")
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][quarter_turns % 4]
    sign = -1 if mirror else 1
    scale = lambda a, v: None if v is None else a * v
    xy = lambda x, y: (scale(c * sign, x) if c else scale(-s, y), scale(s * sign, x) if s else scale(c, y))
    yaw_of = lambda yaw: wrap_yaw(quarter_turns * math.pi / 2 + (math.pi - yaw if mirror else yaw))
    polygon = room["floor_polygon_xy_m"]
    shift = [min(p[q] for p in polygon) - min(xy(*p)[q] for p in polygon) for q in range(2)]
    move = lambda x, y: tuple(None if v is None else _snap(v + d) for v, d in zip(xy(x, y), shift))
    room["floor_polygon_xy_m"] = [list(move(*p)) for p in polygon]
    for obj in room.get("fixed_objects", []):
        if mirror and (obj.get("semantic_front_local") is not None or obj.get("support_surfaces")):
            raise ValueError("mirror cannot transform asset-local fronts or support surfaces")
        obj["bottom_center_m"] = [*move(*obj["bottom_center_m"][:2]), obj["bottom_center_m"][2]]
        obj["yaw_rad"] = yaw_of(obj["yaw_rad"])
    for obj in target.get("objects", []):
        if obj.get("bottom_center_m") is not None:
            obj["bottom_center_m"] = [*move(*obj["bottom_center_m"][:2]), obj["bottom_center_m"][2]]
        if obj.get("yaw_rad") is not None:
            obj["yaw_rad"] = yaw_of(obj["yaw_rad"])
    for constraint in condition.get("constraints", []):
        if "direction_xy" in constraint:
            constraint["direction_xy"] = list(xy(*constraint["direction_xy"]))
        if "polygon_xy_m" in constraint:
            constraint["polygon_xy_m"] = [list(move(*p)) for p in constraint["polygon_xy_m"]]
    if quarter_turns % 2:
        validity["position"] = [[m[1], m[0], m[2]] if isinstance(m, list) else m for m in validity.get("position", [])]


def _regroup(condition: dict, target: dict, validity: dict) -> None:
    """Recompute C5 exchangeable groups after a drop made requests identical or removed references.

    Build-time groups stay certifiable (drops only remove references and merge
    descriptions), so a group whose wider recomputed candidate fails
    certification keeps its members together under a distinct label.
    """
    from .matching import group_labels
    objects, targets = condition["objects"], target.get("objects", [])
    index = {t["id"]: i for i, t in enumerate(targets)}
    position = [_mask_row(validity, "position", index.get(obj["id"]), 3) for obj in objects]
    fresh = dict(zip((obj["id"] for obj in objects), group_labels(objects, condition.get("constraints", []), position)))
    old = validity.get("exchangeable_group") or [None] * len(targets)
    validity["exchangeable_group"] = [fresh.get(t["id"]) or (f"kept_{g}" if g else None) for t, g in zip(targets, old)]


def augment_sample(sample: dict, augment: dict, generator: torch.Generator | None = None) -> dict:
    """Training-only augmentation; deterministic given ``generator``. Returns a new sample.

    ``minimal_form_p`` replaces the condition by ``render_minimal_condition`` when
    ``minimal_form_eligible`` (a boundary-known axis-aligned rectangle): fixed objects, constraints and
    support declarations leave the text, so floor-declared z is learned for that
    sample, and exchangeable groups are recomputed like after a drop.
    """
    options = _augment_options(augment)
    draw = lambda: float(torch.rand(1, generator=generator))
    sample = migrate_legacy_row(sample)
    condition, target = deepcopy(sample["condition"]), deepcopy(sample.get("target", {}))
    validity = deepcopy(sample.get("validity", {}))
    if options["shuffle_objects"]:
        _shuffle_objects(condition, target, validity, generator)
    quarter_turns = int(torch.randint(4, (1,), generator=generator)) if options["rotate90"] else 0
    mirror = options["mirror"] and draw() < .5
    if quarter_turns or mirror:
        _rigid_xy(condition, target, validity, quarter_turns, mirror)
    drops = [draw() < options[key] for key in ("drop_constraints_p", "drop_support_p", "category_only_description_p")]
    if drops[0]:
        condition["constraints"] = []
    for obj in condition["objects"]:
        if drops[1]:
            obj.pop("support_parent", None)
        if drops[2]:
            obj["description"] = obj["category"]
    minimal = draw() < options["minimal_form_p"] and minimal_form_eligible(condition["room"])
    if minimal:
        condition = render_minimal_condition(condition)
    if any(drops) or minimal:
        _regroup(condition, target, validity)
    return {**sample, "condition": condition, "target": target, "validity": validity}


def collate_samples(samples: list[dict], tokenizer: Any, *, max_length: int = 4096, max_objects: int = 128,
                    augment: dict | None = None, generator: torch.Generator | None = None) -> dict:
    """Batch samples; ``augment`` (training loader only) applies ``augment_sample`` to each one first."""
    if not samples:
        raise ValueError("cannot batch zero samples")
    encoded, spans, geometry, origins, scales = [], [], [], [], []
    samples = [migrate_legacy_row(sample) for sample in samples]
    if augment is not None:
        samples = [augment_sample(sample, augment, generator) for sample in samples]
    for sample in samples:
        condition = sample["condition"]
        validate_condition(condition)
        objects = condition.get("objects", [])
        ids = [obj.get("id") for obj in objects]
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("request IDs must be unique nonempty strings")
        if len(objects) > max_objects:
            raise ValueError("request exceeds object budget; rebuild a complete subscene")
        tokens, object_spans = tokenize_condition(condition, tokenizer)
        if len(tokens) > max_length:
            raise ValueError("condition exceeds context budget; no partial truncation is allowed")
        origin, scale = room_normalization(condition.get("room", {}))
        encoded.append(tokens)
        spans.append(object_spans)
        geometry.append(_geometry_rows(sample, origin, scale))
        origins.append(origin)
        scales.append(scale)
    groups = [rows.pop("group") for rows in geometry]
    b, n, t = len(samples), max(len(s) for s in spans), max(len(s) for s in encoded)
    ids = torch.full((b, t), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((b, t), dtype=torch.bool)
    slot_mask = torch.zeros((b, n), dtype=torch.bool)
    object_spans = torch.zeros((b, n, 2), dtype=torch.long)
    tensors = {key: torch.full((b, n, 3), float("nan")) for key in ("position", "size")}
    tensors.update(yaw=torch.full((b, n), float("nan")), symmetry=torch.ones((b, n), dtype=torch.long))
    for key in ("fixed_position", "fixed_size"):
        tensors[key] = torch.zeros((b, n, 3))
    for key in ("position_valid", "size_valid", "fixed_position_mask", "fixed_size_mask"):
        tensors[key] = torch.zeros((b, n, 3), dtype=torch.bool)
    for key in ("yaw_valid", "swap"):
        tensors[key] = torch.zeros((b, n), dtype=torch.bool)
    for row, (tokens, object_span, geom) in enumerate(zip(encoded, spans, geometry)):
        count = len(object_span)
        ids[row, :len(tokens)] = torch.tensor(tokens)
        attention[row, :len(tokens)] = True
        slot_mask[row, :count] = True
        if count:
            object_spans[row, :count] = torch.tensor(object_span)
            for key, values in geom.items():
                tensors[key][row, :count] = torch.tensor(values, dtype=tensors[key].dtype)
    return {
        "input_ids": ids, "attention_mask": attention, "object_spans": object_spans, "slot_mask": slot_mask,
        "origin": torch.tensor(origins, dtype=torch.float32), "scale": torch.tensor(scales, dtype=torch.float32),
        "targets": {"position_normalized": tensors["position"], "size": tensors["size"], "yaw": tensors["yaw"]},
        "validity": {"position": tensors["position_valid"], "size": tensors["size_valid"], "yaw": tensors["yaw_valid"]},
        "fixed_position_normalized": tensors["fixed_position"], "fixed_position_mask": tensors["fixed_position_mask"],
        "fixed_size": tensors["fixed_size"], "fixed_size_mask": tensors["fixed_size_mask"],
        "yaw_symmetry_order": tensors["symmetry"], "size_axis_swap_allowed": tensors["swap"], "exchangeable_group": groups,
        "objects": [s["condition"]["objects"] for s in samples], "conditions": [s["condition"] for s in samples],
        "provenance": [s.get("provenance", {}) for s in samples],
    }
