"""Condition-only tokenization and geometry batching for FastFill v2.

Every JSON segment is encoded without automatic special tokens. Object segment
boundaries therefore have exact token spans for every tokenizer, including the
explicit offline byte tokenizer. Entire samples are rejected on overflow.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch

from .schema import normalize_room, validate_condition


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


def condition_segments(condition: dict[str, Any]) -> list[tuple[str, int | None]]:
    """Serialize all condition fields, tagging exact requested-object segments."""
    objects = condition.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("condition objects must be a list")
    segments = [("Predict target local size, bottom-center and yaw for every requested ID.\nCondition:\n{", None)]
    fields = sorted(set(condition) | {"objects"})
    for index, key in enumerate(fields):
        segments.append((("," if index else "") + _json(key) + ":", None))
        if key != "objects":
            segments.append((_json(condition[key]), None))
            continue
        segments.append(("[", None))
        for object_index, obj in enumerate(objects):
            if object_index:
                segments.append((",", None))
            segments.append((_json(obj), object_index))
        segments.append(("]", None))
    return segments + [("}\nAssistant:\n", None)]


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


def _mask_row(validity: dict, field: str, index: int, dimensions: int) -> list[bool]:
    rows = validity.get(field, [])
    row = rows[index] if index < len(rows) else False
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
                                "fixed_position", "fixed_position_mask", "fixed_size", "fixed_size_mask", "symmetry")}
    room = sample["condition"].get("room", {})
    source_targets = sample.get("target", {}).get("objects", [])
    target_indices = {obj["id"]: index for index, obj in enumerate(source_targets)}
    floor = room.get("floor_z_m")
    floor_known = room.get("floor_known", floor is not None) and isinstance(floor, (int, float)) and math.isfinite(floor)
    for i, (obj, target) in enumerate(zip(objects, _target_rows(sample, objects))):
        # Validity arrays describe target rows, so reorder them by stable target ID.
        label_index = target_indices.get(obj["id"], len(source_targets))
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
        symmetry = symmetries[label_index] if label_index < len(symmetries) else 1
        if not isinstance(symmetry, int) or isinstance(symmetry, bool) or symmetry < 1:
            raise ValueError("yaw symmetry order must be a positive integer")
        rows["symmetry"].append(symmetry)
    return rows


def collate_samples(samples: list[dict], tokenizer: Any, *, max_length: int = 4096, max_objects: int = 128) -> dict:
    if not samples:
        raise ValueError("cannot batch zero samples")
    encoded, spans, geometry, origins, scales = [], [], [], [], []
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
    tensors["yaw_valid"] = torch.zeros((b, n), dtype=torch.bool)
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
        "yaw_symmetry_order": tensors["symmetry"], "objects": [s["condition"]["objects"] for s in samples],
        "conditions": [s["condition"] for s in samples],
        "provenance": [s.get("provenance", {}) for s in samples],
    }
