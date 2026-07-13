"""Provenance hashing and split keys (铁律 4).

Three candidate sources (M3DLayout, IL3D, 3D-SynthPlace) share the 3D-FRONT
upstream, so the same physical room can re-enter the corpus through several
datasets with different category vocabularies and frames. ``geometry_hash``
is therefore deliberately *category-free* and frame-normalized: it hashes the
recentered floor signature plus the multiset of quantized object sizes and
positions. ``layout_hash`` adds categories and yaw for within-source exact
identity.

Known limits (documented, revisit if dedup recall looks low): rooms are not
rotation-normalized across sources, and near-duplicates beyond the 5 cm
quantization step need threshold matching (deferred). Quantization uses
round-half-away-from-zero so hashes are stable across platforms.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from scenesmith.growing_world.fastfill.schema import (
    FloorObjectSpec,
    ProvenanceMeta,
)
from scenesmith.growing_world.fastfill.transforms import (
    polygon_area,
    recenter_polygon,
)

_QUANT_M = 0.05  # 5 cm position/size grid
_QUANT_DEG = 5.0  # 5 degree yaw grid
_HASH_LEN = 16


def _q(value: float, step: float) -> int:
    """Quantize to an integer grid index, half away from zero."""
    scaled = value / step
    return int(scaled + (0.5 if scaled >= 0 else -0.5))


def _digest(parts: Sequence[str]) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:_HASH_LEN]


def _floor_signature(floor_polygon: Sequence[tuple[float, float]]) -> str:
    """Frame-invariant floor descriptor: vertex count + area + sorted edge
    lengths (quantized). Invariant to translation and rotation."""
    verts, _ = recenter_polygon(floor_polygon)
    n = len(verts)
    edges = sorted(
        _q(
            (
                (verts[i][0] - verts[(i + 1) % n][0]) ** 2
                + (verts[i][1] - verts[(i + 1) % n][1]) ** 2
            )
            ** 0.5,
            _QUANT_M,
        )
        for i in range(n)
    )
    area_q = _q(abs(polygon_area(verts)), _QUANT_M)
    return f"n{n}:a{area_q}:e{','.join(map(str, edges))}"


def geometry_hash(
    floor_polygon: Sequence[tuple[float, float]],
    objects: Sequence[FloorObjectSpec],
) -> str:
    """Cross-source near-duplicate key (category-free, see module docstring)."""
    _, offset = recenter_polygon(floor_polygon)
    rows = sorted(
        (
            _q(o.dimensions[0], _QUANT_M),
            _q(o.dimensions[1], _QUANT_M),
            _q(o.dimensions[2], _QUANT_M),
            _q(o.position_xy[0] - offset[0], _QUANT_M),
            _q(o.position_xy[1] - offset[1], _QUANT_M),
        )
        for o in objects
    )
    parts = [_floor_signature(floor_polygon), f"k{len(objects)}"]
    parts.extend(",".join(map(str, r)) for r in rows)
    return _digest(parts)


def layout_hash(
    floor_polygon: Sequence[tuple[float, float]],
    objects: Sequence[FloorObjectSpec],
) -> str:
    """Within-source exact layout identity (categories + yaw included)."""
    _, offset = recenter_polygon(floor_polygon)
    rows = sorted(
        (
            o.category,
            _q(o.dimensions[0], _QUANT_M),
            _q(o.dimensions[1], _QUANT_M),
            _q(o.dimensions[2], _QUANT_M),
            _q(o.position_xy[0] - offset[0], _QUANT_M),
            _q(o.position_xy[1] - offset[1], _QUANT_M),
            _q(o.yaw_deg, _QUANT_DEG),
        )
        for o in objects
    )
    parts = [_floor_signature(floor_polygon), f"k{len(objects)}"]
    parts.extend(":".join(map(str, r)) for r in rows)
    return _digest(parts)


def make_split_key(
    source_dataset: str, source_house_id: str, source_room_id: str = ""
) -> str:
    """House-first split key: every room of a house lands in one split.

    Sources without a house ID (e.g. 3D-SynthPlace) fall back to ROOM
    identity — a room-scope split (documented limitation: cross-file
    copies of one upstream house cannot be grouped without a house map).
    A shared sentinel like ``no_house`` is forbidden: it would fuse every
    houseless sample into ONE split unit. Both ids empty = converter bug.
    """
    if source_house_id:
        return f"{source_dataset}/{source_house_id}"
    if source_room_id:
        return f"{source_dataset}/room:{source_room_id}"
    raise ValueError(
        f"cannot build split key for {source_dataset!r}: "
        "source_house_id and source_room_id are both empty"
    )


def stamp_hashes(
    meta: ProvenanceMeta,
    floor_polygon: Sequence[tuple[float, float]],
    objects: Sequence[FloorObjectSpec],
) -> ProvenanceMeta:
    """Return a copy of ``meta`` with hashes and split key filled in."""
    return meta.model_copy(
        update={
            "geometry_hash": geometry_hash(floor_polygon, objects),
            "layout_hash": layout_hash(floor_polygon, objects),
            "split_key": make_split_key(
                meta.source_dataset, meta.source_house_id, meta.source_room_id
            ),
        }
    )
