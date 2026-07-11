"""Coordinate transforms for the FastFill contract (Z-up, meters, degrees).

The normative contract lives in ``fastfill/README.md``. Every function here
is pure and unit-tested against hand-derived vectors in
``tests/unit/fastfill/test_transforms.py`` — converter bugs here would
silently poison every downstream training run, so nothing in this module may
depend on dataset-specific assumptions (those live in the converters).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


# ------------------------------------------------------------------ axes/yaw


def yup_to_zup_point(p: Sequence[float]) -> Vec3:
    """Map a Y-up right-handed point to the Z-up contract frame.

    ``(x, y, z)_yup -> (x, -z, y)_zup`` — determinant +1, so handedness (and
    therefore yaw sign) is preserved. A rotation about +Y in the source maps
    to the same-sign rotation about +Z here.
    """
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    return (x, -z, y)


def yup_to_zup_ground_point(p: Sequence[float]) -> Vec2:
    """Map a Y-up ground-plane point ``(x, z)`` to the contract ``(x, y)``."""
    x, z = float(p[0]), float(p[1])
    return (x, -z)


def normalize_deg(angle: float) -> float:
    """Normalize an angle in degrees to ``[-180, 180)``."""
    a = math.fmod(float(angle), 360.0)
    if a < -180.0:
        a += 360.0
    elif a >= 180.0:
        a -= 360.0
    return a


def rad_to_yaw_deg(rad: float) -> float:
    """Convert a radian yaw to the contract's normalized degree yaw."""
    return normalize_deg(math.degrees(float(rad)))


# ---------------------------------------------------------------- dimensions


def yup_half_extents_to_dimensions(half: Sequence[float]) -> Vec3:
    """Y-up half-extents ``(hx, hy_up, hz)`` -> full ``[width, depth, height]``.

    Vertical (Y-up) half-extent becomes height; the source z half-extent
    becomes depth (contract local y).
    """
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
    return (2.0 * hx, 2.0 * hz, 2.0 * hy)


def yup_extents_to_dimensions(extents: Sequence[float]) -> Vec3:
    """Y-up full extents ``(ex, ey_up, ez)`` -> ``[width, depth, height]``."""
    ex, ey, ez = float(extents[0]), float(extents[1]), float(extents[2])
    return (ex, ez, ey)


def hwd_cm_to_dimensions(bbox_hwd_cm: Sequence[float]) -> Vec3:
    """3D-SynthPlace bbox ``[height, width, depth]`` in cm-style base-100
    units -> contract ``[width, depth, height]`` in meters."""
    h, w, d = (float(v) / 100.0 for v in bbox_hwd_cm)
    return (w, d, h)


# ------------------------------------------------------------------ polygons


def polygon_area(vertices: Sequence[Sequence[float]]) -> float:
    """Signed shoelace area (positive = CCW)."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i][0], vertices[i][1]
        x2, y2 = vertices[(i + 1) % n][0], vertices[(i + 1) % n][1]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def polygon_centroid(vertices: Sequence[Sequence[float]]) -> Vec2:
    """Area centroid; falls back to the vertex mean for degenerate polygons."""
    a = polygon_area(vertices)
    if abs(a) < 1e-9:
        n = len(vertices)
        return (
            sum(v[0] for v in vertices) / n,
            sum(v[1] for v in vertices) / n,
        )
    cx = cy = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i][0], vertices[i][1]
        x2, y2 = vertices[(i + 1) % n][0], vertices[(i + 1) % n][1]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return (cx / (6.0 * a), cy / (6.0 * a))


def ensure_ccw(vertices: Sequence[Vec2]) -> tuple[Vec2, ...]:
    """Return the polygon with CCW winding (contract for floor polygons)."""
    verts = tuple((float(v[0]), float(v[1])) for v in vertices)
    if polygon_area(verts) < 0.0:
        return tuple(reversed(verts))
    return verts


def recenter_polygon(
    vertices: Sequence[Vec2],
) -> tuple[tuple[Vec2, ...], Vec2]:
    """Translate a polygon so its area centroid sits at the origin.

    Returns ``(recentered_vertices, offset)`` where ``offset`` is the
    translation that was SUBTRACTED — apply the same subtraction to every
    object position from the same source room.
    """
    cx, cy = polygon_centroid(vertices)
    moved = tuple((float(v[0]) - cx, float(v[1]) - cy) for v in vertices)
    return moved, (cx, cy)


def translate_points(points: Iterable[Vec2], offset: Vec2) -> tuple[Vec2, ...]:
    """Subtract ``offset`` from each point (companion to recenter_polygon)."""
    ox, oy = offset
    return tuple((float(p[0]) - ox, float(p[1]) - oy) for p in points)


# -------------------------------------------------------------- local frames


def rotate_2d(point: Vec2, yaw_deg: float) -> Vec2:
    """Rotate a 2D point CCW about the origin by ``yaw_deg``."""
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    x, y = float(point[0]), float(point[1])
    return (c * x - s * y, s * x + c * y)


def world_to_local_2d(point: Vec2, origin: Vec2, yaw_deg: float) -> Vec2:
    """Express a room-frame point in a local frame at ``origin`` rotated by
    ``yaw_deg`` (e.g. small-object world pose -> surface-local pose)."""
    dx = float(point[0]) - float(origin[0])
    dy = float(point[1]) - float(origin[1])
    return rotate_2d((dx, dy), -yaw_deg)


def local_to_world_2d(point: Vec2, origin: Vec2, yaw_deg: float) -> Vec2:
    """Inverse of :func:`world_to_local_2d`."""
    rx, ry = rotate_2d(point, yaw_deg)
    return (rx + float(origin[0]), ry + float(origin[1]))


def footprint_corners(
    position_xy: Vec2, dimensions: Vec3, yaw_deg: float
) -> tuple[Vec2, Vec2, Vec2, Vec2]:
    """The four room-frame corners of an object's yawed footprint rectangle.

    ``dimensions`` is the contract ``[width, depth, height]``; the footprint
    is width x depth centered on ``position_xy``.
    """
    hw, hd = dimensions[0] / 2.0, dimensions[1] / 2.0
    local = ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))
    return tuple(  # type: ignore[return-value]
        local_to_world_2d(c, position_xy, yaw_deg) for c in local
    )


def front_direction(yaw_deg: float) -> Vec2:
    """Unit front vector: at yaw 0 an object faces +Y (README contract)."""
    return rotate_2d((0.0, 1.0), yaw_deg)
