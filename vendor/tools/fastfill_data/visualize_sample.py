"""Top-view visual audit of one FastFillSample (facing-convention checker).

Renders the contract view of a sample: floor polygon outline, doors as red
wall segments with dashed clearance rectangles, windows as blue segments,
each floor object as its yaw-rotated footprint rectangle
(``transforms.footprint_corners``) with a category label and a front arrow
(``transforms.front_direction`` — this is how facing conventions are audited
per dataset), and surface objects as small dots at their WORLD positions
(surface-local -> parent-local -> room frame), colored per surface.

    python tools/fastfill_data/visualize_sample.py \
        --in out/deduped.jsonl --index 0 --out out/fig.png

``render_sample(sample, out_path)`` is importable for programmatic use.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

# Script-mode bootstrap: make `fastfill_data` (tools/) and `scenesmith`
# (repo root, not pip-installed) importable when run as a plain script.
for _extra in (
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2],
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from fastfill_data.common import read_jsonl  # noqa: E402
from scenesmith.growing_world.fastfill.schema import (  # noqa: E402
    FastFillSample,
    FloorObjectSpec,
    Vec2,
)
from scenesmith.growing_world.fastfill.transforms import (  # noqa: E402
    footprint_corners,
    front_direction,
    local_to_world_2d,
    polygon_centroid,
)

DOOR_COLOR = "red"
WINDOW_COLOR = "tab:blue"
FLOOR_FACE = "0.85"
FRONT_ARROW_MIN_M = 0.15  # arrows stay visible on tiny objects
SURFACE_COLORS = plt.get_cmap("tab10")


# ------------------------------------------------------------ wall geometry


def _point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _wall_segment(
    polygon: Sequence[Vec2], center_xy: Vec2, width_m: float
) -> tuple[Vec2, Vec2, Vec2]:
    """``(p1, p2, inward_normal)`` for an opening on the nearest wall edge."""
    n = len(polygon)
    a, b = min(
        ((polygon[i], polygon[(i + 1) % n]) for i in range(n)),
        key=lambda edge: _point_segment_distance(center_xy, *edge),
    )
    ux, uy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / norm, uy / norm
    half = width_m / 2.0
    cx, cy = center_xy
    p1 = (cx - ux * half, cy - uy * half)
    p2 = (cx + ux * half, cy + uy * half)
    nx, ny = -uy, ux  # pick the normal sign that points into the room
    ccx, ccy = polygon_centroid(polygon)
    if (ccx - cx) * nx + (ccy - cy) * ny < 0.0:
        nx, ny = -nx, -ny
    return p1, p2, (nx, ny)


# ------------------------------------------------------------ layer drawing


def _draw_openings(ax: Axes, sample: FastFillSample) -> None:
    polygon = sample.room_context.floor_polygon
    for door in sample.room_context.doors:
        p1, p2, (nx, ny) = _wall_segment(polygon, door.center_xy, door.width_m)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=DOOR_COLOR, lw=3, zorder=4)
        depth = door.clearance_depth_m
        clearance = (
            p1,
            p2,
            (p2[0] + nx * depth, p2[1] + ny * depth),
            (p1[0] + nx * depth, p1[1] + ny * depth),
        )
        ax.add_patch(
            MplPolygon(
                clearance,
                closed=True,
                fill=False,
                edgecolor=DOOR_COLOR,
                linestyle="--",
                lw=1,
                zorder=4,
            )
        )
    for window in sample.room_context.windows:
        p1, p2, _ = _wall_segment(polygon, window.center_xy, window.width_m)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=WINDOW_COLOR, lw=3, zorder=4)


def _draw_floor_object(ax: Axes, obj: FloorObjectSpec) -> None:
    corners = footprint_corners(obj.position_xy, obj.dimensions, obj.yaw_deg)
    ax.add_patch(
        MplPolygon(
            corners,
            closed=True,
            facecolor=FLOOR_FACE,
            edgecolor="black",
            lw=1,
            zorder=2,
        )
    )
    x, y = obj.position_xy
    ax.text(x, y, obj.category, ha="center", va="center", fontsize=6, zorder=5)
    fx, fy = front_direction(obj.yaw_deg)
    length = max(obj.dimensions[1] / 2.0, FRONT_ARROW_MIN_M)
    ax.annotate(
        "",
        xy=(x + fx * length, y + fy * length),
        xytext=(x, y),
        arrowprops={"arrowstyle": "->", "color": "darkgreen", "lw": 1.2},
        zorder=5,
    )


def _draw_surface_objects(ax: Axes, sample: FastFillSample) -> None:
    """Surface objects at WORLD xy: local -> parent frame -> room frame."""
    floor = sample.layout.floor_layout
    surfaces = {s.surface_id: s for s in floor.support_surfaces}
    parents = {o.object_id: o for o in floor.objects}
    for index, group in enumerate(sample.layout.surface_groups):
        surface = surfaces.get(group.surface_id)
        if surface is None:  # unresolvable group — nothing to audit visually
            continue
        parent = parents.get(surface.parent_object_id)
        if parent is None:
            continue
        scx, scy = polygon_centroid(surface.polygon_local)
        color = SURFACE_COLORS(index % 10)
        xs, ys = [], []
        for obj in group.objects:
            parent_local = (scx + obj.position_local[0], scy + obj.position_local[1])
            wx, wy = local_to_world_2d(parent_local, parent.position_xy, parent.yaw_deg)
            xs.append(wx)
            ys.append(wy)
        ax.scatter(xs, ys, s=12, color=color, zorder=6, label=group.surface_id)


# --------------------------------------------------------------- entry point


def render_sample(sample: FastFillSample, out_path: Path) -> Path:
    """Render the top view of ``sample`` to ``out_path`` (PNG)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    polygon = sample.room_context.floor_polygon
    outline = tuple(polygon) + (polygon[0],)
    ax.plot(
        [v[0] for v in outline],
        [v[1] for v in outline],
        color="black",
        lw=2,
        zorder=3,
    )
    for obj in sample.layout.floor_layout.objects:
        _draw_floor_object(ax, obj)
    _draw_openings(ax, sample)
    _draw_surface_objects(ax, sample)
    if sample.layout.surface_groups:
        ax.legend(fontsize=6, loc="upper right", title="surfaces")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{sample.sample_id}  ({sample.room_context.room_type})", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _select_sample(
    in_path: Path, index: int | None, sample_id: str | None
) -> FastFillSample:
    for position, sample in enumerate(read_jsonl(in_path)):
        if sample_id is not None:
            if sample.sample_id == sample_id:
                return sample
        elif position == (index or 0):
            return sample
    wanted = f"sample_id={sample_id!r}" if sample_id else f"index={index or 0}"
    raise SystemExit(f"no sample with {wanted} in {in_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a top-view audit figure for one FastFillSample"
    )
    parser.add_argument(
        "--in", dest="in_path", required=True, help="FastFillSample JSONL"
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--index", type=int, default=None, help="0-based row")
    selector.add_argument("--sample-id", default=None, help="exact sample_id")
    parser.add_argument("--out", required=True, help="output PNG path")
    args = parser.parse_args()

    sample = _select_sample(Path(args.in_path), args.index, args.sample_id)
    out = render_sample(sample, Path(args.out))
    print(f"rendered {sample.sample_id} -> {out}")


if __name__ == "__main__":
    main()
