"""Compact deterministic text codec for FastFill (T1.10 groundwork).

The training/inference codec renders contexts and layouts as short plain-text
blocks: positions and sizes as centimeter integers, angles as degree integers
(quantization error <= 0.5 cm / <= 0.5 deg). ``decode(encode(x))`` restores
``x`` up to that quantization for every encoded field.

Deliberately lossy fields (documented, not bugs): ``object_id`` is
regenerated deterministically (``<category>_<n>`` per category, surface
objects ``<group_id>/<category>_<n>``), ``asset_query`` / ``functional_role``
/ ``intra_relations`` are dropped, and the ``s`` flag records only *whether*
an object wants surface fill (decoded as ``(SurfaceKind.TOP,)``).

Floor object line:      ``category|w,d,h|x,y|yaw|flags``
Surface object line:    ``category|w,d,h|x,y|yaw|t`` (``-`` when optional)
Flags: anchor letter (W/C/F) + ``t`` if required_by_task + ``s`` if
wants_surface_fill. Pattern header field: ``free`` / ``matrix:2x3:sp40,30`` /
``paired:sp15`` / ``circular:r25`` (cm integers).
"""

from __future__ import annotations

from typing import Iterable, Sequence

from scenesmith.growing_world.fastfill.schema import (
    Anchor,
    FloorLayout,
    FloorObjectSpec,
    GroupPattern,
    PatternParams,
    RoomContext,
    SupportContext,
    SurfaceKind,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
    Vec2,
    Vec3,
)
from scenesmith.growing_world.fastfill.transforms import normalize_deg

_ANCHOR_TO_FLAG = {Anchor.WALL: "W", Anchor.CORNER: "C", Anchor.FREE: "F"}
_FLAG_TO_ANCHOR = {v: k for k, v in _ANCHOR_TO_FLAG.items()}


# ------------------------------------------------------------- scalar helpers


def _cm(value: float) -> int:
    return int(round(float(value) * 100.0))


def _deg(value: float) -> int:
    return int(round(float(value)))


def _xy(point: Sequence[float]) -> str:
    return f"{_cm(point[0])},{_cm(point[1])}"


def _dims(dims: Vec3) -> str:
    return f"{_cm(dims[0])},{_cm(dims[1])},{_cm(dims[2])}"


def _poly(vertices: Iterable[Vec2]) -> str:
    return " ".join(_xy(v) for v in vertices)


def _csv(items: Sequence[str]) -> str:
    return ",".join(items) if items else "-"


def _check_token(value: str, what: str) -> str:
    if "|" in value or "\n" in value:
        raise ValueError(f"{what} may not contain '|' or newlines: {value!r}")
    return value


def _parse_xy(token: str) -> Vec2:
    x_s, y_s = token.split(",")
    return (int(x_s) / 100.0, int(y_s) / 100.0)


def _parse_dims(token: str) -> Vec3:
    w_s, d_s, h_s = token.split(",")
    return (int(w_s) / 100.0, int(d_s) / 100.0, int(h_s) / 100.0)


# ------------------------------------------------------------- room context


def encode_room_context(ctx: RoomContext) -> str:
    """Render a RoomContext as a compact deterministic text block."""
    lines = [
        f"room {_check_token(ctx.room_type, 'room_type')} id={ctx.room_id}",
        f"ceil {_cm(ctx.ceiling_height_m)}",
        f"poly {_poly(ctx.floor_polygon)}",
    ]
    for d in ctx.doors:
        lines.append(
            f"door {d.door_id} {_xy(d.center_xy)} w={_cm(d.width_m)} "
            f"d={_cm(d.clearance_depth_m)} wall={d.wall or '-'}"
        )
    for w in ctx.windows:
        keep = " keep" if w.keep_clear else ""
        lines.append(
            f"win {w.window_id} {_xy(w.center_xy)} w={_cm(w.width_m)} "
            f"sill={_cm(w.sill_height_m)} h={_cm(w.height_m)} "
            f"wall={w.wall or '-'}{keep}"
        )
    for p in ctx.portals:
        lines.append(
            f"portal {p.portal_id} {_xy(p.center_xy)} w={_cm(p.width_m)} "
            f"wall={p.wall or '-'}"
        )
    for f in ctx.forbidden_regions:
        lines.append(f"forbid {f.region_id} {_poly(f.polygon)}")
    lines.append(f"task {ctx.task or '-'}")
    lines.append(f"furn {_csv(ctx.expected_furniture)}")
    lines.append(f"manip {_csv(ctx.expected_manipulands)}")
    if ctx.style_hint:
        lines.append(f"style {ctx.style_hint}")
    return "\n".join(lines)


# -------------------------------------------------------------- floor layout


def _floor_flags(obj: FloorObjectSpec) -> str:
    flags = _ANCHOR_TO_FLAG[obj.anchor]
    if obj.required_by_task:
        flags += "t"
    if obj.wants_surface_fill:
        flags += "s"
    return flags


def encode_floor_layout(layout: FloorLayout) -> str:
    """One line per floor object: ``category|w,d,h|x,y|yaw|flags``."""
    lines = []
    for obj in layout.objects:
        cat = _check_token(obj.category, "category")
        lines.append(
            f"{cat}|{_dims(obj.dimensions)}|{_xy(obj.position_xy)}"
            f"|{_deg(obj.yaw_deg)}|{_floor_flags(obj)}"
        )
    return "\n".join(lines)


def _decode_floor_line(line: str, counters: dict[str, int]) -> FloorObjectSpec:
    parts = line.split("|")
    if len(parts) != 5:
        raise ValueError(f"malformed floor object line: {line!r}")
    category, dims_s, pos_s, yaw_s, flags = parts
    if not flags or flags[0] not in _FLAG_TO_ANCHOR:
        raise ValueError(f"bad flags {flags!r} in line: {line!r}")
    index = counters.get(category, 0)
    counters[category] = index + 1
    return FloorObjectSpec(
        object_id=f"{category}_{index}",
        category=category,
        dimensions=_parse_dims(dims_s),
        position_xy=_parse_xy(pos_s),
        yaw_deg=normalize_deg(int(yaw_s)),
        anchor=_FLAG_TO_ANCHOR[flags[0]],
        required_by_task="t" in flags[1:],
        wants_surface_fill=((SurfaceKind.TOP,) if "s" in flags[1:] else ()),
    )


def decode_floor_layout(text: str, room_id: str) -> FloorLayout:
    """Inverse of :func:`encode_floor_layout` (ids regenerated per category)."""
    counters: dict[str, int] = {}
    objects = tuple(
        _decode_floor_line(line, counters) for line in text.splitlines() if line.strip()
    )
    return FloorLayout(room_id=room_id, objects=objects)


# ------------------------------------------------------------ support context


def encode_support_context(sc: SupportContext) -> str:
    """Render one SupportContext as a compact deterministic text block."""
    s = sc.surface
    lines = [
        f"surface {s.surface_id} kind={s.kind.value} "
        f"parent={_check_token(sc.parent_category, 'parent_category')} "
        f"pdims={_dims(sc.parent_dimensions)} h={_cm(s.height_m)} "
        f"cap={s.capacity_max_objects}",
        f"poly {_poly(s.polygon_local)}",
    ]
    for f in s.forbidden_regions_local:
        lines.append(f"forbid {f.region_id} {_poly(f.polygon)}")
    lines.append(f"room {sc.room_type}")
    lines.append(f"task {sc.task or '-'}")
    lines.append(f"manip {_csv(sc.expected_manipulands_here)}")
    lines.append(f"near {_csv(sc.neighbor_objects)}")
    lines.append(f"groups {_csv(sc.desired_groups)}")
    return "\n".join(lines)


# ------------------------------------------------------------- surface groups


def _encode_pattern(pattern: GroupPattern, params: PatternParams) -> str:
    if pattern is GroupPattern.MATRIX:
        return (
            f"matrix:{params.rows}x{params.cols}"
            f":sp{_cm(params.spacing_x_m)},{_cm(params.spacing_y_m)}"
        )
    if pattern is GroupPattern.PAIRED:
        return f"paired:sp{_cm(params.spacing_x_m)}"
    if pattern is GroupPattern.CIRCULAR:
        return f"circular:r{_cm(params.radius_m)}"
    return "free"


def _decode_pattern(token: str) -> tuple[GroupPattern, PatternParams]:
    if token == "free":
        return GroupPattern.FREE, PatternParams()
    kind, _, rest = token.partition(":")
    if kind == "matrix":
        grid, _, sp = rest.partition(":sp")
        rows_s, _, cols_s = grid.partition("x")
        sx_s, _, sy_s = sp.partition(",")
        params = PatternParams(
            rows=int(rows_s),
            cols=int(cols_s),
            spacing_x_m=int(sx_s) / 100.0,
            spacing_y_m=int(sy_s) / 100.0,
        )
        return GroupPattern.MATRIX, params
    if kind == "paired":
        return GroupPattern.PAIRED, PatternParams(
            spacing_x_m=int(rest.removeprefix("sp")) / 100.0
        )
    if kind == "circular":
        return GroupPattern.CIRCULAR, PatternParams(
            radius_m=int(rest.removeprefix("r")) / 100.0
        )
    raise ValueError(f"unknown pattern token: {token!r}")


def encode_surface_groups(groups: Sequence[SurfaceObjectGroup]) -> str:
    """Group header line + one line per surface object."""
    lines: list[str] = []
    for g in groups:
        lines.append(
            f"group {g.group_id} surface={g.surface_id} "
            f"pattern={_encode_pattern(g.pattern, g.pattern_params)} "
            f"anchor={g.anchor_object_id or '-'}"
        )
        for obj in g.objects:
            cat = _check_token(obj.category, "category")
            flag = "t" if obj.required_by_task else "-"
            lines.append(
                f"{cat}|{_dims(obj.dimensions)}|{_xy(obj.position_local)}"
                f"|{_deg(obj.yaw_deg_local)}|{flag}"
            )
    return "\n".join(lines)


def _decode_group_header(line: str) -> SurfaceObjectGroup:
    parts = line.split()
    if len(parts) != 5 or parts[0] != "group":
        raise ValueError(f"malformed group header: {line!r}")
    fields = dict(p.split("=", 1) for p in parts[2:])
    pattern, params = _decode_pattern(fields["pattern"])
    anchor = fields["anchor"]
    return SurfaceObjectGroup(
        group_id=parts[1],
        surface_id=fields["surface"],
        pattern=pattern,
        pattern_params=params,
        anchor_object_id="" if anchor == "-" else anchor,
    )


def _decode_surface_line(
    line: str, group_id: str, counters: dict[str, int]
) -> SurfaceObjectSpec:
    parts = line.split("|")
    if len(parts) != 5:
        raise ValueError(f"malformed surface object line: {line!r}")
    category, dims_s, pos_s, yaw_s, flag = parts
    index = counters.get(category, 0)
    counters[category] = index + 1
    return SurfaceObjectSpec(
        object_id=f"{group_id}/{category}_{index}",
        category=category,
        dimensions=_parse_dims(dims_s),
        position_local=_parse_xy(pos_s),
        yaw_deg_local=normalize_deg(int(yaw_s)),
        required_by_task=flag == "t",
    )


def decode_surface_groups(text: str) -> tuple[SurfaceObjectGroup, ...]:
    """Inverse of :func:`encode_surface_groups`."""
    groups: list[SurfaceObjectGroup] = []
    current: SurfaceObjectGroup | None = None
    objects: list[SurfaceObjectSpec] = []
    counters: dict[str, int] = {}

    def _flush() -> None:
        if current is not None:
            groups.append(current.model_copy(update={"objects": tuple(objects)}))

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("group "):
            _flush()
            current = _decode_group_header(line)
            objects = []
            counters = {}
        else:
            if current is None:
                raise ValueError(f"object line before any group header: {line!r}")
            objects.append(_decode_surface_line(line, current.group_id, counters))
    _flush()
    return tuple(groups)
