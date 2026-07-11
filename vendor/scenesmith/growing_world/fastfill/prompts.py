"""Prompt pairs (system, user) for FastFill's three fixed model calls.

Three builders: FLOOR (RoomContext -> JSON array of floor objects), SURFACE
(batched: JSON array of SupportContexts -> JSON array of groups) and REPAIR
(failed groups + violation text -> regenerated groups only). All prompts
demand RAW JSON matching the pydantic shapes in ``schema.py``; field lists
are derived from ``model_json_schema()`` so prompt and schema cannot drift.
"""

from __future__ import annotations

import json
from typing import Sequence

from pydantic import BaseModel

from scenesmith.growing_world.fastfill.schema import (
    FloorObjectSpec,
    PatternParams,
    RoomContext,
    SupportContext,
    SurfaceObjectGroup,
    SurfaceObjectSpec,
)

_COORDINATE_CONTRACT = """\
COORDINATE CONTRACT (normative):
- Z-up right-handed frame; lengths in METERS; angles in DEGREES.
- yaw_deg rotates CCW about +Z, normalized to [-180, 180).
- At yaw_deg=0 an object's FRONT faces +Y. Example: a bed against the south
  wall (front toward +Y / room interior) has yaw_deg=0; against the north
  wall it has yaw_deg=180 (or -180 -> use -180..179 range).
- dimensions = [width, depth, height] FULL extents along the object's local
  x/y/z axes before yaw (width across the front, depth front-to-back)."""

_JSON_ONLY = "Output RAW JSON only: no prose, no explanations, no markdown fences."


def _short_type(spec: dict) -> str:
    if "type" in spec:
        return str(spec["type"])
    if "$ref" in spec:
        return str(spec["$ref"]).rsplit("/", 1)[-1]
    if "allOf" in spec:
        return _short_type(spec["allOf"][0])
    if "anyOf" in spec:
        return "|".join(_short_type(s) for s in spec["anyOf"])
    return "object"


def _schema_fields(model: type[BaseModel], skip: Sequence[str] = ()) -> str:
    """Compact one-line-per-field description derived from the JSON schema."""
    js = model.model_json_schema()
    required = set(js.get("required", ()))
    lines = []
    for name, spec in js.get("properties", {}).items():
        if name in skip:
            continue
        mark = "required" if name in required else "optional"
        lines.append(f"- {name} ({_short_type(spec)}, {mark})")
    return "\n".join(lines)


def _dump(model: BaseModel, exclude: set[str] | None = None) -> str:
    return json.dumps(
        model.model_dump(mode="json", exclude=exclude), separators=(",", ":")
    )


# ---------------------------------------------------------------- call 1: floor


_FLOOR_EXAMPLE = (
    '[{"object_id":"bed_0","category":"bed","asset_query":"queen bed, wood '
    'frame","dimensions":[1.6,2.0,0.55],"position_xy":[0.0,-0.4],"z":0.0,'
    '"yaw_deg":0.0,"anchor":"wall","functional_role":"sleeping",'
    '"required_by_task":true,"wants_surface_fill":[]},'
    '{"object_id":"nightstand_0","category":"nightstand","asset_query":'
    '"small wooden nightstand","dimensions":[0.5,0.4,0.55],"position_xy":'
    '[1.1,-1.2],"z":0.0,"yaw_deg":0.0,"anchor":"wall","functional_role":'
    '"bedside storage","required_by_task":false,'
    '"wants_surface_fill":["top"]}]'
)

FLOOR_SYSTEM = f"""\
You are FastFill, an expert interior layout engine. Given one room context,
produce ALL floor-standing furniture for the room in a single response.

{_COORDINATE_CONTRACT}
- The room frame origin is the floor-polygon centroid; z=0 is the floor.
- position_xy is the FOOTPRINT CENTER of the object in the room frame.

DESIGN RULES:
1. Every footprint must lie entirely inside floor_polygon.
2. Keep every door clearance free: a rectangle of the door's width extending
   clearance_depth_m from the door center into the room.
3. Never place anything inside a forbidden region.
4. Furniture footprints must not overlap each other.
5. Wall-align heavy furniture (bed, wardrobe, sofa, bookshelf, counters):
   back against the wall, anchor="wall", front (+Y after yaw) into the room.
6. EVERY category listed in expected_furniture MUST appear at least once,
   with required_by_task=true.
7. Set wants_surface_fill (e.g. ["top"]) on furniture whose surfaces should
   later receive small objects.
8. Use realistic sizes; leave walkable space between furniture and the door.

OUTPUT: a JSON array of floor objects. Fields per object:
{_schema_fields(FloorObjectSpec)}
anchor is one of "wall","corner","free"; wants_surface_fill items are from
"top","shelf","seat","counter","inset". {_JSON_ONLY}
Example output shape:
{_FLOOR_EXAMPLE}"""


def build_floor_prompt(ctx: RoomContext) -> tuple[str, str]:
    """(system, user) for call 1: floor furniture generation."""
    user = (
        "ROOM CONTEXT (JSON):\n"
        f"{_dump(ctx, exclude={'meta', 'budget'})}\n"
        "Return the floor furniture JSON array now."
    )
    return FLOOR_SYSTEM, user


# ------------------------------------------------------------- call 2: surface


_GROUP_EXAMPLE = (
    '[{"group_id":"g0","surface_id":"desk_0/top0","pattern":"free",'
    '"pattern_params":{"rows":0,"cols":0,"spacing_x_m":0.0,"spacing_y_m":0.0,'
    '"radius_m":0.0},"anchor_object_id":"","objects":[{"object_id":"g0/lamp_0",'
    '"category":"desk_lamp","asset_query":"small metal desk lamp",'
    '"dimensions":[0.15,0.15,0.4],"position_local":[-0.35,0.1],"z_local":0.0,'
    '"yaw_deg_local":0.0,"functional_role":"lighting",'
    '"required_by_task":false}],"intra_relations":[]}]'
)

SURFACE_SYSTEM = f"""\
You are FastFill's small-object placer. Input: a JSON array of support-surface
contexts (one per surface). Output: ONE JSON array of surface object GROUPS
covering those surfaces (multiple groups per surface are allowed).

{_COORDINATE_CONTRACT}
- position_local is in the SURFACE LOCAL frame: origin at the surface-polygon
  centroid, axes = the parent furniture's local axes, z_local=0 on the
  surface plane; yaw_deg_local is relative to the parent's yaw.

PLACEMENT RULES:
1. Every object must lie inside its surface's polygon_local and OUTSIDE every
   forbidden_regions_local polygon (e.g. sink basins).
2. Respect capacity_max_objects per surface.
3. Group objects functionally (e.g. a desk work set, a dining place setting).
4. Pattern primitives: set pattern to "paired"/"matrix"/"circular" with
   pattern_params (rows, cols, spacing_x_m, spacing_y_m, radius_m) and
   anchor_object_id, then provide ONE template object (plus the optional
   anchor object) — a deterministic expander replicates the template.
   Use "free" and explicit positions otherwise.
5. EVERY category in a surface's expected_manipulands_here MUST appear on
   that surface, with required_by_task=true.
6. Use realistic small-object sizes (meters).

Group fields:
{_schema_fields(SurfaceObjectGroup)}
Object fields:
{_schema_fields(SurfaceObjectSpec)}
pattern_params fields:
{_schema_fields(PatternParams)}
{_JSON_ONLY}
Example output shape:
{_GROUP_EXAMPLE}"""


def build_surface_prompt(
    contexts: Sequence[SupportContext],
) -> tuple[str, str]:
    """(system, user) for call 2: batched surface-group generation."""
    payload = json.dumps(
        [c.model_dump(mode="json") for c in contexts], separators=(",", ":")
    )
    user = (
        "SUPPORT SURFACE CONTEXTS (JSON array):\n"
        f"{payload}\n"
        "Return one JSON array of groups for these surfaces now."
    )
    return SURFACE_SYSTEM, user


# -------------------------------------------------------------- call 3: repair


REPAIR_SYSTEM = f"""\
You are FastFill's semantic repair engine. Some surface object groups failed
validation. Regenerate ONLY those groups so the violations are fixed, keeping
each group's surface_id unchanged. Follow the same coordinate contract and
placement rules as initial surface placement.

{_COORDINATE_CONTRACT}
- position_local is in the SURFACE LOCAL frame (origin at the surface-polygon
  centroid); objects must lie inside polygon_local, outside forbidden
  regions, within capacity, and required manipulands must stay present.

OUTPUT: a JSON array containing ONLY the regenerated groups (same schema as
surface generation). {_JSON_ONLY}"""


def build_repair_prompt(
    failed_groups: Sequence[SurfaceObjectGroup],
    violations_text: str,
    contexts: Sequence[SupportContext],
) -> tuple[str, str]:
    """(system, user) for the single batched semantic-repair call."""
    groups_json = json.dumps(
        [g.model_dump(mode="json") for g in failed_groups],
        separators=(",", ":"),
    )
    contexts_json = json.dumps(
        [c.model_dump(mode="json") for c in contexts], separators=(",", ":")
    )
    user = (
        "FAILED GROUPS (JSON):\n"
        f"{groups_json}\n"
        "VIOLATIONS:\n"
        f"{violations_text}\n"
        "SUPPORT SURFACE CONTEXTS (JSON):\n"
        f"{contexts_json}\n"
        "Return the regenerated groups JSON array now."
    )
    return REPAIR_SYSTEM, user
