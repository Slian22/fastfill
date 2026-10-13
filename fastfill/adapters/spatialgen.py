"""SpatialGen public test set (48 designer rooms). Eval-only: the source release is a test split.

Export: Z-up meters, scene-local, furniture_position = box center, size = scaled local XYZ extents,
yaw about +Z (deg). Boundary = floor polygon. Front = local -Y (front test: beds/wardrobes touch the wall with +Y).
The raw row-major affine transform and unscaled size in source_fields preserve
tilt, scale and reflection; bound all eight source corners at the exported yaw.
"""
import os

from fastfill.adapters.unified import convert_room, iter_records
from fastfill.adapters.upright_bounds import finite_vector, yaw_extents

SOURCE = "SpatialGen"


def _furniture(f):
    """Use the complete source affine box without inventing missing rotations."""
    fields = f.get("source_fields") or {}
    transform = fields.get("transform")
    if not finite_vector(transform, 16) or transform[12:] != [0, 0, 0, 1]:
        return {**f, "furniture_size": None}
    linear = [transform[i:i + 3] for i in (0, 4, 8)]
    bounds = yaw_extents(fields.get("size"), linear, (f.get("furniture_rotation") or {}).get("z"))
    if bounds is None:
        return {**f, "furniture_size": None}
    # Raw translation is the center (the exported position is rounded to 1e-6).
    # Rotation, including x/y=None for tilt and the original yaw, stays intact.
    return {**f, "furniture_size": dict(zip(("width", "length", "height"), bounds)),
            "furniture_position": dict(zip("xyz", (transform[i] for i in (3, 7, 11))))}


def load(root):
    path = os.path.join(root, "BillLin66__3D_Room_Collections/SpatialGen_exported/spatialgen.jsonl")
    for scene in iter_records(path):
        for room in scene["rooms"]:
            corrected = {**room, "furniture": [_furniture(f) for f in room.get("furniture") or []]}
            yield convert_room(corrected, source=SOURCE, uid=room["room_id"], group=f"spatialgen:{scene['scene_id']}",
                               subset="test", boundary_type="polygon", center_z=True,
                               front_offset_deg=-90, meta={"eval_only": True, "front_known": True})
