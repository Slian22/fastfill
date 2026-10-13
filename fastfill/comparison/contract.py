"""Category-only unknown-size pilot, consuming PREPARED room IR before canonical().

bind_room(room, seed) returns a deep-copied, label-bearing IR and a JSON sidecar
with source, uid, protocol, seed, original_to_bound and bound_to_original maps.
Original IDs must still be in each record's ``id``; their upstream provenance is
the caller's responsibility. HMAC-SHA256(seed, compact JSON [source,uid,id])
orders fixed and movable instances separately. No category or geometry is hashed.

The bound IR is private label data, NOT a model input. messages() projects an
explicit allowlist into system/user JSON and emits labels as the assistant turn.
Generation callers must use messages(... )[:2]. No descriptions, source IDs,
derived constraints or movable GT geometry/support appear in those two turns.
Placement sizes must come from the caller; this module cannot certify their
training provenance. The sidecar must also stay outside the model prompt.

IR lengths are meters and yaw is radians. Messages use bottom-center positions,
centimeter display precision for prepared geometry, and integer-degree yaw.
Priors keep the caller's numeric precision. All public functions return copies
and raise ValueError on invalid input; no GT repair or physical feasibility test
is performed. validate_output accepts strict JSON text or a decoded dict and
returns a copied dict in message units, never silently filling in missing size.
"""
import copy
import hashlib
import hmac
import json
import math

from fastfill import scene


DEFAULT_SEED = "unknown-size-v1-20261013"
PROTOCOL = "unknown-size-v1"
_MARKER = "_comparison_protocol"
_COMMON = (
    "You are a room layout model. Place every listed movable object exactly once, in listed order.\n"
    "Frame: meters, right-handed, Z up; floor z=0. boundary is a counter-clockwise floor polygon; "
    "boundary_type polygon denotes walls, hull denotes a scanned floor hull, proxy an approximate boundary. "
    "height is the ceiling height; a missing field means unknown.\n"
    "Objects have opaque instance IDs and explicit categories. Fixed items have explicit categories and "
    "known size, pos and yaw; they cannot move, are not output placements, and must remain unobstructed.\n"
    "size=[sx,sy,sz] denotes positive extents along local X,Y,Z; local +X is the front. "
    "pos=[x,y,z] is the bottom center. yaw is integer degrees in [0,360), counter-clockwise from room +X. "
    "on is the supporting movable or fixed ID; omit on for floor support. "
    "Supports must exist and cannot be self-references or cycles.\n"
)
_SYSTEM = {
    "joint": _COMMON + 'Predict size and placement. Output only JSON {"placements":'
             '[{"id":id,"size":[sx,sy,sz],"pos":[x,y,z],"yaw":deg,"on":support_id},...]}.',
    "placement": _COMMON + 'Use the supplied object sizes. Output only JSON {"placements":'
                 '[{"id":id,"pos":[x,y,z],"yaw":deg,"on":support_id},...]}. '
                 'Do not output size or change supplied dimensions.',
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a nonempty string")
    return value


def _number(value, field, positive=False, bounded=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: expected a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (bounded and abs(value) >= scene.MAX_COORD) or (positive and value <= 0):
        raise ValueError(f"{field}: invalid number (finite, {'positive, ' if positive else ''}"
                         f"{'magnitude < 1000 required' if bounded else 'finite required'})")
    return value


def _vector(value, field, length=3, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field}: expected {length} numbers")
    return [_number(v, field, positive) for v in value]


def _records(room):
    if not isinstance(room, dict):
        raise ValueError("room: expected a dict")
    fixed, objects = room.get("fixed", []), room.get("objects")
    if not isinstance(fixed, list) or not isinstance(objects, list):
        raise ValueError("room: fixed and objects must be lists")
    if not all(isinstance(o, dict) for o in fixed + objects):
        raise ValueError("room: each object must be a dict")
    return fixed, objects


def _support(parents):
    """Validate identities and acyclicity iteratively, including long chains."""
    for oid, parent in parents.items():
        if parent is not None and (not isinstance(parent, str) or parent not in parents):
            raise ValueError(f"parent/support: unknown reference for {oid}")
    done = set()
    for oid in parents:
        path, cur = set(), oid
        while cur is not None and cur not in done:
            if cur in path:
                raise ValueError("parent/support: self-reference or cycle")
            path.add(cur)
            cur = parents[cur]
        done.update(path)


def _validate_room(room):
    fixed, objects = _records(room)
    seen = set()
    for obj in fixed + objects:
        oid = _text(obj.get("id"), "origin id")
        if oid in seen:
            raise ValueError("duplicate origin id across fixed/objects")
        seen.add(oid)
        category = obj.get("category")
        if category is not None and not isinstance(category, str):
            raise ValueError("category: expected a string or None")
        _vector(obj.get("size"), f"{oid}.size", positive=True)
        _vector(obj.get("pos"), f"{oid}.pos")
        _number(obj.get("yaw"), f"{oid}.yaw", bounded=False)
    _support({o["id"]: o.get("parent") for o in fixed + objects})
    boundary = room.get("boundary")
    if not isinstance(boundary, (list, tuple)) or len(boundary) < 3:
        raise ValueError("boundary: expected a polygon with at least three vertices")
    points = [_vector(p, "boundary", length=2) for p in boundary]
    polygon = scene.Polygon(points)
    if not polygon.is_valid or polygon.area <= 0 or not polygon.exterior.is_ccw:
        raise ValueError("boundary: expected a valid, nondegenerate counter-clockwise polygon")
    if room.get("boundary_type") not in ("polygon", "hull", "proxy"):
        raise ValueError("boundary_type: expected polygon, hull or proxy")
    if room.get("height") is not None:
        _number(room["height"], "height", positive=True)
    if room.get("room_type") is not None and not isinstance(room["room_type"], str):
        raise ValueError("room_type: expected a string or None")
    return fixed, objects


def _bound_records(room):
    fixed, objects = _records(room)
    if room.get(_MARKER) != PROTOCOL:
        raise ValueError("room must be bound by bind_room before serialization")
    for records, prefix in ((fixed, "fixed"), (objects, "obj")):
        if [o.get("id") for o in records] != [f"{prefix}_{i:04d}" for i in range(1, len(records) + 1)]:
            raise ValueError("bound IDs/order must match their fixed/obj namespace")
    return fixed, objects


def bind_room(room, seed=DEFAULT_SEED):
    """Bind prepared original IDs; return (deep-copied label IR, identity sidecar).

    No call to scene.canonical: it would sort by target geometry and lose origin
    identity. parent is remapped explicitly for both fixed and movable records.
    Missing/duplicate IDs and unknown/self/cyclic support identities are errors.
    """
    fixed, objects = _validate_room(room)
    if _MARKER in room:
        raise ValueError("room is already bound")
    source, uid = _text(room.get("source"), "source"), _text(room.get("uid"), "uid")
    key = _text(seed, "seed").encode("utf-8")

    def order(obj):
        payload = _json([source, uid, obj["id"]]).encode("utf-8")
        return hmac.new(key, payload, hashlib.sha256).digest(), obj["id"]

    ordered = [("fixed", "fixed", sorted(fixed, key=order)),
               ("objects", "obj", sorted(objects, key=order))]
    ids = {o["id"]: f"{prefix}_{i:04d}" for _, prefix, records in ordered
           for i, o in enumerate(records, 1)}
    rebound = {kind: [{**copy.deepcopy(o), "id": ids[o["id"]],
                      **({"parent": ids[o["parent"]] if o["parent"] is not None else None}
                         if "parent" in o else {})} for o in records]
               for kind, _, records in ordered}
    new_room = {**copy.deepcopy(room), **rebound, _MARKER: PROTOCOL}
    sidecar = {"protocol": PROTOCOL, "seed": seed, "source": source, "uid": uid,
               "original_to_bound": ids, "bound_to_original": {v: k for k, v in ids.items()}}
    return new_room, sidecar


def _arm(arm):
    if not isinstance(arm, str) or arm not in _SYSTEM:
        raise ValueError("arm: expected joint or placement")


def _geometry(obj, include_size):
    pos = [scene.num(v) for v in obj["pos"]]
    _vector(pos, f"{obj['id']}.pos after cm rounding")
    item = {"id": obj["id"], "pos": pos,
            "yaw": round(math.degrees(obj["yaw"] % math.tau)) % 360}
    if include_size:
        size = [scene.num(v) for v in obj["size"]]
        _vector(size, f"{obj['id']}.size after cm rounding", positive=True)
        item = {**item, "size": size}
    return item


def _sizes(arm, estimated_sizes, objects):
    if arm == "joint":
        if estimated_sizes is not None:
            raise ValueError("estimated_sizes: joint arm must not receive a size prior")
        return {}
    if not isinstance(estimated_sizes, dict) or set(estimated_sizes) != {o["id"] for o in objects}:
        raise ValueError("estimated_sizes: expected exactly the bound movable IDs")
    return {o["id"]: _vector(estimated_sizes[o["id"]], "estimated_sizes", positive=True) for o in objects}


def messages(room, arm="joint", estimated_sizes=None):
    """Return system/user/assistant messages; use [:2] for generation.

    Joint requests contain id/category only; targets have id,size,pos,yaw,on?.
    Placement requests add caller-supplied size by bound ID; targets have
    id,pos,yaw,on? only. Original GT dimensions are never a prior fallback.
    """
    _arm(arm)
    fixed, objects = _bound_records(room)
    _validate_room(room)
    sizes = _sizes(arm, estimated_sizes, objects)
    user = {"boundary_type": room["boundary_type"], "boundary": copy.deepcopy(room["boundary"])}
    rt = scene.norm_room_type(room.get("room_type"))
    if rt:
        user = {"room_type": rt, **user}
    if room.get("height") is not None:
        user = {**user, "height": room["height"]}
    if fixed:
        user = {**user, "fixed": [{**_geometry(o, True), "category": scene.norm_cat(o.get("category")) or "object"}
                                  for o in fixed]}
    user = {**user, "objects": [{"id": o["id"], "category": scene.norm_cat(o.get("category")) or "object",
                                  **({"size": sizes[o["id"]]} if arm == "placement" else {})} for o in objects]}
    labels = {"placements": [{**_geometry(o, arm == "joint"),
                               **({"on": o["parent"]} if o.get("parent") is not None else {})} for o in objects]}
    validate_output(labels, room, arm)
    return [{"role": "system", "content": _SYSTEM[arm]}, {"role": "user", "content": _json(user)},
            {"role": "assistant", "content": _json(labels)}]


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("output: duplicate JSON key")
        out[key] = value
    return out


def validate_output(output, room, arm="joint"):
    """Validate exact output schema, finite geometry and support; return a copy.

    Requires each bound movable ID once in request order, no fixed placements,
    no extra keys, positive joint size, and integer yaw in [0,360). Coordinates
    and sizes have magnitude < scene.MAX_COORD (1000 meters). on is optional;
    if present it must be a valid ID (explicit null is not a support identity).
    JSON text must have no duplicate keys, surrounding prose or trailing data.
    """
    _arm(arm)
    fixed, objects = _bound_records(room)
    if isinstance(output, str):
        try:
            output = json.loads(output, object_pairs_hook=_unique_pairs)
        except (ValueError, RecursionError) as exc:
            raise ValueError("output: invalid JSON or duplicate key") from exc
    if not isinstance(output, dict) or set(output) != {"placements"} or not isinstance(output["placements"], list):
        raise ValueError("output: expected only a placements list")
    required = {"id", "pos", "yaw"} | ({"size"} if arm == "joint" else set())
    ids, parents = [], {o["id"]: o.get("parent") for o in fixed}
    for item in output["placements"]:
        if not isinstance(item, dict) or not required <= set(item) or set(item) - required - {"on"}:
            raise ValueError("output: invalid placement fields")
        oid = _text(item["id"], "output id")
        if oid in parents:
            raise ValueError("output: duplicate ID or fixed placement")
        _vector(item["pos"], "output pos")
        yaw = _number(item["yaw"], "output yaw")
        if not isinstance(yaw, int) or not 0 <= yaw < 360:
            raise ValueError("output yaw: expected integer degrees in [0,360)")
        if arm == "joint":
            _vector(item["size"], "output size", positive=True)
        if "on" in item:
            _text(item["on"], "output support")
        ids.append(oid)
        parents[oid] = item.get("on")
    if ids != [o["id"] for o in objects]:
        raise ValueError("output: missing/extra IDs or wrong order")
    _support(parents)
    return copy.deepcopy(output)
