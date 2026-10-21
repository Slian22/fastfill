"""Train-only size medians and component-grouped out-of-fold (OOF) priors.

Accept prepared/bound room dictionaries; never discover data or infer components.
The caller supplies the full house/alias connected-component identity for each UID.
Explicit ``split`` fields must be ``train`` when fitting; absent splits rely on the
caller's train-only contract. Prediction accepts any split and never reads sizes.

Buckets store object counts and exp(median(log(size))) independently on local XYZ.
Prediction uses category/room-type, then category, at ``min_count`` observations,
then the global bucket regardless of count. Unknown room types use the empty key.
UID/component hashes are audit identifiers, not an anonymization guarantee.
"""
import hashlib
import json
import math
import re
from collections.abc import Mapping
from itertools import groupby
from statistics import median


# Mirror fastfill.scene's stdlib normalizers without importing its Shapely runtime.
# The parity test extracts the canonical functions from scene.py to detect drift.
_HASH = re.compile(r"(?=[a-f]*\d)[0-9a-f]{8}\b")
_ALIASES = {"corner/side table": "side table", "three-seat / multi-seat sofa": "three seat sofa",
            "three-seat / multi-person sofa": "three seat sofa", "coffeetablefactory": "coffee table",
            "sidetablefactory": "side table", "boxcomforterfactory": "comforter", "comforterfactory": "comforter"}
_ROOM_WORDS = {"livingroom": "living room", "diningroom": "dining room", "livingdiningroom": "living dining room",
               "masterbedroom": "master bedroom", "secondbedroom": "second bedroom", "kidsroom": "kids room",
               "storageroom": "storage room", "laundryroom": "laundry room", "elderlyroom": "elderly room"}
_UNKNOWN_ROOM = {"misc", "other", "other room", "unknown", "undefined", "none"}


def _norm_cat(c):
    c = (c or "").lower()
    c = _ALIASES.get(re.sub(r"\(.*", "", c).strip(), c)
    c = re.split(r"[,(/]", c)[0]
    c = " ".join(re.sub(r"[^a-z0-9]+", " ", c).split())
    c = " ".join(_HASH.sub(" ", c).split()).split(" or ")[0]
    return re.sub(r"(?<=[a-z])(?:\s*\d+)+$", "", c)


def _norm_room_type(t):
    if not t:
        return None
    t = re.sub(r"\s*/\s*", "/", " ".join(t.lower().replace("_", " ").split())).rstrip(".").strip()
    t = " ".join(_ROOM_WORDS.get(w, w) for w in t.split())
    t = re.sub(r"(?:\s+\d+)+$", "", t)
    t = " ".join(re.sub(r"\b(?:f\d+|apt\d+|apartment \d+|\d+p)\b", " ", t).split())
    t = re.sub(r"\brestrooms\b", "restroom", t)
    return None if not t or t in _UNKNOWN_ROOM else t


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _label(value, name):
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")
    return value


def _size(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("size must be a positive finite numeric triple")
    try:
        valid = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0 for v in value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("size must be a positive finite numeric triple")
    return [float(v) for v in value]


def _objects(room):
    if not isinstance(room, Mapping):
        raise ValueError("room must be a mapping")
    objects = room.get("objects")
    if not isinstance(objects, (list, tuple)) or any(not isinstance(o, Mapping) for o in objects):
        raise ValueError("objects must be a sequence of mappings")
    ids = [_identifier(o.get("id"), "object id") for o in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate object id")
    return objects


def _prepare_rooms(rooms):
    prepared = [_prepare_room(r) for r in rooms]
    uids = [r["uid"] for r in prepared]
    if len(uids) != len(set(uids)):
        raise ValueError("duplicate room uid")
    return sorted(prepared, key=lambda r: r["uid"])


def _prepare_room(room):
    objects = _objects(room)
    if "split" in room and room["split"] != "train":
        raise ValueError("explicit split must be train for fitting")
    fixed = room.get("fixed") or []
    if not isinstance(fixed, (list, tuple)):
        raise ValueError("fixed must be a sequence")
    return {
        "uid": _identifier(room.get("uid"), "room uid"),
        "room_type": _norm_room_type(_label(room.get("room_type"), "room_type")) or "",
        "objects": [{"category": _norm_cat(_label(o.get("category"), "category")),
                     "size": _size(o.get("size"))} for o in objects],
        "fixed_count": len(fixed),
    }


def _bucket(sizes):
    values = list(sizes)
    return {"count": len(values),
            "size": [math.exp(median(math.log(s[axis]) for s in values)) for axis in range(3)]}


def _fit(rooms, min_count):
    observations = [(o["category"], r["room_type"], o["size"]) for r in rooms for o in r["objects"]]
    if not observations:
        raise ValueError("empty fit: no trainable objects")
    by_category = {category: list(rows) for category, rows in
                   groupby(sorted(observations, key=lambda x: (x[0], x[1])), key=lambda x: x[0])}
    pairs = {category: {rt: _bucket(row[2] for row in rows) for rt, rows in
                        groupby(records, key=lambda x: x[1])}
             for category, records in by_category.items()}
    uid_hashes = sorted(_digest(r["uid"]) for r in rooms)
    return {
        "version": 1, "estimator": "axis_log_median", "min_count": min_count,
        "category_room_type": pairs,
        "category": {cat: _bucket(r[2] for r in rows) for cat, rows in by_category.items()},
        "global": _bucket(row[2] for row in observations),
        "provenance": {
            "room_count": len(rooms), "object_count": len(observations),
            "fixed_excluded_count": sum(r["fixed_count"] for r in rooms),
            "category_count": len(by_category), "category_room_type_count": sum(map(len, pairs.values())),
            "fitted_uid_hashes": uid_hashes, "fitted_uid_sha256": _digest(_json(uid_hashes)),
            "component_source": "not_provided", "component_count": None, "component_hashes": [],
        },
    }


def fit_prior(rooms, min_count=5):
    """Fit only supplied training rooms; reject invalid targets and duplicate UIDs.

    Return JSON-compatible buckets and audit counts/hashes. Component metadata is
    explicitly unavailable here: only build_oof_priors receives authoritative groups.
    Empty rooms count toward provenance; a fit with zero movable objects fails.
    """
    _positive_int(min_count, "min_count")
    return _fit(_prepare_rooms(rooms), min_count)


def predict_sizes(room, prior):
    """Return fresh XYZ lists keyed by object ID, with no target-size access."""
    objects = _objects(room)
    rt = _norm_room_type(_label(room.get("room_type"), "room_type")) or ""
    _positive_int(prior["min_count"], "min_count")
    return {o["id"]: _predict(_norm_cat(_label(o.get("category"), "category")), rt, prior)
            for o in objects}


def _predict(category, room_type, prior):
    pair = prior["category_room_type"].get(category, {}).get(room_type)
    cat = prior["category"].get(category)
    for bucket in (pair, cat):
        if bucket is not None and bucket["count"] >= prior["min_count"]:
            return _size(bucket["size"])
    return _size(prior["global"]["size"])


def _with_components(prior, rooms, component_by_uid, held_out_fold=None):
    hashes = sorted({_digest(component_by_uid[r["uid"]]) for r in rooms})
    return {**prior, "provenance": {
        **prior["provenance"], "component_source": "caller_connected_components",
        "component_count": len(hashes), "component_hashes": hashes,
        "held_out_fold": held_out_fold,
    }}


def _component_records(mapping, component_folds):
    members = {component: sorted(_digest(uid) for uid, _ in rows)
               for component, rows in groupby(sorted(mapping.items(), key=lambda item: item[1]),
                                               key=lambda item: item[1])}
    return {_digest(component): {"fold": component_folds[component], "uid_hashes": members[component]}
            for component in sorted(members, key=_digest)}


def build_oof_priors(rooms, component_by_uid, folds=5, seed="unknown-size-v1-20261013"):
    """Fit full and whole-fold-excluded priors, assigning components by stable hash.

    Sort components by SHA-256 of compact UTF-8 JSON [seed, component], tie-break
    by component, then assign round-robin. Require >= folds nonempty component IDs
    represented by supplied rooms. This ensures every fold has rooms; every OOF
    training subset must also contain movable objects. Adding components may change
    fold assignments, so preserve the returned manifest with the frozen input set.
    Extra component-map entries are ignored. No house/alias relationships are inferred.
    """
    _positive_int(folds, "folds", minimum=2)
    _identifier(seed, "seed")
    if not isinstance(component_by_uid, Mapping):
        raise ValueError("component_by_uid must be a mapping")
    prepared = _prepare_rooms(rooms)
    mapping = {r["uid"]: _identifier(component_by_uid.get(r["uid"]), "component") for r in prepared}
    components = sorted(set(mapping.values()), key=lambda c: (_digest(_json([seed, c])), c))
    if len(components) < folds:
        raise ValueError("need at least folds different nonempty component groups")
    component_folds = {component: i % folds for i, component in enumerate(components)}
    fold_by_uid = {uid: component_folds[component] for uid, component in mapping.items()}
    full_prior = _with_components(_fit(prepared, 5), prepared, mapping)
    subsets = {str(fold): [r for r in prepared if fold_by_uid[r["uid"]] != fold] for fold in range(folds)}
    priors = {fold: _with_components(_fit(subset, 5), subset, mapping, int(fold))
              for fold, subset in subsets.items()}
    return {
        "fold_by_uid": fold_by_uid, "priors": priors, "full_prior": full_prior,
        "provenance": {**full_prior["provenance"], "fold_count": folds, "seed": seed,
                       "fold_algorithm": "sha256-json-seed-component-sort-round-robin-v1",
                       "hash_algorithm": "sha256-utf8", "components": _component_records(mapping, component_folds),
                       "training_contract": "caller_prepared_bound_train_only; reject_explicit_nontrain_split"},
    }
