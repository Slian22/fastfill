"""Contract tests use synthetic prepared IR only; no training or network."""
import copy
import hashlib
import hmac
import json
import math

import pytest

from fastfill.comparison import bind_room, messages, validate_output
from fastfill.scene import norm_cat


SEED = "unknown-size-v1-20261013"


def prepared():
    return {
        "source": "private-source", "uid": "private-room", "group": "private-house",
        "room_type": "LivingRoom 2", "boundary_type": "polygon",
        "boundary": [[0, 0], [6, 0], [6, 5], [0, 5]], "height": 3,
        "meta": {"asset_id": "DO_NOT_SHOW"}, "constraints": [["near", "asset-chair", "asset-table", 1]],
        "objects": [
            {"id": "asset-table", "category": "CoffeeTableFactory", "size": [1.2, .8, .75],
             "pos": [2, 2, 0], "yaw": math.pi / 2, "parent": None, "desc": "DO_NOT_SHOW table at 2,2"},
            {"id": "asset-lamp", "category": "Lamp 12", "size": [.2, .2, .4],
             "pos": [2, 2, .75], "yaw": -math.pi / 2, "parent": "asset-table"},
            {"id": "asset-chair", "category": "Chair / Stool", "size": [.5, .5, .9],
             "pos": [4, 3, 0], "yaw": 0, "parent": None}],
        "fixed": [
            {"id": "asset-column", "category": "Column 8", "size": [.3, .3, 3],
             "pos": [5, 4, 0], "yaw": 0},
            {"id": "asset-shelf", "category": "Shelf", "size": [1, .5, 1],
             "pos": [1, 4, 0], "yaw": math.pi}],
    }


def prior(room):
    return {o["id"]: [.61, .72, .83] for o in room["objects"]}


def inputs(room, arm="joint", sizes=None):
    return json.dumps(messages(room, arm, sizes)[:2], ensure_ascii=False).encode()


def target(room, arm="joint"):
    return json.loads(messages(room, arm, prior(room) if arm == "placement" else None)[2]["content"])


def test_hmac_identity_definition_separate_namespaces_and_parent_remap():
    source = prepared()
    room, sidecar = bind_room(source)
    expected = {}
    for kind, prefix in (("fixed", "fixed"), ("objects", "obj")):
        def key(obj):
            payload = json.dumps([source["source"], source["uid"], obj["id"]],
                                 ensure_ascii=False, separators=(",", ":")).encode()
            return hmac.new(SEED.encode(), payload, hashlib.sha256).digest(), obj["id"]
        ordered = sorted(source[kind], key=key)
        expected.update({o["id"]: f"{prefix}_{i:04d}" for i, o in enumerate(ordered, 1)})
        assert [o["id"] for o in room[kind]] == [expected[o["id"]] for o in ordered]
    assert sidecar["original_to_bound"] == expected
    assert sidecar["bound_to_original"] == {v: k for k, v in expected.items()}
    assert sidecar["seed"] == SEED
    assert bind_room(source, SEED) == (room, sidecar)
    lamp = next(o for o in room["objects"] if o["id"] == expected["asset-lamp"])
    assert lamp["parent"] == expected["asset-table"]
    assert lamp["size"] == [.2, .2, .4]
    assert lamp["pos"] == [2, 2, .75]
    assert lamp["yaw"] == -math.pi / 2


@pytest.mark.parametrize("arm", ["joint", "placement"])
@pytest.mark.parametrize("field,value", [("size", [2.3, 1.8, 1.4]), ("pos", [3, 4, .25]),
                                        ("yaw", 1.23), ("parent", "asset-shelf")])
def test_movable_counterfactuals_do_not_change_input_bytes(arm, field, value):
    original = prepared()
    changed = copy.deepcopy(original)
    changed["objects"][1][field] = value
    a, sa = bind_room(original)
    b, sb = bind_room(changed)
    sizes = prior(a) if arm == "placement" else None
    assert sa["original_to_bound"] == sb["original_to_bound"]
    assert inputs(a, arm, sizes) == inputs(b, arm, sizes)
    if arm == "joint" or field != "size":
        assert target(a, arm) != target(b, arm)
    else:
        assert target(a, arm) == target(b, arm)


@pytest.mark.parametrize("arm", ["joint", "placement"])
def test_record_reordering_stable_and_prompt_is_allowlisted(arm):
    source = prepared()
    reordered = {**source, "objects": list(reversed(source["objects"])),
                 "fixed": list(reversed(source["fixed"]))}
    a, sa = bind_room(source)
    b, sb = bind_room(reordered)
    sizes = prior(a) if arm == "placement" else None
    assert sa == sb
    assert messages(a, arm, sizes) == messages(b, arm, sizes)
    prompt = inputs(a, arm, sizes).decode()
    for forbidden in ("asset-", "private-", "DO_NOT_SHOW", "desc", "constraints", '"parent"', '"anchor"'):
        assert forbidden not in prompt
    user = json.loads(messages(a, arm, sizes)[1]["content"])
    assert set(user) == {"room_type", "boundary_type", "boundary", "height", "fixed", "objects"}
    assert user["room_type"] == "living room"
    for obj in user["objects"]:
        assert set(obj) == ({"id", "category", "size"} if arm == "placement" else {"id", "category"})
        if arm == "placement":
            assert obj["size"] == sizes[obj["id"]]
    for obj, raw in zip(user["fixed"], a["fixed"]):
        assert set(obj) == {"id", "category", "size", "pos", "yaw"}
        assert obj["category"] == norm_cat(raw["category"])
        assert obj["size"] == raw["size"] and obj["pos"] == raw["pos"]


def test_category_never_affects_ids_and_fixed_namespace_is_independent():
    source = prepared()
    a, sa = bind_room(source)
    changed = copy.deepcopy(source)
    changed["objects"][0]["category"] = "glassnightstand5c8f77c6"
    b, sb = bind_room(changed)
    assert sa == sb
    assert "glassnightstand" in inputs(b).decode()
    changed["fixed"] = []
    c, sc = bind_room(changed)
    assert b["objects"] == c["objects"]
    assert {k: v for k, v in sc["original_to_bound"].items()} == {
        k: v for k, v in sb["original_to_bound"].items() if v.startswith("obj_")}


@pytest.mark.parametrize("arm", ["joint", "placement"])
def test_target_exact_geometry_units_and_fixed_support(arm):
    raw = prepared()
    raw["objects"][1]["parent"] = "asset-shelf"
    room, sidecar = bind_room(raw)
    result = target(room, arm)
    assert [o["id"] for o in result["placements"]] == [o["id"] for o in room["objects"]]
    for obj, label in zip(result["placements"], room["objects"]):
        assert obj["pos"] == label["pos"]
        assert obj["yaw"] == round(math.degrees(label["yaw"])) % 360
        assert obj.get("on") == label.get("parent")
        assert set(obj) == {"id", "pos", "yaw"} | ({"on"} if label.get("parent") else set()) | (
            {"size"} if arm == "joint" else set())
        if arm == "joint":
            assert obj["size"] == label["size"]
    assert any(o.get("on") == sidecar["original_to_bound"]["asset-shelf"] for o in result["placements"])
    assert validate_output(json.dumps(result), room, arm) == result


def test_deep_copies_and_messages_do_not_mutate_inputs():
    source = prepared()
    snapshot = copy.deepcopy(source)
    room, sidecar = bind_room(source)
    before = copy.deepcopy(room)
    sizes = prior(room)
    sizes_before = copy.deepcopy(sizes)
    messages(room)
    messages(room, "placement", sizes)
    assert source == snapshot and room == before and sizes == sizes_before
    room["objects"][0]["size"][0] = 99
    room["boundary"][0][0] = 99
    room["meta"]["asset_id"] = "changed"
    sidecar["original_to_bound"].clear()
    assert source == snapshot
    room, _ = bind_room(source)
    output = target(room)
    checked = validate_output(output, room)
    checked["placements"][0]["pos"][0] = 99
    assert output == target(room)


@pytest.mark.parametrize("bad", [None, "", " ", 1, [], True])
def test_missing_or_bad_origin_id_rejected(bad):
    raw = prepared()
    raw["objects"][0]["id"] = bad
    with pytest.raises(ValueError, match="id"):
        bind_room(raw)
    del raw["objects"][0]["id"]
    with pytest.raises(ValueError, match="id"):
        bind_room(raw)


@pytest.mark.parametrize("kind", ["fixed", "objects"])
def test_duplicate_ids_across_both_namespaces_rejected(kind):
    raw = prepared()
    raw[kind][1]["id"] = raw["objects"][0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        bind_room(raw)


@pytest.mark.parametrize("parent", ["missing", "", [], True, "asset-lamp"])
def test_bad_parent_identity_rejected(parent):
    raw = prepared()
    raw["objects"][1]["parent"] = parent
    with pytest.raises(ValueError, match="parent|support"):
        bind_room(raw)


def test_support_cycles_rejected():
    raw = prepared()
    raw["objects"][0]["parent"] = "asset-lamp"
    with pytest.raises(ValueError, match="parent|support"):
        bind_room(raw)


BAD_NUMBERS = [True, "1", None, float("nan"), float("inf"), -float("inf"), 10**1000]


@pytest.mark.parametrize("value", BAD_NUMBERS, ids=["bool", "str", "null", "nan", "inf", "-inf", "overflow"])
@pytest.mark.parametrize("field", ["size", "pos", "yaw", "boundary", "height"])
def test_nonfinite_or_nonnumeric_geometry_rejected(field, value):
    raw = prepared()
    if field == "boundary":
        raw[field][0][0] = value
    elif field == "height":
        if value is None:  # Unknown height is legal.
            return
        raw[field] = value
    elif field == "yaw":
        raw["objects"][0][field] = value
    else:
        raw["objects"][0][field][0] = value
    with pytest.raises(ValueError):
        bind_room(raw)


@pytest.mark.parametrize("size", [None, [], [1, 2], [1, 2, 3, 4], "123", [0, 1, 1], [-1, 1, 1]])
def test_malformed_sizes_rejected_everywhere(size):
    raw = prepared()
    raw["fixed"][0]["size"] = size
    with pytest.raises(ValueError):
        bind_room(raw)
    room, _ = bind_room(prepared())
    sizes = prior(room)
    sizes[room["objects"][0]["id"]] = size
    with pytest.raises(ValueError):
        messages(room, "placement", sizes)


@pytest.mark.parametrize("value", BAD_NUMBERS, ids=["bool", "str", "null", "nan", "inf", "-inf", "overflow"])
def test_prior_numbers_are_strict(value):
    room, _ = bind_room(prepared())
    sizes = prior(room)
    sizes[room["objects"][0]["id"]][0] = value
    with pytest.raises(ValueError, match="estimated_sizes"):
        messages(room, "placement", sizes)


def test_prior_keys_must_match_bound_movable_ids_exactly():
    room, _ = bind_room(prepared())
    sizes = prior(room)
    for bad in (None, {}, [], {**sizes, "asset-chair": [1, 1, 1]},
                {**sizes, "fixed_0001": [1, 1, 1]}, {k: v for k, v in sizes.items() if k != "obj_0001"}):
        with pytest.raises(ValueError, match="estimated_sizes"):
            messages(room, "placement", bad)
    with pytest.raises(ValueError):
        messages(room, "joint", sizes)


def test_arm_binding_provenance_and_empty_room_validation():
    for key in ("source", "uid"):
        raw = prepared()
        del raw[key]
        with pytest.raises(ValueError, match=key):
            bind_room(raw)
    with pytest.raises(ValueError, match="seed"):
        bind_room(prepared(), seed=None)
    with pytest.raises(ValueError, match="bound"):
        messages(prepared())
    room, _ = bind_room(prepared())
    with pytest.raises(ValueError, match="bound"):
        bind_room(room)
    with pytest.raises(ValueError, match="arm"):
        messages(room, "other")
    raw = {**prepared(), "objects": [], "fixed": [], "height": None, "room_type": None}
    room, _ = bind_room(raw)
    assert target(room) == {"placements": []}
    assert target(room, "placement") == {"placements": []}
    assert "height" not in json.loads(messages(room)[1]["content"])


@pytest.mark.parametrize("boundary", [[], [[0, 0], [1, 1]], [[0, 0], [1, 0], [2, 0]],
                                       [[0, 0], [2, 2], [0, 2], [2, 0]]])
def test_invalid_boundary_rejected(boundary):
    with pytest.raises(ValueError, match="boundary"):
        bind_room({**prepared(), "boundary": boundary})


@pytest.mark.parametrize("arm", ["joint", "placement"])
def test_output_rejects_wrong_ids_keys_numbers_and_support(arm):
    room, _ = bind_room(prepared())
    good = target(room, arm)
    mutations = [lambda p: p.pop(), lambda p: p.append(copy.deepcopy(p[0])),
                 lambda p: p[0].update(id="fixed_0001"), lambda p: p[0].update(yaw=True),
                 lambda p: p[0].update(yaw=float("nan")), lambda p: p[0].update(yaw=360),
                 lambda p: p[0].update(yaw=1.5), lambda p: p[0].update(pos=[0, 1]),
                 lambda p: p[0].update(pos=[float("inf"), 0, 0]),
                 lambda p: p[0].update(on="not_an_id"), lambda p: p[0].update(on=p[0]["id"]),
                 lambda p: p[0].update(on=None), lambda p: p[0].update(desc="hidden")]
    for mutate in mutations:
        bad = copy.deepcopy(good)
        mutate(bad["placements"])
        with pytest.raises(ValueError):
            validate_output(bad, room, arm)
    bad = copy.deepcopy(good)
    if arm == "joint":
        bad["placements"][0].pop("size")
    else:
        bad["placements"][0]["size"] = [1, 1, 1]
    with pytest.raises(ValueError):
        validate_output(bad, room, arm)
    for bad in ('not JSON', json.dumps(good) + ' trailing', '{"placements": [], "placements": []}',
                {"placements": None}, {"placements": [None]}, {**good, "extra": 1}):
        with pytest.raises(ValueError):
            validate_output(bad, room, arm)


def test_bound_labels_are_revalidated_and_output_support_cycles_rejected():
    room, _ = bind_room(prepared())
    bad = copy.deepcopy(room)
    bad["objects"][0]["yaw"] = float("inf")
    with pytest.raises(ValueError):
        messages(bad)
    bad = target(room)
    a, b = bad["placements"][:2]
    a["on"], b["on"] = b["id"], a["id"]
    with pytest.raises(ValueError, match="support"):
        validate_output(bad, room)


@pytest.mark.parametrize("kind", ["objects", "fixed"])
def test_display_rounding_cannot_emit_out_of_range_geometry(kind):
    raw = prepared()
    raw[kind][0]["pos"][0] = 999.999
    room, _ = bind_room(raw)
    with pytest.raises(ValueError, match="pos"):
        messages(room)


@pytest.mark.parametrize("kind", ["objects", "fixed"])
def test_display_rounding_cannot_silently_clamp_tiny_size(kind):
    raw = prepared()
    raw[kind][0]["size"][0] = .001
    room, _ = bind_room(raw)
    with pytest.raises(ValueError, match="size"):
        messages(room)


@pytest.mark.parametrize("overrides", [{"objects": None}, {"fixed": None}, {"objects": [None]},
                                       {"boundary_type": "invalid"}, {"room_type": []},
                                       {"height": 0}, {"height": -1}, {"height": 1000}])
def test_invalid_room_shapes_and_values_rejected(overrides):
    with pytest.raises(ValueError):
        bind_room({**prepared(), **overrides})


def test_invalid_room_and_bound_identity_rejected():
    with pytest.raises(ValueError):
        bind_room(None)
    raw = prepared()
    raw["objects"][0]["category"] = []
    with pytest.raises(ValueError, match="category"):
        bind_room(raw)
    room, _ = bind_room(prepared())
    room["objects"][0]["id"] = "asset-table"
    with pytest.raises(ValueError, match="bound"):
        messages(room)
    with pytest.raises(ValueError, match="bound"):
        validate_output({"placements": []}, room)


def test_fixed_parent_remapped_without_exposing_support():
    raw = prepared()
    raw["fixed"][0]["parent"] = "asset-shelf"
    room, sidecar = bind_room(raw)
    fixed = {o["id"]: o for o in room["fixed"]}
    assert fixed[sidecar["original_to_bound"]["asset-column"]]["parent"] == sidecar["original_to_bound"]["asset-shelf"]
    assert all("parent" not in o and "on" not in o for o in json.loads(messages(room)[1]["content"])["fixed"])


def test_unknown_categories_and_optional_fields_remain_explicit():
    raw = prepared()
    raw["objects"][0]["category"] = None
    raw["room_type"] = "unknown"
    del raw["fixed"]
    del raw["height"]
    room, _ = bind_room(raw)
    user = json.loads(messages(room)[1]["content"])
    assert {"id", "category"} == set(user["objects"][0])
    assert "object" in [o["category"] for o in user["objects"]]
    assert "fixed" not in user and "height" not in user and "room_type" not in user


@pytest.mark.parametrize("size", [[False, 1, 1], [0, 1, 1], [float("nan"), 1, 1], [1000, 1, 1]])
def test_joint_output_size_validation(size):
    room, _ = bind_room(prepared())
    bad = target(room)
    bad["placements"][0]["size"] = size
    with pytest.raises(ValueError, match="size"):
        validate_output(bad, room)


def test_json_round_trip_binding_and_prior_mapping_order_are_stable():
    room, sidecar = bind_room(prepared())
    reloaded = json.loads(json.dumps(room, allow_nan=False))
    assert messages(reloaded) == messages(room)
    sizes = prior(room)
    assert messages(room, "placement", sizes) == messages(room, "placement", dict(reversed(list(sizes.items()))))
    assert json.loads(json.dumps(sidecar)) == sidecar


@pytest.mark.parametrize("arm", ["joint", "placement"])
def test_output_cycle_through_fixed_parent_is_rejected(arm):
    raw = prepared()
    raw["fixed"][0]["parent"] = "asset-table"
    room, sidecar = bind_room(raw)
    output = target(room, arm)
    table = next(o for o in output["placements"] if o["id"] == sidecar["original_to_bound"]["asset-table"])
    table["on"] = sidecar["original_to_bound"]["asset-column"]
    with pytest.raises(ValueError, match="support"):
        validate_output(output, room, arm)
