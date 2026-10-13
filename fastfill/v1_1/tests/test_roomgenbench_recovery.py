"""Legacy showcase failures must remain failures even with renderable fallbacks."""
from __future__ import annotations

import copy
import base64
import json
import math
from pathlib import Path
import sys

import pytest

from fastfill.v1_1 import roomgenbench_v1_1 as showcase
from fastfill.v1_1.scene_v1_1 import gt_placements_v1_1, parse_v1_1


def _room():
    return {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "objects": [
        {"id": "box_1", "category": "box", "size": [9, 9, 9], "pos": [99, 99, 99], "yaw": 2.1},
        {"id": "box_2", "category": "box", "size": [8, 8, 8], "pos": [88, 88, 88], "yaw": 1.7},
    ]}


def test_invalid_support_cycle_is_replaced_without_reference_geometry():
    room = _room()
    before = copy.deepcopy(room)
    invalid = json.dumps({"placements": [
        {"id": "box_1", "on": "box_2", "size": [.5] * 3, "pos": [6, 2, 0], "yaw": 0},
        {"id": "box_2", "on": "box_1", "size": [.5] * 3, "pos": [6, 2, .5], "yaw": 0},
    ]})
    _, error = parse_v1_1(invalid, ["box_1", "box_2"])
    assert error == "bad_support_ref"
    recovered = showcase._fallback_placements_v1_1(room)
    assert all(p == {"size": [.5, .5, .6], "pos": [2, 2, 0], "yaw": 0., "on": None}
               for p in recovered.values())
    assert room == before


def test_fallback_uses_declared_support_and_wall_metadata_only():
    room = _room()
    room["objects"][1] = {**room["objects"][1], "parent": "box_1", "anchor": "wall"}
    recovered = showcase._fallback_placements_v1_1(room)
    assert recovered["box_2"]["on"] == "box_1"
    assert recovered["box_2"]["pos"][2] == 1.2
    recovered["box_2"]["size"][0] = 10
    assert recovered["box_1"]["size"][0] == .5


@pytest.mark.parametrize("parents", [("box_2", "box_1"), ("box_1", None), ("missing", None)])
def test_invalid_declared_support_fails_before_repair(parents):
    room = _room()
    room["objects"] = [{**obj, "parent": parent} for obj, parent in zip(room["objects"], parents)]
    with pytest.raises(ValueError, match="declared support"):
        showcase._fallback_placements_v1_1(room)


def test_existing_mesh_does_not_fabricate_generator_success(tmp_path):
    target = tmp_path / "results" / "embodiedgen_v2" / "assets"
    target.mkdir(parents=True)
    for index in range(184):
        (target / f"box_{index:03d}.glb").write_bytes(b"geometry not verified")
    assert showcase.ensure_roomgenbench_assets(tmp_path) == target
    receipts = [json.loads(p.read_text()) for p in target.glob("*.json")]
    assert len(receipts) == 184
    assert all(r["status"] == "fallback" for r in receipts)
    assert all("provenance" in r for r in receipts)


def test_legacy_three_field_success_receipt_is_downgraded(tmp_path):
    old = {"asset_key": "box", "method": "embodiedgen_v2", "status": "ok"}
    (tmp_path / "box.json").write_text(json.dumps(old))
    showcase._record_unverified_asset(tmp_path, "box", "embodiedgen_v2")
    receipt = json.loads((tmp_path / "box.json").read_text())
    assert receipt["status"] == "fallback"
    assert receipt["source_status"] == "ok"


def test_existing_generator_receipt_is_preserved(tmp_path):
    recorded = {"asset_key": "box", "method": "embodiedgen_v2", "status": "failed", "reason": "generator failed"}
    path = tmp_path / "box.json"
    path.write_text(json.dumps(recorded))
    prior = path.read_bytes()
    showcase._record_unverified_asset(tmp_path, "box", "embodiedgen_v2")
    assert path.read_bytes() == prior


@pytest.mark.parametrize("broken", [b"{broken", b"\xff"])
def test_one_invalid_existing_receipt_is_preserved_and_does_not_disable_other_assets(tmp_path, broken, capsys):
    target = tmp_path / "results" / "embodiedgen_v2" / "assets"
    target.mkdir(parents=True)
    for index in range(184):
        (target / f"box_{index:03d}.glb").write_bytes(b"geometry not verified")
    bad = target / "box_000.json"
    bad.write_bytes(broken)
    assert showcase.ensure_roomgenbench_assets(tmp_path) == target
    assert bad.read_bytes() == broken
    assert json.loads((target / "box_001.json").read_text())["status"] == "fallback"
    assert "Invalid asset receipt retained" in capsys.readouterr().out


def test_newly_recovered_asset_keeps_invalid_prior_receipt_diagnostic(tmp_path):
    (tmp_path / "box.json").write_text("{broken")
    showcase._record_unverified_asset(tmp_path, "box", "embodiedgen_v2", recovered_from="room.room.json")
    receipt = json.loads((tmp_path / "box.json").read_text())
    assert receipt["status"] == "fallback"
    assert receipt["source_receipt_error"] == "JSONDecodeError"


def test_recovered_web_mesh_is_labelled_fallback_with_original_status(tmp_path):
    recorded = {"asset_key": "box", "method": "embodiedgen_v2", "status": "failed", "reason": "generator failed"}
    path = tmp_path / "box.json"
    path.write_text(json.dumps(recorded))
    showcase._record_unverified_asset(tmp_path, "box", "embodiedgen_v2", recovered_from="bedroom.room.json")
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "fallback"
    assert receipt["source_status"] == "failed"
    assert receipt["source_room_export"] == "bedroom.room.json"


def test_recovered_mesh_receipt_cannot_turn_synthetic_box_into_success(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from fastfill.v2.roomgenbench import _asset_parts, _reference
    (tmp_path / "box.glb").write_bytes(trimesh.creation.box(extents=[.4, .5, .6]).export(file_type="glb"))
    showcase._record_unverified_asset(tmp_path, "box", "embodiedgen_v2")
    module, _ = _reference(Path(__file__).resolve().parents[3] / "RoomGenBench")
    parts, receipt = _asset_parts(module, {"asset_key": "box", "dimensions": {"width": .4, "length": .5, "height": .6}},
                                  "embodiedgen_v2", tmp_path)
    assert parts
    assert receipt["status"] == "fallback"


def test_web_export_recovery_records_fallback_after_extracting_mesh(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    scenes = tmp_path / "bench" / "inputs" / "scenes"
    web = tmp_path / "site" / "data" / "embodiedgen_v2"
    scenes.mkdir(parents=True)
    web.mkdir(parents=True)
    scene = {"objects": [{"asset_key": "box", "position": {"x": 0, "y": 0, "z": 0},
                           "rotation": {"x": 0, "y": 0, "z": 0}}]}
    (scenes / "room.json").write_text(json.dumps(scene))
    mesh = trimesh.Scene()
    mesh.add_geometry(trimesh.creation.box(), node_name="obj_000")
    payload = base64.b64encode(mesh.export(file_type="glb")).decode()
    (web / "room.room.json").write_text(json.dumps({"glb": payload}))
    assets = showcase.ensure_roomgenbench_assets(tmp_path)
    assert (assets / "box.glb").is_file()
    receipt = json.loads((assets / "box.json").read_text())
    assert receipt["status"] == "fallback"
    assert receipt["provenance"] == "recovered_room_export"
    assert receipt["source_room_export"].endswith("room.room.json")
    previous = (assets / "box.json").read_bytes()
    showcase.ensure_roomgenbench_assets(tmp_path)
    assert (assets / "box.json").read_bytes() == previous


def test_benchmark_reference_roundtrip_preserves_all_objects_sizes_yaws_and_openings():
    from fastfill.v2.direct_layout import layout_to_roomgenbench
    root = Path(__file__).resolve().parents[3] / "RoomGenBench" / "bench" / "inputs" / "scenes"
    paths = sorted(root.glob("*.json"))
    assert paths, "The bundled benchmark input scenes are required for this integration check"
    for path in paths:
        raw = json.loads(path.read_text())
        room, meta = showcase.roomgenbench_scene_to_v1_1_room(path)
        condition, layout = showcase.placements_to_v2_handoff(room, meta, gt_placements_v1_1(room))
        restored = layout_to_roomgenbench(condition, layout)
        assert len(restored["objects"]) == len(raw["objects"])
        for index, (actual, expected) in enumerate(zip(restored["objects"], raw["objects"])):
            assert actual["id"] == f"obj_{index:04d}"
            assert actual["type"] == expected["type"]
            assert actual["position"] == expected["position"]
            # V1.1's documented serialization/parser minimum is 1 cm per local axis.
            assert actual["dimensions"] == {k: max(.01, v) for k, v in expected["dimensions"].items()}
            assert abs(math.remainder(actual["rotation"]["z"] - expected["rotation"]["z"], 360.)) < 1e-10
        assert len(room["fixed"]) == sum(len(raw["room"].get(k, [])) for k in ("doors", "windows"))
        assert sum(o.get("anchor") == "wall" for o in room["objects"]) == sum(o["place_id"] == "wall" for o in raw["objects"])


def test_model_parse_failure_exports_truthful_fallback_end_to_end(tmp_path, monkeypatch):
    from fastfill.v1_1 import evaluate_v1_1
    import transformers
    scene_dir = tmp_path / "benchmark" / "bench" / "inputs" / "scenes"
    scene_dir.mkdir(parents=True)
    scene = {"scene_key": "simple", "room_type": "bedroom", "room": {"dimensions": {"width": 4, "length": 4, "height": 2.8}},
             "objects": [{"id": "source_box", "type": "box", "description": "box", "place_id": "floor",
                          "position": {"x": 1, "y": 1, "z": 0}, "rotation": {"x": 0, "y": 0, "z": 0},
                          "dimensions": {"length": .4, "width": .5, "height": .6}}]}
    (scene_dir / "simple.json").write_text(json.dumps(scene))
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    bad = json.dumps({"placements": [{"id": "box_1", "on": "box_1", "size": [.5, .5, .6], "pos": [6, 2, 0], "yaw": 0}]})
    monkeypatch.setattr(evaluate_v1_1, "generate_v1_1", lambda *a, **k: [bad])
    monkeypatch.setattr(showcase, "ensure_roomgenbench_assets", lambda *a, **k: None)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["showcase", "--model", "fake", "--scenes-dir", str(scene_dir), "--out-dir", str(output)])
    showcase.main()
    summary = json.loads((output / "summary.json").read_text())["simple"]
    assert summary["parse_error"] == "bad_support_ref"
    assert summary["used_fallback"] is True
    assert summary["geometry_source"] == "category_prior_fallback"
    assert summary["repaired_valid"] is False
    assert summary["model_generation_succeeded"] is False
    receipt = json.loads((output / "simple" / "generation_receipt.json").read_text())
    assert receipt["geometry_source"] == "category_prior_fallback"
    assert receipt["model_generation_succeeded"] is False
    assert (output / "simple" / "layout.glb").is_file()
    layout = json.loads((output / "simple" / "layout.json").read_text())
    assert layout["objects"][0]["bottom_center_m"] == [2., 2., 0.]
