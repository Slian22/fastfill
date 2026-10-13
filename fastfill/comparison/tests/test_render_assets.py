"""Renderer math, resource, and manifest contracts; no bpy or test assets."""
import copy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "fastfill/comparison/render_assets.py"
spec = importlib.util.spec_from_file_location("render_assets", MODULE_PATH)
render = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = render
spec.loader.exec_module(render)

HOLO = "OptiScene_holodeck"
FRONT = "IL3D_3dfront"
BOUNDS = ((1, 2, 3), (3, 8, 7))


def corners(bounds=BOUNDS):
    return list(itertools.product(*zip(*bounds)))


@pytest.mark.parametrize("source,expected", [(HOLO, (-3, 1, 2)), (FRONT, (3, 1, 2))])
def test_source_axes(source, expected):
    assert render.source_point((1, 2, 3), source) == expected


@pytest.mark.parametrize("source,flip", [(HOLO, True), (FRONT, False)])
def test_fit_union_bottom_center_and_reflection(source, flip):
    transform = render.build_transform(BOUNDS, source, (8, 4, 3), (5, 6, 7), 0)
    world = [transform.point(v) for v in corners()]
    lo, hi = render.vertex_bounds(world)
    assert lo == pytest.approx((1, 4, 7))
    assert hi == pytest.approx((9, 8, 10))
    assert transform.flip_winding is flip


def test_fit_precedes_yaw_and_translation():
    transform = render.build_transform(BOUNDS, FRONT, (8, 4, 3), (5, 6, 7), math.pi / 2)
    assert transform.point((1, 2, 3)) == pytest.approx((7, 2, 7))
    lo, hi = render.vertex_bounds([transform.point(v) for v in corners()])
    assert lo == pytest.approx((3, 2, 7))
    assert hi == pytest.approx((7, 10, 10))


@pytest.mark.parametrize("source,flip", [(HOLO, False), (FRONT, True)])
def test_negative_native_scale_applies_before_basis_and_positive_scale_not_twice(source, flip):
    a = render.build_transform(BOUNDS, source, (8, 4, 3), (0, 0, 0), 0, (-2, 3, 4))
    b = render.build_transform(BOUNDS, source, (8, 4, 3), (0, 0, 0), 0, (-1, 1, 1))
    assert a.flip_winding is flip
    for v in corners():
        assert a.point(v) == pytest.approx(b.point(v))
    assert a.point((1, 2, 3))[1] == pytest.approx(2)


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def test_normals_use_inverse_transpose_and_corrected_winding():
    transform = render.build_transform(((0, 0, 0), (2, 3, 4)), HOLO,
                                       (9, 5, 2), (4, 6, 1), 0.7)
    a, b, c = (0, 0, 0), (1, 1, 0), (0, 1, 1)
    points = [transform.point(v) for v in (a, c, b)]  # reflected winding
    edges = [tuple(p[i]-points[0][i] for i in range(3)) for p in points[1:]]
    normal = transform.normal(cross(b, c))
    face_normal = cross(*edges)
    norm = math.sqrt(sum(v*v for v in face_normal))
    assert normal == pytest.approx([v/norm for v in face_normal])


@pytest.mark.parametrize("kwargs", [
    {"size": (0, 1, 1)}, {"size": (1, -1, 1)}, {"yaw": float("nan")},
    {"pos": (0, float("inf"), 0)}, {"native_scale": (1, 0, 1)},
    {"source": "unknown"}, {"raw_bounds": ((0, 0, 0), (0, 1, 1))},
])
def test_invalid_transform_rejected(kwargs):
    args = dict(raw_bounds=BOUNDS, source=FRONT, size=(1, 2, 3), pos=(0, 0, 0), yaw=0)
    with pytest.raises(render.RenderReject):
        render.build_transform(**{**args, **kwargs})


def test_concave_floor_preserves_area_and_reversed_winding():
    polygon = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
    for boundary in (polygon, polygon[::-1]):
        triangles = render.triangulate_floor(boundary)
        area = sum(abs((boundary[b][0]-boundary[a][0])*(boundary[c][1]-boundary[a][1])
                       -(boundary[b][1]-boundary[a][1])*(boundary[c][0]-boundary[a][0]))/2
                   for a, b, c in triangles)
        assert len(triangles) == 4
        assert area == pytest.approx(12)


@pytest.mark.parametrize("boundary", [[], [(0, 0), (1, 1), (2, 2)],
    [(0, 0), (3, 3), (0, 2), (3, 0)], [(0, 0), (1, 0), (1, 0), (0, 1)]])
def test_bad_polygon_rejected(boundary):
    with pytest.raises(render.RenderReject):
        render.triangulate_floor(boundary)


@pytest.fixture
def asset(tmp_path):
    folder = tmp_path / "asset"
    folder.mkdir()
    mesh = folder / "raw_model.obj"
    mesh.write_text("mtllib model.mtl\no first\nv 1 2 3\nv 3 8 3\no second\nv 1 8 7\n"
                    "vt 0 0\nvt 1 0\nvt 0 1\nvn 1 0 0\nusemtl fabric\nf 1/1/1 2/2/1 3/3/1\n")
    (folder / "model.mtl").write_text("newmtl fabric\nKd 1 0 0\nmap_Kd texture.png\n")
    (folder / "texture.png").write_bytes(b"only resource presence checked without bpy")
    return mesh


def case_for(asset):
    obj = {"id": "chair", "size": [2, 1, 1], "pos": [2, 2, 0], "yaw": 0}
    return {"uid": "synthetic-dev", "source": FRONT, "boundary": [[0, 0], [5, 0], [5, 4], [0, 4]],
            "height": 2.8, "fixed": [], "expected_ids": ["chair"],
            "asset_mapping": [{"prompt_id": "chair", "asset_id": "asset", "mesh_path": str(asset),
                               "source": FRONT, "layout_object": {"scale": [1, 1, 1]}}],
            "layouts": {"rawGT": [copy.deepcopy(obj)], "candidate": [copy.deepcopy(obj)]}}


def test_temporary_obj_reflects_faces_normals_and_keeps_raw_unchanged(asset, tmp_path):
    original = {p.name: p.read_bytes() for p in asset.parent.iterdir()}
    resources = render.inspect_asset(asset, "asset")
    assert resources["bounds"] == BOUNDS
    transform = render.build_transform(BOUNDS, HOLO, (8, 4, 3), (5, 6, 7), 0)
    dest = tmp_path / "copy"
    dest.mkdir()
    prepared = render.prepare_obj(resources, transform, dest)
    text = prepared.read_text()
    assert "f 3/3/1 2/2/1 1/1/1" in text
    assert "vt 1 0" in text
    assert {p.name: p.read_bytes() for p in asset.parent.iterdir()} == original
    assert str((asset.parent / "texture.png").resolve()) in resources["materials"][0]["textures"]


def test_prepared_mtl_uses_local_texture_links_for_blender(asset, tmp_path):
    resources = render.inspect_asset(asset, "asset")
    transform = render.build_transform(BOUNDS, FRONT, (2, 1, 1), (0, 0, 0), 0)
    dest = tmp_path / "local-links"
    dest.mkdir()
    render.prepare_obj(resources, transform, dest)
    line = next(line for line in (dest / "material-0.mtl").read_text().splitlines() if line.startswith("map_Kd"))
    texture = Path(line.split(maxsplit=1)[1])
    assert not texture.is_absolute()  # Blender prepends the OBJ dir even to absolute MTL paths.
    assert (dest / texture).read_bytes() == (asset.parent / "texture.png").read_bytes()


def test_only_documented_mtl_override_is_allowed(asset):
    mtl = asset.parent / "model.mtl"
    mtl.rename(asset.parent / "raw_model.mtl")
    with pytest.raises(render.RenderReject, match="missing_mtl"):
        render.inspect_asset(asset, "asset")
    resources = render.inspect_asset(asset, "f350c1ba-8040-4750-9e6e-f45cfbf916a4")
    assert resources["overrides"] == {"model.mtl": "raw_model.mtl"}


def test_missing_material_definition_or_texture_rejected(asset):
    (asset.parent / "model.mtl").write_text("newmtl wrong\n")
    with pytest.raises(render.RenderReject, match="undefined_material"):
        render.inspect_asset(asset, "asset")
    (asset.parent / "model.mtl").write_text("newmtl fabric\nmap_Kd missing.png\n")
    with pytest.raises(render.RenderReject, match="missing_texture"):
        render.inspect_asset(asset, "asset")


def test_candidate_dimensions_and_camera_independent_of_gt(asset):
    case = case_for(asset)
    original = copy.deepcopy(case)
    case["layouts"]["rawGT"][0]["size"] = [50, 70, 20]
    plan = render.plan_case(case)
    obj = plan["layouts"]["candidate"][0]
    assert obj["size"] == [2, 1, 1]
    assert plan["cameras"] == render.plan_case(original)["cameras"]
    assert case["layouts"]["candidate"] == original["layouts"]["candidate"]


@pytest.mark.parametrize("change,reason", [
    (lambda c: c["layouts"]["candidate"].clear(), "incomplete_inventory"),
    (lambda c: c["layouts"]["candidate"].append(copy.deepcopy(c["layouts"]["candidate"][0])), "duplicate_id"),
    (lambda c: c["layouts"]["candidate"][0].pop("size"), "incomplete_placement"),
    (lambda c: c["layouts"]["candidate"][0].update(size=[float("nan"), 1, 1]), "invalid_number"),
    (lambda c: c.update(asset_mapping=[]), "no_asset"),
    (lambda c: c.update(schema_complete=False), "incomplete_case"),
    (lambda c: c["fixed"].append({"id": "desk", "size": [1, 1, 1], "pos": [0, 0, 0], "yaw": 0}), "no_asset"),
])
def test_case_failures_retained_in_ledger(asset, tmp_path, change, reason):
    case = case_for(asset)
    change(case)
    ledger = render.run_cases([case], tmp_path / "output", dry_run=True)
    assert ledger["cases"][0]["status"] == "rejected"
    assert reason in ledger["cases"][0]["reason"]
    assert json.loads((tmp_path / "output/ledger.json").read_text()) == ledger
    assert not list((tmp_path / "output").rglob("*.png"))


def test_metric_adapter_uses_only_selected_candidate_and_context(asset):
    base = case_for(asset)
    archived = {**base, "stage": "final", "judging_request": {"objects": [{"id": "chair"}]},
                "gt_canonicalobjects": base["layouts"]["rawGT"],
                "raw_layout": base["layouts"]["candidate"], "repaired_layout": [],
                "parsed": True, "schema_complete": True}
    row = {**base["asset_mapping"][0], "uid": base["uid"], "gt_object": {"size": [90, 90, 90]}}
    converted = render.metric_case(archived, [row])
    assert converted["layouts"]["candidate"] == archived["raw_layout"]
    assert converted["expected_ids"] == ["chair"]
    assert render.plan_case(converted)["layouts"]["candidate"][0]["size"] == [2, 1, 1]


def test_dry_run_accepts_structure_context_without_furniture_boxes(asset, tmp_path):
    case = case_for(asset)
    case["fixed"] = [{"id": "door", "category": "door", "structure": True,
                      "size": [1, .1, 2], "pos": [0, 2, 0], "yaw": math.pi/2}]
    receipt = render.run_cases([case], tmp_path / "output", dry_run=True)
    assert receipt["cases"][0]["status"] == "ready"
    assert "bpy" not in sys.modules


def test_portable_asset_root_uses_archived_filename(asset):
    case = case_for(asset)
    case["asset_mapping"][0]["mesh_path"] = "/different/machine/asset/raw_model.obj"
    plan = render.plan_case(case, asset_root=asset.parent.parent)
    assert plan["layouts"]["candidate"][0]["resource"]["mesh_path"] == str(asset.resolve())


@pytest.mark.parametrize("case", [None, [], "bad-case", {"uid": "incomplete"}])
def test_malformed_cases_get_receipts(case, tmp_path):
    ledger = render.run_cases([case], tmp_path / "output", dry_run=True)
    assert ledger["cases"][0]["status"] == "rejected"
    assert ledger["cases"][0]["reason"]


def test_no_overwrite_of_previous_ledger(asset, tmp_path):
    render.run_cases([case_for(asset)], tmp_path / "output", dry_run=True)
    with pytest.raises(render.RenderReject, match="output_exists"):
        render.run_cases([case_for(asset)], tmp_path / "output", dry_run=True)


def test_cli_portable_manifest_and_future_resolution(asset, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [case_for(asset)]}))
    output = tmp_path / "out"
    assert render.main(["--manifest", str(manifest), "--output", str(output),
                        "--dry-run", "--top", "--resolution", "1023"]) == 0
    ledger = json.loads((output / "ledger.json").read_text())
    assert ledger["config"]["resolution"] == 1023
    assert ledger["config"]["top"] is True


def test_cli_denoising_is_explicit_and_recorded(asset, tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case_for(asset)))
    output = tmp_path / "denoised"
    assert render.main(["--case", str(path), "--output", str(output), "--dry-run",
                        "--denoise", "--samples", "32"]) == 0
    config = json.loads((output / "ledger.json").read_text())["config"]
    assert config["denoise"] is True
    assert config["samples"] == 32


def test_image_validation_forces_lazy_blender_decode():
    class LazyImage:
        source, filepath = "FILE", "valid-texture.jpg"
        has_data, size = False, (1024, 1024)

        @property
        def pixels(self):
            self.has_data = True
            return [1, 1, 1, 1]

    img = LazyImage()
    render.validate_images([img])
    assert img.has_data


def test_image_validation_rejects_empty_decodes():
    class EmptyImage:
        source, filepath = "FILE", "broken-texture.jpg"
        has_data, size, pixels = False, (0, 0), []

    with pytest.raises(render.RenderReject, match="texture_decode_failed"):
        render.validate_images([EmptyImage()])


@pytest.mark.parametrize("missing", [False, True])
def test_unknown_height_uses_shared_three_metre_policy(asset, tmp_path, missing):
    case = {**case_for(asset), "height": None}
    if missing:
        case.pop("height")
    original = copy.deepcopy(case)
    plan = render.plan_case(case)
    changed = {**case, "layouts": {**case["layouts"], "candidate": [
        {**case["layouts"]["candidate"][0], "size": [90, 80, 70]}]}}
    assert plan["height"] == 3.0
    assert plan["cameras"] == render.plan_case(changed)["cameras"]
    assert plan["cameras"] == render.plan_case({**case, "height": 3.0})["cameras"]
    receipt = render.run_cases([case], tmp_path / "height", dry_run=True)["cases"][0]
    assert receipt["height_provenance"] == {
        "input_height": None, "effective_height": 3.0,
        "policy": "unknown_uniform_3m", "imputed": True}
    assert case == original


@pytest.mark.parametrize("height,override,effective,policy", [
    (2.8, None, 2.8, "case_height"), (None, 4.1, 4.1, "case_override"),
    (2.8, 4.1, 4.1, "case_override"),
])
def test_explicit_height_policy(asset, height, override, effective, policy):
    case = {**case_for(asset), "height": height}
    if override is not None:
        case = {**case, "render_height": override}
    plan = render.plan_case(case)
    assert plan["height"] == effective
    assert plan["height_provenance"]["policy"] == policy
    assert plan["height_provenance"]["imputed"] is False


@pytest.mark.parametrize("field", ["height", "render_height"])
@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf"), True, "3"])
def test_invalid_height_retains_receipt_and_continues(asset, tmp_path, field, value):
    bad = {**case_for(asset), field: value}
    good = {**case_for(asset), "uid": "next"}
    ledger = render.run_cases([bad, good], tmp_path / "heights", dry_run=True)
    assert [c["status"] for c in ledger["cases"]] == ["rejected", "ready"]
    assert "height" in ledger["cases"][0]["reason"]
    assert ledger["cases"][0]["images"] == {}
    assert json.loads((tmp_path / "heights/ledger.json").read_text()) == ledger


@pytest.mark.parametrize("patch", [
    {"layouts": ["candidate"]}, {"layouts": {"candidate": "bad"}},
    {"layouts": {"candidate": [None]}}, {"asset_mapping": [None]},
    {"asset_mapping": {"chair": None}}, {"fixed": [None]},
    {"expected_ids": "chair"}, {"uid": float("nan")}, {"stage": float("inf")},
])
def test_nested_malformed_case_does_not_abort_later_receipts(asset, tmp_path, patch):
    ledger = render.run_cases([{**case_for(asset), **patch}, case_for(asset)],
                              tmp_path / "malformed", dry_run=True)
    assert [c["status"] for c in ledger["cases"]] == ["rejected", "ready"]
    assert ledger["cases"][0]["reason"]
    assert ledger["cases"][0]["images"] == {}
    assert json.loads((tmp_path / "malformed/ledger.json").read_text()) == ledger


def archive_for(asset):
    base = case_for(asset)
    record = {**base, "stage": "final", "judging_request": {"objects": [{"id": "chair"}]},
              "gt_canonicalobjects": base["layouts"]["rawGT"],
              "raw_layout": base["layouts"]["candidate"],
              "repaired_layout": [{**base["layouts"]["candidate"][0], "pos": [3, 2, 0]}]}
    return record, [{**base["asset_mapping"][0], "uid": base["uid"]}]


@pytest.mark.parametrize("repaired", [False, True])
def test_archive_cli_records_variant_and_handles_missing_height(asset, tmp_path, repaired):
    record, rows = archive_for(asset)
    record.pop("height")
    path, mapping = tmp_path / "archive.json", tmp_path / "mapping.json"
    path.write_text(json.dumps({"cases": [None, record]}))
    mapping.write_text(json.dumps(rows))
    args = ["--metric-inputs", str(path), "--asset-map", str(mapping), "--uid", record["uid"],
            "--stage", "final", "--output", str(tmp_path / "out"), "--dry-run"]
    assert render.main(args + (["--repaired"] if repaired else [])) == 0
    receipt = json.loads((tmp_path / "out/ledger.json").read_text())["cases"][0]
    assert receipt["layout_variant"] == ("repaired" if repaired else "raw")
    selected = record["repaired_layout" if repaired else "raw_layout"]
    expected = hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert receipt["layout_sha256"]["candidate"] == expected
    assert receipt["height_provenance"]["effective_height"] == 3.0


def test_selected_archive_adapter_error_has_ledger(asset, tmp_path):
    record, rows = archive_for(asset)
    record.pop("raw_layout")
    path, mapping = tmp_path / "archive.json", tmp_path / "mapping.json"
    path.write_text(json.dumps({"cases": [record]}))
    mapping.write_text(json.dumps(rows))
    assert render.main(["--metric-inputs", str(path), "--asset-map", str(mapping),
                        "--uid", record["uid"], "--stage", "final", "--dry-run",
                        "--output", str(tmp_path / "out")]) == 1
    receipt = json.loads((tmp_path / "out/ledger.json").read_text())["cases"][0]
    assert receipt["status"] == "rejected"
    assert "raw_layout" in receipt["reason"]
    assert receipt["images"] == {}
    assert receipt["layout_variant"] == "raw"


@pytest.mark.parametrize("contents", ["{broken", '{"cases": null}'])
def test_malformed_input_document_has_ledger(tmp_path, contents):
    path = tmp_path / "manifest.json"
    path.write_text(contents)
    assert render.main(["--manifest", str(path), "--output", str(tmp_path / "out"), "--dry-run"]) == 1
    receipt = json.loads((tmp_path / "out/ledger.json").read_text())["cases"][0]
    assert receipt["status"] == "rejected"
    assert receipt["reason"]


def test_receipt_hashes_bind_layout_material_and_texture_bytes(asset, tmp_path):
    case = case_for(asset)
    first = render.run_cases([case], tmp_path / "first", dry_run=True)["cases"][0]
    texture, mtl = asset.parent / "texture.png", asset.parent / "model.mtl"
    material = first["assets"][0]["materials"][0]
    assert material["sha256"] == hashlib.sha256(mtl.read_bytes()).hexdigest()
    assert material["texture_sha256"] == {str(texture.resolve()): hashlib.sha256(texture.read_bytes()).hexdigest()}
    texture.write_bytes(b"different texture fixture, not an image")
    mtl.write_text(mtl.read_text().replace("Kd 1 0 0", "Kd 0 1 0"))
    changed = {**case, "layouts": {**case["layouts"], "candidate": [
        {**case["layouts"]["candidate"][0], "size": [3, 1, 1]}]}}
    second = render.run_cases([changed], tmp_path / "second", dry_run=True)["cases"][0]
    assert first["input_sha256"] != second["input_sha256"]
    assert first["layout_sha256"]["rawGT"] == second["layout_sha256"]["rawGT"]
    assert first["layout_sha256"]["candidate"] != second["layout_sha256"]["candidate"]
    assert first["assets"][0]["sha256"] == second["assets"][0]["sha256"]
    assert material["sha256"] != second["assets"][0]["materials"][0]["sha256"]
    assert material["texture_sha256"] != second["assets"][0]["materials"][0]["texture_sha256"]
    assert first["image_sha256"] == second["image_sha256"] == {}


@pytest.mark.parametrize("omit_view", [False, True])
def test_post_render_hashes_and_missing_output_atomicity(asset, tmp_path, monkeypatch, omit_view):
    # Stub byte files exercise publication only; no Blender, image generation or decoding.
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(app=SimpleNamespace(version_string="stub")))
    def fake_render(bpy, plan, objects, directory, views, *settings):
        for view in views:
            if not (omit_view and directory.name == "candidate" and view == "top"):
                (directory / f"{view}.png").write_bytes(f"stub:{directory.name}:{view}".encode())
    monkeypatch.setattr(render, "render_layout", fake_render)
    output = tmp_path / "publication"
    receipt = render.run_cases([case_for(asset)], output, top=True)["cases"][0]
    if omit_view:
        assert receipt["status"] == "failed"
        assert receipt["images"] == receipt["image_sha256"] == {}
        assert not list(output.rglob("*.png"))
    else:
        assert receipt["status"] == "rendered"
        for label, views in receipt["images"].items():
            for view, path in views.items():
                assert receipt["image_sha256"][label][view] == hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_layout_named_fixed_retains_its_asset_receipts(asset, tmp_path):
    case = case_for(asset)
    case = {**case, "layouts": {"fixed": case["layouts"]["candidate"]}}
    receipt = render.run_cases([case], tmp_path / "fixed-label", dry_run=True)["cases"][0]
    assert receipt["status"] == "ready"
    assert len(receipt["assets"]) == 1


def test_overflowing_camera_gets_rejection_and_next_case_survives(asset, tmp_path):
    bad = {**case_for(asset), "height": 1e308}
    ledger = render.run_cases([bad, case_for(asset)], tmp_path / "overflow", dry_run=True)
    assert [c["status"] for c in ledger["cases"]] == ["rejected", "ready"]
    assert json.loads((tmp_path / "overflow/ledger.json").read_text()) == ledger


def test_failed_raw_candidate_keeps_variant_and_height_assumption(asset, tmp_path):
    record, rows = archive_for(asset)
    case = render.metric_case({**record, "height": None}, rows)
    bad = {**case, "asset_mapping": []}
    receipt = render.run_cases([bad], tmp_path / "missing-mesh", dry_run=True)["cases"][0]
    assert receipt["status"] == "rejected"
    assert receipt["layout_variant"] == "raw"
    assert receipt["height_provenance"]["imputed"] is True
    assert receipt["images"] == receipt["image_sha256"] == {}


def test_adapter_failure_does_not_abort_next_case(asset, tmp_path):
    bad = {**case_for(asset), "uid": "bad"}
    def adapter(case):
        if case["uid"] == "bad":
            raise KeyError("raw_layout")
        return case
    ledger = render.run_cases([bad, case_for(asset)], tmp_path / "adapter", dry_run=True, case_adapter=adapter)
    assert [c["status"] for c in ledger["cases"]] == ["rejected", "ready"]


@pytest.mark.parametrize("tail,expected", [
    ("-bm 0.5 texture.png", ("-bm 0.5", "texture.png")),
    ('-s 1 2 3 -o -1 0 "texture with spaces.png"', ("-s 1 2 3 -o -1 0", "texture with spaces.png")),
])
def test_texture_options_preserve_parameters_and_filename(tail, expected):
    assert render.texture_spec(tail) == expected


@pytest.mark.parametrize("tail", ["", "-unknown 1 texture.png", "-bm 1"])
def test_invalid_texture_options_are_rejected(tail):
    with pytest.raises(render.RenderReject):
        render.texture_spec(tail)


def test_shared_cutaway_room_preserves_concave_floor(asset, monkeypatch):
    case = {**case_for(asset), "height": None,
            "boundary": [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]}
    plan = render.plan_case(case)
    meshes = []
    monkeypatch.setattr(render, "material", lambda bpy, name, color: name)
    monkeypatch.setattr(render, "make_mesh", lambda bpy, name, vertices, faces, material:
                        meshes.append((name, vertices, faces)))
    render.make_room(None, plan)
    floor = meshes[0]
    assert floor[1] == [(x, y, 0) for x, y in case["boundary"]]
    assert len(floor[2]) == 4
    assert meshes[1:]
    for name, vertices, faces in meshes[1:]:
        assert name == "cutaway-wall"
        assert max(v[2] for v in vertices) == 3.0
        assert render.turn(vertices[0], vertices[1], plan["cameras"]["perspective"]["location"]) >= 0
    forward = copy.deepcopy(meshes)
    meshes.clear()
    render.make_room(None, render.plan_case({**case, "boundary": case["boundary"][::-1]}))
    assert meshes[1:] == forward[1:]


@pytest.mark.parametrize("failure,reason", [
    (None, None), ("empty", "import_empty"), ("faces", "import_empty"),
    ("materials", "import_material_missing"), ("null_material", "import_material_missing"),
    ("axis", "import_axis_mismatch"),
])
def test_import_validation_without_blender_or_images(asset, tmp_path, failure, reason):
    plan = render.plan_case(case_for(asset))
    obj = plan["layouts"]["candidate"][0]
    points = [obj["transform"].point(v) for v in corners()]
    class Identity:
        def __matmul__(self, point):
            return point
    class Imported:
        type = "MESH"
        matrix_world = Identity()
    imported = Imported()
    vertices = [(x + (100 if failure == "axis" else 0), y, z) for x, y, z in points]
    materials = [] if failure == "materials" else [None if failure == "null_material" else object()]
    imported.data = SimpleNamespace(vertices=[SimpleNamespace(co=v) for v in vertices],
                                    polygons=[] if failure == "faces" else [object()], materials=materials)
    objects = []
    def fake_import(**kwargs):
        assert kwargs["forward_axis"] == "Y" and kwargs["up_axis"] == "Z"
        assert Path(kwargs["filepath"]).is_file()
        if failure != "empty":
            objects.append(imported)
    bpy = SimpleNamespace(data=SimpleNamespace(objects=objects),
                          ops=SimpleNamespace(wm=SimpleNamespace(obj_import=fake_import)))
    folder = tmp_path / "import"
    folder.mkdir()
    if reason:
        with pytest.raises(render.RenderReject, match=reason):
            render.import_furniture(bpy, obj, folder)
    else:
        render.import_furniture(bpy, obj, folder)
        assert imported.name == "chair"
