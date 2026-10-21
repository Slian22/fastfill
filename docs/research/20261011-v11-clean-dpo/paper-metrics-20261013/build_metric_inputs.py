"""Offline, diagnostic-only layout export; never load a model or generate responses.

Run with Python from any directory. Outputs contain flat UID/stage cases; each
layout is a list of canonicalobjects (size/pos in metres, bottom-centre, yaw in
radians). Fixed boxes in judging_request retain the original input's degree yaw.
No GT geometry is substituted for missing/invalid model predictions. The embedded
manifest is published only after every saved raw/repaired OOR has been checked.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fastfill.v1_1.postprocess import check_v1_1, repair_v1_1
from fastfill.v1_1.scene_v1_1 import apply_v1_1, parse_v1_1, user_json_v1_1

STAGES = ("sft", "dpo1", "final")
SOURCES = ("IL3D_3dfront", "OptiScene_holodeck")
PAPER_STRATA = tuple((source, kind) for source, kinds in (
    (SOURCES[0], ("bedroom", "living room")),
    (SOURCES[1], ("bedroom", "living room", "kitchen", "bathroom")),
) for kind in kinds)
DEPENDENCIES = (
    "fastfill/__init__.py", "fastfill/v1_1/__init__.py", "fastfill/scene.py",
    "fastfill/anchors.py", "fastfill/validate.py", "fastfill/interface.py",
    "fastfill/build.py", "fastfill/split.py", "fastfill/evaluate.py",
    "fastfill/v1_1/scene_v1_1.py", "fastfill/v1_1/postprocess.py",
)
# Exact source-segment SHA256, verified read-only against frozen server snapshot
# 3e3eef72c515. Source segments avoid Python-version-specific AST dump fields.
FROZEN_FUNCTION_SHA256 = {
    "_size_iou_3d": "36d0b861e5c0752d3790cbbc4255311f886f8fafce181b403d089840cccc40bf",
    "_sizes_realistic": "6204b58f185a8fceaace1d478b5c7a2b24237b396d3fa0a441923af4f857e8cf",
    "geometry_v1_1": "9bdc84fbcb44a3d91446425decc13695754bb8e13a75166e2fbbcaf56618f36d",
    "judge_v1_1": "b3be901f6b932097e2915b3a0fdb513e6ae113f840ca041977b6c88f6d71d8a4",
    "score_v1_1": "1552792e37f4c74acfdf296f6eda825d354562fdb05896cb859cb689e7f7091c",
    "_room_chunks": "739f52739afadb23355e98db776b07a36f41827f0a41013e4fd9dbc74f79a42b",
    "_score_chunk": "bbea0c04fa8fbad471e20db44714cda1188b0c8e9d2ab2ca018f6381496ff739",
}


class BuildError(ValueError):
    """Input identity, frozen implementation, or saved-score mismatch."""


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify_sources() -> dict:
    path = HERE.parent / "source-manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["source_sha256"] != "3e3eef72c51546cc0a22134e4409853570b988c4e5932823182d42007f6197c4":
        raise BuildError("Frozen source package mismatch")
    actual = {name: digest((REPO_ROOT / name).read_bytes()) for name in DEPENDENCIES}
    for name, value in actual.items():
        if value != manifest["files"][name]:
            raise BuildError(f"source hash mismatch: {name}")
    evaluator = REPO_ROOT / "fastfill/v1_1/evaluate_v1_1.py"
    source = evaluator.read_text()
    functions = {node.name: digest(ast.get_source_segment(source, node).encode())
                 for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in FROZEN_FUNCTION_SHA256}
    if functions != FROZEN_FUNCTION_SHA256:
        raise BuildError("evaluator source function mismatch")
    return {"verified": True, "frozen_package_sha256": manifest["source_sha256"],
            "source_manifest_sha256": digest(path.read_bytes()), "files": actual,
            "evaluator_current_sha256": digest(evaluator.read_bytes()),
            "evaluator_frozen_sha256": manifest["files"]["fastfill/v1_1/evaluate_v1_1.py"],
            "evaluator_verified_function_sha256": functions,
            "builder_sha256": digest(Path(__file__).read_bytes())}


def judging_request(room: dict) -> dict:
    """Whitelist declared request fields; never copy movable target geometry."""
    request = json.loads(user_json_v1_1(room))
    return {**{key: request[key] for key in ("room_type", "boundary_type", "boundary", "fixed")
               if key in request},
            "objects": [{**obj, "category": obj["id"].rsplit("_", 1)[0].replace("_", " ")}
                        for obj in request["objects"]]}


def canonicalobjects(objects: list[dict]) -> list[dict]:
    """Keep frozen request order/IDs, without re-canonicalizing or renumbering."""
    return [{"id": obj["id"],
             "category": obj.get("category") or obj["id"].rsplit("_", 1)[0].replace("_", " "),
             **({"desc": obj["desc"]} if obj.get("desc") else {}),
             "size": list(obj["size"]), "pos": list(obj["pos"]),
             "yaw": obj["yaw"], "on": obj.get("parent")}
            for obj in objects]


def layout_objects(room: dict, placements: dict) -> list[dict]:
    predicted = apply_v1_1(room, placements)["objects"]
    known = {obj["id"] for obj in room["objects"]}
    extras = [{"id": oid, **placement, "parent": placement.get("on")}
              for oid, placement in placements.items() if oid not in known]
    return canonicalobjects(predicted + extras)


def assert_saved(saved: dict, result: dict) -> None:
    for key in ("parsed", "error"):
        if key not in saved or saved[key] != result[key]:
            raise BuildError(f"{result['stage']}:{result['evaluation_row']} {key} mismatch")
    for key in ("oor", "oor_repaired"):
        actual, expected = result[key], saved.get(key)
        matches = actual is None and expected is None
        if actual is not None:
            matches = (isinstance(expected, (int, float)) and not isinstance(expected, bool)
                       and math.isfinite(expected)
                       and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
        if not matches:
            raise BuildError(f"{result['stage']}:{result['evaluation_row']} {key} mismatch: {actual} != {expected}")


def build_case(task: tuple) -> dict:
    scene, stage, saved = task
    room, index = scene["record"], scene["evaluation_row"]
    text = saved["output"]
    if text is not None and not isinstance(text, str):
        raise BuildError("Prediction output must be text or null")
    placements, error = ({}, "missing_generation") if text is None else parse_v1_1(
        text, [obj["id"] for obj in room["objects"]])
    parsed = text is not None and error not in ("truncated_or_bad_json", "bad_json")
    request = judging_request(room)
    result = {"uid": room["uid"], "evaluation_row": index, "stage": stage,
              "source": room["source"], "group": room.get("group"),
              "room_type": request.get("room_type"), "asset_mapping": {},
              "judging_request": request,
              "judging_requirement_text": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
              "boundary": room["boundary"], "boundary_type": room["boundary_type"],
              "height": room.get("height"), "fixed": room.get("fixed", []),
              "gt_canonicalobjects": canonicalobjects(room["objects"]),
              "raw_output": text, "raw_layout": layout_objects(room, placements),
              "repaired_layout": None, "parsed": parsed, "schema_complete": error is None,
              "error": error, "oor": None, "oor_repaired": None,
              "valid_raw": False, "valid_repaired": False}
    if error is None:
        raw = check_v1_1(apply_v1_1(room, placements))
        repaired = repair_v1_1(room, placements, resolve_collisions=True)
        fixed = check_v1_1(apply_v1_1(room, repaired))
        result = {**result, "repaired_layout": layout_objects(room, repaired),
                  "oor": raw["oor"], "oor_repaired": fixed["oor"],
                  "valid_raw": raw["valid"], "valid_repaired": fixed["valid"]}
    assert_saved(saved, result)
    return {**result, "saved_oor_verified": True}


def identity(scene: dict) -> tuple:
    return scene["evaluation_row"], scene["record"]["uid"]


def make_tasks(export: dict) -> list[tuple]:
    scenes = export["data"]
    if not scenes or set(export["predictions"]) != set(STAGES):
        raise BuildError("Missing scenes or incorrect stage set")
    keys = [identity(scene) for scene in scenes]
    if len(set(keys)) != len(keys) or len({k[0] for k in keys}) != len(keys):
        raise BuildError("Duplicate evaluation-row identity")
    if any(scene["record"]["source"] not in SOURCES for scene in scenes):
        raise BuildError("Unexpected source")
    indexed = {}
    for stage in STAGES:
        rows = export["predictions"][stage]
        lookup = {(row["evaluation_row"], row["uid"]): row for row in rows}
        if len(lookup) != len(rows) or set(lookup) != set(keys):
            raise BuildError(f"{stage}: missing/extra/duplicate prediction identities")
        indexed[stage] = lookup
    tasks = [(scene, stage, indexed[stage][identity(scene)]) for scene in scenes for stage in STAGES]
    for scene, stage, saved in tasks:
        room = scene["record"]
        if saved["source"] != room["source"] or saved["n"] != len(room["objects"]):
            raise BuildError(f"{stage}: prediction source/object-count mismatch")
        if saved.get("group") != room.get("group"):
            raise BuildError(f"{stage}: prediction group mismatch")
    return tasks


def stage_summary(rows: list[dict]) -> dict:
    complete = [row for row in rows if row["schema_complete"]]
    count = len(complete)
    mean = lambda key: math.fsum(row[key] for row in complete) / count if count else None
    raw, repaired = mean("oor"), mean("oor_repaired")
    return {"requests": len(rows), "complete": count, "failed": len(rows) - count,
            "parse_or_generation_failures": sum(not row["parsed"] for row in rows),
            "schema_failures": sum(row["parsed"] and not row["schema_complete"] for row in rows),
            "failure_by_error": dict(Counter(row["error"] for row in rows if row["error"])),
            "invalid_geometry_raw": sum(not row["valid_raw"] for row in complete),
            "invalid_geometry_repaired": sum(not row["valid_repaired"] for row in complete),
            "oor_denominator_complete": count,
            "mean_oor_fraction": raw, "mean_oor_repaired_fraction": repaired,
            "mean_oor_percent": None if raw is None else raw * 100,
            "mean_oor_repaired_percent": None if repaired is None else repaired * 100}


def aggregate(cases: list[dict]) -> dict:
    return {"scene_count": len({(row["evaluation_row"], row["uid"]) for row in cases}),
            "stages": {stage: stage_summary([row for row in cases if row["stage"] == stage])
                       for stage in STAGES}}


def summarize(cases: list[dict]) -> dict:
    paper = [row for row in cases if (row["source"], row["room_type"]) in PAPER_STRATA]
    strata = {f"{source}/{kind}": aggregate([row for row in paper
              if (row["source"], row["room_type"]) == (source, kind)]) for source, kind in PAPER_STRATA}
    return {"all_sources": aggregate(cases),
            "by_source": {source: aggregate([row for row in cases if row["source"] == source])
                          for source in SOURCES},
            "paper_six_strata": {**aggregate(paper), "strata": strata,
                "excluded_scene_count": (len(cases) - len(paper)) // len(STAGES),
                "selection": "Exact source and normalized room_type; living dining room is not merged into living room.",
                "same_paper_test_cohort_claim": False},
            "verification": {"verified_cases": len(cases),
                "numerical_oor_comparisons": 2 * sum(row["schema_complete"] for row in cases),
                "failure_oor_absence_checks": 2 * sum(not row["schema_complete"] for row in cases),
                "mismatches": 0, "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12}}


def build_documents(export: dict, workers: int = 1) -> tuple[dict, dict]:
    tasks = make_tasks(export)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cases = list(pool.map(build_case, tasks, chunksize=4))
    else:
        cases = [build_case(task) for task in tasks]
    common = {"schema_version": 1, "interpretation": "diagnostic_only",
              "target_independent_claim": False,
              "limitations": ["Target-derived inventory order/IDs are inherited.",
                  "No official-paper cohort-equivalence claim.",
                  "No visual scores or asset mappings supplied.",
                  "OOR means condition on schema-complete outputs; failures stay in request counts.",
                  "Geometry remains bounding-box based; raw and repaired predictions are separate."],
              "coordinate_system": {"units": "metres", "up": "Z", "position": "bbox_bottom_center",
                                    "layout_yaw": "radians", "judging_fixed_yaw": "degrees"}}
    return {**common, "cases": cases}, {**common, **summarize(cases)}


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def build_files(source: Path, destination: Path, workers: int = 1) -> dict:
    outputs = [destination / "metric-inputs.json", destination / "oor-summary.json"]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("Refusing to overwrite metric outputs")
    binding = verify_sources()
    content = source.read_bytes()
    export = json.loads(content)
    inputs, summary = build_documents(export, workers)
    if source.read_bytes() != content or verify_sources() != binding:
        raise BuildError("Input/source identity changed during processing")
    manifest = {"status": "verified", "input_path": str(source.resolve()),
                "input_sha256": digest(content), "scene_count": len(export["data"]),
                "case_count": len(inputs["cases"]), "all_saved_oor_values_match": True,
                "source_binding": binding, "source_test": export.get("source_test"),
                "subset_rule": export.get("subset_rule"), "receipts": export.get("receipts"),
                "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "shapely")}}
    summary_bytes = json_bytes({**summary, "manifest": manifest})
    input_bytes = json_bytes({**inputs, "manifest": {**manifest, "oor_summary_sha256": digest(summary_bytes)}})
    # Validation/serialization is complete before either publication; create-only.
    with outputs[0].open("xb") as handle:
        handle.write(input_bytes)
    with outputs[1].open("xb") as handle:
        handle.write(summary_bytes)
    return {"metric_inputs": str(outputs[0]), "oor_summary": str(outputs[1]),
            "cases": len(inputs["cases"]), "metric_inputs_sha256": digest(input_bytes),
            "oor_summary_sha256": digest(summary_bytes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "inputs/server-paper-source-records.json")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    print(json.dumps(build_files(args.input, args.out_dir, args.workers), indent=2))


if __name__ == "__main__":
    main()
