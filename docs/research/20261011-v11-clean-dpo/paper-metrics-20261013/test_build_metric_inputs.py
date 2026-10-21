"""Offline contract tests; no models, API calls, or real-data generation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_metric_inputs as build
from fastfill.v1_1.evaluate_v1_1 import score_v1_1


def room(uid="one", source="IL3D_3dfront", room_type="bedroom"):
    return {
        "uid": uid, "source": source, "group": "house", "room_type": room_type,
        "boundary_type": "polygon", "boundary": [[0, 0], [6, 0], [6, 6], [0, 6]],
        "height": 3, "meta": {},
        "fixed": [{"id": "door_1", "category": "door", "size": [.1, .7, 2],
                   "pos": [.05, 3, 0], "yaw": 0}],
        "objects": [
            {"id": "chair_1", "category": "chair", "desc": "a wooden chair",
             "size": [.3, .4, .8], "pos": [1, 1, 0], "yaw": 0, "parent": None},
            {"id": "table_1", "category": "table", "desc": "a round table",
             "size": [.6, .7, .9], "pos": [4, 4, 0], "yaw": 0, "parent": None},
        ],
    }


def output():
    return json.dumps({"placements": [
        {"id": "chair_1", "size": [1, 1, 1], "pos": [2, 2, 0], "yaw": 0},
        {"id": "table_1", "size": [1, 1, 1], "pos": [2.5, 2.5, 0], "yaw": 90},
    ]})


def exported(rooms=None, text=None):
    rooms = rooms or [room()]
    text = output() if text is None else text
    records = [{"evaluation_row": i + 10, "record": r} for i, r in enumerate(rooms)]
    predictions = {
        stage: [{**score_v1_1(r, text), "evaluation_row": i + 10, "output": text}
                for i, r in enumerate(rooms)] for stage in build.STAGES
    }
    return {"data": records, "predictions": predictions, "receipts": {},
            "source_test": {"sha256": "fixture", "rows": len(rooms)},
            "subset_rule": "test fixture"}


class MetricInputTests(unittest.TestCase):
    def test_source_binding_matches_frozen_and_detects_drift(self):
        binding = build.verify_sources()
        self.assertTrue(binding["verified"])
        self.assertIn("fastfill/v1_1/postprocess.py", binding["files"])
        with mock.patch.dict(build.FROZEN_FUNCTION_SHA256, {"score_v1_1": "0" * 64}):
            with self.assertRaisesRegex(build.BuildError, "source.*mismatch"):
                build.verify_sources()

    def test_raw_sizes_and_manual_oor_preserved_input_immutable(self):
        data = exported()
        before = copy.deepcopy(data)
        inputs, summary = build.build_documents(data)
        case = inputs["cases"][0]
        self.assertEqual(data, before)
        self.assertEqual(case["raw_layout"][0]["size"], [1, 1, 1])
        self.assertEqual(case["gt_canonicalobjects"][0]["size"], [.3, .4, .8])
        self.assertAlmostEqual(case["oor"], .125)
        self.assertTrue(case["schema_complete"])
        self.assertEqual(case["asset_mapping"], {})
        self.assertEqual(summary["verification"]["verified_cases"], 3)
        self.assertEqual(summary["all_sources"]["stages"]["sft"]["failed"], 0)
        self.assertEqual(inputs["interpretation"], "diagnostic_only")
        self.assertFalse(inputs["target_independent_claim"])

    def test_judging_requirement_excludes_movable_target_geometry(self):
        original = room()
        altered = {**original, "objects": [
            {**obj, "size": [9, 8, 7], "pos": [5, 4, 3], "yaw": 2,
             "parent": "secret_parent", "anchor": "wall"}
            for obj in original["objects"]]}
        self.assertEqual(build.judging_request(original), build.judging_request(altered))
        request = build.judging_request(original)
        self.assertEqual(set(request), {"room_type", "boundary_type", "boundary", "objects", "fixed"})
        self.assertEqual(set(request["objects"][0]), {"id", "category", "desc"})
        self.assertEqual(set(request["fixed"][0]), {"id", "size", "pos", "yaw"})

    def test_parser_schema_failure_is_not_success_or_repaired(self):
        bad = json.loads(output())
        bad["placements"].append({"id": "extra_1", "size": [2, 2, 2], "pos": [4, 4, 0], "yaw": 0})
        data = exported(text=json.dumps(bad))
        with mock.patch.object(build, "repair_v1_1", side_effect=AssertionError("must not repair")):
            inputs, summary = build.build_documents(data)
        case = inputs["cases"][0]
        self.assertTrue(case["parsed"])
        self.assertFalse(case["schema_complete"])
        self.assertEqual(case["error"], "extra_ids")
        self.assertIsNone(case["repaired_layout"])
        self.assertIsNone(case["oor"])
        self.assertIn("extra_1", [obj["id"] for obj in case["raw_layout"]])
        stats = summary["all_sources"]["stages"]["sft"]
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["schema_failures"], 1)
        self.assertIsNone(stats["mean_oor_fraction"])

    def test_bad_json_and_missing_generation_are_retained(self):
        inputs, _ = build.build_documents(exported(text="broken json"))
        self.assertEqual(inputs["cases"][0]["raw_output"], "broken json")
        self.assertEqual(inputs["cases"][0]["raw_layout"], [])
        data = exported()
        for stage in build.STAGES:
            data["predictions"][stage] = [{
                **score_v1_1(room(), None, "missing_generation"),
                "evaluation_row": 10, "output": None}]
        inputs, summary = build.build_documents(data)
        self.assertEqual(inputs["cases"][0]["error"], "missing_generation")
        self.assertEqual(inputs["cases"][0]["raw_layout"], [])
        self.assertEqual(summary["verification"]["numerical_oor_comparisons"], 0)

    def test_six_strata_are_exact_and_do_not_merge_living_dining(self):
        rooms = [room("a"), room("b", room_type="living dining room"),
                 room("c", "OptiScene_holodeck", "bathroom")]
        _, summary = build.build_documents(exported(rooms))
        self.assertEqual(summary["all_sources"]["scene_count"], 3)
        self.assertEqual(summary["paper_six_strata"]["scene_count"], 2)
        self.assertEqual(summary["paper_six_strata"]["excluded_scene_count"], 1)
        self.assertEqual(len(summary["paper_six_strata"]["strata"]), 6)
        self.assertEqual(summary["by_source"]["IL3D_3dfront"]["scene_count"], 2)

    def test_join_uses_identity_and_rejects_missing_duplicate_or_wrong_source(self):
        data = exported([room("a"), room("b")])
        data["predictions"]["sft"].reverse()
        inputs, _ = build.build_documents(data)
        self.assertEqual(inputs["cases"][0]["uid"], "a")
        variants = []
        missing = copy.deepcopy(data)
        missing["predictions"]["sft"].pop()
        variants.append(missing)
        duplicate = copy.deepcopy(data)
        duplicate["predictions"]["sft"].append(duplicate["predictions"]["sft"][0])
        variants.append(duplicate)
        wrong = copy.deepcopy(data)
        wrong["predictions"]["sft"][0]["source"] = "unknown"
        variants.append(wrong)
        for variant in variants:
            with self.subTest(variant=variants.index(variant)), self.assertRaises(build.BuildError):
                build.build_documents(variant)

    def test_mismatch_never_publishes_outputs(self):
        data = exported()
        data["predictions"]["final"][0]["oor_repaired"] += .1
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.json"
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(build.BuildError, "oor_repaired"):
                build.build_files(source, Path(folder))
            self.assertFalse((Path(folder) / "metric-inputs.json").exists())
            self.assertFalse((Path(folder) / "oor-summary.json").exists())

    def test_end_to_end_manifest_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.json"
            source.write_text(json.dumps(exported()))
            result = build.build_files(source, Path(folder))
            inputs = json.loads((Path(folder) / "metric-inputs.json").read_text())
            summary = (Path(folder) / "oor-summary.json").read_bytes()
            self.assertEqual(inputs["manifest"]["oor_summary_sha256"], hashlib.sha256(summary).hexdigest())
            self.assertTrue(inputs["manifest"]["all_saved_oor_values_match"])
            self.assertEqual(result["cases"], 3)
            with self.assertRaises(FileExistsError):
                build.build_files(source, Path(folder))


if __name__ == "__main__":
    unittest.main()
