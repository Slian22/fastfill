"""Synthetic-only checks for the train-only, component-held-out size prior."""
import ast
import copy
import hashlib
import json
import math
import re
import unittest
from pathlib import Path

from fastfill.comparison.size_prior import (
    build_oof_priors, fit_prior, predict_sizes,
)


def obj(oid="chair_1", size=(1, 2, 3), category="chair"):
    return {"id": oid, "category": category, "size": list(size)}


def room(uid, size=(1, 2, 3), category="chair", room_type="bedroom", **kw):
    return {"uid": uid, "room_type": room_type,
            "objects": [obj(size=size, category=category)], **kw}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SizePriorTests(unittest.TestCase):
    def assertSize(self, actual, expected):
        self.assertEqual(len(actual), 3)
        for got, want in zip(actual, expected):
            self.assertTrue(math.isclose(got, want, rel_tol=1e-12))

    def test_even_axis_log_median_and_json_round_trip(self):
        prior = fit_prior([room("a", (1, 4, 9)), room("b", (9, 16, 25))], 2)
        prior = json.loads(json.dumps(prior, allow_nan=False))
        self.assertSize(predict_sizes(room("query"), prior)["chair_1"], (3, 8, 15))
        self.assertEqual(prior["global"]["count"], 2)

    def test_pair_then_category_then_global_fallback(self):
        rooms = [room("a", (2, 3, 4)), room("b", (2, 3, 4)),
                 room("c", (8, 12, 16), room_type="kitchen"),
                 room("d", (8, 12, 16), room_type="kitchen"),
                 room("e", (32, 48, 64), category="sofa")]
        prior = fit_prior(rooms, min_count=2)
        self.assertSize(predict_sizes(room("q"), prior)["chair_1"], (2, 3, 4))
        self.assertSize(predict_sizes(room("q", room_type="office"), prior)["chair_1"],
                        (4, 6, 8))
        self.assertSize(predict_sizes(room("q", category="sofa"), prior)["chair_1"],
                        (8, 12, 16))
        self.assertSize(predict_sizes(room("q", category="unseen"), prior)["chair_1"],
                        (8, 12, 16))

    def test_default_min_count_is_five(self):
        rooms = [room(str(i), (2, 2, 2)) for i in range(4)]
        rooms += [room(str(i), (8, 8, 8), room_type="office") for i in range(4, 8)]
        prior = fit_prior(rooms)
        self.assertEqual(prior["min_count"], 5)
        self.assertSize(predict_sizes(room("q"), prior)["chair_1"], (4, 4, 4))

    def test_normalized_labels_and_unknown_room(self):
        prior = fit_prior([room("a", category="CoffeeTableFactory", room_type="LivingRoom 2")], 1)
        self.assertSize(predict_sizes(room("q", category="coffee table", room_type="living_room"),
                                      prior)["chair_1"], (1, 2, 3))
        prior = fit_prior([room("a", category=None, room_type="unknown")], 1)
        self.assertSize(predict_sizes(room("q", category="", room_type=None), prior)["chair_1"],
                        (1, 2, 3))

    def test_normalizers_match_scene_without_importing_geometry(self):
        # Extract only the stdlib normalization definitions; importing scene needs Shapely.
        source = Path(__file__).resolve().parents[2] / "scene.py"
        names = {"HASH", "ALIASES", "ROOM_WORDS", "UNKNOWN_ROOM", "norm_cat", "norm_room_type"}
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if
                 (isinstance(node, ast.FunctionDef) and node.name in names) or
                 (isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id in names for target in node.targets))]
        canonical = {"re": re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), canonical)
        from fastfill.comparison import size_prior
        for label in (None, "", "Corner/side table", "three-seat / multi-seat sofa",
                      "CoffeeTableFactory", "glassnightstand5c8f77c6", "Chair 12", "chair or stool",
                      "SIDE_TABLE (wood)", "comforterfactory", "boxcomforterfactory"):
            self.assertEqual(size_prior._norm_cat(label), canonical["norm_cat"](label))
        for label in (None, "", "misc", "None", "LivingRoom 2", "restrooms f1", "meeting room 4p",
                      "apartment 2 storage", "living_room / diningroom.", "apt3 kidsroom"):
            self.assertEqual(size_prior._norm_room_type(label), canonical["norm_room_type"](label))

    def test_fixed_excluded_and_queries_ignore_ground_truth(self):
        training = room("a", fixed=[obj("door", (float("nan"), 0, -1))])
        prior = fit_prior([training], 1)
        query = room("q", (float("inf"), 0, -1), fixed=[obj("door", (1e200,) * 3)])
        predicted = predict_sizes(query, prior)
        self.assertEqual(set(predicted), {"chair_1"})
        self.assertSize(predicted["chair_1"], (1, 2, 3))
        query = {**query, "objects": [{"id": "a", "category": "chair"},
                                     {"id": "b", "category": "chair"}]}
        predicted = predict_sizes(query, prior)
        predicted["a"][0] = 99
        self.assertEqual(predicted["b"][0], 1)
        self.assertEqual(prior["global"]["size"][0], 1)
        self.assertEqual(prior["provenance"]["fixed_excluded_count"], 1)

    def test_positive_extreme_finite_sizes(self):
        for value in (1e-300, 1e300):
            prior = fit_prior([room("a", (value,) * 3)], 1)
            self.assertSize(predict_sizes(room("q"), prior)["chair_1"], (value,) * 3)

    def test_invalid_size_triples_fail_closed(self):
        invalid = [None, [], [1, 2], [1, 2, 3, 4], "123", [True, 1, 1],
                   ["1", 2, 3], [0, 1, 1], [-1, 1, 1], [float("nan"), 1, 1],
                   [float("inf"), 1, 1], [1, -float("inf"), 1], [10**1000, 1, 1]]
        for size in invalid:
            with self.subTest(size=str(size)[:50]), self.assertRaises(ValueError):
                fit_prior([room("a", objects=[{"id": "a", "category": "chair", "size": size}])])
        with self.assertRaises(ValueError):
            fit_prior([room("a", objects=[{"id": "a", "category": "chair"}])])

    def test_explicit_nontrain_split_rejected_but_prediction_allowed(self):
        prior = fit_prior([room("a", split="train")])
        for split in ("test", "dev", "validation", "TRAIN", None, ""):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "split"):
                fit_prior([room("q", split=split)])
        self.assertSize(predict_sizes(room("q", split="test"), prior)["chair_1"], (1, 2, 3))

    def test_invalid_min_count_and_empty_fits(self):
        for count in (0, -1, 1.5, True, "5", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                fit_prior([room("a")], count)
        for rooms in ([], [room("a", objects=[])], [room("a", objects=[], fixed=[obj()])]):
            with self.subTest(rooms=rooms), self.assertRaises(ValueError):
                fit_prior(rooms)

    def test_duplicate_or_missing_uid_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate.*uid|duplicate.*UID"):
            fit_prior([room("a"), room("a")])
        for uid in (None, "", " ", 3):
            with self.subTest(uid=uid), self.assertRaises(ValueError):
                fit_prior([room(uid)])

    def test_duplicate_prediction_ids_and_bad_labels_rejected(self):
        prior = fit_prior([room("a")])
        for objects in ([obj(), obj()], [{"category": "chair"}], [{"id": 1}]):
            with self.subTest(objects=objects), self.assertRaises(ValueError):
                predict_sizes({"objects": objects}, prior)
        for kwargs in ({"category": 3}, {"room_type": ["bedroom"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_prior([room("a", **kwargs)])

    def test_inputs_not_mutated_and_generator_supported(self):
        rooms = [room("b"), room("a")]
        snapshot = copy.deepcopy(rooms)
        prior = fit_prior(iter(rooms))
        prior_snapshot = copy.deepcopy(prior)
        predict_sizes(rooms[0], prior)
        self.assertEqual(rooms, snapshot)
        self.assertEqual(prior, prior_snapshot)

    def test_invalid_room_object_and_fixed_containers(self):
        for candidate in (None, [], {"uid": "a"}, room("a", objects={}),
                          room("a", objects=[None]), room("a", fixed="door")):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                fit_prior([candidate])


class OOFPriorTests(unittest.TestCase):
    def setUp(self):
        self.rooms = [room(f"u{i}", (i + 1,) * 3, split="train") for i in range(12)]
        self.components = {f"u{i}": f"private-house-{i // 2}" for i in range(12)}

    def test_components_wholly_held_out_and_five_folds_nonempty(self):
        result = build_oof_priors(self.rooms, self.components)
        self.assertEqual(set(result), {"fold_by_uid", "priors", "full_prior", "provenance"})
        assignments = result["fold_by_uid"]
        self.assertEqual(set(assignments.values()), set(range(5)))
        self.assertEqual(set(result["priors"]), {str(i) for i in range(5)})
        for i in range(0, 12, 2):
            self.assertEqual(assignments[f"u{i}"], assignments[f"u{i + 1}"])
        for fold, prior in result["priors"].items():
            expected = {digest(uid) for uid, assigned in assignments.items() if assigned != int(fold)}
            self.assertEqual(set(prior["provenance"]["fitted_uid_hashes"]), expected)
            held = {digest(self.components[uid]) for uid, assigned in assignments.items()
                    if assigned == int(fold)}
            self.assertFalse(held & set(prior["provenance"]["component_hashes"]))

    def test_extreme_heldout_component_cannot_influence_prior(self):
        baseline = build_oof_priors(self.rooms, self.components, folds=3)
        fold = baseline["fold_by_uid"]["u0"]
        changed = [{**r, "objects": [obj(size=(1e200,) * 3)]}
                   if baseline["fold_by_uid"][r["uid"]] == fold else r for r in self.rooms]
        result = build_oof_priors(changed, self.components, folds=3)
        self.assertEqual(result["priors"][str(fold)], baseline["priors"][str(fold)])
        query = room("q")
        self.assertEqual(predict_sizes(query, result["priors"][str(fold)]),
                         predict_sizes(query, baseline["priors"][str(fold)]))
        self.assertNotEqual(result["full_prior"], baseline["full_prior"])

    def test_deterministic_under_room_object_and_mapping_reordering(self):
        rooms = [{**r, "objects": r["objects"] + [obj("other", (9, 3, 1))]} for r in self.rooms]
        original = copy.deepcopy(rooms)
        result = build_oof_priors(rooms, self.components, 3)
        reversed_rooms = [{**r, "objects": list(reversed(r["objects"]))} for r in reversed(rooms)]
        reordered = build_oof_priors(iter(reversed_rooms), dict(reversed(list(self.components.items()))), 3)
        self.assertEqual(json.dumps(result, allow_nan=False), json.dumps(reordered, allow_nan=False))
        self.assertEqual(rooms, original)

    def test_seeded_hash_assignment_has_known_definition(self):
        result = build_oof_priors(self.rooms, self.components, 3, seed="audit-seed")
        ordered = sorted(set(self.components.values()), key=lambda c: (
            digest(json.dumps(["audit-seed", c], ensure_ascii=False, separators=(",", ":"))), c))
        expected = {component: index % 3 for index, component in enumerate(ordered)}
        self.assertEqual(result["fold_by_uid"],
                         {uid: expected[component] for uid, component in sorted(self.components.items())})

    def test_oof_recomputes_counts_and_fallbacks_from_exact_complement(self):
        assignment = build_oof_priors(self.rooms, self.components, 3)["fold_by_uid"]
        held_uid = next(uid for uid, fold in assignment.items() if fold == 0)
        kept_uid = next(uid for uid, fold in assignment.items() if fold != 0)
        kept_office_uid = next(uid for uid, fold in assignment.items() if fold != 0 and uid != kept_uid)
        additions = {
            held_uid: [obj(f"held-chair-{i}", (100,) * 3) for i in range(3)]
                      + [obj(f"held-sofa-{i}", (100,) * 3, "sofa") for i in range(3)]
                      + [obj("held-lamp", (100,) * 3, "lamp")],
            kept_uid: [obj(f"chair-{i}", (1,) * 3) for i in range(2)]
                      + [obj(f"sofa-{i}", (4,) * 3, "sofa") for i in range(2)],
            kept_office_uid: [obj(f"office-chair-{i}", (9,) * 3) for i in range(3)],
        }
        rooms = [{**r, "room_type": "office" if r["uid"] == kept_office_uid else "bedroom",
                  "objects": [obj("table", (64,) * 3, "table")] + additions.get(r["uid"], [])}
                 for r in self.rooms]
        result = build_oof_priors(rooms, self.components, 3)
        query = room("query", objects=[obj(c, category=c) for c in ("chair", "sofa", "lamp")])
        for fold, prior in result["priors"].items():
            exact = fit_prior([r for r in rooms if assignment[r["uid"]] != int(fold)])
            for key in ("category_room_type", "category", "global"):
                self.assertEqual(prior[key], exact[key])
            self.assertEqual(predict_sizes(query, prior), predict_sizes(query, exact))
        self.assertEqual(result["full_prior"]["category_room_type"]["chair"]["bedroom"]["count"], 5)
        self.assertEqual(result["full_prior"]["category"]["sofa"]["count"], 5)
        oof = result["priors"]["0"]
        self.assertEqual(oof["category_room_type"]["chair"]["bedroom"]["count"], 2)
        self.assertEqual(oof["category"]["sofa"]["count"], 2)
        self.assertNotIn("lamp", oof["category"])
        predictions = predict_sizes(query, oof)
        for category, expected in (("chair", 9), ("sofa", 64), ("lamp", 64)):
            for axis in predictions[category]:
                self.assertTrue(math.isclose(axis, expected, rel_tol=1e-12))

    def test_provenance_counts_hashes_and_no_raw_component_identifiers(self):
        result = build_oof_priors(self.rooms, self.components, 3)
        provenance = result["provenance"]
        self.assertEqual(provenance["seed"], "unknown-size-v1-20261013")
        self.assertEqual(provenance["fold_count"], 3)
        self.assertEqual(provenance["component_count"], 6)
        self.assertIn("sha256", provenance["fold_algorithm"])
        for prior in [result["full_prior"], *result["priors"].values()]:
            p = prior["provenance"]
            self.assertEqual(p["room_count"], len(p["fitted_uid_hashes"]))
            self.assertEqual(p["object_count"], prior["global"]["count"])
            self.assertEqual(p["component_count"], len(p["component_hashes"]))
            self.assertEqual(p["fitted_uid_sha256"], digest(json.dumps(
                sorted(p["fitted_uid_hashes"]), ensure_ascii=False, separators=(",", ":"))))
            serialized = json.dumps(prior, allow_nan=False)
            self.assertNotIn("private-house-", serialized)
            for uid in self.components:
                self.assertNotIn(json.dumps(uid), serialized)
        self.assertEqual(result["full_prior"]["provenance"]["room_count"], 12)
        self.assertEqual(fit_prior(self.rooms)["provenance"]["component_count"], None)
        expected = {digest(c): {"fold": result["fold_by_uid"][uid],
                               "uid_hashes": sorted(digest(u) for u, other in self.components.items()
                                                    if other == c)}
                    for uid, c in self.components.items()}
        self.assertEqual(provenance["components"], expected)

    def test_external_component_entries_ignored_and_manifest_round_trip(self):
        expected = build_oof_priors(self.rooms, self.components, 3)
        actual = build_oof_priors(self.rooms, {**self.components, "not-in-training": None}, 3)
        self.assertEqual(json.loads(json.dumps(actual, allow_nan=False)), expected)
        standalone = fit_prior(self.rooms)
        for key in ("category_room_type", "category", "global", "min_count"):
            self.assertEqual(actual["full_prior"][key], standalone[key])
        with self.assertRaises(ValueError):
            build_oof_priors(self.rooms, [], 3)

    def test_invalid_folds_and_seed(self):
        for folds in (0, 1, 1.5, True, "3", 7):
            with self.subTest(folds=folds), self.assertRaises(ValueError):
                build_oof_priors(self.rooms, self.components, folds=folds)
        with self.assertRaises(ValueError):
            build_oof_priors(self.rooms, self.components, seed=None)

    def test_missing_empty_and_insufficient_components(self):
        invalid = [{}, {uid: "one" for uid in self.components},
                   {**self.components, "u0": ""}, {**self.components, "u0": " "},
                   {**self.components, "u0": None}, {**self.components, "u0": []}]
        for mapping in invalid:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                build_oof_priors(self.rooms, mapping, folds=3)

    def test_oof_rejects_duplicate_uid_and_explicit_nontrain_split(self):
        for rooms in (self.rooms + [self.rooms[0]],
                      [{**self.rooms[0], "split": "test"}, *self.rooms[1:]]):
            with self.subTest(rooms=rooms), self.assertRaises(ValueError):
                build_oof_priors(rooms, self.components, 3)

    def test_empty_needed_training_subset_fails_even_if_full_fit_succeeds(self):
        rooms = [room("a"), room("b", objects=[], fixed=[obj("fixed")])]
        self.assertEqual(fit_prior(rooms)["global"]["count"], 1)
        with self.assertRaisesRegex(ValueError, "empty|no.*object"):
            build_oof_priors(rooms, {"a": "A", "b": "B"}, folds=2)


if __name__ == "__main__":
    unittest.main()
