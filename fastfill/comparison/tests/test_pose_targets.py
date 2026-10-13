"""Synthetic offline-teacher fixtures; no release data, fitting, or training."""
import copy
import json
import math
import unittest
from unittest.mock import patch

from fastfill.comparison.pose_targets import repair
from fastfill.validate import check, support_contains


def item(oid, size=(1, 1, 1), pos=(2, 2, 0), *, parent=None,
         yaw=0, category="cabinet"):
    return {"id": oid, "category": category, "size": list(size),
            "pos": list(pos), "yaw": yaw, "parent": parent,
            "anchor": "object" if parent else "floor", "desc": "keep me"}


def room_of(*objects, width=5, depth=5, height=4, fixed=()):
    return {"uid": "synthetic:teacher", "source": "fixture", "group": "house",
            "room_type": "office", "boundary_type": "polygon",
            "boundary": [[0, 0], [width, 0], [width, depth], [0, depth]],
            "height": height, "objects": list(objects), "fixed": list(fixed),
            "meta": {"nested": ["preserve"]}}


def sizes_of(room):
    return {o["id"]: list(o["size"]) for o in room["objects"]}


class PoseTargetTests(unittest.TestCase):
    def assert_inventory(self, source, sizes, result):
        output = result["room"]
        self.assertEqual({k: v for k, v in output.items() if k != "objects"},
                         {k: v for k, v in source.items() if k != "objects"})
        self.assertEqual(len(source["objects"]), len(output["objects"]))
        for old, new in zip(source["objects"], output["objects"]):
            self.assertEqual({k: v for k, v in old.items() if k not in ("pos", "yaw", "size")},
                             {k: v for k, v in new.items() if k not in ("pos", "yaw", "size")})
            self.assertEqual(json.dumps(new["size"]), json.dumps(sizes[old["id"]]))
            self.assertEqual(new["pos"], [round(v, 2) for v in new["pos"]])
            deg = math.degrees(new["yaw"])
            self.assertAlmostEqual(deg, round(deg))
            self.assertTrue(0 <= deg < 360)
        json.dumps(result, allow_nan=False)

    def assert_accepted(self, room, sizes, result):
        self.assertEqual(result["status"], "ok", result["diagnostics"])
        self.assert_inventory(room, sizes, result)
        audit = check(result["room"])
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["collisions"]["main"], [])
        self.assertEqual(audit["collisions"]["severe"], [])
        self.assertEqual(audit["fixed_blocking"], [])
        self.assertEqual(result["diagnostics"]["reasons"], [])
        self.assertEqual(result["diagnostics"]["after_counts"]["severe"], 0)

    def test_frozen_parent_height_recomputes_entire_support_chain(self):
        room = room_of(item("child", (.4, .4, .3), (2, 2, 1), parent="parent"),
                       item("top", (.2, .2, .2), (2, 2, 1.3), parent="child"),
                       item("parent", (1, 1, 1)))
        sizes = {**sizes_of(room), "parent": [.6, .6, .6]}
        result = repair(room, sizes)
        self.assert_accepted(room, sizes, result)
        by_id = {o["id"]: o for o in result["room"]["objects"]}
        self.assertEqual(by_id["child"]["pos"][2], .6)
        self.assertEqual(by_id["top"]["pos"][2], .9)
        self.assertIn("child", result["diagnostics"]["before"]["support_fail"])

    def test_child_center_is_projected_onto_rotated_shrunk_parent(self):
        room = room_of(item("p", (2, 2, 1), yaw=math.pi / 4),
                       item("c", (.4, .4, .3), (3, 2, 1), parent="p"))
        sizes = {**sizes_of(room), "p": [.5, .3, .6]}
        result = repair(room, sizes)
        self.assert_accepted(room, sizes, result)
        p, c = result["room"]["objects"]
        self.assertTrue(support_contains(p, c["pos"]))
        self.assertEqual(result["diagnostics"]["policy"]["support"], "center")

    def test_declared_support_rule_allows_child_overhang(self):
        room = room_of(item("p", (.4, .4, .6)),
                       item("c", (1, 1, .3), (2, 2, .6), parent="p"))
        self.assert_accepted(room, sizes_of(room), repair(room, sizes_of(room)))

    def test_boundary_translation_with_frozen_dimensions(self):
        room = room_of(item("a", (.2, .2, .4), (.1, 2, .2)))
        sizes = {"a": [1.2, .8, .7]}
        result = repair(room, sizes)
        self.assert_accepted(room, sizes, result)
        self.assertGreater(result["room"]["objects"][0]["pos"][0], .1)
        self.assertEqual(result["room"]["objects"][0]["pos"][2], 0)

    def test_cardinal_yaw_can_fit_long_object(self):
        room = room_of(item("a", pos=(.5, 1.5, 0)), width=1, depth=3)
        sizes = {"a": [2.4, .7, 1]}
        result = repair(room, sizes)
        self.assert_accepted(room, sizes, result)
        self.assertIn(round(math.degrees(result["room"]["objects"][0]["yaw"])), (90, 270))

    def test_oversized_object_is_rejected_without_shrinking_or_dropping(self):
        room = room_of(item("a"), width=2, depth=2)
        sizes = {"a": [4, 4, 1]}
        result = repair(room, sizes)
        self.assertEqual(result["status"], "rejected")
        self.assert_inventory(room, sizes, result)
        self.assertIn("oob", result["diagnostics"]["reasons"])
        self.assertIn("search_exhausted", result["diagnostics"]["reasons"])
        self.assertLessEqual(result["diagnostics"]["candidate_counts"]["a"], 200)
        self.assertLessEqual(result["diagnostics"]["passes"], 4)

    def test_fixed_column_is_avoided_and_preserved(self):
        room = room_of(item("a"), fixed=[item("column", (.6, .6, 3), category="column")])
        result = repair(room, sizes_of(room))
        self.assert_accepted(room, sizes_of(room), result)
        self.assertEqual(len(result["diagnostics"]["before"]["fixed_blocking"]), 1)

    def test_windows_remain_nonblocking_per_existing_checker(self):
        room = room_of(item("a"), fixed=[item("w", (.1, 1, 2), category="window")])
        result = repair(room, sizes_of(room))
        self.assert_accepted(room, sizes_of(room), result)
        self.assertEqual(len(result["diagnostics"]["after"]["fixed_collisions"]), 1)

    def test_tuck_category_semantics_are_retained(self):
        room = room_of(item("a", category="table"), item("b", category="chair"))
        result = repair(room, sizes_of(room))
        self.assert_accepted(room, sizes_of(room), result)
        self.assertEqual(len(result["diagnostics"]["after"]["collisions"]["tuck"]), 1)

    def test_collision_in_supported_subtree_moves_whole_tree(self):
        # The root is clear of the column in Z; only its supported child collides.
        room = room_of(item("p", (.8, .8, .6)),
                       item("c", (1.2, 1.2, .5), (2, 2, .6), parent="p"),
                       fixed=[item("column", (.4, .4, 1), (2, 2, .65), category="column")])
        result = repair(room, sizes_of(room))
        self.assert_accepted(room, sizes_of(room), result)
        p, c = result["room"]["objects"]
        self.assertEqual(p["pos"][:2], c["pos"][:2])
        self.assertNotEqual(p["pos"][:2], [2, 2])

    def test_two_supported_subtrees_avoid_nonancestor_collisions(self):
        room = room_of(item("p", (.6, .6, .6), (2, 2, 0)),
                       item("c", (1.2, 1.2, .5), (2, 2, .6), parent="p"),
                       item("q", (.6, .6, .6), (3, 2, 0)),
                       item("d", (1.2, 1.2, .5), (3, 2, .6), parent="q"))
        self.assertEqual(check(room)["collisions"]["main"], [("c", "d")])
        self.assert_accepted(room, sizes_of(room), repair(room, sizes_of(room)))

    def test_siblings_can_move_independently_on_parent(self):
        room = room_of(item("p", (3, 3, .6)),
                       item("c", (.5, .5, .4), (2, 2, .6), parent="p"),
                       item("d", (.5, .5, .4), (2, 2, .6), parent="p"))
        self.assert_accepted(room, sizes_of(room), repair(room, sizes_of(room)))

    def test_child_footprint_participates_in_boundary_search(self):
        room = room_of(item("p", (.2, .2, .6), (.2, 2, 0)),
                       item("c", (1.8, 1, .5), (.2, 2, .6), parent="p"))
        self.assert_accepted(room, sizes_of(room), repair(room, sizes_of(room)))

    def test_determinism_immutability_and_no_output_aliases(self):
        room = room_of(item("a"), item("b"))
        sizes = sizes_of(room)
        old_room, old_sizes = copy.deepcopy(room), copy.deepcopy(sizes)
        first, second = repair(room, sizes), repair(room, sizes)
        self.assert_accepted(room, sizes, first)
        self.assertEqual(first, second)
        self.assertEqual(room, old_room)
        self.assertEqual(sizes, old_sizes)
        first["room"]["objects"][0]["size"][0] = 999
        first["room"]["meta"]["nested"].append("changed")
        self.assertEqual(room, old_room)
        self.assertEqual(sizes, old_sizes)

    def test_search_is_independent_of_gt_sizes_and_input_order(self):
        room = room_of(item("a"), item("b"))
        sizes = {"a": [1.3, .8, .9], "b": [1.0, 1.0, .9]}
        altered = {**room, "objects": [{**o, "size": [9, 7, 8]}
                                      for o in reversed(room["objects"])]}
        first, second = repair(room, sizes), repair(altered, sizes)
        self.assert_accepted(room, sizes, first)
        self.assertEqual({o["id"]: o for o in first["room"]["objects"]},
                         {o["id"]: o for o in second["room"]["objects"]})

    def test_rounding_is_applied_before_acceptance(self):
        room = room_of(item("p", (.5, .5, .6167), (1.12345, 2.45678, .009), yaw=.12345),
                       item("c", (.2, .2, .2), (1.12345, 2.45678, .6167), parent="p"))
        result = repair(room, sizes_of(room))
        self.assert_accepted(room, sizes_of(room), result)
        self.assertEqual(result["room"]["objects"][1]["pos"][2], .62)

    def test_no_unrounded_boundary_false_acceptance(self):
        # Only the unrounded X=0.003 center fits; neither neighboring cm grid point fits.
        room = room_of(item("a", (1.203, 2.5, .2), (.003, 1.5, 0)), width=1, depth=3)
        room["boundary"] = [[-.499, 0], [.505, 0], [.505, 3], [-.499, 3]]
        self.assertTrue(check(room)["valid"])
        result = repair(room, sizes_of(room))
        self.assertEqual(result["status"], "rejected")
        self.assertIn("oob", result["diagnostics"]["reasons"])

    def test_explicit_tiny_budget_is_honored(self):
        room = room_of(item("a"), item("b"), width=1, depth=1)
        result = repair(room, sizes_of(room), max_candidates=3, max_passes=2)
        self.assertEqual(result["status"], "rejected")
        self.assertLessEqual(result["diagnostics"]["passes"], 2)
        self.assertTrue(all(n <= 3 for n in result["diagnostics"]["candidate_counts"].values()))

    def test_candidate_budget_includes_all_descendant_trials(self):
        room = room_of(item("p", (.3, .3, .6)),
                       item("c", (1, 1, .4), (2, 2, .6), parent="p"),
                       item("d", (1, 1, .4), (2, 2, .6), parent="p"))
        with patch("fastfill.comparison.pose_targets.check", wraps=check) as checker:
            result = repair(room, sizes_of(room), max_candidates=7, max_passes=3)
        self.assertEqual(result["status"], "rejected")
        count = result["diagnostics"]["candidate_counts"]["p"]
        self.assertLessEqual(count, 7)
        # Original audit, normalized audit, and returned-precision recheck are separate.
        self.assertEqual(checker.call_count, count + 3)

    def test_concave_boundary_uses_real_footprints(self):
        room = room_of(item("a", (.7, .7, .6), (3, 3, 0)), width=4, depth=4)
        room["boundary"] = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
        self.assertFalse(check(room)["valid"])
        self.assert_accepted(room, sizes_of(room), repair(room, sizes_of(room)))

    def test_zero_budget_still_normalizes_and_checks(self):
        room = room_of(item("a", pos=(2, 2, .2)))
        result = repair(room, sizes_of(room), max_candidates=0)
        self.assert_accepted(room, sizes_of(room), result)
        self.assertEqual(result["diagnostics"]["candidate_counts"], {"a": 0})

    def test_ceiling_failure_is_explicit_and_inventory_is_complete(self):
        room = room_of(item("p", (1, 1, 1)),
                       item("c", (.4, .4, .5), (2, 2, 1), parent="p"), height=1)
        result = repair(room, sizes_of(room), max_candidates=4)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("ceiling", result["diagnostics"]["reasons"])
        self.assert_inventory(room, sizes_of(room), result)

    def test_bad_support_graphs_are_explicit_rejections(self):
        for objects in [(item("a", parent="absent"),),
                        (item("a", parent="b"), item("b", parent="a")),
                        (item("a", parent="a"),)]:
            room = room_of(*objects)
            result = repair(room, sizes_of(room))
            self.assertEqual(result["status"], "rejected")
            self.assertTrue(result["diagnostics"]["reasons"])
            self.assert_inventory(room, sizes_of(room), result)

    def test_missing_or_invalid_estimates_never_fall_back_to_gt_sizes(self):
        room = room_of(item("a"))
        for sizes in ({}, {"a": [0, 1, 1]}, {"a": [1, 1]},
                      {"a": [True, 1, 1]}, {"a": [1, 1, 1], "extra": [1, 1, 1]},
                      {"a": [10**1000, 1, 1]}, {"a": [float("nan"), 1, 1]},
                      {"a": [float("inf"), 1, 1]}):
            result = repair(room, sizes)
            self.assertEqual(result["status"], "rejected")
            self.assertTrue(result["diagnostics"]["reasons"])
            self.assertEqual(len(result["room"]["objects"]), 1)

    def test_invalid_search_limits_fail_explicitly(self):
        room = room_of(item("a"))
        for kwargs in ({"max_candidates": -1}, {"max_passes": -1},
                       {"max_candidates": 1.5}, {"max_passes": True}):
            self.assertEqual(repair(room, sizes_of(room), **kwargs)["status"], "rejected")

    def test_nonfinite_rejections_are_strict_json_and_keep_inventory(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for field in ("yaw", "pos", "size", "height", "boundary", "fixed"):
                with self.subTest(value=value, field=field):
                    room = room_of(item("a"), fixed=[item("column", category="column")])
                    sizes = sizes_of(room)
                    if field == "yaw":
                        room["objects"][0]["yaw"] = value
                    elif field in ("pos", "size"):
                        room["objects"][0][field][0] = value
                        if field == "size":
                            sizes = sizes_of(room)
                    elif field == "height":
                        room["height"] = value
                    elif field == "boundary":
                        room["boundary"][0][0] = value
                    else:
                        room["fixed"][0]["size"][0] = value
                    room["meta"]["invalid"] = {"nested": [value]}
                    original = json.dumps(room)
                    result = repair(room, sizes)
                    self.assertEqual(result["status"], "rejected")
                    receipt = json.loads(json.dumps(result, allow_nan=False))
                    self.assertTrue(receipt["diagnostics"]["reasons"])
                    self.assertEqual([o["id"] for o in receipt["room"]["objects"]], ["a"])
                    self.assertEqual(receipt["room"]["meta"]["invalid"]["nested"],
                                     [{"nonfinite": str(value)}])
                    self.assertEqual(json.dumps(room), original)
                    result["room"]["meta"]["nested"].append("detached")
                    self.assertEqual(json.dumps(room), original)

    def test_nonfinite_limits_serialize_and_skip_graph_preflight(self):
        room = room_of(item("a"))
        for key in ("max_candidates", "max_passes"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, value=value):
                    with patch("fastfill.comparison.pose_targets._forest") as forest:
                        result = repair(room, sizes_of(room), **{key: value})
                    self.assertEqual(result["status"], "rejected")
                    forest.assert_not_called()
                    receipt = json.loads(json.dumps(result, allow_nan=False))
                    self.assertEqual(receipt["diagnostics"]["reasons"], ["invalid_search_limits"])
                    self.assertEqual(receipt["diagnostics"]["limits"][key],
                                     {"nonfinite": str(value)})

    def test_physical_rejection_keeps_finite_values_and_serializes_bad_metadata(self):
        room = room_of(item("a"), width=1, depth=1)
        room["meta"] = {"measurement": float("nan"), "finite_tuple": (1.23456789, "keep")}
        sizes = {"a": [4.123456789, 4, 1]}
        result = repair(room, sizes, max_candidates=0)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("oob", result["diagnostics"]["reasons"])
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["room"]["objects"][0]["size"], sizes["a"])
        self.assertEqual(result["room"]["meta"]["finite_tuple"], room["meta"]["finite_tuple"])
        self.assertTrue(math.isnan(room["meta"]["measurement"]))

    def test_oversized_graph_is_retained_without_geometry_or_graph_work(self):
        room = room_of(*[item(str(i), parent=str(i - 1) if i else None)
                         for i in range(257)])
        sizes = {o["id"]: [.8, .7, .6] for o in room["objects"]}
        with patch("fastfill.comparison.pose_targets._forest") as forest, \
                patch("fastfill.comparison.pose_targets.check") as checker:
            result = repair(room, sizes, max_candidates=0)
        forest.assert_not_called()
        checker.assert_not_called()
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["diagnostics"]["reasons"], ["movable_limit_exceeded"])
        self.assertEqual(result["diagnostics"]["limits"]["max_objects"], 256)
        self.assert_inventory(room, sizes, result)

    def test_fifty_object_support_chain_remains_accepted(self):
        room = room_of(*[item(str(i), (.2, .2, .02), (2, 2, i * .02),
                              parent=str(i - 1) if i else None) for i in reversed(range(50))])
        sizes = sizes_of(room)
        result = repair(room, sizes, max_candidates=0)
        self.assert_accepted(room, sizes, result)
        self.assertEqual(result["diagnostics"]["candidate_counts"], {"0": 0})

    def test_empty_inventory_and_absent_boundary(self):
        empty = room_of()
        self.assert_accepted(empty, {}, repair(empty, {}))
        room = {**room_of(item("a"), item("b")), "boundary": None, "boundary_type": None}
        self.assert_accepted(room, sizes_of(room), repair(room, sizes_of(room)))

    def test_final_checker_result_is_authoritative(self):
        room = room_of(item("a"))
        real_check = check

        def reject(candidate):
            audit = real_check(candidate)
            return {**audit, "collisions": {**audit["collisions"], "severe": [("a", "ghost")]}}

        with patch("fastfill.comparison.pose_targets.check", side_effect=reject):
            result = repair(room, sizes_of(room), max_candidates=0)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("severe", result["diagnostics"]["reasons"])

    def test_bound_training_prompt_is_unchanged_and_teacher_label_changes(self):
        from fastfill.comparison.contract import bind_room, messages

        room, _ = bind_room(room_of(item("p"),
                                   item("c", (.3, .3, .4), (2, 2, 1), parent="p")))
        sizes = {o["id"]: [.6, .6, .6] for o in room["objects"]}
        before = messages(room, arm="placement", estimated_sizes=sizes)
        result = repair(room, sizes)
        self.assert_accepted(room, sizes, result)
        after = messages(result["room"], arm="placement", estimated_sizes=sizes)
        self.assertEqual(before[:2], after[:2])
        self.assertNotEqual(before[2], after[2])

    def test_invalid_ir_is_rejected_without_input_mutation(self):
        room = room_of(item("a"))
        variants = [None, {**room, "objects": None},
                    {**room, "objects": [item("a"), item("a")]},
                    {**room, "objects": [{**item("a"), "id": None}]},
                    {**room, "objects": [{**item("a"), "yaw": float("inf")}]},
                    {**room, "boundary": [[0, 0]]},
                    {**room, "boundary": [[0, 0], [1, 1], [0, 1], [1, 0]]},
                    {**room, "boundary_type": "invalid"}, {**room, "height": -1}]
        for variant in variants:
            original = copy.deepcopy(variant)
            result = repair(variant, {"a": [1, 1, 1]})
            self.assertEqual(result["status"], "rejected")
            self.assertTrue(result["diagnostics"]["reasons"])
            self.assertEqual(variant, original)


if __name__ == "__main__":
    unittest.main()
