import copy
import json
import math
import unittest

from fastfill.comparison import geometry as g
from fastfill.comparison.contract import bind_room, messages


def fixture():
    room = {'uid': 'r', 'source': 'S', 'group': 'g', 'boundary_type': 'polygon',
            'boundary': [[0, 0], [5, 0], [5, 5], [0, 5]], 'height': 3,
            'objects': [{'id': 'old-a', 'category': 'chair', 'size': [1, 1, 1],
                         'pos': [1, 1, 0], 'yaw': 0}], 'fixed': []}
    return bind_room(room)[0]


class GeometryTests(unittest.TestCase):
    @staticmethod
    def paired_rows(spec):
        def row(uid, source, group, value):
            return {'uid': uid, 'source': source, 'group': group,
                    **{key: value for key in g.SCALARS},
                    **{key: value is not None for key in g.RATES}}
        return tuple([row(uid, source, group, values[arm])
                      for uid, source, group, *values in spec] for arm in (0, 1))

    def test_frozen_cross_source_signatures_survive_metric_failure(self):
        j, p = self.paired_rows([
            ('1S', 'S', 'g1', 0., 0.), ('1T', 'T', 'g1', None, 0.),
            ('2S', 'S', 'g2', 1., 0.), ('2T', 'T', 'g2', 1., 0.)])
        result = g.paired_intervals(j, p, repeats=400)['size_iou']
        self.assertEqual(result['singleton_source_signature_strata'], 0)
        self.assertEqual(result['ci95'], [0., 1.])
        self.assertAlmostEqual(result['estimate_joint_minus_placement'], 2/3)
        macro = result['source_macro']
        self.assertEqual(macro['estimate_joint_minus_placement'], .75)
        self.assertGreater(macro['undefined_bootstrap_draws'], 0)
        self.assertIsNone(macro['ci95'])
        self.assertIn('too_few_valid_bootstrap_draws', macro['warnings'])
        self.assertEqual(macro['source_universe'], ['S', 'T'])

    def test_empty_components_remain_in_sampling_and_undefined_draws_counted(self):
        j, p = self.paired_rows([('a', 'S', 'g1', 1., 0.),
                                 ('b', 'S', 'g2', None, 0.)])
        result = g.paired_intervals(j, p, repeats=400)['size_iou']
        self.assertEqual(result['components'], 2)
        self.assertEqual(result['data_qualified_components'], 1)
        self.assertGreater(result['undefined_bootstrap_draws'], 0)
        self.assertEqual(result['valid_bootstrap_draws'] + result['undefined_bootstrap_draws'], 400)
        self.assertIsNone(result['ci95'])
        self.assertIn('too_few_valid_bootstrap_draws', result['warnings'])

    def test_macro_paired_cohort_fixed_sources_and_shared_component_draws(self):
        j, p = self.paired_rows([
            ('1S', 'S', 'g1', 1., 0.), ('1Sa', 'S', 'g1', 1., 0.),
            ('1T', 'T', 'g1', 0., 0.), ('2S', 'S', 'g2', 0., 0.),
            ('2T', 'T', 'g2', 1., 0.), ('3U', 'U', 'g3', None, 1.)])
        before = copy.deepcopy((j, p))
        result = g.paired_intervals(j, p, repeats=400)['size_iou']
        self.assertEqual(result['source_universe'], ['S', 'T', 'U'])
        self.assertEqual(result['data_qualified_sources'], 2)
        self.assertEqual(result['paired_rooms_by_source'], {'S': 3, 'T': 2, 'U': 0})
        macro = result['source_macro']
        self.assertEqual(macro['source_universe'], ['S', 'T'])
        self.assertAlmostEqual(macro['estimate_joint_minus_placement'], 7/12)
        self.assertAlmostEqual(macro['ci95'][0], .5)
        self.assertAlmostEqual(macro['ci95'][1], 7/12)
        self.assertEqual(macro['undefined_bootstrap_draws'], 0)
        self.assertIn('singleton_source_signature_strata', macro['warnings'])
        self.assertEqual((j, p), before)
        self.assertEqual(g.paired_intervals(j, p, repeats=400),
                         g.paired_intervals(j[::-1], p[::-1], repeats=400))

    def test_bootstrap_all_missing_empty_and_small_draw_budget(self):
        j, p = self.paired_rows([('a', 'S', 'g1', None, None)])
        result = g.paired_intervals(j, p, repeats=100)
        missing = result['size_iou']
        self.assertEqual(missing['components'], 1)
        self.assertEqual(missing['undefined_bootstrap_draws'], 100)
        self.assertEqual(missing['data_qualified_sources'], 0)
        self.assertIsNone(missing['source_macro']['estimate_joint_minus_placement'])
        self.assertEqual(result['complete']['estimate_joint_minus_placement'], 0.)
        self.assertIn('singleton_source_signature_strata', result['complete']['warnings'])
        self.assertIsNone(g.paired_intervals([], [], repeats=100)['size_iou']['ci95'])
        few = g.paired_intervals(j, p, repeats=2)['complete']
        self.assertIn('too_few_valid_bootstrap_draws', few['warnings'])
        self.assertIsNone(few['ci95'])
        for repeats in (True, 1, 1.5):
            with self.assertRaises(ValueError):
                g.paired_intervals(j, p, repeats=repeats)
        with self.assertRaises(ValueError):
            g.paired_intervals(j, [{**p[0], 'source': 'changed'}])

    def test_fixed_support_valid_and_invalid_without_changing_overlap(self):
        room = fixture()
        fixed = {'id': 'fixed_0001', 'category': 'counter', 'size': [2, 2, 1],
                 'pos': [2, 2, 0], 'yaw': 0}
        obj = {**room['objects'][0], 'category': 'lamp', 'size': [.2]*3,
               'pos': [2, 2, 1], 'parent': fixed['id']}
        room = {**room, 'fixed': [fixed], 'objects': [obj]}
        before = copy.deepcopy(room)
        msg = messages(room)
        result = g.score(room, msg[:2], msg[-1]['content'], 'joint')
        self.assertTrue(result['geometry_success'])
        self.assertTrue(result['geometry_success_repaired'])
        self.assertEqual(result['support_fail'], 0)
        self.assertEqual(result['oor'], 0.)
        for change in ({'parent': 'missing'}, {'pos': [2, 2, 1.1]},
                       {'pos': [4, 4, 1]}):
            invalid = g.geometry({**room, 'objects': [{**obj, **change}]})
            self.assertEqual(invalid['support_fail'], 1)
            self.assertFalse(invalid['geometry_success'])
        self.assertEqual(room, before)

    def test_repair_preserves_valid_fixed_support(self):
        room = fixture()
        fixed = {'id': 'fixed_0001', 'category': 'counter', 'size': [.5, 2, 1],
                 'pos': [.25, 2, 0], 'yaw': 0}
        obj = {**room['objects'][0], 'category': 'lamp', 'size': [1.2, .2, .2],
               'pos': [.55, 2, 1], 'parent': fixed['id']}
        room = {**room, 'objects': [obj], 'fixed': [fixed]}
        self.assertTrue(g.geometry(room)['geometry_success'])
        repaired = g.small_repair(room)
        self.assertTrue(g.geometry(repaired)['geometry_success'])
        self.assertEqual(repaired['fixed'], room['fixed'])
        self.assertEqual(repaired['objects'][0]['size'], obj['size'])

    def test_repair_rolls_back_whole_candidate_on_cumulative_budget_excess(self):
        for depth in (2, 3):
            with self.subTest(depth=depth):
                room = fixture()
                objects = [{'id': 'a', 'category': 'table', 'size': [1, 1, 1],
                            'pos': [.45, 2, 0], 'yaw': 0},
                           {'id': 'b', 'category': 'box', 'size': [2, .2, .2],
                            'pos': [.85, 2, 1], 'yaw': 0, 'parent': 'a'}]
                if depth == 3:
                    objects.append({'id': 'c', 'category': 'box', 'size': [2, .2, .2],
                                    'pos': [.85, 2, 1.2], 'yaw': 0, 'parent': 'b'})
                room = {**room, 'objects': objects}
                before = copy.deepcopy(room)
                repaired = g.small_repair(room)
                # Parent moves 5 cm, then its child another 10 cm: the entire
                # candidate must roll back, not clamp or detach the descendants.
                self.assertEqual(repaired, before)
                self.assertEqual(room, before)
                self.assertIsNot(repaired, room)
                self.assertFalse(g.geometry(repaired)['geometry_success'])

    def test_repair_at_budget_translates_support_tree_coherently(self):
        room = {**fixture(), 'objects': [
            {'id': 'a', 'category': 'table', 'size': [1, 1, 1],
             'pos': [.4, 2, 0], 'yaw': 0},
            {'id': 'b', 'category': 'box', 'size': [.2, .2, .2],
             'pos': [.4, 2, 1], 'yaw': 0, 'parent': 'a'}]}
        before = copy.deepcopy(room)
        repaired = g.small_repair(room)
        for raw, fixed in zip(room['objects'], repaired['objects']):
            self.assertAlmostEqual(fixed['pos'][0] - raw['pos'][0], .1)
            self.assertEqual(fixed['pos'][1:], raw['pos'][1:])
            self.assertEqual(fixed['size'], raw['size'])
            self.assertEqual(fixed.get('parent'), raw.get('parent'))
        self.assertTrue(g.geometry(repaired)['geometry_success'])
        self.assertEqual(room, before)

    def test_placement_geometry_independent_of_gt_sizes(self):
        room = fixture()
        msg = messages(room, 'placement', {room['objects'][0]['id']: [.5]*3})
        first = g.score(room, msg[:2], msg[-1]['content'], 'placement')
        changed = {**room, 'objects': [{**o, 'size': [2.]*3} for o in room['objects']]}
        second = g.score(changed, msg[:2], msg[-1]['content'], 'placement')
        self.assertEqual(first['raw_layout'], second['raw_layout'])
        self.assertEqual(first['repaired_layout'], second['repaired_layout'])
        self.assertEqual(first['size_iou'], .125)
        self.assertEqual(second['size_iou'], 1/64)

    def test_scalar_requires_common_complete_and_nonfinite_metrics_rejected(self):
        j, p = self.paired_rows([('a', 'S', 'g1', 1., 0.)])
        result = g.paired_intervals([{**j[0], 'complete': False}], p, repeats=100)
        self.assertEqual(result['size_iou']['paired_rooms'], 0)
        self.assertEqual(result['complete']['paired_rooms'], 1)
        self.assertEqual(result['complete']['estimate_joint_minus_placement'], -1.)
        with self.assertRaises(ValueError):
            g.paired_intervals([{**j[0], 'size_iou': float('inf')}], p)

    def test_exact_and_unknown_size_prior(self):
        room = fixture()
        request = messages(room)
        result = g.score(room, request[:2], request[2]['content'], 'joint')
        self.assertEqual(result['size_iou'], 1)
        self.assertEqual(result['size_log_mae'], 0)
        self.assertEqual(result['oor'], 0)
        self.assertTrue(result['geometry_success'])
        oid = room['objects'][0]['id']
        p = messages(room, 'placement', {oid: [0.5, 0.5, 0.5]})
        result = g.score(room, p[:2], p[2]['content'], 'placement')
        self.assertEqual(result['size_iou'], 0.125)
        self.assertAlmostEqual(result['size_log_mae'], math.log(2))
        self.assertEqual(result['raw_layout'][0]['size'], [0.5]*3)

    def test_failure_retained_no_geometric_imputation(self):
        room = fixture()
        for raw in ('{}', 'garbage', '{"placements": []}', None):
            r = g.score(room, messages(room)[:2], raw, 'joint')
            self.assertFalse(r['complete'])
            self.assertFalse(r['geometry_success'])
            self.assertIsNone(r['oor'])
            self.assertIsNone(r['size_iou'])
        r = g.score(room, messages(room)[:2], messages(room)[2]['content'], 'joint', 'runtime')
        self.assertFalse(r['complete'])
        self.assertEqual(r['error'], 'runtime')

    def test_no_inherited_parent_and_no_gt_mutation(self):
        room = fixture()
        child = {**copy.deepcopy(room['objects'][0]), 'id': 'obj_0002',
                 'category': 'lamp', 'size': [0.2, 0.2, 0.2], 'pos': [1, 1, 1],
                 'parent': room['objects'][0]['id']}
        room['objects'].append(child)
        before = copy.deepcopy(room)
        msg = messages(room)
        target = json.loads(msg[-1]['content'])
        target['placements'][1].pop('on')
        target['placements'][1]['pos'] = [3, 3, 0]
        r = g.score(room, msg[:2], json.dumps(target), 'joint')
        self.assertIsNone(r['raw_layout'][1]['parent'])
        self.assertEqual(r['parent_accuracy'], 0.5)
        self.assertEqual(room, before)

    def test_oor_is_overlap_not_out_of_room(self):
        room = fixture()
        room['objects'].append({**room['objects'][0], 'id': 'obj_0002'})
        msg = messages(room)
        r = g.score(room, msg[:2], msg[-1]['content'], 'joint')
        self.assertEqual(r['oor'], 0.5)
        self.assertEqual(r['oob_objects'], 0)
        self.assertFalse(r['geometry_success'])

    def test_predicted_size_frozen_during_small_repair(self):
        room = fixture()
        msg = messages(room)
        target = json.loads(msg[-1]['content'])
        target['placements'][0].update(size=[2, 1, 1], pos=[0.95, 2, 0])
        r = g.score(room, msg[:2], json.dumps(target), 'joint')
        self.assertEqual(r['repaired_layout'][0]['size'], [2, 1, 1])
        self.assertGreater(r['repaired_layout'][0]['pos'][0], 0.95)

    def test_summary_coverage_macro_and_ci(self):
        room = fixture()
        msg = messages(room)
        okay = g.score(room, msg[:2], msg[-1]['content'], 'joint')
        failed = g.score({**room, 'uid': 'r2', 'source': 'T', 'group': 'g2'},
                         msg[:2], '{}', 'joint')
        s = g.summarize([okay, failed])
        self.assertEqual(s['all']['requests'], 2)
        self.assertEqual(s['all']['complete_rate'], 0.5)
        self.assertEqual(s['all']['size_iou'], 1)
        self.assertEqual(s['source_macro']['complete_rate'], 0.5)
        self.assertEqual(s['source_macro']['size_iou_sources'], 1)
        right = [{**okay, 'size_iou': 0.5}, failed]
        ci = g.paired_intervals([okay, failed], right, repeats=100)
        self.assertEqual(ci['size_iou']['estimate_joint_minus_placement'], 0.5)
        self.assertEqual(ci['size_iou']['paired_rooms'], 1)
        self.assertEqual(ci, g.paired_intervals([okay, failed], right, repeats=100))
        with self.assertRaises(ValueError):
            g.paired_intervals([okay], [])
        with self.assertRaises(ValueError):
            g.paired_intervals([okay, okay], [okay, okay])


if __name__ == '__main__':
    unittest.main()
