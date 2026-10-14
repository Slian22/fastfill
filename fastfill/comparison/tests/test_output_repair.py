import copy
import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastfill.comparison import downstream as d
from fastfill.comparison import geometry as g
from fastfill.comparison import output_repair as r
from fastfill.comparison.output_repair import COMPONENTS
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.validate import check, holds, support_ok

RECT = [[0, 0], [5, 0], [5, 4], [0, 4]]
L_SHAPE = [[0, 0], [6, 0], [6, 3], [3, 3], [3, 6], [0, 6]]


def obj(i, category, size, pos, yaw=0.0, parent=None):
    return {'id': i, 'category': category, 'size': list(size), 'pos': list(pos), 'yaw': yaw, 'parent': parent}


def door(x, y, yaw=math.pi / 2, i='door_1'):
    return {'id': i, 'category': 'door', 'size': [0.1, 0.9, 2.0], 'pos': [x, y, 0], 'yaw': yaw}


def room(objects, fixed=(), boundary=RECT, kind='polygon'):
    return {'uid': 'u', 'boundary_type': kind, 'boundary': boundary, 'height': 2.8, 'fixed': list(fixed),
            'objects': objects}


class RepairTests(unittest.TestCase):
    def assert_invariants(self, raw, result):
        out = result['room']['objects']
        keys = ('id', 'category', 'size', 'yaw', 'parent')
        self.assertEqual([[o.get(k) for k in keys] + [o['pos'][2]] for o in out],
                         [[o.get(k) for k in keys] + [o['pos'][2]] for o in raw['objects']])
        disp = result['diagnostics']['displacement']
        self.assertEqual(len(disp), len(raw['objects']))                       # nobody is left out
        for a, b, moved in zip(raw['objects'], out, disp):
            self.assertEqual(moved, math.dist(a['pos'][:2], b['pos'][:2]))
            if moved == 0:
                self.assertEqual(json.dumps(a), json.dumps(b))                 # untouched: bit for bit
        keepouts = result['diagnostics'].get('keepouts', [])
        held = [['against_wall', o['id']] for o in raw['objects'] if holds(['against_wall', o['id']], raw['objects'],
                                                                           raw['boundary'])]
        before, after = r.audit(raw, keepouts, held), r.audit(result['room'], keepouts, held)
        self.assertTrue(all(b <= a for a, b in zip(before['v'], after['v'])), (before['v'], after['v']))
        return before, after

    def test_door_keepout_covering_the_whole_floor_skips_the_strict_stage(self):
        # structured3d_00488_room_4256 (dev): a 2.5 x 1 m room whose 2.38 m door strip covers the floor.
        wide = {'id': 'door_1', 'category': 'door', 'size': [2.38, 0.1, 2.57], 'pos': [1.24, 1.12, 0], 'yaw': 0.0}
        raw = room([obj('unknown_1', 'cabinet', (.64, .64, .5), (.9, .49, 0))], [wide],
                   boundary=[[0, 0], [2.5, 0], [2.5, 1], [0, 1]])
        result = r.repair(copy.deepcopy(raw))
        self.assertEqual(result['diagnostics']['strict_skipped'], 'empty_floor')
        self.assertFalse(result['diagnostics']['downstream']['raw']['accepted'])
        self.assert_invariants(raw, result)

    def test_colocated_duplicate_wardrobes_separated_deterministically(self):
        raw = room([obj('wardrobe_a', 'wardrobe', (.6, 1.2, 2), (4.6, 2.0, 0), math.pi),
                    obj('wardrobe_b', 'wardrobe', (.6, 1.2, 2), (4.6, 2.0, 0), math.pi),
                    obj('bed', 'bed', (2, 1.6, .5), (1.2, 2.0, 0))], [door(2.5, 0.0)])
        frozen = copy.deepcopy(raw)
        first, second = r.repair(raw), r.repair(copy.deepcopy(raw))
        self.assertEqual(raw, frozen)
        self.assertEqual(json.dumps(first['room']), json.dumps(second['room']))
        self.assertEqual(first['diagnostics']['steps'], second['diagnostics']['steps'])
        self.assertEqual(first['status'], 'ok')
        self.assertFalse(check(first['room'])['collisions']['main'])
        stay = min(('wardrobe_a', 'wardrobe_b'), key=lambda i: hashlib.sha256(i.encode()).hexdigest())
        out = {o['id']: o for o in first['room']['objects']}
        self.assertEqual(out[stay]['pos'], [4.6, 2.0, 0])                    # the first copy stays
        self.assert_invariants(raw, first)
        self.assertTrue(first['diagnostics']['audit_parity'])
        self.assertEqual(first['diagnostics']['keepout_source'], 'doors')

    def test_table_through_wall_comes_back_with_its_cup(self):
        raw = room([obj('table', 'table', (1.2, .8, .75), (1.0, 3.63, 0)),
                    obj('cup', 'cup', (.1, .1, .1), (1.1, 3.7, .75), parent='table')])
        result = r.repair(raw)
        self.assertEqual(r.strict_audit(raw, [])['outside_1mm'], ['table'])
        self.assertEqual(r.strict_audit(result['room'], [])['outside_1mm'], [])
        table, cup = result['room']['objects']
        self.assertTrue(support_ok(cup, table))
        self.assertAlmostEqual(cup['pos'][0] - table['pos'][0], 0.1)
        self.assertAlmostEqual(cup['pos'][1] - table['pos'][1], 0.07)
        self.assertLessEqual(table['pos'][1] + .4, 4.0 + 1e-3)
        self.assert_invariants(raw, result)

    def test_keepout_intrusion_slides_clear_along_its_wall(self):
        raw = room([obj('cabinet', 'cabinet', (.5, 1.0, 1.0), (1.7, .25, 0), math.pi / 2)], [door(2.5, 0.0)])
        self.assertTrue(holds(['against_wall', 'cabinet'], raw['objects'], RECT))
        self.assertEqual(r.strict_audit(raw, r.door_keepouts(raw))['keepout_1cm'], [['cabinet', 0]])
        result = r.repair(raw)
        out = result['room']
        self.assertEqual(r.strict_audit(out, r.door_keepouts(out))['keepout_1cm'], [])
        self.assertTrue(holds(['against_wall', 'cabinet'], out['objects'], RECT))
        self.assertEqual(out['objects'][0]['pos'][1], .25)
        self.assertEqual(result['status'], 'ok')
        self.assert_invariants(raw, result)

    def test_z_sunk_child_is_flagged_and_not_moved(self):
        raw = room([obj('desk', 'desk', (1.2, .6, .75), (2.5, 2, 0)),
                    obj('lamp', 'lamp', (.2, .2, .4), (2.5, 2, .5), parent='desk')])
        result = r.repair(raw)
        self.assertEqual(result['status'], 'unchanged')
        self.assertEqual(result['diagnostics']['flags'], {'lamp': ['support_z']})
        self.assertEqual(result['diagnostics']['unsolved'], {'lamp': ['support_fail', 'support_z']})
        self.assertEqual(json.dumps(result['room']), json.dumps(raw))

    def test_valid_layout_stays_byte_identical(self):
        raw = room([obj('bed', 'bed', (2, 1.6, .5), (1.05, 2.5, 0.0)),
                    obj('nightstand', 'nightstand', (.4, .4, .6), (.3, 1.4, 0)),
                    obj('lamp', 'lamp', (.2, .2, .4), (.3, 1.4, .6), parent='nightstand')], [door(4.0, 0.0)])
        result = r.repair(raw)
        self.assertEqual(result['status'], 'unchanged')
        self.assertEqual(json.dumps(result['room']), json.dumps(raw))
        self.assertEqual(result['diagnostics']['displacement'], [0.0, 0.0, 0.0])

    def test_unsolvable_dense_room_keeps_every_object_and_never_gets_worse(self):
        side = [[0, 0], [2.5, 0], [2.5, 2.5], [0, 2.5]]                      # room for four of the eight boxes
        raw = room([obj(f'box_{k}', 'box', (1, 1, 1), (1.25, 1.25, 0)) for k in range(8)], boundary=side)
        result = r.repair(raw, max_trials=300)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(sorted(result['diagnostics']['unsolved']), [f'box_{k}' for k in range(8)])
        self.assertTrue(result['diagnostics']['budget_exhausted'])
        before, after = self.assert_invariants(raw, result)
        self.assertLess(after['v'][COMPONENTS.index('main')], before['v'][COMPONENTS.index('main')])
        self.assertLess(after['phi'], before['phi'])

    def test_l_shaped_room_moves_support_subtrees_rigidly(self):
        raw = room([obj('sofa', 'sofa', (2, .9, .8), (1.5, 4.0, 0)),
                    obj('table', 'table', (1.2, .8, .75), (1.6, 4.1, 0)),
                    obj('tray', 'tray', (.4, .3, .05), (1.8, 4.2, .75), parent='table'),
                    obj('cup', 'cup', (.1, .1, .1), (1.8, 4.2, .8), parent='tray')], boundary=L_SHAPE)
        result = r.repair(raw)
        out = {o['id']: o for o in result['room']['objects']}
        self.assertFalse(check(result['room'])['collisions']['main'])
        offsets = lambda objs: [round(objs[k]['pos'][a] - objs['table']['pos'][a], 9) for k in ('tray', 'cup') for a in (0, 1)]
        self.assertEqual(offsets(out), offsets({o['id']: o for o in raw['objects']}))
        self.assertTrue(g.geometry(result['room'])['geometry_valid'])
        self.assert_invariants(raw, result)

    def test_floor_covering_in_a_doorway_is_moved_clear_and_accepted_as_is(self):
        raw = room([obj('rug', 'rug', (1.6, 1.2, .01), (2.5, 0.8, 0)),
                    obj('wardrobe_a', 'wardrobe', (.6, 1.2, 2), (4.6, 2.0, 0), math.pi),
                    obj('wardrobe_b', 'wardrobe', (.6, 1.2, 2), (4.6, 2.0, 0), math.pi)], [door(2.5, 0.0)])
        self.assertEqual(r.strict_audit(raw, r.door_keepouts(raw))['outside_1mm'], ['rug'])     # R2: rugs too
        result = r.repair(raw)
        self.assertEqual(result['status'], 'ok')
        v = d.accept(result['room'], result['room']['objects'])
        self.assertTrue(v['accepted'], v['reasons'])
        self.assertEqual(v['snap_moved'], [])                 # the endpoint's own push has nothing left to move
        self.assertEqual(result['diagnostics']['downstream']['s1']['reasons'], v['reasons'])
        self.assert_invariants(raw, result)

    def test_repair_never_turns_a_downstream_flag_false(self):
        # S1 before the a4_aabb component: sofa pushed off the column into a rotated cabinet's AABB (footprints
        # overlap < MIN_AREA, admit A4 counts it) and status ok while the mat stayed in the doorway
        raw = room([obj('mat', 'mat', (1.31, .92, .03), (1.344, .943, 0), math.pi / 2),
                    obj('sofa', 'sofa', (.67, .76, 1.78), (1.099, .712, 0), math.pi / 2),
                    obj('cabinet', 'cabinet', (1.4, 1.03, .58), (.992, 2.499, 0), .528),
                    obj('ottoman', 'ottoman', (1.04, 1.16, .03), (5.233, 1.459, 0))],
                   [door(1.993, 0.0), {'id': 'col', 'category': 'column', 'size': [.3, .3, 2.6], 'pos': [.677, .943, 0],
                                       'yaw': 0.0}], boundary=[[0, 0], [6.657, 0], [6.657, 3.168], [0, 3.168]])
        result = r.repair(raw)
        before, after = (d.accept(x, x['objects']) for x in (raw, result['room']))
        self.assertFalse([k for k in r.DOWNSTREAM_FLAGS if before[k] and not after[k]])
        self.assertTrue(after['accepted'], after['reasons'])
        self.assertEqual(result['status'], 'ok')
        self.assert_invariants(raw, result)

    def test_downstream_regression_rolls_back_to_raw(self):
        raw = room([obj('table', 'table', (1.2, .8, .75), (1.0, 3.63, 0)),
                    obj('cup', 'cup', (.1, .1, .1), (1.1, 3.7, .75), parent='table')])
        real, calls = d.accept, []

        def worse(room_, layout):                         # the repaired room loses admit_ok (e.g. A3, not a component)
            calls.append(real(room_, layout))
            return {**calls[-1], 'admit_ok': len(calls) == 1}
        with mock.patch.object(r.d, 'accept', worse):
            result = r.repair(raw)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['diagnostics']['rolled_back'], 'downstream_worse')
        self.assertEqual(result['status'], 'unchanged')
        self.assertEqual(json.dumps(result['room']), json.dumps(raw))

    def test_four_colocated_copies_separated_deterministically(self):
        raw = room([obj(f'chair_{k}', 'chair', (.5, .5, .9), (2.5, 2.5, 0)) for k in range(4)], [door(1.0, 0.0)])
        first, second = r.repair(raw), r.repair(copy.deepcopy(raw))
        self.assertEqual(json.dumps(first['room']), json.dumps(second['room']))
        self.assertEqual(first['status'], 'ok')
        self.assertTrue(d.accept(first['room'], first['room']['objects'])['accepted'])
        self.assertEqual(sum(o['pos'] == [2.5, 2.5, 0] for o in first['room']['objects']), 1)
        self.assert_invariants(raw, first)

    def test_timeout_keeps_raw_and_flags_every_unsolved_object(self):
        raw = room([obj(f'box_{k}', 'box', (1, 1, 1), (2.5, 2, 0)) for k in range(3)])
        result = r.repair(raw, time_limit=0)
        self.assertTrue(result['diagnostics']['timed_out'])
        self.assertEqual(result['status'], 'unchanged')
        self.assertEqual(json.dumps(result['room']), json.dumps(raw))
        self.assertTrue(all('timeout' in v for v in result['diagnostics']['unsolved'].values()))
        self.assertEqual(sorted(result['diagnostics']['unsolved']), ['box_0', 'box_1', 'box_2'])

    def test_child_moves_on_its_own_seat_before_its_parent(self):
        raw = room([obj('stand_a', 'nightstand', (.5, .5, .6), (1.0, 2.0, 0)),
                    obj('stand_b', 'nightstand', (.5, .5, .6), (1.52, 2.0, 0)),
                    obj('lamp_a', 'lamp', (.3, .3, .5), (1.2, 2.0, .6), parent='stand_a'),
                    obj('lamp_b', 'lamp', (.3, .3, .5), (1.4, 2.0, .6), parent='stand_b')])
        result = r.repair(raw)
        out = {o['id']: o for o in result['room']['objects']}
        self.assertEqual([out[k]['pos'] for k in ('stand_a', 'stand_b')], [[1.0, 2.0, 0], [1.52, 2.0, 0]])
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(support_ok(out['lamp_a'], out['stand_a']) and support_ok(out['lamp_b'], out['stand_b']))
        self.assert_invariants(raw, result)

    def test_malformed_input_is_rejected_unchanged(self):
        raw = room([obj('a', 'box', (1, 1, 1), (float('nan'), 1, 0))])
        result = r.repair(raw)
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['room']['objects'][0]['id'], 'a')


class ParityTests(unittest.TestCase):
    """audit_delta (incremental rows) equals check()/geometry()/strict_audit on 200 random rooms."""

    @staticmethod
    def random_room(rng):
        w, dd = rng.uniform(2.5, 7), rng.uniform(2.5, 6)
        boundary = rng.choice([[[0, 0], [w, 0], [w, dd], [0, dd]],
                               [[0, 0], [w, 0], [w, dd / 2], [w / 2, dd / 2], [w / 2, dd], [0, dd]]])
        cats = ['chair', 'table', 'bed', 'nightstand', 'wardrobe', 'sofa', 'rug', 'box', 'lamp', 'cabinet']
        objects = []
        for k in range(rng.randint(2, 14)):
            size = [rng.uniform(.2, 1.6), rng.uniform(.2, 1.4), rng.choice([.03, rng.uniform(.3, 2.2)])]
            o = obj(f'o{k}', rng.choice(cats), size, [rng.uniform(-.3, w + .3), rng.uniform(-.3, dd + .3), 0],
                    rng.choice([0.0, math.pi / 2, rng.uniform(0, 2 * math.pi)]))
            if objects and rng.random() < .25:
                p = rng.choice(objects)
                o.update(parent=p['id'], pos=[p['pos'][0] + rng.uniform(-.5, .5), p['pos'][1] + rng.uniform(-.5, .5),
                                              p['pos'][2] + p['size'][2] + rng.choice([0, 0, .2])])
            if objects and rng.random() < .15:
                o['pos'] = list(objects[-1]['pos'])
            objects.append(o)
        fixed = [door(rng.uniform(.5, w - .5), 0.0),
                 {'id': 'win', 'category': 'window', 'size': [.1, 1.2, 1.2], 'pos': [w / 2, dd if len(boundary) == 4 else dd / 2, .8], 'yaw': math.pi / 2},
                 {'id': 'col', 'category': 'column', 'size': [.3, .3, 2.6], 'pos': [rng.uniform(0, w), rng.uniform(0, dd / 2), 0], 'yaw': 0.0}]
        if rng.random() < .2:
            objects[-1]['parent'] = 'col'
        return {**room(objects, fixed, boundary, rng.choice(['polygon', 'polygon', 'hull', 'proxy'])),
                'height': rng.choice([2.6, None])}

    def test_incremental_audit_matches_the_frozen_checker(self):
        rng = random.Random(20261014)
        for _ in range(200):
            raw = self.random_room(rng)
            keepouts = r.door_keepouts(raw)
            ctx = r._context(raw, keepouts, ())
            state = r._state(ctx, {i: (0, 0) for i in ctx['ids']})
            for _ in range(3):
                unit = rng.choice(ctx['ids'])
                move = (rng.randint(-600, 600), rng.randint(-600, 600))
                state = r.audit_delta(ctx, state, {j: move for j in ctx['members'][unit]})
            moved = {**raw, 'objects': [state['cur'][i] for i in ctx['ids']]}
            full = r.audit(moved, keepouts, ctx['held'])
            self.assertEqual(state['v'], full['v'])
            self.assertAlmostEqual(state['phi'], full['phi'], places=9)
            c = check(moved)
            pairs = {k for k, (kind, _, _) in state['pair'].items() if kind == 'main'}
            self.assertEqual(pairs, set(map(tuple, c['collisions']['main'])))
            self.assertEqual(state['v'][1], g.geometry(moved)['support_fail'])
            if raw['boundary_type'] == 'proxy':
                continue
            v = d.accept(moved, moved['objects'])
            if not v['snap_moved']:                       # S1's R2 / A2 / A4 lists are accept()'s
                codes = lambda c: [x['ids'] for x in v['violations'] if x['code'] == c]
                self.assertEqual(full['lists']['a4_aabb'], codes('A4_COLLISION'))
                self.assertEqual({i for i, _ in full['lists']['keepout_1cm']}, {i for x in codes('A2_KEEPOUT_BLOCKED') for i in x})
                self.assertEqual(full['lists']['outside_1mm'], [i for x in codes('R2_OUTSIDE_1MM') for i in x])


class RowTests(unittest.TestCase):
    def fixture(self):
        raw = {'uid': 'r', 'source': 'S', 'group': 'g', 'boundary_type': 'polygon', 'room_type': 'bedroom',
               'boundary': [[0, 0], [5, 0], [5, 4], [0, 4]], 'height': 3,
               'objects': [obj('w1', 'wardrobe', (.6, 1.2, 2), (4.6, 2, 0)), obj('w2', 'wardrobe', (.6, 1.2, 2), (4.6, 2, 0)),
                           obj('bed', 'bed', (2, 1.6, .5), (1.2, 2, 0))],
               'fixed': [{'id': 'd1', 'category': 'door', 'size': [.1, .9, 2], 'pos': [2.5, 0, 0], 'yaw': math.pi / 2}]}
        bound = bind_room(raw)[0]
        request = {'uid': 'r', 'source': 'S', 'group': 'g', 'messages': messages(bound, 'joint')[:2]}
        self.placement = {**request, 'messages': messages(bound, 'placement', {o['id']: o['size'] for o in bound['objects']})[:2]}
        answer = json.dumps({'placements': [{'id': o['id'], 'size': o['size'], 'pos': [4.6, 2, 0] if o['category'] == 'wardrobe'
                                             else [1.2, 2, 0], 'yaw': 0} for o in bound['objects']]})
        row = {**g.score(bound, request['messages'], answer, 'joint'), 'arm': 'joint'}
        return bound, request, row

    def test_row_keeps_raw_beside_s1_and_summary_counts_every_request(self):
        bound, request, row = self.fixture()
        out = r.repair_row((row, bound, request))
        self.assertEqual(out['raw_layout'], row['raw_layout'])
        self.assertEqual(out['repaired_layout'], row['repaired_layout'])
        s1 = out['s1']
        self.assertFalse(s1['geometry_success_raw'])
        self.assertTrue(s1['geometry_success_s1'])
        self.assertEqual(s1['stratum'], '1-10')
        self.assertFalse(any(s1['anti_gaming_s1'][k] for k in r.BUGS))
        missing = r.repair_row(({**row, 'uid': 'x', 'complete': False, 'raw_layout': None}, bound, request))
        self.assertIsNone(missing['s1_layout'])
        summary = r.summarize([out, missing])
        self.assertEqual(summary['all']['requests'], 2)
        self.assertEqual(summary['all']['geometry_success_s1_rate'], .5)
        self.assertEqual(summary['by_stratum']['1-10']['requests'], 2)
        self.assertEqual(missing['s1']['rectangle'], True)
        self.assertEqual(missing['s1']['doors'], 1)
        self.assertEqual(summary['by_shape']['rectangle']['requests'], 2)
        self.assertEqual(summary['by_shape']['non_rectangle']['requests'], 0)
        self.assertEqual(summary['by_doors']['doors']['requests'], 2)
        self.assertEqual(summary['by_doors']['no_doors']['requests'], 0)
        disp = summary['all']['displacement']
        self.assertEqual((disp['objects'], disp['moved_objects'], disp['median']), (3, 1, 0.0))
        self.assertGreater(disp['moved_median'], 0)
        self.assertEqual(summary['all']['bugs'], {})

    def test_cli_verifies_inputs_and_publishes_once(self):
        bound, request, row = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, run = root / 'data', root / 'run'
            (data / 'joint').mkdir(parents=True)
            (data / 'placement').mkdir()
            run.mkdir()
            (data / 'rooms.jsonl').write_text(json.dumps(bound) + '\n')
            (data / 'joint' / 'requests.jsonl').write_text(json.dumps(request) + '\n')
            (data / 'placement' / 'requests.jsonl').write_text(json.dumps(self.placement) + '\n')
            h.write_json(data / 'MANIFEST.json', {'files': h.artifact_manifest(data)})
            (run / 'joint-geometry.jsonl').write_text(json.dumps(row) + '\n')
            h.write_json(run / 'MANIFEST.json', {'files': h.artifact_manifest(run)})
            summary = r.run(data, run / 'joint-geometry.jsonl', 'joint', root / 'out')
            self.assertEqual(summary['rows'], 1)
            names = sorted(p.name for p in (root / 'out').iterdir())
            self.assertEqual(names, ['MANIFEST.json', 'joint-geometry-s1.jsonl', 'report.md', 'summary.json'])
            with self.assertRaises(FileExistsError):
                r.run(data, run / 'joint-geometry.jsonl', 'joint', root / 'out')
            with self.assertRaises(ValueError):
                r.run(data, run / 'joint-geometry.jsonl', 'placement', root / 'out2')
            wider = root / 'wider'                        # a second frozen request the geometry file lacks
            (wider / 'joint').mkdir(parents=True)
            (wider / 'rooms.jsonl').write_text(json.dumps(bound) + '\n' + json.dumps({**bound, 'uid': 'r2'}) + '\n')
            (wider / 'joint' / 'requests.jsonl').write_text(json.dumps(request) + '\n'
                                                            + json.dumps({**request, 'uid': 'r2'}) + '\n')
            h.write_json(wider / 'MANIFEST.json', {'files': h.artifact_manifest(wider)})
            with self.assertRaisesRegex(ValueError, 'cover every frozen request'):
                r.run(wider, run / 'joint-geometry.jsonl', 'joint', root / 'out3')

    def test_uids_restrict_the_denominator_to_exactly_those_requests(self):
        bound, request, row = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, run = root / 'data', root / 'run'
            (data / 'joint').mkdir(parents=True)
            run.mkdir()
            (data / 'rooms.jsonl').write_text(json.dumps(bound) + '\n' + json.dumps({**bound, 'uid': 'r2'}) + '\n')
            (data / 'joint' / 'requests.jsonl').write_text(json.dumps(request) + '\n'
                                                           + json.dumps({**request, 'uid': 'r2'}) + '\n')
            h.write_json(data / 'MANIFEST.json', {'files': h.artifact_manifest(data)})
            (run / 'llm_harness-geometry.jsonl').write_text(json.dumps({**row, 'arm': 'llm_harness'}) + '\n')
            h.write_json(run / 'MANIFEST.json', {'files': h.artifact_manifest(run)})
            geometry = run / 'llm_harness-geometry.jsonl'
            cohort, plain = root / 'cohort.json', root / 'uids.json'
            cohort.write_text(json.dumps({'version': 1, 'uids': ['r']}))
            plain.write_text(json.dumps(['r']))
            summary = r.run(data, geometry, 'joint', root / 'out', uids=cohort)
            self.assertEqual((summary['rows'], summary['summary']['all']['requests']), (1, 1))
            self.assertEqual(summary['inputs']['uids'], {'file': 'cohort.json', 'sha256': h.sha256(cohort), 'count': 1})
            self.assertEqual(summary['metric_contract']['rate_denominator'], 'given_uids')
            self.assertIn('--uids 给定的全部 1 个请求', (root / 'out' / 'report.md').read_text())
            self.assertEqual(r.run(data, geometry, 'joint', root / 'out2', uids=plain)['rows'], 1)
            with self.assertRaisesRegex(ValueError, 'cover every frozen request'):
                r.run(data, geometry, 'joint', root / 'out3')
            for bad, match in ((['r', 'r2'], 'cover exactly the given uids'), (['r', 'r'], 'unique uid strings'),
                               ({'uids': []}, 'unique uid strings'), ({'x': ['r']}, 'unique uid strings')):
                plain.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, match):
                    r.run(data, geometry, 'joint', root / 'out4', uids=plain)
            self.assertFalse((root / 'out3').exists() or (root / 'out4').exists())


if __name__ == '__main__':
    unittest.main()
