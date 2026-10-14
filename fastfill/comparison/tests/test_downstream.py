import copy
import math
import unittest

from shapely.geometry import Point, Polygon

from fastfill.comparison import downstream as d

RECT = [[0, 0], [5, 0], [5, 4], [0, 4]]
L_SHAPE = [[0, 0], [6, 0], [6, 3], [3, 3], [3, 6], [0, 6]]


def obj(i, category, size, pos, yaw=0.0, parent=None):
    return {'id': i, 'category': category, 'size': list(size), 'pos': list(pos), 'yaw': yaw, 'parent': parent}


def door(x, y, yaw=math.pi / 2, width=0.9, i='door_1'):
    return {'id': i, 'category': 'door', 'size': [0.1, width, 2.0], 'pos': [x, y, 0], 'yaw': yaw}


def room(objects, fixed=(), boundary=RECT):
    return {'boundary_type': 'polygon', 'boundary': boundary, 'height': 2.8, 'fixed': list(fixed),
            'objects': objects}


def verdict(r):
    return d.accept(r, r['objects'])


class DoorStripTests(unittest.TestCase):
    def test_strip_follows_wall_strip_rule(self):
        south, = d.door_strips(room([], [door(2.5, 0.0)]))
        self.assertEqual(tuple(round(v, 9) for v in south['strip'].bounds), (1.9, 0, 3.1, 1.0))
        self.assertEqual(south['seed'], (2.5, 0.25))
        self.assertAlmostEqual(south['width'], 0.9)
        self.assertFalse(south['detached'])
        east, = d.door_strips(room([], [door(5.0, 2.0, yaw=0.0)]))
        self.assertEqual(tuple(round(v, 9) for v in east['strip'].bounds), (4.0, 1.4, 5.0, 2.6))
        corner, = d.door_strips(room([], [door(0.3, 0.0)]))            # clamped to the wall
        self.assertAlmostEqual(corner['strip'].bounds[0], 0.0)
        middle, = d.door_strips(room([], [door(2.5, 2.0)]))
        self.assertTrue(middle['detached'])
        self.assertEqual(d.door_strips(room([], [{**door(2.5, 0), 'category': 'window'}])), [])


class AcceptTests(unittest.TestCase):
    def test_clean_layout_accepted_and_inputs_untouched(self):
        r = room([obj('bed', 'bed', (2, 1.6, .5), (1.05, 2.5, 0)), obj('lamp', 'lamp', (.2, .2, .4), (1.05, 2.5, .5),
                                                                          parent='bed')], [door(4.0, 0.0)])
        before = copy.deepcopy(r)
        v = verdict(r)
        self.assertTrue(v['accepted'] and v['admit_lenient_ok'] and v['admit_no_required_ok'], v['reasons'])
        self.assertEqual(v['parents'], {'bed': None, 'lamp': 'bed'})
        self.assertTrue(v['rectangle'])
        self.assertEqual(r, before)

    def test_small_intrusion_snapped_by_endpoint_large_one_blocked(self):
        small = verdict(room([obj('chair', 'chair', (.6, .6, .9), (2.5, 1.25, 0))], [door(2.5, 0.0)]))
        self.assertTrue(small['accepted'], small['reasons'])
        self.assertEqual(small['snap_moved'], ['chair'])
        large = verdict(room([obj('chair', 'chair', (.6, .6, .9), (2.5, 0.8, 0))], [door(2.5, 0.0)]))
        self.assertFalse(large['accepted'])
        self.assertIn('A2_KEEPOUT_BLOCKED', large['objects']['chair'])
        self.assertIn('R2_OUTSIDE_1MM', large['reasons'])
        self.assertIn('A3_ENTRY_BLOCKED', large['reasons'])

    def test_floor_covering_exempt_from_a2_but_not_from_endpoint(self):
        v = verdict(room([obj('rug', 'rug', (1.6, 1.2, .01), (2.5, 0.8, 0))], [door(2.5, 0.0)]))
        self.assertNotIn('A2_KEEPOUT_BLOCKED', v['reasons'])
        self.assertTrue(v['admit_ok'])
        self.assertFalse(v['endpoint_ok'])
        self.assertIn('R2_OUTSIDE_1MM', v['objects']['rug'])

    def test_window_overlap_is_reported_never_gated(self):
        window = {'id': 'win', 'category': 'window', 'size': [0.1, 1.2, 1.2], 'pos': [2.5, 4.0, 0.9], 'yaw': math.pi / 2}
        v = verdict(room([obj('dresser', 'dresser', (1.2, .5, 1.0), (2.5, 3.75, 0))], [window]))
        self.assertTrue(v['accepted'], v['reasons'])
        self.assertEqual(v['window_overlap'], [['dresser', 'win']])

    def test_r4_parent_needs_positive_aabb_overlap(self):
        table = obj('table', 'table', (1.2, .8, .75), (1.5, 2, 0))
        v = verdict(room([table, obj('cup', 'cup', (.1, .1, .1), (4, 1, .75), parent='table')]))
        self.assertIsNone(v['parents']['cup'])
        self.assertIn('R2_CHECK', v['reasons'])          # off its parent: the endpoint's check() already refuses it
        v = verdict(room([table, obj('cup', 'cup', (.1, .1, .1), (1.5, 2, .75), parent='table')]))
        self.assertEqual(v['parents']['cup'], 'table')

    def test_tucked_chair_is_an_a4_collision_only_with_a_required_set(self):
        v = verdict(room([obj('desk', 'desk', (1.2, .6, .75), (2.5, 3.6, 0)),
                          obj('chair', 'chair', (.5, .5, .9), (2.5, 3.3, 0))]))
        self.assertIn('A4_COLLISION', v['reasons'])
        self.assertFalse(v['admit_ok'])
        self.assertTrue(v['admit_no_required_ok'] and v['endpoint_ok'] and v['attach_ok'])

    def test_required_object_behind_a_wall_of_shelves_is_unreachable(self):
        shelves = [obj(f's{k}', 'shelf', (1.0, .4, 2.0), (0.5 + k, 2.0, 0)) for k in range(5)]
        v = verdict(room(shelves + [obj('bed', 'bed', (2, 1.4, .5), (2.5, 3.2, 0))], [door(2.5, 0.0)]))
        hit = [x for x in v['violations'] if x['code'] == 'A3_UNREACHABLE_REQUIRED']
        self.assertEqual([x['ids'] for x in hit], [['bed']])
        self.assertTrue(v['admit_no_required_ok'])

    def test_doorless_l_shape_and_its_notch(self):
        inside = verdict(room([obj('sofa', 'sofa', (2, .9, .8), (1.5, 5.4, 0))], boundary=L_SHAPE))
        self.assertTrue(inside['accepted'], inside['reasons'])
        self.assertFalse(inside['rectangle'])
        self.assertIn('A3', inside['not_computed'])
        self.assertEqual(inside['theta'], 0.0)
        notch = verdict(room([obj('sofa', 'sofa', (2, .9, .8), (4.5, 4.5, 0))], boundary=L_SHAPE))
        self.assertIn('R5_OUTSIDE_ENVELOPE', notch['objects']['sofa'])
        self.assertFalse(notch['endpoint_ok'] or notch['attach_ok'])

    def test_rotated_room_uses_its_own_frame(self):
        t = math.radians(30)
        rot = lambda x, y: [x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)]
        boundary = [rot(*p) for p in RECT]
        theta, m = d.frame(boundary)
        self.assertAlmostEqual(theta, t)
        box = d.aabb(obj('b', 'box', (1, .5, .5), rot(2, 2) + [0], yaw=t), m)
        self.assertAlmostEqual(box[2] - box[0], 1)
        self.assertAlmostEqual(box[3] - box[1], .5)
        self.assertAlmostEqual(box[0], 1.5)

    def test_pocket_cut_off_by_two_keepouts_is_not_floor(self):
        corner = room([obj('box', 'box', (.15, .15, .5), (.12, .12, 0))],
                      [door(.9, 0.0), door(0.0, .9, yaw=0.0, i='door_2')])
        v = verdict(corner)
        self.assertNotIn('R1_KEEPOUT_NOT_REPRESENTABLE', v['reasons'])
        self.assertIn('R2_OUTSIDE_1MM', v['objects']['box'])            # interface.py:137 drops the pocket
        ok, floor = d.keepout_floor(Polygon(RECT), [s['strip'] for s in d.door_strips(corner)])
        self.assertTrue(ok)
        self.assertFalse(floor.covers(Point(.12, .12)))
        self.assertTrue(floor.covers(Point(2.5, 2.5)))

    def test_strips_that_split_the_floor_are_not_representable(self):
        strip_room = room([], [door(2.5, 0.0), door(2.5, 2.0, i='door_2')],
                          boundary=[[0, 0], [5, 0], [5, 2], [0, 2]])
        v = verdict(strip_room)
        self.assertIn('R1_KEEPOUT_NOT_REPRESENTABLE', v['reasons'])
        self.assertFalse(v['endpoint_ok'])


class RoomGenBenchTests(unittest.TestCase):
    def test_export_keeps_ids_sizes_and_explicit_placements(self):
        r = room([obj('obj_0001', 'nightstand', (.5, .4, .6), (.3, 3.7, 0), yaw=math.pi * 1.5),
                  obj('obj_0002', 'lamp', (.2, .2, .4), (.3, 3.7, .6), parent='obj_0001')],
                 [{**door(2.5, 0.0), 'id': 'fixed_0001'}])
        names = {'obj_0001': 'nightstand_1', 'obj_0002': 'lamp_1', 'fixed_0001': 'door_1'}
        condition, layout, scene = d.to_roomgenbench(r, r['objects'], names)
        self.assertEqual([o['id'] for o in layout['objects']], ['nightstand_1', 'lamp_1'])
        self.assertEqual([o['place_id'] for o in scene['objects']], ['floor', 'nightstand_1'])
        self.assertEqual([o['place'] for o in scene['objects']], ['floor', 'on_object'])
        self.assertEqual([o['support_status'] for o in scene['objects']], ['inferred', 'declared'])
        stand = scene['objects'][0]
        self.assertEqual(stand['dimensions'], {'width': .4, 'length': .5, 'height': .6})
        self.assertAlmostEqual(stand['rotation']['z'], 180.0 if stand['rotation']['z'] > 0 else -180.0)
        self.assertEqual(condition['room']['fixed_objects'][0]['id'], 'door_1')
        self.assertEqual(len(scene['room']['doors']), 1)

    def test_export_infers_support_and_refuses_an_unplaced_object(self):
        r = room([obj('clock', 'clock', (.05, .3, .3), (.025, 2, 1.5)), obj('bed', 'bed', (2, 1.6, .5), (3, 2, 0)),
                  obj('box', 'box', (.3, .3, .3), (3, 2, .5))])
        _, _, scene = d.to_roomgenbench(r, r['objects'])
        self.assertEqual([(o['place_id'], o['support_status']) for o in scene['objects']],
                         [('wall', 'inferred'), ('floor', 'inferred'), ('bed', 'inferred')])
        floating = room([obj('box', 'box', (.3, .3, .3), (2, 2, .7))])     # no parent, nothing under it, off the wall
        with self.assertRaises(ValueError):
            d.to_roomgenbench(floating, floating['objects'])


if __name__ == '__main__':
    unittest.main()
