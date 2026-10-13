import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastfill.comparison.data import Components, select_release_rows, sha256


class DataTests(unittest.TestCase):
    def fixture(self, path, rows):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return {'sha256': sha256(path), 'lines': len(rows)}

    def test_selection_balances_sources_and_is_stable(self):
        rows = [{'uid': f'{s}:{i}', 'source': s, 'flags': {}}
                for s in ['A', 'B'] for i in range(8)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'train.jsonl'
            info = self.fixture(p, rows)
            a, ledger = select_release_rows(p, info, 3, 'seed', True)
            info = self.fixture(p, list(reversed(rows)))
            b, _ = select_release_rows(p, info, 3, 'seed', True)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)
        self.assertEqual(ledger['selected_by_source'], {'A': 3, 'B': 3})

    def test_flags_excluded_for_train_only(self):
        rows = [{'uid': 'a', 'source': 'A', 'flags': {'fixed_collision': 1}},
                {'uid': 'b', 'source': 'A', 'flags': {}}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x'
            info = self.fixture(p, rows)
            train, _ = select_release_rows(p, info, 10, 'seed', True)
            dev, _ = select_release_rows(p, info, 10, 'seed', False)
        self.assertEqual(set(train), {'b'})
        self.assertEqual(set(dev), {'a', 'b'})

    def test_reject_source_drift_or_duplicate_uid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x'
            info = self.fixture(p, [{'uid': 'a', 'source': 'A', 'flags': {}}])
            p.write_text(p.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                select_release_rows(p, info, 1, 'seed', True)

    def test_reject_replacement_after_initial_hash(self):
        from fastfill.comparison import data
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'train.jsonl'
            info = self.fixture(p, [{'uid': 'original', 'source': 'A'}])
            real_hash = data.sha256

            def swap(path):
                value = real_hash(path)
                p.write_text(json.dumps({'uid': 'replacement', 'source': 'A'}) + '\n')
                return value

            with patch.object(data, 'sha256', side_effect=swap):
                with self.assertRaisesRegex(ValueError, 'hash|changed'):
                    select_release_rows(p, info, 1, 'seed', True)
            info = self.fixture(p, [{'uid': 'a', 'source': 'A'}] * 2)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                select_release_rows(p, info, 1, 'seed', True)

    def test_components_transitive_and_order_independent(self):
        a, b = Components(), Components()
        edges = [('house1', 'visit'), ('house2', 'visit'), ('house2', 'scan')]
        for x, y in edges:
            a.union(x, y)
        for x, y in reversed(edges):
            b.union(y, x)
        self.assertEqual(a.find('house1'), a.find('scan'))
        self.assertEqual(a.find('house1'), b.find('house1'))
        self.assertNotEqual(a.find('separate'), a.find('house1'))

    def build_fixture(self, root):
        ir, release = root / 'ir', root / 'release'
        ir.mkdir()
        release.mkdir()
        rooms = []
        for i in range(14):
            rooms.append({'uid': f'A:{i}', 'source': 'A', 'group': f'g{i}',
                          'room_type': 'bedroom', 'boundary_type': 'polygon',
                          'boundary': [[0, 0], [5+i/10, 0], [5+i/10, 5], [0, 5]],
                          'height': 3, 'meta': {'front_known': True, 'n_incomplete': 0},
                          'objects': [{'id': 'raw-chair', 'category': 'chair',
                                       'size': [0.5+i/100, 0.6, 1], 'pos': [1+i/20, 1, 0],
                                       'yaw': 0, 'anchor': 'floor', 'parent': None}]})
        info = self.fixture(ir / 'A.jsonl', rooms)
        manifest = {'args': {'boundary_types': ['polygon', 'hull'],
                             'source_anchors': {}, 'anchors': ['floor', 'object'],
                             'min_objects': 1, 'max_vertices': 0, 'oob_tol': 0.1,
                             'hidden_max': 0.3, 'reject_flagged': []},
                    'ir_sha256': {'A.jsonl': info['sha256']}, 'files': {}}
        for name, part in [('train', rooms[:12]), ('dev', rooms[12:])]:
            rows = [{'uid': r['uid'], 'source': r['source'], 'flags': {}} for r in part]
            manifest['files'][name+'.jsonl'] = self.fixture(release / (name+'.jsonl'), rows)
        (release / 'MANIFEST.json').write_text(json.dumps(manifest))
        return release, ir, rooms

    def test_end_to_end_atomic_paired_build(self):
        from fastfill.comparison.data import build
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            release, ir, _ = self.build_fixture(root)
            out = root / 'output'
            report = build(release, ir, out, 12, 2)
            self.assertEqual(report['counts'], {'train': 12, 'dev': 2})
            self.assertEqual(report['input_counterfactual_checks'], 28)
            for path, expected in report['files'].items():
                self.assertEqual(sha256(out / path), expected)
            joint = [json.loads(l) for l in (out/'joint/train.jsonl').read_text().splitlines()]
            place = [json.loads(l) for l in (out/'placement/train.jsonl').read_text().splitlines()]
            self.assertEqual([r['uid'] for r in joint], [r['uid'] for r in place])
            j_req = json.loads(joint[0]['messages'][1]['content'])
            p_req = json.loads(place[0]['messages'][1]['content'])
            self.assertNotIn('size', j_req['objects'][0])
            self.assertIn('size', p_req['objects'][0])
            self.assertNotIn('raw-chair', json.dumps(joint[0]['messages'][:2]))
            with self.assertRaises(FileExistsError):
                build(release, ir, out)

    def test_reconstruct_rejects_alias_leakage_and_ir_drift(self):
        from fastfill.comparison.data import reconstruct
        with tempfile.TemporaryDirectory() as d:
            release, ir, rooms = self.build_fixture(Path(d))
            man = json.loads((release/'MANIFEST.json').read_text())
            chosen = {'train': {'A:0': 'A'}, 'dev': {'A:13': 'A'}}
            rooms[0]['meta']['group_aliases'] = ['visit:shared']
            rooms[-1]['meta']['group_aliases'] = ['visit:shared']
            info = self.fixture(ir/'A.jsonl', rooms)
            with self.assertRaisesRegex(ValueError, 'IR hash'):
                reconstruct(ir, man, chosen)
            man['ir_sha256']['A.jsonl'] = info['sha256']
            with self.assertRaisesRegex(ValueError, 'cross-split component'):
                reconstruct(ir, man, chosen)

    def test_reconstruct_rejects_unresolved_and_cross_split_uid(self):
        from fastfill.comparison.data import reconstruct
        with tempfile.TemporaryDirectory() as d:
            release, ir, _ = self.build_fixture(Path(d))
            man = json.loads((release/'MANIFEST.json').read_text())
            with self.assertRaisesRegex(ValueError, 'cross-split UID'):
                reconstruct(ir, man, {'train': {'A:0': 'A'}, 'dev': {'A:0': 'A'}})
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                reconstruct(ir, man, {'train': {'absent': 'A'}})


if __name__ == '__main__':
    unittest.main()
