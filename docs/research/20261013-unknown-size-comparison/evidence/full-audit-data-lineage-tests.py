"""Dataset-free tests; run with python -B. No filesystem writes or Slurm needed."""
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(
    'lineage', Path(__file__).with_name('full-audit-data-lineage.py'))
lineage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lineage)


class IndependentCoreTests(unittest.TestCase):
    def test_raw_only_alias_bridge_and_minimum_root(self):
        dsu = lineage.DSU()
        for left, right in [('g:train', 'a:first'), ('g:unreleased', 'a:first'),
                            ('g:unreleased', 'a:last'), ('g:test', 'a:last'),
                            ('g:train', 'c:001')]:
            dsu.union(left, right)
        self.assertEqual(dsu.find('g:test'), 'a:first')
        self.assertEqual(dsu.find('g:train'), dsu.find('g:test'))
        self.assertNotEqual(dsu.find('g:other'), dsu.find('g:train'))

    def test_folds_are_sorted_hash_round_robin_and_whole_components(self):
        mapping = {f'u{i}': f'g:{i // 2}' for i in range(14)}
        seed = lineage.SEED
        expected_order = sorted(set(mapping.values()), key=lambda c: (
            hashlib.sha256(json.dumps([seed, c], separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest(), c))
        expected = {c: i % 5 for i, c in enumerate(expected_order)}
        folds = lineage.assign_folds(mapping)
        self.assertEqual(folds, {u: expected[c] for u, c in mapping.items()})
        for f in range(5):
            held = {u for u in mapping if folds[u] == f}
            fitted = set(mapping) - held
            self.assertFalse({mapping[u] for u in held} & {mapping[u] for u in fitted})
        with self.assertRaises(ValueError):
            lineage.assign_folds({'a': 'one'})

    def test_double_hash_compact_json_and_unicode(self):
        uids = {'room-é', 'z', 'a'}
        hashes = sorted(hashlib.sha256(u.encode()).hexdigest() for u in uids)
        expected = hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(lineage.uid_provenance(uids), (hashes, expected))
        self.assertNotEqual(expected, lineage.H('\n'.join(hashes)))

    def test_even_logmedian_axis_counts_fixed_and_empty_rooms(self):
        store = lineage.Observations()
        store.add('a', 'bed', [('chair', [1, 9, 4])], 7)
        store.add('b', 'bed', [('chair', [9, 1, 16])], 2)
        store.add('empty', '', [], 3)
        prior = store.fit({'a', 'b', 'empty'})
        self.assertEqual(prior['provenance']['room_count'], 3)
        self.assertEqual(prior['provenance']['fixed_excluded_count'], 12)
        self.assertEqual(prior['global']['count'], 2)
        for x, y in zip(prior['global']['size'], [3, 3, 8]):
            self.assertAlmostEqual(x, y, places=12)
        self.assertEqual(prior['category_room_type']['chair']['bed']['count'], 2)
        self.assertEqual(prior['provenance']['fitted_uid_hashes'], lineage.uid_provenance({'a', 'b', 'empty'})[0])
        with self.assertRaises(ValueError):
            store.fit({'empty'})
        with self.assertRaises(ValueError):
            store.add('bad', '', [('chair', [True, 2, 3])], 0)

    def test_all_six_fits_exclude_extreme_own_fold(self):
        store = lineage.Observations()
        mapping = {f'u{i}': f'g:{i}' for i in range(10)}
        folds = lineage.assign_folds(mapping)
        for i, u in enumerate(mapping):
            store.add(u, '', [('', [2 ** i, 3, 7])], 0)
        full = store.fit(set(mapping))
        self.assertEqual(full['global']['count'], 10)
        fitted_counts = 0
        for f in range(5):
            subset = {u for u in mapping if folds[u] != f}
            fit = store.fit(subset)
            vals = sorted(2 ** int(u[1:]) for u in subset)
            expected = math.sqrt(vals[3] * vals[4])
            self.assertAlmostEqual(fit['global']['size'][0], expected, places=10)
            self.assertEqual(fit['provenance']['fitted_uid_sha256'], lineage.uid_provenance(subset)[1])
            fitted_counts += fit['global']['count']
        self.assertEqual(fitted_counts, 4 * full['global']['count'])

    def test_content_fingerprint_exceptions_and_multiplicity(self):
        room = {'source': 'X', 'objects': [{'size': [1, 2, 3]}] * 3}
        dims = [(20, 40, 60)] * 3
        expected = hashlib.sha1(repr(dims).encode()).hexdigest()
        self.assertEqual(lineage.content_key(room), expected)
        self.assertIsNone(lineage.content_key({**room, 'source': 'MansionWorld'}))
        self.assertIsNone(lineage.content_key({**room, 'objects': room['objects'][:2]}))

    def test_layout_fingerprint_matches_explicit_four_rotations(self):
        room = {'boundary': [[0, 0], [4, 0], [4, 2], [0, 2]],
                'objects': [{'pos': [1, 0.5, 0], 'yaw': 0.0}]}
        forms = [([[0, 0], [4, 0], [4, 2], [0, 2]], [(1, 0.5, 0)]),
                 ([[0, 0], [2, 0], [2, 4], [0, 4]], [(1.5, 1, 90)]),
                 ([[0, 0], [4, 0], [4, 2], [0, 2]], [(3, 1.5, 180)]),
                 ([[0, 0], [2, 0], [2, 4], [0, 4]], [(0.5, 3, 270)])]
        expected = min(hashlib.sha1(repr(x).encode()).hexdigest() for x in forms)
        self.assertEqual(lineage.layout_key(room), expected)
        self.assertIsNone(lineage.layout_key({'boundary': []}))

    def test_historical_flags_and_heldout_closure(self):
        base = {'source': 'X', 'split': 'train', 'component': 'g:a', 'training_flags': {}}
        self.assertEqual(lineage.exclusion_reasons(base, set()), [])
        flagged = {**base, 'training_flags': {'fixed_collision': True, 'oob_objects': 1}}
        self.assertEqual(lineage.exclusion_reasons(flagged, set()),
                         ['historical_fixed_collision', 'historical_oob_objects'])
        self.assertEqual(lineage.exclusion_reasons(base, {'g:a'}), ['heldout_source_component'])
        self.assertEqual(lineage.exclusion_reasons({**base, 'split': 'dev'}, set()), ['not_exact_train'])

    def test_check_counts_examples_and_numeric_mismatch(self):
        audit = lineage.Audit(None)
        for i in range(12):
            audit.check(False, 'bad', i)
        self.assertEqual(audit.checks['bad']['failed'], 12)
        self.assertEqual(audit.checks['bad']['examples'], list(range(5)))
        audit.numeric([1.0, 2.0, 3.0], [1.0 + 1e-13, 2.0, 3.0], 'small')
        audit.numeric([1.0], [1.1], 'large')
        self.assertEqual(audit.checks['median_values']['failed'], 1)
        self.assertGreater(audit.max_delta['absolute'], 0.09)

    def test_stream_hashes_original_lines_and_counts_field_names(self):
        class MemoryPath:
            def open(self, mode):
                return io.BytesIO(payload)
            def __str__(self):
                return 'memory.jsonl'
        payload = b'{"uid":"u1","source":"X","meta":{"ok":true}}\n'
        audit = lineage.Audit(None)
        with patch.object(lineage, 'signature', return_value=(1, 2, len(payload), 4)):
            rows = list(audit.rows(MemoryPath(), hashlib.sha256(payload).hexdigest()))
        self.assertEqual(rows[0][2], hashlib.sha256(payload).hexdigest())
        self.assertEqual(audit.files['memory.jsonl']['fields'], {'uid': 1, 'source': 1, 'meta': 1})
        self.assertEqual(audit.files['memory.jsonl']['records'], 1)
        self.assertEqual(audit.files['memory.jsonl']['by_source'], {'X': 1})


if __name__ == '__main__':
    unittest.main()
