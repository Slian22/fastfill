"""Matched runner contracts: stdlib only, no network, GPU, or real training."""
import ast
import contextlib
import copy
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

# Load only the owned modules: comparison.__init__ belongs to the contract
# owner and eagerly imports scene/numpy/shapely. Helpers must work without them.
ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load_module('train_pair_helpers', ROOT / 'fastfill/comparison/train_pair_helpers.py')
package = types.ModuleType('fastfill.comparison')
package.train_pair_helpers = h
with patch.dict(sys.modules, {'fastfill.comparison': package}):
    runner = load_module('train_pair', ROOT / 'fastfill/comparison/train_pair.py')


class Tokenizer:
    eos_token = '<|im_end|>'
    eos_token_id = 999
    pad_token_id = 0

    def get_vocab(self):
        return {self.eos_token: self.eos_token_id}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return ''.join(m['role'] + ':' + m['content'] + '\n' for m in messages) + 'assistant:'

    def __call__(self, texts, **kwargs):
        assert kwargs == {'add_special_tokens': False}
        if isinstance(texts, str):
            texts = [texts]
        def ids(text):
            if text.endswith(self.eos_token):
                return [ord(c) for c in text[:-len(self.eos_token)]] + [self.eos_token_id]
            return [ord(c) for c in text]
        return {'input_ids': [ids(text) for text in texts]}


def row(uid='train-a', arm='joint', **extra):
    obj = {'id': 'obj_0001', 'category': 'chair'}
    target = {'id': obj['id'], 'pos': [1, 2, 0], 'yaw': 30}
    if arm == 'joint':
        target = {**target, 'size': [1, 1, 2]}
    else:
        obj = {**obj, 'size': [0.8, 0.9, 1.1]}
    return {'uid': uid, 'source': 'synthetic', 'group': uid, **extra, 'messages': [
        {'role': 'system', 'content': 'layout'},
        {'role': 'user', 'content': json.dumps({'objects': [obj]})},
        {'role': 'assistant', 'content': json.dumps({'placements': [target]})}]}


def fixture(root, changes=None):
    files = {}
    for arm in h.ARMS:
        (root / arm).mkdir(exist_ok=True)
        for split in h.SPLITS:
            rows = [row(split + '-a', arm), row(split + '-b', arm)]
            if changes:
                rows = changes(arm, split, rows)
            path = root / arm / (split + '.jsonl')
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'MANIFEST.json').write_text(json.dumps({'files': files, 'counts': {'train': 2, 'dev': 2}}))


def add_dev_rooms(root):
    path = root / 'dev_rooms.jsonl'
    path.write_text(''.join(json.dumps({'uid': 'dev-' + c}) + '\n' for c in 'ab'))
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    manifest['files']['dev_rooms.jsonl'] = h.sha256(path)
    h.write_json(root / 'MANIFEST.json', manifest)


def main_fixture(root):
    policy = 'fixed_estimated_size_pose_repaired_v1'
    def change(arm, split, rows):
        for record in rows:
            record['target_policy'] = policy
            record['target_qualified'] = split == 'train'
            if arm == 'placement' and split == 'train':
                target = json.loads(record['messages'][-1]['content'])
                target['placements'][0].update(pos=[2, 3, 0], yaw=90)
                record['messages'][-1]['content'] = json.dumps(target)
        return rows
    fixture(root, change)
    indexed = {arm: {r['uid']: r for r in map(json.loads,
               (root / arm / 'train.jsonl').read_text().splitlines())} for arm in h.ARMS}
    def evidence(record, arm):
        turns = record['messages']
        digest = lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
        canonical = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        sizes = {o['id']: o['size'] for o in json.loads(turns[1]['content'])['objects']} if arm == 'placement' else None
        return {'qualified': True, 'reasons': [],
                'counts': {'valid': True, **{k: 0 for k in
                    ('oob', 'support_fail', 'ceiling', 'fixed_blocking', 'main', 'severe')}},
                'request_sha256': digest(canonical(turns[:-1])),
                'target_sha256': digest(turns[-1]['content']),
                **({'estimated_sizes_sha256': digest(canonical(sizes))} if sizes is not None else {})}
    ledger = [{'uid': uid, 'accepted': True, 'eligible': True,
               **{arm: evidence(indexed[arm][uid], arm) for arm in h.ARMS}}
              for uid in sorted(indexed['joint'])]
    path = root / 'common-admission.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in ledger))
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    manifest['target_policy'] = policy
    manifest['files'][path.name] = h.sha256(path)
    h.write_json(root / 'MANIFEST.json', manifest)


def base_fixture(root):
    contents = {'config.json': json.dumps({'model_type': 'qwen3', 'max_position_embeddings': 40960}),
                'model.safetensors.index.json': '{"weight_map":{"weight":"model.safetensors"}}',
                'model.safetensors': 'synthetic weights', 'tokenizer.json': '{}',
                'tokenizer_config.json': '{}', 'generation_config.json': '{}'}
    meta = root / '.cache/huggingface/download'
    meta.mkdir(parents=True)
    tree = root / '.cache/huggingface/trees'
    tree.mkdir()
    (tree / (h.REVISION + '.json')).write_text('{}')
    for name, content in contents.items():
        (root / name).write_text(content)
        digest = h.sha256(root / name)
        (meta / (name + '.metadata')).write_text(h.REVISION + '\n' + digest + '\n0')
    return {name: h.sha256(root / name) for name in contents
            if name in ('config.json', 'model.safetensors.index.json', 'model.safetensors')}


class AdmissionTests(unittest.TestCase):
    def test_labels_match_existing_encode_and_include_eos(self):
        tok, record = Tokenizer(), row()
        encoded = h.encode_messages(tok, record['messages'])
        source = Path('fastfill/train.py').read_text()
        tree = ast.parse(source)
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('encode', 'end_token')]
        scope = {'prompt_text': lambda tok, msgs: tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)}
        exec(compile(ast.Module(body=defs, type_ignores=[]), '<existing encode>', 'exec'), scope)
        expected, dropped = scope['encode'](tok, [record['messages']], float('inf'))
        self.assertEqual(encoded, expected[0])
        self.assertEqual(dropped, [])
        prompt_len = encoded['labels'].index(next(v for v in encoded['labels'] if v != -100))
        self.assertGreater(prompt_len, 0)
        self.assertEqual(encoded['labels'][:prompt_len], [-100] * prompt_len)
        self.assertEqual(encoded['labels'][-1], tok.eos_token_id)
        self.assertEqual(encoded['input_ids'][-1], tok.eos_token_id)

    def test_common_admission_no_flag_filter_or_mutation(self):
        rows = {a: [row('b', a), row('a', a, flags={'oob_objects': True})] for a in h.ARMS}
        original = copy.deepcopy(rows)
        samples, report = h.common_admission(Tokenizer(), rows, 8192)
        self.assertEqual(report['accepted_uids'], ['a', 'b'])
        self.assertEqual(report['dropped'], [])
        self.assertEqual([len(samples[a]) for a in h.ARMS], [2, 2])
        self.assertEqual(rows, original)

    def test_overlength_in_either_arm_drops_uid_from_both(self):
        rows = {a: [row('keep', a), row('long-j', a), row('long-p', a)] for a in h.ARMS}
        rows['joint'][1]['messages'][0]['content'] = 'x' * 1000
        rows['placement'][2]['messages'][0]['content'] = 'x' * 1000
        samples, report = h.common_admission(Tokenizer(), rows, 500)
        self.assertEqual(report['accepted_uids'], ['keep'])
        self.assertEqual([len(samples[a]) for a in h.ARMS], [1, 1])
        self.assertEqual([x['uid'] for x in report['dropped']], ['long-j', 'long-p'])
        self.assertEqual(report['dropped'][0]['overlength_arms'], ['joint'])
        self.assertEqual(report['dropped'][1]['overlength_arms'], ['placement'])
        self.assertGreater(report['dropped'][0]['lengths']['joint'], 500)

    def test_exact_length_is_accepted_and_empty_fails(self):
        rows = {a: [row('a', a)] for a in h.ARMS}
        limit = max(len(h.encode_messages(Tokenizer(), rs[0]['messages'])['input_ids']) for rs in rows.values())
        self.assertEqual(h.common_admission(Tokenizer(), rows, limit)[1]['accepted_uids'], ['a'])
        with self.assertRaisesRegex(ValueError, 'empty'):
            h.common_admission(Tokenizer(), rows, 1)

    def test_bad_message_roles_fail(self):
        messages = row()['messages'][:-1]
        with self.assertRaises(ValueError):
            h.encode_messages(Tokenizer(), messages)


class ManifestTests(unittest.TestCase):
    def test_pair_rejects_shared_movable_target_fields(self):
        for field, value in (('pos', [1, 2, 0]), ('yaw', 30),
                             ('on', 'fixed_0001'), ('description', 'at x=1, y=2')):
            with self.subTest(field=field):
                j, p = row(), row(arm='placement')
                for record in (j, p):
                    req = json.loads(record['messages'][1]['content'])
                    obj = {**req['objects'][0], field: value}
                    record['messages'][1]['content'] = json.dumps({**req, 'objects': [obj]})
                with self.assertRaisesRegex(ValueError, 'movable input fields'):
                    h._check_pair(j, p, 'train')

    def test_pair_requires_explicit_nonempty_category(self):
        for category in (None, '', ' ', 123):
            with self.subTest(category=category):
                j, p = row(), row(arm='placement')
                for record in (j, p):
                    req = json.loads(record['messages'][1]['content'])
                    obj = {k: v for k, v in req['objects'][0].items() if k != 'category'}
                    if category is not None:
                        obj = {**obj, 'category': category}
                    record['messages'][1]['content'] = json.dumps({**req, 'objects': [obj]})
                with self.assertRaisesRegex(ValueError, 'movable input fields|category'):
                    h._check_pair(j, p, 'train')

    def test_main_policy_still_requires_equal_valid_support_graphs(self):
        j, p = row(), row(arm='placement')
        for record in (j, p):
            record['target_policy'] = h.MAIN_TARGET_POLICY
            record['target_qualified'] = True
            req = json.loads(record['messages'][1]['content'])
            record['messages'][1]['content'] = json.dumps({**req, 'fixed': [{'id': 'support'}]})
        target = json.loads(p['messages'][-1]['content'])
        target['placements'][0]['on'] = 'support'
        p['messages'][-1]['content'] = json.dumps(target)
        self.assertTrue(h.parse_prediction(p['messages'][-1]['content'], p, 'placement')['valid'])
        with self.assertRaisesRegex(ValueError, 'targets mismatch'):
            h._check_pair(j, p, 'train', h.MAIN_TARGET_POLICY)

    def test_main_policy_accepts_only_qualified_common_admission(self):
        for fault in ('none', 'missing_hash', 'corrupt', 'unqualified', 'not_accepted',
                      'missing_uid', 'false_row', 'missing_policy', 'downgrade', 'graph', 'size'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                main_fixture(root)
                manifest = json.loads((root / 'MANIFEST.json').read_text())
                ledger_path = root / 'common-admission.jsonl'
                ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
                if fault == 'missing_hash':
                    del manifest['files'][ledger_path.name]
                elif fault == 'corrupt':
                    ledger_path.write_text(ledger_path.read_text() + '\n')
                elif fault in ('unqualified', 'not_accepted', 'missing_uid'):
                    if fault == 'unqualified':
                        ledger[0]['placement']['qualified'] = False
                    elif fault == 'not_accepted':
                        ledger[0]['accepted'] = False
                    else:
                        ledger = ledger[1:]
                    ledger_path.write_text(''.join(json.dumps(r) + '\n' for r in ledger))
                    manifest['files'][ledger_path.name] = h.sha256(ledger_path)
                elif fault == 'downgrade':
                    del manifest['target_policy']
                elif fault in ('false_row', 'missing_policy', 'graph', 'size'):
                    path = root / 'placement/train.jsonl'
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if fault == 'false_row':
                        rows[0]['target_qualified'] = False
                    elif fault == 'missing_policy':
                        rows[0].pop('target_policy')
                    else:
                        target = json.loads(rows[0]['messages'][-1]['content'])
                        target['placements'][0][{'graph': 'on', 'size': 'size'}[fault]] = (
                            'obj_0001' if fault == 'graph' else [1, 1, 1])
                        rows[0]['messages'][-1]['content'] = json.dumps(target)
                    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                    manifest['files']['placement/train.jsonl'] = h.sha256(path)
                h.write_json(root / 'MANIFEST.json', manifest)
                if fault == 'none':
                    pairs, receipt = h.load_pair(root)
                    self.assertEqual(receipt['target_policy'], manifest['target_policy'])
                    self.assertEqual(len(pairs['train']['joint']), 2)
                else:
                    with self.assertRaises(ValueError):
                        h.load_pair(root)

    def test_main_ledger_requires_complete_clean_evidence(self):
        faults = ('request_sha256', 'target_sha256', 'estimated_sizes_sha256',
                  'missing_counts', 'missing_count', 'dirty_count', 'bool_count',
                  'float_count', 'false_valid', 'int_valid', 'dirty_reasons')
        for arm in h.ARMS:
            for fault in faults:
                if arm == 'joint' and fault == 'estimated_sizes_sha256':
                    continue
                with self.subTest(arm=arm, fault=fault), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    main_fixture(root)
                    path = root / 'common-admission.jsonl'
                    ledger = [json.loads(line) for line in path.read_text().splitlines()]
                    evidence = ledger[0][arm]
                    if fault.endswith('sha256'):
                        evidence.pop(fault)
                    elif fault == 'missing_counts':
                        evidence.pop('counts')
                    elif fault == 'missing_count':
                        evidence['counts'].pop('oob')
                    elif fault == 'dirty_reasons':
                        evidence['reasons'] = ['unqualified']
                    else:
                        key, value = {'dirty_count': ('oob', 1), 'bool_count': ('oob', False),
                                      'float_count': ('oob', 0.0), 'false_valid': ('valid', False),
                                      'int_valid': ('valid', 1)}[fault]
                        evidence['counts'][key] = value
                    path.write_text(''.join(json.dumps(r) + '\n' for r in ledger))
                    manifest = json.loads((root / 'MANIFEST.json').read_text())
                    manifest['files'][path.name] = h.sha256(path)
                    h.write_json(root / 'MANIFEST.json', manifest)
                    with self.assertRaises(ValueError):
                        h.load_pair(root)

    def test_main_ledger_rejects_rehashed_target_request_and_size_changes(self):
        for fault in ('joint_target', 'placement_target', 'joint_system', 'placement_size',
                      'shared_context', 'size_with_updated_request_hash', 'wrong_target_hash'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                main_fixture(root)
                manifest = json.loads((root / 'MANIFEST.json').read_text())
                ledger_path = root / 'common-admission.jsonl'
                ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
                arms = h.ARMS if fault == 'shared_context' else (
                    'joint' if fault.startswith('joint') else 'placement',)
                for arm in arms:
                    path = root / arm / 'train.jsonl'
                    records = [json.loads(line) for line in path.read_text().splitlines()]
                    turns = records[0]['messages']
                    if fault.endswith('_target'):
                        target = json.loads(turns[-1]['content'])
                        target['placements'][0]['pos'] = [500, 500, 0]
                        turns[-1]['content'] = json.dumps(target)
                    elif fault == 'joint_system':
                        turns[0]['content'] += ' changed'
                    elif fault == 'wrong_target_hash':
                        ledger[0][arm]['target_sha256'] = '0' * 64
                    else:
                        user = json.loads(turns[1]['content'])
                        if fault == 'shared_context':
                            user['height'] = 9
                        else:
                            user['objects'][0]['size'] = [100, 100, 100]
                        turns[1]['content'] = json.dumps(user)
                        if fault == 'size_with_updated_request_hash':
                            canonical = json.dumps(turns[:-1], ensure_ascii=False, allow_nan=False, sort_keys=True)
                            ledger[0][arm]['request_sha256'] = hashlib.sha256(canonical.encode()).hexdigest()
                    path.write_text(''.join(json.dumps(r) + '\n' for r in records))
                    manifest['files'][str(path.relative_to(root))] = h.sha256(path)
                ledger_path.write_text(''.join(json.dumps(r) + '\n' for r in ledger))
                manifest['files'][ledger_path.name] = h.sha256(ledger_path)
                h.write_json(root / 'MANIFEST.json', manifest)
                with self.assertRaises(ValueError):
                    h.load_pair(root)

    def test_target_policy_is_explicit_and_cannot_bypass_pose_or_graph_checks(self):
        for policy in (None, 'identical_poses_v1', 'unknown', 'fixed_estimated_size_pose_repaired_v1'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                def change(arm, split, rows):
                    if arm == 'placement':
                        target = json.loads(rows[0]['messages'][-1]['content'])
                        target['placements'][0]['pos'] = [2, 3, 0]
                        rows[0]['messages'][-1]['content'] = json.dumps(target)
                    return rows
                fixture(root, change)
                manifest = json.loads((root / 'MANIFEST.json').read_text())
                if policy is not None:
                    manifest['target_policy'] = policy
                h.write_json(root / 'MANIFEST.json', manifest)
                with self.assertRaises(ValueError):
                    h.load_pair(root)

    def test_unknown_policy_rejected_even_with_identical_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            manifest = json.loads((root / 'MANIFEST.json').read_text())
            h.write_json(root / 'MANIFEST.json', {**manifest, 'target_policy': 'typo'})
            with self.assertRaisesRegex(ValueError, 'policy'):
                h.load_pair(root)

    def test_paired_manifest_and_parent_flags_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            pairs, receipt = h.load_pair(root)
            self.assertEqual(len(pairs['train']['joint']), 2)
            self.assertEqual(len(receipt['files']), 4)
            self.assertEqual(receipt['manifest_sha256'], h.sha256(root / 'MANIFEST.json'))

    def test_corrupt_data_or_missing_manifest_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / 'joint/train.jsonl'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                h.load_pair(root)
            (root / 'MANIFEST.json').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                h.load_pair(root)

    def test_uid_mismatch_duplicate_and_cross_split_fail(self):
        for kind in ('mismatch', 'duplicate', 'overlap', 'empty'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                def change(arm, split, rows):
                    if kind == 'mismatch' and arm == 'placement':
                        return [{**rows[0], 'uid': split + '-other'}, rows[1]]
                    if kind == 'duplicate':
                        return [rows[0], rows[0]]
                    if kind == 'overlap':
                        return [{**rows[0], 'uid': 'overlap'}, rows[1]]
                    return [] if kind == 'empty' else rows
                fixture(Path(tmp), change)
                with self.assertRaises(ValueError):
                    h.load_pair(Path(tmp))

    def test_pair_request_and_metadata_mismatch_fail(self):
        for field in ('source', 'group', 'request'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                def change(arm, split, rows):
                    if arm == 'placement':
                        if field == 'request':
                            rows[0]['messages'][1]['content'] = '{"objects":[]}'
                        else:
                            rows[0] = {**rows[0], field: 'different'}
                    return rows
                fixture(Path(tmp), change)
                with self.assertRaises(ValueError):
                    h.load_pair(Path(tmp))

    def test_manifest_cannot_escape_root_or_skip_required_file(self):
        for entry in ('../elsewhere', 'joint/train.jsonl'):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root)
                path = root / 'MANIFEST.json'
                manifest = json.loads(path.read_text())
                manifest['files'].pop('joint/train.jsonl')
                if entry.startswith('..'):
                    manifest['files'][entry] = '0' * 64
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    h.load_pair(root)


class SchemaTests(unittest.TestCase):
    def parse(self, record, text=None, arm='joint'):
        return h.parse_prediction(text or record['messages'][-1]['content'], record, arm)

    def test_valid_joint_and_placement_prior_preserved_without_gt(self):
        for arm in h.ARMS:
            record = row(arm=arm)
            result = self.parse(record, arm=arm)
            self.assertTrue(result['valid'])
            obj = result['placements'][0]
            self.assertEqual(obj['size'], [1, 1, 2] if arm == 'joint' else [0.8, 0.9, 1.1])
        changed = row(arm='placement')
        changed['messages'][-1]['content'] = 'GT must never be read'
        self.assertTrue(self.parse(changed, row(arm='placement')['messages'][-1]['content'], 'placement')['valid'])

    def test_extra_duplicate_missing_ids_and_prose_fail(self):
        record = row()
        value = json.loads(record['messages'][-1]['content'])
        obj = value['placements'][0]
        bad = [{'placements': []}, {'placements': [obj, obj]},
               {'placements': [obj, {**obj, 'id': 'extra'}]},
               {'placements': [{**obj, 'id': 'extra'}]}]
        for value in bad:
            self.assertFalse(self.parse(record, json.dumps(value))['valid'])
        for text in ('```json\n{}\n```', 'prefix {}', '{"placements":[],"placements":[]}'):
            self.assertFalse(self.parse(record, text)['valid'])

    def test_bad_sizes_numbers_yaw_and_unknown_fields_fail(self):
        obj = json.loads(row()['messages'][-1]['content'])['placements'][0]
        changes = [{'size': s} for s in ([0, 1, 1], [-1, 1, 1], [True, 1, 1], [1, 2],
                                        [float('nan'), 1, 1], [1000, 1, 1], ['1', 1, 1])]
        changes += [{'pos': [float('inf'), 0, 0]}, {'yaw': True}, {'yaw': 360},
                    {'yaw': 2.5}, {'unexpected': 1}, {'on': 'missing'}, {'on': 'obj_0001'}]
        for change in changes:
            with self.subTest(change=change):
                result = self.parse(row(), json.dumps({'placements': [{**obj, **change}]}))
                self.assertFalse(result['valid'])
                self.assertTrue(result['error'])

    def test_placement_size_output_and_bad_prior_fail(self):
        self.assertFalse(self.parse(row(arm='placement'), row()['messages'][-1]['content'], 'placement')['valid'])
        record = row(arm='placement')
        record['messages'][1]['content'] = '{"objects":[{"id":"obj_0001","size":[0,1,1]}]}'
        self.assertFalse(self.parse(record, arm='placement')['valid'])


class RunnerTests(unittest.TestCase):
    def args(self, *extra):
        return runner.parse_args(['--data', '/data', '--base', '/base', '--out', '/new', *extra])

    def test_budget_defaults_and_exclusive_epochs(self):
        args = self.args()
        self.assertEqual((args.max_steps, args.max_len, args.grad_accum, args.bs), (8, 8192, 4, 1))
        self.assertEqual((args.arm, args.run_purpose, args.max_new_tokens, args.checkpoint_steps),
                         ('both', 'engineering', 2048, None))
        self.assertEqual(self.args('--epochs', '2.5').epochs, 2.5)
        self.assertIsNone(self.args('--epochs', '2.5').max_steps)
        for extra in [('--epochs', '1', '--max-steps', '8'), ('--max-steps', '0'),
                      ('--max-len', '-1'), ('--epochs', 'nan'), ('--bs', '0'),
                      ('--checkpoint-steps', '0'), ('--max-new-tokens', '0'), ('--arm', 'other'),
                      ('--run-purpose', 'matched_sft'), ('--resume-from-checkpoint', '/old')]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*extra)

    def test_single_arm_completion_does_not_claim_matched_pair(self):
        for arm in h.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                args = runner.parse_args(['--data', tmp, '--base', tmp, '--out', str(Path(tmp) / 'out'),
                    '--arm', arm, '--run-purpose', 'matched_sft', '--epochs', '1', '--checkpoint-steps', '2'])
                launched = []
                class Process:
                    exitcode = 0
                    def __init__(self, target, args):
                        launched.append((args[1], target.__name__))
                    def start(self):
                        pass
                    def join(self):
                        directory = args.out / arm
                        directory.mkdir(exist_ok=True)
                        h.write_json(directory / 'training_result.json', {'status': 'TRAINING_SAVED', 'finite_loss': True})
                        h.write_json(directory / 'result.json', {'status': 'TRAINING_COMPLETE',
                                     'clean_load': True, 'dev_count': 2})
                        h.write_json(directory / 'SHA256.json', h.artifact_manifest(directory))
                    def is_alive(self):
                        return False
                    def close(self):
                        pass
                prepared = ({'run_purpose': 'matched_sft'}, {a: [] for a in h.ARMS},
                            {a: [] for a in h.ARMS}, {})
                with patch.object(runner, 'prepare', return_value=prepared), patch.object(
                        runner.multiprocessing, 'get_context', return_value=types.SimpleNamespace(Process=Process)):
                    status = runner.run(args)
                self.assertEqual(launched, [(arm, '_arm_worker'), (arm, '_probe_worker')])
                self.assertTrue(status['process_isolation_complete'])
                self.assertEqual(status['status'], 'TRAINING_COMPLETE')
                self.assertEqual(status['completed_arms'], [arm])
                self.assertFalse(status['pair_complete'])
                self.assertFalse(status['quality_claim'])

    def test_periodic_checkpoint_inventory_requires_training_and_rng_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                     'scheduler.pt', 'rng_state.pth', 'trainer_state.json')
            for name in names:
                (root / name).write_text('{}')
            runner._checkpoint_inventory(root)
            h.verify_artifacts(root, h.strict_json((root / 'SHA256.json').read_text()))
            for name in names:
                (root / name).unlink()
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    runner._checkpoint_inventory(root)
                (root / name).write_text('{}')

    def test_checkpoint_callback_checks_exact_saved_step(self):
        modules = {'transformers': types.SimpleNamespace(TrainerCallback=object)}
        with patch.dict(sys.modules, modules), patch.object(runner, '_checkpoint_inventory') as inventory:
            control = object()
            result = runner._checkpoint_callback().on_save(types.SimpleNamespace(output_dir='/out'),
                types.SimpleNamespace(global_step=4), control)
            self.assertIs(result, control)
            inventory.assert_called_once_with(Path('/out/checkpoint-4'))

    def test_main_preflight_cannot_use_unqualified_pilot_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            args = runner.parse_args(['--data', tmp, '--base', tmp, '--out', str(root / 'out'),
                '--run-purpose', 'matched_sft', '--epochs', '1', '--checkpoint-steps', '2', '--preflight-only'])
            with patch.object(h, 'base_receipt') as base, self.assertRaisesRegex(ValueError, 'policy'):
                runner.run(args)
            base.assert_not_called()

    def test_import_and_help_do_not_load_heavy_dependencies(self):
        script = "from fastfill.comparison import train_pair; import sys; " + \
                 "assert not {'torch','transformers','peft'} & set(sys.modules)"
        subprocess.run([sys.executable, '-B', '-c', script], check=True, capture_output=True)
        result = subprocess.run([sys.executable, '-B', '-m', 'fastfill.comparison.train_pair', '--help'],
                                check=True, capture_output=True, text=True)
        self.assertIn('--preflight-only', result.stdout)

    def test_helpers_import_without_site_packages(self):
        script = ('import importlib.util; s=importlib.util.spec_from_file_location("h", '
                  + repr(str(ROOT / 'fastfill/comparison/train_pair_helpers.py'))
                  + '); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)')
        subprocess.run([sys.executable, '-B', '-S', '-c', script], check=True, capture_output=True)

    def test_artifact_hash_manifest_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'adapter_config.json').write_text('{}')
            manifest = h.artifact_manifest(root)
            h.verify_artifacts(root, manifest)
            (root / 'adapter_config.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'hash'):
                h.verify_artifacts(root, manifest)

    def test_base_receipt_hashes_payload_and_tokenizer_and_pins_revision(self):
        for change in ('none', 'weights', 'tokenizer', 'revision', 'adapter', 'missing_shard'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                hashes = base_fixture(root)
                if change == 'weights':
                    (root / 'model.safetensors').write_text('corrupt')
                elif change == 'tokenizer':
                    (root / 'tokenizer.json').write_text('{"corrupt":true}')
                elif change == 'revision':
                    (root / '.cache/huggingface/download/config.json.metadata').write_text('wrong\nhash')
                elif change == 'adapter':
                    (root / 'adapter_config.json').write_text('{}')
                elif change == 'missing_shard':
                    (root / 'model.safetensors').unlink()
                with patch.object(h, 'BASE_HASHES', hashes):
                    if change == 'none':
                        receipt = h.base_receipt(root)
                        self.assertTrue(receipt['weight_hashes_freshly_verified'])
                        self.assertEqual(receipt['revision'], h.REVISION)
                        self.assertEqual(receipt['files']['model.safetensors'], hashes['model.safetensors'])
                    else:
                        with self.assertRaises(ValueError):
                            h.base_receipt(root)

    def test_preflight_end_to_end_with_fake_tokenizer_never_starts_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, base, out = root / 'data', root / 'base', root / 'out'
            data.mkdir()
            fixture(data)
            add_dev_rooms(data)
            hashes = base_fixture(base)
            args = runner.parse_args(['--data', str(data), '--base', str(base), '--out', str(out), '--preflight-only'])
            with patch.object(h, 'BASE_HASHES', hashes), patch.object(runner, '_tokenizer', return_value=Tokenizer()), \
                    patch.object(runner.multiprocessing, 'get_context') as spawn:
                result = runner.run(args)
                spawn.assert_not_called()
            self.assertEqual(result['status'], 'PREFLIGHT_COMPLETE')
            self.assertFalse(result['training_executed'])
            receipt = json.loads((out / 'run_receipt.json').read_text())
            self.assertEqual(receipt['config']['lora']['r'], 32)
            self.assertEqual(receipt['config']['learning_rate'], 5e-5)
            self.assertIn('fastfill/comparison/contract.py', receipt['code_sha256'])
            self.assertEqual(json.loads((out / 'dev_selection.json').read_text())['uids'], ['dev-a', 'dev-b'])
            h.verify_artifacts(out, json.loads((out / 'SHA256.json').read_text()))
            before = (out / 'status.json').read_bytes()
            with self.assertRaises(FileExistsError):
                runner.run(args)
            self.assertEqual(before, (out / 'status.json').read_bytes())

    def test_empty_admission_keeps_all_length_drops_before_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            args = runner.parse_args(['--data', tmp, '--base', tmp, '--out', str(root / 'out'),
                                      '--preflight-only', '--max-len', '1'])
            with patch.object(h, 'base_receipt', return_value={'max_position_embeddings': 40960}), \
                    patch.object(runner, '_tokenizer', return_value=Tokenizer()), self.assertRaisesRegex(ValueError, 'empty'):
                runner.run(args)
            lengths = json.loads((args.out / 'lengths.json').read_text())
            self.assertEqual(set(lengths), {'train', 'dev'})
            self.assertEqual([len(lengths[s]['dropped']) for s in h.SPLITS], [2, 2])
            self.assertEqual(json.loads((args.out / 'status.json').read_text())['status'], 'FAIL')

    def test_processes_sequential_and_failure_blocks_second_arm(self):
        for failure in (None, '_arm_worker', '_probe_worker', 'hash', 'evidence'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                args = runner.parse_args(['--data', tmp, '--base', tmp, '--out', str(Path(tmp) / 'out')])
                events = []
                class Process:
                    def __init__(self, target, args):
                        self.arm = args[1]
                        self.stage = target.__name__
                        self.exitcode = int(self.stage == failure)
                    def start(self):
                        events.append(('start', self.arm, self.stage))
                    def join(self):
                        events.append(('join', self.arm, self.stage))
                        arm_dir = args.out / self.arm
                        arm_dir.mkdir(exist_ok=True)
                        h.write_json(arm_dir / 'training_result.json', {'status': 'TRAINING_SAVED',
                                                                     'finite_loss': failure != 'evidence'})
                        h.write_json(arm_dir / 'result.json', {'status': 'ENGINEERING_COMPLETE', 'clean_load': True, 'dev_count': 2})
                        h.write_json(arm_dir / 'SHA256.json', h.artifact_manifest(arm_dir))
                        if failure == 'hash':
                            (arm_dir / 'training_result.json').write_text('{}')
                    def is_alive(self):
                        return False
                    def close(self):
                        events.append(('close', self.arm, self.stage))
                prepared = ({}, {a: [] for a in h.ARMS}, {a: [] for a in h.ARMS}, {})
                with patch.object(runner, 'prepare', return_value=prepared), \
                        patch.object(runner.multiprocessing, 'get_context', return_value=types.SimpleNamespace(Process=Process)):
                    if failure:
                        with self.assertRaises((RuntimeError, ValueError)):
                            runner.run(args)
                        starts = [e for e in events if e[0] == 'start']
                        self.assertEqual([e[2] for e in starts], ['_arm_worker', '_probe_worker']
                                         if failure == '_probe_worker' else ['_arm_worker'])
                    else:
                        self.assertEqual(runner.run(args)['status'], 'ENGINEERING_COMPLETE')
                        self.assertEqual(events, [(event, arm, stage) for arm in h.ARMS
                            for stage in ('_arm_worker', '_probe_worker') for event in ('start', 'join', 'close')])

    def test_inventory_blocks_missing_files_and_verifies_complete_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                runner._adapter_inventory(root)
            for name in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json', 'tokenizer.json'):
                (root / 'adapter').mkdir(exist_ok=True)
                (root / 'adapter' / name).write_text('{}')
            (root / 'trainer_state.json').write_text('{}')
            manifest = runner._adapter_inventory(root)
            self.assertEqual(len(manifest), 5)
            h.verify_artifacts(root, manifest)

    def test_gpu_release_detects_lingering_tensor_allocations(self):
        torch = Mock()
        torch.cuda.memory_allocated.return_value = 0
        runner._release_gpu(torch)
        torch.cuda.empty_cache.assert_called_once()
        torch.cuda.memory_allocated.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'allocations'):
            runner._release_gpu(torch)

    def test_nonfinite_microbatch_fails_before_logging(self):
        class Number:
            def __init__(self, value):
                self.value = value
            def detach(self):
                return self
            def item(self):
                return self.value
            def sum(self):
                return self
            def __ne__(self, other):
                return self
        class Trainer:
            def compute_loss(self, model, inputs, *args, **kwargs):
                return Number(model)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'transformers': types.SimpleNamespace(Trainer=Trainer)}):
            trainer = runner._trainer_class()()
            trainer.args = types.SimpleNamespace(output_dir=tmp)
            trainer.state = types.SimpleNamespace(global_step=0)
            trainer.pair_tokens = trainer.pair_supervised = 0
            inputs = {'attention_mask': Number(100), 'labels': Number(20)}
            trainer.compute_loss(1.5, inputs)
            self.assertEqual((trainer.pair_tokens, trainer.pair_supervised), (100, 20))
            with self.assertRaises(FloatingPointError):
                trainer.compute_loss(float('nan'), inputs)
            self.assertEqual(len((Path(tmp) / 'loss.jsonl').read_text().splitlines()), 1)

    def test_training_recipe_and_save_with_stub_trainer(self):
        for epochs in (None, 2.0):
            with self.subTest(epochs=epochs), tempfile.TemporaryDirectory() as tmp:
                calls, seeds = [], []
                class Arguments:
                    def __init__(self, **kwargs):
                        self.values = kwargs
                        calls.append(kwargs)
                    def to_dict(self):
                        return self.values
                class Trainer:
                    def __init__(self, **kwargs):
                        self.state = types.SimpleNamespace(global_step=8 if epochs is None else 2,
                                                            log_history=[{'loss': 1.0}])
                        self.accelerator = types.SimpleNamespace(free_memory=lambda: None)
                    def train(self):
                        return types.SimpleNamespace(training_loss=1.0, metrics={'train_runtime': 0.1})
                    def save_model(self, path):
                        calls.append(('save_model', path))
                    def save_state(self):
                        calls.append('save_state')
                transformers = types.SimpleNamespace(TrainingArguments=Arguments,
                    set_seed=seeds.append, DataCollatorForSeq2Seq=lambda *a, **k: None)
                lora = Mock(side_effect=lambda **kwargs: kwargs)
                peft = types.SimpleNamespace(LoraConfig=lora, get_peft_model=lambda model, config: model)
                config = {'bs': 1, 'grad_accum': 4, 'epochs': epochs, 'max_steps': 8 if epochs is None else None,
                          'checkpoint_steps': 1 if epochs is not None else None}
                tok, model = Mock(), Mock()
                with patch.dict(sys.modules, {'transformers': transformers, 'peft': peft}), \
                        patch.object(runner, '_load_model', return_value=model), \
                        patch.object(runner, '_trainer_class', return_value=Trainer), \
                        patch.object(runner, '_checkpoint_callback', return_value=object()):
                    metrics = runner._train(config, Path(tmp), [1, 2, 3, 4], tok)
                self.assertTrue(metrics['finite_loss'])
                self.assertEqual(seeds, [42, 42])
                self.assertEqual(lora.call_args.kwargs, runner.LORA)
                self.assertEqual(calls[0]['max_steps'], 8 if epochs is None else -1)
                self.assertEqual(calls[0]['learning_rate'], 5e-5)
                self.assertTrue(calls[0]['gradient_checkpointing'])
                self.assertEqual(calls[0]['save_strategy'], 'no' if epochs is None else 'steps')
                self.assertEqual(calls[0]['save_total_limit'], 2)
                self.assertFalse(calls[0]['save_only_model'])
                if epochs is not None:
                    self.assertEqual(calls[0]['save_steps'], 1)
                self.assertIn('save_state', calls)
                tok.save_pretrained.assert_called_once_with(Path(tmp) / 'adapter')

    def test_generation_reloads_fresh_base_and_calls_contract_on_exact_rooms(self):
        class Output(list):
            def __getitem__(self, index):
                return self if isinstance(index, tuple) else super().__getitem__(index)
        class Batch(dict):
            def to(self, device):
                self.device = device
                return self
        tok = Mock()
        tok.return_value = Batch(input_ids=types.SimpleNamespace(shape=(1, 4)))
        tok.get_vocab.return_value = {'<|im_end|>': 999}
        tok.pad_token_id = 0
        tok.decode.return_value = row()['messages'][-1]['content']
        model, base, validate = Mock(), Mock(), Mock()
        model.generate.return_value = Output([1, 999])
        rooms = {'dev-a': {'uid': 'dev-a'}, 'dev-b': {'uid': 'dev-b'}}
        validate.side_effect = [None, ValueError('contract rejection')]
        peft = types.SimpleNamespace(PeftModel=types.SimpleNamespace(from_pretrained=Mock(return_value=model)))
        modules = {'torch': types.SimpleNamespace(inference_mode=contextlib.nullcontext), 'peft': peft,
                   'transformers': types.SimpleNamespace(set_seed=Mock()),
                   'fastfill.comparison.contract': types.SimpleNamespace(validate_output=validate)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.object(runner, '_tokenizer', return_value=tok), \
                patch.object(runner, '_load_model', return_value=base) as fresh:
            result = runner._generate({}, Path(tmp), 'joint', [row('dev-a'), row('dev-b')], rooms)
            self.assertTrue(result['clean_load'])
            self.assertEqual((result['dev_count'], result['schema_valid']), (2, 1))
            fresh.assert_called_once()
            self.assertIs(peft.PeftModel.from_pretrained.call_args.args[0], base)
            self.assertEqual([c.args[1] for c in validate.call_args_list], list(rooms.values()))
            self.assertEqual(model.generate.call_args.kwargs['max_new_tokens'], 2048)
            self.assertFalse(model.generate.call_args.kwargs['do_sample'])
            self.assertEqual(len(json.loads((Path(tmp) / 'dev_generations.json').read_text())), 2)
            for call in tok.apply_chat_template.call_args_list:
                self.assertEqual([m['role'] for m in call.args[0]], ['system', 'user'])
                self.assertFalse(call.kwargs['enable_thinking'])
            validate.side_effect = None
            runner._generate({'max_new_tokens': 2}, Path(tmp), 'joint', [row('dev-a')], rooms)
            self.assertEqual(model.generate.call_args.kwargs['max_new_tokens'], 2)
            self.assertTrue(json.loads((Path(tmp) / 'dev_generations.json').read_text())[0]['reached_token_cap'])

    def test_worker_checks_one_gpu_and_preserves_failure(self):
        torch = Mock()
        torch.cuda.device_count.return_value = 2
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'torch': torch}), \
                self.assertRaisesRegex(RuntimeError, 'one visible GPU'):
            runner._arm_worker({'out': tmp}, 'joint', [], [], {})
            self.fail('worker should have stopped')

    def test_local_only_loaders(self):
        auto_model, auto_tok = Mock(), Mock()
        auto_tok.from_pretrained.return_value.pad_token_id = None
        modules = {'torch': types.SimpleNamespace(bfloat16='bf16'),
                   'transformers': types.SimpleNamespace(AutoTokenizer=auto_tok, AutoModelForCausalLM=auto_model)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            h.write_json(Path(tmp) / 'run_receipt.json', {'base': {'files': {}}})
            runner._tokenizer('/local')
            runner._load_model({'base': '/local', 'out': tmp})
        for loader in (auto_tok, auto_model):
            self.assertTrue(loader.from_pretrained.call_args.kwargs['local_files_only'])
            self.assertFalse(loader.from_pretrained.call_args.kwargs['trust_remote_code'])

    def test_training_worker_saves_evidence_without_in_process_reload_or_release(self):
        torch = Mock()
        torch.cuda.device_count.return_value = 1
        torch.cuda.is_bf16_supported.return_value = True
        torch.cuda.get_device_name.return_value = 'fake'
        torch.cuda.get_device_properties.return_value.total_memory = 1
        for name in ('max_memory_allocated', 'max_memory_reserved', 'memory_allocated'):
            getattr(torch.cuda, name).return_value = 0
        events = []
        def step(name, result):
            def call(*args):
                events.append(name)
                return result
            return call
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'torch': torch}), \
                patch.object(runner, '_tokenizer'), \
                patch.object(runner, '_train', side_effect=step('train', {'finite_loss': True})), \
                patch.object(runner, '_release_gpu', side_effect=step('release', None)), \
                patch.object(runner, '_adapter_inventory', side_effect=step('inventory', {})), \
                patch.object(runner, '_generate', side_effect=step('reload_dev', {'clean_load': True, 'dev_count': 2})):
            runner._arm_worker({'out': tmp, 'base': '/local'}, 'joint', [], [], {})
            self.assertEqual(events, ['train', 'inventory'])
            result = json.loads((Path(tmp) / 'joint/training_result.json').read_text())
            self.assertEqual(result['status'], 'TRAINING_SAVED')
            self.assertNotIn('gpu_allocated_after_release', result)
            self.assertFalse((Path(tmp) / 'joint/result.json').exists())
            arm_dir = Path(tmp) / 'joint'
            h.write_json(arm_dir / 'SHA256.json', h.artifact_manifest(arm_dir))
            runner._probe_worker({'out': tmp, 'base': '/local'}, 'joint', [], {})
            self.assertEqual(events, ['train', 'inventory', 'reload_dev', 'inventory'])
            result = json.loads((arm_dir / 'result.json').read_text())
            self.assertEqual(result['status'], 'ENGINEERING_COMPLETE')
            self.assertIn('training_gpu_peak_allocated_bytes', result)
            self.assertIn('probe_gpu_peak_allocated_bytes', result)
            self.assertNotIn('gpu_allocated_after_release', result)
            (arm_dir / 'training_result.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash'):
                runner._probe_worker({'out': tmp}, 'joint', [], {})
            self.assertEqual(json.loads((arm_dir / 'failure.json').read_text())['phase'], 'fresh_base_reload_and_dev')


if __name__ == '__main__':
    unittest.main()
