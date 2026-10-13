"""CPU-only inference contracts; no model downloads or geometry dependencies."""
import copy
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load('infer_helpers', ROOT / 'fastfill/comparison/train_pair_helpers.py')
package = types.ModuleType('fastfill.comparison')
package.train_pair_helpers = h
with patch.dict(sys.modules, {'fastfill.comparison': package}):
    worker = load('infer', ROOT / 'fastfill/comparison/infer.py')


def row(uid='b', arm='joint'):
    obj = {'id': 'obj_0001', 'category': 'chair'}
    if arm == 'placement':
        obj = {**obj, 'size': [1, 2, 3]}
    return {'uid': uid, 'source': 'synthetic', 'group': 'room-' + uid,
            'messages': [{'role': 'system', 'content': 'layout'},
                         {'role': 'user', 'content': json.dumps({'objects': [obj]})}]}


def prediction(arm='joint'):
    obj = {'id': 'obj_0001', 'pos': [1, 2, 0], 'yaw': 0}
    return json.dumps({'placements': [{**obj, **({'size': [1, 2, 3]} if arm == 'joint' else {})}]})


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert [m['role'] for m in messages] == ['system', 'user']
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return messages[0]['content']

    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, truncation=False)
        return {'input_ids': list(range(len(text)))}

    def decode(self, ids, **kwargs):
        assert kwargs == dict(skip_special_tokens=False)
        return ''.join(chr(i) for i in ids)


def eval_fixture(root):
    files = {}
    for arm in h.ARMS:
        (root / arm).mkdir()
        path = root / arm / 'requests.jsonl'
        path.write_text(''.join(json.dumps(row(uid, arm)) + '\n' for uid in ['b', 'a']))
        files[f'{arm}/requests.jsonl'] = {'sha256': h.sha256(path), 'lines': 2}
    for name in ('rooms.jsonl', 'size-estimates.jsonl'):
        (root / name).write_bytes(b'NOT JSON: private sidecar must never be parsed\n')
        files[name] = h.sha256(root / name)
    h.write_json(root / 'MANIFEST.json', {'expected_uids': ['b', 'a'], 'files': files})


def training_fixture(root):
    base = root / 'base'
    base.mkdir()
    payloads = {'config.json': json.dumps({'model_type': 'qwen3', 'max_position_embeddings': 40960}),
                'model.safetensors.index.json': '{"weight_map":{"w":"model.safetensors"}}',
                'model.safetensors': 'weights', 'tokenizer.json': '{}',
                'tokenizer_config.json': '{}', 'generation_config.json': '{}'}
    meta = base / '.cache/huggingface/download'
    meta.mkdir(parents=True)
    tree = base / '.cache/huggingface/trees'
    tree.mkdir()
    (tree / (h.REVISION + '.json')).write_text('{}')
    for name, content in payloads.items():
        (base / name).write_text(content)
        (meta / (name + '.metadata')).write_text(h.REVISION + '\n' + h.sha256(base / name))
    pins = {n: h.sha256(base / n) for n in ('config.json', 'model.safetensors.index.json', 'model.safetensors')}
    run = root / 'train'
    adapter = run / 'joint/adapter'
    adapter.mkdir(parents=True)
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
        (adapter / name).write_text('{}')
    h.write_json(run / 'joint/result.json', {'status': 'TRAINING_COMPLETE', 'arm': 'joint',
        'finite_loss': True, 'clean_load': True, 'global_step': 4})
    h.write_json(run / 'status.json', {'status': 'TRAINING_COMPLETE', 'training_executed': True,
                                      'completed_arms': ['joint']})
    with patch.object(h, 'BASE_HASHES', pins):
        h.write_json(run / 'run_receipt.json', {'base': h.base_receipt(base),
            'data': {'manifest_sha256': 'a' * 64, 'files': {'joint/train.jsonl': 'b' * 64}}})
    rehash_training(run)
    args = worker.parse_args(['--data', str(root / 'eval'), '--base', str(base), '--adapter', str(adapter),
                              '--training-run', str(run), '--arm', 'joint', '--out', str(root / 'out')])
    return args, pins


def rehash_training(run):
    h.write_json(run / 'joint/SHA256.json', h.artifact_manifest(run / 'joint'))
    h.write_json(run / 'SHA256.json', h.artifact_manifest(run))


class InferenceTests(unittest.TestCase):
    def test_requests_no_assistant_labels_or_mutation(self):
        original = row()
        saved = copy.deepcopy(original)
        self.assertEqual(worker.parse_prediction(prediction(), original, 'joint')['valid'], True)
        self.assertEqual(original, saved)
        for invalid in [dict(original, labels={}), dict(original, source=''),
                        dict(original, messages=original['messages'] + [{'role': 'assistant', 'content': '{}'}])]:
            with self.assertRaises(ValueError):
                worker.validate_request(invalid, 'joint')
        leak = copy.deepcopy(original)
        leak['messages'][1]['content'] = json.dumps({'objects': [{'id': 'obj_0001', 'category': 'chair', 'pos': [1, 2, 3]}]})
        with self.assertRaises(ValueError):
            worker.validate_request(leak, 'joint')

    def test_schema_strict_and_placement_sizes_from_request(self):
        self.assertTrue(worker.parse_prediction(prediction('placement'), row(arm='placement'), 'placement')['valid'])
        self.assertFalse(worker.parse_prediction(prediction(), row(arm='placement'), 'placement')['valid'])
        for raw in ['```json\n' + prediction() + '\n```', '{"placements":[],"placements":[]}',
                    prediction().replace('"yaw": 0', '"yaw": 0.5'), prediction().replace('obj_0001', 'wrong')]:
            self.assertFalse(worker.parse_prediction(raw, row(), 'joint')['valid'])

    def test_manifest_order_opaque_sidecars_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_fixture(root)
            rows, receipt = worker.load_eval(root, 'joint')
            self.assertEqual([r['uid'] for r in rows], ['b', 'a'])
            self.assertEqual(receipt['expected_uids'], ['b', 'a'])
            (root / 'rooms.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                worker.load_eval(root, 'joint')

    def test_manifest_reordered_duplicate_missing_uids_rejected(self):
        for uids in [['a', 'b'], ['b', 'b'], ['b'], []]:
            with self.subTest(uids=uids), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                eval_fixture(root)
                manifest = h.strict_json((root / 'MANIFEST.json').read_text())
                h.write_json(root / 'MANIFEST.json', {**manifest, 'expected_uids': uids})
                with self.assertRaises(ValueError):
                    worker.load_eval(root, 'joint')

    def test_eos_split_counts_only_generated_through_first_eos(self):
        tok = Tokenizer()
        raw, info = worker.decode_output(tok, [0, 9, 65, 999, 0, 0], 2, 999, 4)
        self.assertEqual(raw, 'A')
        self.assertEqual(info, {'generated_tokens': 2, 'reached_token_cap': False, 'stopped_on_eos': True})
        raw, info = worker.decode_output(tok, [9, 65, 66], 1, 999, 2)
        self.assertEqual(raw, 'AB')
        self.assertTrue(info['reached_token_cap'])
        self.assertFalse(info['stopped_on_eos'])
        self.assertTrue(worker.decode_output(tok, [65, 999], 0, 999, 2)[1]['reached_token_cap'])

    def test_batch_overlength_schema_failure_and_no_retry(self):
        records = [row('b'), row('a'), {**row('long'), 'messages': [
            {'role': 'system', 'content': 'x' * 30}, row()['messages'][1]]}]
        calls = []
        def generate(ids):
            calls.append(ids)
            width = max(map(len, ids))
            return [[0] * width + list(map(ord, prediction())) + [999],
                    [0] * width + list(map(ord, 'bad')) + [999]], width
        results = worker.process_batch(records, 'joint', Tokenizer(), generate, 999, 20, 1000, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual([r['uid'] for r in results], ['b', 'a', 'long'])
        self.assertTrue(results[0]['schema']['valid'])
        self.assertIsNone(results[1]['error'])
        self.assertFalse(results[1]['schema']['valid'])
        self.assertEqual(results[2]['error'], 'INPUT_TOO_LONG')
        def fail(ids):
            calls.append(ids)
            raise RuntimeError('OOM')
        failed = worker.process_batch(records[:2], 'joint', Tokenizer(), fail, 999, 20, 1000, 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(r['error'] == 'BATCH_GENERATION_FAILED:RuntimeError' for r in failed))
        self.assertTrue(all(not r['generated'] for r in failed))

    def test_partial_accounting_and_zero_valid_completion(self):
        partial = [{'uid': 'b', 'generated': True, 'schema': {'valid': False}, 'error': None}]
        counts = worker.accounting(['b', 'a'], partial)
        self.assertEqual((counts['recorded_count'], counts['pending_count'], counts['generated_count'], counts['valid_count']), (1, 1, 1, 0))
        with self.assertRaises(ValueError):
            worker.accounting(['a', 'b'], partial)
        with self.assertRaises(ValueError):
            worker.accounting(['b'], partial * 2)

    def test_atomic_receipts_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            worker.write_once(path, {'a': 1})
            with self.assertRaises(FileExistsError):
                worker.write_once(path, {'a': 2})
            self.assertEqual(h.strict_json(path.read_text()), {'a': 1})
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['result.json'])

    def test_cli_requires_all_paths_and_positive_budgets(self):
        argv = ['--data', 'd', '--base', 'b', '--adapter', 'r/joint/adapter', '--training-run', 'r', '--arm', 'joint', '--out', 'o']
        args = worker.parse_args(argv)
        self.assertEqual((args.batch_size, args.max_new_tokens, args.max_input_tokens), (4, 8192, 8192))
        with self.assertRaises(SystemExit):
            worker.parse_args(argv + ['--batch-size', '0'])

    def test_training_identity_and_adapter_base_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, pins = training_fixture(Path(tmp))
            with patch.object(h, 'BASE_HASHES', pins):
                identity = worker.training_identity(args)
                self.assertEqual(identity['base']['revision'], h.REVISION)
                (args.adapter / 'adapter_model.safetensors').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    worker.training_identity(args)
                (args.adapter / 'adapter_model.safetensors').write_text('{}')
                (args.base / 'model.safetensors').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    worker.training_identity(args)

    def test_training_terminal_required_for_requested_arm(self):
        for change in [{'status': 'ENGINEERING_COMPLETE'}, {'finite_loss': False},
                       {'arm': 'placement'}, {'clean_load': False}, {'global_step': 0}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                args, pins = training_fixture(Path(tmp))
                path = args.training_run / 'joint/result.json'
                h.write_json(path, {**h.strict_json(path.read_text()), **change})
                rehash_training(args.training_run)
                with patch.object(h, 'BASE_HASHES', pins), self.assertRaisesRegex(ValueError, 'terminal'):
                    worker.training_identity(args)

    def test_training_identity_wrong_path_extra_artifacts_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, pins = training_fixture(Path(tmp))
            with patch.object(h, 'BASE_HASHES', pins):
                wrong = types.SimpleNamespace(**{**vars(args), 'adapter': args.base})
                with self.assertRaisesRegex(ValueError, 'belong'):
                    worker.training_identity(wrong)
                extra = args.adapter / 'unexpected.bin'
                extra.write_text('unhashed')
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    worker.training_identity(args)
                extra.unlink()
                huge = types.SimpleNamespace(**{**vars(args), 'max_new_tokens': 50000})
                with self.assertRaisesRegex(ValueError, 'context'):
                    worker.training_identity(huge)
                receipt = args.training_run / 'run_receipt.json'
                value = h.strict_json(receipt.read_text())
                value['base']['revision'] = 'wrong'
                h.write_json(receipt, value)
                rehash_training(args.training_run)
                with self.assertRaisesRegex(ValueError, 'base differs'):
                    worker.training_identity(args)

    def test_complete_zero_valid_publishes_ordered_records_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, pins = training_fixture(Path(tmp))
            args.data.mkdir()
            eval_fixture(args.data)
            def generate(ids):
                width = max(map(len, ids))
                return [[0] * width + [ord('!'), 999] for _ in ids], width
            with patch.object(h, 'BASE_HASHES', pins), patch.object(worker, '_runtime', return_value=(Tokenizer(), generate, 999)):
                result = worker.run(args)
                with self.assertRaises(FileExistsError):
                    worker.run(args)
            self.assertEqual(result['status'], 'INFERENCE_COMPLETE')
            self.assertEqual((result['generated_count'], result['valid_count'], result['pending_count']), (2, 0, 0))
            records = [h.strict_json(x) for x in (args.out / 'records.jsonl').read_text().splitlines()]
            self.assertEqual([r['uid'] for r in records], ['b', 'a'])
            self.assertTrue(all(r['error'] is None and r['schema']['error'] for r in records))
            manifest = h.strict_json((args.out / 'MANIFEST.json').read_text())
            h.verify_artifacts(args.out, manifest['files'])
            ledger = [h.strict_json(x) for x in (args.out / 'ledger.jsonl').read_text().splitlines()]
            for entry in ledger:
                h.verify_artifacts(args.out, {entry['receipt']: entry['sha256']})

    def test_interrupt_preserves_partial_accounting_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, pins = training_fixture(Path(tmp))
            args = types.SimpleNamespace(**{**vars(args), 'batch_size': 1})
            args.data.mkdir()
            eval_fixture(args.data)
            calls = []
            def generate(ids):
                calls.append(ids)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return [[0] * len(ids[0]) + [ord('!'), 999]], len(ids[0])
            with patch.object(h, 'BASE_HASHES', pins), patch.object(worker, '_runtime', return_value=(Tokenizer(), generate, 999)):
                with self.assertRaises(KeyboardInterrupt):
                    worker.run(args)
            status = h.strict_json((args.out / 'status.json').read_text())
            self.assertEqual((status['status'], status['recorded_count'], status['pending_count']), ('FAIL', 1, 1))
            self.assertTrue((args.out / 'receipts/00000000.json').is_file())
            self.assertFalse((args.out / 'records.jsonl').exists())
            self.assertFalse((args.out / 'result.json').exists())

    def test_tokenizer_and_cardinality_failures_are_receipts(self):
        class Broken(Tokenizer):
            def __call__(self, *args, **kwargs):
                raise ValueError('bad encoding')
        def unused(_):
            self.fail('generation must not run')
        result = worker.process_batch([row()], 'joint', Broken(), unused, 999, 20, 100, 0)[0]
        self.assertEqual(result['error'], 'TOKENIZATION_FAILED:ValueError')
        result = worker.process_batch([row()], 'joint', Tokenizer(), lambda _: ([], 6), 999, 20, 100, 0)[0]
        self.assertEqual(result['error'], 'BATCH_GENERATION_FAILED:ValueError')
        for sequence in [[], [1] * 4]:
            with self.assertRaises(ValueError):
                worker.decode_output(Tokenizer(), sequence, 0, 999, 3)

    def test_hf_runtime_clean_local_bf16_sdpa_lora_and_left_padding(self):
        class Batch(dict):
            def to(self, device):
                self.device = device
                return self
        batch = Batch(input_ids=types.SimpleNamespace(shape=(2, 6)), attention_mask='mask')
        tok = Mock(pad_token_id=None)
        tok.get_vocab.return_value = {'<|im_end|>': 999}
        tok.pad.return_value = batch
        base, model = Mock(), Mock()
        model.generate.return_value.cpu.return_value.tolist.return_value = [[0] * 6 + [65, 999]] * 2
        torch = types.SimpleNamespace(bfloat16='bf16', inference_mode=contextlib.nullcontext,
            cuda=Mock(device_count=Mock(return_value=1), is_bf16_supported=Mock(return_value=True)))
        transformers = types.SimpleNamespace(
            AutoTokenizer=Mock(from_pretrained=Mock(return_value=tok)),
            AutoModelForCausalLM=Mock(from_pretrained=Mock(return_value=base)),
            GenerationConfig=lambda **kwargs: types.SimpleNamespace(**kwargs))
        peft = types.SimpleNamespace(PeftModel=Mock(from_pretrained=Mock(return_value=model)))
        args = types.SimpleNamespace(adapter=Path('/local/adapter'), base=Path('/local/base'), max_new_tokens=16, context_budget=20)
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers, 'peft': peft}), patch.dict('os.environ', {'WORLD_SIZE': '1'}):
            returned_tok, generate, eos = worker._runtime(args)
            output, width = generate([[1, 2], [1, 2, 3, 4, 5, 6]])
            self.assertEqual((width, eos, returned_tok.padding_side, returned_tok.pad_token_id), (6, 999, 'left', 999))
            self.assertEqual(len(output), 2)
            self.assertEqual(batch.device, 'cuda:0')
            transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with('/local/base',
                local_files_only=True, trust_remote_code=False, dtype='bf16', attn_implementation='sdpa', device_map={'': 0})
            peft.PeftModel.from_pretrained.assert_called_once_with(base, '/local/adapter',
                local_files_only=True, is_trainable=False, autocast_adapter_dtype=False)
            model.eval.assert_called_once()
            config = model.generate.call_args.kwargs['generation_config']
            self.assertEqual(model.generate.call_args.kwargs['logits_to_keep'], 1)
            self.assertEqual((config.do_sample, config.num_beams, config.max_new_tokens, config.eos_token_id), (False, 1, 14, 999))
            self.assertEqual(tok.pad.call_args.args[0]['attention_mask'], [[1, 1], [1] * 6])
            torch.cuda.device_count.return_value = 2
            with self.assertRaisesRegex(RuntimeError, 'one process'):
                worker._runtime(args)
            torch.cuda.device_count.return_value = 1
            tok.get_vocab.return_value = {}
            with self.assertRaisesRegex(ValueError, 'missing im_end'):
                worker._runtime(args)

    def test_context_budget_counts_effective_generation_cap(self):
        tok = Mock()
        tok.apply_chat_template.return_value = 'abcdef'
        tok.return_value = {'input_ids': [1] * 6}
        tok.decode.return_value = 'invalid'
        result = worker.process_batch([row()], 'joint', tok,
            lambda ids: ([[0] * 6 + [65] * 14], 6), 999, 19, 1000, 0, context_budget=20)[0]
        self.assertTrue(result['reached_token_cap'])
        self.assertEqual(result['effective_max_new_tokens'], 14)
        self.assertIsNone(result['error'])

    def test_shared_context_plan_uses_both_inputs_not_labels(self):
        rows = [row(), dict(row(), uid='second')]
        tok = Mock()
        tok.return_value = {'input_ids': [1]}
        tok.side_effect = [{'input_ids': [1] * n} for n in (6, 7, 10, 12)]
        plan = worker.context_plan({'joint': rows, 'placement': rows}, tok, 2, 20, 19, 19)
        self.assertEqual(plan[0]['max_new_tokens'], 8)
        self.assertEqual(plan[0]['shared_padded_width'], 12)
        self.assertEqual(plan[0]['uids'], [r['uid'] for r in rows])
        with self.assertRaisesRegex(ValueError, 'paired'):
            worker.context_plan({'joint': rows, 'placement': rows[:1]}, tok, 2, 20, 19, 19)

    def test_fixed_request_and_invalid_manifest_or_training_hash_metadata(self):
        fixed = {'id': 'fixed_0001', 'category': 'door', 'size': [1, 1, 2], 'pos': [0, 0, 0], 'yaw': 0}
        record = row(arm='placement')
        req = h.strict_json(record['messages'][1]['content'])
        record['messages'][1]['content'] = json.dumps({**req, 'fixed': [fixed]})
        worker.validate_request(record, 'placement')
        record['messages'][1]['content'] = json.dumps({**req, 'fixed': [{**fixed, 'private': 'label'}]})
        with self.assertRaises(ValueError):
            worker.validate_request(record, 'placement')
        record['messages'][1]['content'] = json.dumps({**req, 'private': 'label'})
        with self.assertRaises(ValueError):
            worker.validate_request(record, 'placement')
        with tempfile.TemporaryDirectory() as tmp:
            args, pins = training_fixture(Path(tmp))
            args.data.mkdir()
            h.write_json(args.data / 'MANIFEST.json', {'files': {}})
            with self.assertRaisesRegex(ValueError, 'missing required'):
                worker.load_eval(args.data, 'joint')
            receipt = args.training_run / 'run_receipt.json'
            value = h.strict_json(receipt.read_text())
            h.write_json(receipt, {**value, 'data': {'files': {}}})
            rehash_training(args.training_run)
            with patch.object(h, 'BASE_HASHES', pins), self.assertRaisesRegex(ValueError, 'data hashes missing'):
                worker.training_identity(args)
            h.write_json(args.training_run / 'SHA256.json', {})
            with self.assertRaisesRegex(ValueError, 'lacks terminal'):
                worker.training_identity(args)


if __name__ == '__main__':
    unittest.main()
