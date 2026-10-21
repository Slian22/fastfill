"""CPU/stdlib contracts for the paired SFT driver; no ML or geometry imports."""
import copy
import hashlib
import json
import math
from pathlib import Path

ARMS = ('joint', 'placement')
SPLITS = ('train', 'dev')
PILOT_TARGET_POLICY = 'identical_poses_v1'
MAIN_TARGET_POLICY = 'fixed_estimated_size_pose_repaired_v1'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
BASE_HASHES = {
    'config.json': 'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30',
    'model.safetensors.index.json': 'f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc',
    'model-00001-of-00005.safetensors': '31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f',
    'model-00002-of-00005.safetensors': '5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282',
    'model-00003-of-00005.safetensors': 'c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836',
    'model-00004-of-00005.safetensors': 'b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a',
    'model-00005-of-00005.safetensors': '20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Atomic evidence replacement; no NaN receipts or partly written JSON."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def strict_json(text):
    def reject(_):
        raise ValueError('nonfinite JSON number')
    return json.loads(text, object_pairs_hook=_unique, parse_constant=reject)


def _inside(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('manifest path must stay inside root')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('manifest symlink escapes root')
    return resolved


def _expected_hash(value):
    digest = value.get('sha256') if isinstance(value, dict) else value
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid manifest SHA256')
    return digest


def artifact_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'SHA256.json' and not p.name.endswith('.tmp')}


def verify_artifacts(root, manifest):
    for name, expected in manifest.items():
        path = _inside(Path(root), name)
        if not path.is_file() or sha256(path) != _expected_hash(expected):
            raise ValueError(f'artifact hash mismatch: {name}')


def _rows(root, name, expected):
    payload = _inside(root, name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != _expected_hash(expected):
        raise ValueError(f'data hash mismatch: {name}')
    rows = [strict_json(line) for line in payload.decode('utf-8').splitlines()]
    if not rows:
        raise ValueError(f'empty split: {name}')
    _by_uid(rows)
    if isinstance(expected, dict) and 'lines' in expected and len(rows) != expected['lines']:
        raise ValueError(f'manifest count mismatch: {name}')
    return rows


def _by_uid(rows):
    if any(not isinstance(r, dict) or not isinstance(r.get('uid'), str) or not r['uid'] for r in rows):
        raise ValueError('each row needs a nonempty UID')
    result = {r['uid']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate UID')
    return result


def request(record):
    messages = record.get('messages')
    if (not isinstance(messages, list) or len(messages) != 3 or
            [m.get('role') for m in messages] != ['system', 'user', 'assistant'] or
            any(not isinstance(m.get('content'), str) or not m['content'] for m in messages)):
        raise ValueError('expected nonempty system/user/assistant messages')
    value = strict_json(messages[1]['content'])
    if not isinstance(value, dict) or not isinstance(value.get('objects'), list) or not value['objects']:
        raise ValueError('request requires nonempty objects')
    fixed = value.get('fixed', [])
    if not isinstance(fixed, list):
        raise ValueError('request fixed must be a list')
    if any(not isinstance(obj, dict) for obj in value['objects'] + fixed):
        raise ValueError('request objects must be records')
    ids = [obj.get('id') for obj in value['objects'] + fixed]
    if any(not isinstance(oid, str) or not oid for oid in ids) or len(set(ids)) != len(ids):
        raise ValueError('request IDs must be unique strings')
    return value


def _check_pair(joint, placement, split, target_policy=PILOT_TARGET_POLICY):
    if target_policy not in (PILOT_TARGET_POLICY, MAIN_TARGET_POLICY):
        raise ValueError('unsupported target policy')
    for record in (joint, placement):
        default_policy = target_policy if target_policy == PILOT_TARGET_POLICY else None
        if record.get('target_policy', default_policy) != target_policy:
            raise ValueError('row target policy mismatch')
        if target_policy == MAIN_TARGET_POLICY and split == 'train' and record.get('target_qualified') is not True:
            raise ValueError('main train target must be qualified')
    for field in ('source', 'group', 'split'):
        if joint.get(field) != placement.get(field):
            raise ValueError(f'paired metadata mismatch: {field}')
    if joint.get('split', split) != split:
        raise ValueError('row split mismatch')
    j, p = request(joint), request(placement)
    if any('size' in obj for obj in j['objects']):
        raise ValueError('joint input contains movable size')
    for arm, req in zip(ARMS, (j, p)):
        fields = {'id', 'category'} | ({'size'} if arm == 'placement' else set())
        for obj in req['objects']:
            if set(obj) != fields:
                raise ValueError(f'invalid {arm} movable input fields')
            if not isinstance(obj['category'], str) or not obj['category'].strip():
                raise ValueError('movable input category must be a nonempty string')
    stripped = {**p, 'objects': [{k: v for k, v in obj.items() if k != 'size'} for obj in p['objects']]}
    if j != stripped:
        raise ValueError('paired request mismatch')
    for arm, record in zip(ARMS, (joint, placement)):
        if not parse_prediction(record['messages'][-1]['content'], record, arm)['valid']:
            raise ValueError(f'invalid {arm} training target or request')
    jt = strict_json(joint['messages'][-1]['content'])
    pt = strict_json(placement['messages'][-1]['content'])
    ignored = {'size'} if target_policy == PILOT_TARGET_POLICY else {'size', 'pos', 'yaw'}
    left = [{k: v for k, v in obj.items() if k not in ignored} for obj in jt['placements']]
    right = [{k: v for k, v in obj.items() if k not in ignored} for obj in pt['placements']]
    if left != right:
        raise ValueError('paired placement targets mismatch')


def qualification_hashes(turns, arm):
    """Bind qualification to serialized prompts/targets and displayed P sizes."""
    def digest(text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    def canonical(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    evidence = {'request_sha256': digest(canonical(turns[:-1])),
                'target_sha256': digest(turns[-1]['content'])}
    if arm == 'placement':
        sizes = {o['id']: o['size'] for o in strict_json(turns[1]['content'])['objects']}
        return {**evidence, 'estimated_sizes_sha256': digest(canonical(sizes))}
    return evidence


def _main_admission(root, files, pairs):
    name = 'common-admission.jsonl'
    if name not in files:
        raise ValueError('main target policy requires hashed common-admission ledger')
    ledger = _rows(root, name, files[name])
    accepted = {r['uid']: r for r in ledger if r.get('accepted') is True}
    indexed = {arm: _by_uid(pairs['train'][arm]) for arm in ARMS}
    if set(accepted) != set(indexed['joint']):
        raise ValueError('common-admission accepted UID coverage mismatch')
    count_fields = {'oob', 'support_fail', 'ceiling', 'fixed_blocking', 'main', 'severe'}
    for uid, entry in accepted.items():
        if entry.get('eligible') is not True or any(
                not isinstance(entry.get(arm), dict) or entry[arm].get('qualified') is not True for arm in ARMS):
            raise ValueError('common-admission requires both targets qualified')
        for arm in ARMS:
            evidence = entry[arm]
            counts = evidence.get('counts')
            if (not isinstance(counts, dict) or set(counts) != count_fields | {'valid'} or
                    counts['valid'] is not True or evidence.get('reasons') != [] or
                    any(type(counts[k]) is not int or counts[k] != 0 for k in count_fields)):
                raise ValueError(f'qualified target requires clean counts: {uid}: {arm}')
            expected = qualification_hashes(indexed[arm][uid]['messages'], arm)
            for field, actual in expected.items():
                if evidence.get(field) != actual:
                    raise ValueError(f'qualified {field} mismatch: {uid}: {arm}')


def load_pair(root):
    """Verify every manifest file, exact paired IDs and shared input/target context."""
    root = Path(root)
    payload = (root / 'MANIFEST.json').read_bytes()
    manifest = strict_json(payload)
    policy = manifest.get('target_policy', PILOT_TARGET_POLICY)
    if policy not in (PILOT_TARGET_POLICY, MAIN_TARGET_POLICY):
        raise ValueError('unsupported manifest target policy')
    files = manifest.get('files', {})
    required = {f'{a}/{s}.jsonl' for a in ARMS for s in SPLITS}
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError('manifest missing required paired files')
    verify_artifacts(root, files)
    pairs = {s: {a: _rows(root, f'{a}/{s}.jsonl', files[f'{a}/{s}.jsonl']) for a in ARMS} for s in SPLITS}
    for split, arms in pairs.items():
        j, p = (_by_uid(arms[a]) for a in ARMS)
        if j.keys() != p.keys():
            raise ValueError('paired UID mismatch')
        if manifest.get('counts', {}).get(split, len(j)) != len(j):
            raise ValueError('manifest split count mismatch')
        for uid in j:
            _check_pair(j[uid], p[uid], split, policy)
    if policy == MAIN_TARGET_POLICY:
        _main_admission(root, files, pairs)
    for field in ('uid', 'group'):
        groups = [{r[field] for r in pairs[s]['joint'] if r.get(field) is not None} for s in SPLITS]
        if groups[0] & groups[1]:
            raise ValueError(f'cross-split {field} overlap')
    digest = hashlib.sha256(payload).hexdigest()
    if sha256(root / 'MANIFEST.json') != digest:
        raise ValueError('manifest hash changed while reading')
    return pairs, {'manifest_sha256': digest, 'files': {n: _expected_hash(v) for n, v in files.items()},
                   'counts': {s: len(pairs[s]['joint']) for s in SPLITS}, 'target_policy': policy}


def encode_messages(tok, messages, eos_token_id=None):
    """Same label contract as fastfill.train.encode, without its eager ML imports.

    No truncation: common_admission decides whether BOTH arms retain a UID.
    """
    if (len(messages) != 3 or [m.get('role') for m in messages] != ['system', 'user', 'assistant'] or
            any(not isinstance(m.get('content'), str) or not m['content'] for m in messages)):
        raise ValueError('expected nonempty system/user/assistant messages')
    if eos_token_id is None:
        eos_token_id = tok.get_vocab().get('<|im_end|>')
    if eos_token_id is None:
        raise ValueError('Qwen3 tokenizer missing im_end EOS')
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p, a = tok([prompt, messages[-1]['content'] + '<|im_end|>'], add_special_tokens=False)['input_ids']
    if not p or not a or a[-1] != eos_token_id:
        raise ValueError('empty prompt/answer or unsupervised EOS')
    return {'input_ids': p + a, 'attention_mask': [1] * (len(p) + len(a)), 'labels': [-100] * len(p) + a}


class EmptyAdmission(ValueError):
    def __init__(self, report):
        super().__init__('empty common admission')
        self.report = report


def common_admission(tok, arms, max_len):
    """Sort paired UIDs once; never re-filter parent-selected flags or truncate."""
    indexed = {a: _by_uid(arms[a]) for a in ARMS}
    if indexed['joint'].keys() != indexed['placement'].keys():
        raise ValueError('paired UID mismatch')
    samples, accepted, dropped, lengths = {a: [] for a in ARMS}, [], [], []
    # Qwen's full vocabulary has >150k entries. Materializing it twice per row
    # dominated the real CPU preflight; its immutable EOS ID is shared per split.
    eos_token_id = tok.get_vocab().get('<|im_end|>')
    if eos_token_id is None:
        raise ValueError('Qwen3 tokenizer missing im_end EOS')
    for uid in sorted(indexed['joint']):
        encoded = {a: encode_messages(tok, indexed[a][uid]['messages'], eos_token_id) for a in ARMS}
        sizes = {a: len(encoded[a]['input_ids']) for a in ARMS}
        info = {'uid': uid, 'lengths': sizes,
                'answer_tokens': {a: sum(v != -100 for v in encoded[a]['labels']) for a in ARMS}}
        lengths.append(info)
        over = [a for a in ARMS if sizes[a] > max_len]
        if over:
            dropped.append({**info, 'overlength_arms': over})
        else:
            accepted.append(uid)
            for arm in ARMS:
                samples[arm].append(encoded[arm])
    report = {'accepted_uids': accepted, 'dropped': dropped, 'lengths': lengths,
              'max_len': max_len, 'input_count': len(lengths), 'accepted_count': len(accepted),
              'tokens': {a: {'input': sum(len(r['input_ids']) for r in samples[a]),
                             'supervised': sum(sum(v != -100 for v in r['labels']) for r in samples[a])}
                        for a in ARMS}}
    if not accepted:
        raise EmptyAdmission(report)
    return samples, report


def _number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not -1000 < value < 1000:
        raise ValueError('geometry must be finite numbers with magnitude < 1000')
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError('size must be positive and geometry finite')


def _vector(value, positive=False):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError('geometry vector must have three numbers')
    for number in value:
        _number(number, positive)


def _validate_output(value, req, arm):
    if not isinstance(value, dict) or set(value) != {'placements'} or not isinstance(value['placements'], list):
        raise ValueError('output must contain only placements')
    required = {'id', 'pos', 'yaw'} | ({'size'} if arm == 'joint' else set())
    objects = value['placements']
    for obj in objects:
        if not isinstance(obj, dict) or not required <= obj.keys() or obj.keys() - required - {'on'}:
            raise ValueError('invalid placement fields')
        _vector(obj['pos'])
        _number(obj['yaw'])
        if not isinstance(obj['yaw'], int) or not 0 <= obj['yaw'] < 360:
            raise ValueError('yaw must be integer degrees in [0,360)')
        if arm == 'joint':
            _vector(obj['size'], positive=True)
    if [o['id'] for o in objects] != [o['id'] for o in req['objects']]:
        raise ValueError('missing/extra/duplicate IDs or wrong order')
    parents = {o['id']: None for o in req.get('fixed', [])}
    parents.update({o['id']: o.get('on') for o in objects})
    for obj in objects:
        if 'on' in obj and (not isinstance(obj['on'], str) or obj['on'] not in parents):
            raise ValueError('invalid support ID')
        seen, current = set(), obj['id']
        while current is not None:
            if current in seen:
                raise ValueError('support cycle')
            seen.add(current)
            current = parents[current]
    if arm == 'placement':
        sizes = {o['id']: o['size'] for o in req['objects']}
        for size in sizes.values():
            _vector(size, positive=True)
        return [{**copy.deepcopy(o), 'size': copy.deepcopy(sizes[o['id']])} for o in objects]
    return copy.deepcopy(objects)


def parse_prediction(raw, record, arm):
    """Strict contract v1 schema; placement sizes come ONLY from its input prior.

    Kept stdlib-only so preflight/tests cannot import scene's geometry stack.
    No repair, markdown extraction, GT resize, or failed-example substitution.
    """
    try:
        if arm not in ARMS:
            raise ValueError('unknown arm')
        parsed = _validate_output(strict_json(raw), request(record), arm)
        return {'valid': True, 'error': None, 'placements': parsed}
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        # Do not embed exception content from parsers: it may quote private data.
        return {'valid': False, 'error': type(exc).__name__ + ': invalid contract output/request', 'placements': None}


def base_receipt(base):
    """Rehash the preflight-pinned generic Qwen3-8B, including all weight bytes."""
    base = Path(base)
    if not base.is_dir() or (base / 'adapter_config.json').exists():
        raise ValueError('base must be a local generic Qwen3 directory, not an adapter')
    verify_artifacts(base, BASE_HASHES)
    config = strict_json((base / 'config.json').read_text())
    index = strict_json((base / 'model.safetensors.index.json').read_text())
    shards = set(index.get('weight_map', {}).values())
    if config.get('model_type') != 'qwen3' or shards != {n for n in BASE_HASHES if n.endswith('.safetensors')}:
        raise ValueError('base config or shard inventory mismatch')
    names = set(BASE_HASHES) | {'tokenizer.json', 'tokenizer_config.json', 'generation_config.json'}
    names.update(p.name for p in base.iterdir() if p.is_file() and
                 (p.name.startswith(('tokenizer', 'special_tokens', 'chat_template', 'vocab', 'merges'))))
    receipts = {}
    for name in sorted(names):
        receipt = base / '.cache/huggingface/download' / (name + '.metadata')
        lines = receipt.read_text().splitlines()
        if len(lines) < 2 or lines[0] != REVISION:
            raise ValueError(f'base revision receipt mismatch: {name}')
        receipts[str(receipt.relative_to(base))] = sha256(receipt)
        if name in BASE_HASHES and name.endswith('.safetensors') and lines[1] != BASE_HASHES[name]:
            raise ValueError(f'weight receipt hash mismatch: {name}')
        if not name.endswith('.safetensors'):
            payload = (base / name).read_bytes()
            blob = b'blob ' + str(len(payload)).encode() + b'\0' + payload
            if lines[1] not in (hashlib.sha256(payload).hexdigest(), hashlib.sha1(blob).hexdigest()):
                raise ValueError(f'base file payload does not match download receipt: {name}')
    tree = base / '.cache/huggingface/trees' / (REVISION + '.json')
    receipts[str(tree.relative_to(base))] = sha256(tree)
    hashes = {n: BASE_HASHES[n] if n in BASE_HASHES else sha256(base / n) for n in sorted(names)}
    return {'repository': 'Qwen/Qwen3-8B', 'revision': REVISION, 'path': str(base.resolve()),
            'files': hashes, 'receipt_hashes': receipts, 'weight_hashes_freshly_verified': True,
            'max_position_embeddings': config['max_position_embeddings']}
