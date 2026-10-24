"""Read-only verification of two existing Isambard training runs; run over SSH stdin.

No model loading, GPU use, job submission, installs, or remote output writes.
Large base-weight shards are stat-checked, not rehashed in this inspection.
"""
from pathlib import Path
import datetime
import hashlib
import json
import math
import re
import struct
import subprocess

ROOT = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill')
RUN = ROOT / 'runs/unknownsize-20261013'
CHECKS = []
FILES = []


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def check(label, actual, expected):
    CHECKS.append({'check': label, 'passed': actual == expected,
                   'actual': actual, 'expected': expected})


def capture(path):
    raw = path.read_bytes()
    FILES.append({'path': str(path.relative_to(ROOT)), 'bytes': len(raw),
                  'sha256': hashlib.sha256(raw).hexdigest(), 'text': raw.decode()})


def header(path):
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError('Unexpected safetensors header size')
        values = json.loads(stream.read(length))
    tensors = [v for k, v in values.items() if k != '__metadata__']
    return {'tensor_count': len(tensors),
            'parameters': sum(math.prod(v['shape']) for v in tensors),
            'dtypes': sorted({v['dtype'] for v in tensors})}


result = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'host': subprocess.check_output(['hostname'], text=True).strip(),
          'base_weight_shards_rehashed': False, 'arms': {}}
for arm, model, job, pool_name in [('a021', '27b', '7419267', 'pool'),
                                    ('a028', '27b-joint', '7461666', 'rows')]:
    run = RUN / arm
    model_dir = run / model
    train = model_dir / 'train'
    policy = load(model_dir / 'a021-policy.json')
    receipt = load(model_dir / 'run_receipt.json')
    guard = load(train / 'training_guard.json')
    completion = load(train / 'final/completion.json')
    manifest = load(train / 'run_manifest.json')
    for rel, expected in [
        ('run_receipt.json', policy['training']['run_receipt_sha256']),
        ('train/training_guard.json', policy['training']['guard_sha256']),
        ('train/final/completion.json', policy['training']['completion']['sha256']),
        ('rows/MANIFEST.json', policy['training_data']['rows']['manifest_sha256'])]:
        check(arm + ':' + rel, sha(model_dir / rel), expected)
    check(arm + ':pool_manifest', sha(run / pool_name / 'MANIFEST.json'),
          receipt['pool_manifest_sha256'])
    for name, expected in policy['adapter']['files'].items():
        check(arm + ':adapter/' + name, sha(train / 'final' / name), expected)
    pairs = Path(guard['pairs']['path'])
    check(arm + ':training_rows_sha256', sha(pairs), guard['pairs']['sha256'])
    check(arm + ':training_rows_size', pairs.stat().st_size, guard['pairs']['size_bytes'])
    with pairs.open() as stream:
        uids = [json.loads(line)['uid'] for line in stream if line.strip()]
    check(arm + ':training_row_count', len(uids), receipt['n'])
    check(arm + ':unique_training_rows', len(set(uids)), receipt['n'])
    check(arm + ':recipe_binding', guard['settings'],
          {**receipt['recipe'], **receipt['fixed'], 'n': receipt['n']})
    check(arm + ':trainer_recipe', {k: manifest['args'].get(k) for k in receipt['recipe']},
          receipt['recipe'])
    check(arm + ':code_identity_policy_receipt', policy['code_sha256'], receipt['code_sha256'])
    for name, identity in guard['code'].items():
        path = Path(name)
        check(arm + ':guarded_code/' + path.name, sha(path), identity['sha256'])
    snapshot = next(Path(name) for name in guard['code'] if name.endswith('/comparison/a021_train.py')).parents[1]
    for rel, expected in policy['code_sha256'].items():
        check(arm + ':snapshot_code/' + rel, sha(snapshot / rel), expected)
    base = Path(guard['model']['path'])
    skipped = []
    for name, identity in guard['model']['files'].items():
        path = base / name
        check(arm + ':base_size/' + name, path.stat().st_size, identity['size_bytes'])
        if name.endswith('.safetensors'):
            skipped.append(name)
        else:
            check(arm + ':base_metadata/' + name, sha(path), identity['sha256'])
    for file in [train / 'final/completion.json', run / pool_name / 'MANIFEST.json',
                 run / pool_name / 'summary.json', snapshot / 'v1_1/training_guard.py']:
        capture(file)
    log = ROOT / 'runs/isambard' / ('unknownsize-' + arm[1:] + '-train-' + job + '.out')
    capture(log)
    text = log.read_text()
    rows = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and 'step' in value and 'loss' in value:
            rows.append(value)
    steps = [r['step'] for r in rows]
    check(arm + ':logged_steps', steps, [1] + list(range(10, 10001, 10)))
    check(arm + ':finite_logged_loss', all(math.isfinite(r['loss']) for r in rows), True)
    resume_lines = [line for line in text.splitlines() if re.search(r'\bresum(?:e|ed|ing)\b', line, re.I)]
    result['arms'][arm] = {
        'job_id': job, 'snapshot_fastfill_dir': str(snapshot),
        'policy_sha256': sha(model_dir / 'a021-policy.json'),
        'completion': completion, 'adapter_header': header(train / 'final/adapter_model.safetensors'),
        'checkpoint_adapter_sha256': sha(train / 'ckpt_latest/adapter_model.safetensors'),
        'previous_checkpoint_adapter_sha256': sha(train / 'ckpt_latest.previous/adapter_model.safetensors'),
        'training_rows_sha256': guard['pairs']['sha256'], 'training_rows': len(uids),
        'logged_losses': len(rows), 'first_logged': rows[0], 'last_logged': rows[-1],
        'loss_min': min(r['loss'] for r in rows), 'loss_max': max(r['loss'] for r in rows),
        'resume_log_lines': resume_lines, 'base_weight_shards_stat_only': skipped,
    }
result['checks'] = CHECKS
result['files'] = FILES
result['passed'] = all(c['passed'] for c in CHECKS)
result['observed_end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(result))
