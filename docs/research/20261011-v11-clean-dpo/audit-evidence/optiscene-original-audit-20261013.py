"""Read-only reference audit v1.0. Python 3.13.1; no network or model execution.
Run: PYTHONDONTWRITEBYTECODE=1 python3 /tmp/optiscene-original-audit-20261013.py
All repository inputs are read only; JSON goes to stdout.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import sys, json, re, hashlib, subprocess, collections
from pathlib import Path
from importlib.metadata import version
from datasets import Dataset
ROOT = Path('/Users/slian/Desktop/3D/Worldedge/OptiScene')
VERSIONS = {'datasets': '4.1.1', 'numpy': '2.4.6', 'pyarrow': '21.0.0'}
assert sys.version_info[:3] == (3, 13, 1), sys.version
assert all(version(k) == v for k, v in VERSIONS.items())
sha = lambda b: hashlib.sha256(b).hexdigest()
original = ROOT / 'OptiScene_Original'
result = {'audit_version': '1.0', 'script_sha256': sha(Path(__file__).read_bytes()), 'python': sys.version, 'versions': VERSIONS,
          'commit': subprocess.check_output(['git', '-C', str(original), 'rev-parse', 'HEAD'], text=True).strip(), 'training': {}}
uid_sets = {}
for name in ['sft_prompts', 'dpo_prompts', 'dpo2_prompts']:
    p = original / 'dataset' / (name + '.json'); raw = p.read_bytes(); records = json.loads(raw)
    rows = [r if 'uid' in r else r['combined_prompts'][0] for r in records]
    ids = [r['uid'] for r in rows]; uid_sets[name] = set(ids)
    signatures = [sha((r['instruction'] + r['input'] + '\0' + r['output']).encode()) for r in rows]
    split = Dataset.from_dict({'uid': ids, 'input_chosen': signatures}).train_test_split(test_size=0.2, seed=42)
    pointer = subprocess.check_output(['git', '-C', str(original), 'show', 'HEAD:dataset/' + p.name], text=True)
    checks = {}
    for k in ['uid', 'input_chosen']:
        train = set(split['train'][k]); test = split['test'][k]
        checks[k] = {'shared_unique': len(train & set(test)), 'seen_eval_rows': sum(x in train for x in test)}
    result['training'][name] = {'sha256': sha(raw), 'lfs_match': 'oid sha256:' + sha(raw) in pointer,
        'root_dataset_identical': (ROOT / 'dataset' / p.name).read_bytes() == raw,
        'rows': len(rows), 'unique_uid': len(set(ids)), 'train': len(split['train']), 'eval': len(split['test']),
        'room_counts': dict(collections.Counter(re.search(r'\[Task Room Type\]\s*([^\[\n]+)', r['input']).group(1).strip() for r in rows)),
        'overlap': checks}
result['hf_prompts'] = {}
for name in ['prompts_all_scenes.json', 'prompts.json']:
    records = json.loads((ROOT / 'huggingface' / name).read_bytes()); ids = {r['uid'] for r in records}
    result['hf_prompts'][name] = {'rows': len(records), 'uid_unique': len(ids),
        'literal_uid_overlap_original': {k: len(v & ids) for k, v in uid_sets.items()}}
assets = ROOT / 'huggingface/3D-SynthPlace_Scenes_Assets'
scenes = list((assets / 'scenes').glob('*.json')); sources = collections.Counter(); rooms = collections.Counter()
all_ids = set(); absent = set(); affected = 0; instances = 0; below = collections.Counter()
for p in scenes:
    d = json.loads(p.read_bytes()); source = d.get('source', '<missing>'); sources[source] += 1; rooms[d['room']] += 1
    ids = [o['assetId'] for o in d['objects']]; all_ids.update(ids)
    missing = [i for i in ids if not (assets / 'assets_data' / i).is_dir()]
    absent.update(missing); affected += bool(missing); instances += len(missing)
    minimum = {'bedroom': 6, 'living room': 6, 'kitchen': 3, 'bathroom': 2}[d['room']]
    if len(ids) < minimum: below[source + ':' + d['room']] += 1
result['scenes'] = {'count': len(scenes), 'sources': dict(sources), 'rooms': dict(rooms),
    'asset_ids': len(all_ids), 'missing_asset_folders': len(absent), 'affected_scenes': affected,
    'missing_asset_instances': instances, 'below_paper_minimum': dict(below)}
p = assets / 'assets_data/3db8d396-f996-45ef-b4e0-95dd9ffdde2c/raw_model.obj'
low = [float('inf')] * 3; high = [float('-inf')] * 3
for line in p.open():
    if line.startswith('v '):
        xyz = list(map(float, line.split()[1:4])); low = [min(a,b) for a,b in zip(low,xyz)]; high = [max(a,b) for a,b in zip(high,xyz)]
result['bedroom_0_mesh_xyz_extents'] = [round(b-a, 4) for a,b in zip(low,high)]
result['bedroom_0_declared_bbox'] = json.loads((assets / 'scenes/bedroom-0.json').read_bytes())['objects'][0]['bbox']
print(json.dumps(result, indent=2))
