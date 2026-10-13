"""Freeze the amendment-005 model-free reference rows from local hash-verified inputs.

python -m fastfill.comparison.reference_rows --export reference-test-export.jsonl.gz \
    --joint eval-data-v1/joint/requests.jsonl --placement eval-data-v1/placement/requests.jsonl \
    --manifest eval-data-v1/MANIFEST.json --known-cases full-audit-data-known-cases.json \
    --out REFERENCE.json [--prior size-priors.json] [--workers 8]

Rooms are rebuilt exactly as eval_data._write_rows (bind_room, request component) and must hash
to the manifest's rooms.jsonl; both request files must match the manifest. Per room: the GT answer
scored as a J output by geometry.score (raw and <=10 cm boundary snap) plus rotated-footprint OOR;
the P request sizes against GT (geometry.dimensions; signed log footprint/volume bias). Object
prior level: G when the request size equals the constant fallback; with --prior (the frozen
training size-priors.json, sha-checked against the manifest) exactly P/C/G as size_prior._predict,
and every request size must be reproduced; without it other objects are N (pair or category,
unresolved). GT-pass = UIDs whose raw GT layout has geometry_success. Known-defect UIDs = the
known-cases entries on the test split. Written once; nothing is model output.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from fastfill import interface, scene, validate
from fastfill.comparison import contract, infer, llm_agent, size_prior
from fastfill.comparison import geometry as g
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.eval_data import SEED

# Amendment 005 section 3: real scans, SceneSmith and SpatialGen; every other source is high reuse.
LOW_REUSE = ('InternScenes_3rscan', 'InternScenes_arkit', 'InternScenes_mp3d', 'InternScenes_scannet',
             'MultiScan', 'Scan2CAD', 'SceneSmith', 'SpatialGen')
GT_KEYS = ('gt_success', 'gt_success_snapped', 'gt_oor', 'gt_oor_snapped', 'gt_oor_rotated', 'gt_oor_rotated_snapped')
PRIOR_KEYS = ('prior_size_iou', 'prior_size_log_mae', 'prior_bias_fp', 'prior_bias_vol')
CODES = {'P': 'category x room_type bucket', 'C': 'category bucket', 'G': 'global constant fallback',
         'N': 'pair or category (unresolved without the frozen size-priors.json)'}


def _lines(path):
    return [h.strict_json(line) for line in Path(path).read_text().splitlines()]


def load(export, joint, placement, manifest_path):
    manifest = h.strict_json(Path(manifest_path).read_text())
    files = manifest['files']
    for name, path in (('joint/requests.jsonl', joint), ('placement/requests.jsonl', placement)):
        if h.sha256(path) != files[name]:
            raise ValueError(f'{name} does not match the evaluation manifest')
    requests = {'joint': h._by_uid(_lines(joint)), 'placement': h._by_uid(_lines(placement))}
    with gzip.open(export, 'rt', encoding='utf-8') as stream:
        originals = sorted((h.strict_json(line) for line in stream), key=lambda r: r['uid'])
    rooms, digest = [], hashlib.sha256()
    for original in originals:
        bound, _ = contract.bind_room(original, seed=SEED)
        component = requests['joint'][original['uid']]['component']
        room = {**bound, 'group': component, 'component': component, 'split': 'test'}
        digest.update((json.dumps(room, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
        rooms.append(room)
    if digest.hexdigest() != files['rooms.jsonl'] or [r['uid'] for r in rooms] != manifest['expected_uids']:
        raise ValueError('rebuilt rooms do not reproduce the manifest rooms.jsonl')
    return rooms, requests, manifest


def _level(obj, room_type, prior):
    if prior is None:
        return 'G' if obj['size'] == r2.GLOBAL_FALLBACK else 'N'
    category, kind = size_prior._norm_cat(obj['category']), size_prior._norm_room_type(room_type) or ''
    pair, cat = prior['category_room_type'].get(category, {}).get(kind), prior['category'].get(category)
    if [max(.01, round(v, 2)) for v in size_prior._predict(category, kind, prior)] != obj['size']:
        raise ValueError('frozen prior does not reproduce a placement request size')
    return ('P' if pair and pair['count'] >= prior['min_count'] else
            'C' if cat and cat['count'] >= prior['min_count'] else 'G')


def _room(task):
    room, joint, placement, prior = task
    gt = g.score(room, joint['messages'], contract.messages(room, 'joint')[2]['content'], 'joint')
    request = h.strict_json(placement['messages'][1]['content'])
    if not gt['complete'] or [o['id'] for o in request['objects']] != [o['id'] for o in room['objects']]:
        raise ValueError(f"GT/request does not score for {room['uid']}")
    predicted = request['objects']
    iou, mae = g.dimensions(room['objects'], predicted)
    fp, vol = r2.size_bias(room['objects'], predicted)
    return {'uid': room['uid'], 'source': room['source'], 'room_type': room.get('room_type'),
            'gt_success': gt['geometry_success'], 'gt_success_snapped': gt['geometry_success_repaired'],
            'gt_oor': gt['oor'], 'gt_oor_snapped': gt['oor_repaired'],
            'gt_oor_rotated': r2.rotated_oor(gt['raw_layout']), 'gt_oor_rotated_snapped': r2.rotated_oor(gt['repaired_layout']),
            'prior_size_iou': iou, 'prior_size_log_mae': mae, 'prior_bias_fp': fp, 'prior_bias_vol': vol,
            'levels': ''.join(_level(o, request.get('room_type'), prior) for o in predicted),
            'object_errors': [g.dimensions([a], [b]) for a, b in zip(room['objects'], predicted)]}


def _scope(rows):
    def mean(key):
        return statistics.fmean(r[key] for r in rows) if rows else None
    levels = [lv for r in rows for lv in r['levels']]
    return {'requests': len(rows), 'gt': {k: mean(k) for k in GT_KEYS},
            'prior': {**{k: mean(k) for k in PRIOR_KEYS}, 'objects': len(levels),
                      'level_counts': dict(sorted(collections.Counter(levels).items())),
                      'global_fallback_object_rate': levels.count('G') / len(levels) if levels else None,
                      'rooms_with_global_fallback_rate': mean('has_global')}}


def build_reference(rooms, requests, data_manifest_sha256, known_defect_uids, prior=None, inputs=None, workers=1):
    tasks = [(room, requests['joint'][room['uid']], requests['placement'][room['uid']], prior) for room in rooms]
    if workers == 1:
        rows = [_room(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_room, tasks, chunksize=16))
    rows = [{**r, 'has_global': 'G' in r['levels']} for r in rows]
    uids = [r['uid'] for r in rows]
    if not set(known_defect_uids) <= set(uids):
        raise ValueError('known-defect UID outside the test inventory')
    gt_pass = sorted(r['uid'] for r in rows if r['gt_success'])
    by_level = collections.defaultdict(list)
    for r in rows:
        for level, errors in zip(r['levels'], r['object_errors']):
            by_level[level].append(errors)
    code = (g.__file__, validate.__file__, interface.__file__, contract.__file__, size_prior.__file__,
            scene.__file__, r2.__file__, __file__)
    return {'kind': r2.KIND, 'version': 1, 'amendment': '005', 'data_manifest_sha256': data_manifest_sha256,
            'inputs': inputs or {}, 'code_sha256': {Path(p).name: h.sha256(p) for p in code},
            'global_fallback_size': r2.GLOBAL_FALLBACK,
            'prior_level_method': 'fallback_triple_only' if prior is None else 'exact_frozen_prior',
            'prior_level_codes': CODES, 'low_reuse_sources': list(LOW_REUSE),
            'known_defect_uids': sorted(known_defect_uids),
            'gt_pass': {'count': len(gt_pass), 'sha256': llm_agent._digest(json.dumps(gt_pass)), 'uids': gt_pass},
            'scopes': {name: _scope(subset) for name, subset in r2.scopes(rows, LOW_REUSE).items()},
            'prior_level_objects': {level: {'objects': len(v), 'size_iou': statistics.mean(x[0] for x in v),
                                            'size_log_mae': statistics.mean(x[1] for x in v)}
                                    for level, v in sorted(by_level.items())},
            'prior_levels': {r['uid']: r['levels'] for r in rows}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('export', 'joint', 'placement', 'manifest', 'known-cases', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    rooms, requests, manifest = load(args.export, args.joint, args.placement, args.manifest)
    prior = None
    if args.prior:
        if h.sha256(args.prior) != manifest['input_consumed_sha256']['training']['size-priors.json']:
            raise ValueError('size-priors.json is not the frozen training prior')
        prior = h.strict_json(args.prior.read_text())['full_prior']
    cases = h.strict_json(args.known_cases.read_text())['cases']
    known = sorted({c['uid'] for c in cases if c.get('historical_split') == 'test'})
    inputs = {name: {'file': path.name, 'sha256': h.sha256(path)} for name, path in (
        ('export', args.export), ('joint_requests', args.joint), ('placement_requests', args.placement),
        ('eval_manifest', args.manifest), ('known_cases', args.known_cases),
        *((('size_priors', args.prior),) if args.prior else ()))}
    inputs['rooms_rebuilt_sha256'] = manifest['files']['rooms.jsonl']
    value = build_reference(rooms, requests, h.sha256(args.manifest), known, prior, inputs, args.workers)
    infer.write_once(args.out, value)
    print(json.dumps({'out': str(args.out), 'sha256': h.sha256(args.out), 'rooms': len(rooms),
                      'gt_pass': value['gt_pass']['count'], 'known_defect_uids': known}))


if __name__ == '__main__':
    main()
