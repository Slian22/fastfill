"""Freeze every historical test UID with label-free paired generation requests.

No GT quality admission, asset lookup, rendering, model/API calls, or new blind
benchmark claim. The immutable training full_prior supplies all placement sizes.
A failed binding aborts publication with an EvaluationError.audit UID receipt.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile

from fastfill import scene
from fastfill.comparison import corpus as corpus_io
from fastfill.comparison.contract import DEFAULT_SEED, bind_room, messages
from fastfill.comparison.data import json_dump, sha256
from fastfill.comparison.size_prior import predict_sizes

SEED = DEFAULT_SEED
HELDOUT = ('SceneSmith', 'SpatialGen')
PAPER_STRATA = tuple((source, kind) for source, kinds in (
    ('IL3D_3dfront', ('bedroom', 'living room')),
    ('OptiScene_holodeck', ('bedroom', 'living room', 'kitchen', 'bathroom')),
) for kind in kinds)
CODE = ('comparison/eval_data.py', 'comparison/corpus.py', 'comparison/contract.py',
        'comparison/data.py', 'comparison/size_prior.py', 'scene.py')


class EvaluationError(ValueError):
    """Fatal row failure with a machine-readable counter/failure receipt."""
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = {**audit, 'failures': [dict(r) for r in audit['failures']]}


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _read(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'consumed hash mismatch: {path}')
    return json.loads(data)


def _snapshot(root):
    manifest_hash = sha256(root / 'MANIFEST.json')
    manifest = _read(root / 'MANIFEST.json', manifest_hash)
    files = manifest['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('empty file inventory')
    for name, digest in files.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or name == 'MANIFEST.json':
            raise ValueError(f'invalid manifest file: {name}')
        if any(p.is_symlink() for p in [root / path, *(root / path).parents] if p != root.parent):
            raise ValueError(f'symlink manifest file: {name}')
        if not isinstance(digest, str) or sha256(root / name) != digest:
            raise ValueError(f'input hash mismatch: {root / name}')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(files) | {'MANIFEST.json'}:
        raise ValueError(f'file inventory mismatch: {root}')
    return manifest, manifest_hash


def _identity(root, manifest):
    entries = _read(root / 'identity-manifest.json', manifest['files']['identity-manifest.json'])
    index, owners = {}, {}
    for entry in entries:
        uid, component, split = (entry[k] for k in ('uid', 'component', 'split'))
        if any(not isinstance(entry.get(k), str) or not entry[k] for k in
               ('uid', 'component', 'source', 'original_group')) or split not in ('train', 'dev', 'test'):
            raise ValueError('invalid corpus identity')
        if uid in index:
            raise ValueError(f'duplicate identity UID: {uid}')
        if component in owners and owners[component] != split:
            raise ValueError(f'cross-split component: {component}: {uid}')
        index[uid], owners[component] = entry, split
    counts = {s: sum(r['split'] == s for r in entries) for s in ('train', 'dev', 'test')}
    if counts != manifest['counts']:
        raise ValueError('corpus identity count mismatch')
    return index


def _fit_prior(training, training_manifest, corpus_manifest_hash, identity_hash, identities):
    if training_manifest.get('seed') != SEED:
        raise ValueError('training seed mismatch')
    if training_manifest.get('corpus_manifest_sha256') != corpus_manifest_hash:
        raise ValueError('training/corpus manifest hash mismatch')
    if training_manifest.get('corpus_identity_manifest_sha256') != identity_hash:
        raise ValueError('training/corpus identity hash mismatch')
    oof = _read(training / 'size-priors.json', training_manifest['files']['size-priors.json'])
    if oof['provenance']['seed'] != SEED:
        raise ValueError('prior seed mismatch')
    fitted = set(oof['fold_by_uid'])
    if not fitted or not fitted <= identities.keys():
        raise ValueError('unknown or empty fitted UID pool')
    held_components = {r['component'] for r in identities.values() if r['source'] in HELDOUT}
    test_components = {r['component'] for r in identities.values() if r['split'] == 'test'}
    for uid in sorted(fitted):
        entry = identities[uid]
        if entry['split'] != 'train' or entry['component'] in test_components:
            raise ValueError(f'fit/test leakage: {uid}')
        if entry['source'] in HELDOUT or entry['component'] in held_components:
            raise ValueError(f'source-heldout fit leakage: {uid}')
    if 'prior-fit-pool.json' in training_manifest['files']:
        pool = _read(training / 'prior-fit-pool.json', training_manifest['files']['prior-fit-pool.json'])
        if sorted(pool['uids']) != sorted(fitted):
            raise ValueError('fit pool UID mismatch')
    prior = oof['full_prior']
    expected_uids = sorted(_digest(uid) for uid in fitted)
    expected_components = sorted({_digest(identities[u]['component']) for u in fitted})
    for provenance in (prior['provenance'], oof['provenance']):
        if provenance.get('fitted_uid_hashes') != expected_uids:
            raise ValueError('full_prior fitted UID provenance mismatch')
        if provenance.get('component_hashes') != expected_components:
            raise ValueError('full_prior fitted component provenance mismatch')
        if provenance.get('room_count') != len(fitted) or provenance.get('component_count') != len(expected_components):
            raise ValueError('full_prior provenance count mismatch')
        expected_digest = _digest(json.dumps(expected_uids, ensure_ascii=False, separators=(',', ':')))
        if provenance.get('fitted_uid_sha256') != expected_digest:
            raise ValueError('full_prior fitted UID digest mismatch')
        if provenance.get('component_source') != 'caller_connected_components':
            raise ValueError('full_prior lacks connected component provenance')
    return prior, sorted(fitted)


def _test_rooms(root, manifest, identities, audit):
    rooms, seen = [], set()
    for _, room, digest in corpus_io._rows(root / 'prepared_rooms.jsonl',
                                          manifest['files']['prepared_rooms.jsonl']):
        uid = room.get('uid')
        if uid not in identities or uid in seen:
            raise ValueError(f'unknown/duplicate prepared UID: {uid}')
        seen.add(uid)
        entry = identities[uid]
        if digest != entry['prepared_sha256']:
            raise ValueError(f'prepared row hash mismatch: {uid}')
        if any(room.get(k) != entry[k] for k in ('source', 'split')) or room.get('group') != entry['original_group']:
            raise ValueError(f'prepared identity mismatch: {uid}')
        if entry['split'] == 'test':
            rooms.append(room)
    if seen != identities.keys():
        missing = sorted(identities.keys() - seen)
        raise ValueError(f'missing prepared UIDs: {missing}')
    audit['test_rows'] = len(rooms)
    return sorted(rooms, key=lambda r: r['uid'])


def select_gpt(rooms):
    """Exact normalized paper strata; independent hash ranks, <=30 per stratum.

    Living dining room stays separate, as in the existing paper metric inputs.
    All preselected UIDs are retained; asset eligibility is not known here.
    """
    strata, selected = {}, []
    for source, kind in PAPER_STRATA:
        key = source + '/' + kind
        uids = [r['uid'] for r in rooms if r['source'] == source
                and scene.norm_room_type(r.get('room_type')) == kind]
        ranked = sorted(uids, key=lambda uid: (_digest(SEED + '\0' + key + '\0' + uid), uid))
        chosen = ranked[:30]
        selected.extend(chosen)
        strata[key] = {'population': len(uids), 'selected_count': len(chosen), 'selected_uids': chosen,
                       'inclusion_probability': len(chosen) / len(uids) if uids else 0}
    return {'version': 1, 'seed': SEED, 'max_per_stratum': 30,
            'algorithm': 'sha256(seed\\0source/normalized_room_type\\0uid), uid tie-break',
            'selection_stage': 'before_size_and_layout_predictions', 'asset_eligibility': 'unknown',
            'visual_eligibility': 'not_assessed', 'selected_uids': selected, 'strata': strata}


def _write_line(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def _write_rows(dest, rooms, identities, prior, audit):
    sidecars = {}
    names = ('joint/requests.jsonl', 'placement/requests.jsonl', 'rooms.jsonl', 'size-estimates.jsonl')
    with ExitStack() as stack:
        files = {name: stack.enter_context((dest / name).open('w', encoding='utf-8')) for name in names}
        for original in rooms:
            uid = original['uid']
            try:
                bound, sidecar = bind_room(original, seed=SEED)
                component = identities[uid]['component']
                labels = {**bound, 'group': component, 'component': component, 'split': 'test'}
                # Pass only prediction features: even an accidental prior implementation
                # cannot consult movable GT sizes, positions, asset IDs or descriptions.
                features = {'room_type': bound.get('room_type'),
                            'objects': [{'id': o['id'], 'category': o.get('category')} for o in bound['objects']]}
                sizes = {oid: [max(.01, round(v, 2)) for v in values]
                         for oid, values in predict_sizes(features, prior).items()}
                metadata = {k: labels[k] for k in ('uid', 'source', 'group', 'component', 'split')}
                if labels.get('room_type') is not None:
                    metadata = {**metadata, 'room_type': labels['room_type']}
                requests = {arm: {**metadata, 'messages': messages(labels, arm,
                            **({'estimated_sizes': sizes} if arm == 'placement' else {}))[:2]}
                            for arm in ('joint', 'placement')}
                for arm, record in requests.items():
                    _write_line(files[arm + '/requests.jsonl'], record)
                _write_line(files['rooms.jsonl'], labels)
                _write_line(files['size-estimates.jsonl'], {'uid': uid, 'estimated_sizes': sizes})
                sidecars[uid] = {**sidecar, **identities[uid], 'group': component}
                audit['written_rows'] += 1
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                audit['failed_rows'] += 1
                audit['failures'].append({'uid': uid, 'stage': 'bind_predict_serialize',
                                          'error_type': type(exc).__name__, 'reason': str(exc)})
                raise EvaluationError(f'evaluation contract failed for UID {uid}: {exc}', audit) from exc
    json_dump(dest / 'inventory-sidecar.json', sidecars)


def build(corpus, training_data, out):
    """Validate frozen inputs, retain every test UID, atomically publish NOREPLACE."""
    corpus, training, out = Path(corpus), Path(training_data), Path(out)
    if os.path.lexists(out):
        raise FileExistsError(f'refusing existing output: {out}')
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {name: sha256(code_root / name) for name in CODE}
    corpus_manifest, corpus_hash = _snapshot(corpus)
    training_manifest, training_hash = _snapshot(training)
    if corpus_manifest.get('scope') != 'full_release_prepared_identity_closure':
        raise ValueError('expected full released corpus identity closure')
    identities = _identity(corpus, corpus_manifest)
    identity_hash = corpus_manifest['files']['identity-manifest.json']
    prior, fitted = _fit_prior(training, training_manifest, corpus_hash, identity_hash, identities)
    audit = {'test_rows': 0, 'written_rows': 0, 'failed_rows': 0, 'quality_filtered_rows': 0, 'failures': []}
    rooms = _test_rooms(corpus, corpus_manifest, identities, audit)
    selection = select_gpt(rooms)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temp:
        dest = Path(temp)
        for arm in ('joint', 'placement'):
            (dest / arm).mkdir()
        json_dump(dest / 'gpt-selection.json', selection)
        _write_rows(dest, rooms, identities, prior, audit)
        if audit['test_rows'] != audit['written_rows'] or audit['failed_rows']:
            raise EvaluationError('test row reconciliation failed', audit)
        result = {'version': 1, 'scope': 'frozen_historical_test_evaluation', 'seed': SEED,
                  'expected_uids': [r['uid'] for r in rooms],
                  'corpus_manifest_sha256': corpus_hash, 'corpus_identity_manifest_sha256': identity_hash,
                  'training_manifest_sha256': training_hash,
                  'training_corpus_identity_manifest_sha256': training_manifest['corpus_identity_manifest_sha256'],
                  'input_consumed_sha256': {'corpus': corpus_manifest['files'], 'training': training_manifest['files']},
                  'implementation_sha256': code_hashes, 'counts': {'test': len(rooms), 'gpt': len(selection['selected_uids'])},
                  'by_source': dict(sorted(Counter(r['source'] for r in rooms).items())),
                  'source_heldout': list(HELDOUT), 'prior_fit_count': len(fitted), 'audit': audit,
                  'new_blind_test_claim': False, 'geometry_policy': 'all_released_test_no_GT_quality_filter',
                  'size_policy': 'immutable_training_full_prior; centimeters; min0.01m',
                  'input_policy': 'system_user_only_labels_and_inventory_separate',
                  'files': {p.relative_to(dest).as_posix(): sha256(p) for p in sorted(dest.rglob('*')) if p.is_file()}}
        json_dump(dest / 'MANIFEST.json', result)
        for root, expected in ((corpus, corpus_hash), (training, training_hash)):
            _, actual = _snapshot(root)
            if actual != expected:
                raise ValueError(f'manifest hash changed during build: {root}')
        if any(sha256(code_root / name) != digest for name, digest in code_hashes.items()):
            raise ValueError('implementation hash changed during build')
        corpus_io._publish(dest, out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'training-data', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.corpus, args.training_data, args.out)
    except EvaluationError as exc:
        parser.exit(1, json.dumps({'error': str(exc), 'audit': exc.audit}) + '\n')
    print(json.dumps({'counts': result['counts'], 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
