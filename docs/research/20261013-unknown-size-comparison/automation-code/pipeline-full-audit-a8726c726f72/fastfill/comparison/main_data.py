"""Main paired SFT data from the complete, hash-verified historical corpus.

Freeze the train-only category/room-type logmedian prior with category/global
fallback before decoder admission. J retains its own GT; P receives fixed
estimated sizes and a bounded pose-only teacher. Only
jointly qualified serialized train targets are admitted. Dev is never repaired or
quality filtered. Test geometry is never used and no test prompts are written.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile

from fastfill.comparison.contract import DEFAULT_SEED, bind_room, messages, validate_output
from fastfill.comparison.corpus import _load_manifest, _publish, _rows
from fastfill.comparison.data import EXCLUDE_FLAGS, check_input_invariance, json_dump, sha256
from fastfill.comparison.pose_targets import repair
from fastfill.comparison.size_prior import build_oof_priors, predict_sizes
from fastfill.comparison.train_pair_helpers import qualification_hashes
from fastfill.validate import check

TARGET_POLICY = 'fixed_estimated_size_pose_repaired_v1'
HELDOUT = frozenset(('SceneSmith', 'SpatialGen'))
CODE = ('comparison/main_data.py', 'comparison/corpus.py', 'comparison/data.py',
        'comparison/contract.py', 'comparison/size_prior.py', 'comparison/pose_targets.py',
        'comparison/train_pair_helpers.py',
        'scene.py', 'validate.py', 'anchors.py', 'split.py')


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _identities(root, manifest):
    path = root / 'identity-manifest.json'
    expected = manifest['files'][path.name]
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected or sha256(path) != expected:
        raise ValueError('identity manifest hash mismatch')
    entries = json.loads(payload)
    index, owners, keys = {}, {}, {}
    for entry in entries:
        uid = entry['uid']
        if not isinstance(uid, str) or not uid or uid in index:
            raise ValueError('invalid or duplicate identity UID')
        split, component = entry['split'], entry['component']
        if split not in ('train', 'dev', 'test') or not isinstance(component, str) or not component:
            raise ValueError('invalid split/component identity')
        if not isinstance(entry.get('training_flags'), dict):
            raise ValueError('invalid historical training flags')
        if component in owners and owners[component] != split:
            raise ValueError('cross-split component in full corpus closure')
        owners[component] = split
        for field in ('original_group', 'content_key', 'layout_key'):
            key = entry.get(field)
            if key is not None:
                pair = field, key
                if pair in keys and keys[pair] != component:
                    raise ValueError('inconsistent full corpus component closure')
                keys[pair] = component
        index[uid] = entry
    counts = dict(Counter(e['split'] for e in entries))
    if any(counts.get(s, 0) != manifest['counts'][s] for s in ('train', 'dev', 'test')):
        raise ValueError('corpus identity count mismatch')
    return index


def _eligible(entry, heldout_components):
    reasons = []
    if entry['split'] != 'train':
        reasons.append('not_exact_train')
    if entry['source'] in HELDOUT:
        reasons.append('heldout_source')
    if entry['component'] in heldout_components:
        reasons.append('heldout_source_component')
    reasons += ['historical_' + k for k in sorted(EXCLUDE_FLAGS) if entry['training_flags'].get(k)]
    return reasons


def _rooms(root, manifest, identities):
    seen = set()
    for _, row, digest in _rows(root / 'prepared_rooms.jsonl', manifest['files']['prepared_rooms.jsonl']):
        uid = row['uid']
        if uid not in identities or uid in seen:
            raise ValueError('unresolved or duplicate prepared UID')
        seen.add(uid)
        entry = identities[uid]
        if digest != entry['prepared_sha256']:
            raise ValueError('prepared row hash mismatch: ' + uid)
        if any(row[k] != entry[k] for k in ('source', 'split')) or row['group'] != entry['original_group']:
            raise ValueError('prepared identity mismatch: ' + uid)
        yield row
    if seen != identities.keys():
        raise ValueError('missing prepared identities')


def _freeze_prior(root, manifest, identities, eligible):
    # Only minimal size observations survive this pass; test/dev geometry is unused.
    observations = []
    for room in _rooms(root, manifest, identities):
        if room['uid'] in eligible:
            observations.append({'uid': room['uid'], 'split': 'train', 'room_type': room.get('room_type'),
                'fixed': [None] * len(room.get('fixed', [])),
                'objects': [{k: o[k] for k in ('id', 'category', 'size')} for o in room['objects']]})
    components = {u: identities[u]['component'] for u in sorted(eligible)}
    oof = build_oof_priors(observations, components, seed=DEFAULT_SEED)
    def annotate(prior):
        return {**prior, 'conditioning': 'category_room_type', 'fallback': ['category', 'global']}
    oof = {**oof, 'full_prior': annotate(oof['full_prior']),
           'priors': {k: annotate(v) for k, v in oof['priors'].items()},
           'provenance': {**oof['provenance'], 'conditioning': 'category_room_type',
                          'fallback': ['category', 'global'], 'frozen_before_decoder_admission': True}}
    uids = sorted(eligible)
    pool = {'uids': uids, 'uid_hashes': sorted(map(_digest, uids)),
            'components': sorted(set(components.values())),
            'component_hashes': sorted({_digest(c) for c in components.values()}),
            'by_source': dict(sorted(Counter(identities[u]['source'] for u in uids).items())),
            'members': [{k: identities[u][k] for k in ('uid', 'source', 'component', 'prepared_sha256')}
                        for u in uids], 'frozen_before_decoder_admission': True}
    return oof, pool


def _serialized_room(room, turns, arm):
    user = json.loads(turns[1]['content'])
    target = validate_output(turns[-1]['content'], room, arm)['placements']
    supplied = {o['id']: o for o in user['objects']}
    fixed = {o['id']: o for o in user.get('fixed', [])}
    return {**room, 'boundary': user['boundary'], 'height': user.get('height'),
            'fixed': [{**o, **fixed[o['id']], 'yaw': math.radians(fixed[o['id']]['yaw'])}
                      for o in room.get('fixed', [])],
            'objects': [{**o, 'size': t['size'] if arm == 'joint' else supplied[o['id']]['size'],
                         'pos': t['pos'], 'yaw': math.radians(t['yaw']), 'parent': t.get('on')}
                        for o, t in zip(room['objects'], target)]}


def _qualification(room, arm, sizes=None):
    turns = None
    evidence = {'request_sha256': None, 'target_sha256': None,
                **({'estimated_sizes_sha256': None} if arm == 'placement' else {})}
    try:
        turns = messages(room, arm, estimated_sizes=sizes)
        evidence = qualification_hashes(turns, arm)
        audit = check(_serialized_room(room, turns, arm))
        counts = {'valid': bool(audit['valid']),
                  **{k: len(audit[k]) for k in ('oob', 'support_fail', 'ceiling', 'fixed_blocking')},
                  **{k: len(audit['collisions'][k]) for k in ('main', 'severe')}}
        reasons = [k for k, v in counts.items() if k != 'valid' and v]
        if not counts['valid']:
            reasons.append('invalid')
        return turns, {'qualified': not reasons, 'counts': counts, 'reasons': reasons, **evidence}
    except ValueError as exc:
        return turns, {'qualified': False, 'counts': None, 'reasons': [str(exc)], **evidence}


def _locked(room):
    return {**room, 'objects': [{k: v for k, v in o.items() if k not in ('pos', 'yaw', 'size')}
                               for o in room['objects']]}


def _candidate(task):
    room, entry, sizes, fold, limits = task
    try:
        bound, sidecar = bind_room(room, DEFAULT_SEED)
    except ValueError as exc:
        failure = _excluded(entry, ['binding_failed: ' + str(exc)])
        return room, None, None, None, {**failure, 'eligible': True, 'fold': fold}
    sizes = {sidecar['original_to_bound'][oid]: values for oid, values in sizes.items()}
    joint_turns, joint = _qualification(bound, 'joint')
    result = repair(bound, sizes, **limits)
    repaired = result['room']
    placement_turns, placement = _qualification(repaired, 'placement', sizes)
    frozen = _locked(bound) == _locked(repaired) and all(
        o['size'] == sizes.get(o['id']) for o in repaired['objects'])
    reasons = list(placement['reasons'])
    if result['status'] != 'ok':
        reasons.append('repair_rejected')
    if not frozen:
        reasons.append('inventory_size_support_or_context_changed')
    if joint_turns is not None and placement_turns is not None:
        ju, pu = (json.loads(t[1]['content']) for t in (joint_turns, placement_turns))
        stripped = {**pu, 'objects': [{k: v for k, v in o.items() if k != 'size'} for o in pu['objects']]}
        if ju != stripped:
            reasons.append('input_inventory_changed')
    placement = {**placement, 'qualified': placement['qualified'] and not reasons, 'reasons': reasons}
    accepted = joint['qualified'] and placement['qualified']
    ledger = {'uid': entry['uid'], 'source': entry['source'], 'component': entry['component'],
              'fold': fold, 'eligible': True, 'accepted': accepted, 'joint': joint, 'placement': placement,
              'reasons': ([] if accepted else ['paired_target_not_qualified']),
              'repair': result['diagnostics']}
    if accepted:
        check_input_invariance(bound, sizes)
    return bound, sidecar, joint_turns, placement_turns, ledger


def _ordered(tasks, workers):
    if workers == 1:
        for task in tasks:
            yield _candidate(task)
        return
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        pending = deque()
        for task in tasks:
            pending.append(executor.submit(_candidate, task))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _tasks(root, manifest, identities, eligible, oof, limits):
    for room in _rooms(root, manifest, identities):
        uid = room['uid']
        if uid in eligible:
            fold = oof['fold_by_uid'][uid]
            sizes = {k: [max(.01, round(v, 2)) for v in values]
                     for k, values in predict_sizes(room, oof['priors'][str(fold)]).items()}
            yield room, identities[uid], sizes, fold, limits


def _record(room, entry, turns, qualified):
    return {'uid': room['uid'], 'source': entry['source'], 'split': entry['split'],
            'group': entry['component'], 'flags': entry['training_flags'], 'messages': turns,
            'target_policy': TARGET_POLICY, 'target_qualified': qualified}


def _excluded(entry, reasons):
    failure = {'qualified': False, 'counts': None, 'reasons': reasons,
               'request_sha256': None, 'target_sha256': None, 'estimated_sizes_sha256': None}
    return {'uid': entry['uid'], 'source': entry['source'], 'component': entry['component'],
            'fold': None, 'eligible': False, 'accepted': False, 'reasons': reasons,
            'joint': failure, 'placement': failure}


def _write_data(dest, root, manifest, identities, eligible, excluded, oof, workers, limits, devlimit):
    accepted, source_counts = [], {}
    counts = Counter()
    with ExitStack() as stack:
        names = ['common-admission.jsonl', 'bound_rooms.jsonl', 'train_rooms.jsonl', 'dev_rooms.jsonl']
        names += [f'{a}/{s}.jsonl' for a in ('joint', 'placement') for s in ('train', 'dev')]
        streams = {n: stack.enter_context((dest / n).open('w', encoding='utf-8')) for n in names}
        sidecar = stack.enter_context((dest / 'inventory-sidecar.json').open('w', encoding='utf-8'))
        splitfile = stack.enter_context((dest / 'split-manifest.json').open('w', encoding='utf-8'))
        sidecar.write('{')
        splitfile.write('[')
        def line(name, value):
            streams[name].write(_dump(value) + '\n')
        def evidence(room, entry, mapping, fold):
            comma = ',' if counts['written'] else ''
            sidecar.write(comma + _dump(room['uid']) + ':' + _dump(mapping))
            splitfile.write(comma + _dump({'uid': room['uid'], 'source': entry['source'],
                'split': entry['split'], 'component': entry['component'], 'fold': fold}))
            line('bound_rooms.jsonl', room)
            line(entry['split'] + '_rooms.jsonl', room)
            counts['written'] += 1
        def ledger(value):
            line('common-admission.jsonl', value)
            bucket = source_counts.setdefault(value['source'], Counter())
            bucket['train_identities'] += 1
            bucket['eligible'] += int(value['eligible'])
            bucket['accepted'] += int(value['accepted'])
            bucket['rejected'] += int(not value['accepted'])
            for arm in ('joint', 'placement'):
                bucket[arm + '_qualified'] += int(value[arm]['qualified'])
        for uid in sorted(excluded):
            ledger(_excluded(identities[uid], excluded[uid]))
        tasks = _tasks(root, manifest, identities, eligible, oof, limits)
        processed = 0
        for room, mapping, joint, placement, decision in _ordered(tasks, workers):
            processed += 1
            ledger(decision)
            if decision['accepted']:
                uid = room['uid']
                accepted.append(uid)
                entry = identities[uid]
                for arm, turns in (('joint', joint), ('placement', placement)):
                    line(arm + '/train.jsonl', _record(room, entry, turns, True))
                evidence(room, entry, mapping, decision['fold'])
            if processed % 1000 == 0:
                print(f'processed={processed} accepted={len(accepted)}', flush=True)
        # Geometry is streamed; the optional dev limit depends only on UID identity.
        devuids = sorted(u for u, e in identities.items() if e['split'] == 'dev')
        selected = set(devuids if devlimit is None else devuids[:devlimit])
        for raw in _rooms(root, manifest, identities):
            if raw['uid'] not in selected:
                continue
            room, mapping = bind_room(raw, DEFAULT_SEED)
            sizes = {k: [max(.01, round(v, 2)) for v in values]
                     for k, values in predict_sizes(room, oof['full_prior']).items()}
            entry = identities[room['uid']]
            for arm in ('joint', 'placement'):
                turns = messages(room, arm, sizes if arm == 'placement' else None)
                line(arm + '/dev.jsonl', _record(room, entry, turns, False))
            evidence(room, entry, mapping, None)
            counts['dev'] += 1
        sidecar.write('}\n')
        splitfile.write(']\n')
    return sorted(accepted), counts['dev'], dict(sorted(source_counts.items()))


def build(corpus, out, workers=1, max_candidates=200, max_passes=4, devlimit=None):
    """Publish a new deterministic paired dataset; never refit after admission."""
    root, out = Path(corpus), Path(out)
    for name, value, minimum in (('workers', workers, 1), ('max_candidates', max_candidates, 0),
                                  ('max_passes', max_passes, 0), ('devlimit', devlimit, 1)):
        if name == 'devlimit' and value is None:
            continue
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if os.path.lexists(out):
        raise FileExistsError(f'refusing existing output: {out}')
    manifest, manifest_hash = _load_manifest(root / 'MANIFEST.json')
    if manifest.get('scope') != 'full_release_prepared_identity_closure':
        raise ValueError('requires full historical corpus identity closure')
    if set(manifest['files']) != {'prepared_rooms.jsonl', 'identity-manifest.json'}:
        raise ValueError('unexpected corpus file inventory')
    identities = _identities(root, manifest)
    heldout_components = {e['component'] for e in identities.values() if e['source'] in HELDOUT}
    reasons = {u: _eligible(e, heldout_components) for u, e in identities.items() if e['split'] == 'train'}
    eligible = frozenset(u for u, why in reasons.items() if not why)
    excluded = {u: why for u, why in reasons.items() if why}
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {n: sha256(code_root / n) for n in CODE}
    oof, pool = _freeze_prior(root, manifest, identities, eligible)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temporary:
        dest = Path(temporary)
        for arm in ('joint', 'placement'):
            (dest / arm).mkdir()
        json_dump(dest / 'prior-fit-pool.json', pool)
        json_dump(dest / 'size-priors.json', oof)
        accepted, devcount, sources = _write_data(dest, root, manifest, identities, eligible, excluded,
            oof, workers, {'max_candidates': max_candidates, 'max_passes': max_passes}, devlimit)
        counts = {'train': len(accepted), 'dev': devcount}
        result = {'version': 1, 'scope': 'main_paired_training_historical_release',
            'seed': DEFAULT_SEED, 'target_policy': TARGET_POLICY, 'quality_claim_allowed': False,
            'corpus_manifest_sha256': manifest_hash,
            'corpus_identity_manifest_sha256': manifest['files']['identity-manifest.json'],
            'input_consumed_sha256': {'MANIFEST.json': manifest_hash, **manifest['files']},
            'implementation_sha256': code_hashes,
            'source_release_manifest_sha256': manifest['source_release_manifest_sha256'],
            'source_release_files': manifest['source_release_files'], 'ir_sha256': manifest['ir_sha256'],
            'counts': counts, 'dataset_counts': {'train_identities': manifest['counts']['train'],
                'dev_identities': manifest['counts']['dev'], 'test_identities': manifest['counts']['test'],
                'prior_fit_rooms': len(eligible), 'eligible_candidates': len(eligible),
                'ineligible_train': len(excluded), 'common_decoder_train': len(accepted),
                'failed_candidates': len(eligible) - len(accepted), 'dev_retained': devcount,
                'joint_qualified': sum(c['joint_qualified'] for c in sources.values()),
                'placement_qualified': sum(c['placement_qualified'] for c in sources.values())},
            'by_source': sources, 'excluded_sources': sorted(HELDOUT),
            'prior_exposure': {'may_be_broader_than_common_decoder_pool': True,
                'extra_fit_uids': sorted(eligible - set(accepted)),
                'extra_fit_count': len(eligible) - len(accepted),
                'refit_after_decoder_admission': False},
            'input_policy': 'category_only_frozen_prepared_inventory_no_target_derived_ids',
            'prior_policy': 'category_room_type_axis_log_median_category_global_fallback_5_fold_component_OOF_train_full_prior_dev_test',
            'estimated_size_precision': 'centimeters; min0.01m, same display precision as joint labels',
            'split_policy': 'historical_release_no_reassignment',
            'component_policy': manifest['component_policy'],
            'dev_policy': 'all_dev_no_quality_filter_original_pose_loss_targets_unqualified',
            'devlimit': devlimit, 'test_policy': 'identity_closure_only_no_test_prompts',
            'repair_limits': {'max_candidates': max_candidates, 'max_passes': max_passes}}
        for arm in ('joint', 'placement'):
            json_dump(dest / arm / 'MANIFEST.json', {**result, 'arm': arm})
        result = {**result, 'files': {str(p.relative_to(dest)): sha256(p)
                                    for p in sorted(dest.rglob('*')) if p.is_file()}}
        json_dump(dest / 'MANIFEST.json', result)
        for path, expected in [(root / 'MANIFEST.json', manifest_hash),
                *[(root / n, h) for n, h in manifest['files'].items()],
                *[(code_root / n, h) for n, h in code_hashes.items()]]:
            if sha256(path) != expected:
                raise ValueError(f'input or implementation hash changed: {path}')
        _publish(dest, out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-candidates', type=int, default=200)
    parser.add_argument('--max-passes', type=int, default=4)
    parser.add_argument('--devlimit', '--dev-limit', type=int)
    args = parser.parse_args(argv)
    result = build(**vars(args))
    print(json.dumps({'counts': result['counts'], 'out': str(args.out)}))


if __name__ == '__main__':
    main()
