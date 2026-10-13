"""Tiny, isolated corpus tests for the main paired builder."""
import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fastfill.comparison import main_data as m
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.data import sha256


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def room(uid, split='train', size=1, source='Synthetic'):
    return dict(uid=uid, source=source, group='house:' + uid, split=split,
                boundary_type='polygon', boundary=[[0, 0], [20, 0], [20, 20], [0, 20]],
                height=5, fixed=[], meta={'flags': {}}, room_type='bedroom',
                objects=[dict(id=uid + ':original', category='chair', size=[size, 1, 1],
                              pos=[5, 5, 0], yaw=0, parent=None, anchor='floor')])


def fixture(root, transform=None):
    root.mkdir()
    data = [room('T' + str(i), size=1 + i / 10) for i in range(6)]
    data += [room('flag'), room('smith', source='SceneSmith'), room('spatial', source='SpatialGen'),
             room('D', 'dev', 90), room('E', 'dev'), room('X', 'test', 99)]
    if transform:
        data = transform(data)
    identity = []
    with (root / 'prepared_rooms.jsonl').open('wb') as f:
        for r in data:
            line = (json.dumps(r) + '\n').encode()
            f.write(line)
            identity.append(dict(uid=r['uid'], source=r['source'], split=r['split'],
                                 component='c:' + r['group'], original_group=r['group'],
                                 training_flags={'oob_objects': 1} if r['uid'] == 'flag' else {},
                                 prepared_sha256=hashlib.sha256(line).hexdigest()))
    (root / 'identity-manifest.json').write_text(json.dumps(identity))
    manifest = dict(scope='full_release_prepared_identity_closure',
                    component_policy='all_raw_group_alias_edges_and_all_released_prepared_content_layout_keys',
                    counts={s: sum(r['split'] == s for r in data) for s in ('train', 'dev', 'test')},
                    source_release_manifest_sha256='a' * 64,
                    source_release_files={}, ir_sha256={})
    refresh(root, manifest)
    return root


def refresh(root, manifest=None):
    manifest = manifest or read(root / 'MANIFEST.json')
    manifest['files'] = {n: sha256(root / n) for n in ('prepared_rooms.jsonl', 'identity-manifest.json')}
    (root / 'MANIFEST.json').write_text(json.dumps(manifest))


def build(root, out, **kwargs):
    return m.build(root, out, max_candidates=8, max_passes=1, **kwargs)


def test_frozen_pool_oof_heldout_dev_and_paired_serialization(tmp_path):
    root = fixture(tmp_path / 'corpus')
    out = tmp_path / 'out'
    manifest = build(root, out)
    assert manifest['target_policy'] == 'fixed_estimated_size_pose_repaired_v1'
    assert manifest['quality_claim_allowed'] is False
    fit = read(out / 'prior-fit-pool.json')
    assert fit['uids'] == ['T' + str(i) for i in range(6)]
    prior = read(out / 'size-priors.json')
    assert prior['full_prior']['global']['size'][0] < 2
    assert prior['full_prior']['category_room_type']['chair']['bedroom']['count'] == 6
    assert len(set(prior['fold_by_uid'].values())) == 5
    for uid, fold in prior['fold_by_uid'].items():
        held = {hashlib.sha256(u.encode()).hexdigest() for u, f in prior['fold_by_uid'].items() if f == fold}
        assert not held.intersection(prior['priors'][str(fold)]['provenance']['fitted_uid_hashes'])
    assert manifest['counts']['dev'] == 2
    ledger = rows(out / 'common-admission.jsonl')
    assert len(ledger) == manifest['dataset_counts']['train_identities'] == 9
    assert sum(r['eligible'] for r in ledger) == 6
    assert manifest['dataset_counts']['common_decoder_train'] == sum(r['accepted'] for r in ledger)
    sidecars = read(out / 'inventory-sidecar.json')
    for split in ('train', 'dev'):
        joint, placement = (rows(out / arm / (split + '.jsonl')) for arm in ('joint', 'placement'))
        assert [r['uid'] for r in joint] == [r['uid'] for r in placement]
        for j, p in zip(joint, placement):
            ju, pu = json.loads(j['messages'][1]['content']), json.loads(p['messages'][1]['content'])
            assert ju == {**pu, 'objects': [{k: v for k, v in o.items() if k != 'size'} for o in pu['objects']]}
            assert sidecars[j['uid']]['original_to_bound'] == {j['uid'] + ':original': 'obj_0001'}
            assert all('size' not in o for o in json.loads(p['messages'][2]['content'])['placements'])
            assert all(v == round(v, 2) for o in pu['objects'] for v in o['size'])
            if split == 'dev':
                assert p['target_qualified'] is False
    assert not list(out.rglob('*test*'))
    for name, expected in manifest['files'].items():
        assert sha256(out / name) == expected
    assert manifest['input_consumed_sha256']['prepared_rooms.jsonl'] == sha256(root / 'prepared_rooms.jsonl')


def test_failure_ledger_does_not_refit_and_no_dev_repair(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'corpus')
    repair = m.repair
    called = []
    def reject_one(r, sizes, **kwargs):
        assert r['split'] == 'train'
        called.append(r['uid'])
        result = repair(r, sizes, **kwargs)
        return {**result, 'status': 'rejected'} if r['uid'] == 'T0' else result
    monkeypatch.setattr(m, 'repair', reject_one)
    out = tmp_path / 'out'
    manifest = build(root, out)
    assert len(called) == 6
    assert read(out / 'prior-fit-pool.json')['uids'] == ['T' + str(i) for i in range(6)]
    ledger = {r['uid']: r for r in rows(out / 'common-admission.jsonl')}
    assert ledger['T0']['joint']['qualified'] and not ledger['T0']['accepted']
    assert manifest['prior_exposure']['extra_fit_uids'] == ['T0']
    assert manifest['prior_exposure']['may_be_broader_than_common_decoder_pool'] is True


@pytest.mark.parametrize('mode', ['cross_split', 'heldout_bridge'])
def test_full_identity_closure_before_filtering(tmp_path, mode):
    root = fixture(tmp_path / 'corpus')
    identities = read(root / 'identity-manifest.json')
    target = 'X' if mode == 'cross_split' else 'smith'
    identities = [{**r, 'component': 'c:house:T0'} if r['uid'] == target else r for r in identities]
    (root / 'identity-manifest.json').write_text(json.dumps(identities))
    refresh(root)
    if mode == 'cross_split':
        with pytest.raises(ValueError, match='cross-split'):
            build(root, tmp_path / 'out')
    else:
        build(root, tmp_path / 'out')
        assert 'T0' not in read(tmp_path / 'out/prior-fit-pool.json')['uids']


@pytest.mark.parametrize('name', ['identity-manifest.json', 'prepared_rooms.jsonl'])
def test_file_hash_failure(tmp_path, name):
    root = fixture(tmp_path / 'corpus')
    with (root / name).open('ab') as f:
        f.write(b' ')
    with pytest.raises(ValueError, match='hash'):
        build(root, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_serialized_gt_is_authority_and_test_labels_unused(tmp_path):
    def transform(data):
        return [{**r, 'objects': [{**r['objects'][0], 'size': [.004, 1, 1]}]} if r['uid'] == 'T0'
                else {k: v for k, v in r.items() if k != 'objects'} if r['split'] == 'test' else r for r in data]
    root = fixture(tmp_path / 'corpus', transform)
    out = tmp_path / 'out'
    build(root, out)
    rejected = next(r for r in rows(out / 'common-admission.jsonl') if r['uid'] == 'T0')
    assert not rejected['joint']['qualified'] and not rejected['accepted']
    assert 'T0' in read(out / 'prior-fit-pool.json')['uids']


def test_repeat_parallel_and_dev_limit(tmp_path):
    root = fixture(tmp_path / 'corpus')
    a, b = tmp_path / 'a', tmp_path / 'b'
    ma = build(root, a, workers=1, devlimit=1)
    mb = build(root, b, workers=2, devlimit=1)
    assert ma == mb
    assert ma['counts']['dev'] == 1
    assert ma['dataset_counts']['dev_identities'] == 2
    for name in ma['files']:
        assert (a / name).read_bytes() == (b / name).read_bytes()


@pytest.mark.parametrize('kwargs', [{'workers': 0}, {'workers': True}, {'max_passes': -1},
                                  {'max_candidates': -1}, {'devlimit': 0}])
def test_bad_arguments(tmp_path, kwargs):
    with pytest.raises(ValueError):
        m.build(tmp_path / 'none', tmp_path / 'out', **kwargs)


def test_no_overwrite_and_cli(tmp_path):
    root = fixture(tmp_path / 'corpus')
    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, '-m', 'fastfill.comparison.main_data', '--corpus', str(root),
                             '--out', str(out), '--max-candidates', '0', '--max-passes', '0'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['counts']['dev'] == 2
    with pytest.raises(FileExistsError):
        build(root, out)


@pytest.mark.parametrize('violation', ['drop', 'resize', 'support', 'fixed'])
def test_teacher_cannot_change_frozen_inventory(tmp_path, monkeypatch, violation):
    root = fixture(tmp_path / 'corpus')
    original = m.repair
    def faulty(room, sizes, **kwargs):
        result = original(room, sizes, **kwargs)
        if room['uid'] != 'T0':
            return result
        changed = copy.deepcopy(result['room'])
        if violation == 'drop':
            changed['objects'] = []
        elif violation == 'resize':
            changed['objects'][0]['size'] = [.1, .1, .1]
        elif violation == 'support':
            changed['objects'][0]['parent'] = 'missing'
        else:
            changed['height'] = 6
        return {**result, 'room': changed, 'status': 'ok'}
    monkeypatch.setattr(m, 'repair', faulty)
    out = tmp_path / 'out'
    build(root, out)
    decision = next(r for r in rows(out / 'common-admission.jsonl') if r['uid'] == 'T0')
    assert not decision['accepted'] and not decision['placement']['qualified']
    assert 'inventory_size_support_or_context_changed' in decision['placement']['reasons']


def test_serialized_fixed_geometry_and_collisions_are_checked():
    raw = room('test')
    raw['objects'][0]['pos'] = [5.004, 5, 0]
    raw['fixed'] = [dict(id='original-fixed', category='door', size=[1.004, 1, 1],
                         pos=[5.004, 5, 0], yaw=.001)]
    bound, _ = bind_room(raw)
    turns, result = m._qualification(bound, 'joint')
    serial = m._serialized_room(bound, turns, 'joint')
    assert serial['fixed'][0]['size'] == [1, 1, 1]
    assert serial['fixed'][0]['pos'] == [5, 5, 0]
    assert serial['fixed'][0]['yaw'] == 0
    assert not result['qualified'] and result['counts']['fixed_blocking'] == 1
    raw['fixed'] = []
    raw['objects'].append({**raw['objects'][0], 'id': 'overlap', 'pos': [5.2, 5, 0]})
    _, result = m._qualification(bind_room(raw)[0], 'joint')
    assert result['counts']['valid'] and result['counts']['main'] == 1
    assert result['counts']['severe'] == 1 and not result['qualified']


def test_serialization_rounding_can_invalidate_valid_raw_gt():
    raw = room('test')
    raw['objects'][0]['size'] = [1.009, 1, 1]
    raw['objects'][0]['pos'] = [.4049, 5, 0]
    assert m.check(raw)['valid']
    _, result = m._qualification(bind_room(raw)[0], 'joint')
    assert not result['qualified']
    assert result['counts']['oob'] == 1


def test_consumed_bytes_hash_cannot_be_hidden_by_restoring_disk(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'corpus')
    target = root / 'prepared_rooms.jsonl'
    original = target.read_bytes()
    real_open = Path.open
    def swapped(path, *args, **kwargs):
        if path == target and args == ('rb',):
            return io.BytesIO(b' ' + original)
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', swapped)
    with pytest.raises(ValueError, match='hash'):
        build(root, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_late_input_drift_aborts_publication(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'corpus')
    write = m._write_data
    def drift(*args, **kwargs):
        result = write(*args, **kwargs)
        path = root / 'MANIFEST.json'
        path.write_bytes(path.read_bytes() + b'\n')
        return result
    monkeypatch.setattr(m, '_write_data', drift)
    with pytest.raises(ValueError, match='hash changed'):
        build(root, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_real_loader_and_evaluator_provenance_handshake(tmp_path):
    from fastfill.comparison.train_pair_helpers import load_pair
    from fastfill.comparison.eval_data import _fit_prior
    root = fixture(tmp_path / 'corpus')
    out = tmp_path / 'out'
    manifest = build(root, out)
    pairs, receipt = load_pair(out)
    assert receipt['counts'] == manifest['counts']
    identities = {r['uid']: r for r in read(root / 'identity-manifest.json')}
    prior, fitted = _fit_prior(out, manifest, sha256(root / 'MANIFEST.json'),
                              sha256(root / 'identity-manifest.json'), identities)
    assert fitted == read(out / 'prior-fit-pool.json')['uids']
    assert prior['category_room_type']['chair']['bedroom']['count'] == 6
    ledger = {r['uid']: r for r in rows(out / 'common-admission.jsonl')}
    for arm in ('joint', 'placement'):
        for record in pairs['train'][arm]:
            evidence = ledger[record['uid']][arm]
            assert evidence['target_sha256'] == hashlib.sha256(
                record['messages'][-1]['content'].encode()).hexdigest()
            canonical = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
            assert evidence['request_sha256'] == hashlib.sha256(
                canonical(record['messages'][:-1]).encode()).hexdigest()
            assert evidence['counts'] == {'valid': True, **{k: 0 for k in
                ('oob', 'support_fail', 'ceiling', 'fixed_blocking', 'main', 'severe')}}
            if arm == 'placement':
                supplied = {o['id']: o['size'] for o in json.loads(record['messages'][1]['content'])['objects']}
                assert evidence['estimated_sizes_sha256'] == hashlib.sha256(canonical(supplied).encode()).hexdigest()


def test_binding_failure_is_kept_in_candidate_ledger(tmp_path):
    def transform(data):
        return [{**r, 'boundary': [[0, 0], [0, 0], [0, 0]]} if r['uid'] == 'T0' else r for r in data]
    root = fixture(tmp_path / 'corpus', transform)
    out = tmp_path / 'out'
    manifest = build(root, out)
    rejected = next(r for r in rows(out / 'common-admission.jsonl') if r['uid'] == 'T0')
    assert rejected['eligible'] and not rejected['accepted']
    assert rejected['fold'] is not None
    assert 'binding_failed' in rejected['reasons'][0]
    assert manifest['dataset_counts']['prior_fit_rooms'] == 6
    assert read(out / 'size-priors.json')['full_prior']['provenance']['room_count'] == 6


def test_qualification_hashes_bind_centimeter_serialization():
    bound, _ = bind_room(room('rounded', size=1.234))
    raw_sizes = {bound['objects'][0]['id']: [1.234, 1.236, 1.239]}
    # The builder rounds prior predictions before calling qualification.
    sizes = {oid: [round(v, 2) for v in values] for oid, values in raw_sizes.items()}
    for arm in ('joint', 'placement'):
        turns, evidence = m._qualification(bound, arm, sizes if arm == 'placement' else None)
        assert evidence['qualified']
        assert evidence['target_sha256'] == hashlib.sha256(turns[-1]['content'].encode()).hexdigest()
        assert evidence['request_sha256'] == hashlib.sha256(m._dump(turns[:-1]).encode()).hexdigest()
        if arm == 'joint':
            assert json.loads(turns[-1]['content'])['placements'][0]['size'][0] == 1.23
        else:
            displayed = {o['id']: o['size'] for o in json.loads(turns[1]['content'])['objects']}
            assert next(iter(displayed.values())) == [1.23, 1.24, 1.24]
            assert evidence['estimated_sizes_sha256'] == hashlib.sha256(m._dump(displayed).encode()).hexdigest()
            assert evidence['estimated_sizes_sha256'] != hashlib.sha256(m._dump(raw_sizes).encode()).hexdigest()


def test_target_independent_predictions_and_bounded_pending_tasks(tmp_path, monkeypatch):
    from fastfill.comparison.size_prior import predict_sizes
    root = fixture(tmp_path / 'corpus')
    out = tmp_path / 'out'
    build(root, out)
    prior = read(out / 'size-priors.json')['full_prior']
    original = room('new')
    changed = {**original,
               'objects': [{**o, 'size': [900, 900, 900]} for o in original['objects']]}
    assert predict_sizes(original, prior) == predict_sizes(changed, prior)
    manifest = read(root / 'MANIFEST.json')
    identities = {r['uid']: r for r in read(root / 'identity-manifest.json')}
    oof = read(out / 'size-priors.json')
    tasks = m._tasks(root, manifest, identities, set(oof['fold_by_uid']), oof,
                     {'max_candidates': 0, 'max_passes': 0})
    consumed = []
    def count_tasks():
        for task in tasks:
            consumed.append(task[0]['uid'])
            yield task
    pending = m._ordered(count_tasks(), 2)
    assert next(pending)[0]['uid'] == 'T0'
    assert len(consumed) == 4
    assert len(list(pending)) == 5


def test_room_type_logmedian_and_category_global_fallback_without_query_fit(tmp_path, monkeypatch):
    from fastfill.comparison.size_prior import predict_sizes

    def transform(data):
        chairs = [{**room(f'{kind}{i}', size=size), 'room_type': kind}
                  for kind, size in (('bedroom', 1), ('living room', 4)) for i in range(10)]
        tables = [room(f'table{i}', size=9) for i in range(5)]
        tables = [{**r, 'objects': [{**o, 'category': 'table'} for o in r['objects']]} for r in tables]
        return chairs + tables + [r for r in data if r['split'] != 'train']

    fits = []
    original_fit = m.build_oof_priors
    def fit_once(observations, *args, **kwargs):
        fits.append(sorted(r['uid'] for r in observations))
        return original_fit(observations, *args, **kwargs)

    monkeypatch.setattr(m, 'build_oof_priors', fit_once)
    root = fixture(tmp_path / 'corpus', transform)
    out = tmp_path / 'out'
    manifest = build(root, out)
    oof = read(out / 'size-priors.json')
    prior = oof['full_prior']
    before = copy.deepcopy(prior)
    assert len(fits) == 1 and len(fits[0]) == 25
    assert read(out / 'prior-fit-pool.json')['uids'] == fits[0]
    assert manifest['seed'] == 'unknown-size-v1-20261013'
    assert manifest['prior_policy'].startswith('category_room_type_axis_log_median')
    assert oof['provenance']['conditioning'] == 'category_room_type'
    assert prior['provenance']['category_room_type_count'] == 3

    def predict(prior, kind, category='chair'):
        # Query has no GT sizes/poses and is never passed to fitting.
        query = {'uid': 'query', 'room_type': kind, 'objects': [{'id': 'query:original', 'category': category}]}
        return predict_sizes(query, prior)['query:original'][0]

    assert prior['category_room_type']['chair']['bedroom']['count'] == 10
    assert prior['category_room_type']['chair']['living room']['count'] == 10
    assert predict(prior, 'bedroom') == pytest.approx(1)
    assert predict(prior, 'livingroom') == pytest.approx(4)
    assert predict(prior, 'kitchen') == pytest.approx(2)  # exp(median(log(1), log(4)))
    assert predict(prior, 'bedroom', 'unknown category') == pytest.approx(4)
    for fold_prior in oof['priors'].values():
        assert predict(fold_prior, 'bedroom') == pytest.approx(1)
        assert predict(fold_prior, 'living room') == pytest.approx(4)
    assert prior == before and len(fits) == 1
    assert 'query' not in oof['fold_by_uid']
