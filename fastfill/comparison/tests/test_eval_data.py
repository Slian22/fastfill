"""Small synthetic frozen evaluation fixtures; no production data or APIs."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from fastfill.comparison import eval_data as e
from fastfill.comparison.data import sha256
from fastfill.comparison.size_prior import build_oof_priors


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def dump(path, value):
    path.write_text(json.dumps(value) + '\n')


def room(uid, split='test', source='IL3D_3dfront', kind='bedroom'):
    return dict(uid=uid, source=source, group='house:' + uid, split=split,
                boundary_type='polygon', boundary=[[0, 0], [20, 0], [20, 20], [0, 20]],
                height=4, room_type=kind, fixed=[], meta={'flags': {'oob_objects': 1}},
                objects=[dict(id=uid + ':a', category='chair', size=[9, 8, 7],
                              pos=[30, 5, 0], yaw=0, parent=None),
                         dict(id=uid + ':b', category='table', size=[3, 2, 1],
                              pos=[5, 5, 0], yaw=0, parent=None)])


def fixture(root, tests=None):
    corpus, train = root / 'corpus', root / 'training'
    corpus.mkdir(parents=True)
    train.mkdir()
    fitting = [room('train' + str(i), 'train', 'Synthetic') for i in range(5)]
    fitting = [{**r, 'objects': [{**o, 'size': [.004, 1.234, 2.345]} for o in r['objects']]}
               for r in fitting]
    data = fitting + (tests if tests is not None else [room('test'), room('smith', source='SceneSmith'),
                                                     room('spatial', source='SpatialGen')])
    identities = []
    with (corpus / 'prepared_rooms.jsonl').open('wb') as stream:
        for r in data:
            line = (json.dumps(r) + '\n').encode()
            stream.write(line)
            identities.append(dict(uid=r['uid'], split=r['split'], source=r['source'],
                                   original_group=r['group'], component='c:' + r['group'],
                                   training_flags=r['meta']['flags'],
                                   prepared_sha256=hashlib.sha256(line).hexdigest()))
    dump(corpus / 'identity-manifest.json', identities)
    dump(corpus / 'MANIFEST.json', dict(version=1, scope='full_release_prepared_identity_closure',
         counts={s: sum(r['split'] == s for r in data) for s in ('train', 'dev', 'test')},
         files={n: sha256(corpus / n) for n in ('prepared_rooms.jsonl', 'identity-manifest.json')}))
    prior = build_oof_priors(fitting, {r['uid']: 'c:' + r['group'] for r in fitting})
    dump(train / 'size-priors.json', prior)
    dump(train / 'prior-fit-pool.json', {'uids': sorted(r['uid'] for r in fitting)})
    dump(train / 'MANIFEST.json', dict(version=1, seed=e.SEED,
         corpus_manifest_sha256=sha256(corpus / 'MANIFEST.json'),
         corpus_identity_manifest_sha256=sha256(corpus / 'identity-manifest.json'),
         files={n: sha256(train / n) for n in ('size-priors.json', 'prior-fit-pool.json')}))
    return corpus, train


def refresh(root):
    man = read(root / 'MANIFEST.json')
    man['files'] = {n: sha256(root / n) for n in man['files']}
    dump(root / 'MANIFEST.json', man)


def test_real_inference_loader_accepts_builder_contract(tmp_path):
    from fastfill.comparison.infer import load_eval
    corpus, train = fixture(tmp_path)
    out = tmp_path / 'out'
    result = e.build(corpus, train, out)
    for arm in ('joint', 'placement'):
        loaded, receipt = load_eval(out, arm)
        assert [r['uid'] for r in loaded] == result['expected_uids']
        assert receipt['expected_uids'] == sorted(['test', 'smith', 'spatial'])


def test_all_test_retained_prompts_labels_and_sizes_separate(tmp_path):
    corpus, train = fixture(tmp_path)
    out = tmp_path / 'out'
    result = e.build(corpus, train, out)
    assert result['counts']['test'] == 3
    assert result['audit']['written_rows'] == result['audit']['test_rows'] == 3
    assert result['audit']['failures'] == []
    labels = {r['uid']: r for r in rows(out / 'rooms.jsonl')}
    estimates = {r['uid']: r['estimated_sizes'] for r in rows(out / 'size-estimates.jsonl')}
    sidecars = read(out / 'inventory-sidecar.json')
    for arm in ('joint', 'placement'):
        requests = rows(out / arm / 'requests.jsonl')
        assert {r['uid'] for r in requests} == set(labels)
        for r in requests:
            assert set(r) <= {'uid', 'source', 'group', 'component', 'split', 'room_type', 'messages'}
            assert [m['role'] for m in r['messages']] == ['system', 'user']
            user = json.loads(r['messages'][1]['content'])
            assert all(set(o) == ({'id', 'category'} if arm == 'joint' else {'id', 'category', 'size'})
                       for o in user['objects'])
            assert all(v == [.01, 1.23, 2.35] for v in estimates[r['uid']].values())
            assert labels[r['uid']]['group'] == r['group'] == r['component']
            assert sidecars[r['uid']]['component'] == r['component']
    assert set(result['source_heldout']) == {'SceneSmith', 'SpatialGen'}
    assert result['new_blind_test_claim'] is False
    for name, digest in result['files'].items():
        assert sha256(out / name) == digest


def test_shuffle_and_target_counterfactual_do_not_change_requests(tmp_path):
    test = room('test')
    changed = {**copy.deepcopy(test), 'objects': [{**o, 'size': [1, 2, 3], 'pos': [2, 3, 0],
                                                 'yaw': 1.7} for o in reversed(test['objects'])]}
    a = fixture(tmp_path / 'a', [test])
    b = fixture(tmp_path / 'b', [changed])
    e.build(*a, tmp_path / 'out-a')
    e.build(*b, tmp_path / 'out-b')
    for name in ('joint/requests.jsonl', 'placement/requests.jsonl', 'size-estimates.jsonl',
                 'gpt-selection.json'):
        assert (tmp_path / 'out-a' / name).read_bytes() == (tmp_path / 'out-b' / name).read_bytes()


@pytest.mark.parametrize('mode', ['uid', 'component', 'heldout', 'heldout_bridge'])
def test_fit_test_and_source_heldout_leakage_rejected(tmp_path, mode):
    corpus, train = fixture(tmp_path)
    prior = read(train / 'size-priors.json')
    identity = read(corpus / 'identity-manifest.json')
    if mode == 'uid':
        prior['fold_by_uid']['test'] = 0
    elif mode == 'component':
        prior['full_prior']['provenance']['component_hashes'].append(hashlib.sha256(b'c:house:test').hexdigest())
    else:
        identity = [{**r, 'source': 'SceneSmith'} if r['uid'] == 'train0' and mode == 'heldout'
                    else {**r, 'component': 'c:house:train0'} if r['uid'] == 'smith' else r for r in identity]
        dump(corpus / 'identity-manifest.json', identity)
        refresh(corpus)
        manifest = read(train / 'MANIFEST.json')
        manifest.update(corpus_manifest_sha256=sha256(corpus / 'MANIFEST.json'),
                        corpus_identity_manifest_sha256=sha256(corpus / 'identity-manifest.json'))
        dump(train / 'MANIFEST.json', manifest)
    dump(train / 'size-priors.json', prior)
    refresh(train)
    with pytest.raises(ValueError):
        e.build(corpus, train, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('root,name', [('corpus', 'prepared_rooms.jsonl'), ('corpus', 'identity-manifest.json'),
                                     ('training', 'size-priors.json'), ('training', 'prior-fit-pool.json')])
def test_all_input_hashes_verified(tmp_path, root, name):
    corpus, train = fixture(tmp_path)
    with (tmp_path / root / name).open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError, match='hash'):
        e.build(corpus, train, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_binding_failure_has_uid_receipt_and_no_partial_output(tmp_path):
    bad = room('broken')
    bad['objects'][1]['id'] = bad['objects'][0]['id']
    corpus, train = fixture(tmp_path, [bad])
    with pytest.raises(e.EvaluationError, match='broken') as error:
        e.build(corpus, train, tmp_path / 'out')
    assert error.value.audit['failures'][0]['uid'] == 'broken'
    assert error.value.audit['failed_rows'] == 1
    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('.out-*'))


def test_six_strata_unequal_population_stable_and_frozen_before_prediction(tmp_path, monkeypatch):
    tests = [room(f'{i}-{j}', source=s, kind=k) for i, (s, k) in enumerate(e.PAPER_STRATA)
             for j in range([35, 2, 4, 1, 0, 3][i])]
    tests += [room('combined', kind='livingdiningroom'), room('heldout', source='SceneSmith')]
    corpus, train = fixture(tmp_path, tests)
    original = e.predict_sizes
    def predict(r, prior):
        selected = list(tmp_path.glob('.out-*/gpt-selection.json'))
        assert len(selected) == 1
        assert len(read(selected[0])['selected_uids']) == 40
        return original(r, prior)
    monkeypatch.setattr(e, 'predict_sizes', predict)
    e.build(corpus, train, tmp_path / 'out')
    selected = read(tmp_path / 'out/gpt-selection.json')
    assert [v['population'] for v in selected['strata'].values()] == [35, 2, 4, 1, 0, 3]
    assert [v['selected_count'] for v in selected['strata'].values()] == [30, 2, 4, 1, 0, 3]
    assert selected['asset_eligibility'] == 'unknown'
    assert 'renderable' not in selected
    assert selected == e.select_gpt(list(reversed(tests)))
    assert 'combined' not in selected['selected_uids']


def test_no_replace_and_late_input_change(tmp_path, monkeypatch):
    corpus, train = fixture(tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(FileExistsError):
        e.build(corpus, train, out)
    out.rmdir()
    original = e.predict_sizes
    def drift(r, prior):
        with (train / 'prior-fit-pool.json').open('ab') as stream:
            stream.write(b' ')
        return original(r, prior)
    monkeypatch.setattr(e, 'predict_sizes', drift)
    with pytest.raises(ValueError, match='hash'):
        e.build(corpus, train, out)
    assert not out.exists()


@pytest.mark.parametrize('mode', ['seed', 'corpus_hash', 'identity_hash', 'extra_file', 'missing_file',
                                 'row_hash', 'prior_uid_digest', 'prior_count', 'prior_component_source'])
def test_invalid_frozen_provenance_and_inventory(tmp_path, mode):
    corpus, train = fixture(tmp_path)
    if mode in ('seed', 'corpus_hash', 'identity_hash'):
        manifest = read(train / 'MANIFEST.json')
        key = {'seed': 'seed', 'corpus_hash': 'corpus_manifest_sha256',
               'identity_hash': 'corpus_identity_manifest_sha256'}[mode]
        manifest[key] = 'wrong'
        dump(train / 'MANIFEST.json', manifest)
    elif mode == 'extra_file':
        (train / 'unlisted.json').write_text('{}')
    elif mode == 'missing_file':
        (train / 'prior-fit-pool.json').unlink()
    elif mode == 'row_hash':
        identity = read(corpus / 'identity-manifest.json')
        identity[-1]['prepared_sha256'] = '0' * 64
        dump(corpus / 'identity-manifest.json', identity)
        refresh(corpus)
        manifest = read(train / 'MANIFEST.json')
        manifest.update(corpus_manifest_sha256=sha256(corpus / 'MANIFEST.json'),
                        corpus_identity_manifest_sha256=sha256(corpus / 'identity-manifest.json'))
        dump(train / 'MANIFEST.json', manifest)
    else:
        prior = read(train / 'size-priors.json')
        key = {'prior_uid_digest': 'fitted_uid_sha256', 'prior_count': 'room_count',
               'prior_component_source': 'component_source'}[mode]
        prior['full_prior']['provenance'][key] = 'wrong'
        dump(train / 'size-priors.json', prior)
        refresh(train)
    with pytest.raises((ValueError, FileNotFoundError)):
        e.build(corpus, train, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_atomic_publish_cannot_replace_racing_directory(tmp_path, monkeypatch):
    corpus, train = fixture(tmp_path)
    out = tmp_path / 'out'
    publish = e.corpus_io._publish
    def race(staging, destination):
        destination.mkdir()
        publish(staging, destination)
    monkeypatch.setattr(e.corpus_io, '_publish', race)
    with pytest.raises(FileExistsError):
        e.build(corpus, train, out)
    assert list(out.iterdir()) == []
    assert not list(tmp_path.glob('.out-*'))


def test_cli_success_and_failure_receipt(tmp_path):
    import subprocess
    import sys

    corpus, train = fixture(tmp_path / 'good')
    result = subprocess.run([sys.executable, '-m', 'fastfill.comparison.eval_data', '--corpus', str(corpus),
                             '--training-data', str(train), '--out', str(tmp_path / 'out')],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['counts']['test'] == 3
    bad = room('bad-cli')
    bad['objects'][0]['parent'] = 'unknown'
    corpus, train = fixture(tmp_path / 'bad', [bad])
    result = subprocess.run([sys.executable, '-m', 'fastfill.comparison.eval_data', '--corpus', str(corpus),
                             '--training-data', str(train), '--out', str(tmp_path / 'bad-out')],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert json.loads(result.stderr)['audit']['failures'][0]['uid'] == 'bad-cli'
    assert not (tmp_path / 'bad-out').exists()
