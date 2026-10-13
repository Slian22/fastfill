"""Amendment-005 report/delivery generator and the frozen reference builder (no network, no jobs)."""
import gzip
import json
import math
import random

import pytest

from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer, judge_batch
from fastfill.comparison import llm_agent as L
from fastfill.comparison import reference_rows as RR
from fastfill.comparison import report_v2 as R
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.fid import WEIGHTS_SHA256
from fastfill.comparison.tests.test_llm_agent import FakeApi, args, target, world, write_rows  # noqa: F401
from fastfill.validate import check


def merged(root, data, arm, rooms, raw=None, cap=()):
    """A verified merged-inference dir; `raw` overrides answers, `cap` UIDs hit the cap without EOS."""
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    run = root / ('infer-' + arm)
    run.mkdir()
    receipt = {'data': {'manifest_sha256': h.sha256(data / 'MANIFEST.json')},
               'training_data': {'manifest_sha256': 'a' * 64}, 'base_sha256': 'base', 'config': {'arm': arm}}
    receipt['run_sha256'] = infer._hash(receipt)
    h.write_json(run / 'run_receipt.json', receipt)
    write_rows(run / 'records.jsonl', [
        {'uid': u, 'source': rooms[u]['source'], 'group': rooms[u]['group'], 'arm': arm,
         'raw': (raw or {}).get(u, target(None, arm, rooms[u])), 'error': None,
         'reached_token_cap': u in cap, 'stopped_on_eos': u not in cap, 'generated_tokens': 50, 'input_tokens': 400,
         'batch_id': i // 8, 'duration_seconds': 36., 'run_sha256': receipt['run_sha256']}
        for i, u in enumerate(manifest['expected_uids'])])
    h.write_json(run / 'result.json', {'status': 'INFERENCE_COMPLETE', 'pending_count': 0})
    h.write_json(run / 'MANIFEST.json', {'expected_uids': manifest['expected_uids'], 'files': h.artifact_manifest(run)})
    return run


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def built(world):
    root, rooms = world
    data = root / 'data'
    runs = {'joint': merged(root, data, 'joint', rooms, raw={'x-06': '{"placements": [', 'y-0': '{}'}, cap={'x-05'}),
            'placement': merged(root, data, 'placement', rooms)}
    e.evaluate(data, runs, root / 'geometry-v2', repeats=200)
    L.run(args(root), FakeApi(rooms, {'il-bed-1': ['bad', 'valid'], 'y-0': ['bad', 'worse', 'still bad']}))
    ev.evaluate(data, root / 'cohort.json', root / 'run', root / 'llm-eval', runs, repeats=200)
    requests = {arm: {r['uid']: r for r in _jsonl(data / arm / 'requests.jsonl')} for arm in h.ARMS}
    reference = RR.build_reference(_jsonl(data / 'rooms.jsonl'), requests, h.sha256(data / 'MANIFEST.json'), ['il-bed-2'])
    infer.write_once(root / 'reference.json', reference)
    geometry_sha = h.sha256(root / 'geometry-v2' / 'MANIFEST.json')
    h.write_json(root / 'fid.json', {
        'extractor': 'pytorch_fid:0.3.0:pool3', 'feature_dimension': 2048, 'weights_sha256': WEIGHTS_SHA256,
        'provenance': {'provenance': {'geometry_manifest_sha256': geometry_sha}},
        'strata': {'IL3D_3dfront/bedroom': {'joint_vs_reference': 30., 'placement_vs_reference': 31.,
                                            'counts': {'requested': 3, 'reference_eligible': 3, 'shared_success': 3}}}})
    records = [{'id': f'{u}-{arm}', 'uid': u, 'arm': arm, 'stratum': s, 'group': None, 'launched': True,
                **({'status': 'failed', 'error': 'client_timeout'} if (u, arm) == ('il-bed-2', 'placement') else
                   {'status': 'completed', 'scores': {'functionality_score': 8 + (arm == 'joint'), 'layout_score': 7,
                                                      'aesthetics_score': 9, 'overall_score': 8}})}
               for u, s in (('il-bed-0', 'IL3D_3dfront/bedroom'), ('il-bed-1', 'IL3D_3dfront/bedroom'),
                            ('il-bed-2', 'IL3D_3dfront/bedroom'), ('holo-k-0', 'OptiScene_holodeck/kitchen'))
               for arm in h.ARMS]
    (root / 'judge').mkdir()
    write_rows(root / 'judge' / 'records.jsonl', records)
    h.write_json(root / 'judge' / 'summary.json', {**judge_batch.summarize(records, 'gpt-6.1-sol', len(records)),
                                                   'provenance': {'geometry_manifest_sha256': geometry_sha}})
    return root, rooms, runs


def _build(root, runs, out='report', **extra):
    options = {'agents': {'L': root / 'llm-eval'}, 'fid': root / 'fid.json', 'judge': root / 'judge', 'repeats': 200, **extra}
    return R.build(root / 'data', root / 'geometry-v2', runs, root / 'reference.json', root / out, **options)


def test_end_to_end_report_and_delivery(built):
    root, rooms, runs = built
    result = _build(root, runs)
    t, d = result['tables'], result['delivery']
    n = len(rooms)
    # J answers GT except one cap-without-EOS (valid JSON), one truncated JSON and one schema-invalid output.
    t1 = {row['endpoint'][:4]: row for row in t['T1']}
    assert t1['(a) ']['J'] == pytest.approx((n - 3) / n) and t1['(a) ']['P'] == 1
    assert t1['(b) ']['J']['estimate'] == 0 and t1['(b) ']['delta']['ci95'][1] < 0
    assert [t1['(a) ']['p_holm'], t1['(b) ']['p_holm']] == R.holm([t1['(a) ']['p'], t1['(b) ']['p']])
    # P sizes are GT + 0.1 m, so P is oversized and its (a') factor shrinks it; J is unbiased.
    assert d['scale_factor']['joint'] == 1 and d['scale_factor']['placement'] < 1
    assert d['B_footprint']['P']['estimate'] > 0 and d['B_footprint']['J']['estimate'] == 0
    assert d == {**d, **R.delivery(t1['(b) ']['delta'], t1["(a′)"]['delta'])}
    assert d['condition_i_size_log_mae'] and not d['condition_ii_debiased_success'] and d['delivered'] == 'P'
    t6 = {row['system']: row for row in t['T6']}
    assert (t6['J']['token_cap_without_eos'], t6['J']['cap_rule_flipped_complete'], t6['J']['invalid_json'],
            t6['J']['invalid_schema_or_geometry'], t6['J']['complete']) == (1, 1, 1, 1, n - 3)
    assert t6['P']['complete'] == n and t6['L-raw']['invalid_output'] == 2 and t6['L-harness']['invalid_output'] == 1
    # Reference rows sit beside the arms; sensitivity scopes use the frozen lists.
    assert t['T3'][0]['gt_success'] == 1 and t['T2'][0]['prior_size_log_mae'] > 0
    s1 = {row['scope']: row for row in t['S1']}
    assert s1['排除声明的已知缺陷房间（1）']['requests'] == n - 1 and s1[f'GT 通过冻结几何门（{n}）']['requests'] == n
    assert {row['level'] for row in t['S2']} == {'N'} and 'prior_level_pair_category_unresolved_without_frozen_size_priors' in result['warnings']
    # Paper strata: rotated OOR, FID with small-N flag, GPT from the judge records.
    t4 = {(row['stratum'], row['system']): row for row in t['T4']}
    bed = t4['IL3D_3dfront/bedroom', 'J']
    assert bed['fid'] == 30 and bed['fid_flag'] == 'N<50' and bed['gpt'] == pytest.approx(25 / 3)
    assert t4['IL3D_3dfront/bedroom', 'L-harness']['requests'] == 3 and t4['IL3D_3dfront/bedroom', 'L-raw']['fid'] is None
    gpt = t['T4_gpt'][0]
    assert gpt['n'] == 3 and gpt['delta']['estimate'] == pytest.approx(1 / 3)
    # Agents on the cohort: J/P/agent contrasts on the two pooled scopes only, with cost.
    assert {(r['scope'], r['comparison']) for r in t['T5_contrasts']} == {
        (s, c) for s in ('cohort', 'strata_pooled') for c in ('J−P', 'J−L-raw', 'P−L-raw', 'J−L-harness', 'P−L-harness')}
    assert {r['system'] for r in t['T5']} == {'J', 'P', 'L-raw', 'L-harness'} and result['cohort_rooms'] == 36
    cost = {r['system']: r for r in t['T5_cost']}
    assert cost['L']['model'] == L.EXPECTED_MODEL and cost['J']['gpu_hours_full_test_generation'] == pytest.approx(5 * 36 / 3600)
    out = root / 'report'
    h.verify_artifacts(out, json.loads((out / 'MANIFEST.json').read_text())['files'])
    report = (out / 'report.md').read_text()
    assert '交付：P' in report and '达上限无 EOS' in report and 'Holm' in report
    assert (out / 'T1.csv').read_text().splitlines()[0].startswith('endpoint,J,')
    with pytest.raises(FileExistsError):
        _build(root, runs)


def test_debias_rescoring_matches_frozen_score_and_cap_rule(built):
    root, rooms, runs = built
    data = root / 'data'
    rows = _jsonl(root / 'geometry-v2' / 'joint-geometry.jsonl')
    requests = {r['uid']: r for r in _jsonl(data / 'joint' / 'requests.jsonl')}
    stored = {r['uid']: r for r in _jsonl(data / 'rooms.jsonl')}
    for row in rows:
        derived = R._derive((row, stored[row['uid']], requests[row['uid']], 1.))
        assert derived['success_debiased'] == (row['geometry_success'] if row['complete'] else False)
    capped = R.cap_failure(next(r for r in rows if r['uid'] == 'x-05'))
    assert capped['complete'] is False and capped['size_log_mae'] is None and capped['error'] == R.CAP


@pytest.mark.parametrize('kind', ['reference_data', 'geometry_code', 'fid_lineage', 'agent_runs'])
def test_refuses_unbound_inputs(built, kind):
    root, _, runs = built
    if kind in ('reference_data', 'geometry_code'):
        value = json.loads((root / 'reference.json').read_text())
        if kind == 'reference_data':
            value['data_manifest_sha256'] = 'b' * 64
        else:
            value['code_sha256']['geometry.py'] = 'c' * 64
        (root / 'reference.json').unlink()
        infer.write_once(root / 'reference.json', value)
    elif kind == 'fid_lineage':
        value = json.loads((root / 'fid.json').read_text())
        value['provenance']['provenance']['geometry_manifest_sha256'] = 'd' * 64
        h.write_json(root / 'fid.json', value)
    else:
        path = root / 'llm-eval' / 'summary.json'
        value = json.loads(path.read_text())
        value['inference_run_sha256']['joint'] = 'e' * 64
        h.write_json(path, value)
        manifest = json.loads((root / 'llm-eval' / 'MANIFEST.json').read_text())
        h.write_json(root / 'llm-eval' / 'MANIFEST.json', {'files': {**manifest['files'], 'summary.json': h.sha256(path)}})
    with pytest.raises(ValueError):
        _build(root, runs, out='bad')
    assert not (root / 'bad').exists()


def test_contrasts_reuse_paired_intervals_design_and_draws():
    rng = random.Random(7)
    joint, placement = [], []
    for i in range(80):
        base = {'uid': f'u{i}', 'source': 'ABC'[i % 3], 'group': f'g{i // 3}'}
        for rows in (joint, placement):
            complete = rng.random() > .15
            rows.append({**base, **{k: (rng.random() if complete else None) for k in g.SCALARS},
                         'complete': complete, 'geometry_success': complete and rng.random() > .3,
                         'geometry_success_repaired': complete})
    ours = R.contrasts(joint, placement, ('geometry_success', 'size_log_mae'), 300)
    theirs = g.paired_intervals(joint, placement, repeats=300)
    for field in ours:
        assert ours[field]['estimate'] == theirs[field]['estimate_joint_minus_placement']
        assert ours[field]['ci95'] == theirs[field]['ci95'] and ours[field]['n'] == theirs[field]['paired_rooms']
        assert 0 <= ours[field]['p'] <= 1


def test_delivery_rule_and_holm():
    good = {'estimate': -.1, 'ci95': [-.2, -.01]}
    assert R.delivery(good, {'estimate': 0., 'ci95': [-.02, .01]})['delivered'] == 'J'
    for b, a1 in ((good, {'estimate': 0., 'ci95': [-.0201, .01]}), ({'estimate': -.1, 'ci95': [-.2, .01]}, {'ci95': [0., .1]}),
                  ({'estimate': .1, 'ci95': [.05, .2]}, {'ci95': [0., .1]}), ({'estimate': None, 'ci95': None}, {'ci95': [0., .1]}),
                  (good, {'estimate': None, 'ci95': None})):
        assert R.delivery(b, a1)['delivered'] == 'P'
    assert R.holm([.01, .04]) == [.02, .04] and R.holm([.03, .02]) == [.04, .04] and R.holm([None, .01]) == [None, None]


def test_rotated_oor_uses_rotated_footprints():
    square = {'size': [1, 1, 1], 'pos': [2, 2, 0], 'yaw': 0.}
    pair = [square, {**square, 'yaw': math.pi / 4}]
    assert R.rotated_oor(pair) == pytest.approx((2 * (math.sqrt(2) - 1)) / 2)
    room = {'objects': [{**o, 'id': str(i), 'category': 'box'} for i, o in enumerate(pair)]}
    assert check(room)['oor'] == pytest.approx(.5)
    assert R.rotated_oor([square, {**square, 'pos': [2.5, 2, 0]}]) == pytest.approx(.25)


def test_reference_prior_levels_exact_with_frozen_prior():
    prior = {'min_count': 5, 'global': {'count': 50, 'size': [.4, .5, .65]},
             'category_room_type': {'chair': {'bedroom': {'count': 6, 'size': [.5, .5, .9]},
                                              'kitchen': {'count': 2, 'size': [.3, .3, .3]}}},
             'category': {'chair': {'count': 8, 'size': [.45, .5, .85]}, 'lamp': {'count': 1, 'size': [.2, .2, .2]}}}
    assert RR._level({'category': 'chair', 'size': [.5, .5, .9]}, 'bedroom', prior) == 'P'
    assert RR._level({'category': 'chair', 'size': [.45, .5, .85]}, 'kitchen', prior) == 'C'
    assert RR._level({'category': 'lamp', 'size': [.4, .5, .65]}, 'bedroom', prior) == 'G'
    assert RR._level({'category': 'lamp', 'size': [.4, .5, .65]}, 'bedroom', None) == 'G'
    assert RR._level({'category': 'chair', 'size': [.5, .5, .9]}, 'bedroom', None) == 'N'
    with pytest.raises(ValueError, match='reproduce'):
        RR._level({'category': 'chair', 'size': [.5, .5, .91]}, 'bedroom', prior)


def test_reference_load_rebuilds_rooms_and_checks_hashes(tmp_path):
    raw = {'uid': 'r1', 'source': 'S', 'group': 'house', 'room_type': 'bedroom', 'boundary_type': 'polygon',
           'boundary': [[0, 0], [5, 0], [5, 5], [0, 5]], 'height': 3, 'split': 'test',
           'objects': [{'id': 'a', 'category': 'chair', 'size': [1, 1, 1], 'pos': [1, 1, 0], 'yaw': 0}], 'fixed': []}
    with gzip.open(tmp_path / 'export.jsonl.gz', 'wt') as stream:
        stream.write(json.dumps(raw) + '\n')
    bound, _ = bind_room(raw, seed=RR.SEED)
    room = {**bound, 'group': 'c1', 'component': 'c1', 'split': 'test'}
    meta = {'uid': 'r1', 'source': 'S', 'group': 'c1', 'component': 'c1', 'split': 'test', 'room_type': 'bedroom'}
    sizes = {o['id']: [.4, .5, .65] for o in room['objects']}
    write_rows(tmp_path / 'j.jsonl', [{**meta, 'messages': messages(room, 'joint')[:2]}])
    write_rows(tmp_path / 'p.jsonl', [{**meta, 'messages': messages(room, 'placement', estimated_sizes=sizes)[:2]}])
    (tmp_path / 'rooms.jsonl').write_text(json.dumps(room, ensure_ascii=False) + '\n')
    files = {'joint/requests.jsonl': h.sha256(tmp_path / 'j.jsonl'), 'placement/requests.jsonl': h.sha256(tmp_path / 'p.jsonl'),
             'rooms.jsonl': h.sha256(tmp_path / 'rooms.jsonl')}
    h.write_json(tmp_path / 'MANIFEST.json', {'expected_uids': ['r1'], 'files': files})
    rooms, requests, _ = RR.load(tmp_path / 'export.jsonl.gz', tmp_path / 'j.jsonl', tmp_path / 'p.jsonl', tmp_path / 'MANIFEST.json')
    reference = RR.build_reference(rooms, requests, 'f' * 64, [])
    assert reference['prior_levels'] == {'r1': 'G'} and reference['scopes']['all']['prior']['global_fallback_object_rate'] == 1
    h.write_json(tmp_path / 'MANIFEST.json', {'expected_uids': ['r1'], 'files': {**files, 'rooms.jsonl': '0' * 64}})
    with pytest.raises(ValueError, match='rooms'):
        RR.load(tmp_path / 'export.jsonl.gz', tmp_path / 'j.jsonl', tmp_path / 'p.jsonl', tmp_path / 'MANIFEST.json')
