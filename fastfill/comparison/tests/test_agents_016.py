"""Amendment 016 section 2: size enforcement (addendum B2), GT-size requests, L on P's input over a subset, the pilot
subsample and the size-free vs size-fixed comparison (fixture S1 outputs, no GPU, no network)."""
import json
import math
import shutil

import numpy as np
import pytest

from fastfill.comparison import agents_016 as A
from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import geometry as g
from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import output_repair as r
from fastfill.comparison import scale_curve as sc
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.tests import test_llm_agent as base
from fastfill.comparison.tests import test_test_apply_012 as t12w


def _supported_room():
    raw = {'uid': 'u', 'source': 'S', 'group': 'g', 'boundary_type': 'polygon', 'height': 3,
           'boundary': [[0, 0], [5, 0], [5, 5], [0, 5]],
           'fixed': [{'id': 'shelf', 'category': 'shelf', 'size': [1, .5, 1], 'pos': [1, 1, 0], 'yaw': 0.0,
                      'parent': None}],
           'objects': [{'id': 'table', 'category': 'table', 'size': [1, 1, .75], 'pos': [3, 3, 0], 'yaw': 0.0,
                        'parent': None},
                       {'id': 'cup', 'category': 'cup', 'size': [.1, .1, .1], 'pos': [3, 3, .75], 'yaw': 0.0,
                        'parent': 'table'},
                       {'id': 'lamp', 'category': 'lamp', 'size': [.3, .3, .5], 'pos': [1, 1, 1], 'yaw': 0.0,
                        'parent': 'shelf'}]}
    return bind_room(raw)[0]


def test_enforce_sizes_overwrites_sizes_and_recomputes_supported_z():
    room = _supported_room()
    ids = {o['category']: o['id'] for o in room['objects'] + room['fixed']}
    prior = {ids['table']: [1.2, 1.2, .8], ids['cup']: [.1, .1, .12], ids['lamp']: [.3, .3, .5]}
    record = {'uid': 'u', 'source': 'S', 'group': 'g', 'messages': messages(room, 'placement', prior)[:2]}
    gt_answer = json.loads(base.target(None, 'joint', room))  # Own (GT) sizes; the cup sits on a 0.75 m table.
    by = {p['id']: p for p in gt_answer['placements']}
    assert by[ids['cup']]['pos'][2] == .75 and by[ids['lamp']]['on'] == ids['shelf']
    text, counts = L.enforce_sizes(json.dumps(gt_answer), record)
    assert counts == {'objects': 3, 'size_returned': 3, 'size_changed': 2, 'z_changed': 1}  # lamp size = prior
    fixed = {p['id']: p for p in json.loads(text)['placements']}
    assert all('size' not in p for p in fixed.values())
    assert fixed[ids['cup']]['pos'] == [*by[ids['cup']]['pos'][:2], .8]          # on the 0.8 m prior table
    assert fixed[ids['lamp']]['pos'] == by[ids['lamp']]['pos']                   # fixed shelf top = 1.0: unchanged
    assert fixed[ids['table']]['pos'] == by[ids['table']]['pos']                 # floor object keeps its z
    assert L.parse(json.dumps(gt_answer), record, 'placement')['valid']
    assert not infer.parse_prediction(json.dumps(gt_answer), record, 'placement')['valid']  # P's own contract
    placed = L.parse(json.dumps(gt_answer), record, 'placement')['placements']
    assert [p['size'] for p in placed] == [prior[p['id']] for p in placed]
    compliant = json.dumps({'placements': [{k: v for k, v in p.items() if k != 'size'} | (
        {'pos': [*p['pos'][:2], .8]} if p['id'] == ids['cup'] else {}) for p in gt_answer['placements']]})
    assert L.enforce_sizes(compliant, record) == (compliant, {'objects': 3, 'size_returned': 0, 'size_changed': 0,
                                                              'z_changed': 0})
    for bad in ('nope', '[]', '{"placements": [1]}'):
        assert L.enforce_sizes(bad, record) == (bad, None)
    cycle = json.loads(compliant)
    cycle['placements'][0]['on'] = cycle['placements'][1]['id']
    cycle['placements'][1]['on'] = cycle['placements'][0]['id']
    text, counts = L.enforce_sizes(json.dumps(cycle), record)
    assert counts['z_changed'] == 0 and not L.parse(text, record, 'placement')['valid']  # left for the parser
    assert L.parse(json.dumps(gt_answer), record, 'joint') == infer.parse_prediction(json.dumps(gt_answer), record, 'joint')


def test_gt_request_differs_from_p_only_in_sizes():
    room = _supported_room()
    prior = {o['id']: [.5, .5, .5] for o in room['objects']}
    record = {'uid': 'u', 'source': 'S', 'group': 'g', 'messages': messages(room, 'placement', prior)[:2]}
    gt = L.gt_request(record, room)
    sizes = {o['id']: o['size'] for o in json.loads(gt['messages'][1]['content'])['objects']}
    assert sizes == {o['id']: o['size'] for o in room['objects']} and gt['messages'][0] == record['messages'][0]
    with pytest.raises(ValueError, match='beyond its sizes'):
        L.gt_request(record, {**room, 'height': 4})


@pytest.mark.parametrize('runner_error', [None, 'INTERNAL_ERROR'])
def test_rescoring_keeps_parser_failures_repairable_and_runner_failures_failed(runner_error):
    room = _supported_room()
    record = {k: room[k] for k in ('uid', 'source', 'group')}
    request = {**record, 'messages': messages(room, 'placement', {o['id']: o['size'] for o in room['objects']})[:2]}
    raw = base.target(None, 'joint', room)
    old_schema = infer.parse_prediction(raw, request, 'placement')
    assert not old_schema['valid']  # Old strict parser rejected the now-allowed returned sizes.
    row = {**record, 'arm': 'llm_harness', 'raw': raw, 'schema': old_schema, 'error': runner_error}
    enforced, _ = ev._enforced([row], {room['uid']: request})
    scored = ev._score_llm((room, request, enforced[0], 'placement'))
    assert scored['complete'] is (runner_error is None)
    assert scored['error'] == runner_error


@pytest.mark.parametrize('field', ['pos', 'yaw'])
@pytest.mark.parametrize('number', ['1e309', '-1e309'])
def test_size_enforcement_leaves_nonfinite_geometry_as_a_parse_failure(field, number):
    room = _supported_room()
    request = {'messages': messages(room, 'placement', {o['id']: o['size'] for o in room['objects']})[:2]}
    value = json.loads(base.target(None, 'joint', room))
    if field == 'pos':
        value['placements'][0]['pos'][0] = 'overflow'
    else:
        value['placements'][0]['yaw'] = 'overflow'
    raw = json.dumps(value).replace('"overflow"', number)
    parsed = L.parse(raw, request, 'placement')
    assert not parsed['valid'] and parsed['placements'] is None
    assert L.enforce_sizes(raw, request) == (raw, None)


@pytest.mark.parametrize('field', ['size', 'pos'])
@pytest.mark.parametrize('sign', [-1, 1])
def test_size_enforcement_handles_integers_too_large_for_float(field, sign):
    room = _supported_room()
    prior = {o['id']: [.5, .5, .5] for o in room['objects']}
    request = {'messages': messages(room, 'placement', prior)[:2]}
    value = json.loads(base.target(None, 'joint', room))
    # A huge parent z used to overflow when the supported object's bottom was recomputed.
    value['placements'][0][field][2 if field == 'pos' else 0] = sign * 10 ** 400
    parsed = L.parse(json.dumps(value), request, 'placement')
    if field == 'pos':
        assert not parsed['valid'] and parsed['placements'] is None
    else:  # B2 ignores returned sizes, including unusable values, and uses the request's prior.
        assert parsed['valid'] and all(p['size'] == prior[p['id']] for p in parsed['placements'])


class PlacementApi(base.FakeApi):
    """FakeApi answering P's request: 'valid' = P's GT answer; 'joint' = the GT answer with its (GT) sizes."""
    def uid(self, messages):
        matches = [uid for uid, room in self.rooms.items() if messages[:2] == room['_placement']['messages']]
        assert len(matches) == 1
        return matches[0]

    def __call__(self, config, messages, max_output_tokens, timeout):
        uid = self.uid(messages)
        item = (self.script.get(uid) or [self.default])[0]
        if item in ('valid', 'joint'):
            self.script[uid] = [base.target(None, 'placement' if item == 'valid' else 'joint', self.rooms[uid])]
        return super().__call__(config, messages, max_output_tokens, timeout)


def test_l_runs_p_input_on_a_subset_and_keeps_its_default_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(L, 'BACKOFF_SECONDS', 0)
    data = base.eval_data(tmp_path / 'data')
    base.env_file(tmp_path)
    L.main(['cohort', '--data', str(data), '--out', str(tmp_path / 'cohort.json')])
    rooms = {x['uid']: x for x in map(json.loads, (data / 'rooms.jsonl').read_text().splitlines())}
    for arm, key in (('joint', '_request'), ('placement', '_placement')):
        for request in map(json.loads, (data / f'{arm}/requests.jsonl').read_text().splitlines()):
            rooms[request['uid']][key] = request
    cohort = json.loads((tmp_path / 'cohort.json').read_text())['uids']
    (tmp_path / 'pilot.json').write_text(json.dumps({'uids': [cohort[3], cohort[0]]}))
    api = PlacementApi(rooms, {cohort[0]: ['joint'], cohort[3]: ['bad', 'valid']})
    result = L.run(base.args(tmp_path, arm='placement', uids=tmp_path / 'pilot.json'), api)
    assert result['status'] == 'LLM_COMPLETE' and result['rooms'] == 2 and result['valid_first'] == 1
    assert sorted(u for u, _ in api.calls) == sorted([cohort[0], cohort[3], cohort[3]])  # one repair round
    identity = json.loads((tmp_path / 'run/run_receipt.json').read_text())
    assert identity['config']['request_arm'] == 'placement' and 'enforce_sizes' in identity['config']['parser']
    assert identity['data']['placement_requests_sha256'] == h.sha256(data / 'placement/requests.jsonl')
    assert identity['cohort']['subset']['uids_sha256'] == L._digest(json.dumps([cohort[0], cohort[3]]))
    harness = {x['uid']: x for x in map(json.loads, (tmp_path / 'run/llm_harness/records.jsonl').read_text().splitlines())}
    prior = [o['size'] for o in json.loads(rooms[cohort[0]]['_placement']['messages'][1]['content'])['objects']]
    assert [p['size'] for p in harness[cohort[0]]['schema']['placements']] == prior
    # Default (joint, whole cohort): no amendment-016 key enters the identity.
    joint = L.run(base.args(tmp_path, out=tmp_path / 'joint', limit=1), base.FakeApi(rooms))
    assert joint['status'] == 'LLM_COMPLETE'
    identity = json.loads((tmp_path / 'joint/run_receipt.json').read_text())
    assert identity['config'] == {'model': L.EXPECTED_MODEL, 'reasoning_effort': L.EFFORT, 'max_output_tokens': 65536,
                                  'repair_rounds': 2, 'suffix': L.SUFFIX,
                                  'input': 'joint_system_user_verbatim_then_suffix_user',
                                  'parser': 'infer.parse_prediction(raw, joint_request, joint)',
                                  'repair_message': 'parser_error_text_only', 'limit': 1}
    assert list(identity['data']) == ['manifest_sha256', 'joint_requests_sha256']
    assert list(identity['cohort']) == ['sha256', 'uids_sha256', 'rooms']


def test_pilot_is_stratified_proportional_hash_ranked_and_write_once(tmp_path, monkeypatch):
    one = lambda room: {**room, 'objects': room['objects'][:1]} if room['source'] == 'X' else room
    data = base.eval_data(tmp_path / 'data', one)
    L.main(['cohort', '--data', str(data), '--out', str(tmp_path / 'cohort.json')])
    cohort = json.loads((tmp_path / 'cohort.json').read_text())
    monkeypatch.setattr(sc, 'BINS', (('one', 1, 1), ('two', 2, None)))
    monkeypatch.setattr(A, 'PILOT_SIZE', 7)
    value = A.pilot(data, tmp_path / 'cohort.json', tmp_path / 'pilot.json')
    ones = [u for u in cohort['uids'] if u.startswith('x-')]
    twos = [u for u in cohort['uids'] if not u.startswith('x-')]
    exact = {'one': 7 * len(ones) / len(cohort['uids']), 'two': 7 * len(twos) / len(cohort['uids'])}
    quota = {k: math.floor(v) for k, v in exact.items()}
    quota[max(exact, key=lambda k: exact[k] - quota[k])] += 7 - sum(quota.values())
    assert value['counts'] == {'one': {'cohort': len(ones), 'pilot': quota['one']},
                               'two': {'cohort': len(twos), 'pilot': quota['two']}}
    rank = lambda pool, n: set(sorted(pool, key=lambda u: (L._digest('a016-pilot:' + u), u))[:n])
    assert set(value['uids']) == rank(ones, quota['one']) | rank(twos, quota['two'])
    assert value['uids'] == [u for u in cohort['uids'] if u in set(value['uids'])]
    assert L.subset(cohort, tmp_path / 'pilot.json') == value['uids']
    assert A.pilot(data, tmp_path / 'cohort.json', tmp_path / 'pilot.json') == value  # verified, not rewritten
    monkeypatch.setattr(A, 'PILOT_SIZE', 6)
    with pytest.raises(ValueError, match='differs from its deterministic rebuild'):
        A.pilot(data, tmp_path / 'cohort.json', tmp_path / 'pilot.json')


def test_tau_b():
    assert A.tau(np.array([3., 2, 1]), np.array([3., 2, 1])) == 1
    assert A.tau(np.array([3., 2, 1]), np.array([1., 2, 3])) == -1
    assert A.tau(np.array([3., 2, 1]), np.array([3., 1, 2])) == pytest.approx(1 / 3)
    assert A.tau(np.array([2., 2, 1]), np.array([3., 2, 1])) == pytest.approx(2 / math.sqrt(6))
    assert np.isnan(A.tau(np.array([1., 1, 1]), np.array([3., 2, 1])))
    assert A.tau(np.array([[3., 2, 1], [1, 2, 3]]), np.array([[3., 2, 1], [3, 2, 1]])).tolist() == [1, -1]


def _fixed_world(tmp_path, monkeypatch):
    """test_apply_012's fixture (S1 of P and of L's size-free answers on a 3-room cohort) plus L size-fixed S1
    (placement answers with sizes from P's request) and a copy of L as a second agent M."""
    monkeypatch.setitem(sc.SOURCE_KIND, 'S', ('real', 'fixture source'))
    paths = t12w.world(tmp_path)
    data, cohort, s1 = paths['data'], paths['cohort_path'], paths['s1_root']
    rooms = [json.loads(x) for x in (data / 'rooms.jsonl').read_text().splitlines()]
    requests = [json.loads(x) for x in (data / 'placement/requests.jsonl').read_text().splitlines()]
    evaluation = tmp_path / 'llm-eval-L-P-v1'
    rows = {v: [{**g.score(x, q['messages'], t12w._answer(x, 'placement'), 'placement'), 'arm': v}
                for x, q in zip(rooms, requests) if x['uid'] in ('a', 'b', 'c')] for v in L.VARIANTS}
    t12w._publish(evaluation, {f'{v}-geometry.jsonl': t12w._jsonl(rows[v]) for v in rows})
    fixed = tmp_path / 's1-016-size-fixed'
    for v in A.VARIANTS:
        r.run(data, evaluation / f'llm_{v}-geometry.jsonl', 'placement', fixed / f'agent-L-P-{v}', uids=cohort)
        shutil.copytree(s1 / f'agent-L-{v}', s1 / f'agent-M-{v}')
        shutil.copytree(fixed / f'agent-L-P-{v}', fixed / f'agent-M-P-{v}')
    return paths, fixed


def test_compare_ranks_both_settings_with_tau_differences_and_flips(tmp_path, monkeypatch):
    paths, fixed = _fixed_world(tmp_path, monkeypatch)
    args = dict(data=paths['data'], cohort_path=paths['cohort_path'], s1_012=paths['s1_root'],
                fixed={'L': fixed / 'agent-L-P', 'M': fixed / 'agent-M-P'}, repeats=20)
    result = A.compare(uids_path=paths['cohort_path'], out=tmp_path / 'cmp', **args)
    assert sorted(p.name for p in (tmp_path / 'cmp').iterdir()) == ['MANIFEST.json', 'contrasts.csv', 'levels.csv',
                                                                   'report.md', 'results.json']
    assert result['scope'] == {'rooms': 3, 'cohort_rooms': 3, 'uids_sha256': L._digest(json.dumps(['a', 'b', 'c'])),
                               'uids_file_sha256': h.sha256(paths['cohort_path']), 'is_cohort': True}
    s = result['subsets']['all']
    assert (s['rooms'], s['components']) == (3, 2) and {'objects<=20', 'real', 'synthetic'} <= set(result['subsets'])
    harness = s['harness']
    assert list(harness['levels']) == ['P', 'L-harness', 'M-harness', 'L-P-harness', 'M-P-harness']
    level = lambda label, stage, f: harness['levels'][label][stage][f]['estimate']
    for stage in A.STAGES:
        for f in A.FIELDS:   # size-free minus size-fixed = the difference of the two rates on the same rooms
            assert harness['size_free_minus_fixed']['L'][stage][f]['estimate'] == pytest.approx(
                level('L-harness', stage, f) - level('L-P-harness', stage, f))
        ranks = harness['rankings'][stage]
        assert [n for n, _ in ranks['size_free']] == sorted(['P', 'L', 'M'], key=lambda n: -dict(ranks['size_free'])[n])
        assert dict(ranks['size_fixed'])['P'] == dict(ranks['size_free'])['P'] == level('P', stage, 'downstream')
        point = A.tau(np.array([dict(ranks['size_free'])[n] for n in ('P', 'L', 'M')]),
                      np.array([dict(ranks['size_fixed'])[n] for n in ('P', 'L', 'M')]))
        tau = harness['kendall_tau'][stage]
        assert tau['systems'] == ['P', 'L', 'M'] and (tau['estimate'] is None) == bool(np.isnan(point))
        if tau['estimate'] is not None:
            assert tau['estimate'] == pytest.approx(point)
    assert harness['levels']['L-harness']['raw']['bias_fp'] == pytest.approx(2 * math.log(.5))  # own (halved) sizes
    assert harness['levels']['L-P-harness']['raw']['bias_fp'] == pytest.approx(0.0)              # P's (GT here)
    flips = {(f['comparison'], f['setting'], f['stage']): f['fields'] for f in harness['flips']}
    assert set(flips) == {('P - L', setting, stage) for setting in ('size_free', 'size_fixed') for stage in A.STAGES}
    assert flips['P - L', 'size_free', 'raw']['downstream']['estimate'] == pytest.approx(
        level('P', 'raw', 'downstream') - level('L-harness', 'raw', 'downstream'))
    report = (tmp_path / 'cmp/report.md').read_text()
    assert 'EXPLORATORY' in report and '全队列 3 间' in report and '| P - L | harness | 自选尺寸（J 输入） | raw |' in report
    assert 'OptiScene − P 未计算' in report
    with pytest.raises(FileExistsError):
        A.compare(uids_path=paths['cohort_path'], out=tmp_path / 'cmp', **args)
    # The pilot scope (here rooms a and c) restricts every system; a size-fixed run must cover it.
    (tmp_path / 'pilot.json').write_text(json.dumps({'uids': ['c', 'a']}))
    pilot = A.compare(uids_path=tmp_path / 'pilot.json', out=tmp_path / 'cmp-pilot', **args)
    assert pilot['scope']['rooms'] == 2 and not pilot['scope']['is_cohort']
    assert pilot['subsets']['all']['raw']['levels']['P']['raw']['complete']['estimate'] == 1.0
    (tmp_path / 'ab.json').write_text(json.dumps(['a', 'b']))
    short = tmp_path / 'llm-eval-N-P-v1'
    t12w._publish(short, {f'{v}-geometry.jsonl': ''.join(x + '\n' for x in (
        tmp_path / f'llm-eval-L-P-v1/{v}-geometry.jsonl').read_text().splitlines()[:2]) for v in L.VARIANTS})
    for v in A.VARIANTS:
        r.run(paths['data'], short / f'llm_{v}-geometry.jsonl', 'placement', fixed / f'agent-N-P-{v}',
              uids=tmp_path / 'ab.json')
        shutil.copytree(paths['s1_root'] / f'agent-L-{v}', paths['s1_root'] / f'agent-N-{v}')
    with pytest.raises(ValueError, match='N-P-harness: S1 rows do not cover'):
        A.compare(uids_path=paths['cohort_path'], out=tmp_path / 'cmp-n', **{**args, 'fixed': {'N': fixed / 'agent-N-P'}})
    assert not (tmp_path / 'cmp-n').exists()
