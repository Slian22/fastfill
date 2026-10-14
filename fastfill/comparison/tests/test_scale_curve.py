"""Amendment 016 §1/§6 scale-pass curves on synthetic rooms: a shrinking (tightly packed) system has a steeper curve
than a placement-only one and loses with prior / GT sizes, children keep their support, incomplete and capped rows
fail at every s, the OOB-tolerance variants are monotone, rule shares raw and after S1; and the CPU job."""
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

from fastfill.comparison import geometry as g
from fastfill.comparison import llm_agent
from fastfill.comparison import scale_curve as S
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.validate import support_ok

CODE = Path(__file__).resolve().parents[3]
SCRIPT = CODE / 'fastfill/comparison/submit_016_scale.sh'
PROJECT = '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill'
GAPS = (.70, .74, .78, .82)        # J's cabinet centre distances (its widths are 0.8): overlap resolves at s < 1
RUNS = {'joint': 'j' * 64, 'placement': 'p' * 64}
ROOMS = 20                         # houses h0..h9: the all-test intervals exist (report_v2.NO_CI = 10)


def obj(i, category, size, pos):
    return {'id': i, 'category': category, 'size': list(size), 'pos': list(pos), 'yaw': 0.0, 'parent': None}


def _room(k):
    """6 x 4 m; two cabinets, cabinet a 5 mm (k % 4 == 1) or 5 cm (k % 4 == 2) through the west wall; odd k also a
    chair tucked under a table. Houses h0..h9 (two rooms each), alternately a real and a synthetic source."""
    width = {1: 1.01, 2: 1.10}.get(k % 4, 1.0)
    objects = [obj('a', 'cabinet', (width, .6, 1), (.5 if k % 4 in (1, 2) else 1.5, 2, 0)),
               obj('b', 'cabinet', (1, .6, 1), (4.5, 2, 0))]
    if k % 2:
        objects += [obj('t', 'table', (1.2, .8, .75), (3, 3.2, 0)), obj('c', 'chair', (.5, .5, .9), (3, 2.9, 0))]
    return bind_room({'uid': f'r{k}', 'source': ('SpatialLM', 'MansionWorld')[k // 2 % 2], 'group': f'h{k // 2}',
                      'boundary_type': 'polygon', 'room_type': 'office', 'boundary': [[0, 0], [6, 0], [6, 4], [0, 4]],
                      'height': 3, 'objects': objects, 'fixed': []})[0]


def _answer(room, arm, gap=None):
    """joint: x/y sizes x0.8 (cabinets 0.8 x 0.48), the cabinets `gap` apart about x = 3 (tight); placement: the
    cabinets well apart."""
    cabinets = [o['id'] for o in room['objects'] if o['category'] == 'cabinet']
    size = lambda o: [.8, .48, 1] if o['category'] == 'cabinet' else [x * .8 for x in o['size'][:2]] + o['size'][2:]

    def pos(o):
        if o['category'] != 'cabinet':
            return o['pos']
        side = -1 if o['id'] == cabinets[0] else 1
        return [3 + side * gap / 2, 1, 0] if gap is not None else [3 + side * 1.5, 1, 0]
    return json.dumps({'placements': [{'id': o['id'], 'pos': pos(o), 'yaw': 0,
                                       **({'size': size(o)} if arm == 'joint' else {})} for o in room['objects']]})


def _publish(root, files):
    root.mkdir(parents=True)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    h.write_json(root / 'MANIFEST.json', {'files': h.artifact_manifest(root)})


def _jsonl(rows):
    return ''.join(json.dumps(x) + '\n' for x in rows)


def world(tmp):
    """20 test rooms; J tightly packed and shrunk, P placed with clearance; agent L on a 6-room cohort (harness = P-like
    joint answers, raw: one room missing, one capped); one extra rows file holding two arms."""
    rooms = [_room(k) for k in range(ROOMS)]
    sizes = lambda x: {o['id']: o['size'] for o in x['objects']}
    req = {arm: [{'uid': x['uid'], 'source': x['source'], 'group': x['group'],
                  'messages': messages(x, arm, sizes(x) if arm == 'placement' else None)[:2]} for x in rooms]
           for arm in h.ARMS}
    data = tmp / 'data'
    _publish(data, {'rooms.jsonl': _jsonl(rooms), **{f'{arm}/requests.jsonl': _jsonl(req[arm]) for arm in h.ARMS}})
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    h.write_json(data / 'MANIFEST.json', {**manifest, 'expected_uids': [x['uid'] for x in rooms]})
    data_sha, code = h.sha256(data / 'MANIFEST.json'), {'geometry.py': h.sha256(g.__file__)}
    score = lambda x, q, arm, label, raw: {**g.score(x, q['messages'], raw, arm), 'arm': label}
    answers = {'joint': [_answer(x, 'joint', GAPS[k % 4]) for k, x in enumerate(rooms)],
               'placement': [_answer(x, 'placement') for x in rooms]}
    geo = tmp / 'geometry-v5'
    _publish(geo, {**{f'{arm}-geometry.jsonl': _jsonl(score(x, q, arm, arm, a) for x, q, a in
                                                      zip(rooms, req[arm], answers[arm])) for arm in h.ARMS},
                   'summary.json': json.dumps({'status': 'GEOMETRY_COMPLETE', 'data_manifest_sha256': data_sha,
                                               'inference_run_sha256': RUNS, 'code_sha256': code})})
    uids = [f'r{k}' for k in range(6)]
    cohort = tmp / 'llm-cohort-v1.json'
    cohort.write_text(json.dumps({'version': 1, 'uids_sha256': llm_agent._digest(json.dumps(uids)), 'uids': uids}))
    loose = [json.dumps({'placements': [{**p, 'size': o['size']} for p, o in zip(json.loads(a)['placements'],
                                                                                  x['objects'])]})
             for x, a in zip(rooms, answers['placement'])]
    rows = {'llm_harness': [score(x, q, 'joint', 'llm_harness', a) for x, q, a in zip(rooms, req['joint'], loose)],
            'llm_raw': [score(x, q, 'joint', 'llm_raw', a if x['uid'] != 'r2' else None)
                        for x, q, a in zip(rooms, req['joint'], loose)]}
    for x in rows['llm_raw']:
        x |= {'reached_token_cap': x['uid'] == 'r3', 'stopped_on_eos': x['uid'] != 'r3'}
    agent = tmp / 'llm-eval-L-selected-v1'
    _publish(agent, {**{f'{v}-geometry.jsonl': _jsonl(x for x in rows[v] if x['uid'] in uids) for v in rows},
                     'summary.json': json.dumps(_agent_summary(data_sha, uids))})
    extra = tmp / 'optiscene-score'
    _publish(extra, {'e2-rows.jsonl': _jsonl([{**score(x, q, 'placement', 'os', a), 'mode': mode}
                                              for x, q, a in zip(rooms[:4], req['placement'], answers['placement'])
                                              for mode in ('prior', 'gt')])})
    return {'data': data, 'geometry': geo, 'cohort': cohort, 'agents': {'L': agent},
            'extras': {'OS': f'{extra}/e2-rows.jsonl::arm=os,mode=prior'}}


def _agent_summary(data_sha, uids, **config):
    return {'status': 'LLM_COMPARISON_COMPLETE', 'data_manifest_sha256': data_sha, 'inference_run_sha256': RUNS,
            'code_sha256': {'geometry.py': h.sha256(g.__file__)}, 'llm': {'config': config},
            'cohort': {'uids_sha256': llm_agent._digest(json.dumps(uids))}}


def test_scaled_keeps_bottom_centres_parents_and_support_gaps():
    objects = [obj('t', 'table', (1, 1, .8), (1, 1, 0)), {**obj('b', 'box', (.4, .4, .2), (1, 1, .8)), 'parent': 't'},
               {**obj('c', 'cup', (.1, .1, .1), (1.1, 1, 1.03)), 'parent': 'b'},          # 3 cm gap above the box
               {**obj('v', 'vase', (.2, .2, .3), (5, 5, 1)), 'parent': 'fixed_0001'}]
    xy, xyz = S.scaled(objects, .5, 'xy'), S.scaled(objects, .5, 'xyz')
    assert [o['size'] for o in xy] == [[.5, .5, .8], [.2, .2, .2], [.05, .05, .1], [.1, .1, .3]]
    assert [o['pos'] for o in xy] == [[1, 1, 0], [1, 1, .8], pytest.approx([1.05, 1, 1.03]), [5, 5, 1]]
    assert [o['parent'] for o in xyz] == [None, 't', 'b', 'fixed_0001']
    assert [o['size'][2] for o in xyz] == [.4, .1, .05, .15]
    assert [o['pos'][2] for o in xyz] == pytest.approx([0, .4, .53, 1])                 # gaps kept, fixed parent stays
    assert S.scaled(objects, 1.0, 'xyz') == objects and objects[1]['size'] == [.4, .4, .2]
    assert S.substituted(objects, {o['id']: o['size'] for o in objects}, 'xyz') == objects


def test_a_cup_near_a_table_edge_stays_supported():
    """Review 2026-10-14: with positions kept, s = 0.8 put this cup 7 cm off its table (5 cm tolerance)."""
    yaw = .3
    table = {**obj('t', 'table', (1, 1, .8), (1, 1, 0)), 'yaw': yaw}
    cup = {**obj('c', 'cup', (.1, .1, .1), (1 + .47 * math.cos(yaw), 1 + .47 * math.sin(yaw), .8)), 'parent': 't'}
    assert support_ok(cup, table)
    for variant in S.VARIANTS:
        t, c = S.scaled([table, cup], .8, variant)
        assert support_ok(c, t) and not support_ok({**c, 'pos': cup['pos'][:2] + c['pos'][2:]}, t)
        assert math.dist(c['pos'][:2], t['pos'][:2]) == pytest.approx(.8 * .47)
    turned = {**table, 'size': [2, 1, .8], 'yaw': math.pi / 2}                    # local x = world y
    t, c = S.substituted([turned, {**cup, 'pos': [1, 1.9, .8]}], {'t': [1, 1, 1], 'c': [.1, .1, .1]}, 'xyz')
    assert c['pos'] == pytest.approx([1, 1.45, 1]) and support_ok(c, t)            # same place on the table


def test_curves_slopes_substitution_rules_and_tolerances(tmp_path):
    paths = world(tmp_path)
    out = tmp_path / 'out'
    value = S.run(**paths, out=out, repeats=200)
    assert sorted(p.name for p in out.iterdir()) == ['MANIFEST.json', 'curves.csv', 'curves.png', 'reasons.csv',
                                                     'report.md', 'results.json', 'rows.jsonl', 'rules.csv',
                                                     'summary.csv', 'validity.csv']
    scopes = value['scopes']
    assert {k: (v['rooms'], v['systems']) for k, v in scopes.items()} == {
        'all-test': (20, ['GT', 'J', 'P']), 'cohort': (6, ['GT', 'J', 'P', 'L-raw', 'L-harness']),
        'extra-1': (4, ['GT', 'J', 'P', 'OS'])}
    subsets = scopes['all-test']['subsets']
    assert list(subsets) == ['all', 'objects<=20', 'real', 'synthetic', 'family:other_families',
                             'family:study_office']
    assert (subsets['all']['components'], subsets['all']['flags']) == (10, 'few clusters; small n')
    assert subsets['real']['components'] == 5 and all(                           # house rule: no CI below 10
        p['ci95'] is None for x in subsets['real']['systems'].values() for p in x['curves']['checker']['xy'])
    systems = subsets['all']['systems']
    rate = lambda x, m='checker', v='xy': [p['estimate'] for p in x['curves'][m][v]]
    j, p = rate(systems['J']), rate(systems['P'])
    assert j[S.GRID.index(.8)] == 1 and j[S.ONE] == .5 and j[-1] == 0 and p == [1.0] * len(S.GRID)
    slope = lambda x, m='checker': x['slope'][m]['xy']['estimate']
    assert slope(systems['J']) < slope(systems['P']) == pytest.approx(0)                  # shrinking: steeper
    assert slope(systems['J'], 'downstream') < slope(systems['P'], 'downstream')
    gt = systems['GT']
    assert gt['s_match']['checker']['xy']['estimate'] == 1.0 and gt['s_match']['checker']['xy']['gt_target'] == 1.0
    assert systems['J']['s_match']['checker']['xy']['estimate'] == pytest.approx(.9)   # J meets GT only shrunk to 0.9
    assert systems['J']['curves']['checker']['xy'][0]['ci95'] == [1.0, 1.0]
    swap = lambda x, src, v='xy', m='checker': x['substitution'][m][v][src]['own_minus']['estimate']
    for v in S.VARIANTS:                     # prior = GT sizes here; J's 0.8 x 1 m cabinets collide at full size
        assert swap(systems['J'], 'prior', v) == swap(systems['J'], 'gt', v) == .5
        assert systems['J']['substitution']['checker'][v]['prior']['pass']['estimate'] == 0
        assert swap(systems['P'], 'prior', v) == swap(systems['P'], 'gt', v) == swap(gt, 'gt', v) == 0
    rows = {(x['system'], x['uid']): x for x in map(json.loads, (out / 'rows.jsonl').read_text().splitlines())}
    assert systems['J']['bias_fp']['estimate'] == pytest.approx(statistics.fmean(
        rows['J', f'r{k}']['bias_fp'] for k in range(ROOMS))) and systems['J']['bias_fp']['estimate'] < -.4
    assert systems['P']['bias_fp']['estimate'] == gt['bias_fp']['estimate'] == 0
    v = {k: x['estimate'] for k, x in gt['validity'].items()}                        # 5 of 20 rooms 5 mm out, 5 5 cm
    assert (v['checker_oob_1mm'], v['checker_oob_1cm'], v['checker_oob_10cm']) == (.5, .75, 1.0)
    assert v['checker_oob_10cm'] == rate(gt)[S.ONE] and v['checker_tuck_counted'] == .5
    assert v['downstream_a4_collision'] == .5 and v['downstream'] <= .5               # A4 counts the tucked chair
    assert gt['rules']['raw']['downstream:A4_COLLISION'] == .5 and 'incomplete' not in gt['rules']['raw']
    assert systems['J']['rules'] == {'raw': {'checker:main_collisions': .5, **{
        k: s for k, s in systems['J']['rules']['raw'].items() if k.startswith('downstream:')}}}  # no S1 given
    for scope in scopes.values():
        for sub in scope['subsets'].values():
            for x in sub['systems'].values():
                w = {k: y['estimate'] for k, y in x['validity'].items()}
                assert w['checker_oob_1mm'] <= w['checker_oob_1cm'] <= w['checker_oob_10cm'] and \
                    w['checker_tuck_counted'] <= w['checker_oob_10cm']
    cohort = scopes['cohort']['subsets']['all']['systems']
    assert rate(cohort['L-harness']) == [1.0] * len(S.GRID)
    assert rate(cohort['L-raw']) == [4 / 6] * len(S.GRID)                          # r2 missing, r3 capped
    assert cohort['L-raw']['rules']['raw']['incomplete'] == 2 / 6
    for key in (('L-raw', 'r2'), ('L-raw', 'r3')):
        assert not rows[key]['complete'] and not any(any(c) for m in S.MEASURES for c in rows[key][m].values())
        assert rows[key]['bias_fp'] is None and not any(
            any(x.values()) for m in S.MEASURES for x in rows[key]['substitution'][m].values())
    assert rows['L-harness', 'r3']['complete'] and value['cap_rule'] == {
        'J': 'not applied: no inference runs given', 'P': 'not applied: no inference runs given',
        'L-raw': 'per_row', 'L-harness': 'per_row'}
    assert scopes['all-test']['reasons']['GT']['MansionWorld']['kind'] == 'synthetic'
    with (out / 'summary.csv').open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == sum(len(s['subsets'][k]['systems']) for s in scopes.values() for k in s['subsets']) * 4
    assert {'own_minus_prior', 'own_minus_gt_ci_low', 'bias_fp', 'components', 'flags'} <= set(table[0])
    report = (out / 'report.md').read_text()
    assert 'EXPLORATORY' in report and 'IL3D_3dfront' in report and '## 范围 extra-1（4 个房间，k = 2）' in report
    assert '不能单凭陡峭读成' in report and '### 尺寸替换：下游接受，全部房间' in report and '逐规则失败占比' in report
    assert 'family:study_office' in report and "| OS | " in report and "{'arm': 'os', 'mode': 'prior'}" in report
    assert h.strict_json((out / 'MANIFEST.json').read_text())['files'].keys() == {
        p.name for p in out.iterdir()} - {'MANIFEST.json'}
    with pytest.raises(FileExistsError):
        S.run(**paths, out=out, repeats=200)


def _s1_dir(root, paths, arm='joint', reasons=None):
    """An output_repair dir of the geometry's ARM rows: every room clean after S1 except `reasons` {uid: codes}."""
    source = paths['geometry'] / f'{arm}-geometry.jsonl'
    rows = [{**x, 's1': {'geometry_s1': dict.fromkeys(S.CHECKER_REASONS, 0),
                         'downstream_s1': {'reasons': (reasons or {}).get(x['uid'], [])}}}
            for x in map(json.loads, source.read_text().splitlines())]
    _publish(root, {f'{arm}-geometry-s1.jsonl': _jsonl(rows), 'summary.json': json.dumps(
        {'status': 'S1_COMPLETE', 'requests_arm': arm, 'params': {'max_trials': 2000, 'time_limit_s': 60.0},
         'inputs': {'data_manifest_sha256': h.sha256(paths['data'] / 'MANIFEST.json'), 'geometry': source.name,
                    'geometry_sha256': h.sha256(source)}})})
    return root


def test_cap_rule_from_inference_runs_and_rules_after_s1(tmp_path, monkeypatch):
    paths = world(tmp_path)
    monkeypatch.setattr(S.e, '_inference', lambda root, data, uids: (
        [{'uid': u, 'reached_token_cap': u == 'r6', 'stopped_on_eos': False} for u in uids],
        {'run_sha256': RUNS[Path(root).name]}))
    runs = {arm: tmp_path / arm for arm in h.ARMS}
    s1 = {'J': _s1_dir(tmp_path / 's1-joint', paths, reasons={'r0': ['A3_ENTRY_BLOCKED']})}
    value = S.run(**{**paths, 'agents': {}, 'extras': {}}, out=tmp_path / 'out', runs=runs, s1=s1, repeats=200)
    assert value['cap_rule'] == {'J': 'applied: 1 complete rows failed', 'P': 'applied: 1 complete rows failed'}
    assert value['inputs']['joint_run_sha256'] == RUNS['joint'] and value['s1_systems'] == ['J']
    systems = value['scopes']['all-test']['subsets']['all']['systems']
    assert [p['estimate'] for p in systems['P']['curves']['checker']['xy']] == [19 / 20] * len(S.GRID)
    assert systems['J']['rules'] == {'raw': {**systems['J']['rules']['raw'], 'incomplete': 1 / 20},
                                     's1': {'downstream:A3_ENTRY_BLOCKED': 1 / 20, 'incomplete': 1 / 20}}
    assert set(systems['P']['rules']) == {'raw'}
    rules = list(csv.DictReader((tmp_path / 'out/rules.csv').open()))
    assert {(x['system'], x['stage'], x['rule']) for x in rules if x['subset'] == 'all' and x['stage'] == 's1'} == {
        ('J', 's1', 'downstream:A3_ENTRY_BLOCKED'), ('J', 's1', 'incomplete')}
    assert '| J +S1 |' in (tmp_path / 'out/report.md').read_text()
    with pytest.raises(ValueError, match='does not score'):
        S.run(**paths, out=tmp_path / 'o1', runs={**runs, 'joint': tmp_path / 'placement'})
    with pytest.raises(ValueError, match='not S1 of'):           # P is not what this S1 repaired
        S.run(**paths, out=tmp_path / 'o2', s1={'P': _s1_dir(tmp_path / 's1-p', paths)}, repeats=200)
    with pytest.raises(ValueError, match='not J, P or an agent'):
        S.run(**paths, out=tmp_path / 'o3', s1={'OS': s1['J']}, repeats=200)
    assert S._pairs(['J=x'], '--s1', ('GT',)) == {'J': 'x'}
    assert not any((tmp_path / f'o{k}').exists() for k in (1, 2, 3))


def test_inputs_are_refused_outside_the_contract(tmp_path):
    paths = world(tmp_path)
    rows = (paths['geometry'] / 'joint-geometry.jsonl').read_text().splitlines()
    bad = json.loads(rows[0])
    bad['main_collisions'] += 1                                    # same geometry_success, other checker counts
    (paths['geometry'] / 'joint-geometry.jsonl').write_text('\n'.join([json.dumps(bad), *rows[1:]]) + '\n')
    files = {k: v for k, v in h.artifact_manifest(paths['geometry']).items() if k != 'MANIFEST.json'}
    h.write_json(paths['geometry'] / 'MANIFEST.json', {'files': files})
    with pytest.raises(ValueError, match='does not reproduce'):
        S.run(**paths, out=tmp_path / 'o1', repeats=200)
    with pytest.raises(ValueError, match='no rows'):
        S.run(**{**paths, 'extras': {'OS': paths['extras']['OS'].replace('arm=os', 'arm=none')}}, out=tmp_path / 'o2')
    loose = tmp_path / 'loose.jsonl'                               # an extras file outside any MANIFEST
    loose.write_text((tmp_path / 'optiscene-score/e2-rows.jsonl').read_text())
    (tmp_path / 'MANIFEST.json').write_text(json.dumps({'files': {}}))
    with pytest.raises(ValueError, match='not listed'):
        S.run(**{**paths, 'extras': {'OS': f'{loose}::arm=os'}}, out=tmp_path / 'o3')
    agent = paths['agents']['L']
    data_sha = h.sha256(paths['data'] / 'MANIFEST.json')
    for summary, match in ((_agent_summary(data_sha, [f'r{k}' for k in range(6)], request_arm='placement'),
                            'size-fixed'), ({**_agent_summary('0' * 64, []), 'status': 'LLM_COMPARISON_COMPLETE'},
                                            'other data')):
        (agent / 'summary.json').write_text(json.dumps(summary))
        h.write_json(agent / 'MANIFEST.json', {'files': {k: v for k, v in h.artifact_manifest(agent).items()
                                                         if k != 'MANIFEST.json'}})
        with pytest.raises(ValueError, match=match):
            S.run(**paths, out=tmp_path / 'o4', repeats=200)
    with pytest.raises(ValueError, match='unique NAME=PATH'):
        S._pairs(['J=x'], '--agent')
    with pytest.raises(ValueError, match='real/synthetic'):
        S.kind('NewSource')
    assert not any((tmp_path / f'o{k}').exists() for k in range(1, 5))


def test_s_match_and_slope():
    assert S.s_match([1, 1, 1, .9, .8, .7, .6, .5, .4], .65) == pytest.approx(.975)
    assert S.s_match([.2] * 9, .5) is None and S.s_match([.5] * 9, .5) == 1.0
    assert S.slope([1 - s for s in S.GRID]) == pytest.approx(-1) and S.slope([.3] * 9) == 0


def test_submit_one_cpu_job_and_refusals(tmp_path):
    subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    agents = ('L', 'O1a', 'O1b', 'O2', 'O3', 'O4', 'O6', 'O7')
    s1 = ['joint-sft', 'placement-rpo-dpo1', *(f'agent-{n}-{v}' for n in agents for v in ('harness', 'raw'))]
    for name in ['eval-data-v1/MANIFEST.json', 'geometry-v5/MANIFEST.json', 'infer-joint-002/MANIFEST.json',
                 'infer-placement-005/MANIFEST.json', 'llm-cohort-v1.json',
                 *(f'llm-eval-{n}-selected-v1/MANIFEST.json' for n in agents),
                 *(f's1-012-test/{d}/MANIFEST.json' for d in s1)]:
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text('{}\n')
    (project / 'env/bin').mkdir(parents=True)
    (project / 'env/bin/python').write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    (project / 'env/bin/python').chmod(0o755)
    (tmp_path / 'squeue').write_text('#!/bin/sh\ntest "$*" = "-h -u me -o %i %j" || exit 3\n'
                                     'printf "%s" "$SQUEUE_OUT"\n')
    (tmp_path / 'squeue').chmod(0o755)
    text = SCRIPT.read_text()
    assert text.count(f'PROJECT_ROOT={PROJECT}\n') == 1
    script = tmp_path / 'submit_016_scale.sh'
    script.write_text(text.replace(f'PROJECT_ROOT={PROJECT}\n', f'PROJECT_ROOT={project}\n'))
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    done = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    log = (tmp_path / 'log').read_text().splitlines()
    head = (f'--parsable --account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=72 '
            f'--mem=200G --gres=none --no-requeue --time=08:00:00 --job-name=unknownsize-016-scale '
            f'--output={project}/runs/isambard/unknownsize-016-scale-%j.out --wrap ')
    agent_flags = ' '.join(f'--agent {n}={run}/llm-eval-{n}-selected-v1' for n in agents)
    s1_flags = ' '.join(f'--s1 {k}={run}/s1-012-test/{d}' for k, d in zip(
        ['J', 'P', *(f'{n}-{v}' for n in agents for v in ('harness', 'raw'))], s1))
    assert log == [head + f'cd {CODE} && PYTHONPATH={CODE} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 '
                   f'MKL_NUM_THREADS=1 MPLCONFIGDIR={project}/.cache-jobs/unknownsize-016-scale/mpl '
                   f'{project}/env/bin/python -B -m fastfill.comparison.scale_curve '
                   f'--data {run}/eval-data-v1 --geometry {run}/geometry-v5 --cohort {run}/llm-cohort-v1.json '
                   f'--joint-run {run}/infer-joint-002 --placement-run {run}/infer-placement-005 {agent_flags} '
                   f'{s1_flags} --out {run}/scale-curve-016 --workers 70 --repeats 2000']
    assert 'no OptiScene extras' in done.stdout
    score = run / 'optiscene-014/score'
    score.mkdir(parents=True)
    (score / 'MANIFEST.json').write_text('{}\n')
    (score / 'e2-rows.jsonl').write_text('{}\n')
    (tmp_path / 'log').unlink()
    (run / 'scale-curve-016').mkdir()                               # the first output exists: a new name adds OS
    assert subprocess.run(['bash', str(script), str(CODE), 'scale-curve-016-os'], env=env, capture_output=True,
                          text=True).returncode == 0
    extras = ' '.join(f'--extra OS-{s}-{m}={score}/e2-rows.jsonl::arm={s}-e2-sft,mode={m}'
                      for s in ('sft', 'dpo1', 'dpo2') for m in ('prior', 'gt'))
    assert f'{agent_flags} {extras} {s1_flags} --out {run}/scale-curve-016-os ' in (tmp_path / 'log').read_text()
    for bad in ('../x', '.hidden', 'a b'):
        assert subprocess.run(['bash', str(script), str(CODE), bad], env=env, capture_output=True,
                              text=True).returncode == 2
    (run / 'scale-curve-016').rmdir()
    queued = subprocess.run(['bash', str(script), str(CODE)], env={**env, 'SQUEUE_OUT': '9001 unknownsize-016-scale\n'},
                            capture_output=True, text=True)
    assert queued.returncode == 1 and 'scancel 9001' in queued.stderr
    (run / 'scale-curve-016').mkdir()
    exists = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert exists.returncode == 1 and 'scale-curve-016 exists' in exists.stderr
    (run / 'scale-curve-016').rmdir()
    for name in ('llm-eval-O4-selected-v1/MANIFEST.json', 's1-012-test/agent-O7-raw/MANIFEST.json'):
        (run / name).unlink()
        missing = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
        assert missing.returncode == 1 and name.split('/')[-2] in missing.stderr
        (run / name).write_text('{}\n')
    assert subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True).returncode == 2
    assert len((tmp_path / 'log').read_text().splitlines()) == 1                     # nothing new was submitted
