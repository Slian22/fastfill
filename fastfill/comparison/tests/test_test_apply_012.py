"""Amendment 012 §4 test application: summarize on a fixture run root (real S1 outputs), and its Isambard chain."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from fastfill.comparison import geometry as g
from fastfill.comparison import llm_agent
from fastfill.comparison import output_repair as r
from fastfill.comparison import test_apply_012 as A
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.tests import test_infer_shard as T

CODE = Path(__file__).resolve().parents[3]
SCRIPT = CODE / 'fastfill/comparison/submit_test_012.sh'
PROJECT = '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill'


def obj(i, category, size, pos):
    return {'id': i, 'category': category, 'size': list(size), 'pos': list(pos), 'yaw': 0.0, 'parent': None}


def _room(uid, source, group, room_type):
    raw = {'uid': uid, 'source': source, 'group': group, 'boundary_type': 'polygon', 'room_type': room_type,
           'boundary': [[0, 0], [5, 0], [5, 4], [0, 4]], 'height': 3,
           'objects': [obj('w1', 'wardrobe', (.6, 1.2, 2), (4.6, 1, 0)), obj('w2', 'wardrobe', (.6, 1.2, 2), (4.6, 3, 0)),
                       obj('bed', 'bed', (2, 1.6, .5), (1.2, 2, 0))],
           'fixed': [{'id': 'd1', 'category': 'door', 'size': [.1, .9, 2], 'pos': [2.5, 0, 0], 'yaw': math.pi / 2}]}
    return bind_room(raw)[0]


def _answer(bound, arm, scale=1.0):
    """Both wardrobes on one spot (S1 separates them); joint sizes scaled by `scale`."""
    pos = lambda o: [4.6, 2, 0] if o['category'] == 'wardrobe' else [1.2, 2, 0]
    return json.dumps({'placements': [{'id': o['id'], 'pos': pos(o), 'yaw': 0,
                                       **({'size': [x * scale for x in o['size'][:2]] + o['size'][2:]}
                                          if arm == 'joint' else {})} for o in bound['objects']]})


def _publish(root, files):
    root.mkdir(parents=True)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    h.write_json(root / 'MANIFEST.json', {'files': h.artifact_manifest(root)})


def _jsonl(rows):
    return ''.join(json.dumps(x) + '\n' for x in rows)


def world(tmp, s2=False):
    """eval data (4 test rooms, 3 houses, one paper stratum), J/P geometry, one agent (L) on a 3-room cohort, the dev
    decision, and real output_repair outputs under s1-root."""
    rooms = [_room('a', 'IL3D_3dfront', 'h1', 'bedroom'), _room('b', 'IL3D_3dfront', 'h1', 'bedroom'),
             _room('c', 'S', 'h2', 'office'), _room('d', 'S', 'h3', 'office')]
    sizes = lambda x: {o['id']: o['size'] for o in x['objects']}
    req = {arm: [{'uid': x['uid'], 'source': x['source'], 'group': x['group'],
                  'messages': messages(x, arm, sizes(x) if arm == 'placement' else None)[:2]} for x in rooms]
           for arm in h.ARMS}
    data = tmp / 'data'
    _publish(data, {'rooms.jsonl': _jsonl(rooms), **{f'{arm}/requests.jsonl': _jsonl(req[arm]) for arm in h.ARMS}})
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    h.write_json(data / 'MANIFEST.json', {**manifest, 'expected_uids': [x['uid'] for x in rooms]})
    score = lambda x, q, arm, variant, raw: {**g.score(x, q['messages'], raw, arm), 'arm': variant}
    geo = tmp / 'geometry-v5'
    _publish(geo, {f'{arm}-geometry.jsonl': _jsonl(score(x, q, arm, arm, _answer(x, arm)) for x, q in zip(rooms, req[arm]))
                   for arm in h.ARMS})
    uids = ['a', 'b', 'c']
    cohort = tmp / 'llm-cohort-v1.json'
    cohort.write_text(json.dumps({'version': 1, 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
                                  'uids_sha256': llm_agent._digest(json.dumps(uids)), 'uids': uids}))
    agent = tmp / 'llm-eval-L-selected-v1'
    rows = {'llm_harness': [score(x, q, 'joint', 'llm_harness', _answer(x, 'joint', .5)) for x, q in zip(rooms, req['joint'])],
            'llm_raw': [score(x, q, 'joint', 'llm_raw', _answer(x, 'joint', .5) if x['uid'] != 'c' else None)
                        for x, q in zip(rooms, req['joint'])]}
    rows['llm_raw'][0] |= {'reached_token_cap': True, 'stopped_on_eos': False}
    _publish(agent, {f'{v}-geometry.jsonl': _jsonl(x for x in rows[v] if x['uid'] in uids) for v in rows})
    decision = tmp / 'decision-012'
    _publish(decision, {'decision.json': json.dumps({'status': 'DECISION_012_COMPLETE', 'decisions': {
        's1_adopted': True, 's2_joint': {'passed': s2}, 's2_placement': {'passed': False}}})})
    s1 = tmp / 's1-012-test'
    for label, (name, arm, geometry) in A.OURS.items():
        r.run(data, geo / geometry, arm, s1 / name)
    for variant in A.VARIANTS:
        r.run(data, agent / f'llm_{variant}-geometry.jsonl', 'joint', s1 / f'agent-L-{variant}', uids=cohort)
    return {'data': data, 's1_root': s1, 'cohort_path': cohort, 'decision': decision, 'out': tmp / 'out'}


def test_summarize_reports_raw_beside_s1_with_bias_gt_and_contrasts(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'AGENTS', ('L',))
    paths = world(tmp_path)
    result = A.summarize(**paths, repeats=20)
    assert sorted(p.name for p in paths['out'].iterdir()) == ['MANIFEST.json', 'contrasts.csv', 'levels.csv',
                                                             'report.md', 'results.json']
    assert result['agents'] == ['L'] and result['scopes'] == {
        'test': {'rooms': 4, 'systems': ['J', 'P', 'GT']},
        'cohort': {'rooms': 3, 'systems': ['J', 'P', 'GT', 'L-harness', 'L-raw']}}
    test, cohort = result['levels']['test'], result['levels']['cohort']
    assert list(test['J']) == ['all', 'paper:IL3D_3dfront/bedroom', 'source:IL3D_3dfront', 'source:S']
    j = test['J']['all']
    assert (j['raw']['requests'], j['raw']['checker_rate'], j['s1']['checker_rate']) == (4, 0.0, 1.0)
    assert j['raw']['bias_fp'] == j['s1']['bias_fp'] == pytest.approx(0.0)
    assert j['repair']['requests'] == 4 and j['repair']['bugs'] == {} and j['repair']['displacement']['moved_objects'] == 4
    raw = cohort['L-raw']['all']
    assert (raw['raw']['requests'], raw['raw']['complete_rooms'], raw['raw']['complete_rate']) == (3, 2, 2 / 3)
    assert raw['raw']['bias_fp'] == pytest.approx(2 * math.log(.5))     # agents undersize: B next to acceptance
    assert raw['s1']['checker_rate'] == pytest.approx(2 / 3)
    assert result['cap_rule_would_fail_complete'] == {'J': None, 'P': None, 'L-harness': None, 'L-raw': 1}
    assert set(test['GT']['all']) == {'raw'} and test['GT']['all']['raw']['bias_fp'] == 0.0
    assert cohort['J']['all']['raw']['requests'] == 3 and 'source:S' in cohort['GT']
    contrasts = {(c['scope'], c['comparison'], c['subset']): c for c in result['contrasts']}
    names = {(s, n) for s, n, _ in contrasts}
    assert names == {('test', 'P+S1 - J+S1'), ('test', 'J+S1 - J'), ('test', 'P+S1 - P'),
                     *(('cohort', x) for x in ('P+S1 - L-harness+S1', 'J+S1 - L-harness+S1', 'P+S1 - L-raw+S1',
                                               'J+S1 - L-raw+S1', 'J+S1 - J', 'P+S1 - P', 'L-harness+S1 - L-harness',
                                               'L-raw+S1 - L-raw'))}
    assert {sub for *_, sub in contrasts} == {'all', 'paper:IL3D_3dfront/bedroom'}
    gain = contrasts['test', 'J+S1 - J', 'all']
    assert (gain['rooms'], gain['components'], gain['fields']['checker']['estimate']) == (4, 3, 1.0)
    vs = contrasts['cohort', 'J+S1 - L-raw+S1', 'all']
    assert vs['fields']['complete']['estimate'] == pytest.approx(1 / 3)
    assert vs['bias_fp'] == [pytest.approx(0.0), pytest.approx(2 * math.log(.5))]
    assert contrasts['cohort', 'J+S1 - J', 'paper:IL3D_3dfront/bedroom']['rooms'] == 2
    with (paths['out'] / 'levels.csv').open() as stream:
        levels = list(csv.DictReader(stream))
    row = next(x for x in levels if (x['scope'], x['subset'], x['system'], x['stage']) == ('test', 'all', 'J', 's1'))
    assert float(row['checker_rate']) == 1.0 and float(row['moved_median_m']) > 0 and row['label'] == A.TAG
    report = (paths['out'] / 'report.md').read_text()
    assert 'EXPLORATORY' in report and '确认性结果与交付结果都不变' in report and "{'L-raw': 1}" in report
    assert h.strict_json((paths['out'] / 'MANIFEST.json').read_text())['files'].keys() == {
        'contrasts.csv', 'levels.csv', 'report.md', 'results.json'}
    with pytest.raises(FileExistsError):
        A.summarize(**paths, repeats=20)


def test_summarize_refuses_inputs_outside_the_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'AGENTS', ('L',))
    paths = world(tmp_path / 'w1', s2=True)
    with pytest.raises(ValueError, match='adopt S1 and neither S2'):
        A.summarize(**paths, repeats=20)
    paths = world(tmp_path / 'w2')
    (paths['s1_root'] / 'agent-L-raw/MANIFEST.json').rename(paths['s1_root'] / 'agent-L-raw/MANIFEST.old')
    with pytest.raises(FileNotFoundError):
        A.summarize(**paths, repeats=20)
    (paths['s1_root'] / 'agent-L-raw/MANIFEST.old').rename(paths['s1_root'] / 'agent-L-raw/MANIFEST.json')
    monkeypatch.setattr(A, 'AGENTS', ('L', 'M'))                       # an expected agent without S1 outputs
    with pytest.raises(FileNotFoundError, match='agent-M-harness'):
        A.summarize(**paths, repeats=20)
    monkeypatch.setattr(A, 'AGENTS', ('L',))
    summary = json.loads((paths['s1_root'] / 'joint-sft/summary.json').read_text())
    for key, value, match in ((('params', 'time_limit_s'), 30.0, 'protocol settings'),
                              (('requests_arm',), 'placement', 'protocol settings'),
                              (('params', 'clear_m'), 0.02, 'different parameters or code'),
                              (('code_sha256', 'downstream.py'), '0' * 64, 'different parameters or code')):
        changed = json.loads(json.dumps(summary))
        target = changed
        for k in key[:-1]:
            target = target[k]
        target[key[-1]] = value
        (paths['s1_root'] / 'joint-sft/summary.json').write_text(json.dumps(changed))
        manifest = json.loads((paths['s1_root'] / 'joint-sft/MANIFEST.json').read_text())
        manifest['files']['summary.json'] = h.sha256(paths['s1_root'] / 'joint-sft/summary.json')
        (paths['s1_root'] / 'joint-sft/MANIFEST.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=match):
            A.summarize(**paths, repeats=20)
    cohort = json.loads(paths['cohort_path'].read_text())
    paths['cohort_path'].write_text(json.dumps({**cohort, 'uids': ['a', 'b']}))
    with pytest.raises(ValueError, match='cohort does not bind'):
        A.summarize(**paths, repeats=20)
    assert not paths['out'].exists()


def submit_world(tmp_path):
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    for name in ['eval-data-v1/MANIFEST.json', 'geometry-v5/MANIFEST.json', 'geometry-v5/joint-geometry.jsonl',
                 'geometry-v5/placement-geometry.jsonl', 'llm-cohort-v1.json', 'decision-012/MANIFEST.json',
                 *(f'llm-eval-{n}-selected-v1/{f}' for n in ('L', 'O1a', 'O1b', 'O2', 'O3', 'O4', 'O6', 'O7')
                   for f in ('MANIFEST.json', 'llm_harness-geometry.jsonl', 'llm_raw-geometry.jsonl'))]:
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text('{}\n')
    (run / 'decision-012/decision.json').write_text(json.dumps({'status': 'DECISION_012_COMPLETE', 'decisions': {
        's1_adopted': True, 's2_joint': {'passed': False}, 's2_placement': {'passed': False}}}))
    (project / 'env/bin').mkdir(parents=True)
    (project / 'env/bin/python').write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    (project / 'env/bin/python').chmod(0o755)
    (tmp_path / 'squeue').write_text('#!/bin/sh\ntest "$*" = "-h -u me -o %i %j" || exit 3\nprintf "%s" "$SQUEUE_OUT"\n')
    (tmp_path / 'squeue').chmod(0o755)
    text = SCRIPT.read_text()
    assert text.count(f'PROJECT_ROOT={PROJECT}\n') == 1 and f"AGENTS=({' '.join(A.AGENTS)})" in text
    script = tmp_path / 'submit_test_012.sh'
    script.write_text(text.replace(f'PROJECT_ROOT={PROJECT}\n', f'PROJECT_ROOT={project}\n'))
    return script, run


def test_submit_chain_holds_roots_waits_on_every_s1_and_refuses(tmp_path):
    subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
    script, run = submit_world(tmp_path)
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    done = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    log = (tmp_path / 'log').read_text().splitlines()
    project = run.parents[1]
    cpu = ('--account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=72 --mem=200G '
           '--gres=none --no-requeue --time=04:00:00')
    env_line = f'cd {CODE} && PYTHONPATH={CODE} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1'
    repair = (f'{env_line} {project}/env/bin/python -B -m fastfill.comparison.output_repair --data {run}/eval-data-v1 '
              '--workers 70 --max-trials 2000 --time-limit 60')
    head = lambda label: (f'--job-name=unknownsize-012-test-{label} '
                          f'--output={project}/runs/isambard/unknownsize-012-test-{label}-%j.out')
    assert log[0] == (f'--parsable --hold {cpu} {head("s1-J")} --wrap {repair} --geometry {run}/geometry-v5/joint-geometry.jsonl '
                      f'--requests-arm joint --out {run}/s1-012-test/joint-sft')
    assert log[1].endswith(f'--geometry {run}/geometry-v5/placement-geometry.jsonl --requests-arm placement '
                           f'--out {run}/s1-012-test/placement-rpo-dpo1')
    assert log[2] == (f'--parsable --hold {cpu} {head("s1-L-harness")} --wrap {repair} '
                      f'--geometry {run}/llm-eval-L-selected-v1/llm_harness-geometry.jsonl --requests-arm joint '
                      f'--uids {run}/llm-cohort-v1.json --out {run}/s1-012-test/agent-L-harness')
    assert log[17].endswith(f'--uids {run}/llm-cohort-v1.json --out {run}/s1-012-test/agent-O7-raw')
    assert all(x.startswith('--parsable --hold ') for x in log[:18])
    deps = ':'.join(str(101 + k) for k in range(18))
    assert log[18] == (f'--parsable --kill-on-invalid-dep=yes --dependency=afterok:{deps} {cpu} {head("summarize")} '
                       f'--wrap {env_line} {project}/env/bin/python -B -m fastfill.comparison.test_apply_012 summarize '
                       f'--data {run}/eval-data-v1 --s1-root {run}/s1-012-test --cohort {run}/llm-cohort-v1.json '
                       f'--decision {run}/decision-012 --out {run}/apply-012-test --workers 70')
    assert len(log) == 19
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release ' + deps.replace(':', ',') + '\n'
    failed = tmp_path / 'failed'
    failed.mkdir()
    (failed / 'squeue').write_bytes((tmp_path / 'squeue').read_bytes())
    (failed / 'squeue').chmod(0o755)
    bad = subprocess.run(['bash', str(script), str(CODE)], env={**T.stub_sbatch(failed, fail_on=4), 'USER': 'me'},
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "sbatch failed for 's1-L-raw'" in bad.stderr and 'scancel 101 102 103' in bad.stderr
    assert not (failed / 'log.scontrol').exists()
    queued = subprocess.run(['bash', str(script), str(CODE)], env={**env, 'SQUEUE_OUT': '9001 unknownsize-012-test-s1-J\n'},
                            capture_output=True, text=True)
    assert queued.returncode == 1 and 'scancel 9001' in queued.stderr
    (run / 's1-012-test/agent-O2-raw').mkdir(parents=True)
    exists = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert exists.returncode == 1 and 'agent-O2-raw exists' in exists.stderr
    (run / 's1-012-test/agent-O2-raw').rmdir()
    (run / 'decision-012/decision.json').write_text(json.dumps({'status': 'DECISION_012_COMPLETE', 'decisions': {
        's1_adopted': False, 's2_joint': {'passed': False}, 's2_placement': {'passed': False}}}))
    refused = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert refused.returncode == 1 and 'S1 adopted' in refused.stderr
    (run / 'decision-012/decision.json').unlink()
    missing = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert missing.returncode == 1 and 'missing' in missing.stderr and 'decision-012/decision.json' in missing.stderr
    usage = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert usage.returncode == 2
    assert len((tmp_path / 'log').read_text().splitlines()) == 19  # Nothing new was submitted.
