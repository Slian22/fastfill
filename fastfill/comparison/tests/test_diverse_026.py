"""Amendment 026 addendum A (scope C): the family-stratified uid file, L on test-scope uids, its sbatch modes and
submit_026_diverse.sh. Fakes only (no network, no Slurm)."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from fastfill.comparison import diverse_026 as D
from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import output_repair as r
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.room_families import FAMILIES
from fastfill.comparison.tests import test_llm_agent as base
from fastfill.comparison.tests.test_agents_016 import PlacementApi
from fastfill.comparison.tests.test_llm_agent import world  # noqa: F401 (fixture)

CODE = Path(L.__file__).resolve().parents[2]
PIN = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
RAW = ('bedroom', 'living room', 'dining room', 'kitchen', 'bathroom', 'office', 'meeting room', 'balcony',
       'laundry room', 'garage', 'hallway', 'closet', 'gym', 'studio', None)   # One per FAMILIES entry, in order.
# Taken from the code before amendment 026 (scratch run on this fixture): the run identity without code_sha256 (any
# edit of llm_agent.py changes that) and the llm_agent command lines of isambard_llm_agent.sbatch's existing modes.
GOLDEN_IDENTITY = {'default': 'b2fa2079c7ebaa6fc69a3dc3e2a25dd931a2a7af6b396b0c55a36a6ac5593ded',
                   'pilot': '582610857b04c26ed30c40efd9c74e49651fd573185fd5cffa8155fa2b68bd0d'}
GOLDEN_COMMANDS = 'c4a181aea199b4dc5f4ea0e731dbd712a884bf403cf19ea8358ddf11bb9d5832'


def _placement(root, rooms):
    for request in map(json.loads, (root / 'data/placement/requests.jsonl').read_text().splitlines()):
        rooms[request['uid']]['_placement'] = request
    return rooms


# ---- uid file --------------------------------------------------------------------------------------------------

def test_diverse_takes_the_20_lowest_hashes_of_every_family_once(tmp_path, monkeypatch, capsys):
    data = base.eval_data(tmp_path / 'data', specs=[(f'{k:02d}-{i:02d}', 'S', raw) for k, raw in enumerate(RAW)
                                                    for i in range(22)])
    out = tmp_path / 'a026-diverse-v1.json'
    D.main(['diverse', '--data', str(data), '--out', str(out)])
    value = json.loads(out.read_text())
    assert json.loads(capsys.readouterr().out)['rooms'] == 300
    expected = json.loads((data / 'MANIFEST.json').read_text())['expected_uids']
    assert value['counts'] == {f: {'test': 22, 'diverse': 20} for f in FAMILIES} and len(FAMILIES) == 15
    assert value['uids'] == [u for u in expected if u in set(value['uids'])]  # expected_uids order
    for k in range(len(RAW)):
        pool = [u for u in expected if u.startswith(f'{k:02d}-')]
        lowest = sorted(pool, key=lambda u: hashlib.sha256(('a026c:' + u).encode()).hexdigest())[:20]
        assert set(value['uids']) & set(pool) == set(lowest)
    assert value['uids_sha256'] == L._digest(json.dumps(value['uids']))
    assert value['data_manifest_sha256'] == h.sha256(data / 'MANIFEST.json')
    assert r.load_uids(out) == value['uids']   # output_repair --uids reads it as is
    assert D.diverse(data, out) == value       # deterministic: verified, not rewritten
    out.write_text(json.dumps({**value, 'uids': value['uids'][::-1]}))
    with pytest.raises(ValueError, match='differs from its deterministic rebuild'):
        D.diverse(data, out)
    monkeypatch.setattr(D, 'PER_FAMILY', 23)
    with pytest.raises(ValueError, match='fewer than 23 test rooms'):
        D.diverse(data, tmp_path / 'short.json')
    assert not (tmp_path / 'short.json').exists()


# ---- L on test-scope uids --------------------------------------------------------------------------------------

def test_test_scope_uids_run_under_the_cap_and_are_scored(world):
    root, rooms = world
    rooms = _placement(root, rooms)
    cohort = json.loads((root / 'cohort.json').read_text())['uids']
    expected = json.loads((root / 'data/MANIFEST.json').read_text())['expected_uids']
    outside = [u for u in expected if u not in cohort]
    assert len(outside) == 3
    uids = root / 'c.json'
    uids.write_text(json.dumps({'uids': [outside[2], cohort[5], outside[0]]}))
    order = [u for u in expected if u in {outside[2], cohort[5], outside[0]}]
    run = lambda out, api, **extra: L.run(base.args(root, out=root / out, arm='placement', uids=uids, **extra), api)
    with pytest.raises(ValueError, match='unique cohort uids'):   # The default scope still refuses them.
        run('refused', PlacementApi(rooms))
    # Hard cap: the ledger stops at --max-calls, and a resume under the same cap sends nothing.
    api = PlacementApi(rooms, default='bad')
    assert run('cap', api, uids_scope='test', max_calls=4, concurrency=1)['status'] == 'CALL_CAP_REACHED'
    assert len(api.calls) == 4 and len((root / 'cap/calls.jsonl').read_text().splitlines()) == 4
    again = PlacementApi(rooms)
    assert run('cap', again, uids_scope='test', max_calls=4)['status'] == 'CALL_CAP_REACHED' and not again.calls
    result = run('run', PlacementApi(rooms), uids_scope='test')
    assert result['status'] == 'LLM_COMPLETE' and result['rooms'] == 3 and result['calls_ledger'] == 3
    assert json.loads((root / 'run/MANIFEST.json').read_text())['expected_uids'] == order
    identity = json.loads((root / 'run/run_receipt.json').read_text())
    assert identity['cohort']['subset'] == {'sha256': h.sha256(uids), 'uids_sha256': L._digest(json.dumps(order)),
                                            'scope': 'test'}
    assert identity['config']['request_arm'] == 'placement'
    runs = {arm: base._inference(root, root / 'data', arm, rooms) for arm in h.ARMS}
    with pytest.raises(ValueError, match='unique cohort uids'):
        ev.evaluate(root / 'data', root / 'cohort.json', root / 'run', root / 'e0', runs, repeats=20, uids_path=uids)
    summary = ev.evaluate(root / 'data', root / 'cohort.json', root / 'run', root / 'eval', runs, repeats=20,
                          uids_path=uids, uids_scope='test')
    assert summary['cohort']['subset'] == {'rooms': 3, 'sha256': h.sha256(uids),
                                           'uids_sha256': L._digest(json.dumps(order)), 'scope': 'test'}
    assert {n: a['all']['requests'] for n, a in summary['arms'].items()} == dict.fromkeys(
        ('joint', 'placement', 'llm_raw', 'llm_harness'), 3)
    harness = [json.loads(x)['uid'] for x in (root / 'eval/llm_harness-geometry.jsonl').read_text().splitlines()]
    assert harness == order   # = the uid file's rooms, as output_repair --uids requires
    assert '测试集子集：3 个房间' in (root / 'eval/report.md').read_text()


# ---- existing identities and commands --------------------------------------------------------------------------

def _sbatch_commands(tmp_path, *modes):
    """isambard_llm_agent.sbatch per mode with PROJECT_ROOT = tmp_path; the env python logs its argv and environment."""
    text = (CODE / 'fastfill/comparison/isambard_llm_agent.sbatch').read_text()
    assert text.count(PIN) == 1
    tmp_path.mkdir(exist_ok=True)
    script = tmp_path / 'agent.sbatch'
    script.write_text(text.replace(PIN, f'PROJECT_ROOT={tmp_path}\n'))
    (tmp_path / 'runs/unknownsize-20261013').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'runs/unknownsize-20261013/llm-cohort-v1.json').write_text('{}')
    (tmp_path / 'env/bin').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'bin').mkdir(exist_ok=True)
    python, timeout = tmp_path / 'env/bin/python', tmp_path / 'bin/timeout'
    python.write_text('#!/bin/sh\necho "$(pwd) $PYTHONPATH $FASTFILL_API_ENV ${OPENAI_API_KEY-unset} $*" >> "$LOG"\n')
    timeout.write_text('#!/bin/sh\nshift 3\nexec "$@"\n')
    for path in (python, timeout):
        path.chmod(0o755)
    out = []
    for mode in modes:
        log = tmp_path / ('-'.join(mode) + '.log')
        env = {'PATH': f'{tmp_path}/bin:/usr/bin:/bin', 'LOG': str(log), 'SLURM_JOB_ID': '1', 'OPENAI_API_KEY': 'x'}
        done = subprocess.run(['bash', str(script), str(CODE), *mode], env=env, capture_output=True)
        out.append(log.read_text().replace(str(tmp_path), 'ROOT').replace(str(CODE), 'CODE') if done.returncode == 0
                   else done.returncode)
    return out


def test_existing_identities_and_sbatch_commands_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(L, 'BACKOFF_SECONDS', 0)
    data = base.eval_data(tmp_path / 'data')
    base.env_file(tmp_path)
    L.main(['cohort', '--data', str(data), '--out', str(tmp_path / 'cohort.json')])
    rooms = {x['uid']: x for x in map(json.loads, (data / 'rooms.jsonl').read_text().splitlines())}
    for request in map(json.loads, (data / 'joint/requests.jsonl').read_text().splitlines()):
        rooms[request['uid']]['_request'] = request
    rooms = _placement(tmp_path, rooms)
    cohort = json.loads((tmp_path / 'cohort.json').read_text())['uids']
    (tmp_path / 'pilot.json').write_text(json.dumps({'uids': [cohort[3], cohort[0]]}))
    L.run(base.args(tmp_path, out=tmp_path / 'd'), base.FakeApi(rooms))
    L.run(base.args(tmp_path, out=tmp_path / 'p', arm='placement', uids=tmp_path / 'pilot.json'), PlacementApi(rooms))
    for name, out in (('default', 'd'), ('pilot', 'p')):
        receipt = json.loads((tmp_path / out / 'run_receipt.json').read_text())
        assert infer._hash({k: v for k, v in receipt.items() if k not in ('code_sha256', 'run_sha256')}) \
            == GOLDEN_IDENTITY[name], name
    lines = _sbatch_commands(tmp_path / 'sb', ('smoke',), ('full',), ('pilot-placement',))
    assert hashlib.sha256(''.join(lines).encode()).hexdigest() == GOLDEN_COMMANDS
    smoke, full, *bad = _sbatch_commands(
        tmp_path / 'sb', ('uids-smoke', 'a026-diverse-v1.json', 'llm-agent-P-diverse-smoke-001'),
        ('uids-placement', 'a026-diverse-v1.json', 'llm-agent-P-diverse-001', '900'),
        ('uids-placement', 'a026-diverse-v1.json', 'llm-agent-P-diverse-001'), ('smoke', 'extra'))
    head = ('CODE CODE /home/u6uv/shanliantian.u6uv/.fastfill_api.env unset -B -u -m fastfill.comparison.llm_agent run '
            '--data ROOT/runs/unknownsize-20261013/eval-data-v1 --cohort ROOT/runs/unknownsize-20261013/llm-cohort-v1.json '
            '--out ROOT/runs/unknownsize-20261013/')
    common = ('--env-file /home/u6uv/shanliantian.u6uv/.fastfill_api.env --concurrency 4 --arm placement '
              '--uids ROOT/runs/unknownsize-20261013/a026-diverse-v1.json --uids-scope test')
    assert smoke == (f'{head}llm-agent-P-diverse-smoke-001 {common} --limit 2 --max-calls 6 --max-attempts 99 '
                     '--require-no-api-failed\n')
    assert full == f'{head}llm-agent-P-diverse-001 {common} --max-calls 900\n'
    assert all(code != 0 and isinstance(code, int) for code in bad)   # wrong argument counts never run


# ---- submit_026_diverse.sh -------------------------------------------------------------------------------------

def _submit(tmp_path):
    """submit_026_diverse.sh with PROJECT_ROOT = tmp_path (inputs made) and stub sbatch/squeue/scontrol; the env python
    records the diverse_026 call and runs anything else (the smoke gate) with this interpreter."""
    bin_ = tmp_path / 'bin'
    bin_.mkdir()
    helper = (CODE / 'fastfill/comparison/submit_026_diverse.sh').read_text()
    assert helper.count(PIN) == 1
    script = bin_ / 'submit_026_diverse.sh'
    script.write_text(helper.replace(PIN, f'PROJECT_ROOT={tmp_path}\n'))
    python = tmp_path / 'env/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\ncase "$*" in *diverse_026*) echo "$@" >> "$UIDS_LOG"; exit "${{UIDS_RC:-0}}" ;; esac\n'
                      f'exec {sys.executable} "$@"\n')
    python.chmod(0o755)
    (bin_ / 'sbatch').write_text('#!/bin/sh\necho "$@" >> "$SBATCH_LOG"\nn=$(wc -l < "$SBATCH_LOG")\n'
                                 'if [ "$n" -eq "${FAIL_ON:-0}" ]; then exit 1; fi\necho "$((100 + n));cluster"\n')
    (bin_ / 'scontrol').write_text('#!/bin/sh\necho "$@" >> "$SCONTROL_LOG"\n')
    (bin_ / 'squeue').write_text('#!/bin/sh\ntest "$*" = "-h -u me -o %i %j" || exit 3\nprintf "%s" "$SQUEUE_OUT"\n')
    for name in ('sbatch', 'scontrol', 'squeue'):
        (bin_ / name).chmod(0o755)
    run = tmp_path / 'runs/unknownsize-20261013'
    for name in ('eval-data-v1/MANIFEST.json', 'llm-cohort-v1.json', 'infer-joint-002/MANIFEST.json',
                 'infer-placement-005/MANIFEST.json'):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text('{}')
    env = {'PATH': f'{bin_}:/usr/bin:/bin', 'SBATCH_LOG': str(bin_ / 'log'), 'USER': 'me',
           'SCONTROL_LOG': str(bin_ / 'scontrol.log'), 'UIDS_LOG': str(bin_ / 'uids.log')}
    return script, run, env


def _smoke_done(run, failed=0):
    (run / 'llm-agent-P-diverse-smoke-001').mkdir(exist_ok=True)
    h.write_json(run / 'llm-agent-P-diverse-smoke-001/result.json', {'status': 'LLM_COMPLETE', 'api_failed_rooms': failed})


def test_submit_026_smoke_then_full_chain(tmp_path):
    script, run, env = _submit(tmp_path)
    s, uids = f'{CODE}/fastfill/comparison', f'{run}/a026-diverse-v1.json'
    smoke = subprocess.run(['bash', str(script), str(CODE), 'smoke'], env=env, capture_output=True, text=True, check=True)
    assert (tmp_path / 'bin/log').read_text().splitlines() == [
        f'--parsable --time=00:30:00 {s}/isambard_llm_agent.sbatch {CODE} uids-smoke a026-diverse-v1.json '
        'llm-agent-P-diverse-smoke-001']
    assert (tmp_path / 'bin/uids.log').read_text().split() == [
        '-B', '-m', 'fastfill.comparison.diverse_026', 'diverse', '--data', f'{run}/eval-data-v1', '--out', uids]
    assert not (tmp_path / 'bin/scontrol.log').exists()
    (tmp_path / 'bin/log').unlink()
    _smoke_done(run)
    full = subprocess.run(['bash', str(script), str(CODE), 'full'], env=env, capture_output=True, text=True, check=True)
    log = (tmp_path / 'bin/log').read_text().splitlines()
    assert log[0] == (f'--parsable --hold --time=04:00:00 {s}/isambard_llm_agent.sbatch {CODE} uids-placement '
                      'a026-diverse-v1.json llm-agent-P-diverse-001 900')
    cpu = '--account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --gres=none --no-requeue'
    assert '--kill-on-invalid-dep=yes ' in log[1] and cpu in log[1] and '--dependency=afterok:101 ' in log[1]
    assert log[1].endswith(f'--llm {run}/llm-agent-P-diverse-001 --joint {run}/infer-joint-002 --placement '
                           f'{run}/infer-placement-005 --out {run}/llm-eval-L-P-diverse-v1 --workers 16 --uids {uids} '
                           '--uids-scope test')
    for line, v in zip(log[2:], ('harness', 'raw')):
        assert '--dependency=afterok:102 ' in line and cpu in line and '--cpus-per-task=72' in line
        assert line.endswith(f'--max-trials 2000 --time-limit 60 --geometry {run}/llm-eval-L-P-diverse-v1/llm_{v}-'
                             f'geometry.jsonl --requests-arm placement --uids {uids} --out {run}/s1-026/agent-L-P-diverse-{v}')
    assert len(log) == 4 and (tmp_path / 'bin/scontrol.log').read_text() == 'release 101\n'
    assert full.stdout.splitlines()[-1] == 'released 101'
    for done in (smoke, full):
        assert '.fastfill_api.env' not in done.stdout + done.stderr
    (tmp_path / 'bin/log').unlink()
    bad = subprocess.run(['bash', str(script), str(CODE), 'full'], env={**env, 'FAIL_ON': '2'}, capture_output=True,
                         text=True)
    assert bad.returncode == 1 and 'scancel 101' in bad.stderr


def test_submit_026_refuses_before_submitting(tmp_path):
    script, run, env = _submit(tmp_path)
    _smoke_done(run)
    for mode, setup, extra_env, message in (
            ('full', lambda: (run / 's1-026/agent-L-P-diverse-raw').mkdir(parents=True), {}, 'agent-L-P-diverse-raw exists'),
            ('full', lambda: (run / 's1-026/agent-L-P-diverse-raw').rmdir(),
             {'SQUEUE_OUT': '7400001 unknownsize-llm-agent\n'}, 'scancel 7400001'),
            ('smoke', lambda: None, {}, 'llm-agent-P-diverse-smoke-001 exists'),
            ('full', lambda: _smoke_done(run, failed=1), {}, 'has not completed'),
            ('full', lambda: _smoke_done(run), {'SQUEUE_OUT': '7400002 unknownsize-026-diverse-eval\n'}, 'scancel 7400002'),
            ('full', lambda: None, {'UIDS_RC': '1'}, 'diverse_026 diverse'),
            ('full', lambda: (run / 'infer-placement-005/MANIFEST.json').unlink(), {}, 'missing')):
        setup()
        bad = subprocess.run(['bash', str(script), str(CODE), mode], env={**env, **extra_env}, capture_output=True,
                             text=True)
        assert bad.returncode == 1 and message in bad.stderr and 'nothing submitted' in bad.stderr, bad.stderr
        assert not (tmp_path / 'bin/log').exists()
    for argv in ((str(CODE),), (str(CODE), 'pilot')):
        assert subprocess.run(['bash', str(script), *argv], env=env, capture_output=True).returncode == 2
    subprocess.run(['bash', '-n', str(CODE / 'fastfill/comparison/submit_026_diverse.sh')], check=True)
