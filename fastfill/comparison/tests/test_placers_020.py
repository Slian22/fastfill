"""Amendment 020 placers on CPU: the amendment-014 toy E2 rooms plus a Structured3D garage (E3), fake sizer estimates,
fake OptiScene generations, fake P runs, fake sacct / squeue / sbatch."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from fastfill.comparison import contract, geometry, infer, report_v2
from fastfill.comparison import optiscene_eval as oe
from fastfill.comparison import placers_020 as pl
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests.test_optiscene_eval import (  # noqa: F401 (e2world, world: fixtures)
    INSTR, FakeLLM, e1_stub, e2world, gt_design, publish_gen, toy_families, v1_text, world)

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
BASE_COMMIT = '510f4c7cc'  # optiscene_eval.py before amendment 020: default outputs must not change
S3D = 'structured3d_00001_room_1'
POLICY = {'policy': {'adapter': 'a' * 64}, 'stage': 'dpo1', 'base_sha256': 'b' * 64, 'lineage_sha256': 'c' * 64,
          'training_data': {'manifest_sha256': 'd' * 64}}
REASON = 'invalid size output: every listed object ID exactly once and no other key'  # a sizers_020 failure text
CONFIG = {'base': '/m/Qwen3-8B', 'model': '/r/dpo-003/placement/dpo1-merged', 'arm': 'placement', 'batch_size': 8,
          'lora': 'unmerged_fp32_stage_adapter_on_bf16_training_base'}


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


@pytest.fixture
def w(e2world, monkeypatch):
    """eval-data: holo + il3d (E2, paper strata) and S3D (E3, garage), joint requests added; plans e2/e3 (the 014/015
    plans); sizer L (sizers_020 layout): holo and S3D answered (x/y x 0.9 + float noise), il3d failed (result.json) ->
    plan-{e2,e3}-llm-L."""
    data, tmp = e2world.data, e2world.tmp
    holo, il3d = e2world.labels
    s3d = {**holo, 'uid': S3D, 'source': 'Structured3D', 'room_type': 'garage', 'group': 'c:2', 'component': 'c:2'}
    labels, prior = [holo, il3d, s3d], {**e2world.prior, S3D: e2world.prior[holo['uid']]}
    meta = lambda r: {k: r[k] for k in ('uid', 'source', 'group', 'component', 'split', 'room_type')}
    jsonl(data / 'rooms.jsonl', labels)
    jsonl(data / 'placement/requests.jsonl', [{**meta(r), 'messages': contract.messages(
        r, 'placement', estimated_sizes=prior[r['uid']])[:2]} for r in labels])
    jsonl(data / 'joint/requests.jsonl', [{**meta(r), 'messages': contract.messages(r, 'joint')[:2]} for r in labels])
    jsonl(data / 'size-estimates.jsonl', [{'uid': r['uid'], 'estimated_sizes': prior[r['uid']]} for r in labels])
    names = [p.relative_to(data).as_posix() for p in sorted(data.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json']
    (data / 'MANIFEST.json').write_text(json.dumps({'files': {n: oe._sha(data / n) for n in names}}))
    toy_families(monkeypatch, {'bedroom': 'bedroom', 'LivingRoom': 'living', 'garage': 'garage'})
    e1 = e1_stub(tmp / 'e1', {'HOLODECK-bedroom-7': 0.0})
    oe.plan_e2(data, e2world.official, e1, tmp / 'e2')
    oe.plan_e3(data, e2world.official, e1, tmp / 'e3')
    est = {r['uid']: {o['id']: [o['size'][0] * .9 + 4e-4, o['size'][1] * .9 - 4e-4, o['size'][2]] for o in r['objects']}
           for r in labels}
    oe._publish_files(tmp / 'sizes-020-L', {'estimates.jsonl': oe._jsonl_bytes(
        [{'uid': u, 'estimated_sizes': s} for u, s in est.items() if u != il3d['uid']]), 'result.json': json.dumps(
        {'status': 'SIZES_COMPLETE', 'failed': {il3d['uid']: REASON}, 'gpu_hours': 0}).encode()},
        {'expected_uids': [r['uid'] for r in labels]})
    for e in ('e2', 'e3'):
        getattr(oe, f'plan_{e}')(data, e2world.official, e1, tmp / f'plan-{e}-llm-L', size_mode='llm-L',
                                 estimates=tmp / 'sizes-020-L')
    return SimpleNamespace(data=data, official=e2world.official, e1=e1, tmp=tmp, labels=labels, prior=prior,
                           rooms={r['uid']: r for r in labels}, holo=holo['uid'], il3d=il3d['uid'])


# ---- optiscene_eval: size modes ------------------------------------------------------------------------------------

def test_load_estimates_failure_reasons_and_rounding(tmp_path):
    rooms = [{'uid': u, 'objects': [{'id': 'a'}, {'id': 'b'}]} for u in 'pqrstuvwx']
    rows = [{'uid': 'p', 'estimated_sizes': {'a': [0.004, 0.5449999, 1], 'b': [2, 2, 2]}},
            {'uid': 'q', 'estimated_sizes': None, 'error': 'parse_error'},
            {'uid': 'r', 'estimated_sizes': {'a': [1, 1, 1]}},
            {'uid': 's', 'estimated_sizes': {'a': [1, 1, 1], 'b': [1, 1, 1], 'c': [1, 1, 1]}},
            {'uid': 't', 'estimated_sizes': {'a': [1, 1, 0], 'b': [1, 1, 1]}},
            {'uid': 'u', 'estimated_sizes': {'a': [1, 1, True], 'b': [1, 1, 1]}},
            {'uid': 'w', 'estimated_sizes': {'a': [1, 1, 999.996], 'b': [1, 1, 1]}}]
    def sizer(name, rows, status='SIZES_COMPLETE', uids='pqrstuvwx', result=True):
        files = {'estimates.jsonl': oe._jsonl_bytes(rows)}
        if result:
            files['result.json'] = json.dumps({'status': status, 'failed': {'x': 'API_FAILED'}}).encode()
        oe._publish_files(tmp_path / name, files, {'expected_uids': list(uids)})
        return tmp_path / name
    assert oe.load_estimates(sizer('sizes', rows), rooms) == {
        'p': {'a': [0.01, 0.54, 1.0], 'b': [2.0, 2.0, 2.0]}, 'q': 'parse_error', 'r': 'missing_object',
        's': 'extra_object', 't': 'bad_size', 'u': 'bad_size', 'v': 'missing_row', 'w': 'bad_size', 'x': 'API_FAILED'}
    with pytest.raises(ValueError, match='duplicate'):
        oe.load_estimates(sizer('dup', rows[:1] * 2), rooms)
    for name, kw in (('limited', {'uids': 'pqrstuvw'}), ('incomplete', {'status': 'INCOMPLETE'}),
                     ('no-result', {'result': False})):  # a --limit smoke run, an unfinished run: refused, not failures
        with pytest.raises(ValueError, match='not a complete sizer run'):
            oe.load_estimates(sizer(name, rows, **kw), rooms)


def test_llm_size_mode_plan_changes_only_sizes_and_carries_failed_rooms(w, monkeypatch):
    plan = w.tmp / 'plan-e2-llm-L'
    rows = oe._jsonl(plan / 'requests.jsonl')
    assert [(r['uid'], r['format'], r['mode']) for r in rows] == [
        (u, f, 'llm-L') for u in (w.holo, w.il3d) for f in ('optiscene', 'v1')]
    base = {(r['uid'], r['format']): r for r in oe._jsonl(w.tmp / 'e2' / 'requests.jsonl') if r['mode'] == 'prior'}
    sizes = oe.load_estimates(w.tmp / 'sizes-020-L', w.labels)
    for r in rows:
        old = base[(r['uid'], r['format'])]
        if r['uid'] == w.il3d:  # carried with its reason, no messages: never dropped, never generated
            assert r == {**{k: old[k] for k in ('uid', 'population', 'stratum', 'source', 'format')}, 'mode': 'llm-L',
                         'key': f'{w.il3d}|llm-L|{r["format"]}', 'size_error': REASON, 'messages': {}}
            continue
        same = lambda x: {k: v for k, v in x.items() if k not in ('key', 'mode', 'boxes', 'messages')}
        assert same(r) == same(old) and r['key'] == f'{w.holo}|llm-L|{r["format"]}'
        if r['format'] == 'optiscene':
            s = sizes[w.holo]
            assert r['boxes'] == [[s[i][2], s[i][1], s[i][0]] for i in r['ids']] != old['boxes']
            got, want = (oe.parse_input(x['messages']['sft'][0]['content'][len(INSTR):]) for x in (r, old))
            assert got[:3] == want[:3] and got[3] == r['boxes']
        else:
            user = json.loads(r['messages']['v1'][1]['content'])
            assert {r['id_map'][o['id']]: o['size'] for o in user['objects']} == sizes[w.holo]
    m, default = oe._verified(plan), oe._verified(w.tmp / 'e2')
    assert (m['protocol'], m['size_mode'], m['size_failures']) == ('amendment-020', 'llm-L', {w.il3d: REASON})
    assert m['size_estimates_manifest_sha256'] == oe._sha(w.tmp / 'sizes-020-L' / 'MANIFEST.json')
    new = ('protocol', 'size_mode', 'size_estimates', 'size_estimates_manifest_sha256', 'size_failures', 'files')
    assert {k: v for k, v in m.items() if k not in new} == {k: v for k, v in default.items() if k not in new}
    assert (plan / 'exposure.json').read_bytes() == (w.tmp / 'e2' / 'exposure.json').read_bytes()
    e3 = oe._jsonl(w.tmp / 'plan-e3-llm-L' / 'requests.jsonl')
    assert {(r['uid'], r['stratum'], 'size_error' in r) for r in e3} == {(S3D, 'garage', False)}
    oe._publish_files(w.tmp / 'model', {'model.safetensors': b'w'}, {})
    llm = FakeLLM(lambda p: 'no')
    monkeypatch.setattr(oe, '_engine', lambda engine: (llm, lambda **kw: kw))
    g = oe.generate(plan, 'sft', w.tmp / 'model', w.tmp / 'gen')
    assert (g['requests'], g['protocol']) == (1, 'amendment-020') and len(llm.params) == 1
    for kw in ({'size_mode': 'llm-L'}, {'estimates': w.tmp / 'sizes-020-L'},
               {'size_mode': 'gt', 'estimates': w.tmp / 'sizes-020-L'}):
        with pytest.raises(ValueError, match='needs --estimates'):
            oe.plan_e2(w.data, w.official, w.e1, w.tmp / 'refused', **kw)


def test_main_parses_the_size_mode_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(oe, 'plan_e3', lambda *a, **kw: seen.update(args=a, kw=kw) or {
        k: 0 for k in ('rooms', 'families', 'skipped_families', 'exposure_counts')})
    oe.main(['plan-e3', '--eval-data', 'D', '--data', 'O', '--e1', 'E', '--out', 'X', '--size-mode', 'llm-O2',
             '--estimates', 'S'])
    assert seen['kw'] == {'size_mode': 'llm-O2', 'estimates': Path('S')}
    with pytest.raises(SystemExit):
        oe.main(['plan-e2', '--eval-data', 'D', '--data', 'O', '--e1', 'E', '--out', 'X', '--size-mode', 'prior'])


def old_optiscene_eval(tmp_path, monkeypatch):
    """optiscene_eval.py at BASE_COMMIT, sharing this test's patched constants and data paths."""
    run = subprocess.run(['git', '-C', str(CODE), 'show', f'{BASE_COMMIT}:fastfill/comparison/optiscene_eval.py'],
                         capture_output=True, text=True)
    if run.returncode:
        pytest.skip(f'git history without {BASE_COMMIT}')
    path = tmp_path / 'optiscene_eval_base.py'
    path.write_text(run.stdout)
    spec = importlib.util.spec_from_file_location('optiscene_eval_base', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for name in ('VENDORED', 'DATA_FILES', 'INSTRUCTION', 'families'):
        monkeypatch.setattr(old, name, getattr(oe, name))
    return old


CODE_HASHED = ('implementation_sha256', 'plan_manifest_sha256', 'plans', 'gens', 'plan_e3_manifest_sha256')


def same_outputs(a, b):
    """Every output file byte-identical; MANIFESTs equal apart from code hashes (and hashes of manifests holding them)."""
    files = lambda d: {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob('*'))
                       if p.is_file() and p.name != 'MANIFEST.json'}
    manifest = lambda d: {k: v for k, v in json.loads((d / 'MANIFEST.json').read_text()).items() if k not in CODE_HASHED}
    assert files(a) == files(b) and files(a)
    assert manifest(a) == manifest(b)


def p_run(w):
    """P's test run over eval-data (answer = the GT placement at prior sizes), with its geometry-v5 rows."""
    rows, records = [], []
    for r in w.labels:
        request = contract.messages(r, 'placement', estimated_sizes=w.prior[r['uid']])
        rows.append(geometry.score(r, request[:2], request[2]['content'], 'placement'))
        records.append({'uid': r['uid'], 'reached_token_cap': False, 'stopped_on_eos': True})
    receipt = {**POLICY, 'config': {**CONFIG, 'data': str(w.data), 'plan': '/r/infer-v2-005/plan', 'shard_count': 6},
               'data': {'manifest_sha256': oe._sha(w.data / 'MANIFEST.json')}, 'run_sha256': 'r' * 64}
    oe._publish_files(w.tmp / 'geometry-v5', {'placement-geometry.jsonl': oe._jsonl_bytes(rows), 'summary.json': json.dumps(
        {'inference_run_sha256': {'placement': 'r' * 64}}).encode()}, {})
    oe._publish_files(w.tmp / 'infer-placement-005', {'records.jsonl': oe._jsonl_bytes(records),
                                                      'run_receipt.json': json.dumps(receipt).encode()}, {})
    return {'p_geometry': w.tmp / 'geometry-v5', 'p_run': w.tmp / 'infer-placement-005'}


def gt_records(w, plan, prompts=('sft',)):
    return [{'key': r['key'], 'uid': r['uid'], 'finish_reason': 'stop', 'error': None,
             'text': v1_text(r, w.rooms[r['uid']]) if r['format'] == 'v1' else gt_design({'room': w.rooms[r['uid']], **r})}
            for r in oe._jsonl(plan / 'requests.jsonl') if any(p in r['messages'] for p in prompts)]


def test_default_plan_generate_and_score_outputs_equal_the_pre_020_code(world, w, monkeypatch, tmp_path):
    (tmp_path / 'base').mkdir()
    old = old_optiscene_eval(tmp_path / 'base', monkeypatch)
    e1 = tmp_path / 'e1-full'
    oe.plan_e1(world.hf, world.data, e1)
    oe._publish_files(tmp_path / 'model', {'model.safetensors': b'w'}, {})
    p = p_run(w)
    for mod, root in ((old, tmp_path / 'old'), (oe, tmp_path / 'new')):
        monkeypatch.setattr(mod, '_engine', lambda engine: (FakeLLM(lambda q: 'no'), lambda **kw: kw))
        mod.plan_e2(w.data, w.official, e1, root / 'e2')
        mod.plan_e3(w.data, w.official, e1, root / 'e3')
        mod.generate(root / 'e2', 'sft', tmp_path / 'model', root / 'generated')
        plans = {'e1': e1, 'e2': root / 'e2'}
        for model, plan, prompt in oe.GENS:
            records = gt_records(w, plans[plan], (prompt,)) if plan == 'e2' else [
                {'key': r['key'], 'uid': r['uid'], 'finish_reason': 'stop', 'error': None,
                 'text': gt_design(r) if r['population'] == 'e1-unseen' else 'no answer'}
                for r in oe._jsonl(e1 / 'requests.jsonl') if prompt in r['messages']]
            publish_gen(root / 'gen', f'{model}-{plan}-{prompt}', plans[plan], records)
        for name in ('dpo2-e3-sft', '8b-e3-v1'):
            publish_gen(root / 'gen3', name, root / 'e3', gt_records(w, root / 'e3', (name.rsplit('-', 1)[1],)))
        mod.score(e1, root / 'e2', root / 'gen', w.data, p['p_geometry'], p['p_run'], root / 'score', repeats=2)
        mod.score_e3(root / 'e3', root / 'gen3', w.data, p['p_geometry'], p['p_run'], root / 'score-e3', repeats=2)
    for name in ('e2', 'e3', 'generated', 'score', 'score-e3'):
        same_outputs(tmp_path / 'old' / name, tmp_path / 'new' / name)


# ---- P data --------------------------------------------------------------------------------------------------------

def test_p_data_changes_only_sizes_and_rebuilds_the_prior_requests(w):
    m = pl.p_data(w.data, w.tmp / 'plan-e2-llm-L', w.tmp / 'plan-e3-llm-L', w.tmp / 'sizes-020-L', 'llm-L',
                  w.tmp / 'p-data')
    assert m['expected_uids'] == [w.holo, S3D] and m['universe'] == {'e2': [w.holo, w.il3d], 'e3': [S3D]}
    assert m['size_failures'] == {w.il3d: REASON} and m['prior_identity'] == {'rooms': 3, 'byte_identical': 3}
    sizes = oe.load_estimates(w.tmp / 'sizes-020-L', w.labels)
    placement = {r['uid']: r for r in oe._jsonl(w.data / 'placement/requests.jsonl')}
    joint = {r['uid']: r for r in oe._jsonl(w.data / 'joint/requests.jsonl')}
    rows, _ = infer.load_eval(w.tmp / 'p-data', 'placement')  # what infer_shard plan/shard/merge read
    assert [r['uid'] for r in infer.load_eval(w.tmp / 'p-data', 'joint')[0]] == [w.holo, S3D]
    for r in rows:
        old = placement[r['uid']]
        assert pl.sizes_only(r, old) and r != old
        assert {o['id']: o['size'] for o in json.loads(r['messages'][1]['content'])['objects']} == sizes[r['uid']]
    assert oe._jsonl(w.tmp / 'p-data' / 'joint/requests.jsonl') == [joint[w.holo], joint[S3D]]
    old = placement[w.holo]
    user = json.loads(old['messages'][1]['content'])
    user['boundary'][0][0] += 1
    assert not pl.sizes_only({**old, 'messages': [old['messages'][0], {'role': 'user', 'content': json.dumps(user)}]}, old)
    assert not pl.sizes_only({**old, 'split': 'dev'}, old)
    user = json.loads(old['messages'][1]['content'])
    user['objects'][0]['size'][0] += 0.01
    tampered = {**placement, w.holo: {**old, 'messages': [old['messages'][0], {'role': 'user', 'content': json.dumps(user)}]}}
    with pytest.raises(ValueError, match='do not rebuild'):
        pl.prior_identity(w.rooms, tampered, w.prior)
    with pytest.raises(ValueError, match='both E2 and E3'):
        pl.universe(w.tmp / 'e2', w.tmp / 'e2')


# ---- score and report ----------------------------------------------------------------------------------------------

def p_infer(w, p_data, out, **change):
    """A merged infer_shard run on p_data (answer = GT placement at the request sizes); S3D hits the token cap."""
    manifest = json.loads((p_data / 'MANIFEST.json').read_text())
    sizes = {r['uid']: r['estimated_sizes'] for r in oe._jsonl(p_data / 'size-estimates.jsonl')}
    identity = {**POLICY, 'config': {**CONFIG, 'data': str(p_data), 'plan': str(out.parent / 'plan'), 'shard_count': 4},
                'data': {'manifest_sha256': oe._sha(p_data / 'MANIFEST.json')}, **change}
    run = infer._hash(identity)
    records = [{'uid': u, 'source': w.rooms[u]['source'], 'group': w.rooms[u]['group'], 'arm': 'placement',
                'raw': contract.messages(w.rooms[u], 'placement', estimated_sizes=sizes[u])[2]['content'], 'error': None,
                'run_sha256': run, 'reached_token_cap': u == S3D, 'stopped_on_eos': u != S3D}
               for u in manifest['expected_uids']]
    out.mkdir(parents=True)
    jsonl(out / 'records.jsonl', records)
    (out / 'result.json').write_text(json.dumps({'status': 'INFERENCE_COMPLETE', 'pending_count': 0}))
    (out / 'run_receipt.json').write_text(json.dumps({**identity, 'run_sha256': run}))
    (out / 'MANIFEST.json').write_text(json.dumps({'files': h.artifact_manifest(out),
                                                   'expected_uids': manifest['expected_uids']}))


def test_score_rows_pair_every_room_and_report_table_b(w, monkeypatch):
    tmp, p = w.tmp, p_run(w)
    pl.p_data(w.data, tmp / 'plan-e2-llm-L', tmp / 'plan-e3-llm-L', tmp / 'sizes-020-L', 'llm-L', tmp / 'p-data-llm-L')
    publish_gen(tmp / 'gen14', 'dpo2-e2-sft', tmp / 'e2', gt_records(w, tmp / 'e2'))
    publish_gen(tmp / 'gen15', 'dpo2-e3-sft', tmp / 'e3', gt_records(w, tmp / 'e3'))
    for e in ('e2', 'e3'):
        publish_gen(tmp / 'gen20', f'dpo2-{e}-llm-L-sft', tmp / f'plan-{e}-llm-L', gt_records(w, tmp / f'plan-{e}-llm-L'))
    p_infer(w, tmp / 'p-data-llm-L', tmp / 'p-llm-L' / 'infer')
    common = {'eval_data': w.data, **p}
    ours = {'e2': tmp / 'plan-e2-llm-L', 'e3': tmp / 'plan-e3-llm-L'}
    for mode in ('prior', 'gt'):
        pl.score('os', mode, e2=tmp / 'e2', e3=tmp / 'e3', out=tmp / 'rows' / f'os-{mode}', gen_e2=tmp / 'gen14/dpo2-e2-sft',
                 gen_e3=tmp / 'gen15/dpo2-e3-sft', **common)
    pl.score('os', 'llm-L', **ours, out=tmp / 'rows/os-llm-L', gen_e2=tmp / 'gen20/dpo2-e2-llm-L-sft',
             gen_e3=tmp / 'gen20/dpo2-e3-llm-L-sft', p_data=tmp / 'p-data-llm-L', workers=2, **common)
    m = pl.score('p', 'prior', e2=tmp / 'e2', e3=tmp / 'e3', out=tmp / 'rows/p-prior', **common)
    assert m['inputs']['prior_identity'] == {'rooms': 3, 'byte_identical': 3}
    pl.score('p', 'llm-L', **ours, out=tmp / 'rows/p-llm-L', p_data=tmp / 'p-data-llm-L', p_infer=tmp / 'p-llm-L/infer',
             **common)
    rows = {a: oe._jsonl(tmp / 'rows' / a / 'rows.jsonl') for a in ('os-prior', 'os-gt', 'os-llm-L', 'p-prior', 'p-llm-L')}
    for a, rs in rows.items():
        assert [(r['uid'], r['population'], r['family'], r['arm']) for r in rs] == [
            (w.holo, 'e2', 'bedroom', a), (w.il3d, 'e2', 'living', a), (S3D, 'e3', 'garage', a)]
    assert all(r['complete'] and r['geometry_success'] and r['geometry_success_s1'] and r['size_log_mae'] < 1e-9
               for r in rows['os-gt'])  # GT design, GT sizes
    osl, pll = ({r['uid']: r for r in rows[a]} for a in ('os-llm-L', 'p-llm-L'))
    for failed in (osl[w.il3d], pll[w.il3d]):  # the sizer failure is a failure row in every rate
        assert failed['error'] == 'size:' + REASON and failed['s1_status'] == 'not_complete'
        assert not any(failed[k] for k in pl.RATES) and failed['oor'] is None
    assert osl[w.holo]['complete'] and osl[S3D]['complete'] and osl[w.holo]['size_log_mae'] > 0.05
    assert pll[S3D]['error'] == report_v2.CAP and pll[w.holo]['complete']  # amendment-005 cap rule
    assert osl[w.holo]['size_log_mae'] == pytest.approx(pll[w.holo]['size_log_mae'])  # both placers: the same sizes
    geo = {r['uid']: r for r in oe._jsonl(tmp / 'geometry-v5' / 'placement-geometry.jsonl')}
    assert [(r['complete'], r['geometry_success'], r['oor']) for r in rows['p-prior']] == [
        (geo[u]['complete'], geo[u]['geometry_success'], geo[u]['oor']) for u in (w.holo, w.il3d, S3D)]
    with pytest.raises(ValueError, match='is not the llm-O2 P data'):
        pl.score('os', 'llm-O2', **ours, out=tmp / 'refused', gen_e2=tmp / 'gen20/dpo2-e2-llm-L-sft',
                 gen_e3=tmp / 'gen20/dpo2-e3-llm-L-sft', p_data=tmp / 'p-data-llm-L', **common)
    p_infer(w, tmp / 'p-data-llm-L', tmp / 'p-other' / 'infer', stage='dpo2')
    with pytest.raises(ValueError, match='exact policy'):
        pl.score('p', 'llm-L', **ours, out=tmp / 'refused', p_data=tmp / 'p-data-llm-L', p_infer=tmp / 'p-other/infer',
                 **common)

    pl.report(tmp / 'rows', tmp / 'report', repeats=20)
    s = json.loads((tmp / 'report' / 'summary.json').read_text())
    assert s['not_published'] == ['os-llm-O2', 'p-llm-O2'] and s['not_computed'] == ['llm-O2: OS-DPO2 − P']
    b = s['table_b']
    assert list(b) == ['E2', 'E3', 'bedroom', 'garage', 'living'] and set(b['E2']) == set(rows)
    assert (b['E2']['os-llm-L']['rooms'], b['E2']['os-llm-L']['size_failures'], b['E2']['os-llm-L']['complete']) == (2, 1, .5)
    assert b['E3']['p-llm-L']['complete'] == 0.0 and b['living']['p-llm-L']['rooms'] == 1
    c = s['contrasts']['OS-DPO2: llm-L − prior']['E2']
    assert (c['complete']['estimate'], c['complete']['ci95'], c['components'], c['complete']['n']) == (-.5, None, 2, 2)
    assert s['contrasts']['llm-L: OS-DPO2 − P']['E3']['complete']['estimate'] == 1.0  # P's S3D answer hit the cap
    assert s['e3_minus_e2']['p-llm-L']['complete'] == -.5
    md = (tmp / 'report' / 'report.md').read_text()
    assert '表 B' in md and '| E2 | llm-L → OS-DPO2 | 2 | 1 | 0.500 |' in md and '未发布' in md
    monkeypatch.setattr(report_v2, 'NO_CI', 1)
    pl.report(tmp / 'rows', tmp / 'report-ci', repeats=200)
    ci = json.loads((tmp / 'report-ci' / 'summary.json').read_text())['contrasts']['P: llm-L − prior']['E2']
    assert ci['complete']['ci95'] is not None
    shorter = oe._jsonl(tmp / 'rows/p-prior/rows.jsonl')[:2]
    pl._publish(tmp / 'rows2' / 'p-prior', {'rows.jsonl': oe._jsonl_bytes(shorter)}, {})
    pl._publish(tmp / 'rows2' / 'os-prior', {'rows.jsonl': (tmp / 'rows/os-prior/rows.jsonl').read_bytes()}, {})
    with pytest.raises(ValueError, match='different rooms'):
        pl.report(tmp / 'rows2', tmp / 'report-2', repeats=20)


# ---- GPU budget ----------------------------------------------------------------------------------------------------

SACCT_LINES = [
    '8001|unknownsize-020-sizer-O2|COMPLETED|10800|480|billing=72,cpu=72,gres/gpu=2,mem=220G,node=1|cpu=16,gres/gpu=2',
    '8001.batch|batch|COMPLETED|10800|||',
    '8002|unknownsize-020-sizer-L|COMPLETED|3600|480|billing=1,cpu=1,mem=8G,node=1|cpu=1',
    '8003|placers-020-p-shard-llm-L-0|PENDING|0|60||cpu=16,gres/gpu=1,mem=110G',
    '8004|placers-020-os-gen-llm-L|CANCELLED by 1|1800|60|gres/gpu=1|gres/gpu=1',
    '8005|unknownsize-020-report|RUNNING|100|120|cpu=4|cpu=4',
    '8006|other|RUNNING|100|600|gres/gpu=4|gres/gpu=4']


def test_gpu_budget_charges_sizers_and_earlier_placer_jobs(tmp_path, monkeypatch):
    assert pl.gpu_hours(SACCT_LINES, {'8005'}) == {
        '8001': ('unknownsize-020-sizer-O2', 6.0), '8002': ('unknownsize-020-sizer-L', 0.0),
        '8003': ('placers-020-p-shard-llm-L-0', 1.0), '8004': ('placers-020-os-gen-llm-L', 0.5),
        '8005': ('unknownsize-020-report', 0.0)}
    sizes = tmp_path / 'sizes-020-O2'
    oe._publish_files(sizes, {'result.json': json.dumps({'gpu_hours': 5.5}).encode()}, {})
    calls = []
    run = lambda argv, **kw: calls.append(argv) or SimpleNamespace(stdout='\n'.join(SACCT_LINES) + '\n')
    monkeypatch.setenv('USER', 'me')
    r = pl.budget(30, 14, ['8005'], [sizes, tmp_path / 'missing'], run=run)
    assert (r['sizers_gpu_h'], r['earlier_placer_jobs_gpu_h'], r['total_gpu_h']) == (6.0, 1.5, 21.5)
    assert calls[0][:6] == ['sacct', '-X', '-n', '-P', '-u', 'me'] and r['sizer_receipts_gpu_h'] == {str(sizes): 5.5}
    receipt_only = lambda argv, **kw: SimpleNamespace(stdout='\n'.join(SACCT_LINES[3:5]) + '\n')
    assert pl.budget(30, 14, [], [sizes], run=receipt_only)['total_gpu_h'] == 21.0  # the receipt: 5.5 + 1.5 + 14
    with pytest.raises(ValueError, match='cap exceeded'):
        pl.budget(21, 14, [], [sizes], run=run)
    with pytest.raises(ValueError, match='no sizer GPU use'):
        pl.budget(30, 14, [], [tmp_path / 'missing'], run=receipt_only)
    with pytest.raises(ValueError, match='does not know'):
        pl.budget(30, 14, ['9999'], [], run=run)
    # A job pending on a dependency is missed by the -S window; budget asks sacct for it by id.
    window = [x for x in SACCT_LINES if not x.startswith('8005|')]
    byid = [x for x in SACCT_LINES if x.startswith('8005|')]
    seen = []
    def pending(argv, **kw):
        seen.append(argv)
        return SimpleNamespace(stdout='\n'.join(byid if '-j' in argv else window) + '\n')
    assert pl.budget(30, 14, ['8005'], [sizes], run=pending)['total_gpu_h'] == 21.5
    assert seen[1][:6] == ['sacct', '-X', '-n', '-P', '-j', '8005'] and seen[1][6:] == list(pl.SACCT[2:])


# ---- submit_020_placers.sh -----------------------------------------------------------------------------------------

@pytest.mark.parametrize('args, ok', [
    (['job', 'plan'], False), (['job', 'plan', 'llm-L'], True), (['job', 'plan', 'prior'], False), (['job', 'report'], True), (['job', 'os-gen', 'llm-L'], True), (['job', 'os-gen', 'gt'], False),
    (['job', 'p-shard', 'llm-O2', '5'], True), (['job', 'p-shard', 'llm-O2', '6'], False), (['job', 'p-merge', 'llm-L'], True),
    (['job', 'score', 'os', 'gt'], True), (['job', 'score', 'p', 'gt'], False), (['job', 'score', 'p', 'llm-O2'], True),
    (['job', 'report', 'x'], False), (['job', 'other'], False), (['1:x'], False), (['1:'], False), (['1', '2'], False)])
def test_submit_020_argument_checks(tmp_path, args, ok):
    run = subprocess.run(['bash', str(SCRIPTS / 'submit_020_placers.sh'), str(CODE), *args],
                         env=T.stub_sbatch(tmp_path), capture_output=True, text=True)
    assert (run.returncode, 'Slurm allocation required' in run.stderr) == ((1, True) if ok else (2, False))


def test_submit_020_chain_dependencies_roots_budget_and_refusals(tmp_path):
    project, code = tmp_path / 'project', tmp_path / 'code'
    run = project / 'runs/unknownsize-20261013'
    paths = [run / p for p in ('eval-data-v1/MANIFEST.json', 'geometry-v5/MANIFEST.json', 'dev-eval-010/selection.json',
                               'optiscene-014/e2/MANIFEST.json', 'optiscene-015/e3/MANIFEST.json',
                               'sizes-020-L/MANIFEST.json', 'sizes-020-O2/MANIFEST.json',
                               'dpo-003/placement/dpo1-merged.json')]
    paths += [project / p for p in ('data/optiscene-014-e1/MANIFEST.json', 'data/optiscene-014-e1/hf-coverage.json',
                                    'models/Qwen3-8B/config.json')]
    paths += [project / 'data/optiscene-official' / n for n, _ in oe.DATA_FILES.values()]
    paths += [code / 'fastfill/comparison' / n for n in ('optiscene_eval.py', 'placers_020.py', 'infer_shard.py',
                                                         'submit_020_placers.sh')]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    for py in ('env/bin/python', 'vllm/bin/python'):  # Only the import check and the budget run here: exit codes.
        (project / py).parent.mkdir(parents=True, exist_ok=True)
        (project / py).write_text('#!/bin/sh\necho "$*" >> "$SBATCH_LOG.py"\n'
                                  'case "$*" in *budget*) exit "${BUDGET_EXIT:-0}" ;;\n'
                                  '  *"sizers_020 cap"*) printf "%s" "$CAP_OUT"; exit 1 ;; esac\nexit "${PY_EXIT:-0}"\n')
        (project / py).chmod(0o755)
    text = (SCRIPTS / 'submit_020_placers.sh').read_text()
    for line, new in (('PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n', f'PROJECT_ROOT={project}\n'),
                      ('VLLM_PY=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python\n',
                       f'VLLM_PY={project}/vllm/bin/python\n')):
        assert text.count(line) == 1
        text = text.replace(line, new)
    script = tmp_path / 'submit_020_placers.sh'
    script.write_text(text)
    for name, body in (('squeue', 'test "$*" = "-h -u me -o %i %j" || exit 3\nprintf "%s" "$SQUEUE_OUT"'),
                       ('sacct', 'test "$*" = "-j $2 -X -n -P -o State" || exit 3\necho "$SACCT_STATE"')):
        (tmp_path / name).write_text(f'#!/bin/sh\n{body}\n')
        (tmp_path / name).chmod(0o755)
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me', 'SACCT_STATE': 'COMPLETED', 'CAP_OUT': '{"ok": false}',
           'SQUEUE_OUT': '7001 optiscene-014-merge-dpo2\n7002 optiscene-014-gen-dpo2\n7003 optiscene-015-gen-dpo2\n'}
    submit = lambda *args, **kw: subprocess.run(['bash', str(script), str(code), *args], env={**env, **kw},
                                                capture_output=True, text=True)
    lines = lambda start=0: [x.replace(f'{code}/fastfill/comparison/submit_020_placers.sh {code} job ', 'B ')
                             for x in (tmp_path / 'log').read_text().splitlines()[start:]]
    done = submit()
    assert done.returncode == 0, done.stderr
    k, gpu = '--parsable --kill-on-invalid-dep=yes', '--gpus-per-node=1 --cpus-per-task=16 --mem=110G --time=01:00:00'
    cpu, score = '--cpus-per-task=16 --mem=64G', '--cpus-per-task=32 --mem=128G --time=04:00:00'
    n = lambda label: f'--job-name=placers-020-{label}'
    plan = lambda m, after='': f'{k} --hold{after} {n(f"plan-{m}")} {cpu} --time=02:00:00 B plan {m}'
    shards = lambda m, first, plan, ks=range(6): [
        f'{k} --dependency=afterok:{plan} {n(f"p-shard-{m}-{i}")} {gpu} B p-shard {m} {i}' for i in ks] + [
        f'{k} --dependency=afterok:{":".join(str(first + j) for j in range(len(ks))) or plan} {n(f"p-merge-{m}")} {cpu} '
        f'--time=01:00:00 B p-merge {m}',
        f'{k} --dependency=afterok:{first + len(ks)} {n(f"score-p-{m}")} {score} B score p {m}']
    report = lambda deps: f'{k} --dependency=afterany:{deps} {n("report")} {cpu} --time=04:00:00 B report'
    assert lines() == [
        plan('llm-L'), plan('llm-O2'),
        f'{k} --hold --dependency=afterany:7002:7003 {n("score-os-prior")} {score} B score os prior',
        f'{k} --hold --dependency=afterany:7002:7003 {n("score-os-gt")} {score} B score os gt',
        f'{k} --dependency=afterok:101:7001 {n("os-gen-llm-L")} {gpu} B os-gen llm-L',
        f'{k} --dependency=afterok:105 {n("score-os-llm-L")} {score} B score os llm-L',
        f'{k} --dependency=afterok:102:7001 {n("os-gen-llm-O2")} {gpu} B os-gen llm-O2',
        f'{k} --dependency=afterok:107 {n("score-os-llm-O2")} {score} B score os llm-O2',
        f'{k} --hold {n("score-p-prior")} {score} B score p prior',
        *shards('llm-L', 110, 101), *shards('llm-O2', 118, 102), report('103:104:106:108:109:117:125')]
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101,102,103,104,109\n'
    assert sum(int(x.split('--time=')[1][:2]) for x in lines() if '--gpus-per-node=1' in x) == 14
    budget = (tmp_path / 'log.py').read_text().splitlines()[-1]
    assert budget.endswith(f'budget --cap 30 --planned 14 --sizes {run}/sizes-020-L {run}/sizes-020-O2 --jobs')
    sizers = submit('8001:8002', SQUEUE_OUT='8001 sizer-O2\n')  # 8001 still in Slurm, 8002 ended COMPLETED
    assert sizers.returncode == 0 and 'skip OS-DPO2' in sizers.stdout
    assert lines(26) == [plan('llm-L', ' --dependency=afterok:8001'), plan('llm-O2', ' --dependency=afterok:8001'),
                         f'{k} --hold {n("score-p-prior")} {score} B score p prior', *shards('llm-L', 130, 127),
                         *shards('llm-O2', 138, 128), report('129:137:145')]
    assert (tmp_path / 'log.py').read_text().splitlines()[-1].endswith('--planned 12 --sizes '
                                                                       f'{run}/sizes-020-L {run}/sizes-020-O2 --jobs 8001 8002')
    # The L sizer refused by the API cap: llm-L is dropped; a rerun submits and plans only unpublished GPU steps.
    (run / 'sizes-020-L/MANIFEST.json').unlink()
    (run / 'sizes-020-L/status.json').write_text(json.dumps({'status': 'CAP_REFUSED', 'needed': 3754}, indent=2))
    out = run / 'optiscene-020'
    for d in ('gen/dpo2-e2-llm-O2-sft', 'gen/dpo2-e3-llm-O2-sft', 'p-llm-O2/shard-00', 'p-llm-O2/shard-02'):
        (out / d).mkdir(parents=True)
        (out / d / 'MANIFEST.json').write_text('{}')
    start = len(lines())
    refused_l = submit(SQUEUE_OUT='', CAP_OUT='{"ok": true}')
    assert refused_l.returncode == 0 and 'skip llm-L' in refused_l.stdout and 'llm-L' not in '\n'.join(lines(start))
    assert lines(start) == [plan('llm-O2'), f'{k} --hold {n("score-p-prior")} {score} B score p prior',
                            *shards('llm-O2', 149, 147, (1, 3, 4, 5)), report('148:154')]
    assert (tmp_path / 'log.py').read_text().splitlines()[-1].endswith('--planned 4 --sizes '
                                                                       f'{run}/sizes-020-L {run}/sizes-020-O2 --jobs')
    (run / 'optiscene-014/dpo2-merged').mkdir()
    (run / 'optiscene-014/dpo2-merged/MANIFEST.json').write_text('{}')
    (out / 'p-llm-O2/infer').mkdir()
    (out / 'p-llm-O2/infer/MANIFEST.json').write_text('{}')
    start = len(lines())
    assert submit(SQUEUE_OUT='').returncode == 0
    assert lines(start) == [plan('llm-O2'), f'{k} --hold {n("score-os-prior")} {score} B score os prior',
                            f'{k} --hold {n("score-os-gt")} {score} B score os gt',
                            f'{k} --dependency=afterok:156 {n("score-os-llm-O2")} {score} B score os llm-O2',
                            f'{k} --hold {n("score-p-prior")} {score} B score p prior',
                            *shards('llm-O2', 161, 156, ()), report('157:158:159:160:162')]
    assert '--planned 0 ' in (tmp_path / 'log.py').read_text().splitlines()[-1]
    # No L output yet: the cap preflight decides (refuses -> dropped; passes -> L must be published or awaited).
    (run / 'sizes-020-L/status.json').unlink()
    (run / 'sizes-020-L').rmdir()
    start = len(lines())
    cap = submit(SQUEUE_OUT='', CAP_OUT='{"used": 0, "needed": 3754, "ok": false}')
    assert cap.returncode == 0 and 'cap preflight refuses' in cap.stdout and lines(start)[0] == plan('llm-O2')
    assert 'sizers_020 cap --run-root' in (tmp_path / 'log.py').read_text()
    count = len(lines())
    for args, kw, message in ((('8002',), {'SACCT_STATE': 'FAILED'}, "ended 'FAILED'"),
                              ((), {'CAP_OUT': '{"ok": true}'}, 'sizes-020-L not published'),
                              ((), {'CAP_OUT': 'Traceback'}, 'cap preflight failed'),
                              ((), {'BUDGET_EXIT': '1'}, 'GPU cap'), ((), {'PY_EXIT': '1'}, 'cannot import'),
                              ((), {'SQUEUE_OUT': '9001 placers-020-plan\n'}, 'scancel 9001')):
        refused = submit(*args, **kw)
        assert refused.returncode == 1 and message in refused.stderr and 'nothing submitted' in refused.stderr
    (run / 'sizes-020-O2/MANIFEST.json').unlink()
    refused = submit()
    assert refused.returncode == 1 and 'sizes-020-O2 not published' in refused.stderr
    assert len(lines()) == count  # Nothing new was submitted.
    failed = tmp_path / 'failed'
    failed.mkdir()
    for name in ('squeue', 'sacct'):
        (failed / name).write_bytes((tmp_path / name).read_bytes())
        (failed / name).chmod(0o755)
    bad = subprocess.run(['bash', str(script), str(code), '8001'],
                         env={**env, **T.stub_sbatch(failed, fail_on=3), 'SQUEUE_OUT': '8001 sizer\n'},
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "sbatch failed for 'score-os-gt'" in bad.stderr and 'scancel 101 102' in bad.stderr
    assert not (failed / 'log.scontrol').exists()
