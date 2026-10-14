"""Amendment 018 chain on CPU (SFT-D -> DPO1-D -> DPO2-D, dev selection, test chain): fakes stand in for vLLM,
train_dpo, merge_lora, the models and Slurm."""
import inspect
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from fastfill.comparison import contract
from fastfill.comparison import dev_select as ds
from fastfill.comparison import distill_018_chain as ch
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import infer
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_dev_select as DS
from fastfill.comparison.tests import test_dpo_stages as D
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests import test_onpolicy_dpo as OP
from fastfill.comparison.tests.test_dev_select import dev  # noqa: F401  Fixture: 5 frozen dev rooms.
from fastfill.comparison.tests.test_dpo_stages import pipeline  # noqa: F401  Fixture: dpo-001 on CPU.
from fastfill.comparison.tests.test_onpolicy_dpo import main, world  # noqa: F401  Fixtures: main data; T-A + samples.

CODE, SCRIPTS, KILL, ns, lines = OP.CODE, OP.SCRIPTS, OP.KILL, OP.ns, OP.lines
RPO = ({'stage': 'dpo1', 'training': {'candidate': 'rpo'}}, OP.POLICY)


# ---- amendment-013 identities ----------------------------------------------------------------------

def test_onpolicy_dpo_defaults_stay_amendment_013(world):
    params = inspect.signature(op.build_pairs).parameters
    assert (params['scope'].default, params['sampler'].default, params['short'].default) == (
        op.SUBSET_SCOPE, ('dpo1', 'rpo'), None)  # 013's call still raises below the floors (test_onpolicy_dpo).
    assert inspect.signature(op._scored).parameters['bind'].default is op._bind
    op.build_pairs(OP.pair_args(world, world.root / 'default'))
    op.build_pairs(OP.pair_args(world, world.root / 'explicit'), scope=op.SUBSET_SCOPE, sampler=('dpo1', 'rpo'))
    for name in ('placement.jsonl', 'ledger.jsonl', 'summary.json', 'MANIFEST.json'):
        assert (world.root / 'default' / name).read_bytes() == (world.root / 'explicit' / name).read_bytes()
    with pytest.raises(ValueError, match="rpo-dpo1's exact policy"):  # 013's call still refuses any other sampler.
        op.build_pairs(OP.pair_args(world, world.root / 'x'), sampler=('dpo2', 'dpo1-d'))
    with pytest.raises(ValueError, match='amendment-013 training subset only'):
        op.build_pairs(OP.pair_args(world, world.root / 'y'), scope=ch.TC_SCOPE)


# ---- SFT-D rows ------------------------------------------------------------------------------------

def moved(text, x):
    value = json.loads(text)
    value['placements'][0]['pos'][0] = x
    return contract._json(value)


def teacher(root, targets):
    """A stand-in RUN/distill-018/teacher as distill_018.load_targets reads it."""
    d._publish_dir(root, {'targets.jsonl': d._jsonl(targets),
                          'summary.json': op._json_bytes({'status': 'TEACHER_TARGETS_COMPLETE'})})
    return root


def sft_args(w, out, teacher_dir, dev_dirs=None):
    return ch.parse_args(['sft-data', '--data', str(w.data), '--sft-runs', *map(str, w.runs), '--exclude',
                          str(w.positives), '--teacher', str(teacher_dir), '--base-model',
                          str(w.root / 'dpo-003/placement/dpo1-merged'), '--dev', *map(str, dev_dirs or w.dev),
                          '--test', *map(str, w.test), '--out', str(out)])


def test_sft_data_binds_teacher_targets_to_p_prompts_and_the_frozen_checker(main, monkeypatch):
    monkeypatch.setattr(d, 'exact_policy', lambda path, arm: RPO)
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: OP.Tok())
    train = h._by_uid(h.load_pair(main.data)[0]['train']['placement'])
    answer = {u: train[u]['messages'][2]['content'] for u in train}  # Pool: f1, h4 fail the checker as GT; x1 passes.
    good = {'x1': answer['x1'], 'h4': moved(answer['h4'], 2.5), 'f1': moved(answer['f1'], 4)}
    targets = [{'uid': u, 'set': 'T-A' if u == 'h4' else 'T-B', 'prompt': train[u]['messages'][:2], 'target': t}
               for u, t in good.items()]
    out = main.root / 'distill-018/sft-d-rows'
    result = ch.sft_data(sft_args(main, out, teacher(main.root / 'teacher', targets)))
    rows, summary = lines(out / 'placement.jsonl'), json.loads((out / 'summary.json').read_text())
    assert [r['uid'] for r in rows] == ['f1', 'h4', 'x1'] and all(r['chosen'] == r['rejected'] == good[r['uid']] for r in rows)
    assert all(r['prompt'] == train[r['uid']]['messages'][:2] and r['group'] == train[r['uid']]['group'] for r in rows)
    assert result == {'status': 'A018_SFT_ROWS_COMPLETE', 'rows': 3, 'by_set': {'T-A': 1, 'T-B': 2}, 'overlength': 0}
    assert summary['inputs']['teacher_manifest_sha256'] == h.sha256(main.root / 'teacher/MANIFEST.json')
    assert set(d._verified_dir(out)) == {'placement.jsonl', 'summary.json'}
    with pytest.raises(FileExistsError):
        ch.sft_data(sft_args(main, out, main.root / 'teacher'))
    tall = len(train['x1']['messages'][1]['content']) + len(good['x1']) + 1  # Prompt + target + EOS in Tok tokens.
    admitted = d._admitted(main.runs, h.load_pair(main.data)[1])
    with monkeypatch.context() as m:  # A lower token cap for train_dpo only (the SFT runs' own cap stays 8192).
        m.setattr(d, '_admitted', lambda *a: admitted)
        m.setattr(d, 'MAX_LEN', max(len(train[u]['messages'][1]['content']) + len(good[u]) + 1 for u in ('f1', 'h4')))
        assert tall > d.MAX_LEN
        assert ch.sft_data(sft_args(main, main.root / 'short', main.root / 'teacher'))['overlength'] == 1
        assert [r['uid'] for r in lines(main.root / 'short/placement.jsonl')] == ['f1', 'h4']
    refusals = [
        ([{**targets[0], 'target': answer['x1'].replace(',', ', ')}], 'not canonical'),  # Not P's training format.
        ([{**targets[1], 'target': answer['h4']}], 'not canonical, complete frozen-checker passes'),  # GT collides.
        ([{**targets[0], 'prompt': train['x1']['messages'][:1]}], "not P's SFT training prompt"),
        ([{**targets[0], 'uid': 'h0', 'prompt': train['h0']['messages'][:2], 'target': answer['h0']}], 'DPO1 positives'),
        ([{**targets[0], 'uid': 'h3', 'prompt': train['h3']['messages'][:2], 'target': answer['h3']}], 'length-admitted')]
    for k, (bad, match) in enumerate(refusals):
        with pytest.raises(ValueError, match=match):
            ch.sft_data(sft_args(main, main.root / f'bad{k}', teacher(main.root / f'teacher{k}', bad)))
    leak = OP.held(main.root / 'leak-dev', ['x1'])
    with pytest.raises(ValueError, match='share a UID or house group'):
        ch.sft_data(sft_args(main, main.root / 'leak', main.root / 'teacher', dev_dirs=[leak]))
    with monkeypatch.context() as m:  # train_dpo's tokenizer must reproduce the SFT run's prompt token counts.
        m.setattr(OP.Tok, 'apply_chat_template', lambda self, messages, **kw: messages[1]['content'] + ' ')
        with pytest.raises(ValueError, match='tokenize differently from SFT training'):
            ch.sft_data(sft_args(main, main.root / 'drift', main.root / 'teacher'))
    with monkeypatch.context() as m:
        m.setattr(d, 'exact_policy', lambda path, arm: ({'stage': 'dpo1', 'training': {'candidate': 'e1'}}, OP.POLICY))
        with pytest.raises(ValueError, match='dev-selected rpo-dpo1'):
            ch.sft_data(sft_args(main, main.root / 'e1', main.root / 'teacher'))
    assert not any((main.root / name).exists() for name in ('bad0', 'leak', 'drift', 'e1'))


# ---- DPO1-D pairs (T-A) ----------------------------------------------------------------------------

def spaced(request):
    value = json.loads(request['messages'][1]['content'])
    return OP.text(value, [[1 + 1.5 * j, 1, 0] for j in range(len(value['objects']))])


def sft_rows(root, data, uids, status='A018_SFT_ROWS_COMPLETE', chosen=spaced):
    """A stand-in RUN/distill-018/sft-d-rows: teacher targets for UIDS of the eval-layout DATA."""
    manifest = op.e._verified(data)
    requests = h._by_uid(h._rows(data, 'placement/requests.jsonl', manifest['files']['placement/requests.jsonl']))
    rows = [{**{k: requests[u][k] for k in ('uid', 'source', 'group')}, 'prompt': requests[u]['messages'],
             'chosen': chosen(requests[u]), 'rejected': chosen(requests[u])} for u in uids]
    d._publish_dir(root, {'placement.jsonl': d._jsonl(rows), 'summary.json': op._json_bytes({'status': status})})
    return root


def dpo1_args(w, out, targets, data=None):
    return ch.parse_args(['pairs-dpo1', '--data', str(data or w.data), '--samples', str(w.samples), '--base-model',
                          str(w.model), '--targets', str(targets), '--out', str(out)])


def test_dpo1_pairs_teacher_chosen_vs_rpo_dpo1_failures_in_tier_order(world, monkeypatch):
    targets = sft_rows(world.root / 'sft-d-rows', world.data, ['a', 'b', 'c', 'd'])
    out = world.root / 'dpo1-d-pairs'
    result = ch.pairs_dpo1(dpo1_args(world, out, targets))
    rows, summary = lines(out / 'placement.jsonl'), json.loads((out / 'summary.json').read_text())
    ledger = {x['uid']: x for x in lines(out / 'ledger.jsonl')}
    assert [(r['uid'], r['rejected_k'], r['rejected_tier']) for r in rows] == [
        ('a', 0, 'colocated_duplicate'), ('b', 0, 'main_collision'), ('c', 0, 'other'),
        ('d', 3, 'colocated_duplicate'), ('d', 2, 'main_collision')]  # Tier, then k; at most 2 per room.
    chosen = {r['uid']: r['chosen'] for r in lines(targets / 'placement.jsonl')}
    assert all(r['chosen'] == chosen[r['uid']] and r['chosen_from'] == 'teacher_target' for r in rows)
    assert all(set(r['violations']['chosen'].values()) == {0} and r['rejected'] != r['chosen'] for r in rows)
    assert [x.get('drop') for x in ledger['a']['samples']] == [None] + ['duplicate_text'] * 8  # Identical raws collapse.
    assert [x['class'] for x in ledger['c']['samples']][:3] == ['reject', 'incomplete', 'incomplete']  # Cap hits never.
    assert [(x['k'], x.get('drop')) for x in ledger['d']['samples'] if x['class'] == 'reject'] == [
        (1, 'room_cap'), (2, None), (3, None), (4, 'room_cap'), (5, 'room_cap')]
    assert result == {'status': 'A018_DPO1_PAIRS_COMPLETE', 'pairs': 5, 'rooms_with_pair': 4}
    assert summary['note'] == ch.OFF_POLICY and summary['by_tier'] == {'colocated_duplicate': 2, 'main_collision': 2, 'other': 1}
    assert summary['inputs']['reference_lineage_sha256'] == h.sha256(d.receipt_path(world.model.resolve()))
    assert summary['inputs']['targets_manifest_sha256'] == h.sha256(targets / 'MANIFEST.json')
    ch.pairs_dpo1(dpo1_args(world, world.root / 'two', sft_rows(world.root / 'only-b', world.data, ['b'])))
    assert [r['uid'] for r in lines(world.root / 'two/placement.jsonl')] == ['b']  # T-A rooms without a target: none.
    bad = sft_rows(world.root / 'bad-target', world.data, ['b'], chosen=lambda r: OP.script(
        json.loads(r['messages'][1]['content']), 'b', 0)[0])  # A colliding "teacher target".
    with pytest.raises(ValueError, match='teacher target fails the frozen checker'):
        ch.pairs_dpo1(dpo1_args(world, world.root / 'x1', bad))
    with pytest.raises(ValueError, match='not the SFT-D rows'):
        ch.pairs_dpo1(dpo1_args(world, world.root / 'x2', sft_rows(world.root / 'st', world.data, ['b'], status='X')))
    tc = OP.make_subset(world.root / 'tc', OP.SPEC, scope=ch.TC_SCOPE)
    with pytest.raises(ValueError, match='T-A'):
        ch.pairs_dpo1(dpo1_args(world, world.root / 'x3', targets, data=tc))
    with monkeypatch.context() as m:
        m.setattr(d, 'exact_policy', lambda path, arm: (RPO[0], {**OP.POLICY, 'adapter': '/elsewhere'}))
        with pytest.raises(ValueError, match="rpo-dpo1's exact policy"):
            ch.pairs_dpo1(dpo1_args(world, world.root / 'x4', targets))
    assert not any((world.root / f'x{k}').exists() for k in range(1, 5))


# ---- DPO2-D pairs (T-C, on-policy) -----------------------------------------------------------------

DPO1_D = ({'stage': 'dpo2', 'training': {'candidate': 'dpo1-d'}}, OP.POLICY)


def fake_plan(root, uids, monkeypatch):
    """A stand-in RUN/distill-018/plan holding the SPEC rooms, T-C = UIDS, as distill_018.load_plan returns it."""
    from fastfill.comparison import distill_018
    plan = OP.make_subset(root, OP.SPEC, scope=distill_018.SCOPE)
    manifest = json.loads((plan / 'MANIFEST.json').read_text())
    manifest.update({'split': 'train', 'main_data_manifest_sha256': manifest['training_manifest_sha256'],
                     'input_policy': 'system_user_only_labels_and_inventory_separate',
                     'selection': {'T-C': {'rule': 'stand-in'}}})
    h.write_json(plan / 'MANIFEST.json', manifest)
    (plan / 'uids').mkdir()
    h.write_json(plan / distill_018.UID_FILES['T-C'], {'uids': uids})
    monkeypatch.setattr(distill_018, 'load_plan', lambda path: {'root': Path(path), 'manifest': manifest,
                                                                 'sets': {'T-C': uids}})
    return plan


def test_tc_data_is_the_plans_t_c_rows_in_their_own_sampleable_data_dir(tmp_path, monkeypatch):
    plan = fake_plan(tmp_path / 'plan', ['c', 'a'], monkeypatch)
    out = tmp_path / 'distill-018/t-c'
    args = ch.parse_args(['tc-data', '--plan', str(plan), '--out', str(out)])
    assert ch.tc_data(args)['rooms'] == 2
    manifest, source = (json.loads((root / 'MANIFEST.json').read_text()) for root in (out, plan))
    assert (manifest['scope'], manifest['split'], manifest['expected_uids']) == (ch.TC_SCOPE, 'train', ['a', 'c'])
    assert manifest['sft_prompt_tokens'] == {u: source['sft_prompt_tokens'][u] for u in 'ac'}
    assert manifest['training_manifest_sha256'] == source['training_manifest_sha256']
    assert manifest['selection']['plan']['manifest_sha256'] == h.sha256(plan / 'MANIFEST.json')
    for name in ch.TC_FILES:  # The plan's rows verbatim.
        assert (out / name).read_text().splitlines() == [x for x in (plan / name).read_text().splitlines()
                                                         if json.loads(x)['uid'] in ('a', 'c')]
    assert [r['uid'] for r in infer.load_eval(out, 'placement')[0]] == ['a', 'c']  # sample_vllm's loader.
    with pytest.raises(FileExistsError):
        ch.tc_data(args)


@pytest.fixture
def tc(tmp_path, monkeypatch):
    """T-C (the SPEC rooms, derived by tc-data) sampled by a stand-in DPO1-D exact policy, as sample_vllm runs it."""
    data = tmp_path / 'distill-018/t-c'
    ch.tc_data(ns(plan=fake_plan(tmp_path / 'plan', sorted(OP.SPEC), monkeypatch), out=data))
    model = tmp_path / 'distill-018/placement/dpo1-d-merged'
    model.parent.mkdir(parents=True, exist_ok=True)
    d.receipt_path(model).write_text('{"stage": "dpo2"}\n')
    monkeypatch.setattr(d, 'inference_identity', lambda run: {
        'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': 'a' * 64}, 'stage': 'dpo2',
        'lineage_sha256': h.sha256(d.receipt_path(model.resolve())), 'policy': OP.POLICY})
    monkeypatch.setattr(d, 'exact_policy', lambda path, arm: DPO1_D)
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: OP.Tok())
    monkeypatch.setattr(op.rep, 'repair_row', OP.rounding_repair)
    monkeypatch.setattr(op, 'MIN_PAIRS', 6)
    monkeypatch.setattr(op, 'MIN_ROOMS', 3)
    return ns(root=tmp_path, data=data, model=model, samples=OP.sampled(data, model))


def dpo2_args(w, out, targets, t_a, data=None):
    return ch.parse_args(['pairs-dpo2', '--data', str(data or w.data), '--samples', str(w.samples), '--base-model',
                          str(w.model), '--targets', str(targets), '--t-a', str(t_a), '--out', str(out)])


def test_dpo2_pairs_run_pair_rule_on_t_c_with_the_dpo1_d_policy(tc, monkeypatch):
    targets, t_a = sft_rows(tc.root / 'sft-d-rows', tc.data, []), OP.held(tc.root / 't-a', ['zz'])
    out = tc.root / 'dpo2-d-pairs'
    result = ch.pairs_dpo2(dpo2_args(tc, out, targets, t_a))
    summary = json.loads((out / 'summary.json').read_text())
    assert result['status'] == summary['status'] == 'OP_PAIRS_COMPLETE' and summary['rule'] == op.PAIR_RULE
    assert summary['pairs'] == 6 and summary['rooms_with_pair'] == 3  # The amendment-013 rule on the same rooms.
    assert summary['inputs']['reference_model'] == str(tc.model.resolve())
    assert summary['inputs']['reference_lineage_sha256'] == h.sha256(d.receipt_path(tc.model.resolve()))
    with pytest.raises(ValueError, match='T-A or SFT-D training rooms'):
        ch.pairs_dpo2(dpo2_args(tc, tc.root / 'x1', sft_rows(tc.root / 'trained', tc.data, ['b']), t_a))
    with pytest.raises(ValueError, match='T-A or SFT-D training rooms'):
        ch.pairs_dpo2(dpo2_args(tc, tc.root / 'x2', targets, OP.held(tc.root / 't-a2', ['a'])))
    with pytest.raises(ValueError, match='T-C only'):
        ch.pairs_dpo2(dpo2_args(tc, tc.root / 'x3', targets, t_a, data=OP.make_subset(tc.root / 'ta', OP.SPEC)))
    with monkeypatch.context() as m:  # Sampled by rpo-dpo1 (or any policy but DPO1-D): refused.
        m.setattr(d, 'exact_policy', lambda path, arm: RPO)
        with pytest.raises(ValueError, match='exact policy'):
            ch.pairs_dpo2(dpo2_args(tc, tc.root / 'x4', targets, t_a))
    assert not any((tc.root / f'x{k}').exists() for k in range(1, 5))


def sbatch18(run_root, argv):
    """isambard_018.sbatch ARGV in an allocation with PROJECT_ROOT/runs/unknownsize-20261013 = RUN_ROOT."""
    project = run_root / 'project'
    if not project.exists():
        OP.python_stub(project / 'env/bin/python')
        (project / 'runs').mkdir()
        (project / 'runs/unknownsize-20261013').symlink_to(run_root)
        line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
        (project / 'isambard_018.sbatch').write_text((SCRIPTS / 'isambard_018.sbatch').read_text().replace(
            line, f'PROJECT_ROOT={project}\n'))
    return subprocess.run(['bash', str(project / 'isambard_018.sbatch'), str(CODE), *argv],
                          env={'PATH': '/usr/bin:/bin', 'SLURM_JOB_ID': '1'}, capture_output=True, text=True)


def test_dpo2_below_the_floors_is_recorded_and_its_train_and_dev_jobs_are_no_ops(tc, monkeypatch, capsys):
    targets, t_a = sft_rows(tc.root / 'sft-d-rows', tc.data, []), OP.held(tc.root / 't-a', ['zz'])
    out = tc.root / ch.RUN / ch.ROWS['dpo2-d']
    monkeypatch.setattr(op, 'MIN_PAIRS', 7)  # 6 pairs in 3 rooms: one pair short.
    assert not ch.not_trained(tc.root)
    assert ch.pairs_dpo2(dpo2_args(tc, out, targets, t_a)) == {'status': ch.NOT_TRAINED, 'pairs': 6, 'rooms_with_pair': 3}
    summary = json.loads((out / 'summary.json').read_text())
    assert (summary['status'], summary['pairs'], summary['limits']['min_pairs']) == (ch.NOT_TRAINED, 6, 7)
    assert sorted(p.name for p in out.iterdir()) == ['MANIFEST.json', 'ledger.jsonl', 'summary.json']  # No pairs file.
    assert ch.not_trained(tc.root)
    assert ch.train(ns(run_root=tc.root, candidate='dpo2-d')) == {'status': ch.NOT_TRAINED, 'candidate': 'dpo2-d'}
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        monkeypatch.setenv(key, '1')
    ch.main(['train-dpo2', '--run-root', str(tc.root)])
    assert json.loads(capsys.readouterr().out) == {'status': ch.NOT_TRAINED, 'candidate': 'dpo2-d'}
    for argv in (['shard', 'dev', 'dpo2-d', '0', '2'], ['shard', 'dev', 'dpo2-d', '1', '2', '--resume'],
                 ['dev-merge', 'dpo2-d', '2']):
        done = sbatch18(tc.root, argv)
        assert done.returncode == 0 and ch.NOT_TRAINED in done.stdout, done.stderr
    done = sbatch18(tc.root, ['shard', 'dev', 'dpo1-d', '0', '2'])  # Only DPO2-D is skipped.
    assert done.returncode == 1 and ch.NOT_TRAINED not in done.stdout


# ---- training chain --------------------------------------------------------------------------------

def publish_rows(root, name, rows, summary):
    d._publish_dir(root / ch.RUN / ch.ROWS[name], {'placement.jsonl': d._jsonl(rows), 'summary.json': op._json_bytes(summary)})


def test_train_chain_init_and_reference_wiring_recipes_and_lineage(pipeline, monkeypatch):
    run, calls, commands = pipeline.root.parent, [], []
    monkeypatch.setattr(d.subprocess, 'run', D.fake_train(calls))
    d.train(D.args(dpo_root=run / 'dpo-003', arm='placement', stage='dpo1', candidate='rpo'))
    inner = D.fake_train(calls)
    monkeypatch.setattr(ch.subprocess, 'run', lambda command, check: (commands.append(command), inner(command, check)))
    sft = (pipeline.root / 'placement/sft-merged').resolve()
    rpo = (run / 'dpo-003/placement/dpo1-merged').resolve()
    rows = D.pair_rows(pipeline.root, 'stage2-pairs', 'placement')
    with pytest.raises(FileNotFoundError):  # DPO1-D needs SFT-D first.
        ch.train(ns(run_root=run, candidate='dpo1-d'))
    publish_rows(run, 'sft-d', rows, {'status': 'A018_SFT_ROWS_COMPLETE'})
    receipts = {'sft-d': ch.train(ns(run_root=run, candidate='sft-d'))}
    sft_rows_sha = h.sha256(run / ch.RUN / 'sft-d-rows/MANIFEST.json')
    publish_rows(run, 'dpo1-d', rows, {'status': 'A018_DPO1_PAIRS_COMPLETE', 'inputs': {
        'reference_lineage_sha256': h.sha256(d.receipt_path(rpo)), 'targets_manifest_sha256': '0' * 64}})
    with pytest.raises(ValueError, match='not the dpo1-d training data'):  # Pairs whose chosen are not SFT-D's rows.
        ch.train(ns(run_root=run, candidate='dpo1-d'))
    shutil.rmtree(run / ch.RUN / 'dpo1-d-pairs')
    publish_rows(run, 'dpo1-d', rows, {'status': 'A018_DPO1_PAIRS_COMPLETE', 'inputs': {
        'reference_lineage_sha256': h.sha256(d.receipt_path(rpo)), 'targets_manifest_sha256': sft_rows_sha}})
    receipts['dpo1-d'] = ch.train(ns(run_root=run, candidate='dpo1-d'))
    publish_rows(run, 'dpo2-d', rows, {'status': 'OP_PAIRS_COMPLETE', 'inputs': {  # Sampled by rpo-dpo1, not DPO1-D.
        'reference_lineage_sha256': h.sha256(d.receipt_path(rpo))}})
    with pytest.raises(ValueError, match='not the dpo2-d training data'):
        ch.train(ns(run_root=run, candidate='dpo2-d'))
    shutil.rmtree(run / ch.RUN / 'dpo2-d-pairs')
    publish_rows(run, 'dpo2-d', rows + rows[:1], {'status': 'OP_PAIRS_COMPLETE', 'inputs': {
        'reference_lineage_sha256': h.sha256(d.receipt_path(ch.model_path(run, 'dpo1-d').resolve()))}})
    receipts['dpo2-d'] = ch.train(ns(run_root=run, candidate='dpo2-d'))
    flags = dict(zip(ch.CANDIDATES, calls[1:]))
    init = {'sft-d': run / 'dpo-003/placement/dpo1/final', 'dpo1-d': run / ch.RUN / 'placement/sft-d/final',
            'dpo2-d': run / ch.RUN / 'placement/dpo1-d/final'}
    for name, command in zip(ch.CANDIDATES, commands):
        f = flags[name]
        assert (Path(f['--model']), Path(f['--init_adapter']), Path(f['--out'])) == (
            sft, init[name].resolve(), run / ch.RUN / 'placement' / name)  # Init adapter = policy start = reference.
        assert Path(f['--pairs']) == (run / ch.RUN / ch.ROWS[name] / 'placement.jsonl').resolve()
        assert (f['--epochs'], f['--lr'], f['--beta'], f['--lora_r'], f['--lora_alpha'], f['--seed']) == (
            '2', '1e-05' if name == 'sft-d' else '5e-06', '0.1', '32', '16', '42')
        assert ('--sft_only' in command) == (name == 'sft-d') and ('--rpo_alpha' in command) == (name != 'sft-d')
        assert name == 'sft-d' or f['--rpo_alpha'] == '1.0'
        receipt = receipts[name]
        guard = h.strict_json((run / ch.RUN / 'placement' / name / 'training_guard.json').read_text())
        assert guard['settings'] == receipt['training']['recipe'] == {
            **ch.RECIPES[name], **d.TRAIN_FIXED, 'loss': ch.LOSS[name], 'init_adapter': {
                'path': str(init[name].resolve()), 'sha256': receipt['training']['init']['sha256'],
                'of': ch.INIT[name]}}
        assert receipt['stage'] == 'dpo2' and receipt['parent'] == {
            'stage': 'dpo1', 'path': str(rpo), 'lineage_sha256': h.sha256(d.receipt_path(rpo))}
        assert receipt['training_base']['path'] == str(sft) and receipt['training_base']['is_dpo1_merge'] is False
        assert receipt['training']['candidate'] == name and receipt['training']['chain'] == list(
            ch.CHAIN[:ch.CHAIN.index(name) + 1])
        assert ('note' in receipt['training']) == (name == 'dpo1-d')
        lineage, policy = d.exact_policy(ch.model_path(run, name), 'placement')  # Training base + this stage's adapter.
        assert policy['training_base'] == str(sft) and policy['adapter'] == str(
            (run / ch.RUN / 'placement' / name / 'final').resolve())
    assert receipts['dpo1-d']['training']['init']['lineage_sha256'] == h.sha256(
        d.receipt_path(ch.model_path(run, 'sft-d').resolve()))
    assert receipts['dpo2-d']['training']['pairs']['count'] == len(rows) + 1
    monkeypatch.setattr(d.h, 'base_receipt', lambda base: D.BASE)
    identity = d.inference_identity(ns(model=ch.model_path(run, 'dpo2-d'), arm='placement', base=Path('/models/Qwen3-8B'),
                                       exact_policy=True))  # What sample_vllm / infer_shard --exact-policy load.
    assert identity['stage'] == 'dpo2' and identity['policy']['adapter'].endswith('placement/dpo2-d/final')
    before = len(calls)
    assert ch.train(ns(run_root=run, candidate='dpo2-d'))['model_sha256'] == receipts['dpo2-d']['model_sha256']
    assert len(calls) == before  # A rerun only verifies the published merge.


# ---- dev score, select, merged check, test report --------------------------------------------------

def run18(w, candidate, kind, plan=None, out=None):
    merged = candidate.endswith('-merged')
    source = ['--model', str(ch.model_path(w.root, candidate.removesuffix('-merged')))] + ([] if merged else ['--exact-policy'])
    return OP.run_inference(w.root, w.data, plan or w.plan, out or w.root / ch.EVAL / candidate / 'infer-placement',
                            source, OP.identity(w.training, 'dpo2', merged), kind)


def dev_world(dev, monkeypatch, rpo='collide', kinds=('good', 'good', 'good')):
    root, training = dev.root, h.sha256(dev.data / 'MANIFEST.json')
    plan = DS.plan_for(dev.out, root / 'dev-eval-010/plan')
    for candidate, kind in (('sft', 'collide'), ('rpo-dpo1', rpo)):
        for arm in h.ARMS:
            DS.run_candidate(root, dev.out, plan, arm, candidate, kind, training)
    with monkeypatch.context() as m:
        m.setattr(ds, 'CANDIDATES', ('sft', 'rpo-dpo1'))
        ds.score(ns(data=dev.out, root=root / 'dev-eval-010', out=root / 'dev-eval-010/scores', workers=1,
                    only='selectable'))
    w = ns(root=root, data=dev.out, plan=plan, training=training, eval=root / ch.EVAL)
    for candidate, kind in zip(ch.CANDIDATES, kinds):
        run18(w, candidate, kind)
    return w


def score_args(w, out=None):
    return ns(data=w.data, root=w.eval, ref_root=w.root / 'dev-eval-010', out=out or w.eval / 'scores', workers=1)


def shard_preflight(w, argv):
    """isambard_018.sbatch shard SET CAND 0 2 in an allocation with PROJECT_ROOT/runs/unknownsize-20261013 = w.root."""
    project = w.root / 'project'
    if not project.exists():
        OP.python_stub(project / 'env/bin/python')
        (project / 'runs').mkdir()
        (project / 'runs/unknownsize-20261013').symlink_to(w.root)
        for name in ch.CANDIDATES:
            d.receipt_path(ch.model_path(w.root, name)).parent.mkdir(parents=True, exist_ok=True)
            d.receipt_path(ch.model_path(w.root, name)).write_text('{}')
        original = (SCRIPTS / 'isambard_018.sbatch').read_text()
        line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
        assert original.count(line) == 1
        (project / 'isambard_018.sbatch').write_text(original.replace(line, f'PROJECT_ROOT={project}\n'))
    return subprocess.run(['bash', str(project / 'isambard_018.sbatch'), str(CODE), 'shard', *argv, '0', '2'],
                          env={'PATH': '/usr/bin:/bin', 'SLURM_JOB_ID': '1'}, capture_output=True, text=True)


def test_dev_score_select_later_stage_tie_test_chain_merged_check_and_test_report(dev, monkeypatch):
    w = dev_world(dev, monkeypatch)
    result = ch.dev_score(score_args(w))
    scores = json.loads((w.eval / 'scores/scores.json').read_text())
    c, rooms = scores['candidates'], len(json.loads((w.data / 'MANIFEST.json').read_text())['expected_uids'])
    assert set(c) == {'rpo-dpo1', *ch.CANDIDATES} and scores['protocol'] == ch.SCORES_PROTOCOL
    assert scores['not_trained'] == {} and scores['note'] == ch.OFF_POLICY
    assert (c['rpo-dpo1']['geometry_success_rate'], c['rpo-dpo1']['accepted_rate'], c['rpo-dpo1']['complete_rate']) == (0, 0, 1)
    for name in ch.CANDIDATES:
        m = c[name]
        assert (m['geometry_success_rate'], m['accepted_rate'], m['no_required_rate'], m['complete_rate'], m['requests']) == (
            1, 1, 1, 1, rooms)
        assert m['halves'] == {'paper_sources': {'rooms': 3, 'geometry_success_rate': 1., 'a_prime': 1., 'accepted_rate': 1.},
                               'other_sources': {'rooms': 2, 'geometry_success_rate': 1., 'a_prime': 1., 'accepted_rate': 1.}}
        assert m['inference']['lora'] == ds.EXACT and m['model'] == str(ch.model_path(w.root, name).resolve())
        contrast = scores['contrasts_vs_rpo_dpo1'][name]
        assert contrast['geometry_success']['estimate'] == 1 and contrast['geometry_success']['ci95'] == [1, 1]
        assert set(contrast) == set(ch.FIELDS)
    assert result['dpo2-d'] == {'geometry_success_rate': 1.0, 'accepted_rate': 1.0, 'complete_rate': 1.0}
    geometry = lines(w.eval / 'scores/placement-sft-d-geometry.jsonl')
    assert len(geometry) == rooms and all(r['accepted'] and r['admitted_no_required'] for r in geometry)
    with pytest.raises(FileExistsError):
        ch.dev_score(score_args(w))
    DS.plan_for(T.make_data(w.root), w.root / 'infer-v2-002/plan')  # The test plan the adopted chain copies.
    log, sel = [], w.eval / 'selection.json'
    held = ch.select(ns(scores=w.eval / 'scores', out=sel, code_root=CODE, shards=2, no_submit=True),
                     run=OP.fake_sbatch(log), which=lambda name: name)
    assert held == {'status': 'ADOPTED', 'selected': 'dpo2-d'} and log == []  # Equal checker: the later stage.
    value = json.loads(sel.read_text())
    assert value['eligible'] == value['tied'] == list(ch.CANDIDATES) and value['rule'] == ch.RULE
    assert value['not_trained'] == [] and 'rule 文字沿用 013 写法' in (w.eval / 'selection-report.md').read_text()
    assert '决定：采纳 DPO2-D' in (w.eval / 'selection-report.md').read_text()
    picked = ch.select(ns(scores=w.eval / 'scores', out=sel, code_root=CODE, shards=2), run=OP.fake_sbatch(log),
                       which=lambda name: name)  # submit_018_chain.sh CODE test.
    assert picked['selected'] == 'dpo2-d'
    assert (w.root / 'infer-v2-007/plan/plan.json').read_bytes() == (w.root / 'infer-v2-002/plan/plan.json').read_bytes()
    S, run = str(SCRIPTS / 'isambard_018.sbatch'), str(w.root)
    test, devs = ['sbatch', '--parsable', '--hold', *ch.TEST_SHARD, S, str(CODE), 'shard'], [
        'sbatch', '--parsable', '--hold', *ch.DEV_SHARD, S, str(CODE), 'shard']
    assert log == [
        [*test, 'test', 'dpo2-d', '0', '2'], [*test, 'test', 'dpo2-d', '1', '2'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:101:102', str(SCRIPTS / 'isambard_infer_merge.sbatch'),
         str(CODE), 'placement', '2', f'{run}/infer-v2-007', f'{run}/infer-placement-007'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:103', S, str(CODE), 'test-geometry'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:103', '--time=06:00:00', S, str(CODE), 'test-report'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:104', *ch.S1_CPU, S, str(CODE), 's1-test'],
        [*devs, 'dev', 'dpo2-d-merged', '0', '2'], [*devs, 'dev', 'dpo2-d-merged', '1', '2'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:107:108', S, str(CODE), 'dev-merge', 'dpo2-d-merged', '2'],
        ['sbatch', '--parsable', *KILL, '--dependency=afterok:109', S, str(CODE), 'check-merged'],
        ['scontrol', 'release', '101,102,107,108']]
    with pytest.raises(FileExistsError, match='already submitted'):
        ch.select(ns(scores=w.eval / 'scores', out=sel, code_root=CODE, shards=2), run=OP.fake_sbatch([]), which=lambda n: n)
    run18(w, 'dpo2-d-merged', 'good')
    check = ch.check_merged(ns(data=w.data, root=w.eval, ref_root=w.root / 'dev-eval-010', out=w.eval / 'merged-check',
                               workers=1))
    assert check['decision'] == 'merged_convenience_artifact'
    assert json.loads((w.eval / 'merged-check/check.json').read_text())['merged']['inference']['lora'] == ds.PREMERGED
    for argv, ok in ((['test', 'dpo2-d'], True), (['test', 'sft-d'], False), (['dev', 'dpo2-d-merged'], True),
                     (['dev', 'dpo1-d-merged'], False)):
        done = shard_preflight(w, argv)
        assert ('PREFLIGHT FAIL' not in done.stderr and 'module' in done.stderr) if ok else (
            f"PREFLIGHT FAIL: {argv[1].removesuffix('-merged')} is not the adopted 018 candidate" in done.stderr), done.stderr
    plan = DS.plan_for(w.data, w.root / 'test-plan')  # The dev rooms stand in for eval-data-v1 here.
    run18(w, 'dpo2-d', 'good', plan, w.root / 'infer-placement-007')
    OP.run_inference(w.root, w.data, plan, w.root / 'infer-placement-005',
                     ['--model', str(w.root / 'dpo-003/placement/dpo1-merged'), '--exact-policy'],
                     OP.identity(w.training, 'dpo1'), 'collide')
    report = ch.test_report(ns(data=w.data, selection=sel, p18=w.root / 'infer-placement-007',
                               ref=w.root / 'infer-placement-005', out=w.root / 'test-report-018', workers=1))
    value = json.loads((w.root / 'test-report-018/report.json').read_text())
    assert report == {'status': 'A018_TEST_REPORT_COMPLETE', 'dpo2-d': 1.0, 'rpo-dpo1': 0.0}
    assert value['contrast_selected_minus_rpo_dpo1']['geometry_success']['estimate'] == 1 and value['rooms'] == rooms
    assert value['note'] == ch.OFF_POLICY
    with pytest.raises(ValueError, match='inference is not placement/dpo2-d'):  # Swapped runs are refused.
        ch.test_report(ns(data=w.data, selection=sel, p18=w.root / 'infer-placement-005',
                          ref=w.root / 'infer-placement-007', out=w.root / 'test-report-x', workers=1))


def test_no_adoption_keeps_rpo_dpo1_and_submits_nothing(dev, monkeypatch):
    w = dev_world(dev, monkeypatch, rpo='good', kinds=('collide', 'broken', 'good'))
    ch.dev_score(score_args(w))
    c = json.loads((w.eval / 'scores/scores.json').read_text())['candidates']
    assert c['dpo1-d']['complete_rate'] == c['dpo1-d']['geometry_success_rate'] == c['dpo1-d']['no_required_rate'] == 0
    sel = w.eval / 'selection.json'

    def never(*a, **k):
        raise AssertionError('nothing may be submitted')
    args = ns(scores=w.eval / 'scores', out=sel, code_root=CODE, shards=2)
    result = ch.select(args, run=never, which=lambda name: name)
    value = json.loads(sel.read_text())
    assert result['status'] == 'NOT_ADOPTED' and (value['selected'], value['adopted'], value['test']) == ('rpo-dpo1', False, None)
    assert value['checks']['dpo2-d'] == {'complete': True, 'checker_gain': False, 'strict_admission': True,
                                         'both_halves': True}  # Equal to R: no +0.02 gain, CI [0, 0].
    assert ch.select(args, run=never, which=lambda name: name)['status'] == 'NOT_ADOPTED'  # Idempotent, write-once.
    assert 'P 保持 rpo-dpo1' in (w.eval / 'selection-report.md').read_text() and not (w.root / 'infer-v2-007').exists()
    with pytest.raises(ValueError, match='nothing runs on test'):
        ch.check_merged(ns(data=w.data, root=w.eval, ref_root=w.root / 'dev-eval-010', out=w.eval / 'mc', workers=1))
    sel.unlink()
    sel.write_text(json.dumps({**value, 'selected': 'dpo2-d', 'adopted': True}) + '\n')  # Hand-edited: authorizes nothing.
    with pytest.raises(ValueError, match='exists with another selection'):
        ch.select(args, run=never, which=lambda name: name)
    with pytest.raises(ValueError, match='not bound to its dev scores'):
        ch.load_selection(sel)
    done = shard_preflight(w, ['dev', 'dpo2-d-merged'])
    assert done.returncode == 1 and 'not bound to its dev scores' in done.stderr


def test_untrained_dpo2_d_is_dropped_only_on_its_record(dev, monkeypatch):
    w = dev_world(dev, monkeypatch, kinds=('collide', 'good'))  # No DPO2-D dev run.
    with pytest.raises(FileNotFoundError):  # Without the NOT_TRAINED record every candidate must be scored.
        ch.dev_score(score_args(w))
    record = w.root / ch.RUN / ch.ROWS['dpo2-d']
    d._publish_dir(record, {'ledger.jsonl': d._jsonl([]), 'summary.json': op._json_bytes({'status': ch.NOT_TRAINED})})
    ch.dev_score(score_args(w))
    scores = json.loads((w.eval / 'scores/scores.json').read_text())
    assert set(scores['candidates']) == {'rpo-dpo1', 'sft-d', 'dpo1-d'} and set(scores['contrasts_vs_rpo_dpo1']) == {
        'sft-d', 'dpo1-d'}
    assert scores['not_trained'] == {'dpo2-d': h.sha256(record / 'MANIFEST.json')}
    sel = w.eval / 'selection.json'
    assert ch.select(ns(scores=w.eval / 'scores', out=sel, code_root=CODE, shards=2, no_submit=True),
                     run=OP.fake_sbatch([]), which=lambda name: name) == {'status': 'ADOPTED', 'selected': 'dpo1-d'}
    value, _ = ch.load_selection(sel)
    assert value['not_trained'] == ['dpo2-d'] and set(value['checks']) == {'sft-d', 'dpo1-d'}
    report = (w.eval / 'selection-report.md').read_text()
    assert 'DPO2-D 未训练' in report and '| DPO2-D |' not in report


def metric(complete, checker, admission, paper=None, other=None):
    return {'complete_rate': complete, 'geometry_success_rate': checker, 'accepted_rate': admission,
            'halves': {'paper_sources': {'geometry_success_rate': checker if paper is None else paper},
                       'other_sources': {'geometry_success_rate': checker if other is None else other}}}


def ci(lower):
    return {'geometry_success': {'ci95': None if lower is None else [lower, lower + .1]}}


def test_rule_bounds_ci_halves_and_later_stage_tie_break():
    ref = metric(.90, .50, .40)
    edge = {'rpo-dpo1': ref, 'sft-d': metric(.895, .52, .39), 'dpo1-d': metric(.9, .5, .4), 'dpo2-d': metric(.9, .5, .4)}
    cis = {'sft-d': ci(.001), 'dpo1-d': ci(.001), 'dpo2-d': ci(.001)}
    assert {k: v for k, v in ch.choose(edge, cis).items() if k != 'checks'} == {
        'selected': 'sft-d', 'adopted': True, 'eligible': ['sft-d'], 'tied': ['sft-d'], 'best_checker': .52,
        'not_trained': []}
    for bad in (metric(.894, .6, .4), metric(.9, .519, .4), metric(.9, .6, .389), metric(.9, .6, .4, paper=.49)):
        assert not ch.choose({**edge, 'sft-d': bad}, cis)['adopted']  # Each bound on its own (all inclusive).
    assert ch.choose({**edge, 'sft-d': metric(.9, .6, .4, other=.5)}, cis)['adopted']  # Equal in a half is allowed.
    for lower in (0, None, -.01):  # The CI lower bound must be strictly above 0 (undefined CI: not eligible).
        assert not ch.choose(edge, {**cis, 'sft-d': ci(lower)})['adopted']
    near = {'rpo-dpo1': ref, 'sft-d': metric(.9, .610, .4), 'dpo1-d': metric(.9, .606, .4), 'dpo2-d': metric(.9, .5, .4)}
    choice = ch.choose(near, cis)
    assert (choice['selected'], choice['tied']) == ('dpo1-d', ['sft-d', 'dpo1-d'])  # Within 0.005: the later stage.
    assert ch.choose({**near, 'sft-d': metric(.9, .612, .4)}, cis)['selected'] == 'sft-d'
    assert ch.choose({**near, 'dpo2-d': metric(.9, .7, .4)}, cis)['selected'] == 'dpo2-d'
    with pytest.raises(ValueError, match='must be scored'):
        ch.choose({'rpo-dpo1': ref, 'sft-d': ref}, cis)
    trained = {k: v for k, v in near.items() if k != 'dpo2-d'}  # DPO2-D below the floors: not a candidate.
    choice = ch.choose(trained, cis, ['dpo2-d'])
    assert (choice['selected'], choice['not_trained'], set(choice['checks'])) == ('dpo1-d', ['dpo2-d'], {'sft-d', 'dpo1-d'})
    for untrained, candidates in ((['dpo2-d'], near), (['dpo1-d'], trained), ([], trained)):
        with pytest.raises(ValueError, match='must be scored'):  # Only DPO2-D, only when absent, only when recorded.
            ch.choose(candidates, cis, untrained)


# ---- Slurm scripts ---------------------------------------------------------------------------------

INPUTS = ('main-data-v1/MANIFEST.json', 'sft-joint-001/SHA256.json', 'sft-placement-001/SHA256.json',
          'dpo-001/positives/MANIFEST.json', 'dev-select-v1/MANIFEST.json', 'dev-full-v1/MANIFEST.json',
          'eval-data-v1/MANIFEST.json', 'dpo-003/placement/dpo1-merged.json', 'dpo-001/placement/sft-merged.json',
          'dpo-003/placement/dpo1/final/adapter_model.safetensors', 'dpo-004/train-subset/MANIFEST.json',
          'dpo-004/samples/MANIFEST.json', 'dev-eval-010/plan/MANIFEST.json', 'dev-eval-010/scores/MANIFEST.json',
          'dev-eval-010/sft/infer-placement/MANIFEST.json', 'dev-eval-010/rpo-dpo1/infer-placement/MANIFEST.json',
          'context-capacity-001/SHA256.json', 'infer-v2-002/plan/MANIFEST.json', 'infer-joint-002/MANIFEST.json',
          'infer-placement-005/MANIFEST.json', 'distill-018/teacher/MANIFEST.json', 'distill-018/teacher/targets.jsonl',
          'distill-018/plan/MANIFEST.json')
DOC = 'docs/research/20261013-unknown-size-comparison/amendment-018-distill-from-open-teacher-20261014.md'


def stubs(directory):
    """squeue (own jobs: SQUEUE_OUT; -j ID: SQUEUE_JOB) and sacct (SACCT_OUT) next to the sbatch / scontrol stubs."""
    (directory / 'squeue').write_text('#!/bin/sh\ncase "$*" in\n'
                                      '  "-h -u me -n unknownsize-018 -o %i") printf "%s" "$SQUEUE_OUT" ;;\n'
                                      '  "-h -j "*" -o %i") printf "%s" "$SQUEUE_JOB" ;;\n  *) exit 3 ;;\nesac\n')
    (directory / 'sacct').write_text('#!/bin/sh\ntest "$*" = "-n -X -P -u me -S 2026-10-14 -o JobName,ElapsedRaw,AllocTRES"'
                                     ' || exit 3\nprintf "%s" "$SACCT_OUT"\n')
    for name in ('squeue', 'sacct'):
        (directory / name).chmod(0o755)
    return {**T.stub_sbatch(directory), 'USER': 'me'}


def submit_script(tmp_path):
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    for path in INPUTS:
        (run / path).parent.mkdir(parents=True, exist_ok=True)
        (run / path).write_text('{}')
    (project / 'models/Qwen3-8B').mkdir(parents=True)
    OP.python_stub(project / 'env/bin/python')
    script = tmp_path / 'submit_018_chain.sh'
    original = (SCRIPTS / 'submit_018_chain.sh').read_text()
    line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    assert original.count(line) == 1
    script.write_text(original.replace(line, f'PROJECT_ROOT={project}\n'))
    code = tmp_path / 'main-abc'
    code.mkdir()
    (code / 'fastfill').symlink_to(CODE / 'fastfill')
    (code / DOC).parent.mkdir(parents=True)
    (code / DOC).write_text('# Amendment 018\n')
    return script, run, code


def test_submit_018_chain_dependencies_restart_gpu_cap_and_refusals(tmp_path):
    for name in ('isambard_018.sbatch', 'submit_018_chain.sh'):
        subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    assert '\n#SBATCH --exclude=nid010216\n' in (SCRIPTS / 'isambard_018.sbatch').read_text()  # Every 018 job.
    env = stubs(tmp_path)
    script, run, code = submit_script(tmp_path)
    sent = []

    def submit(*argv, **extra):
        before = len(sent)
        done = subprocess.run(['bash', str(script), str(code), *argv], env={**env, **extra}, capture_output=True, text=True)
        log = tmp_path / 'log'
        sent[:] = log.read_text().splitlines() if log.exists() else []
        return done, [x.replace(f'{code}/fastfill/comparison/', '').replace(f' {code} ', ' CODE ') for x in sent[before:]]
    kill, cpu, train = '--parsable --kill-on-invalid-dep=yes --dependency=afterok:', '--cpus-per-task=72 --mem=192G', \
        '--gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=06:00:00'
    shard, J = '--gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=04:00:00', 'isambard_018.sbatch CODE'
    done, log = submit(SACCT_OUT='unknownsize-018-teacher|36000|billing=72,gres/gpu=4\nunknownsize-016|360000|gres/gpu=4\n')
    assert done.returncode == 0, done.stderr
    assert 'GPU-h: used 40.00 + ceiling 104 <= 190' in done.stdout  # Only unknownsize-018* jobs count.
    assert log == [
        f'--parsable --hold {J} recheck',                                                            # 101
        f'{kill}101 {cpu} --time=04:00:00 {J} sft-data',                                              # 102
        f'{kill}102 {train} {J} train sft-d',                                                         # 103
        f'{kill}102 {cpu} --time=04:00:00 {J} pairs-dpo1',                                            # 104
        f'{kill}103:104 {train} {J} train dpo1-d',                                                    # 105
        f'{kill}101 {J} tc-data',                                                                     # 106
        f'{kill}105:106 --gpus-per-node=1 --cpus-per-task=16 --mem=100G --time=02:00:00 --array=0-3 {J} sample 4',
        f'{kill}107 {J} finalize',                                                                    # 108
        f'{kill}108 {cpu} --time=08:00:00 {J} pairs-dpo2',                                            # 109
        f'{kill}109 {train} {J} train dpo2-d',                                                        # 110
        f'{kill}103 {shard} {J} shard dev sft-d 0 2', f'{kill}103 {shard} {J} shard dev sft-d 1 2',
        f'{kill}111:112 {J} dev-merge sft-d 2',                                                       # 113
        f'{kill}105 {shard} {J} shard dev dpo1-d 0 2', f'{kill}105 {shard} {J} shard dev dpo1-d 1 2',
        f'{kill}114:115 {J} dev-merge dpo1-d 2',                                                      # 116
        f'{kill}110 {shard} {J} shard dev dpo2-d 0 2', f'{kill}110 {shard} {J} shard dev dpo2-d 1 2',
        f'{kill}117:118 {J} dev-merge dpo2-d 2',                                                      # 119
        f'{kill}113:116:119 {J} dev-score', f'{kill}120 {J} select']
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101\n'
    done, log = submit(SACCT_OUT='unknownsize-018-teacher|81000|gres/gpu=4\n')  # 90 + 104 > 190.
    assert log == [] and done.returncode == 1 and 'GPU cap: used 90.00 + ceiling 104 > 190' in done.stderr
    # The teacher filter still running: sft-data waits on it, and its targets need not exist yet.
    for path in ('distill-018/teacher/MANIFEST.json', 'distill-018/teacher/targets.jsonl'):
        (run / path).unlink()
    done, log = submit('sft-data', '7400555', SQUEUE_JOB='7400555')
    assert done.returncode == 0, done.stderr
    assert log[1] == f'{kill}122:7400555 {cpu} --time=04:00:00 {J} sft-data' and len(log) == 21
    done, log = submit()
    assert log == [] and done.returncode == 1 and 'teacher/MANIFEST.json and no TEACHER_JOB' in done.stderr
    done, log = submit('sft-data', '7400555')
    assert log == [] and done.returncode == 1 and 'TEACHER_JOB 7400555 is not in Slurm' in done.stderr
    assert submit('train-sft', '7400555')[0].returncode == 2  # Only sft-data reads the teacher targets.
    (run / 'distill-018/t-c').mkdir()
    (run / 'distill-018/t-c/MANIFEST.json').write_text('{}')  # T-C derived: no second tc-data.
    # A restart from train-dpo1: earlier outputs exist, SFT-D's dev run is merged, a DPO1-D dev shard resumes.
    for path in ('distill-018/sft-d-rows', 'distill-018/dpo1-d-pairs', 'dev-eval-018/sft-d/infer-placement',
                 'dev-eval-018/dpo1-d/placement/shard-01', 'distill-018/placement'):
        (run / path).mkdir(parents=True)
    (run / 'distill-018/placement/sft-d-merged.json').write_text('{}')
    (run / 'dev-eval-018/sft-d/infer-placement/MANIFEST.json').write_text('{}')
    done, log = submit('train-dpo1')
    assert done.returncode == 0, done.stderr
    assert 'GPU-h: used 0.00 + ceiling 72 <= 190' in done.stdout and 'skipped dev sft-d: merged' in done.stdout
    assert log == [  # Ids continue at 143: the held recheck root stands in for every skipped dependency.
        f'--parsable --hold {J} recheck', f'{kill}143 {train} {J} train dpo1-d',                     # 143, 144
        f'{kill}144 --gpus-per-node=1 --cpus-per-task=16 --mem=100G --time=02:00:00 --array=0-3 {J} sample 4',
        f'{kill}145 {J} finalize', f'{kill}146 {cpu} --time=08:00:00 {J} pairs-dpo2',                # 146, 147
        f'{kill}147 {train} {J} train dpo2-d',                                                        # 148
        f'{kill}144 {shard} {J} shard dev dpo1-d 0 2', f'{kill}144 {shard} {J} shard dev dpo1-d 1 2 --resume',
        f'{kill}149:150 {J} dev-merge dpo1-d 2',                                                      # 151
        f'{kill}148 {shard} {J} shard dev dpo2-d 0 2', f'{kill}148 {shard} {J} shard dev dpo2-d 1 2',
        f'{kill}152:153 {J} dev-merge dpo2-d 2',                                                      # 154
        f'{kill}151:154 {J} dev-score', f'{kill}155 {J} select']
    (run / 'distill-018/t-c/MANIFEST.json').unlink()  # A T-C derivation that never finished: refused, not reused.
    done, log = submit('train-dpo1')
    assert log == [] and 'refusing to reuse existing ' + str(run / 'distill-018/t-c') in done.stderr
    (run / 'distill-018/t-c').rmdir()
    (run / 'distill-018/plan/MANIFEST.json').unlink()
    done, log = submit('train-dpo1')
    assert log == [] and 'no ' + str(run / 'distill-018/plan/MANIFEST.json') + ' to derive T-C' in done.stderr
    (run / 'distill-018/t-c').mkdir()
    (run / 'distill-018/t-c/MANIFEST.json').write_text('{}')
    refusals = [
        (('pairs-dpo2',), {}, f"FROM pairs-dpo2 needs {run / 'distill-018/placement/dpo1-d-merged.json'}"),
        (('bogus',), {}, None),
        ((), {}, 'refusing to reuse existing ' + str(run / 'distill-018/sft-d-rows')),
        (('train-dpo1',), {'SQUEUE_OUT': '7400001\n'}, 'scancel 7400001'),
        (('dev-score',), {}, f"FROM dev-score needs {run / 'distill-018/placement/dpo1-d-merged.json'}")]
    for argv, extra, message in refusals:
        done, log = submit(*argv, **extra)
        assert log == [] and (done.returncode, message is None or message in done.stderr) == (
            2 if message is None else 1, True), done.stderr
    (run / 'dev-eval-018/dpo1-d/infer-placement').mkdir()  # A dev merge that never finished: refused, not reused.
    done, log = submit('train-dpo1')
    assert log == [] and 'refusing to reuse existing ' + str(run / 'dev-eval-018/dpo1-d/infer-placement') in done.stderr
    for path in ('placement/dpo1-d-merged.json', 'dpo2-d-samples/MANIFEST.json', 'dpo2-d-pairs', 'placement/dpo2-d-merged.json'):
        (run / 'distill-018' / path).parent.mkdir(parents=True, exist_ok=True)
        (run / 'distill-018' / path).write_text('{}')
    done, log = submit('dev-score')  # Every step before FROM needs its output, the dev merges included.
    assert log == [] and f"FROM dev-score needs {run / 'dev-eval-018/dpo1-d/infer-placement/MANIFEST.json'}" in done.stderr
    # pairs-dpo2 below the floors: a restart needs neither DPO2-D's merge nor its dev run and submits nothing for it.
    (run / 'distill-018/dpo2-d-pairs').unlink()
    (run / 'distill-018/placement/dpo2-d-merged.json').unlink()
    (run / 'dev-eval-018/dpo1-d/infer-placement').rmdir()
    d._publish_dir(run / 'distill-018/dpo2-d-pairs', {'ledger.jsonl': d._jsonl([]),
                                                     'summary.json': op._json_bytes({'status': ch.NOT_TRAINED})})
    for start in ('train-dpo2', 'dev'):
        k = 101 + len(sent)
        done, log = submit(start)
        assert done.returncode == 0 and 'dpo2-d not trained' in done.stdout and 'ceiling 8 <=' in done.stdout, done.stderr
        assert log == [f'--parsable --hold {J} recheck', f'{kill}{k} {shard} {J} shard dev dpo1-d 0 2',
                       f'{kill}{k} {shard} {J} shard dev dpo1-d 1 2 --resume', f'{kill}{k + 1}:{k + 2} {J} dev-merge dpo1-d 2',
                       f'{kill}{k + 3} {J} dev-score', f'{kill}{k + 4} {J} select']
    # test: select again on the login node (it submits the adopted test chain itself), within the GPU cap.
    done, log = submit('test')
    assert log == [] and done.returncode == 1 and 'no ' + str(run / 'dev-eval-018/selection.json') in done.stderr
    (run / 'dev-eval-018/selection.json').write_text('{}')
    done, log = submit('test', SACCT_OUT='unknownsize-018|432000|gres/gpu=1\n')  # 120 + 80 > 190.
    assert log == [] and done.returncode == 1 and 'GPU cap: used 120.00 + ceiling 80 > 190' in done.stderr
    done, log = submit('test', PYLOG=str(tmp_path / 'python.log'))
    eval_root = run / 'dev-eval-018'
    assert done.returncode == 0 and log == [] and (tmp_path / 'python.log').read_text().split() == [
        '-B', '-m', 'fastfill.comparison.distill_018_chain', 'select', '--scores', str(eval_root / 'scores'),
        '--out', str(eval_root / 'selection.json'), '--code-root', str(code), '--shards', '6']
    assert submit('test', '3')[0].returncode == 2


@pytest.mark.parametrize('argv, ok', [
    (['recheck'], True), (['tc-data'], True), (['tc-data', 'x'], False), (['sft-data'], True), (['sft-data', 'x'], False), (['train', 'sft-d'], True),
    (['train', 'dpo2-d'], True), (['train', 'rpo-dpo1'], False), (['train'], False), (['pairs-dpo1'], True),
    (['sample', '1'], True), (['sample', '4'], False), (['finalize'], True), (['pairs-dpo2'], True),
    (['shard', 'dev', 'sft-d', '0', '2'], True), (['shard', 'test', 'dpo1-d', '5', '6', '--resume'], True),
    (['shard', 'test', 'dpo1-d-merged', '0', '2'], False), (['shard', 'dev', 'dpo2-d-merged', '1', '2'], True),
    (['shard', 'dev', 'rpo-dpo1', '0', '2'], False), (['shard', 'dev', 'sft-d', '2', '2'], False),
    (['dev-merge', 'dpo2-d-merged', '2'], True), (['dev-merge', 'sft', '2'], False), (['dev-score'], True),
    (['select'], True), (['test-geometry'], True), (['test-report'], True), (['s1-test'], True),
    (['check-merged'], True), (['bogus'], False)])
def test_sbatch_modes_validate_arguments_before_the_allocation(tmp_path, argv, ok):
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    done = subprocess.run(['bash', str(SCRIPTS / 'isambard_018.sbatch'), str(CODE), *argv], env=env, capture_output=True,
                          text=True)
    assert (done.returncode, 'Slurm allocation required' in done.stderr) == ((1, True) if ok else (2, False))
