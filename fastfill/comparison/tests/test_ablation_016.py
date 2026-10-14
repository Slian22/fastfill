"""Amendment 016 section 3 DPO2 attribution ablation on CPU: fakes stand in for train_dpo, merge_lora, the models and
Slurm (fixtures of the amendment-013 tests)."""
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import types

import pytest

from fastfill.comparison import ablation_016 as ab
from fastfill.comparison import dev_select as ds
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison import visual_pipeline as vp
from fastfill.comparison.tests import test_dpo_stages as D
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests import test_onpolicy_dpo as OP
from fastfill.comparison.tests.test_dev_select import dev  # noqa: F401  Fixture: 5 frozen dev rooms.
from fastfill.comparison.tests.test_dpo_stages import pipeline  # noqa: F401  Fixture: dpo-001 on CPU.
from fastfill.comparison.tests.test_visual_pipeline import agent_config

SCRIPTS = OP.SCRIPTS
COMMITTED = '8a832aa95'  # The committed seed-42 ablation (rft, pert) whose training identity must not change.


def ns(**kw):
    return types.SimpleNamespace(**kw)


def test_take_dedups_pert_pairs_ranks_by_hash_and_cycles_to_the_epoch_rows():
    pairs = [{'prompt': [{'role': 'user', 'content': f'p{k % 4}'}], 'chosen': f'c{k % 4}', 'rejected': f'r{k % 4}',
              'n': k} for k in range(6)]  # Four (prompt, chosen, rejected) pairs; two of them twice.
    rows, info = ab.take(pairs, 'pert', 10)
    assert info == {'source_rows': 6, 'unique': 4, 'rows_per_epoch': 10, 'full_cycles': 2, 'extra_rows': 2}
    assert rows[:4] == rows[4:8] and rows[8:] == rows[:2] and {r['n'] for r in rows} == {0, 1, 2, 3}  # First kept.
    assert ab.take([{**p, 'rejected': f'x{p["n"]}'} for p in pairs], 'pert', 6)[1]['unique'] == 6  # Another rejected.
    shuffled = random.Random(0).sample(pairs[:4], 4)
    assert ab.take(shuffled, 'pert', 10)[0] == rows  # Hash order, not file order.
    subset, info = ab.take(pairs, 'pert', 3)
    assert subset == rows[:3] and info['full_cycles'] == 0 and info['extra_rows'] == 3  # A hash-ranked subset.
    with pytest.raises(ValueError, match='nothing to train on'):
        ab.take([], 'pert', 4)


def test_train_rft_and_pert_from_rpo_dpo1_with_op_dpo2_steps_and_lineage(pipeline, monkeypatch):
    source, calls, commands = pipeline.root, [], []
    run_root = source.parent
    monkeypatch.setattr(d.subprocess, 'run', D.fake_train(calls))
    d.train(D.args(dpo_root=run_root / 'dpo-003', arm='placement', stage='dpo1', candidate='rpo'))
    root, parent, count = OP.op_pairs(pipeline, extra=5)  # 10 op pairs (5 unique): 3 steps per epoch, S = 6.
    op_rows = (root / 'stage2-pairs/placement.jsonl').read_bytes()
    inner = OP.fake_train_epochs(calls)
    with pytest.raises(ValueError, match='after amendment-013 training'):  # Nothing to match the steps to yet.
        ab.train(ns(dpo_root=run_root / 'dpo-005', candidate='rft'))
    assert not (run_root / 'dpo-005').exists()
    monkeypatch.setattr(op.subprocess, 'run', inner)
    op.train(ns(dpo_root=root))
    monkeypatch.setattr(ab.subprocess, 'run', lambda command, check: (commands.append(command), inner(command, check)))
    matched = ab.matched_steps(run_root)
    assert (matched['steps'], matched['epochs'], matched['pairs_count'], count) == (6, 2, 10, 10)
    sft, init = (source / 'placement/sft-merged').resolve(), (run_root / 'dpo-003/placement/dpo1/final').resolve()
    _, policy = d.exact_policy(parent, 'placement')
    rft_info = {'source_rows': 10, 'unique': 5, 'rows_per_epoch': 10, 'verbatim': True}  # op-dpo2's 10 rows as is.
    pert_info = {'source_rows': 5, 'unique': 5, 'rows_per_epoch': 12, 'full_cycles': 2, 'extra_rows': 2}  # 4 x S/2.
    for name, run, info in (('rft', 'dpo-005', rft_info), ('pert', 'dpo-006', pert_info)):
        receipt = ab.train(ns(dpo_root=run_root / run, candidate=name))
        flags, command = calls[-1], commands[-1]
        assert (Path(flags['--model']), Path(flags['--init_adapter']), Path(flags['--out'])) == (
            sft, init, run_root / run / 'placement/dpo2')  # rpo-dpo1's exact policy, as op-dpo2.
        assert (flags['--epochs'], flags['--lr'], flags['--lora_r'], flags['--lora_alpha'], flags['--seed']) == (
            '2', '5e-06', '32', '16', '42')
        if name == 'rft':
            assert command[-2:] == ['--save_every_epoch', '--sft_only'] and '--rpo_alpha' not in command
        else:
            assert command[-1] == '--save_every_epoch' and flags['--rpo_alpha'] == '1.0' and '--sft_only' not in command
        rows = OP.lines(run_root / run / 'train-rows/placement.jsonl')
        t = receipt['training']
        assert len(rows) == t['pairs']['count'] == info['rows_per_epoch'] and t['completion']['steps'] == 6  # = S.
        assert len({json.dumps([r['prompt'], r['chosen']]) for r in rows}) == 5
        assert t['steps_matched_to'] == matched and t['candidate'] == name
        assert {k: v for k, v in t['rows'].items() if k != 'source'} == info
        assert t['recipe']['loss'] == ab.LOSS[name] and t['recipe']['init_adapter'] == {
            'path': str(init), 'sha256': policy['adapter_sha256'], 'of': 'rpo-dpo1'}
        assert receipt['training_base'] == {'stage': 'sft', 'path': str(sft), 'lineage_sha256': h.sha256(d.receipt_path(sft)),
                                            'model_sha256': d.verify_merged(sft, 'placement', 'sft')['model_sha256'],
                                            'is_dpo1_merge': False}
        assert receipt['parent'] == {'stage': 'dpo1', 'path': str(parent), 'lineage_sha256': h.sha256(d.receipt_path(parent))}
        _, exact = d.exact_policy(ab.model_path(run_root, name), 'placement')  # What infer_shard --exact-policy loads.
        assert exact['training_base'] == str(sft) and exact['adapter'] == str((run_root / run / 'placement/dpo2/final').resolve())
    assert (run_root / 'dpo-005/train-rows/placement.jsonl').read_bytes() == op_rows  # Only the DPO term differs.
    pert_rows = OP.lines(run_root / 'dpo-006/train-rows/placement.jsonl')
    assert {r['reject_kind'] for r in pert_rows} <= {'collision_shift', 'wall_oob'}
    assert {r['chosen'] for r in pert_rows} == {r['chosen'] for r in D.pair_rows(source, 'stage2-pairs', 'placement')}
    before = len(calls)
    assert ab.train(ns(dpo_root=run_root / 'dpo-005', candidate='rft'))['training']['candidate'] == 'rft'
    assert len(calls) == before  # A rerun only verifies the finished merge.
    with pytest.raises(ValueError, match='trains in dpo-006'):
        ab.train(ns(dpo_root=run_root / 'dpo-005', candidate='pert'))
    monkeypatch.setattr(ab, 'matched_steps', lambda r: {**matched, 'pairs_manifest_sha256': '0' * 64})
    shutil.rmtree(run_root / 'dpo-005')  # Start rft again: its rows must be the chosen texts op-dpo2 trained on.
    with pytest.raises(ValueError, match='pairs op-dpo2 was trained on'):
        ab.train(ns(dpo_root=run_root / 'dpo-005', candidate='rft'))
    monkeypatch.setattr(ab, 'matched_steps', lambda r: {**matched, 'pairs_count': 11})
    with pytest.raises(ValueError, match='rft: 10 pairs, op-dpo2 was trained on 11'):
        ab.train(ns(dpo_root=run_root / 'dpo-005', candidate='rft'))
    assert not (run_root / 'dpo-005').exists()


def trained_op(pipeline, monkeypatch):
    """dpo-003 rpo-dpo1, dpo-004 op pairs (10 rows, 5 unique: S = 6) and op-dpo2 trained by onpolicy_dpo train ->
    (run_root, flags per train_dpo call, argv per call), with every later train_dpo call recorded the same way."""
    calls, commands = [], []
    monkeypatch.setattr(d.subprocess, 'run', D.fake_train(calls))
    run_root = pipeline.root.parent
    d.train(D.args(dpo_root=run_root / 'dpo-003', arm='placement', stage='dpo1', candidate='rpo'))
    root, _, _ = OP.op_pairs(pipeline, extra=5)
    inner = OP.fake_train_epochs(calls)
    monkeypatch.setattr(op.subprocess, 'run', lambda command, check: (commands.append(command), inner(command, check)))
    op.train(ns(dpo_root=root))
    return run_root, calls, commands


def masked(command, **values):
    """The argv with each --flag's value (asserted equal to VALUES[flag]) replaced by '*'."""
    out = list(command)
    for flag, value in values.items():
        k = out.index('--' + flag)
        assert out[k + 1] == value, (flag, out[k + 1], value)
        out[k + 1] = '*'
    return out


def committed_module(tmp_path):
    """ablation_016.py as committed at COMMITTED, loaded from a copy (skip if this checkout lacks the commit)."""
    done = subprocess.run(['git', '-C', str(OP.CODE), 'show', f'{COMMITTED}:fastfill/comparison/ablation_016.py'],
                          capture_output=True, text=True)
    if done.returncode:
        pytest.skip(f'commit {COMMITTED} not in this checkout')
    path = tmp_path / 'ablation_016_committed.py'
    path.write_text(done.stdout)
    spec = importlib.util.spec_from_file_location('ablation_016_committed', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_42_rft_and_pert_train_identity_equals_the_committed_ablation(pipeline, monkeypatch, tmp_path):
    """Seed 42 bit-identical: argv, guard (code hashes aside), train rows and the lineage receipt (hashes bound to the
    code file aside: guard, completion and ablation_016 sha256) of rft and pert equal what the committed code makes."""
    old = committed_module(tmp_path)
    run_root, _, commands = trained_op(pipeline, monkeypatch)
    seen = {}
    for module in (old, ab):
        for name, run in (('rft', 'dpo-005'), ('pert', 'dpo-006')):
            root = run_root / run
            receipt = module.train(ns(dpo_root=root, candidate=name))
            out = root / 'placement/dpo2'
            guard = h.strict_json((out / 'training_guard.json').read_text())
            text = json.dumps(receipt, sort_keys=True)
            for bound in (out / 'training_guard.json', out / 'final/completion.json', Path(module.__file__)):
                text = text.replace(h.sha256(bound), 'CODE-BOUND')
            files = {p.relative_to(root / 'train-rows').as_posix(): p.read_bytes()
                     for p in sorted((root / 'train-rows').iterdir())}
            code = {Path(k).name: v for k, v in guard.pop('code').items() if Path(k) != Path(module.__file__).resolve()}
            seen[module.__name__, name] = (commands[-1], guard, code, files, text)
            shutil.rmtree(root)
    for name in ('rft', 'pert'):
        assert seen[old.__name__, name] == seen[ab.__name__, name]
    assert seen[ab.__name__, 'rft'][1]['settings']['seed'] == seen[ab.__name__, 'pert'][1]['settings']['seed'] == 42


def test_seed_43_arms_match_their_seed_42_arm_except_seed_and_run(pipeline, monkeypatch):
    run_root, calls, commands = trained_op(pipeline, monkeypatch)
    op_command, op_receipt = commands[-1], d.verify_merged(run_root / 'dpo-004/placement/dpo2-merged', 'placement', 'dpo2')
    matched = ab.matched_steps(run_root)
    receipts, argv = {}, {}
    for name in ab.RUNS:
        receipts[name] = ab.train(ns(dpo_root=run_root / ab.RUNS[name], candidate=name))
        argv[name] = commands[-1]
    t = {n: r['training'] for n, r in receipts.items()}
    assert {n: x['recipe']['seed'] for n, x in t.items()} == ab.SEEDS == {'rft': 42, 'rft-s43': 43, 'pert': 42,
                                                                            'op-dpo2-s43': 43}
    assert all(x['completion']['steps'] == matched['steps'] == 6 for x in t.values())  # Step-matched to op-dpo2 e2.
    out = {n: str(run_root / ab.RUNS[n] / 'placement/dpo2') for n in ab.RUNS}
    # op-dpo2-s43 is onpolicy_dpo train's run with --seed 43: same --model, --pairs, --init_adapter, flags and switches.
    assert masked(argv['op-dpo2-s43'], seed='43', out=out['op-dpo2-s43']) == masked(
        op_command, seed='42', out=str(run_root / 'dpo-004/placement/dpo2'))
    assert t['op-dpo2-s43']['pairs'] == op_receipt['training']['pairs']  # The same dpo-004 pairs file, not a copy.
    assert t['op-dpo2-s43']['recipe'] == {**op_receipt['training']['recipe'], 'seed': 43}  # Frozen reference: the same.
    assert {k: receipts['op-dpo2-s43'][k] for k in ('parent', 'training_base', 'base_sha256', 'training_data')} == {
        k: op_receipt[k] for k in ('parent', 'training_base', 'base_sha256', 'training_data')}
    guards = {n: h.strict_json((Path(p) / 'training_guard.json').read_text())
              for n, p in {**out, 'op': run_root / 'dpo-004/placement/dpo2'}.items()}
    assert {**guards['op']['settings'], 'seed': 43} == guards['op-dpo2-s43']['settings']
    assert guards['op']['pairs'] == guards['op-dpo2-s43']['pairs'] and guards['op']['model'] == guards['op-dpo2-s43']['model']
    assert 'epoch' not in t['op-dpo2-s43'] and not (run_root / 'dpo-007-s43/placement/dpo2-epoch01-merged').exists()
    assert not (run_root / 'dpo-007-s43/train-rows').exists() and t['op-dpo2-s43']['rows']['verbatim']
    # rft-s43 is rft with --seed 43: the same rows (its own train-rows copy, byte for byte) and flags.
    assert masked(argv['rft-s43'], seed='43', out=out['rft-s43'], pairs=t['rft-s43']['pairs']['path']) == masked(
        argv['rft'], seed='42', out=out['rft'], pairs=t['rft']['pairs']['path'])
    assert t['rft-s43']['pairs']['sha256'] == t['rft']['pairs']['sha256'] == op_receipt['training']['pairs']['sha256']
    assert t['rft-s43']['recipe'] == {**t['rft']['recipe'], 'seed': 43}
    selection = h.strict_json((run_root / 'dpo-005-s43/train-rows/selection.json').read_text())
    assert 'seed 43, as op-dpo2-s43' in selection['order'] and selection['candidate'] == 'rft-s43'
    for name in ab.RUNS:  # What infer_shard --exact-policy loads: rpo-dpo1's training base + this run's final adapter.
        _, exact = d.exact_policy(ab.model_path(run_root, name), 'placement')
        assert exact['adapter'] == str((run_root / ab.RUNS[name] / 'placement/dpo2/final').resolve())
    lineage = ab.training(run_root)
    assert {n: x['seed'] for n, x in lineage.items()} == {**ab.SEEDS, 'op-dpo2-e2': 42}
    assert lineage['op-dpo2-e2'] == {**matched, 'seed': 42} and {x['steps'] for x in lineage.values()} == {6}
    monkeypatch.setitem(ab.SEEDS, 'rft-s43', 44)
    with pytest.raises(ValueError, match='step-matched rft-s43 run with seed 44'):
        ab.training(run_root)
    monkeypatch.setitem(ab.SEEDS, 'rft-s43', 43)
    shutil.rmtree(run_root / 'dpo-007-s43')
    monkeypatch.setattr(ab, 'matched_steps', lambda r: {**matched, 'steps': 8})
    with pytest.raises(ValueError, match='op-dpo2-s43 ran 6 optimiser steps, op-dpo2 e2 8'):
        ab.train(ns(dpo_root=run_root / 'dpo-007-s43', candidate='op-dpo2-s43'))
    assert not ab.model_path(run_root, 'op-dpo2-s43').exists()
    with pytest.raises(ValueError, match='trains in dpo-007-s43'):
        ab.train(ns(dpo_root=run_root / 'dpo-005-s43', candidate='op-dpo2-s43'))


def contrast(lo, hi):
    return {'estimate': (lo + hi) / 2 if lo is not None else None, 'ci95': None if lo is None else [lo, hi]}


def both(lo, hi):
    return {'success_debiased': contrast(lo, hi), 'accepted': contrast(lo, hi)}


ABOVE, ZERO, BELOW, NONE = (.01, .05), (-.01, .02), (-.05, -.01), (None, None)


@pytest.mark.parametrize('seed42, seed43, reading', [
    ((ABOVE, ZERO), (ABOVE, ZERO), ('preference_beyond_repaired_data', 'real_failures_add_nothing_over_synthetic')),
    ((ZERO, ABOVE), (ZERO, ABOVE), ('contribution_is_repair_data', 'not_pre_registered')),
    ((ABOVE, ZERO), (ZERO, ZERO), ('seeds_disagree', 'real_failures_add_nothing_over_synthetic')),
    ((ZERO, ZERO), (ZERO, BELOW), ('contribution_is_repair_data', 'seeds_disagree')),
    ((BELOW, NONE), (NONE, ABOVE), ('not_pre_registered', 'not_pre_registered'))])
def test_interpretation_is_the_fixed_rule_per_seed_and_claims_only_when_both_seeds_agree(seed42, seed43, reading):
    contrasts = {'op-dpo2-e2 - rft': both(*seed42[0]), 'op-dpo2-e2 - pert': both(*seed42[1]),
                 'op-dpo2-s43 - rft-s43': both(*seed43[0]), 'op-dpo2-s43 - pert': both(*seed43[1])}
    value = ab.interpret(contrasts)
    assert (value['d_vs_b'], value['c_vs_d']) == reading
    assert value['per_seed']['42']['sides']['op-dpo2-e2 - rft']['accepted'] == ab._side(contrast(*seed42[0]))
    for seed in ('42', '43'):  # Each seed on its own is the amendment-016 rule.
        assert value['per_seed'][seed]['d_vs_b'] in ab.READINGS and value['per_seed'][seed]['d_vs_b'] != 'seeds_disagree'
    mixed = {**contrasts, 'op-dpo2-s43 - rft-s43': {'success_debiased': contrast(*ABOVE), 'accepted': contrast(*ZERO)}}
    assert ab.interpret(mixed)['per_seed']['43']['d_vs_b'] == 'not_pre_registered'  # Both metrics must agree.


def test_tables_stratify_every_metric_by_section_5_strata_with_two_seed_means():
    def row(uid, objects, complete, success, accepted, mae, bias):
        return {'uid': uid, 'objects': objects, 'complete': complete, 'geometry_success': success,
                'success_debiased': success, 'accepted': accepted, 'size_log_mae': mae if complete else None,
                'bias_fp': bias if complete else None}
    rows = {n: [row('a', 5, True, True, k % 2 == 0, .1 * k, -.2), row('b', 30, True, k < 3, False, .2, .1 * k),
                row('c', 60, False, False, False, None, None), row('d', 51, True, False, True, .3, .2)]
            for k, n in enumerate(ab.ORDER)}
    result = {n: {**ab.levels(v), 'requests': 4, 'halves': {x: {'a_prime': .5} for x in op.HALVES}} for n, v in rows.items()}
    rooms = {u: {'uid': u, 'objects': [{}] * n, 'source': s, **({'room_type': t} if t else {})} for u, n, s, t in (
        ('a', 5, 'Structured3D', 'bedroom'), ('b', 30, 'OptiScene_holodeck', 'living room'), ('c', 60, 'IL3D_3dfront', None),
        ('d', 51, 'MultiScan', 'kitchen'))}
    value = ab.tables(result, rows, rooms)
    assert {k: v['rooms'] for k, v in value.items()} == {  # Amendment 016 section 5 (scale_curve.subsets), empty ones out.
        'all': 4, 'objects<=20': 1, 'objects21-50': 1, 'objects>50': 2, 'real': 2, 'synthetic': 2,
        'family:paper_families': 3, 'family:bedroom': 1, 'family:living': 1, 'family:kitchen': 1, 'family:none': 1}
    assert value['real']['candidates']['pert'] == ab.levels([r for r in rows['pert'] if r['uid'] in ('a', 'd')])
    assert value['family:none']['candidates']['rft']['complete_rate'] == 0  # Room c only.
    assert list(value['all']['candidates']) == [*ab.ORDER, 'rft-mean', 'op-dpo2-mean']
    k = {n: i for i, n in enumerate(ab.ORDER)}
    small, middle, dense = (value[b]['candidates'] for b in ('objects<=20', 'objects21-50', 'objects>50'))
    assert small['rft']['size_log_mae_own'] == .1 * k['rft'] and small['rft-s43']['footprint_bias'] == -.2
    assert middle['op-dpo2-s43']['footprint_bias'] == pytest.approx(.1 * k['op-dpo2-s43'])
    assert middle['rft']['a_prime'] == 1 and middle['op-dpo2-e2']['a_prime'] == 0  # k < 3.
    assert dense['pert'] == {'complete_rate': .5, 'a_prime': 0, 'accepted_rate': .5, 'both_pass_rate': 0,
                             'geometry_success_rate': 0, 'size_log_mae_own': .3, 'footprint_bias': .2}
    assert small['rft-mean']['accepted_rate'] == (small['rft']['accepted_rate'] + small['rft-s43']['accepted_rate']) / 2
    assert middle['op-dpo2-mean']['footprint_bias'] == pytest.approx(
        (middle['op-dpo2-e2']['footprint_bias'] + middle['op-dpo2-s43']['footprint_bias']) / 2)
    assert ab.levels([row('c', 60, False, False, False, None, None)])['footprint_bias'] is None
    assert ab.with_means({**value['all']['candidates'], 'rft-s43': {**value['all']['candidates']['rft-s43'],
                                                                     'size_log_mae_own': None}})['rft-mean'][
        'size_log_mae_own'] is None  # A missing seed value gives no mean.
    assert value['all']['candidates']['rft']['a_prime_paper_sources'] == .5


def run_ablation(w, name, kind):
    return OP.run_inference(w.root, w.data, w.plan, w.root / 'dev-eval-016' / name / 'infer-placement',
                            ['--model', str(ab.model_path(w.root, name)), '--exact-policy'],
                            OP.identity(w.training, 'dpo2'), kind)


def test_score_pairs_every_seed_reports_b_bins_means_and_claims_only_on_seed_agreement(dev, monkeypatch):
    w = OP.dev_world(dev, monkeypatch, rpo='collide', ops=('collide', 'good'))
    op.dev_score(OP.score_args(w))
    kinds = {'rft': 'good', 'rft-s43': 'collide', 'pert': 'collide', 'op-dpo2-s43': 'good'}
    for name, kind in kinds.items():
        run_ablation(w, name, kind)
    rows = {'source': {}, 'unique': 5, 'rows_per_epoch': 12, 'full_cycles': 2, 'extra_rows': 2}
    monkeypatch.setattr(ab, 'training', lambda run_root: {  # Receipts are tested above; here no weights or receipts.
        **{n: {'steps': 6, 'rows': rows, 'seed': ab.SEEDS[n]} for n in ab.RUNS},
        'op-dpo2-e2': {'steps': 6, 'epochs': 2, 'seed': 42}})
    out = w.root / 'dev-eval-016/ablation.json'
    args = ns(data=w.data, run_root=w.root, out=out, workers=1)
    result = ab.score(args)
    value = h.strict_json(out.read_text())
    rooms = len(h.strict_json((w.data / 'MANIFEST.json').read_text())['expected_uids'])
    c = value['candidates']
    assert [c[n]['a_prime'] for n in ab.ORDER] == [0, 1, 0, 0, 1, 1] and [c[n]['accepted_rate'] for n in ab.ORDER] == [
        0, 1, 0, 0, 1, 1]
    assert result['rft'] == {'a_prime': 1.0, 'accepted_rate': 1.0, 'complete_rate': 1.0}
    assert c['op-dpo2-s43']['inference']['path'] == str((w.root / 'dev-eval-016/op-dpo2-s43/infer-placement').resolve())
    assert c['op-dpo2-e2']['inference']['path'] == str((w.eval / 'op-dpo2-e2/infer-placement').resolve())  # Seed 42.
    assert list(value['contrasts']) == [f'{x} - {y}' for x, y in ab.PAIRS] and len(ab.PAIRS) == 11
    k = value['contrasts']
    assert k['rft - rpo-dpo1']['accepted']['estimate'] == 1 and k['pert - rft']['success_debiased']['estimate'] == -1
    assert k['op-dpo2-e2 - rft']['success_debiased']['ci95'] == [0, 0]  # Seed 42: (d) ~ (b).
    assert k['op-dpo2-s43 - rft-s43']['success_debiased']['estimate'] == 1  # Seed 43: (d) > (b).
    reading = value['interpretation']
    assert (reading['per_seed']['42']['d_vs_b'], reading['per_seed']['43']['d_vs_b']) == (
        'contribution_is_repair_data', 'preference_beyond_repaired_data')
    assert reading['d_vs_b'] == 'seeds_disagree' and reading['c_vs_d'] == 'not_pre_registered'
    assert result['interpretation'] == {'d_vs_b': 'seeds_disagree', 'c_vs_d': 'not_pre_registered', 'per_seed': {
        '42': {'d_vs_b': 'contribution_is_repair_data', 'c_vs_d': 'not_pre_registered'},
        '43': {'d_vs_b': 'preference_beyond_repaired_data', 'c_vs_d': 'not_pre_registered'}}}
    assert value['seeds']['single_seed'] == ['rpo-dpo1', 'pert'] and value['seeds']['by_seed'] == {
        '42': ['rft', 'op-dpo2-e2'], '43': ['rft-s43', 'op-dpo2-s43']}
    assert all(len(value['rows'][n]) == rooms and set(value['rows'][n][0]) == set(ab.ROW_KEYS) for n in ab.ORDER)
    tab = value['tables']
    for n in ab.ORDER:  # The per-room rows reproduce every reported metric, B included; the bins partition the rooms.
        assert ab.levels(value['rows'][n]) == {m: c[n][m] for m in ab.METRICS}
        assert {m: tab['all']['candidates'][n][m] for m in ab.METRICS} == {m: c[n][m] for m in ab.METRICS}
        assert c[n]['footprint_bias'] is not None
    assert sum(tab[b]['rooms'] for b, _, _ in ab.sc.BINS if b in tab) == tab['all']['rooms'] == rooms  # Empty: left out.
    assert tab['real']['rooms'] + tab['synthetic']['rooms'] == rooms and any(k.startswith('family:') for k in tab)
    assert tab['all']['candidates']['rft-mean']['a_prime'] == .5 and tab['all']['candidates']['op-dpo2-mean']['a_prime'] == 1
    report = (w.root / 'dev-eval-016' / ab.REPORT).read_text()
    assert '修订 016 §3' in report and ab.READINGS['seeds_disagree'] in report and '(b) RFT s43' in report
    assert '| (a) rpo-dpo1 | 42（单种子） |' in report and '| (c) pert-DPO2 | 42（单种子） |' in report
    assert '(d) op-dpo2 e2 两种子均值 | 42、43 均值 |' in report and '足迹偏差 B' in report and '训练先验' in report
    assert all(f'### {b}（' in report for b in tab if b != 'all') and 'LoRA 初始化' in report
    assert 'N = 12，每 epoch 补齐 0 行' in report and '不只差 DPO 项' in report  # rows_per_epoch 12 above.
    assert ('  - seed 42（(d) op-dpo2 e2 s42 − (b) RFT s42）：' + ab.READINGS['contribution_is_repair_data']
            + '（a′ CI 含 0 +0.000 [+0.000, +0.000]，下游接受率 CI 含 0') in report and '±0.01' in report
    with pytest.raises(FileExistsError):
        ab.score(args)
    out.unlink()  # A crash after the report: the same report is accepted, ablation.json written again.
    assert ab.score(args)['status'] == 'ABLATION_016_COMPLETE'
    out.unlink()
    eval_root = w.root / 'dev-eval-016'
    (eval_root / 'rft').rename(eval_root / 'tmp')  # Swapped runs are refused by the model binding.
    (eval_root / 'rft-s43').rename(eval_root / 'rft')
    (eval_root / 'tmp').rename(eval_root / 'rft-s43')
    with pytest.raises(ValueError, match='dev inference is not placement/rft'):
        ab.score(args)
    assert not out.exists()


# ---- Slurm scripts -------------------------------------------------------------------------------

INPUTS = ('dpo-001/placement/sft-merged.json', 'dpo-001/stage2-pairs/MANIFEST.json', 'dpo-003/placement/dpo1-merged.json',
          'dpo-003/placement/dpo1/final/adapter_model.safetensors', 'dpo-004/stage2-pairs/MANIFEST.json',
          'dev-select-v1/MANIFEST.json', 'dev-eval-010/plan/MANIFEST.json', 'dev-eval-010/scores/MANIFEST.json',
          'context-capacity-001/SHA256.json')
TRAINED = ('dpo-004/placement/dpo2-merged.json', 'dpo-004/placement/dpo2/final/completion.json')
DOC = 'docs/research/20261013-unknown-size-comparison/amendment-016-size-confound-and-ablations-20261014.md'


def submit_world(tmp_path):
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    for path in INPUTS + TRAINED:
        (run / path).parent.mkdir(parents=True, exist_ok=True)
        (run / path).write_text('{}')
    (project / 'models/Qwen3-8B').mkdir(parents=True)
    script = tmp_path / 'submit_016_ablation.sh'
    original = (SCRIPTS / 'submit_016_ablation.sh').read_text()
    line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    assert original.count(line) == 1
    script.write_text(original.replace(line, f'PROJECT_ROOT={project}\n'))
    code = tmp_path / 'main-abc'
    code.mkdir()
    (code / 'fastfill').symlink_to(OP.CODE / 'fastfill')
    (code / DOC).parent.mkdir(parents=True)
    (code / DOC).write_text('# Amendment 016\n')
    (tmp_path / 'squeue').write_text(  # This user's 016 jobs ($SQUEUE_OUT); 7400001 is the amendment-013 train job,
        '#!/bin/sh\ncase "$*" in\n  "-h -u me -n unknownsize-016 -o %i") printf "%s" "$SQUEUE_OUT" ;;\n'  # 7400004 its pairs.
        '  "-h -j 7400001 -o %j %l") echo "unknownsize-013 12:00:00" ;;\n'
        '  "-h -j 7400002 -o %j %l") echo "unknownsize-014 12:00:00" ;;\n'
        '  "-h -j 7400004 -o %j %l") echo "unknownsize-013 8:00:00" ;;\n  *) exit 1 ;;\nesac\n')
    (tmp_path / 'squeue').chmod(0o755)
    return script, run, code


KILL, SHARD = '--parsable --kill-on-invalid-dep=yes --dependency=afterok:', ' '.join(op.SHARD)
TRAIN = '--gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=12:00:00 isambard_016.sbatch CODE train'
CANDS = ('rft', 'rft-s43', 'pert', 'op-dpo2-s43')


def chain(first, cands=CANDS, held='--parsable --hold', resume=()):
    """The sbatch lines submit_016_ablation.sh sends (stub job ids from FIRST) and the held roots it releases."""
    log, merges, roots, k = [], [], [], first
    for cand in cands:
        log.append(f'{held} {TRAIN} {cand}')
        roots.append(k)
        log += [f'{KILL}{roots[-1]} {SHARD} isambard_016.sbatch CODE shard {cand} {i} 2'
                + (' --resume' if (cand, i) in resume else '') for i in range(2)]
        log.append(f'{KILL}{k + 1}:{k + 2} isambard_016.sbatch CODE dev-merge {cand} 2')
        merges.append(k + 3)
        k += 4
    log.append(f"{KILL}{':'.join(map(str, merges))} isambard_016.sbatch CODE score" if merges else
               f'{held} isambard_016.sbatch CODE score')
    return log, 'scontrol release ' + ','.join(map(str, roots or [k]))


def test_submit_016_chain_held_trains_after_job_and_refusals(tmp_path):
    for name in ('isambard_016.sbatch', 'submit_016_ablation.sh'):
        subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    script, run, code = submit_world(tmp_path)
    sent = []

    def submit(*argv, **extra):
        before = len(sent)
        done = subprocess.run(['bash', str(script), str(code), *argv], env={**env, **extra}, capture_output=True, text=True)
        log = tmp_path / 'log'
        sent[:] = log.read_text().splitlines() if log.exists() else []
        return done, [x.replace(f'{code}/fastfill/comparison/', '').replace(f' {code} ', ' CODE ') for x in sent[before:]], (
            101 + before)

    def released():
        return (tmp_path / 'log.scontrol').read_text().splitlines()[-1]
    done, log, first = submit()
    assert done.returncode == 0, done.stderr
    expected, release = chain(first)  # Trains rft, rft-s43, pert, op-dpo2-s43 (held roots), shards, merges, one score.
    assert log == expected and len(log) == 17 and released() == release == 'scontrol release 101,105,109,113'
    assert [x.split()[-1] for x in log if ' train ' in x] == list(CANDS)
    for path in TRAINED:  # Before amendment-013 training ends: only with its job to depend on.
        (run / path).unlink()
    done, log, _ = submit()
    assert log == [] and done.returncode == 1 and 'dpo-004/placement/dpo2-merged.json' in done.stderr
    (run / 'dev-eval-016/op-dpo2-s43/placement/shard-01').mkdir(parents=True)  # A timed-out dev shard resumes.
    done, log, first = submit('7400001')
    held = '--parsable --hold --kill-on-invalid-dep=yes --dependency=afterok:7400001'
    assert done.returncode == 0, done.stderr
    assert (log, released()) == chain(first, held=held, resume={('op-dpo2-s43', 1)})
    failed = tmp_path / 'failed'
    failed.mkdir()
    shutil.copy(tmp_path / 'squeue', failed / 'squeue')
    bad = subprocess.run(['bash', str(script), str(code), '7400001'], env={**T.stub_sbatch(failed, fail_on=3), 'USER': 'me'},
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "sbatch failed for 'dev shard rft 1'" in bad.stderr and 'scancel 101 102' in bad.stderr
    assert not (failed / 'log.scontrol').exists()  # The held trains never ran.
    (run / 'dev-eval-016/ablation.json').write_text('{}')
    refusals = [(('7400002',), {}, 'not the amendment-013 train job'), (('123',), {}, 'not the amendment-013 train job'),
                (('7400004',), {}, "not the amendment-013 train job in Slurm (got 'unknownsize-013 8:00:00')"),
                (('x1',), {}, None), (('1', '2'), {}, None),
                (('7400001',), {}, 'refusing to reuse existing ' + str(run / 'dev-eval-016/ablation.json'))]
    for argv, extra, message in refusals:
        done, log, _ = submit(*argv, **extra)
        assert log == [] and (done.returncode, message is None or message in done.stderr) == (2 if message is None else 1, True)
    (run / 'dev-eval-016/ablation.json').unlink()
    for cand in CANDS:  # A merge without its manifest refuses, for every candidate.
        (run / f'dev-eval-016/{cand}/infer-placement').mkdir(parents=True)
        done, log, _ = submit('7400001')
        assert log == [] and done.returncode == 1 and f"to reuse existing {run / 'dev-eval-016' / cand}" in done.stderr
        (run / f'dev-eval-016/{cand}/infer-placement/MANIFEST.json').write_text('{}')
    shutil.rmtree(run / 'dev-eval-016/pert/infer-placement')  # Three merged, pert failed: pert again.
    done, log, first = submit('7400001')
    assert done.returncode == 0 and all(f'skipped {c}: dev inference merged' in done.stdout for c in CANDS if c != 'pert')
    assert (log, released()) == chain(first, ('pert',), held=held)
    (run / 'dev-eval-016/pert/infer-placement').mkdir(parents=True)
    (run / 'dev-eval-016/pert/infer-placement/MANIFEST.json').write_text('{}')  # All merged: score is the held root.
    done, log, first = submit('7400001')
    assert done.returncode == 0 and (log, released()) == chain(first, (), held=held)
    assert log == [f'{held} isambard_016.sbatch CODE score']
    for cand in CANDS:
        shutil.rmtree(run / f'dev-eval-016/{cand}/infer-placement')
    done, log, _ = submit('7400001', SQUEUE_OUT='7400003\n')
    assert log == [] and done.returncode == 1 and 'scancel 7400003' in done.stderr
    (run / 'dpo-001/stage2-pairs/MANIFEST.json').unlink()  # The pert pairs.
    done, log, _ = submit('7400001')
    assert log == [] and done.returncode == 1 and 'dpo-001/stage2-pairs/MANIFEST.json' in done.stderr
    (run / 'dpo-001/stage2-pairs/MANIFEST.json').write_text('{}')
    (code / DOC).unlink()
    done, log, _ = submit('7400001')
    assert log == [] and done.returncode == 1 and 'no amendment-016 protocol doc' in done.stderr


@pytest.mark.parametrize('argv, ok', [
    (['train', 'rft'], True), (['train', 'pert'], True), (['train', 'rft-s43'], True), (['train', 'op-dpo2-s43'], True),
    (['train', 'op'], False), (['train', 'op-dpo2-e2'], False), (['train', 'rft-s44'], False), (['train'], False),
    (['shard', 'rft', '0', '2'], True), (['shard', 'pert', '1', '2', '--resume'], True), (['shard', 'rft', '2', '2'], False),
    (['shard', 'op-dpo2-s43', '1', '2'], True), (['dev-merge', 'rft-s43', '2'], True),
    (['shard', 'op-dpo2-e2', '0', '2'], False), (['shard', 'rft', '0', '2', '--x'], False), (['dev-merge', 'pert', '2'], True),
    (['dev-merge', 'sft', '2'], False), (['score'], True), (['score', 'x'], False), (['bogus'], False)])
def test_sbatch_modes_validate_arguments_before_the_allocation(tmp_path, argv, ok):
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    done = subprocess.run(['bash', str(SCRIPTS / 'isambard_016.sbatch'), str(OP.CODE), *argv], env=env, capture_output=True,
                          text=True)
    assert (done.returncode, 'Slurm allocation required' in done.stderr) == ((1, True) if ok else (2, False))


# ---- label guard: size-fixed evaluations never enter J-input agent tables -------------------------

@pytest.mark.parametrize('arm, ok', [(None, True), ('joint', True), ('placement', False), ('anything', False)])
def test_check_labels_refuses_size_fixed_evaluations_under_any_label(arm, ok):
    for label in ('L', 'O2', 'X'):  # X: not an agent label; a size-fixed summary is refused there too.
        config = {**(agent_config(label) if label != 'X' else {}), **({'request_arm': arm} if arm else {})}
        summaries = {'J': {}, 'P': {}, label: {'llm': {'config': config}}}
        if ok:
            vp.check_labels(summaries)
        else:
            with pytest.raises(ValueError, match=f'system label {label}: a size-fixed'):
                vp.check_labels(summaries)


def test_complete_agents_refuses_size_fixed_evaluations(tmp_path):
    def evaluation(name, **config):
        root = tmp_path / name
        root.mkdir()
        h.write_json(root / 'summary.json', {'status': 'LLM_COMPARISON_COMPLETE', 'llm': {'config': config}})
        h.write_json(root / 'MANIFEST.json', {'files': h.artifact_manifest(root)})
        return root
    size_free, joint = evaluation('free', model='m'), evaluation('joint', model='m', request_arm='joint')
    fixed, unfinished = evaluation('fixed', model='m', request_arm='placement'), tmp_path / 'unfinished'
    agents, excluded = ds.complete_agents([f'O2={size_free}', f'O4={joint}', f'O6={unfinished}'])
    assert agents == {'O2': size_free, 'O4': joint} and list(excluded) == ['O6']  # Absent request_arm = joint.
    with pytest.raises(ValueError, match=f'O7={fixed}: a size-fixed'):
        ds.complete_agents([f'O2={size_free}', f'O7={fixed}'])
