"""Amendment 016 section 3 DPO2 attribution ablation on CPU: fakes stand in for train_dpo, merge_lora, the models and
Slurm (fixtures of the amendment-013 tests)."""
import json
from pathlib import Path
import random
import shutil
import subprocess
import types

import pytest

from fastfill.comparison import ablation_016 as ab
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_dpo_stages as D
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests import test_onpolicy_dpo as OP
from fastfill.comparison.tests.test_dev_select import dev  # noqa: F401  Fixture: 5 frozen dev rooms.
from fastfill.comparison.tests.test_dpo_stages import pipeline  # noqa: F401  Fixture: dpo-001 on CPU.

SCRIPTS = OP.SCRIPTS


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
    lineage = ab.training(run_root)
    assert lineage['op-dpo2-e2'] == matched and lineage['rft']['steps'] == lineage['pert']['steps'] == 6
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


def contrast(lo, hi):
    return {'estimate': (lo + hi) / 2 if lo is not None else None, 'ci95': None if lo is None else [lo, hi]}


@pytest.mark.parametrize('db, dc, reading', [
    ((.01, .05), (-.01, .02), ('preference_beyond_repaired_data', 'real_failures_add_nothing_over_synthetic')),
    ((-.01, .02), (.01, .03), ('contribution_is_repair_data', 'not_pre_registered')),
    ((-.05, -.01), (None, None), ('not_pre_registered', 'not_pre_registered'))])
def test_interpretation_is_the_fixed_rule_on_a_prime_and_acceptance(db, dc, reading):
    contrasts = {'op-dpo2-e2 - rft': {'success_debiased': contrast(*db), 'accepted': contrast(*db)},
                 'op-dpo2-e2 - pert': {'success_debiased': contrast(*dc), 'accepted': contrast(*dc)}}
    assert (ab.interpret(contrasts)['d_vs_b'], ab.interpret(contrasts)['c_vs_d']) == reading
    mixed = {**contrasts, 'op-dpo2-e2 - rft': {'success_debiased': contrast(.01, .05), 'accepted': contrast(-.01, .02)}}
    assert ab.interpret(mixed)['d_vs_b'] == 'not_pre_registered'  # Both metrics must agree.


def run_ablation(w, name, kind):
    return OP.run_inference(w.root, w.data, w.plan, w.root / 'dev-eval-016' / name / 'infer-placement',
                            ['--model', str(ab.model_path(w.root, name)), '--exact-policy'],
                            OP.identity(w.training, 'dpo2'), kind)


def test_score_pairs_the_four_candidates_and_applies_the_fixed_reading(dev, monkeypatch):
    w = OP.dev_world(dev, monkeypatch, rpo='collide', ops=('collide', 'good'))
    op.dev_score(OP.score_args(w))
    run_ablation(w, 'rft', 'good')
    run_ablation(w, 'pert', 'collide')
    rows = {'source': {}, 'unique': 5, 'rows_per_epoch': 12, 'full_cycles': 2, 'extra_rows': 2}
    monkeypatch.setattr(ab, 'training', lambda run_root: {  # Receipts are tested above; here no weights or receipts.
        'rft': {'steps': 6, 'rows': rows}, 'pert': {'steps': 6, 'rows': rows}, 'op-dpo2-e2': {'steps': 6, 'epochs': 2}})
    out = w.root / 'dev-eval-016/ablation.json'
    args = ns(data=w.data, run_root=w.root, out=out, workers=1)
    result = ab.score(args)
    value = h.strict_json(out.read_text())
    rooms = len(h.strict_json((w.data / 'MANIFEST.json').read_text())['expected_uids'])
    c = value['candidates']
    assert [c[n]['a_prime'] for n in ab.ORDER] == [0, 1, 0, 1] and [c[n]['accepted_rate'] for n in ab.ORDER] == [0, 1, 0, 1]
    assert result['rft'] == {'a_prime': 1.0, 'accepted_rate': 1.0, 'complete_rate': 1.0}
    assert c['rft']['inference']['path'] == str((w.root / 'dev-eval-016/rft/infer-placement').resolve())
    assert c['op-dpo2-e2']['inference']['path'] == str((w.eval / 'op-dpo2-e2/infer-placement').resolve())
    assert list(value['contrasts']) == ['rft - rpo-dpo1', 'pert - rpo-dpo1', 'op-dpo2-e2 - rpo-dpo1',
                                        'op-dpo2-e2 - rft', 'op-dpo2-e2 - pert', 'pert - rft']
    k = value['contrasts']
    assert k['rft - rpo-dpo1']['accepted']['estimate'] == 1 and k['pert - rft']['success_debiased']['estimate'] == -1
    assert k['op-dpo2-e2 - rft']['success_debiased'] == {**k['op-dpo2-e2 - rft']['success_debiased'], 'estimate': 0,
                                                         'ci95': [0, 0]}
    assert set(k['op-dpo2-e2 - pert']['size_log_mae']) >= {'estimate', 'ci95', 'n'}
    assert value['interpretation']['d_vs_b'] == 'contribution_is_repair_data'  # (d) and (b) are indistinguishable here.
    assert value['interpretation']['c_vs_d'] == 'not_pre_registered'  # (d) beats (c): no pre-registered reading.
    assert all(len(value['rows'][n]) == rooms and set(value['rows'][n][0]) == set(ab.ROW_KEYS) for n in ab.ORDER)
    report = (w.root / 'dev-eval-016' / ab.REPORT).read_text()
    assert '修订 016 §3' in report and ab.READINGS['contribution_is_repair_data'] in report and '(b) RFT' in report
    assert '（a′ CI 含 0 +0.000 [+0.000, +0.000]，下游接受率 CI 含 0' in report and '±0.01' in report  # CI by the reading.
    with pytest.raises(FileExistsError):
        ab.score(args)
    out.unlink()  # A crash after the report: the same report is accepted, ablation.json written again.
    assert ab.score(args)['status'] == 'ABLATION_016_COMPLETE'
    out.unlink()
    eval_root = w.root / 'dev-eval-016'
    (eval_root / 'rft').rename(eval_root / 'tmp')  # Swapped runs are refused by the model binding.
    (eval_root / 'pert').rename(eval_root / 'rft')
    (eval_root / 'tmp').rename(eval_root / 'pert')
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
        return done, [x.replace(f'{code}/fastfill/comparison/', '').replace(f' {code} ', ' CODE ') for x in sent[before:]]
    done, log = submit()
    assert done.returncode == 0, done.stderr
    kill, shard = '--parsable --kill-on-invalid-dep=yes --dependency=afterok:', ' '.join(op.SHARD)
    train = '--gpus-per-node=4 --cpus-per-task=72 --mem=0 --time=12:00:00 isambard_016.sbatch CODE train'
    assert log == [
        f'--parsable --hold {train} rft',                                                          # 101
        f'{kill}101 {shard} isambard_016.sbatch CODE shard rft 0 2',                               # 102
        f'{kill}101 {shard} isambard_016.sbatch CODE shard rft 1 2',                               # 103
        f'{kill}102:103 isambard_016.sbatch CODE dev-merge rft 2',                                 # 104
        f'--parsable --hold {train} pert',                                                         # 105
        f'{kill}105 {shard} isambard_016.sbatch CODE shard pert 0 2',                              # 106
        f'{kill}105 {shard} isambard_016.sbatch CODE shard pert 1 2',                              # 107
        f'{kill}106:107 isambard_016.sbatch CODE dev-merge pert 2',                                # 108
        f'{kill}104:108 isambard_016.sbatch CODE score']                                           # 109
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101,105\n'
    for path in TRAINED:  # Before amendment-013 training ends: only with its job to depend on.
        (run / path).unlink()
    done, log = submit()
    assert log == [] and done.returncode == 1 and 'dpo-004/placement/dpo2-merged.json' in done.stderr
    (run / 'dev-eval-016/pert/placement/shard-01').mkdir(parents=True)  # A timed-out dev shard resumes.
    done, log = submit('7400001')
    assert done.returncode == 0, done.stderr
    held = '--parsable --hold --kill-on-invalid-dep=yes --dependency=afterok:7400001'
    assert log[0] == f'{held} {train} rft' and log[4] == f'{held} {train} pert'
    assert log[6] == f'{kill}114 {shard} isambard_016.sbatch CODE shard pert 1 2 --resume' and len(log) == 9
    assert (tmp_path / 'log.scontrol').read_text().splitlines()[-1] == 'scontrol release 110,114'
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
        done, log = submit(*argv, **extra)
        assert log == [] and (done.returncode, message is None or message in done.stderr) == (2 if message is None else 1, True)
    (run / 'dev-eval-016/ablation.json').unlink()
    (run / 'dev-eval-016/rft/infer-placement').mkdir(parents=True)  # A merge without its manifest refuses.
    done, log = submit('7400001')
    assert log == [] and done.returncode == 1 and f"to reuse existing {run / 'dev-eval-016/rft'}" in done.stderr
    (run / 'dev-eval-016/rft/infer-placement/MANIFEST.json').write_text('{}')  # rft merged, pert failed: pert again.
    done, log = submit('7400001')
    assert done.returncode == 0 and 'skipped rft: dev inference merged' in done.stdout
    assert log == [f'{held} {train} pert', f'{kill}119 {shard} isambard_016.sbatch CODE shard pert 0 2',
                   f'{kill}119 {shard} isambard_016.sbatch CODE shard pert 1 2 --resume',
                   f'{kill}120:121 isambard_016.sbatch CODE dev-merge pert 2',
                   f'{kill}122 isambard_016.sbatch CODE score']
    assert (tmp_path / 'log.scontrol').read_text().splitlines()[-1] == 'scontrol release 119'
    (run / 'dev-eval-016/pert/infer-placement').mkdir(parents=True)
    (run / 'dev-eval-016/pert/infer-placement/MANIFEST.json').write_text('{}')  # Both merged: score is the held root.
    done, log = submit('7400001')
    assert done.returncode == 0 and log == [f'{held} isambard_016.sbatch CODE score']
    assert (tmp_path / 'log.scontrol').read_text().splitlines()[-1] == 'scontrol release 124'
    shutil.rmtree(run / 'dev-eval-016/rft/infer-placement')
    shutil.rmtree(run / 'dev-eval-016/pert/infer-placement')
    done, log = submit('7400001', SQUEUE_OUT='7400003\n')
    assert log == [] and done.returncode == 1 and 'scancel 7400003' in done.stderr
    (run / 'dpo-001/stage2-pairs/MANIFEST.json').unlink()  # The pert pairs.
    done, log = submit('7400001')
    assert log == [] and done.returncode == 1 and 'dpo-001/stage2-pairs/MANIFEST.json' in done.stderr
    (run / 'dpo-001/stage2-pairs/MANIFEST.json').write_text('{}')
    (code / DOC).unlink()
    done, log = submit('7400001')
    assert log == [] and done.returncode == 1 and 'no amendment-016 protocol doc' in done.stderr


@pytest.mark.parametrize('argv, ok', [
    (['train', 'rft'], True), (['train', 'pert'], True), (['train', 'op'], False), (['train'], False),
    (['shard', 'rft', '0', '2'], True), (['shard', 'pert', '1', '2', '--resume'], True), (['shard', 'rft', '2', '2'], False),
    (['shard', 'op-dpo2-e2', '0', '2'], False), (['shard', 'rft', '0', '2', '--x'], False), (['dev-merge', 'pert', '2'], True),
    (['dev-merge', 'sft', '2'], False), (['score'], True), (['score', 'x'], False), (['bogus'], False)])
def test_sbatch_modes_validate_arguments_before_the_allocation(tmp_path, argv, ok):
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    done = subprocess.run(['bash', str(SCRIPTS / 'isambard_016.sbatch'), str(OP.CODE), *argv], env=env, capture_output=True,
                          text=True)
    assert (done.returncode, 'Slurm allocation required' in done.stderr) == ((1, True) if ok else (2, False))
