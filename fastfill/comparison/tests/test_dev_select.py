"""Amendment 010 dev selection on CPU: fakes stand in for the models, the GPU runtime and Slurm."""
import json
from pathlib import Path
import re
import subprocess
import sys
import types
from unittest.mock import patch

import pytest

from fastfill.comparison import dev_select as ds
from fastfill.comparison import evaluate_pair
from fastfill.comparison import infer
from fastfill.comparison import infer_shard as s
from fastfill.comparison import stage_table
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_dpo_stages as D
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests.test_delivery import implementation  # noqa: F401  Autouse (audit M9).
from fastfill.comparison.tests.test_stage_table import BROKEN, build_selected, build_world

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
RUN = '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013'
DEV = [D.spaced(uid, source) for uid, source in (
    ('d0', 'OptiScene_holodeck'), ('d1', 'OptiScene_holodeck'), ('d2', 'IL3D_3dfront'), ('d3', 'IL3D_3dfront'),
    ('d4', 'Structured3D'), ('d5', 'Structured3D'), ('d6', 'SpatialGen'), ('d7', 'OptiScene_holodeck'))]
ADMITTED = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6']  # d7 is not length-admitted.


def ns(**kw):
    return types.SimpleNamespace(**kw)


@pytest.fixture
def dev(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, 'TARGETS', {'paper_sources': 3, 'other_sources': 2})
    data = D.build_main(tmp_path, dev=DEV)
    runs = D.build_sft(tmp_path, data)
    for run in runs:
        lengths = h.strict_json((run / 'lengths.json').read_text())
        h.write_json(run / 'lengths.json', {**lengths, 'dev': {'accepted_uids': ADMITTED, 'lengths': []}})
        (run / 'SHA256.json').unlink()
        h.write_json(run / 'SHA256.json', h.artifact_manifest(run))
    out = tmp_path / 'dev-select-v1'
    result = ds.build(ns(data=data, sft_runs=runs, out=out))
    return ns(root=tmp_path, data=data, runs=runs, out=out, result=result)


def test_dev_set_is_hash_ranked_length_admitted_eval_format_and_frozen(dev, monkeypatch):
    manifest = h.strict_json((dev.out / 'MANIFEST.json').read_text())
    paper = sorted(['d0', 'd1', 'd2', 'd3'], key=ds.rank)[:3]
    other = sorted(['d4', 'd5', 'd6'], key=ds.rank)[:2]
    assert manifest['expected_uids'] == sorted(paper + other) and 'd7' not in manifest['expected_uids']
    assert manifest['uids_sha256'] == ds._digest(manifest['expected_uids']) == dev.result['uids_sha256']
    assert manifest['training_manifest_sha256'] == h.sha256(dev.data / 'MANIFEST.json')
    main = {arm: h._by_uid([h.strict_json(x) for x in (dev.data / arm / 'dev.jsonl').read_text().splitlines()])
            for arm in h.ARMS}
    for arm in h.ARMS:
        requests, info = s.i.load_eval(dev.out, arm)  # The unchanged eval-data-v1 loader and request validator.
        assert info['expected_uids'] == manifest['expected_uids']
        for r in requests:
            assert r['messages'] == main[arm][r['uid']]['messages'][:2]  # Target turn dropped.
            assert (r['group'], r['component'], r['split']) == (main[arm][r['uid']]['group'],) * 2 + ('dev',)
    rooms = h._by_uid(h._rows(dev.out, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    sizes = h._by_uid(h._rows(dev.out, 'size-estimates.jsonl', manifest['files']['size-estimates.jsonl']))
    for uid, room in rooms.items():
        assert room['group'] == room['component'] == main['joint'][uid]['group'] and room['split'] == 'dev'
        request = json.loads(main['placement'][uid]['messages'][1]['content'])
        assert sizes[uid]['estimated_sizes'] == {o['id']: o['size'] for o in request['objects']}
    assert ds.pick({u: 'x' for u in ['d4', 'd5', 'd6']} | {u: 'IL3D_3dfront' for u in ['d0', 'd1', 'd2']}) == \
        ds.pick({u: 'IL3D_3dfront' for u in ['d2', 'd1', 'd0']} | {u: 'x' for u in ['d6', 'd5', 'd4']})
    with pytest.raises(FileExistsError):
        ds.build(ns(data=dev.data, sft_runs=dev.runs, out=dev.out))
    monkeypatch.setattr(ds, 'TARGETS', {'paper_sources': 3, 'other_sources': 4})
    with pytest.raises(ValueError, match='other_sources'):
        ds.build(ns(data=dev.data, sft_runs=dev.runs, out=dev.root / 'other'))


def fake(kind):
    """good: spaced valid layout; collide: every object on one spot; broken: unparsable."""
    def generate(ids, cap):
        width, out = max(map(len, ids)), []
        for x in ids:
            request = json.loads(''.join(map(chr, x)))
            answer = 'nope' if kind == 'broken' else json.dumps({'placements': [
                {'id': o['id'], **({} if 'size' in o else {'size': [0.5, 0.5, 0.5]}),
                 'pos': [0.6 + (k if kind == 'good' else 0), 0.6, 0], 'yaw': 0}
                for k, o in enumerate(request['objects'])]})
            out.append([0] * (width - len(x)) + x + [ord(c) for c in answer] + [T.EOS])
        return out, width
    return generate


def run_candidate(root, data, plan, arm, candidate, kind, training_sha):
    eval_root = root / 'dev-eval-010'
    if candidate == 'sft':
        source = ['--adapter', str(root / f'sft-{arm}-001' / arm / 'adapter'), '--training-run', str(root / f'sft-{arm}-001')]
        target, identity = 'fastfill.comparison.infer.training_identity', {
            'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': training_sha}}
    else:  # RECIPE-dpoK: the exact policy; RECIPE-dpoK-merged (amendment 011): the merged bytes as is.
        merged = candidate.endswith('-merged')
        source = ['--model', str(ds.model_path(root, arm, candidate.removesuffix('-merged')))] + ([] if merged else
                                                                                               ['--exact-policy'])
        target, identity = 'fastfill.comparison.dpo_stages.inference_identity', {
            'base': {}, 'base_sha256': 'b' * 64, 'training_data': {'manifest_sha256': training_sha},
            'stage': candidate.split('-')[1], 'lineage_sha256': '0' * 64,
            **({'model_sha256': 'f' * 64} if merged else
               {'policy': {'mode': 'x', 'training_base': str(root / 'base'), 'adapter': str(root / 'adapter')}})}
    shards = []
    for k in range(2):
        out = eval_root / candidate / arm / f'shard-{k:02d}'
        args = s.parse_args(['shard', '--data', str(data), '--base', str(root / 'base'), '--arm', arm, '--plan', str(plan),
                             '--shard-index', str(k), '--shard-count', '2', '--out', str(out), *source])
        with patch(target, return_value=identity), patch.object(s.i, '_runtime', return_value=(T.Tok(), fake(kind), T.EOS)):
            assert s.shard(args)['status'] == 'SHARD_COMPLETE'
        shards.append(out)
    merged = eval_root / candidate / f'infer-{arm}'
    s.merge(s.parse_args(['merge', '--data', str(data), '--plan', str(plan), '--arm', arm, '--out', str(merged),
                          '--shards', *map(str, shards)]))
    return merged


def plan_for(data, out):
    args = s.parse_args(['plan', '--data', str(data), '--base', str(out), '--out', str(out), '--batch-size', '2'])
    with patch.object(s, '_plan_tokenizer', return_value=(T.Tok(), {'source': 'test'}, 'b' * 64)):
        s.plan(args)
    return out


KINDS = {'sft': 'collide', 'e1-dpo1': 'good', 'e1-dpo2': 'good', 'rpo-dpo1': 'broken', 'rpo-dpo2': 'collide',
         'a009-dpo1': 'good', 'a009-dpo2': 'broken'}  # Report-only rows; a009-dpo1 would win if it were selectable.


def fake_sbatch(log, fail_on=None):
    def run(argv, capture_output, text, env):
        assert capture_output and text and not any(k.startswith(('SLURM_', 'SBATCH_')) for k in env)
        log.append(argv)
        if argv[0] == 'scontrol':
            return ns(returncode=0, stdout='', stderr='')
        if len(log) == fail_on:
            return ns(returncode=1, stdout='', stderr='sbatch: error: QOSMaxSubmitJobPerUserLimit')
        return ns(returncode=0, stdout=f'{100 + len(log)};cluster\n', stderr='')
    return run


def test_dev_scores_rule_selection_and_test_chain_submission(dev, monkeypatch):
    root, training_sha = dev.root, h.sha256(dev.data / 'MANIFEST.json')
    plan = plan_for(dev.out, root / 'dev-eval-010/plan')
    merged = {(c, a): run_candidate(root, dev.out, plan, a, c, kind, training_sha) for c, kind in KINDS.items() for a in h.ARMS}
    monkeypatch.setenv('SLURM_JOB_ID', '7')  # Must not leak into the selector's sbatch calls.
    with pytest.raises(ValueError, match='not joint/e1-dpo2'):  # Each dev inference is bound to its model.
        ds._bind(h.strict_json((merged['e1-dpo1', 'joint'] / 'run_receipt.json').read_text()), root, 'joint', 'e1-dpo2')
    with pytest.raises(ValueError, match='not joint/a009-dpo1'):  # Amendment-009 rows bind to dpo-001, not dpo-002.
        ds._bind(h.strict_json((merged['e1-dpo1', 'joint'] / 'run_receipt.json').read_text()), root, 'joint', 'a009-dpo1')
    result = ds.score(ns(data=dev.out, root=root / 'dev-eval-010', out=root / 'dev-eval-010/scores', workers=1))
    assert result['joint'] == {'sft': 0.0, 'e1-dpo1': 1.0, 'e1-dpo2': 1.0, 'rpo-dpo1': 0.0, 'rpo-dpo2': 0.0,
                               'a009-dpo1': 1.0, 'a009-dpo2': 0.0}
    scores = h.strict_json((root / 'dev-eval-010/scores/scores.json').read_text())
    joint = scores['candidates']['joint']
    assert set(joint) == set(ds.CANDIDATES) and set(scores['report_only']['joint']) == set(ds.REPORT_ONLY)
    report = scores['report_only']['joint']
    assert report['a009-dpo1']['inference']['path'] == str((root / 'dev-eval-010/a009-dpo1/infer-joint').resolve())
    assert report['a009-dpo1']['a_prime'] == 1.0 and report['a009-dpo2']['complete_rate'] == 0
    assert joint['e1-dpo1']['sft_size_log_mae_common'] == joint['sft']['size_log_mae_common_with_sft']
    assert joint['rpo-dpo1']['complete_rate'] == 0 and joint['rpo-dpo1']['size_log_mae_common_with_sft'] is None
    assert joint['e1-dpo1']['complete_rate'] == 1 and joint['e1-dpo1']['footprint_bias'] < 0  # 0.5 m vs GT 0.6 m.
    assert joint['e1-dpo1']['size_log_mae_common_with_sft'] == joint['sft']['size_log_mae_common_with_sft'] > 0
    assert scores['candidates']['placement']['e1-dpo1']['footprint_bias'] == 0  # P sizes = request sizes = GT here.
    assert joint['e1-dpo1']['inference']['stage'] == 'dpo1' and joint['sft']['inference']['stage'] == 'sft'
    # Amendment 010 addendum 2: selectable and report-only rows scored apart give the same numbers.
    for only in ('selectable', 'report-only'):
        ds.score(ns(data=dev.out, root=root / 'dev-eval-010', out=root / f'dev-eval-010/scores-{only}', workers=1, only=only))
    part = {k: h.strict_json((root / f'dev-eval-010/scores-{k}/scores.json').read_text()) for k in ('selectable', 'report-only')}
    assert part['selectable']['candidates'] == scores['candidates'] and part['selectable']['report_only'] == {'joint': {}, 'placement': {}}
    assert part['report-only']['report_only'] == scores['report_only'] and part['report-only']['only'] == 'report-only'
    assert all(part['report-only']['candidates'][a] == {'sft': scores['candidates'][a]['sft']} for a in h.ARMS)
    # The SFT test inference this run reuses, and the plan the selected DPO test shards copy.
    sft_plan = plan_for(T.make_data(root), root / 'infer-v2-002/plan')
    log = []
    selected = ds.select(ns(scores=root / 'dev-eval-010/scores', out=root / 'dev-eval-010/selection.json',
                            code_root=CODE, shards=2), run=fake_sbatch(log), which=lambda name: '/usr/bin/' + name)
    assert selected['selected'] == {'joint': 'e1-dpo1', 'placement': 'e1-dpo1'}  # Tie with e1-dpo2: fewer stages.
    # (a009-dpo1 ties too, with fewer stages than e1-dpo2, but it is report-only and never considered.)
    value = h.strict_json((root / 'dev-eval-010/selection.json').read_text())
    from fastfill.comparison import stage_table as S  # Addendum 2: the stage table binds the separate a009 rows.
    picked = S.load_selection(root / 'dev-eval-010/selection.json', root / 'dev-eval-010/scores-report-only')
    assert picked['report_only'] == scores['report_only']
    with pytest.raises(ValueError, match='same dev data'):
        S.load_selection(root / 'dev-eval-010/selection.json', root / 'dev-eval-010/scores-selectable')
    unscored = {(r['arm'], r['candidate']): r for r in S.dev_rows({**picked, 'report_only': {a: {} for a in h.ARMS}})}
    assert unscored['J', 'a009-dpo2']['a_prime'] is None and unscored['J', 'a009-dpo2']['missing']['a_prime'] == S.UNSCORED
    assert unscored['J', 'e1-dpo1']['a_prime'] == 1.0
    assert value['arms']['joint']['eligible'] == ['sft', 'e1-dpo1', 'e1-dpo2', 'rpo-dpo2']
    assert value['arms']['joint']['tied'] == ['e1-dpo1', 'e1-dpo2'] and set(value['arms']['joint']['candidates']) == set(ds.CANDIDATES)
    assert (root / 'infer-v2-005/plan/plan.json').read_bytes() == (sft_plan / 'plan.json').read_bytes()
    S, kill = str(SCRIPTS), ['--kill-on-invalid-dep=yes']
    assert log == [
        ['sbatch', '--parsable', '--hold', f'{S}/isambard_010_infer_shard.sbatch', str(CODE), 'test', 'joint', 'e1-dpo1', '0', '2'],
        ['sbatch', '--parsable', '--hold', f'{S}/isambard_010_infer_shard.sbatch', str(CODE), 'test', 'joint', 'e1-dpo1', '1', '2'],
        ['sbatch', '--parsable', *kill, '--dependency=afterok:101:102', f'{S}/isambard_infer_merge.sbatch', str(CODE), 'joint', '2',
         str(root / 'infer-v2-005'), str(root / 'infer-joint-005')],
        ['sbatch', '--parsable', '--hold', f'{S}/isambard_010_infer_shard.sbatch', str(CODE), 'test', 'placement', 'e1-dpo1', '0', '2'],
        ['sbatch', '--parsable', '--hold', f'{S}/isambard_010_infer_shard.sbatch', str(CODE), 'test', 'placement', 'e1-dpo1', '1', '2'],
        ['sbatch', '--parsable', *kill, '--dependency=afterok:104:105', f'{S}/isambard_infer_merge.sbatch', str(CODE), 'placement',
         '2', str(root / 'infer-v2-005'), str(root / 'infer-placement-005')],
        ['sbatch', '--parsable', *kill, '--dependency=afterok:103:106', f'{S}/isambard_010_cpu.sbatch', str(CODE),
         'selected-geometry', str(root / 'infer-joint-005'), str(root / 'infer-placement-005')],
        ['scontrol', 'release', '101,102,104,105']]
    assert [j['id'] for j in h.strict_json((root / 'dev-eval-010/submission.json').read_text())['jobs']] == [
        '101', '102', '103', '104', '105', '106', '107']
    with pytest.raises(FileExistsError, match='already submitted'):
        ds.select(ns(scores=root / 'dev-eval-010/scores', out=root / 'dev-eval-010/selection.json', code_root=CODE, shards=2),
                  run=fake_sbatch([]), which=lambda name: name)
    # The selected-model geometry pairs models of different stages/LoRA loading only under the 'selected' label.
    runs = {'joint': merged['sft', 'joint'], 'placement': merged['e1-dpo1', 'placement']}
    with pytest.raises(ValueError, match='model stage mismatch'):
        evaluate_pair.evaluate(dev.out, runs, root / 'geometry-x', repeats=20)
    summary = evaluate_pair.evaluate(dev.out, runs, root / 'geometry-v5', repeats=20, stage_label='selected')
    assert summary['stage'] == 'selected' and summary['stages'] == {'joint': 'sft', 'placement': 'dpo1'}


@pytest.mark.parametrize('kind, listed', [('good', True), ('collide', False)])
def test_merged_weight_check_scores_merged_and_unmerged_like_score(dev, kind, listed, monkeypatch):
    """Amendment 011 section 5: the delivered P (rpo-dpo1) BF16 merge vs its exact policy on the same dev plan."""
    root, training_sha = dev.root, h.sha256(dev.data / 'MANIFEST.json')
    plan = plan_for(dev.out, root / 'dev-eval-010/plan')
    made = {c: 'good' if c == 'rpo-dpo1' else 'collide' for c in ds.CANDIDATES}  # rpo-dpo1 is selected in both arms.
    for arm in h.ARMS:
        for candidate in ds.CANDIDATES:
            run_candidate(root, dev.out, plan, arm, candidate, made[candidate], training_sha)
    run_candidate(root, dev.out, plan, 'placement', 'rpo-dpo1-merged', kind, training_sha)
    def receipt(candidate):
        return h.strict_json((root / 'dev-eval-010' / candidate / 'infer-placement/run_receipt.json').read_text())
    assert receipt('rpo-dpo1-merged')['config']['lora'] == ds.PREMERGED and receipt('rpo-dpo1')['config']['lora'] == ds.EXACT
    with pytest.raises(ValueError, match='not placement/rpo-dpo1-merged'):  # The exact policy is not the merged bytes,
        ds._bind(receipt('rpo-dpo1'), root, 'placement', 'rpo-dpo1-merged')
    with pytest.raises(ValueError, match='not placement/rpo-dpo1$'):  # and the merged bytes never stand in for it.
        ds._bind(receipt('rpo-dpo1-merged'), root, 'placement', 'rpo-dpo1')
    with pytest.raises(ValueError, match='not joint/rpo-dpo1-merged'):
        ds._bind(receipt('rpo-dpo1-merged'), root, 'joint', 'rpo-dpo1-merged')
    selection, out = root / 'dev-eval-010/selection.json', root / 'dev-eval-010/merged-check'
    args = ds.parse_args(['check-merged', '--data', str(dev.out), '--root', str(root / 'dev-eval-010'), '--out', str(out)])
    ds.score(ns(data=dev.out, root=root / 'dev-eval-010', out=root / 'dev-eval-010/scores', workers=1, only='selectable'))
    selection.write_text('{"selected": {"joint": "sft", "placement": "rpo-dpo1"}}\n')
    with pytest.raises(ValueError, match='not bound to its dev scores'):  # A selection not reproduced from its scores.
        ds.check_merged(args)
    selection.unlink()
    assert ds.select(ns(scores=root / 'dev-eval-010/scores', out=selection, code_root=CODE, shards=2, no_submit=True))[
        'selected'] == {'joint': 'rpo-dpo1', 'placement': 'rpo-dpo1'}
    with monkeypatch.context() as m:
        m.setattr(ds, 'MERGED', ('placement', 'e1-dpo1'))
        with pytest.raises(ValueError, match='not e1-dpo1'):
            ds.check_merged(args)
    with monkeypatch.context() as m:  # Rescoring drift: the unmerged side no longer equals its dev score of record.
        finished = ds._finished
        m.setattr(ds, '_finished', lambda *a: (lambda v, f: ({**v, 'a_prime': v['a_prime'] - .5}, f))(*finished(*a)))
        with pytest.raises(ValueError, match='rescoring differs from its dev score'):
            ds.check_merged(args)
    assert not out.exists()
    result = ds.check_merged(args)
    files = ds.d._verified_dir(out)
    check = h.strict_json((out / 'check.json').read_text())
    assert set(files) == {'check.json', 'placement-rpo-dpo1-geometry.jsonl', 'placement-rpo-dpo1-merged-geometry.jsonl'}
    unmerged, merged = check['unmerged'], check['merged']
    rooms = dev.result['rooms']
    assert check['rooms'] == merged['requests'] == unmerged['requests'] == rooms
    assert unmerged['complete_rate'] == unmerged['a_prime'] == 1 and unmerged['footprint_bias'] == 0  # P sizes = GT.
    assert unmerged['common_with_sft'] == rooms and unmerged['inference']['lora'] == ds.EXACT
    assert merged['inference']['lora'] == ds.PREMERGED and merged['inference']['stage'] == 'dpo1'
    assert merged['model'] == {'path': str(ds.model_path(root, 'placement', 'rpo-dpo1').resolve()),
                               'model_sha256': 'f' * 64, 'lineage_sha256': '0' * 64}
    assert check['sft_inference']['path'] == str((root / 'dev-eval-010/sft/infer-placement').resolve())
    assert check['selection_sha256'] == h.sha256(selection) and check['thresholds'] == {'a_prime': .01, 'complete_rate': .005}
    assert check['merged_listed_as_convenience'] is listed and result['decision'] == check['decision'] == (
        'merged_convenience_artifact' if listed else 'adapter_and_training_base_only')
    if listed:  # Same weights' outputs: identical text, zero deltas.
        assert check['byte_identical'] == {'rooms': rooms, 'fraction': 1.0}
        assert check['delta_merged_minus_unmerged'] == {'complete_rate': 0, 'geometry_success_rate': 0, 'a_prime': 0}
    else:  # Every object on one spot: still complete, never a geometry success.
        assert check['byte_identical']['fraction'] < 1 and merged['complete_rate'] == 1 and merged['a_prime'] == 0
        assert check['delta_merged_minus_unmerged']['a_prime'] == -1
    with pytest.raises(FileExistsError):
        ds.check_merged(args)


def test_selection_without_sbatch_prints_the_exact_commands_and_fails(capsys):
    jobs = ds.chain_jobs(CODE, Path(RUN), {'joint': 'sft', 'placement': 'rpo-dpo2'}, 6)
    assert [j['label'] for j in jobs] == [f'test shard placement {k}' for k in range(6)] + ['test merge placement',
                                                                                          'selected geometry']
    assert jobs[-1]['argv'][-2:] == [f'{RUN}/infer-joint-002', f'{RUN}/infer-placement-005']  # SFT: measured run reused.
    with pytest.raises(SystemExit, match='sbatch/scontrol unavailable'):
        ds.submit(jobs, run=None, which=lambda name: None)
    printed = capsys.readouterr().err.splitlines()
    assert printed[1] == (f"J0=$(sbatch --parsable --hold {SCRIPTS}/isambard_010_infer_shard.sbatch {CODE} test placement "
                          "rpo-dpo2 0 6 | cut -d';' -f1)  # test shard placement 0")
    assert printed[7].startswith('J6=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$J0:$J1:$J2:$J3:$J4:$J5 ')
    assert printed[8].startswith('J7=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$J6 ')
    assert printed[-1] == 'scontrol release $J0,$J1,$J2,$J3,$J4,$J5'
    only_geometry = ds.chain_jobs(CODE, Path(RUN), {'joint': 'sft', 'placement': 'sft'}, 6)
    assert [(j['label'], j['after']) for j in only_geometry] == [('selected geometry', [])]
    log = []
    with pytest.raises(SystemExit, match=r"failed for 'test shard placement 2'.*scancel 101 102"):
        ds.submit(jobs, run=fake_sbatch(log, fail_on=3), which=lambda name: name)
    assert len(log) == 3


def m(complete, mae, a, sft_mae=.30):
    return {'complete_rate': complete, 'size_log_mae_common_with_sft': mae, 'sft_size_log_mae_common': sft_mae,
            'a_prime': a}


def test_rule_eligibility_margins_ties_and_sft_fallback():
    base = {'sft': m(.90, .30, .40), 'e1-dpo1': m(.89, .31, .50), 'e1-dpo2': m(.95, .20, .60),
            'rpo-dpo1': m(.889, .30, .90), 'rpo-dpo2': m(.95, .3101, .90)}
    chosen = ds.choose(base)  # Bounds are inclusive; rpo-* miss them by a hair and are not eligible.
    assert chosen == {'selected': 'e1-dpo2', 'eligible': ['sft', 'e1-dpo1', 'e1-dpo2'], 'tied': ['e1-dpo2'],
                      'best_a_prime': .60}
    assert ds.choose({**base, 'e1-dpo1': m(.9, .3, .596)})['selected'] == 'e1-dpo1'  # Within 0.005: fewer stages.
    assert ds.choose({**base, 'e1-dpo1': m(.9, .3, .594)})['selected'] == 'e1-dpo2'
    assert ds.choose({**base, 'sft': m(.90, .30, .596)})['selected'] == 'sft'  # SFT wins every tie.
    tie = {**base, 'e1-dpo2': m(.9, .3, .5), 'rpo-dpo1': m(.9, .3, .503), 'e1-dpo1': m(.9, .3, .5)}
    assert ds.choose(tie)['selected'] == 'rpo-dpo1'  # Same stage count: higher a'.
    worse = {c: m(.5, 1., .99) for c in ds.CANDIDATES if c != 'sft'}
    assert ds.choose({'sft': m(.9, .3, .1), **worse}) == {'selected': 'sft', 'eligible': ['sft'], 'tied': ['sft'],
                                                         'best_a_prime': .1}
    no_sft_mae = {c: m(.9, None, .9, sft_mae=None) for c in ds.CANDIDATES}  # As metrics() gives for an SFT that
    no_sft_mae['sft'] = m(0., None, 0., sft_mae=None)  # is never complete: no common rooms, nothing compared, SFT stays.
    assert ds.choose(no_sft_mae)['selected'] == 'sft'
    with pytest.raises(ValueError, match='every candidate'):
        ds.choose({'sft': base['sft']})
    # The MAE bound is SFT's MAE on the candidate's common rooms, not SFT's MAE on all its own complete rooms.
    assert ds.choose({**base, 'e1-dpo2': m(.95, .20, .60, sft_mae=.18)})['selected'] == 'e1-dpo1'


def row(complete, mae):
    return {'complete': complete, 'size_log_mae': mae if complete else None, 'geometry_success': complete,
            'bias_fp': 0.0 if complete else None}


def test_mae_eligibility_compares_the_same_rooms():
    """Review regression: both complete 90/100; the candidate is 0.03 worse on all 81 shared rooms and fails
    the 9 hard rooms SFT completes (MAE 0.60), completing 9 SFT failures instead. Its common MAE (0.28) is
    below SFT's own-room MAE + 0.01 (0.295) but above SFT's MAE on the same rooms + 0.01 (0.26)."""
    sft = [row(True, .25)] * 81 + [row(True, .60)] * 9 + [row(False, 0)] * 10
    worse = [row(True, .28)] * 81 + [row(False, 0)] * 9 + [row(True, .40)] * 9 + [row(False, 0)]
    better = [row(True, .24)] * 81 + [row(False, 0)] * 9 + [row(True, .40)] * 9 + [row(False, 0)]
    ref = ds.metrics(sft, sft)
    assert ref['size_log_mae_common_with_sft'] == ref['sft_size_log_mae_common'] == pytest.approx(.285)
    for rows, eligible in ((worse, False), (better, True)):
        value = ds.metrics(rows, sft)
        assert value['complete_rate'] == ref['complete_rate'] == .9 and value['common_with_sft'] == 81
        assert value['sft_size_log_mae_common'] == pytest.approx(.25)
        scored = {c: {**ref, 'a_prime': .1} for c in ds.CANDIDATES}
        chosen = ds.choose({**scored, 'e1-dpo1': {**value, 'a_prime': .9}})
        assert ('e1-dpo1' in chosen['eligible']) is eligible and chosen['selected'] == ('e1-dpo1' if eligible else 'sft')


def test_submit_chain_dependencies_with_stub_sbatch(tmp_path):
    for name in ('isambard_010_cpu.sbatch', 'isambard_010_infer_shard.sbatch', 'isambard_dpo_train.sbatch',
                 'submit_dev_selection.sh'):
        subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    env = T.stub_sbatch(tmp_path)
    run = subprocess.run(['bash', str(SCRIPTS / 'submit_dev_selection.sh'), str(CODE), '1'],
                         env=env, capture_output=True, text=True, check=True)
    log = [line.replace(f'{SCRIPTS}/', '').replace(f' {CODE} ', ' CODE ') for line in (tmp_path / 'log').read_text().splitlines()]
    kill = '--parsable --kill-on-invalid-dep=yes --dependency=afterok:'
    trains = [f'{kill}101 isambard_dpo_train.sbatch CODE {arm} dpo1 {c}' for c in ('e1', 'rpo') for arm in h.ARMS]
    assert log[:2] == ['--parsable --hold isambard_010_cpu.sbatch CODE build-dev',
                       f'{kill}101 isambard_010_cpu.sbatch CODE dev-plan']
    assert log[2:10:2] == trains and log[3] == f'{kill}103 isambard_dpo_train.sbatch CODE joint dpo2 e1'
    assert log[10:14] == [f'{kill}102 isambard_010_infer_shard.sbatch CODE dev joint sft 0 1',
                          f'{kill}111 isambard_010_cpu.sbatch CODE dev-merge joint sft 1',
                          f'{kill}102 isambard_010_infer_shard.sbatch CODE dev placement sft 0 1',
                          f'{kill}113 isambard_010_cpu.sbatch CODE dev-merge placement sft 1']
    after = {'e1-dpo1': (103, 105), 'e1-dpo2': (104, 106), 'rpo-dpo1': (107, 109), 'rpo-dpo2': (108, 110)}
    for k, (cand, jobs) in enumerate(after.items()):
        for j, arm in enumerate(h.ARMS):
            assert log[14 + 4 * k + 2 * j] == f'{kill}102:{jobs[j]} isambard_010_infer_shard.sbatch CODE dev {arm} {cand} 0 1'
    for k, cand in enumerate(ds.REPORT_ONLY):  # Report-only amendment-009 stages: models exist, only the plan.
        for j, arm in enumerate(h.ARMS):
            assert log[30 + 4 * k + 2 * j:32 + 4 * k + 2 * j] == [
                f'{kill}102 isambard_010_infer_shard.sbatch CODE dev {arm} {cand} 0 1',
                f'{kill}{131 + 4 * k + 2 * j} isambard_010_cpu.sbatch CODE dev-merge {arm} {cand} 1']
    merges = ':'.join(str(112 + 2 * k) for k in range(14))
    assert log[38:] == [f'{kill}{merges} isambard_010_cpu.sbatch CODE score', f'{kill}139 isambard_010_cpu.sbatch CODE select']
    assert run.stdout.splitlines()[-1] == 'released build-dev 101'
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101\n'
    failed = tmp_path / 'failed'
    failed.mkdir()
    bad = subprocess.run(['bash', str(SCRIPTS / 'submit_dev_selection.sh'), str(CODE)], env=T.stub_sbatch(failed, fail_on=3),
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "sbatch failed for 'train e1 joint dpo1'" in bad.stderr and 'scancel 101 102' in bad.stderr
    assert not (failed / 'log.scontrol').exists()


def test_final_reports_tables_and_delivers_with_complete_agent_evaluations_only(tmp_path, monkeypatch, capsys):
    run = build_selected(build_world(tmp_path), broken=BROKEN)  # J e1-dpo2 (3 broken rooms), P sft -> delivers P.
    monkeypatch.setattr(ds, 'STAGE_REPORTS', {'sft': 'report-sft', 'dpo1': 'report-dpo1'})
    (run / 'corrections.json').write_text('["il-bed-2"]\n')
    unfinished = run / 'llm-eval-O6-selected-v1'
    unfinished.mkdir()  # An evaluation that died before publishing: no MANIFEST.
    args = ds.parse_args(['final', '--run-root', str(run), '--reference', str(run / 'reference.json'), '--corrections',
                          str(run / 'corrections.json'), '--agent', f'O6={unfinished}', '--agent', f'O4={run / "absent"}',
                          '--repeats', '200'])
    result = ds.final(args)
    assert (result['delivered'], result['candidate'], result['agents']) == ('P', 'sft', [])
    assert set(result['agents_excluded']) == {'O6', 'O4'} and 'FileNotFoundError' in result['agents_excluded']['O6']
    assert json.loads(capsys.readouterr().out.splitlines()[0])['agents_excluded'] == result['agents_excluded']
    report = h.strict_json((run / ds.FINAL['report'] / 'results.json').read_text())
    assert report['stage'] == 'selected' and report['delivery']['binding'] and report['agents'] == {}
    assert report['input_sha256']['corrections'] == h.sha256(run / 'corrections.json')
    table = h.strict_json((run / ds.FINAL['table'] / 'stage_table.json').read_text())
    assert table['stages_run'] == ['sft', 'dpo1', 'selected'] and table['not_run'] == {'dpo2': stage_table.STOPPED}
    package = h.strict_json((run / ds.FINAL['delivery'] / 'delivery.json').read_text())
    assert package['candidate'] == 'sft' and {k: package['agents'][k] for k in ('included', 'excluded')} == {
        'included': [], 'excluded': result['agents_excluded']}
    card = (run / ds.FINAL['delivery'] / 'MODEL_CARD.md').read_text()
    assert 'Excluded (evaluation incomplete; never substituted): O4 (' in card and "PeftModel.from_pretrained(base, 'sft-adapter')" in card
    assert {r['candidate'] for r in table['tables']['dev']} == set(ds.CANDIDATES + ds.REPORT_ONLY)
    with pytest.raises(FileExistsError, match='refusing existing outputs'):
        ds.final(args)
    other = tmp_path / 'other'
    (other / 'dev-eval-010').mkdir(parents=True)
    (other / 'dev-eval-010' / 'selection.json').write_text('{"test": {}}\n')
    with pytest.raises(ValueError, match='unique NAME=DIR'):
        ds.final(ds.parse_args(['final', '--run-root', str(other), '--reference', 'r', '--corrections', 'c',
                                '--agent', 'O-6=x']))


def final_world(tmp_path, geometry_published=False):
    """The submit script with PROJECT_ROOT moved under tmp_path, a stub sbatch and the select job's receipts."""
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    (run / 'dev-eval-010').mkdir(parents=True)
    (project / 'env/bin').mkdir(parents=True)
    (project / 'env/bin/python').symlink_to(sys.executable)
    script = tmp_path / 'submit_final_010.sh'
    original = (SCRIPTS / 'submit_final_010.sh').read_text()
    script.write_text(original.replace(f'PROJECT_ROOT={RUN.removesuffix("/runs/unknownsize-20261013")}\n',
                                       f'PROJECT_ROOT={project}\n', 1))
    assert script.read_text() != original
    infer.write_once(run / 'dev-eval-010/selection.json', {'test': {'joint': {'inference': f'{run}/infer-joint-005'},
                                                                    'placement': {'inference': f'{run}/infer-placement-002'}}})
    h.write_json(run / 'dev-eval-010/submission.json', {'selection_sha256': h.sha256(run / 'dev-eval-010/selection.json'),
                 'jobs': [{'label': 'test merge joint', 'id': '201'}, {'label': 'selected geometry', 'id': '207'}]})
    for name in ('llm-agent-001', 'llm-open-O6-001') + (('geometry-v5',) if geometry_published else ()):
        (run / name).mkdir()
        (run / name / 'MANIFEST.json').write_text('{}\n')
    return script, run


def test_submit_final_chain_with_stub_sbatch(tmp_path):
    subprocess.run(['bash', '-n', str(SCRIPTS / 'submit_final_010.sh')], check=True)
    script, run = final_world(tmp_path)
    env = T.stub_sbatch(tmp_path)
    done = subprocess.run(['bash', str(script), str(CODE), f'L={run}/llm-agent-001', f'O6={run}/llm-open-O6-001',
                           f'O3={run}/llm-open-O3-001:7394845'], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    log = (tmp_path / 'log').read_text().splitlines()
    cpu = ('--account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G '
           '--gres=none --no-requeue --kill-on-invalid-dep=yes')
    project = run.parents[1]
    prefix = f'cd {CODE} && PYTHONPATH={CODE} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {project}/env/bin/python -B -m'
    for k, (name, llm, after) in enumerate((('L', 'llm-agent-001', '207'), ('O6', 'llm-open-O6-001', '207'),
                                            ('O3', 'llm-open-O3-001', '207:7394845'))):
        assert log[k] == (
            f'--parsable --hold --dependency=afterok:{after} {cpu} --time=04:00:00 --job-name=unknownsize-010-eval '
            f'--output={project}/runs/isambard/unknownsize-010-eval-{name}-%j.out --wrap {prefix} fastfill.comparison.evaluate_llm '
            f'--data {run}/eval-data-v1 --cohort {run}/llm-cohort-v1.json --llm {run}/{llm} --joint {run}/infer-joint-005 '
            f'--placement {run}/infer-placement-002 --out {run}/llm-eval-{name}-selected-v1 --workers 16')
    evidence = f'{CODE}/docs/research/20261013-unknown-size-comparison/evidence'
    assert log[3] == (
        f'--parsable --hold --dependency=afterany:101:102:103,afterok:207 {cpu} --time=12:00:00 --job-name=unknownsize-010-final '
        f'--output={project}/runs/isambard/unknownsize-010-final-%j.out --wrap {prefix} fastfill.comparison.dev_select final '
        f'--run-root {run} --reference {evidence}/amendment-005-reference-rows-20261013.json '
        f'--corrections {evidence}/amendment-006-corrections-uids-candidate-003.json --agent L={run}/llm-eval-L-selected-v1 '
        f'--agent O6={run}/llm-eval-O6-selected-v1 --agent O3={run}/llm-eval-O3-selected-v1 --repeats 2000 --workers 16')
    assert len(log) == 4 and (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101,102,103,104\n'
    assert done.stdout.splitlines()[-1] == 'released 101 102 103 104'


@pytest.mark.parametrize('case, match', [
    ('published', None), ('unfinished agent', 'O3 generation not complete'), ('existing output', 'refusing to reuse'),
    ('sbatch fails', "sbatch failed for 'eval O6'.*scancel 101"), ('unbound selection', 'cannot read'),
    ('no submission', 'set GEOMETRY_JOB'), ('override', None)])
def test_submit_final_refusals(tmp_path, case, match):
    script, run = final_world(tmp_path, geometry_published=case == 'published')
    agents = [f'L={run}/llm-agent-001', f'O6={run}/llm-open-O6-001']
    if case == 'unfinished agent':
        agents.append(f'O3={run}/llm-open-O3-001')  # Generation still running: needs :JOBID.
    if case == 'existing output':
        (run / 'llm-eval-O6-selected-v1').mkdir()
    if case == 'unbound selection':
        (run / 'dev-eval-010/selection.json').write_text('{"test": {}}\n')
    if case in ('no submission', 'override'):  # The select job could not sbatch; the chain was submitted by hand.
        (run / 'dev-eval-010/submission.json').unlink()
    env = T.stub_sbatch(tmp_path, fail_on=2 if case == 'sbatch fails' else 0)
    done = subprocess.run(['bash', str(script), str(CODE), *agents], env={**env, **({'GEOMETRY_JOB': '999'} if case == 'override' else {})},
                          capture_output=True, text=True)
    if case == 'override':
        assert done.returncode == 0, done.stderr
        assert '--dependency=afterok:999 ' in (tmp_path / 'log').read_text().splitlines()[0]
        return
    if match is None:  # geometry-v5 already published: the evaluations wait for nothing, the final job only for them.
        log = (tmp_path / 'log').read_text().splitlines()
        assert done.returncode == 0 and log[0].startswith('--parsable --hold --account=') and \
            log[2].startswith('--parsable --hold --dependency=afterany:101:102 --account=')
        return
    assert done.returncode == 1 and re.search(match, done.stderr, re.S)
    assert not (tmp_path / 'log.scontrol').exists() and (case == 'sbatch fails') == (tmp_path / 'log').exists()


@pytest.mark.parametrize('script, argv, ok', [
    ('isambard_010_infer_shard.sbatch', ['dev', 'joint', 'sft', '0', '2'], True),
    ('isambard_010_infer_shard.sbatch', ['test', 'joint', 'rpo-dpo2', '5', '6'], True),
    ('isambard_010_infer_shard.sbatch', ['test', 'joint', 'sft', '0', '6'], False),  # SFT test output is reused.
    ('isambard_010_infer_shard.sbatch', ['dev', 'joint', 'dpo2', '0', '2'], False),
    ('isambard_010_infer_shard.sbatch', ['dev', 'placement', 'a009-dpo2', '1', '2'], True),
    ('isambard_010_infer_shard.sbatch', ['test', 'placement', 'a009-dpo2', '0', '6'], False),  # Dev-only report row.
    ('isambard_010_cpu.sbatch', ['dev-merge', 'placement', 'a009-dpo1', '2'], True),
    ('isambard_010_cpu.sbatch', ['score-report-only'], True),
    ('isambard_dpo_train.sbatch', ['joint', 'dpo1', 'rpo'], True),
    ('isambard_dpo_train.sbatch', ['joint', 'dpo1', 'official'], False),
    ('isambard_010_cpu.sbatch', ['selected-geometry', f'{RUN}/infer-joint-002', f'{RUN}/infer-placement-005'], True),
    ('isambard_010_cpu.sbatch', ['selected-geometry', f'{RUN}/infer-placement-005', f'{RUN}/infer-joint-002'], False),
    ('isambard_010_cpu.sbatch', ['dev-merge', 'joint', 'e1-dpo3', '2'], False),
    ('isambard_010_infer_shard.sbatch', ['dev', 'placement', 'rpo-dpo1-merged', '1', '2'], True),  # Amendment 011.
    ('isambard_010_infer_shard.sbatch', ['dev', 'joint', 'rpo-dpo1-merged', '0', '2'], False),  # P is delivered.
    ('isambard_010_infer_shard.sbatch', ['test', 'placement', 'rpo-dpo1-merged', '0', '6'], False),  # Dev only.
    ('isambard_010_infer_shard.sbatch', ['dev', 'placement', 'e1-dpo1-merged', '0', '2'], False),
    ('isambard_010_cpu.sbatch', ['dev-merge', 'placement', 'rpo-dpo1-merged', '2'], True),
    ('isambard_010_cpu.sbatch', ['dev-merge', 'joint', 'rpo-dpo1-merged', '2'], False),
    ('isambard_010_cpu.sbatch', ['check-merged'], True)])
def test_sbatch_argument_checks(tmp_path, script, argv, ok):
    run = subprocess.run(['bash', str(SCRIPTS / script), str(CODE), *argv], env=T.stub_sbatch(tmp_path),
                         capture_output=True, text=True)
    # Valid arguments pass every check and stop at the missing Slurm allocation; invalid ones exit 2 before it.
    assert (run.returncode, 'Slurm allocation required' in run.stderr) == ((1, True) if ok else (2, False))
