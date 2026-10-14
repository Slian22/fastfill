"""Stage ablation over real report_v2 outputs of a fixture DPO chain (no network, no jobs)."""
import argparse
import hashlib
import json
import shutil

import pytest

from fastfill.comparison import dev_select as ds
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import infer as i
from fastfill.comparison import reference_rows as RR
from fastfill.comparison import report_v2 as R
from fastfill.comparison import stage_table as S
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests.test_llm_agent import eval_data, target, write_rows

DECODE = {'max_new_tokens': 4096, 'max_input_tokens': 36864, 'context_budget': 40960, 'batch_size': 8,
          'logits_to_keep': 1, 'do_sample': False, 'num_beams': 1, 'dtype': 'bfloat16', 'attention': 'sdpa',
          'cap_base': 1024, 'cap_per_object': 128, 'enable_thinking': False, 'padding_side': 'left'}
BASE = {'revision': h.REVISION, 'path': '/base', 'files': {'model.safetensors': '1' * 64}}
TRAINING = {'manifest_sha256': 'a' * 64, 'files': {'train.jsonl': '2' * 64}}
BROKEN = {'x-06': '{"placements": [', 'y-0': '{}', 'x-07': 'nonsense'}  # Three J-DPO2 failures.
SELECTED = {'joint': 'e1-dpo2', 'placement': 'sft'}  # Amendment 010 fixture: one DPO candidate, one SFT.
PAIRS = {'dpo1': 2000, 'dpo2': 2150}
DIAGNOSTIC = {'modules': {'q_proj': {'changed': 3, 'total': 10, 'fraction': .3},
                          'other': {'changed': 0, 'total': 10, 'fraction': 0.}},
              'dtypes': ['torch.bfloat16'], 'tensors_only_in_parent': 0, 'tensors_only_in_merged': 0}


def _hex(*parts):
    return hashlib.sha256('/'.join(parts).encode()).hexdigest()


def _lineage(run, arm, recipe=None):
    """dpo-001's sft -> dpo1 -> dpo2 receipts, or (recipe, amendment 010) the candidate's dpo1 -> dpo2 in its own
    run dir on dpo-001's SFT merge, as dpo_stages.train --candidate writes them."""
    name, links = d.CANDIDATES[recipe][0] if recipe else 'dpo-001', {}
    root, tag = run / name / arm, (recipe,) if recipe else ()
    root.mkdir(parents=True)
    for k, stage in enumerate(S.STAGES):
        if recipe and stage == 'sft':
            links[stage] = json.loads((run / 'dpo-001' / arm / 'sft-merged.json').read_text())
            continue
        files = {'config.json': _hex(*tag, arm, stage, 'config'), 'model.safetensors': _hex(*tag, arm, stage, 'weights')}
        receipt = {'protocol': d.MERGE_PROTOCOL, 'arm': arm, 'stage': stage, 'base': BASE,
                   'base_sha256': i._hash(BASE['files']), 'training_data': TRAINING,
                   'adapter': {'path': str(root / stage / 'final'), 'files': {'adapter_model.safetensors': _hex(*tag, arm, stage)}},
                   'model': {'path': str(root / f'{stage}-merged'), 'files': files}, 'model_sha256': i._hash(files),
                   'merge': 'fastfill.merge_lora: BF16 PeftModel.merge_and_unload', 'merge_diagnostic': DIAGNOSTIC}
        if stage == 'sft':
            receipt.update(parent={'stage': 'generic_base', 'path': '/base'},
                           training={'run': '/sft', 'training_run_sha256': _hex(arm, 'run'),
                                     'recipe': 'SFT as trained: LoRA r32 a64, lr 5e-5, 1 epoch'})
        else:
            previous, pairs = S.STAGES[k - 1], run / 'dpo-001' / d.PAIRS[stage]
            parent = run / ('dpo-001' if previous == 'sft' else name) / arm / f'{previous}-merged.json'
            receipt.update(parent={'stage': previous, 'path': links[previous]['model']['path'], 'lineage_sha256': h.sha256(parent)},
                           training={'pairs': {'path': str(pairs / f'{arm}.jsonl'), 'count': PAIRS[stage],
                                               'manifest_sha256': h.sha256(pairs / 'MANIFEST.json')},
                                     'completion': {'steps': 10 * PAIRS[stage] // 4, 'admitted_pairs': PAIRS[stage]},
                                     'recipe': {**(d.CANDIDATES[recipe][1] if recipe else d.RECIPE), **d.TRAIN_FIXED},
                                     **({'candidate': recipe} if recipe else {})})
        h.write_json(root / f'{stage}-merged.json', receipt)
        links[stage] = receipt
    return links


def _inference(run, data, rooms, arm, stage, links, raw=None, out=None, root='dpo-001'):
    out = run / (out or f'infer-{arm}-{S.TAG[stage]}')
    out.mkdir()
    (out / 'context_plan.json').write_text('{"plan": "shared"}\n')
    link, manifest = links[stage], json.loads((data / 'MANIFEST.json').read_text())
    sft_run = (run / f'sft-{arm}-001').resolve()
    source = ({'training_run': str(sft_run), 'adapter': str(sft_run / arm / 'adapter')} if stage == 'sft'
              else {'model': str((run / root / arm / f'{stage}-merged').resolve())})  # As dev_select._bind reads them.
    receipt = {'data': {'manifest_sha256': h.sha256(data / 'MANIFEST.json')}, 'training_data': TRAINING, 'base': BASE,
               'base_sha256': link['base_sha256'], 'context_plan_sha256': h.sha256(out / 'context_plan.json'),
               'config': {'arm': arm, **DECODE, **source, 'base': '/base', 'lora': S.LORA['sft' if stage == 'sft' else 'dpo']}}
    adapter = i._hash(link['adapter']['files'])
    if stage == 'sft':
        receipt.update(training_run_sha256=link['training']['training_run_sha256'], adapter_sha256=adapter,
                       adapter_files=link['adapter']['files'])
    else:
        parent = links[S.STAGES[S.STAGES.index(stage) - 1]]
        receipt.update(stage=stage, lineage_sha256=h.sha256(run / root / arm / f'{stage}-merged.json'),
                       policy={'mode': 'unmerged_fp32_adapter_on_bf16_training_base', 'training_base': parent['model']['path'],
                               'training_base_sha256': parent['model_sha256'], 'adapter': link['adapter']['path'],
                               'adapter_sha256': adapter})
    receipt['run_sha256'] = i._hash(receipt)
    h.write_json(out / 'run_receipt.json', receipt)
    write_rows(out / 'records.jsonl', [
        {'uid': u, 'source': rooms[u]['source'], 'group': rooms[u]['group'], 'arm': arm,
         'raw': (raw or {}).get(u, target(None, arm, rooms[u])), 'error': None, 'reached_token_cap': False,
         'stopped_on_eos': True, 'generated_tokens': 50, 'input_tokens': 400, 'batch_id': k // 8,
         'duration_seconds': 36., 'run_sha256': receipt['run_sha256']} for k, u in enumerate(manifest['expected_uids'])])
    h.write_json(out / 'result.json', {'status': 'INFERENCE_COMPLETE', 'pending_count': 0})
    h.write_json(out / 'MANIFEST.json', {'expected_uids': manifest['expected_uids'], 'files': h.artifact_manifest(out)})
    return out


def build_world(root, broken_dpo2=True):
    """Run root as the DPO chain leaves it: eval data, prior, reference, pairs, lineage, 3 stages x 2 arms, reports."""
    run = root / 'run'
    data = eval_data(run / 'eval-data-v1')
    prior = run / 'main-data-v1' / 'size-priors.json'
    prior.parent.mkdir(parents=True)
    prior.write_text('{"full_prior": {"min_count": 5}}\n')
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    h.write_json(data / 'MANIFEST.json', {**manifest, 'input_consumed_sha256': {'training': {'size-priors.json': h.sha256(prior)}}})
    rows = [json.loads(line) for line in (data / 'rooms.jsonl').read_text().splitlines()]
    rooms = {r['uid']: r for r in rows}
    requests = {arm: {r['uid']: r for r in map(json.loads, (data / arm / 'requests.jsonl').read_text().splitlines())}
                for arm in h.ARMS}
    i.write_once(run / 'reference.json', RR.build_reference(rows, requests, h.sha256(data / 'MANIFEST.json'), ['il-bed-2']))
    for stage, name in d.PAIRS.items():
        summary = {'positives': 2200, 'kept_both_arms': PAIRS[stage], 'verified_per_arm': {a: PAIRS[stage] + 5 for a in h.ARMS},
                   'dropped_by_other_arm_only': {a: 5 for a in h.ARMS}, 'kinds': {},
                   'kept_by_source': {'IL3D_3dfront': PAIRS[stage] // 2, 'OptiScene_holodeck': PAIRS[stage] // 2}}
        d._publish_dir(run / 'dpo-001' / name, {'summary.json': json.dumps(summary).encode(), 'joint.jsonl': b'',
                                                'placement.jsonl': b''})
    links = {arm: _lineage(run, arm) for arm in h.ARMS}
    for stage in S.STAGES:
        broken = BROKEN if broken_dpo2 and stage == 'dpo2' else None
        runs = {arm: _inference(run, data, rooms, arm, stage, links[arm], broken if arm == 'joint' else None)
                for arm in h.ARMS}
        e.evaluate(data, runs, run / S.GEOMETRY[stage], repeats=200)
        R.build(data, run / S.GEOMETRY[stage], runs, run / 'reference.json', run / f'report-{stage}', repeats=200)
    return run


def dev_metrics(chosen, complete=.9):
    """dev_select.metrics-shaped values; `chosen` has the best a' and every DPO candidate but it is ineligible."""
    return {'requests': 600, 'complete_rate': complete, 'geometry_success_rate': .3, 'size_log_mae_common_with_sft': .25,
            'sft_size_log_mae_common': .25, 'common_with_sft': 500, 'size_log_mae_own': .25, 'footprint_bias': -.05,
            'scale_factor': 1.025, 'a_prime': .6 if chosen else .4, 'token_cap_failures': 0}


def build_selected(run, selected=SELECTED, broken=None):
    """Amendment 010 on top of build_world: e1 candidate lineage (dpo-002), dev scores and selection.json written by
    dev_select.select, the selected DPO test runs (infer-ARM-005; SFT reuses infer-ARM-002), geometry-v5, report."""
    data = run / 'eval-data-v1'
    rooms = {r['uid']: r for r in map(json.loads, (data / 'rooms.jsonl').read_text().splitlines())}
    links = {arm: _lineage(run, arm, 'e1') for arm in h.ARMS}
    scores = {kind: {arm: {c: dev_metrics(c == selected[arm], .9 if c in (selected[arm], 'sft') else .5) for c in names}
                     for arm in h.ARMS} for kind, names in (('candidates', ds.CANDIDATES), ('report_only', ds.REPORT_ONLY))}
    for arm in h.ARMS:  # No common complete rooms: never eligible, rendered unmeasured.
        scores['candidates'][arm]['rpo-dpo1'].update(complete_rate=0., size_log_mae_common_with_sft=None,
                                                    sft_size_log_mae_common=None, common_with_sft=0, footprint_bias=None)
    d._publish_dir(run / 'dev-eval-010' / 'scores', {'scores.json': json.dumps(
        {'protocol': 'amendment-010-dev-scores', 'data_manifest_sha256': 'd' * 64, 'uids_sha256': 'e' * 64, **scores}).encode()})
    ds.select(argparse.Namespace(scores=run / 'dev-eval-010' / 'scores', out=run / 'dev-eval-010' / 'selection.json',
                                 code_root=None, shards=2, no_submit=True))
    runs = {arm: run / f'infer-{arm}-002' if c == 'sft' else
            _inference(run, data, rooms, arm, S.candidate_stage(c), links[arm], broken if arm == 'joint' else None,
                       out=f'infer-{arm}-005', root=d.CANDIDATES[c.split('-')[0]][0]) for arm, c in selected.items()}
    e.evaluate(data, runs, run / 'geometry-v5', repeats=200, stage_label='selected')
    R.build(data, run / 'geometry-v5', runs, run / 'reference.json', run / 'report-selected', repeats=200,
            selection=run / 'dev-eval-010' / 'selection.json')
    return run


def reseal(directory, name, value):
    h.write_json(directory / name, value)
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    manifest['files'][name] = h.sha256(directory / name)
    h.write_json(directory / 'MANIFEST.json', manifest)


@pytest.fixture(scope='module')
def template(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp('stage-world'))


@pytest.fixture
def run(template, tmp_path):
    return shutil.copytree(template, tmp_path / 'run')


def reports(run, stages=S.STAGES):
    return {s: run / f'report-{s}' for s in stages}


def test_three_stage_table_reads_measured_values_only(run, tmp_path):
    result = S.build(run, reports(run), tmp_path / 'table')
    t = result['tables']
    rows = {(r['stage'], r['arm']): r for r in t['stages']}
    n = 39
    dpo2 = json.loads((run / 'report-dpo2' / 'results.json').read_text())['tables']
    assert rows['dpo2', 'J']['geometry_success'] == pytest.approx((n - 3) / n) == {r['endpoint'][:3]: r for r in dpo2['T1']}['(a)']['J']
    assert rows['dpo2', 'J']['complete_rate'] == pytest.approx((n - 3) / n) and rows['sft', 'J']['complete_rate'] == 1
    assert rows['dpo2', 'J']['size_log_mae']['estimate'] == 0 and rows['dpo2', 'P']['size_log_mae']['estimate'] > 0
    summary = json.loads((run / 'geometry-v4' / 'summary.json').read_text())
    assert rows['dpo2', 'P']['size_iou'] == summary['arms']['placement']['all']['size_iou']
    t4 = [r for r in dpo2['T4'] if r['system'] == 'J' and r['complete']]
    assert rows['dpo2', 'J']['oor_rotated_strata'] == pytest.approx(
        sum(r['oor_rotated'] * r['complete'] for r in t4) / sum(r['complete'] for r in t4))
    # Not carried by any input: unmeasured with a reason, never filled in.
    assert rows['sft', 'P']['oor_rotated'] is None and rows['sft', 'P']['missing']['oor_rotated'].startswith('未测')
    assert rows['dpo1', 'J']['missing']['automatic_acceptance'].startswith('未测')
    reference = json.loads((run / 'reference.json').read_text())['scopes']['all']
    gt, prior = t['reference']
    assert gt['geometry_success'] == reference['gt']['gt_success'] and gt['oor_rotated'] == reference['gt']['gt_oor_rotated']
    assert prior['size_log_mae'] == reference['prior']['prior_size_log_mae'] and gt['missing']['size_iou'].startswith('不适用')
    assert [(r['stage'], r.get('rule_outcome')) for r in t['decisions']] == [
        ('sft', 'J'), ('dpo1', 'J'), ('dpo2', 'P'), ('selected', None)]  # A given DPO2 report still renders.
    assert [r['note'] for r in t['decisions']][2:] == ['仅供参考，非交付决定', S.NOT_RUN] and result['not_run'] == {'selected': S.NOT_RUN}
    assert {r['stage'] for r in t['differences']} == set(S.STAGES) and len(t['differences']) == 3 * (3 + len(S.T1B))
    # protocol.md: SFT (a), (b) J−P is the confirmatory endpoint; DPO1/DPO2 are descriptive (amendment 010).
    role = {(r['stage'], r['endpoint'].split(' ')[0]): r['role'] for r in t['differences']}
    assert {k for k, v in role.items() if v.startswith('主要')} == {('sft', '(a)'), ('sft', '(b)')}
    assert not {k for k, v in role.items() if v.startswith('次要')} and role['dpo2', '(a)'] == '描述性'
    lineage = {(r['stage'], r['arm']): r for r in t['lineage']}
    assert lineage['dpo2', 'P']['pairs'] == 2150 and lineage['dpo1', 'J']['steps'] == 5000
    assert lineage['sft', 'J']['merge_changed_fraction'] == .15 and lineage['sft', 'J']['missing']['pairs'].startswith('不适用')
    assert [p['kept_both_arms'] for p in t['pairs']] == [2000, 2150]
    out = tmp_path / 'table'
    h.verify_artifacts(out, json.loads((out / 'MANIFEST.json').read_text())['files'])
    md = (out / 'stage_table.md').read_text()
    assert '修订 009' in md and '运行时 GPU 合并' in md and '| 未测 |' in md and '正式交付决定' not in md
    assert 'SFT 阶段 (a)、(b) J−P 为方案保留的主要（确认性）比较' in md and '属次要比较' in md
    assert '只有 DPO2' not in md and '共同主要终点' not in md and '| sft | (a) 全请求原始几何成功率 | 主要（确认性' in md
    assert (out / 'stage_stages.csv').read_text().splitlines()[0].startswith('stage,arm,requests,complete_rate')
    with pytest.raises(FileExistsError):
        S.build(run, reports(run), out)


def test_stage_without_report_is_not_run_and_dpo2_is_stopped(run, tmp_path):
    result = S.build(run, reports(run, ('sft', 'dpo1')), tmp_path / 'table')
    missing = [r for r in result['tables']['stages'] if r['stage'] == 'dpo2']
    assert len(missing) == 2 and all(r['geometry_success'] is None and r['missing']['geometry_success'] == S.STOPPED
                                     for r in missing)
    assert result['not_run'] == {'dpo2': S.STOPPED, 'selected': S.NOT_RUN}
    md = (tmp_path / 'table' / 'stage_table.md').read_text()
    assert '| dpo2 | J | 已停止（修订 010 补充：测试推理在查看前停止，未测） |' in md and '| selected | P | 未运行 |' in md
    short = '已停止（修订 010 补充：测试推理在查看前停止，未测）'
    assert 'dpo2（' + S.STOPPED + '）' in md and f'| dpo2 | {short} | {short} | {short} | {S.STOPPED} |' in md


def _tamper_receipt(run, *_):
    path = run / 'infer-joint-003' / 'run_receipt.json'
    receipt = json.loads(path.read_text())
    receipt['config']['lora'] = 'premerged_bf16_stage_model'  # A merged, not exact-policy, DPO evaluation.
    receipt.pop('run_sha256')
    h.write_json(path, {**receipt, 'run_sha256': i._hash(receipt)})


def _tamper_parent(run, *_):
    path = run / 'dpo-001' / 'placement' / 'sft-merged.json'
    h.write_json(path, {**json.loads(path.read_text()), 'note': 'edited'})


def _tamper_pairs(run, *_):
    root = run / 'dpo-001' / 'stage2-pairs'
    reseal(root, 'summary.json', {**json.loads((root / 'summary.json').read_text()), 'kept_both_arms': 1})


@pytest.mark.parametrize('tamper, match', [
    (None, 'was not built on geometry-v3'),
    (_tamper_receipt, 'infer-joint-003 does not bind'),
    (_tamper_parent, 'lineage parent hash'),
    (_tamper_pairs, 'trained pairs are not stage2-pairs')])
def test_refuses_broken_lineage(run, tmp_path, tamper, match):
    given = reports(run)
    if tamper is None:  # The DPO2 report passed as DPO1.
        given = {**given, 'dpo1': run / 'report-dpo2', 'dpo2': run / 'report-dpo1'}
    else:
        tamper(run)
    with pytest.raises(ValueError, match=match):
        S.build(run, given, tmp_path / 'table')
    assert not (tmp_path / 'table').exists()


def test_shared_plan_and_merge_diagnostic_rules(run):
    stages = [S.load_stage(run, s, p) for s, p in reports(run).items()]
    S.check_shared(stages)
    stages[2]['receipts']['placement']['context_plan_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='other plan'):
        S.check_shared(stages)
    assert S.merge_fraction({'merge_diagnostic': {'error': 'OSError: boom'}}) == (None, None, '未测：合并诊断失败（OSError: boom）')
    assert S.merge_fraction({})[2] == '未测：lineage 收据无 merge_diagnostic'


def test_report_v2_versions_with_the_same_numbers(run):
    """The amendment-009 SFT/DPO1 reports (report_v2.py of main-281257f25) sit beside an amendment-010 report."""
    assert h.sha256(R.__file__) in S.SAME_NUMBERS  # Re-audit (text-only diff?) before adding a new report_v2.py.
    stages = [S.load_stage(run, s, p) for s, p in reports(run).items()]
    stages[0]['results']['code_sha256']['report_v2.py'] = '181bb2fe3928026e8f2241ed109e835e70f0d9a0c29328cbfa925e2674c254e4'
    S.check_shared(stages)
    stages[0]['results']['code_sha256']['report_v2.py'] = 'c' * 64
    with pytest.raises(ValueError, match='differs from sft in data, reference, bootstrap or code'):
        S.check_shared(stages)


@pytest.fixture(scope='module')
def selected_world(tmp_path_factory):
    """Read-only: receipts carry absolute run paths (as on the cluster), so this world is never copied."""
    return build_selected(build_world(tmp_path_factory.mktemp('selected-world')))


def selected_reports(run):
    return {**reports(run, ('sft', 'dpo1')), 'selected': run / 'report-selected'}


def test_selected_row_and_dev_table(selected_world, tmp_path):
    run, out = selected_world, tmp_path / 'table'
    selection = run / 'dev-eval-010' / 'selection.json'
    result = S.build(run, selected_reports(run), out, selection)
    t = result['tables']
    rows = {(r['stage'], r['arm']): r for r in t['stages']}
    report = json.loads((run / 'report-selected' / 'results.json').read_text())
    t1 = {r['endpoint'].split()[0]: r for r in report['tables']['T1']}
    assert rows['selected（e1-dpo2）', 'J']['geometry_success'] == t1['(a)']['J'] == 1
    assert rows['selected（sft）', 'P']['size_log_mae'] == t1['(b)']['P'] == rows['sft', 'P']['size_log_mae']  # Same run.
    assert rows['dpo2', 'J']['missing']['geometry_success'] == S.STOPPED and result['not_run'] == {'dpo2': S.STOPPED}
    role = {(r['stage'], r['endpoint'].split(' ')[0]): r['role'] for r in t['differences']}
    assert {k for k, v in role.items() if v.startswith('次要')} == {('selected', e) for e in S.ENDPOINTS}
    assert role['selected', '(a)'] == '次要（交付比较，修订 010）' and role['dpo1', '(b)'] == '描述性'
    decision = {r['stage']: r for r in t['decisions']}['selected']
    assert decision['rule_outcome'] == report['delivery']['delivered'] == 'J' and decision['note'].startswith('正式交付决定（修订 010')
    lineage = {(r['stage'], r['arm']): r for r in t['lineage']}
    assert lineage['selected（e1-dpo2）', 'J']['recipe'].startswith('lr 5e-06, 1 epochs') and lineage['selected（e1-dpo2）', 'J']['pairs'] == 2150
    assert lineage['selected（sft）', 'P']['evaluated_policy'] == S.POLICY['sft']
    assert result['inputs']['selected']['candidates'] == SELECTED and result['inputs']['selected']['lineage_receipts']['joint'] == \
        h.sha256(run / 'dpo-002' / 'joint' / 'dpo2-merged.json')
    # Dev table: every candidate plus the report-only amendment-009 rows, eligibility and selection as chosen.
    dev = {(r['arm'], r['candidate']): r for r in t['dev']}
    assert len(dev) == 2 * (len(ds.CANDIDATES) + len(ds.REPORT_ONLY))
    assert dev['J', 'e1-dpo2']['selected'] and dev['J', 'sft']['eligible'] and not dev['J', 'sft']['selected']
    assert not dev['J', 'e1-dpo1']['eligible'] and dev['P', 'sft']['selected'] and dev['J', 'e1-dpo2']['a_prime'] == .6
    assert dev['J', 'a009-dpo1']['eligible'] is None and dev['J', 'a009-dpo1']['missing']['selected'] == S.REPORT_ONLY
    assert dev['P', 'rpo-dpo1']['size_log_mae'] is None and dev['P', 'rpo-dpo1']['missing']['size_log_mae'].startswith('未测')
    assert result['dev_selection']['selection_sha256'] == h.sha256(selection) == report['input_sha256']['selection']
    assert result['dev_selection']['rooms'] == 600 and result['dev_selection']['rule'] == ds.RULE
    md = (out / 'stage_table.md').read_text()
    assert '| J | a009-dpo2 | 600 |' in md and '| 仅报告 | 仅报告 |' in md and ds.RULE in md and '选定：J = e1-dpo2；P = sft。' in md
    assert '| selected（e1-dpo2） | J |' in md and '正式交付决定（修订 010' in md and '| dpo2 | J | 已停止' in md
    assert (out / 'stage_dev.csv').read_text().splitlines()[0].startswith('arm,candidate,requests')
    assert S.build(run, reports(run, ('sft',)), tmp_path / 'dev-only', selection)['tables']['dev']  # Dev table alone.


def _flip_selection(run):
    path = run / 'dev-eval-010' / 'selection.json'
    value = json.loads(path.read_text())
    path.unlink()
    i.write_once(path, {**value, 'selected': {**value['selected'], 'joint': 'e1-dpo1'}})


def _scores_edited(run):
    path = run / 'dev-eval-010' / 'scores' / 'scores.json'
    path.write_text(path.read_text().replace('"a_prime": 0.6', '"a_prime": 0.61', 1))


def _candidate_lineage(run):
    path = run / 'dpo-002' / 'joint' / 'dpo2-merged.json'
    receipt = json.loads(path.read_text())
    h.write_json(path, {**receipt, 'training': {**receipt['training'], 'candidate': 'rpo'}})


@pytest.mark.parametrize('tamper, match', [
    (None, 'needs --selection'),
    (_flip_selection, 'does not reproduce from its dev scores'),
    (_scores_edited, 'artifact hash mismatch: scores.json'),
    (_candidate_lineage, 'lineage receipt is not joint/dpo2')])
def test_selected_row_refuses_unbound_selection_or_lineage(tmp_path, tamper, match):
    run = build_selected(build_world(tmp_path))
    selection = None if tamper is None else run / 'dev-eval-010' / 'selection.json'
    if tamper:
        tamper(run)
    with pytest.raises(ValueError, match=match):
        S.build(run, selected_reports(run), tmp_path / 'table', selection)
    assert not (tmp_path / 'table').exists()
