"""Stage ablation over real report_v2 outputs of a fixture DPO chain (no network, no jobs)."""
import hashlib
import json
import shutil

import pytest

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
BASE = {'revision': h.REVISION, 'files': {'model.safetensors': '1' * 64}}
TRAINING = {'manifest_sha256': 'a' * 64, 'files': {'train.jsonl': '2' * 64}}
BROKEN = {'x-06': '{"placements": [', 'y-0': '{}', 'x-07': 'nonsense'}  # Three J-DPO2 failures.
PAIRS = {'dpo1': 2000, 'dpo2': 2150}
DIAGNOSTIC = {'modules': {'q_proj': {'changed': 3, 'total': 10, 'fraction': .3},
                          'other': {'changed': 0, 'total': 10, 'fraction': 0.}},
              'dtypes': ['torch.bfloat16'], 'tensors_only_in_parent': 0, 'tensors_only_in_merged': 0}


def _hex(*parts):
    return hashlib.sha256('/'.join(parts).encode()).hexdigest()


def _lineage(run, arm):
    root, links = run / 'dpo-001' / arm, {}
    root.mkdir(parents=True)
    for k, stage in enumerate(S.STAGES):
        files = {'config.json': _hex(arm, stage, 'config'), 'model.safetensors': _hex(arm, stage, 'weights')}
        receipt = {'protocol': d.MERGE_PROTOCOL, 'arm': arm, 'stage': stage, 'base': BASE,
                   'base_sha256': i._hash(BASE['files']), 'training_data': TRAINING,
                   'adapter': {'path': str(root / stage / 'final'), 'files': {'adapter_model.safetensors': _hex(arm, stage)}},
                   'model': {'path': str(root / f'{stage}-merged'), 'files': files}, 'model_sha256': i._hash(files),
                   'merge': 'fastfill.merge_lora: BF16 PeftModel.merge_and_unload', 'merge_diagnostic': DIAGNOSTIC}
        if stage == 'sft':
            receipt.update(parent={'stage': 'generic_base', 'path': '/base'},
                           training={'run': '/sft', 'training_run_sha256': _hex(arm, 'run'),
                                     'recipe': 'SFT as trained: LoRA r32 a64, lr 5e-5, 1 epoch'})
        else:
            previous, pairs = S.STAGES[k - 1], run / 'dpo-001' / d.PAIRS[stage]
            receipt.update(parent={'stage': previous, 'path': links[previous]['model']['path'],
                                   'lineage_sha256': h.sha256(root / f'{previous}-merged.json')},
                           training={'pairs': {'path': str(pairs / f'{arm}.jsonl'), 'count': PAIRS[stage],
                                               'manifest_sha256': h.sha256(pairs / 'MANIFEST.json')},
                                     'completion': {'steps': 10 * PAIRS[stage] // 4, 'admitted_pairs': PAIRS[stage]},
                                     'recipe': {**d.RECIPE, **d.TRAIN_FIXED}})
        h.write_json(root / f'{stage}-merged.json', receipt)
        links[stage] = receipt
    return links


def _inference(run, data, rooms, arm, stage, links, raw=None):
    out = run / f'infer-{arm}-{S.TAG[stage]}'
    out.mkdir()
    (out / 'context_plan.json').write_text('{"plan": "shared"}\n')
    link, manifest = links[stage], json.loads((data / 'MANIFEST.json').read_text())
    receipt = {'data': {'manifest_sha256': h.sha256(data / 'MANIFEST.json')}, 'training_data': TRAINING, 'base': BASE,
               'base_sha256': link['base_sha256'], 'context_plan_sha256': h.sha256(out / 'context_plan.json'),
               'config': {'arm': arm, **DECODE, 'lora': S.LORA['sft' if stage == 'sft' else 'dpo']}}
    adapter = i._hash(link['adapter']['files'])
    if stage == 'sft':
        receipt.update(training_run_sha256=link['training']['training_run_sha256'], adapter_sha256=adapter,
                       adapter_files=link['adapter']['files'])
    else:
        parent = links[S.STAGES[S.STAGES.index(stage) - 1]]
        receipt.update(stage=stage, lineage_sha256=h.sha256(run / 'dpo-001' / arm / f'{stage}-merged.json'),
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
    assert [(r['stage'], r['rule_outcome']) for r in t['decisions']] == [('sft', 'J'), ('dpo1', 'J'), ('dpo2', 'P')]
    assert {r['stage'] for r in t['differences']} == set(S.STAGES) and len(t['differences']) == 3 * (3 + len(S.T1B))
    lineage = {(r['stage'], r['arm']): r for r in t['lineage']}
    assert lineage['dpo2', 'P']['pairs'] == 2150 and lineage['dpo1', 'J']['steps'] == 5000
    assert lineage['sft', 'J']['merge_changed_fraction'] == .15 and lineage['sft', 'J']['missing']['pairs'].startswith('不适用')
    assert [p['kept_both_arms'] for p in t['pairs']] == [2000, 2150]
    out = tmp_path / 'table'
    h.verify_artifacts(out, json.loads((out / 'MANIFEST.json').read_text())['files'])
    md = (out / 'stage_table.md').read_text()
    assert '修订 009' in md and '运行时 GPU 合并' in md and '| 未测 |' in md and '正式交付决定' in md
    assert (out / 'stage_stages.csv').read_text().splitlines()[0].startswith('stage,arm,requests,complete_rate')
    with pytest.raises(FileExistsError):
        S.build(run, reports(run), out)


def test_stage_without_report_is_not_run(run, tmp_path):
    result = S.build(run, reports(run, ('sft', 'dpo1')), tmp_path / 'table')
    missing = [r for r in result['tables']['stages'] if r['stage'] == 'dpo2']
    assert len(missing) == 2 and all(r['geometry_success'] is None and r['missing']['geometry_success'] == S.NOT_RUN
                                     for r in missing)
    assert result['not_run'] == {'dpo2': S.NOT_RUN} and '| dpo2 | J | 未运行 |' in (tmp_path / 'table' / 'stage_table.md').read_text()


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
