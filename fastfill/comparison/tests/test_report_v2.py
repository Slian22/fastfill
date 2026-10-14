"""Amendment-005 report/delivery generator and the frozen reference builder (no network, no jobs)."""
import gzip
import json
import math
from pathlib import Path
import random
import statistics

import pytest

from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer, judge_batch
from fastfill.comparison import llm_agent as L
from fastfill.comparison import reference_rows as RR
from fastfill.comparison import report_v2 as R
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.fid import WEIGHTS_SHA256
from fastfill.comparison.tests.test_llm_agent import FakeApi, args, target, world, write_rows  # noqa: F401
from fastfill.validate import check


def merged(root, data, arm, rooms, raw=None, cap=()):
    """A verified merged-inference dir; `raw` overrides answers, `cap` UIDs hit the cap without EOS."""
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    run = root / ('infer-' + arm)
    run.mkdir()
    receipt = {'data': {'manifest_sha256': h.sha256(data / 'MANIFEST.json')},
               'training_data': {'manifest_sha256': 'a' * 64}, 'base_sha256': 'base', 'config': {'arm': arm}}
    receipt['run_sha256'] = infer._hash(receipt)
    h.write_json(run / 'run_receipt.json', receipt)
    write_rows(run / 'records.jsonl', [
        {'uid': u, 'source': rooms[u]['source'], 'group': rooms[u]['group'], 'arm': arm,
         'raw': (raw or {}).get(u, target(None, arm, rooms[u])), 'error': None,
         'reached_token_cap': u in cap, 'stopped_on_eos': u not in cap, 'generated_tokens': 50, 'input_tokens': 400,
         'batch_id': i // 8, 'duration_seconds': 36., 'run_sha256': receipt['run_sha256']}
        for i, u in enumerate(manifest['expected_uids'])])
    h.write_json(run / 'result.json', {'status': 'INFERENCE_COMPLETE', 'pending_count': 0})
    h.write_json(run / 'MANIFEST.json', {'expected_uids': manifest['expected_uids'], 'files': h.artifact_manifest(run)})
    return run


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def built(world):
    root, rooms = world
    data = root / 'data'
    runs = {'joint': merged(root, data, 'joint', rooms, raw={'x-06': '{"placements": [', 'y-0': '{}'}, cap={'x-05'}),
            'placement': merged(root, data, 'placement', rooms)}
    e.evaluate(data, runs, root / 'geometry-v2', repeats=200)
    L.run(args(root), FakeApi(rooms, {'il-bed-1': ['bad', 'valid'], 'y-0': ['bad', 'worse', 'still bad']}))
    ev.evaluate(data, root / 'cohort.json', root / 'run', root / 'llm-eval', runs, repeats=200)
    requests = {arm: {r['uid']: r for r in _jsonl(data / arm / 'requests.jsonl')} for arm in h.ARMS}
    reference = RR.build_reference(_jsonl(data / 'rooms.jsonl'), requests, h.sha256(data / 'MANIFEST.json'), ['il-bed-2'])
    infer.write_once(root / 'reference.json', reference)
    geometry_sha = h.sha256(root / 'geometry-v2' / 'MANIFEST.json')
    h.write_json(root / 'fid.json', {
        'extractor': 'pytorch_fid:0.3.0:pool3', 'feature_dimension': 2048, 'weights_sha256': WEIGHTS_SHA256,
        'provenance': {'provenance': {'geometry_manifest_sha256': geometry_sha}},
        'strata': {'IL3D_3dfront/bedroom': {'joint_vs_reference': 30., 'placement_vs_reference': 31.,
                                            'counts': {'requested': 3, 'reference_eligible': 3, 'shared_success': 3}}}})
    records = [{'id': f'{u}-{arm}', 'uid': u, 'arm': arm, 'stratum': s, 'group': None, 'launched': True,
                **({'status': 'failed', 'error': 'client_timeout'} if (u, arm) == ('il-bed-2', 'placement') else
                   {'status': 'completed', 'scores': {'functionality_score': 8 + (arm == 'joint'), 'layout_score': 7,
                                                      'aesthetics_score': 9, 'overall_score': 8}})}
               for u, s in (('il-bed-0', 'IL3D_3dfront/bedroom'), ('il-bed-1', 'IL3D_3dfront/bedroom'),
                            ('il-bed-2', 'IL3D_3dfront/bedroom'), ('holo-k-0', 'OptiScene_holodeck/kitchen'))
               for arm in h.ARMS]
    (root / 'judge').mkdir()
    write_rows(root / 'judge' / 'records.jsonl', records)
    h.write_json(root / 'judge' / 'summary.json', {**judge_batch.summarize(records, 'gpt-6.1-sol', len(records)),
                                                   'provenance': {'geometry_manifest_sha256': geometry_sha}})
    return root, rooms, runs


def _build(root, runs, out='report', **extra):
    options = {'agents': {'L': root / 'llm-eval'}, 'fid': root / 'fid.json', 'judge': root / 'judge', 'repeats': 200, **extra}
    return R.build(root / 'data', root / 'geometry-v2', runs, root / 'reference.json', root / out, **options)


def test_end_to_end_report_and_delivery(built):
    root, rooms, runs = built
    result = _build(root, runs)
    t, d = result['tables'], result['delivery']
    n = len(rooms)
    # J answers GT except one cap-without-EOS (valid JSON), one truncated JSON and one schema-invalid output.
    t1 = {row['endpoint'][:4]: row for row in t['T1']}
    assert t1['(a) ']['J'] == pytest.approx((n - 3) / n) and t1['(a) ']['P'] == 1
    assert t1['(b) ']['J']['estimate'] == 0 and t1['(b) ']['delta']['ci95'][1] < 0
    assert [t1['(a) ']['p_holm'], t1['(b) ']['p_holm']] == R.holm([t1['(a) ']['p'], t1['(b) ']['p']])
    # P sizes are GT + 0.1 m, so P is oversized and its (a') factor shrinks it; J is unbiased.
    assert d['scale_factor']['joint'] == 1 and d['scale_factor']['placement'] < 1
    assert d['B_footprint']['P']['estimate'] > 0 and d['B_footprint']['J']['estimate'] == 0
    assert d == {**d, **R.delivery(t1['(b) ']['delta'], t1["(a′)"]['delta'])}
    assert d['condition_i_size_log_mae'] and not d['condition_ii_debiased_success'] and d['delivered'] == 'P'
    t6 = {row['system']: row for row in t['T6']}
    assert (t6['J']['token_cap_without_eos'], t6['J']['cap_rule_flipped_complete'], t6['J']['invalid_json'],
            t6['J']['invalid_schema_or_geometry'], t6['J']['complete']) == (1, 1, 1, 1, n - 3)
    assert t6['P']['complete'] == n and t6['L-raw']['invalid_output'] == 2 and t6['L-harness']['invalid_output'] == 1
    # Reference rows sit beside the arms; sensitivity scopes use the frozen lists.
    assert t['T3'][0]['gt_success'] == 1 and t['T2'][0]['prior_size_log_mae'] > 0
    s1 = {row['scope']: row for row in t['S1']}
    assert s1['排除声明的已知缺陷房间（1）']['requests'] == n - 1 and s1[f'GT 通过冻结几何门（{n}）']['requests'] == n
    assert {row['level'] for row in t['S2']} == {'N'} and 'prior_level_pair_category_unresolved_without_frozen_size_priors' in result['warnings']
    # Paper strata: rotated OOR, FID with small-N flag, GPT from the judge records.
    t4 = {(row['stratum'], row['system']): row for row in t['T4']}
    bed = t4['IL3D_3dfront/bedroom', 'J']
    assert bed['fid'] == 30 and bed['fid_flag'] == 'N<50' and bed['gpt'] == pytest.approx(25 / 3)
    assert t4['IL3D_3dfront/bedroom', 'L-harness']['requests'] == 3 and t4['IL3D_3dfront/bedroom', 'L-raw']['fid'] is None
    gpt = t['T4_gpt'][0]
    assert gpt['n'] == 3 and gpt['delta']['estimate'] == pytest.approx(1 / 3)
    # Amendment 011 section 2 on a J/P judge: per-arm pooled means; J−P is the canonical endpoint itself.
    arms = {r['system']: r for r in t['T4_gpt_arms']}
    assert arms['J']['mean'] == pytest.approx(25 / 3) and arms['P']['mean'] == pytest.approx(8)
    assert (arms['P']['success'], arms['P']['requested']) == (3, 4)
    assert [r['comparison'] for r in t['T4_gpt_contrasts']] == ['J−P'] and t['T4_gpt_contrasts'][0]['delta'] == gpt['delta']
    # Agents on the cohort: J/P/agent contrasts on the three pooled scopes only, with cost.
    assert {(r['scope'], r['comparison']) for r in t['T5_contrasts']} == {
        (s, c) for s in ('cohort', 'strata_pooled', 'other_sources')
        for c in ('J−P', 'J−L-raw', 'P−L-raw', 'J−L-harness', 'P−L-harness')}
    assert {r['system'] for r in t['T5']} == {'J', 'P', 'L-raw', 'L-harness'} and result['cohort_rooms'] == 36
    cost = {r['system']: r for r in t['T5_cost']}
    assert cost['L']['model'] == L.EXPECTED_MODEL and cost['J']['gpu_hours_full_test_generation'] == pytest.approx(5 * 36 / 3600)
    out = root / 'report'
    h.verify_artifacts(out, json.loads((out / 'MANIFEST.json').read_text())['files'])
    report = (out / 'report.md').read_text()
    assert '达上限无 EOS' in report and 'Holm' in report
    # Amendment 011 section 4: Holm at the confirmatory SFT stage only; amendment label from the stage.
    assert result['amendment'] == R.AMENDMENT['sft'] == '006' and result['reporting_amendment'] == '011'
    assert result['holm_co_primaries'] == {'a': t1['(a) ']['p_holm'], 'b': t1['(b) ']['p_holm']}
    assert t1['(b) ']['p'] == 0 and '| < 1/200 | < 1/200 |' in report  # Audit minor 7: p = 0 prints as < 1/R.
    # J/P-only judge run: the pre-specified N-system judged set is not claimed.
    assert '评审系统集合为预先指定' not in report and '| J−P |' in report and '此 J/P 视觉运行不产出 KID' in report
    # SFT geometry (no stage label): the amendment-005 rule is informational, delivery is between the dev-selected models.
    assert result['stage'] == 'sft' and d['stage'] == 'sft' and d['binding'] is False and 'models' not in result
    assert '**交付：' not in report and '规则结果：P' in report and '修订 010：交付比较为 dev 选定模型' in report
    assert 'DPO2' not in report.split('## T1 ')[0] and 'selection' not in result['input_sha256']
    assert R.ROLE['sft'] in report and '主要（确认性）' in R.ROLE['sft']
    # House components per T2/T3/S1 scope; no CI below 10 components; few clusters / small n flagged.
    k = len({room['group'] for room in rooms.values()})
    t3 = {r['scope']: r for r in t['T3']}
    assert (t3['all']['components'], t3['all']['flags']) == (k, 'small n') and k >= R.NO_CI
    assert (t3['source/IL3D_3dfront']['components'], t3['source/IL3D_3dfront']['flags']) == (2, 'few clusters; small n')
    assert all({'components', 'flags'} <= set(r) for name in ('T2', 'S1') for r in t[name]) and t['T2'][0]['components'] == k
    lines = report.splitlines()
    il3d = [x for x in lines if x.startswith('| source/IL3D_3dfront | 4 | 2 | few clusters; small n |')]
    assert len(il3d) == 2 and all('[' not in x for x in il3d) and '+0.0000 — (2 components)' in il3d[1]
    assert any('[' in x for x in lines if x.startswith(f'| all | {n} | {k} | small n |'))
    # Every warning results.json carries (report level and inside table cells) is rendered.
    cells = {w for rows in t.values() for r in rows for v in r.values() if isinstance(v, dict) for w in v.get('warnings', [])}
    assert 'singleton_source_signature_strata' in cells
    assert all(f'区间级 `{w}`' in report for w in cells) and all(w in report for w in result['warnings'])
    assert (out / 'T1.csv').read_text().splitlines()[0].startswith('endpoint,J,')
    with pytest.raises(FileExistsError):
        _build(root, runs)


def test_t5_own_bias_and_rescaled_success_for_every_system(built):
    """EXPLORATORY T5: B on rooms where J, P and that system are complete, (a') through _derive with own B."""
    root, rooms, runs = built
    t5 = {(r['scope'], r['system']): r for r in _build(root, runs)['tables']['T5']}
    cohort = [r['uid'] for r in _jsonl(root / 'llm-eval' / 'llm_harness-geometry.jsonl')]
    failed = {'x-05', 'x-06', 'y-0', 'il-bed-1'}  # J (cap rule, truncated, schema) and L-raw (il-bed-1, y-0).
    n = {s: t5['cohort', s]['bias_fp']['n'] for s in ('J', 'P', 'L-raw', 'L-harness')}
    # Per-system masks (J, P and that system complete): a weak agent no longer shrinks J's, P's or L-harness's set.
    assert n['J'] == n['P'] == n['L-harness'] > n['L-raw'] == len(set(cohort) - failed)
    strata_n = {s: t5['strata_pooled', s]['bias_vol']['n'] for s in ('J', 'P', 'L-raw', 'L-harness')}
    assert strata_n['J'] == strata_n['P'] == strata_n['L-harness'] == 4 and strata_n['L-raw'] == 3  # il-bed-1 L-raw
    # J and L answer GT sizes: B = 0, factor 1, (a') = raw (a). P sizes are GT + 0.1 m in every room.
    for s in ('J', 'L-raw', 'L-harness'):
        row = t5['cohort', s]
        assert row['bias_fp']['estimate'] == 0 and row['scale_factor'] == 1
        assert row['success_debiased_rate'] == row['geometry_success_rate']
    p = t5['cohort', 'P']
    b_fp = statistics.mean([math.log(1.1 * 1.1), math.log(1.1 * 2.1 / 2)])
    assert p['bias_fp']['estimate'] == pytest.approx(b_fp) and p['bias_fp']['ci95'] == pytest.approx([b_fp, b_fp])
    assert p['bias_vol']['estimate'] == pytest.approx(statistics.mean([3 * math.log(1.1), math.log(1.1 * 2.1 * 1.1 / 2)]))
    assert p['scale_factor'] == pytest.approx(math.exp(-b_fp / 2))
    stored = {r['uid']: r for r in _jsonl(root / 'data' / 'rooms.jsonl')}
    requests = {r['uid']: r for r in _jsonl(root / 'data' / 'placement' / 'requests.jsonl')}
    rows = {r['uid']: r for r in _jsonl(root / 'geometry-v2' / 'placement-geometry.jsonl')}
    again = [R._derive((rows[u], stored[u], requests[u], p['scale_factor']))['success_debiased'] for u in cohort]
    assert p['success_debiased_rate'] == sum(again) / len(again)
    report = (root / 'report' / 'report.md').read_text()
    assert "(a′) 自身 B 去偏成功率（EXPLORATORY）" in report and '各系统在自身完整输出上的均值' in report


def test_amendment_011_gt_and_equal_size_controls(built):
    """EXPLORATORY controls: GT at each factor (T1_gt, T5_gt), equal-size column, Delta(a') with CIs, other sources."""
    root, rooms, runs = built
    result = _build(root, runs)
    t = result['tables']
    stored = {r['uid']: r for r in _jsonl(root / 'data' / 'rooms.jsonl')}
    requests = {arm: {r['uid']: r for r in _jsonl(root / 'data' / arm / 'requests.jsonl')} for arm in h.ARMS}
    gt = {u: R._gt((stored[u], requests['joint'][u])) for u in stored}
    factor = result['delivery']['scale_factor']
    # T1_gt: GT unscaled is the frozen reference; J is unbiased (factor 1); P's factor shrinks GT.
    unscaled, at_j, at_p = t['T1_gt']
    assert unscaled['gt_success'] == t['T2'][0]['gt_success'] == 1 and at_j['gt_success'] == 1 and factor['joint'] == 1
    manual = [R._control((gt[u], stored[u], requests['joint'][u], factor['placement'])) for u in stored]
    assert at_p['gt_success'] == sum(manual) / len(manual) and at_p['scale_factor'] == factor['placement'] < 1
    assert at_p['system_a_prime'] == result['delivery']['a_prime']['P'] and at_j['system'] == 'J'
    # T5: three scopes; other_sources = cohort rooms outside the six strata.
    t5 = {(r['scope'], r['system']): r for r in t['T5']}
    cohort = [r['uid'] for r in _jsonl(root / 'llm-eval' / 'llm_harness-geometry.jsonl')]
    inside = sum(bool(R.stratum(stored[u])) for u in cohort)
    assert t5['other_sources', 'J']['requests'] == len(cohort) - inside == len(cohort) - t5['strata_pooled', 'J']['requests'] > 0
    # Equal-size control: J and L answer GT sizes (= raw (a)); P re-scored with GT x/y at its own z, pos and yaw.
    for system in ('J', 'L-harness', 'L-raw'):
        assert t5['cohort', system]['equal_size_success_rate'] == t5['cohort', system]['geometry_success_rate']
    rows = {r['uid']: r for r in _jsonl(root / 'geometry-v2' / 'placement-geometry.jsonl')}
    again = [R._control((rows[u], stored[u], requests['placement'][u], None)) for u in cohort]
    assert t5['cohort', 'P']['equal_size_success_rate'] == sum(again) / len(again)
    sized = R._control((rows[cohort[0]], stored[cohort[0]], requests['placement'][cohort[0]], None))
    assert sized is (R._control((gt[cohort[0]], stored[cohort[0]], requests['joint'][cohort[0]], 1.)) and True)
    assert R._control(({**rows[cohort[0]], 'complete': False}, None, None, None)) is False
    # T5_gt per scope: unscaled, and GT at each system's own factor beside that system's (a').
    t5_gt = {(r['scope'], r['control']): r for r in t['T5_gt']}
    for scope in ('cohort', 'strata_pooled', 'other_sources'):
        assert t5_gt[scope, 'GT 未缩放']['gt_success'] == 1 == t5_gt[scope, 'GT × J 自身系数']['gt_success']
        p = t5_gt[scope, 'GT × P 自身系数']
        assert p['scale_factor'] == t5['cohort', 'P']['scale_factor'] and p['system_a_prime'] == t5[scope, 'P']['success_debiased_rate']
    keep = [u for u in cohort if R.stratum(stored[u])]
    shrunk = [R._control((gt[u], stored[u], requests['joint'][u], t5['cohort', 'P']['scale_factor'])) for u in keep]
    assert t5_gt['strata_pooled', 'GT × P 自身系数']['gt_success'] == sum(shrunk) / len(shrunk)
    # Delta(a') with the same design and draws as Delta(a): J−P equals the a' rate difference; J−L-harness = Delta(a).
    contrast = {(r['scope'], r['comparison']): r for r in t['T5_contrasts']}
    for scope in ('cohort', 'strata_pooled', 'other_sources'):
        jp = contrast[scope, 'J−P']['own_debiased']
        assert jp['estimate'] == pytest.approx(t5[scope, 'J']['success_debiased_rate'] - t5[scope, 'P']['success_debiased_rate'])
        assert jp['ci95'] is not None and contrast[scope, 'J−L-harness']['own_debiased']['estimate'] == \
            contrast[scope, 'J−L-harness']['geometry_success']['estimate']
    # Cohort representativeness: source shares against the full test set.
    sources = {r['source']: r for r in t['T5_sources']}
    assert sum(r['cohort_rooms'] for r in sources.values()) == len(cohort) and sum(r['test_rooms'] for r in sources.values()) == len(rooms)
    assert sum(r['cohort_share'] for r in sources.values()) == pytest.approx(1) == sum(r['test_share'] for r in sources.values())
    report = (root / 'report' / 'report.md').read_text()
    assert '**队列不代表测试集**' in report and '### T5-S 队列与全测试集的来源构成' in report and '## T1-GT 对照' in report
    assert '等尺寸对照成功率（EXPLORATORY）' in report and "Δ(a′) [95% CI]（EXPLORATORY）" in report
    assert '| other_sources | J |' in report and '### T5-GT 对照' in report
    assert (root / 'report' / 'T5_gt.csv').exists() and (root / 'report' / 'T1_gt.csv').exists()


def _open_agent(root, rooms, system, script=None):
    """A complete llm_open run of `system` on the fixture cohort, evaluated against the merged J/P runs."""
    from fastfill.comparison import llm_open as O
    from fastfill.comparison.tests.test_llm_open import FakeLLM, argv, loader
    model = root / 'models' / O.SYSTEMS[system]['model']
    model.mkdir(parents=True, exist_ok=True)
    (model / 'config.json').write_text('{}')
    thinking = O.SYSTEMS[system]['thinking']
    fake = FakeLLM(rooms, script, default='<think>plan</think>{valid}' if thinking else '{valid}')
    assert O.main(argv(root, system, f'{system}-run'), loader(fake)) == 0
    return root / f'{system}-run'


def test_amendment_011_agent_disclosures(built):
    """Budgets, modes, harness, per-round cap hits, T6 JSON/schema split, not-run list, O1a support note."""
    root, rooms, runs = built
    cohort = json.loads((root / 'cohort.json').read_text())['uids']
    uid = next(u for u in cohort if len(rooms[u]['objects']) >= 2)
    answer = json.loads(target(None, 'joint', rooms[uid]))  # Schema-valid, but object 0 claims support by object 1.
    answer['placements'][0]['on'] = answer['placements'][1]['id']
    o1a = _open_agent(root, rooms, 'O1a', {uid: [json.dumps(answer)]})
    ev.evaluate(root / 'data', root / 'cohort.json', o1a, root / 'o1a-eval', runs, repeats=200)
    status = root / 'llm-open-status.json'
    h.write_json(status, {'O1a': {'model': 'qwen3-8b-nothink', 'state': 'complete', 'reason': None},
                          'O5': {'model': 'deepseek-v4-flash-0731', 'state': 'not run', 'reason': 'smoke: engine build failed'},
                          'kimi-k3': {'model': 'kimi-k3', 'state': 'not run', 'reason': 'not run (too large)'}})
    reason = 'not run: 4 engine-build failures, never substituted'
    result = _build(root, runs, out='report-011', agents={'L': root / 'llm-eval', 'O1a': root / 'o1a-eval'},
                    agent_runs={'L': root / 'run', 'O1a': o1a}, open_status=status, not_run={'O5': reason})
    t = result['tables']
    assert t['T5_not_run'] == [{'system': 'O5', 'model': 'deepseek-v4-flash-0731', 'state': 'not run', 'reason': reason},
                               {'system': 'kimi-k3', 'model': 'kimi-k3', 'state': 'not run', 'reason': 'not run (too large)'}]
    cost = {r['system']: r for r in t['T5_cost']}
    assert cost['L']['budget'] == '65,536（固定）' and cost['L']['reasoning'] == 'medium' and cost['L']['attempts'] == 3
    assert cost['O1a']['budget'] == 'min(1,024 + 128·物体, 40,960 − 提示)' and cost['O1a']['thinking'] is False
    assert [cost[k][f'cap_round_{i}'] for k in ('L', 'O1a') for i in (1, 2, 3)] == [0] * 6
    assert cost['J']['budget'] is None and cost['J']['answers_reaching_cap'] == 1  # x-05 hits the cap in the cohort.
    # T6 with the agent runs: invalid outputs split into JSON / schema (L's 'bad' answers are not JSON).
    t6 = {r['system']: r for r in t['T6']}
    assert (t6['L-raw']['invalid_json'], t6['L-harness']['invalid_json'], t6['L-raw']['invalid_schema_or_geometry']) == (2, 1, 0)
    assert 'invalid_output' not in t6['L-raw'] and t6['O1a-harness']['complete'] == 36
    assert result['input_sha256']['open_status'] == h.sha256(status)
    assert result['input_sha256']['agent_run_O1a_MANIFEST'] == h.sha256(o1a / 'MANIFEST.json')
    report = (root / 'report-011' / 'report.md').read_text()
    assert '| O5 | deepseek-v4-flash-0731 | not run | ' + reason + ' |' in report and '从不替代' in report
    assert 'O1a-raw 完整 36 间中 1 间支撑检查失败；O1a-harness 完整 36 间中 1 间支撑检查失败（全队列）' in report
    assert '不是布局能力的证据' in report and t5_support(t) == {'O1a-raw': 1, 'O1a-harness': 1, 'L-raw': 0}
    assert '至多 3 次尝试' in report and '不含解析器的错误文本' in report and '达上限回答（全部轮次）' in report
    assert '| O1a | qwen3-8b-nothink | 否 | thinking off |' in report
    # Thinking open model with a cap hit in round 1: budget with the thinking allowance, per-round counts.
    o5 = _open_agent(root, rooms, 'O5', {'il-bed-0': [('<think>plan</think>{valid}', 'length')]})
    ev.evaluate(root / 'data', root / 'cohort.json', o5, root / 'o5-eval', runs, repeats=200)
    again = _build(root, runs, out='report-o5', agents={'O5': root / 'o5-eval'}, agent_runs={'O5': o5}, fid=None, judge=None)
    o5_cost = {r['system']: r for r in again['tables']['T5_cost']}['O5']
    assert [o5_cost[f'cap_round_{i}'] for i in (1, 2, 3)] == [1, 0, 0] == [o5_cost['answers_reaching_cap'], 0, 0]
    assert o5_cost['budget'] == 'min(1,024 + 128·物体 + 16,384（思考）, 40,960 − 提示)' and o5_cost['thinking'] is True
    assert again['tables']['T5_not_run'] == [] and '未提供未运行清单' in (root / 'report-o5' / 'report.md').read_text()


@pytest.mark.parametrize('kind', ['complete_unreported', 'listed_and_reported', 'stale_status', 'run_without_agent',
                                  'other_run'])
def test_amendment_011_disclosure_inputs_fail_closed(built, monkeypatch, kind):
    root, rooms, runs = built
    status = root / 'status.json'
    h.write_json(status, {'O2': {'model': 'qwen3.8-27b', 'state': 'complete'}, 'L': {'state': 'incomplete'}}
                 if kind == 'stale_status' else {'O2': {'model': 'qwen3.8-27b', 'state': 'complete'}})
    options = {'complete_unreported': {'open_status': status}, 'listed_and_reported': {'not_run': {'L': 'not run'}},
               'stale_status': {'open_status': status, 'agents': {'L': root / 'llm-eval', 'O2': root / 'llm-eval'}},
               'run_without_agent': {'agent_runs': {'O5': root / 'run'}},
               'other_run': {'agent_runs': {'L': root / 'other-run'}}}[kind]
    if kind == 'other_run':  # A second, different L run on the same cohort: not the one llm-eval scored.
        L.run(args(root, out=root / 'other-run', max_output_tokens=4096), FakeApi(rooms))
    else:  # Disclosure inputs alone fail before any data is read or scored.
        monkeypatch.setattr(R.e, '_verified', lambda *_: pytest.fail('scoring started before the disclosure check'))
    match = {'complete_unreported': 'O2 is complete', 'listed_and_reported': 'also listed as not run',
             'stale_status': 'also listed as not run', 'run_without_agent': 'without --agent',
             'other_run': 'not the run its evaluation scored'}[kind]
    with pytest.raises(ValueError, match=match):
        _build(root, runs, out='bad', **options)
    assert not (root / 'bad').exists()


def test_gt_control_must_reproduce_the_frozen_reference(built, monkeypatch):
    root, _, runs = built
    score = R._gt
    monkeypatch.setattr(R, '_gt', lambda task: {**score(task), 'geometry_success': task[0]['uid'] != 'il-bed-0'})
    with pytest.raises(ValueError, match='GT layouts do not reproduce the frozen reference gt_pass'):
        _build(root, runs, out='bad')
    assert not (root / 'bad').exists()


def t5_support(tables):
    """Cohort rooms whose complete output fails the support check, for a few systems (T5 support_fail_rooms)."""
    return {r['system']: r['support_fail_rooms'] for r in tables['T5']
            if r['scope'] == 'cohort' and r['system'] in ('O1a-raw', 'O1a-harness', 'L-raw')}


def test_p_text_and_cap_rule():
    assert R.p_text(0., 2000) == '< 1/2000' and R.p_text(.0412, 2000) == '0.0412' and R.p_text(None, 2000) is None
    rows = [{'uid': 'a', 'complete': True}, {'uid': 'b', 'complete': True}, {'uid': 'c', 'complete': False}]
    flags = [{'reached_token_cap': True, 'stopped_on_eos': False}, {'reached_token_cap': True, 'stopped_on_eos': True},
             {'reached_token_cap': True, 'stopped_on_eos': False}]
    capped, flipped = R.cap_rule(rows, flags)
    assert [r['complete'] for r in capped] == [False, True, False] and capped[0]['error'] == R.CAP and flipped == 1
    with pytest.raises(ValueError):
        R.cap_rule(rows, flags[:2])


@pytest.mark.parametrize('stage', ['dpo1', 'dpo2'])
def test_delivery_section_not_binding_at_dpo_stages(built, stage):
    root, _, runs = built
    geometry = root / 'geometry-v2'
    _reseal(geometry, {**json.loads((geometry / 'summary.json').read_text()), 'stage': stage})
    result = _build(root, runs, out='report-' + stage, fid=None, judge=None)  # Visual lineage binds the old MANIFEST.
    report = (root / ('report-' + stage) / 'report.md').read_text()
    assert result['stage'] == stage and result['delivery']['binding'] is False  # Amendment 010: not DPO2 any more.
    assert R.ROLE[stage] in report and all(R.ROLE[s] not in report for s in R.ROLE if s != stage)
    assert '**交付：' not in report and f'（{stage} 阶段，非交付决定；修订 010：交付比较为 dev 选定模型）' in report
    assert f'{stage.upper()}：描述性' in report and '修订 008' not in report
    # Amendment 011: no Holm outside SFT (unadjusted, descriptive); governing amendment 009; visual pending.
    assert all(r.get('p_holm') is None for r in result['tables']['T1']) and 'holm_co_primaries' not in result
    assert result['amendment'] == '009' and '未经 Holm 校正' in report and '| < 1/200 | — |' in report
    assert 'FID/KID：视觉指标待测' in report and 'GPT 评审：视觉指标待测' in report and '此 J/P 视觉运行' not in report
    assert result['tables']['T4_gpt_arms'] == result['tables']['T4_gpt_contrasts'] == []
    again = _build(root, runs, out='report-again-' + stage, fid=None, judge=None, amendment='011')
    assert again['amendment'] == '011'


def _selected(root, runs, stages=('dpo1', 'sft'), inference=None):
    """geometry-v2 relabelled as evaluate_pair --stage-label selected writes it, plus dev_select's selection.json."""
    geometry = root / 'geometry-v2'
    _reseal(geometry, {**json.loads((geometry / 'summary.json').read_text()), 'stage': 'selected',
                       'stages': dict(zip(h.ARMS, stages)),
                       'lora': {'joint': 'unmerged_fp32_stage_adapter_on_bf16_training_base', 'placement': 'merge_and_unload_bf16'}})
    picked = {'joint': 'e1-dpo1', 'placement': 'sft'}
    path = root / f'selection-{len(list(root.glob("selection-*")))}.json'
    infer.write_once(path, {'protocol': 'amendment-010-dev-selection', 'selected': picked, 'test': {
        arm: {'candidate': c, 'inference': str((inference or runs)[arm])} for arm, c in picked.items()}})
    return path


def test_selected_stage_is_the_binding_delivery_comparison(built):
    root, _, runs = built
    selection = _selected(root, runs)
    result = _build(root, runs, out='report-selected', fid=None, judge=None, selection=selection)
    d = result['delivery']
    assert result['stage'] == d['stage'] == 'selected' and d['binding'] is True
    assert result['models'] == {'joint': {'candidate': 'e1-dpo1', 'stage': 'dpo1', 'lora': 'unmerged_fp32_stage_adapter_on_bf16_training_base'},
                                'placement': {'candidate': 'sft', 'stage': 'sft', 'lora': 'merge_and_unload_bf16'}}
    assert result['input_sha256']['selection'] == h.sha256(selection) and result['tables']['T5']  # Agents re-paired.
    report = (root / 'report-selected' / 'report.md').read_text()
    assert R.ROLE['selected'] in report and all(R.ROLE[s] not in report for s in R.ROLE if s != 'selected')
    assert f"**交付：{d['delivered']}**" in report and '## 交付决定（修正案 005 §1；修订 010：交付比较为 dev 选定模型）' in report
    assert 'J = e1-dpo1（stage dpo1，LoRA unmerged_fp32_stage_adapter_on_bf16_training_base）；P = sft（stage sft' in report
    assert '配对 J−P；dev 选定模型：有约束力的交付比较（修订 010）' in report
    assert result['amendment'] == '010' and 'holm_co_primaries' not in result and '属次要比较，p 未经 Holm 校正' in report
    assert 'J、P 为本报告 selected 阶段的模型（J = e1-dpo1，P = sft）。' in report  # Audit M4: T5 states the stage.
    assert all(r['p_holm'] is None for r in result['tables']['T1'][:2]) and result['tables']['T1'][0]['p'] is not None
    manifest = json.loads((root / 'report-selected' / 'MANIFEST.json').read_text())
    assert manifest['input_sha256']['selection'] == h.sha256(selection)
    with pytest.raises(ValueError, match='required for, and only for'):  # The selected stage needs its selection.
        _build(root, runs, out='bad', fid=None, judge=None)
    swapped = _selected(root, runs, inference={'joint': runs['placement'], 'placement': runs['joint']})
    with pytest.raises(ValueError, match='does not name these selected test runs'):
        _build(root, runs, out='bad', fid=None, judge=None, selection=swapped)
    other_stage = _selected(root, runs, stages=('dpo2', 'sft'))  # Geometry says J is a DPO2 model; selection says e1-dpo1.
    with pytest.raises(ValueError, match='does not name these selected test runs'):
        _build(root, runs, out='bad', fid=None, judge=None, selection=other_stage)
    assert not (root / 'bad').exists()


def test_selection_only_for_the_selected_stage(built):
    root, _, runs = built
    selection = root / 'selection.json'
    infer.write_once(selection, {'protocol': 'amendment-010-dev-selection'})
    with pytest.raises(ValueError, match='required for, and only for'):
        _build(root, runs, out='bad', selection=selection)
    assert not (root / 'bad').exists()


def test_component_counts_suppress_and_flag_intervals():
    value = {'estimate': .1, 'ci95': [0., .2]}
    assert R._cell(value, 9) == '+0.1000 — (9 components)'
    assert R._cell(value, 10) == R._cell(value) == '+0.1000 [+0.0000, +0.2000]'
    assert R._design([{'group': i % 30} for i in range(50)]) == {'components': 30, 'flags': None}
    assert R._design([{'group': i % 30} for i in range(49)]) == {'components': 30, 'flags': 'small n'}
    assert R._design([{'group': i % 29} for i in range(60)]) == {'components': 29, 'flags': 'few clusters'}
    lines = R._md([{'scope': 's', 'components': 2, 'delta': value}, {'scope': 't', 'components': 12, 'delta': value}],
                  [('scope', 'scope'), ('components', 'k'), ('delta', 'Δ')])
    assert lines[2] == '| s | 2 | +0.1000 — (2 components) |' and lines[3] == '| t | 12 | +0.1000 [+0.0000, +0.2000] |'


def test_debias_rescoring_matches_frozen_score_and_cap_rule(built):
    root, rooms, runs = built
    data = root / 'data'
    rows = _jsonl(root / 'geometry-v2' / 'joint-geometry.jsonl')
    requests = {r['uid']: r for r in _jsonl(data / 'joint' / 'requests.jsonl')}
    stored = {r['uid']: r for r in _jsonl(data / 'rooms.jsonl')}
    for row in rows:
        derived = R._derive((row, stored[row['uid']], requests[row['uid']], 1.))
        assert derived['success_debiased'] == (row['geometry_success'] if row['complete'] else False)
    capped = R.cap_failure(next(r for r in rows if r['uid'] == 'x-05'))
    assert capped['complete'] is False and capped['size_log_mae'] is None and capped['error'] == R.CAP


def _reseal(directory, summary):
    h.write_json(directory / 'summary.json', summary)
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    h.write_json(directory / 'MANIFEST.json', {**manifest, 'files': {**manifest['files'],
                                                                     'summary.json': h.sha256(directory / 'summary.json')}})


@pytest.mark.parametrize('kind', ['running_geometry', 'agent_geometry_code', 'fid_unlisted_agent', 'judge_arms'])
def test_refuses_other_geometry_code_or_unmatched_visual_systems(built, monkeypatch, kind):
    root, rooms, runs = built
    if kind == 'running_geometry':  # geometry-v2 and the agents were scored by another geometry.py.
        monkeypatch.setattr(g, '__file__', R.__file__)
    if kind == 'agent_geometry_code':
        summary = json.loads((root / 'llm-eval' / 'summary.json').read_text())
        _reseal(root / 'llm-eval', {**summary, 'code_sha256': {**summary['code_sha256'], 'geometry.py': 'c' * 64}})
    if kind == 'fid_unlisted_agent':  # An N-system FID whose O5 has no --agent O5=DIR cannot be bound.
        h.write_json(root / 'fid.json', {**json.loads((root / 'fid.json').read_text()), 'systems': ['J', 'P', 'O5']})
    if kind == 'judge_arms':  # N-system judge without L although L was rendered (judged_systems).
        records = [{**r, 'arm': {'joint': 'J', 'placement': 'P'}[r['arm']]} for r in _jsonl(root / 'judge' / 'records.jsonl')]
        write_rows(root / 'judge' / 'records.jsonl', records)
        h.write_json(root / 'judge' / 'summary.json', {**judge_batch.summarize(records, 'gpt-6.1-sol', len(records), ('J', 'P')),
                                                       'provenance': {'geometry_manifest_sha256': 'x'}})
    match = {'running_geometry': 'geometry-v2 used a geometry.py', 'agent_geometry_code': 'agent evaluation llm-eval',
             'fid_unlisted_agent': '--agent LABEL=DIR', 'judge_arms': 'judged system set'}[kind]
    with pytest.raises(ValueError, match=match):
        _build(root, runs, out='bad')
    assert not (root / 'bad').exists()


def test_p_label_is_amendment_006_verbatim(built):
    root, _, runs = built
    amendment = (Path(R.__file__).parents[2] / 'docs/research/20261013-unknown-size-comparison'
                 / 'amendment-006-agent-modes-budgets-20261013.md').read_text()
    assert f'- P label: "{R.P_RULE}".' in amendment
    _build(root, runs)
    assert R.P_LABEL in (root / 'report' / 'report.md').read_text()


def test_end_to_end_two_agents_and_n_system_visual_outputs(built, monkeypatch):
    """geometry-v2 + API agent L + open agent O5 + FID/judge written by the merged N-system visual code."""
    import numpy as np
    from PIL import Image
    from fastfill.comparison import fid as F
    from fastfill.comparison import llm_open as O
    from fastfill.comparison import render_assets as render
    from fastfill.comparison import visual_pipeline as V
    from fastfill.comparison.tests.test_llm_open import FakeLLM, argv, loader
    root, rooms, runs = built
    model = root / 'models' / O.SYSTEMS['O5']['model']
    model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    # O5 thinks; one first answer parses but hit the cap without EOS, so amendment 005 §3 fails that row.
    fake = FakeLLM(rooms, {'il-bed-0': [('<think>plan</think>{valid}', 'length')]}, default='<think>plan</think>{valid}')
    assert O.main(argv(root, 'O5', 'o5-run'), loader(fake)) == 0
    ev.evaluate(root / 'data', root / 'cohort.json', root / 'o5-run', root / 'o5-eval', runs, repeats=200)
    agents, labels, bed = {'L': root / 'llm-eval', 'O5': root / 'o5-eval'}, ['J', 'P', 'L', 'O5'], 'IL3D_3dfront/bedroom'
    geometry = root / 'geometry-v2'
    lineage = render.json_sha256(V.systems_lineage(V._systems({'J': geometry, 'P': geometry, **agents})))
    visual, cases = root / 'visual', []
    for i, uid in enumerate(('il-bed-0', 'il-bed-1', 'il-bed-2')):
        images = {}
        for j, arm in enumerate(['reference', *labels]):
            if (uid, arm) == ('il-bed-2', 'O5'):  # A failed O5 render: O5's population is smaller than J/P's.
                images[arm] = {'failure': 'render_failed'}
                continue
            (visual / uid).mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (512, 512), (40 * j, 60 * i, 90)).save(visual / uid / f'{arm}.png')
            images[arm] = {'path': f'{uid}/{arm}.png', 'sha256': h.sha256(visual / uid / f'{arm}.png')}
        cases.append({'uid': uid, 'stratum': bed, 'images': images})
    h.write_json(visual / 'fid-manifest.json', {'cases': cases, 'systems': labels, 'judged': labels,
                                                'provenance': {'geometry_manifest_sha256': lineage}})
    monkeypatch.setattr(F, 'verify_weights', lambda path: F.WEIGHTS_SHA256)
    features = lambda refs, **_: np.random.default_rng(0).normal(size=(len(refs), 2048))  # noqa: E731
    fid = F.run_manifest(visual / 'fid-manifest.json', root / 'fid-n.json', vendor=root, weights=root / 'w', extractor=features)
    h.write_json(root / 'fid-n.json', {**fid, 'extractor': 'pytorch_fid:0.3.0:pool3'})  # Inception mocked only.
    base = {'J': 8, 'P': 6, 'L': 7, 'O5': 5}
    records = [{'id': f'{u}-{a}', 'uid': u, 'arm': a, 'stratum': s, 'group': None, 'launched': True, 'status': 'completed',
                'scores': {'functionality_score': base[a] + (u == 'il-bed-1' and a == 'J'), 'layout_score': 7,
                           'aesthetics_score': 9, 'overall_score': 8}}
               for u, s in (('il-bed-0', bed), ('il-bed-1', bed), ('il-bed-2', bed), ('holo-k-0', 'OptiScene_holodeck/kitchen'))
               for a in labels]
    (root / 'judge-n').mkdir()
    write_rows(root / 'judge-n' / 'records.jsonl', records)
    judged = judge_batch.summarize(records, 'gpt-6.1-sol', len(records), tuple(labels))
    h.write_json(root / 'judge-n' / 'summary.json', {**judged, 'provenance': {'geometry_manifest_sha256': lineage}})

    result = _build(root, runs, out='report-n', agents=agents, fid=root / 'fid-n.json', judge=root / 'judge-n')
    t = result['tables']
    assert result['visual_mode'] == {'fid': 'systems', 'judge': 'systems'}
    assert '评审系统集合为预先指定' in (root / 'report-n' / 'report.md').read_text()
    t4 = {(r['stratum'], r['system']): r for r in t['T4']}
    # Amendment 011 populations: J/P on reference∩J∩P; each agent on reference∩J∩P∩agent with J/P on its rooms.
    pops = fid['strata'][bed]['populations']
    for label, key, population, n in (('J', 'J', 'primary', 3), ('P', 'P', 'primary', 3), ('L-harness', 'L', 'agent/L', 3),
                                      ('O5-harness', 'O5', 'agent/O5', 2)):
        row, own = t4[bed, label], pops[population]
        assert row['fid'] == own['fid'][key] and row['kid'] == own['kid'][key]['kid'] and row['fid_n'] == n == own['n']
        assert row['fid_flag'] == 'N<50' and row['kid_se'] is not None
        if population != 'primary':
            assert (row['fid_J_same_rooms'], row['fid_P_same_rooms']) == (own['fid']['J'], own['fid']['P'])
        assert row['gpt'] == pytest.approx(statistics.mean([base[key] + (key == 'J') / 3, 7, 9])) and row['gpt_success'] == 3
    assert t4[bed, 'L-raw']['fid'] is None and t4[bed, 'O5-raw']['gpt'] is None  # Only harness answers are rendered.
    assert t4['OptiScene_holodeck/kitchen', 'O5-harness']['gpt'] == pytest.approx(7)
    endpoint = judged['endpoint']
    assert t['T4_gpt'][0]['delta'] == {**t['T4_gpt'][0]['delta'], 'estimate': endpoint['mean_difference'],
                                       'ci95': endpoint['ci95'], 'n': endpoint['both_complete']}
    arms = {r['system']: r['mean'] for r in t['T4_gpt_arms']}
    assert arms == pytest.approx({k: statistics.mean([base[k], 7, 9]) + (k == 'J') / 12 for k in base})  # 4 rooms.
    contrast = {r['comparison']: r['delta'] for r in t['T4_gpt_contrasts']}
    assert list(contrast) == ['P−L', 'P−O5', 'J−P'] and contrast['J−P'] == t['T4_gpt'][0]['delta']
    # Functionality differs by the base only: room means differ by base / 3 in every room.
    assert contrast['P−L']['estimate'] == pytest.approx(-1 / 3) and contrast['P−L']['ci95'] == pytest.approx([-1 / 3] * 2)
    assert contrast['P−O5']['estimate'] == pytest.approx(1 / 3) and contrast['P−O5']['n'] == 4
    cost = {r['system']: r for r in t['T5_cost']}
    assert cost['O5']['gpu_hours'] == json.loads((root / 'o5-run' / 'cost.json').read_text())['gpu_hours']
    assert cost['O5']['reasoning'] == O.SYSTEMS['O5']['effort'] and cost['L']['model'] == L.EXPECTED_MODEL
    assert all(cost[k][f] is not None for k in ('L', 'O5') for f in ('calls', 'input_tokens', 'output_tokens', 'latency_p50_s'))
    t6 = {r['system']: r for r in t['T6']}
    assert (t6['O5-harness']['token_cap_without_eos'], t6['O5-harness']['cap_rule_flipped_complete']) == (1, 1)
    assert 'token_cap_without_eos' not in t6['L-harness'] and result['agents']['O5']['cap_rule_per_row']
    assert 'agent_answers_reached_output_cap_not_row_applied' not in result['warnings']
    t5 = {(r['scope'], r['system']): r for r in t['T5']}
    assert t5['cohort', 'O5-harness']['complete_rate'] == pytest.approx(35 / 36)
    assert {r['comparison'] for r in t['T5_contrasts']} >= {'J−O5-harness', 'P−L-harness'}
    # Every rendered cell with an input is filled: T4 rows of the rendered systems, T5 and its contrasts.
    report = (root / 'report-n' / 'report.md').read_text()
    lines = report.splitlines()
    for label in ('J', 'P', 'L-harness', 'O5-harness'):  # J/P have no same-room agent columns.
        assert next(x for x in lines if x.startswith(f'| {bed} | {label} |')).count('—') == 2 * (label in 'JP')
    populations = {(r['stratum'], r['population'], r['system']): r for r in t['T4_fid']}
    assert populations[bed, 'judged', 'O5']['n'] == 2 and populations[bed, 'judged', 'O5']['role'].startswith('次要')
    assert populations['pooled', 'primary', 'J']['n'] == 3 and populations['pooled', 'agent/O5', 'P']['n'] == 2
    assert '### T4-FID 修订 011 的 FID/KID 人群' in report and '参考∩J∩P∩该代理' in report
    for scope in ('cohort', 'strata_pooled'):
        for label in ('J', 'P', 'L-raw', 'L-harness', 'O5-raw', 'O5-harness'):
            assert '—' not in next(x for x in lines if x.startswith(f'| {scope} | {label} |'))
        assert all('—' not in x for x in lines if x.startswith(f'| {scope} | J−'))
    assert 'KID 为配对 U 统计量' in report and '（N 系统裁判运行自身的 summary.endpoint）' in report


@pytest.mark.parametrize('kind', ['reference_data', 'geometry_code', 'fid_lineage', 'agent_runs'])
def test_refuses_unbound_inputs(built, kind):
    root, _, runs = built
    if kind in ('reference_data', 'geometry_code'):
        value = json.loads((root / 'reference.json').read_text())
        if kind == 'reference_data':
            value['data_manifest_sha256'] = 'b' * 64
        else:
            value['code_sha256']['geometry.py'] = 'c' * 64
        (root / 'reference.json').unlink()
        infer.write_once(root / 'reference.json', value)
    elif kind == 'fid_lineage':
        value = json.loads((root / 'fid.json').read_text())
        value['provenance']['provenance']['geometry_manifest_sha256'] = 'd' * 64
        h.write_json(root / 'fid.json', value)
    else:
        path = root / 'llm-eval' / 'summary.json'
        value = json.loads(path.read_text())
        value['inference_run_sha256']['joint'] = 'e' * 64
        h.write_json(path, value)
        manifest = json.loads((root / 'llm-eval' / 'MANIFEST.json').read_text())
        h.write_json(root / 'llm-eval' / 'MANIFEST.json', {'files': {**manifest['files'], 'summary.json': h.sha256(path)}})
    with pytest.raises(ValueError):
        _build(root, runs, out='bad')
    assert not (root / 'bad').exists()


def test_contrasts_reuse_paired_intervals_design_and_draws():
    rng = random.Random(7)
    joint, placement = [], []
    for i in range(80):
        base = {'uid': f'u{i}', 'source': 'ABC'[i % 3], 'group': f'g{i // 3}'}
        for rows in (joint, placement):
            complete = rng.random() > .15
            rows.append({**base, **{k: (rng.random() if complete else None) for k in g.SCALARS},
                         'complete': complete, 'geometry_success': complete and rng.random() > .3,
                         'geometry_success_repaired': complete})
    ours = R.contrasts(joint, placement, ('geometry_success', 'size_log_mae'), 300)
    theirs = g.paired_intervals(joint, placement, repeats=300)
    for field in ours:
        assert ours[field]['estimate'] == theirs[field]['estimate_joint_minus_placement']
        assert ours[field]['ci95'] == theirs[field]['ci95'] and ours[field]['n'] == theirs[field]['paired_rooms']
        assert 0 <= ours[field]['p'] <= 1


def test_delivery_rule_and_holm():
    good = {'estimate': -.1, 'ci95': [-.2, -.01]}
    assert R.delivery(good, {'estimate': 0., 'ci95': [-.02, .01]})['delivered'] == 'J'
    for b, a1 in ((good, {'estimate': 0., 'ci95': [-.0201, .01]}), ({'estimate': -.1, 'ci95': [-.2, .01]}, {'ci95': [0., .1]}),
                  ({'estimate': .1, 'ci95': [.05, .2]}, {'ci95': [0., .1]}), ({'estimate': None, 'ci95': None}, {'ci95': [0., .1]}),
                  (good, {'estimate': None, 'ci95': None})):
        assert R.delivery(b, a1)['delivered'] == 'P'
    assert R.holm([.01, .04]) == [.02, .04] and R.holm([.03, .02]) == [.04, .04] and R.holm([None, .01]) == [None, None]


def test_rotated_oor_uses_rotated_footprints():
    square = {'size': [1, 1, 1], 'pos': [2, 2, 0], 'yaw': 0.}
    pair = [square, {**square, 'yaw': math.pi / 4}]
    assert R.rotated_oor(pair) == pytest.approx((2 * (math.sqrt(2) - 1)) / 2)
    room = {'objects': [{**o, 'id': str(i), 'category': 'box'} for i, o in enumerate(pair)]}
    assert check(room)['oor'] == pytest.approx(.5)
    assert R.rotated_oor([square, {**square, 'pos': [2.5, 2, 0]}]) == pytest.approx(.25)


def test_reference_prior_levels_exact_with_frozen_prior():
    prior = {'min_count': 5, 'global': {'count': 50, 'size': [.4, .5, .65]},
             'category_room_type': {'chair': {'bedroom': {'count': 6, 'size': [.5, .5, .9]},
                                              'kitchen': {'count': 2, 'size': [.3, .3, .3]}}},
             'category': {'chair': {'count': 8, 'size': [.45, .5, .85]}, 'lamp': {'count': 1, 'size': [.2, .2, .2]}}}
    assert RR._level({'category': 'chair', 'size': [.5, .5, .9]}, 'bedroom', prior) == 'P'
    assert RR._level({'category': 'chair', 'size': [.45, .5, .85]}, 'kitchen', prior) == 'C'
    assert RR._level({'category': 'lamp', 'size': [.4, .5, .65]}, 'bedroom', prior) == 'G'
    assert RR._level({'category': 'lamp', 'size': [.4, .5, .65]}, 'bedroom', None) == 'G'
    assert RR._level({'category': 'chair', 'size': [.5, .5, .9]}, 'bedroom', None) == 'N'
    with pytest.raises(ValueError, match='reproduce'):
        RR._level({'category': 'chair', 'size': [.5, .5, .91]}, 'bedroom', prior)


def test_reference_load_rebuilds_rooms_and_checks_hashes(tmp_path):
    raw = {'uid': 'r1', 'source': 'S', 'group': 'house', 'room_type': 'bedroom', 'boundary_type': 'polygon',
           'boundary': [[0, 0], [5, 0], [5, 5], [0, 5]], 'height': 3, 'split': 'test',
           'objects': [{'id': 'a', 'category': 'chair', 'size': [1, 1, 1], 'pos': [1, 1, 0], 'yaw': 0}], 'fixed': []}
    with gzip.open(tmp_path / 'export.jsonl.gz', 'wt') as stream:
        stream.write(json.dumps(raw) + '\n')
    bound, _ = bind_room(raw, seed=RR.SEED)
    room = {**bound, 'group': 'c1', 'component': 'c1', 'split': 'test'}
    meta = {'uid': 'r1', 'source': 'S', 'group': 'c1', 'component': 'c1', 'split': 'test', 'room_type': 'bedroom'}
    sizes = {o['id']: [.4, .5, .65] for o in room['objects']}
    write_rows(tmp_path / 'j.jsonl', [{**meta, 'messages': messages(room, 'joint')[:2]}])
    write_rows(tmp_path / 'p.jsonl', [{**meta, 'messages': messages(room, 'placement', estimated_sizes=sizes)[:2]}])
    (tmp_path / 'rooms.jsonl').write_text(json.dumps(room, ensure_ascii=False) + '\n')
    files = {'joint/requests.jsonl': h.sha256(tmp_path / 'j.jsonl'), 'placement/requests.jsonl': h.sha256(tmp_path / 'p.jsonl'),
             'rooms.jsonl': h.sha256(tmp_path / 'rooms.jsonl')}
    h.write_json(tmp_path / 'MANIFEST.json', {'expected_uids': ['r1'], 'files': files})
    rooms, requests, _ = RR.load(tmp_path / 'export.jsonl.gz', tmp_path / 'j.jsonl', tmp_path / 'p.jsonl', tmp_path / 'MANIFEST.json')
    reference = RR.build_reference(rooms, requests, 'f' * 64, [])
    assert reference['prior_levels'] == {'r1': 'G'} and reference['scopes']['all']['prior']['global_fallback_object_rate'] == 1
    h.write_json(tmp_path / 'MANIFEST.json', {'expected_uids': ['r1'], 'files': {**files, 'rooms.jsonl': '0' * 64}})
    with pytest.raises(ValueError, match='rooms'):
        RR.load(tmp_path / 'export.jsonl.gz', tmp_path / 'j.jsonl', tmp_path / 'p.jsonl', tmp_path / 'MANIFEST.json')
