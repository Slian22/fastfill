"""Amendment 026 (+ addendum A): frontier_026 on synthetic rows (choice rule, escalation sets, cost attribution,
contrast sign, a scope-C panel through to the written report)."""
import json

import pytest

from fastfill.comparison import frontier_026 as f
from fastfill.comparison.best_of_n import VIOLATIONS


def row(uid, checker=True, strict=True, noreq=True, viol=0, complete=True, source='IL3D_3dfront'):
    geo = {**dict.fromkeys(VIOLATIONS, 0), 'main_collisions': viol, 'oor': .1}
    flags = lambda stage: {f'geometry_success_{stage}': checker, f'strict_ok_{stage}': strict,
                           f'endpoint_ok_{stage}': noreq, f'attach_ok_{stage}': True,
                           f'admit_no_required_ok_{stage}': True}
    return {'uid': uid, 'source': source, 'group': 'g' + uid, 'complete': complete, 'oor': .1, 'size_log_mae': .2,
            'size_iou': .5, 'raw_layout': [{'size': [1, 1, 1]}],
            's1': {**flags('raw'), **flags('s1'), 'geometry_s1': geo}}


def room(uid, room_type='bedroom', source='IL3D_3dfront'):
    return {'uid': uid, 'source': source, 'group': 'g' + uid, 'room_type': room_type, 'objects': [{'size': [1, 1, 1]}]}


def views(*rows):
    return [f.view(x, room(x['uid'], source=x['source'])) for x in rows]


def test_choice_rule_is_lexicographic_and_ties_keep_local():
    local = views(row('a', checker=False, strict=True, noreq=True), row('b', False, False, False, viol=3),
                  row('c', False, False, False, viol=1), row('d', False, False, False, viol=2))
    big = views(row('a', True, False, False), row('b', False, False, False, viol=1),
                row('c', complete=False), row('d', False, False, False, viol=2))
    out = f.simulate('ESC-CHECK', local, big)
    assert [esc for _, esc, _ in out] == [True] * 4
    # checker beats strict + required; fewer violations wins; an incomplete answer ranks lowest; a tie keeps local
    assert [used for _, _, used in out] == [True, True, False, False]
    assert out[0][0] is big[0] and out[2][0] is local[2] and out[3][0] is local[3]
    assert f.rank(big[2])[3] == float('-inf')


def test_policy_escalation_sets():
    local = views(row('pass'), row('noreq', noreq=False), row('strict', strict=False), row('check', checker=False))
    big = views(*(row(v['uid']) for v in local))
    escalated = {p: [v['uid'] for v, (_, esc, _) in zip(local, f.simulate(p, local, big)) if esc] for p in f.POLICIES}
    assert escalated == {'LOCAL': [], 'ESC-CHECK': ['check'], 'ESC-STRICT': ['strict', 'check'],
                         'ESC-ANY': ['noreq', 'strict', 'check'], 'BIG': ['pass', 'noreq', 'strict', 'check']}
    assert all(used for _, _, used in f.simulate('BIG', local, big))
    assert not any(used for _, _, used in f.simulate('LOCAL', local, big))


def test_cost_is_paid_for_every_escalated_room():
    usage = {'a': {'calls': 1, 'output_tokens': 100}, 'b': {'calls': 3, 'output_tokens': 50}, 'c': {'calls': 1,
                                                                                                   'output_tokens': 0}}
    cost = f.policy_cost({'a', 'b'}, usage, 4, f.LOCAL_GPU_S)
    assert cost['big'] == {'calls': 1.0, 'output_tokens': 37.5} and cost['big_total'] == {'calls': 4, 'output_tokens': 150}
    assert cost['local_gpu_s'] == pytest.approx(59 * 60 / 8647)
    assert f.policy_cost({'a'}, None, 4, 0.)['big'] == f.NOT_MEASURED


def _o2_run(tmp_path, latency_b=5.0):
    run = tmp_path / 'o2'
    (run / 'receipts').mkdir(parents=True)
    (run / 'result.json').write_text(json.dumps({'run_sha256': 'R'}))
    (run / 'cost.json').write_text(json.dumps({'gpus': 2, 'slurm_job_ids': ['1'], 'wall_seconds': 20.,
                                               'generation_seconds': 14.}))
    calls = [{'event': 'start', 'slurm_job_id': '0'}, {'event': 'start', 'slurm_job_id': '1'},
             {'event': 'call', 'slurm_job_id': '1', 'round': 0, 'requests': 2, 'wall_seconds': 10.},
             {'event': 'call', 'slurm_job_id': '1', 'round': 1, 'requests': 1, 'wall_seconds': 4.}]
    (run / 'invocations.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in calls))
    answer = lambda latency, out, valid: {'latency_seconds': latency, 'schema': {'valid': valid},
                                          'usage': {'input_tokens': 10, 'output_tokens': out, 'reasoning_tokens': out}}
    for uid, answers in (('a', [answer(5., 300, False), answer(4., 50, True)]), ('b', [answer(latency_b, 100, True)])):
        (run / 'receipts' / f'{uid}.json').write_text(json.dumps({'uid': uid, 'run_sha256': 'R', 'answers': answers}))
    return run


def test_o2_wall_time_is_split_by_output_tokens_within_each_call(tmp_path):
    run = _o2_run(tmp_path)
    harness, meta = f.receipt_usage(run, 'harness', wall=True)
    assert harness['a']['wall_s'] == pytest.approx(10 * 300 / 400 + 4) and harness['b']['wall_s'] == pytest.approx(2.5)
    assert sum(x['wall_s'] for x in harness.values()) == pytest.approx(14.)      # all generation time, nothing else
    assert harness['a']['wall_equal_s'] == pytest.approx(9.) and harness['a']['gpu_s'] == pytest.approx(23.)
    assert harness['a']['calls'] == 2 and harness['a']['output_tokens'] == 350
    assert meta['unattributed_s'] == 6. and meta['valid'] == {'a': True, 'b': True}
    assert meta['jobs_without_calls'] == ['0']          # a job that never finished a call (its time is not in cost)
    raw, meta = f.receipt_usage(run, 'raw', wall=True)
    assert raw['a']['wall_s'] == pytest.approx(7.5) and raw['a']['calls'] == 1 and raw['a']['output_tokens'] == 300
    assert meta['valid'] == {'a': False, 'b': True}


def test_o2_attribution_refuses_an_answer_without_its_call(tmp_path):
    with pytest.raises(ValueError, match='matches no call'):
        f.receipt_usage(_o2_run(tmp_path, latency_b=4.5), 'harness', wall=True)


def test_run_totals_split_equally_with_exact_calls(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps({'llm': {'result': {
        'status': 'LLM_COMPLETE', 'run_sha256': 'L', 'rooms': 2, 'api_failed_rooms': 0, 'calls_ledger': 2,
        'rounds_histogram': {'1': 2}, 'tokens': {'input_tokens': 10, 'output_tokens': 4, 'reasoning_tokens': 2}}}}))
    usage, meta = f.summary_usage(path, ['a', 'b'])
    assert usage['a'] == {'calls': 1., 'input_tokens': 5., 'output_tokens': 2., 'reasoning_tokens': 1.}
    assert meta['basis'] == 'run_mean' and meta['calls_exact']
    with pytest.raises(ValueError):
        f.summary_usage(path, ['a'])


def test_known_rooms_keep_their_tokens_and_the_rest_share_the_remainder(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps({'llm': {'result': {
        'status': 'LLM_COMPLETE', 'run_sha256': 'L', 'rooms': 3, 'api_failed_rooms': 0, 'calls_ledger': 3,
        'rounds_histogram': {'1': 3}, 'tokens': {'input_tokens': 90, 'output_tokens': 30, 'reasoning_tokens': 12}}}}))
    known = {'a': {'input_tokens': 40, 'output_tokens': 20, 'reasoning_tokens': 6}, 'z': {}}   # z: not a panel room
    usage, meta = f.summary_usage(path, ['a', 'b', 'c'], known)
    assert usage['a'] == {'calls': 1., **known['a']} and meta['known_rooms'] == 1
    assert usage['b'] == usage['c'] == {'calls': 1., 'input_tokens': 25., 'output_tokens': 5., 'reasoning_tokens': 3.}
    assert {k: sum(x[k] for x in usage.values()) for k in f.TOKENS} == {'input_tokens': 90, 'output_tokens': 30,
                                                                        'reasoning_tokens': 12}   # BIG = run total
    with pytest.raises(ValueError, match='exceed'):
        f.summary_usage(path, ['a', 'b', 'c'], {'a': {'input_tokens': 91, 'output_tokens': 0, 'reasoning_tokens': 0}})


def test_estimate_marker_only_on_strict_subsets_and_o2_fallback_columns():
    cost = lambda calls, tokens: {'local_gpu_s': .4, 'big': {'calls': calls, **dict.fromkeys(f.TOKENS, tokens)}}
    p = {'cost_kind': 'api', 'rooms': 4, 'usage': {'basis': 'run_mean', 'calls_exact': True},
         'entries': {'LOCAL': {'escalated': 0, 'cost': cost(0., 0.)}, 'ESC-CHECK': {'escalated': 1, 'cost': cost(.25, 9.)},
                     'BIG': {'escalated': 4, 'cost': cost(1., 30.)},
                     f.SYSTEM: {'escalated': 1, 'cost': {**cost(.5, 9.), 'o2': {'output_tokens': 700., 'gpu_s': 2.5}}}}}
    assert f._cost_cells(p, 'ESC-CHECK') == ['0.250', '≈9', '≈9', '≈9', '0.400']     # calls exact, tokens estimated
    assert f._cost_cells(p, 'BIG') == ['1.000', '30', '30', '30', '0.400']           # all rooms = measured run total
    assert f._cost_cells(p, 'LOCAL', o2=True)[-3:] == ['0', '0.0', '0.400']
    assert f._cost_cells(p, f.SYSTEM, o2=True) == ['0.500', '9', '9', '9', '700', '2.5', '0.400']


def test_scope_c_panel_contrast_sign_and_report(tmp_path):
    """Scope C (addendum A) from path-free synthetic rows: garage rooms fail locally, the big model passes all."""
    rooms, local, big = {}, [], []
    for i in range(24):
        family, source = ('garage', 'MultiScan') if i % 2 else ('kitchen', 'IL3D_3dfront')
        uid = f'r{i:02d}'
        rooms[uid] = room(uid, family, source)
        local.append(row(uid, checker=family == 'kitchen', source=source))
        big.append(row(uid, source=source))
    usage = {u: {'calls': 2, 'input_tokens': 10, 'output_tokens': 5, 'reasoning_tokens': 1} for u in rooms}
    p = f.panel('C', 'C', 'L size-fixed harness (gpt-6.1-sol)', 'api', rooms, local, big, usage,
                {'basis': 'per-room receipts', 'run_sha256': 'x' * 64}, repeats=50)
    garage, kitchen = p['levels']['family:garage'], p['levels']['family:kitchen']
    assert garage['LOCAL']['checker_rate'] == 0 and garage['BIG']['checker_rate'] == 1
    assert kitchen['LOCAL']['checker_rate'] == 1 and garage['ESC-CHECK']['escalated_share'] == 1
    assert p['entries']['ESC-CHECK']['escalated'] == 12 and p['entries']['ESC-CHECK']['cost']['big']['calls'] == 1.
    delta = lambda entry, subset: f._contrast(p, entry, subset)['checker']['estimate']
    assert delta('LOCAL', 'family:garage') == -1 and delta('LOCAL', 'all') < 0      # policy - BIG: worse -> negative
    assert delta('ESC-CHECK', 'all') == 0 and delta('LOCAL', 'family:kitchen') == 0
    result = {'status': 'FRONTIER_026_COMPLETE', 'panels': [p], 'settings': {'repeats': 50, 'seed': 42},
              'missing': []}
    f.write(tmp_path / 'out', result, tmp_path / 'results')
    text = (tmp_path / 'results' / 'report.md').read_text()
    assert '范围 C' in text and '族：garage' in text and (tmp_path / 'out' / 'frontier.png').stat().st_size > 0
    with pytest.raises(FileExistsError):
        f.write(tmp_path / 'out', result)


def test_scope_c_needs_both_paths():
    base = ['--run-root', 'R', '--data', 'D', '--pilot', 'P', '--cohort', 'C', '--out', 'O']
    with pytest.raises(SystemExit):
        f.parse_args([*base, '--uids-c', 'U'])
    assert f.parse_args([*base, '--uids-c', 'U', '--big-c-dir', 'B']).big_c_dir.name == 'B'
