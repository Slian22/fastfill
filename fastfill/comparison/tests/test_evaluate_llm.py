"""Agent reports name the system, its model and the paired J/P stage (audit minor 17); fake vLLM, no GPU."""
import json

import pytest

from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import llm_open as O
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_llm_agent as base
from fastfill.comparison.tests.test_llm_open import FakeLLM, argv, loader, world  # noqa: F401 (fixture)


def _inference(root, arm, rooms, **extra):
    """test_llm_agent._inference plus receipt fields (a DPO stage run records stage and lineage_sha256)."""
    data = root / 'data'
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    run = root / ('infer-' + arm)
    run.mkdir()
    receipt = {'data': {'manifest_sha256': h.sha256(data / 'MANIFEST.json')}, 'training_data': {'manifest_sha256': 'a' * 64},
               'base_sha256': 'base', 'config': {'arm': arm, 'lora': 'lora-' + arm}, **extra}
    receipt['run_sha256'] = infer._hash(receipt)
    h.write_json(run / 'run_receipt.json', receipt)
    base.write_rows(run / 'records.jsonl', [
        {'uid': u, 'source': rooms[u]['source'], 'group': rooms[u]['group'], 'arm': arm,
         'raw': base.target(None, arm, rooms[u]), 'error': None, 'run_sha256': receipt['run_sha256']}
        for u in manifest['expected_uids']])
    h.write_json(run / 'result.json', {'status': 'INFERENCE_COMPLETE', 'pending_count': 0})
    h.write_json(run / 'MANIFEST.json', {'expected_uids': manifest['expected_uids'], 'files': h.artifact_manifest(run)})
    return run, receipt['run_sha256']


def test_open_model_report_labels_rows_by_system_and_states_the_paired_stage(world):
    root, rooms = world
    assert O.main(argv(root, 'O2', 'run'), loader(FakeLLM(rooms, {'y-0': ['bad', '{valid}']}))) == 0
    joint, j_sha = _inference(root, 'joint', rooms)
    placement, p_sha = _inference(root, 'placement', rooms, stage='dpo1', lineage_sha256='c' * 64)
    summary = ev.evaluate(root / 'data', root / 'cohort.json', root / 'run', root / 'eval',
                          {'joint': joint, 'placement': placement}, repeats=20)
    model = O.SYSTEMS['O2']['name']
    assert (summary['system'], summary['model']) == ('O2', model)
    assert summary['paired_stage'] == {
        'joint': {'stage': 'sft', 'run_sha256': j_sha, 'lineage_sha256': None, 'lora': 'lora-joint'},
        'placement': {'stage': 'dpo1', 'run_sha256': p_sha, 'lineage_sha256': 'c' * 64, 'lora': 'lora-placement'}}
    report = (root / 'eval' / 'report.md').read_text()
    assert report.startswith(f'# 未知尺寸：LLM 代理 O2（{model}）与 J／P 同输入对比')
    assert '| O2-raw |' in report and '| O2-harness |' in report and 'J − O2-raw' in report and 'L-raw' not in report
    assert f'J = sft（推理 run {j_sha[:12]}）' in report and f'P = dpo1（推理 run {p_sha[:12]}，谱系 cccccccccccc）' in report
    assert '## 表 3：O2 调用统计' in report and '重试消息只含异常类名' in report and '解析器错误文本。' not in report
    assert ev.labels('L') == ev.LABELS  # The API agent keeps L-raw / L-harness.


def test_placement_arm_agent_is_scored_as_p_on_the_p_request_after_size_enforcement(world):
    """Amendment 016 section 2 + addendum B2: sizes come from P's request (returned sizes are overwritten and counted
    as unauthorised changes); valid answers then get exactly P's scores."""
    root, rooms = world
    joint_answer = base.target(None, 'joint', rooms['y-0'])  # GT sizes: differ from the prior by 0.1 m.
    script = {'y-0': [joint_answer], 'il-bed-1': ['bad', '{valid}'], 'il-bed-2': ['bad'] * 3}
    assert O.main(argv(root, 'O2', 'run', '--arm', 'placement'), loader(FakeLLM(rooms, script, arm='placement'))) == 0
    joint, _ = _inference(root, 'joint', rooms)
    placement, p_sha = _inference(root, 'placement', rooms, stage='dpo1')
    summary = ev.evaluate(root / 'data', root / 'cohort.json', root / 'run', root / 'eval',
                          {'joint': joint, 'placement': placement}, repeats=20)
    assert summary['llm']['config']['request_arm'] == 'placement'
    assert summary['paired_stage']['placement']['run_sha256'] == p_sha
    assert summary['metric_contract']['llm_scoring'] == 'evaluate_pair._score_placement_semantics_on_the_P_request'
    assert summary['arms']['llm_raw']['all']['complete_count'] == 34
    assert summary['arms']['llm_harness']['all']['complete_count'] == 35
    assert summary['size_enforcement']['llm_harness'] == {
        'answers': 35, 'answers_size_returned': 1, 'answers_size_changed': 1, 'answers_z_changed': 0,
        'objects': 70, 'size_returned': 2, 'size_changed': 2, 'z_changed': 0}
    scored = {name: {r['uid']: r for r in map(json.loads, (root / 'eval' / f'{name}-geometry.jsonl').read_text()
                                              .splitlines())} for name in ('placement', 'llm_raw', 'llm_harness')}
    assert not scored['llm_harness']['il-bed-2']['complete'] and scored['llm_harness']['il-bed-2']['error'] is not None
    assert not scored['llm_raw']['il-bed-1']['complete'] and scored['llm_harness']['il-bed-1']['complete']
    keys = ('complete', 'geometry_success', 'oor', 'size_iou', 'size_log_mae', 'raw_layout', 'geometry_success_repaired')
    for uid, row in scored['llm_harness'].items():  # The same placements as P on P's request: P's scores exactly.
        assert row['arm'] == 'llm_harness'
        if uid != 'il-bed-2':  # y-0 too: its GT sizes were replaced by the prior.
            assert {k: row[k] for k in keys} == {k: scored['placement'][uid][k] for k in keys}
    prior = [o['size'] for o in json.loads(rooms['y-0']['_placement']['messages'][1]['content'])['objects']]
    assert [o['size'] for o in scored['llm_harness']['y-0']['raw_layout']] == prior
    report = (root / 'eval' / 'report.md').read_text()
    assert '）固定尺寸重生成（P 的输入）与 J／P 对比' in report.splitlines()[0]
    assert 'O2 收到与 P 完全相同的 system 与 user 消息' in report and '附录 B2' in report
    assert '| O2-P-raw |' in report and '| O2-P-harness |' in report
    assert 'J − O2-P-raw' in report and '| O2-raw |' not in report
    assert '## 表 4：未授权的尺寸改动' in report and '| O2-P-harness | 35 | 1 | 1（0.0286） | 70 | 2（0.0286） | 0 |' in report


def test_subset_and_gt_size_runs_are_scored_on_their_rooms_and_requests(world):
    """--uids (amendment-016 pilot) restricts the agent, J and P; a --sizes gt run is scored on the GT-size request."""
    root, rooms = world
    cohort = json.loads((root / 'cohort.json').read_text())['uids']
    pick = [cohort[0], cohort[4], cohort[20], cohort[35]]
    (root / 'pilot.json').write_text(json.dumps(pick))
    for room in rooms.values():
        room['_gt'] = L.gt_request(room['_placement'], room)
    fake = FakeLLM(rooms, arm='placement')
    fake.key = '_gt'
    gt = ('--arm', 'placement', '--sizes', 'gt', '--uids', str(root / 'pilot.json'))
    assert O.main(argv(root, 'O2', 'gt', *gt), loader(fake)) == 0
    joint, _ = _inference(root, 'joint', rooms)
    placement, _ = _inference(root, 'placement', rooms)
    runs = {'joint': joint, 'placement': placement}
    with pytest.raises(ValueError, match='does not cover exactly the frozen cohort'):
        ev.evaluate(root / 'data', root / 'cohort.json', root / 'gt', root / 'e0', runs, repeats=20)
    summary = ev.evaluate(root / 'data', root / 'cohort.json', root / 'gt', root / 'eval', runs, repeats=20,
                          uids_path=root / 'pilot.json')
    assert summary['cohort']['rooms'] == 36 and summary['cohort']['subset'] == {
        'rooms': 4, 'sha256': h.sha256(root / 'pilot.json'), 'uids_sha256': L._digest(json.dumps(pick))}
    assert {n: a['all']['requests'] for n, a in summary['arms'].items()} == dict.fromkeys(
        ('joint', 'placement', 'llm_raw', 'llm_harness'), 4)
    assert summary['metric_contract']['llm_scoring'].endswith('_on_the_P_request_with_GT_sizes')
    harness = {r['uid']: r for r in map(json.loads, (root / 'eval/llm_harness-geometry.jsonl').read_text().splitlines())}
    assert list(harness) == pick
    for uid, row in harness.items():  # GT sizes and GT placements: the GT layout, size error 0.
        assert row['complete'] and row['size_log_mae'] == 0 and [o['size'] for o in row['raw_layout']] == [
            o['size'] for o in rooms[uid]['objects']]
    assert summary['size_enforcement']['llm_harness']['size_returned'] == 0
    report = (root / 'eval/report.md').read_text()
    assert '| O2-PGT-harness |' in report and '队列子集：4 个房间' in report and 'GT 尺寸' in report.splitlines()[0]
