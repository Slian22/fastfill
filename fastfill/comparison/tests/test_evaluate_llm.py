"""Agent reports name the system, its model and the paired J/P stage (audit minor 17); fake vLLM, no GPU."""
import json

from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import infer
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
