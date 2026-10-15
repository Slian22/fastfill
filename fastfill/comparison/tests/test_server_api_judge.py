"""Amendment-023 options of the judge client; unset they keep the pinned payload."""
import importlib.util
import json

from fastfill.comparison import server_api_judge as judge
from fastfill.comparison.tests.test_judge_batch import CLIENT, bundle


def test_payload_unchanged_by_default_and_reasoning_when_set(tmp_path):
    root = bundle(tmp_path) / 'api-bundle'
    case = json.loads((root / 'manifest.json').read_text())['cases'][0]
    pinned = importlib.util.spec_from_file_location('pinned_judge', CLIENT)
    old = importlib.util.module_from_spec(pinned)
    pinned.loader.exec_module(old)
    config = judge.Config('k', 'https://example.invalid', 'gpt-6.1-sol')
    assert judge.prepare_case(case, root, config) == old.prepare_case(
        case, root, old.Config('k', 'https://example.invalid', 'gpt-6.1-sol'))
    _, body = judge.prepare_case(case, root, judge.Config('k', 'https://example.invalid', 'gpt-6-astra', 'high'))
    payload = json.loads(body)
    assert payload['reasoning'] == {'effort': 'high'} and payload['max_output_tokens'] == 32768
    assert payload['model'] == 'gpt-6-astra'


def test_model_override_replaces_env_model_only(tmp_path, monkeypatch):
    root = bundle(tmp_path) / 'api-bundle'
    manifest = json.loads((root / 'manifest.json').read_text())
    (tmp_path / 'm.json').write_text(json.dumps({**manifest, 'expected_model': 'gpt-6-astra',
        'cases': [{**c, 'prompt_file': str(root / c['prompt_file']),
                   'images': [{**i, 'path': str(root / i['path'])} for i in c['images']]}
                  for c in manifest['cases'][:1]]}))
    monkeypatch.setattr(judge, 'load_config', lambda _: judge.Config('k', 'https://example.invalid', 'gpt-6.1-sol'))
    seen = []

    def call(identity, body, config, timeout):
        seen.append((config.model, config.reasoning_effort, json.loads(body)['model']))
        return {**identity, 'status': 'failed', 'error': 'offline'}
    monkeypatch.setattr(judge, 'call_judge', call)
    judge.evaluate(tmp_path / 'm.json', tmp_path / 'r', 'unused', 1, 5, 'gpt-6-astra', 'high')
    assert seen == [('gpt-6-astra', 'high', 'gpt-6-astra')]
    try:
        judge.evaluate(tmp_path / 'm.json', tmp_path / 'r2', 'unused', 1, 5)
    except judge.JudgeError as exc:
        assert str(exc) == 'configured_model_mismatch'
    else:
        raise AssertionError('env model must still be checked without --model')


def test_amendment_025_defaults_match_pinned_client_end_to_end(tmp_path, monkeypatch):
    """Config.instructions/parse unset: same request body and stored judge record as the run-002 client."""
    from fastfill.comparison.tests.test_critic_client_025 import Response, sse
    root = bundle(tmp_path) / 'api-bundle'
    manifest = json.loads((root / 'manifest.json').read_text())
    (tmp_path / 'm.json').write_text(json.dumps({**manifest, 'cases': [
        {**c, 'prompt_file': str(root / c['prompt_file']),
         'images': [{**i, 'path': str(root / i['path'])} for i in c['images']]} for c in manifest['cases'][:1]]}))
    spec = importlib.util.spec_from_file_location('pinned_judge_025', CLIENT)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    text = json.dumps({'functionality_score': 7, 'layout_score': 6, 'aesthetics_score': 5})
    usage_free = sse(text).replace(b', "output_tokens_details": {"reasoning_tokens": 250}', b'')
    bodies, records = [], []
    for module in (old, judge):
        monkeypatch.setattr(module, 'load_config',
                            lambda _, m=module: m.Config('k', 'https://example.invalid', 'gpt-6.1-sol'))
        monkeypatch.setattr(module, 'open_response', lambda request, _t: bodies.append(request.data) or Response(usage_free))
        out = tmp_path / module.__name__
        (record,) = module.evaluate(tmp_path / 'm.json', out, 'unused', 1, 5)
        records.append({k: v for k, v in record.items() if k != 'finished_at'})
        assert module.evaluate(tmp_path / 'm.json', out, 'unused', 1, 5)[0] == record  # reused, no second call
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert records[0] == records[1] and records[1]['scores']['layout_score'] == 6
