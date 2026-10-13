"""Open-weight agent runner (amendments 003/004); vLLM is replaced by a fake LLM object (no GPU, no network)."""
import copy
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import llm_open as O
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_llm_agent as base


class FakeTokenizer:
    """Prompt = base tokens + one token per character of every turn after the first request turns."""
    def __init__(self, base):
        self.base, self.kwargs = base, []

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=True, **kwargs):
        assert add_generation_prompt and not tokenize and tools is None
        self.kwargs.append(kwargs)
        return 'p' * self.length(messages)

    def length(self, messages):
        return self.base + sum(len(m['content']) for m in messages[3:])

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [0] * len(text)


class FakeLLM:
    """vLLM LLM stand-in: per-UID scripted texts ('{valid}' = the GT answer), or (text, finish_reason)."""
    def __init__(self, rooms, script=None, default='{valid}', prompt_tokens=100, fail_on_call=None):
        self.rooms, self.script, self.default = rooms, script or {}, default
        self.tok, self.fail_on_call, self.calls = FakeTokenizer(prompt_tokens), fail_on_call, []

    def get_tokenizer(self):
        return self.tok

    def uid(self, conversation):
        matches = [uid for uid, room in self.rooms.items() if conversation[:2] == room['_request']['messages']]
        assert len(matches) == 1
        return matches[0]

    def chat(self, messages, sampling_params, use_tqdm=True, chat_template_kwargs=None):
        self.calls.append({'uids': [self.uid(m) for m in messages], 'messages': copy.deepcopy(messages),
                           'params': sampling_params, 'kwargs': chat_template_kwargs, 'use_tqdm': use_tqdm})
        if self.fail_on_call == len(self.calls):
            raise RuntimeError('CUDA error: simulated')
        outputs = []
        for conversation in messages:
            uid = self.uid(conversation)
            queue = self.script.setdefault(uid, [])
            item = queue.pop(0) if queue else self.default
            text, finish = item if isinstance(item, tuple) else (item, 'stop')
            text = text.replace('{valid}', base.target(None, 'joint', self.rooms[uid]))
            outputs.append(SimpleNamespace(prompt_token_ids=[1] * self.tok.length(conversation), outputs=[
                SimpleNamespace(text=text, token_ids=[1] * (len(text) + 1), finish_reason=finish)]))
        return outputs


@pytest.fixture
def world(tmp_path):
    data = base.eval_data(tmp_path / 'data')
    L.main(['cohort', '--data', str(data), '--out', str(tmp_path / 'cohort.json')])
    rooms = {r['uid']: r for r in map(json.loads, (data / 'rooms.jsonl').read_text().splitlines())}
    for request in map(json.loads, (data / 'joint/requests.jsonl').read_text().splitlines()):
        rooms[request['uid']]['_request'] = request
    for spec in O.SYSTEMS.values():
        model = tmp_path / 'models' / spec['model']
        model.mkdir(parents=True, exist_ok=True)
        (model / 'config.json').write_text('{"model_type": "fake"}')
        (model / 'model.safetensors').write_bytes(b'weights are not hashed')
    return tmp_path, rooms


def argv(root, system, out='run', *extra):
    return ['run', '--system', system, '--data', str(root / 'data'), '--cohort', str(root / 'cohort.json'),
            '--model-root', str(root / 'models'), '--out', str(root / out), *extra]


def loader(fake, seen=None):
    def load(engine):
        if seen is not None:
            seen.append(engine)
        return fake, dict
    return load


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_split_reasoning_per_template():
    s = O.split_reasoning
    assert s('<think>plan</think>\n\n{"a":1}', 'think', True, True) == ('<think>plan</think>', '\n\n{"a":1}')
    assert s('plan only, prefilled <think></think>{}', 'think', True, True)[1] == '{}'  # Qwen3.6 / DeepSeek: no opener
    assert s('still planning {"a":1}', 'think', True, False) == ('still planning {"a":1}', '')  # cap hit while thinking
    assert s('{"a":1}', 'think', True, True) == ('', '{"a":1}')  # thinking system that answered directly
    assert s('{"a":1}', 'think', False, False) == ('', '{"a":1}')  # non-thinking text is the answer, even when capped
    assert s('<|channel>thought\nhmm<channel|>{"a":1}', 'gemma', False, True)[1] == '{"a":1}'
    harmony = ('<|channel|>analysis<|message|>think {"x":0}<|end|><|start|>assistant'
               '<|channel|>final<|message|>{"a":1}')
    assert s(harmony, 'harmony', True, True)[1] == '{"a":1}'
    assert s(harmony + '<|return|>', 'harmony', True, True)[1] == '{"a":1}'
    assert s(harmony.replace('final<', 'final <|constrain|>json<'), 'harmony', True, True)[1] == '{"a":1}'
    assert s('<|channel|>analysis<|message|>{"x":0}', 'harmony', True, False) == ('<|channel|>analysis<|message|>{"x":0}', '')


def test_thinking_run_same_input_harness_order_and_evaluator(world):
    root, rooms = world
    fake = FakeLLM(rooms, {'il-bed-1': ['<think>a</think>bad', '<think>b</think>{valid}'],
                           'y-0': ['bad', 'worse', 'still bad']}, default='<think>plan</think>\n\n{valid}')
    engines = []
    assert O.main(argv(root, 'O1b', 'run', '--chunk', '10'), loader(fake, engines)) == 0
    out = root / 'run'
    cohort = json.loads((root / 'cohort.json').read_text())
    assert engines == [{'model': str((root / 'models/Qwen3-8B').resolve()), 'tensor_parallel_size': 1,
                        'max_model_len': 40960, 'seed': 0}]
    first = [c for c in fake.calls if len(c['messages'][0]) == 3]
    assert sum(len(c['uids']) for c in first) == 36 and max(len(c['uids']) for c in fake.calls) == 10
    for call in fake.calls:
        assert call['kwargs'] == {'enable_thinking': True} and call['use_tqdm'] is False
        assert all(p == {'max_tokens': 1024 + 128 * 2 + 16384, **O.SAMPLING} for p in call['params'])
    assert {json.dumps(k) for k in fake.tok.kwargs} == {'{"enable_thinking": true}'}  # Precount = LLM.chat template.
    for call in first:  # Exactly the J messages plus the API agent's fixed suffix.
        assert call['messages'] == [[*rooms[u]['_request']['messages'], {'role': 'user', 'content': L.SUFFIX}]
                                    for u in call['uids']]
    order = [u for c in first for u in c['uids']]
    by_len = sorted(cohort['uids'], key=lambda u: (-len(rooms[u]['_request']['messages'][1]['content']), u))
    assert order == by_len != cohort['uids']
    # Repair turns: the answer without its reasoning, then only the parser's error text (llm_agent rule).
    y = [c['messages'][c['uids'].index('y-0')] for c in fake.calls if 'y-0' in c['uids']]
    error = infer.parse_prediction('bad', rooms['y-0']['_request'], 'joint')['error']
    assert y[1] == [*rooms['y-0']['_request']['messages'], {'role': 'user', 'content': L.SUFFIX},
                    {'role': 'assistant', 'content': 'bad'}, {'role': 'user', 'content': error}]
    assert len(y[2]) == 7
    il = [c['messages'][c['uids'].index('il-bed-1')] for c in fake.calls if 'il-bed-1' in c['uids']]
    assert il[1][-2] == {'role': 'assistant', 'content': 'bad'}
    result = json.loads((out / 'result.json').read_text())
    assert result['status'] == 'LLM_COMPLETE' and result['valid_first'] == 34 and result['valid_final'] == 35
    assert result['rounds_histogram'] == {'1': 34, '2': 1, '3': 1} and result['calls_ledger'] == 34 + 2 + 3
    # Generated minus answer tokens; the fake's extra token stands for the stop token.
    assert result['tokens']['reasoning_tokens'] == 34 * (len('<think>plan</think>') + 1) + 2 * (len('<think>a</think>') + 1)
    for variant in L.VARIANTS:
        records = rows(out / variant / 'records.jsonl')
        assert [r['uid'] for r in records] == cohort['uids'] and not any('<think>' in (r['raw'] or '') for r in records)
    receipt = json.loads((out / 'receipts' / (L._digest('il-bed-0') + '.json')).read_text())
    answer = receipt['answers'][0]
    assert answer['reasoning'] == '<think>plan</think>' and answer['stopped_on_eos'] and not answer['reached_token_cap']
    assert answer['max_tokens'] == 17664 and answer['usage']['input_tokens'] == answer['prompt_tokens_precount'] == 100
    identity = json.loads((out / 'run_receipt.json').read_text())
    assert identity['config']['model_files_sha256'] == {'config.json': h.sha256(root / 'models/Qwen3-8B/config.json')}
    assert identity['config_sha256'] == infer._hash(identity['config']) and 'vllm' in identity['config']['versions']
    cost = json.loads((out / 'cost.json').read_text())
    assert cost['gpus'] == 1 and cost['invocations'] == 1 and cost['generated_requests'] == 39
    assert cost['answers_with_content'] == 39 and cost['config_sha256'] == identity['config_sha256']
    assert cost['prompt_token_precount_mismatches'] == 0
    summary = ev.evaluate(root / 'data', root / 'cohort.json', out, root / 'eval', repeats=20)
    assert summary['arms']['llm_raw']['all']['complete_count'] == 34
    assert summary['arms']['llm_harness']['all']['complete_count'] == 35
    assert summary['llm']['config']['model'] == 'qwen3-8b-think'
    before = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
    again = FakeLLM(rooms)
    assert O.main(argv(root, 'O1b'), loader(again)) == 0 and again.calls == []
    assert before == {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='resume identity'):
        O.main(argv(root, 'O1b', 'run', '--limit', '3'), loader(again))
    assert before == {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}


def test_crash_keeps_finished_chunks_and_resume_runs_only_pending(world):
    root, rooms = world
    crashing = FakeLLM(rooms, fail_on_call=2)
    with pytest.raises(RuntimeError, match='simulated'):
        O.main(argv(root, 'O1a', 'run', '--chunk', '20'), loader(crashing))
    status = json.loads((root / 'run/status.json').read_text())
    assert status == {'status': 'FAIL', 'error_type': 'RuntimeError', 'error': 'CUDA error: simulated'}
    done = set(crashing.calls[0]['uids'])
    assert len(done) == 20 and len(list((root / 'run/receipts').iterdir())) == 20
    resumed = FakeLLM(rooms)
    assert O.main(argv(root, 'O1a', 'run', '--chunk', '20'), loader(resumed)) == 0
    assert not done & {u for c in resumed.calls for u in c['uids']} and len(resumed.calls) == 1
    assert all(p['max_tokens'] == 1024 + 128 * 2 for p in resumed.calls[0]['params'])
    assert resumed.calls[0]['kwargs'] == {'enable_thinking': False}
    cost = json.loads((root / 'run/cost.json').read_text())
    assert cost['invocations'] == 2 and cost['generated_requests'] == 36 and cost['gpu_hours'] >= 0
    assert json.loads((root / 'run/result.json').read_text())['calls_ledger'] == 36


def test_harmony_final_channel_engine_flags_and_context_stop(world):
    root, rooms = world
    final = '<|channel|>analysis<|message|>We place objects.<|end|><|start|>assistant<|channel|>final<|message|>'
    fake = FakeLLM(rooms, default=final + '{valid}')
    engines = []
    assert O.main(argv(root, 'O4', 'gptoss', '--limit', '2'), loader(fake, engines)) == 0
    assert engines[0]['tensor_parallel_size'] == 2 and fake.calls[0]['kwargs'] == {'reasoning_effort': 'medium'}
    records = rows(root / 'gptoss/llm_harness/records.jsonl')
    assert all(r['schema']['valid'] and r['raw'] == base.target(None, 'joint', rooms[r['uid']]) for r in records)
    assert json.loads((root / 'gptoss/run_receipt.json').read_text())['config']['reasoning_effort'] == 'medium'
    deepseek = []
    O.main(argv(root, 'O5', 'ds', '--limit', '1'), loader(FakeLLM(rooms), deepseek))
    assert deepseek[0]['tokenizer_mode'] == 'deepseek_v4' and deepseek[0]['enable_expert_parallel']
    assert deepseek[0]['tensor_parallel_size'] == 4
    # The cap is bounded by the context left after the prompt; a repair that cannot fit is not attempted.
    full = FakeLLM(rooms, default=('x' * 950, 'length'), prompt_tokens=40000)
    assert O.main(argv(root, 'O3', 'long', '--limit', '1'), loader(full)) == 0
    receipt = json.loads(next((root / 'long/receipts').iterdir()).read_text())
    assert full.calls[0]['params'][0]['max_tokens'] == 960 and full.calls[0]['kwargs'] == {'enable_thinking': False}
    assert receipt['repair_stopped'] == 'INPUT_TOO_LONG' and receipt['rounds'] == 1 and len(full.calls) == 1
    assert receipt['answers'][0]['reached_token_cap'] and receipt['error'] is None
    assert json.loads((root / 'long/result.json').read_text())['reached_token_cap'] == 1
    # A first prompt with no room left is never generated and stays a failure in every denominator.
    none_left = FakeLLM(rooms, prompt_tokens=40960)
    assert O.main(argv(root, 'O3', 'none', '--limit', '2'), loader(none_left)) == 0 and none_left.calls == []
    result = json.loads((root / 'none/result.json').read_text())
    assert result['api_failed_rooms'] == 2 and result['valid_final'] == 0
    assert {r['error'] for r in rows(root / 'none/llm_raw/records.jsonl')} == {'INPUT_TOO_LONG'}
    assert json.loads((root / 'none/cost.json').read_text())['input_too_long_rooms'] == 2


def test_smoke_gate_and_status_fail_closed(world):
    root, rooms = world
    run_root = root / 'runs'
    thinking_only = FakeLLM(rooms, default=('<think>still planning', 'length'))
    assert O.main(argv(root, 'O1b', 'runs/' + O.out_name('O1b', 'smoke'), '--limit', '2', '--smoke-gate'),
                  loader(thinking_only)) == 1
    def broken(engine):
        raise RuntimeError('Model architectures [Gemma4ForConditionalGeneration] are not supported')
    with pytest.raises(RuntimeError):
        O.main(argv(root, 'O3', 'runs/' + O.out_name('O3', 'smoke'), '--limit', '2', '--smoke-gate'), broken)
    for mode, extra in (('smoke', ('--limit', '2', '--smoke-gate')), ('full', ())):
        assert O.main(argv(root, 'O1a', 'runs/' + O.out_name('O1a', mode), *extra), loader(FakeLLM(rooms))) == 0
    harmony = '<|channel|>analysis<|message|>ok<|end|><|start|>assistant<|channel|>final<|message|>{valid}'
    assert O.main(argv(root, 'O4', 'runs/' + O.out_name('O4', 'smoke'), '--limit', '2', '--smoke-gate'),
                  loader(FakeLLM(rooms, default=harmony))) == 0
    assert O.main(['status', '--run-root', str(run_root)]) == 0
    status = json.loads((run_root / 'llm-open-status.json').read_text())
    assert status['O1a']['state'] == 'complete' and status['O1a']['cost']['gpus'] == 1
    assert status['O1a']['result']['rooms'] == 36
    assert status['O1b'] == {**status['O1b'], 'state': 'not run',
                             'reason': 'no smoke answer had text after the reasoning was removed'}
    assert status['O3']['state'] == 'not run' and 'Gemma4ForConditionalGeneration' in status['O3']['reason']
    assert status['O3']['reason'].startswith('FAIL: RuntimeError: ')
    assert status['O4']['state'] == 'incomplete'  # smoke passed; full not started
    assert status['O5']['state'] == 'not run' and 'smoke output missing' in status['O5']['reason']
    assert status['kimi-k3']['state'] == status['glm-5.2']['state'] == 'not run'


def test_scripts_match_system_table():
    here = Path(O.__file__).parent
    sbatch = (here / 'isambard_llm_open.sbatch').read_text()
    helper = (here / 'submit_llm_open.sh').read_text()
    systems = '|'.join(O.SYSTEMS)
    assert f'case "$SYSTEM" in {systems}) ;;' in sbatch and f'case "$SYSTEM" in {systems}) ;;' in helper
    assert f'SYSTEMS=({" ".join(O.SYSTEMS)})' in helper
    assert 'smoke) OUT="llm-open-$SYSTEM-smoke-001"; EXTRA=(--limit 2 --smoke-gate)' in sbatch
    assert 'full) OUT="llm-open-$SYSTEM-001"' in sbatch
    assert O.out_name('O2', 'smoke') == 'llm-open-O2-smoke-001' and O.out_name('O2', 'full') == 'llm-open-O2-001'
    assert '--dependency="afterok:$SMOKE"' in helper and '--kill-on-invalid-dep=yes' in helper
    assert re.search(r'VLLM_USE_FLASHINFER_SAMPLER=0', sbatch)
    for name in ('XDG_CACHE_HOME', 'VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
        assert f'export {name}=' in sbatch or f' {name}=' in sbatch
    assert O.slurm_flags('O5', 'full') == '--gpus-per-node=4 --cpus-per-task=64 --mem=400G --time=06:00:00'
    assert O.slurm_flags('O1a', 'smoke') == '--gpus-per-node=1 --cpus-per-task=16 --mem=100G --time=01:30:00'
    assert {s['tp'] for k, s in O.SYSTEMS.items() if k in ('O2', 'O3', 'O4')} == {2}
    for script in ('isambard_llm_open.sbatch', 'submit_llm_open.sh'):
        subprocess.run(['bash', '-n', str(here / script)], check=True)
