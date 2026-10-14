"""Open-weight agent runner (amendments 003/004); vLLM is replaced by a fake LLM object (no GPU, no network)."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
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
    """vLLM LLM stand-in: per-UID scripted texts ('{valid}' = the GT answer), or (text, finish_reason).

    arm='placement' answers P's request (amendment 016 section 2): '{valid}' is then P's GT answer (no size).
    """
    def __init__(self, rooms, script=None, default='{valid}', prompt_tokens=100, fail_on_call=None, arm='joint'):
        self.rooms, self.script, self.default, self.arm = rooms, script or {}, default, arm
        self.tok, self.fail_on_call, self.calls = FakeTokenizer(prompt_tokens), fail_on_call, []
        self.key = '_request' if arm == 'joint' else '_placement'

    def get_tokenizer(self):
        return self.tok

    def uid(self, conversation):
        matches = [uid for uid, room in self.rooms.items() if conversation[:2] == room[self.key]['messages']]
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
            text = text.replace('{valid}', base.target(None, self.arm, self.rooms[uid]))
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
    for request in map(json.loads, (data / 'placement/requests.jsonl').read_text().splitlines()):
        rooms[request['uid']]['_placement'] = request
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
    assert s('plan only, prefilled <think></think>{}', 'think', True, True)[1] == '{}'  # Qwen3.8 / DeepSeek: no opener
    # Qwen3.8 cards: thinking content is '<think>\n...</think>\n\n' before the final response.
    qwen38 = '<think>\nBed first.\n</think>\n\n{"a":1}'
    assert s(qwen38, 'think', True, True) == ('<think>\nBed first.\n</think>', '\n\n{"a":1}')
    assert s('Bed first.\n</think>\n\n{"a":1}', 'think', True, True)[1] == '\n\n{"a":1}'  # opener in the prompt
    # DeepSeek-V4 encoding (thinking mode): the prompt ends with '<think>', the completion closes it.
    assert s('Room is 4 x 3 m.</think>{"a":1}', 'think', True, True) == ('Room is 4 x 3 m.</think>', '{"a":1}')
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
    # Muse-Glimmer ATEM (its chat_template.jinja): generation prompt '<|start|>assistant', reasoning on
    # ' to=self<|message|>...<|eom|>', answer on '<|start|>assistant to=user<|message|>...<|eot|>'.
    muse = ' to=self<|message|>Bed by the wall {"x":0}<|eom|><|start|>assistant to=user<|message|>{"a":1}<|eot|>'
    assert s(muse, 'muse', True, True) == (muse[:muse.index('{"a"')], '{"a":1}')
    assert s(' to=self<|message|>still planning', 'muse', True, False) == (' to=self<|message|>still planning', '')
    assert s(' to=self<|message|>plan<|eom|>', 'muse', True, True)[1] == ''  # reasoning only, no user message
    assert s(' to=user<|message|>{"a":1}<|eot|>', 'muse', True, True) == ('', '{"a":1}')  # answered, no reasoning


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
    # Cost and the scored answer's per-row cap flags ride through to report_v2.
    assert summary['llm']['cost'] == cost and summary['metric_contract']['cap_flags']
    for variant in L.VARIANTS:
        scored = {r['uid']: r for r in rows(root / 'eval' / f'{variant}-geometry.jsonl')}
        assert all(r['reached_token_cap'] is False and r['stopped_on_eos'] is True for r in scored.values())
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
    deepseek, ds = [], FakeLLM(rooms, default='<think>plan</think>{valid}')
    assert O.main(argv(root, 'O5', 'ds', '--limit', '1'), loader(ds, deepseek)) == 0
    assert deepseek[0]['tokenizer_mode'] == 'deepseek_v4' and deepseek[0]['enable_expert_parallel']
    assert deepseek[0]['tensor_parallel_size'] == 4
    # Amendments 006/007: O5 thinks at effort high, so it gets the thinking allowance and the split.
    assert ds.calls[0]['kwargs'] == {'thinking': True, 'reasoning_effort': 'high'} and O.SYSTEMS['O5']['thinking']
    assert ds.calls[0]['params'][0]['max_tokens'] == 1024 + 128 * 2 + 16384
    assert rows(root / 'ds/llm_harness/records.jsonl')[0]['schema']['valid']
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


# Amendment 007 (006 for O1a/O1b/O3/O4): model dir, GPUs, thinking, chat_template_kwargs, engine extras.
MODES = {
    'O1a': ('Qwen3-8B', 1, False, {'enable_thinking': False}, {}),
    'O1b': ('Qwen3-8B', 1, True, {'enable_thinking': True}, {}),
    'O2': ('Qwen3.8-27B', 2, True, {'enable_thinking': True}, {'limit_mm_per_prompt': {'image': 0, 'video': 0}}),
    'O3': ('gemma-4-31b-it', 2, False, {'enable_thinking': False}, {'limit_mm_per_prompt': {'image': 0, 'audio': 0}}),
    'O4': ('gpt-oss-120b', 2, True, {'reasoning_effort': 'medium'}, {}),
    'O5': ('DeepSeek-V4-Flash-0731', 4, True, {'thinking': True, 'reasoning_effort': 'high'},
           {'tokenizer_mode': 'deepseek_v4', 'enable_expert_parallel': True, 'kv_cache_dtype': 'fp8', 'block_size': 256,
            'trust_remote_code': True, 'moe_backend': 'marlin'}),
    'O6': ('Qwen3.8-Flash-Next-FP8', 4, True, {'enable_thinking': True, 'reasoning_effort': 'medium'},
           {'enable_expert_parallel': True, 'moe_backend': 'triton', 'limit_mm_per_prompt': {'image': 0, 'video': 0}}),
    'O7': ('Muse-Glimmer-30B', 2, True, {}, {}),  # Model default: no kwargs, no 'Reasoning strength:' line.
}
SAMPLES = {'think': '<think>\nplan\n</think>\n\n{valid}', 'gemma': '<|channel>thought\nplan<channel|>{valid}',
           'harmony': '<|channel|>analysis<|message|>plan<|end|><|start|>assistant<|channel|>final<|message|>{valid}',
           'muse': ' to=self<|message|>plan<|eom|><|start|>assistant to=user<|message|>{valid}<|eot|>'}


@pytest.mark.parametrize('system', MODES)
def test_amendment_007_modes_reach_engine_and_template(world, system):
    root, rooms = world
    model, tp, thinking, template, engine = MODES[system]
    assert list(O.SYSTEMS) == list(MODES) and O.SYSTEMS[system]['thinking'] is thinking
    fake, engines = FakeLLM(rooms, default=SAMPLES[O.SYSTEMS[system]['split']]), []
    assert O.main(argv(root, system, system, '--limit', '1'), loader(fake, engines)) == 0
    assert engines == [{'model': str((root / 'models' / model).resolve()), 'tensor_parallel_size': tp,
                        'max_model_len': 40960, 'seed': 0, **engine}]
    assert fake.calls[0]['kwargs'] == (template or None) and fake.tok.kwargs == [template]
    record = rows(root / system / 'llm_harness/records.jsonl')[0]
    assert record['schema']['valid'] and 'plan' not in record['raw']


def test_smoke_gate_and_status_fail_closed(world):
    root, rooms = world
    run_root = root / 'runs'
    # Amendment 006: loads and runs but only cap-hit reasoning -> measured failure; the smoke passes.
    thinking_only = FakeLLM(rooms, default=('<think>still planning', 'length'))
    assert O.main(argv(root, 'O1b', 'runs/' + O.out_name('O1b', 'smoke'), '--limit', '2', '--smoke-gate'),
                  loader(thinking_only)) == 0
    measured = json.loads((run_root / O.out_name('O1b', 'smoke') / 'result.json').read_text())
    assert measured['valid_final'] == 0 and measured['reached_token_cap'] == 2 * (L.REPAIRS + 1)
    # Nothing generated at all (no prompt fits) is not evidence that the system executes.
    assert O.main(argv(root, 'O2', 'runs/' + O.out_name('O2', 'smoke'), '--limit', '2', '--smoke-gate'),
                  loader(FakeLLM(rooms, prompt_tokens=40960))) == 1
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
    assert status['O1b']['state'] == 'incomplete' and status['O1b']['reason'] == 'full run not started'
    assert status['O2'] == {**status['O2'], 'state': 'not run', 'reason': 'smoke generated no answer'}
    assert status['O3']['state'] == 'not run' and 'Gemma4ForConditionalGeneration' in status['O3']['reason']
    assert status['O3']['reason'].startswith('FAIL: RuntimeError: ')
    assert status['O4']['state'] == 'incomplete'  # smoke passed; full not started
    assert status['O5']['state'] == 'not run' and 'smoke output missing' in status['O5']['reason']
    assert status['O6']['state'] == status['O7']['state'] == 'not run'  # No smoke output yet.
    assert status['kimi-k3']['state'] == status['glm-5.2']['state'] == status['qwen3.6-27b']['state'] == 'not run'
    assert status['qwen3.6-27b']['reason'] == 'not run (superseded by Qwen3.8-27B)'
    assert not set(O.NOT_RUN) & set(O.SYSTEMS)


@pytest.mark.parametrize('system,text', [('O5', '{valid}'), ('O7', ' to=user<|message|>{valid}<|eot|>')])
def test_thinking_smoke_without_reasoning_fails_closed(world, system, text):
    # A thinking system whose template switch is silently ignored answers without any reasoning:
    # the smoke must not pass, or the full run would start in a mode amendment 006 forbids. O7 (Muse ATEM):
    # an answer on the user channel with no 'to=self' reasoning message.
    root, rooms = world
    out = root / 'runs' / O.out_name(system, 'smoke')
    assert O.main(argv(root, system, 'runs/' + O.out_name(system, 'smoke'), '--limit', '2', '--smoke-gate'),
                  loader(FakeLLM(rooms, default=text))) == 1
    assert O.smoke_failure(out) == 'thinking system produced no reasoning (mode not applied)'


def test_scripts_match_system_table():
    here = Path(O.__file__).parent
    sbatch = (here / 'isambard_llm_open.sbatch').read_text()
    helper = (here / 'submit_llm_open.sh').read_text()
    systems = '|'.join(O.SYSTEMS)
    assert f'case "$SYSTEM" in {systems}) ;;' in sbatch and f'case "$SYSTEM" in {systems}) ;;' in helper
    order = 'O6 O5 O4 O2 O3 O7 O1a O1b'  # Amendment 007 run order.
    assert f'SYSTEMS=({order})' in helper and sorted(order.split()) == sorted(O.SYSTEMS)
    # Preflight mirrors the v1_1 vLLM sbatch and fails fast before any GPU work.
    assert 'module load cudatoolkit || ' in sbatch and 'CUDA_PATH="$CUDA_HOME"' in sbatch
    preflight = sbatch.index("\"$PYTHON\" -B -c 'import shapely, fastfill.comparison.llm_open' || {")
    assert sbatch.index('module load cudatoolkit') < preflight < sbatch.index('nvidia-smi -L')
    assert 'PREFLIGHT FAIL' in sbatch[preflight:sbatch.index('nvidia-smi -L')]
    assert 'smoke) OUT="llm-open-$SYSTEM-smoke-001"; EXTRA=(--limit 2 --smoke-gate)' in sbatch
    assert 'full) OUT="llm-open-$SYSTEM-001"' in sbatch
    assert O.out_name('O2', 'smoke') == 'llm-open-O2-smoke-001' and O.out_name('O2', 'full') == 'llm-open-O2-001'
    assert '--dependency="afterok:$SMOKE,$DEP"' in helper and '--kill-on-invalid-dep=yes' in helper
    assert '#SBATCH --job-name=unknownsize-llm-open\n' in sbatch  # singleton keys on this one name
    assert re.search(r'VLLM_USE_FLASHINFER_SAMPLER=0', sbatch)
    for name in ('XDG_CACHE_HOME', 'VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
        assert f'export {name}=' in sbatch or f' {name}=' in sbatch
    assert O.slurm_flags('O5', 'full') == O.slurm_flags('O6', 'full') == (
        '--gpus-per-node=4 --cpus-per-task=64 --mem=400G --time=06:00:00')
    assert O.slurm_flags('O7', 'full') == '--gpus-per-node=2 --cpus-per-task=32 --mem=200G --time=08:00:00'
    assert O.slurm_flags('O1a', 'smoke') == '--gpus-per-node=1 --cpus-per-task=16 --mem=100G --time=01:30:00'
    assert {s['tp'] for k, s in O.SYSTEMS.items() if k in ('O2', 'O3', 'O4', 'O7')} == {2}
    for script in ('isambard_llm_open.sbatch', 'submit_llm_open.sh'):
        subprocess.run(['bash', '-n', str(here / script)], check=True)


def stub_squeue(tmp_path):
    """squeue stub: prints $SQUEUE_OUT only for the exact open-model query; exits 1 when SQUEUE_FAIL is set."""
    stub = tmp_path / 'squeue'
    stub.write_text('#!/bin/sh\ntest -z "$SQUEUE_FAIL" || exit 1\n'
                    'test "$*" = "-h -u me -n unknownsize-llm-open -o %i" || exit 3\n'
                    'printf "%s" "$SQUEUE_OUT"\n')
    stub.chmod(0o755)


def test_submit_serialises_open_jobs_after_inference_with_stub_sbatch(tmp_path):
    """GPU cap: every open-model job is singleton (one at a time) and waits for the AFTER job ids."""
    code = Path(O.__file__).resolve().parents[2]
    helper = (code / 'fastfill/comparison/submit_llm_open.sh').read_text()
    pin = 'PYTHON="$PROJECT_ROOT/env/bin/python"\n'
    assert pin in helper
    script = tmp_path / 'submit_llm_open.sh'
    script.write_text(helper.replace(pin, f'PYTHON={sys.executable}\n'))
    stub = tmp_path / 'sbatch'
    stub.write_text('#!/bin/sh\necho "$@" >> "$SBATCH_LOG"\necho "$((100 + $(wc -l < "$SBATCH_LOG")));cluster"\n')
    stub.chmod(0o755)
    stub_squeue(tmp_path)
    env = {'PATH': f'{tmp_path}:/usr/bin:/bin', 'SBATCH_LOG': str(tmp_path / 'log'), 'USER': 'me'}
    run = subprocess.run(['bash', str(script), str(code), '7383111:7383121', 'O1a', 'O5'],
                         env=env, capture_output=True, text=True, check=True)
    log = (tmp_path / 'log').read_text().splitlines()
    sb = f"{code}/fastfill/comparison/isambard_llm_open.sbatch {code}"
    dep = 'afterany:7383111:7383121,singleton'
    assert log[0] == f"--parsable --dependency={dep} {O.slurm_flags('O1a', 'smoke')} {sb} O1a smoke"
    assert log[1] == (f"--parsable --kill-on-invalid-dep=yes --dependency=afterok:101,{dep} "
                      f"{O.slurm_flags('O1a', 'full')} {sb} O1a full")
    assert log[2] == f"--parsable --dependency={dep} {O.slurm_flags('O5', 'smoke')} {sb} O5 smoke"
    assert log[3].startswith(f'--parsable --kill-on-invalid-dep=yes --dependency=afterok:103,{dep} ')
    assert '--dependency=afterany:101:102:103:104 ' in log[4] and len(log) == 5
    assert run.stdout.splitlines() == ['O1a smoke 101 full 102', 'O5 smoke 103 full 104', 'status 105']
    (tmp_path / 'log').unlink()
    subprocess.run(['bash', str(script), str(code), '-', 'O2'], env=env, check=True, capture_output=True)
    log = (tmp_path / 'log').read_text().splitlines()
    assert '--dependency=singleton ' in log[0] and '--dependency=afterok:101,singleton ' in log[1]
    for bad in (['O1a'], ['', 'O1a'], ['7383111:', 'O1a'], ['7383111,singleton', 'O1a'], []):
        assert subprocess.run(['bash', str(script), str(code), *bad], env=env, capture_output=True).returncode != 0
    assert len((tmp_path / 'log').read_text().splitlines()) == 3


def test_submit_default_order_and_abort_lists_submitted_jobs(tmp_path):
    code = Path(O.__file__).resolve().parents[2]
    helper = (code / 'fastfill/comparison/submit_llm_open.sh').read_text()
    script = tmp_path / 'submit_llm_open.sh'
    script.write_text(helper.replace('PYTHON="$PROJECT_ROOT/env/bin/python"\n', f'PYTHON={sys.executable}\n'))
    stub = tmp_path / 'sbatch'
    stub.write_text('#!/bin/sh\necho "$@" >> "$SBATCH_LOG"\nn=$(wc -l < "$SBATCH_LOG")\n'
                    'if [ "$n" -eq "$FAIL_ON" ]; then echo "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2; exit 1; fi\n'
                    'echo "$((100 + n));cluster"\n')
    stub.chmod(0o755)
    stub_squeue(tmp_path)
    env = {'PATH': f'{tmp_path}:/usr/bin:/bin', 'SBATCH_LOG': str(tmp_path / 'log'), 'FAIL_ON': '0', 'USER': 'me'}
    run = subprocess.run(['bash', str(script), str(code), '-'], env=env, capture_output=True, text=True, check=True)
    assert [line.split()[0] for line in run.stdout.splitlines()] == [
        'O6', 'O5', 'O4', 'O2', 'O3', 'O7', 'O1a', 'O1b', 'status']
    (tmp_path / 'log').unlink()
    run = subprocess.run(['bash', str(script), str(code), '-'], env={**env, 'FAIL_ON': '4'},
                         capture_output=True, text=True)
    assert run.returncode == 1 and len((tmp_path / 'log').read_text().splitlines()) == 4  # Nothing after it.
    assert "sbatch for 'O5 full' failed" in run.stderr and 'scancel 101 102 103' in run.stderr
    assert run.stdout.splitlines() == ['O6 smoke 101 full 102']
    (tmp_path / 'log').unlink()
    bad = subprocess.run(['bash', str(script), str(code), '-', 'O5', 'O9'], env=env, capture_output=True)
    assert bad.returncode == 2 and not (tmp_path / 'log').exists()  # Systems are checked before any sbatch.


def test_submit_refuses_while_an_open_model_chain_is_in_slurm(tmp_path):
    """An older chain shares the singleton name and output folders; nothing new goes in behind it."""
    code = Path(O.__file__).resolve().parents[2]
    helper = (code / 'fastfill/comparison/submit_llm_open.sh').read_text()
    script = tmp_path / 'submit_llm_open.sh'
    script.write_text(helper.replace('PYTHON="$PROJECT_ROOT/env/bin/python"\n', f'PYTHON={sys.executable}\n'))
    stub = tmp_path / 'sbatch'
    stub.write_text('#!/bin/sh\necho "$@" >> "$SBATCH_LOG"\necho "$((100 + $(wc -l < "$SBATCH_LOG")));cluster"\n')
    stub.chmod(0o755)
    stub_squeue(tmp_path)
    env = {'PATH': f'{tmp_path}:/usr/bin:/bin', 'SBATCH_LOG': str(tmp_path / 'log'), 'USER': 'me'}
    run = subprocess.run(['bash', str(script), str(code), '-', 'O5'], env={**env, 'SQUEUE_OUT': '7384533\n7384535\n'},
                         capture_output=True, text=True)
    assert run.returncode == 1 and not (tmp_path / 'log').exists() and run.stdout == ''
    assert 'scancel 7384533 7384535' in run.stderr
    run = subprocess.run(['bash', str(script), str(code), '-', 'O5'], env={**env, 'SQUEUE_FAIL': '1'},
                         capture_output=True, text=True)
    assert run.returncode == 1 and not (tmp_path / 'log').exists()  # Fails closed when Slurm can't be asked.
    subprocess.run(['bash', str(script), str(code), '-', 'O5'], env=env, check=True, capture_output=True)
    assert len((tmp_path / 'log').read_text().splitlines()) == 3  # Empty queue: smoke, full, status.


def _identity_shape(receipt):
    """sha256 of a run identity with its environment-bound values blanked: keys, key order and constants only."""
    r = copy.deepcopy(receipt)
    r['config'].update(versions=None, engine={**r['config']['engine'], 'model': None})
    r['config_sha256'] = r['run_sha256'] = None
    r['data'] = dict.fromkeys(r['data'])
    r['cohort'].update(sha256=None, uids_sha256=None)
    r['code_sha256'] = dict.fromkeys(r['code_sha256'])
    return hashlib.sha256(json.dumps(r).encode()).hexdigest()


# _identity_shape of each system's joint run (--limit 2, this fixture) under llm_open before amendment 016
# (affcc1fea). The full outputs were also compared then: identical except llm_open.py's own code hash.
JOINT_IDENTITY_SHAPE = {
    'O1a': '12c892c4cd65ee883b2d3e4e5db393e3db9d68f00f73c0cd18d259edc8e5d5ff',
    'O1b': 'e0bfb59630afc9ea34bf80b6dd430f2c97a730e8687fc5043406971119ad19cf',
    'O2': 'a3e5e1564632f18ee8aa2942d8219fbfb242e506e11d3ed5c3f02c03d2272531',
    'O3': '4c7b787f1b25f4963f051507e03029805b4bee15a39d59adf7be15a1fb6a3d25',
    'O4': '3dd84096fb244b850514d8490832d7ec4c99c990012ee73e90fc95e64b6d822c',
    'O5': 'f77b5de92ab4441c7622016bc42bbef8a3b5c5d68fbd6146aaec773fb4a26153',
    'O6': 'e2aff6047049a08270ea0c268fee4252f1a1a7951bff27955a4966eb7a75511b',
    'O7': 'fcc02762ae20d8e42f16de046c6d8322d693dc0dbb79932ad858191177184233',
}


def test_default_joint_identity_is_unchanged_by_the_placement_arm(world):
    root, rooms = world
    assert list(JOINT_IDENTITY_SHAPE) == list(O.SYSTEMS)
    for system, shape in JOINT_IDENTITY_SHAPE.items():
        for out, extra in ((system, ()), (system + '-explicit', ('--arm', 'joint'))):
            assert O.main(argv(root, system, out, '--limit', '2', *extra), loader(FakeLLM(rooms))) == 0
        receipt = (root / system / 'run_receipt.json').read_bytes()
        assert receipt == (root / (system + '-explicit') / 'run_receipt.json').read_bytes()
        identity = json.loads(receipt)
        assert _identity_shape(identity) == shape, system
        assert 'request_arm' not in identity['config']
        assert identity['data'] == {'manifest_sha256': h.sha256(root / 'data/MANIFEST.json'),
                                    'joint_requests_sha256': h.sha256(root / 'data/joint/requests.jsonl')}


def test_placement_arm_sends_p_request_and_enforces_its_sizes(world):
    """Amendment 016 section 2 + addendum B2: P's system+user messages, the same suffix, harness and cap; returned sizes
    are replaced by the request's before the parser judges, so a size-only difference is not invalid."""
    root, rooms = world
    joint_answer = base.target(None, 'joint', rooms['y-0'])  # GT sizes (not the prior): P's own parser rejects it.
    fake = FakeLLM(rooms, {'y-0': [joint_answer], 'il-bed-1': ['<think>a</think>bad', '<think>b</think>{valid}']},
                   default='<think>plan</think>{valid}', arm='placement')
    assert O.main(argv(root, 'O2', 'p', '--arm', 'placement', '--chunk', '10'), loader(fake)) == 0
    first = [c for c in fake.calls if len(c['messages'][0]) == 3]
    assert sum(len(c['uids']) for c in first) == 36
    for call in first:
        assert call['messages'] == [[*rooms[u]['_placement']['messages'], {'role': 'user', 'content': L.SUFFIX}]
                                    for u in call['uids']]
        assert call['kwargs'] == {'enable_thinking': True}
        assert all(p == {'max_tokens': 1024 + 128 * 2 + 16384, **O.SAMPLING} for p in call['params'])
    request = rooms['y-0']['_placement']
    assert 'Use the supplied object sizes' in request['messages'][0]['content']
    sizes = [o['size'] for o in json.loads(request['messages'][1]['content'])['objects']]
    assert sizes != [o['size'] for o in rooms['y-0']['objects']]  # The fixture's prior is GT + 0.1, not GT.
    assert not infer.parse_prediction(joint_answer, request, 'placement')['valid']
    assert L.parse(joint_answer, request, 'placement')['valid']
    repairs = [c for c in fake.calls if len(c['messages'][0]) > 3]
    assert [c['uids'] for c in repairs] == [['il-bed-1']]  # y-0 is valid at once; only the bad answer is repaired.
    bed = rooms['il-bed-1']['_placement']
    assert repairs[0]['messages'][0] == [*bed['messages'], {'role': 'user', 'content': L.SUFFIX},
                                         {'role': 'assistant', 'content': 'bad'},
                                         {'role': 'user', 'content': L.parse('bad', bed, 'placement')['error']}]
    result = json.loads((root / 'p/result.json').read_text())
    assert result['valid_first'] == 35 and result['valid_final'] == 36
    harness = {r['uid']: r for r in rows(root / 'p/llm_harness/records.jsonl')}
    assert [p['size'] for p in harness['y-0']['schema']['placements']] == sizes  # Sizes from the request.
    assert harness['y-0']['raw'] == joint_answer  # The receipt keeps the model's answer as returned.
    identity = json.loads((root / 'p/run_receipt.json').read_text())
    config = identity['config']
    assert config['request_arm'] == 'placement' and config['input'] == 'placement_system_user_verbatim_then_suffix_user'
    assert config['parser'] == ('infer.parse_prediction(enforce_sizes(answer_after_reasoning), placement_request, '
                                'placement)')
    assert 'request_sizes' not in config and 'subset' not in identity['cohort']
    assert identity['data'] == {'manifest_sha256': h.sha256(root / 'data/MANIFEST.json'),
                                'placement_requests_sha256': h.sha256(root / 'data/placement/requests.jsonl')}
    with pytest.raises(ValueError, match='resume identity'):  # A joint invocation never resumes a P run.
        O.main(argv(root, 'O2', 'p', '--chunk', '10'), loader(FakeLLM(rooms)))


def test_pilot_subset_and_gt_sizes_enter_the_identity_only_when_given(world):
    """Addenda A6/B1 (--uids: a cohort subset, run in cohort order) and A1 (--sizes gt: P's request with GT sizes)."""
    root, rooms = world
    cohort = json.loads((root / 'cohort.json').read_text())['uids']
    (root / 'pilot.json').write_text(json.dumps({'uids': [cohort[7], cohort[2], cohort[30]]}))
    for uid, room in rooms.items():
        room['_gt'] = L.gt_request(room['_placement'], room)
    fake = FakeLLM(rooms, arm='placement')
    fake.key = '_gt'  # The fake answers only GT-size requests.
    gt = ('--arm', 'placement', '--sizes', 'gt', '--uids', str(root / 'pilot.json'))
    assert O.main(argv(root, 'O2', 'gt', *gt), loader(fake)) == 0
    assert sorted(u for c in fake.calls for u in c['uids']) == sorted([cohort[2], cohort[7], cohort[30]])
    request = json.loads(rooms[cohort[2]]['_gt']['messages'][1]['content'])
    assert [o['size'] for o in request['objects']] == [o['size'] for o in rooms[cohort[2]]['objects']]
    assert rooms[cohort[2]]['_gt']['messages'][0] == rooms[cohort[2]]['_placement']['messages'][0]
    manifest = json.loads((root / 'gt/MANIFEST.json').read_text())
    assert manifest['expected_uids'] == [cohort[2], cohort[7], cohort[30]]  # Cohort order.
    identity = json.loads((root / 'gt/run_receipt.json').read_text())
    assert identity['config']['request_sizes'] == 'gt' and identity['data']['rooms_sha256'] == h.sha256(
        root / 'data/rooms.jsonl')
    assert identity['cohort']['rooms'] == 3 and identity['cohort']['subset'] == {
        'sha256': h.sha256(root / 'pilot.json'), 'uids_sha256': L._digest(json.dumps(manifest['expected_uids']))}
    with pytest.raises(ValueError, match='needs --arm placement'):
        O.main(argv(root, 'O2', 'bad', '--sizes', 'gt'), loader(FakeLLM(rooms)))
    (root / 'pilot.json').write_text(json.dumps(['not-in-cohort']))
    with pytest.raises(ValueError, match='unique cohort uids'):
        O.main(argv(root, 'O2', 'bad2', '--arm', 'placement', '--uids', str(root / 'pilot.json')), loader(fake))


def _submit_016(tmp_path):
    """submit_016_agents.sh with PROJECT_ROOT = tmp_path (inputs made) and stub sbatch/squeue/scontrol in tmp_path/bin;
    the env python is a stub that prints llm_open's slurm flags and records the agents_016 pilot call."""
    code, run_root, tmp_path = Path(O.__file__).resolve().parents[2], tmp_path, tmp_path / 'bin'
    tmp_path.mkdir()
    helper = (code / 'fastfill/comparison/submit_016_agents.sh').read_text()
    pin = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    assert helper.count(pin) == 1
    script = tmp_path / 'submit_016_agents.sh'
    script.write_text(helper.replace(pin, f'PROJECT_ROOT={run_root}\n'))
    python = run_root / 'env/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\ncase "$*" in *agents_016*) echo "$@" >> "$PILOT_LOG"; exit "${{PILOT_RC:-0}}" ;; esac\n'
                      f'exec {sys.executable} "$@"\n')
    python.chmod(0o755)
    stub = tmp_path / 'sbatch'
    stub.write_text('#!/bin/sh\necho "$@" >> "$SBATCH_LOG"\nn=$(wc -l < "$SBATCH_LOG")\n'
                    'if [ "$n" -eq "${FAIL_ON:-0}" ]; then exit 1; fi\necho "$((100 + n));cluster"\n')
    (tmp_path / 'scontrol').write_text('#!/bin/sh\necho "$@" >> "$SCONTROL_LOG"\n')
    (tmp_path / 'squeue').write_text('#!/bin/sh\ntest -z "$SQUEUE_FAIL" || exit 1\n'
                                     'test "$*" = "-h -u me -o %i %j" || exit 3\nprintf "%s" "$SQUEUE_OUT"\n')
    for name in ('sbatch', 'scontrol', 'squeue'):
        (tmp_path / name).chmod(0o755)
    run = run_root / 'runs/unknownsize-20261013'
    for name in ('eval-data-v1/MANIFEST.json', 'eval-data-v1/placement/requests.jsonl', 'llm-cohort-v1.json',
                 'infer-joint-002/MANIFEST.json', 'infer-placement-005/MANIFEST.json'):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text('{}')
    env = {'PATH': f'{tmp_path}:/usr/bin:/bin', 'SBATCH_LOG': str(tmp_path / 'log'), 'USER': 'me',
           'SCONTROL_LOG': str(tmp_path / 'scontrol.log'), 'PILOT_LOG': str(tmp_path / 'pilot.log')}
    return code, script, run, env


def test_submit_016_runs_pilots_first_then_fulls_under_a_gpu_ceiling(tmp_path):
    code, script, run, env = _submit_016(tmp_path)
    out = subprocess.run(['bash', str(script), str(code), 'O2', 'O6', 'L'], env=env, capture_output=True, text=True,
                         check=True)
    log = (tmp_path / 'bin/log').read_text().splitlines()
    sb, s = f'{code}/fastfill/comparison/isambard_llm_open.sbatch {code}', f'{code}/fastfill/comparison'
    pilot, cohort = f'{run}/a016-pilot-v1.json', f'{run}/llm-cohort-v1.json'
    o2 = O.slurm_flags('O2', 'full').rsplit(' --time=', 1)[0]
    o6 = O.slurm_flags('O6', 'full').rsplit(' --time=', 1)[0]
    assert (tmp_path / 'bin/pilot.log').read_text().split()[-6:] == [
        '--data', f'{run}/eval-data-v1', '--cohort', cohort, '--out', pilot]
    # O2: pilot (held, gated, budgeted), its eval and S1 on the pilot, then the same for O6 and L.
    assert log[0] == f'--parsable --hold --dependency=singleton {o2} --time=01:30:00 {sb} O2 pilot placement'
    assert '--dependency=afterok:101 ' in log[1] and f'--llm {run}/llm-open-O2-P-pilot-001 ' in log[1]
    assert log[1].endswith(f'--out {run}/llm-eval-O2-P-pilot-v1 --workers 16 --uids {pilot}')
    assert '--joint ' + f'{run}/infer-joint-002 --placement {run}/infer-placement-005' in log[1]
    for line, v in zip(log[2:4], ('harness', 'raw')):
        assert '--dependency=afterok:102 ' in line and '--cpus-per-task=72' in line and line.endswith(
            f'--geometry {run}/llm-eval-O2-P-pilot-v1/llm_{v}-geometry.jsonl --requests-arm placement --uids {pilot} '
            f'--out {run}/s1-016-size-fixed/agent-O2-P-pilot-{v}')
        assert '--max-trials 2000 --time-limit 60' in line
    assert log[4] == f'--parsable --hold --dependency=singleton {o6} --time=00:45:00 {sb} O6 pilot placement'
    assert log[8] == f'--parsable --hold --time=04:00:00 {s}/isambard_llm_agent.sbatch {code} pilot-placement'
    assert f'--llm {run}/llm-agent-P-pilot-001 ' in log[9] and f'llm-eval-L-P-pilot-v1 --workers 16 --uids {pilot}' in log[9]
    assert log[10].endswith(f'--out {run}/s1-016-size-fixed/agent-L-P-pilot-harness')
    # Fulls wait for their own pilot (afterok) and every other GPU pilot (afterany): pilots first (A6).
    assert log[12] == (f'--parsable --kill-on-invalid-dep=yes --dependency=afterok:101,afterany:105,singleton {o2} '
                       f'--time=05:00:00 {sb} O2 full placement')
    assert log[13].endswith(f'--out {run}/llm-eval-O2-P-v1 --workers 16') and '--uids' not in log[13]
    assert log[14].endswith(f'--uids {cohort} --out {run}/s1-016-size-fixed/agent-O2-P-harness')
    assert '--dependency=afterok:105,afterany:101,singleton' in log[16] and '--time=01:30:00' in log[16]
    # O2 with GT sizes on the pilot (A1 supplementary), submitted last.
    assert log[20] == (f'--parsable --kill-on-invalid-dep=yes --dependency=afterok:101,singleton {o2} --time=01:30:00 '
                       f'{sb} O2 pilot placement-gt')
    assert f'--llm {run}/llm-open-O2-PGT-pilot-001 ' in log[21] and log[23].endswith('agent-O2-PGT-pilot-raw')
    assert len(log) == 24 and (tmp_path / 'bin/scontrol.log').read_text() == 'release 101,105,109\n'
    # Ceiling: O2 (90 + 300 + GT 90 min) x 2 GPUs + O6 (45 + 90) x 4 = 1,500 GPU-minutes; L uses no GPU.
    assert out.stdout.splitlines()[-1] == 'released 101,105,109 (GPU ceiling 1500 GPU-minutes)'
    # The sbatch writes exactly the folders the chain refuses, evaluates and repairs.
    sbatch = (code / 'fastfill/comparison/isambard_llm_open.sbatch').read_text()
    assert 'pilot) OUT="llm-open-$SYSTEM-pilot-001"; EXTRA=(--uids "$PAIR_RUN_ROOT/a016-pilot-v1.json" --smoke-gate)' in sbatch
    assert 'OUT="llm-open-$SYSTEM-PGT-${OUT#llm-open-$SYSTEM-}"; EXTRA+=(--arm placement --sizes gt)' in sbatch
    agent = (code / 'fastfill/comparison/isambard_llm_agent.sbatch').read_text()
    assert 'pilot-placement) OUT=llm-agent-P-pilot-001' in agent and '--arm placement --uids "$PAIR_RUN_ROOT/a016-pilot-v1.json"' in agent
    for name in ('submit_016_agents.sh', 'isambard_llm_open.sbatch', 'isambard_llm_agent.sbatch'):
        subprocess.run(['bash', '-n', str(code / 'fastfill/comparison' / name)], check=True)
    # Default systems (O2 O6 O4 O7 L) fit the cap: 33.5 GPU-h; a failed sbatch aborts with the ids to cancel.
    (tmp_path / 'bin/log').unlink()
    bad = subprocess.run(['bash', str(script), str(code)], env={**env, 'FAIL_ON': '5'}, capture_output=True, text=True)
    assert bad.returncode == 1 and 'scancel 101 102 103 104' in bad.stderr
    (tmp_path / 'bin/log').unlink()
    ok = subprocess.run(['bash', str(script), str(code)], env=env, capture_output=True, text=True, check=True)
    assert ok.stdout.splitlines()[-1].endswith('(GPU ceiling 2010 GPU-minutes)')


def test_submit_016_refuses_existing_outputs_queued_chains_missing_inputs_and_old_code(tmp_path):
    code, script, run, env = _submit_016(tmp_path)
    for setup, extra_env, message in (
            (lambda: (run / 's1-016-size-fixed/agent-O7-P-pilot-raw').mkdir(parents=True), {}, 'agent-O7-P-pilot-raw exists'),
            (lambda: (run / 's1-016-size-fixed/agent-O7-P-pilot-raw').rmdir(), {'SQUEUE_OUT': '7400001 unknownsize-llm-agent\n'},
             'scancel 7400001'),
            (lambda: None, {'SQUEUE_OUT': '7400002 other-job\n', 'PILOT_RC': '1'}, 'agents_016 pilot'),
            (lambda: (run / 'infer-placement-005/MANIFEST.json').unlink(), {}, 'missing')):
        setup()
        bad = subprocess.run(['bash', str(script), str(code)], env={**env, **extra_env}, capture_output=True, text=True)
        assert bad.returncode == 1 and message in bad.stderr and 'nothing submitted' in bad.stderr, bad.stderr
        assert not (tmp_path / 'bin/log').exists()
    for system in ('O9', 'O3'):  # unknown, and no amendment-016 GPU budget
        assert subprocess.run(['bash', str(script), str(code), system], env=env, capture_output=True).returncode == 2
    old = tmp_path / 'old'  # A CODE_ROOT whose evaluate_llm cannot score size-fixed runs.
    (old / 'fastfill').mkdir(parents=True)
    import shutil
    shutil.copytree(code / 'fastfill/comparison', old / 'fastfill/comparison',
                    ignore=shutil.ignore_patterns('tests', '__pycache__'))
    target = old / 'fastfill/comparison/evaluate_llm.py'
    target.write_text(target.read_text().replace('enforce_sizes', 'enforce'))
    bad = subprocess.run(['bash', str(script), str(old)], env=env, capture_output=True, text=True)
    assert bad.returncode == 1 and 'cannot score size-fixed runs' in bad.stderr
