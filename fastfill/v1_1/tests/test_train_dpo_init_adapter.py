"""Amendment 013: train_dpo --init_adapter on a tiny random Qwen3 on CPU (the policy continues a trained adapter and
a frozen copy of it is the DPO reference); amendment 016: --sft_only (chosen mean token NLL, no reference or rejected
pass). Without either flag training is bit-identical to the pre-013 train_dpo."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip('peft', reason='train_dpo needs peft')
from peft import PeftModel, get_peft_model  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from fastfill.v1_1 import train_dpo  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BEFORE = 'cf518a24b'  # The amendment-013 tooling commit: train_dpo.py before --init_adapter.
R, ALPHA = 4, 8
PAIRS = [{'prompt': [{'role': 'user', 'content': f'room {k}'}], 'chosen': f'{{"a": {k}}}', 'rejected': f'{{"a": {k + 5}}}'}
         for k in range(3)]


class Tok:
    """Characters as ids 2..61, EOS '<e>' = 1; the chat template is the user text plus a separator."""
    eos_token, pad_token = '<e>', None

    @property
    def pad_token_id(self):
        return 1 if self.pad_token == self.eos_token else None

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]['content'] + '|'

    def _ids(self, text):
        eos = text.endswith(self.eos_token)
        return [2 + ord(c) % 60 for c in (text[:-len(self.eos_token)] if eos else text)] + [1] * eos

    def __call__(self, texts, add_special_tokens):
        return {'input_ids': [self._ids(t) for t in texts]}

    def save_pretrained(self, path):
        for name in ('tokenizer_config.json', 'tokenizer.json'):
            (Path(path) / name).write_text('{}')


def base(w, dtype=torch.float32):
    return Qwen3ForCausalLM.from_pretrained(w.model, dtype=dtype, attn_implementation='sdpa')


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A saved tiny Qwen3, a trained-looking adapter on it (LoRA B non-zero) and three pairs."""
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(key, raising=False)
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=128, tie_word_embeddings=False)
    Qwen3ForCausalLM(config).save_pretrained(tmp_path / 'model')
    Tok().save_pretrained(tmp_path / 'model')  # The mocked tokenizer is still part of the on-disk model identity.
    w = types.SimpleNamespace(root=tmp_path, model=tmp_path / 'model', init=tmp_path / 'init', pairs=tmp_path / 'pairs.jsonl')
    adapter = get_peft_model(base(w), train_dpo.lora_config(R, ALPHA))
    with torch.no_grad():
        for name, p in adapter.named_parameters():
            if 'lora_' in name:
                p.add_(torch.randn_like(p) * .3)
    adapter.save_pretrained(w.init)
    w.pairs.write_text(''.join(json.dumps(p) + '\n' for p in PAIRS))
    return w


def run(module, monkeypatch, capsys, w, out, *extra):
    """module.main() in process (one CPU rank) -> its JSON step logs."""
    monkeypatch.setattr(module, 'AutoTokenizer', types.SimpleNamespace(from_pretrained=lambda path: Tok()))
    monkeypatch.setattr(sys, 'argv', ['train_dpo', '--model', str(w.model), '--pairs', str(w.pairs), '--out', str(out),
                                      '--epochs', '2', '--grad_accum', '2', '--lr', '1e-3', '--lora_r', str(R),
                                      '--lora_alpha', str(ALPHA), '--save_steps', '1', '--rpo_alpha', '1.0', *extra,
                                      '--save_every_epoch'])
    capsys.readouterr()
    module.main()
    return [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{')]


def batch(w):
    rows = train_dpo.DPOPairDataset(Tok(), w.pairs)
    tok = Tok()
    tok.pad_token = tok.eos_token
    return train_dpo.collate_dpo([rows[k] for k in range(len(rows))], tok.pad_token_id)


def logps(model, b, side):
    return train_dpo.sequence_logps(model, b[f'{side}_input_ids'], b[f'{side}_attention_mask'], b[f'{side}_labels'])


def lora(model, name):
    return {k: p.detach().clone() for k, p in model.named_parameters() if f'.{name}.' in k and 'lora_' in k}


def test_without_init_adapter_training_is_bit_identical_to_before(tiny, monkeypatch, capsys):
    try:
        source = subprocess.run(['git', '-C', str(ROOT), 'show', f'{BEFORE}:fastfill/v1_1/train_dpo.py'],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f'needs the git history ({BEFORE}) for the pre-amendment-013 train_dpo')
    before = types.ModuleType('train_dpo_before')
    exec(compile(source, 'train_dpo_before.py', 'exec'), before.__dict__)
    logs = [run(m, monkeypatch, capsys, tiny, tiny.root / name) for m, name in ((before, 'before'), (train_dpo, 'after'))]
    assert logs[0] == logs[1] and [x['step'] for x in logs[0]] == [1, 4]
    for name in ('final', 'epoch-01', 'epoch-02', 'ckpt_latest'):
        for file in ('adapter_model.safetensors', 'adapter_config.json'):
            assert (tiny.root / 'before' / name / file).read_bytes() == (tiny.root / 'after' / name / file).read_bytes()
    manifests = [json.loads((tiny.root / name / 'run_manifest.json').read_text())['args'] for name in ('before', 'after')]
    assert [{k: v for k, v in m.items() if k != 'out'} for m in manifests][0] == {
        k: v for k, v in manifests[1].items() if k != 'out'}  # No init_adapter key: byte-identical apart from --out.
    assert not {'init_adapter', 'sft_only'} & manifests[1].keys()  # Amendment 016: --sft_only off leaves no trace.
    assert not set(train_dpo.OPTIONAL) & manifests[1].keys()  # Amendment 021 flags off: none of them either.


def test_init_adapter_policy_equals_frozen_reference_at_step_zero_in_fp32(tiny):
    model = train_dpo.init_policy(base(tiny), tiny.init, train_dpo.lora_config(R, ALPHA))
    policy, reference = lora(model, train_dpo.POLICY), lora(model, train_dpo.REFERENCE)
    assert policy.keys() == {k.replace('.reference.', '.default.') for k in reference} and len(policy) == 28
    assert all(torch.equal(policy[k], reference[k.replace('.default.', '.reference.')]) for k in policy)
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    assert trainable == set(policy) and model.active_adapter == train_dpo.POLICY
    b = batch(tiny)
    model.train()  # As in training: the reference has no dropout, so two reference passes agree exactly.
    refs = []
    for _ in range(2):
        with torch.no_grad(), train_dpo.reference_policy(model, tiny.init):
            refs.append([logps(model, b, s) for s in 'cr'])
    assert model.active_adapter == train_dpo.POLICY and {k for k, p in model.named_parameters() if p.requires_grad} == trainable
    assert all(torch.equal(a, c) for a, c in zip(*refs))
    model.eval()  # The policy without dropout is the reference bit for bit: every pair's DPO loss is log 2.
    with torch.no_grad():
        pol = [logps(model, b, s) for s in 'cr']
        exact = PeftModel.from_pretrained(base(tiny), str(tiny.init)).eval()  # The exact policy: base + adapter.
        want = [logps(exact, b, s) for s in 'cr']
        with exact.disable_adapter():
            plain = logps(exact, b, 'c')
    assert all(torch.equal(p, r) and torch.equal(p, x) for p, r, x in zip(pol, refs[0], want))
    assert not torch.allclose(pol[0], plain)  # The adapter matters, so this is not the disabled-adapter reference.
    _, dpo, _, diff = train_dpo.dpo_loss(*pol, *refs[0], 0.1)
    assert torch.equal(diff, torch.zeros(3)) and math.isclose(dpo.item(), math.log(2), rel_tol=1e-7)
    assert all(torch.equal(-F.logsigmoid(0.1 * d), torch.tensor(math.log(2))) for d in diff)
    bf16 = train_dpo.init_policy(base(tiny, torch.bfloat16), tiny.init, train_dpo.lora_config(R, ALPHA))
    assert {p.dtype for k, p in bf16.named_parameters() if 'lora_' in k} == {torch.float32}  # FP32 as trained.
    with pytest.raises(ValueError, match='config differs'):
        train_dpo.init_policy(base(tiny), tiny.init, train_dpo.lora_config(R * 2, ALPHA))


def test_init_adapter_step_moves_only_the_policy_and_saves_one_plain_adapter(tiny, monkeypatch, capsys):
    model = train_dpo.init_policy(base(tiny), tiny.init, train_dpo.lora_config(R, ALPHA))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    before = {n: lora(model, n) for n in (train_dpo.POLICY, train_dpo.REFERENCE)}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
    b = batch(tiny)
    model.train()
    with torch.no_grad(), train_dpo.reference_policy(model, tiny.init):
        ref = [logps(model, b, s) for s in 'cr']
    loss = train_dpo.dpo_loss(*[logps(model, b, s) for s in 'cr'], *ref, 0.1)[0]
    train_dpo.backward_finite_loss(loss)
    opt.step()
    after = {n: lora(model, n) for n in before}
    assert all(torch.equal(v, after[train_dpo.REFERENCE][k]) for k, v in before[train_dpo.REFERENCE].items())
    assert any(not torch.equal(v, after[train_dpo.POLICY][k]) for k, v in before[train_dpo.POLICY].items())

    out = tiny.root / 'op'
    logs = run(train_dpo, monkeypatch, capsys, tiny, out, '--init_adapter', str(tiny.init))
    assert [x['step'] for x in logs] == [1, 4] and all(math.isfinite(x['loss']) for x in logs)
    init = load_file(str(tiny.init / 'adapter_model.safetensors'))
    for name in ('final', 'epoch-01', 'epoch-02', 'ckpt_latest'):
        assert not any(p.is_dir() for p in (out / name).iterdir())  # No per-adapter subdirectory: one adapter only.
        saved = load_file(str(out / name / 'adapter_model.safetensors'))
        assert saved.keys() == init.keys() and {t.dtype for t in saved.values()} == {torch.float32}
        assert any(not torch.equal(saved[k], init[k]) for k in init)
        assert json.loads((out / name / 'adapter_config.json').read_text())['r'] == R
    loaded = PeftModel.from_pretrained(base(tiny), str(out / 'final'))
    assert list(loaded.peft_config) == ['default']
    assert json.loads((out / 'run_manifest.json').read_text())['args']['init_adapter'] == str(tiny.init)
    final = (out / 'final/adapter_model.safetensors').read_bytes()
    shutil.rmtree(out / 'final')  # A rerun resumes ckpt_latest (the last step) into the policy, not the reference.
    assert run(train_dpo, monkeypatch, capsys, tiny, out, '--init_adapter', str(tiny.init)) == []
    assert (out / 'final/adapter_model.safetensors').read_bytes() == final


def test_sft_only_trains_the_chosen_mean_token_nll_without_reference_or_rejected_pass(tiny, monkeypatch, capsys):
    pol_c, tokens = torch.tensor([-6.0, -2.0]), torch.tensor([3, 2])
    zero = torch.zeros(2)  # sft_loss is exactly dpo_loss's NLL term (RPO without the DPO term).
    assert torch.equal(train_dpo.sft_loss(pol_c, tokens), train_dpo.dpo_loss(pol_c, zero, zero, zero, 0.1, 1.0, tokens)[2])
    assert math.isclose(train_dpo.sft_loss(pol_c, tokens).item(), (6 / 3 + 2 / 2) / 2)
    calls, real = [], train_dpo.sequence_logps

    def logps(model, input_ids, attention_mask, labels):
        calls.append((torch.is_grad_enabled(), input_ids.tolist()))
        return real(model, input_ids, attention_mask, labels)

    def no_reference(*args):
        raise AssertionError('--sft_only must not run the reference')
    monkeypatch.setattr(train_dpo, 'sequence_logps', logps)
    monkeypatch.setattr(train_dpo, 'reference_policy', no_reference)
    out = tiny.root / 'rft'
    logs = run(train_dpo, monkeypatch, capsys, tiny, out, '--init_adapter', str(tiny.init), '--rpo_alpha', '0', '--sft_only')
    assert [x['step'] for x in logs] == [1, 4] and all(set(x) == {'epoch', 'step', 'total_steps', 'chosen_nll', 'loss'}
                                                       and x['loss'] == x['chosen_nll'] > 0 for x in logs)
    chosen = [b['c_input_ids'] for b in train_dpo.DPOPairDataset(Tok(), tiny.pairs).rows]
    assert len(calls) == 2 * 3 and all(grad and ids[0] in chosen for grad, ids in calls)  # One chosen pass per row.
    init, final = (load_file(str(p / 'adapter_model.safetensors')) for p in (tiny.init, out / 'final'))
    assert final.keys() == init.keys() and any(not torch.equal(final[k], init[k]) for k in init)
    args = json.loads((out / 'run_manifest.json').read_text())['args']
    assert args['sft_only'] is True and args['init_adapter'] == str(tiny.init) and args['rpo_alpha'] == 0
    with pytest.raises(ValueError, match='--rpo_alpha must be 0'):  # No hidden DPO term.
        run(train_dpo, monkeypatch, capsys, tiny, tiny.root / 'bad', '--sft_only')


def test_init_policy_loads_both_adapters_on_cpu(tiny, monkeypatch):
    """A bare 'cuda' becomes cuda:0 in safetensors: every DDP rank would open GPU 0 (busy on exclusive-process nodes)."""
    import peft.peft_model as pm
    seen, real = [], pm.load_peft_weights

    def spy(model_id, device=None, **kw):
        seen.append(device)
        return real(model_id, device=device, **kw)
    monkeypatch.setattr(pm, 'load_peft_weights', spy)
    train_dpo.init_policy(base(tiny), tiny.init, train_dpo.lora_config(R, ALPHA))
    assert seen == ['cpu', 'cpu']
