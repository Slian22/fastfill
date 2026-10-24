"""Amendment 021 train_dpo flags on CPU: token-mean loss across ranks and accumulation windows, answer-only logits,
the render / EOS check, and a tiny multimodal Qwen3.5 run (LoRA regex scope, gradient assertion, perf records)."""
import json
from pathlib import Path
import sys
import types

import pytest
import torch

pytest.importorskip('peft', reason='train_dpo needs peft')
from safetensors.torch import load_file  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from fastfill.comparison import a021_train  # noqa: E402
from fastfill.v1_1 import train_dpo  # noqa: E402

STOP = '<|im_end|>'


class Tok:
    """Characters as ids 2..61, <|im_end|> = 1; the chat template is the last turn plus '|'. merge: '|{' is one token 3
    (a render whose prompt is not a token prefix of prompt + answer)."""
    pad_token = None

    def __init__(self, eos=STOP, merge=False):
        self.eos_token, self.merge = eos, merge

    @property
    def pad_token_id(self):
        return 1

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return messages[-1]['content'] + '|'

    def _ids(self, text):
        ids, k = [], 0
        while k < len(text):
            if text.startswith(STOP, k):
                ids, k = ids + [1], k + len(STOP)
            elif self.merge and text.startswith('|{', k):
                ids, k = ids + [3], k + 2
            else:
                ids, k = ids + [2 + ord(text[k]) % 60], k + 1
        return ids

    def __call__(self, texts, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        return {'input_ids': self._ids(texts) if isinstance(texts, str) else [self._ids(t) for t in texts]}

    def save_pretrained(self, path):
        for name in ('tokenizer_config.json', 'tokenizer.json'):
            (Path(path) / name).write_text('{}')


ROWS = [{'uid': f'u{k}', 'prompt': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': f'room {k}' * (k + 1)}],
         'chosen': f'{{"a": {k}}}', 'rejected': f'{{"a": {k}}}'} for k in range(4)]


def pairs(tmp_path, rows=ROWS):
    path = tmp_path / 'rows.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def test_token_mean_loss_is_the_answer_token_mean_over_all_ranks_of_each_window():
    torch.manual_seed(0)
    world, accum, batches = 4, 2, 3  # Windows [0, 1] and the partial [2].
    nll = [[torch.rand(2) * 5 for _ in range(batches)] for _ in range(world)]  # Per sequence: summed answer NLL.
    tokens = [[torch.randint(1, 20, (2,)) for _ in range(batches)] for _ in range(world)]
    counts = [[int(t.sum()) for t in rank] for rank in tokens]
    windows = [train_dpo.window_tokens(c, accum) for c in counts]
    assert train_dpo.window_tokens([3, 4, 5], 2) == [7, 7, 5] and train_dpo.window_tokens([3, 4, 5], 1) == [3, 4, 5]
    w = torch.ones((), requires_grad=True)
    for start, stop in ((0, 2), (2, 3)):
        grads = []
        for r in range(world):  # Each rank accumulates its micro-batches; DDP then averages the rank gradients.
            grad = 0
            for b in range(start, stop):
                total = torch.tensor(sum(windows[q][b] for q in range(world)))  # The all-reduce.
                grad = grad + torch.autograd.grad(train_dpo.token_mean_loss(-w * nll[r][b], total, world), w)[0]
            grads.append(grad)
        want = (sum(nll[r][b].sum() for r in range(world) for b in range(start, stop))
                / sum(counts[r][b] for r in range(world) for b in range(start, stop)))
        assert torch.allclose(sum(grads) / world, want)


def tiny_qwen3(tmp_path):
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=128, tie_word_embeddings=False)
    return Qwen3ForCausalLM(config).eval()


def test_answer_logits_keep_only_answer_positions_and_give_the_same_sums(tmp_path):
    model, calls = tiny_qwen3(tmp_path), []

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model

        def forward(self, **kwargs):
            calls.append(kwargs.get('logits_to_keep'))
            return self.inner(**kwargs)
    rows = train_dpo.DPOPairDataset(Tok(), pairs(tmp_path)).rows
    batch = train_dpo.collate_dpo(rows, 1)  # Right padded; prompts of four lengths.
    args = [batch[k] for k in ('c_input_ids', 'c_attention_mask', 'c_labels')]
    with torch.no_grad():
        full, answer = train_dpo.sequence_logps(Spy(), *args), train_dpo.sequence_logps(Spy(), *args, answer_only=True)
    first = min(r['c_labels'].index(next(v for v in r['c_labels'] if v != -100)) for r in rows)
    assert calls == [None, batch['c_input_ids'].shape[1] - first + 1] and calls[1] < batch['c_input_ids'].shape[1]
    assert torch.allclose(full, answer, atol=1e-5)


def test_render_check_requires_the_stop_token_and_a_token_prefix_render(tmp_path):
    path = pairs(tmp_path)
    assert len(train_dpo.DPOPairDataset(Tok(), path, render_check=True)) == 4
    with pytest.raises(ValueError, match='not the inference stop token'):
        train_dpo.DPOPairDataset(Tok(eos='<e>'), path, render_check=True)
    assert len(train_dpo.DPOPairDataset(Tok(merge=True), path)) == 0  # Without the check: silently skipped.
    with pytest.raises(ValueError, match='not a prefix'):
        train_dpo.DPOPairDataset(Tok(merge=True), path, render_check=True)


def tiny_qwen35(path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    text = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, layer_types=['linear_attention'] * 3 + ['full_attention'],
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                linear_conv_kernel_dim=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=64,
                max_position_embeddings=512, tie_word_embeddings=False)
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=4,
                  spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=16, deepstack_visual_indexes=[])
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False)).save_pretrained(path)
    Tok().save_pretrained(path)  # Persist the mocked tokenizer assets for the trainer's input identity.
    return path


def run(monkeypatch, capsys, model, rows, out, *extra):
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(train_dpo, 'AutoTokenizer', types.SimpleNamespace(from_pretrained=lambda path: Tok()))
    monkeypatch.setattr(sys, 'argv', ['train_dpo', '--model', str(model), '--pairs', str(rows), '--out', str(out),
                                      '--epochs', '1', '--bs', '1', '--grad_accum', '2', '--lr', '1e-3', '--lora_r', '4',
                                      '--lora_alpha', '8', '--save_steps', '0', '--sft_only', *extra])
    capsys.readouterr()
    train_dpo.main()
    return [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{')]


def test_tiny_multimodal_qwen35_trains_only_language_model_linears_with_every_grad(tmp_path, monkeypatch, capsys):
    model, rows = tiny_qwen35(tmp_path / 'qwen35'), pairs(tmp_path)
    seen = []
    monkeypatch.setattr(a021_train, 'base_receipt', lambda path: seen.append(path) or {'path': str(path), 'files': {'f': '0' * 64}})
    flags = ('--sft_token_mean', '--answer_logits', '--check_render', '--image_text_model', '--base_receipt',
             '--lora_regex', a021_train.LORA_27B)
    warmups, schedule = [], train_dpo.get_cosine_schedule_with_warmup
    monkeypatch.setattr(train_dpo, 'get_cosine_schedule_with_warmup',
                        lambda opt, **kw: warmups.append(kw['num_warmup_steps']) or schedule(opt, **kw))
    logs = run(monkeypatch, capsys, model, rows, tmp_path / 'out', *flags, '--perf_out', str(tmp_path / 'perf'),
               '--warmup_ratio', '1.0')
    assert warmups == [2]  # max(1, int(1.0 x 2 steps)); the 5% default gives 1 (the identity tests pin it).
    assert [x['step'] for x in logs] == [1, 2] and all(x['loss'] == x['chosen_nll'] > 0 for x in logs)
    keys = load_file(str(tmp_path / 'out/final/adapter_model.safetensors')).keys()
    modules = {k.split('.lora_')[0].removeprefix('base_model.model.') for k in keys}
    assert all(m.startswith('model.language_model.layers.') for m in modules) and len(modules) == 3 * 6 + 7
    assert {m.split('.', 4)[-1] for m in modules} == {
        'linear_attn.in_proj_qkv', 'linear_attn.in_proj_z', 'linear_attn.out_proj', 'self_attn.q_proj', 'self_attn.k_proj',
        'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'}  # No vision, b/a, head.
    manifest = json.loads((tmp_path / 'out/run_manifest.json').read_text())
    assert manifest['args']['lora_regex'] == a021_train.LORA_27B and manifest['args']['sft_token_mean'] is True
    assert manifest['args']['warmup_ratio'] == 1.0
    assert seen == [str(model)] and manifest['base_receipt']['path'] == str(model)
    perf = json.loads((tmp_path / 'perf/perf-rank0.json').read_text())
    assert [s['step'] for s in perf['steps']] == [1, 2] and perf['max_memory_reserved'] is None  # CPU: no CUDA peak.
    assert all(s['input_tokens'] > 0 and s['seconds'] > 0 for s in perf['steps'])
    vision = r'model\.visual\.blocks\.0\.attn\.qkv|' + a021_train.LORA_27B  # A frozen-path target never gets a grad.
    with pytest.raises(ValueError, match='no gradient after step 1'):
        run(monkeypatch, capsys, model, rows, tmp_path / 'bad', '--image_text_model', '--lora_regex', vision)
