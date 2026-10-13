"""Clean evaluations require traced training, completion and merge provenance."""
import json
from pathlib import Path
import shutil
import sys

import pytest

from fastfill.v1_1 import automate, training_guard as guard
from fastfill.v1_1.evaluation_protocol import file_identity, write_json_atomic
from fastfill.v1_1.tests.test_training_guard import sanitized_pairs, weight
from fastfill.v1_1.tests.test_automate import prepared, fake_evaluator


@pytest.fixture
def lineage_run(tmp_path, sanitized_pairs):
    run = tmp_path / 'fastfill_v1_1_clean_fixture'
    run.mkdir()
    shutil.copytree(sanitized_pairs, run / 'pairs')
    package = Path(guard.__file__).resolve().parent.parent
    frozen = tmp_path / 'frozen' / 'fastfill'
    names = {'v1_1/train_dpo.py', 'v1_1/training_guard.py', 'merge_lora.py'}
    audit = json.loads((run / 'pairs/audit.json').read_text())
    names |= set(audit['code_sha256'])
    for name in names:
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    sft = run / 'sft_merged'
    sft.mkdir()
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (sft / name).write_text('{}')
    weight(sft / 'model.safetensors')
    for stage, tag, base_name, merged_name in (
            (1, 'dpo1', 'sft_merged', 'dpo1_merged'),
            (2, 'dpo2', 'dpo1_merged', 'final_merged')):
        adapter = run / f'{tag}_lora'
        base, merged = run / base_name, run / merged_name
        settings = {'lr': 5e-6 if stage == 1 else 2e-6, 'epochs': 2, 'beta': 0.1,
                    'bs': 1, 'grad_accum': 4, 'max_len': 8192, 'lora_r': 32,
                    'lora_alpha': 64, 'save_steps': 150, 'seed': 42, 'world_size': 4}
        pairs = run / f'pairs/stage{stage}.cleaned.jsonl'
        guard.guard_training(base, pairs, adapter, settings,
                             [frozen / name for name in ('v1_1/train_dpo.py', 'v1_1/training_guard.py', 'merge_lora.py')])
        final = adapter / 'final'
        final.mkdir()
        for name in ('adapter_config.json', 'tokenizer_config.json', 'tokenizer.json'):
            (final / name).write_text('{}')
        weight(final / 'adapter_model.safetensors')
        write_json_atomic(adapter / 'run_manifest.json', {'args': {
            **{k: v for k, v in settings.items() if k != 'world_size'},
            'model': str(base), 'pairs': str(pairs), 'out': str(adapter)}})
        write_json_atomic(final / 'completion.json', {
            'schema_version': 1, 'kind': 'fastfill_v1.1_dpo_completion',
            'training_guard': file_identity(adapter / 'training_guard.json'),
            'admitted_pairs': 1, 'loader_batches': 1, 'world_size': 4,
            'epochs': 2, 'position': {'epoch': 1, 'it': 0, 'step': 2},
            'expected_total_steps': 2,
            'adapter_files': {name: file_identity(final / name) for name in (
                'adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json', 'tokenizer.json')}})
        shutil.copytree(base, merged)
        guard.seal_merge(base, final, merged)
    return run, frozen


def test_model_symlink_target_is_also_protected(prepared, tmp_path):
    _, data, run = prepared
    alias = tmp_path / 'alias-run'
    alias.mkdir()
    (alias / 'sft_merged').symlink_to(run / 'sft_merged', target_is_directory=True)
    with pytest.raises(ValueError, match='isolated'):
        automate.prepare(Path(automate.__file__).resolve().parents[2], data, alias,
                         run / 'sft_merged/evaluation', python=sys.executable)


def test_clean_runner_rejects_untraced_models_before_generation(prepared):
    output, data, run = prepared
    clean = run.with_name('fastfill_v1_1_clean_unbound')
    run.rename(clean)
    clean_out = output.with_name('clean_eval')
    automate.prepare(Path(automate.__file__).resolve().parents[2], data, clean,
                     clean_out, python=sys.executable)
    with pytest.raises((ValueError, FileNotFoundError), match='guard|lineage'):
        automate.run_model(clean_out, 'dpo1', executor=fake_evaluator)
    assert automate.report(clean_out)['completed_evaluations'] == 0


def test_new_completion_traces_both_stages_with_old_frozen_code(lineage_run):
    run, frozen = lineage_run
    result = guard.verify_training_lineage(run, 'final')
    assert result['tag'] == 'final'
    assert result['completion']['evidence'] == 'trainer_completion_receipt'
    assert result['parent']['tag'] == 'dpo1'
    assert result['source_package'] == str(frozen)


@pytest.mark.parametrize('change', ['no_merge', 'wrong_output', 'wrong_base', 'wrong_adapter',
                                     'wrong_pair', 'wrong_guard', 'wrong_code', 'wrong_recipe',
                                     'wrong_manifest', 'wrong_step', 'wrong_world', 'wrong_count'])
def test_lineage_rejects_independently_corrupted_evidence(lineage_run, change):
    run, frozen = lineage_run
    final = run / 'dpo1_lora/final'
    if change == 'no_merge':
        (run / 'dpo1_merged.merge.json').unlink()
    elif change in ('wrong_output', 'wrong_base', 'wrong_adapter'):
        path = {'wrong_output': run / 'dpo1_merged/model.safetensors',
                'wrong_base': run / 'sft_merged/model.safetensors',
                'wrong_adapter': final / 'adapter_model.safetensors'}[change]
        path.write_bytes(path.read_bytes()[:-4] + b'1234')
    elif change == 'wrong_pair':
        (run / 'pairs/stage1.cleaned.jsonl').write_text('{"uid":"changed"}\n')
    elif change == 'wrong_code':
        (frozen / 'v1_1/train_dpo.py').write_text('# changed frozen trainer\n')
    elif change in ('wrong_guard', 'wrong_recipe'):
        path = run / 'dpo1_lora/training_guard.json'
        value = json.loads(path.read_text())
        value['settings']['lr'] = 1.0
        write_json_atomic(path, value)
        if change == 'wrong_recipe':
            completion = json.loads((final / 'completion.json').read_text())
            write_json_atomic(final / 'completion.json', {**completion, 'training_guard': file_identity(path)})
    elif change == 'wrong_manifest':
        path = run / 'dpo1_lora/run_manifest.json'
        value = json.loads(path.read_text())
        value['args']['grad_accum'] = 100
        write_json_atomic(path, value)
    else:
        path = final / 'completion.json'
        value = json.loads(path.read_text())
        if change == 'wrong_step':
            value['position']['step'] = 1
        elif change == 'wrong_world':
            value['world_size'] = 2
        else:
            value['admitted_pairs'] = 0
        write_json_atomic(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        guard.verify_training_lineage(run, 'dpo1')


def test_cli_verifies_without_writing_original_run(lineage_run, monkeypatch, capsys):
    run, _ = lineage_run
    before = {str(p): file_identity(p) for p in run.rglob('*') if p.is_file()}
    monkeypatch.setattr(sys, 'argv', ['training_guard', 'verify-lineage', '--out', str(run), '--tag', 'final'])
    guard.main()
    assert json.loads(capsys.readouterr().out)['tag'] == 'final'
    assert before == {str(p): file_identity(p) for p in run.rglob('*') if p.is_file()}


def test_sft_is_explicitly_legacy_provenance(lineage_run):
    result = guard.verify_training_lineage(lineage_run[0], 'sft')
    assert result['completion']['evidence'] == 'legacy_sft_model_identity_only'


def test_legacy_final_requires_complete_matching_last_checkpoint(lineage_run, monkeypatch):
    import torch
    run, _ = lineage_run
    adapter = run / 'dpo1_lora'
    (adapter / 'final/completion.json').unlink()
    ckpt = adapter / 'ckpt_latest'
    ckpt.mkdir()
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        shutil.copyfile(adapter / 'final' / name, ckpt / name)
    torch.save({'state': {'epoch': 1, 'it': 0, 'step': 2}, 'optimizer': {}, 'scheduler': {}}, ckpt / 'trainer_state.pt')
    monkeypatch.setattr(guard, '_count_admitted_pairs', lambda *args: 1)
    result = guard.verify_training_lineage(run, 'dpo1')
    assert result['completion']['evidence'] == 'legacy_final_matches_complete_checkpoint'
    assert not (adapter / 'final/completion.json').exists()
    torch.save({'state': {'epoch': 1, 'it': 0, 'step': 1}}, ckpt / 'trainer_state.pt')
    with pytest.raises(ValueError, match='completion|step|position'):
        guard.verify_training_lineage(run, 'dpo1')


def test_clean_runner_binds_lineage_and_finishes_with_real_receipts(prepared, lineage_run):
    old_output, data, _ = prepared
    run, _ = lineage_run
    output = old_output.with_name('verified-clean-eval')
    automate.prepare(Path(automate.__file__).resolve().parents[2], data, run, output,
                     python=sys.executable)
    summary = automate.run_model(output, 'dpo1', executor=fake_evaluator)
    assert summary['completed_evaluations'] == 4
    state = json.loads((output / 'state/dpo1.json').read_text())
    assert state['training_lineage']['completion']['evidence'] == 'trainer_completion_receipt'
    guard_path = run / 'dpo1_lora/training_guard.json'
    guard_path.write_text(guard_path.read_text() + '\n')
    with pytest.raises(ValueError, match='completion|lineage'):
        automate.run_model(output, 'dpo1', executor=fake_evaluator)
    assert automate.report(output)['completed_evaluations'] == 0


def test_legacy_final_missing_checkpoint_does_not_synthesize_receipt(lineage_run):
    run, _ = lineage_run
    completion = run / 'dpo1_lora/final/completion.json'
    completion.unlink()
    with pytest.raises(FileNotFoundError):
        guard.verify_training_lineage(run, 'dpo1')
    assert not completion.exists()


def test_token_admission_recount_matches_length_and_prefix_rules(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    class Tokenizer:
        eos_token = '!'
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
            return messages[0]['content']
        def __call__(self, texts, **kwargs):
            assert kwargs == {'add_special_tokens': False}
            def tokenize(text):
                result = list(text.encode())
                return [999, *result[1:]] if text.startswith('boundarybad') else result
            return {'input_ids': [tokenize(text) for text in texts]}

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda _: Tokenizer())
    pairs = tmp_path / 'pairs.jsonl'
    rows = [
        {'prompt': [{'content': 'p'}], 'chosen': 'yes', 'rejected': 'no'},
        {'prompt': [{'content': 'p'}], 'chosen': 'x' * 100, 'rejected': 'no'},
        {'prompt': [{'content': 'boundary'}], 'chosen': 'bad', 'rejected': 'okay'},
    ]
    pairs.write_text('\n' + '\n'.join(json.dumps(row) for row in rows) + '\n')
    assert guard._count_admitted_pairs(tmp_path, pairs, 30) == 1


def test_invalid_tag_and_missing_cli_tag_fail_closed(lineage_run, monkeypatch):
    run, _ = lineage_run
    with pytest.raises(ValueError, match='tag'):
        guard.verify_training_lineage(run, '../invalid')
    monkeypatch.setattr(sys, 'argv', ['training_guard', 'verify-lineage', '--out', str(run)])
    with pytest.raises(SystemExit) as error:
        guard.main()
    assert error.value.code == 2


def test_real_trainer_completion_writer_matches_lineage_contract(lineage_run):
    from fastfill.v1_1.train_dpo import write_dpo_completion

    run, _ = lineage_run
    adapter = run / 'dpo1_lora'
    write_dpo_completion(adapter / 'final', adapter, admitted_pairs=1, loader_batches=1,
                         world_size=4, epochs=2, step=2, total_steps=2)
    result = guard.verify_training_lineage(run, 'dpo1')
    assert result['completion']['evidence'] == 'trainer_completion_receipt'


def test_legacy_recount_refuses_changed_prompt_renderer(tmp_path):
    (tmp_path / 'scene.py').write_text('# different prompt renderer\n')
    with pytest.raises(ValueError, match='prompt rendering'):
        guard._count_admitted_pairs(tmp_path, tmp_path / 'absent-pairs', 8192, tmp_path)
