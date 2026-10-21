"""Offline matched Qwen3 SFT: common CPU admission, sequential one-GPU arms.

python -m fastfill.comparison.train_pair --data PAIRED_ROOT --base LOCAL_QWEN3 \
    --out NEW_DIRECTORY [--preflight-only] [--max-steps 8 | --epochs 2]

Eight steps are the engineering default. Matched SFT requires an explicit
parent-locked budget and checkpoint interval; two dev probes are not quality evidence.
Full epochs require a separate signal/promotion decision; this driver never
publishes, resumes, filters flags, repairs predictions, or downloads models.
Use the parent's shared allocation timeout to bound total wall time.
"""
import argparse
import gc
import importlib.metadata
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time

from fastfill.comparison import train_pair_helpers as h

LORA = {'r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05, 'bias': 'none',
        'task_type': 'CAUSAL_LM',
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
SEED = 42
LR = 5e-5
MAX_NEW_TOKENS = 2048


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def _positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('data', 'base', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--max-steps', type=_positive_int)
    budget.add_argument('--epochs', type=_positive_float)
    parser.add_argument('--max-len', type=_positive_int, default=8192)
    parser.add_argument('--grad-accum', type=_positive_int, default=4)
    parser.add_argument('--bs', type=_positive_int, default=1)
    parser.add_argument('--arm', choices=(*h.ARMS, 'both'), default='both')
    parser.add_argument('--run-purpose', choices=('engineering', 'matched_sft'), default='engineering')
    parser.add_argument('--checkpoint-steps', type=_positive_int)
    parser.add_argument('--max-new-tokens', type=_positive_int, default=MAX_NEW_TOKENS,
                        help='decoding cap for the two dev probes only')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if args.run_purpose == 'matched_sft' and (args.max_steps is None and args.epochs is None):
        parser.error('matched_sft requires an explicit parent-approved step or epoch budget')
    if args.run_purpose == 'matched_sft' and args.checkpoint_steps is None:
        parser.error('matched_sft requires --checkpoint-steps')
    return argparse.Namespace(**{**vars(args), 'max_steps':
        8 if args.max_steps is None and args.epochs is None else args.max_steps})


def _tokenizer(base):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    return tok


def _versions():
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers', 'safetensors'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not_installed'
    return versions


def _config(args):
    return {**{k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'lora': LORA, 'learning_rate': LR, 'seed': SEED, 'data_seed': SEED,
            'dtype': 'bfloat16', 'attention': 'sdpa', 'gradient_checkpointing': True,
            'gradient_checkpointing_kwargs': {'use_reentrant': False},
            'lr_scheduler_type': 'cosine', 'weight_decay': 0.0, 'max_grad_norm': 1.0,
            'dev_generations_per_arm': 2,
            'do_sample': False, 'num_beams': 1, 'enable_thinking': False,
            'visible_gpus_required': 1, 'world_size_required': 1,
            'walltime_control': 'parent allocation/shared timeout',
            'disk_free_bytes_before': shutil.disk_usage(args.out).free,
            'versions': _versions(), 'quality_claim': False,
            'resume_supported': False, 'flag_filter': None,
            'resume_limitation': 'fresh runs only; periodic saves are evidence, not approved resume points',
            'save_strategy': 'steps' if args.checkpoint_steps else 'no', 'save_total_limit': 2,
            'evaluator': 'fastfill.comparison.contract.validate_output',
            'evidence_scope': args.run_purpose + '_execution_two_dev_cases_no_quality_promotion'}


def _code_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ['fastfill/comparison/train_pair.py', 'fastfill/comparison/train_pair_helpers.py',
             'fastfill/comparison/contract.py', 'fastfill/scene.py', 'fastfill/train.py']
    return {p: h.sha256(root / p) for p in paths}


def _dev_rooms(root, data_receipt, uids):
    name = 'dev_rooms.jsonl'
    if name not in data_receipt['files']:
        raise ValueError('root manifest must hash dev_rooms.jsonl for the exact contract evaluator')
    rooms = h._rows(root, name, data_receipt['files'][name])
    indexed = h._by_uid(rooms)
    if set(indexed) != set(uids):
        raise ValueError('dev_rooms UID coverage mismatch')
    return indexed


def prepare(args):
    """Write identity before model loading; persist all joint admission decisions."""
    pairs, data = h.load_pair(args.data)
    if args.run_purpose == 'matched_sft' and data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('matched_sft requires the qualified main target policy')
    base = h.base_receipt(args.base)
    if args.max_len + args.max_new_tokens > base['max_position_embeddings']:
        raise ValueError('max length plus generation budget exceeds base context')
    config = _config(args)
    receipt = {'config': config, 'data': data, 'base': base, 'code_sha256': _code_hashes()}
    h.write_json(args.out / 'run_receipt.json', receipt)
    tok = _tokenizer(args.base)
    reports, encoded = {}, {}
    for split in h.SPLITS:
        try:
            encoded[split], reports[split] = h.common_admission(tok, pairs[split], args.max_len)
        except h.EmptyAdmission as exc:
            reports[split] = exc.report
        h.write_json(args.out / 'lengths.json', reports)
    if any(not r['accepted_uids'] for r in reports.values()):
        raise ValueError('empty common admission; all length drops preserved in lengths.json')
    dev_uids = reports['dev']['accepted_uids'][:2]
    if len(dev_uids) != 2:
        raise ValueError('need at least two commonly admitted dev UIDs')
    rooms = _dev_rooms(args.data, data, [r['uid'] for r in pairs['dev']['joint']])
    dev = {a: [h._by_uid(pairs['dev'][a])[uid] for uid in dev_uids] for a in h.ARMS}
    h.write_json(args.out / 'dev_selection.json', {'uids': dev_uids, 'selection': 'first_two_sorted_common_UIDs'})
    return config, encoded['train'], dev, {uid: rooms[uid] for uid in dev_uids}


def _release_gpu(torch):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    remaining = torch.cuda.memory_allocated()
    if remaining:
        raise RuntimeError('model tensor allocations remain before fresh-base reload')


def _load_model(config):
    import torch
    from transformers import AutoModelForCausalLM
    # The cached base may be shared with other work. Refuse changed bytes on
    # every new load, including clean-load probes and the second arm.
    receipt_path = Path(config['out']) / 'run_receipt.json'
    receipt = h.strict_json(receipt_path.read_text())
    h.verify_artifacts(Path(config['base']), receipt['base']['files'])
    return AutoModelForCausalLM.from_pretrained(
        config['base'], local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': 0})


def _trainer_class():
    from transformers import Trainer

    class FiniteTrainer(Trainer):
        """Stop on the actual microbatch loss before logging can hide NaNs."""
        def compute_loss(self, model, inputs, *args, **kwargs):
            total = int(inputs['attention_mask'].sum().item())
            supervised = int((inputs['labels'] != -100).sum().item())
            result = super().compute_loss(model, inputs, *args, **kwargs)
            loss = result[0] if isinstance(result, tuple) else result
            number = float(loss.detach().item())
            if not math.isfinite(number):
                raise FloatingPointError('nonfinite microbatch loss')
            self.pair_tokens += total
            self.pair_supervised += supervised
            import json
            with (Path(self.args.output_dir) / 'loss.jsonl').open('a') as stream:
                stream.write(json.dumps({'optimizer_step_before': self.state.global_step,
                                         'loss': number, 'input_tokens': total,
                                         'supervised_tokens': supervised}, allow_nan=False) + '\n')
            return result
    return FiniteTrainer


def _train(config, arm_dir, train, tok):
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, TrainingArguments, set_seed
    model = trainer = None
    try:
        set_seed(SEED)
        model = _load_model(config)
        model.config.use_cache = False
        model.enable_input_require_grads()
        set_seed(SEED)
        model = get_peft_model(model, LoraConfig(**LORA))
        steps = config['max_steps'] or math.ceil(
            math.ceil(math.ceil(len(train) / config['bs']) / config['grad_accum']) * config['epochs'])
        args = TrainingArguments(
            output_dir=str(arm_dir), per_device_train_batch_size=config['bs'],
            gradient_accumulation_steps=config['grad_accum'], learning_rate=LR,
            max_steps=config['max_steps'] if config['max_steps'] is not None else -1,
            num_train_epochs=config['epochs'] if config['epochs'] is not None else 1,
            seed=SEED, data_seed=SEED, bf16=True, optim='adamw_torch',
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            lr_scheduler_type='cosine', warmup_steps=max(1, int(0.03 * steps)),
            weight_decay=0.0, max_grad_norm=1.0, logging_steps=1, logging_nan_inf_filter=False,
            eval_strategy='no', save_strategy='steps' if config.get('checkpoint_steps') else 'no',
            save_steps=config.get('checkpoint_steps') or 500, save_total_limit=2,
            save_only_model=False, report_to='none',
            remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=False)
        h.write_json(arm_dir / 'training_args.json', args.to_dict())
        trainer = _trainer_class()(model=model, args=args, train_dataset=train,
            callbacks=[_checkpoint_callback()] if config.get('checkpoint_steps') else [],
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
        trainer.pair_tokens = trainer.pair_supervised = 0
        result = trainer.train()
        if trainer.state.global_step != steps or not math.isfinite(result.training_loss):
            raise RuntimeError('training incomplete or nonfinite aggregate loss')
        for event in trainer.state.log_history:
            if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in event.values()):
                raise FloatingPointError('nonfinite trainer metric')
        trainer.save_model(str(arm_dir / 'adapter'))
        tok.save_pretrained(arm_dir / 'adapter')
        trainer.save_state()
        metrics = {'global_step': trainer.state.global_step, 'finite_loss': True,
                   'training_loss': result.training_loss, 'input_tokens_seen': trainer.pair_tokens,
                   'supervised_tokens_seen': trainer.pair_supervised, 'trainer_metrics': result.metrics}
        h.write_json(arm_dir / 'training_metrics.json', metrics)
        return metrics
    finally:
        if trainer is not None:
            trainer.accelerator.free_memory()
        del trainer, model


def _checkpoint_inventory(directory):
    expected = ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                'scheduler.pt', 'rng_state.pth', 'trainer_state.json')
    if any(not (directory / name).is_file() or not (directory / name).stat().st_size for name in expected):
        raise ValueError('incomplete periodic checkpoint training/RNG state')
    manifest = h.artifact_manifest(directory)
    h.write_json(directory / 'SHA256.json', manifest)
    h.verify_artifacts(directory, manifest)
    return manifest


def _checkpoint_callback():
    from transformers import TrainerCallback

    class CheckpointEvidence(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            _checkpoint_inventory(Path(args.output_dir) / f'checkpoint-{state.global_step}')
            return control
    return CheckpointEvidence()


def _completion_status(config):
    return 'TRAINING_COMPLETE' if config.get('run_purpose') == 'matched_sft' else 'ENGINEERING_COMPLETE'


def _adapter_inventory(arm_dir):
    expected = ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors',
                'adapter/tokenizer_config.json', 'adapter/tokenizer.json', 'trainer_state.json']
    if any(not (arm_dir / name).is_file() or not (arm_dir / name).stat().st_size for name in expected):
        raise ValueError('incomplete saved adapter/tokenizer/trainer state')
    manifest = h.artifact_manifest(arm_dir)
    h.write_json(arm_dir / 'SHA256.json', manifest)
    h.verify_artifacts(arm_dir, manifest)
    return manifest


def _generate(config, arm_dir, arm, dev, rooms):
    import torch
    from peft import PeftModel
    from transformers import set_seed
    from fastfill.comparison.contract import validate_output
    model = base = None
    results = []
    try:
        set_seed(SEED)
        tok = _tokenizer(arm_dir / 'adapter')
        base = _load_model(config)
        model = PeftModel.from_pretrained(base, str(arm_dir / 'adapter'),
                                          local_files_only=True, is_trainable=False)
        model.eval()
        model.config.use_cache = True
        for record in dev:
            prompt = tok.apply_chat_template(record['messages'][:-1], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, add_special_tokens=False, return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, num_beams=1,
                    max_new_tokens=config.get('max_new_tokens', MAX_NEW_TOKENS), eos_token_id=tok.get_vocab()['<|im_end|>'],
                    pad_token_id=tok.pad_token_id)
            generated = output[0, inputs['input_ids'].shape[1]:]
            raw = tok.decode(generated, skip_special_tokens=True)
            parsed = h.parse_prediction(raw, record, arm)
            try:
                validate_output(raw, rooms[record['uid']], arm)
            except ValueError:
                parsed = {'valid': False, 'error': 'contract.validate_output rejected', 'placements': None}
            result = {'uid': record['uid'], 'raw': raw, 'schema': parsed,
                      'generated_tokens': len(generated),
                      'reached_token_cap': len(generated) >= config.get('max_new_tokens', MAX_NEW_TOKENS)}
            results.append(result)
            h.write_json(arm_dir / 'dev_generations.json', results)
            del inputs, output, generated
        return {'clean_load': True, 'dev_count': len(results),
                'schema_valid': sum(r['schema']['valid'] for r in results), 'quality_claim': False}
    finally:
        del model, base


def _arm_worker(config, arm, train, dev, rooms):
    """Training only: the coordinator must join this process before probing."""
    import torch
    arm_dir = Path(config['out']) / arm
    arm_dir.mkdir()
    start, phase = time.monotonic(), 'runtime'
    try:
        if int(os.environ.get('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1:
            raise RuntimeError('requires one process and exactly one visible GPU')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 GPU required')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        h.write_json(arm_dir / 'runtime.json', {'gpu': torch.cuda.get_device_name(0),
            'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory, 'versions': _versions()})
        phase = 'training'
        metrics = _train(config, arm_dir, train, _tokenizer(config['base']))
        phase = 'inventory_and_hash'
        h.write_json(arm_dir / 'training_result.json', {**metrics, 'status': 'TRAINING_SAVED',
            'arm': arm, 'quality_claim': False, 'training_runtime_seconds': time.monotonic() - start,
            'training_gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'training_gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        _adapter_inventory(arm_dir)
    except Exception as exc:
        h.write_json(arm_dir / 'failure.json', {'status': 'FAIL', 'phase': phase,
            'error_type': type(exc).__name__, 'runtime_seconds': time.monotonic() - start})
        raise


def _probe_worker(config, arm, dev, rooms):
    """Fresh CUDA context; no training model or allocator survives into here."""
    import torch
    arm_dir = Path(config['out']) / arm
    start = time.monotonic()
    try:
        if int(os.environ.get('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1:
            raise RuntimeError('requires one process and exactly one visible GPU')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 GPU required')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        h.verify_artifacts(arm_dir, h.strict_json((arm_dir / 'SHA256.json').read_text()))
        training = h.strict_json((arm_dir / 'training_result.json').read_text())
        if training.get('status') != 'TRAINING_SAVED' or training.get('finite_loss') is not True:
            raise RuntimeError('training evidence incomplete before probe')
        evaluation = _generate(config, arm_dir, arm, dev, rooms)
        if evaluation.get('clean_load') is not True or evaluation.get('dev_count') != 2:
            raise RuntimeError('clean-load probe evidence incomplete')
        h.write_json(arm_dir / 'result.json', {**training, **evaluation, 'status': _completion_status(config),
            'run_purpose': config.get('run_purpose', 'engineering'), 'resume_supported': False,
            'probe_runtime_seconds': time.monotonic() - start,
            'probe_gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'probe_gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        _adapter_inventory(arm_dir)
    except Exception as exc:
        h.write_json(arm_dir / 'failure.json', {'status': 'FAIL', 'phase': 'fresh_base_reload_and_dev',
            'error_type': type(exc).__name__, 'runtime_seconds': time.monotonic() - start})
        raise


def _run_stage(out, arm, stage, target, arguments):
    """CUDA-free coordinator: a successful terminal exit gates each next stage."""
    h.write_json(out / 'status.json', {'status': 'UNVERIFIED', 'arm': arm, 'phase': stage,
                                     'training_executed': True, 'quality_claim': False})
    process = multiprocessing.get_context('spawn').Process(target=target, args=arguments)
    process.start()
    try:
        process.join()
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
    exitcode = process.exitcode
    h.write_json(out / f'{arm}.{stage}.exit.json', {'arm': arm, 'stage': stage,
        'pid': getattr(process, 'pid', None), 'exitcode': exitcode, 'process_exited': not process.is_alive()})
    process.close()
    if exitcode != 0:
        raise RuntimeError(f'{arm} failed during {stage}; preserving artifacts, no next stage')
    directory = out / arm
    h.verify_artifacts(directory, h.strict_json((directory / 'SHA256.json').read_text()))


def run(args):
    """New output only. Spawn and join one arm at a time; never auto-retry."""
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    try:
        h.write_json(args.out / 'status.json', {'status': 'UNVERIFIED', 'phase': 'preflight'})
        config, train, dev, rooms = prepare(args)
        selected = list(h.ARMS) if args.arm == 'both' else [args.arm]
        completed = []
        if args.preflight_only:
            status = {'status': 'PREFLIGHT_COMPLETE', 'training_executed': False, 'quality_claim': False}
        else:
            for arm in selected:
                _run_stage(args.out, arm, 'training', _arm_worker, (config, arm, train[arm], dev[arm], rooms))
                training = h.strict_json((args.out / arm / 'training_result.json').read_text())
                if training.get('status') != 'TRAINING_SAVED' or training.get('finite_loss') is not True:
                    raise RuntimeError('training completion evidence incomplete')
                _run_stage(args.out, arm, 'probe', _probe_worker, (config, arm, dev[arm], rooms))
                result = h.strict_json((args.out / arm / 'result.json').read_text())
                if result['status'] != _completion_status(config) or not result['clean_load'] or result['dev_count'] != 2:
                    raise RuntimeError('arm completion evidence incomplete')
                h.verify_artifacts(args.out / arm, h.strict_json((args.out / arm / 'SHA256.json').read_text()))
                completed.append(arm)
            status = {'status': _completion_status(config), 'training_executed': True, 'quality_claim': False}
        status = {**status, 'selected_arms': selected, 'completed_arms': completed,
                  'pair_complete': completed == list(h.ARMS), 'resume_supported': False,
                  'process_isolation_complete': bool(completed) and completed == selected}
        h.write_json(args.out / 'status.json', {**status, 'runtime_seconds': time.monotonic() - start})
        h.write_json(args.out / 'SHA256.json', h.artifact_manifest(args.out))
        return status
    except Exception as exc:
        h.write_json(args.out / 'status.json', {'status': 'FAIL', 'error_type': type(exc).__name__,
            'runtime_seconds': time.monotonic() - start, 'quality_claim': False})
        h.write_json(args.out / 'SHA256.json', h.artifact_manifest(args.out))
        raise


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
