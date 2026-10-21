# Independent current-training QA — 2026-10-13

**Decision: no evidence currently requires stopping or retraining jobs 7365681/7365682. Preserve the running matched SFT pair. Accuracy is unverified; this is a narrow execution audit, not model-quality approval.** Increasing inference allowance to 40,960 total tokens does not, by itself, invalidate SFT on complete sequences of at most 8,192 tokens. The disclosed long-scene omission needs evaluation, not an assumed accuracy failure.

Read-only SSH observations ran from **05:42:27 through 05:50:55 UTC** on 2026-10-13. Only this report was written. No model load, training, retokenization campaign, job submission/cancellation/restart, API inference, environment change, or other repository edit was performed. An independently delegated Model QA sidecar reviewed frozen source and prior evidence; it had no write authority. Existing admission, trainer, integration, data, base-payload and context reviews were read first.

## Scope and evidence identity

Remote host: `u6uv.aip2.isambard`.

- `ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill`
- `SNAP=ROOT/automation/unknownsize-20261013/snapshots/sft-c8f792a65470`
- `RUN=ROOT/runs/unknownsize-20261013`
- J: `RUN/sft-joint-001`, arm directory `joint`, job **7365681**.
- P: `RUN/sft-placement-001`, arm directory `placement`, job **7365682**.

`scontrol show job` binds both submitted commands and their `PYTHONPATH` wrapper to this snapshot and these outputs. At 05:42:27 UTC both were RUNNING, elapsed **01:11:37**, each on one GPU, 24-hour allocation, no restarts. Nodes were `nid010012` and `nid010014`. Both statuses correctly remained `UNVERIFIED`, phase `training`, `quality_claim=false`.

The actual remote snapshot's code hashes match [sft-snapshot.json](sft-snapshot.json). In particular:

| Frozen file / live evidence | SHA256 |
|---|---|
| `SNAP/fastfill/comparison/train_pair.py` | `daf0dcf0c545502947c44e39d41806bd436d25179c88eb08cd24032d094152b0` |
| `SNAP/fastfill/comparison/train_pair_helpers.py` | `6a8268dc576f34a03d79e67e876417b01b7a7e8ce76241b5299b41e50176ea76` |
| `SNAP/fastfill/comparison/contract.py` | `ef1e2cd9e449449f5441348e16d9a2bb2bd2ff8618132e6b6ed378347e64cefc` |
| `SNAP/fastfill/comparison/isambard_main.sbatch` | `1d67bca0ea019b67bb168e4226c33ca1806530f23d804d4350dedad80e1eafee` |
| J `run_receipt.json` | `9bd27d9495f1a24ec6675796bf4f02c3778df4a1dc1465d7cf1f8933bd1df216` |
| P `run_receipt.json` | `0c048b78bfd8009a25c14a8817010f6187a6321133de0aac00e10f9b30d6ff6e` |
| Both `lengths.json` | `d3dd6be2fe3af27c4ee6157fa2da04028d55b91f0015d7759b19447a0e4a950f` |

Both live receipts and length files are unchanged from [matched-sft-admission-verification.json](matched-sft-admission-verification.json). Base, data and code receipt objects are equal; configuration differences are **only `arm` and `out`**. Data manifest SHA256 is `e588e2f7c0def2c6515810816e0fd101b6dfdaa907274ce37cc9cd0d2c7c63e6`.

## Matched population and length accounting

Freshly compared both complete length ledgers, without retokenizing the dataset. Accepted UID lists match exactly, are unique and sorted, partition the input with dropped UIDs, and have **zero mismatches** against the rule “both complete arm sequences are at most 8,192 tokens.”

| Receipt measure | Train | Dev loss admission |
|---|---:|---:|
| Qualified paired input | 98,923 | 8,167 |
| Identical admitted UIDs per arm | **98,754** | **8,151** |
| Whole paired UIDs excluded | 169 (0.170840%) | 16 (0.195910%) |
| Maximum admitted complete J sequence | 8,159 | 8,053 |
| Maximum admitted complete P sequence | 8,179 | 8,118 |

Train admitted-list SHA256 is `a1ce8c3821c905138f03cf81407351a0faf8ae908e476c678bb2a9057fe6e4e2`; dev is `2bf23c10819bb092c4f13445ffc657ceb0afa60a110237e6f2c93346e92e3aa2`, using `json.dumps(uids,separators=(',',':'))` encoded as UTF-8. These hashes identify ordered lists, not newly tokenized data.

Of the 169 train exclusions, 98 are MansionWorld and 66 SAGE-10k; the other five are two InternScenes_gen, two InteriorGS and one Structured3D (raw UID prefix `structured3d_00103_room_141`). The aggregate omission is small but nonrandom. It is **not** an observed 0.17% accuracy penalty. The formal evaluation denominator remains all **8,647** frozen requests, independently of train/dev-loss admission.

Recorded admitted train supervision is J **53,157,852** versus P **34,454,444** tokens; total sequence tokens are 111,845,719 versus 112,329,829. Matched UIDs, epochs and optimizer settings do not mean equal supervised-token budgets. This is consistent with different output tasks and prevents interpreting lower P training loss as better accuracy.

## Prompt, loss-mask and prior audit

The delegated frozen-source review found the following, with paths relative to `SNAP`:

- **No direct movable-target leak in the producer projection.** `fastfill/comparison/contract.py:153–181,201–237` assigns opaque IDs using identity-derived HMAC ordering and projects movable `id/category` for J, adding supplied estimated `size` for P. Movable target position/yaw/support are assistant outputs. Fixed geometry is intentional input context. Missing P estimates fail; they do not fall back to GT sizes.
- **Assistant-only loss.** `train_pair_helpers.py:235–252` templates only the system/user turns, masks prompt labels with `-100`, and supervises the full assistant answer plus `<|im_end|>`. `train_pair.py:230–234` pads labels with `-100`. The sidecar's bounded fake-tokenizer probe confirmed this construction; actual Qwen boundaries and every live batch were not re-instrumented.
- **Drop whole paired examples, never truncate targets.** `train_pair_helpers.py:261–284` encodes both complete sequences before deciding admission. A two-UID synthetic check confirmed unilateral overlength rejection from both arms and admission exactly at the cap. The actual length receipts independently confirm the executed common-admission accounting above.
- **Earlier qualification-binding defects are fixed in this snapshot.** `train_pair_helpers.py:159–197,222–223` verifies exact assistant-target SHA, canonical system/user-request SHA, and displayed P-size SHA against qualification evidence. The sidecar's in-memory fixtures rejected a target-position change, a P-size change with the target SHA preserved, and removal of each required hash. The [earlier integration review](agency-unknownsize-main-integration-review-20261013.md)'s F1/F2 describe an older helper; they are not evidence of corrupted current training rows. Hash binding proves consistency with qualification evidence, not correctness of the estimator itself.
- **P remains the train-only estimated-size-plus-layout comparator.** The frozen lock specifies category/room-type axis log-median with category/global fallback and five-fold group OOF training estimates; dev uses the full-train prior. Existing [data](agency-unknownsize-data-review-20261013.md) and [integration](agency-unknownsize-main-integration-review-20261013.md) reviews provide producer/fold evidence. `size-priors.json` and `prior-fit-pool.json` are bound by both live receipts. The SFT snapshot contains neither `main_data.py` nor `size_prior.py`; its loader verifies bytes but does not independently reconstruct fold complements or medians. The 124,843-row eligible train prior-fit pool precedes the 98,923-row qualified decoder pool and the 98,754-row length-admitted pool. Extra eligible-train prior exposure is disclosed and is not dev/test exposure.

One **Low** hardening gap remains: the paired consumer check does not enforce the producer's complete prompt-field allowlist (`train_pair_helpers.py:107–124,141–146`). It accepted a synthetic pair with the target position inserted in both movable requests. The full main-policy loader would reject this change against an unchanged qualification ledger. A self-consistent bad producer/ledger could pass, so an explicit field allowlist is a useful future safeguard. This establishes a validation gap, not observed leakage in these jobs.

At **05:48:35 UTC**, a bounded check of the actual four train/dev JSONLs selected the next complete row at 64 evenly spaced byte offsets per file. All **256 sampled records** had system/user/assistant roles; all **4,551 sampled movable objects** had exactly the producer's expected fields, and no complete assistant target appeared verbatim in its prompt. The top-level fields were subsets of the expected context fields; `fixed`, `height` and `room_type` are optional, so their absence is not a leakage finding. J samples contained 64/64 admitted rows for each split; P contained 63/64 for each split. Samples were selected independently per file, not as matched UID pairs or a random statistical sample. This bounds the hardening concern with actual-record evidence but cannot certify every prompt or the numerical provenance of P sizes.

| Actual-file sample | SHA256 of sorted sampled UID list |
|---|---|
| J train | `38720e244f7f78cdaf7045f06b778a3109f1eaab2ef2b280b20cc864b2d088ca` |
| J dev | `27344f0cbdb1254e76b300b2110ec905ae753469c341f2bba9eb2866d8b1f329` |
| P train | `f469e07ec3e659b50bd575e1ca26f27dcaf4bc5bf51d6732d9f81c3a04e5643f` |
| P dev | `eec88f36e4aba71b6b330a4dc4b8f9a3531a1378cb5265523ac49cd5a0f537f1` |

## Actual optimizer and progress

Evidence is `SNAP/fastfill/comparison/train_pair.py:203–245`, each arm's live `training_args.json`, and checkpoint optimizer/scheduler metadata. This is **`adamw_torch`**, not an inferred optimizer default. Both argument dictionaries differ only in `output_dir`.

| Setting | Both arms |
|---|---|
| Optimizer | PyTorch AdamW; saved `fused=None`, `foreach=None`, decoupled weight decay enabled |
| Learning rate / betas / epsilon | `5e-5` / `(0.9, 0.999)` / `1e-8` |
| Scheduler / warmup | cosine / **740** optimizer steps |
| Weight decay / gradient clipping | `0.0` / `max_grad_norm=1.0` |
| Batch / accumulation / world size | 1 / 4 / 1 |
| Epochs / planned optimizer updates | 1 / **24,689** (`ceil(98754/4)`) |
| Precision / attention | BF16 / SDPA |
| LoRA | rank 32, alpha 64, dropout 0.05; q/k/v/o and gate/up/down projections |
| Random seeds | model 42, data 42 |
| Evaluation during training | disabled (`eval_strategy=no`) |
| Periodic saves | every 1,000 updates, limit 2, optimizer/scheduler/RNG included |

`max_steps=-1` in the actual TrainingArguments means the epoch budget controls training; it does not mean no training. At saved step 1,000, both schedulers have `last_epoch=1000`, `_step_count=1001`, and optimizer/scheduler LR `4.998546083093497e-5`. The logged update LR is `4.998557244488283e-5`, consistent with recording the update LR before the next scheduler value. Optimizer metadata was inspected with `zipfile` and `pickletools` only: **no unpickling, tensor loading or 699 MB optimizer payload read**.

Latest saved trainer states at 05:42:59 UTC were:

| `checkpoint-1000/trainer_state.json` measure | J | P |
|---|---:|---:|
| Saved UTC | 05:17:25 | 05:20:04 |
| Global step / planned max | 1,000 / 24,689 | 1,000 / 24,689 |
| Saved epoch fraction | 0.0405046884 | 0.0405046884 |
| Logged loss / gradient records | 1,000 / 1,000 | 1,000 / 1,000 |
| Nonfinite loss / gradient records | **0 / 0** | **0 / 0** |
| Latest logged loss | **0.4546863139** | **0.3557975590** |
| Latest logged gradient norm | **0.3555382490** | **0.3303774297** |
| Mean loss, last 20 saved logs | 0.4331512555 | 0.3687955901 |
| Loss min–max | 0.2893609703–1.4051771164 | 0.2731380761–1.0922925472 |
| Gradient norm min–max | 0.1073452011–4.1912312508 | 0.1351424456–6.1762447357 |
| Dev/evaluation loss records | 0 | 0 |

Trainer-state SHA256: J `0edf278c72e662d59f01ed91e6e01faa0ff482a8510ec55196722b55b36c42ea`; P `ecc924d95aa2c02a2bf305e2c93163614b88785aeab1a38fed0620e254fe3ace`. Their checkpoint `SHA256.json` manifests exist; all entries smaller than 1 MB were freshly verified successfully. Large adapter/optimizer/tokenizer payloads were not rehashed in this narrow audit.

Training had advanced beyond those periodic saves. At **05:43:45 UTC**, complete `loss.jsonl` records numbered J **6,691**, P **6,260**, with **zero nonfinite losses**. Last `optimizer_step_before` values were 1,672 and 1,564 respectively; these are microbatch markers, not final completion counters. Subsequent bounded stdout tails reported finite J loss/gradient **0.442 / 0.6141** at epoch 0.06805 and P **0.3582 / 0.1643** at epoch 0.06363. No traceback/OOM/nonfinite-loss marker appeared in those last 32 KiB; this is not a full historical error-log scan.

The embedded collector was then executed successfully at **05:50:55 UTC**. Latest saved checkpoints were still step 1,000 with the same hashes and finite statistics. Complete microbatch records had advanced to J **7,412** and P **6,935**, again **zero nonfinite losses**, with last `optimizer_step_before` 1,852 and 1,733. Receipt equality, all four deterministic prompt-sample digests and zero sample schema/target-copy violations reproduced. The embedded synthetic-probe command also exited zero and reproduced the reported SHA-binding, masking, whole-pair rejection and Low hardening results. Neither command loaded a model or read the full raw training corpus.

Finite losses and observed progress support continued execution. They do not establish generalization, geometry accuracy, convergence, completion within walltime, or a J/P winner. Gradient norms above one are not alone evidence that clipping failed; logged norms can be pre-clipping norms.

## Base lineage and completion boundaries

Both runs bind **`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`**, generic upstream Qwen3-8B, with the same base/tokenizer files and 40,960-position context. [base-payload-verification.json](base-payload-verification.json) records full payload verification; both live receipts also report freshly verified weights. This audit compared their identities and hashes and did not reread all 16.4 GB of base weights.

Frozen `_train` seeds and loads the base independently, then creates a new LoRA with `get_peft_model`; `trainer.train()` is called without a resume checkpoint (`train_pair.py:203–234`). `_load_model` verifies base-file hashes before every load (`:166–176`). Saved adapter configs name the generic base, rank 32/alpha 64 and fresh initialization. Their `revision=null` is not the full provenance record: the revision and content hashes are in `run_receipt.json`. Exported `inference_mode=true` is a saved adapter setting, not proof that training was disabled.

The frozen driver requires full-step finite training, saved artifact hashes, a separate fresh process/base reload, and two dev generations before completion (`train_pair.py:234–246,361–388,392–438`). **These final fresh-load gates have not yet executed for either full run**: neither arm had `training_result.json`, `result.json`, final `adapter/adapter_model.safetensors`, or `failure.json` at observation. Periodic checkpoints are evidence, not approved resume points (`resume_supported=false`). Two successful dev probes would certify reload/execution, not accuracy.

## Findings and decision conditions

| ID / severity | Observation and evidence | Impact | Recommendation |
|---|---|---|---|
| F1 — **Low**, disclosed coverage limitation | Exact length ledgers exclude 169/98,923 train UIDs and 16/8,167 dev-loss UIDs, with concentration in long-scene sources. | Long-tail training coverage is reduced; accuracy impact is unmeasured. This alone does not establish faulty training. | Finish the matched pair. Report completion/schema/geometry and EOS/cap failures by length/source, including all 16 dev exclusions; preserve all test requests in denominators. |
| F2 — **Info**, incomplete accuracy evidence | Both full runs are in progress; no saved eval-loss or final clean-load/quality result exists. | Neither numerical health nor synthetic context capacity supports an accuracy claim. | Withhold quality approval until completion, fresh-load gates and paired evaluation. Use dev evidence for changes; do not tune on test outcomes. |
| F3 — **Info**, comparator interpretation | Same examples/epochs but J/P supervised tokens are 53,157,852/34,454,444. | Training-loss differences cannot rank the two systems or establish equal supervised-token compute. | Describe the primary endpoint as matched-example, matched-epoch SFT J0/P0, with the disclosed task-specific targets and train-only P prior. |
| F4 — **Low**, prompt consumer hardening | The paired check accepts one constructed pair containing extra movable target-position fields; the unchanged qualification ledger would reject that modification. | Self-consistently produced bad prompts could bypass the paired schema check. No actual corpus contamination is established. | Add the producer's movable-field allowlist at the consumer boundary in a later code change. This does not warrant discarding current training absent evidence of contaminated records. |

**No confirmed High or Medium training defect was found that justifies immediate retraining.** A confirmed target/prompt leak, unequal UID admission, truncated supervision, incorrect starting checkpoint, unintended resume/optimizer configuration, or nonfinite optimization would change that conclusion. Failed final reload/hash/completion gates must block downstream promotion; the remedy depends on the failure, rather than assuming every reload or capacity failure requires training again. This read-only assignment does not execute cancellation or retraining.

The [long-context amendment](../long-context-amendment.json) explicitly retains this SFT pair, discloses the 169 exclusions, and allocates inference from remaining **40,960-token total context**, not 40,960 new output tokens. It records batch 8 and a shared paired input-width allowance. The existing [capacity result](context-capacity-result.json) reached 40,960 cached tokens at batch 8, peak allocated **73,998,618,112 bytes**, reserved **100,411,637,760 bytes**, on a **102,005,473,280-byte** device. That was synthetic chunked prefill; it does not certify formal unchunked prefill, natural-EOS throughput, or accuracy beyond 8k. Those remain evaluation gates, not evidence that the current optimizer run is wrong.

## Unverified and intentionally not repeated

Final adapters, completion, full-run finite gradients after the latest saved checkpoint, full payload rehashes, end-to-end resume, natural full-context generation capacity/runtime, long-tail accuracy and formal J0/P0 results remain unverified. No independent model retraining or prediction replication was performed. Earlier source/data/geometry audits and their raw-corpus work were not repeated. Historical exposed test splits, one training seed and category-median comparator limitations still apply. Classification calibration/SHAP/fairness and a broad model-lifecycle signoff are outside this narrow structured-generation execution review.

Observed training environment: Python 3.12 per prior server preflight; torch **2.13.0+cu126**, transformers **5.14.1**, PEFT **0.21.0**, accelerate **1.15.0**, tokenizers **0.22.2**, safetensors **0.8.0**, NVIDIA GH200. The read-only stdlib collector used remote Python **3.14.6**. No dependencies were installed or changed.

The parent's later report of 672 pipeline tests, 90.8% coverage and a real-render pass is separate engineering evidence; it was not independently reverified here and supplies no new model test metrics. Training is not complete. No further SSH session, control-socket change or broad corpus read is needed for this audit.

## Reproducible read-only receipt collector, version 1

This self-contained command summarizes existing raw receipts and trainer logs without importing the project/model, executing pickle objects, printing UID lists, or writing files. Live counters can advance; periodic checkpoints may rotate under `save_total_limit=2`. The hashes above identify the observations used for this report. Source-line inspection and existing audit provenance supplement this collector; it is not a model-training replication script.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 u6uv.aip2.isambard 'python3 -B -' <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, pickletools, statistics, subprocess, sys, zipfile

root = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill')
run = root / 'runs/unknownsize-20261013'
snap = root / 'automation/unknownsize-20261013/snapshots/sft-c8f792a65470'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
emit = lambda kind, obj: print(kind, json.dumps(obj, sort_keys=True))
emit('observation', {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version})
subprocess.run(['squeue', '-j', '7365681,7365682', '-o',
                '%.18i %.12T %.12M %.12l %.25R'], check=True)
emit('source_hashes', {str(p.relative_to(snap)): sha(p)
                      for p in sorted(snap.rglob('*')) if p.is_file()})
receipts, ledgers, arguments = [], [], []
for name, arm in [('sft-joint-001', 'joint'), ('sft-placement-001', 'placement')]:
    path = run / name
    receipt, ledger = read(path / 'run_receipt.json'), read(path / 'lengths.json')
    receipts.append(receipt)
    ledgers.append(ledger)
    emit(name + '/hashes', {f: sha(path / f) for f in
                           ['run_receipt.json', 'lengths.json', 'status.json']})
    emit(name + '/status', read(path / 'status.json'))
    for split, data in ledger.items():
        uids = data['accepted_uids']
        admitted = set(uids)
        dropped = {r['uid'] for r in data['dropped']}
        rows = data['lengths']
        emit(name + '/' + split, {
            'input': data['input_count'], 'admitted': len(uids),
            'unique': len(admitted), 'dropped': len(dropped), 'cap': data['max_len'],
            'uids_sha256': hashlib.sha256(json.dumps(uids, separators=(',', ':')).encode()).hexdigest(),
            'partition_ok': admitted.isdisjoint(dropped) and
                            admitted | dropped == {r['uid'] for r in rows},
            'predicate_mismatches': sum((r['uid'] in admitted) !=
                all(v <= data['max_len'] for v in r['lengths'].values()) for r in rows),
            'tokens': data['tokens']})
        source = Path(receipt['config']['data']) / arm / (split + '.jsonl')
        size, sample = source.stat().st_size, {}
        with source.open('rb') as stream:
            for k in range(64):
                offset = k * size // 64
                stream.seek(offset)
                if offset:
                    stream.readline()
                line = stream.readline()
                if line:
                    row = json.loads(line)
                    sample[row['uid']] = row
        object_keys = {'id', 'category'} | ({'size'} if arm == 'placement' else set())
        context_keys = {'room_type', 'boundary_type', 'boundary', 'height', 'fixed', 'objects'}
        violations, target_copies, objects = 0, 0, 0
        for row in sample.values():
            turns = row['messages']
            request = json.loads(turns[1]['content'])
            objects += len(request['objects'])
            violations += (any(set(o) != object_keys for o in request['objects']) or
                not set(request).issubset(context_keys) or
                [m['role'] for m in turns] != ['system', 'user', 'assistant'])
            answer = turns[-1]['content']
            target_copies += len(answer) > 32 and any(answer in m['content'] for m in turns[:-1])
        emit(name + '/' + split + '/bounded_prompt_sample', {
            'sampled_rows': len(sample), 'admitted_rows': sum(u in admitted for u in sample),
            'sample_uid_digest': hashlib.sha256(json.dumps(sorted(sample), separators=(',', ':')).encode()).hexdigest(),
            'movable_objects': objects, 'schema_violation_rows': violations,
            'verbatim_complete_target_copy_rows': target_copies})
    directory = path / arm
    args = read(directory / 'training_args.json')
    arguments.append(args)
    keys = ['optim', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'learning_rate',
            'warmup_steps', 'lr_scheduler_type', 'weight_decay', 'max_grad_norm',
            'per_device_train_batch_size', 'gradient_accumulation_steps',
            'num_train_epochs', 'max_steps', 'seed', 'data_seed', 'eval_strategy']
    emit(name + '/training_args', {k: args.get(k) for k in keys})
    checkpoints = sorted(directory.glob('checkpoint-*/trainer_state.json'),
                         key=lambda p: int(p.parent.name.rsplit('-', 1)[1]))
    if checkpoints:
        file = checkpoints[-1]
        state = read(file)
        summary = {'path': str(file), 'sha256': sha(file),
                   'global_step': state['global_step'], 'max_steps': state['max_steps'],
                   'epoch': state.get('epoch')}
        for key in ['loss', 'grad_norm', 'learning_rate', 'eval_loss']:
            values = [(r.get('step'), r[key]) for r in state.get('log_history', []) if key in r]
            finite = [(s, v) for s, v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            summary[key] = {'count': len(values), 'nonfinite': len(values) - len(finite),
                            'latest': finite[-1] if finite else None,
                            'min': min((v for _, v in finite), default=None),
                            'max': max((v for _, v in finite), default=None),
                            'mean_last20': statistics.mean(v for _, v in finite[-20:]) if finite else None}
        emit(name + '/latest_saved', summary)
        for filename in ['optimizer.pt', 'scheduler.pt']:
            with zipfile.ZipFile(file.parent / filename) as z:
                member = next(n for n in z.namelist() if n.endswith('/data.pkl'))
                ops = list(pickletools.genops(z.read(member)))
            fields = {'lr', 'betas', 'eps', 'weight_decay', 'fused', 'foreach',
                      'last_epoch', '_step_count', '_last_lr'}
            emit(name + '/' + filename + '/metadata_only', [
                [(o.name, value) for o, value, _ in ops[i:i + 8]]
                for i, (_, value, _) in enumerate(ops)
                if isinstance(value, str) and value in fields])
    data = (directory / 'loss.jsonl').read_bytes()
    logs = [json.loads(line) for line in data.split(b'\n')[:-1] if line]
    emit(name + '/microbatches', {'complete_rows': len(logs),
         'nonfinite': sum(not math.isfinite(r['loss']) for r in logs),
         'latest': logs[-1] if logs else None})
    emit(name + '/completion_files', {f: (directory / f).exists() for f in
         ['training_result.json', 'result.json', 'failure.json', 'adapter/adapter_model.safetensors']})
emit('pair_comparison', {
    'base_equal': receipts[0]['base'] == receipts[1]['base'],
    'data_equal': receipts[0]['data'] == receipts[1]['data'],
    'code_equal': receipts[0]['code_sha256'] == receipts[1]['code_sha256'],
    'lengths_equal': ledgers[0] == ledgers[1],
    'config_differences': {k: [receipts[0]['config'].get(k), receipts[1]['config'].get(k)]
        for k in receipts[0]['config'].keys() | receipts[1]['config'].keys()
        if receipts[0]['config'].get(k) != receipts[1]['config'].get(k)},
    'training_args_differences': {k: [arguments[0].get(k), arguments[1].get(k)]
        for k in arguments[0].keys() | arguments[1].keys()
        if arguments[0].get(k) != arguments[1].get(k)}})
PY
```

## Reproducible bounded synthetic probes, version 1

The sidecar's fixture tests can be reproduced with this condensed equivalent. It invokes only the SHA-pinned, standard-library helper in memory. It isolates `_check_pair` from the stronger main-ledger gate and uses a fake tokenizer; it is not a production-data corruption demonstration or a Transformers collator test.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 u6uv.aip2.isambard 'python3 -B -' <<'PY'
from pathlib import Path
import copy, hashlib, json, runpy
p = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/automation/'
         'unknownsize-20261013/snapshots/sft-c8f792a65470/fastfill/comparison/train_pair_helpers.py')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '6a8268dc576f34a03d79e67e876417b01b7a7e8ce76241b5299b41e50176ea76'
h = runpy.run_path(str(p))
def record(arm):
    obj = {'id': 'obj_0001', 'category': 'chair'}
    target = {'id': 'obj_0001', 'pos': [2, 3, 0], 'yaw': 0}
    obj = {**obj, **({'size': [1, 1, 1]} if arm == 'placement' else {})}
    target = {**target, **({'size': [1, 1, 1]} if arm == 'joint' else {})}
    request = {'boundary_type': 'polygon', 'boundary': [[0,0],[10,0],[10,10],[0,10]], 'objects': [obj]}
    return {'uid': 'fixture', 'source': 'synthetic', 'group': 'synthetic-group', 'split': 'train',
            'target_policy': h['MAIN_TARGET_POLICY'], 'target_qualified': True,
            'messages': [{'role': 'system', 'content': 'Synthetic audit fixture'},
                         {'role': 'user', 'content': json.dumps(request)},
                         {'role': 'assistant', 'content': json.dumps({'placements': [target]})}]}
pairs = {'train': {a: [record(a)] for a in h['ARMS']}}
j, p = (pairs['train'][a][0] for a in h['ARMS'])
h['_check_pair'](j, p, 'train', h['MAIN_TARGET_POLICY'])
injected = copy.deepcopy([j, p])
for row in injected:
    request = json.loads(row['messages'][1]['content'])
    request['objects'] = [{**o, 'pos': [2, 3, 0]} for o in request['objects']]
    row['messages'][1]['content'] = json.dumps(request)
h['_check_pair'](*injected, 'train', h['MAIN_TARGET_POLICY'])
print('LOW: shared target-position injection accepted by paired check only')
counts = {'valid': True, **{k: 0 for k in ['oob','support_fail','ceiling','fixed_blocking','main','severe']}}
ledger = [{'uid': 'fixture', 'accepted': True, 'eligible': True,
           **{a: {'qualified': True, 'reasons': [], 'counts': counts,
                  **h['qualification_hashes'](pairs['train'][a][0]['messages'], a)} for a in h['ARMS']}}]
def check(candidate, expected=None):
    try:
        h['_main_admission'](None, {'common-admission.jsonl': 'fixture'}, candidate)
    except ValueError as exc:
        assert expected and f'qualified {expected} mismatch' in str(exc), str(exc)
    else:
        assert expected is None, expected
namespace = h['_main_admission'].__globals__
original = namespace['_rows']
namespace['_rows'] = lambda *args: copy.deepcopy(ledger)
try:
    check(pairs)
    for index, collection, replacement, expected in [
        (-1, 'placements', {'pos': [500,500,0]}, 'target_sha256'),
        (1, 'objects', {'size': [100,100,100]}, 'request_sha256')]:
        changed = copy.deepcopy(pairs)
        turns = changed['train']['placement'][0]['messages']
        content = json.loads(turns[index]['content'])
        content[collection] = [{**o, **replacement} for o in content[collection]]
        turns[index]['content'] = json.dumps(content)
        check(changed, expected)
    for field in ['target_sha256', 'request_sha256', 'estimated_sizes_sha256']:
        saved = ledger[0]['placement'].pop(field)
        check(pairs, field)
        ledger[0]['placement'][field] = saved
finally:
    namespace['_rows'] = original
print('PASS: clean qualification; stale target/request and missing hashes rejected')
class FakeTokenizer:
    def get_vocab(self): return {'<|im_end|>': 99}
    def apply_chat_template(self, messages, **kwargs):
        assert [m['role'] for m in messages] == ['system', 'user']
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return 'PROMPT'
    def __call__(self, strings, **kwargs):
        assert kwargs == {'add_special_tokens': False}
        assert len(strings) == 2 and strings[1].endswith('<|im_end|>')
        return {'input_ids': [[1, 2], [7] * (3 if 'LONG' in strings[1] else 1) + [99]]}
tok = FakeTokenizer()
assert h['encode_messages'](tok, j['messages']) == {
    'input_ids': [1,2,7,99], 'attention_mask': [1,1,1,1], 'labels': [-100,-100,7,99]}
def length_row(uid, answer):
    return {'uid': uid, 'messages': [{'role': 'system', 'content': 'S'},
        {'role': 'user', 'content': 'U'}, {'role': 'assistant', 'content': answer}]}
arms = {'joint': [length_row('short','SHORT'), length_row('long','LONG')],
        'placement': [length_row('short','SHORT'), length_row('long','SHORT')]}
samples, report = h['common_admission'](tok, arms, 4)
assert report['accepted_uids'] == ['short'] and len(report['dropped']) == 1
assert report['dropped'][0]['overlength_arms'] == ['joint']
assert report['dropped'][0]['lengths'] == {'joint': 6, 'placement': 4}
assert len(samples['joint']) == len(samples['placement']) == 1
print('PASS: prompt masking, assistant EOS, exact-cap admission and whole-pair rejection')
PY
```
