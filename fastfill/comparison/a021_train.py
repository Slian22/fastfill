"""Amendment 021 section 2: Arm A, the 27B student (Qwen3.8-27B = the O2 weights as amendment 007 ran them, thinking
off, LoRA on the language model only) trained by train_dpo --sft_only on the P-only admitted pool; Arm B (the 8B on the
same pool and code path) only as the fallback.

python -m fastfill.comparison.a021_train STEP ...   (sbatch: isambard_021_train.sbatch; submit: submit_021_train.sh)

pool     --data MAIN --corpus RUN/full-corpus-v2 --lineage LINEAGE_JSON --base27 QWEN27 --base8 QWEN3
         --dev DEV... --test TEST... --out RUN/a021/pool [--workers N] [--sample 200]
    Every eligible train room whose P target passes the frozen checker. Candidates are the main-data-v1
    common-admission ledger's placement-qualified rooms: the 98,923 accepted rooms (P's SFT rows, verbatim) plus the
    rooms dropped only because J's GT failed (placement qualified, joint not). The ledger's admission counts must equal
    the lineage audit's (evidence/full-audit-data-lineage-7368003-result.json summary.admission_counts) and main-data's
    dataset_counts, and P-only = placement_qualified - common_decoder_train (19,135). P-only rows are rebuilt with
    main_data._candidate (frozen corpus, size priors, fold and repair limits of main-data-v1); each rebuilt ledger entry
    must equal the stored one (so its request / target / size hashes too). The same rebuild on the --sample
    lowest-rank accepted rooms must reproduce their placement/train.jsonl lines byte for byte and their train rooms.
    Pool = candidates whose P target geometry.score finds complete and geometry_success (the frozen checker), in
    sha256('a021-27b:' + uid) order. A UID or house group shared with any dev/test eval dir raises. Per room and
    tokenizer (27B, 8B): train_dpo's render (prompt_text + target + <|im_end|>) token counts, prefix agreement and fit
    (<= 8192 tokens). Writes sft-rows.jsonl (P's SFT format: prompt = system/user turns, chosen = rejected = target),
    ledger.jsonl, summary.json (counts, P-only breakdown by source / bin / family), MANIFEST.json.
kernels  [--report]   GPU: transformers' qwen3_5 must bind fla's chunk / recurrent gated-delta-rule kernels and
    FusedRMSNormGated (causal_conv1d is not installed: F.conv1d); a tiny GatedDeltaNet layer runs forward and backward
    on both fla and the torch fallback. --report only prints the path taken (the CPU setup job).
smoke    --pool RUN/a021/pool --base QWEN27 --out RUN/a021/smoke
    torchrun train_dpo (Arm A recipe, 4 ranks, bs 1, grad_accum 1, gradient checkpointing) on the WORLD x SMOKE_STEPS
    longest admitted sequences: 200 optimizer steps, per-rank peak memory and step times (--perf_out). verdict.json
    (write-once): SMOKE_STOP if the peak reserved memory of any rank > 90 GB (10^9 bytes, the amendment's unit), or
    the projected wall time for min(40,000, pool) rooms > 12 h; else SMOKE_PASS with N = min(40,000, the most rooms in
    rank order that the measured throughput fits in 10 h) and save_steps = optimizer steps per ~30 min. Projection: per
    optimizer step the longest of its 4 sequences (DistributedSampler, seed 42, epoch 0, as training) / the smoke's
    per-rank tokens per second at each steady step's (after WARMUP) longest sequence. Only those two rules stop Arm A: a
    crashed run or incomplete perf records write no verdict (the attempt is renamed smoke.failed-JOB and the error
    raised), so the smoke can be rerun. The pool's 27B EOS must be <|im_end|> before anything is written.
smoke-vllm --data RUN/dev-select-v1 --smoke RUN/a021/smoke --base QWEN27   (vLLM env)
    After SMOKE_PASS: sample_vllm's engine (+ MULTIMODAL_ENGINE, the probe's) on the pinned 27B loads the smoke's
    adapter (smoke/train/final: every LM linear incl. gated-delta-net in_proj_qkv/in_proj_z/out_proj) as a LoRA and
    decodes the first VLLM_SMOKE_ROOMS probe rooms greedily. Writes smoke/vllm-load.json (status OK) only on success;
    train (Arm A) refuses without it, so an adapter vLLM cannot serve stops Arm A before its training, not at the probe.
Amendment 028 (arm J27, fastfill.comparison.a028): smoke / smoke-vllm --arm J27 and train --arm J27 --pool RUN/a028/rows
    --run-root RUN/a028 --smoke RUN/a028/smoke run Arm A's code, model, recipe and gates on J's own SFT rows (rows file
    joint.jsonl, smoke-vllm on the joint requests); the policy RUN/a028/27b-joint is served as --arm joint.
train    --arm A|B --pool RUN/a021/pool --base BASE --run-root RUN/a021 [--smoke RUN/a021/smoke]   (4 GPUs; resumes)
    A: the first N admitted rooms (smoke verdict + vllm-load.json OK) that fit the 27B; B (only while A has no policy): every admitted room
    that fits the 8B, on the generic Qwen3-8B. run_receipt.json (N, rows, recipe, base, smoke verdict) is written
    before training. training_guard, dpo_stages.preflight, torchrun train_dpo (RECIPE: LoRA r32/a64, lr 5e-5, 1 epoch,
    effective batch 4, P's 3% cosine warmup, token-mean NLL, LoRA regex, answer logits, render check, base receipt; A also
    --image_text_model), completion, then the policy receipt a021-policy.json: dpo_stages.inference_identity /
    sample_vllm take RUN/a021/27b|8b as --model with --exact-policy (FP32 adapter unmerged on the pinned base).
probe-hf | probe-vllm | probe-report   --data RUN/dev-select-v1 --policy RUN/a021/27b --base QWEN27 --out RUN/a021/probe
    The 100 dev-select rooms of lowest sha256('a021-probe:' + uid): greedy HF exact (FP32 adapter unmerged on the BF16
    base, the training class; every adapter tensor must load) and vLLM LoRA (sample_vllm's engine, TP 1, max_lora_rank
    32, image/video 0; k = 0 params). Cap = min(40960 - prompt, 1024 + 128 x objects). report: frozen checker success
    (geometry.score, cap hit without EOS = fail) per engine, vLLM - HF with a house-cluster CI (report_v2.contrasts),
    identical-output share; report.json, report.md (Chinese). Every output is new-only; probe-hf appends each room to
    hf.partial.jsonl (bound to the policy identity), so a rerun after a time limit decodes only the remaining rooms.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from fastfill.comparison import dpo_stages as d
from fastfill.comparison import infer as i
from fastfill.comparison import train_pair_helpers as h

SALT, PROBE_SALT = 'a021-27b:', 'a021-probe:'
EOS = '<|im_end|>'
MAX_LEN = d.MAX_LEN  # 8192, P's SFT --max-len.
WORLD, SEED = 4, 42
N_MAX, FIT_HOURS, STOP_HOURS = 40_000, 10, 12
STOP_MEMORY = 90 * 2 ** 30  # Peak reserved CUDA memory of any rank: 90 GiB (amendment 021 addendum A).
SMOKE_STEPS, WARMUP, CKPT_SECONDS = 200, 10, 1800
PROBE_ROOMS, MAX_MODEL_LEN, VLLM_SMOKE_ROOMS = 100, 40960, 3
B_SAVE_STEPS = 1000
LINEAGE_SHA256 = '3c65c23d639d19f26081a7f0aa35eb9307ab51f88ca817a58dfc7fffc67e7f53'
P_ONLY = 19_135  # placement_qualified 118,058 - common_decoder_train 98,923 (lineage audit, critique section 2.3).
POLICY, POOL_STATUS, VLLM_LOAD = 'a021-policy.json', 'A021_POOL_COMPLETE', 'vllm-load.json'
SMOKE_PASS, SMOKE_STOP = 'SMOKE_PASS', 'SMOKE_STOP'
MODEL_TYPE, IMAGE_TEXT, CAUSAL = 'qwen3_5', 'image_text_to_text', 'causal_lm'
REPOSITORY_27B = 'Qwen3.8-27B (models/Qwen3.8-27B, the O2 weights)'
REVISION_27B = 'amendment-007 O2 run: model_files_sha256 of llm-eval-O2 (every non-weight file); weights by SHA256SUMS'
BASE_27B = {  # Every non-weight file of the O2 model dir as the amendment-007 runs recorded it (llm_open config).
    '.gitattributes': '34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930',
    'LICENSE': 'bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a',
    'README.md': '57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641',
    'SHA256SUMS': 'a0f529ae786e405c97732cbfe24269494fb421e64ad55f7f5ca8bbd263578dfc',
    'chat_template.jinja': 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041',
    'config.json': '191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab',
    'crc32.txt': 'b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83',
    'generation_config.json': 'e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e',
    'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
    'model.safetensors.index.json': '77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df',
    'preprocessor_config.json': '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
    'tokenizer.json': '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3',
    'tokenizer_config.json': 'b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27',
    'video_preprocessor_config.json': '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13',
    'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
}
# LoRA targets: the language model's linear layers (full attention q/k/v/o, gated-delta-net qkv/z/out projections,
# MLP) in the checkpoint's own module names; vision tower, MTP head, tiny b/a projections and lm_head excluded.
LORA_27B = (r'model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)'
            r'|mlp\.(gate|up|down)_proj)')
LORA_8B = r'model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'  # P's seven, LM layers only.
ARMS = {'A': {'dir': '27b', 'tokens': '27b', 'regex': LORA_27B, 'model_class': IMAGE_TEXT, 'model_type': MODEL_TYPE},
        'B': {'dir': '8b', 'tokens': '8b', 'regex': LORA_8B, 'model_class': CAUSAL, 'model_type': 'qwen3'},
        # Amendment 028: Arm A's base, recipe and gates on J's own SFT rows (a028 rows, an a021 pool layout), under
        # --run-root RUN/a028; a joint policy (inference arm 'joint').
        'J27': {'dir': '27b-joint', 'tokens': '27b', 'regex': LORA_27B, 'model_class': IMAGE_TEXT, 'model_type': MODEL_TYPE,
                'rows': 'joint.jsonl', 'infer_arm': 'joint', 'stage': 'a028-27b-joint-sft'}}
RECIPE = {'lr': 5e-5, 'epochs': 1, 'bs': 1, 'grad_accum': 1, 'max_len': MAX_LEN, 'lora_r': 32, 'lora_alpha': 64,
          'seed': SEED, 'warmup_ratio': 0.03}  # P's SFT: LoRA r32/a64, lr 5e-5, 1 epoch, effective batch 4 = 4 x 1 x 1,
# cosine with max(1, int(0.03 x steps)) warmup steps (train_pair).
SWITCHES = ('sft_only', 'sft_token_mean', 'answer_logits', 'check_render', 'base_receipt')
FIXED = {'world_size': WORLD, 'effective_batch': WORLD, 'launcher': 'torchrun --standalone (DDP)', 'lora_dropout': 0.05,
         'dtype': 'bfloat16', 'gradient_checkpointing': True, 'scheduler': "cosine, P's 3% warmup",
         'loss': ('token-mean NLL: answer-token NLL summed over the optimizer window on all ranks / its supervised-token '
                  'count (all-reduced); no DPO term, no reference pass'),
         'deviation_from_p': ('no autocast: BF16 base weights, FP32 LoRA (PEFT) and FP32 log-softmax; P ran under HF '
                              'Trainer bf16 autocast (same BF16 base, FP32 LoRA)')}
RULES = {'stop_memory_bytes': STOP_MEMORY, 'stop_projected_hours': STOP_HOURS, 'n_max': N_MAX, 'fit_hours': FIT_HOURS,
         'warmup_steps': WARMUP, 'smoke_steps': SMOKE_STEPS, 'checkpoint_seconds': CKPT_SECONDS,
         'projection': ("hours(n) = sum over optimizer steps of the longest of its WORLD sequences (DistributedSampler "
                        "seed 42 epoch 0 over the first n admitted rooms) / steady per-rank tokens per second / 3600")}
CODE = ('comparison/a021_train.py', 'comparison/dpo_stages.py', 'comparison/infer.py', 'comparison/main_data.py',
        'comparison/geometry.py', 'comparison/train_pair_helpers.py', 'v1_1/train_dpo.py', 'scene.py', 'validate.py')


def _code():
    package = Path(__file__).resolve().parents[1]
    return {name: h.sha256(package / name) for name in CODE}


def _rows_name(arm):
    return ARMS[arm].get('rows', 'placement.jsonl')


def _infer_arm(arm):
    return ARMS.get(arm, {}).get('infer_arm', 'placement')


def rank(uid, salt=SALT):
    return hashlib.sha256((salt + uid).encode('utf-8')).hexdigest()


def _once(path, payload):
    """Write-once bytes (a hard link, never a replacement)."""
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


# ---- the pinned bases ----------------------------------------------------------------------------------------------

def _sums(path):
    sums = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip('*').removeprefix('./')] = digest
    return sums


def base_receipt(base):
    """The pinned base, weights rehashed: Qwen3.8-27B exactly as O2 ran it (every non-weight file = BASE_27B, no other
    file; weight shards = the index's, by the pinned SHA256SUMS), else the generic Qwen3-8B (h.base_receipt)."""
    base = Path(base)
    config = h.strict_json((base / 'config.json').read_text())
    if config.get('model_type') != MODEL_TYPE:
        return h.base_receipt(base)
    if (base / 'adapter_config.json').exists():
        raise ValueError('base must be the model directory, not an adapter')
    names = {p.name for p in base.iterdir() if p.is_file()}
    weights = {n for n in names if n.endswith('.safetensors')}
    if names - weights != set(BASE_27B):
        raise ValueError(f'27B non-weight files differ from the amendment-007 O2 directory: {sorted(names - weights ^ set(BASE_27B))}')
    h.verify_artifacts(base, BASE_27B)
    sums = _sums(base / 'SHA256SUMS')
    shards = set(h.strict_json((base / 'model.safetensors.index.json').read_text())['weight_map'].values())
    if weights != shards or not shards <= sums.keys() or any(sums[n] != BASE_27B[n] for n in sums.keys() & BASE_27B.keys()):
        raise ValueError('27B weight shards differ from the index or the pinned SHA256SUMS')
    h.verify_artifacts(base, {n: sums[n] for n in sorted(shards)})
    return {'repository': REPOSITORY_27B, 'revision': REVISION_27B, 'path': str(base.resolve()), 'model_type': MODEL_TYPE,
            'files': dict(sorted({**BASE_27B, **{n: sums[n] for n in shards}}.items())),
            'weight_hashes_freshly_verified': True,
            'max_position_embeddings': config['text_config']['max_position_embeddings']}


def _model_type(base):
    return base.get('model_type', 'qwen3')


# ---- the policy receipt (dpo_stages.inference_identity, sample_vllm) -------------------------------------------------

def is_policy(model):
    return (Path(model) / POLICY).is_file()


def inference_identity(run):
    """dpo_stages.inference_identity for RUN/a021/27b|8b (--exact-policy): the FP32 adapter unmerged on the pinned base."""
    if not getattr(run, 'exact_policy', False):
        raise ValueError('an amendment-021 policy is evaluated as its exact policy (--exact-policy)')
    root = Path(run.model).resolve()
    receipt = h.strict_json((root / POLICY).read_text())
    if run.arm != _infer_arm(receipt.get('arm')):  # Arms A/B: placement; J27 (amendment 028): joint.
        raise ValueError(f'an amendment-021 policy is a {_infer_arm(receipt.get("arm"))} policy, not --arm {run.arm}')
    base = base_receipt(run.base)
    if (base['path'], base['files']) != (receipt['base']['path'], receipt['base']['files']):
        raise ValueError('base differs from the amendment-021 policy base')
    adapter = Path(receipt['adapter']['path'])
    if adapter != root / 'train' / 'final' or h.artifact_manifest(adapter) != receipt['adapter']['files']:
        raise ValueError(f'policy adapter bytes differ from {root / POLICY}')
    digest = i._hash(base['files'])
    return {'base': base, 'base_sha256': digest, 'training_data': receipt['training_data'], 'stage': receipt['stage'],
            'policy': {'mode': 'unmerged_fp32_adapter_on_bf16_training_base', 'training_base': base['path'],
                       'training_base_sha256': digest, 'adapter': str(adapter),
                       'adapter_sha256': i._hash(receipt['adapter']['files']), 'model_class': receipt['model_class']},
            'lineage_sha256': h.sha256(root / POLICY)}


# ---- pool ----------------------------------------------------------------------------------------------------------

def admission_counts(ledger):
    rows = list(ledger)
    return {'eligible_candidates': sum(r['eligible'] for r in rows),
            'joint_qualified': sum(r['joint']['qualified'] for r in rows),
            'placement_qualified': sum(r['placement']['qualified'] for r in rows),
            'common_decoder_train': sum(r['accepted'] for r in rows)}


def check_counts(counts, lineage, dataset):
    """The ledger's admission counts are the lineage audit's and main-data's; P-only follows from them."""
    for name, value in counts.items():
        if lineage.get(name) != value or dataset.get(name) != value:
            raise ValueError(f'admission count {name}: ledger {value}, lineage {lineage.get(name)}, main-data {dataset.get(name)}')
    return counts['placement_qualified'] - counts['common_decoder_train']


def _regen(task):
    """main_data._candidate on one prepared room (as its build): bound room, ledger entry and P's SFT row, JSON-normal."""
    from fastfill.comparison import main_data as md
    room, entry = task[0], task[1]
    bound, _, _, turns, ledger = md._candidate(task)
    record = md._record(bound, entry, turns, True) if turns is not None else None
    return json.loads(json.dumps({'uid': room['uid'], 'bound': bound, 'ledger': ledger, 'record': record,
                                  'line': md._dump(record) if record else None}))


def regenerate(corpus, main, manifest, uids, workers):
    from fastfill.comparison import main_data as md
    from fastfill.comparison.corpus import _load_manifest
    from fastfill.comparison.size_prior import predict_sizes
    root = Path(corpus)
    source, digest = _load_manifest(root / 'MANIFEST.json')
    if digest != manifest['corpus_manifest_sha256']:
        raise ValueError('corpus is not the one main-data-v1 was built from')
    identities = md._identities(root, source)
    oof = h.strict_json((Path(main) / 'size-priors.json').read_text())  # Hash-verified by h.load_pair.
    limits = manifest['repair_limits']
    tasks = []
    for room in md._rooms(root, source, identities):
        if room['uid'] in uids:
            fold = oof['fold_by_uid'][room['uid']]
            sizes = {k: [max(.01, round(v, 2)) for v in values]
                     for k, values in predict_sizes(room, oof['priors'][str(fold)]).items()}
            tasks.append((room, identities[room['uid']], sizes, fold, limits))
    found = {r['uid']: r for r in d._map(_regen, tasks, workers)}
    if found.keys() != set(uids):
        raise ValueError('rooms missing from the frozen corpus')
    return found


def _check(task):
    from fastfill.comparison import geometry as g
    room, prompt, chosen = task
    row = g.score(room, prompt, chosen, 'placement')
    return {'complete': row['complete'], 'geometry_success': row['geometry_success']}


def lengths(tok, rows, chunk=2000):
    """train_dpo's render per row: prompt_text, + target + <|im_end|>; token counts, prefix agreement, fit."""
    from fastfill.scene import prompt_text
    result = []
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        prompts = [prompt_text(tok, r['prompt']) for r in part]
        p_ids = tok(prompts, add_special_tokens=False)['input_ids']
        full = tok([p + r['chosen'] + EOS for p, r in zip(prompts, part)], add_special_tokens=False)['input_ids']
        for p, f in zip(p_ids, full):
            ok = f[:len(p)] == p and len(f) > len(p)
            result.append({'prompt': len(p), 'total': len(f), 'prefix_ok': ok, 'fits': ok and len(f) <= MAX_LEN})
    return result


def _counter(values):
    return dict(sorted(Counter(values).items()))


def pool(args):
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    from fastfill.comparison import best_of_n as bo
    from fastfill.comparison import onpolicy_dpo as op
    from fastfill.comparison import room_families as rf
    main = Path(args.data)
    pairs, data = h.load_pair(main)
    if data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('the pool needs the qualified main data')
    manifest = h.strict_json((main / 'MANIFEST.json').read_text())
    ledger = h._by_uid(h._rows(main, 'common-admission.jsonl', data['files']['common-admission.jsonl']))
    counts = admission_counts(ledger.values())
    if h.sha256(args.lineage) != LINEAGE_SHA256:
        raise ValueError(f'{args.lineage} is not the lineage audit result of job 7368003')
    evidence = h.strict_json(Path(args.lineage).read_text())
    expected = check_counts(counts, evidence['summary']['admission_counts'], manifest['dataset_counts'])
    if expected != P_ONLY:
        raise ValueError(f'P-only rooms {expected}, not the audited {P_ONLY}')
    p_only = sorted(u for u, r in ledger.items() if r['placement']['qualified'] and not r['joint']['qualified'])
    accepted = h._by_uid(pairs['train']['placement'])
    if len(p_only) != expected or set(accepted) != {u for u, r in ledger.items() if r['accepted']}:
        raise ValueError('P-only or accepted rooms differ from the admission counts')
    rooms = h._by_uid(h._rows(main, 'train_rooms.jsonl', data['files']['train_rooms.jsonl']))
    sample = sorted(accepted, key=lambda u: (rank(u), u))[:args.sample]
    rebuilt = regenerate(args.corpus, main, manifest, set(p_only) | set(sample), args.workers)
    wanted, lines = set(sample), {}
    with (main / 'placement/train.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            uid = json.loads(line)['uid']
            if uid in wanted:
                lines[uid] = line.rstrip('\n')
    for uid in sample:  # The rebuild reproduces P's SFT rows byte for byte.
        got = rebuilt[uid]
        if got['line'] != lines[uid] or got['ledger'] != ledger[uid] or got['bound'] != rooms[uid]:
            raise ValueError(f'rebuild differs from main-data-v1 for an accepted room: {uid}')
    for uid in p_only:
        if rebuilt[uid]['ledger'] != ledger[uid] or rebuilt[uid]['record'] is None:
            raise ValueError(f'rebuilt P-only ledger entry differs from main-data-v1: {uid}')
    entries = [{'uid': u, 'source': r['source'], 'group': r['group'], 'origin': 'common', 'messages': r['messages'],
                'room': rooms[u]} for u, r in accepted.items()]
    entries += [{**{k: rebuilt[u]['record'][k] for k in ('uid', 'source', 'group', 'messages')}, 'origin': 'p_only',
                 'room': rebuilt[u]['bound']} for u in p_only]
    entries.sort(key=lambda e: (rank(e['uid']), e['uid']))
    held_uids, held_groups, held = op._held_out(args.dev + args.test)
    leaks = [e['uid'] for e in entries if e['uid'] in held_uids or e['group'] in held_groups]
    if leaks:
        raise ValueError(f'{len(leaks)} pool rooms share a UID or house group with dev/test: {leaks[:3]}')
    rows = [{'uid': e['uid'], 'source': e['source'], 'group': e['group'], 'prompt': e['messages'][:2],
             'chosen': e['messages'][2]['content'], 'rejected': e['messages'][2]['content']} for e in entries]
    verdicts = d._map(_check, [(e['room'], r['prompt'], r['chosen']) for e, r in zip(entries, rows)], args.workers)
    bases = {'27b': base_receipt(args.base27), '8b': base_receipt(args.base8)}
    if [_model_type(b) for b in bases.values()] != [MODEL_TYPE, 'qwen3']:
        raise ValueError('--base27 / --base8 are not the 27B and the generic 8B')
    toks = {k: d._model_tokenizer(b['path']) for k, b in bases.items()}
    tokens = {k: lengths(tok, rows) for k, tok in toks.items()}
    ledger_rows, sft_rows = [], []
    for k, (e, row, verdict) in enumerate(zip(entries, rows, verdicts)):
        admitted = verdict['complete'] and verdict['geometry_success']
        ledger_rows.append({'uid': e['uid'], 'source': e['source'], 'group': e['group'], 'origin': e['origin'],
                            'objects': len(e['room']['objects']), 'bin': bo.bin_name(len(e['room']['objects'])),
                            'family': rf.family(e['room'].get('room_type')), 'checker': verdict, 'pool': admitted,
                            'tokens': {key: tokens[key][k] for key in tokens}})
        if admitted:
            sft_rows.append(row)
    admitted = [e for e in ledger_rows if e['pool']]
    only = [e for e in ledger_rows if e['origin'] == 'p_only']
    summary = {
        'status': POOL_STATUS, 'order': f"sha256('{SALT}' + uid)", 'max_len': MAX_LEN,
        'admission_counts': counts, 'p_only_expected': expected,
        'lineage': {'path': str(Path(args.lineage).resolve()), 'sha256': LINEAGE_SHA256,
                    'status': evidence.get('status'), 'note': 'admission_counts were computed before that audit job errored'},
        'p_only': {'rooms': len(only), 'by_source': _counter(e['source'] for e in only),
                   'by_bin': _counter(e['bin'] for e in only), 'by_family': _counter(e['family'] for e in only),
                   'checker_pass': sum(e['pool'] for e in only)},
        'rebuild': {'p_only_ledger_equal': len(p_only), 'accepted_sample_byte_equal': len(sample),
                    'rule': 'main_data._candidate; ledger entry equal; accepted sample: placement/train.jsonl line and '
                            'train room equal'},
        'checker': {'candidates': len(entries), 'pass': len(admitted),
                    'fail_by_origin': _counter(e['origin'] for e in ledger_rows if not e['pool'])},
        'pool': {'rooms': len(admitted), 'by_origin': _counter(e['origin'] for e in admitted),
                 'by_source': _counter(e['source'] for e in admitted), 'by_bin': _counter(e['bin'] for e in admitted),
                 'by_family': _counter(e['family'] for e in admitted),
                 'fits': {k: sum(e['tokens'][k]['fits'] for e in admitted) for k in tokens},
                 'prefix_failures': {k: sum(not e['tokens'][k]['prefix_ok'] for e in admitted) for k in tokens}},
        'held_out': held, 'leaks': 0,
        'tokenizers': {k: {'base_path': b['path'], 'base_files_sha256': i._hash(b['files']),
                           'eos_token': toks[k].eos_token} for k, b in bases.items()},
        'inputs': {'main_manifest_sha256': data['manifest_sha256'],
                   'corpus_manifest_sha256': manifest['corpus_manifest_sha256']},
        'code_sha256': _code()}
    d._publish_dir(out, {'sft-rows.jsonl': d._jsonl(sft_rows), 'ledger.jsonl': d._jsonl(ledger_rows),
                         'summary.json': (json.dumps(summary, ensure_ascii=False, indent=1) + '\n').encode('utf-8')})
    return {'status': POOL_STATUS, 'pool': len(admitted), 'p_only': len(only), 'fits': summary['pool']['fits']}


def load_pool(root):
    root = Path(root)
    files = d._verified_dir(root)
    summary = h.strict_json((root / 'summary.json').read_text())
    if summary.get('status') != POOL_STATUS:
        raise ValueError(f'{root} is not a complete amendment-021 pool')
    ledger = h._rows(root, 'ledger.jsonl', files['ledger.jsonl'])
    return {'root': root, 'summary': summary, 'ledger': ledger, 'files': files,
            'manifest_sha256': h.sha256(root / 'MANIFEST.json')}


def take(pool_value, key, n=None):
    """The first n admitted rooms (rank order) that fit train_dpo under tokenizer key."""
    return [e['uid'] for e in pool_value['ledger'] if e['pool'] and e['tokens'][key]['fits']][:n]


def pool_rows(pool_value, uids):
    rows = h._by_uid(h._rows(pool_value['root'], 'sft-rows.jsonl', pool_value['files']['sft-rows.jsonl']))
    return [rows[u] for u in uids]


# ---- smoke ---------------------------------------------------------------------------------------------------------

def step_tokens(tokens, world=WORLD, seed=SEED):
    """Per optimizer step (bs 1, grad_accum 1) the longest of its world sequences, in train_dpo's epoch-0 order."""
    from torch.utils.data.distributed import DistributedSampler
    ranks = [list(DistributedSampler(range(len(tokens)), num_replicas=world, rank=r, shuffle=True, seed=seed))
             for r in range(world)]
    return [max(tokens[k] for k in step) for step in zip(*ranks)]


def projected_hours(tokens, rate):
    return sum(step_tokens(tokens)) / rate / 3600 if tokens else 0.0


def fit_n(tokens, rate):
    """The most rooms (<= N_MAX, rank order) whose projected wall time is <= FIT_HOURS (bisection)."""
    lo, hi = 0, min(N_MAX, len(tokens))
    if projected_hours(tokens[:hi], rate) <= FIT_HOURS:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if projected_hours(tokens[:mid], rate) <= FIT_HOURS else (lo, mid)
    return lo


def smoke_verdict(perf, tokens, arm="A"):
    """Arm A keeps both stop rules; J27 reduces N on time but still stops on memory or an empty training window.
    Tokens are admitted totals in rank order. Incomplete perf records are never a verdict."""
    perf = sorted(perf, key=lambda p: p['rank'])
    counts = {len(p['steps']) for p in perf}
    if [p['rank'] for p in perf] != list(range(WORLD)) or len(counts) != 1 or counts.pop() <= WARMUP:
        raise RuntimeError(f'incomplete smoke perf records (ranks {[p["rank"] for p in perf]})')
    steps = list(zip(*(p['steps'] for p in perf)))[WARMUP:]
    seconds = sum(max(s['seconds'] for s in step) for step in steps)
    rate = sum(max(s['input_tokens'] for s in step) for step in steps) / seconds
    peak = max(p['max_memory_reserved'] for p in perf)
    full = projected_hours(tokens[:N_MAX], rate)
    n = fit_n(tokens, rate)
    reasons = ((['peak_memory'] if peak > STOP_MEMORY else [])
               + (['projected_hours'] if arm != 'J27' and full > STOP_HOURS else [])
               + (['no_fitting_rooms'] if arm == 'J27' and n == 0 else []))
    rules = ({**RULES, 'stop_projected_hours': None, 'reduce_n_above_projected_hours': STOP_HOURS}
             if arm == 'J27' else RULES)
    per_step = seconds / len(steps)
    return {'status': SMOKE_STOP if reasons else SMOKE_PASS, 'reasons': reasons,
            'peak_memory_reserved': {str(p['rank']): p['max_memory_reserved'] for p in perf},
            'peak_memory_allocated': {str(p['rank']): p['max_memory_allocated'] for p in perf},
            'device_total_memory': perf[0]['device_total_memory'], 'device_name': perf[0].get('device_name'),
            'steady_steps': len(steps), 'seconds_per_step': per_step, 'tokens_per_second_per_rank': rate,
            'tokens_per_second_all_ranks': sum(s['input_tokens'] for step in steps for s in step) / seconds,
            'projected_hours_n_max': full, 'n': n, 'projected_hours_n': projected_hours(tokens[:n], rate),
            'save_steps': max(1, int(CKPT_SECONDS / per_step)), 'rules': rules}


def recipe(arm, save_steps, **extra):
    spec = ARMS[arm]
    return {**RECIPE, 'lora_regex': spec['regex'], 'save_steps': save_steps, **extra, **dict.fromkeys(SWITCHES, True),
            **({'image_text_model': True} if spec['model_class'] == IMAGE_TEXT else {})}


def torchrun(model, rows, out):
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={WORLD}',
            '-m', 'fastfill.v1_1.train_dpo', '--model', str(model), '--pairs', str(rows), '--out', str(out)]


def smoke(args, run=subprocess.run):
    out = Path(args.out)
    if (out / 'verdict.json').exists():
        return h.strict_json((out / 'verdict.json').read_text())
    if os.path.lexists(out):
        raise FileExistsError(f'{out} exists without a verdict; inspect it')
    arm = getattr(args, 'arm', 'A')  # A, or J27 (amendment 028: same model and recipe, J's rows).
    value = load_pool(args.pool)
    _eos(value, '27b')
    base = base_receipt(args.base)
    if _model_type(base) != MODEL_TYPE:
        raise ValueError('the smoke measures Arm A (the 27B)')
    fitting = [e for e in value['ledger'] if e['pool'] and e['tokens']['27b']['fits']]
    longest = sorted(fitting, key=lambda e: (-e['tokens']['27b']['total'], e['uid']))[:WORLD * SMOKE_STEPS]
    if len(longest) != WORLD * SMOKE_STEPS:
        raise ValueError('too few admitted rooms for the smoke')
    out.mkdir(parents=True)
    d._publish_dir(out / 'rows', {_rows_name(arm): d._jsonl(pool_rows(value, [e['uid'] for e in longest]))})
    settings = recipe(arm, 0, perf_out=str(out / 'perf'))
    started = time.monotonic()
    code = run(torchrun(base['path'], out / 'rows' / _rows_name(arm), out / 'train') + d._flags(settings)).returncode
    try:
        if code != 0:
            raise RuntimeError(f'smoke training exited {code}')
        perf = [h.strict_json(p.read_text()) for p in sorted((out / 'perf').glob('perf-rank*.json'))]
        measured = smoke_verdict(perf, [e['tokens']['27b']['total'] for e in fitting], arm=arm)
    except Exception as error:  # A crash is not a stop rule: no verdict; the attempt is kept aside and can be rerun.
        failed = out.with_name(f'{out.name}.failed-{os.environ.get("SLURM_JOB_ID") or os.getpid()}')
        out.rename(failed)
        raise RuntimeError(f'{error}; no verdict, attempt kept as {failed}') from error
    verdict = {**measured,
               'wall_seconds': time.monotonic() - started, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               'pool_manifest_sha256': value['manifest_sha256'], 'base_files_sha256': i._hash(base['files']),
               'rows': {'rooms': len(longest), 'longest_tokens': longest[0]['tokens']['27b']['total'],
                        'shortest_tokens': longest[-1]['tokens']['27b']['total'],
                        'manifest_sha256': h.sha256(out / 'rows/MANIFEST.json')},
               'admitted_fitting_rooms': len(fitting), 'recipe': settings, 'code_sha256': _code()}
    i.write_once(out / 'verdict.json', verdict)
    return verdict


def _eos(value, key):
    """The pool's tokenizer EOS (train_dpo's appended token) must be the inference stop token, before any GPU work."""
    if value['summary']['tokenizers'][key]['eos_token'] != EOS:
        raise ValueError(f"{key} tokenizer EOS {value['summary']['tokenizers'][key]['eos_token']!r} is not {EOS}")


def load_verdict(path, value, base):
    """SMOKE_PASS on this pool and base, and vLLM loaded that smoke's adapter (smoke-vllm)."""
    path = Path(path)
    verdict = h.strict_json((path / 'verdict.json').read_text())
    if verdict['status'] != SMOKE_PASS:
        raise ValueError(f'the smoke stopped Arm A: {verdict["reasons"]}')
    if (verdict['pool_manifest_sha256'], verdict['base_files_sha256']) != (value['manifest_sha256'], i._hash(base['files'])):
        raise ValueError('the smoke measured another pool or base')
    digest = h.sha256(path / 'verdict.json')
    load = h.strict_json((path / VLLM_LOAD).read_text()) if (path / VLLM_LOAD).is_file() else {}
    if (load.get('status'), load.get('verdict_sha256')) != ('OK', digest):
        raise ValueError(f'vLLM has not served this smoke adapter ({path / VLLM_LOAD}); run smoke-vllm first')
    return verdict, digest, h.sha256(path / VLLM_LOAD)


# ---- train ---------------------------------------------------------------------------------------------------------

def train(args, run=subprocess.run):
    spec = ARMS[args.arm]
    root = Path(args.run_root) / spec['dir']
    if is_policy(root):
        return {'status': 'A021_POLICY_COMPLETE', 'arm': args.arm, 'policy_sha256': h.sha256(root / POLICY)}
    value = load_pool(args.pool)
    _eos(value, spec['tokens'])
    base = base_receipt(args.base)
    if _model_type(base) != spec['model_type']:
        raise ValueError(f'Arm {args.arm} trains on {spec["model_type"]}, not {_model_type(base)}')
    if args.arm != 'B':
        verdict, verdict_sha, load_sha = load_verdict(args.smoke, value, base)
        n, save_steps = verdict['n'], verdict['save_steps']
    else:
        if is_policy(Path(args.run_root) / ARMS['A']['dir']):
            raise ValueError('Arm A has a trained policy; Arm B is only the fallback')
        n, save_steps, verdict_sha, load_sha = None, B_SAVE_STEPS, None, None
    uids = take(value, spec['tokens'], n)
    if not uids:
        raise ValueError('no admitted rows')
    rows_dir, name = root / 'rows', _rows_name(args.arm)
    if os.path.lexists(rows_dir):
        files = d._verified_dir(rows_dir)
        if [r['uid'] for r in h._rows(rows_dir, name, files[name])] != uids:
            raise ValueError(f'{rows_dir} holds other rooms')
    else:
        root.mkdir(parents=True, exist_ok=True)
        d._publish_dir(rows_dir, {name: d._jsonl(pool_rows(value, uids))})
    rows = rows_dir / name
    settings = recipe(args.arm, save_steps)
    receipt = {'protocol': 'amendment-021-train-v1', 'arm': args.arm, 'n': len(uids),
               'n_rule': ('min(40,000, rooms fitting 10 h at the smoke throughput), sha256 order' if args.arm != 'B'
                          else 'every admitted room that fits the 8B (fallback)'),
               'smoke_verdict_sha256': verdict_sha, 'smoke_vllm_load_sha256': load_sha,
               'pool_manifest_sha256': value['manifest_sha256'],
               'rows': {'uids_sha256': d._digest(uids), 'manifest_sha256': h.sha256(rows_dir / 'MANIFEST.json')},
               'base': {k: base[k] for k in ('path', 'files', 'repository', 'revision')},
               'model_class': spec['model_class'], 'recipe': settings, 'fixed': FIXED, 'code_sha256': _code()}
    try:  # Written before training starts; a resumed segment must find the same receipt.
        i.write_once(root / 'run_receipt.json', receipt)
    except FileExistsError:
        if h.strict_json((root / 'run_receipt.json').read_text()) != receipt:
            raise ValueError('run identity differs from the existing run_receipt.json') from None
    from fastfill.v1_1 import train_dpo, training_guard
    out = root / 'train'
    training_guard.guard_training(base['path'], rows, out, {**settings, **FIXED, 'n': len(uids)},
                                  [Path(train_dpo.__file__), Path(__file__), Path(d.__file__)])
    if not (out / 'final' / 'completion.json').exists():
        d.preflight(base['path'], rows, len(uids))
        run(torchrun(base['path'], rows, out) + d._flags(settings), check=True)
    completion = d._completion(out, len(uids), settings)
    final = out / 'final'
    policy = {'protocol': 'amendment-021-policy-v1', 'arm': args.arm, 'stage': spec.get('stage', f'a021-{spec["dir"]}-sft'),
              'model_class': spec['model_class'], 'base': receipt['base'],
              'adapter': {'path': str(final.resolve()), 'files': h.artifact_manifest(final)},
              'training_data': {'manifest_sha256': value['summary']['inputs']['main_manifest_sha256'],
                                'pool_manifest_sha256': value['manifest_sha256'], 'rows': receipt['rows'], 'count': len(uids)},
              'training': {'run_receipt_sha256': h.sha256(root / 'run_receipt.json'),
                           'guard_sha256': h.sha256(out / 'training_guard.json'), 'completion': completion},
              'code_sha256': _code()}
    i.write_once(root / POLICY, policy)
    return {'status': 'A021_POLICY_COMPLETE', 'arm': args.arm, 'n': len(uids), 'policy_sha256': h.sha256(root / POLICY)}


# ---- kernels -------------------------------------------------------------------------------------------------------

def kernels(strict=True):
    """The path transformers' qwen3_5 takes; strict (GPU jobs): fla kernels bound and a tiny layer agrees with torch."""
    import importlib.metadata
    import importlib.util
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    def name(f):
        return None if f is None else f'{f.__module__}.{getattr(f, "__qualname__", type(f).__name__)}'
    spec = importlib.util.find_spec('fla')
    path = {'cuda': torch.cuda.is_available(), 'fla_available': bool(m.is_flash_linear_attention_available()),
            'fla': ({'file': spec.origin, 'version': importlib.metadata.version('fla-core')} if spec else None),
            'chunk_gated_delta_rule': name(m.chunk_gated_delta_rule or m.torch_chunk_gated_delta_rule),
            'recurrent_gated_delta_rule': name(m.fused_recurrent_gated_delta_rule or m.torch_recurrent_gated_delta_rule),
            'rms_norm_gated': name(m.FusedRMSNormGated or m.Qwen3_5RMSNormGated),
            'causal_conv1d': name(m.causal_conv1d_fn) or 'torch F.conv1d (causal_conv1d not installed)'}
    if not strict:
        return path
    if not all(path[k].startswith('fla.') for k in ('chunk_gated_delta_rule', 'recurrent_gated_delta_rule', 'rms_norm_gated')):
        raise RuntimeError(f'qwen3_5 would take the torch fallback: {path}')
    from transformers import Qwen3_5TextConfig
    config = Qwen3_5TextConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=1, layer_types=['linear_attention'],
                               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=64,
                               linear_value_head_dim=64, num_attention_heads=4, num_key_value_heads=2, head_dim=64)
    torch.manual_seed(0)
    layer = m.Qwen3_5GatedDeltaNet(config, 0).to('cuda', torch.bfloat16)
    x = torch.randn(1, 128, 256, device='cuda', dtype=torch.bfloat16)
    outputs = []
    for rule in (layer.chunk_gated_delta_rule, m.torch_chunk_gated_delta_rule):
        layer.chunk_gated_delta_rule = rule
        layer.zero_grad(set_to_none=True)
        y = layer(x)
        y.float().square().mean().backward()
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        if not torch.isfinite(y).all() or not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f'non-finite gated delta net output or gradient with {name(rule)}')
        outputs.append(y.float())
    scale = outputs[1].abs().max().item()
    path['tiny_layer_max_abs_diff'] = (outputs[0] - outputs[1]).abs().max().item()
    if path['tiny_layer_max_abs_diff'] > 0.1 * scale + 1e-3:
        raise RuntimeError(f'fla and torch gated delta rule disagree: {path}')
    return path


# ---- fidelity probe ------------------------------------------------------------------------------------------------

def probe_records(data, arm='placement'):
    rows, info = i.load_eval(data, arm)
    return sorted(rows, key=lambda r: (rank(r['uid'], PROBE_SALT), r['uid']))[:PROBE_ROOMS], info


def _cap(record, ids):
    from fastfill.comparison.llm_open import output_cap
    return min(MAX_MODEL_LEN - len(ids), output_cap(record, {'thinking': False}))


def _identity(args):
    return d.inference_identity(SimpleNamespace(base=Path(args.base), model=Path(args.policy), arm='placement',
                                                exact_policy=True))


def _hf_runtime(policy):
    """HF exact: the training class in BF16, the FP32 adapter unmerged; every saved adapter tensor must be loaded."""
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, GenerationConfig
    loader = AutoModelForImageTextToText if policy['model_class'] == IMAGE_TEXT else AutoModelForCausalLM
    base = loader.from_pretrained(policy['training_base'], local_files_only=True, dtype=torch.bfloat16,
                                  attn_implementation='sdpa', device_map={'': 0})
    model = PeftModel.from_pretrained(base, policy['adapter'], is_trainable=False, autocast_adapter_dtype=True).eval()
    saved, loaded = load_file(str(Path(policy['adapter']) / 'adapter_model.safetensors')), get_peft_model_state_dict(model)
    if saved.keys() != loaded.keys() or any(not torch.equal(saved[k], loaded[k].cpu()) for k in saved):
        raise ValueError('the adapter did not load exactly onto the training class')
    tok = d._model_tokenizer(policy['adapter'])
    eos = tok.get_vocab()[EOS]

    def generate(ids, cap):
        x = torch.tensor([ids], device='cuda:0')
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=cap, eos_token_id=eos, pad_token_id=eos,
                                  use_cache=True)
        with torch.inference_mode():
            return model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=config)[0].tolist()
    return tok, eos, generate


def probe_hf(args, runtime=_hf_runtime):
    out = Path(args.out)
    if os.path.lexists(out / 'hf.jsonl'):
        raise FileExistsError(out / 'hf.jsonl')
    identity = _identity(args)
    records, info = probe_records(args.data)
    out.mkdir(parents=True, exist_ok=True)
    partial, digest = out / 'hf.partial.jsonl', i._hash(identity)  # Per room, so a timed-out job resumes.
    rows = [h.strict_json(x) for x in partial.read_text().splitlines()] if partial.exists() else []
    if [(r['uid'], r['identity_sha256']) for r in rows] != [(r['uid'], digest) for r in records[:len(rows)]]:
        raise ValueError(f'{partial} holds other rooms or another policy')
    tok, eos, generate = runtime(identity['policy']) if len(rows) < len(records) else (None, None, None)
    with partial.open('a', encoding='utf-8') as stream:
        for record in records[len(rows):]:
            ids = i.prompt_ids(tok, record)
            cap, start = _cap(record, ids), time.monotonic()
            raw, done = i.decode_output(tok, generate(ids, cap), len(ids), eos, cap)
            rows.append({'uid': record['uid'], 'engine': 'hf_exact', 'input_tokens': len(ids), 'max_tokens': cap,
                         'raw': raw, **done, 'seconds': time.monotonic() - start, 'identity_sha256': digest})
            stream.write(json.dumps(rows[-1], ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    _once(out / 'hf.jsonl', d._jsonl(rows))
    partial.unlink()
    i.write_once(out / 'hf-receipt.json', {'policy': identity['policy'], 'identity_sha256': i._hash(identity),
                                           'data_manifest_sha256': info['manifest_sha256'],
                                           'versions': d._versions('torch', 'transformers', 'peft', 'fla-core'),
                                           'decode': 'greedy, batch 1, eos = pad = <|im_end|>, cache on'})
    return {'status': 'PROBE_HF_DONE', 'rooms': len(rows)}


def _vllm_decode(base, adapter, records, name, load=None):
    """sample_vllm's engine + MULTIMODAL_ENGINE on base, adapter as one LoRA, greedy (k = 0) per record."""
    from fastfill.comparison import sample_vllm as sv
    tok = d._model_tokenizer(adapter)
    eos = tok.get_vocab()[EOS]
    jobs = []
    for record in records:
        ids = i.prompt_ids(tok, record)
        jobs.append((record, ids, _cap(record, ids)))
    engine = {**sv.ENGINE, **sv.MULTIMODAL_ENGINE, 'model': str(base)}
    llm, sampling, lora_request = (load or sv.load_vllm)(engine)
    values = [sv.params(cap, record['uid'], 0, eos) for record, _, cap in jobs]
    start = time.monotonic()
    outputs = llm.generate([{'prompt_token_ids': ids} for _, ids, _ in jobs], [sampling(**p) for p in values],
                           lora_request=lora_request(name, 1, str(adapter)), use_tqdm=False)
    rows = []
    for (record, ids, cap), output in zip(jobs, outputs, strict=True):
        if list(output.prompt_token_ids) != ids or len(output.outputs) != 1:
            raise ValueError('vLLM output does not match its request')
        completion = output.outputs[0]
        raw, done = i.decode_output(tok, list(completion.token_ids), 0, eos, cap)
        rows.append({'uid': record['uid'], 'engine': 'vllm_lora', 'input_tokens': len(ids), 'max_tokens': cap, 'raw': raw,
                     **done, 'finish_reason': completion.finish_reason})
    return rows, {'engine': engine, 'sampling': values[0] if values else None, 'wall_seconds': time.monotonic() - start,
                  'versions': d._versions('vllm', 'torch', 'transformers')}


def smoke_vllm(args, load=None):
    """vLLM serves the smoke's 27B adapter (the probe's engine): vllm-load.json, written only on success."""
    smoke_dir = Path(args.smoke)
    out = smoke_dir / VLLM_LOAD
    if os.path.lexists(out):
        raise FileExistsError(out)
    verdict = h.strict_json((smoke_dir / 'verdict.json').read_text())
    if verdict['status'] != SMOKE_PASS:
        raise ValueError(f'the smoke stopped Arm A: {verdict["reasons"]}')
    if i._hash(base_receipt(args.base)['files']) != verdict['base_files_sha256']:
        raise ValueError('base differs from the smoke base')
    adapter = smoke_dir / 'train' / 'final'
    files = h.artifact_manifest(adapter)
    records, info = probe_records(args.data, _infer_arm(getattr(args, 'arm', 'A')))  # J27: the joint requests.
    rows, run = _vllm_decode(args.base, adapter, records[:VLLM_SMOKE_ROOMS], 'a021-smoke', load)
    i.write_once(out, {'status': 'OK', 'verdict_sha256': h.sha256(smoke_dir / 'verdict.json'),
                       'adapter': {'path': str(adapter.resolve()), 'files': files}, 'base': str(args.base),
                       'data_manifest_sha256': info['manifest_sha256'], 'rows': rows, **run,
                       'rule': f'the first {VLLM_SMOKE_ROOMS} probe rooms; OK = the LoRA loaded and every room decoded'})
    return {'status': 'OK', 'rooms': len(rows), 'stopped_on_eos': sum(r['stopped_on_eos'] for r in rows)}


def probe_vllm(args, load=None):
    out = Path(args.out)
    if os.path.lexists(out / 'vllm.jsonl'):
        raise FileExistsError(out / 'vllm.jsonl')
    identity = _identity(args)
    records, info = probe_records(args.data)
    policy = identity['policy']
    rows, run = _vllm_decode(policy['training_base'], policy['adapter'], records, 'a021', load)
    out.mkdir(parents=True, exist_ok=True)
    _once(out / 'vllm.jsonl', d._jsonl(rows))
    i.write_once(out / 'vllm-receipt.json', {'policy': policy, 'identity_sha256': i._hash(identity), 'engine': run['engine'],
                                             'sampling': run['sampling'], 'data_manifest_sha256': info['manifest_sha256'],
                                             'wall_seconds': run['wall_seconds'], 'versions': run['versions']})
    return {'status': 'PROBE_VLLM_DONE', 'rooms': len(rows)}


def _score(task):
    from fastfill.comparison import geometry as g
    from fastfill.comparison import report_v2 as r2
    room, request, sample = task
    row = g.score(room, request['messages'], sample['raw'], 'placement')
    return r2.cap_failure(row) if sample['reached_token_cap'] and not sample['stopped_on_eos'] else row


def probe_report(args):
    from fastfill.comparison import report_v2 as r2
    out = Path(args.out)
    if any(os.path.lexists(out / n) for n in ('report.json', 'report.md', 'MANIFEST.json')):
        raise FileExistsError(f'probe report exists in {out}')
    records, info = probe_records(args.data)
    rooms = h._by_uid(h._rows(Path(args.data), 'rooms.jsonl', info['files']['rooms.jsonl']))
    runs = {e: h._by_uid([h.strict_json(x) for x in (out / f'{e}.jsonl').read_text().splitlines()]) for e in ('hf', 'vllm')}
    uids = [r['uid'] for r in records]
    if any(list(v) != uids for v in runs.values()):
        raise ValueError('probe runs cover other rooms')
    scored = {e: [_score((rooms[r['uid']], r, runs[e][r['uid']])) for r in records] for e in runs}
    rate = {e: {k: sum(row[k] for row in rows) / len(rows) for k in ('complete', 'geometry_success')}
            for e, rows in scored.items()}
    same = sum(runs['hf'][u]['raw'] == runs['vllm'][u]['raw'] for u in uids)
    contrast = r2.contrasts(scored['vllm'], scored['hf'], ['geometry_success', 'complete'], 2000)
    report = {'status': 'A021_PROBE_COMPLETE', 'rooms': len(uids), 'room_rule': f"lowest sha256('{PROBE_SALT}' + uid)",
              'rates': rate, 'checker_success_vllm_minus_hf': contrast['geometry_success'],
              'complete_vllm_minus_hf': contrast['complete'], 'identical_outputs': {'count': same, 'share': same / len(uids)},
              'receipts': {e: h.sha256(out / f'{e}-receipt.json') for e in runs},
              'data_manifest_sha256': info['manifest_sha256'], 'merged_copy': 'not listed (needs an 011-section-5 check)'}
    ci = contrast['geometry_success']['ci95']
    lines = ['# 修正案 021：27B 保真探针（HF 精确 vs vLLM LoRA）', '',
             f"- 房间：dev-select 中 sha256('{PROBE_SALT}' + uid) 最小的 {len(uids)} 间。",
             f"- 冻结检查器成功率：HF {rate['hf']['geometry_success']:.3f}，vLLM {rate['vllm']['geometry_success']:.3f}；"
             f"完成率：HF {rate['hf']['complete']:.3f}，vLLM {rate['vllm']['complete']:.3f}。",
             f"- vLLM − HF 成功率差：{contrast['geometry_success']['estimate']:+.3f}"
             + (f"（房屋聚类 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]）" if ci else '（CI 未定义）') + '。',
             f'- 输出完全相同的比例：{same}/{len(uids)} = {same / len(uids):.2f}。',
             '- 合并权重副本：未列出（需先做 011 §5 式检查）。', '- 仅描述，不做任何选择。', '']
    _once(out / 'report.json', (json.dumps(report, ensure_ascii=False, indent=1) + '\n').encode('utf-8'))
    _once(out / 'report.md', '\n'.join(lines).encode('utf-8'))
    i.write_once(out / 'MANIFEST.json', {'files': h.artifact_manifest(out)})
    return {k: report[k] for k in ('status', 'rooms', 'rates', 'identical_outputs')}


# ---- cli -----------------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('pool')
    for flag in ('data', 'corpus', 'lineage', 'base27', 'base8', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--dev', type=Path, nargs='+', required=True)
    step.add_argument('--test', type=Path, nargs='+', required=True)
    step.add_argument('--workers', type=int, default=1)
    step.add_argument('--sample', type=int, default=200)
    step = sub.add_parser('kernels')
    step.add_argument('--report', action='store_true')
    step = sub.add_parser('smoke')
    for flag in ('pool', 'base', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--arm', choices=('A', 'J27'), default='A')
    step = sub.add_parser('smoke-vllm')
    for flag in ('data', 'smoke', 'base'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--arm', choices=('A', 'J27'), default='A')
    step = sub.add_parser('train')
    step.add_argument('--arm', choices=tuple(ARMS), required=True)
    for flag in ('pool', 'base', 'run-root'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--smoke', type=Path)
    for name in ('probe-hf', 'probe-vllm', 'probe-report'):
        step = sub.add_parser(name)
        for flag in ('data', 'policy', 'base', 'out'):
            step.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'train' and (args.arm != 'B') != (args.smoke is not None):
        parser.error('--smoke is required for Arm A (and J27) and not taken by Arm B')
    return args


def main(argv=None):
    args = parse_args(argv)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    steps = {'pool': pool, 'smoke': smoke, 'smoke-vllm': smoke_vllm, 'train': train, 'probe-hf': probe_hf, 'probe-vllm': probe_vllm,
             'probe-report': probe_report, 'kernels': lambda a: kernels(strict=not a.report)}
    result = steps[args.command](args)
    print(json.dumps(result, ensure_ascii=False))
    if args.command == 'smoke' and result['status'] != SMOKE_PASS:
        sys.exit(1)


if __name__ == '__main__':
    main()
