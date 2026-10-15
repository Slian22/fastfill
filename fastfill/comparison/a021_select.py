"""Amendment 021 sections 1.4, 3 and 4 (arm P, size-fixed placement): the confirm slice, O2nt reference rows, the
engine check, candidate evaluation, the final cross-amendment selection and the delivered model's test rows.

python -m fastfill.comparison.a021_select STEP ...   (sbatch: isambard_021_eval.sbatch; chain: submit_021_eval.sh)
All outputs live under RUN/a021 and are new-only (a rerun of select must reproduce, code hashes aside).

confirm --data MAIN --sft-runs SFT_J SFT_P --dev RUN/dev-select-v1 --run-root RUN
    Pool: the SFT common length-admitted dev pool (dpo_stages._admitted 'dev', the pool dev_select.build draws from)
    minus every dev-select UID and house group. 3,000 rooms stratified by scale_curve.BINS on the room's object count:
    quotas proportional to the pool's bin counts (largest remainder, ties in bin order), each bin's quota by ascending
    sha256('a021-confirm:' + uid), uid tie-break. Hash order = the chosen rooms by that key. Published as
    a021/confirm-v1 in dev-select-v1's layout (per arm requests.jsonl = system + user turns, rooms.jsonl,
    size-estimates.jsonl = the P request sizes, MANIFEST.json with sorted expected_uids and the selection block), its
    first 1,000 rooms in hash order as a021/engine-v1 (same layout, the section-3 engine check), and the llm_open
    --all-rooms cohorts of dev-select-v1 and confirm-v1 as a021/cohorts/{dev-select,confirm}.json. Any infer_shard plan
    / shard / merge, sample_vllm or llm_open run reads them unchanged.
greedy --data DATA --base BASE (--adapter A --training-run T | --model STAGE-merged --exact-policy) --arm placement
       --out OUT [--batching objects|hash]
    vLLM greedy, BF16 LoRA: sample_vllm's k = 0 request only (its model identity and rehash, prompt ids from the
    adapter dir's tokenizer, cap min(40960 - prompt, 1024 + 128 x objects), stop on <|im_end|> only, HF decode), one
    answer per room, write-once per-room receipts (a rerun resumes under the same identity). Batch composition
    (rooms per LLM.generate call, in the identity): objects = (-objects, uid) in chunks of <= 400k cap tokens; hash =
    sha256('a021-batch:' + uid) order in chunks of <= 100k (R's second run, for the run-to-run spread). A base whose
    config has a vision_config (Qwen3.8-27B) gets image/video inputs 0. Finished: records.jsonl (expected_uids order),
    result.json (INFERENCE_COMPLETE) and MANIFEST.json, the infer.py layout evaluate_pair._inference reads.
o2nt-score --run-root RUN [--workers N]
    Section 1.4, descriptive: the O2nt runs (llm_open --system O2nt --arm placement) on dev-select, confirm and the
    1,454 cohort, scored as evaluate_llm scores a placement run (its _llm binding, addendum-B2 _enforced, _score_llm;
    the cap rule is not applied, as there), both variants. Reasoning tokens must be 0 (asserted again from the result).
    Per slice and variant: all-request checker success, pass among complete and the selection metrics below.
    -> a021/o2nt-scores (scores.json, report.md in Chinese). It decides nothing.
engine-check --run-root RUN [--workers N]
    Section 3: R (rpo-dpo1's exact policy) and P-SFT on engine-v1 under HF exact (infer_shard on a021/hf/engine/plan)
    and vLLM greedy (greedy above). Common vLLM engine iff |checker(R, HF) - checker(R, vLLM)| <= 0.015; else 8B
    models are served on HF exact and the 27B on vLLM. -> a021/engine-check/engine.json.
jobs STAGE --run-root RUN [--arm-kind A|B]   (login node, submit_021_eval.sh)
    The GPU jobs of STAGE (o2nt, engine, candidates, confirm, test), one per line, then 'ceiling GPU-H'. Finished
    runs are left out; a vLLM run resumes; an HF shard dir that exists resumes; an unfinished HF merge dir refuses.
    candidates first writes a021/candidates.json (write-once): R, 013's adopted model (dev-eval-013/selection.json via
    onpolicy_dpo.load_selection), 018's (dev-eval-018/selection.json via distill_018_chain.load_selection; both files
    must exist, adopted or not; 018's HF test run of record is its own infer-placement-007, 013's test run is deferred
    to here) and the 021 arm (A: RUN/a021/27b on Qwen3.8-27B, B: RUN/a021/8b on Qwen3-8B, refused once A's receipt
    exists; the
    a021_train policy dir with its a021-policy.json, loaded --model DIR --exact-policy through
    dpo_stages.inference_identity), each with its section-3 serving engine. Dev-select
    runs of every candidate on its engine (HF: the existing dev-eval-010/013/018 runs are reused), R twice on vLLM
    (objects and hash batching), and the vLLM confirm runs (R and every vLLM-served candidate). HF confirm runs are
    only needed for an HF-served winner: jobs confirm lists them (winner and R) from a021/dev-selection.json.
argv KIND ... --run-root RUN   (isambard_021_eval.sbatch)   the python argv of one job, one token per line.
select --run-root RUN [--workers N]
    Section 4 on dev-select (600 rooms), every candidate on its serving engine, R on every engine in use:
    complete; frozen-checker success over all requests (incomplete = fail; a cap hit without EOS fails, amendment 005);
    OOR (mean over complete rooms); strict admission (downstream.accept, incomplete = refused); GT-relation hold rate;
    checker success in rooms with 21+ objects. GT relations (the amendment names validate.holds without a relation set;
    this tool's reading): the deterministic candidates of build.extract_constraints that hold on GT, i.e. 'on' for every
    GT-supported object and, in polygon rooms, 'against_wall' for every floor object against a wall, both over
    objects that are not flat (anchors.FLAT) and >= 5 cm tall; the rate is held / all GT relations of all requests
    (an incomplete room holds none). Guards vs R on the same engine (inclusive): complete >= R - 0.005, OOR <= R + 0.01,
    admission >= R - 0.01, holds >= R - 0.03, 21+ checker >= R - 0.02 (a value missing in one of the two fails, in both
    passes: nothing to measure). Among R and the guarded
    candidates the best checker wins; within the tie band max(0.01, |checker(R vLLM objects) - checker(R vLLM hash)|)
    the cheaper model (8B before 27B), then R, then higher checker, then candidate order. Confirmation (winner != R):
    winner - R checker on confirm-v1, same engine, paired house-cluster bootstrap (report_v2.contrasts, 2,000 draws,
    seed 42); confirmed iff the 95% CI lower bound > 0, else R stays. Writes a021/dev-selection.json; then, if the
    confirm runs exist (or R won), a021/final-report.md (Chinese) and a021/final-selection.json (written last).
test-geometry --run-root RUN [--workers N]
    The delivered winner and R on all test rooms with the winner's engine (on HF the runs of record are reused: R's
    infer-placement-005, 018's infer-placement-007; nothing reruns on test):
    a021/test-geometry-NAME/placement-geometry.jsonl (evaluate_pair rows plus the cap flags, every request; the input
    output_repair, test_apply_012 and agents_016 take) and summary.json (geometry.summarize, the selection metrics, and
    for the winner the paired contrasts winner - R).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

from fastfill import validate
from fastfill.anchors import FLAT
from fastfill.comparison import distill_018_chain as ch
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import dev_select as ds
from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer
from fastfill.comparison import llm_agent as L
from fastfill.comparison import llm_open as lo
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import sample_vllm as sv
from fastfill.comparison import scale_curve as sc
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.scene import head

ARM, A21 = 'placement', 'a021'
SALT, BATCH_SALT, CONFIRM_ROOMS, ENGINE_ROOMS = 'a021-confirm:', 'a021-batch:', 3000, 1000
BINS = tuple(name for name, _, _ in sc.BINS)
R, PSFT = 'r', 'psft'
ORDER = (R, 'op13', 'd18', 'arm')  # Candidate order (last tie-break).
SIZES = ('8B', '27B')  # Cheaper first (section 4 tie band).
ENGINE_GAP, TIE_MIN, BIG, EPS, REPEATS = 0.015, 0.01, 21, 1e-12, 2000
GUARDS = (('complete_rate', 1, 0.005), ('oor_mean', -1, 0.01), ('accepted_rate', 1, 0.01), ('holds_rate', 1, 0.03),
          ('checker_21plus', 1, 0.02))  # (metric, +1 higher is better / -1 lower is better, allowed slack vs R).
BATCHINGS = {'objects': 400_000, 'hash': 100_000}  # Cap tokens per LLM.generate call.
VLLM_LORA = 'vllm_bf16_lora_on_bf16_base'
GREEDY = 'amendment-021-vllm-greedy-v1'
ARM_DIRS, POLICY = {'A': '27b', 'B': '8b'}, 'a021-policy.json'  # a021_train's policy dirs and receipt (section 2).
O2NT = ('dev-select', 'confirm', 'cohort')
# Slurm time per job (hours). vLLM greedy: 3,000 rooms x 9 samples of an 8B took 0.62 GPU-h (amendment 013); HF exact
# is about 120 rooms per GPU-h (amendment 010 test shards).
VLLM_HOURS = {'8B': {'dev-select': 1, 'engine': 1, 'confirm': 2, 'test': 3},
              '27B': {'dev-select': 2, 'engine': 2, 'confirm': 4, 'test': 8}}
HF_SHARDS = {'dev-select': (2, 4), 'engine': (4, 3), 'confirm': (6, 6), 'test': (6, 14)}  # (shards, hours each)
O2NT_HOURS, O2NT_GPUS = {'dev-select': 1, 'confirm': 3, 'cohort': 2}, lo.EXTRA_SYSTEMS['O2nt']['tp']
ENGINE_RULE = ('common vLLM greedy (BF16 LoRA) engine for every candidate iff |checker(R, HF exact) - checker(R, vLLM '
               'greedy)| <= 0.015 on the first 1,000 confirm rooms in hash order; otherwise each candidate on its serving '
               'engine: 8B models on HF exact, the 27B on vLLM')
RULE = ('section 4: on the 600 dev-select rooms, each candidate on its section-3 engine, guards vs R on the same engine '
        '(complete >= R - 0.005, OOR <= R + 0.01, strict admission >= R - 0.01, GT-relation hold rate >= R - 0.03, '
        'checker success in rooms with 21+ objects >= R - 0.02; inclusive; a value missing in one of the two fails, in '
        'both passes); the best raw frozen-checker success among R and the guarded candidates; within max(0.01, R '
        'run-to-run vLLM spread) the '
        'cheaper model (8B before 27B), then R, then higher checker, then candidate order; a winner other than R is '
        'delivered only if winner - R on the 3,000 confirm rooms (same engine) has a paired house-cluster bootstrap 95% '
        'CI lower bound > 0, else R stays')
RULE_ZH = ('在 dev-select 的 600 间上，每个候选用 §3 的引擎；护栏相对同引擎的 R（含边界）：完整率 ≥ R − 0.005，OOR ≤ R + 0.01，'
           '严格准入 ≥ R − 0.01，GT 关系保持率 ≥ R − 0.03，21 个及以上物体房间的检查器成功率 ≥ R − 0.02（某值只在一方缺失则不满足，'
           '两方都缺失视为无可度量而满足）。R 与通过护栏的候选中取原始冻结检查器成功率（全部请求，未完成算失败）最高者；'
           '在 max(0.01, R 的 vLLM 两次运行差) 的平局带内取更便宜的模型（8B 先于 27B），再 R，再更高的检查器成功率，再候选顺序。'
           '胜者不是 R 时，须在 3,000 间确认切片上（同引擎）胜者 − R 的房屋连通组配对 bootstrap 95% CI 下界 > 0 才交付，否则 R 保持交付。')
LABELS = {R: 'R = rpo-dpo1', PSFT: 'P-SFT', 'op13': '013 采纳', 'd18': '018 采纳', 'arm': '021 臂'}


def _code():
    return {p.name: h.sha256(p) for p in map(Path, (__file__, ds.__file__, e.__file__, g.__file__, r2.__file__,
                                                     ch.__file__, sv.__file__, ev.__file__))}


def _json(value):
    return (json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def _write_once(path, value, what):
    """Write-once JSON; an existing file must hold the same value, code hashes aside."""
    if path.exists():
        if {**h.strict_json(path.read_text()), 'code_sha256': None} != {**value, 'code_sha256': None}:
            raise ValueError(f'{path} exists with another {what}')
    else:
        infer.write_once(path, value)


# ---- paths and model specs -----------------------------------------------------------------------

def models_root(run):
    return Path(run).parent.parent / 'models'  # RUN = PROJECT/runs/unknownsize-20261013


def data_dir(run, slice_):
    run = Path(run)
    return {'dev-select': run / 'dev-select-v1', 'confirm': run / A21 / 'confirm-v1', 'engine': run / A21 / 'engine-v1',
            'test': run / 'eval-data-v1', 'cohort': run / 'eval-data-v1'}[slice_]


def plan_dir(run, slice_):
    run = Path(run)
    return {'dev-select': run / 'dev-eval-010/plan', 'test': run / 'infer-v2-002/plan'}.get(
        slice_, run / A21 / 'hf' / slice_ / 'plan')


def vllm_dir(run, slice_, cand, batching='objects'):
    return Path(run) / A21 / 'vllm' / slice_ / (cand if batching == 'objects' else f'{cand}-{batching}')


def hf_root(run, slice_, cand):
    return Path(run) / A21 / 'hf' / slice_ / cand


def hf_dir(run, slice_, cand, specs):
    """The merged HF inference dir: an existing run of record where one exists, else a021/hf/SLICE/CAND."""
    existing = specs[cand].get('hf', {}).get(slice_)
    return Path(existing) if existing else hf_root(run, slice_, cand) / f'infer-{ARM}'


def run_dir(run, slice_, cand, engine, specs, batching='objects'):
    return vllm_dir(run, slice_, cand, batching) if engine == 'vllm' else hf_dir(run, slice_, cand, specs)


def fixed_specs(run):
    run, base = Path(run), str(models_root(run) / 'Qwen3-8B')
    return {R: {'size': '8B', 'base': base, 'model': str(run / 'dpo-003/placement/dpo1-merged'), 'exact_policy': True,
                'hf': {'dev-select': str(run / 'dev-eval-010/rpo-dpo1/infer-placement'),
                       'test': str(run / 'infer-placement-005')}},
            PSFT: {'size': '8B', 'base': base, 'adapter': str(run / 'sft-placement-001/placement/adapter'),
                   'training_run': str(run / 'sft-placement-001'), 'hf': {}}}


def specs_of(run):
    specs, path = fixed_specs(run), Path(run) / A21 / 'candidates.json'
    if path.exists():
        specs.update(h.strict_json(path.read_text())['candidates'])
    return specs


def model_args(spec):
    if 'model' in spec:
        return ['--model', spec['model'], '--exact-policy']
    return ['--adapter', spec['adapter'], '--training-run', spec['training_run']]


def bind(receipt, spec, engine):
    """The inference must come from this model, loaded the way ENGINE loads it (as dev_select._bind)."""
    config = receipt['config']
    if 'model' in spec:
        ok = (config.get('model') == str(Path(spec['model']).resolve())
              and config.get('lora') == (ds.EXACT if engine == 'hf' else VLLM_LORA))
    else:
        ok = (config.get('adapter') == str(Path(spec['adapter']).resolve()) and 'model' not in config
              and config.get('training_run') == str(Path(spec['training_run']).resolve())
              and (engine == 'hf') == (config.get('lora') != VLLM_LORA))
    if not ok or config.get('arm') != ARM:
        raise ValueError(f'inference is not {ARM} of {spec.get("model") or spec.get("adapter")} on {engine}')


# ---- confirm slice -------------------------------------------------------------------------------

def rank(uid, salt=SALT):
    return hashlib.sha256((salt + uid).encode('utf-8')).hexdigest()


def bin_of(objects):
    found = next((name for name, lo_, hi in sc.BINS if lo_ <= objects and (hi is None or objects <= hi)), None)
    if found is None:
        raise ValueError('room without movable objects')
    return found


def pick(pool, n=CONFIRM_ROOMS):
    """pool {uid: object count} -> (chosen uids in hash order, info); pure and order independent."""
    if len(pool) < n:
        raise ValueError(f'confirm pool has {len(pool)} rooms, {n} required')
    ordered = sorted(pool, key=lambda u: (rank(u), u))
    cells = {b: [u for u in ordered if bin_of(pool[u]) == b] for b in BINS}
    quota = {b: n * len(v) // len(pool) for b, v in cells.items()}
    for b in sorted(BINS, key=lambda b: (-(n * len(cells[b]) % len(pool)), BINS.index(b)))[:n - sum(quota.values())]:
        quota[b] += 1  # Largest remainder; a quota never exceeds its cell (n <= pool).
    chosen = sorted((u for b in BINS for u in cells[b][:quota[b]]), key=lambda u: (rank(u), u))
    return chosen, {'pool_by_bin': {b: len(v) for b, v in cells.items()}, 'quotas': quota}


def _publish_data(out, uids, rows, rooms, extra):
    """dev_select.build's eval-data-v1 layout (both arms; the target turn dropped) for UIDS, published new-only."""
    uids = sorted(uids)
    files = {f'{arm}/requests.jsonl': [] for arm in h.ARMS} | {'rooms.jsonl': [], 'size-estimates.jsonl': []}
    for uid in uids:
        component, room = rows['joint'][uid]['group'], rooms[uid]
        if room['source'] != rows['joint'][uid]['source'] or any(rows[a][uid]['group'] != component for a in h.ARMS):
            raise ValueError(f'dev row/room identity mismatch: {uid}')
        meta = {'uid': uid, 'source': room['source'], 'group': component, 'component': component, 'split': 'dev',
                **({'room_type': room['room_type']} if room.get('room_type') is not None else {})}
        for arm in h.ARMS:
            files[f'{arm}/requests.jsonl'].append(infer.validate_request({**meta, 'messages': rows[arm][uid]['messages'][:2]}, arm))
        files['rooms.jsonl'].append({**room, 'group': component, 'component': component, 'split': 'dev'})
        request = h.strict_json(rows['placement'][uid]['messages'][1]['content'])
        files['size-estimates.jsonl'].append({'uid': uid, 'estimated_sizes': {o['id']: o['size'] for o in request['objects']}})
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        for arm in h.ARMS:
            (dest / arm).mkdir()
        for name, values in files.items():
            (dest / name).write_bytes(d._jsonl(values))
        h.write_json(dest / 'MANIFEST.json', {'version': 1, 'split': 'dev', 'expected_uids': uids,
                                              'uids_sha256': ds._digest(uids), 'counts': {'dev': len(uids)}, **extra,
                                              'input_policy': 'system_user_only_labels_and_inventory_separate',
                                              'code_sha256': {Path(__file__).name: h.sha256(__file__)},
                                              'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return ds._digest(uids)


def confirm(args):
    run = Path(args.run_root)
    root = run / A21
    outs = (root / 'confirm-v1', root / 'engine-v1', root / 'cohorts')
    taken = [str(p) for p in outs if os.path.lexists(p)]
    if taken:
        raise FileExistsError(f'refusing existing outputs: {taken}')
    pairs, data = h.load_pair(args.data)
    if data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('the confirm slice requires the qualified main data')
    admitted, _, lengths = d._admitted(args.sft_runs, data, 'dev')
    rows = {arm: h._by_uid(pairs['dev'][arm]) for arm in h.ARMS}
    rooms = h._by_uid(h._rows(Path(args.data), 'dev_rooms.jsonl', data['files']['dev_rooms.jsonl']))
    if not set(admitted) <= rows['joint'].keys() & rooms.keys():
        raise ValueError('length-admitted dev UID outside the verified dev rows/rooms')
    dev = Path(args.dev)
    if dev.resolve() != data_dir(run, 'dev-select').resolve():
        raise ValueError(f'--dev must be {data_dir(run, "dev-select")}')
    held_uids, held_groups, held = op._held_out([dev])
    pool = {u: len(rooms[u]['objects']) for u in admitted
            if u not in held_uids and rows['joint'][u]['group'] not in held_groups}
    chosen, info = pick(pool, CONFIRM_ROOMS)
    common = {'training_manifest_sha256': data['manifest_sha256'], 'main_data_manifest_sha256': data['manifest_sha256'],
              'sft_lengths_sha256': lengths}
    selection = {'protocol': 'amendment-021-confirm-v1', 'salt': SALT, 'bins': [list(b) for b in sc.BINS],
                 'pool': {'length_admitted_dev': len(admitted), 'eligible': len(pool)},
                 'excluded': {'dev_select_uids': len(held_uids), 'dev_select_groups': len(held_groups)}, **info,
                 'held_out': held, 'hash_order': chosen}
    sha = _publish_data(root / 'confirm-v1', chosen, rows, rooms,
                        {'scope': 'amendment_021_confirm', **common, 'selection': selection})
    engine = chosen[:ENGINE_ROOMS]
    _publish_data(root / 'engine-v1', engine, rows, rooms, {
        'scope': 'amendment_021_engine_check', **common,
        'selection': {'protocol': 'amendment-021-engine-v1', 'rule': f'the first {ENGINE_ROOMS} confirm-v1 rooms in '
                      'hash order', 'confirm_uids_sha256': sha, 'hash_order': engine}})
    cohorts = {name: lo.all_rooms_cohort(data_dir(run, name)) for name in ('dev-select', 'confirm')}
    d._publish_dir(root / 'cohorts', {f'{name}.json': _json(value) for name, value in cohorts.items()})
    return {'status': 'A021_CONFIRM_COMPLETE', 'rooms': len(chosen), 'uids_sha256': sha, 'quotas': info['quotas'],
            'engine_rooms': len(engine)}


# ---- vLLM greedy ---------------------------------------------------------------------------------

def _order(rows, batching):
    if batching == 'objects':
        return sorted(rows, key=lambda r: (-sv.objects(r), r['uid']))
    return sorted(rows, key=lambda r: (rank(r['uid'], BATCH_SALT), r['uid']))


def _chunks(jobs, budget):
    """Consecutive (record, ids, cap) jobs per LLM.generate call: caps sum <= budget, at least one room."""
    groups, used = [], 0
    for job in jobs:
        if not groups or used + job[2] > budget:
            groups.append([])
            used = 0
        groups[-1].append(job)
        used += job[2]
    return groups


def greedy(args, load=sv.load_vllm):
    started, out = time.monotonic(), Path(args.out)
    rows, data = infer.load_eval(args.data, args.arm)
    model, base, adapter = sv.model_identity(args)
    engine = {**sv.ENGINE, 'model': str(base),
              **(sv.MULTIMODAL_ENGINE if 'vision_config' in h.strict_json((base / 'config.json').read_text()) else {})}
    loaded = ({'model': str(Path(args.model).resolve())} if args.model else
              {'adapter': str(Path(args.adapter).resolve()), 'training_run': str(Path(args.training_run).resolve())})
    identity = {'protocol': GREEDY, 'arm': args.arm, 'model': model,
                'config': {'arm': args.arm, **loaded, 'lora': VLLM_LORA, 'base': str(base), 'adapter_loaded': str(adapter),
                           'lora_request': [args.arm, 1, str(adapter)], 'batching': args.batching,
                           'chunk_cap_tokens': BATCHINGS[args.batching]},
                'data': {'manifest_sha256': data['manifest_sha256'],
                         'requests_sha256': data['files'][f'{args.arm}/requests.jsonl']},
                'decoding': {'greedy': 'sample_vllm.params(cap, uid, 0, im_end): temperature 0, n 1, stop on im_end only',
                             'cap': 'min(40960 - prompt tokens, 1024 + 128 x movable objects); < 1: INPUT_TOO_LONG',
                             'raw': 'infer.decode_output (HF decode up to the first stop token)'},
                'engine': engine, 'prompt': sv.PROMPT, 'versions': d._versions('vllm', 'torch', 'transformers'),
                'code_sha256': {p.name: h.sha256(p) for p in map(Path, (__file__, sv.__file__, infer.__file__,
                                                                         d.__file__, h.__file__, lo.__file__))}}
    run_hash = infer._hash(identity)
    receipt = {**identity, 'run_sha256': run_hash}
    (out / 'receipts').mkdir(parents=True, exist_ok=True)
    try:
        infer.write_once(out / 'run_receipt.json', receipt)
    except FileExistsError:
        if h.strict_json((out / 'run_receipt.json').read_text()) != receipt:
            raise ValueError('run identity mismatch with the existing run_receipt.json') from None
    if (out / 'MANIFEST.json').exists():
        return h.strict_json((out / 'result.json').read_text())
    pending = [r for r in _order(rows, args.batching) if not sv._path(out, r['uid']).exists()]
    invocation = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.getpid()}:{time.time_ns()}"

    def log(entry):
        lo._append(out / 'invocations.jsonl', {'invocation': invocation, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                                               'elapsed_seconds': time.monotonic() - started, **entry})

    def write(record, ids, cap, body):
        infer.write_once(sv._path(out, record['uid']), {
            **{k: record[k] for k in ('uid', 'source', 'group')}, 'arm': args.arm, 'run_sha256': run_hash,
            'input_tokens': len(ids), 'effective_max_new_tokens': cap, 'raw': None, 'error': None, 'generated': False,
            'finish_reason': None, 'generated_tokens': 0, 'reached_token_cap': False, 'stopped_on_eos': False, **body})
    log({'event': 'start', 'pending': len(pending)})
    if pending:
        tok = d._model_tokenizer(adapter)
        eos = tok.get_vocab().get(sv.EOS)
        if eos is None:
            raise ValueError('tokenizer missing im_end')
        jobs = []
        for record in pending:
            ids = infer.prompt_ids(tok, record)
            if not ids:
                raise ValueError('empty prompt')
            cap = min(sv.MAX_MODEL_LEN - len(ids), lo.output_cap(record, {'thinking': False}))
            if cap < 1:
                write(record, ids, None, {'error': 'INPUT_TOO_LONG'})
            else:
                jobs.append((record, ids, cap))
        if jobs:
            llm, sampling, lora_request = load(engine)
            lora = lora_request(args.arm, 1, str(adapter))
            log({'event': 'loaded', 'sampling_params': repr(sampling(**sv.params(1, 'uid', 0, eos)))})
            for chunk in _chunks(jobs, BATCHINGS[args.batching]):
                begin = time.monotonic()
                outputs = llm.generate([{'prompt_token_ids': ids} for _, ids, _ in chunk],
                                       [sampling(**sv.params(cap, record['uid'], 0, eos)) for record, _, cap in chunk],
                                       lora_request=lora, use_tqdm=False)
                if len(outputs) != len(chunk):
                    raise ValueError('generation batch cardinality mismatch')
                for (record, ids, cap), output in zip(chunk, outputs):
                    if list(output.prompt_token_ids) != ids or len(output.outputs) != 1:
                        raise ValueError('vLLM output does not match its request')
                    completion = output.outputs[0]
                    raw, info = infer.decode_output(tok, list(completion.token_ids), 0, eos, cap)
                    write(record, ids, cap, {'raw': raw, 'generated': True, 'finish_reason': completion.finish_reason, **info})
                log({'event': 'chunk', 'rooms': len(chunk), 'wall_seconds': time.monotonic() - begin})
    log({'event': 'end'})
    records = []
    for record in rows:
        value = h.strict_json(sv._path(out, record['uid']).read_text())
        if (value['uid'], value['run_sha256'], value['arm']) != (record['uid'], run_hash, args.arm):
            raise ValueError(f"receipt identity mismatch: {record['uid']}")
        records.append(value)
    (out / 'records.jsonl').write_bytes(d._jsonl(records))  # Rewritten whole by a rerun until MANIFEST.json exists.
    events = [h.strict_json(x) for x in (out / 'invocations.jsonl').read_text().splitlines()]
    spans = {}
    for event in events:
        spans[event['invocation']] = max(spans.get(event['invocation'], 0), event['elapsed_seconds'])
    result = {'status': 'INFERENCE_COMPLETE', 'pending_count': 0, 'rooms': len(records), 'run_sha256': run_hash,
              'input_too_long': sum(r['error'] == 'INPUT_TOO_LONG' for r in records),
              'cap_hits_without_eos': sum(r['reached_token_cap'] and not r['stopped_on_eos'] for r in records),
              'gpu_hours': sum(spans.values()) / 3600, 'quality_claim': False,
              'slurm_job_ids': sorted({e['slurm_job_id'] for e in events if e['slurm_job_id']})}
    h.write_json(out / 'result.json', result)
    infer.write_once(out / 'MANIFEST.json', {'files': h.artifact_manifest(out), 'expected_uids': data['expected_uids']})
    return result


# ---- scoring -------------------------------------------------------------------------------------

def gt_relations(room):
    """The deterministic relation candidates of build.extract_constraints that hold on GT: 'on' and 'against_wall'."""
    objs = [o for o in room['objects'] if head(o) not in FLAT and o['size'][2] >= 0.05]
    index = {o['id']: o for o in room['objects']}
    out = [['on', o['id'], o['parent']] for o in objs if o.get('parent')]
    if room['boundary_type'] == 'polygon':
        out += [['against_wall', o['id']] for o in objs
                if not o.get('parent') and validate.holds(['against_wall', o['id']], None, room['boundary'], index)]
    return out


def _extras(task):
    """(strict admission, GT relations held, GT relations) of one scored row; an incomplete row holds none."""
    row, room, request = task
    relations = gt_relations(room)
    if not row['complete']:
        return False, 0, len(relations)
    accepted = ch._admission((row, room, request))[0]
    layout = {o['id']: o for o in row['raw_layout']}
    boundary = h.strict_json(request['messages'][1]['content'])['boundary']
    return accepted, sum(validate.holds(c, None, boundary, layout) for c in relations), len(relations)


def finish(scored, rooms, requests, workers):
    extras = r2._map(_extras, [(s, rooms[s['uid']], requests[s['uid']]) for s in scored], workers)
    return [{**s, 'objects': len(rooms[s['uid']]['objects']), 'accepted': a, 'relations_held': x, 'relations': n}
            for s, (a, x, n) in zip(scored, extras, strict=True)]


def _rate(rows, key):
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else None


def metrics(rows):
    done = [r for r in rows if r['complete']]
    oor = [r['oor'] for r in done if r['oor'] is not None]
    big = [r for r in rows if r['objects'] >= BIG]
    relations = sum(r['relations'] for r in rows)
    return {'requests': len(rows), 'complete_rate': _rate(rows, 'complete'), 'checker': _rate(rows, 'geometry_success'),
            'pass_among_complete': _rate(done, 'geometry_success'), 'oor_mean': statistics.mean(oor) if oor else None,
            'accepted_rate': _rate(rows, 'accepted'),
            'holds_rate': sum(r['relations_held'] for r in rows) / relations if relations else None,
            'relations': relations, 'checker_21plus': _rate(big, 'geometry_success'), 'rooms_21plus': len(big)}


_SLICES = {}


def _slice(data):
    """(expected uids, rooms, placement requests) of a verified eval-layout dir, cached per process."""
    key = str(Path(data).resolve())
    if key not in _SLICES:
        manifest = e._verified(Path(data))
        rooms = h._by_uid(h._rows(Path(data), 'rooms.jsonl', manifest['files']['rooms.jsonl']))
        name = f'{ARM}/requests.jsonl'
        _SLICES[key] = (manifest['expected_uids'], rooms, h._by_uid(h._rows(Path(data), name, manifest['files'][name])))
    return _SLICES[key]


def load_run(run, slice_, cand, engine, specs, workers, batching='objects'):
    """One bound, scored inference -> (rows, records, info); cap hits without EOS fail (amendment 005)."""
    data = data_dir(run, slice_)
    expected, rooms, requests = _slice(data)
    path = run_dir(run, slice_, cand, engine, specs, batching)
    records, receipt = e._inference(path, data, expected)
    bind(receipt, specs[cand], engine)
    scored = r2._map(e._score, [(rooms[x['uid']], requests[x['uid']], x, ARM) for x in records], workers)
    scored, flipped = r2.cap_rule(scored, records)
    info = {'path': str(path.resolve()), 'engine': engine, 'run_sha256': receipt['run_sha256'],
            'manifest_sha256': h.sha256(path / 'MANIFEST.json'), 'cap_failures_of_scorable_rows': flipped,
            **({'batching': batching} if engine == 'vllm' else {})}
    return finish(scored, rooms, requests, workers), records, info


def _complete(path):
    return (Path(path) / 'MANIFEST.json').exists()


# ---- O2nt reference rows -------------------------------------------------------------------------

def o2nt_dir(run, slice_):
    return Path(run) / A21 / 'o2nt' / slice_


def o2nt_cohort(run, slice_):
    return Path(run) / ('llm-cohort-v1.json' if slice_ == 'cohort' else f'{A21}/cohorts/{slice_}.json')


def o2nt_score(args):
    run, out = Path(args.run_root), Path(args.run_root) / A21 / 'o2nt-scores'
    if os.path.lexists(out):
        raise FileExistsError(out)
    value = {'protocol': 'amendment-021-o2nt-reference-rows', 'scope': (
        'descriptive (section 1.4): decides nothing; scored as evaluate_llm scores a placement run (addendum B2 '
        'enforce_sizes, placement semantics, no cap rule); rates over every request of the slice'), 'slices': {}}
    for slice_ in O2NT:
        data, cohort_path = data_dir(run, slice_), o2nt_cohort(run, slice_)
        cohort = h.strict_json(cohort_path.read_text())
        if cohort != (L.build_cohort(data) if slice_ == 'cohort' else lo.all_rooms_cohort(data)):
            raise ValueError(f'{cohort_path} differs from its deterministic rebuild')
        records, receipt, result, cost = ev._llm(o2nt_dir(run, slice_), data, cohort, h.sha256(cohort_path))
        config = receipt['config']
        if (config.get('system'), config.get('request_arm'), config.get('thinking'),
                config.get('chat_template_kwargs')) != ('O2nt', ARM, False, {'enable_thinking': False}) or \
                config.get('reasoning_tokens') not in ('asserted_zero', 'zero_else_failure'):  # Addendum B.
            raise ValueError(f'{o2nt_dir(run, slice_)} is not an O2nt P-input run')
        violations = _violations(o2nt_dir(run, slice_))
        _, rooms, requests = _slice(data)
        variants = {}
        for variant, rows in records.items():
            enforced, counts = ev._enforced(rows, requests)
            scored = r2._map(ev._score_llm, [(rooms[r['uid']], requests[r['uid']], r, ARM) for r in enforced], args.workers)
            variants[variant] = {**metrics(finish(scored, rooms, requests, args.workers)), 'size_enforcement': counts}
        value['slices'][slice_] = {'run': {'path': str(o2nt_dir(run, slice_).resolve()), 'run_sha256': receipt['run_sha256'],
                                           'config_sha256': receipt['config_sha256']},
                                   'cohort_sha256': h.sha256(cohort_path), 'rooms': len(cohort['uids']),
                                   'reasoning_tokens': result['tokens']['reasoning_tokens'],
                                   'no_reasoning_violations': {'rooms': len(violations), 'uids': violations},
                                   'gpu_hours': (cost or {}).get('gpu_hours'), 'variants': variants}
    value['code_sha256'] = _code()
    d._publish_dir(out, {'scores.json': _json(value), 'report.md': _o2nt_report(value).encode('utf-8')})
    return {'status': 'A021_O2NT_SCORED', **{s: round(v['variants']['llm_harness']['checker'], 4)
                                              for s, v in value['slices'].items()}}


def _violations(root):
    """Addendum B: uids whose O2nt answer reasoned; such a room fails (never repaired) and is counted, not a run abort."""
    receipts = (h.strict_json(p.read_text()) for p in sorted((Path(root) / 'receipts').glob('*.json')))
    return sorted(r['uid'] for r in receipts if any(a.get('no_reasoning_violation') for a in r.get('answers', [])))


def _num(x):
    return '—' if x is None else f'{x:.3f}'


COLUMNS = (('complete_rate', '完整率'), ('checker', '检查器成功率（全部请求）'), ('pass_among_complete', '完整中通过率'),
           ('oor_mean', 'OOR'), ('accepted_rate', '严格准入'), ('holds_rate', 'GT 关系保持率'), ('checker_21plus', '21+ 物体检查器'))


def _row(name, m):
    return f'| {name} | {m["requests"]} | ' + ' | '.join(_num(m[k]) for k, _ in COLUMNS) + ' |'


def _header(first):
    return [f'| {first} | 请求数 | ' + ' | '.join(t for _, t in COLUMNS) + ' |', '|---|---:|' + '---:|' * len(COLUMNS)]


def _o2nt_report(value):
    names = {'dev-select': 'dev-select（600 间）', 'confirm': '确认切片（3,000 间）', 'cohort': '测试队列（1,454 间）'}
    lines = ['# 修订 021 §1.4：O2nt 参考行（Qwen3.8-27B，关闭思考，P 输入，零样本）', '',
             '仅作描述（“微调 vs 思考”对照），不参与任何决定。按 evaluate_llm 对 P 输入运行的评分方式：附录 B2 尺寸覆盖，'
             'placement 语义，不应用截断规则；所有比率以该切片全部请求为分母。推理 token 数在每个切片均断言为 0。', '']
    lines += _header('切片 / 变体')
    for slice_, entry in value['slices'].items():
        for variant, m in entry['variants'].items():
            lines.append(_row(f'{names[slice_]} {variant}', m))
    lines += ['', 'GT 关系保持率：GT 上成立的 on 与 against_wall 关系（build.extract_constraints 的确定性候选）在预测布局上'
              '仍成立的比例（validate.holds）；修订未给出关系集合，此为本工具的读法。']
    return '\n'.join(lines) + '\n'


# ---- engine check --------------------------------------------------------------------------------

def engines(common):
    return {'8B': 'vllm' if common else 'hf', '27B': 'vllm'}


def engine_check(args):
    run, out = Path(args.run_root), Path(args.run_root) / A21 / 'engine-check'
    if os.path.lexists(out):
        raise FileExistsError(out)
    specs, runs, raws = fixed_specs(run), {}, {}
    for cand in (R, PSFT):
        runs[cand], raws[cand] = {}, {}
        for engine in ('hf', 'vllm'):
            rows, records, info = load_run(run, 'engine', cand, engine, specs, args.workers)
            runs[cand][engine], raws[cand][engine] = {**info, **metrics(rows)}, [x['raw'] for x in records]
    gap = {c: runs[c]['hf']['checker'] - runs[c]['vllm']['checker'] for c in runs}
    same = {c: sum(a is not None and a == b for a, b in zip(raws[c]['hf'], raws[c]['vllm'], strict=True)) / len(raws[c]['hf'])
            for c in runs}
    common = common_engine(gap[R])
    value = {'protocol': 'amendment-021-engine-check', 'rule': ENGINE_RULE, 'threshold': ENGINE_GAP,
             'data_manifest_sha256': h.sha256(data_dir(run, 'engine') / 'MANIFEST.json'), 'runs': runs,
             'gap_hf_minus_vllm': gap, 'identical_raw_fraction': same, 'common_vllm': common, 'engines': engines(common),
             'code_sha256': _code()}
    d._publish_dir(out, {'engine.json': _json(value)})
    return {'status': 'A021_ENGINE_CHECKED', 'common_vllm': common, 'gap': {c: round(v, 4) for c, v in gap.items()}}


def load_engine(run):
    root = Path(run) / A21 / 'engine-check'
    d._verified_dir(root)
    value = h.strict_json((root / 'engine.json').read_text())
    if value.get('protocol') != 'amendment-021-engine-check' or value['engines'] != engines(value['common_vllm']):
        raise ValueError(f'{root} is not an amendment-021 engine check')
    return value, h.sha256(root / 'MANIFEST.json')


# ---- candidates and jobs -------------------------------------------------------------------------

def write_candidates(run, kind):
    run = Path(run)
    engine, engine_sha = load_engine(run)
    base8, specs, not_adopted = str(models_root(run) / 'Qwen3-8B'), {R: fixed_specs(run)[R]}, {}
    for name, rel, loader, path_of in (('op13', 'dev-eval-013/selection.json', op.load_selection, op.model_path),
                                       ('d18', 'dev-eval-018/selection.json', ch.load_selection, ch.model_path)):
        path = run / rel
        if not path.exists():
            raise FileNotFoundError(f'{path}: that amendment must select (adopted or not) before the amendment-021 '
                                    'candidates are fixed')
        value, _ = loader(path)
        if not value['adopted']:
            not_adopted[name] = h.sha256(path)
            continue
        hf = {'dev-select': str(path.parent / value['selected'] / f'infer-{ARM}')}
        if name == 'd18':  # 018 runs unchanged: its adopted model's HF exact test run is the run of record. 013's test
            hf['test'] = str(run / f'infer-{ARM}-{ch.TEST_TAG}')  # run is deferred to here (013 addendum A).
        specs[name] = {'size': '8B', 'base': base8, 'model': str(path_of(run, value['selected'])), 'exact_policy': True,
                       'selected': value['selected'], 'selection_sha256': h.sha256(path), 'hf': hf}
    if kind not in ARM_DIRS:
        raise ValueError('--arm-kind must be A (27B) or B (8B fallback)')
    if kind == 'B' and (run / A21 / ARM_DIRS['A'] / POLICY).exists():
        raise ValueError('Arm A is trained; Arm B is the fallback only (section 2)')
    arm = run / A21 / ARM_DIRS[kind]
    if not (arm / POLICY).is_file():
        raise FileNotFoundError(f'no policy receipt {arm / POLICY} for the amendment-021 arm')
    specs['arm'] = {'size': '27B' if kind == 'A' else '8B', 'kind': kind, 'exact_policy': True,
                    'base': str(models_root(run) / ('Qwen3.8-27B' if kind == 'A' else 'Qwen3-8B')),
                    'model': str(arm), 'policy_sha256': h.sha256(arm / POLICY), 'hf': {}}
    value = {'protocol': 'amendment-021-candidates', 'engine_check_manifest_sha256': engine_sha,
             'common_vllm': engine['common_vllm'], 'candidates': specs,
             'serving_engine': {c: engine['engines'][s['size']] for c, s in specs.items()}, 'not_adopted': not_adopted,
             'code_sha256': {Path(__file__).name: h.sha256(__file__)}}
    _write_once(run / A21 / 'candidates.json', value, 'candidate set')
    return value


def load_candidates(run):
    value = h.strict_json((Path(run) / A21 / 'candidates.json').read_text())
    engine, engine_sha = load_engine(run)
    if value.get('protocol') != 'amendment-021-candidates' or value['engine_check_manifest_sha256'] != engine_sha or any(
            value['serving_engine'][c] != engine['engines'][s['size']] for c, s in value['candidates'].items()):
        raise ValueError('candidates.json is not bound to the engine check')
    return value


def _vllm_job(run, slice_, cand, specs, batching='objects'):
    return [] if _complete(vllm_dir(run, slice_, cand, batching)) else [
        ('vllm', slice_, cand, batching, VLLM_HOURS[specs[cand]['size']][slice_])]


def _hf_jobs(run, slice_, cand, specs):
    if specs[cand]['size'] != '8B':
        raise ValueError('HF exact serves the 8B models only')
    merged = hf_dir(run, slice_, cand, specs)
    if _complete(merged):
        return []
    if os.path.lexists(merged):
        raise FileExistsError(f'refusing to reuse existing {merged}')
    if specs[cand].get('hf', {}).get(slice_):
        raise FileNotFoundError(f'{merged}: the run of record is missing')
    n, hours = HF_SHARDS[slice_]
    root = hf_root(run, slice_, cand) / ARM
    return [('hf', slice_, cand, str(k), str(n), hours, '--resume' if os.path.lexists(root / f'shard-{k:02d}') else '')
            for k in range(n)] + [('hf-merge', slice_, cand, str(n))]


def job_list(run, stage, kind=None):
    run = Path(run)
    if stage == 'o2nt':
        return [('o2nt', s, O2NT_HOURS[s]) for s in O2NT if not _complete(o2nt_dir(run, s))]
    if stage == 'engine':
        specs = fixed_specs(run)
        return [x for c in (R, PSFT) for x in _hf_jobs(run, 'engine', c, specs) + _vllm_job(run, 'engine', c, specs)]
    if stage == 'candidates':
        value = write_candidates(run, kind)
        specs, serving = specs_of(run), value['serving_engine']
        jobs = _vllm_job(run, 'dev-select', R, specs) + _vllm_job(run, 'dev-select', R, specs, 'hash')
        for c in (c for c in ORDER if c in serving):
            if serving[c] == 'vllm':
                jobs += (_vllm_job(run, 'dev-select', c, specs) if c != R else []) + _vllm_job(run, 'confirm', c, specs)
            else:
                jobs += _hf_jobs(run, 'dev-select', c, specs)
        if 'vllm' in serving.values() and serving[R] != 'vllm':
            jobs += _vllm_job(run, 'confirm', R, specs)  # The same-engine reference of the vLLM-served candidates.
        return jobs
    if stage == 'confirm':  # Only an HF-served dev winner: its confirm runs and R's on HF.
        value = h.strict_json((run / A21 / 'dev-selection.json').read_text())
        winner = value['selected']
        if (run / A21 / 'final-selection.json').exists() or winner == R or value['serving_engine'][winner] != 'hf':
            raise ValueError('no HF confirm runs are needed (R won on dev, a vLLM winner, or already final)')
        specs = specs_of(run)
        return [x for c in (winner, R) for x in _hf_jobs(run, 'confirm', c, specs)]
    if stage == 'test':
        final = load_final(run)
        if final['delivered'] == R:
            raise ValueError('R stays delivered: nothing runs on test (its HF test rows stand)')
        specs, winner, engine = specs_of(run), final['delivered'], final['engine']
        if engine == 'vllm':
            return _vllm_job(run, 'test', winner, specs) + _vllm_job(run, 'test', R, specs)
        return _hf_jobs(run, 'test', winner, specs)
    raise ValueError(f'unknown stage {stage}')


def ceiling(jobs):
    gpus = {'vllm': lambda j: j[4], 'hf': lambda j: j[5], 'o2nt': lambda j: O2NT_GPUS * j[2], 'hf-merge': lambda j: 0}
    return sum(gpus[j[0]](j) for j in jobs)


def argv(run, kind, slice_, *rest):
    """The python argv (after -m) of one sbatch job."""
    run = Path(run)
    data, specs = str(data_dir(run, slice_)), specs_of(run)
    if kind == 'o2nt':
        if slice_ not in O2NT:
            raise ValueError(slice_)
        return ['fastfill.comparison.llm_open', 'run', '--system', 'O2nt', '--arm', ARM, '--data', data,
                '--cohort', str(o2nt_cohort(run, slice_)), '--model-root', str(models_root(run)),
                '--out', str(o2nt_dir(run, slice_)), *([] if slice_ == 'cohort' else ['--all-rooms'])]
    if kind == 'plan':
        return ['fastfill.comparison.infer_shard', 'plan', '--data', data, '--base', specs[R]['base'],
                '--out', str(plan_dir(run, slice_))]
    cand = rest[0]
    spec = specs[cand]
    if kind == 'vllm':
        return ['fastfill.comparison.a021_select', 'greedy', '--data', data, '--base', spec['base'], *model_args(spec),
                '--arm', ARM, '--out', str(vllm_dir(run, slice_, cand, rest[1])), '--batching', rest[1]]
    if spec['size'] != '8B' or spec.get('hf', {}).get(slice_):
        raise ValueError(f'no new HF run of {cand} on {slice_}')
    root, n = hf_root(run, slice_, cand), int(rest[-1])
    if kind == 'hf':
        k = int(rest[1])
        return ['fastfill.comparison.infer_shard', 'shard', '--data', data, '--base', spec['base'], '--arm', ARM,
                *model_args(spec), '--plan', str(plan_dir(run, slice_)), '--shard-index', str(k), '--shard-count',
                str(n), '--out', str(root / ARM / f'shard-{k:02d}')]
    if kind == 'hf-merge':
        return ['fastfill.comparison.infer_shard', 'merge', '--data', data, '--plan', str(plan_dir(run, slice_)),
                '--arm', ARM, '--out', str(root / f'infer-{ARM}'),
                '--shards', *(str(root / ARM / f'shard-{k:02d}') for k in range(n))]
    raise ValueError(f'unknown job kind {kind}')


# ---- selection -----------------------------------------------------------------------------------

def guards(m, ref):
    """Each guard vs R on the same engine; nothing to measure in both (e.g. no 21+ room) passes, in one only fails."""
    out = {}
    for key, sign, slack in GUARDS:
        a, b = m[key], ref[key]
        out[key] = (a is None and b is None) or (a is not None and b is not None and sign * (a - b) >= -slack - EPS)
    return out


def common_engine(gap):
    return abs(gap) <= ENGINE_GAP + EPS


def choose(dev, refs, sizes, serving, spread):
    """Section 4 on dev metrics {name: m} (R included, on its own engine), refs {engine: R metrics on it}."""
    band = max(TIE_MIN, spread)
    checks = {c: guards(dev[c], refs[serving[c]]) for c in dev if c != R}
    eligible = [R] + [c for c in ORDER if c in checks and all(checks[c].values())]
    best = max(dev[c]['checker'] for c in eligible)
    tied = [c for c in eligible if dev[c]['checker'] >= best - band - EPS]
    winner = min(tied, key=lambda c: (SIZES.index(sizes[c]), c != R, -dev[c]['checker'], ORDER.index(c)))
    return {'selected': winner, 'eligible': eligible, 'tied': tied, 'best_checker': best, 'tie_band': band,
            'guards': checks}


def confirmation(winner_rows, r_rows):
    contrast = r2.contrasts(winner_rows, r_rows, ('geometry_success',), REPEATS, r2.SEED)['geometry_success']
    return {**contrast, 'confirmed': contrast['ci95'] is not None and contrast['ci95'][0] > 0}


def load_final(run):
    path = Path(run) / A21 / 'final-selection.json'
    value = h.strict_json(path.read_text())
    if value.get('protocol') != 'amendment-021-final-selection' or value.get('rule') != RULE:
        raise ValueError(f'{path} is not an amendment-021 final selection')
    return value


def select(args):
    run = Path(args.run_root)
    root = run / A21
    cands, specs = load_candidates(run), specs_of(run)
    serving, names = cands['serving_engine'], [c for c in ORDER if c in cands['serving_engine']]
    sizes = {c: specs[c]['size'] for c in names}
    dev, info, refs = {}, {}, {}
    for engine in sorted(set(serving.values()) | {'vllm'}):
        rows, _, info[f'{R}@{engine}'] = load_run(run, 'dev-select', R, engine, specs, args.workers)
        refs[engine] = metrics(rows)
    hash_rows, _, info[f'{R}@vllm-hash'] = load_run(run, 'dev-select', R, 'vllm', specs, args.workers, 'hash')
    hash_checker = metrics(hash_rows)['checker']
    spread = abs(refs['vllm']['checker'] - hash_checker)
    for c in names:
        if c == R:
            dev[c] = refs[serving[R]]
            continue
        rows, _, info[c] = load_run(run, 'dev-select', c, serving[c], specs, args.workers)
        dev[c] = metrics(rows)
    choice = choose(dev, refs, sizes, serving, spread)
    winner = choice['selected']
    engine = serving[winner]
    missing = [c for c in (winner, R) if winner != R and not _complete(run_dir(run, 'confirm', c, engine, specs))]
    if missing and engine == 'vllm':
        raise FileNotFoundError(f'vLLM confirm runs missing for {missing} (the candidates stage submits them)')
    dev_value = {'protocol': 'amendment-021-dev-selection', 'rule': RULE, 'candidates_sha256': h.sha256(root / 'candidates.json'),
                 'serving_engine': serving, 'sizes': sizes, 'dev': dev, 'reference_by_engine': refs, 'runs': dict(info),
                 'spread': {'r_vllm_objects': refs['vllm']['checker'], 'r_vllm_hash': hash_checker, 'abs': spread},
                 **choice, 'code_sha256': _code()}
    _write_once(root / 'dev-selection.json', dev_value, 'dev selection')
    if missing:  # An HF-served winner: submit_021_eval.sh CODE_ROOT select runs them, then this step again.
        return {'status': 'A021_NEEDS_HF_CONFIRM', 'selected': winner, 'missing': missing}
    confirm_value, rows = None, {}
    if winner != R:
        for c in (winner, R):
            rows[c], _, info[f'{c}@confirm'] = load_run(run, 'confirm', c, engine, specs, args.workers)
        confirm_value = {'engine': engine, 'winner': metrics(rows[winner]), 'r': metrics(rows[R]),
                         'contrast_winner_minus_r': confirmation(rows[winner], rows[R])}
    descriptive = {}
    for c in names:  # Every candidate whose confirm run on its serving engine exists (all of them on a common vLLM).
        if c not in rows and _complete(run_dir(run, 'confirm', c, serving[c], specs)):
            descriptive[c] = metrics(load_run(run, 'confirm', c, serving[c], specs, args.workers)[0])
    delivered = winner if confirm_value and confirm_value['contrast_winner_minus_r']['confirmed'] else R
    value = {'protocol': 'amendment-021-final-selection', 'rule': RULE,
             'dev_selection_sha256': h.sha256(root / 'dev-selection.json'), 'dev_winner': winner,
             'confirmation': confirm_value, 'confirm_descriptive': descriptive, 'delivered': delivered,
             'engine': serving[delivered], 'size': sizes[delivered], 'model': specs[delivered],
             'runs': info, 'code_sha256': _code()}
    report = root / 'final-report.md'
    if not report.exists():
        from fastfill.comparison.ablation_016 import _write_new
        _write_new(report, _report(value, dev_value, cands).encode('utf-8'))
    _write_once(root / 'final-selection.json', value, 'final selection')
    return {'status': 'A021_FINAL_SELECTED', 'dev_winner': winner, 'delivered': delivered, 'engine': value['engine']}


def _label(c, cands):
    spec = cands['candidates'][c]
    extra = (f"（{spec['selected']}）" if 'selected' in spec else
             f"（{'A，27B' if spec.get('kind') == 'A' else 'B，8B 后备'}）" if c == 'arm' else '')
    return LABELS[c] + extra


def _report(value, dev_value, cands):
    serving = dev_value['serving_engine']
    eng = {'vllm': 'vLLM 贪心（BF16 LoRA）', 'hf': 'HF 精确'}
    rooms = dev_value['dev'][R]['requests']
    lines = ['# 修订 021 §4：交付模型的最终选择（跨修订 013、018、021）', '',
             f"引擎（§3）：{'共同使用 vLLM 贪心' if cands['common_vllm'] else '各候选用其服务引擎（8B 用 HF 精确，27B 用 vLLM）'}。",
             '', f'## dev-select（{rooms} 间）各候选（各自服务引擎）', ''] + _header('候选（引擎）')
    for c in (c for c in ORDER if c in dev_value['dev']):
        lines.append(_row(f'{_label(c, cands)}（{eng[serving[c]]}）', dev_value['dev'][c]))
    for engine, m in dev_value['reference_by_engine'].items():
        if engine != serving[R]:
            lines.append(_row(f'R 参考（{eng[engine]}，同引擎护栏用）', m))
    lines += ['', f"R 的 vLLM 两次运行（不同批组成）检查器差 {dev_value['spread']['abs']:.4f}，平局带 "
              f"max(0.01, 差) = {dev_value['tie_band']:.4f}。", '']
    for c, checks in dev_value['guards'].items():
        bad = '、'.join(k for k, ok in checks.items() if not ok)
        lines.append(f"- {_label(c, cands)} 护栏（相对同引擎 R）：{'未满足 ' + bad if bad else '全部满足'}")
    lines += ['', f"dev 胜者：{_label(value['dev_winner'], cands)}（合格：{'、'.join(_label(c, cands) for c in dev_value['eligible'])}；"
              f"平局带内：{'、'.join(_label(c, cands) for c in dev_value['tied'])}）。", '']
    if value['confirmation']:
        x = value['confirmation']['contrast_winner_minus_r']
        ci = '未定义' if x['ci95'] is None else f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
        lines += [f"## 确认切片（{value['confirmation']['r']['requests']} 间，胜者的引擎）", ''] + _header('模型')
        lines += [_row(_label(value['dev_winner'], cands), value['confirmation']['winner']), _row('R', value['confirmation']['r'])]
        lines += ['', f"胜者 − R 检查器成功率：{x['estimate']:+.4f}，房屋连通组配对 bootstrap 95% CI {ci}（2,000 次，seed 42）；"
                  f"{'下界 > 0，确认通过' if x['confirmed'] else '下界不大于 0，确认失败'}。", '']
    if value['confirm_descriptive']:
        lines += ['确认切片上其余候选（描述性，同其服务引擎）：', ''] + _header('候选')
        lines += [_row(_label(c, cands), m) for c, m in value['confirm_descriptive'].items()] + ['']
    lines += [f"## 决定：交付 {_label(value['delivered'], cands)}（{eng[value['engine']]}）", '',
              ('R 在 dev 上即为胜者，无需确认。' if value['dev_winner'] == R else
               '确认通过：测试集上只运行该模型与同引擎的 R 各一次。' if value['delivered'] != R else
               '确认失败：R 保持交付，测试集不运行新模型。'), '',
              '规则（修订 021 §4，预注册）：' + RULE_ZH, '',
              'GT 关系保持率：GT 上成立的 on 与 against_wall 关系（build.extract_constraints 的确定性候选，排除扁平与低于 5 cm '
              '的物体）在预测布局上仍成立的比例（validate.holds）；修订只写了 validate.holds，关系集合为本工具的读法。']
    return '\n'.join(lines) + '\n'


# ---- test rows -----------------------------------------------------------------------------------

def test_geometry(args):
    run = Path(args.run_root)
    final = load_final(run)
    winner, engine = final['delivered'], final['engine']
    if winner == R:
        raise ValueError('R stays delivered: nothing ran on test')
    specs = specs_of(run)
    outs = {c: run / A21 / f'test-geometry-{c}' for c in (winner, R)}
    taken = [str(p) for p in outs.values() if os.path.lexists(p)]
    if taken:
        raise FileExistsError(f'refusing existing outputs: {taken}')
    data = data_dir(run, 'test')
    expected, rooms, requests = _slice(data)
    rows, payloads, infos = {}, {}, {}
    for c in (winner, R):
        path = run_dir(run, 'test', c, engine, specs)
        records, receipt = e._inference(path, data, expected)
        bind(receipt, specs[c], engine)
        plain = r2._map(e._score, [(rooms[x['uid']], requests[x['uid']], x, ARM) for x in records], args.workers)
        payloads[c] = [{**s, **{k: x[k] for k in ev.CAP_FLAGS}} for s, x in zip(plain, records, strict=True)]
        rows[c] = finish(r2.cap_rule(plain, records)[0], rooms, requests, args.workers)
        infos[c] = {'path': str(path.resolve()), 'run_sha256': receipt['run_sha256'], 'engine': engine,
                    'manifest_sha256': h.sha256(path / 'MANIFEST.json')}
    for c in (R, winner):  # The winner last: its summary carries the contrast.
        summary = {'status': 'GEOMETRY_COMPLETE', 'protocol': 'amendment-021-test-rows', 'candidate': c,
                   'final_selection_sha256': h.sha256(run / A21 / 'final-selection.json'), 'inference': infos[c],
                   'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'), 'arms': {ARM: g.summarize(payloads[c])},
                   'metrics': metrics(rows[c]),
                   'scope': ('placement-geometry.jsonl = evaluate_pair rows (no cap rule) plus the cap flags; metrics '
                             'apply the amendment-005 cap rule; every test request is a denominator'),
                   **({'contrast_winner_minus_r': r2.contrasts(rows[winner], rows[R], ('complete', 'geometry_success',
                                                                                       'accepted'), REPEATS, r2.SEED)}
                      if c == winner else {}), 'code_sha256': _code()}
        d._publish_dir(outs[c], {'placement-geometry.jsonl': d._jsonl(payloads[c]), 'summary.json': _json(summary)})
    return {'status': 'A021_TEST_ROWS_COMPLETE', **{c: round(metrics(rows[c])['checker'], 4) for c in rows}}


# ---- CLI -----------------------------------------------------------------------------------------

def parse_args(argv_=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('confirm')
    for flag in ('data', 'dev', 'run-root'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--sft-runs', type=Path, nargs=2, required=True, metavar=('SFT_JOINT', 'SFT_PLACEMENT'))
    step = sub.add_parser('greedy')
    for flag in ('data', 'base', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    for flag in ('adapter', 'training-run', 'model'):
        step.add_argument('--' + flag, type=Path)
    step.add_argument('--exact-policy', action='store_true')
    step.add_argument('--arm', choices=(ARM,), required=True)
    step.add_argument('--batching', choices=tuple(BATCHINGS), default='objects')
    for name in ('o2nt-score', 'engine-check', 'select', 'test-geometry'):
        step = sub.add_parser(name)
        step.add_argument('--run-root', type=Path, required=True)
        step.add_argument('--workers', type=int, default=1)
    step = sub.add_parser('jobs')
    step.add_argument('stage', choices=('o2nt', 'engine', 'candidates', 'confirm', 'test'))
    step.add_argument('--run-root', type=Path, required=True)
    step.add_argument('--arm-kind', choices=tuple(ARM_DIRS))
    step = sub.add_parser('argv')
    step.add_argument('items', nargs='+')
    step.add_argument('--run-root', type=Path, required=True)
    return parser.parse_args(argv_)


def main(argv_=None, load=sv.load_vllm):
    args = parse_args(argv_)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    if args.command == 'jobs':
        if (args.stage == 'candidates') != (args.arm_kind is not None):
            raise SystemExit('candidates (only) needs --arm-kind')
        jobs = job_list(args.run_root, args.stage, args.arm_kind)
        print('\n'.join([' '.join(map(str, j)).rstrip() for j in jobs] + [f'ceiling {ceiling(jobs)}']))
        return
    if args.command == 'argv':
        print('\n'.join(argv(args.run_root, *args.items)))
        return
    if args.command == 'greedy':
        result = greedy(args, load)
    else:
        result = {'confirm': confirm, 'o2nt-score': o2nt_score, 'engine-check': engine_check, 'select': select,
                  'test-geometry': test_geometry}[args.command](args)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
