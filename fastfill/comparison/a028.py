"""Amendment 028: the 27B joint arm J27 (Arm A's model, recipe and gates on J's own SFT rows) and the pre-registered
27B-J vs 27B-P comparison on all test rooms.

python -m fastfill.comparison.a028 STEP ...   (sbatch: isambard_021_train.sbatch rows|smoke|smoke-vllm|train J27 and
isambard_021_eval.sbatch a028 greedy|compare; chain: submit_028.sh). Training itself is a021_train --arm J27.

rows     --data RUN/main-data-v1 --sft-runs RUN/sft-joint-001 RUN/sft-placement-001 --base27 QWEN27 --arm-a RUN/a021/27b
         --dev DEV... --test TEST... --out RUN/a028/rows
    J's SFT rows exactly as the 8B J SFT trained them: main-data-v1 joint/train.jsonl over the SFT common length
    admission (dpo_stages._admitted, the matched train pool); prompt = J's system + user turns (infer.validate_request
    'joint': no movable sizes), chosen = rejected = J's target (h.parse_prediction 'joint': size, pos, yaw, parent).
    Order sha256('a021-27b:' + uid), Arm A's rule and salt. A UID or house group shared with a dev/test eval dir raises.
    27B tokenizer: a021_train.lengths per row (train_dpo's render: counts, prefix, fit <= 8192), then train_dpo.check_render
    on the first 40,000 fitting rooms (021: EOS = <|im_end|>; the training prompt render = infer.prompt_ids of the joint
    request with thinking off, token for token, and a prefix of the sequence). Published in a021_train's pool layout
    (sft-rows.jsonl, ledger.jsonl, summary.json, MANIFEST.json), so a021_train smoke / smoke-vllm / train --arm J27 read
    it unchanged (N = the smoke's min(40,000, rooms fitting 10 h), a prefix of this order). summary.selection: the first
    40,000 fitting UIDs (sha256) and their overlap with Arm A's trained rooms (RUN/a021/27b/rows).
greedy   --run-root RUN --slice dev-select|test   (vLLM env, 1 GPU)
    a021_select.greedy unchanged (Arm A's test engine: sample_vllm ENGINE + MULTIMODAL_ENGINE, BF16 LoRA, k = 0,
    objects batching) on --model RUN/a028/27b-joint --exact-policy --arm joint and the slice's joint requests
    -> RUN/a028/vllm/SLICE/j27 (a rerun resumes from its receipts).
compare  --run-root RUN [--workers N] [--repeats 2000]   (CPU)
    First, unless they exist, the geometry rows of both J27 runs: evaluate_pair rows (joint semantics) plus the cap
    flags -> RUN/a028/{dev,test}-geometry-j27/joint-geometry.jsonl and summary.json (geometry.summarize,
    a021_select.metrics with the amendment-005 cap rule), MANIFEST-sealed as a021 test-geometry-arm. Then 27B-J
    (test-geometry-j27) vs 27B-P (RUN/a021/test-geometry-arm; same engine, decoding, prompt and batching, asserted from
    both run receipts) on all test rooms, cap rule per row. Pre-registered: (a) all-request geometry success, (b) size
    log MAE on common-complete rooms (P's sizes = the prior), J - P, paired house-cluster bootstrap 2,000 draws seed 42
    (report_v2.contrasts = geometry.paired_intervals' design), Holm over (a), (b). Secondary (report_v2.jp_scope as
    amendment 005 builds it): footprint bias B, a' (x/y rescaled by exp(-B/2), re-scored), S1 (output_repair.run at
    the amendment-012 settings on both rows -> RUN/a028/s1/*, reused if complete) per scope: all, SceneSmith, rooms with
    / without a prior-fallback object (a P request size = the constant fallback), object-count bins, room families;
    object-level size errors of fallback vs other objects. -> RUN/a028/compare: results.json, report.md (Chinese),
    MANIFEST.json. Descriptive for delivery: the frozen delivery does not change.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

from fastfill.comparison import a021_select as A
from fastfill.comparison import a021_train as t
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import room_families as rf
from fastfill.comparison import train_pair_helpers as h

A28, ARM, REPEATS = 'a028', 'J27', 2000
SLICES = {'dev-select': 'dev', 'test': 'test'}
P_ROWS = ('a021/test-geometry-arm', 'placement-geometry.jsonl')  # 27B-P: Arm A's test rows of record.
SAME = ('engine', 'decoding', 'prompt')  # Run-receipt keys 27B-J and 27B-P must share (plus the batching config).
FIELDS = ('geometry_success', 'size_log_mae')  # (a), (b): Holm over these two.


def _code():
    return {p.name: h.sha256(p) for p in map(Path, (__file__, t.__file__, A.__file__, e.__file__, g.__file__,
                                                     r2.__file__))}


def policy_dir(run):
    return Path(run) / A28 / t.ARMS[ARM]['dir']


def vllm_dir(run, slice_):
    return Path(run) / A28 / 'vllm' / slice_ / 'j27'


def geometry_dir(run, slice_):
    return Path(run) / A28 / f'{SLICES[slice_]}-geometry-j27'


def _uids(root, name):
    return [r['uid'] for r in h._rows(root, name, d._verified_dir(root)[name])]


# ---- rows ----------------------------------------------------------------------------------------

def rows(args):
    from fastfill.scene import prompt_text
    from fastfill.v1_1.train_dpo import check_render
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    main = Path(args.data)
    pairs, data = h.load_pair(main)
    if data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('J27 trains on the qualified main data')
    admitted, _, lengths = d._admitted(args.sft_runs, data, 'train')  # The matched pool the 8B J SFT trained on.
    joint = h._by_uid(pairs['train']['joint'])
    held_uids, held_groups, held = op._held_out(args.dev + args.test)
    leaks = [u for u in admitted if u in held_uids or joint[u]['group'] in held_groups]
    if leaks:
        raise ValueError(f'{len(leaks)} J rows share a UID or house group with dev/test: {leaks[:3]}')
    values = []
    for uid in sorted(admitted, key=lambda u: (t.rank(u), u)):
        row = joint[uid]
        meta = {k: row[k] for k in ('uid', 'source', 'group')}
        infer.validate_request({**meta, 'messages': row['messages'][:2]}, 'joint')  # No movable sizes in the request.
        if not h.parse_prediction(row['messages'][2]['content'], row, 'joint')['valid']:  # size, pos, yaw, parent out.
            raise ValueError(f'not a J target: {uid}')
        values.append({**meta, 'prompt': row['messages'][:2], 'chosen': row['messages'][2]['content'],
                       'rejected': row['messages'][2]['content']})
    base = t.base_receipt(args.base27)
    if t._model_type(base) != t.MODEL_TYPE:
        raise ValueError('--base27 is not the pinned 27B')
    tok = d._model_tokenizer(base['path'])
    tokens = t.lengths(tok, values)
    fitting = [r for r, x in zip(values, tokens) if x['fits']][:t.N_MAX]
    if not fitting:
        raise ValueError('no J row renders as a token prefix within 8192 tokens')
    for start in range(0, len(fitting), 2000):  # Amendment 021's render assertions, on every row J27 can train on.
        part = fitting[start:start + 2000]
        prompts = [prompt_text(tok, r['prompt']) for r in part]
        full = tok([p + r['chosen'] + tok.eos_token for p, r in zip(prompts, part)], add_special_tokens=False)['input_ids']
        check_render(tok, part, tok(prompts, add_special_tokens=False)['input_ids'], full, full)
    arm_a = _uids(Path(args.arm_a) / 'rows', 'placement.jsonl')
    selected = [r['uid'] for r in fitting]
    summary = {
        'status': t.POOL_STATUS,  # a021_train.load_pool reads this layout unchanged.
        'protocol': 'amendment-028-j27-rows', 'order': f"sha256('{t.SALT}' + uid)", 'max_len': t.MAX_LEN,
        'format': "J's SFT rows (main-data-v1 joint/train.jsonl): request without movable sizes; target size, pos, yaw, parent",
        'pool': {'rooms': len(values), 'rule': 'SFT common length admission (dpo_stages._admitted, train)',
                 'fits': {'27b': sum(x['fits'] for x in tokens)},
                 'prefix_failures': {'27b': sum(not x['prefix_ok'] for x in tokens)}},
        'selection': {'n_max': t.N_MAX, 'rooms': len(selected), 'uids_sha256': d._digest(selected),
                      'overlap_with_arm_a': len(set(selected) & set(arm_a)), 'arm_a_rooms': len(arm_a),
                      'arm_a_rows_manifest_sha256': h.sha256(Path(args.arm_a) / 'rows/MANIFEST.json'),
                      'render_check': 'train_dpo.check_render on every selected row'},
        'held_out': held, 'leaks': 0,
        'tokenizers': {'27b': {'base_path': base['path'], 'base_files_sha256': infer._hash(base['files']),
                               'eos_token': tok.eos_token}},
        'inputs': {'main_manifest_sha256': data['manifest_sha256'], 'sft_lengths_sha256': lengths},
        'code_sha256': _code()}
    ledger = [{'uid': r['uid'], 'source': r['source'], 'group': r['group'], 'origin': 'joint', 'pool': True,
               'tokens': {'27b': x}} for r, x in zip(values, tokens)]
    d._publish_dir(out, {'sft-rows.jsonl': d._jsonl(values), 'ledger.jsonl': d._jsonl(ledger),
                         'summary.json': A._json(summary)})
    return {'status': 'A028_ROWS_COMPLETE', 'pool': len(values), 'selected': len(selected),
            'overlap_with_arm_a': summary['selection']['overlap_with_arm_a']}


# ---- greedy and geometry rows --------------------------------------------------------------------

def greedy(args, load=None):
    from fastfill.comparison import sample_vllm as sv
    run = Path(args.run_root)
    return A.greedy(SimpleNamespace(data=A.data_dir(run, args.slice), base=A.models_root(run) / 'Qwen3.8-27B',
                                    model=policy_dir(run), exact_policy=True, adapter=None, training_run=None,
                                    arm='joint', out=vllm_dir(run, args.slice), batching='objects'), load or sv.load_vllm)


def _data(data, arm):
    manifest = e._verified(Path(data))
    rooms = h._by_uid(h._rows(Path(data), 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    name = f'{arm}/requests.jsonl'
    return manifest['expected_uids'], rooms, h._by_uid(h._rows(Path(data), name, manifest['files'][name]))


def score(run, slice_, workers):
    """J27's evaluate_pair rows (joint) + cap flags and its summary, as a021_select.test_geometry writes 27B-P's."""
    run, out = Path(run), geometry_dir(run, slice_)
    if os.path.lexists(out):
        raise FileExistsError(out)
    data, path = A.data_dir(run, slice_), vllm_dir(run, slice_)
    expected, rooms, requests = _data(data, 'joint')
    records, receipt = e._inference(path, data, expected)
    config = receipt['config']
    if (config.get('arm'), config.get('model'), config.get('lora')) != ('joint', str(policy_dir(run).resolve()), A.VLLM_LORA):
        raise ValueError(f'{path} is not the J27 policy on vLLM')
    plain = r2._map(e._score, [(rooms[x['uid']], requests[x['uid']], x, 'joint') for x in records], workers)
    payload = [{**s, **{k: x[k] for k in ev.CAP_FLAGS}} for s, x in zip(plain, records, strict=True)]
    metrics = A.metrics(A.finish(r2.cap_rule(plain, records)[0], rooms, requests, workers))
    summary = {'status': 'GEOMETRY_COMPLETE', 'protocol': 'amendment-028-j27-rows', 'candidate': 'j27', 'slice': slice_,
               'inference': {'path': str(path.resolve()), 'run_sha256': receipt['run_sha256'], 'engine': 'vllm',
                             'manifest_sha256': h.sha256(path / 'MANIFEST.json')},
               'policy_sha256': h.sha256(policy_dir(run) / t.POLICY), 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
               'arms': {'joint': g.summarize(payload)}, 'metrics': metrics,
               'scope': ('joint-geometry.jsonl = evaluate_pair rows (joint, no cap rule) plus the cap flags; metrics apply '
                         'the amendment-005 cap rule; every request is a denominator'), 'code_sha256': _code()}
    d._publish_dir(out, {'joint-geometry.jsonl': d._jsonl(payload), 'summary.json': A._json(summary)})
    return metrics


# ---- compare -------------------------------------------------------------------------------------

def _geometry(root, name, expected):
    """Verified rows in test order and their run receipt (bound by the summary's run_sha256); cap rule per row."""
    rows_ = h._rows(root, name, d._verified_dir(root)[name])
    if [r['uid'] for r in rows_] != expected:
        raise ValueError(f'{root} does not cover the test rooms in order')
    inference = h.strict_json((root / 'summary.json').read_text())['inference']
    receipt = h.strict_json((Path(inference['path']) / 'run_receipt.json').read_text())
    if receipt['run_sha256'] != inference['run_sha256']:
        raise ValueError(f'{root} is not bound to its inference run')
    return r2.cap_rule(rows_, rows_)[0], receipt


def _s1(run, data, root, name, arm, workers):
    """output_repair.run (amendment-012 settings) on one geometry file, reused once published: {uid: s1}."""
    from fastfill.comparison import output_repair as rep
    out = Path(run) / A28 / 's1' / root.name
    if not (out / 'MANIFEST.json').exists():
        rep.run(data, root / name, arm, out, workers)
    stem = Path(name).stem + '-s1.jsonl'
    return {r['uid']: r['s1'] for r in h._rows(out, stem, e._verified(out)['files'][stem])}, h.sha256(out / 'MANIFEST.json')


def _scope(J, P, repeats):
    return {**r2._design(J), **r2.jp_scope(J, P, repeats), 'J_s1': r2._rate(J, 'success_s1'),
            'P_s1': r2._rate(P, 'success_s1'), 's1': r2.contrasts(J, P, ('success_s1',), repeats)['success_s1']}


def compare(args):
    run, repeats = Path(args.run_root), args.repeats
    out = run / A28 / 'compare'
    if os.path.lexists(out):
        raise FileExistsError(out)
    for slice_ in SLICES:
        if not (geometry_dir(run, slice_) / 'MANIFEST.json').exists():
            score(run, slice_, args.workers)
    data = A.data_dir(run, 'test')
    expected, rooms, joint_requests = _data(data, 'joint')
    requests = {'joint': joint_requests, 'placement': _data(data, 'placement')[2]}
    sources = {'joint': (geometry_dir(run, 'test'), 'joint-geometry.jsonl'), 'placement': (run / P_ROWS[0], P_ROWS[1])}
    rows_, receipts, s1, s1_sha = {}, {}, {}, {}
    for arm, (root, name) in sources.items():
        rows_[arm], receipts[arm] = _geometry(root, name, expected)
        s1[arm], s1_sha[arm] = _s1(run, data, root, name, arm, args.workers)
    batching = [{k: r['config'][k] for k in ('batching', 'chunk_cap_tokens', 'lora')} for r in receipts.values()]
    if any(receipts['joint'][k] != receipts['placement'][k] for k in SAME) or batching[0] != batching[1]:
        raise ValueError('27B-J and 27B-P did not run on the same engine, decoding, prompt rule and batching')
    for arm in rows_:  # Amendment 005 as report_v2.build: B on J/P common-complete rooms, a' at exp(-B/2).
        rows_[arm] = [r2._with_bias(r, rooms[r['uid']]) for r in rows_[arm]]
    both = [a['complete'] and b['complete'] for a, b in zip(rows_['joint'], rows_['placement'])]
    bias = {arm: statistics.mean(r['bias_fp'] for r, c in zip(v, both) if c) if any(both) else None
            for arm, v in rows_.items()}
    factor = {arm: None if b is None else math.exp(-b / 2) for arm, b in bias.items()}
    for arm, values in rows_.items():
        derived = r2._map(r2._derive, [(r, rooms[r['uid']], requests[arm][r['uid']], factor[arm]) for r in values],
                          args.workers)
        rows_[arm] = [{**r, **x, 'success_s1': r['error'] != r2.CAP and bool(s1[arm][r['uid']]['geometry_success_s1'])}
                      for r, x in zip(values, derived, strict=True)]
    levels = {u: ''.join('G' if o['size'] == r2.GLOBAL_FALLBACK else 'N'
                         for o in h.strict_json(requests['placement'][u]['messages'][1]['content'])['objects'])
              for u in expected}
    family = {u: rf.family(rooms[u].get('room_type')) for u in expected}
    named = {'all': lambda r: True, 'source/SceneSmith': lambda r: r['source'] == 'SceneSmith',
             'prior_fallback_rooms': lambda r: 'G' in levels[r['uid']],
             'no_prior_fallback_rooms': lambda r: 'G' not in levels[r['uid']],
             **{f'bin/{b}': (lambda r, b=b: A.bin_of(r['objects']) == b) for b in A.BINS},
             **{f'family/{f}': (lambda r, f=f: family[r['uid']] == f) for f in sorted(set(family.values()))}}
    J, P = rows_['joint'], rows_['placement']
    scopes = {}
    for name, keep in named.items():
        mask = [keep(r) for r in J]
        if any(mask):
            scopes[name] = _scope([r for r, k in zip(J, mask) if k], [r for r, k in zip(P, mask) if k], repeats)
    top = scopes['all']
    holm = r2.holm([top[f]['p'] for f in FIELDS])
    level = {FIELDS[0]: (top['J_a'], top['P_a']),  # (b)'s levels: each arm's mean on common-complete rooms.
             FIELDS[1]: (top['size_log_mae_J']['estimate'], top['size_log_mae_P']['estimate'])}
    trained = policy_dir(run) / 'rows'
    trained_uids = _uids(trained, t.ARMS[ARM]['rows'])
    policy = h.strict_json((policy_dir(run) / t.POLICY).read_text())
    result = {
        'status': 'A028_COMPARE_COMPLETE', 'amendment': '028', 'reading': 'descriptive for delivery; the frozen delivery does not change',
        'primary': {f: {**top[f], 'J': level[f][0], 'P': level[f][1], 'p_holm': ph} for f, ph in zip(FIELDS, holm)},
        'bias_B': {'J': top['bias_fp_J'], 'P': top['bias_fp_P']}, 'scale_factor': factor,
        'a_prime': {'J': top['J_a1'], 'P': top['P_a1'], 'delta': top['debiased']},
        's1': {'J': top['J_s1'], 'P': top['P_s1'], 'delta': top['s1']},
        'scopes': scopes, 'object_levels': r2._object_levels(J, P, rooms, levels),
        'training': {'n': policy['training_data']['count'], 'uids_sha256': d._digest(trained_uids),
                     'overlap_with_arm_a': len(set(trained_uids) & set(_uids(run / 'a021/27b/rows', 'placement.jsonl'))),
                     'policy_sha256': h.sha256(policy_dir(run) / t.POLICY)},
        'inputs': {arm: {'geometry_manifest_sha256': h.sha256(root / 'MANIFEST.json'), 'run_sha256': receipts[arm]['run_sha256'],
                         's1_manifest_sha256': s1_sha[arm], 'versions': receipts[arm].get('versions')}
                   for arm, (root, _) in sources.items()},
        'dev_select_j27': h.strict_json((geometry_dir(run, 'dev-select') / 'summary.json').read_text())['metrics'],
        'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'), 'repeats': repeats, 'seed': r2.SEED,
        'code_sha256': _code()}
    d._publish_dir(out, {'results.json': A._json(result), 'report.md': _report(result).encode('utf-8')})
    return {'status': result['status'], **{f: round(v['estimate'], 4) for f, v in result['primary'].items()}}


def _n(x, signed=False):
    return '—' if x is None else f'{x:+.4f}' if signed else f'{x:.4f}'


def _ci(c):
    return '—' if c.get('ci95') is None else f'[{c["ci95"][0]:+.4f}, {c["ci95"][1]:+.4f}]'


def _report(result):
    a, b = (result['primary'][f] for f in FIELDS)
    lines = ['# 修订 028：27B 联合预测（J）vs 27B 查表摆放（P），全部测试房间', '',
             '同一底座（Qwen3.8-27B）、同一训练配方与选房规则、同一 vLLM 引擎（BF16 LoRA，贪心）。J 不输入物体尺寸、'
             '一次输出尺寸与位置朝向；P 的尺寸来自训练集先验（查表）。本对比仅作描述，不改变今天冻结的交付。', '',
             f"训练：J 取 {result['training']['n']:,} 间房（sha256('a021-27b:' + uid) 顺序），其中与 P（臂 A）训练房重合 "
             f"{result['training']['overlap_with_arm_a']:,} 间。dev-select（{result['dev_select_j27']['requests']} 间）J 的"
             f"几何成功率 {_n(result['dev_select_j27']['checker'])}（仅描述）。", '',
             f"## 预注册主要比较（J − P，房屋连通组配对 bootstrap {result['repeats']:,} 次，seed {result['seed']}；Holm 校正两项）", '',
             '| 指标 | J | P | J − P | 95% CI | p | Holm p |', '|---|---:|---:|---:|---:|---:|---:|',
             f"| (a) 全请求几何成功率 | {_n(a['J'])} | {_n(a['P'])} | {_n(a['estimate'], True)} | {_ci(a)} | "
             f"{r2.p_text(a['p'], result['repeats'])} | {r2.p_text(a['p_holm'], result['repeats'])} |",
             f"| (b) 尺寸 log MAE（共同完整，越低越好） | {_n(b['J'])} | {_n(b['P'])} | {_n(b['estimate'], True)} | {_ci(b)} | "
             f"{r2.p_text(b['p'], result['repeats'])} | {r2.p_text(b['p_holm'], result['repeats'])} |", '',
             '## 次要指标', '',
             f"- 足迹偏差 B（log(预测/真值) 面积，共同完整房）：J {_n(result['bias_B']['J']['estimate'], True)} "
             f"{_ci(result['bias_B']['J'])}，P {_n(result['bias_B']['P']['estimate'], True)} {_ci(result['bias_B']['P'])}。",
             f"- a′（x/y 按 exp(−B/2) 去偏后重评）：J {_n(result['a_prime']['J'])}，P {_n(result['a_prime']['P'])}，"
             f"J − P {_n(result['a_prime']['delta']['estimate'], True)} {_ci(result['a_prime']['delta'])}。",
             f"- S1 修复后几何成功率：J {_n(result['s1']['J'])}，P {_n(result['s1']['P'])}，"
             f"J − P {_n(result['s1']['delta']['estimate'], True)} {_ci(result['s1']['delta'])}。", '',
             '## 分范围（各范围内同样的配对 bootstrap，未做多重校正）', '',
             '| 范围 | 请求 | 房屋组 k | J (a) | P (a) | Δ(a) [CI] | J log MAE | P log MAE | Δ(b) [CI] | J S1 | P S1 | Δ S1 [CI] |',
             '|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|']
    for name, s in result['scopes'].items():
        lines.append(f"| {name} | {s['requests']} | {s['components']} | {_n(s['J_a'])} | {_n(s['P_a'])} | "
                     f"{_n(s['geometry_success']['estimate'], True)} {_ci(s['geometry_success'])} | "
                     f"{_n(s['size_log_mae_J']['estimate'])} | {_n(s['size_log_mae_P']['estimate'])} | "
                     f"{_n(s['size_log_mae']['estimate'], True)} {_ci(s['size_log_mae'])} | {_n(s['J_s1'])} | {_n(s['P_s1'])} | "
                     f"{_n(s['s1']['estimate'], True)} {_ci(s['s1'])} |")
    lines += ['', '## 物体级尺寸误差（共同完整房）：G = P 用了常数兜底尺寸 (0.40, 0.50, 0.65 m) 的物体，N = 其余', '',
              '| 级别 | 物体数 | J log MAE | P log MAE | J IoU | P IoU |', '|---|---:|---:|---:|---:|---:|']
    for level, v in result['object_levels'].items():
        lines.append(f"| {level} | {v['objects']} | {_n(v['J_size_log_mae'])} | {_n(v['P_size_log_mae'])} | "
                     f"{_n(v['J_size_iou'])} | {_n(v['P_size_iou'])} |")
    lines += ['', '规则：截断规则（修订 005）——输出到达 token 上限且未停止记为失败；比率以全部请求为分母；尺寸指标只在两者都完整的房间上计算。'
              '仅一个来源或房屋组很少的范围，区间只作参考（见 results.json 的 flags / warnings）。']
    return '\n'.join(lines) + '\n'


# ---- cli -----------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('rows')
    for flag in ('data', 'base27', 'arm-a', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--sft-runs', type=Path, nargs=2, required=True, metavar=('SFT_JOINT', 'SFT_PLACEMENT'))
    step.add_argument('--dev', type=Path, nargs='+', required=True)
    step.add_argument('--test', type=Path, nargs='+', required=True)
    step = sub.add_parser('greedy')
    step.add_argument('--run-root', type=Path, required=True)
    step.add_argument('--slice', choices=tuple(SLICES), required=True)
    step = sub.add_parser('compare')
    step.add_argument('--run-root', type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    step.add_argument('--repeats', type=int, default=REPEATS)
    return parser.parse_args(argv)


def main(argv=None, load=None):
    args = parse_args(argv)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    result = greedy(args, load) if args.command == 'greedy' else {'rows': rows, 'compare': compare}[args.command](args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
