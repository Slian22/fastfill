"""Amendment 020 §3-§5 (EXPLORATORY): the placers OS-DPO2 (amendment-014 final DPO2 merge) and P (rpo-dpo1, exact
policy) given each size source on E2 ∪ E3 (amendment 014 E2, amendment 015 E3), with the amendment-017 metrics.

  p-data  --eval-data RUN/eval-data-v1 --e2 PLAN_E2 --e3 PLAN_E3 --estimates RUN/sizes-020-X --mode llm-L|llm-O2 --out DIR
          P's input in the eval-data layout (infer_shard plan/shard/merge and evaluate_pair read it unchanged) for the
          E2 ∪ E3 rooms the sizer answered: each placement request is eval-data's with only objects[].size replaced
          (p_request; sizes_only checks every room) by optiscene_eval.load_estimates' sizes; the prior sizes must rebuild
          every eval-data placement request byte for byte (prior_identity). Failed rooms are listed (size_failures).
  score   --placer os|p --mode prior|gt|llm-L|llm-O2 --eval-data ... --e2 PLAN_E2 --e3 PLAN_E3 --p-geometry RUN/geometry-v5
          --p-run P_RUN --out DIR [--gen-e2 GEN --gen-e3 GEN] [--p-data DIR] [--p-infer DIR] [--workers N]
          One row per E2 ∪ E3 room (the plans' rooms). OS: optiscene_eval.score_e2 on the generation (the request of the
          size source: prior = eval-data's, gt = GT sizes, llm = p-data's, so the checker scores with those sizes). P prior:
          the geometry-v5 rows of P's test run (amendment-005 cap rule) after prior_identity; P llm: evaluate_pair._score
          of the merged inference (its run identity must equal P_RUN's policy and decoding) with the cap rule. A room the
          sizer failed is a failure row (error size:<reason>). Every row: S1 (output_repair.repair_row at the amendment-012
          settings) gives admission strict (downstream.accept) and required = {} (endpoint_ok, attach_ok and
          admit_no_required_ok), raw and after S1.
  report  --rows OUT/rows --out DIR [--repeats 2000]
          Table B: (size source -> placer) x {E2, E3, family}; paired contrasts (report_v2.contrasts, house-cluster
          bootstrap, seed 42; no CI below report_v2.NO_CI components) llm-L - prior within each placer and OS-DPO2 - P
          within each size source; E3 - E2 descriptive gaps. Arms without rows are listed, never substituted.
  budget  --cap 30 --planned GPU_H [--jobs ID ...] [--sizes DIR ...]
          sizers: the larger of sacct (jobs named unknownsize-020-sizer-* and --jobs) and the published sizer receipts
          (gpu_hours of DIR/result.json); plus earlier placers-020-* jobs (sacct) and the planned GPU-h. sacct charges
          the elapsed time once a job ended, its time limit while pending or running. Exits 1 above the cap, or when no
          sizer GPU use is found at all.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from fastfill.comparison import contract, evaluate_pair, geometry, output_repair, report_v2
from fastfill.comparison import optiscene_eval as oe
from fastfill.comparison import train_pair_helpers as h

ARMS = ('os-prior', 'os-gt', 'os-llm-L', 'os-llm-O2', 'p-prior', 'p-llm-L', 'p-llm-O2')
NAMES = {'os': 'OS-DPO2', 'p': 'P'}
MAX_TRIALS, TIME_LIMIT = 2000, 60.0  # Amendment 012 S1 settings.
FIELDS = ('complete', 'geometry_success', 'geometry_success_s1', 'oor', 'strict_raw', 'strict_s1', 'noreq_raw',
          'noreq_s1')
RATES = ('complete', 'geometry_success', 'geometry_success_s1', 'strict_raw', 'strict_s1', 'noreq_raw', 'noreq_s1')
MEANS = ('oor', 'size_log_mae', 'bias_fp')
CONTRASTS = (('OS-DPO2: llm-L − prior', 'os-llm-L', 'os-prior'), ('P: llm-L − prior', 'p-llm-L', 'p-prior'),
             ('prior: OS-DPO2 − P', 'os-prior', 'p-prior'), ('llm-L: OS-DPO2 − P', 'os-llm-L', 'p-llm-L'),
             ('llm-O2: OS-DPO2 − P', 'os-llm-O2', 'p-llm-O2'))
POLICY = ('policy', 'stage', 'base_sha256', 'lineage_sha256', 'training_data')
RUN_SPECIFIC = ('data', 'plan', 'shard_count')  # infer_shard config keys that name this run, not the policy/decoding.
WAITING = ('PENDING', 'RUNNING', 'REQUEUED', 'RESIZING', 'SUSPENDED', 'CONFIGURING', 'COMPLETING')
PLACERS, SIZERS = 'placers-020-', 'unknownsize-020-sizer-'  # Job-name prefixes (this chain, submit_020_sizers.sh).
SACCT = ('-S', '2026-10-14T00:00:00', '-o', 'JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES')


def _code():
    return {p.name: h.sha256(p) for p in map(Path, (__file__, oe.__file__, output_repair.__file__, geometry.__file__))}


def _publish(out, files, meta):
    """Write-once directory: the files, then MANIFEST.json (meta + every file's sha256), no-replace publish."""
    from fastfill.comparison.corpus import _publish as publish
    out = Path(out)
    oe._refuse(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        for name, payload in files.items():
            (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
            (Path(tmp) / name).write_bytes(payload)
        manifest = {**meta, 'files': h.artifact_manifest(tmp)}
        (Path(tmp) / 'MANIFEST.json').write_text(oe._dumps(manifest, indent=1) + '\n')
        publish(tmp, out)
    return manifest


# ---- rooms and requests --------------------------------------------------------------------------------------------

def universe(e2, e3):
    """{uid: 'e2' | 'e3'} of the two plans' rooms in plan order; a room in both is refused."""
    out = {}
    for pop, plan in (('e2', e2), ('e3', e3)):
        oe._verified(plan)
        for r in oe._jsonl(Path(plan) / 'requests.jsonl'):
            if out.setdefault(r['uid'], pop) != pop:
                raise ValueError(f'{r["uid"]} is in both E2 and E3')
    return out


def _eval(eval_data, uids):
    """eval-data rooms, placement and joint request rows, prior sizes of exactly these uids, and the data manifest."""
    manifest = oe._verified(eval_data)
    load = lambda name: {r['uid']: r for r in oe._jsonl(Path(eval_data) / name) if r['uid'] in uids}
    rooms, placement, joint, prior = (load(n) for n in ('rooms.jsonl', 'placement/requests.jsonl',
                                                        'joint/requests.jsonl', 'size-estimates.jsonl'))
    if any(set(x) != set(uids) for x in (rooms, placement, joint, prior)):
        raise ValueError('eval-data does not cover the plan rooms')
    return rooms, placement, joint, {u: r['estimated_sizes'] for u, r in prior.items()}, manifest


def p_request(room, row, sizes):
    """The eval-data placement request `row` of `room` rebuilt with these object sizes (as eval_data built it)."""
    return {**{k: v for k, v in row.items() if k != 'messages'},
            'messages': contract.messages(room, 'placement', estimated_sizes=sizes)[:2]}


def _without_sizes(row):
    user = h.strict_json(row['messages'][1]['content'])
    user['objects'] = [{k: v for k, v in o.items() if k != 'size'} for o in user['objects']]
    return {k: v for k, v in row.items() if k != 'messages'}, row['messages'][0], row['messages'][1]['role'], user


def sizes_only(new, old):
    """True iff two placement requests differ at most in their objects' size fields."""
    return _without_sizes(new) == _without_sizes(old)


def prior_identity(rooms, placement, prior):
    """The prior sizes rebuild every eval-data placement request byte for byte (P's prior run = eval-data's input)."""
    bad = [u for u in rooms if oe._dumps(p_request(rooms[u], placement[u], prior[u])) != oe._dumps(placement[u])]
    if bad:
        raise ValueError(f'prior sizes do not rebuild the eval-data placement requests of {len(bad)} rooms: {bad[:5]}')
    return {'rooms': len(rooms), 'byte_identical': len(rooms)}


def p_data(eval_data, e2, e3, estimates, mode, out):
    if mode not in oe.SIZE_MODES_020:
        raise ValueError(f'mode must be one of {oe.SIZE_MODES_020}')
    out, eval_data, estimates = Path(out), Path(eval_data), Path(estimates)
    oe._refuse(out)
    pops = universe(e2, e3)
    rooms, placement, joint, prior, manifest = _eval(eval_data, pops)
    identity = prior_identity(rooms, placement, prior)
    sizes = oe.load_estimates(estimates, [rooms[u] for u in pops])
    keep = [u for u in pops if not isinstance(sizes[u], str)]
    if not keep:
        raise ValueError('the sizer answered no room: nothing for P to place')
    requests = {u: p_request(rooms[u], placement[u], sizes[u]) for u in keep}
    if not all(sizes_only(requests[u], placement[u]) for u in keep):
        raise ValueError('a substituted request differs from eval-data beyond the object sizes')
    files = {'joint/requests.jsonl': [joint[u] for u in keep], 'placement/requests.jsonl': [requests[u] for u in keep],
             'rooms.jsonl': [rooms[u] for u in keep],
             'size-estimates.jsonl': [{'uid': u, 'estimated_sizes': sizes[u]} for u in keep]}
    meta = {'protocol': 'amendment-020', 'status': 'EXPLORATORY', 'size_mode': mode, 'expected_uids': keep,
            'universe': {p: [u for u in pops if pops[u] == p] for p in ('e2', 'e3')},
            'size_failures': {u: sizes[u] for u in pops if isinstance(sizes[u], str)}, 'prior_identity': identity,
            'request_rule': 'eval-data placement request with only objects[].size replaced (p_request, sizes_only)',
            'size_estimates': str(estimates), 'size_estimates_manifest_sha256': oe._sha(estimates / 'MANIFEST.json'),
            'eval_data_manifest_sha256': oe._sha(eval_data / 'MANIFEST.json'),
            'training_manifest_sha256': manifest.get('training_manifest_sha256'),
            'plans': {'e2': oe._sha(Path(e2) / 'MANIFEST.json'), 'e3': oe._sha(Path(e3) / 'MANIFEST.json')},
            'code_sha256': _code()}
    return _publish(out, {name: oe._jsonl_bytes(rows) for name, rows in files.items()}, meta)


# ---- scoring -------------------------------------------------------------------------------------------------------

def _compact(room, context, row, arm, family, pop):
    """A scored row + S1 at the amendment-012 settings -> the report row (raw and S1 side by side)."""
    s1 = output_repair.repair_row((row, room, {'messages': context}, MAX_TRIALS, TIME_LIMIT))['s1']
    noreq = lambda a: bool(s1[f'endpoint_ok_{a}'] and s1[f'attach_ok_{a}'] and s1[f'admit_no_required_ok_{a}'])
    return {'uid': room['uid'], 'source': room['source'], 'group': room['group'], 'population': pop, 'family': family,
            'arm': arm, 'complete': row['complete'], 'error': row['error'],
            'geometry_success': bool(row['geometry_success']), 'oor': row['oor'], 'size_log_mae': row['size_log_mae'],
            'bias_fp': report_v2.size_bias(room['objects'], row['raw_layout'])[0] if row['complete'] else None,
            'strict_raw': bool(s1['strict_ok_raw']), 'noreq_raw': noreq('raw'),
            'geometry_success_s1': bool(s1['geometry_success_s1']), 'strict_s1': bool(s1['strict_ok_s1']),
            'noreq_s1': noreq('s1'), 's1_status': s1['status']}


def _row(task):
    """(kind, payload, room, context, arm, family, pop): kind os = optiscene_eval.score_e2 task, record = (request row,
    inference record) scored as evaluate_pair with the amendment-005 cap rule, row = an already scored row."""
    kind, payload, room, *rest = task
    if kind == 'os':
        row = oe.score_e2(payload)
    elif kind == 'record':
        request, record = payload
        row = evaluate_pair._score((room, request, record, 'placement'))
        if record['reached_token_cap'] and not record['stopped_on_eos']:
            row = report_v2.cap_failure(row)
    else:
        row = payload
    return _compact(room, rest[0], row, *rest[1:])


def _same_policy(receipt, p_run):
    ref = h.strict_json((Path(p_run) / 'run_receipt.json').read_text())
    key = lambda r: ({k: r.get(k) for k in POLICY}, {k: v for k, v in r['config'].items() if k not in RUN_SPECIFIC})
    if not receipt.get('policy') or key(receipt) != key(ref):
        raise ValueError(f'the P run is not the exact policy and decoding of {p_run}')


def score(placer, mode, eval_data, e2, e3, p_geometry, p_run, out, gen_e2=None, gen_e3=None, p_data=None,
          p_infer=None, workers=1):
    arm, out, eval_data = f'{placer}-{mode}', Path(out), Path(eval_data)
    if arm not in ARMS:
        raise ValueError(f'no arm {arm}: {ARMS}')
    oe._refuse(out)
    pops = universe(e2, e3)
    rooms, placement, _, prior, _ = _eval(eval_data, pops)
    family = oe.families()[0]
    fam = {u: family(rooms[u].get('room_type')) for u in pops}
    llm, inputs, failures, sized = mode in oe.SIZE_MODES_020, {}, {}, {}
    if llm:
        data = oe._verified(p_data)
        if data.get('size_mode') != mode or data['universe'] != {p: [u for u in pops if pops[u] == p] for p in ('e2', 'e3')}:
            raise ValueError(f'{p_data} is not the {mode} P data of these rooms')
        failures = data['size_failures']
        sized = {r['uid']: r for r in oe._jsonl(Path(p_data) / 'placement/requests.jsonl')}
        inputs['p_data_manifest_sha256'] = oe._sha(Path(p_data) / 'MANIFEST.json')
    context = lambda u: placement[u]['messages']  # S1 reads boundary, height and fixed boxes only: equal in every mode

    def request(u):  # The size source's request: the checker scores the layout with its sizes.
        if llm:
            return sized[u]['messages']
        if mode == 'prior':
            return placement[u]['messages']
        return contract.messages(rooms[u], 'placement', estimated_sizes={o['id']: o['size'] for o in rooms[u]['objects']})[:2]
    tasks = []
    if placer == 'os':
        for pop, plan, gen in (('e2', Path(e2), Path(gen_e2)), ('e3', Path(e3), Path(gen_e3))):
            manifest, rows = oe._verified(plan), oe._jsonl(plan / 'requests.jsonl')
            if llm and (manifest.get('size_mode'), manifest.get('size_estimates_manifest_sha256')) != (
                    mode, data['size_estimates_manifest_sha256']):
                raise ValueError(f'{plan} was not planned from the estimates of {p_data}')
            records = oe._load_gen(gen.parent, gen.name, plan, [r['key'] for r in rows if 'sft' in r['messages']])
            inputs[pop] = {'plan_manifest_sha256': oe._sha(plan / 'MANIFEST.json'),
                           'gen_manifest_sha256': oe._sha(gen / 'MANIFEST.json')}
            for r in rows:
                if r['mode'] != mode or r['format'] != 'optiscene':
                    continue
                u = r['uid']
                if ('sft' in r['messages']) == (u in failures):
                    raise ValueError(f'{u}: plan and P data disagree on the sizer failure')
                score_task = (rooms[u], None, r, {'error': 'size:' + failures[u]}, arm) if u in failures else (
                    rooms[u], request(u), r, records[r['key']], arm)
                tasks.append(('os', score_task, rooms[u], context(u), arm, fam[u], pop))
    elif not llm:
        inputs['prior_identity'] = prior_identity(rooms, placement, prior)
        if h.strict_json((Path(p_run) / 'run_receipt.json').read_text())['data']['manifest_sha256'] != \
                oe._sha(eval_data / 'MANIFEST.json'):
            raise ValueError(f'{p_run} did not run on {eval_data}')
        scored = {r['uid']: r for r in oe.p_test_rows(p_geometry, p_run) if r['uid'] in pops}
        tasks = [('row', scored[u], rooms[u], context(u), arm, fam[u], pops[u]) for u in pops if u in scored]
        inputs.update(p_geometry_manifest_sha256=oe._sha(Path(p_geometry) / 'MANIFEST.json'),
                      p_run_manifest_sha256=oe._sha(Path(p_run) / 'MANIFEST.json'))
    else:
        records, receipt = evaluate_pair._inference(Path(p_infer), Path(p_data), data['expected_uids'])
        _same_policy(receipt, p_run)
        inputs.update(p_infer_manifest_sha256=oe._sha(Path(p_infer) / 'MANIFEST.json'), run_sha256=receipt['run_sha256'],
                      reference_run_manifest_sha256=oe._sha(Path(p_run) / 'MANIFEST.json'))
        tasks = [('record', (sized[r['uid']], r), rooms[r['uid']], context(r['uid']), arm, fam[r['uid']], pops[r['uid']])
                 for r in records]
        tasks += [('row', geometry.score(rooms[u], None, None, 'placement', error='size:' + failures[u]), rooms[u],
                   context(u), arm, fam[u], pops[u]) for u in failures]
    if sorted(t[2]['uid'] for t in tasks) != sorted(pops):
        raise ValueError(f'{arm}: rows do not cover the plan rooms exactly')
    order = {u: k for k, u in enumerate(pops)}
    rows = sorted(oe._map(_row, tasks, workers), key=lambda r: order[r['uid']])
    meta = {'protocol': 'amendment-020', 'status': 'EXPLORATORY', 'arm': arm, 'placer': NAMES[placer], 'mode': mode,
            'label': oe.LABEL if placer == 'os' else 'P = rpo-dpo1, exact policy (amendment 010)',
            'rooms': len(rows), 'populations': dict(collections.Counter(r['population'] for r in rows)),
            'size_failures': len(failures), 'eval_data_manifest_sha256': oe._sha(eval_data / 'MANIFEST.json'),
            'inputs': inputs, 's1': {'max_trials': MAX_TRIALS, 'time_limit_s': TIME_LIMIT},
            'rows': 'one per E2 ∪ E3 room; a sizer failure is a failure row (error size:<reason>); rates use every room',
            'code_sha256': _code()}
    return _publish(out, {'rows.jsonl': oe._jsonl_bytes(rows)}, meta)


# ---- report --------------------------------------------------------------------------------------------------------

def _level(rows):
    n, done = len(rows), [r for r in rows if r['complete']]
    mean = lambda k: oe._mean([r[k] for r in done])
    return {'rooms': n, **{k: sum(bool(r[k]) for r in rows) / n if n else None for k in RATES},
            **{k: mean(k) for k in MEANS}, 'size_failures': sum(str(r['error']).startswith('size:') for r in rows)}


def _contrast(first, second, repeats):
    design, result = report_v2._design(second), report_v2.contrasts(first, second, FIELDS, repeats)
    if design['components'] < report_v2.NO_CI:
        result = {f: {**v, 'ci95': None} for f, v in result.items()}
    return {**design, **result}


def report(rows_root, out, repeats=2000):
    rows_root, out = Path(rows_root), Path(out)
    oe._refuse(out)
    arms = {a: oe._jsonl(rows_root / a / 'rows.jsonl') for a in ARMS if (rows_root / a / 'MANIFEST.json').is_file()
            and oe._verified(rows_root / a)}
    if not arms:
        raise ValueError(f'no published arm under {rows_root}')
    rooms = {a: sorted((r['uid'], r['population'], r['family']) for r in rows) for a, rows in arms.items()}
    if len({tuple(v) for v in rooms.values()}) != 1:
        raise ValueError('the arms cover different rooms')
    scopes = [('E2', 'e2', None), ('E3', 'e3', None)] + [(f, None, f) for f in sorted({x[2] for x in next(iter(
        rooms.values()))})]
    pick = lambda rows, pop, fam: [r for r in rows if pop in (None, r['population']) and fam in (None, r['family'])]
    table = {s: {a: _level(pick(rows, pop, fam)) for a, rows in arms.items()} for s, pop, fam in scopes}
    contrasts = {label: {s: _contrast(pick(arms[a], pop, fam), pick(arms[b], pop, fam), repeats) for s, pop, fam in scopes}
                 for label, a, b in CONTRASTS if a in arms and b in arms}
    gaps = {a: {k: None if table['E3'][a][k] is None or table['E2'][a][k] is None else table['E3'][a][k] - table['E2'][a][k]
                for k in RATES + MEANS} for a in arms}
    summary = {'protocol': 'amendment-020', 'status': 'EXPLORATORY', 'arms': list(arms),
               'not_published': [a for a in ARMS if a not in arms],
               'not_computed': [label for label, a, b in CONTRASTS if not (a in arms and b in arms)],
               'table_b': table, 'contrasts': contrasts, 'e3_minus_e2': gaps, 'repeats': repeats}
    meta = {'protocol': 'amendment-020', 'status': 'EXPLORATORY', 'repeats': repeats, 'seed': report_v2.SEED,
            'rows_manifest_sha256': {a: oe._sha(rows_root / a / 'MANIFEST.json') for a in arms}, 'code_sha256': _code()}
    return _publish(out, {'summary.json': (oe._dumps(summary, indent=1) + '\n').encode('utf-8'),
                          'report.md': _markdown(summary).encode('utf-8')}, meta)


def _f(v):
    return '—' if v is None else f'{v:.3f}'


def _arm_name(a):
    placer, mode = a.split('-', 1)
    return f'{mode} → {NAMES[placer]}'


def _markdown(s):
    lines = ['# 修订 020：两阶段（尺寸来源 → 放置器）按房间族（EXPLORATORY）', '',
             f'放置器：OS-DPO2 = {oe.LABEL} 的 DPO2 终阶段（修订 014）；P = rpo-dpo1 精确策略（修订 010）。尺寸来源：prior = 训练先验'
             '（P 的输入），gt = 真值尺寸（oracle），llm-L / llm-O2 = 仅估计尺寸的 LLM（修订 020 §2）。每个放置器的请求都带该来源的尺寸，'
             '检查器用这些尺寸打分。尺寸估计失败的房间计为失败（不删除），所有比率以 E2 ∪ E3 的全部房间为分母。', '',
             '指标（修订 017）：检查器成功（冻结检查器，主要）、OOR（仅完整房间）；infinite-task-world 准入（严格移植，次要）与 '
             'required = {} 变体，raw 与 S1（修订 012 设定：max_trials 2000、每房间 60 s）。E3 每族至多 100 间（修订 015），'
             '合并 E3 是族平衡样本，不按测试集人口加权。', '']
    if s['not_published']:
        lines += [f"未发布（不替代）：{', '.join(s['not_published'])}。", '']
    lines += ['## 表 B：尺寸来源 → 放置器 × 范围', '',
              '| 范围 | 尺寸来源 → 放置器 | 房间 | 尺寸失败 | 完整 | 检查器成功 raw → S1 | OOR | 严格准入 raw → S1 '
              '| required = {} raw → S1 | 尺寸 log MAE | B |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scope, arms in s['table_b'].items():
        for a, v in arms.items():
            lines.append(f"| {scope} | {_arm_name(a)} | {v['rooms']} | {v['size_failures']} | {_f(v['complete'])} | "
                         f"{_f(v['geometry_success'])} → {_f(v['geometry_success_s1'])} | {_f(v['oor'])} | "
                         f"{_f(v['strict_raw'])} → {_f(v['strict_s1'])} | {_f(v['noreq_raw'])} → {_f(v['noreq_s1'])} | "
                         f"{_f(v['size_log_mae'])} | {_f(v['bias_fp'])} |")
    lines += ['', f"## 配对对比（房屋连通组自举，{s['repeats']:,} 次，seed 42；EXPLORATORY）", '',
              f'估计 = 前者 − 后者（房间微平均）；k = 房屋连通组数，k < {report_v2.NO_CI} 不给区间；flags 见 report_v2._design。', '',
              '| 对比 | 范围 | k | flags | 完整 | 检查器成功 | 检查器成功 S1 | 严格准入 | 严格准入 S1 | required = {} | '
              'required = {} S1 | OOR |', '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    cell = lambda v: _f(v['estimate']) + ('' if v['ci95'] is None else f" [{_f(v['ci95'][0])}, {_f(v['ci95'][1])}]")
    for label, scopes in s['contrasts'].items():
        for scope, c in scopes.items():
            lines.append(f"| {label} | {scope} | {c['components']} | {c['flags'] or ''} | "
                         + ' | '.join(cell(c[f]) for f in ('complete', 'geometry_success', 'geometry_success_s1',
                                                            'strict_raw', 'strict_s1', 'noreq_raw', 'noreq_s1', 'oor'))
                         + ' |')
    if s['not_computed']:
        lines += ['', f"缺少一侧、未计算：{', '.join(s['not_computed'])}。"]
    lines += ['', '## E3 − E2（描述性差距，无区间）', '',
              '| 尺寸来源 → 放置器 | 检查器成功 | 检查器成功 S1 | 严格准入 | required = {} | OOR |', '|---|---:|---:|---:|---:|---:|']
    for a, g in s['e3_minus_e2'].items():
        lines.append(f"| {_arm_name(a)} | " + ' | '.join(_f(g[k]) for k in (
            'geometry_success', 'geometry_success_s1', 'strict_raw', 'noreq_raw', 'oor')) + ' |')
    return '\n'.join(lines) + '\n'


# ---- GPU budget ----------------------------------------------------------------------------------------------------

def gpu_hours(lines, jobs, prefixes=(PLACERS, SIZERS)):
    """{job: (name, GPU-h)} from sacct -P lines (SACCT columns) of the given jobs and every job whose name starts with
    one of the prefixes: elapsed once ended, the time limit (worst case) while pending or running."""
    charged = {}
    for line in lines:
        job, name, state, elapsed, limit, alloc, req = line.split('|')
        if '.' in job or not (job in jobs or name.startswith(prefixes)):
            continue
        gpus = re.search(r'gres/gpu=(\d+)', alloc or req)
        seconds = int(limit) * 60 if state.split()[0] in WAITING else int(elapsed)
        charged[job] = (name, (int(gpus.group(1)) if gpus else 0) * seconds / 3600)
    return charged


def budget(cap, planned, jobs=(), sizes=(), run=subprocess.run):
    done = run(['sacct', '-X', '-n', '-P', '-u', os.environ['USER'], *SACCT], capture_output=True, text=True, check=True)
    charged = gpu_hours([x for x in done.stdout.splitlines() if x.strip()], set(jobs))
    if set(jobs) - set(charged):  # A job pending on a dependency has no eligible time, so the -S window misses it.
        more = run(['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(set(jobs) - set(charged))), *SACCT[2:]],
                   capture_output=True, text=True, check=True)
        charged.update(gpu_hours([x for x in more.stdout.splitlines() if x.strip()], set(jobs)))
    if set(jobs) - set(charged):
        raise ValueError(f'sacct does not know the jobs {sorted(set(jobs) - set(charged))}')
    receipts = {str(d): float(json.loads((Path(d) / 'result.json').read_text()).get('gpu_hours') or 0) for d in sizes
                if (Path(d) / 'MANIFEST.json').is_file() and (Path(d) / 'result.json').is_file()}
    sizer_jobs = {j: g for j, (n, g) in charged.items() if not n.startswith(PLACERS)}
    if not sizer_jobs and not any(receipts.values()):
        raise ValueError('no sizer GPU use found (no unknownsize-020-sizer-* job, SIZERS_JOB or gpu_hours receipt)')
    sizers = max(sum(sizer_jobs.values()), sum(receipts.values()))
    placers = sum(g for n, g in charged.values() if n.startswith(PLACERS))
    total = sizers + placers + planned
    result = {'sizer_jobs_gpu_h': sizer_jobs, 'sizer_receipts_gpu_h': receipts, 'sizers_gpu_h': sizers,
              'earlier_placer_jobs_gpu_h': placers, 'planned_gpu_h': planned, 'total_gpu_h': total, 'cap_gpu_h': cap}
    if total > cap:
        raise ValueError(f'amendment-020 GPU cap exceeded: {result}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('p-data')
    for name in ('eval-data', 'e2', 'e3', 'estimates', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=oe.SIZE_MODES_020, required=True)
    p = sub.add_parser('score')
    p.add_argument('--placer', choices=tuple(NAMES), required=True)
    p.add_argument('--mode', choices=('prior', 'gt', *oe.SIZE_MODES_020), required=True)
    for name in ('eval-data', 'e2', 'e3', 'p-geometry', 'p-run', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('gen-e2', 'gen-e3', 'p-data', 'p-infer'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--workers', type=int, default=1)
    p = sub.add_parser('report')
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=2000)
    p = sub.add_parser('budget')
    p.add_argument('--cap', type=float, required=True)
    p.add_argument('--planned', type=float, required=True)
    p.add_argument('--jobs', nargs='*', default=[])
    p.add_argument('--sizes', type=Path, nargs='*', default=[])
    a = parser.parse_args(argv)
    if a.command == 'p-data':
        m = p_data(a.eval_data, a.e2, a.e3, a.estimates, a.mode, a.out)
        print(oe._dumps({'rooms': len(m['expected_uids']), 'size_failures': len(m['size_failures'])}))
    elif a.command == 'score':
        m = score(a.placer, a.mode, a.eval_data, a.e2, a.e3, a.p_geometry, a.p_run, a.out, a.gen_e2, a.gen_e3,
                  a.p_data, a.p_infer, a.workers)
        print(oe._dumps({k: m[k] for k in ('arm', 'rooms', 'size_failures')}))
    elif a.command == 'report':
        report(a.rows, a.out, a.repeats)
    else:
        try:
            print(oe._dumps(budget(a.cap, a.planned, a.jobs, a.sizes)))
        except ValueError as exc:
            print(f'BUDGET REFUSED: {exc}', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
