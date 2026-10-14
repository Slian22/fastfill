"""Amendment 018: distil arm P from the free open teacher O2 (SFT-D -> DPO1-D -> DPO2-D), selected on dev. EXPLORATORY.

python -m fastfill.comparison.distill_018 STEP ...   (teacher chain: submit_018_teacher.sh)

plan     --data RUN/main-data-v1 --sft-runs SFT_J SFT_P --exclude RUN/dpo-001/positives --ta RUN/dpo-004/train-subset
         --dev RUN/dev-select-v1 RUN/dev-full-v1 --test RUN/eval-data-v1 --out RUN/distill-018/plan
         [--tb 5000] [--tc 3000] [--shards 8]
    Section 1 rooms. T-A = the amendment-013 train subset exactly: its manifest verifies (scope op_dpo2_train_subset,
    same main data) and its request / room / prior files and SFT prompt token counts equal the ones rebuilt here.
    Pool = the SFT common length-admitted train UIDs (dpo_stages._admitted) minus the 2,200 dpo-001 positives minus
    T-A, as onpolicy_dpo subset defines it; a pool or T-A UID or house group in any dev / test expected_uids /
    rooms.jsonl raises (onpolicy_dpo._held_out). T-B = draw(pool, --tb, 'a018-distill:'), then T-C = draw(pool - T-B,
    --tc, 'a018-dpo2:'): n // 3 rooms per scale_curve.BINS object-count bin (<= 20, 21-50, > 50; the first n % 3 bins
    take one more), a bin short of its quota passes the rest to the next denser bin (the densest bin's rest is the
    recorded shortfall); within a bin quotas proportional to source (largest remainder, ties by source name), rooms by
    ascending sha256(salt + uid), uid tie-break. Written new-only in onpolicy_dpo subset's layout (split train, so
    infer.load_eval, llm_open and sample_vllm read it unchanged) for T-A + T-B + T-C: per arm requests.jsonl (the SFT
    system + user turns verbatim), rooms.jsonl, size-estimates.jsonl (P's request sizes = the train prior) and
    MANIFEST.json (sorted expected_uids, sft_prompt_tokens, the selection block), plus cohort-format uid files
    ({uids} in plan order, as llm_agent.subset reads): uids/{ta,tb,tc,teacher}.json (teacher = T-A + T-B) and
    shards/sNN.json (SHARDS stratified round-robin shards of the teacher set: ordered by bin, source,
    sha256('a018-distill:' + uid), uid, then dealt in turn), and cohort.json = llm_open.all_rooms_cohort(plan) (not in
    MANIFEST.json, whose hash it holds; llm_open --all-rooms rebuilds and compares it).
(teacher)   isambard_llm_open.sbatch CODE O2 t018 sNN: llm_open run --system O2 --arm placement --all-rooms --uids
            plan/shards/sNN.json -> RUN/llm-open-O2-T018-sNN-001 (amendment-007 engine and sampling, amendment-016 section
            2 harness and addendum B2; resumes from receipts).
teacher  --plan RUN/distill-018/plan --run-root RUN --reference RUN/llm-open-O2-P-pilot-001 --out RUN/distill-018/teacher
         [--workers 32]
    Teacher filter (CPU). Binding (raises): the plan verifies and its cohort, teacher set and shard files equal their
    rebuild; every shard run is a complete llm_open run (evaluate_llm._llm: manifest, identity hash, data and cohort
    hashes, receipts) on exactly its shard file, with system O2, request_arm placement, cohort_rule all_rooms, no
    limit, and a config equal to the amendment-016 section 2 O2 run --reference apart from cohort_rule and versions
    (versions are compared and recorded). Per room the harness answer (llm_harness) goes through addendum B2
    (evaluate_llm._enforced) and evaluate_llm._score_llm on P's request, exactly as evaluate_llm scores placement rows;
    a cap hit without EOS fails (report_v2.cap_failure, as report_v2). TEACHER_RULE: target iff complete and
    geometry_success (frozen checker, raw, no S1). The target is rebuilt in P's training output format (contract.messages'
    labels: id, pos as scene.num cm values, yaw, on when given; contract._json); P's parser must accept it and it must
    re-parse to the answer's layout up to cm rounding (else raise); it is then re-scored from scratch and must still
    pass, else it is dropped and counted (rounding_broke). Writes targets.jsonl (uid, source, group, set, bin, shard,
    prompt = P's SFT / DPO system + user messages, target, verdict, answer_sha256 of the harness answer as returned),
    ledger.jsonl (every teacher room and its class), summary.json (pass rates per set / bin / source, drops, B2
    counts, GPU-hours from the shards' cost.json), report.md (Chinese) and MANIFEST.json, new-only.
teacher_minutes (called by submit_018_teacher.sh before anything is submitted): the shards' --time, weighted by room
    density (TIME_RULE).
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from fastfill import scene
from fastfill.comparison import contract
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer as i
from fastfill.comparison import llm_agent as L
from fastfill.comparison import llm_open as O
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import scale_curve as sc
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

ARMS, ARM, SYSTEM = h.ARMS, 'placement', 'O2'
SALTS = {'T-B': 'a018-distill:', 'T-C': 'a018-dpo2:'}
ROOMS = {'T-B': 5000, 'T-C': 3000}
SHARDS = 8
SCOPE, PROTOCOL = 'distill_018_plan', 'amendment-018-teacher-v1'
BINS = tuple(name for name, _, _ in sc.BINS)
CHECKS = sc.CHECKER_REASONS
CLASSES = ('target', 'incomplete', 'checker_fail', 'rounding_broke')
ROUNDING = 0.005 + 1e-9  # scene.num rounds to cm: no coordinate may move further.
UID_FILES = {'T-A': 'uids/ta.json', 'T-B': 'uids/tb.json', 'T-C': 'uids/tc.json', 'teacher': 'uids/teacher.json'}
SHARD_RULE = ("teacher set (T-A + T-B) ordered by (scale_curve.BINS bin, source, sha256('a018-distill:' + uid), uid) "
              "and dealt in turn: shard k holds positions k, k + N, ...; each file lists its uids in plan order")
TEACHER_RULE = ("harness answer (llm_harness) of O2 with P's request, addendum B2 enforced, scored as evaluate_llm scores "
                "placement rows (cap hit without EOS = fail, as report_v2); target iff complete and geometry_success "
                "(frozen checker, raw, no S1); rebuilt in P's SFT output format (id, pos as scene.num cm values, yaw, "
                "on; contract._json), re-parsed (same layout up to cm rounding) and re-scored: it must still pass, "
                "else dropped as rounding_broke")
TIME_RULE = ("seconds per output token = the O2 P-input pilot's cost.json generation_seconds / its receipts' output "
             "tokens (every round); a teacher room costs its object-count bin's pilot mean output tokens; teacher bins = "
             "T-A's rooms.jsonl + T-B's draw quotas (the two sparser bins fill; a short densest bin only lowers the cost); a "
             "round-robin shard holds at most ceil(n_bin / SHARDS) rooms per bin; minutes = ceil(1.5 x (LOAD + seconds) / "
             "60), LOAD = pilot wall_seconds - generation_seconds")


def _code():
    return {p.name: h.sha256(p) for p in map(Path, (__file__, op.__file__, d.__file__, O.__file__, L.__file__,
                                                     ev.__file__, e.__file__, g.__file__, contract.__file__,
                                                     scene.__file__, i.__file__, h.__file__))}


def rank(salt, uid):
    return hashlib.sha256((salt + uid).encode('utf-8')).hexdigest()


def bin_of(objects):
    return next(name for name, lo, hi in sc.BINS if lo <= objects and (hi is None or objects <= hi))


def shard_name(k):
    return f's{k:02d}'


def shard_run(run_root, k):
    return Path(run_root) / f'llm-open-{SYSTEM}-T018-{shard_name(k)}-001'


# ---- plan ----------------------------------------------------------------------------------------

def _proportional(cell, n, salt):
    """cell {uid: source} -> n uids: quotas proportional to source (largest remainder, ties by name), hash-ranked."""
    sizes, total = Counter(cell.values()), len(cell)
    quotas = {s: n * c // total for s, c in sizes.items()} if total else {}
    for s in sorted(sizes, key=lambda s: (-(n * sizes[s] % total), s))[:n - sum(quotas.values())]:
        quotas[s] += 1
    ordered = sorted(cell, key=lambda u: (rank(salt, u), u))
    return [u for s in sorted(quotas) for u in [x for x in ordered if cell[x] == s][:quotas[s]]], dict(sorted(quotas.items()))


def bin_quotas(n):
    """n // 3 rooms per object-count bin; the first n % 3 bins take one more."""
    return {b: n // len(BINS) + (k < n % len(BINS)) for k, b in enumerate(BINS)}


def draw(pool, n, salt):
    """pool {uid: (bin, source)} -> (sorted uids, info); pure and input-order independent (module docstring)."""
    quotas = bin_quotas(n)
    carry, uids, bins = 0, [], {}
    for b in BINS:
        cell = {u: source for u, (bb, source) in pool.items() if bb == b}
        want = quotas[b] + carry
        take = min(want, len(cell))
        carry = want - take
        picked, by_source = _proportional(cell, take, salt)
        uids += picked
        bins[b] = {'pool': len(cell), 'quota': quotas[b], 'carried_in': want - quotas[b], 'drawn': take,
                   'by_source': by_source}
    return sorted(uids), {'salt': salt, 'requested': n, 'drawn': len(uids), 'bins': bins, 'shortfall': carry}


def shard_uids(teacher, cells, count):
    """SHARD_RULE: count lists of uids, each sorted (plan order)."""
    order = sorted(teacher, key=lambda u: (BINS.index(cells[u][0]), cells[u][1], rank(SALTS['T-B'], u), u))
    return [sorted(order[k::count]) for k in range(count)]


def _uid_file(name, uids, cells, rule):
    return {'version': 1, 'protocol': PROTOCOL, 'set': name, 'algorithm': rule,
            'counts': {'total': len(uids), 'by_bin': {b: sum(cells[u][0] == b for u in uids) for b in BINS}},
            'uids_sha256': L._digest(json.dumps(uids)), 'uids': uids}


def _layout(uids, rows, rooms):
    """onpolicy_dpo subset's files (split train) for uids, in that order."""
    files = {f'{arm}/requests.jsonl': [] for arm in ARMS} | {'rooms.jsonl': [], 'size-estimates.jsonl': []}
    for uid in uids:
        component, room = rows['joint'][uid]['group'], rooms[uid]
        if room['source'] != rows['joint'][uid]['source'] or any(rows[a][uid]['group'] != component for a in ARMS):
            raise ValueError(f'train row/room identity mismatch: {uid}')
        meta = {'uid': uid, 'source': room['source'], 'group': component, 'component': component, 'split': 'train',
                **({'room_type': room['room_type']} if room.get('room_type') is not None else {})}
        for arm in ARMS:
            files[f'{arm}/requests.jsonl'].append(i.validate_request({**meta, 'messages': rows[arm][uid]['messages'][:2]}, arm))
        files['rooms.jsonl'].append({**room, 'group': component, 'component': component, 'split': 'train'})
        request = h.strict_json(rows[ARM][uid]['messages'][1]['content'])
        files['size-estimates.jsonl'].append({'uid': uid, 'estimated_sizes': {o['id']: o['size'] for o in request['objects']}})
    return files


def plan(args):
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    if args.shards < 1 or args.shards > 100:
        raise ValueError('--shards must be 1..100 (two-digit shard names)')
    pairs, data = h.load_pair(args.data)
    if data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('the teacher plan requires the qualified main data')
    admitted, prompt_tokens, lengths = d._admitted(args.sft_runs, data, 'train')
    rows = {arm: h._by_uid(pairs['train'][arm]) for arm in ARMS}
    rooms = h._by_uid(h._rows(Path(args.data), 'train_rooms.jsonl', data['files']['train_rooms.jsonl']))
    if not set(admitted) <= rows['joint'].keys() & rooms.keys():
        raise ValueError('length-admitted UID outside the verified train rows/rooms')
    positives, _, _, positives_sha = d.load_positives(args.exclude)
    excluded = set(positives['uids'])
    ta_root = Path(args.ta)
    ta_manifest = e._verified(ta_root)
    ta = ta_manifest['expected_uids']
    if (ta_manifest.get('scope') != op.SUBSET_SCOPE or ta_manifest.get('training_manifest_sha256') != data['manifest_sha256']
            or not set(ta) <= set(admitted) - excluded):
        raise ValueError('T-A is not the amendment-013 train subset of this main data')
    if (any((ta_root / name).read_bytes() != d._jsonl(values) for name, values in _layout(ta, rows, rooms).items())
            or ta_manifest.get('sft_prompt_tokens') != {u: prompt_tokens[u][ARM] for u in ta}):
        raise ValueError('T-A rows differ from the ones rebuilt from the main data')
    cells = {u: (bin_of(len(rooms[u]['objects'])), rows['joint'][u]['source']) for u in admitted}
    drop = excluded | set(ta)
    pool = {u: cells[u] for u in admitted if u not in drop}
    held = {name: op._held_out(roots) for name, roots in (('dev', args.dev), ('test', args.test))}
    leaked = sorted(u for u in [*pool, *ta] if any(u in uids or rows['joint'][u]['group'] in groups
                                                   for uids, groups, _ in held.values()))
    if leaked:
        raise ValueError(f'{len(leaked)} plan rooms share a UID or house group with dev/test: {leaked[:5]}')
    tb, tb_info = draw(pool, args.tb, SALTS['T-B'])
    taken = set(tb)
    tc, tc_info = draw({u: c for u, c in pool.items() if u not in taken}, args.tc, SALTS['T-C'])
    sets = {'T-A': sorted(ta), 'T-B': tb, 'T-C': tc, 'teacher': sorted([*ta, *tb])}
    uids = sorted([*ta, *tb, *tc])
    if len(uids) != len(ta) + len(tb) + len(tc):
        raise ValueError('T-A, T-B and T-C overlap')
    shards = shard_uids(sets['teacher'], cells, args.shards)
    rules = {'T-A': 'RUN/dpo-004/train-subset expected_uids (amendment 013)',
             'T-B': f"draw(pool, {args.tb}, '{SALTS['T-B']}')", 'T-C': f"draw(pool - T-B, {args.tc}, '{SALTS['T-C']}')",
             'teacher': 'T-A + T-B'}
    selection = {'protocol': PROTOCOL, 'pool': {'length_admitted_train': len(admitted),
                                                'excluded_positives': len(excluded & set(admitted)),
                                                'excluded_t_a': len(ta), 'eligible': len(pool)},
                 'T-A': {'path': str(ta_root.resolve()), 'manifest_sha256': h.sha256(ta_root / 'MANIFEST.json'),
                         'rooms': len(ta)}, 'T-B': tb_info, 'T-C': tc_info,
                 'by_set': {k: {'by_bin': dict(Counter(cells[u][0] for u in v)),
                                'by_source': dict(sorted(Counter(cells[u][1] for u in v).items()))}
                            for k, v in sets.items()},
                 'shards': {'count': args.shards, 'rule': SHARD_RULE, 'rooms': [len(s) for s in shards]},
                 'excluded_positives_manifest_sha256': positives_sha,
                 'held_out': {name: receipts for name, (_, _, receipts) in held.items()}, 'sft_lengths_sha256': lengths}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        for name in ('uids', 'shards', *ARMS):
            (dest / name).mkdir()
        for name, values in _layout(uids, rows, rooms).items():
            (dest / name).write_bytes(d._jsonl(values))
        for name, path in UID_FILES.items():
            h.write_json(dest / path, _uid_file(name, sets[name], cells, rules[name]))
        for k, values in enumerate(shards):
            h.write_json(dest / 'shards' / f'{shard_name(k)}.json', _uid_file(shard_name(k), values, cells, SHARD_RULE))
        h.write_json(dest / 'MANIFEST.json', {
            'version': 1, 'scope': SCOPE, 'split': 'train', 'expected_uids': uids, 'uids_sha256': d._digest(uids),
            'counts': {'train': len(uids), **{k: len(v) for k, v in sets.items()}},
            'training_manifest_sha256': data['manifest_sha256'], 'main_data_manifest_sha256': data['manifest_sha256'],
            'sft_prompt_tokens': {u: prompt_tokens[u][ARM] for u in uids},
            'input_policy': 'system_user_only_labels_and_inventory_separate', 'shards': args.shards,
            'uid_files': {**UID_FILES, **{shard_name(k): f'shards/{shard_name(k)}.json' for k in range(args.shards)}},
            'selection': selection, 'code_sha256': _code(), 'files': h.artifact_manifest(dest)})
        h.write_json(dest / 'cohort.json', O.all_rooms_cohort(dest))  # After MANIFEST.json: it holds its hash.
        _publish(dest, out)
    return {'status': 'TEACHER_PLAN_COMPLETE', 'rooms': {k: len(v) for k, v in sets.items()},
            'shortfall': {'T-B': tb_info['shortfall'], 'T-C': tc_info['shortfall']}, 'shards': [len(s) for s in shards]}


def load_plan(root):
    """Verified plan -> {manifest, requests (P, by uid), rooms, sets, shards, cohort}; the cohort, the teacher set and
    the shard files must equal their rebuild."""
    root = Path(root)
    manifest = e._verified(root)
    if manifest.get('scope') != SCOPE:
        raise ValueError(f'{root} is not an amendment-018 teacher plan')
    requests = h._by_uid(i.load_eval(root, ARM)[0])
    rooms = h._by_uid(h._rows(root, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    cohort = h.strict_json((root / 'cohort.json').read_text())
    if cohort != O.all_rooms_cohort(root):
        raise ValueError('plan cohort differs from its rebuild')
    sets = {name: h.strict_json((root / path).read_text())['uids'] for name, path in UID_FILES.items()}
    cells = {u: (bin_of(len(r['objects'])), r['source']) for u, r in rooms.items()}
    shards = [L.subset(cohort, root / 'shards' / f'{shard_name(k)}.json') for k in range(manifest['shards'])]
    if (sets['teacher'] != sorted([*sets['T-A'], *sets['T-B']]) or sorted([*sets['teacher'], *sets['T-C']]) != cohort['uids']
            or shards != shard_uids(sets['teacher'], cells, manifest['shards'])):
        raise ValueError('plan uid sets or shards differ from their rebuild')
    return {'root': root, 'manifest': manifest, 'requests': requests, 'rooms': rooms, 'sets': sets, 'cells': cells,
            'shards': shards, 'cohort': cohort}


# ---- teacher filter ------------------------------------------------------------------------------

def training_target(text, room):
    """P's SFT output text (contract.messages' labels) for a valid placement answer."""
    value = contract.validate_output(text, room, ARM)['placements']
    return contract._json({'placements': [{'id': p['id'], 'pos': [scene.num(v) for v in p['pos']], 'yaw': p['yaw'],
                                           **({'on': p['on']} if 'on' in p else {})} for p in value]})


def _same_layout(answer, target, room):
    a, b = (contract.validate_output(t, room, ARM)['placements'] for t in (answer, target))
    return len(a) == len(b) and all(
        {k: v for k, v in x.items() if k != 'pos'} == {k: v for k, v in y.items() if k != 'pos'}
        and all(abs(p - q) <= ROUNDING for p, q in zip(x['pos'], y['pos'], strict=True)) for x, y in zip(a, b))


def _checks(row):
    return {'complete': row['complete'], 'geometry_success': row['geometry_success'], 'error': row['error'],
            **{k: row.get(k) for k in CHECKS}}


def judge(task):
    """One teacher room (row = the B2-enforced harness record) -> {class, answer checks, target, target checks}."""
    room, request, row = task
    scored = ev._score_llm((room, request, row, ARM))
    if scored.get('reached_token_cap') and not scored.get('stopped_on_eos', True):
        scored = r2.cap_failure(scored)
    result = {'class': 'incomplete' if not scored['complete'] else 'checker_fail', 'answer': _checks(scored),
              'target': None, 'target_checks': None}
    if not (scored['complete'] and scored['geometry_success']):
        return result
    target = training_target(row['raw'], room)
    if not i.parse_prediction(target, request, ARM)['valid'] or not _same_layout(row['raw'], target, room):
        raise ValueError(f"training-format target does not re-parse to the teacher layout: {room['uid']}")
    rescored = g.score(room, request['messages'], target, ARM)
    passed = rescored['complete'] and rescored['geometry_success']
    return {**result, 'class': 'target' if passed else 'rounding_broke', 'target': target if passed else None,
            'target_checks': _checks(rescored)}


def _teacher_run(plan_, root, path, uids, reference):
    """Verified shard run -> (harness records as returned, B2-enforced rows, B2 counts, shard info)."""
    records, receipt, result, cost = ev._llm(root, plan_['root'], plan_['cohort'], h.sha256(plan_['root'] / 'cohort.json'),
                                             uids)
    config = receipt['config']
    if (config.get('system') != SYSTEM or config.get('request_arm') != ARM or config.get('cohort_rule') != 'all_rooms'
            or 'request_sizes' in config or config.get('limit') is not None
            or receipt['cohort'].get('subset', {}).get('sha256') != h.sha256(path)):
        raise ValueError(f'{root} is not the O2 P-input teacher run of {path.name}')
    def bare(value):
        return {k: v for k, v in value.items() if k not in ('cohort_rule', 'versions')}
    if bare(config) != bare(reference):
        raise ValueError(f'{root} differs from the amendment-016 section 2 O2 run beyond cohort_rule and versions')
    rows, counts = ev._enforced(records['llm_harness'], plan_['requests'])
    return records['llm_harness'], rows, counts, {
        'run': str(root.resolve()), 'run_sha256': receipt['run_sha256'], 'rooms': len(uids),
        'gpu_hours': (cost or {}).get('gpu_hours'), 'slurm_job_ids': (cost or {}).get('slurm_job_ids'),
        'same_versions_as_reference': config['versions'] == reference['versions'], 'versions': config['versions']}


def _rates(ledger, key):
    groups = {}
    for entry in ledger:
        groups.setdefault(entry[key], []).append(entry['class'] == 'target')
    return {k: {'rooms': len(v), 'targets': sum(v), 'rate': sum(v) / len(v)} for k, v in sorted(groups.items())}


def _report(summary):
    lines = ['# 修订 018 教师数据：O2（Qwen3.8-27B）收到 P 的输入（EXPLORATORY）', '',
             'O2 收到与 P 完全相同的 system 与 user 消息（尺寸为训练集先验），另加同一条“只输出 JSON”后缀与修订 016 §2 的重试规则；'
             '按附录 B2 覆盖回答里的尺寸并重算有支撑物体的 z，然后按 evaluate_llm 给 placement 行评分的方式评分。'
             '教师目标 = 完整且通过冻结检查器（geometry_success，原始输出，不用 S1）。目标改写为 P 的训练输出格式'
             '（id、厘米精度的 pos、yaw、on）后重新解析与评分，必须仍然通过，否则丢弃并计数（rounding_broke）。', '',
             f"房间 {summary['rooms']}，教师目标 {summary['targets']}（{summary['rate']:.4f}）；"
             f"分类：{'，'.join(f'{k} {v}' for k, v in summary['classes'].items())}。"
             f"GPU 小时（各分片 cost.json，进程墙钟 × GPU，sacct 为准）：{summary['gpu_hours']}。", '']
    for title, key in (('按集合', 'by_set'), ('按物体数分箱', 'by_bin'), ('按来源', 'by_source')):
        lines += [f'## {title}', '', '| 组 | 房间 | 教师目标 | 通过率 |', '|---|---:|---:|---:|']
        lines += [f"| {k} | {v['rooms']} | {v['targets']} | {v['rate']:.4f} |" for k, v in summary['pass_rate'][key].items()]
        lines.append('')
    return '\n'.join(lines) + '\n'


def teacher(args):
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    plan_ = load_plan(args.plan)
    reference = h.strict_json((Path(args.reference) / 'run_receipt.json').read_text())
    if reference['config'].get('system') != SYSTEM or reference['config'].get('request_arm') != ARM:
        raise ValueError('--reference is not an amendment-016 section 2 O2 run')
    rooms, requests, sets = plan_['rooms'], plan_['requests'], plan_['sets']
    member = {u: name for name in ('T-A', 'T-B') for u in sets[name]}
    tasks, meta, shards, tally = [], [], [], Counter()
    for k, uids in enumerate(plan_['shards']):
        root = shard_run(args.run_root, k)
        original, rows, counts, info = _teacher_run(plan_, root, plan_['root'] / 'shards' / f'{shard_name(k)}.json',
                                                    uids, reference['config'])
        tally.update(counts)
        shards.append({'shard': shard_name(k), **info})
        for before, row in zip(original, rows, strict=True):
            uid = row['uid']
            tasks.append((rooms[uid], requests[uid], row))
            meta.append({'uid': uid, 'source': row['source'], 'group': row['group'], 'set': member[uid],
                         'bin': plan_['cells'][uid][0], 'shard': shard_name(k), 'rounds': row['rounds'],
                         'answer_sha256': L._digest(before['raw']) if before['raw'] is not None else None})
    judged = d._map(judge, tasks, args.workers)
    ledger = [{**m, 'class': j['class'], 'answer': j['answer'], 'target_checks': j['target_checks']}
              for m, j in zip(meta, judged, strict=True)]
    targets = [{**{k: m[k] for k in ('uid', 'source', 'group', 'set', 'bin', 'shard')}, 'prompt': requests[m['uid']]['messages'],
                'target': j['target'], 'verdict': {'complete': True, 'geometry_success': True, 'answer': j['answer'],
                                                   'target': j['target_checks']}, 'answer_sha256': m['answer_sha256']}
               for m, j in zip(meta, judged) if j['class'] == 'target']
    classes = Counter(x['class'] for x in ledger)
    hours = [s['gpu_hours'] for s in shards]
    summary = {'status': 'TEACHER_TARGETS_COMPLETE', 'protocol': PROTOCOL, 'label': 'EXPLORATORY', 'rule': TEACHER_RULE,
               'rooms': len(ledger), 'targets': len(targets), 'rate': len(targets) / len(ledger),
               'classes': {c: classes[c] for c in CLASSES}, 'drops': {'rounding_broke': classes['rounding_broke']},
               'pass_rate': {'by_set': _rates(ledger, 'set'), 'by_bin': _rates(ledger, 'bin'),
                             'by_source': _rates(ledger, 'source')},
               'size_enforcement': dict(sorted(tally.items())), 'shards': shards,
               'gpu_hours': sum(hours) if None not in hours else None,
               'inputs': {'plan_manifest_sha256': h.sha256(plan_['root'] / 'MANIFEST.json'),
                          'cohort_sha256': h.sha256(plan_['root'] / 'cohort.json'),
                          'reference': {'run': str(Path(args.reference).resolve()),
                                        'run_sha256': reference['run_sha256']}},
               'code_sha256': _code()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        (dest / 'targets.jsonl').write_bytes(d._jsonl(targets))
        (dest / 'ledger.jsonl').write_bytes(d._jsonl(ledger))
        h.write_json(dest / 'summary.json', summary)
        (dest / 'report.md').write_text(_report(summary))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return {'status': summary['status'], 'rooms': summary['rooms'], 'targets': summary['targets'],
            'classes': summary['classes'], 'gpu_hours': summary['gpu_hours']}


def load_targets(root):
    """Verified teacher output -> (targets rows, summary)."""
    root = Path(root)
    manifest = e._verified(root)
    summary = h.strict_json((root / 'summary.json').read_text())
    if summary.get('status') != 'TEACHER_TARGETS_COMPLETE':
        raise ValueError(f'{root} is not a complete teacher filter')
    return h._rows(root, 'targets.jsonl', manifest['files']['targets.jsonl']), summary


# ---- shard time (submit_018_teacher.sh) ----------------------------------------------------------

def teacher_minutes(pilot, pilot_data, ta, shards, tb=ROOMS['T-B']):
    """TIME_RULE -> {minutes, basis}; pilot = the amendment-016 section 2 O2 run on pilot_data (eval-data-v1)."""
    pilot, pilot_data = Path(pilot), Path(pilot_data)
    cost, result, receipt = (h.strict_json((pilot / f'{name}.json').read_text())
                             for name in ('cost', 'result', 'run_receipt'))
    if (result['status'] != 'LLM_COMPLETE' or cost['system'] != SYSTEM or receipt['config'].get('request_arm') != ARM
            or receipt['data']['manifest_sha256'] != h.sha256(pilot_data / 'MANIFEST.json')):
        raise ValueError(f'{pilot} is not a complete O2 P-input run on {pilot_data}')
    def objects(root):
        return {r['uid']: len(r['objects']) for r in map(h.strict_json, (root / 'rooms.jsonl').read_text().splitlines())}
    seen, tokens = objects(pilot_data), {b: [] for b in BINS}
    for path in (pilot / 'receipts').glob('*.json'):
        value = h.strict_json(path.read_text())
        tokens[bin_of(seen[value['uid']])].append(sum(a['usage']['output_tokens'] for a in value['answers']))
    if sum(map(len, tokens.values())) != result['rooms'] or not all(tokens.values()):
        raise ValueError(f'{pilot} receipts must cover its rooms and every object-count bin')
    per_token = cost['generation_seconds'] / sum(map(sum, tokens.values()))
    mean = {b: sum(v) / len(v) for b, v in tokens.items()}
    ta_bins = Counter(bin_of(n) for n in objects(Path(ta)).values())
    need = {b: ta_bins[b] + q for b, q in bin_quotas(tb).items()}
    shard = {b: -(-n // shards) for b, n in need.items()}
    load = cost['wall_seconds'] - cost['generation_seconds']
    return {'minutes': math.ceil(1.5 * (load + per_token * sum(shard[b] * mean[b] for b in BINS)) / 60),
            'rule': TIME_RULE, 'load_seconds': load, 'seconds_per_output_token': per_token,
            'pilot_rooms': {b: len(v) for b, v in tokens.items()}, 'pilot_mean_output_tokens': mean,
            'teacher_rooms': need, 'shard_rooms_max': shard}


# ---- cli -----------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('plan')
    for flag in ('data', 'exclude', 'ta', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--sft-runs', type=Path, nargs=2, required=True, metavar=('SFT_JOINT', 'SFT_PLACEMENT'))
    for flag in ('dev', 'test'):
        step.add_argument('--' + flag, type=Path, nargs='+', required=True)
    step.add_argument('--tb', type=int, default=ROOMS['T-B'])
    step.add_argument('--tc', type=int, default=ROOMS['T-C'])
    step.add_argument('--shards', type=int, default=SHARDS)
    step = sub.add_parser('teacher')
    for flag in ('plan', 'run-root', 'reference', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    return parser.parse_args(argv)


STEPS = {'plan': plan, 'teacher': teacher}  # Other amendment-018 steps add their own entries.


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(STEPS[args.command](args)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
