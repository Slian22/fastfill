"""Amendment 026 + addendum A: quality versus big-model use (local-first escalation vs the big model for every room).

Spec: docs/research/20261013-unknown-size-comparison/amendment-026-quality-cost-frontier-20261015.md. Analysis only:
no model, API or GPU call; every number comes from existing outputs. EXPLORATORY.

  python -m fastfill.comparison.frontier_026 --run-root RUN --data EVAL_DATA --pilot a016-pilot-v1.json \
      --cohort llm-cohort-v1.json --out OUT [--results DIR] [--usage-a LLM_RUN] \
      [--uids-c RUN/a026-diverse-v1.json --big-c-dir S1_DIR [--usage-c LLM_RUN]] [--workers N] [--repeats 2000]

Panels. The local arm is always the 27B test rows + S1 (RUN/s1-022/placement-arm, the amendment-022 placer rows, cap
rule on), restricted to the panel's rooms; big rows are scored as amendment 022 scores agent rows (evaluate.stage_views,
no cap rule); every view is the +S1 stage.
  A          the 300 pilot rooms (--pilot); big = L size-fixed harness (RUN/s1-016-size-fixed/agent-L-P-pilot-harness)
  B-raw      the 1,454 cohort rooms (--cohort); big = O2 size-fixed single answer (.../agent-O2-P-raw)
  B-harness  the same rooms; big = O2 size-fixed harness (.../agent-O2-P-harness)
  C          addendum A, only with --uids-c and --big-c-dir: 300 family-stratified test rooms; big = L size-fixed
             harness, --big-c-dir = the S1 dir of its llm_harness rows; usage from --usage-c (its llm_agent run dir)

Policies (per room): LOCAL; ESC-CHECK (escalate when local + S1 fails the frozen checker: the 022 rule); ESC-STRICT
(fails the checker or strict admission); ESC-ANY (fails the checker, strict or required = {} admission); BIG (the big
answer for every room). An escalated room takes the big answer only if it ranks strictly higher by (checker pass,
strict pass, required = {} pass, fewer checker violations = best_of_n.VIOLATIONS of the S1 geometry, incomplete
worst); otherwise the local layout is kept.

Cost (paid for every escalated room whether or not its answer is used; per-room means over the panel's rooms):
  A, C  calls and input / output / reasoning tokens from per-room llm_agent receipts (--usage-a / --usage-c). Without
        them, A uses its run totals (RUN/llm-eval-L-P-pilot-v1/summary.json llm.result, bound to the big S1 rows by
        that dir's MANIFEST; basis run_mean): calls split equally (exact when every room took exactly one call);
        tokens of the rooms SYSTEM-022 sent to gpt-6.1-sol with a single answer are that answer's receipt (same model,
        effort, input mode, suffix and parser, checked against both configs: input tokens should equal the pilot's,
        output / reasoning are another call's draw), the rest of the run total is split equally over the other rooms.
        Per-room sums over all rooms (BIG) are therefore the measured run total; any other escalated subset is an
        estimate (marked ≈), likely low for hard rooms priced at the remainder mean. C without --usage-c: not measured.
  B     O2 calls and tokens from RUN/llm-open-O2-P-001 receipts (raw: the first answer; harness: every answer). Wall
        time: each llm.chat call (one chunk-round; invocations.jsonl wall_seconds) is split over the answers it
        generated in proportion to their output tokens. An answer is matched to its call by its recorded
        latency_seconds, which llm_open writes as that call's wall / requests (unique per call; the answers must cover
        every call exactly). GPU seconds = wall x the run's GPUs (cost.json, TP 2). Model load and process overhead
        (cost.json wall_seconds - generation_seconds) are not attributed to rooms and are reported separately;
        wall_equal_s (the recorded equal split, sum of latency_seconds) is a sensitivity. Attributed from the all-room
        run, not measured on an escalation-only run.
  local the 27B's own GPU time per room, LOCAL_GPU_S (59 min / 8,647 rooms on 1 GPU: the amendment-021 test inference,
        job 7425973; the figure is amendment 026 §3's, not re-measured here); every policy but BIG pays it.

Panel A also scores the delivered 022 cohort system's actual outputs on its rooms (RUN/system-022/final/cohort/system:
evaluate.load_pipeline / final_view) as SYSTEM-022, a check of the ESC-CHECK simulation (its gpt-6.1-sol calls and
tokens from its receipts, and its O2 fallback: output tokens and token-attributed wall / GPU seconds of its o2 run).

Reported per panel: shares escalated / big used, cost, levels (evaluate.level) on every scale_curve.subsets stratum
(all, object-count bins, real / synthetic, room families), contrasts entry - BIG (evaluate.contrast: report_v2.contrasts,
house-cluster bootstrap, seed 42, no CI below report_v2.NO_CI components) on every stratum, and on A SYSTEM-022 -
ESC-CHECK. -> OUT (write-once): results.json, levels.csv, contrasts.csv, report.md (Chinese), frontier.png,
MANIFEST.json; --results DIR receives a copy (write-once too).
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import llm_agent as L
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import scale_curve as sc
from fastfill.comparison import test_apply_012 as t12
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.best_of_n import VIOLATIONS
from fastfill.comparison.corpus import _publish
from fastfill.system import evaluate as ev

SPEC = 'docs/research/20261013-unknown-size-comparison/amendment-026-quality-cost-frontier-20261015.md'
POLICIES = ('LOCAL', 'ESC-CHECK', 'ESC-STRICT', 'ESC-ANY', 'BIG')
TRIGGER = {'ESC-CHECK': ('checker',), 'ESC-STRICT': ('checker', 'strict'), 'ESC-ANY': ('checker', 'strict', 'noreq')}
SYSTEM = 'SYSTEM-022'
TOKENS = ev.TOKENS
LOCAL_GPU_S = 59 * 60 / 8647
LOCAL_GPU_SOURCE = ('amendment 026 §3: the 27B test inference of amendment 021 (job 7425973), 8,647 rooms in 59 min on '
                    '1 GPU; not re-measured here')
API_COST = ('calls', *TOKENS)                  # cost columns of the API panels (A, C)
GPU_COST = ('output_tokens', 'wall_s', 'gpu_s')  # cost columns of the O2 panels (B)
NOT_MEASURED = ev.NOT_MEASURED
SAME_CALL = ('model', 'reasoning_effort', 'input', 'suffix', 'parser', 'max_output_tokens')   # pilot vs 022 gpt step
RATE_KEYS = ('checker_rate', 'strict_rate', 'noreq_rate')


# ---- the simulation ------------------------------------------------------------------------------

def view(row, room, cap_rule=False):
    """evaluate.stage_views' +S1 view plus the S1 layout's checker violations (None when incomplete)."""
    v = ev.stage_views(row, room, cap_rule)['s1']
    geo = row['s1'].get('geometry_s1')
    return {**v, 'violations': sum(geo[k] for k in VIOLATIONS) if v['complete'] and geo else None}


def rank(v):
    """The pre-registered lexicographic rule: checker, strict, required = {}, fewer violations (incomplete worst)."""
    return (bool(v['checker']), bool(v['strict']), bool(v['noreq']),
            -math.inf if v['violations'] is None else -v['violations'])


def escalates(policy, local):
    if policy in ('LOCAL', 'BIG'):
        return policy == 'BIG'
    return not all(local[k] for k in TRIGGER[policy])


def simulate(policy, local, big):
    """[(chosen view, escalated, big used)] per room; local and big are the same rooms in the same order."""
    out = []
    for a, b in zip(local, big, strict=True):
        if a['uid'] != b['uid']:
            raise ValueError('local and big views are not aligned')
        esc = escalates(policy, a)
        used = policy == 'BIG' or (esc and rank(b) > rank(a))
        out.append((b if used else a, esc, used))
    return out


def policy_cost(escalated, usage, n, local_gpu_s):
    """Per-room means (and totals) of the big model's usage over the escalated rooms; local GPU seconds per room."""
    out = {'local_gpu_s': local_gpu_s}
    if usage is None:
        return {**out, 'big': NOT_MEASURED, 'big_total': NOT_MEASURED}
    keys = sorted({k for x in usage.values() for k in x})
    total = {k: sum(usage[u][k] for u in escalated) for k in keys}
    return {**out, 'big': {k: v / n for k, v in total.items()}, 'big_total': total}


# ---- usage ---------------------------------------------------------------------------------------

def receipt_usage(run, variant, wall=False):
    """({uid: calls and tokens [, wall_s, wall_equal_s, gpu_s]}, meta) from an llm_agent / llm_open run's receipts;
    raw = the first answer only, harness = every answer. wall: the llm_open token-proportional attribution."""
    run = Path(run)
    result = h.strict_json((run / 'result.json').read_text())
    receipts = [h.strict_json(p.read_text()) for p in sorted((run / 'receipts').glob('*.json'))]
    if not receipts or any(x['run_sha256'] != result['run_sha256'] for x in receipts):
        raise ValueError(f'{run}: no receipts, or receipts of another run')
    usage, valid = {}, {}
    for x in receipts:
        answers = x['answers'][:1] if variant == 'raw' else x['answers']
        usage[x['uid']] = {'calls': sum(a.get('tries', 1) for a in answers),
                           **{k: sum((a.get('usage') or {}).get(k, 0) for a in answers) for k in TOKENS}}
        valid[x['uid']] = bool(answers) and answers[-1]['schema']['valid']
    meta = {'basis': 'per-room receipts', 'path': str(run), 'run_sha256': result['run_sha256'], 'variant': variant,
            'valid': valid}
    if wall:
        cost = h.strict_json((run / 'cost.json').read_text())
        meta.update(attribute_wall(run, receipts, usage, variant, cost['gpus']),
                    slurm_job_ids=cost['slurm_job_ids'], process_wall_s=cost['wall_seconds'],
                    generation_s=cost['generation_seconds'],
                    unattributed_s=cost['wall_seconds'] - cost['generation_seconds'])
    return usage, meta


def attribute_wall(run, receipts, usage, variant, gpus):
    """Adds wall_s (token-proportional share of each call), wall_equal_s and gpu_s (wall_s x gpus) to usage; returns
    run-level meta (jobs_without_calls: Slurm jobs of the ledger that never finished a call)."""
    events = L._jsonl(run / 'invocations.jsonl')
    calls = [x for x in events if x['event'] == 'call']
    index = {}
    for i, c in enumerate(calls):
        q = c['wall_seconds'] / c['requests']
        if q in index:
            raise ValueError('two calls share a per-request latency: attribution is ambiguous')
        index[q] = i
    members = collections.defaultdict(list)
    for x in receipts:
        for k, a in enumerate(x['answers']):
            if a['latency_seconds'] not in index:
                raise ValueError(f"{x['uid']}: answer {k} matches no call in invocations.jsonl")
            members[index[a['latency_seconds']]].append((x['uid'], k, a['usage']['output_tokens']))
    if sorted(members) != list(range(len(calls))) or any(len(members[i]) != c['requests'] for i, c in enumerate(calls)):
        raise ValueError('the receipts do not cover the call ledger exactly')
    for x in usage.values():
        x.update(wall_s=0., wall_equal_s=0.)
    for i, group in members.items():
        wall, total = calls[i]['wall_seconds'], sum(t for _, _, t in group)
        for uid, k, tokens in group:
            if variant == 'raw' and k:
                continue
            usage[uid]['wall_s'] += wall * (tokens / total if total else 1 / len(group))
            usage[uid]['wall_equal_s'] += wall / len(group)
    for x in usage.values():
        x['gpu_s'] = x['wall_s'] * gpus
    jobs = lambda xs: {x['slurm_job_id'] for x in xs if 'slurm_job_id' in x}
    return {'gpus': gpus, 'llm_calls': len(calls), 'jobs_without_calls': sorted(jobs(events) - jobs(calls)),
            'rule': 'each llm.chat call (chunk-round) wall split over its answers by output tokens; load and '
                    'process overhead not attributed'}


def summary_usage(path, uids, known=None):
    """Run totals of an llm_agent run (an evaluate_llm summary.json llm.result) over its rooms: calls split equally;
    tokens of the rooms in KNOWN ({uid: tokens} of single answers of another run on the same rooms and prompt) taken
    from it, the rest of the run total split equally over the other rooms (the sum over all rooms stays the run's)."""
    r = h.strict_json(Path(path).read_text())['llm']['result']
    n = r['rooms']
    if r['status'] != 'LLM_COMPLETE' or n != len(uids) or r['api_failed_rooms']:
        raise ValueError(f'{path}: not a complete run over these {len(uids)} rooms')
    known = {u: known[u] for u in uids if u in (known or {})}
    if len(known) == n:
        raise ValueError('every room known: use per-room receipts instead')
    left = {k: (r['tokens'][k] - sum(x[k] for x in known.values())) / (n - len(known)) for k in TOKENS}
    if min(left.values()) < 0:
        raise ValueError('the known rooms exceed the run total')
    return {u: {'calls': r['calls_ledger'] / n, **known.get(u, left)} for u in uids}, {
        'basis': 'run_mean', 'path': str(path), 'run_sha256': r['run_sha256'], 'run_tokens': r['tokens'],
        'rooms': n, 'calls_ledger': r['calls_ledger'], 'rounds_histogram': r['rounds_histogram'],
        'calls_exact': r['rounds_histogram'] == {'1': n} and r['calls_ledger'] == n, 'known_rooms': len(known),
        'remainder_per_room': left,
        'note': 'per-room receipts not available: tokens of known rooms from another run, the rest = run remainder '
                'split equally; exact only summed over all rooms'}


# ---- one panel -----------------------------------------------------------------------------------

def panel(name, scope, big_label, kind, rooms, local_rows, big_rows, usage=None, usage_meta=None, system=None,
          repeats=2000, workers=1):
    """rooms {uid: room}; local_rows / big_rows: S1 rows covering exactly those rooms; usage {uid: numbers} of the big
    model (None = not measured); system: {uid: view, ...} entry of SYSTEM-022 (system_entry) or None."""
    uids = sorted(rooms)
    lb, bb = h._by_uid(local_rows), h._by_uid(big_rows)
    if sorted(lb) != uids or sorted(bb) != uids:
        raise ValueError(f'{name}: local and big rows must cover exactly the panel rooms')
    if usage is not None and not set(uids) <= set(usage):
        raise ValueError(f'{name}: big-model usage misses rooms')
    local = [view(lb[u], rooms[u], cap_rule=True) for u in uids]
    big = [view(bb[u], rooms[u]) for u in uids]
    entries = {}
    for p in POLICIES:
        sim = simulate(p, local, big)
        escalated = {u for u, (_, x, _) in zip(uids, sim) if x}
        entries[p] = {'views': [v for v, _, _ in sim], 'escalated': escalated,
                      'used': {u for u, (_, _, x) in zip(uids, sim) if x},
                      'cost': policy_cost(escalated, usage, len(uids), 0. if p == 'BIG' else LOCAL_GPU_S)}
    if system:
        entries[SYSTEM] = {**system, 'views': [system['views'][u] for u in uids]}
    subsets = {k: {x['uid'] for x in v} for k, v in sc.subsets([rooms[u] for u in uids]).items()}
    levels = {k: {p: {**ev.level([v for v in x['views'] if v['uid'] in keep]),
                      'escalated_share': len(x['escalated'] & keep) / len(keep),
                      'big_used_share': len(x['used'] & keep) / len(keep)} for p, x in entries.items()}
              for k, keep in subsets.items()}
    pick = lambda views, keep: [v for v in views if v['uid'] in keep]
    tasks, meta = [], []
    for k, keep in subsets.items():
        for p in (x for x in entries if x != 'BIG'):
            tasks.append((pick(entries[p]['views'], keep), pick(entries['BIG']['views'], keep), repeats))
            meta.append({'subset': k, 'comparison': f'{p} - BIG'})
    if system:
        tasks.append((entries[SYSTEM]['views'], entries['ESC-CHECK']['views'], repeats))
        meta.append({'subset': 'all', 'comparison': f'{SYSTEM} - ESC-CHECK'})
    contrasts = [{**m, 'rooms': len(t[0]), **c} for m, t, c in zip(meta, tasks, r2._map(ev.contrast, tasks, workers))]
    out = {'panel': name, 'scope': scope, 'big': big_label, 'cost_kind': kind, 'rooms': len(uids),
           'usage': {k: v for k, v in (usage_meta or {}).items() if k != 'valid'} or NOT_MEASURED,
           'entries': {p: {'escalated': len(x['escalated']), 'big_used': len(x['used']), 'cost': x['cost'],
                           **({k: x[k] for k in ('o2_fallback_rooms', 'stages', 'gpt_receipts')} if p == SYSTEM
                              else {})}
                       for p, x in entries.items()},
           'levels': levels, 'contrasts': contrasts}
    out['esc_any_equals_esc_strict'] = entries['ESC-ANY']['escalated'] == entries['ESC-STRICT']['escalated']
    if usage_meta and 'valid' in usage_meta:    # do the receipts belong to these rows? (complete <=> parsed answer)
        out['usage_rows_disagree'] = sum(bool(bb[u]['complete']) != usage_meta['valid'][u] for u in uids)
    if system:
        sim = entries['ESC-CHECK']
        by = lambda key: {v['uid']: v[key] for v in sim['views']}
        sysv = {v['uid']: v for v in entries[SYSTEM]['views']}
        out['system_check'] = {
            'escalated_same_set': system['escalated'] == sim['escalated'],
            'escalated_system_only': len(system['escalated'] - sim['escalated']),
            'escalated_simulation_only': len(sim['escalated'] - system['escalated']),
            'big_used_both': len(system['used'] & sim['used']),
            'big_used_system_only': len(system['used'] - sim['used']),
            'big_used_simulation_only': len(sim['used'] - system['used']),
            **{f'{k}_disagree': sum(bool(sysv[u][k]) != bool(by(k)[u]) for u in uids) for k in ('checker', 'strict', 'noreq')}}
    return out


def system_entry(path, rooms, requests, uids, workers=1):
    """SYSTEM-022 on these rooms: final layouts scored by evaluate.final_view; escalated = any escalation step; big
    used = final stage not the placer; cost = its gpt-6.1-sol calls / tokens from its receipts (+ 27B GPU) and its O2
    fallback (o2: output tokens, wall / GPU seconds attributed as scope B, per panel room); one_answer = {uid: tokens}
    of the rooms whose gpt receipt holds a single answer (no repair round)."""
    path = Path(path)
    identity, rows = ev.load_pipeline(path)
    by = {r['uid']: r for r in rows}
    if not set(uids) <= set(by):
        raise ValueError(f'{path}: the system run lacks rooms of this panel')
    views = r2._map(ev.final_view, [(rooms[u], requests[u], by[u]['layout']) for u in uids], workers)
    gpt = lambda u: [s for s in by[u]['escalation'] if s['step'] == 'gpt']
    usage = {u: {'calls': sum(s['calls'] for s in gpt(u)), **{k: sum(s['usage'][k] for s in gpt(u)) for k in TOKENS}}
             for u in uids}
    escalated = {u for u in uids if by[u]['escalation']}
    keep = set(uids)
    receipts = [x for x in map(lambda q: h.strict_json(q.read_text()), sorted((path / 'api' / 'receipts').glob('*.json')))
                if x['uid'] in keep]
    cost = policy_cost(escalated, usage, len(uids), LOCAL_GPU_S)
    if (path / 'o2' / 'receipts.jsonl').exists():
        o2 = [h.strict_json(x) for x in (path / 'o2' / 'receipts.jsonl').read_text().splitlines()]
        o2_usage = {x['uid']: {'answers': len(x['answers']),
                               'output_tokens': sum(a['usage']['output_tokens'] for a in x['answers'])} for x in o2}
        gpus = h.strict_json((path / 'result.json').read_text())['escalation']['o2']['engine']['tensor_parallel_size']
        attribute_wall(path / 'o2', o2, o2_usage, 'harness', gpus)
        c = policy_cost(set(o2_usage) & keep, o2_usage, len(uids), 0.)
        cost.update(o2=c['big'], o2_total={**c['big_total'], 'rooms': len(set(o2_usage) & keep), 'gpus': gpus})
    return {'views': dict(zip(uids, views)), 'escalated': escalated, 'used': {u for u in uids if by[u]['stage'] != 'placer'},
            'cost': cost,
            'gpt_receipts': {'rooms': len(receipts), 'answers': sum(len(x['answers']) for x in receipts),
                             'calls': sum(usage[u]['calls'] for u in uids),
                             'rounds': dict(collections.Counter(str(x['rounds']) for x in receipts)),
                             'http_errors': dict(collections.Counter(e for x in receipts for e in x['http_errors']))},
            'one_answer': {x['uid']: {k: x['answers'][0]['usage'][k] for k in TOKENS} for x in receipts
                           if len(x['answers']) == 1 and x['answers'][0].get('usage')},
            'o2_fallback_rooms': sum(any(s['step'] == 'o2' for s in by[u]['escalation']) for u in uids),
            'stages': dict(collections.Counter(by[u]['stage'] for u in uids)),
            'identity': {k: identity[k] for k in ('result_sha256', 'rooms_sha256', 'escalate', 'api_calls')}}


# ---- loading -------------------------------------------------------------------------------------

def load_uids(path, data_sha):
    value = h.strict_json(Path(path).read_text())
    uids = value['uids']
    if ('uids_sha256' in value and L._digest(json.dumps(uids)) != value['uids_sha256']) or (
            value.get('data_manifest_sha256', data_sha) != data_sha) or len(set(uids)) != len(uids):
        raise ValueError(f'{path}: uid list does not match its hash, this data, or has duplicates')
    return uids


def s1_rows(path, geometry, data_sha, uids, uids_file=None):
    """The S1 rows of PATH (t12._system: complete S1 on this data at the protocol settings) restricted to uids."""
    path = Path(path)
    rows, info = t12._system(path, 'placement', geometry, data_sha)
    bound = info['inputs'].get('uids')
    text = Path(uids_file).read_bytes() if uids_file is not None else b''
    same = {h.sha256(uids_file), hashlib.sha256(text.rstrip(b'\n') + b'\n').hexdigest()} if text else set()
    if uids_file is not None and bound and bound['sha256'] not in same:   # a local copy may lack the final newline
        raise ValueError(f'{path}: S1 rows of another uid list than {uids_file}')
    keep = set(uids)
    return [x for x in rows if x['uid'] in keep], {
        'path': str(path), 'MANIFEST': h.sha256(path / 'MANIFEST.json'),
        'geometry_manifest_sha256': info['inputs']['geometry_manifest_sha256']}


def build(args):
    run, data = Path(args.run_root), Path(args.data)
    manifest, data_sha = e._verified(data), h.sha256(data / 'MANIFEST.json')
    scopes = {'A': load_uids(args.pilot, data_sha), 'B': load_uids(args.cohort, data_sha)}
    if args.uids_c:
        scopes['C'] = load_uids(args.uids_c, data_sha)
    want = set().union(*scopes.values())
    rows = lambda name: {u: x for u, x in h._by_uid(h._rows(data, name, manifest['files'][name])).items() if u in want}
    rooms, requests = rows('rooms.jsonl'), rows('placement/requests.jsonl')
    if set(rooms) != want:
        raise ValueError('scope uids outside the test data')
    local_dir = run / 's1-022' / 'placement-arm'
    local, local_in = s1_rows(local_dir, 'placement-geometry.jsonl', data_sha, want)
    inputs, missing, panels = {'data_MANIFEST': data_sha, 'local': local_in}, [], []
    of = lambda uids: {u: rooms[u] for u in uids}
    # A: L size-fixed harness on the pilot rooms (+ SYSTEM-022 check)
    a_dir = run / 's1-016-size-fixed' / 'agent-L-P-pilot-harness'
    big, inputs['A'] = s1_rows(a_dir, 'llm_harness-geometry.jsonl', data_sha, scopes['A'], args.pilot)
    system_dir = run / 'system-022' / 'final' / 'cohort' / 'system'
    system = None
    if (system_dir / 'result.json').exists():
        system = system_entry(system_dir, rooms, requests, sorted(scopes['A']), args.workers)
        inputs['system_022'] = {'path': str(system_dir), **system['identity']}
    else:
        missing.append(f'A: {system_dir} (SYSTEM-022 check not measured)')
    if args.usage_a:
        usage, umeta = receipt_usage(args.usage_a, 'harness')
    else:
        evald = run / 'llm-eval-L-P-pilot-v1'
        if h.sha256(evald / 'MANIFEST.json') != inputs['A']['geometry_manifest_sha256']:
            raise ValueError(f'{evald}: not the evaluation the A big S1 rows repaired')
        config = h.strict_json((evald / 'summary.json').read_text())['llm']['config']
        api = h.strict_json((system_dir / 'result.json').read_text())['escalation']['api'] if system else {}
        same = bool(api) and all(config[k] == api[k] for k in SAME_CALL)
        usage, umeta = summary_usage(evald / 'summary.json', scopes['A'], system['one_answer'] if same else None)
        umeta.update(known_source={'run_sha256': api['run_sha256'], 'same_call': SAME_CALL} if same else None)
        missing.append('A: per-room gpt-6.1-sol receipts of the L size-fixed pilot run (run_sha256 '
                       f"{umeta['run_sha256'][:12]}) are not local; tokens of escalated rooms are estimates "
                       f"({umeta['known_rooms']} rooms from SYSTEM-022 receipts, the rest = run remainder mean)")
    panels.append(panel('A', 'A', 'L size-fixed harness (gpt-6.1-sol)', 'api', of(scopes['A']),
                        [x for x in local if x['uid'] in set(scopes['A'])], big, usage, umeta, system,
                        args.repeats, args.workers))
    # B: O2 size-fixed raw and harness on the cohort
    o2_run = run / 'llm-open-O2-P-001'
    for v in ('raw', 'harness'):
        big, inputs[f'B-{v}'] = s1_rows(run / 's1-016-size-fixed' / f'agent-O2-P-{v}', f'llm_{v}-geometry.jsonl',
                                        data_sha, scopes['B'], args.cohort)
        usage, umeta = receipt_usage(o2_run, v, wall=True) if (o2_run / 'result.json').exists() else (None, None)
        if usage is None and v == 'raw':
            missing.append(f'B: {o2_run} (O2 cost not measured)')
        panels.append(panel(f'B-{v}', 'B', f'O2 size-fixed {v} (Qwen3.8-27B thinking)', 'gpu', of(scopes['B']),
                            [x for x in local if x['uid'] in set(scopes['B'])], big, usage, umeta, None,
                            args.repeats, args.workers))
    # C: addendum A, L size-fixed harness on the family-stratified rooms
    if args.uids_c:
        big, inputs['C'] = s1_rows(args.big_c_dir, 'llm_harness-geometry.jsonl', data_sha, scopes['C'], args.uids_c)
        usage, umeta = receipt_usage(args.usage_c, 'harness') if args.usage_c else (None, None)
        if usage is None:
            missing.append('C: --usage-c (per-room receipts of the scope-C L run): cost not measured')
        panels.append(panel('C', 'C', 'L size-fixed harness (gpt-6.1-sol)', 'api', of(scopes['C']),
                            [x for x in local if x['uid'] in set(scopes['C'])], big, usage, umeta, None,
                            args.repeats, args.workers))
    else:
        missing.append('C: scope-C L outputs not given (--uids-c / --big-c-dir); not run')
    return {'status': 'FRONTIER_026_COMPLETE', 'protocol': 'amendment 026 (+ addendum A)', 'spec': SPEC,
            'exploratory': True, 'panels': panels, 'inputs': inputs, 'missing': missing,
            'local_gpu': {'seconds_per_room': LOCAL_GPU_S, 'source': LOCAL_GPU_SOURCE},
            'settings': {'repeats': args.repeats, 'seed': r2.SEED, 'no_ci_below_components': r2.NO_CI},
            'rules': {
                'policies': {'LOCAL': 'no big-model call', **{p: 'escalate when local + S1 fails ' + ' or '.join(k)
                                                              for p, k in TRIGGER.items()}, 'BIG': 'big for every room'},
                'choice': 'escalated room takes the big answer only if (checker, strict, required = {}, -violations) '
                          'is strictly higher; else local',
                'cost': 'paid for every escalated room, used or not; per-room means over the panel rooms',
                'simulation': "the escalated answer is the big model's existing amendment-016 answer (+ S1), not a "
                              "fresh call",
                'local_rows': 'evaluate.stage_views +S1 with the cap rule; big rows without (as 022 scores agents)'},
            'code_sha256': {Path(f).name: h.sha256(f) for f in (__file__, ev.__file__, r2.__file__, sc.__file__)}}


# ---- output --------------------------------------------------------------------------------------

LEVEL_COLS = ('requests', 'complete_rooms', 'escalated_share', 'big_used_share', 'complete_rate', *RATE_KEYS,
              *ev.MEANS)


def _csv(dest, result):
    with (dest / 'levels.csv').open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['panel', 'scope', 'big', 'subset', 'entry', *LEVEL_COLS])
        for p in result['panels']:
            for subset, entries in p['levels'].items():
                for name, x in entries.items():
                    w.writerow([p['panel'], p['scope'], p['big'], subset, name, *(x[k] for k in LEVEL_COLS)])
    with (dest / 'contrasts.csv').open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['panel', 'scope', 'big', 'subset', 'comparison', 'rooms', 'components', 'field', 'estimate',
                    'ci_low', 'ci_high', 'p'])
        for p in result['panels']:
            for c in p['contrasts']:
                for field, x in c['fields'].items():
                    w.writerow([p['panel'], p['scope'], p['big'], c['subset'], c['comparison'], c['rooms'],
                                c['components'], field, x['estimate'], *(x['ci95'] or [None, None]), x['p']])


def _contrast(p, entry, subset='all', other='BIG'):
    return next((c['fields'] for c in p['contrasts'] if c['subset'] == subset and c['comparison'] == f'{entry} - {other}'),
                {})


def _cost_cells(p, entry, o2=False):
    """Cost cells; ≈ marks a run_mean token estimate (a strict, non-empty subset of the rooms: the sum over all rooms is
    the measured run total). o2: SYSTEM-022's O2 fallback columns (0 for the simulated policies, which have none)."""
    cost = p['entries'][entry]['cost']
    cols = API_COST if p['cost_kind'] == 'api' else GPU_COST
    if isinstance(cost['big'], str):
        cells = ['未测量'] * len(cols)
    else:
        approx = (p['usage'] != NOT_MEASURED and p['usage'].get('basis') == 'run_mean' and entry != SYSTEM
                  and 0 < p['entries'][entry]['escalated'] < p['rooms'])
        exact = lambda k: k == 'calls' and p['usage'].get('calls_exact')
        fmt = lambda k, x: (f'{x:.3f}' if k == 'calls' else f'{x:,.0f}' if 'tokens' in k else f'{x:.1f}')
        cells = [('≈' if approx and not exact(k) and cost['big'][k] else '') + fmt(k, cost['big'][k]) for k in cols]
    if o2:
        x = cost.get('o2', {})
        cells += [f"{x.get('output_tokens', 0):,.0f}", f"{x.get('gpu_s', 0):.1f}"]
    return cells + [f"{cost['local_gpu_s']:.3f}"]


COST_HEAD = {'api': ('调用/间', '输入 tok/间', '输出 tok/间', '推理 tok/间'),
             'gpu': ('O2 输出 tok/间', 'O2 墙钟 s/间（分摊）', 'O2 GPU·s/间（分摊）')}
O2_HEAD = ('O2 兜底输出 tok/间', 'O2 兜底 GPU·s/间（分摊）')
SUBSET_ZH = {'all': '全部', 'objects<=20': '≤20 物体', 'objects21-50': '21–50 物体', 'objects>50': '>50 物体',
             'real': '真实/策展来源', 'synthetic': '合成来源', 'family:paper_families': '族：论文四族',
             'family:other_families': '族：其余有标注族'}


def _main_table(panels):
    o2 = any('o2' in x['cost'] for p in panels for x in p['entries'].values())
    head = ('大模型', '策略', '升级', '用大模型', *COST_HEAD[panels[0]['cost_kind']], *(O2_HEAD if o2 else ()),
            '27B GPU·s/间', '检查器', '严格准入',
            'required={}', 'OOR', '尺寸 log MAE', 'B', 'Δ检查器 vs BIG', 'Δ严格 vs BIG', 'Δrequired={} vs BIG')
    lines = ['| ' + ' | '.join(head) + ' |', '|---|---|' + '---:|' * (len(head) - 2)]
    for p in panels:
        for entry in p['entries']:
            x, d = p['levels']['all'][entry], _contrast(p, entry)
            lines.append(f"| {p['big']} | {entry} | {ev._f(x['escalated_share'])} | {ev._f(x['big_used_share'])} | "
                         + ' | '.join(_cost_cells(p, entry, o2)) + ' | '
                         + ' | '.join(ev._f(x[k]) for k in (*RATE_KEYS, 'oor', 'size_log_mae', 'bias_fp')) + ' | '
                         + ' | '.join(ev._ci(d.get(k)) if entry != 'BIG' else '—' for k in ('checker', 'strict', 'noreq'))
                         + ' |')
    return lines


def _attributed(p):
    """Scope B: the attributed wall time is not an escalation-only measurement; its equal-split sensitivity."""
    lv, e = p['levels']['all'], p['entries']
    w = lambda q, k='wall_s': e[q]['cost']['big'][k]
    return ('墙钟与 GPU·s 是从全量运行（每块 256 间）分摊得到的，不是只跑升级房的实测；只跑升级房时总时长受最慢答案支配，'
            f"未测量。ESC-CHECK 的升级房（占 {lv['ESC-CHECK']['escalated_share']:.1%}）分得 {w('ESC-CHECK') / w('BIG'):.0%}"
            f" 的分摊墙钟。等分敏感性（wall_equal_s）：ESC-CHECK {w('ESC-CHECK', 'wall_equal_s'):.2f} s/间（表中 "
            f"{w('ESC-CHECK'):.2f}），ESC-STRICT {w('ESC-STRICT', 'wall_equal_s'):.2f}（表中 {w('ESC-STRICT'):.2f}），"
            f"BIG {w('BIG', 'wall_equal_s'):.2f}（表中 {w('BIG'):.2f}）。")


def _system_cost(s):
    """SYSTEM-022's cost row: what its receipts show (rounds, transport retries) and its O2 fallback."""
    g, o = s['gpt_receipts'], s['cost'].get('o2_total')
    return (f"系统的成本行是其实际用量。gpt-6.1-sol：{g['rooms']} 间、{g['answers']} 个回答（receipts rounds "
            f"{json.dumps(g['rounds'])}：022 允许的 1 轮修复在这些房间上没有发生），{g['calls']} 次调用，其中 "
            f"{g['calls'] - g['answers']} 次是传输重试（{json.dumps(g['http_errors'])}；失败尝试的用量未记录）。"
            + (f"O2 兜底：{o['rooms']} 间、{o['answers']} 个回答、输出 {o['output_tokens']:,} tok、按 token 分摊的墙钟 "
               f"{o['wall_s']:.0f} s × {o['gpus']} 张 GPU（等分 {o['wall_equal_s']:.0f} s），表中 O2 兜底列为每间均值；"
               '“用大模型”含最终取 O2 布局的房间。' if o else 'O2 兜底：本地无记录，未计入。'))


def report(result):
    cell = lambda x: f"{ev._f(x['checker_rate'])} / {ev._f(x['strict_rate'])} / {ev._f(x['noreq_rate'])}"
    lines = ['# 修订 026：本地优先升级策略 vs 每间都用大模型（质量–大模型使用前沿，探索性）', '',
             '只做分析：不调用任何模型、API 或 GPU。所有数字来自已有输出。试点房间也用于设计 022，因此全部结果为 EXPLORATORY；'
             '本文件不采纳任何策略，交付系统仍按 022 锁定。', '',
             '**模拟说明。** 升级房间的大模型答案是修订 016 已有的答案（+S1），不是新调用；其成本是该答案的'
             '记录用量（范围 A 的试点逐房用量不在本地，见该节说明）。升级房间只有在大模型答案按（检查器、严格准入、required={}、更少检查器违规）字典序严格更高时才被采用，'
             '否则保留本地布局。成本按“升级即付费”计算（无论是否采用），为该范围全部房间的每间均值。', '',
             '本地臂 = 27B 测试行 + S1（s1-022/placement-arm，含 token 上限规则）；比率的分母为全部请求；OOR、尺寸 log MAE、'
             'B 为完整房间均值；Δ = 策略 − BIG，房屋连通组 bootstrap 95% CI（'
             f"{result['settings']['repeats']} 次，seed {result['settings']['seed']}，未校正）。尺寸固定：两侧都用训练先验"
             '尺寸，同一房间的尺寸相同，配对 Δ尺寸 log MAE 应为 0（contrasts.csv）；表中尺寸均值只取各自的完整房间，'
             '完整率不同时会不同。', '']
    by_scope = collections.defaultdict(list)
    for p in result['panels']:
        by_scope[p['scope']].append(p)
    titles = {'A': '范围 A：300 间试点房，大模型 = gpt-6.1-sol（尺寸固定，harness）',
              'B': '范围 B：1,454 间代理队列房，大模型 = O2（Qwen3.8-27B 思考，尺寸固定），单次与 harness 分列',
              'C': '范围 C（附录 A）：300 间按房间族分层的测试房（每族 20 间），大模型 = gpt-6.1-sol（尺寸固定，harness）'}
    for k, (scope, panels) in enumerate(by_scope.items(), 1):
        lines += [f'## {k}. {titles[scope]}', '', *_main_table(panels), '']
        for p in panels:
            u = p['usage']
            if isinstance(u, str):
                lines.append(f"- {p['big']}：大模型成本未测量。")
            elif u['basis'] == 'run_mean':
                known = (f"其中 {u['known_rooms']} 间（022 交付系统送往 gpt-6.1-sol 的房间）的 token 取自 022 系统在同一房间上"
                         f"的 receipts（run {u['known_source']['run_sha256'][:12]}；同模型、reasoning effort、P 输入方式、后缀"
                         "与解析器，每间 1 个回答：输入 token 应与试点相同，输出/推理 token 是另一次调用的抽样）；其余 "
                         f"{u['rooms'] - u['known_rooms']} 间平分整次运行总量减去这些房间后的余量。"
                         if u['known_rooms'] else '按房间平均分摊。')
                lines.append(f"- {p['big']}：试点的逐房 receipts 不在本地，只有整次运行总量（{u['calls_ledger']} 次调用，tokens "
                             f"{json.dumps(u['run_tokens'])}）。调用：每间 1 次，为精确值（rounds_histogram "
                             f"{json.dumps(u['rounds_histogram'])}）。token：BIG 为整次运行实测总量（精确）；{known}"
                             '带 ≈ 的格为估计。升级房比平均房更难、输出更长，按余量均值计价的升级房（ESC-STRICT/ESC-ANY 中'
                             '不属于上述房间者）的 token 因此很可能偏低（方向可能，但不是严格下界）。')
            else:
                lines.append(f"- {p['big']}：逐房 receipts（run {u['run_sha256'][:12]}）。"
                             + (f"墙钟按 token 分摊：每次 llm.chat 调用（一块的一轮，共 {u['llm_calls']} 次）的墙钟按各答案"
                                f"输出 token 比例分给房间；GPU·s = 墙钟 × {u['gpus']} 张 GPU；模型加载与进程开销 "
                                f"{u['unattributed_s']:.0f} s（进程墙钟 {u['process_wall_s']:.0f} s − 生成 "
                                f"{u['generation_s']:.0f} s）不分摊；作业 {', '.join(u['slurm_job_ids'])}"
                                + (f"（{', '.join(u['jobs_without_calls'])} 只有 start 事件，其后时间未计入 cost.json；"
                                   'sacct 不在本地）' if u['jobs_without_calls'] else '') + '。' + _attributed(p)
                                if 'gpus' in u else ''))
            if p['esc_any_equals_esc_strict']:
                lines.append(f"  {p['big']}：ESC-ANY 与 ESC-STRICT 的升级集合相同（本地 +S1 通过严格准入的房间也都通过 "
                             'required={}），两行数字因此一致。')
            if 'usage_rows_disagree' in p:
                lines.append(f"  receipts 与 S1 行的完整性不一致的房间：{p['usage_rows_disagree']}（应为 0）。")
        if any('system_check' in p for p in panels):
            p = next(p for p in panels if 'system_check' in p)
            c, s = p['system_check'], p['entries'][SYSTEM]
            lines += ['', f"**{SYSTEM} 实际输出核对（ESC-CHECK 模拟）。** 022 交付系统在这些房间上实际升级 "
                      f"{s['escalated']} 间（模拟 {p['entries']['ESC-CHECK']['escalated']} 间，集合相同："
                      f"{'是' if c['escalated_same_set'] else '否'}；仅系统 {c['escalated_system_only']}，仅模拟 "
                      f"{c['escalated_simulation_only']}）；最终采用升级答案 {s['big_used']} 间（模拟 "
                      f"{p['entries']['ESC-CHECK']['big_used']}；两者都采用 {c['big_used_both']}，仅系统 "
                      f"{c['big_used_system_only']}，仅模拟 {c['big_used_simulation_only']}）；最终来源 "
                      f"{json.dumps(s['stages'])}，其中 O2 兜底 {s['o2_fallback_rooms']} 间。逐房判定不一致：检查器 "
                      f"{c['checker_disagree']}、严格 {c['strict_disagree']}、required={{}} {c['noreq_disagree']}。"
                      f"{SYSTEM} − ESC-CHECK：Δ检查器 {ev._ci(_contrast(p, SYSTEM, other='ESC-CHECK').get('checker'))}，"
                      f"Δ严格 {ev._ci(_contrast(p, SYSTEM, other='ESC-CHECK').get('strict'))}。" + _system_cost(s)]
        lines.append('')
    lines += [f'## {len(by_scope) + 1}. 分层（物体数分箱、真实/合成来源、房间族）', '',
              '每格：检查器 / 严格准入 / required={}（全部请求为分母）。“升级”= ESC-CHECK 的升级占比。'
              'Δ = LOCAL − BIG 的检查器差（k < 10 个连通组时不给区间）。其余策略与字段见 levels.csv / contrasts.csv。', '',
              '| 大模型 | 子集 | n | LOCAL | ESC-CHECK | ESC-ANY | BIG | 升级 | Δ检查器 LOCAL−BIG |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for p in result['panels']:
        for subset, x in p['levels'].items():
            name = SUBSET_ZH.get(subset, subset.replace('family:', '族：'))
            lines.append(f"| {p['big']} | {name} | {x['LOCAL']['requests']} | {cell(x['LOCAL'])} | "
                         f"{cell(x['ESC-CHECK'])} | {cell(x['ESC-ANY'])} | {cell(x['BIG'])} | "
                         f"{ev._f(x['ESC-CHECK']['escalated_share'])} | "
                         f"{ev._ci(_contrast(p, 'LOCAL', subset).get('checker'))} |")
    lines += ['', '## 说明', '',
              f"- 27B 本地 GPU 时间：{LOCAL_GPU_S:.3f} GPU·s/间（{LOCAL_GPU_SOURCE}）；BIG 不运行本地模型，记 0。",
              '- 范围 B 的 O2 成本：单次（raw）只计第一轮答案；harness 计全部轮次。',
              f"- 时间口径不同：27B 的 {LOCAL_GPU_S:.3f} GPU·s/间取自修订文字（59 分钟，可能含模型加载；作业 7425973 的 "
              'sacct 与 receipt 不在本地）；O2 的分摊不含模型加载与进程开销（另列）。',
              '- bootstrap 区间端点带蒙特卡洛误差（换随机数生成器独立复算，端点移动约一个房间 1/n 的量级）；端点离 0 '
              '不到约 1/n 的 Δ（如下界 +0.000、+0.001）不应读作显著。',
              '- 前沿图：frontier.png（横轴 = 送往大模型的房间占比，纵轴 = 比率，每个指标一条线）。点是离散策略；虚线只为'
              '视觉引导，线段之间不是可实现的策略。'] + (
              [f"- 缺失输入（各项写明如何处理）：{'；'.join(result['missing'])}。"] if result['missing'] else []) + [
              '', '完整数字见 results.json、levels.csv、contrasts.csv。']
    return '\n'.join(lines) + '\n'


def plot(path, result):
    """Quality vs the share of rooms sent to the big model; one panel per (scope, big model), one line per metric."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    series = (('checker_rate', '检查器', '#2a78d6'), ('strict_rate', '严格准入', '#eb6834'),
              ('noreq_rate', 'required={}', '#1baf7a'))
    panels = result['panels']
    plt.rcParams.update({'font.size': 12, 'font.sans-serif': ['PingFang SC', 'Heiti SC', 'Arial Unicode MS',
                                                              'Noto Sans CJK SC', 'DejaVu Sans']})
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8), squeeze=False, sharey=True)
    for ax, p in zip(axes[0], panels):
        lv = p['levels']['all']
        order = sorted(POLICIES, key=lambda q: (lv[q]['escalated_share'], POLICIES.index(q)))
        xs = [lv[q]['escalated_share'] for q in order]
        for key, label, color in series:
            ys = [lv[q][key] for q in order]
            ax.plot(xs, ys, ':o', color=color, lw=1.2, ms=7, label=label)   # dotted: guide only, not a policy
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=10,
                        color='#52514e')
        names = collections.defaultdict(list)       # policies at the same share share one label
        for q, x in zip(order, xs):
            names[x].append(q)
        for k, (x, qs) in enumerate(names.items()):     # alternate heights so neighbouring labels never collide
            ax.annotate('\n= '.join(qs), (x, 0.985 - .06 * (k % 2)), xycoords=('data', 'axes fraction'), ha='center', va='top',
                        fontsize=9, color='#52514e')
            ax.axvline(x, color='#c3c2b7', lw=0.8, zorder=0)
        ax.set_title(f"{p['panel']}（{p['rooms']} 间）\n{p['big']}", fontsize=11)
        ax.set_xlabel('送往大模型的房间占比')
        ax.set_xlim(-0.08, 1.25)
        ax.set_xticks([0, .2, .4, .6, .8, 1])
        ax.grid(alpha=.25)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    low = min(lv[k] for p in panels for lv in p['levels']['all'].values() for k in RATE_KEYS)
    axes[0][0].set_ylim(low - .03, 1.12)
    axes[0][0].set_ylabel('比率（全部请求为分母）')
    axes[0][0].legend(loc='lower right', frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write(out, result, results=None):
    out = Path(out)
    if out.exists() or (results and Path(results).exists()):
        raise FileExistsError(f'{out} or {results} exists (write-once)')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.frontier-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'results.json', result)
        _csv(dest, result)
        (dest / 'report.md').write_text(report(result))
        plot(dest / 'frontier.png', result)
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    if results:
        shutil.copytree(out, results)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--usage-a', type=Path)
    parser.add_argument('--uids-c', type=Path)
    parser.add_argument('--big-c-dir', type=Path)
    parser.add_argument('--usage-c', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2000)
    args = parser.parse_args(argv)
    if bool(args.uids_c) != bool(args.big_c_dir) or (args.usage_c and not args.uids_c):
        parser.error('scope C needs both --uids-c and --big-c-dir (--usage-c only with them)')
    return args


def main(argv=None):
    args = parse_args(argv)
    result = build(args)
    write(args.out, result, args.results)
    print(json.dumps({'status': result['status'], 'panels': [p['panel'] for p in result['panels']],
                      'missing': result['missing']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
