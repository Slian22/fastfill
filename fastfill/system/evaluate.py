"""Amendment 022 §2: the delivered system (prior sizes -> placer -> S1 -> escalation), evaluated stage by stage.

Spec: docs/research/20261013-unknown-size-comparison/amendment-022-delivered-system-20261014.md (all rules there).
python -m fastfill.system.evaluate STEP --run-root RUN --mode interim|final [--scope cohort|test] ...
(sbatch: isambard_022.sbatch; chain: submit_022.sh)

Placer (placer()): interim = R (rpo-dpo1); final = the amendment-021 §4 delivered model (a021/final-selection.json).
Its test rows are reused, never re-inferred: R -> infer-placement-005 (HF exact, the run of record), geometry-v5 and
S1 RUN/s1-012-test/placement-rpo-dpo1; a winner W -> its amendment-021 test inference, a021/test-geometry-W and S1
RUN/s1-022/placement-W. bound(): the S1 rows (test_apply_012._system: S1_COMPLETE on this data at max_trials 2000 /
time_limit 60) repaired those geometry rows (MANIFEST sha256), which bind that inference (run_sha256).

s1        final mode, W != R only: output_repair.run on W's test geometry -> RUN/s1-022/placement-W.
gate      cohort, read-only preview of the gate below (login node, submit_022.sh); exit 1 when it refuses.
system    --scope cohort|test [--env-file F] [--workers N]: fastfill.system.pipeline.run on the scope's eval-data joint
          requests, the pinned train prior (main-data-v1/size-priors.json, sha256 from the eval-data MANIFEST) and
          placer = replay(): the delivered placer's verified test inference (e._inference, a021_select.bind, the run
          the S1 rows descend from); each P request the pipeline builds must equal the eval-data request that inference
          answered. O2 = MODELS/Qwen3.8-27B. cohort: escalate l+o2; the gate (gate.json) first refuses, before any call,
          when more than 600 cohort rooms fail after S1 (incomplete or frozen checker: the rooms the system escalates),
          when the API calls left for amendment 022 (1,200 minus every earlier 022 attempt's api/calls.jsonl) are 0, when
          the LLM-agent calls of other runs (sizers_020.calls_used) + 1,200 > 7,700, or when a counted API run is
          unfinished; the pipeline's API cap is the calls left and its room cap 600 (api_max_rooms: the pipeline stops
          before any call when more rooms need the API after its own S1). test: escalate o2 only (no API).
summarize --scope cohort|test [--workers N] [--repeats 2000] [--price-usd-per-mtok IN OUT]
          Stages raw (placer) and s1 (+S1) from the bound S1 rows (a row with a token-cap hit without EOS is
          incomplete, amendment 005, as the pipeline scores its placer stage; agent rows unchanged); system (+escalation) = the pipeline's final layout
          (rooms.jsonl), scored here by final_view (output_repair.predicted_room, geometry.geometry, downstream.accept,
          geometry.dimensions, report_v2.size_bias), which must reproduce the S1 rows' verdicts wherever the final
          layout is the S1 layout (id, size, pose, parent). Per stage: complete, checker (frozen), strict admission (downstream.accept),
          required = {} admission (endpoint_ok and attach_ok and admit_no_required_ok), OOR, size log MAE / IoU and
          footprint bias B; every request is a rate denominator, means over complete rooms. Subsets: scale_curve.subsets
          (all, object-count bins, real / synthetic, room families). Escalation (load_pipeline: candidates, api/receipts,
          o2/receipts.jsonl): share, per step rooms / passes / calls / latency, final stage, API tokens, cost (only with
          --price-usd-per-mtok, else 'not measured'), escalated vs the S1 rows' failing set. Caps fail closed: cohort
          <= 600 API rooms and <= 1,200 calls, test 0 calls; else nothing is published. Contrasts (report_v2.contrasts:
          house-cluster bootstrap, seed 42; no CI below report_v2.NO_CI components): s1 - raw, system - s1,
          system - raw; on the cohort also system - each agent row (single-shot = llm_raw, harness = llm_harness; raw
          and +S1): amendment-012 size-free RUN/s1-012-test/agent-NAME-V (J input) and amendment-016 size-fixed
          RUN/s1-016-size-fixed/agent-NAME-P-V (P input; L on its 300-room pilot). Missing agent outputs are listed,
          never substituted. The cohort system (+escalation, and contrasts with it) is labelled "system with API
          access" everywhere; in the CSVs placer stages and agent rows carry their own labels. Mesh collisions
          (amendment 019): the system is not rendered -> not measured. -> RUN/system-022/MODE/SCOPE/eval: results.json,
          levels.csv, contrasts.csv, report.md (Chinese), MANIFEST.json (write-once).
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
import sys
import tempfile

from fastfill.comparison import a021_select as A
from fastfill.comparison import downstream as dn
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import llm_agent
from fastfill.comparison import llm_open
from fastfill.comparison import output_repair as rep
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import scale_curve as sc
from fastfill.comparison import sizers_020
from fastfill.comparison import test_apply_012 as t12
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.system import pipeline as pl

SPEC = 'docs/research/20261013-unknown-size-comparison/amendment-022-delivered-system-20261014.md'
SCOPES = {'cohort': {'escalate': 'l+o2', 'api_rooms': 600, 'api_calls': 1200},
          'test': {'escalate': 'o2', 'api_rooms': 0, 'api_calls': 0}}
GLOBAL_CAP = 7700           # §2: the amendment-002/020 LLM-agent cap (6,500) raised by 1,200
STAGES = ('raw', 's1', 'system')
RATES = ('complete', 'checker', 'strict', 'noreq')
MEANS = ('oor', 'size_log_mae', 'size_iou', 'bias_fp')
FIELDS = (*RATES, 'oor', 'size_log_mae')
STEPS = ('gpt', 'o2')                                   # pipeline candidate stages that are escalation steps
TOKENS = ('input_tokens', 'output_tokens', 'reasoning_tokens')
POSE = ('id', 'size', 'pos', 'yaw', 'parent')      # compared layout fields (category: request-normalised vs GT raw)
FIXED_AGENTS = ('O2', 'O4', 'O6', 'O7')                 # amendment 016 §2 size-fixed, whole cohort
VARIANTS = ('raw', 'harness')                           # single-shot (llm_raw) and harness (llm_harness)
CONTRAST_SUBSETS = ('all', *(n for n, _, _ in sc.BINS), 'real', 'synthetic', 'family:paper_families',
                    'family:other_families')
LABEL = {'cohort': 'system with API access (gpt-6.1-sol escalation, then O2)', 'test': 'system with free O2 escalation'}
LABEL_ZH = {'cohort': '系统（含 API 访问：gpt-6.1-sol 升级，再 O2）', 'test': '系统（仅 O2 免费升级）'}
STAGE_ZH = {'raw': '原始', 's1': '+S1', 'system': '+升级'}
NOT_MEASURED = 'not measured'


# ---- paths, prior and the placer -----------------------------------------------------------------

def paths(run, mode, scope=None):
    root = Path(run) / 'system-022' / mode
    leaf = root / scope if scope else root
    return {'root': root, 'requests': leaf / 'requests.jsonl', 'gate': leaf / 'gate.json', 'system': leaf / 'system',
            'eval': leaf / 'eval', 'package': root / 'package'}


def prior_table(run):
    """{'path', 'sha256'} of the train prior P's requests were built from (eval-data MANIFEST input_consumed_sha256)."""
    run = Path(run)
    manifest = h.strict_json((run / 'eval-data-v1' / 'MANIFEST.json').read_text())
    return {'path': run / 'main-data-v1' / 'size-priors.json',
            'sha256': h._expected_hash(manifest['input_consumed_sha256']['training']['size-priors.json'])}


def placer(run, mode):
    """The delivered placer and its reusable test rows (nothing reruns on test)."""
    run = Path(run)
    if mode == 'interim':
        cand, engine, spec, final = A.R, 'hf', A.fixed_specs(run)[A.R], None
    elif mode == 'final':
        value = A.load_final(run)              # refuses before amendment 021 §4 has selected
        cand, engine, spec, final = value['delivered'], value['engine'], value['model'], h.sha256(
            run / A.A21 / 'final-selection.json')
    else:
        raise ValueError('mode must be interim or final')
    if cand == A.R:   # 021 §3: R's HF test rows are kept as they are (the run of record)
        rows_engine, inference = 'hf', run / 'infer-placement-005'
        geometry, s1 = run / 'geometry-v5', run / 's1-012-test' / 'placement-rpo-dpo1'
    else:
        rows_engine, inference = engine, A.run_dir(run, 'test', cand, engine, A.specs_of(run))
        geometry, s1 = run / A.A21 / f'test-geometry-{cand}', run / 's1-022' / f'placement-{cand}'
    return {'mode': mode, 'candidate': cand, 'label': A.LABELS.get(cand, cand), 'spec': spec, 'engine': engine,
            'rows_engine': rows_engine, 'inference': inference, 'geometry': geometry, 's1': s1,
            'final_selection_sha256': final}


def bound(p, data):
    """(S1 rows, S1 info, the placer inference run_sha256) of the placer's test rows; refuses a broken chain."""
    data_sha = h.sha256(Path(data) / 'MANIFEST.json')
    summary = h.strict_json((p['geometry'] / 'summary.json').read_text())
    if summary.get('status') != 'GEOMETRY_COMPLETE' or summary.get('data_manifest_sha256') != data_sha:
        raise ValueError(f"{p['geometry']}: not complete geometry rows of this data")
    if p['candidate'] == A.R:
        run_sha = summary['inference_run_sha256']['placement']
    else:
        run_sha = summary['inference']['run_sha256']
        if summary.get('candidate') != p['candidate'] or summary.get('final_selection_sha256') != p['final_selection_sha256']:
            raise ValueError(f"{p['geometry']}: not the delivered placer's amendment-021 test rows")
    rows, info = t12._system(p['s1'], 'placement', 'placement-geometry.jsonl', data_sha)
    if info['inputs']['geometry_manifest_sha256'] != h.sha256(p['geometry'] / 'MANIFEST.json'):
        raise ValueError(f"{p['s1']}: S1 of other geometry rows than {p['geometry']}")
    return rows, info, run_sha


def replay_placer(records, requests, identity):
    """pipeline.Placer answering each P request with the recorded test answer of that exact request (no GPU)."""
    by = h._by_uid(records)

    def generate(batch):
        out = {}
        for r in batch:
            uid = r['uid']
            if uid not in by or r['messages'] != requests[uid]['messages']:
                raise ValueError(f'replay: {uid}: the pipeline built another P request than the recorded one')
            x = by[uid]
            out[uid] = {'raw': x.get('raw'), 'error': x.get('error'),
                        'cap_hit': bool(x.get('reached_token_cap') and not x.get('stopped_on_eos'))}
        return out
    return pl.Placer(identity, generate)


def receipt_policy(receipt):
    """The exact policy an inference receipt loaded: top level in HF (infer_shard) receipts, under 'model' in the
    amendment-021 vLLM greedy receipts (a021_select.greedy: sample_vllm.model_identity)."""
    return receipt.get('policy') or (receipt.get('model') or {}).get('policy') or {}


def replay(p, data, run_sha):
    """replay_placer over the delivered placer's verified test inference; run_sha = what the S1 rows descend from."""
    data = Path(data)
    manifest = e._verified(data)
    records, receipt = e._inference(p['inference'], data, manifest['expected_uids'])
    A.bind(receipt, p['spec'], p['rows_engine'])
    if receipt['run_sha256'] != run_sha:
        raise ValueError('the placer test rows were not generated by this inference')
    requests = h._by_uid(h._rows(data, 'placement/requests.jsonl', manifest['files']['placement/requests.jsonl']))
    policy = receipt_policy(receipt)
    identity = {'engine': f"replay:{p['rows_engine']}", 'model': policy.get('training_base'),
                'adapter': policy.get('adapter'), 'adapter_sha256': policy.get('adapter_sha256'),
                'candidate': p['candidate'], 'serving_engine': p['engine'], 'run_sha256': receipt['run_sha256'],
                'inference': str(Path(p['inference']).resolve()),
                'manifest_sha256': h.sha256(Path(p['inference']) / 'MANIFEST.json'), 'policy': policy}
    return replay_placer(records, requests, identity)


# ---- the pipeline's published run ----------------------------------------------------------------

def load_pipeline(out):
    """(identity, rows) of a finished pipeline.run dir: result.json, rooms.jsonl (final layout, stage, candidates),
    api/receipts/*.json (llm_agent receipts) and o2/receipts.jsonl (llm_open bodies) for per-step calls, latency and
    tokens. A row: uid, layout, stage (placer | gpt | o2 | fallback), escalation = one entry per escalation candidate."""
    out = Path(out)
    result = h.strict_json((out / 'result.json').read_text())
    if result.get('status') != 'SYSTEM_COMPLETE':
        raise ValueError(f'{out}: not a complete system run (status {result.get("status")})')
    api = result['escalation'].get('api') or {}
    identity = {'placer': result['placer'], 'escalate': result['escalate'], 'api_cap': result['api_cap'],
                'api_calls': api.get('calls_ledger', 0), 'api_stopped': api.get('stopped'),
                'prior': result['prior'], 'result_sha256': h.sha256(out / 'result.json'),
                'rooms_sha256': h.sha256(out / 'rooms.jsonl')}
    bodies = {'gpt': {}, 'o2': {}}
    for path in sorted((out / 'api' / 'receipts').glob('*.json')):
        body = h.strict_json(path.read_text())
        bodies['gpt'][body['uid']] = body
    if (out / 'o2' / 'receipts.jsonl').exists():
        bodies['o2'] = h._by_uid([h.strict_json(x) for x in (out / 'o2' / 'receipts.jsonl').read_text().splitlines()])
    rows = []
    for line in (out / 'rooms.jsonl').read_text().splitlines():
        x = h.strict_json(line)
        steps = []
        for c in x['candidates']:
            if c['stage'] not in STEPS:
                continue
            answers = (bodies[c['stage']].get(x['uid']) or {}).get('answers') or []
            steps.append({'step': c['stage'], 'calls': sum(a.get('tries', 1) for a in answers),
                          'latency_s': sum(a['latency_seconds'] for a in answers) if answers else None,
                          'usage': {k: sum((a.get('usage') or {}).get(k, 0) for a in answers) for k in TOKENS},
                          'checker': bool(c['checker_pass'])})
        rows.append({'uid': x['uid'], 'layout': x['layout'], 'stage': x['stage'], 'escalation': steps})
    return identity, rows


# ---- gate (cohort, before any API call) ---------------------------------------------------------

def capped(row):
    """Amendment 005: an answer that hit the token cap without EOS is incomplete (rows carrying the flags, as
    report_v2.cap_rule and the pipeline's placer stage)."""
    return bool(row.get('reached_token_cap')) and not row.get('stopped_on_eos', True)


def failing(s1_rows, uids):
    """The rooms the system escalates: incomplete (cap rule included) or failing the frozen checker after S1."""
    keep = set(uids)
    return [x['uid'] for x in s1_rows if x['uid'] in keep
            and not (x['complete'] and not capped(x) and x['s1']['geometry_success_s1'])]


def spent(run):
    """API calls of every earlier amendment-022 attempt (each pipeline run keeps its own ledger)."""
    return sum(len(llm_agent._jsonl(p)) for p in sorted((Path(run) / 'system-022').glob('*/*/system*/api/calls.jsonl')))


def gate(run, s1_rows, uids):
    cap = SCOPES['cohort']
    would, done = failing(s1_rows, uids), spent(run)
    used = sizers_020.calls_used(run)
    unfinished = sorted(n for n in used if not (Path(run) / n / 'MANIFEST.json').exists())
    left = cap['api_calls'] - done
    reasons = [r for r, bad in (
        (f'{len(would)} rooms would escalate > {cap["api_rooms"]}', len(would) > cap['api_rooms']),
        (f'amendment-022 API calls already spent {done} of {cap["api_calls"]}', left <= 0),
        (f'LLM-agent calls used {sum(used.values())} + {cap["api_calls"]} > {GLOBAL_CAP}',
         sum(used.values()) + cap['api_calls'] > GLOBAL_CAP),
        (f'unfinished API runs {unfinished}', bool(unfinished))) if bad]
    return {'protocol': 'amendment-022-gate', 'rooms': len(uids), 'would_escalate': len(would),
            'would_escalate_uids_sha256': llm_agent._digest(json.dumps(would)), 'cap_rooms': cap['api_rooms'],
            'cap_calls': cap['api_calls'], 'spent_022': done, 'api_cap': max(0, left), 'global_cap': GLOBAL_CAP,
            'calls_used': used, 'ok': not reasons, 'reasons': reasons}


# ---- per-room views ------------------------------------------------------------------------------

def _base(room):
    return {k: room[k] for k in ('uid', 'source', 'group')}


def stage_views(row, room, cap_rule=False):
    """{raw, s1}: the views of one output_repair row (raw beside S1; S1 never resizes); cap_rule (the placer's rows,
    as the pipeline scores its placer stage): a capped row is incomplete."""
    if (row['uid'], row['source'], row['group']) != (room['uid'], room['source'], room['group']):
        raise ValueError('S1 row / room identity mismatch')
    done, s = bool(row['complete']) and not (cap_rule and capped(row)), row['s1']
    size = ({'size_log_mae': row['size_log_mae'], 'size_iou': row['size_iou'],
             'bias_fp': r2.size_bias(room['objects'], row['raw_layout'])[0]} if done else dict.fromkeys(MEANS[1:]))
    return {stage: {**_base(room), **size, 'complete': done, 'checker': done and bool(s[f'geometry_success_{stage}']),
                    'strict': done and bool(s[f'strict_ok_{stage}']),
                    'noreq': done and bool(s[f'endpoint_ok_{stage}'] and s[f'attach_ok_{stage}']
                                           and s[f'admit_no_required_ok_{stage}']),
                    'oor': (row['oor'] if stage == 'raw' else s['geometry_s1']['oor']) if done else None}
            for stage in ('raw', 's1')}


def final_view(task):
    """(room, P request, final layout) -> view; also counts objects whose size differs from the request's."""
    room, request, layout = task
    if layout is None:
        return {**_base(room), 'complete': False, **dict.fromkeys(RATES[1:], False), **dict.fromkeys(MEANS),
                'size_changed': 0}
    if [o['id'] for o in layout] != [o['id'] for o in room['objects']]:
        raise ValueError(f"{room['uid']}: the system layout is not the requested objects in request order")
    pred = rep.predicted_room(room, request['messages'], layout)
    geo, v = g.geometry(pred), dn.accept(pred, layout)
    iou, mae = g.dimensions(room['objects'], layout)
    sizes = {o['id']: o['size'] for o in h.strict_json(request['messages'][1]['content'])['objects']}
    return {**_base(room), 'complete': True, 'checker': geo['geometry_success'], 'strict': v['accepted'],
            'noreq': v['endpoint_ok'] and v['attach_ok'] and v['admit_no_required_ok'], 'oor': geo['oor'],
            'size_log_mae': mae, 'size_iou': iou, 'bias_fp': r2.size_bias(room['objects'], layout)[0],
            'size_changed': sum(any(abs(a - b) > llm_agent.SIZE_TOL for a, b in zip(o['size'], sizes[o['id']]))
                                for o in layout)}


def level(views):
    n, done = len(views), [v for v in views if v['complete']]

    def mean(key):
        values = [v[key] for v in done if v[key] is not None]
        return statistics.fmean(values) if values else None
    return {'requests': n, 'complete_rooms': len(done),
            **{k + '_rate': sum(bool(v[k]) for v in views) / n if n else None for k in RATES}, **{k: mean(k) for k in MEANS}}


def contrast(task):
    first, second, repeats = task
    k = len({v['group'] for v in first})
    if not first:
        return {'components': 0, 'fields': {}}
    fields = r2.contrasts(first, second, FIELDS, repeats, r2.SEED)
    if k < r2.NO_CI:
        fields = {f: {**x, 'ci95': None, 'p': None} for f, x in fields.items()}
    return {'components': k, 'fields': fields}


def _quantiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {'n': len(values), 'mean': statistics.fmean(values), 'p50': g._quantile(values, .5),
            'p95': g._quantile(values, .95), 'max': values[-1]}


def escalation(rows, fail, identity, scope, prices=None):
    """Escalation share, steps, final stages, API calls / tokens / latency / cost; caps fail closed."""
    n, escalated = len(rows), {r['uid'] for r in rows if r['escalation']}
    steps = [s for r in rows for s in r['escalation']]
    of = lambda step: [s for s in steps if s['step'] == step]
    tokens = {k: sum(s['usage'][k] for s in of('gpt')) for k in TOKENS}
    api_rooms = sum(any(s['step'] == 'gpt' and s['calls'] for s in r['escalation']) for r in rows)
    calls = max(int(identity['api_calls']), sum(s['calls'] for s in of('gpt')))
    cap = SCOPES[scope]
    if api_rooms > cap['api_rooms'] or calls > cap['api_calls'] or (identity['api_cap'] or 0) > cap['api_calls']:
        raise ValueError(f'API cap exceeded on {scope}: {api_rooms} rooms / {calls} calls (run cap '
                         f"{identity['api_cap']}) > {cap['api_rooms']} / {cap['api_calls']}; not published (fail closed)")
    cost = (None if prices is None else
            (tokens['input_tokens'] * prices[0] + tokens['output_tokens'] * prices[1]) / 1e6)
    return {'requests': n, 'escalated': len(escalated), 'escalated_share': len(escalated) / n if n else None,
            'steps': {s: {'rooms': len(of(s)), 'passed': sum(x['checker'] for x in of(s)),
                          'calls': sum(x['calls'] for x in of(s)),
                          'latency_s': _quantiles([x['latency_s'] for x in of(s)])} for s in STEPS},
            'final_stage': dict(collections.Counter(r['stage'] for r in rows)),
            'returned_nothing': sum(r['layout'] is None for r in rows),
            'api': {'rooms': api_rooms, 'calls': calls, 'ledger_calls': identity['api_calls'],
                    'run_cap': identity['api_cap'], 'stopped': identity.get('api_stopped'), 'tokens': tokens,
                    'cap_rooms': cap['api_rooms'], 'cap_calls': cap['api_calls'],
                    'cost_usd': NOT_MEASURED if cost is None else cost,
                    'price_usd_per_mtok': NOT_MEASURED if prices is None else {'input': prices[0], 'output': prices[1]}},
            'vs_s1_rows': {'s1_failing': len(fail), 'escalated_not_failing': len(escalated - set(fail)),
                           'failing_not_escalated': len(set(fail) - escalated)}}


# ---- evaluation core -----------------------------------------------------------------------------

def evaluate(rooms, requests, s1_rows, system, scope, agents=None, repeats=2000, workers=1, prices=None):
    """rooms / requests: {uid: ...} of the scope; s1_rows: the placer's bound S1 rows (scope uids); system:
    (identity, rows) from load_pipeline; agents: {label: {'setting', 'variant', 'rows'}} (S1 rows, cohort only)."""
    identity, rows = system
    uids = [x['uid'] for x in s1_rows]
    if sorted(uids) != sorted(rooms) or sorted(r['uid'] for r in rows) != sorted(uids):
        raise ValueError('the S1 rows and the system rows must cover exactly the scope requests')
    if identity['escalate'] != SCOPES[scope]['escalate']:
        raise ValueError(f"system escalation {identity['escalate']!r} is not the {scope} rule")
    by = {r['uid']: r for r in rows}
    views = {'raw': [], 's1': []}
    for x in s1_rows:
        for stage, v in stage_views(x, rooms[x['uid']], cap_rule=True).items():
            views[stage].append(v)
    views['system'] = r2._map(final_view, [(rooms[u], requests[u], by[u]['layout']) for u in uids], workers)
    agree = 0
    pose = lambda layout: [{k: o.get(k) for k in POSE} for o in layout or []]
    for x, s1, final in zip(s1_rows, views['s1'], views['system']):
        if s1['complete'] and pose(by[x['uid']]['layout']) == pose(x['s1_layout']):
            if any(s1[k] != final[k] for k in ('checker', 'strict', 'noreq')):
                raise ValueError(f"{x['uid']}: the final layout is the S1 layout but scores differently")
            agree += 1
    subsets = {k: {x['uid'] for x in v} for k, v in sc.subsets(list(rooms.values())).items()}
    systems = {'system': {stage: views[stage] for stage in STAGES}}
    for label, a in (agents or {}).items():
        systems[label] = {'raw': [], 's1': []}
        for x in a['rows']:
            for stage, v in stage_views(x, rooms[x['uid']]).items():
                systems[label][stage].append(v)
    levels = {name: {label: {stage: level([v for v in vs if v['uid'] in keep]) for stage, vs in stages.items()}
                     for label, stages in systems.items()} for name, keep in subsets.items()}
    tasks, meta = [], []
    for name, keep in subsets.items():
        pick = lambda vs, only=None: [v for v in vs if v['uid'] in keep and (only is None or v['uid'] in only)]
        for first, second in (('s1', 'raw'), ('system', 's1'), ('system', 'raw')):
            tasks.append((pick(views[first]), pick(views[second]), repeats))
            meta.append({'subset': name, 'comparison': f'{first} - {second}'})
        if name not in CONTRAST_SUBSETS:
            continue
        for label, a in (agents or {}).items():
            only = {x['uid'] for x in a['rows']}
            for stage in ('raw', 's1'):
                tasks.append((pick(views['system'], only), pick(systems[label][stage], only), repeats))
                meta.append({'subset': name, 'comparison': f'system - {label} [{stage}]', 'setting': a['setting'],
                             'variant': a['variant']})
    contrasts = [{**m, 'rooms': len(t[0]), **c} for m, t, c in zip(meta, tasks, r2._map(contrast, tasks, workers))]
    return {'status': 'SYSTEM_022_EVAL_COMPLETE', 'protocol': 'amendment-022 §2', 'scope': scope,
            'label': LABEL[scope], 'label_zh': LABEL_ZH[scope], 'requests': len(uids),
            'escalation': escalation(rows, failing(s1_rows, uids), identity, scope, prices),
            'size_changed_objects': sum(v['size_changed'] for v in views['system']), 'final_equals_s1_rows': agree,
            'levels': levels, 'contrasts': contrasts,
            'agents': {label: {'setting': a['setting'], 'variant': a['variant'], 'rooms': len(a['rows']),
                               'path': a.get('path')} for label, a in (agents or {}).items()},
            'mesh_collisions': NOT_MEASURED + ' (amendment 019 applies to rendered rooms; the system is not rendered)',
            'metric_contract': {
                'rate_denominator': 'every request of the scope (incomplete = failure)',
                'checker': 'geometry.geometry geometry_success (frozen checker)', 'strict': 'downstream.accept accepted',
                'noreq': 'endpoint_ok and attach_ok and admit_no_required_ok (required = {})',
                'means': 'over the complete rooms of the subset', 'bias_fp': 'report_v2.size_bias footprint',
                'raw_s1': 'the bound output_repair rows of the placer (amendment-012 settings); a row whose answer hit '
                          'the token cap without EOS is incomplete (amendment 005, as the pipeline scores its placer '
                          'stage); agent rows as their own protocols scored them (amendment 012 §4 / 016 §2: no cap rule)',
                'system': 'the pipeline final layout, scored by final_view',
                'contrasts': 'report_v2.contrasts, first minus second, house-cluster bootstrap, seed 42, p unadjusted',
                'fairness': 'the escalated system has API / extra-model access; never compared as "the model"'}}


# ---- loading -------------------------------------------------------------------------------------

def agent_dirs(run):
    """{label: (S1 dir, requests arm, setting, variant)} of every agent row set the cohort report pairs."""
    run, out = Path(run), {}
    for v in VARIANTS:
        for name in t12.AGENTS:
            out[f'{name}-{v}'] = (run / 's1-012-test' / f'agent-{name}-{v}', 'joint', 'size_free', v)
        for name in FIXED_AGENTS:
            out[f'{name}-P-{v}'] = (run / 's1-016-size-fixed' / f'agent-{name}-P-{v}', 'placement', 'size_fixed', v)
        out[f'L-P-pilot-{v}'] = (run / 's1-016-size-fixed' / f'agent-L-P-pilot-{v}', 'placement', 'size_fixed', v)
    return out


def load_agents(run, uids, data_sha):
    agents, missing, keep = {}, [], set(uids)
    for label, (path, arm, setting, v) in agent_dirs(run).items():
        if not (path / 'MANIFEST.json').exists():
            missing.append(label)
            continue
        rows, _ = t12._system(path, arm, f'llm_{v}-geometry.jsonl', data_sha)
        if not {x['uid'] for x in rows} <= keep:
            raise ValueError(f'{label}: rows outside the cohort')
        agents[label] = {'setting': setting, 'variant': v, 'rows': rows, 'path': str(path)}
    return agents, missing


def scope_uids(run, data, scope):
    manifest = e._verified(Path(data))
    if scope == 'test':
        return manifest['expected_uids']
    cohort = h.strict_json((Path(run) / 'llm-cohort-v1.json').read_text())
    uids = cohort['uids']
    if (cohort.get('data_manifest_sha256') != h.sha256(Path(data) / 'MANIFEST.json')
            or llm_agent._digest(json.dumps(uids)) != cohort.get('uids_sha256')
            or not set(uids) <= set(manifest['expected_uids'])):
        raise ValueError('cohort does not bind this test data or its uid hash')
    return uids


def _scope(run, data, scope):
    manifest = e._verified(Path(data))
    uids = scope_uids(run, data, scope)
    keep = set(uids)
    rows = lambda name: {u: x for u, x in h._by_uid(h._rows(Path(data), name, manifest['files'][name])).items()
                         if u in keep}
    return uids, rows('rooms.jsonl'), rows('placement/requests.jsonl'), rows('joint/requests.jsonl')


# ---- steps ---------------------------------------------------------------------------------------

def step_s1(run, mode, workers=1):
    p = placer(run, mode)
    if p['candidate'] == A.R:
        return {'status': 'S1_REUSED', 'path': str(p['s1'])}
    return rep.run(Path(run) / 'eval-data-v1', p['geometry'] / 'placement-geometry.jsonl', 'placement', p['s1'],
                   workers, t12.MAX_TRIALS, t12.TIME_LIMIT)


def step_system(run, mode, scope, env_file=None, workers=1):
    run, data = Path(run), Path(run) / 'eval-data-v1'
    out, p = paths(run, mode, scope), placer(run, mode)
    if out['system'].exists():
        raise FileExistsError(f"{out['system']} exists (a pipeline run never resumes; move it aside to retry, its "
                              'API ledger still counts)')
    uids, _, _, joint = _scope(run, data, scope)
    s1_rows, _, run_sha = bound(p, data)
    api_cap = 0
    if scope == 'cohort':
        verdict = gate(run, s1_rows, uids)
        out['gate'].parent.mkdir(parents=True, exist_ok=True)
        h.write_json(out['gate'], verdict)
        if not verdict['ok']:
            raise SystemExit('GATE REFUSED: ' + '; '.join(verdict['reasons']))
        api_cap = verdict['api_cap']
    text = ''.join(json.dumps(joint[u], allow_nan=False) + '\n' for u in uids)
    if out['requests'].exists() and out['requests'].read_text() != text:
        raise ValueError(f"{out['requests']} exists with other requests")
    out['requests'].parent.mkdir(parents=True, exist_ok=True)
    out['requests'].write_text(text)
    prior = prior_table(run)
    return pl.run(pl.load_requests(out['requests']), pl.load_prior(prior['path'], prior['sha256']),
                  replay(p, data, run_sha), out['system'], escalate=SCOPES[scope]['escalate'], api_cap=api_cap,
                  api_max_rooms=SCOPES[scope]['api_rooms'], env_file=env_file,
                  o2_model=A.models_root(run) / llm_open.SPECS['O2']['model'], workers=workers)


def step_summarize(run, mode, scope, workers=1, repeats=2000, prices=None):
    run, data = Path(run), Path(run) / 'eval-data-v1'
    out, p = paths(run, mode, scope), placer(run, mode)
    if out['eval'].exists():
        raise FileExistsError(out['eval'])
    uids, rooms, requests, _ = _scope(run, data, scope)
    s1_rows, s1_info, run_sha = bound(p, data)
    s1_rows = [x for x in s1_rows if x['uid'] in rooms]
    identity, rows = load_pipeline(out['system'])
    if identity['placer'].get('run_sha256') != run_sha:
        raise ValueError('the system placer stage did not replay the bound placer test rows')
    if identity['prior']['sha256'] != prior_table(run)['sha256']:
        raise ValueError('the system used another prior table than the one P was evaluated with')
    data_sha = h.sha256(data / 'MANIFEST.json')
    agents, missing = load_agents(run, uids, data_sha) if scope == 'cohort' else ({}, [])
    result = evaluate(rooms, requests, s1_rows, (identity, rows), scope, agents, repeats, workers, prices)
    result.update({
        'mode': mode, 'interim': mode == 'interim', 'agents_missing': missing,
        'placer': {k: (str(v) if isinstance(v, Path) else v) for k, v in p.items()},
        'inputs': {'data_MANIFEST': data_sha, 's1_MANIFEST': h.sha256(p['s1'] / 'MANIFEST.json'),
                   's1_params': s1_info['params'], 'geometry_MANIFEST': h.sha256(p['geometry'] / 'MANIFEST.json'),
                   'placer_run_sha256': run_sha, 'system': identity,
                   'gate': h.strict_json(out['gate'].read_text()) if out['gate'].exists() else None,
                   'requests_sha256': h.sha256(out['requests'])},
        'settings': {'repeats': repeats, 'seed': r2.SEED, 'max_trials': t12.MAX_TRIALS, 'time_limit_s': t12.TIME_LIMIT},
        'spec': SPEC, 'code_sha256': {Path(f).name: h.sha256(f) for f in (__file__, pl.__file__, rep.__file__,
                                                                        dn.__file__, g.__file__, r2.__file__)}})
    out['eval'].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.eval-', dir=out['eval'].parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'results.json', result)
        _csv(dest, result)
        (dest / 'report.md').write_text(report(result))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out['eval'])
    return result


# ---- output --------------------------------------------------------------------------------------

def _f(value, signed=False):
    if value is None or isinstance(value, str):
        return '—' if value is None else value
    return f'{value:+.3f}' if signed else f'{value:.3f}'


def _ci(x):
    if not x:
        return '—'
    ci = x.get('ci95')
    return _f(x['estimate'], True) + (f' [{_f(ci[0], True)}, {_f(ci[1], True)}]' if ci else '')


def _role(result, label, stage):
    """CSV label: the fairness label on the escalated system only; placer stages and agent rows say what they are."""
    if label == 'system':
        return result['label'] if stage == 'system' else 'placer, no escalation'
    a = result['agents'][label]
    return f"{'single-shot' if a['variant'] == 'raw' else 'harness'} agent ({a['setting']})"


def _csv(dest, result):
    with (dest / 'levels.csv').open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['label', 'scope', 'subset', 'system', 'stage', 'requests', 'complete_rooms',
                    *(k + '_rate' for k in RATES), *MEANS])
        for name, systems in result['levels'].items():
            for label, stages in systems.items():
                for stage, x in stages.items():
                    w.writerow([_role(result, label, stage), result['scope'], name, label, stage, x['requests'],
                                x['complete_rooms'],
                                *(x[k + '_rate'] for k in RATES), *(x[k] for k in MEANS)])
    with (dest / 'contrasts.csv').open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['label', 'scope', 'subset', 'comparison', 'rooms', 'components', 'field', 'estimate', 'ci_low',
                    'ci_high', 'p'])
        for c in result['contrasts']:
            for field, x in c['fields'].items():
                w.writerow([_role(result, 'system', c['comparison'].split(' - ')[0]), result['scope'], c['subset'],
                            c['comparison'], c['rooms'], c['components'], field, x['estimate'],
                            *(x['ci95'] or [None, None]), x['p']])


def report(result):
    scope, esc = result['scope'], result['escalation']
    api, name, all_ = esc['api'], result['label_zh'], result['levels']['all']
    cell = lambda x: f"{_f(x['checker_rate'])} / {_f(x['strict_rate'])} / {_f(x['noreq_rate'])}" if x['requests'] else '—'
    lines = [f'# 修订 022 §2：交付系统逐阶段评估（{"代理队列" if scope == "cohort" else "全部测试房间"}，{result["requests"]} 间）', '',
             f'**{name}**。升级阶段使用 API 或额外的开源大模型，因此这是“含升级的系统”，从不作为“模型”与单次代理比较。'
             + ('（临时版 interim：放置器 R = rpo-dpo1，不改变任何决定。）' if result.get('interim') else ''), '',
             f"放置器：{result.get('placer', {}).get('label', '—')}（测试行复用，未重新推理）。阶段：原始 → +S1（修订 012 的确定性"
             '修复）→ +升级（只对 S1 后未完成或检查器失败的房间）。', '',
             '比率的分母为该范围的全部请求（未完成计失败）；OOR、尺寸 log MAE、足迹偏差 B 为完整房间的均值。'
             '严格准入 = downstream.accept；required={} 准入 = 端点、贴附与 admit_no_required 均通过（修订 017 的两个变体）。', '',
             '## 1. 各阶段贡献（全部房间）', '',
             f'“system” 指{name}。代理行（若有）是修订 012（自选尺寸，J 输入）与修订 016（固定尺寸，P 输入；L 只在 300 间'
             '试点）的已有输出：单次（-raw）与 harness（-harness），各含原始与 +S1。', '',
             '| 系统 | 阶段 | 请求 | 完整率 | 检查器成功 | 严格准入 | required={} 准入 | OOR | 尺寸 log MAE | 足迹偏差 B |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, stages in all_.items():
        for stage, x in stages.items():
            lines.append(f"| {label} | {STAGE_ZH[stage]} | {x['requests']} | "
                         + ' | '.join(_f(x[k + '_rate']) for k in RATES) + ' | '
                         + ' | '.join(_f(x[k]) for k in ('oor', 'size_log_mae', 'bias_fp')) + ' |')
    steps = [f"- {s}：{x['rooms']} 间，之后通过检查器 {x['passed']}，调用 {x['calls']}，每间延迟（秒）均值 / p50 / p95："
             + ('—' if not x['latency_s'] else ' / '.join(_f(x['latency_s'][k]) for k in ('mean', 'p50', 'p95')))
             for s, x in esc['steps'].items()]
    cost = '未测量（未给出单价）' if isinstance(api['cost_usd'], str) else _f(api['cost_usd'])
    lines += ['', '## 2. 升级', '',
              f"- 升级房间：{esc['escalated']} / {esc['requests']}（{_f(esc['escalated_share'])}）；S1 行判定失败 "
              f"{esc['vs_s1_rows']['s1_failing']} 间，升级但 S1 行未失败 {esc['vs_s1_rows']['escalated_not_failing']} 间，"
              f"失败但未升级 {esc['vs_s1_rows']['failing_not_escalated']} 间。", *steps,
              f"- 最终输出来源：{json.dumps(esc['final_stage'], ensure_ascii=False)}；返回空输出的房间：{esc['returned_nothing']}。",
              f"- API（gpt-6.1-sol）：{api['rooms']} 间（上限 {api['cap_rooms']}），调用 {api['calls']}（本次运行上限 "
              f"{api['run_cap']}，修订上限 {api['cap_calls']}）" + (f"，因 {api['stopped']} 停止" if api['stopped'] else '')
              + f"；tokens {json.dumps(api['tokens'])}；费用估计（USD）：{cost}。",
              f"- 升级输出相对请求尺寸被改变的物体数（应为 0，B2 强制）：{result['size_changed_objects']}；最终布局等于 S1 布局"
              f"且判定与 S1 行一致的房间：{result['final_equals_s1_rows']}。",
              '- 网格级碰撞（修订 019）：未测量（系统输出未渲染）。', '',
              '## 3. 按物体数分箱、真实 / 合成来源与房间族', '',
              '每格：检查器成功 / 严格准入 / required={} 准入（全部请求为分母）。', '',
              '| 范围 | n | ' + ' | '.join(STAGE_ZH[s] for s in STAGES) + ' |', '|---|---:|' + '---:|' * len(STAGES)]
    for subset, systems in result['levels'].items():
        lines.append(f"| {subset} | {systems['system']['raw']['requests']} | "
                     + ' | '.join(cell(systems['system'][s]) for s in STAGES) + ' |')
    shown = [k for k in CONTRAST_SUBSETS if k in result['levels']]
    lines += ['', '同一格式，系统与各代理行并列（其余房间族见 levels.csv）：', '',
              '| 系统 | 阶段 | ' + ' | '.join(shown) + ' |', '|---|---|' + '---:|' * len(shown)]
    for label, stages in all_.items():
        for stage in stages:
            lines.append(f'| {label} | {STAGE_ZH[stage]} | '
                         + ' | '.join(cell(result['levels'][k][label][stage]) for k in shown) + ' |')
    bins = ('all', *(n for n, _, _ in sc.BINS))
    lines += ['', f"## 4. 配对差（房屋连通组 bootstrap 95% CI，{result.get('settings', {}).get('repeats', '—')} 次，seed {r2.SEED}；未校正）", '',
              f'前者减后者；k < {r2.NO_CI} 个连通组时不给区间。“system” 指{name}。此处只列全部房间与物体数分箱，'
              '其余范围见 contrasts.csv。', '',
              '| 范围 | 比较 | 房间 | k | Δ完整率 | Δ检查器 | Δ严格准入 | Δrequired={} | ΔOOR |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in (c for c in result['contrasts'] if c['subset'] in bins):
        lines.append(f"| {c['subset']} | {c['comparison']} | {c['rooms']} | {c['components']} | "
                     + ' | '.join(_ci(c['fields'].get(k)) for k in ('complete', 'checker', 'strict', 'noreq', 'oor')) + ' |')
    if result.get('agents_missing'):
        lines += ['', f"缺少的代理输出（不替代）：{', '.join(result['agents_missing'])}。"]
    lines += ['', '完整数字见 results.json、levels.csv、contrasts.csv。']
    return '\n'.join(lines) + '\n'


# ---- CLI -----------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('s1', 'gate', 'system', 'summarize'))
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('interim', 'final'), required=True)
    parser.add_argument('--scope', choices=tuple(SCOPES))
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2000)
    parser.add_argument('--price-usd-per-mtok', type=float, nargs=2, metavar=('IN', 'OUT'))
    args = parser.parse_args(argv)
    if args.step in ('system', 'summarize') and not args.scope:
        parser.error(f'{args.step} needs --scope')
    if args.step == 'gate' and args.scope not in (None, 'cohort'):
        parser.error('the gate applies to the cohort (API escalation) only')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.step == 's1':
        print(json.dumps({'status': step_s1(args.run_root, args.mode, args.workers)['status']}))
    elif args.step == 'gate':
        data = args.run_root / 'eval-data-v1'
        value = gate(args.run_root, bound(placer(args.run_root, args.mode), data)[0],
                     scope_uids(args.run_root, data, 'cohort'))
        print(json.dumps(value))
        return 0 if value['ok'] else 1
    elif args.step == 'system':
        value = step_system(args.run_root, args.mode, args.scope, args.env_file, args.workers)
        print(json.dumps({k: value[k] for k in ('status', 'rooms', 'stages')}))
    else:
        value = step_summarize(args.run_root, args.mode, args.scope, args.workers, args.repeats,
                               args.price_usd_per_mtok)
        print(json.dumps({'status': value['status'], 'escalated': value['escalation']['escalated']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
