"""Amendment 021 section 1.1-1.3 diagnostics: descriptive only, they gate and select nothing.

python -m fastfill.comparison.a021_diag STEP ...   (Slurm: submit_021_diag.sh, which is also the batch script)

ledger --data RUN/dpo-004/train-subset --samples RUN/dpo-004/samples [--pairs RUN/dpo-004/stage2-pairs]
       --out RUN/diag-021/ledger [--workers 72]
    Section 1.1, CPU. The amendment-013 samples (k = 0 vLLM greedy, k = 1..8 T 0.7), bound to the subset as
    onpolicy_dpo._sampling binds them, are rescored with onpolicy_dpo._sample (geometry.score, cap rule, downstream.accept
    on the raw layout): checker = complete and geometry_success, admission = complete and downstream accepted (strict),
    pass = both (onpolicy_dpo._passes). Per object-count bin (best_of_n.bin_name): greedy rates, k = 1 rates and k = 1 -
    greedy (report_v2.contrasts: house-cluster bootstrap, 2,000 draws, seed 42). Exact copy: a record whose pos and yaw
    equal an earlier record's in the same output (of which the earlier one is the immediately preceding record / of the
    same normalized category). Co-located duplicate: onpolicy_dpo.tier's rule, a main/severe/contained checker pair of one
    normalized category on one parent (exact: the pair's pos and yaw are equal). Both over complete samples, greedy and
    all k. Recombination ceiling: units = the floor-root support subtrees of the lowest-k complete sample (its `on`
    fields); in request order of their roots, each unit takes the first complete sample k (ascending, greedy first) whose
    records of that unit have no checker main pair with the units already chosen (else that lowest-k sample's); the
    combined records (verbatim, `on` included) are scored like a sample. Memo D4's per-object rule at the critique's
    subtree level; it repairs cross-unit main pairs only, so it is a lower bound on the ceiling. Also best-of-9 (any k
    passes). With --pairs the per-sample classes are compared with the 013 stage2-pairs ledger (pass / reject /
    incomplete).
pool --pool RUN/a021/pool --out RUN/diag-021/pool
    Section 1.2, CPU, minutes. Reads a021_train.pool's output (submit_021_train.sh setup; one implementation, no rerun):
    P-only rooms (origin p_only: placement.qualified, not joint.qualified; 19,135 per the lineage audit, which that job
    enforces), re-verified there by main_data._candidate on the frozen corpus (a ledger difference fails it), and P's
    target scored by the frozen checker (`pool`). Rooms and checker passes by source, bin and family. uids.json: the
    sorted P-only UIDs whose P target passes the checker.
probe --data RUN/dev-select-v1 --ref-root RUN/dev-eval-010 --out RUN/diag-021/probe [--rooms 200] [--max-minutes 24]
    Section 1.3, one GPU (<= 0.5 GPU-h). P = rpo-dpo1's exact policy (dpo_stages.exact_policy, the policy that wrote
    dev-eval-010's HF records, which give P's greedy outputs), loaded as infer._runtime loads it. Rooms by ascending
    sha256('a021-probe:' + uid); a room is used if its record is complete, valid and canonical and has a record k with an
    earlier same-category sibling on the same parent: j = the latest such sibling, pos_j != pos_k, and record k at pos_k
    collides (checker main) with j moved there. Two prefixes, teacher-forced: control = P's own output before record k;
    intervention = the same with j moved onto P's predicted pos_k. Each is cut at the last token boundary at or before
    record k (the tokenizer's offsets on the full text). Greedy continuation (record k's continuation tokens + 16) ->
    its pos; the colliding continuation = record k at the sibling's current position (control pos_j, intervention
    pos_k), the clean one = the other position (only pos differs); log-probabilities summed over the continuation
    tokens, delta = colliding - clean. Intervention - control via report_v2.contrasts. Rooms past --max-minutes
    (counted from the job's start, loading included) are not run (recorded).
Each step writes result.json, rooms.jsonl (pool: uids.json), report.md (Chinese) and MANIFEST.json, new-only.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from fastfill import scene
from fastfill import validate
from fastfill.comparison import a021_train as t
from fastfill.comparison import best_of_n as bo
from fastfill.comparison import contract
from fastfill.comparison import dev_select as ds
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import infer
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h

ARM, REFERENCE, REPEATS = 'placement', 'rpo-dpo1', 2000
FIELDS = ('complete', 'checker', 'admission', 'pass')
BINS = tuple(bo.bin_name(lo) for lo, _ in bo.BINS)
# SLACK: greedy budget = record k's continuation tokens + SLACK (cm-rounded coordinates differ by a few digits).
PROBE_SALT, PROBE_ROOMS, SLACK, CONDITIONS = 'a021-probe:', 200, 16, ('control', 'intervention')


def _code():
    return {p.name: h.sha256(p) for p in map(Path, (__file__, op.__file__, bo.__file__, r2.__file__, t.__file__,
                                                     validate.__file__, contract.__file__, scene.__file__))}


def _publish(out, result, rows, name='rooms.jsonl'):
    payload = d._jsonl(rows) if name.endswith('.jsonl') else op._json_bytes(rows)
    d._publish_dir(out, {'result.json': op._json_bytes(result), name: payload, 'report.md': REPORTS[result['step']](
        result).encode('utf-8')})


def _share(values):
    values = [v for v in values if v is not None]
    return sum(map(bool, values)) / len(values) if values else None


def _by_bin(rows, summary):
    return {'all': summary(rows), **{b: summary([r for r in rows if r['bin'] == b]) for b in BINS}}


def _rates(rows):
    return {'rooms': len(rows), **{f: _share(r[f] for r in rows) for f in FIELDS}}


# ---- 1.1 ledger ----------------------------------------------------------------------------------

def copies(placements, categories):
    """Records whose (pos, yaw) equal an earlier record's: total, of the immediately preceding one, of the same category."""
    seen, total, previous, same = {}, 0, 0, 0
    for k, p in enumerate(placements):
        key = (tuple(p['pos']), p['yaw'])
        earlier = seen.setdefault(key, [])
        if earlier:
            total += 1
            previous += earlier[-1] == k - 1
            same += any(categories[j] == categories[k] for j in earlier)
        earlier.append(k)
    return {'records': len(placements), 'exact_copies': total, 'copy_of_previous': previous, 'copy_same_category': same}


def colocated(layout):
    """onpolicy_dpo.tier's co-located duplicate pairs (unique) and how many are exact pos/yaw copies."""
    by = {o['id']: o for o in layout}
    found = validate.check({'objects': layout})['collisions']
    pairs = {tuple(sorted(p)) for kind in ('main', 'severe', 'contained') for p in found[kind]
             if scene.norm_cat(by[p[0]]['category']) == scene.norm_cat(by[p[1]]['category'])
             and by[p[0]].get('parent') == by[p[1]].get('parent')}
    exact = sum(by[a]['pos'] == by[b]['pos'] and by[a]['yaw'] == by[b]['yaw'] for a, b in pairs)
    return {'colocated_pairs': len(pairs), 'colocated_exact': exact}


def subtrees(placements):
    """Floor-root support subtrees (roots: `on` absent or not a movable id), in request order, members in request order."""
    ids = [p['id'] for p in placements]
    order, kids = {i: k for k, i in enumerate(ids)}, {i: [] for i in ids}
    roots = [p['id'] for p in placements if p.get('on') not in order]
    for p in placements:
        if p.get('on') in order:
            kids[p['on']].append(p['id'])
    units = []
    for root in roots:  # A complete sample is acyclic (contract._support), so every record has a root.
        members, todo = [], [root]
        while todo:
            members.append(todo.pop())
            todo += kids[members[-1]]
        units.append(sorted(members, key=order.get))
    return units


def _crosses(unit, chosen):
    """Whether a unit's objects form a checker main pair with an already chosen object."""
    if not chosen:
        return False
    # ponytail: a full check per trial, O(n^2) per call; fine for <= ~60 objects x 9 samples.
    ids = {o['id'] for o in unit}
    return any((a in ids) != (b in ids) for a, b in validate.check({'objects': chosen + unit})['collisions']['main'])


def recombine(room, request, prior, samples, scored):
    """Section 1.1 recombination ceiling: per floor-root subtree the first non-colliding sampled placement."""
    complete = [k for k, (row, *_) in enumerate(scored) if row['complete']]
    if not complete:
        return {'complete': False, 'checker': False, 'admission': False, 'pass': False, 'units': None,
                'units_clean': None, 'structure_k': None, 'picks': None}
    records = {k: h.strict_json(samples[k]['raw'])['placements'] for k in complete}
    layouts = {k: {o['id']: o for o in scored[k][0]['raw_layout']} for k in complete}
    base, chosen, picks, clean = complete[0], [], [], 0
    units = subtrees(records[base])
    for unit in units:
        pick = next((k for k in complete if not _crosses([layouts[k][i] for i in unit], chosen)), None)
        clean += pick is not None
        pick = base if pick is None else pick
        chosen += [layouts[pick][i] for i in unit]
        picks.append(pick)
    source = {i: k for unit, k in zip(units, picks) for i in unit}
    text = contract._json({'placements': [records[source[p['id']]][n] for n, p in enumerate(records[base])]})
    row, _, verdict = op._sample(room, request, prior, {'k': None, 'raw': text, 'error': None, 'finish_reason': 'stop'})
    done = row['complete']
    return {'complete': done, 'checker': bool(done and row['geometry_success']),
            'admission': bool(done and verdict['accepted']), 'pass': bool(done and op._passes(row, verdict)),
            'units': len(units), 'units_clean': clean, 'structure_k': samples[base]['k'],
            'picks': dict(sorted(Counter(samples[k]['k'] for k in picks).items()))}


def _mine_room(task):
    room, request, prior, samples = task
    scored = [op._sample(room, request, prior, s) for s in samples]
    out = []
    for s, (row, _, verdict) in zip(samples, scored):
        done = row['complete']
        x = {'k': s['k'], 'finish_reason': s['finish_reason'], 'complete': done,
             'checker': bool(done and row['geometry_success']), 'admission': bool(done and verdict['accepted'])}
        x['pass'] = x['checker'] and x['admission']
        if done:
            categories = [scene.norm_cat(o['category']) for o in row['raw_layout']]
            x.update({**copies(h.strict_json(s['raw'])['placements'], categories), **colocated(row['raw_layout'])})
        out.append(x)
    return {'uid': room['uid'], 'source': room['source'], 'group': room['group'],
            'bin': bo.bin_name(len(room['objects'])), 'samples': out, 'any_pass': any(x['pass'] for x in out),
            'recombination': recombine(room, request, prior, samples, scored)}


def _copies(samples):
    done = [s for s in samples if s['complete']]
    records, n = sum(s['records'] for s in done), sum(s['exact_copies'] for s in done)
    pairs = sum(s['colocated_pairs'] for s in done)
    return {'complete_samples': len(done), 'records': records, 'exact_copies': n,
            'exact_copy_rate': n / records if records else None,
            'copy_of_previous_share': sum(s['copy_of_previous'] for s in done) / n if n else None,
            'copy_same_category_share': sum(s['copy_same_category'] for s in done) / n if n else None,
            'samples_with_copy': _share(s['exact_copies'] > 0 for s in done),
            'samples_with_colocated': _share(s['colocated_pairs'] > 0 for s in done), 'colocated_pairs': pairs,
            'colocated_exact_share': sum(s['colocated_exact'] for s in done) / pairs if pairs else None}


def _ident(room, values):
    return {**{k: room[k] for k in ('uid', 'source', 'group', 'bin')}, **{f: values[f] for f in FIELDS}}


def _class(sample):
    return 'pass' if sample['pass'] else 'reject' if sample['complete'] else 'incomplete'


def _agreement(pairs, rooms):
    d._verified_dir(pairs)
    theirs = {x['uid']: [s['class'] for s in x['samples']]
              for x in map(h.strict_json, (pairs / 'ledger.jsonl').read_text().splitlines())}
    if theirs.keys() != {r['uid'] for r in rooms}:
        raise ValueError('the stage2-pairs ledger covers other rooms')
    differ = [(r['uid'], s['k'], theirs[r['uid']][n], _class(s)) for r in rooms for n, s in enumerate(r['samples'])
              if theirs[r['uid']][n] != _class(s)]
    return {'path': str(pairs.resolve()), 'manifest_sha256': h.sha256(pairs / 'MANIFEST.json'),
            'samples': sum(len(r['samples']) for r in rooms), 'differ': len(differ),
            'examples': [dict(zip(('uid', 'k', 'ledger', 'here'), x)) for x in differ[:20]]}


def ledger(args):
    data, samples, out = Path(args.data), Path(args.samples), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    manifest = e._verified(data)
    def rows(name):
        return h._by_uid(h._rows(data, name, manifest['files'][name]))
    rooms, requests, priors = rows('rooms.jsonl'), rows(f'{ARM}/requests.jsonl'), rows('size-estimates.jsonl')
    receipt, grouped = op._sampling(data, samples, manifest)
    uids = manifest['expected_uids']
    mined = r2._map(_mine_room, [(rooms[u], requests[u], priors[u]['estimated_sizes'], grouped[u]) for u in uids],
                    args.workers)
    greedy = [_ident(r, r['samples'][0]) for r in mined]
    first = [_ident(r, r['samples'][1]) for r in mined]
    recombined = [_ident(r, r['recombination']) for r in mined]
    clean = [r['recombination'] for r in mined if r['recombination']['units']]
    result = {
        'step': 'ledger', 'status': 'A021_LEDGER_COMPLETE', 'rooms': len(mined), 'k': receipt['sampling']['n_samples'],
        'greedy': _by_bin(greedy, _rates), 'k1': _by_bin(first, _rates),
        'k1_minus_greedy': r2.contrasts(first, greedy, FIELDS, REPEATS),
        'copies': {'greedy': _by_bin([{'bin': r['bin'], 's': r['samples'][0]} for r in mined],
                                     lambda x: _copies([y['s'] for y in x])),
                   'all_k': _copies([s for r in mined for s in r['samples']])},
        'recombination': _by_bin(recombined, _rates),
        'recombination_minus_greedy': r2.contrasts(recombined, greedy, FIELDS, REPEATS),
        'best_of_k': _by_bin(mined, lambda x: {'rooms': len(x), 'pass': _share(r['any_pass'] for r in x)}),
        'recombination_units': {'rooms': len(clean), 'units': sum(r['units'] for r in clean),
                                'units_clean': sum(r['units_clean'] for r in clean),
                                'structure_not_greedy': sum(r['structure_k'] != 0 for r in clean)},
        'ledger_agreement': _agreement(Path(args.pairs), mined) if args.pairs else None,
        'inputs': {'data': str(data.resolve()), 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
                   'scope': manifest.get('scope'), 'samples': str(samples.resolve()),
                   'samples_manifest_sha256': h.sha256(samples / 'MANIFEST.json'), 'samples_run_sha256': receipt['run_sha256'],
                   'sampling': receipt['sampling'], 'policy': receipt['model'].get('policy'),
                   'lineage_sha256': receipt['model'].get('lineage_sha256')},
        'code_sha256': _code(), 'descriptive_only': True}
    _publish(out, result, mined)
    return {k: result[k] for k in ('status', 'rooms')}


# ---- 1.2 P-only admission pool -------------------------------------------------------------------

def pool(args):
    """Section 1.2 from amendment 021's train pool (a021_train.pool, one implementation): its rebuild re-verifies every
    P-only ledger entry (main_data._candidate on the frozen corpus; any difference fails that job) and its `pool` flag
    is P's target passing the frozen checker (geometry.score geometry_success)."""
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    value = t.load_pool(args.pool)
    summary = value['summary']
    rows = [e for e in value['ledger'] if e['origin'] == 'p_only']

    def table(key):
        cells = Counter((r[key], r['pool']) for r in rows)
        return {v: {'rooms': cells[v, True] + cells[v, False], 'checker_pass': cells[v, True]}
                for v in sorted({r[key] for r in rows})}  # Bin names sort in bin order.
    result = {'step': 'pool', 'status': 'A021_POOL_COMPLETE', 'p_only': len(rows),
              'checker_pass': sum(r['pool'] for r in rows), 'p_only_expected': summary['p_only_expected'],
              'admission_counts': summary['admission_counts'], 'rebuild': summary['rebuild'],
              'lineage': summary['lineage'], 'by_source': table('source'), 'by_bin': table('bin'),
              'by_family': table('family'),
              'inputs': {'pool': str(value['root'].resolve()), 'pool_manifest_sha256': value['manifest_sha256'],
                         **summary['inputs'], 'pool_code_sha256': summary['code_sha256']},
              'code_sha256': _code(), 'descriptive_only': True}
    _publish(out, result, sorted(r['uid'] for r in rows if r['pool']), 'uids.json')
    return {k: result[k] for k in ('status', 'p_only', 'checker_pass')}


# ---- 1.3 prefix-intervention probe ---------------------------------------------------------------

def _head(placements, k):
    """contract._json's text of {'placements': [...]} up to where record k starts."""
    return '{"placements":[' + ''.join(contract._json(p) + ',' for p in placements[:k])


def _ir(objects, records):
    by = {o['id']: o for o in objects}
    return [{'id': r['id'], 'category': by[r['id']]['category'], 'size': by[r['id']]['size'], 'pos': r['pos'],
             'yaw': math.radians(r['yaw']), 'parent': r.get('on')} for r in records]


def hits(objects, prefix, record, other=None):
    """Whether a record forms a checker main pair with an earlier record (with `other`, with that id only)."""
    pairs = validate.check({'objects': _ir(objects, prefix + [record])})['collisions']['main']
    return any(record['id'] in p and (other is None or other in p) for p in pairs)


def probe_item(room, request, record):
    """(None, item) for a usable room, else (reason, None)."""
    if record['reached_token_cap'] and not record['stopped_on_eos']:
        return 'token_cap', None
    raw = record['raw']
    try:
        contract.validate_output(raw, room, ARM)
    except ValueError:
        return 'invalid_output', None
    placements = h.strict_json(raw)['placements']
    if contract._json({'placements': placements}) != raw:
        return 'non_canonical', None
    objects = h.strict_json(request['messages'][1]['content'])['objects']
    for k in range(1, len(placements)):
        same = [j for j in range(k) if objects[j]['category'] == objects[k]['category']
                and placements[j].get('on') == placements[k].get('on')]
        if not same or placements[same[-1]]['pos'] == placements[k]['pos']:
            continue
        j, p_k = same[-1], placements[k]['pos']
        moved = [{**p, 'pos': list(p_k)} if n == j else p for n, p in enumerate(placements)]
        if not hits(objects, moved[:k], placements[k], placements[j]['id']):
            continue
        p_j = placements[j]['pos']
        conditions = {}
        for name, records, at, other in (('control', placements, p_j, p_k), ('intervention', moved, p_k, p_j)):
            conditions[name] = {'full': contract._json({'placements': records}), 'start': len(_head(records, k)),
                                'prefix': records[:k], 'colliding_pos': list(at), 'clean_pos': list(other)}
        return None, {'uid': room['uid'], 'source': room['source'], 'group': room['group'],
                      'bin': bo.bin_name(len(room['objects'])), 'k': k, 'j': j, 'category': objects[k]['category'],
                      'messages': request['messages'], 'input_tokens': record['input_tokens'], 'objects': objects,
                      'record': placements[k], 'conditions': conditions}
    return 'no_colliding_sibling', None


def probe_items(rooms, requests, records, n):
    items, skipped = [], Counter()
    for uid in sorted(records, key=lambda u: (hashlib.sha256((PROBE_SALT + u).encode()).hexdigest(), u)):
        if len(items) == n:
            break
        reason, item = probe_item(rooms[uid], requests[uid], records[uid])
        if item is None:
            skipped[reason] += 1
        else:
            items.append(item)
    return items, dict(sorted(skipped.items()))


def _encode(tok, text):
    value = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(value['input_ids']), [tuple(x) for x in value['offset_mapping']]


def _parse_at(text, head):
    if not text.startswith(head):
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text, len(head))
    except ValueError:
        return None
    pos = value.get('pos') if isinstance(value, dict) else None
    ok = isinstance(pos, list) and len(pos) == 3 and all(type(x) in (int, float) for x in pos)
    return {**value, 'pos': pos} if ok else None


def probe_room(item, tok, generate, logprob):
    prompt = infer.prompt_ids(tok, {'messages': item['messages']})
    if len(prompt) != item['input_tokens']:
        raise ValueError(f"prompt tokens differ from the recorded inference (template drift): {item['uid']}")
    result = {k: item[k] for k in ('uid', 'source', 'group', 'bin', 'k', 'j', 'category')}
    for name in CONDITIONS:
        c = item['conditions'][name]
        ids, offsets = _encode(tok, c['full'])
        n = sum(end <= c['start'] for _, end in offsets)  # Tokens wholly before record k.
        cut = offsets[n - 1][1] if n else 0
        prefix = prompt + ids[:n]
        conts = {kind: _encode(tok, c['full'][cut:c['start']] + contract._json({**item['record'], 'pos': c[kind + '_pos']}))[0]
                 for kind in ('colliding', 'clean')}  # Record k's continuations; greedy budget = the longer + SLACK.
        text = c['full'][:cut] + tok.decode(generate(prefix, max(map(len, conts.values())) + SLACK),
                                            skip_special_tokens=False)
        found = _parse_at(text, c['full'][:c['start']])
        pos = found and found['pos']
        logp = {kind: logprob(prefix + cont, len(cont)) for kind, cont in conts.items()}
        result[name] = {
            'parsed': pos is not None, 'pos': pos, 'at_colliding': None if pos is None else pos == c['colliding_pos'],
            'at_clean': None if pos is None else pos == c['clean_pos'],
            'collides': None if pos is None else hits(item['objects'], c['prefix'], {**item['record'], 'pos': pos}),
            'clean_candidate_collides': hits(item['objects'], c['prefix'], {**item['record'], 'pos': c['clean_pos']}),
            'logp_colliding': logp['colliding'], 'logp_clean': logp['clean'],
            'delta': logp['colliding'] - logp['clean'],
            'move_xy_m': None if pos is None else math.dist(pos[:2], item['record']['pos'][:2])}
    return result


def _condition(rows):
    deltas = [x['delta'] for x in rows]
    return {'parsed': _share(x['parsed'] for x in rows), 'at_colliding': _share(x['at_colliding'] for x in rows),
            'at_clean': _share(x['at_clean'] for x in rows), 'collides': _share(x['collides'] for x in rows),
            'clean_candidate_collides': _share(x['clean_candidate_collides'] for x in rows),
            'delta_mean': statistics.mean(deltas) if deltas else None,
            'delta_median': statistics.median(deltas) if deltas else None,
            'delta_positive': _share(x > 0 for x in deltas)}


def probe_summary(results):
    by = {name: [r[name] for r in results] for name in CONDITIONS}
    moved = [x['move_xy_m'] for x in by['intervention'] if x['parsed'] and not x['at_colliding']]
    rows = {name: [{**{k: r[k] for k in ('uid', 'source', 'group')},
                    **{f: r[name][f] for f in ('at_colliding', 'collides', 'delta')}} for r in results]
            for name in CONDITIONS}
    return {'rooms': len(results), **{name: _condition(v) for name, v in by.items()},
            'control_reproduces_greedy': _share(x['at_clean'] for x in by['control']),
            'intervention_moved': _share(None if x['at_colliding'] is None else not x['at_colliding']
                                         for x in by['intervention']),
            'intervention_move_xy_median_m': statistics.median(moved) if moved else None,
            'intervention_minus_control': r2.contrasts(rows['intervention'], rows['control'],
                                                       ('at_colliding', 'collides', 'delta'), REPEATS) if results else None}


def _dev_records(data, ref):
    """dev-select rooms/requests and rpo-dpo1's HF exact-policy records there, bound to the policy the probe loads."""
    manifest, rooms, requests = ds._dev(data)
    path = ref / REFERENCE / f'infer-{ARM}'
    records, receipt = e._inference(path, data, manifest['expected_uids'])
    if receipt['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
        raise ValueError('dev inference used other training data')
    ds._bind(receipt, ref.parent, ARM, REFERENCE)
    _, policy = d.exact_policy(ds.model_path(ref.parent, ARM, REFERENCE), ARM)
    if receipt.get('policy') != policy:
        raise ValueError('the dev records were not written by the policy the probe loads')
    inputs = {'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'), 'records': str(path.resolve()),
              'records_run_sha256': receipt['run_sha256'], 'policy': policy}
    return rooms, requests[ARM], {r['uid']: r for r in records}, policy, inputs


def _hf(policy):  # pragma: no cover - GPU only. infer._runtime's exact-policy load; it does not expose the model.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('requires exactly one BF16 GPU')
    tok = AutoTokenizer.from_pretrained(policy['adapter'], local_files_only=True, trust_remote_code=False)
    eos = tok.get_vocab()[d.EOS]
    base = AutoModelForCausalLM.from_pretrained(policy['training_base'], local_files_only=True, trust_remote_code=False,
                                                dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0})
    model = PeftModel.from_pretrained(base, policy['adapter'], local_files_only=True, is_trainable=False,
                                      autocast_adapter_dtype=True).eval()

    def generate(ids, n):
        x = torch.tensor([ids], device='cuda:0')
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=n, eos_token_id=eos, pad_token_id=eos,
                                  use_cache=True)
        with torch.inference_mode():
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=config,
                                 logits_to_keep=1)
        return out[0, len(ids):].tolist()

    def logprob(ids, n):
        x = torch.tensor([ids], device='cuda:0')
        with torch.inference_mode():
            logits = model(input_ids=x, logits_to_keep=n + 1).logits[0, :-1].float()
        return torch.log_softmax(logits, -1).gather(1, x[0, -n:, None]).sum().item()
    return tok, generate, logprob


def probe(args):
    data, ref, out = Path(args.data), Path(args.ref_root), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    started = time.monotonic()
    rooms, requests, records, policy, inputs = _dev_records(data, ref)
    items, skipped = probe_items(rooms, requests, records, args.rooms)
    tok, generate, logprob = _hf(policy)
    deadline, results = started + 60 * args.max_minutes, []  # Loading counts against the budget.
    for item in items:
        if time.monotonic() > deadline:
            break
        results.append(probe_room(item, tok, generate, logprob))
    result = {'step': 'probe', 'status': 'A021_PROBE_COMPLETE', 'selected': len(items),
              'not_run_time_limit': len(items) - len(results), 'skipped': skipped, **probe_summary(results),
              'settings': {'rooms': args.rooms, 'max_minutes': args.max_minutes,
                           'max_new_tokens': f'record k continuation + {SLACK}',
                           'salt': PROBE_SALT, 'decoding': 'HF greedy, batch 1, BF16 base + FP32 adapter, SDPA'},
              'gpu_hours': (time.monotonic() - started) / 3600, 'inputs': inputs, 'code_sha256': _code(),
              'descriptive_only': True}
    _publish(out, result, results)
    return {k: result[k] for k in ('status', 'rooms', 'not_run_time_limit')}


# ---- Chinese reports -----------------------------------------------------------------------------

BIN_ROWS = ('all',) + BINS
RATE_COLUMNS = [('scope', '分箱'), ('rooms', '房间'), ('complete', '完整率'), ('checker', '检查器成功'),
                ('admission', '严格准入'), ('pass', '两者通过')]
CONTRAST_COLUMNS = [('field', '指标'), ('value', '差值 [95% 房屋聚类 CI]')]
NAMES = {'complete': '完整率', 'checker': '检查器成功', 'admission': '严格准入', 'pass': '两者通过',
         'at_colliding': '落在碰撞位置', 'collides': '与前缀碰撞', 'delta': 'Δlogp（碰撞 − 干净）'}


def _bins(table):
    return [{'scope': '全部' if b == 'all' else b, **table[b]} for b in BIN_ROWS if b in table and table[b]['rooms']]


def _contrast(table):
    return r2._md([{'field': NAMES[f], 'value': v} for f, v in table.items()], CONTRAST_COLUMNS)


def _ledger_report(r):
    agreement = r['ledger_agreement']
    lines = ['# 修订 021 §1.1：013 样本账本挖掘（描述性，不门控、不选择）', '',
             f"- 输入：dpo-004 train-subset 共 {r['rooms']} 间训练房；每间 k = 0（vLLM 贪心）与 k = 1..{r['k']}（T 0.7）。",
             '- 评分：onpolicy_dpo._sample（冻结检查器 + downstream.accept 严格准入，raw 布局；length 截断记失败）；'
             '两者通过 = 013 的 pass。比率分母为全部房间（不完整 = 失败）。',
             '- 与 013 stage2-pairs ledger 的逐样本类别一致性：' + (
                 '未提供。' if agreement is None else f"{agreement['samples'] - agreement['differ']}/{agreement['samples']} 一致。"),
             '', '## 贪心（k = 0）按物体数分箱', *r2._md(_bins(r['greedy']), RATE_COLUMNS),
             '## k = 1 与贪心', *r2._md(_bins(r['k1']), RATE_COLUMNS), 'k = 1 − 贪心（房屋聚类自助法，2,000 次，种子 42）：', '',
             *_contrast(r['k1_minus_greedy']),
             '## 精确复制与同位重复（仅完整样本）',
             '精确复制 = 某条记录的 pos 与 yaw 与同一输出中更早的一条完全相同；同位重复 = onpolicy_dpo.tier 规则'
             '（同一归一化类别、同一父对象的 main/severe/contained 检查器对）。', '']
    copy_rows = [{'scope': '贪心 ' + ('全部' if b == 'all' else b), **r['copies']['greedy'][b]} for b in BIN_ROWS
                 if r['copies']['greedy'][b]['complete_samples']] + [{'scope': '全部 k', **r['copies']['all_k']}]
    lines += r2._md(copy_rows, [('scope', '范围'), ('complete_samples', '完整样本'), ('records', '记录'),
                                ('exact_copy_rate', '精确复制记录率'), ('copy_of_previous_share', '其中复制紧邻前一条'),
                                ('copy_same_category_share', '其中同类别'), ('samples_with_copy', '含精确复制的样本'),
                                ('samples_with_colocated', '含同位重复的样本'), ('colocated_pairs', '同位重复对'),
                                ('colocated_exact_share', '其中精确复制')])
    units = r['recombination_units']
    lines += ['## 支撑子树重组上限',
              '单元 = 最低 k 完整样本（通常为贪心）的地面根支撑子树；按根的请求顺序，每个单元取第一个（k 升序）与已选单元'
              '无检查器 main 碰撞的样本放置，否则取该最低 k 样本；拼合后的记录（含 on）按样本同样评分。这是按 memo D4 规则在支撑子树'
              '层面顺序贪心（critique 把 D4 从逐物体改到子树层；修订 021 只写"支撑子树层面的重组上限"）：第一个单元总取最低 k 样本，'
              '只替换有跨单元 main 碰撞的单元；单元内部的越界、支撑、固定物阻挡与下游准入失败都不修复，结果依赖单元顺序，'
              '因此是重组上限的下界。最佳 9 选 1 = 任一 k 通过。', '',
              f"- 单元：{units['units']}，其中找到无碰撞放置 {units['units_clean']}；结构样本非贪心的房间 {units['structure_not_greedy']}。", '']
    rows = [{'scope': '全部' if b == 'all' else b, 'greedy': r['greedy'][b]['pass'], 'recomb': r['recombination'][b]['pass'],
             'recomb_checker': r['recombination'][b]['checker'], 'best': r['best_of_k'][b]['pass'],
             'rooms': r['greedy'][b]['rooms']} for b in BIN_ROWS if r['greedy'][b]['rooms']]
    lines += r2._md(rows, [('scope', '分箱'), ('rooms', '房间'), ('greedy', '贪心两者通过'), ('recomb', '重组两者通过'),
                           ('recomb_checker', '重组检查器成功'), ('best', '9 选 1 两者通过')])
    lines += ['重组 − 贪心：', '', *_contrast(r['recombination_minus_greedy'])]
    return '\n'.join(lines) + '\n'


def _pool_report(r):
    columns = [('value', '取值'), ('rooms', '房间'), ('checker_pass', 'P 目标过冻结检查器')]

    def table(key):
        return r2._md([{'value': k, **v} for k, v in r[key].items()], columns)
    counts, rebuild = r['admission_counts'], r['rebuild']
    lines = ['# 修订 021 §1.2：仅 P 准入池（描述性）', '',
             '- 来源：修订 021 训练池（a021_train pool，RUN/a021/pool）的账本，未另行重跑；本步只做拆分与报告。',
             '- 定义：common-admission.jsonl 中 P 目标合格（placement.qualified）而 J 的 GT 目标不合格的训练房，'
             '即配对设计仅因 J 失败而丢弃的房间。',
             f"- 数量：{r['p_only']}（预期 {r['p_only_expected']} = placement_qualified {counts['placement_qualified']} − "
             f"common_decoder_train {counts['common_decoder_train']}；该作业已与谱系审计及 main-data 清单逐项核对，不一致即报错）。",
             f"- 复核：该作业在冻结语料上重跑 main_data._candidate，{rebuild['p_only_ledger_equal']} 间 P-only 账本逐字段相同"
             f"（不同即报错）；另抽 {rebuild['accepted_sample_byte_equal']} 间已接受房逐字节复现 P 的 SFT 行。"
             f"P 目标过冻结检查器（geometry.score 的 geometry_success）：{r['checker_pass']}/{r['p_only']}。",
             '', '## 按来源', *table('by_source'), '## 按物体数分箱', *table('by_bin'), '## 按房间族', *table('by_family')]
    return '\n'.join(lines) + '\n'


def _probe_report(r):
    rows = [{'scope': '对照' if n == 'control' else '干预', **r[n]} for n in CONDITIONS if r.get(n)]
    lines = ['# 修订 021 §1.3：前缀干预探针（描述性）', '',
             f"- 策略：rpo-dpo1 精确策略（HF，BF16 基座 + FP32 适配器，贪心，批大小 1）；dev-select 按哈希顺序选出 {r['selected']} 间，"
             f"实际运行 {r['rooms']} 间（超时未运行 {r['not_run_time_limit']}）；跳过："
             f"{'、'.join(f'{k} {v}' for k, v in r['skipped'].items()) or '无'}。",
             '- 每间选第一个记录 k：其前面有同类别、同父对象的兄弟 j（取最近者），pos_j ≠ pos_k，且把 j 移到 P 预测的 pos_k 后'
             '记录 k 在 pos_k 处与 j 构成检查器 main 碰撞。',
             '- 对照前缀 = P 自己贪心输出中记录 k 之前的部分；干预前缀 = 同一前缀但 j 移到 pos_k。前缀在记录 k 之前最后一个分词边界处截断。',
             '- 碰撞续写 = 记录 k 落在兄弟当前位置（对照 pos_j，干预 pos_k）；干净续写 = 另一位置；两者只有 pos 不同。'
             f"贪心续写上限 = 两种续写中较长者的 token 数 + {SLACK}；时限从作业开始计（含加载）。",
             f"- GPU 时：{r['gpu_hours']:.3f}。", '']
    lines += r2._md(rows, [('scope', '条件'), ('parsed', '可解析'), ('at_colliding', '贪心落在碰撞位置'),
                           ('at_clean', '贪心落在干净位置'), ('collides', '贪心与前缀碰撞'),
                           ('clean_candidate_collides', '干净候选与前缀碰撞'), ('delta_mean', 'Δlogp 均值'),
                           ('delta_median', 'Δlogp 中位数'), ('delta_positive', 'Δlogp > 0 的比例')])
    lines += [f"- 对照复现贪心记录 k：{r2._cell(r['control_reproduces_greedy'])}；干预下记录 k 移动："
              f"{r2._cell(r['intervention_moved'])}（移动距离中位数 {r2._cell(r['intervention_move_xy_median_m'])} m）。", '']
    if r['intervention_minus_control']:
        lines += ['干预 − 对照（房屋聚类自助法，2,000 次，种子 42）：', '', *_contrast(r['intervention_minus_control'])]
    return '\n'.join(lines) + '\n'


REPORTS = {'ledger': _ledger_report, 'pool': _pool_report, 'probe': _probe_report}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='step', required=True)
    p = sub.add_parser('ledger')
    p.add_argument('--data', required=True)
    p.add_argument('--samples', required=True)
    p.add_argument('--pairs')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=1)
    p = sub.add_parser('pool')
    p.add_argument('--pool', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('probe')
    p.add_argument('--data', required=True)
    p.add_argument('--ref-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rooms', type=int, default=PROBE_ROOMS)
    p.add_argument('--max-minutes', type=float, default=24.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps({'ledger': ledger, 'pool': pool, 'probe': probe}[args.step](args)))


if __name__ == '__main__':
    main()
