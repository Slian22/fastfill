"""S2 (amendment-011 draft): score every vLLM sample with the frozen checker and pick one per room; no GT in the rule.

python -m fastfill.comparison.best_of_n select --data EVAL --samples OUT --arm A --out SEL
    [--ks 1,2,4,8] [--workers 32] [--reference HF_INFER_DIR]
python -m fastfill.comparison.best_of_n calibrate --data RUN/dev-select-v1 \
    --geometry RUN/dev-eval-010/scores/joint-sft-geometry.jsonl --bands 0.5,2 0.67,1.5 0.75,1.33

select     Reads a finalized sample_vllm run (SAMPLING_COMPLETE, hash-verified, this arm, this data, a model trained on
           this data's training manifest). Every sample (uid, k) is scored by geometry.score on the request the model
           received; a sample that stopped on 'length' goes through report_v2.cap_failure (amendment 005). Per room and
           m in {0} + ks the rule (RULE) picks among k <= m the complete samples with the fewest violations (sum of
           VIOLATIONS; severe pairs also count in main, by design), ties to the lower k (greedy wins ties). Greedy is in
           every pool; an alternative k >= 1 must also pass the guards against greedy (output-review item 2): the
           per-object size gate against size-estimates.jsonl (the train prior), contained pairs, flat objects and
           same-category stacks each <= greedy's (0 when greedy is incomplete), and footprint bias vs the prior >=
           greedy's - BIAS_TOL when greedy is complete. P sizes are the prior itself, so gate, flat and bias are
           identities there. Nothing complete and admissible: greedy ('fallback_greedy'). 0 violations =
           geometry_success. The rule reads only the request (boundary, fixed objects), the boxes and the train prior,
           so the backend can run it at serve time. The guards are fixed per candidate (greedy is fixed), so the pool
           only grows with m and the selected violations never rise with m. Exploit counts (EXPLOITS): contained pairs
           (validate.check), flat objects (predicted height < 6 cm, prior >= 6 cm), same-category stacks (on a movable
           parent of the same normalized category), parent changed vs GT (reported only).
           Published write-once: ARM-greedy-geometry.jsonl and ARM-boM-geometry.jsonl (M = largest m; full
           geometry.score rows + sample_index, selection, violations, gate_fail, exploit counts and cap flags: the
           evaluate_llm row layout, so geometry.summarize / paired_intervals read them and S1 takes raw_layout),
           ARM-samples.jsonl (compact checker results of every sample, no layouts), summary.json (per m: overall, per
           object-count bin and per source: complete / success / repaired success rates, mean violations, size log MAE,
           fallback rooms, rooms whose pick the guards changed, exploit totals, footprint bias B vs the prior and vs GT
           and its change selected - greedy; plus the fidelity block, the rule, the band, the sample manifest SHA and
           code hashes) and MANIFEST.json.
           --reference: the HF inference dir of the same selected model on dev-select-v1 (dev-eval-010/CANDIDATE/
           infer-ARM). Checked before any scoring: same model, same training manifest, same rooms and request messages,
           same prompt token counts (the vLLM env renders the prompts with its own transformers). vLLM computes LoRA in
           BF16, so the fidelity block reports how often the vLLM greedy text equals the HF text and the discordant
           rooms b (only HF passes) and c (only vLLM greedy passes), each also counted where either side hit its token
           cap (HF: a per-batch cap; S2: a per-room cap). S2 gains are always vLLM selected vs vLLM greedy, never vs HF
           numbers; no LLM baseline has N samples yet, so no J/P-vs-L claim follows.
calibrate  Before the band is locked: per object-count bin, the share of complete J greedy rooms of dev-eval-010 whose
           every object passes each candidate band; a bin below TOO_STRICT means that band is too strict for it. A loose
           band is bounded by the other guards in select, not here.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics

from fastfill import scene
from fastfill import validate
from fastfill.comparison import contract
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h

VIOLATIONS = ('oob_objects', 'support_fail', 'ceiling_fail', 'main_collisions', 'severe_collisions', 'fixed_blocking')
BAND, BINS = (0.67, 1.5), ((1, 10), (11, 30), (31, 50), (51, None))
GUARDED = ('contained_pairs', 'flat_objects', 'stacked_duplicates')  # Each <= greedy's for an alternative.
EXPLOITS = (*GUARDED, 'parent_changed')
CAP_FLAGS = ('finish_reason', 'reached_token_cap', 'stopped_on_eos')
COMPACT = ('uid', 'source', 'group', 'arm', 'objects', 'complete', 'error', 'geometry_success',
           'geometry_success_repaired', *VIOLATIONS, 'oor', 'size_log_mae', 'size_iou')
MODEL_KEYS = ('base_sha256', 'adapter_sha256', 'policy')
FLAT, TOO_STRICT, EPS = 0.06, 0.9, 1e-12  # FLAT: validate.check ignores boxes under 6 cm in collisions.
BIAS_TOL = 0.03  # Footprint log-bias an alternative may lose vs greedy (output-plan B <= 3%).
RULE = ('per room, among samples k <= m (k = 0 greedy) that are complete (strict parse, not cut by the token cap), the '
        'fewest checker violations (oob + support + ceiling + main + severe + fixed_blocking), ties to the lower k. '
        'Greedy is in every pool; an alternative k >= 1 must also pass the guards against greedy: every object x, y '
        'and z within [{lo}, {hi}] x its train-prior size (inclusive); contained pairs, flat objects (< 6 cm, prior >= '
        "6 cm) and same-category stacks each <= greedy's (0 if greedy is incomplete); mean log footprint vs the prior "
        ">= greedy's - {tol} if greedy is complete. Nothing complete and admissible: greedy (fallback_greedy). "
        'Inputs: the request (boundary, fixed objects), the predicted boxes and the train prior; no GT.')


def violations(row):
    """Sum of VIOLATIONS; None if incomplete (severe pairs are also main pairs, by design)."""
    return sum(row[k] for k in VIOLATIONS) if row['complete'] else None


def size_gate(layout, prior, band=BAND):
    """Ids whose predicted x, y or z leaves [lo, hi] x the train-prior size (inclusive)."""
    lo, hi = band
    return [o['id'] for o in layout if not all(lo - EPS <= p / q <= hi + EPS for p, q in zip(o['size'], prior[o['id']]))]


def exploits(row, room, prior):
    if not row['complete']:
        return dict.fromkeys(EXPLOITS)
    layout = row['raw_layout']
    by_id = {o['id']: o for o in layout}
    return {'contained_pairs': len(validate.check({'objects': layout})['collisions']['contained']),
            'flat_objects': sum(o['size'][2] < FLAT <= prior[o['id']][2] for o in layout),
            'stacked_duplicates': sum(o.get('parent') in by_id and scene.norm_cat(o['category']) ==
                                      scene.norm_cat(by_id[o['parent']]['category']) for o in layout),
            'parent_changed': sum(o.get('parent') != gt.get('parent') for o, gt in zip(layout, room['objects']))}


def admissible(c, greedy):
    """A complete alternative passes the guards against greedy (output-review item 2); greedy itself is never guarded."""
    done = greedy['complete']
    return (not c['gate_fail'] and all(c[k] <= (greedy[k] if done else 0) for k in GUARDED)
            and (not done or c['bias_prior'] >= greedy['bias_prior'] - BIAS_TOL))


def select(cands, m, guarded=True):
    """cands: one room's compact rows k = 0..N -> (k, reason); guarded=False only measures what the guards change."""
    greedy = cands[0]
    if greedy['k'] != 0:
        raise ValueError('candidates must start with greedy (k = 0)')
    pool = [c for c in cands if c['k'] <= m and c['complete'] and (c['k'] == 0 or not guarded or admissible(c, greedy))]
    if not pool:
        return 0, 'fallback_greedy'  # Greedy incomplete and no admissible alternative.
    best = min(pool, key=lambda c: (c['violations'], c['k']))  # Greedy wins ties.
    return best['k'], 'greedy' if best['k'] == 0 else 'fewest_violations'


def score_room(task):
    """All samples of one uid -> compact rows, the pick per m, and the greedy / largest-m full rows."""
    room, request, prior, samples, arm, band, ms = task
    if any(s[k] != room[k] or request[k] != room[k] for s in samples for k in ('uid', 'source', 'group')):
        raise ValueError('sample/request/reference identity mismatch')
    reference = [{'size': prior[o['id']]} for o in room['objects']]
    full, compact = [], []
    for s in samples:
        row = g.score(room, request['messages'], s['raw'], arm, s['error'])
        if s['finish_reason'] == 'length':
            row = r2.cap_failure(row)
        done = row['complete']
        extra = {'violations': violations(row), 'gate_fail': size_gate(row['raw_layout'], prior, band) if done else None,
                 **exploits(row, room, prior), **{k: s[k] for k in CAP_FLAGS}}
        full.append({**row, 'sample_index': s['k'], **extra})
        compact.append({**{k: row.get(k) for k in COMPACT}, 'k': s['k'], **extra,
                        'bias_prior': r2.size_bias(reference, row['raw_layout'])[0] if done else None,
                        'bias_gt': r2.size_bias(room['objects'], row['raw_layout'])[0] if done else None})
    picks = {m: select(compact, m) for m in ms}
    return {'compact': compact, 'picks': picks,
            'guard_changed': {m: select(compact, m, guarded=False)[0] != picks[m][0] for m in ms},
            'greedy': {**full[0], 'selection': picks[0][1]},
            'selected': {**full[picks[ms[-1]][0]], 'selection': picks[ms[-1]][1]}}


def bin_name(objects):
    for lo, hi in BINS:
        if objects >= lo and (hi is None or objects <= hi):
            return f'{lo}-{hi}' if hi else f'{lo}+'
    raise ValueError('room without movable objects')


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _stats(picked):
    """picked: (selected compact row, greedy compact row, reason, the guards changed the pick) per room."""
    n, done = len(picked), [s for s, *_ in picked if s['complete']]
    both = [(s, g0) for s, g0, *_ in picked if s['complete'] and g0['complete']]
    return {'requests': n, 'complete_rate': len(done) / n,
            'success_rate': sum(s['geometry_success'] for s, *_ in picked) / n,
            'repaired_success_rate': sum(s['geometry_success_repaired'] for s, *_ in picked) / n,
            'mean_violations': _mean(s['violations'] for s in done), 'size_log_mae': _mean(s['size_log_mae'] for s in done),
            'fallback_rooms': sum(r == 'fallback_greedy' for _, _, r, _ in picked),
            'guard_changed_rooms': sum(changed for *_, changed in picked),
            'changed_from_greedy': sum(s['k'] != 0 for s, *_ in picked),
            'cap_failures': sum(s['error'] == r2.CAP for s, *_ in picked),
            'exploits': {k: sum(s[k] for s in done) for k in EXPLOITS},
            'bias_fp': {ref: _mean(s['bias_' + ref] for s in done) for ref in ('prior', 'gt')},
            'delta_bias_fp_vs_greedy': {**{ref: _mean(s['bias_' + ref] - g0['bias_' + ref] for s, g0 in both)
                                           for ref in ('prior', 'gt')}, 'rooms': len(both)}}


def _scopes(picked):
    bins, sources = defaultdict(list), defaultdict(list)
    for item in picked:
        bins[bin_name(item[0]['objects'])].append(item)
        sources[item[0]['source']].append(item)
    order = [bin_name(lo) for lo, _ in BINS]
    return {'all': _stats(picked), 'by_bin': {b: _stats(bins[b]) for b in order if bins[b]},
            'by_source': {s: _stats(v) for s, v in sorted(sources.items())}}


def reference_check(reference, arm, model, manifest, rooms, requests, samples):
    """Fidelity inputs, checked before any scoring: the HF run of the sampled model on the same rooms and prompts.

    samples: the vLLM greedy (k = 0) sample row per uid; HF and vLLM must see prompts of equal token counts."""
    reference = Path(reference)
    ref_data = Path(h.strict_json((reference / 'run_receipt.json').read_text())['config']['data'])
    ref_rows, info = infer.load_eval(ref_data, arm)  # Verifies every file of the reference data.
    records, receipt = e._inference(reference, ref_data, info['expected_uids'])
    if receipt['config']['arm'] != arm or any(receipt.get(k) != model.get(k) for k in MODEL_KEYS):
        raise ValueError('reference inference is not the sampled model')
    ref_manifest = h.strict_json((ref_data / 'MANIFEST.json').read_text())
    if ref_manifest.get('training_manifest_sha256') != manifest['training_manifest_sha256']:
        raise ValueError('reference data is bound to another training manifest')
    ref_rooms = h._by_uid(h._rows(ref_data, 'rooms.jsonl', ref_manifest['files']['rooms.jsonl']))
    if any(r['uid'] not in requests or r['messages'] != requests[r['uid']]['messages'] or
           ref_rooms[r['uid']] != rooms[r['uid']] for r in ref_rows):
        raise ValueError('reference rooms or requests differ from the sampled ones')
    drift = sum(r['input_tokens'] != samples[r['uid']]['input_tokens'] for r in records)
    if drift:
        raise ValueError(f'{drift} reference prompts differ in token count from the vLLM prompts (template drift)')
    return reference, records, receipt


def fidelity(checked, arm, rooms, requests, greedy, samples):
    """Amendment-011 decision 1: vLLM greedy (k = 0) vs the HF records of the same model on dev-select-v1."""
    reference, records, receipt = checked
    pairs = []  # (HF passes, vLLM greedy passes, either side hit its token cap)
    for record in records:
        row = e._score((rooms[record['uid']], requests[record['uid']], record, arm))
        hf = r2.cap_failure(row) if record['reached_token_cap'] and not record['stopped_on_eos'] else row
        pairs.append((hf['geometry_success'], greedy[record['uid']]['geometry_success'],
                      record['reached_token_cap'] or samples[record['uid']]['reached_token_cap']))
    b = [capped for hf, vllm, capped in pairs if hf and not vllm]
    c = [capped for hf, vllm, capped in pairs if vllm and not hf]
    n, match = len(records), sum(r['raw'] == samples[r['uid']]['raw'] for r in records)
    return {'reference': str(reference.resolve()), 'reference_run_sha256': receipt['run_sha256'],
            'reference_lora': receipt['config'].get('lora'), 'rooms': n, 'exact_match': match, 'exact_match_rate': match / n,
            'prompt_token_counts_equal': n, 'hf_success_rate': sum(hf for hf, _, _ in pairs) / n,
            'vllm_greedy_success_rate': sum(vllm for _, vllm, _ in pairs) / n,
            'both_pass': sum(hf and vllm for hf, vllm, _ in pairs),
            'b_hf_only_pass': len(b), 'b_either_capped': sum(b), 'c_vllm_only_pass': len(c), 'c_either_capped': sum(c),
            'note': ('vLLM applies LoRA in BF16; HF capped per batch (shared padded width), S2 per room, so the '
                     '*_either_capped rooms may differ by the cap rule, not the engine; S2 gains are vLLM selected minus '
                     'vLLM greedy, never vs HF numbers')}


def evaluate(data, samples, arm, out, band=BAND, ks=(1, 2, 4, 8), workers=1, reference=None):
    data, samples, out = Path(data), Path(samples), Path(out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    manifest = e._verified(data)
    expected = manifest['expected_uids']
    def rows(name):
        return h._by_uid(h._rows(data, name, manifest['files'][name]))
    rooms, requests, priors = rows('rooms.jsonl'), rows(f'{arm}/requests.jsonl'), rows('size-estimates.jsonl')
    sampled = e._verified(samples)  # Hashes every receipt and samples.jsonl.
    receipt = h.strict_json((samples / 'run_receipt.json').read_text())
    run_hash, n, model = receipt['run_sha256'], receipt['sampling']['n_samples'], receipt['model']
    if infer._hash({k: v for k, v in receipt.items() if k != 'run_sha256'}) != run_hash:
        raise ValueError('sampling run identity hash mismatch')
    if (h.strict_json((samples / 'result.json').read_text()).get('status') != 'SAMPLING_COMPLETE' or
            receipt['arm'] != arm or sampled.get('expected_uids') != expected or 'samples.jsonl' not in sampled['files']
            or receipt['data']['manifest_sha256'] != h.sha256(data / 'MANIFEST.json')):
        raise ValueError('samples are not a complete sampling run of this arm on this data')
    if model['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
        raise ValueError('sampled model was trained on another training manifest')
    ks = tuple(ks)
    if not ks or list(ks) != sorted(set(ks)) or ks[0] < 1 or ks[-1] > n:
        raise ValueError(f'ks must be increasing within 1..{n}')
    ms, grouped = (0, *ks), {}
    for line in (samples / 'samples.jsonl').read_text().splitlines():
        sample = h.strict_json(line)
        grouped.setdefault(sample['uid'], []).append(sample)
    if list(grouped) != expected or any([s['k'] for s in v] != list(range(n + 1)) or
                                        any(s['run_sha256'] != run_hash or s['arm'] != arm for s in v)
                                        for v in grouped.values()):
        raise ValueError(f'samples.jsonl must hold k = 0..{n} of this run for every expected room, in order')
    first = {u: grouped[u][0] for u in expected}
    checked = None if reference is None else reference_check(reference, arm, model, manifest, rooms, requests, first)
    scored = d._map(score_room, [(rooms[u], requests[u], priors[u]['estimated_sizes'], grouped[u], arm, band, ms)
                                 for u in expected], workers)
    by_m = {str(m): _scopes([(r['compact'][r['picks'][m][0]], r['compact'][0], r['picks'][m][1], r['guard_changed'][m])
                             for r in scored]) for m in ms}
    greedy = {u: r['greedy'] for u, r in zip(expected, scored)}
    summary = {'status': 'BEST_OF_N_COMPLETE', 'arm': arm, 'rule': RULE.format(lo=band[0], hi=band[1], tol=BIAS_TOL),
               'band': list(band), 'guards': {'gated_counts': list(GUARDED), 'bias_tol': BIAS_TOL,
                                              'applies_to': 'alternatives k >= 1 of either arm, never greedy (P sizes are '
                                                            'the prior: gate, flat and bias are identities there)'},
               'ms': list(ms), 'bins': [bin_name(lo) for lo, _ in BINS], 'by_m': by_m,
               'samples': {'path': str(samples.resolve()), 'manifest_sha256': h.sha256(samples / 'MANIFEST.json'),
                           'run_sha256': run_hash, 'sampling': receipt['sampling'], 'engine': receipt['engine']},
               'model': {k: model.get(k) for k in (*MODEL_KEYS, 'stage')},
               'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
               'fidelity': None if checked is None else fidelity(checked, arm, rooms, requests, greedy, first),
               'scope': ('J/P-internal raw checker results (no S1 repair) on dev; no LLM baseline has N samples yet, so '
                         'no J/P-vs-L claim; the backend 1 mm / 1 cm strict gate is not measured here'),
               'quality_claim': False,
               'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(g.__file__), Path(validate.__file__),
                                                             Path(r2.__file__), Path(e.__file__), Path(contract.__file__),
                                                             Path(scene.__file__))}}
    d._publish_dir(out, {f'{arm}-greedy-geometry.jsonl': d._jsonl(r['greedy'] for r in scored),
                         f'{arm}-bo{ks[-1]}-geometry.jsonl': d._jsonl(r['selected'] for r in scored),
                         f'{arm}-samples.jsonl': d._jsonl(c for r in scored for c in r['compact']),
                         'summary.json': (json.dumps(summary, indent=1, allow_nan=False) + '\n').encode()})
    return summary


def calibrate(data, geometry, bands):
    """Per object-count bin: share of complete J greedy rooms whose every object passes each candidate band."""
    data, geometry = Path(data), Path(geometry)
    manifest = e._verified(data)
    if geometry.name not in e._verified(geometry.parent)['files']:
        raise ValueError('geometry file is not in its directory manifest')
    priors = h._by_uid(h._rows(data, 'size-estimates.jsonl', manifest['files']['size-estimates.jsonl']))
    rows = [h.strict_json(x) for x in geometry.read_text().splitlines()]
    if sorted(r['uid'] for r in rows) != sorted(manifest['expected_uids']) or any(r['arm'] != 'joint' for r in rows):
        raise ValueError('geometry rows are not the joint rows of exactly this data')
    done = [r for r in rows if r['complete']]
    result = {}
    for band in bands:
        passed = defaultdict(list)
        for r in done:
            ok = not size_gate(r['raw_layout'], priors[r['uid']]['estimated_sizes'], band)
            passed['all'].append(ok)
            passed[bin_name(r['objects'])].append(ok)
        result[f'{band[0]},{band[1]}'] = {
            name: {'complete_rooms': len(v), 'gate_pass_rate': sum(v) / len(v) if v else None,
                   'too_strict': bool(v) and sum(v) / len(v) < TOO_STRICT}
            for name in ('all', *(bin_name(lo) for lo, _ in BINS)) for v in [passed[name]]}
    return {'protocol': 's2-size-gate-calibration-v1', 'geometry': str(geometry.resolve()),
            'geometry_sha256': h.sha256(geometry), 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
            'rooms': len(rows), 'complete_rooms': len(done), 'too_strict_below': TOO_STRICT, 'bands': result}


def main(argv=None):
    def floats(text):
        return tuple(float(x) for x in text.split(','))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    chooser, calibrator = sub.add_parser('select'), sub.add_parser('calibrate')
    for flag in ('data', 'samples', 'out'):
        chooser.add_argument('--' + flag, type=Path, required=True)
    chooser.add_argument('--arm', choices=h.ARMS, required=True)
    chooser.add_argument('--ks', type=lambda text: tuple(int(x) for x in text.split(',')), default=(1, 2, 4, 8))
    chooser.add_argument('--workers', type=int, default=1)
    chooser.add_argument('--reference', type=Path)
    for flag in ('data', 'geometry'):
        calibrator.add_argument('--' + flag, type=Path, required=True)
    calibrator.add_argument('--bands', type=floats, nargs='+', default=[BAND])
    args = parser.parse_args(argv)
    if args.command == 'calibrate':
        print(json.dumps(calibrate(args.data, args.geometry, args.bands), indent=1))
        return
    summary = evaluate(args.data, args.samples, args.arm, args.out, ks=args.ks, workers=args.workers,
                       reference=args.reference)
    print(json.dumps({'status': summary['status'], 'arm': args.arm, 'fidelity': summary['fidelity'],
                      **{f'm{m}': summary['by_m'][m]['all']['success_rate'] for m in summary['by_m']}}))


if __name__ == '__main__':
    main()
