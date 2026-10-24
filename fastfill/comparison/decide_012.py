"""Amendment 012 §4 dev decision: S1 and S2 adoption from the four S1 outputs; never touches test.

python -m fastfill.comparison.decide_012 --data RUN/dev-full-v1 --s1-root RUN/s1-012 --out RUN/decision-012 [--workers 32]

Inputs: RUN/s1-012/{joint-sft,placement-rpo-dpo1}-{greedy,bo8}/ (output_repair outputs, MANIFEST-verified, every
dev request). Primary metric (amendment 012 §3) = downstream acceptance = s1 'strict_ok' (downstream.accept, path A,
every door entry); also reported: 'accepted' (frozen checker success AND strict_ok) and checker success. Paired
differences use geometry.paired_intervals (house-component bootstrap, source-signature strata, 2,000 draws, seed 42)
with rates over every request (incomplete = 0). Rules, inclusive: S1 adopted iff for BOTH arms G+S1 - G strict_ok has
CI lower bound > 0, the median displacement of moved objects <= 0.30 m and checker success G+S1 - G has CI upper
bound >= 0. S2 adopted (per arm) iff (B8+S1) - (G+S1) strict_ok >= 0.03 with CI lower bound > 0 and, for J, |B(B8) -
B(G)| <= 0.03 (footprint log-bias vs GT on complete rooms). Only B8 was repaired, so the smaller-m fallback of §4 is
not evaluable and is reported as such. A GT reference row (downstream.accept and the checker on the GT layouts) is
descriptive only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import tempfile

from fastfill.comparison import downstream as dn
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

ARMS = {'joint': 'joint-sft', 'placement': 'placement-rpo-dpo1'}
VARIANTS = ('greedy', 'bo8')
S1_GAIN, MAX_MOVED_MEDIAN, S2_GAIN, BIAS_TOL = 0.0, 0.30, 0.03, 0.03


def _rows(root, arm, variant):
    d = Path(root) / f'{ARMS[arm]}-{variant}'
    manifest = e._verified(d)
    name = f'{arm}-{variant}-geometry-s1.jsonl'
    return h._rows(d, name, manifest['files'][name]), h.sha256(d / 'MANIFEST.json')


def _view(rows, stage):
    """Rows as paired_intervals expects: 'geometry_success' := downstream strict_ok, 'geometry_success_repaired' :=
    checker success, 'complete' as is; scalars unused (None)."""
    out = []
    for r in rows:
        s1 = r['s1'] if isinstance(r.get('s1'), dict) else {}
        out.append({'uid': r['uid'], 'source': r['source'], 'group': r['group'], 'complete': bool(r.get('complete')),
                    'geometry_success': bool(s1.get(f'strict_ok_{stage}', False)),
                    'geometry_success_repaired': bool(s1.get(f'geometry_success_{stage}', False)),
                    **{k: None for k in g.SCALARS}})
    return out


def _diff(a, b, label):
    iv = g.paired_intervals(a, b)
    pick = lambda f: {'estimate': iv[f]['estimate_joint_minus_placement'], 'ci95': iv[f]['ci95'],
                      'n': iv[f]['paired_rooms'], 'warnings': iv[f]['warnings']}
    return {'comparison': label, 'downstream': pick('geometry_success'), 'checker': pick('geometry_success_repaired'),
            'complete': pick('complete')}


def _bias(rows, rooms):
    values = [r2.size_bias(rooms[r['uid']]['objects'], r['raw_layout'])[0] for r in rows if r.get('complete')]
    return {'mean': statistics.mean(values) if values else None, 'complete_rooms': len(values)}


def _gt(task):
    room = task
    verdict = dn.accept(room, room['objects'])
    geometry = g.geometry({**room, 'constraints': []})
    return {'uid': room['uid'], 'strict_ok': bool(verdict.get('accepted')), 'checker': bool(geometry['geometry_success'])}


def decide(data, s1_root, out, workers=1):
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = e._verified(data)
    rooms = h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl'])
    by_uid = h._by_uid(rooms)
    inputs, rates, contrasts, displacement = {}, {}, [], {}
    views, joint_bias = {}, {}
    for arm in ARMS:
        for variant in VARIANTS:
            rows, sha = _rows(s1_root, arm, variant)
            if {r['uid'] for r in rows} != set(manifest['expected_uids']):
                raise ValueError(f'{arm}-{variant}: rows must cover every dev request')
            inputs[f'{arm}-{variant}'] = sha
            if arm == 'joint':
                joint_bias[variant] = _bias(rows, by_uid)
            for stage in ('raw', 's1'):
                v = _view(rows, stage)
                views[arm, variant, stage] = v
                rates[f'{arm}-{variant}-{stage}'] = {
                    'downstream': statistics.mean(r['geometry_success'] for r in v),
                    'checker': statistics.mean(r['geometry_success_repaired'] for r in v),
                    'complete': statistics.mean(r['complete'] for r in v)}
            moved = [x for r in rows for x in ((r.get('s1') or {}).get('displacement_m') or []) if x > 0]
            displacement[f'{arm}-{variant}'] = {'moved_objects': len(moved),
                                                'moved_median': statistics.median(moved) if moved else 0.0}
    decisions = {}
    s1_ok = True
    for arm in ARMS:
        c = _diff(views[arm, 'greedy', 's1'], views[arm, 'greedy', 'raw'], f'{arm}: G+S1 - G')
        contrasts.append(c)
        lo = c['downstream']['ci95'][0] if c['downstream']['ci95'] else None
        hi = c['checker']['ci95'][1] if c['checker']['ci95'] else None
        med = displacement[f'{arm}-greedy']['moved_median']
        passed = lo is not None and lo > S1_GAIN and med <= MAX_MOVED_MEDIAN + 1e-12 and hi is not None and hi >= 0
        decisions[f's1_{arm}'] = {'passed': passed, 'ci_low_downstream': lo, 'moved_median_m': med,
                                  'checker_ci_high': hi}
        s1_ok = s1_ok and passed
    decisions['s1_adopted'] = s1_ok
    for arm in ARMS:
        c = _diff(views[arm, 'bo8', 's1'], views[arm, 'greedy', 's1'], f'{arm}: (B8+S1) - (G+S1)')
        contrasts.append(c)
        est, ci = c['downstream']['estimate'], c['downstream']['ci95']
        passed = est is not None and est >= S2_GAIN - 1e-12 and ci is not None and ci[0] > 0
        bias_check = 'P sizes are the prior: not applicable'
        if arm == 'joint':
            greedy, bo8 = (joint_bias[v]['mean'] for v in VARIANTS)
            delta = abs(bo8 - greedy) if bo8 is not None and greedy is not None else None
            bias_ok = delta is not None and delta <= BIAS_TOL + 1e-12
            bias_check = {**joint_bias, 'absolute_difference': delta, 'tolerance': BIAS_TOL, 'passed': bias_ok,
                          'population': 'each variant\'s own complete rooms; room mean log(pred/GT footprint)'}
            passed = passed and bias_ok
        decisions[f's2_{arm}'] = {'passed': passed, 'estimate': est, 'ci95': ci,
                                  'bias_check': bias_check,
                                  'smaller_m_fallback': 'not evaluable: only B8 was repaired by S1'}
    tasks = [r for r in rooms]
    if workers == 1:
        gt = [_gt(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as pool:
            gt = list(pool.map(_gt, tasks, chunksize=8))
    reference = {'rooms': len(gt), 'downstream': statistics.mean(r['strict_ok'] for r in gt),
                 'checker': statistics.mean(r['checker'] for r in gt),
                 'note': 'GT layouts of the same dev rooms; descriptive only (not a decision input)'}
    value = {'status': 'DECISION_012_COMPLETE', 'protocol': 'amendment-012 §4', 'rates': rates,
             'contrasts': contrasts, 'displacement': displacement, 'decisions': decisions, 'gt_reference': reference,
             'inputs': {'data_MANIFEST': h.sha256(data / 'MANIFEST.json'), **inputs},
             'code_sha256': {p.name: h.sha256(p) for p in map(Path, (__file__, g.__file__, dn.__file__, r2.__file__))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'decision.json', value)
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('data', 's1-root', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    value = decide(args.data, args.s1_root, args.out, args.workers)
    print(json.dumps({'status': value['status'], 'decisions': value['decisions'], 'rates': value['rates'],
                      'gt_reference': value['gt_reference']}, default=str))


if __name__ == '__main__':
    main()
