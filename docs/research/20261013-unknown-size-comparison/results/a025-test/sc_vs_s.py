"""Amendment 025 section 3 / 024: per-arm judge means and the paired SC - S judge contrast (house-cluster bootstrap).

usage: python sc_vs_s.py AS_RECORDS SC_RECORDS OUT_JSON
Endpoint = (functionality + layout) / 2 per room; pairs = rooms with both S (024 judge) and SC judged; bootstrap resamples
house groups (2,000 draws, seed 42). Also reports per-arm mean overall and (f+l)/2 for A, S, SC.
"""
import json
import random
import statistics
import sys


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def fl(r):
    s = r['scores']
    return (s['functionality_score'] + s['layout_score']) / 2


def main(as_path, sc_path, out):
    rows = [r for r in load(as_path) + load(sc_path) if r.get('status') == 'completed']
    by = {}
    for r in rows:
        by.setdefault(r['arm'], {})[r['uid']] = r
    arms = {a: {'n': len(v), 'overall': statistics.fmean(x['scores'].get('overall_score', 0) for x in v.values()),
                'fl': statistics.fmean(fl(x) for x in v.values())} for a, v in by.items()}
    pairs = [(u, fl(by['SC'][u]) - fl(by['S'][u]), by['S'][u].get('group')) for u in by.get('SC', {}) if u in by.get('S', {})]
    groups = {}
    for u, d, g in pairs:
        groups.setdefault(g or u, []).append(d)
    keys = sorted(groups)
    est = statistics.fmean(d for _, d, _ in pairs)
    rng, draws = random.Random(42), []
    for _ in range(2000):
        sample = [d for k in (rng.choice(keys) for _ in keys) for d in groups[k]]
        draws.append(statistics.fmean(sample))
    draws.sort()
    value = {'arms': arms, 'pairs': len(pairs), 'groups': len(keys), 'sc_minus_s_fl': est,
             'ci95': [draws[49], draws[1949]], 'selected': {'S': 174, 'SC': 174}}
    json.dump(value, open(out, 'w'), indent=1)
    print(json.dumps(value))


if __name__ == '__main__':
    main(*sys.argv[1:4])
