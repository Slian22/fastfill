"""Amendment 023 §3: two judges on the same judge bundle, side by side.

  python -m fastfill.comparison.judge_agreement_023 --a RUN002/records.jsonl --b RUN023/records.jsonl \
      --names gpt-6.1-sol,gpt-6-astra-high --out DIR

Per-arm means (each judge's own completed cases and the common ones), per-metric Spearman/Pearson on the common
cases, mean per-room Kendall tau of the arms' overall scores (rooms with every arm complete under both judges),
the arm ranking by mean overall, and L's margin over the other arms' mean. Descriptive only.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

from scipy.stats import kendalltau, pearsonr, spearmanr

METRICS = ('functionality', 'layout', 'aesthetics', 'overall')


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {r['id']: r for r in rows if r.get('status') == 'completed'}, {r['id']: r for r in rows}


def score(row, metric):
    return row['scores'].get(metric + '_score')


def means(done, arms, ids=None):
    out = {}
    for arm in arms:
        rows = [r for cid, r in done.items() if r['arm'] == arm and (ids is None or cid in ids)]
        out[arm] = {m: statistics.fmean(v) if (v := [score(r, m) for r in rows if score(r, m) is not None]) else None
                    for m in METRICS} | {'n': len(rows)}
    return out


def compare(a_path, b_path, names):
    (a, a_all), (b, b_all) = load(a_path), load(b_path)
    if set(a_all) != set(b_all):
        raise ValueError('the two runs judged different case sets')
    arms = sorted({r['arm'] for r in a_all.values()})
    common = set(a) & set(b)
    corr = {}
    for m in METRICS:
        pairs = [(score(a[c], m), score(b[c], m)) for c in sorted(common)
                 if score(a[c], m) is not None and score(b[c], m) is not None]
        x, y = zip(*pairs)
        corr[m] = {'n': len(pairs), 'spearman': spearmanr(x, y).statistic, 'pearson': pearsonr(x, y).statistic}
    rooms = {}
    for cid in common:
        rooms.setdefault(a[cid]['uid'], {})[a[cid]['arm']] = (score(a[cid], 'overall'), score(b[cid], 'overall'))
    taus = [kendalltau(*zip(*[v[arm] for arm in arms])).statistic for v in rooms.values()
            if set(v) == set(arms) and all(None not in v[arm] for arm in arms)]
    taus = [t for t in taus if t == t]  # A room where one judge gives every arm the same score has no defined tau.
    out = {'judges': names, 'cases': len(a_all), 'completed': {names[0]: len(a), names[1]: len(b)},
           'common_completed': len(common), 'correlation': corr,
           'room_kendall_tau': {'rooms': len(taus), 'mean': statistics.fmean(taus) if taus else None}}
    for name, done in zip(names, (a, b)):
        own, shared = means(done, arms), means(done, arms, common)
        ranking = sorted(arms, key=lambda arm: -shared[arm]['overall'])
        others = [shared[arm]['overall'] for arm in arms if arm != 'L']
        out[name] = {'means_own': own, 'means_common': shared, 'ranking_common': ranking,
                     'L_margin_common': shared['L']['overall'] - statistics.fmean(others) if 'L' in arms else None}
    out['same_ranking'] = out[names[0]]['ranking_common'] == out[names[1]]['ranking_common']
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--names', required=True, type=lambda v: v.split(','))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare(args.a, args.b, args.names)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'agreement.json').write_text(json.dumps(result, indent=1) + '\n')
    with (args.out / 'means.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['judge', 'cases', 'arm', 'n', *METRICS])
        for name in args.names:
            for kind in ('own', 'common'):
                for arm, row in result[name]['means_' + kind].items():
                    w.writerow([name, kind, arm, row['n'], *[row[m] for m in METRICS]])
    print(json.dumps({k: result[k] for k in ('completed', 'common_completed', 'same_ranking', 'room_kendall_tau')}))


if __name__ == '__main__':
    main()
