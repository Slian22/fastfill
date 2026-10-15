"""Amendment 026 addendum C: paired 27B - O2nt (untuned same base) differences per policy, house-cluster bootstrap.

python ablation_paired.py RUN_ROOT DATA PILOT COHORT OUT_JSON
Reuses frontier_026's own row loading (s1_rows), views (cap rule on the local arm, as in the panels) and simulate().
"""
import json
import random
import statistics
import sys
from pathlib import Path

from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import frontier_026 as f
from fastfill.comparison import train_pair_helpers as h

METRICS = ('checker', 'strict', 'noreq')


def main(run, data, pilot, cohort, out):
    run, data = Path(run), Path(data)
    manifest, data_sha = e._verified(data), h.sha256(data / 'MANIFEST.json')
    scopes = {'A': f.load_uids(pilot, data_sha), 'B': f.load_uids(cohort, data_sha)}
    want = set().union(*scopes.values())
    rooms = {u: x for u, x in h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl'])).items()
             if u in want}
    locals_ = {'27B': f.s1_rows(run / 's1-022' / 'placement-arm', 'placement-geometry.jsonl', data_sha, want)[0],
               'O2nt': f.s1_rows(run / 's1-026' / 'agent-O2nt-P-raw', 'llm_raw-geometry.jsonl', data_sha, want)[0]}
    bigs = {'A': f.s1_rows(run / 's1-016-size-fixed' / 'agent-L-P-pilot-harness', 'llm_harness-geometry.jsonl',
                           data_sha, scopes['A'])[0]}
    for v in ('raw', 'harness'):
        bigs[f'B-{v}'] = f.s1_rows(run / 's1-016-size-fixed' / f'agent-O2-P-{v}', f'llm_{v}-geometry.jsonl', data_sha,
                                   scopes['B'])[0]
    result = {}
    for panel, big_rows in bigs.items():
        uids = sorted(scopes[panel[0]])
        bb = h._by_uid(big_rows)
        big = [f.view(bb[u], rooms[u]) for u in uids]
        sims = {}
        for name, rows in locals_.items():
            lb = h._by_uid(rows)
            local = [f.view(lb[u], rooms[u], cap_rule=True) for u in uids]
            sims[name] = {p: f.simulate(p, local, big) for p in ('LOCAL', 'ESC-CHECK', 'ESC-STRICT')}
        groups = {}
        for i, u in enumerate(uids):
            groups.setdefault(rooms[u].get('group') or u, []).append(i)
        keys = sorted(groups)
        for p in ('LOCAL', 'ESC-CHECK', 'ESC-STRICT'):
            entry = {}
            for m in METRICS + ('escalated',):
                if m == 'escalated':
                    d = [float(a[1]) - float(b[1]) for a, b in zip(sims['27B'][p], sims['O2nt'][p])]
                else:
                    d = [float(bool(a[0][m])) - float(bool(b[0][m])) for a, b in zip(sims['27B'][p], sims['O2nt'][p])]
                rng, draws = random.Random(42), []
                for _ in range(2000):
                    idx = [i for k in (rng.choice(keys) for _ in keys) for i in groups[k]]
                    draws.append(statistics.fmean(d[i] for i in idx))
                draws.sort()
                entry[m] = {'diff_27B_minus_O2nt': statistics.fmean(d), 'ci95': [draws[49], draws[1949]]}
            result[f'{panel}/{p}'] = entry
    Path(out).write_text(json.dumps(result, indent=1) + '\n')
    for k, v in result.items():
        print(k, {m: f"{x['diff_27B_minus_O2nt']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" for m, x in v.items()})


if __name__ == '__main__':
    main(*sys.argv[1:6])
