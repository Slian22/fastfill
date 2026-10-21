"""Table S1 / S2 / S2-F for one split (all rooms of the S1 file), by object-count bins.

python3 s2f_eval.py dev|cohort [DIR under s2f/, default the split]
  S2  = the existing merged S2 (s2-dev or s2-render: S1 -> s2-protect -> merge -> s2c-strict -> merge, recomputed)
  S2F = s2f/{split}/final-s2f.jsonl
checker / strict re-scored with s2_protect._eval (asserted equal to the merged flags); GT relations as s2_protect
(a021_select.gt_relations + validate.holds); companions: retention from raw and from S1 (pairs read from that
layout), GT-based score (GT pairs, evaluation only); moved / displacement relative to S1.
"""
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import companions as C  # noqa: E402
import s2_protect as S  # noqa: E402
import s2f as F  # noqa: E402

P = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021')
H = Path('/Volumes/harddisk/fastfillout/unknownsize-20261013')
SPLITS = {'dev': (P / 's2-dev/s1/placement-geometry-s1.jsonl', H / 'a025-inputs/dev-select-v1', P / 's2-dev/s2-protect', P / 's2-dev/s2c-strict'),
          'cohort': (P / 's2-render/cohort-s1.jsonl', H / 'visual-final-selected-002/inputs/eval-data-v1',
                     P / 's2-render/s2-protect', P / 's2-render/s2c-strict')}
ARMS = ('S1', 'S2', 'S2F')
BINS = (('1-10', 1, 10), ('11-20', 11, 20), ('21+', 21, 10 ** 9))


def L(f):
    return [json.loads(l) for l in open(f)]


def score(task):
    uid, room_gt, request, raw, layouts = task
    rel = S.gt_relations(room_gt)
    out = {'uid': uid, 'relations': len(rel)}
    s1 = layouts['S1']
    for arm, objs in layouts.items():
        if objs is None:
            out[arm] = {'checker': False, 'strict': False, 'held': 0, 'ret_raw': C.retention(raw, None),
                        'ret_s1': C.retention(s1, None), 'gt_comp': C.gt_score(room_gt['objects'], None),
                        'moved': 0, 'disp': [], 'invariant_ok': True}
            continue
        room = S.orp.predicted_room(room_gt, request['messages'], objs)
        e = S._eval(room, objs)
        disp = [math.dist(a['pos'][:2], b['pos'][:2]) for a, b in zip(s1, objs)]
        out[arm] = {'checker': e['checker'], 'strict': e['strict'],
                    'held': sum(S.holds(c, None, room_gt['boundary'], {o['id']: o for o in objs}) for c in rel),
                    'ret_raw': C.retention(raw, objs), 'ret_s1': C.retention(s1, objs),
                    'gt_comp': C.gt_score(room_gt['objects'], objs),
                    'moved': sum(d > 1e-9 for d in disp), 'disp': [d for d in disp if d > 1e-9],
                    'invariant_ok': S._invariant(s1, objs)}
    return out


def _frac(rs, arm, k):
    a, b = sum(r[arm][k][0] for r in rs), sum(r[arm][k][1] for r in rs)
    return a / b if b else None


def agg(rs, objects, runtime):
    out = {'rooms': len(rs)}
    n_obj = sum(objects[r['uid']] for r in rs)
    for arm in ARMS:
        disp = [d for r in rs for d in r[arm]['disp']]
        out[arm] = {'checker': sum(r[arm]['checker'] for r in rs) / len(rs), 'strict': sum(r[arm]['strict'] for r in rs) / len(rs),
                    'gt_relations_held': sum(r[arm]['held'] for r in rs) / max(sum(r['relations'] for r in rs), 1),
                    'comp_retention_from_raw': _frac(rs, arm, 'ret_raw'), 'comp_retention_from_s1': _frac(rs, arm, 'ret_s1'),
                    'comp_gt_score': _frac(rs, arm, 'gt_comp'),
                    'pairs_raw': sum(r[arm]['ret_raw'][1] for r in rs), 'pairs_s1': sum(r[arm]['ret_s1'][1] for r in rs),
                    'pairs_gt': sum(r[arm]['gt_comp'][1] for r in rs),
                    'rooms_changed': sum(r[arm]['moved'] > 0 for r in rs),
                    'moved_share': sum(r[arm]['moved'] for r in rs) / max(n_obj, 1),
                    'moved_median_m': statistics.median(disp) if disp else None,
                    'invariant_violations': sum(not r[arm]['invariant_ok'] for r in rs)}
        if arm in runtime:
            rt = [runtime[arm][r['uid']] for r in rs if r['uid'] in runtime[arm]]
            out[arm]['runtime_rooms'] = len(rt)
            out[arm]['runtime_mean_s'] = statistics.mean(rt) if rt else None
            out[arm]['runtime_max_s'] = max(rt) if rt else None
    for arm in ('S2', 'S2F'):
        for k in ('checker', 'strict'):
            out[arm][f'{k}_gain'] = out[arm][k] - out['S1'][k]
    for k in ('checker', 'strict'):
        g2 = out['S2'][f'{k}_gain']
        out['S2F'][f'{k}_gain_kept_of_S2'] = out['S2F'][f'{k}_gain'] / g2 if g2 else None
    return out


def table(summary):
    f = lambda v, p=3: '—' if v is None else f'{v:.{p}f}'
    cols = [('checker', 'checker'), ('strict', 'strict'), ('gt_relations_held', 'GT rel. held'),
            ('comp_retention_from_raw', 'comp. ret. (raw pairs)'), ('comp_retention_from_s1', 'comp. ret. (S1 pairs)'),
            ('comp_gt_score', 'comp. GT score'), ('moved_share', 'objects moved'), ('moved_median_m', 'median disp. moved (m)'),
            ('rooms_changed', 'rooms changed'), ('runtime_mean_s', 'runtime mean s/room'), ('invariant_violations', 'invariant viol.')]
    lines = ['| bin | rooms | arm | ' + ' | '.join(c[1] for c in cols) + ' |', '|---|---:|---|' + '---:|' * len(cols)]
    for name, a in [('all', summary['all'])] + list(summary['bins'].items()):
        for arm in ARMS:
            cells = []
            for k, _ in cols:
                v = a[arm].get(k)
                cells.append(str(v) if isinstance(v, int) else f(v))
            lines.append(f"| {name} | {a['rooms']} | {arm} | " + ' | '.join(cells) + ' |')
    a = summary['all']['S2F']
    lines += ['', f"S2F keeps {f(a['checker_gain_kept_of_S2'])} of S2's checker gain over S1 and "
                  f"{f(a['strict_gain_kept_of_S2'])} of its strict gain (all rooms)."]
    return '\n'.join(lines) + '\n'


def main():
    split = sys.argv[1]
    s1_f, data, s2c, s2s = SPLITS[split]
    out = P / 's2f' / (sys.argv[2] if len(sys.argv) > 2 else split)      # optional: a dev iteration dir
    S._imports(F.REPO)
    rooms = {json.loads(l)['uid']: json.loads(l) for l in open(data / 'rooms.jsonl')}
    requests = {json.loads(l)['uid']: json.loads(l) for l in open(data / 'placement' / 'requests.jsonl')}
    s1 = L(s1_f)
    st = {'S2': [L(s2c / 'rows.jsonl'), L(s2s / 'rows.jsonl')],
          'S2F': [L(out / 's2f-checker/rows.jsonl'), L(out / 's2f-strict/rows.jsonl')]}
    s2 = F.merge(F.merge(s1, st['S2'][0]), st['S2'][1])
    s2f = L(out / 'final-s2f.jsonl')
    assert [r['uid'] for r in s2f] == [r['uid'] for r in s1]
    (out / 'merged-s2.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in s2))
    runtime = {arm: {} for arm in st}
    for arm, stages in st.items():
        for rows in stages:
            for r in rows:
                runtime[arm][r['uid']] = runtime[arm].get(r['uid'], 0.0) + r['runtime_s']
    stage_inv = {arm: sum(not r['invariant_ok'] for rows in stages for r in rows) for arm, stages in st.items()}
    tasks = [(a['uid'], rooms[a['uid']], requests[a['uid']], a['raw_layout'],
              {'S1': a['s1_layout'], 'S2': b['s1_layout'], 'S2F': c['s1_layout']}) for a, b, c in zip(s1, s2, s2f)]
    with ProcessPoolExecutor(8, initializer=S._imports, initargs=(F.REPO,)) as pool:
        res = list(pool.map(score, tasks, chunksize=4))
    for a, b, c, r in zip(s1, s2, s2f, res):        # re-scored flags agree with the merged flags
        for row, arm in ((a, 'S1'), (b, 'S2'), (c, 'S2F')):
            if row['complete']:
                assert r[arm]['checker'] == row['s1']['geometry_success_s1'] and r[arm]['strict'] == row['s1']['strict_ok_s1'], (arm, r['uid'])
    objects = {r['uid']: r['objects'] for r in s1}
    counts = {f'stage{i + 1}': {k: sum(r[k] for r in st['S2F'][i])
                                for k in ('pairs', 'group_moves', 'group_failed', 'keep_moves', 'fallbacks', 'rejoined', 'left_behind')}
              for i in (0, 1)}
    summary = {'split': split, 'rooms': len(res), 'stage_invariant_violations': stage_inv, 's2f_counts': counts,
               'all': agg(res, objects, runtime),
               'bins': {name: agg([r for r in res if lo <= objects[r['uid']] <= hi], objects, runtime) for name, lo, hi in BINS}}
    (out / 'eval-rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in res))
    (out / 'eval-summary.json').write_text(json.dumps(summary, indent=1))
    t = table(summary)
    (out / 'eval-table.md').write_text(t + '\n' + json.dumps({'stage_invariant_violations': stage_inv, 's2f_counts': counts}) + '\n')
    print(t, stage_inv, counts)


if __name__ == '__main__':
    main()
