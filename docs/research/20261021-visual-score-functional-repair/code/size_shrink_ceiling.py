"""Ceiling probe: at the GT arrangement (poses/yaw/parents frozen), how much of the prior-size infeasibility is removed by
shrinking only the violating objects, never below FLOOR x prior per axis?  Sizes shrink about the object centre (xy for
collisions / OOB / fixed blocking / support, z for the ceiling); children are re-seated on the parent's new top.
Also reports size log-MAE vs GT before and after (does feasibility-driven shrinking move sizes toward the truth?).

python3 size_shrink_ceiling.py EVAL_DATA_DIR OUT_DIR FLOOR [FLOOR ...]
"""
import collections, copy, json, math, statistics, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from fastfill.comparison import geometry as g, output_repair as orp

STEP, ITERS = 0.9, 12


def bin_(n): return '1-10' if n <= 10 else '11-20' if n <= 20 else '21-50' if n <= 50 else '51+'


def reseat(room):
    by = {o['id']: o for o in room['objects'] + (room.get('fixed') or [])}
    mine = {o['id'] for o in room['objects']}
    done = set()
    def top(i):
        o = by[i]
        if i in done:
            return
        done.add(i)
        if i in mine and o.get('parent') in by:
            top(o['parent'])
            p = by[o['parent']]
            o['pos'] = [o['pos'][0], o['pos'][1], p['pos'][2] + p['size'][2]]
    for o in room['objects']:
        top(o['id'])


def log_mae(objs, gt):
    return statistics.mean(abs(math.log(o['size'][k] / gt[o['id']][k])) for o in objs for k in range(3))


def one(args):
    room, est, floors = args
    gt = {o['id']: o['size'] for o in room['objects']}
    base = copy.deepcopy(room)
    for o in base['objects']:
        o['size'] = list(est[o['id']])
    reseat(base)
    out = {'uid': room['uid'], 'n': len(room['objects']), 'gt_pass': g.geometry(room)['geometry_success'],
           'prior_pass': g.geometry(base)['geometry_success'], 'prior_logmae': log_mae(base['objects'], gt)}
    for floor in floors:
        r = copy.deepcopy(base)
        lo = {o['id']: [floor * v for v in est[o['id']]] for o in r['objects']}
        by = {o['id']: o for o in r['objects']}
        ok = out['prior_pass']
        for _ in range(ITERS if not ok else 0):
            L = orp.audit(r, [], [])['lists']
            xy = set(L['oob']) | set(L['support_fail']) | {o for o, _ in L['fixed_blocking']} \
                | {i for pair in L['main'] for i in pair}
            changed = False
            for i in xy & set(by):
                s = by[i]['size']
                new = [max(s[0] * STEP, lo[i][0]), max(s[1] * STEP, lo[i][1]), s[2]]
                changed |= new != s
                by[i]['size'] = new
            for i in set(L['ceiling']) & set(by):
                s = by[i]['size']
                z = max(s[2] * STEP, lo[i][2])
                changed |= z != s[2]
                by[i]['size'] = [s[0], s[1], z]
            reseat(r)
            ok = g.geometry(r)['geometry_success']
            if ok or not changed:
                break
        shrunk = [o for o in r['objects'] if o['size'] != list(est[o['id']])]
        out[f'f{floor}'] = {'pass': ok, 'logmae': log_mae(r['objects'], gt), 'shrunk': len(shrunk),
                            'shrunk_logmae_prior': log_mae([{**o, 'size': est[o['id']]} for o in shrunk], gt) if shrunk else None,
                            'shrunk_logmae_after': log_mae(shrunk, gt) if shrunk else None}
    return out


def main():
    data, out, floors = Path(sys.argv[1]), Path(sys.argv[2]), [float(x) for x in sys.argv[3:]]
    out.mkdir(parents=True, exist_ok=False)
    est = {r['uid']: r['estimated_sizes'] for r in map(json.loads, open(data / 'size-estimates.jsonl'))}
    rooms = [r for r in map(json.loads, open(data / 'rooms.jsonl')) if r['uid'] in est]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(one, [(r, est[r['uid']], floors) for r in rooms], chunksize=16))
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    head = '| 物体数 | 房间 | GT 尺寸 | 先验尺寸 | ' + ' | '.join(f'缩到≥{f}×先验' for f in floors) + \
           ' | 尺寸 logMAE 先验 → ' + ' / '.join(f'{f}' for f in floors) + ' | 被缩物体 logMAE 先验 → 缩后（' + f'{floors[-1]}' + '） |'
    lines = [head, '|---|---:|' + '---:|' * (3 + len(floors))]
    for b in ('1-10', '11-20', '21-50', '51+', 'all'):
        rs = [r for r in rows if b == 'all' or bin_(r['n']) == b]
        rate = lambda k: sum(r[k] for r in rs) / len(rs)
        fr = lambda f: sum(r[f'f{f}']['pass'] for r in rs) / len(rs)
        mae = lambda f: statistics.mean(r[f'f{f}']['logmae'] for r in rs)
        sh = [r[f'f{floors[-1]}'] for r in rs if r[f'f{floors[-1]}']['shrunk']]
        shp = statistics.mean(x['shrunk_logmae_prior'] for x in sh) if sh else float('nan')
        sha = statistics.mean(x['shrunk_logmae_after'] for x in sh) if sh else float('nan')
        lines.append(f"| {b} | {len(rs)} | {rate('gt_pass'):.3f} | {rate('prior_pass'):.3f} | "
                     + ' | '.join(f'{fr(f):.3f}' for f in floors)
                     + f" | {statistics.mean(r['prior_logmae'] for r in rs):.3f} → " + ' / '.join(f'{mae(f):.3f}' for f in floors)
                     + f" | {shp:.3f} → {sha:.3f} |")
    text = '# 先验尺寸下 GT 摆法：只缩违规物体的上限（位姿不动）\n\n' + '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
