"""Headroom for task-aware placement: downstream admission when only the TASK objects must be reachable (upstream's
real contract: required = task objects), vs all categories (current strict) and none. Task objects emulate upstream's
'verb obj1, obj2' specs: 2 floor-standing movable categories per room, picked by hash (no GT peeking beyond the room).
python3 task_reach_headroom.py EVAL_DATA S1_ROWS OUT [S2F_FINAL_JSONL]"""
import collections, hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from fastfill.comparison import downstream as d


def bin_(n): return '1-10' if n <= 10 else '11-20' if n <= 20 else '21+'


def task_cats(room):
    roots = sorted({o['category'] for o in room['objects'] if not o.get('parent') and o['size'][2] >= 0.3})
    roots.sort(key=lambda c: hashlib.sha256(('task:' + room['uid'] + c).encode()).hexdigest())
    return roots[:2]


def scores(room, layout, cats):
    if not layout:
        return {'all': False, 'task': False, 'none': False}
    full = d.accept(room, layout)
    t = d.accept({**room, 'objects': [o for o in room['objects'] if o['category'] in cats]}, layout)
    return {'all': full['accepted'], 'task': t['accepted'], 'none': full['endpoint_ok'] and full['attach_ok'] and full['admit_no_required_ok']}


def one(args):
    room, row, s2f = args
    cats = task_cats(room)
    out = {'uid': room['uid'], 'n': len(room['objects']), 'cats': cats, 'GT': scores(room, room['objects'], cats)}
    out['raw'] = scores(room, row['raw_layout'] if row['complete'] else None, cats)
    out['S1'] = scores(room, row['s1_layout'] if row['complete'] else None, cats)
    if s2f is not None:
        out['S2F'] = scores(room, s2f, cats)
    return out


def main():
    data, s1, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    s2f = {}
    if len(sys.argv) > 4:
        for l in open(sys.argv[4]):
            r = json.loads(l)
            s2f[r['uid']] = r['s1_layout'] if r.get('complete') else None
    out.mkdir(parents=True, exist_ok=False)
    rooms = {r['uid']: r for r in map(json.loads, open(data / 'rooms.jsonl'))}
    rows = [json.loads(l) for l in open(s1)]
    tasks = [(rooms[r['uid']], r, s2f.get(r['uid']) if s2f else None) for r in rows if r['uid'] in rooms and (not s2f or r['uid'] in s2f)]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(one, tasks, chunksize=16))
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in res))
    arms = ['GT', 'raw', 'S1'] + (['S2F'] if s2f else [])
    lines = ['| 物体数 | 房间 | ' + ' | '.join(f'{a} 全类别 / 任务物体 / 不要求' for a in arms) + ' |', '|---|---:|' + '---:|' * len(arms)]
    for b in ('1-10', '11-20', '21+', 'all'):
        rs = [r for r in res if b == 'all' or bin_(r['n']) == b]
        f = lambda a, k: sum(r[a][k] for r in rs) / len(rs)
        lines.append(f'| {b} | {len(rs)} | ' + ' | '.join(f"{f(a,'all'):.3f} / {f(a,'task'):.3f} / {f(a,'none'):.3f}" for a in arms) + ' |')
    text = '# 任务物体可达的提升空间（下游准入：全类别 / 只要求任务物体 / 不要求）\n\n' + '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(text); print(text)


if __name__ == '__main__':
    main()
