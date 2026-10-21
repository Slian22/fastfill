"""Task-scoped reachability repair after S2-F: labbuilder_nav_repair.nav_repair with downstream.accept's required set
limited to the room's TASK objects (task_reach_headroom.task_cats, upstream 'verb obj1, obj2' emulation).
Runs on S2-F final rows that fail task-scoped admission with an A3 code. Reports task-scoped admission before/after,
checker, companion retention (S2-F layout -> repaired). python3 task_nav.py EVAL_DATA S2F_FINAL OUT"""
import collections, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
S = '/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021/scripts'
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene'); sys.path.insert(0, S)
from fastfill.comparison import downstream as d, geometry as g, output_repair as r
import labbuilder_nav_repair as L, companions as C, task_reach_headroom as T

_orig = d.accept
CATS = set()
def _task_accept(room, layout):
    return _orig({**room, 'objects': [o for o in room['objects'] if o['category'] in CATS]}, layout)


def work(task):
    global CATS
    room_gt, request, row = task
    CATS = set(T.task_cats(room_gt))
    pred = r.predicted_room(room_gt, request['messages'], row['s1_layout'])
    before = _task_accept(pred, pred['objects'])
    if before['accepted'] or not any(x.startswith('A3') for x in before['reasons']):
        return None
    L.d.accept = _task_accept
    try:
        out = L.nav_repair(pred)
    finally:
        L.d.accept = _orig
    lay = out['layout']
    kept, tot = C.retention(pred['objects'], lay)
    return {'uid': row['uid'], 'n': len(lay), 'cats': sorted(CATS), 'task_ok_before': False,
            'task_ok_after': _task_accept(pred, lay)['accepted'],
            'checker_before': g.geometry(pred)['geometry_success'],
            'checker_after': g.geometry({**pred, 'objects': lay})['geometry_success'],
            'companions_kept': kept, 'companions': tot, 'moved': out['moved_objects'], 'max_disp_m': out['max_disp_m'],
            'reasons_after': _task_accept(pred, lay)['reasons'], 'runtime_s': out['runtime_s'], 'layout': lay}


def main():
    data, final, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=False)
    rooms = {r_['uid']: r_ for r_ in map(json.loads, open(data / 'rooms.jsonl'))}
    reqs = {r_['uid']: r_ for r_ in map(json.loads, open(data / 'placement' / 'requests.jsonl'))}
    rows = [json.loads(l) for l in open(final)]
    tasks = [(rooms[x['uid']], reqs[x['uid']], x) for x in rows if x.get('complete') and x['uid'] in rooms]
    with ProcessPoolExecutor(8) as ex:
        res = [x for x in ex.map(work, tasks, chunksize=4) if x]
    (out / 'rows.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in res))
    n = len(rows)
    fixed = sum(x['task_ok_after'] for x in res)
    broke = sum(x['checker_before'] and not x['checker_after'] for x in res)
    ck = sum(x['companions_kept'] for x in res); ct = sum(x['companions'] for x in res)
    text = (f"# 任务物体可达修复（S2-F 之后）\n\n- 可视化房 {n} 间；任务口径下失败且含 A3 的 {len(res)} 间\n"
            f"- 修好（任务口径准入通过）{fixed} 间 = 全部房间 +{fixed / n:.3f}\n- checker 从通过变失败 {broke} 间\n"
            f"- 被修房间的搭配关系保持 {ck}/{ct}\n- 修后仍失败的原因: {collections.Counter(c for x in res if not x['task_ok_after'] for c in x['reasons_after']).most_common(6)}\n")
    (out / 'report.md').write_text(text); print(text)


if __name__ == '__main__':
    main()
