"""Proxy for the placer's training targets: run the delivered pose-only teacher (pose_targets.repair, 200/4, prior sizes)
on GT rooms and measure how many GT companion pairs (companions.py rules) survive: GT poses + prior sizes (no repair)
vs teacher output. Also teacher success rate. python3 teacher_companions.py EVAL_DATA OUT [LIMIT]"""
import collections, copy, hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
sys.path.insert(0, str(Path(__file__).parent))
from fastfill.comparison.pose_targets import repair
import companions as C


def bin_(n): return '1-10' if n <= 10 else '11-20' if n <= 20 else '21-50' if n <= 50 else '51+'


def one(args):
    room, est = args
    gt = room['objects']
    ps = C.pairs(gt)
    prior = [{**o, 'size': list(est[o['id']])} for o in gt]
    res = repair(room, est)
    out = res['room']['objects'] if res['status'] == 'ok' else None
    held = lambda objs: sum(C.holds(p, {o['id']: o for o in objs}) for p in ps) if objs else 0
    by_role = collections.Counter(); kept_role = collections.Counter()
    if out:
        idx = {o['id']: o for o in out}
        for p in ps:
            r = C.role({o['id']: o for o in gt}[p[1]]); by_role[r] += 1; kept_role[r] += C.holds(p, idx)
    return {'uid': room['uid'], 'n': len(gt), 'status': res['status'], 'pairs': len(ps),
            'held_prior_nopose': held(prior), 'held_teacher': held(out),
            'moved': sum(1 for a, b in zip(gt, out) if a['pos'][:2] != b['pos'][:2]) if out else None,
            'yaw_changed': sum(1 for a, b in zip(gt, out) if round(a['yaw'], 4) != round(b['yaw'], 4)) if out else None,
            'by_role': dict(by_role), 'kept_role': dict(kept_role)}


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2]); limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out.mkdir(parents=True, exist_ok=False)
    est = {r['uid']: r['estimated_sizes'] for r in map(json.loads, open(data / 'size-estimates.jsonl'))}
    rooms = [r for r in map(json.loads, open(data / 'rooms.jsonl')) if r['uid'] in est]
    rooms.sort(key=lambda r: hashlib.sha256(('teacher-comp:' + r['uid']).encode()).hexdigest())
    rooms = rooms[:limit] if limit else rooms
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(one, [(r, est[r['uid']]) for r in rooms], chunksize=8))
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    lines = ['| 物体数 | 房间 | 修复成功 | 有搭配的房间 | GT 搭配对 | 只换先验尺寸仍保持 | 修复成功房里修复后仍保持 | 修复成功房里移动物体比例 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in ('1-10', '11-20', '21-50', '51+', 'all'):
        rs = [r for r in rows if b == 'all' or bin_(r['n']) == b]
        ok = [r for r in rs if r['status'] == 'ok']
        P = sum(r['pairs'] for r in rs); Pok = sum(r['pairs'] for r in ok)
        mv = sum(r['moved'] for r in ok) / max(1, sum(r['n'] for r in ok))
        lines.append(f"| {b} | {len(rs)} | {len(ok)/len(rs):.3f} | {sum(r['pairs']>0 for r in rs)} | {P} | "
                     f"{sum(r['held_prior_nopose'] for r in rs)/max(1,P):.3f} | {sum(r['held_teacher'] for r in ok)/max(1,Pok):.3f} | {mv:.3f} |")
    roles = collections.Counter(); kept = collections.Counter()
    for r in rows:
        roles.update(r['by_role']); kept.update(r['kept_role'])
    lines += ['', '按角色（修复成功房）: ' + ', '.join(f"{k} {kept[k]}/{v} = {kept[k]/v:.2f}" for k, v in roles.most_common())]
    text = '# 训练目标代理：位姿修复教师会拆掉多少 GT 搭配关系\n\n' + '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(text); print(text)


if __name__ == '__main__':
    main()
