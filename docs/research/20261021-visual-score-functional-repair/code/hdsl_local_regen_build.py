"""API probe builder (free, local CPU): HDSL-style LOCAL SUBTREE REGENERATION on post-S1 failures.

For a 27B+S1 room that still fails the frozen checker, the objects involved in violations are expanded to support
units (hdsl_chunk_hazard.units); those objects become the request's movable list (area-descending), every other
object is locked as an injected fixed record at its S1 pose (hdsl_chunk_requests.chunk_request with a one-chunk
plan). Contract prompt unchanged. N rooms are chosen by sha256 rank inside the strata; one API call each.

python3 -I hdsl_local_regen_build.py S1_JSONL ROOMS_JSONL REQUESTS_JSONL OUT_DIR [--n 100] [--max-regen 15]
Writes OUT_DIR/requests.jsonl: {uid, source, group, room_type, objects, regen, messages[system,user], maps, plan,
baseline:{checker_pass_s1, accepted_s1, violations_s1}} and selection.json.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from hdsl_chunk_hazard import involved, units, bin_of  # noqa: E402
from hdsl_chunk_requests import chunk_request  # noqa: E402
from fastfill.comparison import contract  # noqa: E402

STRATA = ('11-20', '21-50')      # 51+ regen sets are half the room (median 29): out of scope for 1 call


def main():
    s1p, roomsp, reqp, outd = (Path(a) for a in sys.argv[1:5])
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 100
    max_regen = int(sys.argv[sys.argv.index('--max-regen') + 1]) if '--max-regen' in sys.argv else 15
    outd.mkdir(parents=True, exist_ok=True)
    rooms = {r['uid']: r for r in map(json.loads, open(roomsp))}
    reqs = {r['uid']: r for r in map(json.loads, open(reqp))}
    cands = []
    for line in open(s1p):
        r = json.loads(line)
        if not r.get('complete') or not r.get('s1_layout') or bin_of(r['objects']) not in STRATA:
            continue
        s1 = r['s1']
        if s1.get('geometry_success_s1'):
            continue
        room = rooms[r['uid']]
        geo = {'boundary': room['boundary'], 'boundary_type': room['boundary_type'], 'height': room.get('height'),
               'fixed': room.get('fixed') or []}
        fail, ids = involved(geo, r['s1_layout'])
        if not fail:
            continue
        top = units(r['s1_layout'])
        unit_ids = {top[i] for i in ids}
        regen = [o['id'] for o in r['s1_layout'] if top[o['id']] in unit_ids]
        if len(regen) > max_regen:
            continue
        cands.append((hashlib.sha256(('hdsl-local-regen:' + r['uid']).encode()).hexdigest(), r, regen))
    cands.sort(key=lambda t: t[0])
    chosen = cands[:n]
    rows = []
    for _, r, regen in chosen:
        req = reqs[r['uid']]
        user = json.loads(req['messages'][1]['content'])
        by = {o['id']: o for o in user['objects']}
        lay = {o['id']: o for o in r['s1_layout']}
        order = sorted(regen, key=lambda i: (-by[i]['size'][0] * by[i]['size'][1], i))
        locked = [o['id'] for o in user['objects'] if o['id'] not in set(regen)]
        plan = [locked, order]                      # chunk 0 = locked (already placed), chunk 1 = regen set
        placed = {i: {'pos': [round(v, 2) for v in lay[i]['pos']],
                      'yaw': round(__import__('math').degrees(lay[i]['yaw']) % 360) % 360} for i in locked}
        cu, maps = chunk_request(user, plan, 1, placed)
        rows.append({'uid': r['uid'], 'source': r['source'], 'group': r['group'], 'room_type': r.get('room_type'),
                     'objects': r['objects'], 'regen': len(regen), 'stratum': bin_of(r['objects']),
                     'messages': [{'role': 'system', 'content': contract._SYSTEM['placement']},
                                  {'role': 'user', 'content': contract._json(cu)}],
                     'maps': maps, 'plan': plan,
                     'locked': {i: {**placed[i], 'on': lay[i].get('parent')} for i in locked},
                     'baseline': {'checker_pass_s1': False, 'accepted_s1': bool(r['s1'].get('accepted_s1')),
                                  'violations_s1': {k: r['s1']['geometry_s1'][k] for k in
                                                    ('main_collisions', 'severe_collisions', 'oob_objects',
                                                     'fixed_blocking', 'support_fail', 'ceiling_fail')}}})
    with open(outd / 'requests.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    sel = {'candidates': len(cands), 'chosen': len(rows), 'n': n, 'max_regen': max_regen, 'strata': STRATA,
           'regen_sizes': sorted(r['regen'] for r in rows),
           'by_stratum': {s: sum(r['stratum'] == s for r in rows) for s in STRATA},
           'order': "sha256('hdsl-local-regen:'+uid)", 'calls_per_room': 1}
    json.dump(sel, open(outd / 'selection.json', 'w'), indent=1)
    print(json.dumps(sel))


if __name__ == '__main__':
    main()
