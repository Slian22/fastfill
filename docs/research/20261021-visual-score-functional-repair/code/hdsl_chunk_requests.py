"""HDSL-lite chunked placement on the UNCHANGED contract: 'fixed-injection'.

A room with n movable objects is placed in ceil(n/C) calls. Call k lists only chunk k's objects (renumbered
obj_0001..) and injects every object placed in earlier calls as a fixed record (fixed_NNNN continuing the room's
numbering, category/size/pos/yaw) - the schema the placer was trained on. Chunk order = prior footprint area
descending (supports first; orderstat: parent after child 2.9%). 'on' pointing at an injected fixed id is mapped
back to the original movable id at merge; the merged output is request-order {id,pos,yaw,on} and must pass
contract.validate_output. System prompt, user schema and output schema are byte-identical to the delivered arm.

Library: chunk_plan(user, C) / chunk_request(user, plan, k, placed) / merge(user, plan, answers).
Self-check (free, local): python3 -I hdsl_chunk_requests.py --check REQUESTS ROOMS OUT_JSON [--C 10]
  uses the GT labels (contract.messages) as the oracle answer of every chunk, verifies the round trip under
  validate_output, counts support links a chunk cannot express (parent in a later chunk) and the char growth.
"""
import copy
import json
import math
import sys

REPO = '/Users/slian/Desktop/3D/Worldedge/OptiScene'
sys.path.insert(0, REPO)
from fastfill.comparison import contract  # noqa: E402


def _j(v):
    return json.dumps(v, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def chunk_plan(user, C):
    """[[original obj ids of chunk 0], [chunk 1], ...] in area-descending order (ties: request order)."""
    objs = user['objects']
    order = sorted(range(len(objs)), key=lambda i: (-objs[i]['size'][0] * objs[i]['size'][1], i))
    ids = [objs[i]['id'] for i in order]
    return [ids[k:k + C] for k in range(0, len(ids), C)]


def chunk_request(user, plan, k, placed):
    """(chunk user JSON, maps). placed: {original id: {'pos','yaw','on'}} of every earlier chunk (yaw int deg)."""
    by = {o['id']: o for o in user['objects']}
    fixed = copy.deepcopy(user.get('fixed', []))
    maps = {'fixed': {}, 'obj': {}}
    for cid in (i for c in plan[:k] for i in c):
        fid = f'fixed_{len(fixed) + 1:04d}'
        p = placed[cid]
        fixed.append({'id': fid, 'pos': p['pos'], 'yaw': p['yaw'], 'size': by[cid]['size'], 'category': by[cid]['category']})
        maps['fixed'][fid] = cid
    objects = []
    for n, cid in enumerate(plan[k], 1):
        oid = f'obj_{n:04d}'
        objects.append({'id': oid, 'category': by[cid]['category'], 'size': by[cid]['size']})
        maps['obj'][oid] = cid
    cu = {kk: v for kk, v in user.items() if kk not in ('fixed', 'objects')}
    if fixed:
        cu['fixed'] = fixed
    cu['objects'] = objects
    return cu, maps


def merge(user, plan, answers):
    """answers: [(chunk output dict, maps)] per chunk -> request-order output on original ids."""
    placed = {}
    for out, maps in answers:
        back = {**maps['fixed'], **maps['obj']}
        for p in out['placements']:
            item = {'id': maps['obj'][p['id']], 'pos': p['pos'], 'yaw': p['yaw']}
            if 'on' in p:
                item['on'] = back.get(p['on'], p['on'])       # injected fixed -> movable; real fixed stays
            placed[item['id']] = item
    return {'placements': [placed[o['id']] for o in user['objects']]}


def _check(requests_path, rooms_path, out_path, C):
    rooms = {}
    for line in open(rooms_path):
        r = json.loads(line)
        rooms[r['uid']] = r
    stats = {'C': C, 'rooms': 0, 'validate_ok': 0, 'chunks': 0, 'links': 0, 'links_unexpressible': 0,
             'chars_single': 0, 'chars_chunked': 0, 'rooms_multi_chunk': 0}
    for line in open(requests_path):
        req = json.loads(line)
        room = rooms[req['uid']]
        user = json.loads(req['messages'][1]['content'])
        sizes = {o['id']: o['size'] for o in user['objects']}
        labels = json.loads(contract.messages(room, 'placement', sizes)[2]['content'])
        gt = {p['id']: p for p in labels['placements']}
        plan = chunk_plan(user, C)
        stats['rooms'] += 1
        stats['chunks'] += len(plan)
        stats['rooms_multi_chunk'] += len(plan) > 1
        stats['chars_single'] += len(req['messages'][1]['content']) + len(_j(labels))
        placed, answers = {}, []
        for k in range(len(plan)):
            cu, maps = chunk_request(user, plan, k, placed)
            fwd = {v: kk for kk, v in maps['obj'].items()} | {v: kk for kk, v in maps['fixed'].items()}
            out = {'placements': []}
            for cid in plan[k]:
                p = gt[cid]
                item = {'id': fwd[cid], 'pos': p['pos'], 'yaw': p['yaw']}
                if 'on' in p:
                    stats['links'] += 1
                    if p['on'] in fwd or p['on'].startswith('fixed_'):
                        item['on'] = fwd.get(p['on'], p['on'])
                    else:
                        stats['links_unexpressible'] += 1     # parent in a later chunk: floor instead
                out['placements'].append(item)
                placed[cid] = {'pos': p['pos'], 'yaw': p['yaw'], 'on': item.get('on')}
            stats['chars_chunked'] += len(_j(cu)) + len(_j(out))
            answers.append((out, maps))
        merged = merge(user, plan, answers)
        try:
            contract.validate_output(merged, room, 'placement')
            stats['validate_ok'] += 1
        except ValueError as exc:
            stats.setdefault('errors', []).append([req['uid'], str(exc)])
    stats['char_growth'] = stats['chars_chunked'] / stats['chars_single']
    stats['link_loss'] = stats['links_unexpressible'] / max(1, stats['links'])
    json.dump(stats, open(out_path, 'w'), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    if '--check' in sys.argv:
        a = sys.argv[sys.argv.index('--check') + 1:]
        C = int(sys.argv[sys.argv.index('--C') + 1]) if '--C' in sys.argv else 10
        _check(a[0], a[1], a[2], C)
