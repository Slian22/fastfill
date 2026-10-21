"""Building block for the GPU follow-up (FuncRoom-style segment decode, variant V2 'placed-as-fixed'):
turn a frozen P request + the committed (already placed, S1-repaired) movables into the NEXT chunk's request.
Committed movables are appended to `fixed` as fixed_NNNN (bound numbering continues, contract/infer rules kept);
remaining objects keep their obj ids; `on` references to committed objects are rewritten through the id map and
mapped back when the chunk's answer is merged. The final answer is assembled in the ORIGINAL request order and
validated against the ORIGINAL request, so the interface contract is untouched. No GPU here: prompt text only.

python3 chunk_request.py REQUESTS.jsonl   # self-check on the first 300 requests
"""
import json
import math
import sys

sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/OptiScene')
from fastfill.comparison import infer                      # noqa: E402
from fastfill.comparison import train_pair_helpers as h   # noqa: E402

CHUNK = 8


def chunked_request(request, committed, remaining_ids):
    """committed: {obj_id: {'pos': [x,y,z], 'yaw': deg int, 'on': id|None, 'size': [...], 'category': str}}
    (message units: metres, integer degrees, bottom centre). Returns (request, id_map obj->fixed)."""
    user = h.strict_json(request['messages'][1]['content'])
    fixed = list(user.get('fixed', []))
    id_map = {}
    for oid in [o['id'] for o in user['objects'] if o['id'] in committed]:   # request order = stable numbering
        fid = f'fixed_{len(fixed) + 1:04d}'
        id_map[oid] = fid
        c = committed[oid]
        fixed.append({'id': fid, 'pos': [round(v, 2) for v in c['pos']], 'yaw': int(c['yaw']) % 360,
                      'size': c['size'], 'category': c['category']})
    objects = [o for o in user['objects'] if o['id'] in set(remaining_ids)]
    new_user = {**user, 'fixed': fixed, 'objects': objects}
    if not fixed:
        new_user.pop('fixed')
    return {**request, 'messages': [request['messages'][0],
                                    {'role': 'user', 'content': json.dumps(new_user, ensure_ascii=False,
                                                                           separators=(',', ':'))}]}, id_map


def merge_chunk(answer, id_map):
    """A chunk's parsed placements -> original ids (support references to committed objects mapped back)."""
    back = {v: k for k, v in id_map.items()}
    return [{**p, **({'on': back.get(p['on'], p['on'])} if 'on' in p else {})} for p in answer['placements']]


if __name__ == '__main__':
    n = 0
    for line in open(sys.argv[1]):
        req = json.loads(line)
        user = h.strict_json(req['messages'][1]['content'])
        ids = [o['id'] for o in user['objects']]
        if len(ids) <= CHUNK:
            continue
        # pretend the first chunk was placed at the room's first boundary vertex
        x, y = user['boundary'][0]
        committed = {o['id']: {'pos': [x, y, 0], 'yaw': 90, 'on': None, 'size': o['size'], 'category': o['category']}
                     for o in user['objects'][:CHUNK]}
        new, id_map = chunked_request(req, committed, ids[CHUNK:])
        infer.validate_request(new, 'placement')                      # the frozen request schema still holds
        assert [o['id'] for o in h.strict_json(new['messages'][1]['content'])['objects']] == ids[CHUNK:]
        assert len(id_map) == CHUNK and all(v.startswith('fixed_') for v in id_map.values())
        merged = merge_chunk({'placements': [{'id': ids[CHUNK], 'pos': [1, 1, 0], 'yaw': 0, 'on': id_map[ids[0]]}]}, id_map)
        assert merged[0]['on'] == ids[0]
        n += 1
        if n >= 300:
            break
    print('ok', n, 'chunked requests validated')
