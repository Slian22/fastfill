import json, collections, sys
path = sys.argv[1]
n = 0; rooms_dup = 0; floor_dup_objs = 0; child_dup_objs = 0; rooms_floor_only = 0; rooms_child_only = 0; rooms_both = 0
dup_fail = 0; groups_sizes = collections.Counter(); by_cat_parent = collections.Counter()
keys_seen = None
for line in open(path):
    r = json.loads(line); n += 1
    if keys_seen is None: keys_seen = list(r.keys())
    if not r['complete']: continue
    lay = r['raw_layout']
    key = collections.defaultdict(list)
    for o in lay:
        k = (round(o['pos'][0],3), round(o['pos'][1],3), round(o['pos'][2],3), round(o['yaw'],6))
        key[k].append(o)
    fl = ch = 0
    for k, objs in key.items():
        if len(objs) > 1:
            groups_sizes[min(len(objs), 10)] += 1
            for o in objs[1:]:
                if o.get('parent') is None: fl += 1
                else: ch += 1
    if fl or ch:
        rooms_dup += 1; floor_dup_objs += fl; child_dup_objs += ch
        if fl and ch: rooms_both += 1
        elif fl: rooms_floor_only += 1
        else: rooms_child_only += 1
        if not r['geometry_success']: dup_fail += 1
print('rows', n, 'keys', keys_seen)
print('rooms_with_dup', rooms_dup, 'fail', dup_fail, 'floor_dup_objs', floor_dup_objs, 'child_dup_objs', child_dup_objs)
print('rooms floor_only', rooms_floor_only, 'child_only', rooms_child_only, 'both', rooms_both)
print('group sizes', sorted(groups_sizes.items()))
