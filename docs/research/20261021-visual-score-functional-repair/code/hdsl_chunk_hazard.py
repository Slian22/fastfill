"""HDSL transfer probe (free, local CPU): does 27B duplication depend on OUTPUT POSITION (fixable by HDSL-style
small-sibling-set calls) or only on room size, and how big is the post-S1 'subtree regeneration set'?

python3 -I hdsl_chunk_hazard.py GEOMETRY_JSONL S1_JSONL ROOMS_JSONL OUT_DIR [--workers N]

A: per size bin x output-position bucket, the duplicate rate of records in request order (HMAC order, so position
   is sequence length only). B: after the first duplicate, the share of remaining records that are duplicates
   (collapse). C: post-S1 failing rooms: objects involved in frozen-checker violations, expanded to support units
   (topmost movable ancestor + subtree) = what an HDSL-style local regeneration would have to re-emit.
"""
import collections
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = '/Users/slian/Desktop/3D/Worldedge/OptiScene'
sys.path.insert(0, REPO)
from fastfill.validate import check, support_ok  # noqa: E402

BINS = ((1, 10), (11, 20), (21, 50), (51, 10 ** 9))
BUCKETS = ((0, 4), (5, 9), (10, 14), (15, 19), (20, 29), (30, 49), (50, 10 ** 9))
CHUNKS = (8, 10, 15, 20, 30)


def bin_of(n):
    return next(f'{a}-{b}' if b < 10 ** 9 else f'{a}+' for a, b in BINS if a <= n <= b)


def bucket_of(k):
    return next(f'{a}-{b}' if b < 10 ** 9 else f'{a}+' for a, b in BUCKETS if a <= k <= b)


def dup_flags(layout):
    seen, flags = set(), []
    for o in layout:
        key = (tuple(round(v, 3) for v in o['pos']), round(o['yaw'], 4))
        flags.append(key in seen)
        seen.add(key)
    return flags


def units(layout):
    """id -> topmost movable ancestor id (a fixed or missing parent ends the chain)."""
    by = {o['id']: o for o in layout}
    top = {}
    for o in layout:
        cur, path = o['id'], []
        while cur in by and by[cur].get('parent') in by and cur not in path:
            path.append(cur)
            cur = by[cur]['parent']
        top[o['id']] = cur
    return top


def involved(room, layout):
    r = {**room, 'objects': layout}
    c = check(r)
    parents = {o['id']: o for o in layout + (room.get('fixed') or [])}
    sup = [o['id'] for o in layout if (o.get('parent') is not None and o['parent'] not in parents)
           or not support_ok(o, parents.get(o.get('parent')))]
    ids = set(c['oob']) | set(sup) | set(c['ceiling']) | {p[0] for p in c['fixed_blocking']}
    for kind in ('main', 'severe'):
        for a, b in c['collisions'][kind]:
            ids.update((a, b))
    fail = bool(c['oob'] or sup or c['ceiling'] or c['collisions']['main'] or c['collisions']['severe']
                or c['fixed_blocking'])
    return fail, ids


def one(task):
    row, s1row, room = task
    raw = row['raw_layout']
    n = len(raw)
    flags = dup_flags(raw)
    first = next((k for k, f in enumerate(flags) if f), None)
    after = flags[first + 1:] if first is not None else []
    out = {'uid': row['uid'], 'n': n, 'bin': bin_of(n), 'raw_pass': bool(row['geometry_success']),
           'dups': sum(flags), 'first_dup': first,
           'after_first_total': len(after), 'after_first_dups': sum(after),
           'dup_positions': [k for k, f in enumerate(flags) if f]}
    s1 = s1row['s1_layout'] if s1row and s1row.get('s1_layout') else raw
    out['s1_status'] = s1row['s1'].get('status') if s1row else None
    fail, ids = involved(room, s1)
    out['s1_fail'] = fail
    if fail:
        top = units(s1)
        unit_ids = {top[i] for i in ids}
        regen = [o['id'] for o in s1 if top[o['id']] in unit_ids]
        rawdup = {o['id'] for o, f in zip(raw, flags) if f}
        # members of a duplicate group include the first copy: mark the whole group
        groups = collections.defaultdict(list)
        for o in raw:
            groups[(tuple(round(v, 3) for v in o['pos']), round(o['yaw'], 4))].append(o['id'])
        dupgroup = {i for g in groups.values() if len(g) > 1 for i in g}
        out.update({'involved': len(ids), 'units': len(unit_ids), 'regen': len(regen), 'regen_frac': len(regen) / n,
                    'involved_in_dupgroup': len(ids & dupgroup), 'involved_raw_dup': len(ids & rawdup)})
    return out


def main():
    geom, s1p, roomsp, outd = (Path(a) for a in sys.argv[1:5])
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
    outd.mkdir(parents=True, exist_ok=True)
    rooms = {}
    for line in open(roomsp):
        r = json.loads(line)
        rooms[r['uid']] = {'boundary': r['boundary'], 'boundary_type': r['boundary_type'], 'height': r.get('height'),
                           'fixed': r.get('fixed') or []}
    s1 = {}
    for line in open(s1p):
        r = json.loads(line)
        s1[r['uid']] = {'s1_layout': r.get('s1_layout'), 's1': r.get('s1') or {}}
    tasks = []
    for line in open(geom):
        r = json.loads(line)
        if not r.get('complete') or not r.get('raw_layout'):
            continue
        tasks.append((r, s1.get(r['uid']), rooms[r['uid']]))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, tasks, chunksize=32))
    with open(outd / 'rooms.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    # A: hazard by bin x position bucket
    rec = collections.Counter()
    dup = collections.Counter()
    for r in rows:
        pos = set(r['dup_positions'])
        for k in range(r['n']):
            key = (r['bin'], bucket_of(k))
            rec[key] += 1
            dup[key] += k in pos
    hazard = {f'{b}|{p}': {'records': rec[(b, p)], 'dups': dup[(b, p)], 'rate': dup[(b, p)] / rec[(b, p)]}
              for (b, p) in sorted(rec, key=lambda t: (BINS.index(next(x for x in BINS if bin_of(x[0]) == t[0])),
                                                       BUCKETS.index(next(x for x in BUCKETS if bucket_of(x[0]) == t[1]))))}
    # share of duplicates at position >= C (what a chunk of C objects would not reach as same-call context)
    share = {}
    for b in (bin_of(a) for a, _ in BINS):
        allp = [k for r in rows if r['bin'] == b for k in r['dup_positions']]
        share[b] = {str(C): (sum(k >= C for k in allp) / len(allp) if allp else None) for C in CHUNKS} | {'dups': len(allp)}
    # B: collapse after first duplicate
    collapse = {}
    for b in (bin_of(a) for a, _ in BINS):
        rs = [r for r in rows if r['bin'] == b and r['first_dup'] is not None]
        tot = sum(r['after_first_total'] for r in rs)
        collapse[b] = {'rooms_with_dup': len(rs), 'after_first_records': tot,
                       'after_first_dup_rate': (sum(r['after_first_dups'] for r in rs) / tot) if tot else None,
                       'first_dup_median_index': statistics.median(r['first_dup'] for r in rs) if rs else None,
                       'first_dup_median_frac': statistics.median(r['first_dup'] / r['n'] for r in rs) if rs else None}
    # C: regeneration set after S1
    regen = {}
    for b in (bin_of(a) for a, _ in BINS):
        rs = [r for r in rows if r['bin'] == b and r['s1_fail']]
        if not rs:
            regen[b] = {'s1_fail_rooms': 0}
            continue
        fr = [r['regen_frac'] for r in rs]
        regen[b] = {'s1_fail_rooms': len(rs), 'rooms_in_bin': sum(r['bin'] == b for r in rows),
                    'involved_median': statistics.median(r['involved'] for r in rs),
                    'units_median': statistics.median(r['units'] for r in rs),
                    'regen_objects_median': statistics.median(r['regen'] for r in rs),
                    'regen_frac_median': statistics.median(fr), 'regen_frac_mean': statistics.mean(fr),
                    'share_regen_le8': sum(r['regen'] <= 8 for r in rs) / len(rs),
                    'share_regen_le15': sum(r['regen'] <= 15 for r in rs) / len(rs),
                    'share_regen_frac_le_0.3': sum(x <= 0.3 for x in fr) / len(rs),
                    'involved_in_dupgroup_share': sum(r['involved_in_dupgroup'] for r in rs) / max(1, sum(r['involved'] for r in rs))}
    summary = {'rooms': len(rows), 's1_fail_total': sum(r['s1_fail'] for r in rows),
               'raw_pass_total': sum(r['raw_pass'] for r in rows), 'hazard': hazard,
               'dup_share_at_or_beyond_C': share, 'collapse_after_first_dup': collapse, 'regen_after_s1': regen}
    json.dump(summary, open(outd / 'summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
