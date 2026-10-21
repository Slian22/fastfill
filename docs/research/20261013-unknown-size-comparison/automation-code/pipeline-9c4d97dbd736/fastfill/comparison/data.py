"""Build paired category-only engineering data from hash-verified frozen IR.

Never derives identity from canonical target-sorted IDs. Pilot inherits historical
release splits; it is not a newly blind benchmark. Original IR stays untouched.
"""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

EXCLUDE_FLAGS = {'oob_objects', 'overlapping_furniture', 'fixed_collision'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def select_release_rows(path, expected, per_source, seed, training):
    if sha256(path) != expected['sha256']:
        raise ValueError(f'release hash mismatch: {path}')
    buckets, seen = collections.defaultdict(list), set()
    excluded = collections.Counter()
    consumed = hashlib.sha256()
    with open(path, 'rb') as f:
        for line in f:
            consumed.update(line)
            r = json.loads(line)
            uid, source = r['uid'], r['source']
            if uid in seen:
                raise ValueError(f'duplicate release UID: {uid}')
            seen.add(uid)
            flags = r.get('flags') or {}
            if training and any(flags.get(k) for k in EXCLUDE_FLAGS):
                excluded[source] += 1
                continue
            rank = hashlib.sha256((seed + '\0' + uid).encode()).hexdigest()
            buckets[source].append((rank, uid))
    if consumed.hexdigest() != expected['sha256']:
        raise ValueError(f'release hash changed while selecting: {path}')
    if len(seen) != expected['lines']:
        raise ValueError('release line count mismatch')
    selected = {uid: source for source, values in sorted(buckets.items())
                for _, uid in sorted(values)[:per_source]}
    return selected, {'input_rows': len(seen), 'excluded_flags': dict(excluded),
                      'selected_by_source': dict(collections.Counter(selected.values()))}


class Components:
    """Deterministic union-find, private mutable index; never changes room data."""
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            nxt = self.parent[key]
            self.parent[key] = root
            key = nxt
        return root

    def union(self, a, b):
        x, y = self.find(a), self.find(b)
        if x != y:
            self.parent[max(x, y)] = min(x, y)


def reconstruct(ir, release_manifest, selections):
    from fastfill.build import prep
    from fastfill.split import content_key, layout_key

    args = SimpleNamespace(**release_manifest['args'])
    wanted = {uid for ids in selections.values() for uid in ids}
    if sum(map(len, selections.values())) != len(wanted):
        raise ValueError('cross-split UID overlap')
    rooms, groups, components = {}, {}, Components()
    for filename, expected in sorted(release_manifest['ir_sha256'].items()):
        path = ir / filename
        before = sha256(path)
        if before != expected:
            raise ValueError(f'IR hash mismatch: {path}')
        consumed = hashlib.sha256()
        with path.open('rb') as f:
            for line in f:
                consumed.update(line)
                r = json.loads(line)
                group = 'g:' + r['group']
                components.find(group)
                for alias in r.get('meta', {}).get('group_aliases') or []:
                    components.union(group, 'a:' + alias)
                if r['uid'] not in wanted:
                    continue
                if r['uid'] in rooms:
                    raise ValueError('duplicate selected raw UID')
                prepared, reason = prep(r, args)
                if reason:
                    raise ValueError(f'release UID rejected by prep: {r["uid"]}: {reason}')
                rooms[r['uid']] = prepared
                groups[r['uid']] = group
                for prefix, key in [('c:', content_key(prepared)), ('l:', layout_key(prepared))]:
                    if key is not None:
                        components.union(group, prefix + key)
        if consumed.hexdigest() != before or sha256(path) != before:
            raise ValueError(f'IR changed while reading: {path}')
        print(f'joined {filename}: {len(rooms)}/{len(wanted)} selected rooms', flush=True)
    if rooms.keys() != wanted:
        raise ValueError(f'unresolved selected UIDs: {len(wanted - rooms.keys())}')
    component_by_uid = {uid: components.find(g) for uid, g in groups.items()}
    owners = {}
    for split, ids in selections.items():
        for uid in ids:
            c = component_by_uid[uid]
            if c in owners and owners[c] != split:
                raise ValueError('cross-split component overlap')
            owners[c] = split
    return rooms, component_by_uid


def check_input_invariance(room, prior_sizes):
    from fastfill.comparison.contract import messages

    changed = copy.deepcopy(room)
    changed['objects'] = list(reversed([
        {**o, 'size': [7.31, 8.17, 9.13], 'pos': [23, 29, 0],
         'yaw': 1.234, 'parent': None} for o in room['objects']]))
    # Rebinding before serialization preserves the target-independent ID order.
    changed['objects'].sort(key=lambda o: o['id'])
    for arm in ('joint', 'placement'):
        kwargs = {'estimated_sizes': prior_sizes} if arm == 'placement' else {}
        if messages(room, arm=arm, **kwargs)[:2] != messages(changed, arm=arm, **kwargs)[:2]:
            raise ValueError(f'input depends on movable target: {room["uid"]}/{arm}')


def geometry_audit(room, sizes):
    from fastfill.validate import check

    changed = {**room, 'objects': [{**o, 'size': sizes[o['id']]} for o in room['objects']]}
    result = check(changed)
    return {'oob': len(result['oob']), 'support_fail': len(result['support_fail']),
            'ceiling': len(result['ceiling']), 'collisions': len(result['collisions']['main']),
            'fixed_blocking': len(result['fixed_blocking'])}


def json_dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def build(release, ir, out, train_per_source=64, dev_per_source=4, seed='unknown-size-v1-20261013'):
    from fastfill.comparison.contract import bind_room, messages
    from fastfill.comparison.size_prior import build_oof_priors, predict_sizes

    if out.exists():
        raise FileExistsError(f'refusing existing output: {out}')
    code_root = Path(__file__).resolve().parents[1]
    code_names = ['comparison/data.py', 'comparison/contract.py', 'comparison/size_prior.py',
                  'scene.py', 'build.py', 'anchors.py', 'validate.py', 'split.py']
    code_hashes = {name: sha256(code_root / name) for name in code_names}
    man_path = release / 'MANIFEST.json'
    original_hash = sha256(man_path)
    man = json.loads(man_path.read_text())
    selections, selection_ledger = {}, {}
    for split, limit in [('train', train_per_source), ('dev', dev_per_source)]:
        ids, info = select_release_rows(release / f'{split}.jsonl',
                                       man['files'][f'{split}.jsonl'], limit, seed, split == 'train')
        selections[split], selection_ledger[split] = ids, info
    if not all(selections.values()):
        raise ValueError('empty pilot split')
    raw, components = reconstruct(ir, man, selections)
    bound, sidecars = {}, {}
    for uid, room in raw.items():
        bound[uid], sidecars[uid] = bind_room(room, seed=seed)
    train = [bound[u] for u in sorted(selections['train'])]
    oof = build_oof_priors(train, {u: components[u] for u in selections['train']}, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name, dir=out.parent) as tmp:
        dest = Path(tmp)
        for arm in ('joint', 'placement'):
            (dest / arm).mkdir()
        audits, split_manifest = [], []
        for split, ids in selections.items():
            records = {'joint': [], 'placement': []}
            for uid in sorted(ids):
                room = bound[uid]
                fold = oof['fold_by_uid'][uid] if split == 'train' else None
                prior = oof['priors'][str(fold)] if split == 'train' else oof['full_prior']
                sizes = {oid: [max(0.01, round(v, 2)) for v in values]
                         for oid, values in predict_sizes(room, prior).items()}
                check_input_invariance(room, sizes)
                for arm in records:
                    kwargs = {'estimated_sizes': sizes} if arm == 'placement' else {}
                    records[arm].append({'uid': uid, 'source': room['source'], 'split': split,
                                         'group': components[uid], 'flags': room['meta']['flags'],
                                         'messages': messages(room, arm=arm, **kwargs)})
                audits.append({'uid': uid, 'split': split, 'group': components[uid], 'fold': fold,
                               'estimated_sizes': sizes, 'estimated_geometry': geometry_audit(room, sizes),
                               'reference_geometry': geometry_audit(room, {o['id']: o['size'] for o in room['objects']})})
                split_manifest.append({'uid': uid, 'split': split, 'source': room['source'],
                                       'component': components[uid], 'fold': fold})
            for arm, rows in records.items():
                (dest / arm / f'{split}.jsonl').write_text(''.join(
                    json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
            (dest / f'{split}_rooms.jsonl').write_text(''.join(
                json.dumps(bound[u], ensure_ascii=False, allow_nan=False) + '\n' for u in sorted(ids)))
        json_dump(dest / 'inventory-sidecar.json', sidecars)
        json_dump(dest / 'split-manifest.json', split_manifest)
        json_dump(dest / 'size-priors.json', oof)
        json_dump(dest / 'geometry-audit.json', audits)
        data_manifest = {
            'version': 1, 'scope': 'engineering_pilot_not_formal_benchmark', 'seed': seed,
            'source_release_manifest_sha256': original_hash,
            'implementation_sha256': code_hashes,
            'source_release_files': {s: man['files'][f'{s}.jsonl'] for s in selections},
            'ir_sha256': man['ir_sha256'], 'selection': selection_ledger,
            'input_policy': 'category_only_frozen_prepared_inventory_no_target_derived_ids',
            'estimated_size_precision': 'centimeters; min0.01m, same display precision as joint labels',
            'split_policy': 'historical_release_train_dev_only_no_new_blind_test_claim',
            'component_policy': 'all_IR_group_alias_connections_and_selected_room_content_layout_keys',
            'geometry_policy': 'unchanged_reference_positions; estimated_size_infeasibility_audited_not_filtered',
            'counts': {s: len(ids) for s, ids in selections.items()},
            'input_counterfactual_checks': len(bound) * 2,
            'files': {str(p.relative_to(dest)): sha256(p) for p in sorted(dest.rglob('*')) if p.is_file()},
        }
        json_dump(dest / 'MANIFEST.json', data_manifest)
        for arm in ('joint', 'placement'):
            json_dump(dest / arm / 'MANIFEST.json', {**data_manifest, 'arm': arm})
        if sha256(man_path) != original_hash:
            raise ValueError('release manifest changed during build')
        if any(sha256(code_root / name) != value for name, value in code_hashes.items()):
            raise ValueError('implementation changed during build')
        if any(sha256(release / f'{s}.jsonl') != man['files'][f'{s}.jsonl']['sha256']
               for s in selections):
            raise ValueError('source release changed during build')
        os.rename(dest, out)
    return data_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--ir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-per-source', type=int, default=64)
    p.add_argument('--dev-per-source', type=int, default=4)
    args = p.parse_args()
    if min(args.train_per_source, args.dev_per_source) < 1:
        p.error('per-source limits must be positive')
    result = build(args.release, args.ir, args.out, args.train_per_source, args.dev_per_source)
    print(json.dumps({'counts': result['counts'], 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
