"""Independent frozen lineage/OOF audit. Full scans require a bounded CPU Slurm job.

Only category/room normalizers are imported from the SHA-pinned main-data snapshot.
No prep replay, geometry repair, tokenizer, model, API, or binary asset reads.
Runtime writes are confined to a newly created --out directory. Run python -B.
"""
import argparse
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
SEED = 'unknown-size-v1-20261013'
SOURCES = ('HSSD200 IL3D_3dfront IL3D_synthetic InteriorGS InternScenes_3rscan '
           'InternScenes_arkit InternScenes_gen InternScenes_mp3d InternScenes_scannet '
           'MansionWorld MultiScan OptiScene_holodeck SAGE-10k Scan2CAD SceneSmith '
           'SpatialGen SpatialLM Structured3D').split()
HELDOUT = {'SceneSmith', 'SpatialGen'}
FLAGS = {'oob_objects', 'overlapping_furniture', 'fixed_collision'}
PINS = {
    '__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'scene.py': '91712134b9fce57a0893af25374cc8d62b0855f9374dc8f81800b43ceb00ce60',
    'anchors.py': '44a3b3275ebca87d5eb126e221fcd005e4c8e1b7cd36061bcd76c73e620a8780',
    'validate.py': 'ddaff1e5f6dedd949c61267780b92d2a18981926d55939ed038e1e3cdbe17158',
    'build.py': 'fbd36a4f841f4606ab0475e097077c4ea906046caa71aecba0295640c831939f',
    'split.py': '35bbfe4a2ba7b364454093de4a157a2aa2e1600d5316d03c1fa19a1ce4ed3a2f',
    'requirements.txt': 'd79656a13dbaab6143a79e0ba78be72c104a6562cc67d1c257a3f43fdec8a8fe',
    'comparison/__init__.py': '215453181009e47e365023af136e63d360b076d5a68e6c13695ef721ed52e6a1',
    'comparison/contract.py': 'ef1e2cd9e449449f5441348e16d9a2bb2bd2ff8618132e6b6ed378347e64cefc',
    'comparison/size_prior.py': 'e7a546b8af793755f3aeb8bd074e78798858f5528ca615273d5f56eef5d61ffc',
    'comparison/data.py': '8bd85ce71862bafd15d949b2a912edb6e8f6311fcb7d5e338ce92faf593dc049',
    'comparison/corpus.py': '2cb3973d1cc9cd53696b7292c95cd8f12daa913b05f640e27b37ef6ec5385683',
    'comparison/main_data.py': 'f14d8808a3358d7b80bdc559a84006e6b7cae19584759c71e8586a5a3505809d',
    'comparison/pose_targets.py': '0717748c3a794c5341595c0e310beae496497c9802bd2a49225919157af353ba',
    'comparison/eval_data.py': 'ddd0ca5729739f4603db1e0eb5e53f851b1f02539563b663d944c418f087104b',
    'comparison/train_pair_helpers.py': '6a8268dc576f34a03d79e67e876417b01b7a7e8ce76241b5299b41e50176ea76',
}
MANIFEST_PINS = {
    'release': '664eecdcdaea593bc7f8653ea544b46e415f1c0bc8bf7e09065fbe89796caf6e',
    'corpus': 'bf6444ae51f21e8f944168d985576f14029a5ecf9400f6e625ffa62ee5f636c6',
    'main': 'e588e2f7c0def2c6515810816e0fd101b6dfdaa907274ce37cc9cd0d2c7c63e6',
    'eval': '6f9d801b9166a2561dcb6665ee3e89dd449d9cdc293bdc8570b6487af8771604',
}


def H(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def uid_provenance(uids):
    hashes = sorted(H(u) for u in uids)
    return hashes, H(compact(hashes))


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def signature(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def safe_path(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsafe manifest path: ' + str(path))
    return path


class Audit:
    def __init__(self, out):
        self.out, self.start = out, time.monotonic()
        self.checks, self.files, self.summary, self.stamps = {}, {}, {}, {}
        self.max_delta = {'absolute': 0.0, 'relative': 0.0}

    def check(self, ok, name, example=None):
        record = self.checks.setdefault(name, {'passed': 0, 'failed': 0, 'examples': []})
        record['passed' if ok else 'failed'] += 1
        if not ok and len(record['examples']) < 5:
            record['examples'].append(example)
        return bool(ok)

    def numeric(self, expected, actual, label):
        if not self.check(isinstance(actual, list) and len(actual) == len(expected), 'median_shape', label):
            return
        for x, y in zip(expected, actual):
            valid = type(y) in (int, float) and math.isfinite(y) and y > 0
            if valid:
                absolute = abs(x - y)
                relative = absolute / max(abs(x), abs(y))
                self.max_delta['absolute'] = max(self.max_delta['absolute'], absolute)
                self.max_delta['relative'] = max(self.max_delta['relative'], relative)
            self.check(valid and math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12),
                       'median_values', {'bucket': label, 'expected': x, 'actual': y})

    def finish_file(self, path, before, digest, expected, records, fields, sources=None):
        after = signature(path)
        self.check(before == after, 'file_stable_during_read', str(path))
        self.check(expected is None or digest == expected, 'file_sha256', str(path))
        self.stamps[str(path)] = after
        self.files[str(path)] = {'sha256': digest, 'expected_sha256': expected, 'bytes': after[2],
                                'records': records, 'fields': dict(fields),
                                'by_source': dict(sources or {}), 'complete_scan': True}

    def read(self, path, expected=None):
        before = signature(path)
        raw = path.read_bytes()
        value = strict_json(raw)
        rows = value if isinstance(value, list) else ([value] if isinstance(value, dict) else [])
        if path.name == 'inventory-sidecar.json' and isinstance(value, dict):
            rows = value.values()
        fields = Counter(k for r in rows if isinstance(r, dict) for k in r)
        self.finish_file(path, before, hashlib.sha256(raw).hexdigest(), expected, len(rows), fields)
        return value

    def blob(self, path, expected=None):
        before, h = signature(path), hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        self.finish_file(path, before, h.hexdigest(), expected, None, {})
        return h.hexdigest()

    def rows(self, path, expected=None):
        before, h, fields, sources = signature(path), hashlib.sha256(), Counter(), Counter()
        number = 0
        self.files[str(path)] = {'complete_scan': False}
        with path.open('rb') as stream:
            for number, raw in enumerate(stream, 1):
                h.update(raw)
                row = strict_json(raw)
                if not isinstance(row, dict):
                    raise ValueError(f'non-object JSONL: {path}:{number}')
                fields.update(row.keys())
                sources[row.get('source', '<absent>')] += 1
                yield number, row, hashlib.sha256(raw).hexdigest()
        self.finish_file(path, before, h.hexdigest(), expected, number, fields, sources)

    def progress(self, stage, **values):
        print(compact({'stage': stage, 'seconds': round(time.monotonic() - self.start, 2), **values}), flush=True)

    def save(self, status, error=None):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {'status': status, 'error': error, 'utc': datetime.now(timezone.utc).isoformat(),
                  'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'python': sys.version,
                  'elapsed_seconds': time.monotonic() - self.start,
                  'max_rss_bytes': rss * (1 if sys.platform == 'darwin' else 1024),
                  'checks': self.checks, 'files': self.files, 'summary': self.summary,
                  'median_max_delta': self.max_delta, 'median_tolerance': {'abs': 1e-12, 'rel': 1e-12}}
        if self.out is not None:
            temporary = self.out / 'result.json.tmp'
            temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            temporary.replace(self.out / 'result.json')
        return result


class DSU:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            node, self.parent[node] = self.parent[node], root
        return root

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def content_key(room):
    if len(room['objects']) < 3 or room.get('source') == 'MansionWorld':
        return None
    dims = sorted(tuple(sorted(round(x / 0.05) for x in o['size'])) for o in room['objects'])
    return hashlib.sha1(repr(dims).encode()).hexdigest()


def num(value):
    value = round(float(value), 2) + 0.0
    return int(value) if value == int(value) else value


def layout_key(room):
    if not room.get('boundary'):
        return None
    results = []
    for k, (c, s) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        boundary, objects = room['boundary'], room['objects']
        if k:
            rotated = [(c*x-s*y, s*x+c*y) for x, y in boundary]
            mx, my = min(x for x, _ in rotated), min(y for _, y in rotated)
            boundary = [[num(x-mx), num(y-my)] for x, y in rotated]
            first = min(range(len(boundary)), key=lambda i: (sum(v*v for v in boundary[i]), boundary[i]))
            boundary = boundary[first:] + boundary[:first]
            poses = [(num(c*o['pos'][0]-s*o['pos'][1]-mx), num(s*o['pos'][0]+c*o['pos'][1]-my),
                      round(math.degrees((o['yaw'] + k*math.pi/2) % (2*math.pi))) % 360) for o in objects]
        else:
            poses = [(num(o['pos'][0]), num(o['pos'][1]), round(math.degrees(o['yaw'])) % 360) for o in objects]
        results.append(hashlib.sha1(repr((boundary, sorted(poses))).encode()).hexdigest())
    return min(results)


def exclusion_reasons(entry, heldout_components):
    reasons = []
    if entry['split'] != 'train':
        reasons.append('not_exact_train')
    if entry['source'] in HELDOUT:
        reasons.append('heldout_source')
    if entry.get('component') in heldout_components:
        reasons.append('heldout_source_component')
    return reasons + ['historical_' + k for k in sorted(FLAGS) if entry['training_flags'].get(k)]


def assign_folds(mapping):
    components = sorted(set(mapping.values()), key=lambda c: (H(compact([SEED, c])), c))
    if len(components) < 5:
        raise ValueError('fewer than five eligible components')
    folds = {c: i % 5 for i, c in enumerate(components)}
    return {u: folds[c] for u, c in mapping.items()}


class Observations:
    """Compact object arrays; geometry/targets are never retained."""
    def __init__(self):
        self.sizes, self.cats, self.rts = array('d'), array('I'), array('I')
        self.categories, self.room_types, self.rooms = {}, {}, {}

    def add(self, uid, room_type, objects, fixed):
        if uid in self.rooms:
            raise ValueError('duplicate observation UID: ' + uid)
        start = len(self.cats)
        rt = self.room_types.setdefault(room_type, len(self.room_types))
        for cat, size in objects:
            if len(size) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in size):
                raise ValueError('nonpositive/nonfinite size: ' + uid)
            self.sizes.extend(size)
            self.cats.append(self.categories.setdefault(cat, len(self.categories)))
            self.rts.append(rt)
        self.rooms[uid] = (start, len(self.cats), fixed)

    def fit(self, uids):
        import numpy as np
        spans = [self.rooms[u] for u in sorted(uids)]
        indices = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b, _ in spans]) if spans else np.array([], dtype=np.int64)
        if not len(indices):
            raise ValueError('empty fit: no movable objects')
        cats = np.frombuffer(self.cats, dtype=np.uint32)[indices]
        rts = np.frombuffer(self.rts, dtype=np.uint32)[indices]
        logs = np.log(np.frombuffer(self.sizes, dtype=np.float64).reshape(-1, 3)[indices])
        def bucket(values):
            return {'count': len(values), 'size': np.exp(np.median(values, axis=0)).tolist()}
        global_bucket = bucket(logs)
        order = np.lexsort((rts, cats))
        cats, rts, logs = cats[order], rts[order], logs[order]
        cat_names, rt_names = list(self.categories), list(self.room_types)
        boundaries = np.r_[0, np.flatnonzero(cats[1:] != cats[:-1]) + 1, len(cats)]
        categories, pairs = {}, {}
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            cat = cat_names[cats[a]]
            categories[cat] = bucket(logs[a:b])
            edges = np.r_[a, np.flatnonzero(rts[a+1:b] != rts[a:b-1]) + a + 1, b]
            pairs[cat] = {rt_names[rts[x]]: bucket(logs[x:y]) for x, y in zip(edges[:-1], edges[1:])}
        hashes, digest = uid_provenance(uids)
        return {'version': 1, 'estimator': 'axis_log_median', 'min_count': 5,
                'conditioning': 'category_room_type', 'fallback': ['category', 'global'],
                'category_room_type': pairs, 'category': categories, 'global': global_bucket,
                'provenance': {'room_count': len(uids), 'object_count': len(indices),
                    'fixed_excluded_count': sum(n for _, _, n in spans), 'category_count': len(categories),
                    'category_room_type_count': sum(map(len, pairs.values())),
                    'fitted_uid_hashes': hashes, 'fitted_uid_sha256': digest}}


class Profiles:
    """Basic keys/types plus at most five scalar values per relevant field/source."""
    def __init__(self):
        self.data = {}

    def observe(self, source, scope, obj):
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            label = source + '/' + scope + '/' + key
            rec = self.data.setdefault(label, {'count': 0, 'types': Counter(), 'values': Counter(), 'other_values': 0})
            rec['count'] += 1
            rec['types'][type(value).__name__] += 1
            relevant = any(w in key.lower() for w in ('front', 'boundary', 'height', 'wall', 'asset', 'tilt', 'mask'))
            if isinstance(value, bool) or (relevant and (value is None or type(value) in (str, int, float))):
                encoded = compact(value)
                if len(encoded) <= 160 and (encoded in rec['values'] or len(rec['values']) < 5):
                    rec['values'][encoded] += 1
                else:
                    rec['other_values'] += 1

    def room(self, row):
        source = row['source']
        self.observe(source, 'room', row)
        self.observe(source, 'room.meta', row.get('meta'))
        for kind in ('objects', 'fixed'):
            for obj in row.get(kind) or []:
                self.observe(source, kind, obj)
                self.observe(source, kind + '.meta', obj.get('meta'))


def scan_release(audit, root, manifest, by_source):
    entries = {}
    for split in ('train', 'dev', 'test'):
        receipt = manifest['files'][split + '.jsonl']
        count = 0
        for number, row, digest in audit.rows(root / (split + '.jsonl'), receipt['sha256']):
            uid, source = row['uid'], row['source']
            count += 1
            by_source[source]['release_' + split] += 1
            if not audit.check(uid not in entries, 'release_uid_unique', uid):
                continue
            flags = row.get('flags')
            flags = {} if flags is None else flags
            if not isinstance(flags, dict):
                raise ValueError('invalid historical flags: ' + uid)
            entries[uid] = {'uid': uid, 'source': source, 'split': split, 'training_flags': flags,
                            'release_line': number, 'release_row_sha256': digest}
        audit.check(count == receipt['lines'], 'release_record_count', split)
    audit.check(len(entries) == 160964, 'frozen_release_population', len(entries))
    return entries


def scan_raw(audit, root, hashes, entries, dsu, by_source):
    seen, profiles = set(), Profiles()
    audit.check(set(hashes) == {s + '.jsonl' for s in SOURCES}, 'all_18_IR_inventory')
    audit.check({p.name for p in root.iterdir() if p.suffix == '.jsonl'} == set(hashes), 'IR_disk_inventory')
    for name, expected in sorted(hashes.items()):
        for number, raw, digest in audit.rows(safe_path(root, name), expected):
            uid, source = raw['uid'], raw['source']
            audit.check(source in SOURCES, 'known_raw_source', source)
            audit.check(uid not in seen, 'raw_uid_unique', uid)
            seen.add(uid)
            by_source[source]['raw'] += 1
            profiles.room(raw)
            group = 'g:' + raw['group']
            dsu.find(group)
            aliases = raw.get('meta', {}).get('group_aliases') or []
            if not isinstance(aliases, list) or any(not isinstance(x, str) or not x for x in aliases):
                raise ValueError('invalid group aliases: ' + uid)
            for alias in aliases:
                dsu.union(group, 'a:' + alias)
            if uid not in entries:
                by_source[source]['excluded_unreleased'] += 1
                continue
            entry = entries[uid]
            audit.check(source == entry['source'], 'raw_release_source', uid)
            entries[uid] = {**entry, 'original_group': raw['group'], 'raw_file': name,
                            'raw_line': number, 'raw_sha256': digest}
        audit.progress('raw', file=name, total=len(seen))
    audit.summary['raw_profiles'] = profiles.data
    audit.check(set(entries) <= seen, 'released_UIDs_found_in_raw')
    audit.check(len(seen) == 188670, 'frozen_raw_population', len(seen))
    audit.check(len(seen - entries.keys()) == 27706, 'frozen_unreleased_population', len(seen - entries.keys()))
    return len(seen)


def scan_prepared(audit, path, expected, entries, dsu, normalizers, by_source):
    seen, observations = set(), Observations()
    previous = None
    for number, room, digest in audit.rows(path, expected):
        uid = room['uid']
        audit.check(uid not in seen, 'prepared_uid_unique', uid)
        seen.add(uid)
        if not audit.check(uid in entries, 'prepared_released_uid', uid):
            continue
        entry = entries[uid]
        audit.check((room['source'], room['split'], room['group']) ==
                    (entry['source'], entry['split'], entry['original_group']), 'prepared_original_identity', uid)
        location = (entry['raw_file'], entry['raw_line'])
        audit.check(previous is None or previous < location, 'prepared_raw_order', uid)
        previous = location
        content, layout = content_key(room), layout_key(room)
        for prefix, key in (('c:', content), ('l:', layout)):
            if key is not None:
                dsu.union('g:' + entry['original_group'], prefix + key)
        entries[uid] = {**entry, 'content_key': content, 'layout_key': layout,
                        'prepared_flags': room['meta']['flags'], 'prepared_sha256': digest}
        by_source[entry['source']]['prepared_' + entry['split']] += 1
        if not exclusion_reasons(entry, set()):
            objects = room['objects']
            audit.check(len({o['id'] for o in objects}) == len(objects), 'prepared_object_ids_unique', uid)
            observations.add(uid, normalizers._norm_room_type(room.get('room_type')) or '',
                             [(normalizers._norm_cat(o.get('category')), o['size']) for o in objects],
                             len(room.get('fixed', [])))
        if number % 10000 == 0:
            audit.progress('prepared', rows=number)
    audit.check(seen == entries.keys(), 'prepared_full_release_coverage')
    return observations


def close_identities(audit, entries, published, dsu, corpus_manifest):
    owners = {}
    for uid, entry in entries.items():
        component = dsu.find('g:' + entry['original_group'])
        entries[uid] = {**entry, 'component': component}
        audit.check(owners.setdefault(component, entry['split']) == entry['split'], 'component_split_closure', uid)
    seen, previous = set(), None
    for actual in published:
        uid = actual['uid']
        audit.check(uid not in seen, 'identity_uid_unique', uid)
        seen.add(uid)
        audit.check(previous is None or previous < uid, 'identity_uid_sorted', uid)
        previous = uid
        audit.check(uid in entries, 'identity_uid_released', uid)
        if uid in entries:
            for key in set(actual) | set(entries[uid]):
                audit.check(key in actual and key in entries[uid] and actual[key] == entries[uid][key],
                            'identity_field:' + key, uid)
    audit.check(seen == entries.keys(), 'identity_complete_UID_coverage')
    counts = dict(Counter(e['split'] for e in entries.values()))
    audit.check(counts == corpus_manifest['counts'], 'corpus_split_counts')
    audit.check(dict(Counter(e['source'] for e in entries.values())) == corpus_manifest['by_source'], 'corpus_source_counts')
    audit.check(len(owners) == corpus_manifest['audit']['components'], 'corpus_component_count')
    audit.summary['split_counts'] = counts
    audit.summary['component_count'] = len(owners)


def compare_prior(audit, expected, actual, name):
    audit.check(set(expected) == set(actual), 'prior_schema', name)
    for key in ('version', 'estimator', 'min_count', 'conditioning', 'fallback', 'provenance'):
        audit.check(expected[key] == actual.get(key), 'prior_' + key, name)
    def bucket(left, right, label):
        audit.check(isinstance(right, dict) and set(right) == {'count', 'size'}, 'bucket_schema', label)
        right = right if isinstance(right, dict) else {}
        audit.check(type(right.get('count')) is int and left['count'] == right['count'], 'bucket_count', label)
        audit.numeric(left['size'], right.get('size'), label)
    bucket(expected['global'], actual.get('global'), name + '/global')
    for group in ('category', 'category_room_type'):
        found = actual.get(group, {})
        audit.check(set(found) == set(expected[group]), 'bucket_category_keys', name + '/' + group)
        for cat, value in expected[group].items():
            if group == 'category':
                bucket(value, found.get(cat), name + '/category/' + cat)
            else:
                types = found.get(cat, {})
                audit.check(set(types) == set(value), 'bucket_room_type_keys', name + '/' + cat)
                for rt, b in value.items():
                    bucket(b, types.get(rt), name + '/pair/' + cat + '/' + rt)


def audit_priors(audit, entries, observations, pool, oof, by_source):
    held = {e['component'] for e in entries.values() if e['source'] in HELDOUT}
    eligible = {u for u, e in entries.items() if not exclusion_reasons(e, held)}
    mapping = {u: entries[u]['component'] for u in eligible}
    folds = assign_folds(mapping)
    audit.check(len(eligible) == 124843, 'frozen_eligible_population', len(eligible))
    for u in eligible:
        by_source[entries[u]['source']]['prior_eligible'] += 1
    expected_pool = {'uids': sorted(eligible), 'uid_hashes': uid_provenance(eligible)[0],
                     'components': sorted(set(mapping.values())),
                     'component_hashes': sorted({H(c) for c in mapping.values()}),
                     'by_source': dict(Counter(entries[u]['source'] for u in eligible)),
                     'members': [{k: entries[u][k] for k in ('uid', 'source', 'component', 'prepared_sha256')}
                                 for u in sorted(eligible)], 'frozen_before_decoder_admission': True}
    audit.check(set(pool) == set(expected_pool), 'prior_pool_schema')
    for key, value in expected_pool.items():
        audit.check(pool.get(key) == value, 'prior_pool:' + key)
    audit.check(oof['fold_by_uid'] == folds, 'OOF_fold_assignment')
    audit.check(set(oof['priors']) == set(map(str, range(5))), 'OOF_five_prior_keys')
    component_uids = defaultdict(list)
    for uid, component in mapping.items():
        component_uids[component].append(uid)
    components = {H(c): {'fold': folds[us[0]], 'uid_hashes': uid_provenance(us)[0]}
                  for c, us in component_uids.items()}
    audit.summary['priors'] = {}
    for fold in (None, 0, 1, 2, 3, 4):
        subset = eligible if fold is None else {u for u in eligible if folds[u] != fold}
        excluded = eligible - subset
        expected = observations.fit(subset)
        hashes = sorted({H(mapping[u]) for u in subset})
        expected['provenance'].update(component_source='caller_connected_components',
                                      component_count=len(hashes), component_hashes=hashes, held_out_fold=fold)
        name = 'full' if fold is None else str(fold)
        actual = oof['full_prior'] if fold is None else oof['priors'][name]
        compare_prior(audit, expected, actual, name)
        audit.check(not set(actual['provenance']['fitted_uid_hashes']) & set(uid_provenance(excluded)[0]),
                    'OOF_excluded_UID_disjoint', name)
        audit.check(not set(actual['provenance']['component_hashes']) & {H(mapping[u]) for u in excluded},
                    'OOF_excluded_component_disjoint', name)
        p = expected['provenance']
        audit.summary['priors'][name] = {k: p[k] for k in ('room_count', 'object_count', 'component_count',
                                                        'fixed_excluded_count', 'fitted_uid_sha256')}
        audit.summary['priors'][name].update(held_out_rooms=len(excluded), held_out_uid_sha256=uid_provenance(excluded)[1],
                                             component_sha256=H(compact(hashes)))
        if fold is None:
            top = {**p, 'fold_count': 5, 'seed': SEED,
                   'fold_algorithm': 'sha256-json-seed-component-sort-round-robin-v1',
                   'hash_algorithm': 'sha256-utf8', 'components': components,
                   'training_contract': 'caller_prepared_bound_train_only; reject_explicit_nontrain_split',
                   'conditioning': 'category_room_type', 'fallback': ['category', 'global'],
                   'frozen_before_decoder_admission': True}
            for key in set(top) | set(oof['provenance']):
                audit.check(top.get(key) == oof['provenance'].get(key), 'OOF_provenance:' + key)
        audit.progress('prior', name=name, rooms=len(subset))
    return eligible, folds, held


def audit_admission(audit, main, manifest, entries, eligible, folds, held, by_source):
    seen, accepted, sources = set(), set(), defaultdict(Counter)
    for _, row, _ in audit.rows(main / 'common-admission.jsonl', manifest['files']['common-admission.jsonl']):
        uid = row['uid']
        audit.check(uid not in seen, 'admission_uid_unique', uid)
        seen.add(uid)
        e = entries[uid]
        reasons = exclusion_reasons(e, held)
        audit.check(e['split'] == 'train', 'admission_only_train', uid)
        audit.check((row['source'], row['component']) == (e['source'], e['component']), 'admission_identity', uid)
        audit.check(type(row['eligible']) is bool and row['eligible'] == (uid in eligible), 'admission_historical_eligibility', uid)
        audit.check(row['fold'] == folds.get(uid), 'admission_fold', uid)
        expected_accept = uid in eligible and row['joint']['qualified'] and row['placement']['qualified']
        audit.check(type(row['accepted']) is bool and row['accepted'] == expected_accept, 'admission_paired_qualification', uid)
        if uid not in eligible:
            audit.check(row['reasons'] == reasons, 'admission_exclusion_reasons', uid)
            audit.check(not row['joint']['qualified'] and not row['placement']['qualified'], 'ineligible_unqualified', uid)
        if row['accepted']:
            accepted.add(uid)
        stats = sources[e['source']]
        stats.update({'train_identities': 1, 'eligible': int(row['eligible']), 'accepted': int(row['accepted']),
                      'rejected': int(not row['accepted']), 'joint_qualified': int(row['joint']['qualified']),
                      'placement_qualified': int(row['placement']['qualified'])})
        for why in reasons:
            by_source[e['source']]['withheld:' + why] += 1
    train = {u for u, e in entries.items() if e['split'] == 'train'}
    audit.check(seen == train, 'admission_all_historical_train')
    audit.check(dict(sources) == manifest['by_source'], 'admission_source_counts')
    for source, values in sources.items():
        by_source[source].update({'admission_' + k: v for k, v in values.items()})
    dev = {u for u, e in entries.items() if e['split'] == 'dev'}
    audit.check(manifest.get('devlimit') is None, 'all_dev_retained_policy')
    counts = {'train_identities': len(train), 'dev_identities': len(dev),
              'test_identities': sum(e['split'] == 'test' for e in entries.values()),
              'prior_fit_rooms': len(eligible), 'eligible_candidates': len(eligible),
              'ineligible_train': len(train - eligible), 'common_decoder_train': len(accepted),
              'failed_candidates': len(eligible - accepted), 'dev_retained': len(dev),
              'joint_qualified': sum(s['joint_qualified'] for s in sources.values()),
              'placement_qualified': sum(s['placement_qualified'] for s in sources.values())}
    audit.check(manifest['dataset_counts'] == counts, 'main_dataset_counts')
    audit.check(manifest['counts'] == {'train': len(accepted), 'dev': len(dev)}, 'main_counts')
    exposure = manifest['prior_exposure']
    audit.check(exposure == {'may_be_broader_than_common_decoder_pool': True,
                            'extra_fit_uids': sorted(eligible - accepted), 'extra_fit_count': len(eligible - accepted),
                            'refit_after_decoder_admission': False}, 'prior_exposure_exact')
    split_rows = audit.read(main / 'split-manifest.json', manifest['files']['split-manifest.json'])
    split_seen = set()
    for row in split_rows:
        uid = row['uid']
        audit.check(uid not in split_seen, 'main_split_uid_unique', uid)
        split_seen.add(uid)
        e = entries[uid]
        expected = {k: e[k] for k in ('uid', 'source', 'split', 'component')}
        expected['fold'] = folds[uid] if e['split'] == 'train' else None
        audit.check(row == expected, 'main_split_record', uid)
    audit.check(split_seen == accepted | dev, 'main_split_complete_coverage')
    audit.summary['admission_counts'] = counts
    return accepted, dev


def remaining_payloads(audit, root, manifest, entries, populations=None):
    """Read every manifest payload, including the separate eval inventory."""
    for name, receipt in manifest['files'].items():
        path = safe_path(root, name)
        expected = receipt['sha256'] if isinstance(receipt, dict) else receipt
        if str(path) in audit.files and audit.files[str(path)].get('complete_scan'):
            audit.check(audit.files[str(path)]['sha256'] == expected, 'payload_referenced_sha256', str(path))
            continue
        if path.suffix == '.jsonl':
            seen = set()
            for _, row, _ in audit.rows(path, expected):
                uid = row.get('uid')
                audit.check(uid not in seen, 'payload_uid_unique:' + name, uid)
                seen.add(uid)
                if uid in entries:
                    e = entries[uid]
                    for key in ('source', 'split'):
                        if key in row:
                            audit.check(row[key] == e[key], 'payload_identity:' + name + ':' + key, uid)
                    if 'flags' in row:
                        audit.check(row['flags'] == e['training_flags'], 'payload_historical_flags', uid)
                else:
                    audit.check(False, 'payload_released_uid:' + name, uid)
            if populations is not None:
                split = 'test' if 'test' in populations else ('dev' if 'dev' in name else 'train')
                target = populations.get(split, set())
                if name == 'bound_rooms.jsonl':
                    target = set().union(*populations.values())
                audit.check(seen == target, 'payload_UID_coverage:' + name)
        elif path.suffix == '.json':
            value = audit.read(path, expected)
            if path.name == 'MANIFEST.json' and 'files' in value:
                remaining_payloads(audit, path.parent, value, entries)
        elif path.suffix in ('.txt', '.md', '.yaml', '.yml'):
            audit.blob(path, expected)
        else:
            raise ValueError('unexpected non-metadata payload (not loaded): ' + str(path))


def run(audit, root):
    runroot = root / 'runs/unknownsize-20261013'
    inputs = root / 'automation/unknownsize-20261013/inputs'
    snaps = root / 'automation/unknownsize-20261013/snapshots'
    main, corpus, evaluation = (runroot / n for n in ('main-data-v1', 'full-corpus-v2', 'eval-data-v1'))
    paths = {'release': inputs / 'release', 'corpus': corpus, 'main': main, 'eval': evaluation}
    manifests = {key: audit.read(path / 'MANIFEST.json', MANIFEST_PINS[key]) for key, path in paths.items()}
    for key, pin in MANIFEST_PINS.items():
        if audit.files[str(paths[key] / 'MANIFEST.json')]['sha256'] != pin:
            raise ValueError('frozen manifest pin mismatch: ' + key)
    snapshot = snaps / 'main-data-50a274bc4c46/fastfill'
    for name, expected in PINS.items():
        if audit.blob(snapshot / name, expected) != expected:
            raise ValueError('frozen source pin mismatch: ' + name)
    for which, manifest in manifests.items():
        for name, expected in manifest.get('implementation_sha256', {}).items():
            audit.check(PINS.get(name) == expected, 'implementation_inventory_pin', which + ':' + name)
            base = snaps / 'corpus-097e351ad6d3/fastfill' if which == 'corpus' else snapshot
            audit.blob(safe_path(base, name), expected)
    spec = importlib.util.spec_from_file_location('_lineage_frozen_normalizers', snapshot / 'comparison/size_prior.py')
    norm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norm)
    audit.summary['normalizers_source'] = str(Path(norm.__file__).resolve())
    rel, cor, man, ev = (manifests[k] for k in ('release', 'corpus', 'main', 'eval'))
    for child in (cor, man):
        audit.check(child['source_release_manifest_sha256'] == MANIFEST_PINS['release'], 'release_manifest_chain')
        audit.check(child['source_release_files'] == {s: rel['files'][s + '.jsonl'] for s in ('train', 'dev', 'test')}, 'release_files_chain')
        audit.check(child['ir_sha256'] == rel['ir_sha256'], 'IR_hash_chain')
    for child in (man, ev):
        audit.check(child['corpus_manifest_sha256'] == MANIFEST_PINS['corpus'], 'corpus_manifest_chain')
        audit.check(child['corpus_identity_manifest_sha256'] == cor['files']['identity-manifest.json'], 'corpus_identity_chain')
    audit.check(ev['training_manifest_sha256'] == MANIFEST_PINS['main'], 'eval_training_manifest_chain')
    audit.check(cor['scope'] == 'full_release_prepared_identity_closure', 'corpus_scope')
    audit.check(cor['prep_args'] == rel['args'], 'frozen_prep_args')
    by_source = defaultdict(Counter, {s: Counter() for s in SOURCES})
    entries = scan_release(audit, inputs / 'release', rel, by_source)
    dsu = DSU()
    raw_count = scan_raw(audit, inputs / 'ir', rel['ir_sha256'], entries, dsu, by_source)
    observations = scan_prepared(audit, corpus / 'prepared_rooms.jsonl', cor['files']['prepared_rooms.jsonl'], entries, dsu, norm, by_source)
    published = audit.read(corpus / 'identity-manifest.json', cor['files']['identity-manifest.json'])
    close_identities(audit, entries, published, dsu, cor)
    del published, dsu
    audit.check(raw_count == len(entries) + sum(v['excluded_unreleased'] for v in by_source.values()), 'raw_release_unreleased_reconciliation')
    for key, value in {'raw_rows': raw_count, 'release_rows': len(entries), 'prepared_rows': len(entries),
                       'unreleased_raw_rows': raw_count-len(entries)}.items():
        audit.check(cor['audit'][key] == value, 'corpus_audit_count:' + key)
    for key in ('rejects', 'duplicate_release_uids', 'duplicate_raw_uids', 'unmatched_uids', 'source_mismatches', 'cross_split_components'):
        audit.check(cor['audit'][key] == 0, 'corpus_recorded_failures:' + key)
    pool = audit.read(main / 'prior-fit-pool.json', man['files']['prior-fit-pool.json'])
    oof = audit.read(main / 'size-priors.json', man['files']['size-priors.json'])
    eligible, folds, held = audit_priors(audit, entries, observations, pool, oof, by_source)
    del observations, pool, oof
    accepted, dev = audit_admission(audit, main, man, entries, eligible, folds, held, by_source)
    test = {u for u, e in entries.items() if e['split'] == 'test'}
    audit.check(set(ev['expected_uids']) == test and len(ev['expected_uids']) == len(test), 'eval_exact_test_UIDs')
    audit.check(ev['prior_fit_count'] == len(eligible), 'eval_prior_fit_count')
    for key, path in paths.items():
        populations = {'train': accepted, 'dev': dev} if key == 'main' else {'test': test} if key == 'eval' else None
        remaining_payloads(audit, path, manifests[key], entries, populations)
    columns = sorted(set().union(*(set(c) for c in by_source.values())))
    audit.summary['by_source'] = {s: {k: by_source[s][k] for k in columns} for s in sorted(set(SOURCES) | set(by_source))}
    audit.summary['boundaries'] = ['No full prep replay; keys recomputed from hash-verified prepared rows.',
        'No geometry repair/qualification replay; admission booleans and lineage checked.',
        'No P/J target equality claim; independent input and geometry audits cover target contracts.',
        'Raw metadata profile is top-level room/object/fixed plus their meta dictionaries; five scalar values per relevant field.',
        'No assets opened. All declared release/corpus/main/eval payload bytes and main snapshot source inventory read.']
    for path, stamp in audit.stamps.items():
        audit.check(signature(Path(path)) == stamp, 'file_stable_at_completion', path)
    import numpy
    audit.summary['numpy'] = numpy.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('SLURM_JOB_ID required; submit via the parent CPU resource wrapper')
    root, out = args.root.resolve(), args.out.resolve()
    protected = [root / 'automation/unknownsize-20261013/inputs', root / 'automation/unknownsize-20261013/snapshots']
    protected += [root / 'runs/unknownsize-20261013' / n for n in ('main-data-v1', 'full-corpus-v2', 'eval-data-v1')]
    if any(out.is_relative_to(p) for p in protected):
        parser.error('--out must be outside frozen input/source/dataset directories')
    out.mkdir(exist_ok=False)
    audit = Audit(out)
    try:
        if sys.platform.startswith('linux'):
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min(24 * 1024**3, hard) if hard != resource.RLIM_INFINITY else 24 * 1024**3
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            audit.summary['address_space_limit_bytes'] = limit
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        audit.blob(Path(__file__).resolve())
        run(audit, root)
        failed = sum(v['failed'] for v in audit.checks.values())
        audit.save('FAIL' if failed else 'PASS')
        audit.progress('complete', failed_checks=failed, result=str(out / 'result.json'))
        return int(bool(failed))
    except Exception as exc:
        audit.check(False, 'fatal_error', {'type': type(exc).__name__, 'message': str(exc)})
        audit.save('ERROR', traceback.format_exc())
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
