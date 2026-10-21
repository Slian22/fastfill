"""Compare full before/after adapters, preserving historical bytes of unchanged rows."""
from collections import Counter
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
ROOT = Path('/Volumes/harddisk/fastfillout/unknownsize-source-correction-20261013')
SOURCES = ('Scan2CAD', 'SpatialGen', 'IL3D_3dfront')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def diff(a, b, path=''):
    if a == b:
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(a.keys() | b.keys())
                for d in diff(a.get(k), b.get(k), path + '/' + k)]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, path + '/' + str(i))]
    return [{'path': path, 'before': a, 'after': b}]


def source_delta(source, ledger):
    filename = source + '.jsonl'
    paths = [PROJECT / '.release/v3.2/ir' / filename, ROOT / 'baseline-ir' / filename,
             ROOT / 'corrected-ir-001' / filename]
    counts, fields = Counter(), Counter()
    max_numeric = 0.0
    seen = set()
    with paths[0].open('rb') as old, paths[1].open('rb') as before, paths[2].open('rb') as after, \
            (ROOT / 'ir-candidate-001' / filename).open('xb') as output:
        for old_bytes, before_bytes, after_bytes in zip(old, before, after, strict=True):
            historical, a, b = map(json.loads, (old_bytes, before_bytes, after_bytes))
            uid = a['uid']
            if uid in seen or historical['uid'] != uid or b['uid'] != uid:
                raise ValueError('UID/order mismatch: ' + uid)
            seen.add(uid)
            counts['rows'] += 1
            counts['baseline_vs_historical_changed'] += historical != a
            output.write(old_bytes if a == b else after_bytes)
            if a == b:
                counts['unchanged_rows_historical_bytes_retained'] += 1
                continue
            counts['changed_rows'] += 1
            aa, bb = ({o['id']: o for o in r['objects']} for r in (a, b))
            if len(aa) != len(a['objects']) or len(bb) != len(b['objects']):
                raise ValueError('duplicate object ID: ' + uid)
            added, removed = sorted(bb.keys() - aa.keys()), sorted(aa.keys() - bb.keys())
            counts['objects_added'] += len(added)
            counts['objects_removed'] += len(removed)
            objects = []
            for oid in sorted(aa.keys() & bb.keys()):
                changes = diff(aa[oid], bb[oid])
                if not changes:
                    continue
                counts['objects_changed'] += 1
                numeric = [abs(d['before'] - d['after']) for d in changes
                           if type(d['before']) in (int, float) and type(d['after']) in (int, float)]
                if numeric:
                    maximum = max(numeric)
                    max_numeric = max(max_numeric, maximum)
                    counts['objects_max_delta_le_1e-9' if maximum <= 1e-9 else
                           'objects_max_delta_gt_1e-9'] += 1
                fields.update(d['path'] for d in changes)
                objects.append({'id': oid, 'category': aa[oid]['category'], 'changes': changes})
            room_changes = diff({k: v for k, v in a.items() if k != 'objects'},
                                {k: v for k, v in b.items() if k != 'objects'})
            entry = {'source': source, 'uid': uid, 'added_objects': [bb[k] for k in added],
                     'removed_objects': [aa[k] for k in removed], 'changed_objects': objects,
                     'object_order_changed': list(aa) != list(bb), 'room_changes': room_changes,
                     'before_row_sha256': hashlib.sha256(before_bytes).hexdigest(),
                     'after_row_sha256': hashlib.sha256(after_bytes).hexdigest()}
            ledger.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
    return {'counts': dict(counts), 'object_field_differences': dict(fields),
            'max_numeric_object_field_delta': max_numeric,
            'inputs': [{'path': str(p), 'sha256': sha(p)} for p in paths]}


def main():
    with (ROOT / 'source-deltas.jsonl').open('x') as ledger:
        summaries = {s: source_delta(s, ledger) for s in SOURCES}
    files = {p.name: {'sha256': sha(p), 'rows': sum(1 for _ in p.open('rb'))}
             for p in sorted((ROOT / 'ir-candidate-001').glob('*.jsonl'))}
    if len(files) != 18:
        raise ValueError('expected all 18 sources')
    result = {'scope': 'Same-raw before/after adapter differences; not prepared or model effects.',
              'by_source': summaries, 'files': files,
              'parent_manifest_sha256': sha(PROJECT / '.release/v3.2/data/v3.2/MANIFEST.json'),
              'corrected_code_manifest_sha256': sha(ROOT / 'corrected-code-001/CORRECTION-MANIFEST.json'),
              'row_reuse_policy': 'Exact semantic before==after: keep historical line bytes; otherwise new corrected bytes.',
              'source_delta_ledger_sha256': sha(ROOT / 'source-deltas.jsonl')}
    (ROOT / 'IR-MANIFEST.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({s: r['counts'] for s, r in summaries.items()}))


if __name__ == '__main__':
    main()
