"""Full frozen-release prepared corpus and identity closure, before OOF selection.

Retains historical train/dev/test membership and flags. Streams every raw IR
group/alias edge, but prepares only released UIDs, without canonical IDs/prompts.
Memory: identity/DSU metadata plus at most 2 * workers pending room payloads.
Output order: sorted IR filename, then line order; identity entries sort by UID.
identity-manifest.json is a JSON array: UID/source/original_group/component/split,
historical training_flags, regenerated prepared_flags, keys and per-row hashes.
MANIFEST.json hashes both artifacts, all consumed sources, and implementation
dependencies. Counts come from the frozen manifest (v3.2: 160964 UIDs, 18 IRs).
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import copy
import ctypes
import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace

from fastfill.comparison.data import Components, json_dump, sha256

SPLITS = ('train', 'dev', 'test')
CODE = ('__init__.py', 'comparison/__init__.py', 'comparison/corpus.py',
        'comparison/data.py', 'build.py', 'split.py', 'scene.py', 'anchors.py',
        'validate.py', 'requirements.txt')


class CorpusError(ValueError):
    """Fatal reconciliation error; counts remain inspectable without publication."""
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = dict(audit)


def _fail(audit, counter, message, count=1):
    audit[counter] += count
    raise CorpusError(message, audit)


def _text(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f'invalid {field}: expected nonempty string')
    return value


def _load_manifest(path):
    expected = sha256(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('release manifest consumed hash mismatch')
    return json.loads(data), expected


def _inventory(ir, hashes):
    if not hashes or any(Path(n).name != n or not n.endswith('.jsonl') for n in hashes):
        raise ValueError('invalid IR manifest filename inventory')
    actual = {p.name for p in ir.iterdir() if p.suffix == '.jsonl'}
    if actual != set(hashes) or any(not (ir / n).is_file() for n in hashes):
        raise ValueError('IR file inventory differs from frozen manifest')


def _rows(path, expected):
    """Hash the bytes consumed, independently of both preflight and final hashes."""
    if sha256(path) != expected:
        raise ValueError(f'input hash mismatch: {path}')
    consumed = hashlib.sha256()
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            consumed.update(line)
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'expected JSON object: {path}:{number}')
            yield number, row, hashlib.sha256(line).hexdigest()
    if consumed.hexdigest() != expected:
        raise ValueError(f'consumed hash changed: {path}')


def _release_index(release, manifest, audit):
    identities, counts = {}, {}
    for split in SPLITS:
        name = split + '.jsonl'
        expected = manifest['files'][name]
        count = 0
        for number, row, digest in _rows(release / name, expected['sha256']):
            uid = _text(row['uid'], 'release UID')
            if uid in identities:
                _fail(audit, 'duplicate_release_uids', f'duplicate release UID: {uid}')
            flags = row.get('flags')
            if flags is None:
                flags = {}
            if not isinstance(flags, dict):
                raise ValueError(f'invalid training flags: {uid}')
            identities[uid] = {'uid': uid, 'source': _text(row['source'], 'source'),
                               'split': split, 'training_flags': flags,
                               'release_line': number, 'release_row_sha256': digest}
            count += 1
            audit['release_rows'] += 1
        if count != expected['lines']:
            raise ValueError(f'release line count mismatch: {name}')
        counts[split] = count
    if not identities:
        raise ValueError('empty released corpus')
    return identities, counts


def _raw_rows(ir, hashes, identities, components, audit):
    seen = set()
    for filename, expected in sorted(hashes.items()):
        for number, raw, digest in _rows(ir / filename, expected):
            uid = _text(raw['uid'], 'raw UID')
            audit['raw_rows'] += 1
            if uid in seen:
                _fail(audit, 'duplicate_raw_uids', f'duplicate raw UID: {uid}')
            seen.add(uid)
            group = 'g:' + _text(raw['group'], 'raw group')
            components.find(group)
            aliases = raw.get('meta', {}).get('group_aliases')
            if aliases is None:
                aliases = []
            if not isinstance(aliases, list):
                raise ValueError(f'invalid group aliases: {uid}')
            for alias in aliases:
                components.union(group, 'a:' + _text(alias, 'group alias'))
            if uid not in identities:
                audit['unreleased_raw_rows'] += 1
                continue
            if raw['source'] != identities[uid]['source']:
                _fail(audit, 'source_mismatches', f'release/raw source mismatch: {uid}')
            yield {'uid': uid, 'original_group': raw['group'], 'raw_file': filename,
                   'raw_line': number, 'raw_sha256': digest}, raw


def _prepare(raw, args):
    from fastfill.build import prep
    from fastfill.split import content_key, layout_key

    prepared, reason = prep(copy.deepcopy(raw), SimpleNamespace(**args))
    if reason or prepared is None:
        return None, reason or 'missing prepared room', None, None
    if any(prepared[k] != raw[k] for k in ('uid', 'source', 'group')):
        raise ValueError(f'prep changed original identity: {raw["uid"]}')
    return prepared, None, content_key(prepared), layout_key(prepared)


def _ordered(rows, args, workers):
    if workers == 1:
        for metadata, raw in rows:
            yield metadata, _prepare(raw, args)
        return
    # Executor.map eagerly queues its iterable on older Python; bound it explicitly.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        for metadata, raw in rows:
            pending.append((metadata, pool.submit(_prepare, raw, args)))
            if len(pending) >= 2 * workers:
                metadata, result = pending.popleft()
                yield metadata, result.result()
        while pending:
            metadata, result = pending.popleft()
            yield metadata, result.result()


def _write_prepared(path, rows, identities, components, args, workers, audit):
    with path.open('wb') as stream:
        for metadata, (prepared, reason, content, layout) in _ordered(rows, args, workers):
            uid = metadata['uid']
            if reason:
                _fail(audit, 'rejects', f'released UID rejected by prep: {uid}: {reason}')
            entry = identities[uid]
            group = 'g:' + metadata['original_group']
            for prefix, key in (('c:', content), ('l:', layout)):
                if key is not None:
                    components.union(group, prefix + key)
            data = (json.dumps({**prepared, 'split': entry['split']},
                               ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
            stream.write(data)
            identities[uid] = {**entry, **metadata, 'content_key': content, 'layout_key': layout,
                               'prepared_flags': prepared['meta']['flags'],
                               'prepared_sha256': hashlib.sha256(data).hexdigest()}
            audit['prepared_rows'] += 1


def _close(identities, components, audit):
    missing = sum('original_group' not in entry for entry in identities.values())
    if missing:
        _fail(audit, 'unmatched_uids', f'unmatched released UIDs: {missing}', missing)
    owners = {}
    for uid, entry in identities.items():
        component = components.find('g:' + entry['original_group'])
        if component in owners and owners[component] != entry['split']:
            _fail(audit, 'cross_split_components', f'cross-split component: {component}: '
                  f'{owners[component]} / {entry["split"]} (UID {uid})')
        owners[component] = entry['split']
        identities[uid] = {**entry, 'component': component}
    if audit['release_rows'] != audit['prepared_rows']:
        raise CorpusError('release/prepared row reconciliation failed', audit)
    if audit['raw_rows'] != audit['prepared_rows'] + audit['unreleased_raw_rows']:
        raise CorpusError('raw/prepared/unreleased row reconciliation failed', audit)
    audit['components'] = len(owners)


def _write_identities(path, identities):
    with path.open('w', encoding='utf-8') as stream:
        stream.write('[\n')
        for index, uid in enumerate(sorted(identities)):
            if index:
                stream.write(',\n')
            stream.write(json.dumps(identities[uid], ensure_ascii=False, allow_nan=False))
        stream.write('\n]\n')


def _dependencies():
    import numpy
    import shapely

    return {'python': platform.python_version(), 'numpy': numpy.__version__,
            'shapely': shapely.__version__, 'geos': shapely.geos_version_string}


def _verify(release, ir, manifest, manifest_hash, code_root, code_hashes):
    _inventory(ir, manifest['ir_sha256'])
    inputs = [(release / 'MANIFEST.json', manifest_hash)]
    inputs += [(release / (s + '.jsonl'), manifest['files'][s + '.jsonl']['sha256']) for s in SPLITS]
    inputs += [(ir / n, h) for n, h in manifest['ir_sha256'].items()]
    inputs += [(code_root / n, h) for n, h in code_hashes.items()]
    for path, expected in inputs:
        if sha256(path) != expected:
            raise ValueError(f'input or implementation hash changed during build: {path}')


def _publish(staging, destination):
    """Never replace output. On Lustre, publish the completion manifest last.

    Some Lustre servers reject RENAME_NOREPLACE with EINVAL. The fallback
    exclusively reserves the directory and hardlinks immutable staged files,
    each with no-replace semantics. Consumers require MANIFEST.json plus hashes;
    interrupted publication leaves a visible, explicitly incomplete directory.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    source, target = os.fsencode(staging), os.fsencode(destination)
    if sys.platform == 'darwin':
        result = libc.renamex_np(source, target, 4)  # RENAME_EXCL
    elif sys.platform.startswith('linux'):
        result = libc.renameat2(-100, source, -100, target, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise OSError('atomic no-replace directory publication requires Linux or macOS')
    if result:
        error = ctypes.get_errno()
        if error in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            return _publish_manifest_last(Path(staging), Path(destination))
        raise OSError(error, os.strerror(error), str(destination))


def _publish_manifest_last(staging, destination):
    if not (staging / 'MANIFEST.json').is_file():
        raise ValueError('publication requires a completion manifest')
    destination.mkdir(exist_ok=False)
    paths = sorted(staging.rglob('*'), key=lambda p: (len(p.parts), str(p)))
    for path in paths:
        if path.is_symlink():
            raise ValueError('staged artifacts must not be symlinks')
        target = destination / path.relative_to(staging)
        if path.is_dir():
            target.mkdir(exist_ok=False)
        elif path != staging / 'MANIFEST.json':
            os.link(path, target)
    os.link(staging / 'MANIFEST.json', destination / 'MANIFEST.json')


def build(release: Path, ir: Path, out: Path, workers=1):
    """Build all manifest-listed released identities; abort without partial output."""
    release, ir, out = Path(release), Path(ir), Path(out)
    if os.path.lexists(out):
        raise FileExistsError(f'refusing existing output: {out}')
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {name: sha256(code_root / name) for name in CODE}
    manifest, manifest_hash = _load_manifest(release / 'MANIFEST.json')
    _inventory(ir, manifest['ir_sha256'])
    audit = dict.fromkeys(('release_rows', 'raw_rows', 'prepared_rows', 'unreleased_raw_rows',
                          'rejects', 'duplicate_release_uids', 'duplicate_raw_uids',
                          'unmatched_uids', 'source_mismatches', 'cross_split_components'), 0)
    identities, counts = _release_index(release, manifest, audit)
    components = Components()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temp:
        dest = Path(temp)
        rows = _raw_rows(ir, manifest['ir_sha256'], identities, components, audit)
        _write_prepared(dest / 'prepared_rooms.jsonl', rows, identities, components,
                        manifest['args'], workers, audit)
        _close(identities, components, audit)
        _write_identities(dest / 'identity-manifest.json', identities)
        result = {
            'version': 1, 'scope': 'full_release_prepared_identity_closure',
            'source_release_manifest_sha256': manifest_hash,
            'source_release_files': {s: manifest['files'][s + '.jsonl'] for s in SPLITS},
            'ir_sha256': manifest['ir_sha256'], 'implementation_sha256': code_hashes,
            'dependencies': _dependencies(), 'prep_args': manifest['args'],
            'counts': counts, 'audit': audit,
            'by_source': dict(sorted(Counter(r['source'] for r in identities.values()).items())),
            'split_policy': 'historical_release_no_reassignment',
            'training_flags_policy': 'historical_release_labels_only_no_filtering',
            'component_policy': 'all_raw_group_alias_edges_and_all_released_prepared_content_layout_keys',
            'id_policy': 'original_prep_ids_no_canonicalization',
            'row_order': 'sorted_IR_filename_then_line',
            'files': {n: sha256(dest / n) for n in ('prepared_rooms.jsonl', 'identity-manifest.json')},
        }
        json_dump(dest / 'MANIFEST.json', result)
        _verify(release, ir, manifest, manifest_hash, code_root, code_hashes)
        _publish(dest, out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'ir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    result = build(args.release, args.ir, args.out, args.workers)
    print(json.dumps({'counts': result['counts'], 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
