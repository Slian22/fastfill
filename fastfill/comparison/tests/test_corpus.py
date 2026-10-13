"""Synthetic full-release closure tests; never open the real release or assets."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fastfill import build as preparation
from fastfill.comparison import corpus
from fastfill.split import content_key, layout_key


ARGS = dict(boundary_types=['polygon', 'hull'], source_anchors={},
            anchors=['floor', 'object'], min_objects=1, max_vertices=0,
            oob_tol=0.1, hidden_max=0.3, reject_flagged=[])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def room(uid, *, width=12, scale=1, offset=0, aliases=()):
    return {'uid': uid, 'source': 'Synthetic', 'group': 'house:' + uid,
            'boundary_type': 'polygon',
            'boundary': [[0, 0], [width, 0], [width, 10], [0, 10]], 'height': 4,
            'meta': {'front_known': True, 'group_aliases': list(aliases)},
            'objects': [dict(id=f'{uid}/original-{i}', category='chair',
                             size=[scale * (0.4 + i / 10), 0.6, 1],
                             pos=[2 + i * 3 + offset, 2, 0], yaw=0,
                             anchor='floor', parent=None) for i in range(3)]}


def write_rows(path, rows):
    data = ''.join(json.dumps(r) + '\n' for r in rows).encode()
    path.write_bytes(data)
    return {'sha256': digest(data), 'lines': len(rows)}


def fixture(root, rooms=None, splits=None, flags=None):
    root.mkdir(parents=True, exist_ok=True)
    release, ir, out = root / 'release', root / 'ir', root / 'output'
    release.mkdir()
    ir.mkdir()
    rooms = rooms if rooms is not None else [room('A'), room('D', width=14, scale=2),
                                           room('T', width=16, scale=3)]
    splits = splits or {'train': ['A'], 'dev': ['D'], 'test': ['T']}
    flags = flags or {}
    manifest = {'args': ARGS, 'files': {}, 'ir_sha256': {}}
    # Exercise the same 18-file traversal shape as v3.2, without real data.
    for i in range(18):
        name = f'part-{i:02}.jsonl'
        manifest['ir_sha256'][name] = write_rows(ir / name, rooms[i::18])['sha256']
    for split in ('train', 'dev', 'test'):
        rows = [dict(uid=u, source='Synthetic', flags=flags.get(u, {}),
                     messages=['historical canonical labels must not be consumed'])
                for u in splits.get(split, [])]
        manifest['files'][split + '.jsonl'] = write_rows(release / (split + '.jsonl'), rows)
    (release / 'MANIFEST.json').write_text(json.dumps(manifest))
    return release, ir, out, manifest


def rewrite_manifest(release, manifest):
    (release / 'MANIFEST.json').write_text(json.dumps(manifest))


def outputs(out):
    rooms = [json.loads(line) for line in (out / 'prepared_rooms.jsonl').read_text().splitlines()]
    identities = json.loads((out / 'identity-manifest.json').read_text())
    return rooms, {r['uid']: r for r in identities}


def assert_clean(out):
    assert not out.exists()
    assert not list(out.parent.glob('.' + out.name + '*'))


def test_all_release_labels_original_ids_hashes_and_audit(tmp_path, monkeypatch):
    release, ir, out, source = fixture(tmp_path, flags={
        'A': {'oob_objects': 1}, 'D': {'fixed_collision': 1},
        'T': {'overlapping_furniture': True}})
    before = {p: p.read_bytes() for base in (release, ir) for p in base.iterdir()}
    real_prep = preparation.prep
    calls = []

    def checked_prep(raw, args):
        saved = copy.deepcopy(raw)
        result = real_prep(raw, args)
        assert raw == saved
        calls.append(raw['uid'])
        return result

    def forbidden(*args, **kwargs):
        pytest.fail('canonicalization / prompt construction is forbidden')

    monkeypatch.setattr(preparation, 'prep', checked_prep)
    monkeypatch.setattr(preparation, 'canonical', forbidden)
    monkeypatch.setattr(preparation, 'messages', forbidden)
    monkeypatch.setattr('fastfill.scene.canonical', forbidden)
    monkeypatch.setattr('fastfill.scene.messages', forbidden)
    manifest = corpus.build(release, ir, out)
    prepared, identities = outputs(out)
    assert calls == ['A', 'D', 'T']
    assert manifest == json.loads((out / 'MANIFEST.json').read_text())
    assert manifest['counts'] == {'train': 1, 'dev': 1, 'test': 1}
    assert manifest['audit']['release_rows'] == manifest['audit']['prepared_rows'] == 3
    assert manifest['audit']['raw_rows'] == 3
    for key in ('rejects', 'duplicate_release_uids', 'duplicate_raw_uids', 'unmatched_uids'):
        assert manifest['audit'][key] == 0
    for r in prepared:
        entry = identities[r['uid']]
        assert r['split'] == entry['split']
        assert entry['source'] == r['source']
        assert entry['original_group'] == r['group'] == 'house:' + r['uid']
        assert [o['id'] for o in r['objects']] == [f'{r["uid"]}/original-{i}' for i in range(3)]
        assert entry['training_flags']
        assert entry['content_key'] == content_key(r)
        assert entry['layout_key'] == layout_key(r)
        assert 'messages' not in r
    for line in (out / 'prepared_rooms.jsonl').read_bytes().splitlines(keepends=True):
        assert identities[json.loads(line)['uid']]['prepared_sha256'] == digest(line)
    for name, expected in manifest['files'].items():
        assert digest((out / name).read_bytes()) == expected
    assert manifest['ir_sha256'] == source['ir_sha256']
    assert manifest['source_release_manifest_sha256'] == digest(before[release / 'MANIFEST.json'])
    assert set(manifest['source_release_files']) == {'train', 'dev', 'test'}
    for name in ('comparison/corpus.py', 'comparison/data.py', 'build.py', 'split.py',
                 'scene.py', 'anchors.py', 'validate.py'):
        assert manifest['implementation_sha256'][name] == digest((Path('fastfill') / name).read_bytes())
    assert {'python', 'numpy', 'shapely', 'geos'} <= manifest['dependencies'].keys()
    assert all(p.read_bytes() == data for p, data in before.items())


def bridge_rooms():
    # A--content--U--layout--B; A and B have neither key in common.
    return [room('A'), room('U', offset=1), room('B', offset=1, scale=2)]


def test_flagged_unselected_released_room_closes_mixed_edges(tmp_path):
    rooms = bridge_rooms()
    prepped = [preparation.prep(r, SimpleNamespace(**ARGS))[0] for r in rooms]
    assert content_key(prepped[0]) == content_key(prepped[1]) != content_key(prepped[2])
    assert layout_key(prepped[0]) != layout_key(prepped[1]) == layout_key(prepped[2])
    release, ir, out, _ = fixture(tmp_path, rooms, {'train': ['A', 'B', 'U']},
                                   flags={'U': {'fixed_collision': 1}})
    corpus.build(release, ir, out)
    prepared, identities = outputs(out)
    assert len(prepared) == 3
    assert len({r['component'] for r in identities.values()}) == 1
    assert identities['U']['training_flags'] == {'fixed_collision': 1}


@pytest.mark.parametrize('edge', ['mixed', 'alias', 'group'])
def test_cross_split_component_is_fatal(tmp_path, edge):
    rooms = bridge_rooms()
    if edge != 'mixed':
        rooms = [room('A'), room('B', width=20, scale=3)]
        if edge == 'alias':
            rooms = [{**r, 'meta': {**r['meta'], 'group_aliases': ['visit']}} for r in rooms]
        else:
            rooms = [{**r, 'group': 'shared'} for r in rooms]
    release, ir, out, _ = fixture(tmp_path, rooms, {
        'train': ['A'] + (['U'] if edge == 'mixed' else []), 'test': ['B']})
    with pytest.raises(ValueError, match='cross-split component'):
        corpus.build(release, ir, out)
    assert_clean(out)


def test_unreleased_raw_alias_bridge_is_used_but_never_prepared(tmp_path, monkeypatch):
    rooms = [room('A', aliases=['left']), room('B', width=20, scale=3, aliases=['right']),
             room('unreleased', aliases=['left', 'right'])]
    rooms[-1]['meta']['front_known'] = False
    release, ir, out, _ = fixture(tmp_path, rooms, {'train': ['A', 'B']})
    real_prep = preparation.prep
    def guard(raw, args):
        assert raw['uid'] != 'unreleased'
        return real_prep(raw, args)
    monkeypatch.setattr(preparation, 'prep', guard)
    manifest = corpus.build(release, ir, out)
    _, identities = outputs(out)
    assert identities['A']['component'] == identities['B']['component']
    assert manifest['audit']['raw_rows'] == 3
    assert manifest['audit']['unreleased_raw_rows'] == 1


@pytest.mark.parametrize('failure', ['duplicate_release', 'duplicate_cross_split', 'duplicate_raw',
                                    'duplicate_unreleased', 'unmatched', 'reject', 'source', 'lines'])
def test_bad_identity_or_reconciliation_aborts(tmp_path, failure):
    release, ir, out, manifest = fixture(tmp_path)
    if failure.startswith('duplicate_') and failure in ('duplicate_release', 'duplicate_cross_split'):
        name = 'train.jsonl' if failure == 'duplicate_release' else 'dev.jsonl'
        rows = [json.loads(x) for x in (release / name).read_text().splitlines()]
        rows += [dict(uid='A', source='Synthetic', flags={})]
        manifest['files'][name] = write_rows(release / name, rows)
    elif failure in ('duplicate_raw', 'duplicate_unreleased'):
        rows = [room('A')] if failure == 'duplicate_raw' else [room('unused')] * 2
        manifest['ir_sha256']['part-17.jsonl'] = write_rows(ir / 'part-17.jsonl', rows)['sha256']
    elif failure in ('unmatched', 'reject', 'source'):
        raw = room('A')
        if failure == 'reject':
            raw['meta']['front_known'] = False
        elif failure == 'source':
            raw['source'] = 'unexpected'
        manifest['ir_sha256']['part-00.jsonl'] = write_rows(
            ir / 'part-00.jsonl', [] if failure == 'unmatched' else [raw])['sha256']
    else:
        manifest['files']['train.jsonl']['lines'] += 1
    rewrite_manifest(release, manifest)
    pattern = {'unmatched': 'unmatched|unresolved', 'reject': 'reject',
               'source': 'source', 'lines': 'count'}.get(failure, 'duplicate')
    with pytest.raises(ValueError, match=pattern) as error:
        corpus.build(release, ir, out)
    counter = {'duplicate_release': 'duplicate_release_uids',
               'duplicate_cross_split': 'duplicate_release_uids',
               'duplicate_raw': 'duplicate_raw_uids', 'duplicate_unreleased': 'duplicate_raw_uids',
               'unmatched': 'unmatched_uids', 'reject': 'rejects', 'source': 'source_mismatches'}
    if failure in counter:
        assert error.value.audit[counter[failure]] == 1
    assert_clean(out)


@pytest.mark.parametrize('target', ['release', 'ir', 'manifest'])
def test_consumed_hash_rejects_swap_even_when_disk_restored(tmp_path, monkeypatch, target):
    release, ir, out, _ = fixture(tmp_path)
    path = {'release': release / 'train.jsonl', 'ir': ir / 'part-00.jsonl',
            'manifest': release / 'MANIFEST.json'}[target]
    original = path.read_bytes()
    real_open = Path.open
    armed = True
    def swapped_open(self, *args, **kwargs):
        nonlocal armed
        if self == path and armed:
            armed = False
            # Semantically identical JSON, different consumed bytes. Disk never changes.
            return io.BytesIO(b' ' + original)
        return real_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', swapped_open)
    with pytest.raises(ValueError, match='hash|changed'):
        corpus.build(release, ir, out)
    assert path.read_bytes() == original
    assert_clean(out)


@pytest.mark.parametrize('change', ['missing', 'extra', 'unsafe_name', 'bad_hash'])
def test_frozen_ir_inventory_and_hashes(tmp_path, change):
    release, ir, out, manifest = fixture(tmp_path)
    if change == 'missing':
        (ir / 'part-17.jsonl').unlink()
    elif change == 'extra':
        (ir / 'extra.jsonl').write_text('')
    elif change == 'unsafe_name':
        manifest['ir_sha256']['../outside.jsonl'] = '0' * 64
        rewrite_manifest(release, manifest)
    else:
        (ir / 'part-17.jsonl').write_text('\n')
    with pytest.raises(ValueError, match='IR|hash|filename'):
        corpus.build(release, ir, out)
    assert_clean(out)


@pytest.mark.parametrize('kind', ['directory', 'file', 'dangling_symlink', 'race'])
def test_atomic_output_never_replaces_existing_path(tmp_path, monkeypatch, kind):
    release, ir, out, _ = fixture(tmp_path)
    if kind == 'race':
        publish = corpus._publish
        def race(staging, destination):
            destination.mkdir()
            return publish(staging, destination)
        monkeypatch.setattr(corpus, '_publish', race)
    elif kind == 'directory':
        out.mkdir()
    elif kind == 'file':
        out.write_text('existing')
    else:
        out.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        corpus.build(release, ir, out)
    assert out.exists() or out.is_symlink()
    if kind == 'file':
        assert out.read_text() == 'existing'
    assert not list(tmp_path.glob('.output*'))


def test_worker_one_two_and_repeat_are_byte_identical(tmp_path):
    rooms = [room(f'R{i}', width=20 + i, scale=1 + i / 3) for i in range(7)]
    release, ir, first, _ = fixture(tmp_path, rooms, {
        'train': ['R0', 'R2', 'R4'], 'dev': ['R1', 'R3'], 'test': ['R5', 'R6']})
    manifests = [corpus.build(release, ir, out, workers=workers)
                 for out, workers in [(first, 1), (tmp_path / 'parallel', 2), (tmp_path / 'repeat', 1)]]
    assert manifests[0] == manifests[1] == manifests[2]
    for name in [*manifests[0]['files'], 'MANIFEST.json']:
        assert (first / name).read_bytes() == (tmp_path / 'parallel' / name).read_bytes()


def test_unsupported_noreplace_filesystem_publishes_manifest_last(tmp_path, monkeypatch):
    import errno
    from types import SimpleNamespace
    staging, output = tmp_path / 'staging', tmp_path / 'output'
    (staging / 'arm').mkdir(parents=True)
    (staging / 'arm' / 'x.jsonl').write_text('payload')
    (staging / 'MANIFEST.json').write_text('{}')
    monkeypatch.setattr(corpus.ctypes, 'CDLL', lambda *a, **k: SimpleNamespace(
        renamex_np=lambda *a: -1, renameat2=lambda *a: -1))
    monkeypatch.setattr(corpus.ctypes, 'get_errno', lambda: errno.EINVAL)
    linked, real_link = [], corpus.os.link
    def link(src, dst):
        linked.append(Path(dst).relative_to(output).as_posix())
        assert not (output / 'MANIFEST.json').exists()
        real_link(src, dst)
    monkeypatch.setattr(corpus.os, 'link', link)
    corpus._publish(staging, output)
    assert linked == ['arm/x.jsonl', 'MANIFEST.json']
    assert (output / 'arm/x.jsonl').read_text() == 'payload'
    with pytest.raises(FileExistsError):
        corpus._publish(staging, output)


def test_failed_portable_publish_cannot_expose_completion_manifest(tmp_path, monkeypatch):
    import errno
    from types import SimpleNamespace
    staging = tmp_path / 'staging'
    staging.mkdir()
    (staging / 'payload').write_text('payload')
    (staging / 'MANIFEST.json').write_text('{}')
    monkeypatch.setattr(corpus.ctypes, 'CDLL', lambda *a, **k: SimpleNamespace(
        renamex_np=lambda *a: -1, renameat2=lambda *a: -1))
    monkeypatch.setattr(corpus.ctypes, 'get_errno', lambda: errno.EOPNOTSUPP)
    def broken(*args):
        raise OSError('link failure')
    monkeypatch.setattr(corpus.os, 'link', broken)
    output = tmp_path / 'output'
    with pytest.raises(OSError, match='link failure'):
        corpus._publish(staging, output)
    assert output.is_dir()
    assert not (output / 'MANIFEST.json').exists()


@pytest.mark.parametrize('workers', [0, -1, True, 1.5])
def test_invalid_workers(tmp_path, workers):
    release, ir, out, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='workers'):
        corpus.build(release, ir, out, workers=workers)
    assert_clean(out)


def test_cli_builds_all_three_splits(tmp_path):
    release, ir, out, _ = fixture(tmp_path)
    result = subprocess.run([sys.executable, '-m', 'fastfill.comparison.corpus',
                             '--release', str(release), '--ir', str(ir), '--out', str(out),
                             '--workers', '2'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['counts'] == {'train': 1, 'dev': 1, 'test': 1}
    assert len(outputs(out)[0]) == 3


@pytest.mark.parametrize('field,value', [('flags', False), ('flags', []), ('flags', 'x'),
                                         ('aliases', False), ('aliases', ''), ('aliases', 'x'),
                                         ('uid', ''), ('group', '')])
def test_malformed_identity_metadata_cannot_silently_drop_labels_or_edges(tmp_path, field, value):
    release, ir, out, manifest = fixture(tmp_path)
    if field == 'flags':
        manifest['files']['train.jsonl'] = write_rows(release / 'train.jsonl', [
            {'uid': 'A', 'source': 'Synthetic', 'flags': value}])
    else:
        raw = room('A')
        if field == 'aliases':
            raw['meta']['group_aliases'] = value
        else:
            raw[field] = value
        manifest['ir_sha256']['part-00.jsonl'] = write_rows(ir / 'part-00.jsonl', [raw])['sha256']
    rewrite_manifest(release, manifest)
    with pytest.raises(ValueError, match='invalid'):
        corpus.build(release, ir, out)
    assert_clean(out)


@pytest.mark.parametrize('target', ['release', 'ir', 'manifest', 'implementation'])
def test_late_drift_prevents_publication(tmp_path, monkeypatch, target):
    release, ir, out, _ = fixture(tmp_path)
    write = corpus._write_identities
    hashed = corpus.sha256
    late = False
    def change_after_read(path, identities):
        nonlocal late
        write(path, identities)
        late = True
        if target != 'implementation':
            changed = {'release': release / 'train.jsonl', 'ir': ir / 'part-17.jsonl',
                       'manifest': release / 'MANIFEST.json'}[target]
            changed.write_bytes(changed.read_bytes() + b'\n')
    def changed_code_hash(path):
        if late and target == 'implementation' and Path(path).name == 'build.py':
            return '0' * 64
        return hashed(path)
    monkeypatch.setattr(corpus, '_write_identities', change_after_read)
    monkeypatch.setattr(corpus, 'sha256', changed_code_hash)
    with pytest.raises(ValueError, match='hash changed'):
        corpus.build(release, ir, out)
    assert_clean(out)


def test_fixed_and_parent_ids_survive_real_prep_and_input_is_immutable(tmp_path):
    raw = {**room('A'), 'objects': [
        dict(id='table:original', category='table', size=[2, 1, 1], pos=[4, 4, 0],
             yaw=0, anchor='floor', parent=None),
        dict(id='child:original', category='vase', size=[0.3, 0.3, 0.5], pos=[4, 4, 1],
             yaw=0, anchor='object', parent='table:original'),
        dict(id='door:original', category='door', size=[0.15, 1, 2.2], pos=[0.075, 5, 0],
             yaw=0, anchor='wall', parent=None, structure=True)]}
    saved = copy.deepcopy(raw)
    prepared, reason, content, _ = corpus._prepare(raw, ARGS)
    assert raw == saved
    assert reason is None and content is None
    assert [o['id'] for o in prepared['objects']] == ['table:original', 'child:original']
    assert prepared['objects'][1]['parent'] == 'table:original'
    assert [o['id'] for o in prepared['fixed']] == ['door:original']
    release, ir, out, _ = fixture(tmp_path, [raw], {'train': ['A']})
    corpus.build(release, ir, out)
    written, identities = outputs(out)
    assert written == [{**prepared, 'split': 'train'}]
    assert identities['A']['content_key'] is None


def test_workers_do_not_eagerly_consume_the_entire_raw_stream():
    consumed = 0
    def raw_stream():
        nonlocal consumed
        for i in range(9):
            consumed += 1
            yield {'uid': str(i)}, room(str(i))
    ordered = corpus._ordered(raw_stream(), ARGS, workers=2)
    metadata, result = next(ordered)
    assert metadata['uid'] == '0' and result[0]['uid'] == '0'
    assert consumed <= 4
    assert [metadata['uid'] for metadata, _ in ordered] == [str(i) for i in range(1, 9)]


@pytest.mark.parametrize('case', ['not_object', 'empty'])
def test_invalid_or_empty_release_aborts(tmp_path, case):
    release, ir, out, manifest = fixture(tmp_path)
    for split in ('train', 'dev', 'test'):
        manifest['files'][split + '.jsonl'] = write_rows(
            release / (split + '.jsonl'), [[]] if case == 'not_object' else [])
    rewrite_manifest(release, manifest)
    with pytest.raises(ValueError, match='JSON object|empty released corpus'):
        corpus.build(release, ir, out)
    assert_clean(out)


def test_worker_failure_leaves_no_partial_output(tmp_path):
    bad = room('A')
    del bad['objects']
    release, ir, out, _ = fixture(tmp_path, [bad], {'train': ['A']})
    with pytest.raises(KeyError, match='objects'):
        corpus.build(release, ir, out, workers=2)
    assert_clean(out)
