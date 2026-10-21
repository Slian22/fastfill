"""Read-only, point-in-time byte audit; does not replace render-time checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root):
    started = time.monotonic()
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    manifest_hash = digest(root / 'MANIFEST.json')
    assert 'resources.json' in manifest['files'], 'resources not in manifest'
    assert all(digest(root / name) == expected for name, expected in manifest['files'].items())
    resources = json.loads((root / 'resources.json').read_text())
    expected_files = {}
    for resource in resources.values():
        rows = [(resource['mesh_path'], resource['mesh_sha256'])]
        for material in resource['materials']:
            rows += [(material['path'], material['sha256'])]
            rows += list(material['texture_sha256'].items())
        for path, expected in rows:
            assert path not in expected_files or expected_files[path] == expected, path
            expected_files[path] = expected
    records = []
    for name, expected in sorted(expected_files.items()):
        path = Path(name)
        try:
            before = path.stat()
            actual = digest(path)
            after = path.stat()
            stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            records.append(dict(path=name, expected=expected, actual=actual,
                                bytes=after.st_size, passed=stable and actual == expected))
        except OSError as exc:
            records.append(dict(path=name, expected=expected, passed=False, error=str(exc)))
    assert digest(root / 'MANIFEST.json') == manifest_hash, 'reference manifest changed'
    assert digest(root / 'resources.json') == manifest['files']['resources.json']
    return dict(utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                reference_manifest_sha256=manifest_hash, resources=len(resources), files=len(records),
                total_bytes=sum(r.get('bytes', 0) for r in records),
                failed=sum(not r['passed'] for r in records), records=records,
                limitation='Point-in-time bytes only; no geometry/material correctness or future render-time integrity certification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = verify(args.reference)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
    raise SystemExit(bool(result['failed']))
