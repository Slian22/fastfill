"""Local-only acceptance check for the parent's NEW full production snapshot."""
import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_CHANGES = {'fastfill/comparison/geometry.py',
                    'fastfill/comparison/final_tables.py',
                    'fastfill/comparison/train_pair_helpers.py'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(root):
    root = Path(root)
    files = json.loads((root / 'MANIFEST.json').read_text())['files']
    for name, expected in files.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('invalid code manifest path')
        path = root
        for part in rel.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError('code snapshot symlink forbidden')
        if digest(path) != expected:
            raise ValueError('code snapshot hash mismatch: ' + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.relative_to(root).parts and p.suffix != '.pyc'}
    if actual != {*files, 'MANIFEST.json'}:
        raise ValueError('code snapshot has undeclared or missing files')
    return files


def verify(old, new):
    old, new = Path(old).resolve(), Path(new).resolve()
    if old == new:
        raise ValueError('new immutable snapshot required')
    before, after = verified(old), verified(new)
    if before.keys() != after.keys():
        raise ValueError('full production snapshot inventory changed')
    changes = {name for name in before if before[name] != after[name]}
    if changes != EXPECTED_CHANGES:
        raise ValueError('expected exactly the three parent-reviewed fixes: ' + str(sorted(changes)))
    return {'snapshot_manifest_sha256': digest(new / 'MANIFEST.json'),
            'changes': {name: {'before': before[name], 'after': after[name]} for name in sorted(changes)},
            'inference_and_driver_unchanged': True,
            'api_client_sha256': after['server_api_judge.py']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.old, args.new), indent=2, sort_keys=True))
