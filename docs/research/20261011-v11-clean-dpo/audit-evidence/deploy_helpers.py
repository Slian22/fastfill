"""Install a hash-checked helper snapshot, without touching experiment sources."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile

archive, digest, destination = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
    raise ValueError('Archive digest mismatch')
if destination.exists():
    raise FileExistsError(destination)
with tempfile.TemporaryDirectory(prefix='.audit-install-', dir=destination.parent) as temporary:
    staged = Path(temporary)
    with tarfile.open(archive, 'r:gz') as bundle:
        bundle.extractall(staged, filter='data')
    manifest = json.loads((staged / 'audit-source-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = staged / relative
        if not path.resolve().is_relative_to(staged.resolve()) or path.is_symlink():
            raise ValueError('Unexpected path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Installed source mismatch: {relative}')
    staged.rename(destination)
print(json.dumps({'installed': str(destination), 'files_verified': len(manifest['files']),
                  'source_sha256': manifest['source_sha256'], 'archive_sha256': digest}))
