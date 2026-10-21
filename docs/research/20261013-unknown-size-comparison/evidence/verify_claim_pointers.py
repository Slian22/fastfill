"""Read-only exact JSON-pointer verification of the audit's numeric evidence.

ARIS's text matcher is a secondary check over compact projections. This command
checks the original source bytes and the precise field, not an arbitrary number
found elsewhere in the file. No model or API request is made.
"""
import argparse
import hashlib
import json
from pathlib import Path


def verify(claim, root):
    payload = (root / claim['source']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != claim['sha256']:
        raise ValueError(f"source changed: {claim['id']}")
    actual = json.loads(payload)
    for token in claim['json_pointer'].split('/')[1:]:
        key = token.replace('~1', '/').replace('~0', '~')
        actual = actual[int(key)] if isinstance(actual, list) else actual[key]
    if type(actual) is not type(claim['value']) or actual != claim['value']:
        raise ValueError(f"value mismatch: {claim['id']}")
    return {'id': claim['id'], 'status': 'exact_field_verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--claims', type=Path, required=True)
    args = parser.parse_args()
    claims = json.loads(args.claims.read_text())
    results = [verify(claim, args.root) for claim in claims]
    print(json.dumps({'results': results, 'verified': len(results),
                      'scope': 'Exact cited field only; not scientific claim approval.'}, indent=2))


if __name__ == '__main__':
    main()
