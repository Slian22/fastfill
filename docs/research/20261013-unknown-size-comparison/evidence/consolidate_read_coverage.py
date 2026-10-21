"""Consolidate version-bound source-reading evidence; never claim test execution."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    inventory_path = 'evidence/agency-full-code-coverage-20261013.json'
    tests_path = 'evidence/agency-full-test-source-coverage-20261013.json'
    metrics_path = str((OUT / 'agency-full-metrics-audit-20261013.json').relative_to(ROOT))
    inventory = read(inventory_path)
    records = {f['path']: {**f, 'evidence': [inventory_path]}
               for f in inventory['files']
               if f['scope'] in ('primary', 'delegated_metric')}
    inputs = [inventory_path, tests_path, metrics_path]
    for f in read(tests_path)['files']:
        records[f['path']] = {**records[f['path']], **f, 'evidence': [tests_path]}
    for f in read(metrics_path)['code_files_read']:
        if f.get('read_scope') != 'full' or f['path'] not in records:
            continue
        records[f['path']] = {**records[f['path']], 'read_status': 'fresh_full',
                              'sha256': f.get('sha256_at_handoff', f['sha256']),
                              'evidence': [metrics_path]}
    for suffix in ('v2-1', 'v2-2', 'v2-3', 'primary-remaining'):
        source = f'evidence/agency-test-source-{suffix}-20261013.json'
        if not (ROOT / source).exists():
            continue
        inputs.append(source)
        for f in read(source)['files']:
            if f['path'] not in records:
                raise ValueError(f'Unexpected source-review path: {f["path"]}')
            status = 'reused_hash_validated' if f['read_status'] == 'reused_full' else f['read_status']
            records[f['path']] = {**records[f['path']], **f, 'read_status': status,
                                  'evidence': [source]}
    files = []
    for path, entry in sorted(records.items()):
        actual = digest(ROOT / path) if (ROOT / path).is_file() else None
        files.append({'path': path, 'scope': entry['scope'], 'role': entry['role'],
                      'read_status': entry['read_status'], 'lines': entry['lines'],
                      'reviewed_sha256': entry['sha256'], 'current_sha256': actual,
                      'current_hash_matches': actual == entry['sha256'],
                      'evidence': entry['evidence']})
    summaries = {}
    for role in ('production', 'test'):
        rows = [f for f in files if f['role'] == role]
        summaries[role] = {'files': len(rows), 'lines': sum(f['lines'] for f in rows),
                           'read_status_counts': dict(Counter(f['read_status'] for f in rows)),
                           'changed_since_review': [f['path'] for f in rows
                                                    if not f['current_hash_matches']],
                           'reading_gaps': [f['path'] for f in rows
                                            if f['read_status'] not in
                                            ('fresh_full', 'reused_hash_validated')]}
    output = {'schema_version': 1, 'observed_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Original frozen first-party primary + delegated metric source inventory',
              'status': 'incomplete' if any(s['reading_gaps'] for s in summaries.values())
                        else 'all_original_files_reviewed_at_recorded_versions',
              'summary': summaries, 'files': files,
              'input_evidence': [{'path': p, 'sha256': digest(ROOT / p)} for p in inputs],
              'limits': ['Full reading is not test execution, coverage or correctness proof.',
                         'Changed sources require separate patch review; old hashes remain explicit.',
                         'New correction files are outside this frozen inventory and reviewed separately.',
                         'Vendor roots, ancillary audit programs and historical copies are not claimed fully read.']}
    (OUT / 'full-audit-read-coverage.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': output['status'], 'summary': summaries}, ensure_ascii=False))


if __name__ == '__main__':
    main()
