#!/usr/bin/env python3
"""Standalone, bounded formal judge orchestration; imports perform no API calls.

Run only on the credential-owning server, explicitly through the parent:
  python judge_batch.py --bundle VISUAL/api-bundle --output NEW --client CLIENT \
    --max-cases 348 --expected-total 348
Mapping/failures default to sibling api-mapping.json/api-failures.json; explicit
--mapping/--failures paths are supported. FASTFILL_API_ENV may name the existing
.fastfill_api.env (default ~/.fastfill_api.env). Only the child client reads it.
No ambient OPENAI credentials are forwarded. No retries, replacements or resume.

All selected arms (including render failures) count against the hard 348 cap.
Every artifact and the client are frozen before the first subprocess. Absolute
snapshot paths avoid changing relative-path meaning in one-case manifests.
An existing output is rejected, including after interrupted execution. Preserve
it: per-case reservations and records.jsonl retain partial progress. Rerunning
with a different output is a new evaluation and must be authorized separately.

summary.json has status/model/counts, by_arm/by_stratum numeric score means and
score_counts, plus aggregate/paired tables. Overall is never imputed. Paired
means are joint minus placement on UID-matched, jointly completed records with
both values present. They are descriptive complete-case comparisons, not
unbiased estimates under nonrandom failure. No demographic labels are present;
stratum counts expose the available coverage rather than claiming fairness.
provenance.json is required, frozen verbatim and included with all bundle file
hashes in run_identity_sha256. Its geometry lineage is carried in provenance
and at the summary top level; the final aggregator checks the actual geometry.
When the visual MANIFEST.json is present, its bundle hashes are also verified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

CAP = 348
METRICS = ('functionality', 'layout', 'aesthetics', 'overall')
ARMS = ('joint', 'placement')
LINEAGE_HASHES = ('geometry_manifest_sha256', 'reference_manifest_sha256',
                  'data_manifest_sha256', 'corpus_manifest_sha256')


class BatchError(ValueError):
    """Fixed, non-sensitive orchestration error."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def content_hash(case):
    return digest(canonical({'id': case['id'], 'prompt_sha256': case['prompt_sha256'],
                            'image_sha256': [i['sha256'] for i in case['images']]}))


def _write_bytes(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o400)


def _write(path, value):
    _write_bytes(path, canonical(value) + b'\n')


def _client(path):
    spec = importlib.util.spec_from_file_location('_fastfill_frozen_judge', path)
    if spec is None or spec.loader is None:
        raise BatchError('invalid_client')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Avoid leaving bytecode beside the provided client.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _read_json(client, path):
    data = client.bounded_read(path, client.MAX_MANIFEST_BYTES, 'bundle_read_error')
    return client.strict_json(data, 'invalid_bundle_json'), digest(data)


def _metadata(mapping):
    if not isinstance(mapping, dict):
        raise BatchError('invalid_mapping')
    rows = []
    for cid, meta in mapping.items():
        if (not isinstance(cid, str) or not cid or not isinstance(meta, dict)
                or meta.get('arm') not in ARMS
                or any(not isinstance(meta.get(k), str) or not meta[k] for k in ('uid', 'stratum'))
                or 'group' not in meta
                or (meta['group'] is not None and not isinstance(meta['group'], (str, int)))):
            raise BatchError('invalid_mapping_or_reference_arm')
        for alias in ('id', 'case_id', 'caseid'):
            if alias in meta and meta[alias] != cid:
                raise BatchError('mapping_id_mismatch')
        rows.append({'id': cid, **{k: meta[k] for k in ('uid', 'arm', 'stratum', 'group')}})
    if len({(r['uid'], r['arm']) for r in rows}) != len(rows):
        raise BatchError('duplicate_uid_arm')
    for uid in {r['uid'] for r in rows}:
        pair = [r for r in rows if r['uid'] == uid]
        if len({r['stratum'] for r in pair}) != 1:
            raise BatchError('inconsistent_pair_stratum')
        if len({r['group'] for r in pair if r['group'] is not None}) > 1:
            raise BatchError('inconsistent_pair_group')
    return rows


def _inputs(bundle, mapping, failures, client, max_cases, expected_total):
    root = Path(bundle).resolve()
    if (root / 'api-bundle/manifest.json').is_file():
        root = root / 'api-bundle'
    manifest, manifest_hash = _read_json(client, root / 'manifest.json')
    meta_path = Path(mapping) if mapping is not None else root.parent / 'api-mapping.json'
    fail_path = Path(failures) if failures is not None else root.parent / 'api-failures.json'
    meta, mapping_hash = _read_json(client, meta_path)
    failed, failures_hash = _read_json(client, fail_path)
    rows = _metadata(meta)
    if len(rows) > max_cases or (expected_total is not None and len(rows) != expected_total):
        raise BatchError('selected_count_out_of_bounds')
    if (not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int
            or manifest['schema_version'] != 1 or not isinstance(manifest.get('expected_model'), str)
            or not manifest['expected_model'] or not isinstance(manifest.get('cases'), list)):
        raise BatchError('invalid_manifest')
    cases = manifest['cases']
    for case in cases:
        client.validate_case(case)
        if set(case) != {'id', 'prompt_file', 'prompt_sha256', 'images'}:
            raise BatchError('unexpected_case_fields')
        if any(set(i) != {'path', 'sha256'} for i in case['images']):
            raise BatchError('unexpected_image_fields')
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids):
        raise BatchError('duplicate_case_id')
    if not isinstance(failed, list):
        raise BatchError('invalid_failures')
    failure_ids = [r.get('id', r.get('case_id')) if isinstance(r, dict) else None for r in failed]
    if any(not isinstance(cid, str) for cid in failure_ids) or len(set(failure_ids)) != len(failure_ids):
        raise BatchError('duplicate_or_invalid_failure')
    if set(ids) & set(failure_ids) or set(ids) | set(failure_ids) != set(meta):
        raise BatchError('incomplete_or_overlapping_cohort')
    for cid, failure in zip(failure_ids, failed):
        if 'failure' not in failure or failure['failure'] is None:
            raise BatchError('missing_failure_reason')
        if any(k in failure and failure[k] != meta[cid][k] for k in ('uid', 'arm', 'stratum', 'group')):
            raise BatchError('failure_metadata_mismatch')
    provenance = {'bundle_manifest_sha256': manifest_hash, 'mapping_sha256': mapping_hash,
                  'failures_sha256': failures_hash}
    lineage, raw_lineage = _bundle_lineage(client, root, provenance)
    return root, manifest, rows, dict(zip(failure_ids, failed)), {**provenance, **lineage}, raw_lineage


def _bundle_inventory(client, root):
    files = {}
    for path in sorted(root.rglob('*')):
        if not path.resolve().is_relative_to(root):
            raise BatchError('artifact_outside_bundle')
        if path.is_dir():
            if path.is_symlink():
                raise BatchError('bundle_directory_symlink')
            continue
        if path.suffix not in {'.json', '.txt', '.png'} or len(files) >= CAP * 18 + 2:
            raise BatchError('unexpected_bundle_file')
        data = client.bounded_read(path, client.MAX_IMAGE_BYTES, 'bundle_file_read_error')
        files[path.relative_to(root).as_posix()] = digest(data)
    return files


def _bundle_lineage(client, root, inputs):
    source = root / 'provenance.json'
    if not source.resolve().is_relative_to(root):
        raise BatchError('provenance_outside_bundle')
    data = client.bounded_read(source, client.MAX_MANIFEST_BYTES, 'provenance_read_error')
    lineage = client.strict_json(data, 'invalid_provenance_json')
    if (not isinstance(lineage, dict) or set(lineage) != set(LINEAGE_HASHES) | {'schema_version', 'corpus_link'}
            or type(lineage.get('schema_version')) is not int or lineage['schema_version'] != 1
            or any(not client.valid_hash(lineage.get(k)) for k in LINEAGE_HASHES)
            or lineage.get('corpus_link') not in {'verified', 'unverified_without_eval_data'}):
        raise BatchError('invalid_provenance')
    files = _bundle_inventory(client, root)
    if files.get('manifest.json') != inputs['bundle_manifest_sha256'] or files.get('provenance.json') != digest(data):
        raise BatchError('bundle_changed_during_preflight')
    visual_hash = None
    if (root.parent / 'MANIFEST.json').exists():
        visual, visual_hash = _read_json(client, root.parent / 'MANIFEST.json')
        expected = visual.get('files') if isinstance(visual, dict) else None
        bound = {root.name + '/' + name: value for name, value in files.items()}
        bound = {**bound, 'api-mapping.json': inputs['mapping_sha256'], 'api-failures.json': inputs['failures_sha256']}
        if (not isinstance(expected, dict) or any(expected.get(k) != v for k, v in bound.items())
                or {k for k in expected if k.startswith(root.name + '/')} != {root.name + '/' + k for k in files}):
            raise BatchError('visual_manifest_lineage_mismatch')
        if 'provenance' in visual and (not isinstance(visual['provenance'], dict)
                or any(visual['provenance'].get(k) != lineage[k] for k in (*LINEAGE_HASHES, 'corpus_link'))):
            raise BatchError('visual_manifest_provenance_mismatch')
    return {'provenance': lineage, 'provenance_sha256': digest(data), 'bundle_files': files,
            'geometry_manifest_sha256': lineage['geometry_manifest_sha256'],
            'visual_manifest_sha256': visual_hash}, data


def _artifact(client, root, relative, expected, maximum):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise BatchError('artifact_outside_bundle')
    if any(p.lower() in {'reference', 'gt', 'ground_truth', 'ground-truth'} for p in path.relative_to(root).parts):
        raise BatchError('reference_artifact_forbidden')
    return client.verified_artifact(root, relative, expected, maximum)


def _freeze_case(client, root, case, artifact_dir):
    prompt = _artifact(client, root, case['prompt_file'], case['prompt_sha256'], client.MAX_PROMPT_BYTES)
    try:
        if not prompt.decode('utf-8').strip():
            raise BatchError('empty_prompt')
    except UnicodeError:
        raise BatchError('invalid_prompt_utf8') from None
    entries = [(case['prompt_sha256'] + '.txt', prompt)]
    images = []
    for item in case['images']:
        data = _artifact(client, root, item['path'], item['sha256'], client.MAX_IMAGE_BYTES)
        client.verify_png(data)
        name = item['sha256'] + '.png'
        entries.append((name, data))
        images.append({'path': str(artifact_dir / name), 'sha256': item['sha256']})
    for name, data in entries:
        target = artifact_dir / name
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)
            target.chmod(0o400)
    frozen = {**case, 'prompt_file': str(artifact_dir / entries[0][0]), 'images': images}
    # Pure request preparation checks the exact client's payload limits, no config read.
    client.prepare_case(frozen, artifact_dir, client.Config('unused', 'https://unused.invalid', 'unused'))
    return frozen


def _result(client, case, directory, model, returncode):
    stem = digest(case['id'].encode())
    identity = {'schema_version': 1, 'id': case['id'], 'expected_model': model,
                'content_sha256': content_hash(case)}
    attempt = client.read_record(directory / (stem + '.attempt.json'), identity)
    record = client.read_record(directory / (stem + '.result.json'), identity)
    if (attempt.get('status') != 'attempted' or not client.valid_hash(attempt.get('request_sha256'))
            or record.get('request_sha256') != attempt['request_sha256']):
        raise BatchError('invalid_attempt_identity')
    if record.get('status') == 'failed':
        return {'status': 'failed', 'error': 'client_evaluation_failed'}
    if (returncode != 0 or record.get('status') != 'completed' or 'error' in record
            or record.get('returned_model') != model):
        raise BatchError('invalid_completion')
    scores = client.parse_scores(client.canonical(record.get('scores')))
    if client.parse_scores(record.get('raw_final_text')) != scores:
        raise BatchError('score_text_mismatch')
    return {'status': 'completed', 'scores': {k + '_score': scores[k + '_score']
                                             for k in METRICS if k + '_score' in scores}}


def _run_case(client, case, output, client_path, client_hash, model, timeout, env_file):
    directory = output / 'cases' / digest(case['id'].encode())
    directory.mkdir(parents=True)
    _write(directory / 'reservation.json', {'id': case['id'], 'content_sha256': content_hash(case),
                                           'attempt_limit': 1})
    manifest_path = directory / 'manifest.json'
    _write(manifest_path, {'schema_version': 1, 'expected_model': model, 'cases': [case]})
    launched = False
    try:
        if digest(client_path.read_bytes()) != client_hash:
            raise BatchError('client_changed')
        client.prepare_case(case, output, client.Config('unused', 'https://unused.invalid', model))
        command = [sys.executable, str(client_path), '--manifest', str(manifest_path),
                   '--results-dir', str(directory / 'results'), '--env-file', str(env_file),
                   '--limit', '1', '--timeout', str(timeout)]
        environment = {k: os.environ[k] for k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'SYSTEMROOT')
                       if k in os.environ}
        launched = True
        process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False, timeout=timeout + 10, env=environment)
        result = _result(client, case, directory / 'results', model, process.returncode)
    except subprocess.TimeoutExpired:
        result = {'status': 'failed', 'error': 'client_timeout'}
    except (OSError, ValueError, client.JudgeError):
        result = {'status': 'failed', 'error': 'client_or_artifact_failure'}
    return {**result, 'launched': launched}


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _aggregate(rows, arm, stratum):
    successes = [r for r in rows if r['status'] == 'completed']
    scores = {metric: [r['scores'][metric + '_score'] for r in successes
                       if metric + '_score' in r['scores']] for metric in METRICS}
    return {'arm': arm, 'stratum': stratum, 'selected': len(rows), 'success': len(successes),
            'failed': len(rows) - len(successes),
            **{m: {'count': len(values), 'missing': len(rows) - len(values), 'mean': _mean(values)}
               for m, values in scores.items()}}


def _paired(rows, stratum):
    uids = sorted({r['uid'] for r in rows})
    indexed = {(r['uid'], r['arm']): r for r in rows}
    pairs = [(indexed.get((uid, 'joint')), indexed.get((uid, 'placement'))) for uid in uids]
    shared = [(a, b) for a, b in pairs if a and b and a['status'] == b['status'] == 'completed']
    values = {m: [a['scores'][m + '_score'] - b['scores'][m + '_score'] for a, b in shared
                  if m + '_score' in a['scores'] and m + '_score' in b['scores']] for m in METRICS}
    return {'stratum': stratum, 'direction': 'joint_minus_placement', 'selected_pairs': len(pairs),
            'shared_successful': len(shared), 'missing_pairs': len(pairs) - len(shared),
            **{m: {'count': len(v), 'mean_difference': _mean(v)} for m, v in values.items()}}


def _compact(row):
    return {'requested': row['selected'], 'success': row['success'], 'failed': row['failed'],
            **{m + '_score': row[m]['mean'] for m in METRICS},
            'score_counts': {m + '_score': row[m]['count'] for m in METRICS}}


def _validate_compact(row):
    requested, success, failed = (row[k] for k in ('requested', 'success', 'failed'))
    if (any(type(v) is not int or v < 0 for v in (requested, success, failed))
            or not success <= requested <= CAP or failed != requested - success):
        raise BatchError('inconsistent_summary_counts')
    for metric in METRICS:
        key = metric + '_score'
        count, value = row['score_counts'][key], row[key]
        if (type(count) is not int or not 0 <= count <= success
                or (metric != 'overall' and count != success)):
            raise BatchError('inconsistent_score_counts')
        if count == 0:
            if value is not None:
                raise BatchError('zero_success_score_must_be_null')
        elif type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10:
            raise BatchError('invalid_summary_score')


def validate_summary(summary):
    """Validate denominators and reported means before publishing any tables."""
    try:
        counts = summary['counts']
        selected, runnable, launched, success, failed = (counts[k] for k in
            ('selected', 'runnable', 'launched', 'success', 'failed'))
        if (any(type(v) is not int or v < 0 for v in (selected, runnable, launched, success, failed))
                or not success <= launched <= runnable <= selected <= CAP or failed != selected - success):
            raise BatchError('inconsistent_summary_counts')
        groups = summary['by_stratum']
        arms = summary['by_arm']
        for group in [arms, *groups.values()]:
            if set(group) != set(ARMS):
                raise BatchError('invalid_summary_arms')
            for row in group.values():
                _validate_compact(row)
        for field, total in (('requested', selected), ('success', success), ('failed', failed)):
            if sum(arms[arm][field] for arm in ARMS) != total:
                raise BatchError('inconsistent_summary_totals')
            for arm in ARMS:
                if sum(group[arm][field] for group in groups.values()) != arms[arm][field]:
                    raise BatchError('inconsistent_stratum_totals')
        for arm in ARMS:
            for metric in METRICS:
                key = metric + '_score'
                count = sum(group[arm]['score_counts'][key] for group in groups.values())
                if count != arms[arm]['score_counts'][key]:
                    raise BatchError('inconsistent_stratum_score_counts')
        for row in summary['aggregate']:
            target = arms[row['arm']] if row['stratum'] is None else groups[row['stratum']][row['arm']]
            if _compact(row) != target:
                raise BatchError('inconsistent_aggregate_table')
    except (KeyError, TypeError, AttributeError, OverflowError):
        raise BatchError('invalid_summary_structure') from None


def summarize(records, model, runnable):
    strata = sorted({r['stratum'] for r in records})
    aggregate = [_aggregate([r for r in records if r['arm'] == arm and (s is None or r['stratum'] == s)], arm, s)
                 for s in [None, *strata] for arm in ARMS]
    paired = [_paired([r for r in records if s is None or r['stratum'] == s], s) for s in [None, *strata]]
    success = sum(r['status'] == 'completed' for r in records)
    summary = {'schema_version': 1, 'status': 'completed' if success == len(records) else 'partial',
            'model': model, 'counts': {'selected': len(records), 'runnable': runnable,
                'launched': sum(r['launched'] for r in records), 'success': success,
                'failed': len(records) - success}, 'aggregate': aggregate, 'paired': paired,
            'by_arm': {r['arm']: _compact(r) for r in aggregate if r['stratum'] is None},
            'by_stratum': {s: {r['arm']: _compact(r) for r in aggregate if r['stratum'] == s} for s in strata},
            'overall_policy': 'reported_only_no_imputation', 'replacement': False, 'retry': False}
    validate_summary(summary)
    return summary


def _tables(output, summary):
    validate_summary(summary)
    for name, columns in (('aggregate', ('arm', 'stratum', 'selected', 'success', 'failed')),
                          ('paired', ('stratum', 'direction', 'selected_pairs', 'shared_successful', 'missing_pairs'))):
        rows = [{**{k: r[k] for k in columns},
                 **{m + '_' + k: v for m in METRICS for k, v in r[m].items()}}
                for r in summary[name]]
        with (output / (name + '.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Formal GPT evaluation', '',
             'Missing values are NA; overall is reported only. No retries or replacements.', '',
             '| Arm | Stratum | Selected | Success | Functionality | Layout | Aesthetics | Overall |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    def label(value):
        return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')
    def value(metric, key):
        return 'NA (n=0)' if metric[key] is None else f"{metric[key]:.4f} (n={metric['count']})"
    for r in summary['aggregate']:
        cells = [r['arm'], r['stratum'] if r['stratum'] is not None else 'All', r['selected'], r['success']]
        lines.append('| ' + ' | '.join(map(label, cells + [value(r[m], 'mean') for m in METRICS])) + ' |')
    lines += ['', 'Paired differences: joint minus placement, shared successful UIDs only.', '',
              '| Stratum | Selected pairs | Shared success | Functionality | Layout | Aesthetics | Overall |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['paired']:
        cells = [r['stratum'] if r['stratum'] is not None else 'All', r['selected_pairs'], r['shared_successful']]
        lines.append('| ' + ' | '.join(map(label, cells + [value(r[m], 'mean_difference') for m in METRICS])) + ' |')
    (output / 'tables.md').write_text('\n'.join(lines) + '\n')


def evaluate(bundle, output, client, max_cases=CAP, *, mapping=None, failures=None,
             expected_total=None, timeout=90):
    if (type(max_cases) is not int or not 1 <= max_cases <= CAP
            or (expected_total is not None and (type(expected_total) is not int or not 0 <= expected_total <= max_cases))
            or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600):
        raise BatchError('invalid_bounds')
    output = Path(output).absolute()
    if os.path.lexists(output):
        raise BatchError('output_already_exists')
    client_source = Path(client).resolve()
    client_bytes = client_source.read_bytes()
    api = _client(client_source)
    if digest(client_source.read_bytes()) != digest(client_bytes):
        raise BatchError('client_changed_during_load')
    try:
        root, manifest, metadata, failed, provenance, raw_lineage = _inputs(
            bundle, mapping, failures, api, max_cases, expected_total)
    except api.JudgeError as exc:
        raise BatchError(str(exc)) from None
    if output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve()):
        raise BatchError('overlapping_output_bundle')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    artifacts = output / 'artifacts'
    artifacts.mkdir()
    try:
        frozen = [_freeze_case(api, root, c, artifacts) for c in manifest['cases']]
    except api.JudgeError as exc:
        raise BatchError(str(exc)) from None
    client_path = output / 'client.py'
    client_path.write_bytes(client_bytes)
    client_path.chmod(0o400)
    _write_bytes(output / 'provenance.json', raw_lineage)
    run_identity = {'bundle_files': provenance['bundle_files'],
                    'mapping_sha256': provenance['mapping_sha256'],
                    'failures_sha256': provenance['failures_sha256'],
                    'client_sha256': digest(client_bytes)}
    provenance = {**provenance, 'run_identity_sha256': digest(canonical(run_identity))}
    snapshot = {'schema_version': 1, 'model': manifest['expected_model'], 'max_cases': max_cases,
                'expected_total': expected_total, 'replacement': False, 'retry': False,
                'client_sha256': digest(client_bytes), **provenance,
                'cases': frozen, 'mapping': metadata, 'failures': failed,
                'case_sha256': {c['id']: content_hash(c) for c in frozen}}
    _write(output / 'snapshot.json', snapshot)
    env_file = Path(os.environ.get('FASTFILL_API_ENV', '~/.fastfill_api.env')).expanduser()
    indexed = {c['id']: c for c in frozen}
    records = []
    with (output / 'records.jsonl').open('xb') as stream:
        for meta in metadata:
            cid = meta['id']
            result = ({'status': 'failed', 'error': 'preparation_failed', 'failure': failed[cid]['failure'],
                       'launched': False} if cid in failed else
                      _run_case(api, indexed[cid], output, client_path, digest(client_bytes),
                                manifest['expected_model'], timeout, env_file))
            record = {**meta, **result}
            records.append(record)
            stream.write(canonical(record) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
    summary = {**summarize(records, manifest['expected_model'], len(frozen)), **provenance,
               'client_sha256': digest(client_bytes),
               'snapshot_sha256': digest((output / 'snapshot.json').read_bytes())}
    _write(output / 'summary.json', summary)
    _tables(output, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('bundle', 'output', 'client'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('mapping', 'failures'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--max-cases', type=int, default=CAP)
    parser.add_argument('--expected-total', type=int)
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args(argv)
    try:
        summary = evaluate(**vars(args))
    except Exception:
        print('judge_batch_error: validation_or_execution_failed', file=sys.stderr)
        return 1
    print(json.dumps({'status': summary['status'], 'counts': summary['counts']}))
    return int(bool(summary['counts']['failed']))


if __name__ == '__main__':
    sys.exit(main())
