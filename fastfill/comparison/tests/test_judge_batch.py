"""Offline orchestration tests: subprocess replaced, no keys or network read."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace
import zlib

import pytest

from fastfill.comparison import judge_batch as batch

CLIENT = Path(__file__).resolve().parents[3] / 'docs/research/20261013-unknown-size-comparison/api-smoke/server_api_judge.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


PNG = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
       + chunk(b'IDAT', zlib.compress(b'\x00\xff\xff\xff')) + chunk(b'IEND', b''))


def bundle(tmp_path, n=2, arms=('joint', 'placement')):
    root = tmp_path / 'bundle'
    api = root / 'api-bundle'
    api.mkdir(parents=True)
    (api / 'prompt.txt').write_text('Judge supplied prediction renders. Return rubric scores.')
    cases, mapping = [], {}
    for i in range(n):
        for arm in arms:
            cid = f'room-{i}-{arm}'
            image = api / f'{cid}.png'
            image.write_bytes(PNG)
            cases.append(dict(id=cid, prompt_file='prompt.txt',
                prompt_sha256=digest((api / 'prompt.txt').read_bytes()),
                images=[dict(path=image.name, sha256=digest(PNG))]))
            mapping[cid] = dict(case_id=cid, uid=f'room-{i}', arm=arm,
                               stratum='bedroom', group='group-a')
    write(api / 'manifest.json', dict(schema_version=1, expected_model='gpt-6.1-sol', cases=cases))
    write(root / 'api-mapping.json', mapping)
    write(root / 'api-failures.json', [])
    write(api / 'provenance.json', dict(schema_version=1, geometry_manifest_sha256='d' * 64,
        reference_manifest_sha256='e' * 64, data_manifest_sha256='f' * 64,
        corpus_manifest_sha256='a' * 64, corpus_link='verified'))
    return root


def fake_client(calls, outcomes=None, hook=None):
    def run(command, **kwargs):
        assert command[command.index('--limit') + 1] == '1'
        assert kwargs['stdout'] == subprocess.DEVNULL and kwargs['stderr'] == subprocess.DEVNULL
        manifest = json.loads(Path(command[command.index('--manifest') + 1]).read_text())
        assert len(manifest['cases']) == 1
        case = manifest['cases'][0]
        calls.append(case['id'])
        for entry in case['images']:
            path = Path(entry['path'])
            assert path.is_absolute() and digest(path.read_bytes()) == entry['sha256']
        out = Path(command[command.index('--results-dir') + 1])
        out.mkdir(parents=True, exist_ok=True)
        mode = (outcomes or {}).get(case['id'], 'ok')
        identity = dict(schema_version=1, id=case['id'], expected_model=manifest['expected_model'],
            content_sha256=batch.content_hash(case), request_sha256='a' * 64)
        stem = digest(case['id'].encode())
        write(out / (stem + '.attempt.json'), dict(identity, status='attempted'))
        if mode == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        scores = dict(functionality_score=8 if 'joint' in case['id'] or case['id'].endswith('-J') else 4,
                      layout_score=6, aesthetics_score=7)
        if mode != 'no_overall':
            scores['overall_score'] = 7.5
        record = dict(identity, returned_model=manifest['expected_model'], usage={},
                      status='completed', scores=scores, raw_final_text=json.dumps(scores))
        if mode == 'fail':
            record = dict(identity, status='failed', error='timeout', raw_final_text='partial')
        if mode == 'partial':
            record = dict(record, scores={'functionality_score': 8})
        if mode == 'wrong_identity':
            record = dict(record, content_sha256='b' * 64)
        write(out / (stem + '.result.json'), record)
        if hook:
            hook(case)
        return SimpleNamespace(returncode=1 if mode == 'fail' else 0)
    return run


def execute(root, tmp_path, monkeypatch, **kwargs):
    calls = []
    monkeypatch.setattr(batch.subprocess, 'run', fake_client(calls, **kwargs))
    out = tmp_path / 'output'
    return batch.evaluate(root, out, CLIENT), calls, out


def test_continues_after_failure_and_tables_use_shared_success(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    result, calls, out = execute(root, tmp_path, monkeypatch,
                               outcomes={'room-0-joint': 'fail'})
    assert len(calls) == 4 and len(set(calls)) == 4
    assert result['counts'] == dict(selected=4, runnable=4, launched=4, success=3, failed=1)
    assert len((out / 'records.jsonl').read_text().splitlines()) == 4
    joint = next(r for r in result['aggregate'] if r['arm'] == 'joint' and r['stratum'] == 'bedroom')
    assert joint['selected'] == 2 and joint['failed'] == 1
    assert joint['functionality']['count'] == 1
    paired = next(r for r in result['paired'] if r['stratum'] == 'bedroom')
    assert paired['functionality'] == dict(count=1, mean_difference=4.0)
    assert paired['shared_successful'] == 1 and paired['selected_pairs'] == 2
    assert (out / 'tables.md').exists() and (out / 'aggregate.csv').exists()
    assert (out / 'paired.csv').exists()
    snapshot = json.loads((out / 'snapshot.json').read_text())
    assert snapshot['replacement'] is False and len(snapshot['cases']) == 4
    assert snapshot['client_sha256'] == digest(CLIENT.read_bytes())


@pytest.mark.parametrize('mutation', ['duplicate', 'pair_duplicate', 'overcap', 'gt', 'escape', 'changed', 'mapping_missing'])
def test_rejects_bad_bundle_before_any_request(tmp_path, monkeypatch, mutation):
    root = bundle(tmp_path)
    path = root / 'api-bundle/manifest.json'
    manifest = json.loads(path.read_text())
    mapping = json.loads((root / 'api-mapping.json').read_text())
    cap = 348
    if mutation == 'duplicate':
        manifest['cases'].append(manifest['cases'][0])
    if mutation == 'pair_duplicate':
        mapping['room-1-joint']['uid'] = 'room-0'
    if mutation == 'overcap':
        cap = 3
    if mutation == 'gt':
        mapping['room-0-joint']['arm'] = 'reference'
    if mutation == 'escape':
        external = tmp_path / 'outside.png'
        external.write_bytes(PNG)
        manifest['cases'][0]['images'][0]['path'] = str(external)
    if mutation == 'changed':
        (root / 'api-bundle/room-1-placement.png').write_bytes(PNG + b'changed')
    if mutation == 'mapping_missing':
        del mapping['room-0-joint']
    write(path, manifest)
    write(root / 'api-mapping.json', mapping)
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('client launched'))
    with pytest.raises((ValueError, batch.BatchError)):
        batch.evaluate(root, tmp_path / 'output', CLIENT, max_cases=cap)


@pytest.mark.parametrize('cap', [0, 349, True, 1.5])
def test_absolute_request_cap(tmp_path, cap):
    with pytest.raises(ValueError):
        batch.evaluate(tmp_path, tmp_path / 'out', CLIENT, max_cases=cap)


@pytest.mark.parametrize('outcome', ['partial', 'wrong_identity', 'timeout'])
def test_partial_or_unfinished_is_failure_without_retry(tmp_path, monkeypatch, outcome):
    root = bundle(tmp_path, 1)
    result, calls, out = execute(root, tmp_path, monkeypatch, outcomes={'room-0-joint': outcome})
    assert len(calls) == 2 and result['counts']['failed'] == 1
    assert result['paired'][0]['shared_successful'] == 0
    with pytest.raises((ValueError, FileExistsError)):
        batch.evaluate(root, out, CLIENT)
    assert len(calls) == 2


def test_overall_is_optional_never_imputed(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    result, _, _ = execute(root, tmp_path, monkeypatch, outcomes={'room-0-joint': 'no_overall'})
    joint = next(r for r in result['aggregate'] if r['arm'] == 'joint')
    assert joint['overall'] == dict(count=0, missing=1, mean=None)
    assert result['paired'][0]['overall'] == dict(count=0, mean_difference=None)


def test_source_image_mutation_cannot_change_frozen_request(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    def mutate(_case):
        (root / 'api-bundle/room-0-placement.png').write_bytes(b'mutated')
    result, calls, out = execute(root, tmp_path, monkeypatch, hook=mutate)
    assert len(calls) == 2 and result['counts']['success'] == 2
    assert all(c['images'][0]['sha256'] == digest(PNG)
               for c in json.loads((out / 'snapshot.json').read_text())['cases'])


def test_render_missingness_is_retained_without_replacement(tmp_path, monkeypatch):
    root = bundle(tmp_path, 2)
    path = root / 'api-bundle/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['cases'] = manifest['cases'][2:]
    write(path, manifest)
    write(root / 'api-failures.json', [dict(case_id='room-0-' + arm, uid='room-0',
        arm=arm, stratum='bedroom', group='group-a', failure='render_failed')
        for arm in ('joint', 'placement')])
    result, calls, _ = execute(root, tmp_path, monkeypatch)
    assert len(calls) == 2 and result['counts'] == dict(selected=4, runnable=2, launched=2, success=2, failed=2)
    assert result['paired'][0]['selected_pairs'] == 2


def test_credentials_only_pass_existing_path_never_read_or_capture(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    secret_path = tmp_path / '.fastfill_api.env'  # Deliberately nonexistent.
    monkeypatch.setenv('FASTFILL_API_ENV', str(secret_path))
    calls = []
    runner = fake_client(calls)
    def checked(command, **kwargs):
        assert command[command.index('--env-file') + 1] == str(secret_path)
        assert 'OPENAI_API_KEY' not in kwargs['env']
        return runner(command, **kwargs)
    monkeypatch.setenv('OPENAI_API_KEY', 'ambient-must-not-pass')
    monkeypatch.setattr(batch.subprocess, 'run', checked)
    out = tmp_path / 'output'
    batch.evaluate(root, out, CLIENT)
    assert not secret_path.exists()
    assert all(b'ambient-must-not-pass' not in p.read_bytes() for p in out.rglob('*') if p.is_file())


def test_zero_runnable_348_selected_and_exact_provenance(tmp_path, monkeypatch):
    import builtins
    import io
    import os
    root = bundle(tmp_path, 174)
    path = root / 'api-bundle/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['cases'] = []
    write(path, manifest)
    mapping = json.loads((root / 'api-mapping.json').read_text())
    failures = [dict(id=cid, **meta, failure={'reason': 'render_failed'})
                for cid, meta in mapping.items()]
    write(root / 'api-failures.json', failures)
    monkeypatch.setenv('FASTFILL_API_ENV', str(tmp_path / '.fastfill_api.env'))
    def guarded_open(original):
        def checked(path, *args, **kwargs):
            if str(path).endswith('.fastfill_api.env'):
                pytest.fail('credential file accessed')
            return original(path, *args, **kwargs)
        return checked
    for module in (builtins, io, os):
        monkeypatch.setattr(module, 'open', guarded_open(module.open))
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('client launched'))
    out = tmp_path / 'output'
    result = batch.evaluate(root / 'api-bundle', out, CLIENT, expected_total=348,
        mapping=root / 'api-mapping.json', failures=root / 'api-failures.json')
    assert result['counts'] == dict(selected=348, runnable=0, launched=0, success=0, failed=348)
    assert result['bundle_manifest_sha256'] == digest(path.read_bytes())
    assert result['mapping_sha256'] == digest((root / 'api-mapping.json').read_bytes())
    assert result['by_stratum']['bedroom']['joint']['requested'] == 174
    assert result['by_arm']['joint']['overall_score'] is None
    assert result['provenance']['geometry_manifest_sha256'] == 'd' * 64
    assert all(result['by_arm'][arm][metric + '_score'] is None
               for arm in batch.ARMS for metric in batch.METRICS)
    assert len((out / 'records.jsonl').read_text().splitlines()) == 348
    batch.validate_summary(json.loads((out / 'summary.json').read_text()))
    assert (out / 'tables.md').is_file() and (out / 'aggregate.csv').is_file()
    assert (out / 'paired.csv').is_file() and not (out / 'cases').exists()


def test_350_frozen_cases_cannot_hide_behind_render_failures(tmp_path, monkeypatch):
    root = bundle(tmp_path, 175)
    path = root / 'api-bundle/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['cases'] = []
    write(path, manifest)
    mapping = json.loads((root / 'api-mapping.json').read_text())
    write(root / 'api-failures.json', [dict(id=cid, **meta, failure='render_failed')
                                      for cid, meta in mapping.items()])
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('client launched'))
    with pytest.raises(ValueError, match='count'):
        batch.evaluate(root / 'api-bundle', tmp_path / 'out', CLIENT)


def test_mutated_frozen_image_fails_but_later_case_is_retained(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    def mutate(case):
        path = Path(case['images'][0]['path'])
        path.chmod(0o600)
        path.write_bytes(b'changed-frozen-artifact')
    result, calls, out = execute(root, tmp_path, monkeypatch, hook=mutate)
    assert len(calls) == 1 and result['counts']['failed'] == 1
    assert result['counts']['selected'] == 2
    assert len((out / 'records.jsonl').read_text().splitlines()) == 2


def test_duplicate_json_key_and_symlink_escape_are_rejected(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('client launched'))
    mapping = root / 'api-mapping.json'
    original = mapping.read_text()
    mapping.write_text('{"duplicate":{},"duplicate":{}}')
    with pytest.raises(ValueError):
        batch.evaluate(root, tmp_path / 'out', CLIENT)
    mapping.write_text(original)
    image = root / 'api-bundle/room-0-joint.png'
    image.unlink()
    target = tmp_path / 'external.png'
    target.write_bytes(PNG)
    image.symlink_to(target)
    with pytest.raises(ValueError, match='outside'):
        batch.evaluate(root, tmp_path / 'out2', CLIENT)


def test_expected_total_mismatch_and_output_overlap(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('client launched'))
    with pytest.raises(ValueError, match='count'):
        batch.evaluate(root, tmp_path / 'out', CLIENT, expected_total=348)
    with pytest.raises(ValueError, match='overlapping'):
        batch.evaluate(root, root / 'api-bundle/output', CLIENT)


def test_cli_main_and_standalone_help(tmp_path, monkeypatch, capsys):
    # Only --help starts a real process; argparse exits before any client import.
    process = subprocess.run([__import__('sys').executable, batch.__file__, '--help'],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0 and '--expected-total' in process.stdout
    root = bundle(tmp_path, 1)
    calls = []
    monkeypatch.setattr(batch.subprocess, 'run', fake_client(calls))
    args = ['--bundle', str(root / 'api-bundle'), '--output', str(tmp_path / 'out'),
            '--client', str(CLIENT), '--expected-total', '2']
    assert batch.main(args) == 0
    assert json.loads(capsys.readouterr().out)['counts']['success'] == 2
    assert batch.main(args) == 1
    assert 'validation_or_execution_failed' in capsys.readouterr().err


def test_geometry_lineage_changes_identity_with_identical_api_manifest(tmp_path, monkeypatch):
    first = bundle(tmp_path / 'first', 1)
    second = bundle(tmp_path / 'second', 1)
    path = second / 'api-bundle/provenance.json'
    lineage = json.loads(path.read_text())
    write(path, {**lineage, 'geometry_manifest_sha256': 'b' * 64})
    assert (first / 'api-bundle/manifest.json').read_bytes() == (second / 'api-bundle/manifest.json').read_bytes()
    one, _, out = execute(first, tmp_path / 'first', monkeypatch)
    two, _, _ = execute(second, tmp_path / 'second', monkeypatch)
    original = (first / 'api-bundle/provenance.json').read_bytes()
    assert one['provenance'] == json.loads(original)
    assert one['geometry_manifest_sha256'] == 'd' * 64
    assert two['geometry_manifest_sha256'] == 'b' * 64
    assert one['provenance_sha256'] == digest(original)
    assert one['run_identity_sha256'] != two['run_identity_sha256']
    assert (out / 'provenance.json').read_bytes() == original
    snapshot = json.loads((out / 'snapshot.json').read_text())
    assert snapshot['provenance'] == one['provenance']
    assert snapshot['run_identity_sha256'] == one['run_identity_sha256']
    assert snapshot['bundle_files'] == {p.relative_to(first / 'api-bundle').as_posix(): digest(p.read_bytes())
        for p in (first / 'api-bundle').rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['missing', 'bad_geometry_hash', 'missing_data', 'boolean_schema', 'bad_link'])
def test_invalid_provenance_rejected_before_client(tmp_path, monkeypatch, change):
    root = bundle(tmp_path, 1)
    path = root / 'api-bundle/provenance.json'
    original = json.loads(path.read_text())
    changed = dict(original)
    if change == 'missing':
        path.unlink()
    else:
        if change == 'bad_geometry_hash':
            changed['geometry_manifest_sha256'] = 'invalid'
        elif change == 'missing_data':
            del changed['data_manifest_sha256']
        elif change == 'boolean_schema':
            changed['schema_version'] = True
        else:
            changed['corpus_link'] = 'assumed'
        write(path, changed)
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('client launched'))
    with pytest.raises(ValueError):
        batch.evaluate(root, tmp_path / 'output', CLIENT)


def test_visual_manifest_detects_replaced_sidecar(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    source = root / 'api-bundle/provenance.json'
    write(root / 'MANIFEST.json', {'files': {p.relative_to(root).as_posix(): digest(p.read_bytes())
        for p in root.rglob('*') if p.is_file()}})
    write(source, {**json.loads(source.read_text()), 'geometry_manifest_sha256': 'b' * 64})
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('client launched'))
    with pytest.raises(ValueError, match='lineage|provenance|manifest'):
        batch.evaluate(root, tmp_path / 'output', CLIENT)


def test_provenance_mutation_after_freeze_preserves_identity(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    path = root / 'api-bundle/provenance.json'
    original = path.read_bytes()
    def mutate(_case):
        write(path, {**json.loads(original), 'geometry_manifest_sha256': 'b' * 64})
    result, calls, out = execute(root, tmp_path, monkeypatch, hook=mutate)
    assert len(calls) == 2 and result['provenance_sha256'] == digest(original)
    assert result['provenance']['geometry_manifest_sha256'] == 'd' * 64
    assert (out / 'provenance.json').read_bytes() == original


@pytest.mark.parametrize('field,value', [('requested', -1), ('success', 2), ('success', True),
    ('functionality_score', float('inf')), ('layout_score', -0.1), ('aesthetics_score', 10.1),
    ('overall_score', True), ('functionality_score', None)])
def test_summary_validation_rejects_inconsistent_counts_or_scores(tmp_path, monkeypatch, field, value):
    root = bundle(tmp_path, 1)
    summary, _, _ = execute(root, tmp_path, monkeypatch)
    invalid = copy.deepcopy(summary)
    invalid['by_stratum']['bedroom']['joint'][field] = value
    with pytest.raises(ValueError):
        batch.validate_summary(invalid)


def test_summary_validation_zero_success_requires_null_and_consistent_totals(tmp_path, monkeypatch):
    root = bundle(tmp_path, 1)
    summary, _, _ = execute(root, tmp_path, monkeypatch, outcomes={
        'room-0-joint': 'fail', 'room-0-placement': 'fail'})
    batch.validate_summary(summary)
    invalid = copy.deepcopy(summary)
    invalid['by_arm']['joint']['functionality_score'] = 0
    with pytest.raises(ValueError):
        batch.validate_summary(invalid)
    invalid = copy.deepcopy(summary)
    invalid['counts']['success'] = 1
    with pytest.raises(ValueError):
        batch.validate_summary(invalid)


def test_actual_visual_bundle_lineage_contract(tmp_path, monkeypatch):
    from fastfill.comparison.tests.test_visual_pipeline import prepare
    visual, prepared, _, _ = prepare(tmp_path, monkeypatch, n=2)
    summary, calls, output = execute(visual / 'api-bundle', tmp_path, monkeypatch)
    assert len(calls) == 4 and summary['counts']['selected'] == 4
    assert summary['provenance']['geometry_manifest_sha256'] == prepared['provenance']['geometry_manifest_sha256']
    assert summary['visual_manifest_sha256'] == digest((visual / 'MANIFEST.json').read_bytes())
    assert (output / 'provenance.json').read_bytes() == (visual / 'api-bundle/provenance.json').read_bytes()
    batch.validate_summary(summary)


OFFLINE_TRANSPORT = '''
# Test-only injection: real parser, evaluator, reservation, and result writer.
# No config is read and the real HTTP opener is explicitly disabled.
import io

def _forbid_live_transport(*args, **kwargs):
    raise AssertionError("live transport forbidden")

urllib.request.OpenerDirector.open = _forbid_live_transport

def load_config(path):
    assert Path(path).name == ".fastfill_api.env"
    return Config("offline-placeholder", "https://offline.invalid", "gpt-6.1-sol")

class _OfflineResponse(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream"}

def open_response(request, timeout):
    assert request.get_header("User-agent") == "fastfill-evaluation/1.0"
    payload = strict_json(request.data, "invalid_test_payload")
    assert payload["store"] is False and payload["stream"] is True
    manifest = Path(sys.argv[sys.argv.index("--manifest") + 1])
    case = load_manifest(manifest)["cases"][0]
    if case["id"] == "room-0-joint":
        raise urllib.error.URLError("synthetic transport failure")
    scores = {"functionality_score": 7, "layout_score": 6,
              "aesthetics_score": 8, "overall_score": 7.0}
    event = {"type": "response.completed", "response": {
        "model": "gpt-6.1-sol", "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": json.dumps(scores)}]}]}}
    return _OfflineResponse(b"data: " + canonical(event) + b"\\n\\n")

'''


def test_standalone_real_subprocess_with_offline_client_transport(tmp_path):
    import os
    import sys
    root = bundle(tmp_path)
    source, marker, entry = CLIENT.read_text().rpartition('if __name__ == "__main__":')
    assert marker
    offline_client = tmp_path / 'offline_client.py'
    offline_client.write_text(source + OFFLINE_TRANSPORT + marker + entry)
    environment = {k: os.environ[k] for k in ('PATH', 'HOME', 'TMPDIR', 'LANG') if k in os.environ}
    environment['FASTFILL_API_ENV'] = str(tmp_path / '.fastfill_api.env')
    output = tmp_path / 'output'
    command = [sys.executable, '-I', batch.__file__, '--bundle', str(root / 'api-bundle'),
               '--output', str(output), '--client', str(offline_client), '--expected-total', '4',
               '--max-cases', '348', '--timeout', '5']
    process = subprocess.run(command, capture_output=True, text=True, check=False,
                             timeout=30, env=environment)
    assert process.returncode == 1 and process.stderr == ''  # One retained failure.
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['counts'] == dict(selected=4, runnable=4, launched=4, success=3, failed=1)
    assert json.loads(process.stdout)['counts'] == summary['counts']
    batch.validate_summary(summary)
    attempts = sorted((output / 'cases').glob('*/results/*.attempt.json'))
    results = sorted((output / 'cases').glob('*/results/*.result.json'))
    assert len(attempts) == len(results) == 4
    records = [json.loads(path.read_text()) for path in results]
    failed = [row for row in records if row['status'] == 'failed']
    assert len(failed) == 1 and failed[0]['error'] == 'transport_error'
    assert failed[0]['id'] == 'room-0-joint'
    assert summary['provenance']['geometry_manifest_sha256'] == 'd' * 64
    assert not (tmp_path / '.fastfill_api.env').exists()
    before = {p: p.read_bytes() for p in attempts + results}
    repeated = subprocess.run(command, capture_output=True, text=True, check=False,
                              timeout=30, env=environment)
    assert repeated.returncode == 1 and 'validation_or_execution_failed' in repeated.stderr
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize('all_failed', [False, True])
def test_checksum_only_upload_manifest_accepts_actual_visual_bundle(tmp_path, monkeypatch, all_failed):
    import shutil
    from fastfill.comparison.tests.test_visual_pipeline import prepare
    statuses = {f'room-{i}': 'failed' for i in range(2)} if all_failed else None
    visual, prepared, _, _ = prepare(tmp_path, monkeypatch, n=2, statuses=statuses)
    upload = tmp_path / 'upload'
    shutil.copytree(visual / 'api-bundle', upload / 'api-bundle')
    for name in ('api-mapping.json', 'api-failures.json'):
        shutil.copyfile(visual / name, upload / name)
    inventory = {p.relative_to(upload).as_posix(): digest(p.read_bytes())
                 for p in upload.rglob('*') if p.is_file()}
    write(upload / 'MANIFEST.json', {'files': inventory})
    assert set(json.loads((upload / 'MANIFEST.json').read_text())) == {'files'}
    assert digest((upload / 'MANIFEST.json').read_bytes()) != digest((visual / 'MANIFEST.json').read_bytes())
    summary, calls, output = execute(upload / 'api-bundle', tmp_path, monkeypatch)
    assert summary['counts'] == dict(selected=4, runnable=0 if all_failed else 4,
        launched=0 if all_failed else 4, success=0 if all_failed else 4,
        failed=4 if all_failed else 0)
    assert len(calls) == (0 if all_failed else 4)
    assert summary['provenance']['geometry_manifest_sha256'] == prepared['provenance']['geometry_manifest_sha256']
    assert summary['visual_manifest_sha256'] == digest((upload / 'MANIFEST.json').read_bytes())
    assert (output / 'provenance.json').read_bytes() == (upload / 'api-bundle/provenance.json').read_bytes()
    batch.validate_summary(summary)
    # A checksum-only wrapper still enforces every bound file's checksum.
    write(upload / 'MANIFEST.json', {'files': {**inventory, 'api-failures.json': '0' * 64}})
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('client launched'))
    with pytest.raises(ValueError, match='lineage_mismatch'):
        batch.evaluate(upload / 'api-bundle', tmp_path / 'invalid-output', CLIENT)


def test_systems_mode_labels_cap_and_endpoint(tmp_path, monkeypatch):
    root = bundle(tmp_path, n=3, arms=('J', 'P', 'L'))
    calls = []
    monkeypatch.setattr(batch.subprocess, 'run', fake_client(calls, outcomes={'room-2-P': 'fail'}))
    result = batch.evaluate(root, tmp_path / 'output', CLIENT, max_cases=800, arms=('J', 'P', 'L'))
    assert len(calls) == 9 and result['arms'] == ['J', 'P', 'L'] and set(result['by_arm']) == {'J', 'P', 'L'}
    endpoint = result['endpoint']
    assert endpoint['direction'] == 'J_minus_P' and result['paired'][0]['direction'] == 'J_minus_P'
    assert (endpoint['both_complete'], endpoint['selected_pairs']) == (2, 3)
    assert endpoint['mean_difference'] == pytest.approx(4 / 3)  # functionality 8 vs 4 -> room mean +4/3
    assert endpoint['ci95'] == pytest.approx([4 / 3, 4 / 3])
    assert 'Endpoint (J_minus_P' in (tmp_path / 'output/tables.md').read_text()
    batch.validate_summary(result)
    tampered = copy.deepcopy(result)
    tampered['endpoint']['both_complete'] += 1
    with pytest.raises(batch.BatchError, match='endpoint'):
        batch.validate_summary(tampered)
    with pytest.raises(batch.BatchError, match='structure'):
        batch.validate_summary({k: v for k, v in result.items() if k != 'endpoint'})
    assert batch.main(['--bundle', str(root), '--output', str(tmp_path / 'cli'), '--client', str(CLIENT),
                       '--max-cases', '800', '--arms', 'J,P,L']) == 1  # one failed case
    assert json.loads((tmp_path / 'cli/summary.json').read_text())['arms'] == ['J', 'P', 'L']
    with pytest.raises(batch.BatchError, match='invalid_bounds'):
        batch.evaluate(root, tmp_path / 'v1-cap', CLIENT, max_cases=800)
    with pytest.raises(batch.BatchError, match='invalid_bounds'):
        batch.evaluate(root, tmp_path / 'over', CLIENT, max_cases=801, arms=('J', 'P', 'L'))
    with pytest.raises(batch.BatchError, match='arm'):
        batch.evaluate(root, tmp_path / 'subset', CLIENT, max_cases=800, arms=('J', 'P'))


def test_endpoint_resamples_rooms_within_stratum_with_fixed_seed():
    def record(uid, arm, stratum, functionality, status='completed'):
        return dict(uid=uid, arm=arm, stratum=stratum, status=status,
                    scores=dict(functionality_score=functionality, layout_score=5, aesthetics_score=5))
    records = [r for i, (stratum, delta) in enumerate([('a', 0), ('a', 3), ('b', 6)])
               for r in (record(f'u{i}', 'J', stratum, 5 + delta), record(f'u{i}', 'P', stratum, 5))]
    records += [record('u9', 'J', 'b', 9, status='failed'), record('u9', 'P', 'b', 1)]
    endpoint = batch._endpoint(records, 'J', 'P')
    assert (endpoint['both_complete'], endpoint['selected_pairs']) == (3, 4)
    assert endpoint['both_complete_by_stratum'] == {'a': 2, 'b': 1}
    assert endpoint['mean_difference'] == pytest.approx(1.0)  # room-mean deltas 0, 1, 2
    # Stratum b's single room is in every draw, so draws lie in {2/3, 1, 4/3}, never 0.
    assert endpoint['ci95'] == pytest.approx([2 / 3, 4 / 3])
    assert batch._endpoint(records, 'J', 'P') == endpoint
    assert endpoint['bootstrap'] == {'draws': 2000, 'seed': 42, 'resample': 'rooms_within_stratum',
                                     'interval': 'percentile'}
