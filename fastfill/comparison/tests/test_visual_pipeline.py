"""Local synthetic integration tests; Blender is always replaced by a stub."""
import copy
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

from PIL import Image
import pytest

from fastfill.comparison import render_assets as render, visual_cases
from fastfill.comparison import visual_pipeline as pipeline


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v) + '\n' for v in values))


def seal(root, **metadata):
    files = {p.relative_to(root).as_posix(): render.file_sha256(p)
             for p in root.rglob('*') if p.is_file() and p.name != 'MANIFEST.json'}
    write(root / 'MANIFEST.json', {**metadata, 'files': files})


def fixture(tmp_path, n=4, selected=None):
    ref, geo, evaluation = [tmp_path / name for name in ('reference', 'geometry', 'eval')]
    assets = tmp_path / 'assets'
    assets.mkdir(parents=True)
    mesh = assets / 'chair.obj'
    mesh.write_text('mtllib chair.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 1\nusemtl chair\nf 1 2 3\n')
    (assets / 'chair.mtl').write_text('newmtl chair\nKd 1 1 1\nmap_Kd chair.png\n')
    Image.new('RGB', (2, 2), (80, 100, 120)).save(assets / 'chair.png')
    planner, resources = visual_cases._planner()
    cases = []
    for i in range(n):
        case = dict(uid=f'room-{i}', source='IL3D_3dfront', stratum='IL3D_3dfront/bedroom',
                    boundary=[[0, 0], [5, 0], [5, 5], [0, 5]], boundary_type='polygon',
                    height=3, fixed=[], expected_ids=['chair-1'], reference_eligible=True,
                    layout_variant='raw', render_settings=visual_cases.RENDER_SETTINGS,
                    asset_mapping=[dict(prompt_id='chair-1', asset_id='synthetic-chair',
                                        mesh_path=str(mesh))],
                    provenance={'corpus_manifest_sha256': 'a' * 64},
                    layouts={'reference': [dict(id='chair-1', category='chair', size=[1.731, 2, 3],
                         pos=[3.871, 2, 0], yaw=.876, description='SECRET_GT_DESCRIPTION')]})
        cases.append({**case, 'cameras': planner(case)['cameras']})
    uids = [c['uid'] for c in cases]
    selected = uids[:min(n, 29)] if selected is None else selected
    rows(ref / 'reference-cases.jsonl', cases)
    rows(ref / 'eligibility.jsonl', [{'uid': c['uid'], 'eligible': True} for c in cases])
    write(ref / 'resources.json', resources)
    write(ref / 'gpt-selection.json', dict(selected_uids=selected, replacement=False,
        strata={'IL3D_3dfront/bedroom': {'selected_uids': selected}},
        raw_candidates={'strata': {'IL3D_3dfront/bedroom': {'selected_uids': selected}}}))
    seal(ref, expected_uids=uids, provenance={'corpus_manifest_sha256': 'a' * 64},
         render_settings=visual_cases.RENDER_SETTINGS)
    rows(evaluation / 'rooms.jsonl', [{'uid': c['uid'], 'source': c['source'], 'group': 'g'}
                                     for c in cases])
    seal(evaluation, expected_uids=uids, corpus_manifest_sha256='a' * 64)
    write(geo / 'summary.json', {'data_manifest_sha256': render.file_sha256(evaluation / 'MANIFEST.json')})
    for arm in ('joint', 'placement'):
        records = [dict(uid=c['uid'], arm=arm, source=c['source'], group='g', complete=True,
                        raw_layout=copy.deepcopy(c['layouts']['reference'])) for c in cases]
        rows(geo / f'{arm}-geometry.jsonl', records + [dict(uid='outside', complete=False)])
    seal(geo)
    return ref, geo, evaluation, cases


def fake_blender(calls, *, statuses=None, corrupt=None, timeout=False):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert 0 < kwargs['timeout'] <= 86400 and kwargs['check'] is False
        assert kwargs['stderr'] == subprocess.STDOUT
        kwargs['stdout'].write('mock blender\n')
        manifest = Path(command[command.index('--manifest') + 1])
        output = Path(command[command.index('--output') + 1])
        output.mkdir(parents=True, exist_ok=True)
        receipts = []
        for case in json.loads(manifest.read_text())['cases']:
            state = (statuses or {}).get(case['uid'], 'rendered')
            if state == 'missing':
                continue
            receipt = dict(uid=case['uid'], status=state)
            if state == 'rendered':
                images, hashes = {}, {}
                for arm in case['layouts']:
                    images[arm], hashes[arm] = {}, {}
                    for view in ('perspective', 'top'):
                        path = output / case['uid'] / arm / f'{view}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new('RGB', (512, 512), (20, 40, 80)).save(path)
                        images[arm][view], hashes[arm][view] = str(path), render.file_sha256(path)
                receipt = {**receipt, 'images': images, 'image_sha256': hashes,
                    'cameras': case['cameras'], 'input_sha256': render.json_sha256(case),
                    'layout_sha256': {k: render.json_sha256(v) for k, v in case['layouts'].items()},
                    'layout_variant': 'raw'}
                if corrupt:
                    receipt = corrupt(receipt)
            receipts.append(receipt)
        write(output / 'ledger.json', {'cases': receipts, 'config': dict(resolution=512,
              samples=32, threads=int(command[-1]), top=True, denoise=True,
              script_sha256=render.file_sha256(render.__file__))})
        if timeout:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return SimpleNamespace(returncode=1 if statuses else 0)
    return run


def prepare(tmp_path, monkeypatch, *, statuses=None, corrupt=None, timeout=False, n=4):
    ref, geo, evaluation, cases = fixture(tmp_path, n)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_blender(
        calls, statuses=statuses, corrupt=corrupt, timeout=timeout))
    out = tmp_path / 'visual'
    result = pipeline.prepare(ref, geo, out, '/mock/blender', eval_data=evaluation)
    return out, result, calls, cases


def test_prepare_full_chain_and_api_contract(tmp_path, monkeypatch):
    out, result, calls, cases = prepare(tmp_path, monkeypatch)
    assert result['counts'] == {'frozen': 4, 'rendered': 4, 'failed': 0,
                                'gpt_selected': 4, 'api_cases': 8}
    assert len(calls) == 2
    assert all(options['timeout'] == 86400 for _, options in calls)
    assert all('--denoise' in cmd and '--top' in cmd and cmd[-1] == '3' for cmd, _ in calls)
    fid = json.loads((out / 'fid-manifest.json').read_text())
    assert len(fid['cases']) == 4
    assert all(c['failure'] is None for c in fid['cases'])
    assert all(i['view'] == 'perspective' and 'perspective.png' in i['path']
               for c in fid['cases'] for i in c['images'].values())
    api = json.loads((out / 'api-bundle/manifest.json').read_text())
    mapping = json.loads((out / 'api-mapping.json').read_text())
    assert api['expected_model'] == 'gpt-6.1-sol'
    assert len(api['cases']) == 8 and len(mapping) == 8
    for case in api['cases']:
        assert set(case) == {'id', 'prompt_file', 'prompt_sha256', 'images'}
        assert len(case['images']) == 2
        assert mapping[case['id']]['group'] == 'g'
        for entry in case['images']:
            assert render.file_sha256(out / 'api-bundle' / entry['path']) == entry['sha256']
    assert len(list((out / 'api-bundle/images').rglob('*.png'))) == 16
    assert len((out / 'cases.jsonl').read_text().splitlines()) == 4
    assert result['provenance']['corpus_link'] == 'verified'
    _, fid_cases, _ = pipeline.fid._load_manifest(out / 'fid-manifest.json')
    checked, failures = pipeline.fid._inspect_cases(fid_cases, out)
    assert not failures and all(c['candidate'] for c in checked.values())


def test_prompt_category_only_same_for_both_arms(tmp_path, monkeypatch):
    out, _, _, _ = prepare(tmp_path, monkeypatch)
    api = json.loads((out / 'api-bundle/manifest.json').read_text())
    prompts = [(out / 'api-bundle' / c['prompt_file']).read_text() for c in api['cases']]
    assert len(set(prompts)) == 1
    prompt = prompts[0]
    assert 'bedroom' in prompt and 'chair' in prompt
    for forbidden in ('1.731', '3.871', '0.876', 'SECRET_GT_DESCRIPTION', 'synthetic studio', 'bathtub'):
        assert forbidden not in prompt
    assert all(k in prompt for k in ('functionality_score', 'layout_score', 'aesthetics_score', 'overall_score'))


@pytest.mark.parametrize('all_failed', [False, True])
def test_portable_api_provenance_and_complete_bundle_hashes(tmp_path, monkeypatch, all_failed):
    statuses = {f'room-{i}': 'failed' for i in range(4)} if all_failed else None
    out, result, _, _ = prepare(tmp_path, monkeypatch, statuses=statuses)
    provenance = json.loads((out / 'api-bundle/provenance.json').read_text())
    assert provenance == {
        'schema_version': 1,
        'geometry_manifest_sha256': render.file_sha256(tmp_path / 'geometry/MANIFEST.json'),
        'reference_manifest_sha256': render.file_sha256(tmp_path / 'reference/MANIFEST.json'),
        'data_manifest_sha256': render.file_sha256(tmp_path / 'eval/MANIFEST.json'),
        'corpus_manifest_sha256': 'a' * 64, 'corpus_link': 'verified'}
    assert provenance['geometry_manifest_sha256'] == result['provenance']['geometry_manifest_sha256']
    api = json.loads((out / 'api-bundle/manifest.json').read_text())
    assert set(api) == {'schema_version', 'expected_model', 'cases'}
    files = json.loads((out / 'MANIFEST.json').read_text())['files']
    bundle_files = {p.relative_to(out).as_posix() for p in (out / 'api-bundle').rglob('*') if p.is_file()}
    assert {p for p in files if p.startswith('api-bundle/')} == bundle_files
    assert all(render.file_sha256(out / name) == files[name] for name in bundle_files)


def test_identical_api_requests_remain_bound_to_distinct_geometry_runs(tmp_path, monkeypatch):
    first, _, _, _ = prepare(tmp_path, monkeypatch)
    geo = tmp_path / 'geometry'
    summary = json.loads((geo / 'summary.json').read_text())
    write(geo / 'summary.json', {**summary, 'inference_run_sha256': {'joint': 'f' * 64}})
    seal(geo)
    second = tmp_path / 'second'
    pipeline.prepare(tmp_path / 'reference', geo, second, '/mock', eval_data=tmp_path / 'eval')
    assert (first / 'api-bundle/manifest.json').read_bytes() == (second / 'api-bundle/manifest.json').read_bytes()
    old = json.loads((first / 'api-bundle/provenance.json').read_text())
    new = json.loads((second / 'api-bundle/provenance.json').read_text())
    assert old['geometry_manifest_sha256'] != new['geometry_manifest_sha256']
    assert new['geometry_manifest_sha256'] == render.file_sha256(geo / 'MANIFEST.json')
    hashes = [json.loads((root / 'MANIFEST.json').read_text())['files']['api-bundle/provenance.json']
              for root in (first, second)]
    assert hashes[0] != hashes[1]


@pytest.mark.parametrize('root_name', ['reference', 'geometry'])
def test_input_hash_mismatch_before_render(tmp_path, monkeypatch, root_name):
    ref, geo, _, _ = fixture(tmp_path)
    path = ref / 'reference-cases.jsonl' if root_name == 'reference' else geo / 'joint-geometry.jsonl'
    path.write_text(path.read_text() + ' ')
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('render launched'))
    with pytest.raises(ValueError, match='hash mismatch'):
        pipeline.prepare(ref, geo, tmp_path / 'out', '/mock')


@pytest.mark.parametrize('kind', ['hash', 'camera', 'view', 'layout'])
def test_invalid_image_or_receipt_excludes_entire_pair(tmp_path, monkeypatch, kind):
    def corrupt(receipt):
        value = copy.deepcopy(receipt)
        if kind == 'hash':
            value['image_sha256']['joint']['top'] = '0' * 64
        if kind == 'camera':
            value['cameras']['perspective']['location'][0] += 1
        if kind == 'view':
            del value['images']['placement']['perspective']
        if kind == 'layout':
            value['layout_sha256']['joint'] = '0' * 64
        return value
    out, result, _, _ = prepare(tmp_path, monkeypatch, corrupt=corrupt)
    assert result['counts']['rendered'] == 0 and result['counts']['failed'] == 4
    assert json.loads((out / 'api-bundle/manifest.json').read_text())['cases'] == []
    assert all(c['failure'] and c['images'] == {} for c in json.loads((out / 'fid-manifest.json').read_text())['cases'])


def test_partial_timeout_preserves_success_and_distinct_failures(tmp_path, monkeypatch):
    out, result, calls, _ = prepare(tmp_path, monkeypatch, timeout=True, statuses={
        'room-1': 'missing', 'room-2': 'pending', 'room-3': 'failed'})
    assert result['counts']['rendered'] == 1 and len(calls) == 2
    ledger = json.loads((out / 'ledger.json').read_text())['cases']
    assert [c['status'] for c in ledger] == ['rendered', 'missing', 'pending', 'failed']
    assert len((out / 'failures.jsonl').read_text().splitlines()) == 3
    assert len(json.loads((out / 'api-failures.json').read_text())) == 6
    assert all(c['status'] == 'timeout' for c in result['chunks'])


def test_frozen_selection_no_score_fill_and_stable_global_order(tmp_path, monkeypatch):
    ref, geo, _, _ = fixture(tmp_path, n=31)
    selected = json.loads((ref / 'gpt-selection.json').read_text())['selected_uids']
    calls = []
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_blender(calls, statuses={'room-0': 'failed'}))
    result = pipeline.prepare(ref, geo, tmp_path / 'out', '/mock', workers=1)
    api = json.loads((tmp_path / 'out/api-bundle/manifest.json').read_text())
    mapping = json.loads((tmp_path / 'out/api-mapping.json').read_text())
    assert result['counts']['gpt_selected'] == 29 and result['counts']['api_cases'] == 56
    assert {r['uid'] for r in mapping.values()} == set(selected)
    assert [c['id'] for c in api['cases']] == sorted(c['id'] for c in api['cases'])
    assert result['provenance']['corpus_link'] == 'unverified_without_eval_data'


def test_optional_eval_provenance_mismatch(tmp_path, monkeypatch):
    ref, geo, evaluation, _ = fixture(tmp_path)
    seal(evaluation, expected_uids=['wrong'], corpus_manifest_sha256='b' * 64)
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('render launched'))
    with pytest.raises(ValueError, match='evaluation|corpus'):
        pipeline.prepare(ref, geo, tmp_path / 'out', '/mock', eval_data=evaluation)


@pytest.mark.parametrize('kwargs', [{'workers': 3}, {'threads': 4}, {'workers': True},
                                   {'timeout_seconds': 0}, {'timeout_seconds': 86401},
                                   {'timeout_seconds': float('inf')}, {'timeout_seconds': True}])
def test_resource_bounds(tmp_path, kwargs):
    with pytest.raises(ValueError):
        pipeline.prepare(tmp_path, tmp_path, tmp_path / 'out', '/mock', **kwargs)


def test_cli_timeout_override_and_default(tmp_path, monkeypatch, capsys):
    calls = []
    def fake_prepare(**kwargs):
        calls.append(kwargs)
        return {'status': 'prepared', 'counts': {}}
    monkeypatch.setattr(pipeline, 'prepare', fake_prepare)
    argv = ['prepare', '--reference-dir', str(tmp_path), '--geometry-dir', str(tmp_path),
            '--out', str(tmp_path / 'out'), '--blender-path', '/mock']
    assert pipeline.main(argv) == 0
    assert calls[-1]['timeout_seconds'] == 86400
    assert pipeline.main(argv + ['--timeout-seconds', '7200', '--eval-data', str(tmp_path)]) == 0
    assert calls[-1]['timeout_seconds'] == 7200
    assert calls[-1]['eval_data'] == tmp_path
    assert all(json.loads(line)['status'] == 'prepared' for line in capsys.readouterr().out.splitlines())


def test_missing_geometry_and_reference_failure_stay_in_all_ledgers(tmp_path, monkeypatch):
    ref, geo, _, cases = fixture(tmp_path)
    cases = [{**c, 'reference_eligible': False, 'input_error': 'no_asset: fixed'}
             if c['uid'] == 'room-1' else c for c in cases]
    rows(ref / 'reference-cases.jsonl', cases)
    rows(ref / 'eligibility.jsonl', [{'uid': c['uid'], 'eligible': c['reference_eligible']} for c in cases])
    seal(ref, **{k: v for k, v in json.loads((ref / 'MANIFEST.json').read_text()).items() if k != 'files'})
    rows(geo / 'placement-geometry.jsonl', [])
    seal(geo)
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('render launched'))
    out = tmp_path / 'out'
    result = pipeline.prepare(ref, geo, out, '/mock')
    assert result['counts']['failed'] == 4 and result['chunks'] == []
    assert len((out / 'cases.jsonl').read_text().splitlines()) == 4
    ledger = json.loads((out / 'ledger.json').read_text())['cases']
    assert ledger[0]['failure']['errors']['placement'] == 'missing_output'
    assert ledger[1]['failure']['errors']['reference'] == 'no_asset: fixed'
    assert len(json.loads((out / 'api-failures.json').read_text())) == 8


def test_two_processes_concurrent_one_chunk_each_no_retries(tmp_path, monkeypatch):
    ref, geo, _, _ = fixture(tmp_path)
    calls, barrier = [], threading.Barrier(2)
    backend = fake_blender(calls)
    def concurrent(command, **kwargs):
        barrier.wait(timeout=5)
        return backend(command, **kwargs)
    monkeypatch.setattr(pipeline.subprocess, 'run', concurrent)
    result = pipeline.prepare(ref, geo, tmp_path / 'out', '/mock', timeout_seconds=17)
    assert len(calls) == 2
    assert sorted(len(c['expected_uids']) for c in result['chunks']) == [2, 2]
    assert all(c['timeout_seconds'] == 17 and c['attempts'] == 1 for c in result['chunks'])


def test_launch_failure_keeps_requested_denominators(tmp_path, monkeypatch):
    ref, geo, _, _ = fixture(tmp_path)
    def missing(*args, **kwargs):
        raise FileNotFoundError('mock executable missing')
    monkeypatch.setattr(pipeline.subprocess, 'run', missing)
    result = pipeline.prepare(ref, geo, tmp_path / 'out', '/mock')
    assert result['counts']['failed'] == 4 and result['counts']['api_cases'] == 0
    assert all(c['status'] == 'launch_failed' for c in result['chunks'])
    with pytest.raises(FileExistsError):
        pipeline.prepare(ref, geo, tmp_path / 'out', '/mock')


@pytest.mark.parametrize('kind', ['config', 'null_config', 'duplicate', 'foreign', 'truncated'])
def test_malformed_chunk_receipts_fail_closed(tmp_path, monkeypatch, kind):
    ref, geo, _, _ = fixture(tmp_path)
    backend = fake_blender([])
    def broken(command, **kwargs):
        result = backend(command, **kwargs)
        path = Path(command[command.index('--output') + 1]) / 'ledger.json'
        ledger = json.loads(path.read_text())
        if kind == 'truncated':
            path.write_text('{')
            return result
        if kind == 'config':
            ledger['config']['samples'] = 8
        if kind == 'null_config':
            ledger['config'] = None
        if kind == 'duplicate':
            ledger['cases'].append(ledger['cases'][0])
        if kind == 'foreign':
            ledger['cases'].append({'uid': 'foreign', 'status': 'rendered'})
        write(path, ledger)
        return result
    monkeypatch.setattr(pipeline.subprocess, 'run', broken)
    result = pipeline.prepare(ref, geo, tmp_path / 'out', '/mock')
    assert result['counts']['failed'] == 4
    failures = json.loads((tmp_path / 'out/api-failures.json').read_text())
    assert all('invalid_chunk_ledger' in r['failure']['reason'] for r in failures)


@pytest.mark.parametrize('kind', ['corpus', 'uids'])
def test_eval_snapshot_verifies_corpus_and_frozen_uid_membership(tmp_path, monkeypatch, kind):
    ref, geo, evaluation, _ = fixture(tmp_path)
    manifest = json.loads((evaluation / 'MANIFEST.json').read_text())
    if kind == 'corpus':
        manifest['corpus_manifest_sha256'] = 'b' * 64
    else:
        manifest['expected_uids'] = ['foreign']
    write(evaluation / 'MANIFEST.json', manifest)
    write(geo / 'summary.json', {'data_manifest_sha256': render.file_sha256(evaluation / 'MANIFEST.json')})
    seal(geo)
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('render launched'))
    with pytest.raises(ValueError, match='corpus|UID'):
        pipeline.prepare(ref, geo, tmp_path / 'out', '/mock', eval_data=evaluation)


def test_selection_cannot_replace_failed_uid(tmp_path, monkeypatch):
    ref, geo, _, _ = fixture(tmp_path, n=31)
    value = json.loads((ref / 'gpt-selection.json').read_text())
    value['selected_uids'][0] = 'room-30'
    value['strata']['IL3D_3dfront/bedroom']['selected_uids'][0] = 'room-30'
    write(ref / 'gpt-selection.json', value)
    seal(ref, **{k: v for k, v in json.loads((ref / 'MANIFEST.json').read_text()).items() if k != 'files'})
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('render launched'))
    with pytest.raises(ValueError, match='first 29'):
        pipeline.prepare(ref, geo, tmp_path / 'out', '/mock')


def test_cli_reports_fatal_validation(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        pipeline.main(['prepare', '--reference-dir', str(tmp_path), '--geometry-dir', str(tmp_path),
                       '--out', str(tmp_path / 'out'), '--blender-path', '/mock', '--workers', '3'])
    assert exc.value.code == 2
    assert 'workers must be' in capsys.readouterr().err


def test_output_cannot_modify_input_snapshot(tmp_path):
    ref, geo, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='outside input snapshots'):
        pipeline.prepare(ref, geo, ref / 'out', '/mock')
    assert not (ref / 'out').exists()


def test_all_1009_frozen_cases_and_174_selection_accounted_without_render(tmp_path, monkeypatch):
    ref, geo, _, cases = fixture(tmp_path, n=1009)
    strata = visual_cases.STRATA
    cases = [{**c, 'source': strata[i % 6].split('/')[0], 'stratum': strata[i % 6],
              'reference_eligible': i >= 4, **({'input_error': 'no_asset: fixed'} if i < 4 else {})}
             for i, c in enumerate(cases)]
    by_uid = {c['uid']: c for c in cases}
    rows(ref / 'reference-cases.jsonl', cases)
    rows(ref / 'eligibility.jsonl', [{'uid': c['uid'], 'eligible': c['reference_eligible']} for c in cases])
    populations = {s: {'selected_uids': [c['uid'] for c in cases if c['stratum'] == s][:29]} for s in strata}
    selected = [u for value in populations.values() for u in value['selected_uids']]
    write(ref / 'gpt-selection.json', {'selected_uids': selected, 'strata': populations,
          'raw_candidates': {'strata': populations}, 'replacement': False})
    seal(ref, **{k: v for k, v in json.loads((ref / 'MANIFEST.json').read_text()).items() if k != 'files'})
    for arm in ('joint', 'placement'):
        path = geo / f'{arm}-geometry.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        rows(path, [{**r, 'source': by_uid[r['uid']]['source']} if r['uid'] in by_uid else r for r in records])
    seal(geo)
    calls = []
    def unavailable(command, **kwargs):
        calls.append(command)
        raise FileNotFoundError('mock backend')
    monkeypatch.setattr(pipeline.subprocess, 'run', unavailable)
    out = tmp_path / 'out'
    result = pipeline.prepare(ref, geo, out, '/mock')
    assert result['counts'] == {'frozen': 1009, 'rendered': 0, 'failed': 1009,
                                'gpt_selected': 174, 'api_cases': 0}
    assert len(calls) == 2 and sum(len(c['expected_uids']) for c in result['chunks']) == 1005
    assert len(json.loads((out / 'api-mapping.json').read_text())) == 348
    assert len(json.loads((out / 'api-failures.json').read_text())) == 348
    assert len(json.loads((out / 'fid-manifest.json').read_text())['cases']) == 1009


def agent_dir(tmp_path, name, cases, evaluation, fail=(), drop=()):
    root = tmp_path / name
    write(root / 'summary.json', {'status': 'LLM_COMPARISON_COMPLETE',
                                  'data_manifest_sha256': render.file_sha256(evaluation / 'MANIFEST.json')})
    rows(root / 'llm_harness-geometry.jsonl', [dict(
        uid=c['uid'], arm='llm_harness', source=c['source'], group='g', complete=c['uid'] not in fail,
        raw_layout=copy.deepcopy(c['layouts']['reference'])) for c in cases if c['uid'] not in drop])
    seal(root)
    return root


def test_prepare_systems_renders_every_system_and_keeps_per_system_failures(tmp_path, monkeypatch):
    ref, geo, evaluation, cases = fixture(tmp_path)
    llm = agent_dir(tmp_path, 'llm', cases, evaluation, fail={'room-1'}, drop={'room-2'})
    o5, o1a = (agent_dir(tmp_path, name, cases, evaluation) for name in ('o5', 'o1a'))
    calls = []
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_blender(calls))
    out = tmp_path / 'visual'
    argv = ['prepare-systems', '--reference-dir', str(ref), '--out', str(out), '--blender-path', '/mock',
            '--eval-data', str(evaluation)]
    for label, root in (('J', geo), ('P', geo), ('L', llm), ('O5', o5), ('O1a', o1a)):
        argv += ['--system', f'{label}={root}']
    assert pipeline.main(argv) == 0
    result = json.loads((out / 'preparation.json').read_text())
    assert result['judged'] == ['J', 'P', 'L', 'O5']
    assert result['counts']['system_rendered'] == {'J': 4, 'P': 4, 'L': 2, 'O1a': 4, 'O5': 4}
    assert result['counts']['rendered'] == 4 and result['counts']['failed'] == 0
    lineage = pipeline.systems_lineage(pipeline._systems({'J': geo, 'P': geo, 'L': llm, 'O1a': o1a, 'O5': o5}))
    assert result['systems'] == lineage
    assert (lineage['J']['arm'], lineage['P']['arm'], lineage['L']['arm']) == ('joint', 'placement', 'llm_harness')
    provenance = json.loads((out / 'api-bundle/provenance.json').read_text())
    assert provenance['geometry_manifest_sha256'] == render.json_sha256(lineage)
    data, fid_cases, _ = pipeline.fid._load_manifest(out / 'fid-manifest.json')
    assert data['systems'] == ['J', 'P', 'L', 'O1a', 'O5']
    checked, failures = pipeline.fid._inspect_cases(fid_cases, out, pipeline.fid._arms(data))
    assert [u for u, c in checked.items() if c['candidate']] == ['room-0', 'room-3']
    assert {(f['uid'], f['arm']) for f in failures} == {('room-1', 'L'), ('room-2', 'L')}
    mapping = json.loads((out / 'api-mapping.json').read_text())
    api = json.loads((out / 'api-bundle/manifest.json').read_text())
    api_failures = json.loads((out / 'api-failures.json').read_text())
    assert len(mapping) == 16 and len(api['cases']) == 14 and result['counts']['api_cases'] == 14
    assert {m['arm'] for m in mapping.values()} == {'J', 'P', 'L', 'O5'}
    assert {(f['uid'], f['arm'], f['failure']['stage']) for f in api_failures} == {
        ('room-1', 'L', 'attachment'), ('room-2', 'L', 'attachment')}
    assert all(len(c['images']) == 2 for c in api['cases'])


def test_prepare_systems_fails_closed_on_labels_and_mixed_evaluation_data(tmp_path, monkeypatch):
    ref, geo, evaluation, cases = fixture(tmp_path)
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('render launched'))
    for systems in ({'J': geo}, {'J': geo, 'P': geo, 'X': geo}, {'J': geo, 'P': geo, 'o5': geo}):
        with pytest.raises(ValueError, match='systems'):
            pipeline.prepare_systems(ref, systems, tmp_path / 'out', '/mock')
    llm = agent_dir(tmp_path, 'llm', cases, evaluation)
    write(llm / 'summary.json', {'data_manifest_sha256': 'f' * 64})
    seal(llm)
    with pytest.raises(ValueError, match='evaluation manifest'):
        pipeline.prepare_systems(ref, {'J': geo, 'P': geo, 'L': llm}, tmp_path / 'out', '/mock')
    assert not (tmp_path / 'out').exists()
    assert pipeline.judged_systems(['J', 'P', 'O4', 'O5', 'O1a']) == ['J', 'P', 'O5']
    assert pipeline.judged_systems(['J', 'P', 'L', 'O4']) == ['J', 'P', 'L', 'O4']
