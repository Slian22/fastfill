import json
from pathlib import Path
import pytest
from fastfill.comparison import final_tables as tables
from fastfill.comparison import judge_batch
from fastfill.comparison import render_assets as render
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison import visual_pipeline


def inputs(tmp_path):
    geo = tmp_path / 'geo'
    geo.mkdir()
    metrics = dict(requests=10, complete_count=8, complete_rate=.8, oor=.1,
        size_iou=.5, size_log_mae=.2, geometry_success_rate=.6)
    summary = dict(status='GEOMETRY_COMPLETE', expected_rooms_per_arm=10,
        paper_strata={'IL3D_3dfront/bedroom': {a: metrics for a in tables.ARMS}},
        arms={a: dict(all=metrics, by_source={'IL3D_3dfront': metrics}, source_macro={}) for a in tables.ARMS},
        paired={}, data_manifest_sha256='a'*64, inference_run_sha256={a:'b'*64 for a in tables.ARMS})
    h.write_json(geo/'summary.json', summary)
    h.write_json(geo/'MANIFEST.json', {'files': h.artifact_manifest(geo)})
    fid = tmp_path/'fid.json'
    h.write_json(fid, dict(metric='FID', extractor='pytorch_fid:0.3.0:pool3', feature_dimension=2048,
        weights_sha256=tables.WEIGHTS_SHA256, strata={'IL3D_3dfront/bedroom':dict(
        counts={'requested':10,'shared_success':5},joint_vs_reference=12.3,placement_vs_reference=22.4)},
        provenance={'provenance':{'geometry_manifest_sha256':h.sha256(geo/'MANIFEST.json')}}))
    judge=tmp_path/'judge.json'
    records = [dict(uid=uid, arm=arm, stratum='IL3D_3dfront/bedroom',
                    status='failed', launched=False)
               for uid in ('a', 'b') for arm in tables.ARMS]
    judged = judge_batch.summarize(records, 'gpt-6.1-sol', runnable=0)
    h.write_json(judge, {**judged, 'provenance': {'geometry_manifest_sha256': h.sha256(geo/'MANIFEST.json')}})
    return geo,fid,judge


def test_tables_preserve_missing_scores_and_different_denominators(tmp_path):
    geo,fid,judge=inputs(tmp_path)
    result=tables.build(geo,fid,judge,tmp_path/'out')
    assert result['paper_style_rows'][0]['FID'] == 12.3
    assert result['paper_style_rows'][0]['FID_shared_n'] == 5
    assert result['paper_style_rows'][0]['OOR_complete_n'] == 8
    assert result['paper_style_rows'][0]['functionality_score'] is None
    assert result['paper_style_rows'][0]['GPT_requested'] == 2
    assert result['paper_style_rows'][0]['GPT_success'] == 0
    assert sum(row['GPT_requested'] for row in result['paper_style_rows']) == 4
    text=(tmp_path/'out/report.md').read_text()
    assert '未测' in text and '不是论文原始测试集' in text
    assert '12.3000' in text
    h.verify_artifacts(tmp_path/'out',json.loads((tmp_path/'out/MANIFEST.json').read_text())['files'])
    with pytest.raises(FileExistsError):tables.build(geo,fid,judge,tmp_path/'out')


def test_rejects_tampered_geometry_and_injected_fid(tmp_path):
    geo,fid,judge=inputs(tmp_path)
    value=json.loads(fid.read_text());value['extractor']='injected';h.write_json(fid,value)
    with pytest.raises(ValueError,match='extractor'):tables.build(geo,fid,judge,tmp_path/'bad1')
    value['extractor']='pytorch_fid:0.3.0:pool3';h.write_json(fid,value)
    (geo/'summary.json').write_text('{}')
    with pytest.raises(ValueError):tables.build(geo,fid,judge,tmp_path/'bad2')


def test_rejects_wrong_judge_and_fid_geometry_lineage(tmp_path):
    geo,fid,judge=inputs(tmp_path)
    h.write_json(judge,{'model':'different','by_stratum':{}})
    with pytest.raises(ValueError,match='judge'):tables.build(geo,fid,judge,tmp_path/'bad1')
    h.write_json(judge,{'model':'gpt-6.1-sol','by_stratum':{},'provenance':{'geometry_manifest_sha256':h.sha256(geo/'MANIFEST.json')}})
    value=json.loads(fid.read_text());value['provenance']['provenance']['geometry_manifest_sha256']='c'*64;h.write_json(fid,value)
    with pytest.raises(ValueError,match='lineage'):tables.build(geo,fid,judge,tmp_path/'bad2')


def test_judge_wrong_run_and_scores_without_observations_rejected(tmp_path):
    geo,fid,judge=inputs(tmp_path)
    value=json.loads(judge.read_text());value['provenance']['geometry_manifest_sha256']='f'*64;h.write_json(judge,value)
    with pytest.raises(ValueError,match='judge lineage'):tables.build(geo,fid,judge,tmp_path/'bad1')
    value['provenance']['geometry_manifest_sha256']=h.sha256(geo/'MANIFEST.json')
    value['by_stratum']={'IL3D_3dfront/bedroom':{'joint':dict(requested=10,success=0,functionality_score=9.5)}}
    h.write_json(judge,value)
    with pytest.raises(ValueError,match='observations'):tables.build(geo,fid,judge,tmp_path/'bad2')


def test_rejects_judge_missing_stratum_counts_instead_of_publishing_zero(tmp_path):
    geo, fid, judge = inputs(tmp_path)
    value = json.loads(judge.read_text())
    value['by_stratum'] = {}
    h.write_json(judge, value)
    with pytest.raises(ValueError):
        tables.build(geo, fid, judge, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('mutation', ['totals', 'score_count'])
def test_rejects_inconsistent_judge_summary(tmp_path, mutation):
    geo, fid, judge = inputs(tmp_path)
    value = json.loads(judge.read_text())
    if mutation == 'totals':
        value['counts']['selected'] += 1
        value['counts']['failed'] += 1
    else:
        value['by_stratum']['IL3D_3dfront/bedroom']['joint']['score_counts']['functionality_score'] = 1
    h.write_json(judge, value)
    with pytest.raises(ValueError):
        tables.build(geo, fid, judge, tmp_path/'bad')


@pytest.mark.parametrize('mutation', ['missing', 'foreign'])
def test_rejects_valid_judge_summary_with_wrong_stratum_coverage(tmp_path, mutation):
    geo, fid, judge = inputs(tmp_path)
    summary = json.loads((geo/'summary.json').read_text())
    key = '/'.join(tables.PAPER_STRATA[1])
    if mutation == 'missing':
        summary['paper_strata'][key] = summary['paper_strata']['IL3D_3dfront/bedroom']
    else:
        summary['paper_strata'] = {key: summary['paper_strata']['IL3D_3dfront/bedroom']}
    h.write_json(geo/'summary.json', summary)
    h.write_json(geo/'MANIFEST.json', {'files': {'summary.json': h.sha256(geo/'summary.json')}})
    value = json.loads(judge.read_text())
    value['provenance']['geometry_manifest_sha256'] = h.sha256(geo/'MANIFEST.json')
    judge_batch.validate_summary(value)  # Internal totals alone cannot detect the wrong cohort.
    h.write_json(judge, value)
    visual = json.loads(fid.read_text())
    visual['provenance']['provenance']['geometry_manifest_sha256'] = h.sha256(geo/'MANIFEST.json')
    h.write_json(fid, visual)
    with pytest.raises(ValueError, match='stratum coverage'):
        tables.build(geo, fid, judge, tmp_path/'bad')


STRATUM = 'IL3D_3dfront/bedroom'


def seal(root):
    files = {k: v for k, v in h.artifact_manifest(root).items() if k != 'MANIFEST.json'}
    h.write_json(root / 'MANIFEST.json', {'files': files})


def lineage(systems):
    return render.json_sha256(visual_pipeline.systems_lineage(visual_pipeline._systems(systems)))


def judge_summary(path, systems, arms=('J', 'P', 'L')):
    scores = {'J': 8, 'P': 5, 'L': 6, 'O5': 7}
    records = [dict(uid=uid, arm=arm, stratum=STRATUM, status='completed', launched=True,
                    scores=dict(functionality_score=scores[arm], layout_score=6, aesthetics_score=7))
               for uid in ('a', 'b') for arm in arms]
    judged = judge_batch.summarize(records, 'gpt-6.1-sol', runnable=len(records), arms=arms)
    h.write_json(path, {**judged, 'provenance': {'geometry_manifest_sha256': lineage(systems)}})


def systems_inputs(tmp_path):
    metrics = dict(requests=10, complete_count=8, complete_rate=.8, oor=.1, size_iou=.5, size_log_mae=.2,
                   geometry_success_rate=.6)
    geo, llm = tmp_path / 'geo', tmp_path / 'llm'
    for root, status, arms in ((geo, 'GEOMETRY_COMPLETE', ('joint', 'placement')),
                               (llm, 'LLM_COMPARISON_COMPLETE', ('llm_harness',))):
        root.mkdir()
        h.write_json(root / 'summary.json', dict(status=status, data_manifest_sha256='a' * 64,
                                                 paper_strata={STRATUM: {a: metrics for a in arms}}))
        for arm in arms:
            (root / f'{arm}-geometry.jsonl').write_text('{}\n')
        seal(root)
    systems = {'J': geo, 'P': geo, 'L': llm}
    fid = tmp_path / 'fid.json'
    h.write_json(fid, dict(metric='FID', extractor='pytorch_fid:0.3.0:pool3', feature_dimension=2048,
        weights_sha256=tables.WEIGHTS_SHA256, systems=['J', 'P', 'L'], strata={STRATUM: dict(
            counts={'requested': 10, 'shared_success': 5}, small_n=True, J_vs_reference=12.3,
            P_vs_reference=22.4, L_vs_reference=30.1, kid={s: {'kid': k, 'se': k / 10}
                                                          for s, k in (('J', .01), ('P', .02), ('L', .03))})},
        provenance={'provenance': {'geometry_manifest_sha256': lineage(systems)}}))
    judge = tmp_path / 'judge.json'
    judge_summary(judge, systems)
    return systems, fid, judge


def test_system_tables_kid_small_n_gpt_endpoint_and_cli(tmp_path):
    systems, fid, judge = systems_inputs(tmp_path)
    result = tables.build_systems(systems, fid, judge, tmp_path / 'out')
    rows = result['system_rows']
    assert len(rows) == 18 and [r['system'] for r in rows[:3]] == ['J', 'P', 'L']
    j, p, agent = rows[:3]
    assert (j['FID'], agent['KID'], agent['KID_SE'], j['FID_shared_n'], j['FID_small_n']) == (12.3, .03, .003, 5, True)
    assert (agent['OOR'], agent['requested'], agent['OOR_complete_n']) == (.1, 10, 8)
    assert j['GPT_room_mean'] == pytest.approx(7) and p['GPT_room_mean'] == pytest.approx(6)
    assert result['gpt_endpoint']['mean_difference'] == pytest.approx(1)
    assert rows[3]['FID'] is None and rows[3]['GPT_requested'] == 0
    text = (tmp_path / 'out/report.md').read_text()
    assert 'N<50' in text and 'J−P' in text and '30.1000' in text and '0.0300 (0.0030)' in text
    h.verify_artifacts(tmp_path / 'out', json.loads((tmp_path / 'out/MANIFEST.json').read_text())['files'])
    args = [arg for label, root in systems.items() for arg in ('--system', f'{label}={root}')]
    tables.main([*args, '--fid', str(fid), '--judge', str(judge), '--out', str(tmp_path / 'cli')])
    assert json.loads((tmp_path / 'cli/results.json').read_text())['status'] == 'MEASURED_SYSTEM_TABLES_PUBLISHED'
    with pytest.raises(SystemExit):
        tables.main([*args, '--geometry', str(systems['J']), '--fid', str(fid), '--judge', str(judge),
                     '--out', str(tmp_path / 'both')])


@pytest.mark.parametrize('mutation,match', [('judged_set', 'judged system set'), ('fid_systems', 'FID system set'),
                                            ('data_manifest', 'evaluation manifest'), ('incomplete', 'complete'),
                                            ('lineage', 'lineage')])
def test_system_tables_fail_closed(tmp_path, mutation, match):
    systems, fid, judge = systems_inputs(tmp_path)
    llm = systems['L']
    summary = json.loads((llm / 'summary.json').read_text())
    if mutation == 'judged_set':
        judge_summary(judge, systems, arms=('J', 'P'))
    if mutation == 'fid_systems':
        h.write_json(fid, {**json.loads(fid.read_text()), 'systems': ['J', 'P']})
    if mutation in ('data_manifest', 'incomplete', 'lineage'):
        summary = {**summary, **{'data_manifest': {'data_manifest_sha256': 'b' * 64},
                                 'incomplete': {'status': 'RUNNING'},
                                 'lineage': {'note': 'changed after visual run'}}[mutation]}
        h.write_json(llm / 'summary.json', summary)
        seal(llm)
    with pytest.raises(ValueError, match=match):
        tables.build_systems(systems, fid, judge, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_actual_systems_chain_visual_fid_judge_tables_share_lineage(tmp_path, monkeypatch):
    """Blender, Inception and the API client are mocked; every lineage check is real."""
    import numpy as np
    from fastfill.comparison import fid as fid_module
    from fastfill.comparison.tests import test_judge_batch as tj, test_visual_pipeline as tv
    ref, geo, evaluation, cases = tv.fixture(tmp_path)
    llm = tv.agent_dir(tmp_path, 'llm', cases, evaluation, fail={'room-1'})
    metrics = dict(requests=4, complete_count=4, oor=.1)
    for root, arms in ((geo, ('joint', 'placement')), (llm, ('llm_harness',))):
        summary = json.loads((root / 'summary.json').read_text())
        tv.write(root / 'summary.json', {**summary, 'status': summary.get('status', 'GEOMETRY_COMPLETE'),
                                         'paper_strata': {STRATUM: {a: metrics for a in arms}}})
        tv.seal(root)
    systems = {'J': geo, 'P': geo, 'L': llm}
    monkeypatch.setattr(tv.pipeline.subprocess, 'run', tv.fake_blender([]))
    prepared = visual_pipeline.prepare_systems(ref, systems, tmp_path / 'visual', '/mock', eval_data=evaluation)
    monkeypatch.setattr(fid_module, 'verify_weights', lambda path: fid_module.WEIGHTS_SHA256)
    features = lambda refs, **_: np.random.default_rng(0).normal(size=(len(refs), 2048))  # noqa: E731
    result = fid_module.run_manifest(tmp_path / 'visual/fid-manifest.json', tmp_path / 'fid.json',
                                     vendor=tmp_path, weights=tmp_path / 'weights', extractor=features)
    assert result['strata'][STRATUM]['counts']['shared_success'] == 3  # L failed room-1
    h.write_json(tmp_path / 'fid.json', {**result, 'extractor': 'pytorch_fid:0.3.0:pool3'})  # mocked model only
    monkeypatch.setattr(judge_batch.subprocess, 'run', tj.fake_client([]))
    judged = judge_batch.evaluate(tmp_path / 'visual/api-bundle', tmp_path / 'judge', tj.CLIENT, max_cases=800,
                                  arms=tuple(prepared['judged']), expected_total=12)
    assert judged['counts']['success'] == 11 and judged['endpoint']['both_complete'] == 4
    out = tables.build_systems(systems, tmp_path / 'fid.json', tmp_path / 'judge/summary.json', tmp_path / 'tables')
    first = {r['system']: r for r in out['system_rows'] if r['stratum'] == STRATUM}
    assert first['L']['GPT_success'] == 3 and first['J']['FID_shared_n'] == 3 and first['L']['KID'] is not None
    assert out['input_sha256']['systems_lineage'] == prepared['provenance']['geometry_manifest_sha256']
