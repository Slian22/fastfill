import json
from pathlib import Path
import pytest
from fastfill.comparison import final_tables as tables
from fastfill.comparison import judge_batch
from fastfill.comparison import train_pair_helpers as h


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
