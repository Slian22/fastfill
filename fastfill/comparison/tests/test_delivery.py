"""Delivery package from the DPO2 report's amendment-005 decision (fixture DPO chain; no network, no jobs)."""
import json
import shutil

import pytest

from fastfill.comparison import delivery as D
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests.test_stage_table import build_world, reseal


@pytest.fixture(scope='module')
def templates(tmp_path_factory):
    """DPO2 J broken in three rooms -> the rule delivers P; clean DPO2 -> J."""
    return {arm: build_world(tmp_path_factory.mktemp('delivery-' + arm), broken_dpo2=arm == 'P') for arm in 'JP'}


@pytest.fixture
def world(templates, tmp_path):
    return {arm: shutil.copytree(root, tmp_path / arm) for arm, root in templates.items()}


def _results(run):
    return json.loads((run / 'report-dpo2' / 'results.json').read_text())


def test_p_package_lists_prior_model_and_report(world, tmp_path):
    run = world['P']
    value = D.build(run, run / 'report-dpo2', tmp_path / 'pkg')
    results = _results(run)
    assert value['chosen_arm'] == 'P' and value['decision'] == results['delivery']
    receipt = json.loads((run / 'dpo-001' / 'placement' / 'dpo2-merged.json').read_text())
    assert value['model']['model_sha256'] == receipt['model_sha256'] and value['model']['merge_changed_fraction'] == .15
    assert value['model']['base_revision'] == h.REVISION and value['evaluated_policy']['lora'].startswith('unmerged_fp32')
    assert value['size_prior']['sha256'] == h.sha256(run / 'main-data-v1' / 'size-priors.json')
    dests = {c['dest'] for c in value['files_to_copy']}
    assert {'size-priors.json', 'model/model.safetensors', 'lineage/placement-dpo2-merged.json',
            'report-dpo2/results.json', 'report-dpo2/MANIFEST.json'} <= dests
    out = tmp_path / 'pkg'
    h.verify_artifacts(out, json.loads((out / 'MANIFEST.json').read_text())['files'])
    assert len((out / 'SHA256SUMS').read_text().splitlines()) == len(value['files_to_copy'])
    card = (out / 'MODEL_CARD.md').read_text()
    t1 = {r['endpoint'].split()[0]: r for r in results['tables']['T1']}
    assert f"{t1['(a)']['J']:.4f}" in card and D._fmt(t1['(b)']['delta'], True) in card
    assert 'P-DPO2' in card and "predict_sizes(features, prior)" in card and 'GPT judge: not measured' in card
    assert 'Human user rating (UR): not measured' in card and 'exact policy' in card
    requests = [json.loads(line) for line in (run / 'eval-data-v1' / 'placement' / 'requests.jsonl').read_text().splitlines()]
    shortest = min(requests, key=lambda r: (len(r['messages'][1]['content']), r['uid']))
    assert value['example_request_uid'] == shortest['uid'] and shortest['messages'][1]['content'] in card
    with pytest.raises(FileExistsError):
        D.build(run, run / 'report-dpo2', out)


def test_j_package_needs_no_prior(world, tmp_path):
    run = world['J']
    value = D.build(run, run / 'report-dpo2', tmp_path / 'pkg')
    assert value['chosen_arm'] == 'J' and value['size_prior'] is None
    assert not any(c['dest'] == 'size-priors.json' for c in value['files_to_copy'])
    assert 'J needs no size prior' in (tmp_path / 'pkg' / 'MODEL_CARD.md').read_text()


def _drop_decision(run):
    results = _results(run)
    results.pop('delivery')
    reseal(run / 'report-dpo2', 'results.json', results)


def _flip_decision(run):
    results = _results(run)
    reseal(run / 'report-dpo2', 'results.json', {**results, 'delivery': {**results['delivery'], 'delivered': 'J'}})


@pytest.mark.parametrize('tamper, match', [
    (_drop_decision, 'no delivery decision'),
    (_flip_decision, 'disagrees with the amendment-005 rule'),
    (lambda run: (run / 'main-data-v1' / 'size-priors.json').write_text('{}\n'), 'frozen size-priors.json'),
    (lambda run: shutil.rmtree(run / 'report-dpo2') or shutil.copytree(run / 'report-dpo1', run / 'report-dpo2'),
     'not built on geometry-v4')])
def test_refuses_without_a_reproducible_dpo2_decision(world, tmp_path, tamper, match):
    run = world['P']
    tamper(run)
    with pytest.raises(ValueError, match=match):
        D.build(run, run / 'report-dpo2', tmp_path / 'pkg')
    assert not (tmp_path / 'pkg').exists()
