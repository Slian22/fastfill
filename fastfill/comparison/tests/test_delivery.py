"""Delivery package from the selected report's amendment-005 decision (amendment 010 fixture chain; no network, no jobs)."""
import json
import shutil

import pytest

from fastfill.comparison import delivery as D
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests.test_stage_table import BROKEN, build_selected, build_world, reseal


@pytest.fixture(scope='module')
def worlds(tmp_path_factory):
    """J selected e1-dpo2 and P selected sft. Clean J -> the rule delivers J (a DPO candidate); J broken in three
    rooms -> P (SFT). Read-only: receipts carry absolute run paths, as on the cluster."""
    return {arm: build_selected(build_world(tmp_path_factory.mktemp('delivery-' + arm)), broken=BROKEN if arm == 'P' else None)
            for arm in 'JP'}


def _results(run):
    return json.loads((run / 'report-selected' / 'results.json').read_text())


def _card(out):
    h.verify_artifacts(out, json.loads((out / 'MANIFEST.json').read_text())['files'])
    return (out / 'MODEL_CARD.md').read_text()


def test_dpo_candidate_package_lists_merged_model_policy_and_dev_selection(worlds, tmp_path):
    run = worlds['J']
    value = D.build(run, run / 'report-selected', tmp_path / 'pkg')
    results = _results(run)
    assert (value['chosen_arm'], value['candidate'], value['decision']) == ('J', 'e1-dpo2', results['delivery'])
    receipt = json.loads((run / 'dpo-002' / 'joint' / 'dpo2-merged.json').read_text())
    parent = json.loads((run / 'dpo-002' / 'joint' / 'dpo1-merged.json').read_text())
    assert value['model']['model_sha256'] == receipt['model_sha256'] and value['model']['merge_changed_fraction'] == .15
    assert value['model']['lineage_receipt_sha256'] == h.sha256(run / 'dpo-002' / 'joint' / 'dpo2-merged.json')
    assert value['evaluated_policy']['lora'].startswith('unmerged_fp32') and value['size_prior'] is None
    assert value['evaluated_policy']['training_base_sha256'] == parent['model_sha256']
    copies = {c['dest']: c for c in value['files_to_copy']}
    assert copies['training-base/model.safetensors']['sha256'] == parent['model']['files']['model.safetensors']
    assert copies['adapter/adapter_model.safetensors']['sha256'] == receipt['adapter']['files']['adapter_model.safetensors']
    assert {'model/model.safetensors', 'lineage/joint-sft-merged.json', 'lineage/joint-dpo2-merged.json',
            'report-selected/results.json', 'dev-selection/selection.json'} <= set(copies)
    assert copies['lineage/joint-sft-merged.json']['source'].endswith('dpo-001/joint/sft-merged.json')
    assert value['dev_selection']['selected'] == 'e1-dpo2' and value['dev_selection']['tied'] == ['e1-dpo2']
    out = tmp_path / 'pkg'
    assert len((out / 'SHA256SUMS').read_text().splitlines()) == len(value['files_to_copy'])
    card = _card(out)
    t1 = {r['endpoint'].split()[0]: r for r in results['tables']['T1']}
    assert f"{t1['(a)']['J']:.4f}" in card and D._fmt(t1['(b)']['delta'], True) in card
    assert '# FastFill unknown-size layout model: J-selected (e1-dpo2)' in card and '| P-selected (sft) |' in card
    assert 'Evaluated as the exact trained policy' in card and '`training-base/`' in card and 'J needs no size prior' in card
    assert 'selected **e1-dpo2**' in card and value['dev_selection']['rule'] in card and D.RULE in card
    assert '| e1-dpo2 | 0.9000 |' in card and 'The other arm selected sft' in card
    assert 'secondary, not confirmatory' in card and 'primary (confirmatory) comparison stays the same-stage matched SFT' in card
    assert '| Holm p (within (a), (b); secondary) |' in card and 'dpo2 kept 2150 of 2200 positives' in card
    assert 'Human user rating (UR): not measured' in card and 'GPT judge: not measured' in card
    with pytest.raises(FileExistsError):
        D.build(run, run / 'report-selected', out)


def test_sft_package_lists_sft_merge_adapter_base_and_prior(worlds, tmp_path):
    run = worlds['P']
    value = D.build(run, run / 'report-selected', tmp_path / 'pkg')
    assert (value['chosen_arm'], value['candidate'], value['model']['stage']) == ('P', 'sft', 'sft')
    receipt = json.loads((run / 'dpo-001' / 'placement' / 'sft-merged.json').read_text())
    assert value['model']['model_sha256'] == receipt['model_sha256'] and value['evaluated_policy']['lora'] == 'merge_and_unload_bf16'
    assert value['evaluated_policy']['adapter_sha256'] == json.loads(
        (run / 'infer-placement-002' / 'run_receipt.json').read_text())['adapter_sha256']
    assert value['size_prior']['sha256'] == h.sha256(run / 'main-data-v1' / 'size-priors.json')
    copies = {c['dest'] for c in value['files_to_copy']}
    assert {'model/model.safetensors', 'sft-adapter/adapter_model.safetensors', 'base/model.safetensors',
            'size-priors.json', 'lineage/placement-sft-merged.json'} <= copies and not any('dpo' in c for c in copies)
    card = _card(tmp_path / 'pkg')
    assert '# FastFill unknown-size layout model: P-selected (sft)' in card and 'merged at runtime on GPU' in card
    assert "dpo-001's SFT merge (the DPO1 training base)" in card and '`sft-adapter/`' in card and '`base/`' in card
    assert 'Recipe: SFT only' in card and 'predict_sizes(features, prior)' in card and 'n/a (no DPO stage)' in card
    requests = [json.loads(line) for line in (run / 'eval-data-v1' / 'placement' / 'requests.jsonl').read_text().splitlines()]
    shortest = min(requests, key=lambda r: (len(r['messages'][1]['content']), r['uid']))
    assert value['example_request_uid'] == shortest['uid'] and shortest['messages'][1]['content'] in card


def _drop_decision(run):
    results = _results(run)
    results.pop('delivery')
    reseal(run / 'report-selected', 'results.json', results)


def _flip_decision(run):
    results = _results(run)
    reseal(run / 'report-selected', 'results.json', {**results, 'delivery': {**results['delivery'], 'delivered': 'J'}})


@pytest.mark.parametrize('tamper, match', [
    (_drop_decision, 'no delivery decision'),
    (_flip_decision, 'disagrees with the amendment-005 rule'),
    (lambda run: (run / 'main-data-v1' / 'size-priors.json').write_text('{}\n'), 'frozen size-priors.json'),
    (lambda run: shutil.rmtree(run / 'report-selected') or shutil.copytree(run / 'report-dpo2', run / 'report-selected'),
     'only the report of stage selected')])
def test_refuses_without_a_reproducible_selected_decision(tmp_path, tamper, match):
    run = build_selected(build_world(tmp_path), broken=BROKEN)  # Delivers P (SFT, needs the prior).
    tamper(run)
    with pytest.raises(ValueError, match=match):
        D.build(run, run / 'report-selected', tmp_path / 'pkg')
    assert not (tmp_path / 'pkg').exists()
