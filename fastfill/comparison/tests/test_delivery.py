"""Delivery package from the selected report's amendment-005 decision (amendment 010 fixture chain; no network, no jobs)."""
import json
import shutil
import subprocess

import pytest

from fastfill.comparison import delivery as D
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_stage_table as ST
from fastfill.comparison.tests.test_stage_table import BROKEN, build_selected, build_world, reseal


@pytest.fixture(scope='module', autouse=True)
def implementation():
    """The fixture eval data records the request code that built its prompts, as eval_data.py does."""
    def recorded(root, *args, **kwargs):
        root = original(root, *args, **kwargs)
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        h.write_json(root / 'MANIFEST.json', {**manifest, 'implementation_sha256': {
            k: h.sha256(D.CODE_ROOT / 'fastfill' / k) for k in D.IMPLEMENTATION}})
        return root
    original = ST.eval_data
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ST, 'eval_data', recorded)
        yield


@pytest.fixture(scope='module')
def worlds(tmp_path_factory, implementation):
    """J selected e1-dpo2 and P selected sft. Clean J -> the rule delivers J (a DPO candidate); J broken in three
    rooms -> P (SFT). Read-only: receipts carry absolute run paths, as on the cluster."""
    return {arm: build_selected(build_world(tmp_path_factory.mktemp('delivery-' + arm)), broken=BROKEN if arm == 'P' else None)
            for arm in 'JP'}


def _results(run):
    return json.loads((run / 'report-selected' / 'results.json').read_text())


def _card(out):
    h.verify_artifacts(out, json.loads((out / 'MANIFEST.json').read_text())['files'])
    return (out / 'MODEL_CARD.md').read_text()


def test_dpo_candidate_package_leads_with_the_evaluated_policy_and_pins_code(worlds, tmp_path):
    run = worlds['J']
    value = D.build(run, run / 'report-selected', tmp_path / 'pkg')
    results = _results(run)
    assert (value['chosen_arm'], value['candidate'], value['decision']) == ('J', 'e1-dpo2', results['delivery'])
    assert value['status'] == 'MANIFEST_READY' and value['merged_check'] is None and not value['model']['listed']
    assert value['model']['delivered_as'].startswith(  # Amendment 011 §5: no check, no merge.
        'not delivered: amendment 011 §5 merged-weight check not run (15.00 % of target weights changed')
    assert [c['dest'].split('/')[0] for c in value['files_to_copy'][:3]] == ['adapter', 'training-base', 'training-base']
    assert value['agents'] == {'included': [], 'excluded': {}, 'not_run': {}}  # The report's own (no agents here).
    code = value['code']
    assert code['sha256'] == {n: h.sha256(D.CODE_ROOT / n) for n in D.CODE}
    assert code['git_dirty'] is (code['git_commit'] is None)  # A commit only for clean request code.
    assert code['versions'] is None and 'run_receipt.json' in code['versions_source']
    receipt = json.loads((run / 'dpo-002' / 'joint' / 'dpo2-merged.json').read_text())
    parent = json.loads((run / 'dpo-002' / 'joint' / 'dpo1-merged.json').read_text())
    assert value['model']['model_sha256'] == receipt['model_sha256'] and value['model']['merge_changed_fraction'] == .15
    assert value['model']['lineage_receipt_sha256'] == h.sha256(run / 'dpo-002' / 'joint' / 'dpo2-merged.json')
    assert value['evaluated_policy']['lora'].startswith('unmerged_fp32') and value['size_prior'] is None
    assert value['evaluated_policy']['training_base_sha256'] == parent['model_sha256']
    copies = {c['dest']: c for c in value['files_to_copy']}
    assert copies['training-base/model.safetensors']['sha256'] == parent['model']['files']['model.safetensors']
    assert copies['adapter/adapter_model.safetensors']['sha256'] == receipt['adapter']['files']['adapter_model.safetensors']
    assert {'lineage/joint-sft-merged.json', 'lineage/joint-dpo2-merged.json',
            'report-selected/results.json', 'dev-selection/selection.json'} <= set(copies)
    assert not any(c.startswith('model/') for c in copies)
    assert copies['provenance/eval-data-MANIFEST.json']['sha256'] == h.sha256(run / 'eval-data-v1' / 'MANIFEST.json')
    assert copies['provenance/inference-run_receipt.json']['sha256'] == h.sha256(run / 'infer-joint-005' / 'run_receipt.json')
    assert {f'code/{n}' for n in D.CODE} <= set(copies) and copies['code/fastfill/comparison/contract.py']['sha256'] == code[
        'sha256']['fastfill/comparison/contract.py']
    assert copies['lineage/joint-sft-merged.json']['source'].endswith('dpo-001/joint/sft-merged.json')
    assert value['dev_selection']['selected'] == 'e1-dpo2' and value['dev_selection']['tied'] == ['e1-dpo2']
    out = tmp_path / 'pkg'
    assert len((out / 'SHA256SUMS').read_text().splitlines()) == len(value['files_to_copy'])
    card = _card(out)
    t1 = {r['endpoint'].split()[0]: r for r in results['tables']['T1']}
    assert f"{t1['(a)']['J']:.4f}" in card and D._fmt(t1['(b)']['delta'], True) in card
    assert '# FastFill unknown-size layout model: J-selected (e1-dpo2)' in card and '| P-selected (sft) |' in card
    assert 'Evaluated as the exact trained policy' in card and '`training-base/`' in card
    assert '**Delivered policy (the evaluated one, amendment 009):** the stage adapter `adapter/`' in card
    assert '**not delivered: amendment 011 §5 merged-weight check not run (15.00 %' in card
    assert 'It is not in SHA256SUMS.' in card and "from_pretrained('model'" not in card
    assert card.index('**Delivered policy') < card.index('`model/` (BF16 merge') and 'MANIFEST_READY' in card
    assert "eos_token_id=eos" in card and "convert_tokens_to_ids('<|im_end|>')" in card and 'json.loads' not in card
    assert "validate_output(raw, room, 'joint')" in card and "room, sidecar = bind_room(filled)" in card
    assert "msgs = messages(room, 'joint')[:2]" in card and "messages = " not in card and "p['size']" in card
    assert card.index("msgs = messages(") < card.index('apply_chat_template(msgs') < card.index('validate_output(raw')
    assert "PeftModel.from_pretrained(base, 'adapter')" in card and (code['git_commit'] or 'not recorded') in card
    assert 'diagnostic only' in card and 'Not run' not in card and 'sha256sum -c SHA256SUMS' in card
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
    assert {'sft-adapter/adapter_model.safetensors', 'base/model.safetensors',
            'size-priors.json', 'lineage/placement-sft-merged.json'} <= copies and not any('dpo' in c for c in copies)
    assert 'model/model.safetensors' not in copies
    card = _card(tmp_path / 'pkg')
    assert '# FastFill unknown-size layout model: P-selected (sft)' in card and 'merged at runtime on GPU' in card
    assert "dpo-001's SFT merge = the DPO1 training base" in card and '`sft-adapter/`' in card and '`base/`' in card
    assert "merge_and_unload()  # runtime BF16 merge, as evaluated" in card and "sizes[p['id']]" in card
    every = {r['scope']: r for r in _results(run)['tables']['T2']}['all']
    rate = every['rooms_with_global_fallback_rate']
    assert f"{100 * rate:.1f} % of test rooms ({round(rate * every['requests'])} of {every['requests']})" in card
    assert {'provenance/eval-data-MANIFEST.json', 'provenance/inference-run_receipt.json'} <= copies
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


def _check(tmp_path, run, d_a_prime=.01, complete=.998, sha=None, **bound):
    """dev_select check-merged's check.json shape for the J world's delivered merge (e1-dpo2), bound to its
    selection.json (frozen dev uids, file sha256, the candidate's dev a′ as the unmerged a′)."""
    receipt = json.loads((run / 'dpo-002' / 'joint' / 'dpo2-merged.json').read_text())
    selection = run / 'dev-eval-010' / 'selection.json'
    dev = json.loads(selection.read_text())
    side = {'complete_rate': .998, 'geometry_success_rate': .40, 'a_prime': dev['arms']['joint']['candidates']['e1-dpo2']['a_prime']}
    path = tmp_path / 'check.json'
    h.write_json(path, {'protocol': 'amendment-011-merged-weight-check', 'arm': 'joint', 'rooms': 600,
                        'uids_sha256': dev['uids_sha256'], 'selection_sha256': h.sha256(selection),
                        'merged': {**side, 'complete_rate': complete, 'a_prime': side['a_prime'] + d_a_prime,
                                   'geometry_success_rate': .41, 'model': {'model_sha256': sha or receipt['model_sha256']}},
                        'byte_identical': {'rooms': 300, 'fraction': .5}, 'merged_listed_as_convenience': True,
                        **bound, 'unmerged': {**side, **bound.pop('unmerged', {})}})
    return path


@pytest.mark.parametrize('d_a_prime, complete, listed', [(.01, .998, True), (-.01, .993, True),
                                                          (.0102, .998, False), (0, .992, False)])
def test_merged_check_lists_the_merge_only_within_amendment_011_limits(worlds, tmp_path, d_a_prime, complete, listed):
    run = worlds['J']
    value = D.build(run, run / 'report-selected', tmp_path / 'pkg', merged_check=_check(tmp_path, run, d_a_prime, complete))
    check, dests = value['merged_check'], {c['dest'] for c in value['files_to_copy']}
    assert check['merged_listed'] is listed is value['model']['listed']  # The file's own verdict is ignored.
    assert check['delta_merged_minus_unmerged']['a_prime'] == pytest.approx(d_a_prime)
    assert ('model/model.safetensors' in dests) is listed and 'merged-check/check.json' in dests
    card = _card(tmp_path / 'pkg')
    assert ('amendment 011 §5 check passed on 600 dev rooms' in card) is listed
    assert ('**not delivered: amendment 011 §5 check failed' in card and 'It is not in SHA256SUMS.' in card) is (not listed)
    assert 'byte-identical outputs 0.5000' in card and card.index('**Delivered policy') < card.index('`model/` (BF16')


@pytest.mark.parametrize('bound', [{'uids_sha256': 'f' * 64}, {'selection_sha256': 'f' * 64},
                                   {'unmerged': {'a_prime': .123}}])
def test_merged_check_must_bind_the_frozen_dev_rooms_selection_and_dev_run(worlds, tmp_path, bound):
    run = worlds['J']  # E.g. a smoke subset or another selection would pass a model_sha256-only binding.
    with pytest.raises(ValueError, match='not bound to the frozen dev rooms'):
        D.build(run, run / 'report-selected', tmp_path / 'pkg', merged_check=_check(tmp_path, run, **bound))
    assert not (tmp_path / 'pkg').exists()


def test_merged_check_must_bind_the_delivered_merge(worlds, tmp_path):
    run = worlds['J']
    with pytest.raises(ValueError, match='not bound to the delivered merged model'):
        D.build(run, run / 'report-selected', tmp_path / 'pkg', merged_check=_check(tmp_path, run, sha='f' * 64))
    path = _check(tmp_path, run)
    value = json.loads(path.read_text())
    sha = value['merged']['model']['model_sha256']
    h.write_json(path, {**value, 'byte_identical': {'fraction': 1.5}})
    with pytest.raises(ValueError, match=r'rates in \[0, 1\]'):
        D.load_merged_check(path, sha)
    for broken in ({'unmerged': {}}, {k: v for k, v in value.items() if k != 'selection_sha256'},
                   {'complete': .9, 'raw_success': .4, 'a_prime': .4, 'identical_fraction': 1., 'model_sha256': sha,
                    'unmerged': {'complete': .9, 'raw_success': .4, 'a_prime': .4}}):  # The flat form is gone.
        h.write_json(path, broken)
        with pytest.raises(ValueError, match='merged check lacks'):
            D.load_merged_check(path, sha)
    assert not (tmp_path / 'pkg').exists()


def test_refuses_request_code_that_did_not_build_the_evaluated_prompts(worlds, tmp_path, monkeypatch):
    run, code = worlds['J'], tmp_path / 'main-0123abc'  # A snapshot directory outside git: its name's commit.
    for n in D.CODE:
        (code / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(D.CODE_ROOT / n, code / n)
    monkeypatch.setattr(D, 'CODE_ROOT', code)
    assert D.build(run, run / 'report-selected', tmp_path / 'pkg')['code']['git_commit'] == '0123abc'
    with (code / 'fastfill/scene.py').open('a') as stream:
        stream.write('# changed after the evaluation data was built\n')
    with pytest.raises(ValueError, match='implementation_sha256'):
        D.build(run, run / 'report-selected', tmp_path / 'pkg2')
    assert not (tmp_path / 'pkg2').exists()


def test_commit_is_recorded_only_for_clean_request_code(tmp_path):
    def git(*args):
        subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=t', '-c', 'user.email=t@t', '-c',
                        'commit.gpgsign=false', *args], check=True, capture_output=True)
    git('init', '-q')
    (tmp_path / 'a.py').write_text('x\n')
    git('add', 'a.py')
    git('commit', '-qm', 'a')
    commit, dirty = D._commit(tmp_path, ['a.py'])
    assert len(commit) == 40 and dirty is False
    (tmp_path / 'a.py').write_text('y\n')
    assert D._commit(tmp_path, ['a.py']) == (commit, True)


def test_not_run_agents_are_disclosed_via_kwarg_and_cli(worlds, tmp_path, capsys):
    run = worlds['J']
    reason = 'not run: 4 engine-build failures, never substituted'
    D.main(['--run-root', str(run), '--report', str(run / 'report-selected'), '--out', str(tmp_path / 'pkg'),
            '--not-run', f'O5={reason}'])
    assert json.loads(capsys.readouterr().out)['status'] == 'MANIFEST_READY'
    value = json.loads((tmp_path / 'pkg' / 'delivery.json').read_text())
    assert value['agents']['not_run'] == {'O5': reason}
    assert f'Not run (never substituted): O5 ({reason}).' in _card(tmp_path / 'pkg')
    with pytest.raises(ValueError, match='both evaluated/excluded and not run'):
        D.build(run, run / 'report-selected', tmp_path / 'pkg2', {'included': ['O5'], 'excluded': {}}, not_run={'O5': reason})
    with pytest.raises(SystemExit):
        D.main(['--run-root', str(run), '--report', str(run / 'report-selected'), '--out', str(tmp_path / 'pkg3'),
                '--not-run', 'O5'])


ROOM = {'uid': 'u-1', 'source': 'S', 'boundary_type': 'polygon', 'boundary': [[0, 0], [5, 0], [5, 4], [0, 4]],
        'height': 3, 'room_type': 'bedroom',
        'fixed': [{'id': 'door', 'category': 'door', 'size': [1, .1, 2], 'pos': [2, 0, 0], 'yaw': 0}],
        'objects': [{'id': 'bed', 'category': 'bed', 'size': [2, 1.6, .5], 'pos': [2.5, 2, 0], 'yaw': 1.5708},
                    {'id': 'lamp', 'category': 'lamp', 'size': [.3, .3, .5], 'pos': [1, 1, .5], 'yaw': 0, 'parent': 'bed'}]}


@pytest.mark.parametrize('label', ['J', 'P'])
def test_card_request_and_output_code_runs_on_a_room_without_labels(tmp_path, monkeypatch, label):
    """Audit M8/minor 11: ids/categories only give the evaluation's prompt; IDs map back, P gets its request sizes."""
    from fastfill.comparison import contract as C
    from fastfill.comparison import size_prior as SP
    prior = SP.fit_prior([ROOM], min_count=1)
    (tmp_path / 'size-priors.json').write_text(json.dumps({'full_prior': prior}))
    monkeypatch.chdir(tmp_path)
    prepared = {**ROOM, 'objects': [{'id': o['id'], 'category': o['category']} for o in ROOM['objects']]}
    scope = {'prepared_room': prepared}
    exec('\n'.join(D.request_code(label)), scope)
    bound = C.bind_room(ROOM)[0]
    sizes = SP.predict_sizes({'room_type': 'bedroom', 'objects': [{'id': o['id'], 'category': o['category']}
                                                                  for o in bound['objects']]}, prior)
    expected = C.messages(bound, D.ARM[label], **({'estimated_sizes': sizes} if label == 'P' else {}))
    assert scope['msgs'] == expected[:2]
    scope['raw'] = expected[2]['content']
    exec('\n'.join(D.output_code(label)), scope)
    layout = {o['id']: o for o in scope['layout']}
    assert set(layout) == {'bed', 'lamp'} and layout['lamp']['on'] == 'bed' and layout['bed']['yaw'] == 90
    assert layout['bed']['size'] == ([2.0, 1.6, 0.5] if label == 'P' else [2, 1.6, .5])


def test_recipe_renders_rpo_alpha_and_refuses_without_it():
    receipt = {'stage': 'dpo1', 'training': {'recipe': {'lr': 5e-6, 'epochs': 3, 'rpo_alpha': 1.0, 'seed': 42}}}
    assert D.recipe(receipt, True) == 'lr 5e-06, epochs 3, rpo_alpha 1.0'
    del receipt['training']['recipe']['rpo_alpha']
    assert D.recipe(receipt, False) == 'lr 5e-06, epochs 3'
    with pytest.raises(ValueError, match='lacks rpo_alpha'):
        D.recipe(receipt, True)
    assert D.recipe({'training': {'recipe': 'SFT as trained'}}, True) == 'SFT as trained'


def _package(tmp_path, files):
    """A minimal package as build writes it: delivery.json files_to_copy, SHA256SUMS, MANIFEST.json."""
    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    copies = [{'source': str(path), 'dest': dest, 'sha256': h.sha256(path)} for dest, path in files.items()]
    h.write_json(pkg / 'delivery.json', {'status': 'MANIFEST_READY', 'files_to_copy': copies})
    (pkg / 'SHA256SUMS').write_text(D._sums(copies))
    (pkg / 'MODEL_CARD.md').write_text('card\n')
    h.write_json(pkg / 'MANIFEST.json', {'files': h.artifact_manifest(pkg)})
    return pkg, copies


def test_assemble_copies_verifies_and_refuses_changed_sources(tmp_path, capsys):
    sources = {'adapter/adapter_model.safetensors': tmp_path / 'a.bin', 'code/fastfill/scene.py': tmp_path / 'scene.py'}
    for i, path in enumerate(sources.values()):
        path.write_bytes(bytes([i]) * 10)
    pkg, copies = _package(tmp_path, sources)
    D.main(['assemble', '--package', str(pkg), '--target', str(tmp_path / 'full')])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['status'] == 'DELIVERY_PACKAGE_ASSEMBLED' and receipt['files_verified'] == 2
    full = tmp_path / 'full'
    assert all(h.sha256(full / c['dest']) == c['sha256'] for c in copies)
    assert json.loads((full / 'ASSEMBLY.json').read_text()) == receipt
    h.verify_artifacts(full, json.loads((full / 'MANIFEST.json').read_text())['files'])
    with pytest.raises(FileExistsError):
        D.assemble(pkg, full)
    sources['code/fastfill/scene.py'].write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch after copy: code/fastfill/scene.py'):
        D.assemble(pkg, tmp_path / 'again')
    (pkg / 'SHA256SUMS').write_text('')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        D.assemble(pkg, tmp_path / 'again')
    assert not (tmp_path / 'again').exists()


def test_built_package_is_manifest_ready_until_assembled(worlds, tmp_path):
    run = worlds['P']
    D.build(run, run / 'report-selected', tmp_path / 'pkg')
    with pytest.raises(OSError):  # The fixture weights are hash-only: nothing is published.
        D.assemble(tmp_path / 'pkg', tmp_path / 'full')
    assert not (tmp_path / 'full').exists()


def test_agent_table_names_models_modes_bias_and_exploratory_a_prime():
    est = lambda v: {'estimate': v, 'ci95': [v - .01, v + .01]}  # noqa: E731
    row = dict(scope='cohort', requests=1454, complete_rate=1., geometry_success_rate=.5, oor=.1, size_log_mae=.3)
    t = {'T5': [{**row, 'system': 'P', 'bias_fp': est(-.02), 'success_debiased_rate': .4986},
                {**row, 'system': 'O2-harness', 'bias_fp': est(-.45), 'success_debiased_rate': .3271}],
         'T5_cost': [{'system': 'P', 'model': 'Qwen3-8B + LoRA'},
                     {'system': 'O2', 'model': 'qwen3.8-27b', 'reasoning': 'thinking on (model default)',
                      'mode_note': 'template default effort high'}],
         'T5_contrasts': [{'scope': 'cohort', 'comparison': 'P−O2-harness', 'geometry_success': est(-.1),
                           'size_log_mae': est(-.2), 'own_debiased': est(.17)}]}
    t['T5'][1]['equal_size_success_rate'] = .61
    value = {'agents': {'included': ['O2'], 'excluded': {}, 'not_run': {'O5': 'never run'}}}
    text = '\n'.join(D._agents(value, t, {'enable_thinking': False}))
    assert '| cohort | P | Qwen3-8B + LoRA | greedy, thinking off | 1454 | 1.0000 | 0.5000 | 0.1000 | 0.3000 | ' in text
    assert '| cohort | O2-harness | qwen3.8-27b | thinking on (model default); template default effort high |' in text
    assert '| 0.3271 | 0.6100 |' in text and '| 0.4986 | not measured |' in text and 'equal-size control' in text
    assert '| cohort | P−O2-harness | -0.1000 [-0.1100, -0.0900] | -0.2000 [-0.2100, -0.1900] | +0.1700 [+0.1600, +0.1800] |' in text
    assert '-0.4500 [-0.4600, -0.4400]' in text and 'a′ (EXPLORATORY stress test)' in text
    assert 'not representative of the full test set' in text and 'exception class' in text and '`on`' in text
    assert 'Not run (never substituted): O5 (never run).' in text
