"""Amendment 011 final regeneration (fixture run root) and its Isambard chain (stub sbatch), CPU only."""
import json
from pathlib import Path
import re
import shlex
import subprocess

import pytest

from fastfill.comparison import dpo_stages as d
from fastfill.comparison import final_011 as F
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests.test_delivery import implementation  # noqa: F401  Autouse (audit M9).
from fastfill.comparison.tests.test_stage_table import BROKEN, build_selected, build_world

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
RUN = '/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/unknownsize-20261013'


def report_only(run):
    """dev_select score --only report-only, bound to the fixture's dev scores."""
    scores = h.strict_json((run / 'dev-eval-010/scores/scores.json').read_text())
    d._publish_dir(run / 'dev-eval-010/scores-report-only', {'scores.json': json.dumps({
        **{k: scores[k] for k in ('protocol', 'data_manifest_sha256', 'uids_sha256', 'report_only')},
        'candidates': {arm: {'sft': scores['candidates'][arm]['sft']} for arm in h.ARMS}, 'only': 'report-only'}).encode()})


def merged_check(run, tamper=False, **over):
    """dev_select check-merged's check.json shape, bound to the selection and the delivered P model (the fixture
    delivers P sft)."""
    sha = h.strict_json((run / 'dpo-001/placement/sft-merged.json').read_text())['model_sha256']
    selection = h.strict_json((run / 'dev-eval-010/selection.json').read_text())
    a_prime = selection['arms']['placement']['candidates']['sft']['a_prime']  # The unmerged side = its dev score.
    side = {'complete_rate': 1.0, 'geometry_success_rate': .5, 'a_prime': a_prime}
    d._publish_dir(run / 'dev-eval-010/merged-check', {'check.json': json.dumps({
        'protocol': 'amendment-011-merged-weight-check', 'arm': 'placement', 'candidate': 'sft',
        'selection_sha256': h.sha256(run / 'dev-eval-010/selection.json'), 'uids_sha256': selection['uids_sha256'],
        'unmerged': side, 'merged': {**side, 'model': {'model_sha256': sha}}, 'byte_identical': {'fraction': 1.0},
        'decision': 'merged_convenience_artifact', **over}).encode()})
    if tamper:
        (run / 'dev-eval-010/merged-check/check.json').write_text('{"decision": "merged_convenience_artifact"}\n')


@pytest.fixture
def world(tmp_path):
    run = build_selected(build_world(tmp_path), broken=BROKEN)  # J e1-dpo2 (3 broken rooms), P sft -> delivers P.
    (run / 'corrections.json').write_text('["il-bed-2"]\n')
    return run


def args_for(run, *extra, agents=()):
    """Default --agents unless given: the fixture has no agent evaluations, so complete runs pass agents=[]."""
    args = F.parse_args(['--run-root', str(run), '--reference', str(run / 'reference.json'), '--corrections',
                         str(run / 'corrections.json'), '--repeats', '200', *extra])
    if agents != ():
        args.agents = list(agents)
    return args


def test_final_regenerates_every_report_once(world, capsys):
    run = world
    report_only(run)
    h.write_json(run / 'llm-open-status.json', {'O5': {'model': 'DeepSeek-V4-Flash', 'state': 'not run', 'reason': 'x'}})
    merged_check(run)
    result = F.final(args_for(run, agents=[]))
    logged = json.loads(capsys.readouterr().out.splitlines()[0])
    assert logged == {'agents_included': {'sft': [], 'selected': []}, 'agents_excluded': {'sft': {}, 'selected': {}},
                      'not_run': F.NOT_RUN}
    assert (result['status'], result['delivered'], result['candidate']) == ('FINAL_011_COMPLETE', 'P', 'sft')
    assert result['agents'] == {'sft': [], 'selected': []} and result['not_run'] == F.NOT_RUN
    assert result['open_status'] == str(run / 'llm-open-status.json')
    assert result['merged_check'] == str(run / 'dev-eval-010/merged-check/check.json')
    for name in F.OUTS.values():
        d._verified_dir(run / name)
    # J/P numbers do not change: the regenerated reports equal the fixture's reports of record, table for table.
    for stage, old in (('sft', 'report-sft'), ('dpo1', 'report-dpo1'), ('selected', 'report-selected')):
        new = h.strict_json((run / F.OUTS[stage] / 'results.json').read_text())
        before = h.strict_json((run / old / 'results.json').read_text())
        assert new['stage'] == before['stage'] and new['delivery'] == before['delivery']
        assert {k: new['tables'][k] for k in ('T1', 'T1_bias', 'T2', 'T3')} == {k: before['tables'][k] for k in
                                                                              ('T1', 'T1_bias', 'T2', 'T3')}
        assert new['input_sha256']['corrections'] == h.sha256(run / 'corrections.json')
        # The amendment 007 not-run disclosure reaches the agent-paired reports (O5 from the status + --not-run).
        assert [r['system'] for r in new['tables']['T5_not_run']] == (['O5'] if stage != 'dpo1' else [])
    package = h.strict_json((run / F.OUTS['delivery'] / 'delivery.json').read_text())
    assert package['agents'] == {'included': [], 'excluded': {}, 'not_run': F.NOT_RUN}
    # The delivery binds the check to the delivered merged model, re-applies the rule and copies it.
    assert package['merged_check']['merged_listed'] and package['model']['listed']
    assert any(c['dest'] == 'merged-check/check.json' for c in package['files_to_copy'])
    table = h.strict_json((run / F.OUTS['table'] / 'stage_table.json').read_text())
    assert table['stages_run'] == ['sft', 'dpo1', 'selected']
    assert table['dev_selection']['report_only_scores_manifest_sha256'] == h.sha256(
        run / 'dev-eval-010/scores-report-only/MANIFEST.json')
    assert table['inputs']['sft']['report'] == str(run / F.OUTS['sft'])
    with pytest.raises(FileExistsError, match='refusing existing outputs'):
        F.final(args_for(run, agents=[]))


@pytest.mark.parametrize('case, error, match', [
    ('no report-only scores', FileNotFoundError, 'scores-report-only'),
    ('no merged check', FileNotFoundError, 'merged-check'),
    ('tampered merged check', ValueError, 'check.json'),
    ('merged check of another selection', ValueError, 'not the merged check of the selected placement model'),
    ('merged check of another candidate', ValueError, 'not the merged check of the selected placement model'),
    ('merged check on other dev rooms', ValueError, 'not the merged check of the selected placement model'),
    ('merged check of another dev run', ValueError, 'not the merged check of the selected placement model'),
    ('agent also not run', ValueError, 'both evaluated and not run'),
    ('agent evaluations incomplete', ValueError, r"agent evaluations incomplete: .*'L': .*FileNotFoundError"),
    ('agent paired other J/P runs', ValueError, 'paired other J/P runs'),
    ('no open-model status', FileNotFoundError, 'llm-open-status.json'),
    ('complete open model unreported', ValueError, r"complete in llm-open-status.json but unreported: \['O2'\]")])
def test_final_refuses_before_writing_anything(world, case, error, match):
    run = world
    if case != 'no report-only scores':
        report_only(run)
    if case != 'no merged check':
        merged_check(run, tamper=case == 'tampered merged check',
                     **{'merged check of another selection': {'selection_sha256': '0' * 64},
                        'merged check of another candidate': {'candidate': 'rpo-dpo1'},
                        'merged check on other dev rooms': {'uids_sha256': '0' * 64},
                        'merged check of another dev run': {'unmerged': {'complete_rate': 1.0, 'geometry_success_rate': .5,
                                                                         'a_prime': .25}}}.get(case, {}))
    if case != 'no open-model status':  # Generated, but its evaluation is missing: report_v2 would refuse.
        h.write_json(run / 'llm-open-status.json', {'O5': {'state': 'not run'},
                                                    **({'O2': {'state': 'complete'}} if 'unreported' in case else {})})
    if case == 'agent paired other J/P runs':  # Complete, on this data, but scored against other J/P runs.
        summary = {'status': 'LLM_COMPARISON_COMPLETE', 'data_manifest_sha256': h.sha256(run / 'eval-data-v1/MANIFEST.json'),
                   'inference_run_sha256': {'joint': '0' * 64, 'placement': '1' * 64}}
        for suffix in ('v1', 'selected-v1'):
            d._publish_dir(run / f'llm-eval-L-{suffix}', {'summary.json': json.dumps(summary).encode()})
    extra = ['--not-run', 'L=never'] if case == 'agent also not run' else []
    agents = {'agent also not run': (), 'agent evaluations incomplete': (), 'agent paired other J/P runs': ['L']}
    with pytest.raises(error, match=match):
        F.final(args_for(run, *extra, agents=agents.get(case, [])))
    assert not any((run / name).exists() for name in F.OUTS.values())


def submit_world(tmp_path):
    """submit_011.sh with PROJECT_ROOT under tmp_path and the inputs it checks."""
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    names = ['dev-eval-010/plan/MANIFEST.json', 'dev-eval-010/selection.json', 'dpo-003/placement/dpo1-merged.json',
             'dev-eval-010/scores-report-only/MANIFEST.json', 'llm-open-status.json', 'llm-agent-001/MANIFEST.json',
             *(f'llm-eval-{n}-{s}/MANIFEST.json' for n in F.AGENTS for s in ('v1', 'selected-v1')),
             *(f'{F.llm_run(".", n).name}/MANIFEST.json' for n in F.AGENTS if n != 'L')]
    for name in names:
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text('{}\n')
    script = tmp_path / 'submit_011.sh'
    original = (SCRIPTS / 'submit_011.sh').read_text()
    script.write_text(original.replace(f'PROJECT_ROOT={RUN.removesuffix("/runs/unknownsize-20261013")}\n',
                                       f'PROJECT_ROOT={project}\n', 1))
    assert script.read_text() != original
    return script, run


def test_submit_chain_with_stub_sbatch(tmp_path):
    subprocess.run(['bash', '-n', str(SCRIPTS / 'submit_011.sh')], check=True)
    script, run = submit_world(tmp_path)
    done = subprocess.run(['bash', str(script), str(CODE)], env=T.stub_sbatch(tmp_path), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    log = (tmp_path / 'log').read_text().splitlines()
    project, hold, kill = run.parents[1], '--parsable --hold', '--parsable --hold --kill-on-invalid-dep=yes --dependency=afterok:'
    assert log[:4] == [
        f'{hold} {SCRIPTS}/isambard_010_infer_shard.sbatch {CODE} dev placement rpo-dpo1-merged 0 2',
        f'{hold} {SCRIPTS}/isambard_010_infer_shard.sbatch {CODE} dev placement rpo-dpo1-merged 1 2',
        f'{kill}101:102 {SCRIPTS}/isambard_010_cpu.sbatch {CODE} dev-merge placement rpo-dpo1-merged 2',
        f'{kill}103 {SCRIPTS}/isambard_010_cpu.sbatch {CODE} check-merged']
    evidence = f'{CODE}/docs/research/20261013-unknown-size-comparison/evidence'
    command = (f'cd {CODE} && PYTHONPATH={CODE} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
               f'{project}/env/bin/python -B -m fastfill.comparison.final_011 --run-root {run} '
               f'--reference {evidence}/amendment-005-reference-rows-20261013.json '
               f'--corrections {evidence}/amendment-006-corrections-uids-candidate-003.json --repeats 2000 --workers 16')
    assert log[4] == (
        f'{kill}104 --account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=16 '
        f'--mem=64G --gres=none --no-requeue --time=12:00:00 --job-name=unknownsize-011-final '
        f'--output={project}/runs/isambard/unknownsize-011-final-%j.out --wrap {command}')
    parsed = F.parse_args(shlex.split(command.split(' -m fastfill.comparison.final_011 ')[1]))
    assert parsed.agents == list(F.AGENTS) and parsed.not_run is None and parsed.workers == 16
    assert len(log) == 5 and (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101,102,103,104,105\n'
    assert done.stdout.splitlines()[-1] == 'released 101 102 103 104 105'


@pytest.mark.parametrize('case, code, match', [
    ('existing output', 1, 'refusing to reuse existing .*/report-v2-selected-011'),
    ('existing merged run', 1, 'refusing to reuse existing .*/dev-eval-010/rpo-dpo1-merged'),
    ('missing merged model', 1, 'missing input .*/dpo1-merged.json'),
    ('missing report-only scores', 1, 'missing input .*/scores-report-only/MANIFEST.json'),
    ('missing open-model status', 1, 'missing input .*/llm-open-status.json'),
    ('missing agent evaluation', 1, 'missing input .*/llm-eval-O1b-selected-v1/MANIFEST.json'),
    ('missing agent run', 1, 'missing input .*/llm-open-O7-001/MANIFEST.json'),
    ('sbatch fails', 1, "sbatch failed for 'dev merge rpo-dpo1-merged placement'.*scancel 101 102"),
    ('usage', 2, 'usage')])
def test_submit_refusals(tmp_path, case, code, match):
    script, run = submit_world(tmp_path)
    if case == 'existing output':
        (run / 'report-v2-selected-011').mkdir()
    if case == 'existing merged run':
        (run / 'dev-eval-010/rpo-dpo1-merged').mkdir()
    missing = {'missing merged model': 'dpo-003/placement/dpo1-merged.json',
               'missing report-only scores': 'dev-eval-010/scores-report-only/MANIFEST.json',
               'missing open-model status': 'llm-open-status.json',
               'missing agent evaluation': 'llm-eval-O1b-selected-v1/MANIFEST.json',
               'missing agent run': 'llm-open-O7-001/MANIFEST.json'}
    if case in missing:
        (run / missing[case]).unlink()
    argv = ['bash', str(script), str(CODE)] + (['extra'] if case == 'usage' else [])
    done = subprocess.run(argv, env=T.stub_sbatch(tmp_path, fail_on=3 if case == 'sbatch fails' else 0),
                          capture_output=True, text=True)
    assert done.returncode == code and re.search(match, done.stderr, re.S)
    assert not (tmp_path / 'log.scontrol').exists() and (case == 'sbatch fails') == (tmp_path / 'log').exists()
