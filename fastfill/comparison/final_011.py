"""Amendment 011 (clarification): the final reports regenerated once, from one code snapshot (CPU only).

python -m fastfill.comparison.final_011 --run-root RUN --reference REF.json --corrections UIDS.json
       [--agents L O6 ...] [--not-run NAME=REASON ...] [--repeats 2000] [--workers N]
    CPU job of submit_011.sh, afterok the merged-weight check. New output dirs only; refuses if any exists:
    report-v2-sft-011       geometry-v2, infer-ARM-002, the SFT-paired agents RUN/llm-eval-NAME-v1
    report-v2-dpo1-011      geometry-v3, infer-ARM-003, no agents
    report-v2-selected-011  geometry-v5, the dev-selected test runs (dev-eval-010/selection.json), the
                            selected-paired agents RUN/llm-eval-NAME-selected-v1, --selection
    delivery-011            the selected report; the agents; the not-run list; the merged-weight check
                            dev-eval-010/merged-check/check.json (required: the merge is listed only if it passed)
    stage-table-011         the three reports; dev table with dev-eval-010/scores-report-only
    Every agent named (default L O6 O4 O2 O3 O7 O1a O1b) must be complete in both paired stages
    (dev_select.complete_agents) or the job refuses; one never run is left out of --agents and given in --not-run
    (default O5), never substituted. report_v2 gets the open-model status RUN/llm-open-status.json (required;
    llm_open.write_status), the not-run list and each agent's own generation run (L: llm-agent-001, open models:
    llm_open.out_name, MANIFEST-complete) as open_status / not_run / agent_runs; delivery gets merged_check /
    not_run. Checked before any output is written: selection + report-only scores (stage_table.load_selection);
    the merged check's MANIFEST, protocol, (placement, the selected placement candidate), selection_sha256,
    uids_sha256 and unmerged a' (= that candidate's dev score);
    each agent evaluation complete and bound to the data and its stage's J/P run hashes (report_v2._agent); the
    status, with no complete open model unreported; the agents' generation runs MANIFEST-complete. Checked only
    inside the builders (after earlier reports may exist): the merged check against the delivered model's sha256
    and its rates (delivery.load_merged_check), each generation run against the run its evaluation scored
    (report_v2._agent_run). J/P numbers come from the unchanged builders; superseded report dirs stay archived.
"""
import argparse
import json
import os
from pathlib import Path

from fastfill.comparison import delivery, stage_table
from fastfill.comparison import dev_select as ds
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import llm_open
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h

AGENTS = ('L', 'O6', 'O4', 'O2', 'O3', 'O7', 'O1a', 'O1b')
NOT_RUN = {'O5': 'not run: 4 engine-build failures, never substituted'}  # Amendment 011 section 3.
OUTS = {'sft': 'report-v2-sft-011', 'dpo1': 'report-v2-dpo1-011', 'selected': 'report-v2-selected-011',
        'delivery': 'delivery-011', 'table': 'stage-table-011'}
STAGES = {'sft': ('geometry-v2', '002'), 'dpo1': ('geometry-v3', '003')}  # selected: selection.json + geometry-v5.


def llm_run(run, name):
    """The agent's own generation run: L = llm-agent-001 (amendment 002), open models llm_open.out_name."""
    return Path(run) / ('llm-agent-001' if name == 'L' else llm_open.out_name(name, 'full'))


def final(args):
    run = Path(args.run_root)
    outs = {k: run / v for k, v in OUTS.items()}
    taken = [str(p) for p in outs.values() if os.path.lexists(p)]
    if taken:
        raise FileExistsError(f'refusing existing outputs: {taken}')
    modules = (Path(__file__), *(Path(m.__file__) for m in (r2, stage_table, delivery, ds)))
    if len({p.resolve().parent for p in modules}) != 1:
        raise ValueError('report builders are not from one code snapshot')
    not_run = dict(NOT_RUN) if args.not_run is None else dict(item.split('=', 1) for item in args.not_run)
    if set(not_run) & set(args.agents):
        raise ValueError(f'agents both evaluated and not run: {sorted(set(not_run) & set(args.agents))}')
    # Inputs checked before any output is written.
    selection, report_only = run / 'dev-eval-010' / 'selection.json', run / 'dev-eval-010' / 'scores-report-only'
    chosen = stage_table.load_selection(selection, report_only)['value']
    test, candidate = chosen['test'], chosen['test']['placement']['candidate']
    check = run / 'dev-eval-010' / 'merged-check'
    d._verified_dir(check)  # Amendment 011 section 5: required; absent = FileNotFoundError.
    checked = h.strict_json((check / 'check.json').read_text())
    if (checked.get('protocol') != 'amendment-011-merged-weight-check'
            or (checked.get('arm'), checked.get('candidate')) != ('placement', candidate)
            or checked.get('selection_sha256') != h.sha256(selection) or checked.get('uids_sha256') != chosen['uids_sha256']
            or (checked.get('unmerged') or {}).get('a_prime') != chosen['arms']['placement']['candidates'][candidate]['a_prime']):
        raise ValueError(f'{check}/check.json is not the merged check of the selected placement model')
    agents, excluded = {}, {}
    for stage, suffix in (('sft', 'v1'), ('selected', 'selected-v1')):
        agents[stage], excluded[stage] = ds.complete_agents([f'{n}={run}/llm-eval-{n}-{suffix}' for n in args.agents])
    print(json.dumps({'agents_included': {s: sorted(a) for s, a in agents.items()}, 'agents_excluded': excluded,
                      'not_run': not_run}), flush=True)
    if any(excluded.values()):  # The clarification: all agents in both paired stages; never a silent shrink.
        raise ValueError(f'agent evaluations incomplete: {excluded}')
    data = run / 'eval-data-v1'
    runs = {**{s: {arm: run / f'infer-{arm}-{tag}' for arm in h.ARMS} for s, (_, tag) in STAGES.items()},
            'selected': {arm: Path(test[arm]['inference']) for arm in h.ARMS}}
    for stage, names in agents.items():  # Each agent evaluation bound to this data and this stage's J/P runs.
        hashes = {arm: h.strict_json((runs[stage][arm] / 'run_receipt.json').read_text())['run_sha256'] for arm in h.ARMS}
        for root in names.values():
            r2._agent(root, h.sha256(data / 'MANIFEST.json'), hashes)
    status = run / 'llm-open-status.json'  # Required (amendment 007 not-run list); absent = FileNotFoundError.
    done = {k for k, v in h.strict_json(status.read_text()).items() if v.get('state') == 'complete'}
    for stage, names in agents.items():
        unreported = sorted(done - set(names))
        if unreported:  # report_v2 refuses this; stop before any output exists.
            raise ValueError(f'{stage}: open models complete in {status.name} but unreported: {unreported}')
    agent_runs = {s: {n: llm_run(run, n) for n in names} for s, names in agents.items()}
    absent = sorted({str(p) for v in agent_runs.values() for p in v.values() if not (p / 'MANIFEST.json').is_file()})
    if absent:
        raise FileNotFoundError(f'agent generation runs not complete: {absent}')
    common = {'corrections': args.corrections, 'repeats': args.repeats, 'workers': args.workers}
    geometry = {**{s: run / g for s, (g, _) in STAGES.items()}, 'selected': run / ds.GEOMETRY}
    for stage in ('sft', 'dpo1', 'selected'):
        paired = {'not_run': not_run, 'agent_runs': agent_runs[stage], 'open_status': status} if stage in agents else {}
        r2.build(data, geometry[stage], runs[stage], args.reference, outs[stage], agents.get(stage), **common, **paired,
                 **({'selection': selection} if stage == 'selected' else {}))
    # The binding package first, so it never depends on the stage table (as dev_select.final).
    value = delivery.build(run, outs['selected'], outs['delivery'], {'included': sorted(agents['selected']), 'excluded': {}},
                           merged_check=check / 'check.json', not_run=not_run)
    stage_table.build(run, {s: outs[s] for s in ('sft', 'dpo1', 'selected')}, outs['table'], selection, report_only)
    return {'status': 'FINAL_011_COMPLETE', 'outputs': {k: str(p) for k, p in outs.items()},
            'agents': {s: sorted(a) for s, a in agents.items()}, 'not_run': not_run, 'open_status': str(status),
            'merged_check': str(check / 'check.json'), 'delivered': value['chosen_arm'], 'candidate': value['candidate'],
            'code_sha256': {p.name: h.sha256(p) for p in modules}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('run-root', 'reference', 'corrections'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--agents', nargs='+', default=list(AGENTS), metavar='NAME')
    parser.add_argument('--not-run', action='append', metavar='NAME=REASON')  # Default NOT_RUN.
    parser.add_argument('--repeats', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    print(json.dumps(final(parse_args(argv))))


if __name__ == '__main__':
    main()
