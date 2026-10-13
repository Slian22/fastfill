"""Final N-system visual run config (amendments 005 §5, 007, 008) for overnight.py on geometry-v4.

python -m fastfill.comparison.visual_final \
    --template docs/research/20261013-unknown-size-comparison/visual-final-dpo2-template.json \
    --sacct SACCT.txt --local-code-root SNAPSHOT --remote-api-code SERVER_SNAPSHOT --out CONFIG.json \
    [--output-root NEW_DIR] [--ssh-control-path SOCKET]

SACCT.txt is the saved output of
    sacct -n -P -j <every candidate_agents job id> --format=JobIDRaw,State,ExitCode
taken after every candidate DPO2-paired agent evaluation (llm-eval-*-dpo2-v1) has ended. An agent whose
job COMPLETED with exit 0:0 is rendered; any other ended state excludes it with that state recorded; a
job still pending/running, or absent from the output, refuses (launch later). L is required. The judged
set is visual_pipeline.judged_systems: J, P, L and the strongest included open model (O6 > O5 > O4).
Writes a new overnight config, validated by overnight.validate_config, with the selection recorded under
visual_final_selection. It launches nothing; the user starts the run on the Mac:
    caffeinate -dimsu python -m fastfill.comparison.overnight --config CONFIG.json
"""
import argparse
import json
from pathlib import Path

from fastfill.comparison import overnight, visual_pipeline

FILL = '<FILL'


def select(candidates, sacct):
    jobs = {label: str(c['job_id']) for label, c in candidates.items()}
    records = overnight.parse_sacct(sacct, list(jobs.values()))
    included, excluded = [], {}
    for label in (s for s in visual_pipeline.SYSTEMS if s in jobs):
        record = records.get(jobs[label])
        if record is None:
            raise ValueError(f'{label}: job {jobs[label]} is missing from the sacct output')
        if (record['state'], record['exit_code']) == ('COMPLETED', '0:0'):
            included.append(label)
        elif record['state'] in overnight.TERMINAL_FAILURES | {'COMPLETED'}:
            excluded[label] = record
        else:
            raise ValueError(f'{label}: evaluation job {jobs[label]} has not ended ({record["state"]}); launch later')
    if 'L' not in included:
        raise ValueError('the API agent L evaluation did not complete; the judged set requires it')
    return included, excluded


def build(template, sacct, out, local_code_root, remote_api_code, output_root=None, ssh_control_path=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    value = overnight.read_json(template)
    agents = value['candidate_agents']
    included, excluded = select(agents, Path(sacct).read_text())
    config = {**value['config'], 'local_code_root': str(local_code_root), 'remote_api_code': str(remote_api_code),
              **({'output_root': str(output_root)} if output_root else {}),
              **({'ssh_control_path': ssh_control_path} if ssh_control_path else {}),
              'agents': {s: agents[s]['dir'] for s in included},
              'agent_job_ids': {s: str(agents[s]['job_id']) for s in included},
              'visual_final_selection': {'included': included, 'excluded': excluded,
                                         'judged': visual_pipeline.judged_systems(['J', 'P', *included]),
                                         'template_sha256': overnight.sha256(template),
                                         'sacct_sha256': overnight.sha256(sacct)}}
    if FILL in json.dumps(config):
        raise ValueError('template placeholder left unfilled')
    if Path(config['output_root']).exists():
        raise FileExistsError(config['output_root'])
    overnight.validate_config(config)
    overnight.atomic_json(out, config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('template', 'sacct', 'local-code-root', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--remote-api-code', required=True)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--ssh-control-path')
    args = parser.parse_args(argv)
    config = build(args.template, args.sacct, args.out, args.local_code_root.resolve(), args.remote_api_code,
                   args.output_root.resolve() if args.output_root else None, args.ssh_control_path)
    print(json.dumps({'config': str(args.out), **config['visual_final_selection'],
                      'launch': f'caffeinate -dimsu python -m fastfill.comparison.overnight --config {args.out}'}))


if __name__ == '__main__':
    main()
