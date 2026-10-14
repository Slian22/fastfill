"""Amendment 010: per arm, select among sft / e1-dpo1 / e1-dpo2 / rpo-dpo1 / rpo-dpo2 on dev only.

python -m fastfill.comparison.dev_select STEP ...   (sbatch: isambard_010_*.sbatch; chain: submit_dev_selection.sh)

build   --data MAIN --sft-runs SFT_J SFT_P --out RUN/dev-select-v1
    600 dev rooms of the SFT common length-admitted dev pool (both completed runs' lengths.json 'dev'):
    300 from OptiScene_holodeck + IL3D_3dfront and 300 from all other sources, each by ascending
    sha256('dev-select-v1:' + uid); a shortfall fails. Published in the eval-data-v1 layout: per arm
    requests.jsonl = the dev rows' system + user turns (no target), rooms.jsonl = GT rooms for scoring,
    size-estimates.jsonl = the P request sizes, MANIFEST.json with expected_uids (sorted), the frozen
    uids_sha256 and training_manifest_sha256 = MAIN. Built before any candidate is scored.
score   --data DEV --root RUN/dev-eval-010 --out RUN/dev-eval-010/scores [--workers N]
    Each candidate's merged dev inference ROOT/CANDIDATE/infer-ARM (shared length-sorted dev plan,
    amendment 002 caps) is verified (evaluate_pair._inference) and bound to its model (SFT adapter run, or
    the dpo-002/dpo-003 stage's exact policy), then scored by the frozen geometry.score; reached_token_cap
    without EOS is a failure (amendment 005). Per arm and candidate: complete rate, size log MAE of the
    candidate and of SFT on the same rooms (complete in both; amendment 005 common-complete pairing), own
    footprint bias B = mean over the candidate's complete rooms of the room mean log(pred/GT footprint),
    and a' = all-request geometry success after rescaling its x/y sizes about object centres by exp(-B/2)
    (report_v2._derive, amendment 005). The amendment-009 stages a009-dpo1/a009-dpo2 (dpo-001 exact
    policy; amendment 010 addendum) are scored the same way under 'report_only': never selectable.
select  --scores SCORES --out RUN/dev-eval-010/selection.json --code-root CODE [--shards 6] [--no-submit]
    Eligible: SFT always; a DPO candidate iff complete rate >= SFT's - 0.01 and its log MAE <= SFT's log
    MAE on the same common rooms + 0.01. Selected: the eligible candidate with the highest a'; candidates
    within 0.005 of the best tie and go to fewer training stages (sft 0, dpo1 1, dpo2 2), then higher a', then protocol order, so SFT
    wins unless a DPO candidate is eligible and better by more than 0.005. selection.json is write-once
    with every candidate's dev metrics. Then the test chain is submitted with sbatch (fails with the exact
    commands printed if sbatch is unavailable): for each selected DPO candidate, N exact-policy test shards
    on infer-v2-005/plan (byte copy of infer-v2-002/plan) -> merge infer-ARM-005; a selected SFT reuses
    infer-ARM-002. Then evaluate_pair --stage-label selected over the two selected models -> geometry-v5.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import types

from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import geometry as g
from fastfill.comparison import infer as i
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

ARMS = h.ARMS
SALT = 'dev-select-v1:'
PAPER_SOURCES = ('OptiScene_holodeck', 'IL3D_3dfront')
TARGETS = {'paper_sources': 300, 'other_sources': 300}
CANDIDATES = ('sft', 'e1-dpo1', 'e1-dpo2', 'rpo-dpo1', 'rpo-dpo2')  # Protocol order (last tie-break).
REPORT_ONLY = ('a009-dpo1', 'a009-dpo2')  # Amendment 010 addendum: dev-only report rows, never selectable.
RUNS = {**{k: v[0] for k, v in d.CANDIDATES.items()}, 'a009': d.SOURCE_RUN}
STAGE_COUNT = {'sft': 0, 'e1-dpo1': 1, 'rpo-dpo1': 1, 'e1-dpo2': 2, 'rpo-dpo2': 2}
COMPLETE_MARGIN, MAE_MARGIN, TIE = 0.01, 0.01, 0.005
EPS = 1e-12  # Float slack so that the rule's inclusive bounds stay inclusive.
EXACT = 'unmerged_fp32_stage_adapter_on_bf16_training_base'
SFT_TEST_TAG, TEST_TAG, GEOMETRY = '002', '005', 'geometry-v5'
RULE = ("eligible: SFT always; DPO candidate iff dev complete rate >= SFT - 0.01 and its dev size log MAE <= SFT's "
        "dev size log MAE + 0.01, both on the rooms complete in both it and SFT. selected: highest dev a' "
        "(own footprint bias rescale) among eligible; candidates within 0.005 of the best tie and go to fewer training stages, then higher a', then "
        "protocol order sft, e1-dpo1, e1-dpo2, rpo-dpo1, rpo-dpo2")


def rank(uid):
    return hashlib.sha256((SALT + uid).encode('utf-8')).hexdigest()


def _digest(uids):
    return hashlib.sha256(json.dumps(uids).encode('utf-8')).hexdigest()


def pick(pool):
    """pool: {uid: source}; pure and order independent."""
    ordered = sorted(pool, key=lambda u: (rank(u), u))
    groups = {'paper_sources': [u for u in ordered if pool[u] in PAPER_SOURCES],
              'other_sources': [u for u in ordered if pool[u] not in PAPER_SOURCES]}
    chosen = {}
    for name, target in TARGETS.items():
        if len(groups[name]) < target:
            raise ValueError(f'dev pool has {len(groups[name])} {name} rooms, {target} required')
        chosen[name] = groups[name][:target]
    return sorted(u for picked in chosen.values() for u in picked), {k: len(v) for k, v in groups.items()}


# ---- build ---------------------------------------------------------------------------------------

def build(args):
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    pairs, data = h.load_pair(args.data)
    if data['target_policy'] != h.MAIN_TARGET_POLICY:
        raise ValueError('dev selection requires the qualified main data')
    admitted, _, lengths = d._admitted(args.sft_runs, data, 'dev')
    rows = {arm: h._by_uid(pairs['dev'][arm]) for arm in ARMS}
    rooms = h._by_uid(h._rows(Path(args.data), 'dev_rooms.jsonl', data['files']['dev_rooms.jsonl']))
    if not set(admitted) <= rows['joint'].keys() & rooms.keys():
        raise ValueError('length-admitted dev UID outside the verified dev rows/rooms')
    uids, pool = pick({u: rows['joint'][u]['source'] for u in admitted})
    files = {f'{arm}/requests.jsonl': [] for arm in ARMS}
    files.update({'rooms.jsonl': [], 'size-estimates.jsonl': []})
    for uid in uids:
        component, room = rows['joint'][uid]['group'], rooms[uid]
        if room['source'] != rows['joint'][uid]['source'] or any(rows[a][uid]['group'] != component for a in ARMS):
            raise ValueError(f'dev row/room identity mismatch: {uid}')
        meta = {'uid': uid, 'source': room['source'], 'group': component, 'component': component, 'split': 'dev',
                **({'room_type': room['room_type']} if room.get('room_type') is not None else {})}
        for arm in ARMS:  # The eval-data-v1 request: system + user turns only, the target turn dropped.
            files[f'{arm}/requests.jsonl'].append(i.validate_request({**meta, 'messages': rows[arm][uid]['messages'][:2]}, arm))
        files['rooms.jsonl'].append({**room, 'group': component, 'component': component, 'split': 'dev'})
        request = h.strict_json(rows['placement'][uid]['messages'][1]['content'])
        files['size-estimates.jsonl'].append({'uid': uid, 'estimated_sizes': {o['id']: o['size'] for o in request['objects']}})
    by_source = {}
    for uid in uids:
        by_source[rooms[uid]['source']] = by_source.get(rooms[uid]['source'], 0) + 1
    selection = {'protocol': 'amendment-010-dev-select-v1', 'salt': SALT, 'targets': TARGETS,
                 'pool': {'length_admitted_dev': len(admitted), **pool}, 'by_source': dict(sorted(by_source.items())),
                 'uids': uids, 'uids_sha256': _digest(uids)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        for arm in ARMS:
            (dest / arm).mkdir()
        for name, values in files.items():
            (dest / name).write_bytes(d._jsonl(values))
        h.write_json(dest / 'selection.json', selection)
        h.write_json(dest / 'MANIFEST.json', {
            'version': 1, 'scope': 'amendment_010_dev_selection', 'split': 'dev', 'expected_uids': uids,
            'uids_sha256': selection['uids_sha256'], 'counts': {'dev': len(uids)},
            'training_manifest_sha256': data['manifest_sha256'], 'main_data_manifest_sha256': data['manifest_sha256'],
            'sft_lengths_sha256': lengths, 'input_policy': 'system_user_only_labels_and_inventory_separate',
            'code_sha256': {'dev_select.py': h.sha256(__file__)}, 'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return {'status': 'DEV_SELECT_COMPLETE', 'rooms': len(uids), 'uids_sha256': selection['uids_sha256'],
            'by_source': selection['by_source']}


# ---- score ---------------------------------------------------------------------------------------

def model_path(run_root, arm, candidate):
    recipe, stage = candidate.split('-')
    return Path(run_root) / RUNS[recipe] / arm / f'{stage}-merged'


def _bind(receipt, run_root, arm, candidate):
    """The merged dev inference must come from this candidate's model for this arm."""
    config = receipt['config']
    if candidate == 'sft':
        run = (Path(run_root) / f'sft-{arm}-001').resolve()
        ok = (receipt.get('stage') is None and config.get('training_run') == str(run)
              and config.get('adapter') == str(run / arm / 'adapter') and 'model' not in config)
    else:
        ok = (receipt.get('stage') == candidate.split('-')[1] and config.get('lora') == EXACT
              and config.get('model') == str(model_path(run_root, arm, candidate).resolve()))
    if not ok or config['arm'] != arm:
        raise ValueError(f'dev inference is not {arm}/{candidate}')


def metrics(rows, sft_rows):
    """Selection inputs of one candidate (rows after cap failures and with bias_fp)."""
    complete = [r for r in rows if r['complete']]
    common = [(r['size_log_mae'], s['size_log_mae']) for r, s in zip(rows, sft_rows, strict=True)
              if r['complete'] and s['complete']]
    bias = statistics.mean(r['bias_fp'] for r in complete) if complete else None
    return {'requests': len(rows), 'complete_rate': len(complete) / len(rows),
            'geometry_success_rate': sum(r['geometry_success'] for r in rows) / len(rows),
            'size_log_mae_common_with_sft': statistics.mean(x for x, _ in common) if common else None,
            'sft_size_log_mae_common': statistics.mean(y for _, y in common) if common else None,  # Same rooms.
            'common_with_sft': len(common),
            'size_log_mae_own': statistics.mean(r['size_log_mae'] for r in complete) if complete else None,
            'footprint_bias': bias, 'scale_factor': None if bias is None else math.exp(-bias / 2)}


def score(args):
    data, root, out = Path(args.data), Path(args.root), Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    manifest = e._verified(data)
    expected = manifest['expected_uids']
    rooms = h._by_uid(h._rows(data, 'rooms.jsonl', manifest['files']['rooms.jsonl']))
    requests = {arm: h._by_uid(h._rows(data, f'{arm}/requests.jsonl', manifest['files'][f'{arm}/requests.jsonl']))
                for arm in ARMS}
    plans, result, files = set(), {}, {}
    for arm in ARMS:
        rows, result[arm] = {}, {}
        for candidate in CANDIDATES + REPORT_ONLY:
            path = root / candidate / f'infer-{arm}'
            records, receipt = e._inference(path, data, expected)
            if receipt['training_data']['manifest_sha256'] != manifest['training_manifest_sha256']:
                raise ValueError('dev inference used other training data')
            _bind(receipt, root.parent, arm, candidate)
            plans.add(receipt['context_plan_sha256'])
            tasks = [(rooms[r['uid']], requests[arm][r['uid']], r, arm) for r in records]
            scored = r2._map(e._score, tasks, args.workers)
            caps = [r['reached_token_cap'] and not r['stopped_on_eos'] for r in records]
            rows[candidate] = [r2._with_bias(r2.cap_failure(s) if c else s, rooms[s['uid']]) for s, c in zip(scored, caps)]
            result[arm][candidate] = {'inference': {'path': str(path.resolve()), 'run_sha256': receipt['run_sha256'],
                                                    'manifest_sha256': h.sha256(path / 'MANIFEST.json'),
                                                    'stage': receipt.get('stage', 'sft'), 'lora': receipt['config']['lora']},
                                      'token_cap_failures': sum(caps)}
        for candidate in CANDIDATES + REPORT_ONLY:
            value = metrics(rows[candidate], rows['sft'])
            tasks = [(r, rooms[r['uid']], requests[arm][r['uid']], value['scale_factor']) for r in rows[candidate]]
            derived = r2._map(r2._derive, tasks, args.workers)
            value['a_prime'] = sum(bool(x['success_debiased']) for x in derived) / len(derived)
            result[arm][candidate].update(value)
            files[f'{arm}-{candidate}-geometry.jsonl'] = d._jsonl(
                {**r, 'success_debiased': x['success_debiased']} for r, x in zip(rows[candidate], derived))
    if len(plans) != 1 or None in plans:
        raise ValueError('candidates were not run on one shared dev plan')
    value = {'protocol': 'amendment-010-dev-scores', 'data_manifest_sha256': h.sha256(data / 'MANIFEST.json'),
             'uids_sha256': manifest['uids_sha256'], 'plan_sha256': plans.pop(),
             'candidates': {arm: {c: result[arm][c] for c in CANDIDATES} for arm in ARMS},
             'report_only': {arm: {c: result[arm][c] for c in REPORT_ONLY} for arm in ARMS},
             'code_sha256': {p.name: h.sha256(p) for p in (Path(__file__), Path(e.__file__), Path(g.__file__),
                                                           Path(r2.__file__))}}
    if _digest(expected) != value['uids_sha256']:
        raise ValueError('dev UIDs differ from the frozen dev selection')
    d._publish_dir(out, {'scores.json': (json.dumps(value, indent=1) + '\n').encode(), **files})
    return {'status': 'DEV_SCORES_COMPLETE', **{arm: {c: round(m['a_prime'], 4) for c, m in result[arm].items()}
                                                for arm in ARMS}}


# ---- select --------------------------------------------------------------------------------------

def choose(candidates):
    """Amendment 010 rule for one arm; candidates = {name: metrics}, all five present."""
    if set(candidates) != set(CANDIDATES):
        raise ValueError('every candidate must be scored')
    sft = candidates['sft']

    def eligible(name):
        m = candidates[name]
        return name == 'sft' or (  # Both MAEs over the same rooms (complete in it and SFT), amendment 005 pairing.
            m['size_log_mae_common_with_sft'] is not None
            and m['complete_rate'] >= sft['complete_rate'] - COMPLETE_MARGIN - EPS
            and m['size_log_mae_common_with_sft'] <= m['sft_size_log_mae_common'] + MAE_MARGIN + EPS)
    names = [c for c in CANDIDATES if eligible(c)]
    best = max(candidates[c]['a_prime'] for c in names)
    tied = [c for c in names if candidates[c]['a_prime'] >= best - TIE - EPS]
    selected = min(tied, key=lambda c: (STAGE_COUNT[c], -candidates[c]['a_prime'], CANDIDATES.index(c)))
    return {'selected': selected, 'eligible': names, 'tied': tied, 'best_a_prime': best}


def selected_dirs(run_root, selected):
    return {arm: Path(run_root) / f'infer-{arm}-{SFT_TEST_TAG if c == "sft" else TEST_TAG}' for arm, c in selected.items()}


def chain_jobs(code_root, run_root, selected, shards):
    """sbatch jobs (label, afterok labels, script + args) for the selected models' test evaluation."""
    scripts, run_root = Path(code_root) / 'fastfill/comparison', Path(run_root)
    dirs, jobs, merges = selected_dirs(run_root, selected), [], []
    for arm in ARMS:
        if selected[arm] == 'sft':
            continue  # Already measured: infer-ARM-002.
        labels = [f'test shard {arm} {k}' for k in range(shards)]
        jobs += [{'label': label, 'after': [], 'argv': [str(scripts / 'isambard_010_infer_shard.sbatch'), str(code_root),
                                                       'test', arm, selected[arm], str(k), str(shards)]}
                 for k, label in enumerate(labels)]
        jobs.append({'label': f'test merge {arm}', 'after': labels,
                     'argv': [str(scripts / 'isambard_infer_merge.sbatch'), str(code_root), arm, str(shards),
                              str(run_root / f'infer-v2-{TEST_TAG}'), str(dirs[arm])]})
        merges.append(f'test merge {arm}')
    jobs.append({'label': 'selected geometry', 'after': merges,
                 'argv': [str(scripts / 'isambard_010_cpu.sbatch'), str(code_root), 'selected-geometry',
                          str(dirs['joint']), str(dirs['placement'])]})
    return jobs


def shell(jobs):
    """The same chain as login-node shell commands (roots held until everything is submitted)."""
    lines, names = ['set -euo pipefail'], {}
    for k, job in enumerate(jobs):
        names[job['label']] = f'J{k}'
        after = ':'.join('$' + names[label] for label in job['after'])
        flags = f'--kill-on-invalid-dep=yes --dependency=afterok:{after}' if after else '--hold'
        lines.append(f"J{k}=$(sbatch --parsable {flags} {shlex.join(job['argv'])} | cut -d';' -f1)  # {job['label']}")
    roots = ','.join('$' + names[j['label']] for j in jobs if not j['after'])
    return '\n'.join(lines + [f'scontrol release {roots}'])


def submit(jobs, run=subprocess.run, which=shutil.which):
    """Submit with dependencies; roots are held until the whole chain is in, so an abort runs nothing."""
    if which('sbatch') is None or which('scontrol') is None:
        print(shell(jobs), file=sys.stderr)
        raise SystemExit('ABORT: sbatch/scontrol unavailable in this job; run the commands above on the login node')
    env = {k: v for k, v in os.environ.items()  # No parent-job leakage; the cluster config path stays.
           if k == 'SLURM_CONF' or not k.startswith(('SLURM_', 'SBATCH_'))}
    ids, roots = {}, []
    for job in jobs:
        after = [ids[label] for label in job['after']]
        flags = ['--kill-on-invalid-dep=yes', '--dependency=afterok:' + ':'.join(after)] if after else ['--hold']
        done = run(['sbatch', '--parsable', *flags, *job['argv']], capture_output=True, text=True, env=env)
        job_id = done.stdout.strip().split(';')[0]
        if done.returncode or not job_id.isdigit():  # e.g. no submission from compute nodes: hand over the commands.
            print(shell(jobs), file=sys.stderr)
            raise SystemExit(f"ABORT: sbatch failed for '{job['label']}': {done.stderr.strip()}; "
                             f"cancel with: scancel {' '.join(ids.values()) or '(nothing submitted)'}, "
                             'then run the commands above on the login node')
        ids[job['label']] = job_id
        if not after:
            roots.append(job_id)
    done = run(['scontrol', 'release', ','.join(roots)], capture_output=True, text=True, env=env)
    if done.returncode:
        raise SystemExit(f"ABORT: could not release {','.join(roots)}; cancel with: scancel {' '.join(ids.values())}")
    return ids


def select(args, run=subprocess.run, which=shutil.which):
    scores_dir, out = Path(args.scores), Path(args.out)
    d._verified_dir(scores_dir)
    scores = h.strict_json((scores_dir / 'scores.json').read_text())
    run_root = scores_dir.parent.parent
    arms = {arm: {**choose({c: scores['candidates'][arm][c] for c in CANDIDATES}),
                  'candidates': scores['candidates'][arm]} for arm in ARMS}
    selected = {arm: arms[arm]['selected'] for arm in ARMS}
    dirs = selected_dirs(run_root, selected)
    value = {'protocol': 'amendment-010-dev-selection', 'rule': RULE,
             'scores_manifest_sha256': h.sha256(scores_dir / 'MANIFEST.json'),
             'data_manifest_sha256': scores['data_manifest_sha256'], 'uids_sha256': scores['uids_sha256'],
             'arms': arms, 'selected': selected,
             'test': {arm: {'candidate': selected[arm], 'inference': str(dirs[arm]), 'already_measured': selected[arm] == 'sft'}
                      for arm in ARMS}, 'geometry': str(run_root / GEOMETRY),
             'code_sha256': {'dev_select.py': h.sha256(__file__)}}
    if out.exists():
        if h.strict_json(out.read_text()) != value:
            raise ValueError(f'{out} exists with another selection')
    else:
        i.write_once(out, value)
    receipt = out.with_name('submission.json')
    if receipt.exists():
        raise FileExistsError(f'test chain already submitted: {receipt}')
    if getattr(args, 'no_submit', False):
        return {'status': 'SELECTED', 'selected': selected}
    fresh = [run_root / GEOMETRY] + [p for arm in ARMS if selected[arm] != 'sft'
                                     for p in (dirs[arm], run_root / f'infer-v2-{TEST_TAG}' / arm)]
    taken = [str(p) for p in fresh if os.path.lexists(p)]
    if taken:
        raise FileExistsError(f'refusing existing test outputs: {taken}')
    if any(c != 'sft' for c in selected.values()):
        from fastfill.comparison import infer_shard
        src, plan = run_root / f'infer-v2-{SFT_TEST_TAG}/plan', run_root / f'infer-v2-{TEST_TAG}/plan'
        if not plan.exists():
            d.copy_plan(types.SimpleNamespace(src=src, out=[plan]))
        elif infer_shard.load_plan(plan)[1] != infer_shard.load_plan(src)[1]:
            raise ValueError(f'{plan} is not the infer-v2-{SFT_TEST_TAG} plan')
    jobs = chain_jobs(args.code_root, run_root, selected, args.shards)
    ids = submit(jobs, run, which)
    i.write_once(receipt, {'selection_sha256': h.sha256(out), 'jobs': [{**j, 'id': ids[j['label']]} for j in jobs]})
    return {'status': 'TEST_CHAIN_SUBMITTED', 'selected': selected, 'jobs': ids}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    step = sub.add_parser('build')
    step.add_argument('--data', type=Path, required=True)
    step.add_argument('--sft-runs', type=Path, nargs=2, required=True, metavar=('SFT_JOINT', 'SFT_PLACEMENT'))
    step.add_argument('--out', type=Path, required=True)
    step = sub.add_parser('score')
    for flag in ('data', 'root', 'out'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--workers', type=int, default=1)
    step = sub.add_parser('select')
    for flag in ('scores', 'out', 'code-root'):
        step.add_argument('--' + flag, type=Path, required=True)
    step.add_argument('--shards', type=int, default=6)
    step.add_argument('--no-submit', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps({'build': build, 'score': score, 'select': select}[args.command](args)))


if __name__ == '__main__':
    main()
