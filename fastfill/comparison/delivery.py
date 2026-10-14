"""Amendment 010 delivery package: the dev-selected model of the arm the 'selected' report's amendment-005 rule chose.

python -m fastfill.comparison.delivery --run-root RUN --report REPORT_V2_SELECTED_DIR --out NEW_DIR \
    [--merged-check RUN/dev-eval-010/merged-check/check.json] [--not-run NAME=REASON ...]
python -m fastfill.comparison.delivery assemble --package NEW_DIR --target NEW_ASSEMBLED_DIR

Refuses unless the report is a complete report_v2 output of stage 'selected' bound (stage_table.load_stage)
to geometry-v5, both arms' dev-selected test inference receipts and RUN/dev-eval-010/selection.json
(stage_table.load_selection: bound to its dev scores, reproduced by dev_select.choose, sha256 = the
report's); carries a delivery decision that report_v2.delivery reproduces from the report's own recorded
inputs; and both arms' candidate lineage chains (dpo-001 SFT merge, then the candidate's dpo-002/dpo-003
stages; stage_table.selected_lineage) bind the evaluated policies and the stage pair summaries. P also
needs RUN/main-data-v1/size-priors.json with the hash RUN/eval-data-v1/MANIFEST.json recorded. The example
prompt is the shortest frozen test request of the delivered arm.

Delivered policy (amendment 011 §5, audit M1): the evaluated policy leads: a DPO candidate's stage adapter
(FP32, unmerged) on its BF16 training base (amendment 009); SFT: the SFT adapter merged at runtime onto the
generic base. The BF16 merge (model/) is listed only with --merged-check (dev_select check-merged's
check.json, bound to the merged model_sha256, the frozen dev uids_sha256, selection.json's sha256 and the
selected candidate's dev a′) and only iff |Δa′| <= 0.01 and |Δcomplete| <= 0.005 (re-applied here); without
the check, or outside the limits, it is not delivered at all. --not-run records agents that were
never run (e.g. O5), beside the final chain's excluded ones. Writes MODEL_CARD.md, delivery.json, SHA256SUMS
(files to copy, as package-relative destinations for `sha256sum -c` after copying) and MANIFEST.json with
status MANIFEST_READY: weights, provenance files and the request code (contract.py, size_prior.py, scene.py
and their package inits, hashed; contract.py, size_prior.py and scene.py must equal the eval-data MANIFEST's
implementation_sha256, i.e. the code that built the evaluated prompts; the git commit only when those files
are clean, and the inference receipt's library versions) are listed, never copied. `assemble` copies every files_to_copy entry into a new directory, verifies each
SHA256 and writes ASSEMBLY.json (status DELIVERY_PACKAGE_ASSEMBLED). Every number comes from the report,
selection.json, the merged check or the receipts; anything absent is 'not measured'.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from fastfill.comparison import report_v2 as R
from fastfill.comparison import stage_table as S
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

DATA_RELEASE = 'FastFill data v3.2'  # research-log 2026-10-13: the J/P models were trained on v3.2.
ARM = {'J': 'joint', 'P': 'placement'}
NAME = {'J': 'J (joint: predicts each object size, position and yaw)',
        'P': f'P (placement: predicts position and yaw; sizes come from the frozen training prior, {R.P_RULE})'}
RULE = ('amendment 005 §1 rule applied to J-selected vs P-selected (amendment 010: each arm\'s model selected on dev '
        'only; replaces J-DPO2 vs P-DPO2 of amendment 008): deliver J iff J−P size log MAE on common-complete rooms '
        'has its paired 95 % CI below zero and the size-debiased success (a′) J−P has 95 % CI lower bound ≥ −0.02; '
        'otherwise P; any undefined interval delivers P')
INPUT_KEYS = ('b_size_log_mae_joint_minus_placement', 'a_prime_debiased_success_joint_minus_placement')
CODE_ROOT = Path(__file__).resolve().parents[2]
# Audit M9: the request code a user runs (contract -> fastfill.scene -> shapely; size_prior is stdlib only).
CODE = ('fastfill/__init__.py', 'fastfill/scene.py', 'fastfill/comparison/__init__.py',
        'fastfill/comparison/contract.py', 'fastfill/comparison/size_prior.py')
IMPLEMENTATION = ('comparison/contract.py', 'comparison/size_prior.py', 'scene.py')  # eval_data.CODE keys.
MERGED_A_PRIME, MERGED_COMPLETE, EPS = 0.01, 0.005, 1e-12  # Amendment 011 §5; EPS keeps the bounds inclusive.
MERGED_KEYS = {'complete': 'complete_rate', 'raw_success': 'geometry_success_rate', 'a_prime': 'a_prime'}
RECIPE_KEYS = ('lr', 'epochs', 'lora_r', 'lora_alpha', 'lora_dropout', 'beta', 'rpo_alpha', 'effective_batch',
               'max_len', 'scheduler', 'target_modules')


def decision(results):
    """The report's delivery decision, refused when absent or not reproducible from its recorded inputs."""
    chosen = results.get('delivery')
    if not isinstance(chosen, dict) or chosen.get('delivered') not in ARM:
        raise ValueError('selected report carries no delivery decision; refusing to package a model')
    inputs = chosen.get('inputs') or {}
    if any(not isinstance(inputs.get(k), dict) for k in INPUT_KEYS):
        raise ValueError('selected report delivery decision lacks its rule inputs; refusing')
    again = R.delivery(*(inputs[k] for k in INPUT_KEYS))
    if any(again[k] != chosen.get(k) for k in ('delivered', 'condition_i_size_log_mae', 'condition_ii_debiased_success')):
        raise ValueError('selected report delivery decision disagrees with the amendment-005 rule on its own inputs')
    return chosen


def load_merged_check(path, model_sha256):
    """Amendment 011 §5 from dev_select check-merged's check.json (merged/unmerged complete_rate,
    geometry_success_rate, a_prime; byte_identical.fraction; merged.model.model_sha256; uids_sha256;
    selection_sha256). Bound here to the delivered merged model (build binds the rest to selection.json); the
    listing rule is re-applied here, never read from the file."""
    path = Path(path)
    value = h.strict_json(path.read_text())
    try:
        sha = value['merged']['model']['model_sha256']
        sides = {name: {k: value[name][alias] for k, alias in MERGED_KEYS.items()} for name in ('merged', 'unmerged')}
        identical = value['byte_identical']['fraction']
        bound = {k: value[k] for k in ('uids_sha256', 'selection_sha256')}
    except (KeyError, TypeError) as exc:
        raise ValueError(f'merged check lacks {exc}') from exc
    if sha != model_sha256:
        raise ValueError('merged check is not bound to the delivered merged model (model_sha256)')
    numbers = [*sides['merged'].values(), *sides['unmerged'].values(), identical]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x <= 1 for x in numbers):
        raise ValueError('merged check values must be rates in [0, 1]')
    delta = {k: sides['merged'][k] - sides['unmerged'][k] for k in MERGED_KEYS}
    listed = abs(delta['a_prime']) <= MERGED_A_PRIME + EPS and abs(delta['complete']) <= MERGED_COMPLETE + EPS
    return {'path': str(path.resolve()), 'sha256': h.sha256(path), 'rooms': value.get('rooms'), **bound, **sides,
            'delta_merged_minus_unmerged': delta, 'identical_fraction': identical,
            'thresholds': {'a_prime': MERGED_A_PRIME, 'complete': MERGED_COMPLETE}, 'merged_listed': listed}


def merged_status(fraction, check):
    """How model/ (the BF16 merge) is delivered: checked within the amendment 011 §5 limits, else not at all."""
    pct = ('fraction of target weights changed not measured' if fraction is None
           else f'{100 * fraction:.2f} % of target weights changed by the BF16 merge')
    if check is None:
        return f'not delivered: amendment 011 §5 merged-weight check not run ({pct}; its outputs were never scored)'
    d = check['delta_merged_minus_unmerged']
    measured = (f"{check['rooms'] or 'the'} dev rooms: Δa′ {d['a_prime']:+.4f}, Δcomplete {d['complete']:+.4f}, "
                f"Δraw success {d['raw_success']:+.4f}, byte-identical outputs {check['identical_fraction']:.4f}")
    if check['merged_listed']:
        return f'convenience merge ({pct}); amendment 011 §5 check passed on {measured}; no test number comes from it'
    return f'not delivered: amendment 011 §5 check failed on {measured} (limits |Δa′| ≤ 0.01, |Δcomplete| ≤ 0.005)'


def recipe(receipt, rpo):
    """Lineage-receipt hyper-parameters; an rpo candidate's DPO stage must record rpo_alpha (audit minor 9)."""
    value = receipt['training']['recipe']
    if not isinstance(value, dict):
        return value
    if rpo and 'rpo_alpha' not in value:
        raise ValueError(f"rpo lineage receipt for {receipt.get('stage')} lacks rpo_alpha")
    return ', '.join(f'{k} {value[k]}' for k in RECIPE_KEYS if k in value)


def _commit(root, paths=CODE):
    """(git HEAD of the code root, whether paths differ from it per git status --porcelain), else (the commit in a
    main-<commit> snapshot directory name, None), else (None, None)."""
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=True).stdout
    try:
        return git('rev-parse', 'HEAD').strip(), bool(git('status', '--porcelain', '--', *paths).strip())
    except (OSError, subprocess.CalledProcessError):
        match = re.fullmatch(r'main-([0-9a-f]{7,40})', root.name)
        return (match.group(1) if match else None), None


def _fmt(value, signed=False):
    if isinstance(value, dict) and 'estimate' in value:
        if value['estimate'] is None:
            return 'not measured'
        ci, f = value.get('ci95'), '{:+.4f}' if signed else '{:.4f}'
        return f.format(value['estimate']) + (' [' + ', '.join(map(f.format, ci)) + ']' if ci else ' [CI not measured]')
    if value is None:
        return 'not measured'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    return f'{value:.4f}' if isinstance(value, float) else str(value)


def _table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + [
        '| ' + ' | '.join(map(str, r)) + ' |' for r in rows] + ['']


def _files(root, files, dest):
    return [{'source': str(Path(root) / n), 'dest': f'{dest}/{n}', 'sha256': h._expected_hash(v)} for n, v in sorted(files.items())]


def _sums(copies):
    return ''.join(f"{c['sha256']}  {c['dest']}\n" for c in copies)


def build(run, report, out, agents=None, *, merged_check=None, not_run=None):
    """agents (dev_select final): {'included': [names], 'excluded': {name: reason}}; default: the report's agents.
    merged_check: amendment 011 §5 check.json; not_run: {name: reason} for agents never run (never substituted)."""
    run, report, out = Path(run), Path(report), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if S._read(report / 'results.json').get('stage') != 'selected':
        raise ValueError('delivery packages only the report of stage selected (amendment 010)')
    selection = S.load_selection(run / 'dev-eval-010' / 'selection.json')
    stage = S.load_stage(run, 'selected', report, selection)
    results = stage['results']
    chosen = decision(results)
    chain, kept = S.selected_lineage(run, stage)
    label = chosen['delivered']
    arm = ARM[label]
    candidate = stage['candidates'][arm]
    name = S.candidate_stage(candidate)
    link, receipt = chain[arm], chain[arm][name]['receipt']
    for s in link:  # Fail before writing anything: an rpo stage without rpo_alpha.
        recipe(link[s]['receipt'], s != 'sft' and candidate.startswith('rpo-'))
    agents = {**(agents or {'included': sorted(results.get('agents') or {}), 'excluded': {}}), 'not_run': dict(not_run or {})}
    if set(agents['not_run']) & (set(agents['included']) | set(agents['excluded'])):
        raise ValueError('an agent cannot be both evaluated/excluded and not run')
    data = run / 'eval-data-v1'
    if h.sha256(data / 'MANIFEST.json') != results['input_sha256']['data_MANIFEST']:
        raise ValueError('eval-data-v1 is not the data the selected report evaluated')
    manifest = S._read(data / 'MANIFEST.json')
    code = {n: h.sha256(CODE_ROOT / n) for n in CODE}  # Audit M9: the shipped code built the evaluated prompts.
    recorded = manifest.get('implementation_sha256') or {}
    if any(code['fastfill/' + k] != recorded.get(k) for k in IMPLEMENTATION):
        raise ValueError('request code differs from the eval-data MANIFEST implementation_sha256 '
                         '(the code that built the evaluated prompts); refusing')
    requests = h._rows(data, f'{arm}/requests.jsonl', manifest['files'][f'{arm}/requests.jsonl'])
    example = min(requests, key=lambda r: (len(r['messages'][1]['content']), r['uid']))
    prior = None
    if label == 'P':
        path = run / 'main-data-v1' / 'size-priors.json'
        expected = h._expected_hash(manifest['input_consumed_sha256']['training']['size-priors.json'])
        if not path.is_file() or h.sha256(path) != expected:
            raise ValueError('P delivery needs the frozen size-priors.json recorded by the evaluation data')
        prior = {'path': str(path.resolve()), 'sha256': expected, 'key': 'full_prior',
                 'use': 'fastfill.comparison.size_prior.predict_sizes(features, prior), each value max(0.01, round(v, 2))'}
    model = Path(receipt['model']['path'])
    if name == 'sft':  # Evaluated: the SFT adapter merged at runtime on GPU onto the generic base (infer-ARM-002).
        policy = {'sft-adapter': receipt['adapter'], 'base': receipt['base']}
    else:  # Evaluated: the stage adapter unmerged, FP32 as trained, on its BF16 training base (amendment 009).
        policy = {'adapter': receipt['adapter'], 'training-base': link[S.STAGES[S.STAGES.index(name) - 1]]['receipt']['model']}
    fraction, _, why = S.merge_fraction(receipt)
    chosen_dev = selection['value']['arms'][arm]
    check = load_merged_check(merged_check, receipt['model_sha256']) if merged_check else None
    if check and (check['uids_sha256'] != selection['value']['uids_sha256'] or check['selection_sha256'] != selection['sha256']
                  or check['unmerged']['a_prime'] != chosen_dev['candidates'][candidate]['a_prime']):
        raise ValueError('merged check is not bound to the frozen dev rooms, this selection.json and the selected '
                         "candidate's dev run (uids_sha256, selection_sha256, unmerged a_prime)")
    listed = check is not None and check['merged_listed']
    infer = stage['receipts'][arm]
    infer_dir = stage['dirs'][arm]
    copies = [c for dest, part in policy.items() for c in _files(part['path'], part['files'], dest)]
    copies += _files(model, receipt['model']['files'], 'model') if listed else []
    copies += [{'source': link[s]['path'], 'dest': f'lineage/{arm}-{s}-merged.json', 'sha256': link[s]['sha256']}
               for s in link]
    copies += [{'source': prior['path'], 'dest': 'size-priors.json', 'sha256': prior['sha256']}] if prior else []
    report_files = {**S._read(report / 'MANIFEST.json')['files'], 'MANIFEST.json': h.sha256(report / 'MANIFEST.json')}
    copies += _files(report.resolve(), report_files, 'report-selected')
    copies.append({'source': str(selection['path'].resolve()), 'dest': 'dev-selection/selection.json',
                   'sha256': selection['sha256']})
    # Audit minor 14: the prior's provenance (input_consumed_sha256) and the inference run identity.
    copies += [{'source': str(p.resolve()), 'dest': dest, 'sha256': h.sha256(p)} for p, dest in (
        (data / 'MANIFEST.json', 'provenance/eval-data-MANIFEST.json'),
        (infer_dir / 'run_receipt.json', 'provenance/inference-run_receipt.json'))]
    copies += [{'source': check['path'], 'dest': 'merged-check/check.json', 'sha256': check['sha256']}] if check else []
    copies += [{'source': str(CODE_ROOT / n), 'dest': f'code/{n}', 'sha256': code[n]} for n in CODE]
    commit, dirty = _commit(CODE_ROOT)
    value = {'status': 'MANIFEST_READY', 'amendments': ['005', '008', '009', '010', '011'], 'rule': RULE,
             'chosen_arm': label, 'arm': arm, 'candidate': candidate, 'candidates': stage['candidates'], 'decision': chosen,
             'dev_selection': {'rule': selection['value']['rule'], 'selection_sha256': selection['sha256'],
                               'uids_sha256': selection['value']['uids_sha256'],
                               'rooms': chosen_dev['candidates']['sft']['requests'],
                               **{k: chosen_dev[k] for k in ('selected', 'eligible', 'tied', 'best_a_prime', 'candidates')}},
             'model': {'path': str(model), 'stage': name, 'model_sha256': receipt['model_sha256'],
                       'model_sha256_definition': 'sha256 of the sorted JSON file->sha256 manifest (dpo_stages lineage receipt)',
                       'files': receipt['model']['files'], 'merge': receipt['merge'],
                       'merge_changed_fraction': fraction, 'merge_changed_fraction_note': why,
                       'listed': listed, 'delivered_as': merged_status(fraction, check),
                       'lineage_receipt': link[name]['path'], 'lineage_receipt_sha256': link[name]['sha256'],
                       'base_revision': receipt['base'].get('revision'), 'base_sha256': receipt['base_sha256'],
                       'training_data_manifest_sha256': receipt['training_data'].get('manifest_sha256')},
             'merged_check': check,
             'evaluated_policy': {'lora': infer['config']['lora'],
                                  **(infer.get('policy') or {'adapter': infer['config']['adapter'],
                                                             'adapter_sha256': infer['adapter_sha256'],
                                                             'base': infer['config']['base']}),
                                  'package_dirs': {dest: str(part['path']) for dest, part in policy.items()},
                                  'inference_run_sha256': infer['run_sha256'], 'inference_dir': str(infer_dir.resolve())},
             'size_prior': prior,
             'inference': {k: infer['config'].get(k) for k in (*S.DECODE, 'cap_base', 'cap_per_object',
                                                               'enable_thinking', 'padding_side')},
             'code': {'git_commit': None if dirty else commit, 'git_dirty': dirty, 'sha256': code, 'package_dir': 'code',
                      'bound_to': 'eval-data MANIFEST.json implementation_sha256 (contract.py, size_prior.py, scene.py equal)',
                      'versions': infer.get('versions'), 'versions_source': 'inference run_receipt.json versions'},
             'report': {'path': str(report.resolve()), 'results_sha256': h.sha256(report / 'results.json'),
                        'manifest_sha256': h.sha256(report / 'MANIFEST.json')},
             'example_request_uid': example['uid'], 'files_to_copy': copies, 'agents': agents}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'delivery.json', value)
        (dest / 'MODEL_CARD.md').write_text(card(value, results, link, kept[arm], example))
        (dest / 'SHA256SUMS').write_text(_sums(copies))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return value


def assemble(package, target):
    """Copy every files_to_copy source of a verified package into a new directory beside the package files and
    verify each SHA256 (what `sha256sum -c SHA256SUMS` checks); nothing is published unless all match."""
    package, target = Path(package), Path(target)
    if target.exists():
        raise FileExistsError(target)
    names = S._read(package / 'MANIFEST.json')['files']
    h.verify_artifacts(package, names)
    value = S._read(package / 'delivery.json')
    copies = value['files_to_copy']
    if (package / 'SHA256SUMS').read_text() != _sums(copies):
        raise ValueError('SHA256SUMS does not list files_to_copy')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + target.name + '-', dir=target.parent) as tmp:
        dest = Path(tmp)
        for name in (*names, 'MANIFEST.json'):
            shutil.copyfile(package / name, dest / name)
        for c in copies:
            path = h._inside(dest, c['dest'])
            if path.exists():
                raise ValueError(f"duplicate package destination: {c['dest']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(c['source'], path)
            if h.sha256(path) != c['sha256']:
                raise ValueError(f"hash mismatch after copy: {c['dest']}")
        receipt = {'status': 'DELIVERY_PACKAGE_ASSEMBLED', 'package': str(package.resolve()),
                   'package_manifest_sha256': h.sha256(package / 'MANIFEST.json'),
                   'sha256sums_sha256': h.sha256(package / 'SHA256SUMS'), 'files_verified': len(copies)}
        h.write_json(dest / 'ASSEMBLY.json', receipt)
        _publish(dest, target)
    return receipt


def request_code(label):
    """Label-free request builder shown in the card (audit M8, minor 11); tests execute it."""
    lines = ['import json', 'from fastfill.comparison.contract import bind_room, messages, validate_output',
             *(['from fastfill.comparison.size_prior import predict_sizes'] if label == 'P' else []),
             '# prepared_room: source, uid, boundary (counter-clockwise, metres), boundary_type, optional height and',
             '# room_type, fixed items (id, category, size, pos, yaw in radians, optional parent) and the movable',
             '# objects with id and category only. bind_room validates size/pos/yaw on every object; the movable',
             '# placeholders below never reach the prompt (it carries movable id and category only).',
             "filled = {**prepared_room, 'objects': [{'size': [1, 1, 1], 'pos': [0, 0, 0], 'yaw': 0, **o}",
             "                                       for o in prepared_room['objects']]}",
             'room, sidecar = bind_room(filled)  # keep the sidecar: it maps the opaque IDs back']
    if label == 'J':
        return lines + ["msgs = messages(room, 'joint')[:2]  # system + user"]
    return lines + ["prior = json.load(open('size-priors.json'))['full_prior']",
                    "features = {'room_type': room.get('room_type'),",
                    "            'objects': [{'id': o['id'], 'category': o.get('category')} for o in room['objects']]}",
                    'sizes = {k: [max(.01, round(v, 2)) for v in vals] for k, vals in predict_sizes(features, prior).items()}',
                    "msgs = messages(room, 'placement', estimated_sizes=sizes)[:2]  # system + user"]


def output_code(label):
    """Strict parse plus ID mapping (and, for P, size attachment) shown in the card; tests execute it."""
    size = "sizes[p['id']]" if label == 'P' else "p['size']"
    return [f"placements = validate_output(raw, room, '{ARM[label]}')['placements']  # strict, as evaluated",
            "back = sidecar['bound_to_original']  # opaque obj_/fixed_ IDs -> your IDs",
            *(['# P outputs no size: each placement takes the size its request carried.'] if label == 'P' else []),
            '# pos = bottom centre in metres; yaw = integer degrees counter-clockwise from +X; on = supporting ID.',
            f"layout = [{{'id': back[p['id']], 'size': {size}, 'pos': p['pos'], 'yaw': p['yaw'],",
            "           **({'on': back[p['on']]} if 'on' in p else {})} for p in placements]"]


def _generate_code(cfg, sft):
    adapter = 'sft-adapter' if sft else 'adapter'
    return ['import torch', 'from peft import PeftModel', 'from transformers import AutoModelForCausalLM, AutoTokenizer',
            f"tok = AutoTokenizer.from_pretrained('{adapter}')  # the evaluation loaded the tokenizer from the adapter",
            f"base = AutoModelForCausalLM.from_pretrained('{'base' if sft else 'training-base'}', dtype=torch.bfloat16,",
            "                                            attn_implementation='sdpa', device_map='auto')",
            ("model = PeftModel.from_pretrained(base, 'sft-adapter').merge_and_unload()  # runtime BF16 merge, as evaluated"
             if sft else "model = PeftModel.from_pretrained(base, 'adapter')  # FP32 adapter kept unmerged, as evaluated"),
            'text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)',
            "ids = tok(text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)",
            f"if ids.shape[1] > {cfg['max_input_tokens']}:",
            "    raise ValueError('prompt too long: rejected, as evaluated')",
            f"cap = min({cfg['max_new_tokens']}, {cfg['context_budget']} - ids.shape[1], "
            f"{cfg['cap_base']} + {cfg['cap_per_object']} * len(room['objects']))",
            "eos = tok.convert_tokens_to_ids('<|im_end|>')  # the only stop token used (not <|endoftext|>)",
            'new = model.generate(ids, max_new_tokens=cap, do_sample=False, num_beams=1, eos_token_id=eos,',
            '                     pad_token_id=eos)[0, ids.shape[1]:].tolist()',
            'if eos not in new:',
            "    raise ValueError('cap reached without <|im_end|>: a failure, as evaluated')",
            'raw = tok.decode(new[:new.index(eos)], skip_special_tokens=False)']


def _agents(value, t, cfg):
    """Audit M6/M7/M12/M13 and amendment 011 §3: model and mode per row, raw (a), OOR, B, a′ and the equal-size
    control (EXPLORATORY), paired Δ(a), Δ(b) and Δ(a′)."""
    cost = {r['system']: r for r in t.get('T5_cost') or []}
    a = value['agents']
    lines = []
    if t['T5']:
        def system(row):
            return cost.get(row['system'].removesuffix('-raw').removesuffix('-harness'), {})

        def mode(row):  # Audit minor 20: the report's mode note (e.g. a template default effort) stays with the mode.
            c = system(row)
            own = (f"greedy, thinking {'on' if cfg['enable_thinking'] else 'off'}" if row['system'] in ARM
                   else c.get('reasoning') or 'not recorded')
            return own + (f"; {c['mode_note']}" if c.get('mode_note') else '')
        lines += ['## Agent benchmark (EXPLORATORY; estimation only, no significance claims)', '',
                  'Same frozen cohort; the agents receive J\'s input (categories only, no size prior), so P is not '
                  'same-input with them: it receives prior sizes they do not. The cohort over-represents the paper '
                  'strata and is not representative of the full test set; P−L here does not describe the test set. '
                  '"harness" = up to 3 attempts whose retry message carries only the exception class (a generic retry, '
                  'not the parser text). Output budgets differ by system (report T5_cost). a′ rescales each system\'s '
                  'sizes by its own footprint bias with positions fixed, using true sizes: a stress test that mostly '
                  'measures how far a system was rescaled, not what it would score with unbiased sizes. O1a/O1b near-zero '
                  'success is a failure of the support (`on`) contract field, not evidence about layout skill. The '
                  'equal-size control (amendment 011) replaces every predicted object x/y size by its true one (height, '
                  'position, yaw and support kept) and re-scores the room: placement quality without size choices.', '']
        lines += _table(['scope', 'system', 'model', 'mode', 'requests', 'complete rate', '(a) raw geometry success',
                         'raw OOR', '(b) size log MAE', 'footprint bias B', 'a′ (EXPLORATORY stress test)',
                         'equal-size control success (EXPLORATORY)'],
                        [[r['scope'], r['system'], system(r).get('model', 'not recorded'), mode(r),
                          _fmt(r['requests']), _fmt(r['complete_rate']), _fmt(r['geometry_success_rate']),
                          _fmt(r.get('oor')), _fmt(r['size_log_mae']), _fmt(r.get('bias_fp')),
                          _fmt(r.get('success_debiased_rate')), _fmt(r.get('equal_size_success_rate'))] for r in t['T5']])
        lines += _table(['scope', 'comparison', 'Δ(a) [95% CI]', 'Δ(b) [95% CI]', 'Δ(a′) [95% CI] (EXPLORATORY)'],
                        [[r['scope'], r['comparison'], _fmt(r['geometry_success'], True), _fmt(r['size_log_mae'], True),
                          _fmt(r.get('own_debiased'), True)] for r in t['T5_contrasts']])
    else:
        lines += ['Agent benchmark: not measured in this report.', '']
    # Amendment 010: every agent re-paired with the selected models; exclusions stated, never substituted.
    lines += [f"Same-input agents compared in the report: {', '.join(a['included']) or 'none'}."
              + (' Excluded (evaluation incomplete; never substituted): '
                 + '; '.join(f'{k} ({v})' for k, v in sorted(a['excluded'].items())) + '.' if a['excluded'] else '')
              + (' Not run (never substituted): ' + '; '.join(f'{k} ({v})' for k, v in sorted(a['not_run'].items())) + '.'
                 if a['not_run'] else ''), '']
    return lines


def card(value, results, link, kept, example):
    t, label, d, dev = results['tables'], value['chosen_arm'], value['decision'], value['dev_selection']
    bias, t6, t2 = ({r[k]: r for r in t[name]} for name, k in (('T1_bias', 'system'), ('T6', 'system'), ('T2', 'scope')))
    m, cfg, sft = value['model'], value['inference'], value['model']['stage'] == 'sft'
    names = {x: f"{x}-selected ({value['candidates'][ARM[x]]})" for x in 'JP'}
    rpo, policy = value['candidate'].startswith('rpo-'), value['evaluated_policy']
    lines = [f"# FastFill unknown-size layout model: {names[label]}", '',
             f'**Delivered arm:** {NAME[label]}.', '',
             f'Chosen by the {RULE}. Delivery is an artifact choice for the user pipeline, not a claim of scientific '
             'superiority. Single training seed. Every number below is copied from the selected-stage report_v2 output '
             f"(results.json sha256 {value['report']['results_sha256']}), the dev selection.json or the lineage receipts, "
             'and every test number belongs to the evaluated policy (adapter + base below), not to the merged `model/`.', '',
             '## Dev selection (amendment 010)', '',
             f"Each arm's model was selected on {dev['rooms']} dev rooms only (never used for training or a test decision; "
             f"uids sha256 `{dev['uids_sha256']}`, selection.json sha256 `{dev['selection_sha256']}`) by the rule: {dev['rule']}. "
             f"{label}: eligible {', '.join(dev['eligible'])}; best dev a′ {_fmt(dev['best_a_prime'])}; within 0.005 of it: "
             f"{', '.join(dev['tied'])}; selected **{dev['selected']}**. The other arm selected "
             f"{value['candidates'][ARM['P' if label == 'J' else 'J']]}.", '']
    lines += _table(['candidate', 'dev complete rate', 'dev size log MAE', 'SFT size log MAE (same rooms)', 'common rooms',
                     'own footprint bias B', "dev a′", 'eligible', 'selected'],
                    [[c, _fmt(x['complete_rate']), _fmt(x['size_log_mae_common_with_sft']), _fmt(x['sft_size_log_mae_common']),
                      _fmt(x['common_with_sft']), _fmt(x['footprint_bias']), _fmt(x['a_prime']), _fmt(c in dev['eligible']),
                      _fmt(c == dev['selected'])] for c, x in dev['candidates'].items()])
    lines += ['## Model', '',
              f"- Base: Qwen/Qwen3-8B, revision `{m['base_revision']}` (lineage root base_sha256 `{m['base_sha256']}`).",
              ('- **Delivered policy (the evaluated one):** the SFT adapter `sft-adapter/` merged at runtime on GPU '
               f"(BF16 merge_and_unload) onto the generic base `base/` (adapter sha256 `{policy.get('adapter_sha256')}`)."
               if sft else
               '- **Delivered policy (the evaluated one, amendment 009):** the stage adapter `adapter/` (FP32, kept '
               f"unmerged; sha256 `{policy.get('adapter_sha256')}`) on its BF16 training base `training-base/` "
               f"(sha256 `{policy.get('training_base_sha256')}`)."),
              f"- `model/` (BF16 merge `{m['path']}`, {m['merge']}, model_sha256 `{m['model_sha256']}`"
              + (", dpo-001's SFT merge = the DPO1 training base" if sft else '') + f"): **{m['delivered_as']}**."
              + ('' if m['listed'] else ' It is not in SHA256SUMS.'),
              f"- Training data: {DATA_RELEASE} (main-data manifest sha256 `{m['training_data_manifest_sha256']}`).",
              ('- Recipe: SFT only, as trained (no eligible DPO candidate beat it on dev by more than 0.005 a′).' if sft else
               f"- Recipe (OptiScene, amendments 008/009; amendment 010 candidate {value['candidate']}): SFT → DPO-stage1 "
               '(consensus alignment; negatives = own SFT samples)'
               + (' → DPO-stage2 (geometric plausibility; negatives = out-of-boundary / overlap position perturbations '
                  'of the positive)' if m['stage'] == 'dpo2' else '')
               + ('. rpo = DPO plus an NLL term on the chosen response, weighted by rpo_alpha (table below).'
                  if rpo else '. e1 = the official README hyper-parameters for 1 epoch per stage.')
               + ' Each DPO stage trains a LoRA on the BF16 merge of the previous stage. Positives are a programmatic '
               'geometry-verified selection, not human approval.'), '']
    rows = []
    for stage in link:
        r = link[stage]['receipt']
        pairs = (r['training'].get('pairs') or {}).get('count')
        steps = (r['training'].get('completion') or {}).get('steps')
        fraction, _, why = S.merge_fraction(r)
        rows.append([stage, recipe(r, stage != 'sft' and rpo), _fmt(pairs) if stage != 'sft' else 'n/a',
                     _fmt(steps) if stage != 'sft' else 'n/a', _fmt(fraction) if why is None else 'not measured (' + why + ')'])
    lines += _table(['stage', 'hyper-parameters (lineage receipt)', 'DPO pairs', 'optimizer steps',
                     'fraction of weights changed by the BF16 merge'], rows)
    lines += ['Pair summaries: ' + ('; '.join(f"{s} kept {p['kept_both_arms']} of {p['positives']} positives (both arms)"
                                              for s, p in kept.items()) or 'n/a (no DPO stage)') + '.', '',
              '## Input / output contract', '',
              'Two chat messages: the fixed system prompt of the arm and a user JSON with the room boundary, '
              'boundary_type, optional height, room_type, fixed items and the movable objects (opaque IDs, categories'
              + (', and sizes from the frozen prior' if label == 'P' else '') + '). The answer is one JSON object '
              '`{"placements": [...]}` in listed order, as the system prompt states'
              + ('; P outputs no sizes.' if label == 'P' else '.') + ' Example: the shortest frozen '
              f"test request of this arm (uid `{example['uid']}`, historical test split; format illustration only).", '',
              '```text', example['messages'][0]['content'], '```', '', '```json', example['messages'][1]['content'], '```', '']
    c = value['code']
    lines += ['## Inference', '',
              ('Evaluated with the SFT LoRA adapter merged at runtime on GPU onto the generic base '
               if sft else 'Evaluated as the exact trained policy ')
              + f"({policy['lora']}) with transformers, greedy decoding (do_sample={cfg['do_sample']}, "
              f"num_beams={cfg['num_beams']}), dtype {cfg['dtype']}, thinking off (enable_thinking={cfg['enable_thinking']}), "
              f"stopping only at `<|im_end|>`, batch {cfg['batch_size']} with {cfg['padding_side']} padding on one GPU, "
              f"context budget {cfg['context_budget']} tokens, at most {cfg['max_input_tokens']} input tokens. The cap was "
              f"shared per batch: min({cfg['max_new_tokens']}, {cfg['context_budget']} − padded batch width, "
              f"{cfg['cap_base']} + {cfg['cap_per_object']} × the most movable objects in the batch); the per-request cap "
              'below approximates it. An answer that reaches the cap without `<|im_end|>` counts as a failure, and an '
              'answer counts as complete only after a strict parse. Batch-1 greedy BF16 outputs can differ token by '
              'token from the batched evaluation.', '',
              'The request code ships in `code/` (git commit '
              + (f"`{c['git_commit']}`" if c['git_commit'] else 'not recorded'
                 + (': the files differ from git HEAD' if c.get('git_dirty') else ''))
              + '; sha256 per file in delivery.json `code.sha256`; contract.py, size_prior.py and scene.py equal the '
              "evaluation data's implementation_sha256, the code that built the evaluated prompts; `fastfill.scene` "
              'needs shapely). Library versions of the evaluation '
              "(inference run receipt): " + (', '.join(f'{k} {v}' for k, v in sorted(c['versions'].items()))
                                             if c['versions'] else 'not recorded') + '.', '',
              '1. Request (no labels needed):', '', '```python', *request_code(label), '```', '',
              '2. Generation with the delivered (evaluated) policy:', '', '```python', *_generate_code(cfg, sft), '```', '',
              '3. Output:', '', '```python', *output_code(label), '```', '']
    if m['listed']:
        lines += [f"`model/` ({m['delivered_as']}) loads as a plain causal LM: replace the `base = …` and `model = …` statements by "
                  "`model = AutoModelForCausalLM.from_pretrained('model', dtype=torch.bfloat16, attn_implementation='sdpa', "
                  "device_map='auto')` (tokenizer from `model`). vLLM was not used for any reported number.", '']
    if value['size_prior']:
        lines += ['P needs the frozen size prior shipped as `size-priors.json` '
                  f"(sha256 `{value['size_prior']['sha256']}`, key `full_prior`); the request code above builds sizes "
                  'exactly as the evaluation data did.', '']
    lines += [f"## Measured results: secondary delivery comparison {names['J']} vs {names['P']} (one shared inference "
              'plan, frozen geometry code, full test set)', '',
              'This comparison decides which arm is delivered (amendment 010: the two dev-selected models, replacing '
              'J-DPO2 vs P-DPO2 of amendment 008); it is secondary, not confirmatory. '
              "The protocol's primary (confirmatory) comparison stays the same-stage matched SFT J0 vs P0 (protocol.md), "
              'reported in the SFT-stage report and the stage table, not in this card. Holm p below adjusts only '
              'within (a), (b) of this comparison.', '',
              f"Intervals: house-component cluster bootstrap, {results['repeats']} draws, seed {results['seed']}; "
              'they cover test-room sampling only, not training randomness.', '']
    rows = [[r['endpoint'], _fmt(r['J']), _fmt(r['P']), _fmt(r['delta'], True), _fmt(r['n']),
             *(_fmt(r[k]) if k in r else 'n/a (tested: (a), (b) only)' for k in ('p', 'p_holm'))] for r in t['T1']]
    complete = {r['field']: r for r in t['T1b']}['complete']['micro']
    rows.append(['complete rate (all requests)', *(_fmt(t6[x]['complete'] / t6[x]['requests']) for x in 'JP'),
                 _fmt(complete, True), _fmt(complete['n']), 'n/a', 'n/a'])
    rows.append(['footprint size bias B = mean log(pred/GT)', _fmt(bias['J']['footprint']), _fmt(bias['P']['footprint']),
                 'n/a', 'n/a', 'n/a', 'n/a'])
    lines += _table(['endpoint', names['J'], names['P'], 'Δ J−P [95% CI]', 'n', 'bootstrap p',
                     'Holm p (within (a), (b); secondary)'], rows)
    every = t2['all']
    lines += [f"Delivery rule: condition (i) {_fmt(d['condition_i_size_log_mae'])}, condition (ii) "
              f"{_fmt(d['condition_ii_debiased_success'])} → **{label}**. Model-free GT reference on the same requests: "
              f"geometry success {_fmt(every['gt_success'])}, axis-aligned OOR {_fmt(every['gt_oor'])}, rotated OOR "
              f"{_fmt(every['gt_oor_rotated'])}; P prior alone: size IoU {_fmt(every['prior_size_iou'])}, size log MAE "
              f"{_fmt(every['prior_size_log_mae'])}. GT is a reference, not a ceiling.", '',
              'a′ is diagnostic only: it rescales each arm\'s sizes by its footprint bias measured against the true test '
              'sizes (positions fixed), which a deployed model cannot know, so no user obtains a′. It serves the delivery '
              'rule and dev selection; the raw rate (a) is what a user gets.', '']
    lines += _agents(value, t, cfg)
    if t['T4_gpt']:
        lines += [f"GPT judge (gpt-6.1-sol; not interchangeable with paper GPT-4o scores): pooled Δ J−P "
                  f"{_fmt(t['T4_gpt'][0]['delta'], True)} on {t['T4_gpt'][0]['n']} both-complete rooms.", '']
    else:
        lines += ['GPT judge: not measured in this report.', '']
    fid = [r for r in t['T4'] if r['system'] in ('J', 'P') and r['fid'] is not None]
    lines += (_table(['stratum', 'system', 'FID', 'shared N', 'flag'],
                     [[r['stratum'], r['system'], _fmt(r['fid']), _fmt(r['fid_n']), r['fid_flag'] or ''] for r in fid])
              if fid else ['FID/KID: not measured in this report.', ''])
    lines += ['Human user rating (UR): not measured.', '',
              '## Package assembly and provenance', '',
              f"Status `{value['status']}`: this directory holds the manifest only. Assemble with "
              '`python -m fastfill.comparison.delivery assemble --package THIS_DIR --target NEW_DIR`, which copies every '
              '`files_to_copy` source to its destination and verifies each SHA256 (equivalently: copy, then '
              '`sha256sum -c SHA256SUMS`). Provenance shipped: the evaluation data MANIFEST '
              '(`provenance/eval-data-MANIFEST.json`, whose `input_consumed_sha256` binds the size prior), the inference '
              'run receipt (`provenance/inference-run_receipt.json`), the lineage receipts, the selected report and '
              'selection.json' + (', and the merged-weight check (`merged-check/check.json`)' if value['merged_check'] else '')
              + '.', '']
    lines += ['## Limitations', '',
              '- Single training seed; intervals reflect test-room sampling only.',
              '- DPO preferences are programmatic (geometry gate, own SFT samples, position perturbations), not human preferences.',
              '- The test split is historical and was used for earlier diagnostics; it is not a fresh blind test.',
              '- Human UR not measured. Geometry success is a bounding-box check with tolerances and category exemptions, '
              'not mesh-level physics.',
              f"- Evaluated policy ≠ `model/`: every number comes from the evaluated policy ({policy['lora']}); "
              f"`model/` is {m['delivered_as']}.",
              '- a′ (and B) need the true test sizes: diagnostic only, not a number a user can obtain.',
              '- Model selection (amendment 010) used dev rooms only and was locked after the amendment-009 DPO1 test '
              'results had been viewed (disclosed in the amendment); no test output was used to select.']
    if label == 'P':  # Audit minor 15: read from the report (T2 all, S2 level G), never hard-coded.
        level = {r['level']: r for r in t.get('S2') or []}.get('G', {})
        rooms, objects = every.get('rooms_with_global_fallback_rate'), every.get('global_fallback_object_rate')
        lines.append('- P\'s weak spot, the global fallback size: '
                     + (f"{100 * rooms:.1f} % of test rooms ({round(rooms * every['requests'])} of {every['requests']}) "
                        f"contain at least one object sized by it, {100 * objects:.1f} % of objects "
                        f"({round(objects * every['objects'])} of {every['objects']})"
                        if None not in (rooms, objects, every.get('objects')) else 'its room and object shares are not measured')
                     + f"; P size log MAE on those objects {_fmt(level.get('P_size_log_mae'))} (report T2, S2).")
    lines.append(f'- Known source defects: trained on {DATA_RELEASE}. The sensitivity rows below are the report\'s own: the '
                 'GT-pass stratum, the declared known-defect test rooms excluded and, when the report was given that frozen '
                 'list, the test rooms changed by the later v3.3 source correction excluded.')
    sensitivity = [r for r in t['S1'] if r['scope'].startswith(('排除', 'GT 通过'))]
    lines += [''] + _table(['sensitivity scope (report wording)', 'requests', 'Δ(a) J−P', 'Δ(b) J−P', "Δ(a′) J−P"],
                           [[r['scope'], _fmt(r['requests']), _fmt(r['geometry_success'], True),
                             _fmt(r['size_log_mae'], True), _fmt(r['debiased'], True)] for r in sensitivity])
    lines += [f"Report warnings: {', '.join(results.get('warnings') or []) or 'none'}.", '']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['assemble']:
        parser = argparse.ArgumentParser(prog='delivery assemble', description=assemble.__doc__)
        for flag in ('package', 'target'):
            parser.add_argument('--' + flag, type=Path, required=True)
        args = parser.parse_args(argv[1:])
        print(json.dumps(assemble(args.package, args.target)))
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('run-root', 'report', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--merged-check', type=Path)
    parser.add_argument('--not-run', action='append', default=[], metavar='NAME=REASON')
    args = parser.parse_args(argv)
    pairs = [item.split('=', 1) for item in args.not_run]
    if any(len(p) != 2 or not all(p) for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
        parser.error('--not-run needs unique NAME=REASON')
    value = build(args.run_root, args.report, args.out, merged_check=args.merged_check, not_run=dict(pairs))
    print(json.dumps({'status': value['status'], 'chosen_arm': value['chosen_arm'], 'model_sha256': value['model']['model_sha256'],
                      'model_listed': value['model']['listed']}))


if __name__ == '__main__':
    main()
