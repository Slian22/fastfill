"""Amendment 008 delivery package: the merged DPO2 model of the arm the DPO2 report's amendment-005 rule chose.

python -m fastfill.comparison.delivery --run-root RUN --report REPORT_V2_DPO2_DIR --out NEW_DIR

Refuses unless the report is a complete report_v2 output bound (stage_table.load_stage) to geometry-v4
and both DPO2 exact-policy inference receipts; carries a delivery decision that report_v2.delivery
reproduces from the report's own recorded inputs; and both arms' lineage chains sft -> dpo1 -> dpo2
bind the evaluated policies and the stage pair summaries. P also needs RUN/main-data-v1/size-priors.json
with the hash RUN/eval-data-v1/MANIFEST.json recorded. The example prompt is the shortest frozen test
request of the delivered arm. Writes MODEL_CARD.md, delivery.json, SHA256SUMS (files to copy, as
package-relative destinations for `sha256sum -c` after copying) and MANIFEST.json. Weights are listed,
never copied. Every number comes from the report or the receipts; anything absent is 'not measured'.
"""
import argparse
import json
from pathlib import Path
import tempfile

from fastfill.comparison import report_v2 as R
from fastfill.comparison import stage_table as S
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish

DATA_RELEASE = 'FastFill data v3.2'  # research-log 2026-10-13: the J/P models were trained on v3.2.
ARM = {'J': 'joint', 'P': 'placement'}
NAME = {'J': 'J (joint: predicts each object size, position and yaw)',
        'P': f'P (placement: predicts position and yaw; sizes come from the frozen training prior, {R.P_RULE})'}
RULE = ('amendment 005 §1 rule applied to J-DPO2 vs P-DPO2 (amendment 008): deliver J iff J−P size log MAE on '
        'common-complete rooms has its paired 95 % CI below zero and the size-debiased success (a′) J−P has '
        '95 % CI lower bound ≥ −0.02; otherwise P; any undefined interval delivers P')
INPUT_KEYS = ('b_size_log_mae_joint_minus_placement', 'a_prime_debiased_success_joint_minus_placement')


def decision(results):
    """The report's delivery decision, refused when absent or not reproducible from its recorded inputs."""
    chosen = results.get('delivery')
    if not isinstance(chosen, dict) or chosen.get('delivered') not in ARM:
        raise ValueError('DPO2 report carries no delivery decision; refusing to package a model')
    inputs = chosen.get('inputs') or {}
    if any(not isinstance(inputs.get(k), dict) for k in INPUT_KEYS):
        raise ValueError('DPO2 report delivery decision lacks its rule inputs; refusing')
    again = R.delivery(*(inputs[k] for k in INPUT_KEYS))
    if any(again[k] != chosen.get(k) for k in ('delivered', 'condition_i_size_log_mae', 'condition_ii_debiased_success')):
        raise ValueError('DPO2 report delivery decision disagrees with the amendment-005 rule on its own inputs')
    return chosen


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


def build(run, report, out):
    run, report, out = Path(run), Path(report), Path(out)
    if out.exists():
        raise FileExistsError(out)
    stage = S.load_stage(run, 'dpo2', report)
    results = stage['results']
    chosen = decision(results)
    chain = {arm: S.lineage(run, arm, ['dpo2']) for arm in h.ARMS}
    S.bind('dpo2', chain, stage['receipts'])
    pair_summaries = {s: S.pairs(run, s, chain)[0] for s in ('dpo1', 'dpo2')}
    label = chosen['delivered']
    arm = ARM[label]
    link, receipt = chain[arm], chain[arm]['dpo2']['receipt']
    data = run / 'eval-data-v1'
    if h.sha256(data / 'MANIFEST.json') != results['input_sha256']['data_MANIFEST']:
        raise ValueError('eval-data-v1 is not the data the DPO2 report evaluated')
    manifest = S._read(data / 'MANIFEST.json')
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
    copies = [{'source': str(model / n), 'dest': f'model/{n}', 'sha256': h._expected_hash(v)}
              for n, v in sorted(receipt['model']['files'].items())]
    copies += [{'source': link[s]['path'], 'dest': f'lineage/{arm}-{s}-merged.json', 'sha256': link[s]['sha256']}
               for s in S.STAGES]
    copies += [{'source': prior['path'], 'dest': 'size-priors.json', 'sha256': prior['sha256']}] if prior else []
    report_files = {**S._read(report / 'MANIFEST.json')['files'], 'MANIFEST.json': h.sha256(report / 'MANIFEST.json')}
    copies += [{'source': str((report / n).resolve()), 'dest': f'report-dpo2/{n}', 'sha256': h._expected_hash(v)}
               for n, v in sorted(report_files.items())]
    infer = stage['receipts'][arm]
    fraction, _, why = S.merge_fraction(receipt)
    value = {'status': 'DELIVERY_PACKAGE_READY', 'amendments': ['005', '008', '009'], 'rule': RULE,
             'chosen_arm': label, 'arm': arm, 'decision': chosen,
             'model': {'path': str(model), 'model_sha256': receipt['model_sha256'],
                       'model_sha256_definition': 'sha256 of the sorted JSON file->sha256 manifest (dpo_stages lineage receipt)',
                       'files': receipt['model']['files'], 'merge': receipt['merge'],
                       'merge_changed_fraction': fraction, 'merge_changed_fraction_note': why,
                       'lineage_receipt': link['dpo2']['path'], 'lineage_receipt_sha256': link['dpo2']['sha256'],
                       'base_revision': receipt['base'].get('revision'), 'base_sha256': receipt['base_sha256'],
                       'training_data_manifest_sha256': receipt['training_data'].get('manifest_sha256')},
             'evaluated_policy': {'lora': infer['config']['lora'], **infer['policy'],
                                  'inference_run_sha256': infer['run_sha256']},
             'size_prior': prior,
             'inference': {k: infer['config'].get(k) for k in (*S.DECODE, 'cap_base', 'cap_per_object',
                                                               'enable_thinking', 'padding_side')},
             'report': {'path': str(report.resolve()), 'results_sha256': h.sha256(report / 'results.json'),
                        'manifest_sha256': h.sha256(report / 'MANIFEST.json')},
             'example_request_uid': example['uid'], 'files_to_copy': copies}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'delivery.json', value)
        (dest / 'MODEL_CARD.md').write_text(card(value, results, chain, pair_summaries, example))
        (dest / 'SHA256SUMS').write_text(''.join(f"{c['sha256']}  {c['dest']}\n" for c in copies))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return value


def card(value, results, chain, pair_summaries, example):
    t, label, arm, d = results['tables'], value['chosen_arm'], value['arm'], value['decision']
    bias, t6, t2 = ({r[k]: r for r in t[name]} for name, k in (('T1_bias', 'system'), ('T6', 'system'), ('T2', 'scope')))
    m, cfg = value['model'], value['inference']
    lines = [f'# FastFill unknown-size layout model: {label}-DPO2', '',
             f'**Delivered arm:** {NAME[label]}.', '',
             f'Chosen by the {RULE}. Delivery is an artifact choice for the user pipeline, not a claim of scientific '
             'superiority. Single training seed. Every number below is copied from the DPO2 report_v2 output '
             f"(results.json sha256 {value['report']['results_sha256']}) or the lineage receipts.", '',
             '## Model', '',
             f"- Base: Qwen/Qwen3-8B, revision `{m['base_revision']}` (lineage root base_sha256 `{m['base_sha256']}`).",
             f"- Delivered weights: merged BF16 model `{m['path']}` ({m['merge']}); model_sha256 `{m['model_sha256']}` "
             '(per-file hashes in delivery.json and SHA256SUMS).',
             f"- Training data: {DATA_RELEASE} (main-data manifest sha256 `{m['training_data_manifest_sha256']}`).",
             '- Recipe (OptiScene, amendments 008/009): SFT → DPO-stage1 (consensus alignment; negatives = own SFT '
             'samples) → DPO-stage2 (geometric plausibility; negatives = out-of-boundary / overlap position '
             'perturbations of the positive). Each DPO stage trains a LoRA on the BF16 merge of the previous stage '
             'and is merged afterwards. Positives are a programmatic geometry-verified selection, not human approval.', '']
    rows = []
    for stage in S.STAGES:
        r = chain[arm][stage]['receipt']
        recipe = r['training']['recipe']
        if isinstance(recipe, dict):
            recipe = ', '.join(f'{k} {recipe[k]}' for k in ('lr', 'epochs', 'lora_r', 'lora_alpha', 'lora_dropout', 'beta',
                                                          'effective_batch', 'max_len', 'scheduler', 'target_modules')
                               if k in recipe)
        pairs = (r['training'].get('pairs') or {}).get('count')
        steps = (r['training'].get('completion') or {}).get('steps')
        fraction, _, why = S.merge_fraction(r)
        rows.append([stage, recipe, _fmt(pairs) if stage != 'sft' else 'n/a', _fmt(steps) if stage != 'sft' else 'n/a',
                     _fmt(fraction) if why is None else 'not measured (' + why + ')'])
    lines += _table(['stage', 'hyper-parameters (lineage receipt)', 'DPO pairs', 'optimizer steps',
                     'fraction of weights changed by the BF16 merge'], rows)
    lines += [f"Pair summaries: stage1 kept {pair_summaries['dpo1']['kept_both_arms']} of "
              f"{pair_summaries['dpo1']['positives']} positives, stage2 kept {pair_summaries['dpo2']['kept_both_arms']} "
              f"of {pair_summaries['dpo2']['positives']} (both arms).", '',
              '## Input / output contract', '',
              'Two chat messages: the fixed system prompt of the arm and a user JSON with the room boundary, '
              'boundary_type, optional height, room_type, fixed items and the movable objects (opaque IDs, categories'
              + (', and sizes from the frozen prior' if label == 'P' else '') + '). The answer is one JSON object '
              '`{"placements": [...]}` in listed order, as the system prompt states. Example: the shortest frozen '
              f"test request of this arm (uid `{example['uid']}`, historical test split; format illustration only).", '',
              '```text', example['messages'][0]['content'], '```', '', '```json', example['messages'][1]['content'], '```', '',
              '## Inference', '',
              f"Evaluated as the exact trained policy ({value['evaluated_policy']['lora']}) with transformers, "
              f"greedy decoding (do_sample={cfg['do_sample']}, num_beams={cfg['num_beams']}), dtype {cfg['dtype']}, "
              f"thinking off (enable_thinking={cfg['enable_thinking']}), batch {cfg['batch_size']} with "
              f"{cfg['padding_side']} padding, context budget {cfg['context_budget']} tokens, at most "
              f"{cfg['max_input_tokens']} input tokens, and max_new_tokens = min({cfg['max_new_tokens']}, "
              f"{cfg['context_budget']} − prompt tokens, {cfg['cap_base']} + {cfg['cap_per_object']} × movable objects). "
              'An answer that reaches the cap without EOS counts as a failure.', '',
              '```python', 'import json, torch',
              'from transformers import AutoModelForCausalLM, AutoTokenizer',
              "path = 'model'  # the copied merged DPO2 directory",
              'tok = AutoTokenizer.from_pretrained(path)',
              "model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='auto')",
              "messages = request['messages'][:2]  # system + user, exactly as in the example",
              'text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)',
              "ids = tok(text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)",
              "objects = len(json.loads(messages[1]['content'])['objects'])",
              f"cap = min({cfg['max_new_tokens']}, {cfg['context_budget']} - ids.shape[1], "
              f"{cfg['cap_base']} + {cfg['cap_per_object']} * objects)",
              'out = model.generate(ids, max_new_tokens=cap, do_sample=False, num_beams=1)',
              "placements = json.loads(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))['placements']",
              '```', '', 'vLLM (same prompt token ids, greedy):', '', '```python',
              'from vllm import LLM, SamplingParams',
              f"llm = LLM(model='model', dtype='bfloat16', max_model_len={cfg['context_budget']})",
              "ids = llm.get_tokenizer()(text, add_special_tokens=False)['input_ids']",
              "answer = llm.generate([{'prompt_token_ids': ids}], SamplingParams(temperature=0, max_tokens=cap))[0].outputs[0].text",
              '```', '', 'vLLM was not used for the reported numbers; its outputs can differ token by token.', '']
    if value['size_prior']:
        lines += ['P needs the frozen size prior shipped as `size-priors.json` '
                  f"(sha256 `{value['size_prior']['sha256']}`); request sizes are built exactly as the evaluation did:", '',
                  '```python', 'from fastfill.comparison.contract import bind_room, messages',
                  'from fastfill.comparison.size_prior import predict_sizes',
                  "prior = json.load(open('size-priors.json'))['full_prior']",
                  'room = bind_room(prepared_room)[0]',
                  "features = {'room_type': room.get('room_type'),",
                  "            'objects': [{'id': o['id'], 'category': o.get('category')} for o in room['objects']]}",
                  'sizes = {k: [max(.01, round(v, 2)) for v in vals] for k, vals in predict_sizes(features, prior).items()}',
                  "request = {'messages': messages(room, 'placement', estimated_sizes=sizes)[:2]}", '```', '']
    else:
        lines += ['J needs no size prior: `request = {\'messages\': messages(bind_room(prepared_room)[0], \'joint\')[:2]}` '
                  '(fastfill.comparison.contract).', '']
    lines += ['## Measured results (J-DPO2 vs P-DPO2, one shared inference plan, frozen geometry code)', '',
              f"Intervals: house-component cluster bootstrap, {results['repeats']} draws, seed {results['seed']}; "
              'they cover test-room sampling only, not training randomness.', '']
    rows = [[r['endpoint'], _fmt(r['J']), _fmt(r['P']), _fmt(r['delta'], True), _fmt(r['n']),
             *(_fmt(r[k]) if k in r else 'n/a (tested: (a), (b) only)' for k in ('p', 'p_holm'))] for r in t['T1']]
    complete = {r['field']: r for r in t['T1b']}['complete']['micro']
    rows.append(['complete rate (all requests)', *(_fmt(t6[x]['complete'] / t6[x]['requests']) for x in 'JP'),
                 _fmt(complete, True), _fmt(complete['n']), 'n/a', 'n/a'])
    rows.append(['footprint size bias B = mean log(pred/GT)', _fmt(bias['J']['footprint']), _fmt(bias['P']['footprint']),
                 'n/a', 'n/a', 'n/a', 'n/a'])
    lines += _table(['endpoint', 'J-DPO2', 'P-DPO2', 'Δ J−P [95% CI]', 'n', 'bootstrap p', 'Holm p'], rows)
    every = t2['all']
    lines += [f"Delivery rule: condition (i) {_fmt(d['condition_i_size_log_mae'])}, condition (ii) "
              f"{_fmt(d['condition_ii_debiased_success'])} → **{label}**. Model-free GT reference on the same requests: "
              f"geometry success {_fmt(every['gt_success'])}, axis-aligned OOR {_fmt(every['gt_oor'])}, rotated OOR "
              f"{_fmt(every['gt_oor_rotated'])}; P prior alone: size IoU {_fmt(every['prior_size_iou'])}, size log MAE "
              f"{_fmt(every['prior_size_log_mae'])}. GT is a reference, not a ceiling.", '']
    if t['T5']:
        lines += ['Agent benchmark (same frozen cohort and inputs; estimation only, no significance claims):', '']
        lines += _table(['scope', 'system', 'requests', 'complete rate', '(a) geometry success', '(b) size log MAE'],
                        [[r['scope'], r['system'], _fmt(r['requests']), _fmt(r['complete_rate']),
                          _fmt(r['geometry_success_rate']), _fmt(r['size_log_mae'])] for r in t['T5']])
        lines += _table(['scope', 'comparison', 'Δ(a) [95% CI]', 'Δ(b) [95% CI]'],
                        [[r['scope'], r['comparison'], _fmt(r['geometry_success'], True), _fmt(r['size_log_mae'], True)]
                         for r in t['T5_contrasts']])
    else:
        lines += ['Agent benchmark: not measured in this report.', '']
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
              '## Limitations', '',
              '- Single training seed; intervals reflect test-room sampling only.',
              '- DPO preferences are programmatic (geometry gate, own SFT samples, position perturbations), not human preferences.',
              '- The test split is historical and was used for earlier diagnostics; it is not a fresh blind test.',
              '- Human UR not measured. Geometry success is a bounding-box check with tolerances and category exemptions, '
              'not mesh-level physics.',
              f"- Evaluated policy ≠ delivered bytes: the numbers come from the exact policy ({value['evaluated_policy']['lora']}); "
              'the delivered model folds that update into BF16 weights (fraction of weights changed by the final merge: '
              f"{_fmt(m['merge_changed_fraction']) if m['merge_changed_fraction_note'] is None else 'not measured'}). "
              'The merged bytes were not re-evaluated.',
              f'- Known source defects: trained on {DATA_RELEASE}. The sensitivity rows below are the report\'s own: the '
              'GT-pass stratum, the declared known-defect test rooms excluded and, when the report was given that frozen '
              'list, the test rooms changed by the later v3.3 source correction excluded.']
    sensitivity = [r for r in t['S1'] if r['scope'].startswith(('排除', 'GT 通过'))]
    lines += [''] + _table(['sensitivity scope (report wording)', 'requests', 'Δ(a) J−P', 'Δ(b) J−P', "Δ(a′) J−P"],
                           [[r['scope'], _fmt(r['requests']), _fmt(r['geometry_success'], True),
                             _fmt(r['size_log_mae'], True), _fmt(r['debiased'], True)] for r in sensitivity])
    lines += [f"Report warnings: {', '.join(results.get('warnings') or []) or 'none'}.", '']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('run-root', 'report', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    value = build(args.run_root, args.report, args.out)
    print(json.dumps({'status': value['status'], 'chosen_arm': value['chosen_arm'], 'model_sha256': value['model']['model_sha256']}))


if __name__ == '__main__':
    main()
