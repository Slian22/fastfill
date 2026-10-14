"""Amendment 022 §3: the delivery package of the system (placer, prior table, S1 and escalation code, CLI, model card).

python -m fastfill.system.package build --run-root RUN --mode interim|final [--out NEW_DIR]
python -m fastfill.system.package assemble --package DIR --target NEW_DIR    (delivery.assemble: copy, verify sha256)

interim: R = rpo-dpo1, labelled 'interim' everywhere (packaged before amendment 021 §4 finishes; changes no decision).
final: the amendment-021 §4 delivered placer (a021/final-selection.json). Default out: RUN/system-022/MODE/package.
Placer = the exact policy that produced the evaluated test rows (evaluate.placer and evaluate.bound; the inference
receipt self-hash and a021_select.bind; evaluate.receipt_policy): its adapter files (copied, each sha256; their
manifest hash must equal the receipt's adapter_sha256) and its training base: a private merged stage model (R's
sft-merged) is copied to placer/training-base, its manifest hash = the policy's training_base_sha256; a public base
under MODELS (generic Qwen3-8B, Qwen3.8-27B) is listed by path and sha256, not copied. Prior table:
main-data-v1/size-priors.json at the sha256 the eval-data MANIFEST pins. Code: the import closure of
fastfill.system.cli plus the lazily imported RoomGenBench export (fastfill.v2.direct_layout and .validation),
computed in a clean interpreter on this code root, copied with sha256, plus the git commit (delivery._commit; None
when dirty).
Measured numbers only from MANIFEST-verified JSON: RUN/system-022/MODE/{cohort,test}/eval/results.json (evaluate.py)
and, when the placer is R, RUN/apply-012-test/results.json (amendment 012 §4: R raw and +S1 on test). Anything
absent is 'not measured'; no number is typed by hand. Writes (write-once) delivery.json (files_to_copy, the
delivery.py layout delivery.assemble reads), MODEL_CARD.md (Chinese + English), SHA256SUMS and MANIFEST.json.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from fastfill.comparison import a021_select as A
from fastfill.comparison import delivery
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import infer
from fastfill.comparison import llm_agent
from fastfill.comparison import llm_open
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.corpus import _publish
from fastfill.system import evaluate as ev
from fastfill.system import pipeline as pl

CODE_ROOT = Path(__file__).resolve().parents[2]
CLOSURE = ('import json, sys; import fastfill.system.cli, fastfill.v2.direct_layout, fastfill.v2.validation; '
           'print(json.dumps(sorted('
           "m.__file__ for n, m in sys.modules.items() if n.split('.')[0] == 'fastfill' and getattr(m, '__file__', None))))")
NM = 'not measured'
STAGE_EN = {'raw': 'placer raw', 's1': '+S1', 'system': '+escalation'}
RATES = ('complete_rate', 'checker_rate', 'strict_rate', 'noreq_rate')


def code_files(root=CODE_ROOT):
    """Repo-relative paths of every fastfill module the system CLI and the RoomGenBench export import."""
    out = subprocess.run([sys.executable, '-B', '-c', CLOSURE], cwd=root, capture_output=True, text=True, check=True,
                         env={**os.environ, 'PYTHONPATH': str(root)}).stdout
    return sorted(str(Path(f).resolve().relative_to(Path(root).resolve())) for f in json.loads(out))


def _verified_json(root, name='results.json'):
    """A MANIFEST-verified JSON of an output dir, or None when the dir has no MANIFEST (not run yet)."""
    if not (Path(root) / 'MANIFEST.json').exists():
        return None
    e._verified(Path(root))
    return h.strict_json((Path(root) / name).read_text())


def _policy(p, run_sha):
    """The inference receipt that produced the bound test rows -> (receipt, policy, adapter files)."""
    receipt = h.strict_json((Path(p['inference']) / 'run_receipt.json').read_text())
    if receipt.get('run_sha256') != run_sha or infer._hash({k: v for k, v in receipt.items() if k != 'run_sha256'}) != run_sha:
        raise ValueError('the inference receipt is not the run the evaluated test rows descend from')
    A.bind(receipt, p['spec'], p['rows_engine'])
    policy = ev.receipt_policy(receipt)
    if not policy.get('adapter') or not policy.get('training_base'):
        raise ValueError('the delivered placer must be an exact policy (adapter on its training base)')
    files = h.artifact_manifest(policy['adapter'])
    if infer._hash(files) != policy['adapter_sha256']:
        raise ValueError(f"adapter bytes differ from the evaluated policy: {policy['adapter']}")
    return receipt, policy, files


def training_base(run, policy):
    """The base files to copy, or None for a public base under MODELS (listed by path and sha256 only)."""
    base = Path(policy['training_base'])
    if base.resolve().is_relative_to(A.models_root(run).resolve()):
        return None
    files = h.artifact_manifest(base)
    if infer._hash(files) != policy.get('training_base_sha256'):
        raise ValueError(f'training base bytes differ from the evaluated policy: {base}')
    return files


def measured(run, mode, candidate):
    """{'cohort': results | None, 'test': results | None, 'apply_012': results | None} (verified JSON only)."""
    value = {scope: _verified_json(ev.paths(run, mode, scope)['eval']) for scope in ev.SCOPES}
    value['apply_012'] = _verified_json(Path(run) / 'apply-012-test') if candidate == A.R else None
    return value


def build(run, mode, out=None, files=code_files):
    run = Path(run)
    out = Path(out or ev.paths(run, mode)['package'])
    if out.exists():
        raise FileExistsError(out)
    p, data = ev.placer(run, mode), run / 'eval-data-v1'
    _, _, run_sha = ev.bound(p, data)
    receipt, policy, adapter_files = _policy(p, run_sha)
    prior = ev.prior_table(run)
    if h.sha256(prior['path']) != prior['sha256']:
        raise ValueError('size-priors.json differs from the table the evaluation data pins')
    code = {n: h.sha256(CODE_ROOT / n) for n in files()}
    commit, dirty = delivery._commit(CODE_ROOT, list(code))
    results = measured(run, mode, p['candidate'])
    base_files = training_base(run, policy)
    copies = delivery._files(policy['adapter'], adapter_files, 'placer/adapter')
    if base_files is not None:
        copies += delivery._files(policy['training_base'], base_files, 'placer/training-base')
    copies.append({'source': str(Path(prior['path']).resolve()), 'dest': 'size-priors.json', 'sha256': prior['sha256']})
    copies += [{'source': str(CODE_ROOT / n), 'dest': f'code/{n}', 'sha256': s} for n, s in code.items()]
    copies += [{'source': str((Path(src)).resolve()), 'dest': dest, 'sha256': h.sha256(src)} for src, dest in (
        (data / 'MANIFEST.json', 'provenance/eval-data-MANIFEST.json'),
        (Path(p['inference']) / 'run_receipt.json', 'provenance/inference-run_receipt.json'),
        *([(run / A.A21 / 'final-selection.json', 'provenance/final-selection.json')] if mode == 'final' else []))]
    for scope in ev.SCOPES:
        if results[scope] is not None:
            root = ev.paths(run, mode, scope)['eval']
            names = {**e._verified(root)['files'], 'MANIFEST.json': h.sha256(root / 'MANIFEST.json')}
            copies += delivery._files(root.resolve(), names, f'evaluation/{scope}')
    if results['apply_012'] is not None:
        copies.append({'source': str((run / 'apply-012-test' / 'results.json').resolve()),
                       'dest': 'evaluation/apply-012-test/results.json',
                       'sha256': h.sha256(run / 'apply-012-test' / 'results.json')})
    serving = 'vllm' if p['engine'] == 'vllm' else 'hf'
    o2 = A.models_root(run) / llm_open.SPECS['O2']['model']
    value = {
        'status': 'MANIFEST_READY', 'protocol': 'amendment-022 §3', 'mode': mode, 'release': 'interim' if mode == 'interim'
        else 'final', 'spec': ev.SPEC,
        'placer': {'candidate': p['candidate'], 'label': p['label'], 'serving_engine': p['engine'],
                   'evaluated_rows_engine': p['rows_engine'], 'model': p['spec'],
                   'training_base': {'path': policy['training_base'], 'sha256': policy.get('training_base_sha256'),
                                     'copied': base_files is not None,
                                     'package_dir': None if base_files is None else 'placer/training-base'},
                   'adapter': {'path': policy['adapter'], 'sha256': policy['adapter_sha256'], 'files': adapter_files,
                               'package_dir': 'placer/adapter'},
                   'lora': receipt['config'].get('lora'), 'inference_run_sha256': run_sha,
                   'inference_dir': str(Path(p['inference']).resolve()),
                   'final_selection_sha256': p['final_selection_sha256'], 'versions': receipt.get('versions')},
        'size_prior': {'path': str(Path(prior['path']).resolve()), 'sha256': prior['sha256'], 'package_file': 'size-priors.json',
                       'use': 'pipeline.load_prior (full_prior) + size_prior.predict_sizes, rounded as eval_data'},
        's1': pl.S1,
        'escalation': {'order': ['gpt-6.1-sol size-fixed (one answer + at most one repair, B2) + S1', 'O2 size-fixed + S1',
                                 'best candidate: checker pass > complete > fewer violations'],
                       'api_model': llm_agent.EXPECTED_MODEL, 'api_rounds': pl.API_ROUNDS,
                       'api_cap_rooms': ev.SCOPES['cohort']['api_rooms'], 'api_cap_calls': ev.SCOPES['cohort']['api_calls'],
                       'llm_agent_global_cap': ev.GLOBAL_CAP, 'o2_model': llm_open.SPECS['O2']['name'], 'o2_path': str(o2)},
        'cli': usage(serving, policy['training_base'] if base_files is None else 'placer/training-base', prior, o2),
        'code': {'git_commit': None if dirty else commit, 'git_dirty': dirty, 'sha256': code, 'package_dir': 'code',
                 'closure': 'import closure of fastfill.system.cli + fastfill.v2.direct_layout, fastfill.v2.validation '
                            '(clean interpreter)'},
        'measured': {k: summary(v, k) for k, v in results.items()}, 'files_to_copy': copies}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as tmp:
        dest = Path(tmp)
        h.write_json(dest / 'delivery.json', value)
        (dest / 'MODEL_CARD.md').write_text(card(value))
        (dest / 'SHA256SUMS').write_text(delivery._sums(copies))
        h.write_json(dest / 'MANIFEST.json', {'files': h.artifact_manifest(dest)})
        _publish(dest, out)
    return value


def usage(serving, base, prior, o2):
    """Run from the assembled package directory (code/ on PYTHONPATH)."""
    return ('PYTHONPATH=code python -m fastfill.system.cli run --requests ROOMS.jsonl --prior size-priors.json '
            f"--prior-sha256 {prior['sha256']} --placer {serving} --model {base} "
            f"--adapter placer/adapter --escalate l+o2 --api-cap {ev.SCOPES['cohort']['api_calls']} "
            f"--api-max-rooms {ev.SCOPES['cohort']['api_rooms']} --env-file ~/.fastfill_api.env --o2-model {o2} "
            '--workers 32 --out NEW_DIR')


def summary(results, kind):
    """The card's numbers from one verified results JSON; None stays None ('not measured' in the card)."""
    if results is None:
        return None
    if kind == 'apply_012':
        levels = results['levels']['test']['P']['all']
        return {'source': 'amendment-012 §4 test application (R raw / +S1, every test request)',
                'stages': {s: {'complete_rate': levels[s].get('complete_rate'), 'checker_rate': levels[s].get('checker_rate'),
                               'strict_rate': levels[s].get('downstream_rate'), 'noreq_rate': None}
                           for s in ('raw', 's1')}}
    esc, api = results['escalation'], results['escalation']['api']
    return {'source': f"amendment-022 §2 evaluate.py ({results['label']})", 'requests': results['requests'],
            'stages': {s: {k: results['levels']['all']['system'][s][k] for k in (*RATES, 'oor', 'size_log_mae', 'bias_fp')}
                       for s in ev.STAGES},
            'bins': {b: {s: results['levels'][b]['system'][s]['checker_rate'] for s in ev.STAGES}
                     for b, _, _ in ev.sc.BINS if b in results['levels']},
            'escalated_share': esc['escalated_share'], 'api_rooms': api['rooms'], 'api_calls': api['calls'],
            'api_cost_usd': None if isinstance(api['cost_usd'], str) else api['cost_usd'],
            'final_stage': esc['final_stage']}


# ---- model card ----------------------------------------------------------------------------------

def _f(x):
    return NM if x is None else delivery._fmt(x)


def _rows(m, stages_name):
    if m is None:
        return [['—', NM, NM, NM, NM, NM]]
    return [[stages_name.get(s, s), *(_f(x.get(k)) for k in (*RATES, 'oor'))] for s, x in m['stages'].items()]


def card(v):
    m, p, esc = v['measured'], v['placer'], v['escalation']
    interim = v['release'] == 'interim'
    tag_zh = '临时版（interim）：放置器 R = rpo-dpo1，在修订 021 §4 完成前打包，不改变任何决定' if interim else \
        '正式版（final）：放置器为修订 021 §4 选定并确认的模型'
    tag_en = ('INTERIM: placer R = rpo-dpo1, packaged before amendment 021 §4 finishes; it changes no decision'
              if interim else 'FINAL: the placer selected and confirmed by amendment 021 §4')
    zh_stage = {'raw': '放置器原始', 's1': '+S1', 'system': '+升级'}
    header_zh = ['阶段', '完整率', '检查器成功', '严格准入', 'required={} 准入', 'OOR']
    header_en = ['stage', 'complete', 'checker success', 'strict admission', 'required={} admission', 'OOR']
    scopes_zh = {'cohort': '代理队列（含 API 访问的系统：gpt-6.1-sol 升级，再 O2）', 'test': '全部测试房间（仅 O2 免费升级的系统）',
                 'apply_012': '全部测试房间，修订 012 §4（R 原始 / +S1，无升级）'}
    scopes_en = {'cohort': 'agent cohort (system with API access: gpt-6.1-sol escalation, then O2)',
                 'test': 'all test rooms (system with free O2 escalation only)',
                 'apply_012': 'all test rooms, amendment 012 §4 (R raw / +S1, no escalation)'}

    def esc_line(scope, zh):
        x, name = m.get(scope), scopes_zh[scope] if zh else scopes_en[scope]
        if x is None:
            return f'- {name}：{NM}' if zh else f'- {name}: {NM}'
        cost = _f(x['api_cost_usd'])
        if zh:
            return (f"- {scopes_zh[scope]}：升级比例 {_f(x['escalated_share'])}，API 房间 {x['api_rooms']}，API 调用 "
                    f"{x['api_calls']}，费用（USD）{cost}，最终来源 {json.dumps(x['final_stage'], ensure_ascii=False)}")
        return (f"- {scopes_en[scope]}: escalated share {_f(x['escalated_share'])}, API rooms {x['api_rooms']}, "
                f"API calls {x['api_calls']}, cost (USD) {cost}, final stage {json.dumps(x['final_stage'])}")

    lines = [f"# FastFill 交付系统 / FastFill delivered system — {v['release']}", '',
             f'**{tag_zh}。**', '', f'**{tag_en}.**', '', '## 中文', '',
             '**做什么。** 输入一个房间（边界多边形、层高、门窗与固定物体框、房间类型）和需要的家具（类别与数量），输出每件家具的'
             ' bbox 尺寸、位置、朝向与支撑父物体（IR 布局），并导出 RoomGenBench 格式（downstream.to_roomgenbench）。', '',
             '**组成（修订 022 §1）。**',
             f"1. 尺寸：训练集先验表 size-priors.json（sha256 {v['size_prior']['sha256']}）。",
             f"2. 放置器：{p['label']}，服务引擎 {p['serving_engine']}（评估所用测试行的引擎：{p['evaluated_rows_engine']}"
             + ('，与部署引擎不同' if p['evaluated_rows_engine'] != p['serving_engine'] else '') + '）；训练基座 '
             + (f"placer/training-base（随包复制，原路径 {p['training_base']['path']}" if p['training_base']['copied']
                else f"{p['training_base']['path']}（公开基座，不随包复制") + f"，sha256 {_f(p['training_base']['sha256'])}）"
             f"+ adapter（placer/adapter，sha256 {p['adapter']['sha256']}）。",
             f"3. S1：确定性最小位移修复（max_trials {v['s1']['max_trials']}，每间 {v['s1']['time_limit']:.0f} 秒）。",
             f"4. 升级（只对 S1 后未完成或检查器失败的房间）：{esc['api_model']} 固定尺寸（一次回答 + 至多一轮修复）→ "
             f"{esc['o2_model']} 固定尺寸 → 最佳候选。API 上限：≤ {esc['api_cap_rooms']} 间 × 2 次 = {esc['api_cap_calls']} 次调用；"
             f"LLM 代理总上限 {esc['llm_agent_global_cap']}。永不返回空结果、永不丢房间。", '',
             '**实测结果。** 每个数字来自已校验的 results.json；缺失写 not measured。比率的分母为全部请求。', '']
    for scope in ('cohort', 'test', 'apply_012'):
        lines += [f'{scopes_zh[scope]}：', '', *delivery._table(header_zh, _rows(m.get(scope), zh_stage))]
    lines += ['**升级开销。**', esc_line('cohort', True), esc_line('test', True), '',
              '**公平性。** 含升级的系统使用了 API 或额外的开源大模型，只作为“含 API 访问的系统”报告，从不作为“模型”与单次'
              '生成的代理比较。网格级碰撞（修订 019）：未测量（系统输出未渲染）。', '',
              '**用法。**', '', '```', v['cli'], '```', '',
              '输入格式见 fastfill/system/pipeline.py 文档（eval-data 联合请求或简单房间 JSON）。', '',
              f"**代码。** git commit {_f(v['code']['git_commit'])}（dirty: {_f(v['code']['git_dirty'])}）；"
              f"{len(v['code']['sha256'])} 个文件随包复制到 code/，逐个 sha256。", '',
              '## English', '',
              '**What it does.** Input: a room (boundary polygon, ceiling height, door / window / fixed boxes, room type) and '
              'the furniture it needs (category and count). Output: every object\'s bbox size, position, yaw and support '
              'parent (IR layout), exported to RoomGenBench (downstream.to_roomgenbench).', '',
              '**Components (amendment 022 §1).**',
              f"1. Sizes: the train prior table size-priors.json (sha256 {v['size_prior']['sha256']}).",
              f"2. Placer: {p['label']}, served on {p['serving_engine']} (evaluated test rows from "
              f"{p['evaluated_rows_engine']}" + (', not the serving engine' if p['evaluated_rows_engine'] !=
                                                 p['serving_engine'] else '') + '); training base '
              + (f"placer/training-base (copied from {p['training_base']['path']}" if p['training_base']['copied']
                 else f"{p['training_base']['path']} (public base, not copied") + ", sha256 "
              f"{_f(p['training_base']['sha256'])}) + adapter (placer/adapter, sha256 {p['adapter']['sha256']}).",
              f"3. S1: deterministic minimum-displacement repair (max_trials {v['s1']['max_trials']}, "
              f"{v['s1']['time_limit']:.0f} s per room).",
              f"4. Escalation (only rooms incomplete or failing the checker after S1): {esc['api_model']} size-fixed (one "
              f"answer + at most one repair round) -> {esc['o2_model']} size-fixed -> best candidate. API cap: <= "
              f"{esc['api_cap_rooms']} rooms x 2 = {esc['api_cap_calls']} calls; LLM-agent total cap "
              f"{esc['llm_agent_global_cap']}. Never empty, never drops a room.", '',
              '**Measured.** Every number comes from a verified results.json; missing ones read not measured. Rates use '
              'every request as the denominator.', '']
    for scope in ('cohort', 'test', 'apply_012'):
        lines += [f'{scopes_en[scope]}:', '', *delivery._table(header_en, _rows(m.get(scope), STAGE_EN))]
    lines += ['**Escalation cost.**', esc_line('cohort', False), esc_line('test', False), '',
              '**Fairness.** The escalated system uses an API or an extra open model; it is reported as a system with API '
              'access and never compared as "the model" against single-shot agents. Mesh collisions (amendment 019): '
              'not measured (the system output is not rendered).', '',
              '**Usage.**', '', '```', v['cli'], '```', '',
              f"**Code.** git commit {_f(v['code']['git_commit'])} (dirty: {_f(v['code']['git_dirty'])}); "
              f"{len(v['code']['sha256'])} files copied to code/, each with sha256. Provenance: provenance/ (eval-data "
              'MANIFEST, the inference run receipt' + (', final-selection.json' if not interim else '') + ').']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--run-root', type=Path, required=True)
    b.add_argument('--mode', choices=('interim', 'final'), required=True)
    b.add_argument('--out', type=Path)
    a = sub.add_parser('assemble')
    a.add_argument('--package', type=Path, required=True)
    a.add_argument('--target', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        value = build(args.run_root, args.mode, args.out)
        print(json.dumps({'status': value['status'], 'release': value['release'], 'files': len(value['files_to_copy'])}))
    else:
        print(json.dumps(delivery.assemble(args.package, args.target)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
