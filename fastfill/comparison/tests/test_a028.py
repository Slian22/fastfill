"""Amendment 028 (a028 + a021_train --arm J27) on CPU: Arm A's identity and argv pinned, J27 rows (rule, overlap, 021
render/EOS assertions), smoke -> smoke-vllm -> train -> greedy -> compare end to end with fakes, and the Slurm scripts
with stub sbatch / squeue / sacct."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as ns

import pytest

from fastfill.comparison import a021_select as A
from fastfill.comparison import a021_train as t
from fastfill.comparison import a028 as a
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import evaluate_llm as ev
from fastfill.comparison import evaluate_pair as e
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import report_v2 as r2
from fastfill.comparison import sample_vllm as sv
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.tests import test_a021_train as TA
from fastfill.comparison.tests import test_dpo_stages as D
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests import test_sample_vllm as SV
from fastfill.v1_1.tests.test_train_dpo_a021 import Tok

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
DOC = 'docs/research/20261013-unknown-size-comparison/amendment-028-27b-joint-arm-20261015.md'


def test_arm_a_identity_and_training_argv_are_unchanged(tmp_path, monkeypatch):
    """Hashes computed with the amendment-021 code before J27 existed: Arm A's served identity (the model part of its
    greedy run identity) and its torchrun argv."""
    files = {'config.json': '1' * 64}
    monkeypatch.setattr(t, 'base_receipt', lambda p: {'path': str(p), 'files': files, 'repository': 'R', 'revision': 'v',
                                                      'model_type': t.MODEL_TYPE})
    root = tmp_path / 'a021/27b'
    final = root / 'train/final'
    TA.write_files(final, {'adapter_config.json': '{}', 'adapter_model.safetensors': b'lora'})
    (root / t.POLICY).write_text(json.dumps({
        'arm': 'A', 'stage': 'a021-27b-sft', 'model_class': t.IMAGE_TEXT, 'base': {'path': str(tmp_path / 'qwen27'), 'files': files},
        'adapter': {'path': str(final.resolve()), 'files': h.artifact_manifest(final)}, 'training_data': {'manifest_sha256': 'a' * 64}}))
    identity = d.inference_identity(ns(base=tmp_path / 'qwen27', model=root, arm='placement', exact_policy=True))
    assert identity.pop('lineage_sha256') == h.sha256(root / t.POLICY)
    text = json.dumps(identity, sort_keys=True).replace(str(tmp_path.resolve()), '<tmp>').replace(str(tmp_path), '<tmp>')
    assert hashlib.sha256(text.encode()).hexdigest() == '855f5720524cb7ebfa857591d5cc7aeb5e21f8006f019a72a2d32fb34990810b'
    with pytest.raises(ValueError, match='a placement policy, not --arm joint'):
        d.inference_identity(ns(base=tmp_path / 'qwen27', model=root, arm='joint', exact_policy=True))
    argv = t.torchrun(Path('/b'), Path('/r/rows/placement.jsonl'), Path('/o'))[1:] + d._flags(t.recipe('A', 900))
    assert hashlib.sha256(json.dumps(argv).encode()).hexdigest() == '1eadb0b47de7189f1c1fef6191e88c65aecf0ed4b74254f7b2ac2d51baf70ead'
    assert t.recipe('J27', 900) == t.recipe('A', 900)  # J27 = Arm A's recipe exactly.
    assert t.parse_args(['smoke', '--pool', 'p', '--base', 'b', '--out', 'o']).arm == 'A'


# ---- rows ----------------------------------------------------------------------------------------

def run_root(tmp_path):
    return tmp_path / 'project/runs/unknownsize-20261013'


def rows_world(tmp_path, monkeypatch, arm_a=('x0', 'h0', 'zz')):
    run = run_root(tmp_path)
    run.mkdir(parents=True)
    data = D.build_main(run)
    runs = D.build_sft(run, data)
    d._publish_dir(run / 'a021/27b/rows', {'placement.jsonl': d._jsonl({'uid': u} for u in arm_a)})
    held = []
    monkeypatch.setattr(op, '_held_out', lambda roots: (set(held), set(), [{'path': str(r)} for r in roots]))
    monkeypatch.setattr(t, 'base_receipt', TA.receipt)
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: Tok())
    monkeypatch.setattr(d, '_tokenizer', lambda path: Tok())
    args = ns(data=data, sft_runs=runs, base27=tmp_path / 'project/models/Qwen3.8-27B', arm_a=run / 'a021/27b',
              dev=[run / 'dev-select-v1'], test=[run / 'eval-data-v1'], out=run / 'a028/rows')
    return ns(run=run, data=data, args=args, held=held)


def test_rows_are_js_sft_rows_in_arm_a_order_with_overlap_and_021_render_checks(tmp_path, monkeypatch):
    w = rows_world(tmp_path, monkeypatch)
    monkeypatch.setattr(t, 'N_MAX', 3)
    result = a.rows(w.args)
    order = sorted(D.ADMITTED, key=lambda u: hashlib.sha256(f'a021-27b:{u}'.encode()).hexdigest())  # The common pool.
    out = w.args.out
    rows = [json.loads(x) for x in (out / 'sft-rows.jsonl').read_text().splitlines()]
    assert [r['uid'] for r in rows] == order and 'h3' not in order  # h3 failed the SFT length admission.
    trained = {r['uid']: r for r in map(json.loads, (w.data / 'joint/train.jsonl').read_text().splitlines())}
    for row in rows:  # J's SFT format, verbatim: no movable sizes in, size + pos + yaw (+ parent) out.
        assert [row['prompt'], row['chosen'], row['rejected']] == [trained[row['uid']]['messages'][:2]] + [
            trained[row['uid']]['messages'][2]['content']] * 2
        assert all('size' not in o for o in json.loads(row['prompt'][1]['content'])['objects'])
        assert all({'size', 'pos', 'yaw'} <= o.keys() for o in json.loads(row['chosen'])['placements'])
    summary = json.loads((out / 'summary.json').read_text())
    selection = summary['selection']
    assert result == {'status': 'A028_ROWS_COMPLETE', 'pool': 8, 'selected': 3,
                      'overlap_with_arm_a': len(set(order[:3]) & {'x0', 'h0'})}
    assert selection['uids_sha256'] == d._digest(order[:3]) and selection['arm_a_rooms'] == 3
    pool = t.load_pool(out)  # a021_train reads the layout unchanged; its N rule takes a prefix of this order.
    assert t.take(pool, '27b', 3) == order[:3] and summary['tokenizers']['27b']['eos_token'] == t.EOS
    with pytest.raises(FileExistsError):
        a.rows(w.args)


def test_rows_refuse_a_wrong_eos_a_non_prefix_render_and_a_dev_test_leak(tmp_path, monkeypatch):
    w = rows_world(tmp_path, monkeypatch)
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: Tok(eos='<|endoftext|>'))
    with pytest.raises(ValueError, match='training EOS'):
        a.rows(w.args)
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: Tok(merge=True))  # '|{' one token: never a prefix.
    with pytest.raises(ValueError, match='no J row renders as a token prefix'):
        a.rows(w.args)
    real = t.lengths  # Even if the ledger called them prefixes, train_dpo.check_render refuses them.
    monkeypatch.setattr(t, 'lengths', lambda tok, rows: [{**x, 'prefix_ok': True, 'fits': True} for x in real(tok, rows)])
    with pytest.raises(ValueError, match='training render differs from infer.prompt_ids'):
        a.rows(w.args)
    monkeypatch.setattr(t, 'lengths', real)
    w.held.append(D.ADMITTED[0])
    with pytest.raises(ValueError, match='share a UID or house group with dev/test'):
        a.rows(w.args)
    assert not w.args.out.exists()


# ---- smoke -> train -> greedy -> compare ---------------------------------------------------------

def eval_data(root, counts, sources=('SceneSmith', 'S1')):
    """SV.make_data's layout; object o0 of every even room has the prior's constant fallback size (a 'G' object)."""
    rooms = []
    for k, count in enumerate(counts):
        objects = [{'id': f'o{j}', 'category': SV.CATEGORIES[j % 4],
                    'size': list(r2.GLOBAL_FALLBACK) if j == 0 and k % 2 == 0 else [0.6, 0.6, 0.8],
                    'pos': [1 + 1.5 * (j % 4), 1 + 1.5 * (j // 4), 0], 'yaw': 0} for j in range(count)]
        rooms.append(bind_room({'uid': f'r{k}', 'source': sources[k % len(sources)], 'group': f'g{k}', 'room_type': 'bedroom',
                                'boundary_type': 'polygon', 'boundary': [[0, 0], [7 + k / 100, 0], [7 + k / 100, 7], [0, 7]],
                                'height': 3, 'objects': objects, 'fixed': []})[0])
    files = {'rooms.jsonl': rooms, 'size-estimates.jsonl': [], 'joint/requests.jsonl': [], 'placement/requests.jsonl': []}
    for room in rooms:
        sizes = {o['id']: o['size'] for o in room['objects']}
        files['size-estimates.jsonl'].append({'uid': room['uid'], 'estimated_sizes': sizes})
        for arm in h.ARMS:
            files[f'{arm}/requests.jsonl'].append({**{k: room[k] for k in ('uid', 'source', 'group')},
                                                   'messages': messages(room, arm, sizes if arm == 'placement' else None)[:2]})
    for name, values in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(''.join(json.dumps(r) + '\n' for r in values))
    h.write_json(root / 'MANIFEST.json', {'expected_uids': [r['uid'] for r in rooms], 'training_manifest_sha256': 'a' * 64,
                                          'files': h.artifact_manifest(root)})
    return root


def p_rows(run, loader, monkeypatch):
    """27B-P's test rows of record, as a021_select.greedy + test_geometry write them (here from an SFT-style identity)."""
    monkeypatch.setattr(sv.infer, 'training_identity', lambda run_: SV.IDENTITY)
    data, out = A.data_dir(run, 'test'), run / 'a021/vllm/test/arm'
    A.greedy(ns(data=data, base=A.models_root(run) / 'Qwen3.8-27B', adapter=run / 'p/placement/adapter', training_run=run / 'p',
                model=None, exact_policy=False, arm='placement', out=out, batching='objects'), loader)
    expected, rooms, requests = a._data(data, 'placement')
    records, receipt = e._inference(out, data, expected)
    payload = [{**e._score((rooms[x['uid']], requests[x['uid']], x, 'placement')), **{k: x[k] for k in ev.CAP_FLAGS}}
               for x in records]
    d._publish_dir(run / 'a021/test-geometry-arm', {'placement-geometry.jsonl': d._jsonl(payload), 'summary.json': A._json(
        {'inference': {'path': str(out.resolve()), 'run_sha256': receipt['run_sha256']}})})


def answer(request, uid, k):
    """J27's fake answers: room r1 overflows its cap (no EOS), r3 is shifted onto one point (collides)."""
    text, finish = SV.spaced(request, uid, k)
    if uid == 'r3':
        text = json.dumps({'placements': [{**o, 'pos': [1, 1, 0]} for o in json.loads(text)['placements']]})
    return (text * 400, 'length') if uid == 'r1' else (text, finish)


def test_j27_smoke_train_greedy_and_compare_end_to_end(tmp_path, monkeypatch):
    w = rows_world(tmp_path, monkeypatch)
    run = w.run
    b27 = TA.fake_base(tmp_path / 'project/models/Qwen3.8-27B')
    (b27 / 'config.json').write_text('{"model_type": "qwen3_5", "vision_config": {}}')
    a.rows(w.args)
    eval_data(run / 'dev-select-v1', [1, 2, 3])
    eval_data(run / 'eval-data-v1', [2, 3, 2, 3, 1, 2, 3, 2])
    monkeypatch.setattr(t, 'SMOKE_STEPS', 1)
    calls, smoke_dir = [], run / 'a028/smoke'
    verdict = t.smoke(ns(pool=w.args.out, base=b27, out=smoke_dir, arm='J27'), TA.fake_torchrun(calls))
    assert verdict['status'] == t.SMOKE_PASS and verdict['n'] == 8 and (smoke_dir / 'rows/joint.jsonl').is_file()
    swap = {str(smoke_dir / 'rows/joint.jsonl'): '/r/rows/placement.jsonl', str(smoke_dir / 'train'): '/o',
            str(b27): '/b', str(smoke_dir / 'perf'): '/p'}
    assert [swap.get(x, x) for x in calls[0][1:]] == t.torchrun(Path('/b'), Path('/r/rows/placement.jsonl'), Path('/o'))[1:] + \
        d._flags(t.recipe('A', 0, perf_out='/p'))  # Arm A's smoke command, J's rows.
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: SV.Tok())
    fake = SV.FakeLLM(run / 'dev-select-v1')
    loaded = t.smoke_vllm(ns(data=run / 'dev-select-v1', smoke=smoke_dir, base=b27, arm='J27'), SV.loader(fake))
    assert loaded['status'] == 'OK' and all('size' not in o for p in fake.calls[0]['prompts']
                                            for o in json.loads(''.join(map(chr, p['prompt_token_ids'])))['objects'])
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: Tok())
    train = ns(arm='J27', pool=w.args.out, base=b27, run_root=run / 'a028', smoke=smoke_dir)
    assert t.train(train, TA.fake_torchrun(calls))['n'] == 8
    policy = json.loads((a.policy_dir(run) / t.POLICY).read_text())
    assert policy['stage'] == 'a028-27b-joint-sft' and policy['arm'] == 'J27'
    assert [json.loads(x)['uid'] for x in (a.policy_dir(run) / 'rows/joint.jsonl').read_text().splitlines()] == t.take(
        t.load_pool(w.args.out), '27b', 8)
    with pytest.raises(ValueError, match='a joint policy, not --arm placement'):
        d.inference_identity(ns(base=b27, model=a.policy_dir(run), arm='placement', exact_policy=True))
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: SV.Tok())
    engines = []
    for slice_ in a.SLICES:
        fake = SV.FakeLLM(A.data_dir(run, slice_), script=answer)
        a.main(['greedy', '--run-root', str(run), '--slice', slice_], SV.loader(fake, engines))
        assert fake.calls[0]['lora'] == ('joint', 1, str(a.policy_dir(run) / 'train/final'))
    assert engines == [{**sv.ENGINE, **sv.MULTIMODAL_ENGINE, 'model': str(b27)}] * 2  # Arm A's 27B engine.
    p_rows(run, SV.loader(SV.FakeLLM(A.data_dir(run, 'test'))), monkeypatch)
    result = a.compare(ns(run_root=run, workers=1, repeats=200))
    value = json.loads((run / 'a028/compare/results.json').read_text())
    J = [json.loads(x) for x in (run / 'a028/test-geometry-j27/joint-geometry.jsonl').read_text().splitlines()]
    assert [r['uid'] for r in J] == [f'r{k}' for k in range(8)] and J[1]['reached_token_cap'] and not J[1]['stopped_on_eos']
    jp = r2.cap_rule(J, J)[0]
    p_success = [json.loads(x)['geometry_success'] for x in
                 (run / 'a021/test-geometry-arm/placement-geometry.jsonl').read_text().splitlines()]
    a_ = value['primary']['geometry_success']
    assert a_['J'] == sum(r['geometry_success'] for r in jp) / 8 and a_['P'] == sum(p_success) / 8 and not jp[1]['complete']
    assert a_['estimate'] == pytest.approx(a_['J'] - a_['P']) and result['geometry_success'] == round(a_['estimate'], 4)
    b = value['primary']['size_log_mae']
    common = [(x['size_log_mae'], y) for x, y in zip(jp, map(json.loads, (run / 'a021/test-geometry-arm/placement-geometry.jsonl')
                                                         .read_text().splitlines())) if x['complete'] and y['complete']]
    assert b['n'] == len(common) == 7 and b['estimate'] == pytest.approx(sum(x - y['size_log_mae'] for x, y in common) / 7)
    assert [a_['p_holm'], b['p_holm']] == r2.holm([a_['p'], b['p']])
    assert value['training'] == {**value['training'], 'n': 8, 'overlap_with_arm_a': 2}
    assert {'all', 'source/SceneSmith', 'prior_fallback_rooms', 'no_prior_fallback_rooms', 'bin/objects<=20',
            'family/bedroom'} <= value['scopes'].keys()
    assert value['scopes']['source/SceneSmith']['requests'] == 4 and set(value['object_levels']) == {'G', 'N'}
    assert value['s1']['J'] is not None and (run / 'a028/s1/test-geometry-j27/MANIFEST.json').exists()
    assert (run / 'a028/s1/test-geometry-arm/MANIFEST.json').exists() and value['dev_select_j27']['requests'] == 3
    assert '修订 028' in (run / 'a028/compare/report.md').read_text()
    with pytest.raises(FileExistsError):
        a.compare(ns(run_root=run, workers=1, repeats=200))
    receipt = run / 'a021/vllm/test/arm/run_receipt.json'  # 27B-P on another engine: no comparison.
    value = json.loads(receipt.read_text())
    receipt.write_text(json.dumps({**value, 'engine': {**value['engine'], 'max_num_seqs': 1024}}))
    os.rename(run / 'a028/compare', run / 'a028/compare.old')
    with pytest.raises(ValueError, match='did not run on the same engine'):
        a.compare(ns(run_root=run, workers=1, repeats=200))


# ---- Slurm scripts -------------------------------------------------------------------------------

def stubs(directory):
    (directory / 'squeue').write_text('#!/bin/sh\ncase "$*" in\n  "-h -u me -o %i|%j") printf "%s" "$SQUEUE_ALL" ;;\n'
                                      '  *) exit 3 ;;\nesac\n')
    (directory / 'sacct').write_text('#!/bin/sh\nF=JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES\ncase "$*" in\n'
                                     '  "-n -X -P -u me -S 2026-10-15 -o $F") printf "%s" "$SACCT_OUT" ;;\n'
                                     '  "-n -X -P -j $SACCT_IDS -o $F") printf "%s" "$SACCT_BY_ID" ;;\n  *) exit 3 ;;\nesac\n')
    for name in ('squeue', 'sacct'):
        (directory / name).chmod(0o755)
    return {**T.stub_sbatch(directory), 'USER': 'me'}


def submit_world(tmp_path):
    project = tmp_path / 'project'
    run = project / 'runs/unknownsize-20261013'
    for path in ('main-data-v1/MANIFEST.json', 'sft-joint-001/SHA256.json', 'sft-placement-001/SHA256.json',
                 'dev-select-v1/MANIFEST.json', 'dev-full-v1/MANIFEST.json', 'eval-data-v1/MANIFEST.json',
                 'a021/test-geometry-arm/MANIFEST.json', 'a021/27b/rows/MANIFEST.json'):
        TA.write_files((run / path).parent, {Path(path).name: '{}'})
    TA.write_files(project / 'models/Qwen3.8-27B', {'config.json': '{}'})
    TA.write_files(project / 'pylib-021', {'.a021-fla-core-0.5.2': 'fla-core==0.5.2\n'})
    (project / 'env/bin').mkdir(parents=True)
    (project / 'env/bin/python').symlink_to(sys.executable)
    script = tmp_path / 'submit_028.sh'
    line = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    original = (SCRIPTS / 'submit_028.sh').read_text()
    assert original.count(line) == 1
    script.write_text(original.replace(line, f'PROJECT_ROOT={project}\n'))
    code = tmp_path / 'main-abc'
    code.mkdir()
    (code / 'fastfill').symlink_to(CODE / 'fastfill')
    TA.write_files((code / DOC).parent, {Path(DOC).name: '# 028\n'})
    return script, project, run / 'a028', code


def test_submit_028_all_chains_every_stage_under_the_gpu_cap_and_refuses_reuse(tmp_path):
    for name in ('isambard_021_train.sbatch', 'isambard_021_eval.sbatch', 'submit_028.sh'):
        subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    env = stubs(tmp_path)
    script, project, a28, code = submit_world(tmp_path)
    sent = []

    def submit(*argv, **extra):
        before = len(sent)
        done = subprocess.run(['bash', str(script), str(code), *argv], env={**env, **extra}, capture_output=True, text=True)
        log = tmp_path / 'log'
        sent[:] = log.read_text().splitlines() if log.exists() else []
        return done, [x.replace(f'{code}/fastfill/comparison/', '').replace(f' {code}', ' CODE').replace(str(project), 'P')
                      for x in sent[before:]]

    def head(name, dep=None):
        return ('--parsable --account=brics.u6uv --partition=workq --qos=normal --job-name=unknownsize-028-' + name +
                f' --output=P/runs/isambard/unknownsize-028-{name}-%j.out')
    gpu4, gpu1 = '--gpus-per-node=4 --cpus-per-task=72 --mem=0', '--gpus-per-node=1 --cpus-per-task=16'
    TR, EV, kill = 'isambard_021_train.sbatch CODE', 'isambard_021_eval.sbatch CODE', '--kill-on-invalid-dep=yes --dependency='
    done, log = submit('all', SACCT_OUT='1|unknownsize-021-train|COMPLETED|36000|720|gres/gpu=4|gres/gpu=4\n'
                                        '2|unknownsize-028-smoke|COMPLETED|3600|90|gres/gpu=4|gres/gpu=4\n')
    assert done.returncode == 0, done.stderr
    assert 'GPU-h: used 4.00 + ceiling 73 <= 80' in done.stdout  # Only unknownsize-028* jobs count; 7 + 56 + 2 + 8.
    assert log == [f'{head("rows")} --hold {TR} rows J27',
                   f'{head("smoke")} {gpu4} --time=01:30:00 {kill}afterok:101 {TR} smoke J27',
                   f'{head("smoke-vllm")} {gpu1} --mem=200G --time=01:00:00 {kill}afterok:102 {TR} smoke-vllm J27',
                   f'{head("train")} {gpu4} --time=7:00:00 {kill}afterok:103 {TR} train J27 7',
                   f'{head("train")} {gpu4} --time=7:00:00 {kill}afterok:103,afternotok:104 {TR} train J27 7',
                   f'{head("infer-dev-select")} {gpu1} --mem=100G --time=02:00:00 {kill}afterok:104?afterok:105 {EV} a028 greedy dev-select 2',
                   f'{head("infer-test")} {gpu1} --mem=100G --time=08:00:00 {kill}afterok:104?afterok:105 {EV} a028 greedy test 8',
                   f'{head("compare")} --cpus-per-task=72 --mem=200G --time=06:00:00 {kill}afterok:106:107 {EV} a028 compare 6']
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101\n'
    done, log = submit('all', SACCT_OUT='2|unknownsize-028-train|COMPLETED|9000|420|gres/gpu=4|gres/gpu=4\n')
    assert log == [] and 'GPU cap: used 10.00 + ceiling 73 > 80' in done.stderr  # 10 + 73 > 80.
    done, log = submit('all', SQUEUE_ALL='7600001|unknownsize-028-train\n7600002|unknownsize-021-eval\n')
    assert log == [] and 'scancel 7600001' in done.stderr and '7600002' not in done.stderr
    (a28 / 'rows').mkdir(parents=True)
    assert f'refusing to reuse existing {a28 / "rows"}' in submit('all')[0].stderr
    assert 'missing input' in submit('smoke')[0].stderr
    TA.write_files(a28 / 'rows', {'MANIFEST.json': '{}'})
    done, log = submit('smoke')  # The chain from the smoke on (e.g. a crashed smoke was moved aside).
    assert done.returncode == 0 and len(log) == 7 and log[0] == f'{head("smoke")} {gpu4} --time=01:30:00 --hold {TR} smoke J27'
    assert 'ceiling 73' in done.stdout
    TA.write_files(a28 / 'smoke', {'verdict.json': json.dumps({'status': 'SMOKE_STOP'})})
    assert 'refusing to reuse existing' in submit('smoke')[0].stderr
    assert 'no SMOKE_PASS verdict' in submit('smoke-vllm')[0].stderr
    TA.write_files(a28 / 'smoke', {'verdict.json': json.dumps({'status': 'SMOKE_PASS'})})
    assert 'missing input' in submit('smoke-vllm')[0].stderr
    TA.write_files(a28 / 'smoke/train/final', {'adapter_model.safetensors': b'lora'})
    done, log = submit('smoke-vllm')  # A failed vLLM load is rerun with the rest of the chain behind it.
    assert done.returncode == 0 and 'ceiling 67' in done.stdout and len(log) == 6
    assert log[0] == f'{head("smoke-vllm")} {gpu1} --mem=200G --time=01:00:00 --hold {TR} smoke-vllm J27'
    assert 'no SMOKE_PASS + vllm-load.json OK' in submit('train')[0].stderr
    TA.write_files(a28 / 'smoke', {'vllm-load.json': '{"status": "OK"}'})
    assert 'refusing to reuse existing' in submit('smoke-vllm')[0].stderr
    done, log = submit('train')
    assert done.returncode == 0 and 'ceiling 66' in done.stdout and log[0].endswith(f'--hold {TR} train J27 7')
    assert 'missing input' in submit('infer')[0].stderr
    TA.write_files(a28 / '27b-joint', {'a021-policy.json': '{}'})
    assert 'refusing to reuse existing' in submit('train')[0].stderr
    TA.write_files(a28 / 'vllm/dev-select/j27', {'MANIFEST.json': '{}'})
    done, log = submit('infer')  # Only the unfinished run, then compare.
    assert done.returncode == 0 and 'ceiling 8' in done.stdout, done.stderr
    assert log == [f'{head("infer-test")} {gpu1} --mem=100G --time=08:00:00 --hold {EV} a028 greedy test 8',
                   f'{head("compare")} --cpus-per-task=72 --mem=200G --time=06:00:00 {kill}afterok:{100 + len(sent) - 1} {EV} a028 compare 6']
    assert 'missing input' in submit('compare')[0].stderr
    TA.write_files(a28 / 'vllm/test/j27', {'MANIFEST.json': '{}'})
    done, log = submit('compare')
    assert done.returncode == 0 and log == [f'{head("compare")} --cpus-per-task=72 --mem=200G --time=06:00:00 --hold {EV} a028 compare 6']
    (a28 / 'compare').mkdir()
    assert 'refusing to reuse existing' in submit('compare')[0].stderr
    for bad in (('bogus',), ('all', 'x'), ()):
        assert submit(*bad)[0].returncode == 2
    os.remove(project / 'pylib-021/.a021-fla-core-0.5.2')
    assert 'no fla overlay' in submit('train')[0].stderr


@pytest.mark.parametrize('script, argv, ok', [
    ('train', ['rows', 'J27'], True), ('train', ['rows'], False), ('train', ['rows', 'A'], False),
    ('train', ['smoke', 'J27'], True), ('train', ['smoke', 'J27', 'x'], False), ('train', ['smoke-vllm', 'J27'], True),
    ('train', ['smoke-vllm', 'B'], False), ('train', ['train', 'J27', '7'], True), ('train', ['train', 'J27'], False),
    ('eval', ['a028', 'greedy', 'dev-select', '2'], True), ('eval', ['a028', 'greedy', 'test', '8'], True),
    ('eval', ['a028', 'greedy', 'confirm', '2'], False), ('eval', ['a028', 'greedy', 'test'], False),
    ('eval', ['a028', 'compare', '6'], True), ('eval', ['a028', 'compare'], False), ('eval', ['a028', 'score', '1'], False)])
def test_sbatch_j27_modes_validate_arguments_before_the_allocation(tmp_path, script, argv, ok):
    done = subprocess.run(['bash', str(SCRIPTS / f'isambard_021_{script}.sbatch'), str(CODE), *argv],
                          env={'PATH': '/usr/bin:/bin'}, capture_output=True, text=True)
    assert (done.returncode, 'Slurm allocation required' in done.stderr) == ((1, True) if ok else (2, False))


@pytest.mark.parametrize('peak, expected_status, expected_reasons', [
    (50 * TA.GIB, t.SMOKE_PASS, []),
    (90 * TA.GIB + 1, t.SMOKE_STOP, ['peak_memory']),
])
def test_j27_slow_smoke_reduces_n_without_bypassing_memory(peak, expected_status, expected_reasons):
    tokens = [3000] * 50_000  # Full N projects to 16.7 h; 24,000 rooms fit the locked ten-hour window.
    measured = t.smoke_verdict(TA.perf(peak=peak), tokens, arm='J27')
    assert measured['status'] == expected_status and measured['reasons'] == expected_reasons
    assert measured['n'] == 24_000 and measured['projected_hours_n'] == pytest.approx(10)
    assert measured['rules']['stop_projected_hours'] is None
    assert measured['rules']['reduce_n_above_projected_hours'] == 12
    original = t.smoke_verdict(TA.perf(peak=peak), tokens)
    assert original['status'] == t.SMOKE_STOP and 'projected_hours' in original['reasons']
    assert original['rules'] == t.RULES


def test_j27_smoke_stops_if_no_room_fits_the_training_window():
    result = t.smoke_verdict(TA.perf(seconds=60_000), [3000] * 4, arm='J27')
    assert result['status'] == t.SMOKE_STOP and result['reasons'] == ['no_fitting_rooms']
    assert result['n'] == 0


def test_j27_smoke_dispatches_its_recorded_reduced_n_policy(tmp_path, monkeypatch):
    w = rows_world(tmp_path, monkeypatch)
    b27 = TA.fake_base(tmp_path / 'project/models/Qwen3.8-27B')
    a.rows(w.args)
    monkeypatch.setattr(t, 'SMOKE_STEPS', 1)
    tokens = [e['tokens']['27b']['total'] for e in t.load_pool(w.args.out)['ledger']
              if e['pool'] and e['tokens']['27b']['fits']]
    rate = sum(t.step_tokens(tokens)) / (16 * 3600)
    calls = []

    def measured_run(command):
        completed = TA.fake_torchrun(calls)(command)
        output = Path(command[command.index('--perf_out') + 1])
        for record in TA.perf(seconds=1000 / rate):
            (output / f"perf-rank{record['rank']}.json").write_text(json.dumps(record))
        return completed

    verdict = t.smoke(ns(pool=w.args.out, base=b27, out=w.run / 'a028/smoke', arm='J27'), measured_run)
    assert verdict['status'] == t.SMOKE_PASS
    assert 0 < verdict['n'] < len(tokens) and verdict['projected_hours_n'] <= t.FIT_HOURS
    assert verdict['projected_hours_n_max'] == pytest.approx(16)
