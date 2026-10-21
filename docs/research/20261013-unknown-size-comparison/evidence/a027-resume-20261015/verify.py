"""Read-only validation around an unchanged amendment-027 recovery run."""
import json
from pathlib import Path
import sys

from fastfill.comparison import infer, train_pair_helpers as h
from fastfill.system import evaluate as ev, pipeline as pl

ROOT = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill')
OLD = ROOT / 'snapshots/main-498348274'
RUN = ROOT / 'runs/unknownsize-20261013'
HERE = Path(__file__).resolve().parent
HASHES = {'pipeline.py': 'b8956e1b36677fc2ddc0f8c1b42ed9489f6bcc9ec02d4ddacdfed88f8aacdfda',
          'evaluate.py': '3e1538bca317279f01ba803a424fedb8db738b0011994494eb3f52b685c36123',
          'isambard_022.sbatch': '46d6608d2c6bda97ebf7df544831356f35b80dbd4ecf752c5eef587eaa72622a'}


def check_inputs():
    assert Path(pl.__file__).resolve() == OLD / 'fastfill/system/pipeline.py'
    assert all(h.sha256(OLD / 'fastfill/system' / name) == digest for name, digest in HASHES.items())
    out, plan, _, records = ev._planned(RUN, 'final', 'strict')
    assert (plan['status'], plan['rooms'], plan['shards']) == ('PLAN_COMPLETE', 8647, 8)
    assert plan['code_sha256']['pipeline.py'] == HASHES['pipeline.py']
    assert len(plan['escalated']) == len(set(plan['escalated'])) == 6155
    assert plan['shard_rooms'][1] == 775
    baseline, _ = ev.load_pipeline(ev.paths(RUN, 'final', 'test', True)['system'])
    assert baseline['placer'] == plan['placer'] and baseline['prior'] == plan['prior']
    files = [out['plan'] / 'plan.json', out['plan'] / 'placer.jsonl', out['requests']]
    for index in (0, 2, 3, 4, 5, 6, 7):
        directory = out['shards'] / f'shard-{index}'
        files.extend([directory / 'identity.json', directory / 'COMPLETE.json'])
        files.extend(sorted((directory / 'chunks').glob('*.json')))
    preserved = {str(p): h.sha256(p) for p in files}
    saved = HERE / 'preserved.json'
    if saved.exists():
        assert json.loads(saved.read_text()) == preserved, 'Previously completed artifacts changed'
    return out, plan, records, preserved


def check_shards(out, plan, records, all_complete):
    keep = set(plan['escalated'])
    for index in range(8):
        mine = [r for r in records if r['uid'] in keep and pl.shard_of(r['uid'], 8) == index]
        directory = out['shards'] / f'shard-{index}'
        identity = pl._shard_identity(mine, ROOT / 'models/Qwen3.8-27B', index, 8, 256)[3]
        assert json.loads((directory / 'identity.json').read_text()) == identity
        if index == 1 and not all_complete:
            assert not (directory / 'COMPLETE.json').exists()
            continue
        marker = json.loads((directory / 'COMPLETE.json').read_text())
        expected = {'status': 'SHARD_COMPLETE', 'index': index, 'shards': 8, 'rooms': len(mine),
                    'chunks': len(identity['sharding']['chunks']), 'identity_sha256': infer._hash(identity)}
        assert marker == expected, f'Invalid completion marker: shard {index}'
        assert set(pl._shard_bodies(directory, identity)) == {r['uid'] for r in mine}


def check_final(out, plan, records):
    identity, rows = ev.load_pipeline(out['system'])
    assert identity['escalation_trigger'] == 'strict' and identity['api_calls'] == identity['api_cap'] == 0
    assert len(rows) == 8647 and {r['uid'] for r in rows} == {r['uid'] for r in records}
    assert identity['placer'] == plan['placer'] and identity['prior'] == plan['prior']
    assert identity['o2_sharding']['shards'] == 8
    assert identity['o2_sharding']['plan_sha256'] == h.sha256(out['plan'] / 'plan.json')
    receipts = [json.loads(line) for line in (out['system'] / 'o2/receipts.jsonl').read_text().splitlines()]
    assert len(receipts) == len(plan['escalated']) and {r['uid'] for r in receipts} == set(plan['escalated'])
    result = json.loads((out['eval'] / 'results.json').read_text())
    assert (result['status'], result['scope'], result['mode'], result['requests'], result['escalation_trigger']) == (
        'SYSTEM_022_EVAL_COMPLETE', 'test', 'final', 8647, 'strict')
    assert result['size_changed_objects'] == 0
    assert result['inputs']['system']['api_calls'] == 0
    assert result['levels']['all']['system']['system']['complete_rooms'] == 8647
    assert 'replay_prediction' in result and 'checker_trigger_system' in result['inputs']
    assert any(c['comparison'] == f'system - {ev.CHECKER_RUN}' for c in result['contrasts'])
    manifest = json.loads((out['eval'] / 'MANIFEST.json').read_text())['files']
    assert all(h.sha256(out['eval'] / name) == digest for name, digest in manifest.items())
    return {'status': 'RECOVERY_VERIFIED', 'rooms': 8647, 'escalated': len(receipts),
            'level': result['levels']['all']['system']['system'], 'api_calls': 0,
            'eval_manifest_sha256': h.sha256(out['eval'] / 'MANIFEST.json')}


if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode in ('capture', 'before-shard', 'before-finalize', 'after-summarize')
    out, plan, records, preserved = check_inputs()
    check_shards(out, plan, records, mode in ('before-finalize', 'after-summarize'))
    if mode in ('capture', 'before-shard'):
        assert not out['system'].exists() and not out['eval'].exists()
    if mode == 'capture':
        with (HERE / 'preserved.json').open('x') as stream:
            json.dump(preserved, stream, indent=2)
    if mode == 'after-summarize':
        report = check_final(out, plan, records)
        with (HERE / 'verification.json').open('x') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report), flush=True)
    print(f'VERIFIED {mode}: pinned code, plan, identities, receipts and seven preserved shards', flush=True)
