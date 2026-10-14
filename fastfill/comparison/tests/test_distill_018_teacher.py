"""Amendment 018 teacher data (distill_018 plan / teacher, llm_open --all-rooms, submit_018_teacher.sh) on CPU: fakes
stand in for vLLM, the main data, the SFT runs and Slurm."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

from fastfill.comparison import contract
from fastfill.comparison import distill_018 as ds
from fastfill.comparison import dpo_stages as d
from fastfill.comparison import geometry as g
from fastfill.comparison import llm_agent as L
from fastfill.comparison import llm_open as O
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.tests import test_dpo_stages as D
from fastfill.comparison.tests import test_llm_open as T
from fastfill.comparison.tests import test_onpolicy_dpo as OPT
from fastfill.comparison.tests.test_llm_open import world  # noqa: F401  Fixture: the llm_open eval data.

CODE = Path(__file__).resolve().parents[3]
SCRIPTS = CODE / 'fastfill/comparison'
B1, B2, B3 = ds.BINS


# ---- pure selection ------------------------------------------------------------------------------

def test_draw_takes_thirds_spills_to_the_next_denser_bin_and_is_proportional_and_hash_ranked():
    pool = {'s0': (B1, 'A'), **{f'm{k}': (B2, 'A') for k in range(4)}, 'm4': (B2, 'B'), 'd0': (B3, 'A'), 'd1': (B3, 'C')}
    uids, info = ds.draw(pool, 6, 'a018-distill:')
    # Quotas 2 / 2 / 2: the sparse bin has 1 room and passes 1 on; 21-50 draws 3 of 5 (A 2.4 -> 2, B 0.6 -> 1 by
    # largest remainder); > 50 draws its 2.
    assert [info['bins'][b]['drawn'] for b in ds.BINS] == [1, 3, 2] and info['shortfall'] == 0
    assert info['bins'][B2] == {'pool': 5, 'quota': 2, 'carried_in': 1, 'drawn': 3, 'by_source': {'A': 2, 'B': 1}}
    ranked = sorted((u for u in pool if pool[u] == (B2, 'A')), key=lambda u: ds.rank('a018-distill:', u))
    assert uids == sorted(['s0', *ranked[:2], 'm4', 'd0', 'd1'])
    assert ds.draw(dict(reversed(list(pool.items()))), 6, 'a018-distill:') == (uids, info)  # Order free.
    assert ds.rank('a018-dpo2:', 'u') == __import__('hashlib').sha256(b'a018-dpo2:u').hexdigest()
    _, info = ds.draw(pool, 10, 'x:')  # 4 / 3 / 3: the densest bin's rest is the shortfall.
    assert [info['bins'][b]['drawn'] for b in ds.BINS] == [1, 5, 2] and info['shortfall'] == 2
    assert ds.draw({}, 3, 'x:')[0] == [] and ds.draw({}, 3, 'x:')[1]['shortfall'] == 3


def test_teacher_shards_are_stratified_round_robin_and_partition_the_set():
    cells = {f'u{k:02d}': ((B1, B2, B3)[k % 3], 'AB'[k % 2]) for k in range(20)}
    shards = ds.shard_uids(list(cells), cells, 3)
    assert sorted(u for s in shards for u in s) == sorted(cells) and [len(s) for s in shards] == [7, 7, 6]
    for s in shards:
        assert s == sorted(s)
        assert max(sum(cells[u][0] == b for u in s) for b in ds.BINS) - min(
            sum(cells[u][0] == b for u in s) for b in ds.BINS) <= 1
    assert ds.shard_uids(list(reversed(cells)), cells, 3) == shards


# ---- plan ----------------------------------------------------------------------------------------

def grid(uid, source, n, k):
    """n spaced 0.4 m boxes on a 0.8 m grid in a (9 + k / 100) x 9 m room (k keeps every request distinct)."""
    objects = [D.obj(f'o{j}', ('cabinet', 'sofa', 'wardrobe')[j % 3], .6 + .8 * (j % 10), .6 + .8 * (j // 10),
                     size=(.4, .4, .5)) for j in range(n)]
    return {**D.raw_room(uid, source, objects), 'boundary': [[0, 0], [9 + k / 100, 0], [9 + k / 100, 9], [0, 9]]}


SOURCES = {'h': 'OptiScene_holodeck', 'f': 'IL3D_3dfront', 'x': 'Structured3D'}
COUNTS = (3, 4, 5, 25, 26, 27, 52, 53)
ROOMS = [grid(f'{p}{j}', source, n, 10 * i + j) for i, (p, source) in enumerate(SOURCES.items())
         for j, n in enumerate(COUNTS)]


@pytest.fixture
def teacher_world(tmp_path, monkeypatch):
    """main data (24 rooms, 3 sources x 3 bins) -> SFT runs -> positives (5) -> T-A (amendment-013 subset, 4 rooms)."""
    monkeypatch.setattr(d, 'TARGETS', D.TARGETS)
    monkeypatch.setattr(D, 'ADMITTED', sorted(r['uid'] for r in ROOMS if r['uid'] != 'x7'))
    data = D.build_main(tmp_path, ROOMS)
    runs = D.build_sft(tmp_path, data)
    positives = tmp_path / 'dpo-001/positives'
    d.positives(D.args(data=data, sft_runs=runs, out=positives, workers=1))
    w = types.SimpleNamespace(root=tmp_path, data=data, runs=runs, positives=positives,
                              dev=[OPT.held(tmp_path / 'dev-select-v1', ['dev0']),
                                   OPT.held(tmp_path / 'dev-full-v1', ['dev0', 'dev1'])],
                              test=[OPT.held(tmp_path / 'eval-data-v1', ['t0'])], ta=tmp_path / 'dpo-004/train-subset')
    op.subset(OPT.subset_args(w, w.ta, rooms=4))
    return w


def plan_argv(w, out, dev=None, test=None, ta=None, tb=6, tc=6, shards=3):
    return ['plan', '--data', str(w.data), '--sft-runs', *map(str, w.runs), '--exclude', str(w.positives),
            '--ta', str(ta or w.ta), '--dev', *map(str, dev or w.dev), '--test', *map(str, test or w.test),
            '--out', str(out), '--tb', str(tb), '--tc', str(tc), '--shards', str(shards)]


def test_plan_builds_ta_tb_tc_deterministically_disjoint_and_refuses_leaks(teacher_world):
    w = teacher_world
    out = w.root / 'distill-018/plan'
    assert ds.main(plan_argv(w, out)) == 0
    manifest = h.strict_json((out / 'MANIFEST.json').read_text())
    plan = ds.load_plan(out)
    sets = plan['sets']
    ta = h.strict_json((w.ta / 'MANIFEST.json').read_text())['expected_uids']
    positives = set(d.load_positives(w.positives)[0]['uids'])
    assert sets['T-A'] == ta and len(sets['T-B']) == 6
    # 14 pool rooms; T-B (weights 7:2:0, addendum B) takes no > 50 room, so T-C fills.
    assert len(sets['T-C']) == 6 and manifest['selection']['T-C']['shortfall'] == 0
    assert ds.bin_quotas(4500, ds.TB_WEIGHTS) == dict(zip(ds.BINS, (3500, 1000, 0)))
    assert ds.bin_quotas(3000) == dict(zip(ds.BINS, (1000, 1000, 1000))) and ds.bin_quotas(7) == dict(zip(ds.BINS, (3, 2, 2)))
    assert not (set(ta) & set(sets['T-B']) or set(ta) & set(sets['T-C']) or set(sets['T-B']) & set(sets['T-C']))
    assert not positives & set(manifest['expected_uids']) and 'x7' not in manifest['expected_uids']  # x7: not admitted.
    assert manifest['expected_uids'] == sorted([*ta, *sets['T-B'], *sets['T-C']]) == plan['cohort']['uids']
    # T-B / T-C are draw() on the pool (admitted - positives - T-A), then on the pool minus T-B.
    rooms = {r['uid']: r for r in ROOMS}
    cells = {u: (ds.bin_of(len(rooms[u]['objects'])), rooms[u]['source']) for u in D.ADMITTED}
    pool = {u: c for u, c in cells.items() if u not in positives | set(ta)}
    assert (sets['T-B'], manifest['selection']['T-B']) == ds.draw(pool, 6, 'a018-distill:', ds.TB_WEIGHTS)
    rest = {u: c for u, c in pool.items() if u not in set(sets['T-B'])}
    assert (sets['T-C'], manifest['selection']['T-C']) == ds.draw(rest, 6, 'a018-dpo2:')
    assert ds.draw(rest, len(rest) + 2, 'a018-dpo2:')[1]['shortfall'] == 2  # The densest bin's rest is the shortfall.
    assert [sum(cells[u][0] == b for u in sets['T-B']) for b in ds.BINS] == [5, 1, 0]
    # The layout is the amendment-013 subset's: T-A rows are byte-identical to dpo-004/train-subset's.
    for name in ('joint/requests.jsonl', 'placement/requests.jsonl', 'rooms.jsonl', 'size-estimates.jsonl'):
        mine = [x for x in (out / name).read_text().splitlines() if json.loads(x)['uid'] in set(ta)]
        assert mine == (w.ta / name).read_text().splitlines()
    assert manifest['sft_prompt_tokens'] == {u: d._admitted(w.runs, h.load_pair(w.data)[1])[1][u]['placement']
                                              for u in manifest['expected_uids']}
    assert [len(s) for s in plan['shards']] == manifest['selection']['shards']['rooms'] == [4, 3, 3]
    assert sorted(u for s in plan['shards'] for u in s) == sets['teacher'] == sorted([*ta, *sets['T-B']])
    # Deterministic: a second build is byte-identical (cohort included).
    again = w.root / 'again'
    ds.main(plan_argv(w, again))
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()} == {
        p.relative_to(again): p.read_bytes() for p in again.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        ds.main(plan_argv(w, out))
    leak = OPT.held(w.root / 'leak-dev', ['dev0', sorted(pool)[0]])
    with pytest.raises(ValueError, match='share a UID or house group'):
        ds.main(plan_argv(w, w.root / 'x', dev=[leak]))
    house = OPT.held(w.root / 'leak-test', ['t9'], {'t9': rooms[ta[0]]['group']})  # A T-A house, another UID.
    with pytest.raises(ValueError, match='share a UID or house group'):
        ds.main(plan_argv(w, w.root / 'y', test=[house]))
    forged = w.root / 'forged-ta'
    shutil.copytree(w.ta, forged)
    lines = (forged / 'placement/requests.jsonl').read_text().splitlines()
    (forged / 'placement/requests.jsonl').write_text('\n'.join(lines[1:] + lines[:1]) + '\n')
    manifest_ta = h.strict_json((forged / 'MANIFEST.json').read_text())
    h.write_json(forged / 'MANIFEST.json', {**manifest_ta, 'files': {**manifest_ta['files'], 'placement/requests.jsonl':
                                                                         h.sha256(forged / 'placement/requests.jsonl')}})
    with pytest.raises(ValueError, match='T-A rows differ'):
        ds.main(plan_argv(w, w.root / 'z', ta=forged))
    assert not any((w.root / n).exists() for n in ('x', 'y', 'z'))


def test_load_plan_refuses_shards_that_differ_from_their_rebuild(teacher_world):
    w = teacher_world
    out = w.root / 'plan'
    ds.main(plan_argv(w, out))
    forged = w.root / 'forged'
    shutil.copytree(out, forged)
    a, b = (h.strict_json((forged / f'shards/s0{k}.json').read_text()) for k in (0, 1))
    a['uids'][0], b['uids'][0] = b['uids'][0], a['uids'][0]  # Same teacher set, other shards.
    for k, value in enumerate((a, b)):
        h.write_json(forged / f'shards/s0{k}.json', {**value, 'uids': sorted(value['uids'])})
    manifest = h.strict_json((forged / 'MANIFEST.json').read_text())
    for name in ('cohort.json', 'MANIFEST.json'):
        (forged / name).unlink()
    h.write_json(forged / 'MANIFEST.json', {**manifest, 'files': h.artifact_manifest(forged)})
    h.write_json(forged / 'cohort.json', O.all_rooms_cohort(forged))
    with pytest.raises(ValueError, match='shards differ'):
        ds.load_plan(forged)


# ---- llm_open on the plan + teacher filter -------------------------------------------------------

def answer(room, edit=None):
    """P's GT answer for ROOM (the SFT label), optionally edited."""
    value = json.loads(contract.messages(room, 'placement', {o['id']: o['size'] for o in room['objects']})[-1]['content'])
    if edit:
        edit(value['placements'], room)
    return contract._json(value)


def collide(placements, room):
    placements[1]['pos'] = list(placements[0]['pos'])


def near_wall(placements, room):
    """Raw: 30 deg box 0.4 m wide reaches W + 0.0992 (inside the 10 cm OOB tolerance); at cm precision W + 0.1032."""
    width = max(x for x, _ in room['boundary'])
    placements[0]['pos'][0], placements[0]['yaw'] = round(width - .174, 3), 30


def shard_argv(root, plan, k):
    return ['run', '--system', 'O2', '--data', str(plan), '--cohort', str(plan / 'cohort.json'),
            '--model-root', str(root / 'models'), '--out', str(ds.shard_run(root, k)), '--arm', 'placement',
            '--all-rooms', '--uids', str(plan / 'shards' / f's{k:02d}.json')]


def test_llm_open_reads_the_plan_and_the_filter_round_trips_targets(teacher_world):
    w = teacher_world
    plan = w.root / 'distill-018/plan'
    ds.main(plan_argv(w, plan))
    info = ds.load_plan(plan)
    rooms = {u: {**r, '_placement': info['requests'][u]} for u, r in info['rooms'].items()}
    model = w.root / 'models/Qwen3.8-27B'
    model.mkdir(parents=True)
    (model / 'config.json').write_text('{"model_type": "fake"}')
    teacher = info['sets']['teacher']
    bad, clash, wall, repaired, capped, sized = teacher[:6]

    def resize(placements, room):  # Addendum B2 drops every returned size: the target is still P's SFT label.
        for p in placements:
            p['size'] = [9, 9, 9]
    with_sizes = answer(rooms[sized], resize)
    script = {bad: ['bad', 'worse', 'still bad'], clash: [answer(rooms[clash], collide)],
              wall: [answer(rooms[wall], near_wall)], repaired: ['bad', '<think>b</think>{valid}'],
              capped: [('<think>a</think>{valid}', 'length')], sized: ['<think>s</think>' + with_sizes]}
    fake = T.FakeLLM(rooms, script, default='<think>plan</think>\n\n{valid}', arm='placement')
    for k in range(3):
        assert O.main(shard_argv(w.root, plan, k), T.loader(fake)) == 0
    first = [c for c in fake.calls if len(c['messages'][0]) == 3]
    assert sorted(u for c in first for u in c['uids']) == teacher  # T-C is never sent to the teacher.
    for call in first:  # P's exact request, the harness suffix, O2's template.
        assert call['messages'] == [[*info['requests'][u]['messages'], {'role': 'user', 'content': L.SUFFIX}]
                                    for u in call['uids']]
        assert call['kwargs'] == {'enable_thinking': True}
    receipt = json.loads((ds.shard_run(w.root, 0) / 'run_receipt.json').read_text())
    assert receipt['config']['cohort_rule'] == 'all_rooms' and receipt['config']['request_arm'] == 'placement'
    assert receipt['cohort']['subset']['sha256'] == h.sha256(plan / 'shards/s00.json')
    # Resume: a rerun with the same argv finds the published run and changes nothing.
    before = (ds.shard_run(w.root, 0) / 'MANIFEST.json').read_bytes()
    assert O.main(shard_argv(w.root, plan, 0), T.loader(T.FakeLLM(rooms, arm='placement'))) == 0
    assert (ds.shard_run(w.root, 0) / 'MANIFEST.json').read_bytes() == before
    # The amendment-016 section 2 O2 run stands in as --reference (same config; no cohort_rule).
    L.main(['cohort', '--data', str(plan), '--out', str(w.root / 'ref-cohort.json')])
    (w.root / 'ref-uids.json').write_text(json.dumps(json.loads((w.root / 'ref-cohort.json').read_text())['uids'][:1]))
    reference = w.root / 'llm-open-O2-P-pilot-001'
    assert O.main(['run', '--system', 'O2', '--data', str(plan), '--cohort', str(w.root / 'ref-cohort.json'),
                   '--model-root', str(w.root / 'models'), '--out', str(reference), '--arm', 'placement',
                   '--uids', str(w.root / 'ref-uids.json')], T.loader(fake)) == 0
    out = w.root / 'distill-018/teacher'
    argv = ['teacher', '--plan', str(plan), '--run-root', str(w.root), '--reference', str(reference), '--out', str(out)]
    assert ds.main(argv) == 0
    order = [json.loads(x)['uid'] for x in (out / 'ledger.jsonl').read_text().splitlines()]
    ledger = {r['uid']: r for r in map(json.loads, (out / 'ledger.jsonl').read_text().splitlines())}
    assert sorted(ledger) == teacher and order == [u for s in info['shards'] for u in s]
    assert {u: ledger[u]['class'] for u in (bad, clash, wall, repaired, capped, sized)} == {
        bad: 'incomplete', clash: 'checker_fail', wall: 'rounding_broke', repaired: 'target', capped: 'incomplete',
        sized: 'target'}
    assert ledger[wall]['answer']['geometry_success'] and ledger[wall]['target_checks']['oob_objects'] == 1
    assert ledger[clash]['answer']['main_collisions'] == 1 and ledger[repaired]['rounds'] == 2
    targets, summary = ds.load_targets(out)
    assert [t['uid'] for t in targets] == [u for u in order if ledger[u]['class'] == 'target']
    assert len(targets) == len(teacher) - 4 == summary['targets']
    for t in targets:  # GT answers: the rebuilt target is exactly P's SFT label, re-parsed and re-scored.
        room = rooms[t['uid']]
        sft = contract.messages(room, 'placement', {o['id']: o['size'] for o in room['objects']})
        assert t['prompt'] == sft[:2] == info['requests'][t['uid']]['messages'] and t['target'] == sft[2]['content']
        assert g.score(room, t['prompt'], t['target'], 'placement')['geometry_success']
        assert t['set'] in ('T-A', 'T-B') and t['shard'] == ledger[t['uid']]['shard']
        returned = {repaired: t['target'], sized: with_sizes}.get(t['uid'], '\n\n' + t['target'])  # After </think>.
        assert t['answer_sha256'] == L._digest(returned)
    assert summary['classes'] == {'target': len(teacher) - 4, 'incomplete': 2, 'checker_fail': 1, 'rounding_broke': 1}
    assert summary['drops'] == {'rounding_broke': 1} and len(summary['shards']) == 3
    sized_objects = len(rooms[sized]['objects'])
    assert {k: summary['size_enforcement'][k] for k in ('answers_size_returned', 'answers_size_changed', 'size_changed')
            } == {'answers_size_returned': 1, 'answers_size_changed': 1, 'size_changed': sized_objects}
    assert all(s['same_versions_as_reference'] for s in summary['shards'])
    assert sum(v['rooms'] for v in summary['pass_rate']['by_bin'].values()) == len(teacher)
    assert '教师目标' in (out / 'report.md').read_text()
    with pytest.raises(FileExistsError):
        ds.main(argv)
    # A shard run whose config differs from the reference beyond cohort_rule / versions is refused.
    other = w.root / 'other-reference'
    other.mkdir()
    value = json.loads((reference / 'run_receipt.json').read_text())
    value['config']['suffix'] += ' '
    (other / 'run_receipt.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='differs from the amendment-016'):
        ds.main([*argv[:-4], '--reference', str(other), '--out', str(w.root / 't2')])


def test_filter_judges_an_answer_and_its_training_target_identically(teacher_world):
    """Round trip on one room: the answer with mm positions -> cm target that re-parses to the same layout."""
    w = teacher_world
    plan = w.root / 'plan'
    ds.main(plan_argv(w, plan))
    info = ds.load_plan(plan)
    uid = info['sets']['T-B'][0]
    room, request = info['rooms'][uid], info['requests'][uid]

    def jitter(placements, room):
        for p in placements:
            p['pos'] = [p['pos'][0] + .0012, p['pos'][1] - .0031, p['pos'][2]]
    text = answer(room, jitter)
    row = {'uid': uid, 'source': room['source'], 'group': room['group'], 'arm': 'llm_harness', 'raw': text, 'error': None}
    result = ds.judge((room, request, row))
    assert result['class'] == 'target' and result['target'] == answer(room)  # cm rounding recovers the SFT label.
    assert result['answer'] == result['target_checks']
    assert ds._same_layout(text, result['target'], room)
    assert ds.judge((room, request, {**row, 'raw': 'not json'}))['class'] == 'incomplete'


# ---- llm_open identity ---------------------------------------------------------------------------

# _identity_shape of each system's P (--limit 2) and P-pilot (--uids, 3 rooms) run on test_llm_open's fixture under
# llm_open before amendment 018 (1b0d7cdaf): --all-rooms must leave them unchanged.
PLACEMENT_IDENTITY_SHAPE = {
    'O1a': ('76bd01afe6c6b0e41493c57151d62f16f52db77a28c11a8249978d32582720f0',
            'b2704a55c019f80b7c710d8b24afec548b84d75870f55e194267fe65aa183e98'),
    'O1b': ('2f066aa10da04267b0f30e94e34ecb5582f159bf00c486a83784c5e4a51f8319',
            '823cce9cbc726415ee4bec41a210fb6f34517266b884df0a45fe53fd454850df'),
    'O2': ('815d18a22c2d4f0604f65ab5e67a98d9789f71a988a686a55b050ffb7c9a4fac',
           '5571d5402aa56e736277deb596c93cd01003b033f2a01e4f63f9a8e5f88e0b9f'),
    'O3': ('cfcef07455b6f47c16a077a607eb98aafc230dec8dc9b11b54c16374b8d9cde8',
           'd6e2b679c580a87f989003ab3ec96be3bf535ced197a444ab29bdb567f1aff16'),
    'O4': ('47a01de2305a069f9062e0eb735356e532cf3fc01ce906d47080726f240163ee',
           '6b5b768d9e5c08d8d271de09a5398cfae20077b89e0c5339dd865ae074bc41f3'),
    'O5': ('511892f18f77e35deed12077815171ae730d4dbf7a06555aea7840c7ee71f43f',
           'ffc7099aabe704447ef855513265ee7c40f88641912edc48d9b264cdf28e1c30'),
    'O6': ('e677039e7c1eed3d56db30802fd7abd88821da63dbca8c7f8ccdb490e34f22ed',
           '60118a15e2501590c7bcad76af44a116b34f7f15e4d3673c2200543eec48e61b'),
    'O7': ('04d3ebaf8f3070d44d5ec18157bcc24a17905bee9c1d839462a06930373527bc',
           'afbb731e5b5000a24582bf6c12a413e6098e986a7176c2c4f2196a466eef05ef'),
}


def test_all_rooms_leaves_existing_identities_unchanged_and_binds_its_own_cohort(world):
    root, rooms = world
    cohort = json.loads((root / 'cohort.json').read_text())['uids']
    (root / 'pilot.json').write_text(json.dumps({'uids': [cohort[7], cohort[2], cohort[30]]}))
    assert list(PLACEMENT_IDENTITY_SHAPE) == list(O.SYSTEMS)
    for system, shapes in PLACEMENT_IDENTITY_SHAPE.items():
        for tag, extra, shape in (('P', ('--limit', '2'), shapes[0]), ('pilot', ('--uids', str(root / 'pilot.json')), shapes[1])):
            out = f'{system}-{tag}'
            O.main(T.argv(root, system, out, '--arm', 'placement', *extra), T.loader(T.FakeLLM(rooms, arm='placement')))
            identity = json.loads((root / out / 'run_receipt.json').read_text())
            assert T._identity_shape(identity) == shape, out
            assert 'cohort_rule' not in identity['config']
    # --all-rooms binds every request of the data, and only its own cohort file.
    every = O.all_rooms_cohort(root / 'data')
    assert every['uids'] == json.loads((root / 'data/MANIFEST.json').read_text())['expected_uids'] != cohort
    (root / 'all.json').write_text(json.dumps(every))
    with pytest.raises(ValueError, match='deterministic rebuild'):  # The test cohort under --all-rooms.
        O.main(T.argv(root, 'O2', 'a', '--arm', 'placement', '--all-rooms', '--limit', '1'), T.loader(T.FakeLLM(rooms)))
    argv = [*T.argv(root, 'O2', 'b', '--arm', 'placement', '--limit', '1')]
    argv[argv.index('--cohort') + 1] = str(root / 'all.json')
    with pytest.raises(ValueError, match='deterministic rebuild'):  # The all-rooms cohort without --all-rooms.
        O.main(argv, T.loader(T.FakeLLM(rooms, arm='placement')))
    assert O.main([*argv, '--all-rooms'], T.loader(T.FakeLLM(rooms, arm='placement'))) == 0
    config = json.loads((root / 'b/run_receipt.json').read_text())['config']
    assert config['cohort_rule'] == 'all_rooms'
    reference = json.loads((root / 'O2-P/run_receipt.json').read_text())['config']
    assert {k: v for k, v in config.items() if k != 'cohort_rule'} == {**reference, 'limit': 1}


# ---- sbatch / submit -----------------------------------------------------------------------------

@pytest.mark.parametrize('argv, ok', [
    (['O2', 't018', 's00'], True), (['O2', 't018', 's07'], True), (['O2', 't018'], False), (['O2', 't018', 's7'], False),
    (['O2', 't018', 'placement'], False), (['O6', 't018', 's00'], False), (['O2', 'full', 's00'], False),
    (['O2', 'pilot', 'placement'], True), (['O2', 'full'], True)])
def test_llm_open_sbatch_t018_mode_validates_arguments_before_the_allocation(argv, ok):
    run = subprocess.run(['bash', str(SCRIPTS / 'isambard_llm_open.sbatch'), str(CODE), *argv], capture_output=True,
                         text=True, env={'PATH': '/usr/bin:/bin'})
    assert (run.returncode, 'Slurm allocation required' in run.stderr) == ((1, True) if ok else (2, False)), argv
    sbatch = (SCRIPTS / 'isambard_llm_open.sbatch').read_text()
    assert 'DATA="$PAIR_RUN_ROOT/distill-018/plan"; COHORT="$DATA/cohort.json"; OUT="llm-open-$SYSTEM-T018-$4-001"' in sbatch
    assert 'EXTRA=(--arm placement --all-rooms --uids "$DATA/shards/$4.json")' in sbatch
    # Side-by-side shards compile into their own cache; every other mode keeps .cache-jobs/llm-open/$SYSTEM.
    assert 'CACHE_TAG="-T018-$4"' in sbatch and '/.cache-jobs/llm-open/$SYSTEM${CACHE_TAG-}"\n' in sbatch
    assert sbatch.count('CACHE_TAG') == 2


PILOT_TOKENS = {**{f'p{k}': (5, [600, 400] if k == 0 else [1000]) for k in range(8)}, 'p8': (30, [3000]),
                'p9': (60, [6000])}  # uid: (objects, output tokens per round); 8 / 1 / 1 rooms per bin, mostly sparse.
TA_BINS = {5: 2000, 30: 600, 60: 400}  # T-A objects: rooms (bins 2,000 / 600 / 400).


def submit_world(tmp_path, generation=25.5, wall=335.5):
    """submit_018_teacher.sh with PROJECT_ROOT = tmp_path (inputs made; O2 pilot: PILOT_TOKENS on eval-data-v1,
    GENERATION and WALL seconds; T-A: TA_BINS) and stub sbatch / squeue / scontrol; the env python runs this
    interpreter."""
    helper = (SCRIPTS / 'submit_018_teacher.sh').read_text()
    pin = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'
    assert helper.count(pin) == 1
    bin_ = tmp_path / 'bin'
    bin_.mkdir(parents=True)
    script = bin_ / 'submit_018_teacher.sh'
    script.write_text(helper.replace(pin, f'PROJECT_ROOT={tmp_path}\n'))
    python = tmp_path / 'env/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    (bin_ / 'sbatch').write_text('#!/bin/sh\necho "$@" >> "$SBATCH_LOG"\nn=$(wc -l < "$SBATCH_LOG")\n'
                                 'if [ "$n" -eq "${FAIL_ON:-0}" ]; then exit 1; fi\necho "$((100 + n));cluster"\n')
    (bin_ / 'scontrol').write_text('#!/bin/sh\necho "$@" >> "$SCONTROL_LOG"\n')
    (bin_ / 'squeue').write_text('#!/bin/sh\n'
                                 'test "$*" = "-h -u me -n unknownsize-018-plan,unknownsize-018-teacher,unknownsize-018-filter -o %i"'
                                 ' || exit 3\nprintf "%s" "$SQUEUE_OUT"\n')
    python.chmod(0o755)
    for name in ('sbatch', 'scontrol', 'squeue'):
        (bin_ / name).chmod(0o755)
    run = tmp_path / 'runs/unknownsize-20261013'
    for name in ('main-data-v1/MANIFEST.json', 'sft-joint-001/SHA256.json', 'sft-placement-001/SHA256.json',
                 'dpo-001/positives/MANIFEST.json', 'dpo-004/train-subset/MANIFEST.json', 'dev-select-v1/MANIFEST.json',
                 'dev-full-v1/MANIFEST.json', 'eval-data-v1/MANIFEST.json', 'llm-open-O2-P-pilot-001/MANIFEST.json',
                 'llm-open-O2-P-pilot-001/run_receipt.json'):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text('{}')
    (tmp_path / 'models/Qwen3.8-27B').mkdir(parents=True)
    (tmp_path / 'models/Qwen3.8-27B/config.json').write_text('{}')
    pilot = run / 'llm-open-O2-P-pilot-001'
    h.write_json(pilot / 'cost.json', {'system': 'O2', 'gpus': 2, 'wall_seconds': wall, 'generation_seconds': generation})
    h.write_json(pilot / 'result.json', {'status': 'LLM_COMPLETE', 'rooms': len(PILOT_TOKENS)})
    h.write_json(pilot / 'run_receipt.json', {'config': {'system': 'O2', 'request_arm': 'placement'},
                                              'data': {'manifest_sha256': h.sha256(run / 'eval-data-v1/MANIFEST.json')}})
    (pilot / 'receipts').mkdir()
    for uid, (_, tokens) in PILOT_TOKENS.items():
        h.write_json(pilot / 'receipts' / f'{uid}.json', {'uid': uid, 'answers': [{'usage': {'output_tokens': t}}
                                                                                  for t in tokens]})
    def rooms(path, counts):
        path.write_text(''.join(json.dumps({'uid': uid, 'objects': [{}] * n}) + '\n' for uid, n in counts))
    rooms(run / 'eval-data-v1/rooms.jsonl', [(u, n) for u, (n, _) in PILOT_TOKENS.items()] + [('q0', 60)])
    rooms(run / 'dpo-004/train-subset/rooms.jsonl', [(f'a{n}-{k}', n) for n, c in TA_BINS.items() for k in range(c)])
    env = {'PATH': f'{bin_}:/usr/bin:/bin', 'SBATCH_LOG': str(bin_ / 'log'), 'SCONTROL_LOG': str(bin_ / 'scontrol.log'),
           'USER': 'me'}
    return script, run, env


def test_submit_018_teacher_chain_runs_shards_side_by_side_under_the_gpu_cap(tmp_path):
    script, run, env = submit_world(tmp_path)
    out = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True, check=True)
    log = (tmp_path / 'bin/log').read_text().splitlines()
    assert len(log) == 10
    plan = f'{run}/distill-018/plan'
    assert log[0].startswith('--parsable --hold --account=brics.u6uv ') and '--job-name=unknownsize-018-plan' in log[0]
    assert log[0].endswith(f'--dev {run}/dev-select-v1 {run}/dev-full-v1 --test {run}/eval-data-v1 --out {plan} --shards 8')
    assert f'distill_018 plan --data {run}/main-data-v1 --sft-runs {run}/sft-joint-001 {run}/sft-placement-001 ' in log[0]
    assert f'--exclude {run}/dpo-001/positives --ta {run}/dpo-004/train-subset ' in log[0]
    # Pilot: 0.0015 s per output token, mean 1,000 / 3,000 / 6,000 tokens per room by bin, load 310 s. Teacher bins
    # 5,500 / 1,600 / 400 (T-A + T-B at 7:2:0, addendum B) -> at most 688 / 200 / 50 per shard: 1.5 x (310 + 2,382) /
    # 60 = 67.3 -> 68 minutes on O2's 2 GPUs.
    flags = O.slurm_flags('O2', 'full').rsplit(' --time=', 1)[0]
    for k, line in enumerate(log[1:9]):
        assert line == (f'--parsable --kill-on-invalid-dep=yes --dependency=afterok:101 --job-name=unknownsize-018-teacher '
                        f'{flags} --time=01:08:00 {SCRIPTS}/isambard_llm_open.sbatch {CODE} O2 t018 s{k:02d}')
        assert 'singleton' not in line
    assert '--dependency=afterok:102:103:104:105:106:107:108:109 ' in log[9] and '--job-name=unknownsize-018-filter' in log[9]
    assert log[9].endswith(f'distill_018 teacher --plan {plan} --run-root {run} --reference '
                           f'{run}/llm-open-O2-P-pilot-001 --out {run}/distill-018/teacher --workers 32')
    assert (tmp_path / 'bin/scontrol.log').read_text() == 'release 101\n'
    assert 'released 101 (GPU ceiling 1088 GPU-minutes: 8 shards x 01:08:00 x 2 GPUs)' in out.stdout
    basis = json.loads(out.stdout.split('time estimate: ', 1)[1].splitlines()[0])
    assert basis['teacher_rooms'] == dict(zip(ds.BINS, (5500, 1600, 400))) and basis['minutes'] == 68
    assert basis['pilot_rooms'] == dict(zip(ds.BINS, (8, 1, 1))) and basis['load_seconds'] == 310
    assert 'then the filter: sbatch --dependency=afterok:SHARD_JOB --account=brics.u6uv' in out.stdout
    assert '--workers 32"' in out.stdout and log[9].endswith('--workers 32')
    (tmp_path / 'bin/log').unlink()
    subprocess.run(['bash', str(script), str(CODE), '4'], env=env, capture_output=True, check=True)
    log = (tmp_path / 'bin/log').read_text().splitlines()  # 1,375 / 400 / 100: 1.5 x (310 + 4,762.5) / 60 -> 127 min.
    assert len(log) == 6 and all('--time=02:07:00' in line for line in log[1:5]) and log[4].endswith('O2 t018 s03')
    (tmp_path / 'bin/log').unlink()
    bad = subprocess.run(['bash', str(script), str(CODE)], env={**env, 'FAIL_ON': '4'}, capture_output=True, text=True)
    assert bad.returncode == 1 and 'scancel 101 102 103' in bad.stderr
    assert (tmp_path / 'bin/scontrol.log').read_text() == 'release 101\nrelease 101\n'  # Nothing released on ABORT.


def test_submit_018_teacher_refuses_the_gpu_cap_outputs_queued_jobs_inputs_and_old_code(tmp_path):
    script, run, env = submit_world(tmp_path, generation=120, wall=430)
    # 1.5 x (310 + 11,209.4) / 60 -> 288 min; 8 x 288 x 2 = 4,608 > 4,320 GPU-minutes (the 72 GPU-h ceiling cap of
    # amendment 018 addendum B).
    bad = subprocess.run(['bash', str(script), str(CODE)], env=env, capture_output=True, text=True)
    assert bad.returncode == 1 and 'GPU ceiling 4608 GPU-minutes (8 shards x 288 min x 2 GPUs) exceeds' in bad.stderr
    assert 'raise the cap or shrink T-B' in bad.stderr
    assert 'nothing submitted' in bad.stderr and not (tmp_path / 'bin/log').exists()
    script, run, env = submit_world(tmp_path / 'ok')
    pilot = run / 'llm-open-O2-P-pilot-001'
    for setup, extra, message in (
            (lambda: (run / 'llm-open-O2-T018-s07-001').mkdir(), {}, 'llm-open-O2-T018-s07-001 exists'),
            (lambda: (run / 'llm-open-O2-T018-s07-001').rmdir(), {'SQUEUE_OUT': '7411001\n'}, 'scancel 7411001'),
            (lambda: (pilot / 'receipts/p9.json').rename(pilot / 'p9.json'), {}, 'every object-count bin'),
            (lambda: ((pilot / 'p9.json').rename(pilot / 'receipts/p9.json'),
                      (run / 'eval-data-v1/MANIFEST.json').write_text('{"x": 1}')), {}, 'is not a complete O2 P-input run'),
            (lambda: (pilot / 'cost.json').unlink(), {}, 'missing')):
        setup()
        bad = subprocess.run(['bash', str(script), str(CODE)], env={**env, **extra}, capture_output=True, text=True)
        assert bad.returncode == 1 and message in bad.stderr and 'nothing submitted' in bad.stderr, bad.stderr
    assert not (tmp_path / 'ok/bin/log').exists()
    for shards in ('0', '100', 'x'):
        assert subprocess.run(['bash', str(script), str(CODE), shards], env=env, capture_output=True).returncode == 2
    old = tmp_path / 'old'
    shutil.copytree(CODE / 'fastfill/comparison', old / 'fastfill/comparison', ignore=shutil.ignore_patterns('tests', '__pycache__'))
    shutil.copytree(CODE / 'docs/research/20261013-unknown-size-comparison', old / 'docs/research/20261013-unknown-size-comparison',
                    ignore=lambda d_, names: [n for n in names if not n.startswith('amendment-018')])
    target = old / 'fastfill/comparison/llm_open.py'
    target.write_text(target.read_text().replace('--all-rooms', '--every-room'))
    bad = subprocess.run(['bash', str(script), str(old)], env=env, capture_output=True, text=True)
    assert bad.returncode == 1 and 'has no --all-rooms' in bad.stderr
