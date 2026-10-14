"""Amendment 021 section 1 diagnostics on CPU: fakes stand in for vLLM, the HF policy and Slurm."""
import json
import math
import re
import subprocess

import pytest

from fastfill.comparison import a021_diag as a
from fastfill.comparison import a021_train as t
from fastfill.comparison import contract
from fastfill.comparison import onpolicy_dpo as op
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.contract import bind_room, messages
from fastfill.comparison.tests import test_infer_shard as T
from fastfill.comparison.tests.test_onpolicy_dpo import CODE, SCRIPTS, code_root, pair_args, python_stub
from fastfill.comparison.tests.test_onpolicy_dpo import world  # noqa: F401  Fixture: dpo-004-like subset + samples.


def read(path):
    return json.loads(path.read_text())


# ---- 1.1 ledger ----------------------------------------------------------------------------------

def test_copies_colocated_and_subtrees():
    p = [{'id': 'a', 'pos': [1, 1, 0], 'yaw': 0}, {'id': 'b', 'pos': [1, 1, 0], 'yaw': 0},
         {'id': 'c', 'pos': [2, 1, 0], 'yaw': 0}, {'id': 'd', 'pos': [1.0, 1, 0], 'yaw': 0},
         {'id': 'e', 'pos': [2, 1, 0], 'yaw': 90}]
    assert a.copies(p, ['x', 'y', 'x', 'x', 'x']) == {
        'records': 5, 'exact_copies': 2, 'copy_of_previous': 1, 'copy_same_category': 1}  # b of a; d of a and b.
    box = {'size': [.6, .6, .8], 'yaw': 0.0}
    layout = [{'id': 'a', 'category': 'Cabinet', 'pos': [1, 1, 0], **box}, {'id': 'b', 'category': 'cabinet', 'pos': [1, 1, 0], **box},
              {'id': 'c', 'category': 'cabinet', 'pos': [1.1, 1, 0], **box, 'parent': 'x'},
              {'id': 'd', 'category': 'sofa', 'pos': [1.05, 1, 0], **box}]
    assert a.colocated(layout) == {'colocated_pairs': 1, 'colocated_exact': 1}  # c sits on another parent.
    assert a.subtrees([{'id': 't'}, {'id': 'l', 'on': 't'}, {'id': 'c'}, {'id': 'b', 'on': 'l'}, {'id': 'w', 'on': 'win'},
                       {'id': 'm', 'on': 't'}]) == [['t', 'l', 'b', 'm'], ['c'], ['w']]


def ledger_args(w, out, pairs=None):
    return a.parse_args(['ledger', '--data', str(w.data), '--samples', str(w.samples), '--out', str(out),
                         *(['--pairs', str(pairs)] if pairs else [])])


def test_ledger_greedy_k1_copies_recombination_and_013_agreement(world):
    pairs = world.root / 'stage2-pairs'
    op.build_pairs(pair_args(world, pairs))
    out = world.root / 'diag-021/ledger'
    assert a.main(['ledger', '--data', str(world.data), '--samples', str(world.samples), '--out', str(out),
                   '--pairs', str(pairs)]) is None
    r, rooms = read(out / 'result.json'), {x['uid']: x for x in map(json.loads, (out / 'rooms.jsonl').read_text().splitlines())}
    assert set(h.strict_json((out / 'MANIFEST.json').read_text())['files']) == {'result.json', 'rooms.jsonl', 'report.md'}
    # test_onpolicy_dpo.script: a colocated cabinets at every k; b collides at k = 0 only; c floats (k = 1 capped,
    # k = 2 not JSON); d passes at k = 0, k = 1 is out of the room.
    assert r['rooms'] == 4 and r['k'] == 8 and r['ledger_agreement']['differ'] == 0
    assert r['ledger_agreement']['samples'] == 36
    assert r['greedy']['all'] == {'rooms': 4, 'complete': 1.0, 'checker': .25, 'admission': r['greedy']['all']['admission'],
                                  'pass': .25}
    assert r['greedy']['1-10']['rooms'] == 4 and r['greedy']['11-30']['rooms'] == 0
    assert (r['k1']['all']['complete'], r['k1']['all']['pass']) == (.75, .25)
    assert r['k1_minus_greedy']['complete']['estimate'] == -.25 and r['k1_minus_greedy']['pass']['estimate'] == 0
    assert [s['pass'] for s in rooms['b']['samples']] == [False] + [True] * 8
    greedy, every = r['copies']['greedy']['all'], r['copies']['all_k']
    assert (greedy['complete_samples'], greedy['records'], greedy['exact_copies']) == (4, 10, 1)
    assert (greedy['copy_of_previous_share'], greedy['samples_with_copy'], greedy['samples_with_colocated']) == (1.0, .25, .25)
    assert (every['complete_samples'], every['exact_copies'], every['colocated_pairs'], every['colocated_exact_share']) == (
        34, 10, 10, 1.0)  # c's capped and non-JSON samples are not complete; d's k = 3 is the tenth copy.
    # Units go in request order and the first keeps its greedy placement: b's sofa (first) sits on the cabinet's spot
    # at k = 0 and the cabinet is at that spot in every sample, so b stays greedy; a's second cabinet never moves.
    assert r['recombination']['all']['pass'] == .25 and r['best_of_k']['all']['pass'] == .5
    assert {u: x['recombination']['pass'] for u, x in rooms.items()} == {'a': False, 'b': False, 'c': False, 'd': True}
    assert rooms['b']['recombination']['picks'] == {'0': 2} and rooms['a']['recombination']['units_clean'] == 2
    assert r['recombination_units'] == {'rooms': 4, 'units': 10, 'units_clean': 8, 'structure_not_greedy': 0}
    assert r['recombination_minus_greedy']['pass']['estimate'] == 0
    report = (out / 'report.md').read_text()
    assert report.startswith('# 修订 021 §1.1') and '36/36 一致' in report and '支撑子树重组上限' in report
    with pytest.raises(FileExistsError):
        a.ledger(ledger_args(world, out))
    other = world.root / 'other'
    a.ledger(ledger_args(world, other))  # No pairs ledger: reported as not given.
    assert read(other / 'result.json')['ledger_agreement'] is None
    (world.root / 'p2').mkdir()
    for name in ('ledger.jsonl', 'summary.json', 'placement.jsonl'):
        (world.root / 'p2' / name).write_bytes((pairs / name).read_bytes())
    lines = (world.root / 'p2/ledger.jsonl').read_text().splitlines()
    first = json.loads(lines[0])
    first['samples'][0]['class'] = 'pass'
    (world.root / 'p2/ledger.jsonl').write_text('\n'.join([json.dumps(first)] + lines[1:]) + '\n')
    h.write_json(world.root / 'p2/MANIFEST.json', {'files': h.artifact_manifest(world.root / 'p2')})
    a.ledger(ledger_args(world, world.root / 'third', world.root / 'p2'))
    agreement = read(world.root / 'third/result.json')['ledger_agreement']
    assert agreement['differ'] == 1 and agreement['examples'] == [{'uid': 'a', 'k': 0, 'ledger': 'pass', 'here': 'reject'}]


def test_recombine_takes_each_subtree_from_the_first_sample_clear_of_the_chosen_ones():
    room, request, _ = dev_room('r', ['cabinet', 'sofa', 'table', 'lamp'], ROW + [[5.5, 1, 0]])
    prior = {o['id']: o['size'] for o in room['objects']}
    objects = [o['id'] for o in json.loads(request['messages'][1]['content'])['objects']]
    table, lamp = (objects[[o['category'] for o in room['objects']].index(c)] for c in ('table', 'lamp'))
    first, second = (i for i in objects if i not in (table, lamp))

    def sample(k, moves, finish='stop'):
        pos = {i: [1 + 1.5 * n, 1, 0] for n, i in enumerate(objects)}
        pos[lamp] = [pos[table][0], 1, .8]
        pos.update(moves)
        raw = contract._json({'placements': [{'id': i, 'pos': pos[i], 'yaw': 0, **({'on': table} if i == lamp else {})}
                                             for i in objects]})
        return {'k': k, 'raw': raw, 'error': None, 'finish_reason': finish}
    # k = 0 and 2: the second unit sits on the first; k = 1: capped (never a candidate); k = 3: clean.
    clash = {second: [1 + 1.5 * objects.index(first), 1, 0]}
    samples = [sample(0, clash), sample(1, {}, 'length'), sample(2, clash), sample(3, {})]
    scored = [op._sample(room, request, prior, s) for s in samples]
    assert not op._passes(scored[0][0], scored[0][2]) and not scored[1][0]['complete']
    value = a.recombine(room, request, prior, samples, scored)
    assert value == {'complete': True, 'checker': True, 'admission': True, 'pass': True, 'units': 3, 'units_clean': 3,
                     'structure_k': 0, 'picks': {0: 2, 3: 1}}  # The lamp moves with its table (from k = 0).
    assert a.recombine(room, request, prior, [samples[1]], [scored[1]])['units'] is None


# ---- 1.2 pool ------------------------------------------------------------------------------------

def train_pool(root):
    """A published a021_train pool: two common rooms, three P-only rooms (one fails the checker)."""
    root.mkdir(parents=True)
    rows = [('c1', 'common', 'S0', '1-10', 'bedroom', True), ('p1', 'p_only', 'S0', '1-10', 'bedroom', True),
            ('p2', 'p_only', 'S1', '11-30', 'bedroom', False), ('c2', 'common', 'S1', '11-30', 'other', True),
            ('p3', 'p_only', 'S1', '11-30', 'other', True)]
    ledger = [dict(zip(('uid', 'origin', 'source', 'bin', 'family', 'pool'), r)) for r in rows]
    (root / 'ledger.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in ledger))
    (root / 'sft-rows.jsonl').write_text('')
    summary = {'status': t.POOL_STATUS, 'p_only_expected': 3, 'code_sha256': {'a021_train.py': 'x'},
               'admission_counts': {'eligible_candidates': 9, 'joint_qualified': 6, 'placement_qualified': 8,
                                    'common_decoder_train': 5},
               'rebuild': {'p_only_ledger_equal': 3, 'accepted_sample_byte_equal': 2}, 'lineage': {'sha256': 'l'},
               'inputs': {'main_manifest_sha256': 'm', 'corpus_manifest_sha256': 'c'}}
    (root / 'summary.json').write_text(json.dumps(summary))
    h.write_json(root / 'MANIFEST.json', {'files': h.artifact_manifest(root)})
    return root


def test_pool_breaks_down_the_train_pools_p_only_rooms(tmp_path):
    source, out = train_pool(tmp_path / 'a021/pool'), tmp_path / 'diag/pool'
    args = a.parse_args(['pool', '--pool', str(source), '--out', str(out)])
    assert a.pool(args) == {'status': 'A021_POOL_COMPLETE', 'p_only': 3, 'checker_pass': 2}
    r = read(out / 'result.json')
    assert read(out / 'uids.json') == ['p1', 'p3'] and r['p_only_expected'] == 3 and r['descriptive_only'] is True
    assert r['by_source'] == {'S0': {'rooms': 1, 'checker_pass': 1}, 'S1': {'rooms': 2, 'checker_pass': 1}}
    assert r['by_bin'] == {'1-10': {'rooms': 1, 'checker_pass': 1}, '11-30': {'rooms': 2, 'checker_pass': 1}}
    assert r['by_family'] == {'bedroom': {'rooms': 2, 'checker_pass': 1}, 'other': {'rooms': 1, 'checker_pass': 1}}
    assert r['inputs']['pool_manifest_sha256'] == h.sha256(source / 'MANIFEST.json') and r['inputs']['main_manifest_sha256'] == 'm'
    report = (out / 'report.md').read_text()
    assert report.startswith('# 修订 021 §1.2') and '2/3' in report and 'placement_qualified 8' in report
    with pytest.raises(FileExistsError):
        a.pool(args)
    (source / 'ledger.jsonl').write_text('')  # A pool that is not the published one.
    with pytest.raises(ValueError):
        a.pool(a.parse_args(['pool', '--pool', str(source), '--out', str(tmp_path / 'x')]))


def test_pool_reads_what_a021_train_pool_writes(tmp_path, monkeypatch):
    """test_a021_train's world fixture, inlined (its name clashes with the 013 world here): the producer's real output."""
    from types import SimpleNamespace
    from fastfill.comparison import dpo_stages as d
    from fastfill.comparison import main_data as m
    from fastfill.comparison.tests import test_a021_train as TT
    from fastfill.comparison.tests import test_main_data as M
    corpus = M.fixture(tmp_path / 'corpus', TT.collide)
    main = tmp_path / 'main'
    manifest = m.build(corpus, main, max_candidates=8, max_passes=1)
    lineage = tmp_path / 'lineage.json'
    lineage.write_text(json.dumps({'status': 'ERROR', 'summary': {'admission_counts': manifest['dataset_counts']}}))
    monkeypatch.setattr(t, 'LINEAGE_SHA256', h.sha256(lineage))
    monkeypatch.setattr(t, 'P_ONLY', 1)
    monkeypatch.setattr(t, 'base_receipt', TT.receipt)
    monkeypatch.setattr(d, '_model_tokenizer', lambda path: TT.Tok())
    monkeypatch.setattr(op, '_held_out', lambda roots: (set(), set(), []))
    t.pool(SimpleNamespace(data=main, corpus=corpus, lineage=lineage, base27=tmp_path / 'qwen27', base8=tmp_path / 'qwen8',
                           dev=[], test=[], out=tmp_path / 'a021/pool', workers=1, sample=3))
    assert a.pool(a.parse_args(['pool', '--pool', str(tmp_path / 'a021/pool'), '--out', str(tmp_path / 'diag')])) == {
        'status': 'A021_POOL_COMPLETE', 'p_only': 1, 'checker_pass': 1}
    r = read(tmp_path / 'diag/result.json')
    assert r['by_source'] == {'Synthetic': {'rooms': 1, 'checker_pass': 1}} and read(tmp_path / 'diag/uids.json') == ['P0']
    assert r['rebuild']['p_only_ledger_equal'] == 1 and r['p_only_expected'] == 1


# ---- 1.3 probe -----------------------------------------------------------------------------------

SIX = [.6, .6, .8]


class Tok:
    """Prompt = 'P:' + user JSON + newline; one token per character except ',{' (one token, as BPE merges may)."""
    MERGED = 0x10000

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return 'P:' + messages[1]['content'] + '\n'

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=None):
        assert add_special_tokens is False
        ids, offsets, n = [], [], 0
        while n < len(text):
            width = 2 if text.startswith(',{', n) else 1
            ids.append(self.MERGED if width == 2 else ord(text[n]))
            offsets.append((n, n + width))
            n += width
        return {'input_ids': ids, **({'offset_mapping': offsets} if return_offsets_mapping else {})}

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(',{' if i == self.MERGED else chr(i) for i in ids)


def dev_room(uid, categories, positions, on=None, sizes=None):
    """(room, request, record): P's output places object n (bound order) at positions[n], on object on[n] (an index),
    object n of size sizes[n] (default SIX)."""
    room = bind_room({'uid': uid, 'source': 'S0', 'group': 'g-' + uid, 'room_type': 'bedroom', 'boundary_type': 'polygon',
                      'boundary': [[0, 0], [7, 0], [7, 7], [0, 7]], 'height': 3, 'fixed': [],
                      'objects': [{'id': f'o{j}', 'category': 'box', 'size': list(SIX), 'pos': [1 + 1.5 * j, 1, 0], 'yaw': 0}
                                  for j in range(len(categories))]})[0]
    ids = [o['id'] for o in room['objects']]  # Categories and supports are set in bound (request) order.
    parents = [None if not on or on[n] is None else ids[on[n]] for n in range(len(ids))]
    for o, category, parent, size in zip(room['objects'], categories, parents, sizes or [SIX] * len(ids)):
        o.update(category=category, size=list(size), **({'parent': parent} if parent else {}))
    sizes = {o['id']: o['size'] for o in room['objects']}
    request = {'uid': uid, 'source': 'S0', 'group': 'g-' + uid, 'messages': messages(room, 'placement', sizes)[:2]}
    raw = contract._json({'placements': [{'id': i, 'pos': list(p), 'yaw': 0, **({'on': parent} if parent else {})}
                                         for i, p, parent in zip(ids, positions, parents)]})
    tok = Tok()
    prompt = tok.apply_chat_template(request['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    record = {'uid': uid, 'raw': raw, 'reached_token_cap': False, 'stopped_on_eos': True,
              'input_tokens': len(tok(prompt, add_special_tokens=False)['input_ids'])}
    return room, request, record


ROW = [[1, 1, 0], [2.5, 1, 0], [4, 1, 0]]


def test_probe_item_picks_the_colliding_sibling_and_builds_both_prefixes():
    room, request, record = dev_room('p', ['cabinet', 'sofa', 'cabinet'], ROW)
    reason, item = a.probe_item(room, request, record)
    placements = json.loads(record['raw'])['placements']
    assert reason is None and (item['k'], item['j'], item['category']) == (2, 0, 'cabinet')  # Same category, latest.
    control, intervention = item['conditions']['control'], item['conditions']['intervention']
    assert control['full'] == record['raw'] and record['raw'][control['start']] == '{'
    assert control['start'] == len('{"placements":[' + contract._json(placements[0]) + ',' + contract._json(placements[1]) + ',')
    assert json.loads(intervention['full'])['placements'][0]['pos'] == [4, 1, 0] and intervention['start'] == control['start']
    assert (control['colliding_pos'], control['clean_pos']) == ([1, 1, 0], [4, 1, 0])
    assert (intervention['colliding_pos'], intervention['clean_pos']) == ([4, 1, 0], [1, 1, 0])
    cases = {
        'token_cap': dev_room('c', ['cabinet', 'cabinet'], ROW[:2]),
        'invalid_output': dev_room('i', ['cabinet', 'cabinet'], ROW[:2]),
        'non_canonical': dev_room('n', ['cabinet', 'cabinet'], ROW[:2]),
        'no_colliding_sibling': dev_room('s', ['cabinet', 'sofa'], ROW[:2])}
    cases['token_cap'][2].update(reached_token_cap=True, stopped_on_eos=False)
    cases['invalid_output'][2]['raw'] = '{"placements":[]}'
    cases['non_canonical'][2]['raw'] = cases['non_canonical'][2]['raw'].replace(',', ', ')
    for reason, (room, request, record) in cases.items():
        assert a.probe_item(room, request, record) == (reason, None)
    for categories, positions, on in ((['cabinet', 'cabinet'], [ROW[0], ROW[0]], None),  # P already copied.
                                      (['rug', 'rug'], ROW[:2], None),                    # Rugs never collide.
                                      (['table', 'lamp', 'lamp'], [[3, 3, 0], [3, 3, .8], [2, 2, 0]], [None, 0, None])):
        assert a.probe_item(*dev_room('x', categories, positions, on))[0] == 'no_colliding_sibling'
    # k already collides with the sofa in P's output; the big cabinet j moved onto pos_k only contains k: no j-k pair.
    contained = dev_room('y', ['cabinet', 'sofa', 'cabinet'], [[1.5, 1.5, 0], [5.4, 5, 0], [5, 5, 0]],
                         sizes=[[2, 2, .8], SIX, [.5, .5, .8]])
    objects = json.loads(contained[1]['messages'][1]['content'])['objects']
    placements = json.loads(contained[2]['raw'])['placements']
    assert a.hits(objects, placements[:2], placements[2])  # The old any-record rule would have picked this room.
    assert a.probe_item(*contained) == ('no_colliding_sibling', None)
    rooms = {u: dev_room(u, ['cabinet', 'cabinet'], ROW[:2]) for u in 'abcdef'}
    rooms['b'][2]['raw'] = 'x'
    picked, skipped = a.probe_items(*({u: v[n] for u, v in rooms.items()} for n in range(3)), 3)
    order = sorted(rooms, key=lambda u: a.hashlib.sha256((a.PROBE_SALT + u).encode()).hexdigest())
    assert [x['uid'] for x in picked] == [u for u in order if u != 'b'][:3]
    assert skipped == ({'invalid_output': 1} if 'b' in order[:order.index(picked[-1]['uid'])] else {})


def policy(tok, item, mode, calls):
    """Fake P: copier = record k at the last position in the prefix; reactive = P's pos_k unless it is occupied there."""
    def generate(ids, n):
        calls.append(('generate', list(ids), n))
        seen = tok.decode(ids)
        last = json.loads(re.findall(r'"pos":(\[[^\]]*\])', seen)[-1])
        pos = last if mode == 'copier' else [5.5, 1, 0] if last == item['record']['pos'] else item['record']['pos']
        return tok(',' + contract._json({**item['record'], 'pos': pos}) + ',{"id":"o9"', add_special_tokens=False)['input_ids']

    def logprob(ids, n):
        assert list(ids[:-n]) == calls[-1][1]  # Teacher forcing on the prefix that was generated from.
        return -0.001 * sum(ids[-n:])
    return generate, logprob


def score(tok, text):
    return -0.001 * sum(tok(text, add_special_tokens=False)['input_ids'])


def test_probe_room_cuts_at_token_boundaries_and_measures_the_move():
    tok = Tok()
    _, item = a.probe_item(*dev_room('p', ['cabinet', 'cabinet', 'sofa'], ROW))
    assert (item['k'], item['j']) == (1, 0)
    calls = []
    result = a.probe_room(item, tok, *policy(tok, item, 'copier', calls))
    record = item['record']
    rec = lambda pos: ',' + contract._json({**record, 'pos': pos})  # ',{' straddles record k: the cut is before ','.
    prompt = tok(tok.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False),
                 add_special_tokens=False)['input_ids']
    assert calls[0][1][:len(prompt)] == prompt and tok.decode(calls[0][1][len(prompt):]) == item['conditions']['control']['full'][
        :item['conditions']['control']['start'] - 1]
    assert calls[0][2] == len(tok(rec([2.5, 1, 0]), add_special_tokens=False)['input_ids']) + a.SLACK  # The longer one.
    control, intervention = result['control'], result['intervention']
    assert control == {'parsed': True, 'pos': [1, 1, 0], 'at_colliding': True, 'at_clean': False, 'collides': True,
                       'clean_candidate_collides': False, 'logp_colliding': score(tok, rec([1, 1, 0])),
                       'logp_clean': score(tok, rec([2.5, 1, 0])), 'delta': control['delta'], 'move_xy_m': 1.5}
    assert math.isclose(control['delta'], score(tok, rec([1, 1, 0])) - score(tok, rec([2.5, 1, 0])))
    assert (intervention['pos'], intervention['at_colliding'], intervention['collides'], intervention['move_xy_m']) == (
        [2.5, 1, 0], True, True, 0.0)
    assert math.isclose(intervention['delta'], score(tok, rec([2.5, 1, 0])) - score(tok, rec([1, 1, 0])))
    reactive = a.probe_room(item, tok, *policy(tok, item, 'reactive', []))
    assert (reactive['control']['at_clean'], reactive['intervention']['at_colliding'], reactive['intervention']['pos'],
            reactive['intervention']['collides'], reactive['intervention']['move_xy_m']) == (True, False, [5.5, 1, 0], False, 3.0)
    summary = a.probe_summary([result, {**reactive, 'uid': 'q', 'group': 'g-q'}])
    assert (summary['rooms'], summary['control_reproduces_greedy'], summary['intervention_moved']) == (2, .5, .5)
    assert summary['intervention']['at_colliding'] == .5 and summary['intervention_move_xy_median_m'] == 3.0
    assert summary['intervention_minus_control']['at_colliding']['estimate'] == 0.0  # 1 - 1 and 0 - 0.
    garbage = a.probe_room(item, tok, lambda ids, n: tok('nonsense', add_special_tokens=False)['input_ids'],
                           lambda ids, n: 0.0)
    assert garbage['control']['parsed'] is False and garbage['control']['collides'] is None
    with pytest.raises(ValueError, match='template drift'):
        a.probe_room({**item, 'input_tokens': item['input_tokens'] + 1}, tok, *policy(tok, item, 'copier', []))


def test_probe_writes_results_within_the_time_limit(tmp_path, monkeypatch):
    tok = Tok()
    rooms = {u: dev_room(u, ['cabinet', 'cabinet', 'sofa'], ROW) for u in 'abc'}
    loaded = []
    monkeypatch.setattr(a, '_dev_records', lambda data, ref: (
        {u: v[0] for u, v in rooms.items()}, {u: v[1] for u, v in rooms.items()}, {u: v[2] for u, v in rooms.items()},
        {'adapter': 'A'}, {'records': str(ref)}))

    def hf(policy_value):
        loaded.append(policy_value)
        _, item = a.probe_item(*rooms['a'])
        return (tok, *policy(tok, item, 'copier', []))
    monkeypatch.setattr(a, '_hf', hf)
    out = tmp_path / 'probe'
    assert a.probe(a.parse_args(['probe', '--data', 'D', '--ref-root', 'R', '--out', str(out), '--rooms', '2'])) == {
        'status': 'A021_PROBE_COMPLETE', 'rooms': 2, 'not_run_time_limit': 0}
    r = read(out / 'result.json')
    assert loaded == [{'adapter': 'A'}] and r['selected'] == 2 and r['skipped'] == {} and r['descriptive_only'] is True
    assert len((out / 'rooms.jsonl').read_text().splitlines()) == 2 and r['inputs'] == {'records': 'R'}
    assert (out / 'report.md').read_text().startswith('# 修订 021 §1.3')
    late = a.parse_args(['probe', '--data', 'D', '--ref-root', 'R', '--out', str(tmp_path / 'late'), '--max-minutes', '-1'])
    assert a.probe(late) == {'status': 'A021_PROBE_COMPLETE', 'rooms': 0, 'not_run_time_limit': 3}
    assert '超时未运行 3' in (tmp_path / 'late/report.md').read_text()


def test_dev_records_bind_the_loaded_policy(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(a.ds, '_dev', lambda data: ({'expected_uids': ['u'], 'training_manifest_sha256': 't'}, {}, {'placement': {}}))
    monkeypatch.setattr(a.e, '_inference', lambda path, data, expected: (
        [{'uid': 'u'}], {'training_data': {'manifest_sha256': 't'}, 'run_sha256': 'r', 'policy': {'adapter': 'A'}}))
    monkeypatch.setattr(a.ds, '_bind', lambda receipt, root, arm, candidate: calls.append((root, arm, candidate)))
    monkeypatch.setattr(a.d, 'exact_policy', lambda model, arm: (calls.append(model), ({}, {'adapter': 'A'}))[1])
    (tmp_path / 'MANIFEST.json').write_text('{}')
    ref = tmp_path / 'dev-eval-010'
    rooms, requests, records, policy_value, inputs = a._dev_records(tmp_path, ref)
    assert records == {'u': {'uid': 'u'}} and policy_value == {'adapter': 'A'} and inputs['records_run_sha256'] == 'r'
    assert calls == [(tmp_path, 'placement', 'rpo-dpo1'), a.ds.model_path(tmp_path, 'placement', 'rpo-dpo1')]
    monkeypatch.setattr(a.d, 'exact_policy', lambda model, arm: ({}, {'adapter': 'B'}))
    with pytest.raises(ValueError, match='not written by the policy'):
        a._dev_records(tmp_path, ref)


# ---- submit_021_diag.sh --------------------------------------------------------------------------

INPUTS = ('dpo-004/train-subset/MANIFEST.json', 'dpo-004/samples/MANIFEST.json', 'a021/pool/MANIFEST.json',
          'dev-select-v1/MANIFEST.json', 'dev-eval-010/rpo-dpo1/infer-placement/MANIFEST.json',
          'dpo-003/placement/dpo1-merged.json')
DOC = 'docs/research/20261013-unknown-size-comparison/amendment-021-diagnostics-and-one-more-arm-20261014.md'
LINE = 'PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill\n'


def project(tmp_path):
    root = tmp_path / 'project'
    run = root / 'runs/unknownsize-20261013'
    for path in INPUTS:
        (run / path).parent.mkdir(parents=True, exist_ok=True)
        (run / path).write_text('{}')
    python_stub(root / 'env/bin/python')
    original = (SCRIPTS / 'submit_021_diag.sh').read_text()
    assert original.count(LINE) == 1
    script = tmp_path / 'submit_021_diag.sh'
    script.write_text(original.replace(LINE, f'PROJECT_ROOT={root}\n'))
    (tmp_path / 'squeue').write_text('#!/bin/sh\ntest "$1 $2 $3 $4 $6 $7" = "-h -u me -n -o %i" || exit 3\n'
                                     'echo "$5" >> "$SBATCH_LOG.squeue"\nprintf "%s" "$SQUEUE_OUT"\n')
    (tmp_path / 'squeue').chmod(0o755)
    return script, run


def test_submit_021_diag_held_steps_released_together_and_refusals(tmp_path):
    subprocess.run(['bash', '-n', str(SCRIPTS / 'submit_021_diag.sh')], check=True)
    script, run = project(tmp_path)
    code = code_root(tmp_path / 'main-abc', doc=False)
    (code / DOC).parent.mkdir(parents=True)
    (code / DOC).write_text('# Amendment 021\n')
    env = {**T.stub_sbatch(tmp_path), 'USER': 'me'}
    name = '--parsable --hold --job-name=unknownsize-021-diag-'

    def submit(*steps, **extra):
        log = tmp_path / 'log'
        before = len(log.read_text().splitlines()) if log.exists() else 0
        done = subprocess.run(['bash', str(script), str(code), *steps], env={**env, **extra}, capture_output=True, text=True)
        sent = log.read_text().splitlines()[before:] if log.exists() else []
        return done, [x.replace(f'{code}/fastfill/comparison/', '').replace(f' {code} ', ' CODE ') for x in sent]
    done, log = submit('ledger', 'probe')
    assert done.returncode == 0, done.stderr
    assert log == [name + 'ledger submit_021_diag.sh CODE job ledger',
                   name + 'probe --gpus-per-node=1 --cpus-per-task=8 --mem=96G --time=00:30:00 submit_021_diag.sh CODE job probe']
    assert (tmp_path / 'log.squeue').read_text() == 'unknownsize-021-diag-ledger,unknownsize-021-diag-probe\n'
    assert (tmp_path / 'log.scontrol').read_text() == 'scontrol release 101,102\n'
    assert done.stdout.splitlines()[-1] == 'released 101 102'
    (run / 'a021/pool/MANIFEST.json').unlink()  # The pool step waits for a021_train's pool.
    done, log = submit('pool')
    assert (done.returncode, log) == (1, []) and 'a021/pool/MANIFEST.json' in done.stderr
    (run / 'a021/pool/MANIFEST.json').write_text('{}')
    done, log = submit('pool')
    assert done.returncode == 0 and log == [name + 'pool --cpus-per-task=4 --mem=32G --time=00:30:00 submit_021_diag.sh CODE job pool']
    failed = tmp_path / 'failed'
    failed.mkdir()
    (failed / 'squeue').write_bytes((tmp_path / 'squeue').read_bytes())
    (failed / 'squeue').chmod(0o755)
    bad = subprocess.run(['bash', str(script), str(code), 'ledger', 'pool', 'probe'],
                         env={**T.stub_sbatch(failed, fail_on=3), 'USER': 'me'}, capture_output=True, text=True)
    assert bad.returncode == 1 and "sbatch failed for 'probe'" in bad.stderr and 'scancel 101 102' in bad.stderr
    assert not (failed / 'log.scontrol').exists()
    (run / 'diag-021/pool').mkdir(parents=True)
    done, log = submit('ledger', 'pool')
    assert (done.returncode, log) == (1, []) and 'diag-021/pool exists' in done.stderr
    done, log = submit('ledger', SQUEUE_OUT='7400001\n')
    assert (done.returncode, log) == (1, []) and 'scancel 7400001' in done.stderr
    for steps in (('bogus',), ('ledger', 'ledger'), ()):
        done, log = submit(*steps)
        assert (done.returncode, log) == (2, []) and 'usage' in done.stderr
    (run / 'dpo-004/samples/MANIFEST.json').unlink()
    done, log = submit('ledger')
    assert (done.returncode, log) == (1, []) and 'dpo-004/samples/MANIFEST.json' in done.stderr
    done = subprocess.run(['bash', str(script), str(code_root(tmp_path / 'main-nodoc', doc=False)), 'probe'], env=env,
                          capture_output=True, text=True)
    assert done.returncode == 1 and 'no amendment-021 protocol doc' in done.stderr


@pytest.mark.parametrize('argv, ok', [(['job', 'ledger'], True), (['job', 'pool'], True), (['job', 'probe'], True),
                                      (['job', 'bogus'], False), (['run', 'ledger'], False), (['job'], False)])
def test_job_modes_validate_arguments_before_the_allocation(tmp_path, argv, ok):
    done = subprocess.run(['bash', str(SCRIPTS / 'submit_021_diag.sh'), str(CODE), *argv],
                          env={'PATH': '/usr/bin:/bin'}, capture_output=True, text=True)
    assert (done.returncode, 'Slurm allocation required' in done.stderr) == ((1, True) if ok else (2, False))


def test_job_runs_the_diagnostic_with_this_code(tmp_path):
    script, run = project(tmp_path)
    (run / 'dpo-004/stage2-pairs').mkdir()
    (run / 'dpo-004/stage2-pairs/MANIFEST.json').write_text('{}')
    stubs = tmp_path / 'bin'
    stubs.mkdir()
    (stubs / 'timeout').write_text('#!/bin/sh\nshift 3\nexec "$@"\n')
    (stubs / 'timeout').chmod(0o755)
    prefix = ['-B', '-u', '-m', 'fastfill.comparison.a021_diag']
    for mode, argv in (('ledger', ['ledger', '--data', str(run / 'dpo-004/train-subset'), '--samples', str(run / 'dpo-004/samples'),
                                   '--pairs', str(run / 'dpo-004/stage2-pairs'), '--out', str(run / 'diag-021/ledger'),
                                   '--workers', '72']),
                       ('pool', ['pool', '--pool', str(run / 'a021/pool'), '--out', str(run / 'diag-021/pool')])):
        log = tmp_path / f'{mode}.log'
        done = subprocess.run(['bash', str(script), str(CODE), 'job', mode], capture_output=True, text=True,
                              env={'PATH': f'{stubs}:/usr/bin:/bin', 'SLURM_JOB_ID': '1', 'PYLOG': str(log)})
        assert done.returncode == 0, done.stderr
        assert log.read_text().split() == prefix + argv
        a.parse_args(argv)  # The job's arguments parse.
