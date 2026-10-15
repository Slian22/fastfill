"""Amendment 026 addendum A, scope C (EXPLORATORY): 20 test rooms per amendment-015 room family.

python -m fastfill.comparison.diverse_026 diverse --data RUN/eval-data-v1 --out RUN/a026-diverse-v1.json

Every test room (the eval data's joint requests; uid and room_type only, never rooms.jsonl or a target) gets
room_families.family(room_type); per family (room_families.FAMILIES: all 15, other and none included) the PER_FAMILY
rooms with the lowest sha256('a026c:' + uid) (uid tie-break). Refuses when a family has fewer than PER_FAMILY test
rooms. Written once in expected_uids order with per-family counts, in agents_016's pilot-file format; an existing file
must equal its rebuild. The rooms are not cohort rooms: llm_agent run and evaluate_llm take the file as
--uids FILE --uids-scope test, output_repair as --uids FILE.
"""
import argparse
import json
from pathlib import Path

from fastfill.comparison import infer
from fastfill.comparison import llm_agent
from fastfill.comparison import train_pair_helpers as h
from fastfill.comparison.room_families import FAMILIES, family

KEY, PER_FAMILY = 'a026c:', 20
LABEL = 'amendment-026 addendum A, scope C (EXPLORATORY): family-stratified test rooms'


def build(data):
    rows, identity = infer.load_eval(data, 'joint')
    pools = {name: [] for name in FAMILIES}
    for row in rows:
        pools[family(row.get('room_type'))].append(row['uid'])
    short = {k: len(v) for k, v in pools.items() if len(v) < PER_FAMILY}
    if short:
        raise ValueError(f'families with fewer than {PER_FAMILY} test rooms: {short}')
    chosen = {u for v in pools.values() for u in sorted(v, key=lambda u: (llm_agent._digest(KEY + u), u))[:PER_FAMILY]}
    uids = [r['uid'] for r in rows if r['uid'] in chosen]
    return {'version': 1, 'label': LABEL, 'data_manifest_sha256': identity['manifest_sha256'],
            'algorithm': f"room_families.family(room_type of the joint request); per family the {PER_FAMILY} lowest "
                         f"sha256('{KEY}'+uid), uid tie-break; expected_uids order",
            'counts': {k: {'test': len(v), 'diverse': PER_FAMILY} for k, v in pools.items()},
            'uids_sha256': llm_agent._digest(json.dumps(uids)), 'uids': uids}


def diverse(data, out):
    value, out = build(Path(data)), Path(out)
    if out.exists():
        if h.strict_json(out.read_text()) != value:
            raise ValueError(f'{out} differs from its deterministic rebuild')
    else:
        infer.write_once(out, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True).add_parser('diverse')
    for flag in ('data', 'out'):
        sub.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    value = diverse(args.data, args.out)
    print(json.dumps({'rooms': len(value['uids']), 'uids_sha256': value['uids_sha256'],
                      'counts': {k: v['test'] for k, v in value['counts'].items()}}))


if __name__ == '__main__':
    main()
