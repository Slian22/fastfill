"""Freeze reference-only visual eligibility; attach paired raw predictions later.

CPU planning only. No Blender, API, training, or inference is launched. Scope is
reference-asset-assisted visualization, NOT paper Isaac retrieval/native scale.
Original prepared rooms are bound with contract.DEFAULT_SEED before joining the
archived crosswalk by (uid, original_instance_id); old prompt IDs and all archived
GT geometry are discarded. Movables retain their own prepared size/pose. Fixed
geometry is the actual rounded request, with original structure/fixed_kind flags.
Only render_assets' explicit structural categories (door/window/wall/column/beam)
with structure=True or fixed_kind='structure' are procedural. All other fixed
furniture requires a real mapped mesh. Unknown height uses the renderer's 3 m
policy. Malformed candidates remain in both the case and eligibility ledgers.

freeze_reference(export, receipt, assetmap, out) publishes a new directory.
attach_predictions(reference_cases, joint, placement) returns {cases, ledger,
counts, provenance}; inputs are JSONL paths. Prediction files must be restricted
to the reference cohort: foreign/duplicate UIDs are fatal, missing UIDs fail in
the ledger. geometry.score's complete flag is its schema-completion contract;
an explicit schema_complete=False always fails. Only raw_layout is consumed.
All three layouts share reference context/cameras; either arm's failure prevents
the whole paired case. Separate reference rendering can use the frozen cases.

CLI: python -m fastfill.comparison.visual_cases freeze --export ... --receipt ...
     --assetmap ... --out ...
     python -m fastfill.comparison.visual_cases attach --reference-cases ...
     --joint ... --placement ... --out ...
Outputs are atomic and never replace an existing directory. The CPU freeze may
be expensive on real assets; inspection results are cached across rooms locally,
without changing renderer module globals. No full production scan runs on import.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import FunctionType

from fastfill import scene
from fastfill.comparison import contract, eval_data, render_assets as renderer
from fastfill.comparison.corpus import _publish
from fastfill.comparison.data import sha256

SCOPE = 'reference_asset_assisted_visual_comparison'
RENDER_SETTINGS = {'resolution': 512, 'samples': 32, 'denoise': True,
                   'views': ['perspective', 'top'], 'unknown_height_m': 3.0}
STRATA = tuple(source + '/' + kind for source, kind in eval_data.PAPER_STRATA)
ERRORS = (ValueError, TypeError, KeyError, OSError, OverflowError, IndexError)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _write_json(path, value):
    path.write_text(_json(value) + '\n', encoding='utf-8')


def _write_rows(path, records):
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(_json(record) + '\n')


def _read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _index(records):
    result = {}
    for row in records:
        uid = row.get('uid') if isinstance(row, dict) else None
        if not isinstance(uid, str) or not uid or uid in result:
            raise ValueError(f'missing/duplicate UID: {uid}')
        result[uid] = row
    return result


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _stratum(room):
    kind = room.get('room_type')
    return str(room.get('source')) + '/' + ((scene.norm_room_type(kind) or '') if isinstance(kind, str) else '')


def _inputs(export, receipt, assetmap):
    paths = dict(export=Path(export), receipt=Path(receipt), assetmap=Path(assetmap))
    hashes = {key + '_sha256': sha256(path) for key, path in paths.items()}
    info = json.loads(paths['receipt'].read_text())
    if info.get('gzip_sha256') != hashes['export_sha256']:
        raise ValueError('export receipt gzip hash mismatch')
    for key in ('corpus_manifest_sha256', 'consumed_prepared_sha256'):
        value = info.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'invalid receipt {key}')
    rooms = _read_rows(paths['export'])
    _index(rooms)
    if isinstance(info.get('rows'), bool) or info.get('rows') != len(rooms):
        raise ValueError('export receipt row count mismatch')
    asset_rows = json.loads(paths['assetmap'].read_text())
    if not isinstance(asset_rows, list) or any(not isinstance(r, dict) for r in asset_rows):
        raise ValueError('assetmap must be a list of objects')
    return paths, {**hashes, **{k: info[k] for k in ('corpus_manifest_sha256', 'consumed_prepared_sha256')}}, rooms, asset_rows


def _planner():
    """Inject one invocation-local inspection cache into unchanged renderer code.

    plan_case has no cache parameter. Private function copies preserve its actual
    validation/category logic and avoid process-global monkeypatches or a copied
    planner implementation. Both successful and rejected inspections are cached.
    """
    cache, resources = {}, {}

    def inspect(path, asset_id):
        key = (path, asset_id)
        if key not in cache:
            try:
                resource = renderer.inspect_asset(path, asset_id)
                cache[key] = (resource, None)
                resources[_digest(list(key))] = {
                    'mesh_path': resource['mesh_path'], 'asset_id': asset_id,
                    'mesh_sha256': resource['sha256'],
                    'materials': [{k: m[k] for k in ('path', 'sha256', 'texture_sha256')}
                                  for m in resource['materials']], 'overrides': resource['overrides']}
            except ERRORS as exc:
                cache[key] = (None, (type(exc).__name__, str(exc)))
        resource, error = cache[key]
        if error:
            raise renderer.RenderReject(': '.join(error))
        return resource

    mapped = FunctionType(renderer.mapped_object.__code__,
                          {**renderer.mapped_object.__globals__, 'inspect_asset': inspect})
    original = renderer.plan_case
    planner = FunctionType(original.__code__, {**original.__globals__, 'mapped_object': mapped},
                           original.__name__, original.__defaults__, original.__closure__)
    return planner, resources


def _asset_mapping(room, sidecar, rows):
    original = {o['id']: role for field, role in (('objects', 'movable'), ('fixed', 'fixed'))
                for o in room.get(field, [])}
    result, seen = [], set()
    for row in rows:
        oid = row.get('original_instance_id')
        if not isinstance(oid, str) or oid not in original:
            raise ValueError('asset original_instance_id is absent from prepared room')
        if oid in seen or row.get('source') != room['source'] or row.get('role') != original[oid]:
            raise ValueError('ambiguous asset identity/source/role')
        seen.add(oid)
        result.append({'uid': room['uid'], 'original_instance_id': oid,
                       'prompt_id': sidecar['original_to_bound'][oid], 'source': room['source'],
                       'role': original[oid], 'asset_id': row.get('asset_id'), 'mesh_path': row.get('mesh_path'),
                       'layout_object': {'scale': copy.deepcopy(row.get('layout_object', {}).get('scale', [1, 1, 1]))}})
    return sorted(result, key=lambda row: row['prompt_id'])


def _build_case(room, assets, provenance):
    if room.get('split') != 'test':
        raise ValueError('reference export contains a non-test candidate')
    bound, sidecar = contract.bind_room(room, seed=contract.DEFAULT_SEED)
    request = json.loads(contract.messages(bound)[1]['content'])
    original_fixed = {o['id']: o for o in bound.get('fixed', [])}
    fixed = [{**copy.deepcopy(original_fixed[o['id']]), **o, 'yaw': math.radians(o['yaw'])}
             for o in request.get('fixed', [])]
    return {'uid': room['uid'], 'source': room['source'], 'stratum': _stratum(room),
            'boundary': request['boundary'], 'boundary_type': request['boundary_type'],
            'height': request.get('height'), 'fixed': fixed,
            'expected_ids': [o['id'] for o in bound['objects']],
            'layouts': {'reference': copy.deepcopy(bound['objects'])},
            'asset_mapping': _asset_mapping(room, sidecar, assets), 'layout_variant': 'raw',
            'scope': SCOPE, 'binding': sidecar, 'render_settings': copy.deepcopy(RENDER_SETTINGS),
            'provenance': {**provenance, 'prepared_room_sha256': _digest(room),
                           'binding_sha256': _digest(sidecar)}}


def _qualify(room, assets, provenance, planner):
    case = {'uid': room['uid'], 'source': room.get('source'), 'stratum': _stratum(room),
            'layout_variant': 'raw', 'scope': SCOPE, 'provenance': provenance}
    reason = None
    try:
        case = _build_case(room, assets, provenance)
        plan = planner(case)
        case = {**case, 'cameras': plan['cameras'], 'height_provenance': plan['height_provenance']}
    except ERRORS as exc:
        reason = type(exc).__name__ + ': ' + str(exc)
    eligible = reason is None
    case = {**case, 'reference_eligible': eligible, **({'input_error': reason} if reason else {})}
    ledger = {'uid': room['uid'], 'source': room.get('source'), 'stratum': _stratum(room),
              'eligible': eligible, 'reason': reason, 'case_sha256': _digest(case)}
    return case, ledger


def _selection(rooms, ledger):
    raw = eval_data.select_gpt(rooms)
    strata = {key: {**value, 'selected_uids': value['selected_uids'][:29],
                    'selected_count': len(value['selected_uids'][:29]),
                    'inclusion_probability': min(29, value['population']) / value['population']
                    if value['population'] else 0}
              for key, value in raw['strata'].items()}
    uids = [uid for value in strata.values() for uid in value['selected_uids']]
    by_uid = _index(ledger)
    selected = [{**by_uid[uid], 'status': 'selected_pending_predictions' if by_uid[uid]['eligible']
                 else 'selected_failed'} for uid in uids]
    return {'seed': contract.DEFAULT_SEED, 'max_per_stratum': 29, 'raw_candidates': raw,
            'strata': strata, 'selected_uids': uids, 'selected': selected,
            'selection_stage': 'before_reference_eligibility_and_predictions', 'replacement': False,
            'formal_success_cap': 348, 'smoke_allowance': 2, 'budget_limit': 360}


def _code_hashes():
    return {Path(module.__file__).name: sha256(module.__file__)
            for module in (contract, eval_data, renderer, scene)} | {Path(__file__).name: sha256(__file__)}


def _publish_files(out, files, manifest, verify=None):
    out = Path(out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temp:
        dest = Path(temp)
        for name, value in files.items():
            (_write_rows if name.endswith('.jsonl') else _write_json)(dest / name, value)
        result = {**manifest, 'files': {name: sha256(dest / name) for name in files}}
        _write_json(dest / 'MANIFEST.json', result)
        if verify:
            verify()
        _publish(dest, out)
    return result


def freeze_reference(export, receipt, assetmap, out):
    """Freeze all six candidate populations, keeping every eligibility failure."""
    if os.path.lexists(out):
        raise FileExistsError(out)
    paths, provenance, rooms, assets = _inputs(export, receipt, assetmap)
    code = _code_hashes()
    provenance = {**provenance, 'implementation_sha256': code, 'seed': contract.DEFAULT_SEED}
    candidates = sorted((r for r in rooms if _stratum(r) in STRATA), key=lambda r: r['uid'])
    asset_index = defaultdict(list)
    for row in assets:
        asset_index[row.get('uid')].append(row)
    planner, resources = _planner()
    pairs = [_qualify(r, asset_index[r['uid']], provenance, planner) for r in candidates]
    cases, ledger = [p[0] for p in pairs], [p[1] for p in pairs]
    selection = _selection(candidates, ledger)
    counts = {'export_rows': len(rooms), 'candidates': len(cases),
              'eligible': sum(r['eligible'] for r in ledger),
              'failed': sum(not r['eligible'] for r in ledger), 'gpt_selected': len(selection['selected_uids'])}
    per_stratum = {key: {'candidates': sum(r['stratum'] == key for r in ledger),
                        'eligible': sum(r['stratum'] == key and r['eligible'] for r in ledger),
                        'failed': sum(r['stratum'] == key and not r['eligible'] for r in ledger)} for key in STRATA}
    manifest = {'version': 1, 'scope': SCOPE, 'provenance': provenance, 'counts': counts,
                'strata': per_stratum, 'expected_uids': [r['uid'] for r in candidates],
                'export_uid_sha256': _digest(sorted(r['uid'] for r in rooms)),
                'render_settings': RENDER_SETTINGS, 'paper_isaac_retrieval': False,
                'eligibility_policy': 'reference_only_real_resources_no_furniture_proxies',
                'failure_counts': dict(Counter(r['reason'] for r in ledger if not r['eligible']))}

    def verify():
        if any(sha256(path) != provenance[key + '_sha256'] for key, path in paths.items()):
            raise ValueError('input hash changed during freeze')
        if _code_hashes() != code:
            raise ValueError('implementation hash changed during freeze')

    return _publish_files(out, {'reference-cases.jsonl': cases, 'eligibility.jsonl': ledger,
                               'gpt-selection.json': selection, 'resources.json': resources}, manifest, verify)


def _prediction(row, case, arm):
    if row is None:
        raise ValueError('missing_output')
    if row.get('source', case['source']) != case['source'] or row.get('arm', arm) != arm:
        raise ValueError('prediction source/arm mismatch')
    complete = row.get('schema_complete', row.get('complete'))
    if complete is not True or row.get('complete') is False or row.get('parsed') is False or row.get('error'):
        raise ValueError('schema_incomplete')
    objects = row.get('raw_layout')
    if not isinstance(objects, list):
        raise ValueError('missing raw_layout')
    for obj in objects:
        renderer.validate_placement(obj)
    if [obj['id'] for obj in objects] != case['expected_ids']:
        raise ValueError('raw_layout ID inventory/order mismatch')
    # Deliberately no merge with reference records, no repaired layout or GT fill.
    return copy.deepcopy(objects)


def attach_predictions(reference_cases, joint, placement):
    """Return paired cases and an all-UID failure ledger; never replace failures."""
    paths = dict(reference_cases=Path(reference_cases), joint=Path(joint), placement=Path(placement))
    hashes = {key + '_sha256': sha256(path) for key, path in paths.items()}
    refs = _index(_read_rows(paths['reference_cases']))
    predictions = {arm: _index(_read_rows(paths[arm])) for arm in ('joint', 'placement')}
    for arm, records in predictions.items():
        if records.keys() - refs.keys():
            raise ValueError(f'{arm}: foreign prediction UID outside reference cohort')
    cases, ledger = [], []
    for uid, case in refs.items():
        errors, layouts = {}, {}
        if case.get('reference_eligible') is not True:
            errors['reference'] = case.get('input_error', 'reference_ineligible')
        for arm, records in predictions.items():
            try:
                layouts[arm] = _prediction(records.get(uid), case, arm)
            except ERRORS as exc:
                errors[arm] = str(exc)
        if not errors:
            cases.append({**copy.deepcopy(case), 'layouts': {**copy.deepcopy(case['layouts']), **layouts},
                          'schema_complete': True, 'prediction_provenance': hashes})
        ledger.append({'uid': uid, 'stratum': case['stratum'], 'status': 'failed' if errors else 'renderable',
                       'schema_complete': {arm: arm not in errors for arm in predictions}, 'errors': errors})
    if any(sha256(path) != hashes[key + '_sha256'] for key, path in paths.items()):
        raise ValueError('input hash changed during attachment')
    return {'cases': cases, 'ledger': ledger, 'provenance': hashes,
            'counts': {'reference_cohort': len(refs), 'renderable': len(cases), 'failed': len(refs)-len(cases)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze')
    for name in ('export', 'receipt', 'assetmap', 'out'):
        freeze.add_argument('--' + name, type=Path, required=True)
    attach = sub.add_parser('attach')
    for name in ('reference-cases', 'joint', 'placement', 'out'):
        attach.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze_reference(args.export, args.receipt, args.assetmap, args.out)
    else:
        result = attach_predictions(args.reference_cases, args.joint, args.placement)
        result = _publish_files(args.out, {'cases.jsonl': result['cases'], 'failures.jsonl': result['ledger']},
                                {k: result[k] for k in ('counts', 'provenance')})
    print(json.dumps({'counts': result['counts'], 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
