"""Full frozen IR/source-contract exposure audit; never opens source meshes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


class Scan:
    def __init__(self):
        self.files = {}
        self.issues = Counter()

    def rows(self, path):
        before = path.stat()
        h, fields, n = hashlib.sha256(), Counter(), 0
        with path.open('rb') as f:
            for line in f:
                h.update(line)
                row = json.loads(line)
                n += 1
                fields.update(row.keys())
                yield row
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            self.issues['changed_during_scan'] += 1
        self.files[str(path)] = dict(sha256=h.hexdigest(), bytes=after.st_size,
                                    records=n, fields=dict(fields), complete_scan=True)

    def read(self, path):
        raw = path.read_bytes()
        self.files[str(path)] = dict(sha256=sha(raw), bytes=len(raw), complete_scan=True)
        return json.loads(raw)


META_VALUES = {'boundary_source', 'boundary_from', 'poly_source', 'floor_z_source',
               'size_from', 'position_from', 'front_source', 'size_note', 'asset_source',
               'anchor_source', 'front_known', 'is_true_room', 'height_reliable',
               'eval_only', 'source_coverage'}
ASSET_FIELDS = {'furniture_asset_id', 'asset_id', 'model_id', 'mesh_path', 'mesh',
                'model_path', 'glb_path', 'asset_path'}
GEOM_FIELDS = {'size', 'pos', 'yaw', 'anchor', 'parent', 'tilted', 'front_known',
               'z_src', 'fixed_kind', 'category', 'rot_src', 'raw_basis', 'rot_wxyz_yup'}


def profile(room):
    """Count metadata literally; no conversion to unsupported physical truth."""
    c = Counter(rooms=1)
    c['boundary_type:' + str(room.get('boundary_type', '<missing>'))] += 1
    h = room.get('height')
    c['height:finite_positive' if type(h) in (int, float) and math.isfinite(h) and h > 0
      else 'height:missing_invalid_nonpositive'] += 1
    meta = room.get('meta') or {}
    c['room_front_known:' + str(meta.get('front_known', '<missing>'))] += 1
    for key, value in meta.items():
        c['meta_field:' + key] += 1
        if key in META_VALUES and not isinstance(value, (dict, list)):
            c['meta_value:' + key + ':' + str(value)] += 1
        if key.startswith('n_') and type(value) in (int, float) and math.isfinite(value):
            c['meta_sum:' + key] += value
            if value:
                c['meta_nonzero_rooms:' + key] += 1
    for key, value in (meta.get('flags') or {}).items():
        if value:
            c['prepared_flag_rooms:' + key] += 1
    for kind in ['objects', 'fixed']:
        records = room.get(kind) or []
        c[kind] += len(records)
        for obj in records:
            for key in obj:
                c[kind + ':field:' + key] += 1
            c[kind + ':front_known:' + str(obj.get('front_known', '<missing>'))] += 1
            c[kind + ':tilted:' + str(obj.get('tilted', '<missing>'))] += 1
            c[kind + ':anchor:' + str(obj.get('anchor', '<missing>'))] += 1
            c[kind + ':has_asset_reference_field'] += bool(ASSET_FIELDS & set(obj))
            c[kind + ':has_mask_field'] += any('mask' in k.lower() for k in obj)
            c[kind + ':has_parent'] += bool(obj.get('parent'))
            category = str(obj.get('category', '')).lower()
            if 'door' in category or 'window' in category:
                c[kind + ':opening_category'] += 1
    return c


def main(root, out, cases_path):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Full scans require bounded Slurm allocation')
    out.mkdir(parents=True, exist_ok=False)
    start, scan = time.monotonic(), Scan()
    run = root / 'runs/unknownsize-20261013'
    inputs = root / 'automation/unknownsize-20261013/inputs'
    cases_doc = scan.read(cases_path)
    cases = cases_doc['cases']
    unresolved_windows, unresolved_objects = [], []
    for item in cases_doc.get('limitations', []):
        if isinstance(item, dict):
            unresolved_windows.extend(item.get('unresolved_il3d_raw_locators', []))
            unresolved_objects.extend(item.get('unresolved_interiorgs_object_ids', []))
    resolved_locators = []
    by_uid = defaultdict(list)
    for i, case in enumerate(cases):
        case['audit_case_index'] = i
        by_uid[case['uid']].append(i)
    encounters = defaultdict(lambda: defaultdict(dict))
    profiles = defaultdict(lambda: defaultdict(Counter))
    populations = defaultdict(dict)
    admitted = scan.read(run / 'sft-joint-001/lengths.json')
    admitted_train = set(admitted['train']['accepted_uids'])
    admitted_dev = set(admitted['dev']['accepted_uids'])
    del admitted
    ledger = {}
    for row in scan.rows(run / 'main-data-v1/common-admission.jsonl'):
        ledger[row['uid']] = (row['eligible'], row['accepted'])
    released = {}
    for split in ['train', 'dev', 'test']:
        for row in scan.rows(inputs / 'release' / (split + '.jsonl')):
            released[row['uid']] = split

    def observe(room, stages, mapping=None):
        uid, source = room['uid'], room['source']
        counters = profile(room)
        for stage in stages:
            profiles[stage][source].update(counters)
            populations[stage][uid] = source
        if uid not in by_uid:
            return
        objects = {}
        for kind in ['objects', 'fixed']:
            for obj in room.get(kind) or []:
                oid = (mapping or {}).get(obj['id'], obj['id'])
                if oid in objects:
                    scan.issues['duplicate_case_object_original_id'] += 1
                objects[oid] = (kind, obj)
        for idx in by_uid[uid]:
            case = cases[idx]
            found = objects.get(case['object_id'])
            for stage in stages:
                hit = {'room_present': True, 'object_present': found is not None,
                       'source': source, 'split': room.get('split', released.get(uid))}
                if found:
                    kind, obj = found
                    geom = {k: obj[k] for k in sorted(GEOM_FIELDS & set(obj))}
                    hit.update(kind=kind, geometry=clean(geom), current_id=obj['id'],
                               geometry_sha256=sha(json.dumps(clean(geom), sort_keys=True).encode()))
                    if stage == 'raw_all' and 'historical_ir_size_m' in case:
                        hit['matches_historical_ir_size_1e-9'] = all(
                            abs(a-b) <= 1e-9 for a,b in zip(obj['size'],case['historical_ir_size_m']))
                    if stage == 'raw_all' and 'historical_ir_width_m' in case:
                        hit['matches_historical_ir_width_1e-9'] = abs(obj['size'][0]-case['historical_ir_width_m']) <= 1e-9
                encounters[str(idx)][stage] = hit

    def resolve(room):
        candidates = []
        meta, uid = room.get('meta') or {}, room['uid']
        if room['source'] == 'IL3D_3dfront':
            for item in unresolved_windows:
                if (meta.get('house'),meta.get('source_room_id')) == (item['house'],item['source_room']):
                    candidates.append({**item, 'finding_id':'il3d_slanted_window_projected_width',
                        'source':room['source'], 'uid':uid, 'object_id':uid+'::window_'+str(item['opening_index']),
                        'kind':'confirmed_geometry_error', 'resolved_from_source_locator':True})
        if room['source'] == 'InteriorGS':
            ids = {o['id'] for o in room.get('objects', [])}
            for item in unresolved_objects:
                if item['object_id'] in ids:
                    candidates.append({**item, 'finding_id':'interiorgs_thin_box_shortcut',
                        'uid':uid, 'resolved_from_source_locator':True})
        for case in candidates:
            if any(cases[i]['object_id'] == case['object_id'] and cases[i]['finding_id'] == case['finding_id']
                   for i in by_uid[uid]):
                continue
            idx = len(cases)
            cases.append({**case,'audit_case_index':idx})
            by_uid[uid].append(idx)
            resolved_locators.append(idx)

    raw_sources = []
    for path in sorted((inputs / 'ir').glob('*.jsonl')):
        raw_sources.append(path.stem)
        for room in scan.rows(path):
            resolve(room)
            stage = 'raw_released' if room['uid'] in released else 'raw_unreleased'
            observe(room, ['raw_all', stage])
        print(json.dumps({'stage':'raw_source_done','source':path.stem,
                          'elapsed':time.monotonic()-start}), flush=True)
    for room in scan.rows(run / 'full-corpus-v2/prepared_rooms.jsonl'):
        uid, split = room['uid'], room['split']
        stages = ['prepared_all', 'prepared_' + split]
        if split == 'train':
            eligible, qualified = ledger[uid]
            stages += ['prior_eligible_train' if eligible else 'historical_ineligible_train']
            if eligible and not qualified:
                stages += ['eligible_decoder_rejected_train']
            if qualified:
                stages += ['decoder_qualified_train']
            if uid in admitted_train:
                stages += ['length_admitted_train']
            elif qualified:
                stages += ['length_dropped_train']
        elif split == 'dev':
            stages += ['length_admitted_dev' if uid in admitted_dev else 'length_dropped_dev']
        observe(room, stages)
    for dataset, filename in [('main-data-v1', 'bound_rooms.jsonl'), ('eval-data-v1', 'rooms.jsonl')]:
        all_maps = scan.read(run / dataset / 'inventory-sidecar.json')
        maps = {uid: value['bound_to_original'] for uid, value in all_maps.items() if uid in by_uid}
        del all_maps
        for room in scan.rows(run / dataset / filename):
            uid, split = room['uid'], room['split']
            stages = ['actual_bound_' + split]
            if uid in admitted_train:
                stages += ['actual_admitted_train']
            if uid in admitted_dev:
                stages += ['actual_admitted_dev']
            if uid in by_uid and uid not in maps:
                scan.issues['known_case_missing_identity_map'] += 1
            observe(room, stages, maps.get(uid))
        del maps
    paired_target_counts = {}
    for split in ['train', 'dev']:
        count = 0
        left = scan.rows(run / 'main-data-v1/joint' / (split+'.jsonl'))
        right = scan.rows(run / 'main-data-v1/placement' / (split+'.jsonl'))
        for j,p in itertools.zip_longest(left,right):
            if j is None or p is None or j['uid'] != p['uid']:
                scan.issues['target_pair_uid_or_length'] += 1
                continue
            jt,pt = [json.loads(x['messages'][-1]['content']) for x in [j,p]]
            projected = {'placements':[{k:v for k,v in obj.items() if k!='size'} for obj in jt['placements']]}
            if projected != pt:
                scan.issues['P_target_differs_from_J_minus_size'] += 1
            count += 1
        paired_target_counts[split] = count
    groups = defaultdict(lambda: defaultdict(lambda: {'case_indices': [], 'uids': set(), 'kinds': Counter()}))
    for idx, case in enumerate(cases):
        finding = case['finding_id']
        for stage, hit in encounters[str(idx)].items():
            if hit['object_present']:
                rec = groups[finding][stage]
                rec['case_indices'].append(idx)
                rec['uids'].add(case['uid'])
                rec['kinds'][hit['kind']] += 1
    grouped = {finding: {stage: {'objects':len(v['case_indices']), 'rooms':len(v['uids']),
                  'by_kind':dict(v['kinds']), 'case_indices':v['case_indices']}
                  for stage,v in stages.items()} for finding,stages in groups.items()}
    population_summary = {stage:dict(Counter(uids.values())) for stage,uids in populations.items()}
    for first, second in [('length_admitted_train','actual_admitted_train'),
                          ('length_admitted_dev','actual_admitted_dev'),
                          ('prepared_test','actual_bound_test')]:
        if populations[first] != populations[second]:
            scan.issues['population_alignment:' + first + ':' + second] += 1
    script_raw = Path(__file__).read_bytes()
    scan.files[str(Path(__file__))] = dict(sha256=sha(script_raw),bytes=len(script_raw),source_read=True)
    result = {'status':'COMPLETE', 'utc':datetime.now(timezone.utc).isoformat(),
       'slurm_job_id':os.environ['SLURM_JOB_ID'], 'elapsed_seconds':time.monotonic()-start,
       'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'issues':dict(scan.issues), 'source_names':raw_sources,
       'populations_by_source':population_summary,
       'profiles':{stage:{source:dict(c) for source,c in sources.items()} for stage,sources in profiles.items()},
       'known_case_counts':grouped, 'known_cases':cases, 'known_case_encounters':dict(encounters),
       'resolved_locator_case_indices':resolved_locators, 'paired_targets_checked':paired_target_counts,
       'case_evidence':cases_doc.get('evidence', []), 'files':scan.files,
       'limitations':['Every frozen IR and prepared/bound record scanned; no binary assets read.',
          'Known-case joins establish current exposure, not an independent source-mesh remeasurement.',
          'Metadata front_known and boundary_type are declarations, not calibrated ground truth.',
          'Missing case object does not exclude indirect influence on boundary, parent, or eligibility.',
          'Per-room n_* metadata sums may repeat scene-level values; not unique asset-defect counts.',
          'v2 mask/correction counts belong to another artifact and are not imported here.']}
    (out / 'result.json').write_text(json.dumps(clean(result), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status':'COMPLETE','issues':dict(scan.issues),
          'population_totals':{s:sum(v.values()) for s,v in population_summary.items()},
          'known_case_counts':{f:{s:{k:v for k,v in x.items() if k!='case_indices'} for s,x in st.items()}
                               for f,st in grouped.items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.out, args.cases)
