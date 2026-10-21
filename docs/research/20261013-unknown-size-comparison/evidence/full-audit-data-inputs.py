"""Full streaming frozen J/P input audit. CPU Slurm only; no model or API calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import resource
import sqlite3
import sys
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Audit:
    def __init__(self, out):
        self.out = out
        self.files = {}
        self.issues = Counter()
        self.examples = defaultdict(list)
        self.summary = {}
        self.start = time.monotonic()

    def check(self, ok, name, uid=None):
        if not ok:
            self.issues[name] += 1
            if len(self.examples[name]) < 5:
                self.examples[name].append(uid)

    def rows(self, path, expected=None):
        before = path.stat()
        hasher, fields, sources, seen = hashlib.sha256(), Counter(), Counter(), set()
        with path.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                hasher.update(line)
                row = json.loads(line)
                fields.update(row.keys())
                sources[row.get('source', 'unspecified')] += 1
                uid = row.get('uid')
                self.check(uid not in seen, 'duplicate_uid:' + str(path), uid)
                seen.add(uid)
                yield row
        sha = hasher.hexdigest()
        self.check(expected is None or sha == expected, 'file_hash_mismatch', str(path))
        after = path.stat()
        self.check((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                   'file_changed_during_scan', str(path))
        self.files[str(path)] = {'sha256': sha, 'bytes': after.st_size, 'records': sum(sources.values()),
                                 'fields': dict(fields), 'by_source': dict(sources), 'complete_scan': True}

    def read(self, path, expected=None):
        raw = path.read_bytes()
        sha = digest(raw)
        self.check(expected is None or sha == expected, 'file_hash_mismatch', str(path))
        self.files[str(path)] = {'sha256': sha, 'bytes': len(raw), 'complete_scan': True}
        return json.loads(raw)

    def save(self, status):
        self.out.write_text(json.dumps({'status': status, 'utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - self.start, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'python': sys.version, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'issues': dict(self.issues), 'examples': dict(self.examples), 'summary': self.summary,
            'files': self.files}, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def vector(value, positive=False):
    return (isinstance(value, list) and len(value) == 3 and
            all(type(v) in (int, float) and math.isfinite(v) and (v > 0 or not positive) for v in value))


def strict_request(audit, row, arm, split, system):
    uid, turns = row['uid'], row['messages']
    roles = ['system', 'user'] + ([] if split == 'test' else ['assistant'])
    audit.check([m.get('role') for m in turns] == roles, 'message_roles', uid)
    audit.check(turns[0]['content'] == system, 'system_prompt_mismatch', uid)
    request = json.loads(turns[1]['content'])
    allowed = {'boundary', 'boundary_type', 'room_type', 'height', 'fixed', 'objects'}
    audit.check(set(request) <= allowed, 'extra_prompt_top_fields', uid)
    objects = request['objects']
    ids = [o.get('id') for o in objects]
    audit.check(len(ids) == len(set(ids)) and all(isinstance(v, str) and v.startswith('obj_') for v in ids),
                'invalid_or_duplicate_movable_ids', uid)
    expected = {'id', 'category'} | ({'size'} if arm == 'placement' else set())
    for obj in objects:
        audit.check(set(obj) == expected, 'strict_movable_allowlist:' + arm, uid)
        audit.check(isinstance(obj.get('category'), str) and bool(obj['category'].strip()),
                    'invalid_category_type_or_empty:' + arm, uid)
        if arm == 'placement':
            audit.check(vector(obj.get('size'), True), 'invalid_prior_size', uid)
    for obj in request.get('fixed', []):
        audit.check(set(obj) == {'id', 'category', 'size', 'pos', 'yaw'}, 'fixed_field_contract', uid)
    return request


def target(audit, turns, request, arm, uid, expected_parents):
    value = json.loads(turns[-1]['content'])
    audit.check(set(value) == {'placements'}, 'target_top_fields', uid)
    placements = value['placements']
    ids = [o.get('id') for o in placements]
    audit.check(ids == [o['id'] for o in request['objects']], 'target_completeness_order', uid)
    for obj in placements:
        required = {'id', 'pos', 'yaw'} | ({'size'} if arm == 'joint' else set())
        audit.check(required <= set(obj) <= required | {'on'}, 'target_object_fields', uid)
        audit.check(vector(obj.get('pos')), 'target_nonfinite_position', uid)
        audit.check(type(obj.get('yaw')) in (int, float) and math.isfinite(obj['yaw']) and 0 <= obj['yaw'] < 360,
                    'target_invalid_yaw', uid)
        if arm == 'joint':
            audit.check(vector(obj.get('size'), True), 'target_invalid_size', uid)
        audit.check(obj.get('on') == expected_parents.get(obj['id']), 'target_support_identity_mismatch', uid)
    return value


def main(root, out):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Full audit must run inside a bounded Slurm CPU job')
    out.mkdir(parents=True, exist_ok=False)
    audit = Audit(out / 'result.json')
    run = root / 'runs/unknownsize-20261013'
    snap = root / 'automation/unknownsize-20261013/snapshots/sft-c8f792a65470'
    prior_snap = root / 'automation/unknownsize-20261013/snapshots/main-data-50a274bc4c46'
    sys.path.insert(0, str(prior_snap))
    from fastfill.comparison import contract, size_prior
    from transformers import AutoTokenizer
    audit.summary['versions'] = {n: importlib.metadata.version(n) for n in ['transformers','tokenizers','numpy','shapely']}
    audit.summary['resources'] = {'cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
                                'mem_per_node_mb': os.environ.get('SLURM_MEM_PER_NODE'), 'gpu_requested': 0}
    for p in [Path(__file__), prior_snap/'fastfill/comparison/contract.py', prior_snap/'fastfill/comparison/size_prior.py',
              prior_snap/'fastfill/scene.py', snap/'fastfill/comparison/train_pair_helpers.py', snap/'fastfill/comparison/train_pair.py']:
        audit.files[str(p)] = {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size, 'source_read': True}
    manifests = {n: audit.read(run/n/'MANIFEST.json') for n in ['main-data-v1','eval-data-v1']}
    main_manifest = manifests['main-data-v1']
    priors = audit.read(run/'main-data-v1/size-priors.json', main_manifest['files']['size-priors.json'])
    ledgers = [audit.read(run/f'sft-{arm}-001/lengths.json') for arm in ['joint','placement']]
    audit.check(ledgers[0] == ledgers[1], 'unequal_length_ledgers')
    receipts = [audit.read(run/f'sft-{arm}-001/run_receipt.json') for arm in ['joint','placement']]
    audit.check(receipts[0]['data'] == receipts[1]['data'], 'unequal_data_receipts')
    audit.check(receipts[0]['base'] == receipts[1]['base'], 'unequal_base_receipts')
    base = root/'models/Qwen3-8B'
    for n in ['tokenizer.json','tokenizer_config.json','config.json']:
        audit.read(base/n, receipts[0]['base']['files'][n])
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    db = sqlite3.connect(out/'audit-index.sqlite')
    db.execute('CREATE TABLE rooms (uid TEXT PRIMARY KEY, split TEXT, source TEXT, request TEXT, target TEXT, parents TEXT, sizes TEXT)')
    for dataset, filename in [('main-data-v1','bound_rooms.jsonl'), ('eval-data-v1','rooms.jsonl')]:
        for room in audit.rows(run/dataset/filename, manifests[dataset]['files'][filename]):
            uid, split = room['uid'], room['split']
            turns = contract.messages(room, 'joint')
            req, label = [json.loads(t['content']) for t in turns[1:]]
            features = {'room_type': room.get('room_type'), 'objects': [{'id': o['id'], 'category': o.get('category')} for o in room['objects']]}
            prior = priors['priors'][str(priors['fold_by_uid'][uid])] if split == 'train' else priors['full_prior']
            sizes = {k: [max(.01, round(x,2)) for x in v] for k,v in size_prior.predict_sizes(features,prior).items()}
            parents = {o['id']: o.get('parent') for o in room['objects']}
            db.execute('INSERT INTO rooms VALUES (?,?,?,?,?,?,?)',
                (uid, split, room['source'], canonical(req), canonical(label), canonical(parents), canonical(sizes)))
        db.commit()
    del priors
    audit.save('RUNNING')
    results = {}
    all_uids = {}
    for split in ['train','dev','test']:
        dataset = 'eval-data-v1' if split == 'test' else 'main-data-v1'
        filename = 'requests.jsonl' if split == 'test' else split+'.jsonl'
        streams = [audit.rows(run/dataset/arm/filename, manifests[dataset]['files'][arm+'/'+filename]) for arm in ['joint','placement']]
        by_source = Counter()
        counters = Counter()
        accepted = set()
        seen = set()
        lengths = {} if split == 'test' else {r['uid']: r for r in ledgers[0][split]['lengths']}
        recorded_admitted = set() if split == 'test' else set(ledgers[0][split]['accepted_uids'])
        for index, pair in enumerate(itertools.zip_longest(*streams), 1):
            j,p = pair
            if j is None or p is None:
                audit.check(False, 'unequal_arm_row_count', split)
                continue
            uid = j['uid']
            seen.add(uid)
            audit.check(uid == p['uid'], 'paired_uid_or_order_mismatch', uid)
            audit.check(all(j.get(k) == p.get(k) for k in ['source','group','split']), 'paired_metadata_mismatch', uid)
            ref = db.execute('SELECT split,source,request,target,parents,sizes FROM rooms WHERE uid=?',(uid,)).fetchone()
            if ref is None:
                audit.check(False, 'missing_bound_reference', uid)
                continue
            refsplit, source, reqtext, tgttext, parenttext, sizetext = ref
            audit.check(refsplit == split and j.get('split') == split and j['source'] == source, 'record_reference_identity', uid)
            by_source[source] += 1
            expected, expected_parents, sizes = json.loads(reqtext), json.loads(parenttext), json.loads(sizetext)
            totals, answers = {}, {}
            for arm,row in [('joint',j),('placement',p)]:
                req = strict_request(audit,row,arm,split,contract._SYSTEM[arm])
                projected = {**req, 'objects':[{k:v for k,v in o.items() if k != 'size'} for o in req['objects']]}
                audit.check(canonical(projected) == reqtext, 'prompt_differs_from_frozen_inventory:' + arm, uid)
                counters['objects:'+arm] += len(req['objects'])
                if arm == 'placement':
                    audit.check({o['id']:o['size'] for o in req['objects']} == sizes, 'P_size_prior_mismatch', uid)
                turns = row['messages']
                prompt = tokenizer.apply_chat_template(turns[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
                counters['prompt_tokens:'+arm] += len(prompt_ids)
                if split != 'test':
                    label = target(audit,turns,req,arm,uid,expected_parents)
                    if arm == 'joint':
                        audit.check(canonical(label) == tgttext, 'J_target_differs_from_reference', uid)
                    answer_ids = tokenizer(turns[-1]['content']+'<|im_end|>', add_special_tokens=False)['input_ids']
                    totals[arm], answers[arm] = len(prompt_ids)+len(answer_ids),len(answer_ids)
                    audit.check(answer_ids[-1] == tokenizer.convert_tokens_to_ids('<|im_end|>'), 'assistant_EOS_token', uid)
                    if uid in lengths:
                        audit.check(totals[arm] == lengths[uid]['lengths'][arm], 'retokenized_total_length_mismatch:'+arm, uid)
                        audit.check(answers[arm] == lengths[uid]['answer_tokens'][arm], 'retokenized_answer_length_mismatch:'+arm, uid)
                    else:
                        audit.check(False,'missing_length_record',uid)
                else:
                    counters['input_ge_40960:'+arm] += len(prompt_ids) >= 40960
            if split != 'test' and all(v <= 8192 for v in totals.values()):
                accepted.add(uid)
                for arm in ['joint','placement']:
                    counters['admitted_total_tokens:'+arm] += totals[arm]
                    counters['admitted_supervised_tokens:'+arm] += answers[arm]
            if index % 5000 == 0:
                print(json.dumps({'stage':split,'rows':index,'issue_count':sum(audit.issues.values()),'elapsed':time.monotonic()-audit.start}),flush=True)
        expected_uids = {x[0] for x in db.execute('SELECT uid FROM rooms WHERE split=?',(split,))}
        audit.check(seen == expected_uids,'full_UID_coverage:'+split)
        if split != 'test':
            audit.check(accepted == recorded_admitted,'retokenized_admission_mismatch:'+split)
            audit.check(seen == set(lengths),'length_ledger_coverage:'+split)
        all_uids[split] = seen
        results[split] = {'rows_per_arm':len(seen),'by_source':dict(by_source),'admitted_per_arm':len(accepted) if split!='test' else None,
                          'excluded_by_length':len(seen-accepted) if split!='test' else None,'counts':dict(counters)}
        audit.summary['populations'] = results
        audit.save('RUNNING')
    for first,second in itertools.combinations(all_uids,2):
        audit.check(not all_uids[first] & all_uids[second], 'cross_split_UID_overlap:'+first+':'+second)
    # Bind all qualified training hashes against the fully consumed request records.
    ledger_by_uid = {}
    for row in audit.rows(run/'main-data-v1/common-admission.jsonl', main_manifest['files']['common-admission.jsonl']):
        if row['accepted']:
            ledger_by_uid[row['uid']] = row
    audit.check(set(ledger_by_uid)==all_uids['train'],'qualification_UID_coverage')
    for arm in ['joint','placement']:
        for row in audit.rows(run/'main-data-v1'/arm/'train.jsonl',main_manifest['files'][arm+'/train.jsonl']):
            turns,uid = row['messages'],row['uid']
            hashes = {'target_sha256':digest(turns[-1]['content'].encode()),
                      'request_sha256':digest(json.dumps(turns[:-1],ensure_ascii=False,allow_nan=False,sort_keys=True).encode())}
            if arm=='placement':
                values={o['id']:o['size'] for o in json.loads(turns[1]['content'])['objects']}
                hashes['estimated_sizes_sha256']=digest(json.dumps(values,ensure_ascii=False,allow_nan=False,sort_keys=True).encode())
            for key,value in hashes.items():
                audit.check(value==ledger_by_uid[uid][arm][key], 'qualification_'+key+':'+arm,uid)
    db.close()
    audit.save('COMPLETE')
    print(json.dumps({'status':'COMPLETE','issues':dict(audit.issues),'populations':results}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    main(args.root,args.out)
