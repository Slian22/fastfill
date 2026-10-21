"""Supplement: repaired P train poses may differ; unmodified dev poses may not."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path


def main(root, out):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run in bounded Slurm allocation')
    out.mkdir(parents=True, exist_ok=False)
    base = root / 'runs/unknownsize-20261013/main-data-v1'
    manifest = json.loads((base/'MANIFEST.json').read_bytes())
    counts, issues, files = {}, Counter(), {}
    for split in ['train', 'dev']:
        stats = Counter()
        paths = [base/arm/(split+'.jsonl') for arm in ['joint','placement']]
        hashes = [hashlib.sha256(),hashlib.sha256()]
        with paths[0].open('rb') as left,paths[1].open('rb') as right:
            for lines in itertools.zip_longest(left,right):
                if None in lines:
                    issues['different_pair_lengths'] += 1
                    continue
                for h,line in zip(hashes,lines):h.update(line)
                j,p = [json.loads(line) for line in lines]
                if j['uid'] != p['uid']:issues['different_pair_UIDs'] += 1
                jt,pt = [json.loads(row['messages'][-1]['content']) for row in [j,p]]
                projected = {'placements':[{k:v for k,v in obj.items() if k!='size'} for obj in jt['placements']]}
                different = projected != pt
                stats['records'] += 1
                stats['poses_different'] += different
                stats['poses_equal'] += not different
                if split == 'dev' and different:issues['dev_unexpected_target_change'] += 1
                frozen = lambda target:[{k:v for k,v in obj.items() if k not in ['pos','yaw','size']} for obj in target['placements']]
                if frozen(jt) != frozen(pt):issues['target_inventory_or_support_changed'] += 1
        for p,h in zip(paths,hashes):
            digest = h.hexdigest()
            if digest != manifest['files'][str(p.relative_to(base))]:issues['input_hash_mismatch'] += 1
            files[str(p)] = {'sha256':digest,'complete_scan':True,'records':stats['records']}
        counts[split] = dict(stats)
    snapshot = root/'automation/unknownsize-20261013/snapshots/main-data-50a274bc4c46/fastfill/comparison/main_data.py'
    files[str(snapshot)] = {'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'policy_evidence':'_candidate pose repair; _write_data unrepaired dev'}
    files[str(Path(__file__))] = {'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result = {'status':'COMPLETE','utc':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
       'issues':dict(issues),'counts':counts,'files':files,
       'correction':'geometry-7367978 P_target_differs_from_J_minus_size was an invalid audit invariant for pose-repaired TRAIN. Treat its count as descriptive, not a dataset error. This supplement checks the intended split-specific contract. Training target bytes were independently bound to qualification hashes in inputs-7367962.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    main(args.root,args.out)
