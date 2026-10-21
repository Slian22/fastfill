"""Finish actual payload inventory without repeating completed component/OOF fits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def main(root, prior_result, out):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Bounded CPU Slurm allocation required')
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    previous = json.loads(prior_result.read_bytes())
    issues = Counter()
    prior_failures = {k:v for k,v in previous['checks'].items() if v['failed'] and k != 'fatal_error'}
    if prior_failures:issues['previous_core_checks_failed'] += 1
    if set(previous['summary'].get('priors',{})) != {'full','0','1','2','3','4'}:
        issues['incomplete_prior_rebuild'] += 1
    files, missing_aux = {}, []
    run = root/'runs/unknownsize-20261013'
    bases = {'release':root/'automation/unknownsize-20261013/inputs/release',
             **{name:run/name for name in ['full-corpus-v2','main-data-v1','eval-data-v1']}}
    for scope,base in bases.items():
        manifest_path=base/'MANIFEST.json'
        raw=manifest_path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest != previous['files'][str(manifest_path)]['sha256']:
            issues['manifest_changed_since_core_scan'] += 1
        manifest=json.loads(raw)
        files[str(manifest_path)]={'sha256':digest,'bytes':len(raw),'complete_scan':True}
        for name,receipt in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe relative payload path')
            p=base/name
            expected=receipt['sha256'] if isinstance(receipt,dict) else receipt
            if not p.exists():
                detail={'scope':scope,'path':str(p),'expected_sha256':expected}
                if scope == 'release' and name not in ['train.jsonl','dev.jsonl','test.jsonl']:
                    missing_aux.append(detail)
                    continue
                issues['missing_required_payload'] += 1
                files[str(p)]={**detail,'complete_scan':False}
                continue
            before=p.stat()
            h,fields,count=hashlib.sha256(),Counter(),None
            if p.suffix == '.jsonl':
                count=0
                with p.open('rb') as f:
                    for line in f:
                        h.update(line)
                        row=json.loads(line)
                        count+=1
                        fields.update(row.keys())
            elif p.suffix == '.json':
                raw=p.read_bytes();h.update(raw);value=json.loads(raw)
                if isinstance(value,list):
                    count=len(value)
                    fields.update(k for v in value if isinstance(v,dict) for k in v)
                elif isinstance(value,dict):
                    count=len(value) if name=='inventory-sidecar.json' else 1
                    if name!='inventory-sidecar.json':fields.update(value.keys())
                del raw,value
            else:
                with p.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):issues['file_changed_during_read']+=1
            if h.hexdigest()!=expected:issues['payload_hash_mismatch']+=1
            files[str(p)]={'sha256':h.hexdigest(),'expected_sha256':expected,'bytes':after.st_size,
                           'records':count,'fields':dict(fields),'complete_scan':True}
        print(json.dumps({'stage':'payload_scope','scope':scope,'seconds':time.monotonic()-start}),flush=True)
    # Recheck byte identity of every already-scanned raw IR and source file as well.
    for name,record in previous['files'].items():
        if name in files or not record.get('complete_scan'):continue
        p=Path(name);h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=record['sha256']:issues['core_file_changed_since_rebuild']+=1
        files[name]={**record,'post_core_sha256':h.hexdigest()}
    result={'status':'PASS' if not issues else 'FAIL','utc':datetime.now(timezone.utc).isoformat(),
       'slurm_job_id':os.environ['SLURM_JOB_ID'],'elapsed_seconds':time.monotonic()-start,
       'issues':dict(issues),'prior_result':str(prior_result),
       'prior_result_sha256':hashlib.sha256(prior_result.read_bytes()).hexdigest(),
       'core_rebuild_completed_with_zero_nonfatal_failed_checks':not prior_failures,
       'missing_unconsumed_historical_release_auxiliary_files':missing_aux,'files':files,
       'scope':'All actual release train/dev/test, raw IR, corpus, main and eval payload records/bytes verified. Historical release manifest auxiliary files absent from the deliberately staged subset are enumerated, not certified. No component or median fit recomputation in this supplement.'}
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prior-result',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.prior_result,a.out)
