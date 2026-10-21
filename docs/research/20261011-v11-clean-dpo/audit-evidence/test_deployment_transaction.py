import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile

import pytest

ROOT=Path('/tmp/fastfill-v11-reaudit-20261011')
OLD=('7252878','7252879','7252880','7252881','7252889')


@pytest.fixture
def transaction(tmp_path,monkeypatch):
    helper=tmp_path/'original'/'automation'/'new helpers'
    helper.mkdir(parents=True)
    (helper/'audit-source-manifest.json').write_text('{"source_sha256":"test-frozen-source"}')
    run=helper.parent.parent/'runs/fastfill_v1_1_clean_20261011'
    (run/'logs').mkdir(parents=True)
    monkeypatch.setattr(sys,'argv',['replace_pending_jobs.py',str(helper)])
    state={'jobs':{job:{'JobState':'PENDING','RunTime':'00:00:00','Requeue':'0',
                       'ReqTRES':'cpu=32,gres/gpu='+('0' if job=='7252889' else '1'),
                       'Priority':'100','Reason':'Dependency'} for job in OLD},
           'calls':[],'counter':8000000,'inspections':{},'fail_submit':None,'bad_gpu':False,'old_running':False}
    def command(argv,**kwargs):
        state['calls'].append(list(argv))
        stdout=''
        if argv[0]=='sbatch':
            state['counter']+=1
            if state['fail_submit']==state['counter']-8000000:
                raise subprocess.CalledProcessError(1,argv,'injected submit failure')
            assert '--hold' in argv and '--no-requeue' in argv
            job=str(state['counter']); name=next(a.split('=',1)[1] for a in argv if a.startswith('--job-name='))
            gpu='1' if name.startswith('ff-verified-') else '0'
            if state['bad_gpu'] and name=='ff-verified-dpo1-1':gpu='4'
            state['jobs'][job]={'JobState':'PENDING','RunTime':'00:00:00','Requeue':'0',
                                'ReqTRES':'cpu=32,gres/gpu='+gpu,'Priority':'100' if state.get('bad_hold') else '0','Reason':'JobHeldUser'}
            stdout=job+'\n'
        elif argv[:3]==['scontrol','show','job']:
            job=argv[-1]; n=state['inspections'].get(job,0)+1; state['inspections'][job]=n
            if state['old_running'] and job=='7252889' and n>=2:
                state['jobs'][job]={**state['jobs'][job],'JobState':'RUNNING','RunTime':'00:00:01'}
            stdout='JobId='+job+' '+' '.join(k+'='+v for k,v in state['jobs'][job].items())
        elif argv[:2]==['scontrol','hold']:
            job=argv[-1]
            if state['jobs'][job]['JobState']!='PENDING':raise subprocess.CalledProcessError(1,argv)
            state['jobs'][job]={**state['jobs'][job],'Priority':'0','Reason':'JobHeldUser'}
        elif argv[:2]==['scontrol','release']:
            job=argv[-1]; assert job not in OLD
            state['jobs'][job]={**state['jobs'][job],'Priority':'100','Reason':'Dependency'}
        elif argv[0]=='scancel':
            job=argv[-1]; assert job in OLD
            assert state['jobs'][job]['JobState']=='PENDING'
            assert state['jobs'][job]['RunTime']=='00:00:00'
            state['jobs'][job]={**state['jobs'][job],'JobState':'CANCELLED'}
        else:
            raise AssertionError('Forbidden real command: '+str(argv))
        return subprocess.CompletedProcess(argv,0,stdout,'')
    monkeypatch.setattr(subprocess,'run',command)
    return state,run


def run_transaction():
    runpy.run_path(str(ROOT/'replace_pending_jobs.py'),run_name='__main__')


def test_success_holds_and_checks_before_report_first_cancel_then_release(transaction):
    state,run=transaction
    run_transaction()
    receipt=json.loads((run/'audit-launch.json').read_text())
    assert receipt['complete'] is True
    assert len(receipt['jobs'])==7
    assert receipt['gpu_hour_cap_unchanged']==84 and receipt['cpu_audit_max_hours']==12.5
    cancellations=[cmd[-1] for cmd in state['calls'] if cmd[0]=='scancel']
    assert cancellations==['7252889','7252879','7252881','7252878','7252880']
    released=[cmd[-1] for cmd in state['calls'] if cmd[:2]==['scontrol','release']]
    assert len(released)==7
    holds=[(i,cmd[-1]) for i,cmd in enumerate(state['calls']) if cmd[:2]==['scontrol','hold']]
    assert {job for _,job in holds}==set(OLD)
    first_cancel=min(i for i,cmd in enumerate(state['calls']) if cmd[0]=='scancel')
    assert max(i for i,_ in holds)<first_cancel
    last_cancel=max(i for i,cmd in enumerate(state['calls']) if cmd[0]=='scancel')
    first_release=min(i for i,cmd in enumerate(state['calls']) if cmd[:2]==['scontrol','release'])
    assert last_cancel<first_release


@pytest.mark.parametrize('failure',['gpu','submit','old_running','not_held'])
def test_failures_leave_new_jobs_held_and_do_not_release(transaction,failure):
    state,run=transaction
    if failure=='gpu':state['bad_gpu']=True
    if failure=='not_held':state['bad_hold']=True
    if failure=='submit':state['fail_submit']=4
    if failure=='old_running':state['old_running']=True
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        run_transaction()
    assert not any(cmd[:2]==['scontrol','release'] for cmd in state['calls'])
    assert not any(cmd[0]=='scancel' for cmd in state['calls'])
    receipt=json.loads((run/'audit-launch.json').read_text())
    assert receipt['complete'] is False
    if failure!='not_held':
        assert all(job['Priority']=='0' for key,job in state['jobs'].items() if key not in OLD)


def test_existing_receipt_prevents_duplicate_submissions(transaction):
    state,run=transaction
    (run/'audit-launch.json').write_text('{}')
    with pytest.raises(FileExistsError):run_transaction()
    assert not state['calls']


def make_bundle(tmp_path,*,wrong_file_hash=False,escape=False):
    archive=tmp_path/'source.tar.gz'
    code=b'# frozen helper\n'
    expected='0'*64 if wrong_file_hash else hashlib.sha256(code).hexdigest()
    manifest=json.dumps({'files':{'fastfill/helper.py':expected},'source_sha256':'bundle-source'}).encode()
    files={'fastfill/helper.py':code,'audit-source-manifest.json':manifest}
    if escape:files['../escaped.py']=b'not allowed'
    with tarfile.open(archive,'w:gz') as bundle:
        for name,content in files.items():
            info=tarfile.TarInfo(name);info.size=len(content);bundle.addfile(info,io.BytesIO(content))
    return archive,hashlib.sha256(archive.read_bytes()).hexdigest()


def test_valid_source_install_is_atomic_and_cannot_overwrite(tmp_path,monkeypatch):
    archive,digest=make_bundle(tmp_path);destination=tmp_path/'new root'
    monkeypatch.setattr(sys,'argv',['deploy_helpers.py',str(archive),digest,str(destination)])
    runpy.run_path(str(ROOT/'deploy_helpers.py'),run_name='__main__')
    assert (destination/'fastfill/helper.py').read_bytes()==b'# frozen helper\n'
    with pytest.raises(FileExistsError):runpy.run_path(str(ROOT/'deploy_helpers.py'),run_name='__main__')


@pytest.mark.parametrize('bad',['digest','file_hash','escape'])
def test_invalid_source_never_publishes_destination(tmp_path,monkeypatch,bad):
    archive,digest=make_bundle(tmp_path,wrong_file_hash=bad=='file_hash',escape=bad=='escape')
    destination=tmp_path/'new root'
    monkeypatch.setattr(sys,'argv',['deploy_helpers.py',str(archive),'0'*64 if bad=='digest' else digest,str(destination)])
    with pytest.raises((ValueError,tarfile.TarError)):
        runpy.run_path(str(ROOT/'deploy_helpers.py'),run_name='__main__')
    assert not destination.exists()
    assert not (tmp_path/'escaped.py').exists()
