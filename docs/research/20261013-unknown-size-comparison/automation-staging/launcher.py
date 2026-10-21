import json,os,signal,subprocess,sys,threading,time
from pathlib import Path
config_path=Path(sys.argv[1]); c=json.loads(config_path.read_text()); doc=config_path.parent
children=[]
driver=None
def interrupted(signum,frame):
    raise SystemExit(124 if signum==signal.SIGALRM else 130)
for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM):signal.signal(signum,interrupted)
signal.alarm(int(c['total_timeout_seconds']))
def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
def write(name,value):
    path=doc/'evidence'/name; tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
code=2
try:
    sock=c['ssh_control_path']; host=c['ssh_host']
    master_args=['/usr/bin/ssh','-o','BatchMode=yes','-o','ConnectTimeout=30','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ForwardAgent=no','-o','ForwardX11=no','-o','ControlPersist=no','-M','-N','-n','-S',sock,host]
    with (doc/'automation-staging/ssh-master-launchd.log').open('x') as log:
        master=subprocess.Popen(master_args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(master)
    deadline=time.monotonic()+60
    while True:
        ready=subprocess.run(['/usr/bin/ssh','-S',sock,'-O','check',host],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5).returncode==0
        if ready:break
        if master.poll() is not None or time.monotonic()>deadline:raise RuntimeError('SSH master readiness failed')
        time.sleep(1)
    probe=subprocess.run(['/usr/bin/ssh','-v','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath='+sock,host,'true'],capture_output=True,text=True,check=True,timeout=30)
    evidence=[line for line in probe.stderr.splitlines() if 'mux_client' in line]
    if not evidence:raise RuntimeError('SSH multiplex reuse unverified')
    write('ssh-multiplex-launchd-verification.json',{'evidence':evidence,'control_path':sock,'observed_unix':time.time(),'launcher_ppid':os.getppid()})
    env={**os.environ,'PYTHONPATH':c['local_code_root'],'PYTHONDONTWRITEBYTECODE':'1'}
    with (doc/'automation-staging/driver-launchd.log').open('x') as log:
        driver=subprocess.Popen([c['local_python'],'-B','-u','-m','fastfill.comparison.overnight','--config',str(config_path)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=c['local_code_root'],env=env);children.append(driver)
    wake=subprocess.Popen(['/usr/bin/caffeinate','-ims','-w',str(driver.pid)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);children.append(wake)
    write('automation-processes.json',{'launcher_pid':os.getpid(),'launcher_ppid':os.getppid(),'driver_pid':driver.pid,'master_pid':master.pid,'caffeinate_pid':wake.pid,'watchdog':'independent launcher SIGALRM','watchdog_seconds':c['total_timeout_seconds']})
    print('Automation active: driver='+str(driver.pid),flush=True)
    code=driver.wait()
finally:
    signal.alarm(0)
    for proc in reversed(children):
        try:stop(proc)
        except (ProcessLookupError,subprocess.TimeoutExpired):pass
    write('automation-launcher-end.json',{'driver_exit_code':driver.returncode if driver is not None else None,'finished_unix':time.time(),'all_children_stopped':all(x.poll() is not None for x in children)})
raise SystemExit(code)
