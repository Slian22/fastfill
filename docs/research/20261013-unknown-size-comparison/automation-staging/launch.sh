#!/bin/bash
set -u
umask 077
driver_pid=''
master_pid=''
cleanup() {
  if [[ -n "$driver_pid" ]] && kill -0 "$driver_pid" 2>/dev/null; then
    kill -TERM "$driver_pid" 2>/dev/null || true
    wait "$driver_pid" 2>/dev/null || true
  fi
  /usr/bin/ssh -S /tmp/ff-u6uv-44da3imw/control.sock -O exit u6uv.aip2.isambard >/dev/null 2>&1 || true
  if [[ -n "$master_pid" ]]; then
    kill -TERM "$master_pid" 2>/dev/null || true
    wait "$master_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o ForwardAgent=no -o ForwardX11=no -o ControlPersist=no -M -N -n -S /tmp/ff-u6uv-44da3imw/control.sock u6uv.aip2.isambard >/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/automation-staging/ssh-master.log 2>&1 &
master_pid=$!
ready=0
for attempt in {1..60}; do
  if /usr/bin/ssh -S /tmp/ff-u6uv-44da3imw/control.sock -O check u6uv.aip2.isambard >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$master_pid" 2>/dev/null; then break; fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then printf 'SSH master readiness failed\n' >&2; exit 2; fi
export PYTHONPATH=/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/automation-code/pipeline-9c4d97dbd736
export PYTHONDONTWRITEBYTECODE=1
cd /Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/automation-code/pipeline-9c4d97dbd736
/usr/local/bin/python3 -B -u -m fastfill.comparison.overnight --config /Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/automation-config.json >/Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/automation-staging/driver.log 2>&1 &
driver_pid=$!
/usr/bin/caffeinate -ims -w "$driver_pid" &
wake_pid=$!
printf '{"launcher_pid":%s,"driver_pid":%s,"master_pid":%s,"caffeinate_pid":%s}\n' "$$" "$driver_pid" "$master_pid" "$wake_pid" > /Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/evidence/automation-processes.json.tmp
mv /Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/evidence/automation-processes.json.tmp /Users/slian/Desktop/3D/Worldedge/OptiScene/docs/research/20261013-unknown-size-comparison/evidence/automation-processes.json
printf 'Automation active: driver=%s master=%s wake=%s\n' "$driver_pid" "$master_pid" "$wake_pid"
wait "$driver_pid"
rc=$?
printf 'Driver exited: %s\n' "$rc"
exit "$rc"
