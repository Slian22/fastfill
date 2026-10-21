#!/bin/bash
cd /Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021
while pgrep -f wait_api.py >/dev/null; do sleep 20; done
grep -q "image probe ok" judge/wait.log && python3 scripts/judge_drive.py
