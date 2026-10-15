#!/bin/bash
# Amendment 025: the gpt-6.1-sol critic (reasoning medium) on one critic_025 state directory (BUNDLE/critic-manifest.json),
# as one CPU Slurm job. The output is fixed by the bundle: OUT = BUNDLE.smoke or BUNDLE.critic (BUNDLE resolved), and
# OUT/answers is then the round's --answers for `critic_025 round`.
#   submit_025.sh CODE_ROOT smoke  BUNDLE            -> first 2 cases only (--limit 2 --cap 2), 1 worker
#   submit_025.sh CODE_ROOT critic BUNDLE [--cap N]  -> every case, refused above N (default 500), 4 workers
# critic_batch_025 makes at most one call per case: no retries, no replacement, and one critic run per bundle. The API
# key stays in the env file and is never printed. Write-once: nothing is submitted when OUT exists or an
# unknownsize-025 job is in Slurm.
set -euo pipefail
USAGE="usage: submit_025.sh CODE_ROOT smoke|critic BUNDLE [--cap N]"
test "$#" -eq 3 || test "$#" -eq 5 || { echo "$USAGE" >&2; exit 2; }
CODE_ROOT="$1"; STAGE="$2"; CAP=500
if test "$#" -eq 5; then
  test "$4" = --cap && [[ "$5" =~ ^[1-9][0-9]*$ ]] || { echo "$USAGE" >&2; exit 2; }
  CAP="$5"
fi
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PYTHON="$PROJECT_ROOT/env/bin/python"
ENV_FILE=/home/u6uv/shanliantian.u6uv/.fastfill_api.env
test -f "$CODE_ROOT/fastfill/comparison/critic_batch_025.py" && test -f "$3/critic-manifest.json" \
  || { echo "missing critic_batch_025.py or bundle manifest" >&2; exit 1; }
BUNDLE="$(cd "$3" && pwd -P)"
case "$STAGE" in
smoke)  OPTS="--limit 2 --cap 2 --workers 1"; TIME=00:30:00 ;;
critic) OPTS="--cap $CAP --workers 4"; TIME=12:00:00 ;;
*) echo "$USAGE" >&2; exit 2 ;;
esac
OUT="$BUNDLE.$STAGE"
test ! -e "$OUT" && test ! -e "$OUT.completion.json" || { echo "exists: $OUT" >&2; exit 1; }
test -z "$(squeue -h -u "$USER" -n unknownsize-025 -o %i)" || { echo "unknownsize-025 already in Slurm" >&2; exit 1; }
mkdir -p "$PROJECT_ROOT/runs/isambard"
CMD="cd '$CODE_ROOT' && '$PYTHON' -B -u -m fastfill.comparison.critic_batch_025 --bundle '$BUNDLE' --output '$OUT' --env-file '$ENV_FILE' --timeout 600 $OPTS"
sbatch --parsable --no-requeue --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=8G --gres=none --time="$TIME" \
  --account=brics.u6uv --partition=workq --qos=normal --job-name=unknownsize-025 \
  --output="$PROJECT_ROOT/runs/isambard/unknownsize-025-$STAGE-%j.out" \
  --wrap="$CMD; rc=\$?; echo \"{\\\"exit_code\\\": \$rc}\" > '$OUT.completion.json'; exit \$rc"
