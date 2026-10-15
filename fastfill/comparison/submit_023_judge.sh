#!/bin/bash
# Amendment 023: rerun the run-002 GPT-judge bundle with judge gpt-6-astra at reasoning effort high.
#   submit_023_judge.sh CODE_ROOT smoke   -> client alone on the first 2 bundle cases (RUN/judge-023-astra-high/smoke)
#   submit_023_judge.sh CODE_ROOT full    -> judge_batch on all 696 selected / 692 runnable cases, 4 workers
#                                            (RUN/judge-023-astra-high/results); only after the smoke passed.
# One CPU Slurm job each; the API key stays in the env file and is never printed. Write-once outputs; nothing is
# submitted when the output exists or an unknownsize-023 job is already in Slurm.
set -euo pipefail
USAGE="usage: submit_023_judge.sh CODE_ROOT smoke|full"
test "$#" -eq 2 || { echo "$USAGE" >&2; exit 2; }
CODE_ROOT="$1"; STAGE="$2"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
ENV_FILE=/home/u6uv/shanliantian.u6uv/.fastfill_api.env
BUNDLE="$RUN/gpt-863099194937c5259a5aa8ac269f7a86f5b40cf752eabb9b789743d655ebcbb3/bundle"
CLIENT="$CODE_ROOT/fastfill/comparison/server_api_judge.py"
BASE="$RUN/judge-023-astra-high"
OPTS="--model gpt-6-astra --reasoning-effort high"
test -f "$CLIENT" && test -f "$BUNDLE/api-bundle/manifest.json" || { echo "missing client or bundle" >&2; exit 1; }
test -z "$(squeue -h -u "$USER" -n unknownsize-023 -o %i)" || { echo "unknownsize-023 already in Slurm" >&2; exit 1; }
mkdir -p "$BASE" "$PROJECT_ROOT/runs/isambard"
case "$STAGE" in
smoke)
  test ! -e "$BASE/smoke" || { echo "exists: $BASE/smoke" >&2; exit 1; }
  mkdir "$BASE/smoke"
  # First 2 cases with absolute artifact paths; expected_model is the amendment-023 judge.
  "$PYTHON" -c 'import json, sys; from pathlib import Path
root = Path(sys.argv[1]); m = json.loads((root / "manifest.json").read_text())
cases = [{**c, "prompt_file": str(root / c["prompt_file"]), "images": [{**i, "path": str(root / i["path"])} for i in c["images"]]} for c in m["cases"][:2]]
Path(sys.argv[2]).write_text(json.dumps({"schema_version": 1, "expected_model": "gpt-6-astra", "cases": cases}))' \
    "$BUNDLE/api-bundle" "$BASE/smoke/manifest.json"
  CMD="'$PYTHON' -B '$CLIENT' --manifest '$BASE/smoke/manifest.json' --results-dir '$BASE/smoke/results' --env-file '$ENV_FILE' --limit 2 --timeout 600 $OPTS"
  TIME=00:30:00 ;;
full)
  test -e "$BASE/smoke/results" || { echo "run the smoke first" >&2; exit 1; }
  test ! -e "$BASE/results" || { echo "exists: $BASE/results" >&2; exit 1; }
  CMD="cd '$CODE_ROOT' && PYTHONPATH='$CODE_ROOT' FASTFILL_API_ENV='$ENV_FILE' '$PYTHON' -B -u '$CODE_ROOT/fastfill/comparison/judge_batch.py' --bundle '$BUNDLE' --output '$BASE/results' --client '$CLIENT' --max-cases 800 --arms J,P,L,O6 --timeout 600 --expected-total 696 $OPTS --workers 4"
  TIME=12:00:00 ;;
*) echo "$USAGE" >&2; exit 2 ;;
esac
sbatch --parsable --no-requeue --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=8G --gres=none --time="$TIME" \
  --account=brics.u6uv --partition=workq --qos=normal --job-name=unknownsize-023 \
  --output="$PROJECT_ROOT/runs/isambard/unknownsize-023-$STAGE-%j.out" \
  --wrap="$CMD; rc=\$?; echo \"{\\\"exit_code\\\": \$rc}\" > '$BASE/$STAGE.completion.json'; exit \$rc"
