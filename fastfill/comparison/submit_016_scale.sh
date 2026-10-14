#!/bin/bash
# Amendment 016 §1 and §6 (EXPLORATORY, descriptive; no result or decision of amendments 005-015 changes), run by the
# user on the Isambard login node:
#   submit_016_scale.sh CODE_ROOT [OUT_NAME]
# One CPU job (72 cores, 200G, 8 h, no GPU): python -m fastfill.comparison.scale_curve --workers 70 --repeats 2000 on
#   GT, J = SFT (geometry-v5 joint rows, cap rule from infer-joint-002) and P = rpo-dpo1 (geometry-v5 placement rows,
#   infer-placement-005) on every test room; every agent NAME in L O1a O1b O2 O3 O4 O6 O7
#   (RUN/llm-eval-NAME-selected-v1, harness and raw) on RUN/llm-cohort-v1.json; the amendment-012 S1 outputs of all of
#   them (RUN/s1-012-test/{joint-sft,placement-rpo-dpo1,agent-NAME-V}) for the rule-level table after S1; and, when
#   amendment 014's scorer is published (RUN/optiscene-014/score), the reproduced OptiScene stages on their E2 rooms:
#   OS-STAGE-MODE = e2-rows.jsonl::arm=STAGE-e2-sft,mode=MODE for STAGE in sft dpo1 dpo2, MODE in prior gt (the
#   released 8B arm 8b-e2-v1 is not a reproduced stage and is left out)
#   -> RUN/OUT_NAME (default scale-curve-016; write-once). Without the scorer the job runs without OptiScene (printed);
#   once it is published, submit again with a new OUT_NAME (e.g. scale-curve-016-os) to include it.
#   Log: PROJECT_ROOT/runs/isambard/unknownsize-016-scale-JOBID.out; the matplotlib cache goes to
#   PROJECT_ROOT/.cache-jobs/unknownsize-016-scale.
# Nothing is submitted on a missing input, an existing output, a CODE_ROOT that cannot import scale_curve, or an
# unknownsize-016-scale job already in Slurm.
set -euo pipefail
USAGE="usage: submit_016_scale.sh CODE_ROOT [OUT_NAME]"
test "$#" -ge 1 && test "$#" -le 2 || { echo "$USAGE" >&2; exit 2; }
CODE_ROOT="$1"
OUT_NAME="${2:-scale-curve-016}"
case "$OUT_NAME" in ''|.*|*[!A-Za-z0-9._-]*) echo "$USAGE (OUT_NAME: letters, digits, . _ -)" >&2; exit 2 ;; esac
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
OUT="$RUN/$OUT_NAME"
S1="$RUN/s1-012-test"
SCORE="$RUN/optiscene-014/score"
AGENTS=(L O1a O1b O2 O3 O4 O6 O7)
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
test -f "$CODE_ROOT/fastfill/comparison/scale_curve.py" || fail "missing $CODE_ROOT/fastfill/comparison/scale_curve.py"
INPUTS=(eval-data-v1/MANIFEST.json geometry-v5/MANIFEST.json infer-joint-002/MANIFEST.json
        infer-placement-005/MANIFEST.json llm-cohort-v1.json s1-012-test/joint-sft/MANIFEST.json
        s1-012-test/placement-rpo-dpo1/MANIFEST.json)
FLAGS=""
S1FLAGS=" --s1 J=$S1/joint-sft --s1 P=$S1/placement-rpo-dpo1"
for NAME in "${AGENTS[@]}"; do
  INPUTS+=("llm-eval-$NAME-selected-v1/MANIFEST.json")
  FLAGS="$FLAGS --agent $NAME=$RUN/llm-eval-$NAME-selected-v1"
  for V in harness raw; do
    INPUTS+=("s1-012-test/agent-$NAME-$V/MANIFEST.json")
    S1FLAGS="$S1FLAGS --s1 $NAME-$V=$S1/agent-$NAME-$V"
  done
done
for f in "${INPUTS[@]}"; do test -f "$RUN/$f" || fail "missing $RUN/$f"; done
if test -f "$SCORE/MANIFEST.json"; then
  test -f "$SCORE/e2-rows.jsonl" || fail "missing $SCORE/e2-rows.jsonl"
  for STAGE in sft dpo1 dpo2; do
    for MODE in prior gt; do
      FLAGS="$FLAGS --extra OS-$STAGE-$MODE=$SCORE/e2-rows.jsonl::arm=$STAGE-e2-sft,mode=$MODE"
    done
  done
else
  echo "amendment 014 scorer not published ($SCORE): no OptiScene extras (submit again with a new OUT_NAME later)"
fi
test ! -e "$OUT" || fail "$OUT exists"
test -x "$PYTHON" || fail "no executable $PYTHON"
(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -c 'import fastfill.comparison.scale_curve') \
  || fail "$CODE_ROOT cannot import fastfill.comparison.scale_curve"
QUEUED=$(squeue -h -u "$USER" -o '%i %j') || fail "squeue failed"
QUEUED=$(printf '%s\n' "$QUEUED" | awk '$2 ~ /^unknownsize-016-scale/ {print $1}')
test -z "$QUEUED" || fail "unknownsize-016-scale job already in Slurm (scancel $(echo $QUEUED))"
CMD="cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
CMD="$CMD MPLCONFIGDIR=$PROJECT_ROOT/.cache-jobs/unknownsize-016-scale/mpl"
CMD="$CMD $PYTHON -B -m fastfill.comparison.scale_curve --data $RUN/eval-data-v1 --geometry $RUN/geometry-v5"
CMD="$CMD --cohort $RUN/llm-cohort-v1.json --joint-run $RUN/infer-joint-002 --placement-run $RUN/infer-placement-005"
CMD="$CMD$FLAGS$S1FLAGS --out $OUT --workers 70 --repeats 2000"
ID=$(sbatch --parsable --account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=72 \
  --mem=200G --gres=none --no-requeue --time=08:00:00 --job-name=unknownsize-016-scale \
  --output="$PROJECT_ROOT/runs/isambard/unknownsize-016-scale-%j.out" --wrap "$CMD") || ID=""
ID="${ID%%;*}"
case "$ID" in ''|*[!0-9]*) fail "sbatch failed" ;; esac
echo "unknownsize-016-scale $ID"
