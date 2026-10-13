#!/bin/bash
#SBATCH --job-name=unknownsize-llm-agent
#SBATCH --account=brics.u6uv
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/runs/isambard/unknownsize-llm-agent-%j.out
# Usage: sbatch --time=00:30:00 isambard_llm_agent.sbatch CODE_ROOT smoke   (2 rooms, <=6 calls)
#        sbatch isambard_llm_agent.sbatch CODE_ROOT full                    (frozen cohort, <=4794 calls)
# Smoke + full stay within the amendment's 4,800-call budget. Resubmitting the same mode
# resumes from receipts; calls.jsonl keeps the cap across invocations and resubmissions.
# Full sizing: ~1,454 rooms x ~1.3 calls x ~20 s / 4 concurrent ~= 2.6 h; 3 calls/room ~= 6 h.
# Exit 2 (API failure, already retried per call) is retried here at most twice after 10 min.
set -euo pipefail
test "$#" -eq 2
PAIR_CODE_ROOT="$1"
case "$2" in
  # Smoke never finalizes API_FAILED (99 attempts > its 6-call cap) and exits 1 if any room is
  # API_FAILED, so a systemic 400/413/422 (an unsupported parameter) cannot pass afterok.
  smoke) OUT=llm-agent-smoke-001; MINUTES=25; EXTRA=(--limit 2 --max-calls 6 --max-attempts 99 --require-no-api-failed) ;;
  full) OUT=llm-agent-001; MINUTES=470; EXTRA=(--max-calls 4794) ;;
  *) exit 2 ;;
esac
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
PAIR_RUN_ROOT="$PROJECT_ROOT/runs/unknownsize-20261013"
COHORT="$PAIR_RUN_ROOT/llm-cohort-v1.json"
PYTHON="$PROJECT_ROOT/env/bin/python"
: "${SLURM_JOB_ID:?Slurm allocation required}"
export PYTHONPATH="$PAIR_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export FASTFILL_API_ENV=/home/u6uv/shanliantian.u6uv/.fastfill_api.env
unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL
cd "$PAIR_CODE_ROOT"
test -e "$COHORT" || "$PYTHON" -B -m fastfill.comparison.llm_agent cohort \
  --data "$PAIR_RUN_ROOT/eval-data-v1" --out "$COHORT"
END=$((SECONDS + MINUTES * 60))
for invocation in 1 2 3; do
  LEFT=$(( (END - SECONDS) / 60 ))
  test "$LEFT" -ge 5 || exit 124
  rc=0
  timeout --signal=TERM --kill-after=30s "${LEFT}m" "$PYTHON" -B -u \
    -m fastfill.comparison.llm_agent run --data "$PAIR_RUN_ROOT/eval-data-v1" --cohort "$COHORT" \
    --out "$PAIR_RUN_ROOT/$OUT" --env-file "$FASTFILL_API_ENV" --concurrency 4 "${EXTRA[@]}" || rc=$?
  test "$rc" -eq 2 || exit "$rc"
  test "$invocation" -eq 3 || sleep 600
done
exit 2
