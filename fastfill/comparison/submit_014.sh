#!/bin/bash
# Amendment 014 chain (exploratory; "OptiScene code recipe (README, Qwen2.5-7B-Instruct)"), run by the user on the
# Isambard login node after setup_optiscene_env.sh, the dataset rsync and the plan-e1 upload:  submit_014.sh CODE_ROOT
# Preflight, nothing submitted on failure: a passing env.json with the model hashes, the three data files' sha256
# prefixes, the uploaded plan-e1 (MANIFEST + requests hash + built by this CODE_ROOT's optiscene_eval.py, prompt.py
# and UPSTREAM.json), eval-data-v1 / dev-select-v1 / geometry-v5 / dev-eval-010/selection.json / the 8B model, the
# training venv and vLLM pythons, no optiscene-014 job in Slurm, nothing under RUN/optiscene-014.
# Jobs (each --kill-on-invalid-dep=yes; the roots are held until the whole chain is in, then released):
#   plan-e2                                      CPU 16 / 64G, 2 h      root
#   sft                                          1 GPU, 6 h             root
#   merge-sft       afterok sft                  CPU 16 / 64G, 2 h
#   dpo1-s1         afterok merge-sft            1 GPU, 24 h
#   dpo1-s2         afterany dpo1-s1             1 GPU, 24 h (train refuses unless s1 ended COMPLETED or TIMEOUT)
#   merge-dpo1      afterany dpo1-s2             CPU (fails closed on an incomplete stage: report it, new amendment)
#   dpo2-s1 / dpo2-s2 / merge-dpo2               the same after merge-dpo1
#   gen-sft|dpo1|dpo2  afterok merge-X:plan-e2   1 GPU, 2 h
#   gen-8b          afterok plan-e2              1 GPU, 2 h
#   score           afterok every gen            CPU 16 / 64G, 4 h
# GPU cap: 6 + 4 x 24 + 4 x 2 = 110 GPU-h (expected 50-80). On any sbatch failure: ABORT with the ids to scancel.
set -euo pipefail
test "$#" -eq 1 || exit 2
CODE_ROOT="$1"
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
OUT="$RUN/optiscene-014"
S="$CODE_ROOT/fastfill/comparison"
VENV_PY="$PROJECT_ROOT/envs/optiscene-014/bin/python"
VLLM_PY=/home/u6uv/shanliantian.u6uv/miniforge3/envs/vllm/bin/python
B="$S/isambard_014.sbatch"
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
for f in optiscene_eval.py isambard_014.sbatch optiscene_official/main.py optiscene_official/UPSTREAM.json; do
  test -f "$S/$f" || fail "missing $S/$f"
done
for d in eval-data-v1/MANIFEST.json dev-select-v1/MANIFEST.json geometry-v5/MANIFEST.json dev-eval-010/selection.json; do
  test -f "$RUN/$d" || fail "missing $RUN/$d"
done
test -f "$PROJECT_ROOT/models/OptiScene-Qwen-8B-Final/config.json" || fail "missing models/OptiScene-Qwen-8B-Final"
test ! -e "$OUT" || fail "$OUT exists"
for py in "$VENV_PY" "$VLLM_PY"; do test -x "$py" || fail "no executable $py"; done
"$PROJECT_ROOT/env/bin/python" - "$PROJECT_ROOT" "$S" <<'PY' || fail "preflight check failed (see above)"
import hashlib, json, sys
root = sys.argv[1]
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()
env = json.load(open(f'{root}/envs/optiscene-014.env.json'))
assert env.get('ok') is True and (env.get('model') or {}).get('files'), 'env.json: no passing env-check with model hashes'
for name, prefix in (('sft_prompts.json', '12b3ccc10e8e'), ('dpo_prompts.json', '0d331989'),
                     ('dpo2_prompts.json', 'ed6c1bae')):
    digest = sha(f'{root}/data/optiscene-official/{name}')
    assert digest.startswith(prefix), f'{name}: sha256 {digest[:12]} is not {prefix}'
plan = f'{root}/data/optiscene-014-e1'
manifest = json.load(open(f'{plan}/MANIFEST.json'))
assert manifest.get('plan') == 'e1' and manifest['files']['requests.jsonl'] == sha(f'{plan}/requests.jsonl'), 'plan-e1'
s = sys.argv[2]
code = {'optiscene_eval.py': f'{s}/optiscene_eval.py', 'prompt.py': f'{s}/optiscene_official/prompt.py',
        'UPSTREAM.json': f'{s}/optiscene_official/UPSTREAM.json'}
assert manifest.get('implementation_sha256') == {k: sha(v) for k, v in code.items()}, \
    'plan-e1 was built by other code: rebuild it with this CODE_ROOT and upload it again'
PY
QUEUED=$(squeue -h -u "$USER" -o '%i %j') || fail "squeue failed"
QUEUED=$(printf '%s\n' "$QUEUED" | awk '$2 ~ /^optiscene-014/ {print $1}')
test -z "$QUEUED" || fail "optiscene-014 jobs already in Slurm (scancel $(echo $QUEUED))"
SUBMITTED=""
ROOTS=""
submit() {  # submit LABEL DEPENDENCY SBATCH_ARGS...; sets ID; no dependency = a held root. set -e misses $(...).
  local label="$1" dep="$2" out=""
  shift 2
  if test -n "$dep"; then
    out=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="$dep" --job-name="optiscene-014-$label" "$@") || out=""
  else
    out=$(sbatch --parsable --hold --job-name="optiscene-014-$label" "$@") || out=""
  fi
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$label'; chain NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  if test -z "$dep"; then ROOTS="$ROOTS${ROOTS:+,}$ID"; fi
  echo "$label $ID"
}
CPU=(--cpus-per-task=16 --mem=64G)
GPU=(--gpus-per-node=1 --cpus-per-task=16 --mem=110G)
submit plan-e2 "" "${CPU[@]}" --time=02:00:00 "$B" "$CODE_ROOT" plan-e2
PLAN="$ID"
submit sft "" "${GPU[@]}" --time=06:00:00 "$B" "$CODE_ROOT" train sft -
submit merge-sft "afterok:$ID" "${CPU[@]}" --time=02:00:00 "$B" "$CODE_ROOT" merge sft
MERGE="$ID"
submit gen-sft "afterok:$MERGE:$PLAN" "${GPU[@]}" --time=02:00:00 "$B" "$CODE_ROOT" gen sft
GENS="$ID"
for STAGE in dpo1 dpo2; do
  submit "$STAGE-s1" "afterok:$MERGE" "${GPU[@]}" --time=24:00:00 "$B" "$CODE_ROOT" train "$STAGE" -
  FIRST="$ID"
  submit "$STAGE-s2" "afterany:$FIRST" "${GPU[@]}" --time=24:00:00 "$B" "$CODE_ROOT" train "$STAGE" "$FIRST"
  submit "merge-$STAGE" "afterany:$ID" "${CPU[@]}" --time=02:00:00 "$B" "$CODE_ROOT" merge "$STAGE"
  MERGE="$ID"
  submit "gen-$STAGE" "afterok:$MERGE:$PLAN" "${GPU[@]}" --time=02:00:00 "$B" "$CODE_ROOT" gen "$STAGE"
  GENS="$GENS:$ID"
done
submit gen-8b "afterok:$PLAN" "${GPU[@]}" --time=02:00:00 "$B" "$CODE_ROOT" gen 8b
GENS="$GENS:$ID"
submit score "afterok:$GENS" "${CPU[@]}" --time=04:00:00 "$B" "$CODE_ROOT" score
scontrol release "$ROOTS" || { echo "ABORT: could not release $ROOTS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOTS"
