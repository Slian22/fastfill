#!/bin/bash
# Amendment 016 section 2 (size-fixed regeneration, EXPLORATORY) with addenda A1/A2/A6/B1/B2: the agents re-run on the
# frozen cohort with P's input (llm_open / llm_agent --arm placement), pilot first, scored as P and repaired by S1.
# Run by the user on the Isambard login node:
#   submit_016_agents.sh CODE_ROOT [SYSTEM ...]   (default: O2 O6 O4 O7 L; only these have a GPU budget below)
# First (login node, nothing submitted yet): agents_016 pilot writes or verifies RUN/a016-pilot-v1.json (300 cohort
# rooms, stable hash 'a016-pilot:', stratified by object-count bin; L's subsample too, B1).
# Per open system SYS (GPU jobs carry the sbatch's job name unknownsize-llm-open + 'singleton': one at a time):
#   pilot      RUN/llm-open-SYS-P-pilot-001       held root; smoke-gated (it replaces the smoke: same gate, 300 rooms)
#   pilot-eval RUN/llm-eval-SYS-P-pilot-v1        afterok pilot: evaluate_llm --uids pilot, J/P restricted to it
#   pilot-s1   RUN/s1-016-size-fixed/agent-SYS-P-pilot-{harness,raw}   afterok pilot-eval
#   full       RUN/llm-open-SYS-P-001             afterok own pilot + afterany every other pilot (A6: pilots first)
#   eval       RUN/llm-eval-SYS-P-v1              afterok full; J = RUN/infer-joint-002 (SFT), P = RUN/infer-placement-005
#   s1         RUN/s1-016-size-fixed/agent-SYS-P-{harness,raw}         afterok eval (--uids cohort)
# O2 also, A1 supplementary (submitted last): gt-pilot RUN/llm-open-O2-PGT-pilot-001 (GT sizes; afterok O2 pilot)
#   -> RUN/llm-eval-O2-PGT-pilot-v1 -> RUN/s1-016-size-fixed/agent-O2-PGT-pilot-{harness,raw}.
# L (A2/B1; CPU, API, <= 1,200 calls): isambard_llm_agent.sbatch pilot-placement -> RUN/llm-agent-P-pilot-001 (held
#   root) -> RUN/llm-eval-L-P-pilot-v1 -> RUN/s1-016-size-fixed/agent-L-P-pilot-{harness,raw}.
# Evals: CPU 16 / 64G / 4 h. S1: output_repair at the amendment-012 settings (--requests-arm placement, max_trials
# 2000, time_limit 60, --workers 70; CPU 72 / 200G / 4 h). Every dependent job is --kill-on-invalid-dep=yes.
# GPU cap 40 GPU-h (section 2): --time per job from LIMITS below replaces llm_open's limit; the script refuses when the
# ceiling (sum of time limits x GPUs, pilots + fulls + GT pilot) exceeds 40 GPU-h. Defaults: 33.5 GPU-h ceiling;
# the J-input full runs measured 8.7 GPU-h of process wall time (report-v2-selected-001 T5: O2 4.52, O6 1.97, O7 1.62,
# O4 0.55, i.e. 2.26 / 0.49 / 0.81 / 0.27 h wall). Calibrate against their sacct first if possible. A run that hits
# its limit resumes from its receipts under the same identity when resubmitted (count that against the cap).
# Comparison (CPU, by the parent once the S1 outputs exist; amendment-012 S1 inputs under RUN/s1-012-test), e.g.
#   agents_016 compare --data RUN/eval-data-v1 --cohort RUN/llm-cohort-v1.json --uids RUN/a016-pilot-v1.json \
#     --s1-012 RUN/s1-012-test --fixed O2=RUN/s1-016-size-fixed/agent-O2-P-pilot ... --fixed L=.../agent-L-P-pilot \
#     --extra O2-PGT=.../agent-O2-PGT-pilot --out RUN/compare-016-pilot-v1        (pilot reading, labelled as such)
#   then on the full runs: --uids RUN/llm-cohort-v1.json without L, and --uids pilot with L (B1), [--optiscene E2 S1].
# Pilots (and L) are held roots, released once the whole chain is in; any failed sbatch aborts and lists the ids for
# scancel (held, so nothing ran). Refuses (nothing submitted) when any output above exists, an input is missing,
# CODE_ROOT lacks the amendment-016 code, or an unknownsize-llm-open / -llm-agent / -016-agents job is in Slurm.
set -euo pipefail
test "$#" -ge 1 || { echo "usage: submit_016_agents.sh CODE_ROOT [SYSTEM ...]" >&2; exit 2; }
CODE_ROOT="$1"
shift
PROJECT_ROOT=/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill
RUN="$PROJECT_ROOT/runs/unknownsize-20261013"
PYTHON="$PROJECT_ROOT/env/bin/python"
S="$CODE_ROOT/fastfill/comparison"
JOINT="$RUN/infer-joint-002"
PLACEMENT="$RUN/infer-placement-005"
PILOT="$RUN/a016-pilot-v1.json"
COHORT="$RUN/llm-cohort-v1.json"
S1="$RUN/s1-016-size-fixed"
CAP_MINUTES=2400  # 40 GPU-h
GT_MINUTES=90
limits() {  # limits SYSTEM -> "PILOT_MINUTES FULL_MINUTES" (bash 3 on the macOS test host: no associative arrays)
  case "$1" in O2) echo "90 300" ;; O6) echo "45 90" ;; O4) echo "30 60" ;; O7) echo "45 120" ;; *) return 1 ;; esac
}
hhmm() { printf '%02d:%02d:00' $(($1 / 60)) $(($1 % 60)); }
SYSTEMS=("$@")
if test "${#SYSTEMS[@]}" -eq 0; then SYSTEMS=(O2 O6 O4 O7 L); fi
fail() { echo "ABORT: $*; nothing submitted" >&2; exit 1; }
SEEN=" "
OUTS=()
for SYSTEM in "${SYSTEMS[@]}"; do
  case "$SYSTEM" in L) ;; *) limits "$SYSTEM" > /dev/null || { echo "no amendment-016 budget for $SYSTEM" >&2; exit 2; } ;; esac
  case "$SEEN" in *" $SYSTEM "*) echo "duplicate system $SYSTEM" >&2; exit 2 ;; esac
  SEEN="$SEEN$SYSTEM "
  case "$SYSTEM" in L) OUTS+=(llm-agent-P-pilot-001) ;; *) OUTS+=("llm-open-$SYSTEM-P-pilot-001" "llm-open-$SYSTEM-P-001"
    "llm-eval-$SYSTEM-P-v1" "s1-016-size-fixed/agent-$SYSTEM-P-harness" "s1-016-size-fixed/agent-$SYSTEM-P-raw") ;; esac
  OUTS+=("llm-eval-$SYSTEM-P-pilot-v1" "s1-016-size-fixed/agent-$SYSTEM-P-pilot-harness"
         "s1-016-size-fixed/agent-$SYSTEM-P-pilot-raw")
done
case "$SEEN" in *" O2 "*) OUTS+=(llm-open-O2-PGT-pilot-001 llm-eval-O2-PGT-pilot-v1
  s1-016-size-fixed/agent-O2-PGT-pilot-harness s1-016-size-fixed/agent-O2-PGT-pilot-raw) ;; esac
for out in "${OUTS[@]}"; do test ! -e "$RUN/$out" || fail "$RUN/$out exists"; done
for f in llm_open.py llm_agent.py evaluate_llm.py agents_016.py output_repair.py isambard_llm_open.sbatch \
         isambard_llm_agent.sbatch; do
  test -f "$S/$f" || fail "missing $S/$f"
done
grep -q 'pilot) OUT=' "$S/isambard_llm_open.sbatch" || fail "$S/isambard_llm_open.sbatch has no pilot mode"
grep -q 'pilot-placement)' "$S/isambard_llm_agent.sbatch" || fail "$S/isambard_llm_agent.sbatch has no pilot-placement"
# An older evaluate_llm would score P runs with joint rules; an older llm_agent has no addendum-B2 size enforcement.
grep -q request_arm "$S/evaluate_llm.py" && grep -q enforce_sizes "$S/evaluate_llm.py" \
  || fail "$S/evaluate_llm.py cannot score size-fixed runs (request_arm + enforce_sizes)"
for f in eval-data-v1/MANIFEST.json eval-data-v1/placement/requests.jsonl llm-cohort-v1.json \
         infer-joint-002/MANIFEST.json infer-placement-005/MANIFEST.json; do
  test -f "$RUN/$f" || fail "missing $RUN/$f"
done
# An older chain still in Slurm shares the singleton name (or the API budget) and would run first.
QUEUED=$(squeue -h -u "$USER" -o '%i %j') || fail "squeue failed"
QUEUED=$(printf '%s\n' "$QUEUED" | awk '$2 ~ /^unknownsize-(llm-open|llm-agent|016-agents)/ {print $1}')
test -z "$QUEUED" || fail "unknownsize open-model / API-agent / 016-agents jobs already in Slurm (scancel $(echo $QUEUED))"
flags() {  # flags SYSTEM: llm_open's full-run resources for SYSTEM, without its --time (LIMITS set it).
  local f=""
  f=$(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -m fastfill.comparison.llm_open slurm --system "$1" --mode full) \
    || return 1
  echo "${f% --time=*}"
}
PILOT_FLAGS=()  # Parallel to SYSTEMS.
FULL_FLAGS=()
CEILING=0
for SYSTEM in "${SYSTEMS[@]}"; do
  if test "$SYSTEM" = L; then PILOT_FLAGS+=(""); FULL_FLAGS+=(""); continue; fi
  F=$(flags "$SYSTEM") || fail "flags for $SYSTEM"
  read -r PM FM <<< "$(limits "$SYSTEM")"
  GPUS="${F#*--gpus-per-node=}"
  GPUS="${GPUS%% *}"
  CEILING=$((CEILING + (PM + FM) * GPUS))
  if test "$SYSTEM" = O2; then CEILING=$((CEILING + GT_MINUTES * GPUS)); GT_FLAGS="$F --time=$(hhmm "$GT_MINUTES")"; fi
  PILOT_FLAGS+=("$F --time=$(hhmm "$PM")")
  FULL_FLAGS+=("$F --time=$(hhmm "$FM")")
done
test "$CEILING" -le "$CAP_MINUTES" || fail "GPU ceiling $CEILING GPU-minutes exceeds the 40 GPU-h cap"
(cd "$CODE_ROOT" && PYTHONPATH="$CODE_ROOT" "$PYTHON" -B -m fastfill.comparison.agents_016 pilot \
  --data "$RUN/eval-data-v1" --cohort "$COHORT" --out "$PILOT") || fail "agents_016 pilot (RUN/a016-pilot-v1.json)"
SUBMITTED=""
ROOTS=""
submit() {  # submit LABEL SBATCH_ARGS...; sets ID. set -e does not reach $(...), so check here.
  local out=""
  out=$(sbatch --parsable "${@:2}") || out=""
  ID="${out%%;*}"
  case "$ID" in
    ''|*[!0-9]*)
      echo "ABORT: sbatch failed for '$1'; chain NOT complete. Already submitted: ${SUBMITTED:-none}" >&2
      if test -n "$SUBMITTED"; then echo "cancel with: scancel $SUBMITTED" >&2; fi
      exit 1 ;;
  esac
  SUBMITTED="$SUBMITTED${SUBMITTED:+ }$ID"
  echo "$1 $ID"
}
B="$S/isambard_llm_open.sbatch"
ENV="cd $CODE_ROOT && PYTHONPATH=$CODE_ROOT PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
LOGS="$PROJECT_ROOT/runs/isambard"
CPU=(--kill-on-invalid-dep=yes --account=brics.u6uv --partition=workq --qos=normal --nodes=1 --ntasks=1 --gres=none
     --no-requeue --time=04:00:00)
score() {  # score NAME LLM_DIR TAG UIDS AFTER: evaluate_llm (--uids on a pilot), then S1 on both variants.
  local eval_dir="$RUN/llm-eval-$1-$3-v1" select=""
  test "$4" = "$COHORT" || select=" --uids $4"
  submit "$1 $3 eval" "${CPU[@]}" --dependency="afterok:$5" --cpus-per-task=16 --mem=64G \
    --job-name="unknownsize-016-agents-$1-$3-eval" --output="$LOGS/unknownsize-016-agents-$1-$3-eval-%j.out" \
    --wrap "$ENV $PYTHON -B -m fastfill.comparison.evaluate_llm --data $RUN/eval-data-v1 --cohort $COHORT --llm $2 --joint $JOINT --placement $PLACEMENT --out $eval_dir --workers 16$select"
  local scored="$ID"
  for V in harness raw; do
    submit "$1 $3 s1-$V" "${CPU[@]}" --dependency="afterok:$scored" --cpus-per-task=72 --mem=200G \
      --job-name="unknownsize-016-agents-$1-$3-s1" --output="$LOGS/unknownsize-016-agents-$1-$3-s1-$V-%j.out" \
      --wrap "$ENV $PYTHON -B -m fastfill.comparison.output_repair --data $RUN/eval-data-v1 --workers 70 --max-trials 2000 --time-limit 60 --geometry $eval_dir/llm_$V-geometry.jsonl --requests-arm placement --uids $4 --out $S1/agent-$1-$3-$V"
  done
}
PILOTS=()  # Parallel to SYSTEMS ('' for L).
for k in "${!SYSTEMS[@]}"; do
  SYSTEM="${SYSTEMS[$k]}"
  if test "$SYSTEM" = L; then
    submit "L pilot" --hold --time=04:00:00 "$S/isambard_llm_agent.sbatch" "$CODE_ROOT" pilot-placement
    ROOTS="$ROOTS${ROOTS:+,}$ID"
    PILOTS+=("")
    score L "$RUN/llm-agent-P-pilot-001" P-pilot "$PILOT" "$ID"
    continue
  fi
  # shellcheck disable=SC2086  # the flags are space-separated sbatch options
  submit "$SYSTEM pilot" --hold --dependency=singleton ${PILOT_FLAGS[$k]} "$B" "$CODE_ROOT" "$SYSTEM" pilot placement
  ROOTS="$ROOTS${ROOTS:+,}$ID"
  PILOTS+=("$ID")
  score "$SYSTEM" "$RUN/llm-open-$SYSTEM-P-pilot-001" P-pilot "$PILOT" "$ID"
done
for k in "${!SYSTEMS[@]}"; do
  SYSTEM="${SYSTEMS[$k]}"
  test "$SYSTEM" != L || continue
  OTHERS=""
  for j in "${!PILOTS[@]}"; do
    if test "$j" != "$k" -a -n "${PILOTS[$j]}"; then OTHERS="$OTHERS:${PILOTS[$j]}"; fi
  done
  DEP="afterok:${PILOTS[$k]}${OTHERS:+,afterany$OTHERS},singleton"
  # shellcheck disable=SC2086
  submit "$SYSTEM full" --kill-on-invalid-dep=yes --dependency="$DEP" ${FULL_FLAGS[$k]} "$B" "$CODE_ROOT" "$SYSTEM" full placement
  score "$SYSTEM" "$RUN/llm-open-$SYSTEM-P-001" P "$COHORT" "$ID"
done
for k in "${!SYSTEMS[@]}"; do
  test "${SYSTEMS[$k]}" = O2 || continue
  # shellcheck disable=SC2086
  submit "O2 gt-pilot" --kill-on-invalid-dep=yes --dependency="afterok:${PILOTS[$k]},singleton" $GT_FLAGS \
    "$B" "$CODE_ROOT" O2 pilot placement-gt
  score O2 "$RUN/llm-open-O2-PGT-pilot-001" PGT-pilot "$PILOT" "$ID"
done
scontrol release "$ROOTS" || { echo "ABORT: could not release $ROOTS; cancel with: scancel $SUBMITTED" >&2; exit 1; }
echo "released $ROOTS (GPU ceiling $CEILING GPU-minutes)"
