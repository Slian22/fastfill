#!/bin/bash
# FastFill v2 round 15 — run plan after main7-cell05-main-20261008a-e5-yawcls05. NOT RUN YET: start it right after the
# autopilot's phase F (STATUS.json phase done / done-with-failures) and BEFORE the control run, with the user's go-ahead.
# Everything writes under runs/analysis/; nothing else is touched. GPU 0 is never used.
#
#   bash runs/analysis/run_plan.sh gpu      # (1) GPU lanes in parallel on idle cards 1-7 + CPU RoomGenBench   ~35 min
#   bash runs/analysis/run_plan.sh cpu      # (2) CPU only (may run beside the control run)                      ~50 min
#   bash runs/analysis/run_plan.sh report   # (3) the HTML report                                                 ~1 min
#   bash runs/analysis/run_plan.sh all      # (1) (2) (3) in order
#
# (1) gpu: one lane per idle card (a card with no compute process; preflight before every step), at most 5 lanes;
#     with fewer idle cards the lanes queue on them. Every GPU forward pass saves its head outputs; nothing below
#     reads a GPU-decoded layout except as a cross-check. Estimates scale the logged 2026-10-07 test run (13075
#     requests, 164k objects: 21.5 min) by object count (new test 214k, new validation 198k; the baseline skips its
#     > 128-object rooms):
#       lane 0  heads base-val    baseline, NEW validation, minimal + full ............................ ~23 min
#       lane 1  heads final-val   final,    NEW validation, minimal + full ............................ ~28 min
#       lane 2  heads base-test   baseline, NEW test, minimal + full .................................. ~25 min
#       lane 3  heads final-test  final,    NEW test, minimal + full .................................. ~30 min
#       lane 4  heads final-llm   final, the 300 LLM rows, full (< 1 min); then diag_ablation on the 500-row
#               sample: final/new rows, baseline/new rows, baseline/old rows (~5 min each) ............ ~17 min
#       beside: RoomGenBench, CPU: reference check of phase F's (CPU) export of the final; the baseline exported
#               again on the CPU exactly like phase F (run_checkpoint.sh's predict) + reference check ... ~8 min
#     wall ~35 min with >= 5 idle cards (5 parallel lanes); ~120 GPU min in total.
# (2) cpu (<= 8 processes): every head output decoded twice by the same code (spread, argmax; 10 runs) ~15 min;
#     cross-check against phase F's own GPU-decoded reports (identical layouts expected) ~2 min; stratify x16 ~15 min
#     (measured 2026-10-08 on old test: 32 s minimal, 49 s full at 7 workers; x1.3 objects); compare x32 on cached room
#     records ~16 min (measured: 25 s for 4455 rooms, 37 s for 8620 rooms); llm_compare ~1 min.
# Wall time per step: runs/analysis/walltime.tsv (step, start, end, seconds, exit code, card).
set -euo pipefail
R=/home/jovyan/shanliantian/fastfill
A=$R/runs/analysis
cd $R
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
       HF_HOME=/home/jovyan/shanliantian/.huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$R/env/bin/python
NAME=main7-cell05-main-20261008a-e5-yawcls05
BASE=$R/runs/main7-cell05-main-20261007b-e5/model-step-6357
NEW=$R/data/main-20261008a
OLD=$R/data/main-20261007b
LLMROWS=$R/runs/llm-prompt-300/rows.jsonl
SAMPLE=$A/ablation-sample-500   # drawn on CPU before the final checkpoint existed (sample_validation.py, seed 0)
COHORT_NEW=$R/runs/autorun/validation-$(sha256sum $NEW/validation.jsonl | cut -c1-12)-all.jsonl  # the final's select2 rows
COHORT_OLD=$R/runs/autorun/validation-$(sha256sum $OLD/validation.jsonl | cut -c1-12)-all.jsonl  # the baseline's select2 rows
COHORT_OLD1=${COHORT_OLD%-all.jsonl}-3000.jsonl   # the baseline's select1 rows (select1 of 2026-10-08 02:33 ranked its steps)
G=$A/gpu; F=$A/final; CACHE=$A/cache; mkdir -p $G $F

final_checkpoint() {  # the autopilot's select2 winner; refuses before phase F has finished
  $P - <<EOF
import json, sys
s = json.load(open("$R/runs/autorun/STATUS.json"))
if s["phase"] not in ("done", "done-with-failures") or "$NAME" not in s.get("best", {}).get("checkpoint", ""):
    sys.exit(f"refused: autopilot phase {s['phase']}, best {s.get('best', {}).get('checkpoint')}")
print(s["best"]["checkpoint"])
EOF
}

idle() {  # card $1 is not GPU 0 and runs no compute process (a training rank, someone else's job)
  [ "$1" != 0 ] || return 1
  local uuid apps
  uuid=$(nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader) || return 1
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || return 1
  [[ "$apps" != *"$uuid"* ]]
}

# Functions run by timed (or after ||) run with errexit ignored (bash rule): each propagates failures with || return 1.
timed() {  # timed STEP CARD LOG CMD...: runs CMD > LOG, appends its wall time to walltime.tsv, returns its exit code
  local step=$1 card=$2 log=$3 t0 rc=0; shift 3; t0=$(date +%s)
  "$@" > "$log" 2>&1 || rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$step" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u +%FT%TZ)" $(( $(date +%s) - t0 )) $rc "$card" >> $A/walltime.tsv
  echo "$step (card $card): exit $rc after $(( $(date +%s) - t0 )) s"
  return $rc
}

gpu_step() {  # gpu_step STEP CARD LOG CMD...: refuses a busy card right before the step
  idle $2 || { echo "refused: $1 — GPU $2 is GPU 0 or has a compute process"; return 1; }
  local step=$1 card=$2 log=$3; shift 3
  timed $step $card $log env CUDA_VISIBLE_DEVICES=$card "$@"
}

heads() {  # heads NAME CHECKPOINT DATA CARD PROJECTION...: one forward pass per request, head outputs saved
  local name=$1 ckpt=$2 data=$3 card=$4; shift 4
  gpu_step heads-$name $card $G/$name-online.log $P -m fastfill.v2.evaluate --checkpoint $ckpt --data $data \
    --projection "$@" --device cuda --grid-decode spread --output $G/$name-online --save-head-outputs $G/$name-heads
}

ablation() {  # ablation NAME CHECKPOINT ROWS TRAIN CARD; the sample holds exactly 500 eligible rows in a seeded order
  gpu_step $1 $5 $G/$1.log $P $A/diag_ablation.py --checkpoint $2 --validation $3 --train $4 --rows 500 --device cuda \
    --out $G/$1 || return 1
  $P -c "import json,sys; r=json.load(open('$G/$1/report.json')); sys.exit(0 if r['rows'] == 500 else f'$1 scored {r[\"rows\"]} rows')"
}

lane0() { heads base-val $BASE $NEW/validation.jsonl $1 minimal full; }
lane1() { heads final-val $FINAL $NEW/validation.jsonl $1 minimal full; }
lane2() { heads base-test $BASE $NEW/test.jsonl $1 minimal full; }
lane3() { heads final-test $FINAL $NEW/test.jsonl $1 minimal full; }
lane4() {
  local rc=0
  heads final-llm $FINAL $LLMROWS $1 full || rc=1
  ablation ablation-final-new $FINAL $SAMPLE/new.jsonl $NEW/train.jsonl $1 || rc=1   # each checkpoint's own train prior
  ablation ablation-base-new $BASE $SAMPLE/new.jsonl $OLD/train.jsonl $1 || rc=1
  ablation ablation-base-old $BASE $SAMPLE/old.jsonl $OLD/train.jsonl $1 || rc=1
  return $rc
}

rgb_export() {  # the baseline exported on the CPU like phase F's run_checkpoint.sh; both exports reference-checked
  local key
  mkdir -p $A/rgb-base $A/reference-final $A/reference-base
  for req in $R/runs/roomgenbench/requests/*.json; do
    key=$(basename $req .json)
    OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES= $P -m fastfill.v2.predict --checkpoint $BASE --request $req --device cpu \
      --output $A/rgb-base/$key.prediction.json --export-dir $A/rgb-base/$key.handoff || return 1
    for side in final:$EXPORT base:$A/rgb-base; do
      CUDA_VISIBLE_DEVICES= $P -m fastfill.v2.roomgenbench --reference-check ${side#*:}/$key.handoff \
        --scene $R/RoomGenBench/bench/inputs/scenes/$key.json --output $A/reference-${side%%:*}/$key.json || return 1
    done
  done
}

step_rgb() {
  EXPORT=$R/runs/roomgenbench/best-$NAME   # phase F's CPU export of the final (run_checkpoint.sh)
  [ "$(find $EXPORT -path '*.layout_boxes/receipt.json' 2>/dev/null | wc -l)" = 5 ] || { echo "refused: $EXPORT lacks 5 receipts"; return 1; }
  local code
  code=$($P -c "from fastfill.v2.autorun import implementation_sha256; print(implementation_sha256('$R'))")
  [ "$(sed -n 2p $EXPORT/checkpoint.txt)" = "$code" ] || {
    echo "refused: phase F exported the final with other fastfill/v2 code than is on disk now; the baseline would not be comparable"; return 1; }
  timed rgb-export-and-check cpu $A/rgb.log rgb_export
}

step_gpu() {
  FINAL=$(final_checkpoint)
  local cards=() pids=() n k i fail=0
  for c in 1 2 3 4 5 6 7; do if idle $c; then cards+=($c); fi; done
  [ ${#cards[@]} -gt 0 ] || { echo "refused: every card 1-7 has a compute process"; exit 1; }
  n=$(( ${#cards[@]} < 5 ? ${#cards[@]} : 5 ))
  echo "idle cards: ${cards[*]}; $n lanes"
  step_rgb > $A/rgb-step.log 2>&1 & pids+=($!)
  for k in $(seq 0 $((n - 1))); do
    ( rc=0; for i in 0 1 2 3 4; do if [ $((i % n)) = $k ]; then lane$i ${cards[k]} || rc=1; fi; done; exit $rc ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || fail=1; done   # every lane's own exit code
  [ $fail = 0 ] || { echo "GPU-STEP FAILURE: see $A/walltime.tsv, $A/rgb-step.log and the logs in $G"; exit 1; }
  echo GPU-DONE
}

crosscheck() {  # identical layouts expected: phase F's GPU-decoded reports and our online runs vs the CPU decodes
  local pairs=() m o
  pairs+=(--pair "phase F 测试·三字段·spread=$R/runs/test-$NAME/outcomes.jsonl,$G/final-test-spread/outcomes.jsonl")
  pairs+=(--pair "phase F 测试·完整条件·spread=$R/runs/test-$NAME/outcomes-full.jsonl,$G/final-test-spread/outcomes-full.jsonl")
  pairs+=(--pair "phase F 测试·三字段·argmax=$R/runs/test-$NAME-argmax/outcomes.jsonl,$G/final-test-argmax/outcomes.jsonl")
  pairs+=(--pair "phase F LLM 行·spread=$R/runs/llmrows-$NAME/outcomes.jsonl,$G/final-llm-spread/outcomes.jsonl")
  pairs+=(--pair "phase F LLM 行·argmax=$R/runs/llmrows-$NAME-argmax/outcomes.jsonl,$G/final-llm-argmax/outcomes.jsonl")
  for m in base-val final-val base-test final-test final-llm; do for o in outcomes.jsonl outcomes-full.jsonl; do
    if [ -f $G/$m-online/$o ]; then pairs+=(--pair "本计划在线运行 $m（$o）=$G/$m-online/$o,$G/$m-spread/$o"); fi
  done; done
  $P $A/crosscheck.py "${pairs[@]}" > $F/crosscheck.md
}

step_cpu() {
  export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
  local name data proj d o s m sel
  # every saved head output decoded twice by the same code (8 processes at a time)
  for x in base-val:$NEW/validation.jsonl final-val:$NEW/validation.jsonl base-test:$NEW/test.jsonl \
           final-test:$NEW/test.jsonl final-llm:$LLMROWS; do
    name=${x%%:*}; data=${x#*:}; proj="minimal full"; [ $name != final-llm ] || proj=full
    for d in spread argmax; do
      [ -e $G/$name-$d ] || echo "$P -m fastfill.v2.evaluate --from-head-outputs $G/$name-heads --data $data --projection $proj --grid-decode $d --output $G/$name-$d > $G/$name-$d.log 2>&1"
    done
  done > $G/decode-commands.txt
  timed decode cpu $G/decode.log xargs -d '\n' -P 8 -I CMD bash -c CMD < $G/decode-commands.txt
  timed crosscheck cpu $F/crosscheck.log crosscheck
  S() { $P $A/stratify.py --workers 8 --cache $CACHE "$@" > /dev/null; }
  C() { $P $A/compare.py --workers 8 --cache $CACHE "$@" > /dev/null; }
  stratify_all() {
    for m in base final; do for s in val test; do for d in spread argmax; do
      S --eval-dir $G/$m-$s-$d --out $F/$m-$s-minimal-$d.json || return 1
      S --eval-dir $G/$m-$s-$d --outcomes outcomes-full.jsonl --out $F/$m-$s-full-$d.json || return 1
    done; done; done
  }
  compare_all() {  # final minus baseline on rooms both can answer (<= 128 objects); spread minus argmax per model
    for s in val test; do for proj in minimal full; do
      o=outcomes.jsonl; [ $proj = minimal ] || o=outcomes-full.jsonl
      for d in spread argmax; do
        for sel in all "old rule=frozen_prep" "new rule=wall_anchor,support_inside_parent,support_on_added_parent"; do
          set -- $sel
          C --a $G/final-$s-$d --a-outcomes $o --b $G/base-$s-$d --b-outcomes $o --label-a 最终 --label-b 基线 --max-objects 128 \
            ${2:+--where $2} --out $F/compare-$s-$proj-$d-$1.json || return 1
        done
      done
      for m in final base; do
        C --a $G/$m-$s-spread --a-outcomes $o --b $G/$m-$s-argmax --b-outcomes $o --label-a spread --label-b argmax \
          --out $F/decode-$s-$m-$proj.json || return 1
      done
    done; done
  }
  llm_all() {  # FastFill (CPU decodes; latency from the online run) vs the four LLM modes on the frozen 300 rows
    local M=() COH=()
    for m in prompt harness structured structured-harness; do
      M+=(--method "LLM $m=$R/runs/llm-$m-300-eval-4125b82,$R/runs/llm-$m-300/summary.json"); done
    for f in $COHORT_NEW $COHORT_OLD $COHORT_OLD1; do if [ -f $f ]; then COH+=(--selection-cohort $f); fi; done
    $P $A/llm_compare.py --method "FastFill spread=$G/final-llm-spread,$G/final-llm-online" \
      --method "FastFill argmax=$G/final-llm-argmax,$G/final-llm-online" "${M[@]}" "${COH[@]}" --out $F/llm-compare.json --workers 8
  }
  timed stratify cpu $F/stratify.log stratify_all
  timed compare cpu $F/compare.log compare_all
  timed llm-compare cpu $F/llm-compare.log llm_all
  echo CPU-DONE
}

step_report() {
  local args=() m s p d n x
  declare -A M=([base]=基线 [final]=最终) PJ=([minimal]=三字段 [full]=完整条件) SEL=([all]=全部对象 [old]="旧对象 frozen_prep" [new]=新对象)
  for s in val test; do
    for m in base final; do for p in minimal full; do for d in spread argmax; do
      args+=(--$s-run "${M[$m]}·${PJ[$p]}·$d=$F/$m-$s-$p-$d.json"); done; done; done
    for p in minimal full; do for d in spread argmax; do for x in all old new; do
      args+=(--$s-compare "最终 vs 基线·${PJ[$p]}·$d·${SEL[$x]}（≤128 物体）=$F/compare-$s-$p-$d-$x.json"); done; done
      for m in final base; do args+=(--$s-decode "${M[$m]}·${PJ[$p]}：spread vs argmax=$F/decode-$s-$m-$p.json"); done; done
  done
  if [ -f $COHORT_NEW ]; then args+=(--selection-cohort "最终（select2，新验证集全集）=$COHORT_NEW"); fi
  if [ -f $COHORT_OLD ]; then args+=(--selection-cohort "基线（select2，旧验证集全集）=$COHORT_OLD"); fi
  if [ -f $COHORT_OLD1 ]; then args+=(--selection-cohort "基线（select1，旧验证集 3000 行）=$COHORT_OLD1"); fi
  $P $A/report.py --out $F/report.html --title "FastFill v2 评测报告：基线 vs $NAME" --summary $R/runs/autorun/SUMMARY.md \
    "${args[@]}" --llm $F/llm-compare.json \
    --ablation "最终·新行（500 分层样本）=$G/ablation-final-new/report.json" \
    --ablation "基线·新行（同一样本，相同输入）=$G/ablation-base-new/report.json" \
    --ablation "基线·旧行（同一批场景）=$G/ablation-base-old/report.json" --ablation-sample $SAMPLE/sample.json \
    --reference "最终（phase F 的 CPU 导出）=$A/reference-final" --reference "基线（当前代码 CPU 重新导出）=$A/reference-base" \
    --walltime $A/walltime.tsv --notes $F/crosscheck.md
}

case "${1:-}" in
  gpu) step_gpu ;; cpu) step_cpu ;; report) step_report ;;
  all) step_gpu; step_cpu; step_report ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
