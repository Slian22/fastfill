# fastfill/v2/ops — operations and final-analysis scripts (2026-10-08)

Server-side scripts used to rebuild data, run the autopilot and analyse the final model. Paths assume the server
checkout `/home/jovyan/shanliantian/fastfill` (copy a script to `runs/` or run it from the checkout); nothing here
holds credentials — the LLM key lives only in `/home/jovyan/shanliantian/.fastfill_api.env` on the server.

Operations
- `rebuild_and_publish.sh <name>` (Mac): full data chain (legacy_build → legacy_verify → review_data → qualified_data
  → multisource_verify) from a pinned worktree (`SRC=`), HF upload, `NEXT_DATA.json`; `NO_HANDOFF=1` holds the server
  handoff until the reports are checked. Stops at the first failing verifier (set -euo pipefail).
- `start_autorun_retrain.sh`: retrain on new data with logged `--set` overrides (`model.max_objects=256`,
  `loss.yaw_cls=0.5`), `--name-suffix`, LLM step disabled (no paid calls).
- `run_checkpoint.sh <model-step> <tag>`: five-room RoomGenBench hand-off on CPU with `--require-placement`
  (the autopilot calls it; runs from any checkout).
- Isambard-AI (aarch64 GH200, Slurm): from a checkout under `$PROJECTDIR`, `bash fastfill/v2/ops/isambard_submit.sh`
  submits `isambard_setup.sbatch` (conda env with the baseline's package versions, pinned data, backbone and LLM rows,
  tests) and, once it succeeds, two `isambard_autorun.sbatch` jobs (one node each): yaw_cls 0.5 and the 0.08 control,
  both from `configs/main7-cell05-main-20261007b-e5.json` (the baseline's configuration) with `max_objects` 256.
  Status in `runs/autorun-yawcls05/` and `runs/autorun-yawcls008/`, job logs in `runs/isambard/`. Resubmitting an
  autorun job resumes it.
- `run_structured.sh`, `rescore_llm.sh`: the OptiScene-style LLM modes on the frozen 300 rows and re-scoring of saved
  LLM predictions with the current evaluator (no API calls); `run_reference.sh`: reference check of a 5-room export.

Final analysis (`run_plan.sh gpu | cpu | report`; `make_sample.sh` re-tests the whole chain on CPU)
- `stratify.py`: per-object errors split by target selection rule (old vs added targets), place, source and object
  count, reproducing `evaluate`'s report exactly; validator collision / clean-room rates per stratum.
- `compare.py`: paired room-level A−B (bootstrap CI of the mean per-room difference as the primary criterion, Holm
  sign test as robustness); refuses different rows or different decoding code.
- `llm_compare.py`: FastFill (raw / spread) vs the four LLM modes on the frozen 300 rows; transport failures,
  latency and token columns.
- `sample_validation.py`, `diag_ablation.py`, `selftest_shift.py`, `crosscheck.py`, `report.py` (self-contained HTML).
Decisions (spread vs argmax, replacing the baseline) read validation comparisons only; test sections are report-only.
