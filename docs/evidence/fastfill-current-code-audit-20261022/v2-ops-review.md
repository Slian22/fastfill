# FastFill v2 ops current-source review

Snapshot: `825fd13ea18f45835db1a242197359a77ea99f9e`. Read-only production audit of every Python/shell/sbatch file under `fastfill/v2/ops`, including statistical comparison, report generation, frozen-artifact inputs, Isambard orchestration, ablation and VLM rendering/export. Per-file coverage and hashes: `v2-ops-files.json` (also `ops-files.json` for v2 aggregate).

## Verified findings

### OPS-01 — P2: cached stratification can falsely certify baseline reproducibility after its fit input changes

Location: `fastfill/v2/ops/stratify.py:223–228`, external fit dependency at `:234–238`.

`collect(..., cache=...)` includes outcomes, evaluated rows, and Python implementation fingerprints in its cache key, but excludes `report.json` and the external `baseline_fit_file` that the uncached computation reads. Changing that fit file in place returns the previous baselines and previous `reproduces_report_exactly=True` without reading the changed dependency. This matters to the analysis contract: cached and uncached runs on the same current inputs disagree, and a cached comparison/report can keep the green reproduction certificate after drift.

CPU reproduction uses the repository's three-row evaluation fixture, predictions copied from labels, and an external baseline fit file. Initial collection is exact. Doubling valid fit-object sizes, without changing evaluation rows/outcomes/code, produces:

- cached: `reproduces_report_exactly=true`, baseline mismatch requests `0`;
- uncached: `reproduces_report_exactly=false`, baseline mismatch requests `3`.

Suggested fix: bind the report and all external fit dependencies into the cache key before lookup (and preserve the evaluator's original fit identity where available). Add a regression that mutates the fit dependency and compares cached/uncached collection. This does not establish that an existing published report was affected; default evaluation-set leave-one-out fits do not exercise the external-fit trigger.

### OPS-02 — P2: the structured LLM launcher reports success even when both modes and scorers fail

Location: `fastfill/v2/ops/run_structured.sh:8–15` (also absence of `-e` at line 3).

Each background subshell continues to scoring after generation failure; row mismatch only prints a message. The parent calls bare `wait`, which waits for all jobs but returns success rather than collecting their exit statuses, and unconditionally ends with `echo ALL-DONE`. Therefore the command succeeds even if every agent/scorer process fails, defeating a calling shell/job dependency's failure detection and potentially scoring stale files.

CPU shell reproduction ran the unchanged script with its fixed `cd` redirected to a temporary fixture and a local `env/bin/python` shim that always exits `42`; no API calls. Both modes print `ROWS DIFFER`, the wrapper prints `ALL-DONE`, and exits `0`. Suggested fix: fail each mode immediately on generation/row-check failure, capture each background PID, wait each PID and return failure if any failed. Similar historical `rescore_llm.sh` explicitly prints per-mode failures but still exits success; its reporting-only semantics should be made explicit or propagated as well.

## Coverage and strengths

All production files were read in full, including embedded Python heredocs; shell/sbatch syntax checks pass. Fully read directly relevant tests: `test_isambard_inputs.py`, `test_vlm_judge.py`, `test_audit_c2_uniform_yaw.py`. Arm selection requires matching cohort bytes and scoring implementation; downloads are pinned; baseline mutation occurs on a copy; comparisons verify matching rows/code; report separates validation decisions from test/LLM/ablation evidence and calls out selection overlap, placeholder checkpoints, and validation drift. VLM projection tests cover front direction, camera coverage and box rendering.

## Validation limits

No Slurm submission, network/download, GPU, model-scale inference, paid API, repository source/test edit, or production artifact mutation. Full pytest is owned by the root agent. Reproduction script: `v2_ops_repros.py`; actual output: `v2-ops-repros.json`. Current findings are bounded to their demonstrated triggers; no inference that previously published experiments were corrupted. Potential identity limitations for manually mismatched LLM summaries/latency files were inspected but not elevated beyond the verified findings above.
