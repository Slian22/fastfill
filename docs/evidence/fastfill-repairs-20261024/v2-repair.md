# FastFill v2 repairs — 2026-10-24

Reviewed baseline: `873251f0b1f299230c42d510fe5c4f732aa77ce9`. This work addresses F06, F07, F10 and F12 from the current-code audit. Verification used local CPU fixtures, fake API transports and temporary directories only; no SSH, GPU job, API purchase, historical-output rewrite, commit or push was performed by this worker.

## Changes and regression evidence

| Finding | Minimal change | Verified behavior |
| --- | --- | --- |
| F06: stale stratification cache | `ops/stratify.py` includes `report.json` and the resolved external baseline-fit path/content in its cache key. | After changing either the fit samples or a reported metric, cached and uncached collection agree and detect the mismatch. Previous cache files remain untouched. |
| F07: request path traversal | `roomgenbench.py` rejects missing/non-string/empty/dot/path-containing `scene_key` values before creating the output directory, checks the resolved destination parent and creates files exclusively. | Relative traversal, absolute paths, forward/backslash components and invalid keys cannot modify the reference checkout. An invalid second scene leaves no partial output directory. Existing normal RoomGenBench tests pass. |
| F10: valid answer lost on repair transport failure | New `llm_baseline_v2.py` retains the last valid layout on repair transport/parse failure and records `repair_error`. Its summary records `harness_version=repair-v2`, the new implementation hash and the frozen helper implementation hash. | Repair failures do not restart the first-answer loop. Both an initial valid layout and a subsequently repaired valid layout survive later failures. Initial malformed/transport failures still follow the existing retry and failure-accounting contract. |
| F12: wrapper reports success after child failures | `ops/run_structured.sh` stops each failed generation/row-check/evaluation chain and waits for every recorded child PID. | Either or both child failures produce nonzero exit status; row mismatch prevents scoring; `ALL-DONE` appears only when both chains succeed. |

## F10 historical compatibility and safe entry points

`fastfill/v2/llm_baseline.py` is unchanged and remains byte-identical to commit `510f1e0`:

```text
sha256 bc50fa57e9250fe86442ed3981c0e9eef4d6388a7303245df0324e0cd97be459
```

The new module is invoked as `python -m fastfill.v2.llm_baseline_v2`, with the previous baseline's data/output/env/mode options. It requires a fresh output path. Legacy artifacts retain their original identity and interpretation; no historical score is relabelled as repaired.

The public `run` entry point validates nonnegative integer `repairs`/`retries`, positive integer `workers`, and `max_samples=None` or a positive integer before reading inputs, creating output or calling the transport. Boolean, fractional and otherwise invalid counts fail explicitly; zero repairs/retries retain their intended meaning.

The LLM branch of `autorun.py` now defaults to `--llm-baseline legacy`, which only reuses matching historical outputs. Missing identity, input/model/endpoint drift or incomplete output is recorded and skipped, with no deletion and no replacement API call. Fresh generation requires `--llm-baseline repair-v2`, which uses `llm-repair-v2-{mode}-{rows}` output directories. A changed experiment requires a new `--llm-run-tag`; existing directories are never overwritten. Generation metadata is checked after the subprocess completes.

If only the scorer or saved predictions change, autorun can run the CPU evaluator into a new report directory derived from the input/evaluator identity. Existing reports remain byte-for-byte intact. A conflicting pre-existing alternate directory fails closed. The final model comparison selects the requested baseline version's rows. Other training/evaluation branches of the historical autopilot were outside this repair and were not executed.

Tests exercise matching legacy reuse, protected unbound legacy outputs, new-version generation, restart reuse, separate rescoring, identity drift without generation, and explicitly tagged new experiments. The previous autorun test that required destructive regeneration was updated to the new preservation contract.

## Validation

- RED, before production changes: `test_repairs_20261024.py` produced **18 failed, 1 passed**. The passing case was the existing successful shell path. The legacy-output fixture included the actual legacy module so failure could not be masked by a missing implementation file.
- Initial GREEN: all **19** original regressions passed.
- Reviewer-requested parameter boundary RED: **14 failed, 26 deselected**, before the new entry-point validation; the checks require no output directory and no transport invocation. A positive zero-count control was also added.
- Final regression set: **133 passed in 23.91 s**, covering all **41 repair cases** plus `test_autorun.py`, `test_llm_baseline.py`, `test_llm_structured.py`, `test_roomgenbench.py`, `test_roomgenbench_fix20261006.py` and `test_round6_roomgenbench.py`. See [the final test output](v2-final.log).
- New harness statement coverage: **83% (68/82 statements)**. Uncovered code is the CLI wrapper and invalid mode/missing-model rejection. Coverage used the existing temporary QA dependency directory; no dependencies were installed or changed.
- `bash -n fastfill/v2/ops/run_structured.sh` and `git diff --check` passed.
- The existing frozen-harness byte-identity test passed; direct comparison with `git show 510f1e0:fastfill/v2/llm_baseline.py` independently matched the hash above.

Re-run the focused suite from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m pytest --import-mode=importlib \
  fastfill/v2/tests/test_repairs_20261024.py fastfill/v2/tests/test_autorun.py \
  fastfill/v2/tests/test_llm_baseline.py fastfill/v2/tests/test_llm_structured.py \
  fastfill/v2/tests/test_roomgenbench.py fastfill/v2/tests/test_roomgenbench_fix20261006.py \
  fastfill/v2/tests/test_round6_roomgenbench.py -q
```

## Deployment boundary

Deploy corrected source into a new immutable snapshot and run the CPU/fake-transport suite there. Do not restart or replace the completed 27B-P/27B-J training, their frozen source snapshots, training guards, checkpoints, delivery receipts or old `llm-*-300` outputs. The live-training audit found matching code/data/model bindings and complete finite-loss runs; these four fixes provide no evidence that those trained weights require regeneration.

Any future real API experiment must use the explicitly versioned entry point and a fresh output directory. A changed evaluator may rescore existing predictions only into a new directory. Local regression evidence establishes the repaired behavior; it does not establish that each synthetic fault occurred in historical production artifacts. Full-suite validation, independent review and server deployment receipts are recorded separately by the coordinating worker.
