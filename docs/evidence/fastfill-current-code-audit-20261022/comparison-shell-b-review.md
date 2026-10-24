# FastFill comparison submit-shell review, part B

Snapshot: `825fd13ea18f45835db1a242197359a77ea99f9e`. Read-only review of sorted relative comparison `.sh`/`.sbatch` paths 26–50: `submit_011.sh` through `submit_test_012.sh`. All **25 scripts / 3,062 lines** read in full, including embedded Python and job-mode branches. Coverage/hashes: `comparison-shell-b-files.json`.

## Verified finding

### CSB-01 — P2: failed judge smoke still unlocks the full paid run

Location: `fastfill/comparison/submit_023_judge.sh:35–38`, success/failure marker emitted at `:45`; corroborating client `fastfill/comparison/server_api_judge.py:581–594`.

The launcher documents that `full` is allowed only after the two-case smoke passes, but its only gate is `test -e "$BASE/smoke/results"`. The client creates that directory before preparing/calling the judge, so a failed API attempt retains it. The shell separately writes `smoke.completion.json` with the exit code, but `full` does not inspect it or the per-case success/schema/model records. A failed smoke therefore allows a new 696-selected-case batch job against the same failing setup. This defeats the intended pilot gate and can waste paid calls/time; it is not evidence that an actual historical full job was submitted erroneously.

Local reproduction (no Slurm/API): copy the script into a temporary directory, replace only its fixed `PROJECT_ROOT` with a temporary fixture, supply local `squeue`/`sbatch` stubs, make `smoke/results` exist with zero successful cases and `smoke.completion.json={"exit_code":1}`, invoke `full`. Observed wrapper exit `0`, captured submission contains `judge_batch.py --expected-total 696`. Root independently confirmed the real client creates `results_dir` before its API call.

Suggested fix: require the smoke completion marker to indicate exit zero **and** both expected smoke cases to contain successfully validated results for the intended model/config before constructing or submitting the full job. Add regression cases for existing-empty results, API-failed results, malformed responses, missing completion and passing smoke.

## Remaining scope and checks

The other 24 scripts were semantically reviewed for mode/argument dispatch, data/model identity passed to Python entry points, write-once/restart handling, held roots, dependency fan-in, timeout/resource ceilings and propagated failures. No additional defect with sufficient evidence was elevated in this scope. The 018/021 numeric caps differ from earlier header comments because current constants explicitly reference later addenda; those comment differences were not reported as runtime bugs. Historical scripts that intentionally retain old output naming were likewise not misclassified merely for age.

All 25 scripts pass `bash -n`. Relevant existing shell-test sections were inspected in `test_a028.py`, `test_a021_train.py`, and `test_onpolicy_dpo.py`; other matching tests are indexed in the coverage JSON. Root runs the full test suite. No remote submissions, network/API/GPU calls, production directory changes or repository edits occurred. Reproduction source/output: `comparison_shell_b_repros.py`, `comparison-shell-b-repros.json`.
