# Source-correction legacy release staging

This sidecar builds **one diagnostic legacy release candidate** and compares every released UID with the immutable historical train/dev/test files. It does not regenerate IR, edit source adapters, upload files, submit jobs, call APIs, or launch corpus/main-data/training. Only these four staging files were added by this implementation task.

All input paths and manifest pins come from a parent-created configuration. No candidate version is hardcoded or approved here. In particular, candidate001's opening-piece/yaw changes are not implicitly accepted: the parent must finish the source patch/IR gate and supply the final reviewed version (candidate002 or its successor).

## Handoff to the parent

1. Complete the final source-correction review. Freeze the code package and three regenerated source IR files. Upload these; copy the other 15 IR files from the existing server `/automation/unknownsize-20261013/inputs/ir` under the task's server root. Do not regenerate or modify those 15 originals.
2. Assemble a fresh IR directory containing exactly the original 18 `<source>.jsonl` filenames. Its `IR-MANIFEST.json` must be **outside that directory**. Its `files` object maps each filename to at least `{"sha256": "<64 lowercase hex>", "rows": <nonnegative integer>}`. No null row counts. Additional receipt metadata is allowed.
3. The copied code package must contain `CORRECTION-MANIFEST.json` with `files` mapping every package file to its SHA-256, relative to `code_root`. Only the three source adapters may differ from the historical implementation; `adapters/upright_bounds.py` is the sole allowed new implementation module. All manifest-listed files, including tests/docs, are checked. Unmanifested files and symlinks fail; existing `__pycache__` files are ignored for inventory and cannot be used by the isolated build interpreter.
4. Provide an immutable historical release directory containing its original `MANIFEST.json` and all eight manifest-listed output files, including train/dev/test, room files and stats. The old manifest supplies the entire build argument set and the original 18-source inventory. Do not create a replacement old manifest with corrected hashes.
5. Create a new config with the eight required keys below, using final absolute server paths and independent manifest hashes. Choose a nonexistent `output_root` whose parent exists and which is outside all input directories. Existing destinations, including empty directories, are rejected. Then review the runner, tests and Slurm template before the parent submits it once.

```json
{
  "code_root": "/absolute/final-reviewed-code-package",
  "code_manifest_sha256": "<SHA256 of code_root/CORRECTION-MANIFEST.json>",
  "ir_dir": "/absolute/final-reviewed-18-source-ir",
  "ir_manifest": "/absolute/IR-MANIFEST.json",
  "ir_manifest_sha256": "<SHA256 of IR-MANIFEST.json>",
  "old_release_dir": "/absolute/immutable-historical-v3.2-release",
  "old_manifest_sha256": "<SHA256 of old_release_dir/MANIFEST.json>",
  "output_root": "/absolute/nonexistent-release-audit-output"
}
```

The angle-bracket strings are placeholders and are deliberately not runnable pins. No extra config keys are accepted. The historical code path on the local workstation is `.release/v3.2/code`; the historical release is `.release/v3.2/data/v3.2`. Server paths are supplied by the parent, not inferred by this runner.

Parent-only submission command, **not executed in this task**:

```bash
sbatch /absolute/staging/release.sbatch \
  /absolute/final-release-config.json \
  /absolute/staging/run_release.py \
  /absolute/environment/bin/python
```

The explicit runner path is required because Slurm copies the batch script into a spool directory. The wrapper requires Python 3.9+; the selected Python environment must already provide NumPy and the frozen build's compatible Shapely (2.x). The parent supplies that environment; the sidecar installs nothing.

## Execution and integrity checks

`run_release.py --config /absolute/config.json` performs these steps:

- Before any build, verify the pinned code manifest and every listed file; the pinned IR manifest, all 18 file hashes and physical JSONL row counts; and the old release manifest plus every listed file hash/line count. Reject code changes outside the three adapters/helper and require the other 15 IR hashes to equal the historical hashes. Index every old train/dev/test UID, rejecting duplicates, invalid source IDs and missing flags/messages.
- Exclusively create `output_root`. Collect a runtime receipt with Python/executable/platform, NumPy, Shapely and GEOS versions. Invoke `fastfill.build` from the frozen package using an isolated interpreter, a new cache prefix and no bytecode writes; inherited `PYTHONPATH` or existing `.pyc` cannot override the package.
- Pass **every old manifest argument**, replacing only `ir` and `out`. This includes `constraint_frac=0.3`, source order, all filters, empty lists, false booleans and `source_anchors`. Unexpected/missing argument keys fail instead of falling back to defaults. No source or filter overrides are available in this config.
- After the builder returns, even if it fails, recheck config, code, IR, old manifest and old release file pins. Verify the candidate manifest's exact effective arguments, implementation hashes, IR hashes, eight artifact hashes and line counts. The candidate lives at `output_root/release`.
- Index every candidate train/dev/test UID and compare the complete old/new UID union. Hash canonical JSON for `flags` and `messages` independently; message content strings are not normalized. Produce source counts for **all 18 sources**, including zero-population sources. Shared-UID flag/message change counts are attributed to the old source; explicit source-change in/out counts identify source reassignment.

The full old/new datasets are read by the future bounded CPU job. This implementation task ran only synthetic tests; it did not duplicate the parent's IR scans. Peak legacy build memory is governed by the Slurm allocation, not made streaming by this wrapper. UID comparisons retain compact metadata/digests, not all message content.

## Outputs, status and release gates

All generated artifacts are inside the new output root:

| Artifact | Content |
|---|---|
| `invocation.json` | Config/manifest pins, effective arguments, exact command and runtime versions |
| `build.log` | Frozen builder stdout/stderr |
| `release/` | New candidate legacy release; never the old release |
| `identity-map.jsonl` | Every old/new UID, split, source and flags/messages fingerprints |
| `added.jsonl`, `dropped.jsonl`, `moved.jsonl` | Complete UID additions, removals and split moves |
| `source_changed.jsonl` | Shared UIDs whose source changed |
| `changed.jsonl` | Shared UIDs with changed flags and/or messages, with separate booleans |
| `summary.json` | Status, full counts, all-source split counts, blockers, runtime and ledger hashes/row counts |

The identity status deliberately ignores geometry correctness:

- `candidate_complete_identity_preserved`: UID population, source identity and split assignments are identical. Flags/messages can still change, with counts and fingerprints disclosed. CLI exit **0**.
- `candidate_complete_identity_changed`: any addition, removal, source change or split move. All diagnostic candidate outputs and ledgers are retained. CLI exit **2**, so an `afterok` dependency cannot mistake identity changes for a pass.
- `failed`: integrity, malformed metadata, duplicate UID, builder or output verification failure. CLI exit **1**. Preflight failures create no output directory. Failures after creation retain diagnostic files and a failed summary where possible.

Every old dev/test split move is an explicit `old_heldout_split_migration` blocker, including dev↔test and movement into train. Heldout removals are a separate blocker; added/dropped/source-changed UIDs and train moves also fail the identity gate. Equal aggregate split counts cannot conceal UID migration.

**Neither complete status is scientific acceptance.** `refit_ready=false`, `scientifically_accepted=false` and `full_component_closure_checked=false` remain explicit in both. Group/alias/content/layout component closure, source geometry correctness, changed qualification, newly eligible population policy, train-only OOF priors, P repair, token admission and evaluation integrity require the subsequent reviewed corpus/data workflow. Even exit0 must not be used alone to launch main-data/training. This runner contains no such launch.

The Slurm template requests one node/task, **4 CPUs, 96G RAM, no GRES, two hours**, account `brics.u6uv`, partition `workq`, QoS `normal`, and `--no-requeue`. GNU `timeout` applies **115 minutes to the entire wrapper and its child process group**, with a 60-second TERM→KILL grace. Timeout/Slurm termination may leave no complete summary; a missing/invalid summary is not acceptance. Timeout normally returns124, forced termination may return137. A retry must use a fresh output path; the sidecar never deletes or overwrites partial output.

## Validation and file hashes

TDD sequence: tests first failed for missing implementation; new invariant tests were demonstrated failing before fixes. Final validation: **25/25 local tests passed**, including one real isolated subprocess with a tiny synthetic pinned builder and an actual runtime-version probe. Tests otherwise mock the builder and use temporary synthetic metadata/IR; no production build, remote operation or dataset scan was run.

Coverage includes bad code/IR pins before build, manifest row counts (including null rejection), the other15 source invariants, unrelated prep mutation, existing output protection, exact original arguments, code/IR/old-release changes during build, output counts, duplicate UID rejection, equal-count dev/test swaps, heldout→train, added/lost UIDs, source changes/counts, same-flags changed messages, zero-population source entries and nonzero CLI status for identity changes. `bash -n release.sbatch` passed. Standard-library `trace` measured **315/320 executable lines (98.44%)** for `run_release.py`; this is line coverage, not branch coverage or a cluster execution claim.

Run local tests:

```bash
python -B docs/research/20261013-unknown-size-comparison/source-correction-staging/test_run_release.py
bash -n docs/research/20261013-unknown-size-comparison/source-correction-staging/release.sbatch
```

Independent read-only code review identified the CLI-success-on-migration and omitted-empty-source gaps; both were fixed with failing-then-passing tests. No live scheduler validation or full legacy release run was performed.

SHA-256 pins for the executable/test artifacts (README does not embed its own self-referential hash):

| File | SHA-256 |
|---|---|
| `run_release.py` | `4891efa6f9cfebc4b432889f2230616893a944c3b75a9e760c00808153f88e0e` |
| `test_run_release.py` | `47cb9d46e1f9bb1fa3b2c2bc181295f4f182ece9b90caca81901bca26cb56e47` |
| `release.sbatch` | `26cbc3269081116534e9f6140531db315b062a70218e18c00ff5c525d6cca2fb` |
