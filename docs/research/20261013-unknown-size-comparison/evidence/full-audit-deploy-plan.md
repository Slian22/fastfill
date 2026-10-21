# Full-audit deployment: prepared, switch gated

Preparation may freeze and upload the tested production snapshot. **Do not stop
the current controller, cancel a job, submit geometry, or start a new driver
until the parent relays the independent review's no-HIGH result.** All template
approval fields remain false. No API request is part of deployment preparation.

Parent's data-scan amendment: preserve the current run as a **disclosed
diagnostic** because known source annotations were exposed. These scoring fixes
are still needed; this deployment does not turn the run into a fresh blind test.
Parent owns the scientific disclosure/report. No SFT or inference cancellation
is part of this amendment.

## Frozen scope and identities

- New production snapshot: `pipeline-full-audit-a8726c726f72`, the same 33 declared
  files as `pipeline-9c4d97dbd736`, plus the newly generated MANIFEST.json.
- Exactly three production files differ: `train_pair_helpers.py`, `geometry.py`,
  and `final_tables.py`. The pinned API client, overnight driver, inference code,
  visual pipeline, FID code, and all other production bytes remain unchanged.
- Parent validation: `full-audit-final-validation.json`, 681 tests passed. Its
  SHA256 and each approved source hash are in `full-audit-deploy-snapshot.json`.
- New local output: `automation-run-003`. Independent controller logs and
  handoff receipts go in a **new** private controller state directory, not the
  existing `automation-staging/*-launchd.log` exclusive-create paths.
- Keep remote `runs/unknownsize-20261013` and the original input/inference roots.
  The parent explicitly permits reserving the still-unused `geometry-v1` path
  after the old geometry job is confirmed cancelled without ever starting.
  Record this namespace amendment; never remove or overwrite geometry output.
- Protected jobs: main data 7365659; SFT 7365681/7365682; inference
  7367626/7367628. Only geometry 7367629 may be conditionally replaced.

## Deadline: no restart of the 120-hour window

`full-audit-deploy-deadline.json` fixes the conservative earliest original cutoff
at **2026-10-18 05:53:20 UTC (06:53:20 Europe/London)**, derived from run001's
durable timestamp/elapsed receipt. It is earlier than run002's launcher cutoff,
so no interpretation of "original" receives an extension. The staged controller
uses an independent SIGALRM for this absolute deadline. Immediately before
starting run003 it writes a new immutable config with total_timeout_seconds =
min(original cap, floor(deadline - current_time) - 10). Its final alarm also
protects against interpreter startup delay. Existing render 24h, API job 10h,
348-case cap, zero API retries, and polling 120s stay unchanged. Cleanup can add
up to 35 seconds per tracked child; this is recorded separately, not extra work.

## Compatibility assessment

The real `_inference` reader in `evaluate_pair.py` verifies the inference
MANIFEST, UID inventory, completion status, receipt identity, per-record run
identity, context plan, and evaluation data hash. It does **not** require its
current geometry/helper code hash to equal inference receipt code hashes.
`evaluate()` further checks paired training-data identity, base checkpoint,
context plan and decoding settings. Consequently old frozen inference receipts
remain valid inputs to the new scoring code; do not rewrite those receipts.
The new geometry summary records its own evaluate_pair.py/geometry.py hashes and
the original inference run identities. FID/GPT/final tables bind to the newly
produced geometry MANIFEST, not the old code snapshot. One offline staging test
exercises the real inference reader with an original-code receipt and verifies
that artifact tampering still fails. No inference job replacement is needed.

The parent input guard affects validation helpers but does not retroactively
rewrite or revalidate running training with new code. Keep the original training
and inference snapshot paths unchanged. The complete new snapshot manifest
records the helper change even though the geometry summary lists only its two
scoring modules.

## Ordered handoff, ONLY after the parent review gate

1. Preserve the old config, process identity, status/endstate and latest Slurm
   receipts. Confirm run002 still has stage=jobs, api_submission_attempted=false,
   and no geometry output. Recheck immediately before stopping it. If the stage
   changed, stop this plan and report actual state rather than forcing a switch.
2. Verify local and remote production MANIFEST SHA256 and all 33 files against
   the freeze receipt. Freeze the reviewed ops scripts separately; they are not
   extra production files. No edits to either old or new frozen directories.
3. Copy `spec.template.json` to a NEW final spec, set the parent-ready field only
   after the parent's no-HIGH message, and choose a fresh task-private 0700 /tmp
   socket directory (path <=100 bytes, no expansions/traversal). Keep all pinned
   fields/deadline. Copy the reviewed controller to a new private /tmp directory
   and compare SHA256 before launch. Use the already-authorized CLI-host detached
   Popen pattern: start_new_session=True, stdin=DEVNULL, exclusive file logs.
   Do not use launchd, a UI alternative, or change any privacy/credential policy.
4. Start the new **gated master/controller only**. It verifies the frozen code,
   opens a foreground SSH master without ControlPersist/forking, then executes
   `true` with -vv over that exact socket. Its probe disables direct-connection
   fallback and requires both a multiplexed master session ID and exit status 0.
   Wait for its atomic ready.json and confirm process survival in a later read.
   **It cannot start the driver while release.json is absent.** If master
   authentication fails, leave the old controller/master untouched and abort.
   Establish this connection before today's certificate expiry (parent reports
   12:34 local); do not renew or bypass credentials.
5. Parent stops only the old local launcher 41825 after validating its identity
   and that of driver42372/master41826/wake42373. Allow the existing finally
   cleanup to terminate owned children. Verify all four old PIDs are gone and
   run002 has a durable partial/interrupted endstate at jobs, with API unattempted.
   The new gated master remains alive. Do not reuse old PIDs without rechecking.
6. Through the new socket, acquire fresh `sacct` primary records with
   JobIDRaw,State,Start,End,ElapsedRaw,ExitCode and `scontrol show job -o` for
   **7367629 only**. Require PENDING, no actual Start, ElapsedRaw=0, Restarts=0,
   and absent geometry-v1 (including absent symlink). If any fact is unknown or
   suggests it ran, abort the cancellation/replacement and report it.
7. Receipt the exact cancellation intent, then issue once:
   `scancel --ctld --state=PENDING 7367629`. The server-side state filter protects against the
   pending-to-running race. Do not use --name, --user, broad filters, or any
   training/inference job ID. Confirm primary state CANCELLED, no actual execution/elapsed,
   and still-absent output. If it became RUNNING/COMPLETED, do not cancel it or
   submit the replacement. Preserve all evidence. Create the explicit
   geometry-v1 destination amendment receipt now.

   Actual Slurm cancellation semantics observed during this handoff: scontrol
   assigns equal StartTime/EndTime at cancellation, while sacct records Start=None.
   The never-started proof is pre-cancel PENDING/Start=Unknown, the server-side
   pending-only cancellation, zero elapsed/restarts, no allocation, no batch-step
   record/log, and absent geometry output. The too-strict initial post-cancel
   Unknown assertion failed; only read-only recovery checks followed, with no
   cancellation retry. Preserve these exact observations in the receipts.
8. Upload the reviewed geometry.sbatch to a fresh deployment-control directory
   (outside immutable production snapshots). Pin its SHA256. Create an exclusive
   submission claim before invoking sbatch, with immutable attempt/response
   receipts stored remotely and locally. Submit **once**, parsable, no-requeue,
   dependency `afterok:7367626:7367628`, nodes1/tasks1/CPU32/mem96G/gres=none,
   wall2h; use the original absolute deadline, not now+walltime. Arguments are
   NEW_CODE_ROOT ORIGINAL_RUN_ROOT REMOTE_PYTHON. It verifies the new code and
   scores original inference outputs into the unused geometry-v1 destination.
   The shell script retains workers24/repeats2000/115m process timeout.
   Do not automatically retry an ambiguous SSH/submission result. Recover the
   remote claim/submission receipt read-only; never issue sbatch again blindly.
9. Save `full-audit-deploy-geometry-submission.json` with job_id, exact
   dependency, submission_attempts=1, raw sbatch result, script/code hashes,
   cancellation/amendment receipt hashes, resource settings and absolute deadline.
   Verify the new job's dependency and command via read-only Slurm inspection.
10. Create a NEW candidate config from candidate-config.template.json with only
    the new geometry ID and new socket filled in. No protected IDs or caps may
    change. Create final release.json atomically in the new controller state
    directory from release.template.json, with parent_snapshot_ready=true and exact candidate,
    test receipt, submission receipt and snapshot MANIFEST hashes. The controller
    requires the old processes gone and old endstate partial/jobs/no-API, then
    validates the final config through the frozen driver and writes a new effective
    config with the remaining original duration. It starts run003 once.
11. Verify independent new logs, run003 frozen/config receipts, jobs-only status,
    all six primary job states and actual multiplex transport. Record the new
    controller PID/PGID, driver/master/wake PIDs, socket, snapshot, geometry job,
    preserved deadline and saved old partial run. Do not manually start any API
    job: normal driver sequencing handles the later bounded evaluation.

## Failure handling and remaining limits

- Before old-controller termination: any preparation/readiness failure leaves
  run002 and all Slurm jobs unchanged; close only the new task-owned master.
- After old-controller termination: retain run002 partial evidence. If the
  geometry switch cannot be proven safe, keep the new driver gated and report
  the reason. Do not restart old or new drivers automatically.
- Controller and master loss remain possible under tool-host teardown, logout,
  network loss or server policy. Detachment has only the observed parent-exit
  persistence guarantee. No new OS jobs or permissions are introduced.
- Cleanup tracks direct children and their groups; it does not certify every
  separately grouped descendant after forced termination. Never claim otherwise.
- Certificate expiry is not extended. If no fresh authenticated master can be
  established using normal existing authorization, stop the handoff.

## Prepared files and verification

Source staging: `automation-staging/full-audit-deploy-001/` contains controller.py,
geometry.sbatch, verify_freeze.py, test_staging.py, base-config.json and non-launchable
templates. `verify_freeze.py` checks the original inventory, every hash and exactly
three approved differences. Controller release/config tests use only mocks; the
lineage reader test uses synthetic local files. No real driver, render, API,
cancellation, submission or SSH-master launch is part of these staging tests.
