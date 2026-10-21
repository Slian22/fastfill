# Independent visual-binding controller review — 2026-10-13

**PASS — current controller source review. Both previously reported findings, VB-01 and VB-02, are resolved and independently verified. No open findings remain in the reviewed delta. The run003 → run004 local-only handoff remains compatible with existing jobs and frozen remote judge/client code. This is a source-review verdict, not proof of old-controller shutdown or completed deployment.**

Reviewed `automation-staging/resource-binding-deploy-001/controller.py` and `test_controller.py` against `full-audit-deploy-001/controller.py`, the actual launched controller copy, run003 configuration/receipts, and the frozen overnight driver. Only the two requested review files were written. No controller `main`, overnight `Driver` instance, SSH/API request, scheduler operation, or process signal was executed. Offline probes replaced subprocess/network/signal boundaries with failing mocks; PID checks were simulated.

## Re-review of the parent's gate fixes

Current controller SHA-256: `f0fead883eeb50882c77dd82260733ae8ddce5dee723651162d01a86a7803c95`.

Current test SHA-256: `a95df45131c029aa707f99d3ca1a0d8f0fa1c72f475364c9d6c5da20951e3a23`.

Independently verified the precise revision: reversing only the added `ORIGINAL_DEADLINE_UNIX`, revised `remaining`, new `old_identity`, and revised `check_release` reconstructs the prior reviewed controller hash `dd6302efaa5c02b8cf1cc04d8ed5a11d071eb80515955ccd7517e3e38659fa3e`. The four-key allowlist, geometry identity rule, remote-code compatibility, readiness sequence, process cleanup, and all other controller code are unchanged.

`remaining` now independently enforces the finite numeric deadline ceiling **1792302800**. Increasing both spec deadline fields is rejected through `effective_config`; the controller's pre-SSH alarm setup also calls `remaining`. Valid shorter deadlines and the ten-second startup margin still work. **VB-02 is resolved.**

`check_release` now requires `api_submission_attempted is False`. Before any PID liveness checks, `old_identity` binds the old endstate path to the base output directory; verifies the process receipt SHA; requires exactly four unique positive integer PIDs matching launcher/driver/master/wake identities; and binds both receipt/endstate config hashes and the endstate driver PID. **VB-01 is resolved.** Independent probes rejected absent/falsey API fields, missing/short/long/duplicate/wrong/noninteger PID lists, stale run002 endstate paths, wrong process receipt hashes, wrong config hashes, and wrong driver identity before reaching any PID check. A simulated live correctly identified PID still blocks release.

All **6 supplied tests passed**, plus **60 independent offline checks**. The re-review also retained the existing geometry receipt checks, unchanged protected config fields, exhausted-window failures, and compatibility with the frozen overnight config validator. No controller `main`, actual PID query/signal, subprocess deployment, SSH, API, or scheduler operation occurred. Individual results are under `followup_review` in the JSON report; original review results remain historical evidence.

The actual local run003 process receipt at `/tmp/ff-audit-controller-2m973_n1/state/processes.json` passed the new identity check against run003's effective configuration and observed endstate:

- receipt SHA-256: `949ba8681397ba9c961b6c9354f2ab9c25b6c7cd6dd7d1fcab318e68a53825f6`;
- config SHA-256: `13b9ac01984c4842992a8e5af6c828f9bbe1d5f21803905ce10d0dbfb09b6743`;
- launcher/driver/master/wake PIDs: `40191 / 96318 / 40239 / 96319`.

A mocked copy of those exact receipt bytes at a different staged path passes with the same digest; the check does not depend on the original `/tmp` pathname. The parent's planned immutable copy therefore fits the contract. The actual staged copy/spec/release and real process shutdown were not performed or certified here. Those remaining operational steps are not unresolved source findings.

## Resolved findings — initial review retained for provenance

The following two sections describe the superseded `dd6302...` version. Their weaknesses are fixed in the current `f0fead...` version verified above.

### VB-01 — RESOLVED (formerly MEDIUM): incomplete or stale old-run evidence

`controller.py:96–104` accepts a missing `api_submission_attempted` field, or values `null`, `0`, and `""`, as evidence of no API attempt. An empty `old_process_ids` list passes without checking any process. The gate does not bind the old endstate's `config_sha256` or `pid` to the base configuration/current controller receipt. Independent in-memory probes reproduced these acceptances, including a stale run002 endstate with an empty PID list.

The real overnight driver writes a `partial/jobs/api_submission_attempted=false` endstate **while it is running** (`overnight.py:248–266`). At read time run003's endstate had those fields plus `reason=driver_active_or_interrupted`; its status receipt reported `running`. These files are not stop proof. No live PID queries were performed here.

Before release, bind the stop evidence to run003 and require the actual boolean `api_submission_attempted is false`, `status=partial`, and `stage=jobs`, plus absence of every current tracked process. The read run003 receipt identifies launcher **40191**, driver **96318**, master **40239**, and wake process **96319**. Derive identities from the current receipt at handoff rather than blindly copying these numbers. Check the old endstate's config hash against run003's effective config. The controller refuses a supplied PID that exists, but only if the correct nonempty set is supplied.

This gate is inherited unchanged; the geometry/key patch did not cause the weakness. Requiring explicit false and a bound nonempty process set would make it fail closed. No controller edit was made here.

### VB-02 — RESOLVED (formerly MEDIUM): original deadline anchoring

`controller.py:156–158` only checks `deadline_unix <= original_deadline_unix`, with both values supplied by the spec. It does not independently pin **1792302800**. Evaluating the exact AST predicate with both fields set to `1792306400` confirms that this one-hour extension would not be rejected by that guard.

The existing original-deadline receipt, run003 spec, and process receipt consistently say **1792302800 = 2026-10-18 05:53:20 UTC**. Parent preflight must require `original_deadline_unix == 1792302800` and `deadline_unix <= 1792302800`, with the original receipt as provenance. An independent original-receipt pin in the controller would enforce this directly; none was added here.

With a correctly pinned spec, existing mechanics are compatible: `effective_config` returns `min(base_timeout, floor(deadline-now)-10)`, rejects an exhausted startup window, and the controller alarm uses that absolute deadline. Run003's base timeout is **425451 seconds**, below the driver's 432000 maximum. Unchanged shutdown allows 35 seconds per tracked child, normally at most 105 seconds, after the alarm; the original deadline receipt already discloses this grace.

## Verified diff, configuration, and geometry receipt

The initial AST comparison found only **`effective_config`** changed among functions. Other initial changes were the module description and `CHANGED_CONFIG_KEYS`. The subsequent gate fixes are enumerated in the re-review above. Readiness ordering, signals, lifecycle, and cleanup remain unchanged. The old staged controller is byte-identical to `/tmp/ff-audit-controller-2m973_n1/controller.py`.

The new allowlist is exactly `local_code_root`, `output_root`, `ssh_control_path`, and `total_timeout_seconds`. Probes rejected changes to geometry/prerequisites; remote code/client/run paths; client hash/environment; local resource/interpreter paths; Slurm selectors; data-manifest pins; receipt lists; polling/retries; final-table policy; and render/API bounds. Geometry must equal the hashed base config and remain distinct from prerequisites. It is not hardcoded to 7368009, so selecting the correct base matters.

Use **run003's effective config**: `automation-config-run003.json` and `automation-run-003/config.json` are byte-identical, SHA-256 `13b9ac01984c4842992a8e5af6c828f9bbe1d5f21803905ce10d0dbfb09b6743`. Do **not** reuse `full-audit-deploy-001/base-config.json`: it is run002's baseline, with geometry **7367629** and older remote paths. Preserve geometry **7368009** and prerequisites **7365659, 7365681, 7365682, 7367626, 7367628**.

Existing `full-audit-deploy-geometry-submission.json` has SHA-256 `ad57ab0e2720f7c6e7c34fce175c34daf347521ec3fd1aa2d2576674ddaf4bbe`, `job_id=7368009`, `dependency=afterok:7367626:7367628`, and `submission_attempts=1`. It records the original deadline and old frozen geometry snapshot. The new controller accepts this receipt; it does not require a new geometry job or a geometry snapshot matching the new local visual snapshot.

Mocked release probes rejected missing parent-ready authorization, bad candidate/test/geometry receipt hashes, wrong released snapshot, wrong old-run stage/status, a true API-attempt flag, a live supplied PID, wrong geometry ID/dependency, and repeated submission count. Hash matching attests bytes; the controller does not inspect test success, so the parent's snapshot-ready authorization must cover test/review success. Pin the exact existing geometry receipt rather than a newly authored replacement.

Before release the controller requires a new private owned 0700 socket directory and nonexistent socket, creates fresh state, verifies the snapshot, opens/checks a new SSH master, then waits for `release.json`. The driver starts only after release validation, a second manifest verification, config validation, and exclusive config publication. Output reuse is rejected here and by the driver's exclusive directory creation. These lifecycle checks were inspected statically, not exercised against live processes.

## Overnight and frozen remote judge compatibility

Current `overnight.py` is byte-identical to `pipeline-full-audit-a8726c726f72`, SHA-256 `663d33c6ae3cf629d0b6ae0277bc98be1dc0bdd05b5e8fab73a11b76dd05c492`. Its `validate_config` accepts the independently constructed local-only candidate using all other actual run003 fields.

`local_code_root` need not equal `remote_api_code`. Before submission, `launch_api` compares the remote judge hash with the new local snapshot's judge and the remote client with its explicit unchanged pin. A local visual-resource patch is compatible with retaining:

- `remote_api_code`: existing server `.../snapshots/pipeline-full-audit-a8726c726f72`;
- `api_client_path`: that directory's `server_api_judge.py`;
- judge SHA-256: `0d93b727e99118261be14868779a557854afc08a5adf084b2ac7d5545546d7c8`;
- client SHA-256: `8bfd7bc4f2e55b53f3d8701d78bcf00861336c091bdc3132ba10903b61f26b03`.

Workspace and frozen judge bytes match. The eventual new local snapshot must retain those bytes; the client pin is protected by the config allowlist. No code upload is needed. Later API input-bundle transfer and the original planned API judge submission remain unchanged driver behavior, not actions performed here. Remote hashes were not queried during this review; the driver checks them before submission.

Caps remain: total **120 hours** including waiting; render **24 hours**; API **10 hours**; **348 selected arm records**, no replacement/retry; **90 seconds per case** in the driver's judge command; poll **120 seconds**, read retries **3**. The controller test's `api_case_cap` is a synthetic allowlist probe, not a real config field: 348 is hardcoded in the frozen driver/judge.

## Independent checks and release limit

In the initial review, **3 supplied controller tests passed** and **41 additional offline checks passed** for current-config compatibility, protected fields, exhausted deadlines, fresh output, release hashes, old-state checks, supplied-PID liveness, and geometry identity. Separate probes reproduced the then-inherited weaknesses. These are historical results; the revised version passed the 6 tests and 60 checks reported above. The [JSON report](visual-binding-controller-review-20261013.json) preserves both rounds and source identities.

At initial review time the resource-binding staging directory contained controller/test source and a pre-existing `__pycache__`, with no final run004 spec, candidate config, snapshot manifest, or release receipt. The re-review assessed the changed controller/test source and actual run003 process-receipt identity, not a completed deployment. Parent release still needs actual run003 shutdown, finalized spec/snapshot/receipts, and fresh output/socket/state. The code now enforces both previously deficient gates. Scheduler state was not rechecked.

Initially reviewed controller SHA-256: `dd6302efaa5c02b8cf1cc04d8ed5a11d071eb80515955ccd7517e3e38659fa3e`. Current hash is listed in the re-review above.

## Requested follow-up: parent finite-value guard

After the controller probes, inspected the parent's helper/test changes. Replacing only the new `finite_vector` function text with the previously observed version reconstructs the old helper SHA-256 exactly (`37bc56b0b48bce50f800d0a78ddb84eb2fb1ab40f65658b5ac6be45e1dfaf0d9`), confirming the production delta is confined to that function. It rejects booleans before `math.isfinite` and returns false on integer-conversion `OverflowError`. Projection math is unchanged. Tests cover `True` and `10**400` through both adapters. No concern found in this narrow correction.

Independently reran the source-tilt suite with bytecode/cache writes disabled: **61 passed in 1.18 s**. The parent's separate 166-test result is not counted as my execution.

| Current file | SHA-256 |
|---|---|
| `fastfill/adapters/upright_bounds.py` | `2e750b661fe66e7c0ffd094534b82405714de92bc3f5304887518fb60dec1c9f` |
| `fastfill/tests/test_source_tilt_correction.py` | `9a2f82645edb739d86acbb67590b0a58d937a5e35ac50cbbef47e8e941f90a71` |
| `fastfill/adapters/scan2cad.py` | `aef249703105ecf5003be6b3e31bc4ffcdd2492bf96bf6f7b529a29b5987688e` |
| `fastfill/adapters/spatialgen.py` | `57f4642a09420230bb568e73dbea484550e99a286b90a65c0b57a01226ae6826` |

The original source-correction receipt describes the earlier 57-test snapshot. This addendum records the parent-amended 61-test snapshot without rewriting that historical receipt or regenerating a corpus.
