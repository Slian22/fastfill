# Root-owned repair evidence

Base commit: `873251f0b1f299230c42d510fe5c4f732aa77ce9`.

| Finding | Confirmed failure | Repair and regression |
|---|---|---|
| F02 | Batch answers omit identity required by `critic_025.round_` | Preserve model and frozen case-content identity in every answer, including failed calls. A real batch-to-round integration test covers success, transport failure and stale identity. The result dictionary is no longer mutated by `pop`. |
| F08 | A stalled request blocks the serial model server | Bound the complete request-header/body read by one absolute deadline; repeatedly recompute the remaining time so a slow stream cannot reset the budget. Keep model generation serial and allow generation to exceed the input deadline. Real loopback tests cover idle and trickling headers/body, subsequent requests, and serial generation. |
| F11 | QA boundary repair omits the original constraints | Pass the same constraints and room boundary as runtime. Invalid constraint declarations retain their diagnostics and are not repaired as unconstrained input. A `faces` example now produces the same refusal as runtime without mutating the input. |
| F14 | Agreement report crashes on optional or insufficient scores | Report per-metric sample counts; undefined correlation, ranking and margins are `null`. Empty, one-row, constant and missing-overall cases serialize as strict JSON. Existing complete-score behavior remains covered. |

TDD: [root-red.log](root-red.log) records **11 failed, 2 passed** before the production changes. [root-green.log](root-green.log) records **80 passed and 90 subtests passed** across the affected eight modules after the fixes. Tests run with Python 3.12.9; fake API transports make no paid requests, and HTTP checks use loopback sockets.

An independent reviewer inspected all four changes and their tests and found no unresolved blocker. Full-suite and deployment results are recorded separately in the final repair report.

## Historical O2nt impact

The server receipt scan covered dev-select (600 rooms), confirm (3,000), and cohort (1,454). Four final answers carry `reasoning_in_no_reasoning_system`: three confirm rooms and one cohort room. Their first answers do not carry that marker.

The eight selected raw/harness records were joined to the original manifest-bound room and placement request. Both versions were scored after the same B2 size enforcement. **No metric changed:** each affected final answer was already rejected by the parser. Only its error attribution changed from a generic parse failure to the preserved semantic failure. This evidence does not claim a fresh full-cohort rerun or a new model evaluation.

The comparison is in [historical-o2nt-delta.json](historical-o2nt-delta.json). Private input payloads remain outside the repository; the report records their hash and the source data manifest hashes. The original outputs remain untouched.

## Historical critic handoff

All 757 stored `answers/` records across four a025 batches lack the two handoff fields, while their immutable raw client results retain them. [recover_cached_critic.py](recover_cached_critic.py) verifies all source artifact hashes, active case identities, attempt/result identities and original answer text before writing a separate corrected answer bundle. It invokes no API and does not rerun geometry rounds. Actual recovery receipts belong in the final deployment evidence; historical rounds were executed by older consumers, so this format defect alone does not establish that historical rounds failed.
