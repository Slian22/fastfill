# Standalone API evaluator review — 2026-10-13

One consequential P2 defect was reproduced and fixed. No other reproducible P1/P2 correctness issue was identified in the reviewed paths. The existing immutable attempt records, exact request identities, terminal SSE checks, and fixed diagnostic codes provide useful protection against duplicate charges and invalid evidence.

Scope: `docs/research/20261011-v11-clean-dpo/paper-metrics-20261013/server_api_judge.py` and `test_server_api_judge.py` only. Those are the only repository files edited by this review. Other agents’ changes were preserved. This report and coverage data are in `/tmp`. No HTTP requests, paid calls, remote jobs, or benchmark batch were run; no private server credential file was read. Transport was mocked in offline evaluator tests.

## Fixed finding

🟡 **P2 — Valid fractional overall scores became permanent failures.** Before the fix, `parse_scores` applied the strict integer check to `overall_score` as well as the three required components, and `INSTRUCTIONS` explicitly required an integer overall score. A completed response with three valid integer components and an average such as `22 / 3` was marked `invalid_score_value`; its persistent failed reservation prevented reuse/retry and stopped the batch.

The parser now accepts optional `overall_score` as an integer or float in inclusive `[0, 10]`, excluding booleans. Non-finite values, including JSON exponent overflow (`1e400`), fail validation. Functionality, layout, and aesthetics remain strict integers in inclusive `[0, 10]`; comments remain optional strings. The prompt explicitly permits a fractional average. No score is rounded, synthesized, or substituted for a component. Relevant current source: `server_api_judge.py:68` (prompt) and `server_api_judge.py:295` (schema).

Two regression tests were added: numeric overall boundaries/averages (`test_server_api_judge.py:64`) and a full mocked delta/completed SSE evaluation with fractional overall, persistence, and byte-for-byte unchanged reuse without a second transport call (`test_server_api_judge.py:230`). The existing invalid-score test was updated to keep strict component validation and reject invalid overall values.

## Reviewed contracts and limits

- **Krill SSE:** Supports `response.output_text.delta` followed by `response.completed`, including multiline SSE data, CRLF, Unicode, and keepalives. A terminal completed status, final assistant output, matching deltas, and exact returned model are required. Missing completion, refusal, incomplete/failed responses, malformed scores, and mismatched aliases cannot produce successful scores. The new end-to-end offline regression exercises both specified event types through `evaluate`.
- **Provenance:** Manifest model, config model, request model, and response model must agree exactly. Local `api-connectivity.json` records requested and returned `gpt-6.1-sol` at `/codex/v1/responses`, HTTP 200, completed; it explicitly establishes only text connectivity and a provider-reported alias. These offline fixtures do not prove visual quality, model internals, or equivalence to the paper’s judge. No alias normalization or model substitution was introduced.
- **Secret output:** Reviewed transport headers, fixed failure codes, count-only CLI output, credential redaction of stored output, and owner-only config/result files. Existing tests exercise credential echoes in output/comments, HTTP error bodies/headers/messages, and transport exceptions without exposing actual credentials. No new raw logging was added.
- **Reuse and failures:** All selected artifact hashes and existing records are preflighted before a new API call. Request hashes cover endpoint, exact model, instructions, and payload. Exclusive immutable attempt reservation precedes each call; matching completions reuse unchanged, while failed, unfinished, corrupt, or incompatible records stop explicitly. New failures stop later cases, retain evidence, and omit scores. No automatic retries, redirects, or ambient proxies.
- **Bounds:** Default selection is the first one manifest case; `--limit` includes reused cases, and the manifest permits at most 10,000 cases. Each request has a POSIX main-thread wall deadline (default 90 seconds), one transport open, and `max_output_tokens=4096`. Byte limits are 4 MiB manifest, 1 MiB prompt, 32 MiB per image, 64 MiB request, and 4 MiB SSE; at most 16 images. Preflight retains identities instead of all image payloads. Returned result records still scale with selected case count; byte limits are not a whole-process RSS ceiling. These are request/token controls, not a currency-denominated spend cap; no pricing or additional policy mechanism was added.

**Compatibility consequence:** Changing `INSTRUCTIONS` changes the request hash. Existing results from the old instruction text remain deliberately incompatible and are not rewritten or silently pooled. The parent experiment plan should freeze this instruction/schema version alongside the rubric, renders, selection, and exact provider alias before a batch. Keep the three component metrics and failure counts explicit in comparisons.

## Checks

Baseline: 39 offline tests passed. The new tests first reproduced the old bug (fractional/float overall rejection and failed end-to-end result), then passed after the fix.

Final: **41 offline tests passed; 88% branch-inclusive coverage** of `server_api_judge.py` (449 statements, 41 missed; 238 branches, 41 partial). Both files parse successfully; the final diff against pre-review snapshots is limited to the prompt, score validation, and associated tests.

```sh
PYTHONDONTWRITEBYTECODE=1 COVERAGE_FILE=/tmp/agency-api-client-review-20261013.coverage python3 -m coverage run --branch --source=server_api_judge -m unittest discover -s docs/research/20261011-v11-clean-dpo/paper-metrics-20261013 -p test_server_api_judge.py -q
COVERAGE_FILE=/tmp/agency-api-client-review-20261013.coverage python3 -m coverage report
```

The codebase graph had no index for this repository, so review used direct reads of the two explicitly named files and the local sanitized connectivity record. No network-dependent indexing or documentation lookup was performed. The remaining next step belongs to the parent’s fair-comparison design; this review did not launch the benchmark.
