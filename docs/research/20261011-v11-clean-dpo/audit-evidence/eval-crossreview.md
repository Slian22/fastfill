# Independent evaluation cross-review

No deployment-blocking issue found. No production or test files changed.

Fully read verify_outputs.py, evaluate_v1_1.py and their two relevant test files; exact hashes and line counts are in eval-crossreview.json. Focused existing tests: **46 passed**.

A separate integration smoke extracted the original 3e3eef72c515 snapshot, confirmed that it has no verify_outputs.py, created valid frozen fixtures with that old scorer/protocol, then launched the new verifier by absolute script path with the old frozen root and original interpreter, using two spawned workers. All four cells were independently scored successfully; cached_cells=0. Evidence: oldscore-cli-crossreview.json.

The verifier recomputes all scoring fields from persisted raw text and rooms, then rebuilds metrics. It rejects altered but re-sealed scores, metrics, outputs, row identities, and extra scoring fields. Cache reuse checks the full bound artifacts, code, runtime, state and verifier hash. Start/failure states replace prior pass receipts for entered verification attempts, with final identity checks before complete receipts. Payload digests protect both generated and scored resumable chunks; they do not authenticate maliciously replaced and re-signed generations.

Operational constraint: new chunk validation deliberately rejects old chunks without digests. Keep existing outputs attached to the old frozen evaluator and use the external CPU verifier to validate them; do not resume old partial chunks under newly frozen evaluation code. Independent rescoring proves score reproducibility, not model quality or generation authenticity.
