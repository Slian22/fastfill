Read-only bounded review — 2026-10-13

Reviewed `fastfill/comparison/data.py`, `size_prior.py`, their two test files, and `docs/research/20261013-unknown-size-comparison/protocol.md`. Included the current centimeter-rounding change in `data.py:187–188`. No repository edits, production-data scans, bulk tests, training, or evaluation jobs. `contract.py`, `train_pair`, and rendering remain outside this review. Small reads of existing preparation/split/geometry helpers established their caller contracts. The graph service has no index for this checkout, so direct file reads were used.

Decision: no P1 defect established in the requested engineering-pilot scope; one reproducible P2 hash-integrity defect. The OOF complement construction and its tests are particularly strong. The unchanged-pose P targets and selected-only fingerprint closure are explicitly scoped pilot limitations, but both need gates before main training.

🟡 **P2 — Hash the release bytes actually consumed by selection.**

Location: `fastfill/comparison/data.py:28–34`, with the stale receipt propagated at `data.py:212` and only the release MANIFEST rechecked at `data.py:226`.

`select_release_rows` verifies the file, closes it, then reopens the path to parse rows. A same-row-count replacement between those operations is accepted. The manifest can therefore attribute the selected inventory/flags/split membership to a SHA-256 belonging to different bytes. Unlike IR reconstruction, release selection has no post-read hash check. This requires concurrent modification/replacement; it is not evidence that the currently frozen release leaked labels.

Executed a deterministic temporary-file reproduction: hash a one-row release containing UID `original`, replace it immediately after the initial hash calculation with one row containing UID `replacement`, then select. Actual result: `{'replacement': 'A'}` and the current file hash does not match the recorded digest. The regression test below fails with `AssertionError: ValueError not raised`.

Consider hashing the exact byte stream used for parsing and comparing that digest with the release manifest before returning selection. At minimum, mirror the IR before/after check. The existing source-drift test changes the file before selection and does not exercise this gap.

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastfill.comparison import data

class ReleaseReceiptRaceTest(unittest.TestCase):
    def test_rejects_release_replacement_after_initial_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'train.jsonl'
            path.write_text(json.dumps({
                'uid': 'original', 'source': 'A', 'flags': {}}) + '\n')
            expected = {'sha256': data.sha256(path), 'lines': 1}
            real_sha256 = data.sha256
            calls = 0

            def swap_after_hash(candidate):
                nonlocal calls
                digest = real_sha256(candidate)
                calls += 1
                if calls == 1:
                    path.write_text(json.dumps({
                        'uid': 'replacement', 'source': 'A', 'flags': {}}) + '\n')
                return digest

            with patch.object(data, 'sha256', side_effect=swap_after_hash):
                with self.assertRaisesRegex(ValueError, 'hash|changed'):
                    data.select_release_rows(path, expected, 1, 'seed', True)

if __name__ == '__main__':
    unittest.main()
```

**Checks and limits of assurance.**

- Training prior input is constructed exclusively from selected train UIDs. Training P uses that UID's held-out-fold prior; dev uses the full training prior. Prediction reads category/room type and identity, not query dimensions. Synthetic perturbations of all labels in a held-out fold, and separately of dev dimensions, left the applicable predictions unchanged.
- Full raw group/alias connections are processed before selected-room filtering. A synthetic bridge through an unselected room correctly caused cross-split rejection. Supplied connected components stay together in OOF. The selected-only content/layout fingerprint boundary is accurately stated in the data manifest; it is not full-release closure.
- Reconstruction calls `prep` on original IR rather than canonical release targets. Synthetic reconstruction preserved `raw-chair` IDs and unchanged IR file bytes. Final opaque binding/input serialization belongs to the excluded contract review; this report does not certify that layer.
- Both arms are emitted from the same UID loop and have the same admission decisions. Geometry failures are audited without per-arm filtering. Preparation rejection, missing selected UIDs, and component overlap abort rather than silently dropping selected rooms. The existing paired-build test checks matching training UIDs and artifact hashes.
- Current estimates use two-decimal rounding and a 0.01 m floor, consistent numerically with prepared target sizes. The same rounded mapping feeds placement messages, counterfactual checks, and geometry audits. Keeping unrounded estimator buckets is compatible with this declared display policy. No new P1/P2 rounding defect found.
- Payload files are staged in a sibling temporary directory, hashed, and published by rename after validation; an existing output is refused. This supports atomic visibility of completed output. No crash-durability claim or independent contract/consumer audit is made.
- Tests were read in full; only the new hash-race test and small synthetic assertions were executed. No parent bulk checks were repeated. The synthetic assertions covered alias transitivity through unselected data, original IDs, raw preservation, own-fold/dev target independence, and centimeter quantization.

**Gate before main training; engineering pilot may remain unchanged.**

Parent-reported `data/pilot-v1/geometry-audit.json` totals for 965 training rooms: reference OOB 0 / support 0 / main collisions 289; OOF-size substitution at unchanged reference poses gives OOB 324 / support 386 / main collisions 452 / fixed blocking 56. These are audit violation counts, not necessarily counts of unique failing rooms. They were supplied by the parent and were not rescanned here. They show that unchanged reference poses are not generally consistent P supervision. The eight-step loading/gradient/provenance pilot can remain explicitly preliminary; the same target policy should not become the main quality comparison unchanged.

Minimum viable main-training policy:

1. Freeze the component/fold manifest and train-only OOF estimates. Preserve the centimeter-quantized estimated dimensions exactly during P target construction. Recalculate child Z from the updated parent top in support-chain order.
2. Apply deterministic, bounded pose-only repair/projection (position/yaw, preserving objects, IDs, support graph, and estimated sizes) to meet the declared geometry qualification. Support-Z correction alone cannot fix the reported boundary, footprint-support, collision, and fixed-blocking failures. Do not reuse the existing size-mutating v1_1 postprocessor unchanged.
3. Admit a training room to both J and P only when its reference J target and repaired P target satisfy the same declared qualification rule. Keep every exclusion/failure reason and raw/repaired audit. Define the collision rule before running it: the reference itself has 289 main-collision counts, so a blanket zero-collision standard would change the cohort substantially. P may require different valid poses because its dimensions differ; document the resulting supervision policy explicitly.
4. Freeze admitted membership and fitted priors consistently. If common admission changes the declared fitting pool, regenerate priors/targets and requalify before freezing. Guard against indirect OOF leakage from adaptive pruning: held-out labels must not influence which complement rooms fit that fold's prior. Keep fold assignments fixed, retain the final fit UID/component hashes, and repeat an end-to-end own-fold label perturbation check after admission. Preserve matched training budgets. Training admission must not become a post-generation evaluation filter: evaluation failures still belong in all-request denominators.

A minimal acceptance fixture has a reference parent height 1.0 m and child bottom Z 1.0 m, with estimated parent height 0.6 m. The new P target should place the child bottom at 0.6 m while keeping every estimated dimension and object identity unchanged. Include an infeasible layout that cannot be repaired within the bound: it must appear in the rejection ledger and be absent from both training arms. Assert byte-for-byte size equality before/after repair.

Main OOF additionally requires **full-release component closure**, including all released train/dev/test rooms and all raw group/alias edges, before selecting a training subset or assigning folds. Compute eligible released-room content/layout edges even for rooms outside the selected/admitted pool; otherwise an omitted intermediate room can bridge two selected houses. Preserve the existing historical split boundary and reject/resolve cross-split components without silently moving held-out rooms into training. Use held-out identities/fingerprints only for frozen leakage governance, never held-out dimensions for fitting or repair-policy tuning. A regression fixture should connect selected A and B through an unselected released U using different fingerprint edges; A/B must share an OOF fold, or trigger rejection if in different splits.

The build that loaded pre-rounding code remains a preliminary immutable artifact. Regenerate under a distinct output directory with the current code after that job completes; verify the new `estimated_size_precision` receipt field and payload hashes. No rerun was launched by this reviewer.

Reviewed file SHA-256 receipts (captured after the rounding edit):

| File | SHA-256 |
| --- | --- |
| `fastfill/comparison/data.py` | `511f70a153f76b1cbb99359fd26016cdfcb9ec9656737fbafb7906e18713f7d0` |
| `fastfill/comparison/size_prior.py` | `e7a546b8af793755f3aeb8bd074e78798858f5528ca615273d5f56eef5d61ffc` |
| `fastfill/comparison/tests/test_data.py` | `65f9e3e55e4e12f2e5ebe20c98ae79a97278e5e37a3d78491a2ddf043e0b07e0` |
| `fastfill/comparison/tests/test_size_prior.py` | `60072f98f8c59bac4507f143b85363df807e27748e154f1203db681030c3f093` |
| `docs/research/20261013-unknown-size-comparison/protocol.md` | `f2fc11ec56d927f551a438fd14e83ff4ca64e5e4a0e60d4b5e47968e7853706e` |

Close the reproducible hash race, then qualify the main-training target/admission and closure policies before scaling. The pilot's explicit scope and strong OOF complement tests provide a useful foundation.
