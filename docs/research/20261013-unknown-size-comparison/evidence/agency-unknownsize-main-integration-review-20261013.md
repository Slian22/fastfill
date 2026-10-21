# Independent main dataset integration review — 2026-10-13

**Result: two Medium contract findings confirmed.** The current synthetic builder output integrates successfully; the loader does not fully bind qualification evidence to the records it consumes. Fix both bindings before the pending main dataset build/training handoff. No claim is made about actual dataset quality or model performance.

Scope: `fastfill/comparison/main_data.py`, the main target-policy path in `fastfill/comparison/train_pair_helpers.py`, and `fastfill/comparison/eval_data.py`. Only these implementation files were inspected. Associated synthetic tests/fixtures were read and executed; imported dependencies were not independently audited. No repository edits, real-data build, training, model/API/server calls, or nested agents. Source reads began after the requested brief delay; the room-type prior fix was present. The graph search returned no matching indexed nodes, so direct reads were used.

## Findings

### F1 — Medium: qualified ledger target SHA is never compared with the loaded assistant target

**Location:** `train_pair_helpers.py:158–169`; relevant caller `:194–195`. Producer: `main_data.py:145–160`, `:275–284`.

**Observation:** `_main_admission` requires a hashed ledger, exact accepted train UID coverage, eligibility, and both arm qualification booleans. It neither requires `target_sha256` nor compares it with the assistant content. The producer already emits the correct SHA of the exact serialized target. `_check_pair` validates structure/support and intentionally permits different poses for the main policy; it does not repeat geometric qualification.

**Evidence:** Starting from the existing main-data synthetic fixture (six admitted train UIDs), change only T0's placement assistant position to `[500, 500, 0]` in a room bounded by `[0,20]` on both horizontal axes. Refresh the placement JSONL's hash in the root manifest, leaving the qualification ledger untouched. `load_pair()` succeeds. The stale ledger SHA is `d51606044a963e578fa7dd27fe4915fbb85a2363ecb9b748750ebe96bfd146a1`; the consumed target SHA is `b90d8431f335057953d41b6a16c59f3b805da888514284d2c5bd3ea34a30931b`. Removing all target SHAs from the ledger and refreshing its inventory hash also succeeds.

**Impact:** One of six synthetic train targets (16.7%) can be replaced with an out-of-bounds target while still being accepted as qualified. Production incidence is unknown; no real data was inspected. This concerns stale evidence after repackaging, partial regeneration, or incorrect artifact mixing. Ordinary file changes without refreshing the manifest are already rejected; this is a missing semantic cross-check, not a SHA256 failure.

**Minimal patch suggestion:** Index both train arms once, retain existing coverage/boolean checks, then compare each accepted UID/arm's exact UTF-8 assistant string with the ledger value. A direct equality check also rejects absent, malformed, or wrong digests:

```python
indexed = {arm: _by_uid(pairs['train'][arm]) for arm in ARMS}
for uid, entry in accepted.items():
    # Retain existing eligible/qualified checks here.
    for arm in ARMS:
        actual = hashlib.sha256(
            indexed[arm][uid]['messages'][-1]['content'].encode('utf-8')
        ).hexdigest()
        if entry[arm].get('target_sha256') != actual:
            raise ValueError(f'qualified target SHA mismatch: {uid}: {arm}')
```

Update `test_train_pair.py`'s main fixture to emit real target digests: its current success fixture omits them entirely. Add regression cases for missing/wrong digest and a syntactically valid pose change with the file hash refreshed. The existing happy-path builder test checks SHA equality itself, which does not test loader enforcement.

### F2 — Medium: placement request sizes are not bound to qualification evidence

**Location:** `main_data.py:145–160`, `train_pair_helpers.py:140–169` and `:308–312`.

**Observation:** Placement qualification depends on estimated sizes carried in the user request. The ledger hashes only the assistant target, which deliberately contains no placement sizes. `_check_pair` strips placement sizes when checking shared inputs; `parse_prediction` accepts any positive finite supplied sizes. Consequently, implementing F1 alone cannot enforce that the qualified geometry is the geometry consumed for training.

**Evidence:** Restore the original target, then change only T0's placement request size from `[1.35, 1.0, 1.0]` to `[100, 100, 100]`, refreshing only the placement JSONL inventory hash. `load_pair()` succeeds. Its assistant SHA still exactly matches the original qualification ledger. The ledger and prior remain untouched.

**Impact:** One of six synthetic placement requests (16.7%) can silently depart from the frozen OOF size assignment. A 100 m object is accepted in a 20 m room despite the earlier qualified-target flag. This breaks the fixed-size comparison premise and invalidates that record's geometric qualification. No such change was observed in unmodified builder output.

**Minimal patch suggestion:** Add a `request_sha256` to each arm's qualification result, computed from a canonical serialization of `turns[:-1]`, and verify it alongside F1 in `_main_admission`. In the producer, `_digest(_dump(turns[:-1]))` uses the existing canonical helper. In the consumer use the same `json.dumps(..., ensure_ascii=False, allow_nan=False, sort_keys=True)` serialization and UTF-8 SHA256. Include system and user messages so inventory, sizes, boundary, fixed context, and instructions are bound together. Rejected records can carry `None` where no turns exist. Add a regression that changes only a valid placement request size, rehashes its JSONL, and expects rejection with an unchanged target digest.

This preserves the stdlib-only loader and avoids geometry imports or refitting. It binds qualification to its actual inputs; it is not an independent proof that prior fitting was correct. Because the actual main build is pending, emit this field before that build rather than generating a new main artifact with incomplete qualification evidence.

## Verified behavior and limits

| Area | Evidence and assessment |
| --- | --- |
| Train/dev/test separation | `main_data.py:43–85,323–330` validates full identity component ownership before filtering, excludes heldout sources and their connected components, and selects exact eligible train identities. Synthetic cross-split/heldout bridge tests pass. No leakage found in the reviewed producer path. |
| Immutable OOF/fit pool | Fitting precedes decoder admission (`:106–129,330–338`). Train uses its assigned fold prior (`:218–225`); dev uses the frozen full prior (`:288–300`). A synthetic two-UID shared-component probe placed both in the same fold and excluded both from that fold's fitted UID hashes. Forcing T0's teacher rejection left both prior and fit-pool files byte-identical and reported T0 as extra fit exposure. |
| Room-type fix | Minimal observations now retain `room_type` (`:111`). The existing room-type regression passes: 25 fit rooms, one fitting call, bedroom/living-room conditioning, category/global fallback, and no query-time refit. |
| Matched UIDs / support | Builder admits both arms together. Loader requires identical per-split UID sets, shared metadata/input context, and identical non-pose/non-size target fields. Output parsing enforces ordered IDs, valid support references, and no support cycles. Existing support-mismatch and teacher inventory/resize/support/context mutation tests pass. |
| Frozen placement sizes | Builder rounds fold predictions once, passes those sizes to repair, and rejects size/support/context changes (`:175–192`). Producer protection works in synthetic tests; F2 concerns the later consumer boundary. |
| Dev policy | Dev is not repaired or geometrically quality-filtered; targets are explicitly unqualified. Optional `devlimit` selects by sorted UID. The manifest's `dev_policy` says “all_dev” even with this option, but `devlimit` and retained/total counts disclose the restriction; no separate material finding. |
| Eval handoff | Full main fixture → `load_pair()` → `eval_data.build()` succeeds: train=6, dev=2, fit=6, test=1. Eval binds corpus/identity hashes and fitted UID/component provenance (`eval_data.py:89–129`), uses the training full prior without refit, predicts from category/room-type features only (`:191–202`), and emits paired system/user requests with labels separate. |
| Eval population / manifests | Existing tests retain invalid-quality test rooms, verify all test UID reconciliation and deterministic selection, reject wrong hashes/provenance, and exercise the downstream inference-loader contract. No production inference was run. The main → eval probe confirms the actual producer manifest shape is accepted, beyond the separate handcrafted eval fixture. |

Eval currently treats `prior-fit-pool.json` as optional and validates the claimed fit population/provenance rather than reconstructing fitting or checking every OOF fold. The main builder emits the pool; no incompatibility or leakage was demonstrated for its output. This review does not certify arbitrary regenerated artifacts, dependency internals, or the pending real corpus. Lustre publication implementation is outside scope and remains with the parent.

## Validation and source snapshot

Executed from `/Users/slian/Desktop/3D/Worldedge/OptiScene`:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  fastfill/comparison/tests/test_main_data.py \
  fastfill/comparison/tests/test_train_pair.py \
  fastfill/comparison/tests/test_eval_data.py
```

**Result:** `89 passed, 62 subtests passed in 2.44s`.

Environment observed: Python **3.12.9**, pytest **9.1.1**, NumPy **1.26.4**, Shapely **2.1.1**, macOS **26.6.2**, arm64. No dependency installations or environment changes.

Implementation SHAs were unchanged between review checks:

| File | SHA256 |
| --- | --- |
| `main_data.py` | `f59db9d7bcac9eac5d40f2e9e95d28f2e48c57119836c507bd6d1b588977f1f7` |
| `train_pair_helpers.py` | `cb50aeb5540544b3c0caa1471715327857d1417408fc11e4434420b2b0e8401c` |
| `eval_data.py` | `ddd0ca5729739f4603db1e0eb5e53f851b1f02539563b663d944c418f087104b` |

## Reproduction v1 — generated synthetic data only

Run this block from the same repository root with the environment above. It uses the existing tiny fixture, creates all artifacts in a temporary directory, and modifies no source. The assertions reproduce both findings, missing target hashes, the full integration handoff, and admission-independent component folds/fit-pool bytes.

```sh
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from fastfill.comparison.tests.test_main_data import fixture, build, read, rows, refresh
from fastfill.comparison.train_pair_helpers import load_pair, sha256
from fastfill.comparison import eval_data, main_data


def put(path, data):
    path.write_text(''.join(json.dumps(r) + '\n' for r in data))


def rehash(root, name):
    manifest = read(root / 'MANIFEST.json')
    manifest = {**manifest, 'files': {**manifest['files'], name: sha256(root / name)}}
    (root / 'MANIFEST.json').write_text(json.dumps(manifest))


with tempfile.TemporaryDirectory(prefix='main-review-') as temp:
    root = Path(temp).resolve()
    corpus = fixture(root / 'corpus')
    main = root / 'main'
    build(corpus, main)
    _, receipt = load_pair(main)
    evaluation = eval_data.build(corpus, main, root / 'evaluation')
    assert receipt['counts'] == {'train': 6, 'dev': 2}
    assert evaluation['counts']['test'] == 1
    ledger = {r['uid']: r for r in rows(main / 'common-admission.jsonl')}
    name = 'placement/train.jsonl'
    original = (main / name).read_bytes()

    # F1: stale target qualification is accepted after refreshing inventory SHA.
    data = rows(main / name)
    first = data[0]
    target = json.loads(first['messages'][-1]['content'])
    changed_target = {'placements': [
        {**o, 'pos': [500, 500, 0]} for o in target['placements']]}
    changed_first = {**first, 'messages': first['messages'][:-1] + [
        {**first['messages'][-1], 'content': json.dumps(changed_target)}]}
    put(main / name, [changed_first, *data[1:]])
    rehash(main, name)
    loaded, _ = load_pair(main)
    actual = hashlib.sha256(
        loaded['train']['placement'][0]['messages'][-1]['content'].encode()).hexdigest()
    assert actual != ledger[first['uid']]['placement']['target_sha256']
    print('CONFIRMED F1: changed out-of-bounds target accepted')

    # F2: target SHA still matches after changing only frozen placement sizes.
    (main / name).write_bytes(original)
    data = rows(main / name)
    first = data[0]
    user = json.loads(first['messages'][1]['content'])
    changed_user = {**user, 'objects': [
        {**o, 'size': [100, 100, 100]} for o in user['objects']]}
    changed_first = {**first, 'messages': [first['messages'][0],
        {**first['messages'][1], 'content': json.dumps(changed_user)},
        first['messages'][-1]]}
    put(main / name, [changed_first, *data[1:]])
    rehash(main, name)
    loaded, _ = load_pair(main)
    actual = hashlib.sha256(
        loaded['train']['placement'][0]['messages'][-1]['content'].encode()).hexdigest()
    assert actual == ledger[first['uid']]['placement']['target_sha256']
    print('CONFIRMED F2: changed placement sizes accepted with matching target SHA')

    (main / name).write_bytes(original)
    rehash(main, name)
    missing = [{**r, **{a: {k: v for k, v in r[a].items() if k != 'target_sha256'}
                         for a in ('joint', 'placement')}} for r in ledger.values()]
    put(main / 'common-admission.jsonl', missing)
    rehash(main, 'common-admission.jsonl')
    load_pair(main)
    print('CONFIRMED F1: absent target SHAs accepted')

    # Separate baseline to test component folds and rejection independence.
    entries = read(corpus / 'identity-manifest.json')
    entries = [{**e, 'component': 'c:house:T0'} if e['uid'] == 'T1' else e
               for e in entries]
    (corpus / 'identity-manifest.json').write_text(json.dumps(entries))
    refresh(corpus)
    baseline, rejected = root / 'baseline', root / 'rejected'
    build(corpus, baseline)
    repair = main_data.repair
    def reject_one(room, sizes, **kwargs):
        result = repair(room, sizes, **kwargs)
        return {**result, 'status': 'rejected'} if room['uid'] == 'T0' else result
    with patch.object(main_data, 'repair', reject_one):
        build(corpus, rejected)
    oof = read(baseline / 'size-priors.json')
    fold = oof['fold_by_uid']['T0']
    assert fold == oof['fold_by_uid']['T1']
    held = {hashlib.sha256(u.encode()).hexdigest() for u in ('T0', 'T1')}
    assert not held.intersection(oof['priors'][str(fold)]['provenance']['fitted_uid_hashes'])
    for name in ('size-priors.json', 'prior-fit-pool.json'):
        assert (baseline / name).read_bytes() == (rejected / name).read_bytes()
    assert read(rejected / 'MANIFEST.json')['prior_exposure']['extra_fit_uids'] == ['T0']
    print('PASS: component OOF exclusion and admission-independent prior/pool bytes')
PY
```

Review completed within the requested five-minute window. Recommendations only; no patches applied.
