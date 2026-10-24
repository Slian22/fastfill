# Server model-identity failure diagnosis

**F17: a production identity-check weakness was confirmed and fixed locally.** The previous process-wide model hash cache assumed that an unchanged stat tuple implied unchanged bytes. A same-length weight overwrite with an unchanged stat tuple was accepted by `guard_training`; directly hashing the file detected the change. The shared cache has now been removed. Deterministic RED tests failed five times before the fix; the four complete related modules now pass all 124 tests. Server validation of the final source remains owned by the root agent.

The initial diagnosis was read-only. The root agent then stopped the previous full-suite process and authorized the bounded F17 source/test repair below. A subsequent [server stat probe](identity-stat-probe.json) confirms **5 of 5 missed changes**: before/after `(dev, inode, size, mtime_ns, ctime_ns)` are exactly equal, the recorded mtime/ctime values are whole seconds, fresh SHA-256 values differ, and the cached identity reports the old digest. This establishes the actual identical-signature collision on the server filesystem. The evidence does not determine whether filesystem timestamp granularity or attribute caching accounts for all such observations. The deterministic local regression models the observed condition without depending on its platform-specific cause.

## Observed server failures

The [second server log](server-attempt2/pytest.log) reports **8 failed, 389 passed, 2 skipped, 67 subtests passed**. Four are model identity failures:

| Test | Actual failure | Relevant path |
|---|---|---|
| `test_training_guard.py::test_resume_rejects_changed_training_identity[weights]` | No `ValueError` after replacing the last four weight bytes with `1234` | `guard_training` → `model_identity` |
| `test_training_guard.py::test_merged_output_must_match_exact_base_adapter_and_output` | `merge_complete` returns `True` after weight corruption | `merge_complete` → `model_identity(out)` |
| `test_automation_lineage.py::test_lineage_rejects_independently_corrupted_evidence[wrong_output]` | Corrupted output is accepted | `verify_training_lineage` → `model_identity(merged)` |
| Same test, `[wrong_base]` | Corrupted base is accepted | `verify_training_lineage` → parent `model_identity` |

These fixtures write actual new bytes. Their `wrong_adapter`, pair, and code counterparts pass and use `file_identity` directly, which reads the bytes on every call. The cache-specific failure pattern supports the diagnosis. The three missing-matplotlib failures and one copied-read-only-fixture failure are separate and handled by the root agent.

## Source mechanism and scope

At diagnosis, `fastfill/v1_1/evaluation_protocol.py` is unchanged from HEAD and has SHA-256 `750c0582c44e0b07f7646a8a486ec9d3104d496bd3f5c1c94085c106c41699b8`. Its last recorded commit is `051726074`. This cache predates the current repair patch.

- `file_identity` (lines 38–49) streams bytes into SHA-256. It has no hash cache.
- `_model_file_digest` (lines 135–140) is decorated with `@lru_cache(maxsize=512)`.
- `_model_file_identity` (lines 143–147) uses `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)` as the cache signature.
- `model_identity` (lines 150–171) uses that cached helper for model weights and assets.

On a cache hit, no weight bytes are read and no safetensors structural validation runs. Adding ctime to the key cannot be the fix: it is already present. Small files and rapid same-length writes make collisions easier to exercise, but the identity contract applies to real model files as well.

The affected shared entry point is used by guard/resume validation, the direct trainer identity, merge receipts/completion, lineage checks, and evaluation identities. A one-callsite `cache_clear()` would leave other validation boundaries exposed. A fresh process or complete uncached rehash can establish identity independently; this finding alone does not demonstrate corruption in the previously verified P/J training artifacts.

## Deterministic local reproduction against the pre-fix source

Executed before the repair in a separate Python process with temporary files, no production/source edits, no network, and no GPU. `Path.stat` is pinned only for the one temporary weight file to emulate an unchanged metadata signature. The actual bytes are overwritten, and `file_identity` reads them normally. This archived reproduction references the removed cache helper and is intentionally a reproduction of the pre-fix source; the regression tests below validate the current implementation.

```python
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from fastfill.v1_1 import evaluation_protocol as ep
from fastfill.v1_1.training_guard import guard_training

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary).resolve()
    model = root / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (model / name).write_text("{}")
    weight = model / "model.safetensors"
    header = json.dumps({"weight": {
        "dtype": "F32", "shape": [1], "data_offsets": [0, 4]
    }}).encode()
    weight.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
    pairs = root / "pairs.jsonl"
    pairs.write_text('{"uid":"a"}\n')
    code = root / "train.py"
    code.write_text("# fixture\n")
    state, original_stat = weight.stat(), Path.stat

    def pinned_stat(path, *args, **kwargs):
        return state if path == weight else original_stat(path, *args, **kwargs)

    ep._model_file_digest.cache_clear()
    with patch.object(Path, "stat", pinned_stat):
        before = guard_training(model, pairs, root / "run", {"epochs": 2}, [code])
        weight.write_bytes(weight.read_bytes()[:-4] + b"1234")
        fresh = ep.file_identity(weight)
        cached = ep.model_identity(model)["files"]["model.safetensors"]
        assert fresh != cached
        # Incorrect acceptance under the current cached implementation:
        assert guard_training(model, pairs, root / "run", {"epochs": 2}, [code]) == before
        ep._model_file_digest.cache_clear()
        try:
            guard_training(model, pairs, root / "run", {"epochs": 2}, [code])
        except ValueError:
            print("changed weights rejected after cache clear")
        else:
            raise AssertionError("uncached validation should reject changed bytes")
```

Observed diagnostics:

```json
{
  "old_sha256": "56f6037b61405efc1b41070108470fa70cf2e293dd6280b31a1b89049476b271",
  "fresh_sha256": "b60d993f66340190e518f203b02fa3ceba666c4ecd3f6b1db6e8cb2324a8cc6e",
  "cached_sha256": "56f6037b61405efc1b41070108470fa70cf2e293dd6280b31a1b89049476b271",
  "fresh_bytes_changed": true,
  "guard_incorrectly_accepted_changed_weights": true,
  "cache_info_before_clear": "CacheInfo(hits=8, misses=4, maxsize=512, currsize=4)",
  "guard_rejected_after_cache_clear": true
}
```

## Implemented minimal repair

1. Removed the cross-call stat-key digest cache and unused `lru_cache` import from the shared `model_identity` path. `_model_file_identity` now validates safetensors structure and calls the streaming `file_identity` on every invocation. The identity schema is unchanged.
2. Added three deterministic protocol regressions: same-stat, same-length weight changes; same-stat tokenizer changes; same-stat malformed safetensors headers. Strengthened the existing guard-weight and merge-output tests with the same pinned-stat condition. They modify actual temporary file bytes and compare against a fresh direct hash; they do not sleep or change timestamps to avoid the problem.
3. Ran the complete protocol, training guard, automation lineage, and DPO checkpoint recovery modules. Existing lineage `wrong_base` and `wrong_output` cases pass locally. Their original server RED evidence is retained, and the root agent owns the final same-filesystem server GREEN verification.

The cost is repeated full-model reads at verification boundaries; streaming memory remains bounded by the existing 8 MiB block. Explicit reuse of a validated identity within one controlled operation can be considered separately, but a stat tuple alone cannot authorize reuse across independent byte-verification calls.

## Repair validation and frozen files

[f17-red.log](f17-red.log): **5 failed in 4.64 seconds**, before production edits. The failures cover changed weights, changed tokenizer, invalid weight header, training guard acceptance, and merge acceptance under unchanged metadata.

[f17-green.log](f17-green.log): **124 passed in 7.93 seconds** after the minimal shared-helper fix:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/slian/miniforge3/bin/python -B -m pytest -p no:cacheprovider --import-mode=importlib -q \
  fastfill/v1_1/tests/test_evaluation_protocol.py \
  fastfill/v1_1/tests/test_training_guard.py \
  fastfill/v1_1/tests/test_automation_lineage.py \
  fastfill/v1_1/tests/test_dpo_checkpoint_recovery.py
```

An additional isolated unchanged-input comparison executed the pre-fix HEAD module and repaired module on the same four-asset model fixture. Their complete `model_identity` dictionaries are equal: top-level keys remain `files`/`path`; every asset still has `sha256`/`size_bytes`. No historical guard or manifest was rewritten. Scoped `git diff --check` passes.

| Frozen file | SHA-256 |
|---|---|
| `fastfill/v1_1/evaluation_protocol.py` | `3747133048d56591cb71ad95c040141a00b2051907b34e932cd00516bba5aef2` |
| `fastfill/v1_1/tests/test_evaluation_protocol.py` | `4d5ae134dae6b3839548ac79ac1a6660e15a2def168bec4e563a316b01925f9e` |
| `fastfill/v1_1/tests/test_training_guard.py` | `638cfe4edcf3ed1f40ee11cdea0b57094be7b9089635c8a2fac04535fb6b75de` |

These are the only source/test files edited by the F17 repair worker. No SSH, API, GPU, new training, dependency change, commit or push was performed by that worker.
