# Independent trainer review — 2026-10-13

**Disposition: no blocking issue found for the bounded one-GPU, eight-optimizer-update-per-arm engineering pilot. P1: none. P2: one nonblocking termination-evidence concern below.** This is a source review, not evidence that the GPU run completed or fits its allocation.

Scope: `fastfill/comparison/train_pair.py` and `train_pair_helpers.py`, with narrow reads of the invoked contract, existing encoder, installed library implementation, and parent's existing server/base receipts. No nested agents, external network, dataset scans, model loading, training, test-suite runs, or repository edits. Only this requested report was written. Graph discovery returned no matching indexed code; direct reads were used. Full-training methods and coverage work were excluded.

The trainer changed concurrently during review; the updated model-load hash check, pre-loss token counting, and per-arm progress status were re-read. Reviewed snapshots:

- `train_pair.py`: SHA256 `9bb822b6a5f3f5f537393cd6916bff06b7273d88a04d59e1f69179307a9c737f`
- `train_pair_helpers.py`: SHA256 `fd03704c1e0857e140b6de3ce6cab51eece3728d6a6608d2ba4d269f45a75baf`

## Findings

### 🟡 P2 — External termination has no terminal driver receipt

**Location:** `train_pair.py:340–350, 359–365`; worker failure recording at `320–323`.

**Trigger:** The proposed shared GNU timeout reaches 27 minutes, or Slurm terminates the job while an arm is running. Neither process installs a SIGTERM handler; the `except Exception` paths and Python `finally` cleanup are not executed by default SIGTERM termination. Root status remains `UNVERIFIED` with the last arm, and final SHA/failure evidence may be absent. The per-arm phase update improves progress visibility but does not record termination.

**Impact:** The artifact directory alone cannot distinguish an active run from a terminated one, or identify timeout versus another external kill. This does **not** turn an incomplete arm into engineering success. The proposed process-group timeout with `--kill-after=30s`, plus Slurm's 30-minute allocation limit, provides the intended execution bound; this is not a launch blocker under that envelope.

**Suggestion:** Have the parent preserve the timeout/process exit status and Slurm terminal state in a separate termination receipt and require those alongside complete driver evidence. Preserve group-wide termination; killing only the top-level Python PID does not guarantee worker cleanup. Do not interpret an adapter directory or `UNVERIFIED` receipt as completion. No additional trainer test campaign is needed for this operational integration point.

## Production-path checks

- **Eight updates and crash handling — `train_pair.py:45–58, 189–234, 290–323`.** Default budget is eight optimizer updates, passed as Trainer `max_steps`, with batch 1 and accumulation 4. Global-step equality, actual microbatch finite loss, aggregate finite loss, and finite logged metrics are checked before saving a successful adapter. Runtime requires one visible BF16-capable GPU and world size one. Worker failure prevents starting the next arm. No API mismatch was found against locally installed Transformers 5.14.1, PEFT 0.21.0 and Accelerate 1.15.0, matching the parent's server receipt. Local torch is 2.13.0; server receipt specifies 2.13.0+cu126.

- **Memory release — `train_pair.py:143–149, 231–234, 248–287, 305–319`.** Training releases Accelerator references and deletes Trainer/model; the caller then collects cycles, empties the CUDA cache, synchronizes and refuses reload if allocator-tracked tensors remain. No surviving model/tensor reference was identified in returned metrics. Generation drops per-example tensors and its model/base references before the same release check. Each arm additionally exits its own spawned process before the next starts. Actual zero-allocation checks and peak-memory measurements remain GPU-run evidence, not something established by source inspection.

- **Matched UIDs and assistant loss — `train_pair_helpers.py:125–226`.** Exact paired UID sets, split/group separation, request equivalence apart from movable size, and the pilot's intentionally identical placement targets are enforced. Both arms use the same sorted common admission list, dropping a UID from both if either exceeds the limit, without truncation. Shared data seed and identical sample indexing support matched sampling. Prompt labels are -100; assistant answer and Qwen im_end EOS are supervised. The collator pads labels with -100. The updated loss override counts labels before delegating to Trainer and forwards its loss-normalization arguments.

- **P dimensions and evaluation — `train_pair_helpers.py:243–291`; `train_pair.py:263–285`.** Placement sizes come exclusively from each request object's prior; missing/invalid priors fail validation. There is no GT dimension fallback. Generation uses only system/user messages, reloads the saved tokenizer, and retains invalid predictions as failures without repair. The exact contract validator is also invoked. Zero valid generations may still yield ENGINEERING_COMPLETE, consistently with `quality_claim=False`; that status certifies execution/reload, not quality.

- **Base identity and SHA — `train_pair_helpers.py:294–327`; `train_pair.py:115–123, 152–162`.** Initial admission streams full pinned shard hashes, checks config/index, revision receipts and tokenizer/generation payload receipts, and records hashes. Every subsequent model load re-verifies the recorded base files, including the second arm and fresh-base probes. Expected shard hashes agree with the parent's existing payload-verification JSON; this review did not rehash model data. Repeated full-base reads count against the single shared wall-time budget.

- **Output refusal and reload — `train_pair.py:237–260, 326–365`.** Existing output roots are refused before modification. There is no automatic resume/retry. Saved adapter/tokenizer/trainer-state files must exist and be nonempty, are hashed and verified, then the saved adapter is loaded onto a fresh generic base for two dev generations. Parent verifies each arm's completion fields and artifact hashes before continuing. The saved artifact is an adapter plus trainer state, not a resumable optimizer checkpoint; resume is explicitly unsupported.

## Validation and next step

Both reviewed files parsed successfully with Python AST. Library source checks covered loss forwarding, list-dataset sampling/collation, save behavior and Accelerator cleanup; no new tests were written or duplicated. The repository's referenced `docs/CODEX-NAVIGATION-GUIDE.md` is absent.

Proceed within the parent's existing bounded pilot envelope, retaining scheduler/timeout evidence. The separation of fresh arm processes, strict common admission, and checkpoint reload gates is sound for this engineering scope. GPU compatibility, actual memory release, successful reload and completion within the shared timeout still require the authorized pilot itself.
