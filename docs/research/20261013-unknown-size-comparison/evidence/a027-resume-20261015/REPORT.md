# Amendment 027: recovery of strict-trigger test shard 1

Recovery requested by the user on 2026-10-15. This is an operational restart of the existing experiment, with no change to model weights, inference parameters, size priors, requests, or selection rules. It uses local O2 only; API calls remain disabled.

## Failure and preserved results

- Original shard 1 job `7449644` failed on `nid011109`: CUDA device 1 had 86.26 GiB free, below vLLM's original 87.4 GiB startup budget (0.92 × 95 GiB).
- Shards 0 and 2–7 completed. Their identities, completion markers and ordered chunk receipts were validated against the original plan. Their file hashes, plus plan and requests, are recorded in `preserved.json`.
- Shard 1 contains 775 of the 6,155 escalated rooms. The full evaluation contains 8,647 rooms.
- Original finalize `7449651` and summarize `7449652` were cancelled by their failed dependency.

## Restart

- Unchanged code snapshot: `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/snapshots/main-498348274`.
- External wrapper directory: `/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill/ops/a027-resume-20261015-v1`.
- New chain: shard `7454202` → finalize `7454203` → summarize `7454204`. Exact submission commands are in `jobs.json`.
- Root submitted held and released at **2026-10-15 19:39:50 UTC** after independent review. Both successors require `afterok`; failed dependencies cancel them.
- Original GPU allocation preserved: 2 GPUs, 72 CPUs, 200G memory, 12 hours. Failed node excluded. Each GPU must have the original vLLM budget plus 0.5 GiB free before the original script starts. No GPU utilization setting was lowered.
- Shard `7454202` is running on `nid010357`. At 19:41:17 UTC both GPUs passed preflight: 94.448 and 94.446 GiB free versus 87.4 GiB required. The original environment reports torch 2.13.0+cu130, vLLM 0.31.0 and transformers 5.17.0.
- At **19:46:01 UTC**, engine initialization completed successfully (130.99 seconds for profiling, KV cache and warmup). Both workers loaded weights (25.24 GiB each), compiled and captured CUDA graphs; the runtime then entered chat request processing. The optional FlashInfer allreduce fusion failed compilation and was automatically disabled by the unchanged runtime; engine initialization succeeded afterward. No application setting was changed to bypass it.
- First-batch results are not yet claimed: the existing harness disables its progress bar and writes receipts after a whole chunk. The other seven shards took approximately 2h39–3h16 each; allow roughly three hours for this shard, plus CPU merging/statistics and queue time.
- A read-only `srun --jobid=7454202 --overlap ... nvidia-smi` probe at job elapsed 7m37s observed **100% GPU utilization on both allocated cards**. The shard was RUNNING; both CPU successors were PENDING on their intended dependencies.

## Validation

The remote CPU validation passed before submission and again inside the allocation. It checks pinned source hashes, the original request/prior binding, all seven completed shard identities and receipts, and preservation of completed files. Shell syntax and Python parsing checks passed. Darwin independently reviewed both the resume approach and external scripts, reporting no release-blocking bugs.

Before finalization, the wrapper validates all eight completion markers and receipts. After summarization, it validates room coverage, zero API calls, unchanged furniture sizes, paired comparisons with the old trigger, cohort replay comparison, and evaluation manifest hashes. A successful check writes `verification.json` in the server wrapper directory.

The final output remains `runs/unknownsize-20261013/system-022/final/test-strict-sharded/{system,eval}`. This report does **not** claim those outputs are already complete. Include the original failed job and retry in Slurm GPU cost accounting; invocation-only costs can omit failed startup time.
