# Amendment 003 — open-weight LLM agents (vLLM) as same-input baselines

Locked 2026-10-13 ~18:30 UTC, before any J0/P0, API-agent or open-model output exists. User clarification 2026-10-13: the final delivery is **one model plus a benchmark against other methods on the same input** — the existing API agent (`gpt-6.1-sol`) and open-weight models the server can run with the user's vLLM environment.

## Systems (all receive the identical J request: same system + user messages, same fixed JSON-only suffix as the API agent)

| ID | Model | Where | Mode |
|---|---|---|---|
| O1a | Qwen3-8B (same base revision as J/P), `fastfill/models/Qwen3-8B` | 1 GH200 | zero-shot, `enable_thinking=False` (identical template to J/P: isolates the effect of fine-tuning) |
| O1b | same | 1 GH200 | zero-shot, `enable_thinking=True` (agent mode; thinking text stripped before parsing) |
| O2 | Qwen3-VL-32B-Instruct (HF cache, text-only use) | 2 GH200, TP2 | instruct |
| O3 | Qwen3-VL-235B-A22B-Instruct-FP8 (`fastfill/models/...`) | 4 GH200, TP4 | instruct |
| O4 | gpt-oss-120b — **conditional**: only if a download job completes and a 2-room smoke passes; otherwise reported as "not run" with the reason | 1–2 GH200 | reasoning medium |

- Engine: vLLM offline batch inference from the server env `~/miniforge3/envs/vllm` (vLLM 0.31.0, torch 2.13 cu130), inside Slurm jobs; offline HF; `max_model_len` 40,960 (larger only if the model's limit and memory allow, recorded).
- Decoding: greedy (`temperature=0`), fixed seed; output cap per request = the same input-derived cap as amendment 002 for non-thinking modes; thinking modes get an extra fixed 16,384-token thinking allowance.
- Harness identical to the API agent: same parser as J (`parse_prediction(raw, record, 'joint')`), up to 2 repair rounds that return only the parser error text. Variants reported separately: `raw` (first answer) and `harness` (after repair).
- Cohort: the same frozen LLM cohort as amendment 002 (all 1,009 six-strata rooms + up to 30 per source by stable hash). J and P are compared with every agent on exactly this cohort, paired.
- No size tables, priors, statistics or per-request extra information for any agent. No prompt tuning on test rows; the suffix is fixed before any call.

## Tables and visual metrics

- Geometry/size table (cohort, paired vs the delivered model): complete rate, geometry success (all-request), size IoU, size log MAE, raw and repaired OOR, failure taxonomy, cost (GPU-h or API calls, latency, tokens). Raw and harness variants side by side.
- Six-strata OOR for every system. FID (local renders, same assets/cameras) for every system on the six strata where renders succeed. GPT judge (174 rooms, pooled paired difference inferential) for J, P, API-agent harness and O3 harness; judge cap raised to 800 calls.
- Delivery rule unchanged (amendment 002 §5): the delivered model is J or P; agents are comparators only.

## Budget

GPU cap for this protocol raised from 200 to 260 GPU-hours to cover open-model runs (expected: O1a/O1b ~2 GPU-h each, O2 ~4, O3 ~8–16, O4 ~4).
