# Amendment 007 — latest mainstream open-weight set (supersedes the model lists of amendments 003/004/006)

Locked 2026-10-13 ~20:10 UTC. No J/P output exists and no agent answer or score has been looked at (only API-agent progress counters). The user pointed out Qwen3.6 is outdated and asked for mainstream, moderately large models (not 2T-class flagships); multi-node is allowed by the server guide but not needed for this set. Every repo below was verified on the Hugging Face API on 2026-10-13 (existence, created_at, safetensors size, architectures vs the vLLM 0.31 registry, license, gating); evidence in workflow wf_94a1b68c-adf.

## Systems (identical J request + fixed suffix + same parser/repair harness, greedy, seed fixed)

| ID | Repo (created) | GPUs | Mode |
|---|---|---|---|
| O1a | Qwen/Qwen3-8B (2025-04) | 1 | thinking off |
| O1b | Qwen/Qwen3-8B | 1 | thinking on |
| O2 | Qwen/Qwen3.8-27B (2026-08-05), BF16 55.6 GB | 2 | thinking on (default) |
| O3 | google/gemma-4-31B-it (2026-03-11), BF16 62.5 GB | 2 | thinking off (instruct default) |
| O4 | openai/gpt-oss-120b (2025-08), MXFP4 65 GB | 2 | reasoning effort medium |
| O5 | deepseek-ai/DeepSeek-V4-Flash-0731 (2026-07-31), FP8/FP4 167 GB | 4 | thinking on, reasoning effort **high** (the model offers low/high/max, no medium) |
| O6 | Qwen/Qwen3.8-Flash-Next-FP8 (2026-08-24), 180B total / ~6B active, 185.6 GB | 4 | thinking on, reasoning effort **medium**; TP4 + expert parallel, triton MoE backend on Hopper |
| O7 | meta-models/Muse-Glimmer-30B (2026-08-09), BF16 59.6 GB | 2 | model default; no "Reasoning strength:" system line (that would be extra input) |

Qwen3.6-27B (superseded by Qwen3.8-27B) is dropped and never run. Run order (one at a time, after all inference v2 GPU jobs): O6, O5, O4, O2, O3, O7, O1a, O1b. Smoke rules of amendment 006 apply to every system, including the fail-closed check that a thinking mode actually produced reasoning.

## Not run (reported in the tables with reason)

Too large for one 4×GH200 node in a Hopper precision: Qwen3.8-2.4T-A95B, DeepSeek-V4.1-Flash (510 GB) and V4-Pro, GLM-5.2/5.3 (755 GB FP8; GLM-5.3-Flash FP8 328 GB leaves no KV headroom), Kimi-K3/K2.6, MiniMax-M3, MiMo-V2.6-Pro, Mistral-Large-3, Nemotron-3-Ultra, Llama-4-Maverick. Gated without access: Llama-4-Scout (HTTP 403). Fit but not selected (lower adoption/scores or custom vLLM builds): MiMo-V2.6-Flash, MiniMax-M2.7, Step-3.7-Flash, Hy3, Mistral-Small-4, Mistral-Medium-3.5, Nemotron-3-Super. Phi: context too short.

## Visual judge

The open model sent to the GPT judge is the strongest successfully run open model: O6, else O5, else O4. Other systems get FID only.

## Budget

GPU cap stays 400 GPU-hours (expected total ≈ 290 including O6 and O7).
