# Amendment 004 — mainstream open-weight comparison set (supersedes the model list of amendment 003)

Locked 2026-10-13 ~18:40 UTC, before any J/P/agent output exists. The user asked which open models are mainstream for comparison now and to use `gpt-6.1-sol` with medium reasoning for the API agent.

## Evidence used (searched 2026-10-13)

October 2026 open-weight leaders on public aggregators are Kimi K3 (~2.8T), GLM-5.2/5.3 (753B), Xiaomi MiMo-V2.6-Pro, Qwen3.6/3.8, DeepSeek-V4 (Pro/Flash), MiniMax-M3, Gemma 4, gpt-oss-120b and Llama 4. Hugging Face metadata checked the same day: Qwen/Qwen3.6-27B (56 GB, 2026-04-21), google/gemma-4-31b-it (63 GB, 2026-03-11), deepseek-ai/DeepSeek-V4-Flash-0731 (167 GB, FP8, 2026-07-31), openai/gpt-oss-120b, zai-org/GLM-5.2 (1,507 GB), moonshotai/Kimi-K3 (1,561 GB). The vLLM and NVIDIA Dynamo recipes run DeepSeek-V4-Flash on 4× H200 (Hopper); Kimi K3 and GLM-5.2 need more than one 4-GPU GH200 node in BF16/FP8 and are reported as "not run (exceeds one node)".

## Open-weight agents (same input, same suffix, same parser and repair harness as amendment 002 §3)

| ID | Model | GPUs | Why |
|---|---|---|---|
| O1a | Qwen3-8B (same base as J/P), thinking off | 1 | fine-tuning ablation, identical template |
| O1b | Qwen3-8B, thinking on | 1 | same base as an agent |
| O2 | Qwen/Qwen3.6-27B | 1–2 | latest open Qwen (dense) |
| O3 | google/gemma-4-31b-it | 1–2 | Google open family |
| O4 | openai/gpt-oss-120b, reasoning medium | 1–2 | OpenAI open-weight |
| O5 | deepseek-ai/DeepSeek-V4-Flash-0731 (FP8, 284B MoE / 13B active) | 4 | strongest mainstream model that fits one node |

Qwen3-VL-32B and Qwen3-VL-235B (amendment 003) are dropped as older-generation models. Any model whose 2-room smoke fails on the server's vLLM 0.31 is reported "not run" with the error, never silently replaced. Decoding: greedy, fixed seed; thinking/reasoning models get the amendment 003 extra allowance; the chat template is each model's own.

## API agent

`gpt-6.1-sol`, reasoning effort **medium**, unchanged from amendment 002.

## Storage

Model weights are downloaded into `$PROJECTDIR` (`/projects/u6uv/shanliantian.u6uv/fastfill/models/<name>`), caches on scratch, nothing large in `$HOME`.

## Budget

GPU cap stays 260 GPU-hours; O5 is expected to take ≤ 16 GPU-hours on the cohort.
