# Amendment 006 — open-model modes and order, budgets, visual/judge details

Locked 2026-10-13 ~19:10 UTC. No J/P output exists (inference v2 shards wait for SFT); the API-agent run is in progress and only its progress counters (receipt counts, call counts) have been looked at, never answers or scores.

## Open-model modes (amendment 004 left O2/O3/O5 to model defaults)

| ID | Mode |
|---|---|
| O1a | Qwen3-8B, thinking off |
| O1b | Qwen3-8B, thinking on |
| O2 | Qwen3.6-27B, thinking on (model default) |
| O3 | gemma-4-31b-it, thinking off (instruct default) |
| O4 | gpt-oss-120b, reasoning effort medium |
| O5 | DeepSeek-V4-Flash-0731, **thinking on** (its agent mode; it is the open model sent to the GPT judge, and the API agent also reasons at medium effort) |

Reasoning text is stripped before parsing; a thinking model that never closes its reasoning within the cap produces an empty answer and counts as a failure. A smoke that loads and runs but yields only cap-hit reasoning is a measured failure, not "not run"; "not run" is reserved for systems that cannot load or execute.

Run order (one open-model job at a time, after all inference v2 GPU jobs): O5, O4, O2, O3, O1a, O1b. A system whose weights are not complete when its turn comes is submitted later, never replaced.

## Budgets

- GPU cap for this protocol raised to 400 GPU-hours (user mandate: request more GPUs as needed).
- GPT judge: 4 systems (J, P, API-agent harness, O5 harness, else O4) × 174 rooms = 696 sequential calls, call cap 800, API execution cap raised from 10 h to 24 h.

## Reporting details fixed now

- Low-reuse sources (size metrics split): InternScenes_{3rscan, arkit, mp3d, scannet}, MultiScan, Scan2CAD, SceneSmith, SpatialGen. All others, including InteriorGS and SpatialLM, are high-reuse.
- KID is the paired U-statistic over distinct rooms (GT and system images of the same room are correlated); it is secondary and not comparable with literature KID values.
- The amendment 005 cap rule (cap reached without EOS = failure) is applied per row to J and P; for agents, per-answer cap flags are applied where the receipts carry them (open models), otherwise reported as a count with a warning (API).
- P label: "category × room-type median, then category median, with constant fallback (0.40, 0.50, 0.65 m) when neither bucket has ≥ 5 training observations".
- Known-defect sensitivity uses the 3 declared rooms plus the test rooms changed by the final source-correction candidate (UID list frozen before any output is viewed).
