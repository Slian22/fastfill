# Research log

- 2026-10-21: workspace + protocol locked (user away, asked for autoresearch with multiple agents). By-eye labels for the 20 S1-vs-S2 renders fixed beforehand in ../20261013-unknown-size-comparison/method-improvement-research-20261021.md §10 (better #02 #03 #11 #14; worse #01 #05 #06 #09 #12 #15 #17 #18 #19; rest same).
- 2026-10-21: launched 4 background agents (S2-F builder, CLIP metric, GPT pairwise judge, literature). Loop cron 7e9413bc every 20 min. User asked again why CLIP cannot be used -> measured by the CLIP agent, literature agent extracts how others use CLIP.
- 2026-10-21: literature agent done (15 notes + survey.md). Spot-checked VIPScene Table 2 tau values: match. Its judge advice (pros/cons, 3 votes, composite image) differs from the locked single-vote judge; keep the locked design for this run, note as follow-up.
- 2026-10-21 ~14:10: API back 13:56 (image input ok). Judge: ref-vs-raw 13/20, S1-vs-S2 10/10, position A 15/20. S2-F cohort+renders done; interim numbers in findings. Agents S2-F, CLIP, judge still running.
- 2026-10-21: CLIP agent done: H1b supported (CLIP/SigLIP cannot separate GT from raw; no correlation with violations).
- 2026-10-21: S2-F agent done; REPORT.md saved by me (subagent could not write it). H3 partly supported. Waiting for judge on S2F renders.
- 2026-10-21 ~14:50: API busy again; judge set B 9/20 done. Wrote to_human/report.html (interim) and opened it.
- 2026-10-21 ~15:20: user asked for training-side innovation; searched papers; ran teacher-companion proxy (smoke 200: 0.909 -> 0.764). Full run in background. Judge B set stalled on API busy (9/20).
- 2026-10-21: teacher proxy full run done: 0.915 -> 0.795 pairs kept; 21-50 0.848 -> 0.596; sofa group 0.49. Added to report.
- 2026-10-21 ~15:40: API back; judge set B: S2F vs S1 7 win / 10 loss / 3 tie; S2F vs S2 (17/20 so far) 4 win / 13 tie / 0 loss. Combined S2F vs S1 (A+B): 19-18-3.
- 2026-10-21: judge v2 sanity passed marginally (18/20 consistent, GT 13/18). Addendum B: cap 300, v2 on S2F vs S1 (A+B, 80 calls) running.
- 2026-10-21 17:45: API busy/400 since ~15:40 (judge v2 S2F vs S1 stuck at 15/40). Deleted the 20-min cron (idle ticks); the judge script keeps retrying and its completion notification resumes the work.
- 2026-10-21: user said skip; judge v2 stopped at 15/40 (S2F 6 / S1 5 / tie 4; 224 ok, 94 busy, 134 error calls in total). Phase concluded.
- 2026-10-21: phase 2 task-aware placement: upstream task format surveyed (templated, 95% no relations); task-scoped headroom measured; task-scoped blocker push +0.016.
- 2026-10-21: phase 2 reframed for mobile manipulation (user). Addendum C locked (50e7da6a9). Launched 3 agents: MoMa literature survey, task-reachability S2-F, MoMa symbolic checks.
- 2026-10-21: MoMa checks agent done (H5). Robot contract is symbolic; front approach clearance is the main MoMa failure.
