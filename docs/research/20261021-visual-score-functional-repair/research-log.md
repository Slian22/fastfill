# Research log

- 2026-10-21: workspace + protocol locked (user away, asked for autoresearch with multiple agents). By-eye labels for the 20 S1-vs-S2 renders fixed beforehand in ../20261013-unknown-size-comparison/method-improvement-research-20261021.md §10 (better #02 #03 #11 #14; worse #01 #05 #06 #09 #12 #15 #17 #18 #19; rest same).
- 2026-10-21: launched 4 background agents (S2-F builder, CLIP metric, GPT pairwise judge, literature). Loop cron 7e9413bc every 20 min. User asked again why CLIP cannot be used -> measured by the CLIP agent, literature agent extracts how others use CLIP.
- 2026-10-21: literature agent done (15 notes + survey.md). Spot-checked VIPScene Table 2 tau values: match. Its judge advice (pros/cons, 3 votes, composite image) differs from the locked single-vote judge; keep the locked design for this run, note as follow-up.
- 2026-10-21 ~14:10: API back 13:56 (image input ok). Judge: ref-vs-raw 13/20, S1-vs-S2 10/10, position A 15/20. S2-F cohort+renders done; interim numbers in findings. Agents S2-F, CLIP, judge still running.
- 2026-10-21: CLIP agent done: H1b supported (CLIP/SigLIP cannot separate GT from raw; no correlation with violations).
- 2026-10-21: S2-F agent done; REPORT.md saved by me (subagent could not write it). H3 partly supported. Waiting for judge on S2F renders.
- 2026-10-21 ~14:50: API busy again; judge set B 9/20 done. Wrote to_human/report.html (interim) and opened it.
