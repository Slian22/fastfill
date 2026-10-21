# Probe and experiment code (2026-10-21)

Copy of `/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021/scripts/` (the working location; outputs stay
on the external disk). Key scripts: `s2_protect.py` (S2 + wall protection + strict phi), `companions.py` + `s2f.py`
(companion-protected S2-F), `s2f_task.py` (task-object reachability), `moma_checks.py` (mobile-manipulation checks),
`teacher_companions.py` (training-target proxy), `size_shrink_ceiling.py`, `judge_lib.py` / `judge_v2.py` (GPT pairwise
judge), `embed.py` / `analyze.py` / `clip_systems.py` (CLIP), `build_survey_xlsx.py` (survey workbook).
The API key is read from `llm_config.json` at run time and is never stored here.
