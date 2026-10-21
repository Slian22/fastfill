# SceneCode: Executable World Programs for Editable Indoor Scenes with Articulated Objects
- Authors: Puyi Wang, Yuhao Wang, Linjie Li, Zhengyuan Yang, Kevin Qinghong Lin, Yangguang Li, Yu Cheng. arXiv 2605.19587v1 (19 May 2026). https://arxiv.org/html/2605.19587
- What: scenes as executable programs with articulated parts (doors, drawers, lids) as separate functional parts.
- Metrics (App. D.1): Opening-Clearance OPC = "Fraction of architectural openings (doors and windows) whose clearance volume is obstructed by objects", using "an extruded clearance box ... in front of the opening" (box depth/width not given, unverified); NAV = largest connected free floor region / total free floor.
- Results (Table 1, lower OPC better): HSM OPC 10.9 +- 8.8, NAV 99.1; LayoutVLM OPC N/A, NAV 100.0; SceneSmith OPC 7.9 +- 4.8, NAV 97.8; SceneCode OPC 12.4 +- 6.7, NAV 100.0.
- No articulated-part sweep clearance or reachability metric.
- Relevance: 8-12% of room doors/windows are blocked in current generators, while NAV saturates at 98-100% -> NAV-style ratios do not detect door blocking; a door-front box check is needed separately.
- Tags: scene-gen-for-robots, articulated-objects
