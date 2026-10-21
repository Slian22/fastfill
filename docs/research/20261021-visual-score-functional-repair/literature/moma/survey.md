# Survey: what mobile manipulation (MoMa) needs from an indoor room layout

Date 2026-10-21. Our setting: we place a given furniture list in a given room (position, yaw, support parent). A mobile manipulator then has to open the door and enter, navigate, see each task object and achieve born-false pick/place goal relations ("put the cup on the dining table"). Downstream already checks a 0.25 m navigation grid with a 0.2 m agent radius, starting from the entry door.

How to read this: every number comes from a note in this folder (file named in brackets). "(unverified)" means the source text did not confirm the number. "(derived)" means our own calculation: FK on a vendor URDF, or arithmetic. "(code)" means the number is in a repo, not in the paper. The notes cover 93 sources: 44 are pre-2025 or undated (including 4 robot spec sheets and the ADA standard), 48 have 2025-26 arXiv ids, and one is a 2026 company blog. Three notes in the parent folder are reused: [../sceneteract-2603.29798.md], [../sceneeval-2503.14756.md], [../roomer-2608.01973.md].

## Key conclusions (short)

1. **Our 0.2 m agent radius is too small for every common MoMa robot.**
   - Official nav-stack radii: Stretch 0.22, TIAGo 0.275, Fetch 0.30 [robot-stretch, robot-tiago, robot-fetch].
   - HomeRobot OVMM uses 0.25 for episode generation and 0.3 in sim [homerobot-ovmm].
   - Galaxea R1 navigation goals need the centre at least 45 cm from obstacles [robot-galaxea-r1].
   - SceneSmith uses about 35 cm for an RB-Y1 [scenesmith].
   - Stretch's 33 x 34 cm footprint has a half-diagonal of about 0.237 m (derived).
   - The Stretch and TIAGo nav2 costmaps use 0.05 m cells; our 0.25 m cells are 5x coarser.
2. **The most often reported scene-related MoMa failure is the base pose (stand-off and approach side), not reaching the room.**
   - MoManipVLA: Orient_to_place causes 72.09% of OVMM failures [momanipvla].
   - DynaMem: half of the home navigation failures were "a suboptimal location for manipulation" [dynamem].
   - N2M: a geometrically reachable base pose gives 0.16, against 0.93 for the oracle pose [n2m].
   - Mobi-pi: 3.1 cm of base noise halves success on Turn on Stove [mobi-pi].
   - Affordance base placement: an object-centre approach gets 47%, against 85% for an affordance-aware one [affordance-base-placement].
   - Conclusion: a layout must leave a free base *region* on the correct side of each target, not a single point.
3. **Opening clearance for articulated objects is the clearest documented case where the layout alone makes a task fail.**
   - ManiSkill-HAB: Close Fridge is 0.00% on validation scenes against 86.81% on train, "because the fridge door opens into a wall" [maniskill-hab].
   - MolmoSpaces: LLM-generated houses pass its articulation test only 65.2% of the time, because large objects were placed to keep door-to-door paths clear [molmospaces].
   - Room doors and windows are blocked in 8-12% of cases in current generators (SceneCode OPC) [scenecode].
4. **Connectivity metrics saturate.**
   - NAV = largest free component / all free floor scores 97.6-100% for every generator [sceneeval-navacc, scenecode, scenesmith].
   - Per-object front-access ACC still ranges 47-90% [sceneeval-navacc].
   - Conclusion: per-object, robot-calibrated checks are the informative signal.
5. **"Visible" has a standard symbolic definition.** A navigable viewpoint within 1.0 m of the target OBB, from which at least 0.1% of the pixels show the object [homerobot-ovmm (code), hssd-200, objectnav-revisited]. Lenient variants are 1.5 m (MolmoSpaces), 2 m within 60 deg (InfiniteWorld) and 3 m (GRUtopia).
6. **Prior work on layout repair for MoMa is thin.**
   - The closest is Rearrange-for-Co-Activity (ICRA 2023), which reports +14% accessible space and +30% reachable objects. It has no visibility, opening-sweep, height or door-entry terms, and no task-success evaluation [rearrange-coactivity].
   - Generators do the following instead:
     - Agent-box path guidance (PhyScene, SceneFoundry).
     - A pairwise 2r gap between objects (RoboLayout).
     - Rejection sampling (ProcTHOR, EmbodiedGen V2).
     - Post-hoc removal of blocking objects (MolmoSpaces).
     - Rollout filtering (MoMaGen, AutoMoMa, SAGE).
   - None certifies base pose, visibility, opening clearance and entry together for a given robot.

## 1. Landscape 2023-2026: benchmarks, simulators, robots, success criteria

### 1a. Benchmarks and simulators

| Work (year) | Task family | Robot (embodiment numbers) | Success criterion | Note |
|---|---|---|---|---|
| HomeRobot OVMM (2023) + NeurIPS'23 challenge (2024) | open-vocabulary "move object from receptacle A to B", sim HSSD (60 scenes) + real apartment | Stretch + DexWrist; sim radius 0.3 / episode generation 0.25, height 1.41, camera 1.31 m; arm 4 x 0.13 m telescope + lift 0-1.1 (code) | FindObj/FindRec: within 0.1 m of a viewpoint with >= 0.1% pixel coverage; Pick: EE within 0.8 m; Place: contact for 50 steps; any collision fails; 1,250 steps | homerobot-ovmm, homerobot-ovmm-challenge |
| Habitat 2.0 HAB (2021) | TidyHouse / PrepareGroceries / SetTable, ReplicaCAD | Fetch, 7-DoF arm; grasp within 15 cm | objects within 15 cm of goal; nav skill 0.3 m / 0.5 rad; open drawer 90-100% | habitat2-hab |
| Habitat 3.0 (2023), HSSD-200 (2023), PARTNR (2024) | social nav / rearrange; ObjectNav; LLM multi-agent planning | Spot (2-cylinder footprint); LoCoBot radius 0.18 m, height 0.88 m | follow 1-2 m; pick/place within 0.15 m; PARTNR is_next_to <= 0.50 m, is_on_floor within 0.04 m | habitat3, hssd-200, partnr |
| ManiSkill-HAB (2024/25) | HAB with low-level whole-body control, GPU | Fetch; navigation replaced by teleport | place within 0.15 m; open 75% (fridge) / 90% (drawer); close within 1%; 200 steps | maniskill-hab |
| ProcTHOR (2022) / ManipulaTHOR (2021) | procedural houses; ObjectNav, ArmPointNav | LoCoBot; ManipulaTHOR reach = hemisphere of radius 0.6335 m | ObjectNav: target within 1 m and visible; ArmPointNav 200 steps | procthor, manipulathor |
| BEHAVIOR-1K / OmniGibson (2024) + BEHAVIOR Challenge 2025 | 1,000 BDDL activities in 50 scenes (72-1,375 objects per scene); challenge = 50 long tasks | Tiago++ (paper), Galaxea R1 Pro (challenge, 23-dim joint action) | BDDL predicates; joint "open" at 5% above the lower limit; challenge q-score = fraction of satisfied predicates, timeout 2 x mean human time; winner 0.2605 / 0.2599, binary 0.1120 / 0.1240 | behavior-1k, behavior-challenge-2025, vla-failures-b1k |
| RoboCasa (2024) / RoboCasa365 (2026) | kitchen atomic + composite tasks; 120 -> 2,500 kitchens (365 copied from 50 Zillow homes) | Franka Panda on Omron base | door/drawer open >= 0.90 of range (code); NavigateKitchen within 0.20 m and cos(dyaw) >= 0.98 (code); 220/365 tasks need mobility; best multi-task 20.0 (GR00T N1.5) | robocasa, robocasa365, checkvla |
| GRUtopia (2024) | city/home loco-navigation and loco-manipulation, 100 scenes | Unitree H1; Aliengo + Z1 (camera 0.8 m above base) | target in FOV and < 3 m away; loco-manipulation 0 for all methods | grutopia |
| MolmoSpaces (2026) | 230k+ procedural/LLM houses; nav, pick, PnP, open, door | RB-Y1 (holonomic bimanual), Franka FR3 | Navigate-to: visible from head camera and within 1.5 m; door >= 67% open; PnP >= 50% of weight supported | molmospaces |
| MobileManiBench (2026), EMMOE (2025), Kitchen-R (2025), LAMBDA (2024/25), M3Bench (2024/25) | articulated + pick MoMa; long-horizon language MoMa; digital-twin kitchen; whole-body motion | 7-DoF arm on 2-DoF base; Fetch; Mobile ALOHA; ManipulaTHOR/Spot; Kinova on omni base | 300 steps to goal; 20-step plans; base within 10 cm and object within 5 cm; per-subtask points; no collision + goal held 2 s (M3Bench pick <= 25.59%, place <= 5.83%) | mobilemanibench, emmoe, kitchen-r, lambda, m3bench |
| InfiniteWorld (2024), InternScenes (2025), MetaScenes (2025) | Holodeck-based 10K scenes; 39,870 simulatable scenes; ScanNet replicas | "Stretch" (described as omni base + 7-DoF arm); Dingo; SPOC agent | nav: < 2 m and within 60 deg of HFOV; PointNav with 3-10 m start-goal | infiniteworld, internscenes, metascenes |

### 1b. Scene and environment generation aimed at robots (2023-26)

| Work | Robot / eval | Executability checks (numbers) | Note |
|---|---|---|---|
| SceneSmith (ICML 2026) | KUKA iiwa14 on a mobile base with lift; pick/place 16/100 | disk of half-width h_r: erode the floor and dilate OBBs, require one connected component; RB-Y1 (60 x 69 cm) about 35 cm; NAV 97.6, ACC 83.4 at 71.1 objects/room; "robot-sized passage" and door clearance zone with no sizes | scenesmith |
| EmbodiedGen v1 / V2 (2025/26) | Franka grasp tests; RL 9.7 -> 79.8% sim, 21.7 -> 75.0% real | support = 1, sibling footprint IoU = 0; target in the "reachable, forward-facing interaction region"; resample under room-connectivity and navigable-opening constraints; **no thresholds published**; 83.3% of 150 worlds usable | embodiedgen, embodiedgen-v2 |
| SAGE (2026) | Franka on Omron LD-60; MoMa privileged 68.4 / 52.8 (train/test) | collision ratio; unstable if > 0.2 m or > 8 deg after 120 steps; no navigability or reach metric; failures from "unreachable target configurations", "far-away objects" | sage |
| PhyScene (CVPR 2024) / SceneFoundry (2026) | no robot (agent boxes 0.2 / 0.3 / 0.5 m) | IoU of objects with agent boxes along an A* path; articulated boxes enlarged to the open state. R_reach 0.762-0.789, Col_scene 0.36-0.63; articulated collision 0.191 -> 0.109 (SceneFoundry) | physcene, scenefoundry |
| RoboLayout (2026) | none | pairwise centre distance >= e_i + e_j + 2r; no metrics | robolayout |
| MoMa-Kitchen (ICCV 2025), FloAff-Kitchen (2026) | 6 arms on a base; RoboCasa skills | base-pose labels: semicircle of radius = max reach, 10 cm grid (5 cm for FloAff), rollout per cell; random pose Top1 0.080 vs NavAff 0.72 | moma-kitchen, floaff-kitchen |
| FATE (2026) | RidgebackFranka | audit (reachability, interpenetration, stability, scale; no thresholds) + repair ops; task yield 12.6% -> 92.1% | fate |
| AgentWorld (CoRL 2025), Genie Sim 3.0 (2026), RoboLab (2026), SimVLA (2026), RoboGen (2023), GenManip (2025), ACDC (2024), Veo world sim (2025) | various | connectivity only (AgentWorld); rollout/VLM-judged (Genie Sim); stability 0.02 m in 300 steps (RoboLab); "spawn anywhere without collision" (SimVLA); fixed arm / tabletop for the rest | agentworld, geniesim3, robolab, simvla, robogen, genmanip, acdc, veo-world-sim-eval |

### 1c. Policies, data generation, real homes (2024-26)

| Work | Robot | Headline numbers | Scene-related failures | Note |
|---|---|---|---|---|
| pi0.5 (2025) | 2 wheeled holonomic bimanual platforms with torso lift | 3 unseen homes, step rubric (figures only) | drawer handles/cabinets, arm occluding the target, room-to-room memory | pi05 |
| OK-Robot (2024), DynaMem (2024/25), RUM (2024) | Stretch | 58.5% in 10 homes (82.4% decluttered); 70% lab, 9/17 homes; door 47/50, drawer 47/50 when started facing the target | clutter; bed-centre and high objects unreachable; half of the nav failures are a bad manipulation location | ok-robot, dynamem, rum |
| BRS / WB-VIMA (2025), Galaxea G0 (2025), MoMaGen (ICLR 2026) | Galaxea R1 / R1 Lite | ET average 58%; MoMaGen D0 average 63%, D2 0.47 / 0.22 / 0.07 / 0.16 | doorframe and table collisions; "no base pose ... satisfies reachability and visibility"; base-level planning about 26% of planner failures | brs, galaxea-g0, momagen |
| TidyBot / TidyBot++ (2023/24), HoMeR (2025) | Kinova on holonomic base, 50 x 54 cm | 85.0% real; fridge 10/10; holonomic 9/10 vs differential 4/10 (wipe); 79.17% | floor clutter blocks the base; doors/cabinets need sideways base motion; base misalignment hurts reach | tidybot, tidybot-pp, homer |
| Mobile ALOHA (2024), AhaRobot (2025), HOMIE (2025), Figure Helix 2.5 (2026 blog) | 90 x 135 cm base, reach 65-200 cm; 550 x 500 x 1550 mm, Z reach to 1250 mm; G1 squat; humanoid | 30 unseen homes, 56% (Figure) | fixed arm height cannot reach low cabinets; 30 cm elevator clearance; "tight, cluttered spaces" | mobile-aloha, aharobot, homie, figure-helix25 |
| MolmoBot (2026), SimVLA (2026), MobileManiVLA, EchoVLA, CheckVLA, AC-DiT, MoManipVLA, MoTo, UniManip, AnywhereVLA, BUMBLE | RB-Y1; Anubis; Stretch (OVMM); others | door sim 77.7%, real 2/9; SimVLA 54.4%; RoboCasa365 composite-unseen 1.2-14.9; OVMM 49.4% (MoManipVLA) | base collision 2/9 at doors; U/L kitchens need precise base positioning; poor base stop pose; wrong-object picks 38.9% with 20-25 distractors vs 10.0% with 5-10 | molmobot, simvla, mobilemanibench, echovla-momani, checkvla, ac-dit, momanipvla, moto, unimanip, anywherevla, bumble |
| Base placement line: Vahrenkamp IRM (2013), ARPlace (2012), N2M2 (2023), BHyRL (2022), BaSeNet (2024), MoMa-Pos (2024), Mobi-pi (2025), N2M (ICML 2026), affordance base placement (AAAI 2026), visibility-aware grasping (2026) | ARMAR-III, PR2/HSR/TIAGo, TIAGo++, UR5e base, Fetch, RB-Y1 | IRM grasp 10.0% w/o grasp check vs 94.4% with; N2M reach 0.16 vs oracle 0.93; visibility-aware 54.0 -> 70.3% | "reachable" is not the same as "executable"; side approach to articulated objects fails (5/20); poses too close to the table for the base planner | vahrenkamp-irm, stulp-arplace, n2m2, jauhri-bhyrl, basenet, momapos, mobi-pi, n2m, affordance-base-placement, visibility-mobile-grasping |

## 2. Layout properties required or reported as failure causes

### 2a. Navigation clearance / passage width / agent footprint

- **Agent radius in use:**
  - Habitat LoCoBot 0.18 m [objectnav-revisited, hssd-200].
  - Stretch nav2 0.22 m, inflation 0.55 m, 0.05 m cells [robot-stretch].
  - OVMM 0.25 m (episode generation) / 0.3 (sim) [homerobot-ovmm].
  - TIAGo 0.275 m [robot-tiago].
  - Fetch 0.30 m, inflation 0.70 m [robot-fetch].
  - R1: goals at least 45 cm from obstacles [robot-galaxea-r1].
  - RB-Y1 about 35 cm [scenesmith].
  - Humanoid about 20 cm [scenesmith].
  - PhyScene tests agent boxes of 0.2 / 0.3 / 0.5 m [physcene].
- **Footprints:**
  - Stretch 33 x 34 cm [robot-stretch]; Stretch 4 diameter 43 cm (datasheet) / 45 cm (web).
  - TidyBot++ 50 x 54 cm [tidybot-pp].
  - Fetch 51 x 56 cm (TidyBot++ Table 1, secondary).
  - TIAGo diameter 54 cm [robot-tiago].
  - R1 chassis 675 x 636 mm [robot-galaxea-r1].
  - Mobile ALOHA 90 x 135 cm [mobile-aloha].
  - Long bodies are modelled with several circles: OVMM uses 3 circles at 0 and +-0.25 m, tested at 4 rotations every 0.3 m (code) [homerobot-ovmm]; Spot uses 2 cylinders [habitat3].
- **Passage widths seen in experiments:**
  - N2M2: passages "as narrow as 0.85 m" are its hardest map, where TIAGo succeeds 42.0% against PR2 70.0%. A 0.89 m door frame is narrower than the PR2's 0.91 m diagonal [n2m2].
  - ADA: 0.915 m walkway, 0.815 m door clear width, 1.525 m turning circle [ada-2010-standards].
  - InternScenes navigable floor = ESDF > 0.5 m [internscenes].
  - Roomer walkway 0.60 m [../roomer].
- **Grid validation:** ProcTHOR runs BFS on a 0.25 m grid from the agent and needs at least 5 reachable cells per room [procthor]. HSSD removes navmesh islands with radius < 1.5 m (beds, tables) [hssd-200]. OVMM uses only the largest indoor island, with indoor ratio >= 0.85 (code) [homerobot-ovmm].
- **Reported causes:**
  - Furnished houses cut legged navigation from 100% to 58% (H1) and 14% (Aliengo) [grutopia].
  - Real scans are harder than generated scenes (NavDP 48.3 vs 61.9) [internscenes].
  - "Tightly constrained multi-room environments" [homerobot-ovmm].
  - Exploration fails in "cluttered, tight spaces and corridors" [anywherevla].
  - An enclosed spawn room that must be exited gives the lowest scores [lambda].

### 2b. Base-placement feasibility

- **Samplers and their numbers:**
  - OmniGibson: 0.0-1.5 m from the object, yaw in [-pi, pi], 200 attempts, same room, then collision + IK (code) [behavior-1k].
  - MoMaGen: ring around the target, then IK; radii not stated [momagen].
  - RoboCasa: base 0.20 m in front of the fixture (+0.10 m for fridges); door tasks stand beside the swing at fixture width/2 + 0.2 m (code) [robocasa].
  - Affordance base placement: at least 0.4 m from obstacles, preferred 0.7 m, coarse stop within 1.5 m [affordance-base-placement].
  - OK-Robot: stand "at an arms length", A* on a 10 cm grid with 20 cm inflation [ok-robot].
  - AnywhereVLA: base perpendicular to the support edge at a fixed offset [anywherevla].
- **Spawn regions in front of articulated objects:**
  - HAB: fridge 0.9 x 1.2 m, drawer 0.15 x 0.75 m [habitat2-hab].
  - ManiSkill-HAB: about 0.93-1.83 m ahead of a fridge and 0.3-1.5 m ahead of a drawer, +-0.6 m lateral [maniskill-hab].
  - MobileManiBench: 1.0-1.5 m, +-15 deg [mobilemanibench].
- **Reach:**
  - ManipulaTHOR 0.6335 m [manipulathor].
  - Stretch arm extension 0.52 m (official). Grasp centre 0.415-0.935 m *sideways* from the base origin (derived), so the base parks parallel to the receptacle edge [robot-stretch].
  - TIAGo arm 87 cm [robot-tiago].
  - R1 700 mm (860 mm with gripper); R1 Pro radius 716 mm (861 mm) [robot-galaxea-r1]. BRS reports 923 mm per arm [brs]; the two sources differ.
  - Planar reach at height z: delta_r = sqrt(r_l^2 - z_offset^2) [momapos].
- **Pose existence is not enough:**
  - IRM poses without a grasp check grasp only 10.0% / 39.6% of objects, against 94.4-97.9% with the check [basenet].
  - A reachable pose gives 0.16, against 0.93 for the oracle; the good region is about 0.4 x 0.4 m, +-15 deg [n2m].
  - 3.1 cm of base error halves success [mobi-pi].
  - In real runs, poses too close to the table for the base planner dominate failures [jauhri-bhyrl].
- **Reported causes:**
  - Orient_to_place 72.09% of failures [momanipvla].
  - Half of home navigation failures [dynamem].
  - Motion errors 9.7% of trials, from "base ... stopped slightly outside the optimal workspace" [unimanip].
  - "Misadjusting the base" [bumble].
  - "Unfavorable robot base placement" [behavior-1k].
  - Baselines 0 on D1/D2 because "objects are beyond the reachability of replayed base poses" [momagen].

### 2c. Reachable surface heights

- **Robot bands:**
  - Stretch gripper centre 0.11-1.21 m with a level wrist (derived), lift 1.1 m [robot-stretch].
  - N2M2 goal heights: PR2 [0.2, 1.55] m, TIAGo [0.2, 1.5] m, HSR [0.2, 1.4] m [n2m2].
  - Mobile ALOHA 0.65-2.0 m, so it cannot reach low cabinets, ovens or dishwashers [mobile-aloha].
  - R1 / R1 Pro 0-2.0 m; R1 Lite 0-1.7 m [robot-galaxea-r1].
  - AhaRobot floor to 1.25 m [aharobot].
  - TIAGo robot height 1.10-1.45 m; end-effector min/max height not published [robot-tiago].
  - ADA human reach 0.38-1.22 m [ada-2010-standards].
- **Task-object surfaces:** RoboCasa object regions z_range = (0.45, 1.50) m (code) [robocasa]. Household object heights cluster at 0.09, 0.49, 0.94 and 1.43 m in BEHAVIOR-1K [brs].
- **Reported causes:**
  - Objects on high places or in the middle of a bed are unreachable for Stretch [ok-robot].
  - The head camera cannot see the floor in front of the robot or shelves above its head [behavior-challenge-2025].
  - Confined vertical clearance causes grasp failures [molmospaces].

### 2d. Object visibility / occlusion

- **Viewpoint spec:**
  - The viewpoint is navigable and on the main island, at most 1.0 m from the target OBB (code; the paper text says 1.5 m), on a 0.15 m candidate grid, with coverage >= 0.001 over 3 head tilts from a 1.31 m camera [homerobot-ovmm].
  - Same 1 m / 0.1% rule in [hssd-200].
  - ObjectNav: within 1 m of the OBB plus oracle visibility; if visibility is enforced, at least 1 pixel [objectnav-revisited].
  - Instances with no valid viewpoint are *dropped* from the dataset (filtered, not repaired).
- **Success variants:**
  - Visible + 1.5 m [molmospaces].
  - < 2 m and within 60 deg of HFOV [infiniteworld].
  - In FOV and < 3 m [grutopia].
  - Within 1 m and in FOV (RoboTHOR) [objectnav-revisited].
  - MoMaGen: % of frames with the target visible (D0 1.00 / 0.86 / 0.79 / 0.69) [momagen].
- **Measured effects:**
  - Keeping the target visible lifts Fetch grasping from 54.0% to 70.3%; perception failures drop from 7.6 to 2.5 points [visibility-mobile-grasping].
  - Occluding the wrist camera drops pi0.5 to 2%, the third-person camera to 20% [molmospaces].
  - Navigation fails because "a wall or obstacle blocking the view" [infiniteworld].
  - "Target missing or unrecognizable" dominates failed runs [emmoe].
  - Occlusion/contact ambiguity is 18.6% of residual failures [checkvla].
  - SceneTeract isVisible pass rate is 97.7-98.7% on 3D-FRONT [../sceneteract].

### 2e. Articulated-object opening clearance (doors, drawers, fridges)

- **Success thresholds:**
  - Open >= 0.90 of range [robocasa (code)].
  - 75% (fridge) / 90% (drawer) [maniskill-hab].
  - 90-100% [habitat2-hab].
  - Door >= 67% [molmospaces, molmobot].
  - "Open" = 5% above the lower limit [behavior-1k].
  - Door progress > 90% [harmonic-mm].
- **Validity tests:**
  - Joint must move through >= 70% of its range, otherwise blocking objects are removed [molmospaces].
  - Articulated boxes enlarged to the fully open state [physcene]; expanded along the joint axis [scenefoundry].
  - OPC = fraction of doors and windows whose clearance box is obstructed [scenecode].
- **Measured failures:**
  - Close Fridge 0.00% on validation scenes vs 86.81% on train, because the door opens into a wall [maniskill-hab].
  - Articulation pass 65.2% when navigation clearance was prioritised [molmospaces].
  - Articulated-scene Col_scene 0.76 [physcene].
  - Articulated collision still 0.109 with an explicit constraint [scenefoundry].
  - OPC 7.9-12.4% [scenecode].
  - Pull doors 51.2% vs push 80.0% (sim), cause: "reduced space as the door opens, especially in narrow areas" [harmonic-mm].
  - Real door opening: base collision 2/9 [molmobot].
  - Real drawer opening 10.0%, collision with the moving hinge [momanipvla].
  - Side approach to cabinet/dishwasher 5/20 [affordance-base-placement].
  - Baseline fridge/microwave opening 18-82% vs 100% when the swing trajectory is reachable [momapos].
  - Doors/cabinets need sideways base motion [tidybot-pp].
  - Standing beside the swing [robocasa].
- **Human standard for room doors** (ADA Table 404.2.4.1): front approach pull side 1.525 m deep plus 0.455 m beyond the latch; push side 1.22 m deep [ada-2010-standards].

### 2f. Receptacle free area, clutter, object density

- **Placement criteria:**
  - In-receptacle when the centre distance is < 0.7 x the receptacle radius (code) [robocasa].
  - Bbox overlap > 0.5 [m2diffuser].
  - >= 50% of the weight supported [molmospaces].
  - Within 15 cm [habitat2-hab, maniskill-hab].
  - OVMM filters receptacles "too small for object placement" (threshold unverified) [homerobot-ovmm].
  - Objects near receptacle edges fall off; large receptacles cause held-object collisions [homerobot-ovmm-challenge].
  - **No paper gives a numeric free-area or edge-margin threshold.**
- **Clutter:**
  - HAB places 5 clutter objects per receptacle [habitat2-hab].
  - ProcTHOR allows <= 3 of a type per receptacle [procthor].
  - Fridge clutter: 60.62% of pick failures are excessive collisions [maniskill-hab].
  - SR 68.7 vs SR-without-disturbance 39.4 [manipulathor].
  - Wrong-object picks 38.9% with 20-25 distractors vs 10.0% with 5-10 [bumble].
  - 58.5% vs 82.4% decluttered [ok-robot].
  - Distractors 0/1/2 give SR 20.27 / 18.60 / 18.00 [genmanip].
  - TidyBot cannot drive over floor objects [tidybot].
- **Density:**
  - SceneSmith 71.1 objects/room lowers NAV/ACC [scenesmith].
  - InternScenes 41.5 objects/region; real scans are harder [internscenes].
  - BEHAVIOR-1K challenge scenes are "generally not cluttered" [vla-failures-b1k], so clutter is not yet a tested axis there.

## 3. Robot specifications (official sources unless tagged)

| Robot | Base | Height | Reach | Reachable height | Nav radius | Note |
|---|---|---|---|---|---|---|
| Stretch 2 / 3 | 33 x 34 cm, differential, 25 mm ground clearance (RE2) | 141 cm | arm 52 cm (telescoping, sideways) | lift 110 cm; gripper 0.11-1.21 m level wrist (derived); head camera 1.322 m (derived) | 0.22 (nav2) | robot-stretch |
| Stretch 4 (2026) | diameter 43 cm (datasheet) / 45 cm (web), 3 omniwheels | 160 cm | 55 cm + 6 cm wrist | lift 120 cm | not found | robot-stretch |
| Fetch | 50.8 cm wide (secondary); 559 mm diameter (unverified) | 109.6-149.1 cm (secondary) | not official; gripper 1.128 m ahead at zero pose (derived) | shoulder 0.726-1.126 m (derived); torso 400 mm | 0.30, inflation 0.70 | robot-fetch |
| TIAGo / TIAGo++ | diameter 54 cm, 70 kg | 110-145 cm, torso 35 cm | 87 cm (without EE) | not published; N2M2 goals 0.2-1.5 m | 0.275 | robot-tiago, n2m2 |
| TIAGo Pro | 50 x 72 cm, mecanum | 120-155 cm | vertical 92 cm, horizontal 236 cm | not found | not found | robot-tiago |
| Galaxea R1 / R1 Pro | chassis 675 x 636 x 346 mm | 1700 mm | 700 mm (860 with gripper) / radius 716 mm (861 mm) | 0-2000 mm | goals >= 45 cm from obstacles | robot-galaxea-r1, brs |
| Galaxea R1 Lite | chassis 596 x 600 x 250 mm | 1281 / 1560 mm | arm 600 mm | 0-1.7 m (2025) | - | robot-galaxea-r1, galaxea-g0 |

No vendor publishes a minimum passage or door width, or a door-opening spec [robot-*].

## 4. Prior work that optimises or repairs layouts for MoMa executability

| Work | What is optimised / checked | Measured | What is missing |
|---|---|---|---|
| Rearrange Indoor Scenes for Human-Robot Co-Activity (ICRA 2023) | re-optimise furniture: SDF free-space term for the base (radius r_b) + reach-from-interaction-side term (d_max), keeping human functional groups | ~+14% accessible space, ~+30% reachable objects; 2 of 7 scenes rated lower by humans | no visibility, opening sweep, heights or door entry; no task success; r_b and d_max not reported | rearrange-coactivity |
| PhyScene (CVPR 2024), SceneFoundry (2026) | diffusion guidance: agent-box IoU along an A* path; open-state articulated boxes | R_reach 0.76-0.83; articulated collision 0.109 | no arm, height, base pose or visibility; walkability as area ratio | physcene, scenefoundry |
| RoboLayout (2026) | pairwise clearance >= footprints + 2r in differentiable layout | qualitative only | a chain of gaps does not imply a path; no reach | robolayout |
| Make it Home (2011), Haisor (2024), SceneTeract (2026) | human accessible space, viewing frustum, door-to-door path (box penetration costs); RL repair for free space; per-agent functional checks | 3/21 differ from designers; Haisor numbers unverified; SceneTeract adult task success 59.0%, hasClearance 67.3% | human-sized, no robot reach | makeithome, haisor, ../sceneteract |
| Coexistence wheelchair (2020), kitchen co-design (2023) | layout optimisation with a motion planner in the loop (RRT, Reeds-Shepp) | 3-5 min per design; planning 96.06% of compute | path only; expensive | coexistence-wheelchair, codesign-hrc-kitchen |
| FATE (2026) | LLM task+scene generation with audit and repair (TRANSFORM_POSE, SWAP_ASSET, ...) | yield 12.6% -> 92.1%; 298/333 repairs | checks unquantified; edits a task scene, not a room under a fixed furniture list | fate |
| MolmoSpaces (2026), ProcTHOR (2022), EmbodiedGen V2 (2026), SceneSmith (2026) | post-hoc removal of blocking objects; resample whole house; resample under reachability / navigable-opening constraints; agent connectivity tool | articulation pass 65.2-99.1%; >= 5 cells per room; 83.3% usable; NAV 97.6 | no thresholds (EmbodiedGen), no per-object base pose (ProcTHOR), no visibility | molmospaces, procthor, embodiedgen-v2, scenesmith |
| Episode filters: OVMM / HSSD / ManipulaTHOR / MoMaGen / AutoMoMa | drop targets without viewpoints / reachable base poses / feasible trajectories | rates not reported | filter, never repair | homerobot-ovmm, hssd-200, manipulathor, momagen, automoma |
| Base-placement learners (MoMa-Kitchen, FloAff, N2M, Mobi-pi, MoTo) | choose a base pose in a *given* scene | Top1 0.72; SR 0.85 / 0.79 (Layouts split harder); 0.55 vs oracle 0.93 | scene is fixed; layout never changed | moma-kitchen, floaff-kitchen, n2m, mobi-pi, moto |

What is open: see 6(b).

## 5. 2025-2026 trends (ranked by how often they appear)

Counts are tag occurrences over the 49 notes dated 2025-26 in this folder (48 arXiv + Figure blog). The sample comes from our searches, which deliberately targeted benchmarks, scene generation and base placement, so the counts show prevalence in this sample, not in the field.

| Rank | Trend | Count / 49 | What groups do now | Layout relevance |
|---|---|---|---|---|
| 1 | VLA / foundation policies applied to MoMa | 24 | pi0.5 in unseen homes; GR00T N1.5 / pi0.5 on RoboCasa365; MolmoBot / SimVLA trained only on generated scenes; challenge winners are VLAs | success is still low on long tasks: B1K binary 0.1120-0.1240; RoboCasa365 composite-unseen 1.2-14.9; failures are mostly grasp and logic, but base pose, visibility and articulated handles recur |
| 2 | House-scale sim benchmarks | 21 | BEHAVIOR Challenge 2025, RoboCasa365, MolmoSpaces, MobileManiBench, EMMOE, Kitchen-R | layouts are usually fixed or copied from real homes; layout is rarely an experimental variable |
| 3 (tie) | Articulated objects as standard sub-tasks | 18 | doors, drawers, fridges, dishwashers in almost every 2025-26 benchmark | opening clearance and swing-side base space become hard requirements |
| 3 (tie) | Scene generation for robots | 18 | SceneSmith, SAGE, MolmoSpaces, EmbodiedGen V2, InternScenes, Genie Sim 3.0, SceneFoundry, SceneCode | checks are physics stability + collision + connectivity; executability by rollouts; no robot thresholds published |
| 5 | Automated data generation | 14 | MoMaGen, AutoMoMa, MolmoBot, SimVLA, MobileManiBench | infeasible scenes are silently filtered, so the layout determines the data yield |
| 6 (tie) | Wheeled bimanual whole-body robots | 12 | Galaxea R1 family, RB-Y1, TidyBot++, AhaRobot, pi0.5 platforms replace Stretch/Fetch | larger footprints (0.6-0.68 m chassis) mean 0.45 m goal clearance, not 0.2 m |
| 6 (tie) | Base placement as its own problem | 12 | N2M, Mobi-pi, FloAff, MoMa-Kitchen, MoTo, affordance base placement, UniManip | "reachable is not executable": a free 0.4 x 0.4 m region on the correct side is needed |
| 6 (tie) | Long-horizon multi-step tasks | 12 | BEHAVIOR 50 tasks, RoboCasa365 composites, EMMOE | multi-room search and memory; entry and room-to-room paths matter |
| 9 | Teleop data at scale | 8 | BRS JoyLo, Galaxea Open-World 500 h, AgiBot World 2976.4 h | mostly evaluation without navigation |
| 10 (tie) | Real-home deployment, real2sim twins, open-vocabulary nav-pick | 6 each | pi0.5, Figure (30 homes, 56%), SimVLA home; RoboCasa365 Zillow copies, InternScenes | clutter and tight homes are the named constraints; no numeric passage widths reported |
| 13 | RL sim2real / humanoids / 3D memory | 5 / 4 / 2 | WHOLE-MoMa, HOMIE, EchoVLA | humanoids widen height reach by squatting |

## 6(a). MoMa layout-requirement checklist (symbolic checks on top-down layouts with OBBs)

Inputs: room polygon, walls, doors (hinge, swing direction, leaf width), each object's OBB (x, y, yaw, w, d, h, z_bottom), its support parent, articulated parts (type, axis, leaf/drawer depth), the task objects, and the goal receptacles. All checks take a robot profile P. Hard = fail the layout; soft = score or repair priority.

**P (robot profile).** Defaults are in the table below; set r_base, reach band, height band, camera height and arm side per robot.

| Profile | r_base | horizontal reach (from base centre) | target height band | camera h | arm side |
|---|---|---|---|---|---|
| Stretch 3 | 0.22 (nav2) to 0.25-0.3 (OVMM) | 0.415-0.935 m lateral (derived) | 0.11-1.21 m (derived) | 1.31 m (OVMM) | side |
| TIAGo | 0.275 | <= 0.87 m arm from shoulder (offset not modelled) | 0.2-1.5 m (N2M2) | not found | front |
| Fetch | 0.30 | not official | shoulder 0.726-1.126 m (derived) | 1.061-1.461 m (derived) | front |
| Galaxea R1 / Pro | 0.45 goal clearance (vendor rule) | 0.70-0.86 m (0.716-0.861 Pro) | 0-2.0 m | not found | front |
| Conservative generic | 0.30 | 0.4-0.8 m (our choice; OVMM pick within 0.8 m, baselines prefer 0.7 m) | 0.45-1.20 m (our choice; RoboCasa 0.45 low, Stretch / ADA about 1.2 high) | 1.31 m | front |

| ID | Check | How to compute | Threshold (default) | Hard/soft | Sources |
|---|---|---|---|---|---|
| C1 | Free-space grid at robot size | rasterise all floor-standing OBBs (tables and beds are solid: no driving under) and walls; dilate by r_base; cell <= 0.05-0.10 m (0.25 m cells quantise passages) | r_base from P (never < 0.22 for Stretch; 0.2 m is below every vendor nav radius found) | hard | robot-stretch, robot-tiago, robot-fetch, homerobot-ovmm, ok-robot (10 cm grid, 20 cm inflation) |
| C2 | Main component from the entry door | flood fill from the free cell just inside the door; everything below must lie in this component; drop islands | single component; ignore islands < 1.5 m radius | hard | procthor (BFS), hssd-200, scenesmith (single component), homerobot-ovmm (largest island) |
| C3 | Entry door swing + approach | room-door leaf sweep (quarter disc, radius = leaf width, on the swing side) free of OBBs; robot standing rectangle on the pull side free | hard: sweep free + footprint-sized zone beside the latch; comfortable: ADA pull side 1.525 m deep x (door + 0.455 m), push side 1.22 m deep | hard / soft | ada-2010-standards, harmonic-mm (pull narrow spaces), scenecode (OPC 8-12% blocked), scenesmith, lambda |
| C4 | Passage width along entry -> target paths | implied by C1 at r_base; additionally report min corridor width (2 x ESDF) along the shortest path | hard: > 2 r_base; soft: >= 0.85-0.915 m (N2M2 failure regime, ADA); turning space 1.525 m circle for non-holonomic bases | hard / soft | n2m2, ada-2010-standards, internscenes (ESDF > 0.5 m), ../roomer (0.60 m) |
| C5 | Base region per task object and goal receptacle | cells in the C2 component whose distance to the target point (object position, or chosen place point) lies in the reach band, with the arm side facing the target (Stretch: parallel to the edge); use delta_r = sqrt(r^2 - dz^2) for height | hard: >= 1 cell; soft: feasible region >= 0.4 x 0.4 m with +-15 deg yaw tolerance; prefer >= 0.4 m from other obstacles | hard / soft | behavior-1k (0-1.5 m + IK), moma-kitchen (semicircle = reach, 10 cm grid), n2m (0.4 x 0.4 m, +-15 deg), affordance-base-placement (0.4 m, 0.7 m), momapos, robot-stretch |
| C6 | Height reachability | support-top height of the pick object and the goal receptacle within the P height band; for objects inside containers, container opened (C8) | P band; generic 0.45-1.20 m; floor picks only if the profile allows (Stretch yes, Mobile ALOHA no) | hard | robocasa (0.45-1.50), robot-stretch, n2m2, mobile-aloha, ok-robot (high places), ada-2010-standards |
| C7 | Visibility viewpoint | exists a C2 cell within 1.0 m of the target OBB with a 2D line of sight to the target centre not crossed by any OBB whose top is above the camera-to-target line (camera h from P); needed for each pick object and each goal receptacle | hard: within 1.0 m; lenient 1.5 m | hard | homerobot-ovmm (1.0 m, >= 0.1% pixels, 3 tilts), hssd-200, objectnav-revisited, molmospaces (1.5 m), infiniteworld (2 m / 60 deg), visibility-mobile-grasping |
| C8 | Articulated opening sweep | for every openable part involved in a task (and soft for all): swept footprint at the success fraction free of walls and other OBBs (door leaf quarter disc; drawer = depth extruded) | hard: free at >= 0.70 of range (MolmoSpaces test; RoboCasa success needs 0.90, door tasks 0.67) | hard | maniskill-hab (door into wall: 0%), molmospaces, physcene, scenefoundry, robocasa, harmonic-mm |
| C9 | Standing region at articulated objects | free rectangle in front of, and offset to the latch/handle side of, the sweep, inside the C2 component | fridge 0.9 x 1.2 m (HAB); drawer 0.15 x 0.75 m strip (HAB); hinge-side offset width/2 + 0.2 m (RoboCasa) | hard | habitat2-hab, maniskill-hab, robocasa, affordance-base-placement (side approach 5/20), tidybot-pp, molmobot (base collision) |
| C10 | Receptacle free area for born-false goals | on the goal receptacle top, free rectangle (minus other children footprints) >= held object footprint, inside the reach-side half of the top | >= object footprint + margin (margin: our choice, no source threshold); place point on the near side within reach | hard (fit) / soft (margin) | robocasa (0.7 x radius), m2diffuser (overlap > 0.5), molmospaces (>= 50% supported), homerobot-ovmm-challenge (edge falls, large receptacles) |
| C11 | Clutter near targets | count objects within 0.50 m of the pick object, and same-category distractors in the room | soft; flag > 5 per receptacle (HAB hard setting); same-type <= 3 per receptacle | soft | partnr (next_to 0.50 m), habitat2-hab, procthor, bumble (38.9% vs 10.0%), ok-robot, maniskill-hab |
| C12 | Small floor objects on paths | any floor-standing OBB below base height on the C1 path counts as an obstacle | hard (bases cannot drive over objects; Stretch ground clearance 25 mm) | hard | tidybot, robot-stretch |
| C13 | Report robot-calibrated scores | fraction of task objects passing C5+C6+C7 (+C8+C9 for articulated), entry pass, min corridor width | report; NAV-style ratios alone saturate at 97.6-100% | metric | sceneeval-navacc, scenecode, scenesmith |

Implementation note: C1/C2/C4/C5/C7/C9 all reuse one dilated occupancy grid plus a distance transform; C3/C8 are polygon-vs-OBB intersection tests. The OVMM `viewpoints.py` failure labels (not_on_active_island / too_far / down_unnavigable / low_visibility) are a ready-made failure taxonomy for C2/C5/C7.

## 6(b). Open gaps where a layout model for MoMa could contribute

1. **No generator or repair method certifies, for a given robot, that each task object and goal receptacle has a free base region within arm reach, at a reachable height and on the correct arm side.** Rearrange-Co-Activity comes closest, with no heights, visibility or task-level evaluation. PhyScene, SceneFoundry and RoboLayout use agent boxes or pairwise gaps only.
2. **Navigation clearance and opening clearance compete, and no one optimises them jointly.** MolmoSpaces gets 65.2% articulation pass when keeping door-to-door paths clear. A swing-into-wall layout zeroes a task (ManiSkill-HAB Close Fridge 0.00%).
3. **Entry is unchecked.** No layout work checks the room door's swing, the approach zone or the path from the door to every task viewpoint, yet 8-12% of doors/windows are blocked (SceneCode) and exiting/entering a room is the hardest case (LAMBDA).
4. **Visibility from the approach pose is only an episode filter.** OVMM, HSSD and ObjectNav drop failing instances; no layout objective uses it.
5. **Robot-calibrated metrics are missing.** NAV saturates, the agent sizes in use (0.18-0.3 m) are below wide bimanual bases (0.45 m clearance), and no paper reports "fraction of task targets executable for robot P". A layout benchmark that reports C13 per robot profile would be new.
6. **Failures are not attributed to the scene or the policy.** SceneSmith (16/100), SAGE, AgentWorld (4-30%) and M3Bench report low MoMa success with no scene-vs-policy breakdown. Layouts that pass C1-C12 by construction would let benchmarks separate the two.
7. **Repair instead of filtering.** MoMaGen, AutoMoMa, OVMM and ProcTHOR discard infeasible targets or houses. A minimal-edit repair that keeps the furniture list and functional groups is open; FATE repairs task scenes with unquantified checks.
8. **No published thresholds for receptacle free area and edge margin.** These, and the clutter thresholds, have to be defined and validated (e.g. against OVMM edge falls and large-receptacle collisions).
9. **Difficulty control.** Generated layouts are easier than real ones (InternScenes 48.3 vs 61.9; HSSD realism beats scale), and B1K scenes are "generally not cluttered". A layout model that can produce feasible-but-hard rooms (narrow 0.85-0.9 m passages, U/L kitchens, distractor density) is useful for evaluation, not just for feasibility.
10. **Robot-conditioned layouts.** Height and reach bands differ widely: Mobile ALOHA >= 0.65 m, Stretch 0.11-1.21 m side-reach, R1 0-2.0 m. No generator takes a robot profile as input.
