# Visibility-Aware Mobile Grasping in Dynamic Environments
- Authors: Tianrun Hu, Anxing Xiao, David Hsu, Hanbo Zhang. arXiv 2605.02487 (v1 4 May 2026, v4 19 Sep 2026); no venue. https://arxiv.org/html/2605.02487
- What: Fetch with a pan-tilt limited-FOV head camera; gaze policy, subgoal policy that samples grasp/intermediate whole-body configurations keeping the target visible (binary visibility under a gaze, mapped and robot self-occlusion considered; arm configs that self-occlude the target rejected), receding-horizon whole-body planner. Base poses sampled collision-free from a 2D SDF; observation poses at "a fixed safe standoff distance" (value not given); 2,560 pre-grasp samples. FOV values not given.
- Setup: ReplicaCAD, 20 scenes x 20 YCB objects (400 scenarios), 5-10 m navigation; targets on dining table, kitchen counter, workstation, coffee table, sofa.
- Results (Table I, static): ours 70.3% (abstract 70.4%), failures coll 5.0 / grasp 8.0 / IK 0.5 / reach 3.0 / perception 2.5 / planning 10.7; decoupled nav-then-manip 54.0% (coll 7.7, grasp 8.7, perc 7.6, plan 22.1); direct grasping 27.2%. Dynamic: 57.9% vs 48.6%.
- Relevance: visibility of the target from the approach pose measurably matters (perception failures 7.6 -> 2.5 points); there is still no layout-level visibility threshold here.
- Tags: base-placement, sim-benchmark
