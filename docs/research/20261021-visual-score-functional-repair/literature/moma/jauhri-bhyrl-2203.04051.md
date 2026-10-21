# Robot Learning of Mobile Manipulation with Reachability Behavior Priors
- Authors: Snehal Jauhri, Jan Peters, Georgia Chalvatzaki. IEEE RA-L 7(3):8399-8406, 2022. https://arxiv.org/abs/2203.04051 (read via ar5iv); code github.com/iROSA-lab/sampled_reachability_maps
- What: hybrid discrete/continuous RL (HyRL, BHyRL with boosted residual Q) decides base placement and whether to activate the arm for a 6D target; a reachability prior is learned from the arm's operational workspace / inverse reachability (manipulability-weighted). Grid resolution not stated (unverified).
- Robot: TIAGo++ (left arm only; base + torso + arm 10 DoF).
- Results: 6D_Reach_1m IRM 73.74%, HyRL 100%, SAC-hybrid 91.4%; 6D_Fetch (3 m, multi-object) BHyRL 83.27%; 6D_Fetch (4 m, table and sofa) 93.6%; IRM needs about 4.8 base queries per goal vs 2.2 sub-goals for the policy. Real TIAGo++ zero-shot fetch 81.25% over 32 trials.
- Real failures: mostly a sub-goal near the table that the ROS base planner could not reach safely.
- Relevance: a pose that is kinematically reachable but too close to furniture for the base planner is the dominant real failure -> a layout check should demand base-clearance margin around the chosen pose, not just reach.
- Tags: base-placement, RL-sim2real
