# BaSeNet: A Learning-based Mobile Manipulator Base Pose Sequence Planning for Pickup Tasks
- Authors: Lakshadeep Naik, Sinan Kalkan, Sune L. Sorensen, Mikkel B. Kjaergaard, Norbert Krueger (SDU). IROS 2024 (pp. 3666-3673, per search; arXiv page names no venue). https://arxiv.org/html/2406.08653v1
- What: RL policy outputs a sequence of continuous SE(2) base poses from which to pick all objects on a table; baselines use inverse reachability maps (IRM) at 10 cm and 45 deg (Lula kinematics solver), "-GC" variants add a grasp/trajectory feasibility check.
- Robot: UR5e on a mobile base (arm at the right back corner); footprint and reach not given. Base 0.5 m/s, 0.5 rad/s.
- Scene: table 2.0 m x 0.8 m, up to 10 YCB objects (5 classes); 50 random scenes each for 5 and 10 objects; start 2.5-3 m from the table. Episode ends on table collision.
- Results (% objects grasped, 5 / 10 obj): IRM-based PBG 10.0 / 8.9; PBG-GC 94.4 / 89.5; MBP 39.6 / 53.4; MBP-GC 96.0 / 97.9; exact DP 98.4 / 97.1; BaSeNet 97.5 / 94.8. Total time BaSeNet 103.8 / 184.8 s vs DP 219.7 / 406.3 s.
- Finding: pure IRM base poses without a grasp-feasibility check fail most of the time for a 6-DoF arm (10.0% / 39.6% grasped) - IRM reachability is not sufficient.
- Relevance: a check "base pose exists in IRM footprint" overestimates executability; a stricter symbolic check should require clearance in front of the receptacle and approach roughly perpendicular to its edge.
- Tags: base-placement
