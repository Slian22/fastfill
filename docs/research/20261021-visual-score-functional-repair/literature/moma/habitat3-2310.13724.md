# Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots
- Authors: Xavi Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, et al. (23 authors). arXiv 19 Oct 2023. https://arxiv.org/abs/2310.13724
- What: simulator with humanoid avatars + robot; tasks Social Navigation (find and follow a humanoid) and Social Rearrangement (robot + humanoid move 2 objects); human-in-the-loop tool (30 participants).
- Robot: Boston Dynamics Spot with 7-joint arm; collision modelled as two cylinders (centre + front) because of its long body; arm depth 224x171, 55 deg hFOV; max base speed 2.5 m/s (as fetched). Cylinder radii and arm reach not stated (unverified).
- Scenes: HSSD-200 (211 homes, 18k+ objects); 59 scenes used (37 train / 12 val / 10 test), YCB objects; scene sizes small 68.56 / medium 136.11 / large 846.15 m2.
- Success: SocialNav = within 1-2 m of the humanoid while facing it (dot > 0.5), episode ends on collision, 1500 steps, start >= 5 m away; SocialRearrange = both objects at targets, pick/place counted within 0.15 m. Objects on "open receptacles".
- Failure / layout notes: oracle->learned low-level skills drops Plan-Pop3 SR 77.79% -> 41.96% ("low-level execution failures"); SocialNav collision rate 0.51-0.52; robot must "make way in narrow spaces" and backs away in narrow corridors (qualitative, no width threshold).
- Relevance: long bodies need a two-circle footprint, not a single radius; corridors are where human-robot conflicts occur. No numeric clearance thresholds.
- Tags: sim-benchmark, humanoid, long-horizon
