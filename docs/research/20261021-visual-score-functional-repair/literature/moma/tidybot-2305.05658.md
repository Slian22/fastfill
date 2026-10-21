# TidyBot: Personalized Robot Assistance with Large Language Models
- Authors: Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, Andy Zeng, Shuran Song, Jeannette Bohg, Szymon Rusinkiewicz, Thomas Funkhouser. arXiv May 2023 (v2 11 Oct 2023; IROS 2023 + journal extension). https://arxiv.org/abs/2305.05658
- What: LLM (text-davinci-003) summarises user example placements into rules, CLIP picks categories; robot tidies floor objects into receptacles by pick-and-place or pick-and-toss (top-down grasps).
- Robot: Kinova Gen3 7-DoF + Robotiq 2F-85 on a holonomic powered-caster base. No footprint/reach numbers in this version (see tidybot-pp-2412.10447.md).
- Success: object in correct receptacle: real 85.0%; text benchmark 91.2% unseen / 91.8% seen. Benchmark 96 scenarios, 4 room types, 2-5 receptacles each, 1,076 objects, 87 receptacles; real 8 scenarios x 10 objects x 3 runs (240 objects).
- Failure breakdown (real): overhead-camera localisation 92.5%, CLIP category 95.5%, receptacle/primitive selection 100%, primitive execution 96.2%; 15-20 s per object.
- Scene assumptions: receptacle locations "hard-coded for each scenario", receptacles are obstacles in the occupancy map; "mobile robots cannot drive over objects" so it "would not work well in excessive clutter".
- Relevance: floor clutter blocks the base; receptacles must be approachable; tossing relaxes the receptacle-reach requirement. No numeric clearance.
- Tags: real-home-deploy, open-vocab-nav-pick
