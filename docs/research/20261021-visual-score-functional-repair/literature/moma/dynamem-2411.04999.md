# DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation
- Authors: Peiqi Liu, Zhanqiu Guo, Mohit Warke, Soumith Chintala, Chris Paxton, Nur Muhammad Mahi Shafiullah, Lerrel Pinto. arXiv 2411.04999 (v2 29 May 2025). https://arxiv.org/abs/2411.04999
- What: voxel memory that adds and removes points online so pick-and-drop works when objects move; OK-Robot successor.
- Robot: Hello Robot Stretch SE3, head-mounted RealSense D435.
- Eval: lab, 3 dynamic environments x 10 queries (30); homes, 2 NYC apartments, 17 long-horizon trials; offline DynaBench, 9 environments.
- Results: 70% pick-and-drop on non-stationary objects (lab) vs OK-Robot 30%; homes 9/17.
- Failures: OK-Robot 53.3% of trials failed because a moved object could not be found; DynaMem 6.7% failed to navigate to dynamic objects. Static-goal navigation failures 10% (DynaMem VLM-feature), 13.3% (OK-Robot), 20% (DynaMem mLLM-QA). Home: 8 failures = 4 navigation, 3 manipulation, 1 placement; half of the navigation failures were "a suboptimal location for manipulation" (base pose), half target not found/confused.
- Map parameters: voxels above a 0.2 m ground threshold are obstacles; depth beyond 2 m ignored when removing voxels; executes the first seven waypoints (about 0.7-1 m), rescans, replans. Grid size and inflation radius not stated.
- Relevance: even with a good memory, base-pose quality is half of the home navigation failures -> layouts should leave a good stand-off pose, not just a reachable point.
- Tags: open-vocab-nav-pick, 3D-memory/scene-graph, real-home-deploy, base-placement
