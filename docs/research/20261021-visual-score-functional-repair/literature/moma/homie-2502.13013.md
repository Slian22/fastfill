# HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit
- Authors: Qingwei Ben, Feiyu Jia, Jia Zeng, Junting Dong, Dahua Lin, Jiangmiao Pang (Shanghai AI Lab, CUHK). arXiv 18 Feb 2025; RSS 2025. https://arxiv.org/abs/2502.13013
- What: RL whole-body policy (walk + squat to commanded height) driven by a pedal, plus exoskeleton arms and gloves for teleop (~$500); data used for imitation learning.
- Robot: Unitree G1 (main), Fourier GR-1. Commanded height while walking: G1 0.74 m, GR-1 0.90 m; squat range G1 [-0.24, 0.74] m, GR-1 [-0.30, 0.90] m (reference point not defined in text). Hand reach heights not stated.
- Tasks: squat to pick from lower shelves/sofa and place higher, box transfer between shelves, open oven while stepping back. Imitation learning (15 trials each): Squat Pick 73.3%, Pick & Place 80.0%.
- Failure causes: stair collisions and unstable walking (head-mounted LiDAR moves; low-resolution elevation map). No layout dimensions given.
- Relevance: humanoids widen the reachable height band by squatting, so low receptacles matter less than for fixed-lift wheeled bases; terrain/step-free floors still matter.
- Tags: humanoid, teleop-data, RL-sim2real
