# Robot simulator benchmarks for a text-only VLM (Qwen3.5-4B)

Ticket question: which robot simulator benchmarks with actions can run on one Linux workstation (RTX 5090 32 GB, 45 GB RAM, 24 cores, headless), and which of them have published results for a general VLM with no action head?

Scope: this file reports facts only. It does not recommend a benchmark. The user makes the decision.

Method: web research on 2026-11-10. Each claim has its source URL. "Not found" means that the fetched primary source did not state the fact. A "derived" value is arithmetic on sourced numbers.

## Summary table

"Headless" states what the fetched docs say. It does not state a test result on this workstation.

| Candidate | Backend | Headless support (documented) | Native action space | Higher-level interface available | Published no-action-head VLM result |
|---|---|---|---|---|---|
| LIBERO | MuJoCo (EGL variable in docs) | EGL implied, not documented | 7-D vector | No (in benchmark). Privileged state API used by FAEA | Yes (Claude Opus 4.5, code + privileged state) |
| SimplerEnv | SAPIEN (ManiSkill2; ManiSkill3 branch) | Not stated. Needs Vulkan, NVIDIA GPU | 7-D EE delta (xyz, axis-angle, gripper), 3 Hz or 5 Hz | Not found | No (not found) |
| ManiSkill3 | SAPIEN | Not stated on fetched page. Needs Vulkan for rendering | Selectable PD controllers (joint, EE delta pose 6-D, plus gripper) | Absolute EE pose control; motion planning mentioned | Yes (Claude Opus 4.5, code + privileged state) |
| CALVIN | PyBullet | EGL GPU rendering documented | 7-D absolute or relative Cartesian, or joint, 30 Hz | Not found | No (not found) |
| RoboCasa | robosuite v1.5 (MuJoCo) | Not stated | Not found | Not found | No (not found) |
| VLABench | MuJoCo / dm_control | Not stated | EE pose (position + quaternion) through IK, Franka 7-DoF | Yes: skill library with DSL output for VLMs | Yes (VLMs through DSL skills; exact numbers not found as text) |
| RLBench | CoppeliaSim 4.1.0 + PyRep | Needs an X server; headless GPU X setup documented | Configurable action modes (example: joint velocity) | Not stated on fetched page | Yes (Manipulate-Anything with GPT-4V; baselines VoxPoser, CaP) |
| EmbodiedBench EB-Manipulation | CoppeliaSim + PyRep (extends VLMBench) | Yes, through a virtual X display (startx.py) | Discrete 7-D [X, Y, Z, Roll, Pitch, Yaw, Gripper] bins | Yes: detection boxes and object poses in the prompt | Yes (GPT-4o, Claude, Gemini, Qwen2/2.5-VL, InternVL2.5) |
| Meta-World | MuJoCo | Not stated | 4-D: EE dx, dy, dz, gripper, in [-1, 1] | Not found | Yes (Claude Opus 4.5, code + privileged state) |
| BEHAVIOR-1K | OmniGibson on Isaac Sim | Headless install mode documented; headless run not stated | Not found | Not found | No verified number (CaP-Bench wraps it; numbers not found) |
| CaP-Bench (CaP-X, 2026) | Wraps LIBERO-PRO, Robosuite, BEHAVIOR | Inherits from the wrapped simulators | Code-as-policies at levels S1 to S4 | Yes: API levels from pick-and-place to raw primitives | Yes (aggregate only; per-model numbers not found) |

## LIBERO

- What it is: a benchmark "for studying knowledge transfer in multitask and lifelong robot learning problems." Repository: https://github.com/Lifelong-Robot-Learning/LIBERO. Licence: MIT for code, CC BY 4.0 for datasets (same URL).
- Install cost:
  - The README pins Python 3.8.13 and `torch==1.11.0+cu113` (https://github.com/Lifelong-Robot-Learning/LIBERO).
  - Backend: the README does not name robosuite. It uses the variable `MUJOCO_EGL_DEVICE_ID`, which implies MuJoCo with EGL rendering (same URL).
  - Headless rendering: not documented in the README. Known install issues: not found.
- Action space: the README example uses `dummy_action = [0.] * 7`, a 7-D action (same URL). The meaning of each element and the control rate: not found.
- Higher-level interface: not found in LIBERO. The FAEA paper drove LIBERO through Python calls to simulator APIs with "absolute end-effector position control" and privileged ground-truth state (https://arxiv.org/html/2601.20334).
- Task structure: 130 tasks in four suites, LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-100. LIBERO-100 splits into LIBERO-90 and LIBERO-10 (https://github.com/Lifelong-Robot-Learning/LIBERO). A user selects one suite by its name (same URL).
- Time cost:
  - The OpenVLA evaluation script sets `max_steps` to 220 (Spatial), 280 (Object), 300 (Goal), 520 (LIBERO-10), and 400 (LIBERO-90) (https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py).
  - The same script uses 50 trials per task and a 256 px render (same URL). This is the OpenVLA convention, not a rule of LIBERO itself.
  - Derived: one 10-task suite at 50 trials is 500 episodes.
  - Wall-clock time per episode: not found.
- Published no-action-head results:
  - FAEA, "Demonstration-Free Robotic Control via LLM Agents", Claude Opus 4.5, code that calls simulator APIs, privileged state, no RGB (https://arxiv.org/html/2601.20334).
  - FAEA baseline average 84.9%: Spatial 90.0, Object 100.0, Goal 80.0, Long 83.3. With human coaching: average 88.2% (same URL).
  - CaP-Agent0 (code-as-policies, training-free): 18% on LIBERO-PRO (30 tasks). On the same page, pi0.5 gets 13% (https://capgym.github.io/).
  - Result for a general VLM from RGB images on standard LIBERO: not verified at a primary source. See "Open items".

## SimplerEnv

- What it is: a benchmark to evaluate real-world manipulation policies (RT-1, Octo) in simulation for the Google Robot and WidowX + Bridge setups. Repository: https://github.com/simpler-env/SimplerEnv. Licence: MIT (same URL).
- Install cost:
  - Backend: SAPIEN with the "CPU based ManiSkill2 benchmark". A `maniskill3` branch ports the Bridge tasks to ManiSkill3 with GPU parallel simulation, "10-15x faster" (same URL).
  - Python 3.10 or 3.11. Numpy must stay below 2.0, or "errors in IK might occur in pinocchio" (same URL).
  - Full install needs CUDA ">=11.8 and < 13" and `tensorflow==2.15.0` (same URL).
  - An NVIDIA GPU is required, "ideally RTX". Vulkan errors refer to the ManiSkill troubleshooting docs. Headless operation: not stated (same URL).
- Action space: `action[:3]` delta xyz, `action[3:6]` delta rotation in axis-angle, `action[6:7]` gripper. Control rate: Google Robot 3 Hz, Bridge 5 Hz, simulation about 500 Hz (same URL).
- Higher-level interface: not found.
- Task structure:
  - Google Robot: pick coke can, pick object, move near, open drawer, close drawer, place in closed drawer.
  - WidowX/Bridge: spoon on towel, carrot on plate, stack cube, put eggplant in basket.
  - Two settings: Visual Matching and Variant Aggregation (same URL).
  - A user selects one robot setup or one task.
- Time cost: max episode steps not found. The number of episodes is not stated; the scripts sweep object and robot poses (same URL). Wall-clock time: not found.
- Published no-action-head results: not found. A search returned only VLA models with trained action outputs.

## ManiSkill3

- What it is: an open-source framework for robot simulation and training, "powered by SAPIEN", focused on manipulation. Repository: https://github.com/haosulab/ManiSkill. Licence: rigid-body environments under permissive licences "(e.g., Apache-2.0)"; assets under CC BY-NC 4.0 (same URL).
- Install cost:
  - `pip install --upgrade mani_skill` plus a matching torch (https://github.com/haosulab/ManiSkill).
  - Python version: not stated on the fetched pages.
  - "We currently best support Linux based systems." Rendering requires "a GPU with the Vulkan driver installed". State-only simulation needs no extra dependency (https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html).
  - Known issues: Vulkan `ErrorInitializationFailed`, missing `nvidia_icd.json` files, and server driver packages (same URL). Headless rendering is not addressed directly on that page.
- Action space: the user selects a controller. Panda arm options include `arm_pd_joint_pos` (7), `arm_pd_joint_delta_pos` (7), `arm_pd_ee_delta_pos` (3), and `arm_pd_ee_delta_pose` (6), plus a 1-D gripper (https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html). Default control rate: not found.
- Higher-level interface: non-delta EE control "works more similarly to motion planning where you pick a target pose" (same URL). A motion-planning solution module: not found on the fetched pages.
- Task structure: families include table-top, drawing/cleaning, and dexterous tasks for single-arm robots, mobile manipulators, and humanoids. A total count is not stated (https://github.com/haosulab/ManiSkill). A user selects one task by its environment ID.
- Time cost: the project claims "30,000+ FPS on a 4090 GPU" for RGBD and segmentation data (same URL). Steps per episode and the standard number of evaluation episodes: not found.
- Published no-action-head results: FAEA with Claude Opus 4.5, code plus privileged state: 85.7% (60/70) on 14 tasks with 5 seeds each. PegInsertionSide 0%, TurnFaucet 40%, PlugCharger 60% (https://arxiv.org/html/2601.20334).

## CALVIN

- What it is: "an open-source simulated benchmark to learn long-horizon language-conditioned tasks". Repository: https://github.com/mees/calvin. Licence: MIT (same URL).
- Install cost: PyBullet backend, Python 3.8. If `pyhash` fails, downgrade setuptools below 58. EGL moves rendering "from cpu (which is the default) to gpu" (https://github.com/mees/calvin).
- Action space: absolute Cartesian pose or relative Cartesian displacement (position 3, orientation 3, gripper 1), or joint action. Actions go to the robot "at 30hz" (https://github.com/mees/calvin; https://arxiv.org/html/2112.03227v4).
- Higher-level interface: not found.
- Task structure: 34 tasks in four environments A, B, C, D. The evaluation uses 1000 unique chains of five tasks (https://arxiv.org/html/2112.03227v4). Dataset splits are D, ABC, ABCD (https://github.com/mees/calvin). A user selects a split, or one task from the 34.
- Time cost: derived, 1000 chains of 5 subtasks is up to 5000 subtask attempts. Max steps per subtask: not found. Wall-clock time: not found.
- Published no-action-head results: not found.

## RoboCasa

- What it is: "a large-scale simulation framework for training generally capable robots to perform everyday tasks". Repository: https://github.com/robocasa/robocasa. Licence: MIT for code, CC BY 4.0 for assets and datasets (same URL).
- Install cost: robosuite v1.5 backend (MuJoCo implied by `mjpython`). Python 3.11. "Assets to be downloaded are around 10GB." A numba/numpy conflict fix is in the README. Headless rendering: not stated (same URL).
- Action space and control rate: not found.
- Higher-level interface: not found.
- Task structure: RoboCasa365 has "365 tasks", with 300 composite tasks named in an update (https://github.com/robocasa/robocasa). The `pretrain` split has 2500 kitchen scenes. The `target` split has 10 kitchen scenes (https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html). Atomic task count: not found.
- Time cost: "we randomly sample 50 scenarios to run evaluation rollouts on" per task. Horizons increased by 1.5x in v1.0.1; absolute values not found (same URL). Wall-clock time: not found.
- Published no-action-head results: not found.

## VLABench

- What it is: a benchmark for VLA models, embodied agents, and VLMs on language-conditioned manipulation with long-horizon reasoning. Repository: https://github.com/OpenMOSS/VLABench. Licence: MIT (same URL).
- Install cost: MuJoCo and dm_control backend (https://arxiv.org/html/2412.18194). Python 3.10. Assets come from `scripts/download_assets.py`. A Docker quick start exists. Headless rendering: not stated (https://github.com/OpenMOSS/VLABench).
- Action space: EE pose as 3-D position plus quaternion, resolved by IK to 7 joints of a Franka Panda (https://arxiv.org/html/2412.18194). Control rate: not found as a stated value.
- Higher-level interface: yes. VLMs output a DSL skill sequence from a skill library. Two modes: non-interactive scoring against a ground-truth DAG, and interactive execution through a controller (https://arxiv.org/html/2412.18194). The script `scripts/evaluate_vlm.py` supports GPT-4V, Qwen2-VL, InternVL2, MiniCPM-V-2.6, GLM-4V, LLaVA-NeXT (https://github.com/OpenMOSS/VLABench).
- Task structure: 60 primitive tasks and 40 composite tasks in five dimensions: Mesh & Texture, Spatial, Common Sense & World Knowledge, Semantic, Physical Law (https://arxiv.org/html/2412.18194). A user selects one track, one dimension, or one task.
- Time cost: primitive tasks average about 120 timesteps; composite tasks exceed 500 (https://arxiv.org/html/2412.18194). One task in one process takes about 30 minutes to 1 hour; the example uses `--n-sample 20` (https://github.com/OpenMOSS/VLABench).
- Published no-action-head results: the paper evaluates VLMs through the DSL, but the scores are in Figure 5 radar charts. The only text value: models other than GPT-4o "scored around 20 points" on reasoning (https://arxiv.org/html/2412.18194). Exact numbers: not found.

## RLBench

- What it is: "an ambitious large-scale benchmark and learning environment" for vision-guided manipulation. Repository: https://github.com/stepjam/RLBench. Licence: not found (the licence text did not load).
- Install cost:
  - CoppeliaSim v4.1.0 and PyRep. The download is "Edu V4.1.0 for Ubuntu 20.04". Python version: not stated (https://github.com/stepjam/RLBench).
  - Headless use needs an X server: `nvidia-xconfig`, then `sudo nohup X :99 & disown`. Non-sudo use needs an edit of `/etc/X11/Xwrapper.config` (same URL).
- Action space: configurable action modes; the example uses `MoveArmThenGripper` with `JointVelocity` and `Discrete`. Version 1.2.0 changed the action mode API (same URL). An EE pose via planning mode: not stated on the fetched page.
- Higher-level interface: Manipulate-Anything used a "6 DoF end-effector pose" with the M2T2 grasp model and OMPL motion planning (https://arxiv.org/html/2406.18915v3).
- Task structure: few-shot sets FS10, FS25, FS50, FS95 and multi-task sets MT15, MT30, MT55, MT100. Images default to 128x128 (https://github.com/stepjam/RLBench). A user selects one task by its class.
- Time cost: the Gym example uses `episode_length = 40`, which is not a standard (same URL). Standard episode count and wall-clock time: not found.
- Published no-action-head results (https://arxiv.org/html/2406.18915v3, Table 1, 14 tasks, 3 seeds):
  - Manipulate-Anything uses GPT-4V and Qwen-VL.
  - Derived averages of Table 1: MA about 58.9%, VoxPoser about 36.1%, CaP about 20.7%.
  - The CaP and VoxPoser baselines received ground-truth state or segmented point clouds.

## EmbodiedBench (EB-Manipulation)

- What it is: a benchmark of 1,128 tasks in four environments for multimodal LLMs as embodied agents. EB-Manipulation "extends VLMBench" (https://arxiv.org/html/2502.09560). Repository: https://github.com/EmbodiedBench/EmbodiedBench. Licence: not found.
- Install cost: CoppeliaSim V4.1.0 "required for Ubuntu 20.04", PyRep, and Git LFS. Headless servers run `startx.py` for a virtual display, `DISPLAY=:1`. Local models load with `model_type=local`; the example is `Qwen/Qwen2-VL-7B-Instruct` (https://github.com/EmbodiedBench/EmbodiedBench). Python version: not stated.
- Action space: a 7-D vector [X, Y, Z, Roll, Pitch, Yaw, Gripper]. Position uses 100 bins; orientation uses 120 bins (https://arxiv.org/html/2502.09560).
- Higher-level interface: the input has detection boxes with index markers and object pose estimates. The output is JSON, and the agent can plan many steps at once (same URL).
- Task structure: 228 instances in five subsets: Base, Common Sense, Complex Instruction, Visual Appearance (36), Spatial Awareness. Each subset other than Visual Appearance has 48 (same URL). A user selects one subset.
- Time cost: at most 15 environment steps per episode (same URL). Wall-clock time: not found. A 10% debug sample is available through `down_sample_ratio` (https://github.com/EmbodiedBench/EmbodiedBench).
- Published no-action-head results (average success %, https://arxiv.org/html/2502.09560 Table 3):
  - GPT-4o 28.9, Claude-3.5-Sonnet 25.4, Gemini-1.5-Pro 21.1.
  - Qwen2-VL-7B 0.0, Qwen2.5-VL-7B 9.6, Qwen2.5-VL-72B 16.2.
  - InternVL2.5-8B 7.0, InternVL2.5-38B 15.8, InternVL2.5-78B 18.0.
  - Qwen2.5-VL-7B by subset: Base 8.3, Common 8.3, Complex 8.3, Visual 5.6, Spatial 16.7.

## Meta-World

- What it is: an "open source benchmark for developing and evaluating multi-task and meta reinforcement learning algorithms". Repository: https://github.com/Farama-Foundation/Metaworld. Licence: MIT (same URL).
- Install cost: `pip install metaworld`. Python 3.11, 3.12, 3.13 on Linux and macOS. MuJoCo backend; version not stated. Headless rendering: not stated (same URL).
- Action space: `Box(-1.0, 1.0, (4,), float32)`: EE dx, dy, dz, and gripper (https://metaworld.farama.org/benchmark/action_space/). Control rate: not found.
- Higher-level interface: not found.
- Task structure: 50 tasks. Splits MT1, MT10, MT50, ML1, ML10, ML45 (https://github.com/Farama-Foundation/Metaworld). A user selects one task through MT1.
- Time cost: the README example loops 500 steps; a max episode length is not stated (same URL). Standard episode count and wall-clock time: not found.
- Published no-action-head results: FAEA with Claude Opus 4.5, code plus privileged state: 96.0% (48/50 tasks); 100% with coaching (https://arxiv.org/html/2601.20334).

## BEHAVIOR-1K

- What it is: OmniGibson, a simulator "built upon NVIDIA's Omniverse and Isaac Sim platforms", with the BEHAVIOR Challenge of "100 full-length household tasks" (https://behavior.stanford.edu/getting_started/installation.html; https://behavior.stanford.edu/challenge/index.html). Licence: the installer accepts the "NVIDIA Isaac Sim EULA" and the "BEHAVIOR Dataset License"; a code licence is not stated.
- Install cost:
  - Requirements: "Ubuntu 22.04+", "32GB+" RAM, "NVIDIA RTX 2070+", "8GB+" VRAM. Disk and Python version: not stated (https://behavior.stanford.edu/getting_started/installation.html).
  - Docker is "temporarily unavailable". The first run can take "up to 5 minutes". The renderer can hang at "HydraEngine rtx failed creating scene renderer" (same URL).
  - A headless install mode exists. Headless simulation is not stated (same URL).
- Action space: not found.
- Higher-level interface: not found. CaP-Bench wraps BEHAVIOR with code APIs (https://capgym.github.io/).
- Task structure: 100 tasks; one track uses "RGB + depth + proprioception" (https://behavior.stanford.edu/challenge/index.html).
- Time cost: not found.
- Published no-action-head results: the challenge baselines are pi0.5 and GR00T N1.7 (same URL). CaP-Bench includes BEHAVIOR, but its per-benchmark numbers are not found. The Embodied Agent Interface paper (https://arxiv.org/abs/2410.07166) benchmarks LLMs for embodied decision making; its simulators and numbers were not on the fetched page.

## CaP-Bench (CaP-X, 2026)

- What it is: a benchmark of LM agents that write code to control robots, with CaP-Agent0 (training-free) and CaP-RL (https://capgym.github.io/). Repository and paper: links are placeholders on the page. Licence: not found.
- Install cost: inherits LIBERO-PRO, Robosuite, and BEHAVIOR (same URL).
- Action interface: code-as-policies at API levels from S1 (high-level pick-and-place) to S4 (raw primitives) (same URL).
- Task structure: "100+ Manipulation Tasks"; LIBERO-PRO has 30 tasks (same URL).
- Time cost: not found.
- Published no-action-head results: frontier models reach "over 30% average success" zero-shot. CaP-Agent0 gets 18% on LIBERO-PRO. CaP-RL lifts Qwen 2.5 Coder 7B from 20% to 72% in simulation (same URL). Per-model numbers: not found.

## Open items

- LIBERO and SimplerEnv pin old stacks (torch 1.11.0+cu113; TensorFlow 2.15). Support for the RTX 5090 with these builds: not checked.
- CoppeliaSim 4.1.0 targets Ubuntu 20.04 (RLBench, EmbodiedBench). Operation on this workstation's OS: not checked.
- Headless rendering on this machine: not tested for any candidate. ManiSkill3, SimplerEnv, RoboCasa, VLABench, and Meta-World docs that I fetched do not state it.
- Wall-clock time per episode: not found for any candidate except the VLABench estimate.
- VLABench exact VLM scores (Figure 5) and the interactive execution numbers: not found as text.
- CALVIN max steps per subtask; LIBERO control rate; RoboCasa action space and control rate; BEHAVIOR action space; RLBench licence: not found.
- A 2026 search result names MotorMind (arXiv 2609.38078, LIBERO-PRO 66.7%) and World Action Agent (arXiv 2609.29964, LIBERO-PRO 28.9%). I did not verify them at the primary source. Their backbone names did not match models that I could confirm.
- EASG (arXiv 2610.12832) claims VLM text output as actions with no action head. Benchmark, backbone, and numbers: not on the abstract page. Project page: https://general-robo.github.io/EASG/.
- A secondary page reports a LIBERO-10 frontier-model result ("GPT-6 Astra", 44%). Primary source: not found. Treat it as unverified.
- A search snippet says that VLM4VLA (arXiv 2601.03309) finds SimplerEnv Bridge and LIBERO-Long scores do not correlate with general VLM scores, unlike CALVIN. Not verified at the primary source.
- Embodied3DBench (2026) tests VLM spatial intelligence zero-shot, with Qwen3-VL-4B. It appears to be a question benchmark, not closed-loop control. Primary source: not fetched.
- No result found for Qwen3.5-4B or any Qwen3-VL model on a closed-loop control benchmark without an action head.
