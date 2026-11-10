# Qwen spatial outputs and action interfaces for a text-output VLM

Date of research: 2026-11-10. Scope: web sources only. Each claim has its source URL. "Not found" means that this research did not find the fact. It does not mean that the fact does not exist.

Source label: **[3.5]** means that the source describes Qwen3.5. **[3-VL]** means that the source describes Qwen3-VL.

## Part 1. Qwen outputs

### 1.1 What the Qwen3.5 sources say

- [3.5] The Qwen3.5-4B model card gives these 4B scores: ERQA 54.0, CountBench 96.3, RefCOCO(avg) 88.1, EmbSpatialBench 81.3, RefSpatialBench 54.6, LingoQA 74.4, Hypersim 12.5, Nuscene 9.9, ScreenSpot Pro 60.3. https://huggingface.co/Qwen/Qwen3.5-4B
- [3.5] The model card has no text on the output format of points, boxes, or coordinates. It has no text on robot or embodied use. Only the heading "Spatial Intelligence" introduces the rows above. https://huggingface.co/Qwen/Qwen3.5-4B
- [3.5] The Qwen3.5 blog (https://qwen.ai/blog?id=qwen3.5) renders with JavaScript. The fetch returned only the word "Qwen". Blog content: not found.
- [3.5] A dedicated Qwen3.5 GitHub repository: not found. The QwenLM organization listing (https://api.github.com/orgs/QwenLM/repos) has no repository with "3.5" in the name. The model card links only to Qwen-Agent and qwen-code.
- [3.5] A Qwen3.5 cookbook for grounding or spatial output: not found. All cookbooks below are in the Qwen3-VL repository.

### 1.2 Coordinate convention

- [3-VL] The 2D cookbook states: "Qwen3-VL's default coordinate system has been changed from the absolute coordinates used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000. (You don't need to calculate the resized_w)". https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb
- [3-VL] The cookbook code converts to pixels with `value / 1000 * width` for x and `value / 1000 * height` for y. Box order in the code is [x1, y1, x2, y2]. A comment in the plot function says "[y1 x1 y2 x2]", which disagrees with the code. Same URL.
- [3-VL] The technical report (Section 3.2.4) states "a normalized coordinate system scaled to the range [0,1000]" for boxes and points. https://arxiv.org/html/2511.21631
- [3-VL] All cookbook examples call `qwen3-vl-235b-a22b-instruct` through an API. The 3D cookbook uses the placeholder `"your model here"`. No cookbook example uses a 2B to 8B model.

### 1.3 2D boxes [3-VL]

- Prompt (exact): `locate every instance that belongs to the following categories: "plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person". Report bbox coordinates in JSON format.`
- Example output (first item): `{"bbox_2d": [74, 361, 324, 808], "label": "plate/dish"}`
- Single-object prompt (exact): `Locate the top right brown cake, output its bbox coordinates using JSON format.`
- Source: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb

### 1.4 2D points [3-VL]

- The spatial cookbook states: "For spatial pointing tasks, Qwen3-VL now support these formats: point-format: JSON `{"point_2d": [x, y], "label": "object name/description"}`". https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/spatial_understanding.ipynb
- JSON prompt (exact, affordance section): `Locate the free space on the white table on the right in this image. Output the point coordinates in JSON format.`
- Example output: `[{"point_2d": [925, 618], "label": "point_1"}]`. Same URL.
- JSON prompt with attributes (exact): `Locate every person inside the football field with points, report their point coordinates, role(player, referee or unknown) and shirt color in JSON format like this: {"point_2d": [x, y], "label": "person", "role": "player/referee/unknown", "shirt_color": "the person's shirt color"}`. Example output item: `{"point_2d": [346, 328], "label": "person", "role": "referee", "shirt_color": "black"}`. https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb
- XML prompt (exact): `point to the rolling pin on the far side of the table, output its coordinates in XML format <points x y>object</points>`. Example output: `<points x1="707" y1="233" alt="rolling pin on the far side of the table">rolling pin on the far side of the table</points>`. Same URL.

### 1.5 3D grounding [3-VL]

- Prompt (exact): ``Find all cars in this image. For each car, provide its 3D bounding box. The output format required is JSON: `[{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"category"}]`.``
- Short prompt (exact): `Locate the black chair in image and provide 3D bounding boxes results in JSON format.` Parsed output: `{'bbox_3d': [-0.4, -0.26, 2.71, 0.76, 0.96, 0.78, 0.16, 0.13, 0.08], 'label': 'black chair'}`.
- Convention: the center is "in camera coordinates (meters)". Sizes are in meters. The text says that the angles are in radians. The plot code multiplies the angles by 180 before it converts them to radians. The text and the code disagree.
- Camera: the cookbook says that 3D accuracy "highly relies on camera parameters". It loads fx, fy, cx, cy from a file. Without a file, it builds intrinsics from fov=60 degrees.
- Source: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/3d_grounding.ipynb
- The technical report (Section 3.2.5) says that 3D samples use "9-DoF 3D bounding box annotations in a structured JSON format" and estimate 3D position "from a single monocular image". https://arxiv.org/html/2511.21631

### 1.6 Embodied, affordance, trajectory, and robot material

- [3-VL] Spatial cookbook section "2. Perceive Object Affordances". It names prompts such as "Identify a graspable point on the cup's handle". The output is a `point_2d` point (see 1.4). https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/spatial_understanding.ipynb
- [3-VL] Spatial cookbook section "3. Integrate Spatial Reasoning and Action Planning". The prompts are multiple choice over arrows drawn on the image, for example `What color arrow should the robot follow to move the apple in between the green can and the orange? Choices: A. Red. B. Blue. C. Green. D. Orange.` A video prompt asks a robot to fill in "turn back", "turn left", or "turn right" steps for navigation. Same URL.
- [3-VL] Trajectory output in the cookbooks: not found. Grasp pose output: not found.
- [3-VL] Technical report Section 5.8 "Embodied and Spatial Understanding" names ERQA, VSIBench, EmbSpatial, RefSpatial, and RoboSpatialHome. The text gives flagship 235B scores only (EmbSpatial 84.3, RefSpatial 69.9, RoboSpatialHome 73.9). https://arxiv.org/html/2511.21631
- [3.5] Qwen-VLA is "built upon Qwen3.5-4B (vision-language backbone) and a 1.15B DiT flow-matching action decoder". It is trained on robot data. Reported Instruct scores: LIBERO 97.9, RoboCasa-GR1 56.7, Simpler-WidowX 73.7, RoboTwin-Easy 86.1, RoboTwin-Hard 87.2. https://github.com/QwenLM/Qwen-VLA. A weight release statement: not found.
- [3.5] Qwen-RobotManip combines "a Qwen3.5-4B vision-language backbone with a flow-matching DiT action expert". The README states: "There is currently no plan to release the model weights for Qwen-RobotManip or Qwen-RobotNav." https://github.com/QwenLM/Qwen-RobotManip
- [3-VL] Qwen-RobotNav is "built on Qwen3-VL" and uses "a shared waypoint-prediction interface". https://github.com/QwenLM/Qwen-RobotNav
- A zero-shot result of plain Qwen3.5-4B on a robot simulator: not found.

### 1.7 How RefSpatialBench prompts and scores

- The benchmark has 277 rows: location 100, placement 100, unseen 77. The card warns: "If your model is not trained with RefSpatial, Unseen set should not be used for evaluation." https://huggingface.co/datasets/BAAI/RefSpatial-Bench
- The `suffix` field is "the format used by RoboRefer". "Different models may use different suffixes or none." The suffix text: `Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image.` Same URL.
- Gemini protocol: `"Locate the points of " + sample["object"] + "."`. The output is JSON `{"point": [y, x], "label": ...}` normalized to 0-1000. Same URL.
- Molmo protocol: `"Locate several points of " + sample["object"] + "."`. The output is XML `<points x1=".." y1=".." ... />` normalized to 0-100. Same URL.
- Score: "The main metric is average success rate — the percentage of predictions falling within the mask." Same URL.
- The official evaluation code is at https://github.com/Zhoues/RoboRefer.
- The prompt that Qwen used for its 54.6 score: not found. A Qwen-specific protocol on the card: not found.

### 1.8 How ERQA prompts and scores

- ERQA has 400 multiple-choice VQA questions. https://arxiv.org/pdf/2503.20020
- The questions interleave images and text. The answer is one letter (A, B, C, D). https://github.com/embodiedreasoning/ERQA
- The harness sends the question text and the images in order. An added system instruction in the harness: not found (search by keyword only).
- Score (exact code, `eval_harness.py` line 548): `is_correct = response_text.replace(".", "").strip().lower() == answer.strip().lower()`. A reply with any text other than the letter scores as wrong. https://github.com/embodiedreasoning/ERQA/blob/main/eval_harness.py
- The Gemini Robotics report says that chain-of-thought prompting raised the Gemini 2.0 ERQA score. https://arxiv.org/pdf/2503.20020
- The prompt that Qwen used for its 54.0 score: not found.

## Part 2. Action interfaces for a model with no action head

### 2.1 Pixel point or keypoint, then a controller or planner

- **PIVOT** (https://arxiv.org/html/2402.07872). The VLM picks numbered candidate markers drawn on the image, over several rounds. Navigation maps the pixel to 3D with the on-board depth camera. Manipulation arrows are relative Cartesian end-effector moves. Simulator: RAVENS pick and place. RAVENS success numbers: not found in the extracted text. Models: GPT-4V by default, and four unnamed Gemini sizes. Open model results: none.
- **MOKA** (https://arxiv.org/html/2403.03174). GPT-4V picks grasp, function, and target keypoints from marked candidates, and waypoint tiles on a 5x5 grid. Extra system: GroundedSAM, depth and camera parameters to lift points to 3D, and an antipodal grasp sampler. Setting: real Franka only. Zero-shot per-task success: 0.5 to 1.0 per subtask over 10 trials. Simulator results: none.
- **RoboPoint** (https://arxiv.org/html/2406.10721). The model outputs a list of 2D points normalized to 0-1. It is fine-tuned from Vicuna-v1.5-13B. It is not zero-shot. Extra system: a depth map to lift points, then a motion planner. Where2Place accuracy: RoboPoint 46.77, GPT-4o 29.06, Qwen-VL 10.49. Real manipulation: RoboPoint 41/70, GPT-4V 13/70. Simulator: CoppeliaSim navigation only.
- **ReKep** (https://arxiv.org/html/2409.01652v2). GPT-4o writes Python constraint functions over numbered keypoints. Extra system: DINOv2, SAM, calibrated RGB-D cameras, a point tracker, AnyGrasp, and a SciPy solver. Setting: real robots in the main text. Success: ReKep (Auto) 44.3%, VoxPoser 10.0%. Appendix A.12 lists simulation experiments. Its content: not found.
- **Molmo-style pointing.** Molmo emits XML points on a 0-100 scale (RefSpatialBench card, 1.7). A zero-shot simulator manipulation result that uses Molmo points: not found.

### 2.2 Choice among named skills

- **SayCan** (https://arxiv.org/html/2204.01691v2). The LLM scores each skill text by probability. Learned value functions weight the scores. Extra system: 551 skills trained by BC and RL, plus their value functions. Setting: real and mock kitchen. PaLM-SayCan (540B): 84% planning, 74% execution (mock); 81% and 60% (real). FLAN-SayCan (137B): 70% and 61% (real). A size ablation (8B, 62B, 540B) exists in Appendix Table 7. Its numbers: not found. The open-source tabletop version uses CLIPort, ViLD, and GPT-3. Its numbers: not found.
- **EmbodiedBench EB-ALFRED and EB-Habitat** (https://arxiv.org/html/2502.09560). The model picks one high-level skill per step, for example "find an apple". EB-ALFRED has 171 to 298 actions. EB-Habitat has 70 skills. Best: Claude-3.7-Sonnet 67.7 (EB-ALFRED table), Claude-3.5-Sonnet 68.0 (EB-Habitat). 7B to 8B models: InternVL3-8B 10.3 and 24.3; Qwen2.5-VL-7B 4.7 and 14.3; Qwen2-VL-7B 1.7 and 18.3.
- **Agentic Robot on LIBERO** (https://arxiv.org/html/2505.23450v2). GPT-4o plans subgoals from an atomic skill library. OpenVLA (7B) executes the subgoals. LIBERO average: 79.6. A 2B to 8B planner result: not found.
- A zero-shot VLM skill planner result on RoboCasa: not found.

### 2.3 Generated policy code

- **Code as Policies** (https://arxiv.org/pdf/2209.07753). The LLM writes Python that calls perception and pick-and-place APIs. Model: code-davinci-002. In simulation, the APIs list objects and locations "via a scripted object detector". This is privileged simulator state. Real robots use MDETR or ViLD. Simulated tabletop (UR5e, blocks and bowls, 50 trials per task), CaP success: long-horizon 97.2 (seen), 97.6 (unseen attributes), 80.0 (unseen attributes and instructions); spatial-geometric 89.3, 73.3, 62.0. Smaller models (6.7B GPT-3, code-cushman-001) appear only in code-generation pass rates, not in robot success.
- **VoxPoser** (https://arxiv.org/html/2307.05973). GPT-4 writes Python that builds 3D value maps. Extra system: OWL-ViT, SAM, XMem, RGB-D cameras, a motion planner, and an operational space controller. Simulator: custom SAPIEN block world, 13 tasks. Zero-shot success: 58.8% to 77.5% by split. Real world: 88.0% static. Smaller model results: none.

### 2.4 Text tokens for end-effector moves, zero-shot

- **Language Models as Zero-Shot Trajectory Generators** (https://arxiv.org/html/2310.11604v2). GPT-4 outputs end-effector poses (x, y, z, yaw) inside `<trajectory>` tags, or Python that makes the poses, plus gripper commands. Extra system: `detect_object` returns 3D boxes from LangSAM and camera calibration. Setting: real Sawyer. In a comparison with ground-truth poses, the prompt gets 57.3% over 30 tasks. Llama 2 70B gives executable code in 36.0% of cases. Its success rate is in a figure only: not found as a number.
- **EmbodiedBench EB-Manipulation** (https://arxiv.org/html/2502.09560). The model outputs a 7-D discrete action, for example `[57, 61, 20, 10, 60, 25, 1]`. Position uses 100 bins and orientation uses 120 bins. Extra system: YOLO boxes with index markers and object pose estimates. The task suite extends VLMBench. Success: GPT-4o 28.9, Qwen2.5-VL-7B 9.6, InternVL2.5-8B 7.0, Llama-3.2-11B 0.9, Qwen2-VL-7B 0.0.

### 2.5 Where results exist only for frontier models

- PIVOT, MOKA, ReKep, VoxPoser, Code as Policies, and the trajectory-generator paper report zero-shot results for GPT-4V, GPT-4o, GPT-4, Codex, or Gemini only.
- The only zero-shot simulator results for open 7B to 8B VLMs found here are in EmbodiedBench. They are low: 0.0 to 9.6 on EB-Manipulation, 1.7 to 10.3 on EB-ALFRED.
- A zero-shot robot simulator result for any 2B to 4B VLM: not found.

## Comparison table

| Interface | Model output | Extra system needed | Benchmarks | Best published zero-shot result | Smallest model with a published result |
|---|---|---|---|---|---|
| PIVOT | Labels of drawn candidates, iterated | Candidate sampler; depth camera; Cartesian controller | RAVENS (sim); real nav and manipulation | Real nav up to 100% per task (GPT-4V); RAVENS numbers not found | Unnamed small Gemini size; no open model |
| MOKA | Keypoints and grid tiles from marked candidates | GroundedSAM; depth and calibration; grasp sampler | Real Franka only | 0.5 to 1.0 per subtask (GPT-4V) | GPT-4V only |
| RoboPoint | 2D points, 0-1 | Depth map; motion planner | Where2Place; CoppeliaSim nav; real | Not zero-shot (fine-tuned) | 13B (fine-tuned) |
| ReKep | Python cost functions on keypoints | DINOv2; SAM; calibrated RGB-D; tracker; AnyGrasp; solver | Real robots (sim appendix not read) | 44.3% auto (GPT-4o) | GPT-4o only |
| Skill choice (SayCan) | Score per skill text | Trained skills and value functions | Real kitchen | 74% execution, mock kitchen (PaLM 540B) | 137B FLAN (8B ablation numbers not found) |
| Skill choice (EmbodiedBench) | One skill per step | Simulator skill executors | EB-ALFRED; EB-Habitat | 67.7 and 68.0 (Claude) | Qwen2.5-VL-7B: 4.7 and 14.3 |
| Code as Policies | Python calling perception and primitives | Scripted detector (sim state) or MDETR/ViLD; pick-place primitive | Sim tabletop (UR5e); real | 97.6 long-horizon, unseen attributes (Codex) | Codex only for robot success |
| VoxPoser | Python that builds 3D value maps | OWL-ViT; SAM; XMem; RGB-D; planner; controller | SAPIEN sim (13 tasks); real | 77.5% sim split; 88.0% real (GPT-4) | GPT-4 only |
| Direct pose text (trajectory generator) | End-effector poses in tags, or code | 3D boxes from LangSAM and calibration | Real Sawyer; ground-truth pose comparison | 57.3% over 30 tasks (GPT-4) | Llama 2 70B (executability 36.0% only) |
| Direct pose text (EB-Manipulation) | 7-D binned action vector | YOLO boxes with markers; object poses | EB-Manipulation (VLMBench based) | 28.9 (GPT-4o) | Qwen2.5-VL-7B: 9.6 |

## Open items

- The Qwen3.5 blog content is not readable without JavaScript. Check it in a browser for a spatial or embodied section.
- Confirm that Qwen3.5 uses the same `point_2d`, `bbox_2d`, and `bbox_3d` formats and the 0-1000 scale. All format sources found are Qwen3-VL.
- Find the prompts that Qwen used to get the RefSpatialBench 54.6 and ERQA 54.0 scores.
- Check the Qwen-VLA technical report (https://arxiv.org/abs/2605.30280) for a zero-shot Qwen3.5-4B baseline and for a weight release.
- Read the Qwen3-VL technical report tables for 2B, 4B, and 8B embodied scores (text after character 100000 not read).
- Read PIVOT Figure 8 and the RAVENS results, the ReKep simulation appendix A.12, and SayCan Table 7 (8B and 62B).
- Search for zero-shot pointing results with Molmo or other open models in a robot simulator.
- Search for a zero-shot VLM skill planner on LIBERO or RoboCasa with an open model in the 2B to 8B range. One search found none.
