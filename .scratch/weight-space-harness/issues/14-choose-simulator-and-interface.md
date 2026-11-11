# Choose the simulator and the action interface

Type: grilling
Status: resolved
Blocked by: 12, 13

## Question

Which simulator benchmark does the robotics block use, and through which action interface does
Qwen3.5-4B act in it?

A fact for the decision: the blog's edit "could remove an obstacle that hid an ability, but it could
not add an ability the model lacked". Qwen3.5-4B has no action training. The interface must rest on
an ability that the model has, such as pointing or a choice between named skills.

Decide with the user from the two research answers. The answer to this ticket turns the robotics fog
on the map into tickets.

## Answer

Decided by the teacher on 2026-11-10 under the user's delegation.

**Simulator: Meta-World** (50 tasks for a Sawyer arm, MuJoCo, version 3.1.1 from PyPI).

- It installs from PyPI. Headless EGL rendering works on this machine. The built-in expert policy
  solves `pick-place-v3` in 59 steps, and 500 steps take under a second.
- Each task has a success signal and random start positions, so a practice set and a held-out set
  come from different seeds.
- The standard MT10 set gives a target portion (one or two tasks) and a near-transfer set (the other
  tasks).

**Action interface: text waypoints with the scene state in the prompt.** The model sees one camera
image, the task instruction, and the positions of the gripper, the object, and the goal as text. It
writes the next primitive: a move of the gripper to a position, or an open or close of the gripper.
A proportional controller executes each primitive with the 4-D action of Meta-World. The loop
repeats with a new image until the task succeeds or a step limit ends it. This follows the idea of
EmbodiedBench EB-Manipulation, where the prompt holds object poses and the model writes poses as text.

**Why not the other candidates.**

- VLABench has long composite tasks and a skill library, and it fits the goal better. It needs a
  package that exists only as a git repository and two Google Drive archives. The session blocked
  that install. Its model evaluation script also scores a plan against a fixed dataset and does not
  run the simulator.
- EmbodiedBench has published scores for small Qwen models. It needs CoppeliaSim 4.1.0 or an old
  AI2-THOR with an X server. This machine runs Ubuntu 26.04.
- LIBERO has no skill interface. A text model cannot write its 7-D actions at the control rate.

**Limits.** The text interface is ours. No published score exists for this exact setup. The prompt
holds privileged positions from the simulator, as EB-Manipulation does. Meta-World tasks are short.
A change to VLABench is possible if the user allows the install.
