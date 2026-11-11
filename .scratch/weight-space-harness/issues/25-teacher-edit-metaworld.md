# Edit the 4B model for the Meta-World target tasks

Type: task
Status: claimed
Blocked by: 24

## Question

Does a closed-form edit by the teacher raise the success of Qwen3.5-4B on the target tasks of
Meta-World (button-press-topdown, push, window-open), and what does it cost in the other seven tasks
and in the battery?

The target faults come from [Install one simulator benchmark and run the base model](24-simulator-install.md):
the model closes the gripper before a press, and its first move goes to the goal in place of the object.

The practice episodes use seeds 0 to 19. The held-out episodes use seeds 100 to 119. The other seven
tasks are the near-transfer set. Record the six measures of the map. Try the prompt first, as the
playbook says.
