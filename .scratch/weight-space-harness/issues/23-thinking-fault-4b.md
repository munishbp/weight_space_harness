# Test the closing-token recipe on the 4B model

Type: task
Status: open

## Question

The base Qwen3.5-4B reaches the 4,096 token limit in 33% of GSM8K answers with thinking on (battery
run of 2026-11-10: 75.0% correct). This is the fault that the teacher edited in the 0.8B model. Does
the same recipe (find the parts that suppress the closing token of the thinking, then remove that
suppression) work on the 4B model, and how long does it take?

This ticket answers the scaling question on a task where a fault exists. It is not a computer use
task, so it does not replace the computer use verdict.

The task: follow the playbook on GSM8K practice items with thinking on. Measure the thinking-off
prompt first, because it beat every edit on the 0.8B model. Report the six measures of the map.
