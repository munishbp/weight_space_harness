# Choose a student outside the Qwen family

Type: grilling
Status: open
Blocked by: 06

## Question

Both planned students are Qwen3.5 models. Does the map need a run on a student from another family
(for example GLM) to show that the harness is not specific to Qwen, and which model?

Decide with the user after the calibration, when it is clear which parts of the tools know the
Qwen3.5 architecture. A fact for the decision: three of every four Qwen3.5 layers are Gated DeltaNet
layers and not standard attention, so module names and edit recipes differ between families.
