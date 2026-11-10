# Run the LoRA baseline on the same slice

Type: task
Status: open
Blocked by: 09, 10

## Question

What held-out score and what retention does a LoRA fine-tune reach on the same slice, with the same
practice items and the same wall-clock time as the teacher edit?

This ticket waits for the teacher edit, because the wall-clock budget comes from that run. The
ticket reports the same generalization battery and deviation measures for the LoRA model, and it
states the LoRA settings (rank, target modules, learning rate, steps) and why.
