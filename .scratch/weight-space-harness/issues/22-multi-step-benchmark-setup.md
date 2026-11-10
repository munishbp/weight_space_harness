# Check which multi-step computer use benchmark runs here

Type: task
Status: open
Blocked by: 10

## Question

Which multi-step computer use benchmark, OSWorld-Verified or AndroidWorld, installs and runs on this
machine with Qwen3.5-4B as the agent, and how long does one task take?

The task: check the virtual machine or emulator support of the machine (KVM, Docker, disk space).
Install the cheaper candidate. Run a small set of tasks with the base model. Record the setup steps,
the time per task, and the base success rate. The model card reports 35.6 on OSWorld-Verified and
58.6 on AndroidWorld for Qwen3.5-4B.

This ticket waits for [Edit the 4B model for the ScreenSpot Pro slice](10-teacher-edit-4b-slice.md),
because the order of the map puts the fast benchmark first.
