# Run one portion of OSWorld on the 4B model and try the recipe on it

Type: task
Status: open

## Question

MiniWoB++ tasks have 2 to 10 steps. OSWorld is the realistic computer use benchmark that the goal
names: real applications in an Ubuntu virtual machine. What does the base Qwen3.5-4B score on one
portion of it through screenshots, which kind of fault do its failures show, and does the habit
recipe give a held-out gain there?

The user allowed the downloads on 2026-11-11: the `desktop-env` package from PyPI, the Ubuntu VM
image from the Hugging Face Hub, the Docker image `happysixd/osworld-docker`, and a clone of the
OSWorld repository for the task files. The user also allowed the VLABench install.

The task, in order:

1. Bring up the environment through `wsp/bench/osworld_server.py` and `wsp/bench/osworld_screen.py`.
   Record the time per task.
2. Run the base model on a sample of every application. Choose one application as the portion: the
   base score must leave room, and the failures must be readable.
3. Split the tasks of that application into practice and held-out before the search.
4. Sort the failures by cause. Run a forced replay where a wrong action name is the cause.
5. If a fault of the habit kind or the blocked-token kind exists, edit and run the verdict. If the
   failures are wrong positions, record a negative result.

The model card reports 35.6 on OSWorld-Verified for Qwen3.5-4B, with an agent setup that it does
not state.
