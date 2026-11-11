# Check which multi-step computer use benchmark runs here

Type: task
Status: resolved
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

## Answer

Resolved on 2026-11-10 by a research agent (machine checks and primary sources, no install). Findings:
[Multi-step benchmark setup](../research/multi-step-benchmark-setup.md). The decision below is by
the teacher under the user's delegation.

**Facts.**

- This machine can run a virtual machine: KVM and Docker 29.1.3 work with no sudo, and 653 GB of disk
  is free.
- OSWorld-Verified has no hard blocker. It needs a git clone, a 12.3 GB Ubuntu image, a Docker image
  that runs QEMU, and a new agent class that calls the local model. Tasks take 15 to 100 steps. The
  authors state "10+ hours" for a full run on one machine. No source gives the time per task for a
  local 4B model.
- AndroidWorld has 116 tasks at about 3.9 minutes per task with the authors' agent. The emulator
  package, the system image, and the virtual device are absent here. Its Docker route is experimental.
- MiniWoB++ (`miniwob` on PyPI, MIT) has about 125 short browser tasks and runs in the same process
  as the model. It needs a ChromeDriver that matches Chrome. Its default episode limit is 10 seconds
  of browser time. A 2026 paper reports 61.1 for Qwen3.5-4B on MiniWoB through BrowserGym.

**Decision: MiniWoB++ first, then a portion of OSWorld-Verified.**

- The teacher needs many passes, so one pass must take minutes. MiniWoB++ gives that. It is a
  multi-step computer use benchmark, but its tasks are synthetic and short.
- OSWorld-Verified is the realistic target that the goal names ("open world computer use
  benchmark"). One portion of it (the tasks of one application) follows when MiniWoB++ shows which
  kind of fault an edit can fix in a multi-step loop.
- Both installs reach outside PyPI (a ChromeDriver from Google, or a git clone and VM images). The
  session blocked an install from third-party git repositories before. If it blocks these, the user
  must allow them.

The open point of the research (the 10-second limit of MiniWoB++ against the speed of a local model)
needs a check at install time.
