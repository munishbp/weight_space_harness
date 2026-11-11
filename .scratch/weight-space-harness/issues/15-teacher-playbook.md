# Draft the teacher playbook

Type: prototype
Status: resolved
Blocked by: 06

## Question

What must a fresh session of any frontier model read so that it can run the harness as the teacher on
a new student and a new target?

Make a rough playbook from the calibration run and react to it with the user. Candidate contents: how
to start the kernel, the tools and their limits, the edit rules, the order of work (baseline, failure
sort, hypothesis, read, edit, check), the ledger, and the stop conditions.

The playbook is plain text and names no vendor, because the teacher is interchangeable. It must not
rely on a Claude Code feature.

## Answer

`PLAYBOOK.md` in the repository root. It names no vendor and needs only a shell. Sections: start,
rules, tools, order of work, which edit fits which fault, lessons from the runs, when to stop.

The playbook came from five teacher runs (0.8B thinking fault, ScreenSpot Pro, two Meta-World
faults, MiniWoB++ email). The recipe that worked three times is now code (`wsp/habit.py`), and the
playbook states its rules with the failed attempt behind each rule.

The user did not review the draft. The test of the playbook by a teacher that did not write it is
[Show that the teacher is interchangeable](18-teacher-interchange.md).
