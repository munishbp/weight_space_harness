# Set the pass criteria and the time boxes

Type: grilling
Status: resolved

## Question

What result makes the verdict "feasible", and how long does each block run before it ends with a
negative result?

Decide these points with the user:

- The gain on held-out items of the target portion that counts as a real improvement.
- The time box for the LoRA run, if it differs from the teacher's wall-clock time.
- The time box for the teacher on the 0.8B model, on the ScreenSpot Pro slice, and on the simulator.
- How the LoRA comparison is judged: equal wall-clock time, and which side wins a tie.

The generalization measure has its own ticket: [Choose the generalization battery](16-retention-battery.md).

A fact for the decision: a ScreenSpot Pro application has 31 to 93 samples and a group has 196 to
341. A held-out half of one group gives a 95% interval of about 8 points each side. A small gain on a
small slice is not measurable.

## Answer

Resolved by the user on 2026-11-10.

- **No pass and no fail.** The harness is the deliverable. A 15 minute edit is the ideal. A result
  that misses it is still information to share. Every verdict reports its numbers with no threshold.
- **Time boxes.** The user left them to the teacher. They exist only to stop an open-ended search:
  2 hours of teacher work to find the 0.8B fault again, 4 hours for the ScreenSpot Pro edit, 4 hours
  for the simulator edit after its setup, and 3 hours for the 27B edit. A block that reaches its time
  box records what it found and the map moves on.
- **LoRA comparison.** Equal wall-clock time on the same practice items. The verdict reports both
  sides and names no winner by rule.
