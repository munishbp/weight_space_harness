# Choose the ScreenSpot Pro slice and the split

Type: grilling
Status: resolved
Blocked by: 08

## Question

Which portion of ScreenSpot Pro is the target, and how do its samples divide into practice items for
the teacher and held-out items for the verdict?

Decide with the user, from the baseline table:

- The portion: one group, one application, or one `ui_type` (icon targets are 604 samples).
- The split rule, so that the teacher never sees a held-out item.
- Whether the rest of ScreenSpot Pro is reported as a near-target check, next to the battery from
  [Choose the generalization battery](16-retention-battery.md).

## Answer

Decided by the teacher on 2026-11-10 under the user's delegation. File: `runs/ss-pro-4b-teacher/split.json`.

- **The portion: icon targets** (604 items, base 35.6%). It has the most headroom and enough items
  for a measurable result. One application has 31 to 93 items, which is too few.
- **The split.** Inside each application, the icon items in id order alternate between practice
  (311 items, base 33.1%) and held-out (293 items, base 38.2%). Half of the practice items (156, base
  35.9%) form the quick trial set. The teacher does not look at a held-out item during the search.
- **Near transfer.** The 977 text targets of ScreenSpot Pro, next to the generalization battery.
