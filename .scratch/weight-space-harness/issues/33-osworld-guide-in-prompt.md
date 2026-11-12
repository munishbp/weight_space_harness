# Test if the knowledge of the teacher, as a guide in the prompt, lifts possible OSWorld tasks

Type: task
Status: resolved

## Question

The failures on possible OSWorld tasks are procedures that the student does not know
([ticket 30](30-osworld-portion.md)). The teacher knows them. Two results suggest a route: a
sentence with the procedure fixed the skill names in VLABench ([ticket 31](31-vlabench-procedure-edit.md)),
and a sentence showed the hidden ability for `fail` ([ticket 32](32-osworld-fail-edit.md)). Does a
guide for one application in the system prompt lift the practice tasks of that application? A gain
there is the condition for an edit that writes the guide into the weights.

## Answer

**No. The student follows an exact key recipe, and nothing else. No edit ran.**

The portion is VS Code: 12 practice tasks, of which 9 are possible (base 3 of 9). The teacher wrote
each guide from its own knowledge of VS Code and from the practice tasks. It saw no held-out task.

| Prompt | Possible practice tasks solved | Fixed | Broken |
|---|---|---|---|
| Base | 3 of 9 | | |
| Guide 1: procedures in words (settings page, replace, shortcuts, folders) | 2 of 9 | 1 | 2 |
| Guide 2: four exact key recipes, each with its kind of task | 3 of 9 | 1 | 1 |

- Both guides fix the same task: replace all in a file. Its recipe is five fixed key presses, and
  the student writes them exactly.
- A step that needs a look at the screen fails with the guide as without it: the student types
  "on" in place of a click on a checkbox, and it writes `debug.focusEditorOnBreakpoint: false` in
  place of a JSON object with the correct setting name.
- The guides break tasks that the base solved. The student applies a line to a task of another
  kind, and it presses `ctrl+s` where the guide names it for another case.
- The three tasks that are not possible stay at 0 with both guides.

### What it means

- For possible desktop tasks the missing part is not only knowledge of the procedure. The student
  also lacks the judgment at the steps between the key presses. A guide in the prompt does not give
  it, so an edit that writes the guide into the weights has no ceiling to reach.
- This agrees with every earlier result: a prompt sentence or an edit changes a default (one token,
  one fixed list of names, one fixed list of keys). It does not add an ability.
- OSWorld also has a limit of its own for this method. Its 368 tasks have no training split, and a
  group of tasks that share a procedure has 1 to 5 members. A held-out gain needs a behavior that
  crosses the groups, as the refusal of ticket 32 does.

### Harness defect found

`osworld_screen.parse_action` cut an action at the first closing brace, so a `type` action with a
brace in its text was invalid. The parser now decodes one full JSON object. The defect changed no
step of the base, scan, or verdict runs (0 of 2,748 recorded steps). It changed 10 steps in 2 tasks
of the run with guide 2, which asks for typed JSON.
