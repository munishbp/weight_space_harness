# How the blog's teacher found the direction

Type: research
Status: resolved
Blocked by: 06

## Question

How did the blog's teacher compute the repetition direction in the 50-pass run, and did the first
prompt name the repetition fault?

Read `exp8_math/runs/p50c/turns/t001/prompt.txt`, `runs/p50c/work_notes/`, and `explanations.md` in
https://github.com/RikiyaT/lm_understand-logs. Compare the method with what our teacher did in
[Find the 0.8B repetition fault again as the teacher](06-teacher-refinds-fault.md).

This ticket waits for that attempt, so that the attempt does not copy the published method.

Findings: `../research/blog-direction-method.md`

## Answer

Resolved on 2026-11-10 by a research subagent, after our own attempt ended. The findings file gives
the turn number for each fact.

- **The first prompt named the fault.** It says: "if the model often fails because it repeats itself
  and never finishes, finding and fixing what drives that repetition counts."
- **The loop direction comes from hand-written text pairs.** The blog's teacher wrote four pairs of
  short thinking texts. Each pair has the same problem and the same opening steps. One text then
  makes a productive check, and the other repeats one doubt sentence word for word. The direction is
  the mean difference of the residual stream over the last 36 tokens, after layer 15 and after
  layer 19, scaled to unit length.
- **Layers and neurons.** The MLPs of layers 15 and 19 added most to that contrast. In each one the
  teacher kept the 64 columns with the highest score: the write along the direction times the
  activation difference.
- **The strength is 6.** The component of those columns along the direction becomes -5 times its old
  value. The edit reverses the direction. It does not only remove it.
- **No thinking-off test.** The blog's teacher did not measure the base model with thinking off.
- **Token limit.** The correct answers of the blog's edited model have a median of 5,310 to 9,172
  tokens. Our 4,096 limit in the held-out comparison understates the published edit.

**Differences from our method.** Our teacher took a direction from natural text (a difference of
means between repeated and new text) and used a strength of 1 and 2 on layers 15 and 19. That trial
had no effect. The likely causes are the weak strength and the noisier direction. Our working edits
used a different handle: the output-head direction of the closing token.

**Open item.** Our direction at strength 6, and a hand-written pair direction through our harness,
are not yet tried. Either one is a two-minute trial on the 0.8B model.
