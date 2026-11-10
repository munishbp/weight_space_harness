# Find the 0.8B repetition fault again as the teacher

Type: task
Status: open
Blocked by: 04, 05

## Question

Can Claude, as the teacher, find and remove the repetition fault of Qwen3.5-0.8B with the harness
tools and no published edit file, and how much wall-clock time and how many forward passes does it take?

Rules: no gradient, no optimizer, closed-form edits only. The teacher must not load the published
`edit.npz` or read the run notes of the blog. The teacher already knows the summary in
[What the blog method does and what it released](01-blog-method.md), so this test is not blind. The
answer states that limit.

Record the ledger totals, the final edit manifest (edit size and deviation), the token-limit share before and after, and each
tool that the teacher needed and did not have.
