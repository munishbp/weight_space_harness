# The base model on the small test set of OSWorld without the multi-application tasks: 22 tasks in 9 applications.
# The purpose is the choice of the portion: the base rate and the kind of failure per application.
tools.log("teacher_start", note="OSWorld through screenshots, base sample")
sample_tasks = []
for domain, ids in small.items():
    if domain == "multi_apps": continue
    by = {t["id"]: t for t in ow.tasks(domain)}; sample_tasks += [by[i] for i in ids]
print(len(sample_tasks), "tasks", Counter(t["domain"] for t in sample_tasks))
sample = episodes("sample", sample_tasks, f"{RUN}/sample.json")
for domain in sorted({r["domain"] for r in sample}):
    rows = [r for r in sample if r["domain"] == domain]
    print(f"{domain:20s} solved {sum(r['correct'] for r in rows)} of {len(rows)} | invalid actions {sum(r['invalid'] for r in rows)} | mean seconds {sum(r['seconds'] for r in rows) // len(rows)}")
stamp("sample done")
