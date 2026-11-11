for b, a in zip(scan_other, base_again):
    if b["correct"] != a["correct"]: print(f"{b['id']:26s} {'fixed ' if a['correct'] else 'BROKEN'} | first: {acts(b):22s} | repeat: {acts(a)}")
print("edit 3 against the base repeat:", paired_flips([r["correct"] for r in base_again], [r["correct"] for r in edit_other3]))
print("edit 2 against the base repeat:", paired_flips([r["correct"] for r in base_again], [r["correct"] for r in edit_other2]))
print("edit 1 against the base repeat:", paired_flips([r["correct"] for r in base_again], [r["correct"] for r in edit_other]))
