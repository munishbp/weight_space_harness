for b, a in zip(scan_other, edit_other):
    if b["correct"] != a["correct"]:
        print(f"{b['id']:26s} {'fixed ' if a['correct'] else 'BROKEN'} | base: {acts(b):22s} | edit: {acts(a)}")
