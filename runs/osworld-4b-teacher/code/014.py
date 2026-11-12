# A wider base scan for the choice of a portion: every task of the three applications where the sample
# showed some ability. The split is fixed before any look: even positions in `test_all.json` are practice,
# odd positions are held-out. The teacher reads the actions of practice tasks only.
SOCKETS = ("/tmp/wsp-osworld.sock", "/tmp/wsp-osworld2.sock", "/tmp/wsp-osworld3.sock")
APPS = ["thunderbird", "vs_code", "chrome"]
scan_tasks = [t | {"half": "practice" if i % 2 == 0 else "heldout"} for app in APPS for i, t in enumerate(ow.tasks(app))]
HALF = {t["id"]: t["half"] for t in scan_tasks}
json.dump(HALF, open(f"{RUN}/scan_split.json", "w"))
# The 9 tasks of these applications in the first sample keep their results.
if not os.path.exists(f"{RUN}/scan.json"):
    json.dump([r for r in json.load(open(f"{RUN}/sample.json")) if r["id"] in HALF], open(f"{RUN}/scan.json", "w"))
scan = episodes("scan", scan_tasks, f"{RUN}/scan.json", sockets=SOCKETS)
for app in APPS:
    for half in ("practice", "heldout"):
        rows = [r for r in scan if r["domain"] == app and HALF[r["id"]] == half and "error" not in r]
        print(f"{app:12s} {half:9s} solved {sum(r['correct'] for r in rows):2d} of {len(rows):2d}")
stamp("scan done")
