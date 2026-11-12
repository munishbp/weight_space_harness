# The base 27B student on the practice half of Thunderbird and VS Code (20 tasks), then of Chrome (23 tasks).
# Three virtual machines share the model. The model is the slow part: about 16 seconds per step.
if not os.path.exists(f"{RUN}/scan_practice.json"):
    json.dump(json.load(open(f"{RUN}/smoke.json")), open(f"{RUN}/scan_practice.json", "w"))
tools.log("teacher_start", note="OSWorld through screenshots, 27B student, practice scan")
first = episodes("scan", practice_of("thunderbird", "vs_code"), f"{RUN}/scan_practice.json", sockets=SOCKETS)
summary("27B base, thunderbird and vs_code practice", first); stamp("first part of the scan done")
both = episodes("scan", practice_of("thunderbird", "vs_code", "chrome"), f"{RUN}/scan_practice.json", sockets=SOCKETS)
summary("27B base, three applications, practice", both); stamp("scan done")
