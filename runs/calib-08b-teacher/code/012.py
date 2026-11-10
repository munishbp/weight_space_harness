def categories(gen):
    c = Counter(); ex = {}
    for p, r in zip(practice, gen):
        closed = THINK_END in r["ids"]
        k = ("finished " + ("correct" if r["correct"] else "wrong")) if closed and not r["hit_limit"] else ("closed, then hit limit" if closed else "never closed")
        c[k] += 1; ex.setdefault(k, []).append((p, r))
    return c, ex
for label, gen in (("alpha 2, three components", g8), ("alpha 2, four components", g10)):
    c, ex = categories(gen); print(label, dict(c))
    think_len = [r["ids"].index(THINK_END) for r in gen if THINK_END in r["ids"]]
    print("  thinking length when closed: median", sorted(think_len)[len(think_len) // 2], "| min", min(think_len))
c, ex = categories(g8)
p, r = ex["closed, then hit limit"][0]; i = r["ids"].index(THINK_END)
print("=== closed, then hit limit: text around the close, then the tail"); print(tok.decode(r["ids"][max(0, i - 60): i + 80])); print("... TAIL:", tok.decode(r["ids"][-150:]))
p, r = ex["closed, then hit limit"][1]; print("... second TAIL:", tok.decode(r["ids"][-120:]))
print("=== finished wrong (two examples): end of thinking and the answer")
for p, r in ex["finished wrong"][:2]:
    i = r["ids"].index(THINK_END); print("gold:", p["answer"], "| given:", r["answer_given"]); print(tok.decode(r["ids"][max(0, i - 100):])[-420:]); print("-")
