habit.fit(MLP_OUT, [(keys, scale, g)], keep, protect, margin=8.0)
button8 = mw.run_episodes([(BUTTON, s) for s in range(20)], gen, batch_size=20)
print("margin 8 button:", sum(r["correct"] for r in button8), "/ 20")
tools.reset_edit()
