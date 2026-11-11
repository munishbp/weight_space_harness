class Ban:
    """Set the logit of one token to minus infinity: the forced replay of the other choice at one step."""

    def __init__(self, token):
        self.token = token

    def __call__(self, input_ids, scores):
        scores[:, self.token] = -float("inf")
        return scores


def history_length(conv):
    text = conv[1]["content"][1]["text"]
    done = text.split("Primitives so far:\n")[1].split("\nWrite")[0]
    return 0 if done == "none" else len(done.splitlines())


def gen_forced(conversations):
    torch.manual_seed(0)
    if all(history_length(c) == 1 for c in conversations):
        return tools.generate(conversations, max_new_tokens=64, thinking=False, logits_processor=[Ban(CLOSE)])
    return tools.generate(conversations, max_new_tokens=64, thinking=False)


BUTTON = "button-press-topdown-v3"
forced = mw.run_episodes([(BUTTON, s) for s in range(20)], gen_forced, batch_size=20)
print("forced no-close at step 2:", sum(r["correct"] for r in forced), "/ 20")
print(Counter(" ".join(p["action"][0] for p in r["primitives"]) for r in forced).most_common(3))
print(forced[0]["replies"][:3])
