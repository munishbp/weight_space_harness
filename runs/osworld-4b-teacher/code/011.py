# Forced replay of a second fault: the student writes `type` directly after `type`, with no key or click
# between two fields. Here the action name `type` is not possible in the step after a `type`. The student picks the rest.
SOCKETS = ("/tmp/wsp-osworld.sock", "/tmp/wsp-osworld2.sock", "/tmp/wsp-osworld3.sock")
TYPE = tok.encode("type", add_special_tokens=False)[0]


class Ban:
    """Set the logit of one token to minus infinity."""

    def __init__(self, token):
        self.token = token

    def __call__(self, input_ids, scores):
        scores[:, self.token] = -float("inf")
        return scores


def gen_no_type_twice(conversations):
    replies = []
    for conversation in conversations:
        last = last_action(conversation)
        after_type = last is not None and last.get("action") == "type"
        options = {"logits_processor": [Ban(TYPE)]} if after_type else {}
        replies.append(tools.generate([conversation], max_new_tokens=96, thinking=False, **options)[0])
    return replies


def has_type_twice(result):
    names = [a.get("action") for a in result["actions"]]
    return any(a == b == "type" for a, b in zip(names, names[1:]))


targets = [r["id"] for r in sample if not r["correct"] and has_type_twice(r)]
no_tt = episodes("no type twice", [by_id[i] for i in targets], f"{RUN}/no_type_twice.json", generate=gen_no_type_twice, sockets=SOCKETS)
for r in no_tt: print(r["domain"], r["id"][:8], r.get("score"), [json.dumps(a)[:60] for a in r["actions"]])
stamp("forced replay with no type after type done")
