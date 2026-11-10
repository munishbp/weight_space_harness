"""Run a benchmark through the kernel tools and summarize the scores.

A benchmark is a plain module with two functions: `conversation(item)` builds the chat for one
item, and `score(item, text)` returns a dict that holds `correct`. The runner knows nothing else
about a benchmark, so a new target needs one new module and no change here.
"""

from collections import defaultdict
from collections.abc import Callable

from wsp import tools
from wsp.stats import wilson_interval


def run(
    items: list[dict],
    conversation: Callable[[dict], list[dict]],
    score: Callable[[dict, str], dict],
    batch_size: int = 8,
    **generate_options,
) -> list[dict]:
    """Generate one reply per item and score it. Each batch is one forward pass in the ledger."""
    results = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        replies = tools.generate([conversation(item) for item in chunk], **generate_options)
        for item, reply in zip(chunk, replies):
            results.append({"id": item["id"], **reply, **score(item, reply["text"])})
    return results


def summarize(results: list[dict], items: list[dict] | None = None, by: str | None = None) -> dict:
    """The score with a 95% interval, for all results or for each value of the item field `by`."""
    if by is None:
        return {"all": _summary(results)}
    field = {item["id"]: item[by] for item in items}
    groups = defaultdict(list)
    for result in results:
        groups[field[result["id"]]].append(result)
    return {key: _summary(group) for key, group in sorted(groups.items())}


def _summary(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(result["correct"] for result in results)
    low, high = wilson_interval(correct, total)
    return {
        "n": total,
        "correct_pct": round(100 * correct / total, 1),
        "interval_95": [round(low, 1), round(high, 1)],
        "hit_limit_pct": round(100 * sum(result["hit_limit"] for result in results) / total, 1),
        "mean_new_tokens": round(sum(result["new_tokens"] for result in results) / total, 1),
    }
