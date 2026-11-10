"""Uncertainty of a score and of a paired comparison between the original and the edited model."""

import math


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson interval of a proportion, in percent. The default `z` gives a 95% interval."""
    if total <= 0:
        raise ValueError("The interval needs one sample or more.")
    share = successes / total
    centre = share + z * z / (2 * total)
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    scale = 1 + z * z / total
    return 100 * max(0.0, (centre - spread) / scale), 100 * min(1.0, (centre + spread) / scale)


def paired_flips(before: list[bool], after: list[bool]) -> dict:
    """The items that an edit fixes and breaks, with the exact two-sided McNemar p-value.

    `before` and `after` hold the result of the same items in the same order.
    """
    if len(before) != len(after):
        raise ValueError(f"The two lists must cover the same items: {len(before)} against {len(after)}.")
    fixed = sum(1 for b, a in zip(before, after) if a and not b)
    broken = sum(1 for b, a in zip(before, after) if b and not a)
    flips = fixed + broken
    # Under the hypothesis of no effect, each flip is a fair coin toss.
    tail = sum(math.comb(flips, i) for i in range(min(fixed, broken) + 1)) / 2**flips if flips else 0.5
    return {"fixed": fixed, "broken": broken, "p_value": min(1.0, 2 * tail)}
