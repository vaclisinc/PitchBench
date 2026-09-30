"""Order-preserving note alignment shared by D8 and Category F."""

from typing import Any, Callable


def _lcs_length(
    gt: list[Any],
    pred: list[Any],
    match_fn: Callable[[Any, Any], bool],
) -> int:
    """Length of the longest order-preserving one-to-one note alignment."""
    previous = [0] * (len(pred) + 1)
    for gt_item in gt:
        current = [0]
        for j, pred_item in enumerate(pred, start=1):
            if match_fn(gt_item, pred_item):
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def ordered_note_f1(
    gt: list[Any],
    pred: list[Any],
    match_fn: Callable[[Any, Any], bool] = lambda a, b: a == b,
) -> dict[str, float | int]:
    """ROUGE-L-style note precision/recall/F1 over an ordered pitch sequence."""
    matches = _lcs_length(gt, pred, match_fn)
    precision = matches / len(pred) if pred else 0.0
    recall = matches / len(gt) if gt else 0.0
    denom = len(gt) + len(pred)
    f1 = (2 * matches / denom) if denom else 1.0
    return {
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


