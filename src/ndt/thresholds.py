from __future__ import annotations

from typing import Iterable, List


def apply_threshold(
    scores: Iterable[float],
    threshold: float,
    *,
    positive_label: int = 1,
    negative_label: int = 0,
) -> List[int]:
    """
    Convert numerical scores into binary decisions using a fixed threshold.

    A score is classified as positive if score >= threshold,
    and negative otherwise.

    Parameters
    ----------
    scores : Iterable[float]
        Numerical scores to be thresholded.
    threshold : float
        Decision cutoff value.
    positive_label : int, optional
        Label assigned to scores >= threshold.
    negative_label : int, optional
        Label assigned to scores < threshold.

    Returns
    -------
    List[int]
        Binary decisions corresponding to the input scores.

    Notes
    -----
    A common failure mode in threshold-based systems is ambiguity around
    the equality case (score == threshold). This implementation explicitly
    treats equality as a positive decision.
    """
    decisions: List[int] = []

    for score in scores:
        if score >= threshold:
            decisions.append(positive_label)
        else:
            decisions.append(negative_label)

    return decisions
