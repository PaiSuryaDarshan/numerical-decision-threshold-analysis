from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

# A simple data container to hold the four possible outcomes of a binary decision system.
# frozen=True means once created, these values cannot be changed.
# This prevents accidental modification of results. (STRICTLY immutable)
@dataclass(frozen=True)
class ConfusionCounts:
    tp: int  # true positives: predicted positive, actually positive
    fp: int  # false positives: predicted positive, actually negative
    tn: int  # true negatives: predicted negative, actually negative
    fn: int  # false negatives: predicted negative, actually positive


# Internal helper function.
# Converts any iterable into a list of ints and validates the contents.
# This prevents silent bugs later when metrics assume clean integer labels.
def _to_list_int(values: Iterable[int], name: str) -> List[int]:
    # Force evaluation (works for lists, tuples, generators, etc.) 
    try:
        out = list(values)
    except TypeError as e:
        raise TypeError(f"{name} must be an iterable of ints.") from e

    # Enforce that every element is an integer.
    # Metrics rely on exact equality checks, so this must be strict.
    for i, v in enumerate(out):
        if not isinstance(v, int):
            raise TypeError(
                f"{name}[{i}] must be int, got {type(v).__name__}."
            )
    return out


# Compute the confusion matrix counts for binary classification.
# This function does NOT compute metrics — only raw counts.
def confusion_counts(
    y_true: Iterable[int],   # ground-truth labels
    y_pred: Iterable[int],   # predicted labels
    *,
    positive_label: int = 1, # value treated as "positive"
    negative_label: int = 0, # value treated as "negative"
    strict_labels: bool = True,  # whether to reject unknown labels
) -> ConfusionCounts:
    """
    Compute TP / FP / TN / FN for binary labels.

    Rules:
    - A label is positive if label == positive_label
    - A label is negative if label == negative_label

    If strict_labels=True, any other value raises an error.
    """

    # Convert inputs to validated lists of ints
    yt = _to_list_int(y_true, "y_true")
    yp = _to_list_int(y_pred, "y_pred")

    # True and predicted labels must align one-to-one
    if len(yt) != len(yp):
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {len(yt)} and {len(yp)}."
        )

    # Initialize counters
    tp = fp = tn = fn = 0

    # Compare each true/predicted pair
    for i, (t, p) in enumerate(zip(yt, yp)):

        # Optional strict validation to catch unexpected labels early
        if strict_labels:
            if t not in (positive_label, negative_label):
                raise ValueError(
                    f"y_true[{i}]={t} is not in "
                    f"{{{negative_label}, {positive_label}}}."
                )
            if p not in (positive_label, negative_label):
                raise ValueError(
                    f"y_pred[{i}]={p} is not in "
                    f"{{{negative_label}, {positive_label}}}."
                )

        # Convert labels to boolean flags for simpler logic
        t_pos = (t == positive_label)
        p_pos = (p == positive_label)

        # Increment the appropriate counter
        if t_pos and p_pos:
            tp += 1
        elif (not t_pos) and p_pos:
            fp += 1
        elif (not t_pos) and (not p_pos):
            tn += 1
        else:
            fn += 1

    # Return immutable result container
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


# Helper function for safe division.
# Avoids ZeroDivisionError and keeps metrics defined.
def safe_div(n: float, d: float) -> float:
    """Return n / d, or 0.0 if d == 0."""
    return 0.0 if d == 0 else n / d


# Compute standard performance metrics from confusion counts.
# This function assumes counts are already correct.
def metrics_from_counts(c: ConfusionCounts) -> dict:
    """
    Compute common binary metrics from confusion counts.

    Returns:
    - accuracy
    - precision
    - recall (sensitivity)
    - specificity
    - f1 score
    - false positive rate (fpr)
    - false negative rate (fnr)
    """

    # Unpack counts for readability
    tp, fp, tn, fn = c.tp, c.fp, c.tn, c.fn
    total = tp + fp + tn + fn

    # Core metrics
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # Error rates
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    return {
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
    }


# Convenience wrapper.
# Takes labels → computes counts → derives metrics in one call.
def binary_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    positive_label: int = 1,
    negative_label: int = 0,
    strict_labels: bool = True,
) -> dict:
    """Compute binary classification metrics in one step."""
    c = confusion_counts(
        y_true,
        y_pred,
        positive_label=positive_label,
        negative_label=negative_label,
        strict_labels=strict_labels,
    )
    return metrics_from_counts(c)

