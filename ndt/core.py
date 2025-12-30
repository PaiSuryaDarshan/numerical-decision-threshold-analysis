from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ndt.metrics import ConfusionCounts, binary_metrics, confusion_counts
from ndt.thresholds import apply_threshold


# Result object representing the outcome of applying a single threshold.
# Frozen to ensure analysis results are immutable once computed.
@dataclass(frozen=True)
class ThresholdAnalysisResult:
    """
    Immutable container for a single threshold analysis run.

    Design intent:
    - groups all outputs of a threshold evaluation in one place
    - prevents accidental mutation of computed results
    - provides a stable, testable return type for downstream use
    """
    threshold: float           # decision threshold used
    y_pred: List[int]           # predicted binary decisions
    counts: ConfusionCounts     # raw confusion matrix counts
    metrics: dict               # derived performance metrics


def analyze_threshold(
    scores: Iterable[float],
    y_true: Iterable[int],
    threshold: float,
    *,
    positive_label: int = 1,
    negative_label: int = 0,
    strict_labels: bool = True,
) -> ThresholdAnalysisResult:
    """
    Apply a decision threshold to scores and evaluate against ground truth.

    Responsibility boundaries:
    - thresholds.py: converts scores → decisions
    - metrics.py: evaluates decisions → performance
    - core.py: coordinates the two without adding new logic
    """

    # Step 1: convert continuous scores into binary decisions
    y_pred = apply_threshold(
        scores,
        threshold,
        positive_label=positive_label,
        negative_label=negative_label,
    )

    # Step 2: compute raw confusion matrix counts
    # This is kept explicit because counts are often useful on their own
    counts = confusion_counts(
        y_true,
        y_pred,
        positive_label=positive_label,
        negative_label=negative_label,
        strict_labels=strict_labels,
    )

    # Step 3: compute derived metrics (accuracy, precision, recall, etc.)
    metrics = binary_metrics(
        y_true,
        y_pred,
        positive_label=positive_label,
        negative_label=negative_label,
        strict_labels=strict_labels,
    )

    return ThresholdAnalysisResult(
        threshold=threshold,
        y_pred=y_pred,
        counts=counts,
        metrics=metrics,
    )

def sweep_thresholds(
    scores: Iterable[float],
    y_true: Iterable[int],
    thresholds: Iterable[float],
    *,
    positive_label: int = 1,
    negative_label: int = 0,
    strict_labels: bool = True,
) -> List[ThresholdAnalysisResult]:
    """
    Evaluate model performance across multiple thresholds.

    Intended use:
    - comparative analysis of threshold sensitivity
    - ROC / PR-style explorations (without plotting here)
    """

    results: List[ThresholdAnalysisResult] = []

    # Iterate in the order provided to preserve caller intent
    for th in thresholds:
        results.append(
            analyze_threshold(
                scores,
                y_true,
                th,
                positive_label=positive_label,
                negative_label=negative_label,
                strict_labels=strict_labels,
            )
        )

    return results
