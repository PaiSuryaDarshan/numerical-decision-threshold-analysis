import pytest

from ndt.core import analyze_threshold, sweep_thresholds
from ndt.metrics import ConfusionCounts


# Sanity check: core wiring should match the known outcomes for a simple dataset
def test_analyze_threshold_basic_wiring():
    scores = [0.2, 0.7, 0.4, 0.9]
    y_true = [0, 1, 0, 1]
    threshold = 0.5

    result = analyze_threshold(scores, y_true, threshold)

    # Decisions should follow the threshold rule (score >= threshold)
    assert result.y_pred == [0, 1, 0, 1]

    # Confusion counts should reflect perfect classification
    assert result.counts == ConfusionCounts(tp=2, fp=0, tn=2, fn=0)

    # Derived metrics should be consistent with perfect classification
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["precision"] == 1.0
    assert result.metrics["recall"] == 1.0
    assert result.metrics["specificity"] == 1.0
    assert result.metrics["f1"] == 1.0


# Lock in boundary behavior: equality is treated as positive and should affect core outputs
def test_analyze_threshold_equality_boundary_propagates():
    scores = [0.5]
    y_true = [1]
    threshold = 0.5

    result = analyze_threshold(scores, y_true, threshold)
    assert result.y_pred == [1]
    assert result.counts == ConfusionCounts(tp=1, fp=0, tn=0, fn=0)


# Ensure core fails fast on length mismatch (scores drives predictions, so y_true must align)
def test_analyze_threshold_length_mismatch_raises():
    scores = [0.2, 0.7, 0.4]
    y_true = [0, 1]  # mismatch length
    threshold = 0.5

    with pytest.raises(ValueError):
        analyze_threshold(scores, y_true, threshold)


# Sweep should run analyze_threshold repeatedly and preserve threshold ordering
def test_sweep_thresholds_returns_ordered_results():
    scores = [0.2, 0.7, 0.4]
    y_true = [0, 1, 0]
    thresholds = [0.3, 0.5, 0.9]

    results = sweep_thresholds(scores, y_true, thresholds)

    assert [r.threshold for r in results] == thresholds
    assert [r.y_pred for r in results] == [
        [0, 1, 1],  # >=0.3
        [0, 1, 0],  # >=0.5
        [0, 0, 0],  # >=0.9
    ]


# Sweep should return an empty list for an empty thresholds iterable
def test_sweep_thresholds_empty_thresholds():
    scores = [0.2, 0.7]
    y_true = [0, 1]
    thresholds = []

    results = sweep_thresholds(scores, y_true, thresholds)
    assert results == []
