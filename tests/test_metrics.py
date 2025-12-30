import pytest
from ndt.metrics import ConfusionCounts, binary_metrics, confusion_counts, metrics_from_counts


# Sanity check: verify basic TP/FP/TN/FN counting on a small mixed example
def test_confusion_counts_basic():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 1, 0, 0]
    c = confusion_counts(y_true, y_pred)
    assert c == ConfusionCounts(tp=1, fp=1, tn=1, fn=1)


# Validate that standard metric formulas are computed correctly from known counts
def test_metrics_from_counts_values():
    c = ConfusionCounts(tp=1, fp=1, tn=1, fn=1)
    m = metrics_from_counts(c)

    # total = 4
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5   # tp / (tp + fp)
    assert m["recall"] == 0.5      # tp / (tp + fn)
    assert m["specificity"] == 0.5 # tn / (tn + fp)
    assert m["f1"] == 0.5


# Protect against divide-by-zero cases when no positive predictions are made
def test_binary_metrics_all_negative_predictions():
    y_true = [1, 0, 1, 0]
    y_pred = [0, 0, 0, 0]
    m = binary_metrics(y_true, y_pred)

    # tp=0, fp=0, tn=2, fn=2
    assert m["counts"] == {"tp": 0, "fp": 0, "tn": 2, "fn": 2}
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["specificity"] == 1.0


# Ensure mismatched input lengths are rejected early and explicitly
def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_counts([1, 0], [1])


# Enforce strict label validation when strict_labels=True
def test_strict_labels_rejects_unknown_label():
    with pytest.raises(ValueError):
        confusion_counts([2], [1], strict_labels=True)


# Define behavior when strict label checking is disabled
def test_non_strict_labels_treats_non_positive_as_negative():
    # If strict_labels=False, anything not equal to positive_label counts as negative
    c = confusion_counts([2, 1], [1, 2], strict_labels=False)
    # t=[neg,pos], p=[pos,neg] -> fp=1, fn=1
    assert c == ConfusionCounts(tp=0, fp=1, tn=0, fn=1)


# Verify that non-standard label values are supported when explicitly specified
def test_custom_labels():
    y_true = [9, -9, 9, -9]
    y_pred = [9, 9, -9, -9]
    c = confusion_counts(y_true, y_pred, positive_label=9, negative_label=-9)
    assert c == ConfusionCounts(tp=1, fp=1, tn=1, fn=1)
