import pytest
from ndt.thresholds import apply_threshold


# Basic sanity check: standard scores around a typical threshold
def test_apply_threshold_basic_behavior():
    scores = [0.2, 0.7, 0.4]
    threshold = 0.5
    assert apply_threshold(scores, threshold) == [0, 1, 0]


# Explicitly define boundary behavior: equality counts as positive
def test_apply_threshold_equality_is_positive():
    scores = [0.5]
    threshold = 0.5
    assert apply_threshold(scores, threshold) == [1]


# Ensure empty inputs are handled gracefully (no errors, no output)
def test_apply_threshold_empty_input():
    scores = []
    threshold = 0.5
    assert apply_threshold(scores, threshold) == []


# Verify behavior when the threshold itself is negative
def test_apply_threshold_negative_threshold():
    scores = [-1.0, -0.2, 0.0]
    threshold = -0.2
    # rule is score >= threshold
    assert apply_threshold(scores, threshold) == [0, 1, 1]


# Confirm that output labels are not hard-coded and can be customized
def test_apply_threshold_custom_labels():
    scores = [0.2, 0.7]
    threshold = 0.5
    assert apply_threshold(
        scores,
        threshold,
        positive_label=9,
        negative_label=-9,
    ) == [-9, 9]


# Check that the function works with any iterable, not just concrete lists
def test_apply_threshold_accepts_any_iterable():
    scores = (x / 10 for x in [1, 6, 5])  # generator
    threshold = 0.5
    assert apply_threshold(scores, threshold) == [0, 1, 1]
