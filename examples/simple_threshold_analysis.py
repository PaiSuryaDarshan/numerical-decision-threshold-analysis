"""
Minimal example demonstrating threshold-based decision analysis.

This script assumes that continuous scores and ground-truth labels
already exist, and evaluates how performance changes with threshold.
"""

from ndt.core import analyze_threshold, sweep_thresholds


def main():
    # Example model scores (e.g. probabilities or confidence scores)
    scores = [0.12, 0.43, 0.55, 0.78, 0.91]

    # Ground-truth binary labels
    y_true = [0, 0, 1, 1, 1]

    # Analyze a single threshold
    result = analyze_threshold(scores, y_true, threshold=0.5)

    print("Single-threshold analysis")
    print("-------------------------")
    print(f"Threshold: {result.threshold}")
    print(f"Predictions: {result.y_pred}")
    print(f"Confusion counts: {result.counts}")
    print(f"Metrics: {result.metrics}")
    print()

    # Sweep across multiple thresholds
    thresholds = [0.3, 0.5, 0.7]
    results = sweep_thresholds(scores, y_true, thresholds)

    print("Threshold sweep")
    print("---------------")
    for r in results:
        print(
            f"threshold={r.threshold:.2f} | "
            f"accuracy={r.metrics['accuracy']:.2f} | "
            f"precision={r.metrics['precision']:.2f} | "
            f"recall={r.metrics['recall']:.2f}"
        )


if __name__ == "__main__":
    main()
