"""
Example showing programmatic comparison of metrics across thresholds.

This script demonstrates how threshold analysis results can be
consumed directly (e.g., for reporting or downstream analysis),
without relying on printed output.
"""

from ndt.core import sweep_thresholds

def main():
    scores = [0.12, 0.43, 0.55, 0.78, 0.91]
    y_true = [0, 0, 1, 1, 1]
    thresholds = [0.2, 0.4, 0.6, 0.8]

    results = sweep_thresholds(scores, y_true, thresholds)

    # Extract metrics of interest for comparison
    summary = [
        {
            "threshold": r.threshold,
            "accuracy": r.metrics["accuracy"],
            "precision": r.metrics["precision"],
            "recall": r.metrics["recall"],
        }
        for r in results
    ]

    # Example: identify the threshold with highest accuracy
    best = max(summary, key=lambda x: x["accuracy"])

    print("Best threshold by accuracy:")
    print(best)


if __name__ == "__main__":
    main()
