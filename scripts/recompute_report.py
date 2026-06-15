"""Recompute the final metrics report at the EER-calibrated operating point.

Loads the existing checkpoint (no retraining), re-evaluates the held-out
test set at the threshold where FAR == FRR (the EER operating point) instead
of the arbitrary fixed 0.5 cutoff, and rewrites `models/metrics_report.json`
and `models/confusion_matrix.png` accordingly. `models/roc_curve.png` is
unchanged (the EER point it marks is threshold-independent).

Run from the project root:

    python scripts/recompute_report.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import torch

from src.config import (
    PROCESSED_DATA_DIR,
    CHECKPOINT_PATH,
    METRICS_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    TARGET_ACCURACY,
    TARGET_EER,
    TARGET_F1,
    TARGET_PER_CLASS_ACCURACY,
)
from src.models.model import CNNLSTMDetector
from src.models.metrics import evaluate, meets_targets, plot_confusion_matrix


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    arch = checkpoint["architecture"]

    model = CNNLSTMDetector(
        dropout=arch["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_val = torch.from_numpy(np.load(PROCESSED_DATA_DIR / "X_val.npy"))
    y_val = np.load(PROCESSED_DATA_DIR / "y_val.npy")
    X_test = torch.from_numpy(np.load(PROCESSED_DATA_DIR / "X_test.npy"))
    y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")

    with torch.no_grad():
        val_scores = torch.sigmoid(model(X_val)).numpy()
        test_scores = torch.sigmoid(model(X_test)).numpy()

    # Reference report at the default 0.5 cutoff (also gives us eer / eer_threshold,
    # which are threshold-independent properties of the ROC curve).
    test_report_05 = evaluate(y_test, test_scores, threshold=0.5)
    val_report_05 = evaluate(y_val, val_scores, threshold=0.5)

    op_threshold = test_report_05.eer_threshold
    test_report_op = evaluate(y_test, test_scores, threshold=op_threshold)
    val_report_op = evaluate(y_val, val_scores, threshold=op_threshold)

    print(f"EER-calibrated operating threshold: {op_threshold:.4f}")
    print(f"\n=== Test @ 0.5 ===")
    print(f"  accuracy={test_report_05.accuracy:.4f} f1={test_report_05.f1:.4f} "
          f"eer={test_report_05.eer:.4f} auc={test_report_05.auc:.4f}")
    print(f"  per-class: {test_report_05.per_class_accuracy}")
    print(f"  confusion matrix:\n{test_report_05.confusion_matrix}")

    print(f"\n=== Test @ EER operating point ({op_threshold:.4f}) ===")
    print(f"  accuracy={test_report_op.accuracy:.4f} f1={test_report_op.f1:.4f} "
          f"eer={test_report_op.eer:.4f} auc={test_report_op.auc:.4f}")
    print(f"  per-class: {test_report_op.per_class_accuracy}")
    print(f"  confusion matrix:\n{test_report_op.confusion_matrix}")

    targets_met_05 = meets_targets(test_report_05, TARGET_ACCURACY, TARGET_EER, TARGET_F1, TARGET_PER_CLASS_ACCURACY)
    targets_met_op = meets_targets(test_report_op, TARGET_ACCURACY, TARGET_EER, TARGET_F1, TARGET_PER_CLASS_ACCURACY)
    print(f"\ntargets_met @ 0.5: {targets_met_05}")
    print(f"targets_met @ operating point: {targets_met_op}")

    # --- Update metrics_report.json ---
    with open(METRICS_REPORT_PATH) as f:
        full_report = json.load(f)

    full_report["decision_threshold"] = float(op_threshold)
    full_report["validation"] = val_report_op.to_dict()
    full_report["test"] = test_report_op.to_dict()
    full_report["test_at_default_threshold"] = test_report_05.to_dict()
    full_report["targets_met"] = targets_met_op

    with open(METRICS_REPORT_PATH, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nUpdated {METRICS_REPORT_PATH}")

    # --- Regenerate confusion matrix at the operating point ---
    plot_confusion_matrix(test_report_op.confusion_matrix, save_path=CONFUSION_MATRIX_PATH)
    print(f"Updated {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
