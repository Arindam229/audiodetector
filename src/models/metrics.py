"""Evaluation metrics for the deepfake audio detector.

Computes:
  * Overall accuracy and per-class accuracy (genuine vs. deepfake)
  * F1-score and a full confusion matrix (+ Seaborn heatmap)
  * Equal Error Rate (EER): the point on the ROC curve where the False
    Acceptance Rate (FAR == FPR) equals the False Rejection Rate
    (FRR == FNR == 1 - TPR), found by locating the minimal |FAR - FRR|
    crossing of the two curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

from src.config import CLASS_NAMES


@dataclass
class EvaluationReport:
    accuracy: float
    f1: float
    confusion_matrix: np.ndarray
    per_class_accuracy: dict
    eer: float
    eer_threshold: float
    auc: float
    fpr: np.ndarray = field(repr=False)
    tpr: np.ndarray = field(repr=False)
    thresholds: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "accuracy": float(self.accuracy),
            "f1": float(self.f1),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_accuracy": {k: float(v) for k, v in self.per_class_accuracy.items()},
            "eer": float(self.eer),
            "eer_threshold": float(self.eer_threshold),
            "auc": float(self.auc),
            "roc": {
                "fpr": self.fpr.tolist(),
                "tpr": self.tpr.tolist(),
                "thresholds": self.thresholds.tolist(),
            },
        }


def compute_eer(y_true: np.ndarray, y_score: np.ndarray):
    """Locate the Equal Error Rate on the ROC curve.

    FAR (False Acceptance Rate) == FPR
    FRR (False Rejection Rate)  == 1 - TPR (FNR)

    The EER is the value at which these two curves cross, found as the
    point minimizing |FAR - FRR| along the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    return eer, eer_threshold, fpr, tpr, thresholds


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> EvaluationReport:
    """Compute the full evaluation report.

    Parameters
    ----------
    y_true : array of {0, 1} ground-truth labels (1 = deepfake)
    y_score : array of model probabilities for the deepfake class (1)
    threshold : decision threshold applied to y_score to obtain y_pred
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    per_class_acc = {}
    for label, name in CLASS_NAMES.items():
        row_total = cm[label].sum()
        per_class_acc[name] = float(cm[label, label] / row_total) if row_total > 0 else 0.0

    eer, eer_threshold, fpr, tpr, thresholds = compute_eer(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return EvaluationReport(
        accuracy=float(acc),
        f1=float(f1),
        confusion_matrix=cm,
        per_class_accuracy=per_class_acc,
        eer=eer,
        eer_threshold=eer_threshold,
        auc=float(roc_auc),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[Path] = None, ax=None):
    """Render a Seaborn heatmap of the confusion matrix."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4.2))

    labels = [CLASS_NAMES[0].split(" ")[0], CLASS_NAMES[1].split(" ")[0]]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#222",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    if own_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
        return fig
    return ax


def plot_roc_with_eer(fpr: np.ndarray, tpr: np.ndarray, eer: float, roc_auc: float,
                       save_path: Optional[Path] = None, ax=None):
    """Render the ROC curve with the EER operating point marked."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4.2))

    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))

    ax.plot(fpr, tpr, color="#7c3aed", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#555", lw=1, linestyle="--", label="Chance")
    ax.scatter([fpr[idx]], [tpr[idx]], color="#f43f5e", zorder=5,
               label=f"EER = {eer:.3%}")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title("ROC Curve with EER Operating Point")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    if own_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
        return fig
    return ax


def meets_targets(report: EvaluationReport, target_accuracy: float, target_eer: float,
                   target_f1: float, target_per_class_accuracy: float) -> bool:
    """Check the report against the production verification thresholds."""
    per_class_ok = all(v >= target_per_class_accuracy for v in report.per_class_accuracy.values())
    return (
        report.accuracy >= target_accuracy
        and report.eer <= target_eer
        and report.f1 >= target_f1
        and per_class_ok
    )
