"""
Evaluation metrics for DriftGuard.

Computes and visualises the full evaluation suite:
  - 3×3 confusion matrix
  - Per-class precision, recall, F1
  - False Alarm Rate (FAR) on benign drift
  - Missed Attack Rate (MAR) on malicious class
  - ROC curves (one-vs-rest per class)
  - Per-attack-family breakdown

Author: Rishabh Gaikwad (1032231674) — Objective 5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["malicious", "benign_drift", "ordinary_learning"]
CLASS_LABELS = [0, 1, 2]


# ─────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute the 3×3 confusion matrix.

    Returns
    -------
    np.ndarray of shape (3, 3): rows = true class, cols = predicted class.
    """
    return confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)


def false_alarm_rate(cm: np.ndarray) -> float:
    """
    False Alarm Rate on the benign_drift class:
      FAR = FP(benign_drift classified as malicious) / (total benign_drift)

    This is the primary metric DriftGuard optimises — the rate at which
    legitimate preference changes are incorrectly quarantined.
    """
    # cm[1, 0] = true benign_drift predicted as malicious
    true_benign = cm[1, :].sum()
    if true_benign == 0:
        return 0.0
    return float(cm[1, 0]) / float(true_benign)


def missed_attack_rate(cm: np.ndarray) -> float:
    """
    Missed Attack Rate:
      MAR = FN(malicious not caught) / (total malicious)
    """
    true_malicious = cm[0, :].sum()
    if true_malicious == 0:
        return 0.0
    # cm[0, 1] + cm[0, 2] = malicious predicted as benign_drift or ordinary
    return float(cm[0, 1] + cm[0, 2]) / float(true_malicious)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute all DriftGuard evaluation metrics.

    Parameters
    ----------
    y_true   : true integer labels
    y_pred   : predicted integer labels
    y_proba  : predicted probability matrix of shape (n, 3), optional

    Returns
    -------
    dict with keys: cm, FAR, MAR, classification_report, roc_auc (if proba given)
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    far = false_alarm_rate(cm)
    mar = missed_attack_rate(cm)
    report = classification_report(
        y_true, y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
    )

    result: dict[str, Any] = {
        "confusion_matrix": cm,
        "FAR":              far,
        "MAR":              mar,
        "classification_report": report,
        "accuracy":         float(report["accuracy"]),
        "f1_macro":         float(report["macro avg"]["f1-score"]),
    }

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(
                y_true, y_proba,
                multi_class="ovr",
                average="macro",
                labels=CLASS_LABELS,
            )
            result["roc_auc_macro"] = float(roc_auc)
        except Exception as e:
            logger.warning(f"ROC-AUC computation failed: {e}")

    return result


def print_metrics_report(metrics: dict[str, Any]) -> None:
    """Pretty-print a metrics dict to stdout."""
    cm   = metrics["confusion_matrix"]
    far  = metrics["FAR"]
    mar  = metrics["MAR"]
    acc  = metrics["accuracy"]
    f1   = metrics["f1_macro"]
    auc  = metrics.get("roc_auc_macro", "N/A")

    print("\n" + "="*60)
    print("  DriftGuard Evaluation Results")
    print("="*60)
    print(f"\n  Accuracy:           {acc:.4f}")
    print(f"  F1 (macro):         {f1:.4f}")
    if auc != "N/A":
        print(f"  ROC-AUC (macro):    {auc:.4f}")
    print(f"\n  False Alarm Rate:   {far:.4f}  (benign drift → quarantine)")
    print(f"  Missed Attack Rate: {mar:.4f}  (malicious not caught)")
    print("\n  3×3 Confusion Matrix")
    print("  (rows=true, cols=predicted)\n")

    header = "  {:>20s}  {:>15s}  {:>15s}  {:>18s}".format(
        "", *CLASS_NAMES
    )
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  {:>20s}  {:>15d}  {:>15d}  {:>18d}".format(
            name, int(cm[i, 0]), int(cm[i, 1]), int(cm[i, 2])
        )
        print(row)

    print("\n  Per-class Report:")
    report = metrics["classification_report"]
    for cls in CLASS_NAMES:
        m = report.get(cls, {})
        print(
            f"    {cls:<22s}  P={m.get('precision',0):.3f}  "
            f"R={m.get('recall',0):.3f}  F1={m.get('f1-score',0):.3f}"
        )
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str | Path,
    title: str = "DriftGuard 3×3 Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Save a heatmap of the confusion matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_plot = cm.astype(float)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm_plot / row_sums
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Metrics] Confusion matrix saved to {output_path}.")


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save one-vs-rest ROC curves for all three classes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e63946", "#2a9d8f", "#457b9d"]

    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        y_bin = (y_true == cls_idx).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, cls_idx])
        # np.trapz was removed in NumPy 2.0, renamed to np.trapezoid.
        trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        auc_val = trapz_fn(tpr, fpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("DriftGuard ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Metrics] ROC curves saved to {output_path}.")


def plot_family_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_families: list[str],
    output_path: str | Path,
) -> None:
    """Bar chart of per-attack-family detection rates."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "y_true":  y_true,
        "y_pred":  y_pred,
        "family":  attack_families,
    })
    # Only look at malicious entries
    mal = df[df["y_true"] == 0]
    if mal.empty:
        logger.warning("[Metrics] No malicious entries for family breakdown.")
        return

    detection_rates = (
        mal.groupby("family")
        .apply(lambda g: (g["y_pred"] == 0).mean())
        .reset_index()
        .rename(columns={0: "detection_rate"})
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        detection_rates["family"],
        detection_rates["detection_rate"],
        color=["#e63946"] * len(detection_rates),
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", lw=1.5, label="50% baseline")
    ax.set_xlabel("Attack Family", fontsize=12)
    ax.set_ylabel("Detection Rate", fontsize=12)
    ax.set_title("Detection Rate by Attack Family", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=10,
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Metrics] Family breakdown saved to {output_path}.")
