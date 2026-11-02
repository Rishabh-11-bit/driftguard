"""
Extended performance metrics for DriftGuard.

Complements eval/metrics.py (confusion matrix, FAR, MAR, ROC-AUC) with the
quantities a memory-poisoning detector is actually judged on:

  - Agreement metrics: balanced accuracy, MCC, Cohen's kappa
  - Security view (malicious vs. everything else): TPR, FPR, precision, F2,
    attack-containment rate
  - Utility view (benign drift): drift-preservation rate, drift-recognition
    rate, ordinary-learning pass-through rate
  - Probabilistic quality: PR-AUC of the malicious class, multiclass Brier
    score, expected calibration error (ECE)
  - Decision cost: asymmetric expected-risk under a configurable cost matrix
  - Uncertainty: percentile bootstrap confidence intervals
  - Generalisation: per-family detection rates and the in-distribution ->
    held-out generalisation gap

All functions take plain numpy arrays and are deterministic given a seed.

Author: Rishabh Gaikwad (1032231674) — Objective 5
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

MALICIOUS, BENIGN_DRIFT, ORDINARY = 0, 1, 2
N_CLASSES = 3

# cost[i, j] = cost of predicting class j when the truth is class i.
# Letting an attack through (row 0, cols 1-2) is the worst outcome; wrongly
# quarantining benign drift (row 1, col 0) costs a user-visible false alarm;
# mistaking drift for ordinary learning only loses a review flag.
DEFAULT_COST_MATRIX = np.array(
    [
        [0.0, 10.0, 10.0],
        [3.0,  0.0,  0.5],
        [1.0,  0.2,  0.0],
    ]
)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _rate(num: float, den: float) -> float | None:
    """Like _safe_div but None when the denominator class has no support."""
    return float(num) / float(den) if den else None


# ─────────────────────────────────────────────────────────────
# Security / utility rates
# ─────────────────────────────────────────────────────────────

def security_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Binary view: malicious (positive) vs benign_drift + ordinary (negative)."""
    pos_true = y_true == MALICIOUS
    pos_pred = y_pred == MALICIOUS
    tp = int(np.sum(pos_true & pos_pred))
    fp = int(np.sum(~pos_true & pos_pred))
    fn = int(np.sum(pos_true & ~pos_pred))
    tn = int(np.sum(~pos_true & ~pos_pred))

    precision = _safe_div(tp, tp + fp)
    tpr = _safe_div(tp, tp + fn)
    beta2 = 4.0  # F2: recall weighted 4x precision, a missed attack is worse than a false alarm
    f2 = _safe_div((1 + beta2) * precision * tpr, beta2 * precision + tpr)
    return {
        "attack_tpr": tpr,
        "attack_fpr": _safe_div(fp, fp + tn),
        "attack_tnr": _safe_div(tn, tn + fp),
        "attack_precision": precision,
        "attack_f2": f2,
        "attack_containment_rate": tpr,
    }


def utility_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """What the user keeps: benign drift and ordinary learning must survive."""
    drift = y_true == BENIGN_DRIFT
    ordinary = y_true == ORDINARY
    return {
        # drift not quarantined (admitted either as flagged drift or ordinary)
        "drift_preservation_rate": _rate(np.sum(drift & (y_pred != MALICIOUS)), drift.sum()),
        # drift admitted *and* flagged for review (the intended outcome)
        "drift_recognition_rate": _rate(np.sum(drift & (y_pred == BENIGN_DRIFT)), drift.sum()),
        "ordinary_passthrough_rate": _rate(np.sum(ordinary & (y_pred != MALICIOUS)), ordinary.sum()),
        "legit_quarantine_rate": _rate(
            np.sum((drift | ordinary) & (y_pred == MALICIOUS)), (drift | ordinary).sum()
        ),
    }


# ─────────────────────────────────────────────────────────────
# Probabilistic quality
# ─────────────────────────────────────────────────────────────

def multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared distance between the probability vector and the one-hot truth."""
    onehot = np.eye(N_CLASSES)[y_true]
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label ECE: |confidence - accuracy| averaged over equal-width bins."""
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def malicious_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    """Area under the precision-recall curve for the malicious class."""
    y_bin = (y_true == MALICIOUS).astype(int)
    if y_bin.sum() == 0:
        return None
    return float(average_precision_score(y_bin, y_proba[:, MALICIOUS]))


# ─────────────────────────────────────────────────────────────
# Decision cost
# ─────────────────────────────────────────────────────────────

def expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: np.ndarray = DEFAULT_COST_MATRIX,
) -> dict[str, float]:
    """
    Mean per-write cost under `cost_matrix`, plus the cost of the trivial
    policy that admits everything (always predicts ordinary) as reference.
    `cost_reduction` is 1 - cost / admit_all_cost.
    """
    cost = float(np.mean(cost_matrix[y_true, y_pred]))
    admit_all = float(np.mean(cost_matrix[y_true, ORDINARY]))
    return {
        "expected_cost": cost,
        "admit_all_cost": admit_all,
        "cost_reduction": 1.0 - _safe_div(cost, admit_all) if admit_all else 0.0,
    }


# ─────────────────────────────────────────────────────────────
# Uncertainty
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap (1 - alpha) confidence interval of stat_fn."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = stat_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _far(yt: np.ndarray, yp: np.ndarray) -> float:
    d = yt == BENIGN_DRIFT
    return _safe_div(np.sum(d & (yp == MALICIOUS)), d.sum())


def _mar(yt: np.ndarray, yp: np.ndarray) -> float:
    m = yt == MALICIOUS
    return _safe_div(np.sum(m & (yp != MALICIOUS)), m.sum())


def _acc(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(yt == yp))


def confidence_intervals(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> dict[str, tuple[float, float]]:
    """95% bootstrap CIs for the headline metrics."""
    return {
        "accuracy": bootstrap_ci(y_true, y_pred, _acc, n_boot, seed=seed),
        "FAR": bootstrap_ci(y_true, y_pred, _far, n_boot, seed=seed),
        "MAR": bootstrap_ci(y_true, y_pred, _mar, n_boot, seed=seed),
    }


# ─────────────────────────────────────────────────────────────
# Generalisation
# ─────────────────────────────────────────────────────────────

def per_family_detection(
    y_true: np.ndarray, y_pred: np.ndarray, families: list[str]
) -> dict[str, dict[str, float]]:
    """Detection rate and confusion destinations for each attack family."""
    fam = np.asarray(families)
    out: dict[str, dict[str, float]] = {}
    for f in sorted(set(fam[y_true == MALICIOUS])):
        mask = (fam == f) & (y_true == MALICIOUS)
        n = int(mask.sum())
        out[f] = {
            "n": n,
            "detection_rate": _safe_div(np.sum(y_pred[mask] == MALICIOUS), n),
            "to_benign_drift": _safe_div(np.sum(y_pred[mask] == BENIGN_DRIFT), n),
            "to_ordinary": _safe_div(np.sum(y_pred[mask] == ORDINARY), n),
        }
    return out


def generalisation_gap(in_dist: dict[str, Any], held_out: dict[str, Any]) -> dict[str, float]:
    """
    Drop from in-distribution to held-out attack families. A positive
    `mar_gap` means the detector misses more of the unseen attacks.
    """
    return {
        "mar_gap": held_out["MAR"] - in_dist["MAR"],
        "attack_recall_gap": in_dist["attack_tpr"] - held_out["attack_tpr"],
        "accuracy_gap": in_dist["accuracy"] - held_out["accuracy"],
    }


# ─────────────────────────────────────────────────────────────
# Aggregate entry point
# ─────────────────────────────────────────────────────────────

def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    families: list[str] | None = None,
    cost_matrix: np.ndarray = DEFAULT_COST_MATRIX,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute every extended metric; probability-based ones need `y_proba`."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(np.nan_to_num(cohen_kappa_score(y_true, y_pred))),
    }
    result.update(security_metrics(y_true, y_pred))
    result.update(utility_metrics(y_true, y_pred))
    result.update(expected_cost(y_true, y_pred, cost_matrix))
    result["confidence_intervals"] = confidence_intervals(y_true, y_pred, n_boot, seed)

    if y_proba is not None:
        result["brier_score"] = multiclass_brier(y_true, y_proba)
        result["ece"] = expected_calibration_error(y_true, y_proba)
        result["malicious_pr_auc"] = malicious_pr_auc(y_true, y_proba)
    if families is not None:
        result["per_family"] = per_family_detection(y_true, y_pred, families)
    return result


def print_extended_report(ext: dict[str, Any]) -> None:
    """Pretty-print the extended metrics dict to stdout."""
    def row(label: str, key: str) -> None:
        v = ext.get(key)
        if v is not None:
            print(f"    {label:<30s} {v:.4f}")

    print("\n  Extended Performance Metrics")
    print("  Agreement")
    row("Balanced accuracy", "balanced_accuracy")
    row("Matthews corr. (MCC)", "mcc")
    row("Cohen's kappa", "cohen_kappa")
    print("  Security (malicious vs rest)")
    row("Attack recall / containment", "attack_tpr")
    row("Attack precision", "attack_precision")
    row("Attack F2", "attack_f2")
    row("Attack false-positive rate", "attack_fpr")
    print("  Utility (what the user keeps)")
    row("Drift preservation rate", "drift_preservation_rate")
    row("Drift recognition rate", "drift_recognition_rate")
    row("Ordinary pass-through rate", "ordinary_passthrough_rate")
    row("Legit-write quarantine rate", "legit_quarantine_rate")
    print("  Probabilistic quality")
    row("Malicious PR-AUC", "malicious_pr_auc")
    row("Brier score (lower better)", "brier_score")
    row("ECE (lower better)", "ece")
    print("  Decision cost")
    row("Expected cost / write", "expected_cost")
    row("Admit-all reference cost", "admit_all_cost")
    row("Cost reduction vs admit-all", "cost_reduction")
    row("Latency ms / write", "latency_ms_per_write")
    ci = ext.get("confidence_intervals", {})
    if ci:
        print("  95% bootstrap CIs")
        for k, (lo, hi) in ci.items():
            print(f"    {k:<30s} [{lo:.4f}, {hi:.4f}]")
    fam = ext.get("per_family")
    if fam:
        print("  Per-family detection")
        for f, d in fam.items():
            print(f"    {f:<30s} n={d['n']:<4d} detect={d['detection_rate']:.3f} "
                  f"->drift={d['to_benign_drift']:.3f} ->ordinary={d['to_ordinary']:.3f}")
    print()
