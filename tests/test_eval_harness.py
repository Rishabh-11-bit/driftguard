"""Tests for the evaluation harness metrics: confusion matrix shape, FAR/MAR."""
import numpy as np
import pytest

from eval.metrics import (
    compute_confusion_matrix,
    false_alarm_rate,
    missed_attack_rate,
    compute_all_metrics,
    CLASS_NAMES,
)

# Class indices: 0=malicious, 1=benign_drift, 2=ordinary_learning


def test_confusion_matrix_shape_is_3x3():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)


def test_confusion_matrix_perfect_predictions_is_diagonal():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.sum() == len(y_true)
    assert np.array_equal(np.diag(cm), np.array([2, 2, 2]))
    assert cm.sum() - np.trace(cm) == 0


def test_false_alarm_rate_zero_when_no_benign_misclassified():
    # 3 true benign_drift, all correctly predicted as benign_drift
    y_true = np.array([1, 1, 1, 0, 2])
    y_pred = np.array([1, 1, 1, 0, 2])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert false_alarm_rate(cm) == 0.0


def test_false_alarm_rate_counts_benign_flagged_as_malicious():
    # 4 true benign_drift entries; 1 misclassified as malicious
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert false_alarm_rate(cm) == pytest.approx(0.25)


def test_false_alarm_rate_handles_no_benign_entries():
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 2, 2, 0])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert false_alarm_rate(cm) == 0.0


def test_missed_attack_rate_zero_when_all_malicious_caught():
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 1, 2])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert missed_attack_rate(cm) == 0.0


def test_missed_attack_rate_counts_malicious_missed_as_benign_or_ordinary():
    # 4 true malicious; 1 slips through as benign_drift, 1 as ordinary_learning
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 2])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert missed_attack_rate(cm) == pytest.approx(0.5)


def test_missed_attack_rate_handles_no_malicious_entries():
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 2, 2, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert missed_attack_rate(cm) == 0.0


def test_compute_all_metrics_contains_expected_keys():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=60)
    y_pred = rng.integers(0, 3, size=60)
    metrics = compute_all_metrics(y_true, y_pred)

    assert metrics["confusion_matrix"].shape == (3, 3)
    assert 0.0 <= metrics["FAR"] <= 1.0
    assert 0.0 <= metrics["MAR"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert set(metrics["classification_report"].keys()) >= set(CLASS_NAMES)


def test_compute_all_metrics_with_proba_adds_roc_auc():
    rng = np.random.default_rng(1)
    n = 60
    y_true = rng.integers(0, 3, size=n)
    logits = rng.normal(size=(n, 3))
    y_proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_pred = y_proba.argmax(axis=1)

    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    assert "roc_auc_macro" in metrics
    assert 0.0 <= metrics["roc_auc_macro"] <= 1.0
