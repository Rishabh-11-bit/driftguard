"""Tests for eval/extended_metrics.py — values checked against hand computation."""

import numpy as np
import pytest

from eval.extended_metrics import (
    DEFAULT_COST_MATRIX,
    bootstrap_ci,
    compute_extended_metrics,
    expected_calibration_error,
    expected_cost,
    generalisation_gap,
    multiclass_brier,
    per_family_detection,
    security_metrics,
    utility_metrics,
)

# truth:  M M M M | D D D D | O O O O
Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
# pred:   3 caught, 1 -> ordinary | 1 quarantined, 2 drift, 1 ordinary | 3 ordinary, 1 quarantined
Y_PRED = np.array([0, 0, 0, 2, 0, 1, 1, 2, 2, 2, 2, 0])


def test_security_metrics_hand_computed():
    m = security_metrics(Y_TRUE, Y_PRED)
    # TP=3, FN=1, FP=2 (one drift, one ordinary), TN=6
    assert m["attack_tpr"] == pytest.approx(3 / 4)
    assert m["attack_precision"] == pytest.approx(3 / 5)
    assert m["attack_fpr"] == pytest.approx(2 / 8)
    p, r = 3 / 5, 3 / 4
    assert m["attack_f2"] == pytest.approx(5 * p * r / (4 * p + r))


def test_utility_metrics_hand_computed():
    m = utility_metrics(Y_TRUE, Y_PRED)
    assert m["drift_preservation_rate"] == pytest.approx(3 / 4)
    assert m["drift_recognition_rate"] == pytest.approx(2 / 4)
    assert m["ordinary_passthrough_rate"] == pytest.approx(3 / 4)
    assert m["legit_quarantine_rate"] == pytest.approx(2 / 8)


def test_utility_metrics_none_without_support():
    only_attacks = np.zeros(5, dtype=int)
    m = utility_metrics(only_attacks, only_attacks)
    assert all(v is None for v in m.values())


def test_expected_cost_perfect_and_admit_all():
    perfect = expected_cost(Y_TRUE, Y_TRUE)
    assert perfect["expected_cost"] == 0.0
    assert perfect["cost_reduction"] == pytest.approx(1.0)
    admit_all = expected_cost(Y_TRUE, np.full_like(Y_TRUE, 2))
    assert admit_all["cost_reduction"] == pytest.approx(0.0)
    assert admit_all["admit_all_cost"] == pytest.approx(
        DEFAULT_COST_MATRIX[Y_TRUE, 2].mean()
    )


def test_brier_perfect_and_worst():
    onehot = np.eye(3)[Y_TRUE]
    assert multiclass_brier(Y_TRUE, onehot) == 0.0
    wrong = np.eye(3)[(Y_TRUE + 1) % 3]
    assert multiclass_brier(Y_TRUE, wrong) == pytest.approx(2.0)


def test_ece_perfectly_calibrated_and_overconfident():
    y = np.array([0, 0, 0, 0])
    sure_right = np.tile([1.0, 0.0, 0.0], (4, 1))
    assert expected_calibration_error(y, sure_right) == pytest.approx(0.0)
    sure_wrong = np.tile([0.0, 1.0, 0.0], (4, 1))
    assert expected_calibration_error(y, sure_wrong) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point_estimate_and_is_seeded():
    acc = lambda t, p: float(np.mean(t == p))
    lo, hi = bootstrap_ci(Y_TRUE, Y_PRED, acc, n_boot=300, seed=1)
    assert lo <= acc(Y_TRUE, Y_PRED) <= hi
    assert (lo, hi) == bootstrap_ci(Y_TRUE, Y_PRED, acc, n_boot=300, seed=1)


def test_per_family_detection():
    fam = ["a", "a", "b", "b"] + ["-"] * 8
    out = per_family_detection(Y_TRUE, Y_PRED, fam)
    assert out["a"]["detection_rate"] == 1.0
    assert out["b"]["detection_rate"] == 0.5
    assert out["b"]["to_ordinary"] == 0.5
    assert "-" not in out


def test_generalisation_gap_sign():
    g = generalisation_gap(
        {"MAR": 0.0, "attack_tpr": 1.0, "accuracy": 0.99},
        {"MAR": 0.4, "attack_tpr": 0.6, "accuracy": 0.6},
    )
    assert g["mar_gap"] == pytest.approx(0.4)
    assert g["attack_recall_gap"] == pytest.approx(0.4)


def test_compute_extended_metrics_keys_and_ranges():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=len(Y_TRUE))
    out = compute_extended_metrics(Y_TRUE, Y_PRED, proba, ["x"] * 12, n_boot=50)
    for key in ("balanced_accuracy", "mcc", "cohen_kappa", "attack_tpr",
                "drift_preservation_rate", "expected_cost", "brier_score",
                "ece", "malicious_pr_auc", "confidence_intervals", "per_family"):
        assert key in out
    assert -1.0 <= out["mcc"] <= 1.0
    assert 0.0 <= out["ece"] <= 1.0
    assert set(out["confidence_intervals"]) == {"accuracy", "FAR", "MAR"}
