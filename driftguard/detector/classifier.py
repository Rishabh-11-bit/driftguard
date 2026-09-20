"""
DriftGuard three-class classifier.

Primary backend: scikit-learn RandomForestClassifier (interpretable, fast,
enables feature importance for the ablation study).

Optional backend: PyTorch MLP (better on richer feature interactions).

Classes:
  0 — MALICIOUS (poisoning attack)
  1 — BENIGN_DRIFT (legitimate preference change)
  2 — ORDINARY_LEARNING (normal preference write)

Author: Rishabh Gaikwad (1032231674) — Objective 3
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

logger = logging.getLogger(__name__)

Backend = Literal["random_forest", "gradient_boosting", "logistic", "mlp"]

CLASS_NAMES = ["malicious", "benign_drift", "ordinary_learning"]


# ─────────────────────────────────────────────────────────────
# DriftGuardClassifier
# ─────────────────────────────────────────────────────────────

class DriftGuardClassifier:
    """
    Three-class classifier for DriftGuard.

    Parameters
    ----------
    backend : Backend
        ML backend to use:
          "random_forest"    — sklearn RF (default, interpretable)
          "gradient_boosting"— sklearn GBM
          "logistic"         — calibrated logistic regression (fast baseline)
          "mlp"              — PyTorch MLP (requires torch extra)
    n_estimators : int
        Number of trees (for RF and GBM backends).
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        backend: Backend = "random_forest",
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.backend = backend
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Any = None
        self._feature_names: list[str] | None = None
        self._is_fitted = False

    # ── build ────────────────────────────────────────────────

    def _build_model(self) -> Any:
        if self.backend == "random_forest":
            base = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    CalibratedClassifierCV(base, cv=3, method="isotonic")),
            ])

        elif self.backend == "gradient_boosting":
            base = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    base),
            ])

        elif self.backend == "logistic":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=self.random_state,
                )),
            ])

        elif self.backend == "mlp":
            return _PyTorchMLP(random_state=self.random_state)

        else:
            raise ValueError(f"Unknown backend {self.backend!r}")

    # ── training ─────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "DriftGuardClassifier":
        """
        Fit the classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,) with values {0, 1, 2}
        feature_names : list[str] | None
            Optional names for interpretability reporting.
        """
        self._model = self._build_model()
        self._feature_names = feature_names
        logger.info(f"[Classifier] Fitting {self.backend} on {X.shape[0]} samples…")
        self._model.fit(X, y)
        self._is_fitted = True
        logger.info("[Classifier] Fit complete.")
        return self

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> dict[str, float]:
        """
        Run stratified k-fold cross-validation and return metrics.
        """
        model = self._build_model()
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores_acc  = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        scores_f1   = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
        return {
            "accuracy_mean": float(scores_acc.mean()),
            "accuracy_std":  float(scores_acc.std()),
            "f1_macro_mean": float(scores_f1.mean()),
            "f1_macro_std":  float(scores_f1.std()),
        }

    # ── inference ────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(X)

    def predict_one(self, x: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Classify a single feature vector.

        Returns
        -------
        (predicted_class: int, probabilities: np.ndarray of shape (3,))
        """
        self._check_fitted()
        x2d = x.reshape(1, -1)
        pred = int(self._model.predict(x2d)[0])
        proba = self._model.predict_proba(x2d)[0]
        return pred, proba

    # ── interpretability ─────────────────────────────────────

    def feature_importances(self) -> pd.DataFrame | None:
        """
        Return feature importance as a DataFrame (RF and GBM only).
        Returns None for logistic and MLP backends.
        """
        self._check_fitted()
        if self.backend == "mlp":
            logger.warning("Feature importances not available for MLP backend.")
            return None

        # Navigate the sklearn Pipeline
        step = self._model.named_steps.get("clf")
        if step is None:
            return None

        # CalibratedClassifierCV fits one clone of the base estimator per CV
        # fold (in `calibrated_classifiers_`); `step.estimator` is just the
        # original *unfitted* prototype, so average importances across folds.
        if hasattr(step, "calibrated_classifiers_"):
            fold_importances = [
                cc.estimator.feature_importances_
                for cc in step.calibrated_classifiers_
                if hasattr(cc.estimator, "feature_importances_")
            ]
            if not fold_importances:
                return None
            importances = np.mean(fold_importances, axis=0)
        elif hasattr(step, "feature_importances_"):
            importances = step.feature_importances_
        else:
            return None

        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        df = pd.DataFrame({
            "feature":    names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # ── persistence ──────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "backend":       self.backend,
                "n_estimators":  self.n_estimators,
                "random_state":  self.random_state,
                "model":         self._model,
                "feature_names": self._feature_names,
                "is_fitted":     self._is_fitted,
            }, f)
        logger.info(f"[Classifier] Saved to {path}.")

    @classmethod
    def load(cls, path: str | Path) -> "DriftGuardClassifier":
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)
        obj = cls(
            backend=state["backend"],
            n_estimators=state["n_estimators"],
            random_state=state["random_state"],
        )
        obj._model = state["model"]
        obj._feature_names = state["feature_names"]
        obj._is_fitted = state["is_fitted"]
        logger.info(f"[Classifier] Loaded from {path}.")
        return obj

    # ── helpers ──────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Classifier is not fitted. Call fit() or load() first."
            )


# ─────────────────────────────────────────────────────────────
# PyTorch MLP (optional backend)
# ─────────────────────────────────────────────────────────────

class _PyTorchMLP:
    """Lightweight PyTorch MLP wrapped to match sklearn's fit/predict API."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 32),
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._net = None
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_PyTorchMLP":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with: uv sync --extra torch"
            )

        torch.manual_seed(self.random_state)
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        ys = y.astype(np.int64)

        in_dim = Xs.shape[1]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self._net = nn.Sequential(*layers)

        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataset   = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
        loader    = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._net.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimiser.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimiser.step()
        self._net.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        Xs = torch.from_numpy(self._scaler.transform(X).astype(np.float32))
        with torch.no_grad():
            logits = self._net(Xs)
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        Xs = torch.from_numpy(self._scaler.transform(X).astype(np.float32))
        with torch.no_grad():
            logits = self._net(Xs)
            proba  = F.softmax(logits, dim=1)
        return proba.numpy()
