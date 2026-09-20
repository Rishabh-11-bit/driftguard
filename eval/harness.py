"""
Evaluation harness — replays the three-class benchmark through DriftGuard
and binary baselines, producing the full evaluation suite.

Author: Rishabh Gaikwad (1032231674) — Objective 5
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from driftguard.benchmark.labelling import load_benchmark, BenchmarkEntry
from driftguard.detector.features import FeatureExtractor
from driftguard.detector.classifier import DriftGuardClassifier
from eval.metrics import compute_all_metrics, print_metrics_report

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Complete evaluation results for one detector on the benchmark."""
    detector_name: str
    y_true:        np.ndarray
    y_pred:        np.ndarray
    y_proba:       np.ndarray | None
    attack_families: list[str]
    metrics:       dict[str, Any]
    eval_time_sec: float


class EvaluationHarness:
    """
    Replays the labelled benchmark through one or more detectors and
    produces the full evaluation suite.

    Parameters
    ----------
    benchmark_path : str | Path
        Path to the JSONL benchmark file.
    test_size : float
        Fraction held out for evaluation (default 0.2 = 20%).
    random_state : int
        Seed for train/test split.
    """

    def __init__(
        self,
        benchmark_path: str | Path,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.benchmark_path = Path(benchmark_path)
        self.test_size = test_size
        self.random_state = random_state
        self._entries: list[BenchmarkEntry] | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._families: list[str] | None = None

    def load_and_extract(self, extractor: FeatureExtractor | None = None) -> None:
        """Load benchmark and extract features for all entries."""
        logger.info(f"[Harness] Loading benchmark from {self.benchmark_path}…")
        entries = load_benchmark(self.benchmark_path)
        logger.info(f"[Harness] Loaded {len(entries)} entries.")

        extractor = extractor or FeatureExtractor()
        logger.info("[Harness] Extracting features…")

        records = [
            {
                "content":       e.content,
                "provenance":    e.provenance,
                "write_history": e.write_history,
                "session_id":    e.session_id,
                "last_query":    e.last_query,
                "memory_entries":e.memory_entries,
            }
            for e in entries
        ]
        X = extractor.extract_batch(records)
        y = np.array([e.label_int for e in entries], dtype=int)
        families = [e.attack_family for e in entries]

        self._entries  = entries
        self._X        = X
        self._y        = y
        self._families = families
        logger.info(f"[Harness] Feature matrix: {X.shape}  Labels: {np.bincount(y)}")

    def get_train_test_split(self) -> tuple:
        """Return (X_train, X_test, y_train, y_test, families_test)."""
        if self._X is None:
            raise RuntimeError("Call load_and_extract() first.")
        indices = np.arange(len(self._y))
        idx_train, idx_test = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self._y,
        )
        families_test = [self._families[i] for i in idx_test]
        return (
            self._X[idx_train], self._X[idx_test],
            self._y[idx_train], self._y[idx_test],
            families_test,
        )

    def evaluate_driftguard(
        self,
        classifier: DriftGuardClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        families_test: list[str],
    ) -> EvalResult:
        """Evaluate a trained DriftGuardClassifier on the test set."""
        logger.info("[Harness] Running DriftGuard evaluation…")
        t0 = time.perf_counter()
        y_pred  = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
        elapsed = time.perf_counter() - t0

        metrics = compute_all_metrics(y_test, y_pred, y_proba)
        return EvalResult(
            detector_name="DriftGuard",
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            attack_families=families_test,
            metrics=metrics,
            eval_time_sec=elapsed,
        )

    def evaluate_baseline(
        self,
        baseline,
        baseline_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        families_test: list[str],
    ) -> EvalResult:
        """Evaluate a binary baseline detector."""
        logger.info(f"[Harness] Running baseline '{baseline_name}'…")
        t0 = time.perf_counter()
        y_pred  = baseline.predict(X_test)
        elapsed = time.perf_counter() - t0

        try:
            y_proba = baseline.predict_proba(X_test)
        except Exception:
            y_proba = None

        metrics = compute_all_metrics(y_test, y_pred, y_proba)
        return EvalResult(
            detector_name=baseline_name,
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            attack_families=families_test,
            metrics=metrics,
            eval_time_sec=elapsed,
        )

    def evaluate_held_out(
        self,
        classifier: DriftGuardClassifier,
        held_out_path: str | Path,
        extractor: FeatureExtractor | None = None,
    ) -> EvalResult:
        """Evaluate on held-out attack families (generalisation test)."""
        from driftguard.benchmark.held_out_attacks import (
            generate_slow_drip_entries,
            generate_indirect_tool_injection_entries,
        )
        logger.info("[Harness] Running held-out generalisation evaluation…")
        extractor = extractor or FeatureExtractor()

        # Load or generate held-out entries
        held_path = Path(held_out_path)
        if held_path.exists():
            from driftguard.benchmark.labelling import load_benchmark
            entries = load_benchmark(held_path)
        else:
            entries = (
                generate_slow_drip_entries(n=100) +
                generate_indirect_tool_injection_entries(n=100)
            )
            from driftguard.benchmark.labelling import save_benchmark
            save_benchmark(entries, held_path)

        records = [
            {
                "content":       e.content,
                "provenance":    e.provenance,
                "write_history": e.write_history,
                "session_id":    e.session_id,
                "last_query":    e.last_query,
                "memory_entries":e.memory_entries,
            }
            for e in entries
        ]
        X = extractor.extract_batch(records)
        y = np.array([e.label_int for e in entries], dtype=int)
        families = [e.attack_family for e in entries]

        t0 = time.perf_counter()
        y_pred  = classifier.predict(X)
        y_proba = classifier.predict_proba(X)
        elapsed = time.perf_counter() - t0

        metrics = compute_all_metrics(y, y_pred, y_proba)
        return EvalResult(
            detector_name="DriftGuard (held-out)",
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            attack_families=families,
            metrics=metrics,
            eval_time_sec=elapsed,
        )

    @staticmethod
    def save_results(result: EvalResult, output_dir: str | Path) -> None:
        """Save evaluation results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = result.detector_name.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = output_dir / f"{safe_name}_metrics.json"

        serialisable = {
            "detector_name":  result.detector_name,
            "eval_time_sec":  result.eval_time_sec,
            "FAR":            result.metrics["FAR"],
            "MAR":            result.metrics["MAR"],
            "accuracy":       result.metrics["accuracy"],
            "f1_macro":       result.metrics["f1_macro"],
            "roc_auc_macro":  result.metrics.get("roc_auc_macro"),
            "confusion_matrix": result.metrics["confusion_matrix"].tolist(),
            "classification_report": result.metrics["classification_report"],
        }
        with out_path.open("w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"[Harness] Results saved to {out_path}.")
