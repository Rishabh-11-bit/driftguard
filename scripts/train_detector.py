"""
CLI: Train the DriftGuard three-class detector.

Usage:
    uv run python scripts/train_detector.py \\
        --benchmark data/benchmark.jsonl \\
        --model-out models/classifier.pkl \\
        --backend random_forest

Author: Rishabh Gaikwad (1032231674)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import typer
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import logging

logging.basicConfig(level=logging.WARNING)

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.features import FeatureExtractor, FEATURE_DIM
from driftguard.detector.classifier import DriftGuardClassifier
from eval.harness import EvaluationHarness

console = Console()
app = typer.Typer(help="Train the DriftGuard three-class detector.")

FEATURE_NAMES = [
    "origin_index", "trust_level", "session_age_hours",
    "writes_last_10", "writes_last_30", "inter_write_interval_mean_min", "burst_score",
    "cosine_sim_to_centroid", "novelty_score", "same_session_writes",
    "unique_origins_count", "repeated_hash_flag",
    "query_content_overlap", "write_length_chars", "contains_instruction",
    "contains_url", "avg_word_length", "exclamation_ratio",
]


@app.command()
def main(
    benchmark:  Path = typer.Option(Path("data/benchmark.jsonl"), "--benchmark", help="Benchmark JSONL path."),
    model_out:  Path = typer.Option(Path("models/classifier.pkl"), "--model-out", help="Output model path."),
    backend:    str  = typer.Option("random_forest", "--backend", help="Classifier backend: random_forest | gradient_boosting | logistic | mlp"),
    n_estimators: int = typer.Option(200, "--n-estimators", help="Number of trees (RF/GBM)."),
    cv_folds:   int  = typer.Option(5, "--cv-folds", help="Cross-validation folds."),
    test_size:  float = typer.Option(0.2, "--test-size", help="Fraction for test split."),
    seed:       int  = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    """
    Train the DriftGuard three-class detector on the labelled benchmark.
    """
    console.rule("[bold cyan]DriftGuard Detector Training[/bold cyan]")

    # ── Load and extract features ────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        t = p.add_task("Loading benchmark and extracting features…", total=None)
        harness = EvaluationHarness(benchmark, test_size=test_size, random_state=seed)
        harness.load_and_extract()
        p.update(t, description="✓ Features extracted", completed=True)

    X_train, X_test, y_train, y_test, families_test = harness.get_train_test_split()
    console.print(f"\n  Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    console.print(f"  Feature dimension: {X_train.shape[1]}")

    # ── Cross-validation ─────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        t = p.add_task(f"Running {cv_folds}-fold cross-validation ({backend})…", total=None)
        clf_cv = DriftGuardClassifier(backend=backend, n_estimators=n_estimators, random_state=seed)
        cv_results = clf_cv.cross_validate(X_train, y_train, cv=cv_folds)
        p.update(t, description="✓ Cross-validation complete", completed=True)

    cv_table = Table(title="Cross-Validation Results", header_style="bold magenta")
    cv_table.add_column("Metric", style="cyan")
    cv_table.add_column("Mean", style="white")
    cv_table.add_column("Std", style="white")
    cv_table.add_row("Accuracy", f"{cv_results['accuracy_mean']:.4f}", f"±{cv_results['accuracy_std']:.4f}")
    cv_table.add_row("F1 (macro)", f"{cv_results['f1_macro_mean']:.4f}", f"±{cv_results['f1_macro_std']:.4f}")
    console.print(cv_table)

    # ── Final model fit ──────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        t = p.add_task(f"Fitting final {backend} model on full train set…", total=None)
        clf = DriftGuardClassifier(backend=backend, n_estimators=n_estimators, random_state=seed)
        clf.fit(X_train, y_train, feature_names=FEATURE_NAMES)
        p.update(t, description="✓ Model trained", completed=True)

    # ── Test evaluation ──────────────────────────────────────
    result = harness.evaluate_driftguard(clf, X_test, y_test, families_test)
    from eval.metrics import print_metrics_report
    print_metrics_report(result.metrics)

    # ── Feature importances ──────────────────────────────────
    fi = clf.feature_importances()
    if fi is not None:
        fi_table = Table(title="Top-10 Feature Importances", header_style="bold magenta")
        fi_table.add_column("Rank",      style="dim")
        fi_table.add_column("Feature",   style="cyan")
        fi_table.add_column("Importance", style="white")
        for i, row in fi.head(10).iterrows():
            fi_table.add_row(str(i+1), row["feature"], f"{row['importance']:.4f}")
        console.print(fi_table)

    # ── Save model ───────────────────────────────────────────
    clf.save(model_out)
    console.print(f"\n[bold green]✓ Model saved to {model_out}[/bold green]")


if __name__ == "__main__":
    app()
