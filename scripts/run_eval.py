"""
CLI: Run the full DriftGuard evaluation suite.

Usage:
    uv run python scripts/run_eval.py \\
        --benchmark data/benchmark.jsonl \\
        --model models/classifier.pkl \\
        --out results/

Author: Rishabh Gaikwad (1032231674) — Objective 5
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

from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import FeatureExtractor
from eval.harness import EvaluationHarness
from eval.metrics import (
    print_metrics_report,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_family_breakdown,
)
from eval.baselines.rule_based import RuleBasedDetector

console = Console()
app = typer.Typer(help="Run the full DriftGuard evaluation suite.")


@app.command()
def main(
    benchmark:   Path = typer.Option(Path("data/benchmark.jsonl"),   "--benchmark"),
    model:       Path = typer.Option(Path("models/classifier.pkl"),   "--model"),
    out:         Path = typer.Option(Path("results"),                 "--out"),
    held_out:    Path = typer.Option(Path("data/held_out.jsonl"),     "--held-out"),
    test_size:  float = typer.Option(0.2,                             "--test-size"),
    run_baselines: bool = typer.Option(True,                          "--baselines/--no-baselines"),
    seed:         int  = typer.Option(42,                             "--seed"),
) -> None:
    """
    Run the full DriftGuard evaluation suite and save all plots + JSON results.
    """
    console.rule("[bold cyan]DriftGuard Full Evaluation Suite[/bold cyan]")
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────
    console.print(f"  Loading model from [bold]{model}[/bold]…")
    clf = DriftGuardClassifier.load(model)

    # ── Load and extract features ────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        t = p.add_task("Extracting features from benchmark…", total=None)
        harness = EvaluationHarness(benchmark, test_size=test_size, random_state=seed)
        harness.load_and_extract()
        p.update(t, description="✓ Features extracted", completed=True)

    X_train, X_test, y_train, y_test, families_test = harness.get_train_test_split()
    contents_test = [harness._entries[i].content for i in range(len(harness._entries))][-len(X_test):]

    # ── DriftGuard evaluation ────────────────────────────────
    console.print("\n[bold]── DriftGuard Detector ──[/bold]")
    dg_result = harness.evaluate_driftguard(clf, X_test, y_test, families_test)
    print_metrics_report(dg_result.metrics)
    harness.save_results(dg_result, out)

    plot_confusion_matrix(
        dg_result.metrics["confusion_matrix"],
        out / "driftguard_confusion_matrix.png",
        title="DriftGuard 3×3 Confusion Matrix",
    )
    if dg_result.y_proba is not None:
        plot_roc_curves(dg_result.y_true, dg_result.y_proba, out / "driftguard_roc_curves.png")
    plot_family_breakdown(
        dg_result.y_true, dg_result.y_pred, dg_result.attack_families,
        out / "driftguard_family_breakdown.png",
    )

    # ── Baselines ─────────────────────────────────────────────
    if run_baselines:
        console.print("\n[bold]── Rule-Based Baseline ──[/bold]")
        rule_result = harness.evaluate_baseline(
            RuleBasedDetector(), "Rule-Based", X_test, y_test, families_test
        )
        print_metrics_report(rule_result.metrics)
        harness.save_results(rule_result, out)
        plot_confusion_matrix(
            rule_result.metrics["confusion_matrix"],
            out / "rule_based_confusion_matrix.png",
            title="Rule-Based Baseline 3×3 Confusion Matrix",
        )

    # ── Held-out generalisation ──────────────────────────────
    console.print("\n[bold]── Held-Out Generalisation Test ──[/bold]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        t = p.add_task("Evaluating on held-out attack families…", total=None)
        held_result = harness.evaluate_held_out(clf, held_out)
        p.update(t, description="✓ Held-out evaluation complete", completed=True)

    print_metrics_report(held_result.metrics)
    harness.save_results(held_result, out)
    plot_confusion_matrix(
        held_result.metrics["confusion_matrix"],
        out / "held_out_confusion_matrix.png",
        title="DriftGuard — Held-Out Attacks Confusion Matrix",
    )

    # ── Summary comparison table ─────────────────────────────
    results = [dg_result, held_result]
    if run_baselines:
        results.insert(1, rule_result)

    summary = Table(title="Evaluation Summary", header_style="bold magenta")
    summary.add_column("Detector",  style="cyan", width=25)
    summary.add_column("Accuracy",  style="white")
    summary.add_column("F1 Macro",  style="white")
    summary.add_column("FAR ↓",     style="red")
    summary.add_column("MAR ↓",     style="red")

    for r in results:
        m = r.metrics
        summary.add_row(
            r.detector_name,
            f"{m['accuracy']:.4f}",
            f"{m['f1_macro']:.4f}",
            f"{m['FAR']:.4f}",
            f"{m['MAR']:.4f}",
        )
    console.print(summary)
    console.print(f"\n[bold green]✓ All results saved to {out}/[/bold green]")
    console.print(f"  Plots: confusion matrices, ROC curves, family breakdown")


if __name__ == "__main__":
    app()
