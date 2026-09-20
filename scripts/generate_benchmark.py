"""
CLI: Generate the three-class DriftGuard benchmark.

Usage:
    uv run python scripts/generate_benchmark.py \\
        --n-malicious 200 --n-drift 200 --n-ordinary 200 \\
        --out data/benchmark.jsonl

Authors: Tanisha Talathi (benchmark) / Aryaan Sethi (CLI scaffold)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from driftguard.benchmark.attack_generators.minja_style import generate_minja_entries
from driftguard.benchmark.attack_generators.agentpoison_style import generate_agentpoison_entries
from driftguard.benchmark.benign_generators.preference_drift import generate_preference_drift_entries
from driftguard.benchmark.benign_generators.ordinary_learning import generate_ordinary_learning_entries
from driftguard.benchmark.labelling import save_benchmark, benchmark_stats

console = Console()
app = typer.Typer(help="Generate the DriftGuard three-class benchmark.")


@app.command()
def main(
    n_malicious: int = typer.Option(200, "--n-malicious", help="Number of malicious entries."),
    n_drift:     int = typer.Option(200, "--n-drift",     help="Number of benign-drift entries."),
    n_ordinary:  int = typer.Option(200, "--n-ordinary",  help="Number of ordinary-learning entries."),
    out:    Path = typer.Option(Path("data/benchmark.jsonl"), "--out", help="Output JSONL path."),
    seed:   int  = typer.Option(42, "--seed", help="Random seed."),
    minja_ratio: float = typer.Option(0.5, "--minja-ratio", help="Fraction of malicious from MINJA (rest AgentPoison)."),
) -> None:
    """
    Generate the labelled three-class DriftGuard benchmark.
    """
    console.rule("[bold cyan]DriftGuard Benchmark Generator[/bold cyan]")

    n_minja  = int(n_malicious * minja_ratio)
    n_ap     = n_malicious - n_minja

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        t1 = progress.add_task(f"[red]Generating {n_minja} MINJA-style entries…", total=None)
        minja_entries = generate_minja_entries(n=n_minja, seed=seed)
        progress.update(t1, description=f"[red]✓ {n_minja} MINJA entries", completed=True)

        t2 = progress.add_task(f"[red]Generating {n_ap} AgentPoison-style entries…", total=None)
        ap_entries = generate_agentpoison_entries(n=n_ap, seed=seed+1)
        progress.update(t2, description=f"[red]✓ {n_ap} AgentPoison entries", completed=True)

        t3 = progress.add_task(f"[green]Generating {n_drift} benign-drift entries…", total=None)
        drift_entries = generate_preference_drift_entries(n=n_drift, seed=seed+2)
        progress.update(t3, description=f"[green]✓ {n_drift} benign-drift entries", completed=True)

        t4 = progress.add_task(f"[blue]Generating {n_ordinary} ordinary-learning entries…", total=None)
        ordinary_entries = generate_ordinary_learning_entries(n=n_ordinary, seed=seed+3)
        progress.update(t4, description=f"[blue]✓ {n_ordinary} ordinary-learning entries", completed=True)

    all_entries = minja_entries + ap_entries + drift_entries + ordinary_entries
    random.seed(seed)
    random.shuffle(all_entries)

    stats = benchmark_stats(all_entries)

    # Pretty print stats
    table = Table(title="Benchmark Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total entries", str(stats["total"]))
    for cls, count in stats["class_counts"].items():
        table.add_row(f"  Class: {cls}", str(count))
    table.add_row("─" * 25, "─" * 10)
    for fam, count in stats["attack_families"].items():
        table.add_row(f"  Family: {fam}", str(count))
    console.print(table)

    save_benchmark(all_entries, out)
    console.print(f"\n[bold green]✓ Benchmark saved to {out}[/bold green]")


if __name__ == "__main__":
    app()
