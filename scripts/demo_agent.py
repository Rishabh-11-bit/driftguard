"""
CLI: Interactive demo agent with DriftGuard live protection.

Chat with the DriftGuard-protected agent and see real-time memory
classification for every write.

Usage:
    uv run python scripts/demo_agent.py

Set in .env:
    LLM_PROVIDER=openai (or ollama)
    OPENAI_API_KEY=sk-...
    DETECTOR_MODEL_PATH=./models/classifier.pkl  (optional — uses pass-through if missing)

Author: Aryaan Sethi (1032231670) — Objective 4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

load_dotenv()
logging.basicConfig(level=logging.WARNING)

from driftguard.agent.memory_backend import get_memory_backend
from driftguard.agent.llm_interface import get_llm
from driftguard.agent.graph import build_agent_graph, run_agent
from driftguard.provenance.write_pipeline import WritePipeline
from driftguard.provenance.enforcement import EnforcementPolicy, HumanReviewQueue
from driftguard.detector.classifier import DriftGuardClassifier
from langchain_core.messages import HumanMessage, AIMessage

console = Console()

ACTION_STYLES = {
    "quarantined":      ("[red]QUARANTINED[/red]",      "🚨"),
    "admitted_flagged": ("[yellow]FLAGGED[/yellow]",    "⚠️ "),
    "admitted":         ("[green]ADMITTED[/green]",     "✅"),
}


def _load_classifier() -> DriftGuardClassifier | None:
    model_path = os.getenv("DETECTOR_MODEL_PATH", "./models/classifier.pkl")
    p = Path(model_path)
    if p.exists():
        console.print(f"  [dim]Loading detector from {p}…[/dim]")
        return DriftGuardClassifier.load(p)
    console.print(
        f"  [yellow]⚠ No trained model found at {p}. "
        "Running in pass-through mode (all writes admitted as ordinary learning).[/yellow]\n"
        "  Run [bold]scripts/generate_benchmark.py[/bold] then "
        "[bold]scripts/train_detector.py[/bold] first."
    )
    return None


def _print_write_results(write_results: list[dict]) -> None:
    if not write_results:
        return
    table = Table(
        box=box.SIMPLE_HEAVY,
        title="[bold]Memory Write Classification[/bold]",
        show_header=True,
        header_style="bold cyan",
        width=100,
    )
    table.add_column("Action",     style="white",  width=18)
    table.add_column("Class",      style="white",  width=18)
    table.add_column("P(mal)",     style="red",    width=8)
    table.add_column("P(drift)",   style="yellow", width=8)
    table.add_column("P(ord)",     style="green",  width=8)
    table.add_column("Content (truncated)", style="dim", width=44)

    for r in write_results:
        action    = r.get("action", "")
        cls_name  = r.get("class_name", "")
        probas    = r.get("probabilities", [0.0, 0.0, 1.0])
        content   = r.get("candidate_text", "")[:60]
        label, emoji = ACTION_STYLES.get(action, ("?", "?"))
        table.add_row(
            f"{emoji} {label}",
            cls_name,
            f"{probas[0]:.3f}",
            f"{probas[1]:.3f}",
            f"{probas[2]:.3f}",
            content,
        )
    console.print(table)


def main() -> None:
    console.print(Panel.fit(
        "[bold cyan]DriftGuard[/bold cyan] — Provenance-Aware Memory Security\n"
        "[dim]Chat with the protected agent. Type [bold]quit[/bold] to exit, "
        "[bold]memory[/bold] to view stored memories, "
        "[bold]quarantine[/bold] to view the review queue.[/dim]",
        title="🛡️  DriftGuard Demo",
        border_style="cyan",
    ))

    # ── Setup ─────────────────────────────────────────────────
    session_id      = str(uuid.uuid4())[:8]
    memory_backend  = get_memory_backend()
    classifier      = _load_classifier()
    review_queue    = HumanReviewQueue(path="./data/quarantine_queue.jsonl")
    enforcement     = EnforcementPolicy(quarantine_queue=review_queue)
    pipeline        = WritePipeline(
        memory_backend=memory_backend,
        classifier=classifier,
        enforcement=enforcement,
    )
    llm             = get_llm()
    graph           = build_agent_graph(
        memory_backend=memory_backend,
        llm=llm,
        pipeline=pipeline,
    )

    console.print(f"  [dim]Session ID: {session_id}[/dim]")
    console.print(f"  [dim]LLM: {os.getenv('LLM_PROVIDER', 'openai')} / {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}[/dim]")
    console.print(f"  [dim]Memory: {os.getenv('MEMORY_BACKEND', 'faiss')}[/dim]\n")

    history: list = []

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "memory":
            entries = memory_backend.list_all()
            console.print(f"\n[bold]Stored Memories[/bold] ({len(entries)} entries):")
            for e in entries[-10:]:
                flagged = e.get("metadata", {}).get("flagged", False)
                flag_icon = "⚠️" if flagged else "  "
                console.print(f"  {flag_icon} {e['text'][:80]}")
            console.print()
            continue

        if user_input.lower() == "quarantine":
            pending = review_queue.list_pending()
            console.print(f"\n[bold red]Quarantine Queue[/bold red] ({len(pending)} pending):")
            for q in pending:
                console.print(f"  🚨 [{q.quarantine_id[:8]}] {q.candidate_text[:70]}")
            console.print()
            continue

        # ── Run agent turn ─────────────────────────────────────
        try:
            response, write_results = run_agent(
                user_message=user_input,
                session_id=session_id,
                compiled_graph=graph,
                message_history=history,
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if "OPENAI_API_KEY" in str(e) or "api_key" in str(e).lower():
                console.print(
                    "[yellow]Tip: Add your OPENAI_API_KEY to .env, "
                    "or set LLM_PROVIDER=ollama in .env.[/yellow]"
                )
            continue

        console.print(f"\n[bold cyan]Agent:[/bold cyan] {response}\n")
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))

        if write_results:
            _print_write_results(write_results)


if __name__ == "__main__":
    main()
