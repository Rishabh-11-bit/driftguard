"""
Benchmark labelling schema and serialisation.

Defines the BenchmarkEntry dataclass and JSONL-based persistence for the
three-class DriftGuard benchmark.

Classes:
  0 — malicious          (MINJA-style or AgentPoison-style injections)
  1 — benign_drift       (legitimate user preference changes)
  2 — ordinary_learning  (normal fact/preference writes)

Author: Tanisha Talathi (1032231612) — Objective 2
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Label = Literal["malicious", "benign_drift", "ordinary_learning"]
LABEL_TO_INT: dict[str, int] = {
    "malicious":         0,
    "benign_drift":      1,
    "ordinary_learning": 2,
}
INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass
class BenchmarkEntry:
    """
    A single entry in the three-class DriftGuard benchmark.

    Attributes
    ----------
    content : str
        The memory write text to classify.
    label : Label
        Ground-truth class.
    label_int : int
        Integer encoding of label (0, 1, 2).
    provenance : dict
        Serialised ProvenanceRecord.
    attack_family : str
        For malicious entries: "minja", "agentpoison", "slow_drip",
        "indirect_tool_injection", etc.
        For benign entries: "preference_change", "fact_correction", etc.
    session_id : str
        Synthetic session ID.
    last_query : str
        Synthetic triggering query (for causal attribution features).
    memory_entries : list[str]
        Simulated existing user memories at write time (for consistency features).
    write_history : list[dict]
        Simulated recent write history (for temporal features).
    entry_id : str
        Unique benchmark entry ID.
    metadata : dict
        Additional generation metadata (e.g., template index, drift magnitude).
    created_at : datetime
        When this benchmark entry was generated.
    """

    content: str
    label: Label
    label_int: int
    provenance: dict[str, Any]
    attack_family: str
    session_id: str
    last_query: str
    memory_entries: list[str] = field(default_factory=list)
    write_history: list[dict] = field(default_factory=list)
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchmarkEntry":
        d = dict(d)
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**d)


# ─────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────

def save_benchmark(entries: list[BenchmarkEntry], path: str | Path) -> None:
    """Save benchmark to a JSONL file (one entry per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")
    print(f"[Benchmark] Saved {len(entries)} entries to {path}.")


def load_benchmark(path: str | Path) -> list[BenchmarkEntry]:
    """Load benchmark from a JSONL file."""
    path = Path(path)
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(BenchmarkEntry.from_dict(json.loads(line)))
    return entries


def benchmark_stats(entries: list[BenchmarkEntry]) -> dict[str, Any]:
    """Return a summary dict of class counts and attack family breakdown."""
    from collections import Counter
    label_counts = Counter(e.label for e in entries)
    family_counts = Counter(e.attack_family for e in entries)
    return {
        "total":         len(entries),
        "class_counts":  dict(label_counts),
        "attack_families": dict(family_counts),
    }
