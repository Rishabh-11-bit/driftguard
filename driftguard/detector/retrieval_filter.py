"""
Retrieval-time filtering — wraps MemoryBackend.search() to exclude
quarantined and flagged entries from retrieval results.

Author: Rishabh Gaikwad (1032231674) — Objective 3
"""

from __future__ import annotations

import logging
from typing import Any

from driftguard.agent.memory_backend import MemoryBackend

logger = logging.getLogger(__name__)


class RetrievalFilter:
    """
    Wraps a MemoryBackend's search() to apply retrieval-time filtering.

    At retrieval-time DriftGuard applies a second enforcement pass:
      - Quarantined entries are excluded from results entirely.
      - Flagged (benign-drift) entries are included but annotated.

    Parameters
    ----------
    memory_backend : MemoryBackend
        The underlying memory store.
    exclude_quarantined : bool
        If True (default), quarantined entries are removed from results.
    annotate_flagged : bool
        If True (default), flagged entries have a "_flagged" key set to True.
    """

    def __init__(
        self,
        memory_backend: MemoryBackend,
        exclude_quarantined: bool = True,
        annotate_flagged: bool = True,
    ) -> None:
        self._backend = memory_backend
        self.exclude_quarantined = exclude_quarantined
        self.annotate_flagged = annotate_flagged

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve the top-k memories after applying DriftGuard filters.

        Returns
        -------
        list of entry dicts, each with optional "_flagged" annotation.
        """
        # Over-fetch to compensate for filtered entries
        raw = self._backend.search(query, k=k * 3)

        filtered = []
        for entry in raw:
            meta = entry.get("metadata", {})
            dg_action = meta.get("driftguard_action") or meta.get("driftguard.driftguard_action", "")

            if self.exclude_quarantined and dg_action == "quarantined":
                logger.debug(f"[RetrievalFilter] Skipping quarantined entry {entry.get('id','?')[:8]}…")
                continue

            if self.annotate_flagged and dg_action == "admitted_flagged":
                entry["_flagged"] = True
                entry["_flag_reason"] = meta.get("flag_reason") or meta.get("driftguard.flag_reason", "")

            filtered.append(entry)
            if len(filtered) >= k:
                break

        logger.debug(
            f"[RetrievalFilter] query='{query[:40]}' "
            f"raw={len(raw)} → filtered={len(filtered)}"
        )
        return filtered

    # Proxy all other backend methods
    def add(self, text: str, metadata: dict | None = None) -> str:
        return self._backend.add(text, metadata)

    def delete(self, entry_id: str) -> bool:
        return self._backend.delete(entry_id)

    def list_all(self) -> list[dict[str, Any]]:
        return self._backend.list_all()

    def clear(self) -> None:
        self._backend.clear()

    def __len__(self) -> int:
        return len(self._backend)
