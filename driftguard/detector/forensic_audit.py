"""
Forensic audit — periodic post-hoc re-classification of committed memory entries.

The forensic audit re-scores all entries currently in the memory store using
the latest trained classifier. Entries that were originally admitted but are
now classified as malicious are flagged for human review and optionally removed.

Author: Rishabh Gaikwad (1032231674) — Objective 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from driftguard.detector.classifier import DriftGuardClassifier, CLASS_NAMES
from driftguard.detector.features import FeatureExtractor
from driftguard.provenance.schema import ProvenanceRecord

if TYPE_CHECKING:
    # Type-hint only: avoid a hard runtime dependency on driftguard.agent
    # (and therefore on langgraph/langchain) just to run a forensic audit.
    from driftguard.agent.memory_backend import MemoryBackend

logger = logging.getLogger(__name__)


@dataclass
class AuditFinding:
    """A single finding from the forensic audit."""
    entry_id: str
    entry_text: str
    original_class: str
    reclassified_as: str
    probabilities: list[float]
    action_taken: str   # "reclassified_flagged" | "reclassified_quarantined" | "no_change"
    timestamp: datetime


@dataclass
class AuditReport:
    """Summary of a forensic audit pass."""
    total_entries: int
    reclassified_count: int
    newly_malicious: int
    newly_drift: int
    findings: list[AuditFinding]
    run_at: datetime

    def summary(self) -> str:
        return (
            f"Forensic Audit @ {self.run_at.isoformat()}\n"
            f"  Total entries:      {self.total_entries}\n"
            f"  Reclassified:       {self.reclassified_count}\n"
            f"  Newly malicious:    {self.newly_malicious}\n"
            f"  Newly drift:        {self.newly_drift}\n"
        )


class ForensicAuditor:
    """
    Post-hoc forensic audit over committed memory entries.

    Parameters
    ----------
    memory_backend : MemoryBackend
        The memory store to audit.
    classifier : DriftGuardClassifier
        The trained three-class detector.
    extractor : FeatureExtractor | None
        Feature extractor (created automatically if None).
    quarantine_newly_malicious : bool
        If True, entries reclassified as malicious are deleted from active
        memory and logged for review.
    """

    def __init__(
        self,
        memory_backend: MemoryBackend,
        classifier: DriftGuardClassifier,
        extractor: FeatureExtractor | None = None,
        quarantine_newly_malicious: bool = True,
    ) -> None:
        self.memory_backend = memory_backend
        self.classifier = classifier
        self.extractor = extractor or FeatureExtractor()
        self.quarantine_newly_malicious = quarantine_newly_malicious

    def run(self) -> AuditReport:
        """
        Run a full forensic audit over all committed memory entries.

        Returns
        -------
        AuditReport
        """
        logger.info("[ForensicAudit] Starting forensic audit…")
        all_entries = self.memory_backend.list_all()
        findings: list[AuditFinding] = []
        reclassified = 0
        newly_malicious = 0
        newly_drift = 0

        for entry in all_entries:
            finding = self._audit_entry(entry)
            if finding is not None:
                findings.append(finding)
                reclassified += 1
                if finding.reclassified_as == "malicious":
                    newly_malicious += 1
                elif finding.reclassified_as == "benign_drift":
                    newly_drift += 1

        report = AuditReport(
            total_entries=len(all_entries),
            reclassified_count=reclassified,
            newly_malicious=newly_malicious,
            newly_drift=newly_drift,
            findings=findings,
            run_at=datetime.now(timezone.utc),
        )
        logger.info(f"[ForensicAudit] {report.summary()}")
        return report

    def _audit_entry(self, entry: dict[str, Any]) -> AuditFinding | None:
        """Audit a single entry. Returns a finding if reclassification occurred."""
        entry_id   = entry.get("id", "unknown")
        entry_text = entry.get("text", "")
        metadata   = entry.get("metadata", {})

        original_class = metadata.get("driftguard_class") or metadata.get(
            "driftguard.driftguard_class", "unknown"
        )

        # Reconstruct provenance
        prov_dict = metadata.get("provenance") or {}
        if not prov_dict:
            # Flatten chroma metadata back
            prov_dict = {
                k.replace("provenance.", ""): v
                for k, v in metadata.items()
                if k.startswith("provenance.")
            }

        if not prov_dict:
            logger.debug(f"[ForensicAudit] Skipping entry {entry_id[:8]} — no provenance.")
            return None

        try:
            prov = ProvenanceRecord.from_dict(prov_dict)
        except Exception as e:
            logger.warning(f"[ForensicAudit] Could not parse provenance for {entry_id[:8]}: {e}")
            return None

        # Extract features and re-classify
        features = self.extractor.extract(
            content=entry_text,
            provenance=prov,
            write_history=[],  # historical context not available at audit time
            session_id=prov.session_id,
        )
        new_class_int, probas = self.classifier.predict_one(features)
        new_class_name = CLASS_NAMES[new_class_int]

        if new_class_name == original_class:
            return None  # No change

        # Reclassification event
        action = "reclassified_flagged"
        if new_class_name == "malicious" and self.quarantine_newly_malicious:
            self.memory_backend.delete(entry_id)
            action = "reclassified_quarantined"
            logger.warning(
                f"[ForensicAudit] Entry {entry_id[:8]} reclassified as MALICIOUS "
                f"and removed from active memory."
            )
        else:
            logger.info(
                f"[ForensicAudit] Entry {entry_id[:8]}: "
                f"{original_class!r} → {new_class_name!r} (P={probas.tolist()})"
            )

        return AuditFinding(
            entry_id=entry_id,
            entry_text=entry_text,
            original_class=original_class,
            reclassified_as=new_class_name,
            probabilities=probas.tolist(),
            action_taken=action,
            timestamp=datetime.now(timezone.utc),
        )
