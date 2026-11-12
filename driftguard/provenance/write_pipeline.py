"""
Write pipeline — intercepts every memory write, attaches provenance,
classifies with DriftGuard detector, and applies the enforcement policy.

This is the central integration point that Aryaan owns. It sits between
the LangGraph agent's 'extract_writes' node and the vector store's add()
method.

Author: Aryaan Sethi (1032231670) — Objective 1 & 4
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .enforcement import EnforcementPolicy, HumanReviewQueue, WriteResult
from driftguard.security import MAX_QUERY_CHARS, MAX_WRITE_CHARS, normalise_text
from .schema import ProvenanceRecord, make_provenance, Origin

if TYPE_CHECKING:
    from driftguard.detector.classifier import DriftGuardClassifier

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Candidate write
# ─────────────────────────────────────────────────────────────

@dataclass
class CandidateWrite:
    """
    A memory write candidate before it has been classified and committed.

    Attributes
    ----------
    content : str
        The text to be written to long-term memory.
    origin : Origin
        Categorical source (user_turn, tool_output, etc.)
    session_id : str
        The conversation session that produced this write.
    triggering_text : str
        The full interaction text that triggered this write (used for hashing).
    trust_override : float | None
        If provided, overrides the default trust for this origin.
    metadata : dict
        Any extra context (e.g., tool name, retrieved document id).
    """

    content: str
    origin: Origin
    session_id: str
    triggering_text: str
    trust_override: float | None = None
    metadata: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────
# Write pipeline
# ─────────────────────────────────────────────────────────────

class WritePipeline:
    """
    Intercepts candidate memory writes and routes them through DriftGuard.

    Usage
    -----
    pipeline = WritePipeline(
        memory_backend=backend,
        classifier=detector,
        enforcement=EnforcementPolicy(HumanReviewQueue()),
    )

    # Instead of backend.add(text, metadata), call:
    result = pipeline.process(CandidateWrite(content=text, ...))
    # Commit happens automatically if not quarantined.

    Parameters
    ----------
    memory_backend : any object with add(text, metadata) and search(query, k)
        The underlying vector store wrapper.
    classifier : DriftGuardClassifier
        The three-class detector. If None, uses a pass-through that labels
        everything as ORDINARY_LEARNING (useful for testing without a trained model).
    enforcement : EnforcementPolicy
        Enforcement policy to apply after classification.
    write_history : list
        Internal buffer of recent write provenance records for temporal features.
    history_window : int
        Number of recent writes to keep for temporal feature computation.
    """

    def __init__(
        self,
        memory_backend: Any,
        classifier: "DriftGuardClassifier | None" = None,
        enforcement: EnforcementPolicy | None = None,
        history_window: int = 50,
        allow_trust_override: bool = False,
        fail_closed: bool = True,
    ) -> None:
        # allow_trust_override: honour CandidateWrite.trust_override. Off by default -
        #   trust must come from the runtime's origin table, never from content-derived input.
        # fail_closed: if the detector raises, quarantine instead of admitting.
        self.allow_trust_override = allow_trust_override
        self.fail_closed = fail_closed
        self.memory_backend = memory_backend
        self.classifier = classifier
        self.enforcement = enforcement or EnforcementPolicy(HumanReviewQueue())
        self.history_window = history_window
        self._write_history: list[dict[str, Any]] = []
        self._extractor: Any = None  # built once; loading the encoder is slow
        self.last_features: Any = None  # feature vector of the most recent write

    # ── public API ───────────────────────────────────────────

    def process(self, candidate: CandidateWrite) -> WriteResult:
        """
        Process a candidate write:
        1. Attach provenance.
        2. Extract features.
        3. Classify.
        4. Apply enforcement policy.
        5. Commit to memory if not quarantined.

        Returns
        -------
        WriteResult
            Contains action taken, class, quarantine ID (if any), and latency.
        """
        t0 = time.perf_counter()

        # Step 0 — Sanitise untrusted text (invisible chars, NFKC, length cap)
        candidate = CandidateWrite(
            content=normalise_text(candidate.content, MAX_WRITE_CHARS),
            origin=candidate.origin,
            session_id=candidate.session_id,
            triggering_text=normalise_text(candidate.triggering_text or "", MAX_QUERY_CHARS),
            trust_override=candidate.trust_override,
            metadata=candidate.metadata,
        )

        # Step 1 — Provenance tagging
        if candidate.trust_override is not None and not self.allow_trust_override:
            logger.warning("[WritePipeline] trust_override ignored (allow_trust_override=False).")
        prov = make_provenance(
            origin=candidate.origin,
            session_id=candidate.session_id,
            triggering_text=candidate.triggering_text,
            trust_override=candidate.trust_override if self.allow_trust_override else None,
            metadata=candidate.metadata or {},
        )

        # Step 2 — Feature extraction + classification
        predicted_class, probabilities = self._classify(candidate, prov)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Step 3 — Enforcement policy
        result = self.enforcement.apply(
            candidate_text=candidate.content,
            predicted_class=predicted_class,
            probabilities=probabilities,
            provenance_dict=prov.to_dict(),
            latency_ms=latency_ms,
        )

        # Step 4 — Commit to memory if not quarantined
        if result.action != "quarantined":
            self._commit(candidate, prov, result)

        # Step 5 — Update write history for temporal features
        self._update_history(prov, predicted_class)

        logger.info(
            f"[WritePipeline] session={candidate.session_id[:8]} "
            f"class={result.class_name} action={result.action} "
            f"latency={latency_ms:.1f}ms"
        )
        return result

    def inject_history(self, records: list[dict[str, Any]]) -> None:
        """Prepend synthetic prior writes (used by demos to simulate bursts)."""
        self._write_history = (list(records) + self._write_history)[-self.history_window:]

    def get_write_history(self) -> list[dict[str, Any]]:
        """Return the recent write history buffer."""
        return list(self._write_history)

    # ── internals ────────────────────────────────────────────

    def _classify(
        self,
        candidate: CandidateWrite,
        prov: ProvenanceRecord,
    ) -> tuple[int, list[float]]:
        """Run the detector and return (predicted_class, probabilities)."""
        if self.classifier is None:
            # Pass-through: label everything as ordinary learning
            return 2, [0.05, 0.05, 0.90]

        try:
            if self._extractor is None:
                from driftguard.detector.features import FeatureExtractor, SemanticFeatureExtractor
                # v2 models carry the 7 semantic-anchor features; pick the extractor that matches
                n_feat = len(getattr(self.classifier, "_feature_names", None) or [])
                self._extractor = SemanticFeatureExtractor() if n_feat > 18 else FeatureExtractor()
            # The detector is trained with the triggering query and the user's
            # existing memories; omitting them here zeroes three features.
            features = self._extractor.extract(
                content=candidate.content,
                provenance=prov,
                write_history=self._write_history,
                session_id=candidate.session_id,
                last_query=candidate.triggering_text,
                user_memory_entries=self._existing_memory_texts(),
            )
            self.last_features = features
            pred, proba = self.classifier.predict_one(features)
            return int(pred), proba.tolist()
        except Exception as exc:
            if self.fail_closed:
                logger.error(f"[WritePipeline] Classifier error: {exc}. Failing closed (quarantine).")
                return 0, [1.0, 0.0, 0.0]
            logger.error(f"[WritePipeline] Classifier error: {exc}. Defaulting to ordinary.")
            return 2, [0.0, 0.0, 1.0]

    def _existing_memory_texts(self) -> list[str]:
        """Texts currently in the memory store (empty if the backend can't list)."""
        try:
            return [e["text"] for e in self.memory_backend.list_all()]
        except Exception as exc:
            # Features that compare against existing memories degrade to zero; make that visible
            logger.warning("[WritePipeline] could not list existing memories (%s); novelty features zeroed", exc)
            return []

    def _commit(
        self,
        candidate: CandidateWrite,
        prov: ProvenanceRecord,
        result: WriteResult,
    ) -> None:
        """Commit an approved write to the vector store with provenance metadata."""
        # Caller metadata goes first so it can never overwrite the verdict fields below
        # (e.g. an attacker-supplied {"flagged": False} must not hide a flagged write).
        meta = {
            **(candidate.metadata or {}),
            "provenance": prov.to_dict(),
            "driftguard_class": result.class_name,
            "driftguard_action": result.action,
            "flagged": result.action == "admitted_flagged",
            "flag_reason": result.flag_reason,
        }
        self.memory_backend.add(candidate.content, meta)
        logger.debug(f"[WritePipeline] Committed to memory store.")

    def _update_history(self, prov: ProvenanceRecord, predicted_class: int) -> None:
        record = {
            "timestamp": prov.timestamp.isoformat(),
            "origin": prov.origin,
            "trust_level": prov.trust_level,
            "session_id": prov.session_id,
            "interaction_hash": prov.interaction_hash,
            "predicted_class": predicted_class,
        }
        self._write_history.append(record)
        if len(self._write_history) > self.history_window:
            self._write_history.pop(0)
