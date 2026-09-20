"""
Provenance schema for DriftGuard memory writes.

Every candidate memory write is tagged with a ProvenanceRecord before being
passed to the three-class detector. The record captures:
  - origin:        where the write came from (user, tool, retrieved doc, agent)
  - trust_level:   float 0.0–1.0 reflecting source trustworthiness
  - timestamp:     UTC datetime of the write
  - session_id:    ID of the conversation session
  - interaction_hash: SHA-256 of the triggering interaction text
  - metadata:      arbitrary additional context

Author: Aryaan Sethi (1032231670) — Objective 1
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

Origin = Literal["user_turn", "tool_output", "retrieved_doc", "agent_reflection", "unknown"]

# Default trust levels per origin (can be overridden per-write)
ORIGIN_TRUST_DEFAULTS: dict[str, float] = {
    "user_turn":         0.7,   # users are trusted but can be adversarial
    "tool_output":       0.6,   # tool results may be poisoned via environment
    "retrieved_doc":     0.4,   # external retrieved content is less trusted
    "agent_reflection":  0.9,   # agent's own reasoning is high-trust
    "unknown":           0.2,   # unidentified origin gets low trust
}


# ─────────────────────────────────────────────────────────────
# ProvenanceRecord
# ─────────────────────────────────────────────────────────────

@dataclass
class ProvenanceRecord:
    """
    Immutable provenance tag attached to every candidate memory write.

    Fields
    ------
    origin : Origin
        Categorical source of the write.
    trust_level : float
        Continuous trust score in [0.0, 1.0].
    timestamp : datetime
        UTC time the write was intercepted.
    session_id : str
        Conversation / session identifier.
    interaction_hash : str
        SHA-256 hex digest of the triggering interaction text.
    record_id : str
        Unique ID for this provenance record (UUID4 by default).
    metadata : dict
        Arbitrary extra context (e.g., tool name, retrieval query).
    """

    origin: Origin
    trust_level: float
    timestamp: datetime
    session_id: str
    interaction_hash: str
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── validation ──────────────────────────────────────────
    def __post_init__(self) -> None:
        if not 0.0 <= self.trust_level <= 1.0:
            raise ValueError(f"trust_level must be in [0, 1], got {self.trust_level}")
        if self.origin not in ORIGIN_TRUST_DEFAULTS:
            raise ValueError(f"Unknown origin '{self.origin}'")

    # ── serialisation ────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProvenanceRecord":
        """Deserialise from a dict produced by ``to_dict``."""
        d = dict(d)
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "ProvenanceRecord":
        return cls.from_dict(json.loads(s))

    # ── feature vector helpers ───────────────────────────────
    @property
    def origin_index(self) -> int:
        """Integer encoding of origin for feature extraction."""
        return list(ORIGIN_TRUST_DEFAULTS.keys()).index(self.origin)

    @property
    def session_age_seconds(self) -> float:
        """Seconds since the record was created (useful as a temporal feature)."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


# ─────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────

def make_provenance(
    origin: Origin,
    session_id: str,
    triggering_text: str,
    trust_override: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProvenanceRecord:
    """
    Convenience factory for creating a ProvenanceRecord.

    Parameters
    ----------
    origin:
        Categorical source of the write.
    session_id:
        Conversation session ID.
    triggering_text:
        The raw text of the interaction that produced this write.
        Will be SHA-256 hashed before storage — not persisted in plaintext.
    trust_override:
        If supplied, uses this trust level instead of the origin default.
    metadata:
        Additional context to attach.
    """
    trust = trust_override if trust_override is not None else ORIGIN_TRUST_DEFAULTS[origin]
    interaction_hash = hashlib.sha256(triggering_text.encode("utf-8")).hexdigest()
    return ProvenanceRecord(
        origin=origin,
        trust_level=trust,
        timestamp=datetime.now(timezone.utc),
        session_id=session_id,
        interaction_hash=interaction_hash,
        metadata=metadata or {},
    )
