"""
Enforcement layer — quarantine, flag, and admit policies for DriftGuard.

Implements the class-conditional response policy:
  - Class 0 (malicious):        quarantine for human review, keep out of active memory
  - Class 1 (benign drift):     admit but flag (change is auditable)
  - Class 2 (ordinary learning): admit silently, no user-visible friction

Author: Aryaan Sethi (1032231670) — Objective 4
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Class labels
# ─────────────────────────────────────────────────────────────

class MemoryClass(IntEnum):
    MALICIOUS         = 0
    BENIGN_DRIFT      = 1
    ORDINARY_LEARNING = 2


CLASS_NAMES = {
    MemoryClass.MALICIOUS:         "malicious",
    MemoryClass.BENIGN_DRIFT:      "benign_drift",
    MemoryClass.ORDINARY_LEARNING: "ordinary_learning",
}


# ─────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class WriteResult:
    """Result of processing a candidate memory write through DriftGuard."""

    candidate_text: str
    predicted_class: MemoryClass
    class_name: str
    action: str                        # "quarantined" | "admitted_flagged" | "admitted"
    quarantine_id: str | None = None   # set if action == "quarantined"
    flag_reason: str | None = None     # set if action == "admitted_flagged"
    probabilities: list[float] = field(default_factory=list)  # [P(0), P(1), P(2)]
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["predicted_class"] = int(self.predicted_class)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class QuarantineEntry:
    """A quarantined memory candidate pending human review."""

    quarantine_id: str
    candidate_text: str
    probabilities: list[float]
    provenance: dict[str, Any]
    timestamp: datetime
    reviewed: bool = False
    review_decision: str | None = None  # "approve" | "reject"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuarantineEntry":
        d = dict(d)
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


# ─────────────────────────────────────────────────────────────
# Human Review Queue
# ─────────────────────────────────────────────────────────────

class BaseReviewQueue(ABC):
    """
    Interface shared by all quarantine-queue backends ("jsonl" | "memory" |
    "redis" — selectable via the QUARANTINE_BACKEND env var, see
    ``make_queue_from_env``).
    """

    @abstractmethod
    def enqueue(self, entry: QuarantineEntry) -> None: ...

    @abstractmethod
    def list_pending(self) -> list[QuarantineEntry]: ...

    @abstractmethod
    def list_all(self) -> list[QuarantineEntry]: ...

    @abstractmethod
    def mark_reviewed(self, quarantine_id: str, decision: str) -> None: ...

    def __len__(self) -> int:
        return len(self.list_all())


class HumanReviewQueue(BaseReviewQueue):
    """
    Persistent JSONL queue for quarantined memory entries awaiting human review.

    Each line in the JSONL file is one QuarantineEntry (JSON-serialised).
    """

    def __init__(self, path: str | Path = "./data/quarantine_queue.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def enqueue(self, entry: QuarantineEntry) -> None:
        """Append an entry to the queue."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        logger.info(f"[Quarantine] Enqueued entry {entry.quarantine_id[:8]}…")

    def list_pending(self) -> list[QuarantineEntry]:
        """Return all unreviewed entries."""
        entries = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    e = QuarantineEntry.from_dict(json.loads(line))
                    if not e.reviewed:
                        entries.append(e)
        return entries

    def list_all(self) -> list[QuarantineEntry]:
        """Return all entries (reviewed and unreviewed)."""
        entries = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(QuarantineEntry.from_dict(json.loads(line)))
        return entries

    def mark_reviewed(self, quarantine_id: str, decision: str) -> None:
        """
        Mark an entry as reviewed with a decision ('approve' or 'reject').
        Rewrites the entire file (acceptable for research-scale queues).
        """
        entries = self.list_all()
        updated = False
        for e in entries:
            if e.quarantine_id == quarantine_id:
                e.reviewed = True
                e.review_decision = decision
                updated = True
        if not updated:
            raise KeyError(f"Quarantine entry {quarantine_id!r} not found.")
        # Rewrite
        with self.path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e.to_dict()) + "\n")


class InMemoryReviewQueue(BaseReviewQueue):
    """
    Non-persistent queue held in a Python list. Useful for tests and for
    QUARANTINE_BACKEND=memory, where durability across process restarts
    isn't needed.
    """

    def __init__(self) -> None:
        self._entries: list[QuarantineEntry] = []

    def enqueue(self, entry: QuarantineEntry) -> None:
        self._entries.append(entry)
        logger.info(f"[Quarantine] Enqueued entry {entry.quarantine_id[:8]}… (in-memory)")

    def list_pending(self) -> list[QuarantineEntry]:
        return [e for e in self._entries if not e.reviewed]

    def list_all(self) -> list[QuarantineEntry]:
        return list(self._entries)

    def mark_reviewed(self, quarantine_id: str, decision: str) -> None:
        for e in self._entries:
            if e.quarantine_id == quarantine_id:
                e.reviewed = True
                e.review_decision = decision
                return
        raise KeyError(f"Quarantine entry {quarantine_id!r} not found.")


class RedisReviewQueue(BaseReviewQueue):
    """
    Redis-backed quarantine queue (QUARANTINE_BACKEND=redis).

    Entries are stored as JSON strings in a single Redis hash keyed by
    quarantine_id, so lookups and reviewed-status updates are O(1) without
    needing to rewrite the whole queue (unlike the JSONL backend).

    Requires the optional ``redis`` dependency:
        uv sync --extra redis
    """

    def __init__(
        self,
        url: str | None = None,
        hash_key: str = "driftguard:quarantine",
    ) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisReviewQueue requires the 'redis' package. "
                "Install with: uv sync --extra redis"
            ) from exc

        url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._hash_key = hash_key

    def enqueue(self, entry: QuarantineEntry) -> None:
        self._client.hset(self._hash_key, entry.quarantine_id, json.dumps(entry.to_dict()))
        logger.info(f"[Quarantine] Enqueued entry {entry.quarantine_id[:8]}… (redis)")

    def list_pending(self) -> list[QuarantineEntry]:
        return [e for e in self.list_all() if not e.reviewed]

    def list_all(self) -> list[QuarantineEntry]:
        raw = self._client.hgetall(self._hash_key)
        return [QuarantineEntry.from_dict(json.loads(v)) for v in raw.values()]

    def mark_reviewed(self, quarantine_id: str, decision: str) -> None:
        raw = self._client.hget(self._hash_key, quarantine_id)
        if raw is None:
            raise KeyError(f"Quarantine entry {quarantine_id!r} not found.")
        entry = QuarantineEntry.from_dict(json.loads(raw))
        entry.reviewed = True
        entry.review_decision = decision
        self._client.hset(self._hash_key, quarantine_id, json.dumps(entry.to_dict()))


def make_queue_from_env(env: dict[str, str] | None = None) -> BaseReviewQueue:
    """
    Build the quarantine queue backend selected by QUARANTINE_BACKEND
    ("jsonl" | "memory" | "redis", default "jsonl"). See .env.example.
    """
    env = env if env is not None else os.environ
    backend = env.get("QUARANTINE_BACKEND", "jsonl").lower()

    if backend == "jsonl":
        return HumanReviewQueue(path=env.get("QUARANTINE_PATH", "./data/quarantine_queue.jsonl"))
    elif backend == "memory":
        return InMemoryReviewQueue()
    elif backend == "redis":
        return RedisReviewQueue(url=env.get("REDIS_URL"))
    else:
        raise ValueError(
            f"Unknown QUARANTINE_BACKEND={backend!r}. Supported: 'jsonl', 'memory', 'redis'."
        )


# ─────────────────────────────────────────────────────────────
# Enforcement Policy
# ─────────────────────────────────────────────────────────────

class EnforcementPolicy:
    """
    Applies the class-conditional response policy to a classified write.

      Class 0 (malicious)        → quarantine
      Class 1 (benign drift)     → admit + flag
      Class 2 (ordinary learning)→ admit silently

    Parameters
    ----------
    quarantine_queue : BaseReviewQueue
        Queue where malicious writes are parked. Defaults to whatever
        QUARANTINE_BACKEND selects (see ``make_queue_from_env``).
    malicious_threshold : float
        If P(malicious) ≥ threshold, treat as malicious regardless of argmax.
        Default 0.5 provides a balanced policy; raise to reduce false alarms.
    """

    def __init__(
        self,
        quarantine_queue: BaseReviewQueue | None = None,
        malicious_threshold: float = 0.5,
    ) -> None:
        self.queue = quarantine_queue or make_queue_from_env()
        self.malicious_threshold = malicious_threshold

    def apply(
        self,
        candidate_text: str,
        predicted_class: int,
        probabilities: list[float],
        provenance_dict: dict[str, Any],
        latency_ms: float = 0.0,
    ) -> WriteResult:
        """
        Apply enforcement policy and return a WriteResult.

        The caller (WritePipeline) should commit to the vector store only
        if result.action != 'quarantined'.
        """
        mc = MemoryClass(predicted_class)

        # Hard malicious gate — override argmax if P(mal) is above threshold
        if len(probabilities) >= 1 and probabilities[0] >= self.malicious_threshold:
            mc = MemoryClass.MALICIOUS

        if mc == MemoryClass.MALICIOUS:
            qid = str(uuid.uuid4())
            entry = QuarantineEntry(
                quarantine_id=qid,
                candidate_text=candidate_text,
                probabilities=probabilities,
                provenance=provenance_dict,
                timestamp=datetime.now(timezone.utc),
            )
            self.queue.enqueue(entry)
            result = WriteResult(
                candidate_text=candidate_text,
                predicted_class=mc,
                class_name=CLASS_NAMES[mc],
                action="quarantined",
                quarantine_id=qid,
                probabilities=probabilities,
                latency_ms=latency_ms,
            )
            logger.warning(
                f"[DriftGuard] MALICIOUS write quarantined (id={qid[:8]}…) "
                f"P={probabilities}"
            )

        elif mc == MemoryClass.BENIGN_DRIFT:
            reason = (
                f"Benign preference drift detected "
                f"(P_drift={probabilities[1]:.3f} | trust={provenance_dict.get('trust_level', '?')})"
            )
            result = WriteResult(
                candidate_text=candidate_text,
                predicted_class=mc,
                class_name=CLASS_NAMES[mc],
                action="admitted_flagged",
                flag_reason=reason,
                probabilities=probabilities,
                latency_ms=latency_ms,
            )
            logger.info(f"[DriftGuard] BENIGN DRIFT admitted with flag: {reason}")

        else:  # ORDINARY_LEARNING
            result = WriteResult(
                candidate_text=candidate_text,
                predicted_class=mc,
                class_name=CLASS_NAMES[mc],
                action="admitted",
                probabilities=probabilities,
                latency_ms=latency_ms,
            )
            logger.debug(f"[DriftGuard] Ordinary learning admitted silently.")

        return result
