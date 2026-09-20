"""Tests for the quarantine queue backends and QUARANTINE_BACKEND selection."""
import uuid
from datetime import datetime, timezone

import pytest

from driftguard.provenance.enforcement import (
    HumanReviewQueue,
    InMemoryReviewQueue,
    QuarantineEntry,
    make_queue_from_env,
)


def _entry(text: str = "malicious content") -> QuarantineEntry:
    return QuarantineEntry(
        quarantine_id=str(uuid.uuid4()),
        candidate_text=text,
        probabilities=[0.9, 0.05, 0.05],
        provenance={"origin": "user_turn", "trust_level": 0.7},
        timestamp=datetime.now(timezone.utc),
    )


class TestHumanReviewQueue:
    def test_enqueue_and_list_pending(self, tmp_path):
        queue = HumanReviewQueue(path=tmp_path / "q.jsonl")
        entry = _entry()
        queue.enqueue(entry)
        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0].quarantine_id == entry.quarantine_id

    def test_mark_reviewed_removes_from_pending(self, tmp_path):
        queue = HumanReviewQueue(path=tmp_path / "q.jsonl")
        entry = _entry()
        queue.enqueue(entry)
        queue.mark_reviewed(entry.quarantine_id, "approve")
        assert queue.list_pending() == []
        assert len(queue) == 1
        assert queue.list_all()[0].review_decision == "approve"

    def test_mark_reviewed_unknown_id_raises(self, tmp_path):
        queue = HumanReviewQueue(path=tmp_path / "q.jsonl")
        with pytest.raises(KeyError):
            queue.mark_reviewed("does-not-exist", "approve")


class TestInMemoryReviewQueue:
    def test_enqueue_and_list_pending(self):
        queue = InMemoryReviewQueue()
        entry = _entry()
        queue.enqueue(entry)
        assert len(queue) == 1
        assert queue.list_pending()[0].quarantine_id == entry.quarantine_id

    def test_mark_reviewed(self):
        queue = InMemoryReviewQueue()
        entry = _entry()
        queue.enqueue(entry)
        queue.mark_reviewed(entry.quarantine_id, "reject")
        assert queue.list_pending() == []
        assert queue.list_all()[0].review_decision == "reject"


class TestRedisReviewQueue:
    """
    Integration test against a real Redis instance (docker-compose up redis).
    Skipped automatically if Redis isn't reachable on REDIS_URL / localhost:6379.
    """

    @pytest.fixture
    def queue(self):
        from driftguard.provenance.enforcement import RedisReviewQueue

        try:
            q = RedisReviewQueue(hash_key="driftguard:test:quarantine")
            q._client.ping()
        except Exception:
            pytest.skip("Redis is not reachable on REDIS_URL / localhost:6379")
        yield q
        q._client.delete(q._hash_key)

    def test_enqueue_and_list_pending(self, queue):
        entry = _entry()
        queue.enqueue(entry)
        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0].quarantine_id == entry.quarantine_id

    def test_mark_reviewed_persists_in_redis(self, queue):
        entry = _entry()
        queue.enqueue(entry)
        queue.mark_reviewed(entry.quarantine_id, "reject")

        assert queue.list_pending() == []
        assert queue.list_all()[0].review_decision == "reject"

        # Re-read via a fresh queue instance to prove it's server-side state, not client-cached.
        from driftguard.provenance.enforcement import RedisReviewQueue
        reread = RedisReviewQueue(hash_key="driftguard:test:quarantine")
        assert reread.list_all()[0].review_decision == "reject"

    def test_mark_reviewed_unknown_id_raises(self, queue):
        with pytest.raises(KeyError):
            queue.mark_reviewed("does-not-exist", "approve")

    def test_make_queue_from_env_selects_redis(self, queue):
        selected = make_queue_from_env({"QUARANTINE_BACKEND": "redis"})
        assert type(selected).__name__ == "RedisReviewQueue"


class TestMakeQueueFromEnv:
    def test_defaults_to_jsonl(self, tmp_path):
        path = tmp_path / "q.jsonl"
        queue = make_queue_from_env({"QUARANTINE_PATH": str(path)})
        assert isinstance(queue, HumanReviewQueue)
        assert queue.path == path

    def test_explicit_memory_backend(self):
        queue = make_queue_from_env({"QUARANTINE_BACKEND": "memory"})
        assert isinstance(queue, InMemoryReviewQueue)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            make_queue_from_env({"QUARANTINE_BACKEND": "carrier_pigeon"})
