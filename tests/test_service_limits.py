"""Origin binding, failed-auth throttle, per-session history, and the file-backed review queue."""

import json
import threading
from datetime import datetime, timezone

import numpy as np
import pytest

from driftguard.provenance.enforcement import HumanReviewQueue, QuarantineEntry
from service.api import _parse_keys, create_app
from tests.test_service import K1, R1, TestClient, _post


class _Ext:
    def extract(self, content, **kw):
        return np.array([1.0 if "evil" in content else 0.0] * 18)


class _Clf:
    def predict_one(self, f):
        return (0, np.array([0.97, 0.01, 0.02])) if f[0] > 0 else (2, np.array([0.0, 0.0, 1.0]))


def _app(**kw):
    return create_app({K1: "a"}, classifier=_Clf(), extractor=_Ext(), reviewer_keys={R1: "a"}, **kw)


def test_key_bound_to_origins_cannot_claim_user_turn():
    c = TestClient(_app(key_origins={K1: frozenset({"tool_output", "retrieved_doc"})}))
    body = dict(content="hello", session_id="s", triggering_text="")
    ok = c.post("/v1/guard/write", headers={"X-API-Key": K1}, json={**body, "origin": "tool_output"})
    bad = c.post("/v1/guard/write", headers={"X-API-Key": K1}, json={**body, "origin": "user_turn"})
    assert ok.status_code == 200
    assert bad.status_code == 403


def test_parse_keys_origin_suffix_and_validation():
    origins = {}
    keys = _parse_keys("k_agent_runtime_key_001:acme@tool_output|retrieved_doc", origins)
    assert keys == {"k_agent_runtime_key_001": "acme"}
    assert origins["k_agent_runtime_key_001"] == frozenset({"tool_output", "retrieved_doc"})
    with pytest.raises(RuntimeError, match="unknown origin"):
        _parse_keys("k_agent_runtime_key_001:acme@root")
    with pytest.raises(RuntimeError, match="duplicate"):
        _parse_keys("k_agent_runtime_key_001:a,k_agent_runtime_key_001:b")


def test_failed_auth_flood_is_throttled(monkeypatch):
    monkeypatch.setattr("service.api.AUTH_FAILURES_PER_MIN", 3)
    c = TestClient(_app())
    codes = [_post(c, "wrong-key-wrong-key").status_code for _ in range(5)]
    assert codes[:3] == [401, 401, 401] and codes[3] == 429


def test_write_history_is_per_session():
    from service.api import _Tenant
    t = _Tenant(_Clf(), _Ext())
    a, b = t.pipeline_for("s1"), t.pipeline_for("s2")
    assert a is not b and a is t.pipeline_for("s1")
    assert a._write_history is not b._write_history


def test_session_table_is_bounded(monkeypatch):
    monkeypatch.setattr("service.api.MAX_SESSIONS_PER_TENANT", 3)
    from service.api import _Tenant
    t = _Tenant(_Clf(), _Ext())
    for i in range(10):
        t.pipeline_for(f"s{i}")
    assert len(t._pipelines) == 3


def _entry(i: int) -> QuarantineEntry:
    return QuarantineEntry(
        quarantine_id=f"id-{i}", candidate_text=f"text {i}", probabilities=[0.9, 0.05, 0.05],
        provenance={}, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_file_queue_skips_malformed_lines_and_rewrites_atomically(tmp_path):
    q = HumanReviewQueue(tmp_path / "q.jsonl")
    q.enqueue(_entry(1))
    with (tmp_path / "q.jsonl").open("a", encoding="utf-8") as f:
        f.write("{not json\n" + json.dumps({"unexpected": "shape"}) + "\n")
    q.enqueue(_entry(2))
    assert [e.quarantine_id for e in q.list_pending()] == ["id-1", "id-2"]
    q.mark_reviewed("id-1", "reject")
    assert [e.quarantine_id for e in q.list_pending()] == ["id-2"]
    assert not list(tmp_path.glob("*.tmp"))          # no temp files left behind
    with pytest.raises(KeyError):
        q.mark_reviewed("missing", "approve")


def test_file_queue_concurrent_enqueue_loses_nothing(tmp_path):
    q = HumanReviewQueue(tmp_path / "q.jsonl")
    threads = [threading.Thread(target=lambda i=i: q.enqueue(_entry(i))) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(q.list_all()) == 40
