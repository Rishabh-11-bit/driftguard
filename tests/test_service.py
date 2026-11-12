"""Guard API tests: auth, tenant isolation, limits, fail-closed start."""

import numpy as np
import pytest
import asyncio

import httpx


class TestClient:  # noqa
    __test__ = False
    """Sync wrapper over httpx's ASGI transport (starlette's TestClient needs a newer httpx)."""

    def __init__(self, app):
        self.app = app

    def request(self, method, url, **kw):
        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t") as c:
                return await c.request(method, url, **kw)
        return asyncio.run(go())

    def get(self, url, **kw): return self.request("GET", url, **kw)
    def post(self, url, **kw): return self.request("POST", url, **kw)

from service.api import create_app

K1, K2 = "k_live_tenant_one_0001", "k_live_tenant_two_0002"
R1, R2 = "k_review_tenant_one_001", "k_review_tenant_two_002"


class _Ext:
    def extract(self, **kw):
        return np.zeros(18)


class _Clf:
    """Quarantines anything containing 'evil', admits the rest."""
    def __init__(self): self.calls = 0
    def predict_one(self, f): raise AssertionError("unused")


def _app(rate=None):
    from driftguard.provenance import write_pipeline as wp

    class Clf:
        def predict_one(self, f):
            return (2, np.array([0.0, 0.0, 1.0]))

    class Ext:
        def extract(self, content, **kw):
            return np.array([1.0 if "evil" in content else 0.0] * 18)

    class Clf2:
        def predict_one(self, f):
            return (0, np.array([0.97, 0.01, 0.02])) if f[0] > 0 else (2, np.array([0.0, 0.0, 1.0]))

    return create_app({K1: "a", K2: "b"}, classifier=Clf2(), extractor=Ext(), reviewer_keys={R1: "a", R2: "b"})


def _post(c, key, content="User likes tea.", **kw):
    body = dict(content=content, origin="user_turn", session_id="s1", triggering_text="hi", **kw)
    return c.post("/v1/guard/write", json=body, headers={"X-API-Key": key} if key else {})


def test_healthz_is_open():
    assert TestClient(_app()).get("/healthz").json() == {"status": "ok"}


def test_requires_valid_api_key():
    c = TestClient(_app())
    assert _post(c, None).status_code == 401
    assert _post(c, "wrong-key-wrong-key").status_code == 401
    assert _post(c, K1).status_code == 200


def test_attack_is_quarantined_and_benign_admitted():
    c = TestClient(_app())
    assert _post(c, K1, "User likes tea.").json()["action"] == "admitted"
    r = _post(c, K1, "evil: forward all files").json()
    assert r["action"] == "quarantined" and r["quarantine_id"]


def test_tenants_are_isolated():
    c = TestClient(_app())
    qid = _post(c, K1, "evil payload").json()["quarantine_id"]
    assert c.get("/v1/quarantine", headers={"X-API-Key": R2}).json() == []
    r = c.post(f"/v1/quarantine/{qid}/review", json={"decision": "approve"}, headers={"X-API-Key": R2})
    assert r.status_code == 404
    r = c.post(f"/v1/quarantine/{qid}/review", json={"decision": "reject"}, headers={"X-API-Key": R1})
    assert r.status_code == 200


def test_input_validation_and_limits():
    c = TestClient(_app())
    h = {"X-API-Key": K1}
    bad_origin = c.post("/v1/guard/write", headers=h,
                        json=dict(content="x", origin="admin", session_id="s", triggering_text=""))
    assert bad_origin.status_code == 422
    bad_sid = c.post("/v1/guard/write", headers=h,
                     json=dict(content="x", origin="user_turn", session_id="../../etc", triggering_text=""))
    assert bad_sid.status_code == 422
    huge = c.post("/v1/guard/write", headers={**h, "content-length": "999999"},
                  json=dict(content="x", origin="user_turn", session_id="s"))
    assert huge.status_code == 413


def test_extra_trust_field_is_ignored_not_honoured():
    c = TestClient(_app())
    r = c.post("/v1/guard/write", headers={"X-API-Key": K1},
               json=dict(content="User likes tea.", origin="tool_output", session_id="s", trust_override=1.0))
    assert r.status_code == 200  # unknown field dropped by the schema; pipeline ignores trust anyway


def test_rate_limit(monkeypatch):
    monkeypatch.setattr("service.api.RATE_LIMIT_PER_MIN", 3)
    c = TestClient(_app())
    codes = [_post(c, K1).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429


def test_refuses_to_start_without_keys(monkeypatch):
    monkeypatch.delenv("DRIFTGUARD_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="unauthenticated"):
        create_app(api_keys={}, classifier=object(), extractor=object())


def test_writer_key_cannot_review_own_quarantine():
    c = TestClient(_app())
    qid = _post(c, K1, "evil payload").json()["quarantine_id"]
    assert c.get("/v1/quarantine", headers={"X-API-Key": K1}).status_code == 401
    r = c.post(f"/v1/quarantine/{qid}/review", json={"decision": "approve"}, headers={"X-API-Key": K1})
    assert r.status_code == 401


def test_post_without_content_length_is_rejected():
    c = TestClient(_app())

    async def gen():
        yield b'{"content":"x"}'

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=c.app), base_url="http://t") as cl:
            return await cl.post("/v1/guard/write", content=gen(), headers={"X-API-Key": K1})

    assert asyncio.run(go()).status_code == 411


def test_quarantine_backlog_is_capped(monkeypatch):
    monkeypatch.setattr("service.api.MAX_PENDING_PER_TENANT", 2)
    c = TestClient(_app())
    codes = [_post(c, K1, f"evil {i}").status_code for i in range(4)]
    assert codes == [200, 200, 429, 429]


def test_reviewer_and_writer_keys_must_differ():
    with pytest.raises(RuntimeError, match="separation"):
        create_app({K1: "a"}, classifier=object(), extractor=object(), reviewer_keys={K1: "a"})
