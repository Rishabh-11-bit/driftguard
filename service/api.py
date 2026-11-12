"""DriftGuard Guard API — an authenticated HTTP service in front of an agent's memory store.

Run:
    DRIFTGUARD_API_KEYS="k_live_abc:acme,k_live_def:globex" \
    DRIFTGUARD_MODEL_PATH=models/classifier_robust.pkl \
    uvicorn service.api:app --host 127.0.0.1 --port 8080

Security properties (see SECURITY.md):
  * every route except /healthz requires an API key (constant-time compare)
  * keys map to tenants; each tenant gets an isolated memory + quarantine queue
  * per-key sliding-window rate limit, request-size and field-length limits
  * `origin` must be one of the allowed enum values; there is NO trust override field
  * fails closed: missing key config or model refuses to start (no anonymous mode)
  * model file must carry a valid signature (DriftGuardClassifier.load)
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from driftguard.provenance.enforcement import EnforcementPolicy, InMemoryReviewQueue
from driftguard.provenance.write_pipeline import CandidateWrite, WritePipeline
from driftguard.security import MAX_QUERY_CHARS, MAX_WRITE_CHARS

MAX_BODY_BYTES = 16 * 1024
RATE_LIMIT_PER_MIN = int(os.getenv("DRIFTGUARD_RATE_LIMIT", "120"))
MAX_MEMORIES_PER_TENANT = int(os.getenv("DRIFTGUARD_MAX_MEMORIES", "1000"))
MAX_PENDING_PER_TENANT = int(os.getenv("DRIFTGUARD_MAX_PENDING", "500"))
Origin = Literal["user_turn", "tool_output", "retrieved_doc", "agent_reflection", "unknown"]


class WriteRequest(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_WRITE_CHARS * 4)  # sanitiser caps further
    origin: Origin
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    triggering_text: str = Field(default="", max_length=MAX_QUERY_CHARS * 4)


class WriteResponse(BaseModel):
    verdict: Literal["malicious", "benign_drift", "ordinary_learning"]
    action: Literal["quarantined", "admitted_flagged", "admitted"]
    probabilities: list[float]
    quarantine_id: str | None
    latency_ms: float


class ReviewRequest(BaseModel):
    decision: Literal["approve", "reject"]


class MemoryFull(RuntimeError):
    pass


class _ListMemory:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, text: str, metadata: dict | None = None) -> str:
        if len(self.items) >= MAX_MEMORIES_PER_TENANT:
            raise MemoryFull("tenant memory quota reached")
        self.items.append({"text": text, "metadata": metadata or {}})
        return str(len(self.items))

    def list_all(self) -> list[dict]:
        return self.items


class _Tenant:
    def __init__(self, classifier, extractor) -> None:
        self.lock = threading.Lock()
        self.memory = _ListMemory()
        self.queue = InMemoryReviewQueue()
        self.pipeline = WritePipeline(self.memory, classifier, EnforcementPolicy(quarantine_queue=self.queue))
        if extractor is not None:
            self.pipeline._extractor = extractor


def _parse_keys(raw: str) -> dict[str, str]:
    keys: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        key, _, tenant = pair.partition(":")
        if len(key) < 16 or not tenant or ":" in tenant:
            raise RuntimeError("API key entries must look like '<key >=16 chars, no colon>:<tenant>'")
        if key in keys:
            raise RuntimeError("duplicate API key in configuration")
        keys[key] = tenant
    return keys


def create_app(
    api_keys: dict[str, str] | None = None,
    classifier=None,
    extractor=None,
    reviewer_keys: dict[str, str] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    if api_keys is None:
        api_keys = _parse_keys(os.getenv("DRIFTGUARD_API_KEYS", ""))
    if not api_keys:
        raise RuntimeError("No API keys configured (DRIFTGUARD_API_KEYS). Refusing to start unauthenticated.")
    if reviewer_keys is None:
        reviewer_keys = _parse_keys(os.getenv("DRIFTGUARD_REVIEWER_KEYS", ""))
    if set(reviewer_keys) & set(api_keys):
        raise RuntimeError("reviewer keys must differ from writer keys (separation of duties)")
    if classifier is None:
        if not os.getenv("DRIFTGUARD_MODEL_KEY"):
            raise RuntimeError("DRIFTGUARD_MODEL_KEY is required to load a model in the service "
                               "(plain SHA-256 signatures do not stop an attacker who can write the model directory).")
        from driftguard.detector.classifier import DriftGuardClassifier
        classifier = DriftGuardClassifier.load(os.getenv("DRIFTGUARD_MODEL_PATH", "models/classifier_v2.pkl"))
    if extractor is None:
        from driftguard.detector.features import FeatureExtractor, SemanticFeatureExtractor
        n_feat = len(getattr(classifier, "_feature_names", None) or [])
        extractor = SemanticFeatureExtractor() if n_feat > 18 else FeatureExtractor()

    app = FastAPI(title="DriftGuard Guard API", version="0.1.0", docs_url=None, redoc_url=None)
    tenants: dict[str, _Tenant] = {t: _Tenant(classifier, extractor)
                                   for t in set(api_keys.values()) | set(reviewer_keys.values())}
    # keys are looked up by digest: one dict lookup, no per-key compare loop
    writer_idx = {hashlib.sha256(k.encode()).hexdigest(): (k, t) for k, t in api_keys.items()}
    reviewer_idx = {hashlib.sha256(k.encode()).hexdigest(): (k, t) for k, t in reviewer_keys.items()}
    hits: dict[str, deque] = defaultdict(deque)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        from fastapi.responses import JSONResponse
        length = request.headers.get("content-length")
        if request.method in ("POST", "PUT", "PATCH") and length is None:
            return JSONResponse({"detail": "Content-Length required"}, status_code=411)  # no chunked bypass
        if length and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    def _authenticate(x_api_key: str | None, index: dict) -> str:
        if x_api_key:
            entry = index.get(hashlib.sha256(x_api_key.encode()).hexdigest())
            if entry:
                key, tenant = entry
                now = clock()
                q = hits[key]
                while q and now - q[0] > 60:
                    q.popleft()
                if len(q) >= RATE_LIMIT_PER_MIN:
                    raise HTTPException(429, "rate limit exceeded")
                q.append(now)
                return tenant
        raise HTTPException(401, "invalid or missing API key")

    def auth(x_api_key: str | None = Header(default=None)) -> str:
        return _authenticate(x_api_key, writer_idx)

    def reviewer_auth(x_api_key: str | None = Header(default=None)) -> str:
        return _authenticate(x_api_key, reviewer_idx)

    def tenant_state(tenant: str = Depends(auth)) -> _Tenant:
        return tenants[tenant]

    def reviewer_state(tenant: str = Depends(reviewer_auth)) -> _Tenant:
        return tenants[tenant]

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/guard/write", response_model=WriteResponse)
    def guard_write(req: WriteRequest, t: _Tenant = Depends(tenant_state)) -> WriteResponse:
        with t.lock:
            if len(t.queue.list_pending()) >= MAX_PENDING_PER_TENANT:
                raise HTTPException(429, "quarantine backlog full; review pending items")
            if len(t.memory.items) >= MAX_MEMORIES_PER_TENANT:
                raise HTTPException(507, "memory quota reached")
            res = t.pipeline.process(CandidateWrite(
                content=req.content, origin=req.origin, session_id=req.session_id,
                triggering_text=req.triggering_text,
            ))
        return WriteResponse(
            verdict=res.class_name, action=res.action, probabilities=list(res.probabilities),
            quarantine_id=res.quarantine_id, latency_ms=res.latency_ms,
        )

    @app.get("/v1/quarantine")
    def list_quarantine(t: _Tenant = Depends(reviewer_state)) -> list[dict]:
        return [{"quarantine_id": e.quarantine_id, "text": e.candidate_text,
                 "p_malicious": e.probabilities[0]} for e in t.queue.list_pending()]

    @app.post("/v1/quarantine/{qid}/review")
    def review(qid: str, body: ReviewRequest, t: _Tenant = Depends(reviewer_state)) -> dict:
        with t.lock:
            pending = {e.quarantine_id: e for e in t.queue.list_pending()}
            if qid not in pending:  # also prevents cross-tenant access: queues are per tenant
                raise HTTPException(404, "not found")
            if body.decision == "approve":
                try:
                    t.memory.add(pending[qid].candidate_text, {"reviewed": True})
                except MemoryFull:
                    raise HTTPException(507, "memory quota reached")
            t.queue.mark_reviewed(qid, body.decision)
            return {"quarantine_id": qid, "decision": body.decision}

    return app


def _lazy_app() -> FastAPI:  # `uvicorn service.api:app` builds from the environment
    return create_app()


app = None  # populated below only when keys are configured, so importing the module is safe
if os.getenv("DRIFTGUARD_API_KEYS"):
    app = _lazy_app()
