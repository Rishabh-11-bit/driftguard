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
from collections import OrderedDict, defaultdict, deque
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
MAX_SESSIONS_PER_TENANT = int(os.getenv("DRIFTGUARD_MAX_SESSIONS", "200"))
AUTH_FAILURES_PER_MIN = int(os.getenv("DRIFTGUARD_AUTH_FAILURES", "20"))
ALL_ORIGINS = frozenset({"user_turn", "tool_output", "retrieved_doc", "agent_reflection", "unknown"})
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
    """Isolated memory + quarantine queue, with one write pipeline (and so one write history) per session."""

    def __init__(self, classifier, extractor) -> None:
        self.lock = threading.Lock()
        self.memory = _ListMemory()
        self.queue = InMemoryReviewQueue()
        self._classifier, self._extractor = classifier, extractor
        self._pipelines: OrderedDict[str, WritePipeline] = OrderedDict()

    def pipeline_for(self, session_id: str) -> WritePipeline:
        """History is keyed by session, so one session cannot shape the temporal features of another."""
        pl = self._pipelines.get(session_id)
        if pl is None:
            pl = WritePipeline(self.memory, self._classifier, EnforcementPolicy(quarantine_queue=self.queue))
            if self._extractor is not None:
                pl._extractor = self._extractor
            self._pipelines[session_id] = pl
            while len(self._pipelines) > MAX_SESSIONS_PER_TENANT:
                self._pipelines.popitem(last=False)          # evict the oldest session
        else:
            self._pipelines.move_to_end(session_id)
        return pl


def _parse_keys(raw: str, origins: dict[str, frozenset] | None = None) -> dict[str, str]:
    """Parse '<key>:<tenant>[@origin1|origin2]' entries. The optional origin list limits which `origin`
    values that key may assert (e.g. an agent-runtime key may only send tool_output / retrieved_doc)."""
    keys: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        key, _, rest = pair.partition(":")
        tenant, _, allowed = rest.partition("@")
        if len(key) < 16 or not tenant or ":" in tenant:
            raise RuntimeError("API key entries must look like '<key >=16 chars, no colon>:<tenant>[@origin|origin]'")
        if key in keys:
            raise RuntimeError("duplicate API key in configuration")
        if allowed:
            bad = set(allowed.split("|")) - ALL_ORIGINS
            if bad:
                raise RuntimeError(f"unknown origin(s) in key configuration: {sorted(bad)}")
            if origins is not None:
                origins[key] = frozenset(allowed.split("|"))
        keys[key] = tenant
    return keys


def create_app(
    api_keys: dict[str, str] | None = None,
    classifier=None,
    extractor=None,
    reviewer_keys: dict[str, str] | None = None,
    key_origins: dict[str, frozenset] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    if api_keys is None:
        env_origins: dict[str, frozenset] = {}
        api_keys = _parse_keys(os.getenv("DRIFTGUARD_API_KEYS", ""), env_origins)
        key_origins = key_origins or env_origins
    key_origins = key_origins or {}
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

    auth_failures: dict[str, deque] = defaultdict(deque)

    def _authenticate(request: Request, x_api_key: str | None, index: dict) -> tuple[str, str]:
        now = clock()
        host = request.client.host if request.client else "unknown"
        fq = auth_failures[host]
        while fq and now - fq[0] > 60:
            fq.popleft()
        if len(fq) >= AUTH_FAILURES_PER_MIN:
            raise HTTPException(429, "too many failed authentication attempts")
        if x_api_key:
            entry = index.get(hashlib.sha256(x_api_key.encode()).hexdigest())
            if entry:
                key, tenant = entry
                q = hits[key]
                while q and now - q[0] > 60:
                    q.popleft()
                if len(q) >= RATE_LIMIT_PER_MIN:
                    raise HTTPException(429, "rate limit exceeded")
                q.append(now)
                return tenant, key
        fq.append(now)
        raise HTTPException(401, "invalid or missing API key")

    def writer(request: Request, x_api_key: str | None = Header(default=None)) -> tuple[_Tenant, frozenset]:
        tenant, key = _authenticate(request, x_api_key, writer_idx)
        return tenants[tenant], key_origins.get(key, ALL_ORIGINS)

    def reviewer_state(request: Request, x_api_key: str | None = Header(default=None)) -> _Tenant:
        return tenants[_authenticate(request, x_api_key, reviewer_idx)[0]]

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/guard/write", response_model=WriteResponse)
    def guard_write(req: WriteRequest, ctx: tuple[_Tenant, frozenset] = Depends(writer)) -> WriteResponse:
        t, allowed = ctx
        if req.origin not in allowed:
            raise HTTPException(403, f"this API key may not assert origin {req.origin!r}")
        with t.lock:
            if len(t.queue.list_pending()) >= MAX_PENDING_PER_TENANT:
                raise HTTPException(429, "quarantine backlog full; review pending items")
            if len(t.memory.items) >= MAX_MEMORIES_PER_TENANT:
                raise HTTPException(507, "memory quota reached")
            res = t.pipeline_for(req.session_id).process(CandidateWrite(
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
