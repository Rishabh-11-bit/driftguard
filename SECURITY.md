# DriftGuard — Security Audit & Threat Model

Scope: this repository at the commit that introduced this file. Method: manual code review
of every file-, network-, deserialisation- and prompt-facing code path, plus measured
adaptive-attacker experiments (`scripts/run_adversarial.py`, `scripts/run_transform_holdout.py`).
This is an engineering audit, **not** a third-party penetration test or a compliance certification.

## 1. What is being protected

| Asset | Why it matters |
|---|---|
| The agent's long-term memory | Whatever is stored is later injected into the LLM prompt, so a poisoned entry is persistent prompt injection. |
| User personal data in memory and in the quarantine queue | Preferences, health, location, e-mail habits (PII / sensitive). |
| The detector model file | A tampered model silently disables protection — and a pickle can run code. |
| Provenance labels (`origin`, `trust_level`) | The detector's strongest signals; spoofing them is evasion. |

## 2. Attacker model

1. **Remote content attacker** — controls a web page, document, e-mail or tool result the agent reads (indirect prompt injection). Cannot touch the host.
2. **Conversation attacker** — talks to the agent directly and tries to plant or alter memories.
3. **Adaptive attacker** — knows DriftGuard exists and crafts writes to evade it (white-box on features, black-box on the model).
4. **Local/supply-chain attacker** — can replace files on disk (model, queue, index) or poison a dependency.

## 3. Entry points (attack surface)

| # | Entry point | Where | Attacker | Status |
|---|---|---|---|---|
| E1 | Candidate memory text | `WritePipeline.process` | 1, 2, 3 | **Mitigated**: sanitiser (NFKC, invisible chars, length cap); classifier; quarantine. Residual: see §5. |
| E2 | `origin` / `trust_override` fields | `CandidateWrite` | 2, 3 | **Fixed**: `trust_override` ignored unless the operator sets `allow_trust_override=True`. **Open**: `origin` is still caller-supplied (see F-03). |
| E3 | Free-form `metadata` dict | `_commit` | 2 | **Fixed**: caller metadata could overwrite `flagged`/`driftguard_class`; now cannot. |
| E4 | Model file load (`pickle`) | `DriftGuardClassifier.load` | 4 | **Fixed**: refuses unsigned / modified files (SHA-256, or HMAC with `DRIFTGUARD_MODEL_KEY`). Tested with a real malicious pickle. |
| E5 | Classifier failure path | `WritePipeline._classify` | 1, 2 | **Fixed**: used to *fail open* (admit as ordinary); now quarantines (`fail_closed=True`). |
| E6 | Quarantine queue file / Redis | `enforcement.py` | 4 | **Hardened**: owner-only file mode; warning for non-TLS remote Redis. **Open**: no encryption at rest (F-06). |
| E7 | Demo UI rendering | `app/streamlit_app.py` | 1, 2 | **Fixed**: attacker-controlled text rendered with `st.text`, not markdown (markdown images would let a memory entry beacon to an attacker URL). Server bound to `127.0.0.1`, XSRF on. |
| E8 | Agent write extraction | `agent/graph.py` | 1 | **Open**: any line starting `MEMORY_WRITE:` in LLM output becomes a candidate; every candidate is labelled `agent_reflection` regardless of true source (F-03). |
| E9 | Environment / secrets | `.env`, `llm_interface.py` | 4 | OK: `.env` is git-ignored; no secrets in git history (searched for `sk-` patterns); keys read from env only. |
| E10 | Local LLM endpoint | `app/local_llm.py` | 4 | Low: `OLLAMA_BASE_URL` from env; localhost default; no auth on Ollama itself — do not expose port 11434. |

## 4. Findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| F-01 | **High** | `pickle.load` on a user-supplied path = arbitrary code execution (CWE-502). | Fixed — signature required; regression test `test_malicious_pickle_does_not_execute`. |
| F-02 | **High** | Classifier exceptions admitted the write (fail-open). An attacker who can crash feature extraction (e.g. malformed input) bypasses detection. | Fixed — fail closed by default; test added. |
| F-03 | **High** | **Provenance laundering.** The LangGraph agent tags every write `agent_reflection`, so the origin signal does not reflect whether the content really came from the user, a tool, or a retrieved document. Measured effect is small *for the current model* (§6) but the architecture cannot defend against a smarter attacker until real source tracking exists. | **Open** — requires taint tracking in the agent runtime (§7 R1). |
| F-04 | Medium | Caller-controlled `metadata` was merged last and could overwrite `flagged` / `driftguard_class`, hiding a flagged write from the prompt filter. | Fixed + test. |
| F-05 | Medium | Caller-controlled `trust_override` let any caller claim full trust. | Fixed (off by default) + test. |
| F-06 | Medium | Quarantined text (may be PII) stored in plaintext JSONL/Redis, no retention limit. | Open — encrypt at rest, add TTL (§7 R4). |
| F-07 | Medium | Unbounded input size → embedding/CPU exhaustion (DoS). | Fixed — 2 000-char cap on content and query. |
| F-08 | Medium | Markdown rendering of untrusted text in the UI → data-exfiltration beacon via image URL. | Fixed. |
| F-09 | Low | Invisible/full-width Unicode in writes. | Mitigated by NFKC + stripping; cross-script homoglyphs only partly (§5). |
| F-10 | Low | No authentication/rate limiting on the demo app (single-user local tool). | Accepted for demo; required before any hosted deployment (§7 R2). |
| F-11 | Info | No dependency vulnerability scan was run (would require installing a scanner / network). | Open — run `pip-audit` in CI (§7 R5). |

## 5. Residual risks (measured, not assumed)

Adaptive-attacker results on the hardened benchmark (`results_security/adversarial.md`),
attack-detection rate = share of malicious writes still quarantined:

| Evasion | Standard model | Robust model (adversarially trained) |
|---|---|---|
| Single techniques (zero-width, full-width, homoglyph, synonyms, origin spoof, history camouflage) | 0.97 – 1.00 | 1.00 |
| Benign padding | 0.82 | 1.00 |
| **All combined (adaptive attacker)** | **0.33** | **1.00** |

Leave-one-evasion-out (train without the technique, test on it): 0.97 – 1.00 for every technique.

**Honest limits.**
* The "robust" result is against *our* family of evasions, on *synthetic* data. A novel evasion (e.g. LLM-paraphrased attacks, multi-turn slow drip with benign interleaving) is not covered; the leave-one-evasion-out test is evidence of some transfer, not a guarantee.
* Cross-script homoglyphs (Cyrillic "а") are **not** folded by NFKC. Detection survived in our tests because other features carried the signal, not because the text was normalised.
* Held-out attack families scored 1.00 everywhere, which says the held-out set is easy relative to the in-distribution set — treat it as weak evidence of generalisation (see the paper's limitations).
* All data is synthetic; no evaluation on real poisoned memory traces exists yet.

## 6. Mapping to the OWASP Top 10 for LLM Applications

| OWASP LLM | Relevance | Control |
|---|---|---|
| LLM01 Prompt injection | Memory is a persistent injection channel | Write-time detector + quarantine |
| LLM02 Sensitive information disclosure | Memory/queue hold PII | Owner-only queue file; plain-text rendering; R4 open |
| LLM03 Supply chain | Pickled model, dependencies | Signed model; R5 open |
| LLM04 Data & model poisoning | Memory poisoning is the core threat | Core of the project |
| LLM05 Improper output handling | Agent output rendered in UI | `st.text` |
| LLM06 Excessive agency | Agent writes its own memory | Human review for quarantined writes |
| LLM10 Unbounded consumption | Oversized writes | Length caps |

## 7. Recommendations before any production use

* **R1 — Real provenance.** Attach a taint label at ingestion (user message vs tool result vs retrieved chunk) and carry it through the LLM call so the pipeline knows the *true* source of a MEMORY_WRITE; never let the model or content assert its own origin.
* **R2 — Service boundary.** Put the guard behind an authenticated API (API keys per tenant, rate limits, request-size limits, audit log). Do not expose Streamlit or Ollama.
* **R3 — Defence in depth.** Treat the detector as one layer: also filter memories at *retrieval* time (`detector/retrieval_filter.py`), cap what memory can instruct (no tool-call authority from memory), and keep a human-review path.
* **R4 — Privacy.** Encrypt the queue at rest, add retention/TTL, redact PII in logs, support per-user deletion (GDPR/DPDP "right to erasure" applies to memory stores).
* **R5 — Supply chain.** Pin dependencies, run `pip-audit`/Dependabot in CI, sign model releases with `DRIFTGUARD_MODEL_KEY`, prefer a non-pickle format (e.g. skops/ONNX) long-term.
* **R6 — Continuous red-teaming.** Re-run `scripts/run_adversarial.py` on every model change; add new evasions as they are discovered; track false-alarm rate per release.

## 8. Reporting a vulnerability

Open a private security advisory on the GitHub repository, or contact the maintainers (see README).
Do not include real personal data in reports.

## Addendum: findings from the research audit

- **F-12 Wall-clock label leak (fixed).** `session_age_hours` used `now - record.timestamp`; because the benchmark is generated class by class it identified the class (AUROC 0.94 on random labels), and burst-window features were zero in training. All time features are now relative to the write's own timestamp; regression test in `tests/test_detector.py`. All models retrained.
- **F-03 update: provenance laundering.** The original agent labelled every write `agent_reflection`, the highest default trust (0.9). `driftguard/provenance/attribution.py` now attributes a write to the least-trusted source it is grounded in (token containment >= 0.5). This is heuristic: paraphrased injected content can still be attributed to the user.
- **Adaptive attackers.** Scripted evasions: combined evasion drops the standard model to 0.23 detection; the adversarially trained model holds 1.00 in-distribution and 0.90-1.00 leave-one-evasion-out. Query-budgeted LLM attacker (60 attacks, 10 queries), oracle origin: ASR@10 0.30 (hardened/robust), 0.20 (v2), 0.58 (raw). The lower runtime-attribution figures (0.08-0.15) are an artifact: paraphrased rewrites fall back to `agent_reflection`, which the adversarially trained model learned to distrust; do not rely on them. Attacks delivered through the user turn are the hard case (57 % admitted by the robust model under oracle origin). Detection is not a guarantee; add rate limiting on repeated quarantined writes.
- **F-11 Dependencies.** `pip-audit` flagged fastapi 0.109/starlette 0.35 (upgraded, pinned in the `service` extra). Remaining advisories are in the wider environment (torch, transformers, langchain, etc.) and should be re-audited in CI.

## Addendum 2: Guard API hardening after the ECC security review

Fixed: **self-approval of quarantine** (writer keys can no longer list or review; a separate `DRIFTGUARD_REVIEWER_KEYS` role is required and must differ from writer keys); **body-size bypass** (POST without `Content-Length`, e.g. chunked, is rejected with 411); **unbounded growth** (per-tenant caps: 1000 memories, 500 pending quarantine items, with 507/429); **race conditions** (tenants created eagerly, per-tenant lock around write and review); **key handling** (keys are looked up by SHA-256 digest, duplicate keys and tenants containing ':' are rejected); **model signing** (the signature is verified over the exact bytes that are unpickled, malformed signature files raise `ModelIntegrityError`, an HMAC signature without `DRIFTGUARD_MODEL_KEY` is refused, and the service will not load a model without `DRIFTGUARD_MODEL_KEY`); `/docs` disabled. CI `bandit` previously failed on the intentional, signature-gated `pickle.loads`; it is now annotated `# nosec B301` with the reason.

Still open (accepted residual risk, in priority order):
1. `origin` is caller-supplied: a client can claim `user_turn`. Bind each key to an allowed origin set.
2. Write history is shared by all sessions of a tenant and `session_id` is client-chosen, so one session can shape the temporal features seen by others. Key history by (tenant, session).
3. Failed-auth requests are not throttled (the rate limit applies only after a key matches).
4. The JSONL and Redis review queues are not tenant-aware or concurrency-safe; the API uses the in-memory queue only.
5. Quarantined text is stored unencrypted, and `chmod` has no effect on Windows.
6. Pickle remains the model format; HMAC signing mitigates but a non-pickle format (skops/ONNX) would remove the class of risk.
7. Fail-closed quarantine can be abused to fill the review queue with inputs that make feature extraction raise (now bounded by the pending cap).
8. `uvicorn service.api:app` fails with an opaque error when no keys are set; prefer `create_app()` via a factory.
