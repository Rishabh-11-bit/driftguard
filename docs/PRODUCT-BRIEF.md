# DriftGuard — Product Brief & Path to a Sellable Product

Method: ECC `product-lens` (diagnostic + founder review) and `plan-prd` (anti-fluff rule:
anything not evidenced is written as **TBD — needs validation via …**, never invented).
Status: draft for the team; nothing here has been validated with a real customer.

## 1. Diagnosis (product-lens, Mode 1)

| Question | Answer |
|---|---|
| Who is it for? | Teams shipping LLM agents with **persistent memory** (assistants, copilots, support/CRM agents) — the engineer who must stop memory poisoning without breaking personalisation. Specific persona: *TBD — validate with 8–10 interviews.* |
| What is the pain? | Memory is a persistent prompt-injection channel: one poisoned write affects every later session. Blocking *all* memory change kills the product's personalisation value. Existing prompt-injection filters inspect a single message, not a **write to long-term state over time**. |
| What happens today? | Teams either (a) don't protect memory, (b) apply generic input filters, or (c) disable memory. *Competitive landscape: TBD — needs a sourced survey (use ECC `market-research`); do not claim a gap until verified.* |
| Why now? | Agent memory is shipping in mainstream assistants; the research literature (MINJA, AgentPoison) shows practical attacks. *Market-timing evidence: TBD.* |
| The 10-star version | A drop-in guard that sits in front of any agent memory store, uses **true provenance** (taint-tracked source of every write), distinguishes attack vs. legitimate preference change, and gives security teams audit logs + a review inbox. |
| MVP | See §3. |
| Anti-goals | Not a general prompt-injection firewall; not an LLM-hosting platform; not a compliance certification; not a consumer app. |
| How do we know it works? | Detection rate on **real** poisoning traces, false-alarm rate on **real** user memory edits, p95 latency, % of quarantines humans overturn. Current numbers are synthetic-only (see §5). |

**Go / no-go (honest):** *Conditional go for a design-partner pilot — no-go for selling.*
The research prototype is strong enough to demo and to publish, but three things block a sale:
(1) no evidence on real data, (2) provenance in the agent is not trustworthy yet (SECURITY.md F-03),
(3) no service/ops layer. §4 is the plan to close these.

## 2. Founder review (product-lens, Mode 2) — where the repo stands

| Dimension | Score /10 | Evidence |
|---|---|---|
| Technical depth | 7 | 25-feature detector (18 + 7 semantic), calibrated 3-class model, hardened benchmark, LOFO, adaptive-attacker eval, runtime provenance attribution, 126 tests |
| Evidence quality | 3 | Synthetic core; a label leak was found and fixed (accuracy 0.997 -> 0.918); length-matched real text shows no transfer for synthetic-trained models; LLM-written note-style attacks are mostly missed; end-to-end, a simple rule filter is at least as good (ASR 0.24 vs 0.36). No real traces. |
| Security posture | 6 | Signed models verified on the loaded bytes, fail-closed, sanitiser, authenticated API with separate reviewer role, per-tenant quotas/locks, key-to-origin binding, per-session history, failed-login throttle; provenance attribution is heuristic; see SECURITY.md for the remaining list |
| Product readiness | 4 | Streamlit demo + authenticated multi-tenant Guard API + Dockerfile + CI; no persistence for the API (in-memory tenants), no UI for reviewers, no docs site |
| Differentiation | 6 | Write-time, provenance-aware, drift-vs-attack distinction. *Moat unproven.* |
| Commercial readiness | 1 | No licence, no pricing evidence, IP ownership unresolved |

**The one thing that would 10× this:** real provenance (taint tracking) + a design-partner dataset.
**What doesn't matter yet:** more UI polish, more backends (Chroma/Redis), more feature engineering on synthetic data.

## 3. MVP definition (what to build)

**DriftGuard Guard API + SDK** — a service in front of any agent memory store.

* `POST /v1/guard/write` → `{verdict, action, probabilities, reasons[], quarantine_id?}` (p95 target < 100 ms *TBD — current ≈ 0.3 s on CPU incl. embedding*).
* `GET/POST /v1/quarantine` → review inbox (approve / reject), with audit trail.
* Python SDK + LangGraph/LangChain adapters; a 5-line integration.
* Per-tenant API keys, rate limits, request-size limits, structured audit log.
* Dashboard (the existing Streamlit app, behind auth) for security reviewers.
* Self-hostable container first (data never leaves the customer) — likely the deciding factor for privacy-sensitive buyers *(hypothesis — validate)*.

Explicitly **out of MVP**: auto-learning from reviews, multi-modal memory, SaaS billing.

## 4. Roadmap — closing the gaps in order

| Phase | Goal | Deliverables | Exit criterion |
|---|---|---|---|
| **0 (now – 2 wks)** Research-grade | Honest, publishable evidence | Paper draft, SECURITY.md, adaptive-attacker eval, independent-phrasing test, CI | Numbers reproducible with one command |
| **1 (wks 2–6)** Service | Usable by a developer | FastAPI service + auth + rate limits; Docker image; SDK; OpenAPI docs; `pip-audit` + tests in CI | A stranger integrates it in < 30 min (time it — product-lens Mode 3) |
| **2 (wks 4–10)** Real provenance | Close F-03 | Taint tracking in the LangGraph adapter (user / tool / retrieved); retrieval-time filtering; memory cannot grant tool authority | Origin label equals ground truth in an instrumented agent |
| **3 (wks 6–14)** Real data | Replace synthetic claims | 2–3 design partners (or a public agent-memory corpus); labelled real writes; re-train; report CIs | Detection / false-alarm measured on real traces |
| **4 (wks 12+)** Hardening for sale | Trust | Third-party pen-test; encryption at rest + TTL + erasure API; SBOM; SOC 2 / ISO 27001 roadmap *(only if buyers ask)*; SLA | Pen-test report with no open High findings |

## 5. Evidence required before charging money

1. Detection and false-alarm rates on **real** data with confidence intervals (current: synthetic only).
2. Robustness to a **red team that did not write the detector** (current: our own evasion families).
3. At least one design partner running it in a real agent for ≥ 4 weeks, with the % of quarantines humans agreed with.
4. Latency / cost per 1 000 writes at realistic load (current: single-machine, unmeasured under load).

## 6. Business model hypotheses (all TBD — needs validation via customer interviews)

* **Open-core:** open-source detector + SDK; paid hosted/managed service, enterprise review dashboard, SSO, audit export, support SLA.
* **Pricing metric candidates:** per 1 000 guarded writes, per agent, or flat per security seat. *No price point is supported by evidence yet — do not quote one.*
* **Channel hypotheses:** developer-led (GitHub, LangChain ecosystem, security conferences); security-team-led for regulated industries.

## 7. Legal / ownership checklist (do before any commercial step)

* **IP ownership:** this started as an MIT-WPU capstone. Check the university's IP/consultancy policy and the guide's position *before* commercialising or incorporating — in writing.
* **Licence:** the public repo currently has **no LICENSE file**, which legally means "all rights reserved"; choose one deliberately (MIT/Apache-2.0 for open-core).
* **Third-party licences:** audit the dependency tree (sentence-transformers model licence, FAISS, scikit-learn, LangChain).
* **Privacy:** the product processes user memories (personal data). Determine controller/processor role, retention, and erasure obligations (DPDP Act 2023 in India; GDPR if EU users).
* **Names/IDs:** the repo and report contain student names and ERP numbers — decide what stays public.

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Model-vendor memory features add built-in protection | Med | High | Stay vendor-neutral; focus on provenance + audit across vendors |
| Synthetic results don't transfer to real traffic | **High** | **High** | Phase 3 before any sales claim |
| Adaptive attackers defeat the detector | High | High | Defence in depth (R3), continuous red-teaming, human review |
| False alarms annoy users (current ≈ 5–8 %) | Med | Med | Threshold tuning per tenant; review UX; measure overturn rate |
| IP / ownership dispute | Low–Med | High | §7, resolve first |

## 9. Next 5 actions

1. Decide licence + settle IP ownership with the university (blocking).
2. Ship Phase 1 service skeleton (in progress — see `service/`).
3. Recruit 3 design partners / find a public agent-memory dataset.
4. Run ECC `market-research` for a sourced competitor survey before any positioning claims.
5. Time a first-integration walkthrough (product-lens Mode 3) and fix the top 3 frictions.
