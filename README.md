# DriftGuard

> **Distinguishing Malicious Memory Poisoning from Benign Personalisation Drift in LLM Agents**

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: LangGraph](https://img.shields.io/badge/framework-LangGraph-green)](https://github.com/langchain-ai/langgraph)

**Capstone Project — MIT World Peace University (MIT-WPU), CSE Dept.**  
Group C | Panel C | Academic Year 2024–25

| Team Member | PRN | Role |
|---|---|---|
| Tanisha Talathi | 1032231612 | Benchmark construction & adversarial data generation |
| Aryaan Sethi | 1032231670 | Provenance memory layer, agent integration & enforcement |
| Rishabh Gaikwad | 1032231674 | Detector modelling & evaluation |

---

## Problem

Modern LLM agents use **persistent long-term memory** to personalise responses across sessions. This memory is a new attack surface — MINJA achieves ~98% memory injection success through ordinary queries; AgentPoison backdoors an agent with as few as two poisoned entries.

The core challenge: **malicious poisoning and benign preference updates look identical** from the memory store's perspective. Existing binary detectors either raise constant false alarms on legitimate updates, or miss stealthy attacks (~2/3 missed by single-LLM detectors).

## Solution: DriftGuard

DriftGuard reframes detection as a **three-way classification**:

```
Write candidate ──► Provenance Tagger ──► Feature Extractor ──► 3-Class Classifier
                                                                        │
                          ┌─────────────────────────────────────────────┤
                          ▼                    ▼                         ▼
                   MALICIOUS (0)        BENIGN DRIFT (1)      ORDINARY LEARNING (2)
                   Quarantine           Admit + Flag           Admit silently
```

**Key innovations:**
- Provenance tagging on every memory write (origin, trust level, timestamp)
- Three-class benchmark (first of its kind — LOCOMO/LongMemEval ignore poisoning)
- Enforcement at write-time, retrieval-time, and forensic-audit time
- Evaluation metric: FAR on benign drift vs. MAR on malicious (binary baselines fail this)

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Agent Runtime (LangGraph)              │
│  User Query → Retrieve Memory → LLM → Extract Writes ──►  │
└────────────────────────────────────┬───────────────────────┘
                                     │ candidate write
                          ┌──────────▼──────────┐
                          │   Write Pipeline     │
                          │  (provenance tag)    │
                          └──────────┬──────────┘
                                     │ ProvenanceRecord
                          ┌──────────▼──────────┐
                          │  DriftGuard Detector │
                          │  ┌────────────────┐ │
                          │  │ Provenance Feat│ │
                          │  │ Temporal Feat  │ │
                          │  │ Consistency    │ │
                          │  │ Causal Attr.   │ │
                          │  └──────┬─────────┘ │
                          │  RandomForest/MLP    │
                          └──────────┬──────────┘
                                     │ label {0,1,2}
                          ┌──────────▼──────────┐
                          │  Enforcement Layer   │
                          │  0→Quarantine        │
                          │  1→Admit+Flag        │
                          │  2→Admit silently    │
                          └─────────────────────┘
```

---

## Quick Start

### 1. Install (using [uv](https://github.com/astral-sh/uv))

```bash
# Install uv if you don't have it
pip install uv

# Install all dependencies
cd driftguard
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY (or set LLM_PROVIDER=ollama)
```

### 2. Generate Benchmark

```bash
uv run python scripts/generate_benchmark.py \
    --n-malicious 200 --n-drift 200 --n-ordinary 200 \
    --out data/benchmark.jsonl
```

### 3. Train Detector

```bash
uv run python scripts/train_detector.py \
    --benchmark data/benchmark.jsonl \
    --model-out models/classifier.pkl
```

### 4. Run Evaluation

```bash
uv run python scripts/run_eval.py \
    --benchmark data/benchmark.jsonl \
    --model models/classifier.pkl \
    --out results/
```

### 5. Interactive Demo

```bash
uv run python scripts/demo_agent.py
```

### 6. Run Tests

```bash
uv run pytest tests/ -v
```

---

## Project Structure

```
driftguard/
├── driftguard/
│   ├── agent/              # LangGraph agent + LLM/memory interfaces
│   ├── provenance/         # ProvenanceRecord schema + WritePipeline
│   ├── detector/           # Feature extraction + 3-class classifier
│   └── benchmark/          # Attack generators + benign drift synthesisers
├── eval/
│   ├── harness.py          # Benchmark replay harness
│   ├── metrics.py          # 3×3 confusion matrix, FAR, MAR
│   └── baselines/          # Binary detector baselines
├── scripts/                # CLI entry points
├── tests/                  # pytest test suite
├── notebooks/              # Jupyter analysis notebooks
├── data/                   # Generated benchmark + FAISS index
├── models/                 # Trained classifier
└── results/                # Evaluation outputs + plots
```

---

## Benchmark Classes

| Class | Label | Description |
|---|---|---|
| Malicious Poisoning | `0` | MINJA-style (indication prompt + bridging) and AgentPoison-style (backdoor trigger) injections |
| Benign Drift | `1` | Legitimate user preference changes over time (diet, music, tone preferences) |
| Ordinary Learning | `2` | Normal fact learning, corrections, preference reinforcement |

---

## Quarantine Queue Backends

The human-review queue for quarantined (malicious) writes is selected via `QUARANTINE_BACKEND` in `.env`:

| Backend | Value | Notes |
|---|---|---|
| JSONL (default) | `jsonl` | Append-only file at `QUARANTINE_PATH`, no extra services needed |
| In-memory | `memory` | Non-persistent, useful for tests/demos |
| Redis | `redis` | Persistent, O(1) review updates; run `docker-compose up redis` and install with `uv sync --extra redis` |

`driftguard.provenance.enforcement.make_queue_from_env()` builds the right backend from the current environment; `EnforcementPolicy()` uses it by default.

---

## Evaluation Metrics

- **3×3 Confusion Matrix** across all three classes
- **False Alarm Rate (FAR)**: benign drift misclassified as malicious
- **Missed Attack Rate (MAR)**: malicious entries not caught
- **ROC curves** per class
- **Generalisation**: held-out attack families not seen at training time

---

## References

- MINJA: *"A Practical Memory Injection Attack against LLM Agents"* — Dong et al.
- AgentPoison: *"AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases"* — NeurIPS 2024
- OWASP AI Security Top 10: ASI06 — Memory and Context Poisoning
- LOCOMO, LongMemEval — agent memory recall benchmarks (poisoning not evaluated)

---

## License

MIT — See [LICENSE](LICENSE). All attacks are reproduced for **defensive research only** in a controlled laboratory environment against the project's own agent.

## Extended Metrics and the Hardened Benchmark

`eval/extended_metrics.py` adds balanced accuracy, MCC, Cohen's kappa, attack recall/precision/F2,
drift-preservation and drift-recognition rates, PR-AUC, Brier score, ECE, an asymmetric expected-cost
score, bootstrap 95% CIs, per-family detection and the held-out generalisation gap. `run_eval.py`
prints all of them plus leave-one-attack-family-out detection.

The raw generators leave class fingerprints in timing, trust, text length and metadata, so the raw
benchmark is separable at ~100% accuracy. `driftguard/benchmark/hardening.py` removes them:

```bash
python scripts/generate_benchmark.py --hard --out data/benchmark_hard.jsonl
python scripts/generate_benchmark.py --harden-file data/held_out.jsonl --out data/held_out_hard.jsonl
python scripts/train_detector.py --benchmark data/benchmark_hard.jsonl --model-out models/classifier_hard.pkl
python scripts/run_eval.py --benchmark data/benchmark_hard.jsonl --model models/classifier_hard.pkl \
    --held-out data/held_out_hard.jsonl --out results_hard
```

## Demo UI

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Tabs: **Live Guard** (pick a scenario or type a memory write, see the verdict, probabilities,
enforcement action, the memory store and a quarantine inbox with approve/reject), **Evaluation**
(raw vs hardened benchmark metrics and plots), **Benchmark** (dataset explorer) and
**How it works**. It needs `models/classifier_hard.pkl` (see the hardened benchmark commands above).

### Local Llama agent (Ollama)

The demo's **Live Guard** tab has an "Ask the local Llama agent" panel. With
[Ollama](https://ollama.com) running and a model pulled (`ollama pull llama3.2:1b`),
it asks the same local model the same question twice: once with a memory that stored
every write (unprotected) and once with only what DriftGuard admitted. Submit an
attack scenario first to see the poisoned recommendation appear only in the
unprotected answer. No cloud API key is needed.

## Security, evaluation and productisation

* [`SECURITY.md`](SECURITY.md) — threat model, entry-point table, audit findings (fixed / open) and measured adaptive-attacker results.
* `python -m scripts.run_adversarial` — evasion evaluation; `scripts.train_robust` — adversarial training; `scripts.run_transform_holdout` — leave-one-evasion-out.
* `python -m scripts.run_research_eval` — repeated cross-validation with 95 % CIs, baselines, feature-family ablation, McNemar test.
* `python -m scripts.download_external` - public prompt-injection and persona datasets (Hugging Face); `scripts.run_real_text_eval` / `scripts.run_real_text_controlled` - real-text transfer (the controlled one matches length/prefix and adds style baselines).
* `python -m scripts.run_stealth_eval` - provenance-reliability experiment; `scripts.run_generalisation` - leave-one-family-out; `scripts.run_independent_eval` - LLM-written text (`--attack-as-note` for note-style attacks).
* `python -m scripts.train_v2` / `scripts.train_v3` - v2 (semantic features + real text) and v3 (+ LLM-written notes; needs `scripts.generate_notes`); `scripts.run_blackbox_attack` - query-budgeted LLM attacker; `scripts.run_agent_study` - end-to-end agent study with Wilson intervals.
* Start here for the current state of the research: `results_research/OVERNIGHT-REPORT.md`.
* Research paper draft: `paper/driftguard_paper.md`. Threat model and open items: `SECURITY.md`.
* `python -m scripts.run_independent_eval` — scores trained detectors on free-form text written by a local LLM.
* **Models are signed.** `DriftGuardClassifier.load` refuses unsigned or modified pickles. After training, a `.sig` file is written automatically; for models from elsewhere, review then `python -m scripts.sign_model <file>`. Set `DRIFTGUARD_MODEL_KEY` to use an HMAC.
* **Guard API** (`service/api.py`): writer keys `DRIFTGUARD_API_KEYS="<key-16+chars>:<tenant>[@origin|origin]"` (the optional `@` list restricts which `origin` values that key may assert), a **separate** reviewer role `DRIFTGUARD_REVIEWER_KEYS` (writer keys cannot list or approve quarantine), and `DRIFTGUARD_MODEL_KEY` (HMAC key; required to load a model). Start with `uvicorn --factory service.api:create_app --host 127.0.0.1 --no-server-header` and keep a single worker (tenant state is in memory). Per-tenant quotas, per-key rate limits and a failed-login throttle apply. Container: see `Dockerfile`.
* [`docs/PRODUCT-BRIEF.md`](docs/PRODUCT-BRIEF.md) — path from research prototype to a product, and what evidence is still missing.
