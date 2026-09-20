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
