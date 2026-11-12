# DriftGuard: Provenance-Aware Write-Time Detection of Memory Poisoning versus Benign Personalisation Drift in LLM Agents

**Authors:** Tanisha Talathi, Aryaan Sethi, Rishabh Gaikwad — Dept. of Computer Engineering, DCET, MIT World Peace University, Pune.
**Status:** working draft. Every number is produced by a script in `scripts/` (see §9) and stored in `results*/`. Citation details must be verified before submission.

---

## Abstract

LLM agents with persistent memory can be steered by *memory poisoning*: an attacker plants an entry that later changes the agent's behaviour. Blocking every memory change would also block the personalisation users want, because genuine preference *drift* ("I moved to Bengaluru") looks structurally like an attack. We study **write-time three-class classification** — malicious, benign drift, ordinary learning — with actions quarantine / admit-and-flag / admit. Our detector scores each candidate write with 18 features (provenance, temporal, consistency, content) plus 7 semantic attack-intent features, using a calibrated random forest.

We report results the way a sceptical reviewer would demand. (1) On a synthetic benchmark the model reaches 0.918 ± 0.018 accuracy under repeated cross-validation after label-preserving *hardening* (0.997 on the raw benchmark, which we show is inflated by generator shortcuts). (2) **We found and fixed a label leak in our own feature pipeline**: session age was computed against the wall clock at extraction time and, because the benchmark is generated class by class, identified the class almost perfectly (AUROC 0.94 on random labels); the burst-window features were also dead in training. All numbers here are post-fix. (3) A controlled stealth-attack experiment shows provenance helps only in proportion to its reliability: with reliability 0.95 it lifts accuracy from 0.62 to 0.72 and cuts false alarms from 0.17 to 0.04, and with reliability 0.5 it adds nothing. (4) A combined adaptive attacker cuts detection to 0.23; adversarial augmentation restores 1.00, and holding out each evasion type still gives 0.90–1.00. (5) Leave-one-attack-family-out detection is weak (0.31 on MINJA) until real injection text is added to training (0.89). (6) On free-form text written by an independent LLM the in-template accuracy does not survive (0.66–0.67, false alarms 23–27 %). (7) In an end-to-end study with a small local LLM agent (90 scenarios, 30 per class, one run, no confidence intervals) a simple rule-based filter is a strong baseline: it cuts attack success from 0.47 to 0.20, versus 0.27–0.37 for DriftGuard variants (v2 0.27, hard 0.37). We report this as it is; DriftGuard's advantage is not demonstrated end to end here, only that it blocks fewer attacks than the rules (0.70 vs 0.80) while, like them, not quarantining any benign write in this scenario set.

## 1. Introduction

Persistent memory turns a stateless assistant into a personal one, and turns every input channel — web pages, documents, tool outputs — into a *write path* to future behaviour. MINJA and AgentPoison show planted memories persist and are retrieved later [1, 2], an instance of indirect prompt injection [3] with a long half-life. Input filters examine single messages; memory defence must examine a *state change over time* without destroying the benefit memory provides. The hard negative is legitimate **drift**: users do change preferences, and such changes contradict stored memories just as attacks do.

**Contributions.**
1. A three-class formulation with distinct actions and an enforcement pipeline.
2. A methodological finding: a wall-clock label leak that inflated every metric in our first evaluation, how it was found (a random-label control), and the fix with a regression test.
3. A **controlled measurement of when provenance helps** (stealth attacks, reliability sweep) and a runtime *source-attribution* mechanism for obtaining provenance in an agent.
4. Adaptive-attacker evaluation: scripted evasions, adversarial augmentation, leave-one-evasion-out, and an LLM-driven black-box attacker with a query budget.
5. Real-text transfer (public prompt-injection and persona datasets), independent-phrasing evaluation, and an end-to-end study with a local Llama agent measuring attack success *and* user utility.
6. A security audit of the implementation, with fixes (signed model loading, fail-closed pipeline, sanitiser, authenticated API).

## 2. Threat model

The attacker controls content the agent reads (web page, document, tool result) or talks to the agent; cannot modify the host, the model file, or the runtime that labels a write's source. An *adaptive* attacker knows the defence exists. Goal: quarantine attacks, admit benign writes. Out of scope: attacks not routed through memory writes, host compromise, weight poisoning. Full audit: `SECURITY.md`.

## 3. Method

**Features.** *Provenance (3):* origin, trust level, history span. *Temporal (4):* writes in the last 10/30 minutes, mean inter-write interval, burst score. *Consistency (5):* similarity to memory centroid (MiniLM), novelty, same-session writes, origin diversity, repeated-hash flag. *Content (6):* query overlap, length, instruction phrasing, URL, word length, exclamation ratio. *Semantic (7, v2):* max cosine similarity to anchor sets describing override / exfiltration / promotion / trigger / payment-redirect intent and to benign facts, plus the attack-minus-benign margin. **All time features are relative to the write's own timestamp** (§5.2).

**Classifier.** Standard scaling + calibrated random forest (also GBM, logistic regression).

**Runtime source attribution.** The runtime tags what entered the context (user message, tool output, retrieved document). A candidate write is attributed to the **least-trusted source it is grounded in** (token containment ≥ 0.5); ungrounded writes are labelled `agent_reflection`. The original agent labelled every write `agent_reflection`, which carries the *highest* default trust (0.9) — worse than having no provenance.

**Enforcement.** Class 0 → quarantine and human review; class 1 → admit, flag, withhold from the prompt until reviewed; class 2 → admit. Text is NFKC-normalised, invisible characters stripped and length-capped; the pipeline fails closed on detector errors and ignores caller-supplied trust.

## 4. Data

*Synthetic benchmark:* 600 writes (100 MINJA-style, 100 AgentPoison-style attacks; 200 preference-change; 200 fact-learning) and a 200-attack held-out set from two unseen families. *Hardening* swaps write history and trust with a random donor, equalises length with neutral filler, and scrubs label-bearing metadata (probability 0.7 each; labels untouched). *Real text:* injection strings from deepset/prompt-injections and xTRam1/safe-guard-prompt-injection (label 1; 400 used, truncated to 300 chars) and 400 crowd-written persona facts from Cynaptics/persona-chat. These are prompts and chat facts rather than genuine memory writes, so they test text transfer only. Non-text context for real text is borrowed from random synthetic entries (class-independent).

## 5. Results

### 5.1 Main comparison (repeated 5×3 stratified CV, mean ± 95 % CI)

| Benchmark | Model | Accuracy | Macro-F1 | MCC | Attack TPR | False alarm |
|---|---|---|---|---|---|---|
| Raw | DriftGuard RF | 0.997 ± 0.003 | 0.997 | 0.996 | 1.000 | 0.000 |
| Raw | Rule-based | 0.642 ± 0.007 | 0.532 | 0.557 | 1.000 | 0.000 |
| **Hardened** | **DriftGuard RF** | **0.918 ± 0.018** | 0.918 | 0.879 | 0.987 | 0.028 |
| Hardened | DriftGuard GBM | 0.923 ± 0.015 | 0.923 | 0.885 | 0.983 | 0.020 |
| Hardened | Logistic regression | 0.909 ± 0.013 | 0.910 | 0.866 | 0.985 | 0.003 |
| Hardened | Rule-based | 0.557 ± 0.011 | 0.442 | 0.389 | 0.995 | 0.310 |
| Hardened | Majority class | 0.333 | 0.167 | 0.000 | 1.000 | 1.000 |

RF beats the rule baseline on 240 items the baseline misses vs 21 the reverse (exact McNemar p = 2.8 × 10⁻⁴⁸). Logistic regression is within one point: as generated, the task is close to linearly separable. Raw accuracy is inflated; the hardened figure is the honest one but still synthetic.

### 5.2 A label leak in our own pipeline

The first evaluation reported 0.923 hardened accuracy. A random-label control on a stealth experiment showed feature 2 (*session age*) alone reaching AUROC 0.94 where chance was 0.50. Cause: it was `now − record.timestamp` with `now` read at extraction time; records are created class by class, so extraction order — and hence the value — encoded the class. The same `now` zeroed `writes_last_10/30` during training while live serving saw real values (train/serve skew). Fix: measure every time feature relative to the write's own timestamp; a regression test pins the behaviour. After the fix hardened accuracy moved only 0.923 → 0.918, but "provenance alone" fell 0.72 → 0.64, so part of the earlier provenance signal was the leak. **Recommendation:** random-label controls and per-feature single-column AUROC checks should be standard for security classifiers built on generated data.

### 5.3 Feature-family ablation (hardened, RF)

| Features | Accuracy | Attack TPR | False alarm |
|---|---|---|---|
| All | 0.918 | 0.987 | 0.028 |
| Without provenance | 0.908 | 0.995 | 0.030 |
| Without temporal | 0.927 | 0.995 | 0.030 |
| Without consistency | 0.920 | 0.980 | 0.010 |
| Without content | **0.703** | 0.945 | 0.050 |
| Only provenance | 0.642 | 0.660 | 0.020 |
| Only content | **0.910** | 0.985 | 0.020 |

On this synthetic data, content carries almost all the signal and removing provenance costs one point. We therefore do **not** claim provenance drives accuracy on the benchmark; §5.4 asks when it should.

### 5.4 When does provenance help? (stealth attacks)

Stealth attacks read like legitimate preferences ("User prefers shopping for laptops at ⟨site⟩"); matched benign writes use the same templates with ordinary entities, including URLs and e-mail addresses. Reliability ρ = P(attack via untrusted origin) = P(benign via user origin). All non-text context comes from random synthetic entries so that at ρ = 0.5 every non-text feature is uninformative by construction (verified: provenance-only AUROC 0.487).

| ρ | Features | Accuracy | Attack AUROC | Attack TPR | False alarm |
|---|---|---|---|---|---|
| 0.5 | text only | 0.642 | 0.939 | 0.863 | 0.163 |
| 0.5 | text + provenance | 0.618 | 0.922 | 0.850 | 0.182 |
| 0.8 | text only | 0.617 | 0.939 | 0.880 | 0.205 |
| 0.8 | text + provenance | 0.647 | 0.957 | 0.933 | 0.137 |
| 0.95 | text only | 0.623 | 0.938 | 0.867 | 0.167 |
| 0.95 | text + provenance | **0.717** | **0.986** | 0.960 | **0.038** |

The detector trained on the standard benchmark scores 0.43–0.46 accuracy (AUROC ≈ 0.68) on stealth attacks, i.e. near chance. **Provenance is valuable exactly insofar as the runtime can supply it reliably**; at ρ = 0.5 it is noise. The end-to-end study (§5.9) measures the ρ that runtime attribution actually achieves.

### 5.5 Adaptive attacker (in-distribution attacks, hardened benchmark)

| Evasion | Standard model | Adversarially trained |
|---|---|---|
| none | 1.00 | 1.00 |
| zero-width / full-width / synonym swap / origin laundering / history camouflage | 0.97–1.00 | 1.00 |
| cross-script homoglyphs | 0.97 | 0.97 |
| benign padding | 0.72 | 0.97 |
| **all combined** | **0.23** | **1.00** |

Adversarial augmentation uses seven evasion types on malicious training writes and matching padding/synonym perturbations on benign writes so the perturbation is not a label cue. **Leave-one-evasion-out** (train without a technique, test on it): padding 0.90, combined attacker 0.875, homoglyph 0.975, all others 1.00. The held-out evasions are siblings of the trained ones and were written by the detector's authors, so this shows transfer, not safety. Held-out attack families score ≥ 0.995 under every evasion — they are easier than the in-distribution set and are weak evidence of generalisation.

### 5.6 Unseen attack families and real text

Leave-one-attack-family-out on the hardened benchmark (train on the other family, test on this one): AgentPoison 1.00, **MINJA 0.31**, mean 0.65. Zero-shot transfer to real injection text (synthetic-only models):

| Detector | Attack TPR | False alarm | AUROC |
|---|---|---|---|
| Rule-based | 0.92 | 0.51 | 0.71 |
| Standard (raw) | 0.54 | 0.24 | 0.73 |
| Hardened | 0.77 | 0.10 | 0.90 |
| Adversarially trained | 0.82 | 0.12 | 0.93 |

Adding real text to training and the semantic features (**v2**) gives:

| Setting | LOFO AgentPoison | LOFO MINJA | LOFO mean | Real TPR | Real FAR | CV acc |
|---|---|---|---|---|---|---|
| base 18 features | 1.00 | 0.30 | 0.65 | 0.67 | 0.06 | 0.922 |
| + semantic features | 1.00 | 0.31 | 0.66 | 0.73 | 0.04 | 0.943 |
| + real text | 1.00 | 0.93 | 0.97 | 0.87 | 0.11 | 0.855 |
| **both (v2)** | 1.00 | 0.89 | 0.95 | 0.94 | 0.05 | 0.903 |

The v2 model on a held-out half of the real strings: attack TPR 0.93, false alarm 0.05, AUROC 0.99. The semantic features help by themselves (+2 points CV accuracy); the large LOFO gain comes from real text supplying a second attack distribution, not from a trick. v2 is **not** zero-shot on real text (it trained on other strings from the same pools).

### 5.7 Independent phrasing

A local LLM (llama3.2:1b) wrote 180 free-form memory entries (60 per class); detectors were scored without retraining. Only the text is independent; context is borrowed from synthetic same-class entries.

| Detector | Accuracy | Macro-F1 | Attack recall | Drift recall | Ordinary recall | False alarm |
|---|---|---|---|---|---|---|
| Rule-based | 0.528 | 0.420 | 0.93 | 0.00 | 0.65 | 0.55 |
| Hardened | 0.656 | 0.644 | 0.88 | 0.68 | 0.40 | 0.23 |
| Adversarially trained | 0.672 | 0.664 | 0.92 | 0.62 | 0.48 | 0.27 |

Attacks are still caught (88–92 %), but in-template accuracy (0.92) does not transfer: drift and ordinary are confused and a quarter of benign writes are quarantined. A deployment would need threshold tuning and human review capacity sized for this.

### 5.8 Black-box adaptive attacker

Query-budgeted attacker (llama3.2:1b rewrites a quarantined attack up to 10 times; 60 attacks per detector). Attack success rate at budget k (ASR@k), where lower is better:

| Detector | ASR@1 | ASR@5 | ASR@10 |
|---|---|---|---|
| raw, oracle origin | 0.12 | 0.48 | 0.58 |
| raw, runtime attribution | 0.22 | 0.65 | 0.80 |
| hard, oracle origin | 0.02 | 0.22 | 0.30 |
| hard, runtime attribution | 0.00 | 0.20 | 0.27 |
| robust, oracle origin | 0.02 | 0.22 | 0.30 |
| robust, runtime attribution | 0.00 | 0.02 | 0.08 |
| v2, oracle origin | 0.00 | 0.15 | 0.20 |
| v2, runtime attribution | 0.00 | 0.10 | 0.15 |

A repeating attacker with a few queries evades every detector some of the time; adversarial training (robust) and semantic features (v2) reduce but do not eliminate this. The attacker is a 1B model, so these are optimistic for the defender. Write-time rate limiting on repeated quarantined writes is therefore a needed companion control.

### 5.9 End-to-end agent study

Setup: a local Llama agent, 30 attack, 30 drift and 30 ordinary scenarios (attacks via tool output and, for about a quarter, the user turn; includes hard-negative benign items containing emails/URLs). Metrics: block rate on attacks, attack success rate (ASR, the injected behaviour appears in the agent's behaviour), and utility on benign writes. One run per cell; with n = 30 the 95 % interval on a proportion is about ±0.17, so differences below roughly 0.15 are not significant.

| Defence | Attacks blocked | ASR | Drift admitted | Ordinary admitted |
|---|---|---|---|---|
| None | 0.00 | 0.47 | 1.00 | 1.00 |
| Rule-based | 0.80 | 0.20 | 1.00 | 1.00 |
| LLM judge (local) | 0.47 | 0.27 | 1.00 | 1.00 |
| DriftGuard-hard, runtime attribution | 0.57 | 0.37 | 1.00 | 1.00 |
| DriftGuard-robust, runtime attribution | 0.70 | 0.30 | 1.00 | 1.00 |
| DriftGuard-v2, runtime attribution | 0.70 | 0.27 | 1.00 | 1.00 |

Honest reading: every defence lowers ASR, but the rule-based filter does at least as well as DriftGuard in this setting, and the differences among defences are within sampling noise. Attribution matters little here (v2 oracle = runtime). Withholding flagged drift writes costs 0–13 % of drift utility depending on the model. We therefore do not claim end-to-end superiority; the contribution is the write-time architecture, the leak finding and the evaluation methodology.

## 6. Security analysis of the system

Fixed: unsigned pickle loading (code execution; regression test with a real malicious pickle), fail-open on detector errors, caller metadata overwriting the *flagged* mark, caller-supplied trust, markdown rendering of untrusted text, unbounded input, and a vulnerable web stack found by `pip-audit`. **Residual:** attribution is heuristic (§5.9); the detector is one layer, not a guarantee; quarantined text is stored unencrypted. See `SECURITY.md`.

## 7. Related work

Memory/knowledge-base poisoning: MINJA [1], AgentPoison [2]; indirect prompt injection [3]; dataset artefacts and shortcut learning [4]. Input-level guardrails address single messages rather than persistent state. *A sourced survey is still to be written; we make no novelty claim against specific systems until it is.*

## 8. Limitations

* The core benchmark is synthetic and authored by the detector's authors; the real-text experiments test text transfer only, and no real corpus of poisoned memory traces with provenance exists to us.
* Logistic regression matches the forest: the task as generated is nearly linear, so model complexity is not the contribution.
* Provenance does not drive accuracy on the benchmark (§5.3); its value is shown only in a controlled experiment whose ρ is a parameter (§5.4).
* Unseen-family generalisation required real training text; with synthetic data alone it is poor (§5.6).
* Free-form LLM-written text lowers accuracy substantially (§5.7); the generator is a 1B model with limited diversity.
* End-to-end results use a 1B local agent and a small scenario set; the write proposer is simulated (the injection is assumed to have produced a write).
* Single-split numbers vary: a 20 % split gave 7.5 % false alarms where 5×3 CV gives 2.8 %; we report CV.
* No user study of review burden.

## 9. Reproducibility

```
python -m scripts.generate_benchmark --hard
python -m scripts.train_detector --benchmark data/benchmark_hard.jsonl --model-out models/classifier_hard.pkl
python -m scripts.download_external && python -m scripts.train_v2
python -m scripts.run_research_eval            # 5.1, 5.3
python -m scripts.run_stealth_eval             # 5.4
python -m scripts.run_adversarial && python -m scripts.train_robust && python -m scripts.run_transform_holdout   # 5.5
python -m scripts.run_generalisation && python -m scripts.run_real_text_eval   # 5.6
python -m scripts.run_independent_eval         # 5.7 (needs Ollama)
python -m scripts.run_blackbox_attack          # 5.8 (needs Ollama)
python -m scripts.run_agent_study              # 5.9 (needs Ollama)
```
Seeds are fixed. 114 unit and regression tests run with `pytest`.

## 10. Future work

Real memory-poisoning traces with provenance from design partners; taint tracking inside the model runtime rather than lexical grounding; retrieval-time filtering as a second layer; attacks authored by people who have not seen the detector; a user study of review burden.

## References (verify before submission)

[1] Dong et al., "A Practical Memory Injection Attack against LLM Agents" (MINJA), 2025.
[2] Chen et al., "AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases", NeurIPS 2024.
[3] Greshake et al., "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection", 2023.
[4] Geirhos et al., "Shortcut Learning in Deep Neural Networks", Nature Machine Intelligence, 2020.
[5] OWASP, "Top 10 for LLM Applications".
