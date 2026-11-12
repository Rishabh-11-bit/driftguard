# DriftGuard: Provenance-Aware Write-Time Detection of Memory Poisoning versus Benign Personalisation Drift in LLM Agents

**Authors:** Tanisha Talathi, Aryaan Sethi, Rishabh Gaikwad — Dept. of Computer Engineering, DCET, MIT World Peace University, Pune.
**Status:** working draft. Every number is produced by a script in `scripts/` (see §9) and stored in `results*/`. Citation details must be verified before submission.

---

## Abstract

LLM agents with persistent memory can be steered by *memory poisoning*: an attacker plants an entry that later changes the agent's behaviour. Blocking every memory change would also block the personalisation users want, because genuine preference *drift* ("I moved to Bengaluru") looks structurally like an attack. We study **write-time three-class classification** — malicious, benign drift, ordinary learning — with actions quarantine / admit-and-flag / admit. Our detector scores each candidate write with 18 features (provenance, temporal, consistency, content) plus 7 semantic attack-intent features, using a calibrated random forest.

We report results the way a sceptical reviewer would demand. (1) On a synthetic benchmark the model reaches 0.918 ± 0.018 accuracy under repeated cross-validation after label-preserving *hardening* (0.997 on the raw benchmark, which we show is inflated by generator shortcuts). (2) **We found and fixed a label leak in our own feature pipeline**: session age was computed against the wall clock at extraction time and, because the benchmark is generated class by class, identified the class almost perfectly (AUROC 0.94 on random labels); the burst-window features were also dead in training. All numbers here are post-fix. (3) A controlled stealth-attack experiment shows provenance helps only in proportion to its reliability: with reliability 0.95 it lifts accuracy from 0.62 to 0.72 and cuts false alarms from 0.17 to 0.04, and with reliability 0.5 it adds nothing. (4) A combined adaptive attacker cuts detection to 0.23; adversarial augmentation restores 1.00, and holding out each evasion type still gives 0.90–1.00. (5) Leave-one-attack-family-out detection is weak (0.31 on MINJA) and the apparent fix from adding real injection text (0.89) is confounded: on a length-and-prefix-matched real-text set the synthetic-trained models are at chance (AUROC 0.51-0.57) and a TF-IDF baseline reaches 1.00, so we do not claim real-text transfer. (6) On free-form text written by an independent LLM, with context leakage removed, accuracy falls to 0.37-0.52 and attacks phrased as ordinary memory notes are missed 57-77 % of the time, so in-template performance does not transfer. (7) In an end-to-end study with a local 8B agent (300 scenarios, Wilson intervals) every defence reduces attack success from 0.96, but a simple rule-based filter is at least as good as DriftGuard (ASR 0.24 vs 0.36 for v2, overlapping intervals); we do not claim an end-to-end advantage.

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

**These real-text numbers are confounded and should not be read as evidence of intent detection.** Attacks were long injection prompts and benign items short persona lines prefixed "User: ", so length alone separates them. We re-ran the zero-shot test on a **controlled set** (`scripts/run_real_text_controlled.py`): both classes share the "User: " prefix and casing, are truncated at a word boundary to at most 200 characters, are length-matched in 20-character bins (242 attacks, 242 benign; mean length 86.9 vs 85.2), and near-duplicates (token Jaccard >= 0.8) are removed.

| Detector (controlled set) | Attack TPR | False alarm | AUROC |
|---|---|---|---|
| Hardened | 0.22 | 0.13 | 0.51 |
| Adversarially trained | 0.30 | 0.19 | 0.57 |
| v2 (strings seen in training masked; n = 353) | 0.49 | 0.10 | 0.79 |

Baselines on the same set: length-only logistic regression AUROC 0.52 (so the length match worked), and **TF-IDF logistic regression AUROC 1.00**: the two public datasets differ in vocabulary, so real-text learnability proves nothing about memory-poisoning intent. The synthetic-trained hardened and robust models are at chance on matched real text; v2 retains some signal (0.79) but it was trained on real injection strings, so it is not zero-shot. The honest conclusion is that **our detectors do not yet transfer to real injection text** once style cues are controlled, and that the earlier 0.90-0.93 AUROCs were largely a length effect.

Adding real text to training and the semantic features (**v2**) gives (these rows come from the unmatched set and carry the same caveat):

| Setting | LOFO AgentPoison | LOFO MINJA | LOFO mean | Real TPR | Real FAR | CV acc |
|---|---|---|---|---|---|---|
| base 18 features | 1.00 | 0.30 | 0.65 | 0.67 | 0.06 | 0.922 |
| + semantic features | 1.00 | 0.31 | 0.66 | 0.73 | 0.04 | 0.943 |
| + real text | 1.00 | 0.93 | 0.97 | 0.87 | 0.11 | 0.855 |
| **both (v2)** | 1.00 | 0.89 | 0.95 | 0.94 | 0.05 | 0.903 |

The v2 model on a held-out half of the real strings: attack TPR 0.93, false alarm 0.05, AUROC 0.99. The semantic features help by themselves (+2 points CV accuracy); the large LOFO gain comes from real text supplying a second attack distribution, not from a trick. v2 is **not** zero-shot on real text (it trained on other strings from the same pools).

### 5.7 Independent phrasing

A local LLM (llama3.2:1b) wrote 180 free-form entries (60 per class); detectors were scored without retraining. An earlier version of this experiment borrowed context (history, trust, origin) from **same-class** synthetic entries, which leaks the label; we corrected it so context comes from random entries of any class and only the text carries signal. Two conditions: **(A)** attacks written as injected sentences (as before), benign as "User ..." notes; **(B)** attacks written as ordinary-looking "User ..." memory notes that secretly carry an instruction, which removes the stylistic cue that separates A. MAR is the fraction of attacks not quarantined; FAR the fraction of benign writes quarantined.

| Detector | A accuracy | A attack recall | A FAR | B accuracy | B attack recall | B FAR |
|---|---|---|---|---|---|---|
| Rule-based | 0.456 | 0.80 | 0.67 | 0.378 | 0.53 | 0.67 |
| Hardened | 0.522 | 0.68 | 0.35 | 0.378 | 0.23 | 0.35 |
| Adversarially trained | 0.506 | 0.72 | 0.45 | 0.367 | 0.30 | 0.45 |
| v2 | 0.656 | 0.98 | 0.45 | 0.472 | 0.43 | 0.45 |

With the leak removed the 0.66-0.67 reported in an earlier draft becomes 0.51-0.52 for the synthetic-trained models, and in condition B attack recall falls to 0.23-0.43. v2's 0.98 in A is largely a style effect: it drops to 0.43 when the attack imitates the benign note format. The honest reading is that **the detectors do not currently catch LLM-written attacks that look like ordinary memory notes, and they quarantine a third to a half of benign free-form notes.** The generator is a 1B model and n = 60 per class (95 % intervals roughly +/-0.12 on a recall), but the drop is far larger than that.

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

Setup: a local **llama3:latest** agent (8B) with 100 attack, 100 drift and 100 ordinary scenarios (attacks arrive via tool output, and about a quarter via the user turn; benign scenarios include hard negatives that contain emails or URLs). ASR is the fraction of attack scenarios in which the injected behaviour appears in the agent's answer. Values are proportions with Wilson 95 % intervals; one run per cell, defences evaluated on the same scenarios, so intervals that overlap do **not** establish a difference and we did not run a paired test.

| Defence | Attacks blocked | Attack success rate | Drift util. (use) | Drift util. (withhold flagged) |
|---|---|---|---|---|
| None | 0.00 [0.00, 0.04] | 0.96 [0.90, 0.98] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] |
| Rule-based | 0.76 [0.67, 0.83] | 0.24 [0.17, 0.33] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] |
| LLM judge (local) | 0.46 [0.37, 0.56] | 0.50 [0.40, 0.60] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] |
| DriftGuard-hard, runtime attribution | 0.52 [0.42, 0.61] | 0.47 [0.38, 0.57] | 0.92 [0.85, 0.96] | 0.82 [0.73, 0.88] |
| DriftGuard-robust, runtime attribution | 0.62 [0.52, 0.71] | 0.37 [0.28, 0.47] | 0.92 [0.85, 0.96] | 0.85 [0.77, 0.91] |
| DriftGuard-v2, runtime attribution | 0.64 [0.54, 0.73] | 0.36 [0.27, 0.46] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] |
| DriftGuard-v2, oracle origin | 0.64 [0.54, 0.73] | 0.36 [0.27, 0.46] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] |

Attribution accuracy: 82 % of attacks were attributed to an untrusted source, but only 77.5 % of benign writes were attributed to the user, so about a fifth of benign writes carry a wrong, lower-trust origin.

Reading, without spin: the 8B agent is far more obedient than the 1B agent of our pilot (96 % ASR undefended versus 47 %), which makes the problem harder. Every defence helps. The **rule-based filter has the lowest ASR (0.24)**, and DriftGuard-v2 (0.36) is not significantly better or worse than it given the overlap of the intervals ([0.17, 0.33] vs [0.27, 0.46]); v2 is better than the local LLM judge (0.50) and the older DriftGuard variants (0.37-0.47) only suggestively. Oracle and runtime origin give identical results for v2, so attribution quality did not matter for it here. Withholding flagged drift writes costs up to 10 points of drift utility for the hardened/robust models and nothing for v2. **We therefore do not claim that DriftGuard beats a rule-based filter end to end.** A pilot with a 1B agent (30 scenarios per class) also had the rule-based filter best, but ranked the LLM judge second, where the larger study puts it last; the pilot is too small to rank defences and we do not report it as evidence.

## 6. Security analysis of the system

Fixed: unsigned pickle loading (code execution; regression test with a real malicious pickle), fail-open on detector errors, caller metadata overwriting the *flagged* mark, caller-supplied trust, markdown rendering of untrusted text, unbounded input, and a vulnerable web stack found by `pip-audit`. **Residual:** attribution is heuristic (§5.9); the detector is one layer, not a guarantee; quarantined text is stored unencrypted. See `SECURITY.md`.

## 7. Related work

Memory/knowledge-base poisoning: MINJA [1], AgentPoison [2]; indirect prompt injection [3]; dataset artefacts and shortcut learning [4]. Input-level guardrails address single messages rather than persistent state. *A sourced survey is still to be written; we make no novelty claim against specific systems until it is.*

## 8. Limitations

* The core benchmark is synthetic and authored by the detector's authors; the real-text experiments test text transfer only, and no real corpus of poisoned memory traces with provenance exists to us.
* Logistic regression matches the forest: the task as generated is nearly linear, so model complexity is not the contribution.
* Provenance does not drive accuracy on the benchmark (§5.3); its value is shown only in a controlled experiment whose ρ is a parameter (§5.4).
* Unseen-family generalisation required real training text; with synthetic data alone it is poor (§5.6).
* Free-form LLM-written text lowers accuracy to near chance and defeats the detectors on note-style attacks (§5.7); this is the main open weakness and the generator is a 1B model with limited diversity.
* End-to-end results use one 8B local agent, one run, and a scenario set written by the authors; the write proposer is simulated (the injection is assumed to have produced a write). Defences were compared post hoc on the same scenarios, with no paired test and no correction for multiple comparisons.
* The semantic anchors (exfiltration, promotion, trigger, payment, override) were written by the authors after seeing the benchmark's attack goals, so leave-one-family-out results are not a test of truly unseen intent.
* Several confidence intervals in §5.1-5.4 are t-intervals over overlapping CV folds and are too narrow (Nadeau-Bengio); the ablation uses a single repeat. Differences of about one point there should not be read as significant.
* The stealth experiment uses small, disjoint entity pools and a benign drift/ordinary split that is not learnable from text, which caps accuracy; it supports the *direction* of the provenance-reliability effect, not its size.
* Adversarial-training results at 1.00 evaluate a model on the transforms it was trained on; only the leave-one-evasion-out numbers are evidence of transfer.
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
