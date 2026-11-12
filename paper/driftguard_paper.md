# DriftGuard: Provenance-Aware Write-Time Detection of Memory Poisoning versus Benign Personalisation Drift in LLM Agents

**Authors:** Tanisha Talathi, Aryaan Sethi, Rishabh Gaikwad — Dept. of Computer Engineering, DCET, MIT World Peace University, Pune.
**Status:** working draft. Every number is produced by a script in `scripts/` (see §9) and stored in `results*/`. Citation details must be verified before submission.

---

## Abstract

LLM agents with persistent memory can be steered by *memory poisoning*: an attacker plants an entry that later changes the agent's behaviour. Blocking every memory change would also block the personalisation users want, because genuine preference *drift* ("I moved to Bengaluru") looks structurally like an attack. We study **write-time three-class classification** — malicious, benign drift, ordinary learning — with actions quarantine / admit-and-flag / admit. Our detector scores each candidate write with 18 features (provenance, temporal, consistency, content) plus 7 semantic attack-intent features, using a calibrated random forest.

We report results the way a sceptical reviewer would demand. (1) On a synthetic benchmark the model reaches 0.918 ± 0.018 accuracy under repeated cross-validation after label-preserving *hardening* (corrected 95 % CI [0.890, 0.944]) (0.997 on the raw benchmark, which we show is inflated by generator shortcuts). (2) **We found and fixed a label leak in our own feature pipeline**: session age was computed against the wall clock at extraction time and, because the benchmark is generated class by class, identified the class almost perfectly (AUROC 0.94 on random labels); the burst-window features were also dead in training. All numbers here are post-fix. (3) A redesigned, entity-grouped stealth experiment (5 datasets, intervals) shows provenance helps only in proportion to its reliability: at reliability 0.5 it is at chance (AUROC 0.49) and adds nothing; at 0.8 it adds about 0.09 AUROC over text alone (0.851 vs 0.759) and at 0.95 provenance alone reaches 0.95. (4) A combined adaptive attacker cuts detection to 0.23; adversarial augmentation restores 1.00, and holding out each evasion type still gives 0.90–1.00. (5) Leave-one-attack-family-out detection is weak (0.31 on MINJA) and the apparent fix from adding real injection text (0.89) is confounded: on a length-and-prefix-matched real-text set the synthetic-trained models are at chance (AUROC 0.51-0.57) and a TF-IDF baseline reaches 1.00, so we do not claim real-text transfer. (6) On free-form text written by an independent LLM, with context leakage removed, the detector trained only on templates falls to 0.37-0.52 accuracy and misses 57-77 % of note-style attacks; training on 450 LLM-written notes (v3) lifts malicious-vs-rest AUROC on held-out writers and goals to 0.80-0.92 and cuts false alarms (still 16-35 %), but a plain TF-IDF text classifier reaches 0.88-0.99, so the gain is largely vocabulary. (7) In an end-to-end study with a local 8B agent (300 scenarios, Wilson intervals) every defence reduces attack success from 0.96, but a simple rule-based filter is at least as good as DriftGuard (ASR 0.24 vs 0.36 for v2, overlapping intervals); we do not claim an end-to-end advantage.

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

**Corrected statistics (`scripts/run_stats.py`).** The ± values above are plain t-intervals over 15 overlapping folds and are too narrow. Re-running with 10 x 5-fold repeats (50 folds), the Nadeau-Bengio corrected resampled t interval, and Holm correction across the 11 comparisons gives, on the hardened benchmark:

| Model | Accuracy (corrected 95 % CI) | Diff. vs full RF (CI) | Holm p |
|---|---|---|---|
| DriftGuard RF | 0.917 [0.890, 0.944] | - | - |
| Logistic regression | 0.909 [0.883, 0.935] | +0.008 [-0.014, +0.031] | 1.0 |
| Rule-based | 0.557 [0.532, 0.581] | +0.360 [+0.327, +0.394] | 1.6e-25 |
| RF without provenance | 0.910 [0.884, 0.935] | +0.007 [-0.006, +0.020] | 1.0 |
| RF without temporal | 0.920 [0.893, 0.948] | -0.003 [-0.016, +0.009] | 1.0 |
| RF without consistency | 0.917 [0.890, 0.944] | +0.000 [-0.020, +0.020] | 1.0 |
| RF without content | 0.703 [0.670, 0.737] | +0.214 [+0.173, +0.255] | 2.6e-13 |
| RF only content | 0.914 [0.885, 0.942] | +0.003 [-0.020, +0.027] | 1.0 |

With **grouped** CV (texts identical once digits are masked are never split across train and test) accuracy is 0.909 [0.882, 0.937], so template memorisation does not explain the result. Conclusions the data support: RF beats the rule baseline by a wide margin; **RF and logistic regression are statistically indistinguishable; the provenance, temporal and consistency families do not significantly change accuracy; only the content features do.** The earlier single-repeat ablation differences of about one point (0.908 vs 0.918, 0.927 vs 0.918) are not significant. Accuracy here is on a synthetic benchmark we generated, so it bounds nothing about real traffic.

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

On this synthetic data, content carries almost all the signal and removing provenance costs one point, which the corrected test shows is not significant (diff +0.007, CI [-0.006, +0.020]). We therefore do **not** claim provenance drives accuracy on the benchmark; §5.4 asks when it should.

### 5.4 When does provenance help? (stealth attacks)

Stealth attacks read like legitimate preferences ("User prefers shopping for laptops at ⟨site⟩"); matched benign writes use the same templates with ordinary entities, including URLs and e-mail addresses. Reliability ρ = P(attack via untrusted origin) = P(benign via user origin). All non-text context comes from random synthetic entries so that at ρ = 0.5 every non-text feature is uninformative by construction (verified: provenance-only AUROC 0.487).

*First version (kept for transparency; superseded below).* A 3-class experiment on one generated dataset gave text-only accuracy 0.62-0.64 and AUROC 0.94, and text + provenance 0.717 / AUROC 0.986 at rho = 0.95. A review found that attack and benign writes used partly different templates (3 of 6 each), so the 0.94 reflects template wording, benign drift/ordinary labels were not learnable from text (capping 3-class accuracy), entities could recur across train and test folds, and the reported spread was a fold standard deviation on a single dataset.

**Redesigned experiment (`scripts/run_stealth_v2.py`).** Binary attack-vs-benign; `tau` is the fraction of writes that use class-specific templates (`tau = 0`: identical templates, so only the entity - site, mailbox or brand - differs); folds are grouped by entity so test entities are never seen in training; five independently generated datasets of 400 writes, with t-intervals over the five dataset means. Attack AUROC (95 % CI):

| tau | rho | text only | provenance only | text + provenance |
|---|---|---|---|---|
| 0.0 | 0.5 | 0.761 [0.712, 0.810] | 0.494 [0.461, 0.528] | 0.731 [0.668, 0.793] |
| 0.0 | 0.8 | 0.759 [0.698, 0.820] | 0.801 [0.780, 0.823] | 0.851 [0.810, 0.891] |
| 0.0 | 0.95 | 0.758 [0.707, 0.809] | 0.950 [0.935, 0.964] | 0.964 [0.941, 0.986] |
| 0.5 | 0.5 | 0.878 [0.839, 0.917] | 0.468 [0.427, 0.508] | 0.868 [0.837, 0.899] |
| 0.5 | 0.8 | 0.880 [0.859, 0.902] | 0.792 [0.775, 0.810] | 0.925 [0.909, 0.941] |
| 0.5 | 0.95 | 0.888 [0.863, 0.914] | 0.952 [0.936, 0.967] | 0.977 [0.966, 0.988] |

Reading: **provenance is exactly as useful as it is reliable.** At rho = 0.5 it is at chance (AUROC 0.47-0.49) and adding it does not help (text + provenance is no better than text only, nominally worse). At rho = 0.8 it adds about 0.09 AUROC over text alone when templates are identical (0.851 vs 0.759, intervals do not overlap) and about 0.045 when templates differ; at rho = 0.95, provenance alone (0.95) beats text alone (0.76-0.89). Caveat: even with identical templates text-only AUROC is 0.76, because the 18 text features pick up surface properties of the entity strings (length, characters, top-level domain) - it is not reputation, which the features cannot see - so "text only" overstates what a content detector would manage against a perfectly stealthy attacker. The detector trained on the standard benchmark scores near chance (accuracy 0.43-0.46) on the first-version stealth set.

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

### 5.7b Training on LLM-written memory notes (v3)

To address §5.7 we generated memory notes with local LLMs (`scripts/generate_notes.py`) and trained **v3** = v2 + 450 notes written by llama3.2:1b (150 per class: note-style, injected-sentence and first-person attack styles; drift and ordinary notes in note and first-person styles) about *training* attack goals and topics. Held-out sets use disjoint goals/topics: **A** llama3.2:1b, new goals (180 notes); **B** phi3.5 (an unseen writer of a different model family), note-style (90); **C** phi3.5, first-person voice, attacks and benign alike (90). Context features come from random benchmark entries of any class, so only the text carries signal. Attack recall, false-alarm rate (benign quarantined), 3-class accuracy and malicious-vs-rest AUROC, with Wilson 95 % intervals on the rates (`scripts/train_v3.py`):

| Set | Model | Attack recall | False alarm | 3-class acc. | AUROC |
|---|---|---|---|---|---|
| Benchmark test | v2 | 1.00 [0.91, 1.00] | 0.05 [0.02, 0.12] | 0.93 | 1.00 |
| Benchmark test | **v3** | 1.00 [0.91, 1.00] | 0.01 [0.00, 0.07] | 0.93 | 1.00 |
| A (seen writer, new goals) | v2 | 0.40 [0.29, 0.53] | 0.23 [0.16, 0.31] | 0.49 | 0.68 |
| A | **v3** | 0.52 [0.39, 0.64] | 0.16 [0.10, 0.23] | 0.63 | 0.80 |
| A | TF-IDF, text only | 0.47 [0.35, 0.59] | 0.00 [0.00, 0.03] | 0.49 | 0.92 |
| B (unseen writer) | v2 | 0.77 [0.59, 0.88] | 0.57 [0.44, 0.68] | 0.41 | 0.63 |
| B | **v3** | 0.90 [0.74, 0.97] | 0.35 [0.24, 0.48] | 0.57 | 0.84 |
| B | TF-IDF, text only | 0.37 [0.22, 0.55] | 0.02 [0.00, 0.09] | 0.44 | 0.88 |
| C (unseen writer + style) | v2 | 0.97 [0.83, 0.99] | 0.68 [0.56, 0.79] | 0.48 | 0.76 |
| C | **v3** | 1.00 [0.89, 1.00] | 0.35 [0.24, 0.48] | 0.67 | 0.92 |
| C | TF-IDF, text only | 0.83 [0.66, 0.93] | 0.00 [0.00, 0.06] | 0.61 | 0.99 |

On the length-matched real-text set of §5.6 (v2 and v3 both scored on the same 353 strings neither saw in training): v3 AUROC 0.87, attack TPR 0.58, false alarm 0.04, versus v2 0.79 / 0.49 / 0.10.

What this shows, and what it does not. Training on LLM-written notes helps on every held-out set: AUROC rises by 0.12-0.21 and false alarms fall from 0.57-0.68 to 0.35 on the unseen writer, with no loss on the benchmark. It does **not** make the detector good: a third of benign free-form notes are still quarantined, and **a plain bag-of-words classifier trained on the same notes reaches higher AUROC (0.88-0.99) and far fewer false alarms**, so much of the gain is vocabulary of LLM-written attacks rather than structure, provenance or consistency signal. The sets are small (90-180 notes), all notes come from small local models, and attack goals are drawn from the same families of intent; an adaptive human attacker has not been tested. v3 is therefore evidence that the 25-feature detector can learn from realistic text, and a reminder that a text baseline must always be reported alongside it.

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

**Why runtime attribution looks better than oracle origin here, and why that is not real.** For the adversarially trained model, ASR@10 is 0.08 with runtime attribution but 0.30 with oracle origin, which is backwards. A diagnostic (`robust`, all usable rewrites) shows why: 175 of 179 paraphrased rewrites lose their token overlap with the source and fall back to `agent_reflection`, an origin that adversarial training taught the model to treat as malicious when combined with attack-like text (3 % admitted). Under the oracle, attacks delivered through the user turn keep the `user_turn` origin and are admitted 57 % of the time, against 9-11 % for tool-output and retrieved-document attacks. So the oracle column is the meaningful one, user-turn delivery is the hard case (provenance cannot help when the attacker speaks as the user), and the runtime column mostly measures an attribution fallback. We report the runtime column for completeness and do not draw conclusions from it.

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
* Free-form LLM-written text lowers accuracy to near chance and defeats the detectors on note-style attacks (§5.7); this is the main open weakness. Training on LLM-written notes (v3, §5.7b) improves it, but a TF-IDF baseline is still as good; generators are small local models with limited diversity and the test sets are small.
* End-to-end results use one 8B local agent, one run, and a scenario set written by the authors; the write proposer is simulated (the injection is assumed to have produced a write). Defences were compared post hoc on the same scenarios, with no paired test and no correction for multiple comparisons.
* The semantic anchors (exfiltration, promotion, trigger, payment, override) were written by the authors after seeing the benchmark's attack goals, so leave-one-family-out results are not a test of truly unseen intent.
* Several confidence intervals in §5.1-5.4 are t-intervals over overlapping CV folds and are too narrow (Nadeau-Bengio); the ablation uses a single repeat. Differences of about one point there should not be read as significant.
* The stealth experiment uses small, disjoint entity pools and a benign drift/ordinary split that is not learnable from text, which caps accuracy; it supports the *direction* of the provenance-reliability effect, not its size.
* Adversarial-training results at 1.00 evaluate a model on the transforms it was trained on; only the leave-one-evasion-out numbers are evidence of transfer.
* Single-split numbers vary: a 20 % split gave 7.5 % false alarms where 5×3 CV gives 2.8 %; we report CV.
* No user study of review burden.

## 9. Reproducibility

```
python -m scripts.generate_benchmark                 # raw benchmark -> data/benchmark.jsonl
python -m scripts.generate_benchmark --hard --out data/benchmark_hard.jsonl   # hardened benchmark
python -m scripts.generate_benchmark --harden-file data/held_out.jsonl --out data/held_out_hard.jsonl   # hardened held-out set
python -m scripts.train_detector --benchmark data/benchmark_hard.jsonl --model-out models/classifier_hard.pkl
python -m scripts.download_external && python -m scripts.train_v2
python -m scripts.run_research_eval                  # 5.1 (plain CIs), 5.3
python -m scripts.run_stats                          # 5.1 corrected statistics, grouped CV, Holm
python -m scripts.run_stealth_eval && python -m scripts.run_stealth_v2   # 5.4 (v1 superseded by v2)
python -m scripts.run_adversarial && python -m scripts.train_robust && python -m scripts.run_transform_holdout   # 5.5
python -m scripts.run_generalisation && python -m scripts.run_real_text_eval && python -m scripts.run_real_text_controlled   # 5.6
python -m scripts.run_independent_eval && python -m scripts.run_independent_eval --attack-as-note   # 5.7 (needs Ollama)
python -m scripts.generate_notes && python -m scripts.train_v3   # 5.7b (needs Ollama: llama3.2:1b, phi3.5)
python -m scripts.run_blackbox_attack                # 5.8 (needs Ollama)
python -m scripts.run_agent_study --n-per 100 --agent-model llama3:latest --out results_research/agent_big   # 5.9
python -m scripts.make_manifest                      # git SHA, library versions, data hashes, Ollama digests
```
Generation seeds are fixed, but LLM-written text and agent behaviour depend on the local Ollama model builds; their digests and the hashes of all data files are recorded in `results_research/manifest.json`. `data/held_out.jsonl` (two attack families unseen in training) was produced in an earlier session and is not regenerated by a single command here; its hash is in the manifest. Data files under `data/` and models under `models/` are git-ignored: regenerate them with the commands above (the LLM-written sets are cached in `data/independent_texts*.jsonl` and `data/llm_notes.jsonl`). 126 unit and regression tests run with `pytest`.

## 10. Future work

Real memory-poisoning traces with provenance from design partners; taint tracking inside the model runtime rather than lexical grounding; retrieval-time filtering as a second layer; attacks authored by people who have not seen the detector; a user study of review burden.

## References (verify before submission)

[1] Dong et al., "A Practical Memory Injection Attack against LLM Agents" (MINJA), 2025.
[2] Chen et al., "AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases", NeurIPS 2024.
[3] Greshake et al., "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection", 2023.
[4] Geirhos et al., "Shortcut Learning in Deep Neural Networks", Nature Machine Intelligence, 2020.
[5] OWASP, "Top 10 for LLM Applications".
