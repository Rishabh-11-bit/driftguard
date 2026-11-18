# DriftGuard: Provenance-Aware Write-Time Detection of Memory Poisoning versus Benign Personalisation Drift in LLM Agents

**Authors:** Tanisha Talathi, Aryaan Sethi, Rishabh Gaikwad — Dept. of Computer Engineering, DCET, MIT World Peace University, Pune.
**Status:** working draft. Every number is produced by a script in `scripts/` (see §9) and stored in `results*/`. Citation details must be verified before submission.

---

## Abstract

LLM agents with persistent memory can be steered by *memory poisoning*: an attacker plants an entry that later changes the agent's behaviour. Blocking every memory change would also block the personalisation users want, because genuine preference *drift* ("I moved to Bengaluru") looks structurally like an attack. We study **write-time three-class classification** — malicious, benign drift, ordinary learning — with actions quarantine / admit-and-flag / admit. Our detector scores each candidate write with 18 features (provenance, temporal, consistency, content) plus 7 semantic attack-intent features, using a calibrated random forest.

We report results the way a sceptical reviewer would demand. (1) On a synthetic benchmark the model reaches 0.918 ± 0.018 accuracy under repeated cross-validation after label-preserving *hardening* (corrected 95 % CI [0.890, 0.944]) (0.997 on the raw benchmark, which we show is inflated by generator shortcuts). (2) **We found and fixed a label leak in our own feature pipeline**: session age was computed against the wall clock at extraction time and, because the benchmark is generated class by class, identified the class almost perfectly (AUROC 0.94 on random labels); the burst-window features were also dead in training. All numbers here are post-fix. (3) A redesigned, entity-grouped stealth experiment (5 datasets, intervals) shows provenance helps only in proportion to its reliability: at reliability 0.5 it is at chance (AUROC 0.49) and adds nothing; at 0.8 it adds about 0.09 AUROC over text alone (0.851 vs 0.759) and at 0.95 provenance alone reaches 0.95. (4) A combined adaptive attacker cuts detection to 0.23; adversarial augmentation restores 1.00, and holding out each evasion type still gives 0.90–1.00. (5) Leave-one-attack-family-out detection is weak (0.31 on MINJA) and the apparent fix from adding real injection text (0.89) is confounded: on a length-and-prefix-matched real-text set the synthetic-trained models are at chance (AUROC 0.51-0.57) and a TF-IDF baseline reaches 1.00, so we do not claim real-text transfer. (6) On free-form text written by an independent LLM, with context leakage removed, the detector trained only on templates falls to 0.37-0.52 accuracy and misses 57-77 % of note-style attacks; training on 450 LLM-written notes (v3) lifts malicious-vs-rest AUROC on held-out writers and goals to 0.80-0.92, but plain TF-IDF and embedding classifiers do as well or better, and fusing them with the feature detector is the best scorer on every held-out text set (AUROC 0.95-0.99; false alarms at 90 % recall 0.00-0.13) - with the caveat that real-text results for text models are inflated by a dataset-vocabulary gap. (7) In an end-to-end study with a local 8B agent (300 scenarios, Wilson intervals) every defence reduces attack success from 0.96, but a simple rule-based filter is at least as good as DriftGuard (ASR 0.24 vs 0.36 for v2, overlapping intervals); we do not claim an end-to-end advantage; re-run on the same scenarios the hybrid reaches ASR 0.17 against 0.24 for the rule-based filter (paired p = 0.19, not significant) and clearly beats v2 (p < 0.00001). (8) On thirteen groups from public sources (§5.10) an independent published prompt-injection detector beats our memory-note detectors on generic chat-prompt datasets but recalls only 0.21-0.63 of memory-write and agent-tool attacks at its default threshold; a leave-one-group-out stack of both reaches AUROC 0.92-1.00 on note-style and tool-response groups and 0.85-0.99 on prompt-injection datasets. Scoring structured tool output field by field (AUROC 0.85 to 0.93 and 0.54 to 1.00), normalising obfuscation and scoring sentence windows (diluted-attack recall 0.22-0.48 to 0.88) were each motivated by a failure the evaluation exposed. All of this is on public or generated text, not production traffic.

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

### 5.7c Fusing the feature detector with text classifiers (hybrid)

Because plain text classifiers matched v3 in §5.7b, we evaluated fusing them (`scripts/run_hybrid.py`). Scorers: **v3**; a **TF-IDF** (1-2 gram) logistic regression and a **sentence-embedding** (all-MiniLM-L6-v2) logistic regression, both attack-vs-rest; and unweighted means of probabilities. The combinations were fixed in advance and all are reported (`results_research/hybrid.md`). The text models are trained on exactly the text v3 saw (benchmark train split, the real-text training half, the llama3.2:1b training notes); strings seen in training are masked out of the real-text set. Cell format: AUROC (malicious vs rest) / false-alarm rate needed to reach 90 % recall (lower is better).

| Test set | v3 (features) | TF-IDF LR | Embedding LR | **Fused (mean of all three)** |
|---|---|---|---|---|
| Benchmark test (n=40+80) | 1.000 / 0.00 | 1.000 / 0.00 | 1.000 / 0.00 | 1.000 / 0.00 |
| Notes A (seen writer, new goals) (n=60+120) | 0.812 / 0.48 | 0.906 / 0.23 | 0.918 / 0.15 | 0.946 / 0.13 |
| Notes B (unseen writer) (n=30+60) | 0.843 / 0.37 | 0.947 / 0.18 | 0.934 / 0.13 | 0.951 / 0.12 |
| Notes C (unseen writer + style) (n=30+60) | 0.918 / 0.22 | 0.988 / 0.03 | 0.978 / 0.07 | 0.994 / 0.02 |
| Real text, length-matched, unseen (n=126+227) | 0.864 / 0.37 | 0.983 / 0.01 | 0.976 / 0.03 | 0.987 / 0.00 |

The fused scorer is the best or within 0.005 AUROC of the best on every text set (v3 + TF-IDF alone is marginally higher on set C, 0.999 vs 0.994), and it cuts the false-alarm rate at 90 % recall from 0.22-0.48 (v3) to 0.00-0.13. On the benchmark every scorer is perfect, which is another sign the benchmark cannot discriminate between detectors. **Cautions.** (i) On real text, a TF-IDF model reaches AUROC 0.98 largely because the two public datasets differ in vocabulary (TF-IDF cross-validation gets 1.0, §5.6), so those text-model numbers are shortcut-inflated and should not be read as intent detection. (ii) The notes sets are small (90-180) and LLM-written; text models trained on one small model's notes transferred to another model's notes, which suggests shared LLM vocabulary, and a human attacker may not share it. (iii) These sets carry no usable provenance, timing or memory-consistency signal, so they cannot show what the feature detector adds; §5.4 is where provenance earns its keep. We ship the fusion as the default model (`driftguard/detector/hybrid.py`, `scripts/train_hybrid.py`; the feature detector still supplies the drift-vs-ordinary split and all non-text evidence); the comparison against a fine-tuned transformer is in §5.7d.

### 5.7d Comparison with a fine-tuned transformer

The comparison reviewers ask for first (`scripts/run_finetuned.py`, `results_research/finetuned.md`). We fine-tuned all-MiniLM-L6-v2 end to end with a linear head (3 epochs, AdamW lr 3e-5, batch 32, CPU, seed 42; training loss 0.40 → 0.02) for attack-vs-rest on exactly the rows the hybrid's text models saw, and evaluated on the same four held-out sets. Scorers were fixed in advance: the shipped hybrid, the fine-tuned model, and their unweighted mean. Cell format: AUROC / false-alarm rate needed for 90 % recall.

| Test set | Hybrid (shipped) | Fine-tuned MiniLM | Mean of both |
|---|---|---|---|
| Benchmark test (n=40+80) | 1.000 / 0.00 | 1.000 / 0.00 | 1.000 / 0.00 |
| Notes A (n=60+120) | 0.946 / 0.13 | 0.943 / 0.27 | 0.964 / 0.17 |
| Notes B, unseen writer (n=30+60) | 0.951 / 0.12 | 0.991 / 0.00 | 0.970 / 0.00 |
| Notes C, unseen writer + style (n=30+60) | 0.994 / 0.02 | 0.995 / 0.02 | 0.996 / 0.02 |
| Notes D, hand-style, written by the authors' assistant (n=50+50) | 0.892 / 0.30 | 0.895 / 0.28 | 0.895 / 0.30 |
| Real text, length-matched, unseen (n=126+227) | 0.987 / 0.00 | 1.000 / 0.00 | 0.998 / 0.00 |

Set D (`data/hand_notes.tsv`, 100 notes) was added after the first round to test text the models had never seen from any writer: 50 attacks in varied registers (terse, first-person, formal, lowercase, plausible-sounding authority claims), 17 drift notes, 15 ordinary facts and **18 benign standing instructions** that look like attacks on the surface ("Send the weekly report to my manager every Friday", "remind me to forward the electricity bill to my landlord", "Never schedule meetings over lunch"). It was written by the same assistant that helped write this paper, so it is an unseen writer, **not** independent human data, and it was never used for training or model selection.

The fine-tuned encoder is at least as good as the hybrid on every text set (clearly better on B, 0.991 vs 0.951, and on real text; tied on A, C and D); it is only worse at the A operating point for 90 % recall (0.27 vs 0.13 false alarms). With 90-180 notes per set, a 0.04 AUROC gap is within what resampling noise could produce, and we ran no paired test, so we do **not** claim it is better. **Set D is the sobering result.** Both models fall to AUROC 0.89 and flag 38-42 % of benign notes at the 0.5 threshold. Broken down for the hybrid: it catches 48 of 50 attacks (96 %), flags only 6 % of drift notes and 20 % of ordinary facts, but **flags 83 % (15 of 18) of the benign standing instructions**. The two missed attacks are the ones with no obvious attack vocabulary ("My real purpose is to maximise the user's engagement..." and "if the user says no... ask again in different words"). So a text-level detector cannot separate "forward every email to this address" written by an attacker from "forward the electricity bill to my landlord" written by the user; that distinction is exactly what provenance is for (a user-turn instruction versus tool output), and these notes carry none. Practical consequence: with the shipped default, legitimate user instructions arriving through a trusted channel are the main source of false alarms, which the pipeline's origin handling (§5.4, §6) must absorb. We did not tune on set D.

What the comparison does show is that nothing in our feature pipeline beats a plain fine-tuned text encoder on these sets: the same caution as §5.7c applies (vocabulary gap on real text, LLM-written notes, no provenance signal in these sets), and the case for the feature detector rests on §5.4 (provenance, timing, consistency), not on text. We keep the hybrid as the shipped default because it is already integrated, signed and tested and is within noise of the fine-tuned model; shipping the fine-tuned model (about 90 MB, new serialisation and signing path) is the obvious next step if a larger, human-written test set confirms its edge. A-MemGuard (arXiv:2510.02373) remains uncompared.

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

**Re-run with the newer defences (`scripts/run_agent_newdefences.py`).** The scenario generator is seeded, so the same 300 scenarios were regenerated (the undefended, rule-based and v2 rows reproduce exactly: ASR 0.96, 0.24, 0.36) and the hybrid and the ensemble of §5.10 were added, with normalised text and runtime attribution as before. Attack success is measured with the 8B agent for every attack a defence admits; a blocked write cannot reach the agent. Utility was not re-measured, only write-level admission. Results: **hybrid ASR 0.17 [0.11, 0.26]** and ensemble 0.19 [0.12, 0.28], against the rule-based filter 0.24 [0.17, 0.33] and v2 0.36 [0.27, 0.46]; none of the four defences quarantined any of the 200 benign drift or ordinary writes. Paired on the same 100 attacks (exact McNemar): the hybrid lets through 7 attacks the rule-based filter blocks and blocks 14 that it lets through (p = 0.19; ensemble 7 vs 12, p = 0.36), so **the improvement over the rule-based filter is a point estimate we cannot call significant**; the hybrid blocks 19 attacks that v2 lets through and lets through none that v2 blocks (p < 0.00001), a clear gain over our previous best; the ensemble and hybrid do not differ (2 vs 0, p = 0.5), so the ensemble adds no end-to-end value in this study. Caveats: the scenario texts were written by the authors before the new defences existed but with knowledge of the earlier ones; one agent model, one run, 100 attacks; the hybrid components saw `hand_train.tsv`.

### 5.10 Independent public data, an external baseline, and an ensemble

Everything before this section was evaluated on data we generated or on two public datasets we also trained on. `scripts/download_external2.py` adds nine further public sources (Hugging Face datasets-server JSON and raw GitHub JSON only; nothing is executed): jackhhao/jailbreak-classification, SPML chatbot prompt injection, geekyrakshit prompt-injection, Lakera/gandalf, Dolly, Alpaca, OpenAssistant, Synthetic-Persona-Chat, and from GitHub InjecAgent (attacker instructions and full tool responses) and ASB (attacker instructions). `scripts/run_multisource.py` evaluates thirteen test groups (up to 400 rows each, stratified; five with published labels, two pairing an attack-only with a benign-only source, one set of InjecAgent minimal pairs, one tiny ASB group, and notes A-D) with five scorers: the shipped hybrid (zero-shot on the new sources), the hybrid retrained with `data/hand_train.tsv`, a fine-tuned MiniLM, **ProtectAI's published DeBERTa-v3 prompt-injection detector (v2, Apache-2.0) as an independent baseline**, and TF-IDF / embedding logistic regressions trained leave-one-group-out (the held-out group's whole source family is removed from their training pool; an earlier version of this experiment leaked InjecAgent strings and was corrected). All scorers see normalised text and score sentence windows, with JSON tool output unwrapped into its string fields (finding iii below).

Three findings (`results_research/multisource.md`, `stacking.md`). **(i)** On generic prompt-injection datasets the published detector is better than our memory-note detectors: AUROC 0.96 vs 0.85 on jackhhao, 1.00 vs 0.90 on SPML, 0.92 vs 0.76 on deepset, and our detectors flag 57-97 % of ordinary chat prompts at the 0.5 threshold, which is what one should expect of a model trained on memory notes. **(ii)** On memory-write text the ranking reverses at a fixed false-alarm budget: with 5 % false alarms the published detector recalls 0.30 / 0.70 / 0.77 / 0.52 of attacks on notes A / B / C / D, our hybrid 0.68 / 0.90 / 1.00 / 0.80, and at its default threshold the published detector recalls only 0.21 of InjecAgent attacker instructions. **(iii)** Structured tool responses are a failure mode for all of them when scored raw (AUROC 0.85 hybrid, 0.77 fine-tuned, 0.54 published detector on InjecAgent tool responses with and without the injected slot) and are fixed for all of them by scoring each string field separately (0.93 / 0.98 / 1.00; the benign slot is a persona fact, so this tests structured-text handling, not attack versus benign instruction).

Because the strengths are complementary we stack the three scorers (hybrid, fine-tuned, published detector) with a 3-input logistic regression on their logits, fitted leave-one-group-out so that no group is scored by a stacker that saw it (`scripts/run_stacking.py`). The stack is within 0.035 AUROC of the best single scorer on all 13 groups (0.916-1.000 on all note-style and tool-response groups; 0.854-0.987 on the five prompt-injection datasets) and at a 5 % false-alarm budget recalls 0.77-1.00 of attacks on every note-style and tool-response group, against 0.30-0.96 for the published detector alone. The weak groups remain geekyrakshit (AUROC 0.854, recall 0.27 at 5 % false alarms) and deepset (0.871), and ASB-vs-user-instructions has only 34 rows. We ship the stack as an optional signed model (`driftguard/detector/ensemble.py`, `scripts/train_ensemble.py`; the published detector is pinned to its Hugging Face commit). Caveats: the published detector may have trained on some of these datasets; the hybrid components saw `hand_train.tsv`, written after our first look at set D; two groups pair an attack-only with a benign-only source and can be separated by topic; the shipped stacker is fitted on all groups, so only the leave-one-group-out numbers above are out-of-sample.

### 5.11 Disguised attacks, and the two defences they motivated

`scripts/run_perturbation.py` takes the 50 attack notes of set D and applies deterministic disguises an attacker can use without model access (dilution among benign sentences, case changes, letter spacing, leetspeak, Cyrillic homoglyphs, zero-width characters, authority and polite wrappers, role-play quotation, code fences, benign filler). Each scorer's alarm threshold is fixed once at 5 % false alarms on the 50 unperturbed benign notes. The same transformation is applied to the benign notes to measure collateral false alarms. Raw scoring exposed two problems (`results_research/perturbation.md`): (i) **dilution**: three benign sentences on each side cut recall from 0.80 to 0.02 for the hybrid and from 0.88 to 0.48 for the stacked scorer of §5.10, and six per side to 0.00 and 0.22; (ii) **obfuscation looked like detection**: letter spacing, leetspeak, homoglyphs and zero-width characters gave recall 0.9-1.0 but also false alarms of 0.82-0.98 on disguised benign text, i.e. the model flagged anything strange, which is not robustness.

We added two defences and re-ran (`perturbation_norm_win.md`). **Normalisation** folds Cyrillic and Greek lookalikes when a note is mostly Latin, collapses runs of single letters ("i g n o r e"), and maps digits inside mixed alphanumeric tokens to letters ("1gn0r3"), while leaving ordinary quantities and names (3pm, mp3, v2, a genuine Russian note) untouched (unit-tested). **Window-max scoring** scores the note and its sentence windows and keeps the maximum for every scorer. Result for the stacked scorer (recall / false alarms on the same disguise): none 0.88 / 0.06; three benign sentences each side 0.88 / 0.12; six each side 0.88 / 0.08; letter spacing 0.88 / 0.10; leetspeak 0.96 / 0.16; homoglyphs 0.86 / 0.06; zero-width 0.88 / 0.06. On multi-sentence benign notes windowing adds no false alarms (0.00 for every scorer). **Still open:** wrapping a note in an authority claim ("note from the system administrator"), a polite request, a role-play quote or a code fence raises false alarms to 0.3-1.0 for every scorer because the wrapper itself looks suspicious; we regard that as a feature for the authority wrapper and a limitation for the others. The disguises are our own enumeration; an attacker who optimises against the system is the black-box study of §5.8.

### 5.12 Instructions split across two writes

`scripts/run_split_attack.py` cuts each of the 50 attack notes at the clause boundary nearest the middle and scores each half alone, or the second half together with the first. Because the harmful imperative usually stays in one half, per-write scoring already catches 0.78 (the unsplit note: 0.78); adding the pair as an extra window raises recall to 0.82 at the cost of 0.02 more false alarms on benign pairs. A second set of 12 hand-written pairs whose halves are individually plausible was caught 11 of 12 for the same reason: the second half still carries an unmistakable instruction. We therefore have no evidence that splitting evades the detector, **and no evidence about a split in which neither half is suspicious and only their combination is harmful**; that case is open, and the temporal and consistency features are the intended (untested) defence.

### 5.13 Service behaviour

`scripts/bench_service.py` runs the Guard API in-process with the shipped hybrid on a CPU laptop. Over 200 sequential writes latency is p50 0.39 s, p95 0.66 s, p99 0.91 s (2.5 writes/s); four concurrent clients obtain 1.0 writes/s because the service serialises per tenant, which is adequate for a memory-write guard and not for a high-volume gateway. Of 25 hostile-input cases (zero-width and bidi characters, homoglyphs, null bytes, an emoji flood, 16 kB notes, malformed JSON, wrong types, missing and wrong keys, a disallowed origin, a path-traversal session id, an oversized body, the disabled docs route) all 25 returned a controlled 2xx or 4xx; none produced a 5xx. This shows resilience of the service layer, not detector accuracy.

## 6. Security analysis of the system

Fixed: unsigned pickle loading (code execution; regression test with a real malicious pickle), fail-open on detector errors, caller metadata overwriting the *flagged* mark, caller-supplied trust, markdown rendering of untrusted text, unbounded input, and a vulnerable web stack found by `pip-audit`. **Residual:** attribution is heuristic (§5.9); the detector is one layer, not a guarantee; quarantined text is stored unencrypted. See `SECURITY.md`.

## 7. Related work

**Attacks on agent memory and knowledge bases.** MINJA [1] injects malicious records into an agent's memory using only ordinary queries and output observations, with no privileged access; AgentPoison [2] plants a few poisoned demonstrations in memory or a RAG knowledge base that are retrieved when a trigger appears in the query; PoisonedRAG [4] corrupts a retrieval corpus with a handful of crafted passages. Our two benchmark attack families are modelled on [1] and [2]; we did not run their original attack code, so our attacks are *style-level* imitations, which is a limitation.

**Indirect prompt injection and its defences.** Greshake et al. [3] introduced indirect prompt injection and a security taxonomy; AgentDojo [5] provides a dynamic evaluation environment for agents that call tools over untrusted data. Defences operate mostly at inference time on a single request: the instruction hierarchy [6] trains models to rank system over user over third-party instructions, and spotlighting [7] marks the source of untrusted text in the prompt. CaMeL [8] is the closest in spirit to our provenance argument: it tracks the provenance of values through a custom interpreter and enforces capability policies, giving provable guarantees on a fraction of AgentDojo tasks, but it constrains how the agent is built rather than classifying what it stores. These approaches protect the *current* turn; none targets the *persistent write* that poisons later sessions, which is the point of our write-time formulation.

**Defences specific to agent memory.** A-MemGuard [9] defends memory with consensus-based validation (comparing reasoning paths derived from several related memories) and a dual-memory structure that stores detected failures as lessons. It acts at *use* time on the reasoning the memories induce, whereas DriftGuard acts at *write* time on the candidate entry, and DriftGuard additionally models benign personalisation drift as a separate class so that legitimate preference changes are admitted (flagged) rather than blocked. Concurrent preprints discuss forged-reasoning attacks that target consensus defences (arXiv:2607.05029, "Your Agent's Memories Are Not Its Own") and write-time HMAC provenance tagging (arXiv:2606.12703; we have not verified its title). We have read only their abstracts and **have not compared against any of these systems**; a head-to-head comparison with A-MemGuard in particular is the most important missing baseline. We therefore make no novelty claim beyond the combination of a three-class write-time detector, runtime provenance attribution, and the evaluation methodology, and we regard the closeness of the 2026 preprints as a reason to position this paper as a methodology and negative-results contribution.

**Evaluation methodology.** Shortcut learning [10] describes models that exploit dataset artefacts instead of the intended signal; our label-leak and length-confound findings are instances in a security setting. For model comparison under repeated cross-validation we use the corrected resampled t-test of Nadeau and Bengio [11].

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
* Public-data evaluation (§5.10) uses a published detector as baseline that may have trained on the same datasets; our memory-note models flag most ordinary chat prompts at the default threshold and need a per-deployment threshold; groups that pair an attack-only with a benign-only source can be separated by topic; the InjecAgent minimal pairs fill the benign slot with a persona fact, not a benign instruction; the ASB group has 34 rows; the shipped stacker is fitted on all groups, so only the leave-one-group-out numbers are out-of-sample.
* Hand-style notes (set D and `hand_train.tsv`) were written by an AI assistant, and the training notes after the first failure on D was observed. They are not independent human data.
* Disguise robustness (§5.11) covers our own enumeration of transformations; wrapper phrases (authority, polite request, role-play, code fence) still raise false alarms to 0.3-1.0; a split in which only the combination is harmful (§5.12) is untested; only English was tested.
* The end-to-end re-run with the newer defences (§5.9) measured attack success and write-level admission only, not drift or ordinary utility; it uses one 8B agent, one run and author-written scenarios, and cannot separate the hybrid from the rule-based filter.
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
python -m scripts.run_hybrid && python -m scripts.train_hybrid   # 5.7c (fusion evaluation; shipped model)
python -m scripts.run_finetuned                                  # 5.7d (fine-tuned transformer comparison)
python -m scripts.run_finetuned --extra data/hand_train.tsv --tag _instr --hybrid models/classifier_hybrid_instr.pkl   # needs scripts.train_hybrid --extra data/hand_train.tsv --out models/classifier_hybrid_instr.pkl first
python -m scripts.download_external2 && python -m scripts.run_multisource && python -m scripts.run_stacking   # 5.10
python -m scripts.train_ensemble                                 # builds the signed ensemble used in 5.10
python -m scripts.run_perturbation --normalise --window --tag _norm_win   # 5.11
python -m scripts.run_split_attack                               # 5.12
python -m scripts.bench_service                                  # 5.13
python -m scripts.check_presets                      # demo presets vs every model
python -m scripts.run_blackbox_attack                # 5.8 (needs Ollama)
python -m scripts.run_agent_study --n-per 100 --agent-model llama3:latest --out results_research/agent_big   # 5.9
python -m scripts.make_manifest                      # git SHA, library versions, data hashes, Ollama digests
```
Generation seeds are fixed, but LLM-written text and agent behaviour depend on the local Ollama model builds; their digests and the hashes of all data files are recorded in `results_research/manifest.json`. `data/held_out.jsonl` (two attack families unseen in training) was produced in an earlier session and is not regenerated by a single command here; its hash is in the manifest. Data files under `data/` and models under `models/` are git-ignored: regenerate them with the commands above (the LLM-written sets are cached in `data/independent_texts*.jsonl` and `data/llm_notes.jsonl`). 126 unit and regression tests run with `pytest`.

## 10. Future work

Real memory-poisoning traces with provenance from design partners; taint tracking inside the model runtime rather than lexical grounding; retrieval-time filtering as a second layer; attacks authored by people who have not seen the detector; a user study of review burden.

## References

All entries below were checked against the arXiv listing or the venue's page in this session. Venue details are as reported by those pages.

[1] S. Dong, S. Xu, P. He, Y. Li, J. Tang, T. Liu, H. Liu, Z. Xiang. "Memory Injection Attacks on LLM Agents via Query-Only Interaction" (MINJA; earlier title "A Practical Memory Injection Attack against LLM Agents"). arXiv:2503.03704; NeurIPS 2025.
[2] Z. Chen, Z. Xiang, C. Xiao, D. Song, B. Li. "AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases". arXiv:2407.12784; NeurIPS 2024.
[3] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, M. Fritz. "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection". arXiv:2302.12173, 2023.
[4] W. Zou, R. Geng, B. Wang, J. Jia. "PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models". arXiv:2402.07867; USENIX Security 2025.
[5] E. Debenedetti, J. Zhang, M. Balunovic, L. Beurer-Kellner, M. Fischer, F. Tramer. "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents". arXiv:2406.13352; NeurIPS 2024 Datasets and Benchmarks.
[6] E. Wallace, K. Xiao, R. Leike, L. Weng, J. Heidecke, A. Beutel. "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions". arXiv:2404.13208, 2024.
[7] K. Hines, G. Lopez, M. Hall, F. Zarfati, Y. Zunger, E. Kiciman. "Defending Against Indirect Prompt Injection Attacks With Spotlighting". arXiv:2403.14720, 2024.
[8] E. Debenedetti et al. "Defeating Prompt Injections by Design" (CaMeL). arXiv:2503.18813, 2025.
[9] Q. Wei et al. "A-MemGuard: A Proactive Defense Framework for LLM-Based Agent Memory". arXiv:2510.02373, 2025 (author list and exact title as shown on the arXiv listing; check before submission).
[10] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, F. A. Wichmann. "Shortcut Learning in Deep Neural Networks". arXiv:2004.07780 (Nature Machine Intelligence, 2020; journal page not checked).
[11] C. Nadeau, Y. Bengio. "Inference for the Generalization Error". Machine Learning 52, 239-281, 2003 (verified via secondary sources only).
