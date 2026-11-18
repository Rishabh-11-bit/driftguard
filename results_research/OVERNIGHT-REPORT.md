# Overnight report (read this first)

All work below is committed and pushed to `main`. Tests: 126 passed, 4 skipped.

## Headline

DriftGuard is now an honest, well-instrumented research system, but the **detector is not strong enough on
realistic text to claim a result at a top venue**, and the paper says so. The defensible contributions are the
write-time architecture, the label-leak finding, the evaluation methodology, and the negative results.

## What the experiments showed (all in `paper/driftguard_paper.md`)

| Question | Answer |
|---|---|
| Accuracy on the synthetic benchmark | 0.917, corrected 95 % CI [0.890, 0.944] (raw 0.997 was inflated). Grouped CV 0.909. |
| Does RF beat logistic regression? | No: difference +0.008, CI [-0.014, +0.031], Holm p = 1. |
| Which features matter? | Content only. Removing provenance, temporal or consistency features is not significant. |
| Does provenance help? | Only in proportion to its reliability: AUROC 0.49 at reliability 0.5, 0.85 at 0.8 (vs 0.76 text only), 0.95 at 0.95. |
| Real injection text (length-matched) | Synthetic-trained models at chance (AUROC 0.51-0.57). The earlier "0.93" was a length effect. TF-IDF alone gets 1.0 on those two datasets. |
| LLM-written note-style attacks | v2 misses most (recall 0.40 on set A). v3 (trained on 450 LLM notes) AUROC 0.80 / 0.84 / 0.92 on held-out sets A/B/C, false alarms still 16-35 %. **A TF-IDF baseline gets 0.88-0.99 AUROC.** |
| Persistent attacker (oracle origin) | ASR@10 0.20 (v2), 0.30 (hardened/robust), 0.58 (raw). User-turn attacks are the hard case (57 % admitted). The low "runtime attribution" numbers were an artifact of an attribution fallback. |
| End-to-end, 8B agent, 300 scenarios | Undefended ASR 0.96. Rule-based filter 0.24, v2 0.36, LLM judge 0.50. **We do not beat the rule-based filter.** |

## Things that changed in the code
* Guard API: separate reviewer keys, Content-Length required, per-tenant quotas, per-tenant lock, per-session write history, key-to-origin binding (`key:tenant@origin|origin`), failed-login throttle, hashed key lookup, signature checked on the loaded bytes, model key required, docs disabled.
* File review queue: atomic rewrite, lock, tolerates corrupt lines. Dockerfile uses `--factory`, healthcheck, documented mount.
* CI: Python 3.11/3.12 matrix, read-only permissions, `bandit` finding annotated and reviewed. Last two CI runs passed.
* `pyproject.toml`: pytest moved to `dev`, streamlit/plotly to a new `app` extra.
* New scripts: `run_stats`, `run_stealth_v2`, `run_real_text_controlled`, `generate_notes`, `train_v3`, `make_manifest`.
* New models (git-ignored): `models/classifier_v3.pkl` and `models/classifier_hybrid.pkl` (+ `.sig`). **The hybrid is now the default** in the UI and the Guard API (see the second-pass update below).

## Still open (needs a human or more time)
1. **Licence / IP**: no LICENSE file; decide who owns the code (MIT-WPU or the team).
2. ~~Related work~~ done in the second pass (11 verified references); a head-to-head comparison with A-MemGuard is still missing.
3. ~~Report~~ an updated copy with a correction notice and Section 16 is on your Desktop (`DriftGuard_Project_Report_v2.docx` / `.pdf`); the midterm deck is a historical artefact and was left alone.
4. **Detector quality**: the strongest next step is more diverse, human-written or multi-model note data and a proper comparison against a fine-tuned text classifier; a TF-IDF model currently matches v3.
5. **Not done**: `pip-audit` is advisory (the dependency tree carries torch/transformers advisories that do not apply to the service); models are still pickles (HMAC-signed); `SECURITY.md` lists the remaining open security items.

## Caveats you should repeat if you present this
* Every benchmark number is on data we generated. LLM-written test sets are small (90-180 notes) and written by small local models.
* Defences in the agent study were compared post hoc on the same scenarios with no paired test.
* The semantic anchors were written by us after seeing the attack goals.

Reproduce everything with the commands in section 9 of the paper; `results_research/manifest.json` records the commit, library versions, data hashes and Ollama model digests.

## Second-pass update
* **Related work**: sourced section and 11 references checked against arXiv/venue pages (paper section 7). Includes A-MemGuard and two 2026 preprints on write-time provenance tagging / forged-reasoning attacks that we have only read at abstract level and have **not** compared against.
* **Hybrid detector (new default)**: averaging P(malicious) of v3, a TF-IDF LR and an embedding LR is the best or within 0.005 AUROC of the best scorer on every held-out text set (AUROC 0.95-0.99; false alarms at 90 % recall 0.00-0.13 vs 0.22-0.48 for v3). Shipped as `driftguard/detector/hybrid.py` (signed pickle, loaded through the same integrity check), wired through `WritePipeline`, the forensic audit, the Guard API and the UI. Caveat in the paper: text-only numbers on real text are inflated by a dataset-vocabulary gap, and these sets cannot show the value of provenance.
* **Presets**: all 11 demo presets are classified correctly by hard, robust, v2, v3 and the hybrid (`python -m scripts.check_presets`). My earlier note that v2 mislabelled two drift presets no longer holds.
* **Packaging / CI**: wheel ships only `driftguard` and `service` (CLI entry points removed; research scripts run with `python -m scripts.<name>` from a checkout); GitHub Actions pinned to commit SHAs.
* **Report**: `DriftGuard_Project_Report_v2.docx/.pdf` (Desktop): the original, with a correction notice at the top of Sections 1 and 9.2 and a new Section 16 (revised results, what the audit found, hardening). The original files are unchanged.
* **Fine-tuned transformer comparison (third pass, paper §5.7d)**: a fine-tuned MiniLM matches or beats the shipped hybrid on 3 of 4 text sets (AUROC 0.943/0.991/0.995/1.000 vs 0.946/0.951/0.994/0.987), within noise given the small sets and no paired test. Hybrid stays the default; the feature detector's value rests on the provenance results (§5.4), not text. A-MemGuard is still uncompared.
* **Set D (100 hand-style notes written by the assistant, `data/hand_notes.tsv`)**: hybrid and fine-tuned both AUROC 0.89-0.90. The hybrid catches 48/50 attacks but flags 15/18 benign *standing instructions* ("send the weekly report to my manager"): text alone cannot tell them from attacks, so the false-alarm problem sits with legitimate user instructions. Not human-written; not used for training or tuning.
* **Third pass (night of 18-19 Nov): independent public data, an external baseline, hardening, an ensemble.** Read `docs/EVIDENCE.md` first; paper §5.10-5.13.
  * Nine more public sources (HF datasets-server JSON, InjecAgent and ASB from GitHub), 13 test groups, ProtectAI DeBERTa as an independent baseline. Our memory-note detectors are *worse* than it on generic prompt-injection datasets and *better* on memory-write text (it recalls 0.21-0.63 at its default threshold). A leave-one-group-out stack of both ranks within 0.035 AUROC of the best scorer on every group.
  * Found and fixed by the evaluation: raw JSON tool output defeats all scorers (DeBERTa AUROC 0.54); unwrapping string fields gives 0.93 / 0.98 / 1.00. A leave-one-out leak across InjecAgent sources was found and fixed.
  * Disguised attacks: dilution cut stacked recall to 0.22-0.48; obfuscation looked like detection (false alarms 0.82-0.98). Added normalisation (confusables, letter spacing, leetspeak) and sentence-window max: recall 0.86-0.96 at false alarms 0.06-0.16. Wrapper phrases still raise false alarms (open).
  * Split-instruction attacks were not an evasion in our test (the harmful half stays suspicious); a split where only the combination is harmful is untested.
  * Service: 25/25 hostile-input cases, no 5xx, p50 0.4-0.5 s, 1-2.5 writes/s.
  * New optional signed `EnsembleDetector` (`models/ensemble.json`); 141 tests pass; 11/11 presets for every model.
  * End to end (same 300 scenarios, 8B agent, paper §5.9): attack success hybrid 0.17, ensemble 0.19, rule-based 0.24, v2 0.36, undefended 0.96. The hybrid beats v2 decisively (paired p < 0.00001) but its edge over the rule-based filter is not significant (p = 0.19). The ensemble adds nothing end to end.
  * Honest negatives kept: no significant win over the rule-based filter; hand-style notes are AI-written; the published detector may have seen some of these datasets; utility was not re-measured in the end-to-end re-run.
  * CI: the new ensemble code tripped bandit B615 (unpinned encoder download); fixed by pinning the MiniLM revision; green on `20bcb65`.
* **Deliberately not done**: adding a LICENSE file (choosing a licence and copyright holder is a legal/ownership decision for you, the team and MIT-WPU), editing the midterm deck, and claiming novelty over the 2026 preprints.
