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
* New models (git-ignored): `models/classifier_v3.pkl` (+ `.sig`). v2 is still the default in the UI and service; v3 is better on every held-out set I measured, switch when you have checked it on the demo presets.

## Still open (needs a human or more time)
1. **Licence / IP**: no LICENSE file; decide who owns the code (MIT-WPU or the team).
2. **Related work**: the paper has no sourced survey, and its reference list is unverified.
3. **Report and slide deck**: still quote the old pre-fix numbers.
4. **Detector quality**: the strongest next step is more diverse, human-written or multi-model note data and a proper comparison against a fine-tuned text classifier; a TF-IDF model currently matches v3.
5. **Not done**: wheel still ships `eval`/`scripts`/`app` as top-level packages; GitHub Actions pinned by tag, not SHA; `pip-audit` is advisory; models are still pickles (HMAC-signed); `SECURITY.md` lists the remaining open security items.
6. v3 was not run through the Streamlit presets; the earlier v2 mislabelled two drift presets as ordinary.

## Caveats you should repeat if you present this
* Every benchmark number is on data we generated. LLM-written test sets are small (90-180 notes) and written by small local models.
* Defences in the agent study were compared post hoc on the same scenarios with no paired test.
* The semantic anchors were written by us after seeing the attack goals.

Reproduce everything with the commands in section 9 of the paper; `results_research/manifest.json` records the commit, library versions, data hashes and Ollama model digests.
