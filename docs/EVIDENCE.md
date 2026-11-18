# DriftGuard: what we can show, and what we cannot

This page is the ledger of claims. Every number links to a result file and the command that regenerates it.
Nothing here is tuned on its own test data unless it says so. Where a result is unflattering it is kept.

**One-sentence summary.** DriftGuard is a write-time guard for an agent's long-term memory. On four note-style
test sets and one public agent-attack set it separates attacks from benign notes with AUROC 0.96-1.00; on five public
prompt-injection datasets it reaches 0.84-0.99; off-the-shelf prompt-injection detectors miss most memory-write attacks
at their default settings; and the service survives hostile input. It is **not** proven against a determined human
attacker or on real production traffic.

## How the evidence is organised (independent sources first)

| # | Source | Who produced the text | Published labels? |
|---|---|---|---|
| 1 | jackhhao/jailbreak-classification (n=400 used) | public | yes |
| 2 | reshabhs/SPML_Chatbot_Prompt_Injection (400) | public | yes |
| 3 | geekyrakshit/prompt-injection-dataset (400) | public | yes |
| 4 | deepset/prompt-injections (400) | public | yes (our models trained on a slice) |
| 5 | xTRam1/safe-guard-prompt-injection (400) | public | yes (our models trained on a slice) |
| 6 | InjecAgent attacker instructions vs persona facts and user messages (124) | public | paired sources, shortcut risk |
| 7 | Lakera/gandalf attempts vs Dolly/Alpaca instructions (400) | public | paired sources, shortcut risk |
| 8 | Notes A, B, C (LLM-written memory notes, 60-180 each) | llama3.2:1b, phi3.5 | by construction |
| 9 | Notes D (100 hand-style memory notes) | an AI assistant (Claude); **not human** | by construction |

Command: `python -m scripts.download_external2 && python -m scripts.run_multisource`.
Downloaded data is git-ignored (licences differ per dataset; used for local evaluation only).

## Claims

### C1. It finds memory-poisoning text on note-style data
Leave-one-group-out stacker of three text scorers (`results_research/stacking.md`, `python -m scripts.run_stacking`):

| Test set | AUROC | Recall with 5 % false alarms | Generic published detector (ProtectAI DeBERTa) recall, same budget |
|---|---|---|---|
| Notes A | 0.959 | 0.83 | 0.57 |
| Notes B (unseen writer) | 0.992 | 0.97 | 0.70 |
| Notes C (unseen writer + style) | 1.000 | 1.00 | 0.77 |
| Notes D (hand-style, includes 18 benign "standing instructions") | 0.978 | 0.94 | 0.50 |
| InjecAgent vs persona/user text (shortcut risk) | 0.997 | 1.00 | 0.89 |

*Caveats.* Notes A-C were written by small local models; D by an AI assistant. The scorers saw `data/hand_train.tsv`
(written after seeing where the first model failed on D), so D is not fully independent for them. The two-source set
(InjecAgent vs persona facts) can be separated by topic alone.

### C2. It is competitive, not best, on generic prompt-injection datasets
Same stacker, held-out dataset, all other groups used for fitting (`results_research/multisource.md`, `stacking.md`):

| Dataset | Stacked AUROC | Recall at 5 % false alarms | ProtectAI DeBERTa alone |
|---|---|---|---|
| jackhhao jailbreak | 0.958 | 0.86 | 0.88 |
| SPML chatbot | 0.990 | 0.98 | 0.98 |
| geekyrakshit | 0.857 | 0.38 | 0.00 |
| deepset | 0.837 | 0.58 | 0.69 |
| safeguard | 0.992 | 0.98 | 0.96 |

On these our own detectors alone are clearly worse than the published detector (e.g. jackhhao 0.85 vs 0.97 AUROC,
`multisource.md`) and flag 55-95 % of ordinary chat prompts at the default threshold, because they were built for
memory notes, not chat. The stacker closes most of the ranking gap. The published detector may have trained on some
of these datasets (we cannot check), which favours it.

### C3. Generic prompt-injection detectors miss memory-write attacks
At its default 0.5 threshold ProtectAI DeBERTa recalls 15 % of InjecAgent attacker instructions and 35-63 % of
note-style attacks (`multisource.md`, "Recall at threshold 0.5"), while flagging almost no benign text. A guard built
for chat prompts is therefore not a substitute for a memory-write guard. This is the strongest argument for the product.

### C4. Disguised attacks (`results_research/perturbation*.md`, `python -m scripts.run_perturbation`)
Threshold fixed at 5 % false alarms on 50 unperturbed benign notes. Stacked scorer, recall on 50 perturbed attacks;
the pipeline now normalises text (confusable letters, letter spacing, leetspeak, invisible characters) and scores
sentence windows (max):

| Disguise | Before fixes | After fixes | False alarms on the same transformation, after |
|---|---|---|---|
| none | 0.88 | 0.88 | 0.06 |
| 3 benign sentences on each side | 0.48 | 0.88 | 0.12 |
| 6 benign sentences on each side | 0.22 | 0.88 | 0.08 |
| letter spacing | (1.00)* | 0.88 | 0.10 |
| leetspeak | (1.00)* | 0.96 | 0.16 |
| Cyrillic homoglyphs | (0.92)* | 0.86 | 0.06 |
| zero-width characters | (1.00)* | 0.88 | 0.06 |

\* Before the fixes the false-alarm rate on the same disguised *benign* text was 0.82-0.98, so those recalls were
meaningless (everything strange was flagged). Wrapping text in an authority claim, a polite request, a role-play quote
or a code fence still raises false alarms to 0.3-1.0 because the wrapper itself looks suspicious; that is unsolved.
Multi-sentence benign notes: 0.00 false alarms with windowing.

### C5. The service behaves under hostile input (`results_research/service_bench.md`, `python -m scripts.bench_service`)
In-process, real model, CPU only: 25 of 25 hostile-input cases (zero-width, bidi override, homoglyphs, null bytes,
emoji floods, 16 kB notes, malformed JSON, wrong types, missing/wrong keys, bad origin, path-traversal session id,
oversized body, disabled docs) returned a controlled 2xx/4xx and never a 5xx. Latency per write on a CPU laptop:
p50 about 0.4 s, p99 about 0.9 s; throughput 1-2.5 writes/s (the service serialises per tenant). That is adequate for a
memory-write guard, not for a high-volume gateway.

### C6. Provenance helps exactly as much as it can be trusted (paper section 5.4)
AUROC 0.49 / 0.80 / 0.95 when the origin label is right 50 / 80 / 95 % of the time. Provenance is a real signal only
if the agent runtime supplies reliable origins; the note-style sets above carry no provenance at all.

### C7. End to end with a real agent (paper section 5.9)
300 scenarios, 8B local model: attack success 0.96 undefended, 0.24 with a rule-based filter, 0.36 with the v2
detector, 0.50 with an LLM judge. **We do not beat the rule-based filter on that study.**

## What this does not show
- Performance against a human who has read this repository and adapts. The black-box study (paper section 5.8) gave an
  oracle-origin attacker 20-58 % success within 10 attempts.
- Real production traffic, other languages (only English tested), or long documents (notes are capped at 2000 chars).
- That the hand-style notes are representative of human writing: they were produced by an AI assistant.
- A head-to-head against A-MemGuard or other memory-specific defences (no code to run).
- Calibration at deployment: thresholds must be set per deployment from benign traffic.

## Reproduce
```bash
python -m scripts.download_external
python -m scripts.download_external2
python -m scripts.run_multisource      # multisource.{json,md}, multisource_scores.json
python -m scripts.run_stacking         # stacking.{json,md}
python -m scripts.run_perturbation --normalise --window --tag _norm_win
python -m scripts.bench_service        # service_bench.{json,md}
```
`results_research/manifest.json` records versions and hashes.
