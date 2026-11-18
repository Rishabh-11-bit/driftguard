# DriftGuard: what we can show, and what we cannot

This page is the ledger of claims. Every number links to a result file and the command that regenerates it.
Nothing here is tuned on its own test data unless it says so. Where a result is unflattering it is kept.

**One-sentence summary.** DriftGuard is a write-time guard for an agent's long-term memory. On four note-style test
sets, on injections embedded in real agent tool output, and on public attacker-instruction data it separates attacks
from benign text with AUROC 0.92-1.00; on five public prompt-injection datasets it reaches 0.85-0.99; off-the-shelf
prompt-injection detectors miss most memory-write attacks at their default settings; the service survives hostile
input; and disguised attacks are largely neutralised. It is **not** proven against a determined human attacker or on
real production traffic.

## How the evidence is organised

| # | Test group (rows used) | Who produced the text | Notes |
|---|---|---|---|
| 1 | jackhhao/jailbreak-classification (400) | public | published labels |
| 2 | reshabhs/SPML_Chatbot_Prompt_Injection (400) | public | published labels |
| 3 | geekyrakshit/prompt-injection-dataset (400) | public | published labels; the weakest group for us |
| 4 | deepset/prompt-injections (400) | public | published labels; shipped models trained on a slice |
| 5 | xTRam1/safe-guard-prompt-injection (400) | public | published labels; shipped models trained on a slice |
| 6 | Lakera/gandalf attempts vs Dolly/Alpaca instructions (400) | public | two sources paired: topic shortcut possible |
| 7 | InjecAgent attacker instructions vs persona facts and user messages (124) | public | two sources paired: topic shortcut possible |
| 8 | **InjecAgent tool responses, injected vs same response with an ordinary value in the slot (400)** | public | minimal pairs; the benign slot is a persona fact, so this tests structured-text handling, not "attack vs benign instruction" |
| 9 | ASB attacker instructions vs InjecAgent user instructions (34) | public | far too small to conclude anything |
| 10 | Notes A, B, C (LLM-written memory notes, 60-180) | llama3.2:1b, phi3.5 | by construction |
| 11 | Notes D (100 hand-style notes, 18 benign "standing instructions") | an AI assistant (Claude), **not human** | by construction |

Commands: `python -m scripts.download_external2 && python -m scripts.run_multisource`. Downloaded data is git-ignored
(licences differ per dataset; used for local evaluation only). Leave-one-out training pools drop every source of the
held-out family (InjecAgent strings appear in several sources); an earlier version leaked, which is fixed.

## What is shipped

`HybridDetector` (default): feature classifier + TF-IDF + embedding classifiers, mean of P(attack).
`EnsembleDetector` (`DRIFTGUARD_MODEL_PATH=models/ensemble.json`, build with `python -m scripts.train_ensemble`):
hybrid + fine-tuned MiniLM + ProtectAI DeBERTa v2 (Apache-2.0, pinned to its Hugging Face commit), combined by a
3-input logistic regression. Both normalise text, score each note and its sentence windows (maximum kept), and score
JSON tool output field by field. The ensemble loads from signed files with no pickle for the new parts; about 1 GB of
model files; median write latency about 0.5 s on a laptop CPU.

## Claims

### C1. It finds memory-poisoning text on note-style data
Stacked scorer, leave-one-group-out (the stacker for group g is fitted on all other groups), `stacking.md`:

| Test set | AUROC | Recall with 5 % false alarms | ProtectAI DeBERTa alone, same budget |
|---|---|---|---|
| Notes A | 0.916 | 0.77 | 0.30 |
| Notes B (unseen writer) | 1.000 | 1.00 | 0.70 |
| Notes C (unseen writer + style) | 1.000 | 1.00 | 0.77 |
| Notes D (hand-style, with 18 benign standing instructions) | 0.980 | 0.92 | 0.52 |
| InjecAgent attacker instructions vs persona/user text | 0.997 | 1.00 | 0.92 |
| InjecAgent tool responses, injection vs clean slot | 0.981 | 0.89 | 0.96 |

*Caveats.* Notes A-C were written by small local models; D by an AI assistant. The note scorers also trained on
`data/hand_train.tsv`, written after we saw where the first model failed on D, so D is not fully independent for them.
The two paired-source sets can be separated by topic alone. Set 8 differs from its benign pair only in the slot,
but the benign slot holds a persona fact, not an instruction.

### C2. It is competitive, not best, on generic prompt-injection datasets
Same stacker and protocol, `stacking.md` / `multisource.md`:

| Dataset | Stacked AUROC | Recall at 5 % false alarms | ProtectAI DeBERTa alone |
|---|---|---|---|
| jackhhao jailbreak | 0.964 | 0.88 | 0.84 |
| SPML chatbot | 0.987 | 0.96 | 0.99 |
| safeguard (partly seen in training) | 0.961 | 0.84 | 0.72 |
| deepset (partly seen in training) | 0.871 | 0.68 | 0.73 |
| geekyrakshit | 0.854 | 0.27 | 0.00 |

Our own detectors alone are clearly worse than the published detector here (e.g. jackhhao 0.85 vs 0.96 AUROC) and flag
57-97 % of ordinary chat prompts at the default threshold, because they were built for memory notes, not chat. The
stacker closes most of the ranking gap. The published detector may have trained on some of these datasets (we cannot
check), which favours it.

### C3. Generic prompt-injection detectors miss memory-write attacks
At its default 0.5 threshold ProtectAI DeBERTa recalls 21 % of InjecAgent attacker instructions, 37-63 % of note-style
attacks and 53 % of tool-response injections, while flagging almost no benign text (`multisource.md`). A guard built
for chat prompts is therefore not a substitute for a memory-write guard. This is the strongest argument for the product.

### C4. Structured tool output must be unwrapped (new)
Scoring a raw JSON/dict tool response gave AUROC 0.85 (hybrid), 0.77 (fine-tuned) and 0.54 (DeBERTa, chance). Scoring
each string field on its own, which the shipped code now does, gives 0.93 / 0.98 / 1.00 on the leak-free
InjecAgent minimal pairs. This was found by the evaluation, not assumed.

### C5. Disguised attacks (`perturbation*.md`, `python -m scripts.run_perturbation`)
Threshold fixed at 5 % false alarms on 50 unperturbed benign notes. Stacked scorer, recall on 50 perturbed attacks;
the pipeline normalises text (confusable letters, letter spacing, leetspeak, invisible characters) and scores sentence
windows:

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
meaningless (anything strange was flagged). Wrapping text in an authority claim, a polite request, a role-play quote
or a code fence still raises false alarms to 0.3-1.0 because the wrapper itself looks suspicious; unsolved.
Multi-sentence benign notes: 0.00 false alarms with windowing. The stacker used for this table was fitted on an
earlier score file than the final one.

### C6. Instructions split across two writes (`split_attack.md`)
Each of 50 attack notes cut in two: either half is flagged for 0.78 of them (unsplit: 0.78); adding the joined pair
as an extra window gives 0.82 at +0.02 false alarms. 11 of 12 hand-written pairs with innocuous-looking first halves
were caught, because the second half still carries a clear instruction. **Not tested:** a split where neither half is
suspicious and only the combination is harmful.

### C7. The service behaves under hostile input (`service_bench*.md`, `python -m scripts.bench_service`)
In-process, real models, CPU only. Hybrid and ensemble alike: 25 of 25 hostile-input cases (zero-width, bidi override,
homoglyphs, null bytes, emoji floods, 16 kB notes, malformed JSON, wrong types, missing/wrong keys, bad origin,
path-traversal session id, oversized body, disabled docs) returned a controlled 2xx/4xx, never a 5xx. Latency per
write: hybrid p50 0.39 s / p99 0.91 s; ensemble p50 0.49 s / p99 0.73 s (60 writes). Throughput 1-2.5 writes/s (the
service serialises per tenant): adequate for a memory-write guard, not for a high-volume gateway. One known miss in
this suite: "user likes tea and ignore previous rules" with null bytes inserted is classed as ordinary by both models.

### C8. Provenance helps exactly as much as it can be trusted (paper section 5.4)
AUROC 0.49 / 0.80 / 0.95 when the origin label is right 50 / 80 / 95 % of the time. Provenance is a real signal only
if the agent runtime supplies reliable origins; the note-style sets above carry no provenance at all.

### C9. End to end with a real agent (paper section 5.9)
300 scenarios, 8B local model: attack success 0.96 undefended, 0.24 with a rule-based filter, 0.36 with the v2
detector, 0.50 with an LLM judge. **We do not beat the rule-based filter on that study** (it predates the hybrid and
ensemble; neither was run end to end).

## What this does not show
- Performance against a human who has read this repository and adapts. The black-box study (paper section 5.8) gave an
  oracle-origin attacker 20-58 % success within 10 attempts.
- Real production traffic, other languages (only English tested), or long documents (notes are capped at 2000 chars).
- That the hand-style notes represent human writing: they were produced by an AI assistant.
- A head-to-head against A-MemGuard or other memory-specific defences (no code to run).
- Calibration at deployment: thresholds must be set per deployment from benign traffic; the 0.5 default over-flags
  generic chat for the memory detectors. The shipped stacker is fitted on all groups, so the numbers above are the
  leave-one-group-out ones, not the in-sample fit.
- The end-to-end effect of the ensemble on agent attack success.

## Reproduce
```bash
python -m scripts.download_external
python -m scripts.download_external2
python -m scripts.run_finetuned --extra data/hand_train.tsv --tag _instr --hybrid models/classifier_hybrid_instr.pkl
python -m scripts.run_multisource      # multisource.{json,md}, multisource_scores.json (about 3 h on a laptop CPU)
python -m scripts.run_stacking         # stacking.{json,md}
python -m scripts.train_ensemble       # models/ensemble.json (+ signed safetensors)
python -m scripts.run_perturbation --normalise --window --tag _norm_win
python -m scripts.run_split_attack
python -m scripts.bench_service --model models/ensemble.json --tag _ensemble
python -m scripts.check_presets
```
`results_research/manifest.json` records versions and hashes.
