Configuration: normalise=True, window-max=True.

Recall on 50 perturbed attacks (threshold fixed at 5 % false alarms on the original benign notes) / false-alarm rate on the 50 benign notes after the same transformation:

| Transformation | hybrid_instr | finetuned_instr | protectai_deberta | stack |
|---|---|---|---|---|
| original | 0.78 / 0.06 | 0.90 / 0.06 | 0.52 / 0.06 | 0.88 / 0.06 |
| dilute_1_before | 0.72 / 0.02 | 0.90 / 0.06 | 0.68 / 0.08 | 0.90 / 0.10 |
| dilute_3_both | 0.58 / 0.00 | 0.94 / 0.04 | 0.74 / 0.10 | 0.88 / 0.12 |
| dilute_6_both | 0.54 / 0.02 | 0.86 / 0.04 | 0.58 / 0.06 | 0.88 / 0.08 |
| lowercase | 0.80 / 0.06 | 0.90 / 0.06 | 0.60 / 0.06 | 0.90 / 0.04 |
| UPPERCASE | 0.80 / 0.06 | 0.90 / 0.06 | 0.86 / 0.14 | 0.94 / 0.16 |
| letter_spacing | 0.76 / 0.06 | 0.98 / 0.34 | 0.62 / 0.14 | 0.88 / 0.10 |
| leetspeak | 0.62 / 0.04 | 1.00 / 0.62 | 0.80 / 0.06 | 0.96 / 0.16 |
| cyrillic_homoglyphs | 0.76 / 0.06 | 0.90 / 0.06 | 0.52 / 0.06 | 0.86 / 0.06 |
| zero_width_chars | 0.80 / 0.06 | 0.90 / 0.06 | 0.52 / 0.06 | 0.88 / 0.06 |
| authority_prefix | 0.98 / 0.64 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| polite_wrapper | 0.88 / 0.20 | 0.98 / 0.20 | 1.00 / 1.00 | 1.00 / 1.00 |
| roleplay_quote | 0.96 / 0.38 | 1.00 / 1.00 | 0.74 / 0.30 | 1.00 / 0.64 |
| code_fence | 0.84 / 0.18 | 1.00 / 1.00 | 0.64 / 0.12 | 0.96 / 0.44 |
| benign_filler_suffix | 0.50 / 0.00 | 0.88 / 0.06 | 0.58 / 0.12 | 0.86 / 0.14 |
| _benign_multisentence_false_alarm | nan / 0.00 | nan / 0.00 | nan / 0.00 | nan / 0.00 |
