Configuration: normalise=True, window-max=False.

Recall on 50 perturbed attacks (threshold fixed at 5 % false alarms on the original benign notes) / false-alarm rate on the 50 benign notes after the same transformation:

| Transformation | hybrid_instr | finetuned_instr | protectai_deberta | stack |
|---|---|---|---|---|
| original | 0.80 / 0.06 | 0.90 / 0.06 | 0.50 / 0.06 | 0.88 / 0.06 |
| dilute_1_before | 0.52 / 0.00 | 0.78 / 0.00 | 0.62 / 0.04 | 0.82 / 0.06 |
| dilute_3_both | 0.02 / 0.00 | 0.32 / 0.00 | 0.66 / 0.02 | 0.48 / 0.00 |
| dilute_6_both | 0.00 / 0.00 | 0.14 / 0.00 | 0.48 / 0.04 | 0.22 / 0.02 |
| lowercase | 0.80 / 0.04 | 0.90 / 0.06 | 0.58 / 0.06 | 0.92 / 0.08 |
| UPPERCASE | 0.80 / 0.06 | 0.90 / 0.06 | 0.86 / 0.14 | 0.96 / 0.14 |
| letter_spacing | 0.74 / 0.06 | 0.98 / 0.34 | 0.56 / 0.12 | 0.88 / 0.14 |
| leetspeak | 0.64 / 0.12 | 1.00 / 0.62 | 0.80 / 0.06 | 0.94 / 0.16 |
| cyrillic_homoglyphs | 0.82 / 0.04 | 0.90 / 0.06 | 0.50 / 0.06 | 0.90 / 0.06 |
| zero_width_chars | 0.80 / 0.04 | 0.90 / 0.06 | 0.50 / 0.06 | 0.88 / 0.06 |
| authority_prefix | 0.96 / 0.34 | 1.00 / 0.76 | 0.76 / 0.26 | 1.00 / 0.52 |
| polite_wrapper | 0.88 / 0.16 | 0.98 / 0.20 | 1.00 / 0.92 | 1.00 / 0.62 |
| roleplay_quote | 0.94 / 0.20 | 1.00 / 0.50 | 0.74 / 0.30 | 0.98 / 0.44 |
| code_fence | 0.86 / 0.16 | 1.00 / 0.66 | 0.62 / 0.12 | 0.94 / 0.26 |
| benign_filler_suffix | 0.16 / 0.00 | 0.78 / 0.02 | 0.50 / 0.10 | 0.66 / 0.04 |
| _benign_multisentence_false_alarm | nan / 0.00 | nan / 0.00 | nan / 0.00 | nan / 0.00 |
