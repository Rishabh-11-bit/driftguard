| Test group | Stacked AUROC | Recall @0.5 | False alarm @0.5 | Best single (AUROC) | Mean fusion AUROC |
|---|---|---|---|---|---|
| jackhhao_jailbreak | 0.958 | 0.95 | 0.32 | protectai_deberta (0.961) | 0.949 |
| spml_chatbot | 0.990 | 0.98 | 0.09 | protectai_deberta (0.993) | 0.978 |
| geekyrakshit_injection | 0.857 | 0.99 | 0.94 | protectai_deberta (0.808) | 0.826 |
| deepset | 0.837 | 0.82 | 0.46 | protectai_deberta (0.852) | 0.828 |
| safeguard | 0.992 | 0.99 | 0.16 | protectai_deberta (0.985) | 0.987 |
| gandalf_vs_instructions | 1.000 | 1.00 | 0.04 | protectai_deberta (1.000) | 1.000 |
| injecagent_vs_notes | 0.997 | 0.61 | 0.00 | hybrid_instr (0.998) | 0.999 |
| notes_D | 0.978 | 0.78 | 0.04 | finetuned_instr (0.984) | 0.970 |
| notes_A | 0.959 | 0.48 | 0.00 | hybrid_instr (0.950) | 0.957 |
| notes_B | 0.992 | 0.57 | 0.00 | finetuned_instr (0.991) | 0.971 |
| notes_C | 1.000 | 0.77 | 0.00 | finetuned_instr (1.000) | 1.000 |

Recall when at most 5 % of benign items are flagged (threshold set per group on its own benign items, so this compares ranking quality at a fixed false-alarm budget, not a deployable threshold):

| Test group | hybrid_instr | finetuned_instr | protectai_deberta | stack |
|---|---|---|---|---|
| jackhhao_jailbreak | 0.48 | 0.44 | 0.88 | 0.86 |
| spml_chatbot | 0.55 | 0.72 | 0.98 | 0.98 |
| geekyrakshit_injection | 0.26 | 0.14 | 0.00 | 0.38 |
| deepset | 0.25 | 0.14 | 0.69 | 0.58 |
| safeguard | 0.71 | 0.81 | 0.96 | 0.98 |
| gandalf_vs_instructions | 0.85 | 0.68 | 1.00 | 1.00 |
| injecagent_vs_notes | 0.98 | 1.00 | 0.89 | 1.00 |
| notes_D | 0.82 | 0.90 | 0.50 | 0.94 |
| notes_A | 0.80 | 0.78 | 0.57 | 0.83 |
| notes_B | 0.93 | 0.97 | 0.70 | 0.97 |
| notes_C | 1.00 | 1.00 | 0.77 | 1.00 |
