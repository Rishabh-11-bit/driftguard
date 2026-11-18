| Test group | Stacked AUROC | Recall @0.5 | False alarm @0.5 | Best single (AUROC) | Mean fusion AUROC |
|---|---|---|---|---|---|
| jackhhao_jailbreak | 0.964 | 0.97 | 0.41 | protectai_deberta (0.957) | 0.950 |
| spml_chatbot | 0.987 | 0.97 | 0.06 | protectai_deberta (0.996) | 0.982 |
| geekyrakshit_injection | 0.854 | 0.99 | 0.94 | hybrid_instr (0.812) | 0.820 |
| deepset | 0.871 | 0.85 | 0.40 | protectai_deberta (0.902) | 0.875 |
| safeguard | 0.961 | 0.99 | 0.31 | protectai_deberta (0.940) | 0.952 |
| gandalf_vs_instructions | 1.000 | 0.99 | 0.04 | protectai_deberta (1.000) | 0.999 |
| injecagent_vs_notes | 0.997 | 0.65 | 0.00 | hybrid_instr (0.998) | 0.998 |
| injecagent_toolresp | 0.981 | 0.83 | 0.01 | protectai_deberta (0.980) | 0.969 |
| asb_vs_user_instructions | 0.820 | 0.71 | 0.18 | finetuned_instr (0.848) | 0.820 |
| notes_D | 0.980 | 0.78 | 0.02 | finetuned_instr (0.984) | 0.970 |
| notes_A | 0.916 | 0.55 | 0.00 | finetuned_instr (0.912) | 0.920 |
| notes_B | 1.000 | 0.73 | 0.00 | finetuned_instr (1.000) | 0.998 |
| notes_C | 1.000 | 0.73 | 0.00 | hybrid_instr (1.000) | 1.000 |

Recall when at most 5 % of benign items are flagged (threshold set per group on its own benign items, so this compares ranking quality at a fixed false-alarm budget, not a deployable threshold):

| Test group | hybrid_instr | finetuned_instr | protectai_deberta | stack |
|---|---|---|---|---|
| jackhhao_jailbreak | 0.43 | 0.52 | 0.84 | 0.88 |
| spml_chatbot | 0.61 | 0.70 | 0.99 | 0.96 |
| geekyrakshit_injection | 0.15 | 0.18 | 0.00 | 0.27 |
| deepset | 0.30 | 0.18 | 0.73 | 0.68 |
| safeguard | 0.62 | 0.73 | 0.72 | 0.84 |
| gandalf_vs_instructions | 0.84 | 0.69 | 1.00 | 1.00 |
| injecagent_vs_notes | 0.98 | 1.00 | 0.92 | 1.00 |
| injecagent_toolresp | 0.64 | 0.84 | 0.96 | 0.89 |
| asb_vs_user_instructions | 0.65 | 0.65 | 0.24 | 0.65 |
| notes_D | 0.80 | 0.90 | 0.52 | 0.92 |
| notes_A | 0.68 | 0.82 | 0.30 | 0.77 |
| notes_B | 0.90 | 1.00 | 0.70 | 1.00 |
| notes_C | 1.00 | 1.00 | 0.77 | 1.00 |
