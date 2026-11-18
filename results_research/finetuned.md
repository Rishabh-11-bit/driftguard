| Test set | Scorer | AUROC | Recall @0.5 | False alarm @0.5 | False alarm at 90 % recall |
|---|---|---|---|---|---|
| benchmark test (n=40+80) | hybrid | 1.000 | 1.00 | 0.00 | 0.00 |
| benchmark test (n=40+80) | finetuned | 1.000 | 1.00 | 0.00 | 0.00 |
| benchmark test (n=40+80) | hybrid+finetuned | 1.000 | 1.00 | 0.00 | 0.00 |
| notes A (seen writer, new goals) (n=60+120) | hybrid | 0.946 | 0.62 | 0.03 | 0.13 |
| notes A (seen writer, new goals) (n=60+120) | finetuned | 0.943 | 0.78 | 0.01 | 0.27 |
| notes A (seen writer, new goals) (n=60+120) | hybrid+finetuned | 0.964 | 0.78 | 0.01 | 0.17 |
| notes B (unseen writer) (n=30+60) | hybrid | 0.951 | 0.77 | 0.03 | 0.12 |
| notes B (unseen writer) (n=30+60) | finetuned | 0.991 | 0.70 | 0.00 | 0.00 |
| notes B (unseen writer) (n=30+60) | hybrid+finetuned | 0.970 | 0.70 | 0.00 | 0.00 |
| notes C (unseen writer + style) (n=30+60) | hybrid | 0.994 | 1.00 | 0.08 | 0.02 |
| notes C (unseen writer + style) (n=30+60) | finetuned | 0.995 | 0.97 | 0.03 | 0.02 |
| notes C (unseen writer + style) (n=30+60) | hybrid+finetuned | 0.996 | 0.97 | 0.03 | 0.02 |
| notes D (hand-style, 100 notes, 18 benign instruction-like) (n=50+50) | hybrid | 0.892 | 0.96 | 0.38 | 0.30 |
| notes D (hand-style, 100 notes, 18 benign instruction-like) (n=50+50) | finetuned | 0.895 | 0.98 | 0.42 | 0.28 |
| notes D (hand-style, 100 notes, 18 benign instruction-like) (n=50+50) | hybrid+finetuned | 0.895 | 0.98 | 0.42 | 0.30 |
| real text, length-matched, unseen (n=126+227) | hybrid | 0.988 | 0.83 | 0.00 | 0.00 |
| real text, length-matched, unseen (n=126+227) | finetuned | 1.000 | 0.90 | 0.00 | 0.00 |
| real text, length-matched, unseen (n=126+227) | hybrid+finetuned | 0.998 | 0.89 | 0.00 | 0.00 |
