| Test set | Scorer | AUROC | Recall @0.5 | False alarm @0.5 | False alarm at 90 % recall |
|---|---|---|---|---|---|
| benchmark test (n=40+80) | hybrid | 1.000 | 1.00 | 0.00 | 0.00 |
| benchmark test (n=40+80) | finetuned | 1.000 | 1.00 | 0.00 | 0.00 |
| benchmark test (n=40+80) | hybrid+finetuned | 1.000 | 1.00 | 0.00 | 0.00 |
| notes A (seen writer, new goals) (n=60+120) | hybrid | 0.948 | 0.57 | 0.02 | 0.12 |
| notes A (seen writer, new goals) (n=60+120) | finetuned | 0.941 | 0.80 | 0.05 | 0.23 |
| notes A (seen writer, new goals) (n=60+120) | hybrid+finetuned | 0.960 | 0.80 | 0.03 | 0.11 |
| notes B (unseen writer) (n=30+60) | hybrid | 0.953 | 0.80 | 0.05 | 0.10 |
| notes B (unseen writer) (n=30+60) | finetuned | 0.990 | 0.90 | 0.02 | 0.02 |
| notes B (unseen writer) (n=30+60) | hybrid+finetuned | 0.979 | 0.90 | 0.02 | 0.00 |
| notes C (unseen writer + style) (n=30+60) | hybrid | 0.997 | 1.00 | 0.05 | 0.00 |
| notes C (unseen writer + style) (n=30+60) | finetuned | 0.997 | 1.00 | 0.03 | 0.02 |
| notes C (unseen writer + style) (n=30+60) | hybrid+finetuned | 0.999 | 1.00 | 0.02 | 0.00 |
| notes D (hand-style, 100 notes, 18 benign instruction-like) (n=50+50) | hybrid | 0.952 | 0.96 | 0.30 | 0.14 |
| notes D (hand-style, 100 notes, 18 benign instruction-like) (n=50+50) | finetuned | 0.984 | 0.90 | 0.04 | 0.02 |
| notes D (hand-style, 100 notes, 18 benign instruction-like) (n=50+50) | hybrid+finetuned | 0.978 | 0.90 | 0.08 | 0.02 |
| real text, length-matched, unseen (n=126+227) | hybrid | 0.985 | 0.83 | 0.00 | 0.01 |
| real text, length-matched, unseen (n=126+227) | finetuned | 0.998 | 0.94 | 0.00 | 0.00 |
| real text, length-matched, unseen (n=126+227) | hybrid+finetuned | 0.996 | 0.92 | 0.00 | 0.00 |
