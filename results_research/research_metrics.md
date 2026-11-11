
### raw benchmark (n=600, classes=[200, 200, 200])

| Model | Accuracy | Macro-F1 | MCC | Attack TPR | FAR | MAR |
|---|---|---|---|---|---|---|
| DriftGuard (RF) | 0.997 ± 0.003 | 0.997 ± 0.003 | 0.996 ± 0.004 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| DriftGuard (GBM) | 0.998 ± 0.003 | 0.998 ± 0.003 | 0.997 ± 0.004 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Logistic regression | 0.998 ± 0.003 | 0.998 ± 0.003 | 0.997 ± 0.004 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Rule-based baseline | 0.642 ± 0.007 | 0.532 ± 0.006 | 0.557 ± 0.015 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Majority class | 0.333 ± 0.000 | 0.167 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |

| Ablation (RF) | Accuracy | Attack TPR | FAR |
|---|---|---|---|
| all features | 0.997 | 1.000 | 0.000 |
| without provenance | 0.992 | 1.000 | 0.000 |
| only provenance | 0.995 | 1.000 | 0.005 |
| without temporal | 1.000 | 1.000 | 0.000 |
| only temporal | 0.950 | 0.955 | 0.045 |
| without consistency | 0.998 | 1.000 | 0.000 |
| only consistency | 0.747 | 1.000 | 0.000 |
| without causal/content | 0.998 | 1.000 | 0.000 |
| only causal/content | 1.000 | 1.000 | 0.000 |

McNemar RF vs rule-based: RF-only-correct=214, rule-only-correct=1, p=8.20e-63

### hardened benchmark (n=600, classes=[200, 200, 200])

| Model | Accuracy | Macro-F1 | MCC | Attack TPR | FAR | MAR |
|---|---|---|---|---|---|---|
| DriftGuard (RF) | 0.918 ± 0.018 | 0.918 ± 0.018 | 0.879 ± 0.027 | 0.987 ± 0.015 | 0.028 ± 0.014 | 0.013 ± 0.015 |
| DriftGuard (GBM) | 0.923 ± 0.015 | 0.923 ± 0.015 | 0.885 ± 0.022 | 0.983 ± 0.014 | 0.020 ± 0.009 | 0.017 ± 0.014 |
| Logistic regression | 0.909 ± 0.013 | 0.910 ± 0.013 | 0.866 ± 0.020 | 0.985 ± 0.009 | 0.003 ± 0.005 | 0.015 ± 0.009 |
| Rule-based baseline | 0.557 ± 0.011 | 0.442 ± 0.009 | 0.389 ± 0.018 | 0.995 ± 0.006 | 0.310 ± 0.026 | 0.005 ± 0.006 |
| Majority class | 0.333 ± 0.000 | 0.167 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |

| Ablation (RF) | Accuracy | Attack TPR | FAR |
|---|---|---|---|
| all features | 0.918 | 0.987 | 0.028 |
| without provenance | 0.908 | 0.995 | 0.030 |
| only provenance | 0.642 | 0.660 | 0.020 |
| without temporal | 0.927 | 0.995 | 0.030 |
| only temporal | 0.413 | 0.480 | 0.360 |
| without consistency | 0.920 | 0.980 | 0.010 |
| only consistency | 0.630 | 0.945 | 0.045 |
| without causal/content | 0.703 | 0.945 | 0.050 |
| only causal/content | 0.910 | 0.985 | 0.020 |

McNemar RF vs rule-based: RF-only-correct=240, rule-only-correct=21, p=2.84e-48
