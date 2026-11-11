| Defence | block_attack | exposure | asr | drift_admit | drift_util_use | drift_util_withhold | ordinary_admit | ordinary_util |
|---|---|---|---|---|---|---|---|---|
| No defence | 0.00 | 1.00 | 0.47 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Rule-based | 0.80 | 0.20 | 0.20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LLM judge (local) | 0.47 | 0.53 | 0.27 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| DriftGuard-hard, legacy origin | 0.57 | 0.43 | 0.37 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| DriftGuard-hard, runtime attribution | 0.57 | 0.43 | 0.37 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 |
| DriftGuard-robust, runtime attribution | 0.70 | 0.30 | 0.30 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 |
| DriftGuard-robust, oracle origin | 0.60 | 0.40 | 0.37 | 1.00 | 1.00 | 0.87 | 1.00 | 1.00 |
| DriftGuard-v2, runtime attribution | 0.70 | 0.30 | 0.27 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| DriftGuard-v2, oracle origin | 0.70 | 0.30 | 0.27 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
