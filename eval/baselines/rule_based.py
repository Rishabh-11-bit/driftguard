"""
Rule-based binary detector baseline.

Flags writes as malicious (0) if any of three heuristic signals fire:
  1. Low trust level (< 0.5)
  2. High novelty score (cosine distance to memory centroid > threshold)
  3. Contains instruction keywords (override, ignore, forget, etc.)

Classifies everything else as ordinary_learning (2).
Note: This baseline cannot distinguish benign_drift from ordinary_learning,
so it generates false alarms on drift and maps everything to a binary outcome.

Author: Tanisha Talathi (1032231612) — Objective 5
"""

from __future__ import annotations

import numpy as np

# Feature indices (must match driftguard/detector/features.py FEATURE_DIM layout)
IDX_TRUST         = 1
IDX_NOVELTY       = 8
IDX_INSTRUCTION   = 14
IDX_BURST_SCORE   = 6
IDX_WRITES_LAST10 = 3


class RuleBasedDetector:
    """
    Simple heuristic binary detector.

    Decision rule (OR of conditions → malicious):
      - trust_level < trust_threshold
      - novelty_score > novelty_threshold
      - contains_instruction == 1
      - burst_score > burst_threshold AND writes_last_10 > burst_min_writes
    """

    def __init__(
        self,
        trust_threshold:   float = 0.50,
        novelty_threshold: float = 0.75,
        burst_threshold:   float = 0.70,
        burst_min_writes:  int   = 3,
    ) -> None:
        self.trust_threshold   = trust_threshold
        self.novelty_threshold = novelty_threshold
        self.burst_threshold   = burst_threshold
        self.burst_min_writes  = burst_min_writes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns integer predictions: 0 (malicious) or 2 (ordinary).
        Note: never predicts 1 (benign_drift) — a key limitation.
        """
        preds = np.full(len(X), 2, dtype=int)

        low_trust    = X[:, IDX_TRUST]       < self.trust_threshold
        high_novelty = X[:, IDX_NOVELTY]     > self.novelty_threshold
        instruction  = X[:, IDX_INSTRUCTION] == 1.0
        burst        = (X[:, IDX_BURST_SCORE]   > self.burst_threshold) & \
                       (X[:, IDX_WRITES_LAST10] > self.burst_min_writes)

        flagged = low_trust | high_novelty | instruction | burst
        preds[flagged] = 0
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return dummy probabilities — no calibration for rule-based."""
        preds = self.predict(X)
        proba = np.zeros((len(X), 3), dtype=float)
        for i, p in enumerate(preds):
            if p == 0:
                proba[i] = [0.80, 0.10, 0.10]
            else:
                proba[i] = [0.10, 0.10, 0.80]
        return proba
