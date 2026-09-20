"""
Feature extraction for the DriftGuard three-class detector.

Extracts four families of signals from a candidate memory write:

  1. Provenance features   — origin encoding, trust level, session age
  2. Temporal features     — write frequency, burst score, inter-write intervals
  3. Profile consistency   — cosine similarity to user memory centroid, novelty
  4. Causal attribution    — overlap with retrieved docs, query similarity

The combined feature vector has a fixed dimension (see FEATURE_DIM) and
is ready to feed directly into the DriftGuardClassifier.

Author: Rishabh Gaikwad (1032231674) — Objective 3
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from driftguard.provenance.schema import ProvenanceRecord, ORIGIN_TRUST_DEFAULTS

logger = logging.getLogger(__name__)

# Total number of features per candidate write
FEATURE_DIM = 18


# ─────────────────────────────────────────────────────────────
# FeatureExtractor
# ─────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts an 18-dimensional feature vector for a candidate memory write.

    Feature layout:
      [0]    origin_index            (int, 0–4)
      [1]    trust_level             (float, 0–1)
      [2]    session_age_hours       (float)
      ──── temporal (4 features) ────────────────────────
      [3]    writes_last_10          (int)
      [4]    writes_last_30          (int)
      [5]    inter_write_interval_mean_min (float, minutes; 0 if no history)
      [6]    burst_score             (float; ratio last-10 / last-30, ∈[0,1])
      ──── consistency (5 features) ─────────────────────
      [7]    cosine_sim_to_centroid  (float, –1 to 1)
      [8]    novelty_score           (float, 0 to 1; 1 – cosine_sim)
      [9]    same_session_writes     (int)
      [10]   unique_origins_count    (int)
      [11]   repeated_hash_flag      (int; 1 if same interaction_hash seen before)
      ──── causal attribution (7 features) ──────────────
      [12]   query_content_overlap   (float; cosine sim between write and last query)
      [13]   write_length_chars      (int)
      [14]   contains_instruction    (int; 1 if imperative-like patterns present)
      [15]   contains_url            (int)
      [16]   avg_word_length         (float)
      [17]   exclamation_ratio       (float; !/total chars)

    Parameters
    ----------
    embedding_model : str
        SentenceTransformer model name for embedding.
    """

    def __init__(self, embedding_model: str | None = None) -> None:
        self._model_name = (
            embedding_model
            or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self._encoder: SentenceTransformer | None = None  # lazy load

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            logger.info(f"[FeatureExtractor] Loading encoder '{self._model_name}'…")
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def extract(
        self,
        content: str,
        provenance: ProvenanceRecord,
        write_history: list[dict[str, Any]],
        session_id: str,
        last_query: str = "",
        user_memory_entries: list[str] | None = None,
    ) -> np.ndarray:
        """
        Compute and return the 18-dimensional feature vector.

        Parameters
        ----------
        content : str
            The candidate write text.
        provenance : ProvenanceRecord
            Provenance record for this write.
        write_history : list[dict]
            Recent write history from WritePipeline (last N writes).
        session_id : str
            Current session ID.
        last_query : str
            The most recent user query (for causal attribution).
        user_memory_entries : list[str] | None
            Text of all current memory entries (for centroid computation).
        """
        features = np.zeros(FEATURE_DIM, dtype=np.float32)

        # ── 1. Provenance features ───────────────────────────
        features[0] = float(provenance.origin_index)
        features[1] = float(provenance.trust_level)
        features[2] = float(provenance.session_age_seconds) / 3600.0  # hours

        # ── 2. Temporal features ─────────────────────────────
        now = datetime.now(timezone.utc)
        timestamps = []
        for w in write_history:
            try:
                ts = datetime.fromisoformat(w["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                timestamps.append(ts)
            except Exception:
                pass

        last_10 = [t for t in timestamps if (now - t).total_seconds() <= 600]   # 10 min
        last_30 = [t for t in timestamps if (now - t).total_seconds() <= 1800]  # 30 min

        features[3] = float(len(last_10))
        features[4] = float(len(last_30))

        if len(timestamps) >= 2:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60.0
                         for i in range(len(timestamps) - 1)]
            features[5] = float(np.mean(intervals))
        else:
            features[5] = 0.0

        features[6] = (
            float(len(last_10)) / max(float(len(last_30)), 1.0)
        )

        # ── 3. Profile consistency ───────────────────────────
        entries = user_memory_entries or []
        if entries:
            content_emb = self.encoder.encode([content], normalize_embeddings=True)[0]
            entry_embs  = self.encoder.encode(entries,  normalize_embeddings=True)
            centroid    = entry_embs.mean(axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
            sim = float(np.dot(content_emb, centroid_norm))
        else:
            sim = 0.0

        features[7] = sim
        features[8] = 1.0 - max(sim, 0.0)

        # same-session write count
        same_session = sum(1 for w in write_history if w.get("session_id") == session_id)
        features[9] = float(same_session)

        # unique origins in history
        unique_origins = len(set(w.get("origin", "unknown") for w in write_history))
        features[10] = float(unique_origins)

        # repeated interaction hash — has this exact triggering interaction been seen before?
        my_hash = provenance.interaction_hash
        hash_count = sum(1 for w in write_history if w.get("interaction_hash") == my_hash)
        features[11] = float(int(hash_count > 0))

        # ── 4. Causal attribution ────────────────────────────
        if last_query:
            q_emb = self.encoder.encode([last_query], normalize_embeddings=True)[0]
            c_emb = self.encoder.encode([content],    normalize_embeddings=True)[0]
            features[12] = float(np.dot(q_emb, c_emb))
        else:
            features[12] = 0.0

        features[13] = float(len(content))

        # Instruction-like heuristic: starts with verb or contains "you should / do X"
        lower = content.lower()
        instruction_keywords = [
            "ignore", "forget", "pretend", "always", "never", "do not",
            "do not", "you must", "you should", "from now on", "override"
        ]
        features[14] = float(any(kw in lower for kw in instruction_keywords))

        import re
        features[15] = float(bool(re.search(r"https?://\S+", content)))

        words = content.split()
        features[16] = float(np.mean([len(w) for w in words])) if words else 0.0
        features[17] = float(content.count("!")) / max(len(content), 1)

        return features

    def extract_batch(
        self,
        records: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Batch feature extraction for the evaluation harness and training.

        Parameters
        ----------
        records : list[dict]
            Each dict should have keys:
              - "content"      : str
              - "provenance"   : dict  (from ProvenanceRecord.to_dict())
              - "write_history": list  (default [])
              - "session_id"   : str   (default "bench")
              - "last_query"   : str   (default "")
              - "memory_entries": list[str] (default [])

        Returns
        -------
        np.ndarray of shape (n, FEATURE_DIM)
        """
        out = []
        for rec in records:
            prov = ProvenanceRecord.from_dict(rec["provenance"])
            vec = self.extract(
                content=rec["content"],
                provenance=prov,
                write_history=rec.get("write_history", []),
                session_id=rec.get("session_id", "bench"),
                last_query=rec.get("last_query", ""),
                user_memory_entries=rec.get("memory_entries", []),
            )
            out.append(vec)
        return np.array(out, dtype=np.float32)
