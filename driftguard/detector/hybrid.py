"""Hybrid detector: feature classifier + plain text classifiers, fused by averaging P(malicious).

Why: on held-out LLM-written memory notes and length-matched real text, TF-IDF and sentence-embedding logistic
regressions were as good as or better than the 25-feature detector, and their unweighted mean with it was the best
or tied-best scorer on every text set (paper section 5.7c). The feature detector still supplies what text cannot:
provenance, timing and consistency with existing memories, and it supplies the drift-vs-ordinary split.

    P(malicious) = mean(feature detector, TF-IDF LR, embedding LR)
    P(drift), P(ordinary) = the remaining mass split in the feature detector's drift:ordinary ratio

The object keeps the DriftGuardClassifier interface the pipeline uses (predict_one, _feature_names) and adds
`needs_text = True`, so WritePipeline passes it the sanitised write content. It is saved as a signed pickle,
like every other model, and loaded through the same signature check.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

_ENCODERS: dict[str, object] = {}


def _encoder(name: str):
    if name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer
        _ENCODERS[name] = SentenceTransformer(name)
    return _ENCODERS[name]


class HybridDetector:
    needs_text = True

    def __init__(self, base, tfidf, lr_tfidf, lr_embed, encoder_name: str = "all-MiniLM-L6-v2") -> None:
        self.base, self.tfidf, self.lr_tfidf, self.lr_embed = base, tfidf, lr_tfidf, lr_embed
        self.encoder_name = encoder_name

    # -- DriftGuardClassifier-compatible surface ----------------------------------------------
    @property
    def _feature_names(self):
        return self.base._feature_names

    @property
    def _is_fitted(self) -> bool:
        return self.base._is_fitted

    def _text_scores(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        p_tfidf = self.lr_tfidf.predict_proba(self.tfidf.transform(texts))[:, 1]
        emb = _encoder(self.encoder_name).encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return p_tfidf, self.lr_embed.predict_proba(emb)[:, 1]

    def predict_proba_text(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        """Fused 3-class probabilities for a batch: feature rows X and the matching write texts."""
        base = self.base.predict_proba(X)
        p_tfidf, p_embed = self._text_scores(list(texts))
        p_mal = (base[:, 0] + p_tfidf + p_embed) / 3.0
        rest = base[:, 1] + base[:, 2]
        drift_share = np.where(rest > 1e-6, base[:, 1] / np.clip(rest, 1e-6, None), 0.5)   # undefined -> 50/50
        return np.stack([p_mal, (1 - p_mal) * drift_share, (1 - p_mal) * (1 - drift_share)], axis=1)

    def predict_one(self, features: np.ndarray, content: str | None = None):
        if content is None:
            raise ValueError("HybridDetector needs the write content (needs_text=True)")
        proba = self.predict_proba_text(np.asarray(features, dtype=np.float32).reshape(1, -1), [content])[0]
        return int(np.argmax(proba)), proba

    def predict_proba(self, X, texts=None):
        if texts is None:
            raise ValueError("HybridDetector.predict_proba needs texts")
        return self.predict_proba_text(X, texts)

    def predict(self, X, texts=None):
        return np.argmax(self.predict_proba(X, texts), axis=1)

    # -- persistence (signed pickle, same integrity check as DriftGuardClassifier) ------------
    def save(self, path: str | Path) -> None:
        from driftguard.security import sign_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # the encoder is not pickled: it is re-created lazily from its name
        path.write_bytes(pickle.dumps(self))
        sign_file(path)

    @classmethod
    def load(cls, path: str | Path, allow_unsigned: bool = False) -> "HybridDetector":
        path = Path(path)
        if allow_unsigned:
            data = path.read_bytes()
        else:
            from driftguard.security import read_verified
            data = read_verified(path)           # signature is checked over exactly these bytes
        obj = pickle.loads(data)  # nosec B301 - signature verified above
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a HybridDetector")
        return obj


def load_detector(path: str | Path, allow_unsigned: bool = False):
    """Load either a DriftGuardClassifier or a HybridDetector from a signed pickle (one verified read)."""
    from driftguard.detector.classifier import DriftGuardClassifier
    path = Path(path)
    if allow_unsigned:
        data = path.read_bytes()
    else:
        from driftguard.security import read_verified
        data = read_verified(path)
    obj = pickle.loads(data)  # nosec B301 - signature verified above
    if isinstance(obj, HybridDetector):
        return obj
    if isinstance(obj, dict):
        return DriftGuardClassifier._from_state(obj)
    raise TypeError(f"{path}: unrecognised model file")
