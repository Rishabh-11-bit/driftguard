"""Ensemble detector: shipped hybrid + fine-tuned MiniLM + an independent published detector, stacked.

Why (docs/EVIDENCE.md, C1-C3): the memory-note detectors catch memory-poisoning text but over-flag generic chat
prompts; a generic prompt-injection detector (ProtectAI DeBERTa v2, Apache-2.0) ranks generic prompts well but misses
memory-write attacks at its default threshold. A 3-input logistic regression on the logits of the three scorers,
fitted leave-one-group-out on public data (scripts.run_stacking), tracks the best single scorer on every group.

    P(malicious) = sigmoid( coef . [logit(hybrid), logit(finetuned), logit(external)] + intercept - logit(threshold) )
    P(drift), P(ordinary) = the remaining mass split in the hybrid's feature-detector ratio

Each of the three scorers sees the note AND its sentence windows and keeps the maximum, so an attack buried in benign
sentences cannot be diluted (docs/EVIDENCE.md, C4).

Persistence uses no pickle for the new parts: a JSON config (HMAC/SHA-256 signed like every model file), the hybrid's
signed pickle, a signed safetensors file for the fine-tuned weights, and the external model pinned by name + revision.
The JSON must be loaded through load_detector(); the service does this when DRIFTGUARD_MODEL_PATH ends in .json.
Cost: roughly 2-5x the latency of the hybrid alone and ~1 GB of model files (the external detector is a DeBERTa-base).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from driftguard.detector.hybrid import windows

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"   # pinned Hugging Face commit (supply-chain safety)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _wmax(fn, texts: list[str]) -> np.ndarray:
    groups = [windows(t) for t in texts]
    flat = np.asarray(fn([w for g in groups for w in g]), dtype=float)
    out, i = [], 0
    for g in groups:
        out.append(flat[i:i + len(g)].max())
        i += len(g)
    return np.array(out)


class FineTunedEncoder:
    """MiniLM encoder + linear head -> P(attack), weights from signed safetensors bytes."""

    def __init__(self, weights: bytes, name: str = ENCODER) -> None:
        import torch
        from safetensors.torch import load as st_load
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=ENCODER_REVISION)
        self.enc = AutoModel.from_pretrained(name, revision=ENCODER_REVISION)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        state = st_load(weights)
        self.enc.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith("enc.")})
        self.head.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("head.")})
        self.enc.eval(), self.head.eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        torch, out = self._torch, []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                b = self.tok(texts[i:i + 64], padding=True, truncation=True, max_length=96, return_tensors="pt")
                h = self.enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                out.append(torch.sigmoid(self.head((h * m).sum(1) / m.sum(1)).squeeze(-1)).numpy())
        return np.concatenate(out) if out else np.array([])


class ExternalDetector:
    """A published Hugging Face text-classification prompt-injection detector, pinned to a revision."""

    def __init__(self, name: str, revision: str | None = None) -> None:
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=name, revision=revision, truncation=True, max_length=512)

    def __call__(self, texts: list[str]) -> np.ndarray:
        res = self.pipe(list(texts), batch_size=16)
        return np.array([r["score"] if r["label"].upper().startswith("INJ") else 1 - r["score"] for r in res])


class EnsembleDetector:
    needs_text = True

    def __init__(self, hybrid, finetuned, external, coef, intercept: float, threshold: float = 0.5) -> None:
        self.hybrid, self.finetuned, self.external = hybrid, finetuned, external
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.threshold = float(threshold)

    @property
    def _feature_names(self):
        return self.hybrid._feature_names

    @property
    def _is_fitted(self) -> bool:
        return self.hybrid._is_fitted

    def p_malicious(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        p_h = self.hybrid.predict_proba_text(X, texts)[:, 0]
        z = np.stack([_logit(p_h), _logit(_wmax(self.finetuned, texts)), _logit(_wmax(self.external, texts))], axis=1)
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept - _logit(self.threshold))))

    def predict_proba_text(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        p_mal = self.p_malicious(X, texts)
        base = self.hybrid.base.predict_proba(X)
        rest = base[:, 1] + base[:, 2]
        share = np.where(rest > 1e-6, base[:, 1] / np.clip(rest, 1e-6, None), 0.5)
        return np.stack([p_mal, (1 - p_mal) * share, (1 - p_mal) * (1 - share)], axis=1)

    def predict_one(self, features: np.ndarray, content: str | None = None):
        if content is None:
            raise ValueError("EnsembleDetector needs the write content (needs_text=True)")
        proba = self.predict_proba_text(np.asarray(features, dtype=np.float32).reshape(1, -1), [content])[0]
        return int(np.argmax(proba)), proba

    def predict_proba(self, X, texts=None):
        if texts is None:
            raise ValueError("EnsembleDetector.predict_proba needs texts")
        return self.predict_proba_text(X, texts)

    @classmethod
    def from_config(cls, path: str | Path) -> "EnsembleDetector":
        """Load from a signed JSON config; every referenced file is verified before it is used."""
        from driftguard.detector.hybrid import load_detector
        from driftguard.security import read_verified
        path = Path(path)
        cfg = json.loads(read_verified(path))
        base = path.parent
        hybrid = load_detector(base / cfg["hybrid"])
        finetuned = FineTunedEncoder(read_verified(base / cfg["finetuned"]))
        external = ExternalDetector(cfg["external"]["name"], cfg["external"].get("revision"))
        return cls(hybrid, finetuned, external, cfg["coef"], cfg["intercept"], cfg.get("threshold", 0.5))
