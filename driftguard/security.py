"""Security helpers: input normalisation and model-artefact integrity.

Threats addressed (see SECURITY.md):
  * T1  Unicode / invisible-character evasion of the text features
  * T2  Resource exhaustion through oversized writes
  * T3  Arbitrary code execution through a tampered pickle model file
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import unicodedata
from pathlib import Path

MAX_WRITE_CHARS = 2000
MAX_QUERY_CHARS = 2000

# Zero-width, bidi-override and other format/control characters used to hide
# or reorder text. Category "Cf" covers all of them; "Cc" covers C0/C1 controls.
_STRIP_CATEGORIES = {"Cf", "Cc", "Co", "Cs"}
_WS = re.compile(r"\s+")


def normalise_text(text: str, max_chars: int = MAX_WRITE_CHARS) -> str:
    """NFKC-normalise, drop invisible/control characters, collapse whitespace, cap length.

    NFKC folds full-width and compatibility forms (e.g. ＡＣＴＩＶＡＴＥ -> ACTIVATE)
    that otherwise slip past lexical features. It does NOT fold cross-script
    homoglyphs (Cyrillic 'а' vs Latin 'a'); see SECURITY.md for that residual risk.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t ":
            out.append(ch)
        elif cat == "Cc":                 # control characters act as separators
            out.append(" ")
        elif cat in _STRIP_CATEGORIES:    # zero-width / bidi / private-use: delete, don't split words
            continue
        else:
            out.append(ch)
    text = "".join(out)
    text = _WS.sub(" ", text).strip()
    return text[:max_chars]


# ── model-file integrity ────────────────────────────────────

class ModelIntegrityError(RuntimeError):
    """Raised when a model artefact is unsigned or its signature does not match."""


def _sig_path(path: Path) -> Path:
    return path.with_name(path.name + ".sig")


def _digest(path: Path, key: bytes | None) -> tuple[str, str]:
    data = path.read_bytes()
    if key:
        return "hmac-sha256", hmac.new(key, data, hashlib.sha256).hexdigest()
    return "sha256", hashlib.sha256(data).hexdigest()


def _key() -> bytes | None:
    k = os.getenv("DRIFTGUARD_MODEL_KEY")
    return k.encode() if k else None


def sign_file(path: str | Path) -> Path:
    """Write `<path>.sig`. Uses HMAC-SHA256 when DRIFTGUARD_MODEL_KEY is set, else plain SHA-256."""
    path = Path(path)
    alg, digest = _digest(path, _key())
    sig = _sig_path(path)
    sig.write_text(json.dumps({"alg": alg, "digest": digest}), encoding="utf-8")
    return sig


def verify_file(path: str | Path) -> None:
    """Raise ModelIntegrityError unless `<path>.sig` exists and matches the file.

    With DRIFTGUARD_MODEL_KEY set, only an HMAC signature is accepted, so an
    attacker who can write both files still cannot forge it without the key.
    Plain SHA-256 detects corruption and a swapped model file, but not an
    attacker who can also rewrite the .sig file.
    """
    path = Path(path)
    sig = _sig_path(path)
    if not sig.exists():
        raise ModelIntegrityError(
            f"{path} has no signature file. Models are pickles and can execute code "
            f"on load; only load files you created. Re-sign trusted files with "
            f"`python -m scripts.sign_model {path}`."
        )
    meta = json.loads(sig.read_text(encoding="utf-8"))
    key = _key()
    if key and meta.get("alg") != "hmac-sha256":
        raise ModelIntegrityError(f"{path}: DRIFTGUARD_MODEL_KEY is set but signature is not an HMAC.")
    alg, digest = _digest(path, key if meta.get("alg") == "hmac-sha256" else None)
    if not hmac.compare_digest(digest, meta.get("digest", "")):
        raise ModelIntegrityError(f"{path}: signature mismatch - file was modified or is corrupt.")
