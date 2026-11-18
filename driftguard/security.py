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
    text = _fold_confusables(text)
    text = _collapse_letter_spacing(text)
    text = _fold_leet(text)
    return text[:max_chars]


# Cyrillic / Greek letters that render like Latin ones. Folded only when the note is mostly Latin, so genuine
# Russian or Greek text is left alone (a note that is mostly Cyrillic is not touched at all).
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i", "ј": "j", "ѕ": "s",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "ο": "o", "ν": "v", "α": "a", "ε": "e", "ι": "i", "κ": "k", "ρ": "p", "τ": "t", "υ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Χ": "X", "Υ": "Y",
})
_LETTER_SPACED = re.compile(r"(?<![^\W\d_])(?:[^\W\d_] ){3,}[^\W\d_](?![^\W\d_])")
_LEET_TOKEN = re.compile(r"[A-Za-z0-9]{3,}")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
_BENIGN_ALNUM = re.compile(r"^(\d+[A-Za-z]{1,3}|[A-Za-z]{1,3}\d+|\d+[A-Za-z]?\d*)$")   # 3pm, 5km, mp3, v2, 2000, 4k


def _fold_confusables(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return text
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return text.translate(_CONFUSABLES) if latin / len(letters) >= 0.5 else text


def _collapse_letter_spacing(text: str) -> str:
    """'i g n o r e' -> 'ignore' (runs of at least four single letters separated by single spaces)."""
    return _LETTER_SPACED.sub(lambda m: m.group(0).replace(" ", ""), text)


def _fold_leet(text: str) -> str:
    """'1gn0r3' -> 'ignore'. Only mixed letter/digit tokens that are not ordinary quantities or names (3pm, mp3, v2)."""
    def fix(m):
        tok = m.group(0)
        if tok.isdigit() or tok.isalpha() or _BENIGN_ALNUM.match(tok):
            return tok
        return tok.translate(_LEET) if sum(c.isalpha() for c in tok) >= 2 else tok
    return _LEET_TOKEN.sub(fix, text)


# ── model-file integrity ────────────────────────────────────

class ModelIntegrityError(RuntimeError):
    """Raised when a model artefact is unsigned or its signature does not match."""


def _sig_path(path: Path) -> Path:
    return path.with_name(path.name + ".sig")


def _digest(path: Path, key: bytes | None, data: bytes | None = None) -> tuple[str, str]:
    if data is None:
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


def verify_file(path: str | Path, data: bytes | None = None) -> None:
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
    try:
        meta = json.loads(sig.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("signature is not an object")
    except (OSError, ValueError) as exc:
        raise ModelIntegrityError(f"{path}: unreadable signature file ({exc}).") from exc
    key = _key()
    if meta.get("alg") == "hmac-sha256" and not key:
        raise ModelIntegrityError(f"{path}: signature is an HMAC but DRIFTGUARD_MODEL_KEY is not set.")
    if key and meta.get("alg") != "hmac-sha256":
        raise ModelIntegrityError(f"{path}: DRIFTGUARD_MODEL_KEY is set but signature is not an HMAC.")
    alg, digest = _digest(path, key if meta.get("alg") == "hmac-sha256" else None, data)
    if not hmac.compare_digest(digest, str(meta.get("digest", ""))):
        raise ModelIntegrityError(f"{path}: signature mismatch - file was modified or is corrupt.")


def read_verified(path: str | Path) -> bytes:
    """Read the file once and verify the signature over those exact bytes (no check/use gap)."""
    path = Path(path)
    data = path.read_bytes()
    verify_file(path, data)
    return data
