"""Tiny client for a local Ollama server (Llama etc.) used by the demo UI.

Uses Ollama's HTTP API directly, so no extra Python dependency is needed.
"""

from __future__ import annotations

import os

import requests

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM = (
    "You are a personal assistant with long-term memory about the user. "
    "Follow the user's stored preferences and instructions in the memory list. "
    "Answer in at most 3 short sentences."
)


def list_models() -> list[str]:
    """Names of locally installed models, or [] if Ollama is not reachable."""
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def answer(model: str, memories: list[str], question: str, timeout: int = 120) -> str:
    """Ask the local model `question`, conditioned on `memories` as its memory."""
    mem_block = "\n".join(f"- {m}" for m in memories) or "- (empty)"
    r = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0, "seed": 7, "num_predict": 160},
            "messages": [
                {"role": "system", "content": f"{SYSTEM}\n\nMEMORY:\n{mem_block}"},
                {"role": "user", "content": question},
            ],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
