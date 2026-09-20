"""
LLM interface — thin wrapper supporting OpenAI and Ollama providers.

Controlled via environment variable LLM_PROVIDER = "openai" | "ollama".

Author: Aryaan Sethi (1032231670) — Objective 1
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_llm(provider: str | None = None, **kwargs: Any):
    """
    Return a LangChain-compatible chat model.

    Parameters
    ----------
    provider : str | None
        "openai" or "ollama". Reads LLM_PROVIDER env var if not supplied.
    **kwargs
        Passed directly to the underlying chat model constructor.

    Returns
    -------
    BaseChatModel
        A LangChain chat model ready to use in a LangGraph graph.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = kwargs.pop("model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or set LLM_PROVIDER=ollama."
            )
        return ChatOpenAI(model=model, api_key=api_key, **kwargs)

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama

        model = kwargs.pop("model", os.getenv("OLLAMA_MODEL", "mistral:7b"))
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, **kwargs)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. "
            "Supported values: 'openai', 'ollama'."
        )
