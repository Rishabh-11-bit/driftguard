"""
Memory backend — abstract base and concrete FAISS/Chroma implementations.

Every backend exposes a uniform interface so the agent and the write pipeline
are decoupled from the vector store implementation.

Author: Aryaan Sethi (1032231670) — Objective 1
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────

class MemoryBackend(ABC):
    """Abstract long-term memory backend."""

    @abstractmethod
    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a text entry to memory. Returns the entry ID."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the k most semantically similar entries to query."""

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID. Returns True if found and deleted."""

    @abstractmethod
    def list_all(self) -> list[dict[str, Any]]:
        """Return all entries in the memory store."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the store."""

    def __len__(self) -> int:
        return len(self.list_all())


# ─────────────────────────────────────────────────────────────
# FAISS backend (default — no server required)
# ─────────────────────────────────────────────────────────────

class FAISSMemoryBackend(MemoryBackend):
    """
    In-process FAISS-based long-term memory backend.

    Embeddings are computed with SentenceTransformer and stored alongside
    their text + metadata in a parallel JSON store.  The FAISS index and
    the JSON store are both persisted to disk on every write.

    Parameters
    ----------
    index_path : str | Path
        Directory where the FAISS index and metadata JSON are stored.
    embedding_model : str
        SentenceTransformer model name (default: all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        index_path: str | Path = "./data/faiss_index",
        embedding_model: str | None = None,
    ) -> None:
        import faiss  # type: ignore

        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._model_name = (
            embedding_model
            or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self._encoder = SentenceTransformer(self._model_name)
        self._dim = self._encoder.get_sentence_embedding_dimension()

        self._index_file = self.index_path / "index.faiss"
        self._meta_file  = self.index_path / "metadata.json"

        # Load or create index
        if self._index_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            with self._meta_file.open("r") as f:
                self._metadata: list[dict[str, Any]] = json.load(f)
        else:
            self._index = faiss.IndexFlatIP(self._dim)  # inner-product (cosine after norm)
            self._metadata = []

        logger.info(
            f"[FAISSMemory] Loaded {len(self._metadata)} entries "
            f"from {self.index_path}."
        )

    # ── MemoryBackend API ────────────────────────────────────

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        import faiss  # type: ignore

        entry_id = str(uuid.uuid4())
        emb = self._embed([text])[0]
        self._index.add(emb.reshape(1, -1))
        self._metadata.append({
            "id": entry_id,
            "text": text,
            "metadata": metadata or {},
            "index_pos": len(self._metadata),
        })
        self._persist()
        logger.debug(f"[FAISSMemory] Added entry {entry_id[:8]}…")
        return entry_id

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if len(self._metadata) == 0:
            return []
        k = min(k, len(self._metadata))
        emb = self._embed([query])[0].reshape(1, -1)
        scores, indices = self._index.search(emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self._metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    def delete(self, entry_id: str) -> bool:
        # FAISS IndexFlatIP doesn't support deletion natively;
        # we mark as deleted in metadata and rebuild index.
        before = len(self._metadata)
        self._metadata = [e for e in self._metadata if e["id"] != entry_id]
        if len(self._metadata) == before:
            return False
        self._rebuild_index()
        self._persist()
        return True

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._metadata)

    def clear(self) -> None:
        import faiss  # type: ignore
        self._index = faiss.IndexFlatIP(self._dim)
        self._metadata = []
        self._persist()

    # ── internals ────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings for inner-product cosine sim."""
        embs = self._encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype(np.float32)

    def _rebuild_index(self) -> None:
        import faiss  # type: ignore
        self._index = faiss.IndexFlatIP(self._dim)
        if self._metadata:
            texts = [e["text"] for e in self._metadata]
            embs = self._embed(texts)
            self._index.add(embs)

    def _persist(self) -> None:
        import faiss  # type: ignore
        faiss.write_index(self._index, str(self._index_file))
        with self._meta_file.open("w") as f:
            json.dump(self._metadata, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Chroma backend (optional — requires docker-compose or hosted)
# ─────────────────────────────────────────────────────────────

class ChromaMemoryBackend(MemoryBackend):
    """
    ChromaDB-based long-term memory backend.

    Requires a running ChromaDB server (see docker-compose.yml).
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str = "driftguard_memory",
    ) -> None:
        import chromadb  # type: ignore

        host = host or os.getenv("CHROMA_HOST", "localhost")
        port = port or int(os.getenv("CHROMA_PORT", "8000"))
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(collection_name)
        logger.info(
            f"[ChromaMemory] Connected to Chroma at {host}:{port}, "
            f"collection={collection_name!r}"
        )

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        entry_id = str(uuid.uuid4())
        # Chroma handles its own embedding; metadata values must be str/int/float/bool
        safe_meta = _flatten_metadata(metadata or {})
        self._collection.add(
            documents=[text],
            metadatas=[safe_meta],
            ids=[entry_id],
        )
        return entry_id

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        results = self._collection.query(query_texts=[query], n_results=min(k, self._collection.count()))
        out = []
        for doc, meta, dist, eid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            out.append({"id": eid, "text": doc, "metadata": meta, "score": 1 - dist})
        return out

    def delete(self, entry_id: str) -> bool:
        try:
            self._collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False

    def list_all(self) -> list[dict[str, Any]]:
        all_data = self._collection.get()
        out = []
        for eid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            out.append({"id": eid, "text": doc, "metadata": meta})
        return out

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(name)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def get_memory_backend(backend: str | None = None, **kwargs: Any) -> MemoryBackend:
    """
    Return a MemoryBackend instance based on the MEMORY_BACKEND env var.

    Options: "faiss" (default) | "chroma"
    """
    backend = backend or os.getenv("MEMORY_BACKEND", "faiss")
    if backend == "faiss":
        return FAISSMemoryBackend(**kwargs)
    elif backend == "chroma":
        return ChromaMemoryBackend(**kwargs)
    else:
        raise ValueError(f"Unknown MEMORY_BACKEND={backend!r}. Options: 'faiss', 'chroma'.")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _flatten_metadata(meta: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dot-separated string keys with scalar values."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_metadata(v, key))
        elif isinstance(v, (str, int, float, bool)):
            out[key] = v
        else:
            out[key] = str(v)
    return out
