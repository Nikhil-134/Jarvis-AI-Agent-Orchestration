"""ChromaDB-backed vector store implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from memory.exceptions import MemorySearchError, MemoryStorageError
from memory.interfaces import IVectorStore

_logger = logging.getLogger(__name__)


class ChromaVectorStore(IVectorStore):
    """Vector store backed by ChromaDB's persistent client.

    All ChromaDB calls are dispatched to a thread executor so the
    async event loop is never blocked by disk I/O.
    """

    def __init__(self, path: str, collection_name: str = "jarvis_memory") -> None:
        self._path = path
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    async def initialize(self) -> None:
        """Open (or create) the persistent ChromaDB collection."""
        def _init() -> tuple[Any, Any]:
            import chromadb
            client = chromadb.PersistentClient(path=self._path)
            collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return client, collection

        loop = asyncio.get_running_loop()
        self._client, self._collection = await loop.run_in_executor(None, _init)
        _logger.info(
            "ChromaVectorStore initialised at '%s' (collection=%s)",
            self._path, self._collection_name,
        )

    async def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        self._ensure_ready()

        if ids is None:
            from uuid import uuid4
            ids = [str(uuid4()) for _ in vectors]

        try:
            loop = asyncio.get_running_loop()

            def _add() -> None:
                self._collection.add(
                    embeddings=vectors,
                    metadatas=metadata,
                    ids=ids,
                )

            await loop.run_in_executor(None, _add)
            _logger.debug("Stored %d vectors in ChromaDB", len(vectors))
            return ids
        except Exception as exc:
            raise MemoryStorageError(f"Failed to store vectors: {exc}") from exc

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_ready()

        try:
            loop = asyncio.get_running_loop()

            def _search() -> list[dict[str, Any]]:
                where = metadata_filter
                results = self._collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    where=where,
                    include=["metadatas", "distances", "embeddings"],
                )
                return _flatten_results(results)

            return await loop.run_in_executor(None, _search)
        except Exception as exc:
            raise MemorySearchError(f"Vector search failed: {exc}") from exc

    async def delete(self, ids: list[str]) -> None:
        self._ensure_ready()

        try:
            loop = asyncio.get_running_loop()

            def _delete() -> None:
                self._collection.delete(ids=ids)

            await loop.run_in_executor(None, _delete)
            _logger.debug("Deleted %d vectors from ChromaDB", len(ids))
        except Exception as exc:
            raise MemoryStorageError(f"Failed to delete vectors: {exc}") from exc

    async def count(self) -> int:
        """Return the number of vectors in the collection."""
        self._ensure_ready()
        loop = asyncio.get_running_loop()

        def _count() -> int:
            return self._collection.count()

        return await loop.run_in_executor(None, _count)

    def _ensure_ready(self) -> None:
        if self._collection is None:
            raise RuntimeError(
                "ChromaVectorStore not initialised. Call .initialize() first."
            )


def _flatten_results(results: Any) -> list[dict[str, Any]]:
    """Convert ChromaDB's nested query result into a flat list of dicts.

    ChromaDB returns::

        {
            "ids": [["id1", "id2", ...]],
            "distances": [[0.1, 0.2, ...]],
            "metadatas": [[{...}, {...}]],
            "embeddings": [[...]]
        }

    We flatten to::

        [{"id": "id1", "score": 0.9, ...}, {"id": "id2", "score": 0.8, ...}]
    """
    if not results or not results.get("ids"):
        return []

    flat: list[dict[str, Any]] = []
    ids = results["ids"][0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    embeddings = results.get("embeddings", [[]])[0] if results.get("embeddings") else [None] * len(ids)

    for idx, doc_id in enumerate(ids):
        entry: dict[str, Any] = {
            "id": doc_id,
            "score": 1.0 - min(distances[idx], 1.0) if idx < len(distances) else 0.0,
        }
        if idx < len(metadatas) and metadatas[idx]:
            entry.update(metadatas[idx])
        if idx < len(embeddings) and embeddings[idx] is not None:
            entry["embedding"] = embeddings[idx]
        flat.append(entry)

    return flat
