"""Embedding provider implementation using ChromaDB's ONNX-based function.

This avoids the heavy PyTorch / sentence-transformers dependency by
relying on ChromaDB's built-in ONNX runtime embedding function
(all-MiniLM-L6-v2, 384 dimensions).
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from memory.exceptions import MemoryEmbeddingError
from memory.interfaces import IEmbeddingProvider

_logger = logging.getLogger(__name__)


class ChromaEmbeddingProvider(IEmbeddingProvider):
    """Embedding provider wrapping ChromaDB's default ONNX embedding function.

    The model (all-MiniLM-L6-v2) is downloaded on first use and cached
    locally by ChromaDB.  All embedding calls are run in a thread
    executor to avoid blocking the async event loop.
    """

    def __init__(self) -> None:
        self._ef = _get_chroma_embedding_function()
        self._dim: int = 384

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._embed_sync, text)
            return result
        except Exception as exc:
            raise MemoryEmbeddingError(f"Embedding failed: {exc}") from exc

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._embed_many_sync, texts)
        except Exception as exc:
            raise MemoryEmbeddingError(f"Batch embedding failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Synchronous helpers (run in executor)
    # ------------------------------------------------------------------

    def _embed_sync(self, text: str) -> list[float]:
        result = self._ef([text])
        return result[0].tolist()

    def _embed_many_sync(self, texts: list[str]) -> list[list[float]]:
        results = self._ef(texts)
        return [r.tolist() for r in results]


@lru_cache(maxsize=1)
def _get_chroma_embedding_function():
    """Lazily import and cache ChromaDB's embedding function.

    Cached so the ONNX model is loaded only once per process.
    """
    try:
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise MemoryEmbeddingError(
            "chromadb is required for embeddings. Install: pip install chromadb"
        ) from exc
    _logger.info("Loading ChromaDB ONNX embedding model (all-MiniLM-L6-v2) ...")
    ef = embedding_functions.DefaultEmbeddingFunction()
    _logger.info("Embedding model loaded (dimension=%d)", 384)
    return ef
