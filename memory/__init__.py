"""Memory package exports."""

from typing import Any

__all__ = [
    "IDocumentStore",
    "IEmbeddingProvider",
    "IMemoryStore",
    "IVectorStore",
]


def __getattr__(name: str) -> Any:
    if name in {"IVectorStore", "IDocumentStore", "IMemoryStore", "IEmbeddingProvider"}:
        from memory.interfaces import (
            IDocumentStore,
            IEmbeddingProvider,
            IMemoryStore,
            IVectorStore,
        )
        return {
            "IVectorStore": IVectorStore,
            "IDocumentStore": IDocumentStore,
            "IMemoryStore": IMemoryStore,
            "IEmbeddingProvider": IEmbeddingProvider,
        }[name]
    raise AttributeError(f"module 'memory' has no attribute {name!r}")
