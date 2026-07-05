"""Memory package exports."""

from typing import Any

__all__ = [
    "ChromaEmbeddingProvider",
    "ChromaVectorStore",
    "IDocumentStore",
    "IEmbeddingProvider",
    "IMemoryStore",
    "IVectorStore",
    "MemoryItem",
    "MemoryManager",
    "MemoryService",
    "MemoryType",
    "SQLiteDocumentStore",
    "WorkingMemory",
    "calculate_importance",
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
    if name in {"MemoryItem", "MemoryType", "calculate_importance"}:
        from memory.models import MemoryItem, MemoryType, calculate_importance
        return {
            "MemoryItem": MemoryItem,
            "MemoryType": MemoryType,
            "calculate_importance": calculate_importance,
        }[name]
    if name == "ChromaEmbeddingProvider":
        from memory.embedding_provider import ChromaEmbeddingProvider
        return ChromaEmbeddingProvider
    if name == "ChromaVectorStore":
        from memory.vector_store import ChromaVectorStore
        return ChromaVectorStore
    if name == "SQLiteDocumentStore":
        from memory.document_store import SQLiteDocumentStore
        return SQLiteDocumentStore
    if name == "MemoryManager":
        from memory.memory_manager import MemoryManager
        return MemoryManager
    if name == "MemoryService":
        from memory.memory_service import MemoryService
        return MemoryService
    if name == "WorkingMemory":
        from memory.memory_manager import WorkingMemory
        return WorkingMemory
    raise AttributeError(f"module 'memory' has no attribute {name!r}")
