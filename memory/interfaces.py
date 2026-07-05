"""Memory system interface definitions for Jarvis.

These interfaces define the contracts that concrete memory backends
(ChromaDB, SQLite, FAISS, etc.) will implement in future phases.
They are defined now to prevent architectural rewrites when memory
is added.
"""

from abc import ABC, abstractmethod
from typing import Any


class IVectorStore(ABC):
    """Interface for vector similarity search backends.

    Implementations: ChromaDB, FAISS, Qdrant, PGVector.
    """

    @abstractmethod
    async def add_vectors(
        self, vectors: list[list[float]], metadata: list[dict[str, Any]], ids: list[str] | None = None
    ) -> list[str]:
        """Store vectors with associated metadata and return their ids."""

    @abstractmethod
    async def search(
        self, query_vector: list[float], top_k: int = 10, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most similar vectors with metadata matching *metadata_filter*."""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Remove vectors by their ids."""


class IDocumentStore(ABC):
    """Interface for persistent document/episodic storage.

    Implementations: SQLite, Postgres, JSON file.
    """

    @abstractmethod
    async def store(self, collection: str, document: dict[str, Any]) -> str:
        """Store a document in *collection* and return its id."""

    @abstractmethod
    async def retrieve(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        """Return a document by id or None."""

    @abstractmethod
    async def search(
        self, collection: str, filters: dict[str, Any] | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return documents matching *filters*."""

    @abstractmethod
    async def delete(self, collection: str, doc_id: str) -> bool:
        """Delete a document, returning True if it existed."""


class IMemoryStore(ABC):
    """Interface for agent memory (combines vector + document + working memory).

    Implementations: MemoryManager (orchestrates VectorStore + DocumentStore).
    """

    @abstractmethod
    async def remember(self, key: str, data: dict[str, Any]) -> None:
        """Store a memory that can be later recalled."""

    @abstractmethod
    async def recall(self, key: str) -> dict[str, Any] | None:
        """Return a stored memory by key."""

    @abstractmethod
    async def search_similar(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantically search memories similar to *query*."""

    @abstractmethod
    async def forget(self, key: str) -> bool:
        """Remove a memory, returning True if it existed."""


class IEmbeddingProvider(ABC):
    """Interface for text-to-vector embedding models.

    Implementations: sentence-transformers, Ollama embeddings, OpenAI embeddings.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return a vector embedding for *text*."""

    @abstractmethod
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Return vector embeddings for multiple texts (batched)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of produced embeddings."""
