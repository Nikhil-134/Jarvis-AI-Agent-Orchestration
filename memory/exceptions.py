"""Memory system exception types."""


class MemoryError(Exception):
    """Base exception for memory system failures."""


class MemoryStorageError(MemoryError):
    """Raised when a storage backend (vector or document) operation fails."""


class MemoryRetrievalError(MemoryError):
    """Raised when a memory retrieval operation fails."""


class MemorySearchError(MemoryError):
    """Raised when a semantic search operation fails."""


class MemoryEmbeddingError(MemoryError):
    """Raised when embedding generation fails."""


class MemoryNotFoundError(MemoryError):
    """Raised when a requested memory does not exist."""


class MemoryConfigurationError(MemoryError):
    """Raised when memory system configuration is invalid."""


class MemoryDeduplicationError(MemoryError):
    """Raised when deduplication processing fails."""
