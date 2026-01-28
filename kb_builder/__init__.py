"""KB Builder - Knowledge Base Builder for RAG Applications."""

from .builder import build_kb
from .loader import load_kb, KnowledgeBase
from .schemas import ChunkRecord, Manifest

__version__ = "0.1.0"

__all__ = [
    "build_kb",
    "load_kb",
    "KnowledgeBase",
    "ChunkRecord",
    "Manifest",
]
