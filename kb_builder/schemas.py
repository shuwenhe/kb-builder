"""Data schemas for KB Builder."""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """Represents a single chunk in the knowledge base."""
    vector_id: int
    chunk_id: str
    file_path: str
    title_path: Union[List[str], str, None]
    chunk_type: str
    chunk_index: int
    doc_hash: str
    text_for_embedding: str
    excerpt_markdown: str


class Manifest(BaseModel):
    """Knowledge base build manifest with metadata."""
    kb_version: str
    source_dir: str
    build_time: str
    embedding_model: str
    llm_provider_default: str
    faiss_metric: str
    doc_count: int
    chunk_count: int
    failed_files: List[str] = Field(default_factory=list)
