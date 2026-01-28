"""Tests for KB Builder schemas."""

import pytest
from kb_builder.schemas import ChunkRecord, Manifest


def test_chunk_record_creation():
    """Test creating ChunkRecord."""
    chunk = ChunkRecord(
        vector_id=0,
        chunk_id="abc123",
        file_path="docs/test.docx",
        title_path=["Chapter 1", "Section 1.1"],
        chunk_type="paragraph",
        chunk_index=0,
        doc_hash="def456",
        text_for_embedding="Sample text for embedding",
        excerpt_markdown="Sample excerpt",
    )
    
    assert chunk.vector_id == 0
    assert chunk.chunk_id == "abc123"
    assert chunk.chunk_type == "paragraph"
    assert len(chunk.title_path) == 2


def test_manifest_creation():
    """Test creating Manifest."""
    manifest = Manifest(
        kb_version="20240101-120000",
        source_dir="./docs",
        build_time="2024-01-01T12:00:00Z",
        embedding_model="mxbai-embed-large",
        llm_provider_default="ollama",
        faiss_metric="cosine",
        doc_count=10,
        chunk_count=100,
        failed_files=["docs/bad.doc"],
    )
    
    assert manifest.kb_version == "20240101-120000"
    assert manifest.doc_count == 10
    assert manifest.chunk_count == 100
    assert len(manifest.failed_files) == 1


def test_manifest_default_failed_files():
    """Test Manifest with default failed_files."""
    manifest = Manifest(
        kb_version="20240101-120000",
        source_dir="./docs",
        build_time="2024-01-01T12:00:00Z",
        embedding_model="mxbai-embed-large",
        llm_provider_default="ollama",
        faiss_metric="cosine",
        doc_count=5,
        chunk_count=50,
    )
    
    assert manifest.failed_files == []
