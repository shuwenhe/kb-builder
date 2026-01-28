"""Load knowledge bases from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from .schemas import ChunkRecord, Manifest


@dataclass
class KnowledgeBase:
    """In-memory knowledge base representation."""
    index: faiss.Index
    chunks: Dict[int, ChunkRecord]
    manifest: Manifest


def load_kb(kb_path: str) -> KnowledgeBase:
    """
    Load a knowledge base from disk.
    
    Args:
        kb_path: Path to the knowledge base directory (e.g., "kb/current" or "kb/versions/123456")
    
    Returns:
        KnowledgeBase object with index, chunks, and manifest
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data format is invalid
    """
    kb_dir = Path(kb_path)
    
    index_file = kb_dir / "index.faiss"
    chunks_file = kb_dir / "chunks.jsonl"
    manifest_file = kb_dir / "manifest.json"
    
    # Validate files exist
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
    
    # Load FAISS index
    index = faiss.read_index(str(index_file))
    
    # Load chunks
    chunks: Dict[int, ChunkRecord] = {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = ChunkRecord.model_validate_json(line)
                chunks[record.vector_id] = record
    
    # Load manifest
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        manifest = Manifest(**manifest_data)
    
    return KnowledgeBase(
        index=index,
        chunks=chunks,
        manifest=manifest,
    )
