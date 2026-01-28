#!/usr/bin/env python3
"""Example: Build a knowledge base from documents."""

import os
import sys
from langchain_community.embeddings import OllamaEmbeddings

from kb_builder import build_kb


def main():
    # Configuration
    source_dir = os.getenv("SOURCE_DIR", "./docs")
    out_dir = os.getenv("OUT_DIR", "./kb")
    embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    max_len = int(os.getenv("MAX_LEN", "800"))
    overlap = int(os.getenv("OVERLAP", "100"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    
    # Validate source directory
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        print("Set SOURCE_DIR environment variable or create ./docs directory")
        sys.exit(1)
    
    print("=" * 60)
    print("Knowledge Base Builder")
    print("=" * 60)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Embedding model:  {embed_model}")
    print(f"Ollama base URL:  {ollama_base_url}")
    print(f"Max chunk length: {max_len}")
    print(f"Chunk overlap:    {overlap}")
    print(f"Batch size:       {batch_size}")
    print("=" * 60)
    print()
    
    # Create embeddings client
    embeddings = OllamaEmbeddings(
        model=embed_model,
        base_url=ollama_base_url,
    )
    
    # Build KB
    try:
        manifest = build_kb(
            source_dir=source_dir,
            out_dir=out_dir,
            embeddings_client=embeddings,
            embed_model=embed_model,
            provider="ollama",
            max_len=max_len,
            overlap=overlap,
            batch_size=batch_size,
            show_markdown=False,
        )
        
        print()
        print("=" * 60)
        print("Build completed successfully!")
        print("=" * 60)
        print(f"KB version:    {manifest.kb_version}")
        print(f"Document count: {manifest.doc_count}")
        print(f"Chunk count:   {manifest.chunk_count}")
        print(f"Failed files:  {len(manifest.failed_files)}")
        if manifest.failed_files:
            print("\nFailed files:")
            for path in manifest.failed_files[:10]:
                print(f"  - {path}")
            if len(manifest.failed_files) > 10:
                print(f"  ... and {len(manifest.failed_files) - 10} more")
        print()
        print(f"Knowledge base available at: {out_dir}/current")
        print("=" * 60)
    
    except Exception as e:
        print(f"\nError during KB build: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
