#!/usr/bin/env python3
"""Example: Load and query a knowledge base."""

import os
import sys
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings

from kb_builder import load_kb


def main():
    kb_path = os.getenv("KB_PATH", "./kb/current")
    embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    if not os.path.exists(kb_path):
        print(f"Error: Knowledge base not found: {kb_path}")
        print("Run example_build.py first or set KB_PATH environment variable")
        sys.exit(1)
    
    print("=" * 60)
    print("Knowledge Base Query Example")
    print("=" * 60)
    print(f"KB path:         {kb_path}")
    print(f"Embedding model: {embed_model}")
    print(f"Ollama base URL: {ollama_base_url}")
    print()
    
    # Load KB
    print("Loading knowledge base...")
    kb = load_kb(kb_path)
    
    print(f"✓ Loaded successfully")
    print(f"  - KB version:    {kb.manifest.kb_version}")
    print(f"  - Document count: {kb.manifest.doc_count}")
    print(f"  - Chunk count:   {kb.manifest.chunk_count}")
    print(f"  - FAISS index:   {kb.index.ntotal} vectors, {kb.index.d} dimensions")
    print()
    
    # Create embeddings client
    embeddings = OllamaEmbeddings(
        model=embed_model,
        base_url=ollama_base_url,
    )
    
    # Interactive query loop
    print("=" * 60)
    print("Enter queries (or 'quit' to exit)")
    print("=" * 60)
    print()
    
    while True:
        query = input("Query: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        
        print()
        
        try:
            # Embed query
            query_vector = embeddings.embed_query(query)
            query_array = np.array([query_vector], dtype="float32")
            
            # Normalize for cosine similarity
            import faiss
            faiss.normalize_L2(query_array)
            
            # Search
            k = 5
            scores, indices = kb.index.search(query_array, k)
            
            print(f"Top {k} results:")
            print()
            
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
                if idx == -1:
                    break
                
                chunk = kb.chunks.get(int(idx))
                if not chunk:
                    continue
                
                print(f"[{rank}] Score: {score:.4f}")
                print(f"    File:  {chunk.file_path}")
                
                if chunk.title_path and isinstance(chunk.title_path, list):
                    title = " > ".join(chunk.title_path)
                    if title:
                        print(f"    Title: {title}")
                
                excerpt = chunk.excerpt_markdown
                if len(excerpt) > 150:
                    excerpt = excerpt[:150].rstrip() + "..."
                print(f"    Text:  {excerpt}")
                print()
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            print()
    
    print("Goodbye!")


if __name__ == "__main__":
    main()
