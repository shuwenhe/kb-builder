"""Build knowledge base from documents with FAISS indexing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from tqdm import tqdm

import sys
import os
# Import from our docx-parser project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../docx-parser'))
try:
    from docx_parser import blocks_from_text, parse_docx, parse_docx_with_markdown
except ImportError:
    # Fallback: create stub functions if docx-parser not available
    def blocks_from_text(text):
        """Stub function when docx-parser not available."""
        class Block:
            def __init__(self, text):
                self.block_type = "paragraph"
                self.text = text
                self.title_path = []
        return [Block(text)]
    
    def parse_docx(path):
        """Stub function when docx-parser not available."""
        return []
    
    def parse_docx_with_markdown(path):
        """Stub function when docx-parser not available."""
        return "", []

from .schemas import ChunkRecord, Manifest
from .utils import (
    iter_batches,
    normalize_text,
    safe_relpath,
    sha1_file,
    sha1_text,
    split_list_items,
    split_text,
    table_to_linearized_text,
    table_to_markdown,
)


DOC_EXTENSIONS = {".doc", ".docx"}


class UserAbort(Exception):
    """Raised when user aborts the build process."""
    pass


def _is_word_lock(name: str) -> bool:
    """Check if filename is a Word lock file."""
    return name.startswith("~$")


def scan_documents(source_dir: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Scan directory for Word documents.
    
    Returns:
        (included_files, skipped_files_with_reasons)
    """
    included: List[str] = []
    skipped: List[Dict[str, str]] = []
    
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if _is_word_lock(filename):
                skipped.append({"path": os.path.join(root, filename), "reason": "word_lock"})
                continue
            
            ext = os.path.splitext(filename)[1].lower()
            path = os.path.join(root, filename)
            
            try:
                if os.path.getsize(path) == 0:
                    skipped.append({"path": path, "reason": "empty_file"})
                    continue
            except OSError:
                skipped.append({"path": path, "reason": "stat_failed"})
                continue
            
            if ext in DOC_EXTENSIONS:
                included.append(path)
            else:
                skipped.append({"path": path, "reason": "unsupported_extension"})
    
    return included, skipped


def _which_binary(candidates: Iterable[str]) -> Optional[str]:
    """Find first available binary from candidates."""
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _expected_docx_path(path: str, output_dir: str) -> str:
    """Generate expected path for converted docx file."""
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]
    return os.path.join(output_dir, f"{stem}.docx")


def _convert_doc_with_unstructured(path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Try converting .doc to .docx using unstructured library."""
    try:
        from unstructured.partition.common.common import convert_office_doc
    except ImportError:
        return None, "unstructured_missing"
    
    if not _which_binary(["soffice"]):
        return None, "soffice_missing"
    
    try:
        convert_office_doc(
            path,
            output_dir,
            target_format="docx",
            target_filter="MS Word 2007 XML",
        )
    except Exception:
        return None, "doc_convert_failed_unstructured"
    
    converted = _expected_docx_path(path, output_dir)
    if os.path.exists(converted):
        return converted, None
    return None, "doc_convert_failed_unstructured"


def _convert_doc_with_soffice(path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Try converting .doc to .docx using LibreOffice/soffice."""
    binary = _which_binary(["soffice", "libreoffice"])
    if not binary:
        return None, "soffice_missing"
    
    cmd = [
        binary,
        "--headless",
        "--convert-to",
        "docx",
        "--outdir",
        output_dir,
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None, "doc_convert_failed_soffice"
    
    converted = _expected_docx_path(path, output_dir)
    if os.path.exists(converted):
        return converted, None
    return None, "doc_convert_failed_soffice"


def _convert_doc_with_textutil(path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Try converting .doc to .docx using macOS textutil."""
    textutil = _which_binary(["textutil"])
    if not textutil:
        return None, "textutil_missing"
    
    cmd = [textutil, "-convert", "docx", "-output", output_dir, path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "doc_convert_timeout_textutil"
    
    if result.returncode != 0:
        return None, "doc_convert_failed_textutil"
    
    converted = _expected_docx_path(path, output_dir)
    if os.path.exists(converted):
        return converted, None
    return None, "doc_convert_failed_textutil"


def _extract_doc_text_with_antiword(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract plain text from .doc using antiword as fallback."""
    antiword = _which_binary(["antiword"])
    if not antiword:
        return None, "antiword_missing"
    
    try:
        result = subprocess.run(
            [antiword, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None, "antiword_timeout"
    
    if result.returncode != 0:
        return None, "antiword_failed"
    
    return result.stdout, None


def convert_doc_to_docx(path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert legacy .doc to .docx using available converters.
    
    Tries multiple converters in order:
    1. unstructured library
    2. soffice/libreoffice
    3. textutil (macOS)
    
    Returns:
        (converted_path_or_None, error_reason_or_None)
    """
    converted, reason = _convert_doc_with_unstructured(path, output_dir)
    if converted:
        return converted, None
    
    converted, reason = _convert_doc_with_soffice(path, output_dir)
    if converted:
        return converted, None
    
    converted, reason = _convert_doc_with_textutil(path, output_dir)
    if converted:
        return converted, None
    
    return None, reason or "doc_convert_failed"


def _build_chunk_id(file_path: str, doc_hash: str, chunk_index: int, chunk_type: str) -> str:
    """Generate unique chunk ID."""
    return sha1_text(f"{file_path}:{doc_hash}:{chunk_type}:{chunk_index}")


def _title_prefix(title_path: List[str]) -> str:
    """Format title path as hierarchical string."""
    cleaned = [part for part in title_path if part]
    if not cleaned:
        return ""
    return " > ".join(cleaned)


def _attach_title_prefix(title_path: List[str], text: str) -> str:
    """Prepend title path to text for embedding."""
    prefix = _title_prefix(title_path)
    if not prefix:
        return text
    return f"{prefix}\n{text}"


def _find_repeated_short_paragraphs(
    blocks,
    max_len: int = 40,
    min_count: int = 3,
) -> set:
    """Find paragraphs that repeat frequently (likely headers/footers)."""
    counts: Dict[str, int] = {}
    for block in blocks:
        if block.block_type != "paragraph":
            continue
        text = normalize_text(block.text)
        if not text or len(text) > max_len:
            continue
        counts[text] = counts.get(text, 0) + 1
    return {text for text, count in counts.items() if count >= min_count}


def _shorten(text: str, limit: int = 200) -> str:
    """Truncate text to limit with ellipsis."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _split_table_rows(rows: List[List[str]], max_len: int) -> List[List[List[str]]]:
    """Split large table into smaller chunks by rows."""
    if not rows or len(rows) <= 1 or max_len <= 0:
        return [rows]
    
    header = rows[0]
    body = rows[1:]
    chunks: List[List[List[str]]] = []
    current: List[List[str]] = []
    current_len = 0
    
    for row in body:
        row_text = table_to_linearized_text([header, row])
        row_len = len(row_text)
        if current and current_len + row_len > max_len:
            chunks.append([header] + current)
            current = []
            current_len = 0
        current.append(row)
        current_len += row_len
    
    if current:
        chunks.append([header] + current)
    if not chunks:
        chunks.append(rows)
    return chunks


def _embed_documents_with_retry(embeddings, texts: List[str], max_retries: int = 3) -> List[List[float]]:
    """Embed documents with exponential backoff retry."""
    delay = 0.5
    last_exc: Optional[BaseException] = None
    
    for attempt in range(max_retries + 1):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(delay)
            delay *= 2
    
    raise RuntimeError(f"Embedding failed after retries: {last_exc}") from last_exc


def _collect_chunks(
    file_path: str,
    doc_hash: str,
    blocks,
    max_len: int,
    overlap: int,
) -> List[Dict[str, object]]:
    """
    Collect chunks from parsed document blocks.
    
    Handles:
    - Paragraphs with list splitting
    - Tables (split by rows if needed)
    - Title path hierarchy
    - Repeated short paragraph filtering
    """
    chunks: List[Dict[str, object]] = []
    repeated_short = _find_repeated_short_paragraphs(blocks)
    chunk_index = 0
    
    for block in blocks:
        if block.block_type == "paragraph":
            text = normalize_text(block.text)
            if not text:
                continue
            if text in repeated_short:
                continue
            
            # Split list items
            items = split_list_items(text)
            for item in items:
                # Split long text with overlap
                pieces = split_text(item, max_len=max_len, overlap=overlap)
                for piece in pieces:
                    embedding_text = _attach_title_prefix(block.title_path, piece)
                    chunks.append(
                        {
                            "chunk_id": _build_chunk_id(file_path, doc_hash, chunk_index, "paragraph"),
                            "chunk_type": "paragraph",
                            "chunk_index": chunk_index,
                            "file_path": file_path,
                            "title_path": block.title_path,
                            "text_for_embedding": embedding_text,
                            "excerpt_markdown": piece,
                        }
                    )
                    chunk_index += 1
        
        elif block.block_type == "table":
            # Try table_rows attribute first
            rows = getattr(block, "table_rows", None)
            if rows:
                table_chunks = _split_table_rows(rows, max_len=max_len)
                for table_rows in table_chunks:
                    table_markdown = table_to_markdown(table_rows)
                    table_text = table_to_linearized_text(table_rows)
                    if not table_markdown and not table_text:
                        continue
                    embedding_text = _attach_title_prefix(block.title_path, table_text or table_markdown)
                    chunks.append(
                        {
                            "chunk_id": _build_chunk_id(file_path, doc_hash, chunk_index, "table"),
                            "chunk_type": "table",
                            "chunk_index": chunk_index,
                            "file_path": file_path,
                            "title_path": block.title_path,
                            "text_for_embedding": embedding_text,
                            "excerpt_markdown": table_markdown or table_text,
                        }
                    )
                    chunk_index += 1
                continue
            
            # Fallback to precomputed markdown/text
            table_markdown = block.table_markdown or ""
            table_text = block.table_linearized_text or ""
            if not table_markdown and not table_text:
                continue
            
            chunks.append(
                {
                    "chunk_id": _build_chunk_id(file_path, doc_hash, chunk_index, "table"),
                    "chunk_type": "table",
                    "chunk_index": chunk_index,
                    "file_path": file_path,
                    "title_path": block.title_path,
                    "text_for_embedding": _attach_title_prefix(block.title_path, table_text or table_markdown),
                    "excerpt_markdown": table_markdown or table_text,
                }
            )
            chunk_index += 1
    
    return chunks


def build_kb(
    source_dir: str,
    out_dir: str,
    embeddings_client,
    embed_model: str,
    provider: str = "ollama",
    max_len: int = 800,
    overlap: int = 100,
    batch_size: int = 32,
    show_markdown: bool = False,
) -> Manifest:
    """
    Build FAISS knowledge base from documents.
    
    Args:
        source_dir: Directory containing .doc/.docx files
        out_dir: Output directory for KB (creates versions/ subdirectory)
        embeddings_client: LangChain embeddings instance (e.g., OllamaEmbeddings)
        embed_model: Name of embedding model
        provider: Provider name (for manifest metadata)
        max_len: Maximum chunk length
        overlap: Chunk overlap for splitting
        batch_size: Batch size for embedding
        show_markdown: Print debug markdown output
    
    Returns:
        Manifest with build metadata
        
    Raises:
        UserAbort: If user cancels at prompt
        RuntimeError: If no chunks generated or embedding fails
    """
    start_time = time.perf_counter()
    
    # Scan documents
    included, skipped = scan_documents(source_dir)
    suspicious_count = sum(1 for item in skipped if item["reason"] in ("empty_file", "stat_failed"))
    total_count = len(included) + len(skipped)
    
    print(
        "Scan summary: "
        f"total={total_count}, included={len(included)}, skipped={len(skipped)}, suspicious={suspicious_count}"
    )
    print("About to parse documents.")
    
    # Confirm with user if interactive
    if sys.stdin.isatty():
        answer = input("Proceed with rebuild? (y/n): ").strip().lower()
        if answer not in ("y", "yes"):
            raise UserAbort("KB rebuild aborted by user.")
    
    # Setup directories
    os.makedirs(out_dir, exist_ok=True)
    versions_dir = os.path.join(out_dir, "versions")
    os.makedirs(versions_dir, exist_ok=True)
    
    kb_version = datetime.now().strftime("%Y%m%d-%H%M%S")
    version_dir = os.path.join(versions_dir, kb_version)
    os.makedirs(version_dir, exist_ok=True)
    
    failed_files: List[Dict[str, str]] = []
    degraded_files: List[Dict[str, str]] = []
    records: List[ChunkRecord] = []
    temp_dir = tempfile.mkdtemp(prefix="kb_docx_")
    
    try:
        # Parse documents
        print("Starting document parsing...")
        progress = tqdm(included, desc="Parsing documents")
        
        for path in progress:
            progress.set_postfix_str(safe_relpath(path, source_dir))
            
            try:
                doc_hash = sha1_file(path)
                ext = os.path.splitext(path)[1].lower()
                blocks = None
                
                # Handle .doc conversion
                if ext == ".doc":
                    converted, reason = convert_doc_to_docx(path, temp_dir)
                    if converted:
                        if show_markdown:
                            markdown, blocks = parse_docx_with_markdown(converted)
                            tqdm.write(f"[MARKDOWN] {safe_relpath(path, source_dir)}")
                            tqdm.write(markdown or "(empty)")
                        else:
                            blocks = parse_docx(converted)
                    else:
                        # Try antiword fallback
                        text, text_reason = _extract_doc_text_with_antiword(path)
                        if text:
                            degrade_reason = reason or "antiword_fallback"
                            blocks = blocks_from_text(text)
                            degraded_files.append(
                                {
                                    "path": path,
                                    "reason": text_reason or degrade_reason,
                                }
                            )
                            msg = safe_relpath(path, source_dir)
                            if text_reason or degrade_reason:
                                msg = f"{msg} ({text_reason or degrade_reason})"
                            tqdm.write(f"[WARN] degraded parse: {msg}")
                        else:
                            failed_files.append({"path": path, "reason": reason or "doc_convert_failed"})
                            msg = safe_relpath(path, source_dir)
                            if reason:
                                msg = f"{msg} ({reason})"
                            tqdm.write(f"[WARN] convert failed: {msg}")
                            continue
                else:
                    # Parse .docx directly
                    if show_markdown:
                        markdown, blocks = parse_docx_with_markdown(path)
                        tqdm.write(f"[MARKDOWN] {safe_relpath(path, source_dir)}")
                        tqdm.write(markdown or "(empty)")
                    else:
                        blocks = parse_docx(path)
                
                if not blocks:
                    failed_files.append({"path": path, "reason": "parse_empty"})
                    tqdm.write(f"[WARN] parse empty: {safe_relpath(path, source_dir)}")
                    continue
                
                # Collect chunks
                chunks = _collect_chunks(
                    file_path=safe_relpath(path, source_dir),
                    doc_hash=doc_hash,
                    blocks=blocks,
                    max_len=max_len,
                    overlap=overlap,
                )
                
                # Debug output
                if show_markdown:
                    for chunk in chunks:
                        title_path = chunk.get("title_path") or []
                        title_label = " > ".join([part for part in title_path if part])
                        excerpt = str(chunk.get("excerpt_markdown") or "")
                        excerpt = _shorten(excerpt.replace("\n", "\\n"))
                        tqdm.write(
                            "[CHUNK] "
                            f"{chunk['chunk_index']} "
                            f"type={chunk['chunk_type']} "
                            f"len={len(str(chunk.get('text_for_embedding') or ''))} "
                            f"title={title_label or '-'} "
                            f"text={excerpt}"
                        )
                
                # Create records
                for chunk in chunks:
                    records.append(
                        ChunkRecord(
                            vector_id=-1,
                            chunk_id=chunk["chunk_id"],
                            file_path=chunk["file_path"] if "file_path" in chunk else safe_relpath(path, source_dir),
                            title_path=chunk["title_path"],
                            chunk_type=chunk["chunk_type"],
                            chunk_index=chunk["chunk_index"],
                            doc_hash=doc_hash,
                            text_for_embedding=chunk["text_for_embedding"],
                            excerpt_markdown=chunk["excerpt_markdown"],
                        )
                    )
            
            except Exception:
                failed_files.append({"path": path, "reason": "parse_error"})
                tqdm.write(f"[WARN] parse failed: {safe_relpath(path, source_dir)}")
        
        if not records:
            raise RuntimeError("No chunks generated; check source directory and parsers.")
        
        # Embed documents
        print(f"Starting embedding: chunks={len(records)}")
        vectors: List[List[float]] = []
        kept_records: List[ChunkRecord] = []
        embedding_failed: List[Dict[str, str]] = []
        
        with tqdm(total=len(records), desc="Embedding chunks") as progress:
            for batch_records in iter_batches(records, batch_size):
                batch_texts = [record.text_for_embedding for record in batch_records]
                
                try:
                    batch_vectors = _embed_documents_with_retry(embeddings_client, batch_texts)
                except Exception as exc:
                    # Fallback to per-chunk embedding
                    tqdm.write(f"[WARN] embedding batch failed; fallback to per-chunk: {exc}")
                    batch_vectors = []
                    for record in batch_records:
                        try:
                            vector = _embed_documents_with_retry(embeddings_client, [record.text_for_embedding])[0]
                        except Exception as inner_exc:
                            embedding_failed.append(
                                {
                                    "chunk_id": record.chunk_id,
                                    "file_path": record.file_path,
                                    "reason": str(inner_exc),
                                }
                            )
                            tqdm.write(
                                f"[WARN] embedding failed; skipped chunk: {record.file_path}#{record.chunk_index}"
                            )
                            progress.update(1)
                            continue
                        kept_records.append(record)
                        vectors.append(vector)
                        progress.update(1)
                    continue
                
                if len(batch_vectors) != len(batch_records):
                    raise RuntimeError("Embedding batch returned mismatched vector count.")
                
                kept_records.extend(batch_records)
                vectors.extend(batch_vectors)
                progress.update(len(batch_records))
        
        records = kept_records
        if embedding_failed:
            tqdm.write(f"[WARN] embedding failures: {len(embedding_failed)} chunk(s) skipped.")
        
        if not vectors:
            raise RuntimeError("Embedding returned no vectors.")
        
        # Build FAISS index
        print("Building FAISS index...")
        vector_array = np.array(vectors, dtype="float32")
        faiss.normalize_L2(vector_array)
        dim = vector_array.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vector_array)
        
        # Assign vector IDs
        for idx, record in enumerate(records):
            record.vector_id = idx
        
        # Write artifacts
        print("Writing KB artifacts...")
        index_path = os.path.join(version_dir, "index.faiss")
        faiss.write_index(index, index_path)
        
        jsonl_path = os.path.join(version_dir, "chunks.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.model_dump(), ensure_ascii=True))
                handle.write("\n")
        
        manifest = Manifest(
            kb_version=kb_version,
            source_dir=source_dir,
            build_time=datetime.utcnow().isoformat() + "Z",
            embedding_model=embed_model,
            llm_provider_default=provider,
            faiss_metric="cosine",
            doc_count=len(included),
            chunk_count=len(records),
            failed_files=[item["path"] for item in failed_files],
        )
        manifest_path = os.path.join(version_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest.model_dump(), handle, ensure_ascii=False, indent=2)
        
        # Build log
        build_log = {
            "kb_version": kb_version,
            "source_dir": source_dir,
            "included_files": included,
            "skipped_files": skipped,
            "failed_files": failed_files,
            "degraded_files": degraded_files,
            "embedding_failed_chunks": embedding_failed,
        }
        build_log_path = os.path.join(version_dir, "build_log.json")
        with open(build_log_path, "w", encoding="utf-8") as handle:
            json.dump(build_log, handle, ensure_ascii=False, indent=2)
        
        # Activate version
        print("Activating KB version (atomic switch)...")
        _activate_version(out_dir, version_dir)
        print("Atomic switch completed.")
        
        elapsed = time.perf_counter() - start_time
        success_count = max(len(included) - len(failed_files), 0)
        print(
            "Build summary: "
            f"success={success_count}, failed={len(failed_files)}, skipped={len(skipped)}"
        )
        print(f"Build summary: duration={elapsed:.1f}s, output_dir={version_dir}")
        print(f"Build summary: kb_version={kb_version}")
        
        return manifest
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _activate_version(out_dir: str, version_dir: str) -> None:
    """
    Atomically switch 'current' symlink to new version.
    
    Falls back to copying on systems that don't support atomic symlink replacement.
    """
    current_path = os.path.join(out_dir, "current")
    tmp_link = os.path.join(out_dir, "current_tmp")
    relative_target = os.path.relpath(version_dir, out_dir)
    
    # Clean up any existing tmp link
    if os.path.islink(tmp_link) or os.path.exists(tmp_link):
        if os.path.isdir(tmp_link) and not os.path.islink(tmp_link):
            shutil.rmtree(tmp_link)
        else:
            os.unlink(tmp_link)
    
    # Attempt atomic symlink switch
    try:
        os.symlink(relative_target, tmp_link)
        os.replace(tmp_link, current_path)
    except OSError:
        # Fallback to copy on systems without atomic replace
        if os.path.exists(tmp_link):
            os.unlink(tmp_link)
        if os.path.exists(current_path):
            shutil.rmtree(current_path)
        shutil.copytree(version_dir, current_path)
