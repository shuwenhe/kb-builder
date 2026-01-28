"""Tests for KB Builder utilities."""

import pytest

from kb_builder.utils import (
    normalize_text,
    split_text,
    table_to_markdown,
    table_to_linearized_text,
    split_list_items,
    sha1_text,
)


def test_normalize_text():
    """Test text normalization."""
    assert normalize_text("  hello   world  ") == "hello world"
    assert normalize_text("hello\u00a0world") == "hello world"
    assert normalize_text("line1\nline2") == "line1 line2"


def test_split_text_short():
    """Test splitting text that fits in one chunk."""
    text = "hello world"
    chunks = split_text(text, max_len=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_long():
    """Test splitting long text with overlap."""
    text = "a" * 100
    chunks = split_text(text, max_len=30, overlap=5)
    assert len(chunks) > 1
    # Check overlap
    for i in range(len(chunks) - 1):
        assert chunks[i][-5:] == chunks[i + 1][:5]


def test_table_to_markdown():
    """Test table to markdown conversion."""
    rows = [
        ["Name", "Age"],
        ["Alice", "30"],
        ["Bob", "25"],
    ]
    result = table_to_markdown(rows)
    assert "| Name | Age |" in result
    assert "| --- | --- |" in result
    assert "| Alice | 30 |" in result
    assert "| Bob | 25 |" in result


def test_table_to_linearized_text():
    """Test table to linearized text."""
    rows = [
        ["Name", "Age"],
        ["Alice", "30"],
        ["Bob", "25"],
    ]
    result = table_to_linearized_text(rows)
    assert "Name: Alice; Age: 30" in result
    assert "Name: Bob; Age: 25" in result


def test_split_list_items_numbered():
    """Test splitting numbered list items."""
    text = "Prefix: 1. First item 2. Second item 3. Third item"
    items = split_list_items(text)
    assert len(items) >= 3


def test_split_list_items_no_list():
    """Test text without list items."""
    text = "This is plain text without any list markers"
    items = split_list_items(text)
    assert len(items) == 1
    assert items[0] == text


def test_sha1_text():
    """Test SHA1 hash generation."""
    text = "hello world"
    hash1 = sha1_text(text)
    hash2 = sha1_text(text)
    assert hash1 == hash2
    assert len(hash1) == 40  # SHA1 hex length
    
    # Different text should produce different hash
    hash3 = sha1_text("different text")
    assert hash1 != hash3
