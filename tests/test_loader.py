"""
Tests for document loading functionality.
"""

import pytest
from pathlib import Path
from utilities import load_documents


def test_load_documents(test_data_dir):
    """Test loading documents from test data directory."""
    docs = load_documents(
        directory_path=str(test_data_dir),
        pattern="*.txt",
        limit=5
    )

    assert len(docs) == 5
    assert all(hasattr(doc, 'content') for doc in docs)
    assert all(hasattr(doc, 'token_count') for doc in docs)
    assert all(hasattr(doc, 'doc_id') for doc in docs)
    assert all(doc.token_count > 0 for doc in docs)


def test_load_documents_pattern(test_data_dir):
    """Test loading with specific pattern."""
    # Should load all .txt files
    docs_all = load_documents(
        directory_path=str(test_data_dir),
        pattern="*.txt"
    )

    assert len(docs_all) > 0
    # We know there are 62 Sherlock Holmes stories
    assert len(docs_all) == 62


def test_load_documents_limit(test_data_dir):
    """Test limit parameter."""
    docs_limited = load_documents(
        directory_path=str(test_data_dir),
        pattern="*.txt",
        limit=10
    )

    assert len(docs_limited) == 10


def test_load_documents_content(test_data_dir):
    """Test that loaded documents have actual content."""
    docs = load_documents(
        directory_path=str(test_data_dir),
        pattern="*.txt",
        limit=1
    )

    assert len(docs) == 1
    doc = docs[0]

    # Should have substantial content
    assert len(doc.content) > 100
    # Should be actual text
    assert "Sherlock" in doc.content or "Holmes" in doc.content
