"""
Tests for Document and Tokenizer classes.
"""

import pytest
from core import Document, Tokenizer, create_document_with_tokens


def test_tokenizer_initialization():
    """Test that tokenizer initializes correctly."""
    tokenizer = Tokenizer()
    assert tokenizer is not None


def test_tokenizer_count_tokens():
    """Test token counting."""
    tokenizer = Tokenizer()
    text = "Hello, world! This is a test."
    count = tokenizer.count_tokens(text)
    assert count > 0
    assert isinstance(count, int)


def test_create_document_with_tokens():
    """Test document creation with automatic token counting."""
    tokenizer = Tokenizer()
    content = "This is a test document with some content."
    doc_id = "test_001"

    doc = create_document_with_tokens(content, doc_id, tokenizer=tokenizer)

    assert doc.content == content
    assert doc.doc_id == doc_id
    assert doc.token_count > 0
    assert isinstance(doc.metadata, dict)


def test_document_immutability():
    """Test that documents are immutable."""
    tokenizer = Tokenizer()
    doc = create_document_with_tokens("Test content", "test_001", tokenizer=tokenizer)

    with pytest.raises(AttributeError):
        doc.content = "Modified content"


def test_document_validation():
    """Test that document validation works."""
    with pytest.raises(ValueError):
        Document(content="", token_count=100, doc_id="test")

    with pytest.raises(ValueError):
        Document(content="Test", token_count=0, doc_id="test")

    with pytest.raises(ValueError):
        Document(content="Test", token_count=100, doc_id="")
