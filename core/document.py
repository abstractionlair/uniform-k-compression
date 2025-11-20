#!/usr/bin/env python3
"""
Document data model for uniform-K fractal summarization.

A Document represents a single unit in the corpus with its content,
token count, and metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import tiktoken


@dataclass(frozen=True)
class Document:
    """
    Immutable document with content and token count.

    Attributes:
        content: Full text content of the document
        token_count: Number of tokens (must be pre-computed)
        doc_id: Unique identifier
        metadata: Additional information (source, timestamp, chunk info, etc.)
    """
    content: str
    token_count: int
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate document on creation."""
        if self.token_count <= 0:
            raise ValueError(f"token_count must be positive, got {self.token_count}")
        if not self.doc_id:
            raise ValueError("doc_id cannot be empty")
        if not self.content:
            raise ValueError("content cannot be empty")


class Tokenizer:
    """
    Wrapper for tokenization to count tokens accurately.

    Uses tiktoken (GPT tokenizer) as a reasonable approximation for Claude.
    For production use with Claude, should use actual Claude tokenizer.
    """

    def __init__(self, model_name: str = "cl100k_base"):
        """
        Initialize tokenizer.

        Args:
            model_name: Tiktoken model name
                - "cl100k_base": GPT-4, GPT-3.5-turbo (default)
                - "p50k_base": GPT-3 (davinci, curie, etc.)
        """
        try:
            self.encoding = tiktoken.get_encoding(model_name)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer '{model_name}': {e}")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def create_document(self, content: str, doc_id: str,
                       metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Create a Document with automatic token counting.

        Args:
            content: Document text
            doc_id: Unique identifier
            metadata: Optional metadata dict

        Returns:
            Document with token_count computed
        """
        token_count = self.count_tokens(content)
        return Document(
            content=content,
            token_count=token_count,
            doc_id=doc_id,
            metadata=metadata or {}
        )


# Convenience function for quick document creation
def create_document_with_tokens(content: str, doc_id: str,
                                metadata: Optional[Dict[str, Any]] = None,
                                tokenizer: Optional[Tokenizer] = None) -> Document:
    """
    Create a document with automatic tokenization.

    Args:
        content: Document text
        doc_id: Unique identifier
        metadata: Optional metadata
        tokenizer: Tokenizer to use (creates default if None)

    Returns:
        Document with token count
    """
    if tokenizer is None:
        tokenizer = Tokenizer()

    return tokenizer.create_document(content, doc_id, metadata)


if __name__ == "__main__":
    # Example usage
    tokenizer = Tokenizer()

    doc = tokenizer.create_document(
        content="This is a test document with some content.",
        doc_id="test_001",
        metadata={"source": "test", "created": "2025-01-01"}
    )

    print(f"Document: {doc.doc_id}")
    print(f"Tokens: {doc.token_count}")
    print(f"Content length: {len(doc.content)} chars")
    print(f"Metadata: {doc.metadata}")
