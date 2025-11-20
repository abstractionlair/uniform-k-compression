#!/usr/bin/env python3
"""
Document loader utility.

Loads text files from a directory and creates Document objects with
accurate token counts.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from core.document import Document, Tokenizer


def load_documents(
    directory_path: str,
    pattern: str = "*.txt",
    tokenizer: Tokenizer = None,
    limit: int = None
) -> List[Document]:
    """
    Load all documents from a directory.

    Args:
        directory_path: Path to directory containing text files
        pattern: Glob pattern for files (default: *.txt)
        tokenizer: Tokenizer to use (creates default if None)
        limit: Optional limit on number of files to load

    Returns:
        List of Document objects with token counts
    """
    if tokenizer is None:
        tokenizer = Tokenizer()

    directory = Path(directory_path)

    if not directory.exists():
        raise ValueError(f"Directory not found: {directory_path}")

    # Find all matching files
    files = sorted(directory.glob(pattern))

    if limit:
        files = files[:limit]

    if not files:
        raise ValueError(f"No files matching '{pattern}' found in {directory_path}")

    print(f"Loading {len(files)} documents from {directory_path}...")

    documents = []

    for i, file_path in enumerate(files, 1):
        if i % 100 == 0:
            print(f"  Loaded {i}/{len(files)}...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            doc = tokenizer.create_document(
                content=content,
                doc_id=file_path.stem,
                metadata={
                    "source_file": str(file_path),
                    "file_size_bytes": file_path.stat().st_size
                }
            )

            documents.append(doc)

        except Exception as e:
            print(f"  Warning: Failed to load {file_path.name}: {e}")

    print(f"\n✓ Loaded {len(documents)} documents")
    print(f"  Total tokens: {sum(doc.token_count for doc in documents):,}")
    print(f"  Avg tokens/doc: {sum(doc.token_count for doc in documents) / len(documents):.0f}")
    print(f"  Min: {min(doc.token_count for doc in documents):,}, Max: {max(doc.token_count for doc in documents):,}")

    return documents


def load_multiple_directories(
    directories: List[Tuple[str, str]],
    tokenizer: Tokenizer = None
) -> List[Document]:
    """
    Load documents from multiple directories.

    Useful for combining Claude and ChatGPT conversations.

    Args:
        directories: List of (path, pattern) tuples
        tokenizer: Tokenizer to use

    Returns:
        Combined list of Document objects
    """
    if tokenizer is None:
        tokenizer = Tokenizer()

    all_documents = []

    for directory, pattern in directories:
        docs = load_documents(directory, pattern, tokenizer)
        all_documents.extend(docs)

    print(f"\nCombined corpus: {len(all_documents)} documents")

    return all_documents


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 document_loader.py <directory> [pattern] [limit]")
        print("Example: python3 document_loader.py /tmp/test_conversations *.txt 10")
        sys.exit(1)

    directory = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*.txt"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    docs = load_documents(directory, pattern, limit=limit)

    print("\nFirst 3 documents:")
    for doc in docs[:3]:
        print(f"  {doc.doc_id}: {doc.token_count} tokens")
