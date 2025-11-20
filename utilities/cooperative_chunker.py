#!/usr/bin/env python3
"""
Cooperative chunking utility for oversized documents.

Implements methodology §5: break large documents into readable chunks
using LLM-based cooperative protocol.
"""

import sys
from pathlib import Path
from typing import List
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from core.document import Document
from core.llm_interface import LLMInterface
from core.batch_interface import BatchInterface, BatchRequest
from core.config import PreprocessingConfig


def needs_chunking(document: Document, threshold: int) -> bool:
    """Check if document exceeds chunking threshold."""
    return document.token_count > threshold


def build_cooperative_chunking_prompt(
    chunk_num: int,
    total_chunks: int,
    doc_name: str,
    left_context: str,
    target_section: str,
    right_context: str
) -> str:
    """
    Build prompt for cooperative chunking.

    Follows methodology §5.2 exactly.

    Args:
        chunk_num: This chunk number (1-indexed)
        total_chunks: Total number of chunks
        doc_name: Original document name
        left_context: Content before this section
        target_section: This chunk's section
        right_context: Content after this section

    Returns:
        Prompt for LLM
    """
    left_display = left_context if left_context else "(beginning of document)"
    right_display = right_context if right_context else "(end of document)"

    prompt = f"""GOAL: For {total_chunks} model instances to cooperatively divide "{doc_name}" into
{total_chunks} standalone, readable chunks that together preserve all original content.

You are instance {chunk_num} of {total_chunks} and are responsible for YOUR SECTION below.

=== CONTEXT (what came before) ===
{left_display}

=== YOUR SECTION ===
{target_section}

=== CONTEXT (what comes after) ===
{right_display}

Instructions:
Ideally preserve YOUR SECTION verbatim. But you may make changes if needed for
readability. Use your judgment to find natural boundaries - for example:
- Don't split mid-sentence or mid-paragraph
- Don't split code blocks, tables, or quoted sections
- If a structured element (list, code block, table) straddles the boundary,
  either include it entirely in one chunk or overlap it between chunks
- Add context from adjacent sections if needed for standalone comprehension

The goal is readable, standalone chunks. Exercise judgment about what makes sense.

Format:
---
Editor's Note: This is chunk {chunk_num} of {total_chunks} from "{doc_name}".
[Add context here only if needed for standalone comprehension.]
---

[Your chunk content]
"""

    return prompt


def chunk_document(
    document: Document,
    target_chunk_size: int,
    llm: LLMInterface = None,
    batch_interface: BatchInterface = None,
    parallel: bool = True,
    use_batch_api: bool = False
) -> List[Document]:
    """
    Split a large document into chunks using cooperative protocol.

    Args:
        document: Document to chunk
        target_chunk_size: Target size per chunk (tokens)
        llm: LLM interface for real-time calls (required if not using batch)
        batch_interface: BatchInterface for batch API (required if use_batch_api=True)
        parallel: If True, process chunks in parallel (real-time only)
        use_batch_api: If True, use batch API (50% cost, takes hours)

    Returns:
        List of chunk Documents (or batch ID if use_batch_api=True)
    """
    if not use_batch_api and llm is None:
        raise ValueError("llm required for real-time chunking")
    if use_batch_api and batch_interface is None:
        raise ValueError("batch_interface required for batch API chunking")
    # Determine number of chunks
    n_chunks = math.ceil(document.token_count / target_chunk_size)

    print(f"  Chunking {document.doc_id}: {document.token_count} tokens → {n_chunks} chunks")
    if parallel:
        print(f"    Processing {n_chunks} chunks in parallel...")

    # Divide content into regions
    content = document.content
    chunk_size_chars = len(content) // n_chunks

    # Prepare all chunk parameters
    chunk_params = []
    for i in range(n_chunks):
        start = i * chunk_size_chars
        end = (i + 1) * chunk_size_chars if i < n_chunks - 1 else len(content)

        left_context = content[0:start] if i > 0 else ""
        target_section = content[start:end]
        right_context = content[end:] if end < len(content) else ""

        prompt = build_cooperative_chunking_prompt(
            chunk_num=i + 1,
            total_chunks=n_chunks,
            doc_name=document.doc_id,
            left_context=left_context,
            target_section=target_section,
            right_context=right_context
        )

        chunk_params.append((i, prompt))

    # Process chunks
    chunk_results = []

    if use_batch_api:
        # Batch API mode (50% cost, takes hours)
        print(f"    Submitting to batch API (50% cost savings)...")

        batch_requests = [
            BatchRequest(
                custom_id=f"{document.doc_id}_chunk{i+1}",
                prompt=prompt,
                max_tokens=60_000
            )
            for i, prompt in chunk_params
        ]

        # This would return batch_id and require separate polling/retrieval
        # For now, submit and wait
        results = batch_interface.submit_and_wait(batch_requests)

        # Convert results to chunk_results format
        for custom_id, output_text, output_tokens in results:
            # Extract chunk number from custom_id
            chunk_idx = int(custom_id.split('_chunk')[-1]) - 1
            chunk_results.append((chunk_idx, output_text, output_tokens))

        chunk_results.sort(key=lambda x: x[0])

    elif parallel and n_chunks > 1:
        # Parallel real-time processing
        def process_chunk(params):
            chunk_idx, prompt = params
            chunk_content, chunk_tokens, _ = llm.call(prompt, "large", max_tokens=60_000, timeout=600.0)
            return (chunk_idx, chunk_content, chunk_tokens)

        with ThreadPoolExecutor(max_workers=min(n_chunks, 5)) as executor:
            futures = {executor.submit(process_chunk, params): params[0] for params in chunk_params}

            for future in as_completed(futures):
                chunk_idx, chunk_content, chunk_tokens = future.result()
                chunk_results.append((chunk_idx, chunk_content, chunk_tokens))
                print(f"      Chunk {chunk_idx + 1}/{n_chunks} complete ({chunk_tokens} tokens)")

        # Sort by chunk index
        chunk_results.sort(key=lambda x: x[0])

    else:
        # Sequential processing
        for i, prompt in chunk_params:
            print(f"      Processing chunk {i + 1}/{n_chunks}...")
            chunk_content, chunk_tokens, _ = llm.call(prompt, "large", max_tokens=60_000, timeout=600.0)
            chunk_results.append((i, chunk_content, chunk_tokens))

    # Create Document objects
    chunks = []
    for chunk_idx, chunk_content, chunk_tokens in chunk_results:
        chunk_doc = Document(
            content=chunk_content,
            token_count=chunk_tokens,
            doc_id=f"{document.doc_id}_chunk{chunk_idx+1}",
            metadata={
                **document.metadata,
                "original_doc_id": document.doc_id,
                "chunk_num": chunk_idx + 1,
                "total_chunks": n_chunks,
                "is_chunk": True
            }
        )
        chunks.append(chunk_doc)

    print(f"    → Created {len(chunks)} chunks: {[c.token_count for c in chunks]} tokens")

    return chunks


def preprocess_documents(
    documents: List[Document],
    config: PreprocessingConfig,
    llm: LLMInterface = None
) -> List[Document]:
    """
    Preprocess documents: chunk oversized ones, pass through others.

    Args:
        documents: Input documents
        config: Preprocessing configuration
        llm: LLM interface (creates default if None)

    Returns:
        Processed documents (originals + chunks)
    """
    if llm is None:
        llm = LLMInterface()

    processed = []
    chunk_count = 0

    print(f"\nPreprocessing {len(documents)} documents...")
    print(f"  Chunking threshold: {config.chunking_threshold:,} tokens")

    for doc in documents:
        if needs_chunking(doc, config.chunking_threshold):
            chunks = chunk_document(doc, config.target_chunk_size, llm)
            processed.extend(chunks)
            chunk_count += 1
        else:
            processed.append(doc)

    print(f"\n✓ Preprocessing complete:")
    print(f"  Documents chunked: {chunk_count}")
    print(f"  Final document count: {len(processed)} (was {len(documents)})")
    print(f"  Total tokens: {sum(doc.token_count for doc in processed):,}")

    return processed


if __name__ == "__main__":
    # Test document loader
    if len(sys.argv) < 2:
        print("Usage: python3 document_loader.py <directory> [pattern] [limit]")
        print("Example: python3 document_loader.py /tmp/test_docs *.txt 100")
        sys.exit(1)

    directory = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*.txt"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    docs = load_documents(directory, pattern, limit=limit)

    print(f"\nLoaded {len(docs)} documents")

    # Check for any that would need chunking
    large_docs = [d for d in docs if d.token_count > 50_000]
    if large_docs:
        print(f"\nDocuments exceeding 50k tokens: {len(large_docs)}")
        for doc in large_docs[:5]:
            print(f"  {doc.doc_id}: {doc.token_count:,} tokens")
    else:
        print(f"\nNo documents exceed 50k tokens (no chunking needed)")
