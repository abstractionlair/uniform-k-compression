#!/usr/bin/env python3
"""
Clean ChatGPT conversation text files by removing tool call results.

Similar to excluding PDF attachments from Claude conversations, this removes
web search results, code execution output, and other tool results to focus
on the actual dialogue.
"""

import re
from pathlib import Path
from typing import Tuple


def clean_chatgpt_conversation(text: str) -> str:
    """
    Remove tool call results from ChatGPT conversation text.

    Keeps only user and assistant messages, removes tool outputs.

    Args:
        text: Raw conversation text

    Returns:
        Cleaned conversation text
    """
    lines = text.split('\n')
    cleaned_lines = []

    in_tool_section = False
    in_header = True

    for line in lines:
        # Check if we're entering a tool section
        if re.match(r'\[.*\s+tool\]', line):
            in_tool_section = True
            continue

        # Check if we're entering a user/assistant section
        if re.match(r'\[.*\s+(user|assistant)\]', line):
            in_tool_section = False
            in_header = False
            cleaned_lines.append(line)
            continue

        # Keep header lines (before first message)
        if in_header:
            cleaned_lines.append(line)
            continue

        # Skip lines in tool sections
        if in_tool_section:
            continue

        # Keep other lines (part of user/assistant messages)
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def clean_chatgpt_corpus(
    source_dir: Path,
    output_dir: Path,
    pattern: str = "*.txt"
) -> Tuple[int, int, int]:
    """
    Clean entire corpus of ChatGPT conversation files.

    Args:
        source_dir: Directory with raw conversation files
        output_dir: Directory to write cleaned files
        pattern: Glob pattern for files

    Returns:
        Tuple of (files_processed, files_modified, files_failed)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(source_dir.glob(pattern))

    if not txt_files:
        raise ValueError(f"No files found matching {pattern} in {source_dir}")

    print(f"Cleaning {len(txt_files)} ChatGPT conversation files...")

    processed = 0
    modified = 0
    failed = 0

    for i, txt_path in enumerate(txt_files, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(txt_files)}...")

        try:
            # Read original
            with open(txt_path, 'r', encoding='utf-8') as f:
                original = f.read()

            # Clean
            cleaned = clean_chatgpt_conversation(original)

            # Write output
            output_path = output_dir / txt_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)

            processed += 1

            # Track if we actually modified content
            if len(cleaned) < len(original):
                modified += 1

        except Exception as e:
            print(f"  Error cleaning {txt_path.name}: {e}")
            failed += 1

    print("\nCleaning complete:")
    print(f"  Processed: {processed}")
    print(f"  Modified (had tool calls): {modified}")
    print(f"  Failed: {failed}")

    return processed, modified, failed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 chatgpt_cleaner.py <source_dir> <output_dir>")
        print("Example: python3 chatgpt_cleaner.py conversations_text conversations_text_cleaned")
        sys.exit(1)

    source = Path(sys.argv[1])
    output = Path(sys.argv[2])

    if not source.exists():
        print(f"Error: Source directory not found: {source}")
        sys.exit(1)

    clean_chatgpt_corpus(source, output)
