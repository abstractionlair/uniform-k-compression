#!/usr/bin/env python3
"""
Convert Claude JSON conversation exports to readable text format.

The JSON files contain structured conversation data with metadata.
This extracts the actual conversation text for sampling.
"""

import json
from pathlib import Path
from typing import Tuple


def convert_claude_json_to_text(json_path: Path) -> str:
    """
    Convert a Claude JSON conversation file to readable text.

    Args:
        json_path: Path to JSON file

    Returns:
        Formatted conversation text
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = []

    # Header
    lines.append(f"# {data.get('name', 'Untitled Conversation')}")
    lines.append("")
    lines.append(f"**UUID**: {data.get('uuid', 'unknown')}")
    lines.append(f"**Created**: {data.get('created_at', 'unknown')}")
    lines.append(f"**Updated**: {data.get('updated_at', 'unknown')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Extract messages
    messages = data.get('chat_messages', [])

    for msg in messages:
        sender = msg.get('sender', 'unknown')

        # Format sender
        if sender == 'human':
            lines.append("## User")
        elif sender == 'assistant':
            lines.append("## Claude")
        else:
            lines.append(f"## {sender.title()}")

        lines.append("")

        # Extract text from content
        text_content = []

        # Prefer content array over text field (they often duplicate)
        if 'content' in msg and msg['content']:
            for content_block in msg['content']:
                if isinstance(content_block, dict):
                    if content_block.get('type') == 'text' and 'text' in content_block:
                        text_content.append(content_block['text'])
        # Fallback to text field if no content
        elif 'text' in msg and msg['text']:
            text_content.append(msg['text'])

        # Add extracted text
        for text in text_content:
            lines.append(text.strip())
            lines.append("")

    return '\n'.join(lines)


def convert_corpus(
    source_dir: Path,
    output_dir: Path,
    pattern: str = "*.json"
) -> Tuple[int, int]:
    """
    Convert entire corpus of JSON files to text.

    Args:
        source_dir: Directory with JSON files
        output_dir: Directory to write text files
        pattern: Glob pattern for source files

    Returns:
        Tuple of (files_converted, files_failed)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(source_dir.glob(pattern))

    if not json_files:
        raise ValueError(f"No files found matching {pattern} in {source_dir}")

    print(f"Converting {len(json_files)} JSON files to text...")

    converted = 0
    failed = 0

    for i, json_path in enumerate(json_files, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(json_files)}...")

        try:
            # Convert to text
            text = convert_claude_json_to_text(json_path)

            # Write to output
            output_path = output_dir / f"{json_path.stem}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)

            converted += 1

        except Exception as e:
            print(f"  Error converting {json_path.name}: {e}")
            failed += 1

    print("\nConversion complete:")
    print(f"  Converted: {converted}")
    print(f"  Failed: {failed}")

    return converted, failed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 json_converter.py <source_dir> <output_dir>")
        print("Example: python3 json_converter.py conversations_split conversations_text")
        sys.exit(1)

    source = Path(sys.argv[1])
    output = Path(sys.argv[2])

    if not source.exists():
        print(f"Error: Source directory not found: {source}")
        sys.exit(1)

    convert_corpus(source, output)
