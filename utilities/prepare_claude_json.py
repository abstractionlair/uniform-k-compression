#!/usr/bin/env python3
"""
Prepare Claude conversation JSON files for fractal analysis.

- Replaces attachments and tool outputs with markers
- Removes redundant 'text' field (keeps only 'content')
- Outputs separate JSON strings with clear separators
- Preserves conversation structure with explicit sender attribution
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def clean_message_content(content_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean content blocks by replacing attachments/tool outputs with markers.

    Returns cleaned content blocks with markers for removed items.
    """
    cleaned = []

    for block in content_blocks:
        block_type = block.get('type')

        if block_type == 'text':
            # Keep text blocks as-is
            cleaned.append(block)

        elif block_type == 'tool_use':
            # Replace tool use with marker
            tool_name = block.get('name', 'unknown')
            cleaned.append({
                'type': 'text',
                'text': f'[TOOL USE REMOVED: {tool_name}]'
            })

        elif block_type == 'tool_result':
            # Replace tool results with marker
            tool_name = block.get('name', 'unknown')
            if tool_name == 'web_search':
                marker = '[WEB SEARCH RESULTS REMOVED]'
            else:
                marker = f'[TOOL OUTPUT REMOVED: {tool_name}]'
            cleaned.append({
                'type': 'text',
                'text': marker
            })

        elif block_type == 'image':
            # Replace images with marker
            cleaned.append({
                'type': 'text',
                'text': '[IMAGE ATTACHMENT REMOVED]'
            })

        elif block_type in ['document', 'file']:
            # Replace files with marker
            cleaned.append({
                'type': 'text',
                'text': '[FILE ATTACHMENT REMOVED]'
            })

        else:
            # For unknown types, replace with generic marker
            cleaned.append({
                'type': 'text',
                'text': f'[CONTENT REMOVED: {block_type}]'
            })

    return cleaned


def strip_attachments(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean conversation by replacing attachments/tool outputs with markers.

    Also removes the redundant 'text' field to save tokens.
    """
    cleaned = conversation.copy()

    # Clean chat messages
    if 'chat_messages' in cleaned:
        cleaned['chat_messages'] = []
        for msg in conversation['chat_messages']:
            cleaned_msg = {
                'uuid': msg.get('uuid'),
                'sender': msg.get('sender'),
                'created_at': msg.get('created_at'),
                'updated_at': msg.get('updated_at')
            }

            # Remove 'text' field (redundant with content)
            # Keep only 'content' which has structure

            # Clean content blocks
            if 'content' in msg:
                cleaned_msg['content'] = clean_message_content(msg['content'])
            else:
                # If no content, create empty array
                cleaned_msg['content'] = []

            cleaned['chat_messages'].append(cleaned_msg)

    return cleaned


def format_conversation_json(conversation: Dict[str, Any], separator_width: int = 70) -> str:
    """
    Format a single conversation as JSON with separators.

    Returns a string with:
    ===== CONVERSATION: name =====
    {json content}
    ===== END CONVERSATION =====
    """
    name = conversation.get('name', 'Untitled')

    header = f"CONVERSATION: {name}"
    footer = "END CONVERSATION"

    header_line = f"===== {header} ====="
    footer_line = f"===== {footer} ====="

    # Pretty print JSON with 2-space indent
    json_str = json.dumps(conversation, indent=2, ensure_ascii=False)

    return f"{header_line}\n{json_str}\n{footer_line}\n"


def process_directory(
    input_dir: Path,
    output_dir: Path,
    conversations_per_file: int = 1
) -> None:
    """
    Process all JSON files in input directory.

    Args:
        input_dir: Directory containing JSON conversation files
        output_dir: Directory to write processed files
        conversations_per_file: How many conversations per output file (default 1:1)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    print(f"Processing {len(json_files)} JSON files from {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    for i, json_file in enumerate(json_files):
        try:
            with open(json_file) as f:
                conversation = json.load(f)

            # Strip attachments and tool outputs
            cleaned = strip_attachments(conversation)

            # Format with separators
            formatted = format_conversation_json(cleaned)

            # Write to output (keeping same filename)
            output_file = output_dir / f"{json_file.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(json_files)}...")

        except Exception as e:
            print(f"  ⚠️  Error processing {json_file.name}: {e}")
            continue

    print(f"\n✓ Processed {len(json_files)} conversations")
    print(f"  Output: {output_dir}")


def main():
    """
    Convert Claude JSON conversations to cleaned format for analysis.

    Usage:
        python3 prepare_claude_json.py [input_dir] [output_dir]

    Defaults:
        input_dir: ../../../about_me/claude_conversations/conversations_split
        output_dir: ../../../about_me/claude_conversations/conversations_json_cleaned
    """
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    else:
        # Default: relative to this script
        script_dir = Path(__file__).parent
        input_dir = script_dir / "../../../about_me/claude_conversations/conversations_split"

    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "../../../about_me/claude_conversations/conversations_json_cleaned"

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("\nUsage: python3 prepare_claude_json.py [input_dir] [output_dir]")
        sys.exit(1)

    process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()
