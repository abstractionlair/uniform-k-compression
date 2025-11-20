#!/usr/bin/env python3
"""
Prepare ChatGPT conversation JSON for fractal analysis.

- Replaces attachments and tool outputs with markers
- Outputs separate JSON strings with clear separators
- Preserves conversation structure with explicit role attribution
- Distinguishes ChatGPT conversations from Claude conversations
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


def strip_conversation_for_analysis(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean ChatGPT conversation for analysis.

    Keeps only: title, timestamps, conversation_id, and message chain with roles/content.
    Replaces: attachments, file pointers, tool outputs with markers.
    """
    # Extract message chain
    mapping = conversation.get("mapping", {})
    current_id = conversation.get("current_node")

    messages = extract_message_chain(mapping, current_id)

    # Build cleaned conversation
    cleaned = {
        "title": conversation.get("title", "Untitled"),
        "conversation_id": conversation.get("conversation_id", "unknown"),
        "create_time": conversation.get("create_time"),
        "update_time": conversation.get("update_time"),
        "messages": messages
    }

    return cleaned


def extract_message_chain(mapping: Dict[str, Any], current_id: Optional[str]) -> List[Dict[str, Any]]:
    """
    Extract the message chain by traversing from current_node back to root.

    Follows the same logic as the text extractor but outputs structured JSON.
    """
    if not current_id:
        return []

    # Build chain by following parent links
    chain = []
    seen = set()
    node_id = current_id

    while node_id and node_id not in seen:
        seen.add(node_id)
        node = mapping.get(node_id)
        if not node:
            break
        chain.append(node)
        node_id = node.get("parent")

    chain.reverse()

    # Extract messages
    messages = []
    for node in chain:
        msg = node.get("message")
        if not msg:
            continue

        # Skip hidden messages
        if (msg.get("metadata") or {}).get("is_visually_hidden_from_conversation"):
            continue

        role = msg.get("author", {}).get("role", "unknown")

        # Skip tool messages entirely (replace with marker)
        if role == "tool":
            messages.append({
                "role": "tool",
                "content": "[CODE EXECUTION OUTPUT REMOVED]",
                "timestamp": datetime.fromtimestamp(msg["create_time"], tz=timezone.utc).isoformat() if msg.get("create_time") else None
            })
            continue

        # Extract text content (replace attachments/tool outputs with markers)
        text = extract_text_content(msg, role)
        if not text or not text.strip():
            continue

        # Build message dict
        message_dict = {
            "role": role,
            "content": text
        }

        # Add timestamp if available
        if msg.get("create_time") is not None:
            timestamp = datetime.fromtimestamp(msg["create_time"], tz=timezone.utc)
            message_dict["timestamp"] = timestamp.isoformat()

        messages.append(message_dict)

    return messages


def extract_text_content(message: Dict[str, Any], role: str) -> Optional[str]:
    """
    Extract text content from message, replacing attachments/tool outputs with markers.

    Returns text with markers for removed content.
    """
    content = message.get("content")
    if not content:
        return None

    ctype = content.get("content_type")

    # Handle text-based content types
    if ctype in {"text", "reasoning_recap", "thoughts", "user_editable_context"}:
        parts = content.get("parts") or []
        text = "\n".join(part for part in parts if part)
        return text.strip()

    if ctype == "code":
        language = content.get("language") or ""
        text = content.get("text") or ""
        fence = f"```{language}\n" if language else "```\n"
        return f"{fence}{text}\n```".strip()

    if ctype == "execution_output":
        # Replace tool execution output with marker
        return "[CODE EXECUTION OUTPUT REMOVED]"

    if ctype in {"sonic_webpage", "tether_quote", "tether_browsing_display"}:
        # Web content - replace with marker but keep URL if present
        url = content.get("url")
        if url:
            return f"[WEB SEARCH RESULT REMOVED: {url}]"
        return "[WEB SEARCH RESULT REMOVED]"

    if ctype == "multimodal_text":
        # Extract only text parts, replace asset pointers with markers
        parts = []
        for part in content.get("parts", []):
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("content_type")
            if ptype == "text":
                parts.append(part.get("text") or "")
            elif ptype == "audio_transcription":
                # Include transcriptions (text representation of audio)
                transcript = part.get("text") or ""
                if transcript:
                    label = part.get("direction", "audio")
                    parts.append(f"[{label} transcription]\n{transcript}")
            elif ptype == "image_asset_pointer":
                parts.append("[IMAGE REMOVED]")
            elif ptype == "real_time_user_audio_video_asset_pointer":
                parts.append("[AUDIO/VIDEO REMOVED]")
            else:
                parts.append(f"[CONTENT REMOVED: {ptype}]")

        return "\n".join(parts).strip()

    if ctype == "system_error":
        text = content.get("message") or ""
        return text.strip()

    # For unknown types, replace with marker
    return f"[CONTENT REMOVED: {ctype}]"


def format_conversation_json(conversation: Dict[str, Any], separator_width: int = 70) -> str:
    """
    Format a single ChatGPT conversation as JSON with separators.

    Returns a string with:
    ===== CHATGPT CONVERSATION: title =====
    {json content}
    ===== END CONVERSATION =====
    """
    title = conversation.get("title", "Untitled")

    header = f"CHATGPT CONVERSATION: {title}"
    footer = "END CONVERSATION"

    header_line = f"===== {header} ====="
    footer_line = f"===== {footer} ====="

    # Pretty print JSON with 2-space indent
    json_str = json.dumps(conversation, indent=2, ensure_ascii=False)

    return f"{header_line}\n{json_str}\n{footer_line}\n"


def process_chatgpt_export(
    input_file: Path,
    output_dir: Path
) -> None:
    """
    Process ChatGPT export JSON file.

    Args:
        input_file: conversations.json from ChatGPT export
        output_dir: Directory to write processed files
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ChatGPT conversations from {input_file}")
    with open(input_file) as f:
        conversations = json.load(f)

    print(f"Processing {len(conversations)} conversations")
    print(f"Output directory: {output_dir}")
    print()

    for i, conversation in enumerate(conversations):
        try:
            # Clean conversation
            cleaned = strip_conversation_for_analysis(conversation)

            # Skip if no messages
            if not cleaned.get("messages"):
                continue

            # Format with separators
            formatted = format_conversation_json(cleaned)

            # Write to output (numbered files matching text extraction)
            # Use conversation_id or index for filename
            conv_id = conversation.get("conversation_id", f"unknown_{i:04d}")
            output_file = output_dir / f"{i+1:04d}_{conv_id[:8]}.txt"

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(conversations)}...")

        except Exception as e:
            print(f"  ⚠️  Error processing conversation {i+1}: {e}")
            continue

    print(f"\n✓ Processed {len(conversations)} conversations")
    print(f"  Output: {output_dir}")


def main():
    """
    Convert ChatGPT conversation JSON to cleaned format for analysis.

    Usage:
        python3 prepare_chatgpt_json.py [input_file] [output_dir]

    Defaults:
        input_file: ../../../about_me/chatgpt_conversations/conversations.json
        output_dir: ../../../about_me/chatgpt_conversations/conversations_json_cleaned
    """
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        # Default: relative to this script
        script_dir = Path(__file__).parent
        input_file = script_dir / "../../../about_me/chatgpt_conversations/conversations.json"

    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "../../../about_me/chatgpt_conversations/conversations_json_cleaned"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("\nUsage: python3 prepare_chatgpt_json.py [input_file] [output_dir]")
        sys.exit(1)

    process_chatgpt_export(input_file, output_dir)


if __name__ == "__main__":
    main()
