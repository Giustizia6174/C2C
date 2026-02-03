"""Section-aware tokenization for gpt-oss-20b model family.

This module provides precise section detection within tokenized messages,
identifying system, user, tool, and assistant sections with sub-types
(reasoning, tool_call, text).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .interface import TokenSection, TokenizedConversation

# =============================================================================
# Token Marker IDs for gpt-oss-20b
# =============================================================================

OSS_MARKERS = {
    "START": 200006,    # <|start|> - Section start
    "END": 200007,      # <|end|> - Section end (normal)
    "CALL": 200012,     # <|call|> - Section end (tool call)
    "RETURN": 200002,   # <|return|> - Section end (final)
    "CHANNEL": 200005,  # <|channel|> - Channel type marker
    "MESSAGE": 200008,  # <|message|> - Content start
}


def _is_oss_tokenizer(tokenizer) -> bool:
    """Check if tokenizer is from gpt-oss model family."""
    name = getattr(tokenizer, "name_or_path", "")
    return "gpt-oss" in name.lower() or "openai/gpt-oss" in name.lower()


def _parse_section_header(
    tokens: List[int],
    start_idx: int,
    message_idx: int,
    end_marker_id: int,
    tokenizer,
) -> tuple[str, str]:
    """Parse role and content_type from section header tokens.

    Args:
        tokens: Full token list
        start_idx: Index of <|start|> token
        message_idx: Index of <|message|> token
        end_marker_id: Token ID of the section end marker
        tokenizer: Tokenizer for decoding

    Returns:
        (role, content_type) tuple
    """
    # Decode header text (between <|start|> and <|message|>)
    header_tokens = tokens[start_idx + 1 : message_idx]
    header_text = tokenizer.decode(header_tokens, skip_special_tokens=False)

    # Default values
    role = "unknown"
    content_type = "text"

    # Parse role from header
    header_lower = header_text.lower()

    # Tool response starts with "functions." (e.g., "functions.search to=assistant...")
    # Note: Tool CALL starts with "assistant to=functions..." which also contains "functions."
    # So we must check if it STARTS with "functions." to detect tool response
    if header_text.startswith("functions."):
        role = "tool"
        content_type = "tool_response"

    elif header_lower.startswith("assistant") or "assistant" in header_lower:
        role = "assistant"

        # Determine content_type from channel
        if "analysis" in header_lower:
            content_type = "reasoning"
        elif "commentary" in header_lower:
            # Tool call if ends with <|call|>
            if end_marker_id == OSS_MARKERS["CALL"]:
                content_type = "tool_call"
            else:
                content_type = "text"
        elif "final" in header_lower:
            content_type = "text"
        else:
            content_type = "text"

    elif header_lower.startswith("system"):
        role = "system"
        content_type = "text"

    elif header_lower.startswith("developer"):
        # Developer section contains tool definitions
        role = "developer"
        content_type = "tool_definitions"

    elif header_lower.startswith("user"):
        role = "user"
        content_type = "text"

    return role, content_type


def tokenize_conversation_oss(
    messages: List[Dict[str, Any]],
    tokenizer,
    tools: Optional[List[Dict[str, Any]]] = None,
    exclude_final: bool = False,
    conversation_id: Optional[str] = None,
    convert_reasoning: bool = True,
) -> TokenizedConversation:
    """Tokenize conversation with precise section detection for gpt-oss models.

    Args:
        messages: List of message dicts with "role" and "content".
        tokenizer: gpt-oss tokenizer (openai/gpt-oss-*).
        tools: Optional list of tool schemas (OpenAI format).
        exclude_final: If True, exclude final message to include all reasoning.
        conversation_id: Optional identifier for the conversation.
        convert_reasoning: If True, convert reasoning_content to thinking field.

    Returns:
        TokenizedConversation with precise section boundaries.

    Raises:
        NotImplementedError: If tokenizer is not from gpt-oss family.
    """
    if not _is_oss_tokenizer(tokenizer):
        raise NotImplementedError(
            f"tokenize_conversation_oss only supports gpt-oss models. "
            f"Got: {getattr(tokenizer, 'name_or_path', 'unknown')}"
        )

    # Preprocess messages
    processed_messages = []
    for msg in messages:
        new_msg = dict(msg)
        # Convert reasoning_content to thinking for gpt-oss template
        if convert_reasoning and msg.get("reasoning_content"):
            new_msg["thinking"] = msg["reasoning_content"]
        processed_messages.append(new_msg)

    # Optionally exclude final message (for tool-call analysis)
    if exclude_final:
        for i in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[i]
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                processed_messages = processed_messages[:i]
                break

    # Convert tools if needed
    tool_schemas = None
    if tools:
        tool_schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
            elif hasattr(tool, "get_openai_tool_schema"):
                tool_schemas.append(tool.get_openai_tool_schema())
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

    # Render and tokenize
    rendered = tokenizer.apply_chat_template(
        processed_messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tool_schemas,
    )
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    input_ids = torch.tensor(tokens, dtype=torch.long)

    # Find sections by scanning for markers
    sections = _find_sections(tokens, tokenizer)

    return TokenizedConversation(
        input_ids=input_ids,
        sections=sections,
        messages=processed_messages,
        conversation_id=conversation_id,
    )


def _find_sections(tokens: List[int], tokenizer) -> List[TokenSection]:
    """Find all sections in tokenized conversation.

    Scans for <|start|> markers and parses section boundaries.

    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer for decoding headers

    Returns:
        List of TokenSection objects
    """
    START = OSS_MARKERS["START"]
    END = OSS_MARKERS["END"]
    CALL = OSS_MARKERS["CALL"]
    RETURN = OSS_MARKERS["RETURN"]
    MESSAGE = OSS_MARKERS["MESSAGE"]
    END_MARKERS = {END, CALL, RETURN}

    sections = []
    message_idx_counter = 0
    i = 0

    while i < len(tokens):
        # Find next <|start|>
        if tokens[i] != START:
            i += 1
            continue

        start_idx = i

        # Find <|message|> after <|start|>
        message_pos = None
        for j in range(start_idx + 1, min(start_idx + 50, len(tokens))):
            if tokens[j] == MESSAGE:
                message_pos = j
                break

        if message_pos is None:
            i += 1
            continue

        # Find end marker after <|message|>
        end_pos = None
        end_marker_id = None
        for k in range(message_pos + 1, len(tokens)):
            if tokens[k] in END_MARKERS:
                end_pos = k
                end_marker_id = tokens[k]
                break

        if end_pos is None:
            # No end marker found, section extends to end
            end_pos = len(tokens)
            end_marker_id = END

        # Parse header to get role and content_type
        role, content_type = _parse_section_header(
            tokens, start_idx, message_pos, end_marker_id, tokenizer
        )

        # Content is between <|message|> and end marker
        content_start = message_pos + 1
        content_end = end_pos

        # Create section if it has content
        if content_end > content_start:
            section = TokenSection(
                start_idx=content_start,
                end_idx=content_end,
                role=role,
                message_idx=message_idx_counter,
                content_type=content_type,
            )
            sections.append(section)
            message_idx_counter += 1

        # Move past end marker
        i = end_pos + 1

    return sections


def tokenize_conversation_with_sections(
    messages: List[Dict[str, Any]],
    tokenizer,
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> TokenizedConversation:
    """Auto-detect model family and tokenize with section detection.

    This is the main dispatcher function that selects the appropriate
    tokenizer based on the model family.

    Args:
        messages: List of message dicts.
        tokenizer: HuggingFace tokenizer.
        tools: Optional list of tool schemas.
        **kwargs: Additional arguments passed to model-specific tokenizer.

    Returns:
        TokenizedConversation with precise section boundaries.

    Raises:
        NotImplementedError: If model family is not supported.

    Supported models:
        - gpt-oss-* (openai/gpt-oss-20b, etc.)
    """
    model_name = getattr(tokenizer, "name_or_path", "")

    if _is_oss_tokenizer(tokenizer):
        return tokenize_conversation_oss(messages, tokenizer, tools, **kwargs)
    else:
        raise NotImplementedError(
            f"Section-aware tokenization not supported for model: {model_name}. "
            f"Supported model families: gpt-oss-*"
        )


def batch_tokenize_with_sections(
    conversations: List[tuple[str, List[Dict[str, Any]]]],
    tokenizer,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_length: Optional[int] = None,
    show_progress: bool = True,
    **kwargs,
) -> List[TokenizedConversation]:
    """Batch tokenize conversations with section detection.

    Args:
        conversations: List of (id, messages) tuples.
        tokenizer: HuggingFace tokenizer.
        tools: Optional list of tool schemas.
        max_length: Optional maximum sequence length (skip longer).
        show_progress: Whether to show progress bar.
        **kwargs: Additional arguments passed to tokenizer.

    Returns:
        List of TokenizedConversation objects with section boundaries.
    """
    results = []

    iterator = conversations
    if show_progress:
        try:
            from rich.progress import track

            iterator = track(conversations, description="Tokenizing with sections...")
        except ImportError:
            pass

    for conv_id, messages in iterator:
        try:
            conv = tokenize_conversation_with_sections(
                messages,
                tokenizer,
                tools=tools,
                conversation_id=conv_id,
                **kwargs,
            )
            if max_length is not None and conv.seq_len > max_length:
                continue
            results.append(conv)
        except Exception as e:
            print(f"Warning: Failed to tokenize conversation {conv_id}: {e}")
            continue

    return results
