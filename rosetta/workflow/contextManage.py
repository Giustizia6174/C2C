"""Context management utilities for compressing conversation history."""

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from camel.models import BaseModelBackend

from rosetta.workflow.context_prompt import SUMMARIZE_PROMPT, CONTRACT_PROMPT, SMART_SUMMARIZE_TOOL_RESP_PROMPT, SUMMARIZE_REASONING_SYSTEM, SUMMARIZE_CONTENT_SYSTEM
from rosetta.workflow.camel_utils import model_run_sync
from rosetta.workflow.basic_utils import HistoryConfig, ContentMode

def _get_content(msg: Dict) -> str:
    """Extract content from message, converting tool_calls to text if needed."""
    content = msg.get("content", "")
    if content:
        return content
    # If content is empty but has tool_calls, describe the first tool call
    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        tc = tool_calls[0]
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        args = func.get("arguments", "{}")
        return f"[Tool call: {name}] {args}"
    return ""


def _extract_tool_call(msg: Dict, tool_call_id: Optional[str] = None) -> Optional[str]:
    """Extract tool call as string from assistant message.

    Args:
        msg: Assistant message dict with tool_calls.
        tool_call_id: If provided, find the specific tool call with this ID.
                      If None, returns the first tool call.

    Returns:
        String representation of the tool call, or None if not found.
    """
    tool_calls = msg.get("tool_calls", [])
    if not tool_calls:
        return None

    # Find the matching tool call
    tc = None
    if tool_call_id:
        for call in tool_calls:
            if call.get("id") == tool_call_id:
                tc = call
                break
    if tc is None:
        tc = tool_calls[0]  # Default to first

    func = tc.get("function", {})
    name = func.get("name", "unknown")
    args = func.get("arguments", "{}")
    return f"{name}({args})"


def _is_tool_message(msg: Dict) -> bool:
    """Check if message is a tool response."""
    return msg.get("role") == "tool" or "tool_call_id" in msg


def _find_round_boundaries(messages: List[Dict]) -> List[tuple]:
    """Find (start, end) indices for each round in messages.

    A round consists of:
    - One assistant message (with optional tool_calls)
    - Zero or more following tool messages

    Args:
        messages: List of message dicts.

    Returns:
        List of (start_idx, end_idx) tuples where end_idx is exclusive.
    """
    boundaries = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        # Skip system and user messages
        if role in ("system", "user"):
            i += 1
            continue

        # Start of a round: assistant message
        if role == "assistant":
            start = i
            i += 1
            # Include all following tool messages
            while i < len(messages) and _is_tool_message(messages[i]):
                i += 1
            boundaries.append((start, i))
        else:
            i += 1

    return boundaries


def inject_call_context(messages: List[Dict]) -> List[Dict]:
    """Inject _call key into tool messages from their preceding assistant message.

    For each tool message, find the assistant message with matching tool_calls
    and extract the specific call info based on tool_call_id.

    Args:
        messages: List of message dicts.

    Returns:
        Same list with _call keys injected (modifies in place and returns).
    """
    # Find the most recent assistant message with tool_calls for each tool message
    last_assistant_with_tools = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            last_assistant_with_tools = msg
        elif _is_tool_message(msg) and "_call" not in msg:
            if last_assistant_with_tools:
                tool_call_id = msg.get("tool_call_id")
                call_str = _extract_tool_call(last_assistant_with_tools, tool_call_id)
                if call_str:
                    msg["_call"] = call_str
    return messages


def _parse_output(text: str, roles: List[str]) -> List[Dict[str, str]]:
    """Parse LLM output into message dicts."""
    messages = []
    lines = text.strip().split("\n")
    current_role = None
    current_content = []

    for line in lines:
        if line.startswith("Role:"):
            if current_role is not None:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })
            current_role = line[5:].strip().lower()
            current_content = []
        elif line.startswith("Content:"):
            current_content.append(line[8:].strip())
        elif current_role is not None:
            current_content.append(line)

    if current_role is not None:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content).strip()
        })

    # Ensure correct roles if parsing fails
    if len(messages) != len(roles):
        return [{"role": r, "content": "..."} for r in roles]

    for i, role in enumerate(roles):
        messages[i]["role"] = role

    return messages


def summarize_round(
    messages: List[Dict[str, str]],
    model: BaseModelBackend,
) -> List[Dict[str, str]]:
    """Summarize a 2-message exchange while preserving useful information.

    Args:
        messages: List of 2 message dicts with "role" and "content" keys.
        model: CAMEL model backend for summarization.

    Returns:
        List of 2 summarized message dicts with same roles.
    """
    assert len(messages) == 2, "summarize_round requires exactly 2 messages"

    prompt = SUMMARIZE_PROMPT.format(
        role1=messages[0]["role"],
        content1=_get_content(messages[0]),
        role2=messages[1]["role"],
        content2=_get_content(messages[1]),
    )

    response = model_run_sync(model, [
        {"role": "system", "content": "You summarize conversations concisely. Output only the requested format."},
        {"role": "user", "content": prompt},
    ])
    roles = [messages[0]["role"], messages[1]["role"]]
    return _parse_output(response.choices[0].message.content, roles)

def summarize_tool_resp(
    messages: List[Dict[str, str]],
    model: BaseModelBackend,
) -> List[Dict[str, str]]:
    """Summarize tool response(s), keeping other messages unchanged.

    Auto-detects which messages are tool responses and summarizes only those.
    Uses _call context (if present) to filter irrelevant results.

    Args:
        messages: List of message dicts (can be any length).
        model: CAMEL model backend for summarization.

    Returns:
        List of message dicts with tool responses summarized, others unchanged.
    """
    result = []
    for msg in messages:
        if not _is_tool_message(msg):
            # Keep non-tool messages unchanged
            result.append(dict(msg))
            continue

        tool_content = msg.get("content", "")
        if not tool_content or len(tool_content) < 100:
            # Skip summarization for short responses
            result.append(dict(msg))
            continue

        # Get call context if available
        call_context = msg.get("_call", "unknown")

        prompt = SMART_SUMMARIZE_TOOL_RESP_PROMPT.format(
            tool_call=call_context,
            tool_content=tool_content
        )

        response = model_run_sync(model, [
            {"role": "system", "content": "You summarize tool responses concisely, keeping only information relevant to the query. Output only the summarized content."},
            {"role": "user", "content": prompt},
        ])

        summarized_content = response.choices[0].message.content.strip()

        # Build result message, preserving special keys
        result_msg = {"role": msg.get("role", "tool"), "content": summarized_content}
        if "tool_call_id" in msg:
            result_msg["tool_call_id"] = msg["tool_call_id"]
        if "_call" in msg:
            result_msg["_call"] = msg["_call"]

        result.append(result_msg)

    return result


# Backward compatibility alias
summarize_response = summarize_tool_resp


def summarize_reasoning(
    messages: List[Dict],
    model: BaseModelBackend,
    min_length: int = 100,
) -> List[Dict]:
    """Summarize reasoning content while preserving voice and thought process.

    Args:
        messages: List of message dicts.
        model: CAMEL model backend for summarization.
        min_length: Skip summarization for content shorter than this.

    Returns:
        List of message dicts with summarized content, preserving structure.
    """
    result = []
    for msg in messages:
        content = msg.get("content", "")

        # Skip if too short
        if not content or len(content) < min_length:
            result.append(dict(msg))
            continue

        response = model_run_sync(model, [
            {"role": "system", "content": SUMMARIZE_REASONING_SYSTEM},
            {"role": "user", "content": f"Shorten this text:\n\n{content}"},
        ])
        summarized = response.choices[0].message.content
        summarized = summarized.strip() if summarized else ""

        # Preserve all keys except content
        result_msg = {k: v for k, v in msg.items() if k != "content"}
        result_msg["content"] = summarized
        result.append(result_msg)

    return result


def summarize_content(
    messages: List[Dict],
    model: BaseModelBackend,
    min_length: int = 100,
) -> List[Dict]:
    """Summarize content concisely while preserving key information.

    Args:
        messages: List of message dicts.
        model: CAMEL model backend for summarization.
        min_length: Skip summarization for content shorter than this.

    Returns:
        List of message dicts with summarized content, preserving structure.
    """
    result = []
    for msg in messages:
        content = msg.get("content", "")

        # Skip if too short
        if not content or len(content) < min_length:
            result.append(dict(msg))
            continue

        response = model_run_sync(model, [
            {"role": "system", "content": SUMMARIZE_CONTENT_SYSTEM},
            {"role": "user", "content": f"Summarize the following content:\n\n{content}"},
        ])
        summarized = response.choices[0].message.content
        summarized = summarized.strip() if summarized else ""

        # Preserve all keys except content
        result_msg = {k: v for k, v in msg.items() if k != "content"}
        result_msg["content"] = summarized
        result.append(result_msg)

    return result


def contract(
    messages: List[Dict[str, str]],
    model: BaseModelBackend,
) -> List[Dict[str, str]]:
    """Contract 4 messages (2 rounds) into 2 messages (1 round).

    Args:
        messages: List of 4 message dicts (2 conversation rounds).
        model: CAMEL model backend for contraction.

    Returns:
        List of 2 message dicts showing start intent and final result.
    """
    assert len(messages) == 4, "contract requires exactly 4 messages"

    prompt = CONTRACT_PROMPT.format(
        role1=messages[0]["role"],
        content1=_get_content(messages[0]),
        role2=messages[1]["role"],
        content2=_get_content(messages[1]),
        role3=messages[2]["role"],
        content3=_get_content(messages[2]),
        role4=messages[3]["role"],
        content4=_get_content(messages[3]),
    )

    response = model_run_sync(model, [
        {"role": "system", "content": "You merge conversation rounds concisely. Output only the requested format."},
        {"role": "user", "content": prompt},
    ])
    roles = [messages[0]["role"], messages[3]["role"]]
    return _parse_output(response.choices[0].message.content, roles)

@dataclass
class ContextNode:
    """A node in the context tree representing a round of messages."""

    idx: int
    hash: str
    messages: List[Dict]
    source: str  # "original", "summarize", etc.
    token_count: int = 0
    parent_hashes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Detailed string representation of the node."""
        lines = [f"ContextNode[{self.idx}] ({self.source}, {self.token_count} tokens)"]
        lines.append(f"  hash: {self.hash}")
        if self.parent_hashes:
            lines.append(f"  parents: {self.parent_hashes}")
        lines.append("  messages:")
        for msg in self.messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            preview = content[:80] + "..." if len(content) > 80 else content
            preview = " ".join(preview.split())  # Normalize whitespace
            lines.append(f"    [{role}] {preview}")
        return "\n".join(lines)

    def short(self, highlight: bool = False) -> str:
        """Short representation: [idx:tokens]."""
        if highlight:
            return f"\033[93m[{self.idx}:{self.token_count}]\033[0m"
        return f"[{self.idx}:{self.token_count}]"


class ContextManager:
    """Manages context compression with tree-based provenance tracking.

    Tracks how messages transform through operations like summarization.
    Nodes are rounds (2 messages), edges show transformations.
    """

    def __init__(
        self,
        model: BaseModelBackend,
        tokenizer=None,
        history_config: Optional[HistoryConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = history_config or HistoryConfig()
        self._nodes: Dict[str, ContextNode] = {}  # hash -> ContextNode
        self._node_list: List[ContextNode] = []  # For index-based access
        self._edges: List[tuple] = []  # (src_hashes, dst_hash, op_name)
        self._last_input_hashes: List[str] = []  # Track last apply inputs

    @property
    def nodes(self) -> List[ContextNode]:
        """Access nodes by index: ctx_manager.nodes[0]."""
        return self._node_list

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in messages."""
        text = " ".join(m.get("content", "") for m in messages)
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Rough estimate

    @staticmethod
    def _hash(messages: List[Dict]) -> str:
        """Hash messages to create node ID."""
        parts = [f"{m.get('role', '')}:{m.get('content', '')}" for m in messages]
        return hashlib.md5("||".join(parts).encode()).hexdigest()[:12]

    def _register(self, messages: List[Dict], source: str = "original", parents: List[str] = None) -> str:
        """Register messages as a node, return hash."""
        h = self._hash(messages)
        if h not in self._nodes:
            node = ContextNode(
                idx=len(self._node_list),
                hash=h,
                messages=list(messages),
                source=source,
                token_count=self._count_tokens(messages),
                parent_hashes=parents or [],
            )
            self._nodes[h] = node
            self._node_list.append(node)
        return h

    def _summarize_reasoning_text(self, text: str, min_length: int = 100) -> str:
        """Summarize reasoning text using LLM if long enough."""
        if not text or len(text) < min_length:
            return text
        result = summarize_reasoning([{"content": text}], self.model, min_length)
        return result[0]["content"]

    def _summarize_content_text(self, text: str, min_length: int = 100) -> str:
        """Summarize content text using LLM if long enough."""
        if not text or len(text) < min_length:
            return text
        result = summarize_content([{"content": text}], self.model, min_length)
        return result[0]["content"]

    def _apply_assistant_config(self, msg: Dict) -> Dict:
        """Apply history config to assistant message.

        Controls:
            - reasoning_content: API field, kept/removed/summarized based on config.reasoning
            - content: assistant text, kept/removed/summarized based on config.assistant
            - _reasoning: internal field, always preserved
        """
        if msg.get("role") != "assistant":
            return msg

        result = dict(msg)
        content = msg.get("content", "")
        reasoning = msg.get("_reasoning", "")

        # Handle reasoning_content field based on config
        if self.config.reasoning == ContentMode.NONE:
            result.pop("reasoning_content", None)
        elif self.config.reasoning == ContentMode.FULL and reasoning:
            result["reasoning_content"] = reasoning
        elif self.config.reasoning == ContentMode.SUMMARIZED and reasoning:
            result["reasoning_content"] = self._summarize_reasoning_text(reasoning)

        # Handle content field based on config
        if self.config.assistant == ContentMode.NONE:
            result["content"] = ""
        elif self.config.assistant == ContentMode.SUMMARIZED and content:
            result["content"] = self._summarize_content_text(content)
        # FULL: keep as-is (already in result)

        return result

    def _apply_tool_config(self, msg: Dict) -> Dict:
        """Apply history config to tool message."""
        if not _is_tool_message(msg):
            return msg

        if self.config.tool == ContentMode.FULL:
            return msg

        result = dict(msg)
        if self.config.tool == ContentMode.NONE:
            result["content"] = "[executed]"
        elif self.config.tool == ContentMode.SUMMARIZED:
            summarized = summarize_tool_resp([msg], self.model)
            result["content"] = summarized[0]["content"]

        return result

    def _apply_config_to_round(self, messages: List[Dict]) -> List[Dict]:
        """Apply history config to a round (assistant + tool messages)."""
        return [
            self._apply_tool_config(self._apply_assistant_config(m))
            for m in messages
        ]

    def apply(self, messages: List[Dict], dry_run: bool = False) -> List[Dict]:
        """Apply context management with optional delay.

        Operations applied:
        1. Inject _call context into tool messages
        2. Register the latest round as a node
        3. Apply history config to the target round (based on delay setting)

        With delay=0, transforms the latest round immediately.
        With delay=n, transforms the round that is n rounds before the latest.

        A round consists of one assistant message followed by zero or more tool messages.

        Args:
            messages: List of message dicts.
            dry_run: If True, only record nodes without modifying messages.

        Returns:
            Updated messages (unchanged if dry_run=True).
        """
        # Inject _call context into tool messages
        inject_call_context(messages)

        # Find all rounds in messages
        round_boundaries = _find_round_boundaries(messages)
        if not round_boundaries:
            return messages

        # Track existing nodes for highlighting
        self._last_input_hashes = []
        for start, end in round_boundaries:
            round_msgs = messages[start:end]
            round_hash = self._hash(round_msgs)
            if round_hash in self._nodes:
                self._last_input_hashes.append(round_hash)

        # Register the latest round
        last_start, last_end = round_boundaries[-1]
        last_round = messages[last_start:last_end]
        src_hash = self._register(last_round, "original")
        if src_hash not in self._last_input_hashes:
            self._last_input_hashes.append(src_hash)

        if dry_run:
            return messages

        # Determine which round to transform based on delay
        delay = self.config.delay
        target_round_idx = len(round_boundaries) - 1 - delay

        # Check if we have enough rounds for the delay
        if target_round_idx < 0:
            return messages

        # Get target round boundaries
        target_start, target_end = round_boundaries[target_round_idx]
        target_round = messages[target_start:target_end]
        target_hash = self._hash(target_round)

        # Register target if not already registered
        if target_hash not in self._nodes:
            self._register(target_round, "original")

        # Skip if already transformed
        if self._nodes[target_hash].source != "original":
            return messages

        # Determine transformation type for node tracking
        is_default = (
            self.config.reasoning == ContentMode.FULL
            and self.config.assistant == ContentMode.FULL
            and self.config.tool == ContentMode.FULL
        )

        if is_default:
            # Default config: no transformation needed
            return messages

        # Apply config-based transformation
        transformed = self._apply_config_to_round(target_round)

        # Non-default config: register as new node
        source_label = f"config({self.config.reasoning.value},{self.config.assistant.value},{self.config.tool.value})"
        dst_hash = self._register(transformed, source_label, parents=[target_hash])
        self._edges.append(([target_hash], dst_hash, "config"))

        # Replace the target round in messages
        return messages[:target_start] + transformed + messages[target_end:]

    def __str__(self) -> str:
        """Tree visualization with [idx:tokens] format."""
        if not self._node_list:
            return "ContextManager: empty"

        lines = ["Context Tree", "=" * 40]

        # Build parent -> children map
        children: Dict[str, List[str]] = {}
        roots = []
        for node in self._node_list:
            if not node.parent_hashes:
                roots.append(node.hash)
            for ph in node.parent_hashes:
                children.setdefault(ph, []).append(node.hash)

        # Render tree
        def render(h: str, prefix: str = "", is_last: bool = True) -> List[str]:
            node = self._nodes[h]
            highlight = h in self._last_input_hashes
            connector = "└── " if is_last else "├── "
            node_str = node.short(highlight=highlight)
            src_label = f" ({node.source})" if node.source != "original" else ""
            result = [f"{prefix}{connector}{node_str}{src_label}"]

            child_hashes = children.get(h, [])
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, ch in enumerate(child_hashes):
                result.extend(render(ch, child_prefix, i == len(child_hashes) - 1))
            return result

        for i, root in enumerate(roots):
            lines.extend(render(root, "", i == len(roots) - 1))

        lines.append("=" * 40)
        total_tokens = sum(n.token_count for n in self._node_list)
        lines.append(f"Nodes: {len(self._node_list)}, Total tokens: {total_tokens}")
        lines.append("Legend: \033[93m[idx:tokens]\033[0m = last input")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ContextManager(nodes={len(self._node_list)}, edges={len(self._edges)})"

    def get_tree(self) -> Dict:
        """Return tree structure for inspection."""
        return {
            "nodes": {h: {"messages": n.messages, "source": n.source, "tokens": n.token_count}
                      for h, n in self._nodes.items()},
            "edges": list(self._edges),
        }


